//! [`WeightAccel`] implementation for the Radeon 890M iGPU.
//!
//! Holds a live GPU context and a map of weight name → device-resident matrix
//! (Q4_0 or Q6_K). The model uploads its quantized weights once at load; each
//! decode step the model calls [`GpuWeightAccel::gemv`] per projection, which
//! runs the fused dequant→GEMV kernel and reads back the result. Weights the GPU
//! can't adopt (unsupported quant / shape) are declined and stay on the CPU.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use strix_core::accel::{GpuDecodeConfig, WeightAccel};

use crate::qgemv::{GpuQ4, ResidentQ4, ResidentQ6};

/// Nanoseconds spent inside GPU GEMV calls (submit→readback, incl. the sync).
/// Lets the CLI split decode time into GPU-matmul vs CPU-glue. Profiling only.
static GPU_NANOS: AtomicU64 = AtomicU64::new(0);

/// Total time spent in GPU GEMVs since the last [`reset_gpu_time`], in ms.
pub fn gpu_time_ms() -> f64 {
    GPU_NANOS.load(Ordering::Relaxed) as f64 / 1e6
}

/// Reset the GPU-time counter.
pub fn reset_gpu_time() {
    GPU_NANOS.store(0, Ordering::Relaxed);
}

/// Add to the GPU-time counter (used by the ash decode path too).
#[cfg(feature = "ash")]
pub(crate) fn add_gpu_nanos(n: u64) {
    GPU_NANOS.fetch_add(n, Ordering::Relaxed);
}

/// A resident weight in its native quantization.
enum Resident {
    Q4(ResidentQ4),
    Q6(ResidentQ6),
}

/// Resident GPU buffers for the on-device decode forward (allocated once).
struct DecodeScratch {
    h: wgpu::Buffer,
    xn: wgpu::Buffer,
    q: wgpu::Buffer,
    q2: wgpu::Buffer,
    k: wgpu::Buffer,
    k2: wgpu::Buffer,
    v: wgpu::Buffer,
    v2: wgpu::Buffer,
    attn: wgpu::Buffer,
    t_hidden: wgpu::Buffer,
    gate: wgpu::Buffer,
    up: wgpu::Buffer,
    act: wgpu::Buffer,
    logits: wgpu::Buffer,
    logits_staging: wgpu::Buffer,
    ones: wgpu::Buffer,
    k_cache: Vec<wgpu::Buffer>,
    v_cache: Vec<wgpu::Buffer>,
}

/// iGPU-backed weight accelerator.
pub struct GpuWeightAccel {
    gpu: GpuQ4,
    weights: HashMap<String, Resident>,
    /// Small f32 tensors (norm weights, rope_freqs) for the on-device forward.
    f32buf: HashMap<String, wgpu::Buffer>,
    cfg: Option<GpuDecodeConfig>,
    scratch: Option<DecodeScratch>,
    name: String,
}

impl GpuWeightAccel {
    /// Initialize the GPU context. Returns `None` if no Vulkan device is usable.
    pub fn new() -> Option<Self> {
        let gpu = GpuQ4::new().ok()?;
        let name = gpu.adapter_name().to_string();
        Some(GpuWeightAccel {
            gpu,
            weights: HashMap::new(),
            f32buf: HashMap::new(),
            cfg: None,
            scratch: None,
            name,
        })
    }
}

impl WeightAccel for GpuWeightAccel {
    fn upload_q4_0(&mut self, key: &str, bytes: &[u8], in_dim: usize, out_dim: usize) -> bool {
        if in_dim % 32 != 0 {
            return false;
        }
        match self.gpu.resident_from_q4_0(bytes, in_dim, out_dim) {
            Ok(r) => {
                self.weights.insert(key.to_string(), Resident::Q4(r));
                true
            }
            Err(e) => {
                tracing::warn!(weight = key, error = %e, "GPU Q4_0 upload declined");
                false
            }
        }
    }

    fn upload_q6_k(&mut self, key: &str, bytes: &[u8], in_dim: usize, out_dim: usize) -> bool {
        if in_dim % 256 != 0 {
            return false;
        }
        match self.gpu.resident_from_q6_k(bytes, in_dim, out_dim) {
            Ok(r) => {
                self.weights.insert(key.to_string(), Resident::Q6(r));
                true
            }
            Err(e) => {
                tracing::warn!(weight = key, error = %e, "GPU Q6_K upload declined");
                false
            }
        }
    }

    fn gemv(&self, key: &str, x: &[f32]) -> Option<Vec<f32>> {
        let t = Instant::now();
        let r = match self.weights.get(key)? {
            Resident::Q4(r) => self.gpu.gemv(r, x).ok(),
            Resident::Q6(r) => self.gpu.gemv_q6(r, x).ok(),
        };
        GPU_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }

    fn gemv_batch(&self, calls: &[(&str, &[f32])]) -> Vec<Option<Vec<f32>>> {
        // Fast path: if every call hits an adopted Q4_0 weight, run them in one
        // submission. (q/k/v and gate/up are all Q4_0 in Gemma.) Otherwise loop.
        let q4: Option<Vec<(&ResidentQ4, &[f32])>> = calls
            .iter()
            .map(|(k, x)| match self.weights.get(*k) {
                Some(Resident::Q4(r)) => Some((r, *x)),
                _ => None,
            })
            .collect();
        if let Some(items) = q4 {
            let t = Instant::now();
            let out = self.gpu.gemv_batch_q4(&items);
            GPU_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if let Ok(outs) = out {
                return outs.into_iter().map(Some).collect();
            }
        }
        // Fallback: per-call `gemv` (self-times, no double count).
        calls.iter().map(|(k, x)| self.gemv(k, x)).collect()
    }

    fn resident_count(&self) -> usize {
        self.weights.len()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn upload_f32(&mut self, key: &str, data: &[f32]) {
        use wgpu::util::DeviceExt;
        let buf = self
            .gpu
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("f32-weight"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        self.f32buf.insert(key.to_string(), buf);
    }

    fn configure_decode(&mut self, cfg: GpuDecodeConfig) -> bool {
        use wgpu::util::DeviceExt;
        let dev = self.gpu.device();
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let mk = |n: usize| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scratch"),
                size: (n.max(1) * 4) as u64,
                usage: storage,
                mapped_at_creation: false,
            })
        };
        let hidden = cfg.hidden;
        let n_heads = cfg.n_heads;
        let max_hd = cfg.layers.iter().map(|l| l.head_dim).max().unwrap_or(0);
        let q_dim = n_heads * max_hd;
        let kv_dim = cfg
            .layers
            .iter()
            .map(|l| l.n_kv * l.head_dim)
            .max()
            .unwrap_or(0);
        let logits = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits"),
            size: (cfg.vocab * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let logits_staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits-staging"),
            size: (cfg.vocab * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ones = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ones"),
            contents: bytemuck::cast_slice(&vec![1.0f32; (max_hd / 2).max(1)]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let kv = |l: &strix_core::accel::GpuLayerCfg| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kv-cache"),
                size: (l.n_kv * cfg.max_seq * l.head_dim * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let k_cache = cfg.layers.iter().map(kv).collect();
        let v_cache = cfg.layers.iter().map(kv).collect();
        self.scratch = Some(DecodeScratch {
            h: mk(hidden),
            xn: mk(hidden),
            q: mk(q_dim),
            q2: mk(q_dim),
            k: mk(kv_dim),
            k2: mk(kv_dim),
            v: mk(kv_dim),
            v2: mk(kv_dim),
            attn: mk(q_dim),
            t_hidden: mk(hidden),
            gate: mk(cfg.ffn),
            up: mk(cfg.ffn),
            act: mk(cfg.ffn),
            logits,
            logits_staging,
            ones,
            k_cache,
            v_cache,
        });
        self.cfg = Some(cfg);
        true
    }

    fn decode_step(&mut self, h: &[f32], pos: usize) -> Option<Vec<f32>> {
        let cfg = self.cfg.as_ref()?;
        let s = self.scratch.as_ref()?;
        let gpu = &self.gpu;
        let weights = &self.weights;
        let f32buf = &self.f32buf;
        let hidden = cfg.hidden;
        let n_heads = cfg.n_heads;
        let eps = cfg.eps;

        let q4 = |n: &str| match weights.get(n) {
            Some(Resident::Q4(r)) => r,
            _ => panic!("gpu decode: missing Q4 weight {n}"),
        };
        let q6 = |n: &str| match weights.get(n) {
            Some(Resident::Q6(r)) => r,
            _ => panic!("gpu decode: missing Q6 weight {n}"),
        };
        let ff = |n: &str| -> &wgpu::Buffer {
            f32buf
                .get(n)
                .unwrap_or_else(|| panic!("gpu decode: missing f32 weight {n}"))
        };

        let t = Instant::now();
        gpu.queue().write_buffer(&s.h, 0, bytemuck::cast_slice(h));
        let mut enc = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decode"),
            });

        // Diagnostic: matmul-only floor (skips norms/rope/sdpa/adds → garbage
        // output, but isolates matmul + their barriers from element-wise overhead).
        let matmul_only = std::env::var("STRIX_MATMUL_ONLY").is_ok();
        // Per-op skip flags (garbage output; for isolating each op's cost).
        let skip_sdpa = std::env::var("STRIX_SKIP_SDPA").is_ok();
        let skip_geglu = std::env::var("STRIX_SKIP_GEGLU").is_ok();
        let skip_rope = std::env::var("STRIX_SKIP_ROPE").is_ok();
        let skip_norm = std::env::var("STRIX_SKIP_NORM").is_ok();
        if matmul_only {
            for l in 0..cfg.n_layers {
                let pf = |name: &str| format!("blk.{l}.{name}");
                let q4 = |n: &str| match weights.get(n) {
                    Some(Resident::Q4(r)) => r,
                    _ => panic!("missing {n}"),
                };
                gpu.rec_q4_multi(
                    &mut enc,
                    &[
                        (q4(&pf("attn_q.weight")), &s.xn, &s.q),
                        (q4(&pf("attn_k.weight")), &s.xn, &s.k),
                    ],
                );
                gpu.rec_gemv_q4(
                    &mut enc,
                    q4(&pf("attn_output.weight")),
                    &s.attn,
                    &s.t_hidden,
                );
                gpu.rec_q4_multi(
                    &mut enc,
                    &[
                        (q4(&pf("ffn_gate.weight")), &s.xn, &s.gate),
                        (q4(&pf("ffn_up.weight")), &s.xn, &s.up),
                    ],
                );
                gpu.rec_gemv_q4(&mut enc, q4(&pf("ffn_down.weight")), &s.act, &s.t_hidden);
            }
            gpu.rec_gemv_q6(&mut enc, q6("token_embd.weight"), &s.xn, &s.logits);
            enc.copy_buffer_to_buffer(&s.logits, 0, &s.logits_staging, 0, (cfg.vocab * 4) as u64);
            gpu.queue().submit(Some(enc.finish()));
            let slice = s.logits_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            gpu.device().poll(wgpu::Maintain::Wait);
            rx.recv().ok().and_then(|r| r.ok())?;
            let logits: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            s.logits_staging.unmap();
            GPU_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return Some(logits);
        }

        let n_layers = std::env::var("STRIX_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(cfg.n_layers);
        for l in 0..n_layers {
            let lc = &cfg.layers[l];
            let hd = lc.head_dim;
            let n_kv = lc.n_kv;
            let kv_dim = n_kv * hd;
            let groups = (n_heads / n_kv.max(1)).max(1);
            let len = pos + 1;
            let scale = if cfg.attn_rsqrt {
                1.0 / (hd as f32).sqrt()
            } else {
                1.0
            };
            let pf = |name: &str| format!("blk.{l}.{name}");

            // --- Attention ---
            gpu.rec_rmsnorm(
                &mut enc,
                &s.h,
                ff(&pf("attn_norm.weight")),
                &s.xn,
                1,
                hidden,
                eps,
                true,
            );
            // q/k/(v) all consume xn → one compute pass (independent dispatches).
            if lc.k_eq_v {
                gpu.rec_q4_multi(
                    &mut enc,
                    &[
                        (q4(&pf("attn_q.weight")), &s.xn, &s.q),
                        (q4(&pf("attn_k.weight")), &s.xn, &s.k),
                    ],
                );
            } else {
                gpu.rec_q4_multi(
                    &mut enc,
                    &[
                        (q4(&pf("attn_q.weight")), &s.xn, &s.q),
                        (q4(&pf("attn_k.weight")), &s.xn, &s.k),
                        (q4(&pf("attn_v.weight")), &s.xn, &s.v),
                    ],
                );
            }
            // Q/K/V per-head norms in ONE pass (independent). V reads the *raw*
            // k for k_eq_v (no copy) or the v_proj; it's RMS-normed with no
            // weight. K-norm reads raw k too (V captured before K is overwritten,
            // since they write distinct buffers q2/k2/v2).
            let v_src = if lc.k_eq_v { &s.k } else { &s.v };
            let qn = ff(&pf("attn_q_norm.weight"));
            let kn = ff(&pf("attn_k_norm.weight"));
            if skip_norm {
                // diagnostic: skip per-head norms (garbage output)
            } else if cfg.norm_v {
                gpu.rec_rmsnorm_multi(
                    &mut enc,
                    &[
                        (&s.q, qn, &s.q2, n_heads, hd, true),
                        (&s.k, kn, &s.k2, n_kv, hd, true),
                        (v_src, &s.ones, &s.v2, n_kv, hd, false),
                    ],
                );
            } else {
                gpu.rec_rmsnorm_multi(
                    &mut enc,
                    &[
                        (&s.q, qn, &s.q2, n_heads, hd, true),
                        (&s.k, kn, &s.k2, n_kv, hd, true),
                    ],
                );
                enc.copy_buffer_to_buffer(v_src, 0, &s.v2, 0, (kv_dim * 4) as u64);
            }
            // RoPE q,k in ONE pass (freq factors on global layers).
            let ropef = if lc.is_local {
                &s.ones
            } else {
                ff("rope_freqs.weight")
            };
            if !skip_rope {
                gpu.rec_rope_multi(
                    &mut enc,
                    &[
                        (&s.q2, ropef, hd, n_heads, pos, lc.rope_theta),
                        (&s.k2, ropef, hd, n_kv, pos, lc.rope_theta),
                    ],
                );
            }
            // Append k2,v2 to the token-major KV cache at slot `pos` (one copy
            // each: token `pos`'s block is `[kvh][d]` = exactly k2/v2's layout).
            let dst = (pos * kv_dim * 4) as u64;
            let bytes = (kv_dim * 4) as u64;
            enc.copy_buffer_to_buffer(&s.k2, 0, &s.k_cache[l], dst, bytes);
            enc.copy_buffer_to_buffer(&s.v2, 0, &s.v_cache[l], dst, bytes);
            if !skip_sdpa {
                gpu.rec_sdpa(
                    &mut enc,
                    &s.q2,
                    &s.k_cache[l],
                    &s.v_cache[l],
                    &s.attn,
                    hd,
                    n_heads,
                    len,
                    groups,
                    n_kv,
                    scale,
                );
            }
            gpu.rec_gemv_q4(
                &mut enc,
                q4(&pf("attn_output.weight")),
                &s.attn,
                &s.t_hidden,
            );
            // Fused: h = h + rmsnorm(o)*post_attn_w  (norm + residual, one pass).
            gpu.rec_addnorm(
                &mut enc,
                &s.h,
                &s.t_hidden,
                ff(&pf("post_attention_norm.weight")),
                hidden,
                eps,
                1.0,
            );

            // --- MLP (GeGLU) ---
            gpu.rec_rmsnorm(
                &mut enc,
                &s.h,
                ff(&pf("ffn_norm.weight")),
                &s.xn,
                1,
                hidden,
                eps,
                true,
            );
            // gate/up share xn → one pass.
            gpu.rec_q4_multi(
                &mut enc,
                &[
                    (q4(&pf("ffn_gate.weight")), &s.xn, &s.gate),
                    (q4(&pf("ffn_up.weight")), &s.xn, &s.up),
                ],
            );
            if !skip_geglu {
                gpu.rec_geglu(&mut enc, &s.gate, &s.up, &s.act, cfg.ffn);
            }
            gpu.rec_gemv_q4(&mut enc, q4(&pf("ffn_down.weight")), &s.act, &s.t_hidden);
            // Fused: h = (h + rmsnorm(down)*post_ffw_w) * layer_output_scale.
            gpu.rec_addnorm(
                &mut enc,
                &s.h,
                &s.t_hidden,
                ff(&pf("post_ffw_norm.weight")),
                hidden,
                eps,
                lc.output_scale,
            );
        }

        // Final norm + tied lm_head + soft-cap.
        gpu.rec_rmsnorm(
            &mut enc,
            &s.h,
            ff("output_norm.weight"),
            &s.xn,
            1,
            hidden,
            eps,
            true,
        );
        gpu.rec_gemv_q6(&mut enc, q6("token_embd.weight"), &s.xn, &s.logits);
        if cfg.final_softcap > 0.0 {
            gpu.rec_softcap(&mut enc, &s.logits, cfg.vocab, cfg.final_softcap);
        }
        enc.copy_buffer_to_buffer(&s.logits, 0, &s.logits_staging, 0, (cfg.vocab * 4) as u64);
        gpu.queue().submit(Some(enc.finish()));

        let slice = s.logits_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device().poll(wgpu::Maintain::Wait);
        rx.recv().ok().and_then(|r| r.ok())?;
        let logits: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        s.logits_staging.unmap();
        GPU_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if std::env::var("STRIX_DBG").is_ok() && pos < 3 {
            let (mut bi, mut bv) = (0usize, f32::MIN);
            for (i, &x) in logits.iter().enumerate() {
                if x > bv {
                    bv = x;
                    bi = i;
                }
            }
            eprintln!("[wgpu] pos={pos} argmax={bi} max={bv:.4}");
        }
        Some(logits)
    }
}
