//! Full on-device Gemma decode forward on the raw-Vulkan (ash) path.
//!
//! Mirrors `accel.rs`'s wgpu `decode_step` op-for-op, but records the whole
//! forward as ONE command buffer with ~1µs barriers and a single submit/token —
//! removing the ~14ms/token of wgpu per-compute-pass tax that structurally caps
//! that path at ~10 tok/s. Reuses the exact same WGSL kernels (compiled to
//! SPIR-V via naga in [`crate::ash_gpu`]).
//!
//! UMA makes this simpler than wgpu: every buffer is host-visible *and*
//! device-local, so the hidden state `h` is written and the `logits` read
//! through a mapped pointer — no staging copies, no map_async round-trip.
//!
//! Selected at runtime via `STRIX_ASH=1` (see the CLI's `attach_gpu`).

use std::collections::HashMap;

use ash::vk;
use strix_core::accel::{GpuDecodeConfig, WeightAccel};

use crate::ash_gpu::{AshGpu, Buf, Pipeline};

const QK4_0: usize = 32;
const Q4_0_BYTES: usize = 18;
const QK_K: usize = 256;
const Q6_K_BYTES: usize = 210;
const MAX_GRID_DIM: u32 = 65535;

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = if exp == 0 {
        // Subnormal: value = mant * 2^-24. (Gemma's Q6_K superblock `d` values
        // are tiny — often subnormal f16 — so this branch MUST NOT drop to zero.)
        (mant as f32) * 2.0f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// A resident Q4_0 weight: f16 scales (2/u32) + 4 u32 quants per 32-block.
struct ResQ4 {
    scales: Buf,
    quants: Buf,
    in_dim: usize,
    out_dim: usize,
}

/// A resident Q6_K weight: 16 folded f32 scales + 32 u32 `ql` + 16 u32 `qh` / block.
struct ResQ6 {
    scales: Buf,
    ql: Buf,
    qh: Buf,
    in_dim: usize,
    out_dim: usize,
}

/// Which kernel pipeline an op dispatches.
#[derive(Clone, Copy)]
enum Pk {
    Q4,
    Q6,
    Rmsnorm,
    Addnorm,
    Rope,
    Geglu,
    Softcap,
    Sdpa,
}

/// A recorded forward op (replayed into the command buffer each token).
enum Op {
    Dispatch {
        pk: Pk,
        ds: vk::DescriptorSet,
        gx: u32,
        gy: u32,
    },
    /// Append `src` (k2/v2) to a token-major KV cache at slot `pos`.
    KvCopy {
        src: vk::Buffer,
        dst: vk::Buffer,
        bytes: u64,
    },
    /// A memory+execution barrier at a true data-dependency boundary. Dispatches
    /// between two barriers are independent and overlap on the GPU.
    Barrier,
}

/// A uniform whose contents depend on `pos` and must be rewritten each token.
enum PosU {
    Rope {
        idx: usize,
        head_dim: u32,
        n_heads: u32,
        theta: f32,
    },
    Sdpa {
        idx: usize,
        head_dim: u32,
        groups: u32,
        n_kv: u32,
        scale: f32,
    },
}

/// Per-decode resident state (scratch + KV cache + the recorded op list).
/// Most scratch fields are RAII holders: their `vk::Buffer` handles were captured
/// into descriptor sets at build time; the `Buf`s must outlive the command buffer.
#[allow(dead_code)]
struct State {
    // scratch
    h: Buf,
    xn: Buf,
    q: Buf,
    q2: Buf,
    k: Buf,
    k2: Buf,
    v: Buf,
    v2: Buf,
    attn: Buf,
    t_hidden: Buf,
    gate: Buf,
    up: Buf,
    act: Buf,
    logits: Buf,
    ones: Buf,
    k_cache: Vec<Buf>,
    v_cache: Vec<Buf>,
    // descriptor + uniform storage (kept alive)
    pool: vk::DescriptorPool,
    uniforms: Vec<Buf>,
    ops: Vec<Op>,
    pos_uniforms: Vec<PosU>,
    cb: vk::CommandBuffer,
    vocab: usize,
}

/// iGPU decode accelerator on the raw-Vulkan path.
pub struct AshWeightAccel {
    gpu: AshGpu,
    pipes: Vec<Pipeline>,
    q4: HashMap<String, ResQ4>,
    q6: HashMap<String, ResQ6>,
    f32w: HashMap<String, Buf>,
    cfg: Option<GpuDecodeConfig>,
    state: Option<State>,
    name: String,
}

impl AshWeightAccel {
    /// Initialize the ash context and compile every kernel. `None` if no device.
    pub fn new() -> Option<Self> {
        let gpu = AshGpu::new().ok()?;
        let name = gpu.adapter_name().to_string();
        // Order MUST match the `Pk` enum discriminants.
        use crate::qgemv as k;
        let pipes = vec![
            gpu.build_pipeline(k::SHADER_SG, "main", 4, true).ok()?,
            gpu.build_pipeline(k::SHADER_Q6_SG, "main", 5, true).ok()?,
            gpu.build_pipeline(k::SHADER_RMSNORM, "main", 3, true)
                .ok()?,
            gpu.build_pipeline(k::SHADER_ADDNORM, "main", 3, true)
                .ok()?,
            gpu.build_pipeline(k::SHADER_ROPE, "main", 2, true).ok()?,
            gpu.build_pipeline(k::SHADER_GEGLU, "main", 3, true).ok()?,
            gpu.build_pipeline(k::SHADER_SOFTCAP, "main", 1, true)
                .ok()?,
            gpu.build_pipeline(k::SHADER_SDPA256, "main", 4, true)
                .ok()?,
        ];
        Some(Self {
            gpu,
            pipes,
            q4: HashMap::new(),
            q6: HashMap::new(),
            f32w: HashMap::new(),
            cfg: None,
            state: None,
            name,
        })
    }

    fn pipe(&self, pk: Pk) -> &Pipeline {
        &self.pipes[pk as usize]
    }

    /// Scratch / KV buffer (read-write, transfer-capable).
    fn storage(&self, size_bytes: u64) -> Buf {
        self.gpu
            .alloc(
                size_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .expect("ash scratch alloc")
    }

    /// Resident read-only weight buffer, initialized from `data`.
    fn storage_ro<T: Copy>(&self, data: &[T]) -> Buf {
        let b = self
            .gpu
            .alloc(
                std::mem::size_of_val(data) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .expect("ash weight alloc");
        b.write(data);
        b
    }
}

impl WeightAccel for AshWeightAccel {
    fn upload_q4_0(&mut self, key: &str, bytes: &[u8], in_dim: usize, out_dim: usize) -> bool {
        if in_dim % QK4_0 != 0 {
            return false;
        }
        let nblocks = in_dim / QK4_0;
        let total = nblocks * out_dim;
        if bytes.len() != total * Q4_0_BYTES {
            return false;
        }
        let mut scales = vec![0u32; total.div_ceil(2)];
        let mut quants = vec![0u32; total * 4];
        for (b, blk) in bytes.chunks_exact(Q4_0_BYTES).enumerate() {
            let h = u16::from_le_bytes([blk[0], blk[1]]) as u32;
            scales[b >> 1] |= h << (16 * (b & 1));
            let qs = &blk[2..18];
            for w in 0..4 {
                quants[b * 4 + w] =
                    u32::from_le_bytes([qs[w * 4], qs[w * 4 + 1], qs[w * 4 + 2], qs[w * 4 + 3]]);
            }
        }
        let sb = self.storage_ro(&scales);
        let qb = self.storage_ro(&quants);
        self.q4.insert(
            key.to_string(),
            ResQ4 {
                scales: sb,
                quants: qb,
                in_dim,
                out_dim,
            },
        );
        true
    }

    fn upload_q6_k(&mut self, key: &str, bytes: &[u8], in_dim: usize, out_dim: usize) -> bool {
        if in_dim % QK_K != 0 {
            return false;
        }
        let nblocks = in_dim / QK_K;
        let total = nblocks * out_dim;
        if bytes.len() != total * Q6_K_BYTES {
            return false;
        }
        let mut scales = vec![0.0f32; total * 16];
        let mut ql = vec![0u32; total * 32];
        let mut qh = vec![0u32; total * 16];
        for (b, blk) in bytes.chunks_exact(Q6_K_BYTES).enumerate() {
            let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
            for j in 0..16 {
                scales[b * 16 + j] = d * (blk[192 + j] as i8) as f32;
            }
            for w in 0..32 {
                ql[b * 32 + w] = u32::from_le_bytes([
                    blk[w * 4],
                    blk[w * 4 + 1],
                    blk[w * 4 + 2],
                    blk[w * 4 + 3],
                ]);
            }
            for w in 0..16 {
                qh[b * 16 + w] = u32::from_le_bytes([
                    blk[128 + w * 4],
                    blk[128 + w * 4 + 1],
                    blk[128 + w * 4 + 2],
                    blk[128 + w * 4 + 3],
                ]);
            }
        }
        let scb = self.storage_ro(&scales);
        let qlb = self.storage_ro(&ql);
        let qhb = self.storage_ro(&qh);
        self.q6.insert(
            key.to_string(),
            ResQ6 {
                scales: scb,
                ql: qlb,
                qh: qhb,
                in_dim,
                out_dim,
            },
        );
        true
    }

    fn gemv(&self, _key: &str, _x: &[f32]) -> Option<Vec<f32>> {
        // The ash path runs the whole forward in `decode_step`; per-weight gemv
        // is intentionally unsupported (would defeat the single-submit design).
        None
    }

    fn resident_count(&self) -> usize {
        self.q4.len() + self.q6.len()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn upload_f32(&mut self, key: &str, data: &[f32]) {
        let b = self.storage_ro(data);
        self.f32w.insert(key.to_string(), b);
    }

    fn configure_decode(&mut self, cfg: GpuDecodeConfig) -> bool {
        self.build_state(&cfg);
        self.cfg = Some(cfg);
        self.state.is_some()
    }

    fn decode_step(&mut self, h: &[f32], pos: usize) -> Option<Vec<f32>> {
        let t = std::time::Instant::now();
        let logits = self.run_decode(h, pos)?;
        crate::accel::add_gpu_nanos(t.elapsed().as_nanos() as u64);
        Some(logits)
    }
}

/// 2D grid covering `out_dim` rows (dodges the 65535-per-dim workgroup limit).
fn grid(out_dim: usize) -> (u32, u32) {
    let gx = (out_dim as u32).min(MAX_GRID_DIM);
    (gx, (out_dim as u32).div_ceil(gx))
}

impl AshWeightAccel {
    fn build_state(&mut self, cfg: &GpuDecodeConfig) {
        let hidden = cfg.hidden;
        let n_heads = cfg.n_heads;
        let max_hd = cfg.layers.iter().map(|l| l.head_dim).max().unwrap_or(0);
        let q_dim = n_heads * max_hd;
        let kv_dim_max = cfg
            .layers
            .iter()
            .map(|l| l.n_kv * l.head_dim)
            .max()
            .unwrap_or(0);

        // Scratch + KV cache (all host-visible+device-local on UMA).
        let mk = |n: usize| self.storage((n.max(1) * 4) as u64);
        let h = mk(hidden);
        let xn = mk(hidden);
        let q = mk(q_dim);
        let q2 = mk(q_dim);
        let k = mk(kv_dim_max);
        let k2 = mk(kv_dim_max);
        let v = mk(kv_dim_max);
        let v2 = mk(kv_dim_max);
        let attn = mk(q_dim);
        let t_hidden = mk(hidden);
        let gate = mk(cfg.ffn);
        let up = mk(cfg.ffn);
        let act = mk(cfg.ffn);
        let logits = mk(cfg.vocab);
        let ones_buf = self.storage(((max_hd / 2).max(1) * 4) as u64);
        ones_buf.write(&vec![1.0f32; (max_hd / 2).max(1)]);
        let k_cache: Vec<Buf> = cfg
            .layers
            .iter()
            .map(|l| self.storage((l.n_kv * cfg.max_seq * l.head_dim * 4) as u64))
            .collect();
        let v_cache: Vec<Buf> = cfg
            .layers
            .iter()
            .map(|l| self.storage((l.n_kv * cfg.max_seq * l.head_dim * 4) as u64))
            .collect();

        let sets_cap = (cfg.n_layers * 22 + 8) as u32;
        let pool = self
            .gpu
            .create_descriptor_pool(sets_cap, sets_cap * 5, sets_cap)
            .expect("ash descriptor pool");

        let mut ops: Vec<Op> = Vec::new();
        let mut uniforms: Vec<Buf> = Vec::new();
        let mut pos_uniforms: Vec<PosU> = Vec::new();
        let eps_bits = cfg.eps.to_bits();

        // Closure-free dispatch helper (borrows self.gpu/self.pipes immutably).
        // Returns the uniform index (for later pos updates).
        macro_rules! dispatch {
            ($pk:expr, $storage:expr, $words:expr, $gx:expr, $gy:expr) => {{
                let words: Vec<u32> = $words;
                let ub = self
                    .gpu
                    .alloc(
                        (words.len() * 4).max(16) as u64,
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                    )
                    .expect("ash uniform");
                ub.write(&words);
                let storage: Vec<vk::Buffer> = $storage;
                let ds = self
                    .gpu
                    .alloc_set(pool, self.pipe($pk), &storage, Some(ub.buffer))
                    .expect("ash ds");
                let idx = uniforms.len();
                uniforms.push(ub);
                ops.push(Op::Dispatch {
                    pk: $pk,
                    ds,
                    gx: $gx,
                    gy: $gy,
                });
                idx
            }};
        }
        // Barrier marker between dependency groups (dispatches within a group are
        // independent and overlap; only cross-group edges need a barrier).
        macro_rules! bar {
            () => {
                ops.push(Op::Barrier)
            };
        }

        let q4 = |n: &str| {
            self.q4
                .get(n)
                .unwrap_or_else(|| panic!("ash: missing Q4 {n}"))
        };
        let q6 = |n: &str| {
            self.q6
                .get(n)
                .unwrap_or_else(|| panic!("ash: missing Q6 {n}"))
        };
        let ff = |n: &str| {
            self.f32w
                .get(n)
                .unwrap_or_else(|| panic!("ash: missing f32 {n}"))
                .buffer
        };

        let q4_op = |w: &ResQ4, x: vk::Buffer, y: vk::Buffer| {
            let (gx, gy) = grid(w.out_dim);
            (
                vec![w.scales.buffer, w.quants.buffer, x, y],
                vec![w.in_dim as u32, w.out_dim as u32, gx, 0],
                gx,
                gy,
            )
        };

        let n_layers = std::env::var("STRIX_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(cfg.n_layers);
        // Diagnostics: skip element-wise groups to isolate the matmul floor.
        let skip_norm = std::env::var("STRIX_SKIP_NORM").is_ok();
        let skip_sdpa = std::env::var("STRIX_SKIP_SDPA").is_ok();
        let skip_rope = std::env::var("STRIX_SKIP_ROPE").is_ok();
        let skip_geglu = std::env::var("STRIX_SKIP_GEGLU").is_ok();
        for l in 0..n_layers {
            let lc = &cfg.layers[l];
            let hd = lc.head_dim;
            let n_kv = lc.n_kv;
            let kv_dim = n_kv * hd;
            let half = hd / 2;
            let pf = |name: &str| format!("blk.{l}.{name}");
            let theta = lc.rope_theta;

            // 1. attn input rmsnorm: h -> xn
            if !skip_norm {
                dispatch!(
                    Pk::Rmsnorm,
                    vec![h.buffer, ff(&pf("attn_norm.weight")), xn.buffer],
                    vec![hidden as u32, 1, 1, eps_bits],
                    1,
                    1
                );
            }
            bar!(); // xn ready → projections
                    // 2. Q/K(/V) projections from xn (independent → no inter-barrier).
            let (s, w, gx, gy) = q4_op(q4(&pf("attn_q.weight")), xn.buffer, q.buffer);
            dispatch!(Pk::Q4, s, w, gx, gy);
            let (s, w, gx, gy) = q4_op(q4(&pf("attn_k.weight")), xn.buffer, k.buffer);
            dispatch!(Pk::Q4, s, w, gx, gy);
            if !lc.k_eq_v {
                let (s, w, gx, gy) = q4_op(q4(&pf("attn_v.weight")), xn.buffer, v.buffer);
                dispatch!(Pk::Q4, s, w, gx, gy);
            }
            bar!(); // q/k/v ready → per-head norms
                    // 3. Q/K per-head norms (+ V norm or copy).
            let v_src = if lc.k_eq_v { k.buffer } else { v.buffer };
            if !skip_norm {
                dispatch!(
                    Pk::Rmsnorm,
                    vec![q.buffer, ff(&pf("attn_q_norm.weight")), q2.buffer],
                    vec![hd as u32, n_heads as u32, 1, eps_bits],
                    n_heads as u32,
                    1
                );
                dispatch!(
                    Pk::Rmsnorm,
                    vec![k.buffer, ff(&pf("attn_k_norm.weight")), k2.buffer],
                    vec![hd as u32, n_kv as u32, 1, eps_bits],
                    n_kv as u32,
                    1
                );
            }
            if cfg.norm_v && !skip_norm {
                dispatch!(
                    Pk::Rmsnorm,
                    vec![v_src, ones_buf.buffer, v2.buffer],
                    vec![hd as u32, n_kv as u32, 0, eps_bits],
                    n_kv as u32,
                    1
                );
            } else if !cfg.norm_v {
                ops.push(Op::KvCopy {
                    src: v_src,
                    dst: v2.buffer,
                    bytes: (kv_dim * 4) as u64,
                });
                // NB: this copy targets v2 at offset 0 (handled in record).
            }
            bar!(); // q2/k2/v2 ready → rope
                    // 4. RoPE q2, k2 (pos filled per token).
            let ropef = if lc.is_local {
                ones_buf.buffer
            } else {
                ff("rope_freqs.weight")
            };
            if !skip_rope {
                let rq = dispatch!(
                    Pk::Rope,
                    vec![q2.buffer, ropef],
                    vec![hd as u32, n_heads as u32, 0, theta.to_bits()],
                    ((n_heads * half) as u32).div_ceil(64),
                    1
                );
                pos_uniforms.push(PosU::Rope {
                    idx: rq,
                    head_dim: hd as u32,
                    n_heads: n_heads as u32,
                    theta,
                });
                let rk = dispatch!(
                    Pk::Rope,
                    vec![k2.buffer, ropef],
                    vec![hd as u32, n_kv as u32, 0, theta.to_bits()],
                    ((n_kv * half) as u32).div_ceil(64),
                    1
                );
                pos_uniforms.push(PosU::Rope {
                    idx: rk,
                    head_dim: hd as u32,
                    n_heads: n_kv as u32,
                    theta,
                });
            }
            bar!(); // q2 roped, k2 roped → KV append
                    // 5. Append k2,v2 to the token-major KV cache at slot pos.
            ops.push(Op::KvCopy {
                src: k2.buffer,
                dst: k_cache[l].buffer,
                bytes: (kv_dim * 4) as u64,
            });
            ops.push(Op::KvCopy {
                src: v2.buffer,
                dst: v_cache[l].buffer,
                bytes: (kv_dim * 4) as u64,
            });
            bar!(); // KV cache written → SDPA reads it
                    // 6. SDPA (len filled per token).
            let groups = (n_heads / n_kv.max(1)).max(1);
            let scale = if cfg.attn_rsqrt {
                1.0 / (hd as f32).sqrt()
            } else {
                1.0
            };
            if !skip_sdpa {
                let sd = dispatch!(
                    Pk::Sdpa,
                    vec![q2.buffer, k_cache[l].buffer, v_cache[l].buffer, attn.buffer],
                    vec![
                        hd as u32,
                        1,
                        groups as u32,
                        n_kv as u32,
                        scale.to_bits(),
                        0,
                        0,
                        0
                    ],
                    n_heads as u32,
                    1
                );
                pos_uniforms.push(PosU::Sdpa {
                    idx: sd,
                    head_dim: hd as u32,
                    groups: groups as u32,
                    n_kv: n_kv as u32,
                    scale,
                });
            }
            bar!(); // attn (SDPA out) ready → output projection
                    // 7. attn output projection: attn -> t_hidden
            let (s, w, gx, gy) = q4_op(q4(&pf("attn_output.weight")), attn.buffer, t_hidden.buffer);
            dispatch!(Pk::Q4, s, w, gx, gy);
            bar!(); // t_hidden ready → fused residual+norm
                    // 8. h = h + rmsnorm(t_hidden)*post_attn_w
            if !skip_norm {
                dispatch!(
                    Pk::Addnorm,
                    vec![
                        h.buffer,
                        t_hidden.buffer,
                        ff(&pf("post_attention_norm.weight"))
                    ],
                    vec![hidden as u32, eps_bits, 1.0f32.to_bits(), 0],
                    1,
                    1
                );
            }
            bar!(); // h updated → ffn norm
                    // 9. ffn input rmsnorm: h -> xn
            if !skip_norm {
                dispatch!(
                    Pk::Rmsnorm,
                    vec![h.buffer, ff(&pf("ffn_norm.weight")), xn.buffer],
                    vec![hidden as u32, 1, 1, eps_bits],
                    1,
                    1
                );
            }
            bar!(); // xn ready → gate/up
                    // 10. gate/up from xn (independent → no inter-barrier).
            let (s, w, gx, gy) = q4_op(q4(&pf("ffn_gate.weight")), xn.buffer, gate.buffer);
            dispatch!(Pk::Q4, s, w, gx, gy);
            let (s, w, gx, gy) = q4_op(q4(&pf("ffn_up.weight")), xn.buffer, up.buffer);
            dispatch!(Pk::Q4, s, w, gx, gy);
            bar!(); // gate/up ready → geglu
                    // 11. geglu
            if !skip_geglu {
                dispatch!(
                    Pk::Geglu,
                    vec![gate.buffer, up.buffer, act.buffer],
                    vec![cfg.ffn as u32, 0, 0, 0],
                    (cfg.ffn as u32).div_ceil(64),
                    1
                );
            }
            bar!(); // act ready → ffn down
                    // 12. ffn down: act -> t_hidden
            let (s, w, gx, gy) = q4_op(q4(&pf("ffn_down.weight")), act.buffer, t_hidden.buffer);
            dispatch!(Pk::Q4, s, w, gx, gy);
            bar!(); // t_hidden ready → fused residual+norm
                    // 13. h = (h + rmsnorm(t_hidden)*post_ffw_w) * output_scale
            if !skip_norm {
                dispatch!(
                    Pk::Addnorm,
                    vec![h.buffer, t_hidden.buffer, ff(&pf("post_ffw_norm.weight"))],
                    vec![hidden as u32, eps_bits, lc.output_scale.to_bits(), 0],
                    1,
                    1
                );
            }
            bar!(); // h updated → next layer (or final norm)
        }

        // Final norm + tied lm_head (Q6) + optional soft-cap.
        dispatch!(
            Pk::Rmsnorm,
            vec![h.buffer, ff("output_norm.weight"), xn.buffer],
            vec![hidden as u32, 1, 1, eps_bits],
            1,
            1
        );
        bar!(); // xn ready → lm_head
        let emb = q6("token_embd.weight");
        let (gx, gy) = grid(emb.out_dim);
        dispatch!(
            Pk::Q6,
            vec![
                emb.scales.buffer,
                emb.ql.buffer,
                emb.qh.buffer,
                xn.buffer,
                logits.buffer
            ],
            vec![emb.in_dim as u32, emb.out_dim as u32, gx, 0],
            gx,
            gy
        );
        if cfg.final_softcap > 0.0 {
            bar!(); // logits ready → soft-cap
            dispatch!(
                Pk::Softcap,
                vec![logits.buffer],
                vec![cfg.vocab as u32, cfg.final_softcap.to_bits(), 0, 0],
                (cfg.vocab as u32).div_ceil(64),
                1
            );
        }

        let cb = self.gpu.cmd_buffer().expect("ash cmd buffer");
        self.state = Some(State {
            h,
            xn,
            q,
            q2,
            k,
            k2,
            v,
            v2,
            attn,
            t_hidden,
            gate,
            up,
            act,
            logits,
            ones: ones_buf,
            k_cache,
            v_cache,
            pool,
            uniforms,
            ops,
            pos_uniforms,
            cb,
            vocab: cfg.vocab,
        });
    }

    fn run_decode(&mut self, h: &[f32], pos: usize) -> Option<Vec<f32>> {
        let s = self.state.as_ref()?;
        s.h.write(h);

        // Refresh pos-dependent uniforms.
        for pu in &s.pos_uniforms {
            match *pu {
                PosU::Rope {
                    idx,
                    head_dim,
                    n_heads,
                    theta,
                } => s.uniforms[idx].write(&[head_dim, n_heads, pos as u32, theta.to_bits()]),
                PosU::Sdpa {
                    idx,
                    head_dim,
                    groups,
                    n_kv,
                    scale,
                } => s.uniforms[idx].write(&[
                    head_dim,
                    (pos + 1) as u32,
                    groups,
                    n_kv,
                    scale.to_bits(),
                    0,
                    0,
                    0,
                ]),
            }
        }

        let nobar = std::env::var("STRIX_NOBAR").is_ok();
        let d = &self.gpu.device;
        let cb = s.cb;
        unsafe {
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .ok()?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .ok()?;
            let execbar = std::env::var("STRIX_EXECBAR").is_ok();
            let (src_acc, dst_acc) = if execbar {
                (vk::AccessFlags::empty(), vk::AccessFlags::empty())
            } else {
                (
                    vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
                )
            };
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(src_acc)
                .dst_access_mask(dst_acc)];
            let cbar = std::env::var("STRIX_CBAR").is_ok();
            let stages = if cbar {
                vk::PipelineStageFlags::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER
            };
            for op in &s.ops {
                match op {
                    Op::Dispatch { pk, ds, gx, gy } => {
                        let p = self.pipe(*pk);
                        d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, p.pipeline);
                        d.cmd_bind_descriptor_sets(
                            cb,
                            vk::PipelineBindPoint::COMPUTE,
                            p.layout,
                            0,
                            &[*ds],
                            &[],
                        );
                        d.cmd_dispatch(cb, *gx, *gy, 1);
                    }
                    Op::KvCopy { src, dst, bytes } => {
                        let dst_off = pos as u64 * *bytes;
                        // v2-copy (non-norm_v) targets offset 0; KV-cache targets pos.
                        let dst_off = if *dst == s.v2.buffer { 0 } else { dst_off };
                        d.cmd_copy_buffer(
                            cb,
                            *src,
                            *dst,
                            &[vk::BufferCopy::default()
                                .src_offset(0)
                                .dst_offset(dst_off)
                                .size(*bytes)],
                        );
                    }
                    Op::Barrier => {
                        if !nobar {
                            d.cmd_pipeline_barrier(
                                cb,
                                stages,
                                stages,
                                vk::DependencyFlags::empty(),
                                &barrier,
                                &[],
                                &[],
                            );
                        }
                    }
                }
            }
            d.end_command_buffer(cb).ok()?;
        }
        self.gpu.run(cb).ok()?;
        let logits = s.logits.read::<f32>(s.vocab);
        if std::env::var("STRIX_DBG").is_ok() && pos < 3 {
            let (am, mx) = argmax(&logits);
            eprintln!("[ash] pos={pos} argmax={am} max={mx:.4}");
        }
        Some(logits)
    }
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut bi = 0;
    let mut bv = f32::MIN;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    (bi, bv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ash_gpu::Pipeline;

    /// Run one kernel dispatch and return the contents of the output buffers.
    fn run1(
        gpu: &AshGpu,
        pipe: &Pipeline,
        storage: &[&Buf],
        words: &[u32],
        gx: u32,
        gy: u32,
        gz: u32,
    ) {
        let ub = gpu
            .alloc(
                (words.len() * 4).max(16) as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
            .unwrap();
        ub.write(words);
        let pool = gpu.create_descriptor_pool(1, 8, 1).unwrap();
        let bufs: Vec<vk::Buffer> = storage.iter().map(|b| b.buffer).collect();
        let ds = gpu.alloc_set(pool, pipe, &bufs, Some(ub.buffer)).unwrap();
        let cb = gpu.cmd_buffer().unwrap();
        unsafe {
            let d = &gpu.device;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            d.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[ds],
                &[],
            );
            d.cmd_dispatch(cb, gx, gy, gz);
            d.end_command_buffer(cb).unwrap();
        }
        gpu.run(cb).unwrap();
    }

    fn maxerr(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn ash_kernels_match_cpu() {
        let gpu = AshGpu::new().expect("device");
        use crate::qgemv as k;
        let sb = |n: u64| {
            gpu.alloc(
                n,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap()
        };

        // ---- RMSNorm (full vector, with weight) ----
        {
            let dim = 2560usize;
            let eps = 1e-6f32;
            let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
            let w: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 * 0.003).cos()).collect();
            let ms = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let s = 1.0 / (ms + eps).sqrt();
            let want: Vec<f32> = (0..dim).map(|i| x[i] * s * w[i]).collect();

            let xb = sb((dim * 4) as u64);
            let wb = sb((dim * 4) as u64);
            let yb = sb((dim * 4) as u64);
            xb.write(&x);
            wb.write(&w);
            let p = gpu
                .build_pipeline(k::SHADER_RMSNORM, "main", 3, true)
                .unwrap();
            run1(
                &gpu,
                &p,
                &[&xb, &wb, &yb],
                &[dim as u32, 1, 1, eps.to_bits()],
                1,
                1,
                1,
            );
            let got = yb.read::<f32>(dim);
            eprintln!("rmsnorm maxerr {:.2e}", maxerr(&want, &got));
            assert!(maxerr(&want, &got) < 1e-3, "rmsnorm");
        }

        // ---- AddNorm: h = (h + rmsnorm(x)*w) * scale ----
        {
            let dim = 2560usize;
            let eps = 1e-6f32;
            let scale = 0.7f32;
            let h0: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
            let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.013).sin()).collect();
            let w: Vec<f32> = (0..dim).map(|i| 0.4 + (i as f32 * 0.005).sin()).collect();
            let ms = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let s = 1.0 / (ms + eps).sqrt();
            let want: Vec<f32> = (0..dim)
                .map(|i| (h0[i] + x[i] * s * w[i]) * scale)
                .collect();

            let hb = sb((dim * 4) as u64);
            let xb = sb((dim * 4) as u64);
            let wb = sb((dim * 4) as u64);
            hb.write(&h0);
            xb.write(&x);
            wb.write(&w);
            let p = gpu
                .build_pipeline(k::SHADER_ADDNORM, "main", 3, true)
                .unwrap();
            run1(
                &gpu,
                &p,
                &[&hb, &xb, &wb],
                &[dim as u32, eps.to_bits(), scale.to_bits(), 0],
                1,
                1,
                1,
            );
            let got = hb.read::<f32>(dim);
            eprintln!("addnorm maxerr {:.2e}", maxerr(&want, &got));
            assert!(maxerr(&want, &got) < 1e-3, "addnorm");
        }

        // ---- GeGLU ----
        {
            let n = 4096usize;
            let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
            let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();
            let want: Vec<f32> = (0..n)
                .map(|i| {
                    let g = gate[i];
                    let gt = 0.5 * g * (1.0 + (0.7978845608 * (g + 0.044715 * g * g * g)).tanh());
                    gt * up[i]
                })
                .collect();
            let gb = sb((n * 4) as u64);
            let ub = sb((n * 4) as u64);
            let ob = sb((n * 4) as u64);
            gb.write(&gate);
            ub.write(&up);
            let p = gpu
                .build_pipeline(k::SHADER_GEGLU, "main", 3, true)
                .unwrap();
            run1(
                &gpu,
                &p,
                &[&gb, &ub, &ob],
                &[n as u32, 0, 0, 0],
                (n as u32).div_ceil(64),
                1,
                1,
            );
            let got = ob.read::<f32>(n);
            eprintln!("geglu maxerr {:.2e}", maxerr(&want, &got));
            assert!(maxerr(&want, &got) < 1e-3, "geglu");
        }

        // ---- RoPE (single head, no freq factors) ----
        {
            let hd = 256usize;
            let half = hd / 2;
            let theta = 1.0e6f32;
            let pos = 7usize;
            let v: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.05).sin()).collect();
            let ff = vec![1.0f32; half];
            let mut want = v.clone();
            for j in 0..half {
                let inv = theta.powf(-2.0 * j as f32 / hd as f32) / ff[j];
                let ang = pos as f32 * inv;
                let (s, c) = (ang.sin(), ang.cos());
                let x1 = v[j];
                let x2 = v[j + half];
                want[j] = x1 * c - x2 * s;
                want[j + half] = x2 * c + x1 * s;
            }
            let vb = sb((hd * 4) as u64);
            let fb = sb((half * 4) as u64);
            vb.write(&v);
            fb.write(&ff);
            let p = gpu.build_pipeline(k::SHADER_ROPE, "main", 2, true).unwrap();
            run1(
                &gpu,
                &p,
                &[&vb, &fb],
                &[hd as u32, 1, pos as u32, theta.to_bits()],
                ((half) as u32).div_ceil(64),
                1,
                1,
            );
            let got = vb.read::<f32>(hd);
            eprintln!("rope maxerr {:.2e}", maxerr(&want, &got));
            assert!(maxerr(&want, &got) < 1e-3, "rope");
        }

        // ---- SDPA (single query head, no GQA) ----
        {
            let hd = 128usize;
            let len = 9usize;
            let n_kv = 1usize;
            let groups = 1usize;
            let scale = 1.0 / (hd as f32).sqrt();
            let q: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.03).sin()).collect();
            let kc: Vec<f32> = (0..len * hd).map(|i| (i as f32 * 0.017).cos()).collect();
            let vc: Vec<f32> = (0..len * hd).map(|i| (i as f32 * 0.011).sin()).collect();
            // CPU reference.
            let mut scores = vec![0.0f32; len];
            for t in 0..len {
                let mut s = 0.0;
                for d in 0..hd {
                    s += q[d] * kc[t * hd + d];
                }
                scores[t] = s * scale;
            }
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                den += *s;
            }
            let mut want = vec![0.0f32; hd];
            for d in 0..hd {
                let mut acc = 0.0;
                for t in 0..len {
                    acc += scores[t] * vc[t * hd + d];
                }
                want[d] = acc / den;
            }
            let qb = sb((hd * 4) as u64);
            let kb = sb((len * hd * 4) as u64);
            let vb = sb((len * hd * 4) as u64);
            let ob = sb((hd * 4) as u64);
            qb.write(&q);
            kb.write(&kc);
            vb.write(&vc);
            let p = gpu.build_pipeline(k::SHADER_SDPA, "main", 4, true).unwrap();
            run1(
                &gpu,
                &p,
                &[&qb, &kb, &vb, &ob],
                &[
                    hd as u32,
                    len as u32,
                    groups as u32,
                    n_kv as u32,
                    scale.to_bits(),
                    0,
                    0,
                    0,
                ],
                1,
                1,
                1,
            );
            let got = ob.read::<f32>(hd);
            eprintln!("sdpa maxerr {:.2e}", maxerr(&want, &got));
            assert!(maxerr(&want, &got) < 1e-3, "sdpa");
        }

        // ---- Q6_K GEMV ----
        {
            let in_dim = 512usize;
            // > 65535 to exercise the 2D workgroup grid (the lm_head path).
            let out_dim = 200_000usize;
            let nblk = in_dim / 256;
            let total = nblk * out_dim;
            // Synthetic Q6_K bytes.
            let mut bytes = vec![0u8; total * Q6_K_BYTES];
            let mut seed = 0x1234_5678u64;
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 33) as u32
            };
            for blk in bytes.chunks_exact_mut(Q6_K_BYTES) {
                for b in blk[..192].iter_mut() {
                    *b = (next() & 0xff) as u8;
                }
                for j in 0..16 {
                    blk[192 + j] = ((next() % 64) as i32 - 32) as i8 as u8;
                }
                let d = 0.02f32 + (next() % 16) as f32 * 0.002;
                // f16 of d
                let bits = d.to_bits();
                let h = (((bits >> 16) & 0x8000)
                    | ((((bits >> 23) as i32 - 127 + 15) as u32) << 10)
                    | ((bits & 0x7fffff) >> 13)) as u16;
                blk[208] = (h & 0xff) as u8;
                blk[209] = (h >> 8) as u8;
            }
            // Repack exactly as upload_q6_k.
            let mut scales = vec![0.0f32; total * 16];
            let mut ql = vec![0u32; total * 32];
            let mut qh = vec![0u32; total * 16];
            for (b, blk) in bytes.chunks_exact(Q6_K_BYTES).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
                for j in 0..16 {
                    scales[b * 16 + j] = d * (blk[192 + j] as i8) as f32;
                }
                for w in 0..32 {
                    ql[b * 32 + w] = u32::from_le_bytes([
                        blk[w * 4],
                        blk[w * 4 + 1],
                        blk[w * 4 + 2],
                        blk[w * 4 + 3],
                    ]);
                }
                for w in 0..16 {
                    qh[b * 16 + w] = u32::from_le_bytes([
                        blk[128 + w * 4],
                        blk[128 + w * 4 + 1],
                        blk[128 + w * 4 + 2],
                        blk[128 + w * 4 + 3],
                    ]);
                }
            }
            let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.007).sin()).collect();
            // CPU reference using the shader's exact dequant formula.
            let byte_at = |word: u32, idx: usize| (word >> ((idx & 3) * 8)) & 0xff;
            let mut want = vec![0.0f32; out_dim];
            for (row, wr) in want.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for b in 0..nblk {
                    let blk = row * nblk + b;
                    let scbase = blk * 16;
                    let qlbase = blk * 32;
                    let qhbase = blk * 16;
                    let xbase = b * 256;
                    for half in 0..2usize {
                        for l in 0..32usize {
                            let is = l / 16;
                            let qli0 = half * 64 + l;
                            let qli1 = half * 64 + l + 32;
                            let qhi = half * 32 + l;
                            let qlb0 = byte_at(ql[qlbase + qli0 / 4], qli0);
                            let qlb1 = byte_at(ql[qlbase + qli1 / 4], qli1);
                            let qhb = byte_at(qh[qhbase + qhi / 4], qhi);
                            let q1 = ((qlb0 & 0x0f) | ((qhb & 3) << 4)) as i32 - 32;
                            let q2 = ((qlb1 & 0x0f) | (((qhb >> 2) & 3) << 4)) as i32 - 32;
                            let q3 = ((qlb0 >> 4) | (((qhb >> 4) & 3) << 4)) as i32 - 32;
                            let q4 = ((qlb1 >> 4) | (((qhb >> 6) & 3) << 4)) as i32 - 32;
                            let posv = half * 128 + l;
                            let si = half * 8 + is;
                            acc += scales[scbase + si] * q1 as f32 * x[xbase + posv]
                                + scales[scbase + si + 2] * q2 as f32 * x[xbase + posv + 32]
                                + scales[scbase + si + 4] * q3 as f32 * x[xbase + posv + 64]
                                + scales[scbase + si + 6] * q4 as f32 * x[xbase + posv + 96];
                        }
                    }
                }
                *wr = acc;
            }

            let su = vk::BufferUsageFlags::STORAGE_BUFFER;
            let scb = gpu.alloc((scales.len() * 4) as u64, su).unwrap();
            let qlb = gpu.alloc((ql.len() * 4) as u64, su).unwrap();
            let qhb = gpu.alloc((qh.len() * 4) as u64, su).unwrap();
            let xb = sb((in_dim * 4) as u64);
            let yb = sb((out_dim * 4) as u64);
            scb.write(&scales);
            qlb.write(&ql);
            qhb.write(&qh);
            xb.write(&x);
            let (gx, gy) = grid(out_dim);
            let p = gpu
                .build_pipeline(k::SHADER_Q6_SG, "main", 5, true)
                .unwrap();
            run1(
                &gpu,
                &p,
                &[&scb, &qlb, &qhb, &xb, &yb],
                &[in_dim as u32, out_dim as u32, gx, 0],
                gx,
                gy,
                1,
            );
            let got = yb.read::<f32>(out_dim);
            let scale = want.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
            eprintln!(
                "q6 maxerr {:.2e} (rel {:.2e})",
                maxerr(&want, &got),
                maxerr(&want, &got) / scale
            );
            assert!(maxerr(&want, &got) / scale < 1e-3, "q6");
        }
    }
}
