//! SmolLM3-3B (`smollm3`) — a Llama-architecture transformer with GQA, tied
//! embeddings, and **NoPE**: RoPE is skipped on every 4th layer (the layers `il`
//! where `(il + 1) % 4 == 0`). No QK-norm, no biases, no logit softcap, plain
//! SwiGLU FFN. kq_scale = 1/sqrt(head_dim).
//!
//! Verified against refs/llama.cpp/src/models/smollm3.cpp. CPU-only on-the-fly
//! dequant forward (the WMMA/int8 GPU path is mellum-specific and not wired here).
//! Tokenizer is gpt2-BPE (StrixTokenizer is SentencePiece-only) → drive with raw
//! token IDs via STRIX_QWEN_IDS, like mellum.

use rayon::prelude::*;
use strix_core::accel::{GpuDecodeConfig, GpuLayerCfg, WeightAccel};
use strix_core::backend::Decoder;
use strix_core::error::{Result, StrixError};
use strix_core::sampler::Logits;
use strix_models::ggml_quant::{dequantize_into, GgmlType};
use strix_models::gguf::GgufFile;

fn meta_u32(g: &GgufFile, k: &str) -> Result<usize> {
    g.meta_u32(k).map(|v| v as usize)
}
fn meta_f32_or(g: &GgufFile, k: &str, d: f32) -> f32 {
    g.meta_f32(k).unwrap_or(d)
}

pub struct SmolLm3Cfg {
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub nope_step: usize,
}

impl SmolLm3Cfg {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g
            .architecture()
            .ok_or_else(|| StrixError::invalid("gguf: no general.architecture"))?;
        if arch != "smollm3" {
            return Err(StrixError::unsupported(format!(
                "smollm3 loader got `{arch}`"
            )));
        }
        let k = |s: &str| format!("smollm3.{s}");
        let hidden = meta_u32(g, &k("embedding_length"))?;
        let n_heads = meta_u32(g, &k("attention.head_count"))?;
        let n_kv = meta_u32(g, &k("attention.head_count_kv"))?;
        let ffn = meta_u32(g, &k("feed_forward_length"))?;
        let n_layers = meta_u32(g, &k("block_count"))?;
        let eps = meta_f32_or(g, &k("attention.layer_norm_rms_epsilon"), 1e-6);
        let rope_base = meta_f32_or(g, &k("rope.freq_base"), 5_000_000.0);
        let head_dim = meta_u32(g, &k("rope.dimension_count")).unwrap_or(hidden / n_heads);
        let vocab = g
            .tensors()
            .get("token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .map(|v| v as usize)
            .filter(|&v| v > 0)
            .ok_or_else(|| StrixError::invalid("smollm3: cannot determine vocab"))?;
        Ok(SmolLm3Cfg {
            hidden,
            n_heads,
            n_kv,
            head_dim,
            ffn,
            n_layers,
            vocab,
            eps,
            rope_base,
            nope_step: 4,
        })
    }
    pub fn report(&self) -> String {
        format!(
            "smollm3: {}L hidden={} heads={}/{} hd={} ffn={} vocab={} rope={:.0e} NoPE@every-{}",
            self.n_layers,
            self.hidden,
            self.n_heads,
            self.n_kv,
            self.head_dim,
            self.ffn,
            self.vocab,
            self.rope_base,
            self.nope_step
        )
    }
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0.0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let i = c * 8;
        for k in 0..8 {
            acc[k] += a[i + k] * b[i + k];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in (chunks * 8)..n {
        s += a[i] * b[i];
    }
    s
}

fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}

/// NEOX RoPE on a head vector (plain: freq_scale=1, no yarn).
fn rope_neox(vec: &mut [f32], pos: usize, n_dims: usize, freq_base: f32) {
    let half = n_dims / 2;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let mut theta = pos as f32;
    for k in 0..half {
        let (s, c) = theta.sin_cos();
        let x0 = vec[k];
        let x1 = vec[k + half];
        vec[k] = x0 * c - x1 * s;
        vec[k + half] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

/// out[o] = dequant(W row o) · x, parallel over rows. in_dim = row length.
fn qmatmul(out: &mut [f32], x: &[f32], bytes: &[u8], ty: GgmlType, in_dim: usize) {
    let bpr = (in_dim / ty.block_elems()) * ty.block_bytes();
    out.par_iter_mut().enumerate().for_each_init(
        || vec![0.0f32; in_dim],
        |scratch, (o, oref)| {
            dequantize_into(ty, &bytes[o * bpr..o * bpr + bpr], scratch).unwrap();
            *oref = dot_f32(scratch, x);
        },
    );
}

struct LayerNorms {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
}

pub struct SmolLm3Model {
    gguf: GgufFile,
    cfg: SmolLm3Cfg,
    norms: Vec<LayerNorms>,
    output_norm: Vec<f32>,
    // KV cache, per layer: [n_kv * max_seq * head_dim]
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    pos: usize,
    max_seq: usize,
    accel: Option<Box<dyn WeightAccel>>,
    /// True when the accelerator runs the whole forward on-device (Stage C).
    gpu_decode: bool,
    #[cfg(feature = "npu")]
    npu: Option<crate::mellum_npu::SmolLm3Npu>,
}

impl SmolLm3Model {
    pub fn from_gguf(gguf: GgufFile, max_seq: usize) -> Result<Self> {
        let cfg = SmolLm3Cfg::from_gguf(&gguf)?;
        let f32v = |name: &str| -> Result<Vec<f32>> {
            gguf.dequant_tensor(name)
                .map_err(|e| StrixError::invalid(format!("smollm3: {name}: {e}")))
        };
        let mut norms = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            norms.push(LayerNorms {
                attn_norm: f32v(&format!("blk.{l}.attn_norm.weight"))?,
                ffn_norm: f32v(&format!("blk.{l}.ffn_norm.weight"))?,
            });
        }
        let output_norm = f32v("output_norm.weight")?;
        let kvd = cfg.n_kv * cfg.head_dim;
        let kc = (0..cfg.n_layers)
            .map(|_| Vec::with_capacity(kvd * max_seq))
            .collect();
        let vc = (0..cfg.n_layers)
            .map(|_| Vec::with_capacity(kvd * max_seq))
            .collect();
        Ok(SmolLm3Model {
            gguf,
            cfg,
            norms,
            output_norm,
            kc,
            vc,
            pos: 0,
            max_seq,
            accel: None,
            gpu_decode: false,
            #[cfg(feature = "npu")]
            npu: None,
        })
    }

    /// Stage the per-layer projection weights onto the NPU (int8, requantized from
    /// Q4_0). Prefill GEMMs then run on the XDNA2 NPU (~2 W) via fixed M=256 xclbins.
    #[cfg(feature = "npu")]
    pub fn attach_npu(&mut self, mut npu: crate::mellum_npu::SmolLm3Npu) -> Result<usize> {
        let mut n = 0;
        let q_dim = self.cfg.n_heads * self.cfg.head_dim;
        let kv_dim = self.cfg.n_kv * self.cfg.head_dim;
        for li in 0..self.cfg.n_layers {
            let b = |s: &str| format!("blk.{li}.{s}");
            let l = li as u64;
            let stage = |sh: &mut crate::mellum_npu::NpuShape, slot: u64, name: &str| -> Result<()> {
                let (bytes, ty, _, _) = self.w(name)?;
                sh.stage_q8(slot, bytes, ty)
            };
            stage(&mut npu.qo, 2 * l + 1, &b("attn_output.weight"))?;
            stage(&mut npu.down, l, &b("ffn_down.weight"))?;
            n += 2;
            if npu.qkv.is_some() {
                let (qb, ty, _, _) = self.w(&b("attn_q.weight"))?;
                let (kb, _, _, _) = self.w(&b("attn_k.weight"))?;
                let (vb, _, _, _) = self.w(&b("attn_v.weight"))?;
                npu.qkv
                    .as_mut()
                    .unwrap()
                    .stage_q8_triple(l, qb, kb, vb, q_dim, kv_dim, ty)?;
                n += 1;
            } else {
                stage(&mut npu.qo, 2 * l, &b("attn_q.weight"))?;
                stage(&mut npu.kv, 2 * l, &b("attn_k.weight"))?;
                stage(&mut npu.kv, 2 * l + 1, &b("attn_v.weight"))?;
                n += 3;
            }
            if npu.gu2.is_some() {
                let (gb, ty, _, _) = self.w(&b("ffn_gate.weight"))?;
                let (ub, _, _, _) = self.w(&b("ffn_up.weight"))?;
                npu.gu2.as_mut().unwrap().stage_q8_pair(l, gb, ub, ty)?;
                n += 1;
            } else {
                stage(&mut npu.gu, 2 * l, &b("ffn_gate.weight"))?;
                stage(&mut npu.gu, 2 * l + 1, &b("ffn_up.weight"))?;
                n += 2;
            }
        }
        self.npu = Some(npu);
        Ok(n)
    }

    /// Fused q‖k‖v on the NPU (one dispatch, split output), chunked over M_NPU.
    #[cfg(feature = "npu")]
    fn npu_qkv_fused(
        &self,
        il: usize,
        nrm: &[f32],
        m: usize,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Option<Result<()>> {
        let qkv = self.npu.as_ref()?.qkv.as_ref()?;
        if !qkv.has(il as u64) {
            return None;
        }
        let (hidden, q_dim, kv_dim) = (
            self.cfg.hidden,
            self.cfg.n_heads * self.cfg.head_dim,
            self.cfg.n_kv * self.cfg.head_dim,
        );
        let mp = crate::mellum_npu::M_NPU;
        for c in (0..m).step_by(mp) {
            let mc = (m - c).min(mp);
            if let Err(e) = qkv.gemm_split3(
                il as u64,
                &nrm[c * hidden..(c + mc) * hidden],
                mc,
                q_dim,
                kv_dim,
                &mut q[c * q_dim..(c + mc) * q_dim],
                &mut k[c * kv_dim..(c + mc) * kv_dim],
                &mut v[c * kv_dim..(c + mc) * kv_dim],
            ) {
                return Some(Err(e));
            }
        }
        Some(Ok(()))
    }

    /// Fused gate‖up on the NPU (one dispatch, split output), chunked over M_NPU.
    #[cfg(feature = "npu")]
    fn npu_gu_fused(
        &self,
        il: usize,
        nrm: &[f32],
        m: usize,
        gate: &mut [f32],
        up: &mut [f32],
    ) -> Option<Result<()>> {
        let gu2 = self.npu.as_ref()?.gu2.as_ref()?;
        if !gu2.has(il as u64) {
            return None;
        }
        let (hidden, ffn) = (self.cfg.hidden, self.cfg.ffn);
        let mp = crate::mellum_npu::M_NPU;
        for c in (0..m).step_by(mp) {
            let mc = (m - c).min(mp);
            if let Err(e) = gu2.gemm_split2(
                il as u64,
                &nrm[c * hidden..(c + mc) * hidden],
                mc,
                &mut gate[c * ffn..(c + mc) * ffn],
                &mut up[c * ffn..(c + mc) * ffn],
            ) {
                return Some(Err(e));
            }
        }
        Some(Ok(()))
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    fn w<'a>(&'a self, name: &str) -> Result<(&'a [u8], GgmlType, usize, usize)> {
        let t = self
            .gguf
            .tensors()
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("smollm3: missing tensor {name}")))?;
        let bytes = self.gguf.tensor_bytes(name)?;
        let in_dim = t.dims[0] as usize;
        let out_dim = t.dims.get(1).copied().unwrap_or(1) as usize;
        Ok((bytes, t.ggml_type, in_dim, out_dim))
    }

    /// One-token forward. Returns logits.
    /// Upload big projection weights to the GPU accelerator; matmuls then run via
    /// `gemv`. Norms/rope/attention stay CPU. Returns weights staged.
    pub fn attach_accel(&mut self, mut accel: Box<dyn WeightAccel>) -> usize {
        let mut names: Vec<String> = Vec::new();
        for l in 0..self.cfg.n_layers {
            for t in [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ] {
                names.push(format!("blk.{l}.{t}.weight"));
            }
        }
        names.push("token_embd.weight".to_string());
        let mut n = 0;
        for name in &names {
            let Ok((bytes, ty, in_dim, out_dim)) = self.w(name) else {
                continue;
            };
            let ok = match ty {
                GgmlType::Q4_0 => accel.upload_q4_0(name, bytes, in_dim, out_dim),
                GgmlType::Q4_1 => accel.upload_q4_1(name, bytes, in_dim, out_dim),
                GgmlType::Q6K => accel.upload_q6_k(name, bytes, in_dim, out_dim),
                GgmlType::Q8_0 => accel.upload_q8_0(name, bytes, in_dim, out_dim),
                _ => false,
            };
            if ok {
                n += 1;
            }
        }
        // Stage C: upload f32 norms (no QK-norm in smollm3) + describe the arch.
        // NoPE: every `nope_step`-th layer ((il+1)%step==0) skips RoPE entirely.
        for (l, nrm) in self.norms.iter().enumerate() {
            accel.upload_f32(&format!("blk.{l}.attn_norm.weight"), &nrm.attn_norm);
            accel.upload_f32(&format!("blk.{l}.ffn_norm.weight"), &nrm.ffn_norm);
        }
        accel.upload_f32("output_norm.weight", &self.output_norm);
        let step = self.cfg.nope_step;
        let layers = (0..self.cfg.n_layers)
            .map(|l| GpuLayerCfg {
                head_dim: self.cfg.head_dim,
                n_kv: self.cfg.n_kv,
                k_eq_v: false,
                rope_theta: self.cfg.rope_base,
                is_local: false,
                output_scale: 1.0,
                no_rope: (l + 1) % step == 0,
            })
            .collect();
        let gpu_cfg = GpuDecodeConfig {
            hidden: self.cfg.hidden,
            n_heads: self.cfg.n_heads,
            ffn: self.cfg.ffn,
            vocab: self.cfg.vocab,
            n_layers: self.cfg.n_layers,
            eps: self.cfg.eps,
            final_softcap: 0.0,
            attn_rsqrt: true,
            norm_v: false,
            qk_norm: false,
            post_norm: false,
            act_gelu: false,
            gpu_prefill: false,
            n_swa: 0,
            max_seq: self.max_seq,
            layers,
        };
        self.gpu_decode =
            std::env::var("STRIX_GPU_HYBRID").is_err() && accel.configure_decode(gpu_cfg);
        self.accel = Some(accel);
        n
    }

    /// Embedding row for `token` (no scaling for smollm3).
    fn embed(&self, token: u32) -> Result<Vec<f32>> {
        let (eb, ety, ein, _) = self.w("token_embd.weight")?;
        let bpr = (ein / ety.block_elems()) * ety.block_bytes();
        let mut h = vec![0.0f32; self.cfg.hidden];
        dequantize_into(ety, &eb[token as usize * bpr..token as usize * bpr + bpr], &mut h)
            .map_err(|e| StrixError::invalid(format!("smollm3 embd: {e}")))?;
        Ok(h)
    }

    /// On-device decode of one token: embed on CPU, run the whole forward on the
    /// accelerator (~1 submit). Appends to the device KV cache at `self.pos`.
    fn gpu_decode_step(&mut self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= self.cfg.vocab {
            return Err(StrixError::invalid("smollm3 gpu decode: token out of range"));
        }
        let h = self.embed(token)?;
        let pos = self.pos;
        let logits = self
            .accel
            .as_mut()
            .and_then(|a| a.decode_step(&h, pos))
            .ok_or_else(|| StrixError::invalid("smollm3 gpu decode_step failed"))?;
        self.pos += 1;
        Ok(logits)
    }

    /// Upload the CPU/NPU-prefilled KV (self.kc/vc, `self.pos` tokens) into the
    /// device decode KV cache so on-device decode can attend the prompt — keeps
    /// prefill off the iGPU. Layouts match (roped K incl. NoPE layers, raw V).
    fn seed_device_kv(&mut self) -> Result<()> {
        let Some(mut accel) = self.accel.take() else {
            return Ok(());
        };
        let mut ok = true;
        for il in 0..self.cfg.n_layers {
            if !accel.seed_decode_kv(il, &self.kc[il], &self.vc[il]) {
                ok = false;
                break;
            }
        }
        self.accel = Some(accel);
        if ok {
            Ok(())
        } else {
            Err(StrixError::invalid("smollm3: device KV seed failed"))
        }
    }

    /// Matmul by tensor name: GPU `gemv` if resident, else CPU dequant.
    fn mm(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        if let Some(a) = &self.accel {
            if let Some(y) = a.gemv(name, x) {
                return Ok(y);
            }
        }
        let (bytes, ty, in_dim, out_dim) = self.w(name)?;
        let mut y = vec![0.0f32; out_dim];
        qmatmul(&mut y, x, bytes, ty, in_dim);
        Ok(y)
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (hidden, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv);
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let groups = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let pos = self.pos;

        // embedding: row `token` of token_embd (in_dim = hidden)
        let (eb, ety, ein, _) = self.w("token_embd.weight")?;
        let bpr = (ein / ety.block_elems()) * ety.block_bytes();
        let mut h = vec![0.0f32; hidden];
        dequantize_into(
            ety,
            &eb[token as usize * bpr..token as usize * bpr + bpr],
            &mut h,
        )
        .map_err(|e| StrixError::invalid(format!("smollm3 embd: {e}")))?;

        let mut n = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; q_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn = vec![0.0f32; q_dim];
        let mut o = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; cfg.ffn];
        let mut up = vec![0.0f32; cfg.ffn];

        for il in 0..cfg.n_layers {
            let b = |s: &str| format!("blk.{il}.{s}");
            // attn norm
            rmsnorm(&mut n, &h, &self.norms[il].attn_norm, cfg.eps);
            // q/k/v
            q = self.mm(&b("attn_q.weight"), &n)?;
            k = self.mm(&b("attn_k.weight"), &n)?;
            v = self.mm(&b("attn_v.weight"), &n)?;

            // NoPE: skip rope on layers where (il+1) % step == 0
            let use_rope = (il + 1) % cfg.nope_step != 0;
            if use_rope {
                for hh in 0..nh {
                    rope_neox(&mut q[hh * hd..hh * hd + hd], pos, hd, cfg.rope_base);
                }
                for kh in 0..nkv {
                    rope_neox(&mut k[kh * hd..kh * hd + hd], pos, hd, cfg.rope_base);
                }
            }

            // append to KV cache
            self.kc[il].extend_from_slice(&k);
            self.vc[il].extend_from_slice(&v);
            let len = pos + 1;
            let kc = &self.kc[il];
            let vc = &self.vc[il];

            // GQA attention per q-head
            attn.par_chunks_mut(hd).enumerate().for_each(|(hh, oh)| {
                let kvh = hh / groups;
                let qh = &q[hh * hd..hh * hd + hd];
                let mut scores = vec![0.0f32; len];
                for t in 0..len {
                    let kk = &kc[(t * nkv + kvh) * hd..(t * nkv + kvh) * hd + hd];
                    scores[t] = dot_f32(qh, kk) * scale;
                }
                let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for t in 0..len {
                        acc += scores[t] * vc[(t * nkv + kvh) * hd + d];
                    }
                    oh[d] = acc * inv;
                }
            });

            // output proj + residual
            o = self.mm(&b("attn_output.weight"), &attn)?;
            for i in 0..hidden {
                h[i] += o[i];
            }

            // ffn norm + SwiGLU
            rmsnorm(&mut n, &h, &self.norms[il].ffn_norm, cfg.eps);
            gate = self.mm(&b("ffn_gate.weight"), &n)?;
            up = self.mm(&b("ffn_up.weight"), &n)?;
            for i in 0..cfg.ffn {
                let g = gate[i];
                gate[i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            o = self.mm(&b("ffn_down.weight"), &gate)?;
            for i in 0..hidden {
                h[i] += o[i];
            }
        }

        // final norm + lm_head (tied to token_embd)
        rmsnorm(&mut n, &h, &self.output_norm, cfg.eps);
        let head_name = if self.gguf.tensors().contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let logits = self.mm(head_name, &n)?;
        self.pos += 1;
        Ok(logits)
    }

    /// Batched matmul out[m*n] = W·xs[m*k] by tensor name. Routes to the NPU shape
    /// (chunked to M=256) when staged, else CPU dequant (weight read once per chunk).
    #[allow(clippy::too_many_arguments)]
    fn bmm(
        &self,
        name: &str,
        which: NpuW,
        xs: &[f32],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> Result<()> {
        #[cfg(feature = "npu")]
        if let Some(npu) = &self.npu {
            let (sh, slot) = match which {
                NpuW::Q => (&npu.qo, 0),
                NpuW::O => (&npu.qo, 1),
                NpuW::K => (&npu.kv, 0),
                NpuW::V => (&npu.kv, 1),
                NpuW::Gate => (&npu.gu, 0),
                NpuW::Up => (&npu.gu, 1),
                NpuW::Down => (&npu.down, 2),
            };
            // slot encodes (layer, which): caller passes layer via `name`'s blk index.
            let il: u64 = name
                .split('.')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let s = match which {
                NpuW::Down => il,
                _ => 2 * il + slot,
            };
            if sh.k == k && sh.n == n && sh.has(s) {
                let mut okall = true;
                for c in (0..m).step_by(crate::mellum_npu::M_NPU) {
                    let mc = (m - c).min(crate::mellum_npu::M_NPU);
                    if sh
                        .gemm(
                            s,
                            &xs[c * k..(c + mc) * k],
                            mc,
                            &mut out[c * n..(c + mc) * n],
                        )
                        .is_err()
                    {
                        okall = false;
                        break;
                    }
                }
                if okall {
                    return Ok(());
                }
            }
        }
        let _ = which;
        // CPU fallback: dequant each weight row once, dot against all m activations.
        let (bytes, ty, _, _) = self.w(name)?;
        let bpr = (k / ty.block_elems()) * ty.block_bytes();
        let mut rt = vec![0.0f32; n * m];
        rt.par_chunks_mut(m).enumerate().for_each_init(
            || vec![0.0f32; k],
            |scratch, (o, orow)| {
                dequantize_into(ty, &bytes[o * bpr..o * bpr + bpr], scratch).unwrap();
                for t in 0..m {
                    orow[t] = dot_f32(scratch, &xs[t * k..(t + 1) * k]);
                }
            },
        );
        for t in 0..m {
            for o in 0..n {
                out[t * n + o] = rt[o * m + t];
            }
        }
        Ok(())
    }

    /// Batched prefill over all `m` prompt tokens: the 7 projection GEMMs/layer run
    /// on the NPU (low power); norms/QK/RoPE/attention stay on the CPU. Populates the
    /// KV cache and returns the last token's logits. Bit-compatible with `forward`'s
    /// math (same dot order via dot_f32; attention identical).
    fn prefill_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (hidden, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv);
        let (q_dim, kv_dim, groups, ffn) = (nh * hd, nkv * hd, nh / nkv, cfg.ffn);
        let scale = 1.0 / (hd as f32).sqrt();
        let m = tokens.len();

        // embeddings
        let (eb, ety, ein, _) = self.w("token_embd.weight")?;
        let bpr = (ein / ety.block_elems()) * ety.block_bytes();
        let mut h = vec![0.0f32; m * hidden];
        for (t, &tok) in tokens.iter().enumerate() {
            dequantize_into(
                ety,
                &eb[tok as usize * bpr..tok as usize * bpr + bpr],
                &mut h[t * hidden..(t + 1) * hidden],
            )
            .map_err(|e| StrixError::invalid(format!("smollm3 embd: {e}")))?;
        }

        let mut nrm = vec![0.0f32; m * hidden];
        for il in 0..cfg.n_layers {
            let bnm = |s: &str| format!("blk.{il}.{s}");
            for t in 0..m {
                rmsnorm(
                    &mut nrm[t * hidden..(t + 1) * hidden],
                    &h[t * hidden..(t + 1) * hidden],
                    &self.norms[il].attn_norm,
                    cfg.eps,
                );
            }
            let mut q = vec![0.0f32; m * q_dim];
            let mut k = vec![0.0f32; m * kv_dim];
            let mut v = vec![0.0f32; m * kv_dim];
            let mut qkv_done = false;
            #[cfg(feature = "npu")]
            {
                if let Some(r) = self.npu_qkv_fused(il, &nrm, m, &mut q, &mut k, &mut v) {
                    r?;
                    qkv_done = true;
                }
            }
            if !qkv_done {
                self.bmm(&bnm("attn_q.weight"), NpuW::Q, &nrm, m, hidden, q_dim, &mut q)?;
                self.bmm(&bnm("attn_k.weight"), NpuW::K, &nrm, m, hidden, kv_dim, &mut k)?;
                self.bmm(&bnm("attn_v.weight"), NpuW::V, &nrm, m, hidden, kv_dim, &mut v)?;
            }
            let use_rope = (il + 1) % cfg.nope_step != 0;
            for t in 0..m {
                if use_rope {
                    for hh in 0..nh {
                        rope_neox(
                            &mut q[t * q_dim + hh * hd..t * q_dim + hh * hd + hd],
                            t,
                            hd,
                            cfg.rope_base,
                        );
                    }
                    for kh in 0..nkv {
                        rope_neox(
                            &mut k[t * kv_dim + kh * hd..t * kv_dim + kh * hd + hd],
                            t,
                            hd,
                            cfg.rope_base,
                        );
                    }
                }
            }
            self.kc[il].extend_from_slice(&k);
            self.vc[il].extend_from_slice(&v);
            let kc = &self.kc[il];
            let vc = &self.vc[il];
            let mut attn = vec![0.0f32; m * q_dim];
            attn.par_chunks_mut(q_dim)
                .enumerate()
                .for_each(|(t, arow)| {
                    let len = t + 1; // causal: token t attends keys 0..=t
                    for hh in 0..nh {
                        let kvh = hh / groups;
                        let qh = &q[t * q_dim + hh * hd..t * q_dim + hh * hd + hd];
                        let mut sc = vec![0.0f32; len];
                        for j in 0..len {
                            let kk = &kc[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                            sc[j] = dot_f32(qh, kk) * scale;
                        }
                        let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                        let mut sum = 0.0f32;
                        for s in sc.iter_mut() {
                            *s = (*s - mx).exp();
                            sum += *s;
                        }
                        let inv = 1.0 / sum;
                        let oh = &mut arow[hh * hd..hh * hd + hd];
                        for d in 0..hd {
                            let mut acc = 0.0f32;
                            for j in 0..len {
                                acc += sc[j] * vc[(j * nkv + kvh) * hd + d];
                            }
                            oh[d] = acc * inv;
                        }
                    }
                });
            let mut o = vec![0.0f32; m * hidden];
            self.bmm(
                &bnm("attn_output.weight"),
                NpuW::O,
                &attn,
                m,
                q_dim,
                hidden,
                &mut o,
            )?;
            for i in 0..m * hidden {
                h[i] += o[i];
            }
            for t in 0..m {
                rmsnorm(
                    &mut nrm[t * hidden..(t + 1) * hidden],
                    &h[t * hidden..(t + 1) * hidden],
                    &self.norms[il].ffn_norm,
                    cfg.eps,
                );
            }
            let mut gate = vec![0.0f32; m * ffn];
            let mut up = vec![0.0f32; m * ffn];
            let mut gu_done = false;
            #[cfg(feature = "npu")]
            {
                if let Some(r) = self.npu_gu_fused(il, &nrm, m, &mut gate, &mut up) {
                    r?;
                    gu_done = true;
                }
            }
            if !gu_done {
                self.bmm(&bnm("ffn_gate.weight"), NpuW::Gate, &nrm, m, hidden, ffn, &mut gate)?;
                self.bmm(&bnm("ffn_up.weight"), NpuW::Up, &nrm, m, hidden, ffn, &mut up)?;
            }
            for i in 0..m * ffn {
                let g = gate[i];
                gate[i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            self.bmm(
                &bnm("ffn_down.weight"),
                NpuW::Down,
                &gate,
                m,
                ffn,
                hidden,
                &mut o,
            )?;
            for i in 0..m * hidden {
                h[i] += o[i];
            }
        }
        self.pos = m;
        // last token → logits
        let last = &h[(m - 1) * hidden..m * hidden];
        let mut nf = vec![0.0f32; hidden];
        rmsnorm(&mut nf, last, &self.output_norm, cfg.eps);
        let head_name = if self.gguf.tensors().contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        self.mm(head_name, &nf)
    }
}

/// Selector for which projection a batched matmul targets (picks NPU shape+slot).
#[derive(Clone, Copy)]
enum NpuW {
    Q,
    O,
    K,
    V,
    Gate,
    Up,
    Down,
}

impl Decoder for SmolLm3Model {
    fn prefill(&mut self, input_tokens: &[u32]) -> Result<Logits> {
        if input_tokens.is_empty() {
            return Err(StrixError::invalid("smollm3: empty prompt"));
        }
        // Stage C: prefill stays OFF the iGPU (sustained GPU load crashes this box —
        // see never-gpu-prefill). Batched CPU/NPU prefill fills self.kc/vc
        // (prefill_batch uses the model's NPU when attached, else CPU), then seed the
        // device KV cache so on-device decode can attend the prompt. Must take
        // precedence over the NPU-only branch so the seed actually runs.
        if self.gpu_decode {
            let last = self.prefill_batch(input_tokens)?;
            self.seed_device_kv()?;
            return Ok(Logits::new(last));
        }
        // Batched prefill (NPU-routed when attached); else token-by-token.
        #[cfg(feature = "npu")]
        if self.npu.is_some() {
            return Ok(Logits::new(self.prefill_batch(input_tokens)?));
        }
        let mut last = Vec::new();
        for &t in input_tokens {
            last = self.forward(t)?;
        }
        Ok(Logits::new(last))
    }
    fn decode_one(&mut self, token: u32) -> Result<Logits> {
        if self.gpu_decode {
            return Ok(Logits::new(self.gpu_decode_step(token)?));
        }
        Ok(Logits::new(self.forward(token)?))
    }
    fn reset(&mut self) {
        self.pos = 0;
        for c in self.kc.iter_mut() {
            c.clear();
        }
        for c in self.vc.iter_mut() {
            c.clear();
        }
    }
}
