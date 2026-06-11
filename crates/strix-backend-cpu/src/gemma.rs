//! CPU reference forward pass for Gemma-4 (GGUF, quantized weights kept in place).
//!
//! Runs the 12B QAT q4_0 end-to-end with coherent output, verified op-by-op
//! against llama.cpp's `gemma4.cpp` graph. Also handles Gemma-3 (incidental,
//! shares this path). Gemma-4 specifics:
//! - embedding scaled by `sqrt(hidden)`;
//! - sandwich RMSNorm (4 per block) + per-head QK-norm before RoPE;
//! - **per-layer** head_dim (global 512 / local 256) and KV-head count;
//! - `attention_k_eq_v`: global layers have no V projection — V is the *raw*
//!   k_proj, RMS-normalized with NO weight (whereas K gets the k_norm weight);
//! - V is RMS-normalized (no weight) on every layer before caching;
//! - attention scaling 1.0 (QK-norm replaces 1/sqrt(d); Gemma-3 uses 1/sqrt(d));
//! - dual RoPE base (global 1e6 / local 1e4) + proportional-RoPE `rope_freqs`
//!   (freq factors) on global layers;
//! - GeGLU MLP, per-layer `layer_output_scale`, final logit soft-capping.
//!
//! Weights stay quantized in the mmap'd GGUF; each matmul dequantizes one weight
//! row at a time into a reusable buffer (rayon-parallel over output rows), so the
//! 12B q4_0 model runs in ~7 GB instead of ~48 GB of f32.
//!
//! Note: Gemma-4's GGUF stores norm weights with the `+1` already baked in, so
//! plain RMSNorm (`x * weight`) is correct. Sliding-window masking is a no-op for
//! `seq_len <= sliding_window` (1024), so it is deferred (correct for short ctx).

use rayon::prelude::*;
use strix_core::accel::{GpuDecodeConfig, GpuLayerCfg, WeightAccel};
use strix_core::backend::{Decoder, Model};
use strix_core::error::{Result, StrixError};
use strix_core::model_config::{ModelArchitecture, ModelConfig};
use strix_core::sampler::Logits;
use strix_models::ggml_quant::dequantize_into;
use strix_models::gguf::{GgufFile, MetaValue};
use strix_models::GgmlType;

use crate::attention::sdpa_single;
use crate::ops::{geglu, rope_in_place_ff, softcap_inplace};

/// Per-layer geometry (Gemma-4 varies these by layer).
#[derive(Debug, Clone)]
struct LayerCfg {
    head_dim: usize,
    n_kv: usize,
    /// True for sliding-window (local) layers. Reserved for windowed masking
    /// once sequences exceed the sliding window (currently a no-op below it).
    #[allow(dead_code)]
    is_local: bool,
    rope_theta: f32,
    q_dim: usize,
    /// True when this layer has no V projection and reuses K as V
    /// (Gemma-4 `attention_k_eq_v` on global layers).
    k_eq_v: bool,
}

/// Parsed Gemma-4 hyperparameters.
#[derive(Debug, Clone)]
struct GemmaCfg {
    hidden: usize,
    n_heads: usize,
    ffn: usize,
    vocab: usize,
    n_layers: usize,
    eps: f32,
    emb_scale: f32,
    final_softcap: f32,
    /// Use 1/sqrt(head_dim) attention scaling (gemma3) vs 1.0 (gemma4).
    attn_rsqrt: bool,
    /// Gemma-4 RMS-normalizes V (per head, no weight) before caching.
    norm_v: bool,
    /// Sliding-window size for local layers (0 = no windowing). Local-layer
    /// queries attend only the last `n_swa` keys.
    n_swa: usize,
    layers: Vec<LayerCfg>,
}

fn meta_u32(g: &GgufFile, key: &str) -> Result<u32> {
    g.meta(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| StrixError::invalid(format!("gemma: missing metadata `{key}`")))
}

fn meta_f32_or(g: &GgufFile, key: &str, default: f32) -> f32 {
    g.meta(key).and_then(|v| v.as_f32()).unwrap_or(default)
}

fn int_array(g: &GgufFile, key: &str) -> Result<Vec<i64>> {
    g.meta(key)
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|v| v.as_u64().map(|x| x as i64).unwrap_or(0))
                .collect()
        })
        .ok_or_else(|| StrixError::invalid(format!("gemma: missing int array `{key}`")))
}

fn bool_array(g: &GgufFile, key: &str) -> Result<Vec<bool>> {
    g.meta(key)
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|v| matches!(v, MetaValue::Bool(true)))
                .collect()
        })
        .ok_or_else(|| StrixError::invalid(format!("gemma: missing bool array `{key}`")))
}

impl GemmaCfg {
    fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g
            .architecture()
            .ok_or_else(|| StrixError::invalid("gguf: no general.architecture"))?
            .to_string();
        if arch != "gemma4" && arch != "gemma3" {
            return Err(StrixError::unsupported(format!(
                "this loader handles gemma3/gemma4, got `{arch}`"
            )));
        }
        // gemma3 scales attention by 1/sqrt(head_dim); gemma4 uses 1.0 (QK-norm).
        let attn_rsqrt = arch == "gemma3";
        let k = |s: &str| format!("{arch}.{s}");

        let hidden = meta_u32(g, &k("embedding_length"))? as usize;
        let n_heads = meta_u32(g, &k("attention.head_count"))? as usize;
        let ffn = meta_u32(g, &k("feed_forward_length"))? as usize;
        let n_layers = meta_u32(g, &k("block_count"))? as usize;
        let eps = meta_f32_or(g, &k("attention.layer_norm_rms_epsilon"), 1e-6);

        let key_len = meta_u32(g, &k("attention.key_length"))? as usize;
        // Sliding-window layers may use a smaller head_dim (gemma4); fall back to
        // the global head_dim when there is no `_swa` variant (gemma3).
        let key_len_swa =
            meta_u32(g, &k("attention.key_length_swa")).unwrap_or(key_len as u32) as usize;
        let theta_global = meta_f32_or(g, &k("rope.freq_base"), 1_000_000.0);
        let theta_local = meta_f32_or(g, &k("rope.freq_base_swa"), 10_000.0);
        let final_softcap = meta_f32_or(g, &k("final_logit_softcapping"), 0.0);
        // Sliding-window size for local layers (gemma4=1024). 0 ⇒ no windowing.
        // STRIX_NSWA overrides it (for testing windowing on short prompts).
        let n_swa = std::env::var("STRIX_NSWA")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| meta_u32(g, &k("attention.sliding_window")).unwrap_or(0) as usize);

        // head_count_kv is a per-layer array (gemma4) or a scalar (gemma3).
        let kv_counts: Vec<usize> = match int_array(g, &k("attention.head_count_kv")) {
            Ok(a) => a.iter().map(|&x| x as usize).collect(),
            Err(_) => vec![meta_u32(g, &k("attention.head_count_kv"))? as usize; n_layers],
        };
        // sliding_window_pattern is a per-layer bool array (gemma4) or absent
        // (gemma3: every 6th layer is global, the rest sliding/local).
        let pattern: Vec<bool> = match bool_array(g, &k("attention.sliding_window_pattern")) {
            Ok(a) => a,
            Err(_) => (0..n_layers).map(|l| (l + 1) % 6 != 0).collect(),
        };
        if kv_counts.len() != n_layers || pattern.len() != n_layers {
            return Err(StrixError::invalid("gemma: per-layer arrays len mismatch"));
        }

        let vocab = g
            .tensors()
            .get("token_embd.weight")
            .map(|t| t.dims.get(1).copied().unwrap_or(0) as usize)
            .filter(|&v| v > 0)
            .ok_or_else(|| StrixError::invalid("gemma: cannot determine vocab from token_embd"))?;

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let is_local = pattern[l];
            let head_dim = if is_local { key_len_swa } else { key_len };
            let n_kv = kv_counts[l];
            // A missing V projection means this layer reuses K as V (k_eq_v).
            let k_eq_v = !g.tensors().contains_key(&format!("blk.{l}.attn_v.weight"));
            layers.push(LayerCfg {
                head_dim,
                n_kv,
                is_local,
                rope_theta: if is_local { theta_local } else { theta_global },
                q_dim: n_heads * head_dim,
                k_eq_v,
            });
        }

        Ok(GemmaCfg {
            hidden,
            n_heads,
            ffn,
            vocab,
            n_layers,
            eps,
            emb_scale: (hidden as f32).sqrt(),
            final_softcap,
            attn_rsqrt,
            n_swa,
            norm_v: arch == "gemma4",
            layers,
        })
    }
}

/// Pre-loaded small (F32) per-layer tensors; big weights stay in the GGUF.
struct LayerNorms {
    attn_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffw_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    output_scale: f32,
}

/// Per-layer KV cache with layer-specific dims.
struct LayerKv {
    head_dim: usize,
    max_seq: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl LayerKv {
    fn new(n_kv: usize, head_dim: usize, max_seq: usize) -> Self {
        LayerKv {
            head_dim,
            max_seq,
            k: vec![0.0; n_kv * max_seq * head_dim],
            v: vec![0.0; n_kv * max_seq * head_dim],
        }
    }
    #[inline]
    fn row(&self, head: usize, pos: usize) -> usize {
        (head * self.max_seq + pos) * self.head_dim
    }
    fn store(&mut self, head: usize, pos: usize, kk: &[f32], vv: &[f32]) {
        let s = self.row(head, pos);
        self.k[s..s + self.head_dim].copy_from_slice(kk);
        self.v[s..s + self.head_dim].copy_from_slice(vv);
    }
    fn keys(&self, head: usize, len: usize) -> &[f32] {
        let s = self.row(head, 0);
        &self.k[s..s + len * self.head_dim]
    }
    fn values(&self, head: usize, len: usize) -> &[f32] {
        let s = self.row(head, 0);
        &self.v[s..s + len * self.head_dim]
    }
}

/// A loaded Gemma-4 model (weights kept quantized in the mmap'd GGUF).
pub struct GemmaModel {
    gguf: GgufFile,
    cfg: GemmaCfg,
    config: ModelConfig,
    norms: Vec<LayerNorms>,
    output_norm: Vec<f32>,
    /// Proportional-RoPE frequency factors for global layers (gemma4), if present.
    rope_freqs: Option<Vec<f32>>,
    cache: Vec<LayerKv>,
    seq: usize,
    /// Optional GEMV accelerator (iGPU). When attached, Q4_0/Q6_K projection
    /// matmuls run on-device; everything else stays on the CPU.
    accel: Option<Box<dyn WeightAccel>>,
    /// True when the accelerator runs the *entire* decode forward on-device
    /// (Stage C); then `prefill`/`decode_one` route through `accel.decode_step`.
    gpu_decode: bool,
}

/// RMSNorm with selectable affine convention: gain is `weight` (plain) or
/// `1 + weight` (Gemma HF delta convention), depending on `plus1`.
fn norm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let sumsq: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (sumsq / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}

fn norm_inplace(v: &mut [f32], w: &[f32], eps: f32) {
    let n = v.len();
    let sumsq: f32 = v.iter().map(|x| x * x).sum();
    let scale = 1.0 / (sumsq / n as f32 + eps).sqrt();
    for i in 0..n {
        v[i] = v[i] * scale * w[i];
    }
}

/// Quantized matmul: `out[o] = dequant(weight_row_o) · x`, rayon-parallel over
/// output rows (each row dequantized into a reusable scratch buffer).
///
/// STRIX_W8A8 simulates the NPU offload path's quantization to de-risk precision:
/// the weight row is dequantized then RE-quantized to per-row int8, the activation
/// to per-token int8, an integer dot is taken, and the result is rescaled by
/// `wscale[o]*xscale`. If gemma-4 stays coherent under this, the NPU int8 GEMM
/// (same math) preserves quality.
fn qlinear(out: &mut [f32], x: &[f32], bytes: &[u8], ty: GgmlType, in_dim: usize) {
    let bpr = (in_dim / ty.block_elems()) * ty.block_bytes();
    if w8a8_enabled() {
        // per-token int8 activation
        let xmax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-12);
        let xs = xmax / 127.0;
        let x8: Vec<i32> = x.iter().map(|&v| (v / xs).round() as i32).collect();
        out.par_iter_mut().enumerate().for_each_init(
            || vec![0.0f32; in_dim],
            |buf, (o, y)| {
                let row = &bytes[o * bpr..o * bpr + bpr];
                dequantize_into(ty, row, buf).expect("dequant weight row");
                let wmax = buf.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-12);
                let ws = wmax / 127.0;
                let mut acc: i64 = 0;
                for i in 0..in_dim {
                    let w8 = (buf[i] / ws).round() as i32; // per-row int8 weight
                    acc += (w8 * x8[i]) as i64;
                }
                *y = acc as f32 * ws * xs;
            },
        );
        return;
    }
    out.par_iter_mut().enumerate().for_each_init(
        || vec![0.0f32; in_dim],
        |buf, (o, y)| {
            let row = &bytes[o * bpr..o * bpr + bpr];
            dequantize_into(ty, row, buf).expect("dequant weight row");
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += buf[i] * x[i];
            }
            *y = acc;
        },
    );
}

fn w8a8_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("STRIX_W8A8").is_ok())
}

impl GemmaModel {
    /// Open a Gemma-4 GGUF and prepare it for inference.
    pub fn from_gguf(gguf: GgufFile, max_seq: usize) -> Result<Self> {
        let cfg = GemmaCfg::from_gguf(&gguf)?;
        let max_seq = max_seq.max(1);

        let f32_tensor = |name: &str| -> Result<Vec<f32>> { gguf.dequant_tensor(name) };
        let scalar = |name: &str| -> f32 {
            gguf.dequant_tensor(name)
                .ok()
                .and_then(|v| v.first().copied())
                .unwrap_or(1.0)
        };

        let mut norms = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = format!("blk.{l}");
            norms.push(LayerNorms {
                attn_norm: f32_tensor(&format!("{p}.attn_norm.weight"))?,
                post_attn_norm: f32_tensor(&format!("{p}.post_attention_norm.weight"))?,
                ffn_norm: f32_tensor(&format!("{p}.ffn_norm.weight"))?,
                post_ffw_norm: f32_tensor(&format!("{p}.post_ffw_norm.weight"))?,
                q_norm: f32_tensor(&format!("{p}.attn_q_norm.weight"))?,
                k_norm: f32_tensor(&format!("{p}.attn_k_norm.weight"))?,
                output_scale: scalar(&format!("{p}.layer_output_scale.weight")),
            });
        }
        let output_norm = f32_tensor("output_norm.weight")?;
        // Optional proportional-RoPE factors (gemma4 global layers).
        let rope_freqs = gguf.dequant_tensor("rope_freqs.weight").ok();

        // Log the per-layer output scales once — if they're ~1.0 ignoring them is safe.
        let (mn, mx) = norms.iter().fold((f32::MAX, f32::MIN), |(a, b), n| {
            (a.min(n.output_scale), b.max(n.output_scale))
        });
        tracing::info!(
            layer_output_scale_min = mn,
            layer_output_scale_max = mx,
            "gemma layer scales"
        );

        let cache = cfg
            .layers
            .iter()
            .map(|lc| LayerKv::new(lc.n_kv, lc.head_dim, max_seq))
            .collect();

        let config = ModelConfig {
            architecture: ModelArchitecture::Unknown,
            vocab_size: cfg.vocab,
            hidden_size: cfg.hidden,
            intermediate_size: cfg.ffn,
            num_hidden_layers: cfg.n_layers,
            num_attention_heads: cfg.n_heads,
            num_key_value_heads: cfg.layers.first().map(|l| l.n_kv).unwrap_or(0),
            head_dim: cfg.layers.first().map(|l| l.head_dim).unwrap_or(0),
            rms_norm_eps: cfg.eps,
            rope_theta: 1_000_000.0,
            max_position_embeddings: max_seq,
        };
        Ok(GemmaModel {
            gguf,
            cfg,
            config,
            norms,
            output_norm,
            rope_freqs,
            cache,
            seq: 0,
            accel: None,
            gpu_decode: false,
        })
    }

    /// Attach a GEMV accelerator (e.g. the iGPU) and upload every Q4_0/Q6_K
    /// projection weight to it. Weights the accelerator declines (unsupported
    /// quant/shape) silently remain on the CPU. Returns the number adopted.
    pub fn attach_accel(&mut self, mut accel: Box<dyn WeightAccel>) -> usize {
        // The big GEMV weights: per-layer projections + the tied lm_head.
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

        for name in &names {
            let Ok((bytes, ty, in_dim, out_dim)) = Self::weight(&self.gguf, name) else {
                continue; // e.g. attn_v absent on k_eq_v global layers
            };
            match ty {
                GgmlType::Q4_0 => {
                    accel.upload_q4_0(name, bytes, in_dim, out_dim);
                }
                GgmlType::Q6K => {
                    // STRIX_Q4_LM_HEAD: requantize the tied lm_head Q6_K → Q4_0 to
                    // cut its per-token weight read ~30% (826→566 MB). The GPU Q6
                    // token_embd is used ONLY for lm_head (embeddings are dequantized
                    // CPU-side from the gguf), so this needs no extra Q6 copy. Done
                    // row-by-row to avoid a full-tensor f32 materialization.
                    if name == "token_embd.weight" && std::env::var("STRIX_Q4_LM_HEAD").is_ok() {
                        let mut q4 = Vec::with_capacity(out_dim * (in_dim / 32) * 18);
                        let mut row = vec![0.0f32; in_dim];
                        for r in 0..out_dim {
                            if dequant_row(bytes, ty, r, in_dim, &mut row).is_ok() {
                                q4.extend_from_slice(&strix_models::quantize_q4_0(&row));
                            }
                        }
                        accel.upload_q4_0(name, &q4, in_dim, out_dim);
                    } else {
                        accel.upload_q6_k(name, bytes, in_dim, out_dim);
                    }
                }
                _ => {}
            }
        }
        // Upload the small f32 tensors (norm weights + rope_freqs) and describe
        // the architecture, so the accelerator can run the whole decode forward.
        for (l, nrm) in self.norms.iter().enumerate() {
            accel.upload_f32(&format!("blk.{l}.attn_norm.weight"), &nrm.attn_norm);
            accel.upload_f32(
                &format!("blk.{l}.post_attention_norm.weight"),
                &nrm.post_attn_norm,
            );
            accel.upload_f32(&format!("blk.{l}.ffn_norm.weight"), &nrm.ffn_norm);
            accel.upload_f32(&format!("blk.{l}.post_ffw_norm.weight"), &nrm.post_ffw_norm);
            accel.upload_f32(&format!("blk.{l}.attn_q_norm.weight"), &nrm.q_norm);
            accel.upload_f32(&format!("blk.{l}.attn_k_norm.weight"), &nrm.k_norm);
        }
        accel.upload_f32("output_norm.weight", &self.output_norm);
        if let Some(rf) = &self.rope_freqs {
            accel.upload_f32("rope_freqs.weight", rf);
        }

        let layers = self
            .cfg
            .layers
            .iter()
            .zip(self.norms.iter())
            .map(|(lc, nrm)| GpuLayerCfg {
                head_dim: lc.head_dim,
                n_kv: lc.n_kv,
                k_eq_v: lc.k_eq_v,
                rope_theta: lc.rope_theta,
                is_local: lc.is_local,
                output_scale: nrm.output_scale,
                no_rope: false,
            })
            .collect();
        let gpu_cfg = GpuDecodeConfig {
            hidden: self.cfg.hidden,
            n_heads: self.cfg.n_heads,
            ffn: self.cfg.ffn,
            vocab: self.cfg.vocab,
            n_layers: self.cfg.n_layers,
            eps: self.cfg.eps,
            final_softcap: self.cfg.final_softcap,
            attn_rsqrt: self.cfg.attn_rsqrt,
            norm_v: self.cfg.norm_v,
            qk_norm: true,
            post_norm: true,
            act_gelu: true,
            gpu_prefill: true,
            n_swa: self.cfg.n_swa,
            max_seq: self.max_seq(),
            layers,
        };
        // The fused full on-device forward (Stage C) is now the faster path
        // (~8.4 vs ~8.2 tok/s) after op fusion (q/k/v & gate/up share a pass;
        // residual+norm fused), so it's the default. `STRIX_GPU_HYBRID=1` forces
        // the per-call hybrid (matmuls on GPU, tiny ops on CPU) for comparison.
        self.gpu_decode =
            std::env::var("STRIX_GPU_HYBRID").is_err() && accel.configure_decode(gpu_cfg);

        let n = accel.resident_count();
        tracing::info!(
            adapter = accel.name(),
            resident = n,
            full_forward = self.gpu_decode,
            "GPU accelerator attached"
        );
        self.accel = Some(accel);
        n
    }

    /// On-device decode of one token: compute the (scaled) embedding on the CPU,
    /// then run the entire forward + lm_head on the accelerator (~1 submit).
    fn gpu_decode_step(&mut self, token: u32) -> Result<Vec<f32>> {
        let tok = token as usize;
        if tok >= self.cfg.vocab {
            return Err(StrixError::invalid("gpu decode: token out of range"));
        }
        if self.seq >= self.max_seq() {
            return Err(StrixError::invalid("gpu decode: kv cache full"));
        }
        // Embedding row, scaled by sqrt(hidden).
        let (eb, ety, e_in, _) = Self::weight(&self.gguf, "token_embd.weight")?;
        let mut h = vec![0.0f32; self.cfg.hidden];
        dequant_row(eb, ety, tok, e_in, &mut h)?;
        let scale = self.cfg.emb_scale;
        for x in h.iter_mut() {
            *x *= scale;
        }
        let pos = self.seq;
        let logits = self
            .accel
            .as_mut()
            .and_then(|a| a.decode_step(&h, pos))
            .ok_or_else(|| StrixError::invalid("gpu decode_step failed"))?;
        self.seq += 1;
        Ok(logits)
    }

    /// Greedy on-device decode: like [`Self::gpu_decode_step`] but returns only the
    /// argmax token id (no vocab-wide logits readback). `Ok(None)` if the backend
    /// has no on-device argmax (caller falls back to logits + CPU argmax).
    fn gpu_decode_step_argmax(&mut self, token: u32) -> Result<Option<u32>> {
        let tok = token as usize;
        if tok >= self.cfg.vocab {
            return Err(StrixError::invalid("gpu decode: token out of range"));
        }
        if self.seq >= self.max_seq() {
            return Err(StrixError::invalid("gpu decode: kv cache full"));
        }
        let (eb, ety, e_in, _) = Self::weight(&self.gguf, "token_embd.weight")?;
        let mut h = vec![0.0f32; self.cfg.hidden];
        dequant_row(eb, ety, tok, e_in, &mut h)?;
        let scale = self.cfg.emb_scale;
        for x in h.iter_mut() {
            *x *= scale;
        }
        let pos = self.seq;
        let next = self
            .accel
            .as_mut()
            .and_then(|a| a.decode_step_argmax(&h, pos));
        match next {
            Some(t) => {
                self.seq += 1;
                Ok(Some(t))
            }
            None => Ok(None), // unsupported — seq NOT advanced; caller retries via decode_one
        }
    }

    /// Batched on-device prefill: embed all prompt tokens on the CPU, then run
    /// the accelerator's batched forward in chunks of `prefill_max`. Each weight
    /// is read once per chunk (compute-bound GEMM) instead of once per token.
    /// Returns the last token's logits and leaves the GPU KV cache populated.
    fn gpu_prefill_batched(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let cap = self.accel.as_ref().map_or(0, |a| a.prefill_max());
        let hidden = self.cfg.hidden;
        let scale = self.cfg.emb_scale;
        let mut last = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let m = (tokens.len() - i).min(cap);
            if self.seq + m > self.max_seq() {
                return Err(StrixError::invalid("gpu prefill: kv cache full"));
            }
            let mut hb = vec![0.0f32; m * hidden];
            {
                let (eb, ety, e_in, _) = Self::weight(&self.gguf, "token_embd.weight")?;
                for j in 0..m {
                    let tok = tokens[i + j] as usize;
                    if tok >= self.cfg.vocab {
                        return Err(StrixError::invalid("gpu prefill: token out of range"));
                    }
                    let row = &mut hb[j * hidden..(j + 1) * hidden];
                    dequant_row(eb, ety, tok, e_in, row)?;
                    for x in row.iter_mut() {
                        *x *= scale;
                    }
                }
            }
            let start = self.seq;
            last = self
                .accel
                .as_mut()
                .and_then(|a| a.prefill(&hb, start, m))
                .ok_or_else(|| StrixError::invalid("gpu prefill failed"))?;
            self.seq += m;
            i += m;
        }
        Ok(last)
    }

    /// Max sequence length the cache holds.
    pub fn max_seq(&self) -> usize {
        self.cache.first().map(|c| c.max_seq).unwrap_or(0)
    }

    /// Current sequence position (committed KV length).
    pub fn seq(&self) -> usize {
        self.seq
    }

    /// Roll the sequence position back to `s` (speculative rejection). The KV
    /// cache is position-indexed, so stale slots beyond `s` are overwritten
    /// before they're read again.
    pub fn set_seq(&mut self, s: usize) {
        self.seq = s;
    }

    /// Speculative verify: forward `tokens` at positions `[start_pos, start_pos+n)`
    /// in ONE batched pass (weights read once), returning logits for EVERY token.
    /// Fills the KV cache for those positions. Requires the batched accelerator.
    pub fn verify(&mut self, tokens: &[u32], start_pos: usize) -> Result<Vec<Vec<f32>>> {
        let m = tokens.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        let hidden = self.cfg.hidden;
        let scale = self.cfg.emb_scale;
        let vocab = self.cfg.vocab;
        let mut hb = vec![0.0f32; m * hidden];
        {
            let (eb, ety, e_in, _) = Self::weight(&self.gguf, "token_embd.weight")?;
            for (j, &tok) in tokens.iter().enumerate() {
                let row = &mut hb[j * hidden..(j + 1) * hidden];
                dequant_row(eb, ety, tok as usize, e_in, row)?;
                for x in row.iter_mut() {
                    *x *= scale;
                }
            }
        }
        let all = self
            .accel
            .as_mut()
            .and_then(|a| a.verify(&hb, start_pos, m))
            .ok_or_else(|| {
                StrixError::invalid("verify: accelerator does not support batched verify")
            })?;
        self.seq = start_pos + m;
        Ok(all.chunks(vocab).map(|c| c.to_vec()).collect())
    }

    /// Raw bytes + type for a weight tensor.
    fn weight<'a>(gguf: &'a GgufFile, name: &str) -> Result<(&'a [u8], GgmlType, usize, usize)> {
        let info = gguf
            .tensors()
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("gemma: missing weight `{name}`")))?;
        let in_dim = info.dims.first().copied().unwrap_or(0) as usize;
        let out_dim = info.dims.get(1).copied().unwrap_or(0) as usize;
        let bytes = gguf.tensor_bytes(name)?;
        Ok((bytes, info.ggml_type, in_dim, out_dim))
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let tok = token as usize;
        if tok >= self.cfg.vocab {
            return Err(StrixError::invalid(format!(
                "token {tok} out of range (vocab {})",
                self.cfg.vocab
            )));
        }
        if self.seq >= self.max_seq() {
            return Err(StrixError::invalid("gemma: kv cache full"));
        }
        let attn_rsqrt = self.cfg.attn_rsqrt;

        // Split borrows so we can read the gguf while mutating the cache.
        let Self {
            gguf,
            cfg,
            norms,
            output_norm,
            rope_freqs,
            cache,
            seq,
            accel,
            ..
        } = self;
        let hidden = cfg.hidden;
        let eps = cfg.eps;

        // GEMV dispatcher: run on the accelerator if it adopted this weight,
        // otherwise dequantize-and-dot on the CPU. `out` is filled in place.
        let accel_ref = accel.as_deref();
        let run_mm = |name: &str, out: &mut [f32], x: &[f32]| -> Result<()> {
            if let Some(a) = accel_ref {
                if let Some(y) = a.gemv(name, x) {
                    out.copy_from_slice(&y);
                    return Ok(());
                }
            }
            let (b, t, in_dim, _) = Self::weight(gguf, name)?;
            qlinear(out, x, b, t, in_dim);
            Ok(())
        };

        // Batched GEMV for a group of weights that share the input `x` (q/k/v,
        // gate/up): one accelerator submission instead of one per weight. Falls
        // back to per-weight CPU `qlinear` when the accelerator declines any.
        let run_group = |names: &[String], x: &[f32]| -> Result<Vec<Vec<f32>>> {
            if let Some(a) = accel_ref {
                let calls: Vec<(&str, &[f32])> = names.iter().map(|n| (n.as_str(), x)).collect();
                let res = a.gemv_batch(&calls);
                if res.iter().all(Option::is_some) {
                    return Ok(res.into_iter().map(Option::unwrap).collect());
                }
            }
            let mut out = Vec::with_capacity(names.len());
            for n in names {
                let (b, t, in_dim, od) = Self::weight(gguf, n)?;
                let mut y = vec![0.0f32; od];
                qlinear(&mut y, x, b, t, in_dim);
                out.push(y);
            }
            Ok(out)
        };

        // Embedding row (dequantized) scaled by sqrt(hidden).
        let (eb, ety, e_in, _) = Self::weight(gguf, "token_embd.weight")?;
        let mut h = vec![0.0f32; hidden];
        dequant_row(eb, ety, tok, e_in, &mut h)?;
        for x in h.iter_mut() {
            *x *= cfg.emb_scale;
        }

        let pos = *seq;
        *seq += 1;
        let len = pos + 1;

        for l in 0..cfg.n_layers {
            let lc = &cfg.layers[l];
            let nrm = &norms[l];
            let hd = lc.head_dim;
            let p = format!("blk.{l}");

            // ---- Attention ----
            let mut xn = vec![0.0f32; hidden];
            norm(&mut xn, &h, &nrm.attn_norm, eps);

            // q/k/(v) all consume `xn` → one batched submission on the accelerator.
            let mut qkv_names = vec![format!("{p}.attn_q.weight"), format!("{p}.attn_k.weight")];
            if !lc.k_eq_v {
                qkv_names.push(format!("{p}.attn_v.weight"));
            }
            let mut proj = run_group(&qkv_names, &xn)?;
            let mut q = std::mem::take(&mut proj[0]);
            let mut kk = std::mem::take(&mut proj[1]);

            // Q-norm (per head).
            for head in 0..cfg.n_heads {
                norm_inplace(&mut q[head * hd..(head + 1) * hd], &nrm.q_norm, eps);
            }
            // Build V from the *raw* projection: own v_proj (local) or the raw
            // k_proj (global k_eq_v). Gemma-4 RMS-normalizes V per head with NO
            // weight; this must happen BEFORE k_norm is applied to K.
            let mut vv = if lc.k_eq_v {
                kk.clone() // raw k_proj output (pre k_norm)
            } else {
                std::mem::take(&mut proj[2])
            };
            if cfg.norm_v {
                for head in 0..lc.n_kv {
                    let s = &mut vv[head * hd..(head + 1) * hd];
                    let ms = s.iter().map(|x| x * x).sum::<f32>() / hd as f32;
                    let sc = 1.0 / (ms + eps).sqrt();
                    for x in s.iter_mut() {
                        *x *= sc;
                    }
                }
            }
            // K-norm (per head) — applied AFTER V was captured from raw k.
            for head in 0..lc.n_kv {
                norm_inplace(&mut kk[head * hd..(head + 1) * hd], &nrm.k_norm, eps);
            }

            // RoPE applies to Q and K only (V stays unrotated), then cache K/V.
            // Global (full-attention) layers use proportional-RoPE freq factors.
            let q_theta = lc.rope_theta;
            let ff: Option<&[f32]> = if !lc.is_local {
                rope_freqs.as_deref()
            } else {
                None
            };
            for head in 0..cfg.n_heads {
                rope_in_place_ff(&mut q[head * hd..(head + 1) * hd], pos, q_theta, ff);
            }
            for head in 0..lc.n_kv {
                rope_in_place_ff(&mut kk[head * hd..(head + 1) * hd], pos, q_theta, ff);
                cache[l].store(
                    head,
                    pos,
                    &kk[head * hd..(head + 1) * hd],
                    &vv[head * hd..(head + 1) * hd],
                );
            }

            // Causal GQA attention per query head. Local (sliding-window) layers
            // attend only the last `n_swa` keys: query at pos `len-1` sees keys in
            // [len-1 - n_swa + 1, len-1] (llama.cpp: key masked iff p1-p0 >= n_swa).
            // That is exactly the cache suffix of length min(len, n_swa).
            let mut attn = vec![0.0f32; lc.q_dim];
            let groups = cfg.n_heads / lc.n_kv.max(1);
            let win = if lc.is_local && cfg.n_swa > 0 {
                cfg.n_swa
            } else {
                usize::MAX
            };
            let win_start = len.saturating_sub(win);
            let win_len = len - win_start;
            let off = win_start * hd;
            let mut scores = vec![0.0f32; win_len];
            for head in 0..cfg.n_heads {
                let kvh = head / groups;
                sdpa_single(
                    &mut attn[head * hd..(head + 1) * hd],
                    &q[head * hd..(head + 1) * hd],
                    &cache[l].keys(kvh, len)[off..],
                    &cache[l].values(kvh, len)[off..],
                    hd,
                    win_len,
                    if attn_rsqrt {
                        1.0 / (hd as f32).sqrt()
                    } else {
                        1.0
                    },
                    &mut scores,
                );
            }
            // o_proj, post-attn norm, residual.
            let mut o = vec![0.0f32; hidden];
            run_mm(&format!("{p}.attn_output.weight"), &mut o, &attn)?;
            let mut on = vec![0.0f32; hidden];
            norm(&mut on, &o, &nrm.post_attn_norm, eps);
            for i in 0..hidden {
                h[i] += on[i];
            }

            // ---- MLP (GeGLU) ----
            let mut xn2 = vec![0.0f32; hidden];
            norm(&mut xn2, &h, &nrm.ffn_norm, eps);
            // gate/up both consume `xn2` → one batched submission.
            let mut gu = run_group(
                &[format!("{p}.ffn_gate.weight"), format!("{p}.ffn_up.weight")],
                &xn2,
            )?;
            let gate = std::mem::take(&mut gu[0]);
            let up = std::mem::take(&mut gu[1]);
            let mut act = vec![0.0f32; cfg.ffn];
            geglu(&mut act, &gate, &up);
            let mut down = vec![0.0f32; hidden];
            run_mm(&format!("{p}.ffn_down.weight"), &mut down, &act)?;
            let mut dn = vec![0.0f32; hidden];
            norm(&mut dn, &down, &nrm.post_ffw_norm, eps);
            for i in 0..hidden {
                h[i] += dn[i];
            }

            // Per-layer output scalar applied to the whole residual stream
            // (Gemma-4 `hidden_states *= layer_scalar`, final op of the layer).
            let scl = nrm.output_scale;
            for x in h.iter_mut() {
                *x *= scl;
            }
            if std::env::var("STRIX_DBG").is_ok() {
                let n = h.len() as f32;
                let rms = (h.iter().map(|x| x * x).sum::<f32>() / n).sqrt();
                let mx = h.iter().fold(0.0f32, |a, x| a.max(x.abs()));
                eprintln!(
                    "L{l:02} out_scale={scl:.5} hd={hd} n_kv={} ||h||rms={rms:.4} max={mx:.4}",
                    lc.n_kv
                );
            }
        }

        // Final norm + tied LM head (token_embd) + soft-cap.
        let mut hn = vec![0.0f32; hidden];
        norm(&mut hn, &h, output_norm, eps);
        let mut logits = vec![0.0f32; cfg.vocab];
        run_mm("token_embd.weight", &mut logits, &hn)?;
        if cfg.final_softcap > 0.0 {
            softcap_inplace(&mut logits, cfg.final_softcap);
        }
        Ok(logits)
    }
}

/// Dequantize a single row (`row_len` block-aligned elements) of a tensor.
fn dequant_row(
    bytes: &[u8],
    ty: GgmlType,
    row: usize,
    row_len: usize,
    out: &mut [f32],
) -> Result<()> {
    let bpr = (row_len / ty.block_elems()) * ty.block_bytes();
    let start = row * bpr;
    dequantize_into(ty, &bytes[start..start + bpr], out)
}

impl Model for GemmaModel {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Unknown
    }
    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl Decoder for GemmaModel {
    fn prefill(&mut self, input_tokens: &[u32]) -> Result<Logits> {
        if input_tokens.is_empty() {
            return Err(StrixError::invalid("gemma prefill: empty prompt"));
        }
        self.reset();
        // Batched on-device prefill (compute-bound GEMM) when the accelerator
        // supports it — each weight read once per chunk instead of per token.
        if self.gpu_decode && self.accel.as_ref().map_or(0, |a| a.prefill_max()) > 0 {
            let last = self.gpu_prefill_batched(input_tokens)?;
            return Ok(Logits::new(last));
        }
        let mut last = Vec::new();
        for &t in input_tokens {
            // Fallback: token-by-token (route through the on-device decode when
            // enabled so the GPU KV cache is populated for decode).
            last = if self.gpu_decode {
                self.gpu_decode_step(t)?
            } else {
                self.forward(t)?
            };
        }
        Ok(Logits::new(last))
    }

    fn decode_one(&mut self, token: u32) -> Result<Logits> {
        let logits = if self.gpu_decode {
            self.gpu_decode_step(token)?
        } else {
            self.forward(token)?
        };
        Ok(Logits::new(logits))
    }

    fn decode_one_token(&mut self, token: u32) -> Result<u32> {
        if self.gpu_decode && std::env::var("STRIX_NO_ARGMAX").is_err() {
            if let Some(t) = self.gpu_decode_step_argmax(token)? {
                return Ok(t);
            }
        }
        // Fallback: full logits + CPU argmax (default-trait behavior).
        let logits = self.decode_one(token)?;
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in logits.0.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        Ok(bi as u32)
    }

    fn reset(&mut self) {
        self.seq = 0;
    }
}
