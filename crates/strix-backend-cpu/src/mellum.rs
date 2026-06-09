//! JetBrains Mellum2-12B-A2.5B (`mellum`) CPU reference forward.
//!
//! A Mixtral-class sparse-MoE transformer with two twists, both verified against
//! `refs/llama.cpp/src/models/mellum.cpp`:
//!
//! - **Hybrid attention**: every 4th layer (il where `(il+1)%4==0`) is full
//!   attention; the rest are sliding-window (window=1024). Full layers use YaRN
//!   RoPE (freq_scale=1/16, ext_factor=1, mscale via attn_factor); sliding layers
//!   use plain RoPE (freq_scale=1, ext_factor=0). Both share freq_base=500000.
//! - **Per-head Q/K RMSNorm** over head_dim (128) before RoPE.
//!
//! MoE: 64 experts, top-8 softmax + renorm (norm_topk_prob), SiLU, NO shared expert.
//! GQA: 32 q-heads / 4 kv-heads (groups=8), head_dim=128. Separate output.weight.
//!
//! Token-at-a-time forward with a KV cache (mirrors the qwen35 reference). CPU-only,
//! on-the-fly dequant matmul (12B can't fit as f32). Correctness-first.

use std::f32::consts::PI;

use rayon::prelude::*;
use strix_core::backend::Decoder;
use strix_core::error::{Result, StrixError};
use strix_core::sampler::Logits;
use strix_models::ggml_quant::{dequantize_into, GgmlType};
use strix_models::gguf::GgufFile;

fn mu32(g: &GgufFile, key: &str) -> Result<u32> {
    g.meta_u32(key)
}
fn mu32_or(g: &GgufFile, key: &str, d: u32) -> u32 {
    g.meta_u32(key).unwrap_or(d)
}
fn mf32_or(g: &GgufFile, key: &str, d: f32) -> f32 {
    g.meta_f32(key).unwrap_or(d)
}

/// Parsed `mellum` hyper-parameters.
#[derive(Debug, Clone)]
pub struct MellumCfg {
    pub n_layer: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub ctx_len: usize,
    pub rms_eps: f32,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_freq_base: f32,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_ff: usize,
    pub sliding_window: usize,
    pub swa_period: usize,
    // YaRN (full-attention layers only)
    pub yarn_factor: f32,      // rope.scaling.factor (e.g. 16)
    pub yarn_orig_ctx: f32,    // rope.scaling.original_context_length (e.g. 8192)
    pub yarn_attn_factor: f32, // rope.scaling.attn_factor (mscale passed in; usually 1.0)
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
}

impl MellumCfg {
    pub fn from_gguf(g: &GgufFile) -> Result<MellumCfg> {
        let arch = g
            .architecture()
            .ok_or_else(|| StrixError::invalid("mellum: no general.architecture"))?;
        if arch != "mellum" {
            return Err(StrixError::unsupported(format!(
                "mellum: arch `{arch}` is not mellum"
            )));
        }
        let k = |s: &str| format!("mellum.{s}");
        let vocab = g
            .tensors()
            .get("token_embd.weight")
            .and_then(|t| t.dims.last().copied())
            .map(|d| d as usize)
            .or_else(|| g.meta_u32(&k("vocab_size")).ok().map(|v| v as usize))
            .ok_or_else(|| StrixError::invalid("mellum: cannot determine vocab"))?;

        // Mellum has no `rope.dimension_count` key; llama.cpp asserts n_rot == head_dim.
        let head_dim = mu32(g, &k("attention.key_length"))? as usize;
        Ok(MellumCfg {
            n_layer: mu32(g, &k("block_count"))? as usize,
            hidden: mu32(g, &k("embedding_length"))? as usize,
            vocab,
            ctx_len: mu32_or(g, &k("context_length"), 0) as usize,
            rms_eps: mf32_or(g, &k("attention.layer_norm_rms_epsilon"), 1e-6),
            n_head: mu32(g, &k("attention.head_count"))? as usize,
            n_head_kv: mu32(g, &k("attention.head_count_kv"))? as usize,
            head_dim,
            n_rot: mu32_or(g, &k("rope.dimension_count"), head_dim as u32) as usize,
            rope_freq_base: mf32_or(g, &k("rope.freq_base"), 500000.0),
            n_expert: mu32(g, &k("expert_count"))? as usize,
            n_expert_used: mu32(g, &k("expert_used_count"))? as usize,
            expert_ff: mu32(g, &k("expert_feed_forward_length"))? as usize,
            sliding_window: mu32_or(g, &k("attention.sliding_window"), 0) as usize,
            swa_period: mu32_or(g, &k("attention.sliding_window_pattern"), 4) as usize,
            yarn_factor: mf32_or(g, &k("rope.scaling.factor"), 1.0),
            yarn_orig_ctx: mu32_or(g, &k("rope.scaling.original_context_length"), 0) as f32,
            yarn_attn_factor: mf32_or(g, &k("rope.scaling.attn_factor"), 1.0),
            yarn_beta_fast: mf32_or(g, &k("rope.scaling.yarn_beta_fast"), 32.0),
            yarn_beta_slow: mf32_or(g, &k("rope.scaling.yarn_beta_slow"), 1.0),
        })
    }

    /// True if layer `il` is a sliding-window-attention layer (else full attention).
    /// Matches llama.cpp `set_swa_pattern`: every `swa_period`-th layer is full.
    pub fn is_swa(&self, il: usize) -> bool {
        self.sliding_window > 0 && self.swa_period > 0 && (il + 1) % self.swa_period != 0
    }

    pub fn report(&self) -> String {
        let n_full = (0..self.n_layer).filter(|&l| !self.is_swa(l)).count();
        format!(
            "mellum: {} layers ({} sliding / {} full-attn, window={}, period={}), hidden={}, vocab={}, ctx={}\n  \
             attn: head_dim={} n_head={} n_head_kv={} (GQA {}:1), QK-norm, NEOX rope n_rot={} freq_base={:.0}\n  \
             YaRN(full layers): factor={} orig_ctx={} attn_factor={} beta=[{},{}]\n  \
             MoE: {} experts top-{}, expert_ff={} (no shared expert), rms_eps={:.1e}",
            self.n_layer, self.n_layer - n_full, n_full, self.sliding_window, self.swa_period,
            self.hidden, self.vocab, self.ctx_len,
            self.head_dim, self.n_head, self.n_head_kv, self.n_head / self.n_head_kv.max(1),
            self.n_rot, self.rope_freq_base,
            self.yarn_factor, self.yarn_orig_ctx, self.yarn_attn_factor, self.yarn_beta_fast, self.yarn_beta_slow,
            self.n_expert, self.n_expert_used, self.expert_ff, self.rms_eps,
        )
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RMSNorm over a slice with a weight vector (out = x/rms(x) * w).
fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let r = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * r * w[i];
    }
}

// --- YaRN RoPE (replicates ggml `rope_yarn` / `ggml_rope_cache_init`, NEOX) ---

fn yarn_corr_dim(n_dims: usize, n_ctx_orig: f32, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * (n_ctx_orig / (n_rot * 2.0 * PI)).ln() / (2.0 * base.ln())
}

fn yarn_corr_dims(
    n_dims: usize,
    n_ctx_orig: f32,
    base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> [f32; 2] {
    let start = yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, base).floor();
    let end = yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, base).ceil();
    [start.max(0.0), end.min(n_dims as f32 - 1.0)]
}

#[inline]
fn yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
    let y = ((i0 / 2) as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// One YaRN-corrected (cos,sin) for pair index, exactly as ggml `rope_yarn`.
#[inline]
fn rope_yarn(
    theta_extrap: f32,
    freq_scale: f32,
    corr: [f32; 2],
    i0: usize,
    ext_factor: f32,
    mscale_in: f32,
) -> (f32, f32) {
    let theta_interp = freq_scale * theta_extrap;
    let mut theta = theta_interp;
    let mut mscale = mscale_in;
    if ext_factor != 0.0 {
        let ramp_mix = yarn_ramp(corr[0], corr[1], i0) * ext_factor;
        theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
    }
    (theta.cos() * mscale, theta.sin() * mscale)
}

/// NEOX RoPE on a head vector (`vec.len() == head_dim`, rotating the first `n_dims`).
/// `ext_factor`/`freq_scale`/`mscale` select plain (sliding) vs YaRN (full) behaviour.
#[allow(clippy::too_many_arguments)]
fn rope_neox(
    vec: &mut [f32],
    pos: usize,
    n_dims: usize,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    mscale: f32,
    corr: [f32; 2],
) {
    let half = n_dims / 2;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let mut theta = pos as f32;
    for k in 0..half {
        let (c, s) = rope_yarn(theta, freq_scale, corr, 2 * k, ext_factor, mscale);
        let x0 = vec[k];
        let x1 = vec[k + half];
        vec[k] = x0 * c - x1 * s;
        vec[k + half] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

/// On-the-fly dequant matmul: out[o] = sum_i W[o][i]*x[i]. `bytes` = a weight whose
/// rows are `in_dim` elements each (gguf dims [in_dim, out_dim]); out.len() = out_dim.
fn qmatmul(
    out: &mut [f32],
    x: &[f32],
    bytes: &[u8],
    ty: GgmlType,
    in_dim: usize,
    _row: &mut [f32],
) {
    let bpr = (in_dim / ty.block_elems()) * ty.block_bytes();
    // Rows are independent: parallelize across cores with per-thread dequant scratch
    // (the `_row` arg is kept for call-site compatibility but no longer used).
    out.par_iter_mut().enumerate().for_each_init(
        || vec![0.0f32; in_dim],
        |scratch, (o, oref)| {
            dequantize_into(ty, &bytes[o * bpr..o * bpr + bpr], scratch).unwrap();
            *oref = dot_f32(scratch, x);
        },
    );
}

/// Batched on-the-fly dequant matmul: `out[t][o] = W·xs[t]` for m tokens at once.
/// The win over m `qmatmul` calls: each weight row is dequantized ONCE per chunk,
/// not once per token — token-at-a-time prefill is dequant-bound (~99% of cost).
/// xs: [m * in_dim] token-major; out: [m * out_dim]. Per-token dots are `dot_f32`,
/// so each row's value is bit-identical to the token-at-a-time path.
fn qmatmul_batch(out: &mut [f32], xs: &[f32], m: usize, bytes: &[u8], ty: GgmlType, in_dim: usize, out_dim: usize) {
    let bpr = (in_dim / ty.block_elems()) * ty.block_bytes();
    // Parallel over output rows; out[t*out_dim + o] strided writes per row, so write
    // into a row-major scratch [out_dim][m] then transpose at the end.
    let mut rt = vec![0.0f32; out_dim * m];
    rt.par_chunks_mut(m).enumerate().for_each_init(
        || vec![0.0f32; in_dim],
        |scratch, (o, orow)| {
            dequantize_into(ty, &bytes[o * bpr..o * bpr + bpr], scratch).unwrap();
            for t in 0..m {
                orow[t] = dot_f32(scratch, &xs[t * in_dim..(t + 1) * in_dim]);
            }
        },
    );
    for t in 0..m {
        for o in 0..out_dim {
            out[t * out_dim + o] = rt[o * m + t];
        }
    }
}

/// SIMD-friendly dot product. A plain `s += a[i]*b[i]` loop does NOT auto-vectorize
/// (f32 add isn't associative + no fast-math), so split into 8 independent lane
/// accumulators that LLVM lowers to vector FMAs, then reduce. `len` is a multiple of
/// 32 (block size) so the tail loop is dead in practice.
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

/// A resolved weight: raw bytes + ggml type + the row length (in_dim).
struct W<'a> {
    bytes: &'a [u8],
    ty: GgmlType,
    in_dim: usize,
}

pub struct MellumModel {
    gguf: GgufFile,
    cfg: MellumCfg,
    // per layer: KV cache, flat [t * kv_dim ..] (kv_dim = n_head_kv * head_dim)
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    pos: usize,
    // Optional iGPU weight accelerator (per-weight GEMV by tensor name). `None`
    // (default / CPU build) → byte-identical CPU forward. Mellum is all-Q8_0, so this
    // needs a Q8_0-capable accel (RocmWeightAccel). See docs/ideas/moe-accel-plan.md.
    accel: Option<Box<dyn strix_core::WeightAccel>>,
}

impl MellumModel {
    pub fn from_gguf(gguf: GgufFile) -> Result<Self> {
        let cfg = MellumCfg::from_gguf(&gguf)?;
        let nl = cfg.n_layer;
        Ok(MellumModel {
            kc: vec![Vec::new(); nl],
            vc: vec![Vec::new(); nl],
            cfg,
            gguf,
            pos: 0,
            accel: None,
        })
    }

    /// Attach an iGPU accelerator and upload Mellum's Q8_0 weights resident: dense
    /// attn q/k/v/o + output, plus the per-layer MoE experts (capped by
    /// STRIX_GPU_EXPERT_LAYERS). 12B all-Q8_0 → ~12GB, fits the iGPU. Returns the
    /// count adopted. Needs a Q8_0-capable accel (ROCm); Vulkan adopts nothing.
    pub fn attach_accel(&mut self, mut accel: Box<dyn strix_core::WeightAccel>) -> usize {
        let mut n = 0usize;
        let up = |accel: &mut Box<dyn strix_core::WeightAccel>,
                  key: &str,
                  bytes: &[u8],
                  ty: GgmlType,
                  in_dim: usize,
                  out_dim: usize|
         -> bool {
            match ty {
                GgmlType::Q4_0 => accel.upload_q4_0(key, bytes, in_dim, out_dim),
                GgmlType::Q6K => accel.upload_q6_k(key, bytes, in_dim, out_dim),
                GgmlType::Q8_0 => accel.upload_q8_0(key, bytes, in_dim, out_dim),
                _ => false,
            }
        };
        // dense projections + output
        let mut names: Vec<String> = Vec::new();
        for il in 0..self.cfg.n_layer {
            for t in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                names.push(format!("blk.{il}.{t}.weight"));
            }
        }
        if self.gguf.tensors().contains_key("output.weight") {
            names.push("output.weight".to_string());
        }
        for name in &names {
            let Some(ti) = self.gguf.tensors().get(name) else {
                continue;
            };
            let (ty, in_dim) = (ti.ggml_type, ti.dims[0] as usize);
            let out_dim: usize = ti.dims[1..].iter().map(|&d| d as usize).product();
            if let Ok(bytes) = self.gguf.tensor_bytes(name) {
                if up(&mut accel, name, bytes, ty, in_dim, out_dim) {
                    n += 1;
                }
            }
        }
        // MoE experts (per-expert byte slices), capped for memory.
        let hidden = self.cfg.hidden;
        let eff = self.cfg.expert_ff;
        let ne = self.cfg.n_expert;
        let layer_cap = std::env::var("STRIX_GPU_EXPERT_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(self.cfg.n_layer);
        let mut advise: Vec<String> = Vec::new();
        for il in 0..self.cfg.n_layer.min(layer_cap) {
            for (tname, in_dim, out_dim) in [
                ("ffn_gate_exps", hidden, eff),
                ("ffn_up_exps", hidden, eff),
                ("ffn_down_exps", eff, hidden),
            ] {
                let full = format!("blk.{il}.{tname}.weight");
                let Some(ti) = self.gguf.tensors().get(&full) else {
                    continue;
                };
                let ty = ti.ggml_type;
                if !matches!(ty, GgmlType::Q6K | GgmlType::Q4_0 | GgmlType::Q8_0) {
                    continue;
                }
                let bpr = (in_dim / ty.block_elems()) * ty.block_bytes() * out_dim;
                let Ok(bytes) = self.gguf.tensor_bytes(&full) else {
                    continue;
                };
                let mut got = 0usize;
                for e in 0..ne {
                    let slice = &bytes[e * bpr..(e + 1) * bpr];
                    if up(&mut accel, &format!("blk.{il}.{tname}.e{e}"), slice, ty, in_dim, out_dim) {
                        got += 1;
                    }
                }
                n += got;
                if got == ne {
                    advise.push(full.clone());
                }
            }
        }
        for name in &advise {
            self.gguf.advise_dontneed(name);
        }
        self.accel = Some(accel);
        n
    }

    /// GPU gemv for `key` into `out` if adopted; false ⇒ caller uses CPU.
    fn try_gemv(&self, key: &str, x: &[f32], out: &mut [f32]) -> bool {
        if let Some(a) = &self.accel {
            if let Some(y) = a.gemv(key, x) {
                if y.len() == out.len() {
                    out.copy_from_slice(&y);
                    return true;
                }
            }
        }
        false
    }

    fn mm(&self, key: &str, out: &mut [f32], x: &[f32], w: &W<'_>, row: &mut [f32]) {
        if !self.try_gemv(key, x, out) {
            qmatmul(out, x, w.bytes, w.ty, w.in_dim, row);
        }
    }

    fn w(&self, name: &str) -> Result<W<'_>> {
        let ti = self
            .gguf
            .tensors()
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("mellum: missing tensor {name}")))?;
        let in_dim = ti.dims[0] as usize;
        Ok(W {
            bytes: self.gguf.tensor_bytes(name)?,
            ty: ti.ggml_type,
            in_dim,
        })
    }

    fn vecw(&self, name: &str) -> Result<Vec<f32>> {
        let ti = self
            .gguf
            .tensors()
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("mellum: missing tensor {name}")))?;
        let n: usize = ti.dims.iter().map(|&d| d as usize).product();
        let mut out = vec![0.0f32; n];
        dequantize_into(ti.ggml_type, self.gguf.tensor_bytes(name)?, &mut out)?;
        Ok(out)
    }

    /// Batched prefill: process the whole prompt with weight-read-once matmuls.
    /// Token-at-a-time prefill is dequant-bound (m re-dequants of every weight);
    /// batching reads each weight once → prefill becomes ~m× cheaper on dequant.
    /// Bit-identical to the token loop: each row's dot is `dot_f32`, per-token
    /// rope/norm/SDPA are the same code, and each token's MoE accumulates in its
    /// own routed-expert order. Returns logits of the LAST token.
    fn prefill_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let m = tokens.len();
        let hidden = cfg.hidden;
        let eps = cfg.rms_eps;
        let hd = cfg.head_dim;
        let nh = cfg.n_head;
        let nkv = cfg.n_head_kv;
        let groups = nh / nkv;
        let kv_dim = nkv * hd;
        let q_dim = nh * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let corr = yarn_corr_dims(
            cfg.n_rot,
            cfg.yarn_orig_ctx,
            cfg.rope_freq_base,
            cfg.yarn_beta_fast,
            cfg.yarn_beta_slow,
        );

        // embed all tokens
        let emb = self.w("token_embd.weight")?;
        let bpr_e = (hidden / emb.ty.block_elems()) * emb.ty.block_bytes();
        let mut h = vec![0.0f32; m * hidden];
        for (t, &tok) in tokens.iter().enumerate() {
            dequantize_into(
                emb.ty,
                &emb.bytes[tok as usize * bpr_e..(tok as usize + 1) * bpr_e],
                &mut h[t * hidden..(t + 1) * hidden],
            )?;
        }

        let mut n = vec![0.0f32; m * hidden];
        let mut q = vec![0.0f32; m * q_dim];
        let mut k = vec![0.0f32; m * kv_dim];
        let mut v = vec![0.0f32; m * kv_dim];
        let mut attn_out = vec![0.0f32; m * q_dim];
        let mut o = vec![0.0f32; m * hidden];

        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            let an = self.vecw(&b("attn_norm.weight"))?;
            for t in 0..m {
                let (hs, ns) = (&h[t * hidden..(t + 1) * hidden], &mut n[t * hidden..(t + 1) * hidden]);
                rmsnorm(ns, hs, &an, eps);
            }
            {
                let wq = self.w(&b("attn_q.weight"))?;
                qmatmul_batch(&mut q, &n, m, wq.bytes, wq.ty, hidden, q_dim);
                let wk = self.w(&b("attn_k.weight"))?;
                qmatmul_batch(&mut k, &n, m, wk.bytes, wk.ty, hidden, kv_dim);
                let wv = self.w(&b("attn_v.weight"))?;
                qmatmul_batch(&mut v, &n, m, wv.bytes, wv.ty, hidden, kv_dim);
            }
            let is_swa = cfg.is_swa(il);
            let (freq_scale, ext_factor, mscale) = if is_swa {
                (1.0, 0.0, 1.0)
            } else {
                (1.0 / cfg.yarn_factor, 1.0, cfg.yarn_attn_factor)
            };
            let qn = self.vecw(&b("attn_q_norm.weight"))?;
            let kn = self.vecw(&b("attn_k_norm.weight"))?;
            for t in 0..m {
                let pos = self.pos + t;
                for hh in 0..nh {
                    let qh = &mut q[t * q_dim + hh * hd..t * q_dim + hh * hd + hd];
                    let mut tmp = vec![0.0f32; hd];
                    rmsnorm(&mut tmp, qh, &qn, eps);
                    qh.copy_from_slice(&tmp);
                    rope_neox(qh, pos, cfg.n_rot, cfg.rope_freq_base, freq_scale, ext_factor, mscale, corr);
                }
                for kh in 0..nkv {
                    let khv = &mut k[t * kv_dim + kh * hd..t * kv_dim + kh * hd + hd];
                    let mut tmp = vec![0.0f32; hd];
                    rmsnorm(&mut tmp, khv, &kn, eps);
                    khv.copy_from_slice(&tmp);
                    rope_neox(khv, pos, cfg.n_rot, cfg.rope_freq_base, freq_scale, ext_factor, mscale, corr);
                }
            }
            self.kc[il].extend_from_slice(&k[..m * kv_dim]);
            self.vc[il].extend_from_slice(&v[..m * kv_dim]);
            // per-token causal SDPA over the cache (parallel: cache is read-only now)
            let kc = &self.kc[il];
            let vc = &self.vc[il];
            let base = self.pos;
            attn_out
                .par_chunks_mut(q_dim)
                .enumerate()
                .for_each(|(t, ao)| {
                    let pos = base + t;
                    let len = pos + 1;
                    let win_start = if is_swa && cfg.sliding_window > 0 && len > cfg.sliding_window {
                        len - cfg.sliding_window
                    } else {
                        0
                    };
                    let wlen = len - win_start;
                    let mut keys = vec![0.0f32; wlen * hd];
                    let mut vals = vec![0.0f32; wlen * hd];
                    let mut scratch = vec![0.0f32; wlen];
                    for hh in 0..nh {
                        let kvh = hh / groups;
                        for (ti, tt) in (win_start..len).enumerate() {
                            keys[ti * hd..ti * hd + hd].copy_from_slice(
                                &kc[tt * kv_dim + kvh * hd..tt * kv_dim + kvh * hd + hd],
                            );
                            vals[ti * hd..ti * hd + hd].copy_from_slice(
                                &vc[tt * kv_dim + kvh * hd..tt * kv_dim + kvh * hd + hd],
                            );
                        }
                        let mut oh = vec![0.0f32; hd];
                        crate::attention::sdpa_single(
                            &mut oh,
                            &q[t * q_dim + hh * hd..t * q_dim + hh * hd + hd],
                            &keys,
                            &vals,
                            hd,
                            wlen,
                            scale,
                            &mut scratch,
                        );
                        ao[hh * hd..hh * hd + hd].copy_from_slice(&oh);
                    }
                });
            {
                let wo = self.w(&b("attn_output.weight"))?;
                qmatmul_batch(&mut o, &attn_out, m, wo.bytes, wo.ty, q_dim, hidden);
            }
            for i in 0..m * hidden {
                h[i] += o[i];
            }

            // MoE: route per token, group tokens by expert, batch each expert's GEMMs.
            let fnw = self.vecw(&b("ffn_norm.weight"))?;
            for t in 0..m {
                let (hs, ns) = (&h[t * hidden..(t + 1) * hidden], &mut n[t * hidden..(t + 1) * hidden]);
                rmsnorm(ns, hs, &fnw, eps);
            }
            let ne = cfg.n_expert;
            let topk = cfg.n_expert_used;
            let eff = cfg.expert_ff;
            let wgi = self.w(&b("ffn_gate_inp.weight"))?;
            let mut rl = vec![0.0f32; m * ne];
            qmatmul_batch(&mut rl, &n, m, wgi.bytes, wgi.ty, hidden, ne);
            // per-token top-k (same math as moe())
            let mut routes: Vec<Vec<(usize, f32)>> = Vec::with_capacity(m);
            for t in 0..m {
                let logits = &rl[t * ne..(t + 1) * ne];
                let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
                let sum: f32 = probs.iter().sum();
                for p in probs.iter_mut() {
                    *p /= sum;
                }
                let mut idx: Vec<usize> = (0..ne).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(topk);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
                routes.push(idx.into_iter().map(|e| (e, probs[e] / wsum)).collect());
            }
            // group (token, slot) by expert
            let mut by_exp: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
            for (t, route) in routes.iter().enumerate() {
                for (s, &(e, _)) in route.iter().enumerate() {
                    by_exp[e].push((t, s));
                }
            }
            let wge = self.w(&b("ffn_gate_exps.weight"))?;
            let wue = self.w(&b("ffn_up_exps.weight"))?;
            let wde = self.w(&b("ffn_down_exps.weight"))?;
            let g_bpr = (hidden / wge.ty.block_elems()) * wge.ty.block_bytes() * eff;
            let u_bpr = (hidden / wue.ty.block_elems()) * wue.ty.block_bytes() * eff;
            let d_bpr = (eff / wde.ty.block_elems()) * wde.ty.block_bytes() * hidden;
            // dy[t][slot] holds the down-output for token t's slot-th routed expert
            let mut dy = vec![0.0f32; m * topk * hidden];
            for (e, list) in by_exp.iter().enumerate() {
                if list.is_empty() {
                    continue;
                }
                let me = list.len();
                let mut xs = vec![0.0f32; me * hidden];
                for (i, &(t, _)) in list.iter().enumerate() {
                    xs[i * hidden..(i + 1) * hidden].copy_from_slice(&n[t * hidden..(t + 1) * hidden]);
                }
                let mut g = vec![0.0f32; me * eff];
                let mut u = vec![0.0f32; me * eff];
                qmatmul_batch(&mut g, &xs, me, &wge.bytes[e * g_bpr..(e + 1) * g_bpr], wge.ty, hidden, eff);
                qmatmul_batch(&mut u, &xs, me, &wue.bytes[e * u_bpr..(e + 1) * u_bpr], wue.ty, hidden, eff);
                let mut act = vec![0.0f32; me * eff];
                for i in 0..me * eff {
                    act[i] = silu(g[i]) * u[i];
                }
                let mut d = vec![0.0f32; me * hidden];
                qmatmul_batch(&mut d, &act, me, &wde.bytes[e * d_bpr..(e + 1) * d_bpr], wde.ty, eff, hidden);
                for (i, &(t, s)) in list.iter().enumerate() {
                    dy[(t * topk + s) * hidden..(t * topk + s + 1) * hidden]
                        .copy_from_slice(&d[i * hidden..(i + 1) * hidden]);
                }
            }
            // per-token accumulate in routed order into a zeroed buffer, then add to h
            // (exactly moe()'s float association: out = Σ w·d, then h + out).
            for (t, route) in routes.iter().enumerate() {
                let mut out = vec![0.0f32; hidden];
                for (s, &(_, w)) in route.iter().enumerate() {
                    let dys = &dy[(t * topk + s) * hidden..(t * topk + s + 1) * hidden];
                    for i in 0..hidden {
                        out[i] += w * dys[i];
                    }
                }
                let hrow = &mut h[t * hidden..(t + 1) * hidden];
                for i in 0..hidden {
                    hrow[i] += out[i];
                }
            }
        }
        self.pos += m;

        // lm_head on the last token only
        let on = self.vecw("output_norm.weight")?;
        let mut nh_ = vec![0.0f32; hidden];
        rmsnorm(&mut nh_, &h[(m - 1) * hidden..m * hidden], &on, eps);
        let (head_key, head) = if self.gguf.tensors().contains_key("output.weight") {
            ("output.weight", self.w("output.weight")?)
        } else {
            ("token_embd.weight", self.w("token_embd.weight")?)
        };
        let mut logits = vec![0.0f32; cfg.vocab];
        let mut row = vec![0.0f32; 8192];
        self.mm(head_key, &mut logits, &nh_, &head, &mut row);
        Ok(logits)
    }

    fn forward(&mut self, token: u32, want_logits: bool) -> Result<Option<Vec<f32>>> {
        let cfg = self.cfg.clone();
        let pos = self.pos;
        let hidden = cfg.hidden;
        let eps = cfg.rms_eps;
        let mut row = vec![0.0f32; 8192]; // dequant scratch (>= max in_dim)

        // embedding row
        let emb = self.w("token_embd.weight")?;
        let mut h = vec![0.0f32; hidden];
        {
            let bpr = (hidden / emb.ty.block_elems()) * emb.ty.block_bytes();
            dequantize_into(
                emb.ty,
                &emb.bytes[token as usize * bpr..token as usize * bpr + bpr],
                &mut h,
            )?;
        }

        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            let an = self.vecw(&b("attn_norm.weight"))?;
            let mut n = vec![0.0f32; hidden];
            rmsnorm(&mut n, &h, &an, eps);

            let attn = self.attn(&n, il, pos, &mut row)?;
            for i in 0..hidden {
                h[i] += attn[i];
            }

            let ffn_res = h.clone();
            let fn_ = self.vecw(&b("ffn_norm.weight"))?;
            let mut nn = vec![0.0f32; hidden];
            rmsnorm(&mut nn, &h, &fn_, eps);
            let moe = self.moe(&nn, il, &mut row)?;
            for i in 0..hidden {
                h[i] = ffn_res[i] + moe[i];
            }
        }
        self.pos += 1;

        if !want_logits {
            return Ok(None);
        }
        let on = self.vecw("output_norm.weight")?;
        let mut nh = vec![0.0f32; hidden];
        rmsnorm(&mut nh, &h, &on, eps);
        let (head_key, head) = if self.gguf.tensors().contains_key("output.weight") {
            ("output.weight", self.w("output.weight")?)
        } else {
            ("token_embd.weight", self.w("token_embd.weight")?)
        };
        let mut logits = vec![0.0f32; cfg.vocab];
        self.mm(head_key, &mut logits, &nh, &head, &mut row);
        Ok(Some(logits))
    }

    fn attn(&mut self, x: &[f32], il: usize, pos: usize, row: &mut [f32]) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let hd = cfg.head_dim; // 128
        let nh = cfg.n_head; // 32
        let nkv = cfg.n_head_kv; // 4
        let groups = nh / nkv; // 8
        let kv_dim = nkv * hd; // 512
        let b = |s: &str| format!("blk.{il}.{s}");
        let qn = self.vecw(&b("attn_q_norm.weight"))?;
        let kn = self.vecw(&b("attn_k_norm.weight"))?;
        let mut q = vec![0.0f32; hd * nh]; // 4096
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        {
            let wq = self.w(&b("attn_q.weight"))?;
            self.mm(&b("attn_q.weight"), &mut q, x, &wq, row);
            let wk = self.w(&b("attn_k.weight"))?;
            self.mm(&b("attn_k.weight"), &mut k, x, &wk, row);
            let wv = self.w(&b("attn_v.weight"))?;
            self.mm(&b("attn_v.weight"), &mut v, x, &wv, row);
        }

        // RoPE config for this layer: sliding => plain, full => YaRN.
        let is_swa = cfg.is_swa(il);
        let (freq_scale, ext_factor, mscale) = if is_swa {
            (1.0, 0.0, 1.0)
        } else {
            (1.0 / cfg.yarn_factor, 1.0, cfg.yarn_attn_factor)
        };
        let corr = yarn_corr_dims(
            cfg.n_rot,
            cfg.yarn_orig_ctx,
            cfg.rope_freq_base,
            cfg.yarn_beta_fast,
            cfg.yarn_beta_slow,
        );

        // per-head Q-norm + rope; per-kv-head K-norm + rope
        for hh in 0..nh {
            let qh = &mut q[hh * hd..hh * hd + hd];
            let mut tmp = vec![0.0f32; hd];
            rmsnorm(&mut tmp, qh, &qn, cfg.rms_eps);
            qh.copy_from_slice(&tmp);
            rope_neox(
                qh,
                pos,
                cfg.n_rot,
                cfg.rope_freq_base,
                freq_scale,
                ext_factor,
                mscale,
                corr,
            );
        }
        for kh in 0..nkv {
            let khv = &mut k[kh * hd..kh * hd + hd];
            let mut tmp = vec![0.0f32; hd];
            rmsnorm(&mut tmp, khv, &kn, cfg.rms_eps);
            khv.copy_from_slice(&tmp);
            rope_neox(
                khv,
                pos,
                cfg.n_rot,
                cfg.rope_freq_base,
                freq_scale,
                ext_factor,
                mscale,
                corr,
            );
        }

        // append to KV cache
        self.kc[il].extend_from_slice(&k);
        self.vc[il].extend_from_slice(&v);
        let len = pos + 1;
        // sliding window: attend to keys in [win_start, pos]
        let win_start = if is_swa && cfg.sliding_window > 0 && len > cfg.sliding_window {
            len - cfg.sliding_window
        } else {
            0
        };
        let wlen = len - win_start;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut attn_out = vec![0.0f32; hd * nh];
        let mut keys = vec![0.0f32; wlen * hd];
        let mut vals = vec![0.0f32; wlen * hd];
        let mut scratch = vec![0.0f32; wlen];
        for hh in 0..nh {
            let kvh = hh / groups;
            for (ti, t) in (win_start..len).enumerate() {
                keys[ti * hd..ti * hd + hd].copy_from_slice(
                    &self.kc[il][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd],
                );
                vals[ti * hd..ti * hd + hd].copy_from_slice(
                    &self.vc[il][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd],
                );
            }
            let mut oh = vec![0.0f32; hd];
            crate::attention::sdpa_single(
                &mut oh,
                &q[hh * hd..hh * hd + hd],
                &keys,
                &vals,
                hd,
                wlen,
                scale,
                &mut scratch,
            );
            attn_out[hh * hd..hh * hd + hd].copy_from_slice(&oh);
        }
        let mut o = vec![0.0f32; cfg.hidden];
        {
            let wo = self.w(&b("attn_output.weight"))?;
            self.mm(&b("attn_output.weight"), &mut o, &attn_out, &wo, row);
        }
        Ok(o)
    }

    fn moe(&self, x: &[f32], il: usize, row: &mut [f32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let ne = cfg.n_expert; // 64
        let topk = cfg.n_expert_used; // 8
        let eff = cfg.expert_ff; // 896
        let b = |s: &str| format!("blk.{il}.{s}");

        // router: softmax over all experts, take top-k, renormalize (norm_topk_prob)
        let wgi = self.w(&b("ffn_gate_inp.weight"))?;
        let mut logits = vec![0.0f32; ne];
        qmatmul(&mut logits, x, wgi.bytes, wgi.ty, wgi.in_dim, row);
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let mut idx: Vec<usize> = (0..ne).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        idx.truncate(topk);
        let wsum: f32 = idx.iter().map(|&e| probs[e]).sum();

        let wge = self.w(&b("ffn_gate_exps.weight"))?; // [hidden, eff, ne]
        let wue = self.w(&b("ffn_up_exps.weight"))?;
        let wde = self.w(&b("ffn_down_exps.weight"))?; // [eff, hidden, ne]
        let gate_bpr = (hidden / wge.ty.block_elems()) * wge.ty.block_bytes() * eff;
        let up_bpr = (hidden / wue.ty.block_elems()) * wue.ty.block_bytes() * eff;
        let down_bpr = (eff / wde.ty.block_elems()) * wde.ty.block_bytes() * hidden;

        let mut out = vec![0.0f32; hidden];
        let mut g = vec![0.0f32; eff];
        let mut u = vec![0.0f32; eff];
        let mut act = vec![0.0f32; eff];
        let mut dy = vec![0.0f32; hidden];
        for &e in &idx {
            let wexp = probs[e] / wsum;
            let gkey = format!("blk.{il}.ffn_gate_exps.e{e}");
            if !self.try_gemv(&gkey, x, &mut g) {
                qmatmul(&mut g, x, &wge.bytes[e * gate_bpr..(e + 1) * gate_bpr], wge.ty, hidden, row);
            }
            let ukey = format!("blk.{il}.ffn_up_exps.e{e}");
            if !self.try_gemv(&ukey, x, &mut u) {
                qmatmul(&mut u, x, &wue.bytes[e * up_bpr..(e + 1) * up_bpr], wue.ty, hidden, row);
            }
            for i in 0..eff {
                act[i] = silu(g[i]) * u[i];
            }
            let dkey = format!("blk.{il}.ffn_down_exps.e{e}");
            if !self.try_gemv(&dkey, &act, &mut dy) {
                qmatmul(&mut dy, &act, &wde.bytes[e * down_bpr..(e + 1) * down_bpr], wde.ty, eff, row);
            }
            for i in 0..hidden {
                out[i] += wexp * dy[i];
            }
        }
        Ok(out)
    }
}

impl Decoder for MellumModel {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits> {
        if tokens.is_empty() {
            return Err(StrixError::invalid("mellum prefill: empty"));
        }
        self.reset();
        // STRIX_NO_BATCH_PREFILL forces the token-at-a-time reference path.
        if std::env::var("STRIX_NO_BATCH_PREFILL").is_ok() {
            for (i, &t) in tokens.iter().enumerate() {
                let last = i == tokens.len() - 1;
                let o = self.forward(t, last)?;
                if last {
                    return Ok(Logits::new(o.unwrap()));
                }
            }
            unreachable!()
        }
        Ok(Logits::new(self.prefill_batch(tokens)?))
    }

    fn decode_one(&mut self, token: u32) -> Result<Logits> {
        Ok(Logits::new(self.forward(token, true)?.unwrap()))
    }

    fn reset(&mut self) {
        self.pos = 0;
        for v in self.kc.iter_mut() {
            v.clear();
        }
        for v in self.vc.iter_mut() {
            v.clear();
        }
    }
}
