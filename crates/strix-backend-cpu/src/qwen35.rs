//! Qwen3.5/3.6-MoE (`qwen35moe`) bring-up — Phase 0: config parsing + tensor
//! validation. This is a Qwen3-Next-class HYBRID model: most layers are Gated-
//! DeltaNet linear-attention (recurrent, `ssm_*` tensors), every `full_attention_
//! interval`-th layer is full GQA attention; every layer has a 256-expert top-8
//! MoE + a sigmoid-gated shared expert. See docs/qwen36-arch.md for the full spec.
//!
//! P0 only parses the architecture + verifies all expected tensors are present
//! (with correct shapes). The forward (P1 MoE, P2 attn, P3 gated-deltanet) is TODO.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use strix_core::accel::Q35GpuConfig;
use strix_core::backend::Decoder;
static PROF_GEMM: AtomicU64 = AtomicU64::new(0);
static PROF_SCAN: AtomicU64 = AtomicU64::new(0);
static PROF_SDPA: AtomicU64 = AtomicU64::new(0);
// Decode-path breakdown (µs), gated by STRIX_DECODE_PROF. Printed every 8 tokens.
static DPROF_MIX: AtomicU64 = AtomicU64::new(0);
static DPROF_FFN: AtomicU64 = AtomicU64::new(0);
static DPROF_HEAD: AtomicU64 = AtomicU64::new(0);
static DPROF_N: AtomicU64 = AtomicU64::new(0);
fn padd(c: &AtomicU64, t: std::time::Instant) {
    c.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
}
use strix_core::error::{Result, StrixError};
use strix_core::sampler::Logits;
use strix_models::ggml_quant::{dequantize, dequantize_into, quantize_q4_0, GgmlType};
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

/// Parsed `qwen35moe` hyper-parameters.
#[derive(Debug, Clone)]
pub struct Qwen35Cfg {
    pub n_layer: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub ctx_len: usize,
    pub rms_eps: f32,
    // attention (full-attn layers)
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize, // key_length == value_length
    pub rope_freq_base: f32,
    pub n_rot: usize, // rope dimension_count (partial: < head_dim)
    pub rope_sections: [i64; 4],
    pub full_attn_interval: usize,
    // MoE (`qwen35moe`); all zero for the dense `qwen35` arch.
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_ff: usize,
    pub shared_ff: usize,
    // Dense FFN size (`qwen35`); zero for the MoE arch.
    pub dense_ff: usize,
    // SSM / Gated-DeltaNet (recurrent layers)
    pub ssm_d_conv: usize,
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_dt_rank: usize, // = n_v_heads
    pub ssm_n_group: usize, // = n_k_heads
}

impl Qwen35Cfg {
    pub fn from_gguf(g: &GgufFile) -> Result<Qwen35Cfg> {
        let arch = g
            .architecture()
            .ok_or_else(|| StrixError::invalid("qwen35: no general.architecture"))?
            .to_string();
        // `qwen35` = dense, `qwen35moe` = MoE. Both are the same Qwen3.5/3.6-class
        // hybrid (Gated-DeltaNet + full attn); they differ only in the FFN block.
        if arch != "qwen35" && arch != "qwen35moe" {
            return Err(StrixError::unsupported(format!(
                "qwen35: arch `{arch}` is not qwen35/qwen35moe"
            )));
        }
        let k = |s: &str| format!("{arch}.{s}");
        // rope sections [11,11,10,0]
        let mut rope_sections = [0i64; 4];
        if let Some(arr) = g
            .meta(&k("rope.dimension_sections"))
            .and_then(|v| v.as_array())
        {
            for (i, v) in arr.iter().take(4).enumerate() {
                rope_sections[i] = v.as_u64().map(|x| x as i64).unwrap_or(0);
            }
        }
        // vocab from token_embd shape if not in metadata
        let vocab = g
            .tensors()
            .get("token_embd.weight")
            .and_then(|t| t.dims.last().copied())
            .map(|d| d as usize)
            .or_else(|| g.meta_u32(&k("vocab_size")).ok().map(|v| v as usize))
            .ok_or_else(|| StrixError::invalid("qwen35: cannot determine vocab"))?;

        Ok(Qwen35Cfg {
            n_layer: mu32(g, &k("block_count"))? as usize,
            hidden: mu32(g, &k("embedding_length"))? as usize,
            vocab,
            ctx_len: mu32_or(g, &k("context_length"), 0) as usize,
            rms_eps: mf32_or(g, &k("attention.layer_norm_rms_epsilon"), 1e-6),
            n_head: mu32(g, &k("attention.head_count"))? as usize,
            n_head_kv: mu32(g, &k("attention.head_count_kv"))? as usize,
            head_dim: mu32(g, &k("attention.key_length"))? as usize,
            rope_freq_base: mf32_or(g, &k("rope.freq_base"), 1e7),
            n_rot: mu32_or(g, &k("rope.dimension_count"), 0) as usize,
            rope_sections,
            full_attn_interval: mu32_or(g, &k("full_attention_interval"), 4) as usize,
            n_expert: mu32_or(g, &k("expert_count"), 0) as usize,
            n_expert_used: mu32_or(g, &k("expert_used_count"), 0) as usize,
            expert_ff: mu32_or(g, &k("expert_feed_forward_length"), 0) as usize,
            shared_ff: mu32_or(g, &k("expert_shared_feed_forward_length"), 0) as usize,
            dense_ff: mu32_or(g, &k("feed_forward_length"), 0) as usize,
            ssm_d_conv: mu32(g, &k("ssm.conv_kernel"))? as usize,
            ssm_d_inner: mu32(g, &k("ssm.inner_size"))? as usize,
            ssm_d_state: mu32(g, &k("ssm.state_size"))? as usize,
            ssm_dt_rank: mu32(g, &k("ssm.time_step_rank"))? as usize,
            ssm_n_group: mu32(g, &k("ssm.group_count"))? as usize,
        })
    }

    /// True if layer `il` is a recurrent (Gated-DeltaNet) layer; false = full attn.
    /// Matches llama.cpp: `(il+1) % full_attn_interval != 0`.
    pub fn is_recr(&self, il: usize) -> bool {
        self.full_attn_interval == 0 || (il + 1) % self.full_attn_interval != 0
    }

    /// True for the MoE arch (`qwen35moe`); false for the dense `qwen35`.
    pub fn is_moe(&self) -> bool {
        self.n_expert > 0
    }

    pub fn report(&self) -> String {
        let n_attn = (0..self.n_layer).filter(|&l| !self.is_recr(l)).count();
        let ffn = if self.is_moe() {
            format!(
                "MoE: {} experts top-{}, expert_ff={}, shared_ff={}",
                self.n_expert, self.n_expert_used, self.expert_ff, self.shared_ff
            )
        } else {
            format!("dense FFN: ff={}", self.dense_ff)
        };
        format!(
            "{}: {} layers ({} recurrent / {} full-attn), hidden={}, vocab={}, ctx={}\n  \
             attn: head_dim={} n_head={} n_head_kv={} (GQA {}:1), QK-norm, IMRoPE n_rot={} sections={:?} freq_base={:.0}\n  \
             {}\n  \
             SSM(GatedDeltaNet): d_conv={} d_inner={} d_state={} v_heads(dt_rank)={} k_heads(n_group)={}, rms_eps={:.1e}",
            if self.is_moe() { "qwen35moe" } else { "qwen35" },
            self.n_layer, self.n_layer - n_attn, n_attn, self.hidden, self.vocab, self.ctx_len,
            self.head_dim, self.n_head, self.n_head_kv, self.n_head / self.n_head_kv.max(1),
            self.n_rot, self.rope_sections, self.rope_freq_base,
            ffn,
            self.ssm_d_conv, self.ssm_d_inner, self.ssm_d_state, self.ssm_dt_rank, self.ssm_n_group, self.rms_eps,
        )
    }
}

/// P0 validation: parse the config and verify every expected tensor exists with the
/// right shape (per layer type). Returns the cfg + a human report; errors list the
/// first few missing/mismatched tensors.
pub fn p0_validate(g: &GgufFile) -> Result<(Qwen35Cfg, String)> {
    let cfg = Qwen35Cfg::from_gguf(g)?;
    let t = g.tensors();
    let mut missing: Vec<String> = Vec::new();
    let mut checked = 0usize;

    let key_dim = cfg.ssm_d_state * cfg.ssm_n_group; // 128*16 = 2048
    let value_dim = cfg.ssm_d_inner; // 4096 (= head_v_dim * n_v_heads)
    let head_v_dim = cfg.ssm_d_inner / cfg.ssm_dt_rank; // 128
    let q_out = cfg.head_dim * cfg.n_head * 2; // q includes gate: 8192
    let kv_out = cfg.head_dim * cfg.n_head_kv; // 512
    let attn_out_in = cfg.head_dim * cfg.n_head; // 4096

    let mut want = |name: String, dims: &[usize]| {
        checked += 1;
        match t.get(&name) {
            None => missing.push(format!("{name} (MISSING)")),
            Some(ti) => {
                let got: Vec<usize> = ti.dims.iter().map(|&d| d as usize).collect();
                if got != dims {
                    missing.push(format!("{name} shape {got:?} != expected {dims:?}"));
                }
            }
        }
    };

    want("token_embd.weight".into(), &[cfg.hidden, cfg.vocab]);
    want("output_norm.weight".into(), &[cfg.hidden]);
    // output.weight may be tied (absent) — check separately, not required
    for il in 0..cfg.n_layer {
        let b = |s: &str| format!("blk.{il}.{s}");
        want(b("attn_norm.weight"), &[cfg.hidden]);
        want(b("post_attention_norm.weight"), &[cfg.hidden]);
        if cfg.is_recr(il) {
            // Gated-DeltaNet linear-attn tensors
            want(b("attn_qkv.weight"), &[cfg.hidden, key_dim * 2 + value_dim]);
            want(b("attn_gate.weight"), &[cfg.hidden, value_dim]);
            want(
                b("ssm_conv1d.weight"),
                &[cfg.ssm_d_conv, key_dim * 2 + value_dim],
            );
            want(b("ssm_a"), &[cfg.ssm_dt_rank]);
            want(b("ssm_dt.bias"), &[cfg.ssm_dt_rank]);
            want(b("ssm_alpha.weight"), &[cfg.hidden, cfg.ssm_dt_rank]);
            want(b("ssm_beta.weight"), &[cfg.hidden, cfg.ssm_dt_rank]);
            want(b("ssm_norm.weight"), &[head_v_dim]);
            want(b("ssm_out.weight"), &[value_dim, cfg.hidden]);
        } else {
            // full GQA attention tensors
            want(b("attn_q.weight"), &[cfg.hidden, q_out]);
            want(b("attn_k.weight"), &[cfg.hidden, kv_out]);
            want(b("attn_v.weight"), &[cfg.hidden, kv_out]);
            want(b("attn_q_norm.weight"), &[cfg.head_dim]);
            want(b("attn_k_norm.weight"), &[cfg.head_dim]);
            want(b("attn_output.weight"), &[attn_out_in, cfg.hidden]);
        }
        if cfg.is_moe() {
            // MoE FFN (every layer): routed experts + sigmoid-gated shared expert.
            want(b("ffn_gate_inp.weight"), &[cfg.hidden, cfg.n_expert]);
            want(
                b("ffn_gate_exps.weight"),
                &[cfg.hidden, cfg.expert_ff, cfg.n_expert],
            );
            want(
                b("ffn_up_exps.weight"),
                &[cfg.hidden, cfg.expert_ff, cfg.n_expert],
            );
            want(
                b("ffn_down_exps.weight"),
                &[cfg.expert_ff, cfg.hidden, cfg.n_expert],
            );
            want(b("ffn_gate_inp_shexp.weight"), &[cfg.hidden]);
            want(b("ffn_gate_shexp.weight"), &[cfg.hidden, cfg.shared_ff]);
            want(b("ffn_up_shexp.weight"), &[cfg.hidden, cfg.shared_ff]);
            want(b("ffn_down_shexp.weight"), &[cfg.shared_ff, cfg.hidden]);
        } else {
            // Dense SwiGLU FFN (`qwen35`).
            want(b("ffn_gate.weight"), &[cfg.hidden, cfg.dense_ff]);
            want(b("ffn_up.weight"), &[cfg.hidden, cfg.dense_ff]);
            want(b("ffn_down.weight"), &[cfg.dense_ff, cfg.hidden]);
        }
    }

    let tied = !t.contains_key("output.weight");
    let report = format!(
        "{}\n  tensors: {} checked, {} missing/mismatched; lm_head {}",
        cfg.report(),
        checked,
        missing.len(),
        if tied {
            "tied to token_embd"
        } else {
            "separate output.weight"
        },
    );
    if missing.is_empty() {
        Ok((cfg, report))
    } else {
        let head: Vec<_> = missing.iter().take(8).cloned().collect();
        Err(StrixError::invalid(format!(
            "{report}\n  FIRST ISSUES:\n    {}",
            head.join("\n    ")
        )))
    }
}

// ===================== P1-P4: CPU reference forward =====================
// Token-at-a-time forward (the Gated-DeltaNet recurrence is naturally sequential;
// full-attn layers use a KV cache). Weights are dequantized on the fly per matmul
// (a 35B model can't fit as f32). Correctness over speed — this is the reference.

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
fn softplus(x: f32) -> f32 {
    // numerically stable
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
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

/// L2 norm over a slice (ggml_l2_norm): x / sqrt(max(sum x^2, eps)).
fn l2norm(v: &mut [f32], eps: f32) {
    let ss: f32 = v.iter().map(|x| x * x).sum::<f32>();
    let r = 1.0 / ss.max(eps).sqrt();
    for x in v.iter_mut() {
        *x *= r;
    }
}

/// Partial NEOX RoPE on the first `n_rot` dims of a head vector (text-only: IMRoPE
/// sections collapse to standard rope since all mrope position components == pos).
fn partial_rope(vec: &mut [f32], pos: usize, freq_base: f32, n_rot: usize) {
    let half = n_rot / 2;
    for i in 0..half {
        let freq = (freq_base as f64).powf(-2.0 * i as f64 / n_rot as f64) as f32;
        let ang = pos as f32 * freq;
        let (s, c) = ang.sin_cos();
        let a = vec[i];
        let b = vec[i + half];
        vec[i] = a * c - b * s;
        vec[i + half] = a * s + b * c;
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

/// Batched on-the-fly dequant matmul: `out[t][o] = W·xs[t]` for m tokens (each weight
/// row dequantized ONCE per chunk). Same per-row `dot_f32` as `qmatmul` → identical.
fn qmatmul_batch(
    out: &mut [f32],
    xs: &[f32],
    m: usize,
    bytes: &[u8],
    ty: GgmlType,
    in_dim: usize,
    out_dim: usize,
) {
    let bpr = (in_dim / ty.block_elems()) * ty.block_bytes();
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

/// Shareable raw f32 pointer for disjoint-index parallel writes.
struct SendPtrF(*mut f32);
unsafe impl Sync for SendPtrF {}
unsafe impl Send for SendPtrF {}

/// A resolved weight: raw bytes + ggml type + the row length (in_dim).
struct W<'a> {
    bytes: &'a [u8],
    ty: GgmlType,
    in_dim: usize,
}

pub struct Qwen35Model {
    gguf: GgufFile,
    cfg: Qwen35Cfg,
    // full-attn layers: KV cache, per layer flat [pos * kv_dim .. ] (kv_dim = n_head_kv*head_dim)
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    // deltanet layers: recurrent state per layer [n_v_heads * S_v * S_v] (transposed S)
    ssm: Vec<Vec<f32>>,
    // deltanet layers: conv rolling state per layer [conv_dim * (d_conv-1)]
    conv: Vec<Vec<f32>>,
    pos: usize,
    // Optional iGPU weight accelerator (per-weight GEMV by GGUF tensor name). When
    // `None` (default / CPU build), every matmul takes the CPU path → behaviour is
    // byte-identical to the pure-CPU forward. See docs/ideas/moe-accel-plan.md (P1).
    accel: Option<Box<dyn strix_core::WeightAccel>>,
    // Resident GPU decode active (qwen35 DeltaNet+attn forward runs on-device, 1
    // sync/token). Set by `enable_gpu_decode`; prefill stays on CPU/NPU + seeds device state.
    gpu_decode: bool,
    // Optional NPU offload of the dense projections during batched prefill
    // (the 256-expert MoE ~30GB int8 exceeds the BO pool → experts stay CPU).
    #[cfg(feature = "npu")]
    npu: Option<crate::mellum_npu::QwenNpu>,
}

impl Qwen35Model {
    pub fn from_gguf(gguf: GgufFile) -> Result<Self> {
        let cfg = Qwen35Cfg::from_gguf(&gguf)?;
        let nl = cfg.n_layer;
        let s_v = cfg.ssm_d_inner / cfg.ssm_dt_rank; // head_v_dim = 128
        let conv_dim = cfg.ssm_d_state * cfg.ssm_n_group * 2 + cfg.ssm_d_inner; // 8192
        Ok(Qwen35Model {
            kc: vec![Vec::new(); nl],
            vc: vec![Vec::new(); nl],
            ssm: (0..nl)
                .map(|l| {
                    if cfg.is_recr(l) {
                        vec![0.0f32; cfg.ssm_dt_rank * s_v * s_v]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            conv: (0..nl)
                .map(|l| {
                    if cfg.is_recr(l) {
                        vec![0.0f32; conv_dim * (cfg.ssm_d_conv - 1)]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            cfg,
            gguf,
            pos: 0,
            accel: None,
            gpu_decode: false,
            #[cfg(feature = "npu")]
            npu: None,
        })
    }

    /// Stage the dense projections (deltanet qkv/gate/ssm_out + full-attn q/o, all
    /// Q8_0 in the UD quant) onto the NPU. Returns #staged.
    #[cfg(feature = "npu")]
    pub fn attach_npu(&mut self, mut npu: crate::mellum_npu::QwenNpu) -> Result<usize> {
        let mut n = 0usize;
        let g = &self.gguf;
        let mut stage = |sh: &mut crate::mellum_npu::NpuShape, name: String, slot: u64| {
            if let (Some(ti), Ok(bytes)) = (g.tensors().get(&name), g.tensor_bytes(&name)) {
                if sh.stage_q8(slot, bytes, ti.ggml_type).is_ok() {
                    return 1;
                }
            }
            0
        };
        for il in 0..self.cfg.n_layer {
            if self.cfg.is_recr(il) {
                n += stage(
                    &mut npu.p8192,
                    format!("blk.{il}.attn_qkv.weight"),
                    il as u64,
                );
                n += stage(
                    &mut npu.p4096,
                    format!("blk.{il}.attn_gate.weight"),
                    il as u64,
                );
                n += stage(
                    &mut npu.p2048,
                    format!("blk.{il}.ssm_out.weight"),
                    il as u64,
                );
            } else {
                n += stage(&mut npu.p8192, format!("blk.{il}.attn_q.weight"), il as u64);
                n += stage(
                    &mut npu.p2048,
                    format!("blk.{il}.attn_output.weight"),
                    il as u64,
                );
            }
        }
        self.npu = Some(npu);
        Ok(n)
    }

    /// Dequantize the embedding row for `token` → `[hidden]` (CPU; the resident
    /// GPU decode takes this as its per-token input `h`).
    fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        let hidden = self.cfg.hidden;
        let emb = self.w("token_embd.weight")?;
        let bpr = (hidden / emb.ty.block_elems()) * emb.ty.block_bytes();
        let mut h = vec![0.0f32; hidden];
        dequantize_into(
            emb.ty,
            &emb.bytes[token as usize * bpr..token as usize * bpr + bpr],
            &mut h,
        )?;
        Ok(h)
    }

    /// Enable the resident on-device decode (DeltaNet+attn forward on the iGPU, 1
    /// sync/token). Requires an attached accelerator with the Q8_0 weights resident.
    /// Uploads the f32 norm / SSM-parameter weights, then configures the device
    /// scratch + per-layer KV / SSM / conv state buffers (`max_seq` = prompt+gen).
    /// Returns true if the backend accepted the config. Prefill still runs on
    /// CPU/NPU and seeds the device state via `prefill`.
    pub fn enable_gpu_decode(&mut self, max_seq: usize) -> bool {
        // Resident path implements the dense `qwen35` FFN only (not the MoE arch).
        if self.accel.is_none() || self.cfg.is_moe() {
            return false;
        }
        let cfg = self.cfg.clone();
        // Collect the f32 weights (dequantized) before borrowing the accel mutably.
        let mut f32s: Vec<(String, Vec<f32>)> = Vec::new();
        let mut take = |s: &Self, key: String, out: &mut Vec<(String, Vec<f32>)>| {
            if let Ok(d) = s.vecw(&key) {
                out.push((key, d));
            }
        };
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            take(self, b("attn_norm.weight"), &mut f32s);
            take(self, b("post_attention_norm.weight"), &mut f32s);
            if cfg.is_recr(il) {
                take(self, b("ssm_conv1d.weight"), &mut f32s);
                take(self, b("ssm_a"), &mut f32s);
                take(self, b("ssm_dt.bias"), &mut f32s);
                take(self, b("ssm_norm.weight"), &mut f32s);
            } else {
                take(self, b("attn_q_norm.weight"), &mut f32s);
                take(self, b("attn_k_norm.weight"), &mut f32s);
            }
        }
        take(self, "output_norm.weight".into(), &mut f32s);

        let s_v = cfg.ssm_d_inner / cfg.ssm_dt_rank;
        let n_rot = if cfg.n_rot > 0 {
            cfg.n_rot
        } else {
            cfg.head_dim
        };
        let gcfg = Q35GpuConfig {
            n_layer: cfg.n_layer,
            hidden: cfg.hidden,
            vocab: cfg.vocab,
            eps: cfg.rms_eps,
            full_attn_interval: cfg.full_attn_interval,
            n_head: cfg.n_head,
            n_head_kv: cfg.n_head_kv,
            head_dim: cfg.head_dim,
            n_rot,
            rope_theta: cfg.rope_freq_base,
            n_vh: cfg.ssm_dt_rank,
            n_kh: cfg.ssm_n_group,
            s_v,
            dconv: cfg.ssm_d_conv,
            dense_ff: cfg.dense_ff,
            max_seq,
        };
        let accel = self.accel.as_mut().unwrap();
        for (k, d) in &f32s {
            accel.upload_f32(k, d);
        }
        let ok = accel.configure_decode_qwen35(gcfg);
        self.gpu_decode = ok;
        ok
    }

    /// iGPU dense GEMM for batched prefill (resident Q8 weight, chunks of 256).
    fn gpu_gemm(
        &self,
        key: &str,
        xs: &[f32],
        m: usize,
        in_dim: usize,
        out_dim: usize,
        out: &mut [f32],
    ) -> bool {
        let Some(a) = &self.accel else { return false };
        for c in (0..m).step_by(256) {
            let mc = (m - c).min(256);
            match a.prefill_q8_gemm(key, &xs[c * in_dim..(c + mc) * in_dim], mc) {
                Some(y) if y.len() == mc * out_dim => {
                    out[c * out_dim..(c + mc) * out_dim].copy_from_slice(&y);
                }
                _ => return false,
            }
        }
        true
    }

    /// NPU dense projection for layer `il` during batched prefill (`which`: 0 = the
    /// 2048→8192 shape, 1 = 2048→4096, 2 = 4096→2048), chunked to M=256.
    #[cfg(feature = "npu")]
    fn npu_proj(
        &self,
        which: u8,
        il: usize,
        xs: &[f32],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> bool {
        let Some(npu) = &self.npu else { return false };
        if m < crate::mellum_npu::M_MIN {
            return false;
        }
        let sh = match which {
            0 => &npu.p8192,
            1 => &npu.p4096,
            _ => &npu.p2048,
        };
        if sh.k != k || sh.n != n || !sh.has(il as u64) {
            return false;
        }
        for c in (0..m).step_by(crate::mellum_npu::M_NPU) {
            let mc = (m - c).min(crate::mellum_npu::M_NPU);
            if sh
                .gemm(
                    il as u64,
                    &xs[c * k..(c + mc) * k],
                    mc,
                    &mut out[c * n..(c + mc) * n],
                )
                .is_err()
            {
                return false;
            }
        }
        true
    }
    #[cfg(not(feature = "npu"))]
    #[allow(clippy::too_many_arguments)]
    fn npu_proj(
        &self,
        _w: u8,
        _il: usize,
        _xs: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
        _o: &mut [f32],
    ) -> bool {
        false
    }

    /// Attach an iGPU weight accelerator and upload the dense Q6_K/Q4_0 projection
    /// weights (full-attn q/k/v/o + lm_head) resident by GGUF tensor name. Returns
    /// the count adopted. MoE experts + deltanet projections stay on CPU for now (P2).
    /// NOTE: the ROCm backend's `gemv` is a no-op (decode_step-only) — use a Vulkan
    /// accel (`GpuWeightAccel`/`AshWeightAccel`), which implements per-weight gemv.
    pub fn attach_accel(&mut self, mut accel: Box<dyn strix_core::WeightAccel>) -> usize {
        let mut names: Vec<String> = Vec::new();
        for il in 0..self.cfg.n_layer {
            if !self.cfg.is_recr(il) {
                for t in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                    names.push(format!("blk.{il}.{t}.weight"));
                }
            } else {
                for t in ["attn_qkv", "attn_gate", "ssm_out", "ssm_beta", "ssm_alpha"] {
                    names.push(format!("blk.{il}.{t}.weight"));
                }
            }
            if self.cfg.is_moe() {
                for t in [
                    "ffn_gate_shexp",
                    "ffn_up_shexp",
                    "ffn_down_shexp",
                    "ffn_gate_inp_shexp",
                ] {
                    names.push(format!("blk.{il}.{t}.weight"));
                }
            } else {
                // Dense FFN (`qwen35`): the full FFN goes resident on the iGPU.
                for t in ["ffn_gate", "ffn_up", "ffn_down"] {
                    names.push(format!("blk.{il}.{t}.weight"));
                }
            }
        }
        if self.gguf.tensors().contains_key("output.weight") {
            names.push("output.weight".to_string());
        } else {
            // Tied lm_head: upload the embedding matrix so the (large vocab×hidden)
            // head GEMV runs on-device instead of CPU. The embedding *lookup* stays
            // CPU (one row); this resident copy is used only by the head `mm`.
            names.push("token_embd.weight".to_string());
        }
        let mut n = 0usize;
        // Returns true iff the weight was adopted by the accelerator.
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
        // STRIX_Q35_Q4: repack the resident matmul weights Q8_0 → Q4_0 on upload to
        // HALVE the decode weight-bandwidth (~2× the bandwidth-bound resident decode),
        // at a precision cost (lossy — greedy tokens may differ from the Q8 reference).
        // Opt-in; only meaningful for the dense qwen35 resident path.
        let q4_repack = std::env::var("STRIX_Q35_Q4").is_ok() && !self.cfg.is_moe();
        for name in &names {
            let Some(ti) = self.gguf.tensors().get(name) else {
                continue;
            };
            let (ty, in_dim) = (ti.ggml_type, ti.dims[0] as usize);
            let out_dim: usize = ti.dims[1..].iter().map(|&d| d as usize).product();
            if let Ok(bytes) = self.gguf.tensor_bytes(name) {
                let adopted = if q4_repack && ty == GgmlType::Q8_0 {
                    match dequantize(ty, bytes, in_dim * out_dim) {
                        Ok(f) => accel.upload_q4_0(name, &quantize_q4_0(&f), in_dim, out_dim),
                        Err(_) => up(&mut accel, name, bytes, ty, in_dim, out_dim),
                    }
                } else {
                    up(&mut accel, name, bytes, ty, in_dim, out_dim)
                };
                if adopted {
                    n += 1;
                }
            }
        }
        // MoE experts (P2): the bulk of the model. Each 3D `ffn_*_exps` tensor is
        // [in, ff, n_expert]; expert e is a contiguous byte slice = a 2D [out,in]
        // weight. Upload each Q6_K expert slice under `<tensor>.e{e}` so the MoE loop
        // can gemv it by key. (Layers whose exps tensor is Q8_0 fall through to CPU.)
        let hidden = self.cfg.hidden;
        let eff = self.cfg.expert_ff;
        let ne = self.cfg.n_expert;
        let layer_cap = std::env::var("STRIX_GPU_EXPERT_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(self.cfg.n_layer);
        let mut advise: Vec<String> = Vec::new();
        // Whole-layer NATIVE upload (Q6_K/Q8_0, no repack inflation -> the 35B fits)
        // for the fused moe_ffn decode path.
        for il in 0..self.cfg.n_layer.min(layer_cap) {
            let gname = format!("blk.{il}.ffn_gate_exps.weight");
            let uname = format!("blk.{il}.ffn_up_exps.weight");
            let dname = format!("blk.{il}.ffn_down_exps.weight");
            let (Some(gt), Some(dt)) = (
                self.gguf.tensors().get(&gname),
                self.gguf.tensors().get(&dname),
            ) else {
                continue;
            };
            let (gty, dty) = (gt.ggml_type, dt.ggml_type);
            let (Ok(gb), Ok(ub), Ok(db)) = (
                self.gguf.tensor_bytes(&gname),
                self.gguf.tensor_bytes(&uname),
                self.gguf.tensor_bytes(&dname),
            ) else {
                continue;
            };
            let ok = match (gty, dty) {
                (GgmlType::Q6K, GgmlType::Q6K) => {
                    accel.upload_moe_q6(il, gb, ub, db, hidden, eff, ne)
                }
                (GgmlType::Q8_0, GgmlType::Q8_0) => {
                    accel.upload_moe_q8(il, gb, ub, db, hidden, eff, ne)
                }
                _ => false,
            };
            if ok {
                n += 1;
                advise.push(gname);
                advise.push(uname);
                advise.push(dname);
            }
        }
        for name in &advise {
            self.gguf.advise_dontneed(name);
        }
        self.accel = Some(accel);
        n
    }

    /// Try the GPU gemv for `key` into `out`; returns true if the accelerator adopted
    /// the weight and produced a correctly-sized result. False ⇒ caller does the CPU path.
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

    /// Matmul `out = W·x` for weight `key`: GPU gemv if the accelerator adopted it,
    /// else the CPU path. With no accelerator this is exactly `qmatmul`.
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
            .ok_or_else(|| StrixError::invalid(format!("qwen35: missing tensor {name}")))?;
        let in_dim = ti.dims[0] as usize;
        Ok(W {
            bytes: self.gguf.tensor_bytes(name)?,
            ty: ti.ggml_type,
            in_dim,
        })
    }

    /// Dequantize a small F32/quant vector tensor fully (norms, ssm_a, ssm_dt).
    fn vecw(&self, name: &str) -> Result<Vec<f32>> {
        let ti = self
            .gguf
            .tensors()
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("qwen35: missing tensor {name}")))?;
        let n: usize = ti.dims.iter().map(|&d| d as usize).product();
        let mut out = vec![0.0f32; n];
        dequantize_into(ti.ggml_type, self.gguf.tensor_bytes(name)?, &mut out)?;
        Ok(out)
    }

    /// Forward one token at position `self.pos`; returns logits iff `want_logits`.
    /// Batched prefill: same weight-read-once strategy as the Mellum forward —
    /// projections/experts batched over all m tokens; the Gated-DeltaNet conv+scan
    /// stays sequential per token (recurrent in time). Bit-identical per-token math.
    fn prefill_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let m = tokens.len();
        let hidden = cfg.hidden;
        let eps = cfg.rms_eps;
        let mut row = vec![0.0f32; 16384];

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

        let prof = std::env::var("STRIX_PREFILL_PROF").is_ok();
        let (mut t_mix, mut t_moe) = (0.0f64, 0.0f64);
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            let an = self.vecw(&b("attn_norm.weight"))?;
            for t in 0..m {
                let (hs, ns) = (
                    &h[t * hidden..(t + 1) * hidden],
                    &mut n[t * hidden..(t + 1) * hidden],
                );
                rmsnorm(ns, hs, &an, eps);
            }
            let t0 = std::time::Instant::now();
            if cfg.is_recr(il) {
                self.deltanet_batch(&n, m, il, &mut h)?;
            } else {
                self.attn_batch(&n, m, il, &mut h)?;
            }
            t_mix += t0.elapsed().as_secs_f64();
            let pn = self.vecw(&b("post_attention_norm.weight"))?;
            for t in 0..m {
                let (hs, ns) = (
                    &h[t * hidden..(t + 1) * hidden],
                    &mut n[t * hidden..(t + 1) * hidden],
                );
                rmsnorm(ns, hs, &pn, eps);
            }
            let t1 = std::time::Instant::now();
            if cfg.is_moe() {
                self.moe_batch(&n, m, il, &mut h)?;
            } else {
                self.dense_ffn_batch(&n, m, il, &mut h)?;
            }
            t_moe += t1.elapsed().as_secs_f64();
        }
        if prof {
            eprintln!(
                "[prefill prof] mixer {t_mix:.2}s (dn-gemm {:.2}s scan {:.2}s sdpa {:.2}s) | moe {t_moe:.2}s",
                PROF_GEMM.load(Ordering::Relaxed) as f64 / 1e6,
                PROF_SCAN.load(Ordering::Relaxed) as f64 / 1e6,
                PROF_SDPA.load(Ordering::Relaxed) as f64 / 1e6,
            );
        }
        self.pos += m;

        let on = self.vecw("output_norm.weight")?;
        let mut nh = vec![0.0f32; hidden];
        rmsnorm(&mut nh, &h[(m - 1) * hidden..m * hidden], &on, eps);
        let (head_key, head) = if self.gguf.tensors().contains_key("output.weight") {
            ("output.weight", self.w("output.weight")?)
        } else {
            ("token_embd.weight", self.w("token_embd.weight")?)
        };
        let mut logits = vec![0.0f32; cfg.vocab];
        self.mm(head_key, &mut logits, &nh, &head, &mut row);
        Ok(logits)
    }

    /// Batched full-attention layer: q/k/v/o batched; per-token QK-norm+rope+SDPA.
    /// Adds the projection output to `h` rows (residual).
    fn attn_batch(&mut self, x: &[f32], m: usize, il: usize, h: &mut [f32]) -> Result<()> {
        let cfg = self.cfg.clone();
        let hd = cfg.head_dim;
        let nh = cfg.n_head;
        let nkv = cfg.n_head_kv;
        let groups = nh / nkv;
        let kv_dim = nkv * hd;
        let q_dim = nh * hd;
        let hidden = cfg.hidden;
        let b = |s: &str| format!("blk.{il}.{s}");
        let qn = self.vecw(&b("attn_q_norm.weight"))?;
        let kn = self.vecw(&b("attn_k_norm.weight"))?;
        let qg_dim = hd * 2 * nh;
        let mut qg = vec![0.0f32; m * qg_dim];
        let mut k = vec![0.0f32; m * kv_dim];
        let mut v = vec![0.0f32; m * kv_dim];
        {
            if !self.gpu_gemm(&b("attn_q.weight"), x, m, hidden, qg_dim, &mut qg)
                && !self.npu_proj(0, il, x, m, hidden, qg_dim, &mut qg)
            {
                let wq = self.w(&b("attn_q.weight"))?;
                qmatmul_batch(&mut qg, x, m, wq.bytes, wq.ty, hidden, qg_dim);
            }
            let wk = self.w(&b("attn_k.weight"))?;
            qmatmul_batch(&mut k, x, m, wk.bytes, wk.ty, hidden, kv_dim);
            let wv = self.w(&b("attn_v.weight"))?;
            qmatmul_batch(&mut v, x, m, wv.bytes, wv.ty, hidden, kv_dim);
        }
        let mut q = vec![0.0f32; m * q_dim];
        let mut gate = vec![0.0f32; m * q_dim];
        for t in 0..m {
            let pos = self.pos + t;
            for hh in 0..nh {
                let base = t * qg_dim + hh * hd * 2;
                let mut qh = qg[base..base + hd].to_vec();
                let mut tmp = vec![0.0f32; hd];
                rmsnorm(&mut tmp, &qh, &qn, cfg.rms_eps);
                qh.copy_from_slice(&tmp);
                partial_rope(&mut qh, pos, cfg.rope_freq_base, cfg.n_rot);
                q[t * q_dim + hh * hd..t * q_dim + hh * hd + hd].copy_from_slice(&qh);
                gate[t * q_dim + hh * hd..t * q_dim + hh * hd + hd]
                    .copy_from_slice(&qg[base + hd..base + 2 * hd]);
            }
            for kh in 0..nkv {
                let kb = t * kv_dim + kh * hd;
                let mut khv = k[kb..kb + hd].to_vec();
                let mut tmp = vec![0.0f32; hd];
                rmsnorm(&mut tmp, &khv, &kn, cfg.rms_eps);
                khv.copy_from_slice(&tmp);
                partial_rope(&mut khv, pos, cfg.rope_freq_base, cfg.n_rot);
                k[kb..kb + hd].copy_from_slice(&khv);
            }
        }
        self.kc[il].extend_from_slice(&k[..m * kv_dim]);
        self.vc[il].extend_from_slice(&v[..m * kv_dim]);
        let kc = &self.kc[il];
        let vc = &self.vc[il];
        let base_pos = self.pos;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn_out = vec![0.0f32; m * q_dim];
        let tsd = std::time::Instant::now();
        attn_out
            .par_chunks_mut(q_dim)
            .enumerate()
            .for_each(|(t, ao)| {
                let len = base_pos + t + 1;
                let mut keys = vec![0.0f32; len * hd];
                let mut vals = vec![0.0f32; len * hd];
                let mut scratch = vec![0.0f32; len];
                for hh in 0..nh {
                    let kvh = hh / groups;
                    for tt in 0..len {
                        keys[tt * hd..tt * hd + hd].copy_from_slice(
                            &kc[tt * kv_dim + kvh * hd..tt * kv_dim + kvh * hd + hd],
                        );
                        vals[tt * hd..tt * hd + hd].copy_from_slice(
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
                        len,
                        scale,
                        &mut scratch,
                    );
                    for d in 0..hd {
                        oh[d] *= sigmoid(gate[t * q_dim + hh * hd + d]);
                    }
                    ao[hh * hd..hh * hd + hd].copy_from_slice(&oh);
                }
            });
        let mut o = vec![0.0f32; m * hidden];
        if !self.gpu_gemm(
            &b("attn_output.weight"),
            &attn_out,
            m,
            q_dim,
            hidden,
            &mut o,
        ) && !self.npu_proj(2, il, &attn_out, m, q_dim, hidden, &mut o)
        {
            let wo = self.w(&b("attn_output.weight"))?;
            qmatmul_batch(&mut o, &attn_out, m, wo.bytes, wo.ty, q_dim, hidden);
        }
        for i in 0..m * hidden {
            h[i] += o[i];
        }
        Ok(())
    }

    /// Batched Gated-DeltaNet layer: qkv/gate/alpha/beta projections batched; the
    /// causal conv + delta-rule scan run sequentially per token (time recurrence).
    fn deltanet_batch(&mut self, x: &[f32], m: usize, il: usize, h: &mut [f32]) -> Result<()> {
        let cfg = self.cfg.clone();
        let s_v = cfg.ssm_d_inner / cfg.ssm_dt_rank;
        let n_vh = cfg.ssm_dt_rank;
        let n_kh = cfg.ssm_n_group;
        let key_dim = cfg.ssm_d_state * cfg.ssm_n_group;
        let value_dim = cfg.ssm_d_inner;
        let conv_dim = key_dim * 2 + value_dim;
        let dconv = cfg.ssm_d_conv;
        let hidden = cfg.hidden;
        let b = |s: &str| format!("blk.{il}.{s}");

        let ssm_a = self.vecw(&b("ssm_a"))?;
        let ssm_dt = self.vecw(&b("ssm_dt.bias"))?;
        let ssm_norm = self.vecw(&b("ssm_norm.weight"))?;
        let conv_w = self.vecw(&b("ssm_conv1d.weight"))?;

        let mut qkv = vec![0.0f32; m * conv_dim];
        let mut z = vec![0.0f32; m * value_dim];
        let mut beta_raw = vec![0.0f32; m * n_vh];
        let mut alpha_raw = vec![0.0f32; m * n_vh];
        {
            let tp = std::time::Instant::now();
            if !self.gpu_gemm(&b("attn_qkv.weight"), x, m, hidden, conv_dim, &mut qkv)
                && !self.npu_proj(0, il, x, m, hidden, conv_dim, &mut qkv)
            {
                let wqkv = self.w(&b("attn_qkv.weight"))?;
                qmatmul_batch(&mut qkv, x, m, wqkv.bytes, wqkv.ty, hidden, conv_dim);
            }
            if !self.gpu_gemm(&b("attn_gate.weight"), x, m, hidden, value_dim, &mut z)
                && !self.npu_proj(1, il, x, m, hidden, value_dim, &mut z)
            {
                let wgate = self.w(&b("attn_gate.weight"))?;
                qmatmul_batch(&mut z, x, m, wgate.bytes, wgate.ty, hidden, value_dim);
            }
            let wbeta = self.w(&b("ssm_beta.weight"))?;
            qmatmul_batch(&mut beta_raw, x, m, wbeta.bytes, wbeta.ty, hidden, n_vh);
            let walpha = self.w(&b("ssm_alpha.weight"))?;
            qmatmul_batch(&mut alpha_raw, x, m, walpha.bytes, walpha.ty, hidden, n_vh);
            padd(&PROF_GEMM, tp);
        }

        let tscan = std::time::Instant::now();
        // Phase A: conv for ALL tokens — per channel serial over t, parallel channels.
        let mut conv_all = vec![0.0f32; m * conv_dim];
        {
            let cs = &mut self.conv[il];
            let cap = SendPtrF(conv_all.as_mut_ptr());
            let cap = &cap;
            cs.par_chunks_mut(dconv - 1)
                .enumerate()
                .for_each(|(c, csb)| {
                    for t in 0..m {
                        let xv = qkv[t * conv_dim + c];
                        let mut acc = 0.0f32;
                        for kk in 0..dconv - 1 {
                            acc += conv_w[c * dconv + kk] * csb[kk];
                        }
                        acc += conv_w[c * dconv + (dconv - 1)] * xv;
                        unsafe { *cap.0.add(t * conv_dim + c) = silu(acc) };
                        for kk in 0..dconv - 2 {
                            csb[kk] = csb[kk + 1];
                        }
                        csb[dconv - 2] = xv;
                    }
                });
        }
        // Phase B: per-token q/k L2 norms (parallel over tokens).
        let mut qn_all = vec![0.0f32; m * key_dim];
        let mut kn_all = vec![0.0f32; m * key_dim];
        qn_all
            .par_chunks_mut(key_dim)
            .zip(kn_all.par_chunks_mut(key_dim))
            .enumerate()
            .for_each(|(t, (qn, kn))| {
                qn.copy_from_slice(&conv_all[t * conv_dim..t * conv_dim + key_dim]);
                kn.copy_from_slice(&conv_all[t * conv_dim + key_dim..t * conv_dim + 2 * key_dim]);
                for kh in 0..n_kh {
                    l2norm(&mut qn[kh * s_v..kh * s_v + s_v], cfg.rms_eps);
                    l2norm(&mut kn[kh * s_v..kh * s_v + s_v], cfg.rms_eps);
                }
            });
        // Phase C: delta-rule scan — heads parallel, tokens serial per head.
        let scale = 1.0 / (s_v as f32).sqrt();
        let mut core_all = vec![0.0f32; m * value_dim];
        {
            let cap = SendPtrF(core_all.as_mut_ptr());
            let cap = &cap;
            self.ssm[il]
                .par_chunks_mut(s_v * s_v)
                .enumerate()
                .for_each(|(vh, st)| {
                    let kh = vh % n_kh;
                    let mut delta = vec![0.0f32; s_v];
                    for t in 0..m {
                        let qh = &qn_all[t * key_dim + kh * s_v..t * key_dim + kh * s_v + s_v];
                        let kk = &kn_all[t * key_dim + kh * s_v..t * key_dim + kh * s_v + s_v];
                        let vv = &conv_all[t * conv_dim + 2 * key_dim + vh * s_v
                            ..t * conv_dim + 2 * key_dim + vh * s_v + s_v];
                        let g = ssm_a[vh] * softplus(alpha_raw[t * n_vh + vh] + ssm_dt[vh]);
                        let betah = sigmoid(beta_raw[t * n_vh + vh]);
                        let decay = g.exp();
                        for xx in st.iter_mut() {
                            *xx *= decay;
                        }
                        for j in 0..s_v {
                            let rowj = &st[j * s_v..j * s_v + s_v];
                            let mut sum = 0.0f32;
                            for i in 0..s_v {
                                sum += rowj[i] * kk[i];
                            }
                            delta[j] = (vv[j] - sum) * betah;
                        }
                        for j in 0..s_v {
                            let dj = delta[j];
                            let rowj = &mut st[j * s_v..j * s_v + s_v];
                            for i in 0..s_v {
                                rowj[i] += dj * kk[i];
                            }
                        }
                        for j in 0..s_v {
                            let rowj = &st[j * s_v..j * s_v + s_v];
                            let mut sum = 0.0f32;
                            for i in 0..s_v {
                                sum += rowj[i] * qh[i];
                            }
                            unsafe { *cap.0.add(t * value_dim + vh * s_v + j) = sum * scale };
                        }
                    }
                });
        }
        // Phase D: gated norm (parallel over t,heads).
        let mut gated = vec![0.0f32; m * value_dim];
        gated
            .par_chunks_mut(s_v)
            .enumerate()
            .for_each(|(idx, gout)| {
                let t = idx / n_vh;
                let vh = idx % n_vh;
                let mut tmp = vec![0.0f32; s_v];
                rmsnorm(
                    &mut tmp,
                    &core_all[t * value_dim + vh * s_v..t * value_dim + vh * s_v + s_v],
                    &ssm_norm,
                    cfg.rms_eps,
                );
                for j in 0..s_v {
                    gout[j] = tmp[j] * silu(z[t * value_dim + vh * s_v + j]);
                }
            });
        padd(&PROF_SCAN, tscan);
        let mut o = vec![0.0f32; m * hidden];
        if !self.gpu_gemm(&b("ssm_out.weight"), &gated, m, value_dim, hidden, &mut o)
            && !self.npu_proj(2, il, &gated, m, value_dim, hidden, &mut o)
        {
            let wout = self.w(&b("ssm_out.weight"))?;
            qmatmul_batch(&mut o, &gated, m, wout.bytes, wout.ty, value_dim, hidden);
        }
        for i in 0..m * hidden {
            h[i] += o[i];
        }
        Ok(())
    }

    /// Batched MoE: route per token, group by expert, batch each expert + the shared
    /// expert; per-token accumulation matches moe()'s float association.
    fn moe_batch(&mut self, x: &[f32], m: usize, il: usize, h: &mut [f32]) -> Result<()> {
        let cfg = self.cfg.clone();
        let hidden = cfg.hidden;
        let ne = cfg.n_expert;
        let topk = cfg.n_expert_used;
        let eff = cfg.expert_ff;
        let b = |s: &str| format!("blk.{il}.{s}");

        let wgi = self.w(&b("ffn_gate_inp.weight"))?;
        let mut rl = vec![0.0f32; m * ne];
        qmatmul_batch(&mut rl, x, m, wgi.bytes, wgi.ty, hidden, ne);
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
            idx.sort_by(|&a, &bb| probs[bb].partial_cmp(&probs[a]).unwrap());
            idx.truncate(topk);
            let wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
            routes.push(idx.into_iter().map(|e| (e, probs[e] / wsum)).collect());
        }
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
        let mut dy = vec![0.0f32; m * topk * hidden];
        // GPU path: ONE multi-expert GEMM set per layer (3 launches + 1 sync).
        if let Some(a) = &self.accel {
            let rows: usize = by_exp.iter().map(|l| l.len()).sum();
            let mut xs_all = vec![0.0f32; rows * hidden];
            let mut plan: Vec<(i32, i32)> = Vec::new();
            let mut off = 0usize;
            for (e, list) in by_exp.iter().enumerate() {
                if list.is_empty() {
                    continue;
                }
                for &(t, _) in list {
                    xs_all[off * hidden..(off + 1) * hidden]
                        .copy_from_slice(&x[t * hidden..(t + 1) * hidden]);
                    off += 1;
                }
                plan.push((e as i32, list.len() as i32));
            }
            if let Some(d_all) = a.moe_layer_ffn(il, &plan, &xs_all, rows) {
                let mut o = 0usize;
                for list in by_exp.iter() {
                    for &(t, s) in list {
                        dy[(t * topk + s) * hidden..(t * topk + s + 1) * hidden]
                            .copy_from_slice(&d_all[o * hidden..(o + 1) * hidden]);
                        o += 1;
                    }
                }
                return self.moe_finish(x, m, il, h, &routes, &dy);
            }
        }
        for (e, list) in by_exp.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let me = list.len();
            let mut xs = vec![0.0f32; me * hidden];
            for (i, &(t, _)) in list.iter().enumerate() {
                xs[i * hidden..(i + 1) * hidden].copy_from_slice(&x[t * hidden..(t + 1) * hidden]);
            }
            let mut d = vec![0.0f32; me * hidden];
            let gpu_d = self
                .accel
                .as_ref()
                .and_then(|a| a.moe_expert_ffn(il, e, &xs, me))
                .filter(|y| y.len() == me * hidden);
            if let Some(y) = gpu_d {
                d.copy_from_slice(&y);
            } else {
                let mut g = vec![0.0f32; me * eff];
                let mut u = vec![0.0f32; me * eff];
                qmatmul_batch(
                    &mut g,
                    &xs,
                    me,
                    &wge.bytes[e * g_bpr..(e + 1) * g_bpr],
                    wge.ty,
                    hidden,
                    eff,
                );
                qmatmul_batch(
                    &mut u,
                    &xs,
                    me,
                    &wue.bytes[e * u_bpr..(e + 1) * u_bpr],
                    wue.ty,
                    hidden,
                    eff,
                );
                let mut act = vec![0.0f32; me * eff];
                for i in 0..me * eff {
                    act[i] = silu(g[i]) * u[i];
                }
                qmatmul_batch(
                    &mut d,
                    &act,
                    me,
                    &wde.bytes[e * d_bpr..(e + 1) * d_bpr],
                    wde.ty,
                    eff,
                    hidden,
                );
            }
            for (i, &(t, s)) in list.iter().enumerate() {
                dy[(t * topk + s) * hidden..(t * topk + s + 1) * hidden]
                    .copy_from_slice(&d[i * hidden..(i + 1) * hidden]);
            }
        }
        self.moe_finish(x, m, il, h, &routes, &dy)
    }

    /// Shared expert + per-token weighted accumulation (moe_batch tail).
    fn moe_finish(
        &self,
        x: &[f32],
        m: usize,
        il: usize,
        h: &mut [f32],
        routes: &[Vec<(usize, f32)>],
        dy: &[f32],
    ) -> Result<()> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let topk = cfg.n_expert_used;
        let b = |s: &str| format!("blk.{il}.{s}");
        let sff = cfg.shared_ff;
        let mut shared = vec![0.0f32; m * hidden];
        if sff > 0 {
            let wgs = self.w(&b("ffn_gate_shexp.weight"))?;
            let wus = self.w(&b("ffn_up_shexp.weight"))?;
            let wds = self.w(&b("ffn_down_shexp.weight"))?;
            let wgis = self.w(&b("ffn_gate_inp_shexp.weight"))?;
            let mut gs = vec![0.0f32; m * sff];
            let mut us = vec![0.0f32; m * sff];
            if !self.gpu_gemm(&b("ffn_gate_shexp.weight"), x, m, hidden, sff, &mut gs) {
                qmatmul_batch(&mut gs, x, m, wgs.bytes, wgs.ty, hidden, sff);
            }
            if !self.gpu_gemm(&b("ffn_up_shexp.weight"), x, m, hidden, sff, &mut us) {
                qmatmul_batch(&mut us, x, m, wus.bytes, wus.ty, hidden, sff);
            }
            let mut a = vec![0.0f32; m * sff];
            for i in 0..m * sff {
                a[i] = silu(gs[i]) * us[i];
            }
            if !self.gpu_gemm(&b("ffn_down_shexp.weight"), &a, m, sff, hidden, &mut shared) {
                qmatmul_batch(&mut shared, &a, m, wds.bytes, wds.ty, sff, hidden);
            }
            let mut sg = vec![0.0f32; m];
            qmatmul_batch(&mut sg, x, m, wgis.bytes, wgis.ty, hidden, 1);
            for t in 0..m {
                let s = sigmoid(sg[t]);
                for i in 0..hidden {
                    shared[t * hidden + i] *= s;
                }
            }
        }
        for (t, route) in routes.iter().enumerate() {
            let mut out = vec![0.0f32; hidden];
            for (s, &(_, w)) in route.iter().enumerate() {
                let dys = &dy[(t * topk + s) * hidden..(t * topk + s + 1) * hidden];
                for i in 0..hidden {
                    out[i] += w * dys[i];
                }
            }
            for i in 0..hidden {
                out[i] += shared[t * hidden + i];
            }
            let hrow = &mut h[t * hidden..(t + 1) * hidden];
            for i in 0..hidden {
                hrow[i] += out[i];
            }
        }
        Ok(())
    }

    fn forward(&mut self, token: u32, want_logits: bool) -> Result<Option<Vec<f32>>> {
        let cfg = self.cfg.clone();
        let pos = self.pos;
        let hidden = cfg.hidden;
        let eps = cfg.rms_eps;
        let mut row = vec![0.0f32; 16384]; // dequant scratch (>= max in_dim)

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

        let dprof = std::env::var("STRIX_DECODE_PROF").is_ok();
        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            // attn/token-mixer pre-norm
            let an = self.vecw(&b("attn_norm.weight"))?;
            let mut n = vec![0.0f32; hidden];
            rmsnorm(&mut n, &h, &an, eps);

            let tm = std::time::Instant::now();
            let mixed = if cfg.is_recr(il) {
                self.deltanet(&n, il, &mut row)?
            } else {
                self.attn(&n, il, pos, &mut row)?
            };
            if dprof {
                padd(&DPROF_MIX, tm);
            }
            for i in 0..hidden {
                h[i] += mixed[i];
            }

            // FFN block
            let ffn_res = h.clone();
            let pn = self.vecw(&b("post_attention_norm.weight"))?;
            let mut nn = vec![0.0f32; hidden];
            rmsnorm(&mut nn, &h, &pn, eps);
            let tf = std::time::Instant::now();
            let ffn = if cfg.is_moe() {
                self.moe(&nn, il, &mut row)?
            } else {
                self.dense_ffn(&nn, il, &mut row)?
            };
            if dprof {
                padd(&DPROF_FFN, tf);
            }
            for i in 0..hidden {
                h[i] = ffn_res[i] + ffn[i];
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
        let th = std::time::Instant::now();
        self.mm(head_key, &mut logits, &nh, &head, &mut row);
        if dprof {
            padd(&DPROF_HEAD, th);
            let n = DPROF_N.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 8 == 0 {
                let us = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e3 / n as f64;
                eprintln!(
                    "[decode prof] {n} tok avg/tok: mixer {:.1}ms | ffn {:.1}ms | lm_head {:.1}ms",
                    us(&DPROF_MIX),
                    us(&DPROF_FFN),
                    us(&DPROF_HEAD),
                );
            }
        }
        Ok(Some(logits))
    }

    fn attn(&mut self, x: &[f32], il: usize, pos: usize, row: &mut [f32]) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let hd = cfg.head_dim; // 256
        let nh = cfg.n_head; // 16
        let nkv = cfg.n_head_kv; // 2
        let groups = nh / nkv; // 8
        let kv_dim = nkv * hd; // 512
        let b = |s: &str| format!("blk.{il}.{s}");
        let qn = self.vecw(&b("attn_q_norm.weight"))?;
        let kn = self.vecw(&b("attn_k_norm.weight"))?;
        let mut qg = vec![0.0f32; hd * 2 * nh]; // 8192
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        {
            let mut got = false;
            if let Some(a) = &self.accel {
                let r = a.gemv_batch(&[
                    (&b("attn_q.weight"), x),
                    (&b("attn_k.weight"), x),
                    (&b("attn_v.weight"), x),
                ]);
                if let (Some(qv), Some(kv), Some(vv)) = (&r[0], &r[1], &r[2]) {
                    if qv.len() == qg.len() && kv.len() == k.len() && vv.len() == v.len() {
                        qg.copy_from_slice(qv);
                        k.copy_from_slice(kv);
                        v.copy_from_slice(vv);
                        got = true;
                    }
                }
            }
            if !got {
                let wq = self.w(&b("attn_q.weight"))?;
                self.mm(&b("attn_q.weight"), &mut qg, x, &wq, row);
                let wk = self.w(&b("attn_k.weight"))?;
                self.mm(&b("attn_k.weight"), &mut k, x, &wk, row);
                let wv = self.w(&b("attn_v.weight"))?;
                self.mm(&b("attn_v.weight"), &mut v, x, &wv, row);
            }
        }

        // per-head Q (+gate) norm + rope; per-kv-head K norm + rope
        let mut q = vec![0.0f32; hd * nh];
        let mut gate = vec![0.0f32; hd * nh];
        for hh in 0..nh {
            let base = hh * hd * 2;
            let mut qh = qg[base..base + hd].to_vec();
            let gh = &qg[base + hd..base + 2 * hd];
            let mut tmp = vec![0.0f32; hd];
            rmsnorm(&mut tmp, &qh, &qn, cfg.rms_eps);
            qh.copy_from_slice(&tmp);
            partial_rope(&mut qh, pos, cfg.rope_freq_base, cfg.n_rot);
            q[hh * hd..hh * hd + hd].copy_from_slice(&qh);
            gate[hh * hd..hh * hd + hd].copy_from_slice(gh);
        }
        for kh in 0..nkv {
            let mut khv = k[kh * hd..kh * hd + hd].to_vec();
            let mut tmp = vec![0.0f32; hd];
            rmsnorm(&mut tmp, &khv, &kn, cfg.rms_eps);
            khv.copy_from_slice(&tmp);
            partial_rope(&mut khv, pos, cfg.rope_freq_base, cfg.n_rot);
            k[kh * hd..kh * hd + hd].copy_from_slice(&khv);
        }

        // append to KV cache
        self.kc[il].extend_from_slice(&k);
        self.vc[il].extend_from_slice(&v);
        let len = pos + 1;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut attn_out = vec![0.0f32; hd * nh];
        let mut keys = vec![0.0f32; len * hd];
        let mut vals = vec![0.0f32; len * hd];
        let mut scratch = vec![0.0f32; len];
        for hh in 0..nh {
            let kvh = hh / groups;
            for t in 0..len {
                keys[t * hd..t * hd + hd].copy_from_slice(
                    &self.kc[il][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd],
                );
                vals[t * hd..t * hd + hd].copy_from_slice(
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
                len,
                scale,
                &mut scratch,
            );
            // output gating
            for d in 0..hd {
                oh[d] *= sigmoid(gate[hh * hd + d]);
            }
            attn_out[hh * hd..hh * hd + hd].copy_from_slice(&oh);
        }
        let mut o = vec![0.0f32; cfg.hidden];
        {
            let wo = self.w(&b("attn_output.weight"))?;
            self.mm(&b("attn_output.weight"), &mut o, &attn_out, &wo, row);
        }
        Ok(o)
    }

    fn deltanet(&mut self, x: &[f32], il: usize, row: &mut [f32]) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let s_v = cfg.ssm_d_inner / cfg.ssm_dt_rank; // 128
        let n_vh = cfg.ssm_dt_rank; // 32
        let n_kh = cfg.ssm_n_group; // 16
        let key_dim = cfg.ssm_d_state * cfg.ssm_n_group; // 2048
        let value_dim = cfg.ssm_d_inner; // 4096
        let conv_dim = key_dim * 2 + value_dim; // 8192
        let dconv = cfg.ssm_d_conv; // 4
        let b = |s: &str| format!("blk.{il}.{s}");

        let ssm_a = self.vecw(&b("ssm_a"))?;
        let ssm_dt = self.vecw(&b("ssm_dt.bias"))?;
        let ssm_norm = self.vecw(&b("ssm_norm.weight"))?;
        let conv_w = self.vecw(&b("ssm_conv1d.weight"))?; // [d_conv * conv_dim], chan c taps = [c*dconv..]

        let mut qkv = vec![0.0f32; conv_dim];
        let mut z = vec![0.0f32; value_dim];
        let mut beta_raw = vec![0.0f32; n_vh];
        let mut alpha_raw = vec![0.0f32; n_vh];
        {
            let wqkv = self.w(&b("attn_qkv.weight"))?;
            let mut got = false;
            if let Some(a) = &self.accel {
                let r = a.gemv_batch(&[(&b("attn_qkv.weight"), x), (&b("attn_gate.weight"), x)]);
                if let (Some(qv), Some(zv)) = (&r[0], &r[1]) {
                    if qv.len() == qkv.len() && zv.len() == z.len() {
                        qkv.copy_from_slice(qv);
                        z.copy_from_slice(zv);
                        got = true;
                    }
                }
            }
            if !got {
                qmatmul(&mut qkv, x, wqkv.bytes, wqkv.ty, wqkv.in_dim, row);
            }
            let wgate = self.w(&b("attn_gate.weight"))?;
            if !got {
                qmatmul(&mut z, x, wgate.bytes, wgate.ty, wgate.in_dim, row);
            }
            let wbeta = self.w(&b("ssm_beta.weight"))?;
            qmatmul(&mut beta_raw, x, wbeta.bytes, wbeta.ty, wbeta.in_dim, row);
            let walpha = self.w(&b("ssm_alpha.weight"))?;
            qmatmul(
                &mut alpha_raw,
                x,
                walpha.bytes,
                walpha.ty,
                walpha.in_dim,
                row,
            );
        }

        // causal depthwise conv (kernel dconv) over qkv using conv state, then silu
        let cs = &mut self.conv[il]; // [conv_dim * (dconv-1)]
        let mut conv_out = vec![0.0f32; conv_dim];
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            // window: [hist(dconv-1) ..., current]; taps w[c*dconv + k]
            for k in 0..dconv - 1 {
                acc += conv_w[c * dconv + k] * cs[c * (dconv - 1) + k];
            }
            acc += conv_w[c * dconv + (dconv - 1)] * qkv[c];
            conv_out[c] = silu(acc);
            // shift conv state: drop oldest, append current
            for k in 0..dconv - 2 {
                cs[c * (dconv - 1) + k] = cs[c * (dconv - 1) + k + 1];
            }
            cs[c * (dconv - 1) + (dconv - 2)] = qkv[c];
        }

        // split q,k,v from conv_out
        let q = &conv_out[0..key_dim]; // 16 heads x 128
        let k = &conv_out[key_dim..2 * key_dim];
        let v = &conv_out[2 * key_dim..2 * key_dim + value_dim]; // 32 heads x 128
                                                                 // L2-norm q,k per k-head (128)
        let mut qn = vec![0.0f32; key_dim];
        let mut kn = vec![0.0f32; key_dim];
        qn.copy_from_slice(q);
        kn.copy_from_slice(k);
        for kh in 0..n_kh {
            l2norm(&mut qn[kh * s_v..kh * s_v + s_v], cfg.rms_eps);
            l2norm(&mut kn[kh * s_v..kh * s_v + s_v], cfg.rms_eps);
        }

        let scale = 1.0 / (s_v as f32).sqrt();
        let mut core = vec![0.0f32; value_dim]; // 32 heads x 128 (pre-gate-norm)
        for vh in 0..n_vh {
            let kh = vh % n_kh; // tiled broadcast
            let qh = &qn[kh * s_v..kh * s_v + s_v];
            let kk = &kn[kh * s_v..kh * s_v + s_v];
            let vv = &v[vh * s_v..vh * s_v + s_v];
            let g = ssm_a[vh] * softplus(alpha_raw[vh] + ssm_dt[vh]);
            let betah = sigmoid(beta_raw[vh]);
            let decay = g.exp();
            let st = &mut self.ssm[il][vh * s_v * s_v..vh * s_v * s_v + s_v * s_v]; // s[j*s_v+i]=S[i][j]
                                                                                    // 1) decay
            for x in st.iter_mut() {
                *x *= decay;
            }
            // 2) delta[j] = (v[j] - dot(S_row_j, k)) * beta
            let mut delta = vec![0.0f32; s_v];
            for j in 0..s_v {
                let mut sum = 0.0f32;
                let rowj = &st[j * s_v..j * s_v + s_v];
                for i in 0..s_v {
                    sum += rowj[i] * kk[i];
                }
                delta[j] = (vv[j] - sum) * betah;
            }
            // 3) S[i][j] += k[i]*delta[j]  => s[j*s_v+i] += delta[j]*k[i]
            for j in 0..s_v {
                let dj = delta[j];
                let rowj = &mut st[j * s_v..j * s_v + s_v];
                for i in 0..s_v {
                    rowj[i] += dj * kk[i];
                }
            }
            // 4) out[j] = dot(S_row_j, q) * scale
            let outh = &mut core[vh * s_v..vh * s_v + s_v];
            for j in 0..s_v {
                let rowj = &st[j * s_v..j * s_v + s_v];
                let mut sum = 0.0f32;
                for i in 0..s_v {
                    sum += rowj[i] * qh[i];
                }
                outh[j] = sum * scale;
            }
        }

        // gated norm: per v-head, rmsnorm(core_h, ssm_norm) * silu(z_h)
        let mut gated = vec![0.0f32; value_dim];
        for vh in 0..n_vh {
            let mut tmp = vec![0.0f32; s_v];
            rmsnorm(
                &mut tmp,
                &core[vh * s_v..vh * s_v + s_v],
                &ssm_norm,
                cfg.rms_eps,
            );
            for j in 0..s_v {
                gated[vh * s_v + j] = tmp[j] * silu(z[vh * s_v + j]);
            }
        }
        let mut o = vec![0.0f32; cfg.hidden];
        if !self.try_gemv(&b("ssm_out.weight"), &gated, &mut o) {
            let wout = self.w(&b("ssm_out.weight"))?;
            qmatmul(&mut o, &gated, wout.bytes, wout.ty, wout.in_dim, row);
        }
        Ok(o)
    }

    /// Dense SwiGLU FFN (`qwen35`): `down( silu(gate(x)) * up(x) )`. Returns the FFN
    /// output (caller adds the residual). GPU gemv per weight key if resident.
    fn dense_ffn(&self, x: &[f32], il: usize, row: &mut [f32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let ff = cfg.dense_ff;
        let b = |s: &str| format!("blk.{il}.{s}");
        let wg = self.w(&b("ffn_gate.weight"))?;
        let wu = self.w(&b("ffn_up.weight"))?;
        let wd = self.w(&b("ffn_down.weight"))?;
        let mut g = vec![0.0f32; ff];
        let mut u = vec![0.0f32; ff];
        self.mm(&b("ffn_gate.weight"), &mut g, x, &wg, row);
        self.mm(&b("ffn_up.weight"), &mut u, x, &wu, row);
        let mut act = vec![0.0f32; ff];
        for i in 0..ff {
            act[i] = silu(g[i]) * u[i];
        }
        let mut out = vec![0.0f32; hidden];
        self.mm(&b("ffn_down.weight"), &mut out, &act, &wd, row);
        Ok(out)
    }

    /// Batched dense SwiGLU FFN for prefill (weight-read-once); adds the result into
    /// `h` rows (residual), matching `moe_batch`/`moe_finish`'s convention.
    fn dense_ffn_batch(&mut self, x: &[f32], m: usize, il: usize, h: &mut [f32]) -> Result<()> {
        let cfg = self.cfg.clone();
        let hidden = cfg.hidden;
        let ff = cfg.dense_ff;
        let b = |s: &str| format!("blk.{il}.{s}");
        let mut g = vec![0.0f32; m * ff];
        let mut u = vec![0.0f32; m * ff];
        if !self.gpu_gemm(&b("ffn_gate.weight"), x, m, hidden, ff, &mut g) {
            let wg = self.w(&b("ffn_gate.weight"))?;
            qmatmul_batch(&mut g, x, m, wg.bytes, wg.ty, hidden, ff);
        }
        if !self.gpu_gemm(&b("ffn_up.weight"), x, m, hidden, ff, &mut u) {
            let wu = self.w(&b("ffn_up.weight"))?;
            qmatmul_batch(&mut u, x, m, wu.bytes, wu.ty, hidden, ff);
        }
        let mut act = vec![0.0f32; m * ff];
        for i in 0..m * ff {
            act[i] = silu(g[i]) * u[i];
        }
        let mut dy = vec![0.0f32; m * hidden];
        if !self.gpu_gemm(&b("ffn_down.weight"), &act, m, ff, hidden, &mut dy) {
            let wd = self.w(&b("ffn_down.weight"))?;
            qmatmul_batch(&mut dy, &act, m, wd.bytes, wd.ty, ff, hidden);
        }
        for i in 0..m * hidden {
            h[i] += dy[i];
        }
        Ok(())
    }

    fn moe(&self, x: &[f32], il: usize, row: &mut [f32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let ne = cfg.n_expert; // 256
        let topk = cfg.n_expert_used; // 8
        let eff = cfg.expert_ff; // 512
        let b = |s: &str| format!("blk.{il}.{s}");

        // router
        let wgi = self.w(&b("ffn_gate_inp.weight"))?;
        let mut logits = vec![0.0f32; ne];
        qmatmul(&mut logits, x, wgi.bytes, wgi.ty, wgi.in_dim, row);
        // softmax over all experts, then take top-k and renormalize their weights
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

        // Fused GPU MoE for the routed experts (shared expert stays on CPU below).
        let mut gpu_out: Option<Vec<f32>> = None;
        if let Some(a) = &self.accel {
            let ids: Vec<i32> = idx.iter().map(|&e| e as i32).collect();
            let wexp: Vec<f32> = idx.iter().map(|&e| probs[e] / wsum).collect();
            // shared-expert sigmoid gate computed CPU (its router weight is F32);
            // the shexp FFN itself is fused on-GPU under the same sync.
            let mut sgate = 0.0f32;
            if cfg.shared_ff > 0 {
                let wgis = self.w(&b("ffn_gate_inp_shexp.weight"))?;
                let mut sg = [0.0f32; 1];
                qmatmul(&mut sg, x, wgis.bytes, wgis.ty, wgis.in_dim, row);
                sgate = sigmoid(sg[0]);
            }
            gpu_out = a
                .moe_ffn(il, &ids, &wexp, x, sgate)
                .filter(|y| y.len() == hidden);
        }

        let wge = self.w(&b("ffn_gate_exps.weight"))?; // [hidden, eff, ne]
        let wue = self.w(&b("ffn_up_exps.weight"))?;
        let wde = self.w(&b("ffn_down_exps.weight"))?; // [eff, hidden, ne]
        let gate_bpr = (hidden / wge.ty.block_elems()) * wge.ty.block_bytes() * eff; // bytes per expert
        let up_bpr = (hidden / wue.ty.block_elems()) * wue.ty.block_bytes() * eff;
        let down_bpr = (eff / wde.ty.block_elems()) * wde.ty.block_bytes() * hidden;

        let routed_on_gpu = gpu_out.is_some();
        let mut out = gpu_out.unwrap_or_else(|| vec![0.0f32; hidden]);
        let mut g = vec![0.0f32; eff];
        let mut u = vec![0.0f32; eff];
        let mut act = vec![0.0f32; eff];
        let mut dy = vec![0.0f32; hidden];
        for &e in if routed_on_gpu { &idx[..0] } else { &idx[..] } {
            let wexp = probs[e] / wsum;
            // GPU gemv per expert slice (key `blk.{il}.ffn_*_exps.e{e}`) if resident,
            // else CPU on the byte slice. Experts are the bulk of decode compute.
            let gkey = format!("blk.{il}.ffn_gate_exps.e{e}");
            if !self.try_gemv(&gkey, x, &mut g) {
                qmatmul(
                    &mut g,
                    x,
                    &wge.bytes[e * gate_bpr..(e + 1) * gate_bpr],
                    wge.ty,
                    hidden,
                    row,
                );
            }
            let ukey = format!("blk.{il}.ffn_up_exps.e{e}");
            if !self.try_gemv(&ukey, x, &mut u) {
                qmatmul(
                    &mut u,
                    x,
                    &wue.bytes[e * up_bpr..(e + 1) * up_bpr],
                    wue.ty,
                    hidden,
                    row,
                );
            }
            for i in 0..eff {
                act[i] = silu(g[i]) * u[i];
            }
            let dkey = format!("blk.{il}.ffn_down_exps.e{e}");
            if !self.try_gemv(&dkey, &act, &mut dy) {
                qmatmul(
                    &mut dy,
                    &act,
                    &wde.bytes[e * down_bpr..(e + 1) * down_bpr],
                    wde.ty,
                    eff,
                    row,
                );
            }
            for i in 0..hidden {
                out[i] += wexp * dy[i];
            }
        }

        // shared expert (sigmoid-gated). When routed on GPU, moe_ffn already fused
        // the shared expert into the result — skip CPU.
        let sff = cfg.shared_ff;
        if sff > 0 && !routed_on_gpu {
            let wgs = self.w(&b("ffn_gate_shexp.weight"))?;
            let wus = self.w(&b("ffn_up_shexp.weight"))?;
            let wds = self.w(&b("ffn_down_shexp.weight"))?;
            let wgis = self.w(&b("ffn_gate_inp_shexp.weight"))?;
            let mut gs = vec![0.0f32; sff];
            let mut us = vec![0.0f32; sff];
            let mut a = vec![0.0f32; sff];
            let mut ds = vec![0.0f32; hidden];
            let mut got = false;
            if let Some(acc) = &self.accel {
                let r = acc.gemv_batch(&[
                    (&b("ffn_gate_shexp.weight"), x),
                    (&b("ffn_up_shexp.weight"), x),
                ]);
                if let (Some(gv), Some(uv)) = (&r[0], &r[1]) {
                    if gv.len() == sff && uv.len() == sff {
                        gs.copy_from_slice(gv);
                        us.copy_from_slice(uv);
                        got = true;
                    }
                }
            }
            if !got {
                qmatmul(&mut gs, x, wgs.bytes, wgs.ty, wgs.in_dim, row);
                qmatmul(&mut us, x, wus.bytes, wus.ty, wus.in_dim, row);
            }
            for i in 0..sff {
                a[i] = silu(gs[i]) * us[i];
            }
            if !self.try_gemv(&b("ffn_down_shexp.weight"), &a, &mut ds) {
                qmatmul(&mut ds, &a, wds.bytes, wds.ty, wds.in_dim, row);
            }
            let mut sg = [0.0f32; 1];
            qmatmul(&mut sg, x, wgis.bytes, wgis.ty, wgis.in_dim, row);
            let sgate = sigmoid(sg[0]);
            for i in 0..hidden {
                out[i] += ds[i] * sgate;
            }
        }
        Ok(out)
    }
}

impl Decoder for Qwen35Model {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits> {
        if tokens.is_empty() {
            return Err(StrixError::invalid("qwen35 prefill: empty"));
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
        let logits = self.prefill_batch(tokens)?;
        // Resident GPU decode: seed the device KV (full-attn layers) + SSM/conv
        // recurrent state (DeltaNet layers) from the CPU prefill so decode runs
        // wholly on-device. Prefill itself never touches the iGPU (crash-safe).
        if self.gpu_decode {
            let mut accel = self.accel.take();
            if let Some(a) = accel.as_mut() {
                for il in 0..self.cfg.n_layer {
                    if self.cfg.is_recr(il) {
                        a.seed_qwen35_state(il, &self.ssm[il], &self.conv[il]);
                    } else {
                        a.seed_qwen35_kv(il, &self.kc[il], &self.vc[il]);
                    }
                }
            }
            self.accel = accel;
        }
        Ok(Logits::new(logits))
    }

    fn decode_one(&mut self, token: u32) -> Result<Logits> {
        if self.gpu_decode {
            let emb = self.embed_row(token)?;
            let pos = self.pos;
            let mut accel = self.accel.take();
            let out = accel.as_mut().and_then(|a| a.decode_step_qwen35(&emb, pos));
            self.accel = accel;
            self.pos += 1;
            return out
                .map(Logits::new)
                .ok_or_else(|| StrixError::invalid("qwen35 resident GPU decode failed"));
        }
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
        for v in self.ssm.iter_mut() {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
        for v in self.conv.iter_mut() {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }
}
