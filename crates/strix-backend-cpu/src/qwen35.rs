//! Qwen3.5/3.6-MoE (`qwen35moe`) bring-up — Phase 0: config parsing + tensor
//! validation. This is a Qwen3-Next-class HYBRID model: most layers are Gated-
//! DeltaNet linear-attention (recurrent, `ssm_*` tensors), every `full_attention_
//! interval`-th layer is full GQA attention; every layer has a 256-expert top-8
//! MoE + a sigmoid-gated shared expert. See docs/qwen36-arch.md for the full spec.
//!
//! P0 only parses the architecture + verifies all expected tensors are present
//! (with correct shapes). The forward (P1 MoE, P2 attn, P3 gated-deltanet) is TODO.

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
    pub n_rot: usize,          // rope dimension_count (partial: < head_dim)
    pub rope_sections: [i64; 4],
    pub full_attn_interval: usize,
    // MoE
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_ff: usize,
    pub shared_ff: usize,
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
            .ok_or_else(|| StrixError::invalid("qwen35: no general.architecture"))?;
        if arch != "qwen35moe" {
            return Err(StrixError::unsupported(format!(
                "qwen35: arch `{arch}` is not qwen35moe"
            )));
        }
        let k = |s: &str| format!("qwen35moe.{s}");
        // rope sections [11,11,10,0]
        let mut rope_sections = [0i64; 4];
        if let Some(arr) = g.meta(&k("rope.dimension_sections")).and_then(|v| v.as_array()) {
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
            n_expert: mu32(g, &k("expert_count"))? as usize,
            n_expert_used: mu32(g, &k("expert_used_count"))? as usize,
            expert_ff: mu32(g, &k("expert_feed_forward_length"))? as usize,
            shared_ff: mu32_or(g, &k("expert_shared_feed_forward_length"), 0) as usize,
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

    pub fn report(&self) -> String {
        let n_attn = (0..self.n_layer).filter(|&l| !self.is_recr(l)).count();
        format!(
            "qwen35moe: {} layers ({} recurrent / {} full-attn), hidden={}, vocab={}, ctx={}\n  \
             attn: head_dim={} n_head={} n_head_kv={} (GQA {}:1), QK-norm, IMRoPE n_rot={} sections={:?} freq_base={:.0}\n  \
             MoE: {} experts top-{}, expert_ff={}, shared_ff={}\n  \
             SSM(GatedDeltaNet): d_conv={} d_inner={} d_state={} v_heads(dt_rank)={} k_heads(n_group)={}, rms_eps={:.1e}",
            self.n_layer, self.n_layer - n_attn, n_attn, self.hidden, self.vocab, self.ctx_len,
            self.head_dim, self.n_head, self.n_head_kv, self.n_head / self.n_head_kv.max(1),
            self.n_rot, self.rope_sections, self.rope_freq_base,
            self.n_expert, self.n_expert_used, self.expert_ff, self.shared_ff,
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
            want(b("ssm_conv1d.weight"), &[cfg.ssm_d_conv, key_dim * 2 + value_dim]);
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
        // MoE (every layer)
        want(b("ffn_gate_inp.weight"), &[cfg.hidden, cfg.n_expert]);
        want(b("ffn_gate_exps.weight"), &[cfg.hidden, cfg.expert_ff, cfg.n_expert]);
        want(b("ffn_up_exps.weight"), &[cfg.hidden, cfg.expert_ff, cfg.n_expert]);
        want(b("ffn_down_exps.weight"), &[cfg.expert_ff, cfg.hidden, cfg.n_expert]);
        want(b("ffn_gate_inp_shexp.weight"), &[cfg.hidden]);
        want(b("ffn_gate_shexp.weight"), &[cfg.hidden, cfg.shared_ff]);
        want(b("ffn_up_shexp.weight"), &[cfg.hidden, cfg.shared_ff]);
        want(b("ffn_down_shexp.weight"), &[cfg.shared_ff, cfg.hidden]);
    }

    let tied = !t.contains_key("output.weight");
    let report = format!(
        "{}\n  tensors: {} checked, {} missing/mismatched; lm_head {}",
        cfg.report(),
        checked,
        missing.len(),
        if tied { "tied to token_embd" } else { "separate output.weight" },
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
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
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
fn qmatmul(out: &mut [f32], x: &[f32], bytes: &[u8], ty: GgmlType, in_dim: usize, row: &mut [f32]) {
    let bpr = (in_dim / ty.block_elems()) * ty.block_bytes();
    let row = &mut row[..in_dim];
    for (o, oref) in out.iter_mut().enumerate() {
        dequantize_into(ty, &bytes[o * bpr..o * bpr + bpr], row).unwrap();
        let mut s = 0.0f32;
        for i in 0..in_dim {
            s += row[i] * x[i];
        }
        *oref = s;
    }
}

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
        })
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
            dequantize_into(emb.ty, &emb.bytes[token as usize * bpr..token as usize * bpr + bpr], &mut h)?;
        }

        for il in 0..cfg.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            // attn/token-mixer pre-norm
            let an = self.vecw(&b("attn_norm.weight"))?;
            let mut n = vec![0.0f32; hidden];
            rmsnorm(&mut n, &h, &an, eps);

            let mixed = if cfg.is_recr(il) {
                self.deltanet(&n, il, &mut row)?
            } else {
                self.attn(&n, il, pos, &mut row)?
            };
            for i in 0..hidden {
                h[i] += mixed[i];
            }

            // FFN block
            let ffn_res = h.clone();
            let pn = self.vecw(&b("post_attention_norm.weight"))?;
            let mut nn = vec![0.0f32; hidden];
            rmsnorm(&mut nn, &h, &pn, eps);
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
        let head = if self.gguf.tensors().contains_key("output.weight") {
            self.w("output.weight")?
        } else {
            self.w("token_embd.weight")?
        };
        let mut logits = vec![0.0f32; cfg.vocab];
        qmatmul(&mut logits, &nh, head.bytes, head.ty, hidden, &mut row);
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
            let wq = self.w(&b("attn_q.weight"))?;
            qmatmul(&mut qg, x, wq.bytes, wq.ty, wq.in_dim, row);
            let wk = self.w(&b("attn_k.weight"))?;
            qmatmul(&mut k, x, wk.bytes, wk.ty, wk.in_dim, row);
            let wv = self.w(&b("attn_v.weight"))?;
            qmatmul(&mut v, x, wv.bytes, wv.ty, wv.in_dim, row);
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
                keys[t * hd..t * hd + hd]
                    .copy_from_slice(&self.kc[il][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd]);
                vals[t * hd..t * hd + hd]
                    .copy_from_slice(&self.vc[il][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd]);
            }
            let mut oh = vec![0.0f32; hd];
            crate::attention::sdpa_single(&mut oh, &q[hh * hd..hh * hd + hd], &keys, &vals, hd, len, scale, &mut scratch);
            // output gating
            for d in 0..hd {
                oh[d] *= sigmoid(gate[hh * hd + d]);
            }
            attn_out[hh * hd..hh * hd + hd].copy_from_slice(&oh);
        }
        let mut o = vec![0.0f32; cfg.hidden];
        {
            let wo = self.w(&b("attn_output.weight"))?;
            qmatmul(&mut o, &attn_out, wo.bytes, wo.ty, wo.in_dim, row);
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
            qmatmul(&mut qkv, x, wqkv.bytes, wqkv.ty, wqkv.in_dim, row);
            let wgate = self.w(&b("attn_gate.weight"))?;
            qmatmul(&mut z, x, wgate.bytes, wgate.ty, wgate.in_dim, row);
            let wbeta = self.w(&b("ssm_beta.weight"))?;
            qmatmul(&mut beta_raw, x, wbeta.bytes, wbeta.ty, wbeta.in_dim, row);
            let walpha = self.w(&b("ssm_alpha.weight"))?;
            qmatmul(&mut alpha_raw, x, walpha.bytes, walpha.ty, walpha.in_dim, row);
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
            rmsnorm(&mut tmp, &core[vh * s_v..vh * s_v + s_v], &ssm_norm, cfg.rms_eps);
            for j in 0..s_v {
                gated[vh * s_v + j] = tmp[j] * silu(z[vh * s_v + j]);
            }
        }
        let mut o = vec![0.0f32; cfg.hidden];
        {
            let wout = self.w(&b("ssm_out.weight"))?;
            qmatmul(&mut o, &gated, wout.bytes, wout.ty, wout.in_dim, row);
        }
        Ok(o)
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

        let wge = self.w(&b("ffn_gate_exps.weight"))?; // [hidden, eff, ne]
        let wue = self.w(&b("ffn_up_exps.weight"))?;
        let wde = self.w(&b("ffn_down_exps.weight"))?; // [eff, hidden, ne]
        let gate_bpr = (hidden / wge.ty.block_elems()) * wge.ty.block_bytes() * eff; // bytes per expert
        let up_bpr = (hidden / wue.ty.block_elems()) * wue.ty.block_bytes() * eff;
        let down_bpr = (eff / wde.ty.block_elems()) * wde.ty.block_bytes() * hidden;

        let mut out = vec![0.0f32; hidden];
        let mut g = vec![0.0f32; eff];
        let mut u = vec![0.0f32; eff];
        let mut act = vec![0.0f32; eff];
        let mut dy = vec![0.0f32; hidden];
        for &e in &idx {
            let wexp = probs[e] / wsum;
            let ge = &wge.bytes[e * gate_bpr..(e + 1) * gate_bpr];
            let ue = &wue.bytes[e * up_bpr..(e + 1) * up_bpr];
            let de = &wde.bytes[e * down_bpr..(e + 1) * down_bpr];
            qmatmul(&mut g, x, ge, wge.ty, hidden, row);
            qmatmul(&mut u, x, ue, wue.ty, hidden, row);
            for i in 0..eff {
                act[i] = silu(g[i]) * u[i];
            }
            qmatmul(&mut dy, &act, de, wde.ty, eff, row);
            for i in 0..hidden {
                out[i] += wexp * dy[i];
            }
        }

        // shared expert (sigmoid-gated)
        let sff = cfg.shared_ff;
        if sff > 0 {
            let wgs = self.w(&b("ffn_gate_shexp.weight"))?;
            let wus = self.w(&b("ffn_up_shexp.weight"))?;
            let wds = self.w(&b("ffn_down_shexp.weight"))?;
            let wgis = self.w(&b("ffn_gate_inp_shexp.weight"))?;
            let mut gs = vec![0.0f32; sff];
            let mut us = vec![0.0f32; sff];
            let mut a = vec![0.0f32; sff];
            let mut ds = vec![0.0f32; hidden];
            qmatmul(&mut gs, x, wgs.bytes, wgs.ty, wgs.in_dim, row);
            qmatmul(&mut us, x, wus.bytes, wus.ty, wus.in_dim, row);
            for i in 0..sff {
                a[i] = silu(gs[i]) * us[i];
            }
            qmatmul(&mut ds, &a, wds.bytes, wds.ty, wds.in_dim, row);
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
        for (i, &t) in tokens.iter().enumerate() {
            let last = i == tokens.len() - 1;
            let o = self.forward(t, last)?;
            if last {
                return Ok(Logits::new(o.unwrap()));
            }
        }
        unreachable!()
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
        for v in self.ssm.iter_mut() {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
        for v in self.conv.iter_mut() {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }
}
