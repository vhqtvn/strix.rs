//! Qwen3.5/3.6-MoE (`qwen35moe`) bring-up — Phase 0: config parsing + tensor
//! validation. This is a Qwen3-Next-class HYBRID model: most layers are Gated-
//! DeltaNet linear-attention (recurrent, `ssm_*` tensors), every `full_attention_
//! interval`-th layer is full GQA attention; every layer has a 256-expert top-8
//! MoE + a sigmoid-gated shared expert. See docs/qwen36-arch.md for the full spec.
//!
//! P0 only parses the architecture + verifies all expected tensors are present
//! (with correct shapes). The forward (P1 MoE, P2 attn, P3 gated-deltanet) is TODO.

use strix_core::error::{Result, StrixError};
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
