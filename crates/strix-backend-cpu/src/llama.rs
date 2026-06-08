//! CPU reference forward pass for a Llama-like decoder.
//!
//! Covers Llama 3.x / Mistral / Qwen2.5-3 shape models that share the standard
//! decoder block: RMSNorm → QKV → RoPE → causal GQA attention → o_proj →
//! residual → RMSNorm → SwiGLU MLP → residual. Weights are HF-named safetensors
//! materialized to `f32` (see `strix_models::load_safetensors`).
//!
//! One token is processed per `forward` call; `prefill` just loops over the
//! prompt. Slow and allocation-happy on purpose — this is the correctness
//! oracle, not the fast path.

use strix_core::backend::{Decoder, Model};
use strix_core::error::{Result, StrixError};
use strix_core::kv_cache::KvCache;
use strix_core::model_config::{ModelArchitecture, ModelConfig};
use strix_core::sampler::Logits;
use strix_models::TensorMap;

use crate::attention::sdpa_single;
use crate::kv_cache::CpuKvCache;
use crate::ops::{linear, rmsnorm, rope_in_place, swiglu};

/// Per-layer weights, row-major, HF `nn.Linear` layout (`[out, in]`).
struct LayerWeights {
    input_ln: Vec<f32>,  // [hidden]
    q_proj: Vec<f32>,    // [q_dim, hidden]
    k_proj: Vec<f32>,    // [kv_dim, hidden]
    v_proj: Vec<f32>,    // [kv_dim, hidden]
    o_proj: Vec<f32>,    // [hidden, q_dim]
    post_ln: Vec<f32>,   // [hidden]
    gate_proj: Vec<f32>, // [inter, hidden]
    up_proj: Vec<f32>,   // [inter, hidden]
    down_proj: Vec<f32>, // [hidden, inter]
}

/// A loaded Llama-like model plus its single-sequence KV cache.
pub struct LlamaModel {
    config: ModelConfig,
    embed_tokens: Vec<f32>, // [vocab, hidden]
    layers: Vec<LayerWeights>,
    final_norm: Vec<f32>, // [hidden]
    lm_head: Vec<f32>,    // [vocab, hidden] (tied => clone of embed_tokens)
    cache: CpuKvCache,

    // Derived dims, cached for the hot loop.
    hidden: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    inter: usize,
    vocab: usize,
    groups: usize, // n_heads / n_kv
}

/// Remove a tensor from the map, checking its element count.
fn take(map: &mut TensorMap, name: &str, expect: usize) -> Result<Vec<f32>> {
    let t = map
        .remove(name)
        .ok_or_else(|| StrixError::invalid(format!("missing tensor `{name}`")))?;
    if t.data.len() != expect {
        return Err(StrixError::invalid(format!(
            "tensor `{name}` has {} elements, expected {expect}",
            t.data.len()
        )));
    }
    Ok(t.data)
}

impl LlamaModel {
    /// Build a model from a tensor map and config.
    ///
    /// `max_seq` bounds the KV cache (and thus the longest sequence). It is
    /// clamped to the model's `max_position_embeddings`.
    pub fn from_tensors(config: ModelConfig, mut map: TensorMap, max_seq: usize) -> Result<Self> {
        let hidden = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        if n_kv == 0 || n_heads % n_kv != 0 {
            return Err(StrixError::unsupported(format!(
                "num_attention_heads ({n_heads}) must be a multiple of num_key_value_heads ({n_kv})"
            )));
        }
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let groups = n_heads / n_kv;
        let max_seq = max_seq.min(config.max_position_embeddings).max(1);

        let embed_tokens = take(&mut map, "model.embed_tokens.weight", vocab * hidden)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(LayerWeights {
                input_ln: take(&mut map, &format!("{p}.input_layernorm.weight"), hidden)?,
                q_proj: take(
                    &mut map,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim * hidden,
                )?,
                k_proj: take(
                    &mut map,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim * hidden,
                )?,
                v_proj: take(
                    &mut map,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim * hidden,
                )?,
                o_proj: take(
                    &mut map,
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden * q_dim,
                )?,
                post_ln: take(
                    &mut map,
                    &format!("{p}.post_attention_layernorm.weight"),
                    hidden,
                )?,
                gate_proj: take(
                    &mut map,
                    &format!("{p}.mlp.gate_proj.weight"),
                    inter * hidden,
                )?,
                up_proj: take(&mut map, &format!("{p}.mlp.up_proj.weight"), inter * hidden)?,
                down_proj: take(
                    &mut map,
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden * inter,
                )?,
            });
        }

        let final_norm = take(&mut map, "model.norm.weight", hidden)?;

        // lm_head may be tied to the embedding table (common on small models).
        let lm_head = match map.remove("lm_head.weight") {
            Some(t) if t.data.len() == vocab * hidden => t.data,
            Some(t) => {
                return Err(StrixError::invalid(format!(
                    "lm_head.weight has {} elements, expected {}",
                    t.data.len(),
                    vocab * hidden
                )))
            }
            None => {
                tracing::debug!("lm_head.weight absent; using tied embed_tokens");
                embed_tokens.clone()
            }
        };

        let cache = CpuKvCache::new(config.num_hidden_layers, n_kv, head_dim, max_seq);

        Ok(LlamaModel {
            config,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            cache,
            hidden,
            n_heads,
            n_kv,
            head_dim,
            q_dim,
            kv_dim,
            inter,
            vocab,
            groups,
        })
    }

    /// Maximum sequence length this model's cache can hold.
    pub fn max_seq(&self) -> usize {
        self.cache.capacity()
    }

    /// Run one token through the network, updating the cache, returning logits.
    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let tok = token as usize;
        if tok >= self.vocab {
            return Err(StrixError::invalid(format!(
                "token id {tok} out of range (vocab {})",
                self.vocab
            )));
        }
        let eps = self.config.rms_norm_eps;
        let theta = self.config.rope_theta;
        let hd = self.head_dim;

        // Embedding lookup → working hidden state.
        let mut h = self.embed_tokens[tok * self.hidden..(tok + 1) * self.hidden].to_vec();

        let pos = self.cache.advance()?;
        let len = pos + 1;

        // Scratch reused across layers.
        let mut xn = vec![0.0f32; self.hidden];
        let mut q = vec![0.0f32; self.q_dim];
        let mut k = vec![0.0f32; self.kv_dim];
        let mut v = vec![0.0f32; self.kv_dim];
        let mut attn = vec![0.0f32; self.q_dim];
        let mut scores = vec![0.0f32; len];
        let mut o = vec![0.0f32; self.hidden];
        let mut gate = vec![0.0f32; self.inter];
        let mut up = vec![0.0f32; self.inter];
        let mut act = vec![0.0f32; self.inter];
        let mut down = vec![0.0f32; self.hidden];

        for layer in 0..self.layers.len() {
            let lw = &self.layers[layer];

            // --- Attention ---
            rmsnorm(&mut xn, &h, &lw.input_ln, eps);
            linear(&mut q, &xn, &lw.q_proj, self.hidden, self.q_dim);
            linear(&mut k, &xn, &lw.k_proj, self.hidden, self.kv_dim);
            linear(&mut v, &xn, &lw.v_proj, self.hidden, self.kv_dim);

            for head in 0..self.n_heads {
                rope_in_place(&mut q[head * hd..(head + 1) * hd], pos, theta);
            }
            for head in 0..self.n_kv {
                rope_in_place(&mut k[head * hd..(head + 1) * hd], pos, theta);
                self.cache.store(
                    layer,
                    head,
                    pos,
                    &k[head * hd..(head + 1) * hd],
                    &v[head * hd..(head + 1) * hd],
                );
            }

            for head in 0..self.n_heads {
                let kvh = head / self.groups;
                let keys = self.cache.keys(layer, kvh, len);
                let values = self.cache.values(layer, kvh, len);
                sdpa_single(
                    &mut attn[head * hd..(head + 1) * hd],
                    &q[head * hd..(head + 1) * hd],
                    keys,
                    values,
                    hd,
                    len,
                    1.0 / (hd as f32).sqrt(),
                    &mut scores,
                );
            }

            linear(&mut o, &attn, &lw.o_proj, self.q_dim, self.hidden);
            for i in 0..self.hidden {
                h[i] += o[i];
            }

            // --- MLP (SwiGLU) ---
            rmsnorm(&mut xn, &h, &lw.post_ln, eps);
            linear(&mut gate, &xn, &lw.gate_proj, self.hidden, self.inter);
            linear(&mut up, &xn, &lw.up_proj, self.hidden, self.inter);
            swiglu(&mut act, &gate, &up);
            linear(&mut down, &act, &lw.down_proj, self.inter, self.hidden);
            for i in 0..self.hidden {
                h[i] += down[i];
            }
        }

        // Final norm + LM head.
        rmsnorm(&mut xn, &h, &self.final_norm, eps);
        let mut logits = vec![0.0f32; self.vocab];
        linear(&mut logits, &xn, &self.lm_head, self.hidden, self.vocab);
        Ok(logits)
    }
}

impl Model for LlamaModel {
    fn architecture(&self) -> ModelArchitecture {
        self.config.architecture
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl Decoder for LlamaModel {
    fn prefill(&mut self, input_tokens: &[u32]) -> Result<Logits> {
        if input_tokens.is_empty() {
            return Err(StrixError::invalid("prefill: empty prompt"));
        }
        self.cache.clear();
        let mut last = Vec::new();
        for &t in input_tokens {
            last = self.forward(t)?;
        }
        Ok(Logits::new(last))
    }

    fn decode_one(&mut self, token: u32) -> Result<Logits> {
        Ok(Logits::new(self.forward(token)?))
    }

    fn reset(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strix_models::RawTensor;

    // Deterministic pseudo-random fill so weights are non-degenerate but stable.
    fn lcg_fill(n: usize, seed: &mut u64) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map high bits into a small symmetric range.
            let x = ((*seed >> 33) as f32 / u32::MAX as f32) - 0.5;
            out.push(x * 0.2);
        }
        out
    }

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: ModelArchitecture::Llama,
            vocab_size: 5,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1, // GQA: 2 query heads share 1 kv head
            head_dim: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 16,
        }
    }

    fn tiny_map(cfg: &ModelConfig) -> TensorMap {
        let mut seed = 0x1234_5678u64;
        let h = cfg.hidden_size;
        let q = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let mut m = TensorMap::new();
        let mut put = |name: String, shape: Vec<usize>, n: usize, seed: &mut u64| {
            m.insert(
                name,
                RawTensor {
                    shape,
                    data: lcg_fill(n, seed),
                },
            );
        };
        put(
            "model.embed_tokens.weight".into(),
            vec![vocab, h],
            vocab * h,
            &mut seed,
        );
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            put(format!("{p}.input_layernorm.weight"), vec![h], h, &mut seed);
            put(
                format!("{p}.self_attn.q_proj.weight"),
                vec![q, h],
                q * h,
                &mut seed,
            );
            put(
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv, h],
                kv * h,
                &mut seed,
            );
            put(
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv, h],
                kv * h,
                &mut seed,
            );
            put(
                format!("{p}.self_attn.o_proj.weight"),
                vec![h, q],
                h * q,
                &mut seed,
            );
            put(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                h,
                &mut seed,
            );
            put(
                format!("{p}.mlp.gate_proj.weight"),
                vec![inter, h],
                inter * h,
                &mut seed,
            );
            put(
                format!("{p}.mlp.up_proj.weight"),
                vec![inter, h],
                inter * h,
                &mut seed,
            );
            put(
                format!("{p}.mlp.down_proj.weight"),
                vec![h, inter],
                h * inter,
                &mut seed,
            );
        }
        put("model.norm.weight".into(), vec![h], h, &mut seed);
        // No lm_head => exercise the tied-embedding path.
        m
    }

    #[test]
    fn builds_and_produces_vocab_logits() {
        let cfg = tiny_config();
        let vocab = cfg.vocab_size;
        let mut model = LlamaModel::from_tensors(cfg, tiny_map(&tiny_config()), 16).unwrap();
        let logits = model.prefill(&[1, 2, 3]).unwrap();
        assert_eq!(logits.len(), vocab);
        assert!(logits.0.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn is_deterministic() {
        let cfg = tiny_config();
        let mut a = LlamaModel::from_tensors(cfg.clone(), tiny_map(&cfg), 16).unwrap();
        let mut b = LlamaModel::from_tensors(cfg.clone(), tiny_map(&cfg), 16).unwrap();
        let la = a.prefill(&[1, 2, 3, 4]).unwrap();
        let lb = b.prefill(&[1, 2, 3, 4]).unwrap();
        assert_eq!(la.0, lb.0);
    }

    #[test]
    fn prefill_then_decode_matches_full_prefill() {
        // Feeding [1,2,3] then decode(4) must equal prefilling [1,2,3,4]:
        // both produce the logits used to predict the token after position 3.
        let cfg = tiny_config();
        let mut split = LlamaModel::from_tensors(cfg.clone(), tiny_map(&cfg), 16).unwrap();
        split.prefill(&[1, 2, 3]).unwrap();
        let after_decode = split.decode_one(4).unwrap();

        let mut whole = LlamaModel::from_tensors(cfg.clone(), tiny_map(&cfg), 16).unwrap();
        let after_prefill = whole.prefill(&[1, 2, 3, 4]).unwrap();

        for (x, y) in after_decode.0.iter().zip(after_prefill.0.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn reset_gives_reproducible_runs() {
        let cfg = tiny_config();
        let mut m = LlamaModel::from_tensors(cfg.clone(), tiny_map(&cfg), 16).unwrap();
        let first = m.prefill(&[2, 2, 1]).unwrap();
        m.reset();
        let second = m.prefill(&[2, 2, 1]).unwrap();
        assert_eq!(first.0, second.0);
    }

    #[test]
    fn missing_tensor_errors_cleanly() {
        let cfg = tiny_config();
        let mut map = tiny_map(&cfg);
        map.remove("model.norm.weight");
        assert!(LlamaModel::from_tensors(cfg, map, 16).is_err());
    }

    #[test]
    fn out_of_range_token_errors() {
        let cfg = tiny_config();
        let mut m = LlamaModel::from_tensors(cfg.clone(), tiny_map(&cfg), 16).unwrap();
        assert!(m.decode_one(999).is_err());
    }
}
