//! Architecture-agnostic model configuration.
//!
//! This captures the hyperparameters common to Llama-like decoder-only
//! transformers (Llama 3.x, Mistral, Qwen2.5/3). Architecture-specific config
//! parsing lives in `strix-models`; this is the normalized shape the rest of
//! the engine consumes.

use serde::{Deserialize, Serialize};

/// Supported / planned model architecture families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Llama 3.x style decoder.
    Llama,
    /// Mistral style decoder.
    Mistral,
    /// Qwen2.5 / Qwen3 style decoder.
    Qwen,
    /// Recognized shape we don't specialize yet.
    Unknown,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ModelArchitecture::Llama => "llama",
            ModelArchitecture::Mistral => "mistral",
            ModelArchitecture::Qwen => "qwen",
            ModelArchitecture::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

/// Normalized hyperparameters for a Llama-like decoder-only transformer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Architecture family.
    pub architecture: ModelArchitecture,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden / model dimension.
    pub hidden_size: usize,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: usize,
    /// Number of transformer blocks.
    pub num_hidden_layers: usize,
    /// Number of attention (query) heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA). Equals `num_attention_heads` for MHA.
    pub num_key_value_heads: usize,
    /// Per-head dimension (`hidden_size / num_attention_heads` unless overridden).
    pub head_dim: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency (theta).
    pub rope_theta: f32,
    /// Maximum context length the model was trained/configured for.
    pub max_position_embeddings: usize,
}

impl ModelConfig {
    /// Heads-per-KV-group factor for grouped-query attention.
    pub fn gqa_groups(&self) -> usize {
        if self.num_key_value_heads == 0 {
            1
        } else {
            self.num_attention_heads / self.num_key_value_heads
        }
    }
}
