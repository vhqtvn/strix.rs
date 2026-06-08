//! HuggingFace `config.json` parsing for Llama-like decoders.
//!
//! This reads the subset of fields Strix needs and normalizes them into
//! [`strix_core::ModelConfig`]. Qwen/Mistral share the Llama field layout, so a
//! single permissive struct covers all three families in Phase 1.

use serde::Deserialize;
use strix_core::error::{Result, StrixError};
use strix_core::model_config::{ModelArchitecture, ModelConfig};

/// Raw subset of a HuggingFace `config.json`.
///
/// Fields are optional where families disagree so parsing is forgiving; missing
/// required fields are reported when normalizing.
#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    /// `architectures` array, e.g. `["LlamaForCausalLM"]`.
    #[serde(default)]
    pub architectures: Vec<String>,
    /// `model_type`, e.g. `"llama"`, `"mistral"`, `"qwen2"`.
    #[serde(default)]
    pub model_type: Option<String>,

    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    /// BOS token id (always a scalar in practice).
    #[serde(default)]
    pub bos_token_id: Option<serde_json::Value>,
    /// EOS token id — a scalar for most models, an array for Llama 3 chat.
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

/// Extract a single `u32` id from a config field that may be a number or an
/// array of numbers (returns the first element of an array).
fn first_id(v: &serde_json::Value) -> Option<u32> {
    match v {
        serde_json::Value::Number(n) => n.as_u64().map(|x| x as u32),
        serde_json::Value::Array(a) => a.iter().find_map(first_id),
        _ => None,
    }
}

impl HfConfig {
    /// Parse a `config.json` byte buffer.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| StrixError::parse(format!("config.json: {e}")))
    }

    /// Resolved BOS token id, if present.
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_token_id.as_ref().and_then(first_id)
    }

    /// Resolved EOS token id (first one, if an array), if present.
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_token_id.as_ref().and_then(first_id)
    }

    /// Infer the architecture family from `model_type` / `architectures`.
    pub fn architecture(&self) -> ModelArchitecture {
        let hint = self
            .model_type
            .clone()
            .or_else(|| self.architectures.first().cloned())
            .unwrap_or_default()
            .to_lowercase();

        if hint.contains("qwen") {
            ModelArchitecture::Qwen
        } else if hint.contains("mistral") {
            ModelArchitecture::Mistral
        } else if hint.contains("llama") {
            ModelArchitecture::Llama
        } else {
            ModelArchitecture::Unknown
        }
    }

    /// Normalize into the engine's [`ModelConfig`], filling sensible defaults.
    pub fn to_model_config(&self) -> Result<ModelConfig> {
        let required = |v: Option<usize>, name: &str| -> Result<usize> {
            v.ok_or_else(|| StrixError::parse(format!("config.json missing `{name}`")))
        };

        let hidden_size = required(self.hidden_size, "hidden_size")?;
        let num_attention_heads = required(self.num_attention_heads, "num_attention_heads")?;
        // GQA: default kv heads to query heads (MHA) when absent.
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(num_attention_heads);
        // head_dim defaults to hidden/heads when not explicitly provided.
        let head_dim = self.head_dim.unwrap_or_else(|| {
            if num_attention_heads == 0 {
                0
            } else {
                hidden_size / num_attention_heads
            }
        });

        Ok(ModelConfig {
            architecture: self.architecture(),
            vocab_size: required(self.vocab_size, "vocab_size")?,
            hidden_size,
            intermediate_size: required(self.intermediate_size, "intermediate_size")?,
            num_hidden_layers: required(self.num_hidden_layers, "num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-5),
            rope_theta: self.rope_theta.unwrap_or(10_000.0),
            max_position_embeddings: self.max_position_embeddings.unwrap_or(4096),
        })
    }
}
