//! `strix-models` — architecture configs and non-loading model inspection.
//!
//! Phase 1 covers Llama-like families (Llama 3.x, Mistral, Qwen2.5/3) which
//! share a `config.json` layout. Actual weight loading (safetensors, then GGUF)
//! and per-architecture forward passes land in later milestones; this crate
//! currently provides config normalization and directory inspection.

pub mod ggml_quant;
pub mod gguf;
pub mod hf_config;
pub mod inspect;
pub mod safetensors_loader;
pub mod tokenizer;

pub use ggml_quant::{dequantize, dequantize_into, GgmlType};
pub use hf_config::HfConfig;
pub use inspect::{inspect_model, ModelInspection, WeightFormat};
pub use safetensors_loader::{load_safetensors, RawTensor, TensorMap};
pub use tokenizer::StrixTokenizer;

#[cfg(test)]
mod tests {
    use super::*;
    use strix_core::model_config::ModelArchitecture;

    #[test]
    fn parses_llama_like_config() {
        let json = br#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": 128256,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 8192
        }"#;
        let hf = HfConfig::from_json_slice(json).unwrap();
        assert_eq!(hf.architecture(), ModelArchitecture::Llama);
        let cfg = hf.to_model_config().unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_key_value_heads, 8);
        // head_dim derived from hidden/heads.
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.gqa_groups(), 4);
    }

    #[test]
    fn detects_qwen() {
        let json = br#"{"model_type": "qwen2", "vocab_size": 1, "hidden_size": 8,
            "intermediate_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2}"#;
        let hf = HfConfig::from_json_slice(json).unwrap();
        assert_eq!(hf.architecture(), ModelArchitecture::Qwen);
        // kv heads default to attention heads (MHA) when absent.
        let cfg = hf.to_model_config().unwrap();
        assert_eq!(cfg.num_key_value_heads, 2);
    }

    #[test]
    fn missing_required_field_errors() {
        let json = br#"{"model_type": "llama", "hidden_size": 8}"#;
        let hf = HfConfig::from_json_slice(json).unwrap();
        assert!(hf.to_model_config().is_err());
    }
}
