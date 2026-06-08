//! `strix-core` — the shared vocabulary of the Strix LLM runner.
//!
//! This crate holds traits and plain data types only. It does no I/O and runs
//! no math beyond trivial helpers. Backends (`strix-backend-*`) and model
//! definitions (`strix-models`) build on top of it.
//!
//! The central abstractions are [`backend::Backend`], [`backend::Model`], and
//! [`backend::Decoder`]. Everything else (dtype, tensor, sampler, kv_cache,
//! benchmark) supports those.

pub mod accel;
pub mod backend;
pub mod benchmark;
pub mod device;
pub mod dtype;
pub mod error;
pub mod kv_cache;
pub mod model_config;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;

// Re-export the most-used items at the crate root.
pub use accel::{GpuDecodeConfig, GpuLayerCfg, WeightAccel};
pub use backend::{Backend, Decoder, Model};
pub use device::{DeviceInfo, DeviceKind};
pub use dtype::DType;
pub use error::{Result, StrixError};
pub use model_config::{ModelArchitecture, ModelConfig};
pub use sampler::{GreedySampler, Logits, Sampler, SamplingConfig};
pub use tensor::{HostTensor, Shape};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let s = GreedySampler;
        let logits = Logits::new(vec![0.1, 0.9, 0.3, -1.0]);
        assert_eq!(s.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn greedy_errors_on_empty() {
        let s = GreedySampler;
        assert!(s.sample(&Logits::new(vec![])).is_err());
    }

    #[test]
    fn host_tensor_shape_checks() {
        let ok = HostTensor::from_vec(Shape::new(vec![2, 3]), vec![0.0; 6]);
        assert!(ok.is_ok());
        let bad = HostTensor::from_vec(Shape::new(vec![2, 3]), vec![0.0; 5]);
        assert!(bad.is_err());
    }

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::F32.size_in_bytes(), Some(4));
        assert_eq!(DType::F16.size_in_bytes(), Some(2));
        assert_eq!(DType::Q4K.size_in_bytes(), None);
        assert!(DType::Q8_0.is_quantized());
    }

    #[test]
    fn gqa_group_factor() {
        let cfg = ModelConfig {
            architecture: ModelArchitecture::Llama,
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 8192,
        };
        assert_eq!(cfg.gqa_groups(), 4);
    }
}
