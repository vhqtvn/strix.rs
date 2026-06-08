//! Backend, model, and decoder traits.
//!
//! These are the central abstractions. They are deliberately minimal and *will*
//! change as the CPU reference path lands in Milestone 2. Keep them small.

use crate::device::DeviceInfo;
use crate::error::Result;
use crate::model_config::{ModelArchitecture, ModelConfig};
use crate::sampler::Logits;

/// A compute backend (CPU, Vulkan/iGPU, NPU).
///
/// A backend owns a device and knows how to construct decoders for models it
/// supports. Phase 1 only requires identity + device introspection.
pub trait Backend: Send + Sync {
    /// Stable backend name, e.g. `"cpu"` or `"vulkan"`.
    fn name(&self) -> &'static str;

    /// Describe the device this backend drives.
    fn device_info(&self) -> DeviceInfo;
}

/// A loaded model: its architecture and normalized config.
pub trait Model: Send + Sync {
    /// Architecture family.
    fn architecture(&self) -> ModelArchitecture;

    /// Normalized hyperparameters.
    fn config(&self) -> &ModelConfig;
}

/// A streaming decoder over a single sequence.
///
/// The typical loop is: `prefill(prompt)` once, then `decode_one(token)` per
/// generated step. Implementations own their own KV cache.
pub trait Decoder: Send {
    /// Ingest the prompt tokens and populate the KV cache.
    ///
    /// Returns the logits for the final prompt position (used to sample the
    /// first generated token).
    fn prefill(&mut self, input_tokens: &[u32]) -> Result<Logits>;

    /// Advance one step with the previously sampled token; return next logits.
    fn decode_one(&mut self, token: u32) -> Result<Logits>;

    /// Reset decoder state (KV cache, positions) for a fresh sequence.
    fn reset(&mut self);
}
