//! Logits and sampling.
//!
//! Phase 1 ships greedy (argmax) sampling only. Temperature / top-k / top-p are
//! placeholders documented here so the API shape is stable.

use crate::error::{Result, StrixError};

/// Raw output distribution over the vocabulary for a single position.
#[derive(Debug, Clone)]
pub struct Logits(pub Vec<f32>);

impl Logits {
    /// Wrap a logits buffer.
    pub fn new(values: Vec<f32>) -> Self {
        Logits(values)
    }

    /// Vocabulary length.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the logits buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Sampling configuration. Only greedy is honored in Phase 1.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Softmax temperature. `0.0` (or `None` semantics) means greedy.
    pub temperature: f32,
    /// Top-k cutoff (0 = disabled). Reserved for later phases.
    pub top_k: usize,
    /// Top-p / nucleus cutoff (1.0 = disabled). Reserved for later phases.
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// Turns logits into a chosen token id.
pub trait Sampler: Send + Sync {
    /// Select the next token id from the logits.
    fn sample(&self, logits: &Logits) -> Result<u32>;
}

/// Deterministic argmax sampler.
#[derive(Debug, Default, Clone)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&self, logits: &Logits) -> Result<u32> {
        if logits.is_empty() {
            return Err(StrixError::invalid("cannot sample from empty logits"));
        }
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.0.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        Ok(best_idx as u32)
    }
}
