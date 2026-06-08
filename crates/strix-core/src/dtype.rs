//! Data types Strix tensors can hold.
//!
//! Phase 1 targets `f32` (compute) and `f16`/`bf16` (storage). Quantized
//! variants are placeholders for later phases and intentionally carry no
//! implementation yet.

use std::fmt;

/// Element type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit IEEE float — the reference compute type.
    F32,
    /// 16-bit IEEE float (half).
    F16,
    /// bfloat16.
    BF16,
    /// 8-bit symmetric int quantization (planned, Phase 2).
    Q8_0,
    /// 4-bit k-quant family (planned, Phase 3).
    Q4K,
}

impl DType {
    /// Size in bytes of one element, when one is well-defined.
    ///
    /// Block-quantized types do not have a fixed per-element byte size, so they
    /// return `None`.
    pub fn size_in_bytes(self) -> Option<usize> {
        match self {
            DType::F32 => Some(4),
            DType::F16 | DType::BF16 => Some(2),
            DType::Q8_0 | DType::Q4K => None,
        }
    }

    /// Whether this dtype is a (block) quantized format.
    pub fn is_quantized(self) -> bool {
        matches!(self, DType::Q8_0 | DType::Q4K)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::Q8_0 => "q8_0",
            DType::Q4K => "q4_k",
        };
        f.write_str(s)
    }
}
