//! Minimal tensor types for the CPU reference path.
//!
//! This is intentionally simple: a row-major shape plus an owned `f32` buffer.
//! It is *not* a general N-d array library and is *not* fast. Backends are free
//! to keep their own internal representations; this type exists so the core
//! crate can describe shapes and exchange small host-side buffers.

use crate::error::{Result, StrixError};

/// A row-major tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    /// Construct a shape from dimensions.
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Shape(dims.into())
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total element count (product of dims; empty shape == 1 scalar).
    pub fn numel(&self) -> usize {
        self.0
            .iter()
            .product::<usize>()
            .max(if self.0.is_empty() { 1 } else { 0 })
    }

    /// Dimensions slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// An owned, row-major `f32` host tensor.
///
/// Used by the CPU reference backend and for exchanging small buffers. Heavy
/// math should live in the backends, not here.
#[derive(Debug, Clone)]
pub struct HostTensor {
    shape: Shape,
    data: Vec<f32>,
}

impl HostTensor {
    /// Create a tensor from a shape and matching data buffer.
    pub fn from_vec(shape: Shape, data: Vec<f32>) -> Result<Self> {
        if shape.numel() != data.len() {
            return Err(StrixError::invalid(format!(
                "shape {shape} has {} elements but buffer has {}",
                shape.numel(),
                data.len()
            )));
        }
        Ok(HostTensor { shape, data })
    }

    /// Create a zero-filled tensor.
    pub fn zeros(shape: Shape) -> Self {
        let n = shape.numel();
        HostTensor {
            shape,
            data: vec![0.0; n],
        }
    }

    /// Shape accessor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Read-only data slice.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable data slice.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}
