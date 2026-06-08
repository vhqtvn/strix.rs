//! Error type shared across Strix crates.
//!
//! We use a single `thiserror`-derived enum in the library crates so callers can
//! match on failure kinds. The CLI uses `anyhow` on top of this for ergonomic
//! reporting.

use std::result::Result as StdResult;

/// Convenience alias used throughout the workspace.
pub type Result<T> = StdResult<T, StrixError>;

/// The top-level error type for Strix.
#[derive(Debug, thiserror::Error)]
pub enum StrixError {
    /// A feature exists in the API surface but is not implemented yet.
    #[error("not implemented yet: {0}")]
    NotImplemented(String),

    /// A model file, tokenizer, or config could not be found or read.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A config or metadata file failed to parse.
    #[error("parse error: {0}")]
    Parse(String),

    /// The model architecture or format is recognized but unsupported.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A tensor shape / dtype mismatch or other invariant violation.
    #[error("invalid: {0}")]
    Invalid(String),

    /// A backend (CPU/Vulkan/NPU) reported a device-level failure.
    #[error("backend error ({backend}): {message}")]
    Backend {
        /// Name of the backend that failed.
        backend: &'static str,
        /// Human-readable detail.
        message: String,
    },
}

impl StrixError {
    /// Helper for the common "not implemented" case.
    pub fn todo(what: impl Into<String>) -> Self {
        StrixError::NotImplemented(what.into())
    }

    /// Helper for parse failures.
    pub fn parse(msg: impl Into<String>) -> Self {
        StrixError::Parse(msg.into())
    }

    /// Helper for unsupported formats / architectures.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        StrixError::Unsupported(msg.into())
    }

    /// Helper for invariant violations.
    pub fn invalid(msg: impl Into<String>) -> Self {
        StrixError::Invalid(msg.into())
    }

    /// Helper for backend/device-level failures.
    pub fn backend(msg: impl Into<String>) -> Self {
        StrixError::Backend {
            backend: "vulkan",
            message: msg.into(),
        }
    }
}
