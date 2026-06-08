//! Not-yet-implemented commands.
//!
//! These return a clean `NotImplemented` error rather than panicking, so the
//! CLI surface is complete and honest about what works today.

use std::path::Path;

use anyhow::Result;
use strix_core::error::StrixError;

/// `strix chat` — interactive loop, after `generate` works.
pub fn chat(model: &Path) -> Result<()> {
    tracing::info!(model = %model.display(), "chat requested");
    Err(StrixError::todo("chat: planned after the generate path works").into())
}

/// `strix bench` — real benchmarking once a model can run. Use `bench-dummy` now.
pub fn bench(model: &Path, prompt_len: usize, gen_len: usize) -> Result<()> {
    tracing::info!(
        model = %model.display(),
        prompt_len,
        gen_len,
        "bench requested"
    );
    Err(StrixError::todo(
        "bench: needs real inference (Milestone 2). Try `strix bench-dummy` for now",
    )
    .into())
}
