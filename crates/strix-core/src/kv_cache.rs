//! Key/value cache interface.
//!
//! Phase 1 only needs the abstraction, not an implementation. The contiguous
//! per-layer cache will land with the CPU reference decoder (Milestone 2);
//! paged/block caches are a much later concern (and were studied in
//! `docs/mistral-rs-notes.md`).

/// A KV cache holds per-layer key/value tensors across decode steps.
///
/// The trait is intentionally tiny for now — it captures position bookkeeping
/// which every implementation needs. Tensor accessors will be added once the
/// CPU backend defines its concrete storage.
pub trait KvCache: Send {
    /// Number of layers this cache stores.
    fn num_layers(&self) -> usize;

    /// Current sequence length (number of cached positions).
    fn seq_len(&self) -> usize;

    /// Maximum positions this cache can hold.
    fn capacity(&self) -> usize;

    /// Reset to empty, keeping allocated capacity.
    fn clear(&mut self);
}
