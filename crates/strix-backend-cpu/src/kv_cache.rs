//! Contiguous CPU key/value cache.
//!
//! Layout is `[layer][kv_head][max_seq][head_dim]`, all `f32`. The crucial
//! property: for a fixed `(layer, kv_head)`, positions `0..seq` form a
//! contiguous `[seq, head_dim]` slice — exactly the shape
//! [`crate::attention::sdpa_single`] expects, with no copying.
//!
//! This is a single-sequence cache (no batching, no paging). Those are much
//! later concerns; correctness first.

use strix_core::error::{Result, StrixError};
use strix_core::kv_cache::KvCache;

/// A single-sequence, contiguous KV cache.
#[derive(Debug)]
pub struct CpuKvCache {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    seq: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl CpuKvCache {
    /// Allocate a cache sized for the given model geometry.
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let total = num_layers * num_kv_heads * max_seq * head_dim;
        CpuKvCache {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq,
            seq: 0,
            k: vec![0.0; total],
            v: vec![0.0; total],
        }
    }

    /// Per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Byte offset (in elements) of `(layer, head, pos)` row start.
    #[inline]
    fn row(&self, layer: usize, head: usize, pos: usize) -> usize {
        (((layer * self.num_kv_heads + head) * self.max_seq) + pos) * self.head_dim
    }

    /// Advance the sequence length by one position, returning that position's
    /// index. Call once per token *after* writing all layers/heads for it.
    pub fn advance(&mut self) -> Result<usize> {
        if self.seq >= self.max_seq {
            return Err(StrixError::invalid(format!(
                "kv cache full: max_seq={} reached",
                self.max_seq
            )));
        }
        let pos = self.seq;
        self.seq += 1;
        Ok(pos)
    }

    /// Write the K and V vectors for `(layer, kv_head)` at `pos`.
    ///
    /// `k_vec` and `v_vec` must each be `head_dim` long. `pos` must be `< max_seq`.
    pub fn store(&mut self, layer: usize, head: usize, pos: usize, k_vec: &[f32], v_vec: &[f32]) {
        debug_assert_eq!(k_vec.len(), self.head_dim);
        debug_assert_eq!(v_vec.len(), self.head_dim);
        let start = self.row(layer, head, pos);
        self.k[start..start + self.head_dim].copy_from_slice(k_vec);
        self.v[start..start + self.head_dim].copy_from_slice(v_vec);
    }

    /// Keys for `(layer, kv_head)` over positions `0..len` as a contiguous
    /// `[len, head_dim]` slice.
    pub fn keys(&self, layer: usize, head: usize, len: usize) -> &[f32] {
        let start = self.row(layer, head, 0);
        &self.k[start..start + len * self.head_dim]
    }

    /// Values for `(layer, kv_head)` over positions `0..len`.
    pub fn values(&self, layer: usize, head: usize, len: usize) -> &[f32] {
        let start = self.row(layer, head, 0);
        &self.v[start..start + len * self.head_dim]
    }
}

impl KvCache for CpuKvCache {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn seq_len(&self) -> usize {
        self.seq
    }

    fn capacity(&self) -> usize {
        self.max_seq
    }

    fn clear(&mut self) {
        // Keep the allocation; just reset the logical length. Stale data is
        // never read because reads are bounded by `seq`.
        self.seq = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_read_back_contiguous() {
        // 1 layer, 1 kv head, head_dim=2, capacity 3.
        let mut c = CpuKvCache::new(1, 1, 2, 3);
        assert_eq!(c.seq_len(), 0);

        let p0 = c.advance().unwrap();
        c.store(0, 0, p0, &[1.0, 2.0], &[10.0, 20.0]);
        let p1 = c.advance().unwrap();
        c.store(0, 0, p1, &[3.0, 4.0], &[30.0, 40.0]);

        assert_eq!(c.seq_len(), 2);
        assert_eq!(c.keys(0, 0, 2), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.values(0, 0, 2), &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn heads_and_layers_are_isolated() {
        let mut c = CpuKvCache::new(2, 2, 2, 4);
        let p = c.advance().unwrap();
        c.store(0, 0, p, &[1.0, 1.0], &[1.0, 1.0]);
        c.store(0, 1, p, &[2.0, 2.0], &[2.0, 2.0]);
        c.store(1, 0, p, &[3.0, 3.0], &[3.0, 3.0]);
        assert_eq!(c.keys(0, 0, 1), &[1.0, 1.0]);
        assert_eq!(c.keys(0, 1, 1), &[2.0, 2.0]);
        assert_eq!(c.keys(1, 0, 1), &[3.0, 3.0]);
        // Untouched (layer 1, head 1) stays zero.
        assert_eq!(c.keys(1, 1, 1), &[0.0, 0.0]);
    }

    #[test]
    fn overflow_is_an_error_not_a_panic() {
        let mut c = CpuKvCache::new(1, 1, 1, 2);
        assert!(c.advance().is_ok());
        assert!(c.advance().is_ok());
        assert!(c.advance().is_err());
    }

    #[test]
    fn clear_resets_length_keeps_capacity() {
        let mut c = CpuKvCache::new(1, 1, 1, 2);
        c.advance().unwrap();
        c.clear();
        assert_eq!(c.seq_len(), 0);
        assert_eq!(c.capacity(), 2);
        // reusable after clear
        assert!(c.advance().is_ok());
    }
}
