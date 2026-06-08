//! CPU reference scaled dot-product attention.
//!
//! This is single-query attention against a contiguous K/V history — exactly
//! what one decode step needs. Causal masking and grouped-query (GQA) head
//! mapping are the *caller's* responsibility: causality falls out of only
//! passing keys/values for positions `<=` the current one, and GQA falls out of
//! pointing multiple query heads at the same KV head's slices.

use crate::ops::softmax_inplace;

/// Scaled dot-product attention for one query vector against `seq_len` cached
/// key/value vectors.
///
/// - `q`: query, length `head_dim`.
/// - `keys`: row-major `[seq_len, head_dim]`.
/// - `values`: row-major `[seq_len, head_dim]`.
/// - `out`: result, length `head_dim`.
/// - `scratch`: reusable buffer of length `seq_len` for attention weights.
///
/// Computes `softmax((q·kᵀ) * scale) · v`. The caller supplies `scale` (e.g.
/// `1/sqrt(head_dim)` for Llama, `1.0` for Gemma-4 which relies on QK-norm).
/// Passing only the keys/values up to the current position gives causal attention.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_single(
    out: &mut [f32],
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    head_dim: usize,
    seq_len: usize,
    scale: f32,
    scratch: &mut [f32],
) {
    debug_assert_eq!(out.len(), head_dim);
    debug_assert_eq!(q.len(), head_dim);
    debug_assert_eq!(keys.len(), seq_len * head_dim);
    debug_assert_eq!(values.len(), seq_len * head_dim);
    debug_assert_eq!(scratch.len(), seq_len);

    // scores[t] = (q · keys[t]) * scale
    for t in 0..seq_len {
        let k = &keys[t * head_dim..(t + 1) * head_dim];
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * k[d];
        }
        scratch[t] = dot * scale;
    }

    softmax_inplace(&mut scratch[..seq_len]);

    // out = sum_t scores[t] * values[t]
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for t in 0..seq_len {
        let w = scratch[t];
        let val = &values[t * head_dim..(t + 1) * head_dim];
        for d in 0..head_dim {
            out[d] += w * val[d];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn single_position_returns_that_value() {
        // seq_len=1: softmax of one score is 1.0, so out == values[0].
        let q = [0.3, -0.7];
        let keys = [1.0, 2.0];
        let values = [5.0, -9.0];
        let mut out = [0.0; 2];
        let mut scratch = [0.0; 1];
        sdpa_single(
            &mut out,
            &q,
            &keys,
            &values,
            2,
            1,
            1.0 / 2f32.sqrt(),
            &mut scratch,
        );
        assert!(approx(out[0], 5.0, 1e-6));
        assert!(approx(out[1], -9.0, 1e-6));
    }

    #[test]
    fn equal_scores_average_values() {
        // Two identical keys => equal scores => uniform softmax => mean of values.
        let q = [1.0, 1.0];
        let keys = [0.5, 0.5, 0.5, 0.5];
        let values = [2.0, 4.0, 6.0, 8.0]; // v0=[2,4], v1=[6,8]
        let mut out = [0.0; 2];
        let mut scratch = [0.0; 2];
        sdpa_single(
            &mut out,
            &q,
            &keys,
            &values,
            2,
            2,
            1.0 / 2f32.sqrt(),
            &mut scratch,
        );
        assert!(approx(out[0], 4.0, 1e-6)); // mean(2,6)
        assert!(approx(out[1], 6.0, 1e-6)); // mean(4,8)
    }

    #[test]
    fn dominant_key_dominates_output() {
        // One key aligns strongly with q, the other is orthogonal/negative.
        // Softmax should concentrate weight on the aligned key's value.
        let head_dim = 2;
        let q = [10.0, 0.0];
        let keys = [10.0, 0.0, -10.0, 0.0]; // k0 aligned, k1 anti-aligned
        let values = [1.0, 1.0, 99.0, 99.0];
        let mut out = [0.0; 2];
        let mut scratch = [0.0; 2];
        sdpa_single(
            &mut out,
            &q,
            &keys,
            &values,
            head_dim,
            2,
            1.0 / 2f32.sqrt(),
            &mut scratch,
        );
        // Output should be very close to v0 = [1,1], not v1.
        assert!(approx(out[0], 1.0, 1e-3), "{}", out[0]);
        assert!(approx(out[1], 1.0, 1e-3), "{}", out[1]);
    }

    #[test]
    fn output_is_convex_combination_of_values() {
        // Each output element must lie within [min,max] of the value column,
        // since attention weights are a probability distribution.
        let head_dim = 1;
        let q = [0.4];
        let keys = [0.1, 0.9, -0.3];
        let values = [3.0, 7.0, 5.0];
        let mut out = [0.0; 1];
        let mut scratch = [0.0; 3];
        sdpa_single(&mut out, &q, &keys, &values, head_dim, 3, 1.0, &mut scratch);
        assert!(out[0] >= 3.0 - 1e-6 && out[0] <= 7.0 + 1e-6, "{}", out[0]);
    }
}
