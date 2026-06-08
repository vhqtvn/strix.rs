//! CPU reference math primitives.
//!
//! These are the slow, readable building blocks of the reference decoder. They
//! operate on `f32` slices, do no allocation in the hot path where avoidable,
//! and prioritize matching HuggingFace numerics exactly (we load HF
//! safetensors, so RoPE/RMSNorm conventions must agree).
//!
//! Nothing here is optimized. Correctness first; the Vulkan backend is where
//! speed lives later.

/// RMSNorm: `y = x / sqrt(mean(x^2) + eps) * weight`.
///
/// `x`, `weight`, and `out` are all length `dim`. Matches HF `LlamaRMSNorm`
/// (the mean is over the full hidden dimension, normalization is in f32).
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len();
    let mut sumsq = 0.0f32;
    for &v in x {
        sumsq += v * v;
    }
    let mean = sumsq / n as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Matrix-vector product for a bias-free linear layer: `y = W x`.
///
/// `w` is row-major `[out_dim, in_dim]` — i.e. PyTorch / safetensors
/// `nn.Linear.weight` layout, where row `o` holds the weights producing
/// `y[o]`. `x` is length `in_dim`, `y` is length `out_dim`.
pub fn linear(y: &mut [f32], x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(y.len(), out_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(w.len(), in_dim * out_dim);
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
}

/// In-place numerically-stable softmax over a slice.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

/// SiLU / swish activation: `silu(x) = x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU feed-forward combine: `out = silu(gate) * up`, elementwise.
///
/// `gate`, `up`, and `out` are all the same length (the intermediate dim).
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    debug_assert_eq!(out.len(), gate.len());
    debug_assert_eq!(gate.len(), up.len());
    for i in 0..out.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// In-place plain RMSNorm: `v = v / sqrt(mean(v^2) + eps) * weight`.
///
/// Used for Gemma-4 per-head QK-norm. Note: Gemma's GGUF bakes the `(1 + delta)`
/// into the stored norm weights at conversion time, so the *plain* multiply (not
/// `(1 + weight)`) is correct here.
pub fn rmsnorm_inplace(v: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(v.len(), weight.len());
    let n = v.len();
    let mut sumsq = 0.0f32;
    for &x in v.iter() {
        sumsq += x * x;
    }
    let scale = 1.0 / (sumsq / n as f32 + eps).sqrt();
    for i in 0..n {
        v[i] = v[i] * scale * weight[i];
    }
}

/// GELU with the tanh approximation (`gelu_pytorch_tanh`), as used by Gemma.
///
/// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 x³)))`.
pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// GeGLU feed-forward combine: `out = gelu_tanh(gate) * up`, elementwise.
pub fn geglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    debug_assert_eq!(out.len(), gate.len());
    debug_assert_eq!(gate.len(), up.len());
    for i in 0..out.len() {
        out[i] = gelu_tanh(gate[i]) * up[i];
    }
}

/// Logit / attention soft-cap in place: `x = cap * tanh(x / cap)`.
///
/// Used by Gemma to bound logits (Gemma 4 caps final logits at 30).
pub fn softcap_inplace(x: &mut [f32], cap: f32) {
    if cap <= 0.0 {
        return;
    }
    let inv = 1.0 / cap;
    for v in x.iter_mut() {
        *v = cap * (*v * inv).tanh();
    }
}

/// Apply rotary position embedding (RoPE) in place to one head's vector.
///
/// Uses the HuggingFace "rotate_half" convention (GPT-NeoX style), where the
/// `head_dim` is split into two halves and rotated as:
///
/// ```text
/// x[j]        = x1*cos - x2*sin
/// x[j+half]   = x2*cos + x1*sin
/// ```
///
/// with `cos`/`sin` derived from `freq = pos * theta^(-2j/head_dim)`. This
/// matches `LlamaRotaryEmbedding` (NOT llama.cpp's interleaved memory layout),
/// because we load HF-format safetensors.
///
/// `vec` length must equal `head_dim`, which must be even.
pub fn rope_in_place(vec: &mut [f32], pos: usize, theta: f32) {
    rope_in_place_ff(vec, pos, theta, None)
}

/// RoPE (NEOX/rotate_half) with optional per-frequency `freq_factors`
/// (llama.cpp "proportional RoPE": `inv_freq[j] /= freq_factors[j]`). Gemma-4's
/// global/full-attention layers supply `rope_freqs`; others pass `None`.
pub fn rope_in_place_ff(vec: &mut [f32], pos: usize, theta: f32, freq_factors: Option<&[f32]>) {
    let head_dim = vec.len();
    debug_assert_eq!(head_dim % 2, 0, "head_dim must be even");
    let half = head_dim / 2;
    for j in 0..half {
        // inv_freq[j] = theta^(-2j/head_dim), optionally divided by freq_factors.
        let mut inv_freq = theta.powf(-(2.0 * j as f32) / head_dim as f32);
        if let Some(ff) = freq_factors {
            inv_freq /= ff[j];
        }
        let angle = pos as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let x1 = vec[j];
        let x2 = vec[j + half];
        vec[j] = x1 * cos - x2 * sin;
        vec[j + half] = x2 * cos + x1 * sin;
    }
}

/// RoPE with the interleaved (GPT-J / GGML "NORM") convention: rotates adjacent
/// pairs `(2i, 2i+1)` rather than the half-split `(i, i+d/2)`.
pub fn rope_in_place_interleaved(vec: &mut [f32], pos: usize, theta: f32) {
    let head_dim = vec.len();
    debug_assert_eq!(head_dim % 2, 0, "head_dim must be even");
    let half = head_dim / 2;
    for i in 0..half {
        let inv_freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
        let angle = pos as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let x0 = vec[2 * i];
        let x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * cos - x1 * sin;
        vec[2 * i + 1] = x0 * sin + x1 * cos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn rmsnorm_matches_hand_computation() {
        // x = [1,2,3,4], unit weight, eps=0.
        // mean(x^2) = 30/4 = 7.5, scale = 1/sqrt(7.5) = 0.365148.
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 4];
        let mut out = [0.0; 4];
        rmsnorm(&mut out, &x, &w, 0.0);
        assert!(approx(out[0], 0.365148, 1e-5), "{}", out[0]);
        assert!(approx(out[3], 1.460593, 1e-5), "{}", out[3]);
    }

    #[test]
    fn rmsnorm_weight_scales() {
        let x = [2.0, 2.0];
        let w = [3.0, 0.5];
        let mut out = [0.0; 2];
        // mean sq = 4, scale = 0.5; out = x*0.5*w = [3, 0.5]
        rmsnorm(&mut out, &x, &w, 0.0);
        assert!(approx(out[0], 3.0, 1e-6));
        assert!(approx(out[1], 0.5, 1e-6));
    }

    #[test]
    fn linear_row_major_layout() {
        // W = [[1,0],[0,1],[1,1]] (out=3,in=2), x=[1,2] => y=[1,2,3]
        let w = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let x = [1.0, 2.0];
        let mut y = [0.0; 3];
        linear(&mut y, &x, &w, 2, 3);
        assert_eq!(y, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn softmax_uniform_and_sums_to_one() {
        let mut x = [1.0, 1.0, 1.0];
        softmax_inplace(&mut x);
        for &v in &x {
            assert!(approx(v, 1.0 / 3.0, 1e-6));
        }
        let mut y = [0.0, 0.0];
        softmax_inplace(&mut y);
        assert!(approx(y[0], 0.5, 1e-6) && approx(y[1], 0.5, 1e-6));
    }

    #[test]
    fn softmax_is_shift_invariant_and_stable() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [1001.0, 1002.0, 1003.0];
        softmax_inplace(&mut a);
        softmax_inplace(&mut b);
        for i in 0..3 {
            assert!(approx(a[i], b[i], 1e-6));
        }
    }

    #[test]
    fn silu_known_values() {
        assert!(approx(silu(0.0), 0.0, 1e-7));
        // silu(1) = 1*sigmoid(1) = 0.731059
        assert!(approx(silu(1.0), 0.731059, 1e-5));
        // large positive ~ identity
        assert!(approx(silu(20.0), 20.0, 1e-3));
    }

    #[test]
    fn swiglu_combines() {
        let gate = [0.0, 1.0];
        let up = [5.0, 2.0];
        let mut out = [0.0; 2];
        swiglu(&mut out, &gate, &up);
        assert!(approx(out[0], 0.0, 1e-6));
        assert!(approx(out[1], 0.731059 * 2.0, 1e-5));
    }

    #[test]
    fn rmsnorm_inplace_matches_plain_rmsnorm() {
        // Plain RMSNorm in place: x=[2,2], weight=[3,0.5] => normed=[1,1]*w=[3,0.5].
        let mut v = [2.0, 2.0];
        rmsnorm_inplace(&mut v, &[3.0, 0.5], 0.0);
        assert!(approx(v[0], 3.0, 1e-6) && approx(v[1], 0.5, 1e-6));
    }

    #[test]
    fn gelu_tanh_known_values() {
        assert!(approx(gelu_tanh(0.0), 0.0, 1e-7));
        // gelu(1) ≈ 0.8412 (tanh approx)
        assert!(approx(gelu_tanh(1.0), 0.8412, 1e-3));
        // large positive ~ identity, large negative ~ 0
        assert!(approx(gelu_tanh(10.0), 10.0, 1e-3));
        assert!(gelu_tanh(-10.0).abs() < 1e-3);
    }

    #[test]
    fn softcap_bounds_and_is_near_identity_for_small() {
        let mut x = [1000.0, -1000.0, 0.5];
        softcap_inplace(&mut x, 30.0);
        assert!(x[0] <= 30.0 && x[0] > 29.0);
        assert!(x[1] >= -30.0 && x[1] < -29.0);
        // small value barely changes
        assert!(approx(x[2], 30.0 * (0.5f32 / 30.0).tanh(), 1e-6));
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut v = [0.5, -1.0, 2.0, 0.25];
        let orig = v;
        rope_in_place(&mut v, 0, 10000.0);
        for i in 0..4 {
            assert!(approx(v[i], orig[i], 1e-6));
        }
    }

    #[test]
    fn rope_rotates_first_pair_as_expected() {
        // head_dim=2, half=1, j=0: inv_freq = theta^0 = 1, angle = pos.
        // pos=1 => cos(1), sin(1). x=[1,0] => [cos1, sin1].
        let mut v = [1.0, 0.0];
        rope_in_place(&mut v, 1, 10000.0);
        assert!(approx(v[0], 1.0f32.cos(), 1e-6));
        assert!(approx(v[1], 1.0f32.sin(), 1e-6));
    }

    #[test]
    fn rope_preserves_norm() {
        // Rotation is orthogonal => L2 norm unchanged.
        let mut v = [0.3, 0.7, -1.2, 2.5, 0.1, -0.6];
        let n0: f32 = v.iter().map(|x| x * x).sum();
        rope_in_place(&mut v, 7, 10000.0);
        let n1: f32 = v.iter().map(|x| x * x).sum();
        assert!(approx(n0, n1, 1e-4), "norm changed: {n0} -> {n1}");
    }
}
