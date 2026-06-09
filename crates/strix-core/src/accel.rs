//! Weight-matmul accelerator trait — the seam between a model's forward pass
//! and a device backend that can run GEMV (`y = W · x`) on resident weights.
//!
//! The CPU forward (e.g. `gemma.rs`) owns the quantized weight bytes (from the
//! mmap'd GGUF). An accelerator (the Vulkan iGPU backend) can *adopt* a weight —
//! upload it once into device-resident memory in its native quantization — and
//! thereafter compute that weight's GEMV on-device. The model keeps the bytes as
//! the fallback, so any weight the accelerator declines (unsupported quant,
//! odd shape) transparently runs on the CPU.
//!
//! This trait lives in `strix-core` so the CPU backend depends only on the
//! abstraction, never on a specific device crate (Vulkan/ROCm/NPU). The concrete
//! accelerator is injected by the CLI.

/// Per-layer geometry for the full on-device decode forward (plain data so the
/// CPU model can describe its architecture to a device backend without the
/// backend depending on the model crate).
#[derive(Debug, Clone)]
pub struct GpuLayerCfg {
    /// Head dimension for this layer (Gemma-4 varies it: global vs local).
    pub head_dim: usize,
    /// Number of KV heads.
    pub n_kv: usize,
    /// True if this layer reuses K as V (Gemma-4 `attention_k_eq_v`).
    pub k_eq_v: bool,
    /// RoPE base for this layer (global 1e6 / local 1e4).
    pub rope_theta: f32,
    /// True for sliding-window (local) layers; global layers use `rope_freqs`.
    pub is_local: bool,
    /// Per-layer residual-stream scalar (Gemma-4 `layer_output_scale`).
    pub output_scale: f32,
}

/// Whole-model config for the on-device decode forward.
#[derive(Debug, Clone)]
pub struct GpuDecodeConfig {
    pub hidden: usize,
    pub n_heads: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub n_layers: usize,
    pub eps: f32,
    pub final_softcap: f32,
    /// 1/sqrt(head_dim) attention scaling (gemma3) vs 1.0 (gemma4).
    pub attn_rsqrt: bool,
    /// RMS-normalize V per head (no weight) before caching (gemma4).
    pub norm_v: bool,
    /// Sliding-window size for local layers (0 = no windowing). Local-layer
    /// queries attend only the last `n_swa` keys.
    pub n_swa: usize,
    /// Max sequence length the KV cache must hold.
    pub max_seq: usize,
    pub layers: Vec<GpuLayerCfg>,
}

/// A device that can hold weights resident and compute their GEMV.
///
/// `key` is an opaque, caller-chosen weight identifier (we use GGUF tensor
/// names). `upload_*` returns whether the weight was adopted; `gemv` returns
/// `None` for any key the accelerator did not adopt.
pub trait WeightAccel: Send + Sync {
    /// Adopt a Q4_0 weight `[out_dim, in_dim]` (raw GGUF block bytes), uploading
    /// it resident. Returns `true` if adopted, `false` to leave it on the CPU.
    fn upload_q4_0(&mut self, key: &str, bytes: &[u8], in_dim: usize, out_dim: usize) -> bool;

    /// Adopt a Q6_K weight `[out_dim, in_dim]` (raw GGUF block bytes).
    fn upload_q6_k(&mut self, key: &str, bytes: &[u8], in_dim: usize, out_dim: usize) -> bool;

    /// Adopt a Q8_0 weight `[out_dim, in_dim]` (raw GGUF block bytes). Default: not
    /// adopted (returns false) — only backends with a Q8_0 GEMV kernel override this.
    fn upload_q8_0(&mut self, _key: &str, _bytes: &[u8], _in_dim: usize, _out_dim: usize) -> bool {
        false
    }

    /// Compute `y = W · x` for an adopted weight. Returns `None` if `key` was not
    /// adopted, or `Some(y)` of length `out_dim`.
    fn gemv(&self, key: &str, x: &[f32]) -> Option<Vec<f32>>;

    /// Batched GEMV: compute several `(key, x)` pairs together. Accelerators that
    /// can submit them as one device job (one sync instead of many) should
    /// override this; the default just calls [`WeightAccel::gemv`] per item.
    /// Result `i` is `None` iff `calls[i].0` was not adopted.
    fn gemv_batch(&self, calls: &[(&str, &[f32])]) -> Vec<Option<Vec<f32>>> {
        calls.iter().map(|(k, x)| self.gemv(k, x)).collect()
    }

    /// Number of adopted weights (for reporting).
    fn resident_count(&self) -> usize;

    /// Human-readable accelerator name (e.g. the GPU adapter).
    fn name(&self) -> &str;

    // --- Optional full on-device decode forward (Stage C) ---
    // Backends that can run the *entire* decode step on-device implement these;
    // the default impls leave `decode_step` unsupported so the model falls back
    // to per-matmul `gemv`.

    /// Upload a small f32 tensor resident under `key` (norm weights, rope_freqs).
    fn upload_f32(&mut self, _key: &str, _data: &[f32]) {}

    /// Provide the model architecture for the on-device forward. Call after all
    /// weights are uploaded. Returns true if the backend can run `decode_step`.
    fn configure_decode(&mut self, _cfg: GpuDecodeConfig) -> bool {
        false
    }

    /// Run one full decode step entirely on-device: `h` is the (already
    /// embedding-scaled) hidden state for the current token, `pos` its position.
    /// Returns the output logits, or `None` if unsupported. Appends to the
    /// device KV cache as a side effect.
    fn decode_step(&mut self, _h: &[f32], _pos: usize) -> Option<Vec<f32>> {
        None
    }

    /// Greedy decode of one token, returning only the argmax token id (computed
    /// on-device, no vocab-wide logits readback). `None` if unsupported — callers
    /// fall back to `decode_step` + CPU argmax. Same KV side effect as `decode_step`.
    fn decode_step_argmax(&mut self, _h: &[f32], _pos: usize) -> Option<u32> {
        None
    }

    /// Run a BATCHED prefill of `m` tokens in one pass: `h` is the
    /// embedding-scaled hidden states `[m * hidden]` (row-major, token-major) for
    /// positions `start_pos .. start_pos+m`. Fills the device KV cache for those
    /// positions (causal) and returns the logits of the LAST token (to sample the
    /// first generated token), or `None` if unsupported. The win over calling
    /// `decode_step` m times: each weight is read once (batched GEMM) instead of m
    /// times — prefill becomes compute-bound. `m` must fit the backend's cap.
    fn prefill(&mut self, _h: &[f32], _start_pos: usize, _m: usize) -> Option<Vec<f32>> {
        None
    }

    /// Max tokens accepted by a single [`WeightAccel::prefill`] call (0 = prefill
    /// unsupported). The caller chunks longer prompts.
    fn prefill_max(&self) -> usize {
        0
    }

    /// Speculative-decoding verify: like [`WeightAccel::prefill`] but returns
    /// logits for ALL `m` tokens (`m * vocab`, token-major), not just the last.
    /// `None` if unsupported.
    fn verify(&mut self, _h: &[f32], _start_pos: usize, _m: usize) -> Option<Vec<f32>> {
        None
    }
}
