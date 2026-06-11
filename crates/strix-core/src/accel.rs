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
    /// False = skip RoPE entirely for this layer (smollm3 NoPE every 4th layer).
    pub no_rope: bool,
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
    /// Per-head QK-norm before RoPE (gemma3, qwen3). False = no QK-norm (smollm3).
    pub qk_norm: bool,
    /// Sandwich post-norm residuals — `h += rmsnorm(proj_out)·post_w` with
    /// post_attention_norm/post_ffw_norm weights (gemma). False = plain residual
    /// `h += proj_out` (llama-family: smollm3, qwen3).
    pub post_norm: bool,
    /// FFN activation: true = GeGLU (gemma), false = SwiGLU/SiLU (llama-family).
    pub act_gelu: bool,
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

    /// Adopt a Q4_1 weight `[out_dim, in_dim]` (raw GGUF block bytes: per-block
    /// scale+min). Default: not adopted — only backends with a Q4_1 GEMV override.
    fn upload_q4_1(&mut self, _key: &str, _bytes: &[u8], _in_dim: usize, _out_dim: usize) -> bool {
        false
    }

    /// Adopt a Q6_K weight `[out_dim, in_dim]` (raw GGUF block bytes).
    fn upload_q6_k(&mut self, key: &str, bytes: &[u8], in_dim: usize, out_dim: usize) -> bool;

    /// Adopt a Q8_0 weight `[out_dim, in_dim]` (raw GGUF block bytes). Default: not
    /// adopted (returns false) — only backends with a Q8_0 GEMV kernel override this.
    fn upload_q8_0(&mut self, _key: &str, _bytes: &[u8], _in_dim: usize, _out_dim: usize) -> bool {
        false
    }

    /// Adopt a whole MoE layer's expert tensors in NATIVE Q8_0 layout (gate/up:
    /// [hidden→eff]×ne; down: [eff→hidden]×ne). Default: not adopted.
    #[allow(clippy::too_many_arguments)]
    fn upload_moe_q8(
        &mut self,
        _layer: usize,
        _gate: &[u8],
        _up: &[u8],
        _down: &[u8],
        _hidden: usize,
        _eff: usize,
        _ne: usize,
    ) -> bool {
        false
    }

    /// Adopt a whole MoE layer's expert tensors in NATIVE Q6_K layout. Default: no.
    #[allow(clippy::too_many_arguments)]
    fn upload_moe_q6(
        &mut self,
        _layer: usize,
        _gate: &[u8],
        _up: &[u8],
        _down: &[u8],
        _hidden: usize,
        _eff: usize,
        _ne: usize,
    ) -> bool {
        false
    }

    /// Fused MoE FFN for one token: y = Σ_k wexp[k]·down(silu(gate)·up)(x) over the
    /// routed experts — one device round-trip. `None` if layer not adopted.
    /// `sgate`: sigmoid-gated shared-expert scale to fuse (0.0 = no shared expert).
    fn moe_ffn(
        &self,
        _layer: usize,
        _ids: &[i32],
        _wexp: &[f32],
        _x: &[f32],
        _sgate: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Prefill GEMM on a resident Q8 dense weight: y[m][out] = W·xs (one sync).
    fn prefill_q8_gemm(&self, _key: &str, _xs: &[f32], _m: usize) -> Option<Vec<f32>> {
        None
    }
    /// Batched lm_head: argmax per row of W[vocab,hidden]·xs (m rows, weights read once).
    fn lm_head_argmax_rows(&self, _key: &str, _xs: &[f32], _m: usize) -> Option<Vec<u32>> {
        None
    }
    /// Whole expert FFN for m grouped tokens (gate/up/silu/down, one sync).
    fn moe_expert_ffn(&self, _layer: usize, _e: usize, _xs: &[f32], _m: usize) -> Option<Vec<f32>> {
        None
    }
    /// Queue an expert FFN without syncing; result lands at `dy_off` rows in the
    /// device dy pool. Returns false if not resident / capacity exceeded.
    fn moe_expert_queue(
        &self,
        _layer: usize,
        _e: usize,
        _xs: &[f32],
        _m: usize,
        _dy_off: usize,
    ) -> bool {
        false
    }
    /// Whole-layer multi-expert FFN: plan = (expert id, row count); xs_all gathered
    /// expert-major. Returns dy rows. None ⇒ per-expert fallback.
    fn moe_layer_ffn(
        &self,
        _layer: usize,
        _plan: &[(i32, i32)],
        _xs: &[f32],
        _rows: usize,
    ) -> Option<Vec<f32>> {
        None
    }
    /// Q8 planar variant of [`WeightAccel::moe_expert_queue`].
    fn moe_expert_queue_q8(
        &self,
        _layer: usize,
        _e: usize,
        _xs: &[f32],
        _m: usize,
        _dy_off: usize,
    ) -> bool {
        false
    }
    /// Sync + download `rows` of hidden from the dy pool.
    /// Whole-layer MoE on device (int8 prefill): one upload (normed acts) + one
    /// download (m*hidden out). plan = (expert, slot_off, count); slot_tok/wslot per slot.
    /// Whole-layer batched attention on device (int8 prefill): norm acts in,
    /// returns (attn_out m*hidden, roped k m*kv_dim, v m*kv_dim).
    #[allow(clippy::too_many_arguments)]
    fn mlm_attn_prefill(
        &mut self,
        _layer: usize,
        _xs: &[f32],
        _m: usize,
        _base: usize,
        _win: usize,
        _cs: &[f32],
        _sn: &[f32],
    ) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        None
    }

    /// Resident-h prefill: upload h once per chunk, run layers via the fns below,
    /// download once at the end. Returns false/None on missing support.
    fn pf_begin(&mut self, _h: &[f32], _m: usize) -> bool {
        false
    }
    #[allow(clippy::too_many_arguments)]
    fn pf_attn(
        &mut self,
        _l: usize,
        _m: usize,
        _base: usize,
        _win: usize,
        _cs: &[f32],
        _sn: &[f32],
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        None
    }
    fn pf_router(&mut self, _l: usize, _m: usize, _ne: usize) -> Option<Vec<f32>> {
        None
    }
    fn pf_moe(
        &mut self,
        _l: usize,
        _m: usize,
        _plan: &[(usize, usize, usize)],
        _st: &[i32],
        _w: &[f32],
    ) -> bool {
        false
    }
    fn pf_end(&mut self, _m: usize) -> Option<Vec<f32>> {
        None
    }

    fn moe_layer_q8_dev(
        &self,
        _layer: usize,
        _xs: &[f32],
        _m: usize,
        _plan: &[(usize, usize, usize)],
        _slot_tok: &[i32],
        _wslot: &[f32],
    ) -> Option<Vec<f32>> {
        None
    }

    fn moe_expert_flush(&self, _rows: usize, _hidden: usize) -> Option<Vec<f32>> {
        None
    }

    // --- Mellum fused decode: h stays resident on-device across the whole token;
    // the host round-trips only q/k/v (rope+SDPA on CPU) and the router logits.

    /// Upload the token's hidden state; begins a fused decode token. False = unsupported.
    fn mlm_begin(&mut self, _h: &[f32]) -> bool {
        false
    }
    /// Full on-GPU layer (norm→qkv→rope→SDPA→o→norm→router); 1 sync; router logits.
    fn mlm_layer(&mut self, _il: usize, _pos: usize, _win: usize) -> Option<Vec<f32>> {
        None
    }
    /// Full layer with on-GPU router + queued MoE — no sync. False = unsupported.
    fn mlm_layer_nosync(&mut self, _il: usize, _pos: usize, _win: usize, _topk: usize) -> bool {
        false
    }
    /// Allocate device KV caches for the fused token path.
    fn mlm_prepare(&mut self, _n_layers: usize, _kv_dim: usize, _max_seq: usize) -> bool {
        false
    }
    /// Seed the device KV cache for layer `il` from host data (len rows of kv_dim).
    fn mlm_seed_kv(&mut self, _il: usize, _k: &[f32], _v: &[f32]) -> bool {
        false
    }
    /// Upload rope cos/sin tables for the current pos/layer-type.
    fn mlm_rope_tables(&mut self, _cs: &[f32], _sn: &[f32]) -> bool {
        false
    }
    /// attn_norm + q/k/v projections for layer `il`; returns (q,k,v).
    fn mlm_qkv(&mut self, _il: usize) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        None
    }
    /// o-proj + residual + ffn_norm + router for layer `il`; returns router logits.
    fn mlm_post1(&mut self, _il: usize, _attn_out: &[f32]) -> Option<Vec<f32>> {
        None
    }
    /// fused MoE with routed experts + residual for layer `il` (no sync). False = unsupported.
    fn mlm_post2(&mut self, _il: usize, _ids: &[i32], _wexp: &[f32]) -> bool {
        false
    }
    /// Graph token: capture once, replay per token; returns logits.
    fn mlm_token_graph(&mut self, _layers: &[(usize, bool)], _topk: usize) -> Option<Vec<f32>> {
        None
    }
    /// Upload device pos for the graph token.
    fn mlm_set_pos(&mut self, _pos: i32) -> bool {
        false
    }
    /// Upload both rope table sets (sliding, full).
    fn mlm_rope_tables2(
        &mut self,
        _cs_s: &[f32],
        _sn_s: &[f32],
        _cs_f: &[f32],
        _sn_f: &[f32],
    ) -> bool {
        false
    }
    /// output_norm + lm_head on the resident h; ends the token.
    fn mlm_logits(&mut self) -> Option<Vec<f32>> {
        None
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
