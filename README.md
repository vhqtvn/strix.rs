# Strix.rs

Strix.rs is an experimental Rust local LLM runner targeting AMD Ryzen AI
HX 370-class APUs — the **Radeon 890M** iGPU (RDNA 3.5, gfx1150) and the
**XDNA2 NPU** (~50 TOPS), backed by unified LPDDR5x.

It is **not** a production runner and **not** a replacement for `llama.cpp` or
`mistral.rs`. It is a small, focused foundation whose goal is to become
*measurable*: same model, same prompt, same quantization, same context —
compared against `llama.cpp` on prefill, decode, memory, and wattage. Power
efficiency is an explicit priority, which is why the design leans on the NPU.

## Design: hybrid NPU prefill / iGPU decode

The two phases of inference have opposite bottlenecks, so they run on different
engines:

- **Prefill is compute-bound** → offload the dense projection matmuls to the
  **XDNA2 NPU** (W8A8, ~10× more efficient per op, low watts). This is a large
  win: on Qwen3-4B prefill goes from **5.7 tok/s (CPU) → 73 tok/s (NPU)**, ~13×.
- **Decode is bandwidth-bound** (every token re-reads the whole weight set) →
  keep it on the **Radeon 890M iGPU** via ROCm/HIP, where weights stay resident
  and a fused dequant→GEMV kernel runs near the ~90–120 GB/s LPDDR5x ceiling.

This matches AMD's own Lemonade/OGA and FastFlowLM: **prefill = NPU, decode =
iGPU**. The iGPU is never used for sustained prefill (it can trip an SoC reset on
this box under sustained load).

## What works now

- ✅ **ROCm/HIP backend (main GPU path)** for the Radeon 890M (gfx1150): Q4_0 /
  Q6_K / Q8_0 weights upload **resident**; int8-WMMA prefill GEMM, fused
  dequant→GEMV decode, flash-attention SDPA, RoPE/RMSNorm/GeGLU kernels, on-device
  argmax. hipGraph capture for low-dispatch-overhead decode.
- ✅ **XDNA2 NPU prefill offload** via XRT + mlir-aie xclbins: dense projections
  staged W8A8 on the NPU, concurrent split-N with the iGPU. Verified to produce
  **identical** decode tokens to the CPU/iGPU prefill path (KV-bridge correct).
- ✅ **Multiple current-gen architectures run end-to-end** (greedy decode),
  forward verified op-by-op against `llama.cpp`:
  - **Gemma-4-12B-QAT** (Q4_0) — per-layer head_dim/KV, QK-norm, dual+proportional
    RoPE, V-norm, GeGLU, layer scale, logit softcap.
  - **Gemma-3n-E4B** (Q4_0) — MatFormer, with the GPU lm_head (Q4_K→Q8 repack) fix.
  - **Qwen3-4B-2507** (Q4_0, dense) and **SmolLM3-3B** (Q4_0, dense).
  - **Qwen3.5-4B** (Q8_0, dense Gated-DeltaNet + full-attn hybrid) — full **resident
    on-device decode** (DeltaNet recurrence, full attn, dense FFN all on the iGPU,
    1 sync/token); ~26 tok/s decode (see below).
  - **Mellum2-12B-A2.5B** (Q8_0, MoE + hybrid sliding/full attn + YaRN).
  - **Qwen3.6-35B-A3B** (Q6_K, MoE + Gated-DeltaNet SSM + full-attn) — runs, but
    memory-bound (experts spill past the GTT/UMA budget).
  - Gemma-3 runs incidentally (shares the Gemma code path); Llama/Mistral
    (safetensors) remain the CPU correctness oracle.
- ✅ **GGUF loading** + dequant (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K) + embedded tokenizer.
  Quantized weights stay in the mmap and are dequantized per matmul — a 12B Q4_0
  model runs in ~7 GB instead of ~48 GB.
- ✅ **Speculative decoding** — draft-free n-gram lookup (lossless; +27% on Mellum).
- ✅ `device-info` (live iGPU compute check + NPU XRT round-trip), `inspect-model`,
  bench harness (`STRIX_BENCH`).

## Performance (HX 370, this box)

**Best-config benchmark** — NPU prefill + iGPU decode where it helps. Quick
single-shot runs, 64-token synthetic prompt, cold clock (relative config
comparison, not peak; expect higher prefill on long warm prompts):

| Model | Quant | Best config | Prefill tok/s | Decode tok/s |
|---|---|---|---:|---:|
| SmolLM3-3B | Q4_0 | NPU prefill + iGPU decode | 78.8 | 37.1 |
| Qwen3-4B-2507 | Q4_0 | NPU prefill + iGPU decode | 73.3 | 30.5 |
| Mellum2-12B-A2.5B | Q8_0 | iGPU decode (MoE, no NPU) | 181.5 | 25.8 |
| Gemma-3n-E4B | Q4_0 | NPU prefill + iGPU decode | 16.2 | 18.4 |
| Gemma-4-12B-QAT | Q4_0 | iGPU (full forward) | 21.5 | 10.9 |
| Qwen3.6-35B-A3B | Q6_K | NPU prefill + iGPU decode | 3.1 | 1.66 |
| Qwen3.5-4B | Q8_0 | CPU prefill + resident iGPU decode (Q4 dp4a) | 5.7 | **28.3** |

Notes:
- **NPU dense-projection prefill is the headline win** (~13× on Qwen3, ~4.7× on
  Gemma-3n vs CPU prefill). This is distinct from NPU *attention/SDPA*, which was
  measured dead-for-speed and is not on the critical path.
- **Mellum (MoE) is fastest without the NPU** — its 2.5B-active prefill is already
  cheap, and NPU staging diverts weights from the iGPU-resident decode path.
- **Qwen3.6-35B** (27 GB Q6_K) is memory-bound: only the dense weights fit the iGPU,
  MoE experts spill to CPU. Too big for the ~16 GB practical UMA budget.

**Gemma-4-12B-QAT, characterized warm bench** (~2k-token prompt, `STRIX_BENCH`):
prefill **~138 tok/s iGPU / ~132 hybrid**, decode **~7 tok/s @2k ctx (~11 short)** —
at decode **parity** with `llama.cpp`'s ROCm build (11.4 tok/s) on the same model.
The prefill GEMM (int8 WMMA, double-buffered, BM=128, split-K for ffn_down) already
beats llama's pure-matmul `pp512` once SDPA is excluded; the remaining end-to-end gap
is the scalar prefill attention. CPU oracle is ~0.8 tok/s.

**Qwen3.5-4B resident decode (Gated-DeltaNet hybrid).** The whole per-token forward
runs on the iGPU — the **Gated-DeltaNet recurrence** (a from-scratch GPU scan kernel:
1 workgroup/v-head × 128 threads, validated bit-exact vs CPU), full GQA attention with
output gating + partial RoPE, dense SwiGLU FFN, and the tied lm_head — with one sync per
token. Prefill stays on CPU/NPU and seeds the device KV + SSM/conv state (the iGPU is
never used for prefill). Decode arc: CPU **2.5** → per-weight GEMV **~6.6** → resident
**~11.7** → +coalesced DeltaNet state **~14.5** → **+Q8→Q4 weight repack (`STRIX_Q35_Q4`, int8-dp4a GEMV)
~28 tok/s** (~11×). The Q4 repack halves the (bandwidth-bound) weight reads; it's
opt-in/lossy in principle but was token-identical to Q8 on the tested prompt.

### vs llama.cpp (decode, same box + same GGUFs)

Head-to-head against the system `llama.cpp` (ROCm gfx1150, build 9616). **Decode** is the
apples-to-apples comparison (both run it on the iGPU). Crucial methodology note: `llama-bench`'s
default `pp` phase drives the GPU clock high and `tg` then rides it, whereas strix decode starts
cold (its prefill is on CPU/NPU, leaving the iGPU idle) and decode is memory-bound so it never
ramps the clock. So the fair comparison is llama **decode-only** (`-p 0 -n 128`, cold) vs strix
decode:

| Model | strix decode | llama.cpp decode (cold) | |
|---|---:|---:|---|
| Qwen3.5-4B (Q8→Q4) | **28.3** | 23.1 | ✅ strix |
| SmolLM3-3B Q4_0 | **36.5** | 35.0 | ✅ strix |
| Qwen3-4B Q4_0 | **27.4** | 26.2 | ✅ strix |
| Gemma-4-12B Q4_0 | **10.1** | 9.55 | ✅ strix |
| Mellum2-12B-A2.5B Q8_0 | 24.3 | 26.1 | llama +8% |
| Gemma-3n-E4B Q4_0 | 16.8 | 19.6 | llama +17% |

**strix.rs matches or beats llama.cpp on decode for 4 of 6 models.** The two gaps are kernel-efficiency
on already-resident paths: Mellum's Q8 MoE GEMV (Q4 experts go incoherent — need Q4_K/QuaRot), and
Gemma-3n's resident-but-f32 AltUp/Laurel/PLE matmuls (quantizing them to Q8 would ~tie llama, at a
precision risk).

Decode numbers re-confirmed 2026-06-16 (qwen3 27.1, smollm3 36.5, gemma3n 17.4, qwen3.5 27.1 — within
clock noise of the above). These speeds were unchanged by a round of **GPU decode correctness fixes**: a
`q6_gemv` launch-grid bug (`div_ceil(16)` vs the kernel's 8-rows/block → the upper half of the tied
lm_head's output rows went uncomputed) was corrupting high-id-token logits, which surfaced as garbage on
chat/near-tie prompts (and Qwen3's stray-emoji rambling). Plus SmolLM3 was using NEOX RoPE where its
Llama-permuted weights need NORM. With both fixed, the resident GPU decode is now verified **bit-exact
vs the CPU forward** on q6_k lm_heads (and SmolLM3 resident decode is coherent == CPU at 36.5 tok/s) — the
decode wins above are now backed by correct output, not just throughput.

**Prefill**, by contrast, llama is 5–10× faster (pp256: SmolLM3 792, Qwen3 540, Gemma-4 204, Mellum
345 t/s) — *by design*: llama prefills on the iGPU; strix keeps prefill on the NPU/CPU because sustained
iGPU load triggers this box's SoC-reset fault and the NPU path is ~6–7× more energy-efficient. strix
trades prefill throughput for hardware safety + power. (The 35B doesn't fit either engine: 27 GB vs
~4 GB true VRAM + GTT.)

## Build & run

```bash
# ROCm decode + NPU prefill (the full hybrid — needs the all-features build):
cargo build --release -p strix-cli --features "npu-cpu,ryzen-ai,rocm"

# Dense models (Qwen3-4B, SmolLM3): NPU prefill + iGPU decode.
# These use a gpt2-BPE vocab the bundled tokenizer can't encode, so pass raw
# token IDs via STRIX_QWEN_IDS (comma/space separated).
STRIX_ROCM=1 STRIX_NPU=1 STRIX_QWEN_IDS="1,2,3,..." \
  ./target/release/strix generate --model models/qwen3-4b-2507 \
  --prompt x --raw --gpu --max-tokens 40

# Gemma-4-12B-QAT: instruction-tuned, use --chat (wraps in Gemma's turn template).
cargo build --release -p strix-cli --features rocm
STRIX_ROCM=1 ./target/release/strix generate \
  --model models/gemma-4-12b-qat \
  --chat --gpu --prompt "Name three primary colors." --max-tokens 40
# => The three primary colors are: 1. Red 2. Yellow 3. Blue

# Mellum2-12B MoE: iGPU decode only (no STRIX_NPU).
STRIX_ROCM=1 STRIX_QWEN_IDS="1,2,3,..." \
  ./target/release/strix generate --model models/mellum2-12b-a2.5b \
  --prompt x --raw --gpu --max-tokens 40

# Qwen3.5-4B: resident on-device decode + Q8→Q4 weight repack (fastest decode).
STRIX_ROCM=1 STRIX_Q35_Q4=1 STRIX_QWEN_IDS="1,2,3,..." \
  ./target/release/strix generate --model models/qwen3.5-4b/Qwen3.5-4B-Q8_0.gguf \
  --prompt x --raw --gpu --max-tokens 64
```

Drop `--gpu` / `STRIX_ROCM` for the CPU oracle. `STRIX_BENCH=<reps>` gives a
warmup + median/min/max bench (single-shot `generate` numbers are noisy: iGPU
clock ramps ~30 s, ±3–4 tok/s variance).

### Env knobs

- `STRIX_ROCM=1` — select the ROCm/HIP backend (else CPU).
- `STRIX_NPU=1` — enable NPU prefill offload (with `--features ryzen-ai`/`npu-cpu`).
  `STRIX_NPU_COLS=4|8`, `STRIX_NPU_MODE=speed|power`, `STRIX_NPU_DIR=<xclbin dir>`.
- `STRIX_QWEN_IDS="…"` — raw token IDs for models with a gpt2-BPE vocab
  (qwen3, smollm3, gemma3n, mellum, qwen35/qwen35moe).
- `STRIX_Q35_Q4=1` — repack Qwen3.5-4B's Q8_0 weights to Q4_0 for the resident decode
  (~1.8× decode, half the weight bandwidth; lossy in principle). `STRIX_Q35_RESIDENT=0`
  reverts to the per-weight-GEMV decode path.
- `STRIX_F16_KV=1` — f16 KV cache (−7% prefill / +7% decode / 2× context; default f32).
- `STRIX_BENCH=<reps>`, `STRIX_NOGRAPH=1` (disable hipGraph), `STRIX_PROF=1`.

### Feature flags (`strix-cli`)

| feature      | meaning                                                           |
|--------------|-------------------------------------------------------------------|
| `cpu`        | default — CPU reference backend (correctness oracle)              |
| `rocm`       | **main GPU backend** — Radeon 890M / gfx1150 (resident decode + prefill GEMM) |
| `ryzen-ai`   | XDNA2 NPU via XRT (prefill offload xclbins)                        |
| `npu-hybrid` | `rocm` + NPU concurrent split-N prefill                           |
| `npu-cpu`    | NPU prefill offload for the CPU MoE forwards (no iGPU)             |
| `vulkan`     | older wgpu decode path (superseded by `rocm`; still builds)       |

## Layout

```text
crates/
  strix-core            traits + plain data types; the WeightAccel seam
  strix-models          architecture configs + GGUF inspection + quant
  strix-backend-cpu     CPU reference forwards (correctness oracle) + NPU prefill offload
  strix-backend-rocm    main GPU backend (HIP kernels, resident decode, prefill GEMM, hipGraph)
  strix-backend-npu     XDNA2 NPU via XRT (C++ shim + Rust FFI)
  strix-backend-vulkan  older Radeon 890M wgpu path (superseded)
  strix-cli             the `strix` binary
docs/                   dev notes & status (local-only; not tracked)
refs/                   reference impls (study only — do not modify)
```

## Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features
cargo test --all
```

## License

MIT OR Apache-2.0.
