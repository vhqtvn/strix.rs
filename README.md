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
  - **Mellum2-12B-A2.5B** (Q8_0, MoE + hybrid sliding/full attn + YaRN).
  - **Qwen3.6-35B-A3B** (Q6_K, MoE + Gated-DeltaNet SSM + full-attn) — runs, but
    memory-bound (experts spill past the GTT/UMA budget).
  - Gemma-3 runs incidentally (shares the Gemma code path); Llama/Mistral
    (safetensors) remain the CPU correctness oracle.
- ✅ **GGUF loading** + dequant (Q4_0/Q4_1/Q8_0/Q4_K/Q6_K) + embedded tokenizer.
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
```

Drop `--gpu` / `STRIX_ROCM` for the CPU oracle. `STRIX_BENCH=<reps>` gives a
warmup + median/min/max bench (single-shot `generate` numbers are noisy: iGPU
clock ramps ~30 s, ±3–4 tok/s variance).

### Env knobs

- `STRIX_ROCM=1` — select the ROCm/HIP backend (else CPU).
- `STRIX_NPU=1` — enable NPU prefill offload (with `--features ryzen-ai`/`npu-cpu`).
  `STRIX_NPU_COLS=4|8`, `STRIX_NPU_MODE=speed|power`, `STRIX_NPU_DIR=<xclbin dir>`.
- `STRIX_QWEN_IDS="…"` — raw token IDs for models with a gpt2-BPE vocab
  (qwen3, smollm3, gemma3n, mellum, qwen35moe).
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
