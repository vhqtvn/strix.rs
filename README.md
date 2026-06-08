# Strix.rs

Strix.rs is an experimental Rust local LLM runner targeting AMD Ryzen AI
HX 370-class APUs (Radeon 890M iGPU + XDNA2 NPU).

It is **not** a production runner and **not** a replacement for `llama.cpp` or
`mistral.rs`. It is a small, focused foundation whose goal is to become
*measurable*: same model, same prompt, same quantization, same context —
compared against `llama.cpp` on prefill, decode, memory, and wattage.

## Current status

> ⚠️ **This section and `docs/*.md` (except `docs/STATUS.md`) are historical and
> predate the ROCm + NPU-hybrid + prefill-GEMM work.** For the authoritative,
> up-to-date state — current performance (~138 tok/s prefill iGPU / ~132 hybrid on
> Gemma-4-12B-QAT, the ROCm WMMA prefill GEMM, the XDNA2 NPU offload, what's been
> tried, and open next steps — see **[`docs/STATUS.md`](docs/STATUS.md)**.

Milestone 2 reached: **CPU reference inference works end-to-end.** `generate`
produces coherent text from a real Llama-architecture safetensors model.

- ✅ Workspace + crate layout, CLI surface, error handling, logging
- ✅ Model inspection (`config.json` → normalized config; format detection)
- ✅ `device-info` and `bench-dummy`
- ✅ CPU reference backend: RMSNorm, RoPE, GQA attention, SwiGLU MLP, KV cache
- ✅ safetensors loading (f32/f16/bf16 → f32) + HF tokenizer
- ✅ `generate` — greedy decoding on Llama/Mistral (safetensors) + Gemma (GGUF)
- ✅ **GGUF loading** + dequant (Q4_0/Q4_1/Q8_0/Q4_K/Q6_K) + GGUF-embedded tokenizer
- ✅ **Quantized inference**: weights stay quantized in the mmap, dequantized per
  matmul row (rayon-parallel) — a 12B q4_0 model runs in ~7 GB instead of ~48 GB
- ✅ **Gemma-4 (12B QAT q4_0) runs coherently** — verified op-by-op against
  llama.cpp; per-layer head_dim/KV, k_eq_v, QK-norm, dual + proportional RoPE,
  V-norm, GeGLU, layer scale, logit softcap. ~0.8 tok/s on CPU, ~7 GB RAM.
- ✅ Gemma-3 runs coherently too (incidental — shares the Gemma code path).
- ✅ **Radeon 890M iGPU decode** (`--features vulkan`, `generate --gpu`): Q4_0
  and Q6_K weights upload **resident** to the GPU; a **fused dequant→GEMV** WGSL
  kernel (coalesced, workgroup-per-row, shared-memory reduction; resident I/O
  buffers, no per-call allocation) runs every projection + the lm_head on-device.
  End-to-end on Gemma-4 12B QAT: **decode 0.81 → 8.26 tok/s (~10×), prefill 0.83 →
  7.16 tok/s** vs the CPU oracle, identical output. The kernel (coalesced
  workgroup-per-row, subgroup-add reduction, resident I/O buffers, f16-packed
  scales) sustains ~80 GB/s — near bandwidth-bound on the ~120 GB/s LPDDR5x;
  validated against CPU to ~1e-4. `strix bench-matmul` micro-benchmarks it.
  Optimization study + roadmap toward the ~17 tok/s ceiling:
  `docs/igpu-optimization-notes.md`.
- 🟡 **XDNA2 NPU** (`--features ryzen-ai`): opens the NPU via XRT FFI and moves
  data host↔NPU↔host (BO alloc/map/sync/read round-trip works, verified in
  `device-info`). Running matmuls still needs an AI-Engine kernel compiled to an
  `.xclbin` (mlir-aie/Peano) — currently blocked on this box (no prebuilt
  xclbins; the toolchain has no wheels for the system Python 3.14). See
  `docs/hx370-notes.md`.

The target design is **hybrid: NPU prefill / iGPU decode** (see
`docs/hx370-notes.md`). `cargo run -p strix-cli --features "vulkan ryzen-ai" --
device-info` reports CPU, the iGPU (live compute check), and the NPU (live XRT
open + data-path round-trip).

Gemma-4 is an instruction/reasoning model — use `--chat` (wraps the prompt in
Gemma's turn template, `<|turn>`/`<turn|>`):

```bash
cargo run --release -p strix-cli --features vulkan -- generate \
  --model models/gemma-4-12b-qat/gemma-4-12b-it-qat-q4_0.gguf \
  --chat --gpu --prompt "Name three primary colors." --max-tokens 40
# => The three primary colors are: 1. Red 2. Yellow 3. Blue
# (--gpu offloads matmuls to the Radeon 890M; drop it for the CPU oracle)
```

### Known limitations (CPU path)

- **Targets:** Gemma-4 and (latest) Qwen — the current-gen models. Gemma-3 works
  incidentally. Llama/Mistral (safetensors) remain as the CPU correctness oracle.
  Qwen2.5/Qwen3 parse via `inspect-model` but don't run yet.
- Greedy sampling only (no temperature/top-k/top-p); single sequence (no
  batching); slow (correctness oracle, not optimized). Q5_K dequant not yet
  implemented. Gemma sliding-window masking is a no-op for sequences ≤ 1024.

### Try it

```bash
# A small real Llama works out of the box (≈160M params):
#   https://huggingface.co/JackFram/llama-160m  (config.json + tokenizer.json + model.safetensors)
cargo run --release -p strix-cli -- \
  generate --model /path/to/llama-160m --prompt "The capital of France is" --max-tokens 20
# => The capital of France is Paris. The city of Paris is the capital of France...
```

## Layout

```text
crates/
  strix-core            traits + plain data types (no I/O, no math)
  strix-models          architecture configs + non-loading inspection
  strix-backend-cpu     CPU reference backend (correctness over speed)
  strix-backend-vulkan  Radeon 890M iGPU backend (enumeration first)
  strix-cli             the `strix` binary
docs/
  architecture.md       module map + trait design
  benchmark-plan.md     metrics + comparison targets
  hx370-notes.md        hardware-specific strategy
  mistral-rs-notes.md   what to borrow / avoid from the reference
refs/
  mistral.rs            reference implementation (study only — do not modify)
```

## Build & run

```bash
cargo build --workspace

cargo run -p strix-cli -- --help
cargo run -p strix-cli -- device-info
cargo run -p strix-cli -- bench-dummy
cargo run -p strix-cli -- inspect-model --path ./some-model
```

To enumerate the Radeon 890M via Vulkan (pulls in `wgpu`):

```bash
cargo run -p strix-cli --features vulkan -- device-info
```

### Feature flags (`strix-cli`)

| feature     | status      | meaning                                   |
|-------------|-------------|-------------------------------------------|
| `cpu`       | default     | CPU reference backend (correctness oracle)|
| `vulkan`    | **decode**  | Radeon 890M: resident Q4_0/Q6_K GEMV (`generate --gpu`) |
| `rocm`      | placeholder | ROCm/HIP experiment, later                |
| `ryzen-ai`  | data path   | XDNA2 NPU via XRT (access + BO round-trip; kernels pending) |

## Planned commands

```bash
strix inspect-model --path <path>                  # works now
strix device-info                                  # works now
strix bench-dummy                                  # works now
strix generate --model <path> --prompt "hello"     # works now (CPU, greedy)
strix chat --model <path>                          # planned
strix bench --model <path>                         # planned
```

## Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features
cargo test --all
```

## License

MIT OR Apache-2.0.
