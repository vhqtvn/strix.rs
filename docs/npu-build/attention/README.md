# Fused-attention NPU kernel (stage-2 prototype)

Single-core fused attention for the XDNA2 NPU: `scores = Q·K^T → softmax → ·V`,
all intermediates in one AIE core's local memory (no host round-trip — the point
of the fusion). This is the foundational piece of the NPU-fusion stage 2/3 (see
[[npu-fusion-rewrite]] in memory).

## Status (2026-06-12)
✅ **BUILDS** through the off-box pipeline → loadable xclbin + insts. Resolved 3 real
AIE constraints getting there:
1. kernel `.cc` path (attention/ is shallower than matrix_multiplication/<subdir>,
   so makefile-common's 4-`../` `kernels_dir` overshoots — use 3 `../`).
2. **DMA channels**: a tile has only 2 input channels → pack Q‖K‖V into ONE input
   buffer (`attention_bf16(qkv, out)`), 1 in + 1 out fits 2/2.
3. **Tile local memory (64 KB)**: double-buffered packed-QKV overflowed → `depth=1`
   single-buffer the ObjectFifos.

⏳ **NOT yet**: on-device numerical validation (needs a 2-buffer XRT harness; the
existing `npu_loadtest`/`NpuGemm` shim is matmul-3-buffer: a, b/weight, out — the
attention kernel is 2-buffer: in, out). Then: vectorize (scalar matmuls →
`aie::mmul`/vector, currently correctness-over-speed), real shapes (m=256, hd=128/256,
variable causal len, GQA), and integrate into the model forward.

## Files
- `attention.cc` → goes in `aie_kernels/aie2p/`. Reuses `softmax.cc`'s
  `softmax_simple_bf16` (its log2e-scaled exp is numerically sensitive — don't reimplement).
  bf16 end-to-end (matches softmax.cc + mm.cc bf16 → no dtype-conversion kernels between stages).
- `attention.py` → goes in `programming_examples/basic/attention/`. IRON design: one
  worker, packed-QKV in, out out. Accepts the matmul Makefile arg-set (M→queries,
  K→keys/L, N→head_dim/D) so it reuses makefile-common.
- `Makefile` → same dir. Overrides the `.o` rule's kernel path (3 `../`).

## Build (on vha, split pipeline — same as the matmul shapes)
```
make M=64 K=64 N=64 NPU2=1 dtype_in=bf16 dtype_out=bf16 build/final_64x64x64.xclbin
```
(with the fake-xclbinutil split: vha emits main.pdi + 3 JSONs + insts → bundle →
package on HX370 with native xclbinutil 2.21.75. See ../README.md / ../batch.sh.)
