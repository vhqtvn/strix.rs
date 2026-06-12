# Fused-attention NPU kernel (stage-2 prototype)

Single-core fused attention for the XDNA2 NPU: `scores = Q·K^T → softmax → ·V`,
all intermediates in one AIE core's local memory (no host round-trip — the point
of the fusion). This is the foundational piece of the NPU-fusion stage 2/3 (see
[[npu-fusion-rewrite]] in memory).

## Status (2026-06-12)
✅ **BUILDS + VALIDATES on the XDNA2 NPU.** rel-L2 0.037, cosine 0.999684 vs a CPU
reference (m=l=d=64, bf16) — within bf16 tolerance. Run:
`cargo run -p strix-backend-npu --example npu_attn_test --features ryzen-ai -- <xclbin> <insts>`.

Resolved 4 real AIE constraints getting there:
1. kernel `.cc` path (attention/ is shallower than matrix_multiplication/<subdir>,
   so makefile-common's 4-`../` `kernels_dir` overshoots — use 3 `../`).
2. **DMA channels**: a tile has only 2 input channels → pack Q‖K‖V into ONE input
   buffer (`attention_bf16(qkv, out)`), 1 in + 1 out fits 2/2.
3. **Tile local memory (64 KB)**: double-buffered packed-QKV overflowed → `depth=1`
   single-buffer the ObjectFifos.
4. **Stack-array alignment (the NaN bug)**: `softmax_simple_bf16` uses
   `aie::cbegin_restrict_vector<32>` — ALIGNED 512-bit vector loads. The `scores`/`probs`
   stack arrays were only 2-byte aligned, so the vector load read garbage and softmax
   blew up → all-NaN output. Fix: `alignas(64)` on both. (2-buffer XRT harness:
   `strix_npu_attn` in npu_shim.cpp / `run_attn` in lib.rs — the matmul shim is
   3-buffer a/b/out, attention is 2-buffer in/out.)

✅ **VECTORIZED** [commit c13da68]: D-axis Q·K^T dot products and probs·V
accumulation use `aie::load_v` + `aie::mac` into lane-wise `accum` (reduce_add for
scores, scalar-vec mac for V). Bit-identical to scalar on the NPU; ~16-32× fewer
inner-loop ops. Plain row-major layout kept (no blocked-mmul DMA tiling). Requires
D % 32 == 0 (true for all real head_dims 64/128/256).

## ⛔ The resident one-shot design tops out at the 64×64×64 toy shape
The kernel keeps the WHOLE packed Q‖K‖V + out resident in one tile's **64 KB** data
memory. Budget (bf16, 2 B/elem):
- 64×64×64 (toy): packed 3·64·64·2 = 24 KB + out 8 KB ≈ 33 KB. **fits.**
- 64×64×**128** (real head_dim, tiny seq): packed 48 KB + out 16 KB ≈ 67 KB. **overflows.**
- 256×256×128 (real prefill batch): Q/K/V 64 KB each = 192 KB, scores[256,256]bf16
  = 128 KB. **3–5× over.**

⇒ Real shapes CANNOT be reached by bumping the `-D` macros — they need the streaming
flash dataflow below.

## ✅ Streaming flash attention — implemented + validated (lifts the ceiling)
K/V now **stream block-by-block** (one K‖V block resident at a time); Q is resident;
running flash state `(m, l, o)` is carried across blocks in **persistent core-local
Buffers**; **online softmax** rescales the carried state by `exp2(m_old − m_new)` per
block. L is no longer bounded by tile memory.
- `attention.cc`: `block_row` (one query row × one block; first-block detected from the
  `m = −3e38` sentinel) + `attn_block` (uniform streaming loop) + `attn_finalize`.
- `attention.py`: `of_q` (resident) + `of_kv` (streamed, depth 2) + `of_o`; three f32
  `Buffer`s carry `m`/`l`/`o`; ONE multi-object KV fill (3D tap `[NBLK,2·LB,D]`).
- Validated on XDNA2: **64×64×64** cosine 0.9997 / rel-L2 0.035 (== resident kernel,
  byte-identical); **64×256×64 (8 blocks)** cosine 0.9992 / rel-L2 0.039, matches FULL
  (0.27 vs first/last block alone) — carry correct over many blocks.

Two IRON bugs fixed via MLIR inspection + a block-subrange diagnostic: (1) flat `M*D`
DMA tap → `repeat_count > 255` BD limit → 2D `[rows, D]` patterns; (2) separate per-block
`rt.fill`s collide on one fifo slot → one multi-object fill (matmul streaming idiom).

⏳ **REMAINING toward integration** (shape/tiling engineering on the proven core):
query tiling (at D=128, M=64: Q 16 KB + o_buf f32 32 KB + KV depth-2 32 KB + out 16 KB
= 96 KB > 64 KB → tile to Mtile≈32, loop query tiles; prefill m=256 → 8 tiles); GQA
(q-heads share a streamed kv-head); causal masking per block; integrate into the model
forward (replace CPU SDPA).

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
