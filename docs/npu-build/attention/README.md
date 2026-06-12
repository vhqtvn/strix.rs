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

## ✅ Query tiling — implemented + validated (real shapes now fit)
Queries are processed `Mtile` rows at a time (outer loop); each query tile re-streams
all K/V blocks (inner loop). `attn_finalize` re-arms `(m,l)` per tile so one set of
`[Mtile]`-sized Buffers is reused. Resident memory is bounded by `Mtile` (not total M)
and one K/V block (not L) → real head_dims fit the 64 KB tile.
- `attention.py` args: `-M`=Mtile, `--mq`=total queries, `-K`=L, `-N`=D, `--lb`,
  `--kvdepth` (1 when depth-2 won't fit). K/V replicated NQT× in the host buffer
  (the shim DMA has no stride-0 broadcast read; a memtile broadcast would avoid the
  DDR replication — TODO).
- Validated on XDNA2: **128×64×64 (2 query tiles)** cosine 0.9997; **256×256×128
  (8 query tiles × 8 KV blocks, real head_dim 128, kvdepth=1)** cosine 0.9998,
  rel-L2 0.031 — the qwen3/smollm3 per-head prefill attention shape, fully on the NPU.

## ✅ Causal masking — implemented + validated
`block_row` masks `sblk[jj] = -1e4` (NOT -3e38: exp2's floor→int32 exponent
extraction overflows on -3e38 and fails to underflow; -1e4 gives exp2→0 cleanly)
where `key_idx = kb*ATT_LB + jj > query_idx = qt*ATT_M + i`. `qt`/`kb` are passed
as **i32 scalar args from the (unrolled) dataflow loops** — a persisted counter
Buffer was tried first but the Buffer `initial_value` is NOT honored at runtime on
this toolchain (int OR float), so the counters read garbage. Build flag `CAUSAL=1`.
- Validated on XDNA2: **256×256×128 causal** cosine 0.999768, rel-L2 0.026; query 0
  attends only key 0 → out = v[0] exactly. (Loops are Python-unrolled so qt/kb are
  i32 constants; an scf.for index is MLIR `index` ≠ i32. NQT*NBLK calls — fine here;
  very long seqs would want an index_cast in a range_ loop — scalability TODO.)

## ✅ GQA + head_dim 64/128/256 — validated
- **GQA**: NH query heads share one streamed KV head — loop `NQT = NH·TPH` tiles
  head-major, pass the position-tile-within-head (`qti % TPH`) as the causal index
  (kernel unchanged). Validated: 4 q-heads × 1 kv-head, 128×128, D=128, causal →
  cosine 0.9997.
- **head_dim**: 64 / 128 (qwen3, smollm3) / **256 (gemma3n)** all validated
  (D=256 causal: cosine 0.9996). The kernel is D-parametric (DCHUNKS = D/32).

⛔ **The flash kernel is now shape-complete for every target model**: streaming K/V
(any L) + online softmax + query tiling (any M) + head_dim 64/128/256 + causal + GQA.

⏳ **REMAINING — model integration (the actual speedup)**: wire the kernel into the
Rust model forward to replace CPU SDPA (stage-3 endpoint). Needs: per-(kv-head) host
packing of Q/K/V into the streaming+replicated layout, invoke `run_attn`, scatter the
output back. Perf TODO: memtile KV broadcast to drop the DDR replication; an
`index_cast` `range_` loop instead of unrolling for very long seqs.

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
