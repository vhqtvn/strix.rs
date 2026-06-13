# Multi-core + MAC-array flash attention — the multi-week endpoint

Goal: make NPU SDPA actually beat CPU SDPA. The single-core matvec kernel is
correct + integrated but ~400× too slow (386 ms/run bucket-128; doesn't scale to
real prefill lengths). Two orthogonal levers, both needed:

1. **MAC array (`aie::mmul`)** — compute `S = Q·Kᵀ` and `O = P·V` on the systolic
   MAC array instead of scalar-vector matvec. ~10× per core.
2. **Multi-core (whole_array, up to 8 cols × n rows)** — attention is embarrassingly
   parallel over query rows/tiles (each query row is independent — NO cross-core
   reduction, unlike matmul's K-reduction). Broadcast K/V to all cores, distribute
   query tiles, gather outputs. ~Ncores×.

## bf16 mmul facts (aie2p, from mm.cc)
- `aie::mmul<r,s,t,bf16,bf16,accauto>` with `(r,s,t) = (4,8,8)` (no bfp16 emulation).
- `size_A = r*s = 32`, `size_B = s*t = 64`, `size_C = r*t = 32`.
- Operands must be in **blocked tile layout**: A as `[M/r][K/s]` tiles of `[r,s]`,
  B as `[K/s][N/t]` tiles of `[s,t]` (or transposed via `b_row_maj=false`), C as
  `[M/r][N/t]` tiles of `[r,t]`. `matmul_vectorized_2x2_mmul` (mm.cc) is the proven
  2×2-expanded inner loop; needs `rowA%2==0`, `colB%2==0`.

## Layout mapping (per query-tile MT=32, kv-block LB=32, head_dim D)
- **S = Q·Kᵀ**: M=MT, K=D, N=LB. A=Q `[MT,D]`, B=K `[LB,D]` used as **col-major B**
  (`b_row_maj=false` → mmul transposes tiles) so K stays natural layout. C=S `[MT,LB]`.
- **O = P·V**: M=MT, K=LB, N=D. A=P `[MT,LB]`, B=V `[LB,D]` row-major. C=O `[MT,D]`.
- **The hard part — softmax between the two mmuls.** S comes out C-blocked
  `[MT/r][LB/t]` tiles of `[r,t]`. Softmax needs per-query-row (over LB) max+sum;
  in blocked layout a row's LB scores are scattered across `LB/t` col-tiles and the
  `r` row-lanes. Plan: **un-tile S → row-major `[MT,LB]` scratch** (one repack),
  do the online-softmax there (row-wise reduce, the proven code), then **re-tile P
  → A-blocked** for the P·V mmul (second repack). Repacks are MT*LB / MT*D each —
  cheap vs the mmul savings, and avoidable later via DMA `dims_to_stream`.

## Increments (each validated on the existing harness before the next)
- ✅ **I1+I2 DONE [commit 302d6fe]** — `attention_mmul.cc`: full single-core mmul flash
  attn (Q·Kᵀ and P·V on `aie::mmul<4,8,8>`, on-chip repack to blocked layout, per-row
  online softmax on row-major S between). Validated 8×16×64 causal: cosine 0.9998,
  rel-L2 0.023. The MAC-array flash COMPUTE is correct.
- ⛔ **MEMORY BLOCKER (found in I2):** on-chip repack scratch (Qb+KTb+Vb+Ob+S+O ≈ 40KB+
  at MT=32/D=128) overflows the 64KB tile. So the repack CANNOT stay on-chip at real
  shapes — I3 is mandatory, not optional.
- ⏳ **I3 (NEXT, the hard dataflow):** push tiling into the DMA via `dims_to_stream` on
  the Q/K/V `of_q`/`of_kv` fills so blocked tiles arrive directly (no on-chip repack);
  stage via the 512KB memtile. This is the intricate part (the whole_array `a_dims`/
  `b_dims`/`c_dims` access patterns). S→softmax→P still needs an on-chip un-tile/re-tile
  of just the [MT,LB] score tile (small, fits).
- ✅ **I3 DONE [commit de4d5e1]:** host pre-tiles Q/KT/V into blocked layout (free in
  the pack loop, no 3rd DMA channel) → mmul kernel reads blocked operands directly;
  only the small [MT,LB] score tile un-tiles on-chip. Fits 64KB at D=128 (MT=16):
  cosine 0.9997, 50 ms/run (~4× faster/tile than matvec).
- ✅ **I4 DONE [commit c298624], `attention_mc.py`:** N INDEPENDENT pipelines (simpler
  than the whole_array memtile hierarchy — attention has no cross-core reduction, so
  each core owns whole query tiles with its own Q/KV/out fifos + Buffers; host layout
  identical to single-core, each core reads its slice). Validated 128×128×128 causal
  (matvec): **1c 24.24 ms → 2c 6.39 (3.8×) → 4c 1.71 ms/run (14.2×)** — SUPERLINEAR
  (parallelizes per-core DMA + state, not just compute), cosine 0.9998 at every count.
- ✅ **I3+I4 combined + I5 integrated + benchmarked:** smollm3 multi-core matvec attn
  (NH=4, D=128, bucket128, **NC=8**) = 6.5 ms/run (59× over 386 ms single-core). Dropped
  into the model (same host interface, no Rust change); prefill 12 tok 0.2 → **7.0 tok/s**.
  Finding: mmul ≈ matvec per tile after multi-core (softmax + repack dominate) → matvec
  multi-core is the practical winner.

## ⛔ Honest verdict (the whole effort)
The two levers WORK and are validated + integrated (59× over single-core NPU). **But
NPU SDPA does not beat CPU SDPA** for these models: at seq=128, CPU 134.6 tok/s vs NPU
65.5 (~2× slower) — CPU SDPA is already fast; the 144 sequential per-(layer,kv-head)
calls + KV replication + packing dominate. NPU could only win at long seq (512+, CPU
O(n²)) with bigger-bucket xclbins. **Plus a bf16 precision cost**: NPU(bf16) vs CPU(f32)
tokens match at 32 tok, diverge at 64+ (per-layer 0.03 rel-L2 compounds over 36 layers).
⇒ Confirms NPU prefill is the *power* path, already "fast enough"; SDPA-on-NPU is low-ROI
vs the gemma3n hipGraph decode lever. The fast, correct NPU flash kernel is banked for a
future power-optimized or very-long-context path.

## Risks / notes
- Build loop is off-box on vha (~17 min/cycle) → minimize cycles, validate each I.
- Strategic: NPU prefill is the POWER path & already "fast enough"
  ([[never-gpu-prefill]]); this is speed-for-its-own-sake. Endpoint = FastFlowLM-class.
