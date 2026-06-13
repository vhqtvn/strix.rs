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
- **I1 (de-risk mmul):** single-core kernel computes ONLY `S=Q·Kᵀ` via
  `matmul_vectorized` (on-chip repack Q,K → blocked; output S row-major); validate S
  vs CPU. Proves the MAC-array primitive in-kernel. ← START HERE
- **I2:** add online-softmax on row-major S + `O=P·V` mmul; full single-core mmul
  flash attn; validate vs CPU; measure (expect ~10× over matvec).
- **I3:** push the tiling into the DMA (`dims_to_stream` on the Q/K/V fills) to drop
  the on-chip repacks.
- **I4 (multi-core):** N workers, KV broadcast (multiple `.cons()` / memtile forward),
  query tiles distributed (`split`), outputs joined (`join`) — the whole_array
  pattern, but no C-reduction (each core owns whole query rows). Scale 2→8 cores.
- **I5:** integrate the multi-core/bucketed xclbins into the model; benchmark vs CPU.

## Risks / notes
- Build loop is off-box on vha (~17 min/cycle) → minimize cycles, validate each I.
- Strategic: NPU prefill is the POWER path & already "fast enough"
  ([[never-gpu-prefill]]); this is speed-for-its-own-sake. Endpoint = FastFlowLM-class.
