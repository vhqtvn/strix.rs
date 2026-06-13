//===- attention_mmul.cc ------------------------------------*- C++ -*-===//
//
// Single-core flash attention using the MAC array (aie::mmul) for BOTH matmuls:
//   S = Q·Kᵀ   (MT×D · D×LB → MT×LB)      [Increment 1/2 of the multi-week plan]
//   O = P·V    (MT×LB · LB×D → MT×D)
// with the proven per-row online softmax on a row-major S in between (un-tile S →
// softmax → re-tile P). bf16 mmul tile (r,s,t)=(4,8,8). Data is repacked into
// blocked [r,s]/[s,t] tile layout ON-CHIP (the DMA could do this later — I3).
//
// Layout (matches mm.cc matmul_vectorized_2x2_mmul, b_row_maj=c_row_maj=true):
//   A blocked: tile (mb,kb) at A + (mb*KB+kb)*size_A,  within-tile [r,s] row-major
//   B blocked: tile (kb,nb) at B + (kb*NB+nb)*size_B,  within-tile [s,t] row-major
//   C blocked: tile (mb,nb) at C + (mb*NB+nb)*size_C,  within-tile [r,t] row-major
// Host packs KT (K transposed [D,LB]) so the score matmul is plain row-major B.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef ATT_M
#define ATT_M 32
#endif
#ifndef ATT_L
#define ATT_L 64
#endif
#ifndef ATT_D
#define ATT_D 64
#endif
#ifndef ATT_LB
#define ATT_LB 32
#endif
#ifndef ATT_CAUSAL
#define ATT_CAUSAL 0
#endif

#define ATT_LOG2E 1.4426950408889634f
// bf16 mmul micro-tile
#define MM_R 4
#define MM_S 8
#define MM_T 8

using MMUL = aie::mmul<MM_R, MM_S, MM_T, bfloat16, bfloat16, accauto>;

// C[M,N] = A[M,K]·B[K,N], all in blocked [r,s]/[s,t]/[r,t] tile layout. M%r, K%s,
// N%t == 0. Accumulates the full K reduction per output tile.
template <int M, int K, int N>
static inline void mm_blk(const bfloat16 *restrict A, const bfloat16 *restrict B,
                          bfloat16 *restrict C) {
  constexpr int MB = M / MM_R, KB = K / MM_S, NB = N / MM_T;
  for (int mb = 0; mb < MB; mb++) {
    for (int nb = 0; nb < NB; nb++) {
      aie::vector<bfloat16, MMUL::size_C> zc = aie::zeros<bfloat16, MMUL::size_C>();
      MMUL acc(zc);
      for (int kb = 0; kb < KB; kb++) {
        aie::vector<bfloat16, MMUL::size_A> av =
            aie::load_v<MMUL::size_A>(A + (mb * KB + kb) * MMUL::size_A);
        aie::vector<bfloat16, MMUL::size_B> bv =
            aie::load_v<MMUL::size_B>(B + (kb * NB + nb) * MMUL::size_B);
        acc.mac(av, bv);
      }
      aie::store_v(C + (mb * NB + nb) * MMUL::size_C, acc.to_vector<bfloat16>());
    }
  }
}

// row-major [M,K] → blocked [M/r][K/s] tiles of [r,s]
template <int M, int K, int R, int S>
static inline void tile_rm(const bfloat16 *restrict src, bfloat16 *restrict dst) {
  constexpr int MB = M / R, KB = K / S;
  for (int mb = 0; mb < MB; mb++)
    for (int kb = 0; kb < KB; kb++)
      for (int rr = 0; rr < R; rr++)
        for (int ss = 0; ss < S; ss++)
          dst[(mb * KB + kb) * (R * S) + rr * S + ss] =
              src[(mb * R + rr) * K + (kb * S + ss)];
}

// blocked [M/r][N/t] tiles of [r,t] → row-major [M,N]  (T_ is the tile width)
template <int M, int N, int R, int T_>
static inline void untile_rm(const bfloat16 *restrict src, float *restrict dst) {
  constexpr int MB = M / R, NB = N / T_;
  for (int mb = 0; mb < MB; mb++)
    for (int nb = 0; nb < NB; nb++)
      for (int rr = 0; rr < R; rr++)
        for (int tt = 0; tt < T_; tt++)
          dst[(mb * R + rr) * N + (nb * T_ + tt)] =
              (float)src[(mb * NB + nb) * (R * T_) + rr * T_ + tt];
}

extern "C" {

// kv = [KT (D×LB, K transposed) ‖ V (LB×D)]. qt = query-tile index (for causal
// position), kb = block index. Running (m,l,o) per query row in the Buffers.
void attn_block(bfloat16 *restrict q, bfloat16 *restrict kv,
                float *restrict m_buf, float *restrict l_buf,
                float *restrict o_buf, int32_t qt, int32_t kb) {
  event0();
  const bfloat16 *KT = kv;               // [D, LB]
  const bfloat16 *V = kv + ATT_D * ATT_LB; // [LB, D]
  const int qbase = qt * ATT_M, kbase = kb * ATT_LB;

  // --- scratch (blocked + row-major intermediates). static: too big for the
  // small AIE stack, and on a single core static = persistent local data mem.
  // NOTE: at real shapes (MT=32,D=128) this on-chip repack overflows the 64KB
  // tile → the production path tiles in the DMA (dims_to_stream), not here. ---
  alignas(64) static bfloat16 Qb[ATT_M * ATT_D];
  alignas(64) static bfloat16 KTb[ATT_D * ATT_LB];
  alignas(64) static bfloat16 Sb[ATT_M * ATT_LB];
  alignas(64) static bfloat16 Pb[ATT_M * ATT_LB];
  alignas(64) static bfloat16 Vb[ATT_LB * ATT_D];
  alignas(64) static bfloat16 Ob[ATT_M * ATT_D];
  alignas(128) static float S[ATT_M * ATT_LB];
  alignas(64) static bfloat16 P[ATT_M * ATT_LB];
  alignas(128) static float O[ATT_M * ATT_D];

  // 1) S = Q·Kᵀ on the MAC array.
  tile_rm<ATT_M, ATT_D, MM_R, MM_S>(q, Qb);   // A
  tile_rm<ATT_D, ATT_LB, MM_S, MM_T>(KT, KTb); // B (KT is [D,LB] row-major)
  mm_blk<ATT_M, ATT_D, ATT_LB>(Qb, KTb, Sb);
  untile_rm<ATT_M, ATT_LB, MM_R, MM_T>(Sb, S); // → row-major S[MT,LB]

  // 2) per-row online softmax on row-major S (scale + causal mask), update m/l,
  //    rescale o_buf, and build P[MT,LB] row-major for the P·V matmul.
  for (int i = 0; i < ATT_M; i++) {
    float *si = S + i * ATT_LB;
    const int qrow = qbase + i;
    for (int j = 0; j < ATT_LB; j++) {
#if ATT_CAUSAL
      si[j] = (kbase + j > qrow) ? -1.0e4f : si[j] * ATT_LOG2E;
#else
      si[j] *= ATT_LOG2E;
#endif
    }
    aie::vector<float, ATT_LB> sv = aie::load_v<ATT_LB>(si);
    float bmax = aie::reduce_max(sv);
    float m_old = m_buf[i];
    bool first = (m_old < -1.0e30f);
    float m_new = first ? bmax : (m_old > bmax ? m_old : bmax);
    aie::vector<float, ATT_LB> ein = aie::add(sv, aie::broadcast<float, ATT_LB>(-m_new));
    aie::vector<bfloat16, ATT_LB> pb = aie::exp2<bfloat16>(ein);
    aie::store_v(P + i * ATT_LB, pb);
    float block_l = 0.f;
    for (int j = 0; j < ATT_LB; j++) block_l += (float)pb.get(j);
    if (first) {
      l_buf[i] = block_l;
      for (int d = 0; d < ATT_D; d++) o_buf[i * ATT_D + d] = 0.f;
    } else {
      float corr = (float)aie::exp2<bfloat16>(aie::broadcast<float, ATT_LB>(m_old - m_new)).get(0);
      l_buf[i] = l_buf[i] * corr + block_l;
      for (int d = 0; d < ATT_D; d++) o_buf[i * ATT_D + d] *= corr;
    }
    m_buf[i] = m_new;
  }

  // 3) O_block = P·V on the MAC array; add to the running output.
  tile_rm<ATT_M, ATT_LB, MM_R, MM_S>(P, Pb); // A
  tile_rm<ATT_LB, ATT_D, MM_S, MM_T>(V, Vb); // B
  mm_blk<ATT_M, ATT_LB, ATT_D>(Pb, Vb, Ob);
  untile_rm<ATT_M, ATT_D, MM_R, MM_T>(Ob, O);
  for (int i = 0; i < ATT_M * ATT_D; i++) o_buf[i] += O[i];
}

void attn_finalize(float *restrict o_buf, float *restrict l_buf,
                   float *restrict m_buf, bfloat16 *restrict out) {
  event0();
  for (int i = 0; i < ATT_M; i++) {
    float inv = 1.f / l_buf[i];
    for (int d = 0; d < ATT_D; d++)
      out[i * ATT_D + d] = (bfloat16)(o_buf[i * ATT_D + d] * inv);
    m_buf[i] = -3.0e38f;
    l_buf[i] = 0.f;
  }
}

} // extern "C"
