//===- attention.cc -----------------------------------------*- C++ -*-===//
//
// Fused single-core attention for the strix.rs NPU stage-3 prototype, in
// STREAMING flash form: K/V arrive block-by-block (one (K_block‖V_block) at a
// time) so the L axis is NOT resident — this lifts the 64 KB-tile ceiling that
// capped the resident one-shot kernel at 64×64×64.
//
//   scores[i,j] = q_i · k_j ; probs = softmax_L(scores) ; out[i,:] = Σ_j probs·v_j
//
// Online (flash) softmax: per query row i we carry a running max m[i], running
// denom l[i], and running unnormalized output o[i,:] across blocks, rescaling
// the carried state by exp2(m_old - m_new) each block. m/l/o live in persistent
// core-local Buffers (IRON `Buffer`), Q is resident (loaded once), only ONE K/V
// block is resident at a time. bf16 in/out; running state in f32. exp is hand-
// rolled in the log2 domain (aie2p has only the vector aie::exp2 → bf16).
//
//   attn_block_first(q, kv, m, l, o) : block 0 — initializes m/l/o from this block
//   attn_block      (q, kv, m, l, o) : blocks 1.. — online update w/ correction
//   attn_finalize   (o, l, out)      : out[i,:] = o[i,:] / l[i]  (→ bf16)
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef ATT_M
#define ATT_M 64
#endif
#ifndef ATT_L
#define ATT_L 64
#endif
#ifndef ATT_D
#define ATT_D 64
#endif
// KV block size (rows of K/V streamed per step). Must divide ATT_L.
#ifndef ATT_LB
#define ATT_LB 32
#endif

#ifndef ATT_CAUSAL
#define ATT_CAUSAL 0
#endif

#define ATT_VEC 32 // bf16 D-axis vector width (D is a multiple of 32)
#define ATT_LOG2E 1.4426950408889634f

// One query row × one K/V block: update running (m,l,o). The first block is
// detected from the running max sentinel (-3e38, the Buffer init) so a single
// kernel handles every block in a uniform streaming loop — no exp2(-inf).
static inline void block_row(const bfloat16 *restrict qi,
                             const bfloat16 *restrict kk,
                             const bfloat16 *restrict vv, float *restrict m_p,
                             float *restrict l_p, float *restrict oi,
                             int qrow_g, int kbase) {
  constexpr int DCHUNKS = ATT_D / ATT_VEC;
  // scores[LB] = Σ_d q_i[d] · KT[d, :]   (KT = this block's K TRANSPOSED to
  // [D, LB], packed that way by the host). The key reduction lives in the LB
  // lanes via scalar-vector mac over d — NO per-key cross-lane aie::reduce_add,
  // which was the measured bottleneck (a horizontal reduction per key/row).
  alignas(128) float sblk[ATT_LB];
  {
    aie::accum<accfloat, ATT_LB> sacc = aie::zeros<accfloat, ATT_LB>();
    for (int d = 0; d < ATT_D; d++)
      sacc = aie::mac(sacc, qi[d], aie::load_v<ATT_LB>(kk + d * ATT_LB));
    aie::store_v(sblk, sacc.to_vector<float>());
  }
  for (int jj = 0; jj < ATT_LB; jj++) {
#if ATT_CAUSAL
    // causal mask: query qrow_g attends only keys ≤ qrow_g. Use a MODERATE
    // log2-domain sentinel (-1e4): exp2(-1e4) underflows to 0 (real scaled
    // scores are ≤ ~±40), while exp2's floor(x)→int32 exponent extraction would
    // OVERFLOW on -3e38 and NOT underflow → masked keys would wrongly contribute.
    if (kbase + jj > qrow_g) {
      sblk[jj] = -1.0e4f;
      continue;
    }
#endif
    sblk[jj] *= ATT_LOG2E;
  }
  aie::vector<float, ATT_LB> sv = aie::load_v<ATT_LB>(sblk);

  float bmax = aie::reduce_max(sv);
  float m_old = *m_p;
  bool first = (m_old < -1.0e30f); // sentinel from the Buffer init
  float m_new = first ? bmax : (m_old > bmax ? m_old : bmax);

  aie::vector<float, ATT_LB> ein = aie::add(sv, aie::broadcast<float, ATT_LB>(-m_new));
  aie::vector<bfloat16, ATT_LB> probsb = aie::exp2<bfloat16>(ein);
  float block_l = 0.f;
  for (int jj = 0; jj < ATT_LB; jj++) block_l += (float)probsb.get(jj);

  if (first) {
    *l_p = block_l;
    for (int d = 0; d < ATT_D; d++) oi[d] = 0.f;
  } else {
    float corr =
        (float)aie::exp2<bfloat16>(aie::broadcast<float, ATT_LB>(m_old - m_new)).get(0);
    *l_p = *l_p * corr + block_l;
    for (int d = 0; d < ATT_D; d++) oi[d] *= corr;
  }
  // V accumulation: O[:] += Σ_j probs[j] · V[j, :]. jj-outer so each prob lane
  // is extracted ONCE (not per d-chunk) — DCHUNKS f32 accums held across j.
  // Masked jj have probs≈0 so their mac adds ~0.
  aie::accum<accfloat, ATT_VEC> acc[DCHUNKS];
  for (int c = 0; c < DCHUNKS; c++) acc[c] = aie::zeros<accfloat, ATT_VEC>();
  for (int jj = 0; jj < ATT_LB; jj++) {
    bfloat16 p = probsb.get(jj);
    const bfloat16 *vj = vv + jj * ATT_D;
    for (int c = 0; c < DCHUNKS; c++)
      acc[c] = aie::mac(acc[c], p, aie::load_v<ATT_VEC>(vj + c * ATT_VEC));
  }
  for (int c = 0; c < DCHUNKS; c++) {
    aie::vector<float, ATT_VEC> ov = aie::load_v<ATT_VEC>(oi + c * ATT_VEC);
    aie::store_v(oi + c * ATT_VEC, aie::add(ov, acc[c].to_vector<float>()));
  }
  *m_p = m_new;
}

extern "C" {

// kv = [KT_block (D×ATT_LB, K TRANSPOSED) ‖ V_block (ATT_LB×D)] for the current
// block. K is transposed by the host so the score matvec reads contiguous LB-key
// columns. One kernel handles every block (first detected from the m sentinel).
// idx_buf[0] = current query-tile index qt, idx_buf[1] = current block index kb
// (advanced here; reset/advanced across tiles by attn_finalize). Used for causal
// masking — the global query row is qt*ATT_M + i, the block's key base is kb*ATT_LB.
// qt (query-tile index) and kb (block index) are passed directly from the
// dataflow loops — avoids a persisted counter Buffer (whose init wasn't honored
// at runtime). Global query row = qt*ATT_M + i; block key base = kb*ATT_LB.
void attn_block(bfloat16 *restrict q, bfloat16 *restrict kv,
                float *restrict m_buf, float *restrict l_buf,
                float *restrict o_buf, int32_t qt, int32_t kb) {
  event0();
  const bfloat16 *kk = kv;
  const bfloat16 *vv = kv + ATT_LB * ATT_D;
  int qbase = qt * ATT_M, kbase = kb * ATT_LB;
  for (int i = 0; i < ATT_M; i++)
    block_row(q + i * ATT_D, kk, vv, m_buf + i, l_buf + i, o_buf + i * ATT_D,
              qbase + i, kbase);
}

// Normalize one query tile and RE-ARM the running state for the next tile:
// m←sentinel, l←0 (o_buf is overwritten by the next tile's first block). This
// lets a single set of [ATT_M]-sized Buffers be reused across query tiles.
void attn_finalize(float *restrict o_buf, float *restrict l_buf,
                   float *restrict m_buf, bfloat16 *restrict out) {
  event0();
  for (int i = 0; i < ATT_M; i++) {
    float inv = 1.f / l_buf[i];
    const float *oi = o_buf + i * ATT_D;
    bfloat16 *outi = out + i * ATT_D;
    for (int d = 0; d < ATT_D; d++) outi[d] = (bfloat16)(oi[d] * inv);
    m_buf[i] = -3.0e38f; // re-arm running state for the next query tile
    l_buf[i] = 0.f;
  }
}

} // extern "C"
