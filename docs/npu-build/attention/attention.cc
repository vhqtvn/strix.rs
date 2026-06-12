//===- attention.cc -----------------------------------------*- C++ -*-===//
//
// Fused single-core attention for the strix.rs NPU stage-2/3 prototype:
//   scores[M,L] = Q[M,D] · K[L,D]^T ; probs = softmax(scores) ; out[M,D] = probs · V[L,D]
// All intermediates live in this core's local memory (no host round-trip) — the
// point of the fusion. bf16 throughout (matches softmax.cc + mm.cc bf16 path → no
// dtype-conversion kernels between stages). Fixed shape via -D macros.
//
// FLASH/ONLINE-SOFTMAX FORM: the L (key/value) axis is processed in blocks of
// ATT_LB with a running max (m) + running denominator (l) + running unnormalized
// output (oacc), rescaling the carried state by exp2(m_old - m_new) per block.
// This is mathematically EXACT (== global softmax) and is the form required once
// K/V stream block-by-block (stage 3) — here K/V are still resident so we can
// validate the online MATH independently of the streaming dataflow. The exp is
// done in the log2 domain (no scalar exp2 exists on aie2p — only the vector
// aie::exp2 — so the running-max correction goes through a 1-lane broadcast+exp2).
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
// KV block size for the online/flash softmax. Must divide ATT_L. The block's
// scaled scores are processed as one aie::vector<float, ATT_LB>.
#ifndef ATT_LB
#define ATT_LB 32
#endif

// D-axis vector width (bf16). D is a multiple of 32 for all real head_dims.
#define ATT_VEC 32
// log2(e): exp(x) == exp2(x * LOG2E). Full-precision (float domain) so the
// softmax matches the natural-exp CPU reference exactly modulo bf16 inputs.
#define ATT_LOG2E 1.4426950408889634f

extern "C" {

// Inputs packed into one buffer (a single AIE tile has only 2 input DMA channels,
// so Q‖K‖V share one stream): qkv = [Q (M*D) | K (L*D) | V (L*D)].
void attention_bf16(bfloat16 *restrict qkv, bfloat16 *restrict out) {
  event0();
  bfloat16 *restrict q = qkv;
  bfloat16 *restrict k = qkv + ATT_M * ATT_D;
  bfloat16 *restrict v = qkv + ATT_M * ATT_D + ATT_L * ATT_D;
  constexpr int DCHUNKS = ATT_D / ATT_VEC;
  constexpr int NBLK = ATT_L / ATT_LB;

  for (int i = 0; i < ATT_M; i++) {
    const bfloat16 *qi = q + i * ATT_D;
    float m = -3.0e38f; // running max (already × LOG2E, i.e. log2 domain)
    float l = 0.f;      // running softmax denominator
    float oacc[ATT_D];  // running unnormalized output (scalar, float)
    for (int d = 0; d < ATT_D; d++) oacc[d] = 0.f;

    for (int b = 0; b < NBLK; b++) {
      const int base = b * ATT_LB;

      // 1) block scaled scores: sblk[jj] = (q_i · k_{base+jj}) × LOG2E.
      // D-axis dot is vectorized via aie::mac into a lane-wise accum.
      alignas(128) float sblk[ATT_LB];
      for (int jj = 0; jj < ATT_LB; jj++) {
        const bfloat16 *kj = k + (base + jj) * ATT_D;
        aie::accum<accfloat, ATT_VEC> sacc = aie::zeros<accfloat, ATT_VEC>();
        for (int c = 0; c < DCHUNKS; c++) {
          aie::vector<bfloat16, ATT_VEC> qv = aie::load_v<ATT_VEC>(qi + c * ATT_VEC);
          aie::vector<bfloat16, ATT_VEC> kv = aie::load_v<ATT_VEC>(kj + c * ATT_VEC);
          sacc = aie::mac(sacc, qv, kv);
        }
        sblk[jj] = aie::reduce_add(sacc.to_vector<float>()) * ATT_LOG2E;
      }
      aie::vector<float, ATT_LB> sv = aie::load_v<ATT_LB>(sblk);

      // 2) running max over this block.
      float bmax = aie::reduce_max(sv);
      float m_new = (b == 0) ? bmax : (m > bmax ? m : bmax);

      // 3) probs = exp2(sv - m_new). exp2 on aie2p always yields bf16 (matches
      // softmax.cc). add with negated broadcast — float-vec add is the proven
      // primitive; sub is avoided. block_l summed in float from scalar lanes.
      aie::vector<float, ATT_LB> ein = aie::add(sv, aie::broadcast<float, ATT_LB>(-m_new));
      aie::vector<bfloat16, ATT_LB> probsb = aie::exp2<bfloat16>(ein);
      float block_l = 0.f;
      for (int jj = 0; jj < ATT_LB; jj++) block_l += (float)probsb.get(jj);

      // 4) rescale the carried running state by exp2(m_old - m_new) ≤ 1.
      if (b == 0) {
        l = block_l; // oacc already zero
      } else {
        float corr =
            (float)aie::exp2<bfloat16>(aie::broadcast<float, ATT_LB>(m - m_new)).get(0);
        l = l * corr + block_l;
        for (int d = 0; d < ATT_D; d++) oacc[d] *= corr;
      }

      // 5) accumulate probs · V for this block (scalar over D — vectorized in the
      // streaming pass; correctness-first here).
      for (int jj = 0; jj < ATT_LB; jj++) {
        float p = (float)probsb.get(jj);
        const bfloat16 *vj = v + (base + jj) * ATT_D;
        for (int d = 0; d < ATT_D; d++) oacc[d] += p * (float)vj[d];
      }
      m = m_new;
    }

    // 6) normalize by the final denominator and store.
    float inv = 1.f / l;
    bfloat16 *oi = out + i * ATT_D;
    for (int d = 0; d < ATT_D; d++) oi[d] = (bfloat16)(oacc[d] * inv);
  }
}

} // extern "C"
