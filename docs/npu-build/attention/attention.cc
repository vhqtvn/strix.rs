//===- attention.cc -----------------------------------------*- C++ -*-===//
//
// Fused single-core attention for the strix.rs NPU stage-2 prototype:
//   scores[M,L] = Q[M,D] · K[L,D]^T ; probs = softmax(scores) ; out[M,D] = probs · V[L,D]
// All intermediates live in this core's local memory (no host round-trip) — the
// point of the fusion. Scalar matmuls + the proven softmax_simple_bf16 (its
// log2e-scaled exp is numerically sensitive, so we reuse it verbatim). Fixed
// shape via -D macros (ATT_M/ATT_L/ATT_D). bf16 throughout (matches softmax.cc
// + mm.cc bf16 path → no dtype-conversion kernels between stages).
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

// Bring in softmax_simple_bf16 (it's defined non-extern in softmax.cc).
#include "softmax.cc"

#ifndef ATT_M
#define ATT_M 64
#endif
#ifndef ATT_L
#define ATT_L 64
#endif
#ifndef ATT_D
#define ATT_D 64
#endif

extern "C" {

// Inputs packed into one buffer (a single AIE tile has only 2 input DMA channels,
// so Q‖K‖V share one stream): qkv = [Q (M*D) | K (L*D) | V (L*D)].
void attention_bf16(bfloat16 *restrict qkv, bfloat16 *restrict out) {
  event0();
  bfloat16 *restrict q = qkv;
  bfloat16 *restrict k = qkv + ATT_M * ATT_D;
  bfloat16 *restrict v = qkv + ATT_M * ATT_D + ATT_L * ATT_D;
  bfloat16 scores[ATT_L];
  bfloat16 probs[ATT_L];
  for (int i = 0; i < ATT_M; i++) {
    const bfloat16 *qi = q + i * ATT_D;
    // scores[j] = sum_d q[i,d] * k[j,d]   (Q · K^T)
    for (int j = 0; j < ATT_L; j++) {
      const bfloat16 *kj = k + j * ATT_D;
      float s = 0.f;
      for (int d = 0; d < ATT_D; d++)
        s += (float)qi[d] * (float)kj[d];
      scores[j] = (bfloat16)s;
    }
    // probs = softmax(scores) over the L axis (reuse the validated kernel).
    softmax_simple_bf16(scores, probs, ATT_L);
    // out[i,d] = sum_j probs[j] * v[j,d]   (probs · V)
    bfloat16 *oi = out + i * ATT_D;
    for (int d = 0; d < ATT_D; d++) {
      float o = 0.f;
      for (int j = 0; j < ATT_L; j++)
        o += (float)probs[j] * (float)v[j * ATT_D + d];
      oi[d] = (bfloat16)o;
    }
  }
}

} // extern "C"
