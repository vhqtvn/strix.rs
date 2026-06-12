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

// Vector width for the D-axis reductions. D is a multiple of 32 for every real
// head_dim (64/128/256), so no scalar remainder loop is needed.
#define ATT_VEC 32

extern "C" {

// Inputs packed into one buffer (a single AIE tile has only 2 input DMA channels,
// so Q‖K‖V share one stream): qkv = [Q (M*D) | K (L*D) | V (L*D)].
//
// Vectorized (aie::load_v / aie::mul / aie::reduce_add) over the D axis — keeps
// the plain row-major layout (no blocked mmul tiling / DMA TensorTiler), so it's
// a low-risk speedup over the scalar version while staying bit-comparable.
void attention_bf16(bfloat16 *restrict qkv, bfloat16 *restrict out) {
  event0();
  bfloat16 *restrict q = qkv;
  bfloat16 *restrict k = qkv + ATT_M * ATT_D;
  bfloat16 *restrict v = qkv + ATT_M * ATT_D + ATT_L * ATT_D;
  // softmax_simple_bf16 uses aie::cbegin_restrict_vector<32> (ALIGNED 512-bit
  // vector loads), so scores/probs must be 64-byte aligned — plain stack arrays
  // are only 2-byte aligned and the unaligned vector load reads garbage → NaN.
  alignas(64) bfloat16 scores[ATT_L];
  alignas(64) bfloat16 probs[ATT_L];
  constexpr int DCHUNKS = ATT_D / ATT_VEC;
  for (int i = 0; i < ATT_M; i++) {
    const bfloat16 *qi = q + i * ATT_D;
    // scores[j] = sum_d q[i,d] * k[j,d]   (Q · K^T), D-axis vectorized via mac
    // into a lane-wise accum, then reduce_add across lanes.
    for (int j = 0; j < ATT_L; j++) {
      const bfloat16 *kj = k + j * ATT_D;
      aie::accum<accfloat, ATT_VEC> sacc = aie::zeros<accfloat, ATT_VEC>();
      for (int c = 0; c < DCHUNKS; c++) {
        aie::vector<bfloat16, ATT_VEC> qv = aie::load_v<ATT_VEC>(qi + c * ATT_VEC);
        aie::vector<bfloat16, ATT_VEC> kv = aie::load_v<ATT_VEC>(kj + c * ATT_VEC);
        sacc = aie::mac(sacc, qv, kv);
      }
      scores[j] = (bfloat16)aie::reduce_add(sacc.to_vector<float>());
    }
    // probs = softmax(scores) over the L axis (reuse the validated kernel).
    softmax_simple_bf16(scores, probs, ATT_L);
    // out[i,:] = sum_j probs[j] * v[j,:]   (probs · V), j-outer scalar-vector mac;
    // D held in DCHUNKS float accumulators (DCHUNKS ≤ 8 for D ≤ 256).
    bfloat16 *oi = out + i * ATT_D;
    aie::accum<accfloat, ATT_VEC> oacc[DCHUNKS];
    for (int c = 0; c < DCHUNKS; c++) oacc[c] = aie::zeros<accfloat, ATT_VEC>();
    for (int j = 0; j < ATT_L; j++) {
      const bfloat16 pj = probs[j];
      const bfloat16 *vj = v + j * ATT_D;
      for (int c = 0; c < DCHUNKS; c++) {
        aie::vector<bfloat16, ATT_VEC> vv = aie::load_v<ATT_VEC>(vj + c * ATT_VEC);
        oacc[c] = aie::mac(oacc[c], pj, vv);
      }
    }
    for (int c = 0; c < DCHUNKS; c++)
      aie::store_v(oi + c * ATT_VEC, oacc[c].to_vector<bfloat16>());
  }
}

} // extern "C"
