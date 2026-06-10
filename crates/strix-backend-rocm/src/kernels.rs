//! All decode kernels in one HIP C++ module (compiled once via hiprtc).
//! Ported 1:1 from the Vulkan WGSL kernels; shared-memory reductions (no
//! subgroup ops) for portability. Self-contained — no hip headers (hiprtc lacks
//! them on its include path), so f16 decode is done by hand (with subnormal
//! handling — the Gemma Q6_K `d` scales are subnormal f16).

#[cfg(feature = "rocm")]
pub(crate) const KERNELS: &str = r#"
__device__ __forceinline__ float h2f(unsigned short h) {
    unsigned s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0u) { float v = (float)m * 5.9604644775390625e-8f; return s ? -v : v; }
    union { unsigned u; float f; } c;
    c.u = (s << 31) | ((e + 112u) << 23) | (m << 13);
    return c.f;
}

// f32 -> f16 (round to nearest). KV-cache values are post-norm/RoPE (~O(1)).
__device__ __forceinline__ unsigned short f2h(float f) {
    union { unsigned u; float fv; } c; c.fv = f;
    unsigned x = c.u, sign = (x >> 16) & 0x8000u, m = x & 0x7fffffu;
    int e = (int)((x >> 23) & 0xffu) - 112;
    if (e >= 31) return (unsigned short)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (unsigned short)sign;
        m |= 0x800000u; int sh = 14 - e; unsigned h = m >> sh;
        if ((m >> (sh - 1)) & 1u) h += 1u;
        return (unsigned short)(sign | h);
    }
    unsigned h = ((unsigned)e << 10) | (m >> 13);
    if (m & 0x1000u) h += 1u;
    return (unsigned short)(sign | h);
}
// Load a KV element that is f16 (kvf16!=0) or f32, from a base pointer passed as
// const float* (reinterpreted). Lets the SDPA kernels serve either cache format.
#define KVLD(p, i, kvf16) ((kvf16) ? h2f(((const unsigned short*)(p))[(i)]) : (p)[(i)])

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

// Q4_0 GEMV: f16 scales (1/block) + uint4 quants (16 nibble bytes/block).
extern "C" __global__ void q4_gemv(const unsigned short* __restrict__ scales,
                                   const uint4* __restrict__ quants,
                                   const float* __restrict__ x,
                                   float* __restrict__ y, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int nblocks = in_dim / 32, t = threadIdx.x, rowblk = row * nblocks;
    float acc = 0.f;
    for (int b = t; b < nblocks; b += 32) {
        int blk = rowblk + b;
        float d = h2f(scales[blk]);
        uint4 q = quants[blk];
        unsigned ws[4] = {q.x, q.y, q.z, q.w};
        int xbase = b * 32;
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            unsigned word = ws[w];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                int j = w * 4 + k;
                unsigned byte = (word >> (k * 8)) & 0xffu;
                float lo = (float)(byte & 0xf) - 8.f, hi = (float)(byte >> 4) - 8.f;
                acc += d * (lo * x[xbase + j] + hi * x[xbase + j + 16]);
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (t == 0) y[row] = acc;
}

// Quantize activations to Q8 for the dp4a GEMV: per 32-block, scale=max|x|/127,
// q=round(x/scale). Output: xlo/xhi = packed int8 (x[0..15] / x[16..31] as 4
// char4 each), xd = per-block f32 scale, xsum = per-block sum(q) (for the
// (w-8)·x = w·x - 8·sum(x) correction). grid=nblocks, block=32.
extern "C" __global__ void xquant(const float* __restrict__ x, char4* __restrict__ xlo,
                                  char4* __restrict__ xhi, float* __restrict__ xd,
                                  int* __restrict__ xsum, int n) {
    int blk = blockIdx.x, t = threadIdx.x, base = blk * 32;
    __shared__ signed char sh[32];
    __shared__ float ssc;
    float xv = x[base + t];
    float a = fabsf(xv);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_down(a, o));
    if (t == 0) ssc = a / 127.f;
    __syncthreads();
    float inv = ssc > 0.f ? 1.f / ssc : 0.f;
    int qi = (int)rintf(xv * inv);
    qi = max(-127, min(127, qi));
    sh[t] = (signed char)qi;
    int sm = qi;
    for (int o = 16; o > 0; o >>= 1) sm += __shfl_down(sm, o);
    __syncthreads();
    if (t == 0) { xd[blk] = ssc; xsum[blk] = sm; }
    if (t < 4) xlo[blk * 4 + t] = *(char4*)&sh[t * 4];
    else if (t < 8) xhi[blk * 4 + (t - 4)] = *(char4*)&sh[16 + (t - 4) * 4];
}

// RMSNorm directly into Q8 blocks. This avoids the latency-bound rmsnorm
// launch plus the follow-up xquant launch for inputs that are consumed only by
// Q4 GEMV/GEMM kernels. grid=n_rows, block=256, dim must be divisible by 32.
extern "C" __global__ void rmsnorm_xquant(const float* __restrict__ x, const float* __restrict__ w,
                                          char4* __restrict__ xlo, char4* __restrict__ xhi,
                                          float* __restrict__ xd, int* __restrict__ xsum,
                                          int dim, int has_w, float eps) {
    int row = blockIdx.x, t = threadIdx.x, nb = dim / 32;
    const float* xr = x + (size_t)row * dim;
    __shared__ float red[256];
    float ss = 0.f;
    for (int i = t; i < dim; i += 256) { float v = xr[i]; ss += v * v; }
    red[t] = ss; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float rs = rsqrtf(red[0] / dim + eps);
    if (t >= nb) return;

    int base = t * 32;
    signed char qv[32];
    float mx = 0.f;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float g = has_w ? w[base + j] : 1.f;
        float v = xr[base + j] * rs * g;
        mx = fmaxf(mx, fabsf(v));
    }
    float sc = mx / 127.f;
    float inv = sc > 0.f ? 1.f / sc : 0.f;
    int sm = 0;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float g = has_w ? w[base + j] : 1.f;
        float v = xr[base + j] * rs * g;
        int qi = (int)rintf(v * inv);
        qi = max(-127, min(127, qi));
        qv[j] = (signed char)qi;
        sm += qi;
    }
    int ob = row * nb + t;
    xd[ob] = sc;
    xsum[ob] = sm;
    xlo[ob * 4 + 0] = make_char4(qv[0], qv[1], qv[2], qv[3]);
    xlo[ob * 4 + 1] = make_char4(qv[4], qv[5], qv[6], qv[7]);
    xlo[ob * 4 + 2] = make_char4(qv[8], qv[9], qv[10], qv[11]);
    xlo[ob * 4 + 3] = make_char4(qv[12], qv[13], qv[14], qv[15]);
    xhi[ob * 4 + 0] = make_char4(qv[16], qv[17], qv[18], qv[19]);
    xhi[ob * 4 + 1] = make_char4(qv[20], qv[21], qv[22], qv[23]);
    xhi[ob * 4 + 2] = make_char4(qv[24], qv[25], qv[26], qv[27]);
    xhi[ob * 4 + 3] = make_char4(qv[28], qv[29], qv[30], qv[31]);
}

// GeGLU directly into Q8 blocks for the down projection. Saves a full f32
// activation write/read and one launch on the hot MLP path.
extern "C" __global__ void geglu_xquant(const float* __restrict__ gate, const float* __restrict__ up,
                                        char4* __restrict__ xlo, char4* __restrict__ xhi,
                                        float* __restrict__ xd, int* __restrict__ xsum, int n) {
    int blk = blockIdx.x, t = threadIdx.x, base = blk * 32;
    signed char qv[32];
    float vals[32];
    float mx = 0.f;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        int i = base + j;
        float g = gate[i];
        float gt = 0.5f * g * (1.f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
        float v = gt * up[i];
        vals[j] = v;
        mx = fmaxf(mx, fabsf(v));
    }
    float sc = mx / 127.f;
    float inv = sc > 0.f ? 1.f / sc : 0.f;
    int sm = 0;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        int qi = (int)rintf(vals[j] * inv);
        qi = max(-127, min(127, qi));
        qv[j] = (signed char)qi;
        sm += qi;
    }
    xd[blk] = sc;
    xsum[blk] = sm;
    xlo[blk * 4 + 0] = make_char4(qv[0], qv[1], qv[2], qv[3]);
    xlo[blk * 4 + 1] = make_char4(qv[4], qv[5], qv[6], qv[7]);
    xlo[blk * 4 + 2] = make_char4(qv[8], qv[9], qv[10], qv[11]);
    xlo[blk * 4 + 3] = make_char4(qv[12], qv[13], qv[14], qv[15]);
    xhi[blk * 4 + 0] = make_char4(qv[16], qv[17], qv[18], qv[19]);
    xhi[blk * 4 + 1] = make_char4(qv[20], qv[21], qv[22], qv[23]);
    xhi[blk * 4 + 2] = make_char4(qv[24], qv[25], qv[26], qv[27]);
    xhi[blk * 4 + 3] = make_char4(qv[28], qv[29], qv[30], qv[31]);
}

__device__ __forceinline__ void q4_gemv_dp_row(const unsigned short* __restrict__ scales,
                                               const uint4* __restrict__ quants,
                                               const char4* __restrict__ xlo, const char4* __restrict__ xhi,
                                               const float* __restrict__ xd, const int* __restrict__ xsum,
                                               float* __restrict__ y, int in_dim, int out_dim, int row, int lane) {
    if (row >= out_dim) return;
    int nb = in_dim / 32, rb = row * nb;
    float acc = 0.f;
    for (int b = lane; b < nb; b += 32) {
        int blk = rb + b;
        float dw = h2f(scales[blk]);
        uint4 v = quants[blk];
        unsigned ws[4] = {v.x, v.y, v.z, v.w};
        int isum = 0;
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            unsigned lo = ws[g] & 0x0f0f0f0fu, hi = (ws[g] >> 4) & 0x0f0f0f0fu;
            char4 xl = xlo[b * 4 + g], xh = xhi[b * 4 + g];
            int xli = *(int*)&xl, xhi_i = *(int*)&xh;
            isum = __builtin_amdgcn_sudot4(true, (int)lo, true, xli, isum, false);
            isum = __builtin_amdgcn_sudot4(true, (int)hi, true, xhi_i, isum, false);
        }
        isum -= 8 * xsum[b];
        acc += dw * xd[b] * (float)isum;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (lane == 0) y[row] = acc;
}

// Q4_0 GEMV via dp4a (v_dot4_i32_iu8). Raw nibbles (0..15) dotted with the Q8
// activation, corrected by -8*sum(x). 8 rows / 256-thread block (one wave/row).
extern "C" __global__ void q4_gemv_dp(const unsigned short* __restrict__ scales,
                                      const uint4* __restrict__ quants,
                                      const char4* __restrict__ xlo, const char4* __restrict__ xhi,
                                      const float* __restrict__ xd, const int* __restrict__ xsum,
                                      float* __restrict__ y, int in_dim, int out_dim) {
    int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= out_dim) return;
    int nb = in_dim / 32, rb = row * nb;
    float acc = 0.f;
    for (int b = lane; b < nb; b += 32) {
        int blk = rb + b;
        float dw = h2f(scales[blk]);
        uint4 v = quants[blk];
        unsigned ws[4] = {v.x, v.y, v.z, v.w};
        int isum = 0;
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            unsigned lo = ws[g] & 0x0f0f0f0fu, hi = (ws[g] >> 4) & 0x0f0f0f0fu;
            char4 xl = xlo[b * 4 + g], xh = xhi[b * 4 + g];
            int xli = *(int*)&xl, xhi_i = *(int*)&xh;
            isum = __builtin_amdgcn_sudot4(true, (int)lo, true, xli, isum, false);
            isum = __builtin_amdgcn_sudot4(true, (int)hi, true, xhi_i, isum, false);
        }
        isum -= 8 * xsum[b];
        acc += dw * xd[b] * (float)isum;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (lane == 0) y[row] = acc;
}

// Grouped Q4_0 GEMV via dp4a for projections sharing the same activation
// quantization (q/k/v and gate/up). This removes 1-2 kernel launches per group.
extern "C" __global__ void q4_gemv_dp2(const unsigned short* __restrict__ sc0,
                                       const uint4* __restrict__ q0, float* __restrict__ y0, int n0,
                                       const unsigned short* __restrict__ sc1,
                                       const uint4* __restrict__ q1, float* __restrict__ y1, int n1,
                                       const char4* __restrict__ xlo, const char4* __restrict__ xhi,
                                       const float* __restrict__ xd, const int* __restrict__ xsum,
                                       int in_dim) {
    int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, b = blockIdx.x;
    int b0 = (n0 + 7) >> 3;
    if (b < b0) {
        q4_gemv_dp_row(sc0, q0, xlo, xhi, xd, xsum, y0, in_dim, n0, b * 8 + wave, lane);
    } else {
        int bb = b - b0;
        q4_gemv_dp_row(sc1, q1, xlo, xhi, xd, xsum, y1, in_dim, n1, bb * 8 + wave, lane);
    }
}

extern "C" __global__ void q4_gemv_dp3(const unsigned short* __restrict__ sc0,
                                       const uint4* __restrict__ q0, float* __restrict__ y0, int n0,
                                       const unsigned short* __restrict__ sc1,
                                       const uint4* __restrict__ q1, float* __restrict__ y1, int n1,
                                       const unsigned short* __restrict__ sc2,
                                       const uint4* __restrict__ q2, float* __restrict__ y2, int n2,
                                       const char4* __restrict__ xlo, const char4* __restrict__ xhi,
                                       const float* __restrict__ xd, const int* __restrict__ xsum,
                                       int in_dim) {
    int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, b = blockIdx.x;
    int b0 = (n0 + 7) >> 3, b1 = (n1 + 7) >> 3;
    if (b < b0) {
        q4_gemv_dp_row(sc0, q0, xlo, xhi, xd, xsum, y0, in_dim, n0, b * 8 + wave, lane);
    } else if (b < b0 + b1) {
        int bb = b - b0;
        q4_gemv_dp_row(sc1, q1, xlo, xhi, xd, xsum, y1, in_dim, n1, bb * 8 + wave, lane);
    } else {
        int bb = b - b0 - b1;
        q4_gemv_dp_row(sc2, q2, xlo, xhi, xd, xsum, y2, in_dim, n2, bb * 8 + wave, lane);
    }
}

// Register-tiled batched Q4_0×Q8 GEMM via dp4a: Y[M,N] = X[M,K] @ W[N,K]^T.
// This is what makes PREFILL compute-bound (each weight read once for the whole
// prompt, not once per token). Output is tiled 64 tokens × 64 rows per block,
// 256 threads, each thread owning a 4×4 micro-tile in registers — 16 independent
// dp4a accumulators give the ILP to hide dp4a latency, with NO cross-thread
// reduction. K is tiled one 32-block at a time, weight + Q8 activation staged in
// shared and reused across the micro-tile. X is pre-quantized by `xquant`.
// Validated bit-exact vs CPU (rel ~1e-7); ~2.2 TFLOP/s for M>=64 on the 890M
// (~2.2× the un-tiled version). Launch: grid=(ceil(N/64),ceil(M/64)), block=256.
extern "C" __global__ void q4_gemm(const unsigned short* __restrict__ scales,
                                   const uint4* __restrict__ quants,
                                   const char4* __restrict__ xlo, const char4* __restrict__ xhi,
                                   const float* __restrict__ xd, const int* __restrict__ xsum,
                                   float* __restrict__ y, int K, int N, int M, int ns) {
    int nb = K / 32, brow = blockIdx.x * 64, bm = blockIdx.y * 64;
    int t = threadIdx.x, tx = t & 15, ty = t >> 4;
    __shared__ float s_wsc[64];
    __shared__ uint4 s_wq[64];
    __shared__ float s_xd[64];
    __shared__ int s_xs[64];
    __shared__ char4 s_xlo[256];  // 64 tokens × 4 char4
    __shared__ char4 s_xhi[256];
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    for (int b = 0; b < nb; b++) {
        for (int i = t; i < 64; i += 256) {
            int r = brow + i;
            if (r < N) { s_wsc[i] = h2f(scales[(size_t)r * nb + b]); s_wq[i] = quants[(size_t)r * nb + b]; }
            else { s_wsc[i] = 0.f; s_wq[i] = make_uint4(0, 0, 0, 0); }
        }
        for (int i = t; i < 64; i += 256) {
            int m = bm + i;
            if (m < M) {
                s_xd[i] = xd[(size_t)m * nb + b];
                s_xs[i] = xsum[(size_t)m * nb + b];
                for (int g = 0; g < 4; g++) {
                    s_xlo[i * 4 + g] = xlo[((size_t)m * nb + b) * 4 + g];
                    s_xhi[i * 4 + g] = xhi[((size_t)m * nb + b) * 4 + g];
                }
            } else {
                s_xd[i] = 0.f; s_xs[i] = 0;
                for (int g = 0; g < 4; g++) { s_xlo[i * 4 + g] = make_char4(0, 0, 0, 0); s_xhi[i * 4 + g] = make_char4(0, 0, 0, 0); }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int im = 0; im < 4; im++) {
            int li = tx * 4 + im;
            float xdv = s_xd[li];
            int xsv = s_xs[li], xl[4], xh[4];
            #pragma unroll
            for (int g = 0; g < 4; g++) { char4 a = s_xlo[li * 4 + g], bb = s_xhi[li * 4 + g]; xl[g] = *(int*)&a; xh[g] = *(int*)&bb; }
            #pragma unroll
            for (int in = 0; in < 4; in++) {
                int lj = ty * 4 + in;
                uint4 q = s_wq[lj];
                unsigned ws[4] = {q.x, q.y, q.z, q.w};
                int isum = 0;
                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    unsigned lo = ws[g] & 0x0f0f0f0fu, hi = (ws[g] >> 4) & 0x0f0f0f0fu;
                    isum = __builtin_amdgcn_sudot4(true, (int)lo, true, xl[g], isum, false);
                    isum = __builtin_amdgcn_sudot4(true, (int)hi, true, xh[g], isum, false);
                }
                acc[im][in] += s_wsc[lj] * xdv * (float)(isum - 8 * xsv);
            }
        }
        __syncthreads();
    }
    for (int im = 0; im < 4; im++) {
        int m = bm + tx * 4 + im;
        if (m >= M) continue;
        for (int in = 0; in < 4; in++) {
            int r = brow + ty * 4 + in;
            if (r < N) y[(size_t)m * ns + r] = acc[im][in];
        }
    }
}

// Shared-staged WMMA Q4_0×Q8 GEMM (matrix cores). Same I/O as q4_gemm, but
// ~1.7-2.3× faster: per K-block all 256 threads stage the 128-row weight nibbles,
// 64-token activations and per-block scales (wsc/xd/xsum) into shared ONCE, then
// each wave runs WMMA on shared fragments and scales from SHARED (no global loads
// in the hot loop — the dp4a/old-WMMA bottleneck). Output tile BN=128 rows ×
// BM=64 tokens; 8 waves = 4(token-tile)×2(row-group); each wave owns 4 of the
// 16×16 output sub-tiles. BN=128 halves activation re-reads vs BN=64. Raw nibbles
// + −8·xsum correction (avoids __vsubss4). Validated bit-exact vs CPU (rel ~1e-6).
// Launch: grid=(ceil(N/128),ceil(M/64)), block=256.
// Double-buffered + BM=128: ping-pong shared buffers let the next K-block's
// global loads overlap the current block's WMMA (one __syncthreads/iter). Output
// tile BN=128 rows × BM=128 tokens; each wave owns 4 row-tiles × 2 token-tiles =
// 8 WMMA-pairs. BM=128 (vs 64) halves weight global traffic per N-tile (each
// 128-row weight tile staged by M/128 blocks instead of M/64). Token-staging
// covers 128 tokens. Launch: grid=(ceil(N/128),ceil(M/128)), block=256.
// SWAR per-byte (nibble-8) → signed int8, branchless. Bytes are 0..15 (<0x80),
// so set the guard bit, subtract 8 (no cross-byte borrow since 0x80-0x08>0), flip
// it back. Baking -8 into signed weights here lets the inner loop accumulate raw
// int32 WMMA output (no per-element -8·xsum correction; matches llama.cpp's MMQ).
#define SUB8(x) ((int)((((unsigned)(x) | 0x80808080u) - 0x08080808u) ^ 0x80808080u))
#define Q4GEMMW_STAGE(bb, buf) do { \
    for (int i = t; i < 128; i += 256) { \
        int r = brow + i; \
        if (r < N) { \
            uint4 q = quants[(size_t)r * nb + (bb)]; unsigned ws[4] = {q.x, q.y, q.z, q.w}; \
            for (int p = 0; p < 4; p++) { sh_blo[buf][i*4+p] = SUB8(ws[p] & 0x0f0f0f0fu); sh_bhi[buf][i*4+p] = SUB8((ws[p] >> 4) & 0x0f0f0f0fu); } \
            sh_wsc[buf][i] = h2f(scales[(size_t)r * nb + (bb)]); \
        } else { for (int p = 0; p < 4; p++) { sh_blo[buf][i*4+p] = 0; sh_bhi[buf][i*4+p] = 0; } sh_wsc[buf][i] = 0.f; } \
    } \
    for (int i = t; i < 128; i += 256) { \
        int mm = bm + i; \
        if (mm < M) { \
            for (int g = 0; g < 4; g++) { char4 cl = xlo[((size_t)mm*nb+(bb))*4+g], ch = xhi[((size_t)mm*nb+(bb))*4+g]; sh_alo[buf][i*4+g] = *(int*)&cl; sh_ahi[buf][i*4+g] = *(int*)&ch; } \
            sh_xd[buf][i] = xd[(size_t)mm * nb + (bb)]; \
        } else { for (int g = 0; g < 4; g++) { sh_alo[buf][i*4+g] = 0; sh_ahi[buf][i*4+g] = 0; } sh_xd[buf][i] = 0.f; } \
    } \
} while (0)

extern "C" __global__ __launch_bounds__(256, 3) void q4_gemm_w(const unsigned short* __restrict__ scales,
                                     const uint4* __restrict__ quants,
                                     const char4* __restrict__ xlo, const char4* __restrict__ xhi,
                                     const float* __restrict__ xd, const int* __restrict__ xsum,
                                     float* __restrict__ y, int K, int N, int M, int ns) {
    int nb = K / 32, brow = blockIdx.x * 128, bm = blockIdx.y * 128;  // BM=128
    int t = threadIdx.x, wave = t >> 5, l = t & 31;
    int wt = wave & 3, wr = wave >> 2;          // token-base 0..3, row-group 0..1
    __shared__ int   sh_blo[2][512], sh_bhi[2][512];  // 128 rows × 4 ints (weight nibbles lo/hi)
    __shared__ int   sh_alo[2][512], sh_ahi[2][512];  // 128 tokens × 4 ints (activation lo/hi)
    __shared__ float sh_wsc[2][128];
    __shared__ float sh_xd[2][128];
    float acc[4][2][8];   // row-tile × token-tile × 8
    for (int rt = 0; rt < 4; rt++) for (int mt = 0; mt < 2; mt++) for (int k = 0; k < 8; k++) acc[rt][mt][k] = 0.f;

    Q4GEMMW_STAGE(0, 0);
    __syncthreads();
    for (int b = 0; b < nb; b++) {
        int cur = b & 1;
        if (b + 1 < nb) Q4GEMMW_STAGE(b + 1, (b + 1) & 1);  // prefetch next (loads overlap WMMA below)
        v4i alo[2], ahi[2]; float xdv[2][8];
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            int tg = wt + mt * 4;                 // token subtile 0..7
            int ar = tg * 16 + (l & 15);
            for (int g = 0; g < 4; g++) { alo[mt][g] = sh_alo[cur][ar*4+g]; ahi[mt][g] = sh_ahi[cur][ar*4+g]; }
            #pragma unroll
            for (int k = 0; k < 8; k++) { int tk = tg * 16 + 2 * k + (l >> 4); xdv[mt][k] = sh_xd[cur][tk]; }
        }
        #pragma unroll
        for (int rt = 0; rt < 4; rt++) {
            int br = wr * 64 + rt * 16 + (l & 15);
            v4i blo, bhi; for (int g = 0; g < 4; g++) { blo[g] = sh_blo[cur][br*4+g]; bhi[g] = sh_bhi[cur][br*4+g]; }
            float wsc = sh_wsc[cur][wr * 64 + rt * 16 + (l & 15)];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                v8i c = {0,0,0,0,0,0,0,0};
                // signed weights (−8 baked in via SUB8) → both operands signed, raw int32 accumulate
                c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, alo[mt], true, blo, c, false);
                c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ahi[mt], true, bhi, c, false);
                #pragma unroll
                for (int k = 0; k < 8; k++) acc[rt][mt][k] += wsc * xdv[mt][k] * (float)c[k];
            }
        }
        __syncthreads();
    }
    for (int rt = 0; rt < 4; rt++) {
        int rowbase = wr * 64 + rt * 16;
        for (int mt = 0; mt < 2; mt++) {
            int tg = wt + mt * 4;
            for (int k = 0; k < 8; k++) {
                int tok = bm + tg * 16 + 2 * k + (l >> 4), row = brow + rowbase + (l & 15);
                if (tok < M && row < N) y[(size_t)tok * ns + row] = acc[rt][mt][k];
            }
        }
    }
}

// Split-K variant of q4_gemm_w for large-K / small-N GEMMs (e.g. ffn_down,
// K=15360 N=3840): without split-K the grid is too few blocks to hide latency.
// blockIdx.z splits K into gridDim.z independent slices; each slice accumulates
// its partial and atomic-adds into y (which MUST be pre-zeroed). Same optimized
// tiling as q4_gemm_w (BM=128, double-buffered, signed-weight bake, reuses
// Q4GEMMW_STAGE). Launch: grid=(ceil(N/128),ceil(M/128),nsl), block=256.
extern "C" __global__ __launch_bounds__(256, 3) void q4_gemm_w_sk(const unsigned short* __restrict__ scales,
                                        const uint4* __restrict__ quants,
                                        const char4* __restrict__ xlo, const char4* __restrict__ xhi,
                                        const float* __restrict__ xd, const int* __restrict__ xsum,
                                        float* __restrict__ y, int K, int N, int M, int ns) {
    int nb = K / 32, brow = blockIdx.x * 128, bm = blockIdx.y * 128;
    int sk = blockIdx.z, nsl = gridDim.z, bps = nb / nsl, b0 = sk * bps, b1 = b0 + bps;
    int t = threadIdx.x, wave = t >> 5, l = t & 31;
    int wt = wave & 3, wr = wave >> 2;
    __shared__ int   sh_blo[2][512], sh_bhi[2][512];
    __shared__ int   sh_alo[2][512], sh_ahi[2][512];
    __shared__ float sh_wsc[2][128];
    __shared__ float sh_xd[2][128];
    float acc[4][2][8];
    for (int rt = 0; rt < 4; rt++) for (int mt = 0; mt < 2; mt++) for (int k = 0; k < 8; k++) acc[rt][mt][k] = 0.f;

    Q4GEMMW_STAGE(b0, 0);
    __syncthreads();
    for (int b = b0; b < b1; b++) {
        int cur = b & 1;
        if (b + 1 < b1) Q4GEMMW_STAGE(b + 1, (b + 1) & 1);
        v4i alo[2], ahi[2]; float xdv[2][8];
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            int tg = wt + mt * 4;
            int ar = tg * 16 + (l & 15);
            for (int g = 0; g < 4; g++) { alo[mt][g] = sh_alo[cur][ar*4+g]; ahi[mt][g] = sh_ahi[cur][ar*4+g]; }
            #pragma unroll
            for (int k = 0; k < 8; k++) { int tk = tg * 16 + 2 * k + (l >> 4); xdv[mt][k] = sh_xd[cur][tk]; }
        }
        #pragma unroll
        for (int rt = 0; rt < 4; rt++) {
            int br = wr * 64 + rt * 16 + (l & 15);
            v4i blo, bhi; for (int g = 0; g < 4; g++) { blo[g] = sh_blo[cur][br*4+g]; bhi[g] = sh_bhi[cur][br*4+g]; }
            float wsc = sh_wsc[cur][wr * 64 + rt * 16 + (l & 15)];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                v8i c = {0,0,0,0,0,0,0,0};
                c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, alo[mt], true, blo, c, false);
                c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ahi[mt], true, bhi, c, false);
                #pragma unroll
                for (int k = 0; k < 8; k++) acc[rt][mt][k] += wsc * xdv[mt][k] * (float)c[k];
            }
        }
        __syncthreads();
    }
    for (int rt = 0; rt < 4; rt++) {
        int rowbase = wr * 64 + rt * 16;
        for (int mt = 0; mt < 2; mt++) {
            int tg = wt + mt * 4;
            for (int k = 0; k < 8; k++) {
                int tok = bm + tg * 16 + 2 * k + (l >> 4), row = brow + rowbase + (l & 15);
                if (tok < M && row < N) atomicAdd(&y[(size_t)tok * ns + row], acc[rt][mt][k]);
            }
        }
    }
}

// Q6_K GEMV. scales: 16 folded f32/superblock; ql: 32 u32; qh: 16 u32.
// 8 rows per 256-thread block (one wave32 per row) so the tall-skinny lm_head
// (out_dim=vocab, in_dim small → few superblocks/row) keeps full occupancy
// instead of one underfilled block per row.
extern "C" __global__ void q6_gemv(const float* __restrict__ scales,
                                   const unsigned* __restrict__ ql,
                                   const unsigned* __restrict__ qh,
                                   const float* __restrict__ x,
                                   float* __restrict__ y, int in_dim, int out_dim) {
    int wave = threadIdx.x >> 5;     // 0..7
    int l = threadIdx.x & 31;        // lane = the `l` index (0..31)
    int row = blockIdx.x * 8 + wave;
    if (row >= out_dim) return;
    int nb = in_dim / 256, rb = row * nb, is = l / 16;
    float acc = 0.f;
    // 32 lanes cooperate on the SAME superblock (coalesced ql/qh reads), looping
    // superblocks serially — each lane handles its `l` slice of the 256 values.
    for (int bi = 0; bi < nb; bi++) {
        int blk = rb + bi, sc = blk * 16, qlb = blk * 32, qhb = blk * 16, xb = bi * 256;
        for (int half = 0; half < 2; half++) {
            int qli0 = half * 64 + l, qli1 = half * 64 + l + 32, qhi = half * 32 + l;
            unsigned b0 = (ql[qlb + qli0 / 4] >> ((qli0 & 3) * 8)) & 0xffu;
            unsigned b1 = (ql[qlb + qli1 / 4] >> ((qli1 & 3) * 8)) & 0xffu;
            unsigned bh = (qh[qhb + qhi / 4] >> ((qhi & 3) * 8)) & 0xffu;
            int q1 = (int)((b0 & 0xf) | ((bh & 3) << 4)) - 32;
            int q2 = (int)((b1 & 0xf) | (((bh >> 2) & 3) << 4)) - 32;
            int q3 = (int)((b0 >> 4) | (((bh >> 4) & 3) << 4)) - 32;
            int q4 = (int)((b1 >> 4) | (((bh >> 6) & 3) << 4)) - 32;
            int pos = half * 128 + l, si = half * 8 + is;
            acc += scales[sc + si] * q1 * x[xb + pos]
                 + scales[sc + si + 2] * q2 * x[xb + pos + 32]
                 + scales[sc + si + 4] * q3 * x[xb + pos + 64]
                 + scales[sc + si + 6] * q4 * x[xb + pos + 96];
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (l == 0) y[row] = acc;
}

// Q8_0 GEMV. Block = 32 vals: f16 scale d + 32 int8. y[row] = sum_b d_b*sum_i(q*x).
// Fold d_b into each term so a single wave-reduce suffices: acc_l = sum_b d_b*q[b*32+l]*x.
// 8 waves/block (one row each), 32 lanes cooperate on one block (coalesced int8 + x).
// grid = ceil(out_dim/8), block = 256. scales: f32[nb*out_dim], quants: int8[nb*32*out_dim].
extern "C" __global__ void q8_0_gemv(const float* __restrict__ scales,
                                     const signed char* __restrict__ quants,
                                     const float* __restrict__ x,
                                     float* __restrict__ y, int in_dim, int out_dim) {
    // 1 wave per row, wg=32 (q4_gemv recipe: more blocks -> better latency hiding)
    int l = threadIdx.x & 31;
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int nb = in_dim / 32, rb = row * nb;
    // 4 blocks per pass, char4 per lane (coalesced 128B/wave). nb % 4 == 0 holds for
    // all our shapes (in_dim multiple of 128).
    float acc = 0.f;
    for (int b4 = 0; b4 < nb; b4 += 4) {
        int bi = b4 + (l >> 3);
        float d = scales[rb + bi];
        int e = (l & 7) * 4;
        const char4 q = *(const char4*)(quants + ((long long)(rb + bi) * 32 + e));
        const float* xb = x + bi * 32 + e;
        acc += d * (q.x * xb[0] + q.y * xb[1] + q.z * xb[2] + q.w * xb[3]);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (l == 0) y[row] = acc;
}

// ===== Fused MoE decode (planar Q8_0, top-k experts in ONE launch each) =====
// PLANAR layout (repacked at upload): scales f32[ne][out*nb], quants i8[ne][out*nb*32]
// → aligned char4 loads, coalesced 128B/wave. ids[k]: routed experts.
//
// y[k][row] = expert ids[k] row · x.  grid=(ceil(out/8), k), block=256 (8 waves).
extern "C" __global__ void q8_moe_gemv(const float* __restrict__ scales,
                                       const signed char* __restrict__ quants,
                                       const int* __restrict__ ids,
                                       const float* __restrict__ x,
                                       float* __restrict__ y, int in_dim, int out_dim) {
    int l = threadIdx.x & 31;
    int row = blockIdx.x, k = blockIdx.y;
    if (row >= out_dim) return;
    int nb = in_dim / 32;
    long long eoff = (long long)ids[k] * out_dim * nb;
    const float* sc = scales + eoff + (long long)row * nb;
    const signed char* qr = quants + (eoff + (long long)row * nb) * 32;
    float acc = 0.f;
    for (int b4 = 0; b4 < nb; b4 += 4) {
        int bi = b4 + (l >> 3);
        float d = sc[bi];
        int e = (l & 7) * 4;
        const char4 q = *(const char4*)(qr + (long long)bi * 32 + e);
        const float* xb = x + bi * 32 + e;
        acc += d * (q.x * xb[0] + q.y * xb[1] + q.z * xb[2] + q.w * xb[3]);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (l == 0) y[(long long)k * out_dim + row] = acc;
}

// ===== Native-Q6_K MoE GEMV (NO repack: 210 B/superblock as in the GGUF) =====
// w = full 3D expert tensor (native bytes), expert stride ebytes. Per superblock:
// ql[128] qh[64] sc[16xi8] d[f16]. 8 waves/block, 32 lanes on the SAME superblock
// (l = the l-index of dequant), looping superblocks serially. grid=(ceil(out/8), k).
extern "C" __global__ void q6_moe_gemv(const unsigned char* __restrict__ w, long long ebytes,
                                       const int* __restrict__ ids,
                                       const float* __restrict__ x,
                                       float* __restrict__ y, int in_dim, int out_dim) {
    int wave = threadIdx.x >> 5, l = threadIdx.x & 31;
    int row = blockIdx.x * 8 + wave, k = blockIdx.y;
    if (row >= out_dim) return;
    int nb = in_dim / 256;
    const unsigned char* rowp = w + (long long)ids[k] * ebytes + (long long)row * nb * 210;
    // lane l covers positions 4l..4l+3 per half: one unaligned u32 ql load (or hi
    // nibbles of same bytes for p>=64) + one u32 qh load -> coalesced.
    int p0 = 4 * l;             // 0..124
    int lo = p0 < 64;           // lanes 0..15: low nibbles, 16..31: high nibbles
    int qb = lo ? p0 : p0 - 64; // ql byte base for the 4 positions
    float acc = 0.f;
    for (int bi = 0; bi < nb; bi++) {
        const unsigned char* blk = rowp + bi * 210;
        const signed char* sc = (const signed char*)(blk + 192);
        float d = h2f(*(const unsigned short*)(blk + 208));
        int xb = bi * 256;
        for (int half = 0; half < 2; half++) {
            const unsigned char* ql = blk + half * 64;
            const unsigned char* qh = blk + 128 + half * 32;
            unsigned q4b, h4b;
            __builtin_memcpy(&q4b, ql + qb, 4);
            __builtin_memcpy(&h4b, qh + (p0 & 31), 4);
            int hsh = (p0 >> 5) * 2;
            float a4 = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int p = p0 + j;
                int qn = (q4b >> (8 * j)) & 0xFF;
                int qv = (lo ? (qn & 0xF) : (qn >> 4)) | ((((h4b >> (8 * j)) >> hsh) & 3) << 4);
                a4 += sc[half * 8 + p / 16] * (qv - 32) * x[xb + half * 128 + p];
            }
            acc += d * a4;
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (l == 0) y[(long long)k * out_dim + row] = acc;
}

// Q8 GEMM rows variant: y[t][row] = W row . xs[t]; grid=(out_dim, m), block=32 (wave/row).
// 8 tokens/block: weight row read once per 8 tokens. grid=(out_dim, ceil(m/8)).
extern "C" __global__ void q8_gemm_rows(const float* __restrict__ scales,
                                        const signed char* __restrict__ quants,
                                        const float* __restrict__ xs,
                                        float* __restrict__ y, int in_dim, int out_dim, int m) {
    int l = threadIdx.x & 31;
    int row = blockIdx.x, t0 = blockIdx.y * 16;
    if (row >= out_dim) return;
    int tm = m - t0;
    if (tm > 16) tm = 16;
    int nb = in_dim / 32, rb = row * nb;
    int e = (l & 7) * 4;
    float acc[16] = {0.f};
    for (int b4 = 0; b4 < nb; b4 += 4) {
        int bi = b4 + (l >> 3);
        float d = scales[rb + bi];
        const char4 q = *(const char4*)(quants + ((long long)(rb + bi) * 32 + e));
        int xo = bi * 32 + e;
        for (int j = 0; j < tm; j++) {
            const float* xb = xs + (long long)(t0 + j) * in_dim + xo;
            acc[j] += d * (q.x * xb[0] + q.y * xb[1] + q.z * xb[2] + q.w * xb[3]);
        }
    }
    for (int j = 0; j < tm; j++) {
        float a = acc[j];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down(a, o);
        if (l == 0) y[(long long)(t0 + j) * out_dim + row] = a;
    }
}

// Native-Q6 GEMM rows for ONE expert: y[t][row] = expert eid row . xs[t].
// grid=(out_dim, m), block=32.
// 8 tokens/block (weights decoded once per 8 tokens). grid=(out_dim, ceil(m/8)).
extern "C" __global__ void q6_gemm_rows(const unsigned char* __restrict__ w, long long ebytes,
                                        int eid, const float* __restrict__ xs,
                                        float* __restrict__ y, int in_dim, int out_dim, int m) {
    int l = threadIdx.x & 31;
    int row = blockIdx.x, t0 = blockIdx.y * 16;
    if (row >= out_dim) return;
    int tm = m - t0;
    if (tm > 16) tm = 16;
    int nb = in_dim / 256;
    const unsigned char* rowp = w + (long long)eid * ebytes + (long long)row * nb * 210;
    int p0 = 4 * l;
    int lo = p0 < 64;
    int qb = lo ? p0 : p0 - 64;
    float acc[16] = {0.f};
    for (int bi = 0; bi < nb; bi++) {
        const unsigned char* blk = rowp + bi * 210;
        const signed char* sc = (const signed char*)(blk + 192);
        float d = h2f(*(const unsigned short*)(blk + 208));
        int xb = bi * 256;
        for (int half = 0; half < 2; half++) {
            const unsigned char* ql = blk + half * 64;
            const unsigned char* qh = blk + 128 + half * 32;
            unsigned q4b, h4b;
            __builtin_memcpy(&q4b, ql + qb, 4);
            __builtin_memcpy(&h4b, qh + (p0 & 31), 4);
            int hsh = (p0 >> 5) * 2;
            float sq[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int p = p0 + j;
                int qn = (q4b >> (8 * j)) & 0xFF;
                int qv = (lo ? (qn & 0xF) : (qn >> 4)) | ((((h4b >> (8 * j)) >> hsh) & 3) << 4);
                sq[j] = d * (float)sc[half * 8 + p / 16] * (float)(qv - 32);
            }
            int xo = xb + half * 128 + p0;
            for (int t = 0; t < tm; t++) {
                const float* x = xs + (long long)(t0 + t) * in_dim + xo;
                acc[t] += sq[0] * x[0] + sq[1] * x[1] + sq[2] * x[2] + sq[3] * x[3];
            }
        }
    }
    for (int t = 0; t < tm; t++) {
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down(a, o);
        if (l == 0) y[(long long)(t0 + t) * out_dim + row] = a;
    }
}

// Table-driven multi-expert Q6 GEMM: one launch per (gate|up|down) per layer.
// table[gy] = {eid, xrow0, yrow0, mrows}: blockIdx.y processes up to 16 rows of one
// expert's token group, reading the expert's native Q6 weight rows once.
extern "C" __global__ void q6_gemm_moe(const unsigned char* __restrict__ w, long long ebytes,
                                       const int4* __restrict__ table,
                                       const float* __restrict__ xs,
                                       float* __restrict__ y, int in_dim, int out_dim) {
    int l = threadIdx.x & 31;
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int4 e = table[blockIdx.y];
    int eid = e.x, xrow0 = e.y, yrow0 = e.z, tm = e.w;
    int nb = in_dim / 256;
    const unsigned char* rowp = w + (long long)eid * ebytes + (long long)row * nb * 210;
    int p0 = 4 * l;
    int lo = p0 < 64;
    int qb = lo ? p0 : p0 - 64;
    float acc[16] = {0.f};
    for (int bi = 0; bi < nb; bi++) {
        const unsigned char* blk = rowp + bi * 210;
        const signed char* sc = (const signed char*)(blk + 192);
        float d = h2f(*(const unsigned short*)(blk + 208));
        int xb = bi * 256;
        for (int half = 0; half < 2; half++) {
            const unsigned char* ql = blk + half * 64;
            const unsigned char* qh = blk + 128 + half * 32;
            unsigned q4b, h4b;
            __builtin_memcpy(&q4b, ql + qb, 4);
            __builtin_memcpy(&h4b, qh + (p0 & 31), 4);
            int hsh = (p0 >> 5) * 2;
            float sq[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int p = p0 + j;
                int qn = (q4b >> (8 * j)) & 0xFF;
                int qv = (lo ? (qn & 0xF) : (qn >> 4)) | ((((h4b >> (8 * j)) >> hsh) & 3) << 4);
                sq[j] = d * (float)sc[half * 8 + p / 16] * (float)(qv - 32);
            }
            int xo = xb + half * 128 + p0;
            for (int t = 0; t < tm; t++) {
                const float* x = xs + (long long)(xrow0 + t) * in_dim + xo;
                acc[t] += sq[0] * x[0] + sq[1] * x[1] + sq[2] * x[2] + sq[3] * x[3];
            }
        }
    }
    for (int t = 0; t < tm; t++) {
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down(a, o);
        if (l == 0) y[(long long)(yrow0 + t) * out_dim + row] = a;
    }
}

// act[k][i] = silu(g[k][i]) * u[k][i]; n = k*eff.
extern "C" __global__ void moe_silu_mul(const float* __restrict__ g, const float* __restrict__ u,
                                        float* __restrict__ act, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i];
    act[i] = (gv / (1.f + __expf(-gv))) * u[i];
}

// out[i] += sigmoid(sg[0]) * sd[i] (shared-expert combine). grid=ceil(n/256).
extern "C" __global__ void shexp_add(float* __restrict__ out, const float* __restrict__ sd,
                                     float sgate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += sd[i] * sgate;
}

// Table RoPE: pairwise rotate with precomputed cos/sin per dim (yarn/mscale baked).
extern "C" __global__ void rope_tab(float* __restrict__ v, const float* __restrict__ cs,
                                    const float* __restrict__ sn, int head_dim, int n_heads) {
    int idx = blockIdx.x * 64 + threadIdx.x, half = head_dim / 2;
    if (idx >= n_heads * half) return;
    int head = idx / half, j = idx % half;
    int base = head * head_dim;
    float c = cs[j], s = sn[j];
    float x1 = v[base + j], x2 = v[base + j + half];
    v[base + j] = x1 * c - x2 * s;
    v[base + j + half] = x1 * s + x2 * c;
}

// h[i] += x[i] (plain residual). grid=ceil(n/256), block=256.
extern "C" __global__ void vec_add(float* __restrict__ h, const float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] += x[i];
}

// Tiny F32 GEMV (router): y[row] = W[row,:].x. grid=out, block=32.
extern "C" __global__ void f32_gemv(const float* __restrict__ w, const float* __restrict__ x,
                                    float* __restrict__ y, int in_dim, int out_dim) {
    int l = threadIdx.x & 31, row = blockIdx.x;
    if (row >= out_dim) return;
    float acc = 0.f;
    for (int i = l; i < in_dim; i += 32) acc += w[(long long)row * in_dim + i] * x[i];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (l == 0) y[row] = acc;
}

// GPU router top-k: logits[ne] -> ids[k] + renormalized softmax weights[k].
// Single wave; iterative argmax (ne<=256, k<=16). Matches CPU softmax-all+topk+renorm.
extern "C" __global__ void topk_router(const float* __restrict__ logits, int ne, int k,
                                       int* __restrict__ ids, float* __restrict__ w) {
    if (threadIdx.x != 0) return;
    float mx = -3e38f;
    for (int e = 0; e < ne; e++) mx = fmaxf(mx, logits[e]);
    float sum = 0.f;
    for (int e = 0; e < ne; e++) sum += __expf(logits[e] - mx);
    float taken[16];
    int tid[16];
    for (int i = 0; i < k; i++) {
        float best = -3e38f;
        int bj = 0;
        for (int e = 0; e < ne; e++) {
            bool used = false;
            for (int j = 0; j < i; j++) used |= (tid[j] == e);
            if (!used && logits[e] > best) { best = logits[e]; bj = e; }
        }
        tid[i] = bj;
        taken[i] = __expf(best - mx) / sum;
    }
    float ws = 0.f;
    for (int i = 0; i < k; i++) ws += taken[i];
    for (int i = 0; i < k; i++) { ids[i] = tid[i]; w[i] = taken[i] / ws; }
}

// out[o] = sum_k wexp[k]*dy[k][o] — deterministic accumulation in routed order.
extern "C" __global__ void moe_wsum(const float* __restrict__ dy, const float* __restrict__ wexp,
                                    float* __restrict__ out, int n, int k) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    float s = 0.f;
    for (int e = 0; e < k; e++) s += wexp[e] * dy[(long long)e * n + o];
    out[o] = s;
}

// RMSNorm over n_rows rows of `dim`. grid=n_rows, block=256.
extern "C" __global__ void rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                                   float* __restrict__ y, int dim, int has_w, float eps) {
    int row = blockIdx.x, t = threadIdx.x;
    const float* xr = x + (size_t)row * dim;
    float* yr = y + (size_t)row * dim;
    __shared__ float red[256];
    float ss = 0.f;
    for (int i = t; i < dim; i += 256) { float v = xr[i]; ss += v * v; }
    red[t] = ss; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float scale = rsqrtf(red[0] / dim + eps);
    for (int i = t; i < dim; i += 256) { float g = has_w ? w[i] : 1.f; yr[i] = xr[i] * scale * g; }
}

// Fused residual+RMSNorm(+scale): h = (h + rmsnorm(x)*w) * scale. grid=1, block=256.
extern "C" __global__ void addnorm(float* __restrict__ h, const float* __restrict__ x,
                                   const float* __restrict__ w, int dim, float eps, float scale) {
    int t = threadIdx.x;
    __shared__ float red[256];
    float ss = 0.f;
    for (int i = t; i < dim; i += 256) { float v = x[i]; ss += v * v; }
    red[t] = ss; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float rs = rsqrtf(red[0] / dim + eps);
    for (int i = t; i < dim; i += 256) h[i] = (h[i] + x[i] * rs * w[i]) * scale;
}

// RoPE (NEOX) in place. grid=ceil(n_heads*hd/2/64), block=64.
extern "C" __global__ void rope(float* __restrict__ v, const float* __restrict__ ff,
                                int head_dim, int n_heads, int pos, float theta) {
    int idx = blockIdx.x * 64 + threadIdx.x, half = head_dim / 2;
    if (idx >= n_heads * half) return;
    int head = idx / half, j = idx % half;
    float inv = __powf(theta, -2.f * j / head_dim) / ff[j];
    float ang = pos * inv, s = sinf(ang), c = cosf(ang);
    int base = head * head_dim;
    float x1 = v[base + j], x2 = v[base + j + half];
    v[base + j] = x1 * c - x2 * s;
    v[base + j + half] = x2 * c + x1 * s;
}

// Decode-only fused q/k/v post-processing:
// - q: RMSNorm per head + RoPE -> qout
// - k: RMSNorm per KV head + RoPE -> token-major K cache slot
// - v: RMSNorm/no-weight or copy -> token-major V cache slot
// This collapses q_norm + k_norm + v_norm/copy + q_rope + k_rope + K/V copies
// from seven tiny launches into one launch per layer.
extern "C" __global__ void qkv_post(const float* __restrict__ q, const float* __restrict__ k,
                                    const float* __restrict__ v, const float* __restrict__ qw,
                                    const float* __restrict__ kw, const float* __restrict__ ff,
                                    float* __restrict__ qout, float* __restrict__ kdst,
                                    float* __restrict__ vdst, int hd, int n_heads, int n_kv,
                                    int pos, float theta, float eps, int norm_v, int kvf16) {
    int b = blockIdx.x, t = threadIdx.x, half = hd / 2;
    __shared__ float red[256];
    if (b < n_heads) {
        int h = b, base = h * hd;
        float ss = 0.f;
        for (int i = t; i < hd; i += 256) { float x = q[base + i]; ss += x * x; }
        red[t] = ss; __syncthreads();
        for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
        float rs = rsqrtf(red[0] / hd + eps);
        for (int j = t; j < half; j += 256) {
            float inv = __powf(theta, -2.f * j / hd) / ff[j];
            float ang = pos * inv, sn = sinf(ang), cs = cosf(ang);
            float x1 = q[base + j] * rs * qw[j];
            float x2 = q[base + j + half] * rs * qw[j + half];
            qout[base + j] = x1 * cs - x2 * sn;
            qout[base + j + half] = x2 * cs + x1 * sn;
        }
    } else {
        int h = b - n_heads, base = h * hd;
        if (h >= n_kv) return;
        float ss = 0.f;
        for (int i = t; i < hd; i += 256) { float x = k[base + i]; ss += x * x; }
        red[t] = ss; __syncthreads();
        for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
        float rs = rsqrtf(red[0] / hd + eps);
        for (int j = t; j < half; j += 256) {
            float inv = __powf(theta, -2.f * j / hd) / ff[j];
            float ang = pos * inv, sn = sinf(ang), cs = cosf(ang);
            float x1 = k[base + j] * rs * kw[j];
            float x2 = k[base + j + half] * rs * kw[j + half];
            float ka = x1 * cs - x2 * sn, kb2 = x2 * cs + x1 * sn;
            if (kvf16) { ((unsigned short*)kdst)[base + j] = f2h(ka); ((unsigned short*)kdst)[base + j + half] = f2h(kb2); }
            else { kdst[base + j] = ka; kdst[base + j + half] = kb2; }
        }
        if (norm_v) {
            ss = 0.f;
            for (int i = t; i < hd; i += 256) { float x = v[base + i]; ss += x * x; }
            red[t] = ss; __syncthreads();
            for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
            float vrs = rsqrtf(red[0] / hd + eps);
            for (int i = t; i < hd; i += 256) { float vv = v[base + i] * vrs; if (kvf16) ((unsigned short*)vdst)[base + i] = f2h(vv); else vdst[base + i] = vv; }
        } else {
            for (int i = t; i < hd; i += 256) { float vv = v[base + i]; if (kvf16) ((unsigned short*)vdst)[base + i] = f2h(vv); else vdst[base + i] = vv; }
        }
    }
}

// GeGLU: out = gelu_tanh(gate) * up. grid=ceil(n/64), block=64.
extern "C" __global__ void geglu(const float* __restrict__ gate, const float* __restrict__ up,
                                 float* __restrict__ out, int n) {
    int i = blockIdx.x * 64 + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float gt = 0.5f * g * (1.f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
    out[i] = gt * up[i];
}

// Logit soft-cap in place. grid=ceil(n/64), block=64.
extern "C" __global__ void softcap(float* __restrict__ x, int n, float cap) {
    int i = blockIdx.x * 64 + threadIdx.x;
    if (i >= n) return;
    x[i] = cap * tanhf(x[i] / cap);
}

// Greedy argmax over logits[n] in one block. Ties resolve to the LOWEST index
// (matches the CPU GreedySampler). softcap is monotone so argmax(raw)==argmax(capped),
// letting the greedy decode path skip both softcap and the vocab-wide DtoH.
extern "C" __global__ void argmax_f32(const float* __restrict__ x, int n,
                                      int* __restrict__ out_idx, float* __restrict__ out_val) {
    int t = threadIdx.x, nt = blockDim.x;
    float bestv = -3.0e38f; int besti = 0;
    for (int i = t; i < n; i += nt) {
        float v = x[i];
        if (v > bestv) { bestv = v; besti = i; }   // ascending scan → lowest idx on tie
    }
    __shared__ float sv[1024];
    __shared__ int si[1024];
    sv[t] = bestv; si[t] = besti;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (t < s) {
            float ov = sv[t + s]; int oi = si[t + s];
            if (ov > sv[t] || (ov == sv[t] && oi < si[t])) { sv[t] = ov; si[t] = oi; }
        }
        __syncthreads();
    }
    if (t == 0) { *out_idx = si[0]; *out_val = sv[0]; }
}

// Single-query causal SDPA with GQA. grid=n_heads, block=256, scores in shared.
extern "C" __global__ void sdpa(const float* __restrict__ q, const float* __restrict__ k,
                                const float* __restrict__ v, float* __restrict__ out,
                                int hd, int len, int groups, int n_kv, float scale, int kvf16) {
    int h = blockIdx.x, t = threadIdx.x, kvh = h / groups, qbase = h * hd;
    __shared__ float scores[2048];
    __shared__ float red[256];
    for (int i = t; i < len; i += 256) {
        float s = 0.f; int kb = (i * n_kv + kvh) * hd;
        for (int d = 0; d < hd; d++) s += q[qbase + d] * KVLD(k, kb + d, kvf16);
        scores[i] = s * scale;
    }
    __syncthreads();
    float m = -3.0e38f;
    for (int i = t; i < len; i += 256) m = fmaxf(m, scores[i]);
    red[t] = m; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
    float mx = red[0]; __syncthreads();
    float ls = 0.f;
    for (int i = t; i < len; i += 256) { float e = __expf(scores[i] - mx); scores[i] = e; ls += e; }
    red[t] = ls; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float inv = 1.f / red[0]; __syncthreads();
    for (int d = t; d < hd; d += 256) {
        float acc = 0.f;
        for (int i = 0; i < len; i++) acc += scores[i] * KVLD(v, (i * n_kv + kvh) * hd + d, kvf16);
        out[qbase + d] = acc * inv;
    }
}

// Flash-decoding SDPA, pass 1: split the `len` keys into `n_split` chunks; each
// (head, chunk) block computes a partial online-softmax (local max, sum, and
// unnormalized weighted-V). grid=(n_heads, n_split), block=128. Far better
// occupancy than one block/head (which leaves most CUs idle) and scales with
// context length.
extern "C" __global__ void sdpa_split(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, float* __restrict__ pout,
                                      float* __restrict__ pmax, float* __restrict__ psum,
                                      int hd, int len, int groups, int n_kv, float scale, int n_split, int kvf16) {
    int h = blockIdx.x, sp = blockIdx.y, t = threadIdx.x;
    int kvh = h / groups, qbase = h * hd;
    int chunk = (len + n_split - 1) / n_split, t0 = sp * chunk;
    int t1 = min(len, t0 + chunk), n = t1 - t0;
    __shared__ float sc[512];
    __shared__ float red[128];
    for (int i = t; i < n; i += 128) {
        float s = 0.f; int kb = ((t0 + i) * n_kv + kvh) * hd;
        for (int d = 0; d < hd; d++) s += q[qbase + d] * KVLD(k, kb + d, kvf16);
        sc[i] = s * scale;
    }
    __syncthreads();
    float m = -3.0e38f;
    for (int i = t; i < n; i += 128) m = fmaxf(m, sc[i]);
    red[t] = m; __syncthreads();
    for (int s = 64; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
    float M = red[0]; __syncthreads();
    float ls = 0.f;
    for (int i = t; i < n; i += 128) { float e = __expf(sc[i] - M); sc[i] = e; ls += e; }
    red[t] = ls; __syncthreads();
    for (int s = 64; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float L = red[0];
    int base = (h * n_split + sp) * hd;
    for (int d = t; d < hd; d += 128) {
        float acc = 0.f;
        for (int i = 0; i < n; i++) acc += sc[i] * KVLD(v, ((t0 + i) * n_kv + kvh) * hd + d, kvf16);
        pout[base + d] = acc;
    }
    if (t == 0) { pmax[h * n_split + sp] = (n > 0) ? M : -3.0e38f; psum[h * n_split + sp] = L; }
}

// Flash-decoding SDPA, pass 2: combine the per-chunk partials for each head via
// online-softmax rescaling. grid=n_heads, block=128.
extern "C" __global__ void sdpa_combine(const float* __restrict__ pout, const float* __restrict__ pmax,
                                        const float* __restrict__ psum, float* __restrict__ out,
                                        int hd, int n_split) {
    int h = blockIdx.x, t = threadIdx.x;
    float M = -3.0e38f;
    for (int s = 0; s < n_split; s++) M = fmaxf(M, pmax[h * n_split + s]);
    float L = 0.f;
    for (int s = 0; s < n_split; s++) L += __expf(pmax[h * n_split + s] - M) * psum[h * n_split + s];
    float inv = (L > 0.f) ? 1.f / L : 0.f;
    for (int d = t; d < hd; d += 128) {
        float acc = 0.f;
        for (int s = 0; s < n_split; s++)
            acc += __expf(pmax[h * n_split + s] - M) * pout[(h * n_split + s) * hd + d];
        out[h * hd + d] = acc * inv;
    }
}

// NPU hybrid: per-token int8 quantization of an [m,K] f32 activation into the
// NPU's A buffer [MPAD,K] (row-major int8), zero-padding rows m..MPAD, and
// emitting per-token scale xscale[m]. grid=MPAD blocks, block=256.
extern "C" __global__ void xquant_npu(const float* __restrict__ x, signed char* __restrict__ qout,
                                      float* __restrict__ xscale, int m, int K, int mpad) {
    int r = blockIdx.x, t = threadIdx.x;
    __shared__ float red[256];
    if (r >= m) {                       // pad row
        for (int k = t; k < K; k += 256) qout[(size_t)r * K + k] = 0;
        if (t == 0 && r < mpad) xscale[r] = 1.f;
        return;
    }
    const float* xr = x + (size_t)r * K;
    float mx = 0.f;
    for (int k = t; k < K; k += 256) mx = fmaxf(mx, fabsf(xr[k]));
    red[t] = mx; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
    float xs = fmaxf(red[0], 1e-12f) / 127.f;
    if (t == 0) xscale[r] = xs;
    float inv = 1.f / xs;
    for (int k = t; k < K; k += 256) {
        float v = xr[k] * inv;
        int q = (int)(v >= 0.f ? v + 0.5f : v - 0.5f);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        qout[(size_t)r * K + k] = (signed char)q;
    }
}

// NPU hybrid: rescale the int32 NPU output cin[m,N] (row-major) by the per-token
// and per-row(=per-N, the weight's output channel) scales into f32 out[m,N].
// grid=ceil(m*N/256), block=256.
// cin = NPU output [m, N] (row-major, N = NPU's column count); writes into the
// full output `out` (stride nfull) at column offset noff: out[row*nfull+noff+col].
extern "C" __global__ void rescale_npu(const int* __restrict__ cin, const float* __restrict__ xscale,
                                       const float* __restrict__ wscale, float* __restrict__ out,
                                       int m, int N, int nfull, int noff) {
    size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    size_t tot = (size_t)m * N;
    if (i >= tot) return;
    int row = (int)(i / N), col = (int)(i % N);
    out[(size_t)row * nfull + noff + col] = (float)cin[i] * xscale[row] * wscale[col];
}

// Batched RoPE for prefill: M tokens, each at position start_pos+m. v laid out
// [M][n_heads][hd]. grid=ceil(M*n_heads*hd/2 / 64), block=64.
extern "C" __global__ void rope_batch(float* __restrict__ v, const float* __restrict__ ff,
                                      int head_dim, int n_heads, int start_pos, float theta, int M) {
    int idx = blockIdx.x * 64 + threadIdx.x;
    int half = head_dim / 2, per_tok = n_heads * half, total = M * per_tok;
    if (idx >= total) return;
    int m = idx / per_tok, rem = idx % per_tok, head = rem / half, j = rem % half;
    float inv = __powf(theta, -2.f * j / head_dim) / ff[j];
    float ang = (float)(start_pos + m) * inv, s = sinf(ang), c = cosf(ang);
    int base = (m * n_heads + head) * head_dim;
    float x1 = v[base + j], x2 = v[base + j + half];
    v[base + j] = x1 * c - x2 * s;
    v[base + j + half] = x2 * c + x1 * s;
}

// Batched fused residual+RMSNorm over M rows: h[r] = (h[r]+rmsnorm(x[r])*w)*scale.
// grid=M, block=256.
extern "C" __global__ void addnorm_batch(float* __restrict__ h, const float* __restrict__ x,
                                         const float* __restrict__ w, int dim, float eps, float scale) {
    int row = blockIdx.x, t = threadIdx.x;
    float* hr = h + (size_t)row * dim;
    const float* xr = x + (size_t)row * dim;
    __shared__ float red[256];
    float ss = 0.f;
    for (int i = t; i < dim; i += 256) { float v = xr[i]; ss += v * v; }
    red[t] = ss; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float rs = rsqrtf(red[0] / dim + eps);
    for (int i = t; i < dim; i += 256) hr[i] = (hr[i] + xr[i] * rs * w[i]) * scale;
}

// Causal SDPA for prefill: M query tokens (q laid out [M][n_heads][hd]), each at
// position start_pos+m attends KV-cache positions 0..=start_pos+m. grid=M*n_heads,
// block=256. (scores capped at 2048 → prompt length must be <= 2048.)
extern "C" __global__ void sdpa_prefill(const float* __restrict__ q, const float* __restrict__ k,
                                        const float* __restrict__ v, float* __restrict__ out,
                                        int hd, int start_pos, int groups, int n_kv, float scale, int n_heads, int kvf16) {
    int blk = blockIdx.x, m = blk / n_heads, h = blk % n_heads, t = threadIdx.x;
    int kvh = h / groups, qbase = (m * n_heads + h) * hd, len = start_pos + m + 1;
    __shared__ float scores[2048];
    __shared__ float red[256];
    for (int i = t; i < len; i += 256) {
        float s = 0.f; int kb = (i * n_kv + kvh) * hd;
        for (int d = 0; d < hd; d++) s += q[qbase + d] * KVLD(k, kb + d, kvf16);
        scores[i] = s * scale;
    }
    __syncthreads();
    float mx = -3.0e38f;
    for (int i = t; i < len; i += 256) mx = fmaxf(mx, scores[i]);
    red[t] = mx; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
    float m0 = red[0]; __syncthreads();
    float ls = 0.f;
    for (int i = t; i < len; i += 256) { float e = __expf(scores[i] - m0); scores[i] = e; ls += e; }
    red[t] = ls; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float inv = 1.f / red[0]; __syncthreads();
    for (int d = t; d < hd; d += 256) {
        float acc = 0.f;
        for (int i = 0; i < len; i++) acc += scores[i] * KVLD(v, (i * n_kv + kvh) * hd + d, kvf16);
        out[qbase + d] = acc * inv;
    }
}

// Flash-style causal SDPA for prefill. One block = (head h, tile of TQ=8 query
// rows). 256 threads = 8 waves x 32 lanes; wave w handles query row (q0+w), the
// 32 lanes split the hd dimension (lane owns dims lane, lane+32, ...). Each KV
// row is loaded into shared ONCE per block and reused across all TQ queries —
// cutting K/V global traffic ~TQ x vs the per-query kernel (the dominant cost,
// especially on global layers where n_kv=1/groups=16). Online softmax, so no
// score-length cap. q laid out [M][n_heads][hd]; grid=(n_heads, ceil(M/8)).
extern "C" __global__ void sdpa_prefill_f(const float* __restrict__ q, const float* __restrict__ k,
                                          const float* __restrict__ v, float* __restrict__ out,
                                          int hd, int start_pos, int groups, int n_kv,
                                          float scale, int n_heads, int m_total, int n_swa, int kvf16) {
    const int TQ = 8;
    int h = blockIdx.x, kvh = h / groups, q0 = blockIdx.y * TQ;
    int t = threadIdx.x, w = t >> 5, lane = t & 31;
    int mq = q0 + w, acn = hd >> 5;
    extern __shared__ float sh[];           // [2*hd]: ks then vs
    float* ks = sh; float* vs = sh + hd;

    int qbase = (mq * n_heads + h) * hd;
    bool active = mq < m_total;
    float qreg[16], acc[16];
    for (int c = 0; c < acn; c++) { qreg[c] = active ? q[qbase + lane + c*32] : 0.f; acc[c] = 0.f; }
    float run_m = -3.0e38f, run_l = 0.f;

    int my_len = start_pos + mq + 1;                 // causal window of this query
    // local (sliding-window) layers: query at pos p attends keys [p-n_swa+1, p]
    // (key masked iff p - key >= n_swa). swa_lo = this query's earliest key.
    int swa_lo = (n_swa > 0) ? (my_len - n_swa) : 0;       // key < swa_lo is masked
    int max_len = start_pos + q0 + TQ;               // bound across the tile
    int cap = start_pos + m_total;
    if (max_len > cap) max_len = cap;
    // start the shared-load loop at the earliest window across the whole tile
    int i_lo = (n_swa > 0) ? (start_pos + q0 + 1 - n_swa) : 0;
    if (i_lo < 0) i_lo = 0;

    for (int i = i_lo; i < max_len; i++) {
        __syncthreads();
        int kb = (i * n_kv + kvh) * hd;
        for (int d = t; d < hd; d += 256) { ks[d] = KVLD(k, kb + d, kvf16); vs[d] = KVLD(v, kb + d, kvf16); }
        __syncthreads();
        if (!active || i >= my_len || i < swa_lo) continue;  // whole wave skips together
        float part = 0.f;
        for (int c = 0; c < acn; c++) part += qreg[c] * ks[lane + c*32];
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor(part, o);
        float s = part * scale;
        float nm = fmaxf(run_m, s);
        float corr = __expf(run_m - nm), p = __expf(s - nm);
        run_l = run_l * corr + p;
        for (int c = 0; c < acn; c++) acc[c] = acc[c] * corr + p * vs[lane + c*32];
        run_m = nm;
    }
    if (active) {
        float inv = 1.f / run_l;
        for (int c = 0; c < acn; c++) out[qbase + lane + c*32] = acc[c] * inv;
    }
}

// ---- WMMA (matrix-core) flash attention for prefill (gated STRIX_WMMA_SDPA) ----
// Replaces the scalar sdpa_prefill_f's per-(query,key) dot products with f16 WMMA
// for QK^T and P·V — the prefill SDPA is the whole gap vs llama.cpp (which uses
// matrix cores). One block = one wave (32 lanes) handling one (head, 16-query tile);
// grid=(n_heads, ceil(m/16)). Softmax is done in shared S (avoids fragment-layout
// reductions). RDNA3 wave32 16x16x16 f16 WMMA: A/B frag = v16h (lane l → matrix
// row (l&15)'s 16 K-values), C = v8f (lane l, c[k] → element [row 2k+(l>>4)][col l&15]).
typedef __fp16 v16h __attribute__((ext_vector_type(16)));
typedef float   v8f __attribute__((ext_vector_type(8)));

extern "C" __global__ void sdpa_prefill_wmma(const float* __restrict__ q, const float* __restrict__ k,
                                             const float* __restrict__ v, float* __restrict__ out,
                                             int hd, int start_pos, int groups, int n_kv,
                                             float scale, int n_heads, int m_total, int n_swa, int kvf16) {
    // NW waves/block share ONE K/V tile (loaded once, reused) — each wave owns its
    // own 16-query sub-tile, fully independent (no cross-wave combine). NW = blockDim/32.
    const int BK = 16, MAXDC = 16;             // hd<=256 → dchunks<=16
    int NW = blockDim.x >> 5;
    int h = blockIdx.x, kvh = h / groups;
    int t = threadIdx.x, w = t >> 5, lane = t & 31;
    int dchunks = hd >> 4, row = lane & 15, half = lane >> 4;
    int q0 = blockIdx.y * (16 * NW) + w * 16;  // this wave's first query

    extern __shared__ char shm[];
    __fp16* Ks = (__fp16*)shm;                 // [BK*hd] shared by all waves
    __fp16* Vs = Ks + BK * hd;                 // [BK*hd]
    __fp16* Qs = Vs + BK * hd;                 // [NW*16*hd]  (per-wave query tile)
    float*  Ssb = (float*)(Qs + NW * 16 * hd); // [NW*16*BK]
    __fp16* Psb = (__fp16*)(Ssb + NW * 16 * BK); // [NW*16*BK]
    float*  MLb = (float*)(Psb + NW * 16 * BK);  // [NW*32]: per-wave m[16],l[16]
    float* Ss = Ssb + w * 16 * BK;
    __fp16* Ps = Psb + w * 16 * BK;
    float* Mrow = MLb + w * 32, *Lrow = Mrow + 16;

    // O accumulator in registers: Oacc[dc] holds rows {2k+half}, col {dc*16+row}
    v8f Oacc[MAXDC];
    for (int dc = 0; dc < dchunks; dc++) Oacc[dc] = (v8f){0,0,0,0,0,0,0,0};
    // load this wave's Q tile (f32->f16) into shared Qs[w]
    __fp16* Qw = Qs + w * 16 * hd;
    for (int e = lane; e < 16 * hd; e += 32) {
        int r = e / hd, d = e % hd, mq = q0 + r;
        Qw[e] = (mq < m_total) ? (__fp16)q[((size_t)mq * n_heads + h) * hd + d] : (__fp16)0.f;
    }
    if (lane < 16) { Mrow[lane] = -3.0e38f; Lrow[lane] = 0.f; }

    int blk_q0 = blockIdx.y * (16 * NW);        // block's first query
    int last_mq = blk_q0 + 16 * NW - 1;
    int max_len = start_pos + last_mq + 1, cap = start_pos + m_total;
    if (max_len > cap) max_len = cap;
    int i_lo = (n_swa > 0) ? (start_pos + blk_q0 + 1 - n_swa) : 0;
    if (i_lo < 0) i_lo = 0;
    int kt0 = (i_lo / BK) * BK;

    for (int i0 = kt0; i0 < max_len; i0 += BK) {
        __syncthreads();
        for (int e = t; e < BK * hd; e += blockDim.x) {   // cooperative K/V load (all waves)
            int kk = i0 + e / hd, d = e % hd;
            float kv_k = 0.f, kv_v = 0.f;
            if (kk < max_len) { size_t kb = ((size_t)kk * n_kv + kvh) * hd + d; kv_k = KVLD(k, kb, kvf16); kv_v = KVLD(v, kb, kvf16); }
            Ks[e] = (__fp16)kv_k; Vs[e] = (__fp16)kv_v;
        }
        __syncthreads();
        v8f sacc = {0,0,0,0,0,0,0,0};
        for (int dc = 0; dc < dchunks; dc++) {
            v16h a, b;
            for (int j = 0; j < 16; j++) { a[j] = Qw[row * hd + dc*16 + j]; b[j] = Ks[row * hd + dc*16 + j]; }
            sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, sacc);
        }
        for (int kk = 0; kk < 8; kk++) Ss[(2*kk + half) * BK + row] = sacc[kk];
        // softmax (lane r → query row r), produce corr[r] for O rescale
        __syncthreads();   // wave-local: Ss written by this wave
        float corr_r = 1.f;
        if (lane < 16) {
            int r = lane, mq = q0 + r;
            int my_len = start_pos + mq + 1, swa_lo = (n_swa > 0) ? (my_len - n_swa) : 0;
            float tmax = -3.0e38f;
            for (int j = 0; j < BK; j++) {
                int key = i0 + j;
                float s = (mq < m_total && key < my_len && key >= swa_lo) ? Ss[r*BK + j] * scale : -3.0e38f;
                Ss[r*BK + j] = s; tmax = fmaxf(tmax, s);
            }
            float nm = fmaxf(Mrow[r], tmax);
            corr_r = __expf(Mrow[r] - nm);
            float tsum = 0.f;
            for (int j = 0; j < BK; j++) { float p = (Ss[r*BK+j] > -1e30f) ? __expf(Ss[r*BK+j] - nm) : 0.f; Ps[r*BK+j] = (__fp16)p; tsum += p; }
            Lrow[r] = Lrow[r] * corr_r + tsum; Mrow[r] = nm;
            MLb[w*32 + 16 + r] = Lrow[r];   // (Lrow already in MLb)
        }
        __syncthreads();
        // rescale O accumulator by corr (per row 2k+half) — broadcast corr via shuffle
        // corr_r lives on lane==r (r<16); fetch corr for rows 2k+half from those lanes
        for (int kk = 0; kk < 8; kk++) {
            int rr = 2*kk + half;
            float c = __shfl(corr_r, rr);   // lane rr holds corr for row rr
            for (int dc = 0; dc < dchunks; dc++) Oacc[dc][kk] *= c;
        }
        // P·V accumulate
        for (int dc = 0; dc < dchunks; dc++) {
            v16h pa, vb;
            for (int j = 0; j < 16; j++) { pa[j] = Ps[row*BK + j]; vb[j] = Vs[j*hd + dc*16 + row]; }
            v8f pv = {0,0,0,0,0,0,0,0};
            pv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, pv);
            for (int kk = 0; kk < 8; kk++) Oacc[dc][kk] += pv[kk];
        }
    }
    // write O / l : element row 2k+half, col dc*16+row
    for (int kk = 0; kk < 8; kk++) {
        int r = 2*kk + half, mq = q0 + r;
        if (mq < m_total) {
            float inv = (Lrow[r] > 0.f) ? 1.f / Lrow[r] : 0.f;
            size_t ob = ((size_t)mq * n_heads + h) * hd;
            for (int dc = 0; dc < dchunks; dc++) out[ob + dc*16 + row] = Oacc[dc][kk] * inv;
        }
    }
}

// ===== GEMM-based prefill attention (gated STRIX_GEMM_SDPA) =====
// llama's fast prefill attention is QK^T-GEMM -> softmax -> P*V-GEMM (NOT fused
// flash). These 3 kernels read K/V ONCE (vs the fused kernel's per-query-tile
// re-reads) and use f16 WMMA at high occupancy. Handles hd=256 AND hd=512 (the
// O accumulator is a per-tile WMMA frag, not a per-thread [hd] array). One wave
// (32 lanes) per output 16x16 tile. scores buffer: f32 [n_heads * m * len].

// S[h][i][j] = scale * sum_d Q[i,d]*K[j,d]. grid=(n_heads, ceil(m/16), ceil(len/16)), block=32.
extern "C" __global__ void sdpa_qk_wmma(const float* __restrict__ q, const float* __restrict__ k,
                                        __fp16* __restrict__ scores, int hd, int groups, int n_kv,
                                        float scale, int n_heads, int m, int len, int kvf16,
                                        int start_pos) {
    int NW = blockDim.x >> 5, w = threadIdx.x >> 5;
    int h = blockIdx.x, q0 = blockIdx.y * 16, k0 = (blockIdx.z * NW + w) * 16;
    // each wave handles its own key-tile (no shared K — staging it tanks occupancy
    // more than the saved K-reads help, like every prior shared-tiling attempt).
    if (k0 >= len || k0 > start_pos + q0 + 15) return;
    int lane = threadIdx.x & 31, row = lane & 15, half = lane >> 4;
    int kvh = h / groups, dchunks = hd >> 4;
    int mq = q0 + row, ki = k0 + row;          // this lane's A query-row / B key-row
    v8f sacc = {0,0,0,0,0,0,0,0};
    const float* qp = q + ((size_t)mq * n_heads + h) * hd;
    for (int dc = 0; dc < dchunks; dc++) {
        v16h a, b;
        for (int j = 0; j < 16; j++) a[j] = (mq < m) ? (__fp16)qp[dc*16 + j] : (__fp16)0.f;
        for (int j = 0; j < 16; j++) {
            size_t kb = ((size_t)ki * n_kv + kvh) * hd + dc*16 + j;
            b[j] = (ki < len) ? (__fp16)KVLD(k, kb, kvf16) : (__fp16)0.f;
        }
        sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, sacc);
    }
    for (int kk = 0; kk < 8; kk++) {            // element [row 2kk+half][col lane&15]
        int qi = q0 + 2*kk + half, kj = k0 + row;
        if (qi < m && kj < len) scores[((size_t)h * m + qi) * len + kj] = (__fp16)(sacc[kk] * scale);
    }
}

// Row softmax over S[h][i][0:len] with causal + sliding-window masking, in place.
// grid=(n_heads, m), block=256. start_pos = KV position of query 0; n_swa>0 = local layer.
extern "C" __global__ void sdpa_softmax_mask(__fp16* __restrict__ scores, int start_pos, int m,
                                             int n_heads, int len, int n_swa, float* __restrict__ rowsum) {
    int h = blockIdx.x, i = blockIdx.y, t = threadIdx.x;
    if (i >= m) return;
    __fp16* S = scores + ((size_t)h * m + i) * len;
    int my_len = start_pos + i + 1;                 // causal: keys [0, my_len)
    int lo = (n_swa > 0) ? (my_len - n_swa) : 0;     // sliding window: keys < lo masked
    if (lo < 0) lo = 0;
    // f16 S: reductions in f32, recompute the mask each pass. Write UNNORMALIZED exp
    // (P, in [0, ...]) + the per-row sum; PV divides O by the sum (saves the 3rd pass).
    __shared__ float red[256];
    float mx = -3.0e38f;
    for (int j = t; j < len; j += 256) if (j < my_len && j >= lo) mx = fmaxf(mx, (float)S[j]);
    red[t] = mx; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t+s]); __syncthreads(); }
    float m0 = red[0]; __syncthreads();
    float sum = 0.f;
    for (int j = t; j < len; j += 256) {
        float e = (j < my_len && j >= lo) ? __expf((float)S[j] - m0) : 0.f;
        S[j] = (__fp16)e; sum += e;
    }
    red[t] = sum; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (t < s) red[t] += red[t+s]; __syncthreads(); }
    if (t == 0) rowsum[(size_t)h * m + i] = red[0];
}

// O[i][d] = sum_j P[i][j]*V[j][d]. grid=(n_heads, ceil(m/16), ceil(hd/16)), block=32.
extern "C" __global__ void sdpa_pv_wmma(const __fp16* __restrict__ scores, const float* __restrict__ v,
                                        float* __restrict__ out, int hd, int groups, int n_kv,
                                        int n_heads, int m, int len, int kvf16, int start_pos,
                                        const float* __restrict__ rowsum) {
    int NW = blockDim.x >> 5, w = threadIdx.x >> 5, t = threadIdx.x;
    int h = blockIdx.x, q0 = blockIdx.y * 16, d0 = (blockIdx.z * NW + w) * 16;
    int lane = threadIdx.x & 31, row = lane & 15, half = lane >> 4;
    int kvh = h / groups;
    int dcol = d0 + row;                          // B output-col(d) for this wave
    const __fp16* Pbase = scores + ((size_t)h * m + q0) * len;
    v8f oacc = {0,0,0,0,0,0,0,0};
    // causal: keys beyond the last query of this tile (start_pos+q0+15) are all P=0.
    int pv_len = start_pos + q0 + 16; if (pv_len > len) pv_len = len;
    // P[16 queries x 16 keys] tile is identical for all d-tile waves -> stage in shared.
    __shared__ __fp16 sP[16 * 16];
    for (int j0 = 0; j0 < pv_len; j0 += 16) {
        __syncthreads();
        for (int e = t; e < 256; e += blockDim.x) {     // cooperative P-tile load (all waves)
            int qq = e >> 4, kk2 = e & 15;
            sP[e] = (q0+qq < m && j0+kk2 < len) ? Pbase[(size_t)qq * len + j0 + kk2] : (__fp16)0.f;
        }
        __syncthreads();
        if (d0 >= hd) continue;                      // wave has no valid d-tile (still syncs)
        v16h a, b;
        for (int j = 0; j < 16; j++) a[j] = sP[row * 16 + j];
        for (int j = 0; j < 16; j++) {
            size_t vb = ((size_t)(j0+j) * n_kv + kvh) * hd + dcol;   // V[key j0+j][d=dcol]
            b[j] = (j0+j < len) ? (__fp16)KVLD(v, vb, kvf16) : (__fp16)0.f;
        }
        oacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, oacc);
    }
    if (d0 >= hd) return;
    for (int kk = 0; kk < 8; kk++) {            // element [row 2kk+half = query][col lane&15 = d]
        int qi = q0 + 2*kk + half, dd = d0 + row;
        if (qi < m && dd < hd) {
            float sum = rowsum[(size_t)h * m + qi];      // normalize here (softmax skipped it)
            float inv = (sum > 0.f) ? 1.f / sum : 0.f;
            out[((size_t)qi * n_heads + h) * hd + dd] = oacc[kk] * inv;
        }
    }
}

// Plain float copy (KV-cache append; dst pointer pre-offset to the slot).
extern "C" __global__ void copyf(float* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// f32 -> f16 copy (KV-cache append in half precision; STRIX_F16_KV path).
extern "C" __global__ void copyf_h(unsigned short* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = f2h(src[i]);
}
"#;
