//! Resident **quantized** matrix-vector product on the Radeon 890M (`wgpu`).
//!
//! This is the decode-path engine. At generation time the cost is dominated by
//! GEMV (`y = W · x`) against the model's weight matrices, which are stored
//! **Q4_0** in the GGUF and far too large to dequantize to f32 (a 12B model is
//! ~7 GB Q4_0 vs ~48 GB f32). So we:
//!
//! 1. **repack** each Q4_0 weight once into a GPU-friendly, `u32`-aligned layout
//!    (`scales: f32[]` + `quants: u32[]`) and upload it **resident** on the GPU,
//! 2. each decode step upload only the small activation `x`, then run a **fused
//!    dequant→GEMV** WGSL kernel that reads the quantized bytes directly and
//!    never materializes f32 weights.
//!
//! Q4_0 block (32 values): `f16 d` + 16 nibble bytes; `y = d * (nibble - 8)`.
//! Byte `j` holds value `j` (low nibble) and value `j+16` (high nibble). The
//! repack splits this into one `f32` scale per block + four `u32` words per block
//! (the 16 nibble bytes), which the shader unpacks. Repacking is amortized: it
//! happens once when the weight becomes resident.

use strix_core::error::{Result, StrixError};
use wgpu::util::DeviceExt;

/// Values per Q4_0 block.
const QK4_0: usize = 32;
/// On-disk bytes per Q4_0 block: `f16` scale + 16 nibble bytes.
const Q4_0_BYTES: usize = 18;

/// Max workgroups per grid dimension (Vulkan/WebGPU guarantee). We tile the
/// `out_dim` rows into a 2D grid so very tall matrices (e.g. a 262k lm_head)
/// don't exceed it.
const MAX_GRID_DIM: usize = 65535;

/// X-extent of the 2D workgroup grid for `out_dim` rows.
fn grid_x(out_dim: usize) -> u32 {
    out_dim.min(MAX_GRID_DIM) as u32
}

/// Values per Q6_K superblock.
const QK_K: usize = 256;
/// On-disk bytes per Q6_K superblock: 128 `ql` + 64 `qh` + 16 i8 scales + f16 d.
const Q6_K_BYTES: usize = 210;

/// A live GPU context plus the resident-quantized GEMV pipelines (Q4_0, Q6_K)
/// and the Stage-C element-wise op pipelines (RMSNorm, …) used to keep the whole
/// decode forward on-device.
pub struct GpuQ4 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pipeline_q6: wgpu::ComputePipeline,
    pipeline_rmsnorm: wgpu::ComputePipeline,
    pipeline_rope: wgpu::ComputePipeline,
    pipeline_geglu: wgpu::ComputePipeline,
    pipeline_softcap: wgpu::ComputePipeline,
    pipeline_add: wgpu::ComputePipeline,
    pipeline_scale: wgpu::ComputePipeline,
    pipeline_sdpa: wgpu::ComputePipeline,
    pipeline_addnorm: wgpu::ComputePipeline,
    /// Activation → int8 (q8) quantizer feeding the dp4a GEMV.
    pipeline_xquant: wgpu::ComputePipeline,
    /// Hardware-dp4a Q4_0 GEMV (GLSL→SPIR-V, loaded via passthrough). `None` when
    /// the adapter lacks SPIR-V passthrough or subgroups.
    pipeline_dp4a: Option<wgpu::ComputePipeline>,
    dp4a_bgl: Option<wgpu::BindGroupLayout>,
    /// Output rows per Q4_0 workgroup: 2 with the subgroup kernel (`SHADER_SG`),
    /// 1 with the shared-memory fallback (`SHADER`). Determines the dispatch grid.
    rows_per_wg: usize,
    adapter_name: String,
}

/// Activation quantizer: x[in_dim] → int8 `xq` (8 i32/32-block) + per-block f32
/// scale `xd`, matching the dp4a GEMV's expected layout. One thread per block.
const SHADER_XQUANT: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xd: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>; // (nblocks, _, _, _)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blk = gid.x;
    if (blk >= p.x) { return; }
    let base = blk * 32u;
    var amax = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) { amax = max(amax, abs(x[base + i])); }
    let d = amax / 127.0;
    xd[blk] = d;
    let inv = select(0.0, 1.0 / d, d > 0.0);
    for (var w = 0u; w < 8u; w = w + 1u) {
        var word = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let qi = clamp(i32(round(x[base + w * 4u + j] * inv)), -127, 127);
            word = word | ((u32(qi) & 0xffu) << (j * 8u));
        }
        xq[blk * 8u + w] = word;
    }
}
"#;

/// Fused residual-add + RMSNorm (+ optional scalar): `h = (h + rmsnorm(x)*w) * s`.
/// One workgroup over the hidden vector. Collapses the post-attn / post-ffw
/// `norm → add (→ scale)` chain into a single pass. `p=(dim, eps, scale)` bits.
pub(crate) const SHADER_ADDNORM: &str = r#"
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>; // (dim, eps_bits, scale_bits, _)

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let dim = p.x;
    let eps = bitcast<f32>(p.y);
    let scale = bitcast<f32>(p.z);
    let t = lid.x;
    var ss = 0.0;
    for (var i = t; i < dim; i = i + 256u) { let v = x[i]; ss = ss + v * v; }
    red[t] = ss;
    workgroupBarrier();
    var st = 128u;
    loop { if (st == 0u) { break; } if (t < st) { red[t] = red[t] + red[t + st]; } workgroupBarrier(); st = st / 2u; }
    let rscale = inverseSqrt(red[0] / f32(dim) + eps);
    for (var i = t; i < dim; i = i + 256u) {
        h[i] = (h[i] + x[i] * rscale * w[i]) * scale;
    }
}
"#;

/// RMSNorm over independent rows. Input is `n_rows × dim`; each row is normalized
/// `y = x * rsqrt(mean(x^2)+eps) * (w or 1)`. One workgroup per row.
/// `p = (dim, n_rows, has_weight, bitcast(eps))`. Covers the full-vector attn/ffn
/// norms (1 row of hidden), per-head QK-norm (n_heads rows of head_dim, shared
/// weight), and per-head V-norm (n_kv rows, no weight).
pub(crate) const SHADER_RMSNORM: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>; // (dim, n_rows, has_weight, eps_bits)

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let dim = p.x;
    let row = wid.x;
    let eps = bitcast<f32>(p.w);
    let base = row * dim;
    let t = lid.x;

    var ss = 0.0;
    for (var i = t; i < dim; i = i + 256u) {
        let v = x[base + i];
        ss = ss + v * v;
    }
    red[t] = ss;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { red[t] = red[t] + red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let scale = inverseSqrt(red[0] / f32(dim) + eps);
    for (var i = t; i < dim; i = i + 256u) {
        var g = 1.0;
        if (p.z != 0u) { g = w[i]; }
        y[base + i] = x[base + i] * scale * g;
    }
}
"#;

/// RoPE (NEOX/rotate_half) per head, matching `ops::rope_in_place_ff`. One thread
/// per (head, pair) over `n_heads × head_dim/2`. `ff` holds per-pair freq factors
/// (length head_dim/2; all-1.0 when the layer has none → divide is a no-op).
/// `p = (head_dim, n_heads, pos, bitcast(theta))`.
pub(crate) const SHADER_ROPE: &str = r#"
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<storage, read> ff: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head_dim = p.x;
    let n_heads = p.y;
    let pos = f32(p.z);
    let theta = bitcast<f32>(p.w);
    let half = head_dim / 2u;
    let idx = gid.x;
    if (idx >= n_heads * half) { return; }
    let head = idx / half;
    let j = idx % half;
    let inv_freq = pow(theta, -2.0 * f32(j) / f32(head_dim)) / ff[j];
    let angle = pos * inv_freq;
    let s = sin(angle);
    let c = cos(angle);
    let base = head * head_dim;
    let x1 = v[base + j];
    let x2 = v[base + j + half];
    v[base + j] = x1 * c - x2 * s;
    v[base + j + half] = x2 * c + x1 * s;
}
"#;

/// GeGLU: `out = gelu_tanh(gate) * up` (gelu_pytorch_tanh). `p.x = len`.
pub(crate) const SHADER_GEGLU: &str = r#"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.x) { return; }
    let g = gate[i];
    let gt = 0.5 * g * (1.0 + tanh(0.7978845608 * (g + 0.044715 * g * g * g)));
    out[i] = gt * up[i];
}
"#;

/// Logit soft-cap in place: `x = cap * tanh(x / cap)`. `p.x = len`, `p.y = bitcast(cap)`.
pub(crate) const SHADER_SOFTCAP: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.x) { return; }
    let cap = bitcast<f32>(p.y);
    x[i] = cap * tanh(x[i] / cap);
}
"#;

/// Residual add in place: `a += b`. `p.x = len`.
const SHADER_ADD: &str = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.x) { return; }
    a[i] = a[i] + b[i];
}
"#;

/// Scalar multiply in place: `a *= s`. `p.x = len`, `p.y = bitcast(s)`.
const SHADER_SCALE: &str = r#"
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<uniform> p: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.x) { return; }
    a[i] = a[i] * bitcast<f32>(p.y);
}
"#;

/// Single-query causal SDPA with GQA. One workgroup per query head; scores live
/// in **workgroup shared memory** (no global scratch round-trip — the big win
/// for decode). Token-major cache: `K`/`V[t][kvh][d]` at `(t*n_kv+kvh)*hd+d`.
/// `p[0]=(head_dim, len, groups, n_kv)`, `p[1].x=bitcast(scale)`.
/// `len` must be ≤ SDPA_SCORES_CAP (short-context decode; longer needs tiling).
pub(crate) const SHADER_SDPA: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> scores: array<f32, 2048>; // per-head scores in shared memory (8KB → good occupancy)
var<workgroup> red: array<f32, 64>;
var<workgroup> sh: array<f32, 2>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let hd = p[0].x;
    let len = p[0].y;
    let groups = p[0].z;
    let n_kv = p[0].w;
    let scale = bitcast<f32>(p[1].x);
    let h = wid.x;
    let kvh = h / groups;
    let t0 = lid.x;
    let qbase = h * hd;

    for (var t = t0; t < len; t = t + 64u) {
        var s = 0.0;
        let kb = (t * n_kv + kvh) * hd;
        for (var d = 0u; d < hd; d = d + 1u) { s = s + q[qbase + d] * k[kb + d]; }
        scores[t] = s * scale;
    }
    workgroupBarrier();

    var m = -3.0e38;
    for (var t = t0; t < len; t = t + 64u) { m = max(m, scores[t]); }
    red[t0] = m;
    workgroupBarrier();
    var st = 32u;
    loop { if (st == 0u) { break; } if (t0 < st) { red[t0] = max(red[t0], red[t0 + st]); } workgroupBarrier(); st = st / 2u; }
    if (t0 == 0u) { sh[0] = red[0]; }
    workgroupBarrier();
    let mx = sh[0];
    workgroupBarrier();

    var ls = 0.0;
    for (var t = t0; t < len; t = t + 64u) {
        let e = exp(scores[t] - mx);
        scores[t] = e;
        ls = ls + e;
    }
    red[t0] = ls;
    workgroupBarrier();
    st = 32u;
    loop { if (st == 0u) { break; } if (t0 < st) { red[t0] = red[t0] + red[t0 + st]; } workgroupBarrier(); st = st / 2u; }
    if (t0 == 0u) { sh[1] = red[0]; }
    workgroupBarrier();
    let inv = 1.0 / sh[1];

    for (var d = t0; d < hd; d = d + 64u) {
        var acc = 0.0;
        for (var t = 0u; t < len; t = t + 1u) { acc = acc + scores[t] * v[(t * n_kv + kvh) * hd + d]; }
        out[qbase + d] = acc * inv;
    }
}
"#;

/// Decode SDPA at workgroup_size 256 (4 RDNA wavefronts per head). Same math as
/// `SHADER_SDPA` but 4× the threads on the score/output loops and 4 wavefronts/CU
/// → far better latency hiding for the tiny per-head work (the ash path's single
/// biggest element cost was the 1-wavefront 64-thread version).
pub(crate) const SHADER_SDPA256: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> scores: array<f32, 2048>;
var<workgroup> red: array<f32, 256>;
var<workgroup> sh: array<f32, 2>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let hd = p[0].x;
    let len = p[0].y;
    let groups = p[0].z;
    let n_kv = p[0].w;
    let scale = bitcast<f32>(p[1].x);
    let h = wid.x;
    let kvh = h / groups;
    let t0 = lid.x;
    let qbase = h * hd;

    for (var t = t0; t < len; t = t + 256u) {
        var s = 0.0;
        let kb = (t * n_kv + kvh) * hd;
        for (var d = 0u; d < hd; d = d + 1u) { s = s + q[qbase + d] * k[kb + d]; }
        scores[t] = s * scale;
    }
    workgroupBarrier();

    var m = -3.0e38;
    for (var t = t0; t < len; t = t + 256u) { m = max(m, scores[t]); }
    red[t0] = m;
    workgroupBarrier();
    var st = 128u;
    loop { if (st == 0u) { break; } if (t0 < st) { red[t0] = max(red[t0], red[t0 + st]); } workgroupBarrier(); st = st / 2u; }
    if (t0 == 0u) { sh[0] = red[0]; }
    workgroupBarrier();
    let mx = sh[0];
    workgroupBarrier();

    var ls = 0.0;
    for (var t = t0; t < len; t = t + 256u) {
        let e = exp(scores[t] - mx);
        scores[t] = e;
        ls = ls + e;
    }
    red[t0] = ls;
    workgroupBarrier();
    st = 128u;
    loop { if (st == 0u) { break; } if (t0 < st) { red[t0] = red[t0] + red[t0 + st]; } workgroupBarrier(); st = st / 2u; }
    if (t0 == 0u) { sh[1] = red[0]; }
    workgroupBarrier();
    let inv = 1.0 / sh[1];

    for (var d = t0; d < hd; d = d + 256u) {
        var acc = 0.0;
        for (var t = 0u; t < len; t = t + 1u) { acc = acc + scores[t] * v[(t * n_kv + kvh) * hd + d]; }
        out[qbase + d] = acc * inv;
    }
}
"#;

/// A single Q6_K weight matrix `[out_dim, in_dim]`, uploaded once and reused.
///
/// `scales` holds 16 `f32` per superblock (each `d * sub_scale`, folded), `ql`
/// holds 32 `u32` (128 bytes) and `qh` 16 `u32` (64 bytes) per superblock.
pub struct ResidentQ6 {
    #[allow(dead_code)]
    scales: wgpu::Buffer,
    #[allow(dead_code)]
    ql: wgpu::Buffer,
    #[allow(dead_code)]
    qh: wgpu::Buffer,
    #[allow(dead_code)]
    dims: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    y_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    in_dim: usize,
    out_dim: usize,
}

impl ResidentQ6 {
    /// Output dimension (number of rows).
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
    /// Input dimension (number of columns).
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }
}

/// A single Q4_0 weight matrix `[out_dim, in_dim]`, uploaded once and reused.
///
/// `scales` holds one `f32` per 32-value block; `quants` holds four `u32` per
/// block (the 16 nibble bytes). The weight, its I/O buffers (`x`/`y`/staging),
/// and the bind group are all created once at upload and reused every decode
/// step — a `gemv` call only writes `x` and reads back `y`, with zero per-call
/// allocation (allocation churn was a meaningful slice of the per-call overhead).
pub struct ResidentQ4 {
    #[allow(dead_code)]
    scales: wgpu::Buffer,
    #[allow(dead_code)]
    quants: wgpu::Buffer,
    #[allow(dead_code)]
    dims: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    y_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    in_dim: usize,
    out_dim: usize,
}

impl ResidentQ4 {
    /// Output dimension (number of rows).
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
    /// Input dimension (number of columns).
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }
}

// One workgroup of 64 threads cooperatively computes one output row. Thread `t`
// strides over the row's Q4_0 blocks (t, t+64, …) — so on each step the 64
// threads read 64 *consecutive* blocks, giving coalesced loads from `scales`
// and `quants` (the key to bandwidth). Partial sums reduce in shared memory.
// `dims = (in_dim, out_dim, grid_x, _)`; the row is `wid.x + wid.y*grid_x`
// (2D grid works around the 65535 workgroups-per-dimension limit).
//
// (Tried 2 rows/workgroup à la llama.cpp `rm_stdq`: it *regressed* here. This
// GEMV is weight-bandwidth/occupancy-bound, not activation-bound — reusing `x`
// saves nothing against the huge weight stream, and halving the workgroup count
// hurt occupancy. The real lever is per-call submit/readback overhead, not the
// kernel inner loop. Kept single-row.)
const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> scales: array<u32>; // 2 f16 scales / u32
@group(0) @binding(1) var<storage, read> quants: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>; // (in_dim, out_dim, grid_x, _)

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let in_dim = dims.x;
    let out_dim = dims.y;
    let row = wid.x + wid.y * dims.z;
    let t = lid.x;
    let nblocks = in_dim / 32u;          // Q4_0 block = 32 values
    let row_blk = row * nblocks;

    var acc = 0.0;
    if (row < out_dim) {
        for (var b = t; b < nblocks; b = b + 64u) {
            let blk = row_blk + b;
            let d = unpack2x16float(scales[blk >> 1u])[blk & 1u]; // f16 scale
            let qbase = blk * 4u;
            let xbase = b * 32u;         // 32 activations per block
            for (var w = 0u; w < 4u; w = w + 1u) {
                let word = quants[qbase + w];
                for (var k = 0u; k < 4u; k = k + 1u) {
                    let j = w * 4u + k;              // byte index 0..15
                    let byte = (word >> (k * 8u)) & 0xffu;
                    let lo = f32(byte & 0x0fu) - 8.0;
                    let hi = f32(byte >> 4u) - 8.0;
                    acc = acc + d * (lo * x[xbase + j] + hi * x[xbase + j + 16u]);
                }
            }
        }
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u && row < out_dim) { y[row] = partial[0]; }
}
"#;

/// Q4_0 GEMV, subgroup-reduction variant. Same compute as `SHADER`, but the
/// per-row sum reduces with one `subgroupAdd` (+ a tiny cross-subgroup combine)
/// instead of a 6-barrier shared-memory tree — a win for small-K rows where the
/// reduction otherwise rivals the compute. Requires the `SUBGROUP` feature.
pub(crate) const SHADER_SG: &str = r#"
@group(0) @binding(0) var<storage, read> scales: array<u32>; // 2 f16 scales / u32
@group(0) @binding(1) var<storage, read> quants: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>;

// workgroup_size 32 = one subgroup on RDNA: each thread covers 2× the K of a
// 64-wide group (more in-flight FMAs to hide memory latency — llama.cpp's
// `BLOCK_SIZE = subgroup_size` choice) and the per-row reduce is a single
// `subgroupAdd`, no shared-memory combine. (2 rows/workgroup regressed even
// here — fewer workgroups hurt occupancy more than x-reuse helps — so 1 row.)
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let in_dim = dims.x;
    let out_dim = dims.y;
    let row = wid.x + wid.y * dims.z;
    let t = lid.x;
    let nblocks = in_dim / 32u;
    let row_blk = row * nblocks;

    var acc = 0.0;
    if (row < out_dim) {
        for (var b = t; b < nblocks; b = b + 32u) {
            let blk = row_blk + b;
            let d = unpack2x16float(scales[blk >> 1u])[blk & 1u]; // f16 scale
            let qbase = blk * 4u;
            let xbase = b * 32u;
            // Vectorized unpack: each u32 = 4 bytes = 4 low + 4 high nibbles;
            // dequant 4-wide and `dot` with the activations (fewer ALU ops).
            for (var w = 0u; w < 4u; w = w + 1u) {
                let by = unpack4xU8(quants[qbase + w]);
                let lo = vec4<f32>(by & vec4<u32>(0x0fu)) - vec4<f32>(8.0);
                let hi = vec4<f32>(by >> vec4<u32>(4u)) - vec4<f32>(8.0);
                let xb = xbase + w * 4u;
                let xlo = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
                let xhi = vec4<f32>(x[xb + 16u], x[xb + 17u], x[xb + 18u], x[xb + 19u]);
                acc = acc + d * (dot(lo, xlo) + dot(hi, xhi));
            }
        }
    }
    let sg_sum = subgroupAdd(acc);
    if (t == 0u && row < out_dim) { y[row] = sg_sum; }
}
"#;

/// Q6_K GEMV. Superblock = 256 values: `ql`(128 B), `qh`(64 B), 16 sub-scales,
/// and `d`. Value `y = d * sub_scale * (q - 32)` with `q` a 6-bit code (4 low
/// bits in `ql`, 2 high bits in `qh`). Scales are pre-folded to `d * sub_scale`.
const SHADER_Q6: &str = r#"
@group(0) @binding(0) var<storage, read> scales: array<f32>;
@group(0) @binding(1) var<storage, read> ql: array<u32>;     // 32 per block
@group(0) @binding(2) var<storage, read> qh: array<u32>;     // 16 per block
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> dims: vec4<u32>; // (in_dim, out_dim, grid_x, _)

var<workgroup> partial: array<f32, 64>;

fn byte_at(arr_word: u32, idx: u32) -> u32 {
    return (arr_word >> ((idx & 3u) * 8u)) & 0xffu;
}

// One workgroup (64 threads) per row; threads stride over superblocks for
// coalesced loads, then reduce in shared memory (see the Q4_0 kernel).
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let in_dim = dims.x;
    let out_dim = dims.y;
    let row = wid.x + wid.y * dims.z;
    let t = lid.x;
    let nblocks = in_dim / 256u;
    let row_blk = row * nblocks;

    var acc = 0.0;
    if (row < out_dim) {
        for (var b = t; b < nblocks; b = b + 64u) {
            let blk = row_blk + b;
            let scbase = blk * 16u;
            let qlbase = blk * 32u;  // 32 u32 = 128 bytes
            let qhbase = blk * 16u;  // 16 u32 = 64 bytes
            let xbase = b * 256u;

            for (var half = 0u; half < 2u; half = half + 1u) {
                for (var l = 0u; l < 32u; l = l + 1u) {
                    let is = l / 16u;
                    let qli0 = half * 64u + l;
                    let qli1 = half * 64u + l + 32u;
                    let qhi = half * 32u + l;
                    let qlb0 = byte_at(ql[qlbase + qli0 / 4u], qli0);
                    let qlb1 = byte_at(ql[qlbase + qli1 / 4u], qli1);
                    let qhb = byte_at(qh[qhbase + qhi / 4u], qhi);

                    let q1 = i32((qlb0 & 0x0fu) | ((qhb & 3u) << 4u)) - 32;
                    let q2 = i32((qlb1 & 0x0fu) | (((qhb >> 2u) & 3u) << 4u)) - 32;
                    let q3 = i32((qlb0 >> 4u) | (((qhb >> 4u) & 3u) << 4u)) - 32;
                    let q4 = i32((qlb1 >> 4u) | (((qhb >> 6u) & 3u) << 4u)) - 32;

                    let pos = half * 128u + l;
                    let si = half * 8u + is;
                    acc = acc
                        + scales[scbase + si] * f32(q1) * x[xbase + pos]
                        + scales[scbase + si + 2u] * f32(q2) * x[xbase + pos + 32u]
                        + scales[scbase + si + 4u] * f32(q3) * x[xbase + pos + 64u]
                        + scales[scbase + si + 6u] * f32(q4) * x[xbase + pos + 96u];
                }
            }
        }
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u && row < out_dim) { y[row] = partial[0]; }
}
"#;

/// Q6_K GEMV, subgroup-reduction variant (see `SHADER_SG`). Requires `SUBGROUP`.
pub(crate) const SHADER_Q6_SG: &str = r#"
@group(0) @binding(0) var<storage, read> scales: array<f32>;
@group(0) @binding(1) var<storage, read> ql: array<u32>;
@group(0) @binding(2) var<storage, read> qh: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> dims: vec4<u32>;

var<workgroup> partial: array<f32, 2>;

fn byte_at(arr_word: u32, idx: u32) -> u32 {
    return (arr_word >> ((idx & 3u) * 8u)) & 0xffu;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sg_size: u32, @builtin(subgroup_invocation_id) sg_inv: u32) {
    let in_dim = dims.x;
    let out_dim = dims.y;
    let row = wid.x + wid.y * dims.z;
    let t = lid.x;
    let nblocks = in_dim / 256u;
    let row_blk = row * nblocks;

    var acc = 0.0;
    if (row < out_dim) {
        for (var b = t; b < nblocks; b = b + 64u) {
            let blk = row_blk + b;
            let scbase = blk * 16u;
            let qlbase = blk * 32u;
            let qhbase = blk * 16u;
            let xbase = b * 256u;
            for (var half = 0u; half < 2u; half = half + 1u) {
                for (var l = 0u; l < 32u; l = l + 1u) {
                    let is = l / 16u;
                    let qli0 = half * 64u + l;
                    let qli1 = half * 64u + l + 32u;
                    let qhi = half * 32u + l;
                    let qlb0 = byte_at(ql[qlbase + qli0 / 4u], qli0);
                    let qlb1 = byte_at(ql[qlbase + qli1 / 4u], qli1);
                    let qhb = byte_at(qh[qhbase + qhi / 4u], qhi);
                    let q1 = i32((qlb0 & 0x0fu) | ((qhb & 3u) << 4u)) - 32;
                    let q2 = i32((qlb1 & 0x0fu) | (((qhb >> 2u) & 3u) << 4u)) - 32;
                    let q3 = i32((qlb0 >> 4u) | (((qhb >> 4u) & 3u) << 4u)) - 32;
                    let q4 = i32((qlb1 >> 4u) | (((qhb >> 6u) & 3u) << 4u)) - 32;
                    let pos = half * 128u + l;
                    let si = half * 8u + is;
                    acc = acc
                        + scales[scbase + si] * f32(q1) * x[xbase + pos]
                        + scales[scbase + si + 2u] * f32(q2) * x[xbase + pos + 32u]
                        + scales[scbase + si + 4u] * f32(q3) * x[xbase + pos + 64u]
                        + scales[scbase + si + 6u] * f32(q4) * x[xbase + pos + 96u];
                }
            }
        }
    }
    let sg_sum = subgroupAdd(acc);
    if (sg_inv == 0u) { partial[t / sg_size] = sg_sum; }
    workgroupBarrier();
    if (t == 0u && row < out_dim) {
        let n_sg = 64u / sg_size;
        var total = partial[0];
        for (var i = 1u; i < n_sg; i = i + 1u) { total = total + partial[i]; }
        y[row] = total;
    }
}
"#;

impl GpuQ4 {
    /// Initialize the Vulkan device and compile the Q4_0 GEMV kernel.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| StrixError::Backend {
                backend: "vulkan",
                message: "no Vulkan adapter found".into(),
            })?;
        let adapter_name = adapter.get_info().name;

        // Subgroup reduction (one `subgroupAdd` vs a 6-barrier tree) is a win for
        // small-K rows; use it when the adapter supports it (RDNA does).
        let use_sg = adapter.features().contains(wgpu::Features::SUBGROUP);
        // SPIR-V passthrough lets us load a glslc-compiled dp4a kernel (hardware
        // int8 dot) that naga can't emit — the key to matching llama.cpp's
        // integer GEMV without a wgpu upgrade or any new dependency.
        let use_pt = use_sg
            && adapter
                .features()
                .contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH);
        let mut features = wgpu::Features::empty();
        if use_sg {
            features |= wgpu::Features::SUBGROUP;
        }
        if use_pt {
            features |= wgpu::Features::SPIRV_SHADER_PASSTHROUGH;
        }

        // Use the adapter's real limits, not the conservative downlevel
        // defaults: large weights (e.g. a 262k-vocab lm_head ~0.5 GB Q4_0)
        // exceed the default 128 MB max storage-buffer binding size.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("strix-q4"),
                    required_features: features,
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| StrixError::Backend {
                backend: "vulkan",
                message: format!("request_device: {e}"),
            })?;

        let (src_q4, src_q6) = if use_sg {
            (SHADER_SG, SHADER_Q6_SG)
        } else {
            (SHADER, SHADER_Q6)
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q4_gemv"),
            source: wgpu::ShaderSource::Wgsl(src_q4.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q4_gemv"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let module_q6 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q6_gemv"),
            source: wgpu::ShaderSource::Wgsl(src_q6.into()),
        });
        let pipeline_q6 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q6_gemv"),
            layout: None,
            module: &module_q6,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Helper for the simple element-wise pipelines (all entry point "main").
        let mkpipe = |label: &str, src: &str| {
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &m,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline_rmsnorm = mkpipe("rmsnorm", SHADER_RMSNORM);
        let pipeline_rope = mkpipe("rope", SHADER_ROPE);
        let pipeline_geglu = mkpipe("geglu", SHADER_GEGLU);
        let pipeline_softcap = mkpipe("softcap", SHADER_SOFTCAP);
        let pipeline_add = mkpipe("add", SHADER_ADD);
        let pipeline_scale = mkpipe("scale", SHADER_SCALE);
        let pipeline_sdpa = mkpipe("sdpa", SHADER_SDPA);
        let pipeline_addnorm = mkpipe("addnorm", SHADER_ADDNORM);
        let pipeline_xquant = mkpipe("xquant", SHADER_XQUANT);

        // dp4a GEMV: load the glslc-compiled SPIR-V via passthrough with an
        // explicit bind-group layout (no naga reflection). 5 storage + 1 uniform.
        let (pipeline_dp4a, dp4a_bgl) = if use_pt {
            let ro = wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            };
            let rw = wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            };
            let uni = wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            };
            let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            };
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dp4a"),
                entries: &[
                    entry(0, ro),
                    entry(1, ro),
                    entry(2, ro),
                    entry(3, ro),
                    entry(4, rw),
                    entry(5, uni),
                ],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dp4a"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
            // SAFETY: SPIR-V from our own glslc build of q4_dp4a.comp; the explicit
            // layout matches its binding declarations.
            let module = unsafe {
                device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                    label: Some("q4_dp4a"),
                    source: wgpu::util::make_spirv_raw(include_bytes!("shaders/q4_dp4a.spv")),
                })
            };
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("q4_dp4a"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            (Some(p), Some(bgl))
        } else {
            (None, None)
        };

        Ok(GpuQ4 {
            device,
            queue,
            pipeline,
            pipeline_q6,
            pipeline_rmsnorm,
            pipeline_rope,
            pipeline_geglu,
            pipeline_softcap,
            pipeline_add,
            pipeline_scale,
            pipeline_sdpa,
            pipeline_addnorm,
            pipeline_xquant,
            pipeline_dp4a,
            dp4a_bgl,
            rows_per_wg: 1, // 1 row/workgroup; 2-row regressed at both wg=64 and 32.
            adapter_name,
        })
    }

    /// Adapter name (e.g. "AMD Radeon 890M Graphics (RADV STRIX1)").
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Allocate the persistent per-weight I/O buffers: `x` (write each call),
    /// `y` (kernel output), and a mappable `staging` copy of `y`.
    fn alloc_io(
        &self,
        in_dim: usize,
        out_dim: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let dev = &self.device;
        let x_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("x"),
            size: (in_dim * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let y_size = (out_dim * std::mem::size_of::<f32>()) as u64;
        let y_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: y_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: y_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (x_buf, y_buf, staging)
    }

    /// Dispatch a GEMV pipeline over `out_dim` rows and read `y` back to host.
    /// Assumes `x` was already written into the bound input buffer.
    fn run(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        y_buf: &wgpu::Buffer,
        staging: &wgpu::Buffer,
        out_dim: usize,
        groups: usize,
    ) -> Result<Vec<f32>> {
        let dev = &self.device;
        let y_size = (out_dim * std::mem::size_of::<f32>()) as u64;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            // One workgroup per row-group, tiled into a 2D grid.
            let gx = grid_x(groups);
            let gy = (groups as u32).div_ceil(gx);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(y_buf, 0, staging, 0, y_size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::Maintain::Wait);
        rx.recv()
            .ok()
            .and_then(|r| r.ok())
            .ok_or_else(|| StrixError::Backend {
                backend: "vulkan",
                message: "buffer map failed".into(),
            })?;
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(out)
    }

    /// RMSNorm `n_rows` independent rows of length `dim` (host in/out). `w` is a
    /// per-row shared weight of length `dim` (e.g. attn_norm, or the per-head
    /// q/k norm), or `None` for the no-weight V-norm. Standalone form used to
    /// validate the kernel; the Stage-C forward will use an encoder-recording
    /// variant on resident buffers.
    pub fn rmsnorm(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        n_rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if x.len() != n_rows * dim {
            return Err(StrixError::invalid("rmsnorm: x len != n_rows*dim"));
        }
        if let Some(w) = w {
            if w.len() != dim {
                return Err(StrixError::invalid("rmsnorm: weight len != dim"));
            }
        }
        let dev = &self.device;
        let x_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rms-x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let w_owned: Vec<f32> = match w {
            Some(w) => w.to_vec(),
            None => vec![0.0; dim], // dummy (guarded by has_weight=0)
        };
        let w_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rms-w"),
            contents: bytemuck::cast_slice(&w_owned),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let y_size = std::mem::size_of_val(x) as u64;
        let y_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms-y"),
            size: y_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms-staging"),
            size: y_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let p = [
            dim as u32,
            n_rows as u32,
            u32::from(w.is_some()),
            eps.to_bits(),
        ];
        let p_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rms-p"),
            contents: bytemuck::cast_slice(&p),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsnorm"),
            layout: &self.pipeline_rmsnorm.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rmsnorm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_rmsnorm);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_rows as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, y_size);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::Maintain::Wait);
        rx.recv()
            .ok()
            .and_then(|r| r.ok())
            .ok_or_else(|| StrixError::Backend {
                backend: "vulkan",
                message: "buffer map failed".into(),
            })?;
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(out)
    }

    /// Generic single-dispatch element-wise op (host in/out, for validation +
    /// standalone use). `inputs` become STORAGE buffers at bindings `0..n`; the
    /// `params` uniform binds at `n`. Buffer `rw` is read back. `groups` = x-dim
    /// workgroup count.
    fn run_op(
        &self,
        pipeline: &wgpu::ComputePipeline,
        inputs: &[&[f32]],
        rw: usize,
        params: [u32; 4],
        groups: u32,
    ) -> Result<Vec<f32>> {
        let dev = &self.device;
        let bufs: Vec<wgpu::Buffer> = inputs
            .iter()
            .enumerate()
            .map(|(i, data)| {
                let usage = if i == rw {
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                } else {
                    wgpu::BufferUsages::STORAGE
                };
                dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("op-in"),
                    contents: bytemuck::cast_slice(data),
                    usage,
                })
            })
            .collect();
        let p_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("op-p"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let rw_size = std::mem::size_of_val(inputs[rw]) as u64;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("op-staging"),
            size: rw_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: p_buf.as_entire_binding(),
        });
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("op"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("op"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&bufs[rw], 0, &staging, 0, rw_size);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::Maintain::Wait);
        rx.recv()
            .ok()
            .and_then(|r| r.ok())
            .ok_or_else(|| StrixError::Backend {
                backend: "vulkan",
                message: "buffer map failed".into(),
            })?;
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(out)
    }

    /// RoPE per head (host in/out). `v` is `n_heads × head_dim`; `ff` is the
    /// per-pair freq factors (length head_dim/2) or `None` (→ all 1.0).
    pub fn rope(
        &self,
        v: &[f32],
        ff: Option<&[f32]>,
        head_dim: usize,
        n_heads: usize,
        pos: usize,
        theta: f32,
    ) -> Result<Vec<f32>> {
        let half = head_dim / 2;
        if v.len() != n_heads * head_dim {
            return Err(StrixError::invalid("rope: v len != n_heads*head_dim"));
        }
        let ones = vec![1.0f32; half];
        let ffv = ff.unwrap_or(&ones);
        if ffv.len() != half {
            return Err(StrixError::invalid("rope: ff len != head_dim/2"));
        }
        let groups = ((n_heads * half) as u32).div_ceil(64);
        self.run_op(
            &self.pipeline_rope,
            &[v, ffv],
            0,
            [head_dim as u32, n_heads as u32, pos as u32, theta.to_bits()],
            groups,
        )
    }

    /// GeGLU (host in/out): `out = gelu_tanh(gate) * up`.
    pub fn geglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        if gate.len() != up.len() {
            return Err(StrixError::invalid("geglu: gate/up len mismatch"));
        }
        let n = gate.len();
        let out = vec![0.0f32; n];
        self.run_op(
            &self.pipeline_geglu,
            &[gate, up, &out],
            2,
            [n as u32, 0, 0, 0],
            (n as u32).div_ceil(64),
        )
    }

    /// Logit soft-cap (host in/out): `x = cap * tanh(x / cap)`.
    pub fn softcap(&self, x: &[f32], cap: f32) -> Result<Vec<f32>> {
        let n = x.len();
        self.run_op(
            &self.pipeline_softcap,
            &[x],
            0,
            [n as u32, cap.to_bits(), 0, 0],
            (n as u32).div_ceil(64),
        )
    }

    /// Residual add (host in/out): `a + b`.
    pub fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(StrixError::invalid("add: len mismatch"));
        }
        let n = a.len();
        self.run_op(
            &self.pipeline_add,
            &[a, b],
            0,
            [n as u32, 0, 0, 0],
            (n as u32).div_ceil(64),
        )
    }

    /// Scalar multiply (host in/out): `a * s`.
    pub fn scale(&self, a: &[f32], s: f32) -> Result<Vec<f32>> {
        let n = a.len();
        self.run_op(
            &self.pipeline_scale,
            &[a],
            0,
            [n as u32, s.to_bits(), 0, 0],
            (n as u32).div_ceil(64),
        )
    }

    /// Single-query GQA SDPA (host in/out). `q` is `n_heads × head_dim`; `k`/`v`
    /// are `n_kv × len × head_dim`. Returns `n_heads × head_dim`. `scale` is the
    /// attention scale (1.0 for gemma4, 1/√head_dim for gemma3).
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        head_dim: usize,
        n_heads: usize,
        n_kv: usize,
        len: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        if q.len() != n_heads * head_dim
            || k.len() != n_kv * len * head_dim
            || v.len() != n_kv * len * head_dim
        {
            return Err(StrixError::invalid("sdpa: shape mismatch"));
        }
        let groups = (n_heads / n_kv.max(1)).max(1);
        let dev = &self.device;
        let mk = |label, data: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let q_buf = mk("sdpa-q", bytemuck::cast_slice(q));
        let k_buf = mk("sdpa-k", bytemuck::cast_slice(k));
        let v_buf = mk("sdpa-v", bytemuck::cast_slice(v));
        let out_len = n_heads * head_dim;
        let out_size = (out_len * std::mem::size_of::<f32>()) as u64;
        let out_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdpa-out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // p[0]=(head_dim,len,groups,n_kv), p[1].x=scale. Token-major K/V: [len][n_kv][hd].
        let p: [u32; 8] = [
            head_dim as u32,
            len as u32,
            groups as u32,
            n_kv as u32,
            scale.to_bits(),
            0,
            0,
            0,
        ];
        let p_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdpa-p"),
            contents: bytemuck::cast_slice(&p),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdpa-staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdpa"),
            layout: &self.pipeline_sdpa.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdpa"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_sdpa);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_heads as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::Maintain::Wait);
        rx.recv()
            .ok()
            .and_then(|r| r.ok())
            .ok_or_else(|| StrixError::Backend {
                backend: "vulkan",
                message: "buffer map failed".into(),
            })?;
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(out)
    }

    /// Upload a Q4_0 weight `[out_dim, in_dim]` (raw GGUF tensor bytes) to the
    /// GPU, repacked into the resident `scales`/`quants` layout.
    ///
    /// `bytes` must be exactly `out_dim * (in_dim/32) * 18` long, and `in_dim`
    /// must be a multiple of 32 (the Q4_0 block size).
    pub fn resident_from_q4_0(
        &self,
        bytes: &[u8],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<ResidentQ4> {
        if in_dim % QK4_0 != 0 {
            return Err(StrixError::invalid(
                "q4 gemv: in_dim must be a multiple of 32",
            ));
        }
        let nblocks = in_dim / QK4_0;
        let total_blocks = nblocks * out_dim;
        let expected = total_blocks * Q4_0_BYTES;
        if bytes.len() != expected {
            return Err(StrixError::invalid(format!(
                "q4 gemv: expected {expected} bytes for [{out_dim},{in_dim}] Q4_0, got {}",
                bytes.len()
            )));
        }

        // Keep the original f16 scale bits (lossless), packed two per u32; the
        // shader decodes with `unpack2x16float`. Saves 2 B/block of traffic.
        let mut scales = vec![0u32; total_blocks.div_ceil(2)];
        let mut quants = vec![0u32; total_blocks * 4];
        for (b, blk) in bytes.chunks_exact(Q4_0_BYTES).enumerate() {
            let h = u16::from_le_bytes([blk[0], blk[1]]) as u32;
            scales[b >> 1] |= h << (16 * (b & 1));
            let qs = &blk[2..18]; // 16 nibble bytes
            for w in 0..4 {
                quants[b * 4 + w] =
                    u32::from_le_bytes([qs[w * 4], qs[w * 4 + 1], qs[w * 4 + 2], qs[w * 4 + 3]]);
            }
        }

        let dev = &self.device;
        let scales_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4-scales"),
            contents: bytemuck::cast_slice(&scales),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let quants_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4-quants"),
            contents: bytemuck::cast_slice(&quants),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let grid_x = grid_x(out_dim.div_ceil(self.rows_per_wg));
        let dims = [in_dim as u32, out_dim as u32, grid_x, 0];
        let dims_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4-dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (x_buf, y_buf, staging) = self.alloc_io(in_dim, out_dim);
        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q4_gemv"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scales_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: quants_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        Ok(ResidentQ4 {
            scales: scales_buf,
            quants: quants_buf,
            dims: dims_buf,
            x_buf,
            y_buf,
            staging,
            bind_group,
            in_dim,
            out_dim,
        })
    }

    /// Compute `y = W · x` for a resident Q4_0 weight. `x` has length `in_dim`;
    /// the result has length `out_dim`. The weight and all I/O buffers stay on
    /// the GPU; only `x` is written and only `y` is read back — no allocation.
    pub fn gemv(&self, m: &ResidentQ4, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != m.in_dim {
            return Err(StrixError::invalid("q4 gemv: x length != in_dim"));
        }
        self.queue
            .write_buffer(&m.x_buf, 0, bytemuck::cast_slice(x));
        self.run(
            &self.pipeline,
            &m.bind_group,
            &m.y_buf,
            &m.staging,
            m.out_dim,
            m.out_dim.div_ceil(self.rows_per_wg),
        )
    }

    /// True if the hardware-dp4a GEMV path is available.
    pub fn has_dp4a(&self) -> bool {
        self.pipeline_dp4a.is_some()
    }

    /// Q4_0 GEMV via hardware dp4a: quantize `x` to int8, then run the int-dot
    /// kernel against the resident int4 weights. Reuses `m`'s resident weights
    /// and y/staging buffers; allocates transient xq/xd (host-method form).
    pub fn gemv_dp4a(&self, m: &ResidentQ4, x: &[f32]) -> Result<Vec<f32>> {
        let pipeline = self
            .pipeline_dp4a
            .as_ref()
            .ok_or_else(|| StrixError::invalid("dp4a GEMV unavailable"))?;
        let bgl = self.dp4a_bgl.as_ref().unwrap();
        if x.len() != m.in_dim {
            return Err(StrixError::invalid("dp4a gemv: x length != in_dim"));
        }
        let dev = &self.device;
        let nblocks = m.in_dim / QK4_0;
        let x_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dp4a-x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let xq = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dp4a-xq"),
            size: (nblocks * 8 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let xd = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dp4a-xd"),
            size: (nblocks * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let xqp = self.uniform(&[nblocks as u32, 0, 0, 0]);
        let xq_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xquant"),
            layout: &self.pipeline_xquant.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: xq.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: xd.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: xqp.as_entire_binding(),
                },
            ],
        });
        let dp = self.uniform(&[m.in_dim as u32, m.out_dim as u32, grid_x(m.out_dim), 0]);
        let dp_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dp4a"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: m.scales.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: m.quants.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: xq.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: xd.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: m.y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dp.as_entire_binding(),
                },
            ],
        });
        let y_size = (m.out_dim * std::mem::size_of::<f32>()) as u64;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("xquant"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_xquant);
            pass.set_bind_group(0, &xq_bg, &[]);
            pass.dispatch_workgroups((nblocks as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dp4a"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &dp_bg, &[]);
            let gx = grid_x(m.out_dim);
            let gy = (m.out_dim as u32).div_ceil(gx);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&m.y_buf, 0, &m.staging, 0, y_size);
        self.queue.submit(Some(enc.finish()));
        let slice = m.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::Maintain::Wait);
        rx.recv()
            .ok()
            .and_then(|r| r.ok())
            .ok_or_else(|| StrixError::Backend {
                backend: "vulkan",
                message: "buffer map failed".into(),
            })?;
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        m.staging.unmap();
        Ok(out)
    }

    /// Batched Q4_0 GEMV: run several `(weight, x)` pairs in a **single** command
    /// submission with **one** GPU sync, instead of one submit+poll each. The
    /// decode forward uses this for groups that share an input (q/k/v, gate/up),
    /// cutting the per-token CPU↔GPU round-trips. Results are returned in order.
    pub fn gemv_batch_q4(&self, items: &[(&ResidentQ4, &[f32])]) -> Result<Vec<Vec<f32>>> {
        for (m, x) in items {
            if x.len() != m.in_dim {
                return Err(StrixError::invalid("q4 gemv: x length != in_dim"));
            }
            self.queue
                .write_buffer(&m.x_buf, 0, bytemuck::cast_slice(x));
        }

        let dev = &self.device;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (m, _) in items {
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("q4_gemv_batch"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &m.bind_group, &[]);
                let groups = m.out_dim.div_ceil(self.rows_per_wg);
                let gx = grid_x(groups);
                let gy = (groups as u32).div_ceil(gx);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            let y_size = (m.out_dim * std::mem::size_of::<f32>()) as u64;
            enc.copy_buffer_to_buffer(&m.y_buf, 0, &m.staging, 0, y_size);
        }
        self.queue.submit(Some(enc.finish()));

        // Map all staging buffers, then a single poll drives them all to ready.
        let mut rxs = Vec::with_capacity(items.len());
        for (m, _) in items {
            let (tx, rx) = std::sync::mpsc::channel();
            m.staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            rxs.push(rx);
        }
        dev.poll(wgpu::Maintain::Wait);

        let mut out = Vec::with_capacity(items.len());
        for ((m, _), rx) in items.iter().zip(rxs) {
            rx.recv()
                .ok()
                .and_then(|r| r.ok())
                .ok_or_else(|| StrixError::Backend {
                    backend: "vulkan",
                    message: "buffer map failed".into(),
                })?;
            let slice = m.staging.slice(..);
            out.push(bytemuck::cast_slice(&slice.get_mapped_range()).to_vec());
            m.staging.unmap();
        }
        Ok(out)
    }

    /// Upload a Q6_K weight `[out_dim, in_dim]` (raw GGUF bytes) resident,
    /// repacked into folded `scales` (16 f32/block) + `ql`/`qh` `u32` arrays.
    /// `in_dim` must be a multiple of 256 (the Q6_K superblock size).
    pub fn resident_from_q6_k(
        &self,
        bytes: &[u8],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<ResidentQ6> {
        if in_dim % QK_K != 0 {
            return Err(StrixError::invalid(
                "q6 gemv: in_dim must be a multiple of 256",
            ));
        }
        let nblocks = in_dim / QK_K;
        let total_blocks = nblocks * out_dim;
        let expected = total_blocks * Q6_K_BYTES;
        if bytes.len() != expected {
            return Err(StrixError::invalid(format!(
                "q6 gemv: expected {expected} bytes for [{out_dim},{in_dim}] Q6_K, got {}",
                bytes.len()
            )));
        }

        let mut scales = vec![0.0f32; total_blocks * 16];
        let mut ql = vec![0u32; total_blocks * 32];
        let mut qh = vec![0u32; total_blocks * 16];
        for (b, blk) in bytes.chunks_exact(Q6_K_BYTES).enumerate() {
            let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
            for j in 0..16 {
                scales[b * 16 + j] = d * (blk[192 + j] as i8) as f32;
            }
            for w in 0..32 {
                // ql: bytes [0..128]
                ql[b * 32 + w] = u32::from_le_bytes([
                    blk[w * 4],
                    blk[w * 4 + 1],
                    blk[w * 4 + 2],
                    blk[w * 4 + 3],
                ]);
            }
            for w in 0..16 {
                // qh: bytes [128..192]
                qh[b * 16 + w] = u32::from_le_bytes([
                    blk[128 + w * 4],
                    blk[128 + w * 4 + 1],
                    blk[128 + w * 4 + 2],
                    blk[128 + w * 4 + 3],
                ]);
            }
        }

        let dev = &self.device;
        let mk = |label, data: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let scales_buf = mk("q6-scales", bytemuck::cast_slice(&scales));
        let ql_buf = mk("q6-ql", bytemuck::cast_slice(&ql));
        let qh_buf = mk("q6-qh", bytemuck::cast_slice(&qh));
        let dims = [in_dim as u32, out_dim as u32, grid_x(out_dim), 0];
        let dims_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q6-dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (x_buf, y_buf, staging) = self.alloc_io(in_dim, out_dim);
        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q6_gemv"),
            layout: &self.pipeline_q6.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scales_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ql_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: qh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        Ok(ResidentQ6 {
            scales: scales_buf,
            ql: ql_buf,
            qh: qh_buf,
            dims: dims_buf,
            x_buf,
            y_buf,
            staging,
            bind_group,
            in_dim,
            out_dim,
        })
    }

    /// Compute `y = W · x` for a resident Q6_K weight (reuses resident buffers).
    pub fn gemv_q6(&self, m: &ResidentQ6, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != m.in_dim {
            return Err(StrixError::invalid("q6 gemv: x length != in_dim"));
        }
        self.queue
            .write_buffer(&m.x_buf, 0, bytemuck::cast_slice(x));
        self.run(
            &self.pipeline_q6,
            &m.bind_group,
            &m.y_buf,
            &m.staging,
            m.out_dim,
            m.out_dim, // Q6_K kernel is 1 row per workgroup
        )
    }

    // ---- Stage-C: record ops into a shared encoder, on resident GPU buffers ----
    // These never submit or read back; the decoder records a whole token's
    // forward into one encoder and submits once. Transient bind groups / uniform
    // buffers are retained by the command buffer until submission completes.

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn uniform(&self, data: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("op-uniform"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    fn rec(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        storage: &[&wgpu::Buffer],
        uniform: &wgpu::Buffer,
        groups: (u32, u32, u32),
    ) {
        let mut entries: Vec<wgpu::BindGroupEntry> = storage
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: storage.len() as u32,
            resource: uniform.as_entire_binding(),
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("op"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("op"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rec_rmsnorm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        y: &wgpu::Buffer,
        n_rows: usize,
        dim: usize,
        eps: f32,
        has_weight: bool,
    ) {
        let u = self.uniform(&[
            dim as u32,
            n_rows as u32,
            u32::from(has_weight),
            eps.to_bits(),
        ]);
        self.rec(
            enc,
            &self.pipeline_rmsnorm,
            &[x, w, y],
            &u,
            (n_rows as u32, 1, 1),
        );
    }

    pub(crate) fn rec_gemv_q4(
        &self,
        enc: &mut wgpu::CommandEncoder,
        m: &ResidentQ4,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
    ) {
        let groups = m.out_dim.div_ceil(self.rows_per_wg);
        let gx = grid_x(groups);
        let gy = (groups as u32).div_ceil(gx);
        self.rec(
            enc,
            &self.pipeline,
            &[&m.scales, &m.quants, x, y],
            &m.dims,
            (gx, gy, 1),
        );
    }

    pub(crate) fn rec_gemv_q6(
        &self,
        enc: &mut wgpu::CommandEncoder,
        m: &ResidentQ6,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
    ) {
        let gx = grid_x(m.out_dim);
        let gy = (m.out_dim as u32).div_ceil(gx);
        self.rec(
            enc,
            &self.pipeline_q6,
            &[&m.scales, &m.ql, &m.qh, x, y],
            &m.dims,
            (gx, gy, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(crate) fn rec_rope(
        &self,
        enc: &mut wgpu::CommandEncoder,
        v: &wgpu::Buffer,
        ff: &wgpu::Buffer,
        head_dim: usize,
        n_heads: usize,
        pos: usize,
        theta: f32,
    ) {
        let total = (n_heads * (head_dim / 2)) as u32;
        let u = self.uniform(&[head_dim as u32, n_heads as u32, pos as u32, theta.to_bits()]);
        self.rec(
            enc,
            &self.pipeline_rope,
            &[v, ff],
            &u,
            (total.div_ceil(64), 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rec_sdpa(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        out: &wgpu::Buffer,
        head_dim: usize,
        n_heads: usize,
        len: usize,
        groups: usize,
        n_kv: usize,
        scale: f32,
    ) {
        let u = self.uniform(&[
            head_dim as u32,
            len as u32,
            groups as u32,
            n_kv as u32,
            scale.to_bits(),
            0,
            0,
            0,
        ]);
        self.rec(
            enc,
            &self.pipeline_sdpa,
            &[q, k, v, out],
            &u,
            (n_heads as u32, 1, 1),
        );
    }

    pub(crate) fn rec_geglu(
        &self,
        enc: &mut wgpu::CommandEncoder,
        gate: &wgpu::Buffer,
        up: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n: usize,
    ) {
        let u = self.uniform(&[n as u32, 0, 0, 0]);
        self.rec(
            enc,
            &self.pipeline_geglu,
            &[gate, up, out],
            &u,
            ((n as u32).div_ceil(64), 1, 1),
        );
    }

    #[allow(dead_code)]
    pub(crate) fn rec_add(
        &self,
        enc: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        n: usize,
    ) {
        let u = self.uniform(&[n as u32, 0, 0, 0]);
        self.rec(
            enc,
            &self.pipeline_add,
            &[a, b],
            &u,
            ((n as u32).div_ceil(64), 1, 1),
        );
    }

    #[allow(dead_code)]
    pub(crate) fn rec_scale(
        &self,
        enc: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        n: usize,
        s: f32,
    ) {
        let u = self.uniform(&[n as u32, s.to_bits(), 0, 0]);
        self.rec(
            enc,
            &self.pipeline_scale,
            &[a],
            &u,
            ((n as u32).div_ceil(64), 1, 1),
        );
    }

    pub(crate) fn rec_softcap(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        n: usize,
        cap: f32,
    ) {
        let u = self.uniform(&[n as u32, cap.to_bits(), 0, 0]);
        self.rec(
            enc,
            &self.pipeline_softcap,
            &[x],
            &u,
            ((n as u32).div_ceil(64), 1, 1),
        );
    }

    /// Several Q4_0 GEMVs sharing input `x`, distinct outputs, in ONE compute
    /// pass (no barriers between — independent). Used for q/k/v and gate/up.
    pub(crate) fn rec_q4_multi(
        &self,
        enc: &mut wgpu::CommandEncoder,
        items: &[(&ResidentQ4, &wgpu::Buffer, &wgpu::Buffer)],
    ) {
        let bgs: Vec<wgpu::BindGroup> = items
            .iter()
            .map(|(m, x, y)| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("q4-multi"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: m.scales.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: m.quants.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: x.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: y.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: m.dims.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q4-multi"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for ((m, _, _), bg) in items.iter().zip(&bgs) {
            pass.set_bind_group(0, bg, &[]);
            let groups = m.out_dim.div_ceil(self.rows_per_wg);
            let gx = grid_x(groups);
            let gy = (groups as u32).div_ceil(gx);
            pass.dispatch_workgroups(gx, gy, 1);
        }
    }

    /// Fused residual-add + RMSNorm (+ scalar): `h = (h + rmsnorm(x)*w) * scale`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rec_addnorm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        h: &wgpu::Buffer,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        dim: usize,
        eps: f32,
        scale: f32,
    ) {
        let u = self.uniform(&[dim as u32, eps.to_bits(), scale.to_bits(), 0]);
        self.rec(enc, &self.pipeline_addnorm, &[h, x, w], &u, (1, 1, 1));
    }

    /// Several independent RMSNorms in ONE compute pass (no barrier between them).
    /// Each item: `(x, w, y, n_rows, dim, has_weight)`. Used to fuse the per-head
    /// q/k/v norms (each reads a distinct buffer, writes a distinct buffer).
    pub(crate) fn rec_rmsnorm_multi(
        &self,
        enc: &mut wgpu::CommandEncoder,
        items: &[(
            &wgpu::Buffer,
            &wgpu::Buffer,
            &wgpu::Buffer,
            usize,
            usize,
            bool,
        )],
    ) {
        let mut bgs = Vec::with_capacity(items.len());
        let mut groups = Vec::with_capacity(items.len());
        for (x, w, y, n_rows, dim, has_w) in items {
            let u = self.uniform(&[
                *dim as u32,
                *n_rows as u32,
                u32::from(*has_w),
                1e-6f32.to_bits(),
            ]);
            bgs.push((
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("rmsnorm-multi"),
                    layout: &self.pipeline_rmsnorm.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: x.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: w.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: y.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: u.as_entire_binding(),
                        },
                    ],
                }),
                u,
            ));
            groups.push(*n_rows as u32);
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rmsnorm-multi"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_rmsnorm);
        for ((bg, _u), g) in bgs.iter().zip(&groups) {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(*g, 1, 1);
        }
    }

    /// Several independent RoPE applications in ONE compute pass. Each item:
    /// `(v, ff, head_dim, n_heads, pos, theta)`. Used to fuse q and k RoPE.
    pub(crate) fn rec_rope_multi(
        &self,
        enc: &mut wgpu::CommandEncoder,
        items: &[(&wgpu::Buffer, &wgpu::Buffer, usize, usize, usize, f32)],
    ) {
        let mut bgs = Vec::with_capacity(items.len());
        let mut groups = Vec::with_capacity(items.len());
        for (v, ff, head_dim, n_heads, pos, theta) in items {
            let u = self.uniform(&[
                *head_dim as u32,
                *n_heads as u32,
                *pos as u32,
                theta.to_bits(),
            ]);
            bgs.push((
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("rope-multi"),
                    layout: &self.pipeline_rope.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: v.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: ff.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: u.as_entire_binding(),
                        },
                    ],
                }),
                u,
            ));
            groups.push(((n_heads * (head_dim / 2)) as u32).div_ceil(64));
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rope-multi"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_rope);
        for ((bg, _u), g) in bgs.iter().zip(&groups) {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(*g, 1, 1);
        }
    }
}

/// Minimal IEEE-754 half → f32 (matches `half::f16::to_f32`); avoids a dep here.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = if exp == 0 {
        // subnormal
        (mant as f32) * 2.0f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Q4_0 tensor `[out_dim, in_dim]` from deterministic nibbles and
    /// return (raw bytes, dequantized f32 reference).
    fn build_q4_0(in_dim: usize, out_dim: usize) -> (Vec<u8>, Vec<f32>) {
        let nblocks = in_dim / 32;
        let mut bytes = Vec::with_capacity(out_dim * nblocks * Q4_0_BYTES);
        let mut deq = vec![0.0f32; out_dim * in_dim];
        let mut seed = 0xC0FFEEu64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u32
        };
        for o in 0..out_dim {
            for b in 0..nblocks {
                // scale: a small positive f16 value.
                let d = 0.05f32 + (next() % 16) as f32 * 0.01;
                bytes.extend_from_slice(&f32_to_f16(d).to_le_bytes());
                let mut nibbles = [0u8; 32];
                for n in nibbles.iter_mut() {
                    *n = (next() % 16) as u8; // 0..15
                }
                // pack: byte j = low nibble j | (high nibble j+16) << 4
                for j in 0..16 {
                    bytes.push(nibbles[j] | (nibbles[j + 16] << 4));
                }
                // reference dequant: y = d * (nibble - 8)
                for (p, &nib) in nibbles.iter().enumerate() {
                    deq[o * in_dim + b * 32 + p] = d * (nib as f32 - 8.0);
                }
            }
        }
        (bytes, deq)
    }

    fn f32_to_f16(f: f32) -> u16 {
        // round-to-nearest-even, enough for test scales (no inf/nan/subnormal).
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x7fffff;
        if exp <= 0 {
            return sign;
        }
        let mant16 = (mant >> 13) as u16;
        sign | ((exp as u16) << 10) | mant16
    }

    fn cpu_gemv(deq: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        (0..out_dim)
            .map(|o| (0..in_dim).map(|i| deq[o * in_dim + i] * x[i]).sum())
            .collect()
    }

    /// Build a random Q6_K tensor `[out_dim, in_dim]` and its f32 reference
    /// (replicating `ggml_quant::dequant_q6_k`).
    fn build_q6_k(in_dim: usize, out_dim: usize) -> (Vec<u8>, Vec<f32>) {
        let nblocks = in_dim / 256;
        let mut bytes = Vec::with_capacity(out_dim * nblocks * Q6_K_BYTES);
        let mut deq = vec![0.0f32; out_dim * in_dim];
        let mut seed = 0xBEEF_5678u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u32
        };
        for o in 0..out_dim {
            for blk in 0..nblocks {
                let ql: Vec<u8> = (0..128).map(|_| (next() & 0xff) as u8).collect();
                let qh: Vec<u8> = (0..64).map(|_| (next() & 0xff) as u8).collect();
                let sc: Vec<i8> = (0..16).map(|_| ((next() % 64) as i32 - 32) as i8).collect();
                // Round d through f16 so the reference matches what the GPU reads
                // (the on-disk scale is f16); Q6_K's large sc*q amplifies the gap.
                let d = f16_to_f32(f32_to_f16(0.02f32 + (next() % 16) as f32 * 0.002));
                bytes.extend_from_slice(&ql);
                bytes.extend_from_slice(&qh);
                bytes.extend(sc.iter().map(|&s| s as u8));
                bytes.extend_from_slice(&f32_to_f16(d).to_le_bytes());

                let obase = o * in_dim + blk * 256;
                for half in 0..2 {
                    let qlh = &ql[half * 64..half * 64 + 64];
                    let qhh = &qh[half * 32..half * 32 + 32];
                    let sch = &sc[half * 8..half * 8 + 8];
                    let ybase = half * 128;
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((qlh[l] & 0x0F) | ((qhh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) as i32 - 32;
                        deq[obase + ybase + l] = d * (sch[is] as f32) * q1 as f32;
                        deq[obase + ybase + l + 32] = d * (sch[is + 2] as f32) * q2 as f32;
                        deq[obase + ybase + l + 64] = d * (sch[is + 4] as f32) * q3 as f32;
                        deq[obase + ybase + l + 96] = d * (sch[is + 6] as f32) * q4 as f32;
                    }
                }
            }
        }
        (bytes, deq)
    }

    #[test]
    fn gpu_q6_gemv_matches_cpu() {
        let gpu = match GpuQ4::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU Q6 test: {e}");
                return;
            }
        };
        let (in_dim, out_dim) = (512usize, 64usize);
        let (bytes, deq) = build_q6_k(in_dim, out_dim);
        let mut seed = 0x77u64;
        let x: Vec<f32> = (0..in_dim)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        let resident = gpu.resident_from_q6_k(&bytes, in_dim, out_dim).unwrap();
        let got = gpu.gemv_q6(&resident, &x).unwrap();
        let want = cpu_gemv(&deq, &x, in_dim, out_dim);
        let max_err = got
            .iter()
            .zip(&want)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-2, "Q6_K max err {max_err} too large");
        eprintln!("GPU Q6_K resident GEMV validated (max err {max_err:.2e})");
    }

    fn cpu_rmsnorm(x: &[f32], w: Option<&[f32]>, n_rows: usize, dim: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; n_rows * dim];
        for r in 0..n_rows {
            let row = &x[r * dim..(r + 1) * dim];
            let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let scale = 1.0 / (ss + eps).sqrt();
            for i in 0..dim {
                let g = w.map(|w| w[i]).unwrap_or(1.0);
                out[r * dim + i] = row[i] * scale * g;
            }
        }
        out
    }

    #[test]
    fn gpu_rmsnorm_matches_cpu() {
        let gpu = match GpuQ4::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU rmsnorm test: {e}");
                return;
            }
        };
        let mut seed = 0x51u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) - 0.5
        };
        let eps = 1e-6f32;
        // Case 1: full-vector norm with weight (attn/ffn norm).
        let (n_rows, dim) = (1usize, 3840usize);
        let x: Vec<f32> = (0..n_rows * dim).map(|_| rnd() * 4.0).collect();
        let w: Vec<f32> = (0..dim).map(|_| rnd() + 1.0).collect();
        let got = gpu.rmsnorm(&x, Some(&w), n_rows, dim, eps).unwrap();
        let want = cpu_rmsnorm(&x, Some(&w), n_rows, dim, eps);
        let e1 = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(e1 < 1e-4, "rmsnorm full-vector err {e1}");
        // Case 2: per-head no-weight (V-norm): 8 heads of 256.
        let (n_rows, dim) = (8usize, 256usize);
        let x: Vec<f32> = (0..n_rows * dim).map(|_| rnd() * 4.0).collect();
        let got = gpu.rmsnorm(&x, None, n_rows, dim, eps).unwrap();
        let want = cpu_rmsnorm(&x, None, n_rows, dim, eps);
        let e2 = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(e2 < 1e-4, "rmsnorm per-head err {e2}");
        eprintln!("GPU rmsnorm validated (full {e1:.1e}, per-head {e2:.1e})");
    }

    #[test]
    fn gpu_sdpa_matches_cpu() {
        let gpu = match GpuQ4::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU sdpa test: {e}");
                return;
            }
        };
        let mut seed = 0x2468u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) - 0.5
        };
        // GQA: 8 query heads, 2 kv heads (groups=4), head_dim 64, len 37.
        let (hd, n_heads, n_kv, len) = (64usize, 8usize, 2usize, 37usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let q: Vec<f32> = (0..n_heads * hd).map(|_| rnd()).collect();
        let k: Vec<f32> = (0..n_kv * len * hd).map(|_| rnd()).collect();
        let v: Vec<f32> = (0..n_kv * len * hd).map(|_| rnd()).collect();
        let got = gpu.sdpa(&q, &k, &v, hd, n_heads, n_kv, len, scale).unwrap();

        // CPU reference (token-major K/V: index (t*n_kv + kvh)*hd + d).
        let groups = n_heads / n_kv;
        let mut want = vec![0.0f32; n_heads * hd];
        for h in 0..n_heads {
            let kvh = h / groups;
            let mut scores = vec![0.0f32; len];
            let mut mx = f32::MIN;
            for (t, sc) in scores.iter_mut().enumerate() {
                let mut s = 0.0;
                for d in 0..hd {
                    s += q[h * hd + d] * k[(t * n_kv + kvh) * hd + d];
                }
                *sc = s * scale;
                mx = mx.max(*sc);
            }
            let mut sum = 0.0;
            for sc in scores.iter_mut() {
                *sc = (*sc - mx).exp();
                sum += *sc;
            }
            for d in 0..hd {
                let mut acc = 0.0;
                for (t, &sc) in scores.iter().enumerate() {
                    acc += sc * v[(t * n_kv + kvh) * hd + d];
                }
                want[h * hd + d] = acc / sum;
            }
        }
        let e = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(e < 1e-4, "sdpa err {e}");
        eprintln!("GPU SDPA (GQA) validated (max err {e:.1e})");
    }

    #[test]
    fn gpu_rope_geglu_softcap_match_cpu() {
        let gpu = match GpuQ4::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU rope/geglu/softcap test: {e}");
                return;
            }
        };
        let mut seed = 0x99u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) - 0.5
        };

        // RoPE: 4 heads of 256, pos 7, theta 1e6, with freq factors.
        let (head_dim, n_heads, pos, theta) = (256usize, 4usize, 7usize, 1_000_000.0f32);
        let half = head_dim / 2;
        let v: Vec<f32> = (0..n_heads * head_dim).map(|_| rnd() * 2.0).collect();
        let ff: Vec<f32> = (0..half).map(|_| 1.0 + 0.5 * (rnd() + 0.5)).collect();
        let got = gpu
            .rope(&v, Some(&ff), head_dim, n_heads, pos, theta)
            .unwrap();
        let mut want = v.clone();
        for h in 0..n_heads {
            let base = h * head_dim;
            for j in 0..half {
                let inv = theta.powf(-(2.0 * j as f32) / head_dim as f32) / ff[j];
                let (s, c) = (pos as f32 * inv).sin_cos();
                let (x1, x2) = (want[base + j], want[base + j + half]);
                want[base + j] = x1 * c - x2 * s;
                want[base + j + half] = x2 * c + x1 * s;
            }
        }
        let er = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(er < 1e-3, "rope err {er}");

        // GeGLU.
        let gate: Vec<f32> = (0..1024).map(|_| rnd() * 3.0).collect();
        let up: Vec<f32> = (0..1024).map(|_| rnd() * 3.0).collect();
        let got = gpu.geglu(&gate, &up).unwrap();
        let gelu = |x: f32| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh());
        let eg = got
            .iter()
            .enumerate()
            .map(|(i, &g)| (g - gelu(gate[i]) * up[i]).abs())
            .fold(0.0, f32::max);
        assert!(eg < 1e-3, "geglu err {eg}");

        // Softcap.
        let x: Vec<f32> = (0..512).map(|_| rnd() * 100.0).collect();
        let cap = 30.0f32;
        let got = gpu.softcap(&x, cap).unwrap();
        let es = got
            .iter()
            .enumerate()
            .map(|(i, &g)| (g - cap * (x[i] / cap).tanh()).abs())
            .fold(0.0, f32::max);
        assert!(es < 1e-3, "softcap err {es}");

        // add + scale.
        let a: Vec<f32> = (0..300).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..300).map(|_| rnd()).collect();
        let add = gpu.add(&a, &b).unwrap();
        let ea = add
            .iter()
            .enumerate()
            .map(|(i, &g)| (g - (a[i] + b[i])).abs())
            .fold(0.0, f32::max);
        let scaled = gpu.scale(&a, 0.375).unwrap();
        let esc = scaled
            .iter()
            .enumerate()
            .map(|(i, &g)| (g - a[i] * 0.375).abs())
            .fold(0.0, f32::max);
        assert!(ea < 1e-6 && esc < 1e-6, "add {ea} scale {esc}");
        eprintln!("GPU rope/geglu/softcap/add/scale validated (rope {er:.1e}, geglu {eg:.1e}, softcap {es:.1e})");
    }

    #[test]
    fn gpu_dp4a_matches_cpu() {
        let gpu = match GpuQ4::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping dp4a test: {e}");
                return;
            }
        };
        if !gpu.has_dp4a() {
            eprintln!("skipping dp4a test: passthrough/subgroup unavailable");
            return;
        }
        let (in_dim, out_dim) = (256usize, 96usize);
        let (bytes, deq) = build_q4_0(in_dim, out_dim);
        let mut seed = 0x1234u64;
        let x: Vec<f32> = (0..in_dim)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        let resident = gpu.resident_from_q4_0(&bytes, in_dim, out_dim).unwrap();
        let got = gpu.gemv_dp4a(&resident, &x).unwrap();
        let want = cpu_gemv(&deq, &x, in_dim, out_dim);
        // q8 activation quant adds error; tolerance is looser than the f32 path.
        let mut max_rel = 0.0f32;
        let scale = want.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        for (g, c) in got.iter().zip(&want) {
            max_rel = max_rel.max((g - c).abs() / scale);
        }
        assert!(max_rel < 2e-2, "dp4a rel err {max_rel} too large");
        eprintln!("GPU dp4a GEMV validated (rel err {max_rel:.2e})");
    }

    #[test]
    fn gpu_q4_gemv_matches_cpu() {
        let gpu = match GpuQ4::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU Q4 test: {e}");
                return;
            }
        };
        let (in_dim, out_dim) = (256usize, 96usize);
        let (bytes, deq) = build_q4_0(in_dim, out_dim);
        let mut seed = 0x1234u64;
        let x: Vec<f32> = (0..in_dim)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f32 / u32::MAX as f32) - 0.5
            })
            .collect();

        let resident = gpu.resident_from_q4_0(&bytes, in_dim, out_dim).unwrap();
        let got = gpu.gemv(&resident, &x).unwrap();
        let want = cpu_gemv(&deq, &x, in_dim, out_dim);
        assert_eq!(got.len(), out_dim);
        let mut max_err = 0.0f32;
        for (g, c) in got.iter().zip(&want) {
            max_err = max_err.max((g - c).abs());
        }
        assert!(max_err < 1e-2, "max err {max_err} too large");
        eprintln!(
            "GPU Q4_0 resident GEMV validated on {} (max err {max_err:.2e})",
            gpu.adapter_name()
        );
    }
}
