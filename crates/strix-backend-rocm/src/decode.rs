//! Full on-device Gemma decode forward on ROCm/HIP.
//!
//! Mirrors the Vulkan `accel.rs`/`ash_decode.rs` forward, but every op is a HIP
//! kernel launched on a single stream — the stream serializes dependent kernels
//! in-order automatically (no explicit barriers), and kernel args (incl.
//! pos-dependent ones) are passed per-launch (no uniform buffers). One sync +
//! logits readback per token. Selected via `STRIX_ROCM=1`.

use std::collections::HashMap;
use std::os::raw::c_void;

use strix_core::accel::{GpuDecodeConfig, WeightAccel};

use crate::ffi::hipFunction_t;
use crate::hip::{Dbuf, HipGpu};

const QK4_0: usize = 32;
const Q4_0_BYTES: usize = 18;
const QK_K: usize = 256;
const Q6_K_BYTES: usize = 210;
const QK8_0: usize = 32;
const Q8_0_BYTES: usize = 34; // f16 d + 32 int8

// --- lightweight per-kernel profiling (gated by STRIX_PROF) ---
use std::cell::RefCell;
use std::sync::OnceLock;
fn prof_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("STRIX_PROF").is_ok())
}
thread_local! {
    static PROF: RefCell<HashMap<String, (f64, u32)>> = RefCell::new(HashMap::new());
}
fn prof_add(name: &str, secs: f64) {
    PROF.with(|p| {
        let mut m = p.borrow_mut();
        let e = m.entry(name.to_string()).or_insert((0.0, 0));
        e.0 += secs;
        e.1 += 1;
    });
}
/// Print accumulated per-kernel time (descending) and reset.
pub fn prof_dump(tag: &str) {
    PROF.with(|p| {
        let mut m = p.borrow_mut();
        if m.is_empty() {
            return;
        }
        let mut v: Vec<_> = m.iter().map(|(k, &(s, c))| (k.clone(), s, c)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let total: f64 = v.iter().map(|x| x.1).sum();
        eprintln!(
            "== STRIX_PROF [{tag}] total kernel-wall {:.3} ms ==",
            total * 1e3
        );
        for (k, s, c) in &v {
            eprintln!(
                "  {:>16}  {:8.3} ms  {:5} calls  {:5.1}%  ({:.3} ms/call)",
                k,
                s * 1e3,
                c,
                s / total * 100.0,
                s / *c as f64 * 1e3
            );
        }
        m.clear();
    });
}

struct ResQ4 {
    scales: Dbuf,
    quants: Dbuf,
    in_dim: usize,
    out_dim: usize,
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        let v = m as f32 * 5.960_464_5e-8; // 2^-24, subnormal
        return if s == 1 { -v } else { v };
    }
    f32::from_bits((s << 31) | ((e + 112) << 23) | (m << 13))
}

struct ResQ6 {
    scales: Dbuf,
    ql: Dbuf,
    qh: Dbuf,
    in_dim: usize,
    out_dim: usize,
}

struct ResQ8 {
    scales: Dbuf, // f32[nb*out_dim] (one d per 32-block)
    quants: Dbuf, // int8[nb*32*out_dim]
    in_dim: usize,
    out_dim: usize,
}

/// A whole MoE layer resident in PLANAR Q8_0 (f32 scales + int8 quants per tensor):
/// aligned char4/coalesced reads for the fused decode path.
struct ResMoe {
    gate_s: Dbuf,
    gate_q: Dbuf,
    up_s: Dbuf,
    up_q: Dbuf,
    down_s: Dbuf,
    down_q: Dbuf,
    gate_s4: Dbuf,
    gate_q4: Dbuf,
    up_s4: Dbuf,
    up_q4: Dbuf,
    down_s4: Dbuf,
    down_q4: Dbuf,
    hidden: usize,
    eff: usize,
}

/// Like pack_q4 but applies a per-128-chunk normalized Walsh-Hadamard along the
/// in-dim of each output row BEFORE int4 quant (incoherence: spreads outliers).
/// `nb` blocks per row (in_dim = nb*32, must be multiple of 4 → 128/chunk).
fn pack_q4_had(scales: &[f32], quants: &[i8], nb: usize) -> (Vec<f32>, Vec<u8>) {
    assert!(nb % 4 == 0);
    let nrow = scales.len() / nb;
    let mut s4 = vec![0.0f32; scales.len()];
    let mut q4 = vec![0u8; scales.len() * 16];
    let inv_sqrt = 1.0f32 / (128.0f32).sqrt();
    let mut buf = vec![0.0f32; 128];
    for r in 0..nrow {
        for ch in 0..(nb / 4) {
            // dequant 128 vals of this chunk
            for j in 0..128 {
                let b = r * nb + ch * 4 + j / 32;
                buf[j] = scales[b] * quants[b * 32 + (j % 32)] as f32;
            }
            // in-place H128 (Walsh, hierarchical), normalized
            let mut step = 1;
            while step < 128 {
                let mut i = 0;
                while i < 128 {
                    for k in 0..step {
                        let a = buf[i + k];
                        let bb = buf[i + k + step];
                        buf[i + k] = a + bb;
                        buf[i + k + step] = a - bb;
                    }
                    i += step * 2;
                }
                step <<= 1;
            }
            for v in buf.iter_mut() {
                *v *= inv_sqrt;
            }
            // requant the 4 int4 blocks of this chunk
            for blk in 0..4 {
                let b = r * nb + ch * 4 + blk;
                let seg = &buf[blk * 32..blk * 32 + 32];
                let amax = seg.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                if amax == 0.0 {
                    s4[b] = 0.0;
                    for j in 0..16 {
                        q4[b * 16 + j] = 0x88;
                    }
                    continue;
                }
                s4[b] = amax / 7.0;
                let qinv = 7.0 / amax;
                for j in 0..16 {
                    let lo = ((seg[2 * j] * qinv).round() as i32).clamp(-7, 7) + 8;
                    let hi = ((seg[2 * j + 1] * qinv).round() as i32).clamp(-7, 7) + 8;
                    q4[b * 16 + j] = (lo as u8 & 0xF) | ((hi as u8 & 0xF) << 4);
                }
            }
        }
    }
    (s4, q4)
}

/// Repack a whole 3D Q8_0 expert tensor into planar scales f32 + quants i8.
/// Repack planar i8+f32scale to nibble-packed int4 (probe): 16 B/32-block + new scale.
fn pack_q4(scales: &[f32], quants: &[i8]) -> (Vec<f32>, Vec<u8>) {
    let nb = scales.len();
    let mut s4 = vec![0.0f32; nb];
    let mut q4 = vec![0u8; nb * 16];
    for b in 0..nb {
        let blk = &quants[b * 32..b * 32 + 32];
        let qmax = blk.iter().map(|&q| (q as i32).abs()).max().unwrap_or(0);
        if qmax == 0 {
            s4[b] = 0.0;
            for j in 0..16 {
                q4[b * 16 + j] = 0x88;
            }
            continue;
        }
        s4[b] = scales[b] * qmax as f32 / 7.0;
        let inv = 7.0 / qmax as f32;
        for j in 0..16 {
            let lo = ((blk[2 * j] as f32 * inv).round() as i32).clamp(-7, 7) + 8;
            let hi = ((blk[2 * j + 1] as f32 * inv).round() as i32).clamp(-7, 7) + 8;
            q4[b * 16 + j] = (lo as u8 & 0xF) | ((hi as u8 & 0xF) << 4);
        }
    }
    (s4, q4)
}

fn planar_q8(bytes: &[u8]) -> (Vec<f32>, Vec<i8>) {
    let nb_total = bytes.len() / 34;
    let mut scales = vec![0.0f32; nb_total];
    let mut quants = vec![0i8; nb_total * 32];
    for b in 0..nb_total {
        let blk = &bytes[b * 34..(b + 1) * 34];
        scales[b] = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for i in 0..32 {
            quants[b * 32 + i] = blk[2 + i] as i8;
        }
    }
    (scales, quants)
}

/// A whole MoE layer in NATIVE Q6_K bytes (210 B/superblock, no inflation).
struct ResMoe6 {
    gate: Dbuf,
    up: Dbuf,
    down: Dbuf,
    hidden: usize,
    eff: usize,
    gate_eb: i64,
    down_eb: i64,
}

/// Per-launch kernel argument buffer (stable storage the launch points into).
struct Args {
    vals: Vec<[u8; 8]>,
}
impl Args {
    fn new() -> Self {
        Self { vals: Vec::new() }
    }
    fn ptr(mut self, p: *mut c_void) -> Self {
        let mut b = [0u8; 8];
        b.copy_from_slice(&(p as u64).to_ne_bytes());
        self.vals.push(b);
        self
    }
    fn i(mut self, v: i32) -> Self {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&v.to_ne_bytes());
        self.vals.push(b);
        self
    }
    fn f(mut self, v: f32) -> Self {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&v.to_ne_bytes());
        self.vals.push(b);
        self
    }
}

struct Scratch {
    h: Dbuf,
    xn: Dbuf,
    q: Dbuf,
    q2: Dbuf,
    k: Dbuf,
    v: Dbuf,
    attn: Dbuf,
    t_hidden: Dbuf,
    gate: Dbuf,
    up: Dbuf,
    logits: Dbuf,
    argmax_out: Dbuf,
    ones: Dbuf,
    // Q8-quantized activation for the dp4a GEMV (reused per matmul input).
    xq_lo: Dbuf,
    xq_hi: Dbuf,
    xq_d: Dbuf,
    xq_sum: Dbuf,
    // Flash-decoding partials (n_heads × n_split).
    attn_part: Dbuf,
    attn_pmax: Dbuf,
    attn_psum: Dbuf,
    // Prefill (batched, up to M_CHUNK tokens): one set of [M_CHUNK × dim] buffers.
    p_h: Dbuf,
    p_xn: Dbuf,
    p_q: Dbuf,
    p_q2: Dbuf,
    p_k: Dbuf,
    p_k2: Dbuf,
    p_v: Dbuf,
    p_v2: Dbuf,
    p_attn: Dbuf,
    p_th: Dbuf,
    p_gate: Dbuf,
    p_up: Dbuf,
    #[cfg_attr(not(feature = "npu"), allow(dead_code))]
    p_act: Dbuf,
    p_xqlo: Dbuf,
    p_xqhi: Dbuf,
    p_xqd: Dbuf,
    p_xqsum: Dbuf,
    p_scores: Dbuf, // GEMM-attention S[n_heads*M_CHUNK*max_seq] (STRIX_GEMM_SDPA)
    p_rowsum: Dbuf, // GEMM-attention per-row softmax sums [n_heads*M_CHUNK]
    k_cache: Vec<Dbuf>,
    v_cache: Vec<Dbuf>,
}

/// Max flash-decoding key-split factor (buffer sizing cap). The actual split is
/// context-ADAPTIVE: ~len/64 (constant ≈64 keys/chunk → steady occupancy at any
/// context), clamped to [8, N_SPLIT_MAX]. Measured: 8→32 was +14% decode @2k,
/// 64 a further +4% @6.5k; short ctx (len≤1024) doesn't split. STRIX_N_SPLIT pins it.
const N_SPLIT_MAX: usize = 64;
/// Max prompt tokens per prefill call (bounds scratch; SDPA scores cap is 2048).
const M_CHUNK: usize = 256;

/// Per-weight `gemv` scratch caps (floats): x input row, y output row. Cover the
/// MoE models — in_dim ≤ ~2304, out_dim up to vocab (~248320). gemv returns None
/// (→ CPU fallback) for any weight exceeding these.
const GEMV_MAX_IN: usize = 16384;
const GEMV_MAX_OUT: usize = 262144;

/// ROCm/HIP decode accelerator.
pub struct RocmWeightAccel {
    gpu: HipGpu,
    funcs: HashMap<&'static str, hipFunction_t>,
    q4: HashMap<String, ResQ4>,
    q6: HashMap<String, ResQ6>,
    q8: HashMap<String, ResQ8>,
    f32w: HashMap<String, Dbuf>,
    cfg: Option<GpuDecodeConfig>,
    scratch: Option<Scratch>,
    /// Persistent per-weight `gemv` scratch (x input / y output), reused across calls
    /// (Dbuf has no free-on-drop, so per-call alloc would leak). Sized to GEMV_MAX_*.
    gemv_x: Dbuf,
    gemv_y: Dbuf,
    /// Fused-MoE residency (by layer) + per-token scratch (ids/wexp/g/u/act/dy/out).
    moe: HashMap<usize, ResMoe>,
    moe6: HashMap<usize, ResMoe6>,
    moe_ids: Dbuf,
    moe_w: Dbuf,
    moe_g: Dbuf,
    moe_u: Dbuf,
    moe_act: Dbuf,
    moe_dy: Dbuf,
    moe_out: Dbuf,
    /// Mellum fused-token scratch: resident hidden state + norm/proj buffers.
    mlm_h: Dbuf,
    mlm_n: Dbuf,
    mlm_q: Dbuf,
    mlm_k: Dbuf,
    mlm_v: Dbuf,
    mlm_attn: Dbuf,
    mlm_rl: Dbuf,
    mlm_kc: Vec<Dbuf>,
    mlm_vc: Vec<Dbuf>,
    mlm_cs: Dbuf,
    mlm_sn: Dbuf,
    mlm_cs2: Dbuf,
    mlm_sn2: Dbuf,
    mlm_pos: Dbuf,
    /// int8-activation scratch (STRIX_INT8): quantized acts + per-32-block scales.
    mlm_xq: Dbuf,
    mlm_arg: Dbuf,
    mlm_xd: Dbuf,
    mlm_int8: bool,
    use_wmma: bool,
    use_q4: bool,
    use_had: bool,
    use_q4head: bool,
    head_q4: Option<(Dbuf, Dbuf)>,
    mlm_graph: Option<*mut std::os::raw::c_void>,
    mlm_seq: usize,
    batch_x: Vec<Dbuf>,
    batch_y: Vec<Dbuf>,
    pf_x: Dbuf,
    pf_a: Dbuf,
    pf_b: Dbuf,
    pf_y: Dbuf,
    pf_dy: Dbuf,
    pf_tab: Dbuf,
    pf_act: Dbuf,
    pf_xq: Dbuf,
    pf_cs: Dbuf,
    pf_h: Dbuf,
    pf_n: Dbuf,
    pf_sn: Dbuf,
    pf_xd: Dbuf,
    /// f16 KV cache (STRIX_F16_KV=1): halves KV memory + decode KV traffic, at a
    /// small prefill cost (the prefill SDPA is occupancy-bound, so the h2f isn't
    /// free there). Default false (f32) — best for prefill-heavy workloads.
    kv_f16: bool,
    /// Flash-decoding key-split factor (STRIX_N_SPLIT, default 8): more splits =
    /// more blocks = better occupancy at long context. Only raised (chunk must fit
    /// the kernel's sc[512] → n_split >= len/512).
    n_split: usize,
    name: String,
    /// When set, `prefill` returns logits for ALL m tokens (m×vocab), not just
    /// the last — the speculative-decoding "verify" path. Reset after each call.
    verify_all: bool,
    /// Greedy fast path: when set, `decode_step` skips softcap + the vocab-wide
    /// logits DtoH, does an on-device argmax, and returns a 1-element vec holding
    /// the winning token id (as f32). Set/reset around `decode_step_argmax`.
    want_argmax: bool,
    /// NPU offload for ffn_up: raw Q4 bytes stashed at upload (per layer),
    /// consumed at configure into the live `npu` state.
    #[cfg(feature = "npu")]
    npu_pending: std::collections::BTreeMap<usize, Vec<u8>>,
    #[cfg(feature = "npu")]
    npu: Option<crate::npu_hybrid::NpuFfn>,
    #[cfg(feature = "npu")]
    npu_down_pending: std::collections::BTreeMap<usize, Vec<u8>>,
    #[cfg(feature = "npu")]
    npu_down: Option<crate::npu_hybrid::NpuFfn>,
    #[cfg(feature = "npu")]
    npu_o_pending: std::collections::BTreeMap<usize, Vec<u8>>,
    #[cfg(feature = "npu")]
    npu_o: Option<crate::npu_hybrid::NpuFfn>,
    #[cfg(feature = "npu")]
    npu_q_pending: std::collections::BTreeMap<usize, Vec<u8>>,
    #[cfg(feature = "npu")]
    npu_q: Option<crate::npu_hybrid::NpuFfn>,
    // Global-layer variants (q_dim=8192): attn_q out_dim=8192, attn_output in_dim=8192.
    #[cfg(feature = "npu")]
    npu_q_g_pending: std::collections::BTreeMap<usize, Vec<u8>>,
    #[cfg(feature = "npu")]
    npu_q_g: Option<crate::npu_hybrid::NpuFfn>,
    #[cfg(feature = "npu")]
    npu_o_g_pending: std::collections::BTreeMap<usize, Vec<u8>>,
    #[cfg(feature = "npu")]
    npu_o_g: Option<crate::npu_hybrid::NpuFfn>,
}

// Raw HIP handles/pointers; the accelerator is driven single-threaded (or
// externally synchronized via the stream), so sharing is sound.
unsafe impl Send for RocmWeightAccel {}
unsafe impl Sync for RocmWeightAccel {}

impl RocmWeightAccel {
    pub fn new() -> Option<Self> {
        let gpu = HipGpu::new().ok()?;
        let name = gpu.adapter_name().to_string();
        let code = match crate::hip::compile(crate::kernels::KERNELS) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[rocm] kernel compile FAILED: {e}");
                return None;
            }
        };
        let module = gpu.load_module(&code).ok()?;
        let mut funcs = HashMap::new();
        for name in [
            "q4_gemv",
            "q4_gemv_dp",
            "q4_gemv_dp2",
            "q4_gemv_dp3",
            "xquant",
            "rmsnorm_xquant",
            "geglu_xquant",
            "q6_gemv",
            "q8_0_gemv",
            "q8_moe_gemv",
            "q8_moe_gemv_rows",
            "q8_moe_gemv_gu",
            "q8_moe_down",
            "q8_qkv_gemv",
            "xquant8",
            "silu_quant",
            "q8i_gemv",
            "q8i_qkv_gemv",
            "q8i_moe_gu",
            "q8i_moe_down",
            "q4i_moe_gu",
            "q4i_moe_down",
            "fht128",
            "q4i_gemv",
            "q6_moe_gemv",
            "q8_gemm_rows",
            "q8_gemm_rows32",
            "xquant8_rows",
            "q8i_gemm_rows32",
            "q8i_gemm_lds",
            "q8w_gemm",
            "q8w_gemm32",
            "q8i_gemm_lds2",
            "q6_gemm_rows",
            "q6_gemm_moe",
            "moe_silu_mul",
            "moe_wsum",
            "gather_xq8",
            "rmsnorm_heads",
            "f32_gemv_rows",
            "rope_rows",
            "kv_append_rows",
            "sdpa_rows",
            "silu_quant_rows",
            "scatter_add_w",
            "zerof",
            "vec_add",
            "rope_tab",
            "kv_append_pos",
            "sdpa_pos",
            "topk_router",
            "f32_gemv",
            "shexp_add",
            "argmax_f32",
            "rmsnorm",
            "addnorm",
            "rope",
            "qkv_post",
            "geglu",
            "softcap",
            "sdpa",
            "sdpa_split",
            "sdpa_combine",
            "copyf",
            "copyf_h",
            "q4_gemm",
            "q4_gemm_w",
            "q4_gemm_w_sk",
            "rope_batch",
            "addnorm_batch",
            "sdpa_prefill",
            "sdpa_prefill_f",
            "sdpa_prefill_wmma",
            "sdpa_qk_wmma",
            "sdpa_softmax_mask",
            "sdpa_pv_wmma",
            "xquant_npu",
            "rescale_npu",
        ] {
            funcs.insert(name, gpu.get_function(module, name).ok()?);
        }
        let gemv_x = gpu.alloc(GEMV_MAX_IN * 4).ok()?;
        let gemv_y = gpu.alloc(GEMV_MAX_OUT * 4).ok()?;
        // Fused-MoE scratch: top-k ≤ 16, eff/hidden ≤ 4096.
        let moe_ids = gpu.alloc(16 * 4).ok()?;
        let moe_w = gpu.alloc(16 * 4).ok()?;
        let moe_g = gpu.alloc(16 * 4096 * 4).ok()?;
        let moe_u = gpu.alloc(16 * 4096 * 4).ok()?;
        let moe_act = gpu.alloc(16 * 4096 * 4).ok()?;
        let moe_dy = gpu.alloc(16 * 4096 * 4).ok()?;
        let moe_out = gpu.alloc(4096 * 4).ok()?;
        let mlm_h = gpu.alloc(4096 * 4).ok()?;
        let mlm_n = gpu.alloc(4096 * 4).ok()?;
        let mlm_q = gpu.alloc(8192 * 4).ok()?;
        let mlm_k = gpu.alloc(1024 * 4).ok()?;
        let mlm_v = gpu.alloc(1024 * 4).ok()?;
        let mlm_attn = gpu.alloc(8192 * 4).ok()?;
        let mlm_rl = gpu.alloc(256 * 4).ok()?;
        let mlm_cs = gpu.alloc(128 * 4).ok()?;
        let mlm_sn = gpu.alloc(128 * 4).ok()?;
        let mlm_cs2 = gpu.alloc(128 * 4).ok()?;
        let mlm_sn2 = gpu.alloc(128 * 4).ok()?;
        let mlm_pos = gpu.alloc(4).ok()?;
        let mlm_xq = gpu.alloc(16 * 4096).ok()?;
        let mlm_arg = gpu.alloc(8).ok()?;
        let mlm_xd = gpu.alloc(2048 * 4).ok()?;
        let pf_x = gpu.alloc(2048 * 4096 * 4).ok()?;
        let pf_a = gpu.alloc(2048 * 1024 * 4).ok()?;
        let pf_b = gpu.alloc(2048 * 1024 * 4).ok()?;
        let pf_y = gpu.alloc(256 * 8192 * 4).ok()?;
        let pf_dy = gpu.alloc(2048 * 4096 * 4).ok()?;
        let pf_tab = gpu.alloc(64 * 1024).ok()?;
        let pf_act = gpu.alloc(2048 * 1024 * 4).ok()?;
        let pf_xq = gpu.alloc(2048 * 4096).ok()?;
        let pf_cs = gpu.alloc(512 * 64 * 4).ok()?;
        let pf_h = gpu.alloc(512 * 2304 * 4).ok()?;
        let pf_n = gpu.alloc(512 * 2304 * 4).ok()?;
        let pf_sn = gpu.alloc(512 * 64 * 4).ok()?;
        let pf_xd = gpu.alloc(2048 * 128 * 4).ok()?;
        let mut batch_x = Vec::new();
        let mut batch_y = Vec::new();
        for _ in 0..4 {
            batch_x.push(gpu.alloc(GEMV_MAX_IN * 4).ok()?);
            batch_y.push(gpu.alloc(16384 * 4).ok()?);
        }
        Some(Self {
            gpu,
            funcs,
            q4: HashMap::new(),
            q6: HashMap::new(),
            q8: HashMap::new(),
            f32w: HashMap::new(),
            cfg: None,
            scratch: None,
            gemv_x,
            gemv_y,
            moe: HashMap::new(),
            moe6: HashMap::new(),
            moe_ids,
            moe_w,
            moe_g,
            moe_u,
            moe_act,
            moe_dy,
            moe_out,
            mlm_h,
            mlm_n,
            mlm_q,
            mlm_k,
            mlm_v,
            mlm_attn,
            mlm_rl,
            mlm_kc: Vec::new(),
            mlm_vc: Vec::new(),
            mlm_cs,
            mlm_sn,
            mlm_cs2,
            mlm_sn2,
            mlm_pos,
            mlm_xq,
            mlm_arg,
            mlm_xd,
            mlm_int8: std::env::var("STRIX_NO_INT8").is_err(),
            use_wmma: std::env::var("STRIX_NO_WMMA").is_err(),
            use_q4: std::env::var("STRIX_Q4PROBE").is_ok() || std::env::var("STRIX_HAD").is_ok(),
            use_had: std::env::var("STRIX_HAD").is_ok(),
            use_q4head: std::env::var("STRIX_Q4HEAD").is_ok(),
            head_q4: None,
            mlm_graph: None,
            mlm_seq: 0,
            batch_x,
            batch_y,
            pf_x,
            pf_a,
            pf_b,
            pf_y,
            pf_dy,
            pf_tab,
            pf_act,
            pf_xq,
            pf_cs,
            pf_h,
            pf_n,
            pf_sn,
            pf_xd,
            kv_f16: std::env::var("STRIX_F16_KV").is_ok(),
            // pinned split if STRIX_N_SPLIT set, else 0 = context-adaptive (~len/64).
            n_split: std::env::var("STRIX_N_SPLIT")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .map(|v| v.clamp(1, N_SPLIT_MAX))
                .unwrap_or(0),
            name,
            verify_all: false,
            want_argmax: false,
            #[cfg(feature = "npu")]
            npu_pending: std::collections::BTreeMap::new(),
            #[cfg(feature = "npu")]
            npu: None,
            #[cfg(feature = "npu")]
            npu_down_pending: std::collections::BTreeMap::new(),
            #[cfg(feature = "npu")]
            npu_down: None,
            #[cfg(feature = "npu")]
            npu_o_pending: std::collections::BTreeMap::new(),
            #[cfg(feature = "npu")]
            npu_o: None,
            #[cfg(feature = "npu")]
            npu_q_pending: std::collections::BTreeMap::new(),
            #[cfg(feature = "npu")]
            npu_q: None,
            #[cfg(feature = "npu")]
            npu_q_g_pending: std::collections::BTreeMap::new(),
            #[cfg(feature = "npu")]
            npu_q_g: None,
            #[cfg(feature = "npu")]
            npu_o_g_pending: std::collections::BTreeMap::new(),
            #[cfg(feature = "npu")]
            npu_o_g: None,
        })
    }

    /// Profiled variant: sync after attn and after MoE; prints per-layer wall once/token.
    fn mlm_layer_prof(&mut self, il: usize, pos: usize, win: usize, topk: usize) -> bool {
        use std::sync::atomic::{AtomicU64, Ordering};
        static T_ATTN: AtomicU64 = AtomicU64::new(0);
        static T_MOE: AtomicU64 = AtomicU64::new(0);
        let t0 = std::time::Instant::now();
        if !self.mlm_attn_part(il, pos, win) {
            return false;
        }
        let _ = self.gpu.sync();
        T_ATTN.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        let rl_dim = 64usize;
        let t1 = std::time::Instant::now();
        self.launch(
            "topk_router",
            1,
            32,
            0,
            Args::new()
                .ptr(self.mlm_rl.ptr)
                .i(rl_dim as i32)
                .i(topk as i32)
                .ptr(self.moe_ids.ptr)
                .ptr(self.moe_w.ptr),
        );
        if std::env::var("STRIX_RT_DEBUG").is_ok() && il < 3 {
            let _ = self.gpu.sync();
            let rl = self.mlm_rl.download::<f32>(rl_dim).unwrap_or_default();
            let ids = self.moe_ids.download::<i32>(topk).unwrap_or_default();
            let w = self.moe_w.download::<f32>(topk).unwrap_or_default();
            let mut idx: Vec<usize> = (0..rl_dim).collect();
            let mx = rl.iter().cloned().fold(f32::MIN, f32::max);
            let probs: Vec<f32> = rl.iter().map(|&l| (l - mx).exp()).collect();
            let sum: f32 = probs.iter().sum();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            eprintln!(
                "[rt l{il}] gpu_ids={ids:?} w0={:.3} cpu_top={:?} p0={:.3}",
                w.first().copied().unwrap_or(0.0),
                &idx[..topk.min(8)],
                probs[idx[0]] / sum
            );
        }
        let Some(m) = self.moe.get(&il) else {
            return false;
        };
        self.moe_launches(m, topk, self.mlm_n.ptr);
        self.launch(
            "vec_add",
            m.hidden.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_h.ptr)
                .ptr(self.moe_out.ptr)
                .i(m.hidden as i32),
        );
        let _ = self.gpu.sync();
        T_MOE.fetch_add(t1.elapsed().as_micros() as u64, Ordering::Relaxed);
        if il == 27 {
            println!(
                "[mlm prof] attn {:.1}ms moe {:.1}ms (cum)",
                T_ATTN.load(Ordering::Relaxed) as f64 / 1e3,
                T_MOE.load(Ordering::Relaxed) as f64 / 1e3
            );
        }
        true
    }

    /// Pos-indirect attn part (graph-capturable: pos read from device buffer).
    fn mlm_attn_part_pos(&self, il: usize, win: usize, full_tab: bool) -> bool {
        let (
            Some(nw),
            Some(wq),
            Some(wk),
            Some(wv),
            Some(wo),
            Some(fnw),
            Some(wgi),
            Some(qn),
            Some(kn),
        ) = (
            self.f32w.get(&format!("blk.{il}.attn_norm.weight")),
            self.q8.get(&format!("blk.{il}.attn_q.weight")),
            self.q8.get(&format!("blk.{il}.attn_k.weight")),
            self.q8.get(&format!("blk.{il}.attn_v.weight")),
            self.q8.get(&format!("blk.{il}.attn_output.weight")),
            self.f32w.get(&format!("blk.{il}.ffn_norm.weight")),
            self.f32w.get(&format!("blk.{il}.ffn_gate_inp.weight")),
            self.f32w.get(&format!("blk.{il}.attn_q_norm.weight")),
            self.f32w.get(&format!("blk.{il}.attn_k_norm.weight")),
        )
        else {
            return false;
        };
        let hidden = wq.in_dim;
        let q_dim = wq.out_dim;
        let kv_dim = wk.out_dim;
        let hd = 128usize;
        let nh = q_dim / hd;
        let nkv = kv_dim / hd;
        if il >= self.mlm_kc.len() {
            return false;
        }
        let (cs, sn) = if full_tab {
            (self.mlm_cs2.ptr, self.mlm_sn2.ptr)
        } else {
            (self.mlm_cs.ptr, self.mlm_sn.ptr)
        };
        self.norm_launch(self.mlm_h.ptr, nw.ptr, self.mlm_n.ptr, hidden);
        if self.mlm_int8 {
            self.launch(
                "xquant8",
                (hidden / 32) as u32,
                32,
                0,
                Args::new()
                    .ptr(self.mlm_n.ptr)
                    .ptr(self.mlm_xq.ptr)
                    .ptr(self.mlm_xd.ptr)
                    .i((hidden / 32) as i32),
            );
            self.launch(
                "q8i_qkv_gemv",
                (q_dim + 2 * kv_dim) as u32,
                32,
                0,
                Args::new()
                    .ptr(wq.scales.ptr)
                    .ptr(wq.quants.ptr)
                    .ptr(wk.scales.ptr)
                    .ptr(wk.quants.ptr)
                    .ptr(wv.scales.ptr)
                    .ptr(wv.quants.ptr)
                    .ptr(self.mlm_xd.ptr)
                    .ptr(self.mlm_xq.ptr)
                    .ptr(self.mlm_q.ptr)
                    .ptr(self.mlm_k.ptr)
                    .ptr(self.mlm_v.ptr)
                    .i(hidden as i32)
                    .i(q_dim as i32)
                    .i(kv_dim as i32),
            );
        } else {
            self.launch(
                "q8_qkv_gemv",
                (q_dim + 2 * kv_dim) as u32,
                32,
                0,
                Args::new()
                    .ptr(wq.scales.ptr)
                    .ptr(wq.quants.ptr)
                    .ptr(wk.scales.ptr)
                    .ptr(wk.quants.ptr)
                    .ptr(wv.scales.ptr)
                    .ptr(wv.quants.ptr)
                    .ptr(self.mlm_n.ptr)
                    .ptr(self.mlm_q.ptr)
                    .ptr(self.mlm_k.ptr)
                    .ptr(self.mlm_v.ptr)
                    .i(hidden as i32)
                    .i(q_dim as i32)
                    .i(kv_dim as i32),
            );
        }
        self.launch(
            "rmsnorm",
            nh as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_q.ptr)
                .ptr(qn.ptr)
                .ptr(self.mlm_q.ptr)
                .i(hd as i32)
                .i(1)
                .f(1e-6),
        );
        self.launch(
            "rmsnorm",
            nkv as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_k.ptr)
                .ptr(kn.ptr)
                .ptr(self.mlm_k.ptr)
                .i(hd as i32)
                .i(1)
                .f(1e-6),
        );
        let half = hd / 2;
        self.launch(
            "rope_tab",
            ((nh * half).div_ceil(64)) as u32,
            64,
            0,
            Args::new()
                .ptr(self.mlm_q.ptr)
                .ptr(cs)
                .ptr(sn)
                .i(hd as i32)
                .i(nh as i32),
        );
        self.launch(
            "rope_tab",
            ((nkv * half).div_ceil(64)) as u32,
            64,
            0,
            Args::new()
                .ptr(self.mlm_k.ptr)
                .ptr(cs)
                .ptr(sn)
                .i(hd as i32)
                .i(nkv as i32),
        );
        for (src, cache) in [
            (self.mlm_k.ptr, &self.mlm_kc[il]),
            (self.mlm_v.ptr, &self.mlm_vc[il]),
        ] {
            self.launch(
                "kv_append_pos",
                kv_dim.div_ceil(256) as u32,
                256,
                0,
                Args::new()
                    .ptr(src)
                    .ptr(cache.ptr)
                    .ptr(self.mlm_pos.ptr)
                    .i(kv_dim as i32),
            );
        }
        let scale = 1.0 / (hd as f32).sqrt();
        self.launch(
            "sdpa_pos",
            nh as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_q.ptr)
                .ptr(self.mlm_kc[il].ptr)
                .ptr(self.mlm_vc[il].ptr)
                .ptr(self.mlm_attn.ptr)
                .i(hd as i32)
                .ptr(self.mlm_pos.ptr)
                .i(win as i32)
                .i((nh / nkv) as i32)
                .i(nkv as i32)
                .f(scale),
        );
        if self.mlm_int8 {
            let onb = wo.in_dim / 32;
            let xqo = unsafe { (self.mlm_xq.ptr as *mut u8).add(8192) as *mut c_void };
            let xdo = unsafe { (self.mlm_xd.ptr as *mut f32).add(512) as *mut c_void };
            self.launch(
                "xquant8",
                onb as u32,
                32,
                0,
                Args::new()
                    .ptr(self.mlm_attn.ptr)
                    .ptr(xqo)
                    .ptr(xdo)
                    .i(onb as i32),
            );
            self.launch(
                "q8i_gemv",
                wo.out_dim as u32,
                32,
                0,
                Args::new()
                    .ptr(wo.scales.ptr)
                    .ptr(wo.quants.ptr)
                    .ptr(xdo)
                    .ptr(xqo)
                    .ptr(self.moe_out.ptr)
                    .i(wo.in_dim as i32)
                    .i(wo.out_dim as i32),
            );
        } else {
            self.q8_launch(wo, self.mlm_attn.ptr, self.moe_out.ptr);
        }
        self.launch(
            "vec_add",
            hidden.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_h.ptr)
                .ptr(self.moe_out.ptr)
                .i(hidden as i32),
        );
        self.norm_launch(self.mlm_h.ptr, fnw.ptr, self.mlm_n.ptr, hidden);
        if self.mlm_int8 {
            self.launch(
                "xquant8",
                (hidden / 32) as u32,
                32,
                0,
                Args::new()
                    .ptr(self.mlm_n.ptr)
                    .ptr(self.mlm_xq.ptr)
                    .ptr(self.mlm_xd.ptr)
                    .i((hidden / 32) as i32),
            );
        }
        self.launch(
            "f32_gemv",
            64,
            32,
            0,
            Args::new()
                .ptr(wgi.ptr)
                .ptr(self.mlm_n.ptr)
                .ptr(self.mlm_rl.ptr)
                .i(hidden as i32)
                .i(64),
        );
        true
    }

    /// Queue one full pos-indirect layer (attn + GPU router + MoE), no sync.
    fn mlm_layer_pos(&self, il: usize, win: usize, full_tab: bool, topk: usize) -> bool {
        if !self.mlm_attn_part_pos(il, win, full_tab) {
            return false;
        }
        self.launch(
            "topk_router",
            1,
            32,
            0,
            Args::new()
                .ptr(self.mlm_rl.ptr)
                .i(64)
                .i(topk as i32)
                .ptr(self.moe_ids.ptr)
                .ptr(self.moe_w.ptr),
        );
        let Some(m) = self.moe.get(&il) else {
            return false;
        };
        self.moe_launches(m, topk, self.mlm_n.ptr);
        self.launch(
            "vec_add",
            m.hidden.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_h.ptr)
                .ptr(self.moe_out.ptr)
                .i(m.hidden as i32),
        );
        true
    }

    /// Attn + ffn-norm + router-launch portion of a fused Mellum layer (no sync).
    fn mlm_attn_part(&mut self, il: usize, pos: usize, win: usize) -> bool {
        let (
            Some(nw),
            Some(wq),
            Some(wk),
            Some(wv),
            Some(wo),
            Some(fnw),
            Some(wgi),
            Some(qn),
            Some(kn),
        ) = (
            self.f32w.get(&format!("blk.{il}.attn_norm.weight")),
            self.q8.get(&format!("blk.{il}.attn_q.weight")),
            self.q8.get(&format!("blk.{il}.attn_k.weight")),
            self.q8.get(&format!("blk.{il}.attn_v.weight")),
            self.q8.get(&format!("blk.{il}.attn_output.weight")),
            self.f32w.get(&format!("blk.{il}.ffn_norm.weight")),
            self.f32w.get(&format!("blk.{il}.ffn_gate_inp.weight")),
            self.f32w.get(&format!("blk.{il}.attn_q_norm.weight")),
            self.f32w.get(&format!("blk.{il}.attn_k_norm.weight")),
        )
        else {
            {
                return false;
            }
        };
        let hidden = wq.in_dim;
        let q_dim = wq.out_dim;
        let kv_dim = wk.out_dim;
        let hd = 128usize;
        let nh = q_dim / hd;
        let nkv = kv_dim / hd;
        if il >= self.mlm_kc.len() || (pos + 1) > self.mlm_seq {
            {
                return false;
            }
        }
        self.norm_launch(self.mlm_h.ptr, nw.ptr, self.mlm_n.ptr, hidden);
        if self.mlm_int8 {
            self.launch(
                "xquant8",
                (hidden / 32) as u32,
                32,
                0,
                Args::new()
                    .ptr(self.mlm_n.ptr)
                    .ptr(self.mlm_xq.ptr)
                    .ptr(self.mlm_xd.ptr)
                    .i((hidden / 32) as i32),
            );
            self.launch(
                "q8i_qkv_gemv",
                (q_dim + 2 * kv_dim) as u32,
                32,
                0,
                Args::new()
                    .ptr(wq.scales.ptr)
                    .ptr(wq.quants.ptr)
                    .ptr(wk.scales.ptr)
                    .ptr(wk.quants.ptr)
                    .ptr(wv.scales.ptr)
                    .ptr(wv.quants.ptr)
                    .ptr(self.mlm_xd.ptr)
                    .ptr(self.mlm_xq.ptr)
                    .ptr(self.mlm_q.ptr)
                    .ptr(self.mlm_k.ptr)
                    .ptr(self.mlm_v.ptr)
                    .i(hidden as i32)
                    .i(q_dim as i32)
                    .i(kv_dim as i32),
            );
        } else {
            self.launch(
                "q8_qkv_gemv",
                (q_dim + 2 * kv_dim) as u32,
                32,
                0,
                Args::new()
                    .ptr(wq.scales.ptr)
                    .ptr(wq.quants.ptr)
                    .ptr(wk.scales.ptr)
                    .ptr(wk.quants.ptr)
                    .ptr(wv.scales.ptr)
                    .ptr(wv.quants.ptr)
                    .ptr(self.mlm_n.ptr)
                    .ptr(self.mlm_q.ptr)
                    .ptr(self.mlm_k.ptr)
                    .ptr(self.mlm_v.ptr)
                    .i(hidden as i32)
                    .i(q_dim as i32)
                    .i(kv_dim as i32),
            );
        }
        self.launch(
            "rmsnorm",
            nh as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_q.ptr)
                .ptr(qn.ptr)
                .ptr(self.mlm_q.ptr)
                .i(hd as i32)
                .i(1)
                .f(1e-6),
        );
        self.launch(
            "rmsnorm",
            nkv as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_k.ptr)
                .ptr(kn.ptr)
                .ptr(self.mlm_k.ptr)
                .i(hd as i32)
                .i(1)
                .f(1e-6),
        );
        let half = hd / 2;
        self.launch(
            "rope_tab",
            ((nh * half).div_ceil(64)) as u32,
            64,
            0,
            Args::new()
                .ptr(self.mlm_q.ptr)
                .ptr(self.mlm_cs.ptr)
                .ptr(self.mlm_sn.ptr)
                .i(hd as i32)
                .i(nh as i32),
        );
        self.launch(
            "rope_tab",
            ((nkv * half).div_ceil(64)) as u32,
            64,
            0,
            Args::new()
                .ptr(self.mlm_k.ptr)
                .ptr(self.mlm_cs.ptr)
                .ptr(self.mlm_sn.ptr)
                .i(hd as i32)
                .i(nkv as i32),
        );
        let koff = pos * kv_dim;
        for (src, cache) in [
            (self.mlm_q.ptr, None),
            (self.mlm_k.ptr, Some(&self.mlm_kc[il])),
            (self.mlm_v.ptr, Some(&self.mlm_vc[il])),
        ] {
            if let Some(cache) = cache {
                self.launch(
                    "copyf",
                    (kv_dim).div_ceil(256) as u32,
                    256,
                    0,
                    Args::new()
                        .ptr(src)
                        .ptr(unsafe { (cache.ptr as *mut f32).add(koff) } as *mut c_void)
                        .i(kv_dim as i32),
                );
            }
        }
        let len = pos + 1;
        let win_start = if win > 0 && len > win { len - win } else { 0 };
        let wlen = len - win_start;
        if wlen > 2048 {
            {
                return false;
            }
        }
        let kbase =
            unsafe { (self.mlm_kc[il].ptr as *mut f32).add(win_start * kv_dim) } as *mut c_void;
        let vbase =
            unsafe { (self.mlm_vc[il].ptr as *mut f32).add(win_start * kv_dim) } as *mut c_void;
        let scale = 1.0 / (hd as f32).sqrt();
        self.launch(
            "sdpa",
            nh as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_q.ptr)
                .ptr(kbase)
                .ptr(vbase)
                .ptr(self.mlm_attn.ptr)
                .i(hd as i32)
                .i(wlen as i32)
                .i((nh / nkv) as i32)
                .i(nkv as i32)
                .f(scale)
                .i(0),
        );
        if self.mlm_int8 {
            let onb = wo.in_dim / 32;
            let xqo = unsafe { (self.mlm_xq.ptr as *mut u8).add(8192) as *mut c_void };
            let xdo = unsafe { (self.mlm_xd.ptr as *mut f32).add(512) as *mut c_void };
            self.launch(
                "xquant8",
                onb as u32,
                32,
                0,
                Args::new()
                    .ptr(self.mlm_attn.ptr)
                    .ptr(xqo)
                    .ptr(xdo)
                    .i(onb as i32),
            );
            self.launch(
                "q8i_gemv",
                wo.out_dim as u32,
                32,
                0,
                Args::new()
                    .ptr(wo.scales.ptr)
                    .ptr(wo.quants.ptr)
                    .ptr(xdo)
                    .ptr(xqo)
                    .ptr(self.moe_out.ptr)
                    .i(wo.in_dim as i32)
                    .i(wo.out_dim as i32),
            );
        } else {
            self.q8_launch(wo, self.mlm_attn.ptr, self.moe_out.ptr);
        }
        self.launch(
            "vec_add",
            hidden.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_h.ptr)
                .ptr(self.moe_out.ptr)
                .i(hidden as i32),
        );
        self.norm_launch(self.mlm_h.ptr, fnw.ptr, self.mlm_n.ptr, hidden);
        if self.mlm_int8 {
            self.launch(
                "xquant8",
                (hidden / 32) as u32,
                32,
                0,
                Args::new()
                    .ptr(self.mlm_n.ptr)
                    .ptr(self.mlm_xq.ptr)
                    .ptr(self.mlm_xd.ptr)
                    .i((hidden / 32) as i32),
            );
        }
        // router is F32 in the GGUF — tiny f32 GEMV (out = n_expert)
        self.launch(
            "f32_gemv",
            64,
            32,
            0,
            Args::new()
                .ptr(wgi.ptr)
                .ptr(self.mlm_n.ptr)
                .ptr(self.mlm_rl.ptr)
                .i(hidden as i32)
                .i(64),
        );
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_prefill_inner(
        &mut self,
        layer: usize,
        m: usize,
        base: usize,
        win: usize,
        cs: &[f32],
        sn: &[f32],
        src: *mut c_void,
    ) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let wq = self.q8.get(&format!("blk.{layer}.attn_q.weight"))?;
        let wk = self.q8.get(&format!("blk.{layer}.attn_k.weight"))?;
        let wv = self.q8.get(&format!("blk.{layer}.attn_v.weight"))?;
        let wo = self.q8.get(&format!("blk.{layer}.attn_output.weight"))?;
        let qn = self.f32w.get(&format!("blk.{layer}.attn_q_norm.weight"))?;
        let kn = self.f32w.get(&format!("blk.{layer}.attn_k_norm.weight"))?;
        let (hidden, q_dim, kv_dim) = (wq.in_dim, wq.out_dim, wk.out_dim);
        let hd = 128usize;
        let (nh, nkv) = (q_dim / hd, kv_dim / hd);
        if layer >= self.mlm_kc.len() || base + m > self.mlm_seq || m > 512 {
            return None;
        }
        self.gpu.upload_at(&self.pf_cs, 0, cs).ok()?;
        self.gpu.upload_at(&self.pf_sn, 0, sn).ok()?;
        let (xq, xd) = self.pf_quant(src, m, hidden);
        let qb = self.pf_y.ptr;
        let kb = self.pf_a.ptr;
        let vb = self.pf_b.ptr;
        for (w, y, od) in [(&wq, qb, q_dim), (&wk, kb, kv_dim), (&wv, vb, kv_dim)] {
            self.q8i_gemm(
                od as u32,
                m as u32,
                Args::new()
                    .ptr(w.scales.ptr)
                    .ptr(w.quants.ptr)
                    .ptr(xq)
                    .ptr(xd)
                    .ptr(y)
                    .i(hidden as i32)
                    .i(od as i32)
                    .i(m as i32),
            );
        }
        for (y, n, w) in [(qb, nh, qn.ptr), (kb, nkv, kn.ptr)] {
            self.launch2(
                "rmsnorm_heads",
                n as u32,
                m as u32,
                256,
                0,
                Args::new()
                    .ptr(y)
                    .ptr(w)
                    .ptr(y)
                    .i(hd as i32)
                    .i((n * hd) as i32)
                    .i(m as i32),
            );
        }
        let half = hd / 2;
        for (y, n) in [(qb, nh), (kb, nkv)] {
            self.launch2(
                "rope_rows",
                ((n * half).div_ceil(64)) as u32,
                m as u32,
                64,
                0,
                Args::new()
                    .ptr(y)
                    .ptr(self.pf_cs.ptr)
                    .ptr(self.pf_sn.ptr)
                    .i(hd as i32)
                    .i(n as i32)
                    .i(m as i32),
            );
        }
        for (src, cache) in [(kb, &self.mlm_kc[layer]), (vb, &self.mlm_vc[layer])] {
            self.launch2(
                "kv_append_rows",
                kv_dim.div_ceil(256) as u32,
                m as u32,
                256,
                0,
                Args::new()
                    .ptr(src)
                    .ptr(cache.ptr)
                    .i(base as i32)
                    .i(kv_dim as i32)
                    .i(m as i32),
            );
        }
        let attn = self.pf_dy.ptr;
        let scale = 1.0 / (hd as f32).sqrt();
        self.launch2(
            "sdpa_rows",
            nh as u32,
            m as u32,
            256,
            0,
            Args::new()
                .ptr(qb)
                .ptr(self.mlm_kc[layer].ptr)
                .ptr(self.mlm_vc[layer].ptr)
                .ptr(attn)
                .i(hd as i32)
                .i(base as i32)
                .i(win as i32)
                .i((nh / nkv) as i32)
                .i(nkv as i32)
                .f(scale)
                .i(m as i32)
                .i(q_dim as i32),
        );
        let (aq, ad) = self.pf_quant(attn, m, q_dim);
        let outp = unsafe { (self.pf_y.ptr as *mut f32).add(m * q_dim) } as *mut c_void;
        self.q8i_gemm(
            hidden as u32,
            m as u32,
            Args::new()
                .ptr(wo.scales.ptr)
                .ptr(wo.quants.ptr)
                .ptr(aq)
                .ptr(ad)
                .ptr(outp)
                .i(q_dim as i32)
                .i(hidden as i32)
                .i(m as i32),
        );
        let resident = src == self.pf_n.ptr;
        if resident {
            self.launch(
                "vec_add",
                (m * hidden).div_ceil(256) as u32,
                256,
                0,
                Args::new()
                    .ptr(self.pf_h.ptr)
                    .ptr(outp)
                    .i((m * hidden) as i32),
            );
        }
        self.gpu.sync().ok()?;
        let out = if resident {
            Vec::new()
        } else {
            self.pf_y
                .download_at::<f32>(m * q_dim * 4, m * hidden)
                .ok()?
        };
        let kk = self.pf_a.download::<f32>(m * kv_dim).ok()?;
        let vv = self.pf_b.download::<f32>(m * kv_dim).ok()?;
        Some((out, kk, vv))
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_layer_dev_inner(
        &self,
        layer: usize,
        m: usize,
        plan: &[(usize, usize, usize)],
        slot_tok: &[i32],
        wslot: &[f32],
        src: *mut c_void,
        hout: *mut c_void,
    ) -> Option<()> {
        let mo = self.moe.get(&layer)?;
        let (hidden, eff) = (mo.hidden, mo.eff);
        let nb = hidden / 32;
        let slots = slot_tok.len();
        if slots == 0 || slots > 4096 || m > 512 {
            return None;
        }
        self.gpu.upload_at(&self.pf_tab, 0, slot_tok).ok()?;
        self.gpu.upload_at(&self.pf_tab, slots * 4, wslot).ok()?;
        let wptr = unsafe { (self.pf_tab.ptr as *mut u8).add(slots * 4) } as *mut c_void;
        let (xq, xd) = self.pf_quant(src, m, hidden);
        // gathered acts region (after token region): tokens use [0, m), slots at [m, m+slots)
        let gq = unsafe { (self.pf_xq.ptr as *mut u8).add(m * hidden) } as *mut c_void;
        let gd = unsafe { (self.pf_xd.ptr as *mut f32).add(m * nb) } as *mut c_void;
        self.launch2(
            "gather_xq8",
            nb as u32,
            slots as u32,
            32,
            0,
            Args::new()
                .ptr(xq)
                .ptr(xd)
                .ptr(self.pf_tab.ptr)
                .ptr(gq)
                .ptr(gd)
                .i(nb as i32)
                .i(slots as i32),
        );
        let enb = eff / 32;
        for &(e, off, me) in plan {
            let eoff = e * eff * nb;
            let sq = unsafe { (gq as *mut u8).add(off * hidden) } as *mut c_void;
            let sd = unsafe { (gd as *mut f32).add(off * nb) } as *mut c_void;
            for (sb, qb2, y) in [
                (&mo.gate_s, &mo.gate_q, self.pf_a.ptr),
                (&mo.up_s, &mo.up_q, self.pf_b.ptr),
            ] {
                let sp = unsafe { (sb.ptr as *mut f32).add(eoff) } as *mut c_void;
                let qp = unsafe { (qb2.ptr as *mut i8).add(eoff * 32) } as *mut c_void;
                let yp = unsafe { (y as *mut f32).add(off * eff) } as *mut c_void;
                self.q8i_gemm(
                    eff as u32,
                    me as u32,
                    Args::new()
                        .ptr(sp)
                        .ptr(qp)
                        .ptr(sq)
                        .ptr(sd)
                        .ptr(yp)
                        .i(hidden as i32)
                        .i(eff as i32)
                        .i(me as i32),
                );
            }
        }
        // silu+quantize all slot activations, then down per expert
        let aq = unsafe { (self.pf_xq.ptr as *mut u8).add((m + slots) * hidden) } as *mut c_void;
        let ad = unsafe { (self.pf_xd.ptr as *mut f32).add((m + slots) * nb) } as *mut c_void;
        let nblk = slots * enb;
        self.launch(
            "silu_quant_rows",
            nblk as u32,
            32,
            0,
            Args::new()
                .ptr(self.pf_a.ptr)
                .ptr(self.pf_b.ptr)
                .ptr(aq)
                .ptr(ad)
                .i(nblk as i32),
        );
        for &(e, off, me) in plan {
            let deoff = e * hidden * enb;
            let sp = unsafe { (mo.down_s.ptr as *mut f32).add(deoff) } as *mut c_void;
            let qp = unsafe { (mo.down_q.ptr as *mut i8).add(deoff * 32) } as *mut c_void;
            let sq = unsafe { (aq as *mut u8).add(off * eff) } as *mut c_void;
            let sd = unsafe { (ad as *mut f32).add(off * enb) } as *mut c_void;
            let yp = unsafe { (self.pf_dy.ptr as *mut f32).add(off * hidden) } as *mut c_void;
            self.q8i_gemm(
                hidden as u32,
                me as u32,
                Args::new()
                    .ptr(sp)
                    .ptr(qp)
                    .ptr(sq)
                    .ptr(sd)
                    .ptr(yp)
                    .i(eff as i32)
                    .i(hidden as i32)
                    .i(me as i32),
            );
        }
        let dst = if hout.is_null() {
            let n_out = m * hidden;
            self.launch(
                "zerof",
                n_out.div_ceil(256) as u32,
                256,
                0,
                Args::new().ptr(self.pf_y.ptr).i(n_out as i32),
            );
            self.pf_y.ptr
        } else {
            hout // residual: scatter-add straight into resident h
        };
        self.launch2(
            "scatter_add_w",
            hidden.div_ceil(256) as u32,
            slots as u32,
            256,
            0,
            Args::new()
                .ptr(self.pf_dy.ptr)
                .ptr(self.pf_tab.ptr)
                .ptr(wptr)
                .ptr(dst)
                .i(hidden as i32)
                .i(slots as i32),
        );
        Some(())
    }

    /// int8 GEMM launch: WMMA tile (default) or LDS2 fallback (STRIX_NO_WMMA).
    fn q8i_gemm(&self, out: u32, m: u32, args: Args) {
        if self.use_wmma {
            if m >= 24 {
                self.launch2(
                    "q8w_gemm32",
                    out.div_ceil(128),
                    m.div_ceil(32),
                    256,
                    0,
                    args,
                );
            } else {
                self.launch2("q8w_gemm", out.div_ceil(128), m.div_ceil(16), 256, 0, args);
            }
        } else {
            self.launch2(
                "q8i_gemm_lds2",
                out.div_ceil(16),
                m.div_ceil(32),
                256,
                0,
                args,
            );
        }
    }

    fn launch(&self, name: &str, grid: u32, block: u32, shared: u32, args: Args) {
        self.launch2(name, grid, 1, block, shared, args);
    }

    /// No-sync Q8_0 dense GEMV on a resident weight (stream-ordered).
    fn q8_launch(&self, e: &ResQ8, x: *mut c_void, y: *mut c_void) {
        self.launch(
            "q8_0_gemv",
            e.out_dim as u32,
            32,
            0,
            Args::new()
                .ptr(e.scales.ptr)
                .ptr(e.quants.ptr)
                .ptr(x)
                .ptr(y)
                .i(e.in_dim as i32)
                .i(e.out_dim as i32),
        );
    }

    /// Quantize m rows (in_dim each) to i8 + per-32 scales in pf_xq/pf_xd.
    fn pf_quant(&self, x: *mut c_void, m: usize, in_dim: usize) -> (*mut c_void, *mut c_void) {
        let nb = in_dim / 32;
        self.launch2(
            "xquant8_rows",
            nb as u32,
            m as u32,
            32,
            0,
            Args::new()
                .ptr(x)
                .ptr(self.pf_xq.ptr)
                .ptr(self.pf_xd.ptr)
                .i(nb as i32)
                .i(m as i32),
        );
        (self.pf_xq.ptr, self.pf_xd.ptr)
    }

    /// No-sync RMSNorm on resident buffers (single row).
    fn norm_launch(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void, dim: usize) {
        self.launch(
            "rmsnorm",
            1,
            256,
            0,
            Args::new().ptr(x).ptr(w).ptr(y).i(dim as i32).i(1).f(1e-6),
        );
    }

    /// Queue shared-expert launches into moe_out (no sync). x already in gemv_x.
    fn queue_shexp(&self, layer: usize, hidden: usize, sgate: f32) -> bool {
        let (Some(g), Some(u), Some(d)) = (
            self.q8.get(&format!("blk.{layer}.ffn_gate_shexp.weight")),
            self.q8.get(&format!("blk.{layer}.ffn_up_shexp.weight")),
            self.q8.get(&format!("blk.{layer}.ffn_down_shexp.weight")),
        ) else {
            return false;
        };
        let sff = g.out_dim;
        self.q8_launch(g, self.gemv_x.ptr, self.batch_y[0].ptr);
        self.q8_launch(u, self.gemv_x.ptr, self.batch_y[1].ptr);
        self.launch(
            "moe_silu_mul",
            sff.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.batch_y[0].ptr)
                .ptr(self.batch_y[1].ptr)
                .ptr(self.moe_act.ptr)
                .i(sff as i32),
        );
        self.q8_launch(d, self.moe_act.ptr, self.batch_y[2].ptr);
        self.launch(
            "shexp_add",
            hidden.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.moe_out.ptr)
                .ptr(self.batch_y[2].ptr)
                .f(sgate)
                .i(hidden as i32),
        );
        true
    }

    /// No-sync fused Q6_K MoE launches (native bytes) — same scratch as Q8.
    fn moe6_launches(&self, m: &ResMoe6, k: usize, x: *mut c_void) {
        let geb = m.gate_eb as usize as *mut c_void;
        let deb = m.down_eb as usize as *mut c_void;
        for (w, y) in [(&m.gate, self.moe_g.ptr), (&m.up, self.moe_u.ptr)] {
            self.launch2(
                "q6_moe_gemv",
                m.eff.div_ceil(8) as u32,
                k as u32,
                256,
                0,
                Args::new()
                    .ptr(w.ptr)
                    .ptr(geb)
                    .ptr(self.moe_ids.ptr)
                    .ptr(x)
                    .ptr(y)
                    .i(m.hidden as i32)
                    .i(m.eff as i32),
            );
        }
        let n_act = k * m.eff;
        self.launch(
            "moe_silu_mul",
            n_act.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.moe_g.ptr)
                .ptr(self.moe_u.ptr)
                .ptr(self.moe_act.ptr)
                .i(n_act as i32),
        );
        for e in 0..k {
            self.launch2(
                "q6_moe_gemv",
                m.hidden.div_ceil(8) as u32,
                1,
                256,
                0,
                Args::new()
                    .ptr(m.down.ptr)
                    .ptr(deb)
                    .ptr(unsafe { (self.moe_ids.ptr as *mut i32).add(e) } as *mut c_void)
                    .ptr(unsafe { (self.moe_act.ptr as *mut f32).add(e * m.eff) } as *mut c_void)
                    .ptr(unsafe { (self.moe_dy.ptr as *mut f32).add(e * m.hidden) } as *mut c_void)
                    .i(m.eff as i32)
                    .i(m.hidden as i32),
            );
        }
        self.launch(
            "moe_wsum",
            m.hidden.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.moe_dy.ptr)
                .ptr(self.moe_w.ptr)
                .ptr(self.moe_out.ptr)
                .i(m.hidden as i32)
                .i(k as i32),
        );
    }

    /// No-sync fused-MoE launches for layer `layer` over k routed experts
    /// (ids/wexp already uploaded): x → moe_out. Caller syncs.
    fn moe_launches(&self, m: &ResMoe, k: usize, x: *mut c_void) {
        if self.use_had {
            // rotate gu input (hidden) per-128 then re-quantize into mlm_xq/mlm_xd
            self.launch(
                "fht128",
                (m.hidden / 128) as u32,
                128,
                0,
                Args::new().ptr(x).ptr(self.pf_x.ptr).i(m.hidden as i32),
            );
            self.launch(
                "xquant8",
                (m.hidden / 32) as u32,
                32,
                0,
                Args::new()
                    .ptr(self.pf_x.ptr)
                    .ptr(self.mlm_xq.ptr)
                    .ptr(self.mlm_xd.ptr)
                    .i((m.hidden / 32) as i32),
            );
        }
        if self.use_q4 {
            self.launch3(
                "q4i_moe_gu",
                m.eff as u32,
                k as u32,
                2,
                32,
                0,
                Args::new()
                    .ptr(m.gate_s4.ptr)
                    .ptr(m.gate_q4.ptr)
                    .ptr(m.up_s4.ptr)
                    .ptr(m.up_q4.ptr)
                    .ptr(self.moe_ids.ptr)
                    .ptr(self.mlm_xd.ptr)
                    .ptr(self.mlm_xq.ptr)
                    .ptr(self.moe_g.ptr)
                    .ptr(self.moe_u.ptr)
                    .i(m.hidden as i32)
                    .i(m.eff as i32),
            );
        } else if self.mlm_int8 {
            self.launch3(
                "q8i_moe_gu",
                m.eff as u32,
                k as u32,
                2,
                32,
                0,
                Args::new()
                    .ptr(m.gate_s.ptr)
                    .ptr(m.gate_q.ptr)
                    .ptr(m.up_s.ptr)
                    .ptr(m.up_q.ptr)
                    .ptr(self.moe_ids.ptr)
                    .ptr(self.mlm_xd.ptr)
                    .ptr(self.mlm_xq.ptr)
                    .ptr(self.moe_g.ptr)
                    .ptr(self.moe_u.ptr)
                    .i(m.hidden as i32)
                    .i(m.eff as i32),
            );
        } else {
            self.launch3(
                "q8_moe_gemv_gu",
                m.eff as u32,
                k as u32,
                2,
                32,
                0,
                Args::new()
                    .ptr(m.gate_s.ptr)
                    .ptr(m.gate_q.ptr)
                    .ptr(m.up_s.ptr)
                    .ptr(m.up_q.ptr)
                    .ptr(self.moe_ids.ptr)
                    .ptr(x)
                    .ptr(self.moe_g.ptr)
                    .ptr(self.moe_u.ptr)
                    .i(m.hidden as i32)
                    .i(m.eff as i32),
            );
        }
        if self.mlm_int8 {
            let nb_act = k * m.eff / 32;
            let n_act = k * m.eff;
            let xqm = unsafe { (self.mlm_xq.ptr as *mut u8).add(12288) as *mut c_void };
            let xdm = unsafe { (self.mlm_xd.ptr as *mut f32).add(768) as *mut c_void };
            if self.use_had {
                // silu·up → f32 act → per-128 rotate (within each expert's eff) → quant
                self.launch(
                    "moe_silu_mul",
                    n_act.div_ceil(256) as u32,
                    256,
                    0,
                    Args::new()
                        .ptr(self.moe_g.ptr)
                        .ptr(self.moe_u.ptr)
                        .ptr(self.moe_act.ptr)
                        .i(n_act as i32),
                );
                self.launch(
                    "fht128",
                    (n_act / 128) as u32,
                    128,
                    0,
                    Args::new()
                        .ptr(self.moe_act.ptr)
                        .ptr(self.pf_a.ptr)
                        .i(n_act as i32),
                );
                self.launch(
                    "xquant8",
                    nb_act as u32,
                    32,
                    0,
                    Args::new()
                        .ptr(self.pf_a.ptr)
                        .ptr(xqm)
                        .ptr(xdm)
                        .i(nb_act as i32),
                );
            } else {
                self.launch(
                    "silu_quant",
                    nb_act as u32,
                    32,
                    0,
                    Args::new()
                        .ptr(self.moe_g.ptr)
                        .ptr(self.moe_u.ptr)
                        .ptr(xqm)
                        .ptr(xdm)
                        .i(nb_act as i32),
                );
            }
            if self.use_q4 {
                self.launch2(
                    "q4i_moe_down",
                    m.hidden as u32,
                    k as u32,
                    32,
                    0,
                    Args::new()
                        .ptr(m.down_s4.ptr)
                        .ptr(m.down_q4.ptr)
                        .ptr(self.moe_ids.ptr)
                        .ptr(xdm)
                        .ptr(xqm)
                        .ptr(self.moe_dy.ptr)
                        .i(m.eff as i32)
                        .i(m.hidden as i32),
                );
            } else {
                self.launch2(
                    "q8i_moe_down",
                    m.hidden as u32,
                    k as u32,
                    32,
                    0,
                    Args::new()
                        .ptr(m.down_s.ptr)
                        .ptr(m.down_q.ptr)
                        .ptr(self.moe_ids.ptr)
                        .ptr(xdm)
                        .ptr(xqm)
                        .ptr(self.moe_dy.ptr)
                        .i(m.eff as i32)
                        .i(m.hidden as i32),
                );
            }
        } else {
            self.launch2(
                "q8_moe_down",
                m.hidden as u32,
                k as u32,
                32,
                0,
                Args::new()
                    .ptr(m.down_s.ptr)
                    .ptr(m.down_q.ptr)
                    .ptr(self.moe_ids.ptr)
                    .ptr(self.moe_w.ptr)
                    .ptr(self.moe_g.ptr)
                    .ptr(self.moe_u.ptr)
                    .ptr(self.moe_dy.ptr)
                    .i(m.eff as i32)
                    .i(m.hidden as i32)
                    .i(k as i32),
            );
        }
        self.launch(
            "moe_wsum",
            m.hidden.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.moe_dy.ptr)
                .ptr(self.moe_w.ptr)
                .ptr(self.moe_out.ptr)
                .i(m.hidden as i32)
                .i(k as i32),
        );
    }

    /// lm_head GEMV (tied token_embd): Q6_K (gemma-4 target) or Q4_0 (a pure-Q4_0
    /// draft). `x` = final-normed hidden [hidden], `y` = logits out [vocab].
    fn lm_head(&self, x: *mut c_void, y: *mut c_void) {
        if let Some(e) = self.q6.get("token_embd.weight") {
            self.launch(
                "q6_gemv",
                e.out_dim.div_ceil(16) as u32,
                256,
                0,
                Args::new()
                    .ptr(e.scales.ptr)
                    .ptr(e.ql.ptr)
                    .ptr(e.qh.ptr)
                    .ptr(x)
                    .ptr(y)
                    .i(e.in_dim as i32)
                    .i(e.out_dim as i32),
            );
        } else if let Some(e) = self.q4.get("token_embd.weight") {
            self.launch(
                "q4_gemv",
                e.out_dim as u32,
                32,
                0,
                Args::new()
                    .ptr(e.scales.ptr)
                    .ptr(e.quants.ptr)
                    .ptr(x)
                    .ptr(y)
                    .i(e.in_dim as i32)
                    .i(e.out_dim as i32),
            );
        }
    }

    fn launch2(&self, name: &str, gx: u32, gy: u32, block: u32, shared: u32, args: Args) {
        self.launch3(name, gx, gy, 1, block, shared, args);
    }

    /// Stream sync before an NPU dispatch (ensures the activation written by
    /// `xquant_npu` is visible before the NPU reads it). Timed under
    /// STRIX_NPU_TIMING without adding any extra syncs.
    #[cfg(feature = "npu")]
    fn npu_sync(&self) {
        if crate::npu_hybrid::timing_enabled() {
            let t0 = std::time::Instant::now();
            let _ = self.gpu.sync();
            crate::npu_hybrid::npu_time_add("gpu.sync", t0.elapsed().as_secs_f64());
        } else {
            let _ = self.gpu.sync();
        }
    }

    fn launch3(&self, name: &str, gx: u32, gy: u32, gz: u32, block: u32, shared: u32, args: Args) {
        let mut a = args;
        let mut ptrs: Vec<*mut c_void> = a
            .vals
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut c_void)
            .collect();
        if prof_enabled() {
            let _ = self.gpu.sync();
            let t0 = std::time::Instant::now();
            let _ = self.gpu.launch(
                self.funcs[name],
                (gx, gy, gz),
                (block, 1, 1),
                shared,
                &mut ptrs,
            );
            let _ = self.gpu.sync();
            prof_add(name, t0.elapsed().as_secs_f64());
            return;
        }
        let _ = self.gpu.launch(
            self.funcs[name],
            (gx, gy, gz),
            (block, 1, 1),
            shared,
            &mut ptrs,
        );
    }

    /// Open the NPU ffn_up offload (one xclbin for the up shape) and stage all
    /// stashed per-layer Q4 up-weights as int8. No-op if no pending weights.
    #[cfg(feature = "npu")]
    fn npu_init(&mut self) {
        if self.npu_pending.is_empty() {
            return;
        }
        let (k, ffn) = {
            let cfg = self.cfg.as_ref().unwrap();
            (cfg.hidden, cfg.ffn)
        };
        // Split-N: iGPU computes ffn_up cols [0,n_ig); NPU computes [n_ig, ffn).
        // Default n_ig=0 → ffn_up runs FULLY on the NPU (full-N xclbin exists), the
        // max-NPU / min-iGPU posture for this box (the partition doesn't change the
        // math; rescale concatenates). ~1% slower than the speed-optimal 2048 split,
        // but keeps the biggest GEMM (N=15360) off the iGPU. STRIX_NPU_NIG overrides.
        let n_ig = std::env::var("STRIX_NPU_NIG")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0usize);
        let n_npu = ffn - n_ig;
        let dir = std::env::var("STRIX_NPU_XCLBIN_DIR").unwrap_or_else(|_| {
            "external/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build".into()
        });
        // AIE column count for the offload xclbins (4 = 16 cores, 8 = 32 cores).
        // 8-col is ~+35% on the down GEMM (measured), letting the NPU take a
        // bigger offload share. Requires the matching `_{cols}c` xclbins to exist.
        let cols = std::env::var("STRIX_NPU_COLS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8u32);
        let xclbin = format!("{dir}/final_256x{k}x{n_npu}_64x64x64_{cols}c.xclbin");
        let insts_path = format!("{dir}/insts_256x{k}x{n_npu}_64x64x64_{cols}c.txt");
        let raw = match std::fs::read(&insts_path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("npu: no insts {insts_path}: {e}");
                return;
            }
        };
        let instr = match strix_backend_npu::load_instr_bin(&raw) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("npu: bad insts: {e}");
                return;
            }
        };
        let mut npuffn =
            match crate::npu_hybrid::NpuFfn::open(&self.gpu, &xclbin, &instr, k, n_npu, n_ig) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("npu: open failed: {e}");
                    return;
                }
            };
        let pending = std::mem::take(&mut self.npu_pending);
        for (l, bytes) in pending {
            if let Err(e) = npuffn.stage_q4(&self.gpu, l, &bytes) {
                eprintln!("npu: stage layer {l} failed: {e}");
                return;
            }
        }
        eprintln!("npu: split ffn_up — iGPU cols[0,{n_ig}) ∥ NPU cols[{n_ig},{ffn}) ({}/{ffn}); {} layers staged",
            n_npu, npuffn.layers.len());
        self.npu = Some(npuffn);

        // ffn_down split: K=ffn, output N=hidden; NPU does cols [n_ig_d, hidden).
        if !self.npu_down_pending.is_empty() {
            let n_npu_d = std::env::var("STRIX_NPU_NNPU_D")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2560usize);
            let n_ig_d = k - n_npu_d; // k here == hidden (down output width)
            let xclbin = format!("{dir}/final_256x{ffn}x{n_npu_d}_64x64x64_{cols}c.xclbin");
            let insts_path = format!("{dir}/insts_256x{ffn}x{n_npu_d}_64x64x64_{cols}c.txt");
            match std::fs::read(&insts_path)
                .map_err(|e| e.to_string())
                .and_then(|raw| strix_backend_npu::load_instr_bin(&raw))
                .and_then(|instr| {
                    crate::npu_hybrid::NpuFfn::open(
                        &self.gpu, &xclbin, &instr, ffn, n_npu_d, n_ig_d,
                    )
                }) {
                Ok(mut dn) => {
                    let pend = std::mem::take(&mut self.npu_down_pending);
                    let mut ok = true;
                    for (l, bytes) in pend {
                        if let Err(e) = dn.stage_q4(&self.gpu, l, &bytes) {
                            eprintln!("npu: down stage {l}: {e}");
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        eprintln!("npu: split ffn_down — iGPU cols[0,{n_ig_d}) ∥ NPU cols[{n_ig_d},{k}) ({n_npu_d}/{k})");
                        self.npu_down = Some(dn);
                    }
                }
                Err(e) => eprintln!("npu: down open failed ({xclbin}): {e}"),
            }
        }

        // local attn_output split: K=q_dim(4096), output N=hidden; NPU cols [n_ig_o,hidden).
        if !self.npu_o_pending.is_empty() {
            let ko = 4096usize;
            let n_npu_o = std::env::var("STRIX_NPU_NNPU_O")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2048usize);
            let n_ig_o = k - n_npu_o; // k == hidden
            let xclbin = format!("{dir}/final_256x{ko}x{n_npu_o}_64x64x64_{cols}c.xclbin");
            let insts_path = format!("{dir}/insts_256x{ko}x{n_npu_o}_64x64x64_{cols}c.txt");
            match std::fs::read(&insts_path)
                .map_err(|e| e.to_string())
                .and_then(|raw| strix_backend_npu::load_instr_bin(&raw))
                .and_then(|instr| {
                    crate::npu_hybrid::NpuFfn::open(&self.gpu, &xclbin, &instr, ko, n_npu_o, n_ig_o)
                }) {
                Ok(mut on) => {
                    let pend = std::mem::take(&mut self.npu_o_pending);
                    let mut ok = true;
                    for (l, bytes) in pend {
                        if let Err(e) = on.stage_q4(&self.gpu, l, &bytes) {
                            eprintln!("npu: o stage {l}: {e}");
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        eprintln!("npu: split attn_output(local) — iGPU cols[0,{n_ig_o}) ∥ NPU cols[{n_ig_o},{k}) ({n_npu_o}/{k})");
                        self.npu_o = Some(on);
                    }
                }
                Err(e) => eprintln!("npu: o open failed ({xclbin}): {e}"),
            }
        }

        // local attn_q split: K=hidden, output N=q_dim(4096); NPU cols [n_ig_q,4096).
        // Runs during the otherwise-idle qkv window (the iGPU does q[0,n_ig_q]+k+v),
        // so it moves FLOPs off the iGPU for free (power) at no wall-time cost.
        if !self.npu_q_pending.is_empty() {
            let qd = 4096usize; // local q_dim
                                // Default 4096 → attn_q(local) FULLY on NPU (full-N 256x3840x4096_8c xclbin
                                // exists); max-NPU posture, transparent partition. STRIX_NPU_NNPU_Q overrides.
            let n_npu_q = std::env::var("STRIX_NPU_NNPU_Q")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4096usize);
            let n_ig_q = qd - n_npu_q;
            let xclbin = format!("{dir}/final_256x{k}x{n_npu_q}_64x64x64_{cols}c.xclbin");
            let insts_path = format!("{dir}/insts_256x{k}x{n_npu_q}_64x64x64_{cols}c.txt");
            match std::fs::read(&insts_path)
                .map_err(|e| e.to_string())
                .and_then(|raw| strix_backend_npu::load_instr_bin(&raw))
                .and_then(|instr| {
                    crate::npu_hybrid::NpuFfn::open(&self.gpu, &xclbin, &instr, k, n_npu_q, n_ig_q)
                }) {
                Ok(mut qn) => {
                    let pend = std::mem::take(&mut self.npu_q_pending);
                    let mut ok = true;
                    for (l, bytes) in pend {
                        if let Err(e) = qn.stage_q4(&self.gpu, l, &bytes) {
                            eprintln!("npu: q stage {l}: {e}");
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        eprintln!("npu: split attn_q(local) — iGPU cols[0,{n_ig_q}) ∥ NPU cols[{n_ig_q},{qd}) ({n_npu_q}/{qd})");
                        self.npu_q = Some(qn);
                    }
                }
                Err(e) => eprintln!("npu: q open failed ({xclbin}): {e}"),
            }
        }

        // global attn_q split: K=hidden, N=q_dim(8192); NPU cols [n_ig_qg,8192).
        if !self.npu_q_g_pending.is_empty() {
            let qd = 8192usize; // global q_dim
            let n_npu_qg = std::env::var("STRIX_NPU_NNPU_QG")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4096usize);
            let n_ig_qg = qd - n_npu_qg;
            let xclbin = format!("{dir}/final_256x{k}x{n_npu_qg}_64x64x64_{cols}c.xclbin");
            let insts_path = format!("{dir}/insts_256x{k}x{n_npu_qg}_64x64x64_{cols}c.txt");
            match std::fs::read(&insts_path)
                .map_err(|e| e.to_string())
                .and_then(|raw| strix_backend_npu::load_instr_bin(&raw))
                .and_then(|instr| {
                    crate::npu_hybrid::NpuFfn::open(
                        &self.gpu, &xclbin, &instr, k, n_npu_qg, n_ig_qg,
                    )
                }) {
                Ok(mut qn) => {
                    let pend = std::mem::take(&mut self.npu_q_g_pending);
                    let mut ok = true;
                    for (l, bytes) in pend {
                        if let Err(e) = qn.stage_q4(&self.gpu, l, &bytes) {
                            eprintln!("npu: qg stage {l}: {e}");
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        eprintln!("npu: split attn_q(global) — iGPU cols[0,{n_ig_qg}) ∥ NPU cols[{n_ig_qg},{qd}) ({n_npu_qg}/{qd})");
                        self.npu_q_g = Some(qn);
                    }
                }
                Err(e) => eprintln!("npu: qg open failed ({xclbin}): {e}"),
            }
        }

        // global attn_output split: K=q_dim(8192), N=hidden; NPU cols [n_ig_og,hidden).
        if !self.npu_o_g_pending.is_empty() {
            let ko = 8192usize;
            let n_npu_og = std::env::var("STRIX_NPU_NNPU_OG")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2048usize);
            let n_ig_og = k - n_npu_og; // k == hidden
            let xclbin = format!("{dir}/final_256x{ko}x{n_npu_og}_64x64x64_{cols}c.xclbin");
            let insts_path = format!("{dir}/insts_256x{ko}x{n_npu_og}_64x64x64_{cols}c.txt");
            match std::fs::read(&insts_path)
                .map_err(|e| e.to_string())
                .and_then(|raw| strix_backend_npu::load_instr_bin(&raw))
                .and_then(|instr| {
                    crate::npu_hybrid::NpuFfn::open(
                        &self.gpu, &xclbin, &instr, ko, n_npu_og, n_ig_og,
                    )
                }) {
                Ok(mut on) => {
                    let pend = std::mem::take(&mut self.npu_o_g_pending);
                    let mut ok = true;
                    for (l, bytes) in pend {
                        if let Err(e) = on.stage_q4(&self.gpu, l, &bytes) {
                            eprintln!("npu: og stage {l}: {e}");
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        eprintln!("npu: split attn_output(global) — iGPU cols[0,{n_ig_og}) ∥ NPU cols[{n_ig_og},{k}) ({n_npu_og}/{k})");
                        self.npu_o_g = Some(on);
                    }
                }
                Err(e) => eprintln!("npu: og open failed ({xclbin}): {e}"),
            }
        }

        // STRIX_NPU_ROOFLINE: standalone per-shape NPU GEMM roofline (idea #26).
        // Decides whether int4-on-AIE (#13) is worth building — only memory-bound
        // shapes benefit. NPU-only (<2 W), safe to run sustained.
        if std::env::var("STRIX_NPU_ROOFLINE").is_ok() {
            let reps: usize = std::env::var("STRIX_NPU_ROOFLINE")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&r| r > 1)
                .unwrap_or(50);
            for (label, ffn) in [
                ("ffn_up", &self.npu),
                ("ffn_down", &self.npu_down),
                ("attn_o", &self.npu_o),
                ("attn_q", &self.npu_q),
                ("attn_q_g", &self.npu_q_g),
                ("attn_o_g", &self.npu_o_g),
            ] {
                if let Some(f) = ffn {
                    if let Some((wid, _)) = f.layer(0) {
                        f.roofline(label, wid, reps);
                    }
                }
            }
        }
    }
}

impl WeightAccel for RocmWeightAccel {
    fn upload_q4_0(&mut self, key: &str, bytes: &[u8], in_dim: usize, out_dim: usize) -> bool {
        if in_dim % QK4_0 != 0 {
            return false;
        }
        let nblocks = in_dim / QK4_0;
        let total = nblocks * out_dim;
        if bytes.len() != total * Q4_0_BYTES {
            return false;
        }
        let mut scales = vec![0u16; total];
        let mut quants = vec![0u32; total * 4];
        for (b, blk) in bytes.chunks_exact(Q4_0_BYTES).enumerate() {
            scales[b] = u16::from_le_bytes([blk[0], blk[1]]);
            let qs = &blk[2..18];
            for w in 0..4 {
                quants[b * 4 + w] =
                    u32::from_le_bytes([qs[w * 4], qs[w * 4 + 1], qs[w * 4 + 2], qs[w * 4 + 3]]);
            }
        }
        let (Ok(sb), Ok(qb)) = (self.gpu.upload_new(&scales), self.gpu.upload_new(&quants)) else {
            return false;
        };
        self.q4.insert(
            key.to_string(),
            ResQ4 {
                scales: sb,
                quants: qb,
                in_dim,
                out_dim,
            },
        );
        // NPU hybrid: stash ffn_up raw Q4 bytes for offload (repacked at configure).
        #[cfg(feature = "npu")]
        if std::env::var("STRIX_NPU").is_ok() {
            // STRIX_NPU_MODE: "power" (DEFAULT) offloads everything (ffn_up/down +
            // attn_o/q) to the NPU — maximizes NPU use and MINIMIZES sustained iGPU
            // compute. This is the default on this box because heavy/sustained iGPU
            // load is the trigger for its SoC-reset hardware fault; shifting GEMMs to
            // the low-power NPU keeps the iGPU's own rail load down. "speed" offloads
            // only ffn_up (the gate∥up free-parallelism pair), which is marginally
            // faster at high iGPU clock but works the iGPU harder. STRIX_NPU_SKIP
            // (comma list of up,down,o,q) is a fine override.
            let power = std::env::var("STRIX_NPU_MODE")
                .map(|m| m != "speed")
                .unwrap_or(true);
            let skipv = std::env::var("STRIX_NPU_SKIP").unwrap_or_default();
            let skip = |n: &str| {
                skipv.split(',').any(|s| s.trim() == n)
                    || (!power && matches!(n, "down" | "o" | "q"))
            };
            if let Some(l) = key
                .strip_prefix("blk.")
                .and_then(|s| s.strip_suffix(".ffn_up.weight"))
            {
                if let (Ok(l), false) = (l.parse::<usize>(), skip("up")) {
                    self.npu_pending.insert(l, bytes.to_vec());
                }
            } else if let Some(l) = key
                .strip_prefix("blk.")
                .and_then(|s| s.strip_suffix(".ffn_down.weight"))
            {
                if let (Ok(l), false) = (l.parse::<usize>(), skip("down")) {
                    self.npu_down_pending.insert(l, bytes.to_vec());
                }
            } else if in_dim == 4096 || in_dim == 8192 {
                // attn_output: local (q_dim=4096) → npu_o, global (8192) → npu_o_g.
                if let Some(l) = key
                    .strip_prefix("blk.")
                    .and_then(|s| s.strip_suffix(".attn_output.weight"))
                {
                    if let (Ok(l), false) = (l.parse::<usize>(), skip("o")) {
                        if in_dim == 4096 {
                            self.npu_o_pending.insert(l, bytes.to_vec());
                        } else {
                            self.npu_o_g_pending.insert(l, bytes.to_vec());
                        }
                    }
                }
            } else if out_dim == 4096 || out_dim == 8192 {
                // attn_q (K=hidden): local (q_dim=4096) → npu_q, global (8192) → npu_q_g.
                if let Some(l) = key
                    .strip_prefix("blk.")
                    .and_then(|s| s.strip_suffix(".attn_q.weight"))
                {
                    if let (Ok(l), false) = (l.parse::<usize>(), skip("q")) {
                        if out_dim == 4096 {
                            self.npu_q_pending.insert(l, bytes.to_vec());
                        } else {
                            self.npu_q_g_pending.insert(l, bytes.to_vec());
                        }
                    }
                }
            }
        }
        true
    }

    fn upload_q6_k(&mut self, key: &str, bytes: &[u8], in_dim: usize, out_dim: usize) -> bool {
        if in_dim % QK_K != 0 {
            return false;
        }
        let nblocks = in_dim / QK_K;
        let total = nblocks * out_dim;
        if bytes.len() != total * Q6_K_BYTES {
            return false;
        }
        let mut scales = vec![0.0f32; total * 16];
        let mut ql = vec![0u32; total * 32];
        let mut qh = vec![0u32; total * 16];
        for (b, blk) in bytes.chunks_exact(Q6_K_BYTES).enumerate() {
            let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
            for j in 0..16 {
                scales[b * 16 + j] = d * (blk[192 + j] as i8) as f32;
            }
            for w in 0..32 {
                ql[b * 32 + w] = u32::from_le_bytes([
                    blk[w * 4],
                    blk[w * 4 + 1],
                    blk[w * 4 + 2],
                    blk[w * 4 + 3],
                ]);
            }
            for w in 0..16 {
                qh[b * 16 + w] = u32::from_le_bytes([
                    blk[128 + w * 4],
                    blk[128 + w * 4 + 1],
                    blk[128 + w * 4 + 2],
                    blk[128 + w * 4 + 3],
                ]);
            }
        }
        let (Ok(scb), Ok(qlb), Ok(qhb)) = (
            self.gpu.upload_new(&scales),
            self.gpu.upload_new(&ql),
            self.gpu.upload_new(&qh),
        ) else {
            return false;
        };
        self.q6.insert(
            key.to_string(),
            ResQ6 {
                scales: scb,
                ql: qlb,
                qh: qhb,
                in_dim,
                out_dim,
            },
        );
        true
    }

    fn upload_q8_0(&mut self, key: &str, bytes: &[u8], in_dim: usize, out_dim: usize) -> bool {
        if in_dim % QK8_0 != 0 {
            return false;
        }
        let nblocks = in_dim / QK8_0;
        let total = nblocks * out_dim;
        if bytes.len() != total * Q8_0_BYTES {
            return false;
        }
        // Repack to f32 scales [total] + int8 quants [total*32] (matches q8_0_gemv).
        let mut scales = vec![0.0f32; total];
        let mut quants = vec![0i8; total * 32];
        for (b, blk) in bytes.chunks_exact(Q8_0_BYTES).enumerate() {
            scales[b] = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for i in 0..32 {
                quants[b * 32 + i] = blk[2 + i] as i8;
            }
        }
        let (Ok(sb), Ok(qb)) = (self.gpu.upload_new(&scales), self.gpu.upload_new(&quants)) else {
            return false;
        };
        self.q8.insert(
            key.to_string(),
            ResQ8 {
                scales: sb,
                quants: qb,
                in_dim,
                out_dim,
            },
        );
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_moe_q8(
        &mut self,
        layer: usize,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
        hidden: usize,
        eff: usize,
        ne: usize,
    ) -> bool {
        if hidden % QK8_0 != 0 || eff % QK8_0 != 0 || eff > 4096 || hidden > 4096 {
            return false;
        }
        let gate_eb = (hidden / QK8_0) * Q8_0_BYTES * eff;
        let down_eb = (eff / QK8_0) * Q8_0_BYTES * hidden;
        if gate.len() != gate_eb * ne || up.len() != gate_eb * ne || down.len() != down_eb * ne {
            return false;
        }
        let (gs, gq) = planar_q8(gate);
        let (us, uq) = planar_q8(up);
        let (ds, dq) = planar_q8(down);
        let (Ok(gsb), Ok(gqb), Ok(usb), Ok(uqb), Ok(dsb), Ok(dqb)) = (
            self.gpu.upload_new(&gs),
            self.gpu.upload_new(&gq),
            self.gpu.upload_new(&us),
            self.gpu.upload_new(&uq),
            self.gpu.upload_new(&ds),
            self.gpu.upload_new(&dq),
        ) else {
            return false;
        };
        // int4 probe buffers (packed nibbles); 1-elem dummy if disabled.
        let mk4 = |me: &Self, sc: &[f32], q: &[i8], inb: usize| -> Option<(Dbuf, Dbuf)> {
            if me.use_q4 {
                let (s4, q4) = if me.use_had {
                    pack_q4_had(sc, q, inb)
                } else {
                    pack_q4(sc, q)
                };
                Some((me.gpu.upload_new(&s4).ok()?, me.gpu.upload_new(&q4).ok()?))
            } else {
                Some((
                    me.gpu.upload_new(&[0.0f32]).ok()?,
                    me.gpu.upload_new(&[0u8]).ok()?,
                ))
            }
        };
        let (Some((gs4, gq4)), Some((us4, uq4)), Some((ds4, dq4))) = (
            mk4(self, &gs, &gq, hidden / 32),
            mk4(self, &us, &uq, hidden / 32),
            mk4(self, &ds, &dq, eff / 32),
        ) else {
            return false;
        };
        self.moe.insert(
            layer,
            ResMoe {
                gate_s: gsb,
                gate_q: gqb,
                up_s: usb,
                up_q: uqb,
                down_s: dsb,
                down_q: dqb,
                gate_s4: gs4,
                gate_q4: gq4,
                up_s4: us4,
                up_q4: uq4,
                down_s4: ds4,
                down_q4: dq4,
                hidden,
                eff,
            },
        );
        true
    }

    fn upload_moe_q6(
        &mut self,
        layer: usize,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
        hidden: usize,
        eff: usize,
        ne: usize,
    ) -> bool {
        if hidden % QK_K != 0 || eff % QK_K != 0 || eff > 4096 || hidden > 4096 {
            return false;
        }
        let gate_eb = (hidden / QK_K) * Q6_K_BYTES * eff;
        let down_eb = (eff / QK_K) * Q6_K_BYTES * hidden;
        if gate.len() != gate_eb * ne || up.len() != gate_eb * ne || down.len() != down_eb * ne {
            return false;
        }
        let (Ok(gb), Ok(ub), Ok(db)) = (
            self.gpu.upload_new(gate),
            self.gpu.upload_new(up),
            self.gpu.upload_new(down),
        ) else {
            return false;
        };
        self.moe6.insert(
            layer,
            ResMoe6 {
                gate: gb,
                up: ub,
                down: db,
                hidden,
                eff,
                gate_eb: gate_eb as i64,
                down_eb: down_eb as i64,
            },
        );
        true
    }

    fn moe_ffn(
        &self,
        layer: usize,
        ids: &[i32],
        wexp: &[f32],
        x: &[f32],
        sgate: f32,
    ) -> Option<Vec<f32>> {
        if let Some(m) = self.moe6.get(&layer) {
            let k = ids.len();
            if x.len() != m.hidden || wexp.len() != k || k > 16 {
                return None;
            }
            self.gemv_x.upload(x).ok()?;
            self.moe_ids.upload(ids).ok()?;
            self.moe_w.upload(wexp).ok()?;
            self.moe6_launches(m, k, self.gemv_x.ptr);
            if sgate != 0.0 && !self.queue_shexp(layer, m.hidden, sgate) {
                return None;
            }
            self.gpu.sync().ok()?;
            return self.moe_out.download::<f32>(m.hidden).ok();
        }
        let m = self.moe.get(&layer)?;
        let k = ids.len();
        if x.len() != m.hidden || wexp.len() != k || k > 16 {
            return None;
        }
        self.gemv_x.upload(x).ok()?;
        self.moe_ids.upload(ids).ok()?;
        self.moe_w.upload(wexp).ok()?;
        self.moe_launches(m, k, self.gemv_x.ptr);
        if sgate != 0.0 && !self.queue_shexp(layer, m.hidden, sgate) {
            return None;
        }
        self.gpu.sync().ok()?;
        self.moe_out.download::<f32>(m.hidden).ok()
    }

    fn mlm_begin(&mut self, h: &[f32]) -> bool {
        if self.moe.is_empty() || h.len() > 4096 {
            return false;
        }
        self.mlm_h.upload(h).is_ok()
    }

    /// Full on-GPU Mellum layer: norm→q/k/v→QK-norm→rope(table)→KV append→SDPA→o→
    /// residual→ffn norm→router. cs/sn rope tables already uploaded for this layer
    /// type. ONE sync; returns router logits.
    #[allow(clippy::too_many_arguments)]
    fn mlm_layer(&mut self, il: usize, pos: usize, win: usize) -> Option<Vec<f32>> {
        if !self.mlm_attn_part(il, pos, win) {
            return None;
        }
        self.gpu.sync().ok()?;
        self.mlm_rl.download::<f32>(64).ok()
    }

    /// Full layer with on-GPU router top-k + queued MoE — NO sync. Caller syncs once
    /// per token at lm_head. Requires the same residency as mlm_layer.
    fn mlm_layer_nosync(&mut self, il: usize, pos: usize, win: usize, topk: usize) -> bool {
        if std::env::var("STRIX_MLM_PROF").is_ok() {
            return self.mlm_layer_prof(il, pos, win, topk);
        }

        let rl_dim = 64usize; // F32 router (n_expert)
                              // attn part: reuse the mlm_layer body up to the router (without sync/download)
        if !self.mlm_attn_part(il, pos, win) {
            {
                return false;
            }
        }
        self.launch(
            "topk_router",
            1,
            32,
            0,
            Args::new()
                .ptr(self.mlm_rl.ptr)
                .i(rl_dim as i32)
                .i(topk as i32)
                .ptr(self.moe_ids.ptr)
                .ptr(self.moe_w.ptr),
        );
        if std::env::var("STRIX_RT_DEBUG").is_ok() && il < 3 {
            let _ = self.gpu.sync();
            let rl = self.mlm_rl.download::<f32>(rl_dim).unwrap_or_default();
            let ids = self.moe_ids.download::<i32>(topk).unwrap_or_default();
            let w = self.moe_w.download::<f32>(topk).unwrap_or_default();
            let mut idx: Vec<usize> = (0..rl_dim).collect();
            let mx = rl.iter().cloned().fold(f32::MIN, f32::max);
            let probs: Vec<f32> = rl.iter().map(|&l| (l - mx).exp()).collect();
            let sum: f32 = probs.iter().sum();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            eprintln!(
                "[rt l{il}] gpu_ids={ids:?} w0={:.3} cpu_top={:?} p0={:.3}",
                w.first().copied().unwrap_or(0.0),
                &idx[..topk.min(8)],
                probs[idx[0]] / sum
            );
        }
        let Some(m) = self.moe.get(&il) else {
            {
                return false;
            }
        };
        self.moe_launches(m, topk, self.mlm_n.ptr);
        self.launch(
            "vec_add",
            m.hidden.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_h.ptr)
                .ptr(self.moe_out.ptr)
                .i(m.hidden as i32),
        );
        true
    }

    /// Upload rope cos/sin tables (per layer type, current pos) + allocate KV caches.
    fn mlm_prepare(&mut self, n_layers: usize, kv_dim: usize, max_seq: usize) -> bool {
        if self.mlm_kc.len() == n_layers && self.mlm_seq >= max_seq {
            return true;
        }
        self.mlm_kc.clear();
        self.mlm_vc.clear();
        for _ in 0..n_layers {
            let (Ok(k), Ok(v)) = (
                self.gpu.alloc(max_seq * kv_dim * 4),
                self.gpu.alloc(max_seq * kv_dim * 4),
            ) else {
                return false;
            };
            self.mlm_kc.push(k);
            self.mlm_vc.push(v);
        }
        self.mlm_seq = max_seq;
        true
    }

    fn mlm_seed_kv(&mut self, il: usize, k: &[f32], v: &[f32]) -> bool {
        if il >= self.mlm_kc.len() {
            return false;
        }
        self.gpu.upload_at(&self.mlm_kc[il], 0, k).is_ok()
            && self.gpu.upload_at(&self.mlm_vc[il], 0, v).is_ok()
    }

    fn mlm_rope_tables(&mut self, cs: &[f32], sn: &[f32]) -> bool {
        self.mlm_cs.upload(cs).is_ok() && self.mlm_sn.upload(sn).is_ok()
    }

    fn mlm_qkv(&mut self, il: usize) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let nw = self.f32w.get(&format!("blk.{il}.attn_norm.weight"))?;
        let wq = self.q8.get(&format!("blk.{il}.attn_q.weight"))?;
        let wk = self.q8.get(&format!("blk.{il}.attn_k.weight"))?;
        let wv = self.q8.get(&format!("blk.{il}.attn_v.weight"))?;
        self.norm_launch(self.mlm_h.ptr, nw.ptr, self.mlm_n.ptr, wq.in_dim);
        self.q8_launch(wq, self.mlm_n.ptr, self.mlm_q.ptr);
        self.q8_launch(wk, self.mlm_n.ptr, self.mlm_k.ptr);
        self.q8_launch(wv, self.mlm_n.ptr, self.mlm_v.ptr);
        self.gpu.sync().ok()?;
        Some((
            self.mlm_q.download::<f32>(wq.out_dim).ok()?,
            self.mlm_k.download::<f32>(wk.out_dim).ok()?,
            self.mlm_v.download::<f32>(wv.out_dim).ok()?,
        ))
    }

    fn mlm_post1(&mut self, il: usize, attn_out: &[f32]) -> Option<Vec<f32>> {
        let wo = self.q8.get(&format!("blk.{il}.attn_output.weight"))?;
        let fnw = self.f32w.get(&format!("blk.{il}.ffn_norm.weight"))?;
        let wgi = self.q8.get(&format!("blk.{il}.ffn_gate_inp.weight"))?;
        if attn_out.len() != wo.in_dim {
            return None;
        }
        self.mlm_attn.upload(attn_out).ok()?;
        self.q8_launch(wo, self.mlm_attn.ptr, self.moe_out.ptr);
        let hidden = wo.out_dim;
        self.launch(
            "vec_add",
            hidden.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_h.ptr)
                .ptr(self.moe_out.ptr)
                .i(hidden as i32),
        );
        self.norm_launch(self.mlm_h.ptr, fnw.ptr, self.mlm_n.ptr, hidden);
        self.q8_launch(wgi, self.mlm_n.ptr, self.mlm_rl.ptr);
        self.gpu.sync().ok()?;
        self.mlm_rl.download::<f32>(wgi.out_dim).ok()
    }

    fn mlm_post2(&mut self, il: usize, ids: &[i32], wexp: &[f32]) -> bool {
        let Some(m) = self.moe.get(&il) else {
            return false;
        };
        let k = ids.len();
        if self.moe_ids.upload(ids).is_err() || self.moe_w.upload(wexp).is_err() {
            return false;
        }
        self.moe_launches(m, k, self.mlm_n.ptr);
        self.launch(
            "vec_add",
            m.hidden.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.mlm_h.ptr)
                .ptr(self.moe_out.ptr)
                .i(m.hidden as i32),
        );
        true // no sync — next layer's qkv sync covers it
    }

    /// Graph token: capture all layers + lm_head once, then replay per token.
    /// Host per token: upload h/pos/rope tables, replay, sync, download logits.
    fn mlm_token_graph(&mut self, layers: &[(usize, bool)], topk: usize) -> Option<Vec<f32>> {
        let head = self.q8.get("output.weight")?;
        let on = self.f32w.get("output_norm.weight")?;
        let (head_in, head_out) = (head.in_dim, head.out_dim);
        let (head_s, head_q, on_ptr) = (head.scales.ptr, head.quants.ptr, on.ptr);
        if self.use_q4head && self.head_q4.is_none() {
            let nbt = head_out * (head_in / 32);
            if let (Ok(sc), Ok(q)) = (
                head.scales.download::<f32>(nbt),
                head.quants.download::<i8>(nbt * 32),
            ) {
                let (s4, q4) = pack_q4(&sc, &q);
                if let (Ok(sb), Ok(qb)) = (self.gpu.upload_new(&s4), self.gpu.upload_new(&q4)) {
                    self.head_q4 = Some((sb, qb));
                }
            }
        }
        if self.mlm_graph.is_none() {
            unsafe {
                if crate::ffi::hipStreamBeginCapture(self.gpu.stream, 0) != 0 {
                    return None;
                }
            }
            let mut ok = true;
            for (il, &(win, full)) in layers.iter().enumerate() {
                if !self.mlm_layer_pos(il, win, full, topk) {
                    ok = false;
                    break;
                }
            }
            if ok {
                // final norm + lm_head into gemv_y
                self.norm_launch(self.mlm_h.ptr, on_ptr, self.mlm_n.ptr, head_in);
                if self.mlm_int8 {
                    self.launch(
                        "xquant8",
                        (head_in / 32) as u32,
                        32,
                        0,
                        Args::new()
                            .ptr(self.mlm_n.ptr)
                            .ptr(self.mlm_xq.ptr)
                            .ptr(self.mlm_xd.ptr)
                            .i((head_in / 32) as i32),
                    );
                    if let (true, Some((hs4, hq4))) = (self.use_q4head, self.head_q4.as_ref()) {
                        self.launch(
                            "q4i_gemv",
                            head_out as u32,
                            32,
                            0,
                            Args::new()
                                .ptr(hs4.ptr)
                                .ptr(hq4.ptr)
                                .ptr(self.mlm_xd.ptr)
                                .ptr(self.mlm_xq.ptr)
                                .ptr(self.gemv_y.ptr)
                                .i(head_in as i32)
                                .i(head_out as i32),
                        );
                    } else {
                        self.launch(
                            "q8i_gemv",
                            head_out as u32,
                            32,
                            0,
                            Args::new()
                                .ptr(head_s)
                                .ptr(head_q)
                                .ptr(self.mlm_xd.ptr)
                                .ptr(self.mlm_xq.ptr)
                                .ptr(self.gemv_y.ptr)
                                .i(head_in as i32)
                                .i(head_out as i32),
                        );
                    }
                } else {
                    self.launch(
                        "q8_0_gemv",
                        head_out as u32,
                        32,
                        0,
                        Args::new()
                            .ptr(head_s)
                            .ptr(head_q)
                            .ptr(self.mlm_n.ptr)
                            .ptr(self.gemv_y.ptr)
                            .i(head_in as i32)
                            .i(head_out as i32),
                    );
                }
            }
            if ok {
                self.launch(
                    "argmax_f32",
                    1,
                    1024,
                    0,
                    Args::new()
                        .ptr(self.gemv_y.ptr)
                        .i(head_out as i32)
                        .ptr(self.mlm_arg.ptr)
                        .ptr(unsafe { (self.mlm_arg.ptr as *mut f32).add(1) as *mut c_void }),
                );
            }
            let mut graph: *mut c_void = std::ptr::null_mut();
            unsafe {
                if crate::ffi::hipStreamEndCapture(self.gpu.stream, &mut graph) != 0 || !ok {
                    return None;
                }
                let mut exec: *mut c_void = std::ptr::null_mut();
                if crate::ffi::hipGraphInstantiate(
                    &mut exec,
                    graph,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                ) != 0
                {
                    let _ = crate::ffi::hipGraphDestroy(graph);
                    return None;
                }
                let _ = crate::ffi::hipGraphDestroy(graph);
                self.mlm_graph = Some(exec);
            }
        }
        let exec = self.mlm_graph?;
        unsafe {
            if crate::ffi::hipGraphLaunch(exec, self.gpu.stream) != 0 {
                return None;
            }
        }
        self.gpu.sync().ok()?;
        let idx = self.mlm_arg.download::<i32>(1).ok()?;
        let mut oh = vec![0.0f32; head_out];
        let i = idx[0] as usize;
        if i < head_out {
            oh[i] = 1.0;
        }
        Some(oh)
    }

    /// Upload device pos for the graph token.
    fn mlm_set_pos(&mut self, pos: i32) -> bool {
        self.mlm_pos.upload(&[pos]).is_ok()
    }

    /// Upload BOTH rope table sets (sliding, full).
    fn mlm_rope_tables2(&mut self, cs_s: &[f32], sn_s: &[f32], cs_f: &[f32], sn_f: &[f32]) -> bool {
        self.mlm_cs.upload(cs_s).is_ok()
            && self.mlm_sn.upload(sn_s).is_ok()
            && self.mlm_cs2.upload(cs_f).is_ok()
            && self.mlm_sn2.upload(sn_f).is_ok()
    }

    fn mlm_logits(&mut self) -> Option<Vec<f32>> {
        let on = self.f32w.get("output_norm.weight")?;
        let head = self.q8.get("output.weight")?;
        self.norm_launch(self.mlm_h.ptr, on.ptr, self.mlm_n.ptr, head.in_dim);
        self.q8_launch(head, self.mlm_n.ptr, self.gemv_y.ptr);
        self.gpu.sync().ok()?;
        self.gemv_y.download::<f32>(head.out_dim).ok()
    }

    fn gemv_batch(&self, calls: &[(&str, &[f32])]) -> Vec<Option<Vec<f32>>> {
        // All launches queued stream-ordered, ONE sync, then downloads. Falls back to
        // per-call gemv only for keys not Q8-resident.
        if calls.len() > 4 || !calls.iter().all(|(k, _)| self.q8.contains_key(*k)) {
            return calls.iter().map(|(k, x)| self.gemv(k, x)).collect();
        }
        let mut metas = Vec::with_capacity(calls.len());
        for (i, (k, x)) in calls.iter().enumerate() {
            let e = &self.q8[*k];
            if x.len() != e.in_dim || e.out_dim > 16384 {
                return calls.iter().map(|(k, x)| self.gemv(k, x)).collect();
            }
            let xbuf = &self.batch_x[i];
            let ybuf = &self.batch_y[i];
            if xbuf.upload(x).is_err() {
                return calls.iter().map(|(k, x)| self.gemv(k, x)).collect();
            }
            self.q8_launch(e, xbuf.ptr, ybuf.ptr);
            metas.push(e.out_dim);
        }
        if self.gpu.sync().is_err() {
            return calls.iter().map(|_| None).collect();
        }
        metas
            .iter()
            .enumerate()
            .map(|(i, &n)| self.batch_y[i].download::<f32>(n).ok())
            .collect()
    }

    fn prefill_q8_gemm(&self, key: &str, xs: &[f32], m: usize) -> Option<Vec<f32>> {
        let e = self.q8.get(key)?;
        if xs.len() != m * e.in_dim || m > 256 || e.out_dim > 8192 {
            return None;
        }
        self.pf_x.upload(xs).ok()?;
        if self.mlm_int8 {
            let (xq, xd) = self.pf_quant(self.pf_x.ptr, m, e.in_dim);
            self.q8i_gemm(
                e.out_dim as u32,
                m as u32,
                Args::new()
                    .ptr(e.scales.ptr)
                    .ptr(e.quants.ptr)
                    .ptr(xq)
                    .ptr(xd)
                    .ptr(self.pf_y.ptr)
                    .i(e.in_dim as i32)
                    .i(e.out_dim as i32)
                    .i(m as i32),
            );
        } else {
            self.launch2(
                "q8_gemm_rows32",
                e.out_dim as u32,
                m.div_ceil(32) as u32,
                32,
                0,
                Args::new()
                    .ptr(e.scales.ptr)
                    .ptr(e.quants.ptr)
                    .ptr(self.pf_x.ptr)
                    .ptr(self.pf_y.ptr)
                    .i(e.in_dim as i32)
                    .i(e.out_dim as i32)
                    .i(m as i32),
            );
        }
        self.gpu.sync().ok()?;
        self.pf_y.download::<f32>(m * e.out_dim).ok()
    }

    fn moe_expert_ffn(&self, layer: usize, eid: usize, xs: &[f32], m: usize) -> Option<Vec<f32>> {
        let mo = self.moe6.get(&layer)?;
        if xs.len() != m * mo.hidden || m > 256 {
            return None;
        }
        self.pf_x.upload(xs).ok()?;
        let geb = mo.gate_eb as usize as *mut c_void;
        let deb = mo.down_eb as usize as *mut c_void;
        for (w, y) in [(&mo.gate, self.pf_a.ptr), (&mo.up, self.pf_b.ptr)] {
            self.launch2(
                "q6_gemm_rows",
                mo.eff as u32,
                m.div_ceil(16) as u32,
                32,
                0,
                Args::new()
                    .ptr(w.ptr)
                    .ptr(geb)
                    .i(eid as i32)
                    .ptr(self.pf_x.ptr)
                    .ptr(y)
                    .i(mo.hidden as i32)
                    .i(mo.eff as i32)
                    .i(m as i32),
            );
        }
        let n_act = m * mo.eff;
        self.launch(
            "moe_silu_mul",
            n_act.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.pf_a.ptr)
                .ptr(self.pf_b.ptr)
                .ptr(self.pf_x.ptr)
                .i(n_act as i32),
        );
        self.launch2(
            "q6_gemm_rows",
            mo.hidden as u32,
            m.div_ceil(16) as u32,
            32,
            0,
            Args::new()
                .ptr(mo.down.ptr)
                .ptr(deb)
                .i(eid as i32)
                .ptr(self.pf_x.ptr)
                .ptr(self.pf_y.ptr)
                .i(mo.eff as i32)
                .i(mo.hidden as i32)
                .i(m as i32),
        );
        self.gpu.sync().ok()?;
        self.pf_y.download::<f32>(m * mo.hidden).ok()
    }

    fn moe_expert_queue(
        &self,
        layer: usize,
        eid: usize,
        xs: &[f32],
        m: usize,
        dy_off: usize,
    ) -> bool {
        let Some(mo) = self.moe6.get(&layer) else {
            return false;
        };
        if xs.len() != m * mo.hidden || (dy_off + m) * mo.hidden > 2048 * 4096 {
            return false;
        }
        // per-call x staging at offset dy_off in pf_x (reuse pool rows; hidden<=2048, pool 256*8192)
        let xoff = dy_off * mo.hidden;
        if (xoff + m * mo.hidden) > 2048 * 4096 {
            return false;
        }
        let xptr = unsafe { (self.pf_x.ptr as *mut f32).add(xoff) } as *mut c_void;
        if self.gpu.upload_at(&self.pf_x, xoff * 4, xs).is_err() {
            return false;
        }
        let geb = mo.gate_eb as usize as *mut c_void;
        let deb = mo.down_eb as usize as *mut c_void;
        // pf_a/pf_b reused across queued experts — safe: the stream serializes
        // each expert's gate→silu→down chain before the next overwrites them.
        let a = self.pf_a.ptr;
        let b = self.pf_b.ptr;
        for (w, y) in [(&mo.gate, a), (&mo.up, b)] {
            self.launch2(
                "q6_gemm_rows",
                mo.eff as u32,
                m.div_ceil(16) as u32,
                32,
                0,
                Args::new()
                    .ptr(w.ptr)
                    .ptr(geb)
                    .i(eid as i32)
                    .ptr(xptr)
                    .ptr(y)
                    .i(mo.hidden as i32)
                    .i(mo.eff as i32)
                    .i(m as i32),
            );
        }
        let n_act = m * mo.eff;
        self.launch(
            "moe_silu_mul",
            n_act.div_ceil(256) as u32,
            256,
            0,
            Args::new().ptr(a).ptr(b).ptr(a).i(n_act as i32),
        );
        let dyp = unsafe { (self.pf_dy.ptr as *mut f32).add(dy_off * mo.hidden) } as *mut c_void;
        self.launch2(
            "q6_gemm_rows",
            mo.hidden as u32,
            m.div_ceil(16) as u32,
            32,
            0,
            Args::new()
                .ptr(mo.down.ptr)
                .ptr(deb)
                .i(eid as i32)
                .ptr(a)
                .ptr(dyp)
                .i(mo.eff as i32)
                .i(mo.hidden as i32)
                .i(m as i32),
        );
        true
    }

    fn moe_expert_queue_q8(
        &self,
        layer: usize,
        eid: usize,
        xs: &[f32],
        m: usize,
        dy_off: usize,
    ) -> bool {
        let Some(mo) = self.moe.get(&layer) else {
            return false;
        };
        if xs.len() != m * mo.hidden || (dy_off + m) * mo.hidden > 2048 * 4096 {
            return false;
        }
        let xoff = dy_off * mo.hidden;
        if (xoff + m * mo.hidden) > 2048 * 4096
            || self.gpu.upload_at(&self.pf_x, xoff * 4, xs).is_err()
        {
            return false;
        }
        let xptr = unsafe { (self.pf_x.ptr as *mut f32).add(xoff) } as *mut c_void;
        let nb = mo.hidden / 32;
        let eoff = eid * mo.eff * nb;
        let xqd = if self.mlm_int8 {
            Some(self.pf_quant(xptr, m, mo.hidden))
        } else {
            None
        };
        for (sb, qb2, y) in [
            (&mo.gate_s, &mo.gate_q, self.pf_a.ptr),
            (&mo.up_s, &mo.up_q, self.pf_b.ptr),
        ] {
            let sp = unsafe { (sb.ptr as *mut f32).add(eoff) } as *mut c_void;
            let qp = unsafe { (qb2.ptr as *mut i8).add(eoff * 32) } as *mut c_void;
            if let Some((xq, xd)) = xqd {
                self.q8i_gemm(
                    mo.eff as u32,
                    m as u32,
                    Args::new()
                        .ptr(sp)
                        .ptr(qp)
                        .ptr(xq)
                        .ptr(xd)
                        .ptr(y)
                        .i(mo.hidden as i32)
                        .i(mo.eff as i32)
                        .i(m as i32),
                );
            } else {
                self.launch2(
                    "q8_gemm_rows32",
                    mo.eff as u32,
                    m.div_ceil(32) as u32,
                    32,
                    0,
                    Args::new()
                        .ptr(sp)
                        .ptr(qp)
                        .ptr(xptr)
                        .ptr(y)
                        .i(mo.hidden as i32)
                        .i(mo.eff as i32)
                        .i(m as i32),
                );
            }
        }
        let n_act = m * mo.eff;
        self.launch(
            "moe_silu_mul",
            n_act.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.pf_a.ptr)
                .ptr(self.pf_b.ptr)
                .ptr(self.pf_b.ptr)
                .i(n_act as i32),
        );
        let nbd = mo.eff / 32;
        let ds = unsafe { (mo.down_s.ptr as *mut f32).add(eid * mo.hidden * nbd) } as *mut c_void;
        let dq =
            unsafe { (mo.down_q.ptr as *mut i8).add(eid * mo.hidden * nbd * 32) } as *mut c_void;
        let dyp = unsafe { (self.pf_dy.ptr as *mut f32).add(dy_off * mo.hidden) } as *mut c_void;
        if self.mlm_int8 {
            let (xq, xd) = self.pf_quant(self.pf_b.ptr, m, mo.eff);
            self.q8i_gemm(
                mo.hidden as u32,
                m as u32,
                Args::new()
                    .ptr(ds)
                    .ptr(dq)
                    .ptr(xq)
                    .ptr(xd)
                    .ptr(dyp)
                    .i(mo.eff as i32)
                    .i(mo.hidden as i32)
                    .i(m as i32),
            );
        } else {
            self.launch2(
                "q8_gemm_rows32",
                mo.hidden as u32,
                m.div_ceil(32) as u32,
                32,
                0,
                Args::new()
                    .ptr(ds)
                    .ptr(dq)
                    .ptr(self.pf_b.ptr)
                    .ptr(dyp)
                    .i(mo.eff as i32)
                    .i(mo.hidden as i32)
                    .i(m as i32),
            );
        }
        true
    }

    /// Whole-layer multi-expert FFN (Q6 native): xs_all = gathered token rows
    /// (expert-major), plan = (expert id, rows). 3 launches + 1 sync + 1 download.
    fn moe_layer_ffn(
        &self,
        layer: usize,
        plan: &[(i32, i32)],
        xs_all: &[f32],
        rows: usize,
    ) -> Option<Vec<f32>> {
        let mo = self.moe6.get(&layer)?;
        if xs_all.len() != rows * mo.hidden || rows > 2048 {
            return None;
        }
        // entry table: per 16-row tile of each expert group
        let mut tab: Vec<[i32; 4]> = Vec::new();
        let mut off = 0i32;
        for &(eid, me) in plan {
            let mut t0 = 0i32;
            while t0 < me {
                let tm = (me - t0).min(16);
                tab.push([eid, off + t0, off + t0, tm]);
                t0 += 16;
            }
            off += me;
        }
        if tab.len() > 512 {
            return None;
        }
        self.pf_x.upload(xs_all).ok()?;
        self.pf_tab.upload(&tab).ok()?;
        let geb = mo.gate_eb as usize as *mut c_void;
        let deb = mo.down_eb as usize as *mut c_void;
        let gy = tab.len() as u32;
        for (w, y) in [(&mo.gate, self.pf_a.ptr), (&mo.up, self.pf_b.ptr)] {
            self.launch2(
                "q6_gemm_moe",
                mo.eff as u32,
                gy,
                32,
                0,
                Args::new()
                    .ptr(w.ptr)
                    .ptr(geb)
                    .ptr(self.pf_tab.ptr)
                    .ptr(self.pf_x.ptr)
                    .ptr(y)
                    .i(mo.hidden as i32)
                    .i(mo.eff as i32),
            );
        }
        let n_act = rows * mo.eff;
        self.launch(
            "moe_silu_mul",
            n_act.div_ceil(256) as u32,
            256,
            0,
            Args::new()
                .ptr(self.pf_a.ptr)
                .ptr(self.pf_b.ptr)
                .ptr(self.pf_act.ptr)
                .i(n_act as i32),
        );
        self.launch2(
            "q6_gemm_moe",
            mo.hidden as u32,
            gy,
            32,
            0,
            Args::new()
                .ptr(mo.down.ptr)
                .ptr(deb)
                .ptr(self.pf_tab.ptr)
                .ptr(self.pf_act.ptr)
                .ptr(self.pf_dy.ptr)
                .i(mo.eff as i32)
                .i(mo.hidden as i32),
        );
        self.gpu.sync().ok()?;
        self.pf_dy.download::<f32>(rows * mo.hidden).ok()
    }

    fn lm_head_argmax_rows(&self, key: &str, xs: &[f32], m: usize) -> Option<Vec<u32>> {
        let e = self.q8.get(key)?;
        if xs.len() != m * e.in_dim || m > 16 {
            return None;
        }
        self.pf_x.upload(xs).ok()?;
        let nb = e.in_dim / 32;
        let chunk = 8192usize;
        let mut best = vec![(f32::NEG_INFINITY, 0u32); m];
        let mut base = 0usize;
        while base < e.out_dim {
            let oc = (e.out_dim - base).min(chunk);
            let sp = unsafe { (e.scales.ptr as *mut f32).add(base * nb) } as *mut c_void;
            let qp = unsafe { (e.quants.ptr as *mut i8).add(base * nb * 32) } as *mut c_void;
            if self.mlm_int8 {
                let (xq, xd) = self.pf_quant(self.pf_x.ptr, m, e.in_dim);
                self.q8i_gemm(
                    oc as u32,
                    m as u32,
                    Args::new()
                        .ptr(sp)
                        .ptr(qp)
                        .ptr(xq)
                        .ptr(xd)
                        .ptr(self.pf_y.ptr)
                        .i(e.in_dim as i32)
                        .i(oc as i32)
                        .i(m as i32),
                );
            } else {
                self.launch2(
                    "q8_gemm_rows32",
                    oc as u32,
                    m.div_ceil(32) as u32,
                    32,
                    0,
                    Args::new()
                        .ptr(sp)
                        .ptr(qp)
                        .ptr(self.pf_x.ptr)
                        .ptr(self.pf_y.ptr)
                        .i(e.in_dim as i32)
                        .i(oc as i32)
                        .i(m as i32),
                );
            }
            self.gpu.sync().ok()?;
            let y = self.pf_y.download::<f32>(m * oc).ok()?;
            for t in 0..m {
                for r in 0..oc {
                    let v = y[t * oc + r];
                    if v > best[t].0 {
                        best[t] = (v, (base + r) as u32);
                    }
                }
            }
            base += oc;
        }
        Some(best.into_iter().map(|(_, i)| i).collect())
    }

    fn mlm_attn_prefill(
        &mut self,
        layer: usize,
        xs: &[f32],
        m: usize,
        base: usize,
        win: usize,
        cs: &[f32],
        sn: &[f32],
    ) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        self.gpu.upload_at(&self.pf_x, 0, xs).ok()?;
        let _ = xs;
        let (out, kk, vv) = self.attn_prefill_inner(layer, m, base, win, cs, sn, self.pf_x.ptr)?;
        Some((out, kk, vv))
    }

    fn pf_begin(&mut self, h: &[f32], m: usize) -> bool {
        m <= 512 && self.gpu.upload_at(&self.pf_h, 0, h).is_ok()
    }

    fn pf_attn(
        &mut self,
        l: usize,
        m: usize,
        base: usize,
        win: usize,
        cs: &[f32],
        sn: &[f32],
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let an = self.f32w.get(&format!("blk.{l}.attn_norm.weight"))?;
        let hidden = an.bytes / 4;
        self.launch(
            "rmsnorm",
            m as u32,
            256,
            0,
            Args::new()
                .ptr(self.pf_h.ptr)
                .ptr(an.ptr)
                .ptr(self.pf_n.ptr)
                .i(hidden as i32)
                .i(1)
                .f(1e-6),
        );
        // attention reads normed acts from pf_n via the resident path
        let (out, kk, vv) = self.attn_prefill_inner(l, m, base, win, cs, sn, self.pf_n.ptr)?;
        let _ = out;
        Some((kk, vv))
    }

    fn pf_router(&mut self, l: usize, m: usize, ne: usize) -> Option<Vec<f32>> {
        let fnw = self.f32w.get(&format!("blk.{l}.ffn_norm.weight"))?;
        let wgi = self.f32w.get(&format!("blk.{l}.ffn_gate_inp.weight"))?;
        let hidden = fnw.bytes / 4;
        self.launch(
            "rmsnorm",
            m as u32,
            256,
            0,
            Args::new()
                .ptr(self.pf_h.ptr)
                .ptr(fnw.ptr)
                .ptr(self.pf_n.ptr)
                .i(hidden as i32)
                .i(1)
                .f(1e-6),
        );
        self.launch2(
            "f32_gemv_rows",
            ne as u32,
            m as u32,
            32,
            0,
            Args::new()
                .ptr(wgi.ptr)
                .ptr(self.pf_n.ptr)
                .ptr(self.pf_y.ptr)
                .i(hidden as i32)
                .i(ne as i32)
                .i(m as i32),
        );
        self.gpu.sync().ok()?;
        self.pf_y.download::<f32>(m * ne).ok()
    }

    fn pf_moe(
        &mut self,
        l: usize,
        m: usize,
        plan: &[(usize, usize, usize)],
        st: &[i32],
        w: &[f32],
    ) -> bool {
        self.moe_layer_dev_inner(l, m, plan, st, w, self.pf_n.ptr, self.pf_h.ptr)
            .is_some()
    }

    fn pf_end(&mut self, m: usize) -> Option<Vec<f32>> {
        let hidden = 2304usize;
        self.gpu.sync().ok()?;
        self.pf_h.download::<f32>(m * hidden).ok()
    }

    fn moe_expert_flush(&self, rows: usize, hidden: usize) -> Option<Vec<f32>> {
        self.gpu.sync().ok()?;
        self.pf_dy.download::<f32>(rows * hidden).ok()
    }

    fn moe_layer_q8_dev(
        &self,
        layer: usize,
        xs: &[f32],
        m: usize,
        plan: &[(usize, usize, usize)],
        slot_tok: &[i32],
        wslot: &[f32],
    ) -> Option<Vec<f32>> {
        if xs.len() != m * self.moe.get(&layer)?.hidden {
            return None;
        }
        self.gpu.upload_at(&self.pf_x, 0, xs).ok()?;
        self.moe_layer_dev_inner(
            layer,
            m,
            plan,
            slot_tok,
            wslot,
            self.pf_x.ptr,
            std::ptr::null_mut(),
        )?;
        let hidden = self.moe.get(&layer)?.hidden;
        self.gpu.sync().ok()?;
        self.pf_y.download::<f32>(m * hidden).ok()
    }

    fn gemv(&self, key: &str, x: &[f32]) -> Option<Vec<f32>> {
        // Per-weight GEMV on a resident Q6_K / Q4_0 weight, reusing the exact kernels
        // + launch config as `lm_head`. ROCm's stream-ordered launch has near-zero
        // per-call overhead (vs wgpu's ~21µs), so this suits the ~hundreds of expert
        // GEMVs/token an MoE forward issues. Persistent gemv_x/gemv_y scratch (no
        // per-call alloc — Dbuf doesn't free). Returns None ⇒ caller uses CPU.
        if let Some(e) = self.q6.get(key) {
            if x.len() != e.in_dim || e.in_dim > GEMV_MAX_IN || e.out_dim > GEMV_MAX_OUT {
                return None;
            }
            self.gemv_x.upload(x).ok()?;
            self.launch(
                "q6_gemv",
                e.out_dim.div_ceil(16) as u32,
                256,
                0,
                Args::new()
                    .ptr(e.scales.ptr)
                    .ptr(e.ql.ptr)
                    .ptr(e.qh.ptr)
                    .ptr(self.gemv_x.ptr)
                    .ptr(self.gemv_y.ptr)
                    .i(e.in_dim as i32)
                    .i(e.out_dim as i32),
            );
            self.gpu.sync().ok()?;
            return self.gemv_y.download::<f32>(e.out_dim).ok();
        }
        if let Some(e) = self.q4.get(key) {
            if x.len() != e.in_dim || e.in_dim > GEMV_MAX_IN || e.out_dim > GEMV_MAX_OUT {
                return None;
            }
            self.gemv_x.upload(x).ok()?;
            self.launch(
                "q4_gemv",
                e.out_dim as u32,
                32,
                0,
                Args::new()
                    .ptr(e.scales.ptr)
                    .ptr(e.quants.ptr)
                    .ptr(self.gemv_x.ptr)
                    .ptr(self.gemv_y.ptr)
                    .i(e.in_dim as i32)
                    .i(e.out_dim as i32),
            );
            self.gpu.sync().ok()?;
            return self.gemv_y.download::<f32>(e.out_dim).ok();
        }
        if let Some(e) = self.q8.get(key) {
            if x.len() != e.in_dim || e.in_dim > GEMV_MAX_IN || e.out_dim > GEMV_MAX_OUT {
                return None;
            }
            self.gemv_x.upload(x).ok()?;
            self.launch(
                "q8_0_gemv",
                e.out_dim as u32,
                32,
                0,
                Args::new()
                    .ptr(e.scales.ptr)
                    .ptr(e.quants.ptr)
                    .ptr(self.gemv_x.ptr)
                    .ptr(self.gemv_y.ptr)
                    .i(e.in_dim as i32)
                    .i(e.out_dim as i32),
            );
            self.gpu.sync().ok()?;
            return self.gemv_y.download::<f32>(e.out_dim).ok();
        }
        None
    }

    fn resident_count(&self) -> usize {
        self.q4.len() + self.q6.len() + self.q8.len()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn upload_f32(&mut self, key: &str, data: &[f32]) {
        if let Ok(b) = self.gpu.upload_new(data) {
            self.f32w.insert(key.to_string(), b);
        }
    }

    fn configure_decode(&mut self, cfg: GpuDecodeConfig) -> bool {
        let hidden = cfg.hidden;
        let n_heads = cfg.n_heads;
        let max_hd = cfg.layers.iter().map(|l| l.head_dim).max().unwrap_or(0);
        let q_dim = n_heads * max_hd;
        let kv_dim_max = cfg
            .layers
            .iter()
            .map(|l| l.n_kv * l.head_dim)
            .max()
            .unwrap_or(0);
        let mk = |n: usize| self.gpu.alloc((n.max(1)) * 4).expect("rocm scratch");
        // KV cache element size: 2 bytes (f16) when STRIX_F16_KV, else 4 (f32).
        let kvb = if self.kv_f16 { 2 } else { 4 };
        let mkkv = |n: usize| self.gpu.alloc((n.max(1)) * kvb).expect("rocm kv");
        // Largest matmul input (down-proj in=ffn) → max Q8 blocks.
        let max_in = cfg.ffn.max(hidden).max(q_dim);
        let nb_max = max_in.div_ceil(32);
        let ones_buf = mk((max_hd / 2).max(1));
        ones_buf.upload(&vec![1.0f32; (max_hd / 2).max(1)]).ok();
        let k_cache = cfg
            .layers
            .iter()
            .map(|l| mkkv(l.n_kv * cfg.max_seq * l.head_dim))
            .collect();
        let v_cache = cfg
            .layers
            .iter()
            .map(|l| mkkv(l.n_kv * cfg.max_seq * l.head_dim))
            .collect();
        self.scratch = Some(Scratch {
            h: mk(hidden),
            xn: mk(hidden),
            q: mk(q_dim),
            q2: mk(q_dim),
            k: mk(kv_dim_max),
            v: mk(kv_dim_max),
            attn: mk(q_dim),
            t_hidden: mk(hidden),
            gate: mk(cfg.ffn),
            up: mk(cfg.ffn),
            logits: mk(cfg.vocab),
            argmax_out: mk(2), // [0]=idx (i32), [1]=val (f32)
            ones: ones_buf,
            xq_lo: self.gpu.alloc(nb_max * 16).expect("xq_lo"), // 4 char4/block
            xq_hi: self.gpu.alloc(nb_max * 16).expect("xq_hi"),
            xq_d: mk(nb_max),
            xq_sum: mk(nb_max),

            attn_part: mk(n_heads * N_SPLIT_MAX * max_hd),
            attn_pmax: mk(n_heads * N_SPLIT_MAX),
            attn_psum: mk(n_heads * N_SPLIT_MAX),
            p_h: mk(M_CHUNK * hidden),
            p_xn: mk(M_CHUNK * hidden),
            p_q: mk(M_CHUNK * q_dim),
            p_q2: mk(M_CHUNK * q_dim),
            p_k: mk(M_CHUNK * kv_dim_max),
            p_k2: mk(M_CHUNK * kv_dim_max),
            p_v: mk(M_CHUNK * kv_dim_max),
            p_v2: mk(M_CHUNK * kv_dim_max),
            p_attn: mk(M_CHUNK * q_dim),
            p_th: mk(M_CHUNK * hidden),
            p_gate: mk(M_CHUNK * cfg.ffn),
            p_up: mk(M_CHUNK * cfg.ffn),
            p_act: mk(M_CHUNK * cfg.ffn),
            p_xqlo: self.gpu.alloc(M_CHUNK * nb_max * 16).expect("p_xqlo"),
            p_xqhi: self.gpu.alloc(M_CHUNK * nb_max * 16).expect("p_xqhi"),
            p_xqd: mk(M_CHUNK * nb_max),
            p_xqsum: mk(M_CHUNK * nb_max),
            // GEMM-attention scores scratch — only sized when the path is enabled
            // (n_heads*M_CHUNK*max_seq f32 can be 10s-100s of MB).
            p_scores: if std::env::var("STRIX_GEMM_SDPA").is_ok() {
                // f16 scores (2 bytes/elem): n_heads * M_CHUNK * max_seq.
                self.gpu
                    .alloc(n_heads * M_CHUNK * cfg.max_seq * 2)
                    .expect("p_scores")
            } else {
                self.gpu.alloc(2).expect("p_scores")
            },
            p_rowsum: mk(n_heads * M_CHUNK),
            k_cache,
            v_cache,
        });
        self.cfg = Some(cfg);
        #[cfg(feature = "npu")]
        self.npu_init();
        true
    }

    fn decode_step(&mut self, h: &[f32], pos: usize) -> Option<Vec<f32>> {
        let cfg = self.cfg.as_ref()?;
        let s = self.scratch.as_ref()?;
        let eps = cfg.eps;
        let hidden = cfg.hidden;
        let n_heads = cfg.n_heads;
        let dc = |g: usize| g.div_ceil(64) as u32;
        let q4 = |n: &str| {
            self.q4
                .get(n)
                .unwrap_or_else(|| panic!("rocm: missing Q4 {n}"))
        };
        let _ = &self.q6;
        let ff = |n: &str| {
            self.f32w
                .get(n)
                .unwrap_or_else(|| panic!("rocm: missing f32 {n}"))
                .ptr
        };
        let n_layers = std::env::var("STRIX_LAYERS")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(cfg.n_layers);
        let skip_sdpa = matches!(std::env::var("STRIX_SKIP_SDPA").as_deref(), Ok(s) if !s.is_empty() && s != "global" && s != "local");

        s.h.upload(h).ok()?;

        // Q8-quantize an activation into the shared xq buffers (one xquant block
        // per 32-element block).
        let quantize = |src: *mut c_void, in_dim: usize| {
            self.launch(
                "xquant",
                (in_dim / 32) as u32,
                32,
                0,
                Args::new()
                    .ptr(src)
                    .ptr(s.xq_lo.ptr)
                    .ptr(s.xq_hi.ptr)
                    .ptr(s.xq_d.ptr)
                    .ptr(s.xq_sum.ptr)
                    .i(in_dim as i32),
            );
        };
        let rmsnorm_quantize = |src: *mut c_void, weight: *mut c_void, in_dim: usize| {
            self.launch(
                "rmsnorm_xquant",
                1,
                256,
                0,
                Args::new()
                    .ptr(src)
                    .ptr(weight)
                    .ptr(s.xq_lo.ptr)
                    .ptr(s.xq_hi.ptr)
                    .ptr(s.xq_d.ptr)
                    .ptr(s.xq_sum.ptr)
                    .i(in_dim as i32)
                    .i(1)
                    .f(eps),
            );
        };
        let geglu_quantize = |gate: *mut c_void, up: *mut c_void, n: usize| {
            self.launch(
                "geglu_xquant",
                (n / 32) as u32,
                32,
                0,
                Args::new()
                    .ptr(gate)
                    .ptr(up)
                    .ptr(s.xq_lo.ptr)
                    .ptr(s.xq_hi.ptr)
                    .ptr(s.xq_d.ptr)
                    .ptr(s.xq_sum.ptr)
                    .i(n as i32),
            );
        };
        // Q4 GEMV via dp4a, reading the pre-quantized xq buffers. ~85-88 GB/s vs
        // ~74 for the f32 dequant path (the dominant lever to beat llama.cpp).
        let q4dp = |w: &ResQ4, y: *mut c_void| {
            self.launch(
                "q4_gemv_dp",
                (w.out_dim.div_ceil(8)) as u32,
                256,
                0,
                Args::new()
                    .ptr(w.scales.ptr)
                    .ptr(w.quants.ptr)
                    .ptr(s.xq_lo.ptr)
                    .ptr(s.xq_hi.ptr)
                    .ptr(s.xq_d.ptr)
                    .ptr(s.xq_sum.ptr)
                    .ptr(y)
                    .i(w.in_dim as i32)
                    .i(w.out_dim as i32),
            );
        };
        let q4dp2 = |w0: &ResQ4, y0: *mut c_void, w1: &ResQ4, y1: *mut c_void| {
            debug_assert_eq!(w0.in_dim, w1.in_dim);
            self.launch(
                "q4_gemv_dp2",
                (w0.out_dim.div_ceil(8) + w1.out_dim.div_ceil(8)) as u32,
                256,
                0,
                Args::new()
                    .ptr(w0.scales.ptr)
                    .ptr(w0.quants.ptr)
                    .ptr(y0)
                    .i(w0.out_dim as i32)
                    .ptr(w1.scales.ptr)
                    .ptr(w1.quants.ptr)
                    .ptr(y1)
                    .i(w1.out_dim as i32)
                    .ptr(s.xq_lo.ptr)
                    .ptr(s.xq_hi.ptr)
                    .ptr(s.xq_d.ptr)
                    .ptr(s.xq_sum.ptr)
                    .i(w0.in_dim as i32),
            );
        };
        let q4dp3 = |w0: &ResQ4,
                     y0: *mut c_void,
                     w1: &ResQ4,
                     y1: *mut c_void,
                     w2: &ResQ4,
                     y2: *mut c_void| {
            debug_assert_eq!(w0.in_dim, w1.in_dim);
            debug_assert_eq!(w0.in_dim, w2.in_dim);
            self.launch(
                "q4_gemv_dp3",
                (w0.out_dim.div_ceil(8) + w1.out_dim.div_ceil(8) + w2.out_dim.div_ceil(8)) as u32,
                256,
                0,
                Args::new()
                    .ptr(w0.scales.ptr)
                    .ptr(w0.quants.ptr)
                    .ptr(y0)
                    .i(w0.out_dim as i32)
                    .ptr(w1.scales.ptr)
                    .ptr(w1.quants.ptr)
                    .ptr(y1)
                    .i(w1.out_dim as i32)
                    .ptr(w2.scales.ptr)
                    .ptr(w2.quants.ptr)
                    .ptr(y2)
                    .i(w2.out_dim as i32)
                    .ptr(s.xq_lo.ptr)
                    .ptr(s.xq_hi.ptr)
                    .ptr(s.xq_d.ptr)
                    .ptr(s.xq_sum.ptr)
                    .i(w0.in_dim as i32),
            );
        };

        for l in 0..n_layers {
            let lc = &cfg.layers[l];
            let hd = lc.head_dim;
            let n_kv = lc.n_kv;
            let kv_dim = n_kv * hd;
            let groups = (n_heads / n_kv.max(1)).max(1);
            let scale = if cfg.attn_rsqrt {
                1.0 / (hd as f32).sqrt()
            } else {
                1.0
            };
            let pf = |name: &str| format!("blk.{l}.{name}");

            // attn rmsnorm directly to Q8; q/k/v all read the same xq buffers.
            rmsnorm_quantize(s.h.ptr, ff(&pf("attn_norm.weight")), hidden);
            let qw = q4(&pf("attn_q.weight"));
            let kw = q4(&pf("attn_k.weight"));
            if !lc.k_eq_v {
                q4dp3(qw, s.q.ptr, kw, s.k.ptr, q4(&pf("attn_v.weight")), s.v.ptr);
            } else {
                q4dp2(qw, s.q.ptr, kw, s.k.ptr);
            }
            let v_src = if lc.k_eq_v { s.k.ptr } else { s.v.ptr };
            // q/k/v post-process + KV append at slot `pos`.
            let ropef = if lc.is_local {
                s.ones.ptr
            } else {
                self.f32w
                    .get("rope_freqs.weight")
                    .map(|d| d.ptr)
                    .unwrap_or(s.ones.ptr)
            };
            let off = pos * kv_dim * if self.kv_f16 { 2 } else { 4 };
            let kdst = unsafe { (s.k_cache[l].ptr as *mut u8).add(off) as *mut c_void };
            let vdst = unsafe { (s.v_cache[l].ptr as *mut u8).add(off) as *mut c_void };
            self.launch(
                "qkv_post",
                (n_heads + n_kv) as u32,
                256,
                0,
                Args::new()
                    .ptr(s.q.ptr)
                    .ptr(s.k.ptr)
                    .ptr(v_src)
                    .ptr(ff(&pf("attn_q_norm.weight")))
                    .ptr(ff(&pf("attn_k_norm.weight")))
                    .ptr(ropef)
                    .ptr(s.q2.ptr)
                    .ptr(kdst)
                    .ptr(vdst)
                    .i(hd as i32)
                    .i(n_heads as i32)
                    .i(n_kv as i32)
                    .i(pos as i32)
                    .f(lc.rope_theta)
                    .f(eps)
                    .i(if cfg.norm_v { 1 } else { 0 })
                    .i(if self.kv_f16 { 1 } else { 0 }),
            );
            // sdpa. Local (sliding-window) layers attend only the last `n_swa`
            // keys: offset the K/V base to the window start and shorten `len`
            // (query at pos `pos` sees keys [pos-n_swa+1, pos] = cache suffix).
            // Flash-decoding (split keys) wins only at long context.
            let full_len = pos + 1;
            let win = if lc.is_local && cfg.n_swa > 0 {
                cfg.n_swa
            } else {
                usize::MAX
            };
            let win_start = full_len.saturating_sub(win);
            let len = (full_len - win_start) as i32;
            let koff = (win_start * kv_dim * if self.kv_f16 { 2 } else { 4 }) as usize;
            let kbase = unsafe { (s.k_cache[l].ptr as *mut u8).add(koff) as *mut c_void };
            let vbase = unsafe { (s.v_cache[l].ptr as *mut u8).add(koff) as *mut c_void };
            // context-adaptive key-split: ~64 keys/chunk, clamped [8, N_SPLIT_MAX];
            // pinned by STRIX_N_SPLIT (self.n_split>0). Buffers are sized for the cap.
            let ns = if self.n_split > 0 {
                self.n_split
            } else {
                (len as usize).div_ceil(64).clamp(8, N_SPLIT_MAX)
            };
            if !skip_sdpa {
                if len > 1024 {
                    self.launch2(
                        "sdpa_split",
                        n_heads as u32,
                        ns as u32,
                        128,
                        0,
                        Args::new()
                            .ptr(s.q2.ptr)
                            .ptr(kbase)
                            .ptr(vbase)
                            .ptr(s.attn_part.ptr)
                            .ptr(s.attn_pmax.ptr)
                            .ptr(s.attn_psum.ptr)
                            .i(hd as i32)
                            .i(len)
                            .i(groups as i32)
                            .i(n_kv as i32)
                            .f(scale)
                            .i(ns as i32)
                            .i(if self.kv_f16 { 1 } else { 0 }),
                    );
                    self.launch(
                        "sdpa_combine",
                        n_heads as u32,
                        128,
                        0,
                        Args::new()
                            .ptr(s.attn_part.ptr)
                            .ptr(s.attn_pmax.ptr)
                            .ptr(s.attn_psum.ptr)
                            .ptr(s.attn.ptr)
                            .i(hd as i32)
                            .i(ns as i32),
                    );
                } else {
                    self.launch(
                        "sdpa",
                        n_heads as u32,
                        256,
                        0,
                        Args::new()
                            .ptr(s.q2.ptr)
                            .ptr(kbase)
                            .ptr(vbase)
                            .ptr(s.attn.ptr)
                            .i(hd as i32)
                            .i(len)
                            .i(groups as i32)
                            .i(n_kv as i32)
                            .f(scale)
                            .i(if self.kv_f16 { 1 } else { 0 }),
                    );
                }
            }
            // attn output -> t_hidden
            let o = q4(&pf("attn_output.weight"));
            quantize(s.attn.ptr, o.in_dim);
            q4dp(o, s.t_hidden.ptr);
            // h = h + rmsnorm(t_hidden)*post_attn_w
            self.launch(
                "addnorm",
                1,
                256,
                0,
                Args::new()
                    .ptr(s.h.ptr)
                    .ptr(s.t_hidden.ptr)
                    .ptr(ff(&pf("post_attention_norm.weight")))
                    .i(hidden as i32)
                    .f(eps)
                    .f(1.0),
            );
            // ffn rmsnorm h->xn, then grid-parallel Q8 quantize (gate/up read xn)
            rmsnorm_quantize(s.h.ptr, ff(&pf("ffn_norm.weight")), hidden);
            q4dp2(
                q4(&pf("ffn_gate.weight")),
                s.gate.ptr,
                q4(&pf("ffn_up.weight")),
                s.up.ptr,
            );
            // GeGLU directly to Q8 for the down projection.
            let down = q4(&pf("ffn_down.weight"));
            geglu_quantize(s.gate.ptr, s.up.ptr, down.in_dim);
            q4dp(down, s.t_hidden.ptr);
            // h = (h + rmsnorm(t_hidden)*post_ffw_w) * output_scale
            self.launch(
                "addnorm",
                1,
                256,
                0,
                Args::new()
                    .ptr(s.h.ptr)
                    .ptr(s.t_hidden.ptr)
                    .ptr(ff(&pf("post_ffw_norm.weight")))
                    .i(hidden as i32)
                    .f(eps)
                    .f(lc.output_scale),
            );
        }

        // final norm + lm_head (Q6_K target / Q4_0 draft) + softcap
        let skip_lm = std::env::var("STRIX_SKIP_LM").is_ok();
        if !skip_lm {
            self.launch(
                "rmsnorm",
                1,
                256,
                0,
                Args::new()
                    .ptr(s.h.ptr)
                    .ptr(ff("output_norm.weight"))
                    .ptr(s.xn.ptr)
                    .i(hidden as i32)
                    .i(1)
                    .f(eps),
            );
            self.lm_head(s.xn.ptr, s.logits.ptr);
        }
        // Greedy fast path: softcap is monotone (cap·tanh(x/cap)), so argmax(raw)
        // == argmax(capped). Do the argmax on-device and download just the winner —
        // skips the softcap pass and the vocab-wide (~1 MB/token) DtoH copy.
        if self.want_argmax {
            self.launch(
                "argmax_f32",
                1,
                1024,
                0,
                Args::new()
                    .ptr(s.logits.ptr)
                    .i(cfg.vocab as i32)
                    .ptr(s.argmax_out.ptr)
                    .ptr(unsafe { (s.argmax_out.ptr as *mut f32).add(1) as *mut c_void }),
            );
            self.gpu.sync().ok()?;
            let idx = s.argmax_out.download::<i32>(1).ok()?;
            return Some(vec![idx[0] as f32]);
        }
        if !skip_lm && cfg.final_softcap > 0.0 {
            self.launch(
                "softcap",
                dc(cfg.vocab),
                64,
                0,
                Args::new()
                    .ptr(s.logits.ptr)
                    .i(cfg.vocab as i32)
                    .f(cfg.final_softcap),
            );
        }

        self.gpu.sync().ok()?;
        s.logits.download::<f32>(cfg.vocab).ok()
    }

    fn decode_step_argmax(&mut self, h: &[f32], pos: usize) -> Option<u32> {
        self.want_argmax = true;
        let r = self.decode_step(h, pos);
        self.want_argmax = false;
        r.map(|v| v[0] as u32)
    }

    fn prefill_max(&self) -> usize {
        if self.cfg.is_some() {
            M_CHUNK
        } else {
            0
        }
    }

    fn verify(&mut self, h: &[f32], start_pos: usize, m: usize) -> Option<Vec<f32>> {
        self.verify_all = true;
        let r = self.prefill(h, start_pos, m);
        self.verify_all = false;
        r
    }

    fn prefill(&mut self, h: &[f32], start_pos: usize, m: usize) -> Option<Vec<f32>> {
        let all_logits = self.verify_all;
        let cfg = self.cfg.as_ref()?;
        let s = self.scratch.as_ref()?;
        if m == 0 || m > M_CHUNK {
            return None;
        }
        let eps = cfg.eps;
        let hidden = cfg.hidden;
        let n_heads = cfg.n_heads;
        let dc = |g: usize| g.div_ceil(64) as u32;
        let q4 = |n: &str| {
            self.q4
                .get(n)
                .unwrap_or_else(|| panic!("rocm: missing Q4 {n}"))
        };
        let _ = &self.q6;
        let ff = |n: &str| {
            self.f32w
                .get(n)
                .unwrap_or_else(|| panic!("rocm: missing f32 {n}"))
                .ptr
        };
        let n_layers = std::env::var("STRIX_LAYERS")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(cfg.n_layers);
        #[cfg(feature = "npu")]
        let npu_active = self.npu.is_some()
            || self.npu_down.is_some()
            || self.npu_o.is_some()
            || self.npu_q.is_some()
            || self.npu_q_g.is_some()
            || self.npu_o_g.is_some();
        #[cfg(not(feature = "npu"))]
        let npu_active = false;

        s.p_h.upload(&h[..m * hidden]).ok()?;

        // Quantize a batched activation X[rows, k] (row-major) into the shared
        // prefill xq buffers (one xquant block per 32-block; rows*k/32 blocks).
        let quantize = |src: *mut c_void, n: usize| {
            self.launch(
                "xquant",
                (n / 32) as u32,
                32,
                0,
                Args::new()
                    .ptr(src)
                    .ptr(s.p_xqlo.ptr)
                    .ptr(s.p_xqhi.ptr)
                    .ptr(s.p_xqd.ptr)
                    .ptr(s.p_xqsum.ptr)
                    .i(n as i32),
            );
        };
        let rmsnorm_quantize =
            |src: *mut c_void, weight: *mut c_void, rows: usize, in_dim: usize| {
                self.launch(
                    "rmsnorm_xquant",
                    rows as u32,
                    256,
                    0,
                    Args::new()
                        .ptr(src)
                        .ptr(weight)
                        .ptr(s.p_xqlo.ptr)
                        .ptr(s.p_xqhi.ptr)
                        .ptr(s.p_xqd.ptr)
                        .ptr(s.p_xqsum.ptr)
                        .i(in_dim as i32)
                        .i(1)
                        .f(eps),
                );
            };
        let geglu_quantize = |gate: *mut c_void, up: *mut c_void, n: usize| {
            self.launch(
                "geglu_xquant",
                (n / 32) as u32,
                32,
                0,
                Args::new()
                    .ptr(gate)
                    .ptr(up)
                    .ptr(s.p_xqlo.ptr)
                    .ptr(s.p_xqhi.ptr)
                    .ptr(s.p_xqd.ptr)
                    .ptr(s.p_xqsum.ptr)
                    .i(n as i32),
            );
        };
        // Batched GEMM Y[m,out]=X[m,in]@W. WMMA matrix-core kernel by default
        // (~1.7-2.3× the dp4a one): output tile BN=128 rows × BM=64 tokens.
        // STRIX_DP4A_GEMM forces the dp4a path (64×64 tile). For large-K/small-N
        // GEMMs (ffn_down) the grid is too small to hide latency → split-K
        // (q4_gemm_w_sk) accumulates partials over gridDim.z into a pre-zeroed y.
        let use_dp4a = std::env::var("STRIX_DP4A_GEMM").is_ok();
        let no_sk = std::env::var("STRIX_NO_SPLITK").is_ok();
        let gemm_kernel = if use_dp4a { "q4_gemm" } else { "q4_gemm_w" };
        let n_tile = if use_dp4a { 64usize } else { 128usize };
        let q4gemm = |w: &ResQ4, y: *mut c_void| {
            let nb = w.in_dim / 32;
            // split-K only on the WMMA path for large K (only ffn_down qualifies)
            let sk = if !use_dp4a && !no_sk && w.in_dim >= 12288 {
                let want = std::env::var("STRIX_SK")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok());
                // empirically (gfx1150, optimized double-buffered kernel) sk≈5-6 beats
                // 8 for ffn_down (N=3840): fewer atomicAdd-contending slices once each
                // block is efficient. Prefer 5/6, fall back to whatever divides nb.
                match want {
                    Some(s) if s >= 1 && nb % s == 0 => s,
                    _ => [5usize, 6, 8, 10, 4, 2]
                        .into_iter()
                        .find(|s| nb % s == 0)
                        .unwrap_or(1),
                }
            } else {
                1
            };
            if sk > 1 {
                let _ = self.gpu.zero(y, m * w.out_dim * 4);
                self.launch3(
                    "q4_gemm_w_sk",
                    w.out_dim.div_ceil(128) as u32,
                    m.div_ceil(128) as u32,
                    sk as u32,
                    256,
                    0,
                    Args::new()
                        .ptr(w.scales.ptr)
                        .ptr(w.quants.ptr)
                        .ptr(s.p_xqlo.ptr)
                        .ptr(s.p_xqhi.ptr)
                        .ptr(s.p_xqd.ptr)
                        .ptr(s.p_xqsum.ptr)
                        .ptr(y)
                        .i(w.in_dim as i32)
                        .i(w.out_dim as i32)
                        .i(m as i32)
                        .i(w.out_dim as i32),
                );
                return;
            }
            // q4_gemm_w uses BM=128 (token tile); the dp4a path keeps BM=64.
            let bm_tile = if use_dp4a { 64usize } else { 128usize };
            self.launch2(
                gemm_kernel,
                w.out_dim.div_ceil(n_tile) as u32,
                m.div_ceil(bm_tile) as u32,
                256,
                0,
                Args::new()
                    .ptr(w.scales.ptr)
                    .ptr(w.quants.ptr)
                    .ptr(s.p_xqlo.ptr)
                    .ptr(s.p_xqhi.ptr)
                    .ptr(s.p_xqd.ptr)
                    .ptr(s.p_xqsum.ptr)
                    .ptr(y)
                    .i(w.in_dim as i32)
                    .i(w.out_dim as i32)
                    .i(m as i32)
                    .i(w.out_dim as i32),
            );
        };
        // Partial-N WMMA GEMM: compute only output cols [0,n_compute) of weight
        // `w`, written into `y` with row-stride `ns` (for split-N with the NPU).
        #[cfg(feature = "npu")]
        let q4gemm_part = |w: &ResQ4, y: *mut c_void, n_compute: usize, ns: usize| {
            // n_ig=0 (ffn_up/attn_q fully on NPU) → iGPU computes 0 cols: skip the
            // launch entirely (a 0-block grid is an invalid HIP launch).
            if n_compute == 0 {
                return;
            }
            self.launch2(
                "q4_gemm_w",
                n_compute.div_ceil(128) as u32,
                m.div_ceil(128) as u32,
                256,
                0,
                Args::new()
                    .ptr(w.scales.ptr)
                    .ptr(w.quants.ptr)
                    .ptr(s.p_xqlo.ptr)
                    .ptr(s.p_xqhi.ptr)
                    .ptr(s.p_xqd.ptr)
                    .ptr(s.p_xqsum.ptr)
                    .ptr(y)
                    .i(w.in_dim as i32)
                    .i(n_compute as i32)
                    .i(m as i32)
                    .i(ns as i32),
            );
        };

        // iGPU pacer (STRIX_GPU_PACE_MS): sync + sleep after each prefill layer so the
        // iGPU never SUSTAINS ~100% — the trigger for this box's SoC-reset HW fault.
        // DEFAULT 4ms (safe-by-default for this box, per "don't let the GPU hit 100%");
        // set 0 to disable (when the HW is fixed) or higher for more rest/safety.
        // Trades the iGPU/NPU overlap + ~pace_ms*n_layers wall time for periodic rests.
        let pace_ms = std::env::var("STRIX_GPU_PACE_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(4);
        for l in 0..n_layers {
            let lc = &cfg.layers[l];
            let hd = lc.head_dim;
            let n_kv = lc.n_kv;
            let kv_dim = n_kv * hd;
            let half = hd / 2;
            let groups = (n_heads / n_kv.max(1)).max(1);
            let scale = if cfg.attn_rsqrt {
                1.0 / (hd as f32).sqrt()
            } else {
                1.0
            };
            let pf = |name: &str| format!("blk.{l}.{name}");

            // attn rmsnorm (M rows), quantize, q/k(/v) GEMM. Pure iGPU can
            // fuse norm→Q8; NPU hybrid keeps p_xn for xquant_npu zero-copy.
            if npu_active {
                self.launch(
                    "rmsnorm",
                    m as u32,
                    256,
                    0,
                    Args::new()
                        .ptr(s.p_h.ptr)
                        .ptr(ff(&pf("attn_norm.weight")))
                        .ptr(s.p_xn.ptr)
                        .i(hidden as i32)
                        .i(1)
                        .f(eps),
                );
                quantize(s.p_xn.ptr, m * hidden);
            } else {
                rmsnorm_quantize(s.p_h.ptr, ff(&pf("attn_norm.weight")), m, hidden);
            }
            // Split-N attn_q (local layers, K=hidden, N=q_dim=4096): NPU computes
            // cols [n_ig,4096) during the iGPU's q[0,n_ig)+k+v — the NPU is otherwise
            // idle through qkv/sdpa, so this moves iGPU FLOPs onto it for free (power).
            let qw = q4(&pf("attn_q.weight"));
            #[cfg(feature = "npu")]
            let npu_q_sel = match qw.out_dim {
                4096 => self.npu_q.as_ref(),
                8192 => self.npu_q_g.as_ref(),
                _ => None,
            };
            #[cfg(feature = "npu")]
            let did_npu_q = match (m == crate::npu_hybrid::MPAD).then(|| npu_q_sel).flatten() {
                Some(npu) if npu.layer(l).is_some() => {
                    let (wid, wscale) = npu.layer(l).unwrap();
                    let n_ig = npu.n_ig;
                    let qd = qw.out_dim; // full q row width (4096 local / 8192 global)
                    self.launch(
                        "xquant_npu",
                        crate::npu_hybrid::MPAD as u32,
                        256,
                        0,
                        Args::new()
                            .ptr(s.p_xn.ptr)
                            .ptr(npu.a_dev)
                            .ptr(npu.xscale.ptr)
                            .i(m as i32)
                            .i(hidden as i32)
                            .i(crate::npu_hybrid::MPAD as i32),
                    );
                    self.npu_sync();
                    let _ = npu.start(wid);
                    q4gemm_part(qw, s.p_q.ptr, n_ig, qd); // iGPU q cols [0,n_ig)
                    q4gemm(q4(&pf("attn_k.weight")), s.p_k.ptr);
                    if !lc.k_eq_v {
                        q4gemm(q4(&pf("attn_v.weight")), s.p_v.ptr);
                    }
                    let _ = npu.wait();
                    let tot = m * npu.n_npu;
                    self.launch(
                        "rescale_npu",
                        tot.div_ceil(256) as u32,
                        256,
                        0,
                        Args::new()
                            .ptr(npu.out_dev)
                            .ptr(npu.xscale.ptr)
                            .ptr(wscale)
                            .ptr(s.p_q.ptr)
                            .i(m as i32)
                            .i(npu.n_npu as i32)
                            .i(qd as i32)
                            .i(n_ig as i32),
                    );
                    true
                }
                _ => false,
            };
            #[cfg(not(feature = "npu"))]
            let did_npu_q = false;
            if !did_npu_q {
                q4gemm(qw, s.p_q.ptr);
                q4gemm(q4(&pf("attn_k.weight")), s.p_k.ptr);
                if !lc.k_eq_v {
                    q4gemm(q4(&pf("attn_v.weight")), s.p_v.ptr);
                }
            }
            let v_src = if lc.k_eq_v { s.p_k.ptr } else { s.p_v.ptr };
            // per-head Q/K norms (m*heads rows of hd) + V norm/copy
            self.launch(
                "rmsnorm",
                (m * n_heads) as u32,
                256,
                0,
                Args::new()
                    .ptr(s.p_q.ptr)
                    .ptr(ff(&pf("attn_q_norm.weight")))
                    .ptr(s.p_q2.ptr)
                    .i(hd as i32)
                    .i(1)
                    .f(eps),
            );
            self.launch(
                "rmsnorm",
                (m * n_kv) as u32,
                256,
                0,
                Args::new()
                    .ptr(s.p_k.ptr)
                    .ptr(ff(&pf("attn_k_norm.weight")))
                    .ptr(s.p_k2.ptr)
                    .i(hd as i32)
                    .i(1)
                    .f(eps),
            );
            if cfg.norm_v {
                self.launch(
                    "rmsnorm",
                    (m * n_kv) as u32,
                    256,
                    0,
                    Args::new()
                        .ptr(v_src)
                        .ptr(s.ones.ptr)
                        .ptr(s.p_v2.ptr)
                        .i(hd as i32)
                        .i(0)
                        .f(eps),
                );
            } else {
                self.launch(
                    "copyf",
                    (m * kv_dim).div_ceil(256) as u32,
                    256,
                    0,
                    Args::new()
                        .ptr(s.p_v2.ptr)
                        .ptr(v_src)
                        .i((m * kv_dim) as i32),
                );
            }
            // batched RoPE
            let ropef = if lc.is_local {
                s.ones.ptr
            } else {
                self.f32w
                    .get("rope_freqs.weight")
                    .map(|d| d.ptr)
                    .unwrap_or(s.ones.ptr)
            };
            self.launch(
                "rope_batch",
                dc(m * n_heads * half),
                64,
                0,
                Args::new()
                    .ptr(s.p_q2.ptr)
                    .ptr(ropef)
                    .i(hd as i32)
                    .i(n_heads as i32)
                    .i(start_pos as i32)
                    .f(lc.rope_theta)
                    .i(m as i32),
            );
            self.launch(
                "rope_batch",
                dc(m * n_kv * half),
                64,
                0,
                Args::new()
                    .ptr(s.p_k2.ptr)
                    .ptr(ropef)
                    .i(hd as i32)
                    .i(n_kv as i32)
                    .i(start_pos as i32)
                    .f(lc.rope_theta)
                    .i(m as i32),
            );
            // fill KV cache slots [start_pos .. start_pos+m]
            let off = start_pos * kv_dim * if self.kv_f16 { 2 } else { 4 };
            let kdst = unsafe { (s.k_cache[l].ptr as *mut u8).add(off) as *mut c_void };
            let vdst = unsafe { (s.v_cache[l].ptr as *mut u8).add(off) as *mut c_void };
            let kvcopy = if self.kv_f16 { "copyf_h" } else { "copyf" };
            self.launch(
                kvcopy,
                (m * kv_dim).div_ceil(256) as u32,
                256,
                0,
                Args::new().ptr(kdst).ptr(s.p_k2.ptr).i((m * kv_dim) as i32),
            );
            self.launch(
                kvcopy,
                (m * kv_dim).div_ceil(256) as u32,
                256,
                0,
                Args::new().ptr(vdst).ptr(s.p_v2.ptr).i((m * kv_dim) as i32),
            );
            // causal SDPA over m queries
            // STRIX_SKIP_SDPA=all|global|local — diagnostic: skip SDPA on a layer subset
            let skip_kind = std::env::var("STRIX_SKIP_SDPA").unwrap_or_default();
            let skip_this = match skip_kind.as_str() {
                "" => false,
                "global" => !lc.is_local,
                "local" => lc.is_local,
                _ => true, // any other value (incl. "1") = skip all
            };
            if !skip_this {
                if std::env::var("STRIX_GEMM_SDPA").is_ok() {
                    // GEMM-based attention (llama's non-flash path): QK^T -> mask+softmax -> P*V,
                    // all f16 WMMA, K/V read once. len = total keys after this chunk.
                    let len = start_pos + m;
                    let n_swa = if lc.is_local { cfg.n_swa as i32 } else { 0 };
                    // waves/block (each wave = one independent output tile). 8 default.
                    let gnw = std::env::var("STRIX_GEMM_NW")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(8)
                        .clamp(1, 16);
                    self.launch3(
                        "sdpa_qk_wmma",
                        n_heads as u32,
                        m.div_ceil(16) as u32,
                        len.div_ceil(16 * gnw) as u32,
                        (gnw * 32) as u32,
                        0,
                        Args::new()
                            .ptr(s.p_q2.ptr)
                            .ptr(s.k_cache[l].ptr)
                            .ptr(s.p_scores.ptr)
                            .i(hd as i32)
                            .i(groups as i32)
                            .i(n_kv as i32)
                            .f(scale)
                            .i(n_heads as i32)
                            .i(m as i32)
                            .i(len as i32)
                            .i(if self.kv_f16 { 1 } else { 0 })
                            .i(start_pos as i32),
                    );
                    self.launch2(
                        "sdpa_softmax_mask",
                        n_heads as u32,
                        m as u32,
                        256,
                        0,
                        Args::new()
                            .ptr(s.p_scores.ptr)
                            .i(start_pos as i32)
                            .i(m as i32)
                            .i(n_heads as i32)
                            .i(len as i32)
                            .i(n_swa)
                            .ptr(s.p_rowsum.ptr),
                    );
                    self.launch3(
                        "sdpa_pv_wmma",
                        n_heads as u32,
                        m.div_ceil(16) as u32,
                        hd.div_ceil(16 * gnw) as u32,
                        (gnw * 32) as u32,
                        0,
                        Args::new()
                            .ptr(s.p_scores.ptr)
                            .ptr(s.v_cache[l].ptr)
                            .ptr(s.p_attn.ptr)
                            .i(hd as i32)
                            .i(groups as i32)
                            .i(n_kv as i32)
                            .i(n_heads as i32)
                            .i(m as i32)
                            .i(len as i32)
                            .i(if self.kv_f16 { 1 } else { 0 })
                            .i(start_pos as i32)
                            .ptr(s.p_rowsum.ptr),
                    );
                } else if std::env::var("STRIX_OLD_SDPA").is_ok() {
                    self.launch(
                        "sdpa_prefill",
                        (m * n_heads) as u32,
                        256,
                        0,
                        Args::new()
                            .ptr(s.p_q2.ptr)
                            .ptr(s.k_cache[l].ptr)
                            .ptr(s.v_cache[l].ptr)
                            .ptr(s.p_attn.ptr)
                            .i(hd as i32)
                            .i(start_pos as i32)
                            .i(groups as i32)
                            .i(n_kv as i32)
                            .f(scale)
                            .i(n_heads as i32)
                            .i(if self.kv_f16 { 1 } else { 0 }),
                    );
                } else if std::env::var("STRIX_WMMA_SDPA").is_ok() && hd <= 256 {
                    // WMMA (matrix-core) flash attention: NW waves/block share one K/V
                    // tile; each wave owns a 16-query sub-tile. shared = K,V + per-wave Q,S,P.
                    // Gated to hd<=256: dchunks=hd/16<=MAXDC(16) and shared stays <64KB.
                    // Gemma global layers (hd=512) fall through to the scalar path below.
                    let n_swa = if lc.is_local { cfg.n_swa as i32 } else { 0 };
                    let nw = std::env::var("STRIX_WMMA_NW")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(4)
                        .clamp(1, 4);
                    let shbytes = (64 * hd + 32 * nw * hd + 1664 * nw) as u32;
                    self.launch2(
                        "sdpa_prefill_wmma",
                        n_heads as u32,
                        m.div_ceil(16 * nw) as u32,
                        (nw * 32) as u32,
                        shbytes,
                        Args::new()
                            .ptr(s.p_q2.ptr)
                            .ptr(s.k_cache[l].ptr)
                            .ptr(s.v_cache[l].ptr)
                            .ptr(s.p_attn.ptr)
                            .i(hd as i32)
                            .i(start_pos as i32)
                            .i(groups as i32)
                            .i(n_kv as i32)
                            .f(scale)
                            .i(n_heads as i32)
                            .i(m as i32)
                            .i(n_swa)
                            .i(if self.kv_f16 { 1 } else { 0 }),
                    );
                } else {
                    // flash-style: grid=(n_heads, ceil(m/8)), block=256, shared=2*hd floats.
                    // Local layers pass n_swa (per-query sliding window); global pass 0.
                    let n_swa = if lc.is_local { cfg.n_swa as i32 } else { 0 };
                    self.launch2(
                        "sdpa_prefill_f",
                        n_heads as u32,
                        m.div_ceil(16) as u32,
                        256,
                        (2 * hd * 4) as u32,
                        Args::new()
                            .ptr(s.p_q2.ptr)
                            .ptr(s.k_cache[l].ptr)
                            .ptr(s.v_cache[l].ptr)
                            .ptr(s.p_attn.ptr)
                            .i(hd as i32)
                            .i(start_pos as i32)
                            .i(groups as i32)
                            .i(n_kv as i32)
                            .f(scale)
                            .i(n_heads as i32)
                            .i(m as i32)
                            .i(n_swa)
                            .i(if self.kv_f16 { 1 } else { 0 }),
                    );
                }
            }
            // attn output GEMM + fused residual norm
            let o = q4(&pf("attn_output.weight"));
            quantize(s.p_attn.ptr, m * o.in_dim);
            // Split-N attn_output (K=q_dim): iGPU cols [0,n_ig) ∥ NPU [n_ig,hidden).
            // Local (q_dim=4096) → npu_o, global (8192) → npu_o_g.
            #[cfg(feature = "npu")]
            let npu_o_sel = match o.in_dim {
                4096 => self.npu_o.as_ref(),
                8192 => self.npu_o_g.as_ref(),
                _ => None,
            };
            #[cfg(feature = "npu")]
            let did_npu_o = match (m == crate::npu_hybrid::MPAD).then(|| npu_o_sel).flatten() {
                Some(npu) if npu.layer(l).is_some() => {
                    let (wid, wscale) = npu.layer(l).unwrap();
                    let (n_ig, kk) = (npu.n_ig, o.in_dim);
                    self.launch(
                        "xquant_npu",
                        crate::npu_hybrid::MPAD as u32,
                        256,
                        0,
                        Args::new()
                            .ptr(s.p_attn.ptr)
                            .ptr(npu.a_dev)
                            .ptr(npu.xscale.ptr)
                            .i(m as i32)
                            .i(kk as i32)
                            .i(crate::npu_hybrid::MPAD as i32),
                    );
                    self.npu_sync();
                    let _ = npu.start(wid);
                    q4gemm_part(o, s.p_th.ptr, n_ig, hidden);
                    let _ = npu.wait();
                    let tot = m * npu.n_npu;
                    self.launch(
                        "rescale_npu",
                        tot.div_ceil(256) as u32,
                        256,
                        0,
                        Args::new()
                            .ptr(npu.out_dev)
                            .ptr(npu.xscale.ptr)
                            .ptr(wscale)
                            .ptr(s.p_th.ptr)
                            .i(m as i32)
                            .i(npu.n_npu as i32)
                            .i(hidden as i32)
                            .i(n_ig as i32),
                    );
                    true
                }
                _ => false,
            };
            #[cfg(not(feature = "npu"))]
            let did_npu_o = false;
            if !did_npu_o {
                q4gemm(o, s.p_th.ptr);
            }
            self.launch(
                "addnorm_batch",
                m as u32,
                256,
                0,
                Args::new()
                    .ptr(s.p_h.ptr)
                    .ptr(s.p_th.ptr)
                    .ptr(ff(&pf("post_attention_norm.weight")))
                    .i(hidden as i32)
                    .f(eps)
                    .f(1.0),
            );
            // ffn. Pure iGPU can fuse norm→Q8; NPU hybrid keeps p_xn for the
            // ffn_up offload activation buffer.
            if npu_active {
                self.launch(
                    "rmsnorm",
                    m as u32,
                    256,
                    0,
                    Args::new()
                        .ptr(s.p_h.ptr)
                        .ptr(ff(&pf("ffn_norm.weight")))
                        .ptr(s.p_xn.ptr)
                        .i(hidden as i32)
                        .i(1)
                        .f(eps),
                );
                quantize(s.p_xn.ptr, m * hidden);
            } else {
                rmsnorm_quantize(s.p_h.ptr, ff(&pf("ffn_norm.weight")), m, hidden);
            }
            // Hybrid: offload ffn_up to the NPU (int8), concurrent with the iGPU
            // doing ffn_gate. Only on full M=256 chunks (the NPU xclbin is fixed-M).
            #[cfg(feature = "npu")]
            let did_npu_up = match (m == crate::npu_hybrid::MPAD)
                .then(|| self.npu.as_ref())
                .flatten()
            {
                Some(npu) if npu.layer(l).is_some() => {
                    // Split-N ffn_up: iGPU computes cols [0,n_ig) + ffn_gate, the
                    // NPU computes cols [n_ig,ffn) concurrently; both → s.p_up.
                    let (wid, wscale) = npu.layer(l).unwrap();
                    let (n_ig, ffn) = (npu.n_ig, cfg.ffn);
                    self.launch(
                        "xquant_npu",
                        crate::npu_hybrid::MPAD as u32,
                        256,
                        0,
                        Args::new()
                            .ptr(s.p_xn.ptr)
                            .ptr(npu.a_dev)
                            .ptr(npu.xscale.ptr)
                            .i(m as i32)
                            .i(hidden as i32)
                            .i(crate::npu_hybrid::MPAD as i32),
                    );
                    self.npu_sync(); // activation must be written before the NPU reads it
                    let _ = npu.start(wid); // async NPU ffn_up cols [n_ig,ffn)
                    q4gemm(q4(&pf("ffn_gate.weight")), s.p_gate.ptr);
                    q4gemm_part(q4(&pf("ffn_up.weight")), s.p_up.ptr, n_ig, ffn);
                    let _ = npu.wait();
                    let tot = m * npu.n_npu;
                    self.launch(
                        "rescale_npu",
                        tot.div_ceil(256) as u32,
                        256,
                        0,
                        Args::new()
                            .ptr(npu.out_dev)
                            .ptr(npu.xscale.ptr)
                            .ptr(wscale)
                            .ptr(s.p_up.ptr)
                            .i(m as i32)
                            .i(npu.n_npu as i32)
                            .i(ffn as i32)
                            .i(n_ig as i32),
                    );
                    true
                }
                _ => false,
            };
            #[cfg(not(feature = "npu"))]
            let did_npu_up = false;
            if !did_npu_up {
                q4gemm(q4(&pf("ffn_gate.weight")), s.p_gate.ptr);
                q4gemm(q4(&pf("ffn_up.weight")), s.p_up.ptr);
            }
            let down = q4(&pf("ffn_down.weight"));
            if !npu_active {
                geglu_quantize(s.p_gate.ptr, s.p_up.ptr, m * down.in_dim);
                q4gemm(down, s.p_th.ptr);
            } else {
                self.launch(
                    "geglu",
                    dc(m * cfg.ffn),
                    64,
                    0,
                    Args::new()
                        .ptr(s.p_gate.ptr)
                        .ptr(s.p_up.ptr)
                        .ptr(s.p_act.ptr)
                        .i((m * cfg.ffn) as i32),
                );
                quantize(s.p_act.ptr, m * down.in_dim);
                // Split-N ffn_down: iGPU cols [0,n_ig) ∥ NPU cols [n_ig,hidden).
                #[cfg(feature = "npu")]
                let did_npu_down = match (m == crate::npu_hybrid::MPAD)
                    .then(|| self.npu_down.as_ref())
                    .flatten()
                {
                    Some(npu) if npu.layer(l).is_some() => {
                        let (wid, wscale) = npu.layer(l).unwrap();
                        let (n_ig, kk) = (npu.n_ig, down.in_dim); // kk = ffn (down's K)
                        self.launch(
                            "xquant_npu",
                            crate::npu_hybrid::MPAD as u32,
                            256,
                            0,
                            Args::new()
                                .ptr(s.p_act.ptr)
                                .ptr(npu.a_dev)
                                .ptr(npu.xscale.ptr)
                                .i(m as i32)
                                .i(kk as i32)
                                .i(crate::npu_hybrid::MPAD as i32),
                        );
                        self.npu_sync();
                        let _ = npu.start(wid);
                        q4gemm_part(down, s.p_th.ptr, n_ig, hidden); // iGPU down cols [0,n_ig)
                        let _ = npu.wait();
                        let tot = m * npu.n_npu;
                        self.launch(
                            "rescale_npu",
                            tot.div_ceil(256) as u32,
                            256,
                            0,
                            Args::new()
                                .ptr(npu.out_dev)
                                .ptr(npu.xscale.ptr)
                                .ptr(wscale)
                                .ptr(s.p_th.ptr)
                                .i(m as i32)
                                .i(npu.n_npu as i32)
                                .i(hidden as i32)
                                .i(n_ig as i32),
                        );
                        true
                    }
                    _ => false,
                };
                #[cfg(not(feature = "npu"))]
                let did_npu_down = false;
                if !did_npu_down {
                    q4gemm(down, s.p_th.ptr);
                }
            }
            self.launch(
                "addnorm_batch",
                m as u32,
                256,
                0,
                Args::new()
                    .ptr(s.p_h.ptr)
                    .ptr(s.p_th.ptr)
                    .ptr(ff(&pf("post_ffw_norm.weight")))
                    .i(hidden as i32)
                    .f(eps)
                    .f(lc.output_scale),
            );
            // pace: drain the GPU and rest, so it never sustains 100% across the chunk.
            if pace_ms > 0 {
                let _ = self.gpu.sync();
                std::thread::sleep(std::time::Duration::from_millis(pace_ms));
            }
        }

        // Final norm + lm_head. Normally just the LAST token (rmsnorm its row →
        // lm_head). For speculative verify (`verify_all`), do EVERY token row and
        // return m×vocab logits.
        let lm_one = |row: usize| {
            let hr = unsafe { (s.p_h.ptr as *mut u8).add(row * hidden * 4) as *mut c_void };
            self.launch(
                "rmsnorm",
                1,
                256,
                0,
                Args::new()
                    .ptr(hr)
                    .ptr(ff("output_norm.weight"))
                    .ptr(s.xn.ptr)
                    .i(hidden as i32)
                    .i(1)
                    .f(eps),
            );
            self.lm_head(s.xn.ptr, s.logits.ptr);
            if cfg.final_softcap > 0.0 {
                self.launch(
                    "softcap",
                    dc(cfg.vocab),
                    64,
                    0,
                    Args::new()
                        .ptr(s.logits.ptr)
                        .i(cfg.vocab as i32)
                        .f(cfg.final_softcap),
                );
            }
        };
        let out = if all_logits {
            let mut all = Vec::with_capacity(m * cfg.vocab);
            for r in 0..m {
                lm_one(r);
                self.gpu.sync().ok()?;
                all.extend(s.logits.download::<f32>(cfg.vocab).ok()?);
            }
            all
        } else {
            lm_one(m - 1);
            self.gpu.sync().ok()?;
            if prof_enabled() {
                prof_dump(&format!("prefill m={m}"));
            }
            s.logits.download::<f32>(cfg.vocab).ok()?
        };
        #[cfg(feature = "npu")]
        if crate::npu_hybrid::timing_enabled() {
            crate::npu_hybrid::npu_time_dump(&format!("prefill m={m}"));
        }
        Some(out)
    }
}
