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

/// ROCm/HIP decode accelerator.
pub struct RocmWeightAccel {
    gpu: HipGpu,
    funcs: HashMap<&'static str, hipFunction_t>,
    q4: HashMap<String, ResQ4>,
    q6: HashMap<String, ResQ6>,
    f32w: HashMap<String, Dbuf>,
    cfg: Option<GpuDecodeConfig>,
    scratch: Option<Scratch>,
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
        let code = crate::hip::compile(crate::kernels::KERNELS).ok()?;
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
            "xquant_npu",
            "rescale_npu",
        ] {
            funcs.insert(name, gpu.get_function(module, name).ok()?);
        }
        Some(Self {
            gpu,
            funcs,
            q4: HashMap::new(),
            q6: HashMap::new(),
            f32w: HashMap::new(),
            cfg: None,
            scratch: None,
            kv_f16: std::env::var("STRIX_F16_KV").is_ok(),
            // pinned split if STRIX_N_SPLIT set, else 0 = context-adaptive (~len/64).
            n_split: std::env::var("STRIX_N_SPLIT").ok().and_then(|s| s.parse::<usize>().ok()).map(|v| v.clamp(1, N_SPLIT_MAX)).unwrap_or(0),
            name,
            verify_all: false,
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

    fn launch(&self, name: &str, grid: u32, block: u32, shared: u32, args: Args) {
        self.launch2(name, grid, 1, block, shared, args);
    }

    /// lm_head GEMV (tied token_embd): Q6_K (gemma-4 target) or Q4_0 (a pure-Q4_0
    /// draft). `x` = final-normed hidden [hidden], `y` = logits out [vocab].
    fn lm_head(&self, x: *mut c_void, y: *mut c_void) {
        if let Some(e) = self.q6.get("token_embd.weight") {
            self.launch(
                "q6_gemv",
                e.out_dim.div_ceil(8) as u32,
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
        // n_ig tuned so both finish together (iGPU also does ffn_gate).
        let n_ig = std::env::var("STRIX_NPU_NIG")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048usize);
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
            let n_npu_q = std::env::var("STRIX_NPU_NNPU_Q")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2048usize);
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
            // STRIX_NPU_MODE: "speed" (DEFAULT) offloads only ffn_up (the gate∥up
            // pair has free iGPU/NPU parallelism → robustly ≥ pure-iGPU); "power"
            // offloads everything (down/o/q too — more iGPU work moved to the NPU,
            // but they put the NPU on the critical path so prefill can be slower
            // than pure-iGPU at high iGPU clock). MEASURED (thermal-gated bench):
            // up-only 136 > pure-iGPU 125 > all-offloads 122 tok/s. STRIX_NPU_SKIP
            // is a fine override (comma list of up,down,o,q to force-disable).
            let power = std::env::var("STRIX_NPU_MODE").map(|m| m == "power").unwrap_or(false);
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

    fn gemv(&self, _key: &str, _x: &[f32]) -> Option<Vec<f32>> {
        None // whole forward runs in decode_step
    }

    fn resident_count(&self) -> usize {
        self.q4.len() + self.q6.len()
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
        let skip_sdpa = std::env::var("STRIX_SKIP_SDPA").is_ok();

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
            if std::env::var("STRIX_SKIP_SDPA").is_err() {
                if std::env::var("STRIX_OLD_SDPA").is_ok() {
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
                    let nw = std::env::var("STRIX_WMMA_NW").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(4).clamp(1, 4);
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
                        m.div_ceil(8) as u32,
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
