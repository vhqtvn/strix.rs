//! Hybrid NPU+iGPU prefill: offload a prefill GEMM (ffn_up) to the XDNA2 NPU,
//! concurrent with the iGPU doing the sibling GEMM (ffn_gate). The NPU does an
//! int8 A@B GEMM (W8A8: per-row int8 weight, per-token int8 activation), with
//! zero-copy host I/O (hipHostRegister'd XRT BOs) so the iGPU writes the int8
//! activation and reads the int32 output directly over unified LPDDR5x.
#![cfg(feature = "npu")]

use std::os::raw::c_void;

use crate::hip::{Dbuf, HipGpu};
use strix_backend_npu::NpuGemm;

/// The NPU xclbins are compiled for a fixed M (token count); pad chunks to it.
pub const MPAD: usize = 256;

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        let v = (m as f32) * 5.9604645e-8; // subnormal: mant * 2^-24
        return if s == 1 { -v } else { v };
    }
    let bits = (s << 31) | ((e + 112) << 23) | (m << 13);
    f32::from_bits(bits)
}

/// Repack output-channel rows `[rstart,rend)` of a Q4_0 weight `W[N,K]` into the
/// NPU's B layout: int8 `B[K, rend-rstart]` (transposed, row-major) with a
/// per-output-channel scale `wscale[rend-rstart]`. Each row is dequantized then
/// requantized to int8 with one scale (W8A8). Coherent on gemma-4 (STRIX_W8A8).
pub fn repack_q4_0_transposed(
    bytes: &[u8],
    k: usize,
    rstart: usize,
    rend: usize,
) -> (Vec<i8>, Vec<f32>) {
    const QK: usize = 32;
    const BB: usize = 18; // 2-byte f16 scale + 16 nibble bytes
    let nb = k / QK;
    let nout = rend - rstart;
    let mut b = vec![0i8; k * nout];
    let mut wscale = vec![0f32; nout];
    let mut row = vec![0f32; k];
    for r in rstart..rend {
        let oc = r - rstart;
        let base = r * nb * BB;
        // dequantize row r -> f32[k]
        for blk in 0..nb {
            let off = base + blk * BB;
            let d = f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]));
            let q = &bytes[off + 2..off + 18];
            for (i, &byte) in q.iter().enumerate() {
                let lo = (byte & 0x0f) as i32 - 8;
                let hi = (byte >> 4) as i32 - 8;
                row[blk * QK + i] = lo as f32 * d;
                row[blk * QK + i + 16] = hi as f32 * d;
            }
        }
        let wmax = row.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-12);
        let ws = wmax / 127.0;
        wscale[oc] = ws;
        let inv = 1.0 / ws;
        for kk in 0..k {
            let q = (row[kk] * inv).round() as i32;
            b[kk * nout + oc] = q.clamp(-127, 127) as i8; // transposed: B[k][oc]
        }
    }
    (b, wscale)
}

/// NPU offload state for one GEMM shape (e.g. ffn_up): the staged per-layer
/// weights + the zero-copy I/O buffers + per-token/per-channel scale buffers.
pub struct NpuFfn {
    gemm: NpuGemm,
    /// hipHostRegister'd device pointers into the NPU's A (int8 [MPAD,K]) and
    /// out (int32 [MPAD,N]) host buffers — iGPU kernels read/write these.
    pub a_dev: *mut c_void,
    pub out_dev: *mut c_void,
    pub xscale: Dbuf,
    /// per-LAYER weight: staged-weight index + its [n_npu] scale buffer. Keyed by
    /// layer (not all layers are staged — e.g. attn_output only for local layers).
    pub layers: std::collections::HashMap<usize, (i32, Dbuf)>,
    pub k: usize,
    pub n_npu: usize, // output columns the NPU computes
    pub n_ig: usize,  // output columns the iGPU computes (cols [0,n_ig))
}

impl NpuFfn {
    /// Split-N: iGPU computes output cols [0,n_ig); NPU computes [n_ig, n_ig+n_npu).
    pub fn open(
        gpu: &HipGpu,
        xclbin: &str,
        insts: &[u32],
        k: usize,
        n_npu: usize,
        n_ig: usize,
    ) -> Result<Self, String> {
        let a_cap = MPAD * k; // int8
        let out_cap = MPAD * n_npu * 4; // int32
        let gemm = NpuGemm::open(xclbin, "MLIR_AIE", insts, a_cap, out_cap)?;
        let a_dev = gpu
            .register_host(gemm.a_host, a_cap)
            .map_err(|e| format!("register A: {e}"))?;
        let out_dev = gpu
            .register_host(gemm.out_host, out_cap)
            .map_err(|e| format!("register out: {e}"))?;
        let xscale = gpu
            .alloc(MPAD * 4)
            .map_err(|e| format!("alloc xscale: {e}"))?;
        Ok(NpuFfn {
            gemm,
            a_dev,
            out_dev,
            xscale,
            layers: std::collections::HashMap::new(),
            k,
            n_npu,
            n_ig,
        })
    }

    /// Stage layer `layer`'s weight (full Q4_0 bytes) — repack the NPU's row
    /// slice [n_ig, n_ig+n_npu) to int8 + upload its per-channel scale.
    pub fn stage_q4(&mut self, gpu: &HipGpu, layer: usize, q4_bytes: &[u8]) -> Result<(), String> {
        let (b8, ws) = repack_q4_0_transposed(q4_bytes, self.k, self.n_ig, self.n_ig + self.n_npu);
        let wid = self.gemm.stage(&b8)?;
        let buf = gpu
            .upload_new(&ws)
            .map_err(|e| format!("upload wscale: {e}"))?;
        self.layers.insert(layer, (wid, buf));
        Ok(())
    }

    /// (staged-weight index, per-channel scale ptr) for `layer`, if staged.
    pub fn layer(&self, layer: usize) -> Option<(i32, *mut std::os::raw::c_void)> {
        self.layers.get(&layer).map(|(wid, buf)| (*wid, buf.ptr))
    }
    pub fn start(&self, wid: i32) -> Result<(), String> {
        if timing_enabled() {
            let t0 = std::time::Instant::now();
            let r = self.gemm.start(wid);
            npu_time_add("npu.start", t0.elapsed().as_secs_f64());
            return r;
        }
        self.gemm.start(wid)
    }
    pub fn wait(&self) -> Result<(), String> {
        if timing_enabled() {
            let t0 = std::time::Instant::now();
            let r = self.gemm.wait();
            npu_time_add("npu.wait", t0.elapsed().as_secs_f64());
            return r;
        }
        self.gemm.wait()
    }

    /// Standalone NPU GEMM roofline (idea prefill-#26): run `reps` isolated
    /// start+wait of this M=256 × K × n_npu shape and report achieved TOPS / BW /
    /// arithmetic-intensity, classifying compute- vs memory-bound. This is the
    /// decisive test for int4-on-AIE (#13): only a MEMORY-bound shape benefits from
    /// halving the B (weight) bytes. NPU-only (~<2 W) → safe to run sustained.
    pub fn roofline(&self, label: &str, wid: i32, reps: usize) {
        // warm up (first run pays setup) then time the steady-state.
        let _ = self.start(wid).and_then(|_| self.wait());
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            if self.start(wid).and_then(|_| self.wait()).is_err() {
                eprintln!("[npu-roofline {label}] run failed");
                return;
            }
        }
        let t = t0.elapsed().as_secs_f64() / reps as f64;
        let (m, k, n) = (MPAD as f64, self.k as f64, self.n_npu as f64);
        let ops = 2.0 * m * k * n; // MAC = 2 flops
                                   // int8 A (m·k) + int8 B (k·n) + int32 C (m·n·4)
        let bytes = m * k + k * n + m * n * 4.0;
        let tops = ops / t / 1e12;
        let bw = bytes / t / 1e9;
        let ai = ops / bytes;
        // ~50 TOPS int8 peak, ~60 GB/s NPU bandwidth (measured, see memory).
        let bound = if tops / 50.0 > bw / 60.0 {
            "COMPUTE"
        } else {
            "MEMORY"
        };
        eprintln!(
            "[npu-roofline {label}] M=256 K={} N={} | {:.3} ms/run | {:.1} TOPS ({:.0}%) | {:.1} GB/s ({:.0}%) | AI={:.1} | {}-bound{}",
            self.k, self.n_npu, t * 1e3, tops, tops / 50.0 * 100.0, bw, bw / 60.0 * 100.0, ai, bound,
            if bound == "MEMORY" { " (int4 weights WOULD help)" } else { " (int4 weights would NOT help)" }
        );
    }
}

// --- NPU coordination timing (gated by STRIX_NPU_TIMING; does NOT add syncs,
// so it leaves the iGPU∥NPU overlap intact — unlike STRIX_PROF's per-kernel sync). ---
use std::cell::RefCell;
use std::sync::OnceLock;
pub fn timing_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("STRIX_NPU_TIMING").is_ok())
}
thread_local! {
    static NT: RefCell<std::collections::HashMap<String, (f64, u32)>> =
        RefCell::new(std::collections::HashMap::new());
}
pub fn npu_time_add(name: &str, secs: f64) {
    NT.with(|p| {
        let e = p
            .borrow_mut()
            .entry(name.to_string())
            .or_insert((0.0, 0))
            .clone();
        p.borrow_mut()
            .insert(name.to_string(), (e.0 + secs, e.1 + 1));
    });
}
pub fn npu_time_dump(tag: &str) {
    NT.with(|p| {
        let m = p.borrow();
        if m.is_empty() {
            return;
        }
        let mut v: Vec<_> = m.iter().map(|(k, &(s, c))| (k.clone(), s, c)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let total: f64 = v.iter().map(|x| x.1).sum();
        eprintln!(
            "== STRIX_NPU_TIMING [{tag}] total NPU-coord {:.3} ms ==",
            total * 1e3
        );
        for (k, s, c) in &v {
            eprintln!(
                "  {:>12}  {:8.3} ms  {:5} calls  ({:.3} ms/call)",
                k,
                s * 1e3,
                c,
                s / *c as f64 * 1e3
            );
        }
    });
    NT.with(|p| p.borrow_mut().clear());
}
