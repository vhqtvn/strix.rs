//! NPU (XDNA2) prefill offload for the Mellum CPU forward.
//!
//! The NPU runs fixed-shape int8 GEMMs (M=256, K/N fixed per xclbin) with staged
//! int8 weights. We requantize the Q8_0 weights per OUTPUT CHANNEL to int8 at stage
//! time and the f32 activations per ROW (token) at call time, run `Y_i32 = A_i8·B_i8`
//! on the AIE array, then rescale on CPU: `y[t][o] = xs[t]·ws[o]·acc`. The scheme
//! matches the validated Gemma hybrid (npu_hybrid.rs) but is fully CPU-driven —
//! NO iGPU involvement (rescale is cheap on CPU). Not bit-identical to the CPU
//! forward (per-channel int8 weight requant): validate by greedy-coherence.
//!
//! Used by `MellumModel::prefill_batch` per chunk of up to 256 tokens; shorter
//! chunks are zero-padded in M.

use strix_backend_npu::NpuGemm as Gemm;
use strix_backend_npu::{load_instr_bin, load_instr_txt};
use strix_core::error::{Result, StrixError};
use strix_models::ggml_quant::{dequantize_into, GgmlType};

pub const M_NPU: usize = 256;
/// Below this row count CPU wins (per-call NPU latency dominates).
pub const M_MIN: usize = 64;

/// One fixed-shape NPU GEMM (K×N) with N staged weights (per layer / per expert).
pub struct NpuShape {
    gemm: Gemm,
    pub k: usize,
    pub n: usize,
    /// staged weight id + per-channel scale [n], keyed by an opaque slot id
    weights: std::collections::HashMap<u64, (i32, Vec<f32>)>,
}

impl NpuShape {
    pub fn open(dir: &str, k: usize, n: usize, cols: usize) -> Result<NpuShape> {
        let stem = format!("256x{k}x{n}_64x64x64_{cols}c");
        let xclbin = format!("{dir}/final_{stem}.xclbin");
        let insts = {
            let bin = format!("{dir}/insts_{stem}.bin");
            let txt = format!("{dir}/insts_{stem}.txt");
            let raw = std::fs::read(&bin)
                .or_else(|_| std::fs::read(&txt))
                .map_err(|e| StrixError::backend(format!("read insts {stem}: {e}")))?;
            // newer mlir-aie writes binary insts even with a .txt suffix
            match std::str::from_utf8(&raw) {
                Ok(t) => load_instr_txt(t).map_err(StrixError::backend)?,
                Err(_) => load_instr_bin(&raw).map_err(StrixError::backend)?,
            }
        };
        let gemm = Gemm::open(&xclbin, "MLIR_AIE", &insts, M_NPU * k, M_NPU * n * 4)
            .map_err(StrixError::backend)?;
        Ok(NpuShape {
            gemm,
            k,
            n,
            weights: std::collections::HashMap::new(),
        })
    }

    /// Requantize a Q8_0 weight [n rows of k] per OUTPUT CHANNEL to int8 [K,N]
    /// (column-major for the GEMM's B layout = row-major [K][N]) and stage it.
    pub fn stage_q8(&mut self, slot: u64, bytes: &[u8], ty: GgmlType) -> Result<()> {
        let (k, n) = (self.k, self.n);
        let mut rowf = vec![0.0f32; k];
        let bpr = (k / ty.block_elems()) * ty.block_bytes();
        let mut b8 = vec![0i8; k * n];
        let mut ws = vec![0.0f32; n];
        for o in 0..n {
            dequantize_into(ty, &bytes[o * bpr..(o + 1) * bpr], &mut rowf)?;
            let amax = rowf.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            ws[o] = s;
            let inv = 1.0 / s;
            for i in 0..k {
                b8[i * n + o] = (rowf[i] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
        let wid = self.gemm.stage(&b8).map_err(StrixError::backend)?;
        self.weights.insert(slot, (wid, ws));
        Ok(())
    }

    /// Stage two stacked Q8_0 weights (e.g. gate ‖ up, each n/2 rows of k) as one
    /// [K, N] weight: output channels [0,n/2) from `a`, [n/2,n) from `b`.
    pub fn stage_q8_pair(&mut self, slot: u64, a: &[u8], b: &[u8], ty: GgmlType) -> Result<()> {
        let (k, n) = (self.k, self.n);
        let half = n / 2;
        let mut rowf = vec![0.0f32; k];
        let bpr = (k / ty.block_elems()) * ty.block_bytes();
        let mut b8 = vec![0i8; k * n];
        let mut ws = vec![0.0f32; n];
        for o in 0..n {
            let (src, r) = if o < half { (a, o) } else { (b, o - half) };
            dequantize_into(ty, &src[r * bpr..(r + 1) * bpr], &mut rowf)?;
            let amax = rowf.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            ws[o] = s;
            let inv = 1.0 / s;
            for i in 0..k {
                b8[i * n + o] = (rowf[i] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
        let wid = self.gemm.stage(&b8).map_err(StrixError::backend)?;
        self.weights.insert(slot, (wid, ws));
        Ok(())
    }

    pub fn has(&self, slot: u64) -> bool {
        self.weights.contains_key(&slot)
    }

    /// Y[m][n] = W·xs for m≤256 rows. Quantizes activations per row, runs the NPU
    /// GEMM, rescales on CPU. `out` must be m*n floats.
    pub fn gemm(&self, slot: u64, xs: &[f32], m: usize, out: &mut [f32]) -> Result<()> {
        let (k, n) = (self.k, self.n);
        let (wid, ws) = self
            .weights
            .get(&slot)
            .ok_or_else(|| StrixError::backend("npu: slot not staged"))?;
        let mut xsc = vec![0.0f32; m];
        unsafe {
            let a = self.gemm.a_host as *mut i8;
            for t in 0..m {
                let row = &xs[t * k..(t + 1) * k];
                let amax = row.iter().fold(0.0f32, |mx, &v| mx.max(v.abs()));
                let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                xsc[t] = s;
                let inv = 1.0 / s;
                for i in 0..k {
                    *a.add(t * k + i) = (row[i] * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
            // zero-pad remaining rows so stale data can't bleed in
            if m < M_NPU {
                std::ptr::write_bytes(a.add(m * k), 0, (M_NPU - m) * k);
            }
        }
        self.gemm.start(*wid).map_err(StrixError::backend)?;
        self.gemm.wait().map_err(StrixError::backend)?;
        let acc = unsafe { std::slice::from_raw_parts(self.gemm.out_host as *const i32, m * n) };
        for t in 0..m {
            let s = xsc[t];
            for o in 0..n {
                out[t * n + o] = s * ws[o] * acc[t * n + o] as f32;
            }
        }
        Ok(())
    }
}

/// The Mellum NPU offload bundle: dense q + o shapes, fused gate‖up + down shapes.
pub struct MellumNpu {
    pub q: NpuShape,    // [2304 -> 4096]
    pub o: NpuShape,    // [4096 -> 2304]
    pub gu2: NpuShape,  // expert gate‖up fused [2304 -> 1792] (one call → both)
    pub down: NpuShape, // expert down [896 -> 2304]
}

impl MellumNpu {
    pub fn open(dir: &str) -> Result<MellumNpu> {
        Ok(MellumNpu {
            q: NpuShape::open(dir, 2304, 4096, 8)?,
            o: NpuShape::open(dir, 4096, 2304, 4)?,
            gu2: NpuShape::open(dir, 2304, 1792, 4)?,
            down: NpuShape::open(dir, 896, 2304, 4)?,
        })
    }
}
