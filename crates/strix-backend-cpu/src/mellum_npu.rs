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

use rayon::prelude::*;
use strix_backend_npu::NpuGemm as Gemm;
use strix_backend_npu::{load_instr_bin, load_instr_txt};
use strix_core::error::{Result, StrixError};
use strix_models::ggml_quant::{dequantize_into, GgmlType};

pub const M_NPU: usize = 256;
/// Below this row count CPU wins (per-call NPU latency dominates).
pub const M_MIN: usize = 64;

/// Phase profiler for the NPU GEMM path (gated by STRIX_NPU_PROF). Accumulates
/// nanoseconds spent in activation quant / NPU start+wait / CPU rescale across all
/// gemm() calls so we can see the real bottleneck. Dump with `npu_prof_dump()`.
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static QUANT_NS: AtomicU64 = AtomicU64::new(0);
    pub static NPU_NS: AtomicU64 = AtomicU64::new(0);
    pub static RESCALE_NS: AtomicU64 = AtomicU64::new(0);
    pub static ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLS: AtomicU64 = AtomicU64::new(0);
    pub fn on() -> bool {
        std::env::var("STRIX_NPU_PROF").is_ok()
    }
    pub fn add(q: u64, n: u64, r: u64) {
        QUANT_NS.fetch_add(q, Ordering::Relaxed);
        NPU_NS.fetch_add(n, Ordering::Relaxed);
        RESCALE_NS.fetch_add(r, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
    }
    pub fn add_attn(ns: u64) {
        ATTN_NS.fetch_add(ns, Ordering::Relaxed);
    }
    pub fn dump(tag: &str) {
        let (q, n, r, a, c) = (
            QUANT_NS.load(Ordering::Relaxed) as f64 / 1e6,
            NPU_NS.load(Ordering::Relaxed) as f64 / 1e6,
            RESCALE_NS.load(Ordering::Relaxed) as f64 / 1e6,
            ATTN_NS.load(Ordering::Relaxed) as f64 / 1e6,
            CALLS.load(Ordering::Relaxed),
        );
        eprintln!(
            "[npu-prof {tag}] {c} gemms | quant {q:.1}ms | NPU {n:.1}ms | rescale {r:.1}ms | attn(CPU) {a:.1}ms",
        );
    }
}

/// Shareable raw pointer for disjoint-index parallel writes.
struct SendPtr<T>(*mut T);
unsafe impl<T> Sync for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}

/// One fixed-shape NPU GEMM (K×N) with N staged weights (per layer / per expert).
pub struct NpuShape {
    gemm: Gemm,
    pub k: usize,
    pub n: usize,
    /// staged weight id + per-channel scale [n], keyed by an opaque slot id
    weights: std::collections::HashMap<u64, (i32, Vec<f32>)>,
}

// SAFETY: callers serialize all `gemm()` calls on a single thread (the NPU branch
// of the expert split / the chunk loop); the Sync bound is only needed so models
// holding NpuShape can be shared with the CPU rayon pool.
unsafe impl Sync for NpuShape {}

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
        let bpr = (k / ty.block_elems()) * ty.block_bytes();
        let mut b8 = vec![0i8; k * n];
        let mut ws = vec![0.0f32; n];
        // Parallel over output channels: dequant+scale each row, scatter [K,N]-major.
        // b8 columns are disjoint per channel — share the buffer via raw pointer.
        let b8p = SendPtr(b8.as_mut_ptr());
        ws.par_iter_mut().enumerate().for_each_init(
            || vec![0.0f32; k],
            |rowf, (o, wso)| {
                let b8p = &b8p;
                dequantize_into(ty, &bytes[o * bpr..(o + 1) * bpr], rowf).unwrap();
                let amax = rowf.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                *wso = s;
                let inv = 1.0 / s;
                for i in 0..k {
                    unsafe {
                        *b8p.0.add(i * n + o) = (rowf[i] * inv).round().clamp(-127.0, 127.0) as i8
                    };
                }
            },
        );
        let wid = self.gemm.stage(&b8).map_err(StrixError::backend)?;
        self.weights.insert(slot, (wid, ws));
        Ok(())
    }

    /// Stage two stacked Q8_0 weights (e.g. gate ‖ up, each n/2 rows of k) as one
    /// [K, N] weight: output channels [0,n/2) from `a`, [n/2,n) from `b`.
    pub fn stage_q8_pair(&mut self, slot: u64, a: &[u8], b: &[u8], ty: GgmlType) -> Result<()> {
        let (k, n) = (self.k, self.n);
        let half = n / 2;
        let bpr = (k / ty.block_elems()) * ty.block_bytes();
        let mut b8 = vec![0i8; k * n];
        let mut ws = vec![0.0f32; n];
        let b8p = SendPtr(b8.as_mut_ptr());
        ws.par_iter_mut().enumerate().for_each_init(
            || vec![0.0f32; k],
            |rowf, (o, wso)| {
                let b8p = &b8p;
                let (src, r) = if o < half { (a, o) } else { (b, o - half) };
                dequantize_into(ty, &src[r * bpr..(r + 1) * bpr], rowf).unwrap();
                let amax = rowf.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                *wso = s;
                let inv = 1.0 / s;
                for i in 0..k {
                    unsafe {
                        *b8p.0.add(i * n + o) = (rowf[i] * inv).round().clamp(-127.0, 127.0) as i8
                    };
                }
            },
        );
        let wid = self.gemm.stage(&b8).map_err(StrixError::backend)?;
        self.weights.insert(slot, (wid, ws));
        Ok(())
    }

    /// Stage three stacked Q8_0 weights (q‖k‖v) as one [K, N] weight: output
    /// channels [0,n0) from `a`, [n0,n0+n1) from `b`, [n0+n1,n) from `c`.
    pub fn stage_q8_triple(
        &mut self,
        slot: u64,
        a: &[u8],
        b: &[u8],
        c: &[u8],
        n0: usize,
        n1: usize,
        ty: GgmlType,
    ) -> Result<()> {
        let (k, n) = (self.k, self.n);
        let bpr = (k / ty.block_elems()) * ty.block_bytes();
        let mut b8 = vec![0i8; k * n];
        let mut ws = vec![0.0f32; n];
        let b8p = SendPtr(b8.as_mut_ptr());
        ws.par_iter_mut().enumerate().for_each_init(
            || vec![0.0f32; k],
            |rowf, (o, wso)| {
                let b8p = &b8p;
                let (src, r) = if o < n0 {
                    (a, o)
                } else if o < n0 + n1 {
                    (b, o - n0)
                } else {
                    (c, o - n0 - n1)
                };
                dequantize_into(ty, &src[r * bpr..(r + 1) * bpr], rowf).unwrap();
                let amax = rowf.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                *wso = s;
                let inv = 1.0 / s;
                for i in 0..k {
                    unsafe {
                        *b8p.0.add(i * n + o) = (rowf[i] * inv).round().clamp(-127.0, 127.0) as i8
                    };
                }
            },
        );
        let wid = self.gemm.stage(&b8).map_err(StrixError::backend)?;
        self.weights.insert(slot, (wid, ws));
        Ok(())
    }

    /// Fused GEMM that splits the [m,n] output into three column ranges
    /// [0,n0)/[n0,n0+n1)/[n0+n1,n) → `o0`/`o1`/`o2` (each row-major [m, width]).
    pub fn gemm_split3(
        &self,
        slot: u64,
        xs: &[f32],
        m: usize,
        n0: usize,
        n1: usize,
        o0: &mut [f32],
        o1: &mut [f32],
        o2: &mut [f32],
    ) -> Result<()> {
        let n = self.n;
        let mut fused = vec![0.0f32; m * n];
        self.gemm(slot, xs, m, &mut fused)?;
        let n2 = n - n0 - n1;
        for t in 0..m {
            let row = &fused[t * n..(t + 1) * n];
            o0[t * n0..(t + 1) * n0].copy_from_slice(&row[..n0]);
            o1[t * n1..(t + 1) * n1].copy_from_slice(&row[n0..n0 + n1]);
            o2[t * n2..(t + 1) * n2].copy_from_slice(&row[n0 + n1..]);
        }
        Ok(())
    }

    /// Fused GEMM splitting [m,n] into two equal halves → `o0`/`o1` (gate‖up).
    pub fn gemm_split2(
        &self,
        slot: u64,
        xs: &[f32],
        m: usize,
        o0: &mut [f32],
        o1: &mut [f32],
    ) -> Result<()> {
        let n = self.n;
        let half = n / 2;
        let mut fused = vec![0.0f32; m * n];
        self.gemm(slot, xs, m, &mut fused)?;
        for t in 0..m {
            let row = &fused[t * n..(t + 1) * n];
            o0[t * half..(t + 1) * half].copy_from_slice(&row[..half]);
            o1[t * half..(t + 1) * half].copy_from_slice(&row[half..]);
        }
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
        let prof = prof::on();
        let t0 = std::time::Instant::now();
        let mut xsc = vec![0.0f32; m];
        let ap = SendPtr(self.gemm.a_host as *mut i8);
        xsc.par_iter_mut().enumerate().for_each(|(t, sc)| {
            let ap = &ap;
            let row = &xs[t * k..(t + 1) * k];
            let amax = row.iter().fold(0.0f32, |mx, &v| mx.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            *sc = s;
            let inv = 1.0 / s;
            for i in 0..k {
                unsafe { *ap.0.add(t * k + i) = (row[i] * inv).round().clamp(-127.0, 127.0) as i8 };
            }
        });
        // zero-pad remaining rows so stale data can't bleed in
        if m < M_NPU {
            unsafe {
                std::ptr::write_bytes((self.gemm.a_host as *mut i8).add(m * k), 0, (M_NPU - m) * k)
            };
        }
        let t1 = std::time::Instant::now();
        self.gemm.start(*wid).map_err(StrixError::backend)?;
        self.gemm.wait().map_err(StrixError::backend)?;
        let t2 = std::time::Instant::now();
        let acc = unsafe { std::slice::from_raw_parts(self.gemm.out_host as *const i32, m * n) };
        out[..m * n]
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(t, orow)| {
                let s = xsc[t];
                for o in 0..n {
                    orow[o] = s * ws[o] * acc[t * n + o] as f32;
                }
            });
        if prof {
            let t3 = std::time::Instant::now();
            prof::add(
                (t1 - t0).as_nanos() as u64,
                (t2 - t1).as_nanos() as u64,
                (t3 - t2).as_nanos() as u64,
            );
        }
        Ok(())
    }
}

/// The Mellum NPU offload bundle: dense q + o shapes, fused gate‖up + down shapes.
pub struct MellumNpu {
    pub q: NpuShape,    // [2304 -> 4096]
    pub o: NpuShape,    // [4096 -> 2304]
    pub kv: NpuShape,   // [2304 -> 512] attn_k / attn_v (slots il*2 / il*2+1)
    pub gu2: NpuShape,  // expert gate‖up fused [2304 -> 1792] (one call → both)
    pub down: NpuShape, // expert down [896 -> 2304]
}

impl MellumNpu {
    pub fn open(dir: &str) -> Result<MellumNpu> {
        Ok(MellumNpu {
            q: NpuShape::open(dir, 2304, 4096, 8)?,
            o: NpuShape::open(dir, 4096, 2304, 4)?,
            kv: NpuShape::open(dir, 2304, 512, 8)?,
            gu2: NpuShape::open(dir, 2304, 1792, 4)?,
            down: NpuShape::open(dir, 896, 2304, 4)?,
        })
    }
}

/// Qwen3.6 NPU offload: the dense projections (deltanet qkv/gate/ssm_out + full-attn
/// q/o — all Q8_0 in the UD quant). The 256-expert MoE (~30 GB int8) exceeds the BO
/// pool and stays on CPU.
pub struct QwenNpu {
    pub p8192: NpuShape, // [2048 -> 8192] attn_qkv (deltanet) + attn_q (attn layers)
    pub p4096: NpuShape, // [2048 -> 4096] attn_gate (deltanet)
    pub p2048: NpuShape, // [4096 -> 2048] ssm_out (deltanet) + attn_output (attn layers)
}

impl QwenNpu {
    pub fn open(dir: &str) -> Result<QwenNpu> {
        Ok(QwenNpu {
            p8192: NpuShape::open(dir, 2048, 8192, 8)?,
            p4096: NpuShape::open(dir, 2048, 4096, 8)?,
            p2048: NpuShape::open(dir, 4096, 2048, 8)?,
        })
    }
}

/// SmolLM3-3B dense prefill shapes (Q4_0 weights, requantized to int8 at stage).
/// q/o share 2048x2048; k/v share 2048x512; gate/up share 2048x11008; down 11008x2048.
pub struct SmolLm3Npu {
    pub qo: NpuShape,
    pub kv: NpuShape,
    pub gu: NpuShape,
    pub down: NpuShape,
    pub qkv: Option<NpuShape>, // 2048 -> 3072 (2048+512+512)
    pub gu2: Option<NpuShape>, // 2048 -> 22016 (11008+11008)
}

impl SmolLm3Npu {
    pub fn open(dir: &str) -> Result<SmolLm3Npu> {
        Ok(SmolLm3Npu {
            qo: NpuShape::open(dir, 2048, 2048, 8)?,
            kv: NpuShape::open(dir, 2048, 512, 8)?,
            gu: NpuShape::open(dir, 2048, 11008, 4)?,
            down: NpuShape::open(dir, 11008, 2048, 8)?,
            qkv: NpuShape::open(dir, 2048, 3072, 8).ok(),
            gu2: NpuShape::open(dir, 2048, 22016, 8).ok(),
        })
    }
}

/// Qwen3-4B dense prefill shapes.
pub struct Qwen3Npu {
    pub q: NpuShape,    // 2560 -> 4096
    pub kv: NpuShape,   // 2560 -> 1024
    pub o: NpuShape,    // 4096 -> 2560
    pub gu: NpuShape,   // 2560 -> 9728
    pub down: NpuShape, // 9728 -> 2560
    // Fusion (stage 1): one dispatch for q‖k‖v and gate‖up. None if xclbin absent.
    pub qkv: Option<NpuShape>, // 2560 -> 6144 (4096+1024+1024)
    pub gu2: Option<NpuShape>, // 2560 -> 19456 (9728+9728)
}

impl Qwen3Npu {
    pub fn open(dir: &str) -> Result<Qwen3Npu> {
        Ok(Qwen3Npu {
            q: NpuShape::open(dir, 2560, 4096, 8)?,
            kv: NpuShape::open(dir, 2560, 1024, 8)?,
            o: NpuShape::open(dir, 4096, 2560, 8)?,
            gu: NpuShape::open(dir, 2560, 9728, 8)?,
            down: NpuShape::open(dir, 9728, 2560, 8)?,
            qkv: NpuShape::open(dir, 2560, 6144, 8).ok(),
            gu2: NpuShape::open(dir, 2560, 19456, 8).ok(),
        })
    }
}

/// Gemma-3n-E4B dense prefill shapes (q/o + k/v reuse SmolLM3's 2048x2048 / 2048x512).
pub struct Gemma3nNpu {
    pub qo: NpuShape,     // 2048 -> 2048
    pub kv: NpuShape,     // 2048 -> 512
    pub gu: NpuShape,     // 2048 -> 16384
    pub down: NpuShape,   // 16384 -> 2048
    pub plproj: NpuShape, // 2048 -> 8960 (per_layer_model_proj)
    pub qkv: Option<NpuShape>, // 2048 -> 3072 (2048+512+512); gate/up vary per layer → not fused
}

impl Gemma3nNpu {
    pub fn open(dir: &str) -> Result<Gemma3nNpu> {
        Ok(Gemma3nNpu {
            qo: NpuShape::open(dir, 2048, 2048, 8)?,
            kv: NpuShape::open(dir, 2048, 512, 8)?,
            gu: NpuShape::open(dir, 2048, 16384, 8)?,
            down: NpuShape::open(dir, 16384, 2048, 8)?,
            plproj: NpuShape::open(dir, 2048, 8960, 4)?,
            qkv: NpuShape::open(dir, 2048, 3072, 8).ok(),
        })
    }
}
