//! `strix bench-matmul` — Q4_0 GEMV micro-benchmark: CPU vs iGPU.
//!
//! The decode hot path is a sequence of GEMVs (`y = W · x`) against Q4_0 weights.
//! This compares two ways to do one:
//!
//! - **CPU**: dequantize the Q4_0 weight to f32 and dot it with `x` — exactly
//!   what the CPU oracle's `qlinear` does per matmul, per token.
//! - **iGPU**: upload the weight once (resident, repacked Q4_0) and run the fused
//!   dequant→GEMV kernel, uploading only `x` each call.
//!
//! It reports time/iter and effective weight-read bandwidth (the quantity that
//! bounds decode), plus a correctness cross-check.

use std::time::Instant;

use anyhow::Result;
use strix_models::{dequantize, GgmlType};

/// Build a synthetic Q4_0 tensor `[out_dim, in_dim]` (raw GGUF bytes).
fn synth_q4_0(in_dim: usize, out_dim: usize) -> Vec<u8> {
    let nblocks = in_dim / 32;
    let mut bytes = Vec::with_capacity(out_dim * nblocks * 18);
    let mut seed = 0xA5A5_1234u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as u32
    };
    for _ in 0..out_dim * nblocks {
        let d = 0.03f32 + (next() % 32) as f32 * 0.003;
        bytes.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for _ in 0..16 {
            let lo = (next() % 16) as u8;
            let hi = (next() % 16) as u8;
            bytes.push(lo | (hi << 4));
        }
    }
    bytes
}

fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fffff;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

/// CPU GEMV: dequantize the whole Q4_0 weight then dot with `x`.
fn cpu_gemv(weight: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|o| {
            let row = &weight[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x).map(|(w, v)| w * v).sum()
        })
        .collect()
}

pub fn run(in_dim: usize, out_dim: usize, iters: usize) -> Result<()> {
    if in_dim % 32 != 0 {
        anyhow::bail!("in_dim must be a multiple of 32 (Q4_0 block size)");
    }
    let weight_bytes = out_dim * (in_dim / 32) * 18;
    println!("Q4_0 GEMV benchmark");
    println!(
        "  matrix: [{out_dim} x {in_dim}]  ({:.1} MB Q4_0, {} iters)",
        weight_bytes as f64 / 1e6,
        iters
    );
    println!();

    let bytes = synth_q4_0(in_dim, out_dim);
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i as f32 * 0.017).sin()) * 0.5)
        .collect();

    // CPU: dequant + dot per iteration (what the oracle does per token).
    let deq = dequantize(GgmlType::Q4_0, &bytes, out_dim * in_dim)?;
    let _cpu_ref = cpu_gemv(&deq, &x, in_dim, out_dim);
    let t = Instant::now();
    let mut cpu_sink = 0.0f32;
    for _ in 0..iters {
        let deq = dequantize(GgmlType::Q4_0, &bytes, out_dim * in_dim)?;
        let y = cpu_gemv(&deq, &x, in_dim, out_dim);
        cpu_sink += y[0];
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!(
        "  CPU (dequant+dot):  {cpu_ms:8.3} ms/iter   {:6.1} GB/s   (sink {cpu_sink:.3})",
        weight_bytes as f64 / (cpu_ms * 1e-3) / 1e9
    );

    #[cfg(feature = "vulkan")]
    {
        match strix_backend_vulkan::GpuQ4::new() {
            Ok(gpu) => {
                let resident = gpu
                    .resident_from_q4_0(&bytes, in_dim, out_dim)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                // Warm up (pipeline/buffer first-use cost).
                let gpu_ref = gpu
                    .gemv(&resident, &x)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let t = Instant::now();
                let mut gpu_sink = 0.0f32;
                for _ in 0..iters {
                    let y = gpu
                        .gemv(&resident, &x)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    gpu_sink += y[0];
                }
                let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
                println!(
                    "  iGPU (resident Q4): {gpu_ms:8.3} ms/iter   {:6.1} GB/s   (sink {gpu_sink:.3})",
                    weight_bytes as f64 / (gpu_ms * 1e-3) / 1e9
                );
                println!("    on {}", gpu.adapter_name());

                // dp4a (hardware int8 dot) path, if available.
                if gpu.has_dp4a() {
                    let _ = gpu.gemv_dp4a(&resident, &x); // warm
                    let t = Instant::now();
                    let mut s = 0.0f32;
                    for _ in 0..iters {
                        let y = gpu
                            .gemv_dp4a(&resident, &x)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        s += y[0];
                    }
                    let dp_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
                    println!(
                        "  iGPU (dp4a):        {dp_ms:8.3} ms/iter   {:6.1} GB/s   (sink {s:.3})",
                        weight_bytes as f64 / (dp_ms * 1e-3) / 1e9
                    );
                }

                // Correctness cross-check.
                let max_err = _cpu_ref
                    .iter()
                    .zip(&gpu_ref)
                    .map(|(c, g)| (c - g).abs())
                    .fold(0.0f32, f32::max);
                let rel = max_err
                    / _cpu_ref
                        .iter()
                        .map(|v| v.abs())
                        .fold(0.0f32, f32::max)
                        .max(1e-6);
                println!("    max abs err vs CPU: {max_err:.2e}  (rel {rel:.2e})");
                println!();
                if gpu_ms < cpu_ms {
                    println!("  → iGPU {:.2}x faster per GEMV", cpu_ms / gpu_ms);
                } else {
                    println!(
                        "  → CPU still {:.2}x faster (single-GEMV GPU dispatch/readback overhead)",
                        gpu_ms / cpu_ms
                    );
                }
            }
            Err(e) => println!("  iGPU: unavailable ({e})"),
        }
    }
    #[cfg(not(feature = "vulkan"))]
    {
        println!("  iGPU: build with --features vulkan to benchmark the GPU kernel");
    }

    Ok(())
}
