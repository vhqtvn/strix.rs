//! `strix-backend-rocm` — AMD ROCm/HIP backend for the Radeon 890M (gfx1150).
//!
//! ROCm 7.x supports Strix Point (gfx1150) natively. This backend links the
//! system HIP runtime (`libamdhip64`) + runtime compiler (`libhiprtc`) — no
//! heavy Rust crates, no build-time hipcc — and compiles the decode kernels at
//! load time, mirroring the ash/Vulkan path. The motivation: HIP streams
//! serialize dependent kernels with far lower per-edge overhead than Vulkan's
//! explicit barriers, so the full decode forward can approach the matmul floor.
//!
//! The `rocm` feature gates everything; the default build is a clean stub.

#[cfg(feature = "rocm")]
pub mod decode;
#[cfg(feature = "rocm")]
mod ffi;
#[cfg(feature = "rocm")]
pub mod hip;
#[cfg(feature = "rocm")]
mod kernels;
#[cfg(feature = "npu")]
mod npu_hybrid;
#[cfg(feature = "rocm")]
pub use decode::RocmWeightAccel;

use strix_core::device::{DeviceInfo, DeviceKind};

/// Probe the ROCm backend for `device-info`.
pub fn device_info() -> DeviceInfo {
    #[cfg(feature = "rocm")]
    {
        match hip::HipGpu::new() {
            Ok(gpu) => {
                let mut i =
                    DeviceInfo::new(DeviceKind::Gpu, gpu.adapter_name().to_string(), "rocm");
                i.notes
                    .push("HIP runtime + hiprtc available (gfx1150)".into());
                i
            }
            Err(e) => {
                let mut i = DeviceInfo::new(DeviceKind::Gpu, "ROCm (unavailable)", "rocm");
                i.notes.push(format!("{e}"));
                i
            }
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let mut i = DeviceInfo::new(DeviceKind::Gpu, "ROCm (disabled)", "rocm");
        i.notes
            .push("build with --features rocm to link the system HIP runtime".into());
        i
    }
}

/// Q4_0 GEMV kernel (HIP C++). Same repacked layout as the Vulkan kernel:
/// contiguous f16 scales (1/block) + contiguous quants (16 nibble bytes/block,
/// read as one coalesced `uint4`). One wavefront (32 threads, gfx1150 wave32)
/// per output row, reduced with `__shfl_down`.
#[cfg(all(test, feature = "rocm"))]
pub(crate) const Q4_GEMV_SRC: &str = r#"
// Self-contained: no hip headers needed (hiprtc lacks them on its include path).
// f16->f32 with subnormal handling (Gemma Q6_K scales are subnormal — learned
// the hard way on the Vulkan path).
__device__ __forceinline__ float h2f(unsigned short h) {
    unsigned s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0u) { float v = (float)m * 5.9604644775390625e-8f; return s ? -v : v; }
    union { unsigned u; float f; } c;
    c.u = (s << 31) | ((e + 112u) << 23) | (m << 13);
    return c.f;
}
extern "C" __global__ void q4_gemv(const unsigned short* __restrict__ scales,
                                   const uint4* __restrict__ quants,
                                   const float* __restrict__ x,
                                   float* __restrict__ y, int in_dim, int out_dim) {
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int nblocks = in_dim / 32;
    int t = threadIdx.x;
    int rowblk = row * nblocks;
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
                float lo = (float)(byte & 0xf) - 8.f;
                float hi = (float)(byte >> 4) - 8.f;
                acc += d * (lo * x[xbase + j] + hi * x[xbase + j + 16]);
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (t == 0) y[row] = acc;
}
"#;

#[cfg(all(test, feature = "rocm"))]
mod tests {
    use super::*;
    use std::os::raw::c_void;

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
    fn f16_to_f32(h: u16) -> f32 {
        let s = ((h >> 15) & 1) as u32;
        let e = ((h >> 10) & 0x1f) as u32;
        let m = (h & 0x3ff) as u32;
        let bits = if e == 0 {
            s << 31
        } else {
            (s << 31) | ((e + 112) << 23) | (m << 13)
        };
        f32::from_bits(bits)
    }

    #[test]
    #[ignore = "requires a ROCm device"]
    fn rocm_q4_gemv_matches_cpu() {
        let (in_dim, out_dim) = (15360usize, 3840usize);
        let nblk = in_dim / 32;
        let total = nblk * out_dim;
        let mut scales = vec![0u16; total];
        let mut quants = vec![0u32; total * 4];
        let mut seed = 0xA5A5_1234u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u32
        };
        for i in 0..total {
            scales[i] = f32_to_f16(0.03 + (next() % 32) as f32 * 0.003);
            for w in 0..4 {
                quants[i * 4 + w] = next();
            }
        }
        let x: Vec<f32> = (0..in_dim)
            .map(|i| (i as f32 * 0.017).sin() * 0.5)
            .collect();

        // CPU reference (same dequant as the kernel).
        let cpu: Vec<f32> = (0..out_dim)
            .map(|row| {
                let mut acc = 0.0f32;
                for b in 0..nblk {
                    let blk = row * nblk + b;
                    let d = f16_to_f32(scales[blk]);
                    let xbase = b * 32;
                    for w in 0..4 {
                        let word = quants[blk * 4 + w];
                        for k in 0..4 {
                            let j = w * 4 + k;
                            let byte = (word >> (k * 8)) & 0xff;
                            let lo = (byte & 0xf) as f32 - 8.0;
                            let hi = (byte >> 4) as f32 - 8.0;
                            acc += d * (lo * x[xbase + j] + hi * x[xbase + j + 16]);
                        }
                    }
                }
                acc
            })
            .collect();

        let gpu = hip::HipGpu::new().expect("hip device");
        let code = hip::compile(Q4_GEMV_SRC).expect("compile");
        let module = gpu.load_module(&code).expect("module");
        let func = gpu.get_function(module, "q4_gemv").expect("func");

        let sc = gpu.upload_new(&scales).unwrap();
        let q = gpu.upload_new(&quants).unwrap();
        let xb = gpu.upload_new(&x).unwrap();
        let yb = gpu.alloc(out_dim * 4).unwrap();

        let mut p_sc = sc.ptr;
        let mut p_q = q.ptr;
        let mut p_x = xb.ptr;
        let mut p_y = yb.ptr;
        let mut in_i = in_dim as i32;
        let mut out_i = out_dim as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_sc as *mut _ as *mut c_void,
            &mut p_q as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut in_i as *mut _ as *mut c_void,
            &mut out_i as *mut _ as *mut c_void,
        ];
        // warm + time
        gpu.launch(func, (out_dim as u32, 1, 1), (32, 1, 1), 0, &mut params)
            .unwrap();
        gpu.sync().unwrap();
        let got = yb.download::<f32>(out_dim).unwrap();

        let iters = 300;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            gpu.launch(func, (out_dim as u32, 1, 1), (32, 1, 1), 0, &mut params)
                .unwrap();
        }
        gpu.sync().unwrap();
        let per = t.elapsed().as_secs_f64() / iters as f64;

        let max_err = cpu
            .iter()
            .zip(&got)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let scale = cpu.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        let bytes = total as f64 * 18.0;
        eprintln!(
            "ROCm Q4 GEMV [{out_dim}x{in_dim}] on {}: {:.4} ms  {:.1} GB/s  rel_err {:.2e}",
            gpu.adapter_name(),
            per * 1e3,
            bytes / per / 1e9,
            max_err / scale
        );
        assert!(max_err / scale < 1e-3, "rocm gemv diverged");
    }
}
