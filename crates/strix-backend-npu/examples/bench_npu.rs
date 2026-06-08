//! NPU GEMM bench + verify. Args: <xclbin> <insts> M K N [in_bytes=1] [out_elem=4] [iters=200]
//! Row-major A[M,K], B[K,N] int8 (small values), C[M,N] int32. Verifies bit-exact
//! vs a CPU integer reference, then times. The mlir-aie DMA tiles internally, so
//! host data is plain row-major.
//! Build: cargo run -p strix-backend-npu --features ryzen-ai --release --example bench_npu -- ...
fn main() {
    #[cfg(not(feature = "ryzen-ai"))]
    {
        eprintln!("build with --features ryzen-ai");
    }
    #[cfg(feature = "ryzen-ai")]
    {
        let a: Vec<String> = std::env::args().collect();
        if a.len() < 6 {
            eprintln!(
                "usage: bench_npu <xclbin> <insts> M K N [in_bytes=1] [out_elem=4] [iters=200]"
            );
            return;
        }
        let xclbin = &a[1];
        let insts = &a[2];
        let m: usize = a[3].parse().unwrap();
        let k: usize = a[4].parse().unwrap();
        let n: usize = a[5].parse().unwrap();
        let in_b: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(1);
        let out_e: usize = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(4);
        let iters: usize = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(200);

        let raw = std::fs::read(insts).expect("read insts");
        let instr = strix_backend_npu::load_instr_bin(&raw).expect("parse insts");

        // Small int8 values so the i32 accumulation is exact and a CPU ref matches.
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as i64 % 5 - 2) as i8
        };
        let ai: Vec<i8> = (0..m * k).map(|_| rnd()).collect();
        let bi: Vec<i8> = (0..k * n).map(|_| rnd()).collect();
        let abuf: Vec<u8> = ai.iter().map(|&v| v as u8).collect();
        let bbuf: Vec<u8> = bi.iter().map(|&v| v as u8).collect();
        let out_bytes = m * n * out_e;

        let ctx = strix_backend_npu::NpuContext::open(xclbin, "MLIR_AIE").expect("open ctx");
        let out = ctx
            .run_matmul(&instr, &abuf, &bbuf, out_bytes)
            .expect("warm run");

        if out_e == 4 && (m * n) <= (1 << 22) {
            // CPU reference (row-major)
            let mut cpu = vec![0i32; m * n];
            for i in 0..m {
                for kk in 0..k {
                    let av = ai[i * k + kk] as i32;
                    if av == 0 {
                        continue;
                    }
                    let brow = &bi[kk * n..kk * n + n];
                    let crow = &mut cpu[i * n..i * n + n];
                    for j in 0..n {
                        crow[j] += av * brow[j] as i32;
                    }
                }
            }
            let got: Vec<i32> = out
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mism = (0..m * n).filter(|&i| got[i] != cpu[i]).count();
            if mism == 0 {
                println!("VERIFY: EXACT match vs CPU ({m}x{k}x{n} i8->i32)");
            } else {
                let i = (0..m * n).find(|&i| got[i] != cpu[i]).unwrap();
                println!(
                    "VERIFY: {mism}/{} MISMATCH (first @ {i}: cpu={} npu={})",
                    m * n,
                    cpu[i],
                    got[i]
                );
            }
        }

        let t = std::time::Instant::now();
        for _ in 0..iters {
            ctx.run_matmul(&instr, &abuf, &bbuf, out_bytes)
                .expect("run");
        }
        let per = t.elapsed().as_secs_f64() / iters as f64;
        let gflops = 2.0 * (m * k * n) as f64 / per / 1e9;
        println!(
            "{m}x{k}x{n} in{in_b}B out{out_e}B: {:.3} ms/run  {:.1} GFLOP/s",
            per * 1e3,
            gflops
        );
    }
}
