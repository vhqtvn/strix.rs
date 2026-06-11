//! Load + correctness test for a freshly-built xclbin.
//! `cargo run -p strix-backend-npu --example npu_loadtest --features ryzen-ai -- <xclbin> <insts> <K> <N>`
//! Runs a 256xKxN int8 GEMM on the NPU and compares a few entries to a CPU reference.

use strix_backend_npu::{load_instr_bin, load_instr_txt, NpuGemm};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (xclbin, insts_path, k, n): (&str, &str, usize, usize) = (
        &a[1],
        &a[2],
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
    );
    const M: usize = 256;
    let raw = std::fs::read(insts_path).expect("read insts");
    let insts = match std::str::from_utf8(&raw) {
        Ok(t) => load_instr_txt(t).or_else(|_| load_instr_bin(&raw)).expect("insts"),
        Err(_) => load_instr_bin(&raw).expect("insts"),
    };
    println!("insts words: {}", insts.len());

    let mut g = NpuGemm::open(xclbin, "MLIR_AIE", &insts, M * k, M * n * 4)
        .expect("Gemm::open FAILED (xclbin did not load on NPU)");
    println!("xclbin loaded on NPU OK");

    // Deterministic small int8 A [M][K], B [K][N] (row-major).
    let av: Vec<i8> = (0..M * k).map(|i| ((i % 7) as i8) - 3).collect();
    let bv: Vec<i8> = (0..k * n).map(|i| ((i % 5) as i8) - 2).collect();
    unsafe {
        std::ptr::copy_nonoverlapping(av.as_ptr(), g.a_host as *mut i8, M * k);
    }
    let wid = g.stage(&bv).expect("stage");
    g.start(wid).expect("start");
    g.wait().expect("wait");
    let out = unsafe { std::slice::from_raw_parts(g.out_host as *const i32, M * n) };

    // CPU reference for 3 sample (m,n) cells.
    let mut ok = true;
    for &(mm, nn) in &[(0usize, 0usize), (5, 17), (200, 511.min(n - 1))] {
        let mut acc: i64 = 0;
        for kk in 0..k {
            acc += av[mm * k + kk] as i64 * bv[kk * n + nn] as i64;
        }
        let got = out[mm * n + nn] as i64;
        let m = if got == acc { "OK" } else { ok = false; "MISMATCH" };
        println!("C[{mm}][{nn}] npu={got} cpu={acc} {m}");
    }
    println!("{}", if ok { "RESULT: PASS — NPU computes correctly" } else { "RESULT: FAIL" });
}
