//! Numerical validation of the fused-attention NPU kernel (Q·K^T→softmax→·V).
//! `cargo run -p strix-backend-npu --example npu_attn_test --features ryzen-ai -- <xclbin> <insts>`
//! Feeds bf16 Q‖K‖V (packed, M=L=D=64), runs the kernel, compares to a CPU reference.

use strix_backend_npu::{load_instr_bin, load_instr_txt, run_attn};

fn f2bf(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16 // truncating bf16 (top 16 bits of f32)
}
fn bf2f(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (xclbin, insts_path) = (&a[1], &a[2]);
    // optional shape: m l d lb  (default 64 64 64 32)
    let parse = |i: usize, def: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(def);
    let (m, l, d) = (parse(3, 64), parse(4, 64), parse(5, 64));

    let raw = std::fs::read(insts_path).expect("read insts");
    let insts = match std::str::from_utf8(&raw) {
        Ok(t) => load_instr_txt(t).or_else(|_| load_instr_bin(&raw)).expect("insts"),
        Err(_) => load_instr_bin(&raw).expect("insts"),
    };

    // GQA: nh query heads (m positions each) share ONE kv head (l keys).
    let lb = parse(6, 32);
    let mt = parse(7, m);
    let nh = parse(9, 1);
    // deterministic inputs: q is nh*m*d (head-major), k/v are one shared kv head.
    let q: Vec<f32> = (0..nh * m * d).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let k: Vec<f32> = (0..l * d).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let v: Vec<f32> = (0..l * d).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();

    // pack bf16 LE in the GQA + STREAMING + QUERY-TILED layout:
    //   [ Q (nh*m*d, head-major) | KV blocks replicated per query tile ]
    // nqt = nh * (m/mt) query tiles; each re-reads all KV blocks (host replicates
    // the shared kv head's blocks nqt times — no stride-0 DMA).
    let tph = m / mt;
    let nqt = nh * tph;
    let mut inb: Vec<u8> = Vec::with_capacity((nh * m * d + nqt * 2 * l * d) * 2);
    let push = |buf: &mut Vec<u8>, x: f32| buf.extend_from_slice(&f2bf(x).to_le_bytes());
    for &x in q.iter() {
        push(&mut inb, x);
    }
    for _tile in 0..nqt {
        for b in 0..(l / lb) {
            // K TRANSPOSED to [d, lb] (contiguous key-columns for the matvec)
            for dd in 0..d {
                for r in 0..lb {
                    push(&mut inb, k[(b * lb + r) * d + dd]);
                }
            }
            // V stays [lb, d]
            for r in 0..lb {
                for dd in 0..d {
                    push(&mut inb, v[(b * lb + r) * d + dd]);
                }
            }
        }
    }

    // STRIX_ATTN_LOOP=N: open a PERSISTENT context once, run N times, report
    // per-run latency (isolates the kernel+dispatch cost from xclbin reopen).
    if let Ok(n) = std::env::var("STRIX_ATTN_LOOP").map(|s| s.parse::<u32>().unwrap_or(0)) {
        if n > 0 {
            let ctx = strix_backend_npu::NpuAttnCtx::open(xclbin, "MLIR_AIE", &insts, inb.len(), nh * m * d * 2)
                .expect("attn ctx open");
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                ctx.run(&inb, nh * m * d * 2).expect("ctx run");
            }
            let per = t0.elapsed().as_secs_f64() / n as f64;
            println!("persistent ctx: {n} runs, {:.3} ms/run", per * 1000.0);
            return;
        }
    }
    let out = run_attn(xclbin, "MLIR_AIE", &insts, &inb, nh * m * d * 2).expect("run_attn FAILED");
    let raw_bf: Vec<u16> = out.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let npu: Vec<f32> = raw_bf.iter().map(|&b| bf2f(b)).collect();
    // raw byte diagnostics: 0xFFFF / 0x7FC0 patterns ⇒ kernel never wrote the BO.
    let nan_cnt = npu.iter().filter(|x| x.is_nan()).count();
    let inf_cnt = npu.iter().filter(|x| x.is_infinite()).count();
    let zero_cnt = raw_bf.iter().filter(|&&b| b == 0).count();
    println!(
        "raw bf16[0..6]={:04x?} | NaN {nan_cnt}/{} Inf {inf_cnt} zero {zero_cnt}",
        &raw_bf[..6.min(raw_bf.len())],
        npu.len()
    );

    // CPU reference on bf16-rounded inputs (match the kernel's input precision)
    let r = |x: &f32| bf2f(f2bf(*x));
    let (qb, kb, vb): (Vec<f32>, Vec<f32>, Vec<f32>) =
        (q.iter().map(r).collect(), k.iter().map(r).collect(), v.iter().map(r).collect());
    // causal (arg 8): query position i attends keys 0..=i only.
    let causal = parse(8, 0) == 1;
    // per head h, per position i: attention over the SHARED kv head (GQA).
    let mut cpu = vec![0f32; nh * m * d];
    for hi in 0..nh * m {
        let i = hi % m; // position within the head (causal uses this, not the head)
        let qoff = hi * d;
        let jhi = if causal { (i + 1).min(l) } else { l };
        let mut sc = vec![0f32; jhi];
        for j in 0..jhi {
            let mut s = 0.0;
            for dd in 0..d {
                s += qb[qoff + dd] * kb[j * d + dd];
            }
            sc[j] = s;
        }
        let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        for s in sc.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        for dd in 0..d {
            let mut o = 0.0;
            for j in 0..jhi {
                o += sc[j] * vb[j * d + dd];
            }
            cpu[qoff + dd] = o * inv;
        }
    }

    // Relative L2 norm — the standard attention-accuracy metric. Per-element
    // relative error is meaningless here because the outputs are near-zero-mean.
    // diagnostic: attention restricted to a subrange of keys [lo,hi) — if the NPU
    // matches block0-only or block1-only, the streaming carry/second-block is broken.
    let attn_range = |lo: usize, hi: usize| -> Vec<f32> {
        let mut o = vec![0f32; m * d];
        for i in 0..m {
            let mut sc = vec![0f32; hi - lo];
            for (jx, j) in (lo..hi).enumerate() {
                let mut s = 0.0;
                for dd in 0..d {
                    s += qb[i * d + dd] * kb[j * d + dd];
                }
                sc[jx] = s;
            }
            let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for dd in 0..d {
                let mut acc = 0.0;
                for (jx, j) in (lo..hi).enumerate() {
                    acc += sc[jx] * vb[j * d + dd];
                }
                o[i * d + dd] = acc / sum;
            }
        }
        o
    };
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
        for i in 0..a.len() {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        dot / (na.sqrt() * nb.sqrt())
    };
    // diagnostics compare head-0 only (attn_range covers one head); full uses all.
    let b0 = attn_range(0, lb);
    let bl = attn_range(l - lb, l);
    println!(
        "diag cosine: vs full {:.4} | vs first-block-only(h0) {:.4} | vs last-block-only(h0) {:.4}",
        cos(&npu, &cpu),
        cos(&npu[..m * d], &b0),
        cos(&npu[..m * d], &bl)
    );

    let (mut maxabs, mut err2, mut ref2, mut dot, mut na, mut nb) = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
    let mut bad = false;
    for i in 0..nh * m * d {
        let e = npu[i] - cpu[i];
        if !e.is_finite() {
            bad = true;
            continue;
        }
        maxabs = maxabs.max(e.abs());
        err2 += e * e;
        ref2 += cpu[i] * cpu[i];
        dot += npu[i] * cpu[i];
        na += npu[i] * npu[i];
        nb += cpu[i] * cpu[i];
    }
    let rel_l2 = (err2 / ref2).sqrt();
    let cosine = dot / (na.sqrt() * nb.sqrt());
    println!("npu[0..4]={:?}", &npu[..4]);
    println!("cpu[0..4]={:?}", &cpu[..4]);
    println!("max abs err {maxabs:.5} | rel L2 {rel_l2:.4} | cosine {cosine:.6}");
    // bf16 has ~8 mantissa bits + softmax in bf16 → rel-L2 ~1-3% is expected/correct.
    let pass = !bad && nan_cnt == 0 && inf_cnt == 0 && rel_l2 < 0.05 && cosine > 0.999;
    println!(
        "{}",
        if pass {
            "RESULT: PASS — fused attention computes correctly (within bf16 tolerance)"
        } else {
            "RESULT: FAIL"
        }
    );
}
