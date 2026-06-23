//! MXFP4 coherence pre-check (task #80 spike): does block-scaled FP4 reconstruct
//! MoE expert weights meaningfully better than the uniform Q4_0 that went
//! incoherent? Pure CPU, no kernels — a fast filter before any GPU work.
//!
//! For each sampled expert weight tensor we take the Q8_0-dequantized values as
//! the reference (Q8 is ~lossless), then simulate two 4-bit round-trips per 32-elem
//! block and report RMS-relative error + worst-block error:
//!   - Q4_0  : 16 UNIFORM levels, exact per-block amax scale (what we have today).
//!   - MXFP4 : e2m1 NON-UNIFORM levels {0,.5,1,1.5,2,3,4,6}, power-of-2 block scale.
//! If MXFP4 error << Q4_0 (≈ Q8), the kernel is justified; if ≈ Q4_0, skip it.
//!
//!   cargo run --release -p strix-models --example mxfp4_probe -- <gguf> [tensor...]

use strix_models::gguf::GgufFile;

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Quantize one value to the nearest signed e2m1 level (magnitude from E2M1).
fn q_e2m1(v: f32) -> f32 {
    let s = v.signum();
    let a = v.abs();
    let mut best = E2M1[0];
    let mut bd = (a - E2M1[0]).abs();
    for &l in &E2M1[1..] {
        let d = (a - l).abs();
        if d < bd {
            bd = d;
            best = l;
        }
    }
    s * best
}

/// MXFP4 round-trip of a 32-block: power-of-2 shared scale + per-elem e2m1.
/// Searches a few power-of-2 shared exponents and keeps the lowest-error one, so
/// the comparison isn't penalised by a suboptimal (clipping) scale choice — i.e.
/// this is a best-case MXFP4, strengthening any "still no better than Q4_0" verdict.
fn mxfp4_block(x: &[f32], out: &mut [f32]) {
    let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
    if amax == 0.0 {
        out.iter_mut().for_each(|o| *o = 0.0);
        return;
    }
    let base = amax.log2().floor() - 2.0;
    let mut best_err = f64::INFINITY;
    let mut tmp = [0f32; 32];
    for off in [-1.0f32, 0.0, 1.0, 2.0] {
        let scale = 2f32.powf(base + off);
        let mut e = 0f64;
        for (i, &v) in x.iter().enumerate() {
            tmp[i] = q_e2m1(v / scale) * scale;
            let d = (tmp[i] - v) as f64;
            e += d * d;
        }
        if e < best_err {
            best_err = e;
            out[..x.len()].copy_from_slice(&tmp[..x.len()]);
        }
    }
}

/// Q4_0 round-trip of a 32-block: 16 uniform levels, exact amax scale (ggml-style).
fn q4_0_block(x: &[f32], out: &mut [f32]) {
    let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
    if amax == 0.0 {
        out.iter_mut().for_each(|o| *o = 0.0);
        return;
    }
    let d = amax / 8.0; // levels d*(-8..=7)
    for (o, &v) in out.iter_mut().zip(x) {
        let q = (v / d).round().clamp(-8.0, 7.0);
        *o = q * d;
    }
}

fn err_stats(reference: &[f32], approx: impl Fn(&[f32], &mut [f32])) -> (f64, f64) {
    let mut buf = [0f32; 32];
    let mut se = 0f64; // sum sq error
    let mut sr = 0f64; // sum sq reference (for relative)
    let mut worst = 0f64; // worst per-block rms-rel
    let mut nblk = 0u64;
    for chunk in reference.chunks(32) {
        if chunk.len() < 32 {
            break;
        }
        approx(chunk, &mut buf[..]);
        let mut bse = 0f64;
        let mut bsr = 0f64;
        for (i, &r) in chunk.iter().enumerate() {
            let e = (buf[i] - r) as f64;
            se += e * e;
            sr += (r as f64) * (r as f64);
            bse += e * e;
            bsr += (r as f64) * (r as f64);
        }
        if bsr > 0.0 {
            let brel = (bse / bsr).sqrt();
            if brel > worst {
                worst = brel;
            }
        }
        nblk += 1;
    }
    let _ = nblk;
    ((se / sr).sqrt(), worst)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: mxfp4_probe <gguf> [tensor...]");
    let g = GgufFile::open(std::path::Path::new(&path)).expect("open gguf");

    let tensors: Vec<String> = {
        let rest: Vec<String> = args.collect();
        if !rest.is_empty() {
            rest
        } else {
            // default: a representative set of expert weights across depth.
            let mut v = Vec::new();
            for l in [0usize, 13, 27] {
                for t in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                    v.push(format!("blk.{l}.{t}.weight"));
                }
            }
            v
        }
    };

    println!(
        "{:<32} {:>8}  {:>10} {:>10}  {:>10} {:>10}",
        "tensor", "type", "q4_0 rms", "q4_0 worst", "mxfp4 rms", "mxfp4 worst"
    );
    let (mut sum_q4, mut sum_mx, mut n) = (0f64, 0f64, 0u32);
    for name in &tensors {
        let Some(t) = g.tensors().get(name) else {
            println!("{name:<32}  (missing)");
            continue;
        };
        let ty = t.ggml_type.name();
        let f = match g.dequant_tensor(name) {
            Ok(f) => f,
            Err(e) => {
                println!("{name:<32}  (dequant err: {e})");
                continue;
            }
        };
        // Cap work for the huge expert tensors (~132M elems): first 8M is representative.
        let slice = &f[..f.len().min(8_000_000)];
        let (q4r, q4w) = err_stats(slice, q4_0_block);
        let (mxr, mxw) = err_stats(slice, mxfp4_block);
        println!(
            "{name:<32} {ty:>8}  {:>9.4}% {:>9.4}%  {:>9.4}% {:>9.4}%",
            q4r * 100.0,
            q4w * 100.0,
            mxr * 100.0,
            mxw * 100.0
        );
        sum_q4 += q4r;
        sum_mx += mxr;
        n += 1;
    }
    if n > 0 {
        let (aq, am) = (sum_q4 / n as f64, sum_mx / n as f64);
        println!(
            "\nmean rms-rel: q4_0 {:.4}%  mxfp4 {:.4}%  (mxfp4 is {:.2}× {} q4_0)",
            aq * 100.0,
            am * 100.0,
            (aq / am).max(am / aq),
            if am < aq { "better than" } else { "worse than" }
        );
        println!(
            "verdict: {}",
            if am < aq * 0.6 {
                "MXFP4 clearly better → coherence plausible, kernel worth building"
            } else if am < aq * 0.9 {
                "MXFP4 modestly better → uncertain; full coherence test needed"
            } else {
                "MXFP4 ≈ Q4_0 → unlikely to fix incoherence without QAT; skip"
            }
        );
    }
}
