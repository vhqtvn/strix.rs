//! `strix bench-dummy` — a fake prefill/decode loop with real timing.
//!
//! No model is loaded. Each "token" performs a small fixed chunk of floating
//! point work so the reported tok/s reflects the host and the report shape
//! matches what real `bench` will emit ([`BenchReport`]).

use std::time::Instant;

use anyhow::Result;
use strix_backend_cpu::CpuBackend;
use strix_core::backend::Backend;
use strix_core::benchmark::BenchReport;

/// Per-token synthetic work size (elements summed). Tuned to be small.
const WORK_PER_TOKEN: usize = 50_000;

/// Run the dummy benchmark and print the report.
pub fn run(prompt_len: usize, gen_len: usize) -> Result<()> {
    let backend = CpuBackend::new();

    // Prefill: process the whole prompt "at once" (work scales with prompt_len).
    let prefill_start = Instant::now();
    let _ = synthetic_work(prompt_len.max(1) * WORK_PER_TOKEN);
    let prefill_time = prefill_start.elapsed();
    // Time-to-first-token is the prefill latency in this simple model.
    let time_to_first_token = prefill_time;

    // Decode: one synthetic step per generated token.
    let decode_start = Instant::now();
    for _ in 0..gen_len {
        let _ = synthetic_work(WORK_PER_TOKEN);
    }
    let decode_time = decode_start.elapsed();

    let report = BenchReport {
        backend: backend.name().to_string(),
        model: "<dummy>".to_string(),
        prompt_tokens: prompt_len,
        generated_tokens: gen_len,
        time_to_first_token,
        prefill_time,
        decode_time,
        peak_memory_bytes: None,
    };

    print_report(&report);
    Ok(())
}

/// A tiny, optimizer-resistant floating point workload.
fn synthetic_work(n: usize) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..n {
        // Mix of ops so the loop isn't trivially constant-folded.
        acc += ((i as f64) * 0.5 + 1.0).sqrt();
    }
    acc
}

fn print_report(r: &BenchReport) {
    println!("bench-dummy ({} backend)", r.backend);
    println!("  model:            {}", r.model);
    println!("  prompt tokens:    {}", r.prompt_tokens);
    println!("  generated tokens: {}", r.generated_tokens);
    println!(
        "  time to first tok: {:.3} ms",
        r.time_to_first_token.as_secs_f64() * 1e3
    );
    println!(
        "  prefill:          {:.2} tok/s ({:.3} ms)",
        r.prefill_tokens_per_sec(),
        r.prefill_time.as_secs_f64() * 1e3
    );
    println!(
        "  decode:           {:.2} tok/s ({:.3} ms)",
        r.decode_tokens_per_sec(),
        r.decode_time.as_secs_f64() * 1e3
    );
    println!(
        "  total elapsed:    {:.3} ms",
        r.total_time().as_secs_f64() * 1e3
    );
    println!("  note: synthetic workload — not representative of real inference");
}
