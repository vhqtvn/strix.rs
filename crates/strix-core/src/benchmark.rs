//! Benchmark metric types.
//!
//! Shared so `bench` (real) and `bench-dummy` (fake) report identically. The
//! metric set mirrors `docs/benchmark-plan.md`.

use std::time::Duration;

/// One benchmark run's measured throughput and latency.
#[derive(Debug, Clone)]
pub struct BenchReport {
    /// Backend that produced these numbers.
    pub backend: String,
    /// Label for the model / scenario.
    pub model: String,
    /// Number of prompt tokens prefilled.
    pub prompt_tokens: usize,
    /// Number of tokens generated during decode.
    pub generated_tokens: usize,
    /// Time to first token (prefill latency).
    pub time_to_first_token: Duration,
    /// Wall-clock spent in prefill.
    pub prefill_time: Duration,
    /// Wall-clock spent in decode.
    pub decode_time: Duration,
    /// Peak resident memory in bytes, if measured.
    pub peak_memory_bytes: Option<u64>,
}

impl BenchReport {
    /// Prefill throughput in tokens/second.
    pub fn prefill_tokens_per_sec(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill_time)
    }

    /// Decode throughput in tokens/second.
    pub fn decode_tokens_per_sec(&self) -> f64 {
        rate(self.generated_tokens, self.decode_time)
    }

    /// Total wall-clock for the run.
    pub fn total_time(&self) -> Duration {
        self.prefill_time + self.decode_time
    }
}

fn rate(count: usize, dur: Duration) -> f64 {
    let secs = dur.as_secs_f64();
    if secs <= 0.0 {
        0.0
    } else {
        count as f64 / secs
    }
}
