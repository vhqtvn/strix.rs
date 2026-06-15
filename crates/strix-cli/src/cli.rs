//! Command-line surface for `strix`.
//!
//! All planned subcommands are declared here so the API shape is visible from
//! day one, even where the implementation is still a stub.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Strix — an experimental Rust local LLM runner for AMD Ryzen AI HX 370.
#[derive(Debug, Parser)]
#[command(name = "strix", version, about, long_about = None)]
pub struct Cli {
    /// Increase log verbosity (-v debug, -vv trace). Overridden by RUST_LOG.
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Inspect a model directory/file without loading weights.
    InspectModel {
        /// Path to a model directory or single weight file.
        #[arg(long)]
        path: PathBuf,
    },

    /// Interactive chat (not implemented yet — Milestone 2+).
    Chat {
        /// Path to the model directory.
        #[arg(long)]
        model: PathBuf,
    },

    /// Generate a completion for a prompt (not implemented yet — Milestone 2).
    Generate {
        /// Path to the model directory.
        #[arg(long)]
        model: PathBuf,
        /// Prompt text.
        #[arg(long)]
        prompt: String,
        /// Maximum number of tokens to generate.
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Feed the prompt verbatim instead of wrapping it in the Gemma chat
        /// template. Use for base models or debugging; instruction-tuned models
        /// (the default targets) need the template — without it they emit garbage.
        #[arg(long)]
        raw: bool,
        /// Offload Q4_0/Q6_K matmuls to the iGPU (requires `--features vulkan`).
        #[arg(long)]
        gpu: bool,
    },

    /// Serve an OpenAI- and Anthropic-compatible HTTP API for a GGUF model.
    Serve {
        /// Path to the model directory or .gguf file.
        #[arg(long)]
        model: PathBuf,
        /// Bind host.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Bind port.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Offload decode to the iGPU (requires `--features rocm`/`vulkan` + STRIX_ROCM=1).
        #[arg(long)]
        gpu: bool,
        /// Max sequence length (KV cache / context budget).
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
    },

    /// Benchmark real inference (not implemented yet — Milestone 2+).
    Bench {
        /// Path to the model directory.
        #[arg(long)]
        model: PathBuf,
        /// Prompt length (tokens) to prefill.
        #[arg(long, default_value_t = 128)]
        prompt_len: usize,
        /// Number of tokens to generate.
        #[arg(long, default_value_t = 128)]
        gen_len: usize,
    },

    /// Run a fake prefill/decode loop and print the benchmark report shape.
    BenchDummy {
        /// Pretend prompt length (tokens).
        #[arg(long, default_value_t = 128)]
        prompt_len: usize,
        /// Pretend number of generated tokens.
        #[arg(long, default_value_t = 128)]
        gen_len: usize,
    },

    /// Print detected hardware (OS, CPU, RAM, Vulkan adapters, NPU placeholder).
    DeviceInfo,

    /// Micro-benchmark a Q4_0 matrix-vector product: CPU dequant+dot vs the iGPU
    /// resident fused-dequant GEMV (the decode-path kernel). Requires
    /// `--features vulkan` for the GPU half.
    BenchMatmul {
        /// Input dimension (must be a multiple of 32). Default ~ Gemma FFN width.
        #[arg(long, default_value_t = 3840)]
        in_dim: usize,
        /// Output dimension. Default ~ Gemma FFN hidden.
        #[arg(long, default_value_t = 15360)]
        out_dim: usize,
        /// Number of timed GEMV iterations.
        #[arg(long, default_value_t = 50)]
        iters: usize,
    },
}
