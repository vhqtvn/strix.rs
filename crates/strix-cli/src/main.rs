//! `strix` — command-line entry point.

mod cli;
mod commands;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::cli::{Cli, Command};

fn main() -> Result<()> {
    let args = Cli::parse();
    init_tracing(args.verbose);

    match &args.command {
        Command::InspectModel { path } => commands::inspect::run(path),
        Command::DeviceInfo => commands::device_info::run(),
        Command::BenchDummy {
            prompt_len,
            gen_len,
        } => commands::bench_dummy::run(*prompt_len, *gen_len),
        Command::BenchMatmul {
            in_dim,
            out_dim,
            iters,
        } => commands::bench_matmul::run(*in_dim, *out_dim, *iters),

        Command::Generate {
            model,
            prompt,
            max_tokens,
            raw,
            gpu,
        } => commands::generate::run(model, prompt, *max_tokens, !*raw, *gpu),

        // Stubbed until later milestones.
        Command::Chat { model } => commands::stubs::chat(model),
        Command::Bench {
            model,
            prompt_len,
            gen_len,
        } => commands::stubs::bench(model, *prompt_len, *gen_len),
    }
}

/// Initialize tracing. `RUST_LOG` wins; otherwise `-v`/`-vv` set the level.
fn init_tracing(verbose: u8) {
    let default = match verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("strix={default},strix_cli={default}")));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false).with_writer(std::io::stderr))
        .with(filter)
        .init();
}
