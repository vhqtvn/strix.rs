//! `strix inspect-model` — describe a model on disk without loading weights.

use std::path::Path;

use anyhow::Result;
use strix_models::inspect_model;

/// Inspect and pretty-print a model path.
pub fn run(path: &Path) -> Result<()> {
    let report = inspect_model(path)?;

    println!("Model inspection: {}", report.path.display());
    println!("  format:        {}", report.format);
    println!("  tokenizer.json: {}", yes_no(report.has_tokenizer));

    match &report.config {
        Some(cfg) => {
            println!("  architecture:  {}", cfg.architecture);
            println!("  vocab_size:    {}", cfg.vocab_size);
            println!("  hidden_size:   {}", cfg.hidden_size);
            println!("  layers:        {}", cfg.num_hidden_layers);
            println!(
                "  heads:         {} (kv {}, gqa x{})",
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.gqa_groups()
            );
            println!("  head_dim:      {}", cfg.head_dim);
            println!("  rope_theta:    {}", cfg.rope_theta);
            println!("  max_ctx:       {}", cfg.max_position_embeddings);
        }
        None => println!("  config:        <none parsed>"),
    }

    if report.weight_files.is_empty() {
        println!("  weight files:  <none>");
    } else {
        println!("  weight files:  {} file(s)", report.weight_files.len());
        for f in &report.weight_files {
            println!("    - {f}");
        }
    }

    for note in &report.notes {
        println!("  note: {note}");
    }

    Ok(())
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}
