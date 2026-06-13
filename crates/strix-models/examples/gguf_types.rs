//! Dump GGUF tensor types; flag tensors NOT directly GPU-uploadable
//! (uploadable = Q4_0/Q4_1/Q6K/Q8_0/F32) — candidates for the Q8-repack trick.
use std::collections::BTreeMap;
use std::path::Path;
use strix_models::ggml_quant::GgmlType;
use strix_models::gguf::GgufFile;

fn uploadable(t: GgmlType) -> bool {
    matches!(t, GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q6K | GgmlType::Q8_0 | GgmlType::F32)
}

fn main() {
    for path in std::env::args().skip(1) {
        let g = match GgufFile::open(Path::new(&path)) {
            Ok(g) => g,
            Err(e) => { eprintln!("{path}: {e}"); continue; }
        };
        let mut hist: BTreeMap<String, usize> = BTreeMap::new();
        let mut fallbacks: Vec<(String, String, usize)> = vec![];
        for (name, t) in g.tensors() {
            let ty = format!("{:?}", t.ggml_type);
            *hist.entry(ty.clone()).or_insert(0) += 1;
            if !uploadable(t.ggml_type) {
                fallbacks.push((name.clone(), ty, t.numel()));
            }
        }
        println!("\n=== {} (arch {}) ===", path, g.architecture().unwrap_or("?"));
        println!("  types: {hist:?}");
        if fallbacks.is_empty() {
            println!("  CPU-fallback tensors: NONE (all GPU-uploadable)");
        } else {
            fallbacks.sort_by(|a, b| b.2.cmp(&a.2));
            println!("  CPU-fallback tensors ({}): ", fallbacks.len());
            for (n, ty, ne) in fallbacks.iter().take(8) {
                println!("    {n}  {ty}  {:.1}M elems", *ne as f64 / 1e6);
            }
        }
    }
}
