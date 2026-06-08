//! Dev tool: dump a GGUF file's metadata and tensor index.
//!
//! Usage: `cargo run -p strix-models --example dump_gguf -- <file.gguf>`

use std::path::PathBuf;

use strix_models::gguf::{GgufFile, MetaValue};

fn main() {
    let path: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: dump_gguf <file.gguf>")
        .into();
    let g = GgufFile::open(&path).expect("open gguf");

    println!("architecture: {:?}", g.architecture());
    println!("\n== metadata ({}) ==", g.metadata().len());
    let mut keys: Vec<_> = g.metadata().keys().cloned().collect();
    keys.sort();
    for k in keys {
        let v = &g.metadata()[&k];
        println!("  {k} = {}", summarize(v));
    }

    println!("\n== tensors ({}) ==", g.tensors().len());
    let mut names: Vec<_> = g.tensors().keys().cloned().collect();
    names.sort();
    // Print the first block's tensors + the globals, to reveal naming.
    for n in &names {
        if true {
            let t = &g.tensors()[n];
            println!("  {n}: {:?} {:?}", t.ggml_type, t.dims);
        }
    }
}

fn summarize(v: &MetaValue) -> String {
    match v {
        MetaValue::Array(a) => {
            let kind = a.first().map(elem_kind).unwrap_or("?");
            // Show full numeric arrays up to 64 elems (per-layer config); short
            // preview for big string arrays (tokenizer vocab).
            let n = if matches!(kind, "str") {
                6
            } else {
                a.len().min(64)
            };
            let preview: Vec<String> = a.iter().take(n).map(short).collect();
            format!("[{} array; len {}] {:?}", kind, a.len(), preview)
        }
        other => short(other),
    }
}

fn elem_kind(v: &MetaValue) -> &'static str {
    match v {
        MetaValue::String(_) => "str",
        MetaValue::F32(_) => "f32",
        MetaValue::I32(_) => "i32",
        MetaValue::U32(_) => "u32",
        _ => "num",
    }
}

fn short(v: &MetaValue) -> String {
    match v {
        MetaValue::String(s) => {
            let t: String = s.chars().take(40).collect();
            format!("{t:?}")
        }
        MetaValue::Array(a) => format!("[array len {}]", a.len()),
        other => format!("{other:?}"),
    }
}
