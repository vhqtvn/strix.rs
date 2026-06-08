//! Find vocab tokens containing a substring + print the full chat template.
//! Usage: cargo run -p strix-models --example find_token -- <gguf> <substr>
use strix_models::gguf::{GgufFile, MetaValue};
fn main() {
    let mut a = std::env::args().skip(1);
    let g = GgufFile::open(a.next().unwrap().as_ref()).unwrap();
    let sub = a.next().unwrap_or_default();
    if let Some(MetaValue::String(t)) = g.meta("tokenizer.chat_template") {
        println!("=== chat_template ===\n{}\n", &t[..t.len().min(1200)]);
    }
    let toks = match g.meta("tokenizer.ggml.tokens") {
        Some(MetaValue::Array(a)) => a,
        _ => return,
    };
    let types = match g.meta("tokenizer.ggml.token_type") {
        Some(MetaValue::Array(a)) => a,
        _ => return,
    };
    println!("=== tokens containing {sub:?} ===");
    for (i, t) in toks.iter().enumerate() {
        if let Some(s) = t.as_str() {
            if s.contains(&sub) {
                let ty = types.get(i).and_then(|v| v.as_u64()).unwrap_or(0);
                println!("  id {i}: {s:?} (type {ty})");
            }
        }
    }
}
