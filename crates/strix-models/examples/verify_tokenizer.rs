//! Dev tool: verify a `tokenizer.json` matches a GGUF's embedded token table.
//!
//! Usage: `cargo run -p strix-models --example verify_tokenizer -- <file.gguf> <tokenizer.json>`

use strix_models::gguf::{GgufFile, MetaValue};
use tokenizers::Tokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let gguf_path = args
        .next()
        .expect("usage: verify_tokenizer <gguf> <tokenizer.json>");
    let tok_path = args
        .next()
        .expect("usage: verify_tokenizer <gguf> <tokenizer.json>");

    let g = GgufFile::open(gguf_path.as_ref()).expect("open gguf");
    let tokens = match g.meta("tokenizer.ggml.tokens") {
        Some(MetaValue::Array(a)) => a,
        _ => panic!("no tokenizer.ggml.tokens array in GGUF"),
    };
    let gguf_tokens: Vec<&str> = tokens.iter().filter_map(|v| v.as_str()).collect();
    println!("GGUF tokens: {}", gguf_tokens.len());

    let tok = Tokenizer::from_file(&tok_path).expect("load tokenizer.json");
    println!("tokenizer.json vocab: {}", tok.get_vocab_size(true));

    let mut mismatches = 0usize;
    let mut shown = 0usize;
    for (id, gtok) in gguf_tokens.iter().enumerate() {
        let jtok = tok.id_to_token(id as u32);
        if jtok.as_deref() != Some(*gtok) {
            mismatches += 1;
            if shown < 10 {
                println!("  MISMATCH id {id}: gguf={gtok:?} json={jtok:?}");
                shown += 1;
            }
        }
    }
    println!(
        "matched {}/{} ({} mismatches)",
        gguf_tokens.len() - mismatches,
        gguf_tokens.len(),
        mismatches
    );
    if mismatches == 0 {
        println!("RESULT: tokenizer.json EXACTLY matches the GGUF token table ✓");
    } else {
        println!("RESULT: tokenizers DIFFER — do not reuse this tokenizer.json");
    }
}
