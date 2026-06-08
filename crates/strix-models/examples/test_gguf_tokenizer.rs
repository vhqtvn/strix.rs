//! Dev tool: build a tokenizer from a GGUF and sanity-check it.
//!
//! Usage: `cargo run -p strix-models --example test_gguf_tokenizer -- <file.gguf> "some text"`

use strix_core::tokenizer::Tokenizer;
use strix_models::gguf::GgufFile;
use strix_models::StrixTokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let gguf = args.next().expect("usage: <gguf> [text]");
    let text = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());

    let g = GgufFile::open(gguf.as_ref()).expect("open gguf");
    let tok = StrixTokenizer::from_gguf(&g).expect("build tokenizer");

    println!("vocab_size: {}", tok.vocab_size());
    println!(
        "bos: {:?}  eos: {:?}",
        tok.bos_token_id(),
        tok.eos_token_id()
    );

    let ids = tok.encode(&text, false).expect("encode");
    println!("\nencode({text:?}) -> {ids:?}");
    let back = tok.decode(&ids, false).expect("decode");
    println!("decode -> {back:?}");
    println!(
        "round-trip {}",
        if back == text {
            "EXACT ✓"
        } else {
            "differs (may be fine for SPM)"
        }
    );

    // Per-token view to eyeball segmentation.
    print!("pieces: ");
    for id in &ids {
        print!("[{}]", tok.decode(&[*id], false).unwrap_or_default());
    }
    println!();
}
