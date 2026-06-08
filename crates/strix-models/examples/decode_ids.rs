//! Decode specific token ids: cargo run -p strix-models --example decode_ids -- <gguf> <id>...
use strix_core::tokenizer::Tokenizer;
use strix_models::{gguf::GgufFile, StrixTokenizer};
fn main() {
    let mut a = std::env::args().skip(1);
    let g = GgufFile::open(a.next().unwrap().as_ref()).unwrap();
    let tok = StrixTokenizer::from_gguf(&g).unwrap();
    for s in a {
        let id: u32 = s.parse().unwrap();
        println!("{id} => {:?}", tok.decode(&[id], false));
    }
}
