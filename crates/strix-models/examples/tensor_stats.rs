//! Dev tool: print basic stats for named tensors in a GGUF.
//! Usage: cargo run -p strix-models --example tensor_stats -- <gguf> <name1> [name2 ...]

use strix_models::gguf::GgufFile;

fn main() {
    let mut args = std::env::args().skip(1);
    let gguf = args.next().expect("usage: tensor_stats <gguf> <name...>");
    let g = GgufFile::open(gguf.as_ref()).expect("open");
    for name in args {
        match g.dequant_tensor(&name) {
            Ok(v) => {
                let n = v.len() as f32;
                let mean = v.iter().sum::<f32>() / n;
                let min = v.iter().cloned().fold(f32::MAX, f32::min);
                let max = v.iter().cloned().fold(f32::MIN, f32::max);
                let head: Vec<f32> = v.iter().take(6).cloned().collect();
                println!(
                    "{name}: n={} mean={mean:.4} min={min:.4} max={max:.4} head={head:?}",
                    v.len()
                );
            }
            Err(e) => println!("{name}: ERROR {e}"),
        }
    }
}
