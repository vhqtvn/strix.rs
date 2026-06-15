use strix_models::chat_template::ChatTemplate;
use strix_models::gguf::GgufFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf path");
    let g = GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
    let t = ChatTemplate::from_gguf(&g).expect("no chat_template in gguf");
    let msgs = vec![
        serde_json::json!({"role":"system","content":"You are a helpful assistant."}),
        serde_json::json!({"role":"user","content":"What is the capital of France?"}),
    ];
    let out = t.render(&msgs, true, None).expect("render");
    println!("---- rendered prompt ----\n{out}\n---- end ----");
}
