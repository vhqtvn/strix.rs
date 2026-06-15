//! `strix serve` — OpenAI- and Anthropic-compatible HTTP API server.
//!
//! Renders each model's OWN Jinja chat template (via `minijinja`, from the GGUF
//! `tokenizer.chat_template`), tokenizes with the real HF tokenizer
//! (`tokenizer.json`), and drives the resident GPU / CPU decode. A single stateful
//! model holds one KV cache, so requests are **serialized** behind a mutex.
//!
//! Endpoints:
//!   GET  /v1/models
//!   POST /v1/chat/completions   (OpenAI; `stream:true` → SSE)
//!   POST /v1/completions        (OpenAI legacy text completion; `stream` → SSE)
//!   POST /v1/messages           (Anthropic; `stream:true` → SSE event stream)

use std::io::{Read, Write};
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use strix_core::backend::Decoder;
use strix_core::tokenizer::Tokenizer;
use strix_models::chat_template::ChatTemplate;
use strix_models::gguf::GgufFile;
use strix_models::StrixTokenizer;
use tiny_http::{Header, Response, Server};

use super::generate::{build_weight_accel, find_gguf};

/// A loaded model + everything the server needs to drive it.
struct Model {
    decoder: Box<dyn Decoder>,
    tok: StrixTokenizer,
    template: Option<ChatTemplate>,
    /// Token ids that end generation (eos / end-of-turn).
    eos: Vec<u32>,
    arch: String,
    name: String,
}

/// Sampling + stopping parameters for one request.
struct GenParams {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    stop: Vec<String>,
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams {
            max_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            stop: Vec::new(),
        }
    }
}

// ============================== model loading ==============================

fn load_tokenizer(gguf_path: &Path, gguf: &GgufFile) -> Result<StrixTokenizer> {
    // Prefer a real tokenizer.json (full BPE/Unigram); search the model dir and a
    // `tok/` subdir next to the GGUF. Fall back to the GGUF Unigram reconstruction.
    let dir = gguf_path.parent().unwrap_or_else(|| Path::new("."));
    for cand in [dir.join("tokenizer.json"), dir.join("tok/tokenizer.json")] {
        if cand.is_file() {
            if let Ok(t) = StrixTokenizer::from_file(&cand) {
                return Ok(t);
            }
        }
    }
    StrixTokenizer::from_gguf(gguf).context("no tokenizer.json and GGUF tokenizer unsupported")
}

#[allow(unused_variables, unused_mut)]
fn build_decoder(arch: &str, gguf: GgufFile, gpu: bool, ctx: usize) -> Result<Box<dyn Decoder>> {
    use strix_backend_cpu::{
        gemma::GemmaModel, gemma3n::Gemma3nModel, mellum::MellumModel, qwen3::Qwen3Model,
        qwen35::Qwen35Model, smollm3::SmolLm3Model,
    };
    let accel = if gpu { build_weight_accel() } else { None };
    let attach = |n: usize| eprintln!("[serve] {n} weights resident on iGPU");
    match arch {
        "qwen3" => {
            let mut m = Qwen3Model::from_gguf(gguf, ctx).context("build qwen3")?;
            if let Some(a) = accel {
                attach(m.attach_accel(a));
            }
            Ok(Box::new(m))
        }
        "smollm3" => {
            let mut m = SmolLm3Model::from_gguf(gguf, ctx).context("build smollm3")?;
            if let Some(a) = accel {
                attach(m.attach_accel(a));
            }
            Ok(Box::new(m))
        }
        "gemma3n" => {
            let mut m = Gemma3nModel::from_gguf(gguf, ctx).context("build gemma3n")?;
            if let Some(a) = accel {
                attach(m.attach_accel(a));
            }
            Ok(Box::new(m))
        }
        "mellum" => {
            let mut m = MellumModel::from_gguf(gguf).context("build mellum")?;
            if let Some(a) = accel {
                attach(m.attach_accel(a));
            }
            Ok(Box::new(m))
        }
        "qwen35" | "qwen35moe" => {
            let mut m = Qwen35Model::from_gguf(gguf).context("build qwen35")?;
            if let Some(a) = accel {
                attach(m.attach_accel(a));
                if m.enable_gpu_decode(ctx) {
                    eprintln!("[serve] qwen35 resident on-device decode enabled");
                }
            }
            Ok(Box::new(m))
        }
        "gemma4" | "gemma3" | "gemma" => {
            let mut m = GemmaModel::from_gguf(gguf, ctx).context("build gemma")?;
            if let Some(a) = accel {
                attach(m.attach_accel(a));
            }
            Ok(Box::new(m))
        }
        other => Err(anyhow!("serve: unsupported architecture `{other}`")),
    }
}

fn load_model(path: &Path, gpu: bool, ctx: usize) -> Result<Model> {
    let gguf_path = find_gguf(path).ok_or_else(|| anyhow!("no .gguf found at {}", path.display()))?;
    let gguf = GgufFile::open(&gguf_path).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("?").to_string();
    let template = ChatTemplate::from_gguf(&gguf);
    // End-of-generation token ids from GGUF metadata.
    let mut eos = Vec::new();
    {
        let md = gguf.metadata();
        for k in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
            if let Some(v) = md.get(k).and_then(|v| v.as_u64()) {
                eos.push(v as u32);
            }
        }
    }
    let tok = load_tokenizer(&gguf_path, &gguf)?;
    let name = gguf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let decoder = build_decoder(&arch, gguf, gpu, ctx)?;
    eos.dedup();
    Ok(Model {
        decoder,
        tok,
        template,
        eos,
        arch,
        name,
    })
}

// ============================== sampling ==============================

/// Tiny non-cryptographic RNG (xorshift) so we don't pull in `rand`.
struct Rng(u64);
impl Rng {
    fn new() -> Self {
        let s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15)
            | 1;
        Rng(s)
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Sample a token id from logits with temperature / top-k / top-p (nucleus).
/// `temperature <= 0` → greedy argmax.
fn sample_token(logits: &[f32], p: &GenParams, rng: &mut Rng) -> u32 {
    if p.temperature <= 0.0 {
        let mut bi = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        return bi as u32;
    }
    // softmax with temperature over (optionally) top-k candidates.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let k = if p.top_k > 0 {
        p.top_k.min(idx.len())
    } else {
        idx.len()
    };
    idx.truncate(k);
    let max = logits[idx[0]];
    let inv_t = 1.0 / p.temperature;
    let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i] - max) * inv_t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for x in probs.iter_mut() {
        *x /= sum;
    }
    // nucleus: keep the smallest prefix whose cumulative prob >= top_p.
    if p.top_p < 1.0 {
        let mut cum = 0.0;
        let mut cut = probs.len();
        for (i, &pr) in probs.iter().enumerate() {
            cum += pr;
            if cum >= p.top_p {
                cut = i + 1;
                break;
            }
        }
        idx.truncate(cut);
        probs.truncate(cut);
        let s: f32 = probs.iter().sum();
        for x in probs.iter_mut() {
            *x /= s;
        }
    }
    let r = rng.next_f32();
    let mut cum = 0.0;
    for (i, &pr) in probs.iter().enumerate() {
        cum += pr;
        if r <= cum {
            return idx[i] as u32;
        }
    }
    idx[idx.len() - 1] as u32
}

// ============================== generation ==============================

/// Drive prefill + decode. Calls `on_token(piece)` for each new decoded text piece
/// (incremental detokenization). Returns (full_text, finish_reason, n_gen).
fn generate(
    m: &mut Model,
    prompt_ids: &[u32],
    p: &GenParams,
    mut on_token: impl FnMut(&str),
) -> Result<(String, &'static str, usize)> {
    m.decoder.reset();
    let mut rng = Rng::new();
    let logits = m.decoder.prefill(prompt_ids).context("prefill")?;
    let mut next = sample_token(&logits.0, p, &mut rng);

    let mut out_ids: Vec<u32> = Vec::new();
    let mut emitted = String::new(); // text already sent to on_token
    let mut finish = "length";
    for step in 0..p.max_tokens {
        if m.eos.contains(&next) {
            finish = "stop";
            break;
        }
        out_ids.push(next);
        // Incremental detokenize: decode the whole sequence, emit the new suffix.
        // (Robust across BPE merges that span tokens.)
        if let Ok(full) = m.tok.decode(&out_ids, true) {
            if full.len() > emitted.len() && full.is_char_boundary(emitted.len()) {
                let piece = full[emitted.len()..].to_string();
                if !piece.is_empty() {
                    on_token(&piece);
                    emitted = full;
                }
            }
        }
        // stop sequences
        if !p.stop.is_empty() && p.stop.iter().any(|s| emitted.contains(s.as_str())) {
            finish = "stop";
            break;
        }
        if step + 1 >= p.max_tokens {
            break;
        }
        let logits = m.decoder.decode_one(next).context("decode")?;
        next = sample_token(&logits.0, p, &mut rng);
    }
    // trim at the earliest stop sequence
    let mut text = m.tok.decode(&out_ids, true).unwrap_or_default();
    for s in &p.stop {
        if let Some(idx) = text.find(s.as_str()) {
            text.truncate(idx);
        }
    }
    Ok((text, finish, out_ids.len()))
}

/// Render `messages` to prompt ids via the model's chat template + tokenizer.
fn encode_chat(m: &Model, messages: &[Value], tools: Option<&Value>) -> Result<Vec<u32>> {
    let tmpl = m
        .template
        .as_ref()
        .ok_or_else(|| anyhow!("model has no chat template"))?;
    let prompt = tmpl.render(messages, true, tools)?;
    m.tok
        .encode(&prompt, false)
        .map_err(|e| anyhow!("encode: {e}"))
}

// ============================== HTTP server ==============================

pub fn run(path: &Path, host: &str, port: u16, gpu: bool, ctx: usize) -> Result<()> {
    let model = load_model(path, gpu, ctx)?;
    eprintln!(
        "[serve] loaded `{}` (arch {}), chat_template: {}, eos ids: {:?}",
        model.name,
        model.arch,
        if model.template.is_some() { "yes" } else { "NONE" },
        model.eos,
    );
    let model_id = model.name.clone();
    let shared = Arc::new(Mutex::new(model));
    let addr = format!("{host}:{port}");
    let server = Server::http(&addr).map_err(|e| anyhow!("bind {addr}: {e}"))?;
    eprintln!("[serve] listening on http://{addr}  (model id: {model_id})");

    for req in server.incoming_requests() {
        let url = req.url().to_string();
        let method = req.method().to_string();
        if method == "GET" && (url == "/v1/models" || url == "/models") {
            let body = json!({
                "object": "list",
                "data": [{ "id": model_id, "object": "model", "owned_by": "strix" }]
            });
            respond_json(req, 200, &body);
            continue;
        }
        if method != "POST" {
            respond_json(req, 404, &json!({"error": "not found"}));
            continue;
        }
        let route = if url.starts_with("/v1/chat/completions") {
            Route::OpenAiChat
        } else if url.starts_with("/v1/completions") {
            Route::OpenAiText
        } else if url.starts_with("/v1/messages") {
            Route::Anthropic
        } else {
            respond_json(req, 404, &json!({"error": format!("unknown route {url}")}));
            continue;
        };
        handle(req, route, &shared, &model_id);
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Route {
    OpenAiChat,
    OpenAiText,
    Anthropic,
}

fn handle(mut req: tiny_http::Request, route: Route, shared: &Arc<Mutex<Model>>, model_id: &str) {
    let mut body = String::new();
    if req.as_reader().read_to_string(&mut body).is_err() {
        respond_json(req, 400, &json!({"error": "bad body"}));
        return;
    }
    let v: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            respond_json(req, 400, &json!({"error": format!("bad json: {e}")}));
            return;
        }
    };
    let stream = v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let params = parse_params(&v, route);

    // Build the prompt ids (needs the model lock for the tokenizer/template).
    let prompt_ids = {
        let m = shared.lock().unwrap();
        let r = match route {
            Route::OpenAiText => v
                .get("prompt")
                .and_then(|p| p.as_str())
                .map(|s| m.tok.encode(s, true).map_err(|e| anyhow!("{e}")))
                .unwrap_or_else(|| Err(anyhow!("missing prompt"))),
            Route::OpenAiChat => match v.get("messages").and_then(|x| x.as_array()) {
                Some(msgs) => encode_chat(&m, msgs, v.get("tools")),
                None => Err(anyhow!("missing messages")),
            },
            Route::Anthropic => encode_anthropic(&m, &v),
        };
        r
    };
    let prompt_ids = match prompt_ids {
        Ok(ids) if !ids.is_empty() => ids,
        Ok(_) => {
            respond_json(req, 400, &json!({"error": "empty prompt"}));
            return;
        }
        Err(e) => {
            respond_json(req, 400, &json!({"error": e.to_string()}));
            return;
        }
    };

    if stream {
        stream_response(req, route, shared.clone(), prompt_ids, params, model_id.to_string());
    } else {
        blocking_response(req, route, shared, prompt_ids, params, model_id);
    }
}

fn blocking_response(
    req: tiny_http::Request,
    route: Route,
    shared: &Arc<Mutex<Model>>,
    prompt_ids: Vec<u32>,
    params: GenParams,
    model_id: &str,
) {
    let mut m = shared.lock().unwrap();
    let n_prompt = prompt_ids.len();
    let res = generate(&mut m, &prompt_ids, &params, |_| {});
    drop(m);
    match res {
        Ok((text, finish, n_gen)) => {
            let body = match route {
                Route::Anthropic => anthropic_message(&text, finish, model_id, n_prompt, n_gen),
                Route::OpenAiText => openai_text(&text, finish, model_id, n_prompt, n_gen),
                Route::OpenAiChat => openai_chat(&text, finish, model_id, n_prompt, n_gen),
            };
            respond_json(req, 200, &body);
        }
        Err(e) => respond_json(req, 500, &json!({"error": e.to_string()})),
    }
}

fn parse_params(v: &Value, route: Route) -> GenParams {
    let f = |k: &str, d: f32| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(d);
    let max_key = match route {
        Route::Anthropic => "max_tokens",
        _ => "max_tokens",
    };
    let mut stop = Vec::new();
    match v.get("stop").or_else(|| v.get("stop_sequences")) {
        Some(Value::String(s)) => stop.push(s.clone()),
        Some(Value::Array(a)) => {
            for s in a {
                if let Some(s) = s.as_str() {
                    stop.push(s.to_string());
                }
            }
        }
        _ => {}
    }
    GenParams {
        max_tokens: v.get(max_key).and_then(|x| x.as_u64()).unwrap_or(256) as usize,
        temperature: f("temperature", 0.0),
        top_p: f("top_p", 1.0),
        top_k: v.get("top_k").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
        stop,
    }
}

// ---- Anthropic request mapping (system + messages → chat messages) ----
fn encode_anthropic(m: &Model, v: &Value) -> Result<Vec<u32>> {
    let mut msgs: Vec<Value> = Vec::new();
    if let Some(sys) = v.get("system").and_then(|s| s.as_str()) {
        msgs.push(json!({"role": "system", "content": sys}));
    }
    for msg in v.get("messages").and_then(|x| x.as_array()).into_iter().flatten() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        // Anthropic content is a string or an array of {type:text,text}.
        let content = match msg.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        msgs.push(json!({"role": role, "content": content}));
    }
    encode_chat(m, &msgs, None)
}

// ============================== response bodies ==============================

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn openai_chat(text: &str, finish: &str, model: &str, n_prompt: usize, n_gen: usize) -> Value {
    json!({
        "id": format!("chatcmpl-{}", now_secs()),
        "object": "chat.completion",
        "created": now_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish
        }],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen}
    })
}

fn openai_text(text: &str, finish: &str, model: &str, n_prompt: usize, n_gen: usize) -> Value {
    json!({
        "id": format!("cmpl-{}", now_secs()),
        "object": "text_completion",
        "created": now_secs(),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen}
    })
}

fn anthropic_message(text: &str, finish: &str, model: &str, n_prompt: usize, n_gen: usize) -> Value {
    let reason = if finish == "stop" { "end_turn" } else { "max_tokens" };
    json!({
        "id": format!("msg_{}", now_secs()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": reason,
        "stop_sequence": null,
        "usage": {"input_tokens": n_prompt, "output_tokens": n_gen}
    })
}

fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap()
}

fn respond_json(req: tiny_http::Request, status: u16, body: &Value) {
    let data = serde_json::to_vec(body).unwrap_or_default();
    let resp = Response::from_data(data).with_status_code(status).with_header(json_header());
    let _ = req.respond(resp);
}

// ============================== streaming (SSE) ==============================

/// A `Read` that pulls byte chunks from a channel; EOF when the sender drops.
struct ChanReader {
    rx: Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
}
impl Read for ChanReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.buf.len() {
            match self.rx.recv() {
                Ok(chunk) => {
                    self.buf = chunk;
                    self.pos = 0;
                }
                Err(_) => return Ok(0), // sender dropped → EOF
            }
        }
        let n = (self.buf.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

fn sse(tx: &Sender<Vec<u8>>, payload: &str) -> bool {
    tx.send(format!("data: {payload}\n\n").into_bytes()).is_ok()
}
fn sse_event(tx: &Sender<Vec<u8>>, event: &str, payload: &Value) -> bool {
    tx.send(format!("event: {event}\ndata: {payload}\n\n").into_bytes())
        .is_ok()
}

fn stream_response(
    req: tiny_http::Request,
    route: Route,
    shared: Arc<Mutex<Model>>,
    prompt_ids: Vec<u32>,
    params: GenParams,
    model_id: String,
) {
    let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
    // Generation runs in a worker thread that pushes SSE chunks; this thread
    // responds with the channel reader so tiny_http streams chunked output.
    std::thread::spawn(move || {
        let mut m = shared.lock().unwrap();
        let n_prompt = prompt_ids.len();
        match route {
            Route::Anthropic => stream_anthropic(&mut m, &prompt_ids, &params, &tx, &model_id, n_prompt),
            _ => stream_openai(&mut m, &prompt_ids, &params, &tx, &model_id, route),
        }
    });
    let reader = ChanReader {
        rx,
        buf: Vec::new(),
        pos: 0,
    };
    let resp = Response::new(
        tiny_http::StatusCode(200),
        vec![
            Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap(),
            Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).unwrap(),
        ],
        reader,
        None,
        None,
    );
    let _ = req.respond(resp);
}

fn stream_openai(
    m: &mut Model,
    prompt_ids: &[u32],
    params: &GenParams,
    tx: &Sender<Vec<u8>>,
    model_id: &str,
    route: Route,
) {
    let id = format!("chatcmpl-{}", now_secs());
    let (obj, first) = match route {
        Route::OpenAiText => ("text_completion", json!({})),
        _ => (
            "chat.completion.chunk",
            json!({"role": "assistant", "content": ""}),
        ),
    };
    // OpenAI chat: first chunk carries the role.
    if matches!(route, Route::OpenAiChat) {
        let _ = sse(
            tx,
            &json!({"id": id, "object": obj, "created": now_secs(), "model": model_id,
                    "choices": [{"index":0, "delta": first, "finish_reason": null}]})
            .to_string(),
        );
    }
    let res = generate(m, prompt_ids, params, |piece| {
        let delta = match route {
            Route::OpenAiText => json!({"text": piece}),
            _ => json!({"content": piece}),
        };
        let chunk = if matches!(route, Route::OpenAiText) {
            json!({"id": id, "object": obj, "created": now_secs(), "model": model_id,
                   "choices": [{"index":0, "text": piece, "finish_reason": null}]})
        } else {
            json!({"id": id, "object": obj, "created": now_secs(), "model": model_id,
                   "choices": [{"index":0, "delta": delta, "finish_reason": null}]})
        };
        let _ = sse(tx, &chunk.to_string());
    });
    let finish = res.map(|(_, f, _)| f).unwrap_or("stop");
    let last = json!({"id": id, "object": obj, "created": now_secs(), "model": model_id,
        "choices": [{"index":0, "delta": {}, "finish_reason": finish}]});
    let _ = sse(tx, &last.to_string());
    let _ = sse(tx, "[DONE]");
}

fn stream_anthropic(
    m: &mut Model,
    prompt_ids: &[u32],
    params: &GenParams,
    tx: &Sender<Vec<u8>>,
    model_id: &str,
    n_prompt: usize,
) {
    let id = format!("msg_{}", now_secs());
    let _ = sse_event(
        tx,
        "message_start",
        &json!({"type":"message_start","message":{
            "id": id, "type":"message", "role":"assistant", "model": model_id,
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": n_prompt, "output_tokens": 0}}}),
    );
    let _ = sse_event(
        tx,
        "content_block_start",
        &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
    );
    let res = generate(m, prompt_ids, params, |piece| {
        let _ = sse_event(
            tx,
            "content_block_delta",
            &json!({"type":"content_block_delta","index":0,
                    "delta":{"type":"text_delta","text": piece}}),
        );
    });
    let (finish, n_gen) = res.map(|(_, f, n)| (f, n)).unwrap_or(("stop", 0));
    let reason = if finish == "stop" { "end_turn" } else { "max_tokens" };
    let _ = sse_event(tx, "content_block_stop", &json!({"type":"content_block_stop","index":0}));
    let _ = sse_event(
        tx,
        "message_delta",
        &json!({"type":"message_delta","delta":{"stop_reason": reason, "stop_sequence": null},
                "usage": {"output_tokens": n_gen}}),
    );
    let _ = sse_event(tx, "message_stop", &json!({"type":"message_stop"}));
}
