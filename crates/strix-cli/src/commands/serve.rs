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

fn load_model(
    path: &Path,
    gpu: bool,
    ctx: usize,
    template_override: Option<&Path>,
) -> Result<Model> {
    let gguf_path =
        find_gguf(path).ok_or_else(|| anyhow!("no .gguf found at {}", path.display()))?;
    let gguf = GgufFile::open(&gguf_path).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("?").to_string();
    let template = match template_override {
        Some(p) => {
            let src = std::fs::read_to_string(p)
                .with_context(|| format!("read chat template {}", p.display()))?;
            eprintln!("[serve] using chat-template override {}", p.display());
            Some(ChatTemplate::from_gguf_src(&gguf, src))
        }
        None => ChatTemplate::from_gguf(&gguf),
    };
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
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - max) * inv_t).exp())
        .collect();
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

// ============================== constrained JSON decoding ==============================
// A character-level JSON *prefix* validator: lets us mask the decode so the output is
// always a valid JSON prefix (response_format json). Stateless full-reparse per check —
// JSON outputs are short, so this is cheap.

#[derive(PartialEq, Eq, Clone, Copy)]
enum JsonState {
    Invalid,
    Incomplete,
    Complete,
}

fn json_state(s: &str) -> JsonState {
    let b = s.as_bytes();
    let mut i = 0;
    skip_ws(b, &mut i);
    if i >= b.len() {
        return JsonState::Incomplete; // empty / whitespace is a valid prefix
    }
    match parse_value(b, &mut i) {
        None => JsonState::Invalid,
        Some(false) => JsonState::Incomplete,
        Some(true) => {
            skip_ws(b, &mut i);
            if i >= b.len() {
                JsonState::Complete
            } else {
                JsonState::Invalid
            }
        }
    }
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') {
        *i += 1;
    }
}

/// None = invalid, Some(false) = valid but incomplete (ran out), Some(true) = complete.
fn parse_value(b: &[u8], i: &mut usize) -> Option<bool> {
    skip_ws(b, i);
    if *i >= b.len() {
        return Some(false);
    }
    match b[*i] {
        b'{' => parse_obj(b, i),
        b'[' => parse_arr(b, i),
        b'"' => parse_str(b, i),
        b't' => parse_lit(b, i, b"true"),
        b'f' => parse_lit(b, i, b"false"),
        b'n' => parse_lit(b, i, b"null"),
        b'-' | b'0'..=b'9' => parse_num(b, i),
        _ => None,
    }
}

fn parse_lit(b: &[u8], i: &mut usize, word: &[u8]) -> Option<bool> {
    let mut k = 0;
    while *i < b.len() && k < word.len() {
        if b[*i] != word[k] {
            return None;
        }
        *i += 1;
        k += 1;
    }
    Some(k == word.len())
}

fn parse_str(b: &[u8], i: &mut usize) -> Option<bool> {
    *i += 1; // opening quote
    while *i < b.len() {
        match b[*i] {
            b'"' => {
                *i += 1;
                return Some(true);
            }
            b'\\' => {
                *i += 1;
                if *i >= b.len() {
                    return Some(false);
                }
                *i += 1;
            }
            _ => *i += 1,
        }
    }
    Some(false)
}

fn parse_num(b: &[u8], i: &mut usize) -> Option<bool> {
    let start = *i;
    if *i < b.len() && b[*i] == b'-' {
        *i += 1;
    }
    while *i < b.len() && b[*i].is_ascii_digit() {
        *i += 1;
    }
    if *i < b.len() && b[*i] == b'.' {
        *i += 1;
        while *i < b.len() && b[*i].is_ascii_digit() {
            *i += 1;
        }
    }
    if *i < b.len() && (b[*i] == b'e' || b[*i] == b'E') {
        *i += 1;
        if *i < b.len() && (b[*i] == b'+' || b[*i] == b'-') {
            *i += 1;
        }
        while *i < b.len() && b[*i].is_ascii_digit() {
            *i += 1;
        }
    }
    if *i == start {
        return None;
    }
    // At end → could still be extended (incomplete); otherwise a delimiter bounds it.
    Some(*i < b.len())
}

fn parse_obj(b: &[u8], i: &mut usize) -> Option<bool> {
    *i += 1;
    loop {
        skip_ws(b, i);
        if *i >= b.len() {
            return Some(false);
        }
        if b[*i] == b'}' {
            *i += 1;
            return Some(true);
        }
        if b[*i] != b'"' {
            return None;
        }
        if !parse_str(b, i)? {
            return Some(false);
        }
        skip_ws(b, i);
        if *i >= b.len() {
            return Some(false);
        }
        if b[*i] != b':' {
            return None;
        }
        *i += 1;
        if !parse_value(b, i)? {
            return Some(false);
        }
        skip_ws(b, i);
        if *i >= b.len() {
            return Some(false);
        }
        match b[*i] {
            b',' => *i += 1,
            b'}' => {
                *i += 1;
                return Some(true);
            }
            _ => return None,
        }
    }
}

fn parse_arr(b: &[u8], i: &mut usize) -> Option<bool> {
    *i += 1;
    loop {
        skip_ws(b, i);
        if *i >= b.len() {
            return Some(false);
        }
        if b[*i] == b']' {
            *i += 1;
            return Some(true);
        }
        if !parse_value(b, i)? {
            return Some(false);
        }
        skip_ws(b, i);
        if *i >= b.len() {
            return Some(false);
        }
        match b[*i] {
            b',' => *i += 1,
            b']' => {
                *i += 1;
                return Some(true);
            }
            _ => return None,
        }
    }
}

// ===== schema-aware JSON constraint (response_format json_schema) =====
// Same prefix-validator idea as json_state, but driven by a JSON Schema: enforces
// object keys (declaration order), value types, enums, arrays(items). Over-constrains
// objects to emit ALL `properties` in order (good for structured extraction); ranges /
// patterns / optional-key branching are not enforced (MVP).

fn schema_prefix(s: &str, schema: &Value) -> JsonState {
    let b = s.as_bytes();
    let mut i = 0;
    skip_ws(b, &mut i);
    if i >= b.len() {
        return JsonState::Incomplete;
    }
    match consume_schema(b, &mut i, schema) {
        None => JsonState::Invalid,
        Some(false) => JsonState::Incomplete,
        Some(true) => {
            skip_ws(b, &mut i);
            if i >= b.len() {
                JsonState::Complete
            } else {
                JsonState::Invalid
            }
        }
    }
}

fn consume_schema(b: &[u8], i: &mut usize, schema: &Value) -> Option<bool> {
    skip_ws(b, i);
    if *i >= b.len() {
        return Some(false);
    }
    if let Some(en) = schema.get("enum").and_then(|e| e.as_array()) {
        return consume_enum(b, i, en);
    }
    match schema.get("type").and_then(|t| t.as_str()) {
        Some("object") => consume_obj_schema(b, i, schema),
        Some("array") => consume_arr_schema(b, i, schema),
        Some("string") => parse_str(b, i),
        Some("integer") => parse_int(b, i),
        Some("number") => parse_num(b, i),
        Some("boolean") => match b[*i] {
            b't' => parse_lit(b, i, b"true"),
            b'f' => parse_lit(b, i, b"false"),
            _ => None,
        },
        Some("null") => parse_lit(b, i, b"null"),
        _ => parse_value(b, i), // unknown/missing type → any JSON
    }
}

fn parse_int(b: &[u8], i: &mut usize) -> Option<bool> {
    if *i < b.len() && b[*i] == b'-' {
        *i += 1;
    }
    let ds = *i;
    while *i < b.len() && b[*i].is_ascii_digit() {
        *i += 1;
    }
    if *i == ds {
        return if *i >= b.len() { Some(false) } else { None }; // "-"@end ok, "-x" invalid
    }
    Some(*i < b.len())
}

fn consume_enum(b: &[u8], i: &mut usize, vals: &[Value]) -> Option<bool> {
    let rem = &b[*i..];
    let mut any_prefix = false;
    for v in vals {
        let s = v.to_string();
        let sb = s.as_bytes();
        if rem.len() >= sb.len() {
            if &rem[..sb.len()] == sb {
                *i += sb.len();
                return Some(true);
            }
        } else if sb.starts_with(rem) {
            any_prefix = true;
        }
    }
    if any_prefix {
        *i = b.len();
        Some(false)
    } else {
        None
    }
}

fn consume_obj_schema(b: &[u8], i: &mut usize, schema: &Value) -> Option<bool> {
    if b[*i] != b'{' {
        return None;
    }
    *i += 1;
    let props: Vec<(&String, &Value)> = schema
        .get("properties")
        .and_then(|p| p.as_object())
        .map(|o| o.iter().collect())
        .unwrap_or_default();
    for (idx, (name, sub)) in props.iter().enumerate() {
        skip_ws(b, i);
        if *i >= b.len() {
            return Some(false);
        }
        if idx > 0 {
            if b[*i] != b',' {
                return None;
            }
            *i += 1;
            skip_ws(b, i);
            if *i >= b.len() {
                return Some(false);
            }
        }
        let key = format!("\"{name}\"");
        match parse_lit(b, i, key.as_bytes())? {
            false => return Some(false),
            true => {}
        }
        skip_ws(b, i);
        if *i >= b.len() {
            return Some(false);
        }
        if b[*i] != b':' {
            return None;
        }
        *i += 1;
        match consume_schema(b, i, sub)? {
            false => return Some(false),
            true => {}
        }
    }
    skip_ws(b, i);
    if *i >= b.len() {
        return Some(false);
    }
    if b[*i] != b'}' {
        return None;
    }
    *i += 1;
    Some(true)
}

fn consume_arr_schema(b: &[u8], i: &mut usize, schema: &Value) -> Option<bool> {
    if b[*i] != b'[' {
        return None;
    }
    *i += 1;
    let items = schema.get("items");
    loop {
        skip_ws(b, i);
        if *i >= b.len() {
            return Some(false);
        }
        if b[*i] == b']' {
            *i += 1;
            return Some(true);
        }
        let r = match items {
            Some(it) => consume_schema(b, i, it)?,
            None => parse_value(b, i)?,
        };
        if !r {
            return Some(false);
        }
        skip_ws(b, i);
        if *i >= b.len() {
            return Some(false);
        }
        match b[*i] {
            b',' => *i += 1,
            b']' => {
                *i += 1;
                return Some(true);
            }
            _ => return None,
        }
    }
}

/// What the decode is constrained to.
enum Constrain<'a> {
    None,
    Json,
    Schema(&'a Value),
}

impl Constrain<'_> {
    fn state(&self, s: &str) -> JsonState {
        match self {
            Constrain::Json => json_state(s),
            Constrain::Schema(sc) => schema_prefix(s, sc),
            Constrain::None => JsonState::Incomplete,
        }
    }
}

/// Top-`k` token ids by logit, descending.
fn top_k_desc(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &c| {
        logits[c as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

/// Pick the highest-logit token that keeps the decoded output a valid (JSON / schema)
/// prefix per `con`.
fn pick_constrained(tok: &StrixTokenizer, logits: &[f32], out_ids: &[u32], con: &Constrain) -> u32 {
    let committed = tok.decode(out_ids, true).unwrap_or_default();
    let cands = top_k_desc(logits, 48);
    for &c in &cands {
        let mut ids = out_ids.to_vec();
        ids.push(c);
        let full = match tok.decode(&ids, true) {
            Ok(f) => f,
            Err(_) => continue,
        };
        if full.len() < committed.len() {
            continue;
        }
        if con.state(&full) != JsonState::Invalid {
            return c;
        }
    }
    cands.first().copied().unwrap_or(0)
}

/// Drive prefill + decode. Calls `on_token(piece)` for each new decoded text piece
/// (incremental detokenization). When `json`, the decode is constrained so the output
/// is always a valid JSON value (response_format). Returns (text, finish_reason, n_gen).
fn generate(
    m: &mut Model,
    prompt_ids: &[u32],
    p: &GenParams,
    con: &Constrain,
    mut on_token: impl FnMut(&str),
) -> Result<(String, &'static str, usize)> {
    m.decoder.reset();
    let mut rng = Rng::new();
    let mut logits = m.decoder.prefill(prompt_ids).context("prefill")?;

    let constrained = !matches!(con, Constrain::None);
    let mut out_ids: Vec<u32> = Vec::new();
    let mut emitted = String::new(); // text already sent to on_token
    let mut finish = "length";
    for step in 0..p.max_tokens {
        let next = if constrained {
            pick_constrained(&m.tok, &logits.0, &out_ids, con)
        } else {
            sample_token(&logits.0, p, &mut rng)
        };
        if m.eos.contains(&next) {
            finish = "stop";
            break;
        }
        out_ids.push(next);
        // Incremental detokenize: decode the whole sequence, emit the new suffix.
        let full = m.tok.decode(&out_ids, true).unwrap_or_default();
        if full.len() > emitted.len() && full.is_char_boundary(emitted.len()) {
            let piece = full[emitted.len()..].to_string();
            if !piece.is_empty() {
                on_token(&piece);
                emitted = full.clone();
            }
        }
        if !p.stop.is_empty() && p.stop.iter().any(|s| full.contains(s.as_str())) {
            finish = "stop";
            break;
        }
        if constrained && con.state(&full) == JsonState::Complete {
            finish = "stop";
            break;
        }
        if step + 1 >= p.max_tokens {
            break;
        }
        logits = m.decoder.decode_one(next).context("decode")?;
    }
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

pub fn run(
    path: &Path,
    host: &str,
    port: u16,
    gpu: bool,
    ctx: usize,
    chat_template: Option<&Path>,
) -> Result<()> {
    let model = load_model(path, gpu, ctx, chat_template)?;
    eprintln!(
        "[serve] loaded `{}` (arch {}), chat_template: {}, eos ids: {:?}",
        model.name,
        model.arch,
        if model.template.is_some() {
            "yes"
        } else {
            "NONE"
        },
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
                Some(msgs) => {
                    let msgs = apply_response_format(msgs, &v);
                    encode_chat(&m, &msgs, v.get("tools"))
                }
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

    let want_tools = v
        .get("tools")
        .and_then(|t| t.as_array())
        .is_some_and(|a| !a.is_empty());
    let want_json = v
        .get("response_format")
        .and_then(|r| r.get("type"))
        .and_then(|t| t.as_str())
        .is_some_and(|t| t == "json_object" || t == "json_schema");
    // json_schema → the schema to constrain to (None for plain json_object).
    let schema = v
        .get("response_format")
        .filter(|r| r.get("type").and_then(|t| t.as_str()) == Some("json_schema"))
        .and_then(|r| r.get("json_schema"))
        .and_then(|j| j.get("schema"))
        .cloned();
    if stream {
        stream_response(
            req,
            route,
            shared.clone(),
            prompt_ids,
            params,
            model_id.to_string(),
            want_tools,
            want_json,
            schema,
        );
    } else {
        blocking_response(
            req, route, shared, prompt_ids, params, model_id, want_tools, want_json, schema,
        );
    }
}

/// Build the decode constraint: tools disable it; else schema → schema-constrained;
/// else json → valid-JSON-constrained; else unconstrained sampling.
fn constrain_of<'a>(want_json: bool, want_tools: bool, schema: &'a Option<Value>) -> Constrain<'a> {
    if want_tools {
        Constrain::None
    } else if let Some(s) = schema {
        Constrain::Schema(s)
    } else if want_json {
        Constrain::Json
    } else {
        Constrain::None
    }
}

fn blocking_response(
    req: tiny_http::Request,
    route: Route,
    shared: &Arc<Mutex<Model>>,
    prompt_ids: Vec<u32>,
    params: GenParams,
    model_id: &str,
    want_tools: bool,
    want_json: bool,
    schema: Option<Value>,
) {
    let mut m = shared.lock().unwrap();
    let n_prompt = prompt_ids.len();
    let con = constrain_of(want_json, want_tools, &schema);
    let res = generate(&mut m, &prompt_ids, &params, &con, |_| {});
    drop(m);
    match res {
        Ok((raw, finish, n_gen)) => {
            // Pull any `<tool_call>{json}</tool_call>` blocks out of the output.
            let (mut text, calls) = parse_tool_calls(&raw, want_tools);
            // response_format json: return just the first balanced JSON object.
            if want_json && calls.is_empty() {
                if let Some((s, e)) = first_json_object(&text) {
                    text = text[s..e].to_string();
                }
            }
            let body = match route {
                Route::Anthropic => {
                    anthropic_message(&text, finish, model_id, n_prompt, n_gen, &calls)
                }
                Route::OpenAiText => openai_text(&raw, finish, model_id, n_prompt, n_gen),
                Route::OpenAiChat => openai_chat(&text, finish, model_id, n_prompt, n_gen, &calls),
            };
            respond_json(req, 200, &body);
        }
        Err(e) => respond_json(req, 500, &json!({"error": e.to_string()})),
    }
}

/// A tool call parsed from the model's text output (OpenAI-shaped: arguments are a
/// JSON *string*).
struct ToolCall {
    name: String,
    arguments: String,
}

/// Extract `<tool_call>…</tool_call>` blocks (the Qwen/Hermes/Nous de-facto
/// standard) from the model output. Returns (text_without_calls, calls). Tolerates
/// an unterminated final block (truncation) and ```json fences inside the block.
fn parse_tool_calls(text: &str, want_tools: bool) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut clean = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        clean.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        let (body, tail) = match after.find("</tool_call>") {
            Some(end) => (&after[..end], &after[end + "</tool_call>".len()..]),
            None => (after, ""),
        };
        if let Some(tc) = parse_one_call(body) {
            calls.push(tc);
        }
        rest = tail;
    }
    clean.push_str(rest);
    if !calls.is_empty() || !want_tools {
        return (clean.trim().to_string(), calls);
    }
    // Fallback: some models emit the tool-call JSON *bare* (no <tool_call> wrapper).
    // When tools were requested, scan for a top-level {"name":..,"arguments":..} object.
    if let Some((s, e)) = first_json_object(text) {
        if let Some(tc) = parse_strict_call(&text[s..e]) {
            let around = format!("{}{}", text[..s].trim_end(), text[e..].trim_start());
            return (around.trim().to_string(), vec![tc]);
        }
    }
    (clean.trim().to_string(), calls)
}

/// Byte range of the first balanced `{...}` object (string-aware brace matching).
fn first_json_object(s: &str) -> Option<(usize, usize)> {
    let b = s.as_bytes();
    let start = s.find('{')?;
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for i in start..b.len() {
        let c = b[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((start, i + 1));
                }
            }
            _ => {}
        }
    }
    None
}

/// Like `parse_one_call` but requires both `name` and `arguments`/`parameters` —
/// used for the bare-JSON fallback to avoid treating ordinary JSON as a tool call.
fn parse_strict_call(body: &str) -> Option<ToolCall> {
    let v: Value = serde_json::from_str(body.trim()).ok()?;
    if v.get("name").is_none() || (v.get("arguments").is_none() && v.get("parameters").is_none()) {
        return None;
    }
    parse_one_call(body)
}

fn parse_one_call(body: &str) -> Option<ToolCall> {
    let b = body
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let v: Value = serde_json::from_str(b).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = match v.get("arguments").or_else(|| v.get("parameters")) {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    Some(ToolCall { name, arguments })
}

fn parse_params(v: &Value, route: Route) -> GenParams {
    let f = |k: &str, d: f32| {
        v.get(k)
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(d)
    };
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

// ---- Anthropic request mapping (system + messages + tools → chat messages) ----
fn encode_anthropic(m: &Model, v: &Value) -> Result<Vec<u32>> {
    let mut msgs: Vec<Value> = Vec::new();
    if let Some(sys) = v.get("system").and_then(|s| s.as_str()) {
        msgs.push(json!({"role": "system", "content": sys}));
    }
    for msg in v
        .get("messages")
        .and_then(|x| x.as_array())
        .into_iter()
        .flatten()
    {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        match msg.get("content") {
            Some(Value::String(s)) => msgs.push(json!({"role": role, "content": s})),
            Some(Value::Array(parts)) => {
                // Anthropic content blocks: text / tool_use (assistant) / tool_result (user).
                let mut text = String::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                for p in parts {
                    match p.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                                text.push_str(t);
                            }
                        }
                        Some("tool_use") => tool_calls.push(json!({
                            "id": p.get("id").cloned().unwrap_or(Value::Null),
                            "type": "function",
                            "function": {
                                "name": p.get("name").cloned().unwrap_or(Value::Null),
                                "arguments": p.get("input").map(|i| i.to_string()).unwrap_or_else(|| "{}".into())
                            }
                        })),
                        Some("tool_result") => {
                            let c = match p.get("content") {
                                Some(Value::String(s)) => s.clone(),
                                Some(Value::Array(a)) => a
                                    .iter()
                                    .filter_map(|x| x.get("text").and_then(|t| t.as_str()))
                                    .collect::<Vec<_>>()
                                    .join(""),
                                Some(o) => o.to_string(),
                                None => String::new(),
                            };
                            msgs.push(json!({"role": "tool", "content": c}));
                        }
                        _ => {}
                    }
                }
                if !tool_calls.is_empty() {
                    msgs.push(json!({"role": role, "content": text, "tool_calls": tool_calls}));
                } else if !text.is_empty() {
                    msgs.push(json!({"role": role, "content": text}));
                }
            }
            _ => {}
        }
    }
    let tools = v.get("tools").map(anthropic_tools_to_openai);
    encode_chat(m, &msgs, tools.as_ref())
}

/// Anthropic tools `[{name,description,input_schema}]` → OpenAI
/// `[{type:function,function:{name,description,parameters}}]` (the shape chat
/// templates expect).
fn anthropic_tools_to_openai(tools: &Value) -> Value {
    let arr: Vec<Value> = tools
        .as_array()
        .map(|a| {
            a.iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.get("name").cloned().unwrap_or(Value::Null),
                            "description": t.get("description").cloned().unwrap_or(Value::Null),
                            "parameters": t.get("input_schema").cloned().unwrap_or_else(|| json!({"type":"object"}))
                        }
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    json!(arr)
}

/// `response_format` (OpenAI structured output): append a JSON instruction to the
/// last user message. MVP — prompt-guided, not grammar-constrained decoding.
fn apply_response_format(messages: &[Value], v: &Value) -> Vec<Value> {
    let rf = v.get("response_format");
    let instr = match rf.and_then(|r| r.get("type")).and_then(|t| t.as_str()) {
        Some("json_object") => Some(
            "Respond ONLY with a single valid JSON object — no prose, no code fences.".to_string(),
        ),
        Some("json_schema") => {
            let schema = rf
                .and_then(|r| r.get("json_schema"))
                .and_then(|j| j.get("schema"))
                .map(|s| s.to_string())
                .unwrap_or_default();
            Some(format!(
                "Respond ONLY with a single valid JSON object matching this JSON Schema — no prose, no code fences:\n{schema}"
            ))
        }
        _ => None,
    };
    let mut msgs = messages.to_vec();
    if let Some(instr) = instr {
        if let Some(last) = msgs
            .iter_mut()
            .rev()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        {
            if let Some(c) = last.get("content").and_then(|c| c.as_str()) {
                last["content"] = json!(format!("{c}\n\n{instr}"));
            }
        }
    }
    msgs
}

// ============================== response bodies ==============================

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn openai_chat(
    text: &str,
    finish: &str,
    model: &str,
    n_prompt: usize,
    n_gen: usize,
    calls: &[ToolCall],
) -> Value {
    let mut message = json!({"role": "assistant", "content": text});
    let mut finish_reason = finish;
    if !calls.is_empty() {
        let arr: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({
                    "id": format!("call_{}_{i}", now_secs()),
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments}
                })
            })
            .collect();
        message["content"] = if text.is_empty() {
            Value::Null
        } else {
            json!(text)
        };
        message["tool_calls"] = json!(arr);
        finish_reason = "tool_calls";
    }
    json!({
        "id": format!("chatcmpl-{}", now_secs()),
        "object": "chat.completion",
        "created": now_secs(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
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

fn anthropic_message(
    text: &str,
    finish: &str,
    model: &str,
    n_prompt: usize,
    n_gen: usize,
    calls: &[ToolCall],
) -> Value {
    let mut content: Vec<Value> = Vec::new();
    if !text.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }
    for (i, c) in calls.iter().enumerate() {
        let input: Value = serde_json::from_str(&c.arguments).unwrap_or_else(|_| json!({}));
        content.push(json!({
            "type": "tool_use",
            "id": format!("toolu_{}_{i}", now_secs()),
            "name": c.name,
            "input": input
        }));
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }
    let reason = if !calls.is_empty() {
        "tool_use"
    } else if finish == "stop" {
        "end_turn"
    } else {
        "max_tokens"
    };
    json!({
        "id": format!("msg_{}", now_secs()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
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
    let resp = Response::from_data(data)
        .with_status_code(status)
        .with_header(json_header());
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

#[allow(clippy::too_many_arguments)]
fn stream_response(
    req: tiny_http::Request,
    route: Route,
    shared: Arc<Mutex<Model>>,
    prompt_ids: Vec<u32>,
    params: GenParams,
    model_id: String,
    want_tools: bool,
    want_json: bool,
    schema: Option<Value>,
) {
    let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
    // Generation runs in a worker thread that pushes SSE chunks; this thread
    // responds with the channel reader so tiny_http streams chunked output.
    std::thread::spawn(move || {
        let mut m = shared.lock().unwrap();
        let n_prompt = prompt_ids.len();
        match route {
            Route::Anthropic => stream_anthropic(
                &mut m,
                &prompt_ids,
                &params,
                &tx,
                &model_id,
                n_prompt,
                want_tools,
                want_json,
                &schema,
            ),
            _ => stream_openai(
                &mut m,
                &prompt_ids,
                &params,
                &tx,
                &model_id,
                route,
                want_tools,
                want_json,
                &schema,
            ),
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

#[allow(clippy::too_many_arguments)]
fn stream_openai(
    m: &mut Model,
    prompt_ids: &[u32],
    params: &GenParams,
    tx: &Sender<Vec<u8>>,
    model_id: &str,
    route: Route,
    want_tools: bool,
    want_json: bool,
    schema: &Option<Value>,
) {
    let id = format!("chatcmpl-{}", now_secs());
    let obj = match route {
        Route::OpenAiText => "text_completion",
        _ => "chat.completion.chunk",
    };
    // OpenAI chat: first chunk carries the role.
    if matches!(route, Route::OpenAiChat) {
        let _ = sse(
            tx,
            &json!({"id": id, "object": obj, "created": now_secs(), "model": model_id,
                    "choices": [{"index":0, "delta": {"role":"assistant","content":""}, "finish_reason": null}]})
            .to_string(),
        );
    }
    // Tool calling: buffer, then emit the parsed tool calls as delta.tool_calls
    // (incremental token-level tool parsing is overkill; one delta is spec-valid).
    if want_tools && matches!(route, Route::OpenAiChat) {
        let (raw, _f, _) = generate(m, prompt_ids, params, &Constrain::None, |_| {}).unwrap_or((
            String::new(),
            "stop",
            0,
        ));
        let (text, calls) = parse_tool_calls(&raw, true);
        if calls.is_empty() {
            if !text.is_empty() {
                let _ = sse(
                    tx,
                    &json!({"id": id, "object": obj, "created": now_secs(), "model": model_id,
                    "choices": [{"index":0, "delta": {"content": text}, "finish_reason": null}]})
                    .to_string(),
                );
            }
            emit_finish(tx, &id, obj, model_id, "stop");
        } else {
            let arr: Vec<Value> = calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    json!({
                        "index": i, "id": format!("call_{}_{i}", now_secs()), "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments}
                    })
                })
                .collect();
            let _ = sse(
                tx,
                &json!({"id": id, "object": obj, "created": now_secs(), "model": model_id,
                "choices": [{"index":0, "delta": {"tool_calls": arr}, "finish_reason": null}]})
                .to_string(),
            );
            emit_finish(tx, &id, obj, model_id, "tool_calls");
        }
        let _ = sse(tx, "[DONE]");
        return;
    }
    let is_text = matches!(route, Route::OpenAiText);
    let con = constrain_of(want_json, want_tools, schema);
    let res = generate(m, prompt_ids, params, &con, |piece| {
        let chunk = if is_text {
            json!({"id": id, "object": obj, "created": now_secs(), "model": model_id,
                   "choices": [{"index":0, "text": piece, "finish_reason": null}]})
        } else {
            json!({"id": id, "object": obj, "created": now_secs(), "model": model_id,
                   "choices": [{"index":0, "delta": {"content": piece}, "finish_reason": null}]})
        };
        let _ = sse(tx, &chunk.to_string());
    });
    let finish = res.map(|(_, f, _)| f).unwrap_or("stop");
    emit_finish(tx, &id, obj, model_id, finish);
    let _ = sse(tx, "[DONE]");
}

fn emit_finish(tx: &Sender<Vec<u8>>, id: &str, obj: &str, model_id: &str, finish: &str) {
    let last = json!({"id": id, "object": obj, "created": now_secs(), "model": model_id,
        "choices": [{"index":0, "delta": {}, "finish_reason": finish}]});
    let _ = sse(tx, &last.to_string());
}

#[allow(clippy::too_many_arguments)]
fn stream_anthropic(
    m: &mut Model,
    prompt_ids: &[u32],
    params: &GenParams,
    tx: &Sender<Vec<u8>>,
    model_id: &str,
    n_prompt: usize,
    want_tools: bool,
    want_json: bool,
    schema: &Option<Value>,
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
    // Tool calling: buffer, then emit text + tool_use blocks.
    if want_tools {
        let (raw, _f, n_gen) = generate(m, prompt_ids, params, &Constrain::None, |_| {})
            .unwrap_or((String::new(), "stop", 0));
        let (text, calls) = parse_tool_calls(&raw, true);
        let mut idx = 0;
        if !text.is_empty() {
            let _ = sse_event(
                tx,
                "content_block_start",
                &json!({"type":"content_block_start","index":idx,"content_block":{"type":"text","text":""}}),
            );
            let _ = sse_event(
                tx,
                "content_block_delta",
                &json!({"type":"content_block_delta","index":idx,"delta":{"type":"text_delta","text":text}}),
            );
            let _ = sse_event(
                tx,
                "content_block_stop",
                &json!({"type":"content_block_stop","index":idx}),
            );
            idx += 1;
        }
        for c in &calls {
            let input: Value = serde_json::from_str(&c.arguments).unwrap_or_else(|_| json!({}));
            let _ = sse_event(
                tx,
                "content_block_start",
                &json!({"type":"content_block_start","index":idx,
                "content_block":{"type":"tool_use","id":format!("toolu_{}_{idx}",now_secs()),"name":c.name,"input":{}}}),
            );
            let _ = sse_event(
                tx,
                "content_block_delta",
                &json!({"type":"content_block_delta","index":idx,
                "delta":{"type":"input_json_delta","partial_json": input.to_string()}}),
            );
            let _ = sse_event(
                tx,
                "content_block_stop",
                &json!({"type":"content_block_stop","index":idx}),
            );
            idx += 1;
        }
        let reason = if calls.is_empty() {
            "end_turn"
        } else {
            "tool_use"
        };
        let _ = sse_event(
            tx,
            "message_delta",
            &json!({"type":"message_delta","delta":{"stop_reason":reason,"stop_sequence":null},"usage":{"output_tokens":n_gen}}),
        );
        let _ = sse_event(tx, "message_stop", &json!({"type":"message_stop"}));
        return;
    }
    let _ = sse_event(
        tx,
        "content_block_start",
        &json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
    );
    let con = constrain_of(want_json, want_tools, schema);
    let res = generate(m, prompt_ids, params, &con, |piece| {
        let _ = sse_event(
            tx,
            "content_block_delta",
            &json!({"type":"content_block_delta","index":0,
                    "delta":{"type":"text_delta","text": piece}}),
        );
    });
    let (finish, n_gen) = res.map(|(_, f, n)| (f, n)).unwrap_or(("stop", 0));
    let reason = if finish == "stop" {
        "end_turn"
    } else {
        "max_tokens"
    };
    let _ = sse_event(
        tx,
        "content_block_stop",
        &json!({"type":"content_block_stop","index":0}),
    );
    let _ = sse_event(
        tx,
        "message_delta",
        &json!({"type":"message_delta","delta":{"stop_reason": reason, "stop_sequence": null},
                "usage": {"output_tokens": n_gen}}),
    );
    let _ = sse_event(tx, "message_stop", &json!({"type":"message_stop"}));
}
