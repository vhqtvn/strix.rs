//! Chat templating — render chat `messages` through the model's OWN Jinja chat
//! template, exactly as shipped in the GGUF `tokenizer.chat_template` metadata.
//!
//! This mirrors what mature runners do (llama.cpp via `minja`, mistral.rs via
//! `minijinja`): the model ships the correct, often subtle, template (system-message
//! handling, BOS, generation-prompt marker, `<think>` tags, tool-call formatting),
//! so rendering it is correct-by-construction. Re-implementing each model's template
//! by hand is the usual source of silent "garbage output" bugs.

use crate::gguf::GgufFile;
use minijinja::{context, Environment, Error as JErr, ErrorKind, Value};
use strix_core::error::{Result, StrixError};

/// A model's chat template + the special-token strings its Jinja references.
pub struct ChatTemplate {
    src: String,
    bos: String,
    eos: String,
}

impl ChatTemplate {
    /// Pull the Jinja chat template (and bos/eos token strings) from GGUF metadata.
    /// Returns `None` if the model has no embedded template (e.g. a base model).
    pub fn from_gguf(g: &GgufFile) -> Option<ChatTemplate> {
        let md = g.metadata();
        let src = md.get("tokenizer.chat_template")?.as_str()?.to_string();
        // Resolve bos/eos token *strings* from the id + token table (templates that
        // reference `bos_token`/`eos_token` want the literal text, e.g. "<|im_end|>").
        let tokens = md.get("tokenizer.ggml.tokens").and_then(|v| v.as_array());
        let tok_str = |key: &str| -> String {
            let id = md.get(key).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            tokens
                .and_then(|t| t.get(id))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let bos = tok_str("tokenizer.ggml.bos_token_id");
        let eos = tok_str("tokenizer.ggml.eos_token_id");
        Some(ChatTemplate { src, bos, eos })
    }

    /// Build directly from a template string (e.g. tokenizer_config.json fallback).
    pub fn from_str(src: impl Into<String>, bos: impl Into<String>, eos: impl Into<String>) -> Self {
        ChatTemplate {
            src: src.into(),
            bos: bos.into(),
            eos: eos.into(),
        }
    }

    pub fn raw(&self) -> &str {
        &self.src
    }

    /// Render `messages` (each a JSON object with at least `role`/`content`) to the
    /// model's prompt string. `add_generation_prompt` appends the assistant-turn
    /// opener so the model continues as the assistant. `tools` is passed through for
    /// templates that support tool calling.
    pub fn render(
        &self,
        messages: &[serde_json::Value],
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        let mut env = Environment::new();
        // HF templates use Python string methods (.strip/.split/.startswith/...).
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // `raise_exception(msg)` — templates call this to reject bad message orders.
        env.add_function("raise_exception", |msg: String| -> std::result::Result<Value, JErr> {
            Err(JErr::new(ErrorKind::InvalidOperation, msg))
        });
        // `strftime_now(fmt)` — Llama-3.x date stamping. We don't pull in a clock
        // dep; return a stable placeholder (the date rarely affects generation).
        env.add_function("strftime_now", |_fmt: String| -> String { String::new() });
        env.add_template("chat", &self.src)
            .map_err(|e| StrixError::parse(format!("chat template parse: {e}")))?;
        let tmpl = env
            .get_template("chat")
            .map_err(|e| StrixError::parse(format!("chat template: {e}")))?;
        let msgs = Value::from_serialize(messages);
        let tools_v = tools.map(Value::from_serialize).unwrap_or(Value::UNDEFINED);
        let out = tmpl
            .render(context! {
                messages => msgs,
                add_generation_prompt => add_generation_prompt,
                bos_token => self.bos,
                eos_token => self.eos,
                tools => tools_v,
            })
            .map_err(|e| StrixError::parse(format!("chat template render: {e}")))?;
        Ok(out)
    }
}
