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
    /// Resolve bos/eos token *strings* from the GGUF id + token table (templates that
    /// reference `bos_token`/`eos_token` want the literal text, e.g. "<|im_end|>").
    fn bos_eos(g: &GgufFile) -> (String, String) {
        let md = g.metadata();
        let tokens = md.get("tokenizer.ggml.tokens").and_then(|v| v.as_array());
        let tok_str = |key: &str| -> String {
            let id = md.get(key).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            tokens
                .and_then(|t| t.get(id))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        (
            tok_str("tokenizer.ggml.bos_token_id"),
            tok_str("tokenizer.ggml.eos_token_id"),
        )
    }

    /// Pull the Jinja chat template (and bos/eos token strings) from GGUF metadata.
    /// Returns `None` if the model has no embedded template (e.g. a base model).
    pub fn from_gguf(g: &GgufFile) -> Option<ChatTemplate> {
        let src = g
            .metadata()
            .get("tokenizer.chat_template")?
            .as_str()?
            .to_string();
        let (bos, eos) = Self::bos_eos(g);
        Some(ChatTemplate { src, bos, eos })
    }

    /// Use a caller-supplied template source (e.g. a community override file) but
    /// resolve bos/eos from the GGUF — for models whose embedded template handles
    /// tools poorly.
    pub fn from_gguf_src(g: &GgufFile, src: String) -> ChatTemplate {
        let (bos, eos) = Self::bos_eos(g);
        ChatTemplate { src, bos, eos }
    }

    /// Surgically repair known-broken embedded templates without re-authoring them.
    ///
    /// SmolLM3's GGUF ships the Unsloth-edited template, which only emits the
    /// system turn's closing `<|im_end|>` *inside* the `if xml_tools/python_tools`
    /// block — so a chat with no tools renders a system turn that is never closed
    /// and bleeds straight into the user turn, producing garbage output. We move
    /// that single `<|im_end|>` emission out of the tools `if` (still inside the
    /// non-`/system_override` branch) so the system turn always closes.
    pub fn repair_arch(arch: &str, src: String) -> String {
        if arch == "smollm3" {
            let broken = "    {{- \"\\n\\n\" -}}\n    {{- \"<|im_end|>\\n\" -}}\n  {%- endif -%}\n{%- endif -%}";
            let fixed = "    {{- \"\\n\\n\" -}}\n  {%- endif -%}\n  {{- \"<|im_end|>\\n\" -}}\n{%- endif -%}";
            if src.contains(broken) {
                return src.replace(broken, fixed);
            }
        }
        src
    }

    /// Build directly from a template string (e.g. tokenizer_config.json fallback).
    pub fn from_str(
        src: impl Into<String>,
        bos: impl Into<String>,
        eos: impl Into<String>,
    ) -> Self {
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
        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<Value, JErr> {
                Err(JErr::new(ErrorKind::InvalidOperation, msg))
            },
        );
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
