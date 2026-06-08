//! Tokenizer loading, wrapping the HuggingFace `tokenizers` crate.
//!
//! Implements [`strix_core::tokenizer::Tokenizer`] over a `tokenizer.json`.
//! BOS/EOS ids are resolved best-effort by probing common special-token
//! strings, since `tokenizer.json` alone doesn't label them uniformly across
//! model families. A caller that knows the exact ids (from `config.json`) can
//! override them with [`StrixTokenizer::set_special_ids`].

use std::path::Path;

use serde_json::json;
use strix_core::error::{Result, StrixError};
use strix_core::tokenizer::Tokenizer;
use tokenizers::Tokenizer as HfTokenizer;

use crate::gguf::{GgufFile, MetaValue};

// llama.cpp token-type tags.
const TT_CONTROL: i64 = 3;
const TT_USER_DEFINED: i64 = 4;

/// Common BOS marker strings across Llama/Mistral/Qwen families.
const BOS_CANDIDATES: &[&str] = &["<|begin_of_text|>", "<s>", "<|startoftext|>"];
/// Common EOS marker strings (base + chat variants).
const EOS_CANDIDATES: &[&str] = &[
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|im_end|>",
    "<|endoftext|>",
    "</s>",
];

/// A loaded tokenizer.
pub struct StrixTokenizer {
    inner: HfTokenizer,
    bos: Option<u32>,
    eos: Option<u32>,
}

impl StrixTokenizer {
    /// Load from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| StrixError::parse(format!("tokenizer.json: {e}")))?;
        let bos = first_known(&inner, BOS_CANDIDATES);
        let eos = first_known(&inner, EOS_CANDIDATES);
        Ok(StrixTokenizer { inner, bos, eos })
    }

    /// Build a tokenizer from a GGUF's embedded SentencePiece table.
    ///
    /// GGUF stores Gemma/Llama tokenizers as a unigram model: `tokens` +
    /// `scores` + `token_type`, with `▁` for spaces and `<0xXX>` byte-fallback
    /// tokens. We reconstruct a `tokenizers` Unigram model from that table
    /// (authoritative for the model being run) plus Gemma's standard
    /// normalizer/decoder, so decoding is exact by construction.
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let tokens = str_array(g, "tokenizer.ggml.tokens")?;
        let scores = f32_array(g, "tokenizer.ggml.scores")?;
        let types = int_array(g, "tokenizer.ggml.token_type").unwrap_or_default();
        if tokens.len() != scores.len() {
            return Err(StrixError::invalid(format!(
                "GGUF tokenizer: {} tokens vs {} scores",
                tokens.len(),
                scores.len()
            )));
        }

        let unk = g
            .meta("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let bos = g
            .meta("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|x| x as u32);
        let eos = g
            .meta("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|x| x as u32);

        // Unigram vocab is an ordered [token, score] list; index == token id.
        let vocab: Vec<serde_json::Value> = tokens
            .iter()
            .zip(scores.iter())
            .map(|(t, s)| json!([t, *s as f64]))
            .collect();

        // Control / user-defined tokens become non-normalized special tokens so
        // they are matched whole rather than split into pieces.
        let added: Vec<serde_json::Value> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| matches!(types.get(*i).copied(), Some(TT_CONTROL | TT_USER_DEFINED)))
            .map(|(i, t)| {
                json!({
                    "id": i, "content": t, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false, "special": true
                })
            })
            .collect();

        // Gemma SPM config: replace spaces with ▁ on the way in, reverse it plus
        // byte-fallback fuse on the way out.
        let tk_json = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": { "type": "Replace", "pattern": { "String": " " }, "content": "\u{2581}" },
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": { "type": "Sequence", "decoders": [
                { "type": "Replace", "pattern": { "String": "\u{2581}" }, "content": " " },
                { "type": "ByteFallback" },
                { "type": "Fuse" }
            ]},
            "model": {
                "type": "Unigram",
                "unk_id": unk,
                "vocab": vocab,
                "byte_fallback": true
            }
        });

        let bytes = serde_json::to_vec(&tk_json)
            .map_err(|e| StrixError::parse(format!("building tokenizer json: {e}")))?;
        let inner = HfTokenizer::from_bytes(&bytes)
            .map_err(|e| StrixError::parse(format!("GGUF tokenizer: {e}")))?;

        Ok(StrixTokenizer { inner, bos, eos })
    }

    /// Override the resolved BOS/EOS ids (e.g. from `config.json`). `None`
    /// leaves the existing value unchanged.
    pub fn set_special_ids(&mut self, bos: Option<u32>, eos: Option<u32>) {
        if bos.is_some() {
            self.bos = bos;
        }
        if eos.is_some() {
            self.eos = eos;
        }
    }
}

fn first_known(tok: &HfTokenizer, candidates: &[&str]) -> Option<u32> {
    candidates.iter().find_map(|s| tok.token_to_id(s))
}

fn meta_array<'a>(g: &'a GgufFile, key: &str) -> Result<&'a [MetaValue]> {
    g.meta(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| StrixError::invalid(format!("GGUF tokenizer: missing array `{key}`")))
}

fn str_array(g: &GgufFile, key: &str) -> Result<Vec<String>> {
    Ok(meta_array(g, key)?
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect())
}

fn f32_array(g: &GgufFile, key: &str) -> Result<Vec<f32>> {
    Ok(meta_array(g, key)?
        .iter()
        .map(|v| v.as_f32().unwrap_or(0.0))
        .collect())
}

fn int_array(g: &GgufFile, key: &str) -> Result<Vec<i64>> {
    Ok(meta_array(g, key)?
        .iter()
        .map(|v| v.as_u64().map(|x| x as i64).unwrap_or(0))
        .collect())
}

impl Tokenizer for StrixTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| StrixError::parse(format!("encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(tokens, skip_special_tokens)
            .map_err(|e| StrixError::parse(format!("decode: {e}")))
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
