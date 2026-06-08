//! Tokenizer interface.
//!
//! Strix delegates actual tokenization to the HuggingFace `tokenizers` crate in
//! later phases, but the engine only needs this small trait. Keeping it behind
//! a trait means the CPU reference path and tests can use a trivial stub.

use crate::error::Result;

/// Encodes text to token ids and back.
pub trait Tokenizer: Send + Sync {
    /// Encode a string into token ids.
    ///
    /// `add_special_tokens` controls whether BOS/EOS-style markers are added.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Decode token ids back into text.
    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Beginning-of-sequence token id, if the tokenizer defines one.
    fn bos_token_id(&self) -> Option<u32>;

    /// End-of-sequence token id, if the tokenizer defines one.
    fn eos_token_id(&self) -> Option<u32>;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
}
