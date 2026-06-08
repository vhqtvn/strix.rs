//! Lightweight, non-loading model inspection.
//!
//! Walks a model directory (or single file) and reports what Strix can tell
//! *without* loading weights: detected format, config, tokenizer presence,
//! weight shard files. This backs the `inspect-model` CLI command.

use std::fs;
use std::path::{Path, PathBuf};

use strix_core::error::{Result, StrixError};

use crate::hf_config::HfConfig;

/// Weight serialization format detected on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// One or more `.safetensors` files (Phase 1 target).
    Safetensors,
    /// A `.gguf` file (Phase 2 target).
    Gguf,
    /// Could not determine.
    Unknown,
}

impl std::fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            WeightFormat::Safetensors => "safetensors",
            WeightFormat::Gguf => "gguf",
            WeightFormat::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

/// Result of inspecting a model path.
#[derive(Debug, Clone)]
pub struct ModelInspection {
    /// The path that was inspected.
    pub path: PathBuf,
    /// Detected weight format.
    pub format: WeightFormat,
    /// Parsed/normalized config, if a `config.json` was found and understood.
    pub config: Option<strix_core::model_config::ModelConfig>,
    /// Whether a `tokenizer.json` is present.
    pub has_tokenizer: bool,
    /// Weight files found (relative names).
    pub weight_files: Vec<String>,
    /// Non-fatal notes for the user.
    pub notes: Vec<String>,
}

/// Inspect a model directory or single weight file.
pub fn inspect_model(path: &Path) -> Result<ModelInspection> {
    if !path.exists() {
        return Err(StrixError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("path does not exist: {}", path.display()),
        )));
    }

    let mut notes = Vec::new();
    let mut weight_files = Vec::new();
    let mut format = WeightFormat::Unknown;
    let mut has_tokenizer = false;
    let mut config = None;

    if path.is_file() {
        // Single-file case: most likely a GGUF.
        if has_ext(path, "gguf") {
            format = WeightFormat::Gguf;
        } else if has_ext(path, "safetensors") {
            format = WeightFormat::Safetensors;
        }
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            weight_files.push(name.to_string());
        }
        notes.push("single-file input: directory metadata not scanned".to_string());
        return Ok(ModelInspection {
            path: path.to_path_buf(),
            format,
            config,
            has_tokenizer,
            weight_files,
            notes,
        });
    }

    // Directory case: scan entries.
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let p = entry.path();

        if has_ext(&p, "safetensors") {
            format = WeightFormat::Safetensors;
            weight_files.push(name);
        } else if has_ext(&p, "gguf") {
            // GGUF takes precedence only if no safetensors seen.
            if format != WeightFormat::Safetensors {
                format = WeightFormat::Gguf;
            }
            weight_files.push(name);
        } else if name == "tokenizer.json" {
            has_tokenizer = true;
        } else if name == "config.json" {
            match fs::read(&p).map_err(StrixError::from).and_then(|bytes| {
                let hf = HfConfig::from_json_slice(&bytes)?;
                hf.to_model_config()
            }) {
                Ok(cfg) => config = Some(cfg),
                Err(e) => notes.push(format!("could not normalize config.json: {e}")),
            }
        }
    }

    if weight_files.is_empty() {
        notes.push("no .safetensors or .gguf weight files found".to_string());
    }
    if !has_tokenizer {
        notes.push("no tokenizer.json found".to_string());
    }
    weight_files.sort();

    Ok(ModelInspection {
        path: path.to_path_buf(),
        format,
        config,
        has_tokenizer,
        weight_files,
        notes,
    })
}

fn has_ext(p: &Path, ext: &str) -> bool {
    p.extension().and_then(|e| e.to_str()) == Some(ext)
}
