//! `strix generate` — CPU reference text generation.
//!
//! Two paths, dispatched by what `--model` points at:
//! - a `.gguf` file (or a dir containing one) → GGUF/Gemma path (quantized);
//! - an HF dir (config.json + tokenizer.json + *.safetensors) → Llama path.
//!
//! Both prefill the prompt then greedily decode up to `max_tokens`, stopping on
//! EOS. Single-sequence, greedy only.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use strix_backend_cpu::{GemmaModel, LlamaModel};
use strix_core::backend::Decoder;
use strix_core::model_config::ModelConfig;
use strix_core::sampler::{GreedySampler, Sampler};
use strix_core::tokenizer::Tokenizer;
use strix_models::gguf::GgufFile;
use strix_models::{HfConfig, StrixTokenizer};

/// Dispatch to the GGUF or safetensors path based on the model location.
pub fn run(model: &Path, prompt: &str, max_tokens: usize, chat: bool, gpu: bool) -> Result<()> {
    match find_gguf(model) {
        Some(gguf) => run_gguf(&gguf, prompt, max_tokens, chat, gpu),
        None => run_safetensors(model, prompt, max_tokens),
    }
}

/// Build the iGPU weight accelerator and attach it to the model, uploading the
/// Q4_0/Q6_K weights resident. No-op (with a note) unless built `--features
/// vulkan`. Kept here so the GPU dependency lives only in the CLI.
#[allow(unused_variables)]
/// Index of the max logit (greedy argmax).
fn argmax(l: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

fn attach_gpu(model: &mut GemmaModel) {
    #[allow(unused_imports)]
    use strix_core::WeightAccel;
    #[allow(unused_mut, unused_assignments)]
    let mut accel: Option<(Box<dyn WeightAccel>, &str)> = None;

    // `STRIX_ROCM=1` selects the ROCm/HIP backend (gfx1150). Highest priority.
    #[cfg(feature = "rocm")]
    if accel.is_none() && std::env::var("STRIX_ROCM").is_ok() {
        accel = strix_backend_rocm::RocmWeightAccel::new()
            .map(|a| (Box::new(a) as Box<dyn WeightAccel>, "rocm"));
        if accel.is_none() {
            eprintln!("[gpu] STRIX_ROCM set but no ROCm/HIP device available");
        }
    }

    // Vulkan: `STRIX_ASH=1` = raw-Vulkan (ash); else wgpu.
    #[cfg(feature = "vulkan")]
    if accel.is_none() {
        let ash = std::env::var("STRIX_ASH").is_ok();
        accel = if ash {
            strix_backend_vulkan::AshWeightAccel::new()
                .map(|a| (Box::new(a) as Box<dyn WeightAccel>, "ash"))
        } else {
            strix_backend_vulkan::GpuWeightAccel::new()
                .map(|a| (Box::new(a) as Box<dyn WeightAccel>, "wgpu"))
        };
    }

    match accel {
        Some((accel, tag)) => {
            let name = accel.name().to_string();
            let upload_start = Instant::now();
            let n = model.attach_accel(accel);
            eprintln!(
                "[gpu] {n} weights resident on {name} [{tag}] ({:.1}s upload)",
                upload_start.elapsed().as_secs_f64()
            );
        }
        None => {
            #[cfg(any(feature = "vulkan", feature = "rocm"))]
            eprintln!("[gpu] no GPU device available; staying on CPU");
            #[cfg(not(any(feature = "vulkan", feature = "rocm")))]
            eprintln!(
                "[gpu] built without `vulkan`/`rocm`; rebuild with --features vulkan or rocm"
            );
        }
    }
}

/// Locate a GGUF file: the path itself, or the first `*.gguf` in a directory
/// (ignoring multimodal projector shards).
fn find_gguf(model: &Path) -> Option<PathBuf> {
    if model.is_file() && model.extension().and_then(|e| e.to_str()) == Some("gguf") {
        return Some(model.to_path_buf());
    }
    if model.is_dir() {
        let mut found: Vec<PathBuf> = std::fs::read_dir(model)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
            .filter(|p| {
                !p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.contains("mmproj"))
                    .unwrap_or(false)
            })
            .collect();
        found.sort();
        return found.into_iter().next();
    }
    None
}

/// GGUF/Gemma path: tokenizer + config + quantized weights all from the GGUF.
fn run_gguf(path: &Path, prompt: &str, max_tokens: usize, chat: bool, gpu: bool) -> Result<()> {
    let total_start = Instant::now();

    let load_start = Instant::now();
    let gguf = GgufFile::open(path).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("?").to_string();
    tracing::info!(arch = %arch, "loaded gguf");

    let tokenizer = StrixTokenizer::from_gguf(&gguf).context("build tokenizer from gguf")?;

    // In chat mode wrap the prompt in the Gemma turn template. Gemma-4 uses the
    // `<|turn>` / `<turn|>` markers; Gemma-3 uses `<start_of_turn>`/`<end_of_turn>`.
    let g4 = arch == "gemma4";
    let (turn_open, turn_close) = if g4 {
        ("<|turn>", "<turn|>")
    } else {
        ("<start_of_turn>", "<end_of_turn>")
    };
    let text = if chat {
        format!("{turn_open}user\n{prompt}{turn_close}\n{turn_open}model\n")
    } else {
        prompt.to_string()
    };

    // Encode and prepend BOS (Gemma sets add_bos_token=true).
    let mut prompt_ids = tokenizer.encode(&text, false).context("encode prompt")?;
    if let Some(bos) = tokenizer.bos_token_id() {
        prompt_ids.insert(0, bos);
    }
    if prompt_ids.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens");
    }
    // Stop on EOS, and on the end-of-turn token in chat mode.
    let eos_id = tokenizer.eos_token_id();
    let turn_close_id = if chat {
        tokenizer.encode(turn_close, false).ok().and_then(|v| {
            if v.len() == 1 {
                Some(v[0])
            } else {
                None
            }
        })
    } else {
        None
    };

    let max_seq = prompt_ids.len() + max_tokens + 16;
    let mut model = GemmaModel::from_gguf(gguf, max_seq).context("build gemma model")?;
    if gpu {
        attach_gpu(&mut model);
    }
    // Speculative decoding: STRIX_DRAFT=<draft.gguf> loads a small draft model
    // (must share the target's vocab) that proposes tokens the target verifies in
    // one batched pass. STRIX_SPEC_GAMMA sets the draft length (default 4).
    let mut draft: Option<GemmaModel> = None;
    if gpu {
        if let Ok(dpath) = std::env::var("STRIX_DRAFT") {
            let dg = GgufFile::open(std::path::Path::new(&dpath)).context("open draft gguf")?;
            let mut dm = GemmaModel::from_gguf(dg, max_seq).context("build draft model")?;
            attach_gpu(&mut dm);
            eprintln!("[spec] draft model loaded from {dpath}");
            draft = Some(dm);
        }
    }
    let load_secs = load_start.elapsed().as_secs_f64();

    let sampler = GreedySampler;
    let prefill_start = Instant::now();
    let mut logits = model.prefill(&prompt_ids).context("prefill")?;
    let prefill_secs = prefill_start.elapsed().as_secs_f64();

    let mut generated: Vec<u32> = Vec::new();
    #[cfg(feature = "vulkan")]
    strix_backend_vulkan::reset_gpu_time();
    let emit = |tok: u32, generated: &mut Vec<u32>| {
        generated.push(tok);
        if let Ok(piece) = tokenizer.decode(&[tok], true) {
            print!("{piece}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    };
    let is_stop = |t: u32| Some(t) == eos_id || Some(t) == turn_close_id;
    let decode_start = Instant::now();
    let mut spec_stats: Option<(usize, usize)> = None; // (rounds, accepted_drafts)
    if let Some(ref mut draft) = draft {
        // --- Speculative decoding (greedy → lossless vs plain greedy) ---
        let gamma: usize = std::env::var("STRIX_SPEC_GAMMA")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        draft.prefill(&prompt_ids).context("draft prefill")?; // warm draft KV to the prompt
        let mut cur = sampler.sample(&logits)?;
        let mut pos = prompt_ids.len();
        let (mut rounds, mut acc_total) = (0usize, 0usize);
        'spec: while generated.len() < max_tokens {
            // draft proposes gamma tokens, forwarding cur,d0..d(gamma-1) into its KV
            let mut drafts = Vec::with_capacity(gamma);
            let mut d = cur;
            for _ in 0..gamma {
                let dl = draft.decode_one(d)?;
                d = argmax(&dl.0);
                drafts.push(d);
            }
            draft.decode_one(drafts[gamma - 1])?; // commit the last draft into draft KV
                                                  // target verifies [cur, d0..d(gamma-1)] in ONE batched pass → gamma+1 logits
            let mut vin = Vec::with_capacity(gamma + 1);
            vin.push(cur);
            vin.extend_from_slice(&drafts);
            let vlog = model.verify(&vin, pos).context("target verify")?;
            // accept the longest prefix where target argmax == draft
            let mut n = 0;
            while n < gamma && argmax(&vlog[n]) == drafts[n] {
                n += 1;
            }
            let new_cur = argmax(&vlog[n]); // n<gamma: correction; n==gamma: bonus
                                            // commit cur + accepted drafts (new_cur carries to next round)
            let mut batch = Vec::with_capacity(n + 1);
            batch.push(cur);
            batch.extend_from_slice(&drafts[0..n]);
            for tok in batch {
                if is_stop(tok) {
                    break 'spec;
                }
                emit(tok, &mut generated);
                if generated.len() >= max_tokens {
                    break 'spec;
                }
            }
            pos += n + 1;
            model.set_seq(pos);
            draft.set_seq(pos);
            cur = new_cur;
            rounds += 1;
            acc_total += n;
        }
        spec_stats = Some((rounds, acc_total));
    } else {
        for _ in 0..max_tokens {
            let next = sampler.sample(&logits)?;
            if is_stop(next) {
                break;
            }
            emit(next, &mut generated);
            logits = model.decode_one(next)?;
        }
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();
    println!();
    if let Some((rounds, acc)) = spec_stats {
        let n_gen = generated.len().max(1);
        eprintln!(
            "[spec] {rounds} rounds, {acc} drafts accepted, {:.2} tokens/round (acceptance {:.0}%)",
            n_gen as f64 / rounds.max(1) as f64,
            100.0 * acc as f64
                / (rounds.max(1)
                    * std::env::var("STRIX_SPEC_GAMMA")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(4usize)) as f64,
        );
    }

    eprintln!(
        "[{arch} | load {:.2}s | prefill {} tok @ {:.2} tok/s | decode {} tok @ {:.2} tok/s | total {:.1}s]",
        load_secs,
        prompt_ids.len(),
        rate(prompt_ids.len(), prefill_secs),
        generated.len(),
        rate(generated.len(), decode_secs),
        total_start.elapsed().as_secs_f64(),
    );
    #[cfg(feature = "vulkan")]
    if gpu && !generated.is_empty() {
        let gpu_ms = strix_backend_vulkan::gpu_time_ms();
        let total_ms = decode_secs * 1e3;
        eprintln!(
            "[decode split: GPU matmul {:.1} ms/tok ({:.0}%) | CPU+sync glue {:.1} ms/tok ({:.0}%)]",
            gpu_ms / generated.len() as f64,
            100.0 * gpu_ms / total_ms,
            (total_ms - gpu_ms) / generated.len() as f64,
            100.0 * (total_ms - gpu_ms) / total_ms,
        );
    }
    Ok(())
}

/// Safetensors/Llama path (Milestone 2).
fn run_safetensors(model_dir: &Path, prompt: &str, max_tokens: usize) -> Result<()> {
    let total_start = Instant::now();

    // --- Config + tokenizer ---
    let (config, bos, eos) = load_config(model_dir)?;
    let mut tokenizer = load_tokenizer(model_dir)?;
    tokenizer.set_special_ids(bos, eos);

    // --- Encode prompt ---
    let prompt_ids = tokenizer
        .encode(prompt, true)
        .context("failed to encode prompt")?;
    if prompt_ids.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens");
    }
    let eos_id = tokenizer.eos_token_id();

    // --- Load weights + build model ---
    let max_seq = prompt_ids.len() + max_tokens;
    tracing::info!(
        layers = config.num_hidden_layers,
        hidden = config.hidden_size,
        prompt_tokens = prompt_ids.len(),
        max_seq,
        "loading weights"
    );
    let load_start = Instant::now();
    let tensors = strix_models::load_safetensors(model_dir).context("failed to load weights")?;
    let mut model = LlamaModel::from_tensors(config.clone(), tensors, max_seq)
        .context("failed to build model")?;
    let load_secs = load_start.elapsed().as_secs_f64();

    // --- Prefill ---
    let sampler = GreedySampler;
    let prefill_start = Instant::now();
    let mut logits = model.prefill(&prompt_ids).context("prefill failed")?;
    let prefill_secs = prefill_start.elapsed().as_secs_f64();

    // --- Decode loop (greedy) ---
    let mut generated: Vec<u32> = Vec::new();
    let decode_start = Instant::now();
    for _ in 0..max_tokens {
        let next = sampler.sample(&logits)?;
        if Some(next) == eos_id {
            break;
        }
        generated.push(next);
        logits = model.decode_one(next)?;
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();

    // --- Output ---
    // Decode prompt + continuation together so SentencePiece spacing at the
    // seam is correct (decoding the continuation alone drops its leading space).
    let mut full = prompt_ids.clone();
    full.extend_from_slice(&generated);
    let text = tokenizer
        .decode(&full, true)
        .context("failed to decode output")?;

    println!("{text}");
    eprintln!(
        "\n[load {:.2}s | prefill {} tok @ {:.1} tok/s | decode {} tok @ {:.1} tok/s | total {:.2}s]",
        load_secs,
        prompt_ids.len(),
        rate(prompt_ids.len(), prefill_secs),
        generated.len(),
        rate(generated.len(), decode_secs),
        total_start.elapsed().as_secs_f64(),
    );

    Ok(())
}

/// Load and normalize `config.json`, returning the model config and special ids.
fn load_config(dir: &Path) -> Result<(ModelConfig, Option<u32>, Option<u32>)> {
    let path = dir.join("config.json");
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let hf = HfConfig::from_json_slice(&bytes)?;
    let config = hf.to_model_config()?;
    Ok((config, hf.bos_id(), hf.eos_id()))
}

/// Load `tokenizer.json`.
fn load_tokenizer(dir: &Path) -> Result<StrixTokenizer> {
    let path = dir.join("tokenizer.json");
    StrixTokenizer::from_file(&path).with_context(|| format!("loading {}", path.display()))
}

fn rate(count: usize, secs: f64) -> f64 {
    if secs > 0.0 {
        count as f64 / secs
    } else {
        0.0
    }
}
