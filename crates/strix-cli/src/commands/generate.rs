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

/// Qwen3.5/3.6-MoE (`qwen35moe`) CPU-reference path. Hybrid Gated-DeltaNet + full
/// GQA attention + top-8 MoE w/ shared expert (see docs/qwen36-arch.md).
///
/// `StrixTokenizer` is Unigram-only and can't tokenize Qwen's gpt2-BPE vocab, so
/// this path takes **raw token IDs** via `STRIX_QWEN_IDS` (comma/space separated)
/// for now — used to validate the forward against a llama.cpp golden. If unset, it
/// falls back to the `prompt` string parsed as whitespace/comma-separated IDs.
/// Build a per-weight GEMV accelerator for the MoE models (Qwen35/Mellum). `STRIX_ROCM=1`
/// selects the ROCm/HIP backend (near-zero per-call launch overhead — best for the
/// hundreds of expert GEMVs/token an MoE issues); otherwise a Vulkan accel
/// (`GpuWeightAccel` wgpu, or `AshWeightAccel` ash via `STRIX_ASH=1`). All implement
/// per-weight `gemv` for resident Q4_0/Q6_K weights. `None` if no GPU / not built with a
/// GPU feature (the model then stays fully on CPU).
#[allow(unused_mut, unused_assignments, clippy::let_and_return)]
fn build_weight_accel() -> Option<Box<dyn strix_core::WeightAccel>> {
    #[cfg(feature = "rocm")]
    if std::env::var("STRIX_ROCM").is_ok() {
        if let Some(a) = strix_backend_rocm::RocmWeightAccel::new() {
            return Some(Box::new(a) as Box<dyn strix_core::WeightAccel>);
        }
        eprintln!("[gpu] STRIX_ROCM set but no ROCm/HIP device available");
    }
    #[cfg(feature = "vulkan")]
    {
        if std::env::var("STRIX_ASH").is_ok() {
            if let Some(a) = strix_backend_vulkan::AshWeightAccel::new() {
                return Some(Box::new(a) as Box<dyn strix_core::WeightAccel>);
            }
        } else if let Some(a) = strix_backend_vulkan::GpuWeightAccel::new() {
            return Some(Box::new(a) as Box<dyn strix_core::WeightAccel>);
        }
    }
    None
}

fn run_qwen35(gguf: GgufFile, prompt: &str, max_tokens: usize, gpu: bool) -> Result<()> {
    use strix_backend_cpu::qwen35::Qwen35Model;

    let id_src = std::env::var("STRIX_QWEN_IDS").unwrap_or_else(|_| prompt.to_string());
    let prompt_ids: Vec<u32> = id_src
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .context(
            "qwen35moe needs raw token IDs (StrixTokenizer can't do gpt2-BPE). \
             Set STRIX_QWEN_IDS=\"1,2,3\" or pass IDs as the prompt.",
        )?;
    if prompt_ids.is_empty() {
        anyhow::bail!("qwen35moe: no token IDs given (set STRIX_QWEN_IDS)");
    }

    let load_start = Instant::now();
    let mut model = Qwen35Model::from_gguf(gguf).context("build qwen35 model")?;
    eprintln!(
        "[qwen35moe] model built in {:.1}s, prompt = {} tokens",
        load_start.elapsed().as_secs_f64(),
        prompt_ids.len()
    );

    // NPU prefill offload of the dense projections (feature npu-cpu + STRIX_NPU=1).
    #[cfg(feature = "npu-cpu")]
    if std::env::var("STRIX_NPU").is_ok() {
        let dir = std::env::var("STRIX_NPU_DIR").unwrap_or_else(|_| {
            "external/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
                .into()
        });
        match strix_backend_cpu::mellum_npu::QwenNpu::open(&dir) {
            Ok(npu) => {
                let t = Instant::now();
                match model.attach_npu(npu) {
                    Ok(n) => eprintln!(
                        "[qwen35moe] {n} dense projections staged on NPU ({:.1}s)",
                        t.elapsed().as_secs_f64()
                    ),
                    Err(e) => eprintln!("[qwen35moe] NPU staging failed: {e}"),
                }
            }
            Err(e) => eprintln!("[qwen35moe] NPU open failed ({dir}): {e}"),
        }
    }

    // --gpu: offload the dense Q6_K projections (attn q/k/v/o + lm_head) to the iGPU
    // via per-weight gemv (Vulkan). Experts/deltanet stay on CPU (P2). No-op if no
    // Vulkan accel is available.
    if gpu {
        match build_weight_accel() {
            Some(accel) => {
                let name = accel.name().to_string();
                let t = Instant::now();
                let n = model.attach_accel(accel);
                eprintln!(
                    "[qwen35moe] {n} dense weights resident on {name} ({:.1}s upload)",
                    t.elapsed().as_secs_f64()
                );
            }
            None => eprintln!(
                "[qwen35moe] --gpu: no Vulkan accel (rebuild --features vulkan; ROCm gemv is a no-op for MoE)"
            ),
        }
    }

    let sampler = GreedySampler;
    let prefill_start = Instant::now();
    let logits = model.prefill(&prompt_ids).context("qwen35 prefill")?;
    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    let mut next = sampler.sample(&logits)?;

    // Show the top-5 next-token logits for the prompt (sanity vs llama.cpp).
    let mut top: Vec<(usize, f32)> = logits.0.iter().cloned().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("[qwen35moe] prompt next-token top5:");
    for &(id, l) in top.iter().take(5) {
        eprintln!("    id={id:<8} logit={l:.4}");
    }

    let mut generated: Vec<u32> = vec![next];
    let decode_start = Instant::now();
    for _ in 1..max_tokens {
        let l = model.decode_one(next).context("qwen35 decode")?;
        next = sampler.sample(&l)?;
        generated.push(next);
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();

    let ids_str: Vec<String> = generated.iter().map(|t| t.to_string()).collect();
    println!("[qwen35moe] generated token IDs: {}", ids_str.join(","));
    eprintln!(
        "[qwen35moe] prefill {:.1} tok/s ({} tok in {prefill_secs:.2}s) | decode {:.2} tok/s",
        prompt_ids.len() as f64 / prefill_secs,
        prompt_ids.len(),
        (max_tokens.saturating_sub(1)) as f64 / decode_secs.max(1e-9),
    );
    Ok(())
}

/// JetBrains Mellum2 (`mellum`) CPU-reference path. Sparse-MoE transformer with
/// hybrid sliding/full attention + per-layer-type RoPE (YaRN on full layers). Takes
/// raw token IDs via `STRIX_QWEN_IDS` (or the prompt) — same tokenizer caveat as Qwen.
/// SmolLM3-3B (smollm3): CPU-only greedy. gpt2-BPE → raw IDs via STRIX_QWEN_IDS.
fn run_smollm3(gguf: GgufFile, prompt: &str, max_tokens: usize) -> Result<()> {
    use strix_backend_cpu::smollm3::{SmolLm3Cfg, SmolLm3Model};
    use strix_core::backend::Decoder;

    let cfg = SmolLm3Cfg::from_gguf(&gguf).context("parse smollm3 config")?;
    eprintln!("[smollm3] {}", cfg.report());
    let id_src = std::env::var("STRIX_QWEN_IDS").unwrap_or_else(|_| prompt.to_string());
    let prompt_ids: Vec<u32> = id_src
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .context("smollm3 needs raw token IDs (gpt2-BPE). Set STRIX_QWEN_IDS=\"1,2,3\".")?;
    if prompt_ids.is_empty() {
        anyhow::bail!("smollm3: no token IDs given (set STRIX_QWEN_IDS)");
    }
    let max_seq = prompt_ids.len() + max_tokens + 16;
    let load = Instant::now();
    let mut model = SmolLm3Model::from_gguf(gguf, max_seq).context("build smollm3")?;
    eprintln!(
        "[smollm3] built in {:.1}s, prompt = {} tokens",
        load.elapsed().as_secs_f64(),
        prompt_ids.len()
    );
    let sampler = GreedySampler;
    let pf = Instant::now();
    let logits = model.prefill(&prompt_ids).context("smollm3 prefill")?;
    let pf_s = pf.elapsed().as_secs_f64();
    let mut next = sampler.sample(&logits)?;
    let mut generated = vec![next];
    let dec = Instant::now();
    for _ in 1..max_tokens {
        let l = model.decode_one(next).context("smollm3 decode")?;
        next = sampler.sample(&l)?;
        generated.push(next);
    }
    let dec_s = dec.elapsed().as_secs_f64();
    let ids: Vec<String> = generated.iter().map(|t| t.to_string()).collect();
    println!("[smollm3] generated token IDs: {}", ids.join(","));
    eprintln!(
        "[smollm3] prefill {:.1} tok/s ({} tok in {pf_s:.2}s) | decode {:.2} tok/s",
        prompt_ids.len() as f64 / pf_s,
        prompt_ids.len(),
        (max_tokens.saturating_sub(1)) as f64 / dec_s.max(1e-9),
    );
    Ok(())
}

fn run_mellum(gguf: GgufFile, prompt: &str, max_tokens: usize, gpu: bool) -> Result<()> {
    use strix_backend_cpu::mellum::{MellumCfg, MellumModel};

    let cfg = MellumCfg::from_gguf(&gguf).context("parse mellum config")?;
    eprintln!("[mellum] {}", cfg.report());
    // --gpu offload set up after the model is built (below).

    let id_src = std::env::var("STRIX_QWEN_IDS").unwrap_or_else(|_| prompt.to_string());
    let prompt_ids: Vec<u32> = id_src
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .context(
            "mellum needs raw token IDs (StrixTokenizer can't do gpt2-BPE). \
             Set STRIX_QWEN_IDS=\"1,2,3\" or pass IDs as the prompt.",
        )?;
    if prompt_ids.is_empty() {
        anyhow::bail!("mellum: no token IDs given (set STRIX_QWEN_IDS)");
    }

    let load_start = Instant::now();
    let mut model = MellumModel::from_gguf(gguf).context("build mellum model")?;
    eprintln!(
        "[mellum] model built in {:.1}s, prompt = {} tokens",
        load_start.elapsed().as_secs_f64(),
        prompt_ids.len()
    );

    // NPU prefill offload (feature npu-cpu + STRIX_NPU=1): stages dense q/o + experts
    // (cap STRIX_NPU_EXPERT_LAYERS) onto the XDNA2 NPU as per-channel int8. CPU-driven,
    // zero iGPU involvement — fits the min-iGPU posture. xclbin dir via STRIX_NPU_DIR.
    #[cfg(feature = "npu-cpu")]
    if std::env::var("STRIX_NPU").is_ok() {
        let dir = std::env::var("STRIX_NPU_DIR").unwrap_or_else(|_| {
            "external/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
                .into()
        });
        match strix_backend_cpu::mellum_npu::MellumNpu::open(&dir) {
            Ok(npu) => {
                let t = Instant::now();
                match model.attach_npu(npu) {
                    Ok(n) => eprintln!(
                        "[mellum] {n} weights staged on NPU ({:.1}s)",
                        t.elapsed().as_secs_f64()
                    ),
                    Err(e) => eprintln!("[mellum] NPU staging failed: {e}"),
                }
            }
            Err(e) => eprintln!("[mellum] NPU open failed ({dir}): {e}"),
        }
    }

    // --gpu: Mellum is all-Q8_0 → needs the ROCm accel (STRIX_ROCM=1; Vulkan adopts
    // nothing). Uploads dense q/k/v/o + output + per-expert slices (cap via
    // STRIX_GPU_EXPERT_LAYERS). 12B fits resident, unlike the 35B.
    if gpu {
        match build_weight_accel() {
            Some(accel) => {
                let name = accel.name().to_string();
                let t = Instant::now();
                let n = model.attach_accel(accel);
                eprintln!(
                    "[mellum] {n} weights resident on {name} ({:.1}s upload)",
                    t.elapsed().as_secs_f64()
                );
            }
            None => eprintln!("[mellum] --gpu: no accel (build --features rocm + STRIX_ROCM=1)"),
        }
    }

    let sampler = GreedySampler;
    let prefill_start = Instant::now();
    let logits = model.prefill(&prompt_ids).context("mellum prefill")?;
    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    let mut next = sampler.sample(&logits)?;

    let mut top: Vec<(usize, f32)> = logits.0.iter().cloned().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("[mellum] prompt next-token top5:");
    for &(id, l) in top.iter().take(5) {
        eprintln!("    id={id:<8} logit={l:.4}");
    }

    let mut generated: Vec<u32> = vec![next];
    let decode_start = Instant::now();
    // Lookup speculation is LOSSLESS; wins on long generations (+19% @512, more @1k+),
    // ~breakeven short. Default ON; STRIX_NO_LOOKUP=1 disables.
    // Graph decode (24 t/s) beats lookup-verify; lookup now opt-in STRIX_LOOKUP.
    let lookup = std::env::var("STRIX_LOOKUP").is_ok();
    if lookup {
        // Lossless n-gram lookup speculation: propose continuation from history,
        // verify all candidates in ONE batched forward, accept matching prefix.
        // Small-K speculation (Cascade-style): MoE verify reads the expert UNION, so
        // bytes grow with K. K=3 → union ~1.6-2x bytes; net win iff acceptance is good.
        // Utility gate: 4 zero-acceptance rounds in a row → plain decode for 96 tokens.
        let gamma = std::env::var("STRIX_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3usize);
        let mut zero_rounds = 0usize;
        let mut plain_left = 0usize;
        let mut ctx: Vec<u32> = prompt_ids.clone();
        ctx.push(next);
        while generated.len() < max_tokens {
            if plain_left > 0 {
                plain_left -= 1;
                let l = model.decode_one(next).context("mellum decode")?;
                next = sampler.sample(&l)?;
                generated.push(next);
                ctx.push(next);
                continue;
            }
            // propose: longest n-gram (3..1) suffix match in history
            let mut prop: Vec<u32> = Vec::new();
            'find: for n in (1..=3usize).rev() {
                if ctx.len() < n + 1 {
                    continue;
                }
                let pat = &ctx[ctx.len() - n..];
                for st in (0..ctx.len() - n).rev() {
                    if &ctx[st..st + n] == pat {
                        let cont = &ctx[st + n..(st + n + gamma).min(ctx.len() - 1)];
                        if !cont.is_empty() {
                            prop = cont.to_vec();
                            break 'find;
                        }
                    }
                }
            }
            let pos0 = model.pos();
            let mut cand = vec![next];
            cand.extend(&prop);
            let outs = model.verify_tokens(&cand).context("verify")?;
            // accept prop[i] while it equals model's argmax at i
            let mut acc = 0usize;
            while acc < prop.len()
                && outs[acc] == prop[acc]
                && generated.len() + acc + 1 < max_tokens
            {
                acc += 1;
            }
            for t in &prop[..acc] {
                generated.push(*t);
                ctx.push(*t);
            }
            next = outs[acc];
            generated.push(next);
            ctx.push(next);
            // rollback the rejected suffix (we forwarded all gamma candidates)
            model.rollback(pos0 + acc + 1);
            if acc == 0 {
                zero_rounds += 1;
                if zero_rounds >= 4 {
                    zero_rounds = 0;
                    plain_left = 96;
                }
            } else {
                zero_rounds = 0;
            }
        }
    } else {
        for _ in 1..max_tokens {
            let l = model.decode_one(next).context("mellum decode")?;
            next = sampler.sample(&l)?;
            generated.push(next);
        }
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();

    let ids_str: Vec<String> = generated.iter().map(|t| t.to_string()).collect();
    println!("[mellum] generated token IDs: {}", ids_str.join(","));
    eprintln!(
        "[mellum] prefill {:.1} tok/s ({} tok in {prefill_secs:.2}s) | decode {:.2} tok/s",
        prompt_ids.len() as f64 / prefill_secs,
        prompt_ids.len(),
        (max_tokens.saturating_sub(1)) as f64 / decode_secs.max(1e-9),
    );
    Ok(())
}

/// GGUF/Gemma path: tokenizer + config + quantized weights all from the GGUF.
fn run_gguf(path: &Path, prompt: &str, max_tokens: usize, chat: bool, gpu: bool) -> Result<()> {
    let total_start = Instant::now();

    let load_start = Instant::now();
    let gguf = GgufFile::open(path).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("?").to_string();
    tracing::info!(arch = %arch, "loaded gguf");

    // Qwen3.5/3.6-MoE (qwen35moe) bring-up — Phase 0: recognize + parse config +
    // validate tensors. Forward not yet implemented (hybrid Gated-DeltaNet/MoE,
    // see docs/qwen36-arch.md). Reports and exits rather than failing in GemmaModel.
    if arch == "qwen35moe" {
        return run_qwen35(gguf, prompt, max_tokens, gpu);
    }
    if arch == "mellum" {
        return run_mellum(gguf, prompt, max_tokens, gpu);
    }
    if arch == "smollm3" {
        return run_smollm3(gguf, prompt, max_tokens);
    }

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

    // In-process benchmark mode (STRIX_BENCH=<reps>): load ONCE, then time
    // prefill+decode over N reps after a warmup — robust to GPU clock ramp and
    // the ±few tok/s run-to-run noise that makes single-shot `generate` numbers
    // unreliable. Reports median/min/max. (idea: decode-#21 / benchmark harness.)
    if let Ok(bs) = std::env::var("STRIX_BENCH") {
        let reps: usize = bs.parse().unwrap_or(5);
        let dn = max_tokens.max(8); // decode steps to time per rep
        eprintln!(
            "[bench] {reps} timed reps (+1 warmup), prompt {} tok, {dn} decode steps/rep",
            prompt_ids.len()
        );
        // 2 warmup reps: the iGPU clock ramps over ~30 s of sustained load, so a
        // single warmup still measures a cold-ish clock (observed 80→88 tok/s).
        let warm = 2usize;
        let (mut pf, mut dc) = (Vec::new(), Vec::new());
        for r in 0..(reps + warm) {
            model.set_seq(0);
            let t0 = Instant::now();
            let logits = model.prefill(&prompt_ids).context("bench prefill")?;
            let ps = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let mut next = sampler.sample(&logits)?;
            for _ in 0..dn {
                next = model.decode_one_token(next)?; // greedy fast path (on-device argmax)
            }
            let _ = next;
            let ds = t1.elapsed().as_secs_f64();
            if r >= warm {
                let (pr, dr) = (prompt_ids.len() as f64 / ps, dn as f64 / ds);
                pf.push(pr);
                dc.push(dr);
                eprintln!(
                    "[bench] rep{}: prefill {pr:.1} tok/s | decode {dr:.2} tok/s",
                    r - warm + 1
                );
            } else {
                eprintln!("[bench] warmup {} done (discarded)", r + 1);
            }
        }
        let stat = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[v.len() / 2], v[0], v[v.len() - 1])
        };
        let (pm, plo, phi) = stat(&mut pf);
        let (dm, dlo, dhi) = stat(&mut dc);
        eprintln!(
            "[bench RESULT] prefill median {pm:.1} (min {plo:.1} max {phi:.1}) | decode median {dm:.2} (min {dlo:.2} max {dhi:.2}) tok/s | load {load_secs:.1}s"
        );
        return Ok(());
    }

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
    let logits = model.prefill(&prompt_ids).context("prefill failed")?;
    let prefill_secs = prefill_start.elapsed().as_secs_f64();

    // --- Decode loop (greedy) ---
    // First token from the prefill logits, then the on-device argmax fast path
    // (no vocab-wide logits readback per step).
    let mut generated: Vec<u32> = Vec::new();
    let decode_start = Instant::now();
    let mut next = sampler.sample(&logits)?;
    for _ in 0..max_tokens {
        if Some(next) == eos_id {
            break;
        }
        generated.push(next);
        next = model.decode_one_token(next)?;
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
