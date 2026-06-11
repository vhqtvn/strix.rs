//! Gemma-3n-E4B (`gemma3n`) — the MatFormer/effective-param architecture with
//! AltUp (4 parallel residual streams mixed by learned predict/correct), Laurel
//! (low-rank augmented residual), Per-Layer Embeddings (PLE), KV-sharing for the
//! tail layers, Gaussian-topk activation sparsity on early layers, and a
//! sliding-window attention pattern. Single-token CPU forward (n_tokens=1).
//!
//! Verified against refs/llama.cpp/src/models/gemma3n.cpp. Greedy decoding is
//! invariant to the (monotone) final logit softcap, so it is omitted.
//!
//! STATUS [VALIDATED]: forward verified faithful to llama.cpp via layer-by-layer
//! activation diff (eval-callback): per-layer embeddings (Q5_1) bit-match, layer-0
//! intermediates match to 3-4 decimals, final logits within Q4 accumulation noise.
//! With the proper chat template it answers correctly ("What is the capital of
//! France?" → "The capital of France is **Paris**."). The earlier "fluent but
//! wrong" was a bare-prompt artifact (IT model with no chat template), NOT a bug.
//! STRIX_G3N_DUMP dumps token-0 activations for re-validation.
//! Constants for E4B: 35 layers, n_embd=2048, n_altup=4, n_embd_altup=256,
//! laurel_rank=64, 8/2 heads hd=256, attn_scale=1.0, sparsity layers 0..10,
//! KV from layer 20 (full→19, swa→18), swa pattern (il+1)%5!=0, window=512,
//! rope 1e6 global / 1e4 local. Tied lm_head (token_embd). Tokenizer is
//! SentencePiece here but driven by raw IDs (STRIX_QWEN_IDS) for bring-up.

use rayon::prelude::*;
use strix_core::accel::{G3nConfig, WeightAccel};
use strix_core::backend::Decoder;
use strix_core::error::{Result, StrixError};
use strix_core::sampler::Logits;
use strix_models::ggml_quant::{dequantize_into, GgmlType};
use strix_models::gguf::GgufFile;

const N_ALTUP: usize = 4;
const I_ACT: usize = 0;
const LAUREL_RANK: usize = 64;

fn meta_u32(g: &GgufFile, k: &str) -> Result<usize> {
    g.meta_u32(k).map(|v| v as usize)
}

pub struct Gemma3nCfg {
    pub hidden: usize,     // n_embd 2048
    pub embd_altup: usize, // 256
    pub n_heads: usize,    // 8
    pub n_kv: usize,       // 2
    pub head_dim: usize,   // 256
    pub n_layers: usize,   // 35
    pub vocab: usize,      // 262144
    pub eps: f32,
    pub rope_global: f32,
    pub rope_local: f32,
    pub n_swa: usize,
    pub kv_from_start: usize, // 20
    pub n_sparsity: usize,    // 10
    pub sparsity_mul: f32,    // 1.6448535919
    pub ffn: Vec<usize>,      // per-layer
}

impl Gemma3nCfg {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g
            .architecture()
            .ok_or_else(|| StrixError::invalid("gguf: no general.architecture"))?;
        if arch != "gemma3n" {
            return Err(StrixError::unsupported(format!(
                "gemma3n loader got `{arch}`"
            )));
        }
        let k = |s: &str| format!("gemma3n.{s}");
        let hidden = meta_u32(g, &k("embedding_length"))?;
        let embd_altup = meta_u32(g, &k("embedding_length_per_layer_input")).unwrap_or(256);
        let n_heads = meta_u32(g, &k("attention.head_count"))?;
        let n_kv = meta_u32(g, &k("attention.head_count_kv"))?;
        let head_dim = meta_u32(g, &k("attention.key_length")).unwrap_or(256);
        let n_layers = meta_u32(g, &k("block_count"))?;
        let eps = g
            .meta_f32(&k("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);
        let rope_global = g.meta_f32(&k("rope.freq_base")).unwrap_or(1_000_000.0);
        let rope_local = g.meta_f32(&k("rope.freq_base_swa")).unwrap_or(10_000.0);
        let n_swa = meta_u32(g, &k("attention.sliding_window")).unwrap_or(512);
        // per-layer ffn array
        let ffn: Vec<usize> = g
            .meta(&k("feed_forward_length"))
            .and_then(|m| m.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f32().map(|x| x as usize))
                    .collect()
            })
            .filter(|v: &Vec<usize>| v.len() == n_layers)
            .unwrap_or_else(|| {
                vec![meta_u32(g, &k("feed_forward_length")).unwrap_or(16384); n_layers]
            });
        // sparsity scale array → count finite entries
        let n_sparsity = g
            .meta(&k("activation_sparsity_scale"))
            .and_then(|m| m.as_array())
            .map(|a| {
                a.iter()
                    .filter(|v| v.as_f32().map(|x| x.is_finite()).unwrap_or(false))
                    .count()
            })
            .unwrap_or(10);
        let vocab = g
            .tensors()
            .get("token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .map(|v| v as usize)
            .ok_or_else(|| StrixError::invalid("gemma3n: cannot determine vocab"))?;
        Ok(Gemma3nCfg {
            hidden,
            embd_altup,
            n_heads,
            n_kv,
            head_dim,
            n_layers,
            vocab,
            eps,
            rope_global,
            rope_local,
            n_swa,
            kv_from_start: 20,
            n_sparsity,
            sparsity_mul: 1.6448535919,
            ffn,
        })
    }
    fn is_swa(&self, il: usize) -> bool {
        (il + 1) % 5 != 0
    }
    pub fn report(&self) -> String {
        format!(
            "gemma3n-E: {}L hidden={} altup={}x{} heads={}/{} hd={} vocab={} sparsity0..{} kv_from={} swa={}",
            self.n_layers, self.hidden, N_ALTUP, self.embd_altup, self.n_heads, self.n_kv,
            self.head_dim, self.vocab, self.n_sparsity, self.kv_from_start, self.n_swa
        )
    }
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}
fn rmsnorm_nw(out: &mut [f32], x: &[f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale;
    }
}

fn rope_neox(vec: &mut [f32], pos: usize, n_dims: usize, base: f32) {
    let half = n_dims / 2;
    let ts = base.powf(-2.0 / n_dims as f32);
    let mut theta = pos as f32;
    for k in 0..half {
        let (s, c) = theta.sin_cos();
        let x0 = vec[k];
        let x1 = vec[k + half];
        vec[k] = x0 * c - x1 * s;
        vec[k + half] = x0 * s + x1 * c;
        theta *= ts;
    }
}

/// Quantized matmul out[o] = W_row_o · x, parallel over rows.
fn qmatmul(out: &mut [f32], x: &[f32], bytes: &[u8], ty: GgmlType, in_dim: usize) {
    let bpr = (in_dim / ty.block_elems()) * ty.block_bytes();
    out.par_iter_mut().enumerate().for_each_init(
        || vec![0.0f32; in_dim],
        |scratch, (o, oref)| {
            dequantize_into(ty, &bytes[o * bpr..o * bpr + bpr], scratch).unwrap();
            *oref = dot_f32(scratch, x);
        },
    );
}

/// Dense f32 matmul out[o] = W_row_o · x for a [in_dim, out_dim] f32 weight.
fn f32matmul(out: &mut [f32], x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) {
    for o in 0..out_dim {
        out[o] = dot_f32(&w[o * in_dim..o * in_dim + in_dim], x);
    }
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn l2(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Debug: print first-3 + last-3 of a vector (matches llama-eval-callback layout).
fn dbg3(name: &str, v: &[f32]) {
    let n = v.len();
    eprintln!(
        "[g3n] {name:<26} [{:.4}, {:.4}, {:.4} ... {:.4}, {:.4}, {:.4}] sum={:.4}",
        v[0],
        v[1],
        v[2],
        v[n - 3],
        v[n - 2],
        v[n - 1],
        v.iter().sum::<f32>()
    );
}

/// Per-layer small f32 weights.
struct Layer {
    attn_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffw_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    laurel_l: Vec<f32>, // [hidden, rank] in=hidden out=rank
    laurel_r: Vec<f32>, // [rank, hidden] in=rank out=hidden
    laurel_post_norm: Vec<f32>,
    altup_correct_coef: Vec<f32>,  // [4,4] in=4 out=4
    altup_correct_scale: Vec<f32>, // [hidden]
    altup_predict_coef: Vec<f32>,  // [4,16] in=4 out=16
    altup_router: Vec<f32>,        // [hidden,4] in=hidden out=4
    altup_router_norm: Vec<f32>,
    per_layer_inp_gate: Vec<f32>, // [hidden, embd_altup] in=hidden out=embd_altup
    per_layer_proj: Vec<f32>,     // [embd_altup, hidden] in=embd_altup out=hidden
    per_layer_post_norm: Vec<f32>, // [hidden]
}

pub struct Gemma3nModel {
    gguf: GgufFile,
    cfg: Gemma3nCfg,
    layers: Vec<Layer>,
    output_norm: Vec<f32>,
    per_layer_proj_norm: Vec<f32>, // [embd_altup]
    altup_proj: Vec<Vec<f32>>,     // 3 × [hidden,hidden] in=hidden out=hidden
    altup_unembd_proj: Vec<Vec<f32>>,
    // KV cache for the first kv_from_start layers
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    pos: usize,
    max_seq: usize,
    accel: Option<Box<dyn WeightAccel>>,
    /// True when the accelerator runs the whole MatFormer forward on-device.
    gpu_decode: bool,
    #[cfg(feature = "npu")]
    npu: Option<crate::mellum_npu::Gemma3nNpu>,
}

/// Which projection a batched matmul targets (NPU shape+slot selector).
#[derive(Clone, Copy)]
enum NpuW {
    Q,
    O,
    K,
    V,
    Gate,
    Up,
    Down,
    PlProj,
}

impl Gemma3nModel {
    pub fn from_gguf(gguf: GgufFile, max_seq: usize) -> Result<Self> {
        let cfg = Gemma3nCfg::from_gguf(&gguf)?;
        let f32v = |name: &str| -> Result<Vec<f32>> {
            gguf.dequant_tensor(name)
                .map_err(|e| StrixError::invalid(format!("gemma3n: {name}: {e}")))
        };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let b = |s: &str| format!("blk.{l}.{s}");
            layers.push(Layer {
                attn_norm: f32v(&b("attn_norm.weight"))?,
                post_attn_norm: f32v(&b("post_attention_norm.weight"))?,
                ffn_norm: f32v(&b("ffn_norm.weight"))?,
                post_ffw_norm: f32v(&b("post_ffw_norm.weight"))?,
                q_norm: f32v(&b("attn_q_norm.weight"))?,
                k_norm: f32v(&b("attn_k_norm.weight"))?,
                laurel_l: f32v(&b("laurel_l.weight"))?,
                laurel_r: f32v(&b("laurel_r.weight"))?,
                laurel_post_norm: f32v(&b("laurel_post_norm.weight"))?,
                altup_correct_coef: f32v(&b("altup_correct_coef.weight"))?,
                altup_correct_scale: f32v(&b("altup_correct_scale.weight"))?,
                altup_predict_coef: f32v(&b("altup_predict_coef.weight"))?,
                altup_router: f32v(&b("altup_router.weight"))?,
                altup_router_norm: f32v(&b("altup_router_norm.weight"))?,
                per_layer_inp_gate: f32v(&b("inp_gate.weight"))?,
                per_layer_proj: f32v(&b("proj.weight"))?,
                per_layer_post_norm: f32v(&b("post_norm.weight"))?,
            });
        }
        let output_norm = f32v("output_norm.weight")?;
        let per_layer_proj_norm = f32v("per_layer_proj_norm.weight")?;
        // altup_proj/unembd_proj are 3D [hidden,hidden,3]: split into 3 slices.
        let split3 = |name: &str| -> Result<Vec<Vec<f32>>> {
            let flat = f32v(name)?;
            let per = cfg.hidden * cfg.hidden;
            Ok((0..3)
                .map(|i| flat[i * per..(i + 1) * per].to_vec())
                .collect())
        };
        let altup_proj = split3("altup_proj.weight")?;
        let altup_unembd_proj = split3("altup_unembd_proj.weight")?;
        let kvd = cfg.n_kv * cfg.head_dim;
        let kc = (0..cfg.kv_from_start)
            .map(|_| Vec::with_capacity(kvd * max_seq))
            .collect();
        let vc = (0..cfg.kv_from_start)
            .map(|_| Vec::with_capacity(kvd * max_seq))
            .collect();
        Ok(Gemma3nModel {
            gguf,
            cfg,
            layers,
            output_norm,
            per_layer_proj_norm,
            altup_proj,
            altup_unembd_proj,
            kc,
            vc,
            pos: 0,
            max_seq,
            accel: None,
            gpu_decode: false,
            #[cfg(feature = "npu")]
            npu: None,
        })
    }

    /// Stage projections (q/o/k/v/gate/up/down per layer + per_layer_model_proj once)
    /// onto the NPU as int8. Prefill GEMMs run on the XDNA2 NPU (~2 W).
    #[cfg(feature = "npu")]
    pub fn attach_npu(&mut self, mut npu: crate::mellum_npu::Gemma3nNpu) -> Result<usize> {
        let mut n = 0;
        let mut st = |sh: &mut crate::mellum_npu::NpuShape, slot: u64, name: &str| -> Result<()> {
            let (bytes, ty, _, _) = self.wd(name)?;
            sh.stage_q8(slot, bytes, ty)?;
            n += 1;
            Ok(())
        };
        st(&mut npu.plproj, 0, "per_layer_model_proj.weight")?;
        for l in 0..self.cfg.n_layers {
            let b = |s: &str| format!("blk.{l}.{s}");
            let l = l as u64;
            st(&mut npu.qo, 2 * l, &b("attn_q.weight"))?;
            st(&mut npu.qo, 2 * l + 1, &b("attn_output.weight"))?;
            if (l as usize) < self.cfg.kv_from_start {
                st(&mut npu.kv, 2 * l, &b("attn_k.weight"))?;
                st(&mut npu.kv, 2 * l + 1, &b("attn_v.weight"))?;
            }
            st(&mut npu.gu, 2 * l, &b("ffn_gate.weight"))?;
            st(&mut npu.gu, 2 * l + 1, &b("ffn_up.weight"))?;
            st(&mut npu.down, l, &b("ffn_down.weight"))?;
        }
        self.npu = Some(npu);
        Ok(n)
    }

    /// Batched matmul out[m*n] = W·xs[m*k] by name; NPU shape (chunked M=256) if
    /// staged, else CPU dequant. `il` selects the layer slot (ignored for PlProj).
    #[allow(clippy::too_many_arguments)]
    fn bmm(&self, name: &str, which: NpuW, il: usize, xs: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) -> Result<()> {
        #[cfg(feature = "npu")]
        if let Some(npu) = &self.npu {
            let il = il as u64;
            let (sh, s) = match which {
                NpuW::Q => (&npu.qo, 2 * il),
                NpuW::O => (&npu.qo, 2 * il + 1),
                NpuW::K => (&npu.kv, 2 * il),
                NpuW::V => (&npu.kv, 2 * il + 1),
                NpuW::Gate => (&npu.gu, 2 * il),
                NpuW::Up => (&npu.gu, 2 * il + 1),
                NpuW::Down => (&npu.down, il),
                NpuW::PlProj => (&npu.plproj, 0),
            };
            if sh.k == k && sh.n == n && sh.has(s) {
                let mut okall = true;
                for c in (0..m).step_by(crate::mellum_npu::M_NPU) {
                    let mc = (m - c).min(crate::mellum_npu::M_NPU);
                    if sh.gemm(s, &xs[c * k..(c + mc) * k], mc, &mut out[c * n..(c + mc) * n]).is_err() {
                        okall = false;
                        break;
                    }
                }
                if okall {
                    return Ok(());
                }
            }
        }
        let _ = (which, il);
        let (bytes, ty, _, _) = self.wd(name)?;
        let bpr = (k / ty.block_elems()) * ty.block_bytes();
        let mut rt = vec![0.0f32; n * m];
        rt.par_chunks_mut(m).enumerate().for_each_init(
            || vec![0.0f32; k],
            |scratch, (o, orow)| {
                dequantize_into(ty, &bytes[o * bpr..o * bpr + bpr], scratch).unwrap();
                for t in 0..m {
                    orow[t] = dot_f32(scratch, &xs[t * k..(t + 1) * k]);
                }
            },
        );
        for t in 0..m {
            for o in 0..n {
                out[t * n + o] = rt[o * m + t];
            }
        }
        Ok(())
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Upload the quantized projection weights (q/k/v/o, ffn gate/up/down, tied
    /// lm_head, per_layer_model_proj) to the GPU; matmuls run via `gemv` with CPU
    /// fallback. AltUp/Laurel/PLE f32 mixing + norms/rope/attention stay CPU.
    pub fn attach_accel(&mut self, mut accel: Box<dyn WeightAccel>) -> usize {
        let mut names: Vec<String> = Vec::new();
        for l in 0..self.cfg.n_layers {
            for t in [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ] {
                names.push(format!("blk.{l}.{t}.weight"));
            }
        }
        names.push("token_embd.weight".to_string());
        names.push("per_layer_model_proj.weight".to_string());
        let mut n = 0;
        for name in &names {
            let Ok((bytes, ty, in_dim, out_dim)) = self.wd(name) else {
                continue;
            };
            let ok = match ty {
                GgmlType::Q4_0 => accel.upload_q4_0(name, bytes, in_dim, out_dim),
                GgmlType::Q4_1 => accel.upload_q4_1(name, bytes, in_dim, out_dim),
                GgmlType::Q6K => accel.upload_q6_k(name, bytes, in_dim, out_dim),
                GgmlType::Q8_0 => accel.upload_q8_0(name, bytes, in_dim, out_dim),
                _ => false,
            };
            if ok {
                n += 1;
            }
        }
        // Stage C (MatFormer resident decode): upload all the f32 AltUp/Laurel/PLE
        // tensors + describe the arch so decode runs the whole forward on-device.
        for (l, lay) in self.layers.iter().enumerate() {
            let b = |s: &str| format!("blk.{l}.{s}");
            accel.upload_f32(&b("attn_norm.weight"), &lay.attn_norm);
            accel.upload_f32(&b("post_attention_norm.weight"), &lay.post_attn_norm);
            accel.upload_f32(&b("ffn_norm.weight"), &lay.ffn_norm);
            accel.upload_f32(&b("post_ffw_norm.weight"), &lay.post_ffw_norm);
            accel.upload_f32(&b("attn_q_norm.weight"), &lay.q_norm);
            accel.upload_f32(&b("attn_k_norm.weight"), &lay.k_norm);
            accel.upload_f32(&b("laurel_l.weight"), &lay.laurel_l);
            accel.upload_f32(&b("laurel_r.weight"), &lay.laurel_r);
            accel.upload_f32(&b("laurel_post_norm.weight"), &lay.laurel_post_norm);
            accel.upload_f32(&b("altup_correct_coef.weight"), &lay.altup_correct_coef);
            accel.upload_f32(&b("altup_correct_scale.weight"), &lay.altup_correct_scale);
            accel.upload_f32(&b("altup_predict_coef.weight"), &lay.altup_predict_coef);
            accel.upload_f32(&b("altup_router.weight"), &lay.altup_router);
            accel.upload_f32(&b("altup_router_norm.weight"), &lay.altup_router_norm);
            accel.upload_f32(&b("per_layer_inp_gate.weight"), &lay.per_layer_inp_gate);
            accel.upload_f32(&b("per_layer_proj.weight"), &lay.per_layer_proj);
            accel.upload_f32(&b("per_layer_post_norm.weight"), &lay.per_layer_post_norm);
        }
        accel.upload_f32("output_norm.weight", &self.output_norm);
        accel.upload_f32("per_layer_proj_norm.weight", &self.per_layer_proj_norm);
        // per_layer_model_proj is F16 (not a Q-GEMM) — dequant to f32 for f32_gemv.
        if let Ok(w) = self.gguf.dequant_tensor("per_layer_model_proj.weight") {
            accel.upload_f32("per_layer_model_proj.weight", &w);
        }
        for (i, w) in self.altup_proj.iter().enumerate() {
            accel.upload_f32(&format!("altup_proj.{i}"), w);
        }
        for (i, w) in self.altup_unembd_proj.iter().enumerate() {
            accel.upload_f32(&format!("altup_unembd_proj.{i}"), w);
        }
        let cfg = &self.cfg;
        let gcfg = G3nConfig {
            hidden: cfg.hidden,
            embd_altup: cfg.embd_altup,
            n_heads: cfg.n_heads,
            n_kv: cfg.n_kv,
            head_dim: cfg.head_dim,
            n_layers: cfg.n_layers,
            vocab: cfg.vocab,
            eps: cfg.eps,
            rope_global: cfg.rope_global,
            rope_local: cfg.rope_local,
            n_swa: cfg.n_swa,
            kv_from_start: cfg.kv_from_start,
            n_sparsity: cfg.n_sparsity,
            sparsity_mul: cfg.sparsity_mul,
            laurel_rank: LAUREL_RANK,
            max_seq: self.max_seq,
            ffn: cfg.ffn.clone(),
        };
        self.gpu_decode =
            std::env::var("STRIX_GPU_HYBRID").is_err() && accel.configure_decode_g3n(gcfg);
        self.accel = Some(accel);
        n
    }

    /// On-device MatFormer decode of one token: compute the two scaled embeddings
    /// (base + per-layer) on the CPU, run the whole forward on the accelerator.
    fn gpu_decode_step(&mut self, token: u32) -> Result<Vec<f32>> {
        let (hidden, ea, nl) = (self.cfg.hidden, self.cfg.embd_altup, self.cfg.n_layers);
        let mut x0 = vec![0.0f32; hidden];
        self.embd_row("token_embd.weight", token as usize, hidden, &mut x0)?;
        let es = (hidden as f32).sqrt();
        for v in x0.iter_mut() {
            *v *= es;
        }
        let mut pe = vec![0.0f32; ea * nl];
        self.embd_row("per_layer_token_embd.weight", token as usize, ea * nl, &mut pe)?;
        let ps = (ea as f32).sqrt();
        for v in pe.iter_mut() {
            *v *= ps;
        }
        let pos = self.pos;
        // GPU forward returns the final normed hidden; lm_head (Q4_K token_embd) on CPU.
        let nfinal = self
            .accel
            .as_mut()
            .and_then(|a| a.decode_step_g3n(&x0, &pe, pos))
            .ok_or_else(|| StrixError::invalid("gemma3n gpu decode_step_g3n failed"))?;
        self.pos += 1;
        let head_name = if self.gguf.tensors().contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        self.mm(head_name, &nfinal)
    }

    /// Seed the device KV (own layers 0..kv_from_start) from the CPU/NPU prefill.
    fn seed_device_kv(&mut self) -> Result<()> {
        let Some(mut accel) = self.accel.take() else {
            return Ok(());
        };
        let mut ok = true;
        for il in 0..self.cfg.kv_from_start.min(self.kc.len()) {
            if !accel.seed_decode_kv_g3n(il, &self.kc[il], &self.vc[il]) {
                ok = false;
                break;
            }
        }
        self.accel = Some(accel);
        if ok {
            Ok(())
        } else {
            Err(StrixError::invalid("gemma3n: device KV seed failed"))
        }
    }

    fn w<'a>(&'a self, name: &str) -> Result<(&'a [u8], GgmlType, usize)> {
        let t = self
            .gguf
            .tensors()
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("gemma3n: missing tensor {name}")))?;
        Ok((
            self.gguf.tensor_bytes(name)?,
            t.ggml_type,
            t.dims[0] as usize,
        ))
    }

    /// Like `w` but also returns out_dim (dims[1]).
    fn wd<'a>(&'a self, name: &str) -> Result<(&'a [u8], GgmlType, usize, usize)> {
        let t = self
            .gguf
            .tensors()
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("gemma3n: missing tensor {name}")))?;
        let in_dim = t.dims[0] as usize;
        let out_dim = t.dims.get(1).copied().unwrap_or(1) as usize;
        Ok((self.gguf.tensor_bytes(name)?, t.ggml_type, in_dim, out_dim))
    }

    /// Matmul by tensor name: GPU `gemv` if resident, else CPU dequant.
    fn mm(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        if let Some(a) = &self.accel {
            if let Some(y) = a.gemv(name, x) {
                return Ok(y);
            }
        }
        let (bytes, ty, in_dim, out_dim) = self.wd(name)?;
        let mut y = vec![0.0f32; out_dim];
        qmatmul(&mut y, x, bytes, ty, in_dim);
        Ok(y)
    }

    /// dequant a single row `idx` of a quantized [in_dim, *] tensor.
    fn embd_row(&self, name: &str, idx: usize, in_dim: usize, out: &mut [f32]) -> Result<()> {
        let (bytes, ty, _) = self.w(name)?;
        let bpr = (in_dim / ty.block_elems()) * ty.block_bytes();
        dequantize_into(ty, &bytes[idx * bpr..idx * bpr + bpr], out)
            .map_err(|e| StrixError::invalid(format!("gemma3n embd_row {name}: {e}")))
    }

    /// router modalities: tanh(altup_router · (rmsnorm(x,router_norm) * 1/hidden)).
    fn modalities(&self, x: &[f32], il: usize) -> [f32; N_ALTUP] {
        let cfg = &self.cfg;
        let mut rn = vec![0.0f32; cfg.hidden];
        rmsnorm(&mut rn, x, &self.layers[il].altup_router_norm, cfg.eps);
        let sc = 1.0 / cfg.hidden as f32;
        for v in rn.iter_mut() {
            *v *= sc;
        }
        let mut out = [0.0f32; N_ALTUP];
        f32matmul(
            &mut out,
            &rn,
            &self.layers[il].altup_router,
            cfg.hidden,
            N_ALTUP,
        );
        for v in out.iter_mut() {
            *v = v.tanh();
        }
        out
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (hidden, hd, nh, nkv, ea) = (
            cfg.hidden,
            cfg.head_dim,
            cfg.n_heads,
            cfg.n_kv,
            cfg.embd_altup,
        );
        let groups = nh / nkv;
        let kv_dim = nkv * hd;
        let pos = self.pos;

        let dump = pos == 0 && std::env::var("STRIX_G3N_DUMP").is_ok();
        // base embedding (active stream) scaled by sqrt(hidden)
        let mut x0 = vec![0.0f32; hidden];
        self.embd_row("token_embd.weight", token as usize, hidden, &mut x0)?;
        if dump {
            dbg3("embd(raw)", &x0);
        }
        let es = (hidden as f32).sqrt();
        for v in x0.iter_mut() {
            *v *= es;
        }
        if dump {
            dbg3("inp_scaled", &x0);
        }

        // per-layer inputs: pe_tok[il] (256) from per_layer_token_embd row, scaled sqrt(256)
        // projected: per_layer_model_proj·x0 (8960), *1/sqrt(hidden), reshape [256,n_layer], rmsnorm per slice
        let nl = cfg.n_layers;
        let mut pe = vec![0.0f32; ea * nl];
        self.embd_row(
            "per_layer_token_embd.weight",
            token as usize,
            ea * nl,
            &mut pe,
        )?;
        let pscale = (ea as f32).sqrt();
        for v in pe.iter_mut() {
            *v *= pscale;
        }
        let mut proj = self.mm("per_layer_model_proj.weight", &x0)?;
        let pjs = 1.0 / (hidden as f32).sqrt();
        for v in proj.iter_mut() {
            *v *= pjs;
        }
        // inp_per_layer[il][0..ea] = (rmsnorm(proj_slice, per_layer_proj_norm) + pe_slice)/sqrt(2)
        let mut inp_per_layer = vec![0.0f32; ea * nl];
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        for il in 0..nl {
            let mut tmp = vec![0.0f32; ea];
            rmsnorm(
                &mut tmp,
                &proj[il * ea..il * ea + ea],
                &self.per_layer_proj_norm,
                cfg.eps,
            );
            for j in 0..ea {
                inp_per_layer[il * ea + j] = (tmp[j] + pe[il * ea + j]) * inv_sqrt2;
            }
        }
        if dump {
            dbg3("pe(raw)", &pe[..ea]);
            dbg3("inp_per_layer[0]", &inp_per_layer[..ea]);
        }

        // AltUp init: 4 streams. stream0 = x0, streams1..3 = magnitude-matched proj.
        let mut h: Vec<Vec<f32>> = Vec::with_capacity(N_ALTUP);
        h.push(x0.clone());
        let tgt = l2(&x0);
        for i in 0..N_ALTUP - 1 {
            let mut s = vec![0.0f32; hidden];
            f32matmul(&mut s, &x0, &self.altup_proj[i], hidden, hidden);
            let m = l2(&s).max(1e-12);
            let r = tgt / m;
            for v in s.iter_mut() {
                *v *= r;
            }
            h.push(s);
        }

        // scratch
        let mut q = vec![0.0f32; nh * hd];
        let mut kbuf = vec![0.0f32; kv_dim];
        let mut vbuf = vec![0.0f32; kv_dim];

        for il in 0..nl {
            let b = |s: &str| format!("blk.{il}.{s}");
            let lay = &self.layers[il];
            let swa = cfg.is_swa(il);
            let rope_base = if swa { cfg.rope_local } else { cfg.rope_global };

            // --- AltUp predict ---
            let modal = self.modalities(&h[I_ACT], il);
            // all_coefs[16] = predict_coef · modal ; coef(a,b)=all_coefs[a + 4*b]
            let mut all_coefs = vec![0.0f32; N_ALTUP * N_ALTUP];
            f32matmul(
                &mut all_coefs,
                &modal,
                &lay.altup_predict_coef,
                N_ALTUP,
                N_ALTUP * N_ALTUP,
            );
            // predictions[b][e] = h[b][e] + sum_a coef(a,b)*h[a][e]
            let mut pred: Vec<Vec<f32>> = (0..N_ALTUP).map(|b2| h[b2].clone()).collect();
            for b2 in 0..N_ALTUP {
                for a in 0..N_ALTUP {
                    let c = all_coefs[a + N_ALTUP * b2];
                    if c != 0.0 {
                        let ha = &h[a];
                        let pb = &mut pred[b2];
                        for e in 0..hidden {
                            pb[e] += c * ha[e];
                        }
                    }
                }
            }
            let active_pred = pred[I_ACT].clone();
            if dump && il == 0 {
                dbg3("active_prediction-0", &active_pred);
            }

            // --- attention on the predicted active stream ---
            let mut cur = vec![0.0f32; hidden];
            rmsnorm(&mut cur, &active_pred, &lay.attn_norm, cfg.eps);
            if dump && il == 0 {
                dbg3("attn_norm-0", &cur);
            }

            // laurel branch: r(l(cur)) -> norm -> + cur
            let mut lo = vec![0.0f32; LAUREL_RANK];
            f32matmul(&mut lo, &cur, &lay.laurel_l, hidden, LAUREL_RANK);
            let mut lr = vec![0.0f32; hidden];
            f32matmul(&mut lr, &lo, &lay.laurel_r, LAUREL_RANK, hidden);
            let mut ln = vec![0.0f32; hidden];
            rmsnorm(&mut ln, &lr, &lay.laurel_post_norm, cfg.eps);
            let mut laurel_out = vec![0.0f32; hidden];
            for e in 0..hidden {
                laurel_out[e] = ln[e] + cur[e];
            }
            // bisection: STRIX_G3N_NOLAUREL makes laurel a pass-through (=cur), so
            // attn_laurel = (attn_gated + cur)/sqrt2 — isolates the laurel branch.
            if std::env::var("STRIX_G3N_NOLAUREL").is_ok() {
                laurel_out.copy_from_slice(&cur);
            }

            // Q always computed
            q = self.mm(&b("attn_q.weight"), &cur)?;
            // per-head q-norm then rope
            for hh in 0..nh {
                let qh = &mut q[hh * hd..hh * hd + hd];
                let mut t = vec![0.0f32; hd];
                rmsnorm(&mut t, qh, &lay.q_norm, cfg.eps);
                qh.copy_from_slice(&t);
                rope_neox(qh, pos, hd, rope_base);
            }

            // KV: own (il<kv_from_start) or reuse source layer's cache
            let own_kv = il < cfg.kv_from_start;
            if own_kv {
                kbuf = self.mm(&b("attn_k.weight"), &cur)?;
                vbuf = self.mm(&b("attn_v.weight"), &cur)?;
                for kh in 0..nkv {
                    let kk = &mut kbuf[kh * hd..kh * hd + hd];
                    let mut t = vec![0.0f32; hd];
                    rmsnorm(&mut t, kk, &lay.k_norm, cfg.eps);
                    kk.copy_from_slice(&t);
                    rope_neox(kk, pos, hd, rope_base);
                    // V: plain rms-norm (no weight)
                    let vv = &mut vbuf[kh * hd..kh * hd + hd];
                    let mut tv2 = vec![0.0f32; hd];
                    rmsnorm_nw(&mut tv2, vv, cfg.eps);
                    vv.copy_from_slice(&tv2);
                }
                self.kc[il].extend_from_slice(&kbuf);
                self.vc[il].extend_from_slice(&vbuf);
            }
            let src = if own_kv {
                il
            } else if swa {
                cfg.kv_from_start - 2
            } else {
                cfg.kv_from_start - 1
            };
            let len = self.kc[src].len() / kv_dim;
            // sliding window start for swa layers
            let ws = if swa && cfg.n_swa > 0 && len > cfg.n_swa {
                len - cfg.n_swa
            } else {
                0
            };
            let kc = &self.kc[src];
            let vc = &self.vc[src];
            let scale = 1.0f32; // f_attention_scale = 1.0 for gemma3n (verified better than 1/sqrt)
            let mut attn = vec![0.0f32; nh * hd];
            attn.par_chunks_mut(hd).enumerate().for_each(|(hh, oh)| {
                let kvh = hh / groups;
                let qh = &q[hh * hd..hh * hd + hd];
                let mut sc = vec![0.0f32; len - ws];
                for (i, t) in (ws..len).enumerate() {
                    let kk = &kc[(t * nkv + kvh) * hd..(t * nkv + kvh) * hd + hd];
                    sc[i] = dot_f32(qh, kk) * scale;
                }
                let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                let mut sum = 0.0f32;
                for s in sc.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for x in oh.iter_mut() {
                    *x = 0.0;
                }
                for (i, t) in (ws..len).enumerate() {
                    let w = sc[i];
                    let vrow = &vc[(t * nkv + kvh) * hd..(t * nkv + kvh) * hd + hd];
                    for d in 0..hd {
                        oh[d] += w * vrow[d];
                    }
                }
                for x in oh.iter_mut() {
                    *x *= inv;
                }
            });
            // output proj
            let ao = self.mm(&b("attn_output.weight"), &attn)?;
            if dump && il == 0 {
                dbg3("attn_out-0", &ao);
                dbg3("laurel_out-0", &laurel_out);
            }
            // post-attn norm, gated residual with active_pred, then +laurel /sqrt2
            let mut pa = vec![0.0f32; hidden];
            rmsnorm(&mut pa, &ao, &lay.post_attn_norm, cfg.eps);
            for e in 0..hidden {
                pa[e] += active_pred[e];
            }
            let mut attn_laurel = vec![0.0f32; hidden];
            for e in 0..hidden {
                attn_laurel[e] = (pa[e] + laurel_out[e]) * inv_sqrt2;
            }
            if dump && il == 0 {
                dbg3("attn_gated-0", &pa);
                dbg3("attn_laurel-0", &attn_laurel);
            }

            // FFN
            let nff = cfg.ffn[il];
            let mut fn_in = vec![0.0f32; hidden];
            rmsnorm(&mut fn_in, &attn_laurel, &lay.ffn_norm, cfg.eps);
            let mut gate = self.mm(&b("ffn_gate.weight"), &fn_in)?;
            let up = self.mm(&b("ffn_up.weight"), &fn_in)?;
            if il < cfg.n_sparsity && std::env::var("STRIX_G3N_NOSPARSE").is_err() {
                // gaussian_topk: cutoff = mean + mul*std(ddof=1); relu(x-cutoff)
                let mean = gate.iter().sum::<f32>() / nff as f32;
                let var =
                    gate.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (nff as f32 - 1.0);
                let cutoff = mean + cfg.sparsity_mul * var.sqrt();
                for v in gate.iter_mut() {
                    *v = (*v - cutoff).max(0.0);
                }
            }
            for i in 0..nff {
                gate[i] = gelu_tanh(gate[i]) * up[i];
            }
            let ff = self.mm(&b("ffn_down.weight"), &gate)?;
            let mut pf = vec![0.0f32; hidden];
            rmsnorm(&mut pf, &ff, &lay.post_ffw_norm, cfg.eps);
            if dump && il == 0 {
                dbg3("ffn_out-0", &ff);
            }
            // attn_ffw_laurel_gated = pf + attn_laurel
            let mut afg = vec![0.0f32; hidden];
            for e in 0..hidden {
                afg[e] = pf[e] + attn_laurel[e];
            }
            if dump && il == 0 {
                dbg3("attn_ffw_laurel_gated-0", &afg);
            }

            // --- AltUp correct ---
            let modal2 = self.modalities(&afg, il);
            let mut cc = [0.0f32; N_ALTUP];
            f32matmul(&mut cc, &modal2, &lay.altup_correct_coef, N_ALTUP, N_ALTUP);
            for v in cc.iter_mut() {
                *v += 1.0;
            }
            // innovation = afg - pred[act]
            let mut innov = vec![0.0f32; hidden];
            for e in 0..hidden {
                innov[e] = afg[e] - pred[I_ACT][e];
            }
            // corrected[b] = pred[b] + innov*cc[b]
            let mut corrected: Vec<Vec<f32>> = (0..N_ALTUP).map(|b2| pred[b2].clone()).collect();
            for b2 in 0..N_ALTUP {
                let c = cc[b2];
                let cb = &mut corrected[b2];
                for e in 0..hidden {
                    cb[e] += innov[e] * c;
                }
            }

            // first_prediction → add to streams 1..3
            {
                let mut fp = vec![0.0f32; hidden];
                for e in 0..hidden {
                    fp[e] = corrected[I_ACT][e] * lay.altup_correct_scale[e];
                }
                let mut g = vec![0.0f32; ea];
                f32matmul(&mut g, &fp, &lay.per_layer_inp_gate, hidden, ea);
                for j in 0..ea {
                    g[j] = gelu_tanh(g[j]) * inp_per_layer[il * ea + j];
                }
                let mut pj = vec![0.0f32; hidden];
                f32matmul(&mut pj, &g, &lay.per_layer_proj, ea, hidden);
                let mut pn = vec![0.0f32; hidden];
                rmsnorm(&mut pn, &pj, &lay.per_layer_post_norm, cfg.eps);
                for b2 in 1..N_ALTUP {
                    let cb = &mut corrected[b2];
                    for e in 0..hidden {
                        cb[e] += pn[e];
                    }
                }
            }
            if dump && il == 0 {
                dbg3("l_out-0[stream0]", &corrected[0]);
                dbg3("l_out-0[stream1]", &corrected[1]);
                dbg3("l_out-0[stream3]", &corrected[3]);
            }
            h = corrected;
        }

        // merge altup streams back
        let tgt = l2(&h[I_ACT]);
        let mut merged = h[I_ACT].clone();
        for i in 0..N_ALTUP - 1 {
            let mut u = vec![0.0f32; hidden];
            f32matmul(
                &mut u,
                &h[i + 1],
                &self.altup_unembd_proj[i],
                hidden,
                hidden,
            );
            let m = l2(&u).max(1e-12);
            let r = tgt / m;
            for e in 0..hidden {
                merged[e] += u[e] * r;
            }
        }
        let inv_na = 1.0 / N_ALTUP as f32;
        for v in merged.iter_mut() {
            *v *= inv_na;
        }
        let mut nfinal = vec![0.0f32; hidden];
        rmsnorm(&mut nfinal, &merged, &self.output_norm, cfg.eps);

        // tied lm_head (token_embd). softcap omitted (monotone → greedy-invariant).
        let head_name = if self.gguf.tensors().contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let logits = self.mm(head_name, &nfinal)?;
        if std::env::var("STRIX_G3N_DUMP").is_ok() {
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in logits.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            eprintln!("[g3n] pos={pos} merged={:.4},{:.4},{:.4} result_norm={:.4},{:.4},{:.4} logit0..2={:.3},{:.3},{:.3} argmax={bi}",
                merged[0], merged[1], merged[2], nfinal[0], nfinal[1], nfinal[2],
                logits[0], logits[1], logits[2], bi=bi);
        }
        self.pos += 1;
        Ok(logits)
    }

    /// Batched prefill over `m` tokens. The 7 GEMMs/layer + per_layer_model_proj run
    /// on the NPU (~2 W); AltUp(4-stream)/Laurel/PLE/QK-norm/RoPE/attention stay CPU.
    /// Carries all m tokens' 4 AltUp streams through the layers; returns last logits.
    fn prefill_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (hidden, hd, nh, nkv, ea) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv, cfg.embd_altup);
        let (groups, kv_dim, nl) = (nh / nkv, nkv * hd, cfg.n_layers);
        let q_dim = nh * hd;
        let es = (hidden as f32).sqrt();
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        let m = tokens.len();

        // 4 AltUp streams, each [m][hidden].
        let mut h: Vec<Vec<f32>> = (0..N_ALTUP).map(|_| vec![0.0f32; m * hidden]).collect();
        let mut pe = vec![0.0f32; m * ea * nl];
        for (t, &tok) in tokens.iter().enumerate() {
            self.embd_row("token_embd.weight", tok as usize, hidden, &mut h[0][t * hidden..(t + 1) * hidden])?;
            for v in &mut h[0][t * hidden..(t + 1) * hidden] {
                *v *= es;
            }
            self.embd_row("per_layer_token_embd.weight", tok as usize, ea * nl, &mut pe[t * ea * nl..(t + 1) * ea * nl])?;
            let ps = (ea as f32).sqrt();
            for v in &mut pe[t * ea * nl..(t + 1) * ea * nl] {
                *v *= ps;
            }
        }
        // per-layer inputs: proj = per_layer_model_proj·x0 (batched NPU), then per token/layer.
        let mut proj = vec![0.0f32; m * ea * nl];
        self.bmm("per_layer_model_proj.weight", NpuW::PlProj, 0, &h[0], m, hidden, ea * nl, &mut proj)?;
        let pjs = 1.0 / (hidden as f32).sqrt();
        let mut inp_per_layer = vec![0.0f32; m * ea * nl];
        for t in 0..m {
            for il in 0..nl {
                let off = t * ea * nl + il * ea;
                let mut tmp = vec![0.0f32; ea];
                let mut sl = proj[off..off + ea].to_vec();
                for v in &mut sl {
                    *v *= pjs;
                }
                rmsnorm(&mut tmp, &sl, &self.per_layer_proj_norm, cfg.eps);
                for j in 0..ea {
                    inp_per_layer[off + j] = (tmp[j] + pe[off + j]) * inv_sqrt2;
                }
            }
        }
        // AltUp init: streams 1..3 = magnitude-matched altup_proj·x0 (per token).
        for t in 0..m {
            let x0 = h[0][t * hidden..(t + 1) * hidden].to_vec();
            let tgt = l2(&x0);
            for i in 0..N_ALTUP - 1 {
                let mut s = vec![0.0f32; hidden];
                f32matmul(&mut s, &x0, &self.altup_proj[i], hidden, hidden);
                let r = tgt / l2(&s).max(1e-12);
                for (e, sv) in s.iter().enumerate() {
                    h[i + 1][t * hidden + e] = sv * r;
                }
            }
        }

        for il in 0..nl {
            let bnm = |s: &str| format!("blk.{il}.{s}");
            let lay = &self.layers[il];
            let swa = cfg.is_swa(il);
            let rope_base = if swa { cfg.rope_local } else { cfg.rope_global };
            let own_kv = il < cfg.kv_from_start;

            // AltUp predict (per token) + attn-norm + laurel; collect normed cur into N.
            let mut pred: Vec<Vec<f32>> = (0..N_ALTUP).map(|b2| h[b2].clone()).collect();
            let mut active = vec![0.0f32; m * hidden];
            let mut cur_n = vec![0.0f32; m * hidden];
            let mut laurel_out = vec![0.0f32; m * hidden];
            for t in 0..m {
                let hr = |b: usize| &h[b][t * hidden..(t + 1) * hidden];
                let modal = self.modalities(hr(I_ACT), il);
                let mut coefs = vec![0.0f32; N_ALTUP * N_ALTUP];
                f32matmul(&mut coefs, &modal, &lay.altup_predict_coef, N_ALTUP, N_ALTUP * N_ALTUP);
                for b2 in 0..N_ALTUP {
                    for a in 0..N_ALTUP {
                        let c = coefs[a + N_ALTUP * b2];
                        if c != 0.0 {
                            for e in 0..hidden {
                                pred[b2][t * hidden + e] += c * h[a][t * hidden + e];
                            }
                        }
                    }
                }
                let ap = &pred[I_ACT][t * hidden..(t + 1) * hidden];
                active[t * hidden..(t + 1) * hidden].copy_from_slice(ap);
                let cn = &mut cur_n[t * hidden..(t + 1) * hidden];
                rmsnorm(cn, ap, &lay.attn_norm, cfg.eps);
                // laurel
                let mut lo = vec![0.0f32; LAUREL_RANK];
                f32matmul(&mut lo, cn, &lay.laurel_l, hidden, LAUREL_RANK);
                let mut lr = vec![0.0f32; hidden];
                f32matmul(&mut lr, &lo, &lay.laurel_r, LAUREL_RANK, hidden);
                let mut ln = vec![0.0f32; hidden];
                rmsnorm(&mut ln, &lr, &lay.laurel_post_norm, cfg.eps);
                for e in 0..hidden {
                    laurel_out[t * hidden + e] = ln[e] + cn[e];
                }
            }

            let mut q = vec![0.0f32; m * q_dim];
            self.bmm(&bnm("attn_q.weight"), NpuW::Q, il, &cur_n, m, hidden, q_dim, &mut q)?;
            if own_kv {
                let mut k = vec![0.0f32; m * kv_dim];
                let mut v = vec![0.0f32; m * kv_dim];
                self.bmm(&bnm("attn_k.weight"), NpuW::K, il, &cur_n, m, hidden, kv_dim, &mut k)?;
                self.bmm(&bnm("attn_v.weight"), NpuW::V, il, &cur_n, m, hidden, kv_dim, &mut v)?;
                for t in 0..m {
                    for kh in 0..nkv {
                        let kk = &mut k[t * kv_dim + kh * hd..t * kv_dim + kh * hd + hd];
                        let mut tt = vec![0.0f32; hd];
                        rmsnorm(&mut tt, kk, &lay.k_norm, cfg.eps);
                        kk.copy_from_slice(&tt);
                        rope_neox(kk, t, hd, rope_base);
                        let vv = &mut v[t * kv_dim + kh * hd..t * kv_dim + kh * hd + hd];
                        let mut tv = vec![0.0f32; hd];
                        rmsnorm_nw(&mut tv, vv, cfg.eps);
                        vv.copy_from_slice(&tv);
                    }
                }
                self.kc[il].extend_from_slice(&k);
                self.vc[il].extend_from_slice(&v);
            }
            for t in 0..m {
                for hh in 0..nh {
                    let qh = &mut q[t * q_dim + hh * hd..t * q_dim + hh * hd + hd];
                    let mut tt = vec![0.0f32; hd];
                    rmsnorm(&mut tt, qh, &lay.q_norm, cfg.eps);
                    qh.copy_from_slice(&tt);
                    rope_neox(qh, t, hd, rope_base);
                }
            }
            let src = if own_kv {
                il
            } else if swa {
                cfg.kv_from_start - 2
            } else {
                cfg.kv_from_start - 1
            };
            let kc = &self.kc[src];
            let vc = &self.vc[src];
            let mut attn = vec![0.0f32; m * q_dim];
            attn.par_chunks_mut(q_dim).enumerate().for_each(|(t, arow)| {
                let len = t + 1; // causal
                let ws = if swa && cfg.n_swa > 0 && len > cfg.n_swa { len - cfg.n_swa } else { 0 };
                for hh in 0..nh {
                    let kvh = hh / groups;
                    let qh = &q[t * q_dim + hh * hd..t * q_dim + hh * hd + hd];
                    let mut sc = vec![0.0f32; len - ws];
                    for (i, j) in (ws..len).enumerate() {
                        let kk = &kc[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                        sc[i] = dot_f32(qh, kk); // scale = 1.0
                    }
                    let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                    let mut sum = 0.0f32;
                    for s in sc.iter_mut() {
                        *s = (*s - mx).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    let oh = &mut arow[hh * hd..hh * hd + hd];
                    for x in oh.iter_mut() {
                        *x = 0.0;
                    }
                    for (i, j) in (ws..len).enumerate() {
                        let w = sc[i];
                        let vrow = &vc[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                        for d in 0..hd {
                            oh[d] += w * vrow[d];
                        }
                    }
                    for x in oh.iter_mut() {
                        *x *= inv;
                    }
                }
            });
            let mut o = vec![0.0f32; m * hidden];
            self.bmm(&bnm("attn_output.weight"), NpuW::O, il, &attn, m, q_dim, hidden, &mut o)?;

            // post-attn norm + gated residual + laurel, then ffn norm → FN
            let mut attn_laurel = vec![0.0f32; m * hidden];
            let mut fn_in = vec![0.0f32; m * hidden];
            for t in 0..m {
                let mut pa = vec![0.0f32; hidden];
                rmsnorm(&mut pa, &o[t * hidden..(t + 1) * hidden], &lay.post_attn_norm, cfg.eps);
                for e in 0..hidden {
                    pa[e] += active[t * hidden + e];
                    attn_laurel[t * hidden + e] = (pa[e] + laurel_out[t * hidden + e]) * inv_sqrt2;
                }
                rmsnorm(&mut fn_in[t * hidden..(t + 1) * hidden], &attn_laurel[t * hidden..(t + 1) * hidden], &lay.ffn_norm, cfg.eps);
            }
            let nff = cfg.ffn[il];
            let mut gate = vec![0.0f32; m * nff];
            let mut up = vec![0.0f32; m * nff];
            self.bmm(&bnm("ffn_gate.weight"), NpuW::Gate, il, &fn_in, m, hidden, nff, &mut gate)?;
            self.bmm(&bnm("ffn_up.weight"), NpuW::Up, il, &fn_in, m, hidden, nff, &mut up)?;
            for t in 0..m {
                let g = &mut gate[t * nff..(t + 1) * nff];
                if il < cfg.n_sparsity {
                    let mean = g.iter().sum::<f32>() / nff as f32;
                    let var = g.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (nff as f32 - 1.0);
                    let cutoff = mean + cfg.sparsity_mul * var.sqrt();
                    for v in g.iter_mut() {
                        *v = (*v - cutoff).max(0.0);
                    }
                }
                for i in 0..nff {
                    g[i] = gelu_tanh(g[i]) * up[t * nff + i];
                }
            }
            let mut ff = vec![0.0f32; m * hidden];
            self.bmm(&bnm("ffn_down.weight"), NpuW::Down, il, &gate, m, nff, hidden, &mut ff)?;

            // AltUp correct + first_prediction → next streams
            for t in 0..m {
                let mut pf = vec![0.0f32; hidden];
                rmsnorm(&mut pf, &ff[t * hidden..(t + 1) * hidden], &lay.post_ffw_norm, cfg.eps);
                let mut afg = vec![0.0f32; hidden];
                for e in 0..hidden {
                    afg[e] = pf[e] + attn_laurel[t * hidden + e];
                }
                let modal2 = self.modalities(&afg, il);
                let mut cc = [0.0f32; N_ALTUP];
                f32matmul(&mut cc, &modal2, &lay.altup_correct_coef, N_ALTUP, N_ALTUP);
                for v in cc.iter_mut() {
                    *v += 1.0;
                }
                let mut corr: Vec<Vec<f32>> = (0..N_ALTUP).map(|b2| pred[b2][t * hidden..(t + 1) * hidden].to_vec()).collect();
                for e in 0..hidden {
                    let innov = afg[e] - pred[I_ACT][t * hidden + e];
                    for b2 in 0..N_ALTUP {
                        corr[b2][e] += innov * cc[b2];
                    }
                }
                // first_prediction → streams 1..3
                let mut fp = vec![0.0f32; hidden];
                for e in 0..hidden {
                    fp[e] = corr[I_ACT][e] * lay.altup_correct_scale[e];
                }
                let mut g = vec![0.0f32; ea];
                f32matmul(&mut g, &fp, &lay.per_layer_inp_gate, hidden, ea);
                let off = t * ea * nl + il * ea;
                for j in 0..ea {
                    g[j] = gelu_tanh(g[j]) * inp_per_layer[off + j];
                }
                let mut pj = vec![0.0f32; hidden];
                f32matmul(&mut pj, &g, &lay.per_layer_proj, ea, hidden);
                let mut pn = vec![0.0f32; hidden];
                rmsnorm(&mut pn, &pj, &lay.per_layer_post_norm, cfg.eps);
                for b2 in 0..N_ALTUP {
                    for e in 0..hidden {
                        h[b2][t * hidden + e] = corr[b2][e] + if b2 >= 1 { pn[e] } else { 0.0 };
                    }
                }
            }
        }
        self.pos = m;
        // merge streams for the last token only → logits
        let t = m - 1;
        let act = h[I_ACT][t * hidden..(t + 1) * hidden].to_vec();
        let tgt = l2(&act);
        let mut merged = act;
        for i in 0..N_ALTUP - 1 {
            let mut u = vec![0.0f32; hidden];
            f32matmul(&mut u, &h[i + 1][t * hidden..(t + 1) * hidden], &self.altup_unembd_proj[i], hidden, hidden);
            let r = tgt / l2(&u).max(1e-12);
            for e in 0..hidden {
                merged[e] += u[e] * r;
            }
        }
        let inv_na = 1.0 / N_ALTUP as f32;
        for v in merged.iter_mut() {
            *v *= inv_na;
        }
        let mut nfinal = vec![0.0f32; hidden];
        rmsnorm(&mut nfinal, &merged, &self.output_norm, cfg.eps);
        let head_name = if self.gguf.tensors().contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        self.mm(head_name, &nfinal)
    }
}

impl Decoder for Gemma3nModel {
    fn prefill(&mut self, input_tokens: &[u32]) -> Result<Logits> {
        if input_tokens.is_empty() {
            return Err(StrixError::invalid("gemma3n: empty prompt"));
        }
        // Stage C: prefill stays OFF the iGPU (sustained GPU load crashes this box —
        // see never-gpu-prefill). prefill_batch's GEMMs run on CPU/NPU; then seed the
        // device KV (own layers) so the on-device MatFormer decode can attend the prompt.
        if self.gpu_decode {
            let last = self.prefill_batch(input_tokens)?;
            self.seed_device_kv()?;
            return Ok(Logits::new(last));
        }
        #[cfg(feature = "npu")]
        if self.npu.is_some() {
            return Ok(Logits::new(self.prefill_batch(input_tokens)?));
        }
        let mut last = Vec::new();
        for &t in input_tokens {
            last = self.forward(t)?;
        }
        Ok(Logits::new(last))
    }
    fn decode_one(&mut self, token: u32) -> Result<Logits> {
        if self.gpu_decode {
            return Ok(Logits::new(self.gpu_decode_step(token)?));
        }
        Ok(Logits::new(self.forward(token)?))
    }
    fn reset(&mut self) {
        self.pos = 0;
        for c in self.kc.iter_mut() {
            c.clear();
        }
        for c in self.vc.iter_mut() {
            c.clear();
        }
    }
}
