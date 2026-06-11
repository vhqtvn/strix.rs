//! Gemma-3n-E4B (`gemma3n`) — the MatFormer/effective-param architecture with
//! AltUp (4 parallel residual streams mixed by learned predict/correct), Laurel
//! (low-rank augmented residual), Per-Layer Embeddings (PLE), KV-sharing for the
//! tail layers, Gaussian-topk activation sparsity on early layers, and a
//! sliding-window attention pattern. Single-token CPU forward (n_tokens=1).
//!
//! Verified against refs/llama.cpp/src/models/gemma3n.cpp. Greedy decoding is
//! invariant to the (monotone) final logit softcap, so it is omitted.
//!
//! STATUS [bring-up]: loads + runs end-to-end and emits fluent English, but is
//! not yet answer-correct (a subtle scale/index bug remains in one of AltUp /
//! Laurel / PLE / KV-share). Needs a reference-activation diff against llama.cpp
//! to localize. SmolLM3 and Qwen3 (simpler arches) are fully validated.
//! Constants for E4B: 35 layers, n_embd=2048, n_altup=4, n_embd_altup=256,
//! laurel_rank=64, 8/2 heads hd=256, attn_scale=1.0, sparsity layers 0..10,
//! KV from layer 20 (full→19, swa→18), swa pattern (il+1)%5!=0, window=512,
//! rope 1e6 global / 1e4 local. Tied lm_head (token_embd). Tokenizer is
//! SentencePiece here but driven by raw IDs (STRIX_QWEN_IDS) for bring-up.

use rayon::prelude::*;
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
        })
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
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

        // base embedding (active stream) scaled by sqrt(hidden)
        let mut x0 = vec![0.0f32; hidden];
        self.embd_row("token_embd.weight", token as usize, hidden, &mut x0)?;
        let es = (hidden as f32).sqrt();
        for v in x0.iter_mut() {
            *v *= es;
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
        let mut proj = vec![0.0f32; ea * nl];
        {
            let (wb, wt, win) = self.w("per_layer_model_proj.weight")?;
            qmatmul(&mut proj, &x0, wb, wt, win);
        }
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

            // --- attention on the predicted active stream ---
            let mut cur = vec![0.0f32; hidden];
            rmsnorm(&mut cur, &active_pred, &lay.attn_norm, cfg.eps);

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
            {
                let (wq, tq, inq) = self.w(&b("attn_q.weight"))?;
                qmatmul(&mut q, &cur, wq, tq, inq);
            }
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
                let (wk, tk, ink) = self.w(&b("attn_k.weight"))?;
                qmatmul(&mut kbuf, &cur, wk, tk, ink);
                let (wv, tv, inv) = self.w(&b("attn_v.weight"))?;
                qmatmul(&mut vbuf, &cur, wv, tv, inv);
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
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for (i, t) in (ws..len).enumerate() {
                        acc += sc[i] * vc[(t * nkv + kvh) * hd + d];
                    }
                    oh[d] = acc * inv;
                }
            });
            // output proj
            let mut ao = vec![0.0f32; hidden];
            {
                let (wo, to, ino) = self.w(&b("attn_output.weight"))?;
                qmatmul(&mut ao, &attn, wo, to, ino);
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

            // FFN
            let nff = cfg.ffn[il];
            let mut fn_in = vec![0.0f32; hidden];
            rmsnorm(&mut fn_in, &attn_laurel, &lay.ffn_norm, cfg.eps);
            let mut gate = vec![0.0f32; nff];
            let mut up = vec![0.0f32; nff];
            {
                let (wg, tg, ing) = self.w(&b("ffn_gate.weight"))?;
                qmatmul(&mut gate, &fn_in, wg, tg, ing);
                let (wu, tu, inu) = self.w(&b("ffn_up.weight"))?;
                qmatmul(&mut up, &fn_in, wu, tu, inu);
            }
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
            let mut ff = vec![0.0f32; hidden];
            {
                let (wd, td, ind) = self.w(&b("ffn_down.weight"))?;
                qmatmul(&mut ff, &gate, wd, td, ind);
            }
            let mut pf = vec![0.0f32; hidden];
            rmsnorm(&mut pf, &ff, &lay.post_ffw_norm, cfg.eps);
            // attn_ffw_laurel_gated = pf + attn_laurel
            let mut afg = vec![0.0f32; hidden];
            for e in 0..hidden {
                afg[e] = pf[e] + attn_laurel[e];
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
        let (hw, ht, hin) = self.w(head_name)?;
        let mut logits = vec![0.0f32; cfg.vocab];
        qmatmul(&mut logits, &nfinal, hw, ht, hin);
        self.pos += 1;
        Ok(logits)
    }
}

impl Decoder for Gemma3nModel {
    fn prefill(&mut self, input_tokens: &[u32]) -> Result<Logits> {
        if input_tokens.is_empty() {
            return Err(StrixError::invalid("gemma3n: empty prompt"));
        }
        let mut last = Vec::new();
        for &t in input_tokens {
            last = self.forward(t)?;
        }
        Ok(Logits::new(last))
    }
    fn decode_one(&mut self, token: u32) -> Result<Logits> {
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
