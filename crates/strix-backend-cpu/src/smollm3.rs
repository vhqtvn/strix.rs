//! SmolLM3-3B (`smollm3`) — a Llama-architecture transformer with GQA, tied
//! embeddings, and **NoPE**: RoPE is skipped on every 4th layer (the layers `il`
//! where `(il + 1) % 4 == 0`). No QK-norm, no biases, no logit softcap, plain
//! SwiGLU FFN. kq_scale = 1/sqrt(head_dim).
//!
//! Verified against refs/llama.cpp/src/models/smollm3.cpp. CPU-only on-the-fly
//! dequant forward (the WMMA/int8 GPU path is mellum-specific and not wired here).
//! Tokenizer is gpt2-BPE (StrixTokenizer is SentencePiece-only) → drive with raw
//! token IDs via STRIX_QWEN_IDS, like mellum.

use rayon::prelude::*;
use strix_core::backend::Decoder;
use strix_core::error::{Result, StrixError};
use strix_core::sampler::Logits;
use strix_models::ggml_quant::{dequantize_into, GgmlType};
use strix_models::gguf::GgufFile;

fn meta_u32(g: &GgufFile, k: &str) -> Result<usize> {
    g.meta_u32(k).map(|v| v as usize)
}
fn meta_f32_or(g: &GgufFile, k: &str, d: f32) -> f32 {
    g.meta_f32(k).unwrap_or(d)
}

pub struct SmolLm3Cfg {
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub nope_step: usize,
}

impl SmolLm3Cfg {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g
            .architecture()
            .ok_or_else(|| StrixError::invalid("gguf: no general.architecture"))?;
        if arch != "smollm3" {
            return Err(StrixError::unsupported(format!(
                "smollm3 loader got `{arch}`"
            )));
        }
        let k = |s: &str| format!("smollm3.{s}");
        let hidden = meta_u32(g, &k("embedding_length"))?;
        let n_heads = meta_u32(g, &k("attention.head_count"))?;
        let n_kv = meta_u32(g, &k("attention.head_count_kv"))?;
        let ffn = meta_u32(g, &k("feed_forward_length"))?;
        let n_layers = meta_u32(g, &k("block_count"))?;
        let eps = meta_f32_or(g, &k("attention.layer_norm_rms_epsilon"), 1e-6);
        let rope_base = meta_f32_or(g, &k("rope.freq_base"), 5_000_000.0);
        let head_dim = meta_u32(g, &k("rope.dimension_count")).unwrap_or(hidden / n_heads);
        let vocab = g
            .tensors()
            .get("token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .map(|v| v as usize)
            .filter(|&v| v > 0)
            .ok_or_else(|| StrixError::invalid("smollm3: cannot determine vocab"))?;
        Ok(SmolLm3Cfg {
            hidden,
            n_heads,
            n_kv,
            head_dim,
            ffn,
            n_layers,
            vocab,
            eps,
            rope_base,
            nope_step: 4,
        })
    }
    pub fn report(&self) -> String {
        format!(
            "smollm3: {}L hidden={} heads={}/{} hd={} ffn={} vocab={} rope={:.0e} NoPE@every-{}",
            self.n_layers,
            self.hidden,
            self.n_heads,
            self.n_kv,
            self.head_dim,
            self.ffn,
            self.vocab,
            self.rope_base,
            self.nope_step
        )
    }
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0.0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let i = c * 8;
        for k in 0..8 {
            acc[k] += a[i + k] * b[i + k];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in (chunks * 8)..n {
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

/// NEOX RoPE on a head vector (plain: freq_scale=1, no yarn).
fn rope_neox(vec: &mut [f32], pos: usize, n_dims: usize, freq_base: f32) {
    let half = n_dims / 2;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let mut theta = pos as f32;
    for k in 0..half {
        let (s, c) = theta.sin_cos();
        let x0 = vec[k];
        let x1 = vec[k + half];
        vec[k] = x0 * c - x1 * s;
        vec[k + half] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

/// out[o] = dequant(W row o) · x, parallel over rows. in_dim = row length.
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

struct LayerNorms {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
}

pub struct SmolLm3Model {
    gguf: GgufFile,
    cfg: SmolLm3Cfg,
    norms: Vec<LayerNorms>,
    output_norm: Vec<f32>,
    // KV cache, per layer: [n_kv * max_seq * head_dim]
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    pos: usize,
    max_seq: usize,
}

impl SmolLm3Model {
    pub fn from_gguf(gguf: GgufFile, max_seq: usize) -> Result<Self> {
        let cfg = SmolLm3Cfg::from_gguf(&gguf)?;
        let f32v = |name: &str| -> Result<Vec<f32>> {
            gguf.dequant_tensor(name)
                .map_err(|e| StrixError::invalid(format!("smollm3: {name}: {e}")))
        };
        let mut norms = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            norms.push(LayerNorms {
                attn_norm: f32v(&format!("blk.{l}.attn_norm.weight"))?,
                ffn_norm: f32v(&format!("blk.{l}.ffn_norm.weight"))?,
            });
        }
        let output_norm = f32v("output_norm.weight")?;
        let kvd = cfg.n_kv * cfg.head_dim;
        let kc = (0..cfg.n_layers)
            .map(|_| Vec::with_capacity(kvd * max_seq))
            .collect();
        let vc = (0..cfg.n_layers)
            .map(|_| Vec::with_capacity(kvd * max_seq))
            .collect();
        Ok(SmolLm3Model {
            gguf,
            cfg,
            norms,
            output_norm,
            kc,
            vc,
            pos: 0,
            max_seq,
        })
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    fn w<'a>(&'a self, name: &str) -> Result<(&'a [u8], GgmlType, usize, usize)> {
        let t = self
            .gguf
            .tensors()
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("smollm3: missing tensor {name}")))?;
        let bytes = self.gguf.tensor_bytes(name)?;
        let in_dim = t.dims[0] as usize;
        let out_dim = t.dims.get(1).copied().unwrap_or(1) as usize;
        Ok((bytes, t.ggml_type, in_dim, out_dim))
    }

    /// One-token forward. Returns logits.
    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (hidden, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv);
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let groups = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let pos = self.pos;

        // embedding: row `token` of token_embd (in_dim = hidden)
        let (eb, ety, ein, _) = self.w("token_embd.weight")?;
        let bpr = (ein / ety.block_elems()) * ety.block_bytes();
        let mut h = vec![0.0f32; hidden];
        dequantize_into(
            ety,
            &eb[token as usize * bpr..token as usize * bpr + bpr],
            &mut h,
        )
        .map_err(|e| StrixError::invalid(format!("smollm3 embd: {e}")))?;

        let mut n = vec![0.0f32; hidden];
        let mut q = vec![0.0f32; q_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn = vec![0.0f32; q_dim];
        let mut o = vec![0.0f32; hidden];
        let mut gate = vec![0.0f32; cfg.ffn];
        let mut up = vec![0.0f32; cfg.ffn];

        for il in 0..cfg.n_layers {
            let b = |s: &str| format!("blk.{il}.{s}");
            // attn norm
            rmsnorm(&mut n, &h, &self.norms[il].attn_norm, cfg.eps);
            // q/k/v
            let (wq, tq, inq, _) = self.w(&b("attn_q.weight"))?;
            qmatmul(&mut q, &n, wq, tq, inq);
            let (wk, tk, ink, _) = self.w(&b("attn_k.weight"))?;
            qmatmul(&mut k, &n, wk, tk, ink);
            let (wv, tv, inv, _) = self.w(&b("attn_v.weight"))?;
            qmatmul(&mut v, &n, wv, tv, inv);

            // NoPE: skip rope on layers where (il+1) % step == 0
            let use_rope = (il + 1) % cfg.nope_step != 0;
            if use_rope {
                for hh in 0..nh {
                    rope_neox(&mut q[hh * hd..hh * hd + hd], pos, hd, cfg.rope_base);
                }
                for kh in 0..nkv {
                    rope_neox(&mut k[kh * hd..kh * hd + hd], pos, hd, cfg.rope_base);
                }
            }

            // append to KV cache
            self.kc[il].extend_from_slice(&k);
            self.vc[il].extend_from_slice(&v);
            let len = pos + 1;
            let kc = &self.kc[il];
            let vc = &self.vc[il];

            // GQA attention per q-head
            attn.par_chunks_mut(hd).enumerate().for_each(|(hh, oh)| {
                let kvh = hh / groups;
                let qh = &q[hh * hd..hh * hd + hd];
                let mut scores = vec![0.0f32; len];
                for t in 0..len {
                    let kk = &kc[(t * nkv + kvh) * hd..(t * nkv + kvh) * hd + hd];
                    scores[t] = dot_f32(qh, kk) * scale;
                }
                let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for t in 0..len {
                        acc += scores[t] * vc[(t * nkv + kvh) * hd + d];
                    }
                    oh[d] = acc * inv;
                }
            });

            // output proj + residual
            let (wo, to, ino, _) = self.w(&b("attn_output.weight"))?;
            qmatmul(&mut o, &attn, wo, to, ino);
            for i in 0..hidden {
                h[i] += o[i];
            }

            // ffn norm + SwiGLU
            rmsnorm(&mut n, &h, &self.norms[il].ffn_norm, cfg.eps);
            let (wg, tg, ing, _) = self.w(&b("ffn_gate.weight"))?;
            qmatmul(&mut gate, &n, wg, tg, ing);
            let (wu, tu, inu, _) = self.w(&b("ffn_up.weight"))?;
            qmatmul(&mut up, &n, wu, tu, inu);
            for i in 0..cfg.ffn {
                let g = gate[i];
                gate[i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            let (wd, td, ind, _) = self.w(&b("ffn_down.weight"))?;
            qmatmul(&mut o, &gate, wd, td, ind);
            for i in 0..hidden {
                h[i] += o[i];
            }
        }

        // final norm + lm_head (tied to token_embd)
        rmsnorm(&mut n, &h, &self.output_norm, cfg.eps);
        let head_name = if self.gguf.tensors().contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (hw, ht, hin, _) = self.w(head_name)?;
        let mut logits = vec![0.0f32; cfg.vocab];
        qmatmul(&mut logits, &n, hw, ht, hin);
        self.pos += 1;
        Ok(logits)
    }
}

impl Decoder for SmolLm3Model {
    fn prefill(&mut self, input_tokens: &[u32]) -> Result<Logits> {
        if input_tokens.is_empty() {
            return Err(StrixError::invalid("smollm3: empty prompt"));
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
