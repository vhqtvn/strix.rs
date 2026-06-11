//! Qwen3-4B dense (`qwen3`) — Llama-style transformer with GQA, **per-head QK
//! RMSNorm** over head_dim applied to Q and K *before* RoPE, full RoPE on every
//! layer (no NoPE / no sliding window), no biases, no softcap, plain SwiGLU.
//! q_dim (n_head*head_dim) need not equal hidden. kq_scale = 1/sqrt(head_dim).
//! Tied embeddings (lm_head reuses token_embd unless `output.weight` present).
//!
//! Verified against refs/llama.cpp/src/models/qwen3.cpp. CPU on-the-fly dequant.
//! gpt2-BPE tokenizer → drive with raw token IDs via STRIX_QWEN_IDS.

use rayon::prelude::*;
use strix_core::accel::WeightAccel;
use strix_core::backend::Decoder;
use strix_core::error::{Result, StrixError};
use strix_core::sampler::Logits;
use strix_models::ggml_quant::{dequantize_into, GgmlType};
use strix_models::gguf::GgufFile;

fn meta_u32(g: &GgufFile, k: &str) -> Result<usize> {
    g.meta_u32(k).map(|v| v as usize)
}

pub struct Qwen3Cfg {
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
}

impl Qwen3Cfg {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g
            .architecture()
            .ok_or_else(|| StrixError::invalid("gguf: no general.architecture"))?;
        if arch != "qwen3" {
            return Err(StrixError::unsupported(format!(
                "qwen3 loader got `{arch}`"
            )));
        }
        let k = |s: &str| format!("qwen3.{s}");
        let hidden = meta_u32(g, &k("embedding_length"))?;
        let n_heads = meta_u32(g, &k("attention.head_count"))?;
        let n_kv = meta_u32(g, &k("attention.head_count_kv"))?;
        let ffn = meta_u32(g, &k("feed_forward_length"))?;
        let n_layers = meta_u32(g, &k("block_count"))?;
        let eps = g
            .meta_f32(&k("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);
        let rope_base = g.meta_f32(&k("rope.freq_base")).unwrap_or(1_000_000.0);
        let head_dim = meta_u32(g, &k("attention.key_length")).unwrap_or(hidden / n_heads);
        let vocab = g
            .tensors()
            .get("token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .map(|v| v as usize)
            .filter(|&v| v > 0)
            .ok_or_else(|| StrixError::invalid("qwen3: cannot determine vocab"))?;
        Ok(Qwen3Cfg {
            hidden,
            n_heads,
            n_kv,
            head_dim,
            ffn,
            n_layers,
            vocab,
            eps,
            rope_base,
        })
    }
    pub fn report(&self) -> String {
        format!(
            "qwen3: {}L hidden={} heads={}/{} hd={} ffn={} vocab={} rope={:.0e} QK-norm",
            self.n_layers,
            self.hidden,
            self.n_heads,
            self.n_kv,
            self.head_dim,
            self.ffn,
            self.vocab,
            self.rope_base
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

/// In-place per-head RMSNorm over `dim` (QK-norm). gain = `w`.
fn rmsnorm_inplace(v: &mut [f32], w: &[f32], eps: f32) {
    let n = v.len();
    let ss: f32 = v.iter().map(|x| x * x).sum();
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        v[i] = v[i] * scale * w[i];
    }
}

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
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

pub struct Qwen3Model {
    gguf: GgufFile,
    cfg: Qwen3Cfg,
    norms: Vec<LayerNorms>,
    output_norm: Vec<f32>,
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    pos: usize,
    max_seq: usize,
    accel: Option<Box<dyn WeightAccel>>,
    #[cfg(feature = "npu")]
    npu: Option<crate::mellum_npu::Qwen3Npu>,
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
}

impl Qwen3Model {
    pub fn from_gguf(gguf: GgufFile, max_seq: usize) -> Result<Self> {
        let cfg = Qwen3Cfg::from_gguf(&gguf)?;
        let f32v = |name: &str| -> Result<Vec<f32>> {
            gguf.dequant_tensor(name)
                .map_err(|e| StrixError::invalid(format!("qwen3: {name}: {e}")))
        };
        let mut norms = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            norms.push(LayerNorms {
                attn_norm: f32v(&format!("blk.{l}.attn_norm.weight"))?,
                ffn_norm: f32v(&format!("blk.{l}.ffn_norm.weight"))?,
                q_norm: f32v(&format!("blk.{l}.attn_q_norm.weight"))?,
                k_norm: f32v(&format!("blk.{l}.attn_k_norm.weight"))?,
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
        Ok(Qwen3Model {
            gguf,
            cfg,
            norms,
            output_norm,
            kc,
            vc,
            pos: 0,
            max_seq,
            accel: None,
            #[cfg(feature = "npu")]
            npu: None,
        })
    }

    /// Stage per-layer projections onto the NPU (int8 from Q4_0). Prefill GEMMs run
    /// on the XDNA2 NPU (~2 W).
    #[cfg(feature = "npu")]
    pub fn attach_npu(&mut self, mut npu: crate::mellum_npu::Qwen3Npu) -> Result<usize> {
        let mut n = 0;
        for l in 0..self.cfg.n_layers {
            let b = |s: &str| format!("blk.{l}.{s}");
            let mut st =
                |sh: &mut crate::mellum_npu::NpuShape, slot: u64, name: &str| -> Result<()> {
                    let (bytes, ty, _, _) = self.w(name)?;
                    sh.stage_q8(slot, bytes, ty)?;
                    n += 1;
                    Ok(())
                };
            let l = l as u64;
            st(&mut npu.q, l, &b("attn_q.weight"))?;
            st(&mut npu.kv, 2 * l, &b("attn_k.weight"))?;
            st(&mut npu.kv, 2 * l + 1, &b("attn_v.weight"))?;
            st(&mut npu.o, l, &b("attn_output.weight"))?;
            st(&mut npu.gu, 2 * l, &b("ffn_gate.weight"))?;
            st(&mut npu.gu, 2 * l + 1, &b("ffn_up.weight"))?;
            st(&mut npu.down, l, &b("ffn_down.weight"))?;
        }
        self.npu = Some(npu);
        Ok(n)
    }

    /// Batched matmul out[m*n] = W·xs[m*k] by name; NPU shape (chunked M=256) if
    /// staged, else CPU dequant.
    #[allow(clippy::too_many_arguments)]
    fn bmm(
        &self,
        name: &str,
        which: NpuW,
        xs: &[f32],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> Result<()> {
        #[cfg(feature = "npu")]
        if let Some(npu) = &self.npu {
            let il: u64 = name
                .split('.')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let (sh, s) = match which {
                NpuW::Q => (&npu.q, il),
                NpuW::O => (&npu.o, il),
                NpuW::K => (&npu.kv, 2 * il),
                NpuW::V => (&npu.kv, 2 * il + 1),
                NpuW::Gate => (&npu.gu, 2 * il),
                NpuW::Up => (&npu.gu, 2 * il + 1),
                NpuW::Down => (&npu.down, il),
            };
            if sh.k == k && sh.n == n && sh.has(s) {
                let mut okall = true;
                for c in (0..m).step_by(crate::mellum_npu::M_NPU) {
                    let mc = (m - c).min(crate::mellum_npu::M_NPU);
                    if sh
                        .gemm(
                            s,
                            &xs[c * k..(c + mc) * k],
                            mc,
                            &mut out[c * n..(c + mc) * n],
                        )
                        .is_err()
                    {
                        okall = false;
                        break;
                    }
                }
                if okall {
                    return Ok(());
                }
            }
        }
        let _ = which;
        let (bytes, ty, _, _) = self.w(name)?;
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

    /// Batched prefill over `m` tokens, GEMMs on NPU; QK-norm/RoPE/attention on CPU.
    fn prefill_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (hidden, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv);
        let (q_dim, kv_dim, groups, ffn) = (nh * hd, nkv * hd, nh / nkv, cfg.ffn);
        let scale = 1.0 / (hd as f32).sqrt();
        let m = tokens.len();
        let (eb, ety, ein, _) = self.w("token_embd.weight")?;
        let bpr = (ein / ety.block_elems()) * ety.block_bytes();
        let mut h = vec![0.0f32; m * hidden];
        for (t, &tok) in tokens.iter().enumerate() {
            dequantize_into(
                ety,
                &eb[tok as usize * bpr..tok as usize * bpr + bpr],
                &mut h[t * hidden..(t + 1) * hidden],
            )
            .map_err(|e| StrixError::invalid(format!("qwen3 embd: {e}")))?;
        }
        let mut nrm = vec![0.0f32; m * hidden];
        for il in 0..cfg.n_layers {
            let bnm = |s: &str| format!("blk.{il}.{s}");
            for t in 0..m {
                rmsnorm(
                    &mut nrm[t * hidden..(t + 1) * hidden],
                    &h[t * hidden..(t + 1) * hidden],
                    &self.norms[il].attn_norm,
                    cfg.eps,
                );
            }
            let mut q = vec![0.0f32; m * q_dim];
            let mut k = vec![0.0f32; m * kv_dim];
            let mut v = vec![0.0f32; m * kv_dim];
            self.bmm(
                &bnm("attn_q.weight"),
                NpuW::Q,
                &nrm,
                m,
                hidden,
                q_dim,
                &mut q,
            )?;
            self.bmm(
                &bnm("attn_k.weight"),
                NpuW::K,
                &nrm,
                m,
                hidden,
                kv_dim,
                &mut k,
            )?;
            self.bmm(
                &bnm("attn_v.weight"),
                NpuW::V,
                &nrm,
                m,
                hidden,
                kv_dim,
                &mut v,
            )?;
            for t in 0..m {
                for hh in 0..nh {
                    let qh = &mut q[t * q_dim + hh * hd..t * q_dim + hh * hd + hd];
                    rmsnorm_inplace(qh, &self.norms[il].q_norm, cfg.eps);
                    rope_neox(qh, t, hd, cfg.rope_base);
                }
                for kh in 0..nkv {
                    let kk = &mut k[t * kv_dim + kh * hd..t * kv_dim + kh * hd + hd];
                    rmsnorm_inplace(kk, &self.norms[il].k_norm, cfg.eps);
                    rope_neox(kk, t, hd, cfg.rope_base);
                }
            }
            self.kc[il].extend_from_slice(&k);
            self.vc[il].extend_from_slice(&v);
            let kc = &self.kc[il];
            let vc = &self.vc[il];
            let mut attn = vec![0.0f32; m * q_dim];
            attn.par_chunks_mut(q_dim)
                .enumerate()
                .for_each(|(t, arow)| {
                    let len = t + 1;
                    for hh in 0..nh {
                        let kvh = hh / groups;
                        let qh = &q[t * q_dim + hh * hd..t * q_dim + hh * hd + hd];
                        let mut sc = vec![0.0f32; len];
                        for j in 0..len {
                            let kk = &kc[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                            sc[j] = dot_f32(qh, kk) * scale;
                        }
                        let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
                        let mut sum = 0.0f32;
                        for s in sc.iter_mut() {
                            *s = (*s - mx).exp();
                            sum += *s;
                        }
                        let inv = 1.0 / sum;
                        let oh = &mut arow[hh * hd..hh * hd + hd];
                        for d in 0..hd {
                            let mut acc = 0.0f32;
                            for j in 0..len {
                                acc += sc[j] * vc[(j * nkv + kvh) * hd + d];
                            }
                            oh[d] = acc * inv;
                        }
                    }
                });
            let mut o = vec![0.0f32; m * hidden];
            self.bmm(
                &bnm("attn_output.weight"),
                NpuW::O,
                &attn,
                m,
                q_dim,
                hidden,
                &mut o,
            )?;
            for i in 0..m * hidden {
                h[i] += o[i];
            }
            for t in 0..m {
                rmsnorm(
                    &mut nrm[t * hidden..(t + 1) * hidden],
                    &h[t * hidden..(t + 1) * hidden],
                    &self.norms[il].ffn_norm,
                    cfg.eps,
                );
            }
            let mut gate = vec![0.0f32; m * ffn];
            let mut up = vec![0.0f32; m * ffn];
            self.bmm(
                &bnm("ffn_gate.weight"),
                NpuW::Gate,
                &nrm,
                m,
                hidden,
                ffn,
                &mut gate,
            )?;
            self.bmm(
                &bnm("ffn_up.weight"),
                NpuW::Up,
                &nrm,
                m,
                hidden,
                ffn,
                &mut up,
            )?;
            for i in 0..m * ffn {
                let g = gate[i];
                gate[i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            self.bmm(
                &bnm("ffn_down.weight"),
                NpuW::Down,
                &gate,
                m,
                ffn,
                hidden,
                &mut o,
            )?;
            for i in 0..m * hidden {
                h[i] += o[i];
            }
        }
        self.pos = m;
        let last = &h[(m - 1) * hidden..m * hidden];
        let mut nf = vec![0.0f32; hidden];
        rmsnorm(&mut nf, last, &self.output_norm, cfg.eps);
        let head_name = if self.gguf.tensors().contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        self.mm(head_name, &nf)
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Upload the big projection weights (q/k/v/o, ffn gate/up/down, tied lm_head)
    /// to the GPU accelerator. Per-token matmuls then run via `gemv`; norms, rope,
    /// QK-norm and attention stay on the CPU. Returns the number of weights staged.
    ///
    /// PERF [measured 2026-06-11, lossless]: ~6-7x decode speedup — Qwen3 2.3→16 tok/s,
    /// SmolLM3 2.8→18 tok/s (at a throttled 1079MHz sclk; higher at full 2900MHz).
    /// (An earlier "0 speedup" reading was a collapsed-GPU-clock artifact, not real.)
    /// Still per-`gemv` round-trips (upload+sync+download/matmul); a resident
    /// on-device forward (hipGraph, dense `mlm_token_graph` analog) would go further.
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
        let mut n = 0;
        for name in &names {
            let Ok((bytes, ty, in_dim, out_dim)) = self.w(name) else {
                continue;
            };
            let ok = match ty {
                GgmlType::Q4_0 => accel.upload_q4_0(name, bytes, in_dim, out_dim),
                GgmlType::Q6K => accel.upload_q6_k(name, bytes, in_dim, out_dim),
                GgmlType::Q8_0 => accel.upload_q8_0(name, bytes, in_dim, out_dim),
                _ => false,
            };
            if ok {
                n += 1;
            }
        }
        self.accel = Some(accel);
        n
    }

    fn w<'a>(&'a self, name: &str) -> Result<(&'a [u8], GgmlType, usize, usize)> {
        let t = self
            .gguf
            .tensors()
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("qwen3: missing tensor {name}")))?;
        let in_dim = t.dims[0] as usize;
        let out_dim = t.dims.get(1).copied().unwrap_or(1) as usize;
        Ok((self.gguf.tensor_bytes(name)?, t.ggml_type, in_dim, out_dim))
    }

    /// Matmul by tensor name: GPU `gemv` if the weight is resident, else CPU dequant.
    fn mm(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        if let Some(a) = &self.accel {
            if let Some(y) = a.gemv(name, x) {
                return Ok(y);
            }
        }
        let (bytes, ty, in_dim, out_dim) = self.w(name)?;
        let mut y = vec![0.0f32; out_dim];
        qmatmul(&mut y, x, bytes, ty, in_dim);
        Ok(y)
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (hidden, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv);
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let groups = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let pos = self.pos;

        let (eb, ety, ein, _) = self.w("token_embd.weight")?;
        let bpr = (ein / ety.block_elems()) * ety.block_bytes();
        let mut h = vec![0.0f32; hidden];
        dequantize_into(
            ety,
            &eb[token as usize * bpr..token as usize * bpr + bpr],
            &mut h,
        )
        .map_err(|e| StrixError::invalid(format!("qwen3 embd: {e}")))?;

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
            rmsnorm(&mut n, &h, &self.norms[il].attn_norm, cfg.eps);
            q = self.mm(&b("attn_q.weight"), &n)?;
            k = self.mm(&b("attn_k.weight"), &n)?;
            v = self.mm(&b("attn_v.weight"), &n)?;

            // per-head QK-norm (over head_dim) THEN rope
            for hh in 0..nh {
                let qh = &mut q[hh * hd..hh * hd + hd];
                rmsnorm_inplace(qh, &self.norms[il].q_norm, cfg.eps);
                rope_neox(qh, pos, hd, cfg.rope_base);
            }
            for kh in 0..nkv {
                let kk = &mut k[kh * hd..kh * hd + hd];
                rmsnorm_inplace(kk, &self.norms[il].k_norm, cfg.eps);
                rope_neox(kk, pos, hd, cfg.rope_base);
            }

            self.kc[il].extend_from_slice(&k);
            self.vc[il].extend_from_slice(&v);
            let len = pos + 1;
            let kc = &self.kc[il];
            let vc = &self.vc[il];

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

            o = self.mm(&b("attn_output.weight"), &attn)?;
            for i in 0..hidden {
                h[i] += o[i];
            }

            rmsnorm(&mut n, &h, &self.norms[il].ffn_norm, cfg.eps);
            gate = self.mm(&b("ffn_gate.weight"), &n)?;
            up = self.mm(&b("ffn_up.weight"), &n)?;
            for i in 0..cfg.ffn {
                let g = gate[i];
                gate[i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            o = self.mm(&b("ffn_down.weight"), &gate)?;
            for i in 0..hidden {
                h[i] += o[i];
            }
        }

        rmsnorm(&mut n, &h, &self.output_norm, cfg.eps);
        let head_name = if self.gguf.tensors().contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let logits = self.mm(head_name, &n)?;
        self.pos += 1;
        Ok(logits)
    }
}

impl Decoder for Qwen3Model {
    fn prefill(&mut self, input_tokens: &[u32]) -> Result<Logits> {
        if input_tokens.is_empty() {
            return Err(StrixError::invalid("qwen3: empty prompt"));
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
