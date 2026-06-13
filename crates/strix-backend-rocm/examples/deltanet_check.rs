//! Standalone numerical validation of the GPU gated-delta-net decode kernels against
//! a CPU reference mirroring `qwen35.rs::deltanet()` (the compute after the qkv/gate/
//! beta/alpha projections: conv1d → L2norm → decay/beta → scan → gated-RMSNorm).
//!
//! Run: cargo run --release -p strix-backend-rocm --features rocm --example deltanet_check

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("build with --features rocm");
}

#[cfg(feature = "rocm")]
const SRC: &str = r#"
// Causal depthwise conv1d (kernel dconv) over qkv[conv_dim] using conv state
// cs[conv_dim*(dconv-1)] (oldest..newest), + silu, then shift state. grid=ceil(conv_dim/256).
extern "C" __global__ void dn_conv1d(const float* __restrict__ qkv, const float* __restrict__ cw,
                                     float* __restrict__ cs, float* __restrict__ out,
                                     int conv_dim, int dconv) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const float* w = cw + (long long)c * dconv;
    float* st = cs + (long long)c * (dconv - 1);
    float acc = 0.f;
    for (int k = 0; k < dconv - 1; k++) acc += w[k] * st[k];
    acc += w[dconv - 1] * qkv[c];
    out[c] = acc / (1.f + __expf(-acc));   // silu
    for (int k = 0; k < dconv - 2; k++) st[k] = st[k + 1];
    st[dconv - 2] = qkv[c];
}

// L2-normalize q and k per k-head segment [s_v]: x / sqrt(max(sum x^2, eps)).
// grid = 2*n_kh (first n_kh = q segments, rest = k), block = s_v.
extern "C" __global__ void dn_l2norm(const float* __restrict__ q, const float* __restrict__ k,
                                     float* __restrict__ qn, float* __restrict__ kn,
                                     int s_v, int n_kh, float eps) {
    int seg = blockIdx.x, i = threadIdx.x;
    const float* src; float* dst; int kh;
    if (seg < n_kh) { kh = seg;        src = q + (long long)kh * s_v; dst = qn + (long long)kh * s_v; }
    else            { kh = seg - n_kh; src = k + (long long)kh * s_v; dst = kn + (long long)kh * s_v; }
    __shared__ float red[1024];
    float val = (i < s_v) ? src[i] : 0.f;
    red[i] = val * val; __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) { if (i < o) red[i] += red[i + o]; __syncthreads(); }
    float r = rsqrtf(fmaxf(red[0], eps));
    if (i < s_v) dst[i] = val * r;
}

// decay[vh] = exp(ssm_a[vh] * softplus(alpha_raw[vh] + ssm_dt[vh])); beta[vh] = sigmoid(beta_raw[vh]).
extern "C" __global__ void dn_decaybeta(const float* __restrict__ alpha_raw, const float* __restrict__ beta_raw,
                                        const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
                                        float* __restrict__ decay, float* __restrict__ beta, int n_vh) {
    int vh = blockIdx.x * blockDim.x + threadIdx.x;
    if (vh >= n_vh) return;
    float x = alpha_raw[vh] + ssm_dt[vh];
    float sp = (x > 20.f) ? x : log1pf(__expf(x));
    decay[vh] = __expf(ssm_a[vh] * sp);
    beta[vh] = 1.f / (1.f + __expf(-beta_raw[vh]));
}

// Gated DeltaNet single-token recurrent update. grid = n_vh, block = s_v (thread per
// state row j). State st[vh*s_v*s_v + j*s_v + i] = S[i][j]. kh = vh % n_kh.
extern "C" __global__ void deltanet_step(
    float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ decay,
    const float* __restrict__ beta, float* __restrict__ out, int s_v, int n_kh, float scale) {
    int vh = blockIdx.x, j = threadIdx.x;
    if (j >= s_v) return;
    int kh = vh % n_kh;
    extern __shared__ float sh[];
    float* qsh = sh; float* ksh = sh + s_v;
    for (int i = j; i < s_v; i += blockDim.x) { qsh[i] = q[kh*s_v+i]; ksh[i] = k[kh*s_v+i]; }
    __syncthreads();
    float dec = decay[vh], bt = beta[vh];
    float* base = state + (long long)vh * s_v * s_v;   // TRANSPOSED: S[i][j] at base[i*s_v + j]
    float dotk = 0.f;
    for (int i = 0; i < s_v; i++) { float s = base[i*s_v + j] * dec; base[i*s_v + j] = s; dotk += s * ksh[i]; }
    float delta = (v[vh*s_v + j] - dotk) * bt;
    float dotq = 0.f;
    for (int i = 0; i < s_v; i++) { float s = base[i*s_v + j] + delta * ksh[i]; base[i*s_v + j] = s; dotq += s * qsh[i]; }
    out[vh*s_v + j] = dotq * scale;
}

// Per v-head: rmsnorm(core_h, w) * silu(z_h). grid = n_vh, block = s_v.
extern "C" __global__ void dn_gatednorm(const float* __restrict__ core, const float* __restrict__ z,
                                        const float* __restrict__ w, float* __restrict__ out,
                                        int s_v, float eps) {
    int vh = blockIdx.x, i = threadIdx.x;
    const float* c = core + (long long)vh * s_v;
    __shared__ float red[1024];
    float val = (i < s_v) ? c[i] : 0.f;
    red[i] = val * val; __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) { if (i < o) red[i] += red[i + o]; __syncthreads(); }
    float r = rsqrtf(red[0] / s_v + eps);
    if (i < s_v) {
        float zz = z[(long long)vh * s_v + i];
        out[(long long)vh * s_v + i] = (val * r * w[i]) * (zz / (1.f + __expf(-zz)));
    }
}
"#;

#[cfg(feature = "rocm")]
fn main() {
    use std::ffi::c_void;
    use strix_backend_rocm::hip::HipGpu;

    let (n_vh, n_kh, s_v) = (32usize, 16usize, 128usize);
    let dconv = 4usize;
    let eps = 1e-6f32;
    let scale = 1.0f32 / (s_v as f32).sqrt();
    let key_dim = n_kh * s_v; // 2048
    let value_dim = n_vh * s_v; // 4096
    let conv_dim = key_dim * 2 + value_dim; // 8192
    let st_len = n_vh * s_v * s_v;

    let mut seed = 0x12345678u64;
    let mut rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let qkv: Vec<f32> = (0..conv_dim).map(|_| rng()).collect();
    let z: Vec<f32> = (0..value_dim).map(|_| rng()).collect();
    let beta_raw: Vec<f32> = (0..n_vh).map(|_| rng()).collect();
    let alpha_raw: Vec<f32> = (0..n_vh).map(|_| rng()).collect();
    let conv_w: Vec<f32> = (0..conv_dim * dconv).map(|_| rng() * 0.5).collect();
    let conv0: Vec<f32> = (0..conv_dim * (dconv - 1)).map(|_| rng() * 0.1).collect();
    let state0: Vec<f32> = (0..st_len).map(|_| rng() * 0.1).collect();
    let ssm_a: Vec<f32> = (0..n_vh).map(|_| -(0.5 + rng().abs())).collect();
    let ssm_dt: Vec<f32> = (0..n_vh).map(|_| rng() * 0.5).collect();
    let ssm_norm: Vec<f32> = (0..s_v).map(|_| 0.8 + 0.4 * rng().abs()).collect();

    // ---------- CPU reference (qwen35.rs deltanet, post-projection compute) ----------
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    let mut cs = conv0.clone();
    let mut conv_out = vec![0.0f32; conv_dim];
    for c in 0..conv_dim {
        let mut acc = 0.0f32;
        for k in 0..dconv - 1 {
            acc += conv_w[c * dconv + k] * cs[c * (dconv - 1) + k];
        }
        acc += conv_w[c * dconv + (dconv - 1)] * qkv[c];
        conv_out[c] = silu(acc);
        for k in 0..dconv - 2 {
            cs[c * (dconv - 1) + k] = cs[c * (dconv - 1) + k + 1];
        }
        cs[c * (dconv - 1) + (dconv - 2)] = qkv[c];
    }
    let q = &conv_out[0..key_dim];
    let k = &conv_out[key_dim..2 * key_dim];
    let v = &conv_out[2 * key_dim..2 * key_dim + value_dim];
    let mut qn = q.to_vec();
    let mut kn = k.to_vec();
    let l2 = |seg: &mut [f32]| {
        let ss: f32 = seg.iter().map(|x| x * x).sum();
        let r = 1.0 / ss.max(eps).sqrt();
        for x in seg.iter_mut() {
            *x *= r;
        }
    };
    for kh in 0..n_kh {
        l2(&mut qn[kh * s_v..kh * s_v + s_v]);
        l2(&mut kn[kh * s_v..kh * s_v + s_v]);
    }
    let mut st = state0.clone();
    let mut core = vec![0.0f32; value_dim];
    for vh in 0..n_vh {
        let kh = vh % n_kh;
        let qh = &qn[kh * s_v..kh * s_v + s_v];
        let kk = &kn[kh * s_v..kh * s_v + s_v];
        let g = ssm_a[vh] * softplus(alpha_raw[vh] + ssm_dt[vh]);
        let bt = sigmoid(beta_raw[vh]);
        let dec = g.exp();
        let sh = &mut st[vh * s_v * s_v..(vh + 1) * s_v * s_v];
        for x in sh.iter_mut() {
            *x *= dec;
        }
        let mut delta = vec![0.0f32; s_v];
        for j in 0..s_v {
            let rowj = &sh[j * s_v..j * s_v + s_v];
            let sum: f32 = (0..s_v).map(|i| rowj[i] * kk[i]).sum();
            delta[j] = (v[vh * s_v + j] - sum) * bt;
        }
        for j in 0..s_v {
            let dj = delta[j];
            let rowj = &mut sh[j * s_v..j * s_v + s_v];
            for i in 0..s_v {
                rowj[i] += dj * kk[i];
            }
        }
        for j in 0..s_v {
            let rowj = &sh[j * s_v..j * s_v + s_v];
            let sum: f32 = (0..s_v).map(|i| rowj[i] * qh[i]).sum();
            core[vh * s_v + j] = sum * scale;
        }
    }
    let mut gated_cpu = vec![0.0f32; value_dim];
    for vh in 0..n_vh {
        let seg = &core[vh * s_v..vh * s_v + s_v];
        let ss: f32 = seg.iter().map(|x| x * x).sum::<f32>() / s_v as f32;
        let r = 1.0 / (ss + eps).sqrt();
        for j in 0..s_v {
            gated_cpu[vh * s_v + j] = seg[j] * r * ssm_norm[j] * silu(z[vh * s_v + j]);
        }
    }

    // ---------- GPU chain ----------
    let gpu = HipGpu::new().expect("gpu");
    let code = strix_backend_rocm::hip::compile(SRC).expect("compile");
    let module = gpu.load_module(&code).expect("module");
    let f = |n: &str| gpu.get_function(module, n).expect(n);

    let d_qkv = gpu.upload_new(&qkv).unwrap();
    let d_cw = gpu.upload_new(&conv_w).unwrap();
    let d_cs = gpu.upload_new(&conv0).unwrap();
    let d_conv_out = gpu.alloc(conv_dim * 4).unwrap();
    let d_qn = gpu.alloc(key_dim * 4).unwrap();
    let d_kn = gpu.alloc(key_dim * 4).unwrap();
    let d_alpha = gpu.upload_new(&alpha_raw).unwrap();
    let d_beta_raw = gpu.upload_new(&beta_raw).unwrap();
    let d_ssm_a = gpu.upload_new(&ssm_a).unwrap();
    let d_ssm_dt = gpu.upload_new(&ssm_dt).unwrap();
    let d_decay = gpu.alloc(n_vh * 4).unwrap();
    let d_beta = gpu.alloc(n_vh * 4).unwrap();
    // device stores S transposed (i*s_v+j); transpose the CPU init before upload.
    let mut state0_t = vec![0.0f32; st_len];
    for vh in 0..n_vh {
        let base = vh * s_v * s_v;
        for j in 0..s_v {
            for i in 0..s_v {
                state0_t[base + i * s_v + j] = state0[base + j * s_v + i];
            }
        }
    }
    let d_state = gpu.upload_new(&state0_t).unwrap();
    let d_core = gpu.alloc(value_dim * 4).unwrap();
    let d_z = gpu.upload_new(&z).unwrap();
    let d_norm = gpu.upload_new(&ssm_norm).unwrap();
    let d_gated = gpu.alloc(value_dim * 4).unwrap();

    // helpers to build kernel args
    let launch = |name: &str, grid: u32, block: u32, shared: u32, args: &[Arg]| {
        let mut vals: Vec<[u8; 8]> = args
            .iter()
            .map(|a| {
                let mut e = [0u8; 8];
                match a {
                    Arg::P(p) => e.copy_from_slice(&(*p as u64).to_le_bytes()),
                    Arg::I(v) => e[..4].copy_from_slice(&v.to_le_bytes()),
                    Arg::F(v) => e[..4].copy_from_slice(&v.to_le_bytes()),
                }
                e
            })
            .collect();
        let mut ptrs: Vec<*mut c_void> = vals
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut c_void)
            .collect();
        gpu.launch(f(name), (grid, 1, 1), (block, 1, 1), shared, &mut ptrs)
            .unwrap();
    };
    // v region pointer (conv_out + 2*key_dim)
    let v_ptr = (d_conv_out.ptr as *mut f32).wrapping_add(2 * key_dim) as *mut c_void;
    let k_ptr = (d_conv_out.ptr as *mut f32).wrapping_add(key_dim) as *mut c_void;

    launch(
        "dn_conv1d",
        conv_dim.div_ceil(256) as u32,
        256,
        0,
        &[
            Arg::P(d_qkv.ptr),
            Arg::P(d_cw.ptr),
            Arg::P(d_cs.ptr),
            Arg::P(d_conv_out.ptr),
            Arg::I(conv_dim as i32),
            Arg::I(dconv as i32),
        ],
    );
    launch(
        "dn_l2norm",
        (2 * n_kh) as u32,
        s_v as u32,
        0,
        &[
            Arg::P(d_conv_out.ptr),
            Arg::P(k_ptr),
            Arg::P(d_qn.ptr),
            Arg::P(d_kn.ptr),
            Arg::I(s_v as i32),
            Arg::I(n_kh as i32),
            Arg::F(eps),
        ],
    );
    launch(
        "dn_decaybeta",
        n_vh.div_ceil(64) as u32,
        64,
        0,
        &[
            Arg::P(d_alpha.ptr),
            Arg::P(d_beta_raw.ptr),
            Arg::P(d_ssm_a.ptr),
            Arg::P(d_ssm_dt.ptr),
            Arg::P(d_decay.ptr),
            Arg::P(d_beta.ptr),
            Arg::I(n_vh as i32),
        ],
    );
    launch(
        "deltanet_step",
        n_vh as u32,
        s_v as u32,
        (2 * s_v * 4) as u32,
        &[
            Arg::P(d_state.ptr),
            Arg::P(d_qn.ptr),
            Arg::P(d_kn.ptr),
            Arg::P(v_ptr),
            Arg::P(d_decay.ptr),
            Arg::P(d_beta.ptr),
            Arg::P(d_core.ptr),
            Arg::I(s_v as i32),
            Arg::I(n_kh as i32),
            Arg::F(scale),
        ],
    );
    launch(
        "dn_gatednorm",
        n_vh as u32,
        s_v as u32,
        0,
        &[
            Arg::P(d_core.ptr),
            Arg::P(d_z.ptr),
            Arg::P(d_norm.ptr),
            Arg::P(d_gated.ptr),
            Arg::I(s_v as i32),
            Arg::F(eps),
        ],
    );
    gpu.sync().unwrap();

    let gated_gpu = d_gated.download::<f32>(value_dim).unwrap();
    let state_gpu_t = d_state.download::<f32>(st_len).unwrap();
    // transpose device state back to CPU layout (j*s_v+i) for comparison.
    let mut state_gpu = vec![0.0f32; st_len];
    for vh in 0..n_vh {
        let base = vh * s_v * s_v;
        for i in 0..s_v {
            for j in 0..s_v {
                state_gpu[base + j * s_v + i] = state_gpu_t[base + i * s_v + j];
            }
        }
    }
    let conv_gpu = d_cs.download::<f32>(conv0.len()).unwrap();

    let cmp = |name: &str, a: &[f32], b: &[f32]| -> f32 {
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let l2: f32 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        let nrm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "  {name:14} max|Δ| = {max:.3e}  rel-L2 = {:.3e}",
            l2 / nrm.max(1e-9)
        );
        max
    };
    println!("deltanet full-layer compute validation (n_vh={n_vh} n_kh={n_kh} s_v={s_v}):");
    let m1 = cmp("gated_out", &gated_cpu, &gated_gpu);
    let m2 = cmp("ssm_state", &st, &state_gpu);
    let m3 = cmp("conv_state", &cs, &conv_gpu);
    if m1 < 1e-3 && m2 < 1e-3 && m3 < 1e-3 {
        println!("  PASS");
    } else {
        println!("  FAIL");
        std::process::exit(1);
    }
}

#[cfg(feature = "rocm")]
enum Arg {
    P(*mut std::ffi::c_void),
    I(i32),
    F(f32),
}
