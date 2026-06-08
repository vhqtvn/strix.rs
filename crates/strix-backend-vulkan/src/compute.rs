//! Vulkan compute for the Radeon 890M iGPU, via `wgpu`.
//!
//! Phase-3 first step: an f32 matrix-vector product `y = W · x` on the GPU —
//! the operation that dominates LLM inference (every projection is a matvec at
//! decode time). `W` is row-major `[out_dim, in_dim]` (HF/GGML layout). This is
//! the GPU analogue of the CPU `linear`, used to validate correctness against
//! the CPU oracle and to benchmark the iGPU.
//!
//! Quantized weights and a fused dequant+matmul kernel come next; this proves
//! the device → buffer → kernel → readback path works.

use strix_core::error::{Result, StrixError};
use wgpu::util::DeviceExt;

/// A live GPU context (device + queue) plus the matvec pipeline.
pub struct GpuMatvec {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
}

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec2<u32>; // (in_dim, out_dim)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let in_dim = dims.x;
    let out_dim = dims.y;
    if (o >= out_dim) { return; }
    var acc = 0.0;
    let base = o * in_dim;
    for (var i = 0u; i < in_dim; i = i + 1u) {
        acc = acc + w[base + i] * x[i];
    }
    y[o] = acc;
}
"#;

impl GpuMatvec {
    /// Initialize the Vulkan device and compile the matvec kernel.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| StrixError::Backend {
                backend: "vulkan",
                message: "no Vulkan adapter found".into(),
            })?;
        let adapter_name = adapter.get_info().name;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("strix-vulkan"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| StrixError::Backend {
                backend: "vulkan",
                message: format!("request_device: {e}"),
            })?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(GpuMatvec {
            device,
            queue,
            pipeline,
            adapter_name,
        })
    }

    /// Adapter name (e.g. "AMD Radeon 890M Graphics (RADV STRIX1)").
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Compute `y = W · x` on the GPU. `w` is `[out_dim, in_dim]` row-major.
    pub fn matvec(&self, w: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
        if w.len() != in_dim * out_dim || x.len() != in_dim {
            return Err(StrixError::invalid("matvec: shape mismatch"));
        }
        let dev = &self.device;

        let w_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("w"),
            contents: bytemuck::cast_slice(w),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let x_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let y_size = (out_dim * std::mem::size_of::<f32>()) as u64;
        let y_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: y_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims = [in_dim as u32, out_dim as u32];
        let dims_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: y_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matvec"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (out_dim as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, y_size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::Maintain::Wait);
        rx.recv()
            .ok()
            .and_then(|r| r.ok())
            .ok_or_else(|| StrixError::Backend {
                backend: "vulkan",
                message: "buffer map failed".into(),
            })?;
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_matvec(w: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        (0..out_dim)
            .map(|o| (0..in_dim).map(|i| w[o * in_dim + i] * x[i]).sum())
            .collect()
    }

    #[test]
    fn gpu_matvec_matches_cpu() {
        // Skip gracefully if no GPU is available in the environment.
        let gpu = match GpuMatvec::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let (in_dim, out_dim) = (320usize, 128usize);
        let mut seed = 0x9e37u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) - 0.5
        };
        let w: Vec<f32> = (0..in_dim * out_dim).map(|_| rnd()).collect();
        let x: Vec<f32> = (0..in_dim).map(|_| rnd()).collect();

        let got = gpu.matvec(&w, &x, in_dim, out_dim).unwrap();
        let want = cpu_matvec(&w, &x, in_dim, out_dim);
        assert_eq!(got.len(), out_dim);
        for (g, c) in got.iter().zip(&want) {
            assert!((g - c).abs() < 1e-3, "gpu {g} vs cpu {c}");
        }
        eprintln!("GPU matvec validated on {}", gpu.adapter_name());
    }
}
