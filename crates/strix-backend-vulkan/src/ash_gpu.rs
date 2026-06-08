//! Raw-Vulkan (ash) compute path.
//!
//! The wgpu path is correct and fast *per kernel*, but wgpu charges ~21µs of CPU
//! per compute pass (begin/end + barrier bookkeeping). A Gemma decode step issues
//! ~14 dependent stages per layer → ~672 passes → ~14ms of pure framework tax,
//! which alone exceeds the ~6.6ms/token budget needed to beat llama.cpp (11.4
//! tok/s). A microbench proved raw ash records a dispatch + memory barrier for
//! **0.955µs** (22x cheaper), so the whole forward can be one command buffer with
//! fine-grained barriers and a single submit/token.
//!
//! This module is the foundation: a self-contained ash context that
//!
//! 1. creates the Vulkan instance / device / compute queue,
//! 2. compiles our existing **WGSL** kernels to SPIR-V via `naga` (same kernels
//!    the wgpu path runs — no rewrite, no divergence),
//! 3. allocates buffers in host-visible *and* device-local memory (UMA, so a map
//!    is a real pointer into the same RAM the GPU reads — no staging copies),
//! 4. runs the Q4_0 GEMV and validates it against a CPU reference.
//!
//! Once the foundation is proven the full decode forward records here.

use std::ffi::CStr;

use ash::vk;
use strix_core::error::{Result, StrixError};

fn vkerr<T: std::fmt::Debug>(ctx: &str, e: T) -> StrixError {
    StrixError::backend(format!("ash: {ctx}: {e:?}"))
}

/// Compile a WGSL kernel to SPIR-V words via naga (the same compiler wgpu uses).
pub fn compile_wgsl(src: &str, entry: &str) -> Result<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(src)
        .map_err(|e| StrixError::backend(format!("ash: wgsl parse: {e}")))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| StrixError::backend(format!("ash: wgsl validate: {e:?}")))?;

    let opts = naga::back::spv::Options {
        // SPIR-V 1.3 is the floor for subgroup arithmetic (subgroupAdd in SHADER_SG).
        lang_version: (1, 3),
        ..Default::default()
    };
    let pipe = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: entry.to_string(),
    };
    naga::back::spv::write_vec(&module, &info, &opts, Some(&pipe))
        .map_err(|e| StrixError::backend(format!("ash: spv emit: {e:?}")))
}

/// A GPU buffer plus the memory backing it. On UMA both live in the same RAM;
/// `ptr` is a host pointer when the buffer is host-visible (it always is here).
pub struct Buf {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    ptr: *mut u8,
}

// The mapped pointer is a stable persistent mapping into UMA memory. The
// accelerator is driven from a single thread (or externally synchronized via
// the device fence), so sharing the handle across threads is sound.
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    /// Copy `data` into the buffer (host-visible, coherent — no flush needed).
    pub fn write<T: Copy>(&self, data: &[T]) {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes as u64 <= self.size, "ash: write overruns buffer");
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, self.ptr, bytes);
        }
    }

    /// Read `n` elements of `T` from the start of the buffer.
    pub fn read<T: Copy>(&self, n: usize) -> Vec<T> {
        let bytes = n * std::mem::size_of::<T>();
        assert!(bytes as u64 <= self.size, "ash: read overruns buffer");
        let mut out = Vec::with_capacity(n);
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr as *const T, out.as_mut_ptr(), n);
            out.set_len(n);
        }
        out
    }
}

/// Self-contained raw-Vulkan compute context.
pub struct AshGpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub device: ash::Device,
    #[allow(dead_code)]
    pd: vk::PhysicalDevice,
    queue: vk::Queue,
    #[allow(dead_code)]
    qf: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    cmd_pool: vk::CommandPool,
    fence: vk::Fence,
    name: String,
}

impl AshGpu {
    pub fn new() -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load()
                .map_err(|e| StrixError::backend(format!("ash: load loader: {e}")))?;
            let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )
                .map_err(|e| vkerr("create_instance", e))?;

            let pds = instance
                .enumerate_physical_devices()
                .map_err(|e| vkerr("enumerate_physical_devices", e))?;
            let pd = pds
                .iter()
                .copied()
                .find(|&p| {
                    instance.get_physical_device_properties(p).device_type
                        == vk::PhysicalDeviceType::INTEGRATED_GPU
                })
                .or_else(|| {
                    pds.iter().copied().find(|&p| {
                        instance.get_physical_device_properties(p).device_type
                            == vk::PhysicalDeviceType::DISCRETE_GPU
                    })
                })
                .or_else(|| pds.first().copied())
                .ok_or_else(|| StrixError::backend("ash: no Vulkan physical device"))?;

            let props = instance.get_physical_device_properties(pd);
            let name = CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();

            let qf = instance
                .get_physical_device_queue_family_properties(pd)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| StrixError::backend("ash: no compute queue family"))?
                as u32;

            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qf)
                .queue_priorities(&prio)];
            let device = instance
                .create_device(
                    pd,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&qci),
                    None,
                )
                .map_err(|e| vkerr("create_device", e))?;
            let queue = device.get_device_queue(qf, 0);
            let mem_props = instance.get_physical_device_memory_properties(pd);

            let cmd_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(qf)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| vkerr("create_command_pool", e))?;
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| vkerr("create_fence", e))?;

            Ok(Self {
                _entry: entry,
                instance,
                device,
                pd,
                queue,
                qf,
                mem_props,
                cmd_pool,
                fence,
                name,
            })
        }
    }

    pub fn adapter_name(&self) -> &str {
        &self.name
    }

    /// Pick a memory type satisfying `bits` and containing all `flags`.
    fn find_mem_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32> {
        (0..self.mem_props.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && self.mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
            })
            .ok_or_else(|| StrixError::backend("ash: no suitable memory type"))
    }

    /// Allocate a host-visible, coherent, (preferably) device-local buffer and
    /// map it persistently. On UMA the map is a pointer into the GPU's RAM.
    pub fn alloc(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<Buf> {
        let size = size.max(4);
        unsafe {
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(|e| vkerr("create_buffer", e))?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            // Prefer DEVICE_LOCAL too (UMA exposes a host-visible device-local
            // heap); fall back to plain host-visible coherent if not.
            let want =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mt = self
                .find_mem_type(
                    req.memory_type_bits,
                    want | vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .or_else(|_| self.find_mem_type(req.memory_type_bits, want))?;
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mt),
                    None,
                )
                .map_err(|e| vkerr("allocate_memory", e))?;
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| vkerr("bind_buffer_memory", e))?;
            let ptr = self
                .device
                .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
                .map_err(|e| vkerr("map_memory", e))? as *mut u8;
            Ok(Buf {
                buffer,
                memory,
                size,
                ptr,
            })
        }
    }

    /// Build a compute pipeline from WGSL: `n_buffers` storage bindings (0..n-1)
    /// plus an optional trailing uniform binding for the dims/params word.
    pub(crate) fn build_pipeline(
        &self,
        wgsl: &str,
        entry: &str,
        n_storage: u32,
        uniform: bool,
    ) -> Result<Pipeline> {
        let spv = compile_wgsl(wgsl, entry)?;
        unsafe {
            let module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
                .map_err(|e| vkerr("create_shader_module", e))?;

            let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_storage)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            if uniform {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(n_storage)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            let dsl = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| vkerr("create_descriptor_set_layout", e))?;
            let dsls = [dsl];
            let layout = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls),
                    None,
                )
                .map_err(|e| vkerr("create_pipeline_layout", e))?;

            let ep = std::ffi::CString::new(entry).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&ep);
            let pipeline = self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, e)| vkerr("create_compute_pipelines", e))?[0];
            self.device.destroy_shader_module(module, None);

            Ok(Pipeline {
                pipeline,
                layout,
                dsl,
                n_storage,
                uniform,
            })
        }
    }

    /// Allocate + write a descriptor set binding the given storage buffers (in
    /// order) and an optional uniform buffer last.
    #[allow(dead_code)]
    fn make_descriptor_set(
        &self,
        p: &Pipeline,
        storage: &[&Buf],
        uniform: Option<&Buf>,
    ) -> Result<(vk::DescriptorPool, vk::DescriptorSet)> {
        unsafe {
            let mut sizes = vec![vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(p.n_storage.max(1))];
            if p.uniform {
                sizes.push(
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1),
                );
            }
            let pool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&sizes)
                        .max_sets(1),
                    None,
                )
                .map_err(|e| vkerr("create_descriptor_pool", e))?;
            let dsls = [p.dsl];
            let ds = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&dsls),
                )
                .map_err(|e| vkerr("allocate_descriptor_sets", e))?[0];

            // Keep the per-binding info structs alive until update returns.
            let mut infos: Vec<[vk::DescriptorBufferInfo; 1]> = Vec::new();
            for b in storage {
                infos.push([vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]);
            }
            if let Some(u) = uniform {
                infos.push([vk::DescriptorBufferInfo::default()
                    .buffer(u.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]);
            }
            let mut writes = Vec::new();
            for (i, info) in infos.iter().enumerate() {
                let is_uniform = p.uniform && i == storage.len();
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(i as u32)
                        .descriptor_type(if is_uniform {
                            vk::DescriptorType::UNIFORM_BUFFER
                        } else {
                            vk::DescriptorType::STORAGE_BUFFER
                        })
                        .buffer_info(info),
                );
            }
            self.device.update_descriptor_sets(&writes, &[]);
            Ok((pool, ds))
        }
    }

    /// Create a descriptor pool sized for `max_sets` sets drawing on a shared
    /// budget of `n_storage` storage + `n_uniform` uniform descriptors.
    pub(crate) fn create_descriptor_pool(
        &self,
        max_sets: u32,
        n_storage: u32,
        n_uniform: u32,
    ) -> Result<vk::DescriptorPool> {
        unsafe {
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(n_storage.max(1)),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(n_uniform.max(1)),
            ];
            self.device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&sizes)
                        .max_sets(max_sets.max(1)),
                    None,
                )
                .map_err(|e| vkerr("create_descriptor_pool", e))
        }
    }

    /// Allocate one descriptor set from `pool` for pipeline `p`, binding the
    /// given storage buffers in order (bindings 0..) and an optional uniform last.
    pub(crate) fn alloc_set(
        &self,
        pool: vk::DescriptorPool,
        p: &Pipeline,
        storage: &[vk::Buffer],
        uniform: Option<vk::Buffer>,
    ) -> Result<vk::DescriptorSet> {
        unsafe {
            let dsls = [p.dsl];
            let ds = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&dsls),
                )
                .map_err(|e| vkerr("allocate_descriptor_sets", e))?[0];

            let mut infos: Vec<[vk::DescriptorBufferInfo; 1]> = Vec::new();
            for &b in storage {
                infos.push([vk::DescriptorBufferInfo::default()
                    .buffer(b)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]);
            }
            if let Some(u) = uniform {
                infos.push([vk::DescriptorBufferInfo::default()
                    .buffer(u)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]);
            }
            let mut writes = Vec::new();
            for (i, info) in infos.iter().enumerate() {
                let is_uniform = uniform.is_some() && i == storage.len();
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(i as u32)
                        .descriptor_type(if is_uniform {
                            vk::DescriptorType::UNIFORM_BUFFER
                        } else {
                            vk::DescriptorType::STORAGE_BUFFER
                        })
                        .buffer_info(info),
                );
            }
            self.device.update_descriptor_sets(&writes, &[]);
            Ok(ds)
        }
    }

    /// Allocate a primary command buffer from the context's pool.
    pub(crate) fn cmd_buffer(&self) -> Result<vk::CommandBuffer> {
        self.alloc_cmd_buffer()
    }

    /// Submit `cb` and block until it finishes.
    pub(crate) fn run(&self, cb: vk::CommandBuffer) -> Result<()> {
        self.submit_wait(cb)
    }

    fn alloc_cmd_buffer(&self) -> Result<vk::CommandBuffer> {
        unsafe {
            Ok(self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| vkerr("allocate_command_buffers", e))?[0])
        }
    }

    /// Submit `cb` and block until it completes.
    fn submit_wait(&self, cb: vk::CommandBuffer) -> Result<()> {
        unsafe {
            let cbs = [cb];
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&cbs)],
                    self.fence,
                )
                .map_err(|e| vkerr("queue_submit", e))?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| vkerr("wait_for_fences", e))?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(|e| vkerr("reset_fences", e))?;
            Ok(())
        }
    }
}

impl Drop for AshGpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub(crate) struct Pipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) dsl: vk::DescriptorSetLayout,
    pub(crate) n_storage: u32,
    pub(crate) uniform: bool,
}

/// Q4_0 GEMV via the raw-ash path, reusing the exact WGSL subgroup kernel from
/// `qgemv::SHADER_SG`. Repacks the weight to the same (f16-scales, u32-quants)
/// layout, runs `iters` dispatches, and returns (result, seconds_per_iter).
///
/// `bytes` is raw Q4_0 GGUF for a `[out_dim, in_dim]` weight.
#[allow(clippy::too_many_arguments)]
pub fn bench_q4_gemv(
    gpu: &AshGpu,
    bytes: &[u8],
    x: &[f32],
    in_dim: usize,
    out_dim: usize,
    iters: usize,
) -> Result<(Vec<f32>, f64)> {
    const QK: usize = 32;
    const Q4_0_BYTES: usize = 18;
    if in_dim % QK != 0 {
        return Err(StrixError::invalid("ash q4: in_dim not a multiple of 32"));
    }
    let nblocks = in_dim / QK;
    let total = nblocks * out_dim;
    if bytes.len() != total * Q4_0_BYTES {
        return Err(StrixError::invalid("ash q4: byte length mismatch"));
    }

    // Repack: scales as f16 bits, two per u32; quants as 4 u32 per block.
    let mut scales = vec![0u32; total.div_ceil(2)];
    let mut quants = vec![0u32; total * 4];
    for (b, blk) in bytes.chunks_exact(Q4_0_BYTES).enumerate() {
        let h = u16::from_le_bytes([blk[0], blk[1]]) as u32;
        scales[b >> 1] |= h << (16 * (b & 1));
        let qs = &blk[2..18];
        for w in 0..4 {
            quants[b * 4 + w] =
                u32::from_le_bytes([qs[w * 4], qs[w * 4 + 1], qs[w * 4 + 2], qs[w * 4 + 3]]);
        }
    }

    // 2D grid to dodge the 65535-per-dim workgroup limit (matches wgpu path).
    let grid_x = (out_dim as u32).min(32768);
    let grid_y = (out_dim as u32).div_ceil(grid_x);
    let dims = [in_dim as u32, out_dim as u32, grid_x, 0u32];

    let su = vk::BufferUsageFlags::STORAGE_BUFFER;
    let scales_buf = gpu.alloc((scales.len() * 4) as u64, su)?;
    let quants_buf = gpu.alloc((quants.len() * 4) as u64, su)?;
    let x_buf = gpu.alloc((in_dim * 4) as u64, su)?;
    let y_buf = gpu.alloc((out_dim * 4) as u64, su)?;
    let dims_buf = gpu.alloc(16, vk::BufferUsageFlags::UNIFORM_BUFFER)?;
    scales_buf.write(&scales);
    quants_buf.write(&quants);
    x_buf.write(x);
    dims_buf.write(&dims);

    let pipe = gpu.build_pipeline(crate::qgemv::SHADER_SG, "main", 4, true)?;
    let (pool, ds) = gpu.make_descriptor_set(
        &pipe,
        &[&scales_buf, &quants_buf, &x_buf, &y_buf],
        Some(&dims_buf),
    )?;
    let cb = gpu.alloc_cmd_buffer()?;

    let record = |reps: usize| -> Result<()> {
        unsafe {
            let d = &gpu.device;
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .map_err(|e| vkerr("reset_command_buffer", e))?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| vkerr("begin_command_buffer", e))?;
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            d.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[ds],
                &[],
            );
            let bar = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)];
            for _ in 0..reps {
                d.cmd_dispatch(cb, grid_x, grid_y, 1);
                d.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &bar,
                    &[],
                    &[],
                );
            }
            d.end_command_buffer(cb)
                .map_err(|e| vkerr("end_command_buffer", e))?;
        }
        Ok(())
    };

    // Warm up, then time `iters` dispatches recorded into one command buffer.
    record(1)?;
    gpu.submit_wait(cb)?;
    let result = y_buf.read::<f32>(out_dim);

    record(iters)?;
    let t = std::time::Instant::now();
    gpu.submit_wait(cb)?;
    let per_iter = t.elapsed().as_secs_f64() / iters as f64;

    unsafe {
        gpu.device.destroy_descriptor_pool(pool, None);
        gpu.device.destroy_pipeline(pipe.pipeline, None);
        gpu.device.destroy_pipeline_layout(pipe.layout, None);
        gpu.device.destroy_descriptor_set_layout(pipe.dsl, None);
        for b in [&scales_buf, &quants_buf, &x_buf, &y_buf, &dims_buf] {
            gpu.device.unmap_memory(b.memory);
            gpu.device.destroy_buffer(b.buffer, None);
            gpu.device.free_memory(b.memory, None);
        }
    }

    Ok((result, per_iter))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_f16(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x7fffff;
        if exp <= 0 {
            return sign;
        }
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }

    fn f16_to_f32(h: u16) -> f32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let mant = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            sign << 31
        } else {
            (sign << 31) | ((exp + 112) << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }

    fn synth_q4_0(in_dim: usize, out_dim: usize) -> Vec<u8> {
        let nblocks = in_dim / 32;
        let mut bytes = Vec::with_capacity(out_dim * nblocks * 18);
        let mut seed = 0xA5A5_1234u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u32
        };
        for _ in 0..out_dim * nblocks {
            let d = 0.03f32 + (next() % 32) as f32 * 0.003;
            bytes.extend_from_slice(&f32_to_f16(d).to_le_bytes());
            for _ in 0..16 {
                let lo = (next() % 16) as u8;
                let hi = (next() % 16) as u8;
                bytes.push(lo | (hi << 4));
            }
        }
        bytes
    }

    /// CPU Q4_0 GEMV reference straight from the raw bytes (matches the kernel's
    /// dequant exactly: value = d * (nibble - 8), low nibbles then high nibbles).
    fn cpu_q4_gemv(bytes: &[u8], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let nblocks = in_dim / 32;
        (0..out_dim)
            .map(|o| {
                let mut acc = 0.0f32;
                for b in 0..nblocks {
                    let blk = &bytes[(o * nblocks + b) * 18..][..18];
                    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    let xbase = b * 32;
                    for (j, &byte) in blk[2..18].iter().enumerate() {
                        let lo = (byte & 0x0f) as f32 - 8.0;
                        let hi = (byte >> 4) as f32 - 8.0;
                        acc += d * (lo * x[xbase + j] + hi * x[xbase + j + 16]);
                    }
                }
                acc
            })
            .collect()
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn ash_q4_gemv_matches_cpu() {
        let (in_dim, out_dim) = (2048usize, 1024usize);
        let bytes = synth_q4_0(in_dim, out_dim);
        let x: Vec<f32> = (0..in_dim)
            .map(|i| (i as f32 * 0.017).sin() * 0.5)
            .collect();
        let cpu = cpu_q4_gemv(&bytes, &x, in_dim, out_dim);

        let gpu = AshGpu::new().expect("ash device");
        let (got, per_iter) =
            bench_q4_gemv(&gpu, &bytes, &x, in_dim, out_dim, 200).expect("ash gemv");

        let max_err = cpu
            .iter()
            .zip(&got)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let scale = cpu.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        let weight_bytes = out_dim * (in_dim / 32) * 18;
        eprintln!(
            "ash Q4 GEMV on {}: {:.3} ms/iter  {:.1} GB/s  max_err {:.2e} (rel {:.2e})",
            gpu.adapter_name(),
            per_iter * 1e3,
            weight_bytes as f64 / per_iter / 1e9,
            max_err,
            max_err / scale,
        );
        assert!(
            max_err / scale < 1e-3,
            "ash GEMV diverged: rel {}",
            max_err / scale
        );
    }
}
