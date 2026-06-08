//! `strix-backend-vulkan` — the Radeon 890M iGPU backend (skeleton).
//!
//! Phase 1 scope is *device enumeration only*. No buffers, no kernels, no
//! inference. The plan (see `docs/architecture.md` / `docs/hx370-notes.md`):
//!
//! 1. enumerate Vulkan adapters (this file, behind the `vulkan` feature),
//! 2. allocate buffers + upload tensors,
//! 3. one matmul compute shader, benchmarked against CPU,
//! 4. quantized matmul.
//!
//! Enumeration uses `wgpu` for portability. A later low-level compute path may
//! switch to `ash`; that decision is deferred (see hx370-notes).

#[cfg(feature = "vulkan")]
pub mod compute;
#[cfg(feature = "vulkan")]
pub use compute::GpuMatvec;
#[cfg(feature = "vulkan")]
pub mod qgemv;
#[cfg(feature = "vulkan")]
pub use qgemv::{GpuQ4, ResidentQ4, ResidentQ6};
#[cfg(feature = "vulkan")]
pub mod accel;
#[cfg(feature = "vulkan")]
pub use accel::{gpu_time_ms, reset_gpu_time, GpuWeightAccel};
#[cfg(feature = "ash")]
pub mod ash_gpu;
#[cfg(feature = "ash")]
pub use ash_gpu::AshGpu;
#[cfg(feature = "ash")]
pub mod ash_decode;
#[cfg(feature = "ash")]
pub use ash_decode::AshWeightAccel;

use strix_core::backend::Backend;
use strix_core::device::{DeviceInfo, DeviceKind};

/// Vulkan/iGPU backend placeholder.
///
/// Holds a chosen adapter's [`DeviceInfo`] once enumeration succeeds. Until the
/// `vulkan` feature is enabled it advertises a CPU-fallback note so the CLI can
/// still report cleanly.
#[derive(Debug, Clone)]
pub struct VulkanBackend {
    info: DeviceInfo,
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanBackend {
    /// Construct a backend, picking the first available adapter if any.
    pub fn new() -> Self {
        let info = enumerate_adapters().into_iter().next().unwrap_or_else(|| {
            let mut i = DeviceInfo::new(DeviceKind::Gpu, "no Vulkan adapter", "vulkan");
            i.notes
                .push("build with --features vulkan and ensure a Vulkan driver is present".into());
            i
        });
        VulkanBackend { info }
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn device_info(&self) -> DeviceInfo {
        self.info.clone()
    }
}

/// Enumerate Vulkan adapters as [`DeviceInfo`] entries.
///
/// Returns an empty vector when the `vulkan` feature is disabled or no adapter
/// is found. This is the only function the CLI needs in Phase 1.
#[cfg(feature = "vulkan")]
pub fn enumerate_adapters() -> Vec<DeviceInfo> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .into_iter()
        .map(|adapter| {
            let a = adapter.get_info();
            let mut info = DeviceInfo::new(DeviceKind::Gpu, a.name.clone(), "vulkan");
            info.notes.push(format!("device type: {:?}", a.device_type));
            info.notes
                .push(format!("driver: {} {}", a.driver, a.driver_info));
            info.notes.push(format!(
                "vendor: 0x{:04x} device: 0x{:04x}",
                a.vendor, a.device
            ));
            info
        })
        .collect()
}

/// Enumeration stub used when the `vulkan` feature is off.
#[cfg(not(feature = "vulkan"))]
pub fn enumerate_adapters() -> Vec<DeviceInfo> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_identity_is_stable() {
        let b = VulkanBackend::new();
        assert_eq!(b.name(), "vulkan");
        assert_eq!(b.device_info().kind, DeviceKind::Gpu);
    }

    #[cfg(not(feature = "vulkan"))]
    #[test]
    fn enumeration_empty_without_feature() {
        assert!(enumerate_adapters().is_empty());
    }
}
