//! `strix-backend-cpu` — the CPU reference backend.
//!
//! Phase 1 goal: *correctness over speed*. This backend will host the slow,
//! readable reference implementations of RMSNorm, RoPE, attention, and MLP
//! (Milestone 2). For now it exists as a registered [`Backend`] so the CLI has
//! something real to enumerate and `device-info` / `bench-dummy` can run.

pub mod attention;
pub mod gemma;
pub mod kv_cache;
pub mod llama;
pub mod mellum;
#[cfg(feature = "npu")]
pub mod mellum_npu;
pub mod ops;
pub mod qwen35;

pub use gemma::GemmaModel;
pub use llama::LlamaModel;

use strix_core::backend::Backend;
use strix_core::device::{DeviceInfo, DeviceKind};

/// CPU reference backend.
#[derive(Debug, Default, Clone)]
pub struct CpuBackend {
    /// Optional override for the reported CPU name (else best-effort detection).
    name: Option<String>,
}

impl CpuBackend {
    /// Create a CPU backend with best-effort device naming.
    pub fn new() -> Self {
        CpuBackend { name: None }
    }

    /// Create a CPU backend with an explicit display name.
    pub fn with_name(name: impl Into<String>) -> Self {
        CpuBackend {
            name: Some(name.into()),
        }
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device_info(&self) -> DeviceInfo {
        let name = self.name.clone().unwrap_or_else(|| "host CPU".to_string());

        let mut info = DeviceInfo::new(DeviceKind::Cpu, name, "cpu");
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        info.notes.push(format!("logical threads: {threads}"));
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_reports_identity() {
        let b = CpuBackend::new();
        assert_eq!(b.name(), "cpu");
        let info = b.device_info();
        assert_eq!(info.kind, DeviceKind::Cpu);
        assert_eq!(info.backend, "cpu");
    }
}
