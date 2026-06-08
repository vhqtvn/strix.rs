//! Logical device description.
//!
//! `DeviceInfo` is what `device-info` prints and what backends advertise. It is
//! deliberately string-heavy and cheap to construct so backends can fill in
//! only what they cheaply know.

use serde::{Deserialize, Serialize};

/// Which kind of compute device a backend drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceKind {
    /// Host CPU.
    Cpu,
    /// Integrated or discrete GPU (e.g. Radeon 890M via Vulkan).
    Gpu,
    /// AMD XDNA NPU (planned).
    Npu,
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DeviceKind::Cpu => "CPU",
            DeviceKind::Gpu => "GPU",
            DeviceKind::Npu => "NPU",
        };
        f.write_str(s)
    }
}

/// Human-facing description of a compute device exposed by a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device class.
    pub kind: DeviceKind,
    /// Display name (e.g. "AMD Radeon 890M", "host CPU").
    pub name: String,
    /// Backend driving this device (e.g. "cpu", "vulkan").
    pub backend: String,
    /// Total device-visible memory in bytes, if known.
    pub total_memory_bytes: Option<u64>,
    /// Free notes (driver version, compute units, etc.).
    pub notes: Vec<String>,
}

impl DeviceInfo {
    /// Construct a minimal `DeviceInfo`.
    pub fn new(kind: DeviceKind, name: impl Into<String>, backend: impl Into<String>) -> Self {
        DeviceInfo {
            kind,
            name: name.into(),
            backend: backend.into(),
            total_memory_bytes: None,
            notes: Vec::new(),
        }
    }
}
