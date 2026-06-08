//! `strix device-info` — best-effort hardware detection.
//!
//! Detection is intentionally lightweight and dependency-free: we read a couple
//! of `/proc` files on Linux and stub cleanly elsewhere. Vulkan adapters come
//! from the (feature-gated) Vulkan backend. NPU detection is a placeholder.

use anyhow::Result;
use strix_backend_cpu::CpuBackend;
use strix_backend_vulkan::enumerate_adapters;
use strix_core::backend::Backend;

/// Gather and print device information.
pub fn run() -> Result<()> {
    println!("Strix device info");
    println!();

    // OS / arch.
    println!(
        "OS:   {} ({})",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    // CPU.
    let cpu = CpuBackend::new();
    let cpu_info = cpu.device_info();
    let cpu_name = detect_cpu_name().unwrap_or(cpu_info.name);
    println!("CPU:  {cpu_name}");
    for note in &cpu_info.notes {
        println!("      {note}");
    }

    // RAM.
    match detect_total_ram_bytes() {
        Some(bytes) => println!("RAM:  {}", human_bytes(bytes)),
        None => println!("RAM:  <unknown>"),
    }

    // Vulkan adapters (Radeon 890M iGPU target).
    println!();
    let adapters = enumerate_adapters();
    if adapters.is_empty() {
        println!("Vulkan adapters: none detected");
        if cfg!(not(feature = "vulkan")) {
            println!("  (rebuild with `--features vulkan` to enumerate the iGPU)");
        }
    } else {
        println!("Vulkan adapters: {}", adapters.len());
        for a in &adapters {
            println!("  - {} [{}]", a.name, a.backend);
            for note in &a.notes {
                println!("      {note}");
            }
        }
    }
    #[cfg(feature = "vulkan")]
    {
        match strix_backend_vulkan::GpuMatvec::new() {
            Ok(gpu) => {
                // Tiny matvec to confirm the compute path actually executes.
                let ok = gpu.matvec(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0], 2, 2).is_ok();
                println!(
                    "  compute: {} on {}",
                    if ok {
                        "ready (matvec OK)"
                    } else {
                        "init ok, matvec failed"
                    },
                    gpu.adapter_name()
                );
            }
            Err(e) => println!("  compute: unavailable ({e})"),
        }
    }

    // NPU (XDNA2) via XRT.
    println!();
    let npu = strix_backend_npu::probe();
    let ni = npu.device_info();
    println!(
        "NPU (XDNA2): {}",
        if npu.opened {
            "present, XRT access OK"
        } else {
            "not opened"
        }
    );
    for note in &ni.notes {
        println!("  {note}");
    }
    if npu.opened {
        // Exercise the XRT buffer data path (host↔NPU DMA), no kernel needed.
        let bt = strix_backend_npu::buffer_roundtrip(64 * 1024);
        println!(
            "  data path: {} — {}",
            if bt.ok { "OK" } else { "unavailable" },
            bt.detail
        );
    }

    Ok(())
}

/// Best-effort CPU model name (Linux `/proc/cpuinfo`).
fn detect_cpu_name() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in text.lines() {
            if let Some((key, val)) = line.split_once(':') {
                if key.trim() == "model name" {
                    return Some(val.trim().to_string());
                }
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Best-effort total RAM in bytes (Linux `/proc/meminfo`).
fn detect_total_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                // Format: "MemTotal:       32768000 kB"
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Format a byte count as GiB/MiB.
fn human_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else {
        format!("{:.1} MiB", b / MIB)
    }
}
