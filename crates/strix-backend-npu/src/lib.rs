//! `strix-backend-npu` — AMD XDNA2 (Ryzen AI) NPU backend.
//!
//! Phase 1 scope is *access only*: prove we can open and talk to the NPU from
//! Rust via the XRT runtime (`libxrt_coreutil`) — no kernels yet. Running real
//! matmuls on the NPU additionally requires an AI-Engine kernel compiled to an
//! `.xclbin` (the mlir-aie / Peano toolchain), which is a separate milestone.
//!
//! The XRT FFI lives behind the `ryzen-ai` feature so the default build has no
//! system dependency. With the feature on, this links the system XRT and uses
//! its C API (`xrtDeviceOpen` / `xrtDeviceClose`, BO + kernel calls later).

use strix_core::device::{DeviceInfo, DeviceKind};

/// Result of probing the NPU.
#[derive(Debug, Clone)]
pub struct NpuProbe {
    /// True if the XRT device opened successfully.
    pub opened: bool,
    /// Human-readable detail.
    pub detail: String,
}

impl NpuProbe {
    /// Render as a [`DeviceInfo`] for `device-info` output.
    pub fn device_info(&self) -> DeviceInfo {
        let mut i = DeviceInfo::new(DeviceKind::Npu, "AMD XDNA2 (Ryzen AI)", "ryzen-ai");
        i.notes.push(self.detail.clone());
        if self.opened {
            i.notes
                .push("XRT device opened — kernel execution (xclbin) not yet implemented".into());
        }
        i
    }
}

#[cfg(feature = "ryzen-ai")]
mod xrt {
    use std::os::raw::{c_int, c_uint, c_void};

    use std::ffi::CString;
    use std::os::raw::c_char;

    // Minimal slice of the XRT C API (see /usr/include/xrt/{xrt_device,xrt_bo,
    // xrt_kernel}.h). Handles are opaque pointers; *Open/*Alloc return null on
    // failure, *Get/Load/Set/Sync return 0 on success.
    #[link(name = "xrt_coreutil")]
    extern "C" {
        fn xrtDeviceOpen(index: c_uint) -> *mut c_void;
        fn xrtDeviceClose(handle: *mut c_void) -> c_int;

        // xrtBufferFlags = uint64_t, xrtMemoryGroup = uint32_t.
        fn xrtBOAlloc(dev: *mut c_void, size: usize, flags: u64, grp: u32) -> *mut c_void;
        fn xrtBOFree(bo: *mut c_void) -> c_int;
        fn xrtBOMap(bo: *mut c_void) -> *mut c_void;
        fn xrtBOSync(bo: *mut c_void, dir: c_int, size: usize, offset: usize) -> c_int;
        fn xrtBOWrite(bo: *mut c_void, src: *const c_void, size: usize, seek: usize) -> c_int;
        fn xrtBORead(bo: *mut c_void, dst: *mut c_void, size: usize, skip: usize) -> c_int;
    }

    // The XDNA NPU run path (hw_context) has no C API — only C++. These are our
    // C++ shim's entry points (src/npu_shim.cpp), compiled + linked by build.rs.
    extern "C" {
        fn strix_npu_load_probe(
            xclbin: *const c_char,
            uuidbuf: *mut c_char,
            uuidcap: usize,
            errbuf: *mut c_char,
            errcap: usize,
        ) -> c_int;
        #[allow(clippy::too_many_arguments)]
        fn strix_npu_matmul(
            xclbin: *const c_char,
            kernel_name: *const c_char,
            instr: *const u32,
            instr_words: usize,
            a: *const c_void,
            a_bytes: usize,
            b: *const c_void,
            b_bytes: usize,
            out: *mut c_void,
            out_bytes: usize,
            errbuf: *mut c_char,
            errcap: usize,
        ) -> c_int;
        // 2-buffer kernel (fused attention): one packed input, one output.
        fn strix_npu_attn(
            xclbin: *const c_char,
            kernel_name: *const c_char,
            instr: *const u32,
            instr_words: usize,
            input: *const c_void,
            in_bytes: usize,
            out: *mut c_void,
            out_bytes: usize,
            errbuf: *mut c_char,
            errcap: usize,
        ) -> c_int;

        // Persistent context: open the xclbin once, run many matmuls.
        fn strix_npu_open(
            xclbin: *const c_char,
            kernel: *const c_char,
            errbuf: *mut c_char,
            errcap: usize,
        ) -> *mut c_void;
        #[allow(clippy::too_many_arguments)]
        fn strix_npu_ctx_run(
            h: *mut c_void,
            instr: *const u32,
            instr_words: usize,
            a: *const c_void,
            a_bytes: usize,
            b: *const c_void,
            b_bytes: usize,
            out: *mut c_void,
            out_bytes: usize,
            errbuf: *mut c_char,
            errcap: usize,
        ) -> c_int;
        fn strix_npu_close(h: *mut c_void);

        // Hybrid GEMM: staged per-weight BOs + zero-copy host I/O + async run.
        #[allow(clippy::too_many_arguments)]
        fn strix_npu_gemm_open(
            xclbin: *const c_char,
            kernel: *const c_char,
            instr: *const u32,
            instr_words: usize,
            a_cap: usize,
            out_cap: usize,
            a_host: *mut *mut c_void,
            out_host: *mut *mut c_void,
            errbuf: *mut c_char,
            errcap: usize,
        ) -> *mut c_void;
        fn strix_npu_gemm_stage(
            h: *mut c_void,
            b: *const c_void,
            b_bytes: usize,
            errbuf: *mut c_char,
            errcap: usize,
        ) -> c_int;
        fn strix_npu_gemm_start(
            h: *mut c_void,
            wid: c_int,
            errbuf: *mut c_char,
            errcap: usize,
        ) -> c_int;
        fn strix_npu_gemm_wait(h: *mut c_void, errbuf: *mut c_char, errcap: usize) -> c_int;
        fn strix_npu_gemm_close(h: *mut c_void);
    }

    /// A fixed-shape NPU GEMM with staged weights, zero-copy host I/O, and async
    /// start/wait. `a_host`/`out_host` are the activation-in (int8 [M,K]) and
    /// output (int32 [M,N]) host buffers — the iGPU hipHostRegisters them.
    pub struct Gemm {
        handle: *mut c_void,
        pub a_host: *mut c_void,
        pub out_host: *mut c_void,
    }
    unsafe impl Send for Gemm {}

    impl Gemm {
        #[allow(clippy::too_many_arguments)]
        pub fn open(
            xclbin_path: &str,
            kernel_name: &str,
            instr: &[u32],
            a_cap: usize,
            out_cap: usize,
        ) -> Result<Self, String> {
            let cx = CString::new(xclbin_path).map_err(|_| "xclbin NUL")?;
            let ck = CString::new(kernel_name).map_err(|_| "kernel NUL")?;
            let mut err = [0 as c_char; 512];
            let mut a_host: *mut c_void = std::ptr::null_mut();
            let mut out_host: *mut c_void = std::ptr::null_mut();
            let h = unsafe {
                strix_npu_gemm_open(
                    cx.as_ptr(),
                    ck.as_ptr(),
                    instr.as_ptr(),
                    instr.len(),
                    a_cap,
                    out_cap,
                    &mut a_host,
                    &mut out_host,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if h.is_null() {
                Err(format!("npu_gemm_open: {}", cbuf_to_string(&err)))
            } else {
                Ok(Gemm {
                    handle: h,
                    a_host,
                    out_host,
                })
            }
        }

        /// Stage a weight (B int8 [K,N] row-major). Returns its index.
        pub fn stage(&mut self, b: &[i8]) -> Result<i32, String> {
            let mut err = [0 as c_char; 512];
            let r = unsafe {
                strix_npu_gemm_stage(
                    self.handle,
                    b.as_ptr() as *const c_void,
                    b.len(),
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if r < 0 {
                Err(format!("npu_gemm_stage: {}", cbuf_to_string(&err)))
            } else {
                Ok(r)
            }
        }

        /// Start the GEMM with staged weight `wid` (activation already in `a_host`). Non-blocking.
        pub fn start(&self, wid: i32) -> Result<(), String> {
            let mut err = [0 as c_char; 512];
            let r = unsafe { strix_npu_gemm_start(self.handle, wid, err.as_mut_ptr(), err.len()) };
            if r == 0 {
                Ok(())
            } else {
                Err(format!("npu_gemm_start: {}", cbuf_to_string(&err)))
            }
        }

        /// Wait for completion; output is then valid in `out_host`.
        pub fn wait(&self) -> Result<(), String> {
            let mut err = [0 as c_char; 512];
            let r = unsafe { strix_npu_gemm_wait(self.handle, err.as_mut_ptr(), err.len()) };
            if r == 0 {
                Ok(())
            } else {
                Err(format!("npu_gemm_wait: {}", cbuf_to_string(&err)))
            }
        }
    }

    impl Drop for Gemm {
        fn drop(&mut self) {
            unsafe { strix_npu_gemm_close(self.handle) };
        }
    }

    /// A loaded NPU kernel context (xclbin registered once). Run a matmul per
    /// call without re-loading — the basis for NPU prefill (one GEMM per layer).
    pub struct Context {
        handle: *mut c_void,
    }
    // Driven single-threaded; the handle is an opaque XRT context.
    unsafe impl Send for Context {}

    impl Context {
        pub fn open(xclbin_path: &str, kernel_name: &str) -> Result<Self, String> {
            let cx = CString::new(xclbin_path).map_err(|_| "xclbin path NUL")?;
            let ck = CString::new(kernel_name).map_err(|_| "kernel name NUL")?;
            let mut err = [0 as c_char; 512];
            let h =
                unsafe { strix_npu_open(cx.as_ptr(), ck.as_ptr(), err.as_mut_ptr(), err.len()) };
            if h.is_null() {
                Err(format!("strix_npu_open failed: {}", cbuf_to_string(&err)))
            } else {
                Ok(Context { handle: h })
            }
        }

        pub fn run_matmul(
            &self,
            instr: &[u32],
            a: &[u8],
            b: &[u8],
            out_bytes: usize,
        ) -> Result<Vec<u8>, String> {
            let mut out = vec![0u8; out_bytes];
            let mut err = [0 as c_char; 512];
            let rc = unsafe {
                strix_npu_ctx_run(
                    self.handle,
                    instr.as_ptr(),
                    instr.len(),
                    a.as_ptr() as *const c_void,
                    a.len(),
                    b.as_ptr() as *const c_void,
                    b.len(),
                    out.as_mut_ptr() as *mut c_void,
                    out_bytes,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if rc == 0 {
                Ok(out)
            } else {
                Err(format!(
                    "strix_npu_ctx_run failed: {}",
                    cbuf_to_string(&err)
                ))
            }
        }
    }

    impl Drop for Context {
        fn drop(&mut self) {
            unsafe { strix_npu_close(self.handle) };
        }
    }

    // detail/xrt_mem.h: XCL_BO_FLAGS_HOST_ONLY = (1 << 29).
    const XRT_BO_FLAGS_HOST_ONLY: u64 = 1 << 29;
    const XRT_BO_FLAGS_CACHEABLE: u64 = 1 << 24;
    // deprecated/xrt.h: enum xclBOSyncDirection.
    const XCL_BO_SYNC_BO_TO_DEVICE: c_int = 0;
    const XCL_BO_SYNC_BO_FROM_DEVICE: c_int = 1;

    fn cbuf_to_string(buf: &[c_char]) -> String {
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Open NPU device 0 via XRT, then close it. Proves the access layer works.
    pub fn probe() -> super::NpuProbe {
        // SAFETY: standard XRT C API; handle is opaque and only passed back to XRT.
        unsafe {
            let handle = xrtDeviceOpen(0);
            if handle.is_null() {
                return super::NpuProbe {
                    opened: false,
                    detail: "xrtDeviceOpen(0) returned null (no XRT device or busy)".into(),
                };
            }
            xrtDeviceClose(handle);
            super::NpuProbe {
                opened: true,
                detail: "xrtDeviceOpen(0) succeeded via libxrt_coreutil".into(),
            }
        }
    }

    /// Allocate a buffer object on the NPU, DMA a host pattern to the device and
    /// back, and verify it survives the round-trip. This exercises the XRT BO
    /// data path (alloc / map / write / sync-to-device / sync-from-device / read)
    /// — the host↔NPU memory movement a compute kernel would rely on — without
    /// needing an `.xclbin`. Tries a HOST_ONLY buffer first, then CACHEABLE.
    pub fn buffer_roundtrip(n_bytes: usize) -> super::NpuBufferTest {
        // SAFETY: all pointers come from XRT and are only handed back to XRT;
        // the mapped pointer is valid for `n_bytes` between alloc and free.
        unsafe {
            let dev = xrtDeviceOpen(0);
            if dev.is_null() {
                return super::NpuBufferTest::fail("xrtDeviceOpen(0) returned null");
            }

            let mut last = String::new();
            for (flag, label) in [
                (XRT_BO_FLAGS_HOST_ONLY, "HOST_ONLY"),
                (XRT_BO_FLAGS_CACHEABLE, "CACHEABLE"),
            ] {
                let bo = xrtBOAlloc(dev, n_bytes, flag, 0);
                if bo.is_null() {
                    last = format!("xrtBOAlloc({n_bytes}, {label}, grp=0) returned null");
                    continue;
                }

                // Build a deterministic pattern and copy it into the BO.
                let src: Vec<u8> = (0..n_bytes).map(|i| (i as u8) ^ 0xA5).collect();
                let mut ok = true;
                let detail;
                if xrtBOWrite(bo, src.as_ptr() as *const c_void, n_bytes, 0) != 0 {
                    ok = false;
                    detail = format!("xrtBOWrite failed ({label})");
                } else if xrtBOSync(bo, XCL_BO_SYNC_BO_TO_DEVICE, n_bytes, 0) != 0 {
                    ok = false;
                    detail = format!("xrtBOSync TO_DEVICE failed ({label})");
                } else if xrtBOSync(bo, XCL_BO_SYNC_BO_FROM_DEVICE, n_bytes, 0) != 0 {
                    ok = false;
                    detail = format!("xrtBOSync FROM_DEVICE failed ({label})");
                } else {
                    let mut dst = vec![0u8; n_bytes];
                    if xrtBORead(bo, dst.as_mut_ptr() as *mut c_void, n_bytes, 0) != 0 {
                        ok = false;
                        detail = format!("xrtBORead failed ({label})");
                    } else if dst == src {
                        // Also confirm the mapped pointer aliases the same data.
                        let mapped = xrtBOMap(bo);
                        let mapped_ok = !mapped.is_null()
                            && std::slice::from_raw_parts(mapped as *const u8, n_bytes) == &src[..];
                        detail = format!(
                            "{n_bytes} B round-trip OK via {label} BO (write→sync→sync→read{})",
                            if mapped_ok { ", map verified" } else { "" }
                        );
                    } else {
                        ok = false;
                        detail = format!("data mismatch after round-trip ({label})");
                    }
                }

                xrtBOFree(bo);
                if ok {
                    xrtDeviceClose(dev);
                    return super::NpuBufferTest {
                        ok: true,
                        detail,
                        bytes: n_bytes,
                    };
                }
                last = detail;
            }

            xrtDeviceClose(dev);
            super::NpuBufferTest::fail(&last)
        }
    }

    /// Exercise the xclbin load chain on the NPU via the C++ shim (the modern
    /// register_xclbin → hw_context flow). Returns the UUID string.
    pub fn load_probe(xclbin_path: &str) -> Result<String, String> {
        let cpath = CString::new(xclbin_path).map_err(|_| "path has interior NUL")?;
        let mut uuid = [0 as c_char; 64];
        let mut err = [0 as c_char; 512];
        let rc = unsafe {
            strix_npu_load_probe(
                cpath.as_ptr(),
                uuid.as_mut_ptr(),
                uuid.len(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc == 0 {
            Ok(cbuf_to_string(&uuid))
        } else {
            Err(format!("NPU load failed: {}", cbuf_to_string(&err)))
        }
    }

    /// Run a matmul-style NPU kernel from an `.xclbin` via the C++ shim, using
    /// the mlir-aie host ABI (opcode 3, instr BO + length, A/B/out BOs). Inputs/
    /// outputs are opaque bytes packed per the kernel's dtype.
    ///
    /// `kernel_name` is the xclbin's kernel (mlir-aie designs use `"MLIR_AIE"`).
    /// `instr` is the design's DPU sequence (from `insts_*.txt`/`.bin`).
    pub fn run_matmul(
        xclbin_path: &str,
        kernel_name: &str,
        instr: &[u32],
        a: &[u8],
        b: &[u8],
        out_bytes: usize,
    ) -> Result<Vec<u8>, String> {
        let cpath = CString::new(xclbin_path).map_err(|_| "xclbin path has interior NUL")?;
        let kname = CString::new(kernel_name).map_err(|_| "kernel name has interior NUL")?;
        let mut out = vec![0u8; out_bytes];
        let mut err = [0 as c_char; 512];
        let rc = unsafe {
            strix_npu_matmul(
                cpath.as_ptr(),
                kname.as_ptr(),
                instr.as_ptr(),
                instr.len(),
                a.as_ptr() as *const c_void,
                a.len(),
                b.as_ptr() as *const c_void,
                b.len(),
                out.as_mut_ptr() as *mut c_void,
                out_bytes,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc == 0 {
            Ok(out)
        } else {
            Err(format!("NPU matmul failed: {}", cbuf_to_string(&err)))
        }
    }

    /// Run a 2-buffer kernel (fused attention): `input` is the packed Q‖K‖V bytes,
    /// returns `out_bytes` of output.
    pub fn run_attn(
        xclbin_path: &str,
        kernel_name: &str,
        instr: &[u32],
        input: &[u8],
        out_bytes: usize,
    ) -> Result<Vec<u8>, String> {
        let cpath = CString::new(xclbin_path).map_err(|_| "xclbin path has interior NUL")?;
        let kname = CString::new(kernel_name).map_err(|_| "kernel name has interior NUL")?;
        let mut out = vec![0u8; out_bytes];
        let mut err = [0 as c_char; 512];
        let rc = unsafe {
            strix_npu_attn(
                cpath.as_ptr(),
                kname.as_ptr(),
                instr.as_ptr(),
                instr.len(),
                input.as_ptr() as *const c_void,
                input.len(),
                out.as_mut_ptr() as *mut c_void,
                out_bytes,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc == 0 {
            Ok(out)
        } else {
            Err(format!("NPU attn failed: {}", cbuf_to_string(&err)))
        }
    }
}

/// Parse an mlir-aie `insts_*.txt` (one 32-bit hex word per line) into words.
pub fn load_instr_txt(text: &str) -> Result<Vec<u32>, String> {
    text.split_whitespace()
        .map(|tok| {
            let t = tok.strip_prefix("0x").unwrap_or(tok);
            u32::from_str_radix(t, 16).map_err(|e| format!("bad instr word {tok:?}: {e}"))
        })
        .collect()
}

/// Read an mlir-aie `insts_*.bin` (little-endian 32-bit words) into words.
pub fn load_instr_bin(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "instr .bin length {} not a multiple of 4",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Result of the NPU buffer-object data-path test.
#[derive(Debug, Clone)]
pub struct NpuBufferTest {
    /// True if the host→device→host round-trip preserved the data.
    pub ok: bool,
    /// Human-readable detail.
    pub detail: String,
    /// Bytes moved.
    pub bytes: usize,
}

impl NpuBufferTest {
    fn fail(msg: &str) -> Self {
        NpuBufferTest {
            ok: false,
            detail: msg.to_string(),
            bytes: 0,
        }
    }
}

/// Run the NPU buffer-object data-path test (host↔NPU DMA round-trip).
///
/// Returns a clear "feature off" result when built without `ryzen-ai`.
pub fn buffer_roundtrip(n_bytes: usize) -> NpuBufferTest {
    #[cfg(feature = "ryzen-ai")]
    {
        xrt::buffer_roundtrip(n_bytes)
    }
    #[cfg(not(feature = "ryzen-ai"))]
    {
        let _ = n_bytes;
        NpuBufferTest::fail("built without `ryzen-ai` feature")
    }
}

/// Run a matmul-style kernel on the NPU from an `.xclbin` (mlir-aie host ABI:
/// opcode 3, instr BO + length, then A/B/output BOs). Inputs/outputs are opaque
/// bytes packed per the kernel's dtype. Returns the output bytes or an error.
pub fn run_matmul(
    xclbin_path: &str,
    kernel_name: &str,
    instr: &[u32],
    a: &[u8],
    b: &[u8],
    out_bytes: usize,
) -> Result<Vec<u8>, String> {
    #[cfg(feature = "ryzen-ai")]
    {
        xrt::run_matmul(xclbin_path, kernel_name, instr, a, b, out_bytes)
    }
    #[cfg(not(feature = "ryzen-ai"))]
    {
        let _ = (xclbin_path, kernel_name, instr, a, b, out_bytes);
        Err("built without `ryzen-ai` feature".into())
    }
}

/// Run a 2-buffer kernel (fused attention) on the NPU: one packed input → output.
pub fn run_attn(
    xclbin_path: &str,
    kernel_name: &str,
    instr: &[u32],
    input: &[u8],
    out_bytes: usize,
) -> Result<Vec<u8>, String> {
    #[cfg(feature = "ryzen-ai")]
    {
        xrt::run_attn(xclbin_path, kernel_name, instr, input, out_bytes)
    }
    #[cfg(not(feature = "ryzen-ai"))]
    {
        let _ = (xclbin_path, kernel_name, instr, input, out_bytes);
        Err("built without `ryzen-ai` feature".into())
    }
}

/// A persistent NPU kernel context (load the xclbin once, run a matmul per call).
#[cfg(feature = "ryzen-ai")]
pub use xrt::Context as NpuContext;

/// A fixed-shape hybrid NPU GEMM: staged weights, zero-copy host I/O, async run.
#[cfg(feature = "ryzen-ai")]
pub use xrt::Gemm as NpuGemm;

/// Exercise the xclbin load chain (alloc → UUID → load to device) for a real
/// `.xclbin`, proving the NPU host loader works. Returns the UUID hex or error.
pub fn load_probe(xclbin_path: &str) -> Result<String, String> {
    #[cfg(feature = "ryzen-ai")]
    {
        xrt::load_probe(xclbin_path)
    }
    #[cfg(not(feature = "ryzen-ai"))]
    {
        let _ = xclbin_path;
        Err("built without `ryzen-ai` feature".into())
    }
}

#[cfg(all(test, feature = "ryzen-ai"))]
mod tests {
    /// Locate the single_core matmul build dir's xclbin + insts (override with
    /// STRIX_NPU_BUILD). Returns (xclbin_path, insts_path).
    fn find_matmul_artifacts() -> Option<(String, String)> {
        let base = std::env::var("STRIX_NPU_BUILD").unwrap_or_else(|_| {
            "external/mlir-aie/programming_examples/basic/matrix_multiplication/single_core/build"
                .to_string()
        });
        // Try a few cwd-relative roots (tests run from the crate dir).
        let roots = [base.clone(), format!("../../{base}"), format!("../{base}")];
        for r in roots {
            let dir = std::path::Path::new(&r);
            if !dir.is_dir() {
                continue;
            }
            let mut xclbin = None;
            let mut insts = None;
            for e in std::fs::read_dir(dir).ok()?.flatten() {
                let p = e.path();
                let name = p.file_name()?.to_string_lossy().into_owned();
                if name.ends_with(".xclbin") {
                    xclbin = Some(p.to_string_lossy().into_owned());
                } else if name.starts_with("insts")
                    && (name.ends_with(".txt") || name.ends_with(".bin"))
                {
                    insts = Some(p.to_string_lossy().into_owned());
                }
            }
            if let (Some(x), Some(i)) = (xclbin, insts) {
                return Some((x, i));
            }
        }
        None
    }

    #[test]
    #[ignore = "requires the NPU + a built matmul xclbin (run external/setup-npu-toolchain.sh)"]
    fn npu_matmul_matches_cpu() {
        let (xclbin, insts_path) = match find_matmul_artifacts() {
            Some(v) => v,
            None => {
                eprintln!("no matmul xclbin/insts found — run external/setup-npu-toolchain.sh");
                return;
            }
        };
        eprintln!("xclbin: {xclbin}\ninsts:  {insts_path}");

        // single_core defaults: M=K=N=512, A/B i16 row-major, C i32 row-major.
        let (m, k, n) = (512usize, 512, 512);
        // Recent mlir-aie emits the insts as a binary blob (LE u32) even with a
        // .txt name; parse hex text if it happens to be UTF-8, else binary.
        let raw = std::fs::read(&insts_path).expect("read insts");
        let instr = match std::str::from_utf8(&raw) {
            Ok(t)
                if t.lines()
                    .next()
                    .map(|l| l.trim().chars().all(|c| c.is_ascii_hexdigit() || c == 'x'))
                    .unwrap_or(false) =>
            {
                super::load_instr_txt(t).expect("parse insts txt")
            }
            _ => super::load_instr_bin(&raw).expect("parse insts bin"),
        };
        eprintln!("insts: {} words", instr.len());

        // Small values so the K=512 i32 accumulation can't overflow.
        let mut seed = 0x1234_5678u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as i64 % 7 - 3) as i16
        };
        let a: Vec<i16> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<i16> = (0..k * n).map(|_| rnd()).collect();

        // CPU reference.
        let mut cpu = vec![0i32; m * n];
        for i in 0..m {
            for kk in 0..k {
                let av = a[i * k + kk] as i32;
                if av == 0 {
                    continue;
                }
                for j in 0..n {
                    cpu[i * n + j] += av * b[kk * n + j] as i32;
                }
            }
        }

        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out_bytes = m * n * 4;

        let out = super::run_matmul(&xclbin, "MLIR_AIE", &instr, &a_bytes, &b_bytes, out_bytes)
            .expect("npu run_matmul");
        let got: Vec<i32> = out
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut mism = 0usize;
        let mut first = None;
        for i in 0..m * n {
            if got[i] != cpu[i] {
                mism += 1;
                if first.is_none() {
                    first = Some((i, cpu[i], got[i]));
                }
            }
        }
        if let Some((i, c, g)) = first {
            eprintln!(
                "first mismatch @ {i}: cpu={c} npu={g}  ({mism}/{} differ)",
                m * n
            );
        } else {
            eprintln!("NPU matmul EXACT match vs CPU ({}x{}x{})", m, k, n);
        }
        assert_eq!(mism, 0, "NPU matmul disagreed with CPU in {mism} elements");
    }

    #[test]
    #[ignore = "requires the NPU + a built matmul xclbin"]
    fn npu_matmul_throughput() {
        let Some((xclbin, insts_path)) = find_matmul_artifacts() else {
            eprintln!("no matmul xclbin — run external/setup-npu-toolchain.sh");
            return;
        };
        let (m, k, n) = (512usize, 512, 512);
        let raw = std::fs::read(&insts_path).expect("read insts");
        let instr = super::load_instr_bin(&raw).expect("parse insts");
        let a = vec![1u8; m * k * 2];
        let b = vec![1u8; k * n * 2];
        let out_bytes = m * n * 4;

        let ctx = super::NpuContext::open(&xclbin, "MLIR_AIE").expect("open ctx");
        // warm
        ctx.run_matmul(&instr, &a, &b, out_bytes).expect("warm run");
        let iters = 100;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            ctx.run_matmul(&instr, &a, &b, out_bytes).expect("run");
        }
        let per = t.elapsed().as_secs_f64() / iters as f64;
        let gflops = 2.0 * (m * k * n) as f64 / per / 1e9;
        eprintln!(
            "NPU {m}x{k}x{n} i16 matmul: {:.3} ms/run  {:.1} GFLOP/s (single AIE core, incl per-run BO alloc+DMA)",
            per * 1e3,
            gflops
        );
    }

    #[test]
    #[ignore = "requires the XDNA NPU + XRT"]
    fn load_chain_on_real_xclbin() {
        // Strix (17f0) validate xclbin + an AIE2P overlay: both real, device-
        // matched xclbins. Proves alloc → UUID → load-to-device works on the NPU.
        let candidates = [
            "/opt/opt/xilinx/xrt/amdxdna/bins/17f0_10/validate.xclbin",
            "/usr/usr/share/amdxdna/onnx_rt/AMD_AIE2P_4x4_Overlay.xclbin",
            "/usr/usr/share/amdxdna/onnx_rt/1x4.xclbin",
        ];
        let mut any_ok = false;
        for p in candidates {
            if !std::path::Path::new(p).exists() {
                eprintln!("skip (missing): {p}");
                continue;
            }
            match super::load_probe(p) {
                Ok(uuid) => {
                    eprintln!("LOADED {p}\n  uuid={uuid}");
                    any_ok = true;
                }
                Err(e) => eprintln!("load failed {p}: {e}"),
            }
        }
        assert!(any_ok, "no xclbin loaded onto the NPU");
    }
}

/// Probe the NPU: open it via XRT if the `ryzen-ai` feature is built.
pub fn probe() -> NpuProbe {
    #[cfg(feature = "ryzen-ai")]
    {
        xrt::probe()
    }
    #[cfg(not(feature = "ryzen-ai"))]
    {
        NpuProbe {
            opened: false,
            detail: "built without `ryzen-ai` feature (rebuild with --features ryzen-ai)".into(),
        }
    }
}
