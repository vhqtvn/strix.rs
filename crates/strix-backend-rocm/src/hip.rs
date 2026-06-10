//! Self-contained HIP compute context: device init, runtime kernel compilation
//! (hiprtc), device buffers, and kernel launch — the ROCm analogue of the ash
//! Vulkan context. HIP streams serialize dependent kernels in-order with low
//! overhead and no explicit cache-flush barriers (the Vulkan path's ~7ms tax),
//! which is the reason to try ROCm despite near-identical raw matmul bandwidth.

// The launch/load entry points take raw HIP device handles by value; they don't
// deref host pointers unsafely in a way the caller must guard beyond the FFI.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

use strix_core::error::{Result, StrixError};

use crate::ffi::*;

fn ck(err: hipError_t, ctx: &str) -> Result<()> {
    if err == 0 {
        return Ok(());
    }
    let msg = unsafe {
        let p = hipGetErrorString(err);
        if p.is_null() {
            format!("code {err}")
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    Err(StrixError::backend(format!("hip: {ctx}: {msg}")))
}

/// Compile a HIP C++ source string to a gfx code object via hiprtc.
pub fn compile(src: &str) -> Result<Vec<u8>> {
    unsafe {
        let csrc = CString::new(src).unwrap();
        let name = CString::new("strix.hip").unwrap();
        let mut prog: hiprtcProgram = ptr::null_mut();
        let r = hiprtcCreateProgram(
            &mut prog,
            csrc.as_ptr(),
            name.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        );
        if r != 0 {
            return Err(StrixError::backend(format!("hiprtc create: {r}")));
        }
        let arch = CString::new("--offload-arch=gfx1150").unwrap();
        let opts: [*const c_char; 1] = [arch.as_ptr()];
        let cr = hiprtcCompileProgram(prog, 1, opts.as_ptr());
        if cr != 0 {
            // Pull the compile log for diagnostics.
            let mut log_sz = 0usize;
            hiprtcGetProgramLogSize(prog, &mut log_sz);
            let mut log = vec![0u8; log_sz];
            if log_sz > 0 {
                hiprtcGetProgramLog(prog, log.as_mut_ptr() as *mut c_char);
            }
            let log = String::from_utf8_lossy(&log).into_owned();
            return Err(StrixError::backend(format!("hiprtc compile: {cr}\n{log}")));
        }
        let mut sz = 0usize;
        hiprtcGetCodeSize(prog, &mut sz);
        let mut code = vec![0u8; sz];
        let gr = hiprtcGetCode(prog, code.as_mut_ptr() as *mut c_char);
        if gr != 0 {
            return Err(StrixError::backend(format!("hiprtc getcode: {gr}")));
        }
        Ok(code)
    }
}

/// A device buffer (raw HIP allocation).
pub struct Dbuf {
    pub ptr: *mut c_void,
    pub bytes: usize,
}

impl Dbuf {
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        assert!(n <= self.bytes, "rocm: upload overruns buffer");
        ck(
            unsafe { hipMemcpy(self.ptr, data.as_ptr() as *const c_void, n, HIP_MEMCPY_HTOD) },
            "memcpy h2d",
        )
    }
    pub fn download<T: Copy + Default + Clone>(&self, n: usize) -> Result<Vec<T>> {
        let bytes = n * std::mem::size_of::<T>();
        assert!(bytes <= self.bytes, "rocm: download overruns buffer");
        let mut out = vec![T::default(); n];
        ck(
            unsafe {
                hipMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    bytes,
                    HIP_MEMCPY_DTOH,
                )
            },
            "memcpy d2h",
        )?;
        Ok(out)
    }
}

/// A loaded code module plus a stream.
pub struct HipGpu {
    pub stream: hipStream_t,
    name: String,
}

impl HipGpu {
    pub fn new() -> Result<Self> {
        unsafe {
            ck(hipInit(0), "init")?;
            let mut count = 0;
            ck(hipGetDeviceCount(&mut count), "device count")?;
            if count == 0 {
                return Err(StrixError::backend("hip: no device"));
            }
            ck(hipSetDevice(0), "set device")?;
            let mut buf = [0u8; 256];
            let _ = hipDeviceGetName(buf.as_mut_ptr() as *mut c_char, 256, 0);
            let name = CStr::from_ptr(buf.as_ptr() as *const c_char)
                .to_string_lossy()
                .into_owned();
            let mut stream: hipStream_t = ptr::null_mut();
            ck(hipStreamCreate(&mut stream), "stream create")?;
            Ok(Self { stream, name })
        }
    }

    pub fn adapter_name(&self) -> &str {
        &self.name
    }

    pub fn alloc(&self, bytes: usize) -> Result<Dbuf> {
        let bytes = bytes.max(4);
        let mut ptr: *mut c_void = ptr::null_mut();
        ck(unsafe { hipMalloc(&mut ptr, bytes) }, "malloc")?;
        Ok(Dbuf { ptr, bytes })
    }

    /// Upload at a byte offset into an existing buffer.
    pub fn upload_at<T: Copy>(&self, buf: &Dbuf, byte_off: usize, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        let dst = unsafe { (buf.ptr as *mut u8).add(byte_off) } as *mut c_void;
        let r = unsafe { hipMemcpy(dst, data.as_ptr() as *const c_void, n, HIP_MEMCPY_HTOD) };
        if r != 0 {
            return Err(StrixError::backend(format!("hipMemcpy upload_at: {r}")));
        }
        Ok(())
    }

    pub fn upload_new<T: Copy>(&self, data: &[T]) -> Result<Dbuf> {
        let b = self.alloc(std::mem::size_of_val(data))?;
        b.upload(data)?;
        Ok(b)
    }

    pub fn load_module(&self, code: &[u8]) -> Result<hipModule_t> {
        let mut module: hipModule_t = ptr::null_mut();
        ck(
            unsafe { hipModuleLoadData(&mut module, code.as_ptr() as *const c_void) },
            "module load",
        )?;
        Ok(module)
    }

    pub fn get_function(&self, module: hipModule_t, name: &str) -> Result<hipFunction_t> {
        let cname = CString::new(name).unwrap();
        let mut f: hipFunction_t = ptr::null_mut();
        ck(
            unsafe { hipModuleGetFunction(&mut f, module, cname.as_ptr()) },
            "get function",
        )?;
        Ok(f)
    }

    /// Launch `func` with `grid`×`block` threads. `params` are pointers to each
    /// kernel argument (device pointers / scalars), in order.
    pub fn launch(
        &self,
        func: hipFunction_t,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        ck(
            unsafe {
                hipModuleLaunchKernel(
                    func,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared,
                    self.stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launch",
        )
    }

    pub fn sync(&self) -> Result<()> {
        ck(unsafe { hipStreamSynchronize(self.stream) }, "stream sync")
    }

    /// Zero `bytes` of device memory at `ptr` on the stream (no sync).
    pub fn zero(&self, ptr: *mut c_void, bytes: usize) -> Result<()> {
        ck(
            unsafe { hipMemsetAsync(ptr, 0, bytes, self.stream) },
            "memset",
        )
    }

    /// Register host memory (e.g. an XRT BO's host map) for zero-copy iGPU
    /// access on unified memory; returns the device pointer kernels can use.
    pub fn register_host(&self, ptr: *mut c_void, bytes: usize) -> Result<*mut c_void> {
        ck(unsafe { hipHostRegister(ptr, bytes, 0) }, "host register")?;
        let mut dev: *mut c_void = ptr::null_mut();
        ck(
            unsafe { hipHostGetDevicePointer(&mut dev, ptr, 0) },
            "host get dev ptr",
        )?;
        Ok(dev)
    }
}
