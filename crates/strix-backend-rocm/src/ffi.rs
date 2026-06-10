//! Minimal hand-written FFI to the system HIP runtime (`libamdhip64`) and the
//! runtime compiler (`libhiprtc`). Only the handful of entry points the decode
//! path needs — no bindgen, no heavy crate.

#![allow(non_camel_case_types)]
#![allow(dead_code)] // a few entry points are kept for completeness / future use

use std::os::raw::{c_char, c_int, c_uint, c_void};

pub type hipError_t = c_int; // hipSuccess == 0
pub type hiprtcResult = c_int; // HIPRTC_SUCCESS == 0
pub type hipStream_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipDeviceptr_t = *mut c_void;
pub type hiprtcProgram = *mut c_void;

// hipMemcpyKind
pub const HIP_MEMCPY_HTOD: c_int = 1;
pub const HIP_MEMCPY_DTOH: c_int = 2;

#[link(name = "amdhip64")]
extern "C" {
    // hipGraph capture/replay (token-record once, replay per token)
    pub fn hipStreamBeginCapture(stream: hipStream_t, mode: u32) -> i32;
    pub fn hipStreamEndCapture(stream: hipStream_t, graph: *mut *mut c_void) -> i32;
    pub fn hipGraphInstantiate(
        exec: *mut *mut c_void,
        graph: *mut c_void,
        err_node: *mut c_void,
        log: *mut c_void,
        sz: usize,
    ) -> i32;
    pub fn hipGraphLaunch(exec: *mut c_void, stream: hipStream_t) -> i32;
    pub fn hipGraphDestroy(graph: *mut c_void) -> i32;
}

extern "C" {
    pub fn hipInit(flags: c_uint) -> hipError_t;
    pub fn hipSetDevice(device: c_int) -> hipError_t;
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> hipError_t;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> hipError_t;
    pub fn hipMemsetAsync(
        dst: *mut c_void,
        value: c_int,
        size: usize,
        stream: hipStream_t,
    ) -> hipError_t;
    // Zero-copy: register host memory (e.g. an XRT BO's host map) so iGPU kernels
    // can read/write it directly over the unified-memory fabric.
    pub fn hipHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    pub fn hipHostUnregister(ptr: *mut c_void) -> hipError_t;
    pub fn hipHostGetDevicePointer(
        dev: *mut *mut c_void,
        host: *mut c_void,
        flags: c_uint,
    ) -> hipError_t;
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    pub fn hipModuleGetFunction(
        func: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_mem: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
    pub fn hipGetErrorString(err: hipError_t) -> *const c_char;
}

#[link(name = "hiprtc")]
extern "C" {
    pub fn hiprtcCreateProgram(
        prog: *mut hiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> hiprtcResult;
    pub fn hiprtcCompileProgram(
        prog: hiprtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> hiprtcResult;
    pub fn hiprtcGetCodeSize(prog: hiprtcProgram, size: *mut usize) -> hiprtcResult;
    pub fn hiprtcGetCode(prog: hiprtcProgram, code: *mut c_char) -> hiprtcResult;
    pub fn hiprtcGetProgramLogSize(prog: hiprtcProgram, size: *mut usize) -> hiprtcResult;
    pub fn hiprtcGetProgramLog(prog: hiprtcProgram, log: *mut c_char) -> hiprtcResult;
}
