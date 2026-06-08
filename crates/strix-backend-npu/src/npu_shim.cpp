// Thin C ABI over the XRT C++ API for the XDNA2 NPU. The NPU requires the
// hw_context flow (register_xclbin → hw_context → kernel(ctx,name) → run), which
// XRT exposes ONLY in C++ (no C API). This shim is the minimal bridge; Rust
// calls these extern "C" entry points. Built by build.rs under `ryzen-ai`.
//
// Host ABI matches mlir-aie's matrix_multiplication test.cpp:
//   run = kernel(opcode=3, bo_instr, instr_words, bo_a, bo_b, bo_out)

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_uuid.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_hw_context.h>

#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <exception>

static void set_err(char* errbuf, size_t errcap, const std::string& m) {
    if (errbuf && errcap) {
        std::strncpy(errbuf, m.c_str(), errcap - 1);
        errbuf[errcap - 1] = '\0';
    }
}

extern "C" {

// Load an xclbin onto the NPU via the hw_context flow (no kernel run). Proves
// the modern loader works. Returns 0 on success and writes the UUID string.
int strix_npu_load_probe(const char* xclbin_path, char* uuidbuf, size_t uuidcap,
                         char* errbuf, size_t errcap) {
    try {
        xrt::device device(0);
        auto xclbin = xrt::xclbin(std::string(xclbin_path));
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        std::string u = xclbin.get_uuid().to_string();
        if (uuidbuf && uuidcap) {
            std::strncpy(uuidbuf, u.c_str(), uuidcap - 1);
            uuidbuf[uuidcap - 1] = '\0';
        }
        return 0;
    } catch (const std::exception& e) {
        set_err(errbuf, errcap, e.what());
        return 1;
    } catch (...) {
        set_err(errbuf, errcap, "unknown C++ exception");
        return 2;
    }
}

// Run a matmul-style kernel. instr is the DPU sequence (instr_words u32s); a/b
// are inputs, out receives out_bytes. Returns 0 on success.
int strix_npu_matmul(const char* xclbin_path, const char* kernel_name,
                     const uint32_t* instr, size_t instr_words,
                     const void* a, size_t a_bytes,
                     const void* b, size_t b_bytes,
                     void* out, size_t out_bytes,
                     char* errbuf, size_t errcap) {
    try {
        xrt::device device(0);
        auto xclbin = xrt::xclbin(std::string(xclbin_path));
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        xrt::kernel kernel(context, std::string(kernel_name));

        auto bo_instr = xrt::bo(device, instr_words * sizeof(uint32_t),
                                xrt::bo::flags::cacheable, kernel.group_id(1));
        auto bo_a = xrt::bo(device, a_bytes, xrt::bo::flags::host_only, kernel.group_id(3));
        auto bo_b = xrt::bo(device, b_bytes, xrt::bo::flags::host_only, kernel.group_id(4));
        auto bo_out = xrt::bo(device, out_bytes, xrt::bo::flags::host_only, kernel.group_id(5));
        // mlir-aie matmul kernels also take a tmp BO (group 6) and a trace BO
        // (group 7); allocate minimal ones (trace disabled).
        auto bo_tmp1 = xrt::bo(device, 1, xrt::bo::flags::host_only, kernel.group_id(6));
        auto bo_trace = xrt::bo(device, 4, xrt::bo::flags::host_only, kernel.group_id(7));

        std::memcpy(bo_instr.map<void*>(), instr, instr_words * sizeof(uint32_t));
        std::memcpy(bo_a.map<void*>(), a, a_bytes);
        std::memcpy(bo_b.map<void*>(), b, b_bytes);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        unsigned int opcode = 3;
        auto run = kernel(opcode, bo_instr, instr_words, bo_a, bo_b, bo_out, bo_tmp1, bo_trace);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(out, bo_out.map<void*>(), out_bytes);
        return 0;
    } catch (const std::exception& e) {
        set_err(errbuf, errcap, e.what());
        return 1;
    } catch (...) {
        set_err(errbuf, errcap, "unknown C++ exception");
        return 2;
    }
}

} // extern "C"

// --- Persistent context: load the xclbin once, run many matmuls ---
// (xclbin register + hw_context + kernel open are expensive; for prefill we
// open once and run a GEMM per layer.)
struct NpuCtx {
    xrt::device device;
    xrt::hw_context context;
    xrt::kernel kernel;
    // Persistent buffer objects — allocating an xrt::bo is expensive (driver
    // call + mapping), so for repeated same-shape GEMMs (prefill: one per layer)
    // we allocate once and reuse, only re-copying/syncing the data each run.
    xrt::bo bo_instr, bo_a, bo_b, bo_out, bo_tmp1, bo_trace;
    size_t cap_instr = 0, cap_a = 0, cap_b = 0, cap_out = 0;
    bool inited = false;
};

extern "C" {

void* strix_npu_open(const char* xclbin_path, const char* kernel_name,
                     char* errbuf, size_t errcap) {
    try {
        xrt::device device(0);
        auto xclbin = xrt::xclbin(std::string(xclbin_path));
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        xrt::kernel kernel(context, std::string(kernel_name));
        return new NpuCtx{std::move(device), std::move(context), std::move(kernel)};
    } catch (const std::exception& e) {
        set_err(errbuf, errcap, e.what());
        return nullptr;
    } catch (...) {
        set_err(errbuf, errcap, "unknown C++ exception");
        return nullptr;
    }
}

int strix_npu_ctx_run(void* h, const uint32_t* instr, size_t instr_words,
                      const void* a, size_t a_bytes, const void* b, size_t b_bytes,
                      void* out, size_t out_bytes, char* errbuf, size_t errcap) {
    auto* ctx = static_cast<NpuCtx*>(h);
    try {
        auto& kernel = ctx->kernel;
        auto& device = ctx->device;
        const size_t ib = instr_words * sizeof(uint32_t);
        // (Re)allocate persistent BOs only when first used or when a buffer must
        // grow; same-shape repeated calls reuse them (saves the alloc per run).
        if (ctx->cap_instr < ib) {
            ctx->bo_instr = xrt::bo(device, ib, xrt::bo::flags::cacheable, kernel.group_id(1));
            ctx->cap_instr = ib;
        }
        if (ctx->cap_a < a_bytes) {
            ctx->bo_a = xrt::bo(device, a_bytes, xrt::bo::flags::host_only, kernel.group_id(3));
            ctx->cap_a = a_bytes;
        }
        if (ctx->cap_b < b_bytes) {
            ctx->bo_b = xrt::bo(device, b_bytes, xrt::bo::flags::host_only, kernel.group_id(4));
            ctx->cap_b = b_bytes;
        }
        if (ctx->cap_out < out_bytes) {
            ctx->bo_out = xrt::bo(device, out_bytes, xrt::bo::flags::host_only, kernel.group_id(5));
            ctx->cap_out = out_bytes;
        }
        if (!ctx->inited) {
            ctx->bo_tmp1 = xrt::bo(device, 1, xrt::bo::flags::host_only, kernel.group_id(6));
            ctx->bo_trace = xrt::bo(device, 4, xrt::bo::flags::host_only, kernel.group_id(7));
            ctx->inited = true;
        }
        std::memcpy(ctx->bo_instr.map<void*>(), instr, ib);
        std::memcpy(ctx->bo_a.map<void*>(), a, a_bytes);
        std::memcpy(ctx->bo_b.map<void*>(), b, b_bytes);
        ctx->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        ctx->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        ctx->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        unsigned int opcode = 3;
        auto run = kernel(opcode, ctx->bo_instr, instr_words, ctx->bo_a, ctx->bo_b,
                          ctx->bo_out, ctx->bo_tmp1, ctx->bo_trace);
        run.wait();
        ctx->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(out, ctx->bo_out.map<void*>(), out_bytes);
        return 0;
    } catch (const std::exception& e) {
        set_err(errbuf, errcap, e.what());
        return 1;
    } catch (...) {
        set_err(errbuf, errcap, "unknown C++ exception");
        return 2;
    }
}

void strix_npu_close(void* h) { delete static_cast<NpuCtx*>(h); }

// --- Hybrid GEMM: staged per-weight BOs + zero-copy host I/O + async run ---
// For NPU/iGPU concurrent prefill. The xclbin is fixed-shape (M,K,N). Weights
// (B = Wᵀ int8 [K,N]) are staged ONCE into device BOs. The A (activation int8
// [M,K]) and out (int32 [M,N]) BOs are host_only; their host pointers are
// returned so the caller (iGPU/HIP) can hipHostRegister them and read/write
// directly (zero-copy on unified LPDDR5x). start()/wait() are split so the NPU
// runs concurrently with the iGPU.
struct NpuGemm {
    xrt::device device;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo bo_instr, bo_a, bo_out, bo_tmp1, bo_trace;
    std::vector<xrt::bo> weights;
    uint32_t instr_words = 0;
    xrt::run run;
    bool running = false;
};

void* strix_npu_gemm_open(const char* xclbin_path, const char* kernel_name,
                          const uint32_t* instr, size_t instr_words,
                          size_t a_cap, size_t out_cap,
                          void** a_host, void** out_host, char* errbuf, size_t errcap) {
    try {
        xrt::device device(0);
        auto xclbin = xrt::xclbin(std::string(xclbin_path));
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        xrt::kernel kernel(context, std::string(kernel_name));
        auto* g = new NpuGemm{std::move(device), std::move(context), std::move(kernel)};
        const size_t ib = instr_words * sizeof(uint32_t);
        g->bo_instr = xrt::bo(g->device, ib, xrt::bo::flags::cacheable, g->kernel.group_id(1));
        g->bo_a = xrt::bo(g->device, a_cap, xrt::bo::flags::host_only, g->kernel.group_id(3));
        g->bo_out = xrt::bo(g->device, out_cap, xrt::bo::flags::host_only, g->kernel.group_id(5));
        g->bo_tmp1 = xrt::bo(g->device, 1, xrt::bo::flags::host_only, g->kernel.group_id(6));
        g->bo_trace = xrt::bo(g->device, 4, xrt::bo::flags::host_only, g->kernel.group_id(7));
        std::memcpy(g->bo_instr.map<void*>(), instr, ib);
        g->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        g->instr_words = (uint32_t)instr_words;
        *a_host = g->bo_a.map<void*>();
        *out_host = g->bo_out.map<void*>();
        return g;
    } catch (const std::exception& e) { set_err(errbuf, errcap, e.what()); return nullptr; }
    catch (...) { set_err(errbuf, errcap, "unknown C++ exception"); return nullptr; }
}

// Stage a weight (B int8 [K,N], b_bytes). Returns its index, or -1 on error.
int strix_npu_gemm_stage(void* h, const void* b, size_t b_bytes, char* errbuf, size_t errcap) {
    auto* g = static_cast<NpuGemm*>(h);
    try {
        auto bo = xrt::bo(g->device, b_bytes, xrt::bo::flags::host_only, g->kernel.group_id(4));
        std::memcpy(bo.map<void*>(), b, b_bytes);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        g->weights.push_back(std::move(bo));
        return (int)g->weights.size() - 1;
    } catch (const std::exception& e) { set_err(errbuf, errcap, e.what()); return -1; }
    catch (...) { set_err(errbuf, errcap, "unknown C++ exception"); return -1; }
}

// Start the GEMM with staged weight `wid`. The caller has already written the
// activation int8 into *a_host. Non-blocking (returns after enqueue).
int strix_npu_gemm_start(void* h, int wid, char* errbuf, size_t errcap) {
    auto* g = static_cast<NpuGemm*>(h);
    try {
        // host_only BOs live in unified LPDDR5x; on this coherent APU the cache
        // sync may be redundant (the iGPU already wrote a_host via hipHostRegister
        // + a stream sync). STRIX_NPU_NOSYNC skips it to cut per-offload latency.
        static const bool nosync = std::getenv("STRIX_NPU_NOSYNC") != nullptr;
        if (!nosync) g->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        g->run = g->kernel(3u, g->bo_instr, g->instr_words, g->bo_a, g->weights[(size_t)wid],
                           g->bo_out, g->bo_tmp1, g->bo_trace);
        g->running = true;
        return 0;
    } catch (const std::exception& e) { set_err(errbuf, errcap, e.what()); return 1; }
    catch (...) { set_err(errbuf, errcap, "unknown C++ exception"); return 2; }
}

// Wait for the GEMM and sync the output BO host-side (caller reads *out_host).
int strix_npu_gemm_wait(void* h, char* errbuf, size_t errcap) {
    auto* g = static_cast<NpuGemm*>(h);
    try {
        if (g->running) { g->run.wait(); g->running = false; }
        static const bool nosync = std::getenv("STRIX_NPU_NOSYNC") != nullptr;
        if (!nosync) g->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return 0;
    } catch (const std::exception& e) { set_err(errbuf, errcap, e.what()); return 1; }
    catch (...) { set_err(errbuf, errcap, "unknown C++ exception"); return 2; }
}

void strix_npu_gemm_close(void* h) { delete static_cast<NpuGemm*>(h); }

} // extern "C"
