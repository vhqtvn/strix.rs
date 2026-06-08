//! Compile the XRT C++ shim and link the system XRT — only under `ryzen-ai`.
//! The default build has no NPU/XRT dependency.

fn main() {
    if std::env::var("CARGO_FEATURE_RYZEN_AI").is_err() {
        return;
    }
    let xrt = std::env::var("XILINX_XRT").unwrap_or_else(|_| "/usr".to_string());
    // The NPU run path needs the XRT C++ API (hw_context); compile the shim.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/npu_shim.cpp")
        .include(format!("{xrt}/include"))
        .flag_if_supported("-Wno-unused-parameter")
        .compile("strix_npu_shim");
    println!("cargo:rustc-link-search=native={xrt}/lib");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-lib=dylib=uuid"); // xrt::uuid::to_string → libuuid
    println!("cargo:rerun-if-changed=src/npu_shim.cpp");
    println!("cargo:rerun-if-env-changed=XILINX_XRT");
}
