//! Link the system HIP runtime + hiprtc only when the `rocm` feature is on.
//! Default builds have no ROCm dependency.

fn main() {
    if std::env::var("CARGO_FEATURE_ROCM").is_err() {
        return;
    }
    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    println!("cargo:rustc-link-search=native={rocm}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hiprtc");
    // Embed the ROCm lib dir so the binary finds the .so at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm}/lib");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
}
