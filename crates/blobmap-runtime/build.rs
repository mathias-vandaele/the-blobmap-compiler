//! With the `cuda` feature, compiles the research kernels (`cu/research/sweep.cu`, unmodified)
//! with nvcc into a static library. Without it, nothing: the crate needs no CUDA toolkit.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cu/research/sweep.cu");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let cuda = PathBuf::from(std::env::var("CUDA_PATH").unwrap_or_else(|_| "/opt/cuda".into()));
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let o = out.join("sweep.o");
    let s = Command::new(cuda.join("bin/nvcc"))
        .args(["-O3", "-std=c++17", "-arch=sm_89", "-Xcompiler", "-fPIC", "-lineinfo", "-c", "cu/research/sweep.cu", "-o"])
        .arg(&o)
        .status()
        .expect("nvcc");
    assert!(s.success(), "nvcc failed");
    let s = Command::new("ar").arg("crs").arg(out.join("libcchgpu.a")).arg(&o).status().expect("ar");
    assert!(s.success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=cchgpu");
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
