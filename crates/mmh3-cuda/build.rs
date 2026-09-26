use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        // The crate is empty off Linux, so the workspace builds without a CUDA toolchain.
        return;
    }

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_owned());
    // NOTE: the Blackwell family of GB10 and the RTX 50 series, Hopper (H20, H100, H200) and Ada
    // (RTX 4090, L40S). Each gets machine code only. A PTX for sm_89 would let a newer card compile
    // it, but that card has TMA, so it would launch the TMA paths, which sm_89 code only traps in.
    let architectures =
        env::var("MMH3_CUDA_ARCH").unwrap_or_else(|_| "sm_120f,sm_90a,sm_89".to_owned());
    let manifest_directory = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let output_directory = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_directory = manifest_directory.join("kernels");

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=MMH3_CUDA_ARCH");
    println!("cargo:rerun-if-changed={}", kernel_directory.display());

    let mut sources: Vec<PathBuf> = fs::read_dir(&kernel_directory)
        .expect("kernels directory is missing")
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|extension| extension.to_str()),
                Some("cu" | "cuh")
            )
        })
        .collect();
    sources.sort();
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }
    let compiled_sources: Vec<&PathBuf> = sources
        .iter()
        .filter(|path| path.extension().is_some_and(|extension| extension == "cu"))
        .collect();

    let library = output_directory.join("libmmh3_cuda_kernels.a");
    let status = Command::new(PathBuf::from(&cuda_home).join("bin/nvcc"))
        .args([
            "--lib",
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-Xcompiler",
            "-fPIC",
        ])
        .args(architectures.split(',').map(|architecture| {
            let architecture = architecture.trim();
            let virtual_architecture = architecture.replacen("sm_", "compute_", 1);
            format!("-gencode=arch={virtual_architecture},code={architecture}")
        }))
        // Compiles the architectures in parallel.
        .args(["--threads", "0"])
        .arg("-o")
        .arg(&library)
        .args(&compiled_sources)
        .status()
        .expect("failed to run nvcc");
    assert!(status.success(), "nvcc failed");

    println!(
        "cargo:rustc-link-search=native={}",
        output_directory.display()
    );
    println!("cargo:rustc-link-lib=static=mmh3_cuda_kernels");
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    for library in ["stdc++", "dl", "rt", "pthread"] {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
}
