use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_owned());
    let architecture = env::var("MMH3_CUDA_ARCH").unwrap_or_else(|_| "sm_120f".to_owned());
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
        .arg(format!("-arch={architecture}"))
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
