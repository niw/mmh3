use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        // The crate is empty off Linux, so the workspace builds without libibverbs.
        return;
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let source = manifest.join("src/rdma.c");
    println!("cargo:rerun-if-changed={}", source.display());

    let object = out.join("rdma.o");
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    let status = Command::new(&compiler)
        .args([
            "-O2",
            "-fPIC",
            "-std=c11",
            "-D_POSIX_C_SOURCE=200809L",
            "-Wall",
            "-c",
        ])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("failed to run the C compiler");
    assert!(status.success(), "{compiler} failed");

    let library = out.join("libmmh3_rdma.a");
    let status = Command::new("ar")
        .arg("crs")
        .arg(&library)
        .arg(&object)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=mmh3_rdma");
    println!("cargo:rustc-link-lib=dylib=ibverbs");
}
