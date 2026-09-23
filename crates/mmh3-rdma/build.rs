use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        // The crate is empty off Linux, so the workspace builds without libibverbs.
        return;
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let source = manifest.join("src/rdma.c");
    println!("cargo:rerun-if-changed={}", source.display());

    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rustc-check-cfg=cfg(mmh3_rdma_without_verbs)");

    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    if !has_verbs(&compiler) {
        // Without the headers the crate builds with no device to open, and the socket carries
        // everything. Installing libibverbs later needs `cargo clean -p mmh3-rdma` to be noticed.
        println!("cargo:warning=infiniband/verbs.h was not found, so mmh3 builds without RDMA");
        println!("cargo:rustc-cfg=mmh3_rdma_without_verbs");
        return;
    }

    let object = out.join("rdma.o");
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

/// Whether the compiler finds the libibverbs headers.
fn has_verbs(compiler: &str) -> bool {
    let Ok(mut child) = Command::new(compiler)
        .args(["-E", "-x", "c", "-", "-o", "/dev/null"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    let written = child
        .stdin
        .take()
        .unwrap()
        .write_all(b"#include <infiniband/verbs.h>\n")
        .is_ok();
    child.wait().is_ok_and(|status| status.success()) && written
}
