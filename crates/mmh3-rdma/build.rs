use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;

fn main() {
    println!("cargo:rerun-if-changed=src/rdma.c");
    println!("cargo:rustc-check-cfg=cfg(mmh3_rdma_without_verbs)");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        // The crate is empty off Linux, so the workspace builds without libibverbs.
        return;
    }

    let mut build = cc::Build::new();
    build
        .file("src/rdma.c")
        .std("c11")
        .define("_POSIX_C_SOURCE", "200809L")
        .opt_level(2);
    if !has_verbs(&build) {
        // Without the headers the crate builds with no device to open, and the socket carries
        // everything. The directories the compiler searches are watched, so installing
        // libibverbs later runs this again.
        for directory in include_directories(&build) {
            println!("cargo:rerun-if-changed={}", directory.display());
        }
        println!("cargo:warning=infiniband/verbs.h was not found, so mmh3 builds without RDMA");
        println!("cargo:rustc-cfg=mmh3_rdma_without_verbs");
        return;
    }

    build.compile("mmh3_rdma");
    println!("cargo:rustc-link-lib=dylib=ibverbs");
}

/// Whether the compiler finds the libibverbs headers.
fn has_verbs(build: &cc::Build) -> bool {
    let Ok(mut child) = build
        .get_compiler()
        .to_command()
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

/// The existing directories the compiler searches for `#include <...>`.
fn include_directories(build: &cc::Build) -> Vec<PathBuf> {
    let Ok(output) = build
        .get_compiler()
        .to_command()
        .args(["-E", "-v", "-x", "c", "/dev/null", "-o", "/dev/null"])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .skip_while(|line| !line.starts_with("#include <...> search starts here:"))
        .skip(1)
        .take_while(|line| !line.starts_with("End of search list."))
        .map(|line| PathBuf::from(line.trim()))
        .filter(|directory| directory.is_dir())
        .collect()
}
