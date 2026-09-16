use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=native/runtime.swift");
    println!("cargo:rerun-if-changed=kernels/ops.metal");
    println!("cargo:rerun-if-changed=kernels/matmul.metal");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        // Keep Linux CUDA workspace builds working. The crate is empty off macOS.
        return;
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = out.join("runtime.o");
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "aarch64" => "arm64",
        "x86_64" => "x86_64",
        other => panic!("unsupported Metal target architecture {other}"),
    };

    assert!(
        Command::new("xcrun")
            .args([
                "--sdk",
                "macosx",
                "swiftc",
                "-O",
                "-parse-as-library",
                "-emit-object",
                "-module-name",
                "MMH3Metal",
                "-target"
            ])
            .arg(format!("{arch}-apple-macosx15.0"))
            .arg("-module-cache-path")
            .arg(out.join("module-cache"))
            .args(["native/runtime.swift", "-o"])
            .arg(&object)
            .status()
            .expect("Xcode command-line tools are required")
            .success(),
        "compiling the Swift Metal runtime failed"
    );
    assert!(
        Command::new("xcrun")
            .args(["ar", "rcs"])
            .arg(out.join("libmmh3_metal.a"))
            .arg(object)
            .status()
            .unwrap()
            .success()
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=mmh3_metal");

    for framework in ["Foundation", "Metal", "MetalPerformanceShaders"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }

    let sdk = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .unwrap();
    let sdk = String::from_utf8(sdk.stdout).unwrap();
    println!(
        "cargo:rustc-link-search=native={}/usr/lib/swift",
        sdk.trim()
    );
    println!("cargo:rustc-link-search=native=/usr/lib/swift");
    println!("cargo:rustc-link-lib=swiftCore");
    println!("cargo:rustc-link-lib=swiftFoundation");
    println!("cargo:rustc-link-lib=swiftDispatch");
    println!("cargo:rustc-link-lib=swiftObjectiveC");
    println!("cargo:rustc-link-lib=swiftDarwin");
}
