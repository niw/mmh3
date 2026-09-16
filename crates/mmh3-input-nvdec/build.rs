fn main() {
    println!("cargo:rerun-if-changed=native");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        // The crate is empty off Linux, so the workspace builds without a CUDA toolchain.
        return;
    }

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include("native")
        .file("native/nvdec.cpp")
        .compile("mmh3_nvdec");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dl");
    }
}
