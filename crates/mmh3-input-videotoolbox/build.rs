fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        // Keep Linux CUDA workspace builds working. The crate is empty off macOS.
        return;
    }

    mmh3_swift_build::Build::new("MMH3VideoToolboxDecoder", "native/runtime.swift")
        .frameworks(["Foundation", "VideoToolbox", "CoreMedia", "CoreVideo"])
        .compile("mmh3_videotoolbox_decoder");
}
