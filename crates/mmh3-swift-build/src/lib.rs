//! Compiles a crate's Swift runtime into a static library for its build script.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

/// One Swift source compiled as its own module and linked with the frameworks it uses.
pub struct Build {
    module: String,
    source: PathBuf,
    frameworks: Vec<String>,
}

impl Build {
    pub fn new(module: &str, source: impl AsRef<Path>) -> Self {
        Self {
            module: module.to_owned(),
            source: source.as_ref().to_owned(),
            frameworks: Vec::new(),
        }
    }

    pub fn frameworks<'a>(mut self, frameworks: impl IntoIterator<Item = &'a str>) -> Self {
        self.frameworks
            .extend(frameworks.into_iter().map(str::to_owned));
        self
    }

    /// Compiles the source into `lib{library}.a` and tells Cargo to link it with the Swift runtime.
    pub fn compile(self, library: &str) {
        println!("cargo:rerun-if-changed={}", self.source.display());

        let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let object = out.join(format!("{library}.o"));
        let arch = match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
            "aarch64" => "arm64",
            "x86_64" => "x86_64",
            other => panic!("unsupported Swift target architecture {other}"),
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
                    &self.module,
                    "-target",
                ])
                .arg(format!("{arch}-apple-macosx15.0"))
                .arg("-module-cache-path")
                .arg(out.join("module-cache"))
                .arg(&self.source)
                .arg("-o")
                .arg(&object)
                .status()
                .expect("Xcode command-line tools are required")
                .success(),
            "compiling the Swift module {} failed",
            self.module
        );
        let archive = out.join(format!("lib{library}.a"));
        // ar adds to an existing archive, so start from an empty one.
        let _ = fs::remove_file(&archive);
        assert!(
            Command::new("xcrun")
                .args(["ar", "rcs"])
                .arg(&archive)
                .arg(&object)
                .status()
                .unwrap()
                .success()
        );
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static={library}");

        for framework in &self.frameworks {
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
        for library in [
            "swiftCore",
            "swiftFoundation",
            "swiftDispatch",
            "swiftObjectiveC",
            "swiftDarwin",
        ] {
            println!("cargo:rustc-link-lib={library}");
        }
    }
}
