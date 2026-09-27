use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        // The crate is empty off Linux, so the workspace builds without a CUDA toolchain.
        return;
    }

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_owned());
    // NOTE: the Blackwell family of GB10 and the RTX 50 series, and Ada (RTX 4090, L40S). Each gets
    // machine code only. A PTX for sm_89 would let a newer card compile it, but that card has TMA,
    // so it would launch the TMA paths, which sm_89 code only traps in.
    let architectures = env::var("MMH3_CUDA_ARCH").unwrap_or_else(|_| "sm_120f,sm_89".to_owned());
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

    // NOTE: each source compiles on its own, without relocatable device code. `-rdc` would make
    // ptxas compile for a later device link, which changes register allocation and SASS.
    let nvcc = PathBuf::from(&cuda_home).join("bin/nvcc");
    let mut flags: Vec<String> = [
        "-c",
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-Xcompiler",
        "-fPIC",
    ]
    .map(str::to_owned)
    .to_vec();
    flags.extend(architectures.split(',').map(|architecture| {
        let architecture = architecture.trim();
        let virtual_architecture = architecture.replacen("sm_", "compute_", 1);
        format!("-gencode=arch={virtual_architecture},code={architecture}")
    }));
    // The architectures of one source compile in parallel too, since the longest source sets how
    // long the build takes.
    flags.extend(["--threads".to_owned(), "0".to_owned()]);

    // A source compiles again when its object is older than the source or a header it includes,
    // as nvcc listed them last time, or when the compiler or its flags change.
    let flags_path = output_directory.join("kernels.flags");
    let flags_record = format!("{}\n{}\n", nvcc.display(), flags.join("\n"));
    let flags_changed = fs::read_to_string(&flags_path).ok().as_deref() != Some(&flags_record);
    let objects: Vec<PathBuf> = compiled_sources
        .iter()
        .map(|source| {
            output_directory
                .join(source.file_stem().unwrap())
                .with_extension("o")
        })
        .collect();
    let stale: Vec<usize> = (0..compiled_sources.len())
        .filter(|&index| flags_changed || !is_up_to_date(&objects[index]))
        .collect();

    let job_count = env::var("NUM_JOBS")
        .ok()
        .and_then(|jobs| jobs.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, stale.len().max(1));
    let next_stale = AtomicUsize::new(0);
    thread::scope(|scope| {
        for _ in 0..job_count {
            scope.spawn(|| {
                while let Some(&index) = stale.get(next_stale.fetch_add(1, Ordering::Relaxed)) {
                    compile(&nvcc, &flags, compiled_sources[index], &objects[index]);
                }
            });
        }
    });
    fs::write(&flags_path, flags_record).unwrap();

    let library = output_directory.join("libmmh3_cuda_kernels.a");
    // ar adds to an existing archive, so a removed source would otherwise stay in it.
    let _ = fs::remove_file(&library);
    let status = Command::new("ar")
        .arg("crs")
        .arg(&library)
        .args(&objects)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

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

/// Compiles one kernel source into an object file, with the headers it includes listed next to it.
fn compile(nvcc: &Path, flags: &[String], source: &Path, object: &Path) {
    // A failed compilation must not leave an object that looks newer than its source.
    let _ = fs::remove_file(object);
    let output = Command::new(nvcc)
        .args(flags)
        .arg("-MD")
        .arg("-MF")
        .arg(object.with_extension("d"))
        .arg("-o")
        .arg(object)
        .arg(source)
        .output()
        .expect("failed to run nvcc");
    // One write per source keeps the messages of concurrent compilations apart.
    let mut messages = output.stdout;
    messages.extend_from_slice(&output.stderr);
    io::stderr().write_all(&messages).unwrap();
    if !output.status.success() {
        let _ = fs::remove_file(object);
        panic!("nvcc failed on {}", source.display());
    }
}

/// Whether an object is newer than every file its dependency list names.
fn is_up_to_date(object: &Path) -> bool {
    let Ok(object_time) = fs::metadata(object).and_then(|metadata| metadata.modified()) else {
        return false;
    };
    let Ok(dependencies) = fs::read_to_string(object.with_extension("d")) else {
        return false;
    };
    // The list is a make rule, `object: source header ...`, with lines continued by backslashes.
    let Some((_, prerequisites)) = dependencies.split_once(": ") else {
        return false;
    };
    prerequisites
        .split_whitespace()
        .filter(|prerequisite| *prerequisite != "\\")
        .all(|prerequisite| {
            fs::metadata(prerequisite)
                .and_then(|metadata| metadata.modified())
                .is_ok_and(|time| time < object_time)
        })
}
