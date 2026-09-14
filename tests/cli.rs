use std::process::{Command, Output};

const MMH3: &str = env!("CARGO_BIN_EXE_mmh3");
const TOOLS: &str = env!("CARGO_BIN_EXE_mmh3-tools");

fn run(binary: &str, arguments: &[&str]) -> Output {
    Command::new(binary).args(arguments).output().unwrap()
}

#[test]
fn each_binary_lists_its_own_commands() {
    let generation = run(MMH3, &[]);
    assert_eq!(generation.status.code(), Some(2));
    let usage = String::from_utf8(generation.stderr).unwrap();
    assert!(usage.contains("mmh3 generate"));
    assert!(!usage.contains("inspect"));
    assert!(!usage.contains("mmh3-tools"));

    let tools = run(TOOLS, &[]);
    assert_eq!(tools.status.code(), Some(2));
    let usage = String::from_utf8(tools.stderr).unwrap();
    for command in ["inspect", "device", "bench", "check"] {
        assert!(usage.contains(&format!("mmh3-tools {command}")));
    }
    assert!(!usage.contains("mmh3 generate"));
}

#[test]
fn commands_belong_to_only_one_binary() {
    for command in ["inspect", "device", "bench", "check"] {
        assert_eq!(run(MMH3, &[command]).status.code(), Some(2));
    }
    assert_eq!(run(TOOLS, &["generate"]).status.code(), Some(2));
}

#[test]
fn tools_inspects_a_checkpoint() {
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dit_tiny.safetensors");
    let output = run(TOOLS, &["inspect", fixture, "--all"]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains(fixture));
    assert!(text.contains("tensors   "));
    assert!(text.contains("F32"));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn gpu_commands_explain_the_missing_backend() {
    for (binary, command) in [(MMH3, "generate"), (TOOLS, "device"), (TOOLS, "bench"), (TOOLS, "check")] {
        let output = run(binary, &[command]);
        assert_eq!(output.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&output.stderr).contains("Rebuild with --features cuda"));
    }
}

#[cfg(feature = "cuda")]
#[test]
fn argument_errors_use_the_correct_binary_usage() {
    for (binary, arguments, expected, absent) in [
        (MMH3, &["generate", "--unknown", "1"][..], "mmh3 generate", "mmh3-tools"),
        (TOOLS, &["bench", "attention", "--unknown", "1"][..], "mmh3-tools bench", "mmh3 generate"),
        (TOOLS, &["check", "dit", "--unknown", "1"][..], "mmh3-tools check", "mmh3 generate"),
    ] {
        let output = run(binary, arguments);
        assert_eq!(output.status.code(), Some(1));
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains(expected), "{error}");
        assert!(!error.contains(absent), "{error}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn invalid_output_format_is_rejected_before_loading_models() {
    let output = run(MMH3, &["generate", "--prompt", "test", "--out", "out.unknown"]);
    assert_eq!(output.status.code(), Some(1));
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("FILE.webm"), "{error}");
    assert!(error.contains("--ffmpeg"), "{error}");
}

#[cfg(feature = "cuda")]
#[test]
fn invalid_ffmpeg_template_is_rejected_before_loading_models() {
    let output = run(MMH3, &["generate", "--prompt", "test", "--out", "out.mp4", "--ffmpeg", "-i", "{video}"]);
    assert_eq!(output.status.code(), Some(1));
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("templates must include {out}"), "{error}");
}

#[cfg(all(feature = "cuda", not(feature = "mp4")))]
#[test]
fn mp4_feature_error_precedes_model_loading() {
    let output = run(MMH3, &["generate", "--prompt", "test", "--out", "out.mp4"]);
    assert_eq!(output.status.code(), Some(1));
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("--features mp4"), "{error}");
    assert!(error.contains("--ffmpeg"), "{error}");
}

#[cfg(all(feature = "cuda", not(feature = "webm")))]
#[test]
fn webm_feature_error_precedes_model_loading() {
    let output = run(MMH3, &["generate", "--prompt", "test", "--out", "out.webm"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--features webm"));
}
