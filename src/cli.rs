//! Shared option parsing and command exit handling.

use std::collections::HashMap;
use std::error::Error;
use std::process::ExitCode;

/// Parses `--name value` pairs, rejecting names outside `allowed`. A name in `alone` is a whole
/// option by itself and takes nothing after it, so what it says is that it was given at all.
pub fn parse_options<'a>(
    arguments: &'a [String],
    allowed: &[&str],
    alone: &[&str],
    usage: &str,
) -> Result<HashMap<&'a str, &'a str>, Box<dyn Error>> {
    let mut options = HashMap::new();
    let mut remaining = arguments.iter().peekable();
    while let Some(argument) = remaining.next() {
        let name = argument
            .strip_prefix("--")
            .filter(|name| allowed.contains(name) || alone.contains(name))
            .ok_or(usage)?;
        if alone.contains(&name) {
            options.insert(name, "");
            continue;
        }
        let value = remaining
            .next_if(|value| !value.starts_with("--"))
            .ok_or_else(|| format!("--{name} needs a value"))?;
        options.insert(name, value.as_str());
    }
    Ok(options)
}

/// One machine a run was told to borrow: the address of a `--worker`, or None for the
/// `--local-worker` this process starts for itself, with the units of the `--worker-units` that
/// followed it. No units means every one the machine can serve.
pub struct Borrowed<'a> {
    pub address: Option<&'a str>,
    pub units: Option<&'a str>,
}

/// The machines of `--worker` and `--local-worker`, in the order they were given, which is the
/// order their ranks are numbered in. A `--worker-units` belongs to the machine before it, and
/// one before any machine, or a second one for the same machine, is an error.
pub fn borrowed(arguments: &[String]) -> Result<Vec<Borrowed<'_>>, Box<dyn Error>> {
    let mut borrowed: Vec<Borrowed> = Vec::new();
    let mut remaining = arguments.iter().peekable();
    while let Some(argument) = remaining.next() {
        match argument.strip_prefix("--") {
            Some("worker") => borrowed.push(Borrowed {
                address: Some(remaining.next().ok_or("--worker needs an address")?),
                units: None,
            }),
            Some("local-worker") => borrowed.push(Borrowed {
                address: None,
                units: None,
            }),
            Some("worker-units") => {
                let units = remaining.next().ok_or("--worker-units needs its units")?;
                let machine = borrowed
                    .last_mut()
                    .ok_or("--worker-units belongs to the --worker or --local-worker before it")?;
                if machine.units.is_some() {
                    return Err("one --worker-units for each machine".into());
                }
                machine.units = Some(units);
            }
            _ => {}
        }
    }
    Ok(borrowed)
}

/// Every value of `--name` in `arguments` in their order, for an option that may repeat. Parse the
/// arguments with `parse_options` first, which rejects values that start with `--`.
pub fn option_values<'a>(arguments: &'a [String], name: &str) -> Vec<&'a str> {
    arguments
        .windows(2)
        .filter(|pair| pair[0].strip_prefix("--") == Some(name))
        .map(|pair| pair[1].as_str())
        .collect()
}

pub fn option_number(
    options: &HashMap<&str, &str>,
    name: &str,
    default: usize,
) -> Result<usize, Box<dyn Error>> {
    match options.get(name) {
        Some(value) => Ok(value
            .replace('_', "")
            .parse()
            .map_err(|_| format!("--{name} must be a number"))?),
        None => Ok(default),
    }
}

pub fn option_float(
    options: &HashMap<&str, &str>,
    name: &str,
    default: f32,
) -> Result<f32, Box<dyn Error>> {
    let value = match options.get(name) {
        Some(value) => value
            .parse::<f32>()
            .map_err(|_| format!("--{name} must be a number"))?,
        None => default,
    };
    if !value.is_finite() {
        return Err(format!("--{name} must be a finite number").into());
    }
    Ok(value)
}

/// `--ffmpeg` consumes the remaining arguments verbatim, including repeated options.
/// Put all mmh3 options before it. An empty tail selects ffmpeg's default recipe.
pub fn split_ffmpeg_arguments(arguments: &[String]) -> (&[String], Option<&[String]>) {
    match arguments.iter().position(|arg| arg == "--ffmpeg") {
        Some(index) => (&arguments[..index], Some(&arguments[index + 1..])),
        None => (arguments, None),
    }
}

/// Prints a command error and maps its result to a process exit code.
pub fn exit_code(result: Result<(), Box<dyn Error>>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const USAGE: &str = "test usage";

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn ffmpeg_tail_is_not_parsed_as_mmh3_options() {
        let args = arguments(&[
            "--out",
            "a b.mp4",
            "--ffmpeg",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-vf",
            "scale=640:-2",
        ]);
        let (generation, ffmpeg) = split_ffmpeg_arguments(&args);
        assert_eq!(generation, &args[..2]);
        assert_eq!(ffmpeg, Some(&args[3..]));
        let args = arguments(&["--ffmpeg"]);
        assert_eq!(
            split_ffmpeg_arguments(&args),
            (&args[..0], Some(&args[1..]))
        );
        let args = arguments(&["--out", "out.webm"]);
        assert_eq!(split_ffmpeg_arguments(&args), (args.as_slice(), None));
    }

    #[test]
    fn parses_values_and_numeric_defaults() {
        let arguments = arguments(&[
            "--tokens",
            "38_710",
            "--lora-strength",
            "-0.5",
            "--prompt",
            "a rainy street",
        ]);
        let options = parse_options(
            &arguments,
            &["tokens", "lora-strength", "prompt"],
            &[],
            USAGE,
        )
        .unwrap();
        assert_eq!(options["prompt"], "a rainy street");
        assert_eq!(option_number(&options, "tokens", 1).unwrap(), 38_710);
        assert_eq!(option_number(&options, "iterations", 10).unwrap(), 10);
        assert_eq!(option_float(&options, "lora-strength", 1.0).unwrap(), -0.5);
        assert_eq!(option_float(&options, "sparse-tau", 1.3).unwrap(), 1.3);
    }

    #[test]
    fn keeps_the_last_value_of_repeated_options() {
        let arguments = arguments(&["--steps", "20", "--steps", "4"]);
        let options = parse_options(&arguments, &["steps"], &[], USAGE).unwrap();
        assert_eq!(option_number(&options, "steps", 1).unwrap(), 4);
    }

    #[test]
    fn collects_every_value_of_a_repeated_option() {
        let arguments = arguments(&[
            "--reference",
            "cat.png",
            "--seed",
            "3",
            "--reference",
            "dog.jpg",
        ]);
        parse_options(&arguments, &["reference", "seed"], &[], USAGE).unwrap();
        assert_eq!(
            option_values(&arguments, "reference"),
            ["cat.png", "dog.jpg"]
        );
        assert!(option_values(&arguments, "prompt").is_empty());
    }

    #[test]
    fn units_belong_to_the_machine_before_them() {
        let given = arguments(&[
            "--seed",
            "3",
            "--local-worker",
            "--worker-units",
            "video",
            "--worker",
            "spark.local",
            "--worker",
            "mac.local:7834",
            "--worker-units",
            "steps,prompt",
        ]);
        let borrowed = borrowed(&given).unwrap();
        let read: Vec<_> = borrowed
            .iter()
            .map(|machine| (machine.address, machine.units))
            .collect();
        assert_eq!(
            read,
            vec![
                (None, Some("video")),
                (Some("spark.local"), None),
                (Some("mac.local:7834"), Some("steps,prompt")),
            ]
        );
    }

    #[test]
    fn units_without_a_machine_or_twice_over_are_refused() {
        for values in [
            &["--worker-units", "steps"][..],
            &["--seed", "3", "--worker-units", "steps"][..],
            &[
                "--worker",
                "a",
                "--worker-units",
                "steps",
                "--worker-units",
                "video",
            ][..],
            &["--worker"][..],
            &["--worker", "a", "--worker-units"][..],
        ] {
            assert!(borrowed(&arguments(values)).is_err(), "{values:?}");
        }
    }

    #[test]
    fn an_option_that_stands_alone_takes_nothing_after_it() {
        let given = arguments(&["--local-worker", "--seed", "3"]);
        let options = parse_options(&given, &["seed"], &["local-worker"], USAGE).unwrap();
        assert_eq!(options.get("local-worker"), Some(&""));
        assert_eq!(options["seed"], "3");

        // Last of the arguments, with nothing at all after it.
        let given = arguments(&["--seed", "3", "--local-worker"]);
        let options = parse_options(&given, &["seed"], &["local-worker"], USAGE).unwrap();
        assert_eq!(options.get("local-worker"), Some(&""));

        // Not given, which is the whole of what a flag can say otherwise.
        let given = arguments(&["--seed", "3"]);
        let options = parse_options(&given, &["seed"], &["local-worker"], USAGE).unwrap();
        assert_eq!(options.get("local-worker"), None);
    }

    #[test]
    fn rejects_unknown_options_and_positional_arguments() {
        for values in [&["--unknown", "4"][..], &["steps", "4"][..]] {
            assert!(parse_options(&arguments(values), &["steps"], &[], USAGE).is_err());
        }
    }

    #[test]
    fn reports_missing_values_before_the_next_option() {
        for values in [&["--steps"][..], &["--steps", "--seed", "3"][..]] {
            let error =
                parse_options(&arguments(values), &["steps", "seed"], &[], USAGE).unwrap_err();
            assert_eq!(error.to_string(), "--steps needs a value");
        }
    }

    #[test]
    fn rejects_invalid_integers() {
        for value in ["abc", "-1", "1.5", "18446744073709551616"] {
            let options = HashMap::from([("steps", value)]);
            assert!(option_number(&options, "steps", 20).is_err(), "{value}");
        }
    }

    #[test]
    fn rejects_non_finite_floats() {
        for value in ["NaN", "inf", "-inf", "1e100"] {
            let options = HashMap::from([("shift-video", value)]);
            let error = option_float(&options, "shift-video", 12.0).unwrap_err();
            assert_eq!(
                error.to_string(),
                "--shift-video must be a finite number",
                "{value}"
            );
        }
        assert!(
            option_float(
                &HashMap::from([("shift-video", "abc")]),
                "shift-video",
                12.0
            )
            .is_err()
        );
    }
}
