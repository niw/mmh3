//! The final latents of a generation, written without decoding them.

use crate::USAGE;
use mmh3::cli::parse_options;
use mmh3::generation::{OPTIONS, Settings, sample};
use mmh3_core::safetensors::write_f32;
use std::error::Error;
use std::path::Path;

pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let options = parse_options(arguments, &[OPTIONS, &["out"]].concat(), &[], USAGE)?;
    let path = Path::new(options.get("out").ok_or(USAGE)?);
    let settings = Settings::parse(&options, arguments)?;
    let (video, audio) = sample(&options, &settings)?;
    let shape = &settings.shape;
    let values = [
        shape.width.to_string(),
        shape.height.to_string(),
        shape.frames.to_string(),
        settings.steps.to_string(),
        settings.seed.to_string(),
    ];
    let mut metadata: Vec<(&str, &str)> = ["width", "height", "frames", "steps", "seed"]
        .into_iter()
        .zip(values.iter().map(String::as_str))
        .collect();
    if let Some(prompt) = options.get("prompt") {
        metadata.push(("prompt", prompt));
    }
    write_f32(
        path,
        &[
            ("video", &video.shape, &video.data),
            ("audio", &audio.shape, &audio.data),
        ],
        &metadata,
    )?;
    println!(
        "wrote the video latent {:?} to {}",
        video.shape,
        path.display()
    );
    Ok(())
}
