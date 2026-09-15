//! Text-to-video generation with a synchronized soundtrack.

use crate::USAGE;
use mmh3::cli::{parse_options, split_ffmpeg_arguments};
use mmh3::generation::{OPTIONS, Settings, sample};
use mmh3::models::{AUDIO_VAE_FILE, option_path, video_vae_path};
use mmh3_core::safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

/// Generates video and audio, then passes the decoded media to the selected output backend.
pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::generation::FPS;
    use mmh3_cuda::audio_vae::{CudaAudioDecoder, SAMPLE_RATE};
    use mmh3_cuda::vae::{CudaVideoDecoder, DEFAULT_TILE_OVERLAP_MIN, DEFAULT_TILE_SIZE};
    use mmh3_output::MediaSpec;
    use std::time::Instant;

    let (arguments, ffmpeg_arguments) = split_ffmpeg_arguments(arguments);
    let options = parse_options(arguments, &[OPTIONS, &["out", "audio-vae"]].concat(), USAGE)?;
    let video_path = Path::new(options.get("out").ok_or(USAGE)?).to_path_buf();
    let settings = Settings::parse(&options, arguments)?;
    let spec = MediaSpec {
        width: settings.shape.width,
        height: settings.shape.height,
        frames: settings.shape.frames,
        fps: FPS,
        sample_rate: SAMPLE_RATE,
        channels: 2,
    };
    let mut output = mmh3::output::prepare(ffmpeg_arguments, spec, &video_path)?;
    let (video, audio) = sample(&options, &settings)?;

    let started = Instant::now();
    let path = video_vae_path(&options)?;
    let decoded = CudaVideoDecoder::load(
        &SafeTensors::open(Path::new(&path))?,
        "",
        DEFAULT_TILE_SIZE,
        DEFAULT_TILE_OVERLAP_MIN,
    )?
    .decode_device(&video)?;
    output.write_cuda_video(&decoded)?;
    drop(decoded);
    println!(
        "decoded and submitted the video with {path} in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    let path = option_path(&options, "audio-vae", AUDIO_VAE_FILE)?;
    let waveform =
        CudaAudioDecoder::load(&SafeTensors::open(Path::new(&path))?, "")?.decode(&audio)?;
    println!(
        "decoded the audio in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    output.write_audio(waveform)?;
    output.finish()?;
    println!(
        "wrote {} in {:.1} s",
        video_path.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
