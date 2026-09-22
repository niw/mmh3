//! Text-to-video generation with a synchronized soundtrack.

use crate::USAGE;
use mmh3::cli::{parse_options, split_ffmpeg_arguments};
use mmh3::generation::{OPTIONS, Settings, sample};
#[cfg(any(feature = "cuda", feature = "metal"))]
use mmh3::models::{AUDIO_VAE_FILE, option_path, video_vae_path};
#[cfg(any(feature = "cuda", feature = "metal"))]
use mmh3_core::safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

/// Generates video and audio, then passes the decoded media to the selected output backend.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::generation::FPS;
    #[cfg(feature = "cuda")]
    use mmh3_cuda::audio_vae::{CudaAudioDecoder as AudioDecoder, SAMPLE_RATE};
    #[cfg(feature = "cuda")]
    use mmh3_cuda::vae::{
        CudaVideoDecoder as VideoDecoder, DEFAULT_TILE_OVERLAP_MIN, DEFAULT_TILE_SIZE,
    };
    #[cfg(feature = "metal")]
    use mmh3_metal::audio_vae::{MetalAudioDecoder as AudioDecoder, SAMPLE_RATE};
    #[cfg(feature = "metal")]
    use mmh3_metal::vae::{
        DEFAULT_TILE_OVERLAP_MIN, DEFAULT_TILE_SIZE, MetalVideoDecoder as VideoDecoder,
    };
    use mmh3_output::MediaSpec;
    use std::time::Instant;

    let (arguments, ffmpeg_arguments) = split_ffmpeg_arguments(arguments);
    let options = parse_options(
        arguments,
        &[OPTIONS, &["out", "audio-vae"]].concat(),
        &[],
        USAGE,
    )?;
    #[cfg(feature = "cuda")]
    mmh3::models::load_algorithm_cache();
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

    // The audio goes to a worker on a connection of its own, since a worker finishes its share of
    // the video before this machine finishes blending its own and is idle until then.
    #[cfg(feature = "cuda")]
    let remote_audio = mmh3::worker::RemoteAudio::start(
        settings.workers.clone(),
        settings.token.clone(),
        mmh3::worker::AUDIO_VAE_ROLE,
        &audio,
    );

    let started = Instant::now();
    let path = video_vae_path(&options)?;
    let file = SafeTensors::open(Path::new(&path))?;
    // A run with workers hands every chunk out and only puts the canvases together, so it reads
    // the weights that made them only if one never arrives.
    let decoder = match settings.workers.is_empty() {
        true => VideoDecoder::load(&file, "", DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP_MIN)?,
        false => VideoDecoder::to_assemble(&file, "", DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP_MIN)?,
    };
    // Chunks of the decode go to whichever machines answer, and this one walks the rest while
    // they work. A chunk that does not come back is decoded here.
    let mut remote = mmh3::worker::RemoteCanvases::start(
        mmh3::worker::connect_all(&settings.workers_borrowed(), &settings.token),
        mmh3::worker::VIDEO_VAE_ROLE,
        &video,
        decoder.plan(&video)?.chunks,
        DEFAULT_TILE_SIZE,
        DEFAULT_TILE_OVERLAP_MIN,
    );
    if !remote.chunks().is_empty() {
        println!("chunks {:?} decode elsewhere", remote.chunks());
    }
    #[cfg(feature = "cuda")]
    {
        let decoded = decoder.decode_device_with(&video, &mut |chunk| remote.take(chunk))?;
        drop(decoder);
        output.write_cuda_video(&decoded)?;
    }
    #[cfg(feature = "metal")]
    if output.streams_metal_video() {
        decoder.decode_stream_with(&video, &mut |chunk| remote.take(chunk), |frames| {
            output
                .write_metal_video_chunk(frames)
                .map_err(|e| mmh3_metal::Error::new(e.to_string()))
        })?;
        drop(decoder);
    } else {
        let decoded = decoder.decode_device_with(&video, &mut |chunk| remote.take(chunk))?;
        drop(decoder);
        output.write_metal_video(&decoded)?;
    }
    println!(
        "decoded and submitted the video with {path} in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    #[cfg(feature = "cuda")]
    let remote_waveform = remote_audio.take()?;
    #[cfg(not(feature = "cuda"))]
    let remote_waveform: Option<mmh3_core::tensor::Tensor> = None;
    let waveform = match remote_waveform {
        Some(waveform) => {
            println!(
                "took the audio from a worker in {:.1} s",
                started.elapsed().as_secs_f64()
            );
            waveform
        }
        None => {
            let path = option_path(&options, "audio-vae", AUDIO_VAE_FILE)?;
            let waveform =
                AudioDecoder::load(&SafeTensors::open(Path::new(&path))?, "")?.decode(&audio)?;
            println!(
                "decoded the audio in {:.1} s",
                started.elapsed().as_secs_f64()
            );
            waveform
        }
    };
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

/// The same generation on a machine with no device of its own: the prompt, the steps, the decode
/// and the soundtrack all happen on the workers, and this process holds the plan and the file.
#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3::worker::{AUDIO_VAE_ROLE, RemoteAudio, VIDEO_VAE_ROLE, decode_whole_on_worker};
    use mmh3_core::audio::SAMPLE_RATE;
    use mmh3_core::generation::FPS;
    use mmh3_output::MediaSpec;
    use std::time::Instant;

    let (arguments, ffmpeg_arguments) = split_ffmpeg_arguments(arguments);
    let options = parse_options(
        arguments,
        &[OPTIONS, &["out", "audio-vae"]].concat(),
        &[],
        USAGE,
    )?;
    let video_path = Path::new(options.get("out").ok_or(USAGE)?).to_path_buf();
    let settings = Settings::parse(&options, arguments)?;
    if settings.workers.is_empty() {
        return Err("this build runs nothing of its own: name a worker with --worker".into());
    }
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

    // The audio goes on a connection of its own while the video decodes, as it does anywhere else.
    let remote_audio = RemoteAudio::start(
        settings.workers.clone(),
        settings.token.clone(),
        AUDIO_VAE_ROLE,
        &audio,
    );

    let started = Instant::now();
    let frames = decode_whole_on_worker(&settings.workers, &settings.token, VIDEO_VAE_ROLE, &video)
        .ok_or("no worker decoded the video: name one that holds the video VAE with --worker")?;
    println!(
        "took {} frames from a worker in {:.1} s",
        frames.frames,
        started.elapsed().as_secs_f64()
    );
    output.write_host_video(frames)?;

    let started = Instant::now();
    let waveform = remote_audio
        .take()?
        .ok_or("no worker decoded the audio: name one that holds the audio VAE with --worker")?;
    println!(
        "took the audio from a worker in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    output.write_audio(waveform)?;
    output.finish()?;
    println!(
        "wrote {} in {:.1} s",
        video_path.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
