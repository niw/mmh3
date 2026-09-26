//! Text-to-video generation with a synchronized soundtrack, written to a file.
//!
//! The command and the server both arrive here with the same arguments, since a form field and a
//! command-line option stand for the same thing and turning one into the other is all the server
//! does with them. What they generate is therefore the same run.

use crate::cli::{parse_options, split_ffmpeg_arguments};
use crate::generation::{FLAGS, OPTIONS, Settings, sample};
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::models::{AUDIO_VAE_FILE, option_path, video_vae_path};
#[cfg(any(feature = "cuda", feature = "metal"))]
use mmh3_core::safetensors::SafeTensors;
#[cfg(any(feature = "cuda", not(feature = "metal")))]
use mmh3_core::worker::CAPABILITY_DECODE_AUDIO;
use mmh3_core::worker::CAPABILITY_DECODE_VIDEO;
use std::error::Error;
use std::path::Path;

/// Generates video and audio, then passes the decoded media to the selected output backend.
/// `usage` is what an argument this does not understand is answered with.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn run(arguments: &[String], usage: &'static str) -> Result<(), Box<dyn Error>> {
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
        FLAGS,
        usage,
    )?;
    #[cfg(feature = "cuda")]
    crate::models::load_algorithm_cache();
    let video_path = Path::new(options.get("out").ok_or(usage)?).to_path_buf();
    let settings = Settings::parse(&options, arguments)?;
    let spec = MediaSpec {
        width: settings.shape.width,
        height: settings.shape.height,
        frames: settings.shape.frames,
        fps: FPS,
        sample_rate: SAMPLE_RATE,
        channels: 2,
    };
    let mut output = crate::output::prepare(ffmpeg_arguments, spec, &video_path)?;
    let (video, audio) = sample(&options, &settings)?;

    // The audio goes to a worker on a connection of its own, since a worker finishes its share of
    // the video before this machine finishes blending its own and is idle until then.
    #[cfg(feature = "cuda")]
    let remote_audio = crate::worker::RemoteAudio::start(
        settings.workers_owned(CAPABILITY_DECODE_AUDIO),
        settings.token.clone(),
        crate::worker::AUDIO_VAE_ROLE,
        &audio,
    );

    let started = Instant::now();
    let path = video_vae_path(&options)?;
    let file = SafeTensors::open(Path::new(&path))?;
    // A run with workers hands every chunk out and only puts the canvases together, so it reads
    // the weights that made them only if one never arrives.
    let chunks_out = settings.workers_for(CAPABILITY_DECODE_VIDEO);
    #[allow(unused_mut)]
    let mut decoder = match chunks_out.is_empty() {
        true => VideoDecoder::load(&file, "", DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP_MIN)?,
        false => VideoDecoder::to_assemble(&file, "", DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP_MIN)?,
    };
    #[cfg(feature = "metal")]
    decoder.set_precision(
        crate::metal::default_linear_precision(),
        crate::metal::attention_precision(&options)?,
    )?;
    // Chunks of the decode go to whichever machines answer, and this one walks the rest while
    // they work. A chunk that does not come back is decoded here.
    let mut remote = crate::worker::RemoteCanvases::start(
        crate::worker::connect_all(&chunks_out, &settings.token),
        crate::worker::VIDEO_VAE_ROLE,
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
        // The card this machine lends a worker of its own holds its models, and the decode makes
        // room beside them. It tries again only while it has taken no canvas, since one taken is
        // one the workers do not send twice.
        let decoded = crate::resident::with_room(|| {
            let mut taken = false;
            match decoder.decode_device_with(&video, &mut |chunk| {
                let canvas = remote.take(chunk)?;
                taken |= canvas.is_some();
                Ok(canvas)
            }) {
                Ok(decoded) => Ok(decoded),
                Err(error) if taken => Err(format!("decoding the video: {error}").into()),
                Err(error) => Err(error.into()),
            }
        })?;
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
pub fn run(arguments: &[String], usage: &'static str) -> Result<(), Box<dyn Error>> {
    use crate::worker::{AUDIO_VAE_ROLE, RemoteAudio, VIDEO_VAE_ROLE, decode_whole_on_worker};
    use mmh3_core::audio::SAMPLE_RATE;
    use mmh3_core::generation::FPS;
    use mmh3_output::MediaSpec;
    use std::time::Instant;

    let (arguments, ffmpeg_arguments) = split_ffmpeg_arguments(arguments);
    let options = parse_options(
        arguments,
        &[OPTIONS, &["out", "audio-vae"]].concat(),
        FLAGS,
        usage,
    )?;
    let video_path = Path::new(options.get("out").ok_or(usage)?).to_path_buf();
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
    let mut output = crate::output::prepare(ffmpeg_arguments, spec, &video_path)?;
    let (video, audio) = sample(&options, &settings)?;

    // The audio goes on a connection of its own while the video decodes, as it does anywhere else.
    let remote_audio = RemoteAudio::start(
        settings.workers_owned(CAPABILITY_DECODE_AUDIO),
        settings.token.clone(),
        AUDIO_VAE_ROLE,
        &audio,
    );

    let started = Instant::now();
    let frames = decode_whole_on_worker(
        &settings.workers_owned(CAPABILITY_DECODE_VIDEO),
        &settings.token,
        VIDEO_VAE_ROLE,
        &video,
    )
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
