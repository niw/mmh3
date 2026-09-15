//! Model checks against golden data captured from the reference implementation.

use crate::USAGE;
use mmh3::cli::parse_options;
use mmh3::models::{
    AUDIO_VAE_FILE, DIT_FILE, REFERENCE_DIT_FILE, TEXT_ENCODER_FILE, VIDEO_VAE_FILE, load_dit,
    option_path, sparse_attention,
};
use mmh3_core::json;
use mmh3_core::safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    match arguments.first().map(String::as_str) {
        Some("dit") => check_dit(&arguments[1..]),
        Some("sample") => check_sample(&arguments[1..]),
        Some("video-vae") => check_video_vae(&arguments[1..]),
        Some("keyframes") => check_keyframes(&arguments[1..]),
        Some("audio-vae") => check_audio_vae(&arguments[1..]),
        Some("text-encoder") => check_text_encoder(&arguments[1..]),
        _ => Err(USAGE.into()),
    }
}

/// One file of a golden directory written by tools/golden/h3_reference.py.
struct GoldenFile(SafeTensors);

impl GoldenFile {
    fn open(directory: &Path, name: &str) -> Result<Self, Box<dyn Error>> {
        Ok(GoldenFile(SafeTensors::open(&directory.join(name))?))
    }

    fn metadata(&self, key: &str) -> Result<String, Box<dyn Error>> {
        Ok(self
            .0
            .metadata()
            .iter()
            .find(|(name, _)| name == key)
            .ok_or(format!("golden metadata lacks {key}"))?
            .1
            .clone())
    }

    fn tensor(&self, name: &str) -> Result<mmh3_core::tensor::Tensor, Box<dyn Error>> {
        Ok(mmh3_core::tensor::Tensor::load(
            &self.0,
            self.0
                .get(name)
                .ok_or(format!("golden data lacks {name}"))?,
        )?)
    }

    /// The tensor without its leading batch dimension.
    fn unbatched(&self, name: &str) -> Result<mmh3_core::tensor::Tensor, Box<dyn Error>> {
        let mut tensor = self.tensor(name)?;
        tensor.shape.remove(0);
        Ok(tensor)
    }

    fn has_metadata(&self, key: &str) -> bool {
        self.0.metadata().iter().any(|(name, _)| name == key)
    }

    /// The modality of each text token from `token_tags`, empty when every token is text.
    fn context_modalities(
        &self,
    ) -> Result<Vec<mmh3_core::dit::timestep::Modality>, Box<dyn Error>> {
        use mmh3_core::dit::timestep::Modality;

        let info = self
            .0
            .get("token_tags")
            .ok_or("golden data lacks token_tags")?;
        let tags: Vec<i64> = self
            .0
            .data(info)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&bytes| i64::from_le_bytes(bytes))
            .collect();
        if tags.iter().all(|&tag| tag == Modality::Text as i64) {
            return Ok(Vec::new());
        }
        Ok(tags
            .into_iter()
            .map(|tag| match tag {
                0 => Modality::Video,
                2 => Modality::Audio,
                _ => Modality::Text,
            })
            .collect())
    }

    /// The keyframes as the DiT sees them, with ComfyUI's condition noise.
    fn keyframes(&self) -> Result<Vec<mmh3_core::dit::inputs::Keyframe>, Box<dyn Error>> {
        if !self.has_metadata("frame_indices") {
            return Ok(Vec::new());
        }
        json::parse(&self.metadata("frame_indices")?)?
            .as_array()
            .ok_or("bad frame_indices")?
            .iter()
            .enumerate()
            .map(|(index, frame)| {
                Ok(mmh3_core::dit::inputs::Keyframe {
                    frame_index: frame.as_u64().ok_or("bad frame_indices")? as usize,
                    video: Some(self.tensor(&format!("keyframe.{index}.augmented"))?),
                    audio: None,
                })
            })
            .collect()
    }

    /// The reference pictures as the DiT sees them, with ComfyUI's condition noise.
    fn references(&self) -> Result<Vec<mmh3_core::dit::inputs::Reference>, Box<dyn Error>> {
        if !self.has_metadata("reference_pictures") {
            return Ok(Vec::new());
        }
        let count: usize = self
            .metadata("reference_pictures")?
            .parse()
            .map_err(|_| "bad reference_pictures")?;
        (0..count)
            .map(|index| {
                Ok(mmh3_core::dit::inputs::Reference::Picture(
                    self.tensor(&format!("reference.{index}.augmented"))?,
                ))
            })
            .collect()
    }
}

/// The DiT file in the models directory for golden data with or without references.
fn dit_file_for(references: &[mmh3_core::dit::inputs::Reference]) -> &'static str {
    if references.is_empty() {
        DIT_FILE
    } else {
        REFERENCE_DIT_FILE
    }
}

/// Similarity of `actual` to `expected`: cosine, relative L2 error and the largest absolute error
/// against the largest expected magnitude.
fn compare(actual: &[f32], expected: &[f32]) -> (f64, f64, f32, f32) {
    let dot: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&left, &right)| left as f64 * right as f64)
        .sum();
    let norm = |values: &[f32]| {
        values
            .iter()
            .map(|&value| value as f64 * value as f64)
            .sum::<f64>()
            .sqrt()
    };
    let difference: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&left, &right)| (left as f64 - right as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = expected
        .iter()
        .fold(0.0f32, |maximum, &value| maximum.max(value.abs()));
    let worst = actual
        .iter()
        .zip(expected)
        .fold(0.0f32, |maximum, (&left, &right)| {
            maximum.max((left - right).abs())
        });
    (
        dot / (norm(actual) * norm(expected)),
        difference / norm(expected),
        worst,
        scale,
    )
}

/// Runs the CUDA DiT on the captured step of a golden directory written by
/// tools/golden/h3_reference.py.
fn check_dit(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::layout::{PackedLayout, SegmentKind};
    use std::time::Instant;

    let options = parse_options(
        arguments,
        &[
            "golden",
            "models",
            "weights",
            "patch",
            "lora",
            "lora-strength",
            "lora-mode",
            "attention",
            "attention-precision",
            "linear-precision",
            "sparse-tau",
            "vsa-sparsity",
        ],
        USAGE,
    )?;
    let golden = Path::new(options.get("golden").ok_or(USAGE)?);
    let dit_file = GoldenFile::open(golden, "dit.safetensors")?;
    let text_file = GoldenFile::open(golden, "text.safetensors")?;
    let step: usize = dit_file.metadata("capture_step")?.parse()?;
    let stride: usize = dit_file.metadata("block_row_stride")?.parse()?;
    let sigmas = json::parse(&dit_file.metadata("sigmas_video")?)?;
    let sigma = sigmas
        .as_array()
        .and_then(|values| values.get(step))
        .and_then(json::Value::as_f64)
        .ok_or("bad sigmas_video")?;
    let watched: Vec<usize> = json::parse(&dit_file.metadata("watched_blocks")?)?
        .as_array()
        .ok_or("bad watched_blocks")?
        .iter()
        .filter_map(|value| value.as_u64().map(|index| index as usize))
        .collect();
    let inputs = DitInputs {
        video: dit_file.unbatched(&format!("step{step}.input.video"))?,
        audio: dit_file.unbatched(&format!("step{step}.input.audio"))?,
        context: text_file.tensor("context")?,
        context_modalities: text_file.context_modalities()?,
        keyframes: dit_file.keyframes()?,
        references: dit_file.references()?,
        sigma: sigma as f32,
        shift_video: dit_file.metadata("shift_video")?.parse()?,
        shift_audio: dit_file.metadata("shift_audio")?.parse()?,
    };

    let dit = load_dit(&options, "weights", dit_file_for(&inputs.references))?;
    let started = Instant::now();
    let sparse = sparse_attention(&options, dit.has_vsa_gates())?;
    let outputs = dit.forward(&inputs, &watched, sparse.as_ref())?;
    let routing = outputs.routed_fraction.map_or(String::new(), |fraction| {
        format!(", Sol-Attn routed {:.1}%", 100.0 * fraction)
    });
    println!(
        "forward at sigma {sigma:.4} in {:.1} s{routing}",
        started.elapsed().as_secs_f64()
    );

    println!(
        "{:<12} {:>11} {:>11} {:>11} {:>9}",
        "tensor", "cosine", "rel L2", "max error", "scale"
    );
    let report = |name: &str, actual: &[f32], expected: &[f32]| {
        let (cosine, relative, worst, scale) = compare(actual, expected);
        println!("{name:<12} {cosine:>11.7} {relative:>11.3e} {worst:>11.3e} {scale:>9.3}");
    };
    let hidden = dit.config().hidden;
    let layout = PackedLayout::for_inputs(&inputs);
    let segment_of = |token: usize| match layout
        .segments
        .iter()
        .find(|segment| (segment.start..segment.end).contains(&token))
        .map(|segment| segment.kind)
    {
        Some(SegmentKind::Text) => "text",
        Some(SegmentKind::Audio) => "audio",
        Some(SegmentKind::Video) => "video",
        _ => "conditions",
    };
    for (index, residual) in &outputs.blocks {
        let expected = dit_file.tensor(&format!("step{step}.block.{index}"))?;
        let sampled: Vec<f32> = residual
            .chunks_exact(hidden)
            .step_by(stride)
            .flatten()
            .copied()
            .collect();
        report(&format!("block {index}"), &sampled, &expected.data);
        for segment in ["text", "conditions", "audio", "video"] {
            let (mut actual, mut reference) = (Vec::new(), Vec::new());
            for (sample, (row, expected_row)) in sampled
                .chunks_exact(hidden)
                .zip(expected.data.chunks_exact(hidden))
                .enumerate()
            {
                if segment_of(sample * stride) == segment {
                    actual.extend_from_slice(row);
                    reference.extend_from_slice(expected_row);
                }
            }
            if !actual.is_empty() {
                report(&format!("  {segment}"), &actual, &reference);
            }
        }
    }
    // The golden data stores the data-ward prediction u with x0 = x + sigma · u, the negated flow
    // velocity.
    let negated = |name: &str| -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(dit_file
            .tensor(name)?
            .data
            .into_iter()
            .map(|value| -value)
            .collect())
    };
    report(
        "video",
        &outputs.video,
        &negated(&format!("step{step}.velocity.video"))?,
    );
    report(
        "audio",
        &outputs.audio,
        &negated(&format!("step{step}.velocity.audio"))?,
    );
    Ok(())
}

/// Samples every step of a golden directory from its initial noise and compares the latents after
/// each step.
fn check_sample(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::sampler::{Schedule, euler_step};
    use std::time::Instant;

    let options = parse_options(
        arguments,
        &[
            "golden",
            "models",
            "weights",
            "patch",
            "lora",
            "lora-strength",
            "lora-mode",
            "attention",
            "attention-precision",
            "linear-precision",
            "sparse-tau",
            "sparse-start",
            "vsa-sparsity",
        ],
        USAGE,
    )?;
    let golden = Path::new(options.get("golden").ok_or(USAGE)?);
    let dit_file = GoldenFile::open(golden, "dit.safetensors")?;
    let text_file = GoldenFile::open(golden, "text.safetensors")?;
    let (shift_video, shift_audio): (f32, f32) = (
        dit_file.metadata("shift_video")?.parse()?,
        dit_file.metadata("shift_audio")?.parse()?,
    );
    let schedule = Schedule::uniform(
        dit_file.metadata("steps")?.parse()?,
        shift_video,
        shift_audio,
    );

    let mut video = dit_file.unbatched("noise.video")?;
    let mut audio = dit_file.unbatched("noise.audio")?;
    let context = text_file.tensor("context")?;
    let context_modalities = text_file.context_modalities()?;
    let keyframes = dit_file.keyframes()?;
    let references = dit_file.references()?;
    let dit = load_dit(&options, "weights", dit_file_for(&references))?;
    let sparse = sparse_attention(&options, dit.has_vsa_gates())?;
    println!(
        "{:<6} {:>8} {:>11} {:>11} {:>11} {:>11} {:>7}",
        "step", "sigma", "video cos", "video L2", "audio cos", "audio L2", "s"
    );
    for step in 0..schedule.steps() {
        let started = Instant::now();
        let inputs = DitInputs {
            video: video.clone(),
            audio: audio.clone(),
            context: context.clone(),
            context_modalities: context_modalities.clone(),
            keyframes: keyframes.clone(),
            references: references.clone(),
            sigma: schedule.video[step],
            shift_video,
            shift_audio,
        };
        let step_sparse =
            sparse.filter(|settings| settings.applies_to_step(step, schedule.steps()));
        let outputs = dit.forward(&inputs, &[], step_sparse.as_ref())?;
        euler_step(
            &mut video.data,
            &outputs.video,
            schedule.video[step],
            schedule.video[step + 1],
        );
        euler_step(
            &mut audio.data,
            &outputs.audio,
            schedule.audio[step],
            schedule.audio[step + 1],
        );
        let (video_cosine, video_relative, _, _) = compare(
            &video.data,
            &dit_file.tensor(&format!("step{step}.latent.video"))?.data,
        );
        let (audio_cosine, audio_relative, _, _) = compare(
            &audio.data,
            &dit_file.tensor(&format!("step{step}.latent.audio"))?.data,
        );
        println!(
            "{step:<6} {:>8.4} {video_cosine:>11.7} {video_relative:>11.3e} {audio_cosine:>11.7} {audio_relative:>11.3e} {:>7.1}",
            schedule.video[step],
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// Encodes the keyframe or reference pictures of a golden directory with the video VAE's encoder
/// and compares the latents with ComfyUI's, or with those of `--reference`, a file of the same
/// tensors.
fn check_keyframes(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_cuda::video_encoder::CudaVideoEncoder;
    use std::time::Instant;

    let options = parse_options(
        arguments,
        &["golden", "models", "weights", "reference"],
        USAGE,
    )?;
    let golden = Path::new(options.get("golden").ok_or(USAGE)?);
    let weights_path = option_path(&options, "weights", VIDEO_VAE_FILE)?;
    let (name, file) = if golden.join("keyframes.safetensors").exists() {
        ("keyframe", "keyframes.safetensors")
    } else {
        ("reference", "references.safetensors")
    };
    let pictures = GoldenFile::open(golden, file)?;
    let reference = match options.get("reference") {
        Some(path) => GoldenFile(SafeTensors::open(Path::new(path))?),
        None => GoldenFile::open(golden, file)?,
    };
    let count = if name == "keyframe" {
        json::parse(&pictures.metadata("frame_indices")?)?
            .as_array()
            .ok_or("bad frame_indices")?
            .len()
    } else {
        pictures
            .metadata("reference_pictures")?
            .parse()
            .map_err(|_| "bad reference_pictures")?
    };

    let started = Instant::now();
    let encoder = CudaVideoEncoder::load(&SafeTensors::open(Path::new(&weights_path))?)?;
    println!(
        "loaded the encoder of {weights_path} in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    println!(
        "{:<12} {:>11} {:>11} {:>11} {:>9}",
        "latent", "cosine", "rel L2", "max error", "scale"
    );
    for index in 0..count {
        let picture = pictures.tensor(&format!("{name}.{index}.pixels"))?;
        let expected = reference.tensor(&format!("{name}.{index}.latent"))?;
        let started = Instant::now();
        let latent = encoder.encode_picture(&picture)?.mean_latent();
        let elapsed = started.elapsed().as_secs_f64();
        if latent.shape != expected.shape {
            return Err(format!(
                "latent shape {:?} differs from the golden {:?}",
                latent.shape, expected.shape
            )
            .into());
        }
        let (cosine, relative, worst, scale) = compare(&latent.data, &expected.data);
        println!(
            "{:<12} {cosine:>11.7} {relative:>11.3e} {worst:>11.3e} {scale:>9.3} ({elapsed:.2} s)",
            format!("{name} {index}")
        );
    }
    Ok(())
}

/// Decodes the final video latent of a golden directory and compares the pixels with ComfyUI's,
/// or with the decode of the checkpoint given by `--reference`.
fn check_video_vae(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_cuda::vae::{CudaVideoDecoder, DEFAULT_TILE_OVERLAP_MIN, DEFAULT_TILE_SIZE};
    use std::time::Instant;

    let options = parse_options(
        arguments,
        &["golden", "models", "weights", "reference"],
        USAGE,
    )?;
    let golden = Path::new(options.get("golden").ok_or(USAGE)?);
    let weights_path = option_path(&options, "weights", VIDEO_VAE_FILE)?;
    let dit_file = GoldenFile::open(golden, "dit.safetensors")?;
    let steps: usize = dit_file.metadata("steps")?.parse()?;
    let latent = dit_file.unbatched(&format!("step{}.latent.video", steps - 1))?;

    let started = Instant::now();
    let weights = SafeTensors::open(Path::new(&weights_path))?;
    let decoder =
        CudaVideoDecoder::load(&weights, "", DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP_MIN)?;
    println!(
        "loaded {weights_path} in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    let pixels = decoder.decode(&latent, false)?.pixels;
    println!(
        "decoded latent {:?} to {:?} in {:.1} s",
        latent.shape,
        pixels.shape,
        started.elapsed().as_secs_f64()
    );
    drop(decoder);
    let expected = if let Some(reference) = options.get("reference") {
        let weights = SafeTensors::open(Path::new(reference))?;
        CudaVideoDecoder::load(&weights, "", DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP_MIN)?
            .decode(&latent, false)?
            .pixels
    } else if golden.join("decode.safetensors").exists() {
        GoldenFile::open(golden, "decode.safetensors")?.unbatched("video")?
    } else {
        println!("no decode.safetensors in the golden directory, nothing to compare");
        return Ok(());
    };
    if pixels.shape != expected.shape {
        return Err(format!(
            "decoded shape {:?} differs from the golden {:?}",
            pixels.shape, expected.shape
        )
        .into());
    }

    let psnr = |actual: &[f32], expected: &[f32]| {
        let squared: f64 = actual
            .iter()
            .zip(expected)
            .map(|(&left, &right)| (left as f64 - right as f64).powi(2))
            .sum();
        10.0 * (actual.len() as f64 / squared).log10()
    };
    println!(
        "{:<12} {:>11} {:>11} {:>11} {:>9}",
        "pixels", "cosine", "rel L2", "max error", "PSNR dB"
    );
    let report = |name: &str, actual: &[f32], expected: &[f32]| {
        let (cosine, relative, worst, _) = compare(actual, expected);
        println!(
            "{name:<12} {cosine:>11.7} {relative:>11.3e} {worst:>11.3e} {:>9.2}",
            psnr(actual, expected)
        );
    };
    report("all", &pixels.data, &expected.data);
    let (frames, plane) = (pixels.shape[1], pixels.shape[2] * pixels.shape[3]);
    let frame_values = |data: &[f32], frame: usize| -> Vec<f32> {
        (0..pixels.shape[0])
            .flat_map(|channel| data[(channel * frames + frame) * plane..][..plane].to_vec())
            .collect()
    };
    for frame in 0..frames {
        report(
            &format!("frame {frame}"),
            &frame_values(&pixels.data, frame),
            &frame_values(&expected.data, frame),
        );
    }
    Ok(())
}

/// Decodes the final audio latent of a golden directory and compares the waveform with ComfyUI's
/// decodes.
fn check_audio_vae(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_cuda::audio_vae::CudaAudioDecoder;
    use std::time::Instant;

    let options = parse_options(arguments, &["golden", "models", "weights"], USAGE)?;
    let golden = Path::new(options.get("golden").ok_or(USAGE)?);
    let weights_path = option_path(&options, "weights", AUDIO_VAE_FILE)?;
    let dit_file = GoldenFile::open(golden, "dit.safetensors")?;
    let steps: usize = dit_file.metadata("steps")?.parse()?;
    let latent = dit_file.unbatched(&format!("step{}.latent.audio", steps - 1))?;

    let started = Instant::now();
    let weights = SafeTensors::open(Path::new(&weights_path))?;
    let decoder = CudaAudioDecoder::load(&weights, "")?;
    println!(
        "loaded {weights_path} in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    let waveform = decoder.decode(&latent)?;
    println!(
        "decoded latent {:?} to {:?} in {:.2} s",
        latent.shape,
        waveform.shape,
        started.elapsed().as_secs_f64()
    );
    if !golden.join("decode.safetensors").exists() {
        println!("no decode.safetensors in the golden directory, nothing to compare");
        return Ok(());
    }

    let decode_file = GoldenFile::open(golden, "decode.safetensors")?;
    println!(
        "{:<16} {:>11} {:>11} {:>11} {:>8}",
        "against", "cosine", "rel L2", "max error", "SNR dB"
    );
    // NOTE: ComfyUI decodes audio in BF16 by default. The golden script also stores an FP32 decode.
    for (label, name) in [
        ("ComfyUI default", "audio"),
        ("ComfyUI FP32", "audio_float32"),
    ] {
        if decode_file.0.get(name).is_none() {
            continue;
        }
        let expected = decode_file.unbatched(name)?;
        if waveform.shape != expected.shape {
            return Err(format!(
                "decoded shape {:?} differs from the golden {:?}",
                waveform.shape, expected.shape
            )
            .into());
        }
        let (cosine, relative, worst, _) = compare(&waveform.data, &expected.data);
        println!(
            "{label:<16} {cosine:>11.7} {relative:>11.3e} {worst:>11.3e} {:>8.2}",
            -20.0 * relative.log10()
        );
    }
    Ok(())
}

/// Tokenizes and encodes the prompt of a golden file written by tools/golden/text_encoder.py and
/// compares the token ids, the hidden states it holds and the conditioning.
fn check_text_encoder(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::tokenizer::Tokenizer;
    use mmh3_cuda::text_encoder::CudaTextEncoder;
    use std::time::Instant;

    let options = parse_options(arguments, &["golden", "models", "weights"], USAGE)?;
    let golden = GoldenFile(SafeTensors::open(Path::new(
        options.get("golden").ok_or(USAGE)?,
    ))?);
    let prompt = golden.metadata("prompt")?;
    let id_info = golden
        .0
        .get("token_ids")
        .ok_or("golden data lacks token_ids")?;
    // Vision embeddings count as -1 in the golden ids.
    let expected_ids: Vec<u32> = golden
        .0
        .data(id_info)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&bytes| match i64::from_le_bytes(bytes) {
            -1 => mmh3_core::vision::IMAGE_PAD,
            id => id as u32,
        })
        .collect();
    let pictures: Vec<mmh3_core::tensor::Tensor> = (0..)
        .map_while(|index| golden.tensor(&format!("picture.{index}.pixels")).ok())
        .collect();
    let grids: Vec<mmh3_core::vision::VisionGrid> = pictures
        .iter()
        .map(|picture| {
            mmh3_core::vision::VisionGrid::for_picture(picture.shape[0], picture.shape[1])
        })
        .collect();
    let prompt = mmh3_core::vision::vision_prompt(&Tokenizer::h3(), &prompt, &grids);
    let ids = prompt.ids.clone();
    if ids != expected_ids {
        return Err(
            format!("token ids differ from the golden data:\n  expected {expected_ids:?}\n  actual   {ids:?}").into()
        );
    }
    println!("{} tokens match", ids.len());
    if !pictures.is_empty() {
        let tags = golden.context_modalities()?;
        if tags != prompt.modalities {
            return Err("token tags differ from the golden data".into());
        }
        println!("{} pictures, token tags match", pictures.len());
    }

    let started = Instant::now();
    let weights_path = option_path(&options, "weights", TEXT_ENCODER_FILE)?;
    let encoder = CudaTextEncoder::load(&SafeTensors::open(Path::new(&weights_path))?)?;
    println!(
        "loaded {weights_path} in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    let captured: Vec<usize> = golden
        .0
        .tensors()
        .iter()
        .filter_map(|info| info.name.strip_prefix("layer.")?.parse().ok())
        .collect();
    let report = |name: &str, actual: &[f32], expected: &[f32]| {
        let (cosine, relative, worst, scale) = compare(actual, expected);
        println!("{name:<12} {cosine:>11.7} {relative:>11.3e} {worst:>11.3e} {scale:>11.3}");
    };
    let mut embeddings = Vec::new();
    if !pictures.is_empty() {
        let started = Instant::now();
        let vision = mmh3_cuda::vision::CudaVisionEncoder::load(&SafeTensors::open(Path::new(
            &weights_path,
        ))?)?;
        println!(
            "loaded the vision tower in {:.1} s",
            started.elapsed().as_secs_f64()
        );
        println!(
            "{:<12} {:>11} {:>11} {:>11} {:>11}",
            "vision", "cosine", "rel L2", "max error", "scale"
        );
        for (index, picture) in pictures.iter().enumerate() {
            let started = Instant::now();
            let embedded = vision.encode(picture)?;
            mmh3_cuda::synchronize()?;
            println!(
                "picture {index} embedded in {:.3} s",
                started.elapsed().as_secs_f64()
            );
            report(
                &format!("  merged {index}"),
                &embedded.merged.to_f32()?,
                &golden.tensor(&format!("picture.{index}.merged"))?.data,
            );
            for (layer, features) in embedded.deepstack.iter().enumerate() {
                report(
                    &format!("  stack {index}.{layer}"),
                    &features.to_f32()?,
                    &golden
                        .tensor(&format!("picture.{index}.deepstack.{layer}"))?
                        .data,
                );
            }
            embeddings.push(embedded);
        }
    }
    let started = Instant::now();
    let encoding = encoder.encode_prompt(&prompt, &embeddings, &captured)?;
    println!("encoded in {:.3} s", started.elapsed().as_secs_f64());

    println!(
        "{:<12} {:>11} {:>11} {:>11} {:>11}",
        "tensor", "cosine", "rel L2", "max error", "scale"
    );
    for (layer, hidden) in &encoding.layers {
        report(
            &format!("layer {layer}"),
            &hidden.data,
            &golden.tensor(&format!("layer.{layer}"))?.data,
        );
    }
    let expected = golden.tensor("context")?;
    report("context", &encoding.context.data, &expected.data);
    let hidden = encoder.config().hidden;
    let mut boundaries = vec![0];
    for (&start, embedded) in prompt.picture_starts.iter().zip(&embeddings) {
        boundaries.extend([start, start + embedded.tokens]);
    }
    boundaries.push(ids.len());
    for (index, pair) in boundaries.windows(2).enumerate() {
        if pair[1] > pair[0] {
            let name = if index % 2 == 1 {
                format!("  picture {}", index / 2)
            } else {
                format!("  text {}", index / 2)
            };
            report(
                &name,
                &encoding.context.data[pair[0] * hidden..pair[1] * hidden],
                &expected.data[pair[0] * hidden..pair[1] * hidden],
            );
        }
    }
    // NOTE: the first token carries the model's massive activations, so it is reported apart from
    // the rest.
    report(
        "  token 0",
        &encoding.context.data[..hidden],
        &expected.data[..hidden],
    );
    if ids.len() > 1 {
        report(
            "  others",
            &encoding.context.data[hidden..],
            &expected.data[hidden..],
        );
    }
    Ok(())
}
