use crate::{DecodedMedia, MediaSpec, OutputBackend, Result, output_parent, validate_destination};
use mmh3_core::media::{Yuv420, write_wav, write_y4m};
use mmh3_core::tensor::Tensor;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Runs ffmpeg directly, without a shell. Arguments either describe the output,
/// or provide a full invocation using whole-argument {video}, {audio}, {out} placeholders.
pub struct FfmpegOutput {
    pub executable: PathBuf,
    pub arguments: Vec<OsString>,
}

impl FfmpegOutput {
    pub fn new(arguments: Vec<OsString>) -> Self {
        Self {
            executable: "ffmpeg".into(),
            arguments,
        }
    }

    fn template(&self) -> bool {
        self.arguments.iter().any(|arg| {
            ["{video}", "{audio}", "{out}"]
                .iter()
                .any(|placeholder| arg == placeholder)
        })
    }

    fn command_arguments(&self, video: &Path, audio: &Path, out: &Path) -> Result<Vec<OsString>> {
        let mut args: Vec<OsString> = ["-nostdin", "-y"].map(OsString::from).into();
        if self.template() {
            if !self.arguments.iter().any(|arg| arg == "{out}") {
                return Err("ffmpeg templates must include {out}".into());
            }
            for arg in &self.arguments {
                args.push(match arg.to_str() {
                    Some("{video}") => video.as_os_str().to_owned(),
                    Some("{audio}") => audio.as_os_str().to_owned(),
                    Some("{out}") => out.as_os_str().to_owned(),
                    _ => arg.clone(),
                });
            }
        } else {
            args.extend([
                OsString::from("-i"),
                video.into(),
                OsString::from("-i"),
                audio.into(),
            ]);
            if self.arguments.is_empty() {
                args.extend(
                    [
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-colorspace",
                        "bt709",
                        "-color_primaries",
                        "bt709",
                        "-color_trc",
                        "bt709",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        // Audio latents round to 25 ms, so their decoded duration can be a few
                        // milliseconds shorter than the video. Keep every video frame.
                        "-af",
                        "apad",
                        "-shortest",
                    ]
                    .map(OsString::from),
                );
            } else {
                args.extend(self.arguments.iter().cloned());
            }
            args.push(out.into());
        }
        Ok(args)
    }

    /// Writes the media as temporary Y4M and WAV inputs, then runs ffmpeg with `output` as the
    /// output path.
    fn encode(
        &self,
        media: &DecodedMedia<'_>,
        output: &Path,
        global_options: &[&str],
    ) -> Result<()> {
        let inputs = tempfile::tempdir()?;
        let video = inputs.path().join("video.y4m");
        let audio = inputs.path().join("audio.wav");
        let mut writer = BufWriter::new(File::create(&video)?);
        write_y4m(&mut writer, media.video, media.spec.fps)?;
        writer.flush()?;
        drop(writer);
        let mut writer = BufWriter::new(File::create(&audio)?);
        write_wav(&mut writer, media.audio, media.spec.sample_rate)?;
        writer.flush()?;
        drop(writer);
        let status = Command::new(&self.executable)
            .args(global_options)
            .args(self.command_arguments(&video, &audio, output)?)
            .stdin(Stdio::null())
            .status()
            .map_err(|error| format!("cannot run {}: {error}", self.executable.display()))?;
        if !status.success() {
            return Err(format!("ffmpeg failed: {status}").into());
        }
        Ok(())
    }
}

impl OutputBackend for FfmpegOutput {
    /// Encodes one black, silent frame with the same arguments and output extension into a
    /// temporary directory. This catches a missing executable, unknown options or codecs, and
    /// codecs the container does not accept before generation.
    fn validate(&self, spec: &MediaSpec, destination: &Path) -> Result<()> {
        spec.validate()?;
        validate_destination(destination)?;
        self.command_arguments(Path::new("video.y4m"), Path::new("audio.wav"), destination)?;
        let trial_spec = MediaSpec { frames: 1, ..*spec };
        let luma = spec.width * spec.height;
        let mut video_data = vec![16; luma];
        video_data.resize(Yuv420::frame_bytes(spec.height, spec.width), 128);
        let video = Yuv420 {
            frames: 1,
            height: spec.height,
            width: spec.width,
            data: video_data,
        };
        let samples = spec.sample_rate.div_ceil(spec.fps);
        let audio = Tensor::new(
            vec![spec.channels, samples],
            vec![0.0; spec.channels * samples],
        );
        let trial = tempfile::tempdir()?;
        let mut name = OsString::from("trial");
        if let Some(extension) = destination.extension() {
            name.push(".");
            name.push(extension);
        }
        let media = DecodedMedia {
            spec: trial_spec,
            video: &video,
            audio: &audio,
        };
        self.encode(
            &media,
            &trial.path().join(name),
            &["-hide_banner", "-loglevel", "error"],
        )
        .map_err(|error| {
            format!(
                "ffmpeg cannot write {} with these arguments: {error}",
                destination.display()
            )
            .into()
        })
    }

    fn write(&self, media: &DecodedMedia<'_>, destination: &Path) -> Result<()> {
        media.validate()?;
        // Keep the extension for ffmpeg's format inference. Publish only after successful encoding.
        let suffix = destination.extension().unwrap_or(OsStr::new(""));
        let mut suffix_with_dot = OsString::from(".");
        suffix_with_dot.push(suffix);
        let output = tempfile::Builder::new()
            .prefix(".mmh3-")
            .suffix(&suffix_with_dot)
            .tempfile_in(output_parent(destination))?;
        self.encode(media, &output.path().canonicalize()?, &[])?;
        if output.as_file().metadata()?.len() == 0 {
            return Err("ffmpeg did not write the requested output".into());
        }
        output.persist(destination)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_preserve_boundaries_and_override_defaults() {
        let backend = FfmpegOutput::new(vec!["-vf".into(), "drawtext=text='a b;$HOME'".into()]);
        let args = backend
            .command_arguments(
                Path::new("a b.y4m"),
                Path::new("a b.wav"),
                Path::new("out.mp4"),
            )
            .unwrap();
        assert_eq!(
            args,
            [
                "-nostdin",
                "-y",
                "-i",
                "a b.y4m",
                "-i",
                "a b.wav",
                "-vf",
                "drawtext=text='a b;$HOME'",
                "out.mp4"
            ]
            .map(OsString::from)
        );
    }

    #[test]
    fn templates_support_input_options_and_omitting_audio() {
        let backend = FfmpegOutput::new(
            ["-ss", "1", "-i", "{video}", "-an", "{out}"]
                .map(OsString::from)
                .into(),
        );
        let args = backend
            .command_arguments(
                Path::new("a b.y4m"),
                Path::new("audio.wav"),
                Path::new("x.webm"),
            )
            .unwrap();
        assert_eq!(
            args,
            [
                "-nostdin", "-y", "-ss", "1", "-i", "a b.y4m", "-an", "x.webm"
            ]
            .map(OsString::from)
        );
        assert!(
            FfmpegOutput::new(vec!["{video}".into()])
                .command_arguments(Path::new("v"), Path::new("a"), Path::new("o"))
                .is_err()
        );
    }
}
