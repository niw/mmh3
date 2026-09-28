//! Requires VideoToolbox and ffmpeg, which makes the clips and decodes them for comparison.
#![cfg(target_os = "macos")]

use mmh3_input::mp4::Mp4File;
use mmh3_input_videotoolbox::{Decoder, decode_frames};
use std::path::Path;
use std::process::Command;

fn ffmpeg(arguments: &[&str]) -> Vec<u8> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error"])
        .args(arguments)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// A clip of ffmpeg's test pattern, which moves in every frame, so frames out of order differ.
fn clip(path: &Path, size: &str, options: &[&str]) {
    let source = format!("testsrc2=size={size}:rate=24");
    let mut arguments = vec!["-y", "-f", "lavfi", "-i", &source, "-frames:v", "30"];
    arguments.extend(["-c:v", "libx264"]);
    arguments.extend(options);
    arguments.push(path.to_str().unwrap());
    ffmpeg(&arguments);
}

/// ffmpeg's decode of a clip, as NV12 and as RGB in the given matrix and range. Chroma is taken
/// from the nearest sample, as the conversion to RGB takes it.
fn reference(path: &Path, matrix: &str, range: &str) -> (Vec<u8>, Vec<u8>) {
    let path = path.to_str().unwrap();
    let nv12 = ffmpeg(&["-i", path, "-pix_fmt", "nv12", "-f", "rawvideo", "-"]);
    let filter = format!("scale=in_color_matrix={matrix}:in_range={range}:out_range=pc");
    let rgb = ffmpeg(&[
        "-i",
        path,
        "-vf",
        &filter,
        "-sws_flags",
        "neighbor+accurate_rnd+full_chroma_int+bitexact",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]);
    (nv12, rgb)
}

/// Every frame the decoder hands out, as NV12, in its order.
fn nv12_frames(path: &Path) -> Vec<u8> {
    let mut file = Mp4File::open(path).unwrap();
    let mut decoder = Decoder::open(&file.track().parameter_sets.clone()).unwrap();
    let mut frames = Vec::new();
    for (index, position) in file.track().display_positions().into_iter().enumerate() {
        decoder
            .feed(&file.access_unit(index).unwrap(), position)
            .unwrap();
        while let Some(frame) = decoder.take().unwrap() {
            frames.extend(frame);
        }
    }
    decoder.flush().unwrap();
    while let Some(frame) = decoder.take().unwrap() {
        frames.extend(frame);
    }
    frames
}

#[test]
#[ignore = "requires VideoToolbox and ffmpeg"]
fn decodes_frames_in_display_order_in_the_declared_colour() {
    let directory = tempfile::tempdir().unwrap();
    let cases: [(&str, &str, &[&str], &str, &str); 7] = [
        (
            "bt709-limited-bframes",
            "300x200",
            &["-bf", "3", "-pix_fmt", "yuv420p", "-colorspace", "bt709"],
            "bt709",
            "tv",
        ),
        (
            "bt601-limited",
            "300x200",
            &[
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-colorspace",
                "smpte170m",
            ],
            "bt601",
            "tv",
        ),
        (
            "bt709-full-bframes",
            "320x240",
            &["-bf", "2", "-pix_fmt", "yuvj420p", "-colorspace", "bt709"],
            "bt709",
            "pc",
        ),
        (
            "bt601-full-bframes",
            "302x198",
            &["-bf", "2", "-pix_fmt", "yuvj420p", "-colorspace", "bt470bg"],
            "bt601",
            "pc",
        ),
        // A stream that declares no matrix is BT.709 in high definition and BT.601 below it.
        (
            "unspecified-hd",
            "1280x720",
            &["-bf", "3", "-pix_fmt", "yuv420p"],
            "bt709",
            "tv",
        ),
        (
            "unspecified-sd",
            "300x200",
            &["-bf", "3", "-pix_fmt", "yuv420p"],
            "bt601",
            "tv",
        ),
        // Composition offsets below zero, which a version 1 table holds.
        (
            "negative-offsets",
            "300x200",
            &[
                "-bf",
                "3",
                "-pix_fmt",
                "yuv420p",
                "-colorspace",
                "bt709",
                "-movflags",
                "+negative_cts_offsets",
            ],
            "bt709",
            "tv",
        ),
    ];
    for (name, size, options, matrix, range) in cases {
        let path = directory.path().join(format!("{name}.mp4"));
        clip(&path, size, options);
        let (nv12, rgb) = reference(&path, matrix, range);

        // The frames are ffmpeg's, bit for bit and in its order.
        let decoded = nv12_frames(&path);
        let (width, height) = size.split_once('x').unwrap();
        let frame = width.parse::<usize>().unwrap() * height.parse::<usize>().unwrap() * 3 / 2;
        let order: Vec<Option<usize>> = decoded
            .chunks(frame)
            .map(|decoded| nv12.chunks(frame).position(|frame| frame == decoded))
            .collect();
        assert_eq!(
            order,
            (0..nv12.len() / frame).map(Some).collect::<Vec<_>>(),
            "{name}: the frames ffmpeg decodes, by the order they come out"
        );

        let mut file = Mp4File::open(&path).unwrap();
        let frames = decode_frames(&mut file, 100).unwrap();
        let [count, height, width, 3] = frames.shape[..] else {
            panic!("{name}: frames of shape {:?}", frames.shape);
        };
        assert_eq!(format!("{width}x{height}"), size, "{name}");
        assert_eq!(count, 30, "{name}");
        let (mut largest, mut total) = (0.0f32, 0.0f64);
        for (value, &expected) in frames.data.iter().zip(&rgb) {
            let error = (value * 255.0 - f32::from(expected)).abs();
            largest = largest.max(error);
            total += f64::from(error);
        }
        let mean = total / rgb.len() as f64;
        println!("{name}: largest error {largest:.2}, mean {mean:.3} of 255");
        assert_eq!(frames.data.len(), rgb.len(), "{name}");
        assert!(largest < 1.0 && mean < 0.25, "{name}: {largest} {mean}");

        // A shorter clip is the start of the longer one.
        let mut file = Mp4File::open(&path).unwrap();
        let start = decode_frames(&mut file, 7).unwrap();
        assert_eq!(start.shape, vec![7, height, width, 3], "{name}");
        assert_eq!(start.data[..], frames.data[..start.data.len()], "{name}");
    }
}
