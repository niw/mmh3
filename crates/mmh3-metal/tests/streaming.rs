#![cfg(target_os = "macos")]
//! A DiT that reads its blocks on the way computes what the whole one does. Its own test binary,
//! since the limit that makes it stream belongs to the process.

use mmh3_core::dit::inputs::DitInputs;
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_metal::dit::MetalDit;
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/dit_tiny.safetensors"
);

fn input(file: &SafeTensors, name: &str) -> Tensor {
    let mut tensor = Tensor::load(file, file.get(name).unwrap()).unwrap();
    tensor.shape.remove(0);
    tensor
}

#[test]
fn a_streamed_dit_computes_what_the_whole_one_does() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let inputs = DitInputs {
        video: input(&file, "input.video"),
        audio: input(&file, "input.audio"),
        context: input(&file, "input.context"),
        context_modalities: Vec::new(),
        keyframes: Vec::new(),
        references: Vec::new(),
        sigma: 0.7,
        shift_video: 3.0,
        shift_audio: 3.0,
    };
    let whole = MetalDit::load(&file, "weight.")
        .unwrap()
        .forward(&inputs, &[], None)
        .unwrap();

    // Less room than the whole DiT and the headroom beside it, so it keeps no block back.
    mmh3_metal::set_allocation_limit(mmh3_metal::allocated_bytes() + (256 << 20));
    let streamed = MetalDit::load_fitting(&file, "weight.").unwrap();
    // The second call reads its blocks into the regions the first one's have left.
    for _ in 0..2 {
        let output = streamed.forward(&inputs, &[], None).unwrap();
        assert_eq!(output.video, whole.video);
        assert_eq!(output.audio, whole.audio);
    }
    drop(streamed);
    mmh3_metal::set_allocation_limit(0);
}
