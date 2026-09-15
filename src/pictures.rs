//! Picture files, decoded with the `image` crate and turned upright by their EXIF orientation.

use mmh3_core::picture::Picture;
use std::error::Error;
use std::path::Path;

/// Reads a PNG or JPEG file as 8-bit RGB. Alpha is dropped, as Pillow's RGB conversion does.
pub fn load_picture(path: &Path) -> Result<Picture, Box<dyn Error>> {
    use image::{DynamicImage, ImageDecoder, ImageReader};

    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    let rgb = image.to_rgb8();
    Ok(Picture {
        width: rgb.width() as usize,
        height: rgb.height() as usize,
        pixels: rgb.into_raw(),
    })
}
