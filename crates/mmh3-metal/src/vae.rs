//! Video VAE decoding, unpatchification and spatial/temporal blending on Metal.
use crate::{Device, Error, Result, model::Weights, ops::Array};
use mmh3_core::{
    media::Yuv420,
    safetensors::SafeTensors,
    tensor::Tensor,
    vae::{TemporalPlan, TileAxis, rope_angles, split_tiles},
};
use std::cell::OnceCell;

pub const DEFAULT_TILE_SIZE: usize = 256;
pub const DEFAULT_TILE_OVERLAP_MIN: usize = 64;

/// What a decode is told when it asks for a chunk: the canvas another machine decoded, nothing
/// when the chunk was never handed out and this machine is to decode it, or why a chunk that was
/// handed out never came back. Only the middle answer is a reason to read the weights.
///
/// This is `std::result::Result` rather than the crate's own alias, since the error crosses from
/// the caller's world rather than from the device.
pub type RemoteCanvas = std::result::Result<Option<Vec<u8>>, String>;
/// A video decoder that reads its configuration from the checkpoint's header and leaves the
/// weights on disk until something asks it to decode.
///
/// Blending canvases another machine decoded needs no weights at all: the geometry, the tiling and
/// the latent statistics come to a few hundred bytes of header. A leader that hands every chunk to
/// a worker therefore holds no model, and one that has to decode a chunk itself pays for the
/// weights at that point.
pub struct MetalVideoDecoder {
    device: Device,
    file: SafeTensors,
    prefix: String,
    weights: OnceCell<Weights>,
    channels: usize,
    dim: usize,
    layers: usize,
    registers: usize,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    tile_size: usize,
    overlap: usize,
}

pub struct VideoDecoding {
    pub pixels: Tensor,
    pub first_tile: Option<Tensor>,
}

/// Planar RGB frames retained in a Metal buffer. No CPU pixel copy is made until requested.
pub struct MetalVideoFrames {
    start: usize,
    pixels: Array,
    frames: usize,
    height: usize,
    width: usize,
}

impl MetalVideoFrames {
    /// Global frame indices covered by this full video or streamed chunk.
    pub fn frame_range(&self) -> std::ops::Range<usize> {
        self.start..self.start + self.frames
    }

    pub fn shape(&self) -> [usize; 4] {
        [3, self.frames, self.height, self.width]
    }

    pub fn to_pixels(&self) -> Result<Tensor> {
        Ok(Tensor::new(self.shape().to_vec(), self.pixels.to_f32()?))
    }

    /// Uploads planar RGB for native-output tests and callers with existing host frames.
    pub fn from_pixels(device: &crate::Device, pixels: &Tensor) -> Result<Self> {
        Self::from_pixels_at(device, pixels, 0)
    }

    /// Upload a chunk whose first frame has the given global output index.
    pub fn from_pixels_at(device: &crate::Device, pixels: &Tensor, start: usize) -> Result<Self> {
        let &[3, frames, height, width] = pixels.shape.as_slice() else {
            return Err(Error::new(
                "pixels must be [3, frames, height, width]".into(),
            ));
        };

        let columns = frames
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| Error::new("video dimensions overflow".into()))?;
        if start.checked_add(frames).is_none()
            || [frames, height, width].contains(&0)
            || height % 2 != 0
            || width % 2 != 0
            || pixels
                .data
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(Error::new(
                "video needs positive even dimensions and finite RGB in [0, 1]".into(),
            ));
        }

        Ok(Self {
            start,
            pixels: Array::from_f32(device, 3, columns, &pixels.data)?,
            frames,
            height,
            width,
        })
    }

    /// Borrowed MTLBuffer object for native adapters. The caller must keep these frames alive,
    /// must not release or mutate the object, and must finish GPU reads before dropping them.
    pub fn as_metal_buffer(&self) -> Result<*mut std::ffi::c_void> {
        self.pixels.device().synchronize()?;
        Ok(self.pixels.buffer.0.pointer.as_ptr())
    }
}

impl MetalVideoDecoder {
    /// A decoder with its weights on the device, which is what a machine that decodes wants: the
    /// time a caller measures around this call is the time the upload took.
    pub fn load(
        file: &SafeTensors,
        prefix: &str,
        tile_size: usize,
        overlap: usize,
    ) -> Result<Self> {
        let decoder = Self::to_assemble(file, prefix, tile_size, overlap)?;
        decoder.weights()?;
        Ok(decoder)
    }

    /// A decoder for assembling canvases other machines decoded, which needs no weights. It reads
    /// them only if it is left to decode a chunk itself, which is what a leader does with the
    /// chunks that never came back.
    pub fn to_assemble(
        file: &SafeTensors,
        prefix: &str,
        tile_size: usize,
        overlap: usize,
    ) -> Result<Self> {
        if tile_size == 0
            || !tile_size.is_multiple_of(16)
            || !overlap.is_multiple_of(16)
            || overlap >= tile_size
        {
            return Err(Error::new(
                "VAE tile size and overlap must be multiples of 16, with overlap below tile size"
                    .into(),
            ));
        }

        let info = |name: &str| {
            file.get(&format!("{prefix}{name}"))
                .ok_or_else(|| Error::new(format!("missing tensor {prefix}{name}")))
        };
        let dimension = |name: &str, index: usize| -> Result<usize> {
            info(name)?
                .shape
                .get(index)
                .copied()
                .ok_or_else(|| Error::new(format!("{prefix}{name} has no axis {index}")))
        };
        let host = |name: &str| -> Result<Vec<f32>> {
            Ok(Tensor::load(file, info(name)?).map_err(Error::new)?.data)
        };

        let dim = dimension("decoder.x_embedder.weight", 0)?;
        let channels = dimension("decoder.x_embedder.weight", 1)?;
        let registers = dimension("decoder.register_tokens", 1)?;
        let layers = (0..)
            .take_while(|i| {
                file.get(&format!("{prefix}decoder.transformer_blocks.{i}.scale1"))
                    .is_some()
            })
            .count();
        let latents_mean = host("latents_mean")?;
        let latents_std = host("latents_std")?;
        if dim == 0
            || !dim.is_multiple_of(64)
            || layers == 0
            || dimension("decoder.proj_out.weight", 0)? != 3072
            || latents_mean.len() < channels
            || latents_std.len() < channels
        {
            return Err(Error::new("unsupported video decoder configuration".into()));
        }

        Ok(Self {
            device: Device::shared()?,
            file: SafeTensors::open(file.path()).map_err(|error| Error::new(error.to_string()))?,
            prefix: prefix.to_owned(),
            weights: OnceCell::new(),
            channels,
            dim,
            layers,
            registers,
            latents_mean,
            latents_std,
            tile_size,
            overlap,
        })
    }

    /// The weights, uploaded on the first ask. A decoder that only assembles canvases never
    /// asks, so it never pays for the upload.
    fn weights(&self) -> Result<&Weights> {
        if let Some(weights) = self.weights.get() {
            return Ok(weights);
        }

        // NOTE: OnceCell::get_or_try_init is unstable, so the upload happens outside the cell and
        // a second caller racing to it would only build weights that are then dropped. Nothing
        // shares a decoder across threads: the table that keeps one holds it behind a lock.
        let weights = Weights::load_selected(&self.file, &self.prefix, |name| {
            name.starts_with("decoder.") || name.starts_with("post_quant_conv.")
        })?;
        Ok(self.weights.get_or_init(|| weights))
    }

    fn tile(
        &self,
        latent_rows: &[f32],
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Array> {
        let w = self.weights()?;
        let patches = frames * height * width;
        let rows = Array::from_f32(&w.device, patches, self.channels, latent_rows)?;
        let rows = w.linear(&w.linear(&rows, "post_quant_conv")?, "decoder.x_embedder")?;
        let registers = w
            .array("decoder.register_tokens")?
            .reshape(self.registers, self.dim)?;
        let mut x = Array::concat(
            &[rows, registers, Array::zeros(&w.device, 1, self.dim)?],
            false,
        )?;
        let tokens = x.rows;
        let heads = self.dim / 64;
        let angles = Array::from_f32(
            &w.device,
            tokens,
            24,
            &rope_angles(frames, height, width, self.registers + 1, 8, 100.0),
        )?;
        let ones = Array::from_f32(&w.device, 1, 64, &[1.0; 64])?;

        for i in 0..self.layers {
            let p = format!("decoder.transformer_blocks.{i}");
            let qkv = w
                .linear(
                    &w.norm(&x, &format!("{p}.norm1"), 1e-5)?,
                    &format!("{p}.attn.to_qkv"),
                )?
                .reshape(tokens * heads, 192)?;
            let qk = |start| -> Result<Array> {
                qkv.slice(0, tokens * heads, start, 64)?
                    .norm(&ones, 1e-5, false)?
                    .reshape(tokens, self.dim)?
                    .rope(heads, &angles)
            };

            let q = qk(0)?;
            let k = qk(64)?;
            let v = qkv
                .slice(0, tokens * heads, 128, 64)?
                .reshape(tokens, self.dim)?;
            let attended = w.linear(
                &q.attention(&k, &v, heads, heads, false)?,
                &format!("{p}.attn.to_out"),
            )?;
            x = x.add(&attended.mul(&w.vector(&format!("{p}.scale1"))?)?)?;
            let expanded = w.linear(
                &w.norm(&x, &format!("{p}.norm2"), 1e-5)?,
                &format!("{p}.ff.w1"),
            )?;
            let delta = w.linear(&expanded.swiglu()?, &format!("{p}.ff.w2"))?;
            x = x.add(&delta.mul(&w.vector(&format!("{p}.scale2"))?)?)?;
        }

        let projected = w
            .linear(&w.norm(&x, "decoder.norm_out", 1e-5)?, "decoder.proj_out")?
            .slice(0, patches, 0, 3072)?;
        let (f, h, wp) = (frames * 4, height * 16, width * 16);
        let pixels = Array::empty(&w.device, 3, f * h * wp)?;
        w.device.run(
            "video_unpatch",
            &[&projected.buffer, &pixels.buffer],
            &[pixels.len() as u32, f as u32, h as u32, wp as u32],
            pixels.len(),
            false,
        )?;
        Ok(pixels)
    }

    pub fn decode(&self, latent: &Tensor, capture_first_tile: bool) -> Result<VideoDecoding> {
        let (frames, first_tile) =
            self.decode_inner(latent, capture_first_tile, &mut |_| Ok(None))?;
        Ok(VideoDecoding {
            pixels: frames.to_pixels()?,
            first_tile,
        })
    }

    /// The whole video as YUV 4:2:0, which is what a machine with no device of its own asks for.
    /// It cannot blend the chunks, so the blending happens here, where they already are.
    ///
    /// The conversion runs on the host rather than on the device, unlike the CUDA one. It costs a
    /// pass over the pixels once per video, against a kernel that does not exist yet, and what
    /// goes back on the wire is an eighth of what the pixels are either way.
    pub fn decode_yuv420(&self, latent: &Tensor) -> Result<Yuv420> {
        let pixels = self.decode_device(latent)?.to_pixels()?;
        Yuv420::from_pixels(&pixels).map_err(|error| Error::new(error.to_string()))
    }

    pub fn decode_device(&self, latent: &Tensor) -> Result<MetalVideoFrames> {
        self.decode_device_with(latent, &mut |_| Ok(None))
    }

    /// `decode_device` where `remote` may answer with a canvas another machine decoded. It is
    /// asked for the chunks in order and answers with nothing for the ones this machine decodes,
    /// so the temporal tail carries from one chunk to the next as it does without it. It answers
    /// with an error for a chunk it handed out and did not get back, which ends the decode: a
    /// machine that gave its chunks away has no weights to decode them with.
    pub fn decode_device_with(
        &self,
        latent: &Tensor,
        remote: &mut dyn FnMut(usize) -> RemoteCanvas,
    ) -> Result<MetalVideoFrames> {
        Ok(self.decode_inner(latent, false, remote)?.0)
    }

    /// How a latent divides across machines: what a leader needs to hand the chunks round.
    pub fn plan(&self, latent: &Tensor) -> Result<DecodePlan> {
        let geometry = self.geometry(latent)?;
        Ok(DecodePlan {
            chunks: geometry.chunks,
            canvas_frames: geometry.canvas_frames,
            height: geometry.height,
            width: geometry.width,
            tiles: geometry.rows.starts.len() * geometry.columns.starts.len(),
        })
    }

    /// Emit consecutive GPU-resident chunks, without retaining a full RGB video.
    /// A callback error stops decoding immediately.
    pub fn decode_stream(
        &self,
        latent: &Tensor,
        emit: impl FnMut(&MetalVideoFrames) -> Result<()>,
    ) -> Result<()> {
        self.decode_stream_with(latent, &mut |_| Ok(None), emit)
    }

    /// `decode_stream` where `remote` may answer with a canvas another machine decoded, or with
    /// an error for one it handed out and did not get back.
    pub fn decode_stream_with(
        &self,
        latent: &Tensor,
        remote: &mut dyn FnMut(usize) -> RemoteCanvas,
        mut emit: impl FnMut(&MetalVideoFrames) -> Result<()>,
    ) -> Result<()> {
        self.decode_chunks(latent, false, remote, &mut emit)?;
        Ok(())
    }

    /// One chunk's canvas, `[3, canvas frames, height, width]` FP32 as little-endian bytes, before
    /// any blending with its neighbours, which the caller does. Bytes rather than values because
    /// they travel to another machine. Matches the CUDA `decode_chunk` in size and layout, though
    /// not in value: the arithmetic is this backend's own.
    pub fn decode_chunk(&self, latent: &Tensor, chunk: usize) -> Result<Vec<u8>> {
        let geometry = self.geometry(latent)?;
        if chunk >= geometry.chunks {
            return Err(Error::new(format!(
                "chunk {chunk} of a latent with {} chunks",
                geometry.chunks
            )));
        }

        let (canvas, _) = self.fill_canvas(latent, &geometry, chunk, false)?;
        Ok(canvas
            .to_f32()?
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect())
    }

    fn decode_inner(
        &self,
        latent: &Tensor,
        capture_first_tile: bool,
        remote: &mut dyn FnMut(usize) -> RemoteCanvas,
    ) -> Result<(MetalVideoFrames, Option<Tensor>)> {
        let mut output: Option<MetalVideoFrames> = None;
        let first = self.decode_chunks(latent, capture_first_tile, remote, &mut |part| {
            let total = if latent.shape[1] == 1 {
                1
            } else {
                TemporalPlan::new(latent.shape[1]).frames
            };

            let plane = part.height * part.width;
            if output.is_none() {
                output = Some(MetalVideoFrames {
                    start: 0,
                    pixels: Array::empty(&self.device, 3, total * plane)?,
                    frames: total,
                    height: part.height,
                    width: part.width,
                });
            }

            let out = output.as_ref().unwrap();
            self.device.run(
                "copy_rows",
                &[&part.pixels.buffer, &out.pixels.buffer],
                &[
                    part.pixels.len() as u32,
                    part.pixels.cols as u32,
                    out.pixels.cols as u32,
                    0,
                    (part.start * plane) as u32,
                ],
                part.pixels.len(),
                false,
            )
        })?;
        Ok((
            output.ok_or_else(|| Error::new("decoder produced no frames".into()))?,
            first,
        ))
    }

    fn geometry(&self, latent: &Tensor) -> Result<Geometry> {
        let &[channels, latent_frames, lh, lw] = latent.shape.as_slice() else {
            return Err(Error::new(
                "video latent must be [channels, frames, height, width]".into(),
            ));
        };

        if channels != self.channels || [latent_frames, lh, lw].contains(&0) {
            return Err(Error::new("invalid video latent dimensions".into()));
        }

        let (height, width) = (lh * 16, lw * 16);
        let rows = split_tiles(height, self.tile_size, self.overlap);
        let columns = split_tiles(width, self.tile_size, self.overlap);
        let (th, tw) = (rows.length, columns.length);
        let (chunks, cf, output_frames) = if latent_frames == 1 {
            (1, 1, 1)
        } else {
            let plan = TemporalPlan::new(latent_frames);
            (plan.chunks, 7, plan.frames)
        };

        Ok(Geometry {
            channels,
            latent_frames,
            lh,
            lw,
            height,
            width,
            plane: height * width,
            rows,
            columns,
            th,
            tw,
            chunks,
            cf,
            canvas_frames: cf * 4,
            output_frames,
        })
    }

    /// One chunk's canvas, tiled and blended across rows and columns. The temporal tail of the
    /// chunk before it is not blended in: a whole decode does that as it writes frames out, and a
    /// leader handing chunks round does it once the canvases have arrived.
    fn fill_canvas(
        &self,
        latent: &Tensor,
        geometry: &Geometry,
        chunk: usize,
        capture_first_tile: bool,
    ) -> Result<(Array, Option<Tensor>)> {
        let &Geometry {
            channels,
            latent_frames,
            lh,
            lw,
            height,
            width,
            plane,
            th,
            tw,
            cf,
            canvas_frames,
            ..
        } = geometry;
        let (rows, columns) = (&geometry.rows, &geometry.columns);
        let (mean, std) = (&self.latents_mean, &self.latents_std);
        let device = &self.device;
        let canvas = Array::empty(device, 3, canvas_frames * plane)?;
        let mut first_tile = None;
        let mut previous: Vec<Option<Array>> = vec![None; columns.starts.len()];

        for (row, &top) in rows.starts.iter().enumerate() {
            let mut left_tile: Option<Array> = None;
            for (column, &left) in columns.starts.iter().enumerate() {
                let mut values = Vec::with_capacity(cf * (th / 16) * (tw / 16) * channels);
                for t in 0..cf {
                    for y in 0..th / 16 {
                        for x in 0..tw / 16 {
                            for c in 0..channels {
                                let index = ((c * latent_frames
                                    + (chunk * 5 + t).min(latent_frames - 1))
                                    * lh
                                    + top / 16
                                    + y)
                                    * lw
                                    + left / 16
                                    + x;
                                values.push(latent.data[index] * std[c] + mean[c]);
                            }
                        }
                    }
                }

                let tile = self.tile(&values, cf, th / 16, tw / 16)?;
                if capture_first_tile && row == 0 && column == 0 {
                    first_tile = Some(Tensor::new(vec![3, canvas_frames, th, tw], tile.to_f32()?));
                }

                let kept_h = rows
                    .starts
                    .get(row + 1)
                    .map_or(height - top, |next| next - top);
                let kept_w = columns
                    .starts
                    .get(column + 1)
                    .map_or(width - left, |next| next - left);
                let overlap_y = if row == 0 { 0 } else { rows.overlaps[row - 1] };
                let overlap_x = if column == 0 {
                    0
                } else {
                    columns.overlaps[column - 1]
                };

                let above = previous[column].as_ref().unwrap_or(&tile);
                let beside = left_tile.as_ref().unwrap_or(&tile);
                let n = 3 * canvas_frames * kept_h * kept_w;
                device.run(
                    "video_blend_tile",
                    &[&tile.buffer, &above.buffer, &beside.buffer, &canvas.buffer],
                    &[
                        n as u32,
                        width as u32,
                        height as u32,
                        th as u32,
                        tw as u32,
                        top as u32,
                        left as u32,
                        kept_h as u32,
                        kept_w as u32,
                        overlap_y as u32,
                        overlap_x as u32,
                    ],
                    n,
                    false,
                )?;
                previous[column] = Some(tile.clone());
                left_tile = Some(tile);
            }
        }

        Ok((canvas, first_tile))
    }

    /// A canvas another machine decoded, uploaded as it stands. One of the wrong size did not
    /// come from this latent, so it is refused and the caller decodes that chunk here rather than
    /// blending bytes it cannot account for.
    fn upload_canvas(&self, geometry: &Geometry, bytes: &[u8]) -> Option<Array> {
        let columns = geometry.canvas_frames * geometry.plane;
        let expected = 3 * columns * 4;
        if bytes.len() != expected {
            eprintln!(
                "warning: a canvas of {} bytes, not {expected}, so its chunk decodes here",
                bytes.len()
            );
            return None;
        }

        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect();
        Array::from_f32(&self.device, 3, columns, &values).ok()
    }

    fn decode_chunks(
        &self,
        latent: &Tensor,
        capture_first_tile: bool,
        remote: &mut dyn FnMut(usize) -> RemoteCanvas,
        emit: &mut impl FnMut(&MetalVideoFrames) -> Result<()>,
    ) -> Result<Option<Tensor>> {
        let geometry = self.geometry(latent)?;
        let (height, width, plane) = (geometry.height, geometry.width, geometry.plane);
        let (chunks, canvas_frames) = (geometry.chunks, geometry.canvas_frames);
        let (latent_frames, output_frames) = (geometry.latent_frames, geometry.output_frames);
        let device = &self.device;
        let mut first_tile = None;
        let mut tail = Array::zeros(device, 3, 5 * plane)?;
        let mut position = 0;

        for chunk in 0..chunks {
            // A canvas that arrived but does not fit this latent is refused by the upload, which
            // leaves the chunk to be decoded here. Only a chunk that was handed out and never came
            // back is an error.
            let arrived = remote(chunk)
                .map_err(Error::new)?
                .and_then(|bytes| self.upload_canvas(&geometry, &bytes));
            let canvas = match arrived {
                Some(canvas) => canvas,
                None => {
                    let (canvas, tile) = self.fill_canvas(
                        latent,
                        &geometry,
                        chunk,
                        capture_first_tile && chunk == 0,
                    )?;
                    if tile.is_some() {
                        first_tile = tile;
                    }
                    canvas
                }
            };

            let mut write =
                |first: usize, count: usize, blend: bool, position: usize| -> Result<()> {
                    let count = count.min(output_frames.saturating_sub(position));
                    if count == 0 {
                        return Ok(());
                    }

                    let output = Array::empty(device, 3, count * plane)?;
                    let n = output.len();
                    device.run(
                        "video_write_frames",
                        &[&canvas.buffer, &tail.buffer, &output.buffer],
                        &[
                            n as u32,
                            plane as u32,
                            canvas_frames as u32,
                            count as u32,
                            first as u32,
                            count as u32,
                            blend as u32,
                            0,
                        ],
                        n,
                        false,
                    )?;
                    emit(&MetalVideoFrames {
                        start: position,
                        pixels: output,
                        frames: count,
                        height,
                        width,
                    })
                };
            if latent_frames == 1 {
                write(3, 1, false, 0)?;
            } else {
                write(3, 17, chunk > 0, position)?;
                position += 17;
                if chunk + 1 == chunks {
                    write(23, 5, false, position)?;
                } else {
                    tail = canvas.slice(0, 3, 23 * plane, 5 * plane)?;
                }
            }
        }

        Ok(first_tile)
    }
}

/// How a latent divides across machines: the chunks and the canvas each one leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    pub chunks: usize,
    pub canvas_frames: usize,
    pub height: usize,
    pub width: usize,
    pub tiles: usize,
}

impl DecodePlan {
    /// Values in one chunk's canvas, `[3, canvas frames, height, width]`.
    pub fn canvas_values(&self) -> usize {
        3 * self.canvas_frames * self.height * self.width
    }
}

/// How a latent decodes: the tile and chunk geometry that a whole video and a single chunk share.
struct Geometry {
    channels: usize,
    latent_frames: usize,
    lh: usize,
    lw: usize,
    height: usize,
    width: usize,
    plane: usize,
    rows: TileAxis,
    columns: TileAxis,
    th: usize,
    tw: usize,
    chunks: usize,
    cf: usize,
    canvas_frames: usize,
    output_frames: usize,
}
