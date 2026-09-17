//! Video VAE decoding on the GPU.
//!
//! Every spatial tile of a temporal chunk runs through the ViT decoder as one batch. The residual
//! stream stays in FP32 and the linear layers take FP16 activations: FP16 weights run through
//! cuBLASLt, and the INT8 ConvRot weights of the quantized checkpoint through the INT8 GEMM. Tiles
//! and chunks are blended in FP32 into pixels in [0, 1].

use crate::attention::{AttentionLayout, Element, attention_pointers};
use crate::gemm::{Int8Output, int8_pointers, rotate_quantize_pointers};
use crate::loader::Uploader;
use crate::model::{
    CONVROT_GROUP, DeviceTensors, Error, LinearKind, check_quantization, cublaslt_linear,
    host_tensor, i32_buffer,
};
use crate::{CudaError, DeviceBuffer, check};
use mmh3_core::media::Yuv420;
use mmh3_core::numeric::{f16_to_f32, f32_to_f16};
use mmh3_core::safetensors::{DType, SafeTensors};
use mmh3_core::tensor::Tensor;
use mmh3_core::vae::{
    CHUNK_FRAMES, CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, FRAME_OVERLAP, FRAME_PRE_PADDING,
    SPATIAL_RATIO, TEMPORAL_RATIO, TemporalPlan, TileAxis, rope_angles, split_tiles,
};
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_vae_norm(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        tokens: c_int,
        width: c_int,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vae_qk_norm_rope(
        qkv: *mut c_void,
        angles: *const c_void,
        pairs: c_int,
        tile_tokens: c_int,
        tokens: c_int,
        heads: c_int,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vae_residual_add_scaled(
        residual: *mut c_void,
        delta: *const c_void,
        scale: *const c_void,
        tokens: c_int,
        width: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vae_swiglu(
        input: *const c_void,
        output: *mut c_void,
        tokens: c_int,
        width: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_nv12_frame(
        pixels: *const c_void,
        output: *mut c_void,
        frames: c_int,
        height: c_int,
        width: c_int,
        frame: c_int,
        pitch: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_yuv420(
        pixels: *const c_void,
        output: *mut c_void,
        frames: c_int,
        height: c_int,
        width: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vae_add_norm_quantize(
        residual: *mut c_void,
        delta: *const c_void,
        scale: *const c_void,
        weight: *const c_void,
        quantized: *mut c_void,
        scales: *mut c_void,
        tokens: c_int,
        width: c_int,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vae_embed_suffix(
        embedded: *const c_void,
        registers: *const c_void,
        residual: *mut c_void,
        tiles: c_int,
        tile_tokens: c_int,
        patches: c_int,
        registers_count: c_int,
        width: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vae_unpatchify_blend(
        projected: *const c_void,
        starts_y: *const c_void,
        starts_x: *const c_void,
        overlaps_y: *const c_void,
        overlaps_x: *const c_void,
        tiles_y: c_int,
        tiles_x: c_int,
        tile_height: c_int,
        tile_width: c_int,
        latent_frames: c_int,
        tile_tokens: c_int,
        canvas: *mut c_void,
        height: c_int,
        width: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vae_write_frames(
        canvas: *const c_void,
        canvas_frames: c_int,
        first: c_int,
        count: c_int,
        overlap: *const c_void,
        blend_frames: c_int,
        output: *mut c_void,
        output_frames: c_int,
        position: c_int,
        plane: c_int,
        mean: *const f32,
        deviation: *const f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vae_save_overlap(
        canvas: *const c_void,
        canvas_frames: c_int,
        first: c_int,
        count: c_int,
        overlap: *mut c_void,
        plane: c_int,
        stream: *mut c_void,
    ) -> c_int;
}

pub const DEFAULT_TILE_SIZE: usize = 256;
// NOTE: the grid a tile size and this minimum produce, not the minimum itself, sets the decode
// cost: 32 gives 6 x 4 tiles at 1344 x 768 where 48 and above give 7 x 4, for the same distance
// from a finer grid.
pub const DEFAULT_TILE_OVERLAP_MIN: usize = 32;

const HEAD_DIM: usize = 64;
const NORM_EPSILON: f32 = 1e-5;
const ROPE_BASE: f32 = 100.0;
/// Rotary frequencies per axis. Three axes rotate 3 × 8 pairs, the first 48 of the 64 head
/// dimensions.
const ROPE_FREQUENCIES: usize = 8;
const OUTPUT_CHANNELS: usize = 3;
/// Features of one decoded patch: 3 channels × 4 frames × 16 × 16 pixels.
const PATCH_FEATURES: usize = OUTPUT_CHANNELS * TEMPORAL_RATIO * SPATIAL_RATIO * SPATIAL_RATIO;
const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoDecoderConfig {
    pub latent_channels: usize,
    pub dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub ffn: usize,
    pub registers: usize,
}

/// Decoded pixels and, when requested, the raw decoder output of the first tile.
pub struct VideoDecoding {
    /// `[3, frames, height, width]` in [0, 1].
    pub pixels: Tensor,
    /// `[3, tile frames, tile height, tile width]` of the first tile of the first chunk, before any
    /// blending.
    pub first_tile: Option<Tensor>,
}

/// Buffers sized for one chunk of tiles.
struct Workspace {
    latent_rows: DeviceBuffer,
    residual: DeviceBuffer,
    normalized: DeviceBuffer,
    qkv: DeviceBuffer,
    attention: DeviceBuffer,
    delta: DeviceBuffer,
    expanded: DeviceBuffer,
    activated: DeviceBuffer,
    projected: DeviceBuffer,
    /// Rotated INT8 activations and their row scales for INT8 layers.
    quantized: DeviceBuffer,
    scales: DeviceBuffer,
}

impl Workspace {
    fn new(
        config: &VideoDecoderConfig,
        tokens: usize,
        quantized: bool,
        dense_ffn: bool,
    ) -> Result<Self, Error> {
        let dim = config.dim;
        let quantized_rows = if quantized { tokens } else { 1 };
        Ok(Workspace {
            latent_rows: DeviceBuffer::new(tokens * config.latent_channels * 2)?,
            residual: DeviceBuffer::new(tokens * dim * 4)?,
            normalized: DeviceBuffer::new(tokens * dim * 2)?,
            qkv: DeviceBuffer::new(tokens * 3 * dim * 2)?,
            attention: DeviceBuffer::new(tokens * dim * 2)?,
            delta: DeviceBuffer::new(tokens * dim * 2)?,
            // INT8 feed-forward layers write SwiGLU straight from the GEMM.
            expanded: DeviceBuffer::new(if dense_ffn {
                tokens * 2 * config.ffn * 2
            } else {
                1
            })?,
            activated: DeviceBuffer::new(tokens * config.ffn * 2)?,
            projected: DeviceBuffer::new(tokens * PATCH_FEATURES * 2)?,
            quantized: DeviceBuffer::new(quantized_rows * dim.max(config.ffn))?,
            scales: DeviceBuffer::new(quantized_rows * 4)?,
        })
    }
}

/// The tile grid of one decode call, with the geometry on the device for the blend kernel.
struct TileGrid {
    rows: TileAxis,
    columns: TileAxis,
    starts_y: DeviceBuffer,
    starts_x: DeviceBuffer,
    overlaps_y: DeviceBuffer,
    overlaps_x: DeviceBuffer,
}

impl TileGrid {
    fn new(
        height: usize,
        width: usize,
        tile_size: usize,
        overlap_min: usize,
    ) -> Result<Self, Error> {
        let rows = split_tiles(height, tile_size, overlap_min);
        let columns = split_tiles(width, tile_size, overlap_min);
        // A spare element keeps the overlap buffers non-empty for axes with a single tile.
        let padded = |values: &[usize]| [values, &[0]].concat();
        Ok(TileGrid {
            starts_y: i32_buffer(&rows.starts)?,
            starts_x: i32_buffer(&columns.starts)?,
            overlaps_y: i32_buffer(&padded(&rows.overlaps))?,
            overlaps_x: i32_buffer(&padded(&columns.overlaps))?,
            rows,
            columns,
        })
    }

    fn count(&self) -> usize {
        self.rows.starts.len() * self.columns.starts.len()
    }

    fn latent_height(&self) -> usize {
        self.rows.length / SPATIAL_RATIO
    }

    fn latent_width(&self) -> usize {
        self.columns.length / SPATIAL_RATIO
    }
}

/// Converts pixels `[3, frames, height, width]` in [0, 1] on the device into 4:2:0 BT.709
/// limited-range video with the arithmetic of `Yuv420::from_pixels`. Height and width must be even.
pub fn yuv420(
    pixels: &DeviceBuffer,
    frames: usize,
    height: usize,
    width: usize,
) -> Result<Yuv420, CudaError> {
    assert!(
        pixels.bytes() >= OUTPUT_CHANNELS * frames * height * width * 4,
        "pixels are smaller than their shape"
    );
    let output = DeviceBuffer::new(frames * Yuv420::frame_bytes(height, width))?;
    // SAFETY: the pixels hold the shape, checked above, and the output one YUV frame per frame.
    check(unsafe {
        mmh3_yuv420(
            pixels.pointer(),
            output.pointer(),
            frames as c_int,
            height as c_int,
            width as c_int,
            ptr::null_mut(),
        )
    })?;
    let mut data = vec![0u8; output.bytes()];
    output.copy_to_host(&mut data)?;
    Ok(Yuv420 {
        frames,
        height,
        width,
        data,
    })
}

/// Decoded RGB float32 frames retained on the CUDA device. No host readback is performed.
pub struct CudaVideoFrames {
    pixels: DeviceBuffer,
    frames: usize,
    height: usize,
    width: usize,
}

impl CudaVideoFrames {
    pub fn frames(&self) -> usize {
        self.frames
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn width(&self) -> usize {
        self.width
    }
    /// Takes ownership of planar `[3, frames, height, width]` float32 pixels in [0, 1].
    pub fn from_rgb(
        pixels: DeviceBuffer,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Self, Error> {
        let bytes = frames
            .checked_mul(height)
            .and_then(|rows| rows.checked_mul(width))
            .and_then(|pixels| pixels.checked_mul(12));
        if frames == 0
            || height == 0
            || width == 0
            || !height.is_multiple_of(2)
            || !width.is_multiple_of(2)
            || bytes != Some(pixels.bytes())
            || [frames, height, width]
                .iter()
                .any(|&dimension| dimension > i32::MAX as usize)
        {
            return Err(Error::Model("invalid device RGB frame dimensions".into()));
        }
        Ok(Self {
            pixels,
            frames,
            height,
            width,
        })
    }

    /// Converts on the GPU, then copies the YUV planes to host memory.
    pub fn to_yuv420(&self) -> Result<Yuv420, CudaError> {
        yuv420(&self.pixels, self.frames, self.height, self.width)
    }

    /// Converts directly into the encoder's pitched CUDA allocation. The default stream
    /// is synchronized before returning, so a native encoder may immediately read it.
    pub fn write_nv12_frame(
        &self,
        frame: usize,
        output: &mut DeviceBuffer,
        pitch: usize,
    ) -> Result<(), Error> {
        let bytes = pitch
            .checked_mul(self.height)
            .and_then(|luma| luma.checked_mul(3))
            .map(|size| size / 2);
        if frame >= self.frames
            || pitch < self.width
            || bytes.is_none_or(|bytes| bytes > output.bytes())
        {
            return Err(Error::Model(
                "invalid NV12 destination or frame index".into(),
            ));
        }
        // SAFETY: dimensions and allocation bounds are checked above. Both buffers stay alive
        // through synchronization.
        check(unsafe {
            mmh3_nv12_frame(
                self.pixels.pointer(),
                output.pointer(),
                self.frames as i32,
                self.height as i32,
                self.width as i32,
                frame as i32,
                pitch,
                ptr::null_mut(),
            )
        })?;
        crate::synchronize()?;
        Ok(())
    }
}

pub struct CudaVideoDecoder {
    config: VideoDecoderConfig,
    tensors: DeviceTensors,
    /// Whether any linear layer has INT8 ConvRot weights.
    quantized: bool,
    /// Whether any feed-forward layer has FP16 weights and applies SwiGLU separately.
    dense_ffn: bool,
    /// `post_quant_conv` folded into `x_embedder`, FP16 `[dim, latent channels]` and `[dim]`.
    embed_weight: DeviceBuffer,
    embed_bias: DeviceBuffer,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    tile_size: usize,
    tile_overlap_min: usize,
}

fn f16_buffer(values: &[f32]) -> Result<DeviceBuffer, Error> {
    Ok(DeviceBuffer::from_bytes(
        &values
            .iter()
            .flat_map(|&value| f32_to_f16(value).to_le_bytes())
            .collect::<Vec<_>>(),
    )?)
}

impl CudaVideoDecoder {
    /// Uploads the decoder half of a video VAE checkpoint. `prefix` is prepended to every
    /// checkpoint tensor name.
    pub fn load(
        file: &SafeTensors,
        prefix: &str,
        tile_size: usize,
        tile_overlap_min: usize,
    ) -> Result<Self, Error> {
        let shape = |name: &str| {
            file.get(&format!("{prefix}{name}"))
                .map(|info| info.shape.clone())
                .ok_or_else(|| Error::Model(format!("missing tensor {prefix}{name}")))
        };
        let embedder = shape("decoder.x_embedder.weight")?;
        let config = VideoDecoderConfig {
            latent_channels: embedder[1],
            dim: embedder[0],
            heads: embedder[0] / HEAD_DIM,
            layers: (0..)
                .take_while(|layer| {
                    file.get(&format!(
                        "{prefix}decoder.transformer_blocks.{layer}.scale1"
                    ))
                    .is_some()
                })
                .count(),
            ffn: shape("decoder.transformer_blocks.0.ff.w2.weight")?[1],
            registers: shape("decoder.register_tokens")?[1],
        };
        if !config.dim.is_multiple_of(HEAD_DIM)
            || shape("decoder.proj_out.weight")?[0] != PATCH_FEATURES
        {
            return Err(Error::Model(format!(
                "unsupported video decoder {config:?}"
            )));
        }
        if !tile_size.is_multiple_of(SPATIAL_RATIO)
            || !tile_overlap_min.is_multiple_of(SPATIAL_RATIO)
        {
            return Err(Error::Model(format!(
                "tile size {tile_size} and overlap {tile_overlap_min} must be multiples of {SPATIAL_RATIO}"
            )));
        }

        let mut tensors = DeviceTensors::default();
        let mut uploader = Uploader::new(file);
        let mut quantized = false;
        for info in file.tensors() {
            let Some(name) = info
                .name
                .strip_prefix(prefix)
                .and_then(|name| name.strip_prefix("decoder."))
            else {
                continue;
            };
            if name.starts_with("x_embedder.") || name == "mask_token" {
                continue;
            }
            if name.ends_with(".comfy_quant") {
                check_quantization(name, file.data(info))?;
                continue;
            }
            match info.dtype {
                DType::F16 | DType::I8 => tensors.insert(name, file, info, &mut uploader)?,
                DType::F32 if name.ends_with(".weight_scale") => {
                    tensors.insert(name, file, info, &mut uploader)?
                }
                DType::F32 => {
                    // NOTE: ComfyUI runs this decoder in FP16 and casts every unquantized tensor to
                    // FP16, the biases of INT8 layers included. Those stay FP32 here for the INT8
                    // GEMM's epilogue, rounded through FP16.
                    let values = Tensor::load(file, info).map_err(Error::Model)?.data;
                    let int8_layer = name
                        .strip_suffix(".bias")
                        .and_then(|layer| file.get(&format!("{prefix}decoder.{layer}.weight")))
                        .is_some_and(|weight| weight.dtype == DType::I8);
                    if int8_layer {
                        let rounded: Vec<f32> = values
                            .iter()
                            .map(|&value| f16_to_f32(f32_to_f16(value)))
                            .collect();
                        tensors.insert_buffer(
                            name,
                            DeviceBuffer::from_f32(&rounded)?,
                            DType::F32,
                            info.shape.clone(),
                        );
                    } else {
                        tensors.insert_buffer(
                            name,
                            f16_buffer(&values)?,
                            DType::F16,
                            info.shape.clone(),
                        );
                    }
                }
                other => {
                    return Err(Error::Model(format!(
                        "decoder.{name}: unsupported dtype {other}"
                    )));
                }
            }
            quantized |= info.dtype == DType::I8;
        }
        uploader.run()?;
        let mut dense_ffn = false;
        for layer in 0..config.layers {
            let w1 = format!("transformer_blocks.{layer}.ff.w1");
            if tensors.is_int8(&w1) {
                for suffix in ["weight", "weight_scale", "bias"] {
                    tensors.interleave_swiglu(&format!("{w1}.{suffix}"))?;
                }
            } else {
                dense_ffn = true;
            }
        }

        let channels = config.latent_channels;
        let embed = host_tensor(file, &format!("{prefix}decoder.x_embedder.weight"))?.data;
        let embed_bias = host_tensor(file, &format!("{prefix}decoder.x_embedder.bias"))?.data;
        let post = host_tensor(file, &format!("{prefix}post_quant_conv.weight"))?.data;
        let post_bias = host_tensor(file, &format!("{prefix}post_quant_conv.bias"))?.data;
        let mut folded = vec![0.0f32; config.dim * channels];
        let mut folded_bias = embed_bias.clone();
        for output in 0..config.dim {
            for middle in 0..channels {
                let weight = embed[output * channels + middle];
                folded_bias[output] += weight * post_bias[middle];
                for input in 0..channels {
                    folded[output * channels + input] += weight * post[middle * channels + input];
                }
            }
        }

        Ok(CudaVideoDecoder {
            embed_weight: f16_buffer(&folded)?,
            embed_bias: f16_buffer(&folded_bias)?,
            latents_mean: host_tensor(file, &format!("{prefix}latents_mean"))?.data,
            latents_std: host_tensor(file, &format!("{prefix}latents_std"))?.data,
            config,
            tensors,
            quantized,
            dense_ffn,
            tile_size,
            tile_overlap_min,
        })
    }

    pub fn config(&self) -> &VideoDecoderConfig {
        &self.config
    }

    /// Applies layer `name` to FP16 rows. `swiglu` writes `silu(gate) · up` of an INT8 layer whose
    /// rows went through `interleave_swiglu`, `rows × outputs / 2` values, and `input_quantized`
    /// says that the workspace already holds the rotated INT8 rows of an INT8 layer's input.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        name: &str,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        workspace: &Workspace,
        swiglu: bool,
        input_quantized: bool,
    ) -> Result<(), Error> {
        let weight = self.tensors.get(&format!("{name}.weight"))?;
        let bias = self.tensors.pointer(&format!("{name}.bias"))?;
        let (outputs, features) = (weight.shape[0], weight.shape[1]);
        let output_columns = if swiglu { outputs / 2 } else { outputs };
        assert!(
            input.bytes() >= rows * features * 2 && output.bytes() >= rows * output_columns * 2,
            "{name}: buffers are too small"
        );
        if weight.dtype != DType::I8 {
            assert!(!swiglu, "{name}: SwiGLU output needs INT8 weights");
            // SAFETY: both buffers hold the rows, checked above, and the weight and bias come from
            // the checkpoint.
            unsafe {
                cublaslt_linear(
                    LinearKind::F16,
                    input.pointer(),
                    weight.buffer.pointer(),
                    bias,
                    output.pointer(),
                    rows,
                    outputs,
                    features,
                )?
            };
            return Ok(());
        }
        if features % CONVROT_GROUP != 0 {
            return Err(Error::Model(format!("{name}: unsupported INT8 layer")));
        }
        assert!(
            workspace.quantized.bytes() >= rows * features && workspace.scales.bytes() >= rows * 4,
            "{name}: quantization buffers are too small"
        );
        let weight_scales = self.tensors.pointer(&format!("{name}.weight_scale"))?;
        // SAFETY: the quantization buffers hold the rows, checked above, and the bias holds
        // `outputs` f32 values.
        unsafe {
            if !input_quantized {
                rotate_quantize_pointers(
                    input.pointer(),
                    true,
                    workspace.quantized.pointer(),
                    workspace.scales.pointer(),
                    rows,
                    features,
                )?;
            }
            let output = Int8Output {
                pointer: output.pointer(),
                f16: true,
                bias,
                swiglu,
            };
            int8_pointers(
                workspace.quantized.pointer(),
                weight.buffer.pointer(),
                workspace.scales.pointer(),
                weight_scales,
                output,
                rows,
                outputs,
                features,
                None,
            )?;
        }
        Ok(())
    }

    /// Normalizes the residual stream into `workspace.normalized`: RMSNorm, or LayerNorm when
    /// `bias` is given.
    fn normalize(
        &self,
        weight: &str,
        bias: Option<&str>,
        workspace: &Workspace,
        tokens: usize,
    ) -> Result<(), Error> {
        let bias = match bias {
            Some(name) => self.tensors.pointer(name)?,
            None => ptr::null(),
        };
        // SAFETY: the workspace holds `tokens × dim` residual and normalized values.
        check(unsafe {
            mmh3_vae_norm(
                workspace.residual.pointer(),
                self.tensors.pointer(weight)?,
                bias,
                workspace.normalized.pointer(),
                tokens as c_int,
                self.config.dim as c_int,
                NORM_EPSILON,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    /// Adds `scale ⊙ delta` to the residual when `scale` is given, then normalizes the residual
    /// with `weight` and quantizes it into the workspace for an INT8 layer.
    fn add_norm_quantize(
        &self,
        scale: Option<&str>,
        weight: &str,
        workspace: &Workspace,
        tokens: usize,
    ) -> Result<(), Error> {
        let dim = self.config.dim;
        assert!(
            workspace.quantized.bytes() >= tokens * dim && workspace.scales.bytes() >= tokens * 4,
            "the quantization buffers are too small"
        );
        let (delta, scale) = match scale {
            Some(name) => (
                workspace.delta.pointer().cast_const(),
                self.tensors.pointer(name)?,
            ),
            None => (ptr::null(), ptr::null()),
        };
        // SAFETY: the residual and delta hold `tokens × dim` values, the quantization buffers were
        // checked above, and the scale and weight hold `dim` values.
        check(unsafe {
            mmh3_vae_add_norm_quantize(
                workspace.residual.pointer(),
                delta,
                scale,
                self.tensors.pointer(weight)?,
                workspace.quantized.pointer(),
                workspace.scales.pointer(),
                tokens as c_int,
                dim as c_int,
                NORM_EPSILON,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    fn add_scaled(&self, scale: &str, workspace: &Workspace, tokens: usize) -> Result<(), Error> {
        // SAFETY: residual and delta hold `tokens × dim` values and the scale `dim`.
        check(unsafe {
            mmh3_vae_residual_add_scaled(
                workspace.residual.pointer(),
                workspace.delta.pointer(),
                self.tensors.pointer(scale)?,
                tokens as c_int,
                self.config.dim as c_int,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    /// Runs the decoder on `workspace.latent_rows`, `tiles` tiles of `tile_tokens` rows each, and
    /// leaves the patch features in `workspace.projected`.
    fn decode_tiles(
        &self,
        workspace: &Workspace,
        tiles: usize,
        tile_tokens: usize,
        patches: usize,
        angles: &DeviceBuffer,
    ) -> Result<(), Error> {
        let config = &self.config;
        let (dim, tokens) = (config.dim, tiles * tile_tokens);
        // SAFETY: the workspace is sized for `tokens` rows and the embedder is
        // `[dim, latent channels]`.
        unsafe {
            cublaslt_linear(
                LinearKind::F16,
                workspace.latent_rows.pointer(),
                self.embed_weight.pointer(),
                self.embed_bias.pointer(),
                workspace.normalized.pointer(),
                tokens,
                dim,
                config.latent_channels,
            )?;
            check(mmh3_vae_embed_suffix(
                workspace.normalized.pointer(),
                self.tensors.pointer("register_tokens")?,
                workspace.residual.pointer(),
                tiles as c_int,
                tile_tokens as c_int,
                patches as c_int,
                config.registers as c_int,
                dim as c_int,
                ptr::null_mut(),
            ))?;
        }

        let qkv_token_stride = (3 * dim) as i64;
        let layout = AttentionLayout {
            token_stride: [
                qkv_token_stride,
                qkv_token_stride,
                qkv_token_stride,
                dim as i64,
            ],
            head_stride: [
                3 * HEAD_DIM as i64,
                3 * HEAD_DIM as i64,
                3 * HEAD_DIM as i64,
                HEAD_DIM as i64,
            ],
            batch_stride: [
                tile_tokens as i64 * qkv_token_stride,
                tile_tokens as i64 * qkv_token_stride,
                tile_tokens as i64 * qkv_token_stride,
                (tile_tokens * dim) as i64,
            ],
            ..AttentionLayout::default()
        };
        // The second residual add of a block waits for the next block's first normalization when
        // that one quantizes.
        let mut pending_scale: Option<String> = None;
        for layer in 0..config.layers {
            let prefix = format!("transformer_blocks.{layer}");
            let to_qkv = format!("{prefix}.attn.to_qkv");
            if self.tensors.is_int8(&to_qkv) {
                self.add_norm_quantize(
                    pending_scale.take().as_deref(),
                    &format!("{prefix}.norm1.weight"),
                    workspace,
                    tokens,
                )?;
                self.linear(
                    &to_qkv,
                    &workspace.normalized,
                    &workspace.qkv,
                    tokens,
                    workspace,
                    false,
                    true,
                )?;
            } else {
                if let Some(scale) = pending_scale.take() {
                    self.add_scaled(&scale, workspace, tokens)?;
                }
                self.normalize(&format!("{prefix}.norm1.weight"), None, workspace, tokens)?;
                self.linear(
                    &to_qkv,
                    &workspace.normalized,
                    &workspace.qkv,
                    tokens,
                    workspace,
                    false,
                    false,
                )?;
            }
            let qkv = workspace.qkv.pointer().cast::<u16>();
            // SAFETY: qkv holds `tokens × heads × 3 × 64` values, the angles cover a tile and the
            // layout stays within the `tiles × tile_tokens` rows of qkv and attention.
            unsafe {
                check(mmh3_vae_qk_norm_rope(
                    qkv.cast(),
                    angles.pointer(),
                    (3 * ROPE_FREQUENCIES) as c_int,
                    tile_tokens as c_int,
                    tokens as c_int,
                    config.heads as c_int,
                    NORM_EPSILON,
                    ptr::null_mut(),
                ))?;
                attention_pointers(
                    Element::F16,
                    HEAD_DIM,
                    qkv.cast(),
                    qkv.add(HEAD_DIM).cast(),
                    qkv.add(2 * HEAD_DIM).cast(),
                    workspace.attention.pointer(),
                    tile_tokens,
                    config.heads,
                    tiles,
                    &layout,
                    1.0 / (HEAD_DIM as f32).sqrt(),
                )?;
            }
            self.linear(
                &format!("{prefix}.attn.to_out"),
                &workspace.attention,
                &workspace.delta,
                tokens,
                workspace,
                false,
                false,
            )?;

            let w1 = format!("{prefix}.ff.w1");
            if self.tensors.is_int8(&w1) {
                self.add_norm_quantize(
                    Some(&format!("{prefix}.scale1")),
                    &format!("{prefix}.norm2.weight"),
                    workspace,
                    tokens,
                )?;
                self.linear(
                    &w1,
                    &workspace.normalized,
                    &workspace.activated,
                    tokens,
                    workspace,
                    true,
                    true,
                )?;
            } else {
                self.add_scaled(&format!("{prefix}.scale1"), workspace, tokens)?;
                self.normalize(&format!("{prefix}.norm2.weight"), None, workspace, tokens)?;
                self.linear(
                    &w1,
                    &workspace.normalized,
                    &workspace.expanded,
                    tokens,
                    workspace,
                    false,
                    false,
                )?;
                // SAFETY: expanded holds `tokens × 2 × ffn` values and activated `tokens × ffn`.
                check(unsafe {
                    mmh3_vae_swiglu(
                        workspace.expanded.pointer(),
                        workspace.activated.pointer(),
                        tokens as c_int,
                        config.ffn as c_int,
                        ptr::null_mut(),
                    )
                })?;
            }
            self.linear(
                &format!("{prefix}.ff.w2"),
                &workspace.activated,
                &workspace.delta,
                tokens,
                workspace,
                false,
                false,
            )?;
            pending_scale = Some(format!("{prefix}.scale2"));
        }
        if let Some(scale) = pending_scale {
            self.add_scaled(&scale, workspace, tokens)?;
        }
        self.normalize("norm_out.weight", Some("norm_out.bias"), workspace, tokens)?;
        self.linear(
            "proj_out",
            &workspace.normalized,
            &workspace.projected,
            tokens,
            workspace,
            false,
            false,
        )
    }

    /// The latent rows of every tile of one chunk, `[tiles, tile tokens, channels]` in FP16.
    /// `latent` is the denormalized latent `[channels, frames, height, width]` and the suffix rows
    /// stay zero.
    fn tile_rows(
        &self,
        latent: &Tensor,
        first_frame: usize,
        frames: usize,
        grid: &TileGrid,
        tile_tokens: usize,
    ) -> Vec<u8> {
        let channels = self.config.latent_channels;
        let (latent_frames, height, width) = (latent.shape[1], latent.shape[2], latent.shape[3]);
        let (tile_height, tile_width) = (grid.latent_height(), grid.latent_width());
        let mut rows = vec![0u16; grid.count() * tile_tokens * channels];
        let mut tile = 0;
        for &start_y in &grid.rows.starts {
            for &start_x in &grid.columns.starts {
                for frame in 0..frames {
                    // Frames past the end repeat the last latent frame.
                    let source_frame = (first_frame + frame).min(latent_frames - 1);
                    for y in 0..tile_height {
                        for x in 0..tile_width {
                            let token = (frame * tile_height + y) * tile_width + x;
                            let row =
                                &mut rows[(tile * tile_tokens + token) * channels..][..channels];
                            for (channel, value) in row.iter_mut().enumerate() {
                                let source = ((channel * latent_frames + source_frame) * height
                                    + start_y / SPATIAL_RATIO
                                    + y)
                                    * width
                                    + start_x / SPATIAL_RATIO
                                    + x;
                                *value = f32_to_f16(latent.data[source]);
                            }
                        }
                    }
                }
                tile += 1;
            }
        }
        rows.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    /// Reads the first tile's patch features back as `[3, frames, height, width]`.
    fn first_tile(
        &self,
        workspace: &Workspace,
        frames: usize,
        grid: &TileGrid,
    ) -> Result<Tensor, Error> {
        let (tile_height, tile_width) = (grid.latent_height(), grid.latent_width());
        let patches = frames * tile_height * tile_width;
        let mut bytes = vec![0u8; patches * PATCH_FEATURES * 2];
        workspace.projected.copy_range_to_host(0, &mut bytes)?;
        let (pixel_frames, pixel_height, pixel_width) = (
            frames * TEMPORAL_RATIO,
            grid.rows.length,
            grid.columns.length,
        );
        let mut data = vec![0.0f32; OUTPUT_CHANNELS * pixel_frames * pixel_height * pixel_width];
        for (index, &pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
            let (token, feature) = (index / PATCH_FEATURES, index % PATCH_FEATURES);
            let (frame, y, x) = (
                token / (tile_height * tile_width),
                token / tile_width % tile_height,
                token % tile_width,
            );
            let channel = feature / (TEMPORAL_RATIO * SPATIAL_RATIO * SPATIAL_RATIO);
            let sub_frame = feature / (SPATIAL_RATIO * SPATIAL_RATIO) % TEMPORAL_RATIO;
            let (sub_y, sub_x) = (
                feature / SPATIAL_RATIO % SPATIAL_RATIO,
                feature % SPATIAL_RATIO,
            );
            let target_frame = frame * TEMPORAL_RATIO + sub_frame;
            let target = ((channel * pixel_frames + target_frame) * pixel_height
                + y * SPATIAL_RATIO
                + sub_y)
                * pixel_width
                + x * SPATIAL_RATIO
                + sub_x;
            data[target] = f16_to_f32(u16::from_le_bytes(pair));
        }
        Ok(Tensor::new(
            vec![OUTPUT_CHANNELS, pixel_frames, pixel_height, pixel_width],
            data,
        ))
    }

    /// Decodes a normalized latent `[channels, frames, height, width]` into pixels
    /// `[3, frames, height, width]` in [0, 1]. A latent of `f > 1` frames decodes in overlapping
    /// chunks of 7 latent frames.
    pub fn decode(
        &self,
        latent: &Tensor,
        capture_first_tile: bool,
    ) -> Result<VideoDecoding, Error> {
        let (pixels, [frames, height, width], first_tile) =
            self.decode_on_device(latent, capture_first_tile)?;
        Ok(VideoDecoding {
            pixels: Tensor::new(
                vec![OUTPUT_CHANNELS, frames, height, width],
                pixels.to_f32()?,
            ),
            first_tile,
        })
    }

    /// Decodes a latent into RGB float32 frames retained on the GPU for native encoding.
    pub fn decode_device(&self, latent: &Tensor) -> Result<CudaVideoFrames, Error> {
        let (pixels, [frames, height, width], _) = self.decode_on_device(latent, false)?;
        CudaVideoFrames::from_rgb(pixels, frames, height, width)
    }

    /// Decodes a latent into 4:2:0 BT.709 limited-range video, converted on the GPU. The bytes
    /// match `Yuv420::from_pixels` of `decode`'s pixels.
    pub fn decode_yuv420(&self, latent: &Tensor) -> Result<Yuv420, Error> {
        let (pixels, [frames, height, width], _) = self.decode_on_device(latent, false)?;
        Ok(yuv420(&pixels, frames, height, width)?)
    }

    /// Decodes a latent into pixels `[3, frames, height, width]` on the device and returns them
    /// with their shape.
    fn decode_on_device(
        &self,
        latent: &Tensor,
        capture_first_tile: bool,
    ) -> Result<(DeviceBuffer, [usize; 3], Option<Tensor>), Error> {
        let config = &self.config;
        let &[channels, latent_frames, latent_height, latent_width] = latent.shape.as_slice()
        else {
            return Err(Error::Model(format!(
                "latent shape {:?} is not [channels, frames, height, width]",
                latent.shape
            )));
        };
        if channels != config.latent_channels || latent_frames == 0 {
            return Err(Error::Model(format!(
                "latent shape {:?} does not match the decoder",
                latent.shape
            )));
        }
        let plane_size = latent_frames * latent_height * latent_width;
        let denormalized = Tensor::new(
            latent.shape.clone(),
            latent
                .data
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    value * self.latents_std[index / plane_size]
                        + self.latents_mean[index / plane_size]
                })
                .collect(),
        );

        let (height, width) = (latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO);
        let plane = height * width;
        let grid = TileGrid::new(height, width, self.tile_size, self.tile_overlap_min)?;
        let tiles = grid.count();
        // A single latent frame decodes on its own and keeps only its last frame.
        let (chunks, chunk_frames, padded_frames, output_frames) = if latent_frames == 1 {
            (1, 1, 1, 1)
        } else {
            let plan = TemporalPlan::new(latent_frames);
            (
                plan.chunks,
                CHUNK_TOKENS + CHUNK_OVERLAP_TOKENS,
                latent_frames + plan.pad_tokens,
                plan.frames,
            )
        };
        let canvas_frames = chunk_frames * TEMPORAL_RATIO;
        let patches = chunk_frames * grid.latent_height() * grid.latent_width();
        let tile_tokens = patches + config.registers + 1;
        let tokens = tiles * tile_tokens;

        let mut workspace = Workspace::new(config, tokens, self.quantized, self.dense_ffn)?;
        let angles = DeviceBuffer::from_f32(&rope_angles(
            chunk_frames,
            grid.latent_height(),
            grid.latent_width(),
            config.registers + 1,
            ROPE_FREQUENCIES,
            ROPE_BASE,
        ))?;
        let canvas = DeviceBuffer::new(OUTPUT_CHANNELS * canvas_frames * plane * 4)?;
        let overlap = DeviceBuffer::new(OUTPUT_CHANNELS * FRAME_OVERLAP * plane * 4)?;
        let output = DeviceBuffer::new(OUTPUT_CHANNELS * output_frames * plane * 4)?;

        let write =
            |first: usize, count: usize, blend: bool, position: usize| -> Result<usize, Error> {
                // SAFETY: the canvas holds `canvas_frames` frames, the overlap `FRAME_OVERLAP` and
                // the output `output_frames`, and the kernel skips frames past the end of the
                // output.
                check(unsafe {
                    mmh3_vae_write_frames(
                        canvas.pointer(),
                        canvas_frames as c_int,
                        first as c_int,
                        count as c_int,
                        if blend {
                            overlap.pointer()
                        } else {
                            ptr::null_mut()
                        },
                        if blend { FRAME_OVERLAP as c_int } else { 0 },
                        output.pointer(),
                        output_frames as c_int,
                        position as c_int,
                        plane as c_int,
                        PIXEL_MEAN.as_ptr(),
                        PIXEL_STD.as_ptr(),
                        ptr::null_mut(),
                    )
                })?;
                Ok(position + count.min(output_frames - position))
            };

        let mut first_tile = None;
        let mut position = 0;
        for chunk in 0..chunks {
            let (first_frame, last_frame) = TemporalPlan::chunk_tokens(chunk, padded_frames);
            if last_frame - first_frame != chunk_frames {
                return Err(Error::Model(format!(
                    "chunk {chunk} covers latent frames {first_frame}..{last_frame}"
                )));
            }
            workspace.latent_rows.copy_from_host(&self.tile_rows(
                &denormalized,
                first_frame,
                chunk_frames,
                &grid,
                tile_tokens,
            ))?;
            self.decode_tiles(&workspace, tiles, tile_tokens, patches, &angles)?;
            if capture_first_tile && chunk == 0 {
                first_tile = Some(self.first_tile(&workspace, chunk_frames, &grid)?);
            }
            // SAFETY: projected holds every tile's patch features and the canvas `canvas_frames`
            // full frames.
            check(unsafe {
                mmh3_vae_unpatchify_blend(
                    workspace.projected.pointer(),
                    grid.starts_y.pointer(),
                    grid.starts_x.pointer(),
                    grid.overlaps_y.pointer(),
                    grid.overlaps_x.pointer(),
                    grid.rows.starts.len() as c_int,
                    grid.columns.starts.len() as c_int,
                    grid.rows.length as c_int,
                    grid.columns.length as c_int,
                    chunk_frames as c_int,
                    tile_tokens as c_int,
                    canvas.pointer(),
                    height as c_int,
                    width as c_int,
                    ptr::null_mut(),
                )
            })?;
            if latent_frames == 1 {
                write(canvas_frames - 1, 1, false, 0)?;
                continue;
            }
            // NOTE: each chunk splits into frames [3, 20), written after blending its start with
            // the previous chunk's tail, and the tail [23, 28), kept for the next chunk or written
            // after the last one.
            position = write(
                FRAME_PRE_PADDING,
                CHUNK_FRAMES - FRAME_PRE_PADDING,
                chunk > 0,
                position,
            )?;
            let (tail_first, tail_count) = (
                CHUNK_FRAMES + FRAME_PRE_PADDING,
                canvas_frames - CHUNK_FRAMES - FRAME_PRE_PADDING,
            );
            if chunk + 1 < chunks {
                // SAFETY: the canvas holds the tail frames and the overlap buffer `FRAME_OVERLAP`
                // frames.
                check(unsafe {
                    mmh3_vae_save_overlap(
                        canvas.pointer(),
                        canvas_frames as c_int,
                        tail_first as c_int,
                        tail_count as c_int,
                        overlap.pointer(),
                        plane as c_int,
                        ptr::null_mut(),
                    )
                })?;
            } else {
                position = write(tail_first, tail_count, false, position)?;
            }
        }
        Ok((output, [output_frames, height, width], first_tile))
    }
}
