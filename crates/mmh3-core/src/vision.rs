//! Pictures in the prompt of the Qwen3-VL text encoder: the patches its vision tower embeds, their
//! position embeddings and rotary angles, and the prompt that holds the pictures' vision blocks
//! with the 3D rotary positions of the language model.

use crate::dit::timestep::Modality;
use crate::numeric::{bf16_to_f32, f32_to_bf16};
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;

/// Pixels per side of a patch.
pub const PATCH: usize = 16;
/// Patches per side the merger joins into one embedding.
pub const MERGE: usize = 2;
/// Frames of a patch. A picture fills both with itself.
const TEMPORAL_PATCH: usize = 2;
/// Values of one patch: channels, frames, rows and columns.
pub const PATCH_VALUES: usize = 3 * TEMPORAL_PATCH * PATCH * PATCH;
/// The learned position embeddings form a square grid of this many patches per side.
const POSITION_GRID: usize = 48;
/// Rotary frequencies of the vision attention per axis: head dimension 72 / 4.
const ROTARY_FREQUENCIES: usize = 18;
const ROTARY_THETA: f32 = 10_000.0;

pub const VISION_START: u32 = 151_652;
pub const VISION_END: u32 = 151_653;
/// Stands in for a vision embedding in the token ids.
pub const IMAGE_PAD: u32 = 151_655;

/// Patch rows and columns of a picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionGrid {
    pub height: usize,
    pub width: usize,
}

impl VisionGrid {
    /// The grid of a picture whose sides are multiples of 32 pixels.
    pub fn for_picture(height: usize, width: usize) -> Self {
        assert!(
            height.is_multiple_of(PATCH * MERGE) && width.is_multiple_of(PATCH * MERGE),
            "picture sides must be multiples of {}",
            PATCH * MERGE
        );
        VisionGrid {
            height: height / PATCH,
            width: width / PATCH,
        }
    }

    pub fn patches(&self) -> usize {
        self.height * self.width
    }

    /// Embeddings the merger writes, one per 2 × 2 patches.
    pub fn tokens(&self) -> usize {
        self.patches() / (MERGE * MERGE)
    }

    /// Row and column of each patch in the vision tower's order: blocks of 2 × 2 patches in
    /// row-major order, and row-major within a block.
    pub fn patch_order(&self) -> Vec<(usize, usize)> {
        let mut order = Vec::with_capacity(self.patches());
        for block_row in 0..self.height / MERGE {
            for block_column in 0..self.width / MERGE {
                for row in 0..MERGE {
                    for column in 0..MERGE {
                        order.push((block_row * MERGE + row, block_column * MERGE + column));
                    }
                }
            }
        }
        order
    }
}

/// The patches `[patches, 1536]` of a picture `[height, width, 3]` in [0, 1], normalized to [−1, 1]
/// and ordered like `VisionGrid::patch_order`, each with its values ordered (channel, frame, row,
/// column).
pub fn patchify_picture(picture: &Tensor) -> (VisionGrid, Vec<f32>) {
    let [height, width, 3] = picture.shape[..] else {
        panic!("a picture is [height, width, 3]");
    };
    let grid = VisionGrid::for_picture(height, width);
    let mut patches = Vec::with_capacity(grid.patches() * PATCH_VALUES);
    for (row, column) in grid.patch_order() {
        for channel in 0..3 {
            for _ in 0..TEMPORAL_PATCH {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let (pixel_y, pixel_x) = (row * PATCH + y, column * PATCH + x);
                        let value = picture.data[(pixel_y * width + pixel_x) * 3 + channel];
                        patches.push((value - 0.5) / 0.5);
                    }
                }
            }
        }
    }
    (grid, patches)
}

/// `n` points from 0 to `end` like PyTorch's `linspace` in FP32, which steps from both ends.
fn linspace(end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![0.0];
    }
    let step = end / (n - 1) as f32;
    (0..n)
        .map(|index| {
            if index < n / 2 {
                step * index as f32
            } else {
                end - step * (n - 1 - index) as f32
            }
        })
        .collect()
}

fn bf16(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

/// The position embeddings `[patches, hidden]` of `grid` in patch order, bilinearly interpolated
/// from the learned 48 × 48 grid `table` `[2304, hidden]`. The weights, the products and their sum
/// are rounded to BF16 like the reference, which computes them in the dtype of the table.
pub fn position_embeddings(table: &Tensor, grid: VisionGrid) -> Vec<f32> {
    let hidden = table.shape[1];
    let last = (POSITION_GRID - 1) as f32;
    let rows = linspace(last, grid.height);
    let columns = linspace(last, grid.width);
    let corners = |coordinate: f32| {
        let floor = coordinate as usize;
        (
            floor,
            (floor + 1).min(POSITION_GRID - 1),
            coordinate - floor as f32,
        )
    };
    let mut output = Vec::with_capacity(grid.patches() * hidden);
    for (row, column) in grid.patch_order() {
        let (top, bottom, down) = corners(rows[row]);
        let (left, right, across) = corners(columns[column]);
        let taps = [
            (
                top * POSITION_GRID + left,
                bf16((1.0 - down) * (1.0 - across)),
            ),
            (top * POSITION_GRID + right, bf16((1.0 - down) * across)),
            (bottom * POSITION_GRID + left, bf16(down * (1.0 - across))),
            (bottom * POSITION_GRID + right, bf16(down * across)),
        ];
        for index in 0..hidden {
            let mut sum = 0.0;
            for (tap, &(entry, weight)) in taps.iter().enumerate() {
                let product = bf16(weight * table.data[entry * hidden + index]);
                sum = if tap == 0 {
                    product
                } else {
                    bf16(sum + product)
                };
            }
            output.push(sum);
        }
    }
    output
}

/// The rotary angles `[patches, 36]` of the vision attention in patch order: the first 18 pairs
/// rotate by the patch row and the last 18 by its column. Dimension i of a 72-value head pairs
/// with i + 36.
pub fn rotary_angles(grid: VisionGrid) -> Vec<f32> {
    let frequencies: Vec<f32> = (0..ROTARY_FREQUENCIES)
        .map(|index| 1.0 / ROTARY_THETA.powf((2 * index) as f32 / (2 * ROTARY_FREQUENCIES) as f32))
        .collect();
    grid.patch_order()
        .into_iter()
        .flat_map(|(row, column)| {
            let frequencies = frequencies.clone();
            frequencies
                .iter()
                .map(move |frequency| row as f32 * frequency)
                .chain(
                    frequencies
                        .iter()
                        .map(move |frequency| column as f32 * frequency),
                )
                .collect::<Vec<_>>()
        })
        .collect()
}

/// A prompt with pictures for the text encoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionPrompt {
    /// Token ids, with `IMAGE_PAD` for every vision embedding.
    pub ids: Vec<u32>,
    /// The first token of each picture's vision embeddings.
    pub picture_starts: Vec<usize>,
    /// The DiT's AdaLN branch of each token: video for the vision blocks with their start and
    /// end tokens, text otherwise.
    pub modalities: Vec<Modality>,
    /// The (time, height, width) rotary position of each token in the language model.
    pub positions: Vec<[usize; 3]>,
}

/// A reference the prompt introduces: a picture, which the vision tower embeds, or a sound, which
/// only gets a label since H3 has no audio tower.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptReference {
    Picture(VisionGrid),
    Sound,
}

/// H3's presentation of a prompt with references, without a chat template: for each picture
/// `"<Picture i>: "`, `<|vision_start|>`, its vision embeddings and `<|vision_end|>`, for each
/// sound `"<Audio j>: "`, and then the prompt. Both kinds are numbered from one within their own
/// kind. Text tokens take consecutive positions on all three axes, and a picture's embeddings
/// share the time position of their first one and spread over its rows and columns, after which
/// the text continues past the picture's largest position.
pub fn vision_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    references: &[PromptReference],
) -> VisionPrompt {
    fn push_text(prompt: &mut VisionPrompt, next: &mut usize, ids: &[u32], modality: Modality) {
        for &id in ids {
            prompt.ids.push(id);
            prompt.modalities.push(modality);
            prompt.positions.push([*next; 3]);
            *next += 1;
        }
    }

    let mut result = VisionPrompt {
        ids: Vec::new(),
        picture_starts: Vec::new(),
        modalities: Vec::new(),
        positions: Vec::new(),
    };
    let mut next = 0;
    let (mut pictures, mut sounds) = (0, 0);
    for reference in references {
        let grid = match reference {
            PromptReference::Picture(grid) => {
                pictures += 1;
                let label = tokenizer.encode(&format!("<Picture {pictures}>: "));
                push_text(&mut result, &mut next, &label, Modality::Text);
                grid
            }
            PromptReference::Sound => {
                sounds += 1;
                let label = tokenizer.encode(&format!("<Audio {sounds}>: "));
                push_text(&mut result, &mut next, &label, Modality::Text);
                continue;
            }
        };
        push_text(&mut result, &mut next, &[VISION_START], Modality::Video);
        result.picture_starts.push(result.ids.len());
        let (rows, columns) = (grid.height / MERGE, grid.width / MERGE);
        for row in 0..rows {
            for column in 0..columns {
                result.ids.push(IMAGE_PAD);
                result.modalities.push(Modality::Video);
                result.positions.push([next, next + row, next + column]);
            }
        }
        next += rows.max(columns);
        push_text(&mut result, &mut next, &[VISION_END], Modality::Video);
    }
    push_text(
        &mut result,
        &mut next,
        &tokenizer.encode(prompt),
        Modality::Text,
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_from_both_ends_like_pytorch() {
        let points = linspace(47.0, 84);
        assert_eq!((points[0], points[83]), (0.0, 47.0));
        assert_eq!(points.len(), 84);
        assert_eq!(
            linspace(47.0, 48),
            (0..48).map(|index| index as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn places_two_pictures_in_the_prompt() {
        let grid = VisionGrid::for_picture(768, 1344);
        assert_eq!((grid.height, grid.width, grid.tokens()), (48, 84, 1008));
        let picture = PromptReference::Picture(grid);
        let prompt = vision_prompt(&Tokenizer::h3(), "Rain.", &[picture, picture]);
        assert_eq!(&prompt.ids[..6], &[21604, 3826, 220, 16, 26818, 220]);
        assert_eq!(prompt.ids[6], VISION_START);
        assert_eq!(prompt.picture_starts, vec![7, 7 + 1008 + 1 + 6 + 1]);
        let first = prompt.picture_starts[0];
        assert_eq!(prompt.positions[first], [7, 7, 7]);
        assert_eq!(prompt.positions[first + 1007], [7, 30, 48]);
        // vision_end sits one past the largest position of the picture.
        assert_eq!(prompt.positions[first + 1008], [49; 3]);
        let second = prompt.picture_starts[1];
        assert_eq!(prompt.positions[second - 1], [56; 3]);
        assert_eq!(prompt.positions[second], [57, 57, 57]);
        assert_eq!(prompt.positions[second + 1008 + 1], [100; 3]);
        assert_eq!(prompt.modalities[5], Modality::Text);
        assert_eq!(prompt.modalities[6], Modality::Video);
        assert_eq!(prompt.modalities[first + 1008], Modality::Video);
        assert_eq!(*prompt.modalities.last().unwrap(), Modality::Text);
    }

    #[test]
    fn labels_sounds_after_the_pictures() {
        let tokenizer = Tokenizer::h3();
        let grid = VisionGrid {
            height: 4,
            width: 4,
        };
        let prompt = vision_prompt(
            &tokenizer,
            "Rain.",
            &[
                PromptReference::Picture(grid),
                PromptReference::Sound,
                PromptReference::Sound,
            ],
        );
        // One vision block, then the two labels and the prompt as plain text.
        assert_eq!(prompt.picture_starts.len(), 1);
        let after = prompt.picture_starts[0] + grid.tokens() + 1;
        let labels = tokenizer.encode("<Audio 1>: ");
        assert_eq!(&prompt.ids[after..after + labels.len()], &labels[..]);
        let second = after + labels.len();
        // The label and the prompt are tokenized on their own, so no merge spans their boundary.
        let expected: Vec<u32> = tokenizer
            .encode("<Audio 2>: ")
            .into_iter()
            .chain(tokenizer.encode("Rain."))
            .collect();
        assert_eq!(&prompt.ids[second..], &expected[..]);
        assert!(
            prompt.modalities[after..]
                .iter()
                .all(|modality| *modality == Modality::Text)
        );
        // Text positions stay consecutive across the labels.
        for (offset, position) in prompt.positions[after..].iter().enumerate() {
            let first = prompt.positions[after][0];
            assert_eq!(*position, [first + offset; 3]);
        }
    }

    #[test]
    fn orders_patches_in_blocks() {
        let grid = VisionGrid {
            height: 4,
            width: 4,
        };
        assert_eq!(
            &grid.patch_order()[..6],
            &[(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3)]
        );
        let angles = rotary_angles(grid);
        assert_eq!(angles.len(), 16 * 36);
        // Patch (1, 1): rows rotate the first 18 pairs, columns the others.
        assert_eq!(angles[3 * 36], 1.0);
        assert_eq!(angles[3 * 36 + 18], 1.0);
    }
}
