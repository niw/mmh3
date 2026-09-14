//! The file-writing subset of WebM for VP9 + Opus: tracks, clusters, keyframe cues,
//! CodecDelay/SeekPreRoll and final DiscardPadding. No codec code or native muxer is needed.
//! Element definitions: https://www.matroska.org/technical/elements.html
//! Codec mapping: https://www.matroska.org/technical/codec_specs.html

use crate::webm::timestamp;
use crate::{MediaSpec, Result};
use std::io::{Seek, SeekFrom, Write};

pub(crate) struct Packet {
    pub track: u8,
    pub timestamp_ns: u64,
    pub keyframe: bool,
    pub data: Vec<u8>,
    pub discard_padding_ns: u64,
}

const TIMECODE_SCALE: u64 = 1_000_000; // one millisecond
const SEGMENT: u32 = 0x18538067;
const INFO: u32 = 0x1549a966;
const TRACKS: u32 = 0x1654ae6b;
const CLUSTER: u32 = 0x1f43b675;
const CUES: u32 = 0x1c53bb6b;

fn size(value: u64) -> [u8; 8] {
    // Fixed-width sizes let us patch a master element after writing its payload.
    (value | (1 << 56)).to_be_bytes()
}

fn element(id: u32, data: &[u8]) -> Vec<u8> {
    let id_bytes = id.to_be_bytes();
    let first = id_bytes.iter().position(|&byte| byte != 0).unwrap();
    let mut bytes = Vec::with_capacity(12 + data.len());
    bytes.extend_from_slice(&id_bytes[first..]);
    bytes.extend_from_slice(&size(data.len() as u64));
    bytes.extend_from_slice(data);
    bytes
}

fn uint(id: u32, value: u64) -> Vec<u8> {
    element(id, &value.to_be_bytes())
}
fn float(id: u32, value: f64) -> Vec<u8> {
    element(id, &value.to_be_bytes())
}
fn master(id: u32, children: &[Vec<u8>]) -> Vec<u8> {
    element(id, &children.concat())
}

fn seek_head(info: u64, tracks: u64, cues: u64) -> Vec<u8> {
    master(
        0x114d9b74,
        &[
            // SeekHead
            seek_entry(INFO, info),
            seek_entry(TRACKS, tracks),
            seek_entry(CUES, cues),
        ],
    )
}

fn seek_entry(id: u32, offset: u64) -> Vec<u8> {
    master(
        0x4dbb,
        &[element(0x53ab, &id.to_be_bytes()), uint(0x53ac, offset)],
    )
}

pub(crate) fn write(
    writer: &mut (impl Write + Seek),
    spec: &MediaSpec,
    lookahead: usize,
    packets: &[Packet],
) -> Result<()> {
    writer.write_all(&master(
        0x1a45dfa3,
        &[
            uint(0x4286, 1),
            uint(0x42f7, 1), // EBMLVersion, EBMLReadVersion
            uint(0x42f2, 4),
            uint(0x42f3, 8), // EBMLMaxIDLength, EBMLMaxSizeLength
            element(0x4282, b"webm"),
            uint(0x4287, 4),
            uint(0x4285, 2),
        ],
    ))?;
    writer.write_all(&SEGMENT.to_be_bytes())?;
    let segment_size_position = writer.stream_position()?;
    writer.write_all(&size(0))?;
    let segment_start = writer.stream_position()?;
    writer.write_all(&seek_head(0, 0, 0))?;
    let info_position = writer.stream_position()? - segment_start;
    let duration = timestamp(spec.frames as u64, spec.fps as u64) as f64 / TIMECODE_SCALE as f64;
    writer.write_all(&master(
        INFO,
        &[
            uint(0x2ad7b1, TIMECODE_SCALE),
            float(0x4489, duration),
            element(0x4d80, b"mmh3"),
            element(0x5741, b"mmh3"),
        ],
    ))?;

    let mut opus_head = b"OpusHead".to_vec();
    opus_head.extend([1, spec.channels as u8]);
    opus_head.extend_from_slice(&u16::try_from(lookahead)?.to_le_bytes());
    opus_head.extend_from_slice(&(spec.sample_rate as u32).to_le_bytes());
    opus_head.extend([0, 0, 0]); // output gain = 0, mapping family 0 (mono/stereo)
    let tracks_position = writer.stream_position()? - segment_start;
    writer.write_all(&master(
        TRACKS,
        &[
            master(
                0xae,
                &[
                    uint(0xd7, 1),
                    uint(0x73c5, 1),
                    uint(0x83, 1),
                    uint(0x9c, 0),
                    element(0x86, b"V_VP9"),
                    uint(0x23e383, timestamp(1, spec.fps as u64)), // DefaultDuration
                    master(
                        0xe0,
                        &[
                            uint(0xb0, spec.width as u64),
                            uint(0xba, spec.height as u64),
                            master(
                                0x55b0,
                                &[
                                    // Colour: BT.709, 8-bit, 4:2:0, limited range
                                    uint(0x55b1, 1),
                                    uint(0x55b2, 8),
                                    uint(0x55b3, 1),
                                    uint(0x55b4, 1),
                                    uint(0x55b9, 1),
                                    uint(0x55ba, 1),
                                    uint(0x55bb, 1),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            master(
                0xae,
                &[
                    uint(0xd7, 2),
                    uint(0x73c5, 2),
                    uint(0x83, 2),
                    uint(0x9c, 0),
                    element(0x86, b"A_OPUS"),
                    element(0x63a2, &opus_head),
                    uint(0x56aa, timestamp(lookahead as u64, 48_000)), // CodecDelay
                    uint(0x56bb, 80_000_000),                          // SeekPreRoll
                    master(
                        0xe1,
                        &[float(0xb5, 48_000.0), uint(0x9f, spec.channels as u64)],
                    ),
                ],
            ),
        ],
    ))?;

    let mut cluster = Vec::new();
    let mut cluster_time = 0;
    let mut cues = Vec::new();
    // Keep audio and video at the same timestamp together when opening a keyframe cluster.
    let same_time = |previous: &Packet, next: &Packet| {
        previous.timestamp_ns / TIMECODE_SCALE == next.timestamp_ns / TIMECODE_SCALE
    };
    for group in packets.chunk_by(same_time) {
        let time = group[0].timestamp_ns / TIMECODE_SCALE;
        let has_keyframe = group
            .iter()
            .any(|packet| packet.track == 1 && packet.keyframe);
        if !cluster.is_empty() && (has_keyframe || time - cluster_time > 30_000) {
            writer.write_all(&element(CLUSTER, &cluster))?;
            cluster.clear();
        }
        if cluster.is_empty() {
            cluster_time = time;
            cluster.extend(uint(0xe7, cluster_time));
        }
        for packet in group {
            if packet.track == 1 && packet.keyframe {
                cues.push(master(
                    0xbb,
                    &[
                        uint(0xb3, time), // CueTime
                        master(
                            0xb7,
                            &[
                                uint(0xf7, 1),
                                uint(0xf1, writer.stream_position()? - segment_start),
                                // CueRelativePosition, relative to Cluster data
                                uint(0xf0, cluster.len() as u64),
                            ],
                        ),
                    ],
                ));
            }
            let relative = i16::try_from(time - cluster_time)?;
            let mut block = vec![0x80 | packet.track];
            block.extend_from_slice(&relative.to_be_bytes());
            block.push(if packet.keyframe && packet.discard_padding_ns == 0 {
                0x80
            } else {
                0
            });
            block.extend_from_slice(&packet.data);
            if packet.discard_padding_ns != 0 {
                cluster.extend(master(
                    0xa0,
                    &[
                        element(0xa1, &block),
                        uint(0x9b, 20), // Block, BlockDuration (20 ms)
                        element(0x75a2, &(packet.discard_padding_ns as i64).to_be_bytes()),
                    ],
                ));
            } else {
                cluster.extend(element(0xa3, &block)); // SimpleBlock
            }
        }
    }
    if !cluster.is_empty() {
        writer.write_all(&element(CLUSTER, &cluster))?;
    }
    let cues_position = writer.stream_position()? - segment_start;
    writer.write_all(&master(CUES, &cues))?;
    let end = writer.stream_position()?;
    if end - segment_start >= (1 << 56) - 1 {
        return Err("WebM exceeds EBML size limit".into());
    }
    writer.seek(SeekFrom::Start(segment_size_position))?;
    writer.write_all(&size(end - segment_start))?;
    writer.seek(SeekFrom::Start(segment_start))?;
    writer.write_all(&seek_head(info_position, tracks_position, cues_position))?;
    writer.seek(SeekFrom::Start(end))?;
    Ok(())
}
