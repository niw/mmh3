//! ISO BMFF writer for H.264/AAC. The movie header precedes media (fast start).
//! Each sample is one chunk. Audio edits remove encoder priming and end padding.
use super::{AudioTrack, H264Config, Sample};
use crate::{MediaSpec, Result};
use std::io::Write;

fn atom(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut bytes = u32::try_from(data.len() + 8)
        .expect("MP4 metadata exceeds 4 GiB")
        .to_be_bytes()
        .to_vec();
    bytes.extend(kind);
    bytes.extend(data);
    bytes
}
fn full(kind: &[u8; 4], version: u8, flags: u32, data: &[u8]) -> Vec<u8> {
    let mut bytes = ((u32::from(version) << 24) | flags).to_be_bytes().to_vec();
    bytes.extend(data);
    atom(kind, &bytes)
}
fn u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_be_bytes()).collect()
}
fn matrix() -> Vec<u8> {
    u32s(&[0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000])
}
fn descriptor(tag: u8, data: &[u8]) -> Vec<u8> {
    let mut length = data.len() as u32;
    let mut sizes = vec![(length & 127) as u8];
    length >>= 7;
    while length > 0 {
        sizes.push((length & 127) as u8 | 128);
        length >>= 7;
    }
    let mut bytes = vec![tag];
    bytes.extend(sizes.into_iter().rev());
    bytes.extend(data);
    bytes
}
fn avc1(spec: &MediaSpec, config: &H264Config) -> Result<Vec<u8>> {
    let mut avcc = vec![1, config.sps[1], config.sps[2], config.sps[3], 255, 225];
    avcc.extend(u16::try_from(config.sps.len())?.to_be_bytes());
    avcc.extend(&config.sps);
    avcc.push(1);
    avcc.extend(u16::try_from(config.pps.len())?.to_be_bytes());
    avcc.extend(&config.pps);
    // ISO/IEC 14496-15 repeats the chroma format and bit depths for these High profiles.
    if matches!(config.sps[1], 100 | 110 | 122 | 144) {
        let [chroma_format, luma_depth, chroma_depth] = high_profile_format(&config.sps)?;
        avcc.extend([0xfc | chroma_format, 0xf8 | luma_depth, 0xf8 | chroma_depth, 0]);
    }
    let mut entry = vec![0; 6];
    entry.extend(1u16.to_be_bytes());
    entry.extend([0; 16]);
    entry.extend((spec.width as u16).to_be_bytes());
    entry.extend((spec.height as u16).to_be_bytes());
    entry.extend(u32s(&[0x480000, 0x480000, 0]));
    entry.extend(1u16.to_be_bytes());
    entry.extend([0; 32]);
    entry.extend(24u16.to_be_bytes());
    entry.extend(0xffffu16.to_be_bytes());
    entry.extend(atom(b"avcC", &avcc));
    let mut color = b"nclx".to_vec();
    color.extend([0, 1, 0, 1, 0, 1, 0]);
    entry.extend(atom(b"colr", &color));
    Ok(atom(b"avc1", &entry))
}
/// Reads bits from an SPS RBSP, after emulation prevention bytes are removed.
struct BitReader {
    bytes: Vec<u8>,
    position: usize,
}

impl BitReader {
    fn bit(&mut self) -> Result<u8> {
        let byte = self.bytes.get(self.position / 8).ok_or("truncated H.264 SPS")?;
        let bit = (byte >> (7 - self.position % 8)) & 1;
        self.position += 1;
        Ok(bit)
    }

    fn exp_golomb(&mut self) -> Result<u32> {
        let mut leading_zeros = 0;
        while self.bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err("invalid H.264 SPS Exp-Golomb code".into());
            }
        }
        let mut suffix = 0u64;
        for _ in 0..leading_zeros {
            suffix = (suffix << 1) | u64::from(self.bit()?);
        }
        Ok(u32::try_from((1u64 << leading_zeros) - 1 + suffix)?)
    }
}

/// Returns chroma_format_idc, bit_depth_luma_minus8 and bit_depth_chroma_minus8 of a High profile SPS.
fn high_profile_format(sps: &[u8]) -> Result<[u8; 3]> {
    let mut bytes = Vec::with_capacity(sps.len());
    let mut zeros = 0;
    for &byte in &sps[1..] {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        bytes.push(byte);
    }
    // Skip profile_idc, the constraint flags and level_idc.
    let mut reader = BitReader { bytes, position: 24 };
    reader.exp_golomb()?; // seq_parameter_set_id
    let chroma_format = reader.exp_golomb()?;
    if chroma_format == 3 {
        reader.bit()?; // separate_colour_plane_flag
    }
    let luma_depth = reader.exp_golomb()?;
    let chroma_depth = reader.exp_golomb()?;
    if chroma_format > 3 || luma_depth > 7 || chroma_depth > 7 {
        return Err("unsupported H.264 SPS chroma format or bit depth".into());
    }
    Ok([chroma_format as u8, luma_depth as u8, chroma_depth as u8])
}

fn mp4a(audio: &AudioTrack) -> Vec<u8> {
    let mut decoder = vec![0x40, 0x15, 0, 0, 0];
    decoder.extend(u32s(&[audio.bitrate, audio.bitrate]));
    decoder.extend(descriptor(5, &audio.config));
    let mut stream = vec![0, 2, 0];
    stream.extend(descriptor(4, &decoder));
    stream.extend(descriptor(6, &[2]));
    let mut entry = vec![0; 6];
    entry.extend(1u16.to_be_bytes());
    entry.extend([0; 8]);
    entry.extend(audio.channels.to_be_bytes());
    entry.extend(16u16.to_be_bytes());
    entry.extend([0; 4]);
    entry.extend((audio.sample_rate << 16).to_be_bytes());
    entry.extend(full(b"esds", 0, 0, &descriptor(3, &stream)));
    atom(b"mp4a", &entry)
}
fn sample_table(entry: Vec<u8>, packets: &[Sample], offsets: &[u64], video: bool) -> Result<Vec<u8>> {
    let count = u32::try_from(packets.len())?;
    let mut table = full(b"stsd", 0, 0, &[u32s(&[1]), entry].concat());
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for packet in packets {
        if let Some((count, _)) = runs.last_mut().filter(|(_, duration)| *duration == packet.duration) {
            *count += 1;
        } else {
            runs.push((1, packet.duration));
        }
    }
    let mut stts = u32s(&[runs.len() as u32]);
    for (count, duration) in runs {
        stts.extend(u32s(&[count, duration]));
    }
    table.extend(full(b"stts", 0, 0, &stts));
    if packets.iter().any(|packet| packet.pts != packet.dts as i64) {
        let mut ctts = u32s(&[count]);
        for packet in packets {
            ctts.extend(1u32.to_be_bytes());
            ctts.extend(i32::try_from(i128::from(packet.pts) - i128::from(packet.dts))?.to_be_bytes());
        }
        table.extend(full(b"ctts", 1, 0, &ctts));
    }
    table.extend(full(b"stsc", 0, 0, &u32s(&[1, 1, 1, 1])));
    let mut stsz = u32s(&[0, count]);
    for packet in packets {
        stsz.extend(u32::try_from(packet.data.len())?.to_be_bytes());
    }
    table.extend(full(b"stsz", 0, 0, &stsz));
    let mut co64 = u32s(&[count]);
    for offset in offsets {
        co64.extend(offset.to_be_bytes());
    }
    table.extend(full(b"co64", 0, 0, &co64));
    if video {
        let sync: Vec<u32> = packets
            .iter()
            .enumerate()
            .filter(|(_, packet)| packet.keyframe)
            .map(|(index, _)| index as u32 + 1)
            .collect();
        table.extend(full(b"stss", 0, 0, &[u32s(&[sync.len() as u32]), u32s(&sync)].concat()));
    }
    Ok(atom(b"stbl", &table))
}
fn track(
    spec: &MediaSpec,
    config: &H264Config,
    audio: &AudioTrack,
    packets: &[Sample],
    offsets: &[u64],
    video: bool,
) -> Result<Vec<u8>> {
    let timescale = if video { spec.fps as u32 } else { audio.sample_rate };
    let duration: u64 = packets.iter().map(|packet| u64::from(packet.duration)).sum();
    let mut expected = 0;
    for packet in packets {
        if packet.dts != expected || packet.duration == 0 || packet.data.is_empty() {
            return Err("invalid MP4 sample timing".into());
        }
        expected += u64::from(packet.duration);
    }
    let movie_duration = if video {
        spec.frames as u64 * 1_000_000 / spec.fps as u64
    } else {
        audio.samples * 1_000_000 / u64::from(audio.sample_rate)
    };
    let track_id = if video { 1 } else { 2 };
    let mut track_header = vec![0; 16];
    track_header.extend(u32s(&[track_id, 0]));
    track_header.extend(movie_duration.to_be_bytes());
    track_header.extend([0; 8]);
    track_header.extend([0, 0, 0, 0]);
    track_header.extend(if video { [0, 0, 0, 0] } else { [1, 0, 0, 0] });
    track_header.extend(matrix());
    track_header.extend(u32s(&[
        if video { (spec.width as u32) << 16 } else { 0 },
        if video { (spec.height as u32) << 16 } else { 0 },
    ]));
    let mut trak = full(b"tkhd", 1, 3, &track_header);
    if !video {
        let mut edit = u32s(&[1]);
        edit.extend(movie_duration.to_be_bytes());
        edit.extend(i64::from(audio.priming).to_be_bytes());
        edit.extend([0, 1, 0, 0]);
        trak.extend(atom(b"edts", &full(b"elst", 1, 0, &edit)));
    }
    let mut mdhd = vec![0; 16];
    mdhd.extend(timescale.to_be_bytes());
    mdhd.extend(duration.to_be_bytes());
    mdhd.extend([0x55, 0xc4, 0, 0]);
    let mut mdia = full(b"mdhd", 1, 0, &mdhd);
    let mut handler = u32s(&[0]);
    handler.extend(if video { b"vide" } else { b"soun" });
    handler.extend([0; 12]);
    handler.extend(b"mmh3\0");
    mdia.extend(full(b"hdlr", 0, 0, &handler));
    let mut minf = if video {
        full(b"vmhd", 0, 1, &[0; 8])
    } else {
        full(b"smhd", 0, 0, &[0; 4])
    };
    minf.extend(atom(
        b"dinf",
        &full(b"dref", 0, 0, &[u32s(&[1]), full(b"url ", 0, 1, &[])].concat()),
    ));
    minf.extend(sample_table(
        if video { avc1(spec, config)? } else { mp4a(audio) },
        packets,
        offsets,
        video,
    )?);
    mdia.extend(atom(b"minf", &minf));
    trak.extend(atom(b"mdia", &mdia));
    Ok(atom(b"trak", &trak))
}
fn movie(
    spec: &MediaSpec,
    config: &H264Config,
    video: &[Sample],
    audio: &AudioTrack,
    video_offsets: &[u64],
    audio_offsets: &[u64],
) -> Result<Vec<u8>> {
    let mut movie_header = vec![0; 16];
    movie_header.extend(1_000_000u32.to_be_bytes());
    movie_header.extend((spec.frames as u64 * 1_000_000 / spec.fps as u64).to_be_bytes());
    movie_header.extend(0x10000u32.to_be_bytes());
    movie_header.extend([1, 0, 0, 0]);
    movie_header.extend([0; 8]);
    movie_header.extend(matrix());
    movie_header.extend([0; 24]);
    movie_header.extend(3u32.to_be_bytes());
    let mut moov = full(b"mvhd", 1, 0, &movie_header);
    moov.extend(track(spec, config, audio, video, video_offsets, true)?);
    moov.extend(track(spec, config, audio, &audio.packets, audio_offsets, false)?);
    Ok(atom(b"moov", &moov))
}
pub(super) fn write(
    writer: &mut impl Write,
    spec: &MediaSpec,
    config: &H264Config,
    video: &[Sample],
    audio: &AudioTrack,
) -> Result<()> {
    let ftyp = atom(b"ftyp", b"isom\0\0\x02\0isomiso2avc1mp41");
    let header = movie(
        spec,
        config,
        video,
        audio,
        &vec![0; video.len()],
        &vec![0; audio.packets.len()],
    )?;
    let mut offset = (ftyp.len() + header.len() + 16) as u64;
    // Interleave chunks in decoding order to support progressive playback.
    let mut order: Vec<(bool, usize)> = (0..video.len())
        .map(|index| (true, index))
        .chain((0..audio.packets.len()).map(|index| (false, index)))
        .collect();
    order.sort_by_key(|&(is_video, index)| {
        if is_video {
            u128::from(video[index].dts) * u128::from(audio.sample_rate)
        } else {
            u128::from(audio.packets[index].dts) * spec.fps as u128
        }
    });
    let mut video_offsets = vec![0; video.len()];
    let mut audio_offsets = vec![0; audio.packets.len()];
    for &(is_video, index) in &order {
        let packet = if is_video {
            video_offsets[index] = offset;
            &video[index]
        } else {
            audio_offsets[index] = offset;
            &audio.packets[index]
        };
        offset = offset.checked_add(packet.data.len() as u64).ok_or("MP4 size overflow")?;
    }
    let header = movie(spec, config, video, audio, &video_offsets, &audio_offsets)?;
    writer.write_all(&ftyp)?;
    writer.write_all(&header)?;
    writer.write_all(&1u32.to_be_bytes())?;
    writer.write_all(b"mdat")?;
    writer.write_all(&(offset - (ftyp.len() + header.len()) as u64).to_be_bytes())?;
    for (is_video, index) in order {
        writer.write_all(if is_video { &video[index].data } else { &audio.packets[index].data })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avcc(sps: &[u8]) -> Vec<u8> {
        let spec = MediaSpec {
            width: 64,
            height: 64,
            frames: 1,
            fps: 24,
            sample_rate: 32_000,
            channels: 2,
        };
        let config = H264Config {
            sps: sps.to_vec(),
            pps: vec![0x68, 0xee, 0x3c, 0x80],
        };
        let entry = avc1(&spec, &config).unwrap();
        let start = entry.windows(4).position(|kind| kind == b"avcC").unwrap() - 4;
        let size = u32::from_be_bytes(entry[start..start + 4].try_into().unwrap()) as usize;
        entry[start + 8..start + size].to_vec()
    }

    #[test]
    fn high_profile_avcc_repeats_chroma_format_and_bit_depths() {
        // High profile, sps_id 0, 4:2:0, 8-bit luma and chroma.
        let record = avcc(&[0x67, 0x64, 0x00, 0x1f, 0xac, 0xd9, 0x40]);
        assert_eq!(record[..6], [1, 0x64, 0x00, 0x1f, 0xff, 0xe1]);
        assert_eq!(record[record.len() - 4..], [0xfd, 0xf8, 0xf8, 0]);
        // High 4:2:2, 10-bit luma and chroma.
        let record = avcc(&[0x67, 0x7a, 0x00, 0x1f, 0xb6, 0xc0]);
        assert_eq!(record[record.len() - 4..], [0xfe, 0xfa, 0xfa, 0]);
        // High 4:4:4 with separate_colour_plane_flag, 10-bit luma, 8-bit chroma.
        let record = avcc(&[0x67, 0x90, 0x00, 0x1f, 0x92, 0xe0]);
        assert_eq!(record[record.len() - 4..], [0xff, 0xfa, 0xf8, 0]);
        // Main profile has no extension.
        let record = avcc(&[0x67, 0x4d, 0x00, 0x1f, 0x9a]);
        assert_eq!(record.len(), 6 + 2 + 5 + 1 + 2 + 4);
    }
}
