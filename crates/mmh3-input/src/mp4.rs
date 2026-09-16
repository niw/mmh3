//! MP4 demuxing of a file's H.264 video track.
//!
//! Only what a reference clip needs: the track's size, its parameter sets and its samples in
//! decoding order, handed out as Annex B access units for a hardware decoder. Audio itself goes
//! through Symphonia, which reads MP4, but the edit list that hides a codec's priming comes from
//! here, since Symphonia reports the whole media instead.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// Start code of an Annex B NAL unit.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn at<const N: usize>(payload: &[u8], offset: usize) -> io::Result<[u8; N]> {
    payload
        .get(offset..offset + N)
        .ok_or_else(|| invalid("a box ends inside a field"))?
        .try_into()
        .map_err(|_| invalid("a box ends inside a field"))
}

fn u16_at(payload: &[u8], offset: usize) -> io::Result<u16> {
    Ok(u16::from_be_bytes(at(payload, offset)?))
}

fn u32_at(payload: &[u8], offset: usize) -> io::Result<u32> {
    Ok(u32::from_be_bytes(at(payload, offset)?))
}

fn u64_at(payload: &[u8], offset: usize) -> io::Result<u64> {
    Ok(u64::from_be_bytes(at(payload, offset)?))
}

/// The boxes of a container's payload, as (kind, payload) pairs in their order.
fn children(payload: &[u8]) -> impl Iterator<Item = (&[u8; 4], &[u8])> {
    let mut offset = 0;
    std::iter::from_fn(move || {
        if offset + 8 > payload.len() {
            return None;
        }
        let size = u32::from_be_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        let kind: &[u8; 4] = payload[offset + 4..offset + 8].try_into().unwrap();
        // Size 1 means a 64-bit size follows the header, and 0 the rest of the container.
        let (header, size) = match size {
            0 => (8, payload.len() - offset),
            1 => (16, u64_at(payload, offset + 8).ok()? as usize),
            size => (8, size),
        };
        if size < header || offset + size > payload.len() {
            return None;
        }
        let body = &payload[offset + header..offset + size];
        offset += size;
        Some((kind, body))
    })
}

fn child<'a>(payload: &'a [u8], kind: &[u8; 4]) -> io::Result<&'a [u8]> {
    children(payload)
        .find(|(found, _)| *found == kind)
        .map(|(_, body)| body)
        .ok_or_else(|| invalid(format!("no {} box", String::from_utf8_lossy(kind))))
}

/// One coded frame of the video track.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    /// Where the sample starts in the file.
    pub offset: u64,
    pub size: usize,
    /// Ticks of the track's timescale the frame lasts.
    pub duration: u32,
    /// Whether the sample starts a group of pictures.
    pub keyframe: bool,
}

/// An H.264 video track, with its samples in decoding order.
#[derive(Clone, Debug)]
pub struct VideoTrack {
    pub width: usize,
    pub height: usize,
    /// Ticks per second of the sample durations.
    pub timescale: u32,
    /// The parameter sets of the avcC record with Annex B start codes, which a decoder takes
    /// before the first access unit.
    pub parameter_sets: Vec<u8>,
    pub samples: Vec<Sample>,
}

impl VideoTrack {
    /// Frames per second from the mean sample duration.
    pub fn fps(&self) -> f64 {
        let ticks: u64 = self
            .samples
            .iter()
            .map(|sample| u64::from(sample.duration))
            .sum();
        if ticks == 0 || self.samples.is_empty() {
            return 0.0;
        }
        self.samples.len() as f64 * f64::from(self.timescale) / ticks as f64
    }
}

/// A file open for reading its video track's samples.
pub struct Mp4File {
    file: File,
    track: VideoTrack,
    /// Bytes of the length that precedes every NAL unit of a sample.
    nal_length_size: usize,
}

impl Mp4File {
    /// Reads the movie header and the video track's sample table.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let movie = read_top_level(&mut file, b"moov")?;
        let (track, nal_length_size) = video_track(&movie)?;
        Ok(Mp4File {
            file,
            track,
            nal_length_size,
        })
    }

    pub fn track(&self) -> &VideoTrack {
        &self.track
    }

    /// The access unit of sample `index` in Annex B form, its NAL units prefixed with start codes
    /// instead of lengths.
    pub fn access_unit(&mut self, index: usize) -> io::Result<Vec<u8>> {
        let sample = *self
            .track
            .samples
            .get(index)
            .ok_or_else(|| invalid(format!("no sample {index}")))?;
        let mut data = vec![0u8; sample.size];
        self.file.seek(SeekFrom::Start(sample.offset))?;
        self.file.read_exact(&mut data)?;
        let mut unit = Vec::with_capacity(data.len() + 4 * 8);
        let mut offset = 0;
        while offset + self.nal_length_size <= data.len() {
            let mut length = 0usize;
            for byte in &data[offset..offset + self.nal_length_size] {
                length = (length << 8) | usize::from(*byte);
            }
            offset += self.nal_length_size;
            if length == 0 || offset + length > data.len() {
                return Err(invalid("a NAL unit runs past its sample"));
            }
            unit.extend_from_slice(&START_CODE);
            unit.extend_from_slice(&data[offset..offset + length]);
            offset += length;
        }
        if unit.is_empty() {
            return Err(invalid("a sample holds no NAL unit"));
        }
        Ok(unit)
    }
}

/// Reads the whole payload of a top-level box, which the movie header fits in.
fn read_top_level(file: &mut File, kind: &[u8; 4]) -> io::Result<Vec<u8>> {
    let end = file.seek(SeekFrom::End(0))?;
    let mut offset = 0;
    while offset + 8 <= end {
        file.seek(SeekFrom::Start(offset))?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;
        let size = u32::from_be_bytes(header[..4].try_into().unwrap()) as u64;
        let found: &[u8; 4] = header[4..].try_into().unwrap();
        let (header_size, size) = match size {
            0 => (8, end - offset),
            1 => {
                let mut large = [0u8; 8];
                file.read_exact(&mut large)?;
                (16, u64::from_be_bytes(large))
            }
            size => (8, size),
        };
        if size < header_size {
            return Err(invalid("a box is shorter than its header"));
        }
        if found == kind {
            let mut payload = vec![0u8; usize::try_from(size - header_size).map_err(invalid_size)?];
            file.seek(SeekFrom::Start(offset + header_size))?;
            file.read_exact(&mut payload)?;
            return Ok(payload);
        }
        offset += size;
    }
    Err(invalid(format!("no {} box", String::from_utf8_lossy(kind))))
}

fn invalid_size(error: std::num::TryFromIntError) -> io::Error {
    invalid(format!("a box is too large for this machine: {error}"))
}

/// The video track of a movie header, with the bytes of its NAL lengths.
fn video_track(movie: &[u8]) -> io::Result<(VideoTrack, usize)> {
    for (kind, track) in children(movie) {
        if kind != b"trak" {
            continue;
        }
        let media = child(track, b"mdia")?;
        let handler = child(media, b"hdlr")?;
        if at::<4>(handler, 8)? != *b"vide" {
            continue;
        }
        let header = child(media, b"mdhd")?;
        // The full box's version decides the width of its times.
        let timescale = match header[0] {
            0 => u32_at(header, 12)?,
            _ => u32_at(header, 20)?,
        };
        let table = child(child(media, b"minf")?, b"stbl")?;
        let (width, height, parameter_sets, nal_length_size) =
            avc_configuration(child(table, b"stsd")?)?;
        return Ok((
            VideoTrack {
                width,
                height,
                timescale,
                parameter_sets,
                samples: samples(table)?,
            },
            nal_length_size,
        ));
    }
    Err(invalid("the file has no H.264 video track"))
}

/// The size, parameter sets and NAL length of the avc1 entry of a sample description.
fn avc_configuration(description: &[u8]) -> io::Result<(usize, usize, Vec<u8>, usize)> {
    // A full box: version and flags, then the entry count.
    let entries = &description[8..];
    let (kind, entry) = children(entries)
        .find(|(kind, _)| *kind == b"avc1" || *kind == b"avc3")
        .ok_or_else(|| invalid("the video track is not H.264"))?;
    let width = usize::from(u16_at(entry, 24)?);
    let height = usize::from(u16_at(entry, 26)?);
    // A visual sample entry holds 78 bytes of fields after the box header, then its own boxes.
    let fields = entry
        .get(78..)
        .ok_or_else(|| invalid("a visual sample entry ends inside its fields"))?;
    let record = child(fields, b"avcC").map_err(|_| {
        invalid(format!(
            "the {} entry has no avcC record",
            String::from_utf8_lossy(kind)
        ))
    })?;
    let nal_length_size = usize::from((at::<1>(record, 4)?[0] & 0b11) + 1);
    let mut parameter_sets = Vec::new();
    let mut offset = 5;
    for set in 0..2 {
        // The sequence sets come first, five bits of their count in a reserved byte, then the
        // picture sets with a whole byte.
        let count =
            usize::from(at::<1>(record, offset)?[0] & if set == 0 { 0b1_1111 } else { 0xff });
        offset += 1;
        for _ in 0..count {
            let length = usize::from(u16_at(record, offset)?);
            offset += 2;
            let data = record
                .get(offset..offset + length)
                .ok_or_else(|| invalid("a parameter set runs past the avcC record"))?;
            parameter_sets.extend_from_slice(&START_CODE);
            parameter_sets.extend_from_slice(data);
            offset += length;
        }
    }
    if parameter_sets.is_empty() {
        return Err(invalid("the avcC record holds no parameter set"));
    }
    Ok((width, height, parameter_sets, nal_length_size))
}

/// The samples of a sample table in decoding order, walking the chunks they sit in.
fn samples(table: &[u8]) -> io::Result<Vec<Sample>> {
    let sizes = sample_sizes(child(table, b"stsz")?)?;
    let durations = sample_durations(child(table, b"stts")?, sizes.len())?;
    let offsets = chunk_offsets(table)?;
    let runs = sample_to_chunk(child(table, b"stsc")?)?;
    let sync = sync_samples(table)?;

    let mut samples = Vec::with_capacity(sizes.len());
    let mut run = 0;
    for (chunk, &offset) in offsets.iter().enumerate() {
        while run + 1 < runs.len() && runs[run + 1].0 <= chunk {
            run += 1;
        }
        let per_chunk = runs
            .get(run)
            .ok_or_else(|| invalid("the sample to chunk table is empty"))?
            .1;
        let mut position = offset;
        for _ in 0..per_chunk {
            let index = samples.len();
            if index >= sizes.len() {
                break;
            }
            samples.push(Sample {
                offset: position,
                size: sizes[index],
                duration: durations[index],
                keyframe: sync.as_ref().is_none_or(|sync| sync.contains(&index)),
            });
            position += sizes[index] as u64;
        }
    }
    if samples.len() != sizes.len() {
        return Err(invalid(format!(
            "the chunks hold {} of the {} samples",
            samples.len(),
            sizes.len()
        )));
    }
    Ok(samples)
}

fn sample_sizes(payload: &[u8]) -> io::Result<Vec<usize>> {
    let uniform = u32_at(payload, 4)? as usize;
    let count = u32_at(payload, 8)? as usize;
    if uniform > 0 {
        return Ok(vec![uniform; count]);
    }
    (0..count)
        .map(|index| Ok(u32_at(payload, 12 + index * 4)? as usize))
        .collect()
}

fn sample_durations(payload: &[u8], samples: usize) -> io::Result<Vec<u32>> {
    let entries = u32_at(payload, 4)? as usize;
    let mut durations = Vec::with_capacity(samples);
    for entry in 0..entries {
        let count = u32_at(payload, 8 + entry * 8)? as usize;
        let delta = u32_at(payload, 12 + entry * 8)?;
        durations.extend(std::iter::repeat_n(delta, count.min(samples)));
    }
    // A table may cover fewer samples than the file holds; the last duration carries on.
    let last = durations.last().copied().unwrap_or(0);
    durations.resize(samples, last);
    Ok(durations)
}

fn chunk_offsets(table: &[u8]) -> io::Result<Vec<u64>> {
    if let Ok(payload) = child(table, b"stco") {
        let count = u32_at(payload, 4)? as usize;
        return (0..count)
            .map(|index| Ok(u64::from(u32_at(payload, 8 + index * 4)?)))
            .collect();
    }
    let payload = child(table, b"co64")?;
    let count = u32_at(payload, 4)? as usize;
    (0..count)
        .map(|index| u64_at(payload, 8 + index * 8))
        .collect()
}

/// Every run of chunks as (first chunk, samples per chunk), the first chunk counted from zero.
fn sample_to_chunk(payload: &[u8]) -> io::Result<Vec<(usize, usize)>> {
    let entries = u32_at(payload, 4)? as usize;
    (0..entries)
        .map(|entry| {
            let first = u32_at(payload, 8 + entry * 12)? as usize;
            let per_chunk = u32_at(payload, 12 + entry * 12)? as usize;
            Ok((first.saturating_sub(1), per_chunk))
        })
        .collect()
}

/// The samples that start a group of pictures, or `None` when every sample does.
fn sync_samples(table: &[u8]) -> io::Result<Option<std::collections::HashSet<usize>>> {
    let Ok(payload) = child(table, b"stss") else {
        return Ok(None);
    };
    let count = u32_at(payload, 4)? as usize;
    (0..count)
        .map(|index| Ok(u32_at(payload, 8 + index * 4)? as usize - 1))
        .collect::<io::Result<std::collections::HashSet<usize>>>()
        .map(Some)
}

/// What a player keeps of an audio track: the samples it skips at the start, which hide the
/// codec's priming, and the samples it plays, both counted in the track's own sample rate. The
/// edit list carries them, and a file without one keeps everything.
pub fn audio_trim(path: &Path) -> io::Result<Option<(u64, u64)>> {
    let mut file = File::open(path)?;
    let movie = read_top_level(&mut file, b"moov")?;
    let movie_header = child(&movie, b"mvhd")?;
    let movie_timescale = u64::from(match movie_header[0] {
        0 => u32_at(movie_header, 12)?,
        _ => u32_at(movie_header, 20)?,
    });
    for (kind, track) in children(&movie) {
        if kind != b"trak" {
            continue;
        }
        let media = child(track, b"mdia")?;
        if at::<4>(child(media, b"hdlr")?, 8)? != *b"soun" {
            continue;
        }
        let Ok(list) = child(track, b"edts").and_then(|edits| child(edits, b"elst")) else {
            return Ok(None);
        };
        let header = child(media, b"mdhd")?;
        let media_timescale = u64::from(match header[0] {
            0 => u32_at(header, 12)?,
            _ => u32_at(header, 20)?,
        });
        if movie_timescale == 0 || media_timescale == 0 || u32_at(list, 4)? == 0 {
            return Ok(None);
        }
        // The first entry of the list says where the media starts and how long it plays, the
        // duration in the movie's timescale and the start in the media's own.
        let (duration, start) = if list[0] == 0 {
            (
                u64::from(u32_at(list, 8)?),
                i64::from(u32_at(list, 12)? as i32),
            )
        } else {
            (u64_at(list, 8)?, u64_at(list, 16)? as i64)
        };
        let skip = start.max(0) as u64;
        let length = duration * media_timescale / movie_timescale;
        return Ok(Some((skip, length)));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn atom(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut bytes = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        bytes.extend(kind);
        bytes.extend(payload);
        bytes
    }

    /// A full box of `values.len() / per_entry` entries, each `per_entry` words wide.
    fn table(kind: &[u8; 4], values: &[u32], per_entry: usize) -> Vec<u8> {
        let mut payload = vec![0u8; 4];
        payload.extend(((values.len() / per_entry) as u32).to_be_bytes());
        for value in values {
            payload.extend(value.to_be_bytes());
        }
        atom(kind, &payload)
    }

    /// A file with one H.264 track whose samples each sit in their own chunk.
    fn file(samples: &[Vec<u8>]) -> Vec<u8> {
        let sps = [0x67u8, 0x64, 0x00, 0x1f];
        let pps = [0x68u8, 0xee, 0x3c, 0x80];
        let mut record = vec![1, sps[1], sps[2], sps[3], 0xff, 0xe1];
        record.extend((sps.len() as u16).to_be_bytes());
        record.extend(sps);
        record.push(1);
        record.extend((pps.len() as u16).to_be_bytes());
        record.extend(pps);

        let mut entry = vec![0u8; 78];
        entry[24..26].copy_from_slice(&64u16.to_be_bytes());
        entry[26..28].copy_from_slice(&32u16.to_be_bytes());
        entry.extend(atom(b"avcC", &record));
        let mut description = vec![0u8; 4];
        description.extend(1u32.to_be_bytes());
        description.extend(atom(b"avc1", &entry));

        let mut media_header = vec![0u8; 12];
        media_header.extend(1000u32.to_be_bytes());
        media_header.extend(0u32.to_be_bytes());
        media_header.extend([0u8; 4]);

        // A handler box holds its type after the version, flags and a reserved word.
        let mut handler = vec![0u8; 12];
        handler[8..12].copy_from_slice(b"vide");

        let mut sizes = vec![0u8; 4];
        sizes.extend(0u32.to_be_bytes());
        sizes.extend((samples.len() as u32).to_be_bytes());
        for sample in samples {
            sizes.extend((sample.len() as u32).to_be_bytes());
        }

        // The chunk offsets are absolute, so the header is built once to measure it and again
        // with the offsets it puts the payload at.
        let movie = |offsets: &[u32]| {
            let mut stbl = atom(b"stsd", &description);
            stbl.extend(table(b"stts", &[samples.len() as u32, 40], 2));
            stbl.extend(table(b"stsc", &[1, 1, 1], 3));
            stbl.extend(atom(b"stsz", &sizes));
            stbl.extend(table(b"stss", &[1], 1));
            stbl.extend(table(b"stco", offsets, 1));
            let media = [
                atom(b"mdhd", &media_header),
                atom(b"hdlr", &handler),
                atom(b"minf", &atom(b"stbl", &stbl)),
            ]
            .concat();
            atom(b"moov", &atom(b"trak", &atom(b"mdia", &media)))
        };
        let ftyp = atom(b"ftyp", b"isom\0\0\x02\0isom");
        let zeros = vec![0u32; samples.len()];
        let base = (ftyp.len() + movie(&zeros).len() + 8) as u32;
        let mut position = base;
        let offsets: Vec<u32> = samples
            .iter()
            .map(|sample| {
                let offset = position;
                position += sample.len() as u32;
                offset
            })
            .collect();

        let mut bytes = ftyp;
        bytes.extend(movie(&offsets));
        assert_eq!(
            bytes.len() + 8,
            base as usize,
            "the payload starts at the offsets"
        );
        bytes.extend(atom(b"mdat", &samples.concat()));
        bytes
    }

    fn nal(length: usize, value: u8) -> Vec<u8> {
        let mut unit = (length as u32).to_be_bytes().to_vec();
        unit.extend(std::iter::repeat_n(value, length));
        unit
    }

    #[test]
    fn reads_the_sample_table_and_the_parameter_sets() {
        let samples = vec![
            [nal(6, 0xaa), nal(4, 0xbb)].concat(),
            nal(5, 0xcc),
            nal(7, 0xdd),
        ];
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("clip.mp4");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&file(&samples))
            .unwrap();

        let mut reader = Mp4File::open(&path).unwrap();
        let track = reader.track().clone();
        assert_eq!((track.width, track.height), (64, 32));
        assert_eq!(track.timescale, 1000);
        assert_eq!(track.samples.len(), 3);
        // 1000 ticks per second and 40 ticks per frame make 25 frames per second.
        assert!((track.fps() - 25.0).abs() < 1e-9);
        // The sync table names the first sample alone.
        assert_eq!(
            track
                .samples
                .iter()
                .map(|sample| sample.keyframe)
                .collect::<Vec<_>>(),
            vec![true, false, false]
        );
        assert_eq!(
            track
                .samples
                .iter()
                .map(|sample| sample.size)
                .collect::<Vec<_>>(),
            samples.iter().map(Vec::len).collect::<Vec<_>>()
        );
        // The parameter sets come out as Annex B, the sequence set first.
        assert_eq!(&track.parameter_sets[..5], &[0, 0, 0, 1, 0x67]);
        assert_eq!(&track.parameter_sets[8..13], &[0, 0, 0, 1, 0x68]);

        // The lengths of both NAL units of the first sample become start codes.
        let unit = reader.access_unit(0).unwrap();
        assert_eq!(unit.len(), 4 + 6 + 4 + 4);
        assert_eq!(&unit[..5], &[0, 0, 0, 1, 0xaa]);
        assert_eq!(&unit[10..15], &[0, 0, 0, 1, 0xbb]);
        assert_eq!(reader.access_unit(2).unwrap().len(), 4 + 7);
        assert!(reader.access_unit(3).is_err());
    }
}
