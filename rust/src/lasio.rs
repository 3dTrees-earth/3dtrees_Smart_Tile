//! LAS headers and VLRs as raw bytes, and a point format 0 writer laid out as laspy writes it.
//!
//! SmartTile tiles carry XYZ only (point format 0) and copy identity fields
//! and VLRs from the source header byte for byte, so outputs can be compared
//! with Python SmartTile's files directly.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub type Res<T> = Result<T, String>;

fn io<E: std::fmt::Display>(context: impl std::fmt::Display) -> impl FnOnce(E) -> String {
    move |error| format!("{context}: {error}")
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vlr {
    pub reserved: u16,
    pub user_id: [u8; 16],
    pub record_id: u16,
    pub description: [u8; 32],
    pub data: Vec<u8>,
}

impl Vlr {
    pub fn user(&self) -> String {
        let end = self.user_id.iter().position(|&b| b == 0).unwrap_or(16);
        String::from_utf8_lossy(&self.user_id[..end]).into_owned()
    }

    fn fixed<const N: usize>(text: &str) -> [u8; N] {
        let mut out = [0u8; N];
        out[..text.len().min(N)].copy_from_slice(&text.as_bytes()[..text.len().min(N)]);
        out
    }

    pub fn new(user_id: &str, record_id: u16, description: &str, data: Vec<u8>) -> Vlr {
        Vlr { reserved: 0, user_id: Self::fixed(user_id), record_id, description: Self::fixed(description), data }
    }
}

/// The source header fields SmartTile copies into its outputs.
#[derive(Clone, Debug)]
pub struct SourceHeader {
    pub version: (u8, u8),
    pub file_source_id: u16,
    pub global_encoding: u16,
    pub guid: [u8; 16],
    pub system_identifier: [u8; 32],
    pub generating_software: [u8; 32],
    pub creation_day: u16,
    pub creation_year: u16,
    pub point_format: u8, // without the compression bits
    pub record_length: u16,
    pub point_count: u64,
    pub scales: [f64; 3],
    pub offsets: [f64; 3],
    pub mins: [f64; 3],
    pub maxs: [f64; 3],
    pub vlrs: Vec<Vlr>,
    pub evlrs: Vec<Vlr>,
}

fn u16_at(b: &[u8], o: usize) -> u16 { u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) }
fn u32_at(b: &[u8], o: usize) -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn u64_at(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }
fn f64_at(b: &[u8], o: usize) -> f64 { f64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }

pub fn read_header(path: &Path) -> Res<SourceHeader> {
    let mut file = File::open(path).map_err(io(path.display()))?;
    let mut head = vec![0u8; 375];
    let read = file.read(&mut head).map_err(io(path.display()))?;
    if read < 227 || &head[..4] != b"LASF" {
        return Err(format!("{}: not a LAS/LAZ file", path.display()));
    }
    let version = (head[24], head[25]);
    let header_size = u16_at(&head, 94) as u64;
    let num_vlrs = u32_at(&head, 100);
    let mut point_count = u32_at(&head, 107) as u64;
    let (mut evlr_start, mut num_evlrs) = (0u64, 0u32);
    if version >= (1, 4) && read >= 375 {
        evlr_start = u64_at(&head, 235);
        num_evlrs = u32_at(&head, 243);
        let count = u64_at(&head, 247);
        if count != 0 || point_count == 0 {
            point_count = count;
        }
    }
    let triple = |o: usize| [f64_at(&head, o), f64_at(&head, o + 8), f64_at(&head, o + 16)];
    let mut vlrs = Vec::new();
    file.seek(SeekFrom::Start(header_size)).map_err(io(path.display()))?;
    for _ in 0..num_vlrs {
        let mut fixed = [0u8; 54];
        file.read_exact(&mut fixed).map_err(io(path.display()))?;
        let mut data = vec![0u8; u16_at(&fixed, 20) as usize];
        file.read_exact(&mut data).map_err(io(path.display()))?;
        vlrs.push(Vlr { reserved: u16_at(&fixed, 0), user_id: fixed[2..18].try_into().unwrap(),
                        record_id: u16_at(&fixed, 18), description: fixed[22..54].try_into().unwrap(), data });
    }
    let mut evlrs = Vec::new();
    if num_evlrs > 0 && evlr_start > 0 {
        file.seek(SeekFrom::Start(evlr_start)).map_err(io(path.display()))?;
        for _ in 0..num_evlrs {
            let mut fixed = [0u8; 60];
            file.read_exact(&mut fixed).map_err(io(path.display()))?;
            let mut data = vec![0u8; u64_at(&fixed, 20) as usize];
            file.read_exact(&mut data).map_err(io(path.display()))?;
            evlrs.push(Vlr { reserved: u16_at(&fixed, 0), user_id: fixed[2..18].try_into().unwrap(),
                             record_id: u16_at(&fixed, 18), description: fixed[28..60].try_into().unwrap(), data });
        }
    }
    Ok(SourceHeader {
        version,
        file_source_id: u16_at(&head, 4),
        global_encoding: u16_at(&head, 6),
        guid: head[8..24].try_into().unwrap(),
        system_identifier: head[26..58].try_into().unwrap(),
        generating_software: head[58..90].try_into().unwrap(),
        creation_day: u16_at(&head, 90),
        creation_year: u16_at(&head, 92),
        point_format: head[104] & 0x3f,
        record_length: u16_at(&head, 105),
        point_count,
        scales: triple(131),
        offsets: triple(155),
        maxs: [f64_at(&head, 179), f64_at(&head, 195), f64_at(&head, 211)],
        mins: [f64_at(&head, 187), f64_at(&head, 203), f64_at(&head, 219)],
        vlrs,
        evlrs,
    })
}

/// Output header of a tile: `make_center_of_mass_header(dimension_reduction=True)`.
#[derive(Clone, Debug)]
pub struct TileHeader {
    pub source: SourceHeader,
    pub global_encoding: u16,
    /// ExtraBytes after the point format fields (instance tiles, predictions).
    pub extra: Vec<crate::points::OutputExtra>,
    /// Output point format (0 for tiles; the target's format for dense merge tiles).
    pub point_format: u8,
    pub scales: [f64; 3],
    pub offsets: [f64; 3],
    pub vlrs: Vec<Vlr>,
    pub evlrs: Vec<Vlr>,
}

impl TileHeader {
    /// Point format 0 at the source version; COPC records and extra-bytes
    /// descriptions (no extra bytes survive) are dropped.
    pub fn for_tile(source: &SourceHeader, scales: [f64; 3], offsets: [f64; 3]) -> Res<TileHeader> {
        let keep = |v: &&Vlr| v.user() != "copc";
        let mut vlrs = source.vlrs.iter().filter(keep)
            .filter(|v| !(v.user() == "LASF_Spec" && v.record_id == 4))
            .filter(|v| v.user() != "laszip encoded").cloned().collect();
        let mut evlrs = source.evlrs.iter().filter(keep).cloned().collect();
        let mut global_encoding = source.global_encoding;
        // LAS 1.4: exactly one standardized WKT record (`standardize_projection_records`).
        crate::crs::standardize(source.version, &mut global_encoding, source, &mut vlrs, &mut evlrs);
        Ok(TileHeader { source: source.clone(), global_encoding, extra: Vec::new(), point_format: 0, scales, offsets, vlrs, evlrs })
    }
}

pub const RECORD_LEN: usize = 20; // point format 0

impl TileHeader {
    /// laspy `_sync_extra_bytes_vlr`: the ExtraBytes VLR moves to the end of the VLR list.
    pub fn sync_extra_bytes(&mut self) {
        self.vlrs.retain(|v| !(v.user() == "LASF_Spec" && v.record_id == 4));
        if !self.extra.is_empty() {
            self.vlrs.push(crate::points::extra_bytes_vlr(&self.extra));
        }
    }

    /// The header as laspy read it (`LasData.write` of a read cloud): every VLR kept in place.
    pub fn verbatim(source: &SourceHeader) -> Res<TileHeader> {
        let vlrs: Vec<Vlr> = source.vlrs.iter().filter(|v| v.user() != "laszip encoded").cloned().collect();
        let layout = crate::points::PointLayout::of(source)?;
        Ok(TileHeader { source: source.clone(), global_encoding: source.global_encoding,
                        extra: layout.extra.iter().map(|(e, _)| crate::points::OutputExtra::stored(e)).collect(),
                        point_format: source.point_format, scales: source.scales, offsets: source.offsets, vlrs, evlrs: source.evlrs.clone() })
    }

    /// `copy_single_source_header(preserve_extra_dimensions=True)`: the source's
    /// point format, VLRs (ExtraBytes in place, min/max continuing) and EVLRs
    /// without COPC records, CRS records standardized for LAS 1.4.
    pub fn copy_of(source: &SourceHeader) -> Res<TileHeader> {
        let keep = |v: &&Vlr| v.user() != "copc" && v.user() != "laszip encoded";
        let mut vlrs: Vec<Vlr> = source.vlrs.iter().filter(keep).cloned().collect();
        let mut evlrs: Vec<Vlr> = source.evlrs.iter().filter(|v| v.user() != "copc").cloned().collect();
        let mut global_encoding = source.global_encoding;
        crate::crs::standardize(source.version, &mut global_encoding, source, &mut vlrs, &mut evlrs);
        let layout = crate::points::PointLayout::of(source)?;
        Ok(TileHeader { source: source.clone(), global_encoding,
                        extra: layout.extra.iter().map(|(e, _)| crate::points::OutputExtra::stored(e)).collect(),
                        point_format: source.point_format, scales: source.scales, offsets: source.offsets, vlrs, evlrs })
    }
}

impl TileHeader {
    pub fn record_length(&self) -> usize {
        crate::points::base_record_length(self.point_format) + self.extra.iter().map(|e| e.source.size()).sum::<usize>()
    }
}
const LASPY_LASZIP_DESCRIPTION: &str = "http://laszip.org";

fn header_size(version: (u8, u8)) -> u16 {
    match version { (1, 4) => 375, (1, 3) => 235, _ => 227 }
}

fn vlr_bytes(vlr: &Vlr) -> Vec<u8> {
    let mut out = Vec::with_capacity(54 + vlr.data.len());
    out.extend_from_slice(&vlr.reserved.to_le_bytes());
    out.extend_from_slice(&vlr.user_id);
    out.extend_from_slice(&vlr.record_id.to_le_bytes());
    out.extend_from_slice(&(vlr.data.len() as u16).to_le_bytes());
    out.extend_from_slice(&vlr.description);
    out.extend_from_slice(&vlr.data);
    out
}

fn evlr_bytes(vlr: &Vlr) -> Vec<u8> {
    let mut out = Vec::with_capacity(60 + vlr.data.len());
    out.extend_from_slice(&vlr.reserved.to_le_bytes());
    out.extend_from_slice(&vlr.user_id);
    out.extend_from_slice(&vlr.record_id.to_le_bytes());
    out.extend_from_slice(&(vlr.data.len() as u64).to_le_bytes());
    out.extend_from_slice(&vlr.description);
    out.extend_from_slice(&vlr.data);
    out
}

struct Stats { count: u64, mins: [i32; 3], maxs: [i32; 3], returns: [u64; 15] }

fn header_bytes(header: &TileHeader, vlrs: &[Vlr], stats: &Stats, evlr_start: u64, num_evlrs: u32,
                compressed: bool) -> Vec<u8> {
    let s = &header.source;
    let version = s.version;
    let size = header_size(version);
    let mut b = Vec::with_capacity(size as usize);
    b.extend_from_slice(b"LASF");
    b.extend_from_slice(&s.file_source_id.to_le_bytes());
    b.extend_from_slice(&header.global_encoding.to_le_bytes());
    b.extend_from_slice(&s.guid);
    b.push(version.0);
    b.push(version.1);
    b.extend_from_slice(&s.system_identifier);
    b.extend_from_slice(&s.generating_software);
    b.extend_from_slice(&s.creation_day.to_le_bytes());
    b.extend_from_slice(&s.creation_year.to_le_bytes());
    b.extend_from_slice(&size.to_le_bytes());
    let offset_to_points = size as u32 + vlrs.iter().map(|v| 54 + v.data.len() as u32).sum::<u32>();
    b.extend_from_slice(&offset_to_points.to_le_bytes());
    b.extend_from_slice(&(vlrs.len() as u32).to_le_bytes());
    b.push(header.point_format | if compressed { 0x80 } else { 0 });
    b.extend_from_slice(&(header.record_length() as u16).to_le_bytes());
    // laspy writes no legacy point counts in LAS 1.4 files.
    if version < (1, 4) {
        b.extend_from_slice(&(stats.count as u32).to_le_bytes());
        for r in &stats.returns[..5] { b.extend_from_slice(&(*r as u32).to_le_bytes()); }
    } else {
        b.extend_from_slice(&[0u8; 24]);
    }
    for v in header.scales { b.extend_from_slice(&v.to_le_bytes()); }
    for v in header.offsets { b.extend_from_slice(&v.to_le_bytes()); }
    // laspy: mins/maxs of the scaled coordinates (raw * scale + offset).
    let world = |raw: i32, axis: usize| header.scales[axis] * f64::from(raw) + header.offsets[axis];
    for axis in 0..3 {
        let (lo, hi) = if stats.count == 0 { (0.0, 0.0) } else { (world(stats.mins[axis], axis), world(stats.maxs[axis], axis)) };
        b.extend_from_slice(&hi.to_le_bytes());
        b.extend_from_slice(&lo.to_le_bytes());
    }
    if version >= (1, 3) {
        b.extend_from_slice(&0u64.to_le_bytes());
    }
    if version >= (1, 4) {
        b.extend_from_slice(&evlr_start.to_le_bytes());
        b.extend_from_slice(&num_evlrs.to_le_bytes());
        b.extend_from_slice(&stats.count.to_le_bytes());
        for r in &stats.returns { b.extend_from_slice(&r.to_le_bytes()); }
    }
    debug_assert_eq!(b.len(), size as usize);
    b
}

/// Write XYZ points (point format 0) with the tile header; LAZ when `compress`.
/// `batches` streams the points, so memory stays bounded by one batch.
pub fn write_points<I>(path: &Path, header: &TileHeader, batches: I, compress: bool,
                       pool: Option<&rayon::ThreadPool>) -> Res<u64>
where I: IntoIterator<Item = Res<Vec<[i32; 3]>>> {
    let records = batches.into_iter().map(|batch| batch.map(|points| {
        let mut records = vec![0u8; points.len() * RECORD_LEN];
        for (record, p) in records.chunks_exact_mut(RECORD_LEN).zip(&points) {
            for axis in 0..3 {
                record[axis * 4..axis * 4 + 4].copy_from_slice(&p[axis].to_le_bytes());
            }
        }
        records
    }));
    write_records(path, header, records, compress, pool)
}

/// Write raw point records (format 0 plus the header's ExtraBytes); one
/// batch is one laspy `write_points` call (ExtraBytes min/max follow it).
pub fn write_records<I>(path: &Path, header: &TileHeader, batches: I, compress: bool,
                        pool: Option<&rayon::ThreadPool>) -> Res<u64>
where I: IntoIterator<Item = Res<Vec<u8>>> {
    let record_len = header.record_length();
    let mut stats = Stats { count: 0, mins: [i32::MAX; 3], maxs: [i32::MIN; 3], returns: [0; 15] };
    let mut extra = header.extra.clone();
    let return_bits = if header.point_format < 6 { 0x07 } else { 0x0f };
    let mut vlrs = header.vlrs.clone();
    let extra_size = (record_len - crate::points::base_record_length(header.point_format)) as u16;
    // laspy keeps a copied ExtraBytes VLR in place; added dimensions append one.
    let eb_position = vlrs.iter().position(|v| v.user() == "LASF_Spec" && v.record_id == 4);
    match (eb_position, extra.is_empty()) {
        (Some(at), false) => vlrs[at] = crate::points::extra_bytes_vlr(&extra),
        (None, false) => vlrs.push(crate::points::extra_bytes_vlr(&extra)),
        (Some(at), true) => { vlrs.remove(at); }
        (None, true) => {}
    }
    let laz_vlr = if compress {
        let items = laz::LazItemRecordBuilder::default_for_point_format_id(header.point_format, extra_size).map_err(|e| e.to_string())?;
        let laz_vlr = laz::LazVlr::from_laz_items(items);
        let mut data = Vec::new();
        laz_vlr.write_to(&mut data).map_err(|e| e.to_string())?;
        vlrs.push(Vlr::new("laszip encoded", 22204, LASPY_LASZIP_DESCRIPTION, data));
        Some(laz_vlr)
    } else {
        None
    };
    let file = File::create(path).map_err(io(path.display()))?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    out.write_all(&header_bytes(header, &vlrs, &stats, 0, 0, compress)).map_err(io(path.display()))?;
    for vlr in &vlrs {
        out.write_all(&vlr_bytes(vlr)).map_err(io(path.display()))?;
    }
    enum Sink<W: Write + Seek + Send + Sync> { Plain(W), Laz(laz::ParLasZipCompressor<W>) }
    let mut sink = match laz_vlr {
        Some(laz_vlr) => Sink::Laz(laz::ParLasZipCompressor::new(out, laz_vlr).map_err(|e| e.to_string())?),
        None => Sink::Plain(out),
    };
    for batch in batches {
        let records = batch?;
        if records.is_empty() {
            continue;
        }
        for record in records.chunks_exact(record_len) {
            let return_number = (record[14] & return_bits) as usize;
            if return_number > 0 {
                stats.returns[return_number - 1] += 1;
            }
            for axis in 0..3 {
                let v = i32::from_le_bytes(record[axis * 4..axis * 4 + 4].try_into().unwrap());
                stats.mins[axis] = stats.mins[axis].min(v);
                stats.maxs[axis] = stats.maxs[axis].max(v);
            }
        }
        let mut at = crate::points::base_record_length(header.point_format);
        for e in &mut extra {
            let size = e.source.size();
            e.grow(&records[at..at + size]);
            at += size;
        }
        stats.count += (records.len() / record_len) as u64;
        match &mut sink {
            Sink::Plain(out) => out.write_all(&records).map_err(io(path.display()))?,
            // LAZ chunks compress on the task's worker pool, not the assembly pool.
            Sink::Laz(compressor) => match pool {
                Some(pool) => pool.install(|| compressor.compress_many(&records)),
                None => compressor.compress_many(&records),
            }.map_err(io(path.display()))?,
        }
    }
    let mut out = match sink {
        Sink::Plain(out) => out,
        Sink::Laz(mut compressor) => {
            compressor.done().map_err(|e| e.to_string())?;
            compressor.into_inner()
        }
    };
    let mut evlr_start = 0;
    if !header.evlrs.is_empty() {
        evlr_start = out.stream_position().map_err(io(path.display()))?;
        for vlr in &header.evlrs {
            out.write_all(&evlr_bytes(vlr)).map_err(io(path.display()))?;
        }
    }
    // laspy rewrites the header and the VLRs (ExtraBytes min/max) on close.
    if !extra.is_empty() {
        let at = vlrs.iter().position(|v| v.user() == "LASF_Spec" && v.record_id == 4).unwrap();
        vlrs[at] = crate::points::extra_bytes_vlr(&extra);
    }
    out.seek(SeekFrom::Start(0)).map_err(io(path.display()))?;
    out.write_all(&header_bytes(header, &vlrs, &stats, evlr_start, header.evlrs.len() as u32, compress)).map_err(io(path.display()))?;
    for vlr in &vlrs {
        out.write_all(&vlr_bytes(vlr)).map_err(io(path.display()))?;
    }
    out.flush().map_err(io(path.display()))?;
    Ok(stats.count)
}
