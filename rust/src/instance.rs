//! Instance tiles (`instance_tiles.py`): tile clouds that already carry
//! instance IDs so that every instance appears exactly once and whole.
//!
//! 1. A statistics read sums every instance's points and XY per file; IDs
//!    reused across files at unrelated places fail the run.
//! 2. An instance belongs to the tile whose core holds its XY centroid
//!    (nearest core otherwise); background (ID 0) to the core holding the point.
//! 3. A second read spills every point to its owner tile in XY blocks; voxels
//!    are keyed by (instance, voxel) and keep their point nearest the mean.
//!
//! Point order, float summation order and fragment batches follow Python, so
//! outputs are identical.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use smarttile_native::voxel_keys;

use crate::lasio::{self, Res, SourceHeader, TileHeader};
use crate::points::{ExtraBytes, Layout, OutputExtra, PointLayout, Scalar};
use crate::scan::{self, Router, Tile, CHUNK_POINTS};

const REUSE_GAP: f64 = 5.0;
const REUSE_LIMIT: (f64, f64) = (10.0, 0.05);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InstanceFormat { Laz, Copc, Ply }

pub struct InstanceRequest {
    pub dimensions: Vec<String>,
    pub attributes: Vec<String>,
    pub resolutions: Vec<f64>,
    pub formats: Vec<InstanceFormat>,
    /// One directory per resolution, per dimension.
    pub directories: BTreeMap<String, Vec<PathBuf>>,
    pub check_ids: bool,
    pub wide: f64,
}

/// What a carried attribute is in the sources and the output.
#[derive(Clone)]
struct Carried { name: String, scalar: Scalar, extra: Option<ExtraBytes>, standard_in_output: bool }

/// `_collection_schema`: the instance dimensions and attributes, checked in every source.
fn collection_schema(sources: &[PathBuf], layouts: &[PointLayout], request: &InstanceRequest) -> Res<(Vec<Carried>, Vec<Carried>)> {
    let names: Vec<&String> = request.dimensions.iter().chain(&request.attributes).collect();
    let mut unique = names.clone();
    unique.sort();
    unique.dedup();
    if unique.len() != names.len() {
        return Err("Instance dimensions and carried attributes must be distinct".into());
    }
    let standard0 = PointLayout::of(&SourceHeader { point_format: 0, record_length: 20, ..layouts_dummy() })?.standard;
    let mut schema: BTreeMap<String, Carried> = BTreeMap::new();
    for (source, layout) in sources.iter().zip(layouts) {
        let name_of = || source.file_name().unwrap().to_string_lossy().into_owned();
        let missing: Vec<&&String> = names.iter().filter(|n| layout.find(n).is_none()).collect();
        if !missing.is_empty() {
            return Err(format!("{}: missing dimensions {:?}", name_of(), missing));
        }
        for name in &names {
            let (dim_layout, extra) = layout.find(name).unwrap();
            let is_dimension = request.dimensions.contains(name);
            match extra {
                None => {
                    let in0 = standard0.iter().find(|(n, _)| n == *name);
                    if is_dimension || in0.is_none() || dim_layout.num_bits() > in0.unwrap().1.num_bits() {
                        return Err(format!("{}: {name} cannot be carried by instance tiles (point format 0)", name_of()));
                    }
                    schema.entry(name.to_string()).or_insert(Carried { name: name.to_string(), scalar: dim_layout.scalar(), extra: None, standard_in_output: true });
                }
                Some(eb) => {
                    if standard0.iter().any(|(n, _)| n == *name) {
                        return Err(format!("{}: ExtraByte {name} conflicts with a standard output dimension", name_of()));
                    }
                    let scalar = eb.scalar().ok_or_else(|| format!("{}: {name} is not a scalar ExtraByte", name_of()))?;
                    if is_dimension {
                        if !scalar.is_integer() {
                            return Err(format!("{}: instance dimension {name} must contain scalar integer labels", name_of()));
                        }
                        if (eb.has_scale() && eb.scale[0] != 1.0) || (eb.has_offset() && eb.offset[0] != 0.0) {
                            return Err(format!("{}: scaled instance dimension {name} needs explicit normalization", name_of()));
                        }
                    }
                    let carried = Carried { name: name.to_string(), scalar, extra: Some(eb.clone()), standard_in_output: false };
                    match schema.get(name.as_str()) {
                        None => { schema.insert(name.to_string(), carried); }
                        Some(current) => {
                            let a = current.extra.as_ref().unwrap();
                            let same_meaning = a.has_no_data() == eb.has_no_data() && (!a.has_no_data() || a.no_data[..8] == eb.no_data[..8])
                                && a.has_scale() == eb.has_scale() && a.scale[0] == eb.scale[0]
                                && a.has_offset() == eb.has_offset() && a.offset[0] == eb.offset[0];
                            if !same_meaning {
                                return Err(format!("{}: incompatible schema for {name}", name_of()));
                            }
                            if current.scalar != scalar {
                                return Err(format!("{}: {name} has another type in another file; type promotion is not ported yet", name_of()));
                            }
                        }
                    }
                }
            }
        }
    }
    let pick = |list: &Vec<String>| list.iter().map(|n| schema[n].clone()).collect::<Vec<_>>();
    Ok((pick(&request.dimensions), pick(&request.attributes)))
}

fn layouts_dummy() -> SourceHeader {
    SourceHeader { version: (1, 2), file_source_id: 0, global_encoding: 0, guid: [0; 16], system_identifier: [0; 32],
                   generating_software: [0; 32], creation_day: 0, creation_year: 0, point_format: 0, record_length: 20,
                   point_count: 0, scales: [1.0; 3], offsets: [0.0; 3], mins: [0.0; 3], maxs: [0.0; 3], vlrs: Vec::new(), evlrs: Vec::new() }
}

struct Range { id: usize, file: usize, source: PathBuf, start: u64, count: u64 }

/// Raw records of one point range in CHUNK_POINTS chunks, with world XYZ.
fn for_chunks(range: &Range, mut f: impl FnMut(&las::PointData) -> Res<()>) -> Res<()> {
    let mut reader = las::Reader::from_path(&range.source).map_err(|e| format!("{}: {e}", range.source.display()))?;
    reader.seek(range.start).map_err(|e| e.to_string())?;
    let mut read = 0;
    while read < range.count {
        let chunk = reader.read_points(CHUNK_POINTS.min(range.count - read)).map_err(|e| e.to_string())?;
        if chunk.is_empty() {
            return Err(format!("{}: expected {} points from {}, read {read}", range.source.display(), range.count, range.start));
        }
        read += chunk.len() as u64;
        f(&chunk)?;
    }
    Ok(())
}

fn ids_of(chunk: &las::PointData, layout: Layout, name: &str) -> Res<Vec<u64>> {
    let mut scratch = [0u8; 8];
    let record_len = chunk.record_len();
    chunk.raw_bytes().chunks_exact(record_len).map(|r| {
        let value = layout.scalar().read_int(layout.get(r, &mut scratch));
        if value < 0 { Err(format!("Instance dimension {name} contains negative IDs")) } else { Ok(value as u64) }
    }).collect()
}

/// Per (instance, file): count, sum x, sum y, box — one row per chunk and ID, as `_statistics`.
#[derive(Clone)]
struct Row { id: u64, file: usize, count: f64, sx: f64, sy: f64, minx: f64, maxx: f64, miny: f64, maxy: f64 }

fn statistics(range: &Range, dims: &[(String, Layout)]) -> Res<Vec<Vec<Row>>> {
    let mut parts: Vec<Vec<Row>> = dims.iter().map(|_| Vec::new()).collect();
    for_chunks(range, |chunk| {
        let (x, y): (Vec<f64>, Vec<f64>) = (chunk.x().collect(), chunk.y().collect());
        for (d, (name, layout)) in dims.iter().enumerate() {
            let ids = ids_of(chunk, *layout, name)?;
            // np.unique order; np.bincount sums in point order.
            let mut rows: BTreeMap<u64, Row> = BTreeMap::new();
            for (i, &id) in ids.iter().enumerate() {
                if id == 0 { continue; }
                let row = rows.entry(id).or_insert(Row { id, file: range.file, count: 0.0, sx: 0.0, sy: 0.0,
                    minx: f64::INFINITY, maxx: f64::NEG_INFINITY, miny: f64::INFINITY, maxy: f64::NEG_INFINITY });
                row.count += 1.0;
                row.sx += x[i];
                row.sy += y[i];
                row.minx = row.minx.min(x[i]); row.maxx = row.maxx.max(x[i]);
                row.miny = row.miny.min(y[i]); row.maxy = row.maxy.max(y[i]);
            }
            parts[d].extend(rows.into_values());
        }
        Ok(())
    })?;
    Ok(parts)
}

struct Combined { ids: Vec<u64>, count: Vec<f64>, sx: Vec<f64>, sy: Vec<f64>, width: Vec<f64>, reused: Vec<u64>, shared: usize }

/// `_combine`: per instance totals (sums in row order) and the cross-file reuse check.
fn combine(rows: &[Row]) -> Combined {
    let mut totals: BTreeMap<u64, Row> = BTreeMap::new();
    let mut per_file: BTreeMap<(u64, usize), Row> = BTreeMap::new();
    for r in rows {
        let t = totals.entry(r.id).or_insert(Row { count: 0.0, sx: 0.0, sy: 0.0, minx: f64::INFINITY, maxx: f64::NEG_INFINITY,
                                                   miny: f64::INFINITY, maxy: f64::NEG_INFINITY, ..r.clone() });
        t.count += r.count; t.sx += r.sx; t.sy += r.sy;
        t.minx = t.minx.min(r.minx); t.maxx = t.maxx.max(r.maxx); t.miny = t.miny.min(r.miny); t.maxy = t.maxy.max(r.maxy);
        let f = per_file.entry((r.id, r.file)).or_insert(Row { minx: f64::INFINITY, maxx: f64::NEG_INFINITY,
                                                               miny: f64::INFINITY, maxy: f64::NEG_INFINITY, ..r.clone() });
        f.minx = f.minx.min(r.minx); f.maxx = f.maxx.max(r.maxx); f.miny = f.miny.min(r.miny); f.maxy = f.maxy.max(r.maxy);
    }
    let mut by_id: BTreeMap<u64, Vec<&Row>> = BTreeMap::new();
    for row in per_file.values() {
        by_id.entry(row.id).or_default().push(row);
    }
    let (mut reused, mut shared) = (Vec::new(), 0);
    for (id, rows) in &by_id {
        if rows.len() < 2 { continue; }
        shared += 1;
        let lo_x = rows.iter().map(|r| r.minx).fold(f64::NEG_INFINITY, f64::max);
        let hi_x = rows.iter().map(|r| r.maxx).fold(f64::INFINITY, f64::min);
        let lo_y = rows.iter().map(|r| r.miny).fold(f64::NEG_INFINITY, f64::max);
        let hi_y = rows.iter().map(|r| r.maxy).fold(f64::INFINITY, f64::min);
        if (lo_x - hi_x).max(0.0).hypot((lo_y - hi_y).max(0.0)) > REUSE_GAP {
            reused.push(*id);
        }
    }
    let t: Vec<&Row> = totals.values().collect();
    Combined { ids: t.iter().map(|r| r.id).collect(), count: t.iter().map(|r| r.count).collect(), sx: t.iter().map(|r| r.sx).collect(),
               sy: t.iter().map(|r| r.sy).collect(), width: t.iter().map(|r| (r.maxx - r.minx).max(r.maxy - r.miny)).collect(), reused, shared }
}

/// `check_unique_instances`.
fn check_unique(c: &Combined, dimension: &str, enabled: bool, wide: f64) -> Res<()> {
    let total = c.ids.len();
    let mut wide_ids: Vec<usize> = (0..total).filter(|&i| c.width[i] > wide).collect();
    if !wide_ids.is_empty() {
        wide_ids.sort_by(|&a, &b| c.width[b].total_cmp(&c.width[a]));
        let examples: Vec<String> = wide_ids.iter().take(5).map(|&i| format!("{} ({:.0} m)", c.ids[i], c.width[i])).collect();
        println!("  Warning: {dimension}: {} of {total} instances are wider than {} m, e.g. {}; their tiles extend accordingly",
                 wide_ids.len(), crate::pyfmt::float_repr(wide).trim_end_matches(".0"), examples.join(", "));
    }
    let limit = REUSE_LIMIT.0.max(REUSE_LIMIT.1 * total as f64);
    if enabled && c.reused.len() as f64 > limit {
        let examples: Vec<String> = c.reused.iter().take(8).map(|i| i.to_string()).collect();
        return Err(format!("{dimension}: {} of {total} instance IDs occur in several input files more than {} m apart (e.g. {}). Instance IDs must be unique across all input files; separately segmented files that each number from 1 reuse IDs. Renumber them, or pass --allow-reused-instance-ids if these really are single instances.",
                           c.reused.len(), crate::pyfmt::float_repr(REUSE_GAP).trim_end_matches(".0"), examples.join(", ")));
    }
    let _ = c.shared;
    Ok(())
}

/// `_core_index`: first tile whose closed core holds (x, y), else the nearest core.
fn core_index(x: f64, y: f64, cores: &[[f64; 4]]) -> usize {
    if let Some(i) = cores.iter().position(|c| x >= c[0] && x <= c[2] && y >= c[1] && y <= c[3]) {
        return i;
    }
    let mut best = (f64::INFINITY, 0);
    for (i, c) in cores.iter().enumerate() {
        let dx = (c[0] - x).max(0.0).max(x - c[2]);
        let dy = (c[1] - y).max(0.0).max(y - c[3]);
        let d = dx * dx + dy * dy;
        if d < best.0 { best = (d, i); }
    }
    best.1
}

/// Spill record: X, Y, Z (<i4), resolution mask (u1), instance (<u8), carried attributes (raw).
struct RecordLayout { attributes: Vec<usize>, size: usize }

impl RecordLayout {
    fn new(attributes: &[Carried]) -> RecordLayout {
        let sizes: Vec<usize> = attributes.iter().map(|a| a.scalar.size()).collect();
        RecordLayout { size: 21 + sizes.iter().sum::<usize>(), attributes: sizes }
    }
    fn xym(&self, r: &[u8]) -> (i32, i32, u8) {
        (i32::from_le_bytes(r[0..4].try_into().unwrap()), i32::from_le_bytes(r[4..8].try_into().unwrap()), r[12])
    }
    fn xyz(&self, r: &[u8]) -> [i32; 3] {
        std::array::from_fn(|a| i32::from_le_bytes(r[a * 4..a * 4 + 4].try_into().unwrap()))
    }
    fn instance(&self, r: &[u8]) -> u64 { u64::from_le_bytes(r[13..21].try_into().unwrap()) }
}

fn append_blocks(directory: &Path, name: &str, records: &[u8], layout: &RecordLayout, routed: &[(usize, [i64; 2], u8)]) -> Res<()> {
    let mut by_block: BTreeMap<[i64; 2], Vec<u8>> = BTreeMap::new();
    for &(row, block, bits) in routed {
        let r = &records[row * layout.size..(row + 1) * layout.size];
        let out = by_block.entry(block).or_default();
        let start = out.len();
        out.extend_from_slice(r);
        out[start + 12] = bits;
    }
    for (block, bytes) in by_block {
        let dir = directory.join(format!("{}_{}", block[0], block[1]));
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(name);
        OpenOptions::new().create(true).append(true).open(&path).and_then(|mut f| f.write_all(&bytes))
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn spill(range: &Range, dims: &[(String, Layout)], owners: &[BTreeMap<u64, usize>], tiles: &[Tile], routers: &[Router],
         attributes: &[(Layout, usize)], layout: &RecordLayout, levels: usize, spill_dir: &Path) -> Res<Vec<Vec<u64>>> {
    let full_mask = ((1u16 << levels) - 1) as u8;
    let cores: Vec<[f64; 4]> = tiles.iter().map(|t| t.core.unwrap()).collect();
    let name = format!("{:06}.bin", range.id);
    let mut counts: Vec<Vec<u64>> = dims.iter().map(|_| vec![0; tiles.len()]).collect();
    let mut scratch = [0u8; 8];
    for_chunks(range, |chunk| {
        let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = (chunk.x().collect(), chunk.y().collect(), chunk.z().collect());
        let record_len = chunk.record_len();
        let raw: Vec<&[u8]> = chunk.raw_bytes().chunks_exact(record_len).collect();
        let mut background: Option<Vec<usize>> = None;
        for (d, (dim_name, dim_layout)) in dims.iter().enumerate() {
            let ids = ids_of(chunk, *dim_layout, dim_name)?;
            if ids.contains(&0) && background.is_none() {
                background = Some((0..x.len()).map(|i| core_index(x[i], y[i], &cores)).collect());
            }
            let owner: Vec<usize> = ids.iter().enumerate().map(|(i, id)| if *id > 0 { owners[d][id] } else { background.as_ref().unwrap()[i] }).collect();
            let mut indices: Vec<usize> = owner.clone();
            indices.sort_unstable();
            indices.dedup();
            for index in indices {
                let tile = &tiles[index];
                let rows: Vec<usize> = (0..owner.len()).filter(|&i| owner[i] == index).collect();
                let mut records = vec![0u8; rows.len() * layout.size];
                for (k, &i) in rows.iter().enumerate() {
                    let r = &mut records[k * layout.size..(k + 1) * layout.size];
                    for (axis, v) in [x[i], y[i], z[i]].into_iter().enumerate() {
                        let encoded = scan::encode(v, tile.scales[axis], tile.offsets[axis])
                            .map_err(|_| "Instance tile coordinates exceed the LAS int32 range".to_string())?;
                        r[axis * 4..axis * 4 + 4].copy_from_slice(&encoded.to_le_bytes());
                    }
                    r[12] = full_mask;
                    r[13..21].copy_from_slice(&ids[i].to_le_bytes());
                    let mut at = 21;
                    for (j, (attr_layout, size)) in attributes.iter().enumerate() {
                        let value = attr_layout.get(raw[i], &mut scratch);
                        r[at..at + size].copy_from_slice(&value[..*size]);
                        at += layout.attributes[j];
                    }
                }
                let xym: Vec<(i32, i32, u8)> = (0..rows.len()).map(|k| layout.xym(&records[k * layout.size..])).collect();
                let routed = routers[index].route_xym(&xym)?;
                append_blocks(&spill_dir.join(format!("dimension_{d:03}")).join(&tile.label), &name, &records, layout, &routed)?;
                counts[d][index] += rows.len() as u64;
            }
        }
        Ok(())
    })?;
    Ok(counts)
}

/// Output record (point format 0 + ExtraBytes) of a spilled record.
struct Packer { dimension: (Layout, Scalar), attributes: Vec<Layout>, record_length: usize }

impl Packer {
    fn pack(&self, layout: &RecordLayout, r: &[u8], out: &mut Vec<u8>) -> Res<()> {
        let start = out.len();
        out.resize(start + self.record_length, 0);
        let o = &mut out[start..];
        o[..12].copy_from_slice(&r[..12]);
        let id = layout.instance(r);
        let (dim_layout, scalar) = self.dimension;
        let max = if scalar.is_signed() { (1u128 << (scalar.size() * 8 - 1)) - 1 } else { (1u128 << (scalar.size() * 8)) - 1 };
        if id as u128 > max {
            return Err(format!("Prediction dimension value {id} does not fit output type {scalar:?}"));
        }
        dim_layout.set(o, &id.to_le_bytes());
        let mut at = 21;
        for (j, l) in self.attributes.iter().enumerate() {
            let size = layout.attributes[j];
            l.set(o, &r[at..at + size]);
            at += size;
        }
        Ok(())
    }
}

/// `_reduce`: original records, or per (instance, voxel) the point nearest the mean.
#[allow(clippy::too_many_arguments)]
fn reduce_block(block: &Path, resolutions: &[f64], tile: &Tile, layout: &RecordLayout, packer: &Packer, fragments: &[PathBuf]) -> Res<Vec<(u64, u64)>> {
    let files = scan::sorted_dir(block, Some("bin"))?;
    let mut every = Vec::new();
    for path in &files {
        every.extend(fs::read(path).map_err(|e| e.to_string())?);
    }
    let records: Vec<&[u8]> = every.chunks_exact(layout.size).collect();
    let mut results = Vec::new();
    for (index, (&resolution, fragment)) in resolutions.iter().zip(fragments).enumerate() {
        let selected: Vec<&[u8]> = records.iter().copied().filter(|r| r[12] & (1 << index) != 0).collect();
        if selected.is_empty() {
            results.push((0, 0));
            continue;
        }
        let mut chosen: Vec<&[u8]> = selected.clone();
        if resolution > 0.0 {
            let xyz: Vec<[i32; 3]> = selected.iter().map(|r| layout.xyz(r)).collect();
            let c = scan::level_coefficients(&tile.scales, &tile.offsets, resolution)?;
            let keys = voxel_keys(&xyz, &c).map_err(scan::reduce_error)?;
            let mut order: Vec<usize> = (0..selected.len()).collect();
            order.sort_by(|&a, &b| (layout.instance(selected[a]), keys[a], xyz[a]).cmp(&(layout.instance(selected[b]), keys[b], xyz[b])));
            chosen.clear();
            let mut start = 0;
            while start < order.len() {
                let group = (layout.instance(selected[order[start]]), keys[order[start]]);
                let mut end = start + 1;
                while end < order.len() && (layout.instance(selected[order[end]]), keys[order[end]]) == group { end += 1; }
                let members = &order[start..end];
                let mut sums = [0i64; 3];
                for &m in members { for a in 0..3 { sums[a] += i64::from(xyz[m][a]); } }
                let n = members.len() as f64;
                let mean: [f64; 3] = std::array::from_fn(|a| sums[a] as f64 / n);
                let distance = |m: usize| {
                    let d: [f64; 3] = std::array::from_fn(|a| (f64::from(xyz[m][a]) - mean[a]) * tile.scales[a]);
                    (d[0] * d[0] + d[2] * d[2]) + d[1] * d[1]
                };
                let best = members.iter().copied().min_by(|&a, &b| distance(a).total_cmp(&distance(b)).then(xyz[a].cmp(&xyz[b]))).unwrap();
                chosen.push(selected[best]);
                start = end;
            }
        }
        let mut out = Vec::with_capacity(chosen.len() * packer.record_length);
        for r in &chosen {
            packer.pack(layout, r, &mut out)?;
        }
        fs::write(fragment, out).map_err(|e| e.to_string())?;
        results.push((chosen.len() as u64, selected.len() as u64));
    }
    fs::remove_dir_all(block).map_err(|e| e.to_string())?;
    Ok(results)
}

/// `write_instance_ply`: XYZ doubles, then the instance dimension and attributes.
fn write_ply(laz_header: &SourceHeader, header: &TileHeader, fields: &[(String, Layout, Scalar, bool)], fragments: &[PathBuf],
             target: &Path, count: u64) -> Res<()> {
    let mut lines = vec!["ply".to_string(), "format binary_little_endian 1.0".to_string()];
    if let Some(crs) = crate::crs::ply_comment(laz_header) {
        lines.push(format!("comment crs: {crs}"));
    }
    lines.push(format!("element vertex {count}"));
    lines.extend(["x", "y", "z"].iter().map(|a| format!("property double {a}")));
    for (name, _, scalar, scaled) in fields {
        let ply = if *scaled { "double" } else { match scalar { Scalar::U64 => "uint", Scalar::I64 => "int", s => s.ply() } };
        lines.push(format!("property {ply} {name}"));
    }
    lines.push("end_header".into());
    let mut out = std::io::BufWriter::new(fs::File::create(target).map_err(|e| e.to_string())?);
    out.write_all((lines.join("\n") + "\n").as_bytes()).map_err(|e| e.to_string())?;
    let record_length = header.record_length();
    let mut scratch = [0u8; 8];
    let mut written = 0;
    for fragment in fragments {
        let bytes = fs::read(fragment).map_err(|e| e.to_string())?;
        for r in bytes.chunks_exact(record_length) {
            for axis in 0..3 {
                let raw = i32::from_le_bytes(r[axis * 4..axis * 4 + 4].try_into().unwrap());
                out.write_all(&(header.scales[axis] * f64::from(raw) + header.offsets[axis]).to_le_bytes()).map_err(|e| e.to_string())?;
            }
            for (name, l, scalar, scaled) in fields {
                let v = l.get(r, &mut scratch);
                if *scaled {
                    let extra = header.extra.iter().find(|e| &e.source.name == name).unwrap();
                    let raw = match scalar { Scalar::F32 => f32::from_le_bytes(v[..4].try_into().unwrap()) as f64,
                                             Scalar::F64 => f64::from_le_bytes(v[..8].try_into().unwrap()), s => s.read_int(v) as f64 };
                    let value = raw * extra.source.scale[0] + extra.source.offset[0];
                    out.write_all(&value.to_le_bytes()).map_err(|e| e.to_string())?;
                    continue;
                }
                match scalar {
                    Scalar::U64 | Scalar::I64 => {
                        let value = scalar.read_int(v);
                        let (lo, hi) = if *scalar == Scalar::U64 { (0, u32::MAX as i128) } else { (i32::MIN as i128, i32::MAX as i128) };
                        if value < lo || value > hi {
                            return Err(format!("PLY attribute {name} exceeds the 32-bit range; use LAZ or COPC"));
                        }
                        out.write_all(&(value as i64 as i32).to_le_bytes()[..4]).map_err(|e| e.to_string())?;
                    }
                    _ => out.write_all(v).map_err(|e| e.to_string())?,
                }
            }
            written += 1;
        }
    }
    out.flush().map_err(|e| e.to_string())?;
    if written != count {
        return Err(format!("{}: wrote {written} of {count} points", target.display()));
    }
    Ok(())
}

pub struct InstanceResult { pub dimension: String, pub instances: usize, pub outputs: Vec<(f64, PathBuf, usize, u64)> }

/// `write_instance_tiles`: every (dimension, resolution) collection from two reads.
pub fn write_instance_tiles(sources: &[PathBuf], headers: &[SourceHeader], mut tiles: Vec<Tile>, request: &InstanceRequest,
                            scratch: &Path, workers: usize, copc: &(dyn Fn(&Path, &Path, u64, bool) -> Res<()> + Sync)) -> Res<(Vec<InstanceResult>, Vec<(PathBuf, PathBuf)>)> {
    let resolutions = &request.resolutions;
    if resolutions.is_empty() || resolutions.iter().any(|r| !r.is_finite() || *r < 0.0) {
        return Err("Instance tiles need nonnegative resolutions (zero keeps original points)".into());
    }
    let layouts = headers.iter().map(PointLayout::of).collect::<Res<Vec<_>>>()?;
    let (dims, attributes) = collection_schema(sources, &layouts, request)?;
    if resolutions.contains(&0.0) {
        tiles = crate::tile::original_coordinate_tiles(headers, tiles)?;
    }
    let started = std::time::Instant::now();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers.max(1)).build().map_err(|e| e.to_string())?;
    let total: u64 = headers.iter().map(|h| h.point_count).sum();
    let size = CHUNK_POINTS.max(total.div_ceil((2 * workers.max(1)) as u64));
    let mut ranges = Vec::new();
    for (file, (source, h)) in sources.iter().zip(headers).enumerate() {
        let mut start = 0;
        while start < h.point_count {
            ranges.push(Range { id: ranges.len(), file, source: source.clone(), start, count: size.min(h.point_count - start) });
            start += size;
        }
    }
    // Layouts of the instance dimensions and attributes, per source (they must agree).
    let dim_layouts: Vec<Vec<(String, Layout)>> = layouts.iter().map(|l| dims.iter().map(|d| (d.name.clone(), l.find(&d.name).unwrap().0)).collect()).collect();
    let attr_layouts: Vec<Vec<(Layout, usize)>> = layouts.iter().map(|l| attributes.iter().map(|a| (l.find(&a.name).unwrap().0, a.scalar.size())).collect()).collect();

    let stats: Vec<Res<Vec<Vec<Row>>>> = pool.install(|| ranges.par_iter().map(|r| statistics(r, &dim_layouts[r.file])).collect());
    let mut parts: Vec<Vec<Row>> = dims.iter().map(|_| Vec::new()).collect();
    for s in stats {
        for (d, rows) in s?.into_iter().enumerate() {
            parts[d].extend(rows);
        }
    }
    let cores: Vec<[f64; 4]> = tiles.iter().map(|t| t.core.unwrap()).collect();
    let mut owners = Vec::new();
    let mut instance_counts = Vec::new();
    for (d, dim) in dims.iter().enumerate() {
        let c = combine(&parts[d]);
        check_unique(&c, &dim.name, request.check_ids, request.wide)?;
        owners.push(c.ids.iter().enumerate().map(|(i, id)| {
            let n = c.count[i].max(1.0);
            (*id, core_index(c.sx[i] / n, c.sy[i] / n, &cores))
        }).collect::<BTreeMap<u64, usize>>());
        instance_counts.push(c.ids.len());
    }
    let area = if cores.iter().flatten().all(|v| v.is_finite()) {
        (cores.iter().map(|c| c[2]).fold(f64::NEG_INFINITY, f64::max) - cores.iter().map(|c| c[0]).fold(f64::INFINITY, f64::min))
            * (cores.iter().map(|c| c[3]).fold(f64::NEG_INFINITY, f64::max) - cores.iter().map(|c| c[1]).fold(f64::INFINITY, f64::min))
    } else { 1.0 }.max(1e-9);
    let block = scan::choose_block_size(total, area, resolutions, &scan::sample_xy(sources, headers)?);
    let cells = scan::block_cells(resolutions, block);
    let routers = tiles.iter().map(|t| Router::new(resolutions, &cells, &t.scales, &t.offsets)).collect::<Res<Vec<_>>>()?;
    let layout = RecordLayout::new(&attributes);
    let spill_dir = scratch.join("blocks");
    let spilled: Vec<Res<Vec<Vec<u64>>>> = pool.install(|| ranges.par_iter().map(|r| spill(r, &dim_layouts[r.file], &owners, &tiles, &routers,
        &attr_layouts[r.file], &layout, resolutions.len(), &spill_dir)).collect());
    let mut tile_points: Vec<Vec<u64>> = dims.iter().map(|_| vec![0; tiles.len()]).collect();
    for s in spilled {
        for (d, counts) in s?.into_iter().enumerate() {
            for (i, c) in counts.into_iter().enumerate() { tile_points[d][i] += c; }
        }
    }
    let spill_seconds = started.elapsed().as_secs_f64();
    let mut results = Vec::new();
    let mut products = Vec::new();
    for (d, dim) in dims.iter().enumerate() {
        let routed: u64 = tile_points[d].iter().sum();
        if routed != total {
            return Err(format!("{}: routed {routed} of {total} points", dim.name));
        }
        let staged = scratch.join("out").join(format!("dimension_{d:03}"));
        let mut outputs: Vec<(f64, PathBuf, usize, u64)> = resolutions.iter().enumerate().map(|(slot, r)| (*r, request.directories[&dim.name][slot].clone(), 0, 0)).collect();
        for (index, tile) in tiles.iter().enumerate() {
            let tile_dir = spill_dir.join(format!("dimension_{d:03}")).join(&tile.label);
            if !tile_dir.exists() {
                continue;
            }
            // `_output_header`: point format 0 with the source CRS, then the instance and non-standard attributes.
            let mut header = TileHeader::for_tile(&headers[0], tile.scales, tile.offsets)?;
            header.extra = std::iter::once(dim).chain(attributes.iter().filter(|a| !a.standard_in_output))
                .map(|c| OutputExtra::new(c.extra.as_ref().unwrap())).collect();
            let mut offset = lasio::RECORD_LEN;
            let mut out_layout: BTreeMap<String, Layout> = BTreeMap::new();
            for e in &header.extra {
                out_layout.insert(e.source.name.clone(), Layout::Bytes { offset, scalar: e.source.scalar().unwrap() });
                offset += e.source.size();
            }
            let format0 = PointLayout::of(&SourceHeader { point_format: 0, record_length: 20, ..layouts_dummy() })?;
            let field = |name: &str| out_layout.get(name).copied().or_else(|| format0.find(name).map(|f| f.0)).unwrap();
            let packer = Packer { dimension: (field(&dim.name), dim.scalar), attributes: attributes.iter().map(|a| field(&a.name)).collect(),
                                  record_length: header.record_length() };
            let fragments: Vec<PathBuf> = (0..resolutions.len()).map(|slot| staged.join(&tile.label).join(slot.to_string())).collect();
            for f in &fragments { fs::create_dir_all(f).map_err(|e| e.to_string())?; }
            let blocks = scan::block_dirs(&tile_dir)?;
            let block_results: Vec<Res<Vec<(u64, u64)>>> = pool.install(|| blocks.par_iter().enumerate().map(|(n, b)| {
                let files: Vec<PathBuf> = fragments.iter().map(|f| f.join(format!("{n:09}.bin"))).collect();
                reduce_block(b, resolutions, tile, &layout, &packer, &files)
            }).collect());
            let mut kept = vec![0u64; resolutions.len()];
            let mut used = vec![0u64; resolutions.len()];
            for r in block_results {
                for (slot, (k, u)) in r?.into_iter().enumerate() { kept[slot] += k; used[slot] += u; }
            }
            for (slot, &resolution) in resolutions.iter().enumerate() {
                if used[slot] != tile_points[d][index] {
                    return Err(format!("{} {} at {resolution} m: voxels represent {} of {} points", dim.name, tile.label, used[slot], tile_points[d][index]));
                }
                let files = scan::sorted_dir(&fragments[slot], Some("bin"))?;
                let laz = staged.join(format!("{}.{slot}.laz", tile.label));
                let batches = files.iter().map(|f| fs::read(f).map_err(|e| e.to_string()));
                lasio::write_records(&laz, &header, batches, true, Some(&pool))?;
                let target = match request.formats[slot] {
                    InstanceFormat::Laz => laz,
                    InstanceFormat::Ply => {
                        let ply = laz.with_extension("ply");
                        let fields: Vec<(String, Layout, Scalar, bool)> = std::iter::once(&dim.name).chain(attributes.iter().map(|a| &a.name))
                            .map(|n| {
                                let l = field(n);
                                let scaled = header.extra.iter().find(|e| &e.source.name == n)
                                    .is_some_and(|e| (e.source.has_scale() && e.source.scale[0] != 1.0) || (e.source.has_offset() && e.source.offset[0] != 0.0));
                                if scaled && matches!(l.scalar(), Scalar::U64 | Scalar::I64) {
                                    return Err(format!("PLY cannot retain scaled 64-bit integer attribute {n}; use LAZ or COPC"));
                                }
                                Ok((n.clone(), l, l.scalar(), scaled))
                            }).collect::<Res<Vec<_>>>()?;
                        write_ply(&lasio::read_header(&laz)?, &header, &fields, &files, &ply, kept[slot])?;
                        fs::remove_file(&laz).map_err(|e| e.to_string())?;
                        ply
                    }
                    InstanceFormat::Copc => {
                        if header.extra.iter().any(|e| e.source.has_no_data()) {
                            return Err("COPC instance tiles with ExtraBytes no-data values are not ported yet".into());
                        }
                        let copc_path = PathBuf::from(format!("{}.copc.laz", laz.with_extension("").display()));
                        copc(&laz, &copc_path, kept[slot], true)?;
                        fs::remove_file(&laz).map_err(|e| e.to_string())?;
                        copc_path
                    }
                };
                let extension = match request.formats[slot] { InstanceFormat::Laz => "laz", InstanceFormat::Ply => "ply", InstanceFormat::Copc => "copc.laz" };
                products.push((target, outputs[slot].1.join(format!("{}.{extension}", tile.label))));
                outputs[slot].2 += 1;
                outputs[slot].3 += kept[slot];
            }
            let _ = fs::remove_dir_all(staged.join(&tile.label));
        }
        results.push(InstanceResult { dimension: dim.name.clone(), instances: instance_counts[d], outputs });
    }
    let labels: Vec<String> = dims.iter().map(|d| format!("'{}'", d.name)).collect();
    let res: Vec<String> = resolutions.iter().map(|r| crate::pyfmt::float_repr(*r)).collect();
    println!("  Instance tiles [{}] at [{}] m: {} points read twice in {spill_seconds:.1} s; written in {:.1} s",
             labels.join(", "), res.join(", "), scan::group_thousands(total), started.elapsed().as_secs_f64() - spill_seconds);
    Ok((results, products))
}
