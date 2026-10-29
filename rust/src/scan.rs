//! Single-scan tile subsampling (`tile_centroids.py`): read every source point
//! once, spill it into XY blocks aligned to every output voxel grid, reduce
//! blocks independently and assemble each tile's outputs in block order.
//!
//! Spill files, block order, fragment names and the oversized-block split
//! follow the Python module step for step, so outputs (including the point
//! order of original-resolution tiles) are identical.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;
use smarttile_native::{reduce, voxel_keys, Coefficients, ReduceError};

use crate::decimal::{coefficients, py_round};
use crate::lasio::{self, Res, SourceHeader, TileHeader};

pub const CHUNK_POINTS: u64 = 1_000_000;
pub const BLOCK_POINTS: usize = 4_000_000;
const MAX_BLOCK_EDGE: f64 = 50.0;
const MIN_BLOCK_EDGE: f64 = 5.0;
const ASSEMBLY_POINTS: usize = 5_000_000;
const RECORD: usize = 13; // X, Y, Z (<i4) and a resolution bit mask (u1)

#[derive(Clone, Debug)]
pub struct Tile {
    pub label: String,
    pub bounds: [f64; 4], // xmin, ymin, xmax, ymax (inclusive)
    pub scales: [f64; 3],
    pub offsets: [f64; 3],
    pub core: Option<[f64; 4]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format { Laz, Copc, Ply }

pub struct Output {
    pub resolution: f64, // 0 = original points
    pub paths: BTreeMap<String, PathBuf>, // tile label -> final path
    pub format: Format,
}

#[derive(Clone, Copy)]
struct Rec { xyz: [i32; 3], m: u8 }

fn rec_bytes(records: &[Rec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * RECORD);
    for r in records {
        for v in r.xyz { out.extend_from_slice(&v.to_le_bytes()); }
        out.push(r.m);
    }
    out
}

fn read_records(path: &Path) -> Res<Vec<Rec>> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(parse_records(&bytes))
}

fn parse_records(bytes: &[u8]) -> Vec<Rec> {
    bytes.chunks_exact(RECORD).map(|c| Rec {
        xyz: std::array::from_fn(|a| i32::from_le_bytes(c[a * 4..a * 4 + 4].try_into().unwrap())),
        m: c[12],
    }).collect()
}

/// Reads a record file `limit` records at a time (`np.fromfile(handle, count=limit)`).
struct RecordBatches { reader: BufReader<File>, limit: usize }

impl Iterator for RecordBatches {
    type Item = Res<Vec<Rec>>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut bytes = Vec::with_capacity(self.limit * RECORD);
        match (&mut self.reader).take((self.limit * RECORD) as u64).read_to_end(&mut bytes) {
            Ok(0) => None,
            Ok(_) => Some(Ok(parse_records(&bytes))),
            Err(e) => Some(Err(e.to_string())),
        }
    }
}

fn batches(path: &Path, limit: usize) -> Res<RecordBatches> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(RecordBatches { reader: BufReader::with_capacity(1 << 20, file), limit })
}

fn routing_resolutions(resolutions: &[f64]) -> Vec<f64> {
    let grid = resolutions.iter().copied().filter(|r| *r > 0.0).fold(f64::INFINITY, f64::min);
    let grid = if grid.is_finite() { grid } else { 1.0 };
    resolutions.iter().map(|&r| if r > 0.0 { r } else { grid }).collect()
}

pub fn block_cells(resolutions: &[f64], block_size: f64) -> Vec<i64> {
    routing_resolutions(resolutions).iter().map(|r| (py_round(block_size / r) as i64).max(1)).collect()
}

/// `choose_block_size`: the largest edge leaving at most 10% of the sample in
/// blocks above BLOCK_POINTS, as whole coarsest voxels.
pub fn choose_block_size(point_count: u64, area: f64, resolutions: &[f64], sample: &[[f64; 2]]) -> f64 {
    let coarse = routing_resolutions(resolutions).into_iter().fold(f64::NEG_INFINITY, f64::max);
    let aligned = |edge: f64| coarse.max((edge / coarse).floor() * coarse);
    let mut edge = MAX_BLOCK_EDGE;
    if !sample.is_empty() {
        let scale = point_count as f64 / sample.len() as f64;
        while edge / 2.0 >= MIN_BLOCK_EDGE {
            let size = aligned(edge);
            let mut counts: BTreeMap<(i64, i64), u64> = BTreeMap::new();
            for p in sample {
                *counts.entry(((p[0] / size).floor() as i64, (p[1] / size).floor() as i64)).or_default() += 1;
            }
            let heavy: u64 = counts.values().filter(|&&c| c as f64 * scale > BLOCK_POINTS as f64).sum();
            if heavy as f64 <= 0.1 * sample.len() as f64 {
                break;
            }
            edge /= 2.0;
        }
        return aligned(edge);
    }
    let density = point_count as f64 / area.max(1e-9);
    aligned(MAX_BLOCK_EDGE.min((BLOCK_POINTS as f64 / density.max(1e-12)).sqrt()))
}

/// `np.linspace(0, stop, num).astype(np.int64)`.
fn linspace_starts(stop: f64, num: usize) -> Vec<u64> {
    if num == 1 {
        return vec![0];
    }
    let step = stop / (num - 1) as f64;
    (0..num).map(|i| if i + 1 == num { stop as u64 } else { (i as f64 * step) as u64 }).collect()
}

/// `sample_xy`: XY of evenly spaced point runs across all sources.
pub fn sample_xy(sources: &[PathBuf], headers: &[SourceHeader]) -> Res<Vec<[f64; 2]>> {
    const POINTS: f64 = 2_000_000.0;
    const RUN: u64 = 20_000;
    let total: u64 = headers.iter().map(|h| h.point_count).sum();
    let mut out = Vec::new();
    for (source, header) in sources.iter().zip(headers) {
        let count = header.point_count;
        let runs = ((POINTS * count as f64 / total.max(1) as f64 / RUN as f64).ceil() as usize).max(1);
        let mut reader = las::Reader::from_path(source).map_err(|e| format!("{}: {e}", source.display()))?;
        for start in linspace_starts(count.saturating_sub(RUN) as f64, runs) {
            reader.seek(start).map_err(|e| e.to_string())?;
            let chunk = reader.read_points(RUN.min(count - start)).map_err(|e| e.to_string())?;
            out.extend(chunk.x().zip(chunk.y()).map(|(x, y)| [x, y]));
        }
    }
    Ok(out)
}

pub(crate) fn level_coefficients(scales: &[f64; 3], offsets: &[f64; 3], resolution: f64) -> Res<Coefficients> {
    Ok([coefficients(scales[0], offsets[0], resolution)?, coefficients(scales[1], offsets[1], resolution)?,
        coefficients(scales[2], offsets[2], resolution)?])
}

pub(crate) fn reduce_error(error: ReduceError) -> String {
    match error {
        ReduceError::KeyOverflow => "Voxel keys exceed the int64 range".into(),
        ReduceError::Unsupported => "LAS encoding outside the native reducer's integer range".into(),
    }
}

/// XY routing context of one tile: per resolution the XY voxel coefficients and block cells.
pub(crate) struct Router { levels: Vec<(Coefficients, i64)> }

impl Router {
    pub(crate) fn new(resolutions: &[f64], cells: &[i64], scales: &[f64; 3], offsets: &[f64; 3]) -> Res<Router> {
        let mut levels = Vec::new();
        for (r, &k) in routing_resolutions(resolutions).into_iter().zip(cells) {
            let mut c = level_coefficients(scales, offsets, r)?;
            c[2] = (1, 0, 1); // Z is not routed
            levels.push((c, k));
        }
        Ok(Router { levels })
    }

    /// `route`: (row, block, bits) in Python's order: points whose block is the
    /// same at every resolution first, then per resolution the split copies.
    fn route(&self, records: &[Rec]) -> Res<Vec<(usize, [i64; 2], u8)>> {
        self.route_xym(&records.iter().map(|r| (r.xyz[0], r.xyz[1], r.m)).collect::<Vec<_>>())
    }

    /// `route` on (X, Y, mask) of any record type.
    pub(crate) fn route_xym(&self, records: &[(i32, i32, u8)]) -> Res<Vec<(usize, [i64; 2], u8)>> {
        let raw: Vec<[i32; 3]> = records.iter().map(|r| [r.0, r.1, 0]).collect();
        let mut blocks = Vec::with_capacity(self.levels.len());
        for (c, k) in &self.levels {
            let keys = voxel_keys(&raw, c).map_err(reduce_error)?;
            blocks.push(keys.into_iter().map(|key| [key[0].div_euclid(*k), key[1].div_euclid(*k)]).collect::<Vec<_>>());
        }
        let same: Vec<bool> = (0..records.len()).map(|i| blocks.iter().all(|b| b[i] == blocks[0][i])).collect();
        let mut out: Vec<(usize, [i64; 2], u8)> = (0..records.len()).filter(|&i| same[i]).map(|i| (i, blocks[0][i], records[i].2)).collect();
        for (index, block) in blocks.iter().enumerate() {
            let bit = 1u8 << index;
            out.extend((0..records.len()).filter(|&i| !same[i] && records[i].2 & bit != 0).map(|i| (i, block[i], bit)));
        }
        Ok(out)
    }
}

/// `_write_blocks`: append routed records to `directory/<bx>_<by>/<name>`, grouped by block in stable order.
fn write_blocks(directory: &Path, name: &str, records: &[Rec], routed: &[(usize, [i64; 2], u8)]) -> Res<()> {
    let mut by_block: BTreeMap<[i64; 2], Vec<Rec>> = BTreeMap::new();
    for &(row, block, bits) in routed {
        by_block.entry(block).or_default().push(Rec { xyz: records[row].xyz, m: bits });
    }
    for (block, recs) in by_block {
        let dir = directory.join(format!("{}_{}", block[0], block[1]));
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = dir.join(name);
        let mut file = OpenOptions::new().create(true).append(true).open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        file.write_all(&rec_bytes(&recs)).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

pub(crate) fn encode(value: f64, scale: f64, offset: f64) -> Res<i32> {
    let encoded = ((value - offset) / scale).round_ties_even();
    if encoded < i32::MIN as f64 || encoded > i32::MAX as f64 {
        return Err("Tile coordinates exceed the LAS int32 range".into());
    }
    Ok(encoded as i32)
}

struct SpillTask { id: usize, source: PathBuf, start: u64, count: u64 }

struct SpillCounts { tiles: Vec<u64>, cores: Vec<u64>, uncovered: u64 }

/// `_spill_range`: one point range of one source into every tile it falls in.
fn spill_range(task: &SpillTask, tiles: &[Tile], routers: &[Router], spill_dir: &Path) -> Res<SpillCounts> {
    let mut counts = SpillCounts { tiles: vec![0; tiles.len()], cores: vec![0; tiles.len()], uncovered: 0 };
    let full_mask = ((1u16 << routers[0].levels.len()) - 1) as u8;
    let name = format!("{:06}.bin", task.id);
    let mut reader = las::Reader::from_path(&task.source).map_err(|e| format!("{}: {e}", task.source.display()))?;
    reader.seek(task.start).map_err(|e| e.to_string())?;
    let mut read = 0u64;
    while read < task.count {
        let chunk = reader.read_points(CHUNK_POINTS.min(task.count - read)).map_err(|e| e.to_string())?;
        if chunk.is_empty() {
            return Err(format!("{}: expected {} points from {}, read {read}", task.source.display(), task.count, task.start));
        }
        read += chunk.len() as u64;
        let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = (chunk.x().collect(), chunk.y().collect(), chunk.z().collect());
        let (xmin, xmax) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let (ymin, ymax) = y.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let mut covered = vec![false; x.len()];
        for (i, tile) in tiles.iter().enumerate() {
            let b = tile.bounds;
            if !(b[2] >= xmin && b[0] <= xmax && b[3] >= ymin && b[1] <= ymax) {
                continue;
            }
            let rows: Vec<usize> = (0..x.len()).filter(|&r| x[r] >= b[0] && x[r] <= b[2] && y[r] >= b[1] && y[r] <= b[3]).collect();
            if rows.is_empty() {
                continue;
            }
            match tile.core {
                None => rows.iter().for_each(|&r| covered[r] = true),
                Some(c) => for &r in &rows {
                    if x[r] >= c[0] && x[r] <= c[2] && y[r] >= c[1] && y[r] <= c[3] {
                        counts.cores[i] += 1;
                        covered[r] = true;
                    }
                },
            }
            let mut records = Vec::with_capacity(rows.len());
            for &r in &rows {
                records.push(Rec { xyz: [encode(x[r], tile.scales[0], tile.offsets[0])?, encode(y[r], tile.scales[1], tile.offsets[1])?,
                                         encode(z[r], tile.scales[2], tile.offsets[2])?], m: full_mask });
            }
            let routed = routers[i].route(&records)?;
            write_blocks(&spill_dir.join(&tile.label), &name, &records, &routed)?;
            counts.tiles[i] += rows.len() as u64;
        }
        counts.uncovered += covered.iter().filter(|c| !**c).count() as u64;
    }
    Ok(counts)
}

pub(crate) fn block_key(path: &Path) -> (i64, i64) {
    let name = path.file_name().unwrap().to_string_lossy();
    let (bx, by) = name.split_once('_').unwrap();
    (bx.parse().unwrap(), by.parse().unwrap())
}

pub(crate) fn sorted_dir(dir: &Path, extension: Option<&str>) -> Res<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| extension.is_none_or(|ext| p.extension().is_some_and(|e| e == ext)))
        .collect();
    out.sort();
    Ok(out)
}

pub(crate) fn block_dirs(dir: &Path) -> Res<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
    out.sort_by_key(|p| block_key(p));
    Ok(out)
}

struct Level { resolution: f64, coefficients: Option<Coefficients> }

/// `_reduce_block` (originals excluded): one result per resolution, splitting
/// a block above `limit` points once into enough finer children.
fn reduce_block(block: &Path, levels: &[Level], resolutions: &[f64], cells: &[i64], tile: &Tile, limit: usize,
                nearest: bool) -> Res<Vec<(Vec<[i64; 3]>, u64)>> {
    let files = sorted_dir(block, Some("bin"))?;
    let total: u64 = files.iter().map(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0)).sum::<u64>() / RECORD as u64;
    let factor = if total as usize > limit {
        2i64.pow(((total.max(1) as f64 / limit as f64).sqrt().log2().ceil() as i64).max(1) as u32)
    } else { 1 };
    let finer: Vec<i64> = cells.iter().map(|&c| (c / factor).max(1)).collect();
    if total as usize <= limit || finer == cells {
        let mut records = Vec::with_capacity(total as usize);
        for path in &files {
            records.extend(read_records(path)?);
        }
        return reduce_records(&records, levels, tile, nearest);
    }
    let children = block.join("split");
    let router = Router::new(resolutions, &finer, &tile.scales, &tile.offsets)?;
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        for records in batches(path, limit)? {
            let records = records?;
            let routed = router.route(&records)?;
            write_blocks(&children, &name, &records, &routed)?;
        }
    }
    let mut parts: Vec<(Vec<[i64; 3]>, u64)> = levels.iter().map(|_| (Vec::new(), 0)).collect();
    for child in block_dirs(&children)? {
        for (part, (points, used)) in parts.iter_mut().zip(reduce_block(&child, levels, resolutions, &finer, tile, limit, nearest)?) {
            part.0.extend(points);
            part.1 += used;
        }
    }
    fs::remove_dir_all(&children).map_err(|e| e.to_string())?;
    Ok(parts)
}

/// `reduce_records(include_original=False)` on integer points.
fn reduce_records(records: &[Rec], levels: &[Level], tile: &Tile, nearest: bool) -> Res<Vec<(Vec<[i64; 3]>, u64)>> {
    let mut out = Vec::with_capacity(levels.len());
    for (index, level) in levels.iter().enumerate() {
        let Some(c) = level.coefficients.as_ref() else {
            out.push((Vec::new(), 0));
            continue;
        };
        let selected: Vec<[i32; 3]> = records.iter().filter(|r| r.m & (1 << index) != 0).map(|r| r.xyz).collect();
        let points = if selected.is_empty() { Vec::new() } else { reduce(&selected, c, tile.scales, nearest).map_err(reduce_error)? };
        out.push((points, selected.len() as u64));
        let _ = level.resolution;
    }
    Ok(out)
}

fn write_fragment(path: &Path, points: impl Iterator<Item = [i32; 3]>) -> Res<()> {
    let mut bytes = Vec::new();
    for p in points {
        for v in p { bytes.extend_from_slice(&v.to_le_bytes()); }
    }
    fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

struct Job { tile: usize, block: PathBuf, name: String }

/// `_reduce_and_write`: fragments for one block; [(output points, represented points)] per output.
fn reduce_and_write(job: &Job, tile: &Tile, levels: &[Level], resolutions: &[f64], cells: &[i64],
                    slot_dirs: &[PathBuf], nearest: bool) -> Res<Vec<(u64, u64)>> {
    let originals: Vec<usize> = (0..levels.len()).filter(|&i| levels[i].coefficients.is_none()).collect();
    let mut original_counts = vec![0u64; levels.len()];
    if !originals.is_empty() {
        let mut batch = 0usize;
        for source in sorted_dir(&job.block, Some("bin"))? {
            for points in batches(&source, BLOCK_POINTS)? {
                let points = points?;
                for &slot in &originals {
                    let selected: Vec<[i32; 3]> = points.iter().filter(|r| r.m & (1 << slot) != 0).map(|r| r.xyz).collect();
                    if selected.is_empty() {
                        continue;
                    }
                    original_counts[slot] += selected.len() as u64;
                    write_fragment(&slot_dirs[slot].join(format!("{}_{batch:09}.bin", job.name)), selected.into_iter())?;
                }
                batch += 1;
            }
        }
    }
    let results = if levels.iter().any(|l| l.coefficients.is_some()) {
        reduce_block(&job.block, levels, resolutions, cells, tile, BLOCK_POINTS, nearest)?
    } else {
        levels.iter().map(|_| (Vec::new(), 0)).collect()
    };
    fs::remove_dir_all(&job.block).map_err(|e| e.to_string())?;
    let mut counts = Vec::new();
    for (slot, (points, used)) in results.into_iter().enumerate() {
        if originals.contains(&slot) {
            counts.push((original_counts[slot], original_counts[slot]));
            continue;
        }
        if !points.is_empty() {
            // Representatives lie on the tile's LAS grid already.
            write_fragment(&slot_dirs[slot].join(format!("{}.bin", job.name)),
                           points.iter().map(|p| p.map(|v| v as i32)))?;
        }
        counts.push((points.len() as u64, used));
    }
    Ok(counts)
}

/// Fragments of one output in file-name order, about ASSEMBLY_POINTS at a time.
fn fragment_batches(directory: &Path) -> Res<impl Iterator<Item = Res<Vec<[i32; 3]>>>> {
    let fragments = sorted_dir(directory, Some("bin"))?;
    let mut iter = fragments.into_iter().peekable();
    Ok(std::iter::from_fn(move || {
        let mut pending = Vec::new();
        while let Some(path) = iter.next() {
            let bytes = match fs::read(&path) {
                Ok(b) => b,
                Err(e) => return Some(Err(format!("{}: {e}", path.display()))),
            };
            pending.extend(bytes.chunks_exact(12).map(|c| std::array::from_fn(|a| i32::from_le_bytes(c[a * 4..a * 4 + 4].try_into().unwrap()))));
            if pending.len() >= ASSEMBLY_POINTS {
                return Some(Ok(pending));
            }
        }
        if pending.is_empty() { None } else { Some(Ok(pending)) }
    }))
}

/// `_assemble_ply`: binary little-endian PLY of float64 XYZ, the CRS as `comment crs:`.
fn assemble_ply(directory: &Path, target: &Path, header: &TileHeader, expected: u64, crs: Option<&str>) -> Res<()> {
    let mut out = std::io::BufWriter::new(File::create(target).map_err(|e| format!("{}: {e}", target.display()))?);
    let comment = crs.map(|c| format!("comment crs: {c}\n")).unwrap_or_default();
    write!(out, "ply\nformat binary_little_endian 1.0\n{comment}element vertex {expected}\nproperty double x\nproperty double y\nproperty double z\nend_header\n")
        .map_err(|e| e.to_string())?;
    let mut written = 0u64;
    for batch in fragment_batches(directory)? {
        for p in batch? {
            for axis in 0..3 {
                let v = header.scales[axis] * f64::from(p[axis]) + header.offsets[axis];
                out.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
            }
            written += 1;
        }
    }
    out.flush().map_err(|e| e.to_string())?;
    if written != expected {
        return Err(format!("{}: wrote {written} of {expected} points", target.display()));
    }
    Ok(())
}

pub struct Summary { pub written: BTreeMap<String, Vec<u64>>, pub occupied: Vec<bool>, pub staged: Vec<(PathBuf, PathBuf)> }

/// `write_tile_centroids`. Returns per tile its output point counts, the
/// occupancy flags and the staged (scratch, final) files; the caller publishes.
#[allow(clippy::too_many_arguments)]
pub fn write_tile_centroids(sources: &[PathBuf], headers: &[SourceHeader], tiles: &[Tile], outputs: &[Output],
                            workers: usize, scratch: &Path, converters: usize, nearest: bool,
                            require_coverage: bool, copc: &(dyn Fn(&Path, &Path, u64) -> Res<()> + Sync),
                            ply_crs: Option<&str>) -> Res<Summary> {
    let resolutions: Vec<f64> = outputs.iter().map(|o| o.resolution).collect();
    if tiles.is_empty() || resolutions.is_empty() || resolutions.iter().any(|r| !r.is_finite() || *r < 0.0) {
        return Err("Tile outputs need tiles and non-negative resolutions (0 means original)".into());
    }
    if resolutions.len() > 8 {
        return Err("At most eight resolutions per scan".into());
    }
    let started = std::time::Instant::now();
    let spill_dir = scratch.join("blocks");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers.max(1)).build().map_err(|e| e.to_string())?;

    // `_range_tasks`: about two point ranges per worker, none tiny.
    let mut extent = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for h in headers {
        extent = [extent[0].min(h.mins[0]), extent[1].min(h.mins[1]), extent[2].max(h.maxs[0]), extent[3].max(h.maxs[1])];
    }
    let tiles_extent = tiles.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
        |e, t| [e[0].min(t.bounds[0]), e[1].min(t.bounds[1]), e[2].max(t.bounds[2]), e[3].max(t.bounds[3])]);
    let width = extent[2].min(tiles_extent[2]) - extent[0].max(tiles_extent[0]);
    let height = extent[3].min(tiles_extent[3]) - extent[1].max(tiles_extent[1]);
    let area = width.max(0.0) * height.max(0.0);
    let total_points: u64 = headers.iter().map(|h| h.point_count).sum();
    let size = CHUNK_POINTS.max(total_points.div_ceil((2 * workers.max(1)) as u64));
    let mut tasks = Vec::new();
    for (source, header) in sources.iter().zip(headers) {
        let mut start = 0;
        while start < header.point_count {
            tasks.push(SpillTask { id: tasks.len(), source: source.clone(), start, count: size.min(header.point_count - start) });
            start += size;
        }
    }
    let block_size = choose_block_size(total_points, area, &resolutions, &sample_xy(sources, headers)?);
    let cells = block_cells(&resolutions, block_size);
    let routers = tiles.iter().map(|t| Router::new(&resolutions, &cells, &t.scales, &t.offsets)).collect::<Res<Vec<_>>>()?;

    let results: Vec<Res<SpillCounts>> = pool.install(|| tasks.par_iter().map(|t| spill_range(t, tiles, &routers, &spill_dir)).collect());
    let mut tile_points = vec![0u64; tiles.len()];
    let mut core_points = vec![0u64; tiles.len()];
    let mut uncovered = 0;
    for r in results {
        let r = r?;
        for i in 0..tiles.len() {
            tile_points[i] += r.tiles[i];
            core_points[i] += r.cores[i];
        }
        uncovered += r.uncovered;
    }
    if require_coverage && uncovered > 0 {
        return Err(format!("{uncovered} source points lie outside every planned tile core (input header bounds do not cover the points)"));
    }
    let occupied: Vec<bool> = tiles.iter().enumerate().map(|(i, t)| t.core.is_none() || core_points[i] > 0).collect();
    if !occupied.iter().any(|o| *o) {
        return Err("No planned tile core contains source points".into());
    }
    for (i, tile) in tiles.iter().enumerate() {
        if !occupied[i] {
            let _ = fs::remove_dir_all(spill_dir.join(&tile.label));
        }
    }
    let spilled = started.elapsed().as_secs_f64();
    let spill_bytes = dir_size(&spill_dir);
    println!("  Read {} source points once with {workers} workers in {spilled:.1} s; {block_size:.1} m blocks; {:.2} GiB spilled",
             group_thousands(total_points), spill_bytes as f64 / (1u64 << 30) as f64);

    // Jobs: every occupied tile's blocks, in tile then block order.
    let mut jobs = Vec::new();
    let mut slot_dirs = BTreeMap::new();
    let mut headers_out = BTreeMap::new();
    let mut level_sets = BTreeMap::new();
    for (index, tile) in tiles.iter().enumerate() {
        if !occupied[index] {
            continue;
        }
        let tile_out = scratch.join("out").join(&tile.label);
        let dirs: Vec<PathBuf> = (0..outputs.len()).map(|slot| tile_out.join(slot.to_string())).collect();
        for d in &dirs {
            fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        slot_dirs.insert(index, dirs);
        headers_out.insert(index, TileHeader::for_tile(&headers[0], tile.scales, tile.offsets)?);
        let levels = resolutions.iter().map(|&r| Ok(Level { resolution: r,
            coefficients: if r > 0.0 { Some(level_coefficients(&tile.scales, &tile.offsets, r)?) } else { None } }))
            .collect::<Res<Vec<_>>>()?;
        level_sets.insert(index, levels);
        let tile_dir = spill_dir.join(&tile.label);
        if tile_dir.exists() {
            for block in block_dirs(&tile_dir)? {
                jobs.push(Job { tile: index, block, name: format!("{:09}", jobs.len()) });
            }
        }
    }
    let done = AtomicU64::new(0);
    let results: Vec<Res<Vec<(u64, u64)>>> = pool.install(|| jobs.par_iter().map(|job| {
        let r = reduce_and_write(job, &tiles[job.tile], &level_sets[&job.tile], &resolutions, &cells,
                                 &slot_dirs[&job.tile], nearest);
        done.fetch_add(1, Ordering::Relaxed);
        r
    }).collect());
    let mut totals: BTreeMap<usize, Vec<(u64, u64)>> = BTreeMap::new();
    for (job, result) in jobs.iter().zip(results) {
        let entry = totals.entry(job.tile).or_insert_with(|| vec![(0, 0); outputs.len()]);
        for (slot, (count, used)) in result?.into_iter().enumerate() {
            entry[slot].0 += count;
            entry[slot].1 += used;
        }
    }
    let reduced = started.elapsed().as_secs_f64();

    // Assemble every tile output (`converters` at a time).
    let mut finishing = Vec::new();
    let mut written = BTreeMap::new();
    for (&index, entry) in &totals {
        let tile = &tiles[index];
        for (slot, output) in outputs.iter().enumerate() {
            let (count, used) = entry[slot];
            if used != tile_points[index] || count == 0 {
                return Err(format!("{} at {} m: voxels represent {used} points of {}; {count} outputs",
                                   tile.label, output.resolution, tile_points[index]));
            }
            let extension = match output.format { Format::Laz => "laz", Format::Copc => "copc.laz", Format::Ply => "ply" };
            let target = scratch.join(format!("{}.{slot}.{extension}", tile.label));
            finishing.push((index, slot, target, output.paths[&tile.label].clone(), count));
        }
        written.insert(tile.label.clone(), entry.iter().map(|(c, _)| *c).collect::<Vec<_>>());
    }
    // `converters` plain threads take outputs in order; LAZ chunks compress on
    // the worker pool (a rayon thread waiting on another pool would run
    // unrelated assembly work meanwhile and stall its own output).
    let next = std::sync::atomic::AtomicUsize::new(0);
    let finished: Vec<std::sync::Mutex<Option<Res<()>>>> = finishing.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..converters.max(1) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((index, slot, target, _, count)) = finishing.get(i) else { break };
                let item_started = std::time::Instant::now();
                let directory = &slot_dirs[index][*slot];
                let header = &headers_out[index];
                let result = (|| match outputs[*slot].format {
                    Format::Laz => lasio::write_points(target, header, fragment_batches(directory)?, true, Some(&pool)).map(|_| ()),
                    Format::Copc => {
                        let merged = PathBuf::from(format!("{}.las", directory.display()));
                        lasio::write_points(&merged, header, fragment_batches(directory)?, false, None)?;
                        let converted = copc(&merged, target, *count);
                        let _ = fs::remove_file(&merged);
                        converted
                    }
                    Format::Ply => assemble_ply(directory, target, header, *count, ply_crs),
                })();
                let _ = fs::remove_dir_all(directory);
                if std::env::var_os("SMARTTILE_TIMINGS").is_some() {
                    eprintln!("    assembled {} in {:.1} s", target.display(), item_started.elapsed().as_secs_f64());
                }
                *finished[i].lock().unwrap() = Some(result);
            });
        }
    });
    let finished: Vec<Res<()>> = finished.into_iter().map(|m| m.into_inner().unwrap().unwrap_or(Err("output not assembled".into()))).collect();
    for r in finished {
        r?;
    }
    let staged = finishing.into_iter().map(|(_, _, target, final_path, _)| (target, final_path)).collect();
    println!("  Tile centroids: {} tiles, {:?} m, in {:.1} s: spill {spilled:.1} s, reduce/write {:.1} s, remaining conversion/assembly {:.1} s",
             written.len(), resolutions, started.elapsed().as_secs_f64(), reduced - spilled, started.elapsed().as_secs_f64() - reduced);
    let _ = (sources, done);
    Ok(Summary { written, occupied, staged })
}

pub(crate) fn dir_size(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else { return 0 };
    entries.filter_map(|e| e.ok()).map(|e| {
        let p = e.path();
        if p.is_dir() { dir_size(&p) } else { e.metadata().map(|m| m.len()).unwrap_or(0) }
    }).sum()
}

pub fn group_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 { out.push(','); }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_numpy() {
        // np.linspace(0, 1000, 4).astype(np.int64) == [0, 333, 666, 1000]
        assert_eq!(linspace_starts(1000.0, 4), vec![0, 333, 666, 1000]);
        assert_eq!(linspace_starts(0.0, 1), vec![0]);
    }

    #[test]
    fn nested_resolutions_share_block_edges() {
        assert_eq!(block_cells(&[0.1, 0.02, 0.0], 12.5), vec![125, 625, 625]);
    }
}
