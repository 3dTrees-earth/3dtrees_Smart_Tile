//! The `remap` task (`remap_first_pipeline.strict_remap`): every original
//! point takes the predictions of its nearest final merge point within
//! √3 × resolution_1, with 100% coverage required; instance IDs are then
//! compacted to 1..N across the dataset (`instance_finalization.py`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde_json::{json, Value};

use crate::lasio::{self, Res, SourceHeader, TileHeader, Vlr};
use crate::ld::local_coordinate;
use crate::points::{ExtraBytes, Layout, OutputExtra, PointLayout, Scalar};
use crate::parts::Parts;

const MAX_BATCH_POINTS: usize = 32_768;
const ORIGINAL_WRITE_POINTS: u64 = 1_000_000;
const REGION_SIZE: f64 = 4.0;
const CELL_SIZE: f64 = 2.0;
const BLOCK_POINTS: f64 = 4_000_000.0;
const BLOCK_POINT_CAP: usize = 8_000_000;
const MAX_BLOCK: f64 = 64.0;
/// `READ_POINTS`: spill reads and oversized-block splits go a million records at a time.
const READ_POINTS: u64 = 1_000_000;
/// Bytes of spill records buffered across all open blocks or cells before they are appended.
const SPILL_BUFFER: usize = 64 << 20;
/// Query batches held at a time during enrichment (about two million points).
const QUERY_WINDOW: usize = 64;

pub struct RemapParams {
    pub collections: Vec<PathBuf>,
    pub originals: PathBuf,
    pub output: PathBuf,
    pub workers: usize,
    pub requested_workers: usize,
    pub instance_dimension: String,
    pub target_dims: Option<BTreeSet<String>>,
    /// `--match-instances` and `--prediction-resolution-original`.
    pub match_instances: bool,
    pub prediction_resolution_original: bool,
    /// `--resolution-1` (overrides the manifests), `--remap-tolerance` (radius override), `--baseline-1cm-folders`.
    pub resolution_1: Option<f64>,
    pub remap_tolerance: Option<f64>,
    pub baseline_folders: Option<Vec<PathBuf>>,
}

fn local(raw: &[u8], h: &SourceHeader, origin: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|a| local_coordinate(i32::from_le_bytes(raw[a * 4..a * 4 + 4].try_into().unwrap()), h.scales[a], h.offsets[a], origin[a]))
}

fn chunks(path: &Path, size: u64) -> Res<impl Iterator<Item = Res<las::PointData>>> {
    let mut reader = las::Reader::from_path(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(std::iter::from_fn(move || match reader.read_points(size) {
        Ok(c) if c.is_empty() => None,
        Ok(c) => Some(Ok(c)),
        Err(e) => Some(Err(e.to_string())),
    }))
}

/// One prediction collection: its dimensions and its points with raw values, part by part.
pub struct Collection { pub name: String, pub instance: Option<String>, pub dimensions: Vec<ExtraBytes>, pub parts: Parts, pub width: usize, pub maximum: u64 }

/// A collection's files become parts (tile = file number, point index = file order) under `scratch`.
pub fn load_collection(path: &Path, origin: &[f64; 3], radius: f64, instance_dimension: &str, scratch: &Path) -> Res<Collection> {
    load(path, origin, radius, instance_dimension, true, scratch)
}

/// The geometry of a baseline collection (validated as `load_collection`), queried only for
/// nearest distances: no values, no tile or point references.
pub fn load_baseline(path: &Path, origin: &[f64; 3], radius: f64, instance_dimension: &str, scratch: &Path) -> Res<Parts> {
    Ok(load(path, origin, radius, instance_dimension, false, scratch)?.parts)
}

fn load(path: &Path, origin: &[f64; 3], radius: f64, instance_dimension: &str, with_values: bool, scratch: &Path) -> Res<Collection> {
    // `prediction_collection_files`: a collection folder, or one LAS/LAZ file (`--merged-laz`).
    let files = if path.is_file() {
        let name = path.file_name().unwrap().to_string_lossy().to_lowercase();
        if name.ends_with(".las") || name.ends_with(".laz") { vec![path.to_path_buf()] } else { Vec::new() }
    } else {
        crate::tile::point_cloud_files(path)?
    };
    if files.is_empty() {
        return Err(format!("No prediction files in {}", path.display()));
    }
    let headers = files.iter().map(|f| lasio::read_header(f)).collect::<Res<Vec<_>>>()?;
    let standard = crate::points::all_standard_names();
    let layout0 = PointLayout::of(&headers[0])?;
    let names: Vec<String> = layout0.extra.iter().map(|(e, _)| e.name.clone()).collect();
    let mut instances: Vec<String> = names.iter().filter(|n| n.starts_with("PredInstance")).cloned().collect();
    if names.iter().any(|n| n == instance_dimension) && !instances.iter().any(|n| n == instance_dimension) {
        instances.push(instance_dimension.into());
    }
    if instances.is_empty() && names.iter().any(|n| n == "treeID") { instances.push("treeID".into()); }
    if instances.len() > 1 {
        return Err(format!("{}: expected one model instance dimension, got {instances:?}", path.display()));
    }
    let instance = instances.pop();
    let dimensions: Vec<ExtraBytes> = layout0.extra.iter().filter(|(e, _)| !standard.contains(&e.name)).map(|(e, _)| e.clone()).collect();
    let width: usize = dimensions.iter().map(|d| d.size()).sum();
    let mut parts = Parts::new(scratch, radius, if with_values { width } else { 0 }, with_values)?;
    // Files are read in parallel, each written as its part as soon as it is read.
    let written: Vec<Res<(Vec<crate::parts::Part>, u64)>> = files.par_iter().zip(&headers).enumerate().map(|(tile, (file, header))| {
        let layout = PointLayout::of(header)?;
        let fields: Vec<Layout> = dimensions.iter().map(|d| layout.find(&d.name).map(|f| f.0).ok_or(format!("{}: inconsistent prediction schema", file.display()))).collect::<Res<_>>()?;
        let instance_field = instance.as_ref().map(|n| layout.find(n).unwrap().0);
        let (mut xyz, mut values, mut labels, mut maximum) = (Vec::new(), Vec::new(), Vec::new(), 0u64);
        let mut scratch = [0u8; 8];
        for chunk in chunks(file, READ_POINTS)? {
            let chunk = chunk?;
            for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
                xyz.push(local(r, header, origin));
                if with_values { for f in &fields { values.extend_from_slice(f.get(r, &mut scratch)); } }
                let mut label = 0;
                if let Some(f) = instance_field {
                    let v = f.scalar().read_int(f.get(r, &mut scratch));
                    if v < 0 { return Err(format!("{}: invalid or missing labels", file.display())); }
                    maximum = maximum.max(v as u64);
                    label = v as u64;
                }
                if with_values { labels.push(label); }
            }
        }
        let index: Vec<i64> = if with_values { (0..xyz.len() as i64).collect() } else { Vec::new() };
        Ok((parts.write(tile, tile as u32, &xyz, &index, &labels, &values)?, maximum))
    }).collect();
    let mut maximum = 0;
    for item in written {
        let (part, m) = item?;
        parts.push(part);
        maximum = maximum.max(m);
    }
    Ok(Collection { name: path.file_name().unwrap().to_string_lossy().into_owned(), instance, dimensions, parts, width, maximum })
}

/// `block_size`: the spill block edge in whole 4 m regions.
fn block_size(count: u64, h: &SourceHeader) -> f64 {
    let area = ((h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1])).max(1e-9);
    let edge = (BLOCK_POINTS * area / (count.max(1) as f64)).sqrt();
    MAX_BLOCK.min(REGION_SIZE.max((edge / REGION_SIZE).floor() * REGION_SIZE))
}

/// A limit, lowered by an environment variable in tests (spill and oversized-block paths on small data).
fn test_limit(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// (point indices, local XYZ) of one query batch.
type QueryBatch = (Vec<u32>, Vec<[f64; 3]>);

/// `spill_original` + `block_batches`: (point indices, local XYZ) batches, block by block,
/// ordered by 2 m cell then point index, 32,768 at a time. Originals up to one block
/// (`BLOCK_POINT_CAP`) are ordered in memory; larger ones spill to `scratch` first, so memory
/// does not grow with the original (`SMARTTILE_SPILL_POINTS` lowers the threshold for tests).
fn query_batches<'a>(file: &Path, header: &'a SourceHeader, origin: &'a [f64; 3], scratch: &Path)
                     -> Res<Box<dyn Iterator<Item = Res<QueryBatch>> + 'a>> {
    if header.point_count <= test_limit("SMARTTILE_SPILL_POINTS", BLOCK_POINT_CAP) as u64 {
        return Ok(Box::new(memory_batches(file, header, origin)?.into_iter().map(Ok)));
    }
    if header.point_count >= 1 << 32 {
        return Err(format!("{}: more than 2^32 points per original are not supported", file.display()));
    }
    let blocks = spill_original(file, header, origin, scratch)?;
    Ok(Box::new(spilled_batches(blocks, header, origin)))
}

fn memory_batches(file: &Path, header: &SourceHeader, origin: &[f64; 3]) -> Res<Vec<QueryBatch>> {
    let size = block_size(header.point_count, header);
    let mut blocks: BTreeMap<(i64, i64), Vec<(u32, [f64; 3])>> = BTreeMap::new();
    let mut i = 0u32;
    for chunk in chunks(file, READ_POINTS)? {
        let chunk = chunk?;
        for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
            let p = local(r, header, origin);
            blocks.entry(((p[0] / size).floor() as i64, (p[1] / size).floor() as i64)).or_default().push((i, p));
            i += 1;
        }
    }
    let mut out = Vec::new();
    for (_, mut points) in blocks {
        let cell = |p: &[f64; 3]| ((p[0] / CELL_SIZE).floor() as i64, (p[1] / CELL_SIZE).floor() as i64);
        points.sort_by_key(|(i, p)| (cell(p), *i));
        if points.len() <= test_limit("SMARTTILE_BLOCK_POINT_CAP", BLOCK_POINT_CAP) {
            for batch in points.chunks(MAX_BATCH_POINTS) {
                out.push((batch.iter().map(|b| b.0).collect(), batch.iter().map(|b| b.1).collect()));
            }
        } else {
            // Oversized blocks stream cell by cell, ordered by 4 m region.
            let mut cells: BTreeMap<(i64, i64, i64, i64), Vec<(u32, [f64; 3])>> = BTreeMap::new();
            let per_region = (REGION_SIZE / CELL_SIZE).round() as i64;
            for (i, p) in points {
                let (cx, cy) = cell(&p);
                cells.entry((cx.div_euclid(per_region), cy.div_euclid(per_region), cx, cy)).or_default().push((i, p));
            }
            for (_, points) in cells {
                for batch in points.chunks(MAX_BATCH_POINTS) {
                    out.push((batch.iter().map(|b| b.0).collect(), batch.iter().map(|b| b.1).collect()));
                }
            }
        }
    }
    Ok(out)
}

/// A spilled record: point index and raw X, Y, Z (`SPILL`).
const SPILL_RECORD: usize = 16;

fn spill_record(index: u32, raw: &[u8]) -> [u8; SPILL_RECORD] {
    let mut r = [0u8; SPILL_RECORD];
    r[..4].copy_from_slice(&index.to_le_bytes());
    r[4..].copy_from_slice(&raw[..12]);
    r
}

fn spilled_point(r: &[u8], h: &SourceHeader, origin: &[f64; 3]) -> (u32, [f64; 3]) {
    (u32::from_le_bytes(r[..4].try_into().unwrap()), local(&r[4..], h, origin))
}

/// Records appended to one file per key, buffered across keys up to `SPILL_BUFFER` bytes.
struct Appender<K: Ord + Copy> { buffers: BTreeMap<K, Vec<u8>>, buffered: usize, path: Box<dyn Fn(K) -> PathBuf> }

impl<K: Ord + Copy> Appender<K> {
    fn push(&mut self, key: K, record: &[u8]) -> Res<()> {
        self.buffers.entry(key).or_default().extend_from_slice(record);
        self.buffered += record.len();
        if self.buffered >= SPILL_BUFFER { self.flush()?; }
        Ok(())
    }

    fn flush(&mut self) -> Res<()> {
        use std::io::Write;
        for (key, buffer) in &mut self.buffers {
            if buffer.is_empty() { continue; }
            let path = (self.path)(*key);
            let mut out = fs::OpenOptions::new().create(true).append(true).open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            out.write_all(buffer).map_err(|e| format!("{}: {e}", path.display()))?;
            buffer.clear();
        }
        self.buffered = 0;
        Ok(())
    }
}

/// `spill_original`: every point to its XY block file under `dir`, in point order;
/// the blocks in key order with their files and point counts.
fn spill_original(file: &Path, header: &SourceHeader, origin: &[f64; 3], dir: &Path) -> Res<Vec<(PathBuf, u64)>> {
    let size = block_size(header.point_count, header);
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let root = dir.to_path_buf();
    let mut out = Appender { buffers: BTreeMap::new(), buffered: 0, path: Box::new(move |(bx, by): (i64, i64)| root.join(format!("{bx}_{by}.bin"))) };
    let mut counts: BTreeMap<(i64, i64), u64> = BTreeMap::new();
    let mut start = 0u32;
    for chunk in chunks(file, READ_POINTS)? {
        let chunk = chunk?;
        let record_len = chunk.record_len();
        let keyed: Vec<Vec<((i64, i64), [u8; SPILL_RECORD])>> = chunk.raw_bytes().par_chunks(MAX_BATCH_POINTS * record_len).enumerate().map(|(n, batch)| {
            batch.chunks_exact(record_len).enumerate().map(|(k, r)| {
                let p = local(r, header, origin);
                (((p[0] / size).floor() as i64, (p[1] / size).floor() as i64), spill_record(start + (n * MAX_BATCH_POINTS + k) as u32, r))
            }).collect()
        }).collect();
        for (key, record) in keyed.into_iter().flatten() {
            *counts.entry(key).or_default() += 1;
            out.push(key, &record)?;
        }
        start += (chunk.raw_bytes().len() / record_len) as u32;
    }
    out.flush()?;
    Ok(counts.into_iter().map(|((bx, by), n)| (dir.join(format!("{bx}_{by}.bin")), n)).collect())
}

/// Spill records of `path`, `count` at a time.
fn spill_reader(path: &Path) -> Res<impl FnMut(usize) -> Res<Vec<u8>>> {
    use std::io::Read;
    let mut reader = std::io::BufReader::with_capacity(1 << 20, fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?);
    let name = path.display().to_string();
    Ok(move |count: usize| {
        let mut buffer = Vec::with_capacity(count * SPILL_RECORD);
        reader.by_ref().take((count * SPILL_RECORD) as u64).read_to_end(&mut buffer).map_err(|e| format!("{name}: {e}"))?;
        Ok(buffer)
    })
}

/// `block_batches` over spilled blocks: a block up to `BLOCK_POINT_CAP` is sorted in memory;
/// a larger one is split into one file per 2 m cell and streamed cell by cell (4 m regions first).
fn spilled_batches<'a>(blocks: Vec<(PathBuf, u64)>, header: &'a SourceHeader, origin: &'a [f64; 3]) -> impl Iterator<Item = Res<QueryBatch>> + 'a {
    let mut blocks: std::collections::VecDeque<(PathBuf, u64)> = blocks.into();
    let mut ready: std::collections::VecDeque<QueryBatch> = std::collections::VecDeque::new();
    let mut cells: std::collections::VecDeque<PathBuf> = std::collections::VecDeque::new();
    let mut current: Option<(PathBuf, Box<dyn FnMut(usize) -> Res<Vec<u8>>>)> = None;
    let mut failed = false;
    let cell_of = |p: &[f64; 3]| ((p[0] / CELL_SIZE).floor() as i64, (p[1] / CELL_SIZE).floor() as i64);
    let mut next = move || -> Res<Option<QueryBatch>> {
        loop {
            if let Some(batch) = ready.pop_front() { return Ok(Some(batch)); }
            if let Some((path, read)) = current.as_mut() {
                let bytes = read(MAX_BATCH_POINTS)?;
                if !bytes.is_empty() {
                    return Ok(Some(bytes.chunks_exact(SPILL_RECORD).map(|r| spilled_point(r, header, origin)).unzip()));
                }
                fs::remove_file(&*path).map_err(|e| format!("{}: {e}", path.display()))?;
                current = None;
            }
            if let Some(cell) = cells.pop_front() {
                let read = spill_reader(&cell)?;
                current = Some((cell, Box::new(read)));
                continue;
            }
            let Some((path, count)) = blocks.pop_front() else { return Ok(None) };
            if count as usize <= test_limit("SMARTTILE_BLOCK_POINT_CAP", BLOCK_POINT_CAP) {
                let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let mut points: Vec<(u32, [f64; 3])> = bytes.par_chunks(SPILL_RECORD).map(|r| spilled_point(r, header, origin)).collect();
                points.par_sort_by_key(|(i, p)| (cell_of(p), *i));
                for batch in points.chunks(MAX_BATCH_POINTS) {
                    ready.push_back((batch.iter().map(|b| b.0).collect(), batch.iter().map(|b| b.1).collect()));
                }
            } else {
                // Records stay in point order within each cell file.
                let dir = path.with_extension("cells");
                fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                let root = dir.clone();
                let mut out = Appender { buffers: BTreeMap::new(), buffered: 0, path: Box::new(move |(cx, cy): (i64, i64)| root.join(format!("{cx}_{cy}.bin"))) };
                let mut read = spill_reader(&path)?;
                loop {
                    let bytes = read(READ_POINTS as usize)?;
                    if bytes.is_empty() { break; }
                    let keyed: Vec<(i64, i64)> = bytes.par_chunks(SPILL_RECORD).map(|r| cell_of(&spilled_point(r, header, origin).1)).collect();
                    for (key, r) in keyed.into_iter().zip(bytes.chunks_exact(SPILL_RECORD)) { out.push(key, r)?; }
                }
                out.flush()?;
                let per_region = (REGION_SIZE / CELL_SIZE).round() as i64;
                let mut keys: Vec<(i64, i64)> = out.buffers.keys().copied().collect();
                keys.sort_by_key(|&(cx, cy)| (cx.div_euclid(per_region), cy.div_euclid(per_region), cx, cy));
                cells.extend(keys.into_iter().map(|(cx, cy)| dir.join(format!("{cx}_{cy}.bin"))));
            }
            fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    };
    std::iter::from_fn(move || {
        if failed { return None; }
        match next() {
            Ok(batch) => batch.map(Ok),
            Err(e) => { failed = true; Some(Err(e)) }
        }
    })
}

fn write_json(path: &Path, value: &Value, newline: bool) -> Res<()> {
    let text = crate::pyfmt::dumps(value) + if newline { "\n" } else { "" };
    fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn run(params: &RemapParams) -> Res<()> {
    let started = std::time::Instant::now();
    if params.output.exists() && fs::read_dir(&params.output).map(|mut d| d.next().is_some()).unwrap_or(true) {
        return Err(format!("Refusing to mix this run with existing output: {}", params.output.display()));
    }
    // Resolution and baseline from the merge manifests.
    let mut resolutions = BTreeSet::new();
    let mut baselines = Vec::new();
    for c in &params.collections {
        let manifest = c.join("smarttile_merge.json");
        if !manifest.is_file() { baselines.push(None); continue; }
        let m: Value = serde_json::from_str(&fs::read_to_string(&manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if let Some(r) = m["resolution_1_m"].as_f64() { resolutions.insert(r.to_bits()); }
        baselines.push(m["baseline"].as_str().map(|b| c.join(b)).filter(|b| b.is_dir()));
    }
    if !params.prediction_resolution_original && resolutions.len() > 1 {
        return Err("Prediction collections declare conflicting first-stage resolutions".into());
    }
    let resolution = params.resolution_1.unwrap_or_else(|| resolutions.iter().next().map(|b| f64::from_bits(*b)).unwrap_or(0.01));
    if !(resolution.is_finite() && resolution > 0.0) {
        return Err("First subsampling resolution must be finite and positive".into());
    }
    if params.remap_tolerance.is_some_and(|r| !(r.is_finite() && r > 0.0)) {
        return Err("Original remap tolerance must be finite and positive".into());
    }
    // `--baseline-1cm-folders`: one shared baseline or one per model, instead of the manifests'.
    if let Some(folders) = &params.baseline_folders {
        if folders.len() != 1 && folders.len() != params.collections.len() {
            return Err("Provide one shared baseline or one baseline per model".into());
        }
        baselines = (0..params.collections.len()).map(|i| Some(folders[if folders.len() == 1 { 0 } else { i }].clone())).collect();
    }
    // `original_prediction_radius`: half the diagonal of each collection's coarsest XYZ steps plus an 8-ULP encoding guard.
    let radii: Vec<f64> = if params.prediction_resolution_original {
        params.collections.iter().map(|c| {
            let files = if c.is_file() { vec![c.clone()] } else { crate::tile::point_cloud_files(c)? };
            if files.is_empty() { return Err(format!("No prediction files in {}", c.display())); }
            let (mut scales, mut magnitude) = ([0f64; 3], 1f64);
            for file in &files {
                let h = lasio::read_header(file)?;
                if h.scales.iter().any(|s| !s.is_finite() || *s <= 0.0) {
                    return Err(format!("{}: original-resolution predictions need positive finite XYZ scales", file.display()));
                }
                for a in 0..3 { scales[a] = scales[a].max(h.scales[a]); magnitude = magnitude.max(h.mins[a].abs()).max(h.maxs[a].abs()); }
            }
            let half: Vec<f64> = scales.iter().map(|s| s / 2.0).collect();
            Ok(((half[0] * half[0] + half[1] * half[1]) + half[2] * half[2]).sqrt() + 8.0 * (f64::from_bits(magnitude.to_bits() + 1) - magnitude))
        }).collect::<Res<_>>()?
    } else {
        vec![params.remap_tolerance.unwrap_or(3f64.sqrt() * resolution); params.collections.len()]
    };
    let radius = radii.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let originals = original_files(&params.originals)?;
    let origin = lasio::read_header(&originals[0])?.offsets;
    let indexing_start = std::time::Instant::now();
    let mut collections = Vec::new();
    let mut baseline_sets = Vec::new();
    // Indexed parts live in the staging folder until the remap ends.
    let staging = params.output.parent().unwrap_or(Path::new(".")).join(format!(".strict-remap-{}", std::process::id()));
    for (i, (c, baseline)) in params.collections.iter().zip(&baselines).enumerate() {
        collections.push(load_collection(c, &origin, radius, &params.instance_dimension, &staging.join(format!("parts_{i}")))?);
        baseline_sets.push(match baseline {
            Some(b) => Some(load_baseline(b, &origin, radius, &params.instance_dimension, &staging.join(format!("baseline_{i}")))?),
            None => None,
        });
    }
    let indexing_seconds = indexing_start.elapsed().as_secs_f64();
    let matches: Vec<Option<String>> = if params.match_instances {
        collections.iter().map(|c| c.instance.clone().map(Some).ok_or_else(|| format!("{}: matching instances needs one PredInstance* dimension per collection", c.name))).collect::<Res<_>>()?
    } else {
        vec![None; collections.len()]
    };
    let (selected, used) = select_dimensions(&collections, params.target_dims.as_ref(), &matches)?;
    let parent = params.output.parent().unwrap_or(Path::new("."));
    let report_path = parent.join(format!("{}_coverage.json", params.output.file_name().unwrap().to_string_lossy()));
    let f = crate::pyfmt::float;
    let mut report = json!({
        "contract": "3DT-2101/v6-nearest-core", "state": "running", "models": [],
        "duplicate_radius_m": 0.01, "resolution_1_m": f(resolution), "original_radius_m": f(radius),
        "required_match_fraction": 1.0, "batch_point_limit": MAX_BATCH_POINTS,
        "distance_boundary": "inclusive Euclidean XYZ; 8 float64 ULP guard in local coordinates",
        "survivor_order": "source tile filename, then original point order",
        "nearest_ties": "within 8 local-coordinate float64 ULPs of the minimum distance; stable tile/point order",
        "original_coverage": [],
    });
    report["parallelism"] = json!({"requested_workers": params.requested_workers, "query_workers": params.workers,
        "scope": "bounded spatial queries; small batches run serially"});
    if params.prediction_resolution_original {
        report["resolution_1_m"] = Value::Null;
        report["original_radii_m"] = json!(radii.iter().map(|r| f(*r)).collect::<Vec<_>>());
        report["prediction_resolution"] = json!("original");
        report["original_radius_policy"] = json!("half diagonal of each prediction collection's coarsest LAS XYZ scales; 8 global-coordinate float64 ULP encoding guard");
    }
    report["timings"] = json!({"indexing_seconds": f(indexing_seconds)});
    let stage_dir = staging.join("originals");
    fs::create_dir_all(&stage_dir).map_err(|e| e.to_string())?;
    let result = enrich(&collections, &baseline_sets, &selected, &used, &originals, &origin, &radii, params.workers, &stage_dir, &mut report,
                        &matches, params.prediction_resolution_original);
    let outcome = result.and_then(|ids| {
        report["instance_finalization"] = compact(&stage_dir, &ids, &BTreeMap::new())?;
        let mut names: Vec<PathBuf> = fs::read_dir(&stage_dir).map_err(|e| e.to_string())?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        names.sort();
        let mut checksums = serde_json::Map::new();
        for path in names { checksums.insert(path.file_name().unwrap().to_string_lossy().into_owned(), json!(crate::sha256::file_hex(&path)?)); }
        report["checksums"] = Value::Object(checksums);
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        if params.output.is_dir() { fs::remove_dir(&params.output).map_err(|e| e.to_string())?; }
        fs::rename(&stage_dir, &params.output).map_err(|e| e.to_string())?;
        report["state"] = json!("validated");
        Ok(())
    });
    let _ = fs::remove_dir_all(&staging);
    if let Err(e) = &outcome {
        report["state"] = json!("failed");
        report["error"] = json!(e);
    }
    report["wall_seconds"] = f(started.elapsed().as_secs_f64());
    write_json(&report_path, &report, false)?;
    outcome?;
    println!("Strict original remap complete: {} ({:.1} s)", params.output.display(), started.elapsed().as_secs_f64());
    Ok(())
}

/// `original_files`: the raw LAS/LAZ originals; a COPC without its LAS/LAZ source fails.
pub fn original_files(directory: &Path) -> Res<Vec<PathBuf>> {
    let mut all: Vec<PathBuf> = fs::read_dir(directory).map_err(|e| format!("{}: {e}", directory.display()))?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && { let n = p.file_name().unwrap().to_string_lossy().to_lowercase(); n.ends_with(".laz") || n.ends_with(".las") })
        .collect();
    all.sort_by_key(|p| p.file_name().unwrap().to_string_lossy().to_lowercase());
    let is_copc = |p: &PathBuf| p.file_name().unwrap().to_string_lossy().to_lowercase().ends_with(".copc.laz");
    let raw_keys: BTreeSet<String> = all.iter().filter(|p| !is_copc(p)).map(|p| crate::collections::file_stem(p).to_lowercase()).collect();
    let orphans: Vec<String> = all.iter().filter(|p| is_copc(p) && !raw_keys.contains(&crate::collections::file_stem(p).to_lowercase()))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
    if !orphans.is_empty() {
        return Err(format!("COPC-only originals cannot be enriched; provide the LAS/LAZ source for: {}", orphans.join(", ")));
    }
    let raw: Vec<PathBuf> = all.into_iter().filter(|p| !is_copc(p)).collect();
    if raw.is_empty() {
        return Err(format!("No raw LAS/LAZ originals found in {}", directory.display()));
    }
    Ok(raw)
}

/// Selected dimensions per model; the instance becomes uint16/32 (`instance_extra_bytes_params`).
pub fn select_dimensions(collections: &[Collection], target_dims: Option<&BTreeSet<String>>, matches: &[Option<String>])
                         -> Res<(Vec<Vec<(usize, ExtraBytes)>>, BTreeSet<String>)> {
    let mut used = BTreeSet::new();
    let mut selected: Vec<Vec<(usize, ExtraBytes)>> = Vec::new();
    for (k, c) in collections.iter().enumerate() {
        let matched = matches.get(k).cloned().flatten();
        let dims: Vec<(usize, ExtraBytes)> = c.dimensions.iter().enumerate()
            .filter(|(_, d)| target_dims.is_none_or(|t| t.contains(&d.name)) && matched.as_deref() != Some(d.name.as_str())).map(|(i, d)| {
                let mut d = d.clone();
                if c.instance.as_deref() == Some(d.name.as_str()) {
                    d.data_type = if c.maximum > u16::MAX as u64 { Scalar::U32 } else { Scalar::U16 }.extra_bytes_id();
                    d.options = 0;
                    d.description = [0; 32];
                }
                (i, d)
            }).collect();
        if dims.is_empty() { return Err(format!("No selected prediction dimensions for model {}", c.name)); }
        for (_, d) in &dims {
            if !used.insert(d.name.clone()) { return Err(format!("Duplicate prediction dimension name across models: {}", d.name)); }
        }
        selected.push(dims);
    }
    Ok((selected, used))
}

/// `enrich_originals`: every original point takes the selected values of its nearest final
/// prediction (and is measured against the baseline geometry); enriched copies go to `stage_dir`.
/// Records parallelism, coverage, policy and timings in `report`; fails unless coverage is complete.
#[allow(clippy::too_many_arguments)]
pub fn enrich(collections: &[Collection], baseline_sets: &[Option<Parts>], selected: &[Vec<(usize, ExtraBytes)>], used: &BTreeSet<String>,
              originals: &[PathBuf], origin: &[f64; 3], radii: &[f64], workers: usize, stage_dir: &Path, report: &mut Value,
              matches: &[Option<String>], original_geometry: bool)
              -> Res<BTreeMap<String, BTreeMap<String, BTreeSet<u64>>>> {
    let radius = report["original_radius_m"].as_f64().unwrap_or(radii[0]);
    let baseline_stage: &'static str = if original_geometry { "unfiltered_original" } else { "unfiltered_1cm" };
    let f = crate::pyfmt::float;
    let source_points: u64 = originals.iter().map(|o| lasio::read_header(o).map(|h| h.point_count)).sum::<Res<u64>>()?;
    let processes = workers.min((source_points / MAX_BATCH_POINTS as u64).max(1) as usize);
    let parallelism = report["parallelism"].as_object_mut().ok_or("report without parallelism")?;
    parallelism.insert("enrichment_processes".into(), json!(processes));
    parallelism.insert("enrichment_query_workers".into(), json!(if processes > 1 { 1 } else { workers }));
    parallelism.insert("max_pending_batches".into(), json!(2 * processes));
    parallelism.insert("enrichment_scope".into(), json!("bounded process-parallel original remap; ordered writer"));
    report["original_coverage"] = json!([]);
    let mut policy = serde_json::Map::new();
    policy.insert(baseline_stage.into(), json!("100% required for every model with baseline geometry"));
    policy.insert("final_survivors".into(), json!(if original_geometry { "100% required for every model" }
                                                  else { "100% required except RCT; unmatched RCT predictions become background 0" }));
    report["original_coverage_policy"] = Value::Object(policy);
    // Matching compares the collection's instance label (each part's `labels`).
    for (c, m) in collections.iter().zip(matches) {
        if m.is_some() && m.as_ref() != c.instance.as_ref() { return Err(format!("{}: matching needs the instance dimension", c.name)); }
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers.max(1)).build().map_err(|e| e.to_string())?;
    let enrichment_start = std::time::Instant::now();
    let mut phases = [0.0f64; 3]; // spill, query, write
    let mut metrics: Vec<Coverage> = Vec::new();
    let result = (|| -> Res<BTreeMap<String, BTreeMap<String, BTreeSet<u64>>>> {
        let mut ids_by_dimension: BTreeMap<String, BTreeMap<String, BTreeSet<u64>>> = BTreeMap::new();
        for (file_number, file) in originals.iter().enumerate() {
            let file_name = file.file_name().unwrap().to_string_lossy().into_owned();
            let header = lasio::read_header(file)?;
            let layout = PointLayout::of(&header)?;
            if let Some(d) = used.iter().find(|d| layout.find(d).is_some()) {
                return Err(format!("{}: original already contains prediction dimensions [{d}]", file.display()));
            }
            // `copy_single_source_header` + `update_extra_dimensions`: new dimensions after the original's.
            let mut target = TileHeader::copy_of(&header)?;
            target.extra.extend(selected.iter().flatten().map(|(_, d)| OutputExtra::new(d)));
            target.sync_extra_bytes();
            let total = header.point_count as usize;
            let widths: Vec<usize> = selected.iter().flatten().map(|(_, d)| d.size()).collect();
            let width: usize = widths.iter().sum();
            let mut store = vec![0u8; total * width];
            let tick = std::time::Instant::now();
            // Spill files of a large original, beside the stage (`.enrich-*` as in Python).
            let scratch = stage_dir.parent().unwrap_or(stage_dir).join(format!(".enrich-{}-{file_number}", std::process::id()));
            let mut batches = query_batches(file, &header, origin, &scratch)?;
            phases[0] += tick.elapsed().as_secs_f64();
            let tick = std::time::Instant::now();
            let mut file_ids: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
            let mut finals: Vec<Coverage> = collections.iter().zip(radii).zip(matches).map(|((c, r), m)| {
                let mut metric = Coverage::new(&file_name, &c.name, "final_survivors", *r);
                metric.match_dimension = m.clone();
                metric
            }).collect();
            let mut bases: Vec<Option<Coverage>> = collections.iter().zip(baseline_sets).zip(radii)
                .map(|((c, b), r)| b.as_ref().map(|_| Coverage::new(&file_name, &c.name, baseline_stage, *r))).collect();
            // The originals' own labels of every matched dimension, by point.
            let own_labels: BTreeMap<String, Vec<u64>> = matches.iter().flatten().collect::<BTreeSet<_>>().into_iter()
                .map(|name| Ok((name.clone(), original_labels(file, &header, name)?))).collect::<Res<_>>()?;
            // Batches go in windows of `QUERY_WINDOW`, so only those are held at a time.
            let mut assigned = 0usize;
            loop {
                let batches: Vec<QueryBatch> = batches.by_ref().take(QUERY_WINDOW).collect::<Res<_>>()?;
                if batches.is_empty() { break; }
                assigned += batches.iter().map(|(i, _)| i.len()).sum::<usize>();
                let (lo, hi) = crate::parts::bounds(&batches.iter().flat_map(|(_, xyz)| xyz.iter().copied()).collect::<Vec<_>>());
                for (m, c) in collections.iter().enumerate() {
                    let radius = radii[m];
                    // The parts within reach of this window, indexed.
                    let reached = c.parts.reach(lo, hi, radius)?;
                    let sets: Vec<&crate::spatial::PointSet> = reached.iter().map(|l| &l.set).collect();
                    let base_reached = match &baseline_sets[m] { Some(b) => Some(b.reach(lo, hi, radius)?), None => None };
                    let base_sets: Option<Vec<&crate::spatial::PointSet>> = base_reached.as_ref().map(|r| r.iter().map(|l| &l.set).collect());
                    let results: Vec<(Vec<Option<(usize, u32, f64)>>, Option<Vec<f64>>)> = pool.install(|| batches.par_iter().map(|(indices, xyz)| {
                        let base = base_sets.as_ref().map(|b| crate::spatial::nearest_distances_across(b, xyz, radius));
                        let found = match &matches[m] {
                            // `nearest_matching`: per label, the nearest among points of the same label.
                            Some(name) => {
                                let ours = &own_labels[name];
                                let mut by_label: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
                                for (k, &i) in indices.iter().enumerate() { by_label.entry(ours[i as usize]).or_default().push(k); }
                                let mut found = vec![None; xyz.len()];
                                for (label, rows) in by_label {
                                    let pts: Vec<[f64; 3]> = rows.iter().map(|&k| xyz[k]).collect();
                                    let same = |s: usize, id: u32, _: &[f64; 3]| reached[s].labels[id as usize] == label;
                                    for (k, hit) in rows.iter().zip(crate::spatial::nearest_across(&sets, &pts, radius, &same)) { found[*k] = hit; }
                                }
                                found
                            }
                            None => crate::spatial::nearest_across(&sets, xyz, radius, &|_, _, _| true),
                        };
                        (found, base)
                    }).collect());
                    // Byte offset of this model's selected dimensions in a stored row.
                    let start: usize = selected[..m].iter().flatten().map(|(_, d)| d.size()).sum();
                    let instance_pos = selected[m].iter().position(|(_, d)| c.instance.as_deref() == Some(d.name.as_str()));
                    let mut source_offsets = Vec::new();
                    let mut o = 0;
                    for d in &c.dimensions { source_offsets.push(o); o += d.size(); }
                    for ((indices, xyz), (found, base)) in batches.iter().zip(&results) {
                        if let (Some(metric), Some(base)) = (bases[m].as_mut(), base) { metric.record(xyz, base, origin); }
                        let distances: Vec<f64> = found.iter().map(|x| x.map_or(f64::INFINITY, |(_, _, d)| d)).collect();
                        finals[m].record(xyz, &distances, origin);
                        for (k, &i) in indices.iter().enumerate() {
                            let Some((s, id, _)) = found[k] else { continue };
                            let row = &reached[s].values[id as usize * c.width..(id as usize + 1) * c.width];
                            let mut at = start;
                            for (j, (src, d)) in selected[m].iter().enumerate() {
                                let raw = &row[source_offsets[*src]..source_offsets[*src] + c.dimensions[*src].size()];
                                let dst = &mut store[i as usize * width + at..i as usize * width + at + d.size()];
                                if Some(j) == instance_pos {
                                    let v = c.dimensions[*src].scalar().unwrap().read_int(raw) as u64;
                                    dst.copy_from_slice(&v.to_le_bytes()[..d.size()]);
                                    if v > 0 { file_ids.entry(d.name.clone()).or_default().insert(v); }
                                } else {
                                    dst.copy_from_slice(raw);
                                }
                                at += d.size();
                            }
                        }
                    }
                }
            }
            drop(batches);
            if scratch.exists() { fs::remove_dir_all(&scratch).map_err(|e| format!("{}: {e}", scratch.display()))?; }
            if assigned != total {
                return Err(format!("{file_name}: enrichment queried {} of {} points", crate::pyfmt::thousands(assigned as u64), crate::pyfmt::thousands(total as u64)));
            }
            metrics.extend(bases.into_iter().flatten());
            metrics.extend(finals);
            phases[1] += tick.elapsed().as_secs_f64();
            let tick = std::time::Instant::now();
            // Enriched original in its own point order, 1,000,000 points per write.
            let base_len = header.record_length as usize;
            let mut position = 0usize;
            let batches = chunks(file, ORIGINAL_WRITE_POINTS)?.map(|chunk| chunk.map(|chunk| {
                let mut out = Vec::with_capacity(chunk.len() * (base_len + width));
                for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
                    out.extend_from_slice(&r[..base_len]);
                    out.extend_from_slice(&store[position * width..(position + 1) * width]);
                    position += 1;
                }
                out
            }));
            lasio::write_records(&stage_dir.join(file.file_name().unwrap()), &target, batches, true, Some(&pool))?;
            phases[2] += tick.elapsed().as_secs_f64();
            for (name, ids) in file_ids { ids_by_dimension.entry(name).or_default().insert(file_name.clone(), ids); }
        }
        Ok(ids_by_dimension)
    })();
    for c in collections { c.parts.release(); }
    for b in baseline_sets.iter().flatten() { b.release(); }
    // A failed file leaves its spill behind.
    for n in 0..originals.len() {
        let _ = fs::remove_dir_all(stage_dir.parent().unwrap_or(stage_dir).join(format!(".enrich-{}-{n}", std::process::id())));
    }
    report["original_coverage"] = Value::Array(metrics.iter().map(Coverage::json).collect());
    let timings = report.as_object_mut().unwrap().entry("timings").or_insert_with(|| json!({}));
    timings["enrichment_seconds"] = f(enrichment_start.elapsed().as_secs_f64());
    timings["enrichment_phase_seconds"] = json!({"spill": f(phases[0]), "query": f(phases[1]), "write": f(phases[2])});
    let ids = result?;
    report["background_assigned_points"] = json!(0);
    let _ = radius;
    let failures: Vec<&Coverage> = metrics.iter().filter(|m| m.matched != m.total).collect();
    if !failures.is_empty() {
        report["coverage_failures"] = Value::Array(failures.iter().map(|m| json!({"file": m.file, "model": m.model, "stage": m.stage,
                                                                                "missing": m.total - m.matched})).collect());
        return Err(format!("100% original coverage within {} m is required; {}", g8(report["original_radius_m"].as_f64().unwrap_or(radius)),
                           failures.iter().map(|m| format!("{}/{}/{}: {}/{}", m.file, m.model, m.stage, m.matched, m.total)).collect::<Vec<_>>().join("; ")));
    }
    Ok(ids)
}

/// Python's `f"{x:.8g}"` for a positive finite number.
fn g8(x: f64) -> String {
    let exponent = x.abs().log10().floor() as i32;
    if !(-5..8).contains(&exponent) { return format!("{x:.7e}"); }
    let text = format!("{:.*}", (7 - exponent).max(0) as usize, x);
    if text.contains('.') { text.trim_end_matches('0').trim_end_matches('.').to_string() } else { text }
}

/// `_coverage_metric` / `_record_coverage`: one original, model and stage.
struct Coverage { file: String, model: String, stage: &'static str, radius: f64, total: u64, matched: u64, max: f64, histogram: [u64; 5], missing: Vec<[f64; 3]>,
                  match_dimension: Option<String> }

impl Coverage {
    fn new(file: &str, model: &str, stage: &'static str, radius: f64) -> Coverage {
        Coverage { file: file.into(), model: model.into(), stage, radius, total: 0, matched: 0, max: 0.0, histogram: [0; 5], missing: Vec::new(), match_dimension: None }
    }

    fn bounds(&self) -> [f64; 4] { [0.25, 0.5, 0.75, 1.0].map(|q| self.radius * q) }

    fn record(&mut self, xyz: &[[f64; 3]], distances: &[f64], origin: &[f64; 3]) {
        let bounds = self.bounds();
        self.total += xyz.len() as u64;
        for (p, &d) in xyz.iter().zip(distances) {
            if d.is_finite() {
                self.matched += 1;
                self.max = self.max.max(d);
                // np.searchsorted(side="left"): the number of upper bounds below d.
                self.histogram[bounds.iter().filter(|b| **b < d).count()] += 1;
            } else if self.missing.len() < 5 {
                self.missing.push(std::array::from_fn(|a| p[a] + origin[a]));
            }
        }
    }

    fn json(&self) -> Value {
        let f = crate::pyfmt::float;
        let mut value = json!({"file": self.file, "model": self.model, "stage": self.stage, "radius_m": f(self.radius), "total": self.total,
               "matched": self.matched, "max_matched_distance_m": f(self.max), "distance_histogram": self.histogram,
               "histogram_upper_bounds_m": self.bounds().map(f),
               "missing_examples": self.missing.iter().map(|p| p.map(f).to_vec()).collect::<Vec<_>>()});
        if let Some(name) = &self.match_dimension { value["match_instance_dimension"] = json!(name); }
        value
    }
}

/// The original's own values of `name`, by point (`spill_original` label stores).
fn original_labels(file: &Path, header: &SourceHeader, name: &str) -> Res<Vec<u64>> {
    let layout = PointLayout::of(header)?;
    let (field, _) = layout.find(name).ok_or_else(|| format!("{}: original lacks the matched instance dimension {name}", file.display()))?;
    let mut out = Vec::with_capacity(header.point_count as usize);
    let mut scratch = [0u8; 8];
    for chunk in chunks(file, ORIGINAL_WRITE_POINTS)? {
        let chunk = chunk?;
        for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) { out.push(field.scalar().read_int(field.get(r, &mut scratch)) as u64); }
    }
    Ok(out)
}

/// `compact_originals`: dataset-wide IDs 1..N per instance dimension, uint32, plus the mapping file.
/// `aliases` ({dimension: {source: target}}, small-instance reassignment) give a source its target's final ID.
pub fn compact(dir: &Path, ids_by_dimension: &BTreeMap<String, BTreeMap<String, BTreeSet<u64>>>,
               aliases: &BTreeMap<String, BTreeMap<u64, u64>>) -> Res<Value> {
    if ids_by_dimension.is_empty() {
        return Ok(json!({"mapping": null, "counts": {}}));
    }
    let mut mappings: BTreeMap<String, BTreeMap<u64, u32>> = BTreeMap::new();
    let mut models = serde_json::Map::new();
    let mut files: BTreeSet<String> = BTreeSet::new();
    for (name, by_file) in ids_by_dimension {
        let sources: BTreeSet<u64> = by_file.values().flatten().copied().filter(|v| *v > 0).collect();
        let empty = BTreeMap::new();
        let alias = aliases.get(name).unwrap_or(&empty);
        let finals: BTreeSet<u64> = sources.iter().map(|s| *alias.get(s).unwrap_or(s)).collect();
        let rank: BTreeMap<u64, u32> = finals.iter().enumerate().map(|(i, s)| (*s, i as u32 + 1)).collect();
        let mapping: BTreeMap<u64, u32> = sources.iter().map(|s| (*s, rank[alias.get(s).unwrap_or(s)])).collect();
        models.insert(name.clone(), json!({"scope": "dataset", "instance_count": finals.len(),
            "instances": sources.iter().map(|s| {
                let mut entry = json!({"source_instance_id": s, "instance_id": mapping[s]});
                if let Some(t) = alias.get(s) { entry["reassigned_to"] = json!(t); }
                entry
            }).collect::<Vec<_>>()}));
        files.extend(by_file.keys().cloned());
        mappings.insert(name.clone(), mapping);
    }
    for name in &files {
        let file = dir.join(name);
        let header = lasio::read_header(&file)?;
        let layout = PointLayout::of(&header)?;
        let mut target = TileHeader::copy_of(&header)?;
        // `update_extra_dimensions(replace=True)`: the instance dimensions leave and return as uint32 at the end.
        let replaced: Vec<&String> = mappings.keys().collect();
        let fields: Vec<(Layout, Layout)> = Vec::new();
        let _ = fields;
        let kept: Vec<OutputExtra> = target.extra.iter().filter(|e| !replaced.contains(&&e.source.name)).cloned().collect();
        target.extra = kept;
        for name in &replaced {
            let mut d = layout.extra.iter().find(|(e, _)| &e.name == *name).unwrap().0.clone();
            d.data_type = Scalar::U32.extra_bytes_id();
            d.options = 0;
            let text = b"Dataset-wide compact instance ID";
            d.description = [0; 32];
            d.description[..text.len()].copy_from_slice(text);
            target.extra.push(OutputExtra::new(&d));
        }
        target.sync_extra_bytes();
        target.vlrs.retain(|v| !(v.user() == "3DTrees" && v.record_id == 24004));
        let payload = format!("{{\"version\": 1, \"scope\": \"dataset_per_model\", \"background\": 0, \"dimensions\": [{}]}}",
                              mappings.keys().map(|k| format!("\"{k}\"")).collect::<Vec<_>>().join(", "));
        target.vlrs.push(Vlr::new("3DTrees", 24004, "Compact instance IDs v1", payload.into_bytes()));
        // Output record: the source record without the replaced fields, then the new fields.
        let base = crate::points::base_record_length(header.point_format);
        let mut copies: Vec<(usize, usize)> = Vec::new(); // (source offset, size) of kept extras, in order
        for (e, offset) in &layout.extra {
            if !replaced.contains(&&e.name) { copies.push((*offset, e.size())); }
        }
        let instance_fields: Vec<(Layout, &BTreeMap<u64, u32>)> = replaced.iter().map(|n| (layout.find(n).unwrap().0, &mappings[*n])).collect();
        let tmp = dir.join(format!(".compact-{name}"));
        let batches = chunks(&file, MAX_BATCH_POINTS as u64)?.map(|chunk| chunk.and_then(|chunk| {
            let mut out = Vec::new();
            let mut scratch = [0u8; 8];
            for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
                out.extend_from_slice(&r[..base]);
                for (offset, size) in &copies { out.extend_from_slice(&r[*offset..offset + size]); }
                for (f, mapping) in &instance_fields {
                    let v = f.scalar().read_int(f.get(r, &mut scratch)) as u64;
                    let compact = if v == 0 { 0 } else { *mapping.get(&v).ok_or(format!("{name}: IDs changed after remapping"))? };
                    out.extend_from_slice(&compact.to_le_bytes());
                }
            }
            Ok(out)
        }));
        lasio::write_records(&tmp, &target, batches, true, None)?;
        fs::rename(&tmp, &file).map_err(|e| e.to_string())?;
    }
    write_json(&dir.join("instance_mapping.json"), &json!({"version": 1, "background": 0, "models": Value::Object(models)}), true)?;
    Ok(json!({"mapping": "instance_mapping.json",
              "counts": mappings.iter().map(|(k, m)| (k.clone(), json!(m.values().max().copied().unwrap_or(0)))).collect::<serde_json::Map<_, _>>()}))
}

#[cfg(test)]
mod tests {
    use super::g8;

    #[test]
    fn eight_significant_digits_like_python() {
        assert_eq!(g8(0.017320508075688773), "0.017320508");
        assert_eq!(g8(0.17320508075688773), "0.17320508");
        assert_eq!(g8(0.5), "0.5");
        assert_eq!(g8(2.0), "2");
    }
}
