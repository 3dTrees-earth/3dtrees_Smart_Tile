//! The `merge` task (`remap_first_pipeline.merge_collections`), ported stage by stage.
//!
//! Stage 1 (`prepare_dense`): every 1 cm target point of a tile takes the
//! predictions of its nearest point in the tile's own prediction file
//! (within the transfer radius; every point must match). The dense tiles are
//! the published `output_tiles_unfiltered_1cm` baseline.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde_json::{Map, Value};

use crate::lasio::{self, Res, SourceHeader, TileHeader};
use crate::ld::local_coordinate;
use crate::points::{ExtraBytes, Layout, OutputExtra, PointLayout};
use crate::decimal::py_round;
use crate::spatial::PointSet;

pub const MAX_BATCH_POINTS: u64 = 32_768;

pub struct MergeParams {
    /// One or more prediction collections (`--segmented-folders` or the single segmented folder).
    pub collections: Vec<PathBuf>,
    /// The 1 cm targets; None for ready predictions (`--segmented-remapped-folder`, or `--segmented-folders` alone).
    pub target: Option<PathBuf>,
    /// `--resolution-1` (first-stage resolution) and `--remap-tolerance` (original radius override).
    pub resolution_1: f64,
    pub remap_tolerance: Option<f64>,
    pub tile_bounds_json: PathBuf,
    pub work: PathBuf,
    pub workers: usize,
    pub transfer_radius: f64,
    pub instance_dimension: String,
    pub overlap_threshold: f64,
    /// `--workers` as given (the report records it beside the capped worker count).
    pub requested_workers: usize,
    /// Raw originals to enrich in the same run, and where the enriched copies go.
    pub originals: Option<PathBuf>,
    pub original_output: Option<PathBuf>,
    /// `--skip-merged-file`: no `merged_tiles.laz` (never written for several collections).
    pub skip_merged_file: bool,
    /// `--output-tiles-folder` (default `<work>/output_tiles`) and `--output-merged-laz`.
    pub output_tiles: PathBuf,
    pub output_merged_laz: Option<PathBuf>,
    /// `--disable-matching`, `--filter-anchor`, `--skip-unfiltered-tiles`, `--remap-dims`.
    pub matching: bool,
    pub anchor: Anchor,
    pub keep_unfiltered: bool,
    pub target_dims: Option<std::collections::BTreeSet<String>>,
    /// `--reassign-small-instances`: (max cluster size, max bounding-box volume in m³).
    pub small_instances: Option<(u64, f64)>,
    /// Development: stop after this stage (only "dense" is ported).
    pub stop_after: Option<String>,
}

/// `describe_model`: the prediction dimensions, instance and semantic of a collection.
pub struct Model { pub name: String, pub instance: String, pub semantic: Option<String>, pub dimensions: Vec<ExtraBytes> }

fn describe_model(files: &[PathBuf], headers: &[SourceHeader], source: &Path, instance_dimension: &str) -> Res<Model> {
    let standard = crate::points::all_standard_names();
    let layout = PointLayout::of(&headers[0])?;
    let names: Vec<String> = layout.extra.iter().map(|(e, _)| e.name.clone()).collect();
    let mut instances: Vec<String> = names.iter().filter(|n| n.starts_with("PredInstance")).cloned().collect();
    if names.iter().any(|n| n == instance_dimension) && !instances.iter().any(|n| n == instance_dimension) {
        instances.push(instance_dimension.to_string());
    }
    if instances.is_empty() && names.iter().any(|n| n == "treeID") {
        instances.push("treeID".into());
    }
    if instances.len() != 1 {
        return Err(format!("{}: expected one model instance dimension, got {instances:?}; supply each model as a separate collection", source.display()));
    }
    let instance = instances.remove(0);
    let semantic = Some(instance.replacen("PredInstance", "PredSemantic", 1)).filter(|s| *s != instance && names.contains(s));
    let dimensions: Vec<ExtraBytes> = layout.extra.iter().filter(|(e, _)| !standard.contains(&e.name)).map(|(e, _)| e.clone()).collect();
    for (file, header) in files.iter().zip(headers) {
        let actual: Vec<ExtraBytes> = PointLayout::of(header)?.extra.into_iter().filter(|(e, _)| !standard.contains(&e.name)).map(|(e, _)| e).collect();
        let same = actual.len() == dimensions.len() && actual.iter().zip(&dimensions).all(|(a, b)| {
            a.name == b.name && a.data_type == b.data_type && a.has_no_data() == b.has_no_data()
                && a.has_scale() == b.has_scale() && a.has_offset() == b.has_offset() && a.scale[0] == b.scale[0] && a.offset[0] == b.offset[0]
        });
        if !same {
            return Err(format!("{}: inconsistent prediction schema in {} (type promotion is not ported yet)", source.display(), file.display()));
        }
    }
    for name in std::iter::once(&instance).chain(semantic.as_ref()) {
        let eb = dimensions.iter().find(|e| &e.name == name).unwrap();
        if (eb.has_scale() && eb.scale[0] != 1.0) || (eb.has_offset() && eb.offset[0] != 0.0) {
            return Err(format!("{}: scaled prediction label {name} needs explicit normalization", source.display()));
        }
    }
    Ok(Model { name: source.file_name().unwrap().to_string_lossy().into_owned(), instance, semantic, dimensions })
}

/// `match_tiles_to_json_bounds`: stepwise bounds, then centre tolerances.
fn match_to_layout(tiles: &[(String, [f64; 4])], json_bounds: &[[f64; 4]], centers: &[[f64; 2]]) -> Res<BTreeMap<String, usize>> {
    let mut matched: BTreeMap<String, usize> = BTreeMap::new();
    let mut used = std::collections::BTreeSet::new();
    if tiles.len() == 1 && json_bounds.len() == 1 {
        matched.insert(tiles[0].0.clone(), 0);
        return Ok(matched);
    }
    for tol in [0.1, 0.5, 1.0, 2.0, 5.0] {
        for (name, b) in tiles {
            if matched.contains_key(name) { continue; }
            let mut best: Option<(f64, usize)> = None;
            for (j, j_b) in json_bounds.iter().enumerate() {
                if used.contains(&j) { continue; }
                let diffs = [(b[0] - j_b[0]).abs(), (b[1] - j_b[1]).abs(), (b[2] - j_b[2]).abs(), (b[3] - j_b[3]).abs()];
                if diffs.iter().all(|d| *d <= tol) {
                    let l1 = diffs[0] + diffs[1] + diffs[2] + diffs[3];
                    if best.is_none_or(|(x, _)| l1 < x) { best = Some((l1, j)); }
                }
            }
            if let Some((_, j)) = best { matched.insert(name.clone(), j); used.insert(j); }
        }
        for (name, b) in tiles {
            if matched.contains_key(name) { continue; }
            let (cx, cy) = ((b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5);
            let mut best: Option<(f64, usize)> = None;
            for (j, c) in centers.iter().enumerate() {
                if used.contains(&j) { continue; }
                let d = (c[0] - cx).hypot(c[1] - cy);
                if d <= tol && best.is_none_or(|(x, _)| d < x) { best = Some((d, j)); }
            }
            if let Some((_, j)) = best { matched.insert(name.clone(), j); used.insert(j); }
        }
        if matched.len() == tiles.len() { break; }
    }
    if matched.len() != tiles.len() {
        let missing: Vec<&String> = tiles.iter().map(|t| &t.0).filter(|n| !matched.contains_key(*n)).collect();
        return Err(format!("Failed to match all tiles to entries in tile_bounds_tindex.json. Unmatched tiles: {missing:?}"));
    }
    Ok(matched)
}

/// (minx, maxx, miny, maxy) of a file header.
fn file_bounds(h: &SourceHeader) -> [f64; 4] {
    [h.mins[0], h.maxs[0], h.mins[1], h.maxs[1]]
}

struct Pair { prediction: PathBuf, target: PathBuf, layout_tile: Option<usize>, single: Option<Region> }

/// `ownership_regions`: the declared core and the declared neighbours of a tile.
#[derive(Clone, Debug)]
pub struct Region { pub core: [[f64; 2]; 2], pub west: bool, pub east: bool, pub south: bool, pub north: bool,
                    /// Layout tile index and neighbouring layout tiles (east, west, north, south), as reported.
                    pub layout_tile: Option<usize>, pub neighbors: [Option<usize>; 4],
                    /// `layout_recovery` of a legacy single-cloud layout.
                    pub recovery: Option<Value> }

impl Region {
    /// `dict(region)`: layout_tile, core and neighbors, in the report's key order.
    fn json(&self) -> Map<String, Value> {
        let f = crate::pyfmt::float;
        let n = |v: Option<usize>| v.map_or(Value::Null, |i| serde_json::json!(i));
        let mut m = Map::new();
        m.insert("layout_tile".into(), serde_json::json!(self.layout_tile));
        m.insert("core".into(), serde_json::json!([[f(self.core[0][0]), f(self.core[0][1])], [f(self.core[1][0]), f(self.core[1][1])]]));
        m.insert("neighbors".into(), serde_json::json!({"east": n(self.neighbors[0]), "west": n(self.neighbors[1]),
                                                         "north": n(self.neighbors[2]), "south": n(self.neighbors[3])}));
        if let Some(recovery) = &self.recovery { m.insert("layout_recovery".into(), recovery.clone()); }
        m
    }
}

/// `is_legacy_single_cloud_layout`: an unused historical grid around one verified whole cloud.
fn is_legacy_single_cloud(layout: &Value, b: [f64; 4]) -> bool {
    let Some(tiles) = layout["tiles"].as_array() else { return false };
    if tiles.len() < 2 || layout.get("tiling_skipped") == Some(&Value::Bool(false)) { return false; }
    let tolerance = 0.01 + 1e-9;
    let e = &layout["proj_extent"];
    let Some(expected) = ["minx", "maxx", "miny", "maxy"].iter().map(|k| e[*k].as_f64()).collect::<Option<Vec<f64>>>() else { return false };
    if !b.iter().chain(&expected).all(|v| v.is_finite()) || b.iter().zip(&expected).any(|(a, x)| (a - x).abs() > tolerance) { return false; }
    let [x0, x1, y0, y1] = b;
    if x0 >= x1 || y0 >= y1 { return false; }
    tiles.iter().all(|t| {
        if t.get("bounds").is_none() || t["bounds"] != t["planned_bounds"] { return false; }
        let v = |i: usize, j: usize| t["bounds"][i][j].as_f64().unwrap_or(f64::NAN);
        let (a0, a1, b0, b1) = (v(0, 0), v(0, 1), v(1, 0), v(1, 1));
        [a0, a1, b0, b1].iter().all(|v| v.is_finite()) && a0 < a1 && b0 < b1
            && !(x0 >= a0 - tolerance && x1 <= a1 + tolerance && y0 >= b0 - tolerance && y1 <= b1 + tolerance)
    })
}

/// `ownership_regions` for one cloud without a grid tile: a legacy single-cloud layout, or a
/// layout without tiles (the single-file bypass). None when the cloud belongs to a grid tile.
fn single_cloud_region(target: &Path, layout: &Value) -> Res<Option<Region>> {
    let [x0, x1, y0, y1] = file_bounds(&lasio::read_header(target)?);
    let tiles = layout["tiles"].as_array().map(Vec::len).unwrap_or(0);
    let recovery = if is_legacy_single_cloud(layout, [x0, x1, y0, y1]) {
        println!("Recovered legacy single-cloud layout for {}: full projected extent verified; ignoring {tiles} unused planned tiles",
                 target.file_name().unwrap().to_string_lossy());
        Some(serde_json::json!({"reason": "legacy_single_cloud_bypass", "planned_tile_count": tiles, "extent_tolerance_m": 0.01}))
    } else if tiles == 0 {
        None
    } else {
        return Ok(None);
    };
    Ok(Some(Region { core: [[x0, x1], [y0, y1]], west: false, east: false, south: false, north: false, layout_tile: None,
                     neighbors: [None; 4], recovery }))
}

fn regions(pairs: &[Pair], layout: &Value) -> Res<Vec<Region>> {
    let empty = Vec::new();
    let tiles = layout["tiles"].as_array().unwrap_or(&empty);
    let at: BTreeMap<(i64, i64), usize> = tiles.iter().enumerate()
        .filter_map(|(i, t)| Some(((t["col"].as_i64()?, t["row"].as_i64()?), i))).collect();
    pairs.iter().map(|p| {
        if let Some(region) = &p.single { return Ok(region.clone()); }
        let index = p.layout_tile.ok_or("pair without a layout tile")?;
        let t = &tiles[index];
        let (Some(c), Some(r)) = (t["col"].as_i64(), t["row"].as_i64()) else {
            return Err("layouts without tile col/row are not ported to Rust yet".to_string());
        };
        let f = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
        let neighbors = [at.get(&(c + 1, r)).copied(), at.get(&(c - 1, r)).copied(), at.get(&(c, r + 1)).copied(), at.get(&(c, r - 1)).copied()];
        let core = if t.get("core").is_some_and(|c| !c.is_null()) {
            [[f(&t["core"][0][0]), f(&t["core"][0][1])], [f(&t["core"][1][0]), f(&t["core"][1][1])]]
        } else {
            // Older layouts carry a buffer width instead of explicit cores.
            let buffer = match layout.get("tile_buffer").and_then(Value::as_f64) {
                Some(b) => b,
                None if neighbors.iter().any(Option::is_some) => return Err("Instance ownership requires core bounds or tile_buffer".into()),
                None => 0.0,
            };
            if !buffer.is_finite() || buffer < 0.0 { return Err("Instance ownership requires a finite nonnegative tile_buffer".into()); }
            let (x0, x1, y0, y1) = (f(&t["bounds"][0][0]), f(&t["bounds"][0][1]), f(&t["bounds"][1][0]), f(&t["bounds"][1][1]));
            let pad = |n: Option<usize>| if n.is_some() { buffer } else { 0.0 };
            [[x0 + pad(neighbors[1]), x1 - pad(neighbors[0])], [y0 + pad(neighbors[3]), y1 - pad(neighbors[2])]]
        };
        if core.iter().flatten().any(|v| !v.is_finite()) || core[0][0] > core[0][1] || core[1][0] > core[1][1] {
            return Err(format!("Invalid core bounds for {}", p.target.file_name().unwrap().to_string_lossy()));
        }
        Ok(Region { core, west: neighbors[1].is_some(), east: neighbors[0].is_some(), south: neighbors[3].is_some(), north: neighbors[2].is_some(),
                    layout_tile: Some(index), neighbors, recovery: None })
    }).collect()
}

/// `owned_anchor`: inclusive core edges towards declared neighbours, unbounded outer edges.
fn owned_anchor(x: f64, y: f64, r: &Region) -> bool {
    !((r.west && x < r.core[0][0]) || (r.east && x > r.core[0][1]) || (r.south && y < r.core[1][0]) || (r.north && y > r.core[1][1]))
}

/// `owned_background`: half-open cores (east/north neighbours own shared edges).
fn owned_background(x: f64, y: f64, r: &Region) -> bool {
    (!r.west || x >= r.core[0][0]) && (!r.east || x < r.core[0][1]) && (!r.south || y >= r.core[1][0]) && (!r.north || y < r.core[1][1])
}

/// `find_matching_files` + `_tile_labels`: prediction/target pairs by shared layout tile, sorted by prediction path.
fn pairs(segmented: &Path, target: &Path, layout: &Value) -> Res<(Vec<Pair>, Vec<String>)> {
    let sources = crate::tile::point_cloud_files(segmented)?;
    let targets = crate::tile::point_cloud_files(target)?;
    let tiles = layout["tiles"].as_array().cloned().unwrap_or_default();
    let json_bounds: Vec<[f64; 4]> = tiles.iter().map(|t| [t["bounds"][0][0].as_f64().unwrap(), t["bounds"][0][1].as_f64().unwrap(),
                                                         t["bounds"][1][0].as_f64().unwrap(), t["bounds"][1][1].as_f64().unwrap()]).collect();
    let centers: Vec<[f64; 2]> = json_bounds.iter().map(|b| [(b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5]).collect();
    let bounds_of = |files: &[PathBuf]| -> Res<Vec<(String, [f64; 4])>> {
        files.iter().map(|f| Ok((crate::collections::file_stem(f), file_bounds(&lasio::read_header(f)?)))).collect()
    };
    let (source_bounds, target_bounds) = (bounds_of(&sources)?, bounds_of(&targets)?);
    let mut out = Vec::new();
    if sources.len() == 1 && targets.len() == 1 {
        out.push(Pair { prediction: sources[0].clone(), target: targets[0].clone(), layout_tile: None, single: None });
    } else {
        let s = match_to_layout(&source_bounds, &json_bounds, &centers)?;
        let t = match_to_layout(&target_bounds, &json_bounds, &centers)?;
        let by_json_s: BTreeMap<usize, &String> = s.iter().map(|(k, v)| (*v, k)).collect();
        let by_json_t: BTreeMap<usize, &String> = t.iter().map(|(k, v)| (*v, k)).collect();
        for j in 0..json_bounds.len() {
            match (by_json_s.get(&j), by_json_t.get(&j)) {
                (Some(stem), Some(tstem)) => {
                    let prediction = sources.iter().find(|f| crate::collections::file_stem(f) == **stem).unwrap().clone();
                    let target = targets.iter().find(|f| crate::collections::file_stem(f) == **tstem).unwrap().clone();
                    out.push(Pair { prediction, target, layout_tile: None, single: None });
                }
                (Some(stem), None) => return Err(format!("Source file {stem} matched JSON tile index {j} but no target file matched that tile. Cannot remap.")),
                _ => {}
            }
        }
    }
    if out.len() != targets.len() {
        return Err("every 1 cm target tile must have exactly one prediction tile".into());
    }
    out.sort_by(|a, b| a.prediction.cmp(&b.prediction));
    // `ownership_regions`: each target's layout tile, by its file bounds.
    if out.len() == 1 {
        if let Some(region) = single_cloud_region(&out[0].target, layout)? {
            out[0].single = Some(region);
            return Ok((out, vec!["c00_r00".into()]));
        }
    }
    let target_map = match_to_layout(&out.iter().map(|p| (p.target.display().to_string(), file_bounds(&lasio::read_header(&p.target).unwrap()))).collect::<Vec<_>>(), &json_bounds, &centers)?;
    let mut labels = Vec::new();
    for p in &mut out {
        p.layout_tile = Some(target_map[&p.target.display().to_string()]);
        let t = &tiles[p.layout_tile.unwrap()];
        labels.push(format!("c{:02}_r{:02}", t["col"].as_i64().unwrap(), t["row"].as_i64().unwrap()));
    }
    Ok((out, labels))
}

/// Ready predictions: each tile is its own target (`(p, p, stem)`), sorted by path, on its layout tile.
fn ready_pairs(segmented: &Path, layout: &Value) -> Res<(Vec<Pair>, Vec<String>)> {
    let mut sources = crate::tile::point_cloud_files(segmented)?;
    sources.sort();
    let tiles = layout["tiles"].as_array().cloned().unwrap_or_default();
    let json_bounds: Vec<[f64; 4]> = tiles.iter().map(|t| [t["bounds"][0][0].as_f64().unwrap(), t["bounds"][0][1].as_f64().unwrap(),
                                                         t["bounds"][1][0].as_f64().unwrap(), t["bounds"][1][1].as_f64().unwrap()]).collect();
    let centers: Vec<[f64; 2]> = json_bounds.iter().map(|b| [(b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5]).collect();
    let bounds: Vec<(String, [f64; 4])> = sources.iter().map(|f| Ok((f.display().to_string(), file_bounds(&lasio::read_header(f)?)))).collect::<Res<_>>()?;
    if sources.len() == 1 {
        if let Some(region) = single_cloud_region(&sources[0], layout)? {
            return Ok((vec![Pair { prediction: sources[0].clone(), target: sources[0].clone(), layout_tile: None, single: Some(region) }], vec!["c00_r00".into()]));
        }
    }
    let map = match_to_layout(&bounds, &json_bounds, &centers)?;
    let mut out = Vec::new();
    let mut labels = Vec::new();
    for f in sources {
        let index = map[&f.display().to_string()];
        let t = &tiles[index];
        labels.push(format!("c{:02}_r{:02}", t["col"].as_i64().unwrap(), t["row"].as_i64().unwrap()));
        out.push(Pair { prediction: f.clone(), target: f, layout_tile: Some(index), single: None });
    }
    Ok((out, labels))
}

/// `prepare_dense` with ready predictions: validate the labels and copy each tile with the model
/// dimensions re-declared at the end (`target_header(ready=True)`); every point is assigned.
fn ready_tile(model: &Model, n: usize, pair: &Pair, output: &Path) -> Res<(u64, u64)> {
    let header = lasio::read_header(&pair.prediction)?;
    let layout = PointLayout::of(&header)?;
    let mut target = TileHeader::copy_of(&header)?;
    let names: Vec<&str> = model.dimensions.iter().map(|d| d.name.as_str()).collect();
    target.extra.retain(|e| !names.contains(&e.source.name.as_str()));
    let kept: Vec<(usize, usize)> = layout.extra.iter().filter(|(e, _)| !names.contains(&e.name.as_str())).map(|(e, o)| (*o, e.size())).collect();
    target.extra.extend(model.dimensions.iter().map(OutputExtra::new));
    target.sync_extra_bytes();
    let fields: Vec<(usize, usize)> = model.dimensions.iter().map(|d| { let (_, o) = layout.extra.iter().find(|(e, _)| e.name == d.name).unwrap(); (*o, d.size()) }).collect();
    let labels: Vec<(Layout, Option<[u8; 24]>)> = std::iter::once(&model.instance).chain(model.semantic.as_ref())
        .map(|name| { let (l, e) = layout.find(name).unwrap(); (l, e.filter(|e| e.has_no_data()).map(|e| e.no_data)) }).collect();
    let base = crate::points::base_record_length(header.point_format);
    let file = pair.prediction.file_name().unwrap().to_string_lossy().into_owned();
    let mut total = 0u64;
    let mut scratch = [0u8; 8];
    let mut batches = Vec::new();
    for chunk in chunks(&pair.prediction)? {
        let chunk = chunk?;
        let mut out = Vec::with_capacity(chunk.len() * target.record_length());
        for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
            for (k, (field, no_data)) in labels.iter().enumerate() {
                let raw = field.get(r, &mut scratch);
                let value = match field.scalar() {
                    crate::points::Scalar::F32 => f32::from_le_bytes(raw[..4].try_into().unwrap()) as f64,
                    crate::points::Scalar::F64 => f64::from_le_bytes(raw[..8].try_into().unwrap()),
                    s => s.read_int(raw) as f64,
                };
                let name = if k == 0 { &model.instance } else { model.semantic.as_ref().unwrap() };
                if !value.is_finite() || value < 0.0 || value != value.floor() {
                    return Err(format!("{}/{file}: invalid or missing {name} labels", model.name));
                }
                if k == 0 && value > u32::MAX as f64 {
                    return Err(format!("{}/{file}: instance IDs exceed uint32", model.name));
                }
                if no_data.is_some_and(|nd| nd[..raw.len()] == *raw) {
                    return Err(format!("{}/{file}: missing {name} prediction (no-data sentinel)", model.name));
                }
            }
            out.extend_from_slice(&r[..base]);
            for (o, size) in kept.iter().chain(&fields) { out.extend_from_slice(&r[*o..o + size]); }
            total += 1;
        }
        batches.push(Ok(out));
    }
    lasio::write_records(&output.join(format!("tile_{n:05}.laz")), &target, batches, true, None)?;
    Ok((total, total))
}

/// One prediction tile in memory: local coordinates, raw prediction values, and its spatial hash.
struct Source { set: PointSet, values: Vec<u8>, width: usize }

fn read_records(path: &Path, mut f: impl FnMut(&las::PointData) -> Res<()>) -> Res<()> {
    let mut reader = las::Reader::from_path(path).map_err(|e| format!("{}: {e}", path.display()))?;
    loop {
        let chunk = reader.read_points(MAX_BATCH_POINTS).map_err(|e| e.to_string())?;
        if chunk.is_empty() { break; }
        f(&chunk)?;
    }
    Ok(())
}

fn local(raw: &[u8], h: &SourceHeader, origin: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|a| local_coordinate(i32::from_le_bytes(raw[a * 4..a * 4 + 4].try_into().unwrap()), h.scales[a], h.offsets[a], origin[a]))
}

/// `validate_labels` for one record.
fn validate(model: &Model, layout: &PointLayout, record: &[u8], file: &Path) -> Res<()> {
    let mut scratch = [0u8; 8];
    for name in std::iter::once(&model.instance).chain(model.semantic.as_ref()) {
        let (l, eb) = layout.find(name).unwrap();
        let v = l.get(record, &mut scratch);
        let value = l.scalar().read_int(v);
        if value < 0 || (name == &model.instance && value > u32::MAX as i128) {
            return Err(format!("{}/{}: invalid or missing {name} labels", model.name, file.display()));
        }
        if let Some(eb) = eb {
            if eb.has_no_data() && v == &eb.no_data[..v.len()] {
                return Err(format!("{}/{}: missing {name} prediction (no-data sentinel)", model.name, file.display()));
            }
        }
    }
    Ok(())
}

fn load_source(path: &Path, model: &Model, origin: &[f64; 3], radius: f64) -> Res<Source> {
    let header = lasio::read_header(path)?;
    let layout = PointLayout::of(&header)?;
    let fields: Vec<Layout> = model.dimensions.iter().map(|d| layout.find(&d.name).unwrap().0).collect();
    let width: usize = model.dimensions.iter().map(|d| d.size()).sum();
    // The transfer radius spans several 10 cm points: 27 radius-sized cells beat 8 larger ones.
    let mut source = Source { set: PointSet::with_cell(radius * 1.001), values: Vec::new(), width };
    let mut index = 0i64;
    let mut scratch = [0u8; 8];
    read_records(path, |chunk| {
        for record in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
            validate(model, &layout, record, path)?;
            source.set.push(local(record, &header, origin), 0, index);
            for f in &fields {
                source.values.extend_from_slice(f.get(record, &mut scratch));
            }
            index += 1;
        }
        Ok(())
    })?;
    source.set.build();
    Ok(source)
}

/// `_transfer_tile`: the dense tile of one pair, written as `tile_NNNNN.laz`.
fn transfer_tile(model: &Model, n: usize, pair: &Pair, output: &Path, origin: &[f64; 3], radius: f64,
                 pool: &rayon::ThreadPool) -> Res<(u64, u64)> {
    let t0 = std::time::Instant::now();
    let source = load_source(&pair.prediction, model, origin, radius)?;
    let loaded = t0.elapsed().as_secs_f64();
    let (decode, query) = (std::cell::Cell::new(0.0), std::cell::Cell::new(0.0));
    let target_header = lasio::read_header(&pair.target)?;
    let target_layout = PointLayout::of(&target_header)?;
    if let Some(d) = model.dimensions.iter().find(|d| target_layout.find(&d.name).is_some()) {
        return Err(format!("Target already contains prediction dimension {}", d.name));
    }
    // `target_header`: a copy of the target (CRS standardized), its own ExtraBytes kept, plus the prediction dimensions.
    let mut header = TileHeader::for_tile(&target_header, target_header.scales, target_header.offsets)?;
    header.point_format = target_header.point_format;
    header.extra = target_layout.extra.iter().map(|(e, _)| OutputExtra::stored(e)).chain(model.dimensions.iter().map(OutputExtra::new)).collect();
    if !target_layout.extra.is_empty() { header.sync_extra_bytes(); }
    let base = target_layout.record_length;
    let out_len = header.record_length();
    let (mut total, mut matched) = (0u64, 0u64);
    let mut reader = las::Reader::from_path(&pair.target).map_err(|e| format!("{}: {e}", pair.target.display()))?;
    let (total_ref, matched_ref) = (&mut total, &mut matched);
    // One 32,768-point batch at a time, as `chunk_iterator(MAX_BATCH_POINTS)`.
    // 32,768-point batches (`chunk_iterator(MAX_BATCH_POINTS)`), queried
    // GROUP at a time on the worker pool and written in order.
    const GROUP: usize = 32;
    let record_len = target_header.record_length as usize;
    let transfer = |chunk: &[u8]| -> (Vec<u8>, u64, u64) {
        let records: Vec<&[u8]> = chunk.chunks_exact(record_len).collect();
        let xyz: Vec<[f64; 3]> = records.iter().map(|r| local(r, &target_header, origin)).collect();
        let nearest = source.set.nearest(&xyz, radius, &|_| true);
        let mut out = Vec::with_capacity(records.len() * out_len);
        let mut hits = 0;
        for (k, r) in records.iter().enumerate() {
            out.extend_from_slice(&r[..base]);
            match nearest[k] {
                Some((id, _)) => {
                    hits += 1;
                    let at = id as usize * source.width;
                    out.extend_from_slice(&source.values[at..at + source.width]);
                }
                None => out.extend(std::iter::repeat_n(0u8, source.width)),
            }
        }
        (out, records.len() as u64, hits)
    };
    let mut pending: std::collections::VecDeque<Vec<u8>> = std::collections::VecDeque::new();
    let mut failed = None;
    let batches = std::iter::from_fn(|| {
        if pending.is_empty() && failed.is_none() {
            let t = std::time::Instant::now();
            // One large read (decompressed in parallel), processed in 32,768-point batches.
            let block = match reader.read_points(GROUP as u64 * MAX_BATCH_POINTS) {
                Ok(c) => Some(c),
                Err(e) => { failed = Some(e.to_string()); None }
            };
            let chunks: Vec<&[u8]> = block.as_ref().map(|b| b.raw_bytes().chunks(MAX_BATCH_POINTS as usize * record_len).collect()).unwrap_or_default();
            decode.set(decode.get() + t.elapsed().as_secs_f64());
            let t = std::time::Instant::now();
            let results: Vec<(Vec<u8>, u64, u64)> = pool.install(|| chunks.par_iter().map(|c| transfer(c)).collect());
            query.set(query.get() + t.elapsed().as_secs_f64());
            for (out, n, hits) in results {
                *total_ref += n;
                *matched_ref += hits;
                pending.push_back(out);
            }
        }
        match pending.pop_front() {
            Some(out) => Some(Ok(out)),
            None => failed.take().map(Err),
        }
    });
    lasio::write_records(&output.join(format!("tile_{n:05}.laz")), &header, batches, true, Some(pool))?;
    if std::env::var_os("SMARTTILE_TIMINGS").is_some() {
        eprintln!("    tile {n}: load {loaded:.1} s, decode {:.1} s, query {:.1} s, total {:.1} s", decode.get(), query.get(), t0.elapsed().as_secs_f64());
    }
    if matched != total {
        return Err(format!("{}: incomplete prediction assignment on {}: {matched}/{total} within {radius} m",
                           model.name, pair.target.file_name().unwrap().to_string_lossy()));
    }
    Ok((total, matched))
}

fn spacing(x: f64) -> f64 { f64::from_bits(x.to_bits() + 1) - x }

/// `tile_overlaps`: for each tile, the intersection with every earlier tile's buffered bounds,
/// in local coordinates, widened by 8 ULPs of the global coordinates.
pub type Overlaps = Vec<BTreeMap<usize, ([f64; 2], [f64; 2])>>;

fn tile_overlaps(pairs: &[Pair], layout: &Value, origin: &[f64; 3]) -> Overlaps {
    if pairs.len() == 1 {
        return vec![BTreeMap::new()];
    }
    let tiles = layout["tiles"].as_array().unwrap();
    let bounds: Vec<[f64; 4]> = pairs.iter().map(|p| {
        let t = &tiles[p.layout_tile.unwrap()];
        [t["bounds"][0][0].as_f64().unwrap(), t["bounds"][0][1].as_f64().unwrap(), t["bounds"][1][0].as_f64().unwrap(), t["bounds"][1][1].as_f64().unwrap()]
    }).collect();
    bounds.iter().enumerate().map(|(i, &[x0, x1, y0, y1])| {
        let mut previous = BTreeMap::new();
        for (j, &[a0, a1, b0, b1]) in bounds[..i].iter().enumerate() {
            let lo = [x0.max(a0), y0.max(b0)];
            let hi = [x1.min(a1), y1.min(b1)];
            if hi[0] > lo[0] && hi[1] > lo[1] {
                let allowance: [f64; 2] = std::array::from_fn(|a| 8.0 * spacing(1f64.max(lo[a].abs().max(hi[a].abs()).max(origin[a].abs()))));
                previous.insert(j, ([lo[0] - origin[0] - allowance[0], lo[1] - origin[1] - allowance[1]],
                                    [hi[0] - origin[0] + allowance[0], hi[1] - origin[1] + allowance[1]]));
            }
        }
        previous
    }).collect()
}

/// Every point of a set of tile files, with its tile, point index and instance label.
pub struct TileIndex { pub set: PointSet, pub labels: Vec<u64> }

fn build_index(files: &[PathBuf], model: &Model, origin: &[f64; 3], radius: f64) -> Res<TileIndex> {
    Ok(build_index_where(files, model, origin, radius, None, None)?.0)
}

/// Which points of a tile an index keeps (tile, local point).
type Keep<'a> = &'a (dyn Fn(usize, &[f64; 3]) -> bool + Sync);

/// The index of `build_index`, optionally restricted to the points `keep` accepts (their ids
/// stay in tile and point order, their `index` the point's position in its tile), and with the
/// `semantic` dimension of each kept point.
fn build_index_where(files: &[PathBuf], model: &Model, origin: &[f64; 3], radius: f64, keep: Option<Keep>, semantic: Option<&str>)
                     -> Res<(TileIndex, Vec<u64>)> {
    let started = std::time::Instant::now();
    type Part = (Vec<[f64; 3]>, Vec<u64>, Vec<u64>, Vec<i64>); // xyz, labels, semantics, point index
    // Tiles decode in parallel, then join in tile order.
    let parts: Vec<Res<Part>> = files.par_iter().enumerate().map(|(tile, file)| {
        let header = lasio::read_header(file)?;
        let layout = PointLayout::of(&header)?;
        let (field, _) = layout.find(&model.instance).unwrap();
        let semantic_field = match semantic {
            Some(name) => Some(layout.find(name).ok_or_else(|| format!("{}: no {name}", file.display()))?.0),
            None => None,
        };
        let capacity = if keep.is_some() { 0 } else { header.point_count as usize };
        let mut part: Part = (Vec::with_capacity(capacity), Vec::with_capacity(capacity), Vec::new(), Vec::new());
        let record_len = header.record_length as usize;
        let mut offset = 0i64;
        for block in blocks(file)? {
            let block = block?;
            let batch = MAX_BATCH_POINTS as usize * record_len;
            let parts: Vec<Part> = block.raw_bytes().par_chunks(batch).enumerate().map(|(n, chunk)| {
                let mut scratch = [0u8; 8];
                let mut out: Part = Default::default();
                for (k, r) in chunk.chunks_exact(record_len).enumerate() {
                    let p = local(r, &header, origin);
                    if keep.is_some_and(|keep| !keep(tile, &p)) { continue; }
                    out.0.push(p);
                    out.1.push(field.scalar().read_int(field.get(r, &mut scratch)) as u64);
                    if let Some(f) = semantic_field { out.2.push(f.scalar().read_int(f.get(r, &mut scratch)) as u64); }
                    out.3.push(offset + (n * MAX_BATCH_POINTS as usize + k) as i64);
                }
                out
            }).collect();
            offset += (block.raw_bytes().len() / record_len) as i64;
            for (x, l, s, i) in parts { part.0.extend(x); part.1.extend(l); part.2.extend(s); part.3.extend(i); }
        }
        Ok(part)
    }).collect();
    let t_read = started.elapsed().as_secs_f64();
    let mut index = TileIndex { set: PointSet::new(radius), labels: Vec::new() };
    let mut semantics = Vec::new();
    for (tile, part) in parts.into_iter().enumerate() {
        let (xyz, labels, s, at) = part?;
        for (p, n) in xyz.into_iter().zip(at) {
            index.set.push(p, tile as u32, n);
        }
        index.labels.extend(labels);
        semantics.extend(s);
    }
    if semantic.is_none() { semantics.resize(index.labels.len(), 0); }
    let t_push = started.elapsed().as_secs_f64();
    index.set.build();
    if std::env::var_os("SMARTTILE_TIMINGS").is_some() {
        eprintln!("    index: read {t_read:.1} s, push {:.1} s, build {:.1} s, {} points", t_push - t_read, started.elapsed().as_secs_f64() - t_push, index.set.len());
    }
    Ok((index, semantics))
}

/// Large blocks of raw records (32 batches, decompressed in parallel); callers split them
/// into `MAX_BATCH_POINTS` batches, which keeps `chunk_iterator` boundaries.
fn blocks(path: &Path) -> Res<impl Iterator<Item = Res<las::PointData>>> {
    let mut reader = las::Reader::from_path(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(std::iter::from_fn(move || match reader.read_points(32 * MAX_BATCH_POINTS) {
        Ok(c) if c.is_empty() => None,
        Ok(c) => Some(Ok(c)),
        Err(e) => Some(Err(e.to_string())),
    }))
}

/// Batches of raw records of a LAS/LAZ file (`chunk_iterator(MAX_BATCH_POINTS)`).
fn chunks(path: &Path) -> Res<impl Iterator<Item = Res<las::PointData>>> {
    let mut reader = las::Reader::from_path(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(std::iter::from_fn(move || match reader.read_points(MAX_BATCH_POINTS) {
        Ok(c) if c.is_empty() => None,
        Ok(c) => Some(Ok(c)),
        Err(e) => Some(Err(e.to_string())),
    }))
}

/// `--filter-anchor`: the point that decides which tile owns an instance.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Anchor { Centroid, Highest, Lowest }

impl Anchor {
    pub fn parse(value: &str) -> Res<Anchor> {
        match value {
            "centroid" => Ok(Anchor::Centroid),
            "highest_point" => Ok(Anchor::Highest),
            "lowest_point" => Ok(Anchor::Lowest),
            other => Err(format!("Unknown filter anchor: {other}")),
        }
    }
    fn name(self) -> &'static str {
        match self { Anchor::Centroid => "centroid", Anchor::Highest => "highest_point", Anchor::Lowest => "lowest_point" }
    }
}

pub struct Ownership { pub removed: Vec<u64>, pub counts: BTreeMap<u64, u64>, pub sizes: BTreeMap<u64, u64>, pub background_input: u64, pub background_removed: u64,
                       pub input: u64, pub surviving: u64, pub decisions: Vec<Decision> }

/// One instance's provisional ownership decision (`_filter_tile`), world coordinates.
pub struct Decision { pub instance: u64, pub points: u64, pub anchor: [f64; 3], pub lower: [f64; 3], pub upper: [f64; 3], pub kept: bool }

/// `_filter_tile` (centroid anchor): keep owned instances and core-owned background.
fn filter_tile(model: &Model, file: &Path, region: &Region, output: &Path, origin: &[f64; 3], anchor: Anchor) -> Res<Ownership> {
    let header = lasio::read_header(file)?;
    let layout = PointLayout::of(&header)?;
    let (label_field, _) = layout.find(&model.instance).unwrap();
    // Per instance: size, XYZ sum (per batch, then added), box. Batches of a block run in parallel.
    type Aggregate = (u64, [f64; 3], [f64; 3], [f64; 3]);
    let mut aggregates: BTreeMap<u64, Aggregate> = BTreeMap::new();
    let record_len = header.record_length as usize;
    let aggregate = |chunk: &[u8]| -> BTreeMap<u64, Aggregate> {
        let mut scratch = [0u8; 8];
        let records: Vec<&[u8]> = chunk.chunks_exact(record_len).collect();
        let mut batch: BTreeMap<u64, Aggregate> = BTreeMap::new();
        for r in &records {
            let label = label_field.scalar().read_int(label_field.get(r, &mut scratch)) as u64;
            if label == 0 { continue; }
            let p = local(r, &header, origin);
            let e = batch.entry(label).or_insert((0, [0.0; 3], [f64::INFINITY; 3], [f64::NEG_INFINITY; 3]));
            // Centroid: the sum; highest/lowest: the first extreme point in record order.
            match anchor {
                Anchor::Centroid => for a in 0..3 { e.1[a] += p[a]; },
                Anchor::Highest => if e.0 == 0 || p[2] > e.1[2] { e.1 = p; },
                Anchor::Lowest => if e.0 == 0 || p[2] < e.1[2] { e.1 = p; },
            }
            e.0 += 1;
            for a in 0..3 { e.2[a] = e.2[a].min(p[a]); e.3[a] = e.3[a].max(p[a]); }
        }
        batch
    };
    for block in blocks(file)? {
        let block = block?;
        let parts: Vec<BTreeMap<u64, Aggregate>> = block.raw_bytes().par_chunks(MAX_BATCH_POINTS as usize * record_len).map(aggregate).collect();
      for batch in parts {
        for (label, (n, value, lo, hi)) in batch {
            let e = aggregates.entry(label).or_insert((0, [0.0; 3], [f64::INFINITY; 3], [f64::NEG_INFINITY; 3]));
            let first = e.0 == 0;
            e.0 += n;
            match anchor {
                Anchor::Centroid => for a in 0..3 { e.1[a] = if first { value[a] } else { e.1[a] + value[a] }; },
                Anchor::Highest => if first || value[2] > e.1[2] { e.1 = value; },
                Anchor::Lowest => if first || value[2] < e.1[2] { e.1 = value; },
            }
            for a in 0..3 {
                e.2[a] = e.2[a].min(lo[a]);
                e.3[a] = e.3[a].max(hi[a]);
            }
        }
      }
    }
    let mut out = Ownership { removed: Vec::new(), counts: BTreeMap::new(), sizes: aggregates.iter().map(|(k, v)| (*k, v.0)).collect(),
                              background_input: 0, background_removed: 0, input: 0, surviving: 0, decisions: Vec::new() };
    for (&label, &(n, sum, lo, hi)) in &aggregates {
        let point: [f64; 3] = std::array::from_fn(|a| if anchor == Anchor::Centroid { sum[a] / n as f64 } else { sum[a] } + origin[a]);
        let kept = owned_anchor(point[0], point[1], region);
        if kept {
            out.counts.insert(label, n);
        } else {
            out.removed.push(label);
        }
        out.decisions.push(Decision { instance: label, points: n, anchor: point, lower: std::array::from_fn(|a| lo[a] + origin[a]),
                                      upper: std::array::from_fn(|a| hi[a] + origin[a]), kept });
    }
    let target = TileHeader::copy_of(&header)?;
    let removed: std::collections::BTreeSet<u64> = out.removed.iter().copied().collect();
    let mut background = (0u64, 0u64);
    let keep_chunk = |chunk: &[u8]| -> (Vec<u8>, u64, u64) {
        let (mut kept, mut seen, mut dropped) = (Vec::new(), 0u64, 0u64);
        let mut scratch = [0u8; 8];
        for r in chunk.chunks_exact(record_len) {
            let label = label_field.scalar().read_int(label_field.get(r, &mut scratch)) as u64;
            let keep = if label == 0 {
                let p = local(r, &header, origin);
                seen += 1;
                let keep = owned_background(p[0] + origin[0], p[1] + origin[1], region);
                if !keep { dropped += 1; }
                keep
            } else {
                !removed.contains(&label)
            };
            if keep { kept.extend_from_slice(r); }
        }
        (kept, seen, dropped)
    };
    let mut batches: Vec<Res<Vec<u8>>> = Vec::new();
    for block in blocks(file)? {
        let block = block?;
        let parts: Vec<(Vec<u8>, u64, u64)> = block.raw_bytes().par_chunks(MAX_BATCH_POINTS as usize * record_len).map(keep_chunk).collect();
        for (kept, seen, dropped) in parts {
            background.0 += seen;
            background.1 += dropped;
            batches.push(Ok(kept));
        }
    }
    out.surviving = lasio::write_records(output, &target, batches, false, None)?;
    out.input = header.point_count;
    out.background_input = background.0;
    out.background_removed = background.1;
    Ok(out)
}

/// `recover_orphaned_instances`: admit rejected whole instances that supply the most
/// uncovered core samples (greedy, ties by tile then ID). Returns the admitted (tile, ID)s.
/// `orphan_recovery` report numbers: claimants, distinct uncovered samples, winners with
/// their scores, and the samples the selection checked (SQLite R*Tree candidates).
#[derive(Default)]
pub struct OrphanReport { pub candidates: usize, pub uncovered: usize, pub admitted: Vec<(usize, u64, i64)>, pub checked: u64 }

/// SQLite R*Tree storage: 32-bit bounds rounded outward.
fn f32_down(x: f64) -> f64 {
    let f = x as f32;
    if (f as f64) <= x { f as f64 } else { f32_step(f, false) as f64 }
}

fn f32_up(x: f64) -> f64 {
    let f = x as f32;
    if (f as f64) >= x { f as f64 } else { f32_step(f, true) as f64 }
}

fn f32_step(f: f32, up: bool) -> f32 {
    if f == 0.0 { return if up { f32::from_bits(1) } else { -f32::from_bits(1) }; }
    let bits = f.to_bits();
    f32::from_bits(if (f > 0.0) == up { bits + 1 } else { bits - 1 })
}

/// The 1 m XY cells (local coordinates) within 2 cm of a rejected instance's box.
fn rejected_cells(ownership: &[Ownership], origin: &[f64; 3]) -> std::collections::HashSet<(i64, i64)> {
    const MARGIN: f64 = 0.02;
    let mut cells = std::collections::HashSet::new();
    for d in ownership.iter().flat_map(|o| &o.decisions).filter(|d| !d.kept) {
        let edge = |v: f64, a: usize, m: f64| (v - origin[a] + m).floor() as i64;
        for x in edge(d.lower[0], 0, -MARGIN)..=edge(d.upper[0], 0, MARGIN) {
            for y in edge(d.lower[1], 1, -MARGIN)..=edge(d.upper[1], 1, MARGIN) { cells.insert((x, y)); }
        }
    }
    cells
}

fn orphan_recovery(model: &Model, dense: &[PathBuf], ownership: &[Ownership], owned_index: &TileIndex, regions: &[Region],
                   overlaps: &Overlaps, origin: &[f64; 3], report: &mut OrphanReport) -> Res<Vec<(usize, u64)>> {
    use std::collections::{BTreeSet, HashMap};
    const RADIUS: f64 = 0.01;
    let mut locations: HashMap<[u64; 3], u32> = HashMap::new();
    let mut xyz_of: Vec<[f64; 3]> = Vec::new();
    let mut claims: BTreeMap<(usize, u64), BTreeSet<[u8; 24]>> = BTreeMap::new();
    let positive = |id: u32| owned_index.labels[id as usize] > 0;
    // Uncovered core samples of rejected instances, per tile in parallel.
    let gathered: Vec<Res<Vec<(u64, [f64; 3])>>> = dense.par_iter().enumerate().map(|(tile, file)| {
        let mut out = Vec::new();
        let rejected: BTreeSet<u64> = ownership[tile].removed.iter().copied().collect();
        if rejected.is_empty() { return Ok(out); }
        let header = lasio::read_header(file)?;
        let layout = PointLayout::of(&header)?;
        let (field, _) = layout.find(&model.instance).unwrap();
        // `_core_union_mask`: the tile's own core or that of a tile overlapping it.
        let cores: Vec<[f64; 4]> = regions.iter().enumerate()
            .filter(|(other, _)| *other == tile || overlaps[tile.max(*other)].contains_key(&tile.min(*other)))
            .map(|(_, r)| [r.core[0][0] - origin[0], r.core[0][1] - origin[0], r.core[1][0] - origin[1], r.core[1][1] - origin[1]]).collect();
        let record_len = header.record_length as usize;
        // Each 32,768-point batch is one query (its eligible points); batches run in parallel.
        let claim = |chunk: &[u8]| -> Vec<(u64, [f64; 3])> {
            let mut scratch = [0u8; 8];
            let mut labels = Vec::new();
            let mut xyz = Vec::new();
            for r in chunk.chunks_exact(record_len) {
                let label = field.scalar().read_int(field.get(r, &mut scratch)) as u64;
                if !rejected.contains(&label) { continue; }
                let p = local(r, &header, origin);
                if cores.iter().any(|c| p[0] >= c[0] && p[0] <= c[1] && p[1] >= c[2] && p[1] <= c[3]) {
                    labels.push(label);
                    xyz.push(p);
                }
            }
            if xyz.is_empty() { return Vec::new(); }
            let covered = owned_index.set.covered(&xyz, RADIUS, &positive);
            (0..xyz.len()).filter(|&k| !covered[k]).map(|k| (labels[k], xyz[k])).collect()
        };
        for block in blocks(file)? {
            let block = block?;
            let parts: Vec<Vec<(u64, [f64; 3])>> = block.raw_bytes().par_chunks(MAX_BATCH_POINTS as usize * record_len).map(claim).collect();
            for part in parts { out.extend(part); }
        }
        Ok(out)
    }).collect();
    for (tile, part) in gathered.into_iter().enumerate() {
        for (label, p) in part? {
            let mut loc = [0u8; 24];
            for a in 0..3 { loc[a * 8..a * 8 + 8].copy_from_slice(&p[a].to_le_bytes()); }
            claims.entry((tile, label)).or_default().insert(loc);
            locations.entry(p.map(f64::to_bits)).or_insert_with(|| { xyz_of.push(p); (xyz_of.len() - 1) as u32 });
        }
    }
    // `select_claims`: greedy winners; a winner's samples cover nearby uncovered samples.
    let loc_id = |loc: &[u8; 24]| -> u32 {
        let bits: [u64; 3] = std::array::from_fn(|a| u64::from_le_bytes(loc[a * 8..a * 8 + 8].try_into().unwrap()));
        locations[&bits]
    };
    let t_gathered = std::time::Instant::now();
    report.candidates = claims.len();
    report.uncovered = xyz_of.len();
    let mut claimants: Vec<Vec<(usize, u64)>> = vec![Vec::new(); xyz_of.len()];
    let mut scores: BTreeMap<(usize, u64), i64> = BTreeMap::new();
    for (key, locs) in &claims {
        scores.insert(*key, locs.len() as i64);
        for loc in locs { claimants[loc_id(loc) as usize].push(*key); }
    }
    let mut location_set = PointSet::new(0.5);
    for (i, p) in xyz_of.iter().enumerate() { location_set.push(*p, 0, i as i64); }
    location_set.build();
    let padding = {
        let mut padding = RADIUS;
        let all: Vec<[f64; 3]> = xyz_of.clone();
        for chunk in all.chunks(MAX_BATCH_POINTS as usize) {
            padding = padding.max(RADIUS + crate::spatial::distance_roundoff(chunk));
        }
        padding
    };
    CLAIMS.with(|c| *c.borrow_mut() = claims.iter().map(|(k, locs)| (*k, locs.iter().map(|loc| xyz_of[loc_id(loc) as usize]).collect())).collect());
    let mut covered = vec![false; xyz_of.len()];
    let mut selected = Vec::new();
    loop {
        let best = scores.iter().filter(|(_, s)| **s > 0).max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).map(|(k, s)| (*k, *s));
        let Some((key, score)) = best else { break };
        let winner: Vec<[f64; 3]> = claims[&key].iter().map(|loc| xyz_of[loc_id(loc) as usize]).collect();
        for batch in winner.chunks(MAX_BATCH_POINTS as usize) {
            for group in crate::spatial::spatial_batches(batch) {
                let points: Vec<[f64; 3]> = group.iter().map(|&i| batch[i]).collect();
                let lo: [f64; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).fold(f64::INFINITY, f64::min) - padding);
                let hi: [f64; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).fold(f64::NEG_INFINITY, f64::max) + padding);
                // The R*Tree query returns every uncovered sample whose outward-rounded box meets [lo, hi].
                // A margin above any 32-bit rounding (2^-23 relative); the exact test follows.
                let margin = |a: usize| 1e-3f64.max(lo[a].abs().max(hi[a].abs()) * 1e-6);
                let wide_lo: [f64; 3] = std::array::from_fn(|a| lo[a] - margin(a));
                let wide_hi: [f64; 3] = std::array::from_fn(|a| hi[a] + margin(a));
                report.checked += location_set.in_box(&wide_lo, &wide_hi).into_iter().filter(|&id| {
                    let p = &xyz_of[id as usize];
                    !covered[id as usize] && (0..3).all(|a| f32_up(p[a]) >= lo[a] && f32_down(p[a]) <= hi[a])
                }).count() as u64;
                let candidates: Vec<u32> = location_set.in_box(&lo, &hi).into_iter().filter(|&id| !covered[id as usize]).collect();
                if candidates.is_empty() { continue; }
                let cand_xyz: Vec<[f64; 3]> = candidates.iter().map(|&id| xyz_of[id as usize]).collect();
                let mut group_set = PointSet::new(RADIUS);
                for p in &points { group_set.push(*p, 0, 0); }
                group_set.build();
                let supported = group_set.covered(&cand_xyz, RADIUS, &|_| true);
                for (k, &id) in candidates.iter().enumerate() {
                    if supported[k] {
                        covered[id as usize] = true;
                        for c in &claimants[id as usize] { *scores.get_mut(c).unwrap() -= 1; }
                    }
                }
            }
        }
        selected.push(key);
        report.admitted.push((key.0, key.1, score));
    }
    if std::env::var_os("SMARTTILE_TIMINGS").is_some() { eprintln!("    orphan: selection {:.1} s", t_gathered.elapsed().as_secs_f64()); }
    Ok(selected)
}

/// Rewrite a tile from its dense file keeping recovered instances (`recover_orphaned_instances`).
fn rewrite_recovered(model: &Model, dense: &Path, rejected: &std::collections::BTreeSet<u64>, region: &Region, output: &Path, origin: &[f64; 3]) -> Res<u64> {
    let header = lasio::read_header(dense)?;
    let layout = PointLayout::of(&header)?;
    let (field, _) = layout.find(&model.instance).unwrap();
    let target = TileHeader::copy_of(&header)?;
    let record_len = header.record_length as usize;
    let keep = |chunk: &[u8]| -> Vec<u8> {
        let mut kept = Vec::new();
        let mut scratch = [0u8; 8];
        for r in chunk.chunks_exact(record_len) {
            let label = field.scalar().read_int(field.get(r, &mut scratch)) as u64;
            let keep = if label == 0 {
                let p = local(r, &header, origin);
                owned_background(p[0] + origin[0], p[1] + origin[1], region)
            } else {
                !rejected.contains(&label)
            };
            if keep { kept.extend_from_slice(r); }
        }
        kept
    };
    let mut batches: Vec<Res<Vec<u8>>> = Vec::new();
    for block in blocks(dense)? {
        let block = block?;
        let parts: Vec<Vec<u8>> = block.raw_bytes().par_chunks(MAX_BATCH_POINTS as usize * record_len).map(keep).collect();
        batches.extend(parts.into_iter().map(Ok));
    }
    lasio::write_records(output, &target, batches, false, None)
}

/// The uncovered core samples each admitted orphan claimed (sorted as the claims table).
#[allow(clippy::too_many_arguments)]
fn claims_of(model: &Model, dense: &[PathBuf], ownership: &[Ownership], regions: &[Region], overlaps: &Overlaps,
             origin: &[f64; 3], admitted: &[(usize, u64)]) -> Res<BTreeMap<(usize, u64), Vec<[f64; 3]>>> {
    let _ = (model, dense, ownership, regions, overlaps, origin);
    let mut out = BTreeMap::new();
    for key in admitted {
        out.insert(*key, CLAIMS.with(|c| c.borrow().get(key).cloned().unwrap_or_default()));
    }
    Ok(out)
}

thread_local! {
    /// The orphan claims of this run, kept for the final support check.
    static CLAIMS: std::cell::RefCell<BTreeMap<(usize, u64), Vec<[f64; 3]>>> = std::cell::RefCell::new(BTreeMap::new());
}

/// Checked and unsupported samples: a positive survivor within 1 cm (plus guard) of every sample.
fn recovered_support(admitted: &[(usize, u64)], claims: &BTreeMap<(usize, u64), Vec<[f64; 3]>>, survivors: &crate::parts::Parts)
                     -> Res<(u64, u64, Vec<(usize, u64, [f64; 3])>)> {
    let (mut checked, mut missing, mut examples) = (0, 0, Vec::new());
    for key in admitted {
        for batch in claims[key].chunks(MAX_BATCH_POINTS as usize) {
            let (lo, hi) = crate::parts::bounds(batch);
            let reached = survivors.reach(lo, hi, 0.01)?;
            let sets: Vec<&PointSet> = reached.iter().map(|l| &l.set).collect();
            let covered = crate::spatial::covered_across(&sets, batch, 0.01, &|s, id| reached[s].labels[id as usize] > 0);
            checked += batch.len() as u64;
            missing += covered.iter().filter(|c| !**c).count() as u64;
            for (k, c) in covered.iter().enumerate() {
                if !*c && examples.len() < 5 { examples.push((key.0, key.1, batch[k])); }
            }
        }
    }
    Ok((checked, missing, examples))
}

/// `inside_xy`: inclusive XY box membership with an 8-ULP allowance of the box magnitude.
fn inside_xy(p: &[f64; 3], b: &([f64; 2], [f64; 2])) -> bool {
    (0..2).all(|a| {
        let allowance = 8.0 * spacing(1f64.max(b.0[a].abs().max(b.1[a].abs())));
        p[a] >= b.0[a] - allowance && p[a] <= b.1[a] + allowance
    })
}

/// `reconcile_instances`: mutual-best overlapping instance pairs joined into
/// consistent cross-tile groups; returns (tile, local ID) -> dataset ID.
#[allow(clippy::too_many_arguments)]
/// `reconciliation` report numbers: accepted pairs, rejected recovery bridges and the query counters.
#[derive(Default)]
pub struct ReconcileReport { pub accepted: Vec<(usize, u64, usize, u64, u64)>, pub bridges: Vec<(usize, u64, usize, u64, u64)>, pub bridge_count: u64, pub counters: [u64; 4] }

#[allow(clippy::too_many_arguments)]
fn reconcile(model: &Model, files: &[PathBuf], index: &TileIndex, origin: &[f64; 3], counts: &BTreeMap<(usize, u64), u64>,
             overlap_threshold: f64, radius: f64, overlaps: &Overlaps, normal_keys: Option<&std::collections::BTreeSet<(usize, u64)>>,
             pool: &rayon::ThreadPool, report: &mut ReconcileReport, enabled: bool) -> Res<BTreeMap<(usize, u64), u64>> {
    // `_count_instance_pairs` per tile, against every earlier overlapping tile.
    // Per tile: pair counters and (files_read, points_read, query_points, skipped_query_points).
    type TileCounts = (BTreeMap<usize, BTreeMap<(u64, u64), u64>>, [u64; 4]);
    let per_tile: Vec<Res<TileCounts>> = pool.install(|| files.par_iter().enumerate().map(|(tile, file)| {
        let neighbors: Vec<usize> = if enabled { (0..tile).filter(|o| overlaps[tile].contains_key(o)).collect() } else { Vec::new() };
        let mut counters: BTreeMap<usize, BTreeMap<(u64, u64), u64>> = neighbors.iter().map(|o| (*o, BTreeMap::new())).collect();
        let mut metrics = [0u64; 4];
        if neighbors.is_empty() { return Ok((counters, metrics)); }
        metrics[0] = 1;
        let header = lasio::read_header(file)?;
        let layout = PointLayout::of(&header)?;
        let (field, _) = layout.find(&model.instance).unwrap();
        let mut scratch = [0u8; 8];
        let record_len = header.record_length as usize;
        // Per 32,768-point batch (in parallel): pair counts per earlier neighbour and the counters; summed below.
        let count = |chunk: &[u8]| -> (BTreeMap<usize, BTreeMap<(u64, u64), u64>>, [u64; 4]) {
            let mut scratch = [0u8; 8];
            let mut part: BTreeMap<usize, BTreeMap<(u64, u64), u64>> = BTreeMap::new();
            let mut m = [0u64; 4];
            let records: Vec<&[u8]> = chunk.chunks_exact(record_len).collect();
            let xyz: Vec<[f64; 3]> = records.iter().map(|r| local(r, &header, origin)).collect();
            let left: Vec<u64> = records.iter().map(|r| field.scalar().read_int(field.get(r, &mut scratch)) as u64).collect();
            let groups = crate::spatial::spatial_batches(&xyz);
            m[1] += records.len() as u64;
            for &other in &neighbors {
                let bounds = &overlaps[tile][&other];
                let positions: Vec<usize> = groups.iter().filter(|g| g.iter().any(|&i| left[i] > 0 && inside_xy(&xyz[i], bounds)))
                    .flat_map(|g| g.iter().copied()).collect();
                m[2] += positions.len() as u64;
                m[3] += (records.len() - positions.len()) as u64;
                if positions.is_empty() { continue; }
                let queries: Vec<[f64; 3]> = positions.iter().map(|&i| xyz[i]).collect();
                let allowed = |id: u32| index.set.tile[id as usize] as usize == other && inside_xy(&index.set.xyz[id as usize], bounds);
                let found = index.set.nearest(&queries, radius, &allowed);
                let counter = part.entry(other).or_default();
                for (k, &i) in positions.iter().enumerate() {
                    if !inside_xy(&xyz[i], bounds) { continue; }
                    if let Some((id, _)) = found[k] {
                        let right = index.labels[id as usize];
                        if left[i] > 0 && right > 0 {
                            *counter.entry((left[i], right)).or_default() += 1;
                        }
                    }
                }
            }
            (part, m)
        };
        for block in blocks(file)? {
            let block = block?;
            let parts: Vec<_> = block.raw_bytes().par_chunks(MAX_BATCH_POINTS as usize * record_len).map(count).collect();
            for (part, m) in parts {
                for k in 1..4 { metrics[k] += m[k]; }
                for (other, pairs) in part {
                    let counter = counters.get_mut(&other).unwrap();
                    for (pair, n) in pairs { *counter.entry(pair).or_default() += n; }
                }
            }
        }
        let _ = &mut scratch;
        Ok((counters, metrics))
    }).collect());
    let mut edges: Vec<(usize, u64, usize, u64, u64)> = Vec::new();
    for (tile, part) in per_tile.into_iter().enumerate() {
        let (counters, metrics) = part?;
        for k in 0..4 { report.counters[k] += metrics[k]; }
        for (other, pairs) in counters {
            let mut by_left: BTreeMap<u64, Vec<(u64, u64)>> = BTreeMap::new();
            let mut by_right: BTreeMap<u64, Vec<(u64, u64)>> = BTreeMap::new();
            for (&(a, b), &n) in &pairs {
                by_left.entry(a).or_default().push((n, b));
                by_right.entry(b).or_default().push((n, a));
            }
            for v in by_left.values_mut().chain(by_right.values_mut()) { v.sort_by(|x, y| y.cmp(x)); }
            for (&(a, b), &n) in &pairs {
                let (lb, rb) = (&by_left[&a], &by_right[&b]);
                if lb[0] != (n, b) || rb[0] != (n, a) || (lb.len() > 1 && lb[1].0 == n) || (rb.len() > 1 && rb[1].0 == n) {
                    continue;
                }
                if n as f64 / counts[&(tile, a)].min(counts[&(other, b)]) as f64 >= overlap_threshold {
                    edges.push((tile, a, other, b, n));
                }
            }
        }
    }
    if let Some(normal) = normal_keys {
        let is_normal = |e: &(usize, u64, usize, u64, u64)| normal.contains(&(e.0, e.1)) && normal.contains(&(e.2, e.3));
        let mut ordered: Vec<(usize, (usize, u64, usize, u64, u64))> = edges.into_iter().enumerate().collect();
        ordered.sort_by_key(|(i, e)| (if is_normal(e) { 0 } else { 1 }, if is_normal(e) { 0 } else { -(e.4 as i64) }, *i));
        edges = ordered.into_iter().map(|(_, e)| e).collect();
    }
    let keys: Vec<(usize, u64)> = counts.keys().copied().collect();
    let mut parent: BTreeMap<(usize, u64), (usize, u64)> = keys.iter().map(|k| (*k, *k)).collect();
    let mut members: BTreeMap<(usize, u64), BTreeMap<usize, u64>> = keys.iter().map(|k| (*k, BTreeMap::from([(k.0, k.1)]))).collect();
    let mut normal_component: Option<BTreeMap<(usize, u64), bool>> = normal_keys.map(|n| keys.iter().map(|k| (*k, n.contains(k))).collect());
    fn root(parent: &mut BTreeMap<(usize, u64), (usize, u64)>, mut key: (usize, u64)) -> (usize, u64) {
        while parent[&key] != key {
            let grand = parent[&parent[&key]];
            parent.insert(key, grand);
            key = grand;
        }
        key
    }
    for (tile, a, other, b, n) in edges {
        let (left, right) = (root(&mut parent, (tile, a)), root(&mut parent, (other, b)));
        if left == right {
            report.accepted.push((tile, a, other, b, n));
            continue;
        }
        if members[&left].iter().any(|(t, label)| members[&right].get(t).is_some_and(|l| l != label)) { continue; }
        if let (Some(nc), Some(normal)) = (&normal_component, normal_keys) {
            if nc[&left] && nc[&right] && !(normal.contains(&(tile, a)) && normal.contains(&(other, b))) {
                report.bridge_count += 1;
                if report.bridges.len() < 10 { report.bridges.push((tile, a, other, b, n)); }
                continue;
            }
        }
        let (survivor, removed) = if left < right { (left, right) } else { (right, left) };
        parent.insert(removed, survivor);
        let moved = members[&removed].clone();
        members.get_mut(&survivor).unwrap().extend(moved);
        if let Some(nc) = &mut normal_component {
            let r = nc[&removed];
            *nc.get_mut(&survivor).unwrap() |= r;
        }
        report.accepted.push((tile, a, other, b, n));
    }
    let roots: std::collections::BTreeSet<(usize, u64)> = keys.iter().map(|k| root(&mut parent, *k)).collect();
    let ids: BTreeMap<(usize, u64), u64> = roots.iter().enumerate().map(|(i, k)| (*k, i as u64 + 1)).collect();
    Ok(keys.iter().map(|k| { let r = root(&mut parent, *k); (*k, ids[&r]) }).collect())
}

/// `core_distance`: XY distance (local coordinates) to a tile's closed core.
fn core_distance(p: &[f64; 3], r: &Region, origin: &[f64; 3]) -> f64 {
    let d: [f64; 2] = std::array::from_fn(|a| ((r.core[a][0] - origin[a]) - p[a]).max(p[a] - (r.core[a][1] - origin[a])).max(0.0));
    (d[0] * d[0] + d[1] * d[1]).sqrt()
}

/// `preferred_core`: `first` beats `second` by core distance, then tile order.
fn preferred_core(p: &[f64; 3], first: usize, second: usize, regions: &[Region], origin: &[f64; 3]) -> bool {
    let (a, b) = (core_distance(p, &regions[first], origin), core_distance(p, &regions[second], origin));
    a < b || (a == b && first < second)
}

/// `assign_shared_points`: drop points a competing tile's retained tree owns
/// (tree claims by nearest core; with `background_only`, background under any tree).
#[allow(clippy::too_many_arguments)]
/// Per tile of a shared-point stage: removed points, up to five examples (point, owner tile, owner
/// record, distance), the queried points and their 4 m query regions (`OwnershipQuerySpool`).
pub struct SharedTile { pub removed: u64, pub examples: Vec<(u64, usize, i64, f64)>, pub query_points: u64, pub query_blocks: u64 }

#[allow(clippy::too_many_arguments)]
fn assign_shared_points(model: &Model, files: &[PathBuf], index: &TileIndex, regions: &[Region], mapping: &BTreeMap<(usize, u64), u64>,
                        output_dir: &Path, origin: &[f64; 3], overlaps: &Overlaps, background_only: bool, pool: &rayon::ThreadPool)
                        -> Res<Vec<(PathBuf, SharedTile)>> {
    const RADIUS: f64 = 0.01;
    fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    let mapped: Vec<u64> = (0..index.labels.len()).map(|i| {
        let label = index.labels[i];
        if label == 0 { 0 } else { mapping[&(index.set.tile[i] as usize, label)] }
    }).collect();
    let results: Vec<Res<(PathBuf, SharedTile)>> = pool.install(|| files.par_iter().enumerate().map(|(tile, file)| {
        let competitors: BTreeMap<usize, ([f64; 2], [f64; 2])> = (0..files.len()).filter(|o| *o != tile)
            .filter_map(|o| overlaps[tile.max(o)].get(&tile.min(o)).map(|b| (o, *b))).collect();
        let header = lasio::read_header(file)?;
        let layout = PointLayout::of(&header)?;
        let (field, _) = layout.find(&model.instance).unwrap();
        let mut scratch = [0u8; 8];
        let mut removed: Vec<bool> = Vec::new();
        let mut metric = SharedTile { removed: 0, examples: Vec::new(), query_points: 0, query_blocks: 0 };
        let mut blocks: std::collections::BTreeSet<(i64, i64)> = std::collections::BTreeSet::new();
        for chunk in chunks(file)? {
            let chunk = chunk?;
            let records: Vec<&[u8]> = chunk.raw_bytes().chunks_exact(chunk.record_len()).collect();
            let xyz: Vec<[f64; 3]> = records.iter().map(|r| local(r, &header, origin)).collect();
            let labels: Vec<u64> = records.iter().map(|r| {
                let l = field.scalar().read_int(field.get(r, &mut scratch)) as u64;
                if l == 0 { 0 } else { mapping[&(tile, l)] }
            }).collect();
            let eligible: Vec<usize> = (0..records.len()).filter(|&i| if background_only { labels[i] == 0 } else {
                labels[i] > 0 && competitors.iter().any(|(o, b)| inside_xy(&xyz[i], b) && preferred_core(&xyz[i], *o, tile, regions, origin))
            }).collect();
            let base = removed.len();
            removed.resize(base + records.len(), false);
            let points: Vec<[f64; 3]> = eligible.iter().map(|&i| xyz[i]).collect();
            metric.query_points += points.len() as u64;
            for group in crate::spatial::spatial_batches(&points) {
                let pts: Vec<[f64; 3]> = group.iter().map(|&k| points[k]).collect();
                blocks.insert(((pts[0][0] / 4.0).floor() as i64, (pts[0][1] / 4.0).floor() as i64));
                let limit = RADIUS + crate::spatial::distance_roundoff(&pts);
                let mut best_core: Vec<f64> = pts.iter().map(|p| core_distance(p, &regions[tile], origin)).collect();
                let mut best_tile: Vec<usize> = vec![tile; pts.len()];
                // Examples of this group by position: (owner tile, owner record, distance).
                let mut examples: BTreeMap<usize, (usize, i64, f64)> = BTreeMap::new();
                for (&other, bounds) in &competitors {
                    let any_allowed = pts.iter().any(|p| inside_xy(p, bounds) && (background_only || preferred_core(p, other, tile, regions, origin)));
                    if !any_allowed { continue; }
                    let candidate = |id: u32| {
                        let i = id as usize;
                        index.set.tile[i] as usize == other && mapped[i] > 0 && inside_xy(&index.set.xyz[i], bounds)
                            && (background_only || preferred_core(&index.set.xyz[i], other, tile, regions, origin))
                    };
                    let mut remove = vec![None; pts.len()];
                    for k in 0..pts.len() {
                        let mut allowed = inside_xy(&pts[k], bounds);
                        let other_core = core_distance(&pts[k], &regions[other], origin);
                        if !background_only {
                            allowed &= other_core < best_core[k] || (other_core == best_core[k] && other < best_tile[k]);
                        } else {
                            allowed &= !removed[base + eligible[group[k]]];
                        }
                        if !allowed { continue; }
                        // The nearest eligible record of the competitor (ties: lowest record).
                        let mut nearest: Option<(f64, i64)> = None;
                        index.set.within_pub(&pts[k], limit, |id, d| if candidate(id) {
                            let record = index.set.index[id as usize];
                            if nearest.is_none_or(|(bd, br)| d < bd || (d == bd && record < br)) { nearest = Some((d, record)); }
                        });
                        let Some((d, record)) = nearest else { continue };
                        remove[k] = Some((record, d));
                        if !background_only {
                            best_core[k] = other_core;
                            best_tile[k] = other;
                        }
                        removed[base + eligible[group[k]]] = true;
                    }
                    // `examples`: the first five removals, plus updates of positions already listed.
                    let mut listed: std::collections::BTreeSet<usize> = remove.iter().enumerate().filter(|(_, r)| r.is_some()).map(|(k, _)| k).take(5).collect();
                    listed.extend(examples.keys().copied().filter(|k| remove[*k].is_some()));
                    for k in listed {
                        if examples.contains_key(&k) || examples.len() < 5 {
                            let (record, d) = remove[k].unwrap();
                            examples.insert(k, (other, record, d));
                        }
                    }
                }
                let offset = (base) as u64;
                metric.examples.extend(examples.into_iter().map(|(k, (o, r, d))| (offset + eligible[group[k]] as u64, o, r, d)));
                metric.examples.sort_by_key(|e| e.0);
                metric.examples.truncate(5);
            }
        }
        metric.query_blocks = blocks.len() as u64;
        metric.removed = removed.iter().filter(|r| **r).count() as u64;
        let output = output_dir.join(file.file_name().unwrap());
        let target = TileHeader::copy_of(&header)?;
        let mut offset = 0usize;
        let batches = chunks(file)?.map(|chunk| chunk.map(|chunk| {
            let mut kept = Vec::new();
            for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
                if !removed[offset] { kept.extend_from_slice(r); }
                offset += 1;
            }
            kept
        }));
        lasio::write_records(&output, &target, batches, false, None)?;
        Ok((output, metric))
    }).collect());
    results.into_iter().collect()
}

/// Per positive instance: count, XYZ box and sums of the final local coordinates (`InstanceStatistics`).
#[derive(Default)]
pub struct Statistics { pub rows: BTreeMap<u64, (u64, [f64; 3], [f64; 3], [f64; 3])> }

impl Statistics {
    /// One batch: `np.add.at` accumulates in point order.
    fn add(&mut self, ids: &[u64], xyz: &[[f64; 3]]) {
        for (id, p) in ids.iter().zip(xyz) {
            if *id == 0 { continue; }
            let e = self.rows.entry(*id).or_insert((0, [f64::INFINITY; 3], [f64::NEG_INFINITY; 3], [0.0; 3]));
            e.0 += 1;
            for a in 0..3 { e.1[a] = e.1[a].min(p[a]); e.2[a] = e.2[a].max(p[a]); e.3[a] += p[a]; }
        }
    }
}

/// `instance_output_dtype`: uint16 unless an ID exceeds it.
fn instance_scalar(max_id: u64) -> crate::points::Scalar {
    if max_id > u16::MAX as u64 { crate::points::Scalar::U32 } else { crate::points::Scalar::U16 }
}

/// Per tile, the overlap boxes later tiles share with it: only points inside one can match a later tile's query.
fn later_overlaps(overlaps: &Overlaps) -> Vec<Vec<([f64; 2], [f64; 2])>> {
    let mut later = vec![Vec::new(); overlaps.len()];
    for (u, bounds) in overlaps.iter().enumerate() {
        for (&t, b) in bounds { if t < u { later[t].push(*b); } }
    }
    later
}

/// The 1 cm index of the points inside any overlap of their tile (the only competitors of shared points).
fn overlap_index(files: &[PathBuf], model: &Model, origin: &[f64; 3], overlaps: &Overlaps) -> Res<TileIndex> {
    let later = later_overlaps(overlaps);
    let keep = |tile: usize, p: &[f64; 3]| overlaps[tile].values().chain(&later[tile]).any(|b| inside_xy(p, b));
    Ok(build_index_where(files, model, origin, 0.01, Some(&keep), None)?.0)
}

/// Per tile of deduplication: input, survivors and up to five removals (point, survivor tile, survivor record, distance).
pub struct DedupTile { pub input: u64, pub surviving: u64, pub examples: Vec<(i64, u32, i64, f64)> }

/// `deduplicate`: tiles in order; a point leaves when an earlier tile's survivor
/// within 1 cm carries the same (dataset instance, semantic) labels.
#[allow(clippy::too_many_arguments)]
fn deduplicate(model: &Model, files: &[PathBuf], mapping: &BTreeMap<(usize, u64), u64>, output_dir: &Path, origin: &[f64; 3],
               overlaps: &Overlaps, statistics: &mut Statistics, regions: &[Region], stage_index: TileIndex,
               stage_semantics: Vec<u64>, pool: &rayon::ThreadPool, metrics: &mut Vec<DedupTile>,
               survivors: Option<&mut crate::parts::Parts>) -> Res<Vec<PathBuf>> {
    let mut timing = [0f64; 2]; // nearest + write, survivor index
    const RADIUS: f64 = 0.01;
    fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    let groups = mapping.values().collect::<std::collections::BTreeSet<_>>().len() as u64;
    let instance_type = instance_scalar(groups);
    // Each finished tile's survivors inside an overlap with a later tile (the only ones a later
    // query can accept), indexed once with their labels; `survivors` (when kept) gets every
    // tile's survivors as one part.
    let later = later_overlaps(overlaps);
    let mut tile_sets: Vec<(PointSet, Vec<(u64, u64)>)> = Vec::new();
    let mut survivors = survivors;
    let mut outputs = Vec::new();
    for (tile, file) in files.iter().enumerate() {
        let header = lasio::read_header(file)?;
        let layout = PointLayout::of(&header)?;
        let (instance_field, _) = layout.find(&model.instance).unwrap();
        let semantic_field = model.semantic.as_ref().map(|n| layout.find(n).unwrap().0);
        // `target_header(ready=True)`: the model dimensions replace the tile's, the instance as uint16/32.
        let mut target = TileHeader::copy_of(&header)?;
        target.vlrs.retain(|v| !(v.user() == "LASF_Spec" && v.record_id == 4));
        // Other ExtraBytes of the tile (e.g. of the 1 cm targets) stay first, unchanged.
        let names: Vec<&str> = model.dimensions.iter().map(|d| d.name.as_str()).collect();
        let kept: Vec<(usize, usize)> = layout.extra.iter().filter(|(e, _)| !names.contains(&e.name.as_str())).map(|(e, o)| (*o, e.size())).collect();
        let kept_width: usize = kept.iter().map(|k| k.1).sum();
        let kept_extra: Vec<OutputExtra> = target.extra.iter().filter(|e| !names.contains(&e.source.name.as_str())).cloned().collect();
        target.extra = kept_extra.into_iter().chain(model.dimensions.iter().map(|d| {
            if d.name == model.instance {
                let mut e = d.clone();
                e.data_type = instance_type.extra_bytes_id();
                e.options = 0;
                e.description = [0; 32];
                OutputExtra::new(&e)
            } else {
                OutputExtra::new(d)
            }
        })).collect();
        let base = crate::points::base_record_length(header.point_format);
        let source_fields: Vec<Layout> = model.dimensions.iter().map(|d| layout.find(&d.name).unwrap().0).collect();
        let mut out_layout = Vec::new();
        let mut at = base + kept_width;
        for e in &target.extra[kept.len()..] { out_layout.push(Layout::Bytes { offset: at, scalar: e.source.scalar().unwrap() }); at += e.source.size(); }
        let out_len = target.record_length();
        let bounds = &overlaps[tile];
        let mut offset = 0i64;
        let mut metric = DedupTile { input: 0, surviving: 0, examples: Vec::new() };
        let mut new_points: Vec<([f64; 3], i64, (u64, u64))> = Vec::new();
        let mut new_values: Vec<u8> = Vec::new();
        // `survivors.nearest(before_tile=tile, overlaps=overlaps[tile])`: only earlier tiles overlapping
        // this one can hold a duplicate. Within a tile the batches are independent (survivors join
        // after the tile), so they run in parallel and are merged in order.
        let earlier: Vec<usize> = (0..tile).filter(|t| bounds.contains_key(t)).collect();
        let sets: Vec<&PointSet> = earlier.iter().map(|&t| &tile_sets[t].0).collect();
        let allowed = |s: usize, id: u32, q: &[f64; 3]| {
            let b = &bounds[&earlier[s]];
            inside_xy(&sets[s].xyz[id as usize], b) && inside_xy(q, b)
        };
        let record_len = header.record_length as usize;
        struct Batch { out: Vec<u8>, kept_ids: Vec<u64>, kept_xyz: Vec<[f64; 3]>, points: Vec<([f64; 3], i64, (u64, u64))>, values: Vec<u8>,
                       examples: Vec<(i64, u32, i64, f64)>, input: u64, conflict: Option<String> }
        let process = |chunk: &[u8], offset: i64| -> Batch {
            let mut scratch = [0u8; 8];
            let records: Vec<&[u8]> = chunk.chunks_exact(record_len).collect();
            let xyz: Vec<[f64; 3]> = records.iter().map(|r| local(r, &header, origin)).collect();
            let labels: Vec<(u64, u64)> = records.iter().map(|r| {
                let l = instance_field.scalar().read_int(instance_field.get(r, &mut scratch)) as u64;
                let s = semantic_field.map(|f| f.scalar().read_int(f.get(r, &mut scratch)) as u64).unwrap_or(0);
                (if l == 0 { 0 } else { mapping[&(tile, l)] }, s)
            }).collect();
            let groups = crate::spatial::spatial_batches(&xyz);
            let mut found: Vec<Option<(usize, u32, f64)>> = vec![None; xyz.len()]; // (earlier tile, its survivor, distance)
            for g in &groups {
                // A group with no point inside an earlier overlap has no candidates.
                if !g.iter().any(|&i| earlier.iter().any(|t| inside_xy(&xyz[i], &bounds[t]))) { continue; }
                let pts: Vec<[f64; 3]> = g.iter().map(|&i| xyz[i]).collect();
                let near = crate::spatial::nearest_across(&sets, &pts, RADIUS, &allowed);
                for (&i, n) in g.iter().zip(near) { found[i] = n.map(|(s, id, d)| (earlier[s], id, d)); }
            }
            // `conflicting_match`: no earlier tile's record within 1 cm may carry other labels,
            // except background semantics and two trees each owning its own location.
            let conflict = groups.iter().find_map(|g| {
                let pts: Vec<[f64; 3]> = g.iter().map(|&i| xyz[i]).collect();
                let limit = RADIUS + crate::spatial::distance_roundoff(&pts);
                for (k, &i) in g.iter().enumerate() {
                    let q = &pts[k];
                    if !bounds.iter().any(|(t, b)| *t < tile && inside_xy(q, b)) { continue; }
                    let mut best: BTreeMap<(usize, (u64, u64)), (f64, u32)> = BTreeMap::new();
                    stage_index.set.within_pub(q, limit, |id, d| {
                        let t = stage_index.set.tile[id as usize] as usize;
                        if t >= tile { return; }
                        let Some(b) = bounds.get(&t) else { return };
                        if !inside_xy(q, b) || !inside_xy(&stage_index.set.xyz[id as usize], b) { return; }
                        let raw = stage_index.labels[id as usize];
                        let label = (if raw == 0 { 0 } else { mapping[&(t, raw)] }, stage_semantics[id as usize]);
                        let e = best.entry((t, label)).or_insert((f64::INFINITY, id));
                        if d < e.0 { *e = (d, id); }
                    });
                    for ((t, label), (_, id)) in best {
                        let mut disagrees = labels[i] != label;
                        if label.0 == 0 { disagrees &= labels[i].0 != 0; }
                        let other_pt = &stage_index.set.xyz[id as usize];
                        let separate = preferred_core(q, tile, t, regions, origin) && preferred_core(other_pt, t, tile, regions, origin);
                        disagrees &= !(labels[i].0 > 0 && label.0 > 0 && separate);
                        if disagrees {
                            return Some(format!("Unresolved cross-tile label conflict: tile {tile} point {} labels {:?} vs tile {t} labels {label:?}",
                                                offset + i as i64, labels[i]));
                        }
                    }
                }
                None
            });
            let mut b = Batch { out: Vec::with_capacity(records.len() * out_len), kept_ids: Vec::new(), kept_xyz: Vec::new(), points: Vec::new(),
                                values: Vec::new(), examples: Vec::new(), input: records.len() as u64, conflict };
            for (k, r) in records.iter().enumerate() {
                let duplicate = found[k].is_some_and(|(t, id, _)| tile_sets[t].1[id as usize] == labels[k]);
                if duplicate {
                    if b.examples.len() < 5 {
                        let (t, id, d) = found[k].unwrap();
                        b.examples.push((offset + k as i64, t as u32, tile_sets[t].0.index[id as usize], d));
                    }
                    continue;
                }
                let start = b.out.len();
                b.out.extend_from_slice(&r[..base]);
                for (o, size) in &kept { b.out.extend_from_slice(&r[*o..o + size]); }
                b.out.resize(start + out_len, 0);
                for (j, (src, dst)) in source_fields.iter().zip(&out_layout).enumerate() {
                    if model.dimensions[j].name == model.instance {
                        dst.set(&mut b.out[start..], &labels[k].0.to_le_bytes());
                    } else {
                        let v = src.get(r, &mut scratch).to_vec();
                        dst.set(&mut b.out[start..], &v);
                    }
                }
                b.points.push((xyz[k], offset + k as i64, labels[k]));
                b.values.extend_from_slice(&b.out[start + base + kept_width..start + out_len]);
                b.kept_ids.push(labels[k].0);
                b.kept_xyz.push(xyz[k]);
            }
            b
        };
        // Batches stream into the writer, so a tile's output is never held whole.
        let t_found = std::time::Instant::now();
        let mut source = blocks(file)?;
        let mut pending: std::collections::VecDeque<Batch> = std::collections::VecDeque::new();
        let mut failed = false;
        let stream = std::iter::from_fn(|| loop {
            if failed { return None; }
            if let Some(b) = pending.pop_front() {
                if let Some(conflict) = b.conflict { failed = true; return Some(Err(conflict)); }
                for e in b.examples { if metric.examples.len() < 5 { metric.examples.push(e); } }
                statistics.add(&b.kept_ids, &b.kept_xyz);
                metric.input += b.input;
                metric.surviving += b.kept_ids.len() as u64;
                offset += b.input as i64;
                new_points.extend(b.points);
                new_values.extend_from_slice(&b.values);
                return Some(Ok(b.out));
            }
            let block = match source.next()? {
                Ok(block) => block,
                Err(e) => { failed = true; return Some(Err(e)); }
            };
            let pieces: Vec<&[u8]> = block.raw_bytes().chunks(MAX_BATCH_POINTS as usize * record_len).collect();
            let base_offset = offset;
            pending.extend(pool.install(|| {
                use rayon::prelude::*;
                pieces.par_iter().enumerate().map(|(n, c)| process(c, base_offset + (n as i64) * MAX_BATCH_POINTS as i64)).collect::<Vec<Batch>>()
            }));
        });
        let output = output_dir.join(file.file_name().unwrap());
        if let Err(e) = lasio::write_records(&output, &target, stream, true, Some(pool)) {
            let _ = fs::remove_file(&output);
            return Err(e);
        }
        timing[0] += t_found.elapsed().as_secs_f64();
        let t_build = std::time::Instant::now();
        let mut set = (PointSet::new(RADIUS), Vec::new());
        for (p, index, labels) in &new_points {
            if later[tile].iter().any(|b| inside_xy(p, b)) {
                set.0.push(*p, tile as u32, *index);
                set.1.push(*labels);
            }
        }
        if let Some(parts) = survivors.as_deref_mut() {
            let xyz: Vec<[f64; 3]> = new_points.iter().map(|p| p.0).collect();
            let index: Vec<i64> = new_points.iter().map(|p| p.1).collect();
            let labels: Vec<u64> = new_points.iter().map(|p| p.2 .0).collect();
            drop(new_points);
            parts.add(tile as u32, &xyz, &index, &labels, &new_values)?;
        }
        set.0.build();
        tile_sets.push(set);
        timing[1] += t_build.elapsed().as_secs_f64();
        outputs.push(output);
        metrics.push(metric);
    }
    if std::env::var_os("SMARTTILE_TIMINGS").is_some() {
        eprintln!("    dedup: nearest + write {:.1} s, survivor index {:.1} s", timing[0], timing[1]);
    }
    Ok(outputs)
}

/// Raw records of a file, chunk by chunk, with a field reader for the instance label.
fn instance_reader(file: &Path, name: &str) -> Res<(SourceHeader, Layout)> {
    let header = lasio::read_header(file)?;
    let layout = PointLayout::of(&header)?;
    let (field, _) = layout.find(name).ok_or_else(|| format!("{}: no {name}", file.display()))?;
    Ok((header, field))
}

/// `consolidate_instances`: each positive instance moves whole into one owner tile
/// (core holding its XY centroid, else most points; ties: lowest tile). Files are rewritten in place.
fn consolidate(files: &[PathBuf], instance: &str, regions: &[Region], work: &Path, pool: &rayon::ThreadPool) -> Res<(usize, u64, Vec<(String, u64)>)> {
    // `instance_tiles`: points per tile and world XY sums per instance.
    let mut counts: BTreeMap<u64, BTreeMap<usize, u64>> = BTreeMap::new();
    let mut sums: BTreeMap<u64, [f64; 2]> = BTreeMap::new();
    let mut scratch = [0u8; 8];
    // Per tile (read in parallel), per chunk: the instance batches; summed below in tile and chunk order.
    type Batches = Vec<BTreeMap<u64, (u64, f64, f64)>>;
    let per_tile: Vec<Res<Batches>> = pool.install(|| files.par_iter().map(|file| {
        let (header, field) = instance_reader(file, instance)?;
        let mut scratch = [0u8; 8];
        let mut out = Vec::new();
        for chunk in chunks(file)? {
            let chunk = chunk?;
            let mut batch: BTreeMap<u64, (u64, f64, f64)> = BTreeMap::new();
            for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
                let id = field.scalar().read_int(field.get(r, &mut scratch)) as u64;
                if id == 0 { continue; }
                let x = header.scales[0] * f64::from(i32::from_le_bytes(r[0..4].try_into().unwrap())) + header.offsets[0];
                let y = header.scales[1] * f64::from(i32::from_le_bytes(r[4..8].try_into().unwrap())) + header.offsets[1];
                let e = batch.entry(id).or_insert((0, 0.0, 0.0));
                e.0 += 1; e.1 += x; e.2 += y;
            }
            out.push(batch);
        }
        Ok(out)
    }).collect());
    for (tile, batches) in per_tile.into_iter().enumerate() {
        for batch in batches? {
            for (id, (n, x, y)) in batch {
                *counts.entry(id).or_default().entry(tile).or_default() += n;
                let t = sums.entry(id).or_insert([0.0, 0.0]);
                t[0] += x; t[1] += y;
            }
        }
    }
    // `plan_owners`
    let mut owners: BTreeMap<u64, usize> = BTreeMap::new();
    for (id, per_tile) in &counts {
        if per_tile.len() < 2 { continue; }
        let n: u64 = per_tile.values().sum();
        let (cx, cy) = (sums[id][0] / n as f64, sums[id][1] / n as f64);
        let inside = per_tile.keys().find(|&&t| regions[t].core[0][0] <= cx && cx <= regions[t].core[0][1] && regions[t].core[1][0] <= cy && cy <= regions[t].core[1][1]);
        let owner = match inside {
            Some(t) => *t,
            None => { let best = per_tile.values().max().unwrap(); *per_tile.iter().find(|(_, v)| *v == best).unwrap().0 }
        };
        owners.insert(*id, owner);
    }
    if owners.is_empty() {
        return Ok((0, 0, Vec::new()));
    }
    let mut per_owner: Vec<(String, u64)> = Vec::new();
    let headers: Vec<SourceHeader> = files.iter().map(|f| lasio::read_header(f)).collect::<Res<_>>()?;
    let mut incoming: BTreeMap<usize, Vec<Vec<u8>>> = BTreeMap::new();
    let mut moved = 0u64;
    // Pass 1 (tiles in parallel): each tile without its departing points; departing points in the
    // owner's encoding, per chunk in owner order.
    type Departures = Vec<(usize, Vec<u8>, u64)>;
    let pass1: Vec<Res<(Departures, u64)>> = pool.install(|| files.par_iter().enumerate().map(|(tile, file)| {
        let moving: std::collections::BTreeSet<u64> = owners.iter().filter(|(id, o)| **o != tile && counts[*id].contains_key(&tile)).map(|(id, _)| *id).collect();
        if moving.is_empty() { return Ok((Vec::new(), 0)); }
        let (header, field) = instance_reader(file, instance)?;
        let target = TileHeader::copy_of(&header)?;
        let mut scratch = [0u8; 8];
        let mut departing: Departures = Vec::new();
        let mut moved = 0u64;
        // Kept records stream into the rewritten tile, chunk by chunk.
        let batches = chunks(file)?.map(|chunk| -> Res<Vec<u8>> {
            let chunk = chunk?;
            let mut kept = Vec::new();
            let mut parts: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
            for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
                let id = field.scalar().read_int(field.get(r, &mut scratch)) as u64;
                if !moving.contains(&id) { kept.extend_from_slice(r); continue; }
                let owner = owners[&id];
                let h = &headers[owner];
                let mut moved_record = r.to_vec();
                for a in 0..3 {
                    let world = header.scales[a] * f64::from(i32::from_le_bytes(r[a * 4..a * 4 + 4].try_into().unwrap())) + header.offsets[a];
                    let encoded = ((world - h.offsets[a]) / h.scales[a]).round_ties_even();
                    if encoded < i32::MIN as f64 || encoded > i32::MAX as f64 {
                        return Err("Moved instance points exceed the LAS int32 range of their owner tile".to_string());
                    }
                    moved_record[a * 4..a * 4 + 4].copy_from_slice(&(encoded as i32).to_le_bytes());
                }
                parts.entry(owner).or_default().extend_from_slice(&moved_record);
                moved += 1;
            }
            for (owner, bytes) in parts {
                let n = (bytes.len() / chunk.record_len()) as u64;
                departing.push((owner, bytes, n));
            }
            Ok(kept)
        });
        let temporary = file.with_extension("consolidating.laz");
        if let Err(e) = lasio::write_records(&temporary, &target, batches, true, None) {
            let _ = fs::remove_file(&temporary);
            return Err(e);
        }
        fs::rename(&temporary, file).map_err(|e| e.to_string())?;
        Ok((departing, moved))
    }).collect());
    for result in pass1 {
        let (departing, n) = result?;
        moved += n;
        for (owner, bytes, count) in departing {
            let key = files[owner].file_name().unwrap().to_string_lossy().into_owned();
            match per_owner.iter_mut().find(|(k, _)| *k == key) { Some(e) => e.1 += count, None => per_owner.push((key, count)) }
            incoming.entry(owner).or_default().push(bytes);
        }
    }
    let _ = &mut scratch;
    // Pass 2 (owners in parallel): each owner's own points, then its incoming points (32,768-point chunks).
    let pass2: Vec<Res<()>> = pool.install(|| incoming.into_par_iter().map(|(owner, parts)| {
        let file = &files[owner];
        let target = TileHeader::copy_of(&headers[owner])?;
        let record_len = target.record_length();
        let all: Vec<u8> = parts.concat();
        let batches = chunks(file)?.map(|chunk| chunk.map(|c| c.raw_bytes().to_vec()))
            .chain(all.chunks(MAX_BATCH_POINTS as usize * record_len).map(|chunk| Ok(chunk.to_vec())));
        let temporary = file.with_extension("consolidating.laz");
        lasio::write_records(&temporary, &target, batches, true, None)?;
        fs::rename(&temporary, file).map_err(|e| e.to_string())
    }).collect());
    pass2.into_iter().collect::<Res<Vec<()>>>()?;
    let _ = work;
    Ok((owners.len(), moved, per_owner))
}

/// `report["parallelism"][key] = value` on a model report.
fn set_parallelism(report: &mut Map<String, Value>, key: &str, value: Value) {
    report.entry("parallelism").or_insert_with(|| Value::Object(Map::new())).as_object_mut().unwrap().insert(key.into(), value);
}

/// `time.process_time()`: user plus system CPU seconds of this process (Linux /proc).
fn process_cpu_seconds() -> f64 {
    let stat = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<&str> = stat.rsplit(')').next().unwrap_or("").split_whitespace().collect();
    let ticks = |i: usize| fields.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (ticks(11) + ticks(12)) / 100.0
}

fn timed<T>(name: &str, model: &Model, report: &mut Map<String, Value>, f: impl FnOnce() -> Res<T>) -> Res<T> {
    let started = std::time::Instant::now();
    let result = f()?;
    release_free_memory();
    let seconds = started.elapsed().as_secs_f64();
    println!("{}: {name} finished in {seconds:.3}s", model.name);
    if std::env::var_os("SMARTTILE_TIMINGS").is_some() { eprintln!("    resident after {name}: {} MiB", resident_mib()); }
    report.entry("phase_seconds").or_insert_with(|| Value::Object(Map::new())).as_object_mut().unwrap()
        .insert(name.into(), crate::pyfmt::float(seconds));
    Ok(result)
}

/// Hands freed heap pages back to the system after a stage (glibc keeps them per thread arena),
/// so the resident size follows the live data.
pub fn release_free_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        extern "C" { fn malloc_trim(pad: usize) -> i32; }
        // SAFETY: glibc's malloc_trim only releases free memory and takes no pointers.
        unsafe { malloc_trim(0); }
    }
}

/// The process's resident anonymous memory (`RssAnon`), for the timing log.
pub fn resident_mib() -> u64 {
    fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("RssAnon:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())).unwrap_or(0) / 1024
}

fn write_json(path: &Path, value: &Value) -> Res<()> {
    fs::write(path, crate::pyfmt::dumps(value)).map_err(|e| format!("{}: {e}", path.display()))
}

/// `instance_summary` (no reassignment aliases): one row per final instance, in the source CRS.
fn summary_document(model: &Model, statistics: &Statistics, origin: &[f64; 3], resolution_1: f64,
                    aliases: &BTreeMap<u64, u64>, reassignment_enabled: bool) -> Value {
    use crate::pyfmt::float;
    // `instance_summary`: sources fold into their targets (in ID order).
    let mut folded: BTreeMap<u64, (u64, [f64; 3], [f64; 3], [f64; 3])> = BTreeMap::new();
    for (id, (n, lo, hi, sum)) in &statistics.rows {
        let e = folded.entry(*aliases.get(id).unwrap_or(id)).or_insert((0, [f64::INFINITY; 3], [f64::NEG_INFINITY; 3], [0.0; 3]));
        e.0 += n;
        for a in 0..3 { e.1[a] = e.1[a].min(lo[a]); e.2[a] = e.2[a].max(hi[a]); e.3[a] += sum[a]; }
    }
    let instances: Vec<Value> = folded.iter().map(|(id, (n, lo, hi, sum))| {
        let from: Vec<u64> = aliases.iter().filter(|(_, t)| *t == id).map(|(s, _)| *s).collect();
        let volume = ((hi[0] - lo[0]) * (hi[1] - lo[1])) * (hi[2] - lo[2]);
        let centroid: [f64; 3] = std::array::from_fn(|a| sum[a] / *n as f64 + origin[a]);
        serde_json::json!({
            "id": id, "points": n,
            "bbox_min": lo.iter().zip(origin).map(|(l, o)| float(l + o)).collect::<Vec<_>>(),
            "bbox_max": hi.iter().zip(origin).map(|(h, o)| float(h + o)).collect::<Vec<_>>(),
            "bbox_volume_m3": float(volume),
            "centroid": centroid.iter().map(|c| float(*c)).collect::<Vec<_>>(),
            "centroid_xy": [float(sum[0] / *n as f64 + origin[0]), float(sum[1] / *n as f64 + origin[1])],
            "reassigned_from": from})
    }).collect();
    serde_json::json!({"model": model.name, "instance_dimension": model.instance, "resolution_1_m": float(resolution_1),
                       "coordinates": "source CRS", "id_encoding": "reconciled dataset ID",
                       "small_instance_reassignment": reassignment_enabled, "instances": instances})
}

/// `plan_reassignment`: small instances (fewer points and a smaller box) take the ID of the
/// non-small instance with the nearest XY centroid within 5 m. Returns the mapping and its report.
fn plan_reassignment(statistics: &Statistics, max_cluster_size: u64, max_volume: f64) -> (BTreeMap<u64, u64>, Map<String, Value>) {
    const SEARCH_RADIUS: f64 = 5.0;
    let f = crate::pyfmt::float;
    let rows: Vec<(u64, u64, f64, [f64; 2])> = statistics.rows.iter().map(|(id, (n, lo, hi, sum))| {
        (*id, *n, ((hi[0] - lo[0]) * (hi[1] - lo[1])) * (hi[2] - lo[2]), [sum[0] / *n as f64, sum[1] / *n as f64])
    }).collect();
    let small: Vec<bool> = rows.iter().map(|r| r.1 < max_cluster_size && r.2 < max_volume).collect();
    let targets: Vec<usize> = (0..rows.len()).filter(|&i| !small[i]).collect();
    let (mut mapping, mut reassigned, mut kept) = (BTreeMap::new(), Vec::new(), Vec::new());
    if small.iter().any(|s| *s) && !targets.is_empty() {
        for (i, row) in rows.iter().enumerate().filter(|(i, _)| small[*i]) {
            let _ = i;
            // cKDTree nearest: the smallest distance, ties to the first target.
            let (distance, target) = targets.iter().map(|&t| {
                let (dx, dy) = (rows[t].3[0] - row.3[0], rows[t].3[1] - row.3[1]);
                (((0.0 + dx * dx) + dy * dy).sqrt(), t)
            }).fold((f64::INFINITY, usize::MAX), |best, c| if c.0 < best.0 { c } else { best });
            let mut decision = serde_json::json!({"instance": row.0, "points": row.1, "bbox_volume_m3": f(row.2), "xy_distance_m": f(distance)});
            if distance <= SEARCH_RADIUS {
                mapping.insert(row.0, rows[target].0);
                decision["target"] = serde_json::json!(rows[target].0);
                reassigned.push(decision);
            } else {
                decision["reason"] = serde_json::json!(format!("no non-small instance within {} m", crate::pyfmt::float_repr(SEARCH_RADIUS)));
                kept.push(decision);
            }
        }
    } else {
        kept = rows.iter().zip(&small).filter(|(_, s)| **s).map(|(r, _)| serde_json::json!({"instance": r.0, "reason": "no non-small instance"})).collect();
    }
    let moved: u64 = rows.iter().filter(|r| mapping.contains_key(&r.0)).map(|r| r.1).sum();
    let mut report = Map::new();
    report.insert("policy".into(), "fewer than max_cluster_size points and bounding box below max_volume_m3; nearest non-small instance XY centroid (horizontal distance) within search_radius_m".into());
    report.insert("max_cluster_size".into(), serde_json::json!(max_cluster_size));
    report.insert("max_volume_m3".into(), f(max_volume));
    report.insert("search_radius_m".into(), f(SEARCH_RADIUS));
    report.insert("instances".into(), serde_json::json!(rows.len()));
    report.insert("small".into(), serde_json::json!(small.iter().filter(|s| **s).count()));
    report.insert("reassigned_count".into(), serde_json::json!(mapping.len()));
    report.insert("reassigned_points".into(), serde_json::json!(moved));
    reassigned.truncate(1000);
    kept.truncate(1000);
    report.insert("reassigned".into(), Value::Array(reassigned));
    report.insert("kept".into(), Value::Array(kept));
    (mapping, report)
}

/// `relabel_tiles`: instance IDs rewritten in place (32,768-point batches); relabeled points per tile.
fn relabel_tiles(files: &[PathBuf], instance: &str, mapping: &BTreeMap<u64, u64>) -> Res<Vec<u64>> {
    let mut changed = Vec::new();
    for file in files {
        let (header, field) = instance_reader(file, instance)?;
        let target = TileHeader::copy_of(&header)?;
        let mut points = 0u64;
        let mut batches = Vec::new();
        let mut scratch = [0u8; 8];
        for chunk in chunks(file)? {
            let chunk = chunk?;
            let mut out = chunk.raw_bytes().to_vec();
            for r in out.chunks_exact_mut(chunk.record_len()) {
                let id = field.scalar().read_int(field.get(r, &mut scratch)) as u64;
                if let Some(to) = mapping.get(&id) {
                    field.set(r, &to.to_le_bytes());
                    points += 1;
                }
            }
            batches.push(Ok(out));
        }
        let temporary = file.with_file_name(format!(".{}.relabel", file.file_name().unwrap().to_string_lossy()));
        lasio::write_records(&temporary, &target, batches, true, None)?;
        fs::rename(&temporary, file).map_err(|e| e.to_string())?;
        changed.push(points);
    }
    Ok(changed)
}

/// `merge_file`: the final tiles concatenated in tile order, in the first tile's encoding.
fn merge_file(files: &[PathBuf], output: &Path, pool: &rayon::ThreadPool) -> Res<()> {
    let first = lasio::read_header(&files[0])?;
    let target = TileHeader::copy_of(&first)?;
    // Tiles decode in parallel; their 32,768-point batches are written in tile order.
    let per_file: Vec<Res<Vec<Vec<u8>>>> = pool.install(|| files.par_iter().map(|file| {
        let h = lasio::read_header(file)?;
        if h.point_format != first.point_format || h.record_length != first.record_length {
            return Err(format!("Cannot concatenate incompatible target schemas: {}", file.display()));
        }
        let mut out = Vec::new();
        for chunk in chunks(file)? {
            let chunk = chunk?;
            let mut bytes = chunk.raw_bytes().to_vec();
            for r in bytes.chunks_exact_mut(chunk.record_len()) {
                for a in 0..3 {
                    // laspy change_scaling: round((X * scale + offset - new offset) / new scale).
                    let world = h.scales[a] * f64::from(i32::from_le_bytes(r[a * 4..a * 4 + 4].try_into().unwrap())) + h.offsets[a];
                    let encoded = ((world - first.offsets[a]) / first.scales[a]).round_ties_even() as i32;
                    r[a * 4..a * 4 + 4].copy_from_slice(&encoded.to_le_bytes());
                }
            }
            out.push(bytes);
        }
        Ok(out)
    }).collect());
    let mut batches: Vec<Res<Vec<u8>>> = Vec::new();
    for part in per_file { batches.extend(part?.into_iter().map(Ok)); }
    lasio::write_records(output, &target, batches, true, Some(pool))?;
    Ok(())
}

fn available_destination(path: &Path) -> Res<()> {
    if path.exists() && (!path.is_dir() || fs::read_dir(path).map(|mut d| d.next().is_some()).unwrap_or(true)) {
        return Err(format!("Refusing to mix this run with existing output: {}", path.display()));
    }
    Ok(())
}

/// One model's merge (`merge_collections` loop body): dense transfer through consolidation and
/// the metadata beside its final tiles, under `scratch/<stage>/<suffix>`. Fills `report` as it goes.
#[allow(clippy::too_many_arguments)]
fn merge_model(params: &MergeParams, collection: &Path, suffix: &str, layout: &Value, origin: [f64; 3], scratch: &Path,
               pool: &rayon::ThreadPool, report: &mut Map<String, Value>) -> Res<ModelOutcome> {
    let files = crate::tile::point_cloud_files(collection)?;
    if files.is_empty() {
        return Err(format!("No predictions in {}", collection.display()));
    }
    let headers = files.iter().map(|f| lasio::read_header(f)).collect::<Res<Vec<_>>>()?;
    let model = describe_model(&files, &headers, collection, &params.instance_dimension)?;
    let (pairs, labels) = match &params.target {
        Some(target) => pairs(collection, target, layout)?,
        None => ready_pairs(collection, layout)?,
    };
    let qw = params.workers.max(1);
    let n = pairs.len();
    let f = crate::pyfmt::float;
    report.insert("model".into(), Value::String(model.name.clone()));
    report.insert("source".into(), Value::String(collection.display().to_string()));
    report.insert("instance_dimension".into(), Value::String(model.instance.clone()));
    report.insert("semantic_dimension".into(), model.semantic.clone().map(Value::String).unwrap_or(Value::Null));
    report.insert("tile_sources".into(), Value::Array(pairs.iter().enumerate().map(|(n, p)| serde_json::json!({
        "tile": n, "prediction": p.prediction.display().to_string(), "geometry": p.target.display().to_string()})).collect()));
    // `query_process_shares`: query processes per tile in proportion to its 1 cm points.
    let sizes: Vec<u64> = pairs.iter().map(|p| lasio::read_header(&p.target).map(|h| h.point_count)).collect::<Res<_>>()?;
    let total: u64 = sizes.iter().sum::<u64>().max(1);
    let shares: Vec<u64> = sizes.iter().map(|s| if qw == 1 || params.target.is_none() { 1 } else { 1u64.max(py_round(qw as f64 * *s as f64 / total as f64) as u64) }).collect();
    let mut parallelism = Map::new();
    parallelism.insert("dense_transfer_processes".into(), serde_json::json!(qw.min(n)));
    parallelism.insert("dense_query_processes".into(), serde_json::json!(shares));
    report.insert("parallelism".into(), Value::Object(parallelism));
    report.insert("transfer".into(), Value::Null);
    let dense = scratch.join("dense").join(suffix);
    fs::create_dir_all(&dense).map_err(|e| e.to_string())?;
    let dense_files: Vec<PathBuf> = (0..pairs.len()).map(|n| dense.join(format!("tile_{n:05}.laz"))).collect();
    let transferred = timed("dense_transfer_and_index", &model, report, || {
        let results: Vec<Res<(u64, u64)>> = pool.install(|| pairs.par_iter().enumerate()
            .map(|(n, p)| if params.target.is_none() { ready_tile(&model, n, p, &dense) }
                          else { transfer_tile(&model, n, p, &dense, &origin, params.transfer_radius, pool) }).collect());
        results.into_iter().collect::<Res<Vec<_>>>()
    })?;
    report.insert("transfer".into(), Value::Array(pairs.iter().zip(&transferred).enumerate().map(|(n, (p, (total, matched)))| serde_json::json!({
        "source": p.prediction.display().to_string(), "target": p.target.display().to_string(), "tile": n,
        "total": total, "matched": matched, "missing_examples": [],
            "radius_m": if params.target.is_none() { Value::Null } else { f(params.transfer_radius) }})).collect()));
    let regions = regions(&pairs, layout)?;
    let overlaps = tile_overlaps(&pairs, layout, &origin);
    let owned = scratch.join("owned").join(suffix);
    fs::create_dir_all(&owned).map_err(|e| e.to_string())?;
    let owned_files: Vec<PathBuf> = (0..pairs.len()).map(|n| owned.join(format!("tile_{n:05}.laz"))).collect();
    report.insert("instance_ownership".into(), Value::Null);
    set_parallelism(report, "core_filter_processes", serde_json::json!(qw.min(n)));
    let ownership = timed("core_filter_and_index", &model, report, || {
        let results: Vec<Res<Ownership>> = pool.install(|| (0..pairs.len()).into_par_iter()
            .map(|n| filter_tile(&model, &dense_files[n], &regions[n], &owned_files[n], &origin, params.anchor)).collect());
        results.into_iter().collect::<Res<Vec<_>>>()
    })?;
    // `instance_ownership`: provisional decisions per tile; recovery revises them below.
    let mut tiles_json: Vec<Map<String, Value>> = ownership.iter().enumerate().map(|(tile, o)| {
        let mut m = regions[tile].json();
        let prediction = pairs[tile].prediction.display().to_string();
        let removed = o.decisions.iter().filter(|d| !d.kept).count();
        for (k, v) in [("tile", serde_json::json!(tile)), ("input", serde_json::json!(o.input)), ("surviving", serde_json::json!(o.surviving)),
                       ("removed", serde_json::json!(o.input - o.surviving)), ("background_input", serde_json::json!(o.background_input)),
                       ("background_removed", serde_json::json!(o.background_removed)), ("kept_instances", serde_json::json!(o.decisions.len() - removed)),
                       ("removed_instances", serde_json::json!(removed))] {
            m.insert(k.into(), v);
        }
        m.insert("instances".into(), Value::Array(o.decisions.iter().map(|d| serde_json::json!({
            "instance": d.instance, "points": d.points, "anchor_xy": [f(d.anchor[0]), f(d.anchor[1])], "anchor_xyz": d.anchor.map(f),
            "bbox_xyz": [d.lower.map(f), d.upper.map(f)], "source_prediction": prediction, "normal_kept": d.kept, "kept": d.kept,
            "disposition": if d.kept { "normal_owner" } else { "rejected_buffer" }})).collect()));
        m
    }).collect();
    report.insert("orphan_recovery".into(), Value::Null);
    let recovered = scratch.join("recovered").join(suffix);
    let mut orphan = OrphanReport::default();
    let (admitted, current, revised) = timed("orphan_recovery", &model, report, || {
        let started = std::time::Instant::now();
        // Only owned points near a rejected instance can cover its samples.
        let near = rejected_cells(&ownership, &origin);
        let keep = |_: usize, p: &[f64; 3]| near.contains(&(p[0].floor() as i64, p[1].floor() as i64));
        let owned_index = build_index_where(&owned_files, &model, &origin, 0.01, Some(&keep), None)?.0;
        let t_index = started.elapsed().as_secs_f64();
        let admitted = orphan_recovery(&model, &dense_files, &ownership, &owned_index, &regions, &overlaps, &origin, &mut orphan)?;
        fs::create_dir_all(&recovered).map_err(|e| e.to_string())?;
        let mut by_tile: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
        for (tile, id) in &admitted { by_tile.entry(*tile).or_default().push(*id); }
        let rewritten: Vec<Res<(usize, u64)>> = pool.install(|| by_tile.par_iter().map(|(tile, ids)| {
            let rejected: std::collections::BTreeSet<u64> = ownership[*tile].removed.iter().copied().filter(|r| !ids.contains(r)).collect();
            Ok((*tile, rewrite_recovered(&model, &dense_files[*tile], &rejected, &regions[*tile], &recovered.join(format!("tile_{tile:05}.laz")), &origin)?))
        }).collect());
        let revised: BTreeMap<usize, u64> = rewritten.into_iter().collect::<Res<_>>()?;
        let current: Vec<PathBuf> = (0..pairs.len()).map(|n| if by_tile.contains_key(&n) { recovered.join(format!("tile_{n:05}.laz")) } else { owned_files[n].clone() }).collect();
        if std::env::var_os("SMARTTILE_TIMINGS").is_some() { eprintln!("    orphan: owned index {t_index:.1} s, total {:.1} s", started.elapsed().as_secs_f64()); }
        Ok((admitted, current, revised))
    })?;
    for (tile, id) in &admitted {
        let m = &mut tiles_json[*tile];
        if let Some(d) = m["instances"].as_array_mut().unwrap().iter_mut().find(|d| d["instance"] == serde_json::json!(id)) {
            d["kept"] = Value::Bool(true);
            d["disposition"] = Value::String("recovered_orphan".into());
            d.as_object_mut().unwrap().insert("recovered".into(), Value::Bool(true));
        }
    }
    for (tile, count) in &revised {
        let ids: Vec<u64> = { let mut v: Vec<u64> = admitted.iter().filter(|(t, _)| t == tile).map(|(_, i)| *i).collect(); v.sort(); v };
        let m = &mut tiles_json[*tile];
        let input = m["input"].as_u64().unwrap();
        m["surviving"] = serde_json::json!(count);
        m["removed"] = serde_json::json!(input - count);
        m["kept_instances"] = serde_json::json!(m["kept_instances"].as_u64().unwrap() + ids.len() as u64);
        m["removed_instances"] = serde_json::json!(m["removed_instances"].as_u64().unwrap() - ids.len() as u64);
        m.insert("recovered_instances".into(), serde_json::json!(ids));
    }
    report.insert("instance_ownership".into(), serde_json::json!({"anchor": params.anchor.name(), "geometry": "remapped_1cm",
        "background": "spatial core owner; half-open east/north edges", "semantics": "preserve per-point values from owning tile",
        "boundary": "inclusive core; outer edges retained", "tiles": tiles_json}));
    let mut orphan_json = serde_json::json!({
        "policy": "whole rejected instance only for unsupported positive core geometry", "support_radius_m": 0.01,
        "tie_break": "descending new distinct XYZ core samples, then source tile and local ID",
        "candidates": orphan.candidates, "uncovered_locations": orphan.uncovered,
        "admitted": orphan.admitted.iter().map(|(t, i, score)| serde_json::json!({"tile": t, "local_instance": i, "new_locations": score,
            "source": pairs[*t].prediction.display().to_string()})).collect::<Vec<_>>(),
        "final_support": if admitted.is_empty() { "no recovery needed" } else { "pending" },
        "claim_seconds": 0.0, "selection_checked_locations": orphan.checked, "selection_seconds": 0.0});
    report.insert("orphan_recovery".into(), orphan_json.clone());
    let mut counts: BTreeMap<(usize, u64), u64> = BTreeMap::new();
    for (tile, o) in ownership.iter().enumerate() { for (id, n) in &o.counts { counts.insert((tile, *id), *n); } }
    let normal_keys: std::collections::BTreeSet<(usize, u64)> = counts.keys().copied().collect();
    for (tile, id) in &admitted { counts.insert((*tile, *id), ownership[*tile].sizes[id]); }
    report.insert("reconciliation".into(), Value::Null);
    if params.matching { set_parallelism(report, "reconciliation_processes", serde_json::json!(qw.min(n))); }
    let mut reconciliation = ReconcileReport::default();
    let mapping = timed("reconciliation", &model, report, || {
        // A match needs a point inside an overlap with a later tile.
        let later = later_overlaps(&overlaps);
        let keep = |tile: usize, p: &[f64; 3]| later[tile].iter().any(|b| inside_xy(p, b));
        let index = build_index_where(&current, &model, &origin, 0.05, Some(&keep), None)?.0;
        reconcile(&model, &current, &index, &origin, &counts, params.overlap_threshold, 0.05, &overlaps,
                  (!admitted.is_empty()).then_some(&normal_keys), &pool, &mut reconciliation, params.matching)
    })?;
    let ids: std::collections::BTreeSet<u64> = mapping.values().copied().collect();
    let pair = |e: &(usize, u64, usize, u64, u64)| serde_json::json!({"tiles": [e.0, e.2], "instances": [e.1, e.3], "matches": e.4});
    let c = reconciliation.counters;
    report.insert("reconciliation".into(), serde_json::json!({"radius_m": 0.05, "overlap_threshold": f(params.overlap_threshold),
        "groups": ids.len(), "accepted_pairs": reconciliation.accepted.iter().map(pair).collect::<Vec<_>>(),
        "rejected_recovery_bridge_count": reconciliation.bridge_count,
        "rejected_recovery_bridges": reconciliation.bridges.iter().map(|e| { let mut v = pair(e); v["reason"] = "distinct retained groups".into(); v }).collect::<Vec<_>>(),
        "ids": mapping.iter().map(|((t, l), id)| serde_json::json!([t, l, id])).collect::<Vec<_>>(),
        "files_read": c[0], "points_read": c[1], "query_points": c[2], "skipped_query_points": c[3]}));
    report.insert("semantic_ownership".into(), serde_json::json!({
        "policy": "preserve every member's unique points; shared points use nearest-core tile ownership",
        "scope": "per point, including members of the same reconciled instance", "tie_break": "stable source tile filename order"}));
    let shared_json = |background_only: bool, tiles: &[SharedTile]| serde_json::json!({
        "radius_m": 0.01,
        "policy": if background_only { "retained tree wins over background, preserving its attributes" }
                  else { "all retained tree claims, including merged members: nearest claimant core in XY" },
        "tie_break": "stable source tile filename order", "boundary": "closed tree cores; preserve unshared buffer points",
        "tiles": tiles.iter().enumerate().map(|(tile, t)| serde_json::json!({"tile": tile, "removed": t.removed,
            "removal_examples": t.examples.iter().map(|(p, o, r, d)| serde_json::json!({"point": p, "owner": [o, r], "distance_m": f(*d)})).collect::<Vec<_>>(),
            "query_points": t.query_points, "query_blocks": t.query_blocks})).collect::<Vec<_>>()});
    let mut files = current;
    if files.len() > 1 {
        let resolved = scratch.join("resolved").join(suffix);
        report.insert("shared_point_ownership".into(), Value::Null);
        set_parallelism(report, "shared_point_ownership_processes", serde_json::json!(qw.min(n)));
        let stage = timed("shared_point_ownership", &model, report, || {
            let index = overlap_index(&files, &model, &origin, &overlaps)?;
            assign_shared_points(&model, &files, &index, &regions, &mapping, &resolved, &origin, &overlaps, false, &pool)
        })?;
        let (next, tiles): (Vec<PathBuf>, Vec<SharedTile>) = stage.into_iter().unzip();
        report.insert("shared_point_ownership".into(), shared_json(false, &tiles));
        files = next;
        if ownership.iter().any(|o| o.background_input > o.background_removed) {
            let priority = scratch.join("tree_priority").join(suffix);
            report.insert("tree_background_ownership".into(), Value::Null);
            set_parallelism(report, "tree_background_ownership_processes", serde_json::json!(qw.min(n)));
            let stage = timed("tree_background_ownership", &model, report, || {
                let index = overlap_index(&files, &model, &origin, &overlaps)?;
                assign_shared_points(&model, &files, &index, &regions, &mapping, &priority, &origin, &overlaps, true, &pool)
            })?;
            let (next, tiles): (Vec<PathBuf>, Vec<SharedTile>) = stage.into_iter().unzip();
            report.insert("tree_background_ownership".into(), shared_json(true, &tiles));
            files = next;
        }
    }
    let final_dir = scratch.join("final").join(suffix);
    let mut statistics = Statistics::default();
    report.insert("deduplication".into(), Value::Null);
    set_parallelism(report, "deduplication_query_processes", serde_json::json!(qw));
    let parallel_tiles: Vec<usize> = files.iter().enumerate().filter(|(tile, file)| qw > 1 && *tile > 0
        && lasio::read_header(file).map(|h| h.point_count > 2 * MAX_BATCH_POINTS).unwrap_or(false)).map(|(t, _)| t).collect();
    set_parallelism(report, "deduplication_parallel_tiles", serde_json::json!(parallel_tiles));
    let mut dedup = Vec::new();
    // The survivors serve the recovered-geometry check and the originals (when kept), as one
    // part per tile indexed at the originals' radius (instance as label, model values as rows).
    let keep_survivors = params.originals.is_some() || !admitted.is_empty();
    let survivor_width: usize = model.dimensions.iter().map(|d| if d.name == model.instance { instance_scalar(ids.len() as u64).size() } else { d.size() }).sum();
    let mut survivors = crate::parts::Parts::new(&scratch.join("survivors").join(suffix), params.remap_tolerance.unwrap_or(3f64.sqrt() * params.resolution_1),
                                                 survivor_width, true)?;
    let final_files = timed("deduplication", &model, report, || {
        let t_index = std::time::Instant::now();
        // Only points inside an overlap with a later tile can be a candidate.
        let later = later_overlaps(&overlaps);
        let keep = |tile: usize, p: &[f64; 3]| later[tile].iter().any(|b| inside_xy(p, b));
        let (stage_index, semantics) = build_index_where(&files, &model, &origin, 0.01, Some(&keep), model.semantic.as_deref())?;
        if std::env::var_os("SMARTTILE_TIMINGS").is_some() { eprintln!("    dedup: stage index {:.1} s", t_index.elapsed().as_secs_f64()); }
        deduplicate(&model, &files, &mapping, &final_dir, &origin, &overlaps, &mut statistics, &regions, stage_index, semantics, &pool,
                    &mut dedup, keep_survivors.then_some(&mut survivors))
    })?;
    report.insert("deduplication".into(), serde_json::json!({"radius_m": 0.01,
        "tiles": dedup.iter().enumerate().map(|(tile, t)| serde_json::json!({"tile": tile, "input": t.input, "surviving": t.surviving,
            "removed": t.input - t.surviving,
            "removal_examples": t.examples.iter().map(|(p, st, si, d)| serde_json::json!({"point": p, "survivor": [st, si], "distance_m": f(*d)})).collect::<Vec<_>>()
        })).collect::<Vec<_>>(), "conflicts": []}));
    // `validate_recovered_geometry`: every claimed core sample of a recovered tree is still supported.
    if !admitted.is_empty() {
        let (checked, missing, examples) = recovered_support(&admitted, &claims_of(&model, &dense_files, &ownership, &regions, &overlaps, &origin, &admitted)?, &survivors)?;
        survivors.release();
        orphan_json["final_support"] = serde_json::json!({"checked_samples": checked, "missing_samples": missing,
            "missing_examples": examples.iter().map(|(t, i, p)| serde_json::json!({"tile": t, "local_instance": i,
                "xyz": [f(p[0] + origin[0]), f(p[1] + origin[1]), f(p[2] + origin[2])]})).collect::<Vec<_>>()});
        report.insert("orphan_recovery".into(), orphan_json.clone());
        if missing > 0 {
            return Err(format!("{}: recovered tree geometry lost after ownership and deduplication: {missing}/{checked} core samples unsupported; see orphan_recovery.final_support in report", model.name));
        }
    }
    let groups = ids.len() as u64;
    let mut reassigned: BTreeMap<u64, u64> = BTreeMap::new();
    if let Some((max_cluster_size, max_volume)) = params.small_instances {
        report.insert("small_instance_reassignment".into(), Value::Null);
        let (plan, mut stage) = timed("small_instance_reassignment", &model, report, || {
            let (plan, mut stage) = plan_reassignment(&statistics, max_cluster_size, max_volume);
            if !plan.is_empty() {
                stage.insert("relabeled_points_per_tile".into(), serde_json::json!(relabel_tiles(&final_files, &model.instance, &plan)?));
            }
            Ok((plan, stage))
        })?;
        report.insert("small_instance_reassignment".into(), Value::Object(std::mem::take(&mut stage)));
        reassigned = plan;
    }
    let final_ids: std::collections::BTreeSet<u64> = ids.iter().map(|i| *reassigned.get(i).unwrap_or(i)).collect();
    let added: std::collections::BTreeSet<u64> = reassigned.values().copied().collect();
    report.insert("instance_consolidation".into(), Value::Null);
    let (split, moved, per_owner) = timed("instance_consolidation", &model, report, || consolidate(&final_files, &model.instance, &regions, &scratch.join("consolidation").join(suffix), pool))?;
    report.insert("instance_consolidation".into(), serde_json::json!({
        "policy": "each positive instance in exactly one tile: owner core holds its XY centroid, else most points (ties: lowest tile); points are moved, never dropped",
        "split_instances": split, "moved_points": moved,
        "moved_points_per_owner": per_owner.iter().map(|(k, v)| (k.clone(), serde_json::json!(v))).collect::<Map<_, _>>()}));
    // Final metadata beside the tiles.
    let csv: String = std::iter::once("PredInstance,has_added_clusters\r\n".replace("PredInstance", &model.instance))
        .chain(final_ids.iter().map(|id| format!("{id},{}\r\n", u8::from(added.contains(id))))).collect();
    fs::write(final_dir.join("instance_metadata.csv"), &csv).map_err(|e| e.to_string())?;
    let summary_name = format!("{}_summary.json", model.instance);
    write_json(&final_dir.join(&summary_name), &summary_document(&model, &statistics, &origin, params.resolution_1, &reassigned, params.small_instances.is_some()))?;
    // The layout used for ownership; a recovered legacy layout becomes the single-cloud layout.
    let published = match regions.first().and_then(|r| r.recovery.clone().map(|rec| (r.core, rec))).filter(|_| regions.len() == 1) {
        Some((core, recovery)) => {
            let mut l = layout.as_object().cloned().unwrap_or_default();
            let xy = serde_json::json!([[f(core[0][0]), f(core[0][1])], [f(core[1][0]), f(core[1][1])]]);
            l.insert("tiling_skipped".into(), Value::Bool(true));
            l.insert("tile_buffer".into(), f(0.0));
            l.insert("proj_extent".into(), serde_json::json!({"minx": f(core[0][0]), "miny": f(core[1][0]), "maxx": f(core[0][1]), "maxy": f(core[1][1])}));
            l.insert("grid_bounds".into(), serde_json::json!({"xmin": f(core[0][0]), "xmax": f(core[0][1]), "ymin": f(core[1][0]), "ymax": f(core[1][1])}));
            l.insert("tiles".into(), serde_json::json!([{"col": 0, "row": 0, "bounds": xy, "core": xy, "planned_bounds": xy}]));
            let mut recovery = recovery.as_object().cloned().unwrap_or_default();
            recovery.insert("source_sha256".into(), Value::String(crate::sha256::file_hex(&params.tile_bounds_json)?));
            l.insert("layout_recovery".into(), Value::Object(recovery));
            Value::Object(l)
        }
        None => layout.clone(),
    };
    write_json(&final_dir.join("tile_bounds.json"), &published)?;
    let mut manifest = Map::new();
    manifest.insert("contract".into(), Value::String("3DT-2183/v8-group-geometry".into()));
    manifest.insert("model".into(), Value::String(model.name.clone()));
    manifest.insert("resolution_1_m".into(), crate::pyfmt::float(params.resolution_1));
    if params.keep_unfiltered {
        let name = format!("{}_unfiltered_1cm", params.output_tiles.file_name().unwrap().to_string_lossy());
        let baseline = if suffix.is_empty() { format!("../{name}") } else { format!("../../{name}/{suffix}") };
        manifest.insert("baseline".into(), Value::String(baseline));
    }
    manifest.insert("instance_summary".into(), Value::String(summary_name));
    manifest.insert("tile_bounds_json".into(), Value::String("tile_bounds.json".into()));
    manifest.insert("tile_bounds_sha256".into(), Value::String(crate::sha256::file_hex(&final_dir.join("tile_bounds.json"))?));
    write_json(&final_dir.join("smarttile_merge.json"), &Value::Object(manifest))?;
    Ok(ModelOutcome { model, labels, final_dir, dense, final_files, survivors: keep_survivors.then_some(survivors), groups, reassigned, csv })
}

/// What a model's merge leaves for enrichment and publication.
struct ModelOutcome { model: Model, labels: Vec<String>, final_dir: PathBuf, dense: PathBuf, final_files: Vec<PathBuf>,
                      survivors: Option<crate::parts::Parts>, groups: u64, reassigned: BTreeMap<u64, u64>, csv: String }

pub fn run(params: &MergeParams) -> Res<()> {
    let started = std::time::Instant::now();
    let layout: Value = serde_json::from_str(&fs::read_to_string(&params.tile_bounds_json).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let output_tiles = params.output_tiles.clone();
    let root = output_tiles.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let baseline_output = output_tiles.with_file_name(format!("{}_unfiltered_1cm", output_tiles.file_name().unwrap().to_string_lossy()));
    let single = params.collections.len() == 1;
    if params.output_merged_laz.as_ref().is_some_and(|m| m.file_name().unwrap().to_string_lossy().to_lowercase().ends_with(".copc.laz")) {
        return Err("Processed merge writes regular LAZ; use create_merged_file for COPC products".into());
    }
    if params.output_merged_laz.is_some() && !params.skip_merged_file && !single {
        return Err("A single --output-merged-laz cannot combine model geometry; use --skip-merged-file for multiple collections".into());
    }
    let merged_output = if params.skip_merged_file { None } else { params.output_merged_laz.clone().or_else(|| single.then(|| params.work.join("merged_tiles.laz"))) };
    let merged_metadata = merged_output.as_ref().map(|m| m.with_file_name(format!("{}_instance_metadata.csv", crate::collections::file_stem(m))))
        .unwrap_or_else(|| params.work.join("merged_tiles_instance_metadata.csv"));
    let mut destinations: Vec<&PathBuf> = vec![&output_tiles];
    if params.keep_unfiltered { destinations.push(&baseline_output); }
    destinations.extend(merged_output.as_ref());
    for path in &destinations { available_destination(path)?; }
    let first = crate::tile::point_cloud_files(&params.collections[0])?;
    if first.is_empty() {
        return Err(format!("No predictions in {}", params.collections[0].display()));
    }
    let origin = lasio::read_header(&first[0])?.offsets;
    let qw = params.workers.max(1);
    let f = crate::pyfmt::float;
    let scratch = params.work.join(format!(".remap-first-{}", std::process::id()));
    let mut checksums = Map::new();
    let original_output = params.originals.as_ref().map(|_| params.original_output.clone().unwrap_or_else(|| root.join("original_with_predictions")));
    if let Some(destination) = &original_output { available_destination(destination)?; }
    let resolution = params.resolution_1;
    let original_radius = params.remap_tolerance.unwrap_or(3f64.sqrt() * resolution);
    let mut top_report = serde_json::json!({
        "contract": "3DT-2183/v8-group-geometry", "state": "running", "models": [], "duplicate_radius_m": f(0.01),
        "resolution_1_m": f(resolution), "original_radius_m": f(original_radius), "required_match_fraction": f(1.0),
        "batch_point_limit": MAX_BATCH_POINTS, "distance_boundary": "inclusive Euclidean XYZ; 8 float64 ULP guard in local coordinates",
        "survivor_order": "source tile filename, then original point order",
        "nearest_ties": "within 8 local-coordinate float64 ULPs of the minimum distance; stable tile/point order",
        "original_coverage": "pending: originals not supplied",
        "instance_policy": "one ID per accepted group; union member geometry; shared-point tile ownership",
        "parallelism": {"requested_workers": params.requested_workers, "query_workers": qw,
                        "scope": "bounded spatial queries; small batches run serially"},
        "origin": origin.iter().map(|o| f(*o)).collect::<Vec<_>>()});
    let mut model_reports: Vec<Map<String, Value>> = Vec::new();
    let top = &mut top_report;
    let result = (|| -> Res<()> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(params.workers.max(1)).build().map_err(|e| e.to_string())?;
        let mut outcomes = Vec::new();
        for (i, collection) in params.collections.iter().enumerate() {
            let suffix = if single { String::new() } else { format!("model_{i:03}") };
            model_reports.push(Map::new());
            let outcome = merge_model(params, collection, &suffix, &layout, origin, &scratch, &pool, model_reports.last_mut().unwrap())?;
            outcomes.push(outcome);
        }
        // `enrich_originals` on every model's final survivors, measured against its dense baseline.
        let mut originals_stage = None;
        if let Some(originals) = &params.originals {
            if std::env::var_os("SMARTTILE_TIMINGS").is_some() { eprintln!("    resident before originals: {} MiB", resident_mib()); }
            let radius = original_radius;
            let mut collections = Vec::new();
            let mut baselines = Vec::new();
            for (i, o) in outcomes.iter_mut().enumerate() {
                let mut dimensions = o.model.dimensions.clone();
                for d in dimensions.iter_mut().filter(|d| d.name == o.model.instance) {
                    d.data_type = instance_scalar(o.groups).extra_bytes_id();
                    d.options = 0;
                    d.description = [0; 32];
                }
                let width = dimensions.iter().map(|d| d.size()).sum();
                collections.push(crate::remap::Collection { name: o.model.name.clone(), instance: Some(o.model.instance.clone()), dimensions,
                    parts: o.survivors.take().expect("survivors kept for originals"), width, maximum: o.groups });
                baselines.push(Some(crate::remap::load_baseline(&o.dense, &origin, radius, &o.model.instance, &scratch.join(format!("baseline_{i:03}")))?));
            }
            if std::env::var_os("SMARTTILE_TIMINGS").is_some() { eprintln!("    resident with baselines: {} MiB", resident_mib()); }
            let (selected, used) = crate::remap::select_dimensions(&collections, params.target_dims.as_ref(), &[])?;
            let files = crate::remap::original_files(originals)?;
            let stage = scratch.join("originals");
            fs::create_dir_all(&stage).map_err(|e| e.to_string())?;
            let ids_by_dimension = crate::remap::enrich(&collections, &baselines, &selected, &used, &files, &origin, &vec![radius; collections.len()], qw, &stage, top,
                                                         &vec![None; collections.len()], false)?;
            let aliases: BTreeMap<String, BTreeMap<u64, u64>> = outcomes.iter().map(|o| (o.model.instance.clone(), o.reassigned.clone())).collect();
            top["instance_finalization"] = crate::remap::compact(&stage, &ids_by_dimension, &aliases)?;
            originals_stage = Some(stage);
        }
        let merged = scratch.join("merged.laz");
        let metadata = scratch.join("merged_instance_metadata.csv");
        if merged_output.is_some() {
            merge_file(&outcomes[0].final_files, &merged, &pool)?;
            fs::write(&metadata, &outcomes[0].csv).map_err(|e| e.to_string())?;
        }
        // Name the tiles after the layout, then publish.
        for (o, report) in outcomes.iter().zip(model_reports.iter_mut()) {
            for directory in [&o.final_dir, &o.dense] {
                for (n, label) in o.labels.iter().enumerate() {
                    fs::rename(directory.join(format!("tile_{n:05}.laz")), directory.join(format!("{label}.laz"))).map_err(|e| e.to_string())?;
                }
            }
            report.insert("tile_names".into(), Value::Array(o.labels.iter().map(|l| Value::String(l.clone())).collect()));
        }
        // `checksums`: every LAS/LAZ product (and the metadata file) at its published path.
        let final_root = scratch.join("final");
        let dense_root = scratch.join("dense");
        let mut sources: Vec<(PathBuf, PathBuf)> = vec![(final_root.clone(), output_tiles.clone())];
        if params.keep_unfiltered { sources.push((dense_root.clone(), baseline_output.clone())); }
        if let (Some(stage), Some(destination)) = (&originals_stage, &original_output) { sources.push((stage.clone(), destination.clone())); }
        let mut listed: Vec<(String, PathBuf)> = Vec::new();
        for (source, destination) in &sources {
            for path in laz_files_recursive(source)? {
                let relative = path.strip_prefix(source).unwrap().to_path_buf();
                listed.push((destination.join(relative).display().to_string(), path));
            }
        }
        let digests: Vec<Res<String>> = pool.install(|| listed.par_iter().map(|(_, path)| crate::sha256::file_hex(path)).collect());
        for ((key, _), digest) in listed.into_iter().zip(digests) {
            checksums.insert(key, Value::String(digest?));
        }
        if let Some(out) = &merged_output {
            checksums.insert(out.display().to_string(), Value::String(crate::sha256::file_hex(&merged)?));
            checksums.insert(merged_metadata.display().to_string(), Value::String(crate::sha256::file_hex(&metadata)?));
        }
        let mut products: Vec<(PathBuf, PathBuf)> = sources;
        if let Some(out) = &merged_output { products.push((merged.clone(), out.clone())); }
        for (source, destination) in &products {
            if destination.is_dir() { fs::remove_dir(destination).map_err(|e| e.to_string())?; }
            if let Some(parent) = destination.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
            fs::rename(source, destination).map_err(|e| format!("publishing {}: {e}", destination.display()))?;
        }
        if merged_output.is_some() { fs::rename(&metadata, &merged_metadata).map_err(|e| e.to_string())?; }
        Ok(())
    })();
    if std::env::var_os("SMARTTILE_KEEP_STAGES").is_none() {
        let _ = fs::remove_dir_all(&scratch);
    }
    let state = match (&result, &params.originals) { (Err(_), _) => "failed", (Ok(_), Some(_)) => "validated",
                                                      (Ok(_), None) => "intermediate; final original coverage pending" };
    let mut full = top_report.as_object().unwrap().clone();
    full.insert("state".into(), Value::String(state.into()));
    full.insert("models".into(), Value::Array(model_reports.into_iter().map(Value::Object).collect()));
    if result.is_ok() { full.insert("checksums".into(), Value::Object(checksums)); }
    if let Err(e) = &result { full.insert("error".into(), Value::String(e.clone())); }
    full.insert("wall_seconds".into(), f(started.elapsed().as_secs_f64()));
    full.insert("cpu_seconds".into(), f(process_cpu_seconds()));
    // A failure before any output must still report itself, not the missing folder.
    let _ = fs::create_dir_all(&root);
    let written = write_json(&root.join("remap_first_report.json"), &Value::Object(full));
    result?;
    written?;
    println!("Merge complete: {state}; report: {}", root.join("remap_first_report.json").display());
    Ok(())
}

/// `sorted(source.rglob("*.la*"))`: LAS/LAZ files below a directory, in path order.
fn laz_files_recursive(directory: &Path) -> Res<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_dir() { stack.push(path); } else if path.file_name().unwrap().to_string_lossy().contains(".la") { out.push(path); }
        }
    }
    out.sort();
    Ok(out)
}
