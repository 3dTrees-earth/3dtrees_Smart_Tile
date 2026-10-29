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
use crate::spatial::PointSet;

const MAX_BATCH_POINTS: usize = 32_768;
const ORIGINAL_WRITE_POINTS: u64 = 1_000_000;
const REGION_SIZE: f64 = 4.0;
const CELL_SIZE: f64 = 2.0;
const BLOCK_POINTS: f64 = 4_000_000.0;
const BLOCK_POINT_CAP: usize = 8_000_000;
const MAX_BLOCK: f64 = 64.0;

pub struct RemapParams {
    pub collections: Vec<PathBuf>,
    pub originals: PathBuf,
    pub output: PathBuf,
    pub workers: usize,
    pub requested_workers: usize,
    pub instance_dimension: String,
    pub target_dims: Option<BTreeSet<String>>,
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

/// One prediction collection: its dimensions and every point with raw values.
pub struct Collection { pub name: String, pub instance: Option<String>, pub dimensions: Vec<ExtraBytes>, pub set: PointSet, pub values: Vec<u8>, pub width: usize, pub maximum: u64 }

pub fn load_collection(path: &Path, origin: &[f64; 3], radius: f64, instance_dimension: &str) -> Res<Collection> {
    let files = crate::tile::point_cloud_files(path)?;
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
    let parts: Vec<Res<(Vec<[f64; 3]>, Vec<u8>, u64)>> = files.par_iter().zip(&headers).map(|(file, header)| {
        let layout = PointLayout::of(header)?;
        let fields: Vec<Layout> = dimensions.iter().map(|d| layout.find(&d.name).map(|f| f.0).ok_or(format!("{}: inconsistent prediction schema", file.display()))).collect::<Res<_>>()?;
        let instance_field = instance.as_ref().map(|n| layout.find(n).unwrap().0);
        let (mut xyz, mut values, mut maximum) = (Vec::new(), Vec::new(), 0u64);
        let mut scratch = [0u8; 8];
        for chunk in chunks(file, MAX_BATCH_POINTS as u64)? {
            let chunk = chunk?;
            for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
                xyz.push(local(r, header, origin));
                for f in &fields { values.extend_from_slice(f.get(r, &mut scratch)); }
                if let Some(f) = instance_field {
                    let v = f.scalar().read_int(f.get(r, &mut scratch));
                    if v < 0 { return Err(format!("{}: invalid or missing labels", file.display())); }
                    maximum = maximum.max(v as u64);
                }
            }
        }
        Ok((xyz, values, maximum))
    }).collect();
    let mut set = PointSet::new(radius);
    let mut all_values = Vec::new();
    let mut maximum = 0;
    for (tile, part) in parts.into_iter().enumerate() {
        let (xyz, values, m) = part?;
        for (n, p) in xyz.into_iter().enumerate() { set.push(p, tile as u32, n as i64); }
        all_values.extend(values);
        maximum = maximum.max(m);
    }
    set.build();
    Ok(Collection { name: path.file_name().unwrap().to_string_lossy().into_owned(), instance, dimensions, set, values: all_values, width, maximum })
}

/// `block_size`: the spill block edge in whole 4 m regions.
fn block_size(count: u64, h: &SourceHeader) -> f64 {
    let area = ((h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1])).max(1e-9);
    let edge = (BLOCK_POINTS * area / (count.max(1) as f64)).sqrt();
    MAX_BLOCK.min(REGION_SIZE.max((edge / REGION_SIZE).floor() * REGION_SIZE))
}

/// `spill_original` + `block_batches`: (point indices, local XYZ) batches, block by block,
/// ordered by 2 m cell then point index, 32,768 at a time.
fn query_batches(file: &Path, header: &SourceHeader, origin: &[f64; 3]) -> Res<Vec<(Vec<u32>, Vec<[f64; 3]>)>> {
    let size = block_size(header.point_count, header);
    let mut blocks: BTreeMap<(i64, i64), Vec<(u32, [f64; 3])>> = BTreeMap::new();
    let mut i = 0u32;
    for chunk in chunks(file, 1_000_000)? {
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
        if points.len() <= BLOCK_POINT_CAP {
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
    if resolutions.len() > 1 {
        return Err("Prediction collections declare conflicting first-stage resolutions".into());
    }
    let resolution = resolutions.iter().next().map(|b| f64::from_bits(*b)).unwrap_or(0.01);
    let radius = 3f64.sqrt() * resolution;
    let originals = original_files(&params.originals)?;
    let origin = lasio::read_header(&originals[0])?.offsets;
    let indexing_start = std::time::Instant::now();
    let mut collections = Vec::new();
    let mut baseline_sets = Vec::new();
    for (c, baseline) in params.collections.iter().zip(&baselines) {
        collections.push(load_collection(c, &origin, radius, &params.instance_dimension)?);
        baseline_sets.push(match baseline {
            Some(b) => Some(load_collection(b, &origin, radius, &params.instance_dimension)?.set),
            None => None,
        });
    }
    let indexing_seconds = indexing_start.elapsed().as_secs_f64();
    let (selected, used) = select_dimensions(&collections, params.target_dims.as_ref())?;
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
    report["timings"] = json!({"indexing_seconds": f(indexing_seconds)});
    let staging = parent.join(format!(".strict-remap-{}", std::process::id()));
    let stage_dir = staging.join("originals");
    fs::create_dir_all(&stage_dir).map_err(|e| e.to_string())?;
    let result = enrich(&collections, &baseline_sets, &selected, &used, &originals, &origin, radius, params.workers, &stage_dir, &mut report);
    let outcome = result.and_then(|ids| {
        report["instance_finalization"] = compact(&stage_dir, &ids)?;
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
pub fn select_dimensions(collections: &[Collection], target_dims: Option<&BTreeSet<String>>) -> Res<(Vec<Vec<(usize, ExtraBytes)>>, BTreeSet<String>)> {
    let mut used = BTreeSet::new();
    let mut selected: Vec<Vec<(usize, ExtraBytes)>> = Vec::new();
    for c in collections {
        let dims: Vec<(usize, ExtraBytes)> = c.dimensions.iter().enumerate()
            .filter(|(_, d)| target_dims.is_none_or(|t| t.contains(&d.name))).map(|(i, d)| {
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
pub fn enrich(collections: &[Collection], baseline_sets: &[Option<PointSet>], selected: &[Vec<(usize, ExtraBytes)>], used: &BTreeSet<String>,
              originals: &[PathBuf], origin: &[f64; 3], radius: f64, workers: usize, stage_dir: &Path, report: &mut Value)
              -> Res<BTreeMap<String, BTreeMap<String, BTreeSet<u64>>>> {
    let f = crate::pyfmt::float;
    let source_points: u64 = originals.iter().map(|o| lasio::read_header(o).map(|h| h.point_count)).sum::<Res<u64>>()?;
    let processes = workers.min((source_points / MAX_BATCH_POINTS as u64).max(1) as usize);
    let parallelism = report["parallelism"].as_object_mut().ok_or("report without parallelism")?;
    parallelism.insert("enrichment_processes".into(), json!(processes));
    parallelism.insert("enrichment_query_workers".into(), json!(if processes > 1 { 1 } else { workers }));
    parallelism.insert("max_pending_batches".into(), json!(2 * processes));
    parallelism.insert("enrichment_scope".into(), json!("bounded process-parallel original remap; ordered writer"));
    report["original_coverage"] = json!([]);
    report["original_coverage_policy"] = json!({"unfiltered_1cm": "100% required for every model with baseline geometry",
        "final_survivors": "100% required except RCT; unmatched RCT predictions become background 0"});
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers.max(1)).build().map_err(|e| e.to_string())?;
    let enrichment_start = std::time::Instant::now();
    let mut phases = [0.0f64; 3]; // spill, query, write
    let mut metrics: Vec<Coverage> = Vec::new();
    let result = (|| -> Res<BTreeMap<String, BTreeMap<String, BTreeSet<u64>>>> {
        let mut ids_by_dimension: BTreeMap<String, BTreeMap<String, BTreeSet<u64>>> = BTreeMap::new();
        for file in originals {
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
            let batches = query_batches(file, &header, origin)?;
            phases[0] += tick.elapsed().as_secs_f64();
            let tick = std::time::Instant::now();
            let mut file_ids: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
            let mut finals: Vec<Coverage> = collections.iter().map(|c| Coverage::new(&file_name, &c.name, "final_survivors", radius)).collect();
            let mut bases: Vec<Option<Coverage>> = collections.iter().zip(baseline_sets)
                .map(|(c, b)| b.as_ref().map(|_| Coverage::new(&file_name, &c.name, "unfiltered_1cm", radius))).collect();
            for (m, c) in collections.iter().enumerate() {
                let results: Vec<(Vec<Option<(u32, f64)>>, Option<Vec<f64>>)> = pool.install(|| batches.par_iter().map(|(_, xyz)| {
                    let base = baseline_sets[m].as_ref().map(|b| b.nearest_distances(xyz, radius));
                    (c.set.nearest(xyz, radius, &|_| true), base)
                }).collect());
                // Byte offset of this model's selected dimensions in a stored row.
                let start: usize = selected[..m].iter().flatten().map(|(_, d)| d.size()).sum();
                let instance_pos = selected[m].iter().position(|(_, d)| c.instance.as_deref() == Some(d.name.as_str()));
                let mut source_offsets = Vec::new();
                let mut o = 0;
                for d in &c.dimensions { source_offsets.push(o); o += d.size(); }
                for ((indices, xyz), (found, base)) in batches.iter().zip(&results) {
                    if let (Some(metric), Some(base)) = (bases[m].as_mut(), base) { metric.record(xyz, base, origin); }
                    let distances: Vec<f64> = found.iter().map(|x| x.map_or(f64::INFINITY, |(_, d)| d)).collect();
                    finals[m].record(xyz, &distances, origin);
                    for (k, &i) in indices.iter().enumerate() {
                        let Some((id, _)) = found[k] else { continue };
                        let row = &c.values[id as usize * c.width..(id as usize + 1) * c.width];
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
            let assigned: usize = batches.iter().map(|(i, _)| i.len()).sum();
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
    report["original_coverage"] = Value::Array(metrics.iter().map(Coverage::json).collect());
    let timings = report.as_object_mut().unwrap().entry("timings").or_insert_with(|| json!({}));
    timings["enrichment_seconds"] = f(enrichment_start.elapsed().as_secs_f64());
    timings["enrichment_phase_seconds"] = json!({"spill": f(phases[0]), "query": f(phases[1]), "write": f(phases[2])});
    let ids = result?;
    report["background_assigned_points"] = json!(0);
    let failures: Vec<&Coverage> = metrics.iter().filter(|m| m.matched != m.total).collect();
    if !failures.is_empty() {
        report["coverage_failures"] = Value::Array(failures.iter().map(|m| json!({"file": m.file, "model": m.model, "stage": m.stage,
                                                                                "missing": m.total - m.matched})).collect());
        return Err(format!("100% original coverage within {} m is required; {}", g8(radius),
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
struct Coverage { file: String, model: String, stage: &'static str, radius: f64, total: u64, matched: u64, max: f64, histogram: [u64; 5], missing: Vec<[f64; 3]> }

impl Coverage {
    fn new(file: &str, model: &str, stage: &'static str, radius: f64) -> Coverage {
        Coverage { file: file.into(), model: model.into(), stage, radius, total: 0, matched: 0, max: 0.0, histogram: [0; 5], missing: Vec::new() }
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
        json!({"file": self.file, "model": self.model, "stage": self.stage, "radius_m": f(self.radius), "total": self.total,
               "matched": self.matched, "max_matched_distance_m": f(self.max), "distance_histogram": self.histogram,
               "histogram_upper_bounds_m": self.bounds().map(f),
               "missing_examples": self.missing.iter().map(|p| p.map(f).to_vec()).collect::<Vec<_>>()})
    }
}

/// `compact_originals`: dataset-wide IDs 1..N per instance dimension, uint32, plus the mapping file.
pub fn compact(dir: &Path, ids_by_dimension: &BTreeMap<String, BTreeMap<String, BTreeSet<u64>>>) -> Res<Value> {
    if ids_by_dimension.is_empty() {
        return Ok(json!({"mapping": null, "counts": {}}));
    }
    let mut mappings: BTreeMap<String, BTreeMap<u64, u32>> = BTreeMap::new();
    let mut models = serde_json::Map::new();
    let mut files: BTreeSet<String> = BTreeSet::new();
    for (name, by_file) in ids_by_dimension {
        let sources: BTreeSet<u64> = by_file.values().flatten().copied().filter(|v| *v > 0).collect();
        let mapping: BTreeMap<u64, u32> = sources.iter().enumerate().map(|(i, s)| (*s, i as u32 + 1)).collect();
        models.insert(name.clone(), json!({"scope": "dataset", "instance_count": sources.len(),
            "instances": sources.iter().map(|s| json!({"source_instance_id": s, "instance_id": mapping[s]})).collect::<Vec<_>>()}));
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
              "counts": mappings.iter().map(|(k, m)| (k.clone(), json!(m.len()))).collect::<serde_json::Map<_, _>>()}))
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
