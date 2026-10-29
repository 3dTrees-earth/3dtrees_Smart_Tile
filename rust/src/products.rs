//! The `create_merged_file` task (`main_create_merged_file.py`,
//! `full_resolution_merge.py`): the full-resolution product written natively
//! in record order with `OriginalPointIndex`, and subsampled products from
//! staged COPCs, read and written by PDAL/Untwine exactly as in Python, with
//! the voxel selection (`select_voxel_points`) done here.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use rayon::prelude::*;
use serde_json::{json, Value};

use crate::decimal::{py_round, Ratio};
use crate::lasio::{self, Res, SourceHeader, TileHeader};
use crate::points::{ExtraBytes, OutputExtra, PointLayout, Scalar};
use crate::pyfmt::float_repr;

const READ_POINTS: u64 = 1_000_000;
const PRODUCT_CHUNK_POINTS: f64 = 2_000_000.0;
const SELECTIVE: [&str; 3] = ["PredInstance", "PredInstance_SAT", "PredInstance_FM"];

pub struct ProductParams {
    pub input: PathBuf,
    pub output_dir: PathBuf,
    pub resolutions: Vec<(String, f64)>,
    pub formats: Vec<String>,
    pub workers: usize,
    pub original_point_index: bool,
    pub staged_copc_dir: Option<PathBuf>,
}

fn chunks(path: &Path, size: u64) -> Res<impl Iterator<Item = Res<las::PointData>>> {
    let mut reader = las::Reader::from_path(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(std::iter::from_fn(move || match reader.read_points(size) {
        Ok(c) if c.is_empty() => None,
        Ok(c) => Some(Ok(c)),
        Err(e) => Some(Err(e.to_string())),
    }))
}

/// `resolution_label` of create_merged_file.
pub fn resolution_label(resolution: f64) -> String {
    let cm = resolution * 100.0;
    if (cm - py_round(cm)).abs() <= 1e-9 { format!("{}cm", py_round(cm) as i64) } else { format!("{}cm", g_format(cm)) }
}

fn g_format(x: f64) -> String {
    let s = format!("{x:.6}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// `parse_merged_resolutions`.
pub fn parse_resolutions(value: &str, res1: f64, res2: f64) -> Res<Vec<(String, f64)>> {
    let mut out: Vec<(String, f64)> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for token in value.split(',').map(|t| t.trim().to_lowercase()).filter(|t| !t.is_empty()) {
        if token == "full" || token == "original" {
            if !seen.contains(&"full".to_string()) { seen.push("full".into()); out.push(("full".into(), 0.0)); }
            continue;
        }
        let resolution = match token.as_str() {
            "res1" | "resolution1" | "resolution_1" | "resolution-1" => res1,
            "res2" | "resolution2" | "resolution_2" | "resolution-2" => res2,
            t if t.ends_with("cm") => t[..t.len() - 2].trim().parse::<f64>().map_err(|_| format!("bad resolution {t}"))? / 100.0,
            t => t.parse::<f64>().map_err(|_| format!("bad resolution {t}"))?,
        };
        if resolution <= 0.0 { return Err("merged resolutions must be positive".into()); }
        let key = format!("{:.9}", resolution);
        if seen.contains(&key) { continue; }
        seen.push(key);
        out.push((resolution_label(resolution), resolution));
    }
    if out.is_empty() { return Err("No merged resolutions selected".into()); }
    Ok(out)
}

fn output_path(dir: &Path, label: &str, format: &str) -> PathBuf {
    if label == "full" { return dir.join(format!("merged.{format}")); }
    // re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    let mut safe = String::new();
    for c in label.chars() {
        if c.is_ascii_alphanumeric() || "_.-".contains(c) { safe.push(c) } else if !safe.ends_with('_') || safe.is_empty() { safe.push('_') }
    }
    dir.join(format!("prod_merged_{}.{format}", safe.trim_matches('_')))
}

fn source_key(path: &Path) -> String { crate::collections::file_stem(path).to_lowercase() }

/// `merged_header` + records: every input record in order, OriginalPointIndex first.
/// `ply`: `_ply_header`, the index as uint32 and no 64-bit integer attributes.
fn write_full(inputs: &[PathBuf], output: &Path, compress: bool, original_point_index: bool, ply: bool) -> Res<u64> {
    let headers = inputs.iter().map(|p| lasio::read_header(p)).collect::<Res<Vec<_>>>()?;
    let first = &headers[0];
    let first_layout = PointLayout::of(first)?;
    let extras: Vec<ExtraBytes> = first_layout.extra.iter().map(|(e, _)| e.clone()).collect();
    for (path, h) in inputs.iter().zip(&headers).skip(1) {
        let l = PointLayout::of(h)?;
        if h.point_format != first.point_format || l.extra.len() != extras.len()
            || l.extra.iter().zip(&extras).any(|((a, _), b)| a.name != b.name || a.data_type != b.data_type) {
            return Err(format!("{}: point format or ExtraBytes differ; a full-resolution merge needs one schema", path.display()));
        }
        let same_crs = match (crate::crs::readable_crs(first), crate::crs::readable_crs(h)) {
            (None, None) => true,
            (Some(a), Some(b)) => crate::proj::crs_equivalent(&a, &b),
            _ => false,
        };
        if !same_crs { return Err(format!("{}: CRS differs from {}", path.display(), inputs[0].display())); }
    }
    if original_point_index && extras.iter().any(|e| e.name == "OriginalPointIndex") {
        return Err("Inputs already contain OriginalPointIndex; disable --original-point-index".into());
    }
    let expected: u64 = headers.iter().map(|h| h.point_count).sum();
    if ply {
        if extras.iter().any(|e| e.name == "OriginalPointIndex") {
            return Err("PLY from inputs that already hold OriginalPointIndex is not ported to Rust yet".into());
        }
        if let Some(e) = extras.iter().find(|e| matches!(e.scalar(), Some(Scalar::U64 | Scalar::I64))) {
            return Err(format!("PLY cannot represent 64-bit integer attribute {}; use LAZ or COPC to retain its values", e.name));
        }
        if expected > u64::from(u32::MAX) {
            return Err(format!("PLY point count {expected} exceeds the writer limit {}; use LAZ or COPC", u32::MAX));
        }
    }
    let scales: [f64; 3] = std::array::from_fn(|a| headers.iter().map(|h| h.scales[a]).fold(f64::INFINITY, f64::min));
    let mins: [f64; 3] = std::array::from_fn(|a| headers.iter().map(|h| h.mins[a]).fold(f64::INFINITY, f64::min));
    let maxs: [f64; 3] = std::array::from_fn(|a| headers.iter().map(|h| h.maxs[a]).fold(f64::NEG_INFINITY, f64::max));
    let mut offsets = first.offsets;
    if (0..3).any(|a| (mins[a] - offsets[a]) / scales[a] < i32::MIN as f64 || (maxs[a] - offsets[a]) / scales[a] > i32::MAX as f64) {
        offsets = mins.map(f64::floor);
    }
    let mut header = TileHeader::copy_of(first)?;
    header.scales = scales;
    header.offsets = offsets;
    let ordered: Vec<ExtraBytes> = SELECTIVE.iter().filter_map(|n| extras.iter().find(|e| e.name == *n))
        .chain(extras.iter().filter(|e| !SELECTIVE.contains(&e.name.as_str()))).cloned().collect();
    header.extra.clear();
    if original_point_index {
        let kind = if ply { Scalar::U32 } else { Scalar::U64 };
        let mut opi = ExtraBytes { name: "OriginalPointIndex".into(), data_type: kind.extra_bytes_id(), options: 0, no_data: [0; 24],
            scale: [0.0; 3], offset: [0.0; 3], description: [0; 32], min: [0; 24], max: [0; 24] };
        let text = b"Record index in input order";
        opi.description[..text.len()].copy_from_slice(text);
        header.extra.push(OutputExtra::new(&opi));
    }
    header.extra.extend(ordered.iter().map(OutputExtra::new));
    header.sync_extra_bytes();
    let base = crate::points::base_record_length(first.point_format);
    let record_length = header.record_length();
    let mut plans = Vec::new();
    for (path, h) in inputs.iter().zip(&headers) {
        let layout = PointLayout::of(h)?;
        let fields: Vec<(usize, usize)> = ordered.iter().map(|e| { let (_, o) = layout.extra.iter().find(|(x, _)| x.name == e.name).unwrap(); (*o, e.size()) }).collect();
        plans.push((path.clone(), h.scales, h.offsets, fields));
    }
    // Streamed: one READ_POINTS batch at a time, file by file.
    let mut index = 0u64;
    let mut plans = plans.into_iter();
    let mut current: Option<(Box<dyn Iterator<Item = Res<las::PointData>>>, [f64; 3], [f64; 3], Vec<(usize, usize)>)> = None;
    let batches = std::iter::from_fn(move || loop {
        if current.is_none() {
            let (path, s, o, fields) = plans.next()?;
            match chunks(&path, READ_POINTS) {
                Ok(it) => current = Some((Box::new(it), s, o, fields)),
                Err(e) => return Some(Err(e)),
            }
        }
        let (it, source_scales, source_offsets, fields) = current.as_mut().unwrap();
        let chunk = match it.next() {
            None => { current = None; continue; }
            Some(Err(e)) => return Some(Err(e)),
            Some(Ok(c)) => c,
        };
        let same_grid = *source_scales == scales && *source_offsets == offsets;
        let mut out = Vec::with_capacity(chunk.len() * record_length);
        for r in chunk.raw_bytes().chunks_exact(chunk.record_len()) {
            let start = out.len();
            out.extend_from_slice(&r[..base]);
            if !same_grid {
                for a in 0..3 {
                    let world = source_scales[a] * f64::from(i32::from_le_bytes(r[a * 4..a * 4 + 4].try_into().unwrap())) + source_offsets[a];
                    let v = ((world - offsets[a]) / scales[a]).round_ties_even() as i32;
                    out[start + a * 4..start + a * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            if original_point_index {
                if ply { out.extend_from_slice(&(index as u32).to_le_bytes()) } else { out.extend_from_slice(&index.to_le_bytes()) }
            }
            for (o, size) in fields.iter() { out.extend_from_slice(&r[*o..o + size]); }
            index += 1;
        }
        return Some(Ok(out));
    });
    let written = lasio::write_records(output, &header, batches, compress, None)?;
    if written != expected {
        return Err(format!("Full-resolution merge wrote {written} of {expected} points"));
    }
    Ok(written)
}

/// `add_crs_comment_to_ply`: `comment crs:` after the format line and the case of
/// application property names restored (PDAL lowercases them); only the header changes.
fn annotate_ply(path: &Path, crs: Option<String>, names: &[String]) -> Res<()> {
    use std::io::{BufRead, BufReader, Write};
    let mut by_key: BTreeMap<String, String> = BTreeMap::new();
    for name in names {
        let key = name.to_lowercase();
        if let Some(other) = by_key.get(&key) {
            if other != name { return Err(format!("PLY property names differ only in case: {other}, {name}")); }
        }
        by_key.insert(key, name.clone());
    }
    by_key.retain(|k, v| !["x", "y", "z", "red", "green", "blue"].contains(&k.as_str()) && k != v);
    if crs.is_none() && by_key.is_empty() { return Ok(()); }
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::with_capacity(16 << 20, file);
    let mut lines: Vec<Vec<u8>> = Vec::new();
    loop {
        let mut line = Vec::new();
        reader.read_until(b'\n', &mut line).map_err(|e| e.to_string())?;
        if line.is_empty() || lines.len() >= 200 || line.len() > 8192 || !line.ends_with(b"\n") {
            return Err(format!("{}: invalid or oversized PLY header", path.display()));
        }
        let end = line.strip_suffix(b"\n").map(|l| l.strip_suffix(b"\r").unwrap_or(l)) == Some(&b"end_header"[..]);
        lines.push(line);
        if end { break; }
    }
    if lines[0].strip_suffix(b"\n").map(|l| l.strip_suffix(b"\r").unwrap_or(l)) != Some(&b"ply"[..]) || !lines[1].starts_with(b"format ") {
        return Err(format!("{}: not a PLY file", path.display()));
    }
    let newline: &[u8] = if lines[0].ends_with(b"\r\n") { b"\r\n" } else { b"\n" };
    let mut kept: Vec<Vec<u8>> = lines.into_iter().map(|line| {
        if !line.starts_with(b"property ") { return line; }
        let body_len = line.len() - if line.ends_with(b"\r\n") { 2 } else { 1 };
        let body = &line[..body_len];
        let split = body.iter().rposition(|&c| c == b' ').unwrap();
        match by_key.get(std::str::from_utf8(&body[split + 1..]).unwrap_or("")) {
            None => line.clone(),
            Some(original) => [&body[..=split], original.as_bytes(), &line[body_len..]].concat(),
        }
    }).collect();
    if let Some(value) = crs {
        kept.retain(|l| !l.to_ascii_lowercase().starts_with(b"comment crs:"));
        kept.insert(2, [b"comment crs: ".as_slice(), value.as_bytes(), newline].concat());
    }
    let temporary = path.with_file_name(format!(".{}.crs", path.file_name().unwrap().to_string_lossy()));
    let result = (|| -> std::io::Result<()> {
        let mut out = std::io::BufWriter::with_capacity(16 << 20, fs::File::create(&temporary)?);
        for line in &kept { out.write_all(line)?; }
        std::io::copy(&mut reader, &mut out)?;
        out.flush()
    })();
    if let Err(e) = result {
        let _ = fs::remove_file(&temporary);
        return Err(e.to_string());
    }
    fs::rename(&temporary, path).map_err(|e| e.to_string())
}

/// `_finalize_product` for PLY: the CRS comment and the application property names.
fn finalize_ply(output: &Path, sources: &[PathBuf], metadata_source: &Path) -> Res<()> {
    let mut names = vec!["OriginalPointIndex".to_string()];
    for s in sources {
        names.extend(PointLayout::of(&lasio::read_header(s)?)?.extra.iter().map(|(e, _)| e.name.clone()));
    }
    names.sort();
    names.dedup();
    annotate_ply(output, crate::crs::ply_comment(&lasio::read_header(metadata_source)?), &names)
}

/// Raw LAS/LAZ (one per source key) and COPC files of a directory, by lowercase name.
fn listed_files(directory: &Path) -> Res<(Vec<PathBuf>, Vec<PathBuf>)> {
    let mut files: Vec<PathBuf> = fs::read_dir(directory).map_err(|e| format!("{}: {e}", directory.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && { let n = p.file_name().unwrap().to_string_lossy().to_lowercase(); n.ends_with(".laz") || n.ends_with(".las") })
        .collect();
    files.sort_by_key(|p| p.file_name().unwrap().to_string_lossy().to_lowercase());
    let is_copc = |p: &PathBuf| p.file_name().unwrap().to_string_lossy().to_lowercase().ends_with(".copc.laz");
    let (copc, raw): (Vec<PathBuf>, Vec<PathBuf>) = files.into_iter().partition(is_copc);
    let mut keys: BTreeMap<String, &PathBuf> = BTreeMap::new();
    for path in &raw {
        if let Some(other) = keys.insert(source_key(path), path) {
            return Err(format!("{} and {} name the same source; rename one of them",
                               other.file_name().unwrap().to_string_lossy(), path.file_name().unwrap().to_string_lossy()));
        }
    }
    Ok((raw, copc))
}

/// `extra_bytes_params_compatible`: same storage type, scale, offset and no-data (NaN equal to NaN).
fn eb_compatible(a: &ExtraBytes, b: &ExtraBytes) -> bool {
    let same_no_data = match (a.has_no_data(), b.has_no_data()) {
        (false, false) => true,
        (true, true) => match a.scalar() {
            Some(Scalar::F32 | Scalar::F64) => {
                let (x, y) = (f64::from_le_bytes(a.no_data[..8].try_into().unwrap()), f64::from_le_bytes(b.no_data[..8].try_into().unwrap()));
                x == y || (x.is_nan() && y.is_nan())
            }
            _ => a.no_data[..8] == b.no_data[..8],
        },
        _ => false,
    };
    a.data_type == b.data_type && same_no_data
        && a.has_scale() == b.has_scale() && (!a.has_scale() || a.scale[0] == b.scale[0])
        && a.has_offset() == b.has_offset() && (!a.has_offset() || a.offset[0] == b.offset[0])
}

/// `validate_collection_extra_bytes`: one ExtraBytes interpretation across the sources.
fn validate_collection(paths: &[PathBuf]) -> Res<()> {
    let mut expected: Option<Vec<ExtraBytes>> = None;
    for path in paths {
        let actual: Vec<ExtraBytes> = PointLayout::of(&lasio::read_header(path)?)?.extra.into_iter().map(|(e, _)| e).collect();
        match &expected {
            None => expected = Some(actual),
            Some(first) => {
                let mut a: Vec<&str> = first.iter().map(|e| e.name.as_str()).collect();
                let mut b: Vec<&str> = actual.iter().map(|e| e.name.as_str()).collect();
                a.sort(); b.sort();
                if a != b || first.iter().any(|e| !actual.iter().any(|x| x.name == e.name && eb_compatible(e, x))) {
                    return Err(format!("{}: ExtraBytes encoding or missingness differs from {}",
                                       path.file_name().unwrap().to_string_lossy(), paths[0].file_name().unwrap().to_string_lossy()));
                }
            }
        }
    }
    Ok(())
}

fn source_fingerprint(source: &Path) -> Res<Option<Value>> {
    let Some(header) = header_signature(source)? else { return Ok(None) };
    let meta = fs::metadata(source).map_err(|e| e.to_string())?;
    let mtime = meta.modified().map_err(|e| e.to_string())?.duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos() as u64;
    Ok(Some(json!({"header": header, "mtime_ns": mtime, "name": source.file_name().unwrap().to_string_lossy(), "size": meta.len()})))
}

const MANIFEST: &str = ".smarttile_copc_manifest.json";

fn read_manifest(dir: &Path) -> serde_json::Map<String, Value> {
    let mut manifest = fs::read_to_string(dir.join(MANIFEST)).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.as_object().cloned()).unwrap_or_default();
    manifest.entry("version").or_insert(json!(1));
    if !manifest.get("sources").is_some_and(Value::is_object) { manifest.insert("sources".into(), json!({})); }
    manifest
}

/// `staged_copc_matches_source`: only a COPC recorded in a staging manifest for this exact source is reused.
fn staged_matches(copc: &Path, source: Option<&Path>) -> bool {
    let Some(source) = source else { return true };
    let manifest = read_manifest(copc.parent().unwrap());
    let Some(entry) = manifest["sources"].get(source_key(source)) else { return false };
    if entry.get("copc_file").and_then(Value::as_str) != Some(&copc.file_name().unwrap().to_string_lossy()) { return false; }
    match (source_fingerprint(source), header_signature(copc)) {
        (Ok(Some(fingerprint)), Ok(Some(header))) => entry.get("source") == Some(&fingerprint) && entry.get("copc_header") == Some(&header),
        _ => false,
    }
}

fn reusable_copc(path: &Path) -> bool {
    match lasio::read_header(path) {
        Ok(h) => h.point_count > 0,
        Err(e) => { println!("    Ignoring unreadable staged COPC {}: {e}", path.file_name().unwrap().to_string_lossy()); false }
    }
}

/// `_copc_by_source`: fresh staged COPCs of a directory by source key.
fn copc_by_source(dir: Option<&Path>, raw_by_key: &BTreeMap<String, PathBuf>) -> Res<BTreeMap<String, PathBuf>> {
    let mut found = BTreeMap::new();
    let Some(dir) = dir.filter(|d| d.exists()) else { return Ok(found) };
    for path in listed_files(dir)?.1 {
        let key = source_key(&path);
        let source = raw_by_key.get(&key);
        if reusable_copc(&path) && staged_matches(&path, source.map(PathBuf::as_path)) {
            found.insert(key, path);
        } else if let Some(source) = source {
            println!("    Ignoring stale staged COPC {}: no fresh source manifest for {}",
                     path.file_name().unwrap().to_string_lossy(), source.file_name().unwrap().to_string_lossy());
        }
    }
    Ok(found)
}

/// `write_copc_stage_manifest_entry` (json.dump(indent=2, sort_keys=True)).
fn write_manifest_entry(source: &Path, copc: &Path) -> Res<()> {
    let dir = copc.parent().unwrap();
    let (Some(fingerprint), Some(header)) = (source_fingerprint(source)?, header_signature(copc)?) else {
        println!("    COPC staging manifest entry skipped: ExtraBytes with no-data, scale or offset are not described by the Rust port yet");
        return Ok(());
    };
    let mut manifest = read_manifest(dir);
    manifest["sources"].as_object_mut().unwrap().insert(source_key(source),
        json!({"copc_file": copc.file_name().unwrap().to_string_lossy(), "copc_header": header, "source": fingerprint}));
    fs::write(dir.join(MANIFEST), crate::pyfmt::dumps(&sort_keys(&Value::Object(manifest)))).map_err(|e| e.to_string())
}

fn sort_keys(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(keys.into_iter().map(|k| (k.clone(), sort_keys(&map[k]))).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_keys).collect()),
        other => other.clone(),
    }
}

/// `prepare_copc_inputs`: one staged COPC per source, reusing fresh ones, converting the rest.
fn stage_copcs(raw: &[PathBuf], copc_inputs: &[PathBuf], output_dir: &Path, staged_dir: Option<&Path>) -> Res<Vec<PathBuf>> {
    let existing: Vec<&PathBuf> = copc_inputs.iter().filter(|p| reusable_copc(p)).collect();
    let copc_dir = output_dir.join("original_with_predictions_copc");
    fs::create_dir_all(&copc_dir).map_err(|e| e.to_string())?;
    let raw_by_key: BTreeMap<String, PathBuf> = raw.iter().map(|p| (source_key(p), p.clone())).collect();
    let mut expected: Vec<String> = raw.iter().chain(existing.iter().copied()).map(|p| source_key(p)).collect();
    expected.sort();
    expected.dedup();
    let mut selected: BTreeMap<String, PathBuf> = BTreeMap::new();
    for path in &existing {
        let key = source_key(path);
        if raw_by_key.get(&key).is_none_or(|source| staged_matches(path, Some(source))) {
            selected.insert(key, (*path).clone());
        }
    }
    for dir in [staged_dir, Some(copc_dir.as_path())] {
        for (key, path) in copc_by_source(dir, &raw_by_key)? {
            if expected.contains(&key) { selected.entry(key).or_insert(path); }
        }
    }
    for input in raw {
        let key = source_key(input);
        if let Some(found) = selected.get(&key) {
            println!("    Reusing existing COPC for {}: {}", input.file_name().unwrap().to_string_lossy(), found.file_name().unwrap().to_string_lossy());
            continue;
        }
        let target = copc_dir.join(format!("{key}.copc.laz"));
        println!("    Converting {} -> {}", input.file_name().unwrap().to_string_lossy(), target.file_name().unwrap().to_string_lossy());
        crate::tile::copc_conversion(input, &target, lasio::read_header(input)?.point_count, true)?;
        write_manifest_entry(input, &target)?;
        selected.insert(key, target);
    }
    Ok(expected.iter().map(|k| selected[k].clone()).collect())
}

/// `join_attribute_collections`: the same originals from several collections, every collection's ExtraBytes.
pub fn join_collections(directories: &[PathBuf], output_dir: &Path) -> Res<()> {
    let mut collections: Vec<BTreeMap<String, PathBuf>> = Vec::new();
    for dir in directories {
        let raw = listed_files(dir)?.0;
        if raw.is_empty() { return Err(format!("{}: no LAS/LAZ files (COPC inputs have lost their record order)", dir.display())); }
        collections.push(raw.into_iter().map(|p| (source_key(&p), p)).collect());
    }
    let keys: Vec<String> = collections[0].keys().cloned().collect();
    for (dir, c) in directories.iter().zip(&collections).skip(1) {
        if c.keys().cloned().collect::<Vec<_>>() != keys {
            return Err(format!("{} holds other originals than {}", dir.display(), directories[0].display()));
        }
    }
    fs::create_dir(output_dir).map_err(|e| format!("{}: {e}", output_dir.display()))?;
    for key in &keys {
        let paths: Vec<PathBuf> = collections.iter().map(|c| c[key].clone()).collect();
        let count = join_file(&paths, &output_dir.join(format!("{key}.laz")))?;
        println!("    Joined {key}: {} points from {} collections", crate::pyfmt::thousands(count), paths.len());
    }
    Ok(())
}

/// `join_file`: the first collection's points and standard fields plus every collection's ExtraBytes.
fn join_file(paths: &[PathBuf], output: &Path) -> Res<u64> {
    let headers = paths.iter().map(|p| lasio::read_header(p)).collect::<Res<Vec<_>>>()?;
    let base = &headers[0];
    for (path, h) in paths.iter().zip(&headers).skip(1) {
        if h.point_count != base.point_count {
            return Err(format!("{}: {} points, {} has {}; collections must hold the same points", path.display(),
                               crate::pyfmt::thousands(h.point_count), paths[0].display(), crate::pyfmt::thousands(base.point_count)));
        }
    }
    let layouts = headers.iter().map(PointLayout::of).collect::<Res<Vec<_>>>()?;
    let standard: Vec<String> = layouts[0].standard.iter().map(|(n, _)| n.clone()).collect();
    // name -> (descriptor, holders as (collection, byte offset))
    let mut fields: Vec<(ExtraBytes, Vec<(usize, usize)>)> = layouts[0].extra.iter().map(|(e, o)| (e.clone(), vec![(0, *o)])).collect();
    for (index, layout) in layouts.iter().enumerate().skip(1) {
        for (e, o) in &layout.extra {
            if standard.contains(&e.name) { continue; }
            if let Some((known, holders)) = fields.iter_mut().find(|(k, _)| k.name == e.name) {
                if !eb_compatible(known, e) {
                    return Err(format!("{} has an incompatible schema in {} and {}", e.name, paths[holders[0].0].display(), paths[index].display()));
                }
                holders.push((index, *o));
            } else {
                fields.push((e.clone(), vec![(index, *o)]));
            }
        }
    }
    if fields.iter().any(|(e, _)| e.scalar().is_none()) {
        return Err("joining vector ExtraBytes is not ported to Rust yet".into());
    }
    let mut header = TileHeader::copy_of(base)?;
    let existing = layouts[0].extra.len();
    header.extra.extend(fields[existing..].iter().map(|(e, _)| OutputExtra::new(e)));
    header.sync_extra_bytes();
    let record_base = crate::points::base_record_length(base.point_format);
    let tolerance = headers.iter().flat_map(|h| h.scales).fold(0.0, f64::max) / 2.0;
    let mut readers = paths.iter().map(|p| chunks(p, READ_POINTS)).collect::<Res<Vec<_>>>()?;
    let mut written = 0u64;
    let batches = std::iter::from_fn(|| {
        let mut current = Vec::with_capacity(readers.len());
        for r in readers.iter_mut() {
            match r.next() {
                None => return None,
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(c)) => current.push(c),
            }
        }
        let n = current[0].len();
        let rows: Vec<Vec<&[u8]>> = current.iter().map(|c| c.raw_bytes().chunks_exact(c.record_len()).collect()).collect();
        for (k, rs) in rows.iter().enumerate().skip(1) {
            let same = rs.len() == n && (0..n).all(|i| {
                let (a, b) = (rows[0][i], rs[i]);
                if headers[k].scales == base.scales && headers[k].offsets == base.offsets { a[..12] == b[..12] } else {
                    (0..3).all(|ax| {
                        let w = |r: &[u8], h: &SourceHeader| h.scales[ax] * f64::from(i32::from_le_bytes(r[ax * 4..ax * 4 + 4].try_into().unwrap())) + h.offsets[ax];
                        (w(b, &headers[k]) - w(a, base)).abs() <= tolerance
                    })
                }
            });
            if !same {
                return Some(Err(format!("{} and {} differ in point coordinates or order near record {}; collections must hold the same points",
                                        paths[k].display(), paths[0].display(), crate::pyfmt::thousands(written))));
            }
        }
        let mut out = Vec::with_capacity(n * header.record_length());
        for i in 0..n {
            out.extend_from_slice(&rows[0][i][..record_base]);
            for (e, holders) in &fields {
                let size = e.size();
                let (c, o) = holders[0];
                let value = &rows[c][i][o..o + size];
                for &(other, oo) in &holders[1..] {
                    let theirs = &rows[other][i][oo..oo + size];
                    if crate::points::compare_field(e.scalar().unwrap(), value, theirs) != Ordering::Equal {
                        return Some(Err(format!("{} differs between {} and {}; rename it in one collection", e.name, paths[c].display(), paths[other].display())));
                    }
                }
                out.extend_from_slice(value);
            }
        }
        written += n as u64;
        Some(Ok(out))
    });
    let count = lasio::write_records(output, &header, batches, true, None)?;
    if count != base.point_count {
        return Err(format!("Joined {count} of {} points of {}", base.point_count, paths[0].display()));
    }
    Ok(count)
}

/// `point_cloud_header_signature` (copc_staging.py); None for ExtraBytes it cannot describe exactly yet.
fn header_signature(path: &Path) -> Res<Option<Value>> {
    let h = lasio::read_header(path)?;
    let mut extras = Vec::new();
    for (e, _) in &PointLayout::of(&h)?.extra {
        let Some(scalar) = e.scalar() else { return Ok(None) };
        if e.has_no_data() || e.has_scale() || e.has_offset() { return Ok(None); }
        let dtype = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "float32", "float64"][scalar as usize];
        extras.push(json!({"dtype": dtype, "name": e.name, "no_data": null, "offsets": null, "scales": null}));
    }
    let f = crate::pyfmt::float;
    Ok(Some(json!({
        "bounds": [f(h.mins[0]), f(h.maxs[0]), f(h.mins[1]), f(h.maxs[1]), f(h.mins[2]), f(h.maxs[2])],
        "extra_dimensions": extras,
        "offsets": h.offsets.map(f), "point_count": h.point_count, "point_format": h.point_format,
        "scales": h.scales.map(f), "version": format!("{}.{}", h.version.0, h.version.1),
    })))
}

fn untwine(inputs: &[PathBuf], output: &Path, srs_from: &Path, temp: Option<&Path>) -> Res<()> {
    let mut command = Command::new("untwine");
    for i in inputs { command.arg("-i").arg(i); }
    command.arg("-o").arg(output);
    if let Some(srs) = crate::crs::readable_crs(&lasio::read_header(srs_from)?).and_then(|c| c.authority_string()) {
        command.arg("--a_srs").arg(srs);
    }
    if let Some(t) = temp {
        fs::create_dir_all(t).map_err(|e| e.to_string())?;
        command.arg("--temp_dir").arg(t);
    }
    let result = command.output().map_err(|e| format!("untwine: {e}"))?;
    if !result.status.success() || !output.exists() {
        return Err(format!("untwine failed: {}", String::from_utf8_lossy(&result.stderr)));
    }
    Ok(())
}

fn pdal_pipeline(pipeline: &Value) -> Res<()> {
    use std::io::Write as _;
    let mut child = Command::new("pdal").args(["pipeline", "--stdin"]).stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::piped()).spawn().map_err(|e| format!("pdal: {e}"))?;
    child.stdin.take().unwrap().write_all(pipeline.to_string().as_bytes()).map_err(|e| e.to_string())?;
    let result = child.wait_with_output().map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(format!("PDAL pipeline failed: {}", String::from_utf8_lossy(&result.stderr)));
    }
    Ok(())
}

/// `_finalize_product`: every source dimension survived and the CRS is the source's.
fn finalize(output: &Path, sources: &[PathBuf], metadata_source: &Path) -> Res<()> {
    let names = |h: &SourceHeader| -> Res<Vec<String>> {
        let l = PointLayout::of(h)?;
        Ok(l.standard.iter().map(|(n, _)| n.clone()).chain(l.extra.iter().map(|(e, _)| e.name.clone())).collect())
    };
    let out = lasio::read_header(output)?;
    let have = names(&out)?;
    for s in sources {
        for n in names(&lasio::read_header(s)?)? {
            if !have.contains(&n) && !(n == "scan_angle_rank" && have.contains(&"scan_angle".to_string())) {
                return Err(format!("prod-merged output dropped point dimension {n}"));
            }
        }
    }
    // `preserve_extra_bytes_missingness`: the repair of a dropped no-data value is not ported; refuse instead.
    let source = lasio::read_header(metadata_source)?;
    let out_layout = PointLayout::of(&out)?;
    for (e, _) in &PointLayout::of(&source)?.extra {
        if e.has_no_data() && !out_layout.extra.iter().any(|(o, _)| o.name == e.name && o.has_no_data()) {
            return Err(format!("{}: no_data of {} was dropped; its repair is not ported to Rust yet", output.display(), e.name));
        }
    }
    let (ok, message) = crate::crs::validate_single_crs_record(&source, &out);
    if !ok { return Err(format!("CRS validation failed: {message}")); }
    Ok(())
}

/// `aligned_edges`: chunk edges on the absolute decimal voxel grid, clipped to [start, stop].
fn aligned_edges(start: f64, stop: f64, step: f64, align: f64) -> Res<Vec<(f64, f64)>> {
    let grid = Ratio::from_f64(align)?;
    let (s, e) = (Ratio::from_f64(start)?, Ratio::from_f64(stop)?);
    let cells = (py_round(step / align) as i128).max(1);
    let mut position = Ratio::int(s.floor_div(grid).ok_or("overflow")?).checked_mul(grid).ok_or("overflow")?;
    let stride = Ratio::int(cells).checked_mul(grid).ok_or("overflow")?;
    let lt = |a: Ratio, b: Ratio| a.num * b.den < b.num * a.den;
    let mut edges = Vec::new();
    while lt(position, e) {
        let next = position.checked_add(stride).ok_or("overflow")?;
        let end = if lt(next, e) { next } else { e };
        if lt(s, end) {
            edges.push(((if lt(position, s) { s } else { position }).to_f64(), end.to_f64()));
        }
        position = end;
    }
    Ok(edges)
}

/// `select_voxel_points`: one original point per voxel (nearest the mean; ties by all fields), in GPS-time order.
fn select_voxel_points(records: &[&[u8]], h: &SourceHeader, resolution: f64) -> Res<Vec<usize>> {
    let layout = PointLayout::of(h)?;
    let mut fields = crate::points::raw_fields(h.point_format);
    for (e, o) in &layout.extra {
        let scalar = e.scalar().ok_or("vector ExtraBytes are not ported in product selection")?;
        fields.push((*o, scalar));
    }
    let gps_offset = match h.point_format { 1 | 3 | 4 | 5 => Some(20), 6..=10 => Some(22), _ => None };
    let n = records.len();
    let mut canonical: Vec<usize> = (0..n).collect();
    canonical.sort_by(|&a, &b| {
        for (o, s) in &fields {
            let c = crate::points::compare_field(*s, &records[a][*o..], &records[b][*o..]);
            if c != Ordering::Equal { return c; }
        }
        Ordering::Equal
    });
    let xyz: Vec<[i32; 3]> = canonical.iter().map(|&i| std::array::from_fn(|a| i32::from_le_bytes(records[i][a * 4..a * 4 + 4].try_into().unwrap()))).collect();
    let c = [crate::decimal::coefficients(h.scales[0], h.offsets[0], resolution)?, crate::decimal::coefficients(h.scales[1], h.offsets[1], resolution)?,
             crate::decimal::coefficients(h.scales[2], h.offsets[2], resolution)?];
    let keys = smarttile_native::voxel_keys(&xyz, &c).map_err(|_| "Voxel keys exceed the int64 range".to_string())?;
    let mut group: Vec<usize> = (0..n).collect();
    group.sort_by_key(|&k| (keys[k], k));
    let mut chosen = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && keys[group[end]] == keys[group[start]] { end += 1; }
        let members = &group[start..end];
        let mut sums = [0i64; 3];
        for &m in members { for a in 0..3 { sums[a] += i64::from(xyz[m][a]); } }
        let count = members.len() as f64;
        let mean: [f64; 3] = std::array::from_fn(|a| sums[a] as f64 / count);
        let distance = |m: usize| {
            let d: [f64; 3] = std::array::from_fn(|a| (f64::from(xyz[m][a]) - mean[a]) * h.scales[a]);
            (d[0] * d[0] + d[2] * d[2]) + d[1] * d[1]
        };
        // Lowest distance; exact ties to the lowest X, Y, Z, then canonical order.
        let best = members.iter().copied().min_by(|&a, &b| distance(a).total_cmp(&distance(b)).then(xyz[a].cmp(&xyz[b])).then(a.cmp(&b))).unwrap();
        chosen.push(best);
        start = end;
    }
    // GPS time, then canonical rank.
    chosen.sort_by(|&a, &b| {
        let by_time = gps_offset.map(|o| crate::points::compare_field(Scalar::F64, &records[canonical[a]][o..], &records[canonical[b]][o..])).unwrap_or(Ordering::Equal);
        by_time.then(a.cmp(&b))
    });
    Ok(chosen.into_iter().map(|rank| canonical[rank]).collect())
}

/// One spatial strip: PDAL reads it from the staged COPCs, this keeps the owned voxels' points.
#[allow(clippy::too_many_arguments)]
fn product_chunk(copcs: &[PathBuf], work: &Path, index: usize, bounds: &str, lo: Option<i64>, hi: Option<i64>, resolution: f64,
                 scale_offset: &serde_json::Map<String, Value>) -> Res<Option<PathBuf>> {
    let raw = work.join(format!("prod_merged_chunk{index:04}.raw.las"));
    let chunk = work.join(format!("prod_merged_chunk{index:04}.laz"));
    let mut stages: Vec<Value> = copcs.iter().map(|c| json!({"type": "readers.copc", "filename": c.display().to_string(), "bounds": bounds})).collect();
    if copcs.len() > 1 { stages.push(json!({"type": "filters.merge"})); }
    let mut writer = json!({"type": "writers.las", "filename": raw.display().to_string(), "compression": false, "minor_version": 4,
                            "forward": "all", "extra_dims": "all"});
    for (k, v) in scale_offset { writer[k] = v.clone(); }
    stages.push(writer);
    pdal_pipeline(&json!({"pipeline": stages}))?;
    if !raw.exists() || fs::metadata(&raw).map(|m| m.len()).unwrap_or(0) == 0 { return Ok(None); }
    let h = lasio::read_header(&raw)?;
    let mut bytes = Vec::new();
    for c in chunks(&raw, u64::MAX >> 1)? { bytes.extend_from_slice(c?.raw_bytes()); }
    let record_len = h.record_length as usize;
    let c0 = crate::decimal::coefficients(h.scales[0], h.offsets[0], resolution)?;
    let records: Vec<&[u8]> = bytes.chunks_exact(record_len).filter(|r| {
        let x = i32::from_le_bytes(r[0..4].try_into().unwrap());
        let column = (i128::from(x) * c0.0 + c0.1).div_euclid(c0.2) as i64;
        lo.is_none_or(|l| column >= l) && hi.is_none_or(|h| column < h)
    }).collect();
    let _ = fs::remove_file(&raw);
    if records.is_empty() { return Ok(None); }
    let selected = select_voxel_points(&records, &h, resolution)?;
    let mut out = Vec::with_capacity(selected.len() * record_len);
    for i in selected { out.extend_from_slice(records[i]); }
    lasio::write_records(&chunk, &TileHeader::verbatim(&h)?, vec![Ok(out)], true, None)?;
    Ok(Some(chunk))
}

/// Returns the published products: full resolution first, each resolution in every format.
pub fn run(params: &ProductParams) -> Res<Vec<PathBuf>> {
    let started = std::time::Instant::now();
    let (raw, copc_inputs) = listed_files(&params.input)?;
    // A colocated COPC is another encoding of its raw twin; one without a twin is a source of its own.
    let raw_keys: Vec<String> = raw.iter().map(|p| source_key(p)).collect();
    let mut sources: Vec<PathBuf> = raw.iter().chain(copc_inputs.iter().filter(|p| !raw_keys.contains(&source_key(p)))).cloned().collect();
    sources.sort_by_key(|p| p.file_name().unwrap().to_string_lossy().to_lowercase());
    if sources.is_empty() { return Err(format!("No LAZ/LAS files found in {}", params.input.display())); }
    validate_collection(&sources)?;
    let mut selections = params.resolutions.clone();
    selections.sort_by_key(|(label, _)| label != "full");
    let selections_in_order = selections.clone();
    fs::create_dir_all(&params.output_dir).map_err(|e| e.to_string())?;
    let scratch = params.output_dir.join(format!(".products-{}", std::process::id()));
    fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
    let mut staged: Vec<(PathBuf, PathBuf)> = Vec::new();
    let result = (|| -> Res<()> {
        if selections.iter().any(|(l, _)| l == "full") {
            for format in &params.formats {
                let target = scratch.join(output_path(Path::new(""), "full", format));
                println!("  Creating {} at full resolution from {} file(s) in record order{}", target.file_name().unwrap().to_string_lossy(), sources.len(),
                         if params.original_point_index { " with OriginalPointIndex" } else { "" });
                match format.as_str() {
                    "laz" => { write_full(&sources, &target, true, params.original_point_index, false)?; }
                    "copc.laz" => {
                        let las = scratch.join("full-merge.las");
                        write_full(&sources, &las, false, params.original_point_index, false)?;
                        let count = lasio::read_header(&las)?.point_count;
                        crate::tile::copc_conversion(&las, &target, count, true)?;
                        let _ = fs::remove_file(&las);
                    }
                    _ => {
                        let las = scratch.join("full-merge.las");
                        write_full(&sources, &las, false, params.original_point_index, true)?;
                        // LAS 1.2/1.3 can carry ExtraBytes VLRs, which PDAL ignores by default.
                        let result = Command::new("pdal").arg("translate").arg(&las).arg(&target)
                            .args(["--readers.las.use_eb_vlr=true", "--writers.ply.storage_mode=little endian"])
                            .output().map_err(|e| format!("pdal: {e}"))?;
                        let _ = fs::remove_file(&las);
                        if !result.status.success() {
                            return Err(format!("PLY conversion failed: {}", String::from_utf8_lossy(&result.stderr)));
                        }
                    }
                }
                if format == "ply" { finalize_ply(&target, &sources, &sources[0])? } else { finalize(&target, &sources, &sources[0])? }
                staged.push((target, output_path(&params.output_dir, "full", format)));
            }
        }
        let subsampled: Vec<(String, f64)> = selections.into_iter().filter(|(l, _)| l != "full").collect();
        if subsampled.is_empty() { return Ok(()); }
        println!("  Preparing COPC inputs with SmartTile untwine/PDAL fallback conversion");
        let copcs = stage_copcs(&raw, &copc_inputs, &params.output_dir, params.staged_copc_dir.as_deref())?;
        validate_collection(&copcs)?;
        let headers = copcs.iter().map(|c| lasio::read_header(c)).collect::<Res<Vec<_>>>()?;
        let scales: [f64; 3] = std::array::from_fn(|a| headers.iter().map(|h| h.scales[a]).fold(f64::INFINITY, f64::min));
        let mins: [f64; 3] = std::array::from_fn(|a| headers.iter().map(|h| h.mins[a]).fold(f64::INFINITY, f64::min));
        let maxs: [f64; 3] = std::array::from_fn(|a| headers.iter().map(|h| h.maxs[a]).fold(f64::NEG_INFINITY, f64::max));
        let mut offsets = headers[0].offsets;
        if (0..3).any(|a| (mins[a] - offsets[a]) / scales[a] < i32::MIN as f64 || (maxs[a] - offsets[a]) / scales[a] > i32::MAX as f64) {
            offsets = mins.map(f64::floor);
        }
        let mut scale_offset = serde_json::Map::new();
        for (a, axis) in ["x", "y", "z"].iter().enumerate() {
            scale_offset.insert(format!("scale_{axis}"), crate::pyfmt::float(scales[a]));
            scale_offset.insert(format!("offset_{axis}"), crate::pyfmt::float(offsets[a]));
        }
        for (label, resolution) in subsampled {
            // `_prod_merged_chunk_bounds`
            let margin = headers.iter().map(|h| h.scales[0].max(h.scales[1])).fold(0.0, f64::max);
            let (minx, maxx) = (mins[0], maxs[0]);
            let (miny, maxy) = (mins[1] - margin, maxs[1] + margin);
            let total: u64 = headers.iter().map(|h| h.point_count).sum();
            let count = (params.workers.max(1) as f64).max((total as f64 / PRODUCT_CHUNK_POINTS).ceil()) as usize;
            let raw_step = (maxx - minx) / count as f64;
            let x_step = resolution.max((raw_step / resolution).ceil() * resolution);
            let r = float_repr;
            let edges = if maxx - minx == 0.0 || maxy - miny == 0.0 { vec![(minx, maxx)] } else { aligned_edges(minx, maxx, x_step, resolution)? };
            let bounds: Vec<(String, Option<i64>, Option<i64>)> = edges.iter().enumerate().map(|(i, (a, b))| (
                format!("([{},{}],[{},{}])", r(a - margin), r(b + margin), r(miny), r(maxy)),
                if i == 0 { None } else { Some(py_round(a / resolution) as i64) },
                if i + 1 == edges.len() { None } else { Some(py_round(b / resolution) as i64) })).collect();
            let work = scratch.join(format!("_{label}_shared_chunks"));
            fs::create_dir_all(&work).map_err(|e| e.to_string())?;
            println!("    Spatial chunks: {}", bounds.len());
            let pool = rayon::ThreadPoolBuilder::new().num_threads(params.workers.max(1).min(bounds.len())).build().map_err(|e| e.to_string())?;
            let made: Vec<Res<Option<PathBuf>>> = pool.install(|| bounds.par_iter().enumerate()
                .map(|(i, (b, lo, hi))| product_chunk(&copcs, &work, i, b, *lo, *hi, resolution, &scale_offset)).collect());
            let chunk_files: Vec<PathBuf> = made.into_iter().collect::<Res<Vec<_>>>()?.into_iter().flatten().collect();
            if chunk_files.is_empty() { return Err("No prod-merged chunks were created".into()); }
            for format in &params.formats {
                let target = scratch.join(output_path(Path::new(""), &label, format));
                println!("  Creating {} at {}m with nearest-to-centroid", target.file_name().unwrap().to_string_lossy(), g_format(resolution));
                match format.as_str() {
                    "copc.laz" => {
                        untwine(&chunk_files, &target, &copcs[0], Some(&work.join("_untwine_tmp")))?;
                        let (ok, message) = crate::crs::validate_single_crs_record(&lasio::read_header(&copcs[0])?, &lasio::read_header(&target)?);
                        if !ok { return Err(format!("CRS validation failed: {message}")); }
                    }
                    "laz" => {
                        let mut stages: Vec<Value> = chunk_files.iter().map(|c| json!({"type": "readers.las", "filename": c.display().to_string()})).collect();
                        if chunk_files.len() > 1 { stages.push(json!({"type": "filters.merge"})); }
                        let mut writer = json!({"type": "writers.las", "filename": target.display().to_string(), "compression": true, "forward": "all", "extra_dims": "all"});
                        for (k, v) in &scale_offset { writer[k] = v.clone(); }
                        stages.push(writer);
                        pdal_pipeline(&json!({"pipeline": stages}))?;
                        finalize(&target, &chunk_files, &copcs[0])?;
                    }
                    _ => {
                        let mut stages: Vec<Value> = chunk_files.iter().map(|c| json!({"type": "readers.las", "filename": c.display().to_string()})).collect();
                        if chunk_files.len() > 1 { stages.push(json!({"type": "filters.merge"})); }
                        stages.push(json!({"type": "writers.ply", "filename": target.display().to_string(), "storage_mode": "little endian"}));
                        pdal_pipeline(&json!({"pipeline": stages}))?;
                        finalize_ply(&target, &chunk_files, &copcs[0])?;
                    }
                }
                staged.push((target, output_path(&params.output_dir, &label, format)));
            }
            let _ = fs::remove_dir_all(&work);
        }
        Ok(())
    })();
    if result.is_ok() {
        for (from, to) in &staged {
            fs::rename(from, to).map_err(|e| format!("publishing {}: {e}", to.display()))?;
        }
    }
    let _ = fs::remove_dir_all(&scratch);
    result?;
    println!("  Prod-merged outputs ({:.1} s):", started.elapsed().as_secs_f64());
    let mut made: Vec<PathBuf> = Vec::new();
    for (label, _) in &selections_in_order {
        for format in &params.formats {
            made.push(output_path(&params.output_dir, label, format));
        }
    }
    for path in &made { println!("    {}", path.display()); }
    Ok(made)
}

/// `merged_product_collections`: products by format, collections in order of first appearance.
pub fn collections(products: &[PathBuf]) -> Res<Vec<(String, Vec<PathBuf>)>> {
    let mut out: Vec<(String, Vec<PathBuf>)> = Vec::new();
    for path in products {
        let name = path.file_name().unwrap().to_string_lossy().to_lowercase();
        let collection = if name.ends_with(".copc.laz") { "merged_copc" } else if name.ends_with(".laz") { "merged_laz" }
                         else if name.ends_with(".ply") { "merged_ply" } else { return Err(format!("Unexpected prod-merged product: {}", path.display())) };
        match out.iter_mut().find(|(c, _)| c == collection) {
            Some((_, paths)) => paths.push(path.clone()),
            None => out.push((collection.to_string(), vec![path.clone()])),
        }
    }
    Ok(out)
}
