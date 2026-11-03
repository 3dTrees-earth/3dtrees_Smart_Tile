//! SmartTile v2.5: the SmartTile tasks in Rust, ported task by task from
//! Python SmartTile 2.4 and checked against its outputs.
//!
//! The command line follows `python src/run.py` so the Galaxy wrapper can
//! switch binaries per task. Flags of features not ported yet are rejected.

mod collections;
mod crs;
mod decimal;
mod instance;
mod lasio;
mod merge;
mod ld;
mod overview;
mod points;
mod proj;
mod products;
mod pyfmt;
mod remap;
mod scan;
mod sha256;
mod spatial;
mod tile;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use scan::Format;

const PORTED_TASKS: &[&str] = &["tile", "merge", "remap", "create_merged_file"];

fn parse_args() -> Result<BTreeMap<String, String>, String> {
    let mut args = BTreeMap::new();
    let mut iter = std::env::args().skip(1).peekable();
    while let Some(arg) = iter.next() {
        let Some(name) = arg.strip_prefix("--") else {
            return Err(format!("Unexpected argument {arg}"));
        };
        let (name, value) = match name.split_once('=') {
            Some((n, v)) => (n.to_string(), v.to_string()),
            None => match iter.peek() {
                Some(next) if !next.starts_with("--") => (name.to_string(), iter.next().unwrap()),
                _ => (name.to_string(), "true".to_string()),
            },
        };
        args.insert(name.replace('_', "-"), value);
    }
    Ok(args)
}

fn number(args: &BTreeMap<String, String>, name: &str) -> Result<Option<f64>, String> {
    args.get(name).map(|v| v.parse::<f64>().map_err(|_| format!("--{name} must be a number, not {v}"))).transpose()
}

/// `parse_tile_outputs`: 'resolution:format' pairs, `original` meaning every point.
fn parse_tile_outputs(value: &str) -> Result<Vec<(f64, Format)>, String> {
    let mut outputs: Vec<(f64, Format)> = Vec::new();
    for token in value.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let (resolution, format) = token.split_once(':').ok_or_else(|| format!("Tile output '{token}' must be 'resolution:format'"))?;
        let format = match format.trim().to_lowercase().as_str() {
            "laz" => Format::Laz,
            "copc" | "copc.laz" => Format::Copc,
            "ply" => Format::Ply,
            other => return Err(format!("Tile output '{token}': unknown format {other}")),
        };
        let original = resolution.trim().eq_ignore_ascii_case("original");
        let resolution = if original { 0.0 } else { resolution.trim().parse::<f64>().unwrap_or(f64::NAN) };
        if !(original || (resolution > 0.0 && resolution < 1e6)) {
            return Err(format!("Tile output '{token}' must use 'original' or a positive resolution in metres"));
        }
        if outputs.iter().any(|(r, _)| *r == resolution) {
            return Err(format!("Tile output '{token}': resolution {resolution} m is listed twice; each resolution may appear once"));
        }
        outputs.push((resolution, format));
    }
    Ok(outputs)
}

fn allocated_cpus() -> usize {
    let mut limit = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if let Some(slots) = std::env::var("GALAXY_SLOTS").ok().and_then(|s| s.parse::<usize>().ok()) {
        limit = limit.min(slots.max(1));
    }
    limit
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let task = args.get("task").map(String::as_str).unwrap_or("");
    if !PORTED_TASKS.contains(&task) {
        return Err(format!("--task {task:?} is not ported to Rust yet (ported: {})", PORTED_TASKS.join(", ")));
    }
    if task == "merge" {
        return run_merge(&args);
    }
    if task == "remap" {
        return run_remap(&args);
    }
    if task == "create_merged_file" {
        return run_create_merged_file(&args);
    }
    let known = ["task", "input-dir", "output-dir", "tile-length", "tile-buffer", "tiling-threshold", "workers",
                 "number-of-threads", "tile-outputs", "subsampling-method", "grid-origin-x", "grid-origin-y",
                 "resolution-1", "resolution-2", "output-copc-res1", "output-copc-res2", "collection-dir",
                 "instance-tiles", "instance-tile-dims", "instance-outputs", "instance-resolutions", "max-instance-extent",
                 "allow-reused-instance-ids"];
    let unknown: Vec<&String> = args.keys().filter(|k| !known.contains(&k.as_str())).collect();
    if !unknown.is_empty() {
        return Err(format!("Unknown or not yet ported SmartTile argument(s): {}",
                           unknown.iter().map(|k| format!("--{k}")).collect::<Vec<_>>().join(" ")));
    }
    let required = |name: &str| args.get(name).map(PathBuf::from).ok_or_else(|| format!("--{name} is required for the tile task"));
    let workers = number(&args, "workers")?.or(number(&args, "number-of-threads")?).map(|w| w as usize).unwrap_or_else(allocated_cpus);
    if workers < 1 {
        return Err("--workers must be positive".into());
    }
    let flag = |name: &str, default: bool| args.get(name).map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes")).unwrap_or(default);
    let csv = |name: &str| -> Vec<String> { args.get(name).map(|v| v.split(',').map(|s| s.trim().to_string()).collect()).unwrap_or_default() };
    let instance_dimensions = csv("instance-tiles");
    if args.contains_key("instance-tiles") && instance_dimensions.iter().any(String::is_empty) {
        return Err("--instance-tiles must contain nonempty comma-separated field names".into());
    }
    let mut distinct = instance_dimensions.clone();
    distinct.sort();
    distinct.dedup();
    if distinct.len() != instance_dimensions.len() {
        return Err("--instance-tiles dimensions must be distinct; duplicate fields were selected".into());
    }
    let instance_outputs = match args.get("instance-outputs") {
        None => None,
        Some(value) => {
            if instance_dimensions.is_empty() {
                return Err("--instance-outputs requires at least one --instance-tiles dimension".into());
            }
            if args.contains_key("tile-outputs") || args.contains_key("instance-resolutions") {
                return Err("--instance-outputs cannot be combined with --tile-outputs or --instance-resolutions".into());
            }
            let outputs = parse_tile_outputs(value)?;
            if outputs.is_empty() {
                return Err("--instance-outputs lists no outputs".into());
            }
            Some(outputs)
        }
    };
    let outputs = match args.get("tile-outputs") {
        Some(value) => parse_tile_outputs(value)?,
        None if !instance_dimensions.is_empty() => Vec::new(),
        None => vec![(number(&args, "resolution-1")?.unwrap_or(0.01), if flag("output-copc-res1", true) { Format::Copc } else { Format::Laz }),
                     (number(&args, "resolution-2")?.unwrap_or(0.1), if flag("output-copc-res2", false) { Format::Copc } else { Format::Laz })],
    };
    if outputs.is_empty() && instance_dimensions.is_empty() {
        return Err("--tile-outputs lists no outputs".into());
    }
    let instance = (!instance_dimensions.is_empty()).then(|| -> Result<tile::InstanceParams, String> {
        let resolutions = args.get("instance-resolutions").map(String::as_str).unwrap_or("0.01").split(',').map(|r| r.trim().parse::<f64>()
            .map_err(|_| format!("--instance-resolutions: {r} is not a number"))).collect::<Result<Vec<_>, _>>()?;
        Ok(tile::InstanceParams {
            dimensions: instance_dimensions.clone(),
            attributes: csv("instance-tile-dims").into_iter().filter(|s| !s.is_empty()).collect(),
            outputs: instance_outputs.clone(),
            resolutions,
            wide: number(&args, "max-instance-extent")?.unwrap_or(50.0),
            check_ids: !flag("allow-reused-instance-ids", false),
        })
    }).transpose()?;
    let nearest = match args.get("subsampling-method").map(String::as_str).unwrap_or("center-of-mass") {
        "center-of-mass" | "center_of_mass" | "com" => false,
        "nearest-to-centroid" | "nearest_to_centroid" | "nearest" => true,
        other => return Err(format!("Unknown --subsampling-method {other}")),
    };
    let grid_origin = match (number(&args, "grid-origin-x")?, number(&args, "grid-origin-y")?) {
        (Some(x), Some(y)) => Some((x, y)),
        (None, None) => None,
        _ => return Err("Both grid-origin-x and grid-origin-y are required".into()),
    };
    let params = tile::TileParams {
        input_dir: required("input-dir")?,
        output_dir: required("output-dir")?,
        tile_length: number(&args, "tile-length")?.unwrap_or(100.0).trunc(),
        tile_buffer: number(&args, "tile-buffer")?.unwrap_or(20.0).trunc(),
        tiling_threshold: number(&args, "tiling-threshold")?,
        workers: workers.min(allocated_cpus()),
        outputs,
        nearest,
        grid_origin,
        collection_dir: args.get("collection-dir").map(PathBuf::from),
        instance,
    };
    tile::run(&params)
}

fn run_merge(args: &BTreeMap<String, String>) -> Result<(), String> {
    let known = ["task", "subsampled-segmented-folder", "subsampled-10cm-folder", "segmented-folders", "segmented-remapped-folder",
                 "original-tiles-dir", "remap-tolerance", "output-tiles-folder", "output-merged-laz", "disable-matching", "no-disable-matching",
                 "filter-anchor", "skip-unfiltered-tiles", "no-skip-unfiltered-tiles", "remap-dims", "original-copc-input-dir", "subsampled-target-folder", "tile-bounds-json",
                 "output-folder", "output-dir", "workers", "number-of-threads", "prediction-transfer-tolerance", "instance-dimension", "stop-after", "overlap-threshold",
                 "original-laz-input-dir", "original-input-dir", "original-laz-output-dir", "collection-dir",
                 "transfer-original-dims-to-merged", "no-transfer-original-dims-to-merged",
                 "reassign-small-instances", "no-reassign-small-instances", "skip-merged-file", "no-skip-merged-file", "max-cluster-size", "max-volume-for-merge"];
    let unknown: Vec<&String> = args.keys().filter(|k| !known.contains(&k.as_str()) && !PRODUCT_FLAGS.contains(&k.as_str())).collect();
    if !unknown.is_empty() {
        return Err(format!("Unknown or not yet ported merge argument(s): {}", unknown.iter().map(|k| format!("--{k}")).collect::<Vec<_>>().join(" ")));
    }
    let tenth = args.get("subsampled-segmented-folder").or(args.get("subsampled-10cm-folder"));
    let remapped = args.get("segmented-remapped-folder");
    let collections: Vec<PathBuf> = match (tenth, args.get("segmented-folders"), remapped) {
        (Some(one), None, None) | (None, None, Some(one)) => vec![PathBuf::from(one)],
        (None, Some(many), None) => many.split(',').map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from).collect(),
        _ => return Err("provide exactly one segmented input mode".into()),
    };
    let segmented = collections[0].clone();
    let mut target = args.get("subsampled-target-folder").or(args.get("original-tiles-dir")).map(PathBuf::from);
    if target.is_none() && tenth.is_some() {
        target = Some(segmented.parent().unwrap().join("subsampled_res1"));
    }
    let ready = remapped.is_some() || (args.contains_key("segmented-folders") && target.is_none());
    let workers = number(args, "workers")?.or(number(args, "number-of-threads")?).map(|w| w as usize).unwrap_or_else(allocated_cpus);
    let params = merge::MergeParams {
        work: args.get("output-folder").or(args.get("output-dir")).map(PathBuf::from)
            .or_else(|| args.get("output-tiles-folder").map(|p| Path::new(p).parent().unwrap_or(Path::new(".")).to_path_buf()))
            .or_else(|| args.get("output-merged-laz").map(|p| Path::new(p).parent().unwrap_or(Path::new(".")).to_path_buf()))
            .unwrap_or_else(|| segmented.parent().unwrap().to_path_buf()),
        collections,
        target: if ready { None } else { target },
        resolution_1: number(args, "resolution-1")?.unwrap_or(0.01),
        remap_tolerance: number(args, "remap-tolerance")?,
        tile_bounds_json: args.get("tile-bounds-json").map(PathBuf::from).ok_or("--tile-bounds-json is required")?,
        workers: workers.min(allocated_cpus()),
        requested_workers: workers,
        transfer_radius: number(args, "prediction-transfer-tolerance")?.unwrap_or(0.1732),
        instance_dimension: args.get("instance-dimension").cloned().unwrap_or_else(|| "PredInstance".into()),
        stop_after: args.get("stop-after").cloned(),
        overlap_threshold: number(args, "overlap-threshold")?.unwrap_or(0.3),
        originals: args.get("original-laz-input-dir").or(args.get("original-input-dir")).map(PathBuf::from),
        original_output: args.get("original-laz-output-dir").map(PathBuf::from),
        skip_merged_file: flag(args, "skip-merged-file"),
        output_tiles: PathBuf::new(),
        output_merged_laz: args.get("output-merged-laz").map(PathBuf::from),
        matching: !flag(args, "disable-matching"),
        anchor: merge::Anchor::parse(args.get("filter-anchor").map(String::as_str).unwrap_or("centroid"))?,
        keep_unfiltered: !flag(args, "skip-unfiltered-tiles"),
        target_dims: args.get("remap-dims").map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()),
        small_instances: if flag(args, "reassign-small-instances") {
            let size = number(args, "max-cluster-size")?.unwrap_or(3000.0);
            let volume = number(args, "max-volume-for-merge")?.unwrap_or(4.0);
            if size < 1.0 || size.fract() != 0.0 { return Err("max_cluster_size must be at least 1".into()); }
            if !(volume.is_finite() && volume > 0.0) { return Err("max_volume_for_merge must be a finite positive volume".into()); }
            Some((size as u64, volume))
        } else { None },
    };
    if let (Some(input), output) = (&params.originals, &params.original_output) {
        if !input.exists() { return Err(format!("LAZ original input directory not found: {}", input.display())); }
        let output = output.clone().unwrap_or_else(|| params.work.join("original_with_predictions"));
        if output.canonicalize().ok() == input.canonicalize().ok() && output.exists() {
            return Err("--original-laz-output-dir/--output-dir must differ from --original-laz-input-dir.".into());
        }
    }
    let mut params = params;
    params.output_tiles = args.get("output-tiles-folder").map(PathBuf::from).unwrap_or_else(|| params.work.join("output_tiles"));
    if let Some(copc) = args.get("original-copc-input-dir") {
        let raw = params.originals.as_ref().ok_or("--original-laz-input-dir is required with --original-copc-input-dir")?;
        validate_copc_twins(Path::new(copc), raw)?;
    }
    if !(params.resolution_1.is_finite() && params.resolution_1 > 0.0) {
        return Err("First subsampling resolution must be finite and positive".into());
    }
    if params.remap_tolerance.is_some_and(|r| !(r.is_finite() && r > 0.0)) {
        return Err("Original remap tolerance must be finite and positive".into());
    }
    let collection_dir = args.get("collection-dir").map(PathBuf::from);
    if let Some(dir) = &collection_dir { collections::check(dir, params.output_tiles.parent().unwrap_or(Path::new(".")))?; }
    if flag(args, "transfer-original-dims-to-merged") { product_settings(args)?; }
    merge::run(&params)?;
    let root = params.output_tiles.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let original_output = params.original_output.clone().unwrap_or_else(|| root.join("original_with_predictions"));
    let made = if params.originals.is_some() { prod_merged_after(args, &original_output, &root, params.workers)? } else { Vec::new() };
    if let Some(dir) = &collection_dir {
        let mut published = vec![("merged_tiles".to_string(), collections::files(&params.output_tiles, &[".laz"]))];
        if params.originals.is_some() {
            published.push(("originals_with_predictions".to_string(), collections::files(&original_output, &[".laz", ".las"])));
        }
        published.extend(products::collections(&made)?);
        collections::publish(dir, &published, &[])?;
    }
    Ok(())
}

/// `_validate_copc_original_lane` + `_validate_copc_laz_source_pairs`: every raw original has a
/// COPC twin with the same point count and bounds (within 2 cm).
fn validate_copc_twins(copc_dir: &Path, raw_dir: &Path) -> Result<(), String> {
    if !copc_dir.exists() { return Err(format!("COPC original input directory not found: {}", copc_dir.display())); }
    let list = |dir: &Path, copc: bool| -> Result<BTreeMap<String, PathBuf>, String> {
        Ok(fs::read_dir(dir).map_err(|e| e.to_string())?.filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| { let n = p.file_name().unwrap().to_string_lossy().to_lowercase(); p.is_file() && (n.ends_with(".laz") || n.ends_with(".las")) && n.ends_with(".copc.laz") == copc })
            .map(|p| (collections::file_stem(&p).to_lowercase(), p)).collect())
    };
    let copcs = list(copc_dir, true)?;
    if copcs.is_empty() { return Err(format!("No COPC LAZ files found in original COPC input dir: {}", copc_dir.display())); }
    let raws = list(raw_dir, false)?;
    if raws.is_empty() { return Err(format!("No raw LAZ/LAS files found in original input dir: {}", raw_dir.display())); }
    let missing: Vec<&String> = raws.keys().filter(|k| !copcs.contains_key(*k)).collect();
    if !missing.is_empty() {
        return Err(format!("COPC original input dir is missing COPC twins for raw originals: {}", missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
    }
    for (key, raw) in &raws {
        let (r, c) = (lasio::read_header(raw)?, lasio::read_header(&copcs[key])?);
        if r.point_count != c.point_count {
            return Err(format!("COPC/raw source pair point-count mismatch for {key}: {} has {}, {} has {}",
                copcs[key].file_name().unwrap().to_string_lossy(), pyfmt::thousands(c.point_count), raw.file_name().unwrap().to_string_lossy(), pyfmt::thousands(r.point_count)));
        }
        if (0..3).any(|a| (r.mins[a] - c.mins[a]).abs() > 0.02 || (r.maxs[a] - c.maxs[a]).abs() > 0.02) {
            return Err(format!("COPC/raw source pair bounds mismatch for {key}"));
        }
    }
    Ok(())
}

fn run_remap(args: &BTreeMap<String, String>) -> Result<(), String> {
    let known = ["task", "segmented-folders", "merged-laz", "original-laz-input-dir", "original-input-dir", "original-laz-output-dir", "output-dir",
                 "workers", "number-of-threads", "remap-dims", "instance-dimension", "collection-dir",
                 "transfer-original-dims-to-merged", "no-transfer-original-dims-to-merged", "original-copc-input-dir"];
    let unknown: Vec<&String> = args.keys().filter(|k| !known.contains(&k.as_str()) && !PRODUCT_FLAGS.contains(&k.as_str())).collect();
    if !unknown.is_empty() {
        return Err(format!("Unknown or not yet ported remap argument(s): {}", unknown.iter().map(|k| format!("--{k}")).collect::<Vec<_>>().join(" ")));
    }
    let originals = args.get("original-laz-input-dir").or(args.get("original-input-dir")).map(PathBuf::from);
    let (Some(originals), true) = (originals, args.contains_key("segmented-folders") != args.contains_key("merged-laz")) else {
        return Err("remap requires raw originals and exactly one of --segmented-folders or --merged-laz".into());
    };
    let collections: Vec<PathBuf> = match args.get("segmented-folders") {
        Some(folders) => folders.split(',').map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from).collect(),
        None => vec![PathBuf::from(&args["merged-laz"])],
    };
    let output = args.get("original-laz-output-dir").or(args.get("output-dir")).map(PathBuf::from)
        .unwrap_or_else(|| originals.parent().unwrap().join("original_with_predictions"));
    let workers = number(args, "workers")?.or(number(args, "number-of-threads")?).map(|w| w as usize).unwrap_or_else(allocated_cpus);
    let root = output.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let collection_dir = args.get("collection-dir").map(PathBuf::from);
    if let Some(dir) = &collection_dir { collections::check(dir, &root)?; }
    if let Some(copc) = args.get("original-copc-input-dir") { validate_copc_twins(Path::new(copc), &originals)?; }
    if flag(args, "transfer-original-dims-to-merged") { product_settings(args)?; }
    if number(args, "resolution-1")?.is_some_and(|r| r != 0.01) {
        return Err("remap with --resolution-1 other than 0.01 is not ported yet".into());
    }
    remap::run(&remap::RemapParams {
        collections, originals, output: output.clone(), workers: workers.min(allocated_cpus()), requested_workers: workers,
        instance_dimension: args.get("instance-dimension").cloned().unwrap_or_else(|| "PredInstance".into()),
        target_dims: args.get("remap-dims").map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()),
    })?;
    let made = prod_merged_after(args, &output, &root, workers.min(allocated_cpus()))?;
    if let Some(dir) = &collection_dir {
        let mut published = vec![("originals_with_predictions".to_string(), collections::files(&output, &[".laz", ".las"]))];
        published.extend(products::collections(&made)?);
        collections::publish(dir, &published, &[])?;
    }
    Ok(())
}

/// The create_merged_file product options (also used by `--transfer-original-dims-to-merged`).
struct ProductSettings { resolutions: Vec<(String, f64)>, formats: Vec<String>, original_point_index: bool, staged_copc_dir: Option<PathBuf> }

const PRODUCT_FLAGS: [&str; 7] = ["merged-resolutions", "merged-output-formats", "resolution-1", "resolution-2", "original-point-index",
                                  "no-original-point-index", "staged-copc-dir"];

fn product_settings(args: &BTreeMap<String, String>) -> Result<ProductSettings, String> {
    let resolutions = products::parse_resolutions(args.get("merged-resolutions").map(String::as_str).unwrap_or("res1,res2"),
                                                  number(args, "resolution-1")?.unwrap_or(0.01), number(args, "resolution-2")?.unwrap_or(0.1))?;
    let mut formats: Vec<String> = Vec::new();
    for token in args.get("merged-output-formats").map(String::as_str).unwrap_or("copc.laz").trim_matches(['[', ']']).split(',') {
        let token = token.trim().trim_matches(['\'', '"']).to_lowercase();
        if token.is_empty() { continue; }
        let format = match token.as_str() {
            "las" | "laz" | ".laz" => "laz",
            "copc" | "copc_laz" | "copc-laz" | "copc.laz" | ".copc.laz" => "copc.laz",
            "ply" | ".ply" => "ply",
            _ => return Err(format!("Unsupported merged output format '{token}'. Use laz, copc.laz, or ply.")),
        };
        if !formats.iter().any(|f| f == format) { formats.push(format.into()); }
    }
    if formats.is_empty() {
        return Err("No merged output formats selected".into());
    }
    let original_point_index = !args.contains_key("no-original-point-index")
        && args.get("original-point-index").map(|v| !matches!(v.to_lowercase().as_str(), "false" | "0" | "no")).unwrap_or(true);
    Ok(ProductSettings { resolutions, formats, original_point_index, staged_copc_dir: args.get("staged-copc-dir").map(PathBuf::from) })
}

fn flag(args: &BTreeMap<String, String>, name: &str) -> bool {
    !args.contains_key(&format!("no-{name}")) && args.get(name).is_some_and(|v| !matches!(v.to_lowercase().as_str(), "false" | "0" | "no"))
}

/// `_create_prod_merged_outputs` after merge or remap (`--transfer-original-dims-to-merged`).
fn prod_merged_after(args: &BTreeMap<String, String>, originals: &Path, output_dir: &Path, workers: usize) -> Result<Vec<PathBuf>, String> {
    if !flag(args, "transfer-original-dims-to-merged") || !originals.exists() {
        return Ok(Vec::new());
    }
    let settings = product_settings(args)?;
    println!("\n{}\nCreating Prod-Merged Files (Rust SmartTile)\n{}", "=".repeat(60), "=".repeat(60));
    products::run(&products::ProductParams {
        input: originals.to_path_buf(), output_dir: output_dir.to_path_buf(), resolutions: settings.resolutions, formats: settings.formats,
        workers, original_point_index: settings.original_point_index, staged_copc_dir: settings.staged_copc_dir,
    })
}

fn run_create_merged_file(args: &BTreeMap<String, String>) -> Result<(), String> {
    let known = ["task", "original-with-predictions-dir", "input-dir", "output-dir", "workers", "number-of-threads",
                 "merged-resolutions", "merged-output-formats", "resolution-1", "resolution-2", "original-point-index",
                 "no-original-point-index", "staged-copc-dir", "collection-dir"];
    let unknown: Vec<&String> = args.keys().filter(|k| !known.contains(&k.as_str())).collect();
    if !unknown.is_empty() {
        return Err(format!("Unknown or not yet ported create_merged_file argument(s): {}", unknown.iter().map(|k| format!("--{k}")).collect::<Vec<_>>().join(" ")));
    }
    let input = args.get("original-with-predictions-dir").or(args.get("input-dir"))
        .ok_or("--original-with-predictions-dir or --input-dir is required for create_merged_file task")?;
    let inputs: Vec<PathBuf> = input.split(',').map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from).collect();
    let missing: Vec<String> = inputs.iter().filter(|d| !d.exists()).map(|d| d.display().to_string()).collect();
    if inputs.is_empty() || !missing.is_empty() {
        return Err(format!("Original-with-predictions directory not found: {}", missing.join(", ")));
    }
    let output_dir = args.get("output-dir").map(PathBuf::from).unwrap_or_else(|| inputs[0].parent().unwrap().to_path_buf());
    let collection_dir = args.get("collection-dir").map(PathBuf::from);
    if let Some(dir) = &collection_dir {
        collections::check(dir, &output_dir)?;
    }
    let workers = number(args, "workers")?.or(number(args, "number-of-threads")?).map(|w| w as usize).unwrap_or_else(allocated_cpus);
    if workers < 1 {
        return Err("Query workers must be positive".into());
    }
    let settings = product_settings(args)?;
    println!("{}\nCreating Prod-Merged Files (Rust SmartTile)\n{}", "=".repeat(60), "=".repeat(60));
    let run = |input: PathBuf| products::run(&products::ProductParams {
        input, output_dir: output_dir.clone(), resolutions: settings.resolutions.clone(), formats: settings.formats.clone(),
        workers: workers.min(allocated_cpus()), original_point_index: settings.original_point_index, staged_copc_dir: settings.staged_copc_dir.clone(),
    });
    let made = if inputs.len() > 1 {
        fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;
        println!("Joining attributes of {} collections of the same originals", inputs.len());
        let work = output_dir.join(format!(".joined-originals-{}", std::process::id()));
        let joined = work.join("originals");
        let result = fs::create_dir_all(&work).map_err(|e| e.to_string())
            .and_then(|_| products::join_collections(&inputs, &joined)).and_then(|_| run(joined.clone()));
        let _ = fs::remove_dir_all(&work);
        result?
    } else {
        println!("Original-with-predictions dir: {}\nOutput dir: {}", inputs[0].display(), output_dir.display());
        run(inputs[0].clone())?
    };
    if let Some(dir) = &collection_dir {
        collections::publish(dir, &products::collections(&made)?, &[])?;
    }
    println!("\nCreate merged file complete.");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            println!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}
