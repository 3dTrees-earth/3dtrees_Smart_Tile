//! The `tile` task (`run.py --task tile`, `main_tile.py`): index the sources,
//! plan the buffered tile grid, write every tile output from one read, and
//! publish the task directory only after everything succeeded.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Map, Value};

use crate::decimal::{py_round, Ratio};
use crate::lasio::{self, Res, SourceHeader};
use crate::proj::Transformer;
use crate::pyfmt::{dumps, float, float_repr};
use crate::scan::{self, Format, Output, Tile};

pub struct TileParams {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub tile_length: f64,
    pub tile_buffer: f64,
    pub tiling_threshold: Option<f64>,
    pub workers: usize,
    pub outputs: Vec<(f64, Format)>,
    pub nearest: bool,
    pub grid_origin: Option<(f64, f64)>,
    pub collection_dir: Option<PathBuf>,
    pub instance: Option<InstanceParams>,
}

/// `--instance-tiles` and its options.
pub struct InstanceParams {
    pub dimensions: Vec<String>,
    pub attributes: Vec<String>,
    /// `--instance-outputs`, or None for `--instance-resolutions` with LAZ outputs.
    pub outputs: Option<Vec<(f64, Format)>>,
    pub resolutions: Vec<f64>,
    pub wide: f64,
    pub check_ids: bool,
}

impl InstanceParams {
    /// Resolutions, formats and per-dimension output directories, as `_write_tile_task` lays them out.
    fn selection(&self, work: &Path) -> (Vec<f64>, Vec<crate::instance::InstanceFormat>, BTreeMap<String, Vec<PathBuf>>) {
        use crate::instance::InstanceFormat;
        let convert = |f: Format| match f { Format::Laz => InstanceFormat::Laz, Format::Copc => InstanceFormat::Copc, Format::Ply => InstanceFormat::Ply };
        match &self.outputs {
            Some(outputs) => {
                let directories = self.dimensions.iter().enumerate().map(|(index, d)| {
                    let base = if index == 0 { work.to_path_buf() } else { work.join(format!("instance_dim{}", index + 1)) };
                    (d.clone(), (0..outputs.len()).map(|slot| base.join(format!("subsampled_res{}", slot + 1))).collect())
                }).collect();
                (outputs.iter().map(|o| o.0).collect(), outputs.iter().map(|o| convert(o.1)).collect(), directories)
            }
            None => {
                let quote = |d: &str| d.bytes().map(|b| if b.is_ascii_alphanumeric() || b"_.-~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect::<String>();
                let directories = self.dimensions.iter().map(|d| (d.clone(), self.resolutions.iter()
                    .map(|r| work.join(format!("instance_tiles_{}_{}", quote(d), resolution_label(*r)))).collect())).collect();
                (self.resolutions.clone(), vec![InstanceFormat::Laz; self.resolutions.len()], directories)
            }
        }
    }
}

fn file_stem(path: &Path) -> String {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    if name.to_lowercase().ends_with(".copc.laz") {
        return name[..name.len() - ".copc.laz".len()].to_string();
    }
    Path::new(&name).file_stem().unwrap().to_string_lossy().into_owned()
}

/// `point_cloud_files`: LAS/LAZ sorted by lower-case name, a COPC preferred over its LAS/LAZ twin.
pub fn point_cloud_files(directory: &Path) -> Res<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(directory).map_err(|e| format!("{}: {e}", directory.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && { let n = p.file_name().unwrap().to_string_lossy().to_lowercase(); n.ends_with(".laz") || n.ends_with(".las") })
        .collect();
    files.sort_by_key(|p| p.file_name().unwrap().to_string_lossy().to_lowercase());
    let mut by_source: BTreeMap<String, PathBuf> = BTreeMap::new();
    for path in files {
        let key = file_stem(&path).to_lowercase();
        let is_copc = path.file_name().unwrap().to_string_lossy().to_lowercase().ends_with(".copc.laz");
        if let Some(existing) = by_source.get(&key) {
            if existing.file_name().unwrap().to_string_lossy().to_lowercase().ends_with(".copc.laz") == is_copc {
                return Err(format!("{} and {} name the same source; rename one of them",
                                   existing.file_name().unwrap().to_string_lossy(), path.file_name().unwrap().to_string_lossy()));
            }
        }
        if !by_source.contains_key(&key) || is_copc {
            by_source.insert(key, path);
        }
    }
    Ok(by_source.into_values().collect())
}

/// `build_tiles`: cores of `length` from the south-west corner, buffered by `buffer`.
fn build_tiles(minx: f64, miny: f64, maxx: f64, maxy: f64, length: f64, buffer: f64,
               grid_origin: Option<(f64, f64)>) -> Res<(Vec<Value>, [f64; 4])> {
    if ![minx, miny, maxx, maxy].iter().all(|v| v.is_finite()) {
        return Err("Invalid bounds detected (infinity or NaN)".into());
    }
    if length <= 0.0 || !length.is_finite() || buffer < 0.0 || !buffer.is_finite() {
        return Err("Tile length must be positive and buffer nonnegative".into());
    }
    let (start_x, start_y) = match grid_origin {
        Some((x, y)) if x > minx || y > miny => return Err("Grid origin must not exclude source minimum bounds".into()),
        Some(origin) => origin,
        None => (minx, miny),
    };
    let nx = (((maxx - start_x) / length).ceil() as i64).max(1);
    let ny = (((maxy - start_y) / length).ceil() as i64).max(1);
    if nx * ny > 1_000_000 {
        return Err(format!("Would create {} tiles ({nx} x {ny}), which exceeds the maximum of 1,000,000.", nx * ny));
    }
    let mut tiles = Vec::new();
    for col in 0..nx {
        let x = start_x + col as f64 * length;
        for row in 0..ny {
            let y = start_y + row as f64 * length;
            let core = json!([[float(x), float(x + length)], [float(y), float(y + length)]]);
            let buffered = json!([[float(x - buffer), float(x + length + buffer)], [float(y - buffer), float(y + length + buffer)]]);
            tiles.push(json!({"col": col, "row": row, "core": core, "planned_bounds": buffered.clone(), "bounds": buffered}));
        }
    }
    let grid = [start_x - buffer, start_x + nx as f64 * length + buffer, start_y - buffer, start_y + ny as f64 * length + buffer];
    Ok((tiles, grid))
}

fn pair(v: &Value) -> [f64; 2] {
    [v[0].as_f64().unwrap(), v[1].as_f64().unwrap()]
}

fn label(tile: &Value) -> String {
    format!("c{:02}_r{:02}", tile["col"].as_i64().unwrap(), tile["row"].as_i64().unwrap())
}

/// `write_job_list`: label|projected bounds|geographic bounds per tile.
fn write_job_list(layout: &Value, jobs_file: &Path) -> Res<()> {
    let srs = layout.get("proj_srs").or(layout.get("tindex_srs")).and_then(Value::as_str).unwrap_or("missing");
    let transformer = if srs == "missing" { None } else { Transformer::new(srs, "EPSG:4326").ok() };
    let mut lines = Vec::new();
    for tile in layout["tiles"].as_array().unwrap() {
        let (bx, by) = (pair(&tile["bounds"][0]), pair(&tile["bounds"][1]));
        let (gx, gy) = match &transformer {
            None => (bx, by),
            Some(t) => {
                let corners = [t.transform(bx[0], by[0]), t.transform(bx[0], by[1]), t.transform(bx[1], by[0]), t.transform(bx[1], by[1])];
                let lons = corners.map(|c| c.0);
                let lats = corners.map(|c| c.1);
                ([lons.iter().copied().fold(f64::INFINITY, f64::min), lons.iter().copied().fold(f64::NEG_INFINITY, f64::max)],
                 [lats.iter().copied().fold(f64::INFINITY, f64::min), lats.iter().copied().fold(f64::NEG_INFINITY, f64::max)])
            }
        };
        let r = float_repr;
        lines.push(format!("{}|([{},{}],[{},{}])|([{},{}],[{},{}])", label(tile), r(bx[0]), r(bx[1]), r(by[0]), r(by[1]),
                           r(gx[0]), r(gx[1]), r(gy[0]), r(gy[1])));
    }
    fs::write(jobs_file, lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    eprintln!("[prepare_tile_jobs] wrote {} jobs to {}", lines.len(), jobs_file.display());
    Ok(())
}

/// `build_tindex`: the source index GeoPackage, written by PDAL as in Python SmartTile.
fn build_tindex(sources: &[PathBuf], headers: &[SourceHeader], output: &Path) -> Res<Option<String>> {
    println!("\n{}\nStep 1: Building spatial index (tindex)\n{}", "=".repeat(60), "=".repeat(60));
    let list = output.with_extension("filelist.txt");
    let text: String = sources.iter().map(|s| format!("{}\n", fs::canonicalize(s).unwrap_or(s.clone()).display())).collect();
    fs::write(&list, text).map_err(|e| e.to_string())?;
    let tmp = output.with_extension("tmp.gpkg");
    let mut command = Command::new("pdal");
    command.args(["tindex", "create"]).arg(&tmp).arg("--filelist").arg(&list)
        .args(["--tindex_name=Location", "--ogrdriver=GPKG", "--fast_boundary", "--write_absolute_path"]);
    // As Python: PDAL's compound WKT of the first source as t_srs, the header CRS as a_srs.
    let info = Command::new("pdal").args(["info", "--metadata"]).arg(&sources[0]).output().ok()
        .filter(|o| o.status.success())
        .and_then(|o| serde_json::from_slice::<serde_json::Value>(&o.stdout).ok());
    let t_srs = info.as_ref().and_then(|m| {
        let meta = &m["metadata"];
        [&meta["srs"]["compoundwkt"], &meta["spatialreference"]].into_iter()
            .filter_map(|v| v.as_str()).find(|s| !s.is_empty()).map(str::to_string)
    });
    // `_common_header_srs`: only when every source header declares an equivalent CRS.
    let mut common: Option<crate::proj::Crs> = None;
    let mut all = true;
    for h in headers {
        match crate::crs::readable_crs(h) {
            Some(crs) if common.as_ref().is_none_or(|c| crate::proj::crs_equivalent(c, &crs)) => { if common.is_none() { common = Some(crs); } }
            _ => { all = false; break; }
        }
    }
    let a_srs = if all { common.map(|c| c.to_wkt("WKT2_2019")).transpose()? } else { None };
    let t_srs = t_srs.or(a_srs.clone());
    if let Some(t_srs) = &t_srs {
        command.arg(format!("--t_srs={t_srs}"));
    }
    if let Some(a_srs) = &a_srs {
        command.arg(format!("--a_srs={a_srs}"));
    }
    println!("  Found {} source files\n  Output: {}", sources.len(), output.display());
    let result = command.output().map_err(|e| format!("pdal tindex: {e}"))?;
    let _ = fs::remove_file(&list);
    if !result.status.success() {
        return Err(format!("pdal tindex failed: {}", String::from_utf8_lossy(&result.stderr)));
    }
    fs::rename(&tmp, output).map_err(|e| e.to_string())?;
    println!("  ✓ Tindex created: {}", output.display());
    Ok(t_srs)
}

/// `tile_offsets`: the tile centre in whole metres and the source Z offset.
fn tile_offsets(bounds: [f64; 4], z_offset: f64) -> [f64; 3] {
    [py_round((bounds[0] + bounds[2]) / 2.0), py_round((bounds[1] + bounds[3]) / 2.0), z_offset]
}

/// `original_coordinate_tiles._on_grid`: the lattice point origin + k * scale nearest `value`.
fn on_grid(origin: f64, value: f64, scale: f64) -> Res<f64> {
    let steps = py_round((value - origin) / scale) as i128;
    let exact = Ratio::from_f64(origin)?.checked_add(Ratio::int(steps).checked_mul(Ratio::from_f64(scale)?).ok_or("overflow")?)
        .ok_or("overflow")?;
    Ok(exact.to_f64())
}

pub(crate) fn original_coordinate_tiles(headers: &[SourceHeader], tiles: Vec<Tile>) -> Res<Vec<Tile>> {
    let origin = headers[0].offsets;
    let scales = tiles[0].scales;
    if tiles.iter().any(|t| t.scales != scales) {
        return Err("Original-resolution tiles require one coordinate scale".into());
    }
    for h in headers {
        for axis in 0..3 {
            let ratio = h.scales[axis] / scales[axis];
            let phase = (h.offsets[axis] - origin[axis]) / scales[axis];
            if (ratio - ratio.round_ties_even()).abs() > 1e-9 || (phase - phase.round_ties_even()).abs() > 1e-6 {
                return Err("coordinate grid cannot be preserved at original resolution".into());
            }
        }
    }
    tiles.into_iter().map(|mut t| {
        for axis in 0..3 {
            t.offsets[axis] = on_grid(origin[axis], t.offsets[axis], scales[axis])?;
        }
        Ok(t)
    }).collect()
}

fn resolution_label(resolution: f64) -> String {
    if resolution == 0.0 {
        return "original".into();
    }
    let cm = resolution * 100.0;
    if (cm - py_round(cm)).abs() < 1e-6 && py_round(cm) >= 1.0 {
        return format!("{}cm", py_round(cm) as i64);
    }
    format!("{}mm", py_round(resolution * 1000.0) as i64)
}

/// `subsample_output_name` (output_copc=False): the tile id or cleaned stem, then the resolution label.
fn subsample_output_name(stem: &str, resolution: f64, prefix: &str) -> String {
    let label = resolution_label(resolution);
    let re = |p: &str| regex::Regex::new(p).unwrap();
    let mut base = re(r"_subsampled[\d.]+m$").replace(stem, "").into_owned();
    base = re(r"_subsampled_(?:original|\d+(?:\.\d+)?(?:cm|mm))$").replace(&base, "").into_owned();
    base = re(r"_\d+cm$").replace(&base, "").into_owned();
    if !prefix.is_empty() {
        if let Some(rest) = base.strip_prefix(&format!("{prefix}_")) {
            base = rest.to_string();
        }
    }
    let strip = re(r"^[^_]+_\d+m_");
    base = strip.replace(&base, "").into_owned();
    if let Some(m) = re(r"(c\d+_r\d+)").find(&base) {
        let mut before = base[..m.start()].to_string();
        if before.ends_with('_') {
            before.pop();
        }
        let before = strip.replace(&before, "").into_owned();
        return if before.is_empty() { format!("{}_subsampled_{label}.laz", m.as_str()) }
               else { format!("{before}_{}_subsampled_{label}.laz", m.as_str()) };
    }
    format!("{}_subsampled_{label}.laz", strip.replace(&base, ""))
}

fn output_name(stem: &str, resolution: f64, prefix: &str, format: Format) -> String {
    let name = subsample_output_name(stem, resolution, prefix);
    format!("{}.{}", &name[..name.len() - ".laz".len()], extension(format))
}

/// `single_cloud_layout`: the actual cloud as the one tile, keeping CRS/context fields.
fn single_cloud_layout(layout: &mut Value, header: &SourceHeader) {
    let (x0, x1, y0, y1) = (header.mins[0], header.maxs[0], header.mins[1], header.maxs[1]);
    let xy = json!([[float(x0), float(x1)], [float(y0), float(y1)]]);
    let map = layout.as_object_mut().unwrap();
    map.insert("tiling_skipped".into(), json!(true));
    map.insert("tile_buffer".into(), float(0.0));
    map.insert("proj_extent".into(), json!({"minx": float(x0), "miny": float(y0), "maxx": float(x1), "maxy": float(y1)}));
    map.insert("grid_bounds".into(), json!({"xmin": float(x0), "xmax": float(x1), "ymin": float(y0), "ymax": float(y1)}));
    map.insert("tiles".into(), json!([{"col": 0, "row": 0, "bounds": xy.clone(), "core": xy.clone(), "planned_bounds": xy}]));
}

/// `plot_overview`: source extents (header bounds, as the index stores them) and the layout's tiles.
fn plot_overview(sources: &[PathBuf], headers: &[SourceHeader], layout: &Value, output: &Path) -> Res<()> {
    println!("Loading tile extents...");
    let tiles: Vec<crate::overview::Rect> = layout["tiles"].as_array().unwrap().iter().map(|t| {
        let (bx, by) = (pair(&t["bounds"][0]), pair(&t["bounds"][1]));
        crate::overview::Rect { xmin: bx[0], ymin: by[0], xmax: bx[1], ymax: by[1], label: label(t) }
    }).collect();
    let srs = layout["proj_srs"].as_str().unwrap_or("EPSG:32632");
    println!("Found {} tiles\nTiles are in CRS: {srs}\nLoading source file extents from tindex...", tiles.len());
    let rects: Vec<crate::overview::Rect> = sources.iter().zip(headers).map(|(s, h)| crate::overview::Rect {
        xmin: h.mins[0], ymin: h.mins[1], xmax: h.maxs[0], ymax: h.maxs[1], label: s.file_name().unwrap().to_string_lossy().into_owned() }).collect();
    println!("Found {} source files", rects.len());
    crate::overview::plot(&rects, &tiles, srs, output)?;
    println!("\nVisualization saved to: {}\n  - {} source files (blue)\n  - {} tiles (light red)", output.display(), rects.len(), tiles.len());
    Ok(())
}

fn extension(format: Format) -> &'static str {
    match format { Format::Laz => "laz", Format::Copc => "copc.laz", Format::Ply => "ply" }
}

/// `convert_laz_output_to_copc`: Untwine (`--a_srs` from the input's CRS), then
/// the semantic CRS check; PDAL's COPC writer when either fails.
pub(crate) fn copc_conversion(input: &Path, output: &Path, expected: u64, preserve_extra_dims: bool) -> Res<()> {
    let source = lasio::read_header(input)?;
    let mut command = Command::new("untwine");
    command.arg("-i").arg(input).arg("-o").arg(output);
    if let Some(srs) = crate::crs::readable_crs(&source).and_then(|c| c.authority_string()) {
        command.arg("--a_srs").arg(srs);
    }
    let untwined = command.output().map(|r| r.status.success()).unwrap_or(false) && output.exists();
    let mut valid = false;
    if untwined {
        let (ok, message) = crate::crs::validate_single_crs_record(&source, &lasio::read_header(output)?);
        if !ok {
            println!("  Warning: untwine COPC CRS validation failed for {}: {message}; retrying with PDAL",
                     output.file_name().unwrap().to_string_lossy());
        }
        valid = ok;
    }
    if !valid {
        let _ = fs::remove_file(output);
        let mut writer = json!({"type": "writers.copc", "filename": output.display().to_string(), "forward": "all"});
        if preserve_extra_dims {
            writer["extra_dims"] = json!("all");
        }
        let pipeline = json!({"pipeline": [{"type": "readers.las", "filename": input.display().to_string()}, writer]});
        let mut child = Command::new("pdal").args(["pipeline", "--stdin"]).stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::piped()).spawn().map_err(|e| format!("pdal: {e}"))?;
        use std::io::Write as _;
        child.stdin.take().unwrap().write_all(pipeline.to_string().as_bytes()).map_err(|e| e.to_string())?;
        let result = child.wait_with_output().map_err(|e| e.to_string())?;
        if !result.status.success() || !output.exists() {
            return Err(format!("COPC conversion failed for {}: {}", output.display(), String::from_utf8_lossy(&result.stderr)));
        }
        let (ok, message) = crate::crs::validate_single_crs_record(&source, &lasio::read_header(output)?);
        if !ok {
            return Err(format!("COPC CRS validation failed for {}: {message}", output.display()));
        }
    }
    let header = lasio::read_header(output)?;
    if header.point_count != expected {
        return Err(format!("{}: COPC holds {} of {expected} points", output.display(), header.point_count));
    }
    Ok(())
}

/// Run the tile task inside `work` (the unpublished task directory).
fn write_tile_task(params: &TileParams, work: &Path, published: &Path) -> Res<Vec<PathBuf>> {
    println!("{}\nRunning Tile Task (Rust SmartTile)\n{}", "=".repeat(60), "=".repeat(60));
    let sources = point_cloud_files(&params.input_dir)?;
    if sources.is_empty() {
        return Err(format!("No LAZ/LAS files found in {}", params.input_dir.display()));
    }
    let headers = sources.iter().map(|s| lasio::read_header(s)).collect::<Res<Vec<_>>>()?;
    let empty: Vec<String> = sources.iter().zip(&headers).filter(|(_, h)| h.point_count == 0)
        .map(|(s, _)| s.file_name().unwrap().to_string_lossy().into_owned()).collect();
    if !empty.is_empty() {
        return Err(format!("Input file(s) without points: {}; remove them and run again", empty.join(", ")));
    }
    let skip_tiling = match params.tiling_threshold {
        Some(threshold) if sources.len() == 1 => (fs::metadata(&sources[0]).map_err(|e| e.to_string())?.len() as f64 / (1024.0 * 1024.0)) < threshold,
        _ => false,
    };
    if skip_tiling {
        let size = fs::metadata(&sources[0]).map_err(|e| e.to_string())?.len() as f64 / (1024.0 * 1024.0);
        println!("{}\nTiling Threshold Check\n{}", "=".repeat(60), "=".repeat(60));
        println!("  Single file detected: {}", sources[0].file_name().unwrap().to_string_lossy());
        println!("  Original file size: {size:.2} MB\n  Threshold: {} MB\n  Decision: one whole-cloud tile, no buffer\n{}\n",
                 float_repr(params.tiling_threshold.unwrap()), "=".repeat(60));
    }
    crate::crs::require_metric_tiling_crs(&sources, &headers)?;
    let length_label = params.tile_length as i64;
    let tindex = work.join(format!("tindex_{length_label}m.gpkg"));
    let index_srs = build_tindex(&sources, &headers, &tindex)?;

    // Step 2: layout from the index extent (PDAL's fast boundaries are the header bounds).
    println!("\n{}\nStep 2: Calculating tile bounds\n{}", "=".repeat(60), "=".repeat(60));
    let (mut minx, mut miny, mut maxx, mut maxy) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for h in &headers {
        minx = minx.min(h.mins[0]); miny = miny.min(h.mins[1]);
        maxx = maxx.max(h.maxs[0]); maxy = maxy.max(h.maxs[1]);
    }
    // pyproj `CRS.to_string()` of the index CRS; PDAL labels a CRS-less index EPSG:4326.
    let srs = match &index_srs {
        None => "EPSG:4326".to_string(),
        Some(definition) => {
            let crs = crate::proj::Crs::new(definition)?;
            crs.to_authority(None, 100).map(|(a, c)| format!("{a}:{c}")).unwrap_or_else(|| definition.clone())
        }
    };
    let (tiles_json, grid) = build_tiles(minx, miny, maxx, maxy, params.tile_length, params.tile_buffer, params.grid_origin)?;
    let extent = json!({"minx": float(minx), "miny": float(miny), "maxx": float(maxx), "maxy": float(maxy)});
    let mut layout = Map::new();
    layout.insert("tindex".into(), json!(tindex.display().to_string()));
    layout.insert("tindex_srs".into(), json!(srs));
    layout.insert("proj_srs".into(), json!(srs));
    layout.insert("proj_extent".into(), extent.clone());
    layout.insert("geo_extent".into(), extent);
    layout.insert("tile_length".into(), float(params.tile_length));
    layout.insert("tile_buffer".into(), float(params.tile_buffer));
    layout.insert("grid_bounds".into(), json!({"xmin": float(grid[0]), "xmax": float(grid[1]), "ymin": float(grid[2]), "ymax": float(grid[3])}));
    layout.insert("tiles".into(), Value::Array(tiles_json));
    let mut layout = Value::Object(layout);
    let bounds_json = work.join("tile_bounds_tindex.json");
    fs::write(&bounds_json, dumps(&layout)).map_err(|e| e.to_string())?;
    let jobs_file = work.join(format!("tile_jobs_{length_label}m.txt"));
    write_job_list(&layout, &jobs_file)?;
    println!("  ✓ Calculated {} tiles", layout["tiles"].as_array().unwrap().len());
    std::os::unix::fs::symlink(tindex.file_name().unwrap(), work.join("tindex.gpkg")).map_err(|e| e.to_string())?;
    if skip_tiling {
        single_cloud_layout(&mut layout, &headers[0]);
        fs::write(&bounds_json, dumps(&layout) + "\n").map_err(|e| e.to_string())?;
        write_job_list(&layout, &jobs_file)?;
        plot_overview(&sources, &headers, &layout, &work.join("overview_copc_tiles.png"))?;
    }

    // Tiles: planned buffered bounds, finest source scale, tile-centre offsets.
    let scales: [f64; 3] = std::array::from_fn(|a| headers.iter().map(|h| h.scales[a]).fold(f64::INFINITY, f64::min));
    let planned_tiles = |layout: &Value, z_offset: f64, whole_core: Option<[f64; 4]>| -> Vec<Tile> {
        let mut tiles: Vec<Tile> = if skip_tiling {
            vec![Tile { label: file_stem(&sources[0]), bounds: [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY],
                        scales, offsets: headers[0].offsets, core: whole_core }]
        } else {
            layout["tiles"].as_array().unwrap().iter().map(|t| {
                let (bx, by, cx, cy) = (pair(&t["bounds"][0]), pair(&t["bounds"][1]), pair(&t["core"][0]), pair(&t["core"][1]));
                let bounds = [bx[0], by[0], bx[1], by[1]];
                Tile { label: label(t), bounds, scales, offsets: tile_offsets(bounds, z_offset), core: Some([cx[0], cy[0], cx[1], cy[1]]) }
            }).collect()
        };
        tiles.sort_by(|a, b| a.label.cmp(&b.label));
        tiles
    };
    let mut output_dirs = Vec::new();
    if !params.outputs.is_empty() {
        let mut tiles = planned_tiles(&layout, headers[0].offsets[2], None);
        if params.outputs.iter().any(|(r, _)| *r == 0.0) {
            tiles = original_coordinate_tiles(&headers, tiles)?;
        }
        let prefix = format!("{}_{length_label}m", published.file_name().unwrap().to_string_lossy());
        output_dirs = (0..params.outputs.len()).map(|i| work.join(format!("subsampled_res{}", i + 1))).collect();
        let outputs: Vec<Output> = params.outputs.iter().zip(&output_dirs).map(|(&(resolution, format), dir)| Output {
            resolution, format,
            paths: tiles.iter().map(|t| (t.label.clone(), dir.join(output_name(&t.label, resolution, &prefix, format)))).collect(),
        }).collect();
        println!("\n{}\nSingle-scan tile subsampling\n{}", "=".repeat(60), "=".repeat(60));
        let labels: Vec<String> = params.outputs.iter().map(|(r, _)| if *r == 0.0 { "original".into() } else { format!("{} m", float_repr(*r)) }).collect();
        println!("  Tiles: {}; resolutions: {}", tiles.len(), labels.join(", "));
        let scratch = work.join(format!(".tile-centroids-{}", std::process::id()));
        fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
        let converters = (params.workers / 2).clamp(1, 4);
        let summary = scan::write_tile_centroids(&sources, &headers, &tiles, &outputs, params.workers, &scratch, converters,
                                                 params.nearest, true, &|i: &Path, o: &Path, n: u64| copc_conversion(i, o, n, false),
                                                 crate::crs::ply_comment(&headers[0]).as_deref());
        let summary = match summary {
            Ok(s) => s,
            Err(e) => { let _ = fs::remove_dir_all(&scratch); return Err(e); }
        };
        for (staged, destination) in &summary.staged {
            fs::create_dir_all(destination.parent().unwrap()).map_err(|e| e.to_string())?;
            fs::rename(staged, destination).map_err(|e| e.to_string())?;
        }
        fs::remove_dir_all(&scratch).map_err(|e| e.to_string())?;
        if !skip_tiling {
            let by_label: BTreeMap<String, bool> = tiles.iter().map(|t| t.label.clone()).zip(summary.occupied.iter().copied()).collect();
            publish_occupancy(&mut layout, &by_label, &jobs_file)?;
            plot_overview(&sources, &headers, &layout, &work.join("overview_copc_tiles.png"))?;
        }
    }

    if let Some(instance) = &params.instance {
        // `instance_tiles_from_plan`: the Z offset of the last source, as Python reads it.
        let z_offset = headers.last().unwrap().offsets[2];
        let tiles = planned_tiles(&layout, z_offset, Some([f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY]));
        let (resolutions, formats, directories) = instance.selection(work);
        let request = crate::instance::InstanceRequest {
            dimensions: instance.dimensions.clone(), attributes: instance.attributes.clone(), resolutions, formats,
            directories, check_ids: instance.check_ids, wide: instance.wide,
        };
        let scratch = work.join(format!(".instance-tiles-{}", std::process::id()));
        fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
        let result = crate::instance::write_instance_tiles(&sources, &headers, tiles, &request, &scratch, params.workers,
                                                           &|i: &Path, o: &Path, n: u64, keep: bool| copc_conversion(i, o, n, keep));
        let (results, products) = match result {
            Ok(r) => r,
            Err(e) => { let _ = fs::remove_dir_all(&scratch); return Err(e); }
        };
        for (staged, destination) in &products {
            fs::create_dir_all(destination.parent().unwrap()).map_err(|e| e.to_string())?;
            fs::rename(staged, destination).map_err(|e| e.to_string())?;
        }
        fs::remove_dir_all(&scratch).map_err(|e| e.to_string())?;
        if params.outputs.is_empty() && !skip_tiling {
            let mut occupied = std::collections::BTreeSet::new();
            for (_, destination) in &products {
                let name = destination.file_name().unwrap().to_string_lossy().into_owned();
                occupied.insert(name.split('.').next().unwrap().to_string());
            }
            let by_label: BTreeMap<String, bool> = layout["tiles"].as_array().unwrap().iter().map(|t| { let l = label(t); let o = occupied.contains(&l); (l, o) }).collect();
            publish_occupancy(&mut layout, &by_label, &jobs_file)?;
            plot_overview(&sources, &headers, &layout, &work.join("overview_copc_tiles.png"))?;
        }
        for r in &results {
            for (resolution, directory, count, points) in &r.outputs {
                println!("Instance tiles {} at {} m: {count} tiles, {} instances, {} points: {}", r.dimension, float_repr(*resolution),
                         r.instances, scan::group_thousands(*points), directory.display());
            }
        }
    }

    // Step 7: tile bounds from the written LAS/LAZ headers (`update_tile_bounds_json_from_files`).
    let las_dirs: Vec<&PathBuf> = output_dirs.iter().zip(&params.outputs).filter(|(_, (_, f))| *f != Format::Ply).map(|(d, _)| d).collect();
    if skip_tiling {
        if let Some(dir) = las_dirs.first() {
            let files = point_cloud_files(dir)?;
            if files.len() != 1 {
                return Err("Skipped tiling requires exactly one cloud and one layout entry".into());
            }
            single_cloud_layout(&mut layout, &lasio::read_header(&files[0])?);
            println!("  Updated tile_bounds_tindex.json with the bounds of 1 tile(s)");
        }
    } else {
        let mut extents: BTreeMap<String, [f64; 4]> = BTreeMap::new();
        for dir in &las_dirs {
            for path in scan_sorted(dir)? {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                let Some(label) = tile_label(&name) else { continue };
                let h = lasio::read_header(&path)?;
                let b = [h.mins[0], h.maxs[0], h.mins[1], h.maxs[1]];
                extents.entry(label).and_modify(|e| *e = [e[0].min(b[0]), e[1].max(b[1]), e[2].min(b[2]), e[3].max(b[3])]).or_insert(b);
            }
        }
        let mut updated = 0;
        for tile in layout["tiles"].as_array_mut().unwrap() {
            if let Some(b) = extents.get(&label(tile)) {
                tile["bounds"] = json!([[float(b[0]), float(b[1])], [float(b[2]), float(b[3])]]);
                updated += 1;
            }
        }
        if updated > 0 {
            println!("  Updated tile_bounds_tindex.json with the bounds of {updated} tile(s)");
        }
    }
    // The layout points at the published index, not the scratch directory.
    let relative = tindex.strip_prefix(work).unwrap();
    layout["tindex"] = json!(fs::canonicalize(published.parent().unwrap()).unwrap().join(published.file_name().unwrap()).join(relative).display().to_string());
    fs::write(&bounds_json, dumps(&layout) + "\n").map_err(|e| e.to_string())?;
    Ok(output_dirs)
}

/// `publish_occupied_tiles`: keep only the populated cores in the layout and the job list.
fn publish_occupancy(layout: &mut Value, occupied: &BTreeMap<String, bool>, jobs_file: &Path) -> Res<()> {
    let planned: Vec<Value> = layout["tiles"].as_array().unwrap().clone();
    let (kept, skipped): (Vec<Value>, Vec<Value>) = planned.iter().cloned().partition(|t| occupied[&label(t)]);
    if kept.is_empty() {
        return Err("No planned tile core contains source points".into());
    }
    let retained = kept.len();
    layout["tiles"] = Value::Array(kept);
    layout.as_object_mut().unwrap().insert("tile_count".into(), json!(retained));
    let skipped: Vec<Value> = skipped.into_iter().map(|mut t| { t.as_object_mut().unwrap().insert("reason".into(), json!("no_source_points_in_core")); t }).collect();
    layout.as_object_mut().unwrap().insert("core_occupancy".into(), json!({
        "policy": "at least one original source point inside the inclusive XY core",
        "planned_tiles": planned.len(), "retained_tiles": retained, "skipped_tiles": skipped}));
    write_job_list(layout, jobs_file)?;
    println!("  Core occupancy: {retained} tiles retained; {} empty cores skipped", planned.len() - retained);
    Ok(())
}

fn scan_sorted(dir: &Path) -> Res<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = fs::read_dir(dir).map_err(|e| e.to_string())?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "laz")).collect();
    out.sort();
    Ok(out)
}

fn tile_label(name: &str) -> Option<String> {
    let bytes = name.as_bytes();
    for start in 0..bytes.len() {
        if bytes[start] != b'c' { continue; }
        let mut i = start + 1;
        let d0 = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
        if i == d0 || i + 1 >= bytes.len() || bytes[i] != b'_' || bytes[i + 1] != b'r' { continue; }
        let mut j = i + 2;
        let d1 = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() { j += 1; }
        if j > d1 {
            return Some(name[start..j].to_string());
        }
    }
    None
}

/// `run_tile_task`: publish the complete task directory only after success.
pub fn run(params: &TileParams) -> Res<()> {
    let started = std::time::Instant::now();
    let destination = &params.output_dir;
    if !params.input_dir.is_dir() {
        return Err(format!("Input directory does not exist: {}", params.input_dir.display()));
    }
    if destination.exists() && (!destination.is_dir() || fs::read_dir(destination).map_err(|e| e.to_string())?.next().is_some()) {
        return Err(format!("Refusing to mix this run with existing output: {}", destination.display()));
    }
    let parent = destination.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    if let Some(dir) = &params.collection_dir {
        crate::collections::check(dir, destination)?;
    }
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let temporary = parent.join(format!(".smarttile-task-{}", std::process::id()));
    let work = temporary.join(destination.file_name().unwrap());
    fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let result = write_tile_task(params, &work, destination).and_then(|_| {
        fs::rename(&work, destination).map_err(|e| format!("publishing {}: {e}", destination.display()))
    });
    let _ = fs::remove_dir_all(&temporary);
    result?;
    if let Some(dir) = &params.collection_dir {
        crate::collections::publish(dir, &crate::collections::tile_collections(destination),
            &[("tile_bounds_tindex.json", destination.join("tile_bounds_tindex.json")),
              ("overview_copc_tiles.png", destination.join("overview_copc_tiles.png"))])?;
    }
    println!("Tile task complete: {} ({:.1} s)", destination.display(), started.elapsed().as_secs_f64());
    Ok(())
}
