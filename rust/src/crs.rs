//! CRS records of LAS headers (`crs_records.py`, `tile_crs.py`, laspy `parse_crs`).
//!
//! Outputs carry exactly one standardized CRS record; tiling accepts only
//! metric CRSs shared by every input; COPC outputs are checked semantically
//! against their source.

use std::path::PathBuf;

use crate::lasio::{Res, SourceHeader, Vlr};
use crate::proj::{crs_equivalent, Crs};

const WKT_RECORD_ID: u16 = 2112;
const GEOKEY_RECORD_ID: u16 = 34735;
const PROJECTION_RECORD_IDS: [u16; 6] = [2111, 2112, 2113, 34735, 34736, 34737];
const STANDARD_EPSG_CONFIDENCE: i32 = 70;
const PLY_COMMENT_MAX_BYTES: usize = 8192;
const MAX_STAND_IN_RADIUS: f64 = 6_000_000.0;

fn projection_records(header: &SourceHeader) -> Vec<&Vlr> {
    header.vlrs.iter().chain(&header.evlrs).filter(|v| v.user() == "LASF_Projection").collect()
}

fn wkt_text(data: &[u8]) -> String {
    String::from_utf8_lossy(data).trim_end_matches('\0').to_string()
}

/// GeoTIFF projected (3072) or geographic (2048) EPSG code in 1024..=32766; projected wins.
fn geokey_epsg(data: &[u8]) -> Option<u32> {
    let words: Vec<u16> = data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let count = *words.get(3)? as usize;
    let (mut projected, mut geographic) = (None, None);
    for entry in words.get(4..4 + 4 * count)?.chunks_exact(4) {
        let valid = (1024..=32766).contains(&entry[3]);
        match entry[0] {
            3072 if valid => projected = Some(entry[3] as u32),
            2048 if valid => geographic = Some(entry[3] as u32),
            _ => {}
        }
    }
    projected.or(geographic)
}

/// laspy `header.parse_crs()`; `Err` when a record cannot be parsed (pyproj raises).
pub fn parse_crs(header: &SourceHeader) -> Result<Option<Crs>, String> {
    let (mut wkt, mut geokeys) = (None, None);
    for record in projection_records(header) {
        match record.record_id {
            WKT_RECORD_ID => {
                let text = wkt_text(&record.data);
                if !text.is_empty() {
                    wkt = Some(Crs::new(&text)?);
                }
            }
            GEOKEY_RECORD_ID => {
                if let Some(code) = geokey_epsg(&record.data) {
                    geokeys = Some(Crs::from_epsg(code)?);
                }
            }
            _ => {}
        }
    }
    Ok(wkt.or(geokeys))
}

/// `copc_metadata.parse_crs`: None on any parse failure.
pub fn readable_crs(header: &SourceHeader) -> Option<Crs> {
    parse_crs(header).ok().flatten()
}

/// `standardized_wkt_bytes`: an identified EPSG CRS as its WKT1_GDAL definition, else the text; NUL-terminated.
fn standardized_wkt(text: &str) -> Res<Vec<u8>> {
    let text = text.trim_end_matches('\0').trim();
    let crs = Crs::new(text)?;
    let out = match crs.to_epsg(STANDARD_EPSG_CONFIDENCE) {
        Some(code) => Crs::from_epsg(code)?.to_wkt("WKT1_GDAL")?,
        None => text.to_string(),
    };
    let mut bytes = out.into_bytes();
    bytes.push(0);
    Ok(bytes)
}

/// `standardized_crs_record`: standardized WKT of the effective CRS, None without a readable one.
pub fn standardized_crs_record(header: &SourceHeader) -> Option<Vec<u8>> {
    for record in projection_records(header).into_iter().rev() {
        if record.record_id == WKT_RECORD_ID && !wkt_text(&record.data).trim().is_empty() {
            match standardized_wkt(&wkt_text(&record.data)) {
                Ok(bytes) => return Some(bytes),
                Err(_) => break,
            }
        }
    }
    let crs = readable_crs(header)?;
    standardized_wkt(&crs.to_wkt("WKT1_GDAL").ok()?).ok()
}

/// `standardize_projection_records` for an output header: LAS 1.4 keeps one
/// standardized WKT VLR (and the WKT global-encoding bit); older versions are unchanged.
pub fn standardize(version: (u8, u8), global_encoding: &mut u16, source: &SourceHeader, vlrs: &mut Vec<Vlr>, evlrs: &mut Vec<Vlr>) {
    if version != (1, 4) {
        return;
    }
    let Some(data) = standardized_crs_record(source) else { return };
    vlrs.retain(|v| v.user() != "LASF_Projection");
    vlrs.push(Vlr::new("LASF_Projection", WKT_RECORD_ID, "OGC Transformation Record", data));
    evlrs.retain(|v| v.user() != "LASF_Projection");
    *global_encoding |= 0x10;
}

/// `crs_comment_value`: `EPSG:<code>` or one-line standardized WKT for PLY headers.
pub fn ply_comment(header: &SourceHeader) -> Option<String> {
    let data = standardized_crs_record(header)?;
    let text = wkt_text(&data);
    let value = match Crs::new(&text).ok().and_then(|c| c.to_epsg(STANDARD_EPSG_CONFIDENCE)) {
        Some(code) => format!("EPSG:{code}"),
        None => text.split_whitespace().collect::<Vec<_>>().join(" "),
    };
    if !value.is_ascii() || value.len() + 14 > PLY_COMMENT_MAX_BYTES {
        println!("  Warning: CRS cannot be written as a PLY comment ({} characters)", value.len());
        return None;
    }
    Some(value)
}

/// The CRS that defines X/Y: source of a bound CRS, first part of a compound CRS.
fn horizontal(crs: Crs) -> Crs {
    let crs = crs.unbound();
    if crs.is_compound() {
        if let Some(first) = crs.sub_crs_list().into_iter().next() {
            return first.unbound();
        }
    }
    crs
}

fn label(crs: &Crs) -> String {
    match crs.authority_string() {
        Some(authority) => format!("{authority} ({})", crs.name()),
        None => format!("{:?}", crs.name()),
    }
}

/// `unit_problem`: why `crs` cannot be tiled in metres.
fn unit_problem(crs: &Crs) -> Res<Option<String>> {
    let h = horizontal(Crs::new(&crs.to_wkt("WKT2_2019")?)?);
    if h.is_geocentric() {
        return Ok(Some("is geocentric (Earth-centred X/Y/Z), not a projected map CRS".into()));
    }
    if h.is_geographic() {
        return Ok(Some("is geographic: X/Y are degrees of longitude/latitude, not metres".into()));
    }
    let whole = Crs::new(&crs.to_wkt("WKT2_2019")?)?.unbound();
    let vertical: Vec<_> = if whole.is_compound() {
        whole.sub_crs_list().into_iter().skip(1).flat_map(|s| s.unbound().axis_info()).collect()
    } else {
        h.axis_info().into_iter().skip(2).collect()
    };
    for axis in h.axis_info().into_iter().take(2).chain(vertical) {
        if axis.unit_factor != 0.0 && (axis.unit_factor - 1.0).abs() > 1e-12 {
            return Ok(Some(format!("uses linear unit '{}', not metres", axis.unit_name)));
        }
    }
    Ok(None)
}

fn is_local_stand_in(crs: &Crs, header: &SourceHeader) -> Res<bool> {
    if !horizontal(Crs::new(&crs.to_wkt("WKT2_2019")?)?).is_geocentric() {
        return Ok(false);
    }
    let mut radius: f64 = 0.0;
    for x in [header.mins[0], header.maxs[0]] {
        for y in [header.mins[1], header.maxs[1]] {
            for z in [header.mins[2], header.maxs[2]] {
                radius = radius.max((x * x + y * y + z * z).sqrt());
            }
        }
    }
    Ok(radius < MAX_STAND_IN_RADIUS)
}

const REPROJECT_HINT: &str = "SmartTile tile size, buffer and subsampling resolutions are metres; reproject the point clouds to a projected CRS in metres (e.g. the local UTM zone) and upload them again.";

fn names(files: &[&PathBuf]) -> String {
    files.iter().take(5).map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect::<Vec<_>>().join(", ")
}

/// `require_metric_tiling_crs`: the common CRS of the inputs (None if none declares one).
pub fn require_metric_tiling_crs(files: &[PathBuf], headers: &[SourceHeader]) -> Res<Option<Crs>> {
    let (mut declared, mut undeclared, mut unreadable, mut stand_ins) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (path, header) in files.iter().zip(headers) {
        let crs = readable_crs(header);
        if crs.is_none() && !projection_records(header).is_empty() {
            unreadable.push(path);
        }
        match crs {
            Some(crs) => {
                if is_local_stand_in(&crs, header)? {
                    stand_ins.push(path);
                }
                declared.push((path, crs));
            }
            None => undeclared.push(path),
        }
    }
    if !unreadable.is_empty() {
        return Err(format!("Unreadable CRS records in {}: their units cannot be checked. {REPROJECT_HINT}", names(&unreadable)));
    }
    if !declared.is_empty() && !undeclared.is_empty() {
        let more = if undeclared.len() > 5 { format!(" (and {} more files)", undeclared.len() - 5) } else { String::new() };
        return Err(format!("{} input file(s) have no CRS ({}{more}) while {} declares {}. All point clouds of one dataset must share one projected CRS in metres; assign or reproject and upload again.",
                           undeclared.len(), names(&undeclared), declared[0].0.file_name().unwrap().to_string_lossy(), label(&declared[0].1)));
    }
    if !stand_ins.is_empty() {
        println!("Warning: {}: geocentric CRS with points far inside the Earth; tiling as local metres (3Dtrees standardization stand-in).", names(&stand_ins));
    }
    let mut problems = Vec::new();
    for (path, crs) in &declared {
        if stand_ins.contains(path) {
            continue;
        }
        if let Some(why) = unit_problem(crs)? {
            problems.push(format!("{}: {} {why}", path.file_name().unwrap().to_string_lossy(), label(crs)));
        }
    }
    if !problems.is_empty() {
        let more = if problems.len() > 5 { format!(" (and {} more files)", problems.len() - 5) } else { String::new() };
        return Err(format!("Unsupported input CRS for tiling: {}{more}. {REPROJECT_HINT}", problems[..problems.len().min(5)].join("; ")));
    }
    let mut distinct: Vec<(&PathBuf, Crs)> = Vec::new();
    for (path, crs) in declared {
        if !distinct.iter().any(|(_, seen)| crs_equivalent(&crs, seen)) {
            distinct.push((path, crs));
        }
    }
    if distinct.len() > 1 {
        let shown: Vec<String> = distinct.iter().take(5).map(|(p, c)| format!("{}, e.g. {}", label(c), p.file_name().unwrap().to_string_lossy())).collect();
        return Err(format!("Input files use {} different CRSs: {}. All point clouds of one dataset must share one projected CRS in metres; reproject them to a common CRS and upload them again.",
                           distinct.len(), shown.join("; ")));
    }
    Ok(distinct.into_iter().next().map(|(_, c)| c))
}

/// `validate_single_crs_record`: the output carries exactly one CRS record, equal to the source CRS.
pub fn validate_single_crs_record(source: &SourceHeader, output: &SourceHeader) -> (bool, String) {
    let source_crs = readable_crs(source);
    let source_content: Vec<&Vlr> = projection_records(source).into_iter()
        .filter(|v| PROJECTION_RECORD_IDS.contains(&v.record_id) && !wkt_text(&v.data).trim().is_empty()).collect();
    if source_crs.is_none() && source_content.is_empty() {
        return (true, "source has no CRS metadata".into());
    }
    let records = projection_records(output);
    if records.is_empty() {
        return (false, "source CRS missing in output".into());
    }
    let Some(source_crs) = source_crs else {
        let kept = source_content.iter().all(|s| records.iter().any(|r| r.data == s.data));
        return if kept { (true, "unreadable source CRS records kept unchanged".into()) }
               else { (false, "unreadable source CRS records missing or changed in output".into()) };
    };
    let Some(output_crs) = readable_crs(output) else { return (false, "source CRS missing in output".into()) };
    if output.version == (1, 4) && records.len() != 1 {
        return (false, format!("output carries {} CRS records; exactly one standardized WKT record is required", records.len()));
    }
    if !crs_equivalent(&source_crs, &output_crs) {
        // `same_horizontal_crs_unknown_vertical` is not ported: such outputs fail here.
        return (false, format!("output CRS {:?} differs from source CRS {:?}", output_crs.name(), source_crs.name()));
    }
    (true, "single CRS record, equal to the source CRS".into())
}
