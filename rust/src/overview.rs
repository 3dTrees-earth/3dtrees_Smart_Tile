//! `overview_copc_tiles.png` (`plot_tiles_and_copc.plot_extents`): source file
//! extents and tile extents on one map, drawn as SVG and rasterized with resvg.
//!
//! Same content, colours and layout as the matplotlib figure; pixels differ.

use std::path::Path;

use resvg::{tiny_skia, usvg};

use crate::lasio::Res;

static REGULAR: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");
static BOLD: &[u8] = include_bytes!("../assets/DejaVuSans-Bold.ttf");

pub struct Rect { pub xmin: f64, pub ymin: f64, pub xmax: f64, pub ymax: f64, pub label: String }

fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Round tick steps (1, 2, 2.5, 5 x 10^k) giving at most `max` intervals.
fn ticks(lo: f64, hi: f64, max: usize) -> Vec<f64> {
    let span = (hi - lo).max(f64::MIN_POSITIVE);
    let raw = span / max as f64;
    let magnitude = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|m| m * magnitude).find(|s| *s >= raw).unwrap_or(10.0 * magnitude);
    let mut out = Vec::new();
    let mut k = (lo / step).ceil();
    while k * step <= hi + step * 1e-9 {
        out.push(k * step);
        k += 1.0;
    }
    out
}

fn tick_label(value: f64, step: f64) -> String {
    let decimals = if step >= 1.0 { 0 } else { (-step.log10().floor()) as usize + 1 };
    format!("{value:.decimals$}")
}

pub fn plot(sources: &[Rect], tiles: &[Rect], crs_label: &str, output: &Path) -> Res<()> {
    let all = sources.iter().chain(tiles);
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for r in all {
        x0 = x0.min(r.xmin); y0 = y0.min(r.ymin); x1 = x1.max(r.xmax); y1 = y1.max(r.ymax);
    }
    let (px, py) = ((x1 - x0) * 0.05, (y1 - y0) * 0.05);
    let (x0, x1, y0, y1) = (x0 - px, x1 + px, y0 - py, y1 + py);
    // Axes box in points, equal aspect, within matplotlib's 16 x 12 inch figure.
    let (max_w, max_h) = (1000.0, 760.0);
    let scale = (max_w / (x1 - x0)).min(max_h / (y1 - y0));
    let (w, h) = ((x1 - x0) * scale, (y1 - y0) * scale);
    let (left, top) = (110.0, 70.0);
    let (width, height) = (left + w + 30.0, top + h + 75.0);
    let sx = |x: f64| left + (x - x0) * scale;
    let sy = |y: f64| top + (y1 - y) * scale;
    let mut svg = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="DejaVu Sans"><rect width="100%" height="100%" fill="white"/>"#);
    svg += &format!(r#"<text x="{}" y="28" font-size="14" font-weight="bold" text-anchor="middle">Source Files and Generated Tiles</text><text x="{}" y="47" font-size="14" font-weight="bold" text-anchor="middle">(Blue = source LAZ/LAS, Light red = Tiles)</text>"#, left + w / 2.0, left + w / 2.0);
    svg += &format!(r#"<svg x="{left}" y="{top}" width="{w}" height="{h}" overflow="hidden"><g transform="translate({},{})">"#, -left, -top);
    let (xt, yt) = (ticks(x0, x1, 8), ticks(y0, y1, 8));
    for &x in &xt {
        svg += &format!(r##"<line x1="{0}" x2="{0}" y1="{top}" y2="{1}" stroke="#b0b0b0" stroke-opacity="0.3" stroke-width="0.8"/>"##, sx(x), top + h);
    }
    for &y in &yt {
        svg += &format!(r##"<line y1="{0}" y2="{0}" x1="{left}" x2="{1}" stroke="#b0b0b0" stroke-opacity="0.3" stroke-width="0.8"/>"##, sy(y), left + w);
    }
    for r in sources {
        svg += &format!(r#"<rect x="{}" y="{}" width="{}" height="{}" fill="lightblue" stroke="blue" fill-opacity="0.5" stroke-opacity="0.5" stroke-width="1.5"/>"#,
                        sx(r.xmin), sy(r.ymax), (r.xmax - r.xmin) * scale, (r.ymax - r.ymin) * scale);
    }
    for r in tiles {
        svg += &format!(r#"<rect x="{}" y="{}" width="{}" height="{}" fill="mistyrose" stroke="indianred" fill-opacity="0.4" stroke-opacity="0.4" stroke-width="2"/>"#,
                        sx(r.xmin), sy(r.ymax), (r.xmax - r.xmin) * scale, (r.ymax - r.ymin) * scale);
    }
    let label = |r: &Rect, size: f64, colour: &str, alpha: f64, pad: f64| {
        let (cx, cy) = (sx((r.xmin + r.xmax) / 2.0), sy((r.ymin + r.ymax) / 2.0));
        let text_w = r.label.chars().count() as f64 * size * 0.62 + 2.0 * pad * size;
        let text_h = size * 1.2 + 2.0 * pad * size;
        format!(r#"<rect x="{}" y="{}" width="{text_w}" height="{text_h}" rx="{}" fill="white" fill-opacity="{alpha}"/><text x="{cx}" y="{}" font-size="{size}" font-weight="bold" fill="{colour}" text-anchor="middle">{}</text>"#,
                cx - text_w / 2.0, cy - text_h / 2.0, pad * size, cy + size * 0.36, escape(&r.label))
    };
    for r in sources {
        svg += &label(r, 8.0, "darkblue", 0.7, 0.3);
    }
    for r in tiles {
        svg += &label(r, 10.0, "darkred", 0.8, 0.5);
    }
    svg += "</g></svg>";
    svg += &format!(r#"<rect x="{left}" y="{top}" width="{w}" height="{h}" fill="none" stroke="black" stroke-width="0.8"/>"#);
    for &x in &xt {
        let step = xt.get(1).map(|b| b - xt[0]).unwrap_or(1.0);
        svg += &format!(r#"<line x1="{0}" x2="{0}" y1="{1}" y2="{2}" stroke="black" stroke-width="0.8"/><text x="{0}" y="{3}" font-size="10" text-anchor="middle">{4}</text>"#,
                        sx(x), top + h, top + h + 3.5, top + h + 16.0, tick_label(x, step));
    }
    for &y in &yt {
        let step = yt.get(1).map(|b| b - yt[0]).unwrap_or(1.0);
        svg += &format!(r#"<line y1="{0}" y2="{0}" x1="{1}" x2="{2}" stroke="black" stroke-width="0.8"/><text x="{3}" y="{4}" font-size="10" text-anchor="end">{5}</text>"#,
                        sy(y), left - 3.5, left, left - 6.0, sy(y) + 3.5, tick_label(y, step));
    }
    let axis = escape(crs_label);
    svg += &format!(r#"<text x="{}" y="{}" font-size="12" text-anchor="middle">X (CRS: {axis})</text>"#, left + w / 2.0, top + h + 38.0);
    svg += &format!(r#"<text transform="translate(22,{}) rotate(-90)" font-size="12" text-anchor="middle">Y (CRS: {axis})</text>"#, top + h / 2.0);
    // Legend (upper right) and statistics (upper left), as in the matplotlib figure.
    let (lx, ly) = (left + w - 160.0, top + 8.0);
    svg += &format!(r##"<rect x="{lx}" y="{ly}" width="152" height="44" rx="3" fill="white" fill-opacity="0.8" stroke="#cccccc"/>
<rect x="{}" y="{}" width="20" height="10" fill="lightblue" fill-opacity="0.5"/><text x="{}" y="{}" font-size="10">Source file extent</text>
<rect x="{}" y="{}" width="20" height="10" fill="mistyrose" stroke="indianred" stroke-width="2" fill-opacity="0.4" stroke-opacity="0.4"/><text x="{}" y="{}" font-size="10">Tile extent</text>"##,
                    lx + 8.0, ly + 7.0, lx + 34.0, ly + 16.0, lx + 8.0, ly + 25.0, lx + 34.0, ly + 34.0);
    svg += &format!(r#"<rect x="{}" y="{}" width="92" height="34" rx="4" fill="wheat" fill-opacity="0.5"/><text x="{}" y="{}" font-size="10">Source files: {}</text><text x="{}" y="{}" font-size="10">Tiles: {}</text>"#,
                    left + 8.0, top + 8.0, left + 14.0, top + 21.0, sources.len(), left + 14.0, top + 35.0, tiles.len());
    svg += "</svg>";

    let mut options = usvg::Options::default();
    options.fontdb_mut().load_font_data(REGULAR.to_vec());
    options.fontdb_mut().load_font_data(BOLD.to_vec());
    let tree = usvg::Tree::from_str(&svg, &options).map_err(|e| format!("overview SVG: {e}"))?;
    let zoom = 300.0 / 72.0; // matplotlib saves at 300 dpi
    let mut pixmap = tiny_skia::Pixmap::new((width * zoom).ceil() as u32, (height * zoom).ceil() as u32).ok_or("overview size")?;
    resvg::render(&tree, tiny_skia::Transform::from_scale(zoom as f32, zoom as f32), &mut pixmap.as_mut());
    pixmap.save_png(output).map_err(|e| format!("{}: {e}", output.display()))
}
