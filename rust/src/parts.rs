//! Prediction points kept as parts (merge tiles or prediction files) in scratch files and
//! indexed part by part when a query window reaches them, within a point budget, so the
//! memory of enrichment does not grow with the dataset (Python: SQLite point indexes
//! behind a spatial query cache). Results equal one index over every point: a window
//! loads every part whose bounds lie within the query radius, and ties go by (tile, point
//! index), not by part.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::lasio::Res;
use crate::spatial::PointSet;

/// Points kept indexed across every live collection, shared equally (`SMARTTILE_INDEX_POINTS` overrides).
const DEFAULT_BUDGET: usize = 24_000_000;
/// Live collections (`Parts`), which share the budget.
static LIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Points per stored square of a tile.
const PART_POINTS: usize = 1_000_000;

/// One indexed part: its points (tile and point index set unless distance-only),
/// a label per point (instance, 0 when none) and the raw value row of each point.
pub struct Loaded { pub set: PointSet, pub labels: Vec<u64>, pub values: Vec<u8> }

/// A written square of a tile, registered with `Parts::push` (tiles may be written in parallel).
pub struct Part { tile: u32, path: PathBuf, count: usize, lo: [f64; 2], hi: [f64; 2] }

struct Cache { loaded: Vec<Option<Arc<Loaded>>>, used: Vec<u64>, tick: u64, points: usize }

/// The parts of one collection. `references`: points carry tile, point index and label
/// (nearest-point queries); without, only geometry (distance queries).
pub struct Parts {
    dir: PathBuf,
    radius: f64,
    width: usize,
    references: bool,
    parts: Vec<Part>,
    cache: Mutex<Cache>,
}

impl Parts {
    pub fn new(dir: &Path, radius: f64, width: usize, references: bool) -> Res<Parts> {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        LIVE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(Parts { dir: dir.to_path_buf(), radius, width, references, parts: Vec::new(),
                   cache: Mutex::new(Cache { loaded: Vec::new(), used: Vec::new(), tick: 0, points: 0 }) })
    }

    fn record_len(&self) -> usize { 24 + if self.references { 16 } else { 0 } + self.width }

    /// Appends one tile's points: `index` and `labels` per point when `references`, `values` rows of `width`.
    pub fn add(&mut self, tile: u32, xyz: &[[f64; 3]], index: &[i64], labels: &[u64], values: &[u8]) -> Res<()> {
        let parts = self.write(self.parts.len(), tile, xyz, index, labels, values)?;
        self.push(parts);
        Ok(())
    }

    /// Writes tile number `n` (pushed later, in order) as XY squares of about `PART_POINTS`
    /// points each, so a query window loads only the squares near it.
    pub fn write(&self, n: usize, tile: u32, xyz: &[[f64; 3]], index: &[i64], labels: &[u64], values: &[u8]) -> Res<Vec<Part>> {
        let (lo, hi) = bounds(xyz);
        let area = ((hi[0] - lo[0]) * (hi[1] - lo[1])).max(1e-9);
        let edge = (area * PART_POINTS as f64 / xyz.len().max(1) as f64).sqrt().max(1.0);
        let square = |p: &[f64; 3]| (((p[0] - lo[0]) / edge).floor() as i64, ((p[1] - lo[1]) / edge).floor() as i64);
        struct Open { out: BufWriter<fs::File>, path: PathBuf, count: usize, lo: [f64; 2], hi: [f64; 2] }
        let mut open: std::collections::BTreeMap<(i64, i64), Open> = std::collections::BTreeMap::new();
        let mut record = Vec::with_capacity(self.record_len());
        for (k, p) in xyz.iter().enumerate() {
            let key = square(p);
            if !open.contains_key(&key) {
                let path = self.dir.join(format!("part_{n:05}_{}_{}.bin", key.0, key.1));
                let file = fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                open.insert(key, Open { out: BufWriter::with_capacity(1 << 18, file), path, count: 0, lo: [f64::INFINITY; 2], hi: [f64::NEG_INFINITY; 2] });
            }
            let o = open.get_mut(&key).unwrap();
            for a in 0..2 { o.lo[a] = o.lo[a].min(p[a]); o.hi[a] = o.hi[a].max(p[a]); }
            o.count += 1;
            record.clear();
            for v in p { record.extend_from_slice(&v.to_le_bytes()); }
            if self.references {
                record.extend_from_slice(&index[k].to_le_bytes());
                record.extend_from_slice(&labels[k].to_le_bytes());
            }
            record.extend_from_slice(&values[k * self.width..(k + 1) * self.width]);
            o.out.write_all(&record).map_err(|e| format!("{}: {e}", o.path.display()))?;
        }
        open.into_values().map(|mut o| {
            o.out.flush().map_err(|e| format!("{}: {e}", o.path.display()))?;
            Ok(Part { tile, path: o.path, count: o.count, lo: o.lo, hi: o.hi })
        }).collect()
    }

    pub fn push(&mut self, parts: Vec<Part>) {
        let cache = self.cache.get_mut().unwrap();
        for part in parts {
            self.parts.push(part);
            cache.loaded.push(None);
            cache.used.push(0);
        }
    }

    fn load(&self, part: &Part) -> Res<Loaded> {
        let bytes = fs::read(&part.path).map_err(|e| format!("{}: {e}", part.path.display()))?;
        let f = |r: &[u8], at: usize| f64::from_le_bytes(r[at..at + 8].try_into().unwrap());
        let mut set = PointSet::new(self.radius);
        let mut labels = Vec::new();
        let mut values = Vec::with_capacity(part.count * self.width);
        set.xyz.reserve_exact(part.count);
        for r in bytes.chunks_exact(self.record_len()) {
            set.xyz.push([f(r, 0), f(r, 8), f(r, 16)]);
            let mut at = 24;
            if self.references {
                set.tile.push(part.tile);
                set.index.push(i64::from_le_bytes(r[24..32].try_into().unwrap()));
                labels.push(u64::from_le_bytes(r[32..40].try_into().unwrap()));
                at = 40;
            }
            values.extend_from_slice(&r[at..at + self.width]);
        }
        set.build();
        Ok(Loaded { set, labels, values })
    }

    /// The indexed parts (in part order) a query within `limit` of the box `lo`..`hi` can reach.
    pub fn reach(&self, lo: [f64; 2], hi: [f64; 2], limit: f64) -> Res<Vec<Arc<Loaded>>> {
        // A millimetre beyond the limit: part bounds and queries share local coordinates.
        let margin = limit + 1e-3;
        let wanted: Vec<usize> = self.parts.iter().enumerate()
            .filter(|(_, p)| p.count > 0 && (0..2).all(|a| p.lo[a] <= hi[a] + margin && p.hi[a] >= lo[a] - margin)).map(|(n, _)| n).collect();
        let missing: Vec<usize> = { let c = self.cache.lock().unwrap(); wanted.iter().copied().filter(|&n| c.loaded[n].is_none()).collect() };
        let fresh: Vec<Res<(usize, Loaded)>> = missing.par_iter().map(|&n| Ok((n, self.load(&self.parts[n])?))).collect();
        let mut cache = self.cache.lock().unwrap();
        for item in fresh {
            let (n, loaded) = item?;
            cache.points += self.parts[n].count;
            cache.loaded[n] = Some(Arc::new(loaded));
        }
        cache.tick += 1;
        let tick = cache.tick;
        for &n in &wanted { cache.used[n] = tick; }
        let out = wanted.iter().map(|&n| cache.loaded[n].clone().unwrap()).collect();
        // Least recently used parts leave while over budget (never those of this window).
        let total = std::env::var("SMARTTILE_INDEX_POINTS").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_BUDGET);
        let budget = total / LIVE.load(std::sync::atomic::Ordering::SeqCst).max(1);
        while cache.points > budget {
            let Some(n) = (0..self.parts.len()).filter(|&n| cache.loaded[n].is_some() && cache.used[n] < tick).min_by_key(|&n| cache.used[n]) else { break };
            cache.loaded[n] = None;
            cache.points -= self.parts[n].count;
        }
        Ok(out)
    }
}

impl Parts {
    /// Drops every indexed part (they reload when reached again).
    pub fn release(&self) {
        let mut cache = self.cache.lock().unwrap();
        for slot in cache.loaded.iter_mut() { *slot = None; }
        cache.points = 0;
    }
}

impl Drop for Parts {
    fn drop(&mut self) {
        LIVE.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// The XY box of query points.
pub fn bounds(xyz: &[[f64; 3]]) -> ([f64; 2], [f64; 2]) {
    xyz.iter().fold(([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]), |(mut lo, mut hi), p| {
        for a in 0..2 { lo[a] = lo[a].min(p[a]); hi[a] = hi[a].max(p[a]); }
        (lo, hi)
    })
}
