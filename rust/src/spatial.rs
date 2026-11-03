//! Nearest-point queries with the merge's explicit numerical rules
//! (`bounded_point_index.py`): inclusive radius plus an 8-ULP float64 guard,
//! and among candidates within that guard of the minimum distance the lowest
//! (tile, point index) wins. Distances are computed as scipy's cKDTree does:
//! sqrt(((dx² + dy²) + dz²)).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A fast hash for integer cell keys (FxHash).
#[derive(Default)]
pub struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes { self.0 = (self.0.rotate_left(5) ^ *b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95); }
    }
    fn write_i64(&mut self, v: i64) { self.0 = (self.0.rotate_left(5) ^ v as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95); }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

pub const CELL_SIZE: f64 = 2.0;

/// `np.spacing(x)` for x >= 0.
fn spacing(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1) - x
}

/// `distance_roundoff`: 8 ULPs of the group's largest coordinate magnitude (at least 1).
pub fn distance_roundoff(points: &[[f64; 3]]) -> f64 {
    let magnitude = points.iter().flatten().fold(1.0f64, |m, v| m.max(v.abs()));
    8.0 * spacing(magnitude)
}

pub fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (((0.0 + dx * dx) + dy * dy) + dz * dz).sqrt()
}

/// `spatial_batches`: index groups per 2 m XY cell, cells in (x, y) order, points in input order.
pub fn spatial_batches(xyz: &[[f64; 3]]) -> Vec<Vec<usize>> {
    let mut keyed: Vec<((i64, i64), usize)> = xyz.iter().enumerate()
        .map(|(i, p)| (((p[0] / CELL_SIZE).floor() as i64, (p[1] / CELL_SIZE).floor() as i64), i)).collect();
    keyed.sort(); // stable by construction: (cell, index)
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut last = None;
    for (cell, i) in keyed {
        if last != Some(cell) {
            groups.push(Vec::new());
            last = Some(cell);
        }
        groups.last_mut().unwrap().push(i);
    }
    groups
}

/// Points of one or more tiles, bucketed in cubic cells of the query radius.
/// Push every point, then `build` once before querying.
pub struct PointSet {
    pub xyz: Vec<[f64; 3]>,
    pub tile: Vec<u32>,
    pub index: Vec<i64>,
    cell: f64,
    cells: FxMap<(i64, i64, i64), (u32, u32)>, // cell -> (start, len) in `order`
    order: Vec<u32>,
}

impl PointSet {
    /// `radius`: the query radius. Cells are slightly more than twice as large, so a query
    /// within the radius (plus the 8-ULP guard) touches only the 2×2×2 cells around it.
    pub fn new(radius: f64) -> PointSet {
        PointSet { xyz: Vec::new(), tile: Vec::new(), index: Vec::new(), cell: radius * 2.004, cells: FxMap::default(), order: Vec::new() }
    }

    fn key(&self, p: &[f64; 3]) -> (i64, i64, i64) {
        ((p[0] / self.cell).floor() as i64, (p[1] / self.cell).floor() as i64, (p[2] / self.cell).floor() as i64)
    }

    pub fn push(&mut self, p: [f64; 3], tile: u32, index: i64) {
        self.xyz.push(p);
        self.tile.push(tile);
        self.index.push(index);
    }

    /// Bucket the points (ids in insertion order within a cell).
    pub fn build(&mut self) {
        use rayon::prelude::*;
        let keys: Vec<(i64, i64, i64)> = self.xyz.par_iter().map(|p| self.key(p)).collect();
        let mut order: Vec<u32> = (0..self.xyz.len() as u32).collect();
        order.par_sort_by_key(|&i| (keys[i as usize], i));
        let mut cells = FxMap::default();
        let mut start = 0;
        while start < order.len() {
            let key = keys[order[start] as usize];
            let mut end = start + 1;
            while end < order.len() && keys[order[end] as usize] == key { end += 1; }
            cells.insert(key, (start as u32, (end - start) as u32));
            start = end;
        }
        self.cells = cells;
        self.order = order;
    }

    pub fn len(&self) -> usize { self.xyz.len() }

    fn ids(&self, key: &(i64, i64, i64)) -> &[u32] {
        match self.cells.get(key) {
            Some(&(start, len)) => &self.order[start as usize..(start + len) as usize],
            None => &[],
        }
    }

    /// Every stored point within `limit` of `q` (with its distance).
    pub fn within_pub(&self, q: &[f64; 3], limit: f64, f: impl FnMut(u32, f64)) {
        self.within(q, limit, f)
    }

    fn within(&self, q: &[f64; 3], limit: f64, mut f: impl FnMut(u32, f64)) {
        let (cx, cy, cz) = self.key(q);
        if limit < self.cell * 0.5 {
            // Only the neighbour on the query's side of each axis can hold a point within `limit`.
            let side = |v: f64, c: i64| if v / self.cell - (c as f64) < 0.5 { -1 } else { 1 };
            let (sx, sy, sz) = (side(q[0], cx), side(q[1], cy), side(q[2], cz));
            for dx in [0, sx] {
                for dy in [0, sy] {
                    for dz in [0, sz] {
                        for &id in self.ids(&(cx + dx, cy + dy, cz + dz)) {
                            let d = distance(q, &self.xyz[id as usize]);
                            if d <= limit {
                                f(id, d);
                            }
                        }
                    }
                }
            }
            return;
        }
        let reach = (limit / self.cell).ceil() as i64;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    for &id in self.ids(&(cx + dx, cy + dy, cz + dz)) {
                        let d = distance(q, &self.xyz[id as usize]);
                        if d <= limit {
                            f(id, d);
                        }
                    }
                }
            }
        }
    }

    /// Stored ids whose point lies in the closed box [lo, hi].
    pub fn in_box(&self, lo: &[f64; 3], hi: &[f64; 3]) -> Vec<u32> {
        let (a, b) = (self.key(lo), self.key(hi));
        let mut out = Vec::new();
        for x in a.0..=b.0 { for y in a.1..=b.1 { for z in a.2..=b.2 {
            out.extend(self.ids(&(x, y, z)).iter().copied().filter(|&id| { let p = &self.xyz[id as usize]; (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]) }));
        }}}
        out
    }

    /// `PointIndex.covered`: whether an allowed stored point lies within the radius
    /// (plus the group's 8-ULP guard) of each query.
    pub fn covered(&self, queries: &[[f64; 3]], radius: f64, allowed: &dyn Fn(u32) -> bool) -> Vec<bool> {
        let mut out = vec![false; queries.len()];
        for group in spatial_batches(queries) {
            let points: Vec<[f64; 3]> = group.iter().map(|&i| queries[i]).collect();
            let limit = radius + distance_roundoff(&points);
            for &qi in &group {
                let mut found = false;
                self.within(&queries[qi], limit, |id, _| if !found && allowed(id) { found = true });
                out[qi] = found;
            }
        }
        out
    }

    /// `PointIndex.nearest_distances`: the minimum distance within the radius plus the
    /// group's 8-ULP guard (infinity when none), for baseline coverage.
    pub fn nearest_distances(&self, queries: &[[f64; 3]], radius: f64) -> Vec<f64> {
        let mut out = vec![f64::INFINITY; queries.len()];
        for group in spatial_batches(queries) {
            let points: Vec<[f64; 3]> = group.iter().map(|&i| queries[i]).collect();
            let limit = radius + distance_roundoff(&points);
            for &qi in &group {
                let mut best = f64::INFINITY;
                self.within(&queries[qi], limit, |_, d| best = best.min(d));
                out[qi] = best;
            }
        }
        out
    }

    /// `PointIndex.nearest`: per query, the stored point (id) and distance chosen by the merge rule.
    /// `allowed(id)` filters stored points (tile, labels, overlaps).
    pub fn nearest(&self, queries: &[[f64; 3]], radius: f64, allowed: &dyn Fn(u32) -> bool) -> Vec<Option<(u32, f64)>> {
        self.nearest_for(queries, radius, &|id, _| allowed(id))
    }

    /// `nearest` with a filter that also sees the query point (per-tile overlap boxes).
    pub fn nearest_for(&self, queries: &[[f64; 3]], radius: f64, allowed: &dyn Fn(u32, &[f64; 3]) -> bool) -> Vec<Option<(u32, f64)>> {
        let mut out = vec![None; queries.len()];
        for group in spatial_batches(queries) {
            let points: Vec<[f64; 3]> = group.iter().map(|&i| queries[i]).collect();
            let roundoff = distance_roundoff(&points);
            let limit = radius + roundoff;
            for &qi in &group {
                let q = &queries[qi];
                let mut candidates: Vec<(u32, f64)> = Vec::new();
                self.within(q, limit, |id, d| if allowed(id, q) { candidates.push((id, d)) });
                let Some(minimum) = candidates.iter().map(|c| c.1).reduce(f64::min) else { continue };
                let ceiling = limit.min(minimum + roundoff);
                out[qi] = candidates.into_iter().filter(|c| c.1 <= ceiling)
                    .min_by_key(|c| (self.tile[c.0 as usize], self.index[c.0 as usize]));
            }
        }
        out
    }
}

/// `nearest_for` over several point sets queried together (e.g. one per tile): candidates from every
/// set compete under the same rule. Returns (set, local id, distance).
pub fn nearest_across(sets: &[&PointSet], queries: &[[f64; 3]], radius: f64, allowed: &dyn Fn(usize, u32, &[f64; 3]) -> bool)
                      -> Vec<Option<(usize, u32, f64)>> {
    let mut out = vec![None; queries.len()];
    for group in spatial_batches(queries) {
        let points: Vec<[f64; 3]> = group.iter().map(|&i| queries[i]).collect();
        let roundoff = distance_roundoff(&points);
        let limit = radius + roundoff;
        for &qi in &group {
            let q = &queries[qi];
            let mut candidates: Vec<(usize, u32, f64)> = Vec::new();
            for (s, set) in sets.iter().enumerate() {
                set.within(q, limit, |id, d| if allowed(s, id, q) { candidates.push((s, id, d)) });
            }
            let Some(minimum) = candidates.iter().map(|c| c.2).reduce(f64::min) else { continue };
            let ceiling = limit.min(minimum + roundoff);
            out[qi] = candidates.into_iter().filter(|c| c.2 <= ceiling)
                .min_by_key(|c| (sets[c.0].tile[c.1 as usize], sets[c.0].index[c.1 as usize]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_within_the_guard_take_the_lowest_reference() {
        let mut set = PointSet::new(0.2);
        set.push([0.1, 0.0, 0.0], 0, 7);
        set.push([-0.1, 0.0, 0.0], 0, 3);
        set.push([0.0, 0.05, 0.0], 1, 0);
        set.build();
        let found = set.nearest(&[[0.0, 0.0, 0.0]], 0.2, &|_| true);
        assert_eq!(found[0].unwrap().0, 2); // clearly nearest
        let found = set.nearest(&[[0.0, 0.0, 0.0]], 0.2, &|id| id != 2);
        assert_eq!(set.index[found[0].unwrap().0 as usize], 3); // exact tie: lowest point index
    }

    #[test]
    fn batches_follow_cells_then_input_order() {
        let xyz = [[3.0, 0.0, 0.0], [0.5, 0.5, 0.0], [2.5, 0.1, 0.0], [0.1, 3.0, 0.0]];
        assert_eq!(spatial_batches(&xyz), vec![vec![1], vec![3], vec![0, 2]]);
    }
}
