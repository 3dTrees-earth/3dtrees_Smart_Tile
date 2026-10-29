//! Native voxel reduction for SmartTile tiling (`tile_centroids.reduce_records`).
//!
//! Exactly the NumPy reducer's rules: absolute voxel keys from exact decimal
//! coefficients, voxels in key order, and per voxel either the integer mean
//! rounded half-to-even once or the point nearest that mean (ties: lowest
//! X, Y, Z). Output stays on the LAS integer grid; Python scales it.

/// Per axis `(a, b, denominator)`: key = floor((raw * a + b) / denominator).
pub type Coefficients = [(i128, i128, i128); 3];

#[derive(Debug, PartialEq, Eq)]
pub enum ReduceError {
    /// A voxel key does not fit in int64 (the NumPy reducer rejects it too).
    KeyOverflow,
    /// Beyond this reducer's integer widths; the caller uses the Python path.
    Unsupported,
}

/// One representative per occupied voxel, in voxel-key order.
pub fn reduce(points: &[[i32; 3]], coefficients: &Coefficients, scales: [f64; 3], nearest: bool)
    -> Result<Vec<[i64; 3]>, ReduceError> {
    if points.is_empty() {
        return Ok(Vec::new());
    }
    if points.len() > u32::MAX as usize {
        return Err(ReduceError::Unsupported);
    }
    let keys = voxel_keys(points, coefficients)?;
    let (order, ends) = group(&keys);
    let mut output = Vec::with_capacity(ends.len());
    let mut start = 0;
    for end in ends {
        let members = &order[start..end];
        let mut sums = [0i64; 3];
        for &i in members {
            for axis in 0..3 {
                sums[axis] = sums[axis].checked_add(i64::from(points[i as usize][axis]))
                    .ok_or(ReduceError::Unsupported)?;
            }
        }
        let count = members.len() as i64;
        output.push(if nearest {
            nearest_to_mean(points, members, sums, count, scales)
        } else {
            std::array::from_fn(|axis| round_half_even(sums[axis], count))
        });
        start = end;
    }
    Ok(output)
}

/// Absolute voxel keys of LAS integer points (`voxel_grid.voxel_keys`).
pub fn voxel_keys(points: &[[i32; 3]], coefficients: &Coefficients) -> Result<Vec<[i64; 3]>, ReduceError> {
    let mut keys = vec![[0i64; 3]; points.len()];
    for (axis, &(a, b, denominator)) in coefficients.iter().enumerate() {
        if a <= 0 || denominator <= 0 {
            return Err(ReduceError::Unsupported);
        }
        let (lo, hi) = points.iter().fold((i32::MAX, i32::MIN), |(lo, hi), p| (lo.min(p[axis]), hi.max(p[axis])));
        // `a` > 0, so keys are monotonic in the raw value: checking the
        // extremes bounds every key.
        let key = |v: i32| -> Result<i128, ReduceError> {
            i128::from(v).checked_mul(a).and_then(|n| n.checked_add(b))
                .map(|n| n.div_euclid(denominator)).ok_or(ReduceError::Unsupported)
        };
        if key(lo)? < i128::from(i64::MIN) || key(hi)? > i128::from(i64::MAX) {
            return Err(ReduceError::KeyOverflow);
        }
        // As in voxel_grid: whole origin voxels split off keep the
        // multiplication inside int64 for ordinary LAS encodings.
        let (origin, remainder) = (b.div_euclid(denominator), b.rem_euclid(denominator));
        let fits = |n: i128| i128::from(i64::MIN) <= n && n <= i128::from(i64::MAX);
        if fits(a) && fits(denominator) && fits(origin) && fits(i128::from(lo) * a)
            && fits(i128::from(hi) * a + remainder) {
            let (a, remainder, denominator, origin) = (a as i64, remainder as i64, denominator as i64, origin as i64);
            for (key, p) in keys.iter_mut().zip(points) {
                key[axis] = (i64::from(p[axis]) * a + remainder).div_euclid(denominator) + origin;
            }
        } else {
            for (k, p) in keys.iter_mut().zip(points) {
                k[axis] = key(p[axis])? as i64;
            }
        }
    }
    Ok(keys)
}

/// Point indices sorted by voxel key, and the end of each voxel's run.
fn group(keys: &[[i64; 3]]) -> (Vec<u32>, Vec<usize>) {
    let mut lo = keys[0];
    let mut hi = keys[0];
    for key in keys {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(key[axis]);
            hi[axis] = hi[axis].max(key[axis]);
        }
    }
    let span: [u128; 3] = std::array::from_fn(|axis| (i128::from(hi[axis]) - i128::from(lo[axis]) + 1) as u128);
    let order: Vec<u32>;
    let mut ends = Vec::new();
    if span[0].checked_mul(span[1]).and_then(|n| n.checked_mul(span[2])).is_some_and(|n| n < 1 << 62) {
        // Packed keys sort in the same (X, Y, Z) order at half the width.
        let offset = |key: &[i64; 3], axis: usize| (key[axis] - lo[axis]) as u64;
        let mut packed: Vec<(u64, u32)> = keys.iter().enumerate().map(|(i, key)| {
            ((offset(key, 0) * span[1] as u64 + offset(key, 1)) * span[2] as u64 + offset(key, 2), i as u32)
        }).collect();
        packed.sort_unstable();
        ends.extend((1..packed.len()).filter(|&i| packed[i].0 != packed[i - 1].0));
        order = packed.into_iter().map(|(_, i)| i).collect();
    } else {
        let mut sorted: Vec<([i64; 3], u32)> = keys.iter().enumerate().map(|(i, &key)| (key, i as u32)).collect();
        sorted.sort_unstable();
        ends.extend((1..sorted.len()).filter(|&i| sorted[i].0 != sorted[i - 1].0));
        order = sorted.into_iter().map(|(_, i)| i).collect();
    }
    ends.push(order.len());
    (order, ends)
}

/// Exact round-half-to-even of numerator / denominator (denominator > 0).
pub fn round_half_even(numerator: i64, denominator: i64) -> i64 {
    let quotient = numerator.div_euclid(denominator);
    let twice = 2 * i128::from(numerator.rem_euclid(denominator));
    let denominator = i128::from(denominator);
    quotient + i64::from(twice > denominator || (twice == denominator && quotient.rem_euclid(2) == 1))
}

fn nearest_to_mean(points: &[[i32; 3]], members: &[u32], sums: [i64; 3], count: i64, scales: [f64; 3]) -> [i64; 3] {
    let mean: [f64; 3] = std::array::from_fn(|axis| sums[axis] as f64 / count as f64);
    let distance = |p: &[i32; 3]| {
        let d: [f64; 3] = std::array::from_fn(|axis| (f64::from(p[axis]) - mean[axis]) * scales[axis]);
        // NumPy's einsum("ij,ij->i") over three columns sums X and Z, then
        // adds Y. Exact ties depend on that association, so keep it.
        (d[0] * d[0] + d[2] * d[2]) + d[1] * d[1]
    };
    let best = members.iter().map(|&i| &points[i as usize])
        .min_by(|a, b| distance(a).total_cmp(&distance(b)).then_with(|| a.cmp(b)))
        .expect("every voxel has a point");
    best.map(i64::from)
}

#[cfg(feature = "python")]
mod python {
    use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
    use pyo3::exceptions::{PyOverflowError, PyValueError};
    use pyo3::prelude::*;

    use super::{reduce, Coefficients, ReduceError};

    /// `reduce_voxels(x, y, z, coefficients, scales, nearest)` -> int64 (n, 3)
    /// representatives on the LAS integer grid, in voxel-key order.
    /// OverflowError means "use the Python reducer"; ValueError matches it.
    #[pyfunction]
    fn reduce_voxels<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, i32>, y: PyReadonlyArray1<'py, i32>,
                          z: PyReadonlyArray1<'py, i32>, coefficients: [(i128, i128, i128); 3],
                          scales: [f64; 3], nearest: bool) -> PyResult<Bound<'py, PyArray2<i64>>> {
        // Fields of a structured array are strided views: require copies.
        let contiguous = |_| PyValueError::new_err("X, Y and Z must be contiguous int32 arrays");
        let (x, y, z) = (x.as_slice().map_err(contiguous)?, y.as_slice().map_err(contiguous)?,
                         z.as_slice().map_err(contiguous)?);
        if x.len() != y.len() || x.len() != z.len() {
            return Err(PyValueError::new_err("X, Y and Z must have the same length"));
        }
        let points: Vec<[i32; 3]> = x.iter().zip(y).zip(z).map(|((&x, &y), &z)| [x, y, z]).collect();
        let coefficients: Coefficients = coefficients;
        let reduced = py.detach(|| reduce(&points, &coefficients, scales, nearest));
        match reduced {
            Ok(rows) => {
                let count = rows.len();
                PyArray1::from_vec(py, rows.into_flattened()).reshape([count, 3])
            }
            Err(ReduceError::KeyOverflow) => Err(PyValueError::new_err("Voxel keys exceed the int64 range")),
            Err(ReduceError::Unsupported) => Err(PyOverflowError::new_err("Outside the native reducer's integer range")),
        }
    }

    #[pymodule]
    fn smarttile_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(reduce_voxels, module)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CM: Coefficients = [(1, 0, 10), (1, 0, 10), (1, 0, 10)];

    #[test]
    fn half_even_including_negative() {
        assert_eq!((-5..=5).map(|n| round_half_even(n, 2)).collect::<Vec<_>>(), vec![-2, -2, -2, -1, 0, 0, 0, 1, 2, 2, 2]);
    }

    #[test]
    fn decimal_faces_and_negative_keys_floor() {
        // 0.001 m scale, 500000 m offset, 0.1 m voxels: 500001.1 is in voxel 5000011.
        let coefficients = [(1, 500_000_000, 100), (1, 0, 100), (1, 0, 100)];
        assert_eq!(voxel_keys(&[[1100, -1, -100]], &coefficients).unwrap(), vec![[5_000_011, -1, -1]]);
    }

    #[test]
    fn voxels_come_out_in_key_order_with_rounded_means() {
        let points = [[15, 0, 0], [3, 0, 0], [1, 0, 0], [12, 0, 0], [-1, 0, 0]];
        assert_eq!(reduce(&points, &CM, [0.001; 3], false).unwrap(), vec![[-1, 0, 0], [2, 0, 0], [14, 0, 0]]);
    }

    #[test]
    fn nearest_ties_take_lowest_xyz() {
        let points = [[4, 5, 0], [3, 6, 0], [2, 5, 0], [3, 4, 0]];
        assert_eq!(reduce(&points, &CM, [0.001; 3], true).unwrap(), vec![[2, 5, 0]]);
    }

    #[test]
    fn wide_spans_use_the_unpacked_sort() {
        let points = [[i32::MAX, 0, 0], [i32::MIN, 5, 0], [i32::MIN, 0, i32::MAX]];
        let unit = [(1, 0, 1), (1, 0, 1), (1, 0, 1)];
        assert_eq!(reduce(&points, &unit, [1.0; 3], false).unwrap(),
                   vec![[i64::from(i32::MIN), 0, i64::from(i32::MAX)], [i64::from(i32::MIN), 5, 0], [i64::from(i32::MAX), 0, 0]]);
    }

    #[test]
    fn keys_outside_int64_are_rejected() {
        let huge = [(1, i128::from(i64::MAX) * 4, 1), (1, 0, 1), (1, 0, 1)];
        assert_eq!(reduce(&[[1, 0, 0]], &huge, [1.0; 3], false), Err(ReduceError::KeyOverflow));
    }
}
