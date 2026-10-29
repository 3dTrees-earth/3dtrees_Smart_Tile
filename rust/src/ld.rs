//! x87 extended precision (`np.longdouble` on x86-64): a 64-bit significand,
//! round to nearest even. Merge computes local coordinates as
//! `(X * scale + (offset - origin))` in it, then rounds to float64
//! (`bounded_point_index.coordinates`); Rust has no such type.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F80 { negative: bool, mantissa: u64, exponent: i32 } // value = mantissa * 2^exponent; zero has mantissa 0

/// Round an exact value `m * 2^e` (m < 2^128) to `bits` significant bits, nearest even.
fn round_bits(m: u128, e: i32, bits: u32) -> (u128, i32) {
    if m == 0 {
        return (0, 0);
    }
    let width = 128 - m.leading_zeros();
    if width <= bits {
        return (m, e);
    }
    let shift = width - bits;
    let kept = m >> shift;
    let rest = m & ((1u128 << shift) - 1);
    let half = 1u128 << (shift - 1);
    let rounded = if rest > half || (rest == half && kept & 1 == 1) { kept + 1 } else { kept };
    if rounded >> bits != 0 {
        (rounded >> 1, e + shift as i32 + 1)
    } else {
        (rounded, e + shift as i32)
    }
}

impl F80 {
    pub const ZERO: F80 = F80 { negative: false, mantissa: 0, exponent: 0 };

    pub fn from_f64(x: f64) -> F80 {
        assert!(x.is_finite());
        if x == 0.0 {
            return F80::ZERO;
        }
        let bits = x.to_bits();
        let negative = bits >> 63 == 1;
        let raw_exp = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & ((1u64 << 52) - 1);
        let (mantissa, exponent) = if raw_exp == 0 { (frac, -1074) } else { (frac | (1u64 << 52), raw_exp - 1075) };
        F80 { negative, mantissa, exponent }
    }

    pub fn from_i64(x: i64) -> F80 {
        let (m, e) = round_bits(x.unsigned_abs() as u128, 0, 64);
        F80 { negative: x < 0, mantissa: m as u64, exponent: e }
    }

    pub fn mul(self, other: F80) -> F80 {
        if self.mantissa == 0 || other.mantissa == 0 {
            return F80::ZERO;
        }
        let (m, e) = round_bits(self.mantissa as u128 * other.mantissa as u128, self.exponent + other.exponent, 64);
        F80 { negative: self.negative != other.negative, mantissa: m as u64, exponent: e }
    }

    pub fn neg(self) -> F80 {
        F80 { negative: !self.negative && self.mantissa != 0, ..self }
    }

    /// The same value with a full 64-bit significand.
    fn normalized(self) -> F80 {
        let shift = self.mantissa.leading_zeros();
        F80 { mantissa: self.mantissa << shift, exponent: self.exponent - shift as i32, ..self }
    }

    pub fn add(self, other: F80) -> F80 {
        if self.mantissa == 0 { return other; }
        if other.mantissa == 0 { return self; }
        let (a, b) = (self.normalized(), other.normalized());
        let (a, b) = if a.exponent >= b.exponent { (a, b) } else { (b, a) };
        let diff = (a.exponent - b.exponent) as u32;
        let (am, bm, e) = if diff <= 64 {
            ((a.mantissa as u128) << diff, b.mantissa as u128, b.exponent)
        } else {
            // b lies below a's last bit: three guard bits and a sticky bit suffice.
            let shift = diff - 3;
            let kept = if shift >= 64 { 0 } else { (b.mantissa >> shift) as u128 };
            ((a.mantissa as u128) << 3, kept | 1, a.exponent - 3)
        };
        let (negative, m) = if a.negative == b.negative {
            (a.negative, am + bm)
        } else if am >= bm {
            (a.negative, am - bm)
        } else {
            (b.negative, bm - am)
        };
        if m == 0 {
            return F80::ZERO;
        }
        let (m, e) = round_bits(m, e, 64);
        F80 { negative, mantissa: m as u64, exponent: e }
    }

    pub fn sub(self, other: F80) -> F80 {
        self.add(other.neg())
    }

    /// Nearest float64, ties to even (normal range).
    pub fn to_f64(self) -> f64 {
        if self.mantissa == 0 {
            return 0.0;
        }
        let (m, e) = round_bits(self.mantissa as u128, self.exponent, 53);
        let value = (m as f64) * 2f64.powi(e);
        if self.negative { -value } else { value }
    }
}

/// `bounded_point_index.coordinates` for one axis value.
pub fn local_coordinate(raw: i32, scale: f64, offset: f64, origin: f64) -> f64 {
    F80::from_i64(raw as i64).mul(F80::from_f64(scale)).add(F80::from_f64(offset).sub(F80::from_f64(origin))).to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values_round_trip() {
        for x in [0.001, 437180.0, -0.5, 1e-300, 123456.789] {
            assert_eq!(F80::from_f64(x).to_f64(), x);
        }
    }

    #[test]
    fn matches_numpy_longdouble_samples() {
        // (X, scale, offset, origin) -> np.longdouble result as float64, from numpy 2.5.3 on x86-64.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/longdouble_samples.txt");
        let text = std::fs::read_to_string(path).unwrap();
        for line in text.lines() {
            let v: Vec<&str> = line.split_whitespace().collect();
            let raw: i32 = v[0].parse().unwrap();
            let f = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).unwrap());
            assert_eq!(local_coordinate(raw, f(v[1]), f(v[2]), f(v[3])).to_bits(), f(v[4]).to_bits(), "{line}");
        }
    }
}
