//! Exact decimals of floats, as Python's `Fraction(str(x))` reads them.
//!
//! LAS scales/offsets and voxel resolutions are decimal lengths; SmartTile
//! works with their shortest decimal spelling, not their binary value.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ratio {
    pub num: i128,
    pub den: i128, // > 0, coprime with num
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Ratio {
    pub fn new(num: i128, den: i128) -> Ratio {
        assert!(den != 0, "zero denominator");
        let g = gcd(num, den).max(1);
        let sign = if den < 0 { -1 } else { 1 };
        Ratio { num: sign * num / g, den: sign * den / g }
    }

    pub fn int(n: i128) -> Ratio {
        Ratio { num: n, den: 1 }
    }

    /// The decimal Python writes for `x` (shortest round-trip spelling).
    pub fn from_f64(x: f64) -> Result<Ratio, String> {
        if !x.is_finite() {
            return Err(format!("{x} is not a finite decimal"));
        }
        // Rust's Display, like Python's repr, prints the shortest digits that
        // round-trip; it never uses an exponent, so parsing is plain.
        let text = format!("{x}");
        let (negative, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.as_str()),
        };
        let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
        let mut num: i128 = 0;
        for c in whole.chars().chain(fraction.chars()) {
            num = num.checked_mul(10).and_then(|n| n.checked_add(i128::from(c as u8 - b'0')))
                .ok_or_else(|| format!("{x} has too many decimal digits"))?;
        }
        let den = 10i128.checked_pow(fraction.len() as u32).ok_or_else(|| format!("{x} is too fine"))?;
        Ok(Ratio::new(if negative { -num } else { num }, den))
    }

    pub fn checked_add(self, other: Ratio) -> Option<Ratio> {
        let num = self.num.checked_mul(other.den)?.checked_add(other.num.checked_mul(self.den)?)?;
        Some(Ratio::new(num, self.den.checked_mul(other.den)?))
    }

    pub fn checked_mul(self, other: Ratio) -> Option<Ratio> {
        Some(Ratio::new(self.num.checked_mul(other.num)?, self.den.checked_mul(other.den)?))
    }

    pub fn checked_div(self, other: Ratio) -> Option<Ratio> {
        if other.num == 0 {
            return None;
        }
        Some(Ratio::new(self.num.checked_mul(other.den)?, self.den.checked_mul(other.num)?))
    }

    /// Floor division, as Python's `Fraction // Fraction`.
    pub fn floor_div(self, other: Ratio) -> Option<i128> {
        let q = self.checked_div(other)?;
        Some(q.num.div_euclid(q.den))
    }

    /// The nearest float (correctly rounded), as Python's `float(Fraction)`.
    pub fn to_f64(self) -> f64 {
        if self.den == 1 {
            return format!("{}", self.num).parse().unwrap();
        }
        // A terminating decimal (den = 2^a 5^b) spelled out exactly, then
        // parsed: Rust's parser rounds correctly. Others fall back to division.
        let (mut d, mut twos, mut fives) = (self.den, 0u32, 0u32);
        while d % 2 == 0 { d /= 2; twos += 1; }
        while d % 5 == 0 { d /= 5; fives += 1; }
        if d == 1 {
            let places = twos.max(fives);
            if let Some(scale) = 10i128.checked_pow(places).and_then(|p| p.checked_div(self.den)) {
                if let Some(scaled) = self.num.checked_mul(scale) {
                    let sign = if scaled < 0 { "-" } else { "" };
                    let digits = format!("{:0>width$}", scaled.unsigned_abs(), width = places as usize + 1);
                    let (whole, fraction) = digits.split_at(digits.len() - places as usize);
                    return format!("{sign}{whole}.{fraction}").parse().unwrap();
                }
            }
        }
        self.num as f64 / self.den as f64
    }
}

impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

/// `voxel_grid._coefficients`: (a, b, denominator) with key = (raw * a + b) // denominator.
pub fn coefficients(scale: f64, offset: f64, resolution: f64) -> Result<(i128, i128, i128), String> {
    let overflow = || format!("voxel coefficients overflow for scale {scale}, offset {offset}, resolution {resolution}");
    let resolution = Ratio::from_f64(resolution)?;
    let step = Ratio::from_f64(scale)?.checked_div(resolution).ok_or_else(overflow)?;
    let origin = Ratio::from_f64(offset)?.checked_div(resolution).ok_or_else(overflow)?;
    let den = step.den / gcd(step.den, origin.den) * origin.den;
    Ok((step.num.checked_mul(den / step.den).ok_or_else(overflow)?,
        origin.num.checked_mul(den / origin.den).ok_or_else(overflow)?, den))
}

/// Python's `round(x)` for floats: nearest integer, ties to even.
pub fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimals_follow_the_shortest_spelling() {
        assert_eq!(Ratio::from_f64(0.1).unwrap(), Ratio::new(1, 10));
        assert_eq!(Ratio::from_f64(-500000.1).unwrap(), Ratio::new(-5000001, 10));
        assert_eq!(Ratio::from_f64(32704.229999999996).unwrap(), Ratio::new(32704229999999996, 1_000_000_000_000));
    }

    #[test]
    fn coefficients_match_python() {
        // _coefficients(0.001, 615000.0, 0.01) == (1, 615000000, 10)
        assert_eq!(coefficients(0.001, 615000.0, 0.01).unwrap(), (1, 615_000_000, 10));
        assert_eq!(coefficients(0.001, 615000.0, 0.1).unwrap(), (1, 615_000_000, 100));
    }

    #[test]
    fn exact_ratios_round_to_the_nearest_float() {
        assert_eq!(Ratio::new(3270423, 100).to_f64(), 32704.23);
        assert_eq!(Ratio::new(-1, 3).to_f64(), -1.0 / 3.0);
    }
}
