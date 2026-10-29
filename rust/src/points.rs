//! LAS point records as raw bytes: standard dimensions of every point format,
//! ExtraBytes descriptions, and laspy-compatible ExtraBytes VLRs.

use crate::lasio::{Res, SourceHeader, Vlr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar { U8, I8, U16, I16, U32, I32, U64, I64, F32, F64 }

impl Scalar {
    pub fn size(self) -> usize {
        match self { Scalar::U8 | Scalar::I8 => 1, Scalar::U16 | Scalar::I16 => 2, Scalar::U32 | Scalar::I32 | Scalar::F32 => 4, _ => 8 }
    }
    pub fn is_integer(self) -> bool { !matches!(self, Scalar::F32 | Scalar::F64) }
    pub fn is_signed(self) -> bool { matches!(self, Scalar::I8 | Scalar::I16 | Scalar::I32 | Scalar::I64) }
    /// LAS ExtraBytes data type (1..=10).
    pub fn extra_bytes_id(self) -> u8 {
        match self { Scalar::U8 => 1, Scalar::I8 => 2, Scalar::U16 => 3, Scalar::I16 => 4, Scalar::U32 => 5,
                     Scalar::I32 => 6, Scalar::U64 => 7, Scalar::I64 => 8, Scalar::F32 => 9, Scalar::F64 => 10 }
    }
    pub fn from_extra_bytes_id(id: u8) -> Option<Scalar> {
        Some(match id { 1 => Scalar::U8, 2 => Scalar::I8, 3 => Scalar::U16, 4 => Scalar::I16, 5 => Scalar::U32,
                        6 => Scalar::I32, 7 => Scalar::U64, 8 => Scalar::I64, 9 => Scalar::F32, 10 => Scalar::F64, _ => return None })
    }
    /// PLY property type name.
    pub fn ply(self) -> &'static str {
        match self { Scalar::U8 => "uchar", Scalar::I8 => "char", Scalar::U16 => "ushort", Scalar::I16 => "short",
                     Scalar::U32 => "uint", Scalar::I32 => "int", Scalar::F32 => "float", Scalar::F64 => "double",
                     Scalar::U64 | Scalar::I64 => "double" }
    }
    /// Read as i128 (integers) for IDs and comparisons.
    pub fn read_int(self, b: &[u8]) -> i128 {
        match self {
            Scalar::U8 => b[0] as i128, Scalar::I8 => b[0] as i8 as i128,
            Scalar::U16 => u16::from_le_bytes([b[0], b[1]]) as i128, Scalar::I16 => i16::from_le_bytes([b[0], b[1]]) as i128,
            Scalar::U32 => u32::from_le_bytes(b[..4].try_into().unwrap()) as i128, Scalar::I32 => i32::from_le_bytes(b[..4].try_into().unwrap()) as i128,
            Scalar::U64 => u64::from_le_bytes(b[..8].try_into().unwrap()) as i128, Scalar::I64 => i64::from_le_bytes(b[..8].try_into().unwrap()) as i128,
            Scalar::F32 => f32::from_le_bytes(b[..4].try_into().unwrap()) as i128, Scalar::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()) as i128,
        }
    }
}

/// Where a dimension lives in a record: whole bytes or a bit field of one byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout { Bytes { offset: usize, scalar: Scalar }, Bits { offset: usize, shift: u8, bits: u8 } }

impl Layout {
    pub fn num_bits(self) -> u32 {
        match self { Layout::Bytes { scalar, .. } => scalar.size() as u32 * 8, Layout::Bits { bits, .. } => bits as u32 }
    }
    /// The stored value (bit fields as uint8), little-endian bytes of its storage type.
    pub fn get<'a>(self, record: &'a [u8], scratch: &'a mut [u8; 8]) -> &'a [u8] {
        match self {
            Layout::Bytes { offset, scalar } => &record[offset..offset + scalar.size()],
            Layout::Bits { offset, shift, bits } => {
                scratch[0] = (record[offset] >> shift) & ((1u16 << bits) - 1) as u8;
                &scratch[..1]
            }
        }
    }
    pub fn set(self, record: &mut [u8], value: &[u8]) {
        match self {
            Layout::Bytes { offset, scalar } => record[offset..offset + scalar.size()].copy_from_slice(&value[..scalar.size()]),
            Layout::Bits { offset, shift, bits } => {
                let mask = (((1u16 << bits) - 1) as u8) << shift;
                record[offset] = (record[offset] & !mask) | ((value[0] << shift) & mask);
            }
        }
    }
    pub fn scalar(self) -> Scalar {
        match self { Layout::Bytes { scalar, .. } => scalar, Layout::Bits { .. } => Scalar::U8 }
    }
}

/// One ExtraBytes description (192-byte struct of the LASF_Spec 4 VLR).
#[derive(Clone, Debug, PartialEq)]
pub struct ExtraBytes {
    pub name: String,
    pub data_type: u8,
    pub options: u8,
    pub no_data: [u8; 24],
    pub scale: [f64; 3],
    pub offset: [f64; 3],
    pub description: [u8; 32],
    /// Stored minimum/maximum (raw 64-bit values), as read.
    pub min: [u8; 24],
    pub max: [u8; 24],
}

impl ExtraBytes {
    pub fn scalar(&self) -> Option<Scalar> { Scalar::from_extra_bytes_id(self.data_type) }
    pub fn size(&self) -> usize {
        match self.data_type { 0 => self.options as usize, 1..=10 => self.scalar().unwrap().size(),
                               t @ 11..=20 => 2 * Scalar::from_extra_bytes_id(t - 10).unwrap().size(),
                               t => 3 * Scalar::from_extra_bytes_id(t - 20).unwrap().size() }
    }
    pub fn has_no_data(&self) -> bool { self.data_type != 0 && self.options & 1 != 0 }
    pub fn has_scale(&self) -> bool { self.data_type != 0 && self.options & 8 != 0 }
    pub fn has_offset(&self) -> bool { self.data_type != 0 && self.options & 16 != 0 }

    fn parse(b: &[u8]) -> ExtraBytes {
        let name_end = b[4..36].iter().position(|&c| c == 0).unwrap_or(32);
        let f = |o: usize| f64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        ExtraBytes {
            name: String::from_utf8_lossy(&b[4..4 + name_end]).into_owned(),
            data_type: b[2], options: b[3],
            no_data: b[40..64].try_into().unwrap(),
            scale: [f(112), f(120), f(128)], offset: [f(136), f(144), f(152)],
            description: b[160..192].try_into().unwrap(),
            min: b[64..88].try_into().unwrap(), max: b[88..112].try_into().unwrap(),
        }
    }
}

/// Every dimension of a point format, standard first, then extra bytes.
#[derive(Clone, Debug)]
pub struct PointLayout {
    pub format: u8,
    pub record_length: usize,
    pub standard: Vec<(String, Layout)>,
    pub extra: Vec<(ExtraBytes, usize)>, // (description, byte offset)
}

fn standard_dims(format: u8) -> Vec<(&'static str, Layout)> {
    use Layout::*;
    use Scalar::*;
    let mut d = vec![("X", Bytes { offset: 0, scalar: I32 }), ("Y", Bytes { offset: 4, scalar: I32 }),
                     ("Z", Bytes { offset: 8, scalar: I32 }), ("intensity", Bytes { offset: 12, scalar: U16 })];
    let (time, rgb, nir) = if format < 6 {
        d.extend([("return_number", Bits { offset: 14, shift: 0, bits: 3 }), ("number_of_returns", Bits { offset: 14, shift: 3, bits: 3 }),
                  ("scan_direction_flag", Bits { offset: 14, shift: 6, bits: 1 }), ("edge_of_flight_line", Bits { offset: 14, shift: 7, bits: 1 }),
                  ("classification", Bits { offset: 15, shift: 0, bits: 5 }), ("synthetic", Bits { offset: 15, shift: 5, bits: 1 }),
                  ("key_point", Bits { offset: 15, shift: 6, bits: 1 }), ("withheld", Bits { offset: 15, shift: 7, bits: 1 }),
                  ("scan_angle_rank", Bytes { offset: 16, scalar: I8 }), ("user_data", Bytes { offset: 17, scalar: U8 }),
                  ("point_source_id", Bytes { offset: 18, scalar: U16 })]);
        let time = matches!(format, 1 | 3 | 4 | 5).then_some(20);
        let rgb = match format { 2 => Some(20), 3 | 5 => Some(28), _ => None };
        (time, rgb, None)
    } else {
        d.extend([("return_number", Bits { offset: 14, shift: 0, bits: 4 }), ("number_of_returns", Bits { offset: 14, shift: 4, bits: 4 }),
                  ("synthetic", Bits { offset: 15, shift: 0, bits: 1 }), ("key_point", Bits { offset: 15, shift: 1, bits: 1 }),
                  ("withheld", Bits { offset: 15, shift: 2, bits: 1 }), ("overlap", Bits { offset: 15, shift: 3, bits: 1 }),
                  ("scanner_channel", Bits { offset: 15, shift: 4, bits: 2 }), ("scan_direction_flag", Bits { offset: 15, shift: 6, bits: 1 }),
                  ("edge_of_flight_line", Bits { offset: 15, shift: 7, bits: 1 }), ("classification", Bytes { offset: 16, scalar: U8 }),
                  ("user_data", Bytes { offset: 17, scalar: U8 }), ("scan_angle", Bytes { offset: 18, scalar: I16 }),
                  ("point_source_id", Bytes { offset: 20, scalar: U16 })]);
        let rgb = matches!(format, 7 | 8 | 10).then_some(30);
        let nir = matches!(format, 8 | 10).then_some(36);
        (Some(22), rgb, nir)
    };
    if let Some(o) = time { d.push(("gps_time", Bytes { offset: o, scalar: F64 })); }
    if let Some(o) = rgb {
        d.extend([("red", Bytes { offset: o, scalar: U16 }), ("green", Bytes { offset: o + 2, scalar: U16 }), ("blue", Bytes { offset: o + 4, scalar: U16 })]);
    }
    if let Some(o) = nir { d.push(("nir", Bytes { offset: o, scalar: U16 })); }
    d
}

/// laspy's standard dimension names of every point format (never predictions).
pub fn all_standard_names() -> Vec<String> {
    let mut names: Vec<String> = (0..=10u8).flat_map(standard_dims).map(|(n, _)| n.to_string()).collect();
    names.extend(["wavepacket_index", "wavepacket_offset", "wavepacket_size", "return_point_wave_location", "x_t", "y_t", "z_t",
                  "OriginalPointIndex"].map(String::from));
    names.sort();
    names.dedup();
    names
}

/// laspy's raw record fields of a point format (whole bytes, in record order): the
/// sort key of numpy's structured-array comparison.
pub fn raw_fields(format: u8) -> Vec<(usize, Scalar)> {
    use Scalar::*;
    let mut f = vec![(0, I32), (4, I32), (8, I32), (12, U16)];
    let mut at;
    if format < 6 {
        f.extend([(14, U8), (15, U8), (16, I8), (17, U8), (18, U16)]);
        at = 20;
        if matches!(format, 1 | 3 | 4 | 5) { f.push((at, F64)); at += 8; }
        if matches!(format, 2 | 3 | 5) { f.extend([(at, U16), (at + 2, U16), (at + 4, U16)]); at += 6; }
        if matches!(format, 4 | 5) { f.extend([(at, U8), (at + 1, U64), (at + 9, U32), (at + 13, F32), (at + 17, F32), (at + 21, F32), (at + 25, F32)]); }
    } else {
        f.extend([(14, U8), (15, U8), (16, U8), (17, U8), (18, I16), (20, U16), (22, F64)]);
        at = 30;
        if matches!(format, 7 | 8 | 10) { f.extend([(at, U16), (at + 2, U16), (at + 4, U16)]); at += 6; }
        if matches!(format, 8 | 10) { f.push((at, U16)); at += 2; }
        if matches!(format, 9 | 10) { f.extend([(at, U8), (at + 1, U64), (at + 9, U32), (at + 13, F32), (at + 17, F32), (at + 21, F32), (at + 25, F32)]); }
    }
    f
}

/// numpy's ordering of one field (NaN after numbers).
pub fn compare_field(scalar: Scalar, a: &[u8], b: &[u8]) -> std::cmp::Ordering {
    match scalar {
        Scalar::F32 | Scalar::F64 => {
            let (x, y) = if scalar == Scalar::F32 { (f32::from_le_bytes(a[..4].try_into().unwrap()) as f64, f32::from_le_bytes(b[..4].try_into().unwrap()) as f64) }
                         else { (f64::from_le_bytes(a[..8].try_into().unwrap()), f64::from_le_bytes(b[..8].try_into().unwrap())) };
            x.partial_cmp(&y).unwrap_or_else(|| x.is_nan().cmp(&y.is_nan()))
        }
        s => s.read_int(a).cmp(&s.read_int(b)),
    }
}

pub fn base_record_length(format: u8) -> usize {
    [20, 28, 26, 34, 57, 63, 30, 36, 38, 59, 67][format as usize]
}

impl PointLayout {
    pub fn of(header: &SourceHeader) -> Res<PointLayout> {
        if header.point_format > 10 {
            return Err(format!("unsupported point format {}", header.point_format));
        }
        let mut extra = Vec::new();
        let mut offset = base_record_length(header.point_format);
        if let Some(vlr) = header.vlrs.iter().chain(&header.evlrs).find(|v| v.user() == "LASF_Spec" && v.record_id == 4) {
            for chunk in vlr.data.chunks_exact(192) {
                let eb = ExtraBytes::parse(chunk);
                let size = eb.size();
                extra.push((eb, offset));
                offset += size;
            }
        }
        Ok(PointLayout {
            format: header.point_format,
            record_length: header.record_length as usize,
            standard: standard_dims(header.point_format).into_iter().map(|(n, l)| (n.to_string(), l)).collect(),
            extra,
        })
    }

    /// A dimension by name: its layout and, for ExtraBytes, its description.
    pub fn find(&self, name: &str) -> Option<(Layout, Option<&ExtraBytes>)> {
        if let Some((_, l)) = self.standard.iter().find(|(n, _)| n == name) {
            return Some((*l, None));
        }
        self.extra.iter().find(|(e, _)| e.name == name).and_then(|(e, o)| {
            e.scalar().map(|scalar| (Layout::Bytes { offset: *o, scalar }, Some(e)))
        })
    }
}

/// (smallest, largest) value of the 64-bit type laspy tracks a scalar in.
fn reset_bounds(scalar: Scalar) -> ([u8; 8], [u8; 8]) {
    match scalar {
        Scalar::F32 | Scalar::F64 => ((-f64::MAX).to_le_bytes(), f64::MAX.to_le_bytes()),
        s if s.is_signed() => (i64::MIN.to_le_bytes(), i64::MAX.to_le_bytes()),
        _ => (u64::MIN.to_le_bytes(), u64::MAX.to_le_bytes()),
    }
}

/// laspy `ExtraBytesStruct` for a written extra dimension, with laspy's min/max tracking.
#[derive(Clone, Debug)]
pub struct OutputExtra {
    pub source: ExtraBytes, // name, type, no-data/scale/offset bits and values, description
    pub options: u8,
    pub min: [u8; 24],
    pub max: [u8; 24],
}

impl OutputExtra {
    /// Options as laspy sets them: no-data/scale/offset from the source, plus min and max.
    pub fn new(source: &ExtraBytes) -> OutputExtra {
        let scalar = source.scalar().expect("scalar extra dimension");
        let mut source = source.clone();
        // A new laspy struct holds only the scalar's first element.
        source.scale[1] = 0.0; source.scale[2] = 0.0;
        source.offset[1] = 0.0; source.offset[2] = 0.0;
        source.no_data[8..].fill(0);
        // laspy: a scale without an offset (or vice versa) gets the default other one.
        if source.has_scale() != source.has_offset() {
            if !source.has_scale() { source.scale = [1.0; 3]; }
            if !source.has_offset() { source.offset = [0.0; 3]; }
            source.options |= 8 | 16;
        }
        let (mut min, mut max) = ([0u8; 24], [0u8; 24]);
        // partial_reset: min = largest, max = smallest value of the 64-bit type.
        let (lo, hi) = reset_bounds(scalar);
        min[..8].copy_from_slice(&hi);
        max[..8].copy_from_slice(&lo);
        let options = (if source.has_no_data() { 1 } else { 0 }) | (if source.has_scale() { 8 } else { 0 })
            | (if source.has_offset() { 16 } else { 0 }) | 2 | 4;
        OutputExtra { source, options, min, max }
    }

    /// A descriptor kept from a copied header: its options and stored min/max continue.
    /// laspy resets min/max when a writer opens (`partial_reset`), keeping the options.
    pub fn stored(source: &ExtraBytes) -> OutputExtra {
        let mut out = OutputExtra { source: source.clone(), options: source.options, min: source.min, max: source.max };
        if let Some(scalar) = source.scalar() {
            let (lo, hi) = reset_bounds(scalar);
            out.min[..8].copy_from_slice(&hi);
            out.max[..8].copy_from_slice(&lo);
        }
        out
    }

    /// laspy `ExtraBytesStruct.grow` on one written batch. A scalar field is
    /// indexed `pts[..., 0]`, so only the batch's first point counts, and a
    /// declared no-data value stops the update.
    pub fn grow(&mut self, first_value: &[u8]) {
        if self.source.data_type == 0 || self.options & 6 == 0 || self.source.has_no_data() {
            return;
        }
        let scalar = self.source.scalar().unwrap();
        let widen = |b: &[u8]| -> [u8; 8] {
            match scalar {
                Scalar::F32 => (f32::from_le_bytes(b[..4].try_into().unwrap()) as f64).to_le_bytes(),
                Scalar::F64 => b[..8].try_into().unwrap(),
                s if s.is_signed() => (s.read_int(b) as i64).to_le_bytes(),
                s => (s.read_int(b) as u64).to_le_bytes(),
            }
        };
        let value = widen(first_value);
        let cmp = |a: &[u8; 8], b: &[u8; 8]| -> std::cmp::Ordering {
            match scalar {
                Scalar::F32 | Scalar::F64 => f64::from_le_bytes(*a).total_cmp(&f64::from_le_bytes(*b)),
                s if s.is_signed() => i64::from_le_bytes(*a).cmp(&i64::from_le_bytes(*b)),
                _ => u64::from_le_bytes(*a).cmp(&u64::from_le_bytes(*b)),
            }
        };
        let current_min: [u8; 8] = self.min[..8].try_into().unwrap();
        let current_max: [u8; 8] = self.max[..8].try_into().unwrap();
        if self.options & 2 != 0 && cmp(&value, &current_min).is_lt() { self.min[..8].copy_from_slice(&value); }
        if self.options & 4 != 0 && cmp(&value, &current_max).is_gt() { self.max[..8].copy_from_slice(&value); }
    }

    pub fn bytes(&self) -> Vec<u8> {
        let s = &self.source;
        let mut b = vec![0u8; 192];
        b[2] = s.data_type;
        b[3] = self.options;
        let name = s.name.as_bytes();
        b[4..4 + name.len().min(32)].copy_from_slice(&name[..name.len().min(32)]);
        if self.options & 1 != 0 { b[40..64].copy_from_slice(&s.no_data); }
        b[64..88].copy_from_slice(&self.min);
        b[88..112].copy_from_slice(&self.max);
        if self.options & 8 != 0 { for i in 0..3 { b[112 + 8 * i..120 + 8 * i].copy_from_slice(&s.scale[i].to_le_bytes()); } }
        if self.options & 16 != 0 { for i in 0..3 { b[136 + 8 * i..144 + 8 * i].copy_from_slice(&s.offset[i].to_le_bytes()); } }
        b[160..192].copy_from_slice(&s.description);
        b
    }
}

pub fn extra_bytes_vlr(extras: &[OutputExtra]) -> Vlr {
    Vlr::new("LASF_Spec", 4, "Extra Bytes Record", extras.iter().flat_map(|e| e.bytes()).collect())
}
