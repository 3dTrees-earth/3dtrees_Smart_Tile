//! Python's text spellings: `repr(float)` and `json.dumps(..., indent=2)`.
//!
//! Layout JSON and job lists are read by other tools and compared with
//! Python SmartTile's files, so numbers are written exactly as Python does.

use serde_json::Value;

/// `repr(x)` for a Python float.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Shortest round-trip digits, as `d.ddde±X`.
    let sci = format!("{x:e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if negative { "-" } else { "" };
    if (-4..16).contains(&exponent) {
        let point = exponent + 1; // digits before the decimal point
        let text = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{text}")
    } else {
        let tail = if digits.len() > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        format!("{sign}{}{}e{}{:02}", &digits[..1], tail, if exponent < 0 { '-' } else { '+' }, exponent.abs())
    }
}

fn string_json(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                // ensure_ascii: UTF-16 escapes, surrogate pairs above the BMP.
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_value(value: &Value, indent: usize, out: &mut String) {
    let pad = |n: usize| "  ".repeat(n);
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&float_repr(n.as_f64().unwrap()));
            }
        }
        Value::String(s) => string_json(s, out),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                out.push_str(&pad(indent + 1));
                write_value(item, indent + 1, out);
                out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad(indent));
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (i, (key, item)) in map.iter().enumerate() {
                out.push_str(&pad(indent + 1));
                string_json(key, out);
                out.push_str(": ");
                write_value(item, indent + 1, out);
                out.push_str(if i + 1 < map.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad(indent));
            out.push('}');
        }
    }
}

/// `json.dumps(value, indent=2)`.
pub fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, 0, &mut out);
    out
}

/// A JSON float that keeps Python's float spelling (20.0, not 20).
pub fn float(x: f64) -> Value {
    Value::Number(serde_json::Number::from_f64(x).expect("finite float"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_spell_like_python() {
        for (x, text) in [(20.0, "20.0"), (437149.888, "437149.888"), (1e16, "1e+16"), (1e-5, "1e-05"),
                          (0.0001, "0.0001"), (-117.93069221151073, "-117.93069221151073"),
                          (1.5e300, "1.5e+300"), (123456789012345.6, "123456789012345.6")] {
            assert_eq!(float_repr(x), text);
        }
    }

    #[test]
    fn json_matches_python_indent_2() {
        let value = serde_json::json!({"a": [float(1.0), 2], "b": {}, "c": "é"});
        assert_eq!(dumps(&value), "{\n  \"a\": [\n    1.0,\n    2\n  ],\n  \"b\": {},\n  \"c\": \"\\u00e9\"\n}");
    }
}

/// Python's `f"{n:,}"` for an integer.
pub fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 { out.push(','); }
        out.push(c);
    }
    out
}
