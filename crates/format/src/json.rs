//! JSON files as FORMAT §1 writes them: keys sorted, 2-space indentation, one trailing
//! newline. The keys are sorted here, so the output does not depend on how a `Map` is
//! ordered by whichever serde_json features the build happens to enable.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

/// The canonical text of a JSON value.
pub fn to_string(v: &Value) -> String {
    let mut out = String::new();
    write(v, 0, &mut out);
    out.push('\n');
    out
}

pub fn write_file(path: &Path, v: &Value) -> Result<()> {
    std::fs::write(path, to_string(v)).with_context(|| format!("writing {}", path.display()))
}

pub fn read_file(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn indent(depth: usize, out: &mut String) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn write(v: &Value, depth: usize, out: &mut String) {
    match v {
        Value::Array(a) if a.is_empty() => out.push_str("[]"),
        Value::Array(a) => {
            out.push_str("[\n");
            for (i, x) in a.iter().enumerate() {
                indent(depth + 1, out);
                write(x, depth + 1, out);
                if i + 1 < a.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            indent(depth, out);
            out.push(']');
        }
        Value::Object(m) if m.is_empty() => out.push_str("{}"),
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push_str("{\n");
            for (i, k) in keys.iter().enumerate() {
                indent(depth + 1, out);
                out.push_str(&Value::String((*k).clone()).to_string());
                out.push_str(": ");
                write(&m[*k], depth + 1, out);
                if i + 1 < keys.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            indent(depth, out);
            out.push('}');
        }
        // Scalars: serde_json prints integers without exponent and escapes strings.
        _ => out.push_str(&v.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn sorted_and_indented() {
        let v = json!({"b": [1, {"z": true, "a": null}], "a": "x\"y", "c": {}, "d": []});
        assert_eq!(
            super::to_string(&v),
            "{\n  \"a\": \"x\\\"y\",\n  \"b\": [\n    1,\n    {\n      \"a\": null,\n      \"z\": true\n    }\n  ],\n  \"c\": {},\n  \"d\": []\n}\n"
        );
    }
}
