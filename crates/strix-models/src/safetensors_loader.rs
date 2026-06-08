//! Safetensors weight loading, materialized to `f32`.
//!
//! Phase 1 path: memory-map each `*.safetensors` file in a directory, convert
//! every tensor to an owned `f32` buffer, and hand back a name→tensor map. This
//! is memory-heavy (4 bytes/param, dtype upcast) but dead simple and correct —
//! the right tradeoff for the reference backend on small models. Quantized /
//! lazy / sharded-by-need loading comes later.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use strix_core::error::{Result, StrixError};

/// A weight tensor materialized as row-major `f32`.
#[derive(Debug, Clone)]
pub struct RawTensor {
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Row-major data, upcast to `f32`.
    pub data: Vec<f32>,
}

impl RawTensor {
    /// Total element count.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// All tensors found across a model's safetensors files, keyed by name.
pub type TensorMap = HashMap<String, RawTensor>;

/// Load all `*.safetensors` tensors under `path` (a directory or single file).
///
/// On a directory, every `*.safetensors` file is loaded and merged; sharded
/// checkpoints therefore work without parsing the index. Duplicate tensor names
/// across shards are an error (they should never happen in valid checkpoints).
pub fn load_safetensors(path: &Path) -> Result<TensorMap> {
    let files = collect_files(path)?;
    if files.is_empty() {
        return Err(StrixError::unsupported(format!(
            "no .safetensors files found at {}",
            path.display()
        )));
    }

    let mut map = TensorMap::new();
    for file in files {
        load_one_file(&file, &mut map)?;
    }
    Ok(map)
}

fn collect_files(path: &Path) -> Result<Vec<std::path::PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(path)? {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            files.push(p);
        }
    }
    files.sort();
    Ok(files)
}

fn load_one_file(file: &Path, map: &mut TensorMap) -> Result<()> {
    let f = fs::File::open(file)?;
    // SAFETY: we only read the mapping, and it lives until the end of this fn.
    let mmap = unsafe { Mmap::map(&f)? };
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| StrixError::parse(format!("{}: {e}", file.display())))?;

    for (name, view) in st.tensors() {
        if map.contains_key(&name) {
            return Err(StrixError::invalid(format!(
                "duplicate tensor `{name}` across shards"
            )));
        }
        let shape = view.shape().to_vec();
        let data = convert_to_f32(view.dtype(), view.data(), &name)?;
        map.insert(name, RawTensor { shape, data });
    }
    Ok(())
}

/// Convert raw little-endian tensor bytes of `dtype` into `f32`.
fn convert_to_f32(dtype: Dtype, bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(StrixError::parse(format!("{name}: F32 byte length not /4")));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(StrixError::parse(format!("{name}: F16 byte length not /2")));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err(StrixError::parse(format!(
                    "{name}: BF16 byte length not /2"
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        other => Err(StrixError::unsupported(format!(
            "{name}: dtype {other:?} not supported by the CPU reference loader (want F32/F16/BF16)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_and_bf16_roundtrip_to_f32() {
        // 1.5 in f16/bf16 is exact.
        let h = f16::from_f32(1.5);
        let bytes = h.to_le_bytes();
        let got = convert_to_f32(Dtype::F16, &bytes, "t").unwrap();
        assert_eq!(got, vec![1.5]);

        let b = bf16::from_f32(-2.0);
        let got = convert_to_f32(Dtype::BF16, &b.to_le_bytes(), "t").unwrap();
        assert_eq!(got, vec![-2.0]);
    }

    #[test]
    fn f32_passthrough() {
        let vals = [0.25f32, -3.0, 7.5];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let got = convert_to_f32(Dtype::F32, &bytes, "t").unwrap();
        assert_eq!(got, vals.to_vec());
    }

    #[test]
    fn unsupported_dtype_errors() {
        assert!(convert_to_f32(Dtype::I32, &[0, 0, 0, 0], "t").is_err());
    }
}
