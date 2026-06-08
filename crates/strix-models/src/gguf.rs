//! GGUF container parser (v2/v3).
//!
//! GGUF packs everything a runtime needs into one file: a metadata key/value
//! store (hyperparameters + the full tokenizer) followed by a tensor-info table
//! and aligned tensor data. We memory-map the file and decode lazily: metadata
//! and the tensor index are parsed up front; tensor *data* is dequantized on
//! demand via [`crate::ggml_quant`].
//!
//! Reference: the GGUF spec in ggml-org/ggml (`docs/gguf.md`). Only the value
//! types we encounter in Gemma/Llama GGUFs are handled.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use strix_core::error::{Result, StrixError};

use crate::ggml_quant::{dequantize, GgmlType};

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: usize = 32;

/// A GGUF metadata value.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// Interpret as an unsigned integer, widening from any int type.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            MetaValue::U8(v) => *v as u64,
            MetaValue::U16(v) => *v as u64,
            MetaValue::U32(v) => *v as u64,
            MetaValue::U64(v) => *v,
            MetaValue::I8(v) if *v >= 0 => *v as u64,
            MetaValue::I16(v) if *v >= 0 => *v as u64,
            MetaValue::I32(v) if *v >= 0 => *v as u64,
            MetaValue::I64(v) if *v >= 0 => *v as u64,
            _ => return None,
        })
    }

    /// Interpret as `f32` (also accepts integers and `f64`).
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::F32(v) => Some(*v),
            MetaValue::F64(v) => Some(*v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }

    /// Borrow as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Borrow as an array.
    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// Where a tensor lives in the file and how it's encoded.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g. `blk.0.attn_q.weight`).
    pub name: String,
    /// Dimensions, GGUF order (fastest-varying first).
    pub dims: Vec<u64>,
    /// Element encoding.
    pub ggml_type: GgmlType,
    /// Offset from the start of the tensor-data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total element count.
    pub fn numel(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

/// A parsed, memory-mapped GGUF file.
pub struct GgufFile {
    mmap: Mmap,
    metadata: HashMap<String, MetaValue>,
    tensors: HashMap<String, TensorInfo>,
    data_offset: usize,
}

impl GgufFile {
    /// Open and parse a GGUF file's header (metadata + tensor index).
    pub fn open(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path)?;
        // SAFETY: read-only mapping kept alive for the lifetime of GgufFile.
        let mmap = unsafe { Mmap::map(&f)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let mut c = Cursor::new(&mmap);
        let magic = c.u32()?;
        if magic != MAGIC {
            return Err(StrixError::parse(format!(
                "not a GGUF file (magic 0x{magic:08x})"
            )));
        }
        let version = c.u32()?;
        if version != 2 && version != 3 {
            return Err(StrixError::unsupported(format!("GGUF version {version}")));
        }
        let tensor_count = c.u64()? as usize;
        let kv_count = c.u64()? as usize;

        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = c.string()?;
            let value = c.metavalue()?;
            metadata.insert(key, value);
        }

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let ggml_type = GgmlType::from_u32(c.u32()?)?;
            let offset = c.u64()?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    ggml_type,
                    offset,
                },
            );
        }

        // Tensor data begins at the next alignment boundary after the header.
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT as u64) as usize;
        let data_offset = align_up(c.pos(), alignment);
        if data_offset > mmap.len() {
            return Err(StrixError::parse("GGUF tensor data offset past EOF"));
        }

        Ok(GgufFile {
            mmap,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// All metadata.
    pub fn metadata(&self) -> &HashMap<String, MetaValue> {
        &self.metadata
    }

    /// Look up a metadata value.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// `general.architecture` (e.g. `"gemma3"`, `"llama"`).
    pub fn architecture(&self) -> Option<&str> {
        self.meta("general.architecture").and_then(|v| v.as_str())
    }

    /// Required `u32` metadata, supporting the `{arch}.` prefix convention.
    pub fn meta_u32(&self, key: &str) -> Result<u32> {
        self.meta(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| StrixError::invalid(format!("GGUF metadata missing u32 `{key}`")))
    }

    /// Required `f32` metadata.
    pub fn meta_f32(&self, key: &str) -> Result<f32> {
        self.meta(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| StrixError::invalid(format!("GGUF metadata missing f32 `{key}`")))
    }

    /// The tensor index.
    pub fn tensors(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }

    /// Raw bytes for a tensor.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("GGUF tensor `{name}` not found")))?;
        let start = self.data_offset + info.offset as usize;
        let nbytes = (info.numel() / info.ggml_type.block_elems()) * info.ggml_type.block_bytes();
        let end = start + nbytes;
        if end > self.mmap.len() {
            return Err(StrixError::parse(format!(
                "tensor `{name}` data [{start}..{end}] past EOF {}",
                self.mmap.len()
            )));
        }
        Ok(&self.mmap[start..end])
    }

    /// Dequantize a tensor to `f32`.
    pub fn dequant_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| StrixError::invalid(format!("GGUF tensor `{name}` not found")))?;
        let bytes = self.tensor_bytes(name)?;
        dequantize(info.ggml_type, bytes, info.numel())
    }
}

#[inline]
fn align_up(x: usize, a: usize) -> usize {
    if a == 0 {
        x
    } else {
        x.div_ceil(a) * a
    }
}

/// Little-endian byte cursor with bounds checking.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| StrixError::parse("GGUF: unexpected EOF"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Read a metadata value (type tag + payload).
    fn metavalue(&mut self) -> Result<MetaValue> {
        let ty = self.u32()?;
        self.metavalue_of(ty)
    }

    fn metavalue_of(&mut self, ty: u32) -> Result<MetaValue> {
        Ok(match ty {
            0 => MetaValue::U8(self.u8()?),
            1 => MetaValue::I8(self.u8()? as i8),
            2 => MetaValue::U16(self.u16()?),
            3 => MetaValue::I16(self.u16()? as i16),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(self.u32()? as i32),
            6 => MetaValue::F32(self.f32()?),
            7 => MetaValue::Bool(self.u8()? != 0),
            8 => MetaValue::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let count = self.u64()? as usize;
                if elem_ty == 9 {
                    return Err(StrixError::unsupported("GGUF nested arrays"));
                }
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.metavalue_of(elem_ty)?);
                }
                MetaValue::Array(items)
            }
            10 => MetaValue::U64(self.u64()?),
            11 => MetaValue::I64(self.u64()? as i64),
            12 => MetaValue::F64(self.f64()?),
            other => {
                return Err(StrixError::unsupported(format!(
                    "GGUF metadata value type {other}"
                )))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    // Build a minimal in-memory GGUF with one metadata string, one u32, and one
    // F16 tensor of 2 elements, then verify round-trip parsing + dequant.
    fn build_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };

        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&2u64.to_le_bytes()); // kv_count

        // kv: general.architecture = "test"
        push_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes()); // type STRING
        push_str(&mut b, "test");
        // kv: test.block_count = 7 (u32)
        push_str(&mut b, "test.block_count");
        b.extend_from_slice(&4u32.to_le_bytes()); // type U32
        b.extend_from_slice(&7u32.to_le_bytes());

        // tensor info: name="w", 1 dim [2], type F16(1), offset 0
        push_str(&mut b, "w");
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&2u64.to_le_bytes()); // dim0
        b.extend_from_slice(&1u32.to_le_bytes()); // ggml_type F16
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        // align to 32
        while b.len() % 32 != 0 {
            b.push(0);
        }
        // tensor data: [1.5, -0.5] as f16
        b.extend_from_slice(&f16::from_f32(1.5).to_le_bytes());
        b.extend_from_slice(&f16::from_f32(-0.5).to_le_bytes());
        b
    }

    fn parse_bytes(bytes: Vec<u8>) -> GgufFile {
        // Write to a temp file and open (GgufFile::open expects a path).
        let dir = std::env::temp_dir();
        let path = dir.join(format!("strix_test_{}.gguf", bytes.len()));
        std::fs::write(&path, &bytes).unwrap();
        let g = GgufFile::open(&path).unwrap();
        std::fs::remove_file(&path).ok();
        g
    }

    #[test]
    fn parses_metadata_and_tensor() {
        let g = parse_bytes(build_gguf());
        assert_eq!(g.architecture(), Some("test"));
        assert_eq!(g.meta_u32("test.block_count").unwrap(), 7);
        let info = &g.tensors()["w"];
        assert_eq!(info.dims, vec![2]);
        assert_eq!(info.ggml_type, GgmlType::F16);
        let data = g.dequant_tensor("w").unwrap();
        assert_eq!(data, vec![1.5, -0.5]);
    }

    #[test]
    fn bad_magic_errors() {
        let dir = std::env::temp_dir();
        let path = dir.join("strix_test_badmagic.gguf");
        std::fs::write(&path, vec![0u8; 64]).unwrap();
        let r = GgufFile::open(&path);
        std::fs::remove_file(&path).ok();
        assert!(r.is_err());
    }
}
