//! GGML block-quantization formats → `f32`.
//!
//! GGUF weights are stored in GGML block formats. This module decodes the ones
//! we need, matching llama.cpp's `ggml-quants.c` bit-for-bit:
//!
//! - **F32 / F16**: trivial (raw).
//! - **Q8_0**: 32 values/block = `f16` scale + 32×`int8`. `y = d * q`.
//! - **Q4_0**: 32 values/block = `f16` scale + 16 packed nibbles.
//!   `y = d * (q - 8)`. **This is the Gemma QAT format.**
//! - **Q4_K**: 256-value superblock with 8 sub-block 6-bit scales/mins.
//!
//! Everything is pure and unit-tested against hand-built blocks. The block
//! layouts are a stable on-disk ABI, so these constants must not drift.

use half::f16;
use strix_core::error::{Result, StrixError};

/// GGML tensor element type, tagged with GGUF type ids.
///
/// We recognize the full standard set (so any standard-quant GGUF *parses* and
/// can be listed), but only a subset has a `dequantize` implementation; the rest
/// error at dequant time, not parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlType {
    /// Map a GGUF `ggml_type` id to our enum.
    pub fn from_u32(id: u32) -> Result<Self> {
        Ok(match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            other => {
                return Err(StrixError::unsupported(format!(
                    "GGML type id {other} unrecognized (IQ/TQ types not handled)"
                )))
            }
        })
    }

    /// Short name for display.
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q4_1 => "Q4_1",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q8_1 => "Q8_1",
            GgmlType::Q2K => "Q2_K",
            GgmlType::Q3K => "Q3_K",
            GgmlType::Q4K => "Q4_K",
            GgmlType::Q5K => "Q5_K",
            GgmlType::Q6K => "Q6_K",
            GgmlType::Q8K => "Q8_K",
        }
    }

    /// Number of elements per block.
    pub fn block_elems(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 => 1,
            GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1 => 32,
            GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q8K => 256,
        }
    }

    /// Bytes per block (on-disk GGML layout).
    pub fn block_bytes(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 18, // d(f16) + 16
            GgmlType::Q4_1 => 20, // d,m(f16) + 16
            GgmlType::Q5_0 => 22, // d(f16) + qh(4) + 16
            GgmlType::Q5_1 => 24, // d,m(f16) + qh(4) + 16
            GgmlType::Q8_0 => 34, // d(f16) + 32
            GgmlType::Q8_1 => 36, // d,s(f16) + 32
            GgmlType::Q2K => 84,
            GgmlType::Q3K => 110,
            GgmlType::Q4K => 144, // d,dmin(f16) + 12 + 128
            GgmlType::Q5K => 176,
            GgmlType::Q6K => 210, // 128 ql + 64 qh + 16 sc + d(f16)
            GgmlType::Q8K => 292,
        }
    }
}

/// Decode `n_elements` of `ty` from `bytes` into a freshly allocated buffer.
pub fn dequantize(ty: GgmlType, bytes: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; n_elements];
    dequantize_into(ty, bytes, &mut out)?;
    Ok(out)
}

/// Decode into a caller-provided buffer (`out.len()` elements).
///
/// This is the allocation-free path used by the quantized matmul: dequantize a
/// single weight row into a reusable scratch buffer per dot product, so a large
/// quantized tensor is never fully materialized as `f32`.
pub fn dequantize_into(ty: GgmlType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let n_elements = out.len();
    let be = ty.block_elems();
    if n_elements % be != 0 {
        return Err(StrixError::invalid(format!(
            "{ty:?}: {n_elements} elements not a multiple of block size {be}"
        )));
    }
    let nblocks = n_elements / be;
    let need = nblocks * ty.block_bytes();
    if bytes.len() < need {
        return Err(StrixError::invalid(format!(
            "{ty:?}: need {need} bytes for {n_elements} elements, have {}",
            bytes.len()
        )));
    }
    let bytes = &bytes[..need];
    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        GgmlType::Q8_0 => dequant_q8_0(bytes, out),
        GgmlType::Q4_0 => dequant_q4_0(bytes, out),
        GgmlType::Q4_1 => dequant_q4_1(bytes, out),
        GgmlType::Q5_0 => dequant_q5_0(bytes, out),
        GgmlType::Q5_1 => dequant_q5_1(bytes, out),
        GgmlType::Q4K => dequant_q4_k(bytes, out),
        GgmlType::Q6K => dequant_q6_k(bytes, out),
        other => {
            return Err(StrixError::unsupported(format!(
                "dequant for {} not implemented yet",
                other.name()
            )))
        }
    }
    Ok(())
}

#[inline]
fn read_f16(b: &[u8]) -> f32 {
    f16::from_le_bytes([b[0], b[1]]).to_f32()
}

/// Quantize `f32` values to GGML Q4_0 blocks (32 vals → [f16 d][16 nibble bytes]),
/// matching `ggml_quantize_row_q4_0`. `n_elements` must be a multiple of 32.
/// Used to build a lighter Q4_0 lm_head from a Q6_K tied embedding.
pub fn quantize_q4_0(x: &[f32]) -> Vec<u8> {
    const QK: usize = 32;
    assert!(x.len() % QK == 0, "quantize_q4_0: len not multiple of 32");
    let mut out = Vec::with_capacity((x.len() / QK) * 18);
    for blk in x.chunks_exact(QK) {
        // amax = element with the largest absolute value (signed), per ggml.
        let mut amax = 0.0f32;
        let mut vmax = 0.0f32;
        for &v in blk {
            if v.abs() > amax {
                amax = v.abs();
                vmax = v;
            }
        }
        let d = vmax / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        // nibble j packs x[j] (low) and x[j+16] (high).
        for j in 0..16 {
            let q0 = (((blk[j] * id) + 8.5) as i32).clamp(0, 15) as u8;
            let q1 = (((blk[j + 16] * id) + 8.5) as i32).clamp(0, 15) as u8;
            out.push(q0 | (q1 << 4));
        }
    }
    out
}

/// Q8_0: [f16 d][32×i8 q]; y = d * q.
fn dequant_q8_0(bytes: &[u8], out: &mut [f32]) {
    for (blk, ob) in bytes.chunks_exact(34).zip(out.chunks_mut(32)) {
        let d = read_f16(&blk[0..2]);
        for (o, &q) in ob.iter_mut().zip(&blk[2..34]) {
            *o = d * (q as i8) as f32;
        }
    }
}

/// Q4_0: [f16 d][16 bytes of packed nibbles]; y = d * (nibble - 8).
/// Nibble layout: byte j holds value j (low) and value j+16 (high).
fn dequant_q4_0(bytes: &[u8], out: &mut [f32]) {
    for (blk, ob) in bytes.chunks_exact(18).zip(out.chunks_mut(32)) {
        let d = read_f16(&blk[0..2]);
        let qs = &blk[2..18];
        for j in 0..16 {
            let lo = (qs[j] & 0x0F) as i32 - 8;
            let hi = (qs[j] >> 4) as i32 - 8;
            ob[j] = d * lo as f32;
            ob[j + 16] = d * hi as f32;
        }
    }
}

/// Q4_1: [f16 d][f16 m][16 packed nibbles]; y = d * nibble + m.
fn dequant_q4_1(bytes: &[u8], out: &mut [f32]) {
    for (blk, ob) in bytes.chunks_exact(20).zip(out.chunks_mut(32)) {
        let d = read_f16(&blk[0..2]);
        let m = read_f16(&blk[2..4]);
        let qs = &blk[4..20];
        for j in 0..16 {
            ob[j] = d * (qs[j] & 0x0F) as f32 + m;
            ob[j + 16] = d * (qs[j] >> 4) as f32 + m;
        }
    }
}

/// Q5_0: [f16 d][u32 qh][16 nibbles]; 5th bit per value from qh; y = d*(q-16).
fn dequant_q5_0(bytes: &[u8], out: &mut [f32]) {
    for (blk, ob) in bytes.chunks_exact(22).zip(out.chunks_mut(32)) {
        let d = read_f16(&blk[0..2]);
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((qs[j] & 0x0F) | xh0) as i32 - 16;
            let x1 = ((qs[j] >> 4) | xh1) as i32 - 16;
            ob[j] = x0 as f32 * d;
            ob[j + 16] = x1 as f32 * d;
        }
    }
}

/// Q5_1: [f16 d][f16 m][u32 qh][16 nibbles]; y = d*q + m, q 5-bit (0..31).
fn dequant_q5_1(bytes: &[u8], out: &mut [f32]) {
    for (blk, ob) in bytes.chunks_exact(24).zip(out.chunks_mut(32)) {
        let d = read_f16(&blk[0..2]);
        let m = read_f16(&blk[2..4]);
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        let qs = &blk[8..24];
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((qs[j] & 0x0F) | xh0) as f32;
            let x1 = ((qs[j] >> 4) | xh1) as f32;
            ob[j] = x0 * d + m;
            ob[j + 16] = x1 * d + m;
        }
    }
}

/// Unpack a 6-bit scale and 6-bit min for sub-block `j` from the 12 packed
/// `scales` bytes (llama.cpp `get_scale_min_k4`).
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Q4_K superblock (256 values): [f16 d][f16 dmin][12 scale bytes][128 nibble bytes].
/// 8 sub-blocks of 32, each with 6-bit scale/min: `y = d*sc*q - dmin*m`.
fn dequant_q4_k(bytes: &[u8], out: &mut [f32]) {
    for (blk, ob) in bytes.chunks_exact(144).zip(out.chunks_mut(256)) {
        let d = read_f16(&blk[0..2]);
        let dmin = read_f16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];

        let mut is = 0usize;
        let mut w = 0usize; // write cursor within the 256-value block
                            // Process 256 values in 4 chunks of 64; each chunk uses qs[off..off+32]
                            // for its low nibbles (first 32) and high nibbles (next 32).
        for chunk in 0..4 {
            let q = &qs[chunk * 32..chunk * 32 + 32];
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;
            for &b in q {
                ob[w] = d1 * (b & 0x0F) as f32 - min1;
                w += 1;
            }
            for &b in q {
                ob[w] = d2 * (b >> 4) as f32 - min2;
                w += 1;
            }
            is += 2;
        }
    }
}

/// Q6_K superblock (256 values): [128 ql][64 qh][16 i8 scales][f16 d].
/// 6-bit quants (4 low bits in `ql`, 2 high bits in `qh`), 16 int8 sub-scales.
/// `y = d * scale * (q - 32)`. Matches llama.cpp `dequantize_row_q6_K`.
fn dequant_q6_k(bytes: &[u8], out: &mut [f32]) {
    for (blk, ob) in bytes.chunks_exact(210).zip(out.chunks_mut(256)) {
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208]; // 16 × int8
        let d = read_f16(&blk[208..210]);

        // Two halves of 128 values.
        for half in 0..2 {
            let ql = &ql[half * 64..half * 64 + 64];
            let qh = &qh[half * 32..half * 32 + 32];
            let sc = &sc[half * 8..half * 8 + 8];
            let ybase = half * 128;
            for l in 0..32 {
                let is = l / 16; // 0 or 1
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                ob[ybase + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                ob[ybase + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                ob[ybase + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                ob[ybase + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk_q8_0(d: f32, qs: [i8; 32]) -> Vec<u8> {
        let mut b = f16::from_f32(d).to_le_bytes().to_vec();
        b.extend(qs.iter().map(|&x| x as u8));
        b
    }

    #[test]
    fn q8_0_scales_each_int() {
        let mut qs = [0i8; 32];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = i as i8 - 4;
        }
        let bytes = blk_q8_0(0.5, qs);
        let y = dequantize(GgmlType::Q8_0, &bytes, 32).unwrap();
        for (i, &val) in y.iter().enumerate() {
            assert!((val - 0.5 * (i as f32 - 4.0)).abs() < 1e-3, "{i}: {val}");
        }
    }

    #[test]
    fn q4_0_nibble_layout_and_bias() {
        // d=2.0; byte 0 = 0x80 => low nibble 0 -> (0-8)*2=-16; high nibble 8 -> 0.
        // byte 1 = 0xF1 => low 1 -> (1-8)*2=-14; high 15 -> (15-8)*2=14.
        let mut qs = [0u8; 16];
        qs[0] = 0x80;
        qs[1] = 0xF1;
        let mut bytes = f16::from_f32(2.0).to_le_bytes().to_vec();
        bytes.extend_from_slice(&qs);
        let y = dequantize(GgmlType::Q4_0, &bytes, 32).unwrap();
        assert_eq!(y[0], -16.0); // byte0 low
        assert_eq!(y[16], 0.0); // byte0 high
        assert_eq!(y[1], -14.0); // byte1 low
        assert_eq!(y[17], 14.0); // byte1 high
    }

    #[test]
    fn q4_k_with_zero_min_decodes_to_scaled_nibbles() {
        // dmin=0 kills the min term. d=1.0. Set scales so sub-block 0 scale=1.
        // Then first 32 outputs = 1*1*(low nibble) - 0 = nibble values.
        let mut blk = Vec::new();
        blk.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // d
        blk.extend_from_slice(&f16::from_f32(0.0).to_le_bytes()); // dmin
        let mut scales = [0u8; 12];
        scales[0] = 1; // j=0 (<4): sc = scales[0]&63 = 1
        blk.extend_from_slice(&scales);
        let mut qs = [0u8; 128];
        // First 32 low-nibbles: set qs[0..32] low nibble to l%16.
        for (l, q) in qs.iter_mut().enumerate().take(32) {
            *q = (l % 16) as u8; // low nibble = l%16, high nibble = 0
        }
        blk.extend_from_slice(&qs);
        assert_eq!(blk.len(), 144);

        let y = dequantize(GgmlType::Q4K, &blk, 256).unwrap();
        for (l, &val) in y.iter().take(32).enumerate() {
            assert!((val - (l % 16) as f32).abs() < 1e-4, "{l}: {val}");
        }
        assert_eq!(y.len(), 256);
    }

    #[test]
    fn q6_k_decodes_known_values() {
        // d=1, all scales=1. q = low4 | (high2<<4); y = q - 32.
        let mut blk = vec![0u8; 210];
        blk[0] = 0x05; // ql[0] low nibble = 5
        blk[128] = 0x01; // qh[0]: (>>0)&3 = 1 -> contributes to q1's value 0
        for s in blk.iter_mut().skip(192).take(16) {
            *s = 1; // all sub-scales = 1
        }
        blk[208..210].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());

        let y = dequantize(GgmlType::Q6K, &blk, 256).unwrap();
        // q1 for l=0: (5 | (1<<4)) - 32 = 21 - 32 = -11
        assert_eq!(y[0], -11.0);
        // everything else in the block derives from zero quants: 0 - 32 = -32
        assert_eq!(y[1], -32.0);
        assert_eq!(y[32], -32.0);
        assert_eq!(y.len(), 256);
    }

    #[test]
    fn unimplemented_quant_errors_at_dequant_not_parse() {
        // Q5_K is recognized (block sizes known) but not yet dequantized.
        assert_eq!(GgmlType::from_u32(13).unwrap(), GgmlType::Q5K);
        assert!(dequantize(GgmlType::Q5K, &[0u8; 176], 256).is_err());
    }

    #[test]
    fn f16_and_f32_passthrough() {
        let f32b: Vec<u8> = [1.0f32, -2.5, 3.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            dequantize(GgmlType::F32, &f32b, 3).unwrap(),
            vec![1.0, -2.5, 3.25]
        );
        let f16b: Vec<u8> = [f16::from_f32(1.5), f16::from_f32(-0.25)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            dequantize(GgmlType::F16, &f16b, 2).unwrap(),
            vec![1.5, -0.25]
        );
    }

    #[test]
    fn rejects_bad_sizes() {
        assert!(dequantize(GgmlType::Q8_0, &[0u8; 34], 16).is_err()); // 16 not mult of 32
        assert!(dequantize(GgmlType::Q4_0, &[0u8; 4], 32).is_err()); // too few bytes
    }

    #[test]
    fn type_ids_map() {
        assert_eq!(GgmlType::from_u32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(2).unwrap(), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_u32(8).unwrap(), GgmlType::Q8_0);
        assert_eq!(GgmlType::from_u32(12).unwrap(), GgmlType::Q4K);
        assert!(GgmlType::from_u32(99).is_err());
    }
}
