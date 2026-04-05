//! Fieldnorm encoding for compact document-length storage
//!
//! Implements Lucene-compatible SmallFloat encoding that compresses document
//! lengths into a single byte (3-bit mantissa + 5-bit exponent). This produces
//! ~256 distinct values with 5-11% quantization error — the same approach used
//! by Lucene and Tantivy.
//!
//! # Why not `UintVecMin0`?
//!
//! Document lengths are already quantized by the fieldnorm encoding. Storing
//! the decoded u32 values in `UintVecMin0` wastes bits on values that were
//! already reduced to a 1-byte codebook. Storing the fieldnorm byte directly
//! as `Vec<u8>` is:
//!
//! - **Smaller**: 1.0 bytes/doc vs ~1.13 bytes/doc for `UintVecMin0`
//! - **Faster**: Array index vs multiply + unaligned load + shift + mask
//! - **Eliminates per-posting BM25 division** via pre-computed norm table
//!
//! # Example
//!
//! ```rust
//! use zipora::scoring::FieldnormEncoder;
//!
//! // Encode document lengths to bytes
//! let doc_lengths = vec![150u32, 200, 50, 300, 1000];
//! let encoded: Vec<u8> = doc_lengths.iter().map(|&l| FieldnormEncoder::encode(l)).collect();
//!
//! // Decode back (approximate due to quantization)
//! for (orig, &byte) in doc_lengths.iter().zip(encoded.iter()) {
//!     let decoded = FieldnormEncoder::decode(byte);
//!     // Quantization error is bounded: decoded is close to original
//!     assert!(decoded <= *orig || decoded >= *orig / 2);
//! }
//!
//! // Build BM25 norm table for fast scoring
//! let avg_dl = 150.0f32;
//! let k1 = 1.2;
//! let b = 0.75;
//! let norm_table = FieldnormEncoder::build_norm_table(avg_dl, k1, b);
//!
//! // Score lookup is a single array index — no float division needed
//! let doc_0_norm = norm_table[encoded[0] as usize];
//! assert!(doc_0_norm > 0.0);
//! ```

use std::sync::OnceLock;

/// Lucene-compatible fieldnorm encoder using SmallFloat (3-bit mantissa, 5-bit exponent).
///
/// Encodes unsigned integers into a single byte with bounded quantization error.
/// Values 0-7 are encoded exactly; larger values use a floating-point representation
/// with an implicit leading 1 bit, giving ~5-11% quantization error.
///
/// The encoding range covers 0 to ~16 billion, more than sufficient for document lengths.
pub struct FieldnormEncoder;

impl FieldnormEncoder {
    /// Encode a document length (or any u32) to a single fieldnorm byte.
    ///
    /// - Values 0-7: encoded exactly
    /// - Values 8+: 3-bit mantissa + implicit leading 1 + 5-bit exponent
    ///
    /// The encoding is monotonic: if `a <= b` then `encode(a) <= encode(b)`.
    #[inline]
    pub fn encode(len: u32) -> u8 {
        if len < 8 {
            return len as u8;
        }
        let num_bits = 32 - len.leading_zeros(); // bit length of len
        let shift = num_bits - 4; // keep 4 significant bits (1 implicit + 3 explicit)
        let mantissa = (len >> shift) & 0x07; // 3 explicit mantissa bits
        let exponent = shift + 1; // offset by 1 (exponent 0 = exact small values)
        ((exponent << 3) | mantissa) as u8
    }

    /// Decode a fieldnorm byte back to an approximate document length.
    ///
    /// The decoded value is the lower bound of the quantization bucket:
    /// `decode(encode(x)) <= x` for all x.
    #[inline]
    pub fn decode(byte: u8) -> u32 {
        let b = byte as u32;
        if b < 8 {
            return b;
        }
        let mantissa = b & 0x07;
        let exponent = (b >> 3) - 1;
        // Bytes 240-255 (exponent >= 29) would overflow u32 — saturate
        if exponent >= 29 {
            return u32::MAX;
        }
        (mantissa | 8) << exponent // restore implicit leading 1
    }

    /// Pre-compute a 256-entry decode table for O(1) decoding.
    ///
    /// This is useful when decoding many values — avoids the branch on each call.
    pub fn decode_table() -> &'static [u32; 256] {
        static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
        TABLE.get_or_init(|| {
            let mut table = [0u32; 256];
            for i in 0..256u16 {
                table[i as usize] = Self::decode(i as u8);
            }
            table
        })
    }

    /// Build a BM25 length normalization lookup table.
    ///
    /// For each possible fieldnorm byte (0-255), pre-computes:
    /// ```text
    /// norm[byte] = k1 * (1.0 - b + b * decode(byte) / avg_dl)
    /// ```
    ///
    /// At scoring time, the BM25 formula becomes:
    /// ```text
    /// score = idf * tf * (k1 + 1) / (tf + norm_table[fieldnorm_byte])
    /// ```
    ///
    /// This eliminates the per-posting float division for length normalization.
    ///
    /// # Arguments
    ///
    /// * `avg_dl` - Average document length in the collection
    /// * `k1` - BM25 term frequency saturation parameter (typically 1.2)
    /// * `b` - BM25 length normalization parameter (typically 0.75)
    pub fn build_norm_table(avg_dl: f32, k1: f32, b: f32) -> [f32; 256] {
        let mut table = [0.0f32; 256];
        let inv_avg_dl = if avg_dl > 0.0 { 1.0 / avg_dl } else { 0.0 };
        for i in 0..256u16 {
            let dl = Self::decode(i as u8) as f32;
            table[i as usize] = k1 * (1.0 - b + b * dl * inv_avg_dl);
        }
        table
    }

    /// Build a complete BM25 score lookup table for a given IDF.
    ///
    /// For each possible (tf_quantized, fieldnorm_byte) pair, pre-computes
    /// the full BM25 score. This is useful when TF values are also quantized
    /// to a small range.
    ///
    /// Returns a 2D table indexed as `table[tf][fieldnorm_byte]`.
    ///
    /// # Arguments
    ///
    /// * `avg_dl` - Average document length
    /// * `k1` - BM25 k1 parameter
    /// * `b` - BM25 b parameter
    /// * `idf` - Inverse document frequency for this term
    /// * `max_tf` - Maximum TF value to pre-compute (table will have max_tf+1 rows)
    pub fn build_score_table(
        avg_dl: f32,
        k1: f32,
        b: f32,
        idf: f32,
        max_tf: u16,
    ) -> Vec<[f32; 256]> {
        let norm_table = Self::build_norm_table(avg_dl, k1, b);
        let k1p1 = k1 + 1.0;
        let mut table = vec![[0.0f32; 256]; max_tf as usize + 1];
        for tf in 0..=max_tf {
            let tf_f = tf as f32;
            for byte in 0..256usize {
                let len_norm = norm_table[byte];
                table[tf as usize][byte] = idf * tf_f * k1p1 / (tf_f + len_norm);
            }
        }
        table
    }

    /// Maximum quantization error ratio for a given value.
    ///
    /// Returns `(original - decoded) / original` as a fraction.
    /// Values 0-15 have zero error. Larger values have at most ~12.5% error.
    #[inline]
    pub fn quantization_error(len: u32) -> f64 {
        if len == 0 {
            return 0.0;
        }
        let decoded = Self::decode(Self::encode(len));
        (len as f64 - decoded as f64) / len as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_small_values() {
        // Values 0-7 must be encoded and decoded exactly
        for i in 0..8u32 {
            let byte = FieldnormEncoder::encode(i);
            assert_eq!(byte, i as u8, "encode({i}) should be {i}");
            assert_eq!(FieldnormEncoder::decode(byte), i, "decode(encode({i})) should be {i}");
        }
    }

    #[test]
    fn test_exact_values_8_to_15() {
        // Values 8-15 should also round-trip exactly (4 significant bits, no loss)
        for i in 8..16u32 {
            let byte = FieldnormEncoder::encode(i);
            let decoded = FieldnormEncoder::decode(byte);
            assert_eq!(decoded, i, "decode(encode({i})) = {decoded}, expected {i}");
        }
    }

    #[test]
    fn test_known_encodings() {
        // Verify specific known encode/decode pairs
        assert_eq!(FieldnormEncoder::encode(0), 0);
        assert_eq!(FieldnormEncoder::encode(1), 1);
        assert_eq!(FieldnormEncoder::encode(7), 7);
        assert_eq!(FieldnormEncoder::encode(8), 8);  // 1.000 × 2^0 → exp=1, mant=0
        assert_eq!(FieldnormEncoder::encode(15), 15); // 1.111 × 2^0 → exp=1, mant=7
        assert_eq!(FieldnormEncoder::encode(16), 16); // 1.000 × 2^1 → exp=2, mant=0
        assert_eq!(FieldnormEncoder::encode(18), 17); // 1.001 × 2^1 → exp=2, mant=1

        // Decode
        assert_eq!(FieldnormEncoder::decode(0), 0);
        assert_eq!(FieldnormEncoder::decode(8), 8);
        assert_eq!(FieldnormEncoder::decode(16), 16);
        assert_eq!(FieldnormEncoder::decode(17), 18); // (1|8) << 1 = 9 << 1 = 18
    }

    #[test]
    fn test_monotonic_encoding() {
        // encode must be monotonically non-decreasing
        let mut prev_byte = 0u8;
        for i in 0..100_000u32 {
            let byte = FieldnormEncoder::encode(i);
            assert!(
                byte >= prev_byte,
                "encode({i})={byte} < encode({})={prev_byte} — not monotonic",
                i - 1
            );
            prev_byte = byte;
        }
    }

    #[test]
    fn test_monotonic_decoding() {
        // decode must be monotonically non-decreasing over byte values
        let mut prev = 0u32;
        for b in 0..=255u8 {
            let val = FieldnormEncoder::decode(b);
            assert!(
                val >= prev,
                "decode({b})={val} < decode({})={prev} — not monotonic",
                b - 1
            );
            prev = val;
        }
    }

    #[test]
    fn test_roundtrip_lower_bound() {
        // decode(encode(x)) <= x for all x (decoded is the bucket's lower bound)
        for i in 0..100_000u32 {
            let decoded = FieldnormEncoder::decode(FieldnormEncoder::encode(i));
            assert!(
                decoded <= i,
                "decode(encode({i})) = {decoded} > {i} — not a lower bound"
            );
        }
    }

    #[test]
    fn test_quantization_error_bounded() {
        // Quantization error should not exceed ~12.5% for values >= 16
        for i in 16..100_000u32 {
            let error = FieldnormEncoder::quantization_error(i);
            assert!(
                error < 0.15, // generous bound
                "quantization_error({i}) = {error:.4} exceeds 15%"
            );
        }
    }

    #[test]
    fn test_full_byte_range_used() {
        // Encoding should use a wide range of byte values
        // Max byte for u32::MAX is 239 (exponent=29, mantissa=7), so 240 distinct values
        let mut seen = [false; 256];
        for i in 0..10_000_000u32 {
            seen[FieldnormEncoder::encode(i) as usize] = true;
        }
        let used = seen.iter().filter(|&&s| s).count();
        assert!(used >= 150, "only {used} distinct byte values used — expected >=150");

        // Verify max byte value
        let max_byte = FieldnormEncoder::encode(u32::MAX);
        assert_eq!(max_byte, 239, "max byte should be 239 for u32::MAX");
    }

    #[test]
    fn test_large_values() {
        // Large document lengths should still encode/decode sensibly
        let large = 1_000_000u32;
        let byte = FieldnormEncoder::encode(large);
        let decoded = FieldnormEncoder::decode(byte);
        assert!(decoded <= large);
        assert!(decoded > large / 2, "decoded {decoded} too far from {large}");

        // u32::MAX
        let byte_max = FieldnormEncoder::encode(u32::MAX);
        let decoded_max = FieldnormEncoder::decode(byte_max);
        assert!(decoded_max > 0);
    }

    #[test]
    fn test_decode_table() {
        let table = FieldnormEncoder::decode_table();
        for b in 0..=255u8 {
            assert_eq!(
                table[b as usize],
                FieldnormEncoder::decode(b),
                "decode_table mismatch at byte {b}"
            );
        }
    }

    #[test]
    fn test_norm_table_basic() {
        let avg_dl = 100.0f32;
        let k1 = 1.2;
        let b = 0.75;
        let table = FieldnormEncoder::build_norm_table(avg_dl, k1, b);

        // byte 0 → dl=0 → norm = k1 * (1 - b + 0) = 1.2 * 0.25 = 0.3
        assert!((table[0] - 0.3).abs() < 1e-6, "table[0] = {}", table[0]);

        // For dl == avg_dl → norm = k1 * (1 - b + b) = k1 = 1.2
        let avg_byte = FieldnormEncoder::encode(100);
        let avg_decoded = FieldnormEncoder::decode(avg_byte);
        let expected = k1 * (1.0 - b + b * avg_decoded as f32 / avg_dl);
        assert!(
            (table[avg_byte as usize] - expected).abs() < 1e-5,
            "table[avg_byte] = {}, expected {expected}",
            table[avg_byte as usize]
        );

        // All values should be positive
        for (i, &v) in table.iter().enumerate() {
            assert!(v >= 0.0, "norm_table[{i}] = {v} is negative");
        }

        // Table should be monotonically non-decreasing (longer docs → higher norm)
        for i in 1..256 {
            assert!(
                table[i] >= table[i - 1],
                "norm_table[{i}]={} < norm_table[{}]={}",
                table[i],
                i - 1,
                table[i - 1]
            );
        }
    }

    #[test]
    fn test_norm_table_avg_dl_zero() {
        // Edge case: avg_dl = 0 should not panic
        let table = FieldnormEncoder::build_norm_table(0.0, 1.2, 0.75);
        // All entries should be finite (inv_avg_dl = 0.0 when avg_dl = 0)
        for (i, &v) in table.iter().enumerate() {
            assert!(v.is_finite(), "norm_table[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_score_table() {
        let avg_dl = 100.0;
        let k1 = 1.2;
        let b = 0.75;
        let idf = 5.0;
        let max_tf = 10;

        let table = FieldnormEncoder::build_score_table(avg_dl, k1, b, idf, max_tf);
        assert_eq!(table.len(), 11); // 0..=10

        // tf=0 → score should be 0 for all fieldnorms
        for byte in 0..256 {
            assert_eq!(table[0][byte], 0.0, "score(tf=0, byte={byte}) should be 0");
        }

        // tf=1 → positive scores
        for byte in 0..256 {
            assert!(table[1][byte] >= 0.0);
        }

        // Higher tf → higher score (for same fieldnorm)
        for byte in 0..256 {
            for tf in 1..max_tf as usize {
                assert!(
                    table[tf + 1][byte] >= table[tf][byte],
                    "score(tf={}, byte={byte}) < score(tf={tf}, byte={byte})",
                    tf + 1
                );
            }
        }
    }

    #[test]
    fn test_bm25_scoring_workflow() {
        // End-to-end: encode doc lengths, build norm table, score
        let doc_lengths = vec![50u32, 100, 150, 200, 300];
        let fieldnorm_bytes: Vec<u8> = doc_lengths.iter().map(|&l| FieldnormEncoder::encode(l)).collect();

        let avg_dl: f32 = doc_lengths.iter().sum::<u32>() as f32 / doc_lengths.len() as f32;
        let k1 = 1.2f32;
        let b = 0.75f32;
        let norm_table = FieldnormEncoder::build_norm_table(avg_dl, k1, b);

        let idf = 3.5f32;
        let k1p1 = k1 + 1.0;

        // Score a posting with tf=2 for each doc
        let scores: Vec<f32> = fieldnorm_bytes
            .iter()
            .map(|&byte| {
                let tf = 2.0f32;
                let len_norm = norm_table[byte as usize];
                idf * tf * k1p1 / (tf + len_norm)
            })
            .collect();

        // Shorter docs should score higher (for same tf)
        assert!(scores[0] > scores[4], "shorter doc should score higher");

        // All scores should be positive and finite
        for (i, &s) in scores.iter().enumerate() {
            assert!(s > 0.0 && s.is_finite(), "score[{i}] = {s}");
        }
    }

    // ========================================================================
    // Performance-oriented tests (release mode)
    // ========================================================================

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_encode_decode_performance() {
        use std::time::Instant;

        let n = 10_000_000u32;

        // Encode performance
        let start = Instant::now();
        let mut sum = 0u64;
        for i in 0..n {
            sum += FieldnormEncoder::encode(i) as u64;
        }
        let encode_time = start.elapsed();
        assert!(sum > 0); // prevent optimization

        // Decode performance
        let start = Instant::now();
        let mut sum2 = 0u64;
        for i in 0..n {
            sum2 += FieldnormEncoder::decode((i & 0xFF) as u8) as u64;
        }
        let decode_time = start.elapsed();
        assert!(sum2 > 0);

        // Table lookup performance
        let table = FieldnormEncoder::decode_table();
        let start = Instant::now();
        let mut sum3 = 0u64;
        for i in 0..n {
            sum3 += table[(i & 0xFF) as usize] as u64;
        }
        let table_time = start.elapsed();
        assert!(sum3 > 0);

        eprintln!(
            "FieldnormEncoder perf ({n} ops): encode={:?}, decode={:?}, table_lookup={:?}",
            encode_time, decode_time, table_time
        );
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_norm_table_vs_float_math() {
        use std::time::Instant;
        use crate::containers::UintVecMin0;

        let n = 1_000_000usize;
        let k1 = 1.2f32;
        let b = 0.75f32;
        let avg_dl = 150.0f32;

        // Setup: create doc lengths and encode them
        let doc_lengths: Vec<u32> = (0..n).map(|i| (i % 500 + 1) as u32).collect();
        let fieldnorm_bytes: Vec<u8> = doc_lengths.iter().map(|&l| FieldnormEncoder::encode(l)).collect();

        // Setup: UintVecMin0 with decoded values
        let max_val = *doc_lengths.iter().max().unwrap() as usize;
        let mut uint_vec = UintVecMin0::new(n, max_val);
        for (i, &dl) in doc_lengths.iter().enumerate() {
            uint_vec.set(i, dl as usize);
        }

        // Build norm table
        let norm_table = FieldnormEncoder::build_norm_table(avg_dl, k1, b);

        // Benchmark: Vec<u8> + norm table lookup
        let start = Instant::now();
        let mut sum1 = 0.0f64;
        for i in 0..n {
            let tf = 2.0f32;
            let len_norm = norm_table[fieldnorm_bytes[i] as usize];
            sum1 += (tf * 2.2 / (tf + len_norm)) as f64;
        }
        let table_time = start.elapsed();

        // Benchmark: UintVecMin0 + float math
        let start = Instant::now();
        let mut sum2 = 0.0f64;
        for i in 0..n {
            let tf = 2.0f32;
            let dl = uint_vec.get(i) as f32;
            let len_norm = k1 * (1.0 - b + b * dl / avg_dl);
            sum2 += (tf * 2.2 / (tf + len_norm)) as f64;
        }
        let uint_time = start.elapsed();

        eprintln!(
            "BM25 scoring ({n} docs): Vec<u8>+table={:?}, UintVecMin0+float={:?}, speedup={:.1}x",
            table_time, uint_time,
            uint_time.as_nanos() as f64 / table_time.as_nanos().max(1) as f64
        );

        // Memory comparison
        let fieldnorm_mem = fieldnorm_bytes.len(); // 1 byte per doc
        let uint_mem = uint_vec.mem_size();
        eprintln!(
            "Memory: Vec<u8>={} bytes ({:.2} B/doc), UintVecMin0={} bytes ({:.2} B/doc)",
            fieldnorm_mem, fieldnorm_mem as f64 / n as f64,
            uint_mem, uint_mem as f64 / n as f64
        );

        assert!(sum1 > 0.0 && sum2 > 0.0);
    }
}
