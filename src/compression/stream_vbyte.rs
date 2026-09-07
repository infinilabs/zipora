//! Stream VByte — variable-byte integer encoding with SIMD decoding.
//!
//! Encodes sorted u32 sequences using delta + variable-byte coding.
//! Control bytes are separated from data bytes, following the Stream VByte
//! layout (Lemire et al.). Decoding uses the SSSE3 shuffle-table fast path
//! (one `pshufb` expands a full 4-value group; 256-entry precomputed mask
//! table) with a scalar fallback for stream tails and non-x86 targets.
//! Encoding is scalar.
//!
//! # Format
//!
//! For each group of 4 integers:
//! - 1 control byte: 2 bits per integer (0=1byte, 1=2bytes, 2=3bytes, 3=4bytes)
//! - Data bytes: packed sequentially
//!
//! # Examples
//!
//! ```rust
//! use zipora::compression::stream_vbyte::StreamVByte;
//!
//! let values = vec![1, 5, 100, 300, 1000, 70000];
//! let encoded = StreamVByte::encode_deltas(&values);
//!
//! let decoded = StreamVByte::decode_deltas(&encoded, values.len());
//! assert_eq!(decoded, values);
//! ```

/// Stream VByte encoder/decoder.
pub struct StreamVByte;

/// Per-control-byte pshufb masks: expand a group's 1-4 byte packed values
/// into four 32-bit lanes. 0xFF (high bit set) zero-fills the lane's
/// unused bytes.
#[cfg(target_arch = "x86_64")]
const SHUFFLE_TABLE: [[u8; 16]; 256] = build_shuffle_table();

/// Per-control-byte total data bytes consumed by a full group (4..=16).
#[cfg(target_arch = "x86_64")]
const LENGTH_TABLE: [u8; 256] = build_length_table();

#[cfg(target_arch = "x86_64")]
const fn build_shuffle_table() -> [[u8; 16]; 256] {
    let mut table = [[0u8; 16]; 256];
    let mut ctrl = 0usize;
    while ctrl < 256 {
        let mut offset = 0u8;
        let mut k = 0;
        while k < 4 {
            let len = ((ctrl >> (k * 2)) & 0x03) + 1;
            let mut j = 0;
            while j < 4 {
                table[ctrl][k * 4 + j] = if j < len { offset + j as u8 } else { 0xFF };
                j += 1;
            }
            offset += len as u8;
            k += 1;
        }
        ctrl += 1;
    }
    table
}

#[cfg(target_arch = "x86_64")]
const fn build_length_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut ctrl = 0usize;
    while ctrl < 256 {
        let mut total = 0u8;
        let mut k = 0;
        while k < 4 {
            total += (((ctrl >> (k * 2)) & 0x03) + 1) as u8;
            k += 1;
        }
        table[ctrl] = total;
        ctrl += 1;
    }
    table
}

/// Encoded stream: control bytes followed by data bytes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncodedStream {
    /// Control bytes (2 bits per value, packed 4 per byte)
    pub controls: Vec<u8>,
    /// Data bytes (variable-length encoded values)
    pub data: Vec<u8>,
    /// Number of encoded values
    pub count: usize,
}

impl StreamVByte {
    /// Encode a sorted u32 slice using delta + stream vbyte.
    /// Delta-encodes first (val[i] - val[i-1]), then vbyte-encodes the deltas.
    pub fn encode_deltas(values: &[u32]) -> EncodedStream {
        if values.is_empty() {
            return EncodedStream {
                controls: Vec::new(),
                data: Vec::new(),
                count: 0,
            };
        }

        // Delta encode
        let mut deltas = Vec::with_capacity(values.len());
        deltas.push(values[0]);
        for i in 1..values.len() {
            deltas.push(values[i] - values[i - 1]);
        }

        Self::encode_raw(&deltas)
    }

    /// Encode raw u32 values (no delta encoding).
    pub fn encode_raw(values: &[u32]) -> EncodedStream {
        let n = values.len();
        let num_groups = n.div_ceil(4);

        let mut controls = Vec::with_capacity(num_groups);
        let mut data = Vec::with_capacity(n * 2); // Estimate

        let mut i = 0;
        while i + 4 <= n {
            let mut ctrl = 0u8;
            for k in 0..4 {
                let v = values[i + k];
                let len = Self::byte_length(v);
                ctrl |= ((len - 1) as u8) << (k * 2);
                Self::write_value(&mut data, v, len);
            }
            controls.push(ctrl);
            i += 4;
        }

        // Handle remaining values (< 4)
        if i < n {
            let mut ctrl = 0u8;
            for k in 0..(n - i) {
                let v = values[i + k];
                let len = Self::byte_length(v);
                ctrl |= ((len - 1) as u8) << (k * 2);
                Self::write_value(&mut data, v, len);
            }
            controls.push(ctrl);
        }

        EncodedStream {
            controls,
            data,
            count: n,
        }
    }

    /// Decode delta-encoded stream back to sorted u32 values.
    ///
    /// Returns the decoded values. If the input stream is truncated or malformed,
    /// only the successfully decoded prefix is returned (partial decode), and the
    /// returned `Vec` will contain fewer than `count` values without panicking.
    /// Callers requiring complete stream integrity should verify `result.len() == count`.
    pub fn decode_deltas(stream: &EncodedStream, count: usize) -> Vec<u32> {
        // Prefix sum in place on the buffer decode_raw already allocated;
        // a second Vec would be an extra allocation plus an O(n) copy.
        let mut values = Self::decode_raw(stream, count);
        let mut acc = 0u32;
        for v in &mut values {
            acc += *v;
            *v = acc;
        }
        values
    }

    /// Decode raw values from stream.
    ///
    /// Returns the decoded values. If the input stream is truncated, corrupted, or
    /// specifies fewer elements than `count`, decoding terminates safely at the corruption
    /// boundary and returns only the successfully decoded prefix (`result.len() < count`).
    /// Allocations are bounded by `count.min(stream.controls.len() * 4)` to prevent
    /// memory exhaustion on hostile or untrusted stream headers with arbitrarily large `count`.
    pub fn decode_raw(stream: &EncodedStream, count: usize) -> Vec<u32> {
        let max_possible = stream.controls.len().saturating_mul(4);
        let alloc_count = count.min(max_possible);
        let mut values = vec![0u32; alloc_count];
        let decoded = Self::decode_into(stream, alloc_count, &mut values);
        values.truncate(decoded);
        values
    }

    /// Decode directly into a pre-allocated buffer.
    /// Returns the number of values successfully decoded. If the stream is truncated
    /// or malformed, fewer than `count` values may be returned without panicking.
    pub fn decode_into(stream: &EncodedStream, count: usize, output: &mut [u32]) -> usize {
        DECODE_INTO_IMPL(stream, count, output)
    }

    /// Scalar decode continuing from a mid-stream position: tail groups
    /// within 16 bytes of the data end, partial final group, and the full
    /// decode on non-SSSE3 targets (start position 0/0/0).
    fn decode_scalar_from(
        stream: &EncodedStream,
        count: usize,
        output: &mut [u32],
        mut data_pos: usize,
        mut out_idx: usize,
        mut ctrl_idx: usize,
    ) -> usize {
        while ctrl_idx < stream.controls.len() && out_idx < count {
            let ctrl = stream.controls[ctrl_idx];
            let group_size = (count - out_idx).min(4);

            for k in 0..group_size {
                let len = ((ctrl >> (k * 2)) & 0x03) as usize + 1;
                if data_pos + len > stream.data.len() {
                    return out_idx;
                }
                output[out_idx] = Self::read_value(&stream.data, data_pos, len);
                data_pos += len;
                out_idx += 1;
            }
            ctrl_idx += 1;
        }

        out_idx
    }

    fn decode_into_scalar(stream: &EncodedStream, count: usize, output: &mut [u32]) -> usize {
        Self::decode_scalar_from(stream, count, output, 0, 0, 0)
    }

    /// SSSE3 bulk path: one pshufb expands a whole 4-value group, then the
    /// shared scalar loop finishes the tail.
    #[cfg(target_arch = "x86_64")]
    fn decode_into_ssse3(stream: &EncodedStream, count: usize, output: &mut [u32]) -> usize {
        let simd_limit = count.min(output.len());
        // SAFETY: SSSE3 verified by resolve_decode_into before this pointer
        // is ever published; all loads and stores are bounds-guarded inside.
        let (data_pos, out_idx, ctrl_idx) = unsafe {
            Self::decode_groups_ssse3(&stream.controls, &stream.data, simd_limit, output)
        };
        Self::decode_scalar_from(stream, count, output, data_pos, out_idx, ctrl_idx)
    }

    /// Decode full groups with SSSE3 shuffle-table expansion (Lemire et al.).
    ///
    /// Runs while a full 16-byte load from the data stream and a full
    /// 4-lane store to the output stay in bounds; the caller's scalar loop
    /// finishes the rest. Returns (data_pos, out_idx, ctrl_idx).
    ///
    /// # Safety
    ///
    /// Caller must ensure SSSE3 is available and `limit <= output.len()`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "ssse3")]
    unsafe fn decode_groups_ssse3(
        controls: &[u8],
        data: &[u8],
        limit: usize,
        output: &mut [u32],
    ) -> (usize, usize, usize) {
        use std::arch::x86_64::*;

        let mut data_pos = 0usize;
        let mut out_idx = 0usize;
        let mut ctrl_idx = 0usize;

        while ctrl_idx < controls.len() && out_idx + 4 <= limit && data_pos + 16 <= data.len() {
            let ctrl = controls[ctrl_idx] as usize;
            // SAFETY: data_pos + 16 <= data.len() checked by the loop
            // condition; a group consumes at most 16 bytes, so the load
            // covers every byte the shuffle mask can reference.
            let input = unsafe { _mm_loadu_si128(data.as_ptr().add(data_pos) as *const __m128i) };
            let mask =
                // SAFETY: SHUFFLE_TABLE entries are 16 bytes.
                unsafe { _mm_loadu_si128(SHUFFLE_TABLE[ctrl].as_ptr() as *const __m128i) };
            let expanded = _mm_shuffle_epi8(input, mask);
            // SAFETY: out_idx + 4 <= limit <= output.len() checked by the
            // loop condition.
            unsafe {
                _mm_storeu_si128(output.as_mut_ptr().add(out_idx) as *mut __m128i, expanded);
            }

            data_pos += LENGTH_TABLE[ctrl] as usize;
            out_idx += 4;
            ctrl_idx += 1;
        }

        (data_pos, out_idx, ctrl_idx)
    }

    /// Compression ratio: encoded size / raw size.
    pub fn compression_ratio(stream: &EncodedStream) -> f64 {
        let raw_size = stream.count * 4; // 4 bytes per u32
        let encoded_size = stream.controls.len() + stream.data.len();
        if raw_size == 0 {
            return 1.0;
        }
        encoded_size as f64 / raw_size as f64
    }

    // --- Internal helpers ---

    /// Number of bytes needed to encode a u32 value.
    #[inline(always)]
    fn byte_length(v: u32) -> usize {
        if v < (1 << 8) {
            1
        } else if v < (1 << 16) {
            2
        } else if v < (1 << 24) {
            3
        } else {
            4
        }
    }

    /// Write a value using `len` bytes (little-endian).
    #[inline]
    fn write_value(data: &mut Vec<u8>, v: u32, len: usize) {
        let bytes = v.to_le_bytes();
        data.extend_from_slice(&bytes[..len]);
    }

    /// Read a value of `len` bytes from data at position (little-endian).
    #[inline]
    fn read_value(data: &[u8], pos: usize, len: usize) -> u32 {
        let mut bytes = [0u8; 4];
        bytes[..len].copy_from_slice(&data[pos..pos + len]);
        u32::from_le_bytes(bytes)
    }
}

crate::ifunc_dispatch!(
    static DECODE_INTO_IMPL: fn(&EncodedStream, usize, &mut [u32]) -> usize = resolve_decode_into;
);

/// Picks the decode variant for this machine. Runs once; the chosen safe
/// entry is cached in `DECODE_INTO_IMPL`.
fn resolve_decode_into() -> fn(&EncodedStream, usize, &mut [u32]) -> usize {
    if cfg!(miri) {
        return StreamVByte::decode_into_scalar; // Miri cannot execute SSSE3 intrinsics
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("ssse3") {
            return StreamVByte::decode_into_ssse3;
        }
    }
    StreamVByte::decode_into_scalar
}

/// Group Varint encoder/decoder — encodes 4 integers with shared length byte.
pub struct GroupVarint;

impl GroupVarint {
    /// Encode sorted values with delta + group varint.
    pub fn encode_deltas(values: &[u32]) -> Vec<u8> {
        if values.is_empty() {
            return Vec::new();
        }

        let mut deltas = Vec::with_capacity(values.len());
        deltas.push(values[0]);
        for i in 1..values.len() {
            deltas.push(values[i] - values[i - 1]);
        }

        Self::encode_raw(&deltas)
    }

    /// Encode raw u32 values.
    pub fn encode_raw(values: &[u32]) -> Vec<u8> {
        let mut output = Vec::with_capacity(values.len() * 3);
        let n = values.len();
        let mut i = 0;

        while i + 4 <= n {
            let lengths = [
                StreamVByte::byte_length(values[i]),
                StreamVByte::byte_length(values[i + 1]),
                StreamVByte::byte_length(values[i + 2]),
                StreamVByte::byte_length(values[i + 3]),
            ];

            // Control byte
            let ctrl = ((lengths[0] - 1)
                | ((lengths[1] - 1) << 2)
                | ((lengths[2] - 1) << 4)
                | ((lengths[3] - 1) << 6)) as u8;
            output.push(ctrl);

            // Data
            for k in 0..4 {
                let bytes = values[i + k].to_le_bytes();
                output.extend_from_slice(&bytes[..lengths[k]]);
            }

            i += 4;
        }

        // Remaining values (stored as raw u32)
        for j in i..n {
            output.extend_from_slice(&values[j].to_le_bytes());
        }

        // Store count of remaining values in last byte if not multiple of 4
        if !n.is_multiple_of(4) {
            output.push((n % 4) as u8);
        } else {
            output.push(0); // No remainder
        }

        output
    }

    /// Decode group varint with delta reconstruction.
    pub fn decode_deltas(data: &[u8], count: usize) -> Vec<u32> {
        let mut values = Self::decode_raw(data, count);
        let mut acc = 0u32;
        for v in &mut values {
            acc += *v;
            *v = acc;
        }
        values
    }

    /// Decode raw values.
    pub fn decode_raw(data: &[u8], count: usize) -> Vec<u32> {
        let mut values = Vec::with_capacity(count);
        let mut pos = 0;
        let mut remaining = count;

        while remaining >= 4 && pos < data.len() {
            let ctrl = data[pos];
            pos += 1;

            for k in 0..4 {
                let len = ((ctrl >> (k * 2)) & 0x03) as usize + 1;
                if pos + len > data.len() {
                    break;
                }
                let mut bytes = [0u8; 4];
                bytes[..len].copy_from_slice(&data[pos..pos + len]);
                values.push(u32::from_le_bytes(bytes));
                pos += len;
            }

            remaining -= 4;
        }

        // Decode remaining raw u32s
        while remaining > 0 && pos + 4 <= data.len() {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&data[pos..pos + 4]);
            values.push(u32::from_le_bytes(bytes));
            pos += 4;
            remaining -= 1;
        }

        values.truncate(count);
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- StreamVByte tests ---

    #[test]
    fn test_stream_vbyte_empty() {
        let encoded = StreamVByte::encode_deltas(&[]);
        assert_eq!(encoded.count, 0);
        let decoded = StreamVByte::decode_deltas(&encoded, 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_stream_vbyte_single() {
        let values = vec![42];
        let encoded = StreamVByte::encode_deltas(&values);
        let decoded = StreamVByte::decode_deltas(&encoded, values.len());
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_stream_vbyte_small_values() {
        let values = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let encoded = StreamVByte::encode_deltas(&values);
        let decoded = StreamVByte::decode_deltas(&encoded, values.len());
        assert_eq!(decoded, values);

        // Small deltas should compress well
        let ratio = StreamVByte::compression_ratio(&encoded);
        assert!(
            ratio < 0.5,
            "ratio should be < 0.5 for small values, got {}",
            ratio
        );
    }

    #[test]
    fn test_stream_vbyte_large_values() {
        let values = vec![1000, 2000, 100000, 200000, u32::MAX - 1, u32::MAX];
        let encoded = StreamVByte::encode_deltas(&values);
        let decoded = StreamVByte::decode_deltas(&encoded, values.len());
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_stream_vbyte_posting_list() {
        // Simulate a posting list: 1000 doc IDs in universe of 1M
        let values: Vec<u32> = (0..1000).map(|i| i * 1000 + i % 17).collect();
        let encoded = StreamVByte::encode_deltas(&values);
        let decoded = StreamVByte::decode_deltas(&encoded, values.len());
        assert_eq!(decoded, values);

        let ratio = StreamVByte::compression_ratio(&encoded);
        eprintln!(
            "StreamVByte: 1000 posting IDs, ratio={:.2}, {} bytes",
            ratio,
            encoded.controls.len() + encoded.data.len()
        );
        assert!(
            ratio < 0.75,
            "Should compress posting list well, got {}",
            ratio
        );
    }

    #[test]
    fn test_stream_vbyte_decode_into() {
        let values = vec![10, 20, 30, 40, 50];
        let encoded = StreamVByte::encode_deltas(&values);
        let mut output = vec![0u32; 5];

        // Decode deltas manually
        let deltas = StreamVByte::decode_raw(&encoded, 5);
        let mut acc = 0u32;
        for (i, d) in deltas.iter().enumerate() {
            acc += d;
            output[i] = acc;
        }

        assert_eq!(output, values);
    }

    #[test]
    fn test_stream_vbyte_not_multiple_of_4() {
        for n in 1..=15 {
            let values: Vec<u32> = (0..n).map(|i| i * 10 + 1).collect();
            let encoded = StreamVByte::encode_deltas(&values);
            let decoded = StreamVByte::decode_deltas(&encoded, values.len());
            assert_eq!(decoded, values, "Failed for n={}", n);
        }
    }

    #[test]
    fn test_stream_vbyte_raw_roundtrip() {
        let values = vec![
            0,
            1,
            127,
            128,
            255,
            256,
            65535,
            65536,
            16777215,
            16777216,
            u32::MAX,
        ];
        let encoded = StreamVByte::encode_raw(&values);
        let decoded = StreamVByte::decode_raw(&encoded, values.len());
        assert_eq!(decoded, values);
    }

    /// Every control byte (all 256 length combinations of a 4-value group)
    /// must round-trip — this exercises each shuffle-table entry of the
    /// SIMD decode path and the scalar fallback identically.
    #[test]
    fn test_stream_vbyte_all_control_combinations() {
        let mut values = Vec::with_capacity(256 * 4);
        for ctrl in 0..256u32 {
            for k in 0..4 {
                let len = ((ctrl >> (k * 2)) & 3) + 1;
                let base: u32 = match len {
                    1 => 0x21,
                    2 => 0x1234,
                    3 => 0x123456,
                    _ => 0x12345678,
                };
                values.push(base | (ctrl & 0x7F));
            }
        }
        // Also cover non-multiple-of-4 tails near the end of the data.
        for tail in 0..4 {
            let vals = &values[..values.len() - tail];
            let encoded = StreamVByte::encode_raw(vals);
            let decoded = StreamVByte::decode_raw(&encoded, vals.len());
            assert_eq!(decoded, vals, "tail={}", tail);
        }
    }

    /// decode_into / decode_raw must match a naive per-byte reference
    /// decoder on a large mixed-length input (covers the SIMD bulk loop,
    /// the <16-bytes-remaining boundary, and the scalar tail).
    #[test]
    fn test_stream_vbyte_matches_naive_reference() {
        let values: Vec<u32> = (0..10_001u32)
            .map(|i| i.wrapping_mul(2654435761) >> (i % 29))
            .collect();
        let encoded = StreamVByte::encode_raw(&values);

        // Naive reference decode
        let mut reference = Vec::with_capacity(values.len());
        let mut pos = 0usize;
        'outer: for &ctrl in &encoded.controls {
            for k in 0..4 {
                if reference.len() == values.len() {
                    break 'outer;
                }
                let len = ((ctrl >> (k * 2)) & 0x03) as usize + 1;
                let mut bytes = [0u8; 4];
                bytes[..len].copy_from_slice(&encoded.data[pos..pos + len]);
                reference.push(u32::from_le_bytes(bytes));
                pos += len;
            }
        }

        assert_eq!(reference, values);
        assert_eq!(StreamVByte::decode_raw(&encoded, values.len()), values);
        let mut output = vec![0u32; values.len()];
        StreamVByte::decode_into(&encoded, values.len(), &mut output);
        assert_eq!(output, values);
    }

    // --- GroupVarint tests ---

    #[test]
    fn test_group_varint_basic() {
        let values = vec![1, 5, 100, 300, 1000, 70000, 100000, 200000];
        let encoded = GroupVarint::encode_deltas(&values);
        let decoded = GroupVarint::decode_deltas(&encoded, values.len());
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_group_varint_small() {
        let values = vec![1, 2, 3];
        let encoded = GroupVarint::encode_deltas(&values);
        let decoded = GroupVarint::decode_deltas(&encoded, values.len());
        assert_eq!(decoded, values);
    }

    // --- Performance tests ---

    #[test]
    #[cfg_attr(miri, ignore)] // 100 × 100K-value decodes — hours under Miri
    fn test_stream_vbyte_performance() {
        let values: Vec<u32> = (0..100000).map(|i| i * 10).collect();

        let start = std::time::Instant::now();
        let encoded = StreamVByte::encode_deltas(&values);
        let _encode_time = start.elapsed();

        let start = std::time::Instant::now();
        let mut _total = 0usize;
        for _ in 0..100 {
            let decoded = StreamVByte::decode_deltas(&encoded, values.len());
            _total += decoded.len();
        }
        let _decode_time = start.elapsed();

        #[cfg(not(debug_assertions))]
        {
            let ratio = StreamVByte::compression_ratio(&encoded);
            let decode_per_call = _decode_time / 100;
            eprintln!(
                "StreamVByte 100K values: encode={:?}, decode={:?}/call, ratio={:.2}",
                _encode_time, decode_per_call, ratio
            );
        }
    }

    // --- Const-LUT reference-model tests ---
    //
    // The shuffle/length tables are built by const fns; these tests check
    // them against an independent first-principles model of the format
    // (2-bit lane lengths, little-endian packing) rather than reusing the
    // table-construction logic.

    /// Independent re-derivation of a group's lane lengths from a ctrl byte.
    fn ref_lane_lens(ctrl: u8) -> [usize; 4] {
        let mut lens = [0usize; 4];
        for (k, len) in lens.iter_mut().enumerate() {
            *len = ((ctrl as usize >> (k * 2)) & 0x03) + 1;
        }
        lens
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_tables_match_reference_model_exhaustive() {
        // Synthetic packed data: byte i has value i, so lane extraction is
        // recognizable byte-for-byte.
        let data: Vec<u8> = (0u8..16).collect();

        for ctrl in 0u16..256 {
            let ctrl = ctrl as u8;
            let lens = ref_lane_lens(ctrl);

            // LENGTH_TABLE == sum of lane lengths.
            let expected_total: usize = lens.iter().sum();
            assert_eq!(
                LENGTH_TABLE[ctrl as usize] as usize, expected_total,
                "LENGTH_TABLE mismatch for ctrl={ctrl:#04x}"
            );

            // Scalar emulation of the pshufb mask must reproduce exactly the
            // lanes a scalar little-endian decode of the same bytes yields.
            let mask = &SHUFFLE_TABLE[ctrl as usize];
            let mut offset = 0usize;
            for (k, &len) in lens.iter().enumerate() {
                let mut lane_bytes = [0u8; 4];
                for (j, out) in lane_bytes.iter_mut().enumerate() {
                    let m = mask[k * 4 + j];
                    // pshufb semantics: high bit set => lane byte is zero.
                    *out = if m & 0x80 != 0 { 0 } else { data[m as usize] };
                }
                let via_table = u32::from_le_bytes(lane_bytes);
                let via_scalar = StreamVByte::read_value(&data, offset, len);
                assert_eq!(
                    via_table, via_scalar,
                    "lane {k} mismatch for ctrl={ctrl:#04x}"
                );
                offset += len;
            }
        }
    }

    #[test]
    fn test_decode_simd_matches_scalar_all_ctrl_bytes() {
        // 5 groups per ctrl value: enough for the SSSE3 bulk loop to run and
        // for the scalar tail (last group is within 16 bytes of data end).
        const GROUPS: usize = 5;

        for ctrl in 0u16..256 {
            let ctrl = ctrl as u8;
            let lens = ref_lane_lens(ctrl);
            let group_bytes: usize = lens.iter().sum();

            for payload in 0u32..3 {
                // Deterministic xorshift payload, distinct per (ctrl, payload).
                let mut state = (u32::from(ctrl) << 8) ^ (payload + 0x9E37_79B9);
                let mut data = Vec::with_capacity(GROUPS * group_bytes);
                for _ in 0..GROUPS * group_bytes {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    data.push(state as u8);
                }

                let count = GROUPS * 4;
                let stream = EncodedStream {
                    controls: vec![ctrl; GROUPS],
                    data,
                    count,
                };

                let mut out_scalar = vec![0u32; count];
                let n_scalar = StreamVByte::decode_into_scalar(&stream, count, &mut out_scalar);

                let mut out_dispatch = vec![0u32; count];
                let n_dispatch = StreamVByte::decode_into(&stream, count, &mut out_dispatch);

                assert_eq!(n_scalar, n_dispatch, "count mismatch for ctrl={ctrl:#04x}");
                assert_eq!(
                    out_scalar, out_dispatch,
                    "dispatch mismatch for ctrl={ctrl:#04x} payload={payload}"
                );

                // On x86_64 hardware with SSSE3, force the SIMD variant so
                // this test provably exercises it (never under Miri: dispatch
                // is scalar-only there and the intrinsics can't execute).
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                {
                    if std::arch::is_x86_feature_detected!("ssse3") {
                        let mut out_simd = vec![0u32; count];
                        let n_simd =
                            StreamVByte::decode_into_ssse3(&stream, count, &mut out_simd);
                        assert_eq!(n_scalar, n_simd, "SIMD count mismatch ctrl={ctrl:#04x}");
                        assert_eq!(
                            out_scalar, out_simd,
                            "SIMD mismatch for ctrl={ctrl:#04x} payload={payload}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_stream_vbyte_truncated_stream_safety() {
        // Construct a stream where control bytes say there are 4 values of 4 bytes each (16 bytes data),
        // but stream.data only has 5 bytes (truncated).
        let stream = EncodedStream {
            controls: vec![0xFF], // four 4-byte integers
            data: vec![1, 2, 3, 4, 5], // only 5 bytes instead of 16
            count: 4,
        };

        // decode_raw should safely return without panicking, decoding only the 1st 4-byte integer that fits
        let decoded = StreamVByte::decode_raw(&stream, stream.count);
        assert_eq!(decoded.len(), 1); // Only 1st 4-byte integer fits in 5 bytes
        assert_eq!(decoded[0], u32::from_le_bytes([1, 2, 3, 4]));

        // Completely empty data with non-empty controls
        let empty_data_stream = EncodedStream {
            controls: vec![0x00],
            data: vec![],
            count: 4,
        };
        let decoded_empty = StreamVByte::decode_raw(&empty_data_stream, empty_data_stream.count);
        assert_eq!(decoded_empty.len(), 0);
    }

    #[test]
    fn test_stream_vbyte_unbounded_count_dos_protection() {
        // Untrusted stream with arbitrarily large count (e.g. 1 << 40 or usize::MAX)
        // should NOT attempt a huge allocation and must bound allocation to controls.len() * 4.
        let stream = EncodedStream {
            controls: vec![0x00], // 1 control byte -> max 4 values
            data: vec![10, 20, 30, 40],
            count: usize::MAX,
        };

        let decoded = StreamVByte::decode_raw(&stream, stream.count);
        assert_eq!(decoded, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_stream_vbyte_simd_to_scalar_handoff_truncated_stream_safety() {
        // 10 groups of 1-byte values = 40 values, requiring 40 data bytes.
        // SSSE3 bulk loop runs while data_pos + 16 <= data.len().
        // If data has only 26 bytes:
        // - Group 0 (pos 0..4, +16=20 <= 26): decoded by SSSE3 (4 values)
        // - Group 1 (pos 4..8, +16=24 <= 26): decoded by SSSE3 (4 values)
        // - Group 2 (pos 8..12, +16=24 <= 26): decoded by SSSE3 (4 values)
        // - Group 3 (pos 12..16, +16=28 > 26): SSSE3 loop stops and hands off to scalar loop
        // - Scalar loop decodes Group 3 (pos 12..16), Group 4 (pos 16..20), Group 5 (pos 20..24),
        //   and 2 values of Group 6 (pos 24..25, 25..26), then stops cleanly.
        let mut full_data = Vec::with_capacity(40);
        for i in 0..40u8 {
            full_data.push(i + 1);
        }

        let truncated_data = full_data[..26].to_vec();
        let stream = EncodedStream {
            controls: vec![0x00; 10], // 10 groups of 1-byte integers
            data: truncated_data,
            count: 40,
        };

        let mut out_scalar = vec![0u32; 40];
        let n_scalar = StreamVByte::decode_into_scalar(&stream, 40, &mut out_scalar);
        assert_eq!(n_scalar, 26);
        for (i, &val) in out_scalar[..26].iter().enumerate() {
            assert_eq!(val, (i + 1) as u32);
        }

        let decoded = StreamVByte::decode_raw(&stream, 40);
        assert_eq!(decoded.len(), 26);
        assert_eq!(decoded, out_scalar[..26]);

        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            if std::arch::is_x86_feature_detected!("ssse3") {
                let mut out_simd = vec![0u32; 40];
                let n_simd = StreamVByte::decode_into_ssse3(&stream, 40, &mut out_simd);
                assert_eq!(n_simd, 26, "SIMD handoff count mismatch");
                assert_eq!(out_simd[..26], out_scalar[..26], "SIMD handoff output mismatch");
            }
        }
    }
}
