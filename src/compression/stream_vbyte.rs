//! Stream VByte — SIMD-accelerated variable-byte integer encoding.
//!
//! Encodes sorted u32 sequences using delta + variable-byte coding.
//! Control bytes are separated from data bytes for SIMD-friendly decoding.
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
    pub fn decode_deltas(stream: &EncodedStream, count: usize) -> Vec<u32> {
        let deltas = Self::decode_raw(stream, count);

        // Prefix sum to recover original values
        let mut values = Vec::with_capacity(deltas.len());
        let mut acc = 0u32;
        for d in deltas {
            acc += d;
            values.push(acc);
        }

        values
    }

    /// Decode raw values from stream.
    pub fn decode_raw(stream: &EncodedStream, count: usize) -> Vec<u32> {
        let mut values = Vec::with_capacity(count);
        let mut data_pos = 0usize;
        let mut remaining = count;

        for &ctrl in &stream.controls {
            let group_size = remaining.min(4);

            for k in 0..group_size {
                let len = ((ctrl >> (k * 2)) & 0x03) as usize + 1;
                let val = Self::read_value(&stream.data, data_pos, len);
                values.push(val);
                data_pos += len;
            }

            remaining -= group_size;
            if remaining == 0 {
                break;
            }
        }

        values
    }

    /// Decode directly into a pre-allocated buffer.
    pub fn decode_into(stream: &EncodedStream, count: usize, output: &mut [u32]) {
        let mut data_pos = 0usize;
        let mut out_idx = 0usize;
        let mut remaining = count;

        for &ctrl in &stream.controls {
            let group_size = remaining.min(4);

            for k in 0..group_size {
                let len = ((ctrl >> (k * 2)) & 0x03) as usize + 1;
                output[out_idx] = Self::read_value(&stream.data, data_pos, len);
                data_pos += len;
                out_idx += 1;
            }

            remaining -= group_size;
            if remaining == 0 {
                break;
            }
        }
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
        let raw = Self::decode_raw(data, count);
        let mut values = Vec::with_capacity(raw.len());
        let mut acc = 0u32;
        for d in raw {
            acc += d;
            values.push(acc);
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
}
