//! Plan v2 (reliability-first 5.0.x) regression tests, group G4.
//!
//! Every test here goes through the public API. Tests whose name ends in
//! `_guard` pass at v5.0.0 and pin behaviour a fix must keep; the others fail
//! at v5.0.0 for the reason given in their doc comment.

use std::io::Write;

// ---------------------------------------------------------------------------
// C5.3 `MemoryMappedInput::read_var_int` (src/io/mmap.rs)
// ---------------------------------------------------------------------------

#[cfg(feature = "mmap")]
mod mmap_var_int {
    use super::*;
    use zipora::DataInput;
    use zipora::io::{MemoryMappedInput, VarInt};

    /// Writes `bytes` followed by `pad` zero bytes and opens it. A file of at
    /// most 4 KiB uses the buffered strategy, a larger one a real mapping;
    /// both share `read_var_int`.
    fn open(bytes: &[u8], pad: usize) -> (tempfile::NamedTempFile, MemoryMappedInput) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(bytes).unwrap();
        file.write_all(&vec![0u8; pad]).unwrap();
        file.flush().unwrap();
        let input = MemoryMappedInput::from_path(file.path()).unwrap();
        (file, input)
    }

    fn ten_byte_varint(last: u8) -> Vec<u8> {
        let mut bytes = vec![0xFF; 9];
        bytes.push(last);
        bytes
    }

    /// The 10th byte of a LEB128 `u64` carries bit 63 only. `VarInt::read_from`
    /// rejects `byte > 1` at shift 63; the mmap reader has no such check and
    /// `(byte & 0x7F) << 63` silently drops the high bits, returning
    /// `Ok(0x7FFF_FFFF_FFFF_FFFF)` (or `Ok(u64::MAX)`) for an encoding of a
    /// value >= 2^64.
    #[test]
    fn test_v2_mmap_read_var_int_rejects_u64_overflow() {
        for last in [0x02u8, 0x03, 0x7F] {
            let bytes = ten_byte_varint(last);
            assert!(
                VarInt::decode(&bytes).is_err(),
                "reference decoder rejects 0x{last:02X}"
            );
            for pad in [0usize, 8192] {
                let (_file, mut input) = open(&bytes, pad);
                let got = input.read_var_int();
                assert!(
                    got.is_err(),
                    "10th varint byte 0x{last:02X} (file {} bytes, strategy {:?}): \
                     read_var_int returned {got:?} for a value that overflows u64",
                    bytes.len() + pad,
                    input.strategy()
                );
            }
        }
    }

    /// Guard: the longest valid encodings still decode and agree with
    /// `VarInt::decode`.
    #[test]
    fn test_v2_mmap_read_var_int_max_values_guard() {
        for value in [u64::MAX, 1u64 << 63, (1u64 << 63) - 1, 0, 127, 128] {
            let bytes = VarInt::encode(value);
            assert_eq!(VarInt::decode(&bytes).unwrap().0, value);
            for pad in [0usize, 8192] {
                let (_file, mut input) = open(&bytes, pad);
                assert_eq!(input.read_var_int().unwrap(), value, "pad {pad}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C5.4 `EntropyBitOps::pack_bits` (src/entropy/bit_ops.rs)
// ---------------------------------------------------------------------------

mod pack_bits {
    use zipora::entropy::EntropyBitOps;

    /// `pack_bits` validates `width <= 32` and `offset <= 64` but not
    /// `offset + width <= 64`, so `64 - offset - width` underflows: a panic
    /// with overflow checks (test profile), a silent `Ok` that drops the value
    /// in release. It must return `Err`.
    #[test]
    fn test_v2_pack_bits_rejects_field_past_bit_64() {
        let ops = EntropyBitOps::new();
        for (offset, width) in [(60u32, 8u32), (64, 1), (33, 32), (57, 8)] {
            let mut stream = 0u64;
            let got = ops.pack_bits(&mut stream, 0xFF, offset, width);
            assert!(
                got.is_err(),
                "offset {offset} + width {width} > 64 must be rejected, got {got:?}"
            );
            assert_eq!(stream, 0, "rejected call must not modify the stream");
        }
    }

    /// Guard: fields that end exactly at bit 64 (or are empty) still pack.
    #[test]
    fn test_v2_pack_bits_boundaries_guard() {
        let ops = EntropyBitOps::new();
        let cases = [
            (0xFFu32, 56u32, 8u32, 0xFFu64),
            (0xFFFF_FFFF, 32, 32, 0xFFFF_FFFF),
            (0xFFFF_FFFF, 0, 32, 0xFFFF_FFFF_0000_0000),
            (0x1, 63, 1, 0x1),
            (0x0, 64, 0, 0x0),
            (0xAB, 0, 0, 0x0),
        ];
        for (value, offset, width, expected) in cases {
            let mut stream = 0u64;
            ops.pack_bits(&mut stream, value, offset, width).unwrap();
            assert_eq!(stream, expected, "offset {offset} width {width}");
        }
        let mut stream = 0u64;
        assert!(ops.pack_bits(&mut stream, 1, 65, 0).is_err());
        assert!(ops.pack_bits(&mut stream, 1, 0, 33).is_err());
    }
}
