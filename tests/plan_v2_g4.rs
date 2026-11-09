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
