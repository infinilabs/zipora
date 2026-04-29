//! SIMD Data Validation — CRC32C hardware-accelerated checksums.
//!
//! UTF-8 validation removed — `std::str::from_utf8` is already SIMD-optimized internally.

pub mod checksum;

pub use checksum::{
    Crc32cImpl, crc32c, crc32c_finalize, crc32c_hash, crc32c_update, detect_crc32c_impl,
};
