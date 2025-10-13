//! # SIMD Data Validation Module
//!
//! High-performance data validation using SIMD instructions following zipora's
//! 6-tier SIMD framework.
//!
//! ## Modules
//! - `utf8`: UTF-8 encoding validation (15+ GB/s with AVX2)
//! - `checksum`: CRC32C hardware-accelerated checksums (25 GB/s with SSE4.2)
//!
//! ## Performance
//! - **UTF-8 Validation**: 15+ GB/s (AVX2), 8-12 GB/s (SSE), 2-3 GB/s (scalar)
//! - **CRC32C Checksum**: 25 GB/s (x86_64 SSE4.2), 15+ GB/s (ARM64 CRC), 1-2 GB/s (scalar)
//!
//! ## Example
//!
//! ```
//! use zipora::io::simd_validation::{utf8, checksum};
//!
//! // Validate UTF-8 encoded data
//! assert!(utf8::is_valid_utf8(b"Hello, World!"));
//! assert!(utf8::is_valid_utf8("Hello, 世界! 🦀".as_bytes()));
//! assert!(!utf8::is_valid_utf8(&[0xFF, 0xFE, 0xFD]));
//!
//! // Compute CRC32C checksum
//! let data = b"Hello, world!";
//! let crc = checksum::crc32c(data, 0).unwrap();
//! ```

pub mod utf8;
pub mod checksum;

// Re-export commonly used types and functions
pub use utf8::{is_valid_utf8, validate_utf8, Utf8SimdTier, Utf8Validator};
pub use checksum::{crc32c, crc32c_hash, crc32c_update, crc32c_finalize, detect_crc32c_impl, Crc32cImpl};
