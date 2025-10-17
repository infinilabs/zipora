//! # Hardware-Accelerated CRC32C Checksum
//!
//! CRC32C (Castagnoli) checksum implementation with hardware acceleration.
//!
//! ## Architecture
//!
//! Implements zipora's 6-tier SIMD framework:
//! - **Tier 5**: AVX-512 (not applicable for CRC32C)
//! - **Tier 4**: x86_64 SSE4.2 `_mm_crc32_u64` instruction (25 GB/s target)
//! - **Tier 3**: BMI2 (not applicable for CRC32C)
//! - **Tier 2**: ARM64 CRC32 extensions `__crc32cd` instruction
//! - **Tier 1**: ARM NEON (not applicable for CRC32C)
//! - **Tier 0**: Scalar table-based fallback
//!
//! ## Performance Targets
//!
//! - **x86_64 SSE4.2**: 25 GB/s with hardware CRC instruction
//! - **ARM64 CRC**: 15+ GB/s with hardware CRC instruction
//! - **Scalar**: 1-2 GB/s with table-based implementation
//!
//! ## Usage
//!
//! ```rust
//! use zipora::io::simd_validation::checksum::crc32c_hash;
//!
//! let data = b"Hello, world!";
//! let checksum = crc32c_hash(data).unwrap();
//! assert_ne!(checksum, 0);
//! ```
//!
//! ## Streaming Usage
//!
//! ```rust
//! use zipora::io::simd_validation::checksum::{crc32c_update, crc32c_finalize};
//!
//! let mut crc = 0xFFFFFFFF;
//! crc = crc32c_update(crc, b"Hello, ").unwrap();
//! crc = crc32c_update(crc, b"world!").unwrap();
//! let final_crc = crc32c_finalize(crc);
//! assert_ne!(final_crc, 0);
//! ```

use crate::error::{Result, ZiporaError};
use crate::system::cpu_features::get_cpu_features;
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// CRC32C polynomial (Castagnoli)
const CRC32C_POLY: u32 = 0x82f63b78;

/// CRC32C lookup table for scalar implementation
static CRC32C_TABLE: OnceLock<[u32; 256]> = OnceLock::new();

/// Get or initialize the CRC32C lookup table
fn get_crc32c_table() -> &'static [u32; 256] {
    CRC32C_TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for i in 0..256 {
            let mut crc = i as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ CRC32C_POLY;
                } else {
                    crc >>= 1;
                }
            }
            table[i] = crc;
        }
        table
    })
}

/// CRC32C implementation selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crc32cImpl {
    /// x86_64 SSE4.2 hardware CRC instruction
    Sse42,
    /// ARM64 CRC hardware extension
    ArmCrc32,
    /// Scalar table-based fallback
    Scalar,
}

/// Detect optimal CRC32C implementation
pub fn detect_crc32c_impl() -> Crc32cImpl {
    let features = get_cpu_features();

    #[cfg(target_arch = "x86_64")]
    {
        if features.has_sse42 {
            return Crc32cImpl::Sse42;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if features.has_crc32 {
            return Crc32cImpl::ArmCrc32;
        }
    }

    Crc32cImpl::Scalar
}

/// Compute CRC32C checksum for data
///
/// # Arguments
///
/// * `data` - Input data to checksum
/// * `init` - Initial CRC value (0xFFFFFFFF for new checksum, or previous value for streaming)
///
/// # Returns
///
/// CRC32C checksum value
///
/// # Example
///
/// ```rust
/// use zipora::io::simd_validation::checksum::crc32c;
///
/// let data = b"Hello, world!";
/// let checksum = crc32c(data, 0xFFFFFFFF).unwrap();
/// let checksum = !checksum; // Apply final XOR
/// assert_ne!(checksum, 0);
/// ```
///
/// # Note
///
/// For standard CRC32C usage, use `crc32c_hash()` which handles initialization and finalization.
pub fn crc32c(data: &[u8], init: u32) -> Result<u32> {
    if data.is_empty() {
        return Ok(init);
    }

    let impl_type = detect_crc32c_impl();

    #[cfg(target_arch = "x86_64")]
    {
        match impl_type {
            Crc32cImpl::Sse42 => unsafe { Ok(crc32c_sse42(data, init)) },
            Crc32cImpl::Scalar => Ok(crc32c_scalar(data, init)),
            _ => Ok(crc32c_scalar(data, init)),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        match impl_type {
            Crc32cImpl::ArmCrc32 => unsafe { Ok(crc32c_arm(data, init)) },
            Crc32cImpl::Scalar => Ok(crc32c_scalar(data, init)),
            _ => Ok(crc32c_scalar(data, init)),
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Ok(crc32c_scalar(data, init))
    }
}

/// Compute CRC32C hash (standard interface with proper initialization/finalization)
///
/// This is the recommended interface for computing CRC32C checksums.
///
/// # Arguments
///
/// * `data` - Input data to checksum
///
/// # Returns
///
/// CRC32C checksum value
///
/// # Example
///
/// ```rust
/// use zipora::io::simd_validation::checksum::crc32c_hash;
///
/// let data = b"123456789";
/// let checksum = crc32c_hash(data).unwrap();
/// assert_eq!(checksum, 0xe3069283);
/// ```
pub fn crc32c_hash(data: &[u8]) -> Result<u32> {
    let crc = crc32c(data, 0xFFFFFFFF)?;
    Ok(!crc)
}

/// Update CRC32C checksum (streaming interface)
///
/// # Arguments
///
/// * `crc` - Current CRC value
/// * `data` - New data to process
///
/// # Returns
///
/// Updated CRC32C value
///
/// # Example
///
/// ```rust
/// use zipora::io::simd_validation::checksum::{crc32c_update, crc32c_finalize};
///
/// let mut crc = 0;
/// crc = crc32c_update(crc, b"Hello, ").unwrap();
/// crc = crc32c_update(crc, b"world!").unwrap();
/// let final_crc = crc32c_finalize(crc);
/// ```
pub fn crc32c_update(crc: u32, data: &[u8]) -> Result<u32> {
    crc32c(data, crc)
}

/// Finalize CRC32C checksum
///
/// Applies the final XOR to complete the CRC32C calculation.
///
/// # Arguments
///
/// * `crc` - Current CRC value (from crc32c_update)
///
/// # Returns
///
/// Final CRC32C value
pub fn crc32c_finalize(crc: u32) -> u32 {
    !crc
}

// ============================================================================
// Scalar Implementation (Tier 0)
// ============================================================================

/// Scalar CRC32C implementation using lookup table
///
/// This is the portable fallback implementation that works on all platforms.
/// Performance: ~1-2 GB/s
fn crc32c_scalar(data: &[u8], mut crc: u32) -> u32 {
    let table = get_crc32c_table();

    // Process one byte at a time
    for &byte in data {
        let index = ((crc as u8) ^ byte) as usize;
        crc = (crc >> 8) ^ table[index];
    }

    crc
}

// ============================================================================
// x86_64 SSE4.2 Implementation (Tier 4)
// ============================================================================

#[cfg(target_arch = "x86_64")]
/// x86_64 SSE4.2 CRC32C implementation using hardware instruction
///
/// Performance: ~25 GB/s with hardware acceleration
///
/// # Safety
///
/// Requires SSE4.2 support (checked at runtime)
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_sse42(data: &[u8], mut crc: u32) -> u32 {
    let mut ptr = data.as_ptr();
    let mut remaining = data.len();

    // Process 8 bytes at a time with _mm_crc32_u64
    while remaining >= 8 {
        // Read 8 bytes as u64 (handles unaligned access)
        // SAFETY: We check remaining >= 8, so this is safe
        let value = unsafe { ptr.cast::<u64>().read_unaligned() };
        crc = _mm_crc32_u64(crc as u64, value) as u32;
        // SAFETY: We just checked we have at least 8 bytes
        ptr = unsafe { ptr.add(8) };
        remaining -= 8;
    }

    // Process 4 bytes with _mm_crc32_u32
    if remaining >= 4 {
        // SAFETY: We check remaining >= 4, so this is safe
        let value = unsafe { ptr.cast::<u32>().read_unaligned() };
        crc = _mm_crc32_u32(crc, value);
        // SAFETY: We just checked we have at least 4 bytes
        ptr = unsafe { ptr.add(4) };
        remaining -= 4;
    }

    // Process 2 bytes with _mm_crc32_u16
    if remaining >= 2 {
        // SAFETY: We check remaining >= 2, so this is safe
        let value = unsafe { ptr.cast::<u16>().read_unaligned() };
        crc = _mm_crc32_u16(crc, value);
        // SAFETY: We just checked we have at least 2 bytes
        ptr = unsafe { ptr.add(2) };
        remaining -= 2;
    }

    // Process remaining byte with _mm_crc32_u8
    if remaining == 1 {
        // SAFETY: We check remaining == 1, so this is safe
        crc = _mm_crc32_u8(crc, unsafe { *ptr });
    }

    crc
}

// ============================================================================
// ARM64 CRC Implementation (Tier 2)
// ============================================================================

#[cfg(target_arch = "aarch64")]
/// ARM64 CRC32C implementation using hardware CRC extensions
///
/// Performance: ~15+ GB/s with hardware acceleration
///
/// # Safety
///
/// Requires ARM CRC32 extensions (checked at runtime)
#[target_feature(enable = "crc")]
unsafe fn crc32c_arm(data: &[u8], mut crc: u32) -> u32 {
    let mut ptr = data.as_ptr();
    let mut remaining = data.len();

    // Process 8 bytes at a time with __crc32cd
    while remaining >= 8 {
        // SAFETY: We check remaining >= 8, so this is safe
        let value = unsafe { ptr.cast::<u64>().read_unaligned() };
        crc = __crc32cd(crc, value);
        // SAFETY: We just checked we have at least 8 bytes
        ptr = unsafe { ptr.add(8) };
        remaining -= 8;
    }

    // Process 4 bytes with __crc32cw
    if remaining >= 4 {
        // SAFETY: We check remaining >= 4, so this is safe
        let value = unsafe { ptr.cast::<u32>().read_unaligned() };
        crc = __crc32cw(crc, value);
        // SAFETY: We just checked we have at least 4 bytes
        ptr = unsafe { ptr.add(4) };
        remaining -= 4;
    }

    // Process 2 bytes with __crc32ch
    if remaining >= 2 {
        // SAFETY: We check remaining >= 2, so this is safe
        let value = unsafe { ptr.cast::<u16>().read_unaligned() };
        crc = __crc32ch(crc, value);
        // SAFETY: We just checked we have at least 2 bytes
        ptr = unsafe { ptr.add(2) };
        remaining -= 2;
    }

    // Process remaining byte with __crc32cb
    if remaining == 1 {
        // SAFETY: We check remaining == 1, so this is safe
        crc = __crc32cb(crc, unsafe { *ptr });
    }

    crc
}

// ARM CRC32C intrinsics (Castagnoli variant)
#[cfg(target_arch = "aarch64")]
extern "C" {
    #[link_name = "llvm.aarch64.crc32cb"]
    fn __crc32cb(crc: u32, data: u8) -> u32;

    #[link_name = "llvm.aarch64.crc32ch"]
    fn __crc32ch(crc: u32, data: u16) -> u32;

    #[link_name = "llvm.aarch64.crc32cw"]
    fn __crc32cw(crc: u32, data: u32) -> u32;

    #[link_name = "llvm.aarch64.crc32cd"]
    fn __crc32cd(crc: u32, data: u64) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32c_empty() {
        let crc = crc32c_hash(&[]).unwrap();
        assert_eq!(crc, !0xFFFFFFFF); // Empty data should give NOT of initial value
    }

    #[test]
    fn test_crc32c_single_byte() {
        let crc = crc32c_hash(b"a").unwrap();
        assert_ne!(crc, 0);
    }

    #[test]
    fn test_crc32c_hello_world() {
        let data = b"Hello, world!";
        let crc = crc32c_hash(data).unwrap();
        assert_ne!(crc, 0);

        // CRC32C should be deterministic
        let crc2 = crc32c_hash(data).unwrap();
        assert_eq!(crc, crc2);
    }

    #[test]
    fn test_crc32c_known_value() {
        // CRC32C("123456789") = 0xe3069283
        let data = b"123456789";
        let crc = crc32c_hash(data).unwrap();
        assert_eq!(crc, 0xe3069283);
    }

    #[test]
    fn test_crc32c_incremental() {
        let data = b"Hello, world!";

        // Single-shot CRC
        let crc_full = crc32c_hash(data).unwrap();

        // Incremental CRC
        let mut crc_inc = 0xFFFFFFFF;
        crc_inc = crc32c_update(crc_inc, b"Hello, ").unwrap();
        crc_inc = crc32c_update(crc_inc, b"world!").unwrap();
        let crc_inc = crc32c_finalize(crc_inc);

        assert_eq!(crc_full, crc_inc);
    }

    #[test]
    fn test_crc32c_streaming() {
        let data = b"The quick brown fox jumps over the lazy dog";

        // Single-shot
        let crc_full = crc32c_hash(data).unwrap();

        // Streaming
        let mut crc_stream = 0xFFFFFFFF;
        for chunk in data.chunks(7) {
            crc_stream = crc32c_update(crc_stream, chunk).unwrap();
        }
        let crc_stream = crc32c_finalize(crc_stream);

        assert_eq!(crc_full, crc_stream);
    }

    #[test]
    fn test_crc32c_all_implementations() {
        let data = b"Test data for CRC32C";

        // Scalar implementation
        let crc_scalar = crc32c_scalar(data, 0xFFFFFFFF);

        // Hardware implementations (if available)
        #[cfg(target_arch = "x86_64")]
        {
            let features = get_cpu_features();
            if features.has_sse42 {
                let crc_sse42 = unsafe { crc32c_sse42(data, 0xFFFFFFFF) };
                assert_eq!(crc_scalar, crc_sse42, "SSE4.2 CRC mismatch");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let features = get_cpu_features();
            if features.has_crc32 {
                let crc_arm = unsafe { crc32c_arm(data, 0xFFFFFFFF) };
                assert_eq!(crc_scalar, crc_arm, "ARM CRC mismatch");
            }
        }
    }

    #[test]
    fn test_crc32c_different_sizes() {
        // Test various sizes to ensure proper handling
        for size in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024] {
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let crc = crc32c_hash(&data).unwrap();

            // Verify incremental matches
            let mut crc_inc = 0xFFFFFFFF;
            for chunk in data.chunks(13) {
                crc_inc = crc32c_update(crc_inc, chunk).unwrap();
            }
            let crc_inc = crc32c_finalize(crc_inc);

            assert_eq!(crc, crc_inc, "CRC mismatch for size {}", size);
        }
    }

    #[test]
    fn test_crc32c_unaligned() {
        // Test unaligned data access
        let data: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();

        for offset in 0..8 {
            if offset >= data.len() {
                break;
            }
            let slice = &data[offset..];
            let crc = crc32c_hash(slice).unwrap();
            assert_ne!(crc, 0);
        }
    }

    #[test]
    fn test_crc32c_impl_detection() {
        let impl_type = detect_crc32c_impl();

        #[cfg(target_arch = "x86_64")]
        {
            let features = get_cpu_features();
            if features.has_sse42 {
                assert_eq!(impl_type, Crc32cImpl::Sse42);
            } else {
                assert_eq!(impl_type, Crc32cImpl::Scalar);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let features = get_cpu_features();
            if features.has_crc32 {
                assert_eq!(impl_type, Crc32cImpl::ArmCrc32);
            } else {
                assert_eq!(impl_type, Crc32cImpl::Scalar);
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            assert_eq!(impl_type, Crc32cImpl::Scalar);
        }
    }

    #[test]
    fn test_crc32c_large_data() {
        // Test with large data to ensure hardware acceleration is used
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let crc = crc32c_hash(&data).unwrap();
        assert_ne!(crc, 0);

        // Verify consistency
        let crc2 = crc32c_hash(&data).unwrap();
        assert_eq!(crc, crc2);
    }
}

#[cfg(all(test, not(miri)))]
mod benches {
    use super::*;
    use std::time::Instant;

    fn benchmark_crc32c(name: &str, data: &[u8], iterations: usize) {
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = crc32c_hash(data).unwrap();
        }

        let elapsed = start.elapsed();
        let bytes_processed = data.len() * iterations;
        let throughput_gbps = (bytes_processed as f64 / elapsed.as_secs_f64()) / 1_000_000_000.0;

        println!("{}: {:.2} GB/s ({} iterations, {} bytes)",
                 name, throughput_gbps, iterations, data.len());
    }

    #[test]
    fn bench_crc32c_small() {
        let data: Vec<u8> = (0..64).map(|i| (i % 256) as u8).collect();
        benchmark_crc32c("CRC32C (64 bytes)", &data, 1_000_000);
    }

    #[test]
    fn bench_crc32c_medium() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        benchmark_crc32c("CRC32C (1 KB)", &data, 100_000);
    }

    #[test]
    fn bench_crc32c_large() {
        let data: Vec<u8> = (0..65536).map(|i| (i % 256) as u8).collect();
        benchmark_crc32c("CRC32C (64 KB)", &data, 10_000);
    }

    #[test]
    fn bench_crc32c_scalar() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

        let start = Instant::now();
        for _ in 0..100_000 {
            let _ = !crc32c_scalar(&data, 0xFFFFFFFF);
        }
        let elapsed = start.elapsed();

        let bytes_processed = data.len() * 100_000;
        let throughput_gbps = (bytes_processed as f64 / elapsed.as_secs_f64()) / 1_000_000_000.0;

        println!("CRC32C Scalar (1 KB): {:.2} GB/s", throughput_gbps);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn bench_crc32c_sse42() {
        let features = get_cpu_features();
        if !features.has_sse42 {
            println!("SSE4.2 not available, skipping benchmark");
            return;
        }

        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

        let start = Instant::now();
        for _ in 0..100_000 {
            unsafe {
                let _ = !crc32c_sse42(&data, 0xFFFFFFFF);
            }
        }
        let elapsed = start.elapsed();

        let bytes_processed = data.len() * 100_000;
        let throughput_gbps = (bytes_processed as f64 / elapsed.as_secs_f64()) / 1_000_000_000.0;

        println!("CRC32C SSE4.2 (1 KB): {:.2} GB/s (target: 25 GB/s)", throughput_gbps);
    }
}
