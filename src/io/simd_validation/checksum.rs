//! # Hardware-Accelerated CRC32C Checksum
//!
//! CRC32C (Castagnoli) checksum implementation with hardware acceleration.
//!
//! ## Architecture
//!
//! [`detect_crc32c_impl`] picks one of three implementations at run time:
//! - **x86_64 SSE4.2**: the `CRC32` instruction through `_mm_crc32_u64` and
//!   its 32/16/8-bit forms
//! - **aarch64 CRC extension**: the `CRC32C` instructions through
//!   `__crc32cd` and its 32/16/8-bit forms
//! - **Scalar**: a 256-entry lookup table, one byte per step
//!
//! ## Performance
//!
//! Limitation: no throughput figure is claimed. Each hardware kernel runs
//! one dependent chain of CRC instructions, so it is bound by their
//! latency. On Linux aarch64, `CpuFeatures::has_crc32` is never set, so
//! [`crc32c`] runs the scalar code there.
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

use crate::error::Result;
use crate::system::cpu_features::get_cpu_features;
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{__crc32cb, __crc32cd, __crc32ch, __crc32cw};

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
            // SAFETY: detect_crc32c_impl verified SSE4.2 availability
            Crc32cImpl::Sse42 => unsafe { Ok(crc32c_sse42(data, init)) },
            Crc32cImpl::Scalar => Ok(crc32c_scalar(data, init)),
            _ => Ok(crc32c_scalar(data, init)),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        match impl_type {
            // SAFETY: detect_crc32c_impl verified ARM CRC32 availability
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
/// * `crc` - Current CRC value: `0xFFFF_FFFF` for a new checksum, otherwise
///   the value the previous call returned
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
/// let mut crc = 0xFFFF_FFFF;
/// crc = crc32c_update(crc, b"12345").unwrap();
/// crc = crc32c_update(crc, b"6789").unwrap();
/// // The CRC32C check value of "123456789"
/// assert_eq!(crc32c_finalize(crc), 0xE306_9283);
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
// Scalar Implementation
// ============================================================================

/// Scalar CRC32C implementation using lookup table
///
/// This is the portable fallback implementation that works on all platforms.
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
// x86_64 SSE4.2 Implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
/// x86_64 SSE4.2 CRC32C implementation using hardware instruction
///
/// # Safety
///
/// The CPU must support SSE4.2. There are no requirements on `data`.
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_sse42(data: &[u8], mut crc: u32) -> u32 {
    // Invariant: `ptr` points at the first unprocessed byte of `data`, and
    // `remaining` bytes of `data` start there.
    let mut ptr = data.as_ptr();
    let mut remaining = data.len();

    // Process 8 bytes at a time with _mm_crc32_u64
    while remaining >= 8 {
        // Read 8 bytes as u64 (handles unaligned access)
        // SAFETY: by the invariant and `remaining >= 8`, the 8 bytes at `ptr`
        // lie inside `data`; `read_unaligned` needs no alignment.
        let value = unsafe { ptr.cast::<u64>().read_unaligned() };
        crc = _mm_crc32_u64(crc as u64, value) as u32;
        // SAFETY: `remaining >= 8`, so `ptr + 8` is inside `data` or one
        // past its end.
        ptr = unsafe { ptr.add(8) };
        remaining -= 8;
    }

    // Process 4 bytes with _mm_crc32_u32
    if remaining >= 4 {
        // SAFETY: by the invariant and `remaining >= 4`, the 4 bytes at `ptr`
        // lie inside `data`; `read_unaligned` needs no alignment.
        let value = unsafe { ptr.cast::<u32>().read_unaligned() };
        crc = _mm_crc32_u32(crc, value);
        // SAFETY: `remaining >= 4`, so `ptr + 4` is inside `data` or one
        // past its end.
        ptr = unsafe { ptr.add(4) };
        remaining -= 4;
    }

    // Process 2 bytes with _mm_crc32_u16
    if remaining >= 2 {
        // SAFETY: by the invariant and `remaining >= 2`, the 2 bytes at `ptr`
        // lie inside `data`; `read_unaligned` needs no alignment.
        let value = unsafe { ptr.cast::<u16>().read_unaligned() };
        crc = _mm_crc32_u16(crc, value);
        // SAFETY: `remaining >= 2`, so `ptr + 2` is inside `data` or one
        // past its end.
        ptr = unsafe { ptr.add(2) };
        remaining -= 2;
    }

    // Process remaining byte with _mm_crc32_u8
    if remaining == 1 {
        // SAFETY: by the invariant and `remaining == 1`, `ptr` points at the
        // last byte of `data`.
        crc = _mm_crc32_u8(crc, unsafe { *ptr });
    }

    crc
}

// ============================================================================
// ARM64 CRC Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
/// ARM64 CRC32C implementation using hardware CRC extensions
///
/// Feeds `data` to the `std::arch::aarch64` CRC32C intrinsics as
/// little-endian 8-, 4-, 2- and 1-byte words, the byte order of the
/// reflected CRC32C, so the result matches [`crc32c_scalar`].
///
/// # Safety
///
/// The CPU must support the ARMv8 CRC extension (target feature `crc`).
/// There are no requirements on `data`.
#[target_feature(enable = "crc")]
unsafe fn crc32c_arm(data: &[u8], mut crc: u32) -> u32 {
    // Process 8 bytes at a time with __crc32cd
    let (words, mut rest) = data.as_chunks::<8>();
    for word in words {
        crc = __crc32cd(crc, u64::from_le_bytes(*word));
    }

    // Process 4 bytes with __crc32cw
    if let Some((word, tail)) = rest.split_first_chunk::<4>() {
        crc = __crc32cw(crc, u32::from_le_bytes(*word));
        rest = tail;
    }

    // Process 2 bytes with __crc32ch
    if let Some((half, tail)) = rest.split_first_chunk::<2>() {
        crc = __crc32ch(crc, u16::from_le_bytes(*half));
        rest = tail;
    }

    // Process remaining byte with __crc32cb
    if let Some(&byte) = rest.first() {
        crc = __crc32cb(crc, byte);
    }

    crc
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
                // SAFETY: has_sse42 verified above
                let crc_sse42 = unsafe { crc32c_sse42(data, 0xFFFFFFFF) };
                assert_eq!(crc_scalar, crc_sse42, "SSE4.2 CRC mismatch");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let features = get_cpu_features();
            if features.has_crc32 {
                // SAFETY: has_crc32 verified above
                let crc_arm = unsafe { crc32c_arm(data, 0xFFFFFFFF) };
                assert_eq!(crc_scalar, crc_arm, "ARM CRC mismatch");
            }
        }
    }

    #[test]
    fn test_crc32c_different_sizes() {
        // Test various sizes to ensure proper handling
        for size in [
            0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024,
        ] {
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
        let n = if cfg!(miri) { 512 } else { 10000 };
        let data: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let crc = crc32c_hash(&data).unwrap();
        assert_ne!(crc, 0);

        // Verify consistency
        let crc2 = crc32c_hash(&data).unwrap();
        assert_eq!(crc, crc2);
    }

    /// C9 C1/C3: the hardware kernels against the table-driven code for
    /// every length 0..=64, eight start offsets (most inputs misaligned) and
    /// three initial values. The kernels are called directly behind the
    /// runtime feature check: under Miri `CpuFeatures` reports nothing, so
    /// `crc32c` itself always takes the scalar path there, while
    /// `is_x86_feature_detected!` reports the build's target features, so
    /// the Miri SIMD build runs `crc32c_sse42` here (C3). On aarch64 with the
    /// CRC extension the test runs `crc32c_arm` (C1).
    #[test]
    fn test_c9_c1_c3_hardware_kernels_match_scalar_for_lengths_0_to_64_and_offsets() {
        let buf: Vec<u8> = (0..72u32).map(|i| (i.wrapping_mul(151) ^ (i >> 3)) as u8).collect();
        let mut hardware_checked = 0usize;
        let lens: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 64]
        } else {
            (0..=64).collect()
        };
        for offset in 0..8 {
            for &len in &lens {
                let data = &buf[offset..offset + len];
                for init in [0u32, 0xFFFF_FFFF, 0x89AB_CDEF] {
                    let expected = crc32c_scalar(data, init);
                    let what = (offset, len, init);
                    assert_eq!(crc32c(data, init).unwrap(), expected, "crc32c {what:?}");
                    #[cfg(target_arch = "x86_64")]
                    {
                        if is_x86_feature_detected!("sse4.2") {
                            // SAFETY: SSE4.2 support was verified at runtime just above.
                            let got = unsafe { crc32c_sse42(data, init) };
                            assert_eq!(got, expected, "crc32c_sse42 {what:?}");
                            hardware_checked += 1;
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("crc") {
                            // SAFETY: the CRC extension was verified at runtime just above.
                            let got = unsafe { crc32c_arm(data, init) };
                            assert_eq!(got, expected, "crc32c_arm {what:?}");
                            hardware_checked += 1;
                        }
                    }
                }
            }
        }
        println!("C1/C3: hardware CRC32C kernel compared with scalar on {hardware_checked} inputs");
        assert_eq!(crc32c_hash(b"123456789").unwrap(), 0xE306_9283);
    }
}

#[cfg(all(test, not(miri)))]
mod benches {
    use super::*;
    use std::time::Instant;

    fn benchmark_crc32c(name: &str, data: &[u8], iterations: usize) {
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = crc32c_hash(data).expect("CRC32C hash computation");
        }

        let elapsed = start.elapsed();
        let bytes_processed = data.len() * iterations;
        let throughput_gbps = (bytes_processed as f64 / elapsed.as_secs_f64()) / 1_000_000_000.0;

        println!(
            "{}: {:.2} GB/s ({} iterations, {} bytes)",
            name,
            throughput_gbps,
            iterations,
            data.len()
        );
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
            // SAFETY: Test verified SSE4.2 available above
            unsafe {
                let _ = !crc32c_sse42(&data, 0xFFFFFFFF);
            }
        }
        let elapsed = start.elapsed();

        let bytes_processed = data.len() * 100_000;
        let throughput_gbps = (bytes_processed as f64 / elapsed.as_secs_f64()) / 1_000_000_000.0;

        println!("CRC32C SSE4.2 (1 KB): {:.2} GB/s", throughput_gbps);
    }
}
