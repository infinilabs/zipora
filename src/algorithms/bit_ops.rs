//! SIMD-accelerated bit operations.
//!
//! Provides high-performance population count over `&[u64]` slices,
//! automatically selecting the fastest available implementation:
//! AVX-512 VPOPCNTDQ → AVX2 vpshufb (Mula) → hardware POPCNT → NEON → scalar.
//!
//! # Examples
//!
//! ```rust
//! use zipora::algorithms::bit_ops::popcount_slice;
//!
//! let words = [0xFFu64, 0xFF00, 0];
//! assert_eq!(popcount_slice(&words), 16); // 8 + 8 + 0
//!
//! assert_eq!(popcount_slice(&[]), 0);
//! assert_eq!(popcount_slice(&[u64::MAX]), 64);
//! ```

/// Minimum words to justify SIMD setup overhead.
/// Below this threshold, scalar `count_ones()` is faster.
const SIMD_THRESHOLD: usize = 16;

/// SIMD-accelerated population count over a u64 slice.
///
/// Returns the total number of set bits across all words.
/// Automatically selects the fastest available implementation:
///
/// | Tier | Platform | Method | Throughput |
/// |------|----------|--------|------------|
/// | 0 | AVX-512 (x86_64, avx512 feature) | `_mm512_popcnt_epi64` | ~64 words/cycle |
/// | 1 | POPCNT (x86_64) | Unrolled `_popcnt64` 4× | ~4 words/cycle |
/// | 2 | AVX2 (x86_64) | vpshufb nibble lookup (Mula) | >20 words/cycle |
/// | 3 | NEON (aarch64) | `vcntq_u8` + horizontal sum | ~8 words/cycle |
/// | 4 | Scalar | `u64::count_ones()` | ~1 word/cycle |
///
/// Note: POPCNT is checked before AVX2 because all CPUs with AVX2 also have
/// POPCNT, and hardware `popcnt` is faster than vpshufb nibble-lookup.
///
/// For slices shorter than 16 words (128 bytes), skips SIMD setup and uses
/// scalar directly, as the overhead exceeds the benefit.
///
/// Handles both aligned and unaligned input. When the pointer is 32-byte
/// aligned, uses aligned loads for better throughput on AVX2.
#[inline]
pub fn popcount_slice(words: &[u64]) -> usize {
    if words.len() < SIMD_THRESHOLD {
        return popcount_scalar(words);
    }

    #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512vpopcntdq")
            && std::arch::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: AVX-512 VPOPCNTDQ support verified by runtime feature check.
            return unsafe { popcount_avx512(words) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // POPCNT before AVX2: on CPUs with both (Haswell+), hardware POPCNT
        // beats vpshufb nibble-lookup because popcnt is single-cycle per word.
        // AVX2 vpshufb is only useful on (theoretical) CPUs with AVX2 but no POPCNT.
        if std::arch::is_x86_feature_detected!("popcnt") {
            // SAFETY: POPCNT support verified by runtime feature check.
            return unsafe { popcount_hw(words) };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support verified by runtime feature check.
            return unsafe { popcount_avx2(words) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        return unsafe { popcount_neon(words) };
    }

    popcount_scalar(words)
}

// ============================================================================
// Tier 4: Scalar fallback (always available)
// ============================================================================

/// Scalar popcount using `u64::count_ones()`.
#[inline]
fn popcount_scalar(words: &[u64]) -> usize {
    words.iter().map(|w| w.count_ones() as usize).sum()
}

// ============================================================================
// Tier 0: AVX-512 VPOPCNTDQ (nightly + avx512 feature)
// ============================================================================

#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn popcount_avx512(words: &[u64]) -> usize {
    use std::arch::x86_64::*;

    let chunks = words.len() / 8;

    // SAFETY: All intrinsics safe under AVX-512 guarantee from #[target_feature].
    // Pointer arithmetic bounded by chunks = words.len() / 8.
    unsafe {
        let ptr = words.as_ptr() as *const __m512i;
        let mut acc = _mm512_setzero_si512();

        for i in 0..chunks {
            let v = _mm512_loadu_si512(ptr.add(i));
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(v));
        }

        // Horizontal sum: extract 8 × u64 and sum
        let mut buf = [0u64; 8];
        _mm512_storeu_si512(buf.as_mut_ptr() as *mut _, acc);
        let mut sum: usize = buf.iter().sum::<u64>() as usize;

        // Scalar tail (0..7 remaining words)
        for &w in &words[chunks * 8..] {
            sum += w.count_ones() as usize;
        }
        sum
    }
}

// ============================================================================
// Tier 1: AVX2 vpshufb — Mula's nibble-lookup algorithm
// ============================================================================
//
// Processes 4 u64 words (32 bytes) per iteration:
// 1. Split each byte into low/high nibbles
// 2. vpshufb: nibble → popcount via precomputed 16-entry LUT
// 3. Accumulate byte counts with vpaddb
// 4. Every 31 iterations, reduce with vpsadbw to prevent u8 overflow
//    (max accumulated value per byte lane after 31 iters: 31 × 8 = 248 ≤ 255)
// 5. Final horizontal sum of u64 accumulators

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn popcount_avx2(words: &[u64]) -> usize {
    use std::arch::x86_64::*;

    let bytes = words.as_ptr() as *const u8;
    let total_bytes = words.len() * 8;
    let chunks = total_bytes / 32; // 32 bytes per AVX2 register

    // SAFETY: All intrinsics below are safe under AVX2 guarantee from #[target_feature].
    // Pointer arithmetic is bounded by chunks = total_bytes / 32.
    unsafe {
        // Nibble → popcount lookup table: LUT[nibble] = popcount(nibble)
        let lut = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        );
        let lo_mask = _mm256_set1_epi8(0x0F);
        let mut total_acc = _mm256_setzero_si256(); // u64 accumulators
        let mut local_acc = _mm256_setzero_si256(); // u8 accumulators
        let mut since_reduce = 0u32;

        for i in 0..chunks {
            // Every 31 iterations, reduce u8 accumulators to u64 to prevent overflow
            since_reduce += 1;
            if since_reduce == 31 {
                total_acc = _mm256_add_epi64(
                    total_acc,
                    _mm256_sad_epu8(local_acc, _mm256_setzero_si256()),
                );
                local_acc = _mm256_setzero_si256();
                since_reduce = 0;
            }

            // Always use unaligned loads — on modern CPUs (Haswell+),
            // loadu has no penalty when data is naturally aligned.
            let v = _mm256_loadu_si256(bytes.add(i * 32) as *const __m256i);

            // Split bytes into nibbles and look up popcount
            let lo = _mm256_and_si256(v, lo_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask);

            let popcnt_lo = _mm256_shuffle_epi8(lut, lo);
            let popcnt_hi = _mm256_shuffle_epi8(lut, hi);

            local_acc = _mm256_add_epi8(local_acc, _mm256_add_epi8(popcnt_lo, popcnt_hi));
        }

        // Final reduction of remaining local_acc
        total_acc = _mm256_add_epi64(
            total_acc,
            _mm256_sad_epu8(local_acc, _mm256_setzero_si256()),
        );

        // Horizontal sum of 4 × u64 accumulators
        let lo128 = _mm256_castsi256_si128(total_acc);
        let hi128 = _mm256_extracti128_si256(total_acc, 1);
        let sum128 = _mm_add_epi64(lo128, hi128);
        let hi64 = _mm_unpackhi_epi64(sum128, sum128);
        let total = _mm_add_epi64(sum128, hi64);
        let mut sum = _mm_cvtsi128_si64(total) as usize;

        // Scalar tail (0..3 remaining words from incomplete 32-byte chunk)
        let processed_bytes = chunks * 32;
        let remaining_words = (total_bytes - processed_bytes) / 8;
        let tail_start = processed_bytes / 8;
        for &w in &words[tail_start..tail_start + remaining_words] {
            sum += w.count_ones() as usize;
        }

        sum
    }
}

// ============================================================================
// Tier 2: Hardware POPCNT instruction (x86_64)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn popcount_hw(words: &[u64]) -> usize {
    use std::arch::x86_64::_popcnt64;

    // Simple iterator loop — LLVM will auto-unroll and pipeline.
    // This matches LLVM's own optimization of count_ones() on POPCNT-capable
    // CPUs, but guarantees hardware POPCNT regardless of -C target-cpu.
    let mut sum: usize = 0;
    for &w in words {
        // SAFETY: POPCNT guaranteed by #[target_feature(enable = "popcnt")].
        sum += unsafe { _popcnt64(w as i64) } as usize;
    }
    sum
}

// ============================================================================
// Tier 3: NEON (aarch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn popcount_neon(words: &[u64]) -> usize {
    use std::arch::aarch64::*;

    let chunks = words.len() / 2; // 2 × u64 = 128 bits per NEON register

    // SAFETY: All intrinsics safe under NEON guarantee (always available on aarch64).
    // Pointer arithmetic bounded by chunks = words.len() / 2.
    unsafe {
        let mut acc = vdupq_n_u64(0);

        for i in 0..chunks {
            let base = i * 2;
            let v = vld1q_u64(words.as_ptr().add(base));
            // vcntq_u8: popcount per byte, then pairwise add to u64
            let byte_counts = vcntq_u8(vreinterpretq_u8_u64(v));
            let pair_sums = vpaddlq_u8(byte_counts);   // u8 → u16
            let quad_sums = vpaddlq_u16(pair_sums);     // u16 → u32
            let oct_sums = vpaddlq_u32(quad_sums);      // u32 → u64
            acc = vaddq_u64(acc, oct_sums);
        }

        // Extract horizontal sum
        let sum = vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);
        let mut total = sum as usize;

        // Scalar tail
        for &w in &words[chunks * 2..] {
            total += w.count_ones() as usize;
        }

        total
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_slice() {
        assert_eq!(popcount_slice(&[]), 0);
    }

    #[test]
    fn test_single_word() {
        assert_eq!(popcount_slice(&[0]), 0);
        assert_eq!(popcount_slice(&[1]), 1);
        assert_eq!(popcount_slice(&[u64::MAX]), 64);
        assert_eq!(popcount_slice(&[0xFF]), 8);
        assert_eq!(popcount_slice(&[0xAAAA_AAAA_AAAA_AAAA]), 32);
    }

    #[test]
    fn test_all_zeros() {
        let words = vec![0u64; 100];
        assert_eq!(popcount_slice(&words), 0);
    }

    #[test]
    fn test_all_ones() {
        let words = vec![u64::MAX; 100];
        assert_eq!(popcount_slice(&words), 6400);
    }

    #[test]
    fn test_matches_scalar_small() {
        // Below SIMD_THRESHOLD — exercises scalar path
        for len in 0..SIMD_THRESHOLD {
            let words: Vec<u64> = (0..len as u64).map(|i| i.wrapping_mul(0x1234_5678_9ABC_DEF0)).collect();
            let expected: usize = words.iter().map(|w| w.count_ones() as usize).sum();
            assert_eq!(popcount_slice(&words), expected, "mismatch at len={len}");
        }
    }

    #[test]
    fn test_matches_scalar_simd_range() {
        // At and above SIMD_THRESHOLD — exercises SIMD paths
        for len in [16, 17, 31, 32, 33, 63, 64, 100, 127, 128, 255, 256, 500, 1000] {
            let words: Vec<u64> = (0..len as u64)
                .map(|i| i.wrapping_mul(0xDEAD_BEEF_CAFE_BABE).wrapping_add(i))
                .collect();
            let expected: usize = words.iter().map(|w| w.count_ones() as usize).sum();
            assert_eq!(popcount_slice(&words), expected, "mismatch at len={len}");
        }
    }

    #[test]
    fn test_alternating_bits() {
        let words = vec![0x5555_5555_5555_5555u64; 64]; // every other bit set
        assert_eq!(popcount_slice(&words), 64 * 32);
    }

    #[test]
    fn test_single_bit_per_word() {
        let words: Vec<u64> = (0..64).map(|i| 1u64 << i).collect();
        assert_eq!(popcount_slice(&words), 64);
    }

    #[test]
    fn test_boundary_at_31_iterations() {
        // 31 iterations × 4 words = 124 words — exactly at the AVX2 reduction boundary
        let words = vec![u64::MAX; 124];
        assert_eq!(popcount_slice(&words), 124 * 64);

        // 125 words — crosses the reduction boundary
        let words = vec![u64::MAX; 125];
        assert_eq!(popcount_slice(&words), 125 * 64);
    }

    #[test]
    fn test_avx2_reduction_overflow_boundary() {
        // Mula's algorithm reduces every 31 iterations to prevent u8 overflow.
        // Test at exact multiples of 31 × 4 words = 124 word boundaries.
        for n in [124, 248, 372, 496] {
            let words = vec![u64::MAX; n];
            assert_eq!(popcount_slice(&words), n * 64, "mismatch at n={n}");
        }
    }

    #[test]
    fn test_large_slice() {
        // 10K words (~80KB) — exercises sustained SIMD processing
        let words: Vec<u64> = (0..10_000u64)
            .map(|i| i.wrapping_mul(0x0123_4567_89AB_CDEF))
            .collect();
        let expected: usize = words.iter().map(|w| w.count_ones() as usize).sum();
        assert_eq!(popcount_slice(&words), expected);
    }

    #[test]
    fn test_tier_consistency() {
        // All tiers must produce identical results
        let words: Vec<u64> = (0..256u64)
            .map(|i| i.wrapping_mul(0xFEDC_BA98_7654_3210).wrapping_add(i * 17))
            .collect();
        let scalar = popcount_scalar(&words);
        let dispatch = popcount_slice(&words);
        assert_eq!(dispatch, scalar, "dispatch vs scalar mismatch");

        // Also test the internal SIMD functions directly where available
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let avx2 = unsafe { popcount_avx2(&words) };
                assert_eq!(avx2, scalar, "AVX2 vs scalar mismatch");
            }
            if std::arch::is_x86_feature_detected!("popcnt") {
                let hw = unsafe { popcount_hw(&words) };
                assert_eq!(hw, scalar, "POPCNT vs scalar mismatch");
            }
        }
    }

    /// Verify popcount_slice gives correct results for the typical union counting
    /// workload: a bitset of ~50K doc_ids (781 words) with scattered set bits.
    #[test]
    fn test_union_counting_workload() {
        let num_words = (50_000 >> 6) + 1; // 782 words
        let mut bits = vec![0u64; num_words];

        // Scatter 1000 doc_ids
        let doc_ids: Vec<u32> = (0..1000).map(|i| (i * 47) % 50_000).collect();
        for &doc_id in &doc_ids {
            let w = doc_id as usize >> 6;
            let b = doc_id as usize & 63;
            bits[w] |= 1u64 << b;
        }

        let expected: usize = bits.iter().map(|w| w.count_ones() as usize).sum();
        assert_eq!(popcount_slice(&bits), expected);
    }
}

#[cfg(test)]
#[cfg(feature = "simd")]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Benchmark popcount_slice across different sizes.
    /// Only meaningful in --release mode.
    #[test]
    fn bench_popcount_slice_throughput() {
        if cfg!(debug_assertions) {
            eprintln!("Skipping benchmark in debug mode");
            return;
        }

        let sizes = [16, 100, 781, 1_000, 10_000];
        let iterations = 100_000;

        for &size in &sizes {
            let words: Vec<u64> = (0..size as u64)
                .map(|i| i.wrapping_mul(0xDEAD_BEEF_CAFE_BABE))
                .collect();

            // Warmup
            let mut sink = 0usize;
            for _ in 0..1000 {
                sink += popcount_slice(&words);
            }

            let start = Instant::now();
            for _ in 0..iterations {
                sink += popcount_slice(&words);
            }
            let elapsed = start.elapsed();

            let ns_per_call = elapsed.as_nanos() as f64 / iterations as f64;
            let words_per_ns = size as f64 / ns_per_call;
            eprintln!(
                "popcount_slice({size:>6} words = {:>6} bytes): {ns_per_call:>8.1} ns/call, \
                 {words_per_ns:.2} words/ns ({:.0} Mwords/s) [sink={sink}]",
                size * 8,
                words_per_ns * 1000.0,
            );
        }
    }

    /// Compare SIMD popcount_slice vs scalar to measure speedup.
    #[test]
    fn bench_popcount_simd_vs_scalar() {
        if cfg!(debug_assertions) {
            eprintln!("Skipping benchmark in debug mode");
            return;
        }

        let size = 1000;
        let iterations = 200_000;
        let words: Vec<u64> = (0..size as u64)
            .map(|i| i.wrapping_mul(0xCAFE_BABE_DEAD_BEEF))
            .collect();

        let mut sink = 0usize;

        // Warmup
        for _ in 0..1000 {
            sink += popcount_scalar(&words);
            sink += popcount_slice(&words);
        }

        // Scalar
        let start = Instant::now();
        for _ in 0..iterations {
            sink += popcount_scalar(&words);
        }
        let scalar_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        // SIMD dispatch
        let start = Instant::now();
        for _ in 0..iterations {
            sink += popcount_slice(&words);
        }
        let simd_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        let speedup = scalar_ns / simd_ns;
        eprintln!(
            "popcount {size} words: scalar={scalar_ns:.1}ns, simd={simd_ns:.1}ns, \
             speedup={speedup:.1}× [sink={sink}]"
        );
    }
}
