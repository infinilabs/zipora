//! # SSE4.2 PCMPESTRI String Search
//!
//! String search with SSE4.2 PCMPESTRI kernels on x86_64 and scalar code
//! everywhere else.
//!
//! ## Architecture
//!
//! - **Two tiers**: SSE4.2 (x86_64, detected at runtime) and scalar. The
//!   [`SearchTier::AVX2`], [`SearchTier::AVX512`] and [`SearchTier::NEON`]
//!   variants are never selected; see [`SearchTier`].
//! - **PCMPESTRI Instructions**: `find_char`, `find_pattern` and
//!   `find_any_of` use PCMPESTRI; `compare_strings` uses SSE2 PCMPEQB.
//! - **Hybrid Strategy**: needles of up to 32 bytes are matched by their
//!   first `min(n, 8)` bytes with PCMPESTRI (17..=32-byte needles through
//!   their first 16 bytes), longer needles by their first byte; every
//!   candidate is compared in full.
//! - **In-bounds loads**: every 16-byte load lies inside the caller's
//!   slice; a window shorter than 16 bytes is copied into a zeroed stack
//!   buffer first.
//!
//! ## Performance
//!
//! The substring kernel runs one PCMPESTRI per 16-byte window on the first
//! `p = min(n, 8)` bytes of an `n`-byte needle. After a window without a
//! candidate it moves `17 - p` (at least 9) bytes on, and one byte past a
//! rejected candidate, instead of one byte per PCMPESTRI. Limitation: no
//! speedup over `memchr` or the scalar code is claimed; the ignored test
//! `test_c9_s4_strstr_timing_64k_absent_needle` prints timings for one
//! input on the machine that runs it.

use std::cmp::Ordering;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::system::cpu_features::get_cpu_features;

/// SIMD tier for string search operations
///
/// [`SimdStringSearch::tier`] reports [`SearchTier::SSE42`] or
/// [`SearchTier::Scalar`], the two sets of kernels this module has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchTier {
    /// Scalar fallback (portable)
    Scalar,
    /// SSE4.2 PCMPESTRI (x86_64)
    SSE42,
    /// AVX2 (x86_64). Limitation: never selected; this module has no AVX2
    /// kernels, so AVX2 hosts get [`SearchTier::SSE42`]. The variant is kept
    /// so that existing `match`es keep compiling.
    AVX2,
    /// AVX-512 (x86_64). Limitation: never selected; this module has no
    /// AVX-512 kernels. The variant is kept so that existing `match`es keep
    /// compiling.
    AVX512,
    /// ARM NEON (aarch64). Limitation: never selected; this module has no
    /// NEON kernels, so aarch64 hosts get [`SearchTier::Scalar`]. The variant
    /// is kept so that existing `match`es keep compiling.
    NEON,
}

/// Configuration for SIMD string search
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Allow the SSE4.2 PCMPESTRI kernels when the CPU has SSE4.2. With
    /// `false`, every operation runs the scalar code.
    pub enable_sse42: bool,
    /// Limitation: no effect; this module has no AVX2 kernels.
    pub enable_avx2: bool,
    /// Limitation: no effect; this module has no AVX-512 kernels.
    pub enable_avx512: bool,
    /// Limitation: no effect; this module has no NEON kernels.
    pub enable_neon: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            enable_sse42: true,
            enable_avx2: true,
            enable_avx512: true,
            enable_neon: true,
        }
    }
}

/// SIMD-accelerated string search operations
pub struct SimdStringSearch {
    tier: SearchTier,
}

impl SimdStringSearch {
    /// Create a new SIMD string search instance with runtime detection
    pub fn new() -> Self {
        Self::with_config(SearchConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SearchConfig) -> Self {
        let tier = Self::detect_optimal_tier(&config);
        Self { tier }
    }

    /// Picks [`SearchTier::SSE42`] when `config.enable_sse42` is set and the
    /// CPU has SSE4.2 (x86_64 only), [`SearchTier::Scalar`] otherwise.
    fn detect_optimal_tier(config: &SearchConfig) -> SearchTier {
        #[cfg(target_arch = "x86_64")]
        {
            if config.enable_sse42 && get_cpu_features().has_sse42 {
                return SearchTier::SSE42;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = config;

        SearchTier::Scalar
    }

    /// Get current SIMD tier
    pub fn tier(&self) -> SearchTier {
        self.tier
    }

    /// Find first occurrence of character in haystack
    ///
    /// # Performance
    ///
    /// - SSE4.2 tier: one PCMPESTRI per 16-byte chunk, scalar code for the
    ///   last `haystack.len() % 16` bytes
    /// - Scalar tier: `Iterator::position`
    ///
    /// # Example
    ///
    /// ```
    /// use zipora::io::simd_memory::search::SimdStringSearch;
    ///
    /// let searcher = SimdStringSearch::new();
    /// let haystack = b"hello world";
    /// let pos = searcher.find_char(haystack, b'w');
    /// assert_eq!(pos, Some(6));
    /// ```
    pub fn find_char(&self, haystack: &[u8], ch: u8) -> Option<usize> {
        #[cfg(target_arch = "x86_64")]
        {
            if self.tier == SearchTier::SSE42 {
                return sse42_strchr(haystack, ch);
            }
        }
        scalar_strchr(haystack, ch)
    }

    /// Find first occurrence of pattern in haystack
    ///
    /// # Performance
    ///
    /// - SSE4.2 tier: needles of up to 32 bytes are matched by their first
    ///   `min(n, 8)` bytes with one PCMPESTRI per 16-byte window, longer
    ///   needles by their first byte; candidates are compared in full
    /// - Scalar tier: `slice::windows`
    ///
    /// # Example
    ///
    /// ```
    /// use zipora::io::simd_memory::search::SimdStringSearch;
    ///
    /// let searcher = SimdStringSearch::new();
    /// let haystack = b"hello world";
    /// let needle = b"world";
    /// let pos = searcher.find_pattern(haystack, needle);
    /// assert_eq!(pos, Some(6));
    /// ```
    pub fn find_pattern(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if haystack.len() < needle.len() {
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.tier == SearchTier::SSE42 {
                return sse42_strstr(haystack, needle);
            }
        }
        scalar_strstr(haystack, needle)
    }

    /// Find any character from a set in haystack
    ///
    /// `chars` may have any length. On the SSE4.2 tier a set of up to 16
    /// bytes is matched with PCMPESTRI; a larger set is matched through a
    /// 256-entry membership table.
    ///
    /// # Example
    ///
    /// ```
    /// use zipora::io::simd_memory::search::SimdStringSearch;
    ///
    /// let searcher = SimdStringSearch::new();
    /// let haystack = b"hello world";
    /// let chars = b"aeiou";
    /// let pos = searcher.find_any_of(haystack, chars);
    /// assert_eq!(pos, Some(1)); // 'e' in "hello"
    /// ```
    pub fn find_any_of(&self, haystack: &[u8], chars: &[u8]) -> Option<usize> {
        if chars.is_empty() {
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.tier == SearchTier::SSE42 {
                return sse42_multi_search(haystack, chars);
            }
        }
        scalar_multi_search(haystack, chars)
    }

    /// Compare two strings lexicographically
    ///
    /// # Performance
    ///
    /// - SSE4.2 tier: SSE2 PCMPEQB over 16-byte chunks of the common prefix
    ///   (not PCMPESTRI); strings shorter than 16 bytes are compared by the
    ///   scalar loop
    /// - Scalar tier: `<[u8]>::cmp`
    ///
    /// # Example
    ///
    /// ```
    /// use zipora::io::simd_memory::search::SimdStringSearch;
    /// use std::cmp::Ordering;
    ///
    /// let searcher = SimdStringSearch::new();
    /// let result = searcher.compare_strings(b"abc", b"abd");
    /// assert_eq!(result, Ordering::Less);
    /// ```
    pub fn compare_strings(&self, s1: &[u8], s2: &[u8]) -> Ordering {
        #[cfg(target_arch = "x86_64")]
        {
            if self.tier == SearchTier::SSE42 {
                return sse42_strcmp(s1, s2);
            }
        }
        scalar_strcmp(s1, s2)
    }
}

impl Default for SimdStringSearch {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SSE4.2 PCMPESTRI Implementation
// ============================================================================

/// Find character using SSE4.2 PCMPESTRI
///
/// Uses _mm_cmpestri with _SIDD_CMP_EQUAL_ANY mode on each full 16-byte
/// chunk and scalar code for the remaining bytes. Runs the scalar search
/// when the CPU lacks SSE4.2.
#[cfg(target_arch = "x86_64")]
pub fn sse42_strchr(haystack: &[u8], ch: u8) -> Option<usize> {
    if !is_x86_feature_detected!("sse4.2") {
        return scalar_strchr(haystack, ch);
    }

    // SAFETY: SSE4.2 was detected at runtime just above; that is the only
    // precondition of `sse42_strchr_impl`.
    unsafe { sse42_strchr_impl(haystack, ch) }
}

/// SSE4.2 body of [`sse42_strchr`]. It loads only full 16-byte chunks that
/// lie inside `haystack` and scans the rest with scalar code.
///
/// # Safety
///
/// The CPU must support SSE4.2. There are no requirements on `haystack`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_strchr_impl(haystack: &[u8], ch: u8) -> Option<usize> {
    const CHUNK_SIZE: usize = 16;

    // Create needle vector with single character
    let mut needle_bytes = [0u8; 16];
    needle_bytes[0] = ch;
    // SAFETY: `needle_bytes` is a 16-byte stack array, so the unaligned
    // 16-byte load reads only its bytes.
    let needle = unsafe { _mm_loadu_si128(needle_bytes.as_ptr() as *const __m128i) };

    let mut offset = 0;
    let len = haystack.len();

    // Process 16-byte chunks
    while offset + CHUNK_SIZE <= len {
        // SAFETY: the loop condition gives `offset + 16 <= haystack.len()`,
        // so the unaligned 16-byte load stays inside `haystack`.
        let chunk = unsafe { _mm_loadu_si128(haystack.as_ptr().add(offset) as *const __m128i) };

        // _SIDD_CMP_EQUAL_ANY: Match any character in needle
        // _SIDD_UBYTE_OPS: Operate on unsigned bytes
        let idx = _mm_cmpestri(
            needle,
            1, // needle length = 1
            chunk,
            CHUNK_SIZE as i32,
            _SIDD_CMP_EQUAL_ANY | _SIDD_UBYTE_OPS,
        );

        if idx < CHUNK_SIZE as i32 {
            return Some(offset + idx as usize);
        }

        offset += CHUNK_SIZE;
    }

    // Handle remaining bytes with scalar
    haystack[offset..]
        .iter()
        .position(|&b| b == ch)
        .map(|i| offset + i)
}

/// Find pattern using SSE4.2 PCMPESTRI with hybrid strategy
///
/// Needles of up to 32 bytes are matched by their first `min(n, 8)` bytes
/// with PCMPESTRI (17..=32-byte needles through their first 16 bytes),
/// longer needles by their first byte; every candidate is compared in full.
/// Runs the scalar search when the CPU lacks SSE4.2.
#[cfg(target_arch = "x86_64")]
pub fn sse42_strstr(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if haystack.len() < needle.len() {
        return None;
    }
    if !is_x86_feature_detected!("sse4.2") {
        return scalar_strstr(haystack, needle);
    }

    // SAFETY: SSE4.2 was detected at runtime just above; that is the only
    // precondition of `sse42_strstr_impl`.
    unsafe { sse42_strstr_impl(haystack, needle) }
}

/// Picks the SSE4.2 substring kernel by needle length.
///
/// # Safety
///
/// The CPU must support SSE4.2. There are no requirements on the slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_strstr_impl(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    match needle.len() {
        0 => Some(0),
        // SAFETY: SSE4.2 support is this function's own precondition, and it
        // is the only precondition of `sse42_strstr_short`.
        1..=16 => unsafe { sse42_strstr_short(haystack, needle) },
        // SAFETY: as above, for `sse42_strstr_medium`.
        17..=32 => unsafe { sse42_strstr_medium(haystack, needle) },
        // SAFETY: as above, for `sse42_strstr_long`.
        _ => unsafe { sse42_strstr_long(haystack, needle) },
    }
}

/// Short-needle search (1..=16 bytes): PCMPESTRI in equal-ordered mode on
/// the first `p = min(n, 8)` bytes of the `n`-byte needle, over 16-byte
/// windows of `haystack`.
///
/// A window with 16 bytes left is loaded straight from `haystack`; the last
/// window, with fewer bytes left, is first copied into a zeroed stack
/// buffer, so no load reads past the end of the slice.
///
/// PCMPESTRI returns the first window offset at which the `p` prefix bytes
/// match. That candidate is compared against the whole needle; if it
/// fails, the next window starts one byte after it. A 16-byte window
/// without a candidate rules out the `17 - p` offsets at which the prefix
/// fits inside it, so the next window starts `17 - p` (at least 9) bytes
/// on. A shorter last window holds every remaining start position, so the
/// search ends there. Needles longer than 16 bytes go to [`scalar_strstr`].
///
/// The kernel does not depend on whether PCMPESTRI also reports a prefix
/// that runs off the end of a 16-byte window (hardware does, Miri does
/// not): such a report is one more candidate to compare.
///
/// # Safety
///
/// The CPU must support SSE4.2. There are no requirements on the slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_strstr_short(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    const WINDOW: usize = 16;
    // Needle bytes matched by PCMPESTRI; a longer prefix rules out fewer
    // offsets per window, a shorter one yields more false candidates.
    const MAX_PREFIX: usize = 8;
    let needle_len = needle.len();
    if needle_len == 0 {
        return Some(0);
    }
    if needle_len > WINDOW {
        return scalar_strstr(haystack, needle);
    }
    let last_start = haystack.len().checked_sub(needle_len)?;
    let prefix_len = needle_len.min(MAX_PREFIX);
    // Offsets 0..=WINDOW - prefix_len of a 16-byte window hold the whole
    // prefix; a window without a candidate rules all of them out.
    let miss_step = WINDOW - prefix_len + 1;

    // Load the prefix (padded with zeros to 16 bytes)
    let mut prefix_bytes = [0u8; WINDOW];
    prefix_bytes[..prefix_len].copy_from_slice(&needle[..prefix_len]);
    // SAFETY: `prefix_bytes` is a 16-byte stack array, so the unaligned
    // 16-byte load reads only its bytes.
    let prefix_vec = unsafe { _mm_loadu_si128(prefix_bytes.as_ptr() as *const __m128i) };

    let mut start = 0;
    while start <= last_start {
        let rest = &haystack[start..];
        let window_len = rest.len().min(WINDOW);
        let window = if window_len == WINDOW {
            // SAFETY: `rest` has at least 16 bytes (`window_len == 16`), so
            // the unaligned 16-byte load stays inside `haystack`.
            unsafe { _mm_loadu_si128(rest.as_ptr() as *const __m128i) }
        } else {
            let mut tail = [0u8; WINDOW];
            tail[..window_len].copy_from_slice(rest);
            // SAFETY: `tail` is a 16-byte stack array.
            unsafe { _mm_loadu_si128(tail.as_ptr() as *const __m128i) }
        };

        // _SIDD_CMP_EQUAL_ORDERED: Match ordered substring. The result is the
        // first candidate offset inside the window, or 16 for none.
        let idx = _mm_cmpestri(
            prefix_vec,
            prefix_len as i32,
            window,
            window_len as i32,
            _SIDD_CMP_EQUAL_ORDERED | _SIDD_UBYTE_OPS,
        ) as usize;

        if idx < window_len {
            let candidate = start + idx;
            if haystack.get(candidate..candidate + needle_len) == Some(needle) {
                return Some(candidate);
            }
            start = candidate + 1;
        } else if window_len == WINDOW {
            start += miss_step;
        } else {
            break;
        }
    }

    None
}

/// Medium pattern search (17-32 bytes): finds candidates for the first 16
/// needle bytes with [`sse42_strstr_short`], then compares the whole needle.
///
/// # Safety
///
/// The CPU must support SSE4.2. There are no requirements on the slices; a
/// needle of 16 bytes or fewer is passed to [`sse42_strstr_short`] whole.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_strstr_medium(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let needle_len = needle.len();
    let Some(first_16) = needle.get(..16) else {
        // SAFETY: SSE4.2 support is this function's own precondition.
        return unsafe { sse42_strstr_short(haystack, needle) };
    };
    let last_start = haystack.len().checked_sub(needle_len)?;
    // A prefix candidate at or before `last_start` ends at or before
    // `last_start + 16 <= haystack.len()`.
    let prefix_area = &haystack[..last_start + first_16.len()];

    let mut offset = 0;
    while offset <= last_start {
        // SAFETY: SSE4.2 support is this function's own precondition.
        let found = unsafe { sse42_strstr_short(&prefix_area[offset..], first_16) }?;
        let candidate = offset + found;
        if haystack.get(candidate..candidate + needle_len) == Some(needle) {
            return Some(candidate);
        }
        offset = candidate + 1;
    }

    None
}

/// Long pattern search (>32 bytes): finds candidates for the first needle
/// byte with [`sse42_strchr_impl`], then compares the whole needle.
///
/// # Safety
///
/// The CPU must support SSE4.2. There are no requirements on the slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_strstr_long(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let Some(&first_byte) = needle.first() else {
        return Some(0);
    };
    let last_start = haystack.len().checked_sub(needle.len())?;
    // Only positions up to `last_start` can start a match.
    let starts = &haystack[..=last_start];

    let mut offset = 0;
    while offset <= last_start {
        // SAFETY: SSE4.2 support is this function's own precondition, and it
        // is the only precondition of `sse42_strchr_impl`.
        let found = unsafe { sse42_strchr_impl(&starts[offset..], first_byte) }?;
        let candidate = offset + found;
        // Verify full pattern with scalar comparison
        if haystack.get(candidate..candidate + needle.len()) == Some(needle) {
            return Some(candidate);
        }
        offset = candidate + 1;
    }

    None
}

/// Find any character from set using SSE4.2
///
/// A set of up to 16 bytes is matched with PCMPESTRI in equal-any mode; a
/// larger set through a 256-entry membership table, because PCMPESTRI
/// compares against at most 16 set bytes. Runs the scalar search when the
/// CPU lacks SSE4.2.
#[cfg(target_arch = "x86_64")]
pub fn sse42_multi_search(haystack: &[u8], chars: &[u8]) -> Option<usize> {
    if chars.is_empty() {
        return None;
    }
    if !is_x86_feature_detected!("sse4.2") {
        return scalar_multi_search(haystack, chars);
    }

    // SAFETY: SSE4.2 was detected at runtime just above; that is the only
    // precondition of `sse42_multi_search_impl`.
    unsafe { sse42_multi_search_impl(haystack, chars) }
}

/// SSE4.2 body of [`sse42_multi_search`]. It loads only full 16-byte chunks
/// that lie inside `haystack` and scans the rest with scalar code.
///
/// # Safety
///
/// The CPU must support SSE4.2. There are no requirements on the slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_multi_search_impl(haystack: &[u8], chars: &[u8]) -> Option<usize> {
    const CHUNK_SIZE: usize = 16;
    if chars.len() > CHUNK_SIZE {
        // PCMPESTRI compares against at most 16 set bytes.
        return table_multi_search(haystack, chars);
    }
    let chars_len = chars.len();

    // Load character set
    let mut chars_bytes = [0u8; 16];
    chars_bytes[..chars_len].copy_from_slice(chars);
    // SAFETY: `chars_bytes` is a 16-byte stack array, so the unaligned
    // 16-byte load reads only its bytes.
    let chars_vec = unsafe { _mm_loadu_si128(chars_bytes.as_ptr() as *const __m128i) };

    let mut offset = 0;
    let len = haystack.len();

    while offset + CHUNK_SIZE <= len {
        // SAFETY: the loop condition gives `offset + 16 <= haystack.len()`,
        // so the unaligned 16-byte load stays inside `haystack`.
        let chunk = unsafe { _mm_loadu_si128(haystack.as_ptr().add(offset) as *const __m128i) };

        let idx = _mm_cmpestri(
            chars_vec,
            chars_len as i32,
            chunk,
            CHUNK_SIZE as i32,
            _SIDD_CMP_EQUAL_ANY | _SIDD_UBYTE_OPS,
        );

        if idx < CHUNK_SIZE as i32 {
            return Some(offset + idx as usize);
        }

        offset += CHUNK_SIZE;
    }

    // Handle remaining bytes
    haystack[offset..]
        .iter()
        .position(|&b| chars.contains(&b))
        .map(|i| offset + i)
}

/// Byte-set search through a 256-entry membership table, for sets that do
/// not fit in one PCMPESTRI operand.
#[cfg(target_arch = "x86_64")]
fn table_multi_search(haystack: &[u8], chars: &[u8]) -> Option<usize> {
    let mut member = [false; 256];
    for &c in chars {
        member[usize::from(c)] = true;
    }
    haystack.iter().position(|&b| member[usize::from(b)])
}

/// Compare strings with SSE2 PCMPEQB over 16-byte chunks
///
/// Despite the name it uses no PCMPESTRI: it is gated on SSE4.2 together
/// with the other kernels of this module. Runs the scalar comparison when
/// the CPU lacks SSE4.2.
#[cfg(target_arch = "x86_64")]
pub fn sse42_strcmp(s1: &[u8], s2: &[u8]) -> Ordering {
    if !is_x86_feature_detected!("sse4.2") {
        return scalar_strcmp(s1, s2);
    }

    // SAFETY: SSE4.2 was detected at runtime just above; that is the only
    // precondition of `sse42_strcmp_impl`.
    unsafe { sse42_strcmp_impl(s1, s2) }
}

/// SIMD body of [`sse42_strcmp`]. It loads only full 16-byte chunks of the
/// common prefix, which lie inside both slices.
///
/// # Safety
///
/// The CPU must support SSE4.2 (the function is compiled with that target
/// feature, although it executes only SSE2 instructions). There are no
/// requirements on the slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn sse42_strcmp_impl(s1: &[u8], s2: &[u8]) -> Ordering {
    const CHUNK_SIZE: usize = 16;
    let min_len = s1.len().min(s2.len());

    let mut offset = 0;

    // Compare 16-byte chunks
    while offset + CHUNK_SIZE <= min_len {
        // SAFETY: the loop condition gives `offset + 16 <= min_len <=
        // s1.len()`, so the unaligned 16-byte load stays inside `s1`.
        let chunk1 = unsafe { _mm_loadu_si128(s1.as_ptr().add(offset) as *const __m128i) };
        // SAFETY: likewise `offset + 16 <= min_len <= s2.len()`.
        let chunk2 = unsafe { _mm_loadu_si128(s2.as_ptr().add(offset) as *const __m128i) };

        // Check for difference using PCMPEQB
        let cmp = _mm_cmpeq_epi8(chunk1, chunk2);
        let mask = _mm_movemask_epi8(cmp);

        if mask != 0xFFFF {
            // Found difference - use scalar comparison to determine order
            for i in 0..CHUNK_SIZE {
                let b1 = s1[offset + i];
                let b2 = s2[offset + i];
                if b1 != b2 {
                    return b1.cmp(&b2);
                }
            }
        }

        offset += CHUNK_SIZE;
    }

    // Compare remaining bytes
    for i in offset..min_len {
        match s1[i].cmp(&s2[i]) {
            Ordering::Equal => continue,
            other => return other,
        }
    }

    s1.len().cmp(&s2.len())
}

// ============================================================================
// Scalar Fallback Implementation
// ============================================================================

/// Scalar character search fallback
pub fn scalar_strchr(haystack: &[u8], ch: u8) -> Option<usize> {
    haystack.iter().position(|&b| b == ch)
}

/// Scalar pattern search fallback
pub fn scalar_strstr(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if haystack.len() < needle.len() {
        return None;
    }

    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Scalar multi-character search fallback
pub fn scalar_multi_search(haystack: &[u8], chars: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| chars.contains(&b))
}

/// Scalar string comparison fallback
pub fn scalar_strcmp(s1: &[u8], s2: &[u8]) -> Ordering {
    s1.cmp(s2)
}

// ============================================================================
// Public convenience functions
// ============================================================================

/// Find character with automatic SIMD selection
pub fn find_char(haystack: &[u8], ch: u8) -> Option<usize> {
    SimdStringSearch::new().find_char(haystack, ch)
}

/// Find pattern with automatic SIMD selection
pub fn find_pattern(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    SimdStringSearch::new().find_pattern(haystack, needle)
}

/// Find any character from set with automatic SIMD selection
pub fn find_any_of(haystack: &[u8], chars: &[u8]) -> Option<usize> {
    SimdStringSearch::new().find_any_of(haystack, chars)
}

/// Compare strings with automatic SIMD selection
pub fn compare_strings(s1: &[u8], s2: &[u8]) -> Ordering {
    SimdStringSearch::new().compare_strings(s1, s2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_tier_detection() {
        let searcher = SimdStringSearch::new();
        let tier = searcher.tier();

        // Should detect some SIMD tier on modern hardware
        #[cfg(target_arch = "x86_64")]
        {
            // x86_64 should at least have SSE4.2 on modern CPUs
            assert!(matches!(
                tier,
                SearchTier::SSE42 | SearchTier::AVX2 | SearchTier::AVX512 | SearchTier::Scalar
            ));
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 should typically have NEON
            assert!(matches!(tier, SearchTier::NEON | SearchTier::Scalar));
        }
    }

    #[test]
    fn test_find_char_basic() {
        let searcher = SimdStringSearch::new();

        // Found cases
        assert_eq!(searcher.find_char(b"hello", b'h'), Some(0));
        assert_eq!(searcher.find_char(b"hello", b'e'), Some(1));
        assert_eq!(searcher.find_char(b"hello", b'o'), Some(4));

        // Not found
        assert_eq!(searcher.find_char(b"hello", b'x'), None);

        // Empty haystack
        assert_eq!(searcher.find_char(b"", b'a'), None);
    }

    #[test]
    fn test_find_char_long_string() {
        let searcher = SimdStringSearch::new();

        // Create long string to trigger SIMD path
        let haystack = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaax";
        assert_eq!(searcher.find_char(haystack, b'x'), Some(31));

        // Multiple occurrences - should find first
        let haystack = b"aaaaaxaaaaaxaaaaax";
        assert_eq!(searcher.find_char(haystack, b'x'), Some(5));
    }

    #[test]
    fn test_find_pattern_basic() {
        let searcher = SimdStringSearch::new();

        // Basic matches
        assert_eq!(searcher.find_pattern(b"hello world", b"world"), Some(6));
        assert_eq!(searcher.find_pattern(b"hello world", b"hello"), Some(0));
        assert_eq!(searcher.find_pattern(b"hello world", b"lo wo"), Some(3));

        // Not found
        assert_eq!(searcher.find_pattern(b"hello world", b"xyz"), None);

        // Empty needle
        assert_eq!(searcher.find_pattern(b"hello", b""), Some(0));

        // Needle longer than haystack
        assert_eq!(searcher.find_pattern(b"hi", b"hello"), None);
    }

    #[test]
    fn test_find_pattern_short() {
        let searcher = SimdStringSearch::new();

        // Short patterns (≤16 bytes) - single PCMPESTRI
        assert_eq!(searcher.find_pattern(b"abcdefgh", b"cde"), Some(2));
        assert_eq!(searcher.find_pattern(b"0123456789abcdef", b"789a"), Some(7));

        // Exact 16-byte pattern
        let haystack = b"prefix_0123456789abcdef_suffix";
        let needle = b"0123456789abcdef";
        assert_eq!(searcher.find_pattern(haystack, needle), Some(7));
    }

    #[test]
    fn test_find_pattern_medium() {
        let searcher = SimdStringSearch::new();

        // Medium patterns (17-32 bytes) - cascaded PCMPESTRI
        let needle = b"0123456789abcdefghij"; // 20 bytes
        let haystack = b"prefix_0123456789abcdefghij_suffix";
        assert_eq!(searcher.find_pattern(haystack, needle), Some(7));
    }

    #[test]
    fn test_find_pattern_long() {
        let searcher = SimdStringSearch::new();

        // Long patterns (>32 bytes) - chunked processing
        let needle = b"0123456789abcdefghijklmnopqrstuvwxyz"; // 36 bytes
        let haystack = b"prefix_0123456789abcdefghijklmnopqrstuvwxyz_suffix";
        assert_eq!(searcher.find_pattern(haystack, needle), Some(7));

        // Very long haystack
        let mut long_haystack = vec![b'a'; 1000];
        long_haystack.extend_from_slice(needle);
        long_haystack.extend_from_slice(b"_suffix");
        assert_eq!(searcher.find_pattern(&long_haystack, needle), Some(1000));
    }

    #[test]
    fn test_find_pattern_overlapping() {
        let searcher = SimdStringSearch::new();

        // Overlapping matches - should find first
        assert_eq!(searcher.find_pattern(b"aaaa", b"aa"), Some(0));
        assert_eq!(searcher.find_pattern(b"ababab", b"abab"), Some(0));
    }

    #[test]
    fn test_find_any_of_basic() {
        let searcher = SimdStringSearch::new();

        // Find vowels
        assert_eq!(searcher.find_any_of(b"hello", b"aeiou"), Some(1)); // 'e'
        assert_eq!(searcher.find_any_of(b"world", b"aeiou"), Some(1)); // 'o'

        // Not found
        assert_eq!(searcher.find_any_of(b"xyz", b"aeiou"), None);

        // Empty char set
        assert_eq!(searcher.find_any_of(b"hello", b""), None);
    }

    #[test]
    fn test_find_any_of_multiple() {
        let searcher = SimdStringSearch::new();

        // Multiple matching characters - should find first
        let haystack = b"abcdefghijklmnop";
        let chars = b"xyz";
        assert_eq!(searcher.find_any_of(haystack, chars), None);

        let chars = b"efg";
        assert_eq!(searcher.find_any_of(haystack, chars), Some(4)); // 'e'
    }

    #[test]
    fn test_compare_strings_basic() {
        let searcher = SimdStringSearch::new();

        // Equal
        assert_eq!(
            searcher.compare_strings(b"hello", b"hello"),
            Ordering::Equal
        );

        // Less than
        assert_eq!(searcher.compare_strings(b"abc", b"abd"), Ordering::Less);

        // Greater than
        assert_eq!(searcher.compare_strings(b"xyz", b"abc"), Ordering::Greater);

        // Different lengths
        assert_eq!(searcher.compare_strings(b"abc", b"abcd"), Ordering::Less);
        assert_eq!(searcher.compare_strings(b"abcd", b"abc"), Ordering::Greater);
    }

    #[test]
    fn test_compare_strings_long() {
        let searcher = SimdStringSearch::new();

        // Long identical strings
        let s1 = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let s2 = b"0123456789abcdefghijklmnopqrstuvwxyz";
        assert_eq!(searcher.compare_strings(s1, s2), Ordering::Equal);

        // Long strings with difference
        let s1 = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let s2 = b"0123456789abcdefghijklmnopqrstuvwxyZ"; // Last char different
        assert_eq!(searcher.compare_strings(s1, s2), Ordering::Greater); // 'z' > 'Z'
    }

    #[test]
    fn test_convenience_functions() {
        // Test public convenience functions
        assert_eq!(find_char(b"hello", b'e'), Some(1));
        assert_eq!(find_pattern(b"hello world", b"world"), Some(6));
        assert_eq!(find_any_of(b"hello", b"aeiou"), Some(1));
        assert_eq!(compare_strings(b"abc", b"abc"), Ordering::Equal);
    }

    #[test]
    fn test_scalar_fallback() {
        // Test scalar implementations directly
        assert_eq!(scalar_strchr(b"hello", b'e'), Some(1));
        assert_eq!(scalar_strstr(b"hello world", b"world"), Some(6));
        assert_eq!(scalar_multi_search(b"hello", b"aeiou"), Some(1));
        assert_eq!(scalar_strcmp(b"abc", b"abc"), Ordering::Equal);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_sse42_implementation() {
        if !is_x86_feature_detected!("sse4.2") {
            println!("SSE4.2 not available, skipping test");
            return;
        }

        // Test SSE4.2 implementations directly
        assert_eq!(sse42_strchr(b"hello world", b'w'), Some(6));
        assert_eq!(sse42_strstr(b"hello world", b"world"), Some(6));
        assert_eq!(sse42_multi_search(b"hello", b"aeiou"), Some(1));
        assert_eq!(sse42_strcmp(b"abc", b"abc"), Ordering::Equal);
    }

    #[test]
    fn test_edge_cases() {
        let searcher = SimdStringSearch::new();

        // Single character haystack
        assert_eq!(searcher.find_char(b"a", b'a'), Some(0));
        assert_eq!(searcher.find_char(b"a", b'b'), None);

        // Single character pattern
        assert_eq!(searcher.find_pattern(b"hello", b"h"), Some(0));

        // Pattern at end
        assert_eq!(searcher.find_pattern(b"hello", b"lo"), Some(3));

        // All same characters
        assert_eq!(searcher.find_char(b"aaaa", b'a'), Some(0));
        assert_eq!(searcher.find_pattern(b"aaaa", b"aa"), Some(0));
    }

    #[test]
    fn test_custom_config() {
        // Test with custom configuration
        let config = SearchConfig {
            enable_sse42: false,
            enable_avx2: false,
            enable_avx512: false,
            enable_neon: false,
        };

        let searcher = SimdStringSearch::with_config(config);
        assert_eq!(searcher.tier(), SearchTier::Scalar);

        // Should still work correctly with scalar fallback
        assert_eq!(searcher.find_char(b"hello", b'e'), Some(1));
        assert_eq!(searcher.find_pattern(b"hello world", b"world"), Some(6));
    }

    /// Reference substring search for the C9 differential tests.
    fn reference_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    /// Checks every substring search reachable from this module against
    /// `reference_find`. `what` is formatted only when an assertion fails.
    fn assert_strstr_agrees(haystack: &[u8], needle: &[u8], what: (&str, usize, usize, usize)) {
        let expected = reference_find(haystack, needle);
        assert_eq!(scalar_strstr(haystack, needle), expected, "scalar_strstr {what:?}");
        assert_eq!(find_pattern(haystack, needle), expected, "find_pattern {what:?}");
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.2") {
                assert_eq!(sse42_strstr(haystack, needle), expected, "sse42_strstr {what:?}");
            }
        }
    }

    /// Deterministic text over {a, b}: short needles occur often and windows
    /// end in partial matches of every length.
    fn ab_text(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b'a' + (state & 1) as u8
            })
            .collect()
    }

    /// C9 S1: haystacks shorter than 16 bytes and absent needles of 1..=16
    /// bytes. The SSE4.2 short-needle kernel must not read past the end of
    /// the haystack (each haystack is its own exact-size heap allocation, so
    /// Miri and ASan see any over-read).
    #[test]
    fn test_c9_s1_sse42_strstr_short_haystacks_absent_needles_stay_in_bounds() {
        for len in 1..=20usize {
            let haystack = vec![b'a'; len];
            for n in 1..=len.min(16) {
                let needle = vec![b'b'; n];
                assert_eq!(find_pattern(&haystack, &needle), None, "len {len} n {n}");
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("sse4.2") {
                        assert_eq!(sse42_strstr(&haystack, &needle), None, "len {len} n {n}");
                    }
                }
            }
        }
        // The `find_pattern` doc example: an 11-byte haystack.
        let haystack = b"hello world".to_vec();
        assert_eq!(find_pattern(&haystack, b"world"), Some(6));
        assert_eq!(find_pattern(&haystack, b"worlds"), None);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.2") {
                assert_eq!(sse42_strstr(&haystack, b"world"), Some(6));
                assert_eq!(sse42_strstr(&haystack, b"worlds"), None);
            }
        }
    }

    /// C9 S1/S4: the windowed SSE4.2 search returns what a naive search
    /// returns. Haystack lengths 0..=64, needle lengths 1..=17; each needle is
    /// planted at offset 0, at 15 and at 17 - n (across the 15/16 window
    /// boundary), at len - n, or left out. Two backgrounds: all 'a' with the
    /// needle a^(n-1)b, so every 16-byte window ends in a partial match, and
    /// a pseudo-random a/b text. Hardware PCMPESTRI reports a needle that
    /// runs off the end of a 16-byte window, Miri's does not, so the Miri run
    /// checks that the kernel does not depend on those reports.
    #[test]
    fn test_c9_s1_strstr_matches_reference_search() {
        for len in 0..=64usize {
            for n in 1..=17usize {
                let starts = [0, 15, 17usize.saturating_sub(n), len.saturating_sub(n)];

                let mut needle = vec![b'a'; n];
                needle[n - 1] = b'b';
                let background = vec![b'a'; len];
                assert_strstr_agrees(&background, &needle, ("a-background absent", len, n, 0));
                for &at in &starts {
                    if at + n <= len {
                        let mut haystack = background.clone();
                        haystack[at..at + n].copy_from_slice(&needle);
                        assert_strstr_agrees(&haystack, &needle, ("a-background", len, n, at));
                    }
                }

                let text = ab_text(len, (len * 64 + n) as u64);
                for &at in &starts {
                    if at + n <= len {
                        let present = text[at..at + n].to_vec();
                        assert_strstr_agrees(&text, &present, ("ab-text", len, n, at));
                    }
                }
                let mut absent = needle.clone();
                absent[n - 1] = b'c';
                assert_strstr_agrees(&text, &absent, ("ab-text absent", len, n, 0));
            }
        }
    }

    /// C9 S1 (native, Linux): the haystack ends exactly where a `PROT_NONE`
    /// page begins, so any load past its end faults. The searches run in a
    /// child process (this test binary, re-run with `PROBE_ENV` set and
    /// `--exact`), so a fault shows up as a failed assertion here instead of
    /// killing the whole test run. Miri cannot run `mmap`/`mprotect`, and the
    /// in-bounds property is checked under Miri by the tests above.
    #[cfg(all(target_os = "linux", not(miri)))]
    #[test]
    fn test_c9_s1_search_ending_at_protected_page_does_not_fault() {
        const PROBE_ENV: &str = "ZIPORA_C9_S1_PAGE_END_PROBE";
        const NAME: &str = "test_c9_s1_search_ending_at_protected_page_does_not_fault";
        if std::env::var_os(PROBE_ENV).is_some() {
            page_end_probe();
            return;
        }
        let module = module_path!();
        let module = module.split_once("::").map_or(module, |(_, rest)| rest);
        let exact_name = format!("{module}::{NAME}");
        let exe = std::env::current_exe().expect("path of the test binary");
        let output = std::process::Command::new(exe)
            .args([exact_name.as_str(), "--exact", "--test-threads=1", "--nocapture"])
            .env(PROBE_ENV, "1")
            .output()
            .expect("spawn the page-end probe");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "page-end probe child failed: {}\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}",
            output.status
        );
    }

    /// Child-process body of the page-end test: every search entry point on
    /// a haystack whose last byte is the last byte of a readable page.
    #[cfg(all(target_os = "linux", not(miri)))]
    fn page_end_probe() {
        // SAFETY: `sysconf` has no preconditions.
        let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .expect("page size");
        let map_len = 2 * page;
        // SAFETY: anonymous private mapping with no address hint and no fd.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED, "mmap failed");
        let base = base.cast::<u8>();
        // SAFETY: `base + page` is the page-aligned start of the second page
        // of the mapping created above.
        let rc = unsafe { libc::mprotect(base.add(page).cast(), page, libc::PROT_NONE) };
        assert_eq!(rc, 0, "mprotect failed");
        // SAFETY: the first page is mapped read/write and `page` bytes long.
        unsafe { base.write_bytes(b'a', page) };
        let hay_len = page.min(4096);
        {
            // SAFETY: the last `hay_len` bytes of the first page are mapped,
            // initialised above, and not written while `haystack` is alive;
            // the mapping is unmapped only after this block.
            let haystack: &[u8] =
                unsafe { std::slice::from_raw_parts(base.add(page - hay_len), hay_len) };
            for n in 1..=40usize {
                let absent = vec![b'b'; n];
                let mut near_miss = vec![b'a'; n];
                near_miss[n - 1] = b'b';
                assert_eq!(find_pattern(haystack, &absent), None, "absent n {n}");
                assert_eq!(find_pattern(haystack, &near_miss), None, "near miss n {n}");
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("sse4.2") {
                        assert_eq!(sse42_strstr(haystack, &absent), None, "absent n {n}");
                        assert_eq!(sse42_strstr(haystack, &near_miss), None, "near miss n {n}");
                    }
                }
            }
            assert_eq!(find_char(haystack, b'b'), None);
            assert_eq!(find_any_of(haystack, b"bcd"), None);
            assert_eq!(find_any_of(haystack, b"0123456789bcdefghij"), None);
            assert_eq!(compare_strings(haystack, haystack), Ordering::Equal);
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse4.2") {
                    assert_eq!(sse42_strchr(haystack, b'b'), None);
                    assert_eq!(sse42_multi_search(haystack, b"bcd"), None);
                    assert_eq!(sse42_strcmp(haystack, haystack), Ordering::Equal);
                }
            }
        }
        // SAFETY: `base`/`map_len` describe the mapping created above, and no
        // reference into it outlives the block above.
        let rc = unsafe { libc::munmap(base.cast(), map_len) };
        assert_eq!(rc, 0, "munmap failed");
    }

    /// C9 S4: time per search over a 64 KiB haystack that does not contain
    /// the needle. A measurement, not a pass/fail check; run it with
    /// `cargo test --release --lib -- --ignored --nocapture test_c9_s4`.
    #[test]
    #[ignore = "timing measurement; run with --release -- --ignored --nocapture"]
    fn test_c9_s4_strstr_timing_64k_absent_needle() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const ITERS: u32 = 200;
        fn per_search(f: &dyn Fn() -> Option<usize>) -> Duration {
            let start = Instant::now();
            for _ in 0..ITERS {
                black_box(f());
            }
            start.elapsed() / ITERS
        }

        let haystack: Vec<u8> = (0..65_536u32).map(|i| b'a' + (i % 23) as u8).collect();
        for n in [1usize, 2, 4, 8, 12, 16] {
            let mut needle: Vec<u8> = (0..n).map(|i| b'a' + (i % 23) as u8).collect();
            needle[n - 1] = b'Z'; // never in the haystack
            let scalar = per_search(&|| scalar_strstr(black_box(&haystack), black_box(&needle)));
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse4.2") {
                    assert_eq!(sse42_strstr(&haystack, &needle), None);
                    let sse = per_search(&|| sse42_strstr(black_box(&haystack), black_box(&needle)));
                    println!("needle {n:2} B: sse42_strstr {sse:?}/search, scalar_strstr {scalar:?}/search");
                    continue;
                }
            }
            println!("needle {n:2} B: scalar_strstr {scalar:?}/search (no SSE4.2)");
        }
    }

    /// C9 S3: needles of 17..=32 bytes take `sse42_strstr_medium`, longer
    /// ones `sse42_strstr_long`. Needles are planted at the edges of 16-byte
    /// windows, after a decoy that matches the first 16 bytes only, or left
    /// out.
    #[test]
    fn test_c9_s3_sse42_strstr_medium_and_long_needles_match_reference() {
        for n in 17..=40usize {
            let mut needle: Vec<u8> = (0..n).map(|i| b'a' + (i % 7) as u8).collect();
            needle[n - 1] = b'z';
            let mut decoy = needle.clone();
            decoy[n - 1] = b'y';
            for len in [n - 1, n, n + 1, n + 15, n + 16, n + 17, 96] {
                let background = vec![b'a'; len];
                assert_strstr_agrees(&background, &needle, ("absent", len, n, 0));
                for at in [0usize, 1, 15, 16, 31, 32, len.saturating_sub(n)] {
                    if at + n <= len {
                        let mut haystack = background.clone();
                        haystack[at..at + n].copy_from_slice(&needle);
                        assert_strstr_agrees(&haystack, &needle, ("planted", len, n, at));
                    }
                }
                if 2 * n < len {
                    let mut haystack = background.clone();
                    haystack[..n].copy_from_slice(&decoy);
                    haystack[n + 1..2 * n + 1].copy_from_slice(&needle);
                    assert_strstr_agrees(&haystack, &needle, ("after decoy", len, n, n + 1));
                }
                let text = ab_text(len, (len * 64 + n) as u64);
                assert_strstr_agrees(&text, &needle, ("ab-text absent", len, n, 0));
                if n <= len {
                    let at = len - n;
                    let present = text[at..].to_vec();
                    assert_strstr_agrees(&text, &present, ("ab-text", len, n, at));
                }
            }
        }
    }

    /// Checks every byte-set search reachable from this module against a
    /// naive scan. `what` is formatted only when an assertion fails.
    fn assert_any_of_agrees(haystack: &[u8], chars: &[u8], what: (&str, usize, usize, usize)) {
        let expected = haystack.iter().position(|b| chars.contains(b));
        assert_eq!(scalar_multi_search(haystack, chars), expected, "scalar {what:?}");
        assert_eq!(find_any_of(haystack, chars), expected, "find_any_of {what:?}");
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.2") {
                assert_eq!(sse42_multi_search(haystack, chars), expected, "sse42 {what:?}");
            }
        }
    }

    /// C9 S2: the auditor's inputs. The 17th byte of the set ('X') must count
    /// both inside a full 16-byte chunk and before a hit in the tail.
    #[test]
    fn test_c9_s2_find_any_of_uses_every_byte_of_a_17_byte_set() {
        let chars = b"0123456789abcdefX";
        assert_eq!(scalar_multi_search(&[b'X'; 16], chars), Some(0));
        assert_eq!(find_any_of(&[b'X'; 16], chars), Some(0));
        assert_eq!(find_any_of(b"XXXXXXXXXXXXXXXXa", chars), Some(0));
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.2") {
                assert_eq!(sse42_multi_search(&[b'X'; 16], chars), Some(0));
                assert_eq!(sse42_multi_search(b"XXXXXXXXXXXXXXXXa", chars), Some(0));
            }
        }
    }

    /// C9 S2: sets of 1..=40 distinct bytes. The planted member is the
    /// first, the middle or the last one of the set (the last is past the
    /// 16th byte once the set has more than 16), at the edges of 16-byte
    /// chunks or in the tail. Also sets with duplicates, a NUL member and all
    /// 256 byte values.
    #[test]
    fn test_c9_s2_find_any_of_matches_scalar_for_set_sizes_1_to_40() {
        let (lens, spots): (&[usize], &[usize]) = if cfg!(miri) {
            (&[0, 15, 16, 17, 33], &[0, 15, 16, usize::MAX])
        } else {
            (
                &[0, 1, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65],
                &[0, 1, 15, 16, 17, 31, 32, usize::MAX],
            )
        };
        let filler = b'~'; // outside every set below
        for k in 1..=40usize {
            let set: Vec<u8> = (0..k).map(|i| b'!' + i as u8).collect();
            for &len in lens {
                let background = vec![filler; len];
                assert_any_of_agrees(&background, &set, ("absent", k, len, 0));
                for member in [set[0], set[k / 2], set[k - 1]] {
                    for &spot in spots {
                        let at = if spot == usize::MAX { len.wrapping_sub(1) } else { spot };
                        if at < len {
                            let mut haystack = background.clone();
                            haystack[at] = member;
                            assert_any_of_agrees(&haystack, &set, ("planted", k, len, at));
                        }
                    }
                }
            }
        }

        let mut duplicates = vec![b'q'; 20];
        duplicates.push(b'Q');
        let mut haystack = vec![filler; 40];
        haystack[37] = b'Q';
        assert_any_of_agrees(&haystack, &duplicates, ("duplicates", 21, 40, 37));
        let mut with_nul: Vec<u8> = (b'A'..=b'T').collect();
        with_nul.push(0);
        let mut haystack = vec![filler; 40];
        haystack[20] = 0;
        assert_any_of_agrees(&haystack, &with_nul, ("nul member", 21, 40, 20));
        let all: Vec<u8> = (0..=255u8).collect();
        assert_any_of_agrees(&[filler; 33], &all, ("all bytes", 256, 33, 0));
    }

    /// C9 S5/S6: `tier()` names the kernels that actually run. This module
    /// has SSE4.2 kernels and scalar code only, so no other tier is reported.
    #[test]
    fn test_c9_s5_tier_names_the_kernels_that_run() {
        let tier = SimdStringSearch::new().tier();
        #[cfg(target_arch = "x86_64")]
        {
            let expected = if get_cpu_features().has_sse42 {
                SearchTier::SSE42
            } else {
                SearchTier::Scalar
            };
            assert_eq!(tier, expected);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            assert_eq!(tier, SearchTier::Scalar);
        }
    }

    /// C9 S5: `enable_sse42: false` must keep SSE4.2 code from running, even
    /// with the other flags at their defaults.
    #[test]
    fn test_c9_s5_enable_sse42_false_selects_scalar() {
        let searcher = SimdStringSearch::with_config(SearchConfig {
            enable_sse42: false,
            ..SearchConfig::default()
        });
        assert_eq!(searcher.tier(), SearchTier::Scalar);
        assert_eq!(searcher.find_char(b"hello", b'l'), Some(2));
        assert_eq!(searcher.find_pattern(b"hello world", b"world"), Some(6));
        assert_eq!(searcher.find_any_of(b"hello", b"ol"), Some(2));
        assert_eq!(searcher.compare_strings(b"abc", b"abd"), Ordering::Less);
    }
}
