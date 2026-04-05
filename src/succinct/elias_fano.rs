//! Elias-Fano Encoding — quasi-succinct monotone integer sequence.
//!
//! The gold standard for compressed posting lists in information retrieval.
//! Uses ~2 + log(u/n) bits per element where u = universe size, n = count.
//!
//! - O(1) random access via `get(i)`
//! - O(1) amortized `next_geq(target)` via rank/select on high bits
//! - Immutable after construction (build once, query many)
//!
//! # Examples
//!
//! ```rust
//! use zipora::succinct::elias_fano::EliasFano;
//!
//! let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
//! let ef = EliasFano::from_sorted(&docs);
//!
//! assert_eq!(ef.len(), 8);
//! assert_eq!(ef.get(0), Some(3));
//! assert_eq!(ef.get(7), Some(63));
//!
//! // next_geq: find first element >= target (core posting list operation)
//! assert_eq!(ef.next_geq(10), Some((2, 11)));   // index=2, value=11
//! assert_eq!(ef.next_geq(42), Some((5, 42)));   // exact match
//! assert_eq!(ef.next_geq(64), None);             // past end
//! ```

/// Sampling rate for rank acceleration.
/// One cumulative popcount sample per RANK_SAMPLE_RATE words (8 words = 512 bits).
/// Memory: 4 bytes per 64 bytes of high_bits = 6.25% overhead.
const RANK_SAMPLE_RATE: usize = 8;

/// Sampling rate for select1 acceleration.
/// One position sample per SELECT1_SAMPLE_RATE set bits.
/// Memory: 4 bytes per 256 elements ≈ negligible.
const SELECT1_SAMPLE_RATE: usize = 256;

/// Sampling rate for select0 acceleration.
/// Denser than select1 because `next_geq` uses select0 on the hot path.
/// At 64, max post-sample scan = 1 word (vs 4 words at 256).
/// Memory: 4 bytes per 64 zero-bits — still negligible.
const SELECT0_SAMPLE_RATE: usize = 64;

use crate::algorithms::bit_ops::select_in_word;

/// Elias-Fano encoded monotone integer sequence.
///
/// Layout:
/// - `low_bits`: packed array of the lower L bits of each element
/// - `high_bits`: unary-coded upper bits in a bitvector (1 for value, 0 for bucket boundary)
/// - Sampling indices for O(1) rank/select on high_bits
/// - L = floor(log2(universe / n)), chosen to minimize total space
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EliasFano {
    /// Lower L bits of each element, packed consecutively
    low_bits: Vec<u64>,
    /// Upper bits in unary: position i has a 1-bit for each element in bucket i
    /// Total length = n + (max >> low_bit_width) + 1
    high_bits: Vec<u64>,
    /// Number of elements
    len: usize,
    /// Bit width of low part
    low_bit_width: u32,
    /// Universe upper bound (exclusive)
    universe: u64,
    /// Total number of valid bits in high_bits
    high_len_bits: usize,
    /// Cumulative popcount at every RANK_SAMPLE_RATE words.
    /// rank_samples[i] = popcount(high_bits[0..i*RANK_SAMPLE_RATE])
    rank_samples: Vec<u32>,
    /// Position of every SELECT1_SAMPLE_RATE-th set bit.
    select1_samples: Vec<u32>,
    /// Position of every SELECT0_SAMPLE_RATE-th clear bit (denser for next_geq).
    select0_samples: Vec<u32>,
}

impl EliasFano {
    /// Build from a sorted slice of u32 values. Zero-allocation — processes
    /// u32 values directly without converting to an intermediate `Vec<u64>`.
    pub fn from_sorted(values: &[u32]) -> Self {
        if values.is_empty() {
            return Self::from_sorted_u64(&[]);
        }
        // Process u32 directly: cast on-the-fly, no intermediate Vec<u64>
        let universe = values[values.len() - 1] as u64 + 1;
        Self::from_sorted_impl(values.len(), universe, |i| values[i] as u64)
    }

    /// Build from sorted u64 values.
    pub fn from_sorted_u64(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self {
                low_bits: Vec::new(),
                high_bits: Vec::new(),
                len: 0,
                low_bit_width: 0,
                universe: 0,
                high_len_bits: 0,
                rank_samples: Vec::new(),
                select1_samples: Vec::new(),
                select0_samples: Vec::new(),
            };
        }
        let universe = values[values.len() - 1] + 1;
        Self::from_sorted_impl(values.len(), universe, |i| values[i])
    }

    /// Shared construction logic. `get_val(i)` returns the i-th value as u64.
    fn from_sorted_impl(n: usize, universe: u64, get_val: impl Fn(usize) -> u64) -> Self {
        // L = floor(log2(universe / n)), at least 0
        let low_bit_width = if n >= universe as usize {
            0
        } else {
            (64 - (universe / n as u64).leading_zeros()).saturating_sub(1)
        };

        let low_mask = if low_bit_width == 0 { 0u64 } else { (1u64 << low_bit_width) - 1 };

        // Pack low bits (+1 padding word for branchless u128 extraction)
        let total_low_bits = n as u64 * low_bit_width as u64;
        let low_words = ((total_low_bits + 63) / 64) as usize;
        let mut low_bits = vec![0u64; low_words + 1];

        for i in 0..n {
            if low_bit_width > 0 {
                let low_val = get_val(i) & low_mask;
                let bit_pos = i as u64 * low_bit_width as u64;
                let word_idx = (bit_pos / 64) as usize;
                let bit_idx = (bit_pos % 64) as u32;

                low_bits[word_idx] |= low_val << bit_idx;
                if bit_idx + low_bit_width > 64 && word_idx + 1 < low_words {
                    low_bits[word_idx + 1] |= low_val >> (64 - bit_idx);
                }
            }
        }

        // Build high bits in unary encoding
        let last_val = get_val(n - 1);
        let max_high = last_val >> low_bit_width;
        let high_len_bits = n + max_high as usize + 1;
        let high_words = (high_len_bits + 63) / 64;
        let mut high_bits = vec![0u64; high_words];

        let mut pos = 0usize;
        let mut prev_high = 0u64;

        for i in 0..n {
            let high = get_val(i) >> low_bit_width;
            pos += (high - prev_high) as usize;
            let word_idx = pos / 64;
            let bit_idx = pos % 64;
            if word_idx < high_words {
                high_bits[word_idx] |= 1u64 << bit_idx;
            }
            pos += 1;
            prev_high = high;
        }

        let (rank_samples, select1_samples, select0_samples) =
            Self::build_samples(&high_bits, high_len_bits);

        Self {
            low_bits,
            high_bits,
            len: n,
            low_bit_width,
            universe,
            high_len_bits,
            rank_samples,
            select1_samples,
            select0_samples,
        }
    }

    /// Build rank and select sampling tables from high_bits in a single pass.
    ///
    /// - `rank_samples[i]` = popcount(high_bits[0 .. i*RANK_SAMPLE_RATE*64])
    /// - `select1_samples[k]` = word-start bit-pos containing the (k*S)-th 1-bit
    /// - `select0_samples[k]` = word-start bit-pos containing the (k*S)-th 0-bit
    fn build_samples(high_bits: &[u64], high_len_bits: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut rank_samples = Vec::new();
        let mut select1_samples = Vec::new();
        let mut select0_samples = Vec::new();

        let mut cumul_ones: u32 = 0;
        let mut cumul_zeros: u32 = 0;
        // Next select threshold to record
        let mut next_sel1: u32 = 0;
        let mut next_sel0: u32 = 0;

        for (word_idx, &word) in high_bits.iter().enumerate() {
            // Rank: sample at block boundaries
            if word_idx % RANK_SAMPLE_RATE == 0 {
                rank_samples.push(cumul_ones);
            }

            let ones = word.count_ones();
            let valid = if (word_idx + 1) * 64 <= high_len_bits {
                64u32
            } else {
                (high_len_bits - word_idx * 64) as u32
            };
            let zeros = valid - ones;

            // Select1: record word position for each threshold crossed
            let after_ones = cumul_ones + ones;
            while next_sel1 < after_ones {
                select1_samples.push((word_idx * 64) as u32);
                next_sel1 += SELECT1_SAMPLE_RATE as u32;
            }

            // Select0: denser sampling for next_geq hot path
            let after_zeros = cumul_zeros + zeros;
            while next_sel0 < after_zeros {
                select0_samples.push((word_idx * 64) as u32);
                next_sel0 += SELECT0_SAMPLE_RATE as u32;
            }

            cumul_ones = after_ones;
            cumul_zeros = after_zeros;
        }

        // Final rank sentinel
        rank_samples.push(cumul_ones);

        (rank_samples, select1_samples, select0_samples)
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize { self.len }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Memory usage in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.low_bits.len() * 8
            + self.high_bits.len() * 8
            + self.rank_samples.len() * 4
            + self.select1_samples.len() * 4
            + self.select0_samples.len() * 4
            + 32 // metadata fields
    }

    /// Bits per element.
    #[inline]
    pub fn bits_per_element(&self) -> f64 {
        if self.len == 0 { return 0.0; }
        (self.size_bytes() * 8) as f64 / self.len as f64
    }

    /// Get the i-th element. O(1) via select1 on high bits.
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len { return None; }

        let low = self.get_low(index);
        let high_pos = self.select1(index)?;

        // high_position = high_value + index (unary encoding)
        // so high_value = high_position - index
        let high_val = high_pos.checked_sub(index)? as u64;

        Some((high_val << self.low_bit_width) | low)
    }

    /// Find the first element >= target. Returns (index, value).
    ///
    /// Uses select0 to jump to the target bucket, then scans high_bits
    /// directly (no `select1` or `get()` calls). Each element costs only
    /// `get_low` (1-2 word reads) + bit scan (~1 cycle).
    #[inline]
    pub fn next_geq(&self, target: u64) -> Option<(usize, u64)> {
        if self.len == 0 || target >= self.universe { return None; }

        let target_high = target >> self.low_bit_width;

        // Jump to bucket via select0
        let bucket_start_pos = if target_high == 0 {
            0
        } else {
            match self.select0(target_high as usize - 1) {
                Some(p) => p + 1,
                None => return None,
            }
        };

        let elem_before = self.rank1(bucket_start_pos);

        // Scan high_bits directly from bucket_start_pos — no get()/select1 calls
        let mut idx = elem_before;
        let mut word_idx = bucket_start_pos / 64;
        if word_idx >= self.high_bits.len() { return None; }

        // Start scanning from the bit within the first word
        let start_bit = bucket_start_pos % 64;
        let mut word = self.high_bits[word_idx] >> start_bit;
        let mut bit_pos = bucket_start_pos;

        while idx < self.len {
            if word == 0 {
                word_idx += 1;
                if word_idx >= self.high_bits.len() { return None; }
                word = self.high_bits[word_idx];
                bit_pos = word_idx * 64;
                continue;
            }

            // Skip to the next set bit (trailing zeros = 0-bits to skip)
            let tz = word.trailing_zeros() as usize;
            bit_pos += tz;
            word >>= tz;

            // This is a 1-bit — element `idx` at high_pos = bit_pos
            let high_val = (bit_pos - idx) as u64;
            let low = self.get_low(idx);
            let val = (high_val << self.low_bit_width) | low;

            if val >= target {
                return Some((idx, val));
            }

            // Advance past this 1-bit
            idx += 1;
            bit_pos += 1;
            word >>= 1;
        }

        None
    }

    /// Iterator over all elements.
    pub fn iter(&self) -> EliasFanoIter<'_> {
        EliasFanoIter { ef: self, pos: 0, high_pos: 0 }
    }

    // --- Internal helpers ---

    /// Get the low bits of the i-th element.
    /// Branchless: always loads 2 words via u128, padding word ensures safety.
    #[inline]
    fn get_low(&self, index: usize) -> u64 {
        if self.low_bit_width == 0 { return 0; }

        let bit_pos = index as u64 * self.low_bit_width as u64;
        let word_idx = (bit_pos / 64) as usize;
        let bit_idx = (bit_pos % 64) as u32;

        // Branchless u128 extraction: always reads 2 words (padding word at end
        // ensures word_idx+1 is valid). Compiles to shrd on x86-64.
        let w0 = self.low_bits[word_idx];
        let w1 = self.low_bits[word_idx + 1];
        let combined = w0 as u128 | ((w1 as u128) << 64);
        (combined >> bit_idx) as u64 & ((1u64 << self.low_bit_width) - 1)
    }

    /// Select1(rank): find position of the rank-th set bit (0-indexed).
    /// O(1) via sampling + rank: jump to neighbourhood, compute exact offset, scan ≤ S/64 words.
    #[inline]
    fn select1(&self, rank: usize) -> Option<usize> {
        if rank >= self.len { return None; }

        // Jump to sample: select1_samples[k] = word-start of k*S-th 1-bit
        let sample_idx = rank / SELECT1_SAMPLE_RATE;
        let start_word = if sample_idx < self.select1_samples.len() {
            self.select1_samples[sample_idx] as usize / 64
        } else {
            0
        };

        // Use rank_samples to compute exact cumulative ones at start_word
        let rank_block = start_word / RANK_SAMPLE_RATE;
        let mut cumul = self.rank_samples[rank_block] as usize;
        for w in (rank_block * RANK_SAMPLE_RATE)..start_word {
            cumul += self.high_bits[w].count_ones() as usize;
        }
        let mut remaining = rank - cumul;

        // Scan forward from start_word
        for word_idx in start_word..self.high_bits.len() {
            let word = self.high_bits[word_idx];
            let ones = word.count_ones() as usize;
            if remaining < ones {
                return Some(word_idx * 64 + select_in_word(word, remaining));
            }
            remaining -= ones;
        }
        None
    }

    /// Select0(rank): find position of the rank-th clear bit (0-indexed).
    /// O(1) via sampling + rank.
    #[inline]
    fn select0(&self, rank: usize) -> Option<usize> {
        let sample_idx = rank / SELECT0_SAMPLE_RATE;
        let start_word = if sample_idx < self.select0_samples.len() {
            self.select0_samples[sample_idx] as usize / 64
        } else {
            0
        };

        // Compute exact cumulative zeros at start_word using rank_samples
        let rank_block = start_word / RANK_SAMPLE_RATE;
        let cumul_ones = {
            let mut c = self.rank_samples[rank_block] as usize;
            for w in (rank_block * RANK_SAMPLE_RATE)..start_word {
                c += self.high_bits[w].count_ones() as usize;
            }
            c
        };
        // Total bits up to start_word
        let total_bits_before = start_word * 64;
        let cumul_zeros = total_bits_before - cumul_ones;
        let mut remaining = rank - cumul_zeros;

        for word_idx in start_word..self.high_bits.len() {
            let word = self.high_bits[word_idx];
            let valid_bits = if (word_idx + 1) * 64 <= self.high_len_bits {
                64
            } else {
                self.high_len_bits - word_idx * 64
            };
            let actual_zeros = valid_bits - word.count_ones() as usize;

            if remaining < actual_zeros {
                let inverted = !word;
                return Some(word_idx * 64 + select_in_word(inverted, remaining));
            }
            remaining -= actual_zeros;
        }
        None
    }

    /// Rank1(pos): count set bits in high_bits[0..pos).
    /// O(1) via cumulative popcount sampling.
    #[inline]
    fn rank1(&self, pos: usize) -> usize {
        let full_words = pos / 64;
        let remaining_bits = pos % 64;

        // Jump to nearest sample
        let sample_idx = full_words / RANK_SAMPLE_RATE;
        let mut count = self.rank_samples[sample_idx] as usize;

        // Scan at most RANK_SAMPLE_RATE words from the sample
        let scan_start = sample_idx * RANK_SAMPLE_RATE;
        for i in scan_start..full_words {
            count += self.high_bits[i].count_ones() as usize;
        }

        // Handle partial word
        if remaining_bits > 0 && full_words < self.high_bits.len() {
            let mask = (1u64 << remaining_bits) - 1;
            count += (self.high_bits[full_words] & mask).count_ones() as usize;
        }
        count
    }
}

/// Iterator over Elias-Fano encoded elements.
///
/// Uses `trailing_zeros` to skip 0-bits in bulk — O(1) amortized per element,
/// same technique as `EliasFanoCursor`.
pub struct EliasFanoIter<'a> {
    ef: &'a EliasFano,
    pos: usize,
    high_pos: usize,
}

impl<'a> Iterator for EliasFanoIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.ef.len { return None; }

        // Find next 1-bit using trailing_zeros (bulk skip over 0-bits)
        let next_pos = self.high_pos;
        let mut word_idx = next_pos / 64;
        if word_idx >= self.ef.high_bits.len() { return None; }

        // Mask out bits below current position within the first word
        let bit_offset = next_pos % 64;
        let mut word = self.ef.high_bits[word_idx] >> bit_offset;

        if word != 0 {
            let tz = word.trailing_zeros() as usize;
            self.high_pos = word_idx * 64 + bit_offset + tz;
        } else {
            // Scan subsequent words
            loop {
                word_idx += 1;
                if word_idx >= self.ef.high_bits.len() { return None; }
                word = self.ef.high_bits[word_idx];
                if word != 0 {
                    self.high_pos = word_idx * 64 + word.trailing_zeros() as usize;
                    break;
                }
            }
        }

        let high_val = (self.high_pos - self.pos) as u64;
        let low = self.ef.get_low(self.pos);
        let val = (high_val << self.ef.low_bit_width) | low;
        self.pos += 1;
        self.high_pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ef.len - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for EliasFanoIter<'a> {}

/// Stateful cursor for O(1) amortized sequential access over Elias-Fano.
///
/// Tracks position in `high_bits` so advancing to the next element costs
/// only one `trailing_zeros` + one `get_low` — no `select1` calls.
///
/// # Examples
///
/// ```rust
/// use zipora::succinct::elias_fano::EliasFano;
///
/// let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
/// let ef = EliasFano::from_sorted(&docs);
/// let mut cursor = ef.cursor();
///
/// assert_eq!(cursor.current(), Some(3));
/// assert!(cursor.advance());
/// assert_eq!(cursor.current(), Some(5));
///
/// // Jump to first element >= 30
/// assert!(cursor.advance_to_geq(30));
/// assert_eq!(cursor.current(), Some(31));
/// ```
pub struct EliasFanoCursor<'a> {
    ef: &'a EliasFano,
    /// Current element index (0..len).
    index: usize,
    /// Current bit position in high_bits (position of the current 1-bit).
    high_pos: usize,
}

impl<'a> EliasFanoCursor<'a> {
    /// Create a cursor positioned at the first element.
    fn new(ef: &'a EliasFano) -> Self {
        if ef.is_empty() {
            return Self { ef, index: 0, high_pos: 0 };
        }
        // Find the first 1-bit in high_bits
        let mut high_pos = 0;
        for (word_idx, &word) in ef.high_bits.iter().enumerate() {
            if word != 0 {
                high_pos = word_idx * 64 + word.trailing_zeros() as usize;
                break;
            }
        }
        Self { ef, index: 0, high_pos }
    }

    /// Current element value. O(1) — no select needed.
    #[inline]
    pub fn current(&self) -> Option<u64> {
        if self.index >= self.ef.len { return None; }
        let high_val = (self.high_pos - self.index) as u64;
        let low = self.ef.get_low(self.index);
        Some((high_val << self.ef.low_bit_width) | low)
    }

    /// Current element index.
    #[inline]
    pub fn index(&self) -> usize { self.index }

    /// Whether the cursor is past the last element.
    #[inline]
    pub fn is_exhausted(&self) -> bool { self.index >= self.ef.len }

    /// Advance to the next element. O(1) amortized — just find the next 1-bit.
    #[inline]
    pub fn advance(&mut self) -> bool {
        self.index += 1;
        if self.index >= self.ef.len { return false; }

        // Move past current 1-bit
        let next_pos = self.high_pos + 1;
        let mut word_idx = next_pos / 64;
        if word_idx >= self.ef.high_bits.len() { return false; }

        // Mask out bits at or below current position within the word
        let bit_in_word = next_pos % 64;
        let mut word = self.ef.high_bits[word_idx] >> bit_in_word;

        // Find next 1-bit
        if word != 0 {
            self.high_pos = word_idx * 64 + bit_in_word + word.trailing_zeros() as usize;
            return true;
        }

        // Scan subsequent words
        loop {
            word_idx += 1;
            if word_idx >= self.ef.high_bits.len() { return false; }
            word = self.ef.high_bits[word_idx];
            if word != 0 {
                self.high_pos = word_idx * 64 + word.trailing_zeros() as usize;
                return true;
            }
        }
    }

    /// Advance to the first element >= target.
    ///
    /// If the current element is already >= target, does nothing and returns true.
    /// Otherwise, jumps to the target bucket via `select0` and scans high_bits
    /// directly — maintaining `high_pos` inline with NO redundant `select1` call.
    #[inline]
    pub fn advance_to_geq(&mut self, target: u64) -> bool {
        // Fast path: current is already >= target
        if let Some(val) = self.current() {
            if val >= target { return true; }
        } else {
            return false;
        }

        if target >= self.ef.universe { self.index = self.ef.len; return false; }

        let target_high = target >> self.ef.low_bit_width;

        // Jump to target bucket via select0
        let bucket_start_pos = if target_high == 0 {
            0
        } else {
            match self.ef.select0(target_high as usize - 1) {
                Some(p) => p + 1,
                None => { self.index = self.ef.len; return false; }
            }
        };

        let elem_before = self.ef.rank1(bucket_start_pos);

        // Scan high_bits directly — maintaining index + high_pos inline
        let mut idx = elem_before;
        let mut word_idx = bucket_start_pos / 64;
        if word_idx >= self.ef.high_bits.len() { self.index = self.ef.len; return false; }

        let start_bit = bucket_start_pos % 64;
        let mut word = self.ef.high_bits[word_idx] >> start_bit;
        let mut bit_pos = bucket_start_pos;

        while idx < self.ef.len {
            if word == 0 {
                word_idx += 1;
                if word_idx >= self.ef.high_bits.len() { break; }
                word = self.ef.high_bits[word_idx];
                bit_pos = word_idx * 64;
                continue;
            }

            let tz = word.trailing_zeros() as usize;
            bit_pos += tz;
            word >>= tz;

            // This is element `idx` at high_pos = bit_pos
            let high_val = (bit_pos - idx) as u64;
            let low = self.ef.get_low(idx);
            let val = (high_val << self.ef.low_bit_width) | low;

            if val >= target {
                // Found it — update cursor state directly
                self.index = idx;
                self.high_pos = bit_pos;
                return true;
            }

            idx += 1;
            bit_pos += 1;
            word >>= 1;
        }

        self.index = self.ef.len;
        false
    }

    /// Reposition the cursor directly to element at `idx`.
    /// After this call, `self.index() == idx` and `self.current()` returns `ef.get(idx)`.
    /// Subsequent `advance()` and `advance_to_geq()` continue from this position.
    ///
    /// Returns false (cursor unchanged) if `idx >= ef.len()`.
    #[inline]
    pub fn advance_to_index(&mut self, idx: usize) -> bool {
        if idx >= self.ef.len { return false; }
        // Fast path: same position
        if idx == self.index { return true; }
        // Use select1 to find the bit position of the idx-th 1-bit in high_bits
        if let Some(pos) = self.ef.select1(idx) {
            self.index = idx;
            self.high_pos = pos;
            true
        } else {
            false
        }
    }

    /// Reset cursor to the first element.
    pub fn reset(&mut self) {
        *self = Self::new(self.ef);
    }
}

impl EliasFano {
    /// Create a stateful cursor for O(1) sequential access.
    #[inline]
    pub fn cursor(&self) -> EliasFanoCursor<'_> {
        EliasFanoCursor::new(self)
    }
}

// ============================================================================
// Partitioned Elias-Fano (Uniform 128-element chunks)
// ============================================================================

/// Chunk size for partitioned Elias-Fano. 128 elements per chunk fits well
/// in L1 cache (typically 32-64 KB) and matches standard block-based index sizes.
const PEF_CHUNK_SIZE: usize = 128;

/// Lightweight metadata for a chunk in flat contiguous storage.
///
/// 24 bytes per chunk vs 80+ bytes for the old per-chunk `Vec<u64>` layout.
/// Stores word-offsets into shared flat arrays, eliminating pointer chasing —
/// all chunk data lives in two contiguous `Vec<u64>` arrays owned by the
/// parent `PartitionedEliasFano` or `OptimalPartitionedEliasFano`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct PefChunkMeta {
    /// Minimum value in the chunk (base for delta encoding).
    min_value: u64,
    /// Offset (in u64 words) into the parent's `all_low_bits` array.
    low_offset: u32,
    /// Offset (in u64 words) into the parent's `all_high_bits` array.
    high_offset: u32,
    /// Number of elements in this chunk (1..=512 for OPEF, 1..=128 for PEF).
    count: u16,
    /// Total valid bits in this chunk's high_bits region.
    high_len_bits: u16,
    /// Pre-computed number of u64 words in this chunk's high_bits region.
    high_words: u16,
    /// Pre-computed number of u64 words in this chunk's low_bits region (excluding padding).
    low_words: u16,
    /// Bit width of the low part for this chunk (0..=64).
    low_bit_width: u8,
}

/// Lightweight borrowing view into a chunk's data from flat arrays.
/// Created on the stack for each query — zero heap allocation.
struct ChunkView<'a> {
    low_bits: &'a [u64],
    high_bits: &'a [u64],
    low_bit_width: u32,
    count: usize,
    min_value: u64,
    high_len_bits: usize,
}

/// Partitioned Elias-Fano with uniform 128-element chunks.
///
/// Splits the sorted integer sequence into fixed-size chunks, each encoded
/// with its own local Elias-Fano. Per-chunk delta encoding adapts the bit
/// width to local density, improving cache locality for `next_geq`.
///
/// **Cache locality advantage:** Each chunk's data (low_bits + high_bits)
/// fits in 1-3 cache lines. A `next_geq` query touches only the target
/// chunk's data, avoiding polluting L1 with distant parts of the sequence.
///
/// **When to use:** For posting lists > 256 elements where `next_geq`
/// performance matters. For small lists (< 256), plain `EliasFano` has
/// less overhead.
///
/// # Examples
///
/// ```rust
/// use zipora::succinct::elias_fano::PartitionedEliasFano;
///
/// let docs: Vec<u32> = (0..1000).map(|i| i * 10).collect();
/// let pef = PartitionedEliasFano::from_sorted(&docs);
///
/// assert_eq!(pef.len(), 1000);
/// assert_eq!(pef.get(0), Some(0));
/// assert_eq!(pef.get(999), Some(9990));
///
/// // next_geq: find first element >= target
/// assert_eq!(pef.next_geq(55), Some((6, 60)));
/// assert_eq!(pef.next_geq(9990), Some((999, 9990)));
/// assert_eq!(pef.next_geq(9991), None);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PartitionedEliasFano {
    /// All chunks' low bits concatenated into one contiguous array.
    all_low_bits: Vec<u64>,
    /// All chunks' high bits concatenated into one contiguous array.
    all_high_bits: Vec<u64>,
    /// Per-chunk metadata (offsets into the flat arrays).
    meta: Vec<PefChunkMeta>,
    /// Upper bound (last value) of each chunk, for binary search.
    chunk_upper_bounds: Vec<u64>,
    /// Total number of elements.
    len: usize,
    /// Universe upper bound (exclusive).
    universe: u64,
}

impl PartitionedEliasFano {
    /// Build from a sorted slice of u32 values.
    pub fn from_sorted(values: &[u32]) -> Self {
        if values.is_empty() {
            return Self {
                all_low_bits: Vec::new(), all_high_bits: Vec::new(),
                meta: Vec::new(), chunk_upper_bounds: Vec::new(), len: 0, universe: 0,
            };
        }
        let universe = values[values.len() - 1] as u64 + 1;
        Self::from_sorted_impl(values.len(), universe, |i| values[i] as u64)
    }

    /// Build from sorted u64 values.
    pub fn from_sorted_u64(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self {
                all_low_bits: Vec::new(), all_high_bits: Vec::new(),
                meta: Vec::new(), chunk_upper_bounds: Vec::new(), len: 0, universe: 0,
            };
        }
        let universe = values[values.len() - 1] + 1;
        Self::from_sorted_impl(values.len(), universe, |i| values[i])
    }

    fn from_sorted_impl(n: usize, universe: u64, get_val: impl Fn(usize) -> u64) -> Self {
        let num_chunks = (n + PEF_CHUNK_SIZE - 1) / PEF_CHUNK_SIZE;
        let mut all_low_bits = Vec::new();
        let mut all_high_bits = Vec::new();
        let mut meta = Vec::with_capacity(num_chunks);
        let mut chunk_upper_bounds = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * PEF_CHUNK_SIZE;
            let end = (start + PEF_CHUNK_SIZE).min(n);
            let count = end - start;

            let min_val = get_val(start);
            let max_val = get_val(end - 1);
            let local_universe = max_val - min_val + 1;

            // Compute local low_bit_width
            let low_bit_width = if count as u64 >= local_universe {
                0
            } else {
                (64 - (local_universe / count as u64).leading_zeros()).saturating_sub(1)
            };

            let low_mask = if low_bit_width == 0 { 0u64 } else { (1u64 << low_bit_width) - 1 };

            // Pack low bits directly into flat array
            let total_low_bits = count as u64 * low_bit_width as u64;
            let low_words = ((total_low_bits + 63) / 64) as usize;
            let low_offset = all_low_bits.len();
            all_low_bits.resize(low_offset + low_words, 0);

            for i in 0..count {
                if low_bit_width > 0 {
                    let delta = get_val(start + i) - min_val;
                    let low_val = delta & low_mask;
                    let bit_pos = i as u64 * low_bit_width as u64;
                    let word_idx = (bit_pos / 64) as usize;
                    let bit_idx = (bit_pos % 64) as u32;
                    all_low_bits[low_offset + word_idx] |= low_val << bit_idx;
                    if bit_idx + low_bit_width > 64 && word_idx + 1 < low_words {
                        all_low_bits[low_offset + word_idx + 1] |= low_val >> (64 - bit_idx);
                    }
                }
            }

            // Build high bits directly into flat array
            let last_delta = max_val - min_val;
            let max_high = last_delta >> low_bit_width;
            let high_len_bits = count + max_high as usize + 1;
            let high_words = (high_len_bits + 63) / 64;
            let high_offset = all_high_bits.len();
            all_high_bits.resize(high_offset + high_words, 0);

            let mut pos = 0usize;
            let mut prev_high = 0u64;

            for i in 0..count {
                let delta = get_val(start + i) - min_val;
                let high = delta >> low_bit_width;
                pos += (high - prev_high) as usize;
                let word_idx = pos / 64;
                let bit_idx = pos % 64;
                if word_idx < high_words {
                    all_high_bits[high_offset + word_idx] |= 1u64 << bit_idx;
                }
                pos += 1;
                prev_high = high;
            }

            meta.push(PefChunkMeta {
                min_value: min_val,
                low_offset: low_offset as u32,
                high_offset: high_offset as u32,
                count: count as u16,
                high_len_bits: high_len_bits as u16,
                high_words: high_words as u16,
                low_words: low_words as u16,
                low_bit_width: low_bit_width as u8,
            });
            chunk_upper_bounds.push(max_val);
        }

        // Padding word for branchless u128 extraction in chunk_get_low
        all_low_bits.push(0);

        Self { all_low_bits, all_high_bits, meta, chunk_upper_bounds, len: n, universe }
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize { self.len }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.all_low_bits.len() * 8
            + self.all_high_bits.len() * 8
            + self.meta.len() * std::mem::size_of::<PefChunkMeta>()
            + self.chunk_upper_bounds.len() * 8
            + 48 // struct fields
    }

    /// Bits per element.
    #[inline]
    pub fn bits_per_element(&self) -> f64 {
        if self.len == 0 { return 0.0; }
        (self.size_bytes() * 8) as f64 / self.len as f64
    }

    /// Get the i-th element. O(1) via chunk index arithmetic.
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len { return None; }
        let chunk_idx = index / PEF_CHUNK_SIZE;
        let local_idx = index % PEF_CHUNK_SIZE;
        let view = self.chunk_view(chunk_idx);
        let delta = chunk_get_delta(&view, local_idx);
        Some(view.min_value + delta)
    }

    /// Find the first element >= target. Returns (global_index, value).
    ///
    /// Binary search on chunk upper bounds, then select0-based skip within
    /// the chunk to jump directly to the target's high-value bucket.
    /// Only scans elements in the target bucket and beyond — typically
    /// 1-5 elements instead of the full 128-element chunk.
    #[inline]
    pub fn next_geq(&self, target: u64) -> Option<(usize, u64)> {
        if self.len == 0 || target >= self.universe { return None; }

        // Binary search: find first chunk whose upper_bound >= target
        let chunk_idx = match self.chunk_upper_bounds.binary_search(&target) {
            Ok(i) => i,      // exact match on upper bound
            Err(i) => {
                if i >= self.meta.len() { return None; }
                i
            }
        };

        let view = self.chunk_view(chunk_idx);
        let global_offset = chunk_idx * PEF_CHUNK_SIZE;

        // If target <= chunk.min_value, first element of chunk is the answer
        if target <= view.min_value {
            let delta = chunk_get_delta(&view, 0);
            return Some((global_offset, view.min_value + delta));
        }

        let target_delta = target - view.min_value;

        // Skip directly to the target_high bucket via select0 on high_bits.
        // This avoids scanning all preceding elements (the old 10-41× regression).
        let target_high = (target_delta >> view.low_bit_width) as usize;
        let (start_idx, start_pos) = chunk_skip_to_high(&view, target_high);

        // Scan only from the skip point — typically 1-5 elements
        if let Some((local_idx, delta, _)) = chunk_scan_geq(&view, target_delta, start_idx, start_pos) {
            return Some((global_offset + local_idx, view.min_value + delta));
        }

        // Not found in this chunk — first element of next chunk
        let next_chunk = chunk_idx + 1;
        if next_chunk < self.meta.len() {
            let nv = self.chunk_view(next_chunk);
            let delta = chunk_get_delta(&nv, 0);
            Some((next_chunk * PEF_CHUNK_SIZE, nv.min_value + delta))
        } else {
            None
        }
    }

    /// Iterator over all elements.
    pub fn iter(&self) -> PartitionedEliasFanoIter<'_> {
        if self.is_empty() {
            return PartitionedEliasFanoIter {
                pef: self,
                chunk_idx: 0,
                local_idx: 0,
                local_high_pos: 0,
                cached_high_bits: &[],
                cached_low_bits: &[],
                cached_low_bit_width: 0,
                cached_count: 0,
                cached_min_value: 0,
            };
        }
        let view = self.chunk_view(0);
        PartitionedEliasFanoIter {
            pef: self,
            chunk_idx: 0,
            local_idx: 0,
            local_high_pos: 0,
            cached_high_bits: view.high_bits,
            cached_low_bits: view.low_bits,
            cached_low_bit_width: view.low_bit_width,
            cached_count: view.count,
            cached_min_value: view.min_value,
        }
    }

    /// Create a stateful cursor for sequential access.
    #[inline]
    pub fn cursor(&self) -> PartitionedEliasFanoCursor<'_> {
        PartitionedEliasFanoCursor::new(self)
    }

    // --- Internal helpers ---

    /// Create a lightweight borrowing view into chunk `idx`.
    #[inline]
    fn chunk_view(&self, idx: usize) -> ChunkView<'_> {
        let m = &self.meta[idx];
        let low_start = m.low_offset as usize;
        // +1 for branchless u128 extraction (padding word at end ensures safety)
        let low_end = (low_start + m.low_words as usize + 1).min(self.all_low_bits.len());
        let high_start = m.high_offset as usize;
        ChunkView {
            low_bits: &self.all_low_bits[low_start..low_end],
            high_bits: &self.all_high_bits[high_start..high_start + m.high_words as usize],
            low_bit_width: m.low_bit_width as u32,
            count: m.count as usize,
            min_value: m.min_value,
            high_len_bits: m.high_len_bits as usize,
        }
    }
}

// ============================================================================
// Shared chunk helpers (used by both PEF and OPEF via ChunkView)
// ============================================================================

/// Get the delta value (value - min_value) of the local_idx-th element in a chunk.
#[inline]
fn chunk_get_delta(chunk: &ChunkView<'_>, local_idx: usize) -> u64 {
    let high_pos = chunk_select1(chunk, local_idx);
    let high_val = (high_pos - local_idx) as u64;
    let low = chunk_get_low(chunk, local_idx);
    (high_val << chunk.low_bit_width) | low
}

/// Select1 on a chunk's high_bits — find position of the k-th set bit.
/// No sampling needed (≤ 8 words for PEF, ≤ 16 for OPEF), just word-level scan.
#[inline]
fn chunk_select1(chunk: &ChunkView<'_>, rank: usize) -> usize {
    let mut remaining = rank;
    for (word_idx, &word) in chunk.high_bits.iter().enumerate() {
        let ones = word.count_ones() as usize;
        if remaining < ones {
            return word_idx * 64 + select_in_word(word, remaining);
        }
        remaining -= ones;
    }
    chunk.high_len_bits
}

/// Get low bits of the i-th element within a chunk.
/// Branchless: always loads 2 words via u128, ChunkView slice includes +1 padding.
#[inline]
fn chunk_get_low(chunk: &ChunkView<'_>, index: usize) -> u64 {
    if chunk.low_bit_width == 0 { return 0; }
    let bit_pos = index as u64 * chunk.low_bit_width as u64;
    let word_idx = (bit_pos / 64) as usize;
    let bit_idx = (bit_pos % 64) as u32;
    let w0 = chunk.low_bits[word_idx];
    let w1 = chunk.low_bits[word_idx + 1];
    let combined = w0 as u128 | ((w1 as u128) << 64);
    (combined >> bit_idx) as u64 & ((1u64 << chunk.low_bit_width) - 1)
}

/// Skip forward in a chunk's high_bits to where high_val >= target_high.
///
/// Returns `(element_index, bit_position)` — the scanning start point.
/// This is equivalent to `select0(target_high - 1) + 1` on the chunk's
/// high_bits, followed by `rank1` to get the element count before that point.
///
/// For chunks with ≤ 16 words of high_bits, this is a fast word-level scan
/// with no sampling overhead — analogous to plain EF's select0 jump but
/// without auxiliary structures.
#[inline]
fn chunk_skip_to_high(chunk: &ChunkView<'_>, target_high: usize) -> (usize, usize) {
    if target_high == 0 {
        return (0, 0);
    }

    let mut zeros_remaining = target_high;
    let mut ones_count = 0usize;
    let last_wi = chunk.high_bits.len().saturating_sub(1);

    for (wi, &w) in chunk.high_bits.iter().enumerate() {
        // Mask out padding bits in the last word
        let valid_bits = if wi == last_wi {
            let rem = chunk.high_len_bits % 64;
            if rem == 0 && chunk.high_len_bits > 0 { 64 } else { rem }
        } else {
            64
        };
        let valid_mask = if valid_bits >= 64 { u64::MAX } else { (1u64 << valid_bits) - 1 };
        let masked = w & valid_mask;

        let ones = masked.count_ones() as usize;
        let zeros = valid_bits - ones;

        if zeros_remaining <= zeros {
            // The target zero is in this word. Find position of the
            // zeros_remaining-th zero bit (1-indexed → select the
            // (zeros_remaining-1)-th set bit in the inverted word).
            let inverted = (!w) & valid_mask;
            let zero_pos_in_word = select_in_word(inverted, zeros_remaining - 1);
            let abs_pos = wi * 64 + zero_pos_in_word;

            // Count ones before abs_pos + 1 (rank1 up to and including abs_pos)
            let bits_up_to = zero_pos_in_word + 1;
            let partial_mask = if bits_up_to >= 64 { u64::MAX } else { (1u64 << bits_up_to) - 1 };
            ones_count += (w & partial_mask).count_ones() as usize;

            // Start scanning from the bit after the target zero
            return (ones_count, abs_pos + 1);
        }

        zeros_remaining -= zeros;
        ones_count += ones;
    }

    // All zeros consumed — past the end of the chunk
    (chunk.count, chunk.high_len_bits)
}

/// Find position of the first set bit in a chunk's high_bits.
#[inline]
fn chunk_first_one(chunk: &ChunkView<'_>) -> usize {
    chunk_first_one_cached(chunk.high_bits)
}

/// Find position of the first set bit in a high_bits slice (no ChunkView needed).
#[inline]
fn chunk_first_one_cached(high_bits: &[u64]) -> usize {
    for (wi, &w) in high_bits.iter().enumerate() {
        if w != 0 {
            return wi * 64 + w.trailing_zeros() as usize;
        }
    }
    0
}

/// Linear scan within a chunk's high_bits starting from a given position.
/// Returns `Some((local_index, delta, bit_pos))` for the first element with
/// `delta >= target_delta`. The returned `bit_pos` is the position of that
/// element's set bit in high_bits, eliminating the need for a subsequent
/// `chunk_select1` call.
#[inline]
fn chunk_scan_geq(
    chunk: &ChunkView<'_>,
    target_delta: u64,
    start_idx: usize,
    start_pos: usize,
) -> Option<(usize, u64, usize)> {
    let mut idx = start_idx;
    let mut bit_pos = start_pos;
    let mut word_idx = bit_pos / 64;
    if word_idx >= chunk.high_bits.len() {
        return None;
    }

    let start_bit = bit_pos % 64;
    let mut word = chunk.high_bits[word_idx] >> start_bit;
    bit_pos = word_idx * 64 + start_bit;

    while idx < chunk.count {
        if word == 0 {
            word_idx += 1;
            if word_idx >= chunk.high_bits.len() { break; }
            word = chunk.high_bits[word_idx];
            bit_pos = word_idx * 64;
            continue;
        }

        let tz = word.trailing_zeros() as usize;
        bit_pos += tz;
        word >>= tz;

        let high_val = (bit_pos - idx) as u64;
        let low = chunk_get_low(chunk, idx);
        let delta = (high_val << chunk.low_bit_width) | low;

        if delta >= target_delta {
            return Some((idx, delta, bit_pos));
        }

        idx += 1;
        bit_pos += 1;
        word >>= 1;
    }

    None
}

/// Iterator over PartitionedEliasFano elements.
///
/// Caches the current chunk's slice pointers to avoid per-element `chunk_view()` overhead.
/// Slices are refreshed only on chunk boundary transitions.
pub struct PartitionedEliasFanoIter<'a> {
    pef: &'a PartitionedEliasFano,
    chunk_idx: usize,
    local_idx: usize,
    local_high_pos: usize,
    // Cached from current chunk — refreshed on chunk transitions only
    cached_high_bits: &'a [u64],
    cached_low_bits: &'a [u64],
    cached_low_bit_width: u32,
    cached_count: usize,
    cached_min_value: u64,
}

impl<'a> PartitionedEliasFanoIter<'a> {
    /// Refresh cached chunk data for the current chunk_idx.
    #[inline]
    fn refresh_chunk_cache(&mut self) {
        let view = self.pef.chunk_view(self.chunk_idx);
        self.cached_high_bits = view.high_bits;
        self.cached_low_bits = view.low_bits;
        self.cached_low_bit_width = view.low_bit_width;
        self.cached_count = view.count;
        self.cached_min_value = view.min_value;
    }
}

impl<'a> Iterator for PartitionedEliasFanoIter<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk_idx >= self.pef.meta.len() { return None; }
        if self.local_idx >= self.cached_count {
            // Move to next chunk — refresh cache
            self.chunk_idx += 1;
            self.local_idx = 0;
            self.local_high_pos = 0;
            if self.chunk_idx >= self.pef.meta.len() { return None; }
            self.refresh_chunk_cache();
            return self.next(); // recurse once to process the new chunk
        }

        // Find next 1-bit in cached high_bits
        let next_pos = self.local_high_pos;
        let mut word_idx = next_pos / 64;
        if word_idx >= self.cached_high_bits.len() { return None; }

        let bit_offset = next_pos % 64;
        let mut word = self.cached_high_bits[word_idx] >> bit_offset;

        if word != 0 {
            let tz = word.trailing_zeros() as usize;
            self.local_high_pos = word_idx * 64 + bit_offset + tz;
        } else {
            loop {
                word_idx += 1;
                if word_idx >= self.cached_high_bits.len() { return None; }
                word = self.cached_high_bits[word_idx];
                if word != 0 {
                    self.local_high_pos = word_idx * 64 + word.trailing_zeros() as usize;
                    break;
                }
            }
        }

        let high_val = (self.local_high_pos - self.local_idx) as u64;
        // Inline low-bit extraction using cached slice
        let low = if self.cached_low_bit_width == 0 {
            0
        } else {
            let lbw = self.cached_low_bit_width as u64;
            let bit_pos = self.local_idx as u64 * lbw;
            let w_idx = (bit_pos / 64) as usize;
            let b_idx = (bit_pos % 64) as u32;
            let w0 = self.cached_low_bits[w_idx];
            let w1 = self.cached_low_bits[w_idx + 1]; // safe: padding word
            let combined = w0 as u128 | ((w1 as u128) << 64);
            (combined >> b_idx) as u64 & ((1u64 << lbw) - 1)
        };
        let delta = (high_val << self.cached_low_bit_width) | low;
        let val = self.cached_min_value + delta;

        self.local_idx += 1;
        self.local_high_pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let consumed = self.chunk_idx * PEF_CHUNK_SIZE + self.local_idx;
        let remaining = self.pef.len.saturating_sub(consumed);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for PartitionedEliasFanoIter<'a> {}

/// Stateful cursor for O(1) amortized sequential access over PartitionedEliasFano.
///
/// Caches the current chunk's slice pointers to avoid per-element `chunk_view()` overhead.
pub struct PartitionedEliasFanoCursor<'a> {
    pef: &'a PartitionedEliasFano,
    chunk_idx: usize,
    local_idx: usize,
    local_high_pos: usize,
    global_idx: usize,
    /// Cached current element value — avoids recomputing in advance_to_geq.
    cached_value: u64,
    // Cached from current chunk — refreshed on chunk transitions only
    cached_high_bits: &'a [u64],
    cached_low_bits: &'a [u64],
    cached_low_bit_width: u32,
    cached_count: usize,
    cached_min_value: u64,
}

impl<'a> PartitionedEliasFanoCursor<'a> {
    fn new(pef: &'a PartitionedEliasFano) -> Self {
        if pef.is_empty() {
            return Self {
                pef, chunk_idx: 0, local_idx: 0, local_high_pos: 0, global_idx: 0,
                cached_value: 0,
                cached_high_bits: &[], cached_low_bits: &[],
                cached_low_bit_width: 0, cached_count: 0, cached_min_value: 0,
            };
        }
        let view = pef.chunk_view(0);
        let mut high_pos = 0;
        for (word_idx, &word) in view.high_bits.iter().enumerate() {
            if word != 0 {
                high_pos = word_idx * 64 + word.trailing_zeros() as usize;
                break;
            }
        }
        // Compute initial element value
        let high_val = high_pos as u64; // local_idx is 0, so high_val = high_pos - 0
        let low = chunk_get_low(&view, 0);
        let initial_val = view.min_value + (high_val << view.low_bit_width) + low;
        Self {
            pef, chunk_idx: 0, local_idx: 0, local_high_pos: high_pos, global_idx: 0,
            cached_value: initial_val,
            cached_high_bits: view.high_bits, cached_low_bits: view.low_bits,
            cached_low_bit_width: view.low_bit_width, cached_count: view.count,
            cached_min_value: view.min_value,
        }
    }

    /// Refresh cached chunk data for the current chunk_idx.
    #[inline]
    fn refresh_chunk_cache(&mut self) {
        let view = self.pef.chunk_view(self.chunk_idx);
        self.cached_high_bits = view.high_bits;
        self.cached_low_bits = view.low_bits;
        self.cached_low_bit_width = view.low_bit_width;
        self.cached_count = view.count;
        self.cached_min_value = view.min_value;
    }

    /// Inline low-bit extraction from cached slices.
    #[inline]
    fn get_low_cached(&self, local_idx: usize) -> u64 {
        if self.cached_low_bit_width == 0 { return 0; }
        let lbw = self.cached_low_bit_width as u64;
        let bit_pos = local_idx as u64 * lbw;
        let w_idx = (bit_pos / 64) as usize;
        let b_idx = (bit_pos % 64) as u32;
        let w0 = self.cached_low_bits[w_idx];
        let w1 = self.cached_low_bits[w_idx + 1]; // safe: padding word
        let combined = w0 as u128 | ((w1 as u128) << 64);
        (combined >> b_idx) as u64 & ((1u64 << lbw) - 1)
    }

    /// Compute and cache the current element value from cursor state.
    #[inline]
    fn recompute_value(&mut self) {
        let high_val = (self.local_high_pos - self.local_idx) as u64;
        let low = self.get_low_cached(self.local_idx);
        self.cached_value = self.cached_min_value + (high_val << self.cached_low_bit_width) + low;
    }

    /// Current element value — O(1) from cached value.
    #[inline]
    pub fn current(&self) -> Option<u64> {
        if self.global_idx >= self.pef.len { None } else { Some(self.cached_value) }
    }

    /// Current global element index.
    #[inline]
    pub fn index(&self) -> usize { self.global_idx }

    /// Whether the cursor is past the last element.
    #[inline]
    pub fn is_exhausted(&self) -> bool { self.global_idx >= self.pef.len }

    /// Advance to the next element.
    #[inline]
    pub fn advance(&mut self) -> bool {
        self.global_idx += 1;
        if self.global_idx >= self.pef.len { return false; }

        self.local_idx += 1;

        // Check if we need to cross a chunk boundary
        if self.local_idx >= self.cached_count {
            self.chunk_idx += 1;
            self.local_idx = 0;
            if self.chunk_idx >= self.pef.meta.len() { return false; }
            self.refresh_chunk_cache();
            self.local_high_pos = chunk_first_one_cached(self.cached_high_bits);
            self.recompute_value();
            return true;
        }

        // Find next 1-bit within current chunk — use cached high_bits
        let next_pos = self.local_high_pos + 1;
        let mut word_idx = next_pos / 64;
        if word_idx >= self.cached_high_bits.len() { return false; }

        let bit_in_word = next_pos % 64;
        let mut word = self.cached_high_bits[word_idx] >> bit_in_word;

        if word != 0 {
            self.local_high_pos = word_idx * 64 + bit_in_word + word.trailing_zeros() as usize;
        } else {
            loop {
                word_idx += 1;
                if word_idx >= self.cached_high_bits.len() { return false; }
                word = self.cached_high_bits[word_idx];
                if word != 0 {
                    self.local_high_pos = word_idx * 64 + word.trailing_zeros() as usize;
                    break;
                }
            }
        }
        self.recompute_value();
        true
    }

    /// Advance to the first element >= target.
    ///
    /// Optimized with three fast paths:
    /// 1. **Current >= target**: O(1) — single comparison against cached value.
    /// 2. **Same-chunk**: Scan forward from cursor position directly (no
    ///    `chunk_skip_to_high`). O(d) where d = distance to target in elements.
    /// 3. **Galloping search**: O(log d) where d = distance in chunks to target,
    ///    instead of O(log C) binary search over all C chunks.
    #[inline]
    pub fn advance_to_geq(&mut self, target: u64) -> bool {
        // Fast path 1: cached_value >= target — O(1), no computation
        if self.global_idx >= self.pef.len { return false; }
        if self.cached_value >= target { return true; }

        let num_chunks = self.pef.meta.len();

        // Fast path 2: target is within current chunk's range
        // Scan directly from cursor position — no chunk_skip_to_high overhead.
        // For sorted posting list queries, targets are close together, so this
        // typically scans only a few elements.
        if target <= self.pef.chunk_upper_bounds[self.chunk_idx] {
            let target_delta = target - self.cached_min_value;
            let view = ChunkView {
                low_bits: self.cached_low_bits,
                high_bits: self.cached_high_bits,
                low_bit_width: self.cached_low_bit_width,
                count: self.cached_count,
                min_value: self.cached_min_value,
                high_len_bits: self.pef.meta[self.chunk_idx].high_len_bits as usize,
            };
            if let Some((local_idx, delta, hp)) = chunk_scan_geq(&view, target_delta, self.local_idx, self.local_high_pos) {
                self.local_idx = local_idx;
                self.global_idx = self.chunk_idx * PEF_CHUNK_SIZE + local_idx;
                self.local_high_pos = hp;
                self.cached_value = self.cached_min_value + delta;
                return true;
            }
        }

        // Galloping (exponential) search forward from current chunk
        let mut lo = self.chunk_idx;
        let mut step = 1usize;
        let target_chunk = loop {
            let hi = (lo + step).min(num_chunks - 1);
            if self.pef.chunk_upper_bounds[hi] >= target {
                let s = lo + 1;
                if s > hi {
                    break hi;
                }
                break s + self.pef.chunk_upper_bounds[s..=hi]
                    .partition_point(|&x| x < target);
            }
            if hi == num_chunks - 1 {
                self.global_idx = self.pef.len;
                return false;
            }
            lo = hi;
            step *= 2;
        };

        // Reposition cursor to target chunk and refresh cache
        self.chunk_idx = target_chunk;
        self.refresh_chunk_cache();
        let global_offset = target_chunk * PEF_CHUNK_SIZE;

        if target <= self.cached_min_value {
            self.local_idx = 0;
            self.global_idx = global_offset;
            self.local_high_pos = chunk_first_one_cached(self.cached_high_bits);
            self.recompute_value();
            return true;
        }

        let view = ChunkView {
            low_bits: self.cached_low_bits, high_bits: self.cached_high_bits,
            low_bit_width: self.cached_low_bit_width, count: self.cached_count,
            min_value: self.cached_min_value,
            high_len_bits: self.pef.meta[self.chunk_idx].high_len_bits as usize,
        };
        let target_delta = target - self.cached_min_value;
        let target_high = (target_delta >> view.low_bit_width) as usize;
        let (si, sp) = chunk_skip_to_high(&view, target_high);

        if let Some((local_idx, delta, hp)) = chunk_scan_geq(&view, target_delta, si, sp) {
            self.local_idx = local_idx;
            self.global_idx = global_offset + local_idx;
            self.local_high_pos = hp;
            self.cached_value = self.cached_min_value + delta;
            return true;
        }

        // First element of next chunk
        let next = target_chunk + 1;
        if next >= num_chunks {
            self.global_idx = self.pef.len;
            return false;
        }
        self.chunk_idx = next;
        self.refresh_chunk_cache();
        self.local_idx = 0;
        self.global_idx = next * PEF_CHUNK_SIZE;
        self.local_high_pos = chunk_first_one_cached(self.cached_high_bits);
        self.recompute_value();
        true
    }

    /// Reposition the cursor directly to element at global index `idx`.
    /// Returns false (cursor unchanged) if `idx >= self.pef.len`.
    #[inline]
    pub fn advance_to_index(&mut self, idx: usize) -> bool {
        if idx >= self.pef.len { return false; }
        if idx == self.global_idx { return true; }

        let chunk_idx = idx / PEF_CHUNK_SIZE;
        let local_idx = idx % PEF_CHUNK_SIZE;

        // Switch chunk if needed
        if chunk_idx != self.chunk_idx {
            self.chunk_idx = chunk_idx;
            self.refresh_chunk_cache();
        }

        self.local_idx = local_idx;
        self.global_idx = idx;

        // Find local_high_pos by scanning for the local_idx-th 1-bit in cached_high_bits
        self.local_high_pos = Self::find_nth_one(self.cached_high_bits, local_idx);
        self.recompute_value();
        true
    }

    /// Find the position of the n-th set bit (0-indexed) in a high_bits slice.
    #[inline]
    fn find_nth_one(high_bits: &[u64], n: usize) -> usize {
        let mut remaining = n;
        for (word_idx, &word) in high_bits.iter().enumerate() {
            let ones = word.count_ones() as usize;
            if remaining < ones {
                return word_idx * 64 + select_in_word(word, remaining);
            }
            remaining -= ones;
        }
        0 // Should not happen for valid index
    }

    /// Reset cursor to the first element.
    pub fn reset(&mut self) {
        *self = Self::new(self.pef);
    }
}

// ============================================================================
// Batch Sequential Decoder (Phase 2)
// ============================================================================

/// Batch size for buffered decoding. 8 elements fits in registers and
/// amortizes refill overhead while keeping the buffer small enough for L1.
const BATCH_SIZE: usize = 8;

/// Batch cursor for EliasFano with 8-element decode buffer.
///
/// Decodes elements in batches of 8 to amortize the cost of packed low-bits
/// extraction across word boundaries. The high-bits scanner maintains state
/// between batches, so each refill is O(BATCH_SIZE) amortized.
///
/// **When to use:** Sequential iteration over large posting lists where
/// throughput matters more than latency. For random access or `next_geq`,
/// use `EliasFanoCursor` or `EliasFano::next_geq` directly.
pub struct EliasFanoBatchCursor<'a> {
    ef: &'a EliasFano,
    /// Decoded values buffer.
    buffer: [u64; BATCH_SIZE],
    /// Current position within buffer.
    buf_pos: usize,
    /// Number of valid entries in buffer.
    buf_len: usize,
    /// Next element index to decode into the buffer.
    next_elem: usize,
    /// Bit position in high_bits for next_elem.
    high_pos: usize,
}

impl<'a> EliasFanoBatchCursor<'a> {
    fn new(ef: &'a EliasFano) -> Self {
        let mut cursor = Self {
            ef,
            buffer: [0; BATCH_SIZE],
            buf_pos: 0,
            buf_len: 0,
            next_elem: 0,
            high_pos: 0,
        };
        if !ef.is_empty() {
            cursor.refill();
        }
        cursor
    }

    /// Current element value.
    #[inline]
    pub fn current(&self) -> Option<u64> {
        if self.buf_pos < self.buf_len {
            Some(self.buffer[self.buf_pos])
        } else {
            None
        }
    }

    /// Current global element index.
    #[inline]
    pub fn index(&self) -> usize {
        self.next_elem - self.buf_len + self.buf_pos
    }

    /// Whether the cursor is past the last element.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.buf_pos >= self.buf_len && self.next_elem >= self.ef.len
    }

    /// Advance to the next element. O(1) amortized — refill every 8 advances.
    #[inline]
    pub fn advance(&mut self) -> bool {
        self.buf_pos += 1;
        if self.buf_pos < self.buf_len {
            return true;
        }
        if self.next_elem >= self.ef.len {
            return false;
        }
        self.refill();
        self.buf_len > 0
    }

    /// Refill the buffer with up to BATCH_SIZE decoded elements.
    ///
    /// Optimized with incremental low-bit tracking: instead of recomputing
    /// word_idx/bit_idx from scratch for each element's get_low(), we track
    /// the low-bit position and pre-loaded words across the batch, advancing
    /// by low_bit_width per element.
    fn refill(&mut self) {
        let count = (self.ef.len - self.next_elem).min(BATCH_SIZE);
        if count == 0 {
            self.buf_len = 0;
            return;
        }

        let start_elem = self.next_elem;
        let lbw = self.ef.low_bit_width;

        // --- High-bit scanning state ---
        let mut bit_pos = self.high_pos;
        let mut word_idx = bit_pos / 64;
        let start_bit = bit_pos % 64;
        let mut word = if word_idx < self.ef.high_bits.len() {
            self.ef.high_bits[word_idx] >> start_bit
        } else {
            0
        };
        bit_pos = word_idx * 64 + start_bit;

        // --- Low-bit incremental state ---
        let low_mask = if lbw == 0 { 0u64 } else { (1u64 << lbw) - 1 };
        let mut low_bit_pos = start_elem as u64 * lbw as u64;
        let mut low_word_idx = (low_bit_pos / 64) as usize;
        let mut low_bit_idx = (low_bit_pos % 64) as u32;
        let mut low_word = if lbw > 0 && low_word_idx < self.ef.low_bits.len() {
            self.ef.low_bits[low_word_idx]
        } else { 0 };
        let mut low_word_next = if lbw > 0 && low_word_idx + 1 < self.ef.low_bits.len() {
            self.ef.low_bits[low_word_idx + 1]
        } else { 0 };

        for k in 0..count {
            let idx = start_elem + k;

            // Find next 1-bit in high_bits
            while word == 0 {
                word_idx += 1;
                word = self.ef.high_bits[word_idx];
                bit_pos = word_idx * 64;
            }
            let tz = word.trailing_zeros() as usize;
            bit_pos += tz;
            word >>= tz;

            let high_val = (bit_pos - idx) as u64;

            // Inline low-bit extraction with incremental tracking
            let low = if lbw == 0 {
                0u64
            } else {
                let mut val = low_word >> low_bit_idx;
                if low_bit_idx + lbw > 64 {
                    val |= low_word_next << (64 - low_bit_idx);
                }
                val & low_mask
            };

            self.buffer[k] = (high_val << lbw) | low;

            bit_pos += 1;
            word >>= 1;

            // Advance low-bit position incrementally
            if lbw > 0 {
                low_bit_idx += lbw;
                if low_bit_idx >= 64 {
                    low_bit_idx -= 64;
                    low_word_idx += 1;
                    low_word = low_word_next;
                    low_word_next = if low_word_idx + 1 < self.ef.low_bits.len() {
                        self.ef.low_bits[low_word_idx + 1]
                    } else { 0 };
                }
            }
        }

        self.buf_pos = 0;
        self.buf_len = count;
        self.next_elem = start_elem + count;
        self.high_pos = bit_pos;
    }

    /// Reset cursor to the first element.
    pub fn reset(&mut self) {
        *self = Self::new(self.ef);
    }
}

impl EliasFano {
    /// Create a batch cursor for high-throughput sequential iteration.
    #[inline]
    pub fn batch_cursor(&self) -> EliasFanoBatchCursor<'_> {
        EliasFanoBatchCursor::new(self)
    }
}

/// Batch cursor for PartitionedEliasFano with 8-element decode buffer.
pub struct PartitionedEliasFanoBatchCursor<'a> {
    pef: &'a PartitionedEliasFano,
    buffer: [u64; BATCH_SIZE],
    buf_pos: usize,
    buf_len: usize,
    /// Next global element index to decode.
    next_elem: usize,
    /// Current chunk index.
    chunk_idx: usize,
    /// Next local index within current chunk.
    local_idx: usize,
    /// Bit position in current chunk's high_bits.
    local_high_pos: usize,
}

impl<'a> PartitionedEliasFanoBatchCursor<'a> {
    fn new(pef: &'a PartitionedEliasFano) -> Self {
        let mut cursor = Self {
            pef,
            buffer: [0; BATCH_SIZE],
            buf_pos: 0,
            buf_len: 0,
            next_elem: 0,
            chunk_idx: 0,
            local_idx: 0,
            local_high_pos: 0,
        };
        if !pef.is_empty() {
            cursor.refill();
        }
        cursor
    }

    #[inline]
    pub fn current(&self) -> Option<u64> {
        if self.buf_pos < self.buf_len {
            Some(self.buffer[self.buf_pos])
        } else {
            None
        }
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.next_elem - self.buf_len + self.buf_pos
    }

    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.buf_pos >= self.buf_len && self.next_elem >= self.pef.len
    }

    #[inline]
    pub fn advance(&mut self) -> bool {
        self.buf_pos += 1;
        if self.buf_pos < self.buf_len {
            return true;
        }
        if self.next_elem >= self.pef.len {
            return false;
        }
        self.refill();
        self.buf_len > 0
    }

    fn refill(&mut self) {
        let remaining = self.pef.len - self.next_elem;
        if remaining == 0 {
            self.buf_len = 0;
            return;
        }
        let count = remaining.min(BATCH_SIZE);
        let mut filled = 0;

        while filled < count {
            if self.chunk_idx >= self.pef.meta.len() {
                break;
            }
            let view = self.pef.chunk_view(self.chunk_idx);

            // If starting a new chunk, find the first 1-bit
            if self.local_idx == 0 {
                self.local_high_pos = 0;
                for (wi, &w) in view.high_bits.iter().enumerate() {
                    if w != 0 {
                        self.local_high_pos = wi * 64 + w.trailing_zeros() as usize;
                        break;
                    }
                }
            }

            let chunk_remaining = view.count - self.local_idx;
            let to_decode = (count - filled).min(chunk_remaining);

            let lbw = view.low_bit_width;
            let low_mask = if lbw == 0 { 0u64 } else { (1u64 << lbw) - 1 };

            // --- High-bit scanning state ---
            let mut bit_pos = self.local_high_pos;
            let mut word_idx = bit_pos / 64;
            let start_bit = bit_pos % 64;
            let mut word = if word_idx < view.high_bits.len() {
                view.high_bits[word_idx] >> start_bit
            } else {
                0
            };
            bit_pos = word_idx * 64 + start_bit;

            // --- Low-bit incremental state ---
            let mut low_bit_pos = self.local_idx as u64 * lbw as u64;
            let mut low_word_idx = (low_bit_pos / 64) as usize;
            let mut low_bit_idx = (low_bit_pos % 64) as u32;
            let mut low_word = if lbw > 0 && low_word_idx < view.low_bits.len() {
                view.low_bits[low_word_idx]
            } else { 0 };
            let mut low_word_next = if lbw > 0 && low_word_idx + 1 < view.low_bits.len() {
                view.low_bits[low_word_idx + 1]
            } else { 0 };

            for k in 0..to_decode {
                let li = self.local_idx + k;

                while word == 0 {
                    word_idx += 1;
                    if word_idx >= view.high_bits.len() { break; }
                    word = view.high_bits[word_idx];
                    bit_pos = word_idx * 64;
                }
                let tz = word.trailing_zeros() as usize;
                bit_pos += tz;
                word >>= tz;

                let high_val = (bit_pos - li) as u64;

                // Inline low-bit extraction with incremental tracking
                let low = if lbw == 0 {
                    0u64
                } else {
                    let mut val = low_word >> low_bit_idx;
                    if low_bit_idx + lbw > 64 {
                        val |= low_word_next << (64 - low_bit_idx);
                    }
                    val & low_mask
                };

                let delta = (high_val << lbw) | low;
                self.buffer[filled + k] = view.min_value + delta;

                bit_pos += 1;
                word >>= 1;

                // Advance low-bit position
                if lbw > 0 {
                    low_bit_idx += lbw;
                    if low_bit_idx >= 64 {
                        low_bit_idx -= 64;
                        low_word_idx += 1;
                        low_word = low_word_next;
                        low_word_next = if low_word_idx + 1 < view.low_bits.len() {
                            view.low_bits[low_word_idx + 1]
                        } else { 0 };
                    }
                }
            }

            self.local_idx += to_decode;
            self.local_high_pos = bit_pos;
            filled += to_decode;

            // Move to next chunk if current is exhausted
            if self.local_idx >= view.count {
                self.chunk_idx += 1;
                self.local_idx = 0;
                self.local_high_pos = 0;
            }
        }

        self.buf_pos = 0;
        self.buf_len = filled;
        self.next_elem += filled;
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.pef);
    }
}

impl PartitionedEliasFano {
    /// Create a batch cursor for high-throughput sequential iteration.
    #[inline]
    pub fn batch_cursor(&self) -> PartitionedEliasFanoBatchCursor<'_> {
        PartitionedEliasFanoBatchCursor::new(self)
    }
}

// ============================================================================
// DP-Optimal Partitioned Elias-Fano (Phase 2)
// ============================================================================

/// Minimum chunk size for DP-optimal partitioning.
/// Below this, per-chunk overhead dominates.
const MIN_CHUNK_SIZE: usize = 32;

/// Maximum chunk size for DP-optimal partitioning.
/// Above this, we lose cache locality benefits.
const MAX_CHUNK_SIZE: usize = 512;

/// Per-chunk overhead in bits (min_value, count, low_bit_width, Vec headers).
const CHUNK_OVERHEAD_BITS: usize = 256;

/// DP-Optimal Partitioned Elias-Fano with variable-length chunks.
///
/// Uses linear-time dynamic programming to find the partition that minimizes
/// total encoding size. Each chunk adapts its size to local data density:
/// dense regions get smaller chunks (better compression), sparse regions
/// get larger chunks (less overhead).
///
/// **Space savings:** 5-15% smaller than uniform 128-element PEF.
///
/// **When to use:** When compression ratio matters and you can afford
/// slightly more build time. For real-time construction, use
/// `PartitionedEliasFano` (uniform chunks) instead.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimalPartitionedEliasFano {
    /// All chunks' low bits concatenated into one contiguous array.
    all_low_bits: Vec<u64>,
    /// All chunks' high bits concatenated into one contiguous array.
    all_high_bits: Vec<u64>,
    /// Per-chunk metadata (offsets into the flat arrays).
    meta: Vec<PefChunkMeta>,
    /// Cumulative element count: chunk_starts[i] = start index of chunk i.
    chunk_starts: Vec<usize>,
    /// Upper bound (last value) of each chunk.
    chunk_upper_bounds: Vec<u64>,
    len: usize,
    universe: u64,
}

impl OptimalPartitionedEliasFano {
    /// Build from a sorted slice of u32 values.
    pub fn from_sorted(values: &[u32]) -> Self {
        if values.is_empty() {
            return Self {
                all_low_bits: Vec::new(), all_high_bits: Vec::new(),
                meta: Vec::new(), chunk_starts: Vec::new(),
                chunk_upper_bounds: Vec::new(), len: 0, universe: 0,
            };
        }
        let universe = values[values.len() - 1] as u64 + 1;
        Self::from_sorted_impl(values.len(), universe, |i| values[i] as u64)
    }

    /// Build from sorted u64 values.
    pub fn from_sorted_u64(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self {
                all_low_bits: Vec::new(), all_high_bits: Vec::new(),
                meta: Vec::new(), chunk_starts: Vec::new(),
                chunk_upper_bounds: Vec::new(), len: 0, universe: 0,
            };
        }
        let universe = values[values.len() - 1] + 1;
        Self::from_sorted_impl(values.len(), universe, |i| values[i])
    }

    /// Compute the EF encoding cost in bits for values[start..end].
    #[inline]
    fn chunk_cost(n: usize, local_universe: u64) -> usize {
        if n == 0 { return 0; }
        let low_bit_width = if n as u64 >= local_universe {
            0u32
        } else {
            (64 - (local_universe / n as u64).leading_zeros()).saturating_sub(1)
        };
        let low_bits = n * low_bit_width as usize;
        let max_high = if low_bit_width == 0 {
            local_universe.saturating_sub(1) as usize
        } else {
            (local_universe.saturating_sub(1) >> low_bit_width) as usize
        };
        let high_bits = n + max_high + 1;
        low_bits + high_bits + CHUNK_OVERHEAD_BITS
    }

    fn from_sorted_impl(n: usize, universe: u64, get_val: impl Fn(usize) -> u64) -> Self {
        // DP: dp[i] = minimum cost to encode values[0..i]
        // back[i] = optimal chunk start for the chunk ending at i
        let mut dp = vec![usize::MAX; n + 1];
        let mut back = vec![0usize; n + 1];
        dp[0] = 0;

        for i in 1..=n {
            let max_j_start = if i > MAX_CHUNK_SIZE { i - MAX_CHUNK_SIZE } else { 0 };
            let min_j_end = if i >= MIN_CHUNK_SIZE { i - MIN_CHUNK_SIZE + 1 } else { 0 };

            // Only consider valid chunk sizes [MIN_CHUNK_SIZE, MAX_CHUNK_SIZE]
            // except the last chunk which can be smaller
            let j_start = max_j_start;
            let j_end = if i == n {
                // Last chunk: allow any remaining size >= 1
                i
            } else {
                min_j_end
            };

            for j in j_start..j_end {
                let chunk_n = i - j;
                if chunk_n < 1 { continue; }
                // For non-last chunks, enforce minimum size
                if i < n && chunk_n < MIN_CHUNK_SIZE { continue; }
                if chunk_n > MAX_CHUNK_SIZE { continue; }

                if dp[j] == usize::MAX { continue; }

                let min_val = get_val(j);
                let max_val = get_val(i - 1);
                let local_u = max_val - min_val + 1;
                let cost = dp[j] + Self::chunk_cost(chunk_n, local_u);

                if cost < dp[i] {
                    dp[i] = cost;
                    back[i] = j;
                }
            }
        }

        // Trace back the optimal partition
        let mut partition_points = Vec::new();
        let mut pos = n;
        while pos > 0 {
            partition_points.push(pos);
            pos = back[pos];
        }
        partition_points.reverse();

        // Build chunks into flat arrays
        let num_chunks = partition_points.len();
        let mut all_low_bits = Vec::new();
        let mut all_high_bits = Vec::new();
        let mut meta = Vec::with_capacity(num_chunks);
        let mut chunk_starts = Vec::with_capacity(num_chunks);
        let mut chunk_upper_bounds = Vec::with_capacity(num_chunks);

        let mut start = 0;
        for &end in &partition_points {
            let count = end - start;
            let min_val = get_val(start);
            let max_val = get_val(end - 1);
            let local_universe = max_val - min_val + 1;

            let low_bit_width = if count as u64 >= local_universe {
                0
            } else {
                (64 - (local_universe / count as u64).leading_zeros()).saturating_sub(1)
            };
            let low_mask = if low_bit_width == 0 { 0u64 } else { (1u64 << low_bit_width) - 1 };

            // Pack low bits directly into flat array
            let total_low_bits = count as u64 * low_bit_width as u64;
            let low_words = ((total_low_bits + 63) / 64) as usize;
            let low_offset = all_low_bits.len();
            all_low_bits.resize(low_offset + low_words, 0);
            for i in 0..count {
                if low_bit_width > 0 {
                    let delta = get_val(start + i) - min_val;
                    let low_val = delta & low_mask;
                    let bit_pos = i as u64 * low_bit_width as u64;
                    let word_idx = (bit_pos / 64) as usize;
                    let bit_idx = (bit_pos % 64) as u32;
                    all_low_bits[low_offset + word_idx] |= low_val << bit_idx;
                    if bit_idx + low_bit_width > 64 && word_idx + 1 < low_words {
                        all_low_bits[low_offset + word_idx + 1] |= low_val >> (64 - bit_idx);
                    }
                }
            }

            // Build high bits directly into flat array
            let last_delta = max_val - min_val;
            let max_high = last_delta >> low_bit_width;
            let high_len_bits = count + max_high as usize + 1;
            let high_words = (high_len_bits + 63) / 64;
            let high_offset = all_high_bits.len();
            all_high_bits.resize(high_offset + high_words, 0);
            let mut hpos = 0usize;
            let mut prev_high = 0u64;
            for i in 0..count {
                let delta = get_val(start + i) - min_val;
                let high = delta >> low_bit_width;
                hpos += (high - prev_high) as usize;
                let word_idx = hpos / 64;
                let bit_idx = hpos % 64;
                if word_idx < high_words {
                    all_high_bits[high_offset + word_idx] |= 1u64 << bit_idx;
                }
                hpos += 1;
                prev_high = high;
            }

            chunk_starts.push(start);
            chunk_upper_bounds.push(max_val);
            meta.push(PefChunkMeta {
                min_value: min_val,
                low_offset: low_offset as u32,
                high_offset: high_offset as u32,
                count: count as u16,
                high_len_bits: high_len_bits as u16,
                high_words: high_words as u16,
                low_words: low_words as u16,
                low_bit_width: low_bit_width as u8,
            });
            start = end;
        }

        // Padding word for branchless u128 extraction in chunk_get_low
        all_low_bits.push(0);

        Self { all_low_bits, all_high_bits, meta, chunk_starts, chunk_upper_bounds, len: n, universe }
    }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn size_bytes(&self) -> usize {
        self.all_low_bits.len() * 8
            + self.all_high_bits.len() * 8
            + self.meta.len() * std::mem::size_of::<PefChunkMeta>()
            + self.chunk_starts.len() * 8
            + self.chunk_upper_bounds.len() * 8
            + 56 // struct fields
    }

    #[inline]
    pub fn bits_per_element(&self) -> f64 {
        if self.len == 0 { return 0.0; }
        (self.size_bytes() * 8) as f64 / self.len as f64
    }

    /// Get the i-th element.
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len { return None; }
        // Binary search for the chunk containing this index
        let chunk_idx = match self.chunk_starts.binary_search(&index) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let view = self.chunk_view(chunk_idx);
        let local_idx = index - self.chunk_starts[chunk_idx];
        let delta = chunk_get_delta(&view, local_idx);
        Some(view.min_value + delta)
    }

    /// Find the first element >= target.
    ///
    /// Binary search on chunk upper bounds, then select0-based skip within
    /// the chunk to jump directly to the target's high-value bucket.
    #[inline]
    pub fn next_geq(&self, target: u64) -> Option<(usize, u64)> {
        if self.len == 0 || target >= self.universe { return None; }

        let chunk_idx = match self.chunk_upper_bounds.binary_search(&target) {
            Ok(i) => i,
            Err(i) => {
                if i >= self.meta.len() { return None; }
                i
            }
        };

        let view = self.chunk_view(chunk_idx);
        let global_offset = self.chunk_starts[chunk_idx];

        if target <= view.min_value {
            let delta = chunk_get_delta(&view, 0);
            return Some((global_offset, view.min_value + delta));
        }

        let target_delta = target - view.min_value;

        // Skip directly to the target_high bucket via select0 on high_bits.
        let target_high = (target_delta >> view.low_bit_width) as usize;
        let (start_idx, start_pos) = chunk_skip_to_high(&view, target_high);

        // Scan only from the skip point
        if let Some((local_idx, delta, _)) = chunk_scan_geq(&view, target_delta, start_idx, start_pos) {
            return Some((global_offset + local_idx, view.min_value + delta));
        }

        // Fallback to next chunk
        let next = chunk_idx + 1;
        if next < self.meta.len() {
            let nv = self.chunk_view(next);
            let delta = chunk_get_delta(&nv, 0);
            Some((self.chunk_starts[next], nv.min_value + delta))
        } else {
            None
        }
    }

    /// Iterator over all elements.
    pub fn iter(&self) -> OptimalPefIter<'_> {
        if self.is_empty() {
            return OptimalPefIter {
                opef: self,
                chunk_idx: 0,
                local_idx: 0,
                local_high_pos: 0,
                cached_high_bits: &[],
                cached_low_bits: &[],
                cached_low_bit_width: 0,
                cached_count: 0,
                cached_min_value: 0,
            };
        }
        let view = self.chunk_view(0);
        OptimalPefIter {
            opef: self,
            chunk_idx: 0,
            local_idx: 0,
            local_high_pos: 0,
            cached_high_bits: view.high_bits,
            cached_low_bits: view.low_bits,
            cached_low_bit_width: view.low_bit_width,
            cached_count: view.count,
            cached_min_value: view.min_value,
        }
    }

    /// Create a stateful cursor with `advance_to_geq` for posting list intersection.
    #[inline]
    pub fn cursor(&self) -> OptimalPefCursor<'_> {
        OptimalPefCursor::new(self)
    }

    /// Create a batch cursor for high-throughput sequential iteration.
    #[inline]
    pub fn batch_cursor(&self) -> OptimalPefBatchCursor<'_> {
        OptimalPefBatchCursor::new(self)
    }

    /// Create a lightweight borrowing view into chunk `idx`.
    #[inline]
    fn chunk_view(&self, idx: usize) -> ChunkView<'_> {
        let m = &self.meta[idx];
        let low_start = m.low_offset as usize;
        let low_end = (low_start + m.low_words as usize + 1).min(self.all_low_bits.len());
        let high_start = m.high_offset as usize;
        ChunkView {
            low_bits: &self.all_low_bits[low_start..low_end],
            high_bits: &self.all_high_bits[high_start..high_start + m.high_words as usize],
            low_bit_width: m.low_bit_width as u32,
            count: m.count as usize,
            min_value: m.min_value,
            high_len_bits: m.high_len_bits as usize,
        }
    }
}

/// Iterator over OptimalPartitionedEliasFano elements.
///
/// Caches current chunk's slice pointers to avoid per-element `chunk_view()` overhead.
pub struct OptimalPefIter<'a> {
    opef: &'a OptimalPartitionedEliasFano,
    chunk_idx: usize,
    local_idx: usize,
    local_high_pos: usize,
    // Cached from current chunk — refreshed on chunk transitions only
    cached_high_bits: &'a [u64],
    cached_low_bits: &'a [u64],
    cached_low_bit_width: u32,
    cached_count: usize,
    cached_min_value: u64,
}

impl<'a> OptimalPefIter<'a> {
    #[inline]
    fn refresh_chunk_cache(&mut self) {
        let view = self.opef.chunk_view(self.chunk_idx);
        self.cached_high_bits = view.high_bits;
        self.cached_low_bits = view.low_bits;
        self.cached_low_bit_width = view.low_bit_width;
        self.cached_count = view.count;
        self.cached_min_value = view.min_value;
    }
}

impl<'a> Iterator for OptimalPefIter<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk_idx >= self.opef.meta.len() { return None; }
        if self.local_idx >= self.cached_count {
            self.chunk_idx += 1;
            self.local_idx = 0;
            self.local_high_pos = 0;
            if self.chunk_idx >= self.opef.meta.len() { return None; }
            self.refresh_chunk_cache();
            return self.next();
        }

        let next_pos = self.local_high_pos;
        let mut word_idx = next_pos / 64;
        if word_idx >= self.cached_high_bits.len() { return None; }

        let bit_offset = next_pos % 64;
        let mut word = self.cached_high_bits[word_idx] >> bit_offset;

        if word != 0 {
            let tz = word.trailing_zeros() as usize;
            self.local_high_pos = word_idx * 64 + bit_offset + tz;
        } else {
            loop {
                word_idx += 1;
                if word_idx >= self.cached_high_bits.len() { return None; }
                word = self.cached_high_bits[word_idx];
                if word != 0 {
                    self.local_high_pos = word_idx * 64 + word.trailing_zeros() as usize;
                    break;
                }
            }
        }

        let high_val = (self.local_high_pos - self.local_idx) as u64;
        // Inline low-bit extraction using cached slice
        let low = if self.cached_low_bit_width == 0 {
            0
        } else {
            let lbw = self.cached_low_bit_width as u64;
            let bit_pos = self.local_idx as u64 * lbw;
            let w_idx = (bit_pos / 64) as usize;
            let b_idx = (bit_pos % 64) as u32;
            let w0 = self.cached_low_bits[w_idx];
            let w1 = self.cached_low_bits[w_idx + 1]; // safe: padding word
            let combined = w0 as u128 | ((w1 as u128) << 64);
            (combined >> b_idx) as u64 & ((1u64 << lbw) - 1)
        };
        let delta = (high_val << self.cached_low_bit_width) | low;
        let val = self.cached_min_value + delta;

        self.local_idx += 1;
        self.local_high_pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let consumed: usize = if self.chunk_idx < self.opef.meta.len() {
            self.opef.chunk_starts[self.chunk_idx] + self.local_idx
        } else {
            self.opef.len
        };
        let remaining = self.opef.len.saturating_sub(consumed);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for OptimalPefIter<'a> {}

/// Stateful cursor for O(1) amortized access over OptimalPartitionedEliasFano.
///
/// Caches chunk slice pointers for zero-overhead sequential access.
/// Provides `advance_to_geq` with same-chunk fast path + galloping search.
pub struct OptimalPefCursor<'a> {
    opef: &'a OptimalPartitionedEliasFano,
    chunk_idx: usize,
    local_idx: usize,
    local_high_pos: usize,
    global_idx: usize,
    cached_value: u64,
    // Cached from current chunk
    cached_high_bits: &'a [u64],
    cached_low_bits: &'a [u64],
    cached_low_bit_width: u32,
    cached_count: usize,
    cached_min_value: u64,
}

impl<'a> OptimalPefCursor<'a> {
    fn new(opef: &'a OptimalPartitionedEliasFano) -> Self {
        if opef.is_empty() {
            return Self {
                opef, chunk_idx: 0, local_idx: 0, local_high_pos: 0, global_idx: 0,
                cached_value: 0,
                cached_high_bits: &[], cached_low_bits: &[],
                cached_low_bit_width: 0, cached_count: 0, cached_min_value: 0,
            };
        }
        let view = opef.chunk_view(0);
        let mut high_pos = 0;
        for (word_idx, &word) in view.high_bits.iter().enumerate() {
            if word != 0 {
                high_pos = word_idx * 64 + word.trailing_zeros() as usize;
                break;
            }
        }
        let high_val = high_pos as u64;
        let low = chunk_get_low(&view, 0);
        let initial_val = view.min_value + (high_val << view.low_bit_width) + low;
        Self {
            opef, chunk_idx: 0, local_idx: 0, local_high_pos: high_pos, global_idx: 0,
            cached_value: initial_val,
            cached_high_bits: view.high_bits, cached_low_bits: view.low_bits,
            cached_low_bit_width: view.low_bit_width, cached_count: view.count,
            cached_min_value: view.min_value,
        }
    }

    #[inline]
    fn refresh_chunk_cache(&mut self) {
        let view = self.opef.chunk_view(self.chunk_idx);
        self.cached_high_bits = view.high_bits;
        self.cached_low_bits = view.low_bits;
        self.cached_low_bit_width = view.low_bit_width;
        self.cached_count = view.count;
        self.cached_min_value = view.min_value;
    }

    #[inline]
    fn get_low_cached(&self, local_idx: usize) -> u64 {
        if self.cached_low_bit_width == 0 { return 0; }
        let lbw = self.cached_low_bit_width as u64;
        let bit_pos = local_idx as u64 * lbw;
        let w_idx = (bit_pos / 64) as usize;
        let b_idx = (bit_pos % 64) as u32;
        let w0 = self.cached_low_bits[w_idx];
        let w1 = self.cached_low_bits[w_idx + 1];
        let combined = w0 as u128 | ((w1 as u128) << 64);
        (combined >> b_idx) as u64 & ((1u64 << lbw) - 1)
    }

    #[inline]
    fn recompute_value(&mut self) {
        let high_val = (self.local_high_pos - self.local_idx) as u64;
        let low = self.get_low_cached(self.local_idx);
        self.cached_value = self.cached_min_value + (high_val << self.cached_low_bit_width) + low;
    }

    /// Build a ChunkView from cached data (for helper functions).
    #[inline]
    fn cached_view(&self) -> ChunkView<'a> {
        ChunkView {
            low_bits: self.cached_low_bits,
            high_bits: self.cached_high_bits,
            low_bit_width: self.cached_low_bit_width,
            count: self.cached_count,
            min_value: self.cached_min_value,
            high_len_bits: self.opef.meta[self.chunk_idx].high_len_bits as usize,
        }
    }

    /// Current element value — O(1) from cached value.
    #[inline]
    pub fn current(&self) -> Option<u64> {
        if self.global_idx >= self.opef.len { None } else { Some(self.cached_value) }
    }

    #[inline]
    pub fn index(&self) -> usize { self.global_idx }

    #[inline]
    pub fn is_exhausted(&self) -> bool { self.global_idx >= self.opef.len }

    /// Advance to the next element.
    #[inline]
    pub fn advance(&mut self) -> bool {
        self.global_idx += 1;
        if self.global_idx >= self.opef.len { return false; }

        self.local_idx += 1;

        if self.local_idx >= self.cached_count {
            self.chunk_idx += 1;
            self.local_idx = 0;
            if self.chunk_idx >= self.opef.meta.len() { return false; }
            self.refresh_chunk_cache();
            self.local_high_pos = chunk_first_one_cached(self.cached_high_bits);
            self.recompute_value();
            return true;
        }

        let next_pos = self.local_high_pos + 1;
        let mut word_idx = next_pos / 64;
        if word_idx >= self.cached_high_bits.len() { return false; }

        let bit_in_word = next_pos % 64;
        let mut word = self.cached_high_bits[word_idx] >> bit_in_word;

        if word != 0 {
            self.local_high_pos = word_idx * 64 + bit_in_word + word.trailing_zeros() as usize;
        } else {
            loop {
                word_idx += 1;
                if word_idx >= self.cached_high_bits.len() { return false; }
                word = self.cached_high_bits[word_idx];
                if word != 0 {
                    self.local_high_pos = word_idx * 64 + word.trailing_zeros() as usize;
                    break;
                }
            }
        }
        self.recompute_value();
        true
    }

    /// Advance to the first element >= target.
    ///
    /// O(1) cached value check, same-chunk direct scan, galloping for cross-chunk.
    #[inline]
    pub fn advance_to_geq(&mut self, target: u64) -> bool {
        if self.global_idx >= self.opef.len { return false; }
        if self.cached_value >= target { return true; }

        let num_chunks = self.opef.meta.len();

        // Same-chunk: scan directly from cursor position
        if target <= self.opef.chunk_upper_bounds[self.chunk_idx] {
            let target_delta = target - self.cached_min_value;
            let view = self.cached_view();
            if let Some((local_idx, delta, hp)) = chunk_scan_geq(&view, target_delta, self.local_idx, self.local_high_pos) {
                self.local_idx = local_idx;
                self.global_idx = self.opef.chunk_starts[self.chunk_idx] + local_idx;
                self.local_high_pos = hp;
                self.cached_value = self.cached_min_value + delta;
                return true;
            }
        }

        // Galloping search forward from current chunk
        let mut lo = self.chunk_idx;
        let mut step = 1usize;
        let target_chunk = loop {
            let hi = (lo + step).min(num_chunks - 1);
            if self.opef.chunk_upper_bounds[hi] >= target {
                let s = lo + 1;
                if s > hi {
                    break hi;
                }
                break s + self.opef.chunk_upper_bounds[s..=hi]
                    .partition_point(|&x| x < target);
            }
            if hi == num_chunks - 1 {
                self.global_idx = self.opef.len;
                return false;
            }
            lo = hi;
            step *= 2;
        };

        // Reposition to target chunk
        self.chunk_idx = target_chunk;
        self.refresh_chunk_cache();
        let global_offset = self.opef.chunk_starts[target_chunk];

        if target <= self.cached_min_value {
            self.local_idx = 0;
            self.global_idx = global_offset;
            self.local_high_pos = chunk_first_one_cached(self.cached_high_bits);
            self.recompute_value();
            return true;
        }

        let view = self.cached_view();
        let target_delta = target - self.cached_min_value;
        let target_high = (target_delta >> view.low_bit_width) as usize;
        let (si, sp) = chunk_skip_to_high(&view, target_high);

        if let Some((local_idx, delta, hp)) = chunk_scan_geq(&view, target_delta, si, sp) {
            self.local_idx = local_idx;
            self.global_idx = global_offset + local_idx;
            self.local_high_pos = hp;
            self.cached_value = self.cached_min_value + delta;
            return true;
        }

        // First element of next chunk
        let next = target_chunk + 1;
        if next >= num_chunks {
            self.global_idx = self.opef.len;
            return false;
        }
        self.chunk_idx = next;
        self.refresh_chunk_cache();
        self.local_idx = 0;
        self.global_idx = self.opef.chunk_starts[next];
        self.local_high_pos = chunk_first_one_cached(self.cached_high_bits);
        self.recompute_value();
        true
    }

    /// Reposition the cursor directly to element at global index `idx`.
    /// Returns false (cursor unchanged) if `idx >= self.opef.len`.
    #[inline]
    pub fn advance_to_index(&mut self, idx: usize) -> bool {
        if idx >= self.opef.len { return false; }
        if idx == self.global_idx { return true; }

        // Binary search chunk_starts to find which chunk contains idx
        let chunk_idx = self.opef.chunk_starts.partition_point(|&s| s <= idx) - 1;
        let local_idx = idx - self.opef.chunk_starts[chunk_idx];

        // Switch chunk if needed
        if chunk_idx != self.chunk_idx {
            self.chunk_idx = chunk_idx;
            self.refresh_chunk_cache();
        }

        self.local_idx = local_idx;
        self.global_idx = idx;

        // Find local_high_pos by scanning for the local_idx-th 1-bit
        self.local_high_pos = Self::find_nth_one(self.cached_high_bits, local_idx);
        self.recompute_value();
        true
    }

    /// Find the position of the n-th set bit (0-indexed) in a high_bits slice.
    #[inline]
    fn find_nth_one(high_bits: &[u64], n: usize) -> usize {
        let mut remaining = n;
        for (word_idx, &word) in high_bits.iter().enumerate() {
            let ones = word.count_ones() as usize;
            if remaining < ones {
                return word_idx * 64 + select_in_word(word, remaining);
            }
            remaining -= ones;
        }
        0
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.opef);
    }
}

/// Batch cursor for OptimalPartitionedEliasFano.
pub struct OptimalPefBatchCursor<'a> {
    opef: &'a OptimalPartitionedEliasFano,
    buffer: [u64; BATCH_SIZE],
    buf_pos: usize,
    buf_len: usize,
    next_elem: usize,
    chunk_idx: usize,
    local_idx: usize,
    local_high_pos: usize,
}

impl<'a> OptimalPefBatchCursor<'a> {
    fn new(opef: &'a OptimalPartitionedEliasFano) -> Self {
        let mut cursor = Self {
            opef,
            buffer: [0; BATCH_SIZE],
            buf_pos: 0,
            buf_len: 0,
            next_elem: 0,
            chunk_idx: 0,
            local_idx: 0,
            local_high_pos: 0,
        };
        if !opef.is_empty() {
            cursor.refill();
        }
        cursor
    }

    #[inline]
    pub fn current(&self) -> Option<u64> {
        if self.buf_pos < self.buf_len {
            Some(self.buffer[self.buf_pos])
        } else {
            None
        }
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.next_elem - self.buf_len + self.buf_pos
    }

    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.buf_pos >= self.buf_len && self.next_elem >= self.opef.len
    }

    #[inline]
    pub fn advance(&mut self) -> bool {
        self.buf_pos += 1;
        if self.buf_pos < self.buf_len {
            return true;
        }
        if self.next_elem >= self.opef.len {
            return false;
        }
        self.refill();
        self.buf_len > 0
    }

    fn refill(&mut self) {
        let remaining = self.opef.len - self.next_elem;
        if remaining == 0 {
            self.buf_len = 0;
            return;
        }
        let count = remaining.min(BATCH_SIZE);
        let mut filled = 0;

        while filled < count {
            if self.chunk_idx >= self.opef.meta.len() { break; }
            let view = self.opef.chunk_view(self.chunk_idx);

            if self.local_idx == 0 {
                self.local_high_pos = 0;
                for (wi, &w) in view.high_bits.iter().enumerate() {
                    if w != 0 {
                        self.local_high_pos = wi * 64 + w.trailing_zeros() as usize;
                        break;
                    }
                }
            }

            let chunk_remaining = view.count - self.local_idx;
            let to_decode = (count - filled).min(chunk_remaining);

            let lbw = view.low_bit_width;
            let low_mask = if lbw == 0 { 0u64 } else { (1u64 << lbw) - 1 };

            // --- High-bit scanning state ---
            let mut bit_pos = self.local_high_pos;
            let mut word_idx = bit_pos / 64;
            let start_bit = bit_pos % 64;
            let mut word = if word_idx < view.high_bits.len() {
                view.high_bits[word_idx] >> start_bit
            } else {
                0
            };
            bit_pos = word_idx * 64 + start_bit;

            // --- Low-bit incremental state ---
            let mut low_bit_pos = self.local_idx as u64 * lbw as u64;
            let mut low_word_idx = (low_bit_pos / 64) as usize;
            let mut low_bit_idx = (low_bit_pos % 64) as u32;
            let mut low_word = if lbw > 0 && low_word_idx < view.low_bits.len() {
                view.low_bits[low_word_idx]
            } else { 0 };
            let mut low_word_next = if lbw > 0 && low_word_idx + 1 < view.low_bits.len() {
                view.low_bits[low_word_idx + 1]
            } else { 0 };

            for k in 0..to_decode {
                let li = self.local_idx + k;
                while word == 0 {
                    word_idx += 1;
                    if word_idx >= view.high_bits.len() { break; }
                    word = view.high_bits[word_idx];
                    bit_pos = word_idx * 64;
                }
                let tz = word.trailing_zeros() as usize;
                bit_pos += tz;
                word >>= tz;

                let high_val = (bit_pos - li) as u64;

                let low = if lbw == 0 {
                    0u64
                } else {
                    let mut val = low_word >> low_bit_idx;
                    if low_bit_idx + lbw > 64 {
                        val |= low_word_next << (64 - low_bit_idx);
                    }
                    val & low_mask
                };

                let delta = (high_val << lbw) | low;
                self.buffer[filled + k] = view.min_value + delta;

                bit_pos += 1;
                word >>= 1;

                if lbw > 0 {
                    low_bit_idx += lbw;
                    if low_bit_idx >= 64 {
                        low_bit_idx -= 64;
                        low_word_idx += 1;
                        low_word = low_word_next;
                        low_word_next = if low_word_idx + 1 < view.low_bits.len() {
                            view.low_bits[low_word_idx + 1]
                        } else { 0 };
                    }
                }
            }

            self.local_idx += to_decode;
            self.local_high_pos = bit_pos;
            filled += to_decode;

            if self.local_idx >= view.count {
                self.chunk_idx += 1;
                self.local_idx = 0;
                self.local_high_pos = 0;
            }
        }

        self.buf_pos = 0;
        self.buf_len = filled;
        self.next_elem += filled;
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.opef);
    }
}

// ============================================================================
// Hybrid Posting List (Phase 3)
// ============================================================================

/// Threshold below which a dense array is used instead of EF encoding.
const DENSE_THRESHOLD: usize = 64;

/// Threshold above which partitioned EF is preferred over plain EF.
const PARTITION_THRESHOLD: usize = 256;

/// Threshold above which DP-optimal partitioning is used.
const OPTIMAL_THRESHOLD: usize = 4096;

/// Adaptive posting list that selects the best encoding based on list statistics.
///
/// - **Dense** (< 64 elements): Raw `Vec<u32>` — zero decode cost, best for short lists.
/// - **EliasFano** (64-256 elements): Plain EF — good compression, no chunk overhead.
/// - **PartitionedEF** (256-4096 elements): Uniform 128-chunk PEF — cache-local `next_geq`.
/// - **OptimalPEF** (> 4096 elements): DP-optimal variable chunks — best compression.
///
/// All variants support the same query interface: `get`, `next_geq`, `iter`, `len`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HybridPostingList {
    /// Raw sorted array for very short lists.
    Dense(Vec<u32>),
    /// Plain Elias-Fano for medium lists.
    EliasFano(EliasFano),
    /// Uniform Partitioned EF for large lists.
    Partitioned(PartitionedEliasFano),
    /// DP-Optimal Partitioned EF for very large lists.
    Optimal(OptimalPartitionedEliasFano),
}

impl HybridPostingList {
    /// Build from a sorted slice, automatically selecting the best encoding.
    pub fn from_sorted(values: &[u32]) -> Self {
        let n = values.len();
        if n <= DENSE_THRESHOLD {
            Self::Dense(values.to_vec())
        } else if n <= PARTITION_THRESHOLD {
            Self::EliasFano(EliasFano::from_sorted(values))
        } else if n <= OPTIMAL_THRESHOLD {
            Self::Partitioned(PartitionedEliasFano::from_sorted(values))
        } else {
            Self::Optimal(OptimalPartitionedEliasFano::from_sorted(values))
        }
    }

    /// Build from sorted u64 values.
    pub fn from_sorted_u64(values: &[u64]) -> Self {
        let n = values.len();
        if n <= DENSE_THRESHOLD {
            Self::Dense(values.iter().map(|&v| v as u32).collect())
        } else if n <= PARTITION_THRESHOLD {
            Self::EliasFano(EliasFano::from_sorted_u64(values))
        } else if n <= OPTIMAL_THRESHOLD {
            Self::Partitioned(PartitionedEliasFano::from_sorted_u64(values))
        } else {
            Self::Optimal(OptimalPartitionedEliasFano::from_sorted_u64(values))
        }
    }

    /// Force a specific encoding regardless of list size.
    pub fn with_encoding(values: &[u32], encoding: PostingEncoding) -> Self {
        match encoding {
            PostingEncoding::Dense => Self::Dense(values.to_vec()),
            PostingEncoding::EliasFano => Self::EliasFano(EliasFano::from_sorted(values)),
            PostingEncoding::Partitioned => Self::Partitioned(PartitionedEliasFano::from_sorted(values)),
            PostingEncoding::Optimal => Self::Optimal(OptimalPartitionedEliasFano::from_sorted(values)),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::EliasFano(ef) => ef.len(),
            Self::Partitioned(pef) => pef.len(),
            Self::Optimal(opef) => opef.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Dense(v) => v.len() * 4 + 8, // discriminant + vec overhead
            Self::EliasFano(ef) => ef.size_bytes() + 8,
            Self::Partitioned(pef) => pef.size_bytes() + 8,
            Self::Optimal(opef) => opef.size_bytes() + 8,
        }
    }

    #[inline]
    pub fn bits_per_element(&self) -> f64 {
        if self.len() == 0 { return 0.0; }
        (self.size_bytes() * 8) as f64 / self.len() as f64
    }

    /// Get the i-th element.
    pub fn get(&self, index: usize) -> Option<u64> {
        match self {
            Self::Dense(v) => v.get(index).map(|&x| x as u64),
            Self::EliasFano(ef) => ef.get(index),
            Self::Partitioned(pef) => pef.get(index),
            Self::Optimal(opef) => opef.get(index),
        }
    }

    /// Find the first element >= target.
    #[inline]
    pub fn next_geq(&self, target: u64) -> Option<(usize, u64)> {
        match self {
            Self::Dense(v) => {
                // Binary search on sorted array
                match v.binary_search(&(target as u32)) {
                    Ok(i) => Some((i, v[i] as u64)),
                    Err(i) => {
                        if i < v.len() {
                            Some((i, v[i] as u64))
                        } else {
                            None
                        }
                    }
                }
            }
            Self::EliasFano(ef) => ef.next_geq(target),
            Self::Partitioned(pef) => pef.next_geq(target),
            Self::Optimal(opef) => opef.next_geq(target),
        }
    }

    /// Which encoding was selected.
    pub fn encoding(&self) -> PostingEncoding {
        match self {
            Self::Dense(_) => PostingEncoding::Dense,
            Self::EliasFano(_) => PostingEncoding::EliasFano,
            Self::Partitioned(_) => PostingEncoding::Partitioned,
            Self::Optimal(_) => PostingEncoding::Optimal,
        }
    }
}

/// Encoding strategy for posting lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PostingEncoding {
    Dense,
    EliasFano,
    Partitioned,
    Optimal,
}

impl std::fmt::Display for PostingEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense => write!(f, "Dense"),
            Self::EliasFano => write!(f, "EliasFano"),
            Self::Partitioned => write!(f, "PartitionedEF"),
            Self::Optimal => write!(f, "OptimalPEF"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let ef = EliasFano::from_sorted(&[]);
        assert_eq!(ef.len(), 0);
        assert!(ef.is_empty());
        assert_eq!(ef.get(0), None);
        assert_eq!(ef.next_geq(0), None);
    }

    #[test]
    fn test_single() {
        let ef = EliasFano::from_sorted(&[42]);
        assert_eq!(ef.len(), 1);
        assert_eq!(ef.get(0), Some(42));
        assert_eq!(ef.next_geq(0), Some((0, 42)));
        assert_eq!(ef.next_geq(42), Some((0, 42)));
        assert_eq!(ef.next_geq(43), None);
    }

    #[test]
    fn test_basic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), 8);

        // Random access
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(ef.get(i), Some(v as u64), "get({}) failed", i);
        }
        assert_eq!(ef.get(8), None);
    }

    #[test]
    fn test_next_geq() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);

        // Exact matches
        assert_eq!(ef.next_geq(3), Some((0, 3)));
        assert_eq!(ef.next_geq(42), Some((5, 42)));
        assert_eq!(ef.next_geq(63), Some((7, 63)));

        // Between values
        assert_eq!(ef.next_geq(0), Some((0, 3)));
        assert_eq!(ef.next_geq(4), Some((1, 5)));
        assert_eq!(ef.next_geq(10), Some((2, 11)));
        assert_eq!(ef.next_geq(28), Some((4, 31)));
        assert_eq!(ef.next_geq(59), Some((7, 63)));

        // Past end
        assert_eq!(ef.next_geq(64), None);
        assert_eq!(ef.next_geq(1000), None);
    }

    #[test]
    fn test_iterator() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);

        let collected: Vec<u64> = ef.iter().collect();
        let expected: Vec<u64> = docs.iter().map(|&v| v as u64).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_consecutive() {
        let docs: Vec<u32> = (0..100).collect();
        let ef = EliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), 100);
        for i in 0..100 {
            assert_eq!(ef.get(i), Some(i as u64));
        }
        assert_eq!(ef.next_geq(50), Some((50, 50)));
    }

    #[test]
    fn test_sparse() {
        let docs: Vec<u32> = (0..100).map(|i| i * 1000).collect();
        let ef = EliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), 100);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(ef.get(i), Some(v as u64), "get({}) failed", i);
        }

        assert_eq!(ef.next_geq(500), Some((1, 1000)));
        assert_eq!(ef.next_geq(1000), Some((1, 1000)));
        assert_eq!(ef.next_geq(1001), Some((2, 2000)));
    }

    #[test]
    fn test_large_posting_list() {
        // Simulate a posting list: 10K doc IDs in universe of 1M
        let docs: Vec<u32> = (0..10000).map(|i| i * 100 + i % 7).collect();
        let ef = EliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), 10000);

        // Verify all elements
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(ef.get(i), Some(v as u64), "get({}) = {:?}, expected {}", i, ef.get(i), v);
        }

        // Verify iterator matches
        let from_iter: Vec<u64> = ef.iter().collect();
        assert_eq!(from_iter.len(), 10000);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(from_iter[i], v as u64);
        }

        // Space efficiency
        let bits_per_elem = ef.bits_per_element();
        assert!(bits_per_elem < 32.0, "Should be much less than 32 bits/elem, got {:.1}", bits_per_elem);
    }

    #[test]
    fn test_next_geq_scan() {
        // Simulate posting list intersection via next_geq
        let docs: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let ef = EliasFano::from_sorted(&docs);

        // Find all multiples of 30 via next_geq
        let mut target = 0u64;
        let mut found = Vec::new();
        while let Some((_, val)) = ef.next_geq(target) {
            if val % 30 == 0 {
                found.push(val as u32);
            }
            target = val + 1;
        }

        let expected: Vec<u32> = (0..1000).map(|i| i * 10).filter(|v| v % 30 == 0).collect();
        assert_eq!(found, expected);
    }

    #[test]
    fn test_space_efficiency() {
        // 10K elements in universe of 1M should use ~12 bits/elem
        let docs: Vec<u32> = (0..10000).map(|i| i * 100).collect();
        let ef = EliasFano::from_sorted(&docs);

        let bpe = ef.bits_per_element();
        eprintln!("Elias-Fano: {} elements, {:.1} bits/elem, {} bytes total",
            ef.len(), bpe, ef.size_bytes());

        // Theoretical: 2 + log2(1M/10K) ≈ 2 + 6.6 ≈ 8.6 bits
        // Practical overhead pushes to ~10-15 bits
        assert!(bpe < 20.0, "Too many bits per element: {:.1}", bpe);
    }

    #[test]
    fn test_max_values() {
        let docs = vec![0, u32::MAX / 2, u32::MAX];
        let ef = EliasFano::from_sorted(&docs);
        assert_eq!(ef.get(0), Some(0));
        assert_eq!(ef.get(1), Some(u32::MAX as u64 / 2));
        assert_eq!(ef.get(2), Some(u32::MAX as u64));
    }

    /// Performance — release only.
    #[test]
    fn test_performance_next_geq() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        let ef = EliasFano::from_sorted(&docs);

        let targets: Vec<u64> = (0..10000).map(|i| (i * 100) as u64).collect();

        let start = std::time::Instant::now();
        let mut found = 0usize;
        for _ in 0..100 {
            for &t in &targets {
                if ef.next_geq(t).is_some() { found += 1; }
            }
        }
        let elapsed = start.elapsed();

        #[cfg(not(debug_assertions))]
        {
            let per_call = elapsed / (100 * targets.len() as u32);
            eprintln!("Elias-Fano next_geq: {:?}/call, {} found", per_call, found);
        }
    }

    // --- Cursor tests ---

    #[test]
    fn test_cursor_basic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);
        let mut cursor = ef.cursor();

        // Sequential access matches get()
        for (i, &v) in docs.iter().enumerate() {
            assert!(!cursor.is_exhausted());
            assert_eq!(cursor.index(), i);
            assert_eq!(cursor.current(), Some(v as u64), "cursor at {} failed", i);
            if i < docs.len() - 1 {
                assert!(cursor.advance());
            }
        }
        assert!(!cursor.advance()); // past end
        assert!(cursor.is_exhausted());
    }

    #[test]
    fn test_cursor_advance_to_geq() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);
        let mut cursor = ef.cursor();

        // Jump to >= 30
        assert!(cursor.advance_to_geq(30));
        assert_eq!(cursor.current(), Some(31));

        // Already >= 42 from current pos? No, 31 < 42
        assert!(cursor.advance_to_geq(42));
        assert_eq!(cursor.current(), Some(42));

        // Jump past end
        assert!(!cursor.advance_to_geq(100));
        assert!(cursor.is_exhausted());
    }

    #[test]
    fn test_cursor_matches_iterator() {
        let docs: Vec<u32> = (0..10000).map(|i| i * 10 + i % 7).collect();
        let ef = EliasFano::from_sorted(&docs);

        // Cursor should produce same values as iterator
        let from_iter: Vec<u64> = ef.iter().collect();
        let mut from_cursor = Vec::new();
        let mut cursor = ef.cursor();
        if let Some(v) = cursor.current() {
            from_cursor.push(v);
            while cursor.advance() {
                from_cursor.push(cursor.current().unwrap());
            }
        }

        assert_eq!(from_cursor.len(), from_iter.len());
        for (i, (&a, &b)) in from_cursor.iter().zip(from_iter.iter()).enumerate() {
            assert_eq!(a, b, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_cursor_reset() {
        let docs = vec![10, 20, 30];
        let ef = EliasFano::from_sorted(&docs);
        let mut cursor = ef.cursor();

        cursor.advance();
        cursor.advance();
        assert_eq!(cursor.current(), Some(30));

        cursor.reset();
        assert_eq!(cursor.current(), Some(10));
        assert_eq!(cursor.index(), 0);
    }

    /// Performance: cursor sequential vs get() sequential
    #[test]
    fn test_cursor_performance() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        let ef = EliasFano::from_sorted(&docs);

        #[cfg(not(debug_assertions))]
        {
            // get(i) sequential
            let start = std::time::Instant::now();
            let mut sum1 = 0u64;
            for _ in 0..10 {
                for i in 0..ef.len() {
                    sum1 += ef.get(i).unwrap();
                }
            }
            let get_time = start.elapsed();

            // cursor sequential
            let start = std::time::Instant::now();
            let mut sum2 = 0u64;
            for _ in 0..10 {
                let mut cursor = ef.cursor();
                sum2 += cursor.current().unwrap();
                while cursor.advance() {
                    sum2 += cursor.current().unwrap();
                }
            }
            let cursor_time = start.elapsed();

            assert_eq!(sum1, sum2, "cursor and get must produce same sum");

            let speedup = get_time.as_nanos() as f64 / cursor_time.as_nanos() as f64;
            eprintln!("Sequential 100K: get={:?}, cursor={:?}, speedup={:.1}×",
                get_time, cursor_time, speedup);

            assert!(speedup > 2.0,
                "cursor should be at least 2× faster than get(), got {:.1}×", speedup);
        }
    }

    // ========================================================================
    // PartitionedEliasFano tests
    // ========================================================================

    #[test]
    fn test_pef_empty() {
        let pef = PartitionedEliasFano::from_sorted(&[]);
        assert_eq!(pef.len(), 0);
        assert!(pef.is_empty());
        assert_eq!(pef.get(0), None);
        assert_eq!(pef.next_geq(0), None);
    }

    #[test]
    fn test_pef_single() {
        let pef = PartitionedEliasFano::from_sorted(&[42]);
        assert_eq!(pef.len(), 1);
        assert_eq!(pef.get(0), Some(42));
        assert_eq!(pef.next_geq(0), Some((0, 42)));
        assert_eq!(pef.next_geq(42), Some((0, 42)));
        assert_eq!(pef.next_geq(43), None);
    }

    #[test]
    fn test_pef_basic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let pef = PartitionedEliasFano::from_sorted(&docs);
        assert_eq!(pef.len(), 8);

        // Random access
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(pef.get(i), Some(v as u64), "get({}) failed", i);
        }
        assert_eq!(pef.get(8), None);
    }

    #[test]
    fn test_pef_next_geq() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let pef = PartitionedEliasFano::from_sorted(&docs);

        // Exact matches
        assert_eq!(pef.next_geq(3), Some((0, 3)));
        assert_eq!(pef.next_geq(42), Some((5, 42)));
        assert_eq!(pef.next_geq(63), Some((7, 63)));

        // Between values
        assert_eq!(pef.next_geq(0), Some((0, 3)));
        assert_eq!(pef.next_geq(4), Some((1, 5)));
        assert_eq!(pef.next_geq(10), Some((2, 11)));
        assert_eq!(pef.next_geq(28), Some((4, 31)));
        assert_eq!(pef.next_geq(59), Some((7, 63)));

        // Past end
        assert_eq!(pef.next_geq(64), None);
        assert_eq!(pef.next_geq(1000), None);
    }

    #[test]
    fn test_pef_iterator() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let pef = PartitionedEliasFano::from_sorted(&docs);

        let collected: Vec<u64> = pef.iter().collect();
        let expected: Vec<u64> = docs.iter().map(|&v| v as u64).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_pef_consecutive() {
        let docs: Vec<u32> = (0..100).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        assert_eq!(pef.len(), 100);
        for i in 0..100 {
            assert_eq!(pef.get(i), Some(i as u64));
        }
        assert_eq!(pef.next_geq(50), Some((50, 50)));
    }

    #[test]
    fn test_pef_sparse() {
        let docs: Vec<u32> = (0..100).map(|i| i * 1000).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        assert_eq!(pef.len(), 100);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(pef.get(i), Some(v as u64), "get({}) failed", i);
        }

        assert_eq!(pef.next_geq(500), Some((1, 1000)));
        assert_eq!(pef.next_geq(1000), Some((1, 1000)));
        assert_eq!(pef.next_geq(1001), Some((2, 2000)));
    }

    #[test]
    fn test_pef_multi_chunk() {
        // Force multiple chunks: 300 elements > 128 chunk size
        let docs: Vec<u32> = (0..300).map(|i| i * 10 + i % 7).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        assert_eq!(pef.len(), 300);

        // Verify all elements via get()
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(pef.get(i), Some(v as u64), "get({}) failed", i);
        }

        // Verify iterator matches
        let from_iter: Vec<u64> = pef.iter().collect();
        assert_eq!(from_iter.len(), 300);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(from_iter[i], v as u64, "iter mismatch at {}", i);
        }

        // Verify next_geq across chunk boundaries
        for &v in &docs {
            let (idx, found) = pef.next_geq(v as u64).unwrap();
            assert_eq!(found, v as u64, "next_geq({}) returned {}", v, found);
            assert_eq!(pef.get(idx), Some(found));
        }
    }

    #[test]
    fn test_pef_matches_ef() {
        // Verify PEF produces same results as plain EF for all operations
        let docs: Vec<u32> = (0..500).map(|i| i * 10 + i % 7).collect();
        let ef = EliasFano::from_sorted(&docs);
        let pef = PartitionedEliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), pef.len());

        // get() must match
        for i in 0..docs.len() {
            assert_eq!(ef.get(i), pef.get(i), "get({}) mismatch", i);
        }

        // next_geq() must match
        for target in (0..5010).step_by(7) {
            assert_eq!(
                ef.next_geq(target), pef.next_geq(target),
                "next_geq({}) mismatch", target
            );
        }

        // iter() must match
        let ef_iter: Vec<u64> = ef.iter().collect();
        let pef_iter: Vec<u64> = pef.iter().collect();
        assert_eq!(ef_iter, pef_iter);
    }

    #[test]
    fn test_pef_large_posting_list() {
        // 10K doc IDs in universe of 1M — realistic posting list
        let docs: Vec<u32> = (0..10000).map(|i| i * 100 + i % 7).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        assert_eq!(pef.len(), 10000);

        // Sample verification
        for i in (0..10000).step_by(100) {
            assert_eq!(pef.get(i), Some(docs[i] as u64), "get({}) failed", i);
        }

        // Iterator must produce all elements
        let from_iter: Vec<u64> = pef.iter().collect();
        assert_eq!(from_iter.len(), 10000);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(from_iter[i], v as u64, "iter[{}] mismatch", i);
        }
    }

    #[test]
    fn test_pef_cursor_basic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let mut cursor = pef.cursor();

        for (i, &v) in docs.iter().enumerate() {
            assert!(!cursor.is_exhausted());
            assert_eq!(cursor.index(), i);
            assert_eq!(cursor.current(), Some(v as u64), "cursor at {} failed", i);
            if i < docs.len() - 1 {
                assert!(cursor.advance());
            }
        }
        assert!(!cursor.advance());
        assert!(cursor.is_exhausted());
    }

    #[test]
    fn test_pef_cursor_multi_chunk() {
        let docs: Vec<u32> = (0..300).map(|i| i * 10).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let mut cursor = pef.cursor();

        let mut collected = Vec::new();
        if let Some(v) = cursor.current() {
            collected.push(v);
            while cursor.advance() {
                collected.push(cursor.current().unwrap());
            }
        }

        let expected: Vec<u64> = docs.iter().map(|&v| v as u64).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_pef_cursor_advance_to_geq() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let mut cursor = pef.cursor();

        // Jump across chunks
        assert!(cursor.advance_to_geq(1500));
        assert_eq!(cursor.current(), Some(1500));

        assert!(cursor.advance_to_geq(3005));
        assert_eq!(cursor.current(), Some(3010));

        assert!(!cursor.advance_to_geq(5000));
        assert!(cursor.is_exhausted());
    }

    #[test]
    fn test_pef_cursor_reset() {
        let docs = vec![10, 20, 30];
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let mut cursor = pef.cursor();

        cursor.advance();
        cursor.advance();
        assert_eq!(cursor.current(), Some(30));

        cursor.reset();
        assert_eq!(cursor.current(), Some(10));
        assert_eq!(cursor.index(), 0);
    }

    #[test]
    fn test_pef_space_efficiency() {
        let docs: Vec<u32> = (0..10000).map(|i| i * 100).collect();
        let ef = EliasFano::from_sorted(&docs);
        let pef = PartitionedEliasFano::from_sorted(&docs);

        eprintln!("EF:  {} elements, {:.1} bits/elem, {} bytes",
            ef.len(), ef.bits_per_element(), ef.size_bytes());
        eprintln!("PEF: {} elements, {:.1} bits/elem, {} bytes",
            pef.len(), pef.bits_per_element(), pef.size_bytes());

        // PEF should be within 2x of EF space (chunk overhead)
        assert!(pef.bits_per_element() < ef.bits_per_element() * 2.5,
            "PEF too large: {:.1} vs EF {:.1}", pef.bits_per_element(), ef.bits_per_element());
    }

    #[test]
    fn test_pef_max_values() {
        let docs = vec![0, u32::MAX / 2, u32::MAX];
        let pef = PartitionedEliasFano::from_sorted(&docs);
        assert_eq!(pef.get(0), Some(0));
        assert_eq!(pef.get(1), Some(u32::MAX as u64 / 2));
        assert_eq!(pef.get(2), Some(u32::MAX as u64));
    }

    #[test]
    fn test_pef_next_geq_scan() {
        let docs: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        let mut target = 0u64;
        let mut found = Vec::new();
        while let Some((_, val)) = pef.next_geq(target) {
            if val % 30 == 0 {
                found.push(val as u32);
            }
            target = val + 1;
        }

        let expected: Vec<u32> = (0..1000).map(|i| i * 10).filter(|v| v % 30 == 0).collect();
        assert_eq!(found, expected);
    }

    /// Performance: PEF next_geq vs plain EF next_geq — release only.
    #[test]
    fn test_pef_performance_next_geq() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        let ef = EliasFano::from_sorted(&docs);
        let pef = PartitionedEliasFano::from_sorted(&docs);

        let targets: Vec<u64> = (0..10000).map(|i| (i * 100) as u64).collect();

        #[cfg(not(debug_assertions))]
        {
            let mut sink = 0usize;

            // Warmup
            for &t in &targets {
                if ef.next_geq(t).is_some() { sink += 1; }
                if pef.next_geq(t).is_some() { sink += 1; }
            }

            let iterations = 100;

            let start = std::time::Instant::now();
            for _ in 0..iterations {
                for &t in &targets {
                    if ef.next_geq(t).is_some() { sink += 1; }
                }
            }
            let ef_time = start.elapsed();

            let start = std::time::Instant::now();
            for _ in 0..iterations {
                for &t in &targets {
                    if pef.next_geq(t).is_some() { sink += 1; }
                }
            }
            let pef_time = start.elapsed();

            let ef_ns = ef_time.as_nanos() as f64 / (iterations as f64 * targets.len() as f64);
            let pef_ns = pef_time.as_nanos() as f64 / (iterations as f64 * targets.len() as f64);
            let ratio = ef_ns / pef_ns;

            eprintln!(
                "next_geq 100K elements: EF={ef_ns:.1}ns, PEF={pef_ns:.1}ns, \
                 PEF is {ratio:.2}× (>1 = PEF faster) [sink={sink}]"
            );
        }
    }

    /// Performance: PEF cursor vs plain EF cursor — release only.
    #[test]
    fn test_pef_performance_cursor() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        let ef = EliasFano::from_sorted(&docs);
        let pef = PartitionedEliasFano::from_sorted(&docs);

        #[cfg(not(debug_assertions))]
        {
            let iterations = 10;

            // EF cursor
            let start = std::time::Instant::now();
            let mut sum1 = 0u64;
            for _ in 0..iterations {
                let mut cursor = ef.cursor();
                sum1 += cursor.current().unwrap();
                while cursor.advance() {
                    sum1 += cursor.current().unwrap();
                }
            }
            let ef_time = start.elapsed();

            // PEF cursor
            let start = std::time::Instant::now();
            let mut sum2 = 0u64;
            for _ in 0..iterations {
                let mut cursor = pef.cursor();
                sum2 += cursor.current().unwrap();
                while cursor.advance() {
                    sum2 += cursor.current().unwrap();
                }
            }
            let pef_time = start.elapsed();

            assert_eq!(sum1, sum2, "cursor sums must match");

            let ratio = ef_time.as_nanos() as f64 / pef_time.as_nanos() as f64;
            eprintln!(
                "Cursor 100K: EF={:?}, PEF={:?}, PEF is {ratio:.2}×",
                ef_time, pef_time
            );
        }
    }

    // ========================================================================
    // Batch Cursor tests
    // ========================================================================

    #[test]
    fn test_batch_cursor_ef_basic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);
        let mut bc = ef.batch_cursor();

        let mut collected = Vec::new();
        if let Some(v) = bc.current() {
            collected.push(v);
            while bc.advance() {
                collected.push(bc.current().unwrap());
            }
        }

        let expected: Vec<u64> = docs.iter().map(|&v| v as u64).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_batch_cursor_ef_matches_iter() {
        let docs: Vec<u32> = (0..10000).map(|i| i * 10 + i % 7).collect();
        let ef = EliasFano::from_sorted(&docs);

        let from_iter: Vec<u64> = ef.iter().collect();
        let mut from_batch = Vec::new();
        let mut bc = ef.batch_cursor();
        if let Some(v) = bc.current() {
            from_batch.push(v);
            while bc.advance() {
                from_batch.push(bc.current().unwrap());
            }
        }

        assert_eq!(from_batch.len(), from_iter.len());
        assert_eq!(from_batch, from_iter);
    }

    #[test]
    fn test_batch_cursor_ef_index() {
        let docs: Vec<u32> = (0..100).map(|i| i * 5).collect();
        let ef = EliasFano::from_sorted(&docs);
        let mut bc = ef.batch_cursor();

        for i in 0..100 {
            assert_eq!(bc.index(), i, "index mismatch at {}", i);
            assert_eq!(bc.current(), Some(docs[i] as u64));
            if i < 99 { bc.advance(); }
        }
    }

    #[test]
    fn test_batch_cursor_ef_reset() {
        let docs = vec![10, 20, 30, 40, 50];
        let ef = EliasFano::from_sorted(&docs);
        let mut bc = ef.batch_cursor();

        bc.advance();
        bc.advance();
        assert_eq!(bc.current(), Some(30));

        bc.reset();
        assert_eq!(bc.current(), Some(10));
        assert_eq!(bc.index(), 0);
    }

    #[test]
    fn test_batch_cursor_pef_matches_iter() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10 + i % 7).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        let from_iter: Vec<u64> = pef.iter().collect();
        let mut from_batch = Vec::new();
        let mut bc = pef.batch_cursor();
        if let Some(v) = bc.current() {
            from_batch.push(v);
            while bc.advance() {
                from_batch.push(bc.current().unwrap());
            }
        }

        assert_eq!(from_batch.len(), from_iter.len());
        assert_eq!(from_batch, from_iter);
    }

    #[test]
    fn test_batch_cursor_pef_cross_chunk() {
        // 300 elements spans 3 chunks of 128
        let docs: Vec<u32> = (0..300).map(|i| i * 10).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        let mut bc = pef.batch_cursor();
        let mut count = 0;
        if bc.current().is_some() {
            count += 1;
            while bc.advance() { count += 1; }
        }
        assert_eq!(count, 300);
    }

    /// Performance: batch cursor vs regular cursor — release only.
    #[test]
    fn test_batch_cursor_performance() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        let ef = EliasFano::from_sorted(&docs);

        #[cfg(not(debug_assertions))]
        {
            let iterations = 10;

            // Regular cursor
            let start = std::time::Instant::now();
            let mut sum1 = 0u64;
            for _ in 0..iterations {
                let mut cursor = ef.cursor();
                sum1 += cursor.current().unwrap();
                while cursor.advance() {
                    sum1 += cursor.current().unwrap();
                }
            }
            let cursor_time = start.elapsed();

            // Batch cursor
            let start = std::time::Instant::now();
            let mut sum2 = 0u64;
            for _ in 0..iterations {
                let mut bc = ef.batch_cursor();
                sum2 += bc.current().unwrap();
                while bc.advance() {
                    sum2 += bc.current().unwrap();
                }
            }
            let batch_time = start.elapsed();

            assert_eq!(sum1, sum2, "batch cursor and cursor must produce same sum");

            eprintln!(
                "Sequential 100K: cursor={:?}, batch={:?}, ratio={:.2}×",
                cursor_time, batch_time,
                cursor_time.as_nanos() as f64 / batch_time.as_nanos() as f64
            );
        }
    }

    // ========================================================================
    // DP-Optimal PEF tests
    // ========================================================================

    #[test]
    fn test_opef_empty() {
        let opef = OptimalPartitionedEliasFano::from_sorted(&[]);
        assert_eq!(opef.len(), 0);
        assert!(opef.is_empty());
        assert_eq!(opef.get(0), None);
        assert_eq!(opef.next_geq(0), None);
    }

    #[test]
    fn test_opef_small() {
        // Below MIN_CHUNK_SIZE — should still work (last chunk exception)
        let docs = vec![3, 5, 11, 27, 31];
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);
        assert_eq!(opef.len(), 5);

        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(opef.get(i), Some(v as u64), "get({}) failed", i);
        }
    }

    #[test]
    fn test_opef_matches_ef() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10 + i % 7).collect();
        let ef = EliasFano::from_sorted(&docs);
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), opef.len());

        // get() must match
        for i in 0..docs.len() {
            assert_eq!(ef.get(i), opef.get(i), "get({}) mismatch: ef={:?} opef={:?}",
                i, ef.get(i), opef.get(i));
        }

        // next_geq() must match
        for target in (0..5010).step_by(7) {
            assert_eq!(
                ef.next_geq(target), opef.next_geq(target),
                "next_geq({}) mismatch", target
            );
        }

        // iter() must match
        let ef_vals: Vec<u64> = ef.iter().collect();
        let opef_vals: Vec<u64> = opef.iter().collect();
        assert_eq!(ef_vals, opef_vals);
    }

    #[test]
    fn test_opef_large() {
        let docs: Vec<u32> = (0..10000).map(|i| i * 100 + i % 7).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        assert_eq!(opef.len(), 10000);

        // Sample verification
        for i in (0..10000).step_by(100) {
            assert_eq!(opef.get(i), Some(docs[i] as u64), "get({}) failed", i);
        }

        // Iterator
        let from_iter: Vec<u64> = opef.iter().collect();
        assert_eq!(from_iter.len(), 10000);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(from_iter[i], v as u64, "iter[{}] mismatch", i);
        }
    }

    #[test]
    fn test_opef_space_vs_uniform() {
        let docs: Vec<u32> = (0..10000).map(|i| i * 100).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        eprintln!("Uniform PEF: {:.1} bits/elem, {} bytes",
            pef.bits_per_element(), pef.size_bytes());
        eprintln!("Optimal PEF: {:.1} bits/elem, {} bytes",
            opef.bits_per_element(), opef.size_bytes());

        // Optimal should be no worse than 1.5× uniform (usually better)
        assert!(opef.bits_per_element() < pef.bits_per_element() * 1.5,
            "Optimal PEF too large: {:.1} vs uniform {:.1}",
            opef.bits_per_element(), pef.bits_per_element());
    }

    #[test]
    fn test_opef_next_geq() {
        let docs: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        assert_eq!(opef.next_geq(0), Some((0, 0)));
        assert_eq!(opef.next_geq(55), Some((6, 60)));
        assert_eq!(opef.next_geq(9990), Some((999, 9990)));
        assert_eq!(opef.next_geq(9991), None);
    }

    #[test]
    fn test_opef_cursor_sequential() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10 + i % 7).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        // Cursor sequential should match iterator
        let from_iter: Vec<u64> = opef.iter().collect();
        let mut from_cursor = Vec::new();
        let mut cursor = opef.cursor();
        if let Some(v) = cursor.current() {
            from_cursor.push(v);
            while cursor.advance() {
                from_cursor.push(cursor.current().unwrap());
            }
        }
        assert_eq!(from_cursor, from_iter);
    }

    #[test]
    fn test_opef_cursor_advance_to_geq() {
        let docs: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);
        let mut cursor = opef.cursor();

        // Same-chunk fast path
        assert!(cursor.advance_to_geq(50));
        assert_eq!(cursor.current(), Some(50));

        // Advance further
        assert!(cursor.advance_to_geq(100));
        assert_eq!(cursor.current(), Some(100));

        // Already at target
        assert!(cursor.advance_to_geq(100));
        assert_eq!(cursor.current(), Some(100));

        // Cross-chunk jump
        assert!(cursor.advance_to_geq(5000));
        assert_eq!(cursor.current(), Some(5000));

        // Near end
        assert!(cursor.advance_to_geq(9990));
        assert_eq!(cursor.current(), Some(9990));

        // Past end
        assert!(!cursor.advance_to_geq(10000));
    }

    #[test]
    fn test_opef_cursor_advance_to_geq_sorted_targets() {
        // Simulates posting list intersection: sorted targets
        let docs: Vec<u32> = (0..1000).map(|i| i * 7 + i % 3).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);
        let targets: Vec<u64> = (0..200).map(|i| (i * 35) as u64).collect();

        let mut cursor = opef.cursor();
        let mut cursor_results = Vec::new();
        for &t in &targets {
            if cursor.advance_to_geq(t) {
                cursor_results.push(cursor.current().unwrap());
            }
        }

        // Verify against stateless next_geq
        let mut stateless_results = Vec::new();
        for &t in &targets {
            if let Some((_, v)) = opef.next_geq(t as u64) {
                stateless_results.push(v);
            }
        }

        // Cursor should give same or later results (since it maintains position)
        // For sorted targets, results should match
        assert_eq!(cursor_results.len(), stateless_results.len());
        for (c, s) in cursor_results.iter().zip(stateless_results.iter()) {
            assert!(*c >= *s, "cursor {} should be >= stateless {}", c, s);
        }
    }

    #[test]
    fn test_opef_batch_cursor() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10 + i % 7).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        let from_iter: Vec<u64> = opef.iter().collect();
        let mut from_batch = Vec::new();
        let mut bc = opef.batch_cursor();
        if let Some(v) = bc.current() {
            from_batch.push(v);
            while bc.advance() {
                from_batch.push(bc.current().unwrap());
            }
        }

        assert_eq!(from_batch, from_iter);
    }

    // ========================================================================
    // Hybrid Posting List tests
    // ========================================================================

    #[test]
    fn test_hybrid_dense() {
        let docs: Vec<u32> = vec![1, 5, 10, 20, 50];
        let h = HybridPostingList::from_sorted(&docs);
        assert_eq!(h.encoding(), PostingEncoding::Dense);
        assert_eq!(h.len(), 5);
        assert_eq!(h.get(0), Some(1));
        assert_eq!(h.get(4), Some(50));
        assert_eq!(h.next_geq(6), Some((2, 10)));
    }

    #[test]
    fn test_hybrid_ef() {
        let docs: Vec<u32> = (0..100).map(|i| i * 10).collect();
        let h = HybridPostingList::from_sorted(&docs);
        assert_eq!(h.encoding(), PostingEncoding::EliasFano);
        assert_eq!(h.len(), 100);
        assert_eq!(h.get(50), Some(500));
        assert_eq!(h.next_geq(55), Some((6, 60)));
    }

    #[test]
    fn test_hybrid_partitioned() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10).collect();
        let h = HybridPostingList::from_sorted(&docs);
        assert_eq!(h.encoding(), PostingEncoding::Partitioned);
        assert_eq!(h.len(), 500);
        assert_eq!(h.get(0), Some(0));
        assert_eq!(h.next_geq(4990), Some((499, 4990)));
        assert_eq!(h.next_geq(4991), None);
    }

    #[test]
    fn test_hybrid_optimal() {
        let docs: Vec<u32> = (0..5000).map(|i| i * 10).collect();
        let h = HybridPostingList::from_sorted(&docs);
        assert_eq!(h.encoding(), PostingEncoding::Optimal);
        assert_eq!(h.len(), 5000);

        // Verify correctness matches plain EF
        let ef = EliasFano::from_sorted(&docs);
        for i in (0..5000).step_by(100) {
            assert_eq!(h.get(i), ef.get(i), "get({}) mismatch", i);
        }
        for t in (0..50000).step_by(77) {
            assert_eq!(h.next_geq(t), ef.next_geq(t), "next_geq({}) mismatch", t);
        }
    }

    #[test]
    fn test_hybrid_force_encoding() {
        let docs: Vec<u32> = (0..100).map(|i| i * 10).collect();

        let dense = HybridPostingList::with_encoding(&docs, PostingEncoding::Dense);
        assert_eq!(dense.encoding(), PostingEncoding::Dense);

        let ef = HybridPostingList::with_encoding(&docs, PostingEncoding::EliasFano);
        assert_eq!(ef.encoding(), PostingEncoding::EliasFano);

        // Both should give same results
        for i in 0..100 {
            assert_eq!(dense.get(i), ef.get(i));
        }
    }

    #[test]
    fn test_hybrid_empty() {
        let h = HybridPostingList::from_sorted(&[]);
        assert_eq!(h.encoding(), PostingEncoding::Dense);
        assert_eq!(h.len(), 0);
        assert_eq!(h.get(0), None);
        assert_eq!(h.next_geq(0), None);
    }

    /// Performance comparison across encodings — release only.
    #[test]
    fn test_hybrid_performance_comparison() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        let targets: Vec<u64> = (0..10000).map(|i| (i * 100) as u64).collect();

        let dense = HybridPostingList::with_encoding(&docs, PostingEncoding::Dense);
        let ef = HybridPostingList::with_encoding(&docs, PostingEncoding::EliasFano);
        let pef = HybridPostingList::with_encoding(&docs, PostingEncoding::Partitioned);
        let opef = HybridPostingList::with_encoding(&docs, PostingEncoding::Optimal);

        #[cfg(not(debug_assertions))]
        {
            let iters = 50;
            for (name, h) in [("Dense", &dense), ("EF", &ef), ("PEF", &pef), ("OPEF", &opef)] {
                let start = std::time::Instant::now();
                let mut sink = 0usize;
                for _ in 0..iters {
                    for &t in &targets {
                        if h.next_geq(t).is_some() { sink += 1; }
                    }
                }
                let elapsed = start.elapsed();
                let per_call = elapsed.as_nanos() as f64 / (iters as f64 * targets.len() as f64);
                eprintln!("{name}: {per_call:.1}ns/call, {:.1} bits/elem [sink={sink}]",
                    h.bits_per_element());
            }
        }
    }

    #[test]
    fn test_cursor_advance_to_index_basic() {
        let vals = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // advance_to_index(0) matches get(0)
        assert!(cursor.advance_to_index(0));
        assert_eq!(cursor.current(), Some(3));
        assert_eq!(cursor.index(), 0);

        // advance_to_index(7) matches get(7) — last element
        assert!(cursor.advance_to_index(7));
        assert_eq!(cursor.current(), Some(63));
        assert_eq!(cursor.index(), 7);

        // advance_to_index(8) — past end
        assert!(!cursor.advance_to_index(8));
        // Cursor should be unchanged
        assert_eq!(cursor.index(), 7);
        assert_eq!(cursor.current(), Some(63));
    }

    #[test]
    fn test_cursor_advance_to_index_then_advance() {
        let vals = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Jump to index 5, then advance should give index 6
        assert!(cursor.advance_to_index(5));
        assert_eq!(cursor.current(), Some(42));
        assert!(cursor.advance());
        assert_eq!(cursor.current(), Some(58));
        assert_eq!(cursor.index(), 6);
    }

    #[test]
    fn test_cursor_advance_to_index_then_geq() {
        let vals = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Jump to index 2, then advance_to_geq(40) should find 42 at index 5
        assert!(cursor.advance_to_index(2));
        assert_eq!(cursor.current(), Some(11));
        assert!(cursor.advance_to_geq(40));
        assert_eq!(cursor.current(), Some(42));
        assert_eq!(cursor.index(), 5);
    }

    #[test]
    fn test_cursor_advance_to_index_backward() {
        let vals = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Forward to 5, then backward to 2
        assert!(cursor.advance_to_index(5));
        assert_eq!(cursor.current(), Some(42));
        assert!(cursor.advance_to_index(2));
        assert_eq!(cursor.current(), Some(11));
        assert_eq!(cursor.index(), 2);
    }

    #[test]
    fn test_cursor_advance_to_index_roundtrip() {
        let vals = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Visit every index, verify matches get()
        for i in 0..vals.len() {
            assert!(cursor.advance_to_index(i));
            assert_eq!(cursor.current(), ef.get(i));
            assert_eq!(cursor.index(), i);
        }
    }

    #[test]
    fn test_cursor_advance_to_index_same_position() {
        let vals = vec![10, 20, 30];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        assert!(cursor.advance_to_index(1));
        assert_eq!(cursor.current(), Some(20));
        // Same position — should be no-op
        assert!(cursor.advance_to_index(1));
        assert_eq!(cursor.current(), Some(20));
    }

    #[test]
    fn test_cursor_advance_to_index_empty() {
        let ef = EliasFano::from_sorted(&[]);
        let mut cursor = ef.cursor();
        assert!(!cursor.advance_to_index(0));
    }

    #[test]
    fn test_pef_cursor_advance_to_index() {
        // Need 200+ elements to exercise multiple chunks
        let vals: Vec<u32> = (0..500).map(|i| i * 3 + 1).collect();
        let pef = PartitionedEliasFano::from_sorted(&vals);
        let mut cursor = pef.cursor();

        // Test various indices across chunks
        for &idx in &[0, 1, 127, 128, 129, 255, 256, 300, 499] {
            assert!(cursor.advance_to_index(idx), "advance_to_index({idx}) failed");
            assert_eq!(cursor.current(), Some(vals[idx] as u64), "wrong value at {idx}");
            assert_eq!(cursor.index(), idx);
        }

        // Past end
        assert!(!cursor.advance_to_index(500));

        // Backward jump
        assert!(cursor.advance_to_index(300));
        assert!(cursor.advance_to_index(50));
        assert_eq!(cursor.current(), Some(vals[50] as u64));
    }

    #[test]
    fn test_opef_cursor_advance_to_index() {
        let vals: Vec<u32> = (0..500).map(|i| i * 3 + 1).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&vals);
        let mut cursor = opef.cursor();

        for &idx in &[0, 1, 50, 100, 200, 300, 499] {
            assert!(cursor.advance_to_index(idx), "advance_to_index({idx}) failed");
            assert_eq!(cursor.current(), Some(vals[idx] as u64), "wrong value at {idx}");
            assert_eq!(cursor.index(), idx);
        }

        assert!(!cursor.advance_to_index(500));

        // Backward jump
        assert!(cursor.advance_to_index(400));
        assert!(cursor.advance_to_index(100));
        assert_eq!(cursor.current(), Some(vals[100] as u64));
    }

    #[test]
    fn test_cursor_advance_to_index_large_values() {
        // Test with values near u32::MAX to verify select1 with large universes
        let vals = vec![0, 1000, u32::MAX / 2, u32::MAX - 1000, u32::MAX];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        for i in 0..vals.len() {
            assert!(cursor.advance_to_index(i));
            assert_eq!(cursor.current(), Some(vals[i] as u64));
            assert_eq!(cursor.index(), i);
        }
    }

    #[test]
    fn test_cursor_advance_to_index_large_sequence() {
        // 1500 elements exercises select1 sampling (every 256 elements)
        let vals: Vec<u32> = (0..1500).map(|i| i * 7 + 3).collect();
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Every 10th index
        for i in (0..1500).step_by(10) {
            assert!(cursor.advance_to_index(i));
            assert_eq!(cursor.current(), Some(vals[i] as u64));
            assert_eq!(cursor.index(), i);
        }

        // Backward jumps across the sequence
        assert!(cursor.advance_to_index(1000));
        assert!(cursor.advance_to_index(500));
        assert_eq!(cursor.current(), Some(vals[500] as u64));
        assert!(cursor.advance_to_index(1499));
        assert_eq!(cursor.current(), Some(vals[1499] as u64));
        assert!(cursor.advance_to_index(0));
        assert_eq!(cursor.current(), Some(vals[0] as u64));
    }

    #[test]
    fn test_cursor_advance_to_index_multiple_backward_jumps() {
        let vals: Vec<u32> = (0..200).map(|i| i * 5).collect();
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Zigzag: forward, backward, forward, backward
        assert!(cursor.advance_to_index(100));
        assert_eq!(cursor.current(), Some(500));
        assert!(cursor.advance_to_index(50));
        assert_eq!(cursor.current(), Some(250));
        assert!(cursor.advance_to_index(150));
        assert_eq!(cursor.current(), Some(750));
        assert!(cursor.advance_to_index(25));
        assert_eq!(cursor.current(), Some(125));

        // Verify advance() still works after backward jump
        assert!(cursor.advance());
        assert_eq!(cursor.index(), 26);
        assert_eq!(cursor.current(), Some(130));
    }

    #[test]
    fn test_empty_pef_advance_to_index() {
        let pef = PartitionedEliasFano::from_sorted(&[]);
        let mut cursor = pef.cursor();
        assert!(!cursor.advance_to_index(0));
    }

    #[test]
    fn test_empty_opef_advance_to_index() {
        let opef = OptimalPartitionedEliasFano::from_sorted(&[]);
        let mut cursor = opef.cursor();
        assert!(!cursor.advance_to_index(0));
    }

    #[test]
    fn test_pef_cursor_within_chunk_jumps() {
        let vals: Vec<u32> = (0..500).map(|i| i * 3).collect();
        let pef = PartitionedEliasFano::from_sorted(&vals);
        let mut cursor = pef.cursor();

        // Forward within chunk 0
        assert!(cursor.advance_to_index(10));
        assert_eq!(cursor.current(), Some(30));
        assert!(cursor.advance_to_index(20));
        assert_eq!(cursor.current(), Some(60));

        // Backward within chunk 0
        assert!(cursor.advance_to_index(5));
        assert_eq!(cursor.current(), Some(15));

        // Last element of chunk 0
        assert!(cursor.advance_to_index(127));
        assert_eq!(cursor.current(), Some(vals[127] as u64));
        assert_eq!(cursor.index(), 127);

        // First element of chunk 1, then advance()
        assert!(cursor.advance_to_index(128));
        assert_eq!(cursor.current(), Some(vals[128] as u64));
        assert_eq!(cursor.index(), 128);
        assert!(cursor.advance());
        assert_eq!(cursor.index(), 129);
        assert_eq!(cursor.current(), Some(vals[129] as u64));
    }

    #[test]
    fn test_pef_cursor_chunk_boundary_roundtrip() {
        let vals: Vec<u32> = (0..500).map(|i| i * 3).collect();
        let pef = PartitionedEliasFano::from_sorted(&vals);
        let mut cursor = pef.cursor();

        // Sequential calls crossing chunk boundary
        for i in 125..132 {
            assert!(cursor.advance_to_index(i));
            assert_eq!(cursor.current(), Some(vals[i] as u64));
            assert_eq!(cursor.index(), i);
        }
    }

    #[test]
    fn test_pef_cursor_advance_to_index_large_values() {
        let vals = vec![0, 1000, u32::MAX / 2, u32::MAX - 1000, u32::MAX];
        let pef = PartitionedEliasFano::from_sorted(&vals);
        let mut cursor = pef.cursor();

        for i in 0..vals.len() {
            assert!(cursor.advance_to_index(i));
            assert_eq!(cursor.current(), Some(vals[i] as u64));
        }
    }

    #[test]
    fn test_opef_cursor_advance_to_index_within_chunk() {
        let vals: Vec<u32> = (0..1000).map(|i| i * 5).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&vals);
        let mut cursor = opef.cursor();

        // Forward within a chunk
        assert!(cursor.advance_to_index(10));
        assert_eq!(cursor.current(), Some(50));
        assert!(cursor.advance_to_index(20));
        assert_eq!(cursor.current(), Some(100));

        // Backward within the same chunk
        assert!(cursor.advance_to_index(5));
        assert_eq!(cursor.current(), Some(25));

        // Jump far forward then far backward
        assert!(cursor.advance_to_index(900));
        assert_eq!(cursor.current(), Some(vals[900] as u64));
        assert!(cursor.advance_to_index(50));
        assert_eq!(cursor.current(), Some(vals[50] as u64));
    }

    #[test]
    fn test_opef_cursor_advance_to_index_then_advance() {
        let vals: Vec<u32> = (0..500).map(|i| i * 3 + 1).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&vals);
        let mut cursor = opef.cursor();

        // Jump to an index, then call advance()
        assert!(cursor.advance_to_index(250));
        assert_eq!(cursor.current(), Some(vals[250] as u64));
        assert!(cursor.advance());
        assert_eq!(cursor.index(), 251);
        assert_eq!(cursor.current(), Some(vals[251] as u64));

        // Jump to an index, then call advance_to_geq()
        assert!(cursor.advance_to_index(100));
        assert_eq!(cursor.current(), Some(vals[100] as u64));
        let target = vals[200] as u64;
        assert!(cursor.advance_to_geq(target));
        assert_eq!(cursor.index(), 200);
    }
}
