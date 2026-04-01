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

/// Select the `rank`-th set bit in a 64-bit word (0-indexed).
/// Returns the bit position (0..63).
///
/// Uses BMI2 `_pdep_u64` on x86_64 for O(1) hardware select.
/// Falls back to broadword bit-clearing loop on other architectures.
#[inline(always)]
fn select_in_word(word: u64, rank: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            // SAFETY: _pdep_u64 is safe when bmi2 is available.
            // pdep(1 << rank, word) places a 1 at the position of the rank-th
            // set bit in `word`. trailing_zeros gives the position.
            return unsafe {
                let mask = std::arch::x86_64::_pdep_u64(1u64 << rank, word);
                mask.trailing_zeros() as usize
            };
        }
    }
    // Scalar fallback: clear the lowest `rank` set bits, then find the next one.
    let mut w = word;
    for _ in 0..rank {
        w &= w - 1; // Clear lowest set bit
    }
    w.trailing_zeros() as usize
}

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

        // Pack low bits
        let total_low_bits = n as u64 * low_bit_width as u64;
        let low_words = ((total_low_bits + 63) / 64) as usize;
        let mut low_bits = vec![0u64; low_words];

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
    #[inline]
    fn get_low(&self, index: usize) -> u64 {
        if self.low_bit_width == 0 { return 0; }

        let bit_pos = index as u64 * self.low_bit_width as u64;
        let word_idx = (bit_pos / 64) as usize;
        let bit_idx = (bit_pos % 64) as u32;

        let mut val = self.low_bits[word_idx] >> bit_idx;

        if bit_idx + self.low_bit_width > 64 && word_idx + 1 < self.low_bits.len() {
            val |= self.low_bits[word_idx + 1] << (64 - bit_idx);
        }

        val & ((1u64 << self.low_bit_width) - 1)
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
}
