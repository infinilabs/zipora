const BATCH_SIZE: usize = 8;
use crate::error::{Result, ZiporaError};
use crate::algorithms::bit_ops::select_in_word;
use crate::succinct::BitVector;
use std::cmp::Ordering;

/// Sampling rate for rank acceleration.
const RANK_SAMPLE_RATE: usize = 8;
/// Sampling rate for select1 acceleration.
const SELECT1_SAMPLE_RATE: usize = 256;
/// Sampling rate for select0 acceleration.
const SELECT0_SAMPLE_RATE: usize = 64;

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

impl EliasFano {
    /// Create a stateful cursor for O(1) sequential access.
    #[inline]
    pub fn cursor(&self) -> EliasFanoCursor<'_> {
        EliasFanoCursor::new(self)
    }
}

impl EliasFano {
    /// Create a batch cursor for high-throughput sequential iteration.
    #[inline]
    pub fn batch_cursor(&self) -> EliasFanoBatchCursor<'_> {
        EliasFanoBatchCursor::new(self)
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

