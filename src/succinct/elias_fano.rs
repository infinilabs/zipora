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

/// Sampling rate for select acceleration.
/// One position sample per SELECT_SAMPLE_RATE set/clear bits.
/// Memory: 4 bytes per 256 elements ≈ negligible.
const SELECT_SAMPLE_RATE: usize = 256;

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
    /// Position of every SELECT_SAMPLE_RATE-th set bit.
    /// select1_samples[i] = bit-position of the (i*SELECT_SAMPLE_RATE)-th 1-bit
    select1_samples: Vec<u32>,
    /// Position of every SELECT_SAMPLE_RATE-th clear bit.
    /// select0_samples[i] = bit-position of the (i*SELECT_SAMPLE_RATE)-th 0-bit
    select0_samples: Vec<u32>,
}

impl EliasFano {
    /// Build from a sorted slice of values. Values must be in ascending order.
    pub fn from_sorted(values: &[u32]) -> Self {
        Self::from_sorted_u64(&values.iter().map(|&v| v as u64).collect::<Vec<_>>())
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

        let n = values.len();
        let universe = values[n - 1] + 1;

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

        for (i, &val) in values.iter().enumerate() {
            if low_bit_width > 0 {
                let low_val = val & low_mask;
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
        let max_high = values[n - 1] >> low_bit_width;
        let high_len_bits = n + max_high as usize + 1;
        let high_words = (high_len_bits + 63) / 64;
        let mut high_bits = vec![0u64; high_words];

        let mut pos = 0usize;
        let mut prev_high = 0u64;

        for &val in values {
            let high = val >> low_bit_width;
            pos += (high - prev_high) as usize;
            let word_idx = pos / 64;
            let bit_idx = pos % 64;
            if word_idx < high_words {
                high_bits[word_idx] |= 1u64 << bit_idx;
            }
            pos += 1;
            prev_high = high;
        }

        // Build sampling indices for O(1) rank/select
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
                next_sel1 += SELECT_SAMPLE_RATE as u32;
            }

            // Select0: same for zero bits
            let after_zeros = cumul_zeros + zeros;
            while next_sel0 < after_zeros {
                select0_samples.push((word_idx * 64) as u32);
                next_sel0 += SELECT_SAMPLE_RATE as u32;
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
    /// O(1) amortized — the core operation for posting list intersection.
    pub fn next_geq(&self, target: u64) -> Option<(usize, u64)> {
        if self.len == 0 || target >= self.universe { return None; }

        let target_high = target >> self.low_bit_width;
        let target_low = if self.low_bit_width > 0 {
            target & ((1u64 << self.low_bit_width) - 1)
        } else {
            0
        };

        // Find the first element in bucket >= target_high using select0
        // Bucket b starts at position select0(b) in high_bits
        let bucket_start_pos = if target_high == 0 {
            0
        } else {
            match self.select0(target_high as usize - 1) {
                Some(p) => p + 1,
                None => return None,
            }
        };

        // Count elements before this bucket (= number of 1-bits before bucket_start_pos)
        let elem_before = self.rank1(bucket_start_pos);

        // Scan forward from elem_before to find first element >= target
        for idx in elem_before..self.len {
            if let Some(val) = self.get(idx) {
                if val >= target {
                    return Some((idx, val));
                }
            }
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
        let sample_idx = rank / SELECT_SAMPLE_RATE;
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
        let sample_idx = rank / SELECT_SAMPLE_RATE;
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
pub struct EliasFanoIter<'a> {
    ef: &'a EliasFano,
    pos: usize,
    high_pos: usize,
}

impl<'a> Iterator for EliasFanoIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.ef.len { return None; }

        // Find next 1-bit in high_bits starting from high_pos
        while self.high_pos < self.ef.high_len_bits {
            let word_idx = self.high_pos / 64;
            let bit_idx = self.high_pos % 64;

            if word_idx >= self.ef.high_bits.len() { return None; }

            if (self.ef.high_bits[word_idx] >> bit_idx) & 1 == 1 {
                // Found a 1-bit — this is an element
                let high_val = (self.high_pos - self.pos) as u64;
                let low = self.ef.get_low(self.pos);
                let val = (high_val << self.ef.low_bit_width) | low;
                self.pos += 1;
                self.high_pos += 1;
                return Some(val);
            }
            self.high_pos += 1; // Skip 0-bit (bucket boundary)
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ef.len - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for EliasFanoIter<'a> {}

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
}
