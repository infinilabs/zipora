const BATCH_SIZE: usize = 8;
const CHUNK_OVERHEAD_BITS: usize = 192;
const MAX_CHUNK_SIZE: usize = 512;
const MIN_CHUNK_SIZE: usize = 1;
use crate::error::{Result, ZiporaError};
use crate::algorithms::bit_ops::select_in_word;
use crate::succinct::BitVector;
use std::cmp::Ordering;

use super::chunk::{PefChunkMeta, ChunkView, chunk_skip_to_high, chunk_scan_geq, chunk_get_delta, chunk_select1, chunk_get_low, chunk_first_one_cached};
use super::basic::EliasFano;
use super::partitioned::PartitionedEliasFano;


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
            let max_j_start = i.saturating_sub(MAX_CHUNK_SIZE);
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
            let low_words = total_low_bits.div_ceil(64) as usize;
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
            let high_words = high_len_bits.div_ceil(64);
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

