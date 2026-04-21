const BATCH_SIZE: usize = 8;
const PEF_CHUNK_SIZE: usize = 128;
use crate::error::{Result, ZiporaError};
use crate::algorithms::bit_ops::select_in_word;
use crate::succinct::BitVector;
use std::cmp::Ordering;

use super::chunk::{PefChunkMeta, ChunkView, chunk_skip_to_high, chunk_scan_geq, chunk_get_delta, chunk_select1, chunk_get_low, chunk_first_one_cached};
use super::basic::EliasFano;


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

impl PartitionedEliasFano {
    /// Create a batch cursor for high-throughput sequential iteration.
    #[inline]
    pub fn batch_cursor(&self) -> PartitionedEliasFanoBatchCursor<'_> {
        PartitionedEliasFanoBatchCursor::new(self)
    }
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

