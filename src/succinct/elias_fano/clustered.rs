//! Clustered Elias-Fano (code_review.md §5.2, Phase 4).
//!
//! Like [`PartitionedEliasFano`](super::partitioned::PartitionedEliasFano), the
//! sorted sequence is split into fixed 128-element chunks. The difference: each
//! chunk independently chooses the cheapest of **three container types** based on
//! its local density (the Roaring-bitmap insight applied per chunk):
//!
//! - **Run** — the chunk is fully contiguous (`max - min + 1 == count`). Stored
//!   as nothing but `(min, count)`: ~0 bits/element, O(1) `next_geq`/`get`.
//! - **Bitmap** — dense but not contiguous, where a raw bitmap (`local_universe`
//!   bits) is smaller than local Elias-Fano. SIMD-friendly for intersection.
//! - **EliasFano** — sparse fallback, identical encoding to PEF (reuses the
//!   shared [`chunk`](super::chunk) helpers).
//!
//! **Why a separate type** (not an edit to PEF/OPEF): zero regression risk to the
//! existing encoders and a clean opt-in path via
//! [`HybridPostingList`](super::hybrid). Phase 0 benchmarks showed run/bitmap
//! containers beat OPEF on bursty-clustered and fully-dense data, both for space
//! and (with [`intersect_count`](ClusteredEliasFano::intersect_count)) intersection.

use super::chunk::{ChunkView, chunk_get_delta, chunk_scan_geq, chunk_skip_to_high};

const CEF_CHUNK_SIZE: usize = 128;

/// Per-chunk container type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChunkKind {
    /// Fully contiguous run — stores only `(min, count)`.
    Run,
    /// Dense bitmap of `local_universe` bits.
    Bitmap,
    /// Sparse Elias-Fano (delta-from-min), same layout as PEF.
    EliasFano,
}

/// Metadata for one clustered chunk. Fields are interpreted per `kind`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct CefChunkMeta {
    kind: ChunkKind,
    /// Minimum value (base) — valid for all kinds.
    min_value: u64,
    /// Number of elements in the chunk.
    count: u16,
    // --- EliasFano fields (offsets in u64 words into the flat arrays) ---
    low_offset: u32,
    high_offset: u32,
    high_len_bits: u16,
    low_bit_width: u8,
    // --- Bitmap fields ---
    /// Offset (in u64 words) into `all_dense_bits`.
    bitmap_offset: u32,
    /// Number of u64 words occupied by this chunk's bitmap.
    bitmap_words: u16,
}

/// Clustered Elias-Fano posting list.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClusteredEliasFano {
    all_low_bits: Vec<u64>,
    all_high_bits: Vec<u64>,
    /// Raw bitmap words for `Bitmap` chunks (concatenated).
    all_dense_bits: Vec<u64>,
    meta: Vec<CefChunkMeta>,
    /// Last value of each chunk, for binary-search chunk selection.
    chunk_upper_bounds: Vec<u64>,
    len: usize,
    /// Exclusive upper bound (max value + 1). u128 so a value of u64::MAX fits.
    universe: u128,
}

impl ClusteredEliasFano {
    /// Build from a sorted slice of u32 values.
    pub fn from_sorted(values: &[u32]) -> Self {
        if values.is_empty() {
            return Self::empty();
        }
        let universe = values[values.len() - 1] as u128 + 1;
        Self::from_sorted_impl(values.len(), universe, |i| values[i] as u64)
    }

    /// Build from sorted u64 values.
    pub fn from_sorted_u64(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self::empty();
        }
        // u128 exclusive upper bound so a value of u64::MAX is representable
        // (u64::MAX + 1) and next_geq(u64::MAX) still finds it.
        let universe = values[values.len() - 1] as u128 + 1;
        Self::from_sorted_impl(values.len(), universe, |i| values[i])
    }

    fn empty() -> Self {
        Self {
            all_low_bits: Vec::new(),
            all_high_bits: Vec::new(),
            all_dense_bits: Vec::new(),
            meta: Vec::new(),
            chunk_upper_bounds: Vec::new(),
            len: 0,
            universe: 0,
        }
    }

    fn from_sorted_impl(n: usize, universe: u128, get_val: impl Fn(usize) -> u64) -> Self {
        let num_chunks = n.div_ceil(CEF_CHUNK_SIZE);
        let mut all_low_bits = Vec::new();
        let mut all_high_bits = Vec::new();
        let mut all_dense_bits = Vec::new();
        let mut meta = Vec::with_capacity(num_chunks);
        let mut chunk_upper_bounds = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * CEF_CHUNK_SIZE;
            let end = (start + CEF_CHUNK_SIZE).min(n);
            let count = end - start;
            let min_val = get_val(start);
            let max_val = get_val(end - 1);
            // saturating: guards the pathological case where the chunk spans up to
            // u64::MAX (e.g. from_sorted_u64([0, u64::MAX])). Without it this wraps
            // to 0 and later selects a 0-word Bitmap that indexes out of bounds.
            let local_universe = (max_val - min_val).saturating_add(1);
            chunk_upper_bounds.push(max_val);

            // --- Container selection ---------------------------------------
            // Run: fully contiguous (cheapest, O(1) queries).
            if local_universe == count as u64 {
                meta.push(CefChunkMeta {
                    kind: ChunkKind::Run,
                    min_value: min_val,
                    count: count as u16,
                    low_offset: 0,
                    high_offset: 0,
                    high_len_bits: 0,
                    low_bit_width: 0,
                    bitmap_offset: 0,
                    bitmap_words: 0,
                });
                continue;
            }

            // EF cost model (matches PEF): low_bit_width adaptive to density.
            let low_bit_width = if count as u64 >= local_universe {
                0
            } else {
                (64 - (local_universe / count as u64).leading_zeros()).saturating_sub(1)
            };
            let last_delta = max_val - min_val;
            let max_high = last_delta >> low_bit_width;
            let ef_high_len_bits = count + max_high as usize + 1;
            let ef_total_bits = count * low_bit_width as usize + ef_high_len_bits;

            // Bitmap cost: local_universe bits (rounded to words). saturating_mul
            // so a saturated local_universe can never make bitmap look cheaper
            // than EF (an astronomically wide chunk stays EliasFano).
            let bitmap_words = local_universe.div_ceil(64) as usize;
            let bitmap_total_bits = bitmap_words.saturating_mul(64);

            if bitmap_total_bits < ef_total_bits {
                // --- Bitmap container ---
                let bitmap_offset = all_dense_bits.len();
                all_dense_bits.resize(bitmap_offset + bitmap_words, 0);
                for i in 0..count {
                    let delta = (get_val(start + i) - min_val) as usize;
                    all_dense_bits[bitmap_offset + delta / 64] |= 1u64 << (delta % 64);
                }
                meta.push(CefChunkMeta {
                    kind: ChunkKind::Bitmap,
                    min_value: min_val,
                    count: count as u16,
                    low_offset: 0,
                    high_offset: 0,
                    high_len_bits: 0,
                    low_bit_width: 0,
                    bitmap_offset: bitmap_offset as u32,
                    bitmap_words: bitmap_words as u16,
                });
                continue;
            }

            // --- EliasFano container (same layout as PEF) ---
            let low_mask = if low_bit_width == 0 {
                0u64
            } else {
                (1u64 << low_bit_width) - 1
            };
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
            let high_words = ef_high_len_bits.div_ceil(64);
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
            meta.push(CefChunkMeta {
                kind: ChunkKind::EliasFano,
                min_value: min_val,
                count: count as u16,
                low_offset: low_offset as u32,
                high_offset: high_offset as u32,
                high_len_bits: ef_high_len_bits as u16,
                low_bit_width: low_bit_width as u8,
                bitmap_offset: 0,
                bitmap_words: 0,
            });
        }

        // Padding word for branchless u128 low-bit extraction (chunk_get_low).
        all_low_bits.push(0);

        Self {
            all_low_bits,
            all_high_bits,
            all_dense_bits,
            meta,
            chunk_upper_bounds,
            len: n,
            universe,
        }
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Estimated memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.all_low_bits.len() * 8
            + self.all_high_bits.len() * 8
            + self.all_dense_bits.len() * 8
            + self.meta.len() * std::mem::size_of::<CefChunkMeta>()
            + self.chunk_upper_bounds.len() * 8
            + std::mem::size_of::<Self>()
    }

    /// Bits per element.
    #[inline]
    pub fn bits_per_element(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        (self.size_bytes() * 8) as f64 / self.len as f64
    }

    /// Build an EF `ChunkView` for an EliasFano-kind chunk.
    #[inline]
    fn ef_view(&self, m: &CefChunkMeta) -> ChunkView<'_> {
        let low_start = m.low_offset as usize;
        let low_words = (m.count as u64 * m.low_bit_width as u64).div_ceil(64) as usize;
        let high_start = m.high_offset as usize;
        let high_words = (m.high_len_bits as usize).div_ceil(64);
        ChunkView {
            // +1 padding word is always available (pushed at end of build).
            low_bits: &self.all_low_bits[low_start..low_start + low_words + 1],
            high_bits: &self.all_high_bits[high_start..high_start + high_words],
            low_bit_width: m.low_bit_width as u32,
            count: m.count as usize,
            min_value: m.min_value,
            high_len_bits: m.high_len_bits as usize,
        }
    }

    /// Bitmap words for a Bitmap-kind chunk.
    #[inline]
    fn bitmap_slice(&self, m: &CefChunkMeta) -> &[u64] {
        let s = m.bitmap_offset as usize;
        &self.all_dense_bits[s..s + m.bitmap_words as usize]
    }

    /// Get the i-th element (global index).
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }
        let chunk_idx = index / CEF_CHUNK_SIZE;
        let local_idx = index % CEF_CHUNK_SIZE;
        let m = &self.meta[chunk_idx];
        match m.kind {
            ChunkKind::Run => Some(m.min_value + local_idx as u64),
            ChunkKind::Bitmap => {
                let pos = bitmap_select1(self.bitmap_slice(m), local_idx);
                Some(m.min_value + pos as u64)
            }
            ChunkKind::EliasFano => {
                let view = self.ef_view(m);
                Some(m.min_value + chunk_get_delta(&view, local_idx))
            }
        }
    }

    /// Find the first element >= target. Returns (global_index, value).
    #[inline]
    pub fn next_geq(&self, target: u64) -> Option<(usize, u64)> {
        if self.len == 0 || target as u128 >= self.universe {
            return None;
        }
        let chunk_idx = match self.chunk_upper_bounds.binary_search(&target) {
            Ok(i) => i,
            Err(i) => {
                if i >= self.meta.len() {
                    return None;
                }
                i
            }
        };
        let global_offset = chunk_idx * CEF_CHUNK_SIZE;
        let m = &self.meta[chunk_idx];

        // target <= chunk min → first element of this chunk is the answer, and
        // the first element of every chunk kind is exactly `min_value` (Run: min+0;
        // Bitmap: delta 0 is always set; EF: first delta is 0).
        if target <= m.min_value {
            return Some((global_offset, m.min_value));
        }

        match m.kind {
            ChunkKind::Run => {
                // Values are min, min+1, ..., min+count-1. target is in-range
                // (target <= upper_bound from binary search, target > min).
                let local = (target - m.min_value) as usize;
                Some((global_offset + local, target))
            }
            ChunkKind::Bitmap => {
                let bits = self.bitmap_slice(m);
                let target_delta = target - m.min_value;
                bitmap_next_geq(bits, target_delta)
                    .map(|(local, delta)| (global_offset + local, m.min_value + delta))
            }
            ChunkKind::EliasFano => {
                let view = self.ef_view(m);
                let target_delta = target - m.min_value;
                let target_high = (target_delta >> view.low_bit_width) as usize;
                let (start_idx, start_pos) = chunk_skip_to_high(&view, target_high);
                chunk_scan_geq(&view, target_delta, start_idx, start_pos)
                    .map(|(local, delta, _)| (global_offset + local, m.min_value + delta))
            }
        }
        // Found-in-chunk above always succeeds because the binary search
        // guarantees target <= this chunk's upper bound.
    }

    /// Per-chunk container kinds (diagnostics / tests).
    pub fn chunk_kinds(&self) -> Vec<ChunkKind> {
        self.meta.iter().map(|m| m.kind).collect()
    }

    /// Collect a chunk's values into `buf` (cleared first). At most 128 values.
    #[inline]
    fn collect_chunk(&self, idx: usize, buf: &mut Vec<u64>) {
        buf.clear();
        let m = &self.meta[idx];
        match m.kind {
            ChunkKind::Run => {
                for i in 0..m.count as u64 {
                    buf.push(m.min_value + i);
                }
            }
            ChunkKind::Bitmap => {
                let bits = self.bitmap_slice(m);
                for (wi, &w) in bits.iter().enumerate() {
                    let mut x = w;
                    while x != 0 {
                        let b = x.trailing_zeros() as usize;
                        buf.push(m.min_value + (wi * 64 + b) as u64);
                        x &= x - 1;
                    }
                }
            }
            ChunkKind::EliasFano => {
                let view = self.ef_view(m);
                for i in 0..m.count as usize {
                    buf.push(m.min_value + chunk_get_delta(&view, i));
                }
            }
        }
    }

    /// Count elements common to both lists, using block-level fast paths.
    ///
    /// The key win over leapfrog `next_geq`: two overlapping **Run** chunks
    /// intersect in O(1) — the overlap of two contiguous intervals is a single
    /// subtraction, counting up to 128² value-pairs without touching memory.
    /// Mixed/sparse chunk pairs fall back to a bounded (≤128-element) two-pointer
    /// merge. Correctness is identical to leapfrog; only the cost model changes.
    pub fn intersect_count(&self, other: &Self) -> usize {
        if self.is_empty() || other.is_empty() {
            return 0;
        }
        let mut count = 0usize;
        let mut ia = 0usize;
        let mut ib = 0usize;
        let mut buf_a: Vec<u64> = Vec::with_capacity(CEF_CHUNK_SIZE);
        let mut buf_b: Vec<u64> = Vec::with_capacity(CEF_CHUNK_SIZE);

        while ia < self.meta.len() && ib < other.meta.len() {
            let ma = self.meta[ia].min_value;
            let mb = other.meta[ib].min_value;
            let ua = self.chunk_upper_bounds[ia];
            let ub = other.chunk_upper_bounds[ib];

            if ua < mb {
                ia += 1;
                continue;
            }
            if ub < ma {
                ib += 1;
                continue;
            }

            // Ranges overlap — count matches in this chunk pair.
            let ka = self.meta[ia].kind;
            let kb = other.meta[ib].kind;
            if ka == ChunkKind::Run && kb == ChunkKind::Run {
                // O(1): overlap of [ma, ua] and [mb, ub].
                let lo = ma.max(mb);
                let hi = ua.min(ub);
                count += (hi - lo + 1) as usize; // hi >= lo guaranteed by overlap
            } else {
                self.collect_chunk(ia, &mut buf_a);
                other.collect_chunk(ib, &mut buf_b);
                count += merge_count(&buf_a, &buf_b);
            }

            // Advance the chunk whose range ends first.
            if ua <= ub {
                ia += 1;
            } else {
                ib += 1;
            }
        }
        count
    }
}

/// Two-pointer intersection count of two sorted slices.
#[inline]
fn merge_count(a: &[u64], b: &[u64]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut c = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                c += 1;
                i += 1;
                j += 1;
            }
        }
    }
    c
}

impl super::PostingList for ClusteredEliasFano {
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> Option<u64> {
        self.get(index)
    }
    fn next_geq(&self, target: u64) -> Option<(usize, u64)> {
        self.next_geq(target)
    }
    fn size_bytes(&self) -> usize {
        self.size_bytes()
    }
}

// ---------------------------------------------------------------------------
// Bitmap chunk helpers (scalar by design; §5.2 SIMD batching deliberately skipped
// — the intersection win is algorithmic, not vectorization).
// ---------------------------------------------------------------------------

/// Position of the `rank`-th set bit (0-indexed) in a chunk bitmap.
#[inline]
fn bitmap_select1(bits: &[u64], rank: usize) -> usize {
    let mut remaining = rank;
    for (wi, &w) in bits.iter().enumerate() {
        let ones = w.count_ones() as usize;
        if remaining < ones {
            return wi * 64
                + crate::algorithms::bit_ops::select_in_word(w, remaining);
        }
        remaining -= ones;
    }
    // rank out of range — caller guarantees in-range; return end.
    bits.len() * 64
}

/// First set bit at position >= `target_delta`. Returns `(local_index, delta)`
/// where `local_index` is the rank of that bit within the chunk.
#[inline]
fn bitmap_next_geq(bits: &[u64], target_delta: u64) -> Option<(usize, u64)> {
    let start_word = (target_delta / 64) as usize;
    if start_word >= bits.len() {
        return None;
    }
    // Rank of bits strictly before start_word.
    let mut rank: usize = bits[..start_word].iter().map(|w| w.count_ones() as usize).sum();

    let start_bit = (target_delta % 64) as u32;
    let mut w = bits[start_word] & (!0u64 << start_bit);
    let mut wi = start_word;
    loop {
        if w != 0 {
            let tz = w.trailing_zeros() as usize;
            let pos = wi * 64 + tz;
            // rank already counts whole words < wi; add ones below this bit in wi.
            let below = bits[wi] & ((1u64 << tz) - 1);
            return Some((rank + below.count_ones() as usize, pos as u64));
        }
        rank += bits[wi].count_ones() as usize;
        wi += 1;
        if wi >= bits.len() {
            return None;
        }
        w = bits[wi];
    }
}

#[cfg(test)]
#[path = "clustered_tests.rs"]
mod clustered_tests;
