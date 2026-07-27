use crate::algorithms::bit_ops::select_in_word;


/// Lightweight metadata for a chunk in flat contiguous storage.
///
/// 24 bytes per chunk vs 80+ bytes for the old per-chunk `Vec<u64>` layout.
/// Stores word-offsets into shared flat arrays, eliminating pointer chasing —
/// all chunk data lives in two contiguous `Vec<u64>` arrays owned by the
/// parent `PartitionedEliasFano` or `OptimalPartitionedEliasFano`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct PefChunkMeta {
    /// Minimum value in the chunk (base for delta encoding).
    pub(crate) min_value: u64,
    /// Offset (in u64 words) into the parent's `all_low_bits` array.
    pub(crate) low_offset: u32,
    /// Offset (in u64 words) into the parent's `all_high_bits` array.
    pub(crate) high_offset: u32,
    /// Number of elements in this chunk (1..=512 for OPEF, 1..=128 for PEF).
    pub(crate) count: u16,
    /// Total valid bits in this chunk's high_bits region.
    pub(crate) high_len_bits: u16,
    /// Pre-computed number of u64 words in this chunk's high_bits region.
    pub(crate) high_words: u16,
    /// Pre-computed number of u64 words in this chunk's low_bits region (excluding padding).
    pub(crate) low_words: u16,
    /// Bit width of the low part for this chunk (0..=64).
    pub(crate) low_bit_width: u8,
}

/// Lightweight borrowing view into a chunk's data from flat arrays.
/// Created on the stack for each query — zero heap allocation.
pub(crate) struct ChunkView<'a> {
    pub(crate) low_bits: &'a [u64],
    pub(crate) high_bits: &'a [u64],
    pub(crate) low_bit_width: u32,
    pub(crate) count: usize,
    pub(crate) min_value: u64,
    pub(crate) high_len_bits: usize,
}

// ============================================================================
// Shared chunk helpers (used by both PEF and OPEF via ChunkView)
// ============================================================================

/// Get the delta value (value - min_value) of the local_idx-th element in a chunk.
#[inline]
pub(crate) fn chunk_get_delta(chunk: &ChunkView<'_>, local_idx: usize) -> u64 {
    let high_pos = chunk_select1(chunk, local_idx);
    let high_val = (high_pos - local_idx) as u64;
    let low = chunk_get_low(chunk, local_idx);
    (high_val << chunk.low_bit_width) | low
}

/// Select1 on a chunk's high_bits — find position of the k-th set bit.
/// No sampling needed (≤ 8 words for PEF, ≤ 16 for OPEF), just word-level scan.
#[inline]
pub(crate) fn chunk_select1(chunk: &ChunkView<'_>, rank: usize) -> usize {
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
pub(crate) fn chunk_get_low(chunk: &ChunkView<'_>, index: usize) -> u64 {
    if chunk.low_bit_width == 0 {
        return 0;
    }
    let bit_pos = index as u64 * chunk.low_bit_width as u64;
    let word_idx = (bit_pos / 64) as usize;
    let bit_idx = (bit_pos % 64) as u32;

    // SAFETY: The `low_bits` array is padded during construction, ensuring `word_idx + 1` is safe.
    let (w0, w1) = unsafe {
        (
            *chunk.low_bits.get_unchecked(word_idx),
            *chunk.low_bits.get_unchecked(word_idx + 1),
        )
    };
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
pub(crate) fn chunk_skip_to_high(chunk: &ChunkView<'_>, target_high: usize) -> (usize, usize) {
    if target_high == 0 {
        return (0, 0);
    }

    let mut zeros_remaining = target_high;
    let mut ones_count = 0usize;
    let last_wi = chunk.high_bits.len().saturating_sub(1);

    // Pre-compute the mask for the last word (all other words use u64::MAX)
    let last_valid_bits = {
        let rem = chunk.high_len_bits % 64;
        if rem == 0 && chunk.high_len_bits > 0 {
            64
        } else {
            rem
        }
    };
    let last_word_mask = if last_valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << last_valid_bits) - 1
    };

    for (wi, &w) in chunk.high_bits.iter().enumerate() {
        let valid_bits = if wi == last_wi { last_valid_bits } else { 64 };
        let valid_mask = if wi == last_wi {
            last_word_mask
        } else {
            u64::MAX
        };
        let masked = w & valid_mask;

        let ones = masked.count_ones() as usize;
        let zeros = valid_bits - ones;

        if zeros_remaining <= zeros {
            // The target zero is in this word. Find position of the
            // zeros_remaining-th zero bit (1-indexed → select the
            // (zeros_remaining-1)-th set bit in the inverted word).
            //
            // select_in_word precondition (H12): `zeros_remaining >= 1` holds
            // because target_high == 0 returned early and the subtraction
            // branch below keeps it positive; `zeros_remaining <= zeros ==
            // popcount(inverted)` is this branch's guard. Together:
            // `rank = zeros_remaining - 1 < popcount(inverted)`, which also
            // implies `inverted != 0` (an all-ones word has zeros == 0 and
            // never enters this branch).
            debug_assert!(zeros_remaining >= 1 && zeros_remaining <= zeros);
            let inverted = (!w) & valid_mask;
            let zero_pos_in_word = select_in_word(inverted, zeros_remaining - 1);
            let abs_pos = wi * 64 + zero_pos_in_word;

            // Count ones before abs_pos + 1 (rank1 up to and including abs_pos)
            let bits_up_to = zero_pos_in_word + 1;
            let partial_mask = if bits_up_to >= 64 {
                u64::MAX
            } else {
                (1u64 << bits_up_to) - 1
            };
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

/// Linear scan within a chunk's high_bits starting from a given position.
/// Returns `Some((local_index, delta, bit_pos))` for the first element with
/// `delta >= target_delta`. The returned `bit_pos` is the position of that
/// element's set bit in high_bits, eliminating the need for a subsequent
/// `chunk_select1` call.
#[inline]
pub(crate) fn chunk_scan_geq(
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
    // SAFETY: word_idx is verified < len above
    let mut word = unsafe { *chunk.high_bits.get_unchecked(word_idx) } >> start_bit;
    bit_pos = word_idx * 64 + start_bit;

    while idx < chunk.count {
        if word == 0 {
            word_idx += 1;
            if word_idx >= chunk.high_bits.len() {
                break;
            }
            // SAFETY: word_idx is verified < len above
            word = unsafe { *chunk.high_bits.get_unchecked(word_idx) };
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

#[inline]
pub(crate) fn chunk_first_one_cached(high_bits: &[u64]) -> usize {
    for (wi, &w) in high_bits.iter().enumerate() {
        if w != 0 {
            return wi * 64 + w.trailing_zeros() as usize;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(high_bits: &'a [u64], low_bits: &'a [u64], count: usize, len: usize) -> ChunkView<'a> {
        ChunkView {
            low_bits,
            high_bits,
            low_bit_width: 0,
            count,
            min_value: 0,
            high_len_bits: len,
        }
    }

    /// H12 edge: a word that is all ones within the valid mask has zero
    /// zeros — `chunk_skip_to_high` must skip over it without ever calling
    /// `select_in_word` on its all-zero inverted word.
    #[test]
    fn test_skip_to_high_all_ones_word() {
        // word0 = all ones (64 elements, no zeros); word1 = 0b0101 with
        // high_len_bits = 68 → valid bits {64:1, 65:0, 66:1, 67:0}.
        let high_bits = [u64::MAX, 0b0101];
        let low_bits = [0u64, 0];
        let chunk = view(&high_bits, &low_bits, 66, 68);

        // First zero lives at abs bit 65; 65 ones precede-or-include it.
        assert_eq!(chunk_skip_to_high(&chunk, 1), (65, 66));
        // Second zero at abs bit 67; 66 ones before it.
        assert_eq!(chunk_skip_to_high(&chunk, 2), (66, 68));
        // Past all zeros → clamps to (count, high_len_bits).
        assert_eq!(chunk_skip_to_high(&chunk, 3), (66, 68));
    }

    /// H12 edge: an all-zeros word inverts to an all-ones word — the
    /// `select_in_word(inverted, 63)` call at the word boundary must stay
    /// in range (this is the "all-ones inverted word" case).
    #[test]
    fn test_skip_to_high_all_zeros_word() {
        let high_bits = [0u64, u64::MAX];
        let low_bits = [0u64, 0];
        let chunk = view(&high_bits, &low_bits, 64, 128);

        // 64th zero is bit 63 (rank 63 in the all-ones inverted word).
        assert_eq!(chunk_skip_to_high(&chunk, 64), (0, 64));
        // First zero is bit 0.
        assert_eq!(chunk_skip_to_high(&chunk, 1), (0, 1));
    }
}
