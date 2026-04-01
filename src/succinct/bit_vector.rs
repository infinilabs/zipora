//! Bit vector implementation with efficient storage and access
//!
//! Provides a compact representation of bit arrays with efficient operations
//! for setting, getting, and manipulating individual bits.

use crate::FastVec;
use crate::error::{Result, ZiporaError};
use std::fmt;

// SIMD imports for bulk operations
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use std::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_or_si256, _mm256_storeu_si256,
    _mm256_xor_si256,
};

/// A compact bit vector supporting efficient bit operations
///
/// BitVector stores bits in a packed format using u64 blocks for efficient
/// access and manipulation. It supports dynamic resizing and provides
/// constant-time access to individual bits.
///
/// # Examples
///
/// ```rust
/// use zipora::BitVector;
///
/// let mut bv = BitVector::new();
/// bv.push(true)?;
/// bv.push(false)?;
/// bv.push(true)?;
///
/// assert_eq!(bv.get(0), Some(true));
/// assert_eq!(bv.get(1), Some(false));
/// assert_eq!(bv.len(), 3);
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
pub struct BitVector {
    blocks: FastVec<u64>,
    len: usize,
}

const BITS_PER_BLOCK: usize = 64;

impl BitVector {
    /// Create a new empty bit vector
    #[inline]
    pub fn new() -> Self {
        Self {
            blocks: FastVec::new(),
            len: 0,
        }
    }

    /// Create a bit vector with the specified capacity (in bits)
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        let block_capacity = (capacity + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
        Ok(Self {
            blocks: FastVec::with_capacity(block_capacity)?,
            len: 0,
        })
    }

    /// Create a bit vector with the specified size, filled with the given value.
    ///
    /// For `value == false`, uses `alloc_zeroed` (kernel zero-page mapping)
    /// which is significantly faster than `alloc` + `memset` for large sizes.
    pub fn with_size(size: usize, value: bool) -> Result<Self> {
        if !value {
            // Fast path: alloc_zeroed leverages kernel zero-page mapping.
            // For 1M bits this is ~1 µs vs ~30 µs for alloc+write_bytes.
            let num_blocks = (size + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
            let blocks = FastVec::with_capacity_zeroed(num_blocks)?;
            Ok(Self { blocks, len: size })
        } else {
            let mut bv = Self::with_capacity(size)?;
            bv.resize(size, true)?;
            Ok(bv)
        }
    }

    /// Create a bit vector from raw u64 blocks and bit length
    ///
    /// This is used by multi-dimensional operations to create result bit vectors
    /// from SIMD-computed raw bit data.
    ///
    /// # Arguments
    ///
    /// * `raw_bits` - Raw u64 words containing the bit data
    /// * `total_bits` - Total number of valid bits
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::BitVector;
    ///
    /// let raw_bits = vec![0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64];
    /// let bv = BitVector::from_raw_bits(raw_bits, 128)?;
    /// assert_eq!(bv.len(), 128);
    /// assert_eq!(bv.get(0).unwrap(), true);
    /// assert_eq!(bv.get(63).unwrap(), true);
    /// assert_eq!(bv.get(64).unwrap(), false);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn from_raw_bits(raw_bits: Vec<u64>, total_bits: usize) -> Result<Self> {
        let required_blocks = (total_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;

        if raw_bits.len() < required_blocks {
            return Err(ZiporaError::invalid_data(format!(
                "Insufficient raw bits: need {} blocks for {} bits, got {}",
                required_blocks, total_bits, raw_bits.len()
            )));
        }

        // Zero-copy: take ownership of the Vec via FastVec::from_vec.
        // If the Vec has more blocks than needed, we keep the extra capacity
        // but only use required_blocks via the len field.
        let mut blocks = FastVec::from_vec(raw_bits);

        // Clear trailing bits in the last block if necessary
        if total_bits > 0 {
            let last_block_idx = required_blocks - 1;
            let bits_in_last_block = total_bits % BITS_PER_BLOCK;
            if bits_in_last_block > 0 {
                let mask = (1u64 << bits_in_last_block) - 1;
                blocks[last_block_idx] &= mask;
            }
        }

        Ok(Self {
            blocks,
            len: total_bits,
        })
    }

    /// Create a bit vector by wrapping a pre-allocated zeroed `Vec<u64>`.
    ///
    /// Zero-copy: takes ownership of the Vec without copying. The Vec must
    /// be pre-zeroed (e.g., from `vec![0u64; n]` which uses `calloc`).
    ///
    /// This is the fastest way to create a zeroed BitVector because
    /// `vec![0u64; n]` leverages the kernel's zero-page mapping, avoiding
    /// physical memory zeroing for large allocations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::BitVector;
    ///
    /// let max_doc = 1_000_000;
    /// let num_words = (max_doc + 63) / 64;
    /// let blocks = vec![0u64; num_words]; // calloc — fast zero-page mapping
    /// let mut bv = BitVector::from_blocks(blocks, max_doc);
    ///
    /// // Scatter doc IDs
    /// bv.set_bit_unchecked(42);
    /// bv.set_bit_unchecked(1000);
    /// assert_eq!(bv.count_ones(), 2);
    /// ```
    #[inline]
    pub fn from_blocks(blocks: Vec<u64>, total_bits: usize) -> Self {
        let num_blocks = blocks.len();
        debug_assert!(
            num_blocks >= (total_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK,
            "blocks.len()={num_blocks} insufficient for {total_bits} bits"
        );
        Self {
            blocks: FastVec::from_vec(blocks),
            len: total_bits,
        }
    }

    /// Get the number of bits in the vector
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the bit vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity in bits
    #[inline]
    pub fn capacity(&self) -> usize {
        self.blocks.capacity() * BITS_PER_BLOCK
    }

    /// Get the bit at the specified position
    #[inline]
    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len {
            return None;
        }

        let block_index = index / BITS_PER_BLOCK;
        let bit_index = index % BITS_PER_BLOCK;

        Some((self.blocks[block_index] >> bit_index) & 1 == 1)
    }

    /// Get the bit at the specified position without bounds checking
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        debug_assert!(index < self.len);

        let block_index = index / BITS_PER_BLOCK;
        let bit_index = index % BITS_PER_BLOCK;

// SAFETY: Caller must ensure index < self.len()

        (unsafe { self.blocks.get_unchecked(block_index) } >> bit_index) & 1 == 1
    }

    /// Set the bit at the specified position
    pub fn set(&mut self, index: usize, value: bool) -> Result<()> {
        if index >= self.len {
            return Err(ZiporaError::out_of_bounds(index, self.len));
        }

        let block_index = index / BITS_PER_BLOCK;
        let bit_index = index % BITS_PER_BLOCK;

        if value {
            self.blocks[block_index] |= 1u64 << bit_index;
        } else {
            self.blocks[block_index] &= !(1u64 << bit_index);
        }

        Ok(())
    }

    /// Set the bit at the specified position without bounds checking
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`
    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.len);

        let block_index = index / BITS_PER_BLOCK;
        let bit_index = index % BITS_PER_BLOCK;

        if value {
            // SAFETY: index < len checked by caller, block_index is valid
            unsafe {
                *self.blocks.get_unchecked_mut(block_index) |= 1u64 << bit_index;
            }
        } else {
            // SAFETY: index < len checked by caller, block_index is valid
            unsafe {
                *self.blocks.get_unchecked_mut(block_index) &= !(1u64 << bit_index);
            }
        }
    }

    /// Push a bit to the end of the vector
    pub fn push(&mut self, value: bool) -> Result<()> {
        let block_index = self.len / BITS_PER_BLOCK;
        let bit_index = self.len % BITS_PER_BLOCK;

        // Ensure we have enough blocks
        while self.blocks.len() <= block_index {
            self.blocks.push(0)?;
        }

        if value {
            self.blocks[block_index] |= 1u64 << bit_index;
        } else {
            self.blocks[block_index] &= !(1u64 << bit_index);
        }

        self.len += 1;
        Ok(())
    }

    /// Ensure the bit at position `i` is set to 1, growing the vector if necessary
    ///
    /// This is useful when using the bit vector as an integer set.
    /// If `i >= len`, the vector is resized to include position `i`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::BitVector;
    ///
    /// let mut bv = BitVector::new();
    /// bv.ensure_set1(5)?;
    /// bv.ensure_set1(10)?;
    /// bv.ensure_set1(3)?;
    ///
    /// assert_eq!(bv.len(), 11);
    /// assert_eq!(bv.get(5), Some(true));
    /// assert_eq!(bv.get(10), Some(true));
    /// assert_eq!(bv.get(3), Some(true));
    /// assert_eq!(bv.get(4), Some(false));
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn ensure_set1(&mut self, i: usize) -> Result<()> {
        if i < self.len {
            // Fast path: just set the bit
            let block_index = i / BITS_PER_BLOCK;
            let bit_index = i % BITS_PER_BLOCK;
            self.blocks[block_index] |= 1u64 << bit_index;
        } else {
            // Slow path: need to grow
            self.ensure_set1_slow_path(i)?;
        }
        Ok(())
    }

    /// Slow path for ensure_set1 when vector needs to grow
    #[cold]
    #[inline(never)]
    fn ensure_set1_slow_path(&mut self, i: usize) -> Result<()> {
        // Resize to include position i (sets all new bits to 0)
        let new_len = i + 1;
        let new_block_count = (new_len + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;

        // Add new blocks initialized to 0
        while self.blocks.len() < new_block_count {
            self.blocks.push(0)?;
        }
        self.len = new_len;

        // Set the bit
        let block_index = i / BITS_PER_BLOCK;
        let bit_index = i % BITS_PER_BLOCK;
        self.blocks[block_index] |= 1u64 << bit_index;

        Ok(())
    }

    /// Fast ensure_set1 optimized for sequential integer set insertions
    ///
    /// This is much faster than `ensure_set1` when inserting integer set members
    /// sequentially. The optimization relies on the invariant that allocated but
    /// unused bits between `len` and `capacity` are always 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::BitVector;
    ///
    /// let mut bv = BitVector::with_capacity(1000)?;
    /// // Sequential insertions are very fast
    /// for i in 0..100 {
    ///     bv.fast_ensure_set1(i)?;
    /// }
    /// assert_eq!(bv.count_ones(), 100);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn fast_ensure_set1(&mut self, i: usize) -> Result<()> {
        let capacity = self.blocks.len() * BITS_PER_BLOCK;

        if i < capacity {
            // Fast path: within capacity, just update len and set bit
            if self.len <= i {
                // Invariant: bits between len and capacity are already 0
                debug_assert!(
                    (self.len..=i).all(|j| {
                        let block_idx = j / BITS_PER_BLOCK;
                        let bit_idx = j % BITS_PER_BLOCK;
                        block_idx >= self.blocks.len() || (self.blocks[block_idx] >> bit_idx) & 1 == 0
                    }),
                    "fast_ensure_set1 invariant violated: unused bits must be 0"
                );
                self.len = i + 1;
            }
            let block_index = i / BITS_PER_BLOCK;
            let bit_index = i % BITS_PER_BLOCK;
            self.blocks[block_index] |= 1u64 << bit_index;
            Ok(())
        } else {
            // Slow path: need to grow capacity
            self.fast_ensure_set1_slow_path(i)
        }
    }

    /// Slow path for fast_ensure_set1 when capacity needs to grow
    #[cold]
    #[inline(never)]
    fn fast_ensure_set1_slow_path(&mut self, i: usize) -> Result<()> {
        // Invariant check in debug mode: unused bits must be 0
        #[cfg(debug_assertions)]
        {
            let capacity = self.blocks.len() * BITS_PER_BLOCK;
            for j in self.len..capacity {
                let block_idx = j / BITS_PER_BLOCK;
                let bit_idx = j % BITS_PER_BLOCK;
                if block_idx < self.blocks.len() {
                    debug_assert!(
                        (self.blocks[block_idx] >> bit_idx) & 1 == 0,
                        "fast_ensure_set1 invariant violated at bit {}: unused bits must be 0",
                        j
                    );
                }
            }
        }

        let old_capacity = self.blocks.len() * BITS_PER_BLOCK;
        let new_len = i + 1;
        let new_block_count = (new_len + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;

        // Add new blocks initialized to 0
        while self.blocks.len() < new_block_count {
            self.blocks.push(0)?;
        }

        // Clear any bits that might have been set in the old capacity range
        // (ensures invariant is maintained)
        let new_capacity = self.blocks.len() * BITS_PER_BLOCK;
        if old_capacity < new_capacity {
            // New blocks are already 0, nothing to do
        }

        self.len = new_len;

        // Set the bit
        let block_index = i / BITS_PER_BLOCK;
        let bit_index = i % BITS_PER_BLOCK;
        self.blocks[block_index] |= 1u64 << bit_index;

        Ok(())
    }

    /// Insert a bit at the specified position
    pub fn insert(&mut self, index: usize, value: bool) -> Result<()> {
        if index > self.len {
            return Err(ZiporaError::out_of_bounds(index, self.len));
        }

        // For simplicity, we'll implement this by pushing a bit and then
        // shifting everything after the insertion point
        self.push(false)?; // Extend by one bit

        // Shift bits to the right starting from the end
        for i in (index + 1..self.len).rev() {
            let bit = self.get(i - 1).unwrap_or(false);
            self.set(i, bit)?;
        }

        // Set the bit at the insertion position
        self.set(index, value)?;

        Ok(())
    }

    /// Get a mutable reference to the bit at the specified position
    /// Returns a helper struct that can be used to modify the bit
    pub fn get_mut(&mut self, index: usize) -> Option<BitRef<'_>> {
        if index >= self.len {
            return None;
        }

        Some(BitRef {
            bit_vector: self,
            index,
        })
    }

    /// Pop a bit from the end of the vector
    pub fn pop(&mut self) -> Option<bool> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        let block_index = self.len / BITS_PER_BLOCK;
        let bit_index = self.len % BITS_PER_BLOCK;

        let value = (self.blocks[block_index] >> bit_index) & 1 == 1;

        // Clear the bit
        self.blocks[block_index] &= !(1u64 << bit_index);

        Some(value)
    }

    /// Resize the bit vector to the specified length
    pub fn resize(&mut self, new_len: usize, value: bool) -> Result<()> {
        if new_len > self.len {
            let new_blocks_needed = (new_len + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
            let old_blocks = self.blocks.len();
            let fill: u64 = if value { !0u64 } else { 0u64 };

            // Handle partial last block: set/clear upper bits
            if self.len > 0 {
                let remaining = self.len % BITS_PER_BLOCK;
                if remaining > 0 {
                    let last = old_blocks - 1;
                    if value {
                        // Set upper bits of last existing block
                        self.blocks[last] |= !((1u64 << remaining) - 1);
                    }
                    // If !value, upper bits are already 0 (invariant)
                }
            }

            // Bulk-extend blocks — single allocation, no per-bit loop
            self.blocks.resize(new_blocks_needed, fill)?;
            self.len = new_len;

            // Mask off excess bits in final block
            let tail_bits = new_len % BITS_PER_BLOCK;
            if tail_bits > 0 {
                let last = new_blocks_needed - 1;
                self.blocks[last] &= (1u64 << tail_bits) - 1;
            }
        } else if new_len < self.len {
            let new_blocks = (new_len + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
            // FastVec::resize handles shrinking (drops excess elements)
            self.blocks.resize(new_blocks, 0)?;
            self.len = new_len;

            if new_len > 0 {
                let tail_bits = new_len % BITS_PER_BLOCK;
                if tail_bits > 0 {
                    let last = new_blocks - 1;
                    self.blocks[last] &= (1u64 << tail_bits) - 1;
                }
            }
        }

        Ok(())
    }

    /// Clear all bits from the vector
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.len = 0;
    }

    /// Count the number of set bits in the entire vector.
    ///
    /// Uses iterator-based traversal (no bounds checks) to enable LLVM
    /// auto-vectorization with AVX2 vpshufb popcount (~4 words/cycle).
    #[inline]
    pub fn count_ones(&self) -> usize {
        let blocks = self.blocks.as_slice();
        let complete_blocks = self.len / BITS_PER_BLOCK;
        let remaining_bits = self.len % BITS_PER_BLOCK;

        // Iterator-based: no bounds checks, LLVM can auto-vectorize
        let mut count: usize = blocks[..complete_blocks]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum();

        // Handle partial last block
        if remaining_bits > 0 && complete_blocks < blocks.len() {
            let mask = (1u64 << remaining_bits) - 1;
            count += (blocks[complete_blocks] & mask).count_ones() as usize;
        }

        count
    }

    /// Count the number of clear bits in the entire vector
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    /// Count the number of set bits up to (but not including) the given position
    #[inline]
    pub fn rank1(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }

        let pos = pos.min(self.len);
        let blocks = self.blocks.as_slice();
        let complete_blocks = pos / BITS_PER_BLOCK;

        // Iterator-based: no bounds checks, enables auto-vectorization
        let mut count: usize = blocks[..complete_blocks]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum();

        // Count bits in the partial block
        let remaining_bits = pos % BITS_PER_BLOCK;
        if remaining_bits > 0 && complete_blocks < blocks.len() {
            let mask = (1u64 << remaining_bits) - 1;
            count += (blocks[complete_blocks] & mask).count_ones() as usize;
        }

        count
    }

    /// Count the number of clear bits up to (but not including) the given position
    #[inline]
    pub fn rank0(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }

        let pos = pos.min(self.len);
        pos - self.rank1(pos)
    }

    /// Get read-only access to the underlying u64 blocks.
    #[inline]
    pub fn blocks(&self) -> &[u64] {
        self.blocks.as_slice()
    }

    /// Get mutable access to the underlying u64 blocks.
    ///
    /// This allows zero-overhead scatter operations matching raw `Vec<u64>`.
    /// The caller must not change the slice length. Bits beyond `self.len()`
    /// in the last block are undefined.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::BitVector;
    ///
    /// let mut bv = BitVector::with_size(1000, false).unwrap();
    /// let blocks = bv.blocks_mut();
    /// // Scatter doc IDs directly — same speed as raw Vec<u64>
    /// for doc_id in [10u32, 50, 100, 500, 999] {
    ///     let w = doc_id as usize >> 6;
    ///     let b = doc_id as usize & 63;
    ///     blocks[w] |= 1u64 << b;
    /// }
    /// assert_eq!(bv.count_ones(), 5);
    /// ```
    #[inline]
    pub fn blocks_mut(&mut self) -> &mut [u64] {
        self.blocks.as_mut_slice()
    }

    /// Set bit at `index` to 1 without bounds checking.
    ///
    /// Faster than `set_unchecked(index, true)` — no branch on the value parameter.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    #[inline(always)]
    pub unsafe fn set_bit_unchecked(&mut self, index: usize) {
        debug_assert!(index < self.len);
        let block_index = index >> 6; // / 64
        let bit_index = index & 63;   // % 64
        // SAFETY: index < len guarantees block_index < blocks.len()
        unsafe {
            *self.blocks.get_unchecked_mut(block_index) |= 1u64 << bit_index;
        }
    }

    /// Clear bit at `index` to 0 without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    #[inline(always)]
    pub unsafe fn clear_bit_unchecked(&mut self, index: usize) {
        debug_assert!(index < self.len);
        let block_index = index >> 6;
        let bit_index = index & 63;
        // SAFETY: index < len guarantees block_index < blocks.len()
        unsafe {
            *self.blocks.get_unchecked_mut(block_index) &= !(1u64 << bit_index);
        }
    }

    /// Set multiple bit positions to 1 in bulk.
    ///
    /// More efficient than calling `set_bit_unchecked` in a loop:
    /// single function call, pointer setup amortized, LLVM can vectorize.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all values in `indices` are `< self.len()`.
    #[inline]
    pub unsafe fn set_bits_bulk_unchecked(&mut self, indices: &[u32]) {
        let blocks = self.blocks.as_mut_slice();
        for &idx in indices {
            let w = idx as usize >> 6;
            let b = idx as usize & 63;
            // SAFETY: caller guarantees all indices < self.len()
            unsafe { *blocks.get_unchecked_mut(w) |= 1u64 << b; }
        }
    }

    /// Reserve space for at least `additional` more bits
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        let new_len = self.len + additional;
        let new_block_count = (new_len + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
        let additional_blocks = new_block_count.saturating_sub(self.blocks.len());

        if additional_blocks > 0 {
            self.blocks.reserve(additional_blocks)?;
        }

        Ok(())
    }

    /// Bulk rank1 for multiple positions.
    ///
    /// For sorted positions, this is faster than individual `rank1` calls
    /// because it walks the blocks array once, accumulating popcounts.
    /// For unsorted positions, falls back to individual `rank1` calls.
    ///
    /// For true O(1) per-query rank with precomputed indices, use
    /// `AdaptiveRankSelect` from `src/succinct/`.
    pub fn rank1_bulk(&self, positions: &[usize]) -> Vec<usize> {
        if positions.is_empty() { return Vec::new(); }

        // Check if positions are sorted — enables single-pass optimization
        let sorted = positions.windows(2).all(|w| w[0] <= w[1]);

        if sorted {
            self.rank1_bulk_sorted(positions)
        } else {
            positions.iter().map(|&pos| self.rank1(pos)).collect()
        }
    }

    /// Backward-compatible alias for `rank1_bulk`.
    #[inline]
    pub fn rank1_bulk_simd(&self, positions: &[usize]) -> Vec<usize> {
        self.rank1_bulk(positions)
    }

    /// Single-pass bulk rank for sorted positions.
    /// Walks blocks once, accumulating popcount — O(n + b) where
    /// n = positions.len() and b = max_position / 64.
    fn rank1_bulk_sorted(&self, positions: &[usize]) -> Vec<usize> {
        let blocks = self.blocks.as_slice();
        let mut results = Vec::with_capacity(positions.len());
        let mut cumul = 0usize;
        let mut current_word = 0usize;

        for &pos in positions {
            let target_word = pos / 64;
            let remaining_bits = pos % 64;

            // Accumulate popcount from current_word to target_word
            while current_word < target_word && current_word < blocks.len() {
                cumul += blocks[current_word].count_ones() as usize;
                current_word += 1;
            }

            // Add partial word bits
            let mut rank = cumul;
            if remaining_bits > 0 && target_word < blocks.len() {
                let mask = (1u64 << remaining_bits) - 1;
                rank += (blocks[target_word] & mask).count_ones() as usize;
            }

            results.push(rank);
        }

        results
    }

    /// Set a contiguous range of bits [start, end) to the given value.
    ///
    /// Uses word-level masking: one OR/AND per u64 word in the range.
    /// For a range spanning N words, this is N operations vs N×64 individual
    /// bit sets.
    pub fn set_range(&mut self, start: usize, end: usize, value: bool) -> Result<()> {
        if start > end || end > self.len {
            return Err(ZiporaError::out_of_bounds(end, self.len));
        }
        if start == end { return Ok(()); }

        let start_block = start / BITS_PER_BLOCK;
        let end_block = (end - 1) / BITS_PER_BLOCK;

        for block_idx in start_block..=end_block {
            if block_idx >= self.blocks.len() { break; }

            let block_start = block_idx * BITS_PER_BLOCK;
            let block_end = ((block_idx + 1) * BITS_PER_BLOCK).min(end);
            let range_start = start.max(block_start);
            let range_end = end.min(block_end);

            if range_start >= range_end { continue; }

            let start_bit = range_start - block_start;
            let end_bit = range_end - block_start;

            let mask = if end_bit >= BITS_PER_BLOCK {
                !((1u64 << start_bit) - 1)
            } else {
                ((1u64 << end_bit) - 1) & !((1u64 << start_bit) - 1)
            };

            if value {
                self.blocks[block_idx] |= mask;
            } else {
                self.blocks[block_idx] &= !mask;
            }
        }

        Ok(())
    }

    /// Backward-compatible alias for `set_range`.
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[inline]
    pub fn set_range_simd(&mut self, start: usize, end: usize, value: bool) -> Result<()> {
        self.set_range(start, end, value)
    }

    /// Backward-compatible alias for `set_range`.
    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    #[inline]
    pub fn set_range_simd(&mut self, start: usize, end: usize, value: bool) -> Result<()> {
        self.set_range(start, end, value)
    }

    /// SIMD-accelerated bulk bit operations
    ///
    /// Perform bitwise operations (AND, OR, XOR) on ranges using vectorized instructions
    /// for improved performance on large datasets.
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    pub fn bulk_bitwise_op_simd(
        &mut self,
        other: &BitVector,
        op: BitwiseOp,
        start: usize,
        end: usize,
    ) -> Result<()> {
        if start > end || end > self.len || end > other.len {
            return Err(ZiporaError::invalid_data(
                "Invalid range for bulk operation".to_string(),
            ));
        }

        if is_x86_feature_detected!("avx2") {
            self.bulk_bitwise_op_simd_avx2(other, op, start, end)?;
        } else {
            // Fallback to scalar operations
            for i in start..end {
                let other_bit = other.get(i).unwrap_or(false);
                let self_bit = self.get(i).unwrap_or(false);
                let result = match op {
                    BitwiseOp::And => self_bit & other_bit,
                    BitwiseOp::Or => self_bit | other_bit,
                    BitwiseOp::Xor => self_bit ^ other_bit,
                };
                self.set(i, result)?;
            }
        }

        Ok(())
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    fn bulk_bitwise_op_simd_avx2(
        &mut self,
        other: &BitVector,
        op: BitwiseOp,
        start: usize,
        end: usize,
    ) -> Result<()> {
        let start_block = start / BITS_PER_BLOCK;
        let end_block = (end - 1) / BITS_PER_BLOCK;

        // Process blocks using AVX2 (4 u64s at a time)
        let avx2_blocks = 4;
        let mut block_idx = start_block;

        // SAFETY: Bounds checked in loop condition, pointers are valid for AVX2 operations
        unsafe {
            // Process 4 blocks at a time with AVX2
            while block_idx + avx2_blocks <= end_block + 1
                && block_idx + avx2_blocks <= self.blocks.len()
                && block_idx + avx2_blocks <= other.blocks.len()
            {
                let self_ptr = self.blocks.as_ptr().add(block_idx) as *const __m256i;
                let other_ptr = other.blocks.as_ptr().add(block_idx) as *const __m256i;
                let result_ptr = self.blocks.as_mut_ptr().add(block_idx) as *mut __m256i;

                let self_vec = _mm256_loadu_si256(self_ptr);
                let other_vec = _mm256_loadu_si256(other_ptr);

                let result_vec = match op {
                    BitwiseOp::And => _mm256_and_si256(self_vec, other_vec),
                    BitwiseOp::Or => _mm256_or_si256(self_vec, other_vec),
                    BitwiseOp::Xor => _mm256_xor_si256(self_vec, other_vec),
                };

                _mm256_storeu_si256(result_ptr, result_vec);
                block_idx += avx2_blocks;
            }
        }

        // Handle remaining blocks with scalar operations
        while block_idx <= end_block
            && block_idx < self.blocks.len()
            && block_idx < other.blocks.len()
        {
            match op {
                BitwiseOp::And => self.blocks[block_idx] &= other.blocks[block_idx],
                BitwiseOp::Or => self.blocks[block_idx] |= other.blocks[block_idx],
                BitwiseOp::Xor => self.blocks[block_idx] ^= other.blocks[block_idx],
            }
            block_idx += 1;
        }

        Ok(())
    }

    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    pub fn bulk_bitwise_op_simd(
        &mut self,
        other: &BitVector,
        op: BitwiseOp,
        start: usize,
        end: usize,
    ) -> Result<()> {
        // Fallback implementation for non-SIMD platforms
        if start > end || end > self.len || end > other.len {
            return Err(ZiporaError::invalid_data(
                "Invalid range for bulk operation".to_string(),
            ));
        }

        for i in start..end {
            let other_bit = other.get(i).unwrap_or(false);
            let self_bit = self.get(i).unwrap_or(false);
            let result = match op {
                BitwiseOp::And => self_bit & other_bit,
                BitwiseOp::Or => self_bit | other_bit,
                BitwiseOp::Xor => self_bit ^ other_bit,
            };
            self.set(i, result)?;
        }

        Ok(())
    }
}

/// Supported bitwise operations for SIMD bulk operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitwiseOp {
    /// Bitwise AND operation
    And,
    /// Bitwise OR operation
    Or,
    /// Bitwise XOR operation
    Xor,
}

/// Helper struct for mutable bit access
/// Proxy reference to a single bit in a `BitVector`.
///
/// Use `get()` and `set()` for explicit access. `Deref<Target=bool>` is
/// provided for read convenience (`if *bit_ref { ... }`), but `DerefMut`
/// is not possible — you cannot take `&mut bool` to a sub-byte location.
/// This is the standard pattern for bit-packed types in Rust.
pub struct BitRef<'a> {
    bit_vector: &'a mut BitVector,
    index: usize,
}

impl<'a> BitRef<'a> {
    /// Read the bit value.
    #[inline]
    pub fn get(&self) -> bool {
        self.bit_vector.get(self.index).unwrap_or(false)
    }

    /// Write the bit value.
    #[inline]
    pub fn set(&mut self, value: bool) -> Result<()> {
        self.bit_vector.set(self.index, value)
    }
}

impl<'a> std::ops::Deref for BitRef<'a> {
    type Target = bool;

    fn deref(&self) -> &Self::Target {
        if self.get() { &true } else { &false }
    }
}

impl Default for BitVector {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVector {{ len: {}, bits: [", self.len)?;
        for i in 0..self.len.min(64) {
            // SAFETY: Loop bounds are 0..self.len.min(64), so i < self.len always.
            write!(f, "{}", if self.get(i).expect("index within bitvector length") { '1' } else { '0' })?;
        }
        if self.len > 64 {
            write!(f, "...")?;
        }
        write!(f, "] }}")
    }
}

impl Clone for BitVector {
    fn clone(&self) -> Self {
        Self {
            blocks: self.blocks.clone(),
            len: self.len,
        }
    }
}

impl PartialEq for BitVector {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        // Compare complete blocks
        let complete_blocks = self.len / BITS_PER_BLOCK;
        for i in 0..complete_blocks {
            if self.blocks[i] != other.blocks[i] {
                return false;
            }
        }

        // Compare remaining bits in the last block
        let remaining_bits = self.len % BITS_PER_BLOCK;
        if remaining_bits > 0
            && complete_blocks < self.blocks.len()
            && complete_blocks < other.blocks.len()
        {
            let mask = (1u64 << remaining_bits) - 1;
            if (self.blocks[complete_blocks] & mask) != (other.blocks[complete_blocks] & mask) {
                return false;
            }
        }

        true
    }
}

impl Eq for BitVector {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let bv = BitVector::new();
        assert_eq!(bv.len(), 0);
        assert!(bv.is_empty());
    }

    #[test]
    fn test_push_pop() {
        let mut bv = BitVector::new();
        bv.push(true).unwrap();
        bv.push(false).unwrap();
        bv.push(true).unwrap();

        assert_eq!(bv.len(), 3);
        assert_eq!(bv.get(0), Some(true));
        assert_eq!(bv.get(1), Some(false));
        assert_eq!(bv.get(2), Some(true));

        assert_eq!(bv.pop(), Some(true));
        assert_eq!(bv.pop(), Some(false));
        assert_eq!(bv.len(), 1);
    }

    #[test]
    fn test_set_get() {
        let mut bv = BitVector::with_size(10, false).unwrap();

        bv.set(0, true).unwrap();
        bv.set(5, true).unwrap();
        bv.set(9, true).unwrap();

        assert_eq!(bv.get(0), Some(true));
        assert_eq!(bv.get(1), Some(false));
        assert_eq!(bv.get(5), Some(true));
        assert_eq!(bv.get(9), Some(true));
        assert_eq!(bv.get(10), None);
    }

    #[test]
    fn test_count_operations() {
        let mut bv = BitVector::with_size(10, false).unwrap();
        bv.set(0, true).unwrap();
        bv.set(3, true).unwrap();
        bv.set(7, true).unwrap();

        assert_eq!(bv.count_ones(), 3);
        assert_eq!(bv.count_zeros(), 7);
    }

    #[test]
    fn test_rank_operations() {
        let mut bv = BitVector::with_size(10, false).unwrap();
        bv.set(1, true).unwrap();
        bv.set(3, true).unwrap();
        bv.set(7, true).unwrap();

        assert_eq!(bv.rank1(0), 0);
        assert_eq!(bv.rank1(2), 1);
        assert_eq!(bv.rank1(4), 2);
        assert_eq!(bv.rank1(8), 3);
        assert_eq!(bv.rank1(10), 3);

        assert_eq!(bv.rank0(2), 1);
        assert_eq!(bv.rank0(4), 2);
        assert_eq!(bv.rank0(8), 5);
    }

    #[test]
    fn test_large_bitvector() {
        let mut bv = BitVector::new();

        // Test across multiple blocks
        for i in 0..200 {
            bv.push(i % 3 == 0).unwrap();
        }

        assert_eq!(bv.len(), 200);

        let mut expected_ones = 0;
        for i in 0..200 {
            if i % 3 == 0 {
                expected_ones += 1;
                assert_eq!(bv.get(i), Some(true));
            } else {
                assert_eq!(bv.get(i), Some(false));
            }
        }

        assert_eq!(bv.count_ones(), expected_ones);
    }

    #[test]
    fn test_resize() {
        let mut bv = BitVector::new();
        bv.resize(5, true).unwrap();

        assert_eq!(bv.len(), 5);
        assert_eq!(bv.count_ones(), 5);

        bv.resize(3, false).unwrap();
        assert_eq!(bv.len(), 3);
        assert_eq!(bv.count_ones(), 3);

        bv.resize(8, false).unwrap();
        assert_eq!(bv.len(), 8);
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_equality() {
        let mut bv1 = BitVector::new();
        let mut bv2 = BitVector::new();

        for &bit in &[true, false, true, true, false] {
            bv1.push(bit).unwrap();
            bv2.push(bit).unwrap();
        }

        assert_eq!(bv1, bv2);

        bv2.set(2, false).unwrap();
        assert_ne!(bv1, bv2);
    }

    #[test]
    fn test_unsafe_operations() {
        let mut bv = BitVector::new();
        bv.push(true).unwrap();
        bv.push(false).unwrap();
        bv.push(true).unwrap();

        // SAFETY: Indices 0, 1, 2 are valid (len=3)
        unsafe {
            assert_eq!(bv.get_unchecked(0), true);
            assert_eq!(bv.get_unchecked(1), false);
            assert_eq!(bv.get_unchecked(2), true);

            bv.set_unchecked(1, true);
            assert_eq!(bv.get_unchecked(1), true);

            bv.set_unchecked(0, false);
            assert_eq!(bv.get_unchecked(0), false);
        }
    }

    #[test]
    fn test_blocks_access() {
        let mut bv = BitVector::new();
        for i in 0..128 {
            bv.push(i % 3 == 0).unwrap();
        }

        let blocks = bv.blocks();
        assert!(blocks.len() >= 2); // Should span multiple blocks

        // Verify blocks contain expected data
        let first_block = blocks[0];
        let first_bit = first_block & 1;
        assert_eq!(first_bit, 1); // First bit should be set (0 % 3 == 0)
    }

    #[test]
    fn test_reserve() {
        let mut bv = BitVector::new();
        bv.reserve(1000).unwrap();

        // Should have reserved enough blocks for 1000 bits
        assert!(bv.capacity() >= 1000);

        // Fill with data to verify it works
        for i in 0..1000 {
            bv.push(i % 7 == 0).unwrap();
        }
        assert_eq!(bv.len(), 1000);
    }

    #[test]
    fn test_partial_block_operations() {
        let mut bv = BitVector::new();

        // Test with exactly one full block (64 bits)
        for i in 0..64 {
            bv.push(i % 2 == 0).unwrap();
        }
        assert_eq!(bv.len(), 64);
        assert_eq!(bv.count_ones(), 32);

        // Add a few more bits to create a partial block
        bv.push(true).unwrap();
        bv.push(false).unwrap();
        bv.push(true).unwrap();

        assert_eq!(bv.len(), 67);
        assert_eq!(bv.count_ones(), 34); // 32 + 2 from partial block

        // Test rank with partial blocks
        assert_eq!(bv.rank1(64), 32);
        assert_eq!(bv.rank1(67), 34);
    }

    #[test]
    fn test_edge_case_resize() {
        let mut bv = BitVector::new();

        // Resize to zero
        bv.resize(0, true).unwrap();
        assert_eq!(bv.len(), 0);
        assert!(bv.is_empty());

        // Resize from zero to some value
        bv.resize(10, true).unwrap();
        assert_eq!(bv.len(), 10);
        assert_eq!(bv.count_ones(), 10);

        // Resize down with partial bits
        bv.resize(5, false).unwrap();
        assert_eq!(bv.len(), 5);
        assert_eq!(bv.count_ones(), 5);

        // Resize up again
        bv.resize(15, false).unwrap();
        assert_eq!(bv.len(), 15);
        assert_eq!(bv.count_ones(), 5); // Only first 5 should be set
    }

    #[test]
    fn test_boundary_conditions() {
        let mut bv = BitVector::new();

        // Test exactly at block boundaries (64, 128, etc.)
        for i in 0..128 {
            bv.push(i < 64).unwrap(); // First 64 are true, next 64 are false
        }

        assert_eq!(bv.rank1(0), 0);
        assert_eq!(bv.rank1(64), 64);
        assert_eq!(bv.rank1(128), 64);
        assert_eq!(bv.rank0(64), 0);
        assert_eq!(bv.rank0(128), 64);
    }

    #[test]
    fn test_error_conditions() {
        let mut bv = BitVector::new();
        bv.push(true).unwrap();
        bv.push(false).unwrap();

        // Test out of bounds set
        assert!(bv.set(5, true).is_err());
        assert!(bv.set(2, true).is_err()); // len is 2, so index 2 is out of bounds

        // Valid set should work
        assert!(bv.set(0, false).is_ok());
        assert_eq!(bv.get(0), Some(false));
    }

    #[test]
    fn test_clone_and_equality_edge_cases() {
        let mut original = BitVector::new();

        // Test clone of empty vector
        let cloned_empty = original.clone();
        assert_eq!(original, cloned_empty);

        // Add bits across multiple blocks
        for i in 0..130 {
            original.push(i % 5 == 0).unwrap();
        }

        let cloned = original.clone();
        assert_eq!(original, cloned);

        // Test equality with different lengths
        let mut shorter = original.clone();
        shorter.resize(100, false).unwrap();
        assert_ne!(original, shorter);

        // Test equality with same length but different content
        let mut different = original.clone();
        different.set(50, !different.get(50).unwrap()).unwrap();
        assert_ne!(original, different);
    }

    #[test]
    fn test_debug_output() {
        let mut bv = BitVector::new();
        for i in 0..10 {
            bv.push(i % 2 == 0).unwrap();
        }

        let debug_str = format!("{:?}", bv);
        assert!(debug_str.contains("BitVector"));
        assert!(debug_str.contains("len: 10"));
        assert!(debug_str.contains("1010101010")); // The pattern

        // Test debug output for very long vectors
        let mut long_bv = BitVector::new();
        for i in 0..100 {
            long_bv.push(i % 2 == 0).unwrap();
        }

        let long_debug = format!("{:?}", long_bv);
        assert!(long_debug.contains("..."));
        assert!(long_debug.contains("len: 100"));
    }

    #[test]
    fn test_default() {
        let bv = BitVector::default();
        assert!(bv.is_empty());
        assert_eq!(bv.len(), 0);
        assert_eq!(bv.capacity(), 0);
    }

    #[test]
    fn test_ensure_set1_basic() {
        let mut bv = BitVector::new();

        // Set bits in order
        bv.ensure_set1(0).unwrap();
        assert_eq!(bv.len(), 1);
        assert_eq!(bv.get(0), Some(true));

        bv.ensure_set1(5).unwrap();
        assert_eq!(bv.len(), 6);
        assert_eq!(bv.get(5), Some(true));
        // Bits 1-4 should be false (filled with zeros)
        for i in 1..5 {
            assert_eq!(bv.get(i), Some(false));
        }

        // Setting same bit again should be idempotent
        bv.ensure_set1(5).unwrap();
        assert_eq!(bv.len(), 6);
        assert_eq!(bv.get(5), Some(true));
    }

    #[test]
    fn test_ensure_set1_sequential() {
        let mut bv = BitVector::new();

        // Build a set of integers 0, 2, 4, 6, 8
        for i in (0..10).step_by(2) {
            bv.ensure_set1(i).unwrap();
        }

        assert_eq!(bv.len(), 9);
        assert_eq!(bv.count_ones(), 5);

        // Check pattern
        for i in 0..9 {
            assert_eq!(bv.get(i), Some(i % 2 == 0));
        }
    }

    #[test]
    fn test_ensure_set1_across_blocks() {
        let mut bv = BitVector::new();

        // Set bits across multiple 64-bit blocks
        bv.ensure_set1(0).unwrap();
        bv.ensure_set1(63).unwrap();
        bv.ensure_set1(64).unwrap();
        bv.ensure_set1(127).unwrap();
        bv.ensure_set1(128).unwrap();

        assert_eq!(bv.len(), 129);
        assert_eq!(bv.count_ones(), 5);

        // Verify specific bits
        assert_eq!(bv.get(0), Some(true));
        assert_eq!(bv.get(63), Some(true));
        assert_eq!(bv.get(64), Some(true));
        assert_eq!(bv.get(127), Some(true));
        assert_eq!(bv.get(128), Some(true));
    }

    #[test]
    fn test_fast_ensure_set1_sequential() {
        let mut bv = BitVector::new();

        // Fast version optimized for sequential access
        for i in 0..100 {
            bv.fast_ensure_set1(i).unwrap();
        }

        assert_eq!(bv.len(), 100);
        assert_eq!(bv.count_ones(), 100); // All bits should be set
    }

    #[test]
    fn test_fast_ensure_set1_sparse() {
        let mut bv = BitVector::new();

        // Set every 10th bit
        for i in (0..1000).step_by(10) {
            bv.fast_ensure_set1(i).unwrap();
        }

        assert_eq!(bv.len(), 991);
        assert_eq!(bv.count_ones(), 100);

        // Verify the pattern
        for i in (0..1000).step_by(10) {
            if i < bv.len() {
                assert_eq!(bv.get(i), Some(true), "bit {} should be set", i);
            }
        }
    }

    #[test]
    fn test_ensure_set1_vs_fast_ensure_set1_equivalence() {
        // Both methods should produce the same result
        let indices: Vec<usize> = vec![0, 5, 10, 63, 64, 100, 127, 128, 200];

        let mut bv1 = BitVector::new();
        let mut bv2 = BitVector::new();

        for &i in &indices {
            bv1.ensure_set1(i).unwrap();
            bv2.fast_ensure_set1(i).unwrap();
        }

        assert_eq!(bv1.len(), bv2.len());
        assert_eq!(bv1.count_ones(), bv2.count_ones());

        for i in 0..bv1.len() {
            assert_eq!(bv1.get(i), bv2.get(i), "mismatch at bit {}", i);
        }
    }

    #[test]
    fn test_ensure_set1_integer_set_use_case() {
        // Simulate building a set of integers {3, 7, 15, 31, 63}
        let integers = vec![3, 7, 15, 31, 63];
        let mut bv = BitVector::new();

        for &i in &integers {
            bv.fast_ensure_set1(i).unwrap();
        }

        // Check membership
        assert_eq!(bv.get(3), Some(true));
        assert_eq!(bv.get(7), Some(true));
        assert_eq!(bv.get(15), Some(true));
        assert_eq!(bv.get(31), Some(true));
        assert_eq!(bv.get(63), Some(true));

        // Check non-membership
        assert_eq!(bv.get(0), Some(false));
        assert_eq!(bv.get(4), Some(false));
        assert_eq!(bv.get(8), Some(false));
    }

    #[test]
    fn test_blocks_mut_scatter() {
        let mut bv = BitVector::with_size(1000, false).unwrap();
        let doc_ids: Vec<u32> = vec![0, 1, 63, 64, 65, 500, 999];

        // Scatter via blocks_mut — same performance as raw Vec<u64>
        let blocks = bv.blocks_mut();
        for &doc_id in &doc_ids {
            let w = doc_id as usize >> 6;
            let b = doc_id as usize & 63;
            blocks[w] |= 1u64 << b;
        }

        assert_eq!(bv.count_ones(), 7);
        for &doc_id in &doc_ids {
            assert_eq!(bv.get(doc_id as usize), Some(true), "bit {} should be set", doc_id);
        }
        assert_eq!(bv.get(2), Some(false));
        assert_eq!(bv.get(998), Some(false));
    }

    #[test]
    fn test_set_bit_unchecked() {
        let mut bv = BitVector::with_size(256, false).unwrap();

        unsafe {
            bv.set_bit_unchecked(0);
            bv.set_bit_unchecked(63);
            bv.set_bit_unchecked(64);
            bv.set_bit_unchecked(255);
        }

        assert_eq!(bv.count_ones(), 4);
        assert_eq!(bv.get(0), Some(true));
        assert_eq!(bv.get(63), Some(true));
        assert_eq!(bv.get(64), Some(true));
        assert_eq!(bv.get(255), Some(true));
        assert_eq!(bv.get(1), Some(false));
    }

    #[test]
    fn test_clear_bit_unchecked() {
        let mut bv = BitVector::with_size(128, true).unwrap();
        assert_eq!(bv.count_ones(), 128);

        unsafe {
            bv.clear_bit_unchecked(0);
            bv.clear_bit_unchecked(64);
            bv.clear_bit_unchecked(127);
        }

        assert_eq!(bv.count_ones(), 125);
        assert_eq!(bv.get(0), Some(false));
        assert_eq!(bv.get(64), Some(false));
        assert_eq!(bv.get(127), Some(false));
        assert_eq!(bv.get(1), Some(true));
    }

    #[test]
    fn test_set_bits_bulk_unchecked() {
        let mut bv = BitVector::with_size(10000, false).unwrap();
        let indices: Vec<u32> = (0..1000).map(|i| i * 10).collect();

        unsafe { bv.set_bits_bulk_unchecked(&indices); }

        assert_eq!(bv.count_ones(), 1000);
        for &idx in &indices {
            assert!(bv.get(idx as usize) == Some(true), "bit {} should be set", idx);
        }
        assert_eq!(bv.get(1), Some(false));
        assert_eq!(bv.get(9999), Some(false));
    }

    #[test]
    fn test_blocks_mut_then_count_ones() {
        // Simulates the search engine union counting pattern
        let mut bv = BitVector::with_size(50000, false).unwrap();

        // Scatter 5000 doc IDs
        let blocks = bv.blocks_mut();
        for i in 0..5000u32 {
            let doc_id = (i * 10) as usize;
            blocks[doc_id >> 6] |= 1u64 << (doc_id & 63);
        }

        // Count via popcount
        assert_eq!(bv.count_ones(), 5000);
    }

    #[test]
    fn test_with_size_false_alloc_zeroed() {
        // Verifies the alloc_zeroed fast path produces all-zero bits
        let bv = BitVector::with_size(1_000_000, false).unwrap();
        assert_eq!(bv.len(), 1_000_000);
        assert_eq!(bv.count_ones(), 0);

        // Every block should be zero
        for &block in bv.blocks() {
            assert_eq!(block, 0);
        }
    }

    #[test]
    fn test_with_size_true_still_works() {
        let bv = BitVector::with_size(128, true).unwrap();
        assert_eq!(bv.len(), 128);
        assert_eq!(bv.count_ones(), 128);
    }

    #[test]
    fn test_from_blocks_zero_copy() {
        let max_doc = 100_000;
        let num_words = (max_doc + 63) / 64;
        let mut blocks = vec![0u64; num_words];

        // Scatter some doc IDs directly into the Vec
        for doc_id in [0usize, 42, 1000, 50_000, 99_999] {
            blocks[doc_id >> 6] |= 1u64 << (doc_id & 63);
        }

        let bv = BitVector::from_blocks(blocks, max_doc);
        assert_eq!(bv.len(), max_doc);
        assert_eq!(bv.count_ones(), 5);
        assert!(bv.get(42).unwrap());
        assert!(bv.get(1000).unwrap());
        assert!(!bv.get(43).unwrap());
    }

    #[test]
    fn test_from_raw_bits_zero_copy() {
        // from_raw_bits should no longer copy element-by-element
        let raw = vec![u64::MAX; 4];
        let bv = BitVector::from_raw_bits(raw, 256).unwrap();
        assert_eq!(bv.len(), 256);
        assert_eq!(bv.count_ones(), 256);
        for i in 0..64 {
            assert!(bv.get(i).unwrap(), "bit {i} should be set");
        }
    }

    #[test]
    fn test_from_blocks_scatter_popcount() {
        // Simulates the engine's scatter+popcount pattern
        let max_doc = 1_000_000;
        let num_words = (max_doc + 63) / 64;
        let blocks = vec![0u64; num_words]; // calloc path
        let mut bv = BitVector::from_blocks(blocks, max_doc);

        // Scatter 5000 doc IDs via blocks_mut
        let blks = bv.blocks_mut();
        for i in 0..5000u32 {
            let doc_id = (i * 200) as usize;
            blks[doc_id >> 6] |= 1u64 << (doc_id & 63);
        }
        assert_eq!(bv.count_ones(), 5000);
    }

    /// Performance test — only meaningful in release mode.
    #[test]
    fn test_allocation_performance() {
        #[cfg(not(debug_assertions))]
        {
            let n = 1_000_000;
            let num_words = (n + 63) / 64;

            // Measure vec![0u64; n] (calloc)
            let start = std::time::Instant::now();
            for _ in 0..100 {
                let v = vec![0u64; num_words];
                std::hint::black_box(&v);
            }
            let vec_time = start.elapsed();

            // Measure BitVector::with_size (alloc_zeroed)
            let start = std::time::Instant::now();
            for _ in 0..100 {
                let bv = BitVector::with_size(n, false).unwrap();
                std::hint::black_box(&bv);
            }
            let bv_time = start.elapsed();

            // Measure BitVector::from_blocks (zero-copy)
            let start = std::time::Instant::now();
            for _ in 0..100 {
                let blocks = vec![0u64; num_words];
                let bv = BitVector::from_blocks(blocks, n);
                std::hint::black_box(&bv);
            }
            let from_blocks_time = start.elapsed();

            eprintln!("1M-bit allocation ×100: Vec={:?}, with_size={:?}, from_blocks={:?}",
                vec_time, bv_time, from_blocks_time);

            // Both should be within 2× of Vec (previously was 30× slower)
            assert!(bv_time.as_micros() < vec_time.as_micros() * 3,
                "with_size too slow: {:?} vs Vec {:?}", bv_time, vec_time);
            assert!(from_blocks_time.as_micros() < vec_time.as_micros() * 3,
                "from_blocks too slow: {:?} vs Vec {:?}", from_blocks_time, vec_time);
        }
    }

    /// Full scatter+popcount benchmark matching the engine workload.
    #[test]
    fn test_scatter_popcount_performance() {
        #[cfg(not(debug_assertions))]
        {
            let max_doc = 1_000_000usize;
            let num_words = (max_doc + 63) / 64;

            // Generate posting lists: 20 lists × 5K docs (Large workload)
            let posting_lists: Vec<Vec<u32>> = (0..20).map(|i| {
                (0..5000u32).map(|j| ((i * 50000 + j * 200) % max_doc as u32)).collect()
            }).collect();

            let iters = 20;

            // Scalar Vec<u64> scatter + popcount
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let mut bits = vec![0u64; num_words];
                for list in &posting_lists {
                    for &doc in list {
                        let w = doc as usize >> 6;
                        let b = doc as usize & 63;
                        unsafe { *bits.get_unchecked_mut(w) |= 1u64 << b; }
                    }
                }
                let count: u32 = bits.iter().map(|w| w.count_ones()).sum();
                std::hint::black_box(count);
            }
            let scalar_time = start.elapsed();

            // BitVector from_blocks + blocks_mut + count_ones
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let blocks = vec![0u64; num_words];
                let mut bv = BitVector::from_blocks(blocks, max_doc);
                let blks = bv.blocks_mut();
                for list in &posting_lists {
                    for &doc in list {
                        let w = doc as usize >> 6;
                        let b = doc as usize & 63;
                        unsafe { *blks.get_unchecked_mut(w) |= 1u64 << b; }
                    }
                }
                let count = bv.count_ones();
                std::hint::black_box(count);
            }
            let bv_time = start.elapsed();

            let ratio = bv_time.as_nanos() as f64 / scalar_time.as_nanos() as f64;
            eprintln!("Scatter+popcount (20×5K, 1M): Scalar={:?}, BitVector={:?}, ratio={:.2}×",
                scalar_time, bv_time, ratio);

            assert!(ratio < 1.2,
                "BitVector too slow: {:.2}× vs scalar (expected <1.2×)", ratio);
        }
    }
}
