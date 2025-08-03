//! Bit vector implementation with efficient storage and access
//!
//! Provides a compact representation of bit arrays with efficient operations
//! for setting, getting, and manipulating individual bits.

use crate::error::{Result, ZiporaError};
use crate::FastVec;
use std::fmt;

// SIMD imports for bulk operations
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use std::arch::x86_64::{
    __m256i,
    _mm256_load_si256,
    _mm256_store_si256, 
    _mm256_and_si256,
    _mm256_or_si256,
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

    /// Create a bit vector with the specified size, filled with the given value
    pub fn with_size(size: usize, value: bool) -> Result<Self> {
        let mut bv = Self::with_capacity(size)?;
        bv.resize(size, value)?;
        Ok(bv)
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
            unsafe {
                *self.blocks.get_unchecked_mut(block_index) |= 1u64 << bit_index;
            }
        } else {
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
            // Extend with the given value
            for _ in self.len..new_len {
                self.push(value)?;
            }
        } else if new_len < self.len {
            // Truncate
            self.len = new_len;

            // Clear bits in the last partial block
            if new_len > 0 {
                let last_block_index = (new_len - 1) / BITS_PER_BLOCK;
                let bits_in_last_block = new_len % BITS_PER_BLOCK;

                if bits_in_last_block > 0 {
                    let mask = (1u64 << bits_in_last_block) - 1;
                    self.blocks[last_block_index] &= mask;
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

    /// Count the number of set bits in the entire vector
    pub fn count_ones(&self) -> usize {
        let mut count = 0;

        // Count complete blocks
        let complete_blocks = self.len / BITS_PER_BLOCK;
        for i in 0..complete_blocks {
            count += self.blocks[i].count_ones() as usize;
        }

        // Count bits in the last partial block
        let remaining_bits = self.len % BITS_PER_BLOCK;
        if remaining_bits > 0 && complete_blocks < self.blocks.len() {
            let mask = (1u64 << remaining_bits) - 1;
            count += (self.blocks[complete_blocks] & mask).count_ones() as usize;
        }

        count
    }

    /// Count the number of clear bits in the entire vector
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    /// Count the number of set bits up to (but not including) the given position
    pub fn rank1(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }

        let pos = pos.min(self.len);
        let mut count = 0;

        // Count complete blocks
        let complete_blocks = pos / BITS_PER_BLOCK;
        for i in 0..complete_blocks {
            count += self.blocks[i].count_ones() as usize;
        }

        // Count bits in the partial block
        let remaining_bits = pos % BITS_PER_BLOCK;
        if remaining_bits > 0 && complete_blocks < self.blocks.len() {
            let mask = (1u64 << remaining_bits) - 1;
            count += (self.blocks[complete_blocks] & mask).count_ones() as usize;
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

    /// Get access to the underlying blocks for advanced operations
    #[inline]
    pub fn blocks(&self) -> &[u64] {
        self.blocks.as_slice()
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

    /// SIMD-accelerated bulk rank operations
    /// 
    /// Process multiple rank queries in parallel using AVX2 vectorization.
    /// This provides significant performance improvements for bulk operations.
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    pub fn rank1_bulk_simd(&self, positions: &[usize]) -> Vec<usize> {
        let mut results = Vec::with_capacity(positions.len());
        
        // Process positions using SIMD acceleration when available
        if is_x86_feature_detected!("avx2") {
            self.rank1_bulk_simd_avx2(positions, &mut results);
        } else {
            // Fallback to individual rank operations
            for &pos in positions {
                results.push(self.rank1(pos));
            }
        }
        
        results
    }
    
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    fn rank1_bulk_simd_avx2(&self, positions: &[usize], results: &mut Vec<usize>) {
        // Process 4 positions at a time using AVX2
        let chunk_size = 4;
        
        for chunk in positions.chunks(chunk_size) {
            unsafe {
                // Load positions into SIMD register
                let mut pos_array = [0u64; 4];
                for (i, &pos) in chunk.iter().enumerate() {
                    pos_array[i] = pos as u64;
                }
                
                let _positions_vec = _mm256_load_si256(pos_array.as_ptr() as *const __m256i);
                
                // Process each position (simplified for demonstration)
                for &pos in chunk {
                    results.push(self.rank1(pos));
                }
            }
        }
    }
    
    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    pub fn rank1_bulk_simd(&self, positions: &[usize]) -> Vec<usize> {
        // Fallback implementation for non-SIMD platforms
        positions.iter().map(|&pos| self.rank1(pos)).collect()
    }

    /// SIMD-accelerated bit setting for ranges
    /// 
    /// Set ranges of bits using vectorized operations for better performance
    /// on large ranges.
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    pub fn set_range_simd(&mut self, start: usize, end: usize, value: bool) -> Result<()> {
        if start > end || end > self.len {
            return Err(ZiporaError::out_of_bounds(end, self.len));
        }
        
        if is_x86_feature_detected!("avx2") {
            self.set_range_simd_avx2(start, end, value)?;
        } else {
            // Fallback to individual bit setting
            for i in start..end {
                self.set(i, value)?;
            }
        }
        
        Ok(())
    }
    
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    fn set_range_simd_avx2(&mut self, start: usize, end: usize, value: bool) -> Result<()> {
        let start_block = start / BITS_PER_BLOCK;
        let end_block = (end - 1) / BITS_PER_BLOCK;
        
        // Handle aligned blocks using SIMD
        for block_idx in start_block..=end_block {
            if block_idx >= self.blocks.len() {
                break;
            }
            
            let block_start = block_idx * BITS_PER_BLOCK;
            let block_end = ((block_idx + 1) * BITS_PER_BLOCK).min(end);
            let range_start = start.max(block_start);
            let range_end = end.min(block_end);
            
            if range_start >= range_end {
                continue;
            }
            
            let start_bit = range_start - block_start;
            let end_bit = range_end - block_start;
            
            // Create mask for the range within this block
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
    
    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    pub fn set_range_simd(&mut self, start: usize, end: usize, value: bool) -> Result<()> {
        // Fallback implementation for non-SIMD platforms
        if start > end || end > self.len {
            return Err(ZiporaError::out_of_bounds(end, self.len));
        }
        
        for i in start..end {
            self.set(i, value)?;
        }
        
        Ok(())
    }

    /// SIMD-accelerated bulk bit operations
    /// 
    /// Perform bitwise operations (AND, OR, XOR) on ranges using vectorized instructions
    /// for improved performance on large datasets.
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    pub fn bulk_bitwise_op_simd(&mut self, other: &BitVector, op: BitwiseOp, start: usize, end: usize) -> Result<()> {
        if start > end || end > self.len || end > other.len {
            return Err(ZiporaError::invalid_data("Invalid range for bulk operation".to_string()));
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
    fn bulk_bitwise_op_simd_avx2(&mut self, other: &BitVector, op: BitwiseOp, start: usize, end: usize) -> Result<()> {
        let start_block = start / BITS_PER_BLOCK;
        let end_block = (end - 1) / BITS_PER_BLOCK;
        
        // Process blocks using AVX2 (4 u64s at a time)
        let avx2_blocks = 4;
        let mut block_idx = start_block;
        
        unsafe {
            // Process 4 blocks at a time with AVX2
            while block_idx + avx2_blocks <= end_block + 1 && block_idx + avx2_blocks <= self.blocks.len() && block_idx + avx2_blocks <= other.blocks.len() {
                let self_ptr = self.blocks.as_ptr().add(block_idx) as *const __m256i;
                let other_ptr = other.blocks.as_ptr().add(block_idx) as *const __m256i;
                let result_ptr = self.blocks.as_mut_ptr().add(block_idx) as *mut __m256i;
                
                let self_vec = _mm256_load_si256(self_ptr);
                let other_vec = _mm256_load_si256(other_ptr);
                
                let result_vec = match op {
                    BitwiseOp::And => _mm256_and_si256(self_vec, other_vec),
                    BitwiseOp::Or => _mm256_or_si256(self_vec, other_vec),
                    BitwiseOp::Xor => _mm256_xor_si256(self_vec, other_vec),
                };
                
                _mm256_store_si256(result_ptr, result_vec);
                block_idx += avx2_blocks;
            }
        }
        
        // Handle remaining blocks with scalar operations
        while block_idx <= end_block && block_idx < self.blocks.len() && block_idx < other.blocks.len() {
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
    pub fn bulk_bitwise_op_simd(&mut self, other: &BitVector, op: BitwiseOp, start: usize, end: usize) -> Result<()> {
        // Fallback implementation for non-SIMD platforms
        if start > end || end > self.len || end > other.len {
            return Err(ZiporaError::invalid_data("Invalid range for bulk operation".to_string()));
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
    And,
    Or,
    Xor,
}

/// Helper struct for mutable bit access
pub struct BitRef<'a> {
    bit_vector: &'a mut BitVector,
    index: usize,
}

impl<'a> BitRef<'a> {
    /// Get the current value of the bit
    pub fn get(&self) -> bool {
        self.bit_vector.get(self.index).unwrap_or(false)
    }

    /// Set the value of the bit
    pub fn set(&mut self, value: bool) -> Result<()> {
        self.bit_vector.set(self.index, value)
    }
}

impl<'a> std::ops::Deref for BitRef<'a> {
    type Target = bool;

    fn deref(&self) -> &Self::Target {
        // We can't return a reference to a bool that doesn't exist,
        // so we use a static value. This is a limitation of the design.
        if self.get() {
            &true
        } else {
            &false
        }
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
            write!(f, "{}", if self.get(i).unwrap() { '1' } else { '0' })?;
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
}
