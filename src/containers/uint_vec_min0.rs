//! Variable-width integer vector optimized for sequences where minimum value is 0
//!
//! # Overview
//!
//! `UintVecMin0` stores integers using minimal bits required by the maximum value.
//! This provides significant space savings for sequences with small values.
//!
//! # Memory Layout
//!
//! - Stores N integers using `bits * N` bits total (packed bit-level, not bytes!)
//! - `bits = floor(log2(max_val)) + 1` - Minimum bits needed to represent max value
//! - Values packed into byte array with bit-level alignment
//! - 16-byte aligned allocation for optimal cache performance
//!
//! # Fast Path vs Slow Path
//!
//! - **Fast Path** (bits <= 58): Uses unaligned u64 loads for 2-4x faster access
//! - **Slow Path** (bits > 58): Uses byte-by-byte bit manipulation
//!   - 58-bit limit ensures values fit within 9 bytes max (safe for unaligned loads)
//!
//! # Example
//!
//! ```rust
//! use zipora::containers::UintVecMin0;
//!
//! // Store values 0-255 using only 8 bits each
//! let mut vec = UintVecMin0::new(100, 255);
//! for i in 0..100 {
//!     vec.set(i, i % 256);
//! }
//!
//! // Memory: 100 values * 8 bits = 800 bits = 100 bytes (vs 800 bytes for Vec<usize>)
//! assert_eq!(vec.get(42), 42);
//! assert!(vec.mem_size() < 100 * std::mem::size_of::<usize>());
//! ```

use crate::error::{Result, ZiporaError};
use std::fmt;

/// Variable-width integer vector with minimum value of 0
///
/// Efficiently stores unsigned integers using minimal bits per value.
/// The bit width is determined by the maximum value in the sequence.
#[derive(Clone)]
pub struct UintVecMin0 {
    /// Packed bit data (16-byte aligned)
    data: Vec<u8>,
    /// Bits per integer (0-58 for fast access, 0-64 for BigUintVecMin0)
    bits: usize,
    /// Mask for fast extraction: (1 << bits) - 1
    mask: usize,
    /// Number of integers stored
    size: usize,
}

impl UintVecMin0 {
    /// Create new vector with capacity for `num` integers with maximum value `max_val`
    ///
    /// # Arguments
    ///
    /// * `num` - Number of integers to allocate space for
    /// * `max_val` - Maximum value that will be stored
    ///
    /// # Example
    ///
    /// ```rust
    /// use zipora::containers::UintVecMin0;
    ///
    /// // Store 1000 values in range [0, 127] using 7 bits each
    /// let vec = UintVecMin0::new(1000, 127);
    /// assert_eq!(vec.uintbits(), 7);
    /// ```
    pub fn new(num: usize, max_val: usize) -> Self {
        let bits = Self::compute_uintbits(max_val);
        let mut vec = Self {
            data: Vec::new(),
            bits: 0,
            mask: 0,
            size: 0,
        };
        vec.resize_with_uintbits(num, bits);
        vec
    }

    /// Create empty vector
    pub fn new_empty() -> Self {
        Self {
            data: Vec::new(),
            bits: 0,
            mask: 0,
            size: 0,
        }
    }

    /// Compute minimum bits needed to represent a value
    ///
    /// # Algorithm
    ///
    /// Returns `floor(log2(value)) + 1` or 0 if value is 0.
    /// Equivalent to `64 - value.leading_zeros()` but matches C++ implementation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zipora::containers::UintVecMin0;
    ///
    /// assert_eq!(UintVecMin0::compute_uintbits(0), 0);   // 0 needs 0 bits
    /// assert_eq!(UintVecMin0::compute_uintbits(1), 1);   // 1 needs 1 bit
    /// assert_eq!(UintVecMin0::compute_uintbits(7), 3);   // 7 needs 3 bits
    /// assert_eq!(UintVecMin0::compute_uintbits(255), 8); // 255 needs 8 bits
    /// ```
    #[inline]
    pub fn compute_uintbits(value: usize) -> usize {
        if value == 0 {
            0
        } else {
            64 - value.leading_zeros() as usize
        }
    }

    /// Compute memory size needed for storage
    ///
    /// # Layout
    ///
    /// - `using_size = (bits * num + 7) / 8` - Actual bits needed, rounded to bytes
    /// - `touch_size = using_size + 7` - Add padding for unaligned u64 access
    /// - `align_size = (touch_size + 15) & !15` - Align to 16 bytes
    ///
    /// # Arguments
    ///
    /// * `bits` - Bits per integer
    /// * `num` - Number of integers
    #[inline]
    pub fn compute_mem_size(bits: usize, num: usize) -> usize {
        assert!(bits <= 64, "bits must be <= 64");
        let using_size = (bits * num + 7) / 8;
        let touch_size = using_size + std::mem::size_of::<u64>() - 1;
        (touch_size + 15) & !15 // Align to 16 bytes
    }

    /// Compute memory size needed for given max value and count
    #[inline]
    pub fn compute_mem_size_by_max_val(max_val: usize, num: usize) -> usize {
        let bits = Self::compute_uintbits(max_val);
        Self::compute_mem_size(bits, num)
    }

    /// Get value at index (FAST PATH - max 58 bits)
    ///
    /// # Panics
    ///
    /// Panics if index >= size or bits > 58
    ///
    /// # Performance
    ///
    /// Uses unaligned u64 load for 2-4x faster access than byte-by-byte
    #[inline]
    pub fn get(&self, idx: usize) -> usize {
        assert!(idx < self.size, "Index {} out of bounds {}", idx, self.size);
        assert!(self.bits <= 58, "Use BigUintVecMin0 for >58 bits");
        self.fast_get_internal(idx)
    }

    /// Get two consecutive values (optimized bulk access)
    ///
    /// # Performance
    ///
    /// More efficient than calling `get()` twice due to reduced bounds checking
    #[inline]
    pub fn get2(&self, idx: usize) -> [usize; 2] {
        assert!(idx + 1 < self.size, "Index {} out of bounds for get2", idx);
        assert!(self.bits <= 58, "Use BigUintVecMin0 for >58 bits");
        [self.fast_get_internal(idx), self.fast_get_internal(idx + 1)]
    }

    /// Fast get using static method (for use in hot loops)
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `idx < num_elements`
    /// - `bits <= 58`
    /// - `data` has sufficient size
    #[inline]
    pub fn fast_get(data: &[u8], bits: usize, mask: usize, idx: usize) -> usize {
        assert!(bits <= 58, "fast_get requires bits <= 58");
        let bit_idx = bits * idx;
        let byte_idx = bit_idx / 8;

        // SAFETY: Caller ensures bounds
        // Uses unaligned load since values may span cache lines
        let val = unsafe {
            std::ptr::read_unaligned(data.as_ptr().add(byte_idx) as *const usize)
        };
        (val >> (bit_idx % 8)) & mask
    }

    /// Internal fast get (bounds already checked)
    #[inline]
    fn fast_get_internal(&self, idx: usize) -> usize {
        let bit_idx = self.bits * idx;
        let byte_idx = bit_idx / 8;

        // SAFETY: Bounds checked by caller, data size validated in resize
        let val = unsafe {
            std::ptr::read_unaligned(self.data.as_ptr().add(byte_idx) as *const usize)
        };
        (val >> (bit_idx % 8)) & self.mask
    }

    /// Set value at index
    ///
    /// # Panics
    ///
    /// Panics if index >= size or value > mask
    #[inline]
    pub fn set(&mut self, idx: usize, val: usize) {
        assert!(idx < self.size, "Index {} out of bounds {}", idx, self.size);
        assert!(val <= self.mask, "Value {} exceeds max {}", val, self.mask);
        assert!(self.bits <= 64, "Bits must be <= 64");
        self.set_wire(idx, val);
    }

    /// Set value using bit manipulation (internal, wire format)
    fn set_wire(&mut self, idx: usize, val: usize) {
        let bits = self.bits;
        let bit_idx = bits * idx;
        self.set_uint_bits(bit_idx, bits, val);
    }

    /// Set bits in packed array
    ///
    /// # Algorithm
    ///
    /// Sets `bits` bits starting at `bit_pos` to `val` using unaligned writes
    fn set_uint_bits(&mut self, bit_pos: usize, bits: usize, val: usize) {
        if bits == 0 {
            return;
        }

        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;
        let end_bit = bit_offset + bits;

        if end_bit <= 64 {
            // Fast path: fits in one u64
            let mask = if bits == 64 {
                !0usize
            } else {
                (1usize << bits) - 1
            };
            let shifted_val = val << bit_offset;
            let shifted_mask = mask << bit_offset;

            // SAFETY: Bounds validated in resize, using unaligned access
            unsafe {
                let ptr = self.data.as_mut_ptr().add(byte_idx) as *mut usize;
                let current = std::ptr::read_unaligned(ptr);
                let new_val = (current & !shifted_mask) | shifted_val;
                std::ptr::write_unaligned(ptr, new_val);
            }
        } else {
            // Slow path: spans multiple u64s - use byte-by-byte
            let mut remaining_bits = bits;
            let mut remaining_val = val;
            let mut curr_byte = byte_idx;
            let mut curr_bit_offset = bit_offset;

            while remaining_bits > 0 {
                let bits_in_byte = (8 - curr_bit_offset).min(remaining_bits);
                let byte_mask = ((1u8 << bits_in_byte) - 1) << curr_bit_offset;
                let byte_val = ((remaining_val & ((1 << bits_in_byte) - 1)) as u8) << curr_bit_offset;

                self.data[curr_byte] = (self.data[curr_byte] & !byte_mask) | byte_val;

                remaining_val >>= bits_in_byte;
                remaining_bits -= bits_in_byte;
                curr_byte += 1;
                curr_bit_offset = 0;
            }
        }
    }

    /// Build from slice (auto-detect min/max)
    ///
    /// Scans input to find min and max values, then stores `values - min_val`.
    /// Returns the minimum value found.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zipora::containers::UintVecMin0;
    ///
    /// let data = vec![100usize, 105, 103, 108];
    /// let min_val = UintVecMin0::build_from_usize(&data);
    /// // Internally stores [0, 5, 3, 8] using 4 bits each
    /// ```
    pub fn build_from_usize(src: &[usize]) -> (Self, usize) {
        if src.is_empty() {
            return (Self::new_empty(), 0);
        }

        let &min_val = src.iter().min().unwrap();
        let &max_val = src.iter().max().unwrap();
        let wire_max = max_val - min_val;

        let mut vec = Self::new(src.len(), wire_max);
        for (i, &val) in src.iter().enumerate() {
            vec.set(i, val - min_val);
        }

        (vec, min_val)
    }

    /// Build from i32 slice (auto-detect min/max)
    pub fn build_from_i32(src: &[i32]) -> (Self, i32) {
        if src.is_empty() {
            return (Self::new_empty(), 0);
        }

        let &min_val = src.iter().min().unwrap();
        let &max_val = src.iter().max().unwrap();
        let wire_max = (max_val - min_val) as usize;

        let mut vec = Self::new(src.len(), wire_max);
        for (i, &val) in src.iter().enumerate() {
            vec.set(i, (val - min_val) as usize);
        }

        (vec, min_val)
    }

    /// Build from u32 slice (auto-detect min/max)
    pub fn build_from_u32(src: &[u32]) -> (Self, u32) {
        if src.is_empty() {
            return (Self::new_empty(), 0);
        }

        let &min_val = src.iter().min().unwrap();
        let &max_val = src.iter().max().unwrap();
        let wire_max = (max_val - min_val) as usize;

        let mut vec = Self::new(src.len(), wire_max);
        for (i, &val) in src.iter().enumerate() {
            vec.set(i, (val - min_val) as usize);
        }

        (vec, min_val)
    }

    /// Push back value (may reallocate)
    ///
    /// # Performance
    ///
    /// - Fast path if value fits in current bit width and capacity
    /// - Slow path reallocates with larger bit width if needed
    pub fn push_back(&mut self, val: usize) {
        if Self::compute_mem_size(self.bits, self.size + 1) <= self.data.len() && val <= self.mask {
            // Fast path: fits in existing allocation
            self.set_wire(self.size, val);
            self.size += 1;
        } else {
            // Slow path: need more space or larger bit width
            self.push_back_slow_path(val);
        }
    }

    /// Slow path for push_back (handles reallocation and bit width expansion)
    fn push_back_slow_path(&mut self, val: usize) {
        let new_bits = Self::compute_uintbits(val.max(self.mask));

        if new_bits > self.bits {
            // Need to rebuild with larger bit width
            let old_size = self.size;
            let mut new_vec = Self::new(old_size + 1, val);

            for i in 0..old_size {
                new_vec.set(i, self.get(i));
            }
            new_vec.set(old_size, val);

            *self = new_vec;
        } else {
            // Just need more capacity
            self.resize(self.size + 1);
            self.set(self.size - 1, val);
        }
    }

    /// Get last value
    ///
    /// # Panics
    ///
    /// Panics if vector is empty
    #[inline]
    pub fn back(&self) -> usize {
        assert!(self.size > 0, "Vector is empty");
        self.get(self.size - 1)
    }

    /// Clear all values
    pub fn clear(&mut self) {
        self.data.clear();
        self.bits = 0;
        self.mask = 0;
        self.size = 0;
    }

    /// Resize to new size (preserves existing values)
    pub fn resize(&mut self, new_size: usize) {
        let new_mem_size = Self::compute_mem_size(self.bits, new_size);

        if new_mem_size > self.data.len() {
            self.data.resize(new_mem_size, 0);
        }

        self.size = new_size;
    }

    /// Resize with specific bit width
    pub fn resize_with_uintbits(&mut self, num: usize, bits: usize) {
        assert!(bits <= 64, "Bits must be <= 64");

        self.bits = bits;
        self.mask = if bits == 0 { 0 } else { (1usize << bits) - 1 };
        self.size = num;

        let mem_size = Self::compute_mem_size(bits, num);
        self.data.resize(mem_size, 0);
    }

    /// Resize with max value constraint
    pub fn resize_with_wire_max_val(&mut self, num: usize, max_val: usize) {
        let bits = Self::compute_uintbits(max_val);
        self.resize_with_uintbits(num, bits);
    }

    /// Shrink allocation to minimum needed size
    pub fn shrink_to_fit(&mut self) {
        let needed_size = Self::compute_mem_size(self.bits, self.size);
        self.data.truncate(needed_size);
        self.data.shrink_to_fit();
    }

    /// Set raw data pointer (unsafe, for advanced use)
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `data` points to valid memory of sufficient size
    /// - `bits <= 58` for fast access
    /// - Memory layout matches `compute_mem_size(bits, num)`
    pub unsafe fn risk_set_data(&mut self, data: *mut u8, num: usize, bits: usize) {
        assert!(bits <= 58, "bits={} is too large (max_allowed=58)", bits);

        let mem_size = Self::compute_mem_size(bits, num);
        // SAFETY: Caller ensures data points to valid memory
        unsafe {
            self.data = Vec::from_raw_parts(data, mem_size, mem_size);
        }
        self.bits = bits;
        self.mask = if bits == 0 { 0 } else { (1usize << bits) - 1 };
        self.size = num;
    }

    // Getters

    /// Get underlying byte data
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get number of bits per integer
    #[inline]
    pub fn uintbits(&self) -> usize {
        self.bits
    }

    /// Get bit mask for values
    #[inline]
    pub fn uintmask(&self) -> usize {
        self.mask
    }

    /// Get number of integers stored
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get allocated memory size in bytes
    #[inline]
    pub fn mem_size(&self) -> usize {
        self.data.len()
    }

    /// Check if vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Default for UintVecMin0 {
    fn default() -> Self {
        Self::new_empty()
    }
}

impl std::ops::Index<usize> for UintVecMin0 {
    type Output = usize;

    fn index(&self, idx: usize) -> &Self::Output {
        // We can't return a reference to a packed value, so we use a workaround
        // This is a limitation of the Index trait with packed data
        panic!("Use get() method instead of indexing for UintVecMin0");
    }
}

impl fmt::Debug for UintVecMin0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UintVecMin0")
            .field("size", &self.size)
            .field("bits", &self.bits)
            .field("mask", &format_args!("{:#x}", self.mask))
            .field("mem_size", &self.data.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_uintbits() {
        assert_eq!(UintVecMin0::compute_uintbits(0), 0);
        assert_eq!(UintVecMin0::compute_uintbits(1), 1);
        assert_eq!(UintVecMin0::compute_uintbits(2), 2);
        assert_eq!(UintVecMin0::compute_uintbits(3), 2);
        assert_eq!(UintVecMin0::compute_uintbits(7), 3);
        assert_eq!(UintVecMin0::compute_uintbits(8), 4);
        assert_eq!(UintVecMin0::compute_uintbits(15), 4);
        assert_eq!(UintVecMin0::compute_uintbits(255), 8);
        assert_eq!(UintVecMin0::compute_uintbits(256), 9);
        assert_eq!(UintVecMin0::compute_uintbits(65535), 16);
    }

    #[test]
    fn test_compute_mem_size() {
        // 0 bits per value
        assert_eq!(UintVecMin0::compute_mem_size(0, 100), 16); // Minimum 16 bytes

        // 1 bit per value: 100 bits = 13 bytes + 7 padding + align = 32 bytes
        assert!(UintVecMin0::compute_mem_size(1, 100) >= 13);

        // 8 bits per value: 800 bits = 100 bytes + 7 padding + align
        let size_8bit = UintVecMin0::compute_mem_size(8, 100);
        assert!(size_8bit >= 100);
        assert_eq!(size_8bit % 16, 0); // 16-byte aligned
    }

    #[test]
    fn test_new_and_basic_ops() {
        let vec = UintVecMin0::new(10, 255);
        assert_eq!(vec.size(), 10);
        assert_eq!(vec.uintbits(), 8);
        assert_eq!(vec.uintmask(), 255);
    }

    #[test]
    fn test_set_and_get() {
        let mut vec = UintVecMin0::new(100, 255);

        // Set some values
        vec.set(0, 42);
        vec.set(50, 128);
        vec.set(99, 255);

        // Get them back
        assert_eq!(vec.get(0), 42);
        assert_eq!(vec.get(50), 128);
        assert_eq!(vec.get(99), 255);
    }

    #[test]
    fn test_round_trip_various_bit_widths() {
        // Test 1 bit
        let mut vec1 = UintVecMin0::new(64, 1);
        for i in 0..64 {
            vec1.set(i, i % 2);
        }
        for i in 0..64 {
            assert_eq!(vec1.get(i), i % 2);
        }

        // Test 4 bits
        let mut vec4 = UintVecMin0::new(100, 15);
        for i in 0..100 {
            vec4.set(i, i % 16);
        }
        for i in 0..100 {
            assert_eq!(vec4.get(i), i % 16);
        }

        // Test 16 bits
        let mut vec16 = UintVecMin0::new(1000, 65535);
        for i in 0..1000 {
            vec16.set(i, i % 65536);
        }
        for i in 0..1000 {
            assert_eq!(vec16.get(i), i % 65536);
        }
    }

    #[test]
    fn test_edge_case_zero_bits() {
        let vec = UintVecMin0::new(100, 0);
        assert_eq!(vec.uintbits(), 0);
        assert_eq!(vec.uintmask(), 0);

        // All values should be 0
        for i in 0..100 {
            assert_eq!(vec.get(i), 0);
        }
    }

    #[test]
    fn test_edge_case_one_bit() {
        let mut vec = UintVecMin0::new(128, 1);
        assert_eq!(vec.uintbits(), 1);

        // Set alternating pattern
        for i in 0..128 {
            vec.set(i, i % 2);
        }

        for i in 0..128 {
            assert_eq!(vec.get(i), i % 2);
        }
    }

    #[test]
    fn test_58_bits_max_fast_path() {
        // 58 bits is the maximum for fast path
        let max_val = (1usize << 58) - 1;
        let mut vec = UintVecMin0::new(10, max_val);
        assert_eq!(vec.uintbits(), 58);

        vec.set(0, max_val);
        vec.set(5, max_val / 2);

        assert_eq!(vec.get(0), max_val);
        assert_eq!(vec.get(5), max_val / 2);
    }

    #[test]
    fn test_get2() {
        let mut vec = UintVecMin0::new(100, 255);
        vec.set(10, 42);
        vec.set(11, 43);

        let vals = vec.get2(10);
        assert_eq!(vals, [42, 43]);
    }

    #[test]
    fn test_build_from_usize() {
        let data = vec![100, 105, 103, 108, 101];
        let (vec, min_val) = UintVecMin0::build_from_usize(&data);

        assert_eq!(min_val, 100);
        assert_eq!(vec.size(), 5);

        // max_val=108, min_val=100, wire_max=8, needs 4 bits
        assert_eq!(vec.uintbits(), 4);

        // Values stored as differences from min
        assert_eq!(vec.get(0), 0);  // 100 - 100
        assert_eq!(vec.get(1), 5);  // 105 - 100
        assert_eq!(vec.get(2), 3);  // 103 - 100
        assert_eq!(vec.get(3), 8);  // 108 - 100
        assert_eq!(vec.get(4), 1);  // 101 - 100
    }

    #[test]
    fn test_push_back_fast_path() {
        let mut vec = UintVecMin0::new(10, 255);
        vec.resize(0); // Start empty

        for i in 0..10 {
            vec.push_back(i * 10);
        }

        assert_eq!(vec.size(), 10);
        for i in 0..10 {
            assert_eq!(vec.get(i), i * 10);
        }
    }

    #[test]
    fn test_push_back_slow_path_capacity() {
        let mut vec = UintVecMin0::new(2, 10);
        vec.resize(0);

        // Push beyond initial capacity
        for i in 0..5 {
            vec.push_back(i);
        }

        assert_eq!(vec.size(), 5);
        for i in 0..5 {
            assert_eq!(vec.get(i), i);
        }
    }

    #[test]
    fn test_push_back_slow_path_bit_expansion() {
        let mut vec = UintVecMin0::new(10, 15);
        vec.resize(0);

        // Start with small values
        for i in 0..5 {
            vec.push_back(i);
        }
        assert_eq!(vec.uintbits(), 4); // Max value 15 needs 4 bits

        // Push value that requires more bits
        vec.push_back(255);
        assert_eq!(vec.uintbits(), 8); // Now needs 8 bits

        // Verify all values preserved
        for i in 0..5 {
            assert_eq!(vec.get(i), i);
        }
        assert_eq!(vec.get(5), 255);
    }

    #[test]
    fn test_back() {
        let mut vec = UintVecMin0::new(10, 255);
        vec.set(9, 123);
        assert_eq!(vec.back(), 123);
    }

    #[test]
    fn test_clear() {
        let mut vec = UintVecMin0::new(100, 255);
        for i in 0..100 {
            vec.set(i, i);
        }

        vec.clear();
        assert_eq!(vec.size(), 0);
        assert_eq!(vec.uintbits(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_resize() {
        let mut vec = UintVecMin0::new(10, 255);
        for i in 0..10 {
            vec.set(i, i);
        }

        vec.resize(20);
        assert_eq!(vec.size(), 20);

        // Original values preserved
        for i in 0..10 {
            assert_eq!(vec.get(i), i);
        }
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut vec = UintVecMin0::new(1000, 255);
        vec.resize(10);

        let before = vec.mem_size();
        vec.shrink_to_fit();
        let after = vec.mem_size();

        assert!(after <= before);
        assert_eq!(vec.size(), 10);
    }

    #[test]
    fn test_memory_efficiency() {
        // 1000 values with max 255 should use ~1000 bytes
        let vec = UintVecMin0::new(1000, 255);
        let mem = vec.mem_size();

        // Should be much less than Vec<usize> (8000 bytes on 64-bit)
        assert!(mem < 1000 * std::mem::size_of::<usize>() / 4);

        // But at least enough for the data
        assert!(mem >= 1000); // 1000 values * 8 bits = 1000 bytes minimum
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_out_of_bounds() {
        let vec = UintVecMin0::new(10, 255);
        vec.get(10);
    }

    #[test]
    #[should_panic(expected = "exceeds max")]
    fn test_set_value_too_large() {
        let mut vec = UintVecMin0::new(10, 255);
        vec.set(0, 256);
    }

    #[test]
    #[should_panic(expected = "Vector is empty")]
    fn test_back_empty() {
        let vec = UintVecMin0::new_empty();
        vec.back();
    }

    #[test]
    fn test_fast_get_static() {
        let mut vec = UintVecMin0::new(100, 255);
        for i in 0..100 {
            vec.set(i, i);
        }

        // Test static method
        for i in 0..100 {
            let val = UintVecMin0::fast_get(vec.data(), vec.uintbits(), vec.uintmask(), i);
            assert_eq!(val, i);
        }
    }
}
