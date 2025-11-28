//! Compressed integer vector storing values as `min_val + UintVecMin0`
//!
//! # Overview
//!
//! `ZipIntVec` is a thin wrapper over `UintVecMin0` that automatically determines
//! bit width from the range [min_val, max_val]. This provides optimal space
//! efficiency for integer sequences with non-zero minimum values.
//!
//! # Memory Layout
//!
//! Internally stores `value - min_val` for each element, allowing the underlying
//! `UintVecMin0` to use minimal bits based on the value range rather than absolute values.
//!
//! # Example
//!
//! ```rust
//! use zipora::containers::ZipIntVec;
//!
//! // Store values in range [1000, 1255] using only 8 bits each
//! let mut vec = ZipIntVec::new(100, 1000, 1255);
//! for i in 0..100 {
//!     vec.set(i, 1000 + i % 256);
//! }
//!
//! assert_eq!(vec.get(42), 1042);
//! assert_eq!(vec.uintbits(), 8); // Only 8 bits despite values > 1000
//! ```

use super::uint_vec_min0::UintVecMin0;
use std::fmt;

/// Compressed integer vector with automatic range-based compression
///
/// Stores integers as `min_val + offset` where offsets are stored in `UintVecMin0`.
/// This allows efficient storage of sequences with large absolute values but small ranges.
#[derive(Clone)]
pub struct ZipIntVec {
    /// Underlying compressed storage (stores value - min_val)
    inner: UintVecMin0,
    /// Minimum value in the sequence
    min_val: usize,
}

impl ZipIntVec {
    /// Create new compressed vector for range [min_val, max_val]
    ///
    /// # Arguments
    ///
    /// * `num` - Number of integers to allocate space for
    /// * `min_val` - Minimum value that will be stored
    /// * `max_val` - Maximum value that will be stored
    ///
    /// # Panics
    ///
    /// Panics if `min_val >= max_val`
    ///
    /// # Example
    ///
    /// ```rust
    /// use zipora::containers::ZipIntVec;
    ///
    /// // Store timestamps around epoch 1700000000 efficiently
    /// let vec = ZipIntVec::new(1000, 1700000000, 1700001000);
    /// // Uses only 10 bits per value despite large absolute values
    /// assert_eq!(vec.uintbits(), 10);
    /// ```
    pub fn new(num: usize, min_val: usize, max_val: usize) -> Self {
        assert!(min_val < max_val, "min_val must be less than max_val");
        let wire_max = max_val - min_val;
        Self {
            inner: UintVecMin0::new(num, wire_max),
            min_val,
        }
    }

    /// Create empty vector
    pub fn new_empty() -> Self {
        Self {
            inner: UintVecMin0::new_empty(),
            min_val: 0,
        }
    }

    /// Get value at index
    ///
    /// # Panics
    ///
    /// Panics if index >= size
    #[inline]
    pub fn get(&self, idx: usize) -> usize {
        self.min_val + self.inner.get(idx)
    }

    /// Get two consecutive values (optimized bulk access)
    #[inline]
    pub fn get2(&self, idx: usize) -> [usize; 2] {
        let offsets = self.inner.get2(idx);
        [self.min_val + offsets[0], self.min_val + offsets[1]]
    }

    /// Fast static get method for hot loops
    ///
    /// # Arguments
    ///
    /// * `data` - Packed bit data
    /// * `bits` - Bits per value
    /// * `mask` - Value mask
    /// * `min_val` - Minimum value offset
    /// * `idx` - Index to retrieve
    ///
    /// SAFETY FIX (v2.1.1): Now returns Result for bounds checking
    #[inline]
    pub fn fast_get(data: &[u8], bits: usize, mask: usize, min_val: usize, idx: usize) -> crate::error::Result<usize> {
        Ok(min_val + UintVecMin0::fast_get(data, bits, mask, idx)?)
    }

    /// Set value at index
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - index >= size
    /// - value < min_val
    /// - value > min_val + uintmask
    #[inline]
    pub fn set(&mut self, idx: usize, val: usize) {
        assert!(val >= self.min_val, "Value {} below minimum {}", val, self.min_val);
        let max_val = self.min_val + self.inner.uintmask();
        assert!(val <= max_val, "Value {} exceeds maximum {}", val, max_val);
        self.inner.set(idx, val - self.min_val);
    }

    /// Build from usize slice (auto-detect min/max)
    ///
    /// Scans input to find min and max values, then creates optimally sized vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zipora::containers::ZipIntVec;
    ///
    /// let data = vec![1000, 1005, 1003, 1008];
    /// let vec = ZipIntVec::build_from_usize(&data);
    /// assert_eq!(vec.min_val(), 1000);
    /// assert_eq!(vec.uintbits(), 4); // Range 0-8 needs 4 bits
    /// ```
    pub fn build_from_usize(src: &[usize]) -> Self {
        if src.is_empty() {
            return Self::new_empty();
        }

        let &min_val = src.iter().min().unwrap();
        let &max_val = src.iter().max().unwrap();

        if min_val == max_val {
            // All values are the same
            let mut vec = Self::new(src.len(), min_val, min_val + 1);
            for i in 0..src.len() {
                vec.set(i, min_val);
            }
            return vec;
        }

        let mut vec = Self::new(src.len(), min_val, max_val);
        for (i, &val) in src.iter().enumerate() {
            vec.set(i, val);
        }

        vec
    }

    /// Build from u32 slice (auto-detect min/max)
    pub fn build_from_u32(src: &[u32]) -> Self {
        if src.is_empty() {
            return Self::new_empty();
        }

        let &min_val = src.iter().min().unwrap();
        let &max_val = src.iter().max().unwrap();

        if min_val == max_val {
            // All values are the same
            let mut vec = Self::new(src.len(), min_val as usize, (min_val + 1) as usize);
            for i in 0..src.len() {
                vec.set(i, min_val as usize);
            }
            return vec;
        }

        let mut vec = Self::new(src.len(), min_val as usize, max_val as usize);
        for (i, &val) in src.iter().enumerate() {
            vec.set(i, val as usize);
        }

        vec
    }

    /// Push back value (may reallocate)
    pub fn push_back(&mut self, val: usize) {
        assert!(val >= self.min_val, "Value {} below minimum {}", val, self.min_val);
        self.inner.push_back(val - self.min_val);
    }

    /// Get last value
    ///
    /// # Panics
    ///
    /// Panics if vector is empty
    #[inline]
    pub fn back(&self) -> usize {
        self.min_val + self.inner.back()
    }

    /// Clear all values
    pub fn clear(&mut self) {
        self.inner.clear();
        self.min_val = 0;
    }

    /// Resize to new size (preserves existing values)
    pub fn resize(&mut self, new_size: usize) {
        self.inner.resize(new_size);
    }

    /// Resize with new value range
    pub fn resize_with_range(&mut self, num: usize, min_val: usize, max_val: usize) {
        assert!(min_val < max_val, "min_val must be less than max_val");
        let wire_max = max_val - min_val;
        self.inner.resize_with_wire_max_val(num, wire_max);
        self.min_val = min_val;
    }

    /// Shrink allocation to minimum needed size
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    /// Set raw data pointer (unsafe, for advanced use)
    ///
    /// # Safety
    ///
    /// Caller must ensure all safety requirements of `UintVecMin0::risk_set_data`
    pub unsafe fn risk_set_data(&mut self, data: *mut u8, num: usize, min_val: usize, bits: usize) {
        // SAFETY: Caller ensures all safety requirements
        unsafe {
            self.inner.risk_set_data(data, num, bits);
        }
        self.min_val = min_val;
    }

    /// Swap contents with another vector
    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.inner, &mut other.inner);
        std::mem::swap(&mut self.min_val, &mut other.min_val);
    }

    // Getters

    /// Get minimum value
    #[inline]
    pub fn min_val(&self) -> usize {
        self.min_val
    }

    /// Get maximum value that can be stored
    #[inline]
    pub fn max_val(&self) -> usize {
        self.min_val + self.inner.uintmask()
    }

    /// Get underlying byte data
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.inner.data()
    }

    /// Get number of bits per integer
    #[inline]
    pub fn uintbits(&self) -> usize {
        self.inner.uintbits()
    }

    /// Get bit mask for offset values
    #[inline]
    pub fn uintmask(&self) -> usize {
        self.inner.uintmask()
    }

    /// Get number of integers stored
    #[inline]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    /// Get allocated memory size in bytes
    #[inline]
    pub fn mem_size(&self) -> usize {
        self.inner.mem_size() + std::mem::size_of::<usize>()
    }

    /// Check if vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get reference to inner UintVecMin0
    #[inline]
    pub fn inner(&self) -> &UintVecMin0 {
        &self.inner
    }
}

impl Default for ZipIntVec {
    fn default() -> Self {
        Self::new_empty()
    }
}

impl fmt::Debug for ZipIntVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZipIntVec")
            .field("size", &self.size())
            .field("min_val", &self.min_val)
            .field("max_val", &self.max_val())
            .field("bits", &self.uintbits())
            .field("mem_size", &self.mem_size())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_basic_ops() {
        let vec = ZipIntVec::new(10, 1000, 1255);
        assert_eq!(vec.size(), 10);
        assert_eq!(vec.min_val(), 1000);
        assert_eq!(vec.max_val(), 1255);
        assert_eq!(vec.uintbits(), 8); // 255 needs 8 bits
    }

    #[test]
    fn test_set_and_get() {
        let mut vec = ZipIntVec::new(100, 1000, 1255);

        vec.set(0, 1042);
        vec.set(50, 1128);
        vec.set(99, 1255);

        assert_eq!(vec.get(0), 1042);
        assert_eq!(vec.get(50), 1128);
        assert_eq!(vec.get(99), 1255);
    }

    #[test]
    fn test_round_trip_large_values() {
        let min = 1_000_000;
        let max = 1_001_000;
        let mut vec = ZipIntVec::new(1000, min, max);

        for i in 0..1000 {
            let val = min + (i % 1001);
            vec.set(i, val);
        }

        for i in 0..1000 {
            let expected = min + (i % 1001);
            assert_eq!(vec.get(i), expected);
        }
    }

    #[test]
    fn test_get2() {
        let mut vec = ZipIntVec::new(100, 1000, 1255);
        vec.set(10, 1042);
        vec.set(11, 1043);

        let vals = vec.get2(10);
        assert_eq!(vals, [1042, 1043]);
    }

    #[test]
    fn test_build_from() {
        let data = vec![1000, 1005, 1003, 1008, 1001];
        let vec = ZipIntVec::build_from_usize(&data);

        assert_eq!(vec.min_val(), 1000);
        assert_eq!(vec.size(), 5);
        assert_eq!(vec.uintbits(), 4); // Range 0-8 needs 4 bits
        // max_val() returns min + mask, not necessarily the actual max in data
        // With 4 bits, mask = 15, so max_val() = 1000 + 15 = 1015
        assert_eq!(vec.max_val(), 1015);

        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(vec.get(i), expected);
        }
    }

    #[test]
    fn test_build_from_all_same() {
        let data = vec![42, 42, 42, 42];
        let vec = ZipIntVec::build_from_usize(&data);

        assert_eq!(vec.min_val(), 42);
        assert_eq!(vec.size(), 4);

        for i in 0..4 {
            assert_eq!(vec.get(i), 42);
        }
    }

    #[test]
    fn test_push_back() {
        let mut vec = ZipIntVec::new(10, 1000, 1255);
        vec.resize(0); // Start empty

        for i in 0..10 {
            vec.push_back(1000 + i * 10);
        }

        assert_eq!(vec.size(), 10);
        for i in 0..10 {
            assert_eq!(vec.get(i), 1000 + i * 10);
        }
    }

    #[test]
    fn test_back() {
        let mut vec = ZipIntVec::new(10, 1000, 1255);
        vec.set(9, 1123);
        assert_eq!(vec.back(), 1123);
    }

    #[test]
    fn test_clear() {
        let mut vec = ZipIntVec::new(100, 1000, 1255);
        for i in 0..100 {
            vec.set(i, 1000 + i);
        }

        vec.clear();
        assert_eq!(vec.size(), 0);
        assert_eq!(vec.min_val(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_resize() {
        let mut vec = ZipIntVec::new(10, 1000, 1255);
        for i in 0..10 {
            vec.set(i, 1000 + i);
        }

        vec.resize(20);
        assert_eq!(vec.size(), 20);

        // Original values preserved
        for i in 0..10 {
            assert_eq!(vec.get(i), 1000 + i);
        }
    }

    #[test]
    fn test_resize_with_range() {
        let mut vec = ZipIntVec::new(10, 1000, 1255);
        vec.resize_with_range(20, 2000, 2511);

        assert_eq!(vec.size(), 20);
        assert_eq!(vec.min_val(), 2000);
        assert_eq!(vec.max_val(), 2511);
    }

    #[test]
    fn test_swap() {
        let mut vec1 = ZipIntVec::new(10, 1000, 1255);
        vec1.set(0, 1042);

        let mut vec2 = ZipIntVec::new(10, 2000, 2255);
        vec2.set(0, 2042);

        vec1.swap(&mut vec2);

        assert_eq!(vec1.min_val(), 2000);
        assert_eq!(vec1.get(0), 2042);
        assert_eq!(vec2.min_val(), 1000);
        assert_eq!(vec2.get(0), 1042);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut vec = ZipIntVec::new(1000, 1000, 1255);
        vec.resize(10);

        let before = vec.mem_size();
        vec.shrink_to_fit();
        let after = vec.mem_size();

        assert!(after <= before);
        assert_eq!(vec.size(), 10);
    }

    #[test]
    fn test_memory_efficiency() {
        // 1000 values in range [1000000, 1000255] should use ~1000 bytes
        let vec = ZipIntVec::new(1000, 1000000, 1000255);
        let mem = vec.mem_size();

        // Should be much less than Vec<usize> (8000 bytes on 64-bit)
        assert!(mem < 1000 * std::mem::size_of::<usize>() / 4);

        // But at least enough for the data
        assert!(mem >= 1000); // 1000 values * 8 bits = 1000 bytes minimum
    }

    #[test]
    fn test_fast_get_static() {
        let mut vec = ZipIntVec::new(100, 1000, 1255);
        for i in 0..100 {
            vec.set(i, 1000 + i);
        }

        // Test static method
        for i in 0..100 {
            let val = ZipIntVec::fast_get(
                vec.data(),
                vec.uintbits(),
                vec.uintmask(),
                vec.min_val(),
                i,
            );
            assert_eq!(val.expect("fast_get should succeed"), 1000 + i);
        }
    }

    #[test]
    fn test_zero_range() {
        // Edge case: min_val at zero
        let mut vec = ZipIntVec::new(10, 0, 255);
        vec.set(0, 0);
        vec.set(5, 128);
        vec.set(9, 255);

        assert_eq!(vec.get(0), 0);
        assert_eq!(vec.get(5), 128);
        assert_eq!(vec.get(9), 255);
    }

    #[test]
    fn test_single_bit_range() {
        // Range [100, 101] needs 1 bit
        let mut vec = ZipIntVec::new(64, 100, 101);
        assert_eq!(vec.uintbits(), 1);

        for i in 0..64 {
            vec.set(i, 100 + (i % 2));
        }

        for i in 0..64 {
            assert_eq!(vec.get(i), 100 + (i % 2));
        }
    }

    #[test]
    #[should_panic(expected = "must be less than max_val")]
    fn test_new_invalid_range() {
        ZipIntVec::new(10, 1000, 1000);
    }

    #[test]
    #[should_panic(expected = "below minimum")]
    fn test_set_below_min() {
        let mut vec = ZipIntVec::new(10, 1000, 1255);
        vec.set(0, 999);
    }

    #[test]
    #[should_panic(expected = "exceeds maximum")]
    fn test_set_above_max() {
        let mut vec = ZipIntVec::new(10, 1000, 1255);
        vec.set(0, 1256);
    }

    #[test]
    #[should_panic(expected = "Vector is empty")]
    fn test_back_empty() {
        let vec = ZipIntVec::new_empty();
        vec.back();
    }

    #[test]
    fn test_large_range_efficiency() {
        // Large absolute values but small range
        let min = 1_000_000_000;
        let max = 1_000_001_023; // Range of 1024, needs 10 bits
        let vec = ZipIntVec::new(1000, min, max);

        assert_eq!(vec.uintbits(), 10);
        assert_eq!(vec.min_val(), min);
        assert_eq!(vec.max_val(), max);

        // Memory should be ~1250 bytes, not 8000 bytes
        let mem = vec.mem_size();
        assert!(mem < 1000 * std::mem::size_of::<usize>() / 4);
    }

    #[test]
    fn test_build_from_u32() {
        let data: Vec<u32> = vec![100, 105, 103, 108, 101];
        let vec = ZipIntVec::build_from_u32(&data);

        assert_eq!(vec.min_val(), 100);
        assert_eq!(vec.size(), 5);

        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(vec.get(i), expected as usize);
        }
    }

    #[test]
    fn test_timestamp_compression() {
        // Realistic use case: compress timestamps
        let base_timestamp = 1700000000; // Unix timestamp
        let mut timestamps = Vec::new();
        for i in 0..1000 {
            timestamps.push(base_timestamp + i * 60); // 1-minute intervals
        }

        let vec = ZipIntVec::build_from_usize(&timestamps);

        // Should use ~16 bits per timestamp (range 0-59940)
        assert!(vec.uintbits() <= 16);

        // Verify all timestamps
        for (i, &expected) in timestamps.iter().enumerate() {
            assert_eq!(vec.get(i), expected);
        }

        // Memory should be much less than Vec<usize>
        let compression_ratio = (1000 * std::mem::size_of::<usize>()) as f64 / vec.mem_size() as f64;
        assert!(compression_ratio > 3.0); // At least 3x compression
    }
}
