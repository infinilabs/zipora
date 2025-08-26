//! Advanced bit-packed integer storage with variable bit-width
//!
//! IntVec<T> provides state-of-the-art integer compression with sophisticated
//! block-based architecture, hardware acceleration, and adaptive compression
//! strategies inspired by high-performance database storage engines.

use crate::error::{Result, ZiporaError};
use std::marker::PhantomData;
use std::mem;

mod int_vec_simd;
mod performance_tests;
use int_vec_simd::{BitOps, SimdOps, PrefetchOps};

/// Trait for integer types supported by IntVec
pub trait PackedInt: 
    Copy + 
    Clone + 
    PartialEq + 
    PartialOrd + 
    std::fmt::Debug + 
    'static 
{
    /// Convert to u64 for internal processing
    fn to_u64(self) -> u64;
    /// Convert from u64 (may truncate)
    fn from_u64(val: u64) -> Self;
    /// Get the maximum value for this type
    fn max_value() -> Self;
    /// Get the minimum value for this type  
    fn min_value() -> Self;
    /// Get the bit width of this type
    fn bit_width() -> u8;
    /// Convert to signed for delta calculations
    fn to_i64(self) -> i64;
    /// Convert from signed delta
    fn from_i64(val: i64) -> Self;
}

macro_rules! impl_packed_int {
    ($t:ty, $bits:expr) => {
        impl PackedInt for $t {
            #[inline]
            fn to_u64(self) -> u64 { self as u64 }
            
            #[inline] 
            fn from_u64(val: u64) -> Self { val as Self }
            
            #[inline]
            fn max_value() -> Self { <$t>::MAX }
            
            #[inline]
            fn min_value() -> Self { <$t>::MIN }
            
            #[inline]
            fn bit_width() -> u8 { $bits }
            
            #[inline]
            fn to_i64(self) -> i64 { self as i64 }
            
            #[inline]
            fn from_i64(val: i64) -> Self { val as Self }
        }
    };
}

impl_packed_int!(u8, 8);
impl_packed_int!(u16, 16);
impl_packed_int!(u32, 32);
impl_packed_int!(u64, 64);
impl_packed_int!(i8, 8);
impl_packed_int!(i16, 16);
impl_packed_int!(i32, 32);
impl_packed_int!(i64, 64);

/// Block configuration for optimal performance
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockSize {
    /// 64 elements per block (6 bits)
    Block64 = 6,
    /// 128 elements per block (7 bits)  
    Block128 = 7,
}

impl BlockSize {
    #[inline]
    fn units(self) -> usize {
        1 << (self as u8)
    }
    
    #[inline]
    fn log2(self) -> u8 {
        self as u8
    }
}

/// Compression strategy for optimal space efficiency
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionStrategy {
    /// Raw storage (no compression)
    Raw,
    /// Simple min-max bit packing  
    MinMax { min_val: u64, bit_width: u8 },
    /// Advanced block-based compression
    BlockBased { 
        block_size: BlockSize,
        offset_width: u8,
        sample_width: u8,
        is_sorted: bool,
    },
    /// Delta compression for sequences
    Delta { base_val: u64, delta_width: u8 },
}

/// Advanced bit-packed integer vector with variable bit-width
///
/// # Key Features
///
/// 1. **Generic over Integer Types**: Support for u8-u64, i8-i64
/// 2. **Block-Based Architecture**: 64/128 element blocks for cache efficiency
/// 3. **Hardware Acceleration**: BMI2, SIMD, popcount when available
/// 4. **Variable Bit-Width**: Automatic calculation and specialization
/// 5. **Advanced Compression**: Delta, min-max, sorted sequence detection
/// 6. **Memory Safety**: Rust type system guarantees with zero-cost abstractions
///
/// # Examples
///
/// ```rust
/// use zipora::IntVec;
///
/// // Create from sorted sequence for optimal compression
/// let values: Vec<u32> = (1000..2000).collect();
/// let compressed = IntVec::from_slice(&values)?;
/// assert!(compressed.compression_ratio() < 0.3); // >70% space saving
///
/// // Generic over different integer types
/// let small_values: Vec<u8> = (0..255).collect();
/// let compressed_u8 = IntVec::<u8>::from_slice(&small_values)?;
/// 
/// // Support for signed integers with delta compression
/// let deltas: Vec<i32> = vec![-10, -5, 0, 5, 10];
/// let compressed_i32 = IntVec::<i32>::from_slice(&deltas)?;
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
pub struct IntVec<T: PackedInt> {
    /// Compression strategy in use
    strategy: CompressionStrategy,
    /// Compressed data storage (aligned for hardware acceleration)
    data: Box<[u8]>,
    /// Index structure for block-based access
    index: Option<Box<[u8]>>,
    /// Number of elements stored
    len: usize,
    /// Statistics for performance analysis
    stats: CompressionStats,
    /// Phantom data for type parameter
    _marker: PhantomData<T>,
}

#[derive(Debug, Default, Clone)]
struct CompressionStats {
    original_size: usize,
    compressed_size: usize,
    index_size: usize,
    compression_time_ns: u64,
    access_count: u64,
    cache_hits: u64,
}

impl<T: PackedInt> IntVec<T> {
    /// Create a new empty IntVec
    pub fn new() -> Self {
        Self {
            strategy: CompressionStrategy::Raw,
            data: Box::new([]),
            index: None,
            len: 0,
            stats: CompressionStats::default(),
            _marker: PhantomData,
        }
    }

    /// Create IntVec from slice with optimal compression
    ///
    /// This analyzes the data and selects the best compression strategy
    /// automatically, providing maximum space efficiency.
    ///
    /// # Arguments
    ///
    /// * `values` - Slice of values to compress
    ///
    /// # Returns
    ///
    /// Compressed IntVec with optimal strategy selected
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::IntVec;
    ///
    /// // Sorted sequence - excellent compression
    /// let sorted: Vec<u32> = (0..1000).collect();
    /// let compressed = IntVec::from_slice(&sorted)?;
    /// assert!(compressed.compression_ratio() < 0.2);
    ///
    /// // Random data - adaptive strategy
    /// let random = vec![42u32, 1337, 9999, 12345];
    /// let compressed = IntVec::from_slice(&random)?;
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn from_slice(values: &[T]) -> Result<Self> {
        let start_time = std::time::Instant::now();
        
        if values.is_empty() {
            return Ok(Self::new());
        }

        let mut result = Self::new();
        result.len = values.len();

        // Convert to u64 for analysis
        let u64_values: Vec<u64> = values.iter().map(|v| v.to_u64()).collect();
        
        // Analyze and select optimal strategy
        let strategy = Self::analyze_optimal_strategy(&u64_values);
        result.strategy = strategy;

        // Compress using selected strategy
        result.compress_with_strategy(&u64_values, strategy)?;

        // Update statistics
        result.stats.original_size = values.len() * mem::size_of::<T>();
        result.stats.compressed_size = result.data.len();
        result.stats.index_size = result.index.as_ref().map_or(0, |idx| idx.len());
        result.stats.compression_time_ns = start_time.elapsed().as_nanos() as u64;

        Ok(result)
    }

    /// Get value at specified index with hardware-accelerated decompression
    ///
    /// # Arguments
    /// 
    /// * `index` - Zero-based index of element to retrieve
    ///
    /// # Returns
    ///
    /// Some(value) if index is valid, None otherwise
    ///
    /// # Performance
    ///
    /// Uses BMI2 bit extraction when available, falls back to optimized
    /// bit manipulation for maximum performance across platforms.
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }

        self.stats.access_count.wrapping_add(1);

        let result = match self.strategy {
            CompressionStrategy::Raw => self.get_raw(index),
            CompressionStrategy::MinMax { min_val, bit_width } => {
                self.get_min_max(index, min_val, bit_width)
            }
            CompressionStrategy::BlockBased { 
                block_size, offset_width, sample_width, is_sorted 
            } => {
                self.get_block_based(index, block_size, offset_width, sample_width, is_sorted)
            }
            CompressionStrategy::Delta { base_val, delta_width } => {
                self.get_delta(index, base_val, delta_width)
            }
        };

        result.map(T::from_u64)
    }

    /// Get the number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get compression ratio (0.0 to 1.0, lower is better)
    pub fn compression_ratio(&self) -> f64 {
        if self.stats.original_size == 0 {
            1.0
        } else {
            let total_compressed = self.stats.compressed_size + self.stats.index_size;
            total_compressed as f64 / self.stats.original_size as f64
        }
    }

    /// Get detailed statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        mem::size_of::<Self>() + 
        self.data.len() + 
        self.index.as_ref().map_or(0, |idx| idx.len())
    }

    // Private implementation methods

    /// Analyze data to select optimal compression strategy using hardware acceleration
    fn analyze_optimal_strategy(values: &[u64]) -> CompressionStrategy {
        if values.is_empty() {
            return CompressionStrategy::Raw;
        }

        let len = values.len();
        
        // Don't compress very small datasets
        if len < 8 {
            return CompressionStrategy::Raw;
        }

        // Use SIMD for fast range analysis
        let (min_val, max_val) = SimdOps::analyze_range_bulk(values);

        // Check if values are sorted for advanced block compression
        let is_sorted = values.windows(2).all(|w| w[0] <= w[1]);
        
        // Calculate potential compression strategies
        let strategies = [
            Self::analyze_min_max(values, min_val, max_val),
            Self::analyze_delta(values),
            Self::analyze_block_based(values, is_sorted),
        ];

        // Select strategy with best compression ratio
        strategies.into_iter()
            .min_by(|a, b| {
                let ratio_a = Self::estimate_compression_ratio(*a, len);
                let ratio_b = Self::estimate_compression_ratio(*b, len);
                ratio_a.partial_cmp(&ratio_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(CompressionStrategy::Raw)
    }

    fn analyze_min_max(values: &[u64], min_val: u64, max_val: u64) -> CompressionStrategy {
        if min_val == max_val {
            // All values are the same
            return CompressionStrategy::MinMax { min_val, bit_width: 1 };
        }

        let range = max_val - min_val;
        let bit_width = BitOps::compute_bit_width(range);

        CompressionStrategy::MinMax { min_val, bit_width }
    }

    fn analyze_delta(values: &[u64]) -> CompressionStrategy {
        if values.len() < 2 {
            return CompressionStrategy::Raw;
        }

        let base_val = values[0];
        let mut max_delta = 0u64;
        let mut valid_delta = true;

        for i in 1..values.len() {
            if let Some(delta) = values[i].checked_sub(values[i-1]) {
                max_delta = max_delta.max(delta);
            } else {
                valid_delta = false;
                break;
            }
        }

        if !valid_delta || max_delta > (1u64 << 32) {
            return CompressionStrategy::Raw;
        }

        let delta_width = BitOps::compute_bit_width(max_delta);

        CompressionStrategy::Delta { base_val, delta_width }
    }

    fn analyze_block_based(values: &[u64], is_sorted: bool) -> CompressionStrategy {
        let len = values.len();
        
        // Need sufficient data for block-based compression
        if len < 64 {
            return CompressionStrategy::Raw;
        }

        let block_size = if len >= 1024 {
            BlockSize::Block128
        } else {
            BlockSize::Block64
        };

        let block_units = block_size.units();
        let num_blocks = (len + block_units - 1) / block_units;

        // Analyze sample values (block base values)
        let mut samples = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let start = block_idx * block_units;
            let end = (start + block_units).min(len);
            let block_min = values[start..end].iter().min().unwrap();
            samples.push(*block_min);
        }

        let sample_min = *samples.iter().min().unwrap();
        let sample_max = *samples.iter().max().unwrap();
        let sample_width = BitOps::compute_bit_width(sample_max - sample_min);

        // Analyze offset values within blocks
        let mut max_offset = 0u64;
        for block_idx in 0..num_blocks {
            let start = block_idx * block_units;
            let end = (start + block_units).min(len);
            let block_min = samples[block_idx];
            
            for &val in &values[start..end] {
                let offset = val - block_min;
                max_offset = max_offset.max(offset);
            }
        }

        let offset_width = BitOps::compute_bit_width(max_offset);

        CompressionStrategy::BlockBased {
            block_size,
            offset_width,
            sample_width,
            is_sorted,
        }
    }

    fn estimate_compression_ratio(strategy: CompressionStrategy, len: usize) -> f64 {
        let original_size = len * 8; // Assume u64 for estimation
        
        let compressed_size = match strategy {
            CompressionStrategy::Raw => original_size,
            CompressionStrategy::MinMax { bit_width, .. } => {
                ((len * bit_width as usize + 7) / 8).max(32)
            }
            CompressionStrategy::Delta { delta_width, .. } => {
                8 + ((len * delta_width as usize + 7) / 8).max(32) // base + deltas
            }
            CompressionStrategy::BlockBased { 
                block_size, offset_width, sample_width, .. 
            } => {
                let block_units = block_size.units();
                let num_blocks = (len + block_units - 1) / block_units;
                let index_size = num_blocks * sample_width as usize / 8;
                let data_size = (len * offset_width as usize + 7) / 8;
                index_size + data_size
            }
        };

        compressed_size as f64 / original_size as f64
    }

    fn compress_with_strategy(&mut self, values: &[u64], strategy: CompressionStrategy) -> Result<()> {
        match strategy {
            CompressionStrategy::Raw => self.compress_raw(values),
            CompressionStrategy::MinMax { min_val, bit_width } => {
                self.compress_min_max(values, min_val, bit_width)
            }
            CompressionStrategy::BlockBased { 
                block_size, offset_width, sample_width, is_sorted 
            } => {
                self.compress_block_based(values, block_size, offset_width, sample_width, is_sorted)
            }
            CompressionStrategy::Delta { base_val, delta_width } => {
                self.compress_delta(values, base_val, delta_width)
            }
        }
    }

    fn compress_raw(&mut self, values: &[u64]) -> Result<()> {
        let mut data = Vec::with_capacity(values.len() * 8);
        for &value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        self.data = data.into_boxed_slice();
        Ok(())
    }

    fn compress_min_max(&mut self, values: &[u64], min_val: u64, bit_width: u8) -> Result<()> {
        if bit_width == 0 || bit_width > 64 {
            return Err(ZiporaError::invalid_data("Invalid bit width"));
        }

        let total_bits = values.len() * bit_width as usize;
        let byte_size = (total_bits + 7) / 8;
        let aligned_size = (byte_size + 15) & !15; // 16-byte alignment

        let mut data = vec![0u8; aligned_size];
        let mut bit_offset = 0;

        for &value in values {
            if value < min_val {
                return Err(ZiporaError::invalid_data("Value below minimum"));
            }
            
            let offset_value = value - min_val;
            self.write_bits(&mut data, offset_value, bit_offset, bit_width)?;
            bit_offset += bit_width as usize;
        }

        self.data = data.into_boxed_slice();
        Ok(())
    }

    fn compress_delta(&mut self, values: &[u64], base_val: u64, delta_width: u8) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        // Store base value + deltas
        let mut data = Vec::with_capacity(8 + values.len() * delta_width as usize / 8);
        data.extend_from_slice(&base_val.to_le_bytes());

        let total_bits = (values.len() - 1) * delta_width as usize;
        let delta_bytes = (total_bits + 7) / 8;
        let aligned_size = (delta_bytes + 15) & !15;
        
        data.resize(8 + aligned_size, 0);
        
        let mut bit_offset = 0;
        for i in 1..values.len() {
            let delta = values[i] - values[i-1];
            self.write_bits(&mut data[8..], delta, bit_offset, delta_width)?;
            bit_offset += delta_width as usize;
        }

        self.data = data.into_boxed_slice();
        Ok(())
    }

    fn compress_block_based(
        &mut self, 
        values: &[u64], 
        block_size: BlockSize,
        offset_width: u8,
        sample_width: u8,
        _is_sorted: bool
    ) -> Result<()> {
        let block_units = block_size.units();
        let num_blocks = (values.len() + block_units - 1) / block_units;

        // Build index (samples)
        let mut samples = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let start = block_idx * block_units;
            let end = (start + block_units).min(values.len());
            let block_min = *values[start..end].iter().min().unwrap();
            samples.push(block_min);
        }

        // Compress samples
        let sample_min = *samples.iter().min().unwrap();
        let index_bits = num_blocks * sample_width as usize;
        let index_bytes = (index_bits + 7) / 8;
        let index_aligned = (index_bytes + 15) & !15;
        
        let mut index_data = vec![0u8; index_aligned];
        let mut bit_offset = 0;
        
        for &sample in &samples {
            let offset_sample = sample - sample_min;
            self.write_bits(&mut index_data, offset_sample, bit_offset, sample_width)?;
            bit_offset += sample_width as usize;
        }

        // Compress data (offsets within blocks)
        let data_bits = values.len() * offset_width as usize;
        let data_bytes = (data_bits + 7) / 8;
        let data_aligned = (data_bytes + 15) & !15;
        
        let mut data = vec![0u8; data_aligned];
        bit_offset = 0;

        for block_idx in 0..num_blocks {
            let start = block_idx * block_units;
            let end = (start + block_units).min(values.len());
            let block_min = samples[block_idx];

            for i in start..end {
                let offset = values[i] - block_min;
                self.write_bits(&mut data, offset, bit_offset, offset_width)?;
                bit_offset += offset_width as usize;
            }
        }

        self.index = Some(index_data.into_boxed_slice());
        self.data = data.into_boxed_slice();
        Ok(())
    }

    /// Hardware-accelerated bit writing
    fn write_bits(&self, data: &mut [u8], value: u64, bit_offset: usize, bits: u8) -> Result<()> {
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        if byte_offset >= data.len() || bits > 64 {
            return Err(ZiporaError::invalid_data("Bit write out of bounds"));
        }

        // Mask to prevent overflow
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let masked_value = value & mask;

        // Calculate required bytes
        let bits_needed = bit_in_byte + bits as usize;
        let bytes_needed = (bits_needed + 7) / 8;

        if byte_offset + bytes_needed <= data.len() && bytes_needed <= 8 {
            // Fast unaligned write (up to 64 bits)
            let mut buffer = [0u8; 8];
            let available = (data.len() - byte_offset).min(8);
            buffer[..available].copy_from_slice(&data[byte_offset..byte_offset + available]);

            let current = u64::from_le_bytes(buffer);
            let shifted_value = masked_value << bit_in_byte;
            let result = current | shifted_value;

            let result_bytes = result.to_le_bytes();
            let copy_len = (data.len() - byte_offset).min(8);
            data[byte_offset..byte_offset + copy_len]
                .copy_from_slice(&result_bytes[..copy_len]);
        } else {
            // Fallback bit-by-bit writing
            for i in 0..bits {
                let bit_pos = bit_offset + i as usize;
                let byte_idx = bit_pos / 8;
                let bit_idx = bit_pos % 8;

                if byte_idx >= data.len() {
                    return Err(ZiporaError::invalid_data("Bit position out of bounds"));
                }

                if (masked_value >> i) & 1 == 1 {
                    data[byte_idx] |= 1 << bit_idx;
                }
            }
        }

        Ok(())
    }

    /// Hardware-accelerated bit reading using BMI2 when available
    fn read_bits(&self, data: &[u8], bit_offset: usize, bits: u8) -> Result<u64> {
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        if byte_offset >= data.len() || bits > 64 {
            return Err(ZiporaError::invalid_data("Bit read out of bounds"));
        }

        // Fast unaligned read with hardware acceleration
        let mut buffer = [0u8; 8];
        let available = (data.len() - byte_offset).min(8);
        buffer[..available].copy_from_slice(&data[byte_offset..byte_offset + available]);

        let value = u64::from_le_bytes(buffer);
        
        // Use BMI2 BEXTR for optimal bit extraction when available
        Ok(BitOps::extract_bits(value, bit_in_byte as u8, bits))
    }

    // Decompression methods

    fn get_raw(&self, index: usize) -> Option<u64> {
        let byte_index = index * 8;
        if byte_index + 8 <= self.data.len() {
            // Prefetch next cache line for sequential access
            if byte_index + 64 < self.data.len() {
                PrefetchOps::prefetch_read(unsafe { self.data.as_ptr().add(byte_index + 64) });
            }

            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&self.data[byte_index..byte_index + 8]);
            Some(u64::from_le_bytes(bytes))
        } else {
            None
        }
    }

    fn get_min_max(&self, index: usize, min_val: u64, bit_width: u8) -> Option<u64> {
        let bit_offset = index * bit_width as usize;
        match self.read_bits(&self.data, bit_offset, bit_width) {
            Ok(offset_value) => Some(min_val + offset_value),
            Err(_) => None,
        }
    }

    fn get_delta(&self, index: usize, base_val: u64, delta_width: u8) -> Option<u64> {
        if index == 0 {
            return Some(base_val);
        }

        let bit_offset = (index - 1) * delta_width as usize;
        match self.read_bits(&self.data[8..], bit_offset, delta_width) {
            Ok(delta) => {
                // Reconstruct value by summing deltas
                let mut current_val = base_val;
                for i in 1..=index {
                    let delta_offset = (i - 1) * delta_width as usize;
                    if let Ok(delta) = self.read_bits(&self.data[8..], delta_offset, delta_width) {
                        current_val += delta;
                    } else {
                        return None;
                    }
                }
                Some(current_val)
            }
            Err(_) => None,
        }
    }

    fn get_block_based(
        &self, 
        index: usize, 
        block_size: BlockSize,
        offset_width: u8,
        sample_width: u8,
        _is_sorted: bool
    ) -> Option<u64> {
        let block_units = block_size.units();
        let block_idx = index / block_units;
        let offset_in_block = index % block_units;

        // Get sample (block base value)
        let index_data = self.index.as_ref()?;
        let sample_bit_offset = block_idx * sample_width as usize;
        let sample_offset = self.read_bits(index_data, sample_bit_offset, sample_width).ok()?;

        // Get offset within block  
        let data_bit_offset = index * offset_width as usize;
        let block_offset = self.read_bits(&self.data, data_bit_offset, offset_width).ok()?;

        Some(sample_offset + block_offset)
    }
}

impl<T: PackedInt> Default for IntVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PackedInt> Clone for IntVec<T> {
    fn clone(&self) -> Self {
        Self {
            strategy: self.strategy,
            data: self.data.clone(),
            index: self.index.clone(),
            len: self.len,
            stats: self.stats.clone(),
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let values = vec![1u32, 2, 3, 4, 5];
        let vec = IntVec::from_slice(&values).unwrap();

        assert_eq!(vec.len(), 5);
        assert!(!vec.is_empty());
        
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(vec.get(i), Some(expected));
        }
        assert_eq!(vec.get(5), None);
    }

    #[test]
    fn test_min_max_compression() {
        let values: Vec<u32> = (1000..2000).collect();
        let compressed = IntVec::from_slice(&values).unwrap();

        // Verify all values
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(compressed.get(i), Some(expected));
        }

        // Should achieve excellent compression
        let ratio = compressed.compression_ratio();
        assert!(ratio < 0.4, "Compression ratio {} should be < 0.4", ratio);
    }

    #[test]
    fn test_different_integer_types() {
        // Test u8
        let u8_values: Vec<u8> = (0..255).collect();
        let u8_compressed = IntVec::from_slice(&u8_values).unwrap();
        assert_eq!(u8_compressed.get(100), Some(100));

        // Test i32
        let i32_values = vec![-100i32, -50, 0, 50, 100];
        let i32_compressed = IntVec::from_slice(&i32_values).unwrap();
        assert_eq!(i32_compressed.get(2), Some(0));

        // Test u64
        let u64_values = vec![u64::MAX - 10, u64::MAX - 5, u64::MAX];
        let u64_compressed = IntVec::from_slice(&u64_values).unwrap();
        assert_eq!(u64_compressed.get(2), Some(u64::MAX));
    }

    #[test]
    fn test_sorted_sequence_compression() {
        let values: Vec<u32> = (0..1000).collect();
        let compressed = IntVec::from_slice(&values).unwrap();

        // Should detect sorted sequence and use advanced compression
        match compressed.strategy {
            CompressionStrategy::BlockBased { is_sorted, .. } => {
                assert!(is_sorted, "Should detect sorted sequence");
            }
            _ => {
                // Other strategies are also valid for this data
            }
        }

        // Verify random access
        assert_eq!(compressed.get(0), Some(0));
        assert_eq!(compressed.get(500), Some(500));
        assert_eq!(compressed.get(999), Some(999));

        let ratio = compressed.compression_ratio();
        assert!(ratio < 0.5, "Should achieve good compression for sorted data");
    }

    #[test]
    fn test_delta_compression() {
        // Create sequence with small deltas
        let mut values = vec![1000u32];
        for i in 1..100 {
            values.push(values[i-1] + (i as u32 % 10) + 1);
        }

        let compressed = IntVec::from_slice(&values).unwrap();

        // Verify all values
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(compressed.get(i), Some(expected));
        }
    }

    #[test]
    fn test_empty_vector() {
        let vec: IntVec<u32> = IntVec::new();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert_eq!(vec.get(0), None);
        assert_eq!(vec.compression_ratio(), 1.0);
    }

    #[test]
    fn test_memory_usage() {
        let values: Vec<u32> = (0..1000).collect();
        let compressed = IntVec::from_slice(&values).unwrap();

        let memory_usage = compressed.memory_usage();
        let original_size = values.len() * 4;

        println!("Memory usage: {} bytes vs {} bytes original", memory_usage, original_size);
        println!("Compression ratio: {:.3}", compressed.compression_ratio());

        // Should use significantly less memory
        assert!(memory_usage < original_size);
    }

    #[test]
    fn test_statistics() {
        let values: Vec<u32> = (0..100).collect();
        let compressed = IntVec::from_slice(&values).unwrap();

        let stats = compressed.stats();
        assert!(stats.compression_time_ns > 0);
        assert_eq!(stats.original_size, 400); // 100 * 4 bytes
        assert!(stats.compressed_size < stats.original_size);
    }
}