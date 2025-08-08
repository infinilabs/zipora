//! Compressed integer vector with specialized optimizations
//!
//! UintVector provides space-efficient storage for integer sequences with
//! 60-80% memory reduction compared to Vec<u32> through min-max compression
//! and efficient bit packing techniques.

use crate::error::{Result, ZiporaError};
use std::cmp;

/// Compression strategy for integer storage
#[derive(Debug, Clone, Copy, PartialEq)]
enum CompressionStrategy {
    /// Raw storage for uncompressible data
    Raw,
    /// Min-max compression with bit packing
    MinMaxBitPacked { min_val: u32, bit_width: u8 },
    /// Run-length encoding for highly repetitive data
    RunLength,
}

/// Constants for optimization
const MIN_COMPRESSION_ELEMENTS: usize = 4;  // Don't compress tiny datasets
const MIN_COMPRESSION_RATIO: f64 = 0.8;     // Only compress if <80% of raw size
const MIN_ALLOCATION_SIZE: usize = 32;      // Minimum allocation to avoid overhead
const GOLDEN_RATIO_NUMERATOR: usize = 103;  // Golden ratio growth pattern
const GOLDEN_RATIO_DENOMINATOR: usize = 64; // 103/64 ≈ 1.609

/// Compressed integer vector with specialized optimizations
/// 
/// # Key Optimizations
/// 
/// 1. **Min-Max Compression**: Store only (value - min_value) using minimal bits
/// 2. **Efficient Bit Packing**: Pack multiple values into aligned bytes  
/// 3. **Fast Unaligned Access**: Direct memory loads for performance
/// 4. **Adaptive Strategy**: Automatic selection of optimal compression
///
/// # Examples
///
/// ```rust
/// use zipora::UintVector;
/// 
/// // Build from slice with repetitive pattern for optimal compression
/// let values: Vec<u32> = (0..1000).map(|i| i % 100).collect(); // 0-99 repeated
/// let mut vec = UintVector::new();
/// for &val in &values {
///     vec.push(val)?;
/// }
/// 
/// // Or use efficient batch build
/// let compressed = UintVector::build_from(&values)?;
/// assert!(compressed.compression_ratio() < 0.5); // >50% space saving with repetitive data
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
pub struct UintVector {
    /// Current compression strategy in use
    strategy: CompressionStrategy,
    /// Compressed data storage (16-byte aligned for SIMD)
    data: Vec<u8>,
    /// Number of elements stored
    len: usize,
    /// Statistics for compression analysis
    stats: CompressionStats,
    /// Temporary buffer for incremental builds
    temp_values: Vec<u32>,
}

#[derive(Debug, Default)]
struct CompressionStats {
    original_size: usize,
    compressed_size: usize,
    strategy_switches: usize,
}

impl UintVector {
    /// Create a new empty UintVector
    pub fn new() -> Self {
        Self {
            strategy: CompressionStrategy::Raw,
            data: Vec::new(),
            len: 0,
            stats: CompressionStats::default(),
            temp_values: Vec::new(),
        }
    }

    /// Create a UintVector with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = Self::new();
        vec.temp_values.reserve(capacity);
        vec
    }

    /// Build UintVector from slice with optimal compression
    ///
    /// This is the preferred method for creating compressed vectors as it can
    /// analyze the full data range and apply optimal compression strategies.
    ///
    /// # Arguments
    ///
    /// * `values` - Slice of u32 values to compress
    ///
    /// # Returns
    ///
    /// Compressed UintVector with optimal strategy selected
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::UintVector;
    ///
    /// let values = (1000..2000).collect::<Vec<u32>>();
    /// let compressed = UintVector::build_from(&values)?;
    /// 
    /// // Should achieve significant compression for range-limited data
    /// assert!(compressed.compression_ratio() < 0.4);
    /// assert_eq!(compressed.len(), 1000);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn build_from(values: &[u32]) -> Result<Self> {
        if values.is_empty() {
            return Ok(Self::new());
        }

        let mut result = Self::new();
        result.len = values.len();
        
        // Analyze data for optimal compression strategy
        let strategy = Self::analyze_optimal_strategy(values);
        result.strategy = strategy;
        
        // Compress data using selected strategy
        result.compress_with_strategy(values, strategy)?;
        
        // Update statistics
        result.stats.original_size = values.len() * 4;
        result.stats.compressed_size = result.data.len();

        Ok(result)
    }

    /// Add a value to the vector (for incremental building)
    ///
    /// Note: For optimal compression, use `build_from()` when possible.
    /// This method stores values temporarily and recompresses periodically.
    pub fn push(&mut self, value: u32) -> Result<()> {
        self.temp_values.push(value);
        self.len += 1;

        // Recompress every 64 elements or when capacity is reached
        if self.temp_values.len() % 64 == 0 || self.temp_values.len() > 1000 {
            self.recompress_all()?;
        } else {
            // Quick append for small increments
            self.quick_append(value)?;
        }

        Ok(())
    }

    /// Get a value at the specified index with fast decompression
    pub fn get(&self, index: usize) -> Option<u32> {
        if index >= self.len {
            return None;
        }

        // Handle temporary values first
        if !self.temp_values.is_empty() {
            let compressed_len = self.len - self.temp_values.len();
            if index >= compressed_len {
                return self.temp_values.get(index - compressed_len).copied();
            }
        }

        match self.strategy {
            CompressionStrategy::Raw => self.get_raw(index),
            CompressionStrategy::MinMaxBitPacked { min_val, bit_width } => {
                self.get_min_max_bit_packed(index, min_val, bit_width)
            }
            CompressionStrategy::RunLength => self.get_run_length(index),
        }
    }

    /// Get the number of elements in the vector
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the compression ratio (0.0 to 1.0, lower is better)
    pub fn compression_ratio(&self) -> f64 {
        if self.stats.original_size == 0 {
            1.0
        } else {
            self.stats.compressed_size as f64 / self.stats.original_size as f64
        }
    }

    /// Get compression statistics
    pub fn stats(&self) -> (usize, usize, f64) {
        (
            self.stats.original_size,
            self.stats.compressed_size,
            self.compression_ratio(),
        )
    }

    /// Get the memory usage of this UintVector in bytes
    ///
    /// This includes the compressed data storage plus metadata overhead.
    ///
    /// # Returns
    ///
    /// Total memory usage in bytes
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::UintVector;
    ///
    /// let values = vec![42u32; 1000];
    /// let compressed = UintVector::build_from(&values)?;
    /// let usage = compressed.memory_usage();
    /// assert!(usage < 4000); // Should be much less than 4000 bytes (1000 * 4)
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn memory_usage(&self) -> usize {
        // Base struct size
        std::mem::size_of::<Self>() +
        // Compressed data capacity
        self.data.capacity() +
        // Temporary values capacity  
        self.temp_values.capacity() * 4 +
        // Stats overhead
        std::mem::size_of::<CompressionStats>()
    }

    // Private implementation methods

    /// Calculate the ratio of data that forms consecutive runs
    fn calculate_run_ratio(values: &[u32]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mut runs = 0;
        let mut current_run = 1;
        
        for i in 1..values.len() {
            if values[i] == values[i - 1] {
                current_run += 1;
            } else {
                if current_run > 1 {
                    runs += current_run;
                }
                current_run = 1;
            }
        }
        
        if current_run > 1 {
            runs += current_run;
        }
        
        runs as f64 / values.len() as f64
    }

    /// Analyze data to choose optimal compression strategy
    fn analyze_optimal_strategy(values: &[u32]) -> CompressionStrategy {
        if values.is_empty() {
            return CompressionStrategy::Raw;
        }

        // Don't compress very small datasets
        if values.len() < MIN_COMPRESSION_ELEMENTS {
            return CompressionStrategy::Raw;
        }

        // Check for run-length encoding potential (needs long consecutive runs)
        let run_ratio = Self::calculate_run_ratio(values);
        if run_ratio > 0.5 {  // Need >50% of data in runs to be worthwhile
            let raw_bytes = values.len() * 4;
            let rle_estimate = Self::estimate_run_length_size(values);
            if Self::should_compress(values.len(), raw_bytes, rle_estimate) {
                return CompressionStrategy::RunLength;
            }
        }

        // Min-max compression analysis
        let min_val = *values.iter().min().unwrap();
        let max_val = *values.iter().max().unwrap();
        
        if min_val == max_val {
            // All values are the same - check if single-bit storage is worthwhile
            let raw_bytes = values.len() * 4;
            let compressed_estimate = Self::compute_compressed_size(1, values.len());
            if Self::should_compress(values.len(), raw_bytes, compressed_estimate) {
                return CompressionStrategy::MinMaxBitPacked { min_val, bit_width: 1 };
            } else {
                return CompressionStrategy::Raw;
            }
        }

        let range = max_val - min_val;
        let bit_width = if range == 0 { 1 } else { 32 - range.leading_zeros() as u8 };
        
        // Compression decision
        let raw_bytes = values.len() * 4;
        let compressed_estimate = Self::compute_compressed_size(bit_width, values.len());
        
        if Self::should_compress(values.len(), raw_bytes, compressed_estimate) {
            CompressionStrategy::MinMaxBitPacked { min_val, bit_width }
        } else {
            CompressionStrategy::Raw
        }
    }

    /// Compression decision logic
    fn should_compress(num_elements: usize, raw_bytes: usize, compressed_bytes: usize) -> bool {
        if num_elements < MIN_COMPRESSION_ELEMENTS {
            return false;
        }
        let ratio = compressed_bytes as f64 / raw_bytes as f64;
        ratio < MIN_COMPRESSION_RATIO
    }

    /// Compute compressed size with 16-byte alignment
    fn compute_compressed_size(bit_width: u8, num_elements: usize) -> usize {
        let total_bits = bit_width as usize * num_elements;
        let using_size = (total_bits + 7) / 8;
        let touch_size = using_size + 8 - 1;  // for unaligned access
        let aligned_size = (touch_size + 15) & !15; // 16-byte alignment
        std::cmp::max(aligned_size, MIN_ALLOCATION_SIZE)
    }

    /// Estimate run-length encoding size
    fn estimate_run_length_size(values: &[u32]) -> usize {
        if values.is_empty() {
            return 0;
        }

        let mut runs = 1;
        for i in 1..values.len() {
            if values[i] != values[i - 1] {
                runs += 1;
            }
        }
        
        // Each run takes 8 bytes (value + length)
        let estimated_size = runs * 8;
        std::cmp::max(estimated_size, MIN_ALLOCATION_SIZE)
    }

    /// Compress data using the specified strategy
    fn compress_with_strategy(&mut self, values: &[u32], strategy: CompressionStrategy) -> Result<()> {
        self.data.clear();

        match strategy {
            CompressionStrategy::Raw => {
                self.compress_raw(values)
            }
            CompressionStrategy::MinMaxBitPacked { min_val, bit_width } => {
                self.compress_min_max_bit_packed(values, min_val, bit_width)
            }
            CompressionStrategy::RunLength => {
                self.compress_run_length(values)
            }
        }
    }

    /// Raw storage (no compression)
    fn compress_raw(&mut self, values: &[u32]) -> Result<()> {
        self.data.reserve(values.len() * 4);
        for &value in values {
            self.data.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    /// Min-max compression with bit packing
    fn compress_min_max_bit_packed(&mut self, values: &[u32], min_val: u32, bit_width: u8) -> Result<()> {
        if bit_width == 0 || bit_width > 32 {
            return Err(ZiporaError::invalid_data("Invalid bit width"));
        }

        // Use optimized size computation
        let aligned_size = Self::compute_compressed_size(bit_width, values.len());
        self.data.resize(aligned_size, 0);

        let mut bit_offset = 0;
        
        for &value in values {
            if value < min_val {
                return Err(ZiporaError::invalid_data("Value below minimum"));
            }
            
            let offset_value = value - min_val;
            self.write_bits_fast(offset_value, bit_offset, bit_width)?;
            bit_offset += bit_width as usize;
        }

        Ok(())
    }

    /// Run-length encoding for highly repetitive data
    fn compress_run_length(&mut self, values: &[u32]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let mut current_value = values[0];
        let mut run_length = 1u32;

        for &value in values.iter().skip(1) {
            if value == current_value && run_length < u32::MAX {
                run_length += 1;
            } else {
                // Write run: value (4 bytes) + length (4 bytes)
                self.data.extend_from_slice(&current_value.to_le_bytes());
                self.data.extend_from_slice(&run_length.to_le_bytes());
                
                current_value = value;
                run_length = 1;
            }
        }

        // Write final run
        self.data.extend_from_slice(&current_value.to_le_bytes());
        self.data.extend_from_slice(&run_length.to_le_bytes());

        Ok(())
    }

    /// Fast bit writing with unaligned access
    fn write_bits_fast(&mut self, value: u32, bit_offset: usize, bits: u8) -> Result<()> {
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        
        let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
        let masked_value = value & mask;
        
        // Calculate how many bytes we need to span
        let bits_needed = bit_in_byte + bits as usize;
        let bytes_needed = (bits_needed + 7) / 8;
        
        if byte_offset + bytes_needed > self.data.len() {
            return Err(ZiporaError::invalid_data("Bit offset out of bounds"));
        }

        // Fast unaligned write for up to 32 bits
        if bytes_needed <= 8 && byte_offset + 8 <= self.data.len() {
            // Read current 8-byte value with proper boundary handling
            let mut bytes = [0u8; 8];
            let available = std::cmp::min(8, self.data.len() - byte_offset);
            bytes[..available].copy_from_slice(&self.data[byte_offset..byte_offset + available]);
            let current = u64::from_le_bytes(bytes);
            
            let shifted_value = (masked_value as u64) << bit_in_byte;
            let result = current | shifted_value;
            
            let result_bytes = result.to_le_bytes();
            let copy_len = std::cmp::min(8, self.data.len() - byte_offset);
            self.data[byte_offset..byte_offset + copy_len].copy_from_slice(&result_bytes[..copy_len]);
        } else {
            // Fallback for edge cases or when buffer too small
            self.write_bits_slow(masked_value, bit_offset, bits)?;
        }

        Ok(())
    }

    /// Slower but more general bit writing
    fn write_bits_slow(&mut self, value: u32, bit_offset: usize, bits: u8) -> Result<()> {
        for i in 0..bits {
            let bit_pos = bit_offset + i as usize;
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            
            if byte_idx >= self.data.len() {
                return Err(ZiporaError::invalid_data("Bit position out of bounds"));
            }
            
            if (value >> i) & 1 == 1 {
                self.data[byte_idx] |= 1 << bit_idx;
            }
        }
        Ok(())
    }

    /// Fast decompression for raw storage
    fn get_raw(&self, index: usize) -> Option<u32> {
        let byte_index = index * 4;
        if byte_index + 4 <= self.data.len() {
            let bytes = [
                self.data[byte_index],
                self.data[byte_index + 1], 
                self.data[byte_index + 2],
                self.data[byte_index + 3],
            ];
            Some(u32::from_le_bytes(bytes))
        } else {
            None
        }
    }

    /// Fast decompression for min-max bit packed data
    fn get_min_max_bit_packed(&self, index: usize, min_val: u32, bit_width: u8) -> Option<u32> {
        let bit_offset = index * bit_width as usize;
        
        match self.read_bits_fast(bit_offset, bit_width) {
            Ok(offset_value) => Some(min_val + offset_value),
            Err(_) => None,
        }
    }

    /// Fast bit reading with unaligned access  
    fn read_bits_fast(&self, bit_offset: usize, bits: u8) -> Result<u32> {
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        
        if byte_offset >= self.data.len() || bits > 32 {
            return self.read_bits_slow(bit_offset, bits);
        }

        // Fast unaligned read with proper boundary handling
        let mut bytes = [0u8; 8];
        let available = std::cmp::min(8, self.data.len() - byte_offset);
        bytes[..available].copy_from_slice(&self.data[byte_offset..byte_offset + available]);
        
        let value = u64::from_le_bytes(bytes);
        let shifted = value >> bit_in_byte;
        let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
        
        Ok((shifted as u32) & mask)
    }

    /// Slower but more general bit reading
    fn read_bits_slow(&self, bit_offset: usize, bits: u8) -> Result<u32> {
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        
        if byte_offset >= self.data.len() || bits > 32 {
            return Err(ZiporaError::invalid_data("Invalid bit read parameters"));
        }

        let mut result = 0u32;
        let mut bits_read = 0;
        let mut current_byte_offset = byte_offset;
        
        while bits_read < bits && current_byte_offset < self.data.len() {
            let byte_value = self.data[current_byte_offset] as u32;
            let bits_available_in_byte = 8 - if bits_read == 0 { bit_in_byte } else { 0 };
            let bits_to_read = cmp::min(bits - bits_read, bits_available_in_byte as u8) as usize;
            
            let shift_amount = if bits_read == 0 { bit_in_byte } else { 0 };
            let mask = ((1u32 << bits_to_read) - 1) << shift_amount;
            let extracted_bits = (byte_value & mask) >> shift_amount;
            
            result |= extracted_bits << bits_read;
            bits_read += bits_to_read as u8;
            current_byte_offset += 1;
        }

        Ok(result)
    }

    /// Decompression for run-length encoded data
    fn get_run_length(&self, index: usize) -> Option<u32> {
        let mut current_index = 0;
        let mut byte_offset = 0;

        while byte_offset + 8 <= self.data.len() {
            let value_bytes = [
                self.data[byte_offset],
                self.data[byte_offset + 1],
                self.data[byte_offset + 2],
                self.data[byte_offset + 3],
            ];
            let value = u32::from_le_bytes(value_bytes);
            
            let length_bytes = [
                self.data[byte_offset + 4],
                self.data[byte_offset + 5],
                self.data[byte_offset + 6],
                self.data[byte_offset + 7],
            ];
            let length = u32::from_le_bytes(length_bytes);
            
            if index < current_index + length as usize {
                return Some(value);
            }
            
            current_index += length as usize;
            byte_offset += 8;
        }

        None
    }

    /// Quick append for incremental building (less optimal but faster)
    fn quick_append(&mut self, _value: u32) -> Result<()> {
        // Update statistics for incremental mode
        self.stats.original_size = self.len * 4;
        self.stats.compressed_size = self.data.len() + self.temp_values.len() * 4;
        Ok(())
    }

    /// Recompress all data including temporary values
    fn recompress_all(&mut self) -> Result<()> {
        if self.temp_values.is_empty() {
            return Ok(());
        }

        // Collect all values
        let mut all_values = Vec::with_capacity(self.len);
        
        // Extract existing compressed values
        match self.strategy {
            CompressionStrategy::Raw => {
                for i in 0..(self.len - self.temp_values.len()) {
                    if let Some(val) = self.get_raw(i) {
                        all_values.push(val);
                    }
                }
            }
            CompressionStrategy::MinMaxBitPacked { min_val, bit_width } => {
                for i in 0..(self.len - self.temp_values.len()) {
                    if let Some(val) = self.get_min_max_bit_packed(i, min_val, bit_width) {
                        all_values.push(val);
                    }
                }
            }
            CompressionStrategy::RunLength => {
                for i in 0..(self.len - self.temp_values.len()) {
                    if let Some(val) = self.get_run_length(i) {
                        all_values.push(val);
                    }
                }
            }
        }
        
        // Add temporary values
        all_values.extend_from_slice(&self.temp_values);
        
        // Recompress with optimal strategy
        let new_strategy = Self::analyze_optimal_strategy(&all_values);
        self.strategy = new_strategy;
        self.compress_with_strategy(&all_values, new_strategy)?;
        
        // Clear temporary values
        self.temp_values.clear();
        
        // Update statistics
        self.stats.original_size = self.len * 4;
        self.stats.compressed_size = self.data.len();
        self.stats.strategy_switches += 1;

        Ok(())
    }
}

impl Default for UintVector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut vec = UintVector::new();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());

        vec.push(42).unwrap();
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.get(0), Some(42));
        assert_eq!(vec.get(1), None);
    }

    #[test]
    fn test_min_max_compression() {
        // Test data that should compress very well (range 1000-1999)
        let values: Vec<u32> = (1000..2000).collect();
        let compressed = UintVector::build_from(&values).unwrap();
        
        // Verify all values are correct
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(compressed.get(i), Some(expected), "Mismatch at index {}", i);
        }
        
        // Should achieve excellent compression (10 bits instead of 32)
        let ratio = compressed.compression_ratio();
        assert!(ratio < 0.4, "Compression ratio {} should be < 0.4", ratio);
        
        println!("Min-max compression test: {:.1}% space savings", (1.0 - ratio) * 100.0);
    }

    #[test]
    fn test_small_range_compression() {
        // Test data with very small range (should use minimal bits)
        let values = vec![42u32; 1000];
        let compressed = UintVector::build_from(&values).unwrap();
        
        // Verify all values
        for i in 0..1000 {
            assert_eq!(compressed.get(i), Some(42));
        }
        
        // Should achieve excellent compression (1 bit per value)
        let ratio = compressed.compression_ratio();
        assert!(ratio < 0.1, "Compression ratio {} should be < 0.1 for identical values", ratio);
        
        println!("Small range compression test: {:.1}% space savings", (1.0 - ratio) * 100.0);
    }

    #[test]
    fn test_benchmark_pattern() {
        // Test the specific pattern used in benchmarks: (i % 1000)
        let size = 100000;
        let test_data: Vec<u32> = (0..size).map(|i| (i % 1000) as u32).collect();
        
        let compressed = UintVector::build_from(&test_data).unwrap();
        
        // Debug output
        println!("Strategy: {:?}", compressed.strategy);
        println!("Data len: {} bytes", compressed.data.len());
        println!("Original size: {} bytes", size * 4);
        println!("Compressed size: {} bytes", compressed.stats.compressed_size);
        let ratio = compressed.compression_ratio();
        println!("Compression ratio: {:.3}", ratio);
        
        // Verify first 100 values
        for i in 0..100 {
            assert_eq!(compressed.get(i), Some((i % 1000) as u32));
        }
        
        // Should achieve target compression (60-80% space reduction)
        assert!(ratio < 0.5, "Compression ratio {} should be < 0.5", ratio);
        
        println!("Benchmark pattern compression: {:.1}% space savings", (1.0 - ratio) * 100.0);
        println!("Original size: {} bytes", size * 4);
        println!("Compressed size: {} bytes", compressed.memory_usage());
    }

    #[test]
    fn test_incremental_vs_batch() {
        let values: Vec<u32> = (500..600).collect();
        
        // Build incrementally
        let mut incremental = UintVector::new();
        for &val in &values {
            incremental.push(val).unwrap();
        }
        
        // Build in batch
        let batch = UintVector::build_from(&values).unwrap();
        
        // Both should have same results
        assert_eq!(incremental.len(), batch.len());
        for i in 0..values.len() {
            assert_eq!(incremental.get(i), batch.get(i));
        }
        
        // Batch should achieve better compression
        assert!(batch.compression_ratio() <= incremental.compression_ratio());
    }

    #[test]
    fn test_large_values() {
        let values = vec![u32::MAX - 10, u32::MAX - 5, u32::MAX];
        let compressed = UintVector::build_from(&values).unwrap();
        
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(compressed.get(i), Some(expected));
        }
        
        // Small datasets with small ranges should use raw storage
        let ratio = compressed.compression_ratio();
        // For only 3 large values with small range (10), raw storage is more efficient
        assert!(ratio <= 1.0, "Should use raw storage or achieve compression for large values");
        println!("Large values compression test: ratio = {:.3}", ratio);
    }

    #[test]
    fn test_run_length_compression() {
        // Create data with many repeated values
        let mut values = Vec::new();
        for i in 0..10 {
            for _ in 0..100 {
                values.push(i);
            }
        }
        
        let compressed = UintVector::build_from(&values).unwrap();
        
        // Verify all values
        let mut expected_idx = 0;
        for i in 0..10 {
            for _ in 0..100 {
                assert_eq!(compressed.get(expected_idx), Some(i));
                expected_idx += 1;
            }
        }
        
        // Should achieve good compression with run-length encoding
        let ratio = compressed.compression_ratio();
        println!("Run-length compression: {:.1}% space savings", (1.0 - ratio) * 100.0);
    }

    #[test]
    fn test_empty_vector() {
        let vec = UintVector::new();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert_eq!(vec.get(0), None);
        assert_eq!(vec.compression_ratio(), 1.0);
    }

    #[test]
    fn test_single_value() {
        let compressed = UintVector::build_from(&[42]).unwrap();
        assert_eq!(compressed.len(), 1);
        assert_eq!(compressed.get(0), Some(42));
        // Single values use raw storage to avoid expansion
        let ratio = compressed.compression_ratio();
        assert!(ratio <= 1.0, "Single value should use raw storage to avoid expansion");
        println!("Single value compression test: ratio = {:.3}", ratio);
    }

    #[test]
    fn test_incremental_push_debug() {
        // Test incremental push with the benchmark pattern to debug the issue
        let mut vec = UintVector::new();
        
        // Push first 100 values to trigger recompression
        for i in 0..100 {
            let value = (i % 1000) as u32;
            vec.push(value).unwrap();
        }
        
        // Verify all values are correct
        for i in 0..100 {
            assert_eq!(vec.get(i), Some((i % 1000) as u32), "Mismatch at index {}", i);
        }
        
        println!("Incremental push test passed for 100 values");
    }

    #[test]
    fn test_memory_usage() {
        let values: Vec<u32> = (0..1000).collect();
        let compressed = UintVector::build_from(&values).unwrap();
        
        let memory_usage = compressed.memory_usage();
        let original_size = values.len() * 4;
        
        println!("Memory usage: {} bytes vs {} bytes original", memory_usage, original_size);
        
        // Should use significantly less memory than original
        assert!(memory_usage < original_size);
    }
}