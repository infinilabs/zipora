//! Advanced hash functions optimized for performance
//!
//! This module provides specialized hash functions inspired by advanced research,
//! including the FaboHashCombine function and golden ratio constants for optimal
//! hash distribution and memory utilization.


/// Golden ratio constant as a fraction (103/64 ≈ 1.609375 ≈ φ)
/// Used for optimal hash table growth and load factor calculations
pub const GOLDEN_RATIO_FRAC_NUM: u64 = 103;
pub const GOLDEN_RATIO_FRAC_DEN: u64 = 64;

/// Alternative golden ratio approximation (13/8 = 1.625)
/// Sometimes used for specific optimization scenarios
pub const GOLDEN_RATIO_ALT_NUM: u64 = 13;
pub const GOLDEN_RATIO_ALT_DEN: u64 = 8;

/// Optimal load factor based on golden ratio (≈ 0.618)
/// Expressed as a fraction of 256 for fast integer arithmetic
pub const GOLDEN_LOAD_FACTOR: u8 = 158; // 158/256 ≈ 0.618

/// FaboHashCombine function inspired by advanced research
/// 
/// This is provided through specialized implementations for u32 and u64.
/// The generic version is removed to avoid complex trait bounds.

/// Specialized FaboHashCombine for u32 values (most common case)
#[inline]
pub fn fabo_hash_combine_u32(hash: u32, value: u32) -> u32 {
    hash.rotate_left(5).wrapping_add(value)
}

/// Specialized FaboHashCombine for u64 values
#[inline]
pub fn fabo_hash_combine_u64(hash: u64, value: u64) -> u64 {
    hash.rotate_left(5).wrapping_add(value)
}


/// Calculate the next size using golden ratio growth
/// 
/// This function computes the next capacity for a hash table or container
/// using the golden ratio for optimal memory utilization and performance.
/// 
/// # Parameters
/// - `current_size`: The current capacity
/// 
/// # Returns
/// The next optimal capacity
pub fn golden_ratio_next_size(current_size: usize) -> usize {
    if current_size == 0 {
        return 16; // Reasonable default starting size
    }
    
    // Use golden ratio: new_size = current_size * 103 / 64
    // Add 1 to ensure growth even for small sizes
    ((current_size as u64 * GOLDEN_RATIO_FRAC_NUM) / GOLDEN_RATIO_FRAC_DEN + 1) as usize
}

/// Calculate optimal bucket count for a hash table
/// 
/// Returns the next power of 2 that can accommodate the desired capacity
/// with the golden ratio load factor.
/// 
/// # Parameters
/// - `desired_capacity`: The desired number of elements
/// 
/// # Returns
/// The optimal bucket count (power of 2)
pub fn optimal_bucket_count(desired_capacity: usize) -> usize {
    if desired_capacity == 0 {
        return 16;
    }
    
    // Calculate required bucket count to maintain golden load factor
    // bucket_count = desired_capacity / load_factor
    let required_buckets = (desired_capacity as u64 * 256) / (GOLDEN_LOAD_FACTOR as u64);
    
    // Round up to next power of 2
    (required_buckets as usize).next_power_of_two().max(16)
}

/// Advanced hash combiner that uses multiple techniques for better distribution
/// 
/// This function combines multiple hash values using a sophisticated approach
/// that includes bit rotation, prime multiplication, and XOR operations.
/// 
/// # Parameters
/// - `hashes`: A slice of hash values to combine
/// 
/// # Returns
/// The combined hash value
pub fn advanced_hash_combine(hashes: &[u64]) -> u64 {
    if hashes.is_empty() {
        return 0;
    }
    
    let mut result = hashes[0];
    
    for &hash in &hashes[1..] {
        // Use FaboHashCombine as the base
        result = fabo_hash_combine_u64(result, hash);
        
        // Add additional mixing for better distribution
        result ^= hash.rotate_right(17);
        result = result.wrapping_mul(0x9e3779b97f4a7c15); // Large prime for mixing
    }
    
    // Final avalanche step
    result ^= result >> 30;
    result = result.wrapping_mul(0xbf58476d1ce4e5b9);
    result ^= result >> 27;
    result = result.wrapping_mul(0x94d049bb133111eb);
    result ^= result >> 31;
    
    result
}

/// Hash function builder that creates optimal hash functions for specific types
/// 
/// This struct allows for creating specialized hash functions that are optimized
/// for particular data patterns or performance requirements.
pub struct HashFunctionBuilder {
    rotation_amount: u32,
    combine_strategy: CombineStrategy,
}

/// Strategy for combining hash values
#[derive(Debug, Clone, Copy)]
pub enum CombineStrategy {
    /// Simple addition (fastest)
    Addition,
    /// XOR combination (good distribution)
    Xor,
    /// FaboHashCombine (balanced performance/distribution)
    Fabo,
    /// Advanced mixing (best distribution, slower)
    Advanced,
}

impl HashFunctionBuilder {
    /// Create a new hash function builder with default settings
    pub fn new() -> Self {
        Self {
            rotation_amount: 5,
            combine_strategy: CombineStrategy::Fabo,
        }
    }
    
    /// Set the rotation amount for bit rotation operations
    pub fn with_rotation(mut self, amount: u32) -> Self {
        self.rotation_amount = amount;
        self
    }
    
    /// Set the combine strategy
    pub fn with_strategy(mut self, strategy: CombineStrategy) -> Self {
        self.combine_strategy = strategy;
        self
    }
    
    /// Build a hash function for u32 values
    pub fn build_u32(self) -> impl Fn(u32, u32) -> u32 {
        let rotation = self.rotation_amount;
        
        move |hash: u32, value: u32| -> u32 {
            match self.combine_strategy {
                CombineStrategy::Addition => hash.rotate_left(rotation).wrapping_add(value),
                CombineStrategy::Xor => hash.rotate_left(rotation) ^ value,
                CombineStrategy::Fabo => fabo_hash_combine_u32(hash, value),
                CombineStrategy::Advanced => {
                    let mut result = hash.rotate_left(rotation).wrapping_add(value);
                    result ^= value.rotate_right(17);
                    result = result.wrapping_mul(0x9e3779b9);
                    result ^= result >> 16;
                    result
                }
            }
        }
    }
    
    /// Build a hash function for u64 values
    pub fn build_u64(self) -> impl Fn(u64, u64) -> u64 {
        let rotation = self.rotation_amount;
        
        move |hash: u64, value: u64| -> u64 {
            match self.combine_strategy {
                CombineStrategy::Addition => hash.rotate_left(rotation).wrapping_add(value),
                CombineStrategy::Xor => hash.rotate_left(rotation) ^ value,
                CombineStrategy::Fabo => fabo_hash_combine_u64(hash, value),
                CombineStrategy::Advanced => {
                    let mut result = hash.rotate_left(rotation).wrapping_add(value);
                    result ^= value.rotate_right(17);
                    result = result.wrapping_mul(0x9e3779b97f4a7c15);
                    result ^= result >> 30;
                    result
                }
            }
        }
    }
}

impl Default for HashFunctionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper trait for types that can be used in hash combining operations
pub trait HashCombinable {
    type Output;
    
    /// Combine this value with a hash using the FaboHashCombine algorithm
    fn fabo_combine(self, hash: Self::Output) -> Self::Output;
    
    /// Get the recommended rotation amount for this type
    fn rotation_amount() -> u32 {
        5
    }
}

impl HashCombinable for u32 {
    type Output = u32;
    
    fn fabo_combine(self, hash: Self::Output) -> Self::Output {
        fabo_hash_combine_u32(hash, self)
    }
}

impl HashCombinable for u64 {
    type Output = u64;
    
    fn fabo_combine(self, hash: Self::Output) -> Self::Output {
        fabo_hash_combine_u64(hash, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fabo_hash_combine_u32() {
        let hash = 0x12345678u32;
        let value = 0xabcdef00u32;
        
        let result = fabo_hash_combine_u32(hash, value);
        
        // Result should be different from both inputs
        assert_ne!(result, hash);
        assert_ne!(result, value);
        
        // Function should be deterministic
        assert_eq!(result, fabo_hash_combine_u32(hash, value));
    }

    #[test]
    fn test_fabo_hash_combine_u64() {
        let hash = 0x123456789abcdef0u64;
        let value = 0xfedcba9876543210u64;
        
        let result = fabo_hash_combine_u64(hash, value);
        
        // Result should be different from both inputs
        assert_ne!(result, hash);
        assert_ne!(result, value);
        
        // Function should be deterministic
        assert_eq!(result, fabo_hash_combine_u64(hash, value));
    }

    #[test]
    fn test_golden_ratio_next_size() {
        assert_eq!(golden_ratio_next_size(0), 16);
        assert_eq!(golden_ratio_next_size(1), 2); // (1 * 103) / 64 + 1 = 2
        assert_eq!(golden_ratio_next_size(64), 104); // (64 * 103) / 64 + 1 = 104
        
        // Test that growth is consistent
        let size1 = 100;
        let size2 = golden_ratio_next_size(size1);
        assert!(size2 > size1);
        
        // Should approximate golden ratio growth
        let ratio = size2 as f64 / size1 as f64;
        assert!(ratio > 1.5 && ratio < 1.7); // Should be close to 1.609
    }

    #[test]
    fn test_optimal_bucket_count() {
        assert_eq!(optimal_bucket_count(0), 16);
        
        // Test that bucket count is always a power of 2
        for capacity in [1, 10, 100, 1000] {
            let bucket_count = optimal_bucket_count(capacity);
            assert!(bucket_count.is_power_of_two());
            assert!(bucket_count >= capacity);
        }
    }

    #[test]
    fn test_advanced_hash_combine() {
        let hashes = [0x123456789abcdef0u64, 0xfedcba9876543210u64, 0x0f0f0f0f0f0f0f0fu64];
        
        let result = advanced_hash_combine(&hashes);
        
        // Result should be different from all inputs
        for &hash in &hashes {
            assert_ne!(result, hash);
        }
        
        // Should be deterministic
        assert_eq!(result, advanced_hash_combine(&hashes));
        
        // Empty slice should return 0
        assert_eq!(advanced_hash_combine(&[]), 0);
        
        // Single element should equal input (possibly modified)
        let single_result = advanced_hash_combine(&[hashes[0]]);
        // Due to avalanche step, this will be different from input
        assert_ne!(single_result, hashes[0]);
    }

    #[test]
    fn test_hash_function_builder() {
        let builder = HashFunctionBuilder::new()
            .with_rotation(7)
            .with_strategy(CombineStrategy::Fabo);
        
        let hash_fn = builder.build_u32();
        
        let result = hash_fn(0x12345678u32, 0xabcdef00u32);
        
        // Should be deterministic
        assert_eq!(result, hash_fn(0x12345678u32, 0xabcdef00u32));
    }

    #[test]
    fn test_hash_combinable_trait() {
        let hash = 0x12345678u32;
        let value = 0xabcdef00u32;
        
        let result1 = value.fabo_combine(hash);
        let result2 = fabo_hash_combine_u32(hash, value);
        
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_combine_strategies() {
        let test_cases = [
            (0x12345678u32, 0xabcdef00u32),
            (0x87654321u32, 0x00fedcbau32),
            (0xfedcba98u32, 0x12345678u32),
        ];
        
        for (hash, value) in test_cases.iter() {
            // Test all strategies produce different results
            let addition = HashFunctionBuilder::new()
                .with_strategy(CombineStrategy::Addition)
                .build_u32()(*hash, *value);
            
            let xor = HashFunctionBuilder::new()
                .with_strategy(CombineStrategy::Xor)
                .build_u32()(*hash, *value);
            
            let fabo = HashFunctionBuilder::new()
                .with_strategy(CombineStrategy::Fabo)
                .build_u32()(*hash, *value);
            
            let advanced = HashFunctionBuilder::new()
                .with_strategy(CombineStrategy::Advanced)
                .build_u32()(*hash, *value);
            
            // Verify that each strategy produces a result
            // (not necessarily all different, but should be deterministic)
            assert_ne!(addition, 0);
            assert_ne!(xor, 0);
            assert_ne!(fabo, 0);
            assert_ne!(advanced, 0);
            
            // Verify strategies are deterministic
            let addition2 = HashFunctionBuilder::new()
                .with_strategy(CombineStrategy::Addition)
                .build_u32()(*hash, *value);
            assert_eq!(addition, addition2);
        }
    }
}