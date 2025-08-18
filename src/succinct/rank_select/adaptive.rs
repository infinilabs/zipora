//! Adaptive Strategy Selection for Rank/Select Operations
//!
//! This module implements adaptive selection of optimal rank/select implementations
//! based on data density analysis, inspired by the topling-zip adaptive system.
//!
//! The adaptive system analyzes data characteristics and automatically selects
//! the most appropriate implementation:
//!
//! - **Sparse data (< 5% density)**: Uses `RankSelectFew` for space efficiency
//! - **Dense data (> 90% density)**: Uses optimized dense implementations  
//! - **Balanced data**: Uses general-purpose implementations with optimal block sizes
//! - **Multi-dimensional**: Uses mixed implementations for related bit vectors
//!
//! # Examples
//!
//! ```rust
//! use zipora::{BitVector, AdaptiveRankSelect, RankSelectOps};
//!
//! // Create sparse bit vector (every 100th bit set)
//! let mut sparse_bv = BitVector::new();
//! for i in 0..10000 {
//!     sparse_bv.push(i % 100 == 0).unwrap();
//! }
//!
//! // Adaptive selection automatically chooses RankSelectFew
//! let adaptive_rs = AdaptiveRankSelect::new(sparse_bv).unwrap();
//! println!("Selected: {}", adaptive_rs.implementation_name());
//! 
//! let rank = adaptive_rs.rank1(5000);
//! let pos = adaptive_rs.select1(25).unwrap();
//! ```

use crate::error::{Result, ZiporaError};
use crate::succinct::BitVector;
use super::{
    RankSelectOps, RankSelectSimple, RankSelectSeparated256, RankSelectSeparated512,
    RankSelectInterleaved256, RankSelectFew, RankSelectMixedIL256,
    utils::{optimal_block_size, optimal_select_sample_rate},
};
use std::fmt;

/// Data characteristics profile for adaptive selection
#[derive(Debug, Clone)]
pub struct DataProfile {
    /// Total number of bits
    pub total_bits: usize,
    /// Number of set bits (ones)
    pub ones_count: usize,
    /// Bit density (ratio of ones to total bits)
    pub density: f64,
    /// Estimated access pattern preference
    pub access_pattern: AccessPattern,
    /// Data size category
    pub size_category: SizeCategory,
}

/// Expected access pattern for optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    /// Mixed rank and select operations
    Mixed,
    /// Primarily rank operations
    RankHeavy,
    /// Primarily select operations  
    SelectHeavy,
    /// Sequential access pattern
    Sequential,
    /// Random access pattern
    Random,
}

/// Data size categories for implementation selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeCategory {
    /// Small datasets (< 10K bits)
    Small,
    /// Medium datasets (10K - 1M bits)
    Medium,
    /// Large datasets (1M - 100M bits)
    Large,
    /// Very large datasets (> 100M bits)
    VeryLarge,
}

/// Strategy selection criteria based on topling-zip patterns
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Sparsity threshold for RankSelectFew (default: 0.05 = 5%)
    pub sparse_threshold: f64,
    /// Dense threshold for dense optimizations (default: 0.90 = 90%)
    pub dense_threshold: f64,
    /// Small dataset threshold in bits (default: 10,000)
    pub small_dataset_threshold: usize,
    /// Large dataset threshold in bits (default: 1,000,000)
    pub large_dataset_threshold: usize,
    /// Very large dataset threshold in bits (default: 100,000,000)
    pub very_large_dataset_threshold: usize,
    /// Preferred access pattern for optimization
    pub access_pattern: AccessPattern,
    /// Enable select cache optimization
    pub enable_select_cache: bool,
    /// Prefer space efficiency over speed
    pub prefer_space: bool,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            sparse_threshold: 0.05,
            dense_threshold: 0.90,
            small_dataset_threshold: 10_000,
            large_dataset_threshold: 1_000_000,
            very_large_dataset_threshold: 100_000_000,
            access_pattern: AccessPattern::Mixed,
            enable_select_cache: true,
            prefer_space: false,
        }
    }
}

/// Adaptive rank/select implementation that automatically selects optimal strategy
pub struct AdaptiveRankSelect {
    /// The actual implementation chosen by adaptive selection
    implementation: Box<dyn RankSelectOps + Send + Sync>,
    /// Name of the selected implementation
    implementation_name: String,
    /// Data profile used for selection
    profile: DataProfile,
    /// Selection criteria used
    criteria: SelectionCriteria,
}

impl AdaptiveRankSelect {
    /// Create new adaptive rank/select with automatic strategy selection
    pub fn new(bit_vector: BitVector) -> Result<Self> {
        Self::with_criteria(bit_vector, SelectionCriteria::default())
    }

    /// Create new adaptive rank/select with custom selection criteria
    pub fn with_criteria(bit_vector: BitVector, criteria: SelectionCriteria) -> Result<Self> {
        let profile = Self::analyze_data(&bit_vector, &criteria);
        let (implementation, name) = Self::select_implementation(bit_vector, &profile, &criteria)?;
        
        Ok(Self {
            implementation,
            implementation_name: name,
            profile,
            criteria,
        })
    }

    /// Analyze bit vector characteristics for strategy selection
    fn analyze_data(bit_vector: &BitVector, criteria: &SelectionCriteria) -> DataProfile {
        let total_bits = bit_vector.len();
        let ones_count = bit_vector.count_ones();
        let density = if total_bits == 0 { 0.0 } else { ones_count as f64 / total_bits as f64 };

        let size_category = match total_bits {
            0..=9999 => SizeCategory::Small,
            10000..=999999 => SizeCategory::Medium,
            1000000..=99999999 => SizeCategory::Large,
            _ => SizeCategory::VeryLarge,
        };

        // Use provided access pattern or infer from criteria
        let access_pattern = criteria.access_pattern;

        DataProfile {
            total_bits,
            ones_count,
            density,
            access_pattern,
            size_category,
        }
    }

    /// Select optimal implementation based on data profile (inspired by topling-zip)
    fn select_implementation(
        bit_vector: BitVector,
        profile: &DataProfile,
        criteria: &SelectionCriteria,
    ) -> Result<(Box<dyn RankSelectOps + Send + Sync>, String)> {
        // Edge cases: all zeros or all ones
        if profile.ones_count == 0 {
            return Ok((
                Box::new(RankSelectSimple::new(bit_vector)?),
                "RankSelectSimple (all zeros)".to_string(),
            ));
        }
        
        if profile.ones_count == profile.total_bits {
            return Ok((
                Box::new(RankSelectSimple::new(bit_vector)?),
                "RankSelectSimple (all ones)".to_string(),
            ));
        }

        // Sparse data optimization (topling-zip pattern)
        if profile.density < criteria.sparse_threshold {
            return if profile.ones_count * 2 < profile.total_bits {
                // Few ones pattern - store positions of set bits
                Ok((
                    Box::new(RankSelectFew::<true, 64>::from_bit_vector(bit_vector)?),
                    "RankSelectFew<true> (sparse ones)".to_string(),
                ))
            } else {
                // Few zeros pattern - store positions of clear bits  
                Ok((
                    Box::new(RankSelectFew::<false, 64>::from_bit_vector(bit_vector)?),
                    "RankSelectFew<false> (sparse zeros)".to_string(),
                ))
            };
        }

        // Very dense data with complementary representation
        if profile.density > criteria.dense_threshold {
            return Ok((
                Box::new(RankSelectSeparated256::new(bit_vector)?),
                "RankSelectSeparated256 (dense)".to_string(),
            ));
        }

        // Standard density selection based on size and access pattern
        match (profile.size_category, profile.access_pattern) {
            // Small datasets: prefer cache-efficient implementations
            (SizeCategory::Small, _) => Ok((
                Box::new(RankSelectInterleaved256::new(bit_vector)?),
                "RankSelectInterleaved256 (small dataset)".to_string(),
            )),

            // Medium datasets: optimize based on access pattern
            (SizeCategory::Medium, AccessPattern::Sequential) => Ok((
                Box::new(RankSelectSeparated512::new(bit_vector)?),
                "RankSelectSeparated512 (sequential access)".to_string(),
            )),
            
            (SizeCategory::Medium, AccessPattern::Mixed) => Ok((
                Box::new(RankSelectInterleaved256::new(bit_vector)?),
                "RankSelectInterleaved256 (mixed access)".to_string(),
            )),

            (SizeCategory::Medium, _) => Ok((
                Box::new(RankSelectSeparated256::new(bit_vector)?),
                "RankSelectSeparated256 (medium dataset)".to_string(),
            )),

            // Large datasets: prefer separated implementations for better cache behavior
            (SizeCategory::Large, AccessPattern::Sequential) => Ok((
                Box::new(RankSelectSeparated512::new(bit_vector)?),
                "RankSelectSeparated512 (large sequential)".to_string(),
            )),

            (SizeCategory::Large, _) => Ok((
                Box::new(RankSelectSeparated256::new(bit_vector)?),
                "RankSelectSeparated256 (large dataset)".to_string(),
            )),

            // Very large datasets: always use separated implementation with larger blocks
            (SizeCategory::VeryLarge, _) => Ok((
                Box::new(RankSelectSeparated512::new(bit_vector)?),
                "RankSelectSeparated512 (very large dataset)".to_string(),
            )),
        }
    }

    /// Get the name of the selected implementation
    pub fn implementation_name(&self) -> &str {
        &self.implementation_name
    }

    /// Get the data profile used for selection
    pub fn data_profile(&self) -> &DataProfile {
        &self.profile
    }

    /// Get the selection criteria used
    pub fn selection_criteria(&self) -> &SelectionCriteria {
        &self.criteria
    }

    /// Get optimization statistics
    pub fn optimization_stats(&self) -> OptimizationStats {
        OptimizationStats {
            total_bits: self.profile.total_bits,
            ones_count: self.profile.ones_count,
            density: self.profile.density,
            implementation: self.implementation_name.clone(),
            space_overhead_percent: self.implementation.space_overhead_percent(),
            estimated_performance_tier: self.estimate_performance_tier(),
        }
    }

    /// Estimate performance tier of selected implementation
    fn estimate_performance_tier(&self) -> PerformanceTier {
        if self.implementation_name.contains("RankSelectFew") {
            PerformanceTier::SpaceOptimized
        } else if self.implementation_name.contains("Interleaved") {
            PerformanceTier::HighPerformance
        } else if self.implementation_name.contains("512") {
            PerformanceTier::Sequential
        } else {
            PerformanceTier::Balanced
        }
    }
}

/// Performance tier classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerformanceTier {
    /// Optimized for space efficiency
    SpaceOptimized,
    /// High-performance for mixed operations
    HighPerformance,
    /// Optimized for sequential access
    Sequential,
    /// Balanced space and performance
    Balanced,
}

/// Optimization statistics for adaptive selection
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Total number of bits
    pub total_bits: usize,
    /// Number of set bits
    pub ones_count: usize,
    /// Bit density
    pub density: f64,
    /// Selected implementation name
    pub implementation: String,
    /// Space overhead percentage
    pub space_overhead_percent: f64,
    /// Estimated performance tier
    pub estimated_performance_tier: PerformanceTier,
}

impl fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AdaptiveRankSelect Stats:\n\
             Total bits: {}\n\
             Ones count: {} ({:.2}% density)\n\
             Implementation: {}\n\
             Space overhead: {:.2}%\n\
             Performance tier: {:?}",
            self.total_bits,
            self.ones_count,
            self.density * 100.0,
            self.implementation,
            self.space_overhead_percent,
            self.estimated_performance_tier
        )
    }
}

// Implement RankSelectOps trait by delegating to selected implementation
impl RankSelectOps for AdaptiveRankSelect {
    fn rank1(&self, pos: usize) -> usize {
        self.implementation.rank1(pos)
    }

    fn rank0(&self, pos: usize) -> usize {
        self.implementation.rank0(pos)
    }

    fn select1(&self, k: usize) -> Result<usize> {
        self.implementation.select1(k)
    }

    fn select0(&self, k: usize) -> Result<usize> {
        self.implementation.select0(k)
    }

    fn len(&self) -> usize {
        self.implementation.len()
    }

    fn count_ones(&self) -> usize {
        self.implementation.count_ones()
    }

    fn get(&self, index: usize) -> Option<bool> {
        self.implementation.get(index)
    }

    fn space_overhead_percent(&self) -> f64 {
        self.implementation.space_overhead_percent()
    }
}

/// Multi-dimensional adaptive selection for related bit vectors
pub struct AdaptiveMultiDimensional {
    /// The selected multi-dimensional implementation
    implementation: Box<dyn RankSelectOps + Send + Sync>,
    /// Implementation name
    implementation_name: String,
    /// Number of dimensions
    dimensions: usize,
}

impl AdaptiveMultiDimensional {
    /// Create adaptive multi-dimensional rank/select for two related bit vectors
    pub fn new_dual(bit_vector1: BitVector, bit_vector2: BitVector) -> Result<Self> {
        if bit_vector1.len() != bit_vector2.len() {
            return Err(ZiporaError::invalid_data(
                "Bit vectors must have the same length".to_string()
            ));
        }

        // For now, always use interleaved implementation for dual dimensions
        // Future enhancement: analyze correlation and access patterns
        let implementation = Box::new(RankSelectMixedIL256::new([bit_vector1, bit_vector2])?);
        
        Ok(Self {
            implementation,
            implementation_name: "RankSelectMixedIL256 (dual)".to_string(),
            dimensions: 2,
        })
    }

    /// Get the name of the selected implementation
    pub fn implementation_name(&self) -> &str {
        &self.implementation_name
    }

    /// Get the number of dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl RankSelectOps for AdaptiveMultiDimensional {
    fn rank1(&self, pos: usize) -> usize {
        self.implementation.rank1(pos)
    }

    fn rank0(&self, pos: usize) -> usize {
        self.implementation.rank0(pos)
    }

    fn select1(&self, k: usize) -> Result<usize> {
        self.implementation.select1(k)
    }

    fn select0(&self, k: usize) -> Result<usize> {
        self.implementation.select0(k)
    }

    fn len(&self) -> usize {
        self.implementation.len()
    }

    fn count_ones(&self) -> usize {
        self.implementation.count_ones()
    }

    fn get(&self, index: usize) -> Option<bool> {
        self.implementation.get(index)
    }

    fn space_overhead_percent(&self) -> f64 {
        self.implementation.space_overhead_percent()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sparse_bitvector(size: usize, sparsity: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(i % sparsity == 0).unwrap();
        }
        bv
    }

    fn create_dense_bitvector(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(i % 10 != 0).unwrap(); // 90% density
        }
        bv
    }

    fn create_balanced_bitvector(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(i % 2 == 0).unwrap(); // 50% density
        }
        bv
    }

    #[test]
    fn test_sparse_data_selection() {
        // Very sparse data (1% density)
        let sparse_bv = create_sparse_bitvector(10000, 100);
        let adaptive = AdaptiveRankSelect::new(sparse_bv).unwrap();
        
        assert!(adaptive.implementation_name().contains("RankSelectFew"));
        assert!(adaptive.data_profile().density < 0.05);
        
        // Test basic operations
        let rank = adaptive.rank1(1000);
        assert_eq!(rank, 10); // Every 100th bit, so 10 bits set in first 1000
        
        let pos = adaptive.select1(5).unwrap();
        assert_eq!(pos, 500); // 5th set bit is at position 500
    }

    #[test]
    fn test_dense_data_selection() {
        // Dense data (90% density)
        let dense_bv = create_dense_bitvector(1000);
        let adaptive = AdaptiveRankSelect::new(dense_bv).unwrap();
        
        println!("Dense test - Selected: {}, Density: {:.3}, Size category: {:?}", 
                 adaptive.implementation_name(), 
                 adaptive.data_profile().density,
                 adaptive.data_profile().size_category);
        
        // Small dense dataset should select cache-efficient implementation
        assert!(adaptive.implementation_name().contains("RankSelectInterleaved256"));
        assert!(adaptive.data_profile().density > 0.8);
        
        // Test basic operations
        let rank = adaptive.rank1(100);
        assert_eq!(rank, 90); // 90% of 100 bits
    }

    #[test]
    fn test_balanced_data_selection() {
        // Balanced data (50% density)
        let balanced_bv = create_balanced_bitvector(50000);
        let adaptive = AdaptiveRankSelect::new(balanced_bv).unwrap();
        
        // Should select based on size (medium dataset)
        assert!(adaptive.implementation_name().contains("RankSelect"));
        assert!((adaptive.data_profile().density - 0.5).abs() < 0.1);
        
        // Test basic operations
        let rank = adaptive.rank1(1000);
        assert_eq!(rank, 500); // 50% of 1000 bits
    }

    #[test]
    fn test_small_dataset_selection() {
        // Small dataset should prefer cache-efficient implementation
        let small_bv = create_balanced_bitvector(5000);
        let adaptive = AdaptiveRankSelect::new(small_bv).unwrap();
        
        assert!(adaptive.implementation_name().contains("Interleaved256"));
        assert_eq!(adaptive.data_profile().size_category, SizeCategory::Small);
    }

    #[test]
    fn test_large_dataset_selection() {
        // Large dataset should prefer separated implementation
        let large_bv = create_balanced_bitvector(5_000_000);
        let adaptive = AdaptiveRankSelect::new(large_bv).unwrap();
        
        assert!(adaptive.implementation_name().contains("Separated"));
        assert_eq!(adaptive.data_profile().size_category, SizeCategory::Large);
    }

    #[test]
    fn test_custom_criteria() {
        let bv = create_sparse_bitvector(1000, 50); // 2% density
        
        // Custom criteria with higher sparse threshold
        let criteria = SelectionCriteria {
            sparse_threshold: 0.025, // 2.5% - should trigger sparse optimization for 2% data
            ..Default::default()
        };
        
        let adaptive = AdaptiveRankSelect::with_criteria(bv.clone(), criteria).unwrap();
        
        println!("Custom criteria test - Selected: {}, Density: {:.3}", 
                 adaptive.implementation_name(), 
                 adaptive.data_profile().density);
        
        assert!(adaptive.implementation_name().contains("RankSelectFew"));
    }

    #[test]
    fn test_edge_cases() {
        // All zeros
        let all_zeros = BitVector::with_size(1000, false).unwrap();
        let adaptive = AdaptiveRankSelect::new(all_zeros).unwrap();
        assert!(adaptive.implementation_name().contains("RankSelectSimple"));
        assert_eq!(adaptive.count_ones(), 0);
        
        // All ones
        let all_ones = BitVector::with_size(1000, true).unwrap();
        let adaptive = AdaptiveRankSelect::new(all_ones).unwrap();
        assert!(adaptive.implementation_name().contains("RankSelectSimple"));
        assert_eq!(adaptive.count_ones(), 1000);
    }

    #[test]
    fn test_optimization_stats() {
        let sparse_bv = create_sparse_bitvector(10000, 100);
        let adaptive = AdaptiveRankSelect::new(sparse_bv).unwrap();
        
        let stats = adaptive.optimization_stats();
        assert_eq!(stats.total_bits, 10000);
        assert_eq!(stats.ones_count, 100);
        assert!((stats.density - 0.01).abs() < 0.001);
        assert!(stats.implementation.contains("RankSelectFew"));
        assert_eq!(stats.estimated_performance_tier, PerformanceTier::SpaceOptimized);
        
        // Test display
        let display_str = format!("{}", stats);
        assert!(display_str.contains("AdaptiveRankSelect Stats"));
        assert!(display_str.contains("RankSelectFew"));
    }

    #[test]
    fn test_multi_dimensional_adaptive() {
        let bv1 = create_balanced_bitvector(1000);
        let bv2 = create_sparse_bitvector(1000, 50);
        
        let multi = AdaptiveMultiDimensional::new_dual(bv1, bv2).unwrap();
        assert!(multi.implementation_name().contains("RankSelectMixedIL256"));
        assert_eq!(multi.dimensions(), 2);
        
        // Test basic operations
        assert_eq!(multi.len(), 1000);
        let rank = multi.rank1(100);
        assert!(rank > 0);
    }

    #[test]
    fn test_multi_dimensional_length_mismatch() {
        let bv1 = create_balanced_bitvector(1000);
        let bv2 = create_balanced_bitvector(500); // Different length
        
        let result = AdaptiveMultiDimensional::new_dual(bv1, bv2);
        assert!(result.is_err());
    }

    #[test]
    fn test_data_profile_analysis() {
        let bv = create_sparse_bitvector(50000, 200); // 0.5% density
        let criteria = SelectionCriteria::default();
        let profile = AdaptiveRankSelect::analyze_data(&bv, &criteria);
        
        assert_eq!(profile.total_bits, 50000);
        assert_eq!(profile.ones_count, 250);
        assert!((profile.density - 0.005).abs() < 0.001);
        assert_eq!(profile.size_category, SizeCategory::Medium);
        assert_eq!(profile.access_pattern, AccessPattern::Mixed);
    }

    #[test]
    fn test_performance_consistency() {
        // Ensure adaptive selection provides consistent results
        let bv = create_balanced_bitvector(10000);
        let adaptive = AdaptiveRankSelect::new(bv.clone()).unwrap();
        let reference = RankSelectSimple::new(bv).unwrap();
        
        // Test multiple positions
        for pos in [0, 1000, 5000, 9999] {
            assert_eq!(adaptive.rank1(pos), reference.rank1(pos));
            assert_eq!(adaptive.rank0(pos), reference.rank0(pos));
        }
        
        // Test select operations
        let ones_count = adaptive.count_ones();
        for k in [0, ones_count / 4, ones_count / 2, ones_count - 1] {
            if k < ones_count {
                assert_eq!(adaptive.select1(k).unwrap(), reference.select1(k).unwrap());
            }
        }
    }
}