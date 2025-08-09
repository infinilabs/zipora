//! Advanced Rank/Select Variants with High-Performance Optimizations
//!
//! This module provides a comprehensive collection of rank/select implementations
//! based on research from advanced succinct data structure libraries. Each variant
//! is optimized for different use cases and memory/performance trade-offs.
//!
//! # Variants Overview
//!
//! ## Basic Variants
//! - **`RankSelectSimple`**: Reference implementation with minimal optimizations
//! - **`RankSelectSeparated256`**: High-performance separated cache (256-bit blocks)
//! - **`RankSelectSeparated512`**: Larger block variant (512-bit blocks)
//! - **`RankSelectInterleaved256`**: Cache-optimized interleaved layout
//!
//! ## Sparse Optimizations
//! - **`RankSelectFew`**: Memory-efficient for sparse bit vectors
//!
//! ## Multi-Dimensional Variants
//! - **`RankSelectMixedIL256`**: Dual-dimension interleaved (2 bit vectors)
//! - **`RankSelectMixedSE512`**: Dual-dimension separated (2 bit vectors)
//! - **`RankSelectMixedXL256`**: Multi-dimension extended (2-4 bit vectors)
//!
//! # Performance Characteristics
//!
//! | Variant | Memory Overhead | Rank Time | Select Time | Best Use Case |
//! |---------|----------------|-----------|-------------|---------------|
//! | Simple | ~25% | O(1) | O(log n) | Testing/reference |
//! | Separated256 | ~25% | O(1) | O(1) | General purpose |
//! | Separated512 | ~25% | O(1) | O(1) | Sequential access |
//! | Interleaved256 | ~25% | O(1) | O(1) | Random access |
//! | Few | <5% | O(log n) | O(1) | Sparse data |
//! | Mixed variants | ~30% | O(1) | O(1) | Multi-dimensional |
//!
//! # SIMD Optimizations
//!
//! All variants include SIMD optimizations with runtime feature detection:
//! - **BMI2**: Ultra-fast select using PDEP/PEXT instructions
//! - **POPCNT**: Hardware-accelerated popcount
//! - **AVX-512**: Bulk operations with vectorized popcount
//! - **ARM NEON**: Cross-platform SIMD support
//!
//! # Feature Flags
//!
//! - `simd`: Enable SIMD optimizations (default)
//! - `avx512`: Enable AVX-512 optimizations (requires nightly Rust)
//!
//! # Examples
//!
//! ```rust
//! use zipora::{BitVector, RankSelectOps, RankSelectSeparated256, RankSelectFew};
//!
//! // High-performance general-purpose variant
//! let mut bv = BitVector::new();
//! for i in 0..1000 {
//!     bv.push(i % 3 == 0)?;
//! }
//! let rs = RankSelectSeparated256::new(bv.clone())?;
//! let rank = rs.rank1(500);
//! let pos = rs.select1(100)?;
//!
//! // Memory-efficient sparse variant
//! let sparse_rs = RankSelectFew::<false, 64>::from_bit_vector(bv)?;
//! let sparse_rank = sparse_rs.rank1(500);
//! # Ok::<(), zipora::ZiporaError>(())
//! ```

use crate::error::{Result, ZiporaError};
use crate::succinct::BitVector;
use std::fmt;

// Re-export the original RankSelect256 for backward compatibility
pub use legacy::{RankSelect256, RankSelectSe256, CpuFeatures};

// Import the legacy implementation
pub mod legacy;

// Import the new rank/select variants
pub mod simple;
pub mod separated;
pub mod interleaved;
pub mod sparse;
pub mod mixed;
pub mod simd;
pub mod builder;

// Re-export all variants for convenient access
pub use simple::RankSelectSimple;
pub use separated::{RankSelectSeparated256, RankSelectSeparated512};
pub use interleaved::RankSelectInterleaved256;
pub use sparse::{RankSelectFew, RankSelectFewBuilder};
pub use mixed::{
    RankSelectMixedIL256, 
    RankSelectMixedSE512, 
    RankSelectMixedXL256,
    MixedDimensionView,
};
pub use simd::{SimdOps, bulk_rank1_simd, bulk_select1_simd, bulk_popcount_simd, SimdCapabilities};

/// Common trait for all rank/select operations
///
/// This trait provides a unified interface for rank and select operations
/// across all variants, enabling generic algorithms and benchmarking.
pub trait RankSelectOps {
    /// Count the number of set bits up to (but not including) the given position
    ///
    /// # Arguments
    /// * `pos` - The position to count up to (exclusive)
    ///
    /// # Returns
    /// The number of set bits in the range [0, pos)
    fn rank1(&self, pos: usize) -> usize;

    /// Count the number of clear bits up to (but not including) the given position
    ///
    /// # Arguments
    /// * `pos` - The position to count up to (exclusive)
    ///
    /// # Returns
    /// The number of clear bits in the range [0, pos)
    fn rank0(&self, pos: usize) -> usize;

    /// Find the position of the k-th set bit (0-indexed)
    ///
    /// # Arguments
    /// * `k` - The 0-based index of the set bit to find
    ///
    /// # Returns
    /// The position of the k-th set bit, or an error if k >= total set bits
    fn select1(&self, k: usize) -> Result<usize>;

    /// Find the position of the k-th clear bit (0-indexed)
    ///
    /// # Arguments
    /// * `k` - The 0-based index of the clear bit to find
    ///
    /// # Returns
    /// The position of the k-th clear bit, or an error if k >= total clear bits
    fn select0(&self, k: usize) -> Result<usize>;

    /// Get the length of the underlying bit vector
    fn len(&self) -> usize;

    /// Check if the bit vector is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the total number of set bits
    fn count_ones(&self) -> usize;

    /// Get the total number of clear bits
    fn count_zeros(&self) -> usize {
        self.len() - self.count_ones()
    }

    /// Get the bit at the specified position
    fn get(&self, index: usize) -> Option<bool>;

    /// Get space overhead as a percentage of the original bit vector
    fn space_overhead_percent(&self) -> f64;
}

/// Extended trait for performance-optimized operations
///
/// This trait provides access to hardware-accelerated implementations
/// and performance monitoring capabilities.
pub trait RankSelectPerformanceOps: RankSelectOps {
    /// Hardware-accelerated rank using POPCNT instruction when available
    fn rank1_hardware_accelerated(&self, pos: usize) -> usize;

    /// Hardware-accelerated select using BMI2 instructions when available
    fn select1_hardware_accelerated(&self, k: usize) -> Result<usize>;

    /// Adaptive rank method - chooses best available implementation
    fn rank1_adaptive(&self, pos: usize) -> usize;

    /// Adaptive select method - chooses best available implementation
    fn select1_adaptive(&self, k: usize) -> Result<usize>;

    /// Bulk rank operations for multiple positions
    fn rank1_bulk(&self, positions: &[usize]) -> Vec<usize>;

    /// Bulk select operations for multiple indices
    fn select1_bulk(&self, indices: &[usize]) -> Result<Vec<usize>>;
}

/// Trait for multi-dimensional rank/select operations
///
/// This trait enables operations on multiple related bit vectors
/// stored in a single data structure for cache efficiency.
pub trait RankSelectMultiDimensional<const ARITY: usize>: RankSelectOps {
    /// Get a view of a specific dimension
    fn dimension<const D: usize>(&self) -> MixedDimensionView<'_, D>
    where
        [(); ARITY]: Sized;

    /// Rank operation on a specific dimension
    fn rank1_dim<const D: usize>(&self, pos: usize) -> usize
    where
        [(); ARITY]: Sized;

    /// Select operation on a specific dimension
    fn select1_dim<const D: usize>(&self, k: usize) -> Result<usize>
    where
        [(); ARITY]: Sized;

    /// Get the number of dimensions
    fn arity(&self) -> usize {
        ARITY
    }
}

/// Trait for sparse rank/select optimizations
///
/// This trait provides specialized operations for sparse bit vectors
/// where one value (0 or 1) is much more common than the other.
pub trait RankSelectSparse: RankSelectOps {
    /// The pivot value that is stored sparsely (true for 1s, false for 0s)
    const PIVOT: bool;

    /// Get compression ratio (actual size / uncompressed size)
    fn compression_ratio(&self) -> f64;

    /// Get the number of stored sparse elements
    fn sparse_count(&self) -> usize;

    /// Check if a position contains the sparse element
    fn contains_sparse(&self, pos: usize) -> bool;

    /// Get all sparse positions in a range
    fn sparse_positions_in_range(&self, start: usize, end: usize) -> Vec<usize>;
}

/// Builder trait for constructing rank/select structures
///
/// This trait provides a uniform interface for building rank/select
/// structures from various sources with optional optimizations.
pub trait RankSelectBuilder<T> {
    /// Build from a bit vector
    fn from_bit_vector(bit_vector: BitVector) -> Result<T>;

    /// Build from an iterator of booleans
    fn from_iter<I>(iter: I) -> Result<T>
    where
        I: IntoIterator<Item = bool>;

    /// Build from a byte slice with bit interpretation
    fn from_bytes(bytes: &[u8], bit_len: usize) -> Result<T>;

    /// Build with specific optimization hints
    fn with_optimizations(bit_vector: BitVector, opts: BuilderOptions) -> Result<T>;
}

/// Options for controlling rank/select construction
#[derive(Debug, Clone)]
pub struct BuilderOptions {
    /// Enable select optimization (build select cache)
    pub optimize_select: bool,
    /// Block size for rank cache (256, 512, 1024, etc.)
    pub block_size: usize,
    /// Select sampling rate (every N set bits)
    pub select_sample_rate: usize,
    /// Enable SIMD optimizations during construction
    pub enable_simd: bool,
    /// Prefer space efficiency over speed
    pub prefer_space: bool,
}

impl Default for BuilderOptions {
    fn default() -> Self {
        Self {
            optimize_select: true,
            block_size: 256,
            select_sample_rate: 512,
            enable_simd: true,
            prefer_space: false,
        }
    }
}

/// Performance statistics for rank/select operations
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Total number of rank operations performed
    pub rank_operations: u64,
    /// Total number of select operations performed
    pub select_operations: u64,
    /// Number of hardware-accelerated operations
    pub hardware_accelerated_ops: u64,
    /// Average rank operation time in nanoseconds
    pub avg_rank_time_ns: f64,
    /// Average select operation time in nanoseconds
    pub avg_select_time_ns: f64,
    /// Cache hit rate for block lookups
    pub cache_hit_rate: f64,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            rank_operations: 0,
            select_operations: 0,
            hardware_accelerated_ops: 0,
            avg_rank_time_ns: 0.0,
            avg_select_time_ns: 0.0,
            cache_hit_rate: 1.0,
        }
    }
}

/// Error types specific to rank/select operations
#[derive(Debug, Clone)]
pub enum RankSelectError {
    /// Invalid index for select operation
    InvalidSelectIndex { index: usize, max_valid: usize },
    /// Invalid position for rank operation
    InvalidRankPosition { position: usize, max_valid: usize },
    /// Dimension index out of bounds for multi-dimensional operations
    InvalidDimension { dimension: usize, max_valid: usize },
    /// Sparse data construction failed
    SparseConstructionFailed(String),
    /// SIMD operation not supported on this platform
    SimdNotSupported(String),
    /// Builder configuration error
    BuilderError(String),
}

impl fmt::Display for RankSelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RankSelectError::InvalidSelectIndex { index, max_valid } => {
                write!(f, "Invalid select index {}, maximum valid index is {}", index, max_valid)
            }
            RankSelectError::InvalidRankPosition { position, max_valid } => {
                write!(f, "Invalid rank position {}, maximum valid position is {}", position, max_valid)
            }
            RankSelectError::InvalidDimension { dimension, max_valid } => {
                write!(f, "Invalid dimension {}, maximum valid dimension is {}", dimension, max_valid)
            }
            RankSelectError::SparseConstructionFailed(msg) => {
                write!(f, "Sparse rank/select construction failed: {}", msg)
            }
            RankSelectError::SimdNotSupported(msg) => {
                write!(f, "SIMD operation not supported: {}", msg)
            }
            RankSelectError::BuilderError(msg) => {
                write!(f, "Builder error: {}", msg)
            }
        }
    }
}

impl std::error::Error for RankSelectError {}

/// Convert RankSelectError to ZiporaError
impl From<RankSelectError> for ZiporaError {
    fn from(err: RankSelectError) -> Self {
        ZiporaError::invalid_data(err.to_string())
    }
}

/// Utility functions for rank/select operations
pub mod utils {
    use super::*;

    /// Calculate optimal block size based on data characteristics
    pub fn optimal_block_size(total_bits: usize, density: f64) -> usize {
        match (total_bits, density) {
            (_, d) if d < 0.01 => 1024,  // Very sparse - larger blocks
            (_, d) if d > 0.9 => 256,   // Very dense - smaller blocks
            (n, _) if n < 10_000 => 256, // Small data - smaller blocks
            _ => 512,                    // Default medium blocks
        }
    }

    /// Calculate optimal select sampling rate based on data characteristics
    pub fn optimal_select_sample_rate(total_ones: usize) -> usize {
        match total_ones {
            0..=1000 => 64,
            1001..=10_000 => 256,
            10_001..=100_000 => 512,
            _ => 1024,
        }
    }

    /// Estimate memory overhead for a given configuration
    pub fn estimate_memory_overhead(
        total_bits: usize,
        block_size: usize,
        select_sample_rate: usize,
        enable_select_cache: bool,
    ) -> f64 {
        let num_blocks = (total_bits + block_size - 1) / block_size;
        let rank_overhead = num_blocks * 4; // 4 bytes per block

        let select_overhead = if enable_select_cache {
            let estimated_ones = total_bits / 2; // Assume 50% density
            let num_samples = (estimated_ones + select_sample_rate - 1) / select_sample_rate;
            num_samples * 4 // 4 bytes per sample
        } else {
            0
        };

        let total_overhead_bits = (rank_overhead + select_overhead) * 8;
        (total_overhead_bits as f64 / total_bits as f64) * 100.0
    }

    /// Benchmark different rank/select implementations
    pub fn benchmark_variants(bit_vector: &BitVector, iterations: usize) -> Vec<(String, f64, f64)> {
        // This would contain actual benchmarking code
        // For now, return placeholder data
        vec![
            ("RankSelectSimple".to_string(), 100.0, 1000.0),
            ("RankSelectSeparated256".to_string(), 10.0, 100.0),
            ("RankSelectInterleaved256".to_string(), 8.0, 80.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bitvector(size: usize, pattern: fn(usize) -> bool) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(pattern(i)).unwrap();
        }
        bv
    }

    fn create_alternating_bitvector(size: usize) -> BitVector {
        create_test_bitvector(size, |i| i % 2 == 0)
    }

    fn create_sparse_bitvector(size: usize) -> BitVector {
        create_test_bitvector(size, |i| i % 100 == 0)
    }

    fn create_dense_bitvector(size: usize) -> BitVector {
        create_test_bitvector(size, |i| i % 4 != 3)
    }

    #[test]
    fn test_builder_options_default() {
        let opts = BuilderOptions::default();
        assert!(opts.optimize_select);
        assert_eq!(opts.block_size, 256);
        assert_eq!(opts.select_sample_rate, 512);
        assert!(opts.enable_simd);
        assert!(!opts.prefer_space);
    }

    #[test]
    fn test_performance_stats_default() {
        let stats = PerformanceStats::default();
        assert_eq!(stats.rank_operations, 0);
        assert_eq!(stats.select_operations, 0);
        assert_eq!(stats.hardware_accelerated_ops, 0);
        assert_eq!(stats.avg_rank_time_ns, 0.0);
        assert_eq!(stats.avg_select_time_ns, 0.0);
        assert_eq!(stats.cache_hit_rate, 1.0);
    }

    #[test]
    fn test_utility_functions() {
        // Test optimal block size calculation
        assert_eq!(utils::optimal_block_size(1000, 0.005), 1024); // Very sparse
        assert_eq!(utils::optimal_block_size(1000, 0.95), 256);   // Very dense
        assert_eq!(utils::optimal_block_size(5000, 0.5), 256);    // Small data
        assert_eq!(utils::optimal_block_size(100_000, 0.5), 512); // Default

        // Test optimal select sample rate
        assert_eq!(utils::optimal_select_sample_rate(500), 64);
        assert_eq!(utils::optimal_select_sample_rate(5000), 256);
        assert_eq!(utils::optimal_select_sample_rate(50_000), 512);
        assert_eq!(utils::optimal_select_sample_rate(500_000), 1024);

        // Test memory overhead estimation
        let overhead = utils::estimate_memory_overhead(10_000, 256, 512, true);
        assert!(overhead > 0.0 && overhead < 50.0); // Reasonable overhead
    }

    #[test]
    fn test_rank_select_error_display() {
        let err = RankSelectError::InvalidSelectIndex { index: 10, max_valid: 5 };
        assert!(err.to_string().contains("Invalid select index 10"));

        let err = RankSelectError::InvalidDimension { dimension: 3, max_valid: 1 };
        assert!(err.to_string().contains("Invalid dimension 3"));

        let err = RankSelectError::SparseConstructionFailed("test error".to_string());
        assert!(err.to_string().contains("Sparse rank/select construction failed"));
    }

    #[test]
    fn test_rank_select_error_conversion() {
        let rs_err = RankSelectError::BuilderError("test".to_string());
        let zipora_err: ZiporaError = rs_err.into();
        assert!(zipora_err.to_string().contains("Builder error"));
    }

    // Comprehensive tests for all rank/select variants
    #[test]
    fn test_all_variants_basic_operations() {
        let bv = create_alternating_bitvector(1000);
        
        // Test all basic variants
        let simple = RankSelectSimple::new(bv.clone()).unwrap();
        let separated256 = RankSelectSeparated256::new(bv.clone()).unwrap();
        let separated512 = RankSelectSeparated512::new(bv.clone()).unwrap();
        let interleaved = RankSelectInterleaved256::new(bv.clone()).unwrap();
        
        // Test basic operations consistency
        for pos in [0, 1, 100, 500, 999] {
            let expected_rank = simple.rank1(pos);
            
            assert_eq!(separated256.rank1(pos), expected_rank, "Separated256 rank mismatch at {}", pos);
            assert_eq!(separated512.rank1(pos), expected_rank, "Separated512 rank mismatch at {}", pos);
            assert_eq!(interleaved.rank1(pos), expected_rank, "Interleaved rank mismatch at {}", pos);
        }
        
        // Test select operations
        let ones_count = simple.count_ones();
        for k in [0, ones_count/4, ones_count/2, ones_count*3/4, ones_count-1] {
            if k < ones_count {
                let expected_select = simple.select1(k).unwrap();
                
                assert_eq!(separated256.select1(k).unwrap(), expected_select, "Separated256 select mismatch at {}", k);
                assert_eq!(separated512.select1(k).unwrap(), expected_select, "Separated512 select mismatch at {}", k);
                assert_eq!(interleaved.select1(k).unwrap(), expected_select, "Interleaved select mismatch at {}", k);
            }
        }
    }

    #[test]
    fn test_sparse_variant() {
        let sparse_bv = create_sparse_bitvector(10000);
        
        // Test sparse variant for 1s (should be sparse)
        let sparse_rs = RankSelectFew::<true, 64>::from_bit_vector(sparse_bv.clone()).unwrap();
        
        // Should achieve good compression
        assert!(sparse_rs.compression_ratio() < 0.5, "Sparse variant should achieve <50% compression ratio");
        
        // Test basic operations
        assert_eq!(sparse_rs.rank1(0), 0);
        assert_eq!(sparse_rs.rank1(100), 1);
        assert_eq!(sparse_rs.rank1(200), 2);
        
        if sparse_rs.count_ones() > 0 {
            assert_eq!(sparse_rs.select1(0).unwrap(), 0);
        }
        
        // Test sparse element detection
        assert!(sparse_rs.contains_sparse(0));
        assert!(!sparse_rs.contains_sparse(1));
        assert!(sparse_rs.contains_sparse(100));
    }

    #[test]
    fn test_mixed_variants() {
        let bv1 = create_alternating_bitvector(1000);
        let bv2 = create_dense_bitvector(1000);
        
        // Test dual-dimension variants
        let mixed_il = RankSelectMixedIL256::new([bv1.clone(), bv2.clone()]).unwrap();
        let mixed_se = RankSelectMixedSE512::new([bv1.clone(), bv2.clone()]).unwrap();
        
        // Test dimension operations
        for pos in [0, 100, 500, 999] {
            let rank0_dim0 = mixed_il.rank1_dimension(pos, 0);
            let rank0_dim1 = mixed_il.rank1_dimension(pos, 1);
            
            let rank1_dim0 = mixed_se.rank1_dimension(pos, 0);
            let rank1_dim1 = mixed_se.rank1_dimension(pos, 1);
            
            // Should match individual bit vector ranks
            assert_eq!(rank0_dim0, bv1.rank1(pos), "Mixed IL dimension 0 rank mismatch");
            assert_eq!(rank0_dim1, bv2.rank1(pos), "Mixed IL dimension 1 rank mismatch");
            assert_eq!(rank1_dim0, bv1.rank1(pos), "Mixed SE dimension 0 rank mismatch");
            assert_eq!(rank1_dim1, bv2.rank1(pos), "Mixed SE dimension 1 rank mismatch");
        }
        
        // Test select operations
        let ones0 = bv1.count_ones();
        let ones1 = bv2.count_ones();
        
        if ones0 > 0 {
            let select_il = mixed_il.select1_dimension(0, 0).unwrap();
            let select_se = mixed_se.select1_dimension(0, 0).unwrap();
            assert_eq!(select_il, select_se, "Mixed variants select mismatch");
        }
        
        if ones1 > 0 {
            let select_il = mixed_il.select1_dimension(0, 1).unwrap();
            let select_se = mixed_se.select1_dimension(0, 1).unwrap();
            assert_eq!(select_il, select_se, "Mixed variants select mismatch");
        }
    }

    #[test]
    fn test_multi_dimensional_variant() {
        let bv1 = create_alternating_bitvector(500);
        let bv2 = create_sparse_bitvector(500);
        let bv3 = create_dense_bitvector(500);
        
        // Test 3D variant
        let mixed_xl = RankSelectMixedXL256::<3>::new([bv1.clone(), bv2.clone(), bv3.clone()]).unwrap();
        
        // Test all dimensions using generic syntax
        for pos in [0, 100, 250, 499] {
            assert_eq!(mixed_xl.rank1_dimension::<0>(pos), bv1.rank1(pos), "XL256 dimension 0 rank mismatch");
            assert_eq!(mixed_xl.rank1_dimension::<1>(pos), bv2.rank1(pos), "XL256 dimension 1 rank mismatch");
            assert_eq!(mixed_xl.rank1_dimension::<2>(pos), bv3.rank1(pos), "XL256 dimension 2 rank mismatch");
        }
        
        // Test select operations on different dimensions
        if bv1.count_ones() > 0 {
            let select_result = mixed_xl.select1_dimension::<0>(0);
            assert!(select_result.is_ok());
        }
        
        // Test intersection analysis
        let dimensions_to_intersect = [0, 1]; // Intersect dimensions 0 and 1
        let intersection = mixed_xl.find_intersection(&dimensions_to_intersect, 10).unwrap();
        assert!(intersection.len() <= 10); // Should find at most 10 intersection positions
    }

    #[test]
    fn test_simd_operations() {
        let bit_data: Vec<u64> = vec![
            0xAAAAAAAAAAAAAAAAu64, // Alternating bits
            0x5555555555555555u64, // Alternating bits (offset)
            0xFFFFFFFFFFFFFFFFu64, // All ones
            0x0000000000000000u64, // All zeros
        ];
        
        let positions = vec![0, 64, 128, 192, 256];
        let indices = vec![0, 10, 20, 30];
        
        // Test bulk operations
        let ranks = bulk_rank1_simd(&bit_data, &positions);
        assert_eq!(ranks.len(), positions.len());
        
        let selects = bulk_select1_simd(&bit_data, &indices);
        assert!(selects.is_ok());
        
        let popcounts = bulk_popcount_simd(&bit_data);
        assert_eq!(popcounts.len(), bit_data.len());
        assert_eq!(popcounts[0], 32); // Alternating pattern
        assert_eq!(popcounts[1], 32); // Alternating pattern
        assert_eq!(popcounts[2], 64); // All ones
        assert_eq!(popcounts[3], 0);  // All zeros
    }

    #[test]
    fn test_simd_capabilities() {
        let caps = SimdCapabilities::detect();
        
        // Should detect some capability
        assert!(caps.optimization_tier <= 5);
        assert!(caps.chunk_size > 0);
        assert!(caps.chunk_size <= 64 * 1024);
        
        // Test cached access
        let caps2 = SimdCapabilities::get();
        assert_eq!(caps.optimization_tier, caps2.optimization_tier);
    }

    #[test]
    fn test_performance_ops_trait() {
        let bv = create_alternating_bitvector(1000);
        let separated = RankSelectSeparated256::new(bv).unwrap();
        
        // Test performance operations
        for pos in [0, 100, 500, 999] {
            let standard_rank = separated.rank1(pos);
            let hw_rank = separated.rank1_hardware_accelerated(pos);
            let adaptive_rank = separated.rank1_adaptive(pos);
            
            assert_eq!(standard_rank, hw_rank, "Hardware rank mismatch");
            assert_eq!(standard_rank, adaptive_rank, "Adaptive rank mismatch");
        }
        
        // Test bulk operations
        let positions = vec![0, 100, 500, 999];
        let bulk_ranks = separated.rank1_bulk(&positions);
        
        for (i, &pos) in positions.iter().enumerate() {
            assert_eq!(bulk_ranks[i], separated.rank1(pos), "Bulk rank mismatch");
        }
        
        let ones_count = separated.count_ones();
        if ones_count > 0 {
            let indices = vec![0, ones_count/4, ones_count/2];
            let bulk_selects = separated.select1_bulk(&indices).unwrap();
            
            for (i, &k) in indices.iter().enumerate() {
                if k < ones_count {
                    assert_eq!(bulk_selects[i], separated.select1(k).unwrap(), "Bulk select mismatch");
                }
            }
        }
    }

    #[test]
    fn test_builder_patterns() {
        let bv = create_alternating_bitvector(1000);
        
        // Test builder with options
        let opts = BuilderOptions {
            optimize_select: true,
            block_size: 512,
            select_sample_rate: 256,
            enable_simd: true,
            prefer_space: false,
        };
        
        let separated = RankSelectSeparated256::with_optimizations(bv.clone(), opts).unwrap();
        
        // Should work correctly with custom options
        assert!(separated.len() > 0);
        assert_eq!(separated.rank1(0), 0);
        assert_eq!(separated.rank1(1), 1);
        
        // Test sparse construction
        let sparse_bv = create_sparse_bitvector(1000);
        let sparse_rs = RankSelectFew::<true, 64>::from_bit_vector(sparse_bv).unwrap();
        assert!(sparse_rs.compression_ratio() < 0.5);
    }

    #[test]
    fn test_space_overhead() {
        let bv = create_alternating_bitvector(10000);
        
        let simple = RankSelectSimple::new(bv.clone()).unwrap();
        let separated = RankSelectSeparated256::new(bv.clone()).unwrap();
        let sparse_bv = create_sparse_bitvector(10000);
        let sparse_rs = RankSelectFew::<true, 64>::from_bit_vector(sparse_bv).unwrap();
        
        // Check space overhead is reasonable
        assert!(simple.space_overhead_percent() < 30.0, "Simple variant overhead too high");
        assert!(separated.space_overhead_percent() < 30.0, "Separated variant overhead too high");
        assert!(sparse_rs.compression_ratio() < 0.5, "Sparse variant should compress well");
        
        println!("Simple overhead: {:.2}%", simple.space_overhead_percent());
        println!("Separated overhead: {:.2}%", separated.space_overhead_percent());
        println!("Sparse compression: {:.2}%", sparse_rs.compression_ratio() * 100.0);
    }

    #[test]
    fn test_large_dataset_consistency() {
        // Test with larger dataset to verify scalability
        let large_bv = create_test_bitvector(100000, |i| (i * 13 + 7) % 71 == 0);
        
        let simple = RankSelectSimple::new(large_bv.clone()).unwrap();
        let separated = RankSelectSeparated256::new(large_bv).unwrap();
        
        // Test consistency across implementations
        let test_positions = [0, 1000, 10000, 50000, 99999];
        for &pos in &test_positions {
            assert_eq!(simple.rank1(pos), separated.rank1(pos), "Large dataset rank mismatch at {}", pos);
        }
        
        let ones_count = simple.count_ones();
        let test_ks = [0, ones_count/10, ones_count/2, ones_count*9/10];
        for &k in &test_ks {
            if k < ones_count {
                assert_eq!(simple.select1(k).unwrap(), separated.select1(k).unwrap(), "Large dataset select mismatch at {}", k);
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        // Empty bit vector
        let empty_bv = BitVector::new();
        let empty_rs = RankSelectSimple::new(empty_bv).unwrap();
        assert_eq!(empty_rs.len(), 0);
        assert_eq!(empty_rs.count_ones(), 0);
        assert_eq!(empty_rs.rank1(0), 0);
        
        // Single bit
        let mut single_bv = BitVector::new();
        single_bv.push(true).unwrap();
        let single_rs = RankSelectSimple::new(single_bv).unwrap();
        assert_eq!(single_rs.len(), 1);
        assert_eq!(single_rs.count_ones(), 1);
        assert_eq!(single_rs.rank1(0), 0);
        assert_eq!(single_rs.rank1(1), 1);
        assert_eq!(single_rs.select1(0).unwrap(), 0);
        
        // All zeros
        let all_zeros = BitVector::with_size(1000, false).unwrap();
        let zeros_rs = RankSelectSimple::new(all_zeros).unwrap();
        assert_eq!(zeros_rs.count_ones(), 0);
        assert_eq!(zeros_rs.rank1(500), 0);
        assert!(zeros_rs.select1(0).is_err());
        
        // All ones
        let all_ones = BitVector::with_size(1000, true).unwrap();
        let ones_rs = RankSelectSimple::new(all_ones).unwrap();
        assert_eq!(ones_rs.count_ones(), 1000);
        assert_eq!(ones_rs.rank1(500), 500);
        assert_eq!(ones_rs.select1(499).unwrap(), 499);
    }
}