//! Specialized algorithms for high-performance data processing
//!
//! This module provides implementations of advanced algorithms commonly used
//! in data compression, indexing, and sorting applications.

pub mod cache_oblivious;
pub mod external_sort;
pub mod multiway_merge;
pub mod radix_sort;
pub mod set_operations;
pub mod simd_merge;
pub mod suffix_array;
pub mod tournament_tree;

// Re-export main types
pub use cache_oblivious::{
    AdaptiveAlgorithmSelector, CacheObliviousConfig, CacheObliviousSort,
    DataCharacteristics as CacheObliviousDataCharacteristics, 
    CacheObliviousSortingStrategy, VanEmdeBoas
};
pub use external_sort::{ExternalSort, ReplaceSelectSort, ReplaceSelectSortConfig};
pub use multiway_merge::{MergeSource, MultiWayMerge};
pub use radix_sort::{
    AdvancedRadixSort, AdvancedRadixSortConfig, AdvancedStringRadixSort, AdvancedU32RadixSort, 
    AdvancedU64RadixSort, CpuFeatures, DataCharacteristics, RadixSort, RadixSortConfig, 
    RadixSortable, SortingStrategy as RadixSortingStrategy
};
pub use set_operations::{SetOperations, SetOperationsConfig, SetOperationStats};
pub use simd_merge::{SimdComparator, SimdConfig, SimdOperations};
pub use suffix_array::{LcpArray, SuffixArray, SuffixArrayBuilder};
pub use tournament_tree::{EnhancedLoserTree, LoserTreeConfig, TournamentNode, CacheAlignedNode};

/// Configuration for algorithm behavior
#[derive(Debug, Clone)]
pub struct AlgorithmConfig {
    /// Enable SIMD optimizations when available
    pub use_simd: bool,
    /// Use parallel processing for large datasets
    pub use_parallel: bool,
    /// Threshold for switching to parallel processing
    pub parallel_threshold: usize,
    /// Memory budget for algorithms in bytes
    pub memory_budget: usize,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            use_simd: cfg!(feature = "simd"),
            use_parallel: true,
            parallel_threshold: 10_000,
            memory_budget: 64 * 1024 * 1024, // 64MB
        }
    }
}

/// Performance statistics for algorithm execution
#[derive(Debug, Clone)]
pub struct AlgorithmStats {
    /// Total items processed
    pub items_processed: usize,
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Memory used in bytes
    pub memory_used: usize,
    /// Whether parallel processing was used
    pub used_parallel: bool,
    /// Whether SIMD optimizations were used
    pub used_simd: bool,
}

impl AlgorithmStats {
    /// Calculate processing rate in items per second
    pub fn items_per_second(&self) -> f64 {
        if self.processing_time_us == 0 {
            return 0.0;
        }
        (self.items_processed as f64) / (self.processing_time_us as f64 / 1_000_000.0)
    }

    /// Calculate memory efficiency in items per byte
    pub fn items_per_byte(&self) -> f64 {
        if self.memory_used == 0 {
            return 0.0;
        }
        self.items_processed as f64 / self.memory_used as f64
    }
}

/// Trait for algorithms that can be benchmarked and configured
pub trait Algorithm {
    /// Configuration type for this algorithm
    type Config;

    /// Input type for this algorithm
    type Input;

    /// Output type for this algorithm
    type Output;

    /// Execute the algorithm with the given configuration and input
    fn execute(&self, config: &Self::Config, input: Self::Input) -> crate::Result<Self::Output>;

    /// Get performance statistics from the last execution
    fn stats(&self) -> AlgorithmStats;

    /// Estimate memory requirements for the given input size
    fn estimate_memory(&self, input_size: usize) -> usize;

    /// Check if the algorithm supports parallel execution
    fn supports_parallel(&self) -> bool {
        false
    }

    /// Check if the algorithm supports SIMD optimizations
    fn supports_simd(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_config_default() {
        let config = AlgorithmConfig::default();
        assert_eq!(config.parallel_threshold, 10_000);
        assert_eq!(config.memory_budget, 64 * 1024 * 1024);

        #[cfg(feature = "simd")]
        assert!(config.use_simd);

        #[cfg(not(feature = "simd"))]
        assert!(!config.use_simd);
    }

    #[test]
    fn test_algorithm_stats() {
        let stats = AlgorithmStats {
            items_processed: 1000,
            processing_time_us: 1000, // 1ms
            memory_used: 1024,
            used_parallel: false,
            used_simd: false,
        };

        assert_eq!(stats.items_per_second(), 1_000_000.0); // 1M items/sec
        assert_eq!(stats.items_per_byte(), 1000.0 / 1024.0);
    }

    #[test]
    fn test_algorithm_stats_edge_cases() {
        let stats = AlgorithmStats {
            items_processed: 1000,
            processing_time_us: 0,
            memory_used: 0,
            used_parallel: false,
            used_simd: false,
        };

        assert_eq!(stats.items_per_second(), 0.0);
        assert_eq!(stats.items_per_byte(), 0.0);
    }
}
