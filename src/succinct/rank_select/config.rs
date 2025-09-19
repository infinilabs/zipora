//! Configuration Builder Pattern for Separated Storage Rank/Select Structures
//!
//! This module provides a comprehensive configuration system for all separated storage
//! variants, inspired by advanced design patterns. It allows fine-grained
//! control over memory layout, hardware acceleration, caching strategies, and performance
//! trade-offs.
//!
//! # Design Philosophy
//!
//! The configuration system follows these principles from advanced research:
//! - **Adaptive Strategy Selection**: Automatically choose optimal settings based on data characteristics
//! - **Hardware Acceleration Integration**: Seamless BMI2, POPCNT, and SIMD support
//! - **Memory Layout Optimization**: Separated vs interleaved storage with cache-aware design  
//! - **Multi-dimensional Support**: Unified configuration for 2-4 dimensional bit vectors
//! - **Hierarchical Rank Caching**: Bit-packed relative ranks for space efficiency
//!
//! # Examples
//!
//! ```rust
//! use zipora::succinct::rank_select::{
//!     SeparatedStorageConfig,
//!     RankSelectInterleaved256,
//!     AdaptiveRankSelect,
//!     RankSelectOps
//! };
//! use zipora::succinct::BitVector;
//!
//! // High-performance configuration for large datasets
//! let mut bit_vector = BitVector::new();
//! for i in 0..1000 {
//!     bit_vector.push(i % 3 == 0)?;
//! }
//!
//! let config = SeparatedStorageConfig::new()
//!     .block_size(512)
//!     .enable_select_acceleration(true)
//!     .enable_hardware_acceleration(true)
//!     .superblock_size(32)
//!     .select_sample_rate(256)
//!     .build();
//!
//! // Use best-performing implementation with configuration hints
//! let rs = RankSelectInterleaved256::new(bit_vector.clone())?;
//! let rank = rs.rank1(500);
//! let pos = rs.select1(100)?;
//!
//! // Adaptive selection - automatically chooses optimal implementation
//! let adaptive_rs = AdaptiveRankSelect::new(bit_vector)?;
//! println!("Selected: {}", adaptive_rs.implementation_name());
//! # Ok::<(), zipora::error::ZiporaError>(())
//! ```

use crate::error::{Result, ZiporaError};
use std::fmt;

/// Comprehensive configuration for separated storage rank/select structures
///
/// This configuration system provides fine-grained control over all aspects of
/// separated storage implementations, including memory layout, hardware acceleration,
/// caching strategies, and multi-dimensional support.
#[derive(Debug, Clone)]
pub struct SeparatedStorageConfig {
    /// Primary block size for rank caching (256, 512, 1024)
    pub block_size: usize,
    /// Enable select acceleration with dedicated select cache
    pub enable_select_acceleration: bool,
    /// Select sampling rate (every N set bits get cached)
    pub select_sample_rate: usize,
    /// Enable hardware acceleration (BMI2, POPCNT, SIMD)
    pub enable_hardware_acceleration: bool,
    /// Use bit-packed hierarchical rank caching
    pub enable_bit_packed_ranks: bool,
    /// Number of blocks per superblock for hierarchical caching
    pub superblock_size: usize,
    /// Number of bits per relative rank in bit-packed mode (7-9 recommended)
    pub relative_rank_bits: usize,
    /// Storage layout strategy
    pub storage_layout: StorageLayout,
    /// Memory optimization strategy
    pub memory_strategy: MemoryStrategy,
    /// Cache alignment strategy
    pub cache_alignment: CacheAlignment,
    /// Multi-dimensional configuration
    pub multi_dimensional: Option<MultiDimensionalConfig>,
    /// Hardware-specific optimizations
    pub hardware_optimizations: HardwareOptimizations,
    /// Performance tuning parameters
    pub performance_tuning: PerformanceTuning,
}

/// Storage layout strategy for bit data and rank caches
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageLayout {
    /// Interleaved: Rank cache and bit data stored together for cache locality
    Interleaved,
    /// Separated: Rank cache and bit data stored separately for flexibility
    Separated,
    /// Mixed: Adaptive layout based on access patterns
    Mixed,
    /// Hierarchical: Multi-level caching with different layouts per level
    Hierarchical,
}

/// Memory optimization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStrategy {
    /// Optimize for minimum space usage
    MinimizeSpace,
    /// Balance space and performance
    Balanced,
    /// Optimize for maximum performance
    MaximizePerformance,
    /// Adaptive based on data characteristics
    Adaptive,
}

/// Cache alignment strategy for optimal memory access patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAlignment {
    /// Align to 32-byte cache lines
    CacheLine32,
    /// Align to 64-byte cache lines (most common)
    CacheLine64,
    /// Align to 128-byte cache lines (some modern CPUs)
    CacheLine128,
    /// Adaptive alignment based on CPU detection
    Adaptive,
}

/// Configuration for multi-dimensional rank/select structures
#[derive(Debug, Clone)]
pub struct MultiDimensionalConfig {
    /// Number of dimensions (2-4)
    pub arity: usize,
    /// Per-dimension optimization hints
    pub dimension_hints: Vec<DimensionHint>,
    /// Enable cross-dimension correlation analysis
    pub enable_correlation_analysis: bool,
    /// Shared vs separate caching strategy
    pub cache_sharing_strategy: CacheSharingStrategy,
}

/// Optimization hint for a specific dimension
#[derive(Debug, Clone)]
pub struct DimensionHint {
    /// Expected access frequency for this dimension
    pub access_frequency: AccessFrequency,
    /// Expected data density for this dimension
    pub data_density: DataDensity,
    /// Preferred select cache density for this dimension
    pub select_cache_density: SelectCacheDensity,
}

/// Access frequency hint for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessFrequency {
    /// Very frequent access - optimize for speed
    VeryHigh,
    /// Frequent access - balance speed and space
    High,
    /// Moderate access - default optimizations
    Medium,
    /// Infrequent access - optimize for space
    Low,
    /// Very infrequent access - minimize space
    VeryLow,
}

/// Data density hint for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataDensity {
    /// Very dense (>80% ones)
    VeryDense,
    /// Dense (60-80% ones)
    Dense,
    /// Balanced (40-60% ones)
    Balanced,
    /// Sparse (20-40% ones)
    Sparse,
    /// Very sparse (<20% ones)
    VerySparse,
}

/// Select cache density configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectCacheDensity {
    /// No select cache (space-optimized)
    None,
    /// Sparse select cache (every 1024 bits)
    Sparse,
    /// Normal select cache (every 512 bits)
    Normal,
    /// Dense select cache (every 256 bits)
    Dense,
    /// Very dense select cache (every 128 bits)
    VeryDense,
}

/// Cache sharing strategy for multi-dimensional structures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSharingStrategy {
    /// Separate caches per dimension
    Separate,
    /// Shared rank cache, separate select caches
    SharedRank,
    /// Fully shared caches
    FullyShared,
    /// Adaptive sharing based on access patterns
    Adaptive,
}

/// Hardware acceleration configuration
#[derive(Debug, Clone)]
pub struct HardwareOptimizations {
    /// Enable BMI2 instructions (PDEP/PEXT) for select acceleration
    pub enable_bmi2: bool,
    /// Enable BMI1 instructions (LZCNT/TZCNT/POPCNT)
    pub enable_bmi1: bool,
    /// Enable SIMD instructions (SSE/AVX)
    pub enable_simd: bool,
    /// Enable AVX-512 (requires nightly Rust)
    pub enable_avx512: bool,
    /// Enable prefetching hints
    pub enable_prefetch: bool,
    /// CPU feature detection mode
    pub feature_detection: FeatureDetection,
}

/// CPU feature detection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureDetection {
    /// Runtime detection (recommended)
    Runtime,
    /// Compile-time detection
    CompileTime,
    /// Force enable all features
    ForceEnable,
    /// Disable all hardware acceleration
    Disable,
}

/// Performance tuning parameters
#[derive(Debug, Clone)]
pub struct PerformanceTuning {
    /// Prefetch distance for rank operations
    pub rank_prefetch_distance: usize,
    /// Prefetch distance for select operations
    pub select_prefetch_distance: usize,
    /// Enable branch prediction optimization
    pub optimize_branch_prediction: bool,
    /// Enable loop unrolling for bulk operations
    pub enable_loop_unrolling: bool,
    /// Target cache level for optimization (L1, L2, L3)
    pub target_cache_level: CacheLevel,
}

/// Target cache level for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    /// Optimize for L1 cache (32KB typical)
    L1,
    /// Optimize for L2 cache (256KB typical)
    L2,
    /// Optimize for L3 cache (several MB typical)
    L3,
    /// Adaptive based on dataset size
    Adaptive,
}

impl SeparatedStorageConfig {
    /// Create a new configuration with default settings
    pub fn new() -> SeparatedStorageConfigBuilder {
        SeparatedStorageConfigBuilder::new()
    }

    /// Create configuration optimized for high performance
    pub fn high_performance() -> SeparatedStorageConfigBuilder {
        SeparatedStorageConfigBuilder::new()
            .block_size(512)
            .enable_select_acceleration(true)
            .enable_hardware_acceleration(true)
            .memory_strategy(MemoryStrategy::MaximizePerformance)
            .enable_bit_packed_ranks(true)
            .superblock_size(32)
            .select_sample_rate(256)
    }

    /// Create configuration optimized for low memory usage
    pub fn low_memory() -> SeparatedStorageConfigBuilder {
        SeparatedStorageConfigBuilder::new()
            .block_size(1024)
            .enable_select_acceleration(false)
            .memory_strategy(MemoryStrategy::MinimizeSpace)
            .enable_bit_packed_ranks(true)
            .superblock_size(64)
            .select_sample_rate(1024)
    }

    /// Create configuration optimized for multi-dimensional data
    pub fn multi_dimensional(arity: usize) -> SeparatedStorageConfigBuilder {
        SeparatedStorageConfigBuilder::new()
            .block_size(256)
            .enable_select_acceleration(true)
            .enable_bit_packed_ranks(true)
            .superblock_size(16)
            .multi_dimensional_arity(arity)
            .memory_strategy(MemoryStrategy::Balanced)
    }

    /// Validate the configuration for consistency
    pub fn validate(&self) -> Result<()> {
        // Validate block size
        if !matches!(self.block_size, 256 | 512 | 1024 | 2048) {
            return Err(ZiporaError::invalid_data(format!(
                "Invalid block size {}. Must be 256, 512, 1024, or 2048",
                self.block_size
            )));
        }

        // Validate relative rank bits
        if self.enable_bit_packed_ranks && (self.relative_rank_bits < 7 || self.relative_rank_bits > 12) {
            return Err(ZiporaError::invalid_data(format!(
                "Invalid relative_rank_bits {}. Must be between 7 and 12",
                self.relative_rank_bits
            )));
        }

        // Validate superblock size
        if self.superblock_size == 0 || self.superblock_size > 1024 {
            return Err(ZiporaError::invalid_data(format!(
                "Invalid superblock_size {}. Must be between 1 and 1024",
                self.superblock_size
            )));
        }

        // Validate multi-dimensional configuration
        if let Some(ref multi_config) = self.multi_dimensional {
            if multi_config.arity < 2 || multi_config.arity > 4 {
                return Err(ZiporaError::invalid_data(format!(
                    "Invalid multi-dimensional arity {}. Must be between 2 and 4",
                    multi_config.arity
                )));
            }
            
            if multi_config.dimension_hints.len() != multi_config.arity {
                return Err(ZiporaError::invalid_data(format!(
                    "Dimension hints count {} does not match arity {}",
                    multi_config.dimension_hints.len(),
                    multi_config.arity
                )));
            }
        }

        Ok(())
    }

    /// Calculate expected memory overhead percentage
    pub fn estimated_memory_overhead(&self) -> f64 {
        let mut overhead = 0.0;

        // Base rank cache overhead
        overhead += match self.block_size {
            256 => 25.0,
            512 => 12.5,
            1024 => 6.25,
            2048 => 3.125,
            _ => 25.0,
        };

        // Select cache overhead
        if self.enable_select_acceleration {
            overhead += 100.0 / self.select_sample_rate as f64;
        }

        // Bit-packed ranks reduce overhead
        if self.enable_bit_packed_ranks {
            overhead *= 0.6; // ~40% reduction
        }

        // Multi-dimensional overhead
        if let Some(ref multi_config) = self.multi_dimensional {
            overhead *= multi_config.arity as f64;
        }

        overhead
    }

    /// Get recommended select sample rate based on configuration
    pub fn recommended_select_sample_rate(&self) -> usize {
        match self.memory_strategy {
            MemoryStrategy::MinimizeSpace => 1024,
            MemoryStrategy::Balanced => 512,
            MemoryStrategy::MaximizePerformance => 256,
            MemoryStrategy::Adaptive => {
                if self.enable_bit_packed_ranks { 512 } else { 256 }
            }
        }
    }
}

impl Default for SeparatedStorageConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            enable_select_acceleration: true,
            select_sample_rate: 512,
            enable_hardware_acceleration: true,
            enable_bit_packed_ranks: false,
            superblock_size: 16,
            relative_rank_bits: 9,
            storage_layout: StorageLayout::Separated,
            memory_strategy: MemoryStrategy::Balanced,
            cache_alignment: CacheAlignment::CacheLine64,
            multi_dimensional: None,
            hardware_optimizations: HardwareOptimizations::default(),
            performance_tuning: PerformanceTuning::default(),
        }
    }
}

impl Default for HardwareOptimizations {
    fn default() -> Self {
        Self {
            enable_bmi2: true,
            enable_bmi1: true,
            enable_simd: true,
            enable_avx512: false,
            enable_prefetch: true,
            feature_detection: FeatureDetection::Runtime,
        }
    }
}

impl Default for PerformanceTuning {
    fn default() -> Self {
        Self {
            rank_prefetch_distance: 2,
            select_prefetch_distance: 1,
            optimize_branch_prediction: true,
            enable_loop_unrolling: true,
            target_cache_level: CacheLevel::L2,
        }
    }
}

impl Default for DimensionHint {
    fn default() -> Self {
        Self {
            access_frequency: AccessFrequency::Medium,
            data_density: DataDensity::Balanced,
            select_cache_density: SelectCacheDensity::Normal,
        }
    }
}

/// Builder pattern for constructing SeparatedStorageConfig
pub struct SeparatedStorageConfigBuilder {
    config: SeparatedStorageConfig,
}

impl SeparatedStorageConfigBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            config: SeparatedStorageConfig::default(),
        }
    }

    /// Set the block size for rank caching
    pub fn block_size(mut self, size: usize) -> Self {
        self.config.block_size = size;
        self
    }

    /// Enable or disable select acceleration
    pub fn enable_select_acceleration(mut self, enable: bool) -> Self {
        self.config.enable_select_acceleration = enable;
        self
    }

    /// Set the select sampling rate
    pub fn select_sample_rate(mut self, rate: usize) -> Self {
        self.config.select_sample_rate = rate;
        self
    }

    /// Enable or disable hardware acceleration
    pub fn enable_hardware_acceleration(mut self, enable: bool) -> Self {
        self.config.enable_hardware_acceleration = enable;
        self
    }

    /// Enable or disable bit-packed hierarchical ranks
    pub fn enable_bit_packed_ranks(mut self, enable: bool) -> Self {
        self.config.enable_bit_packed_ranks = enable;
        self
    }

    /// Set the superblock size for hierarchical caching
    pub fn superblock_size(mut self, size: usize) -> Self {
        self.config.superblock_size = size;
        self
    }

    /// Set the number of bits per relative rank
    pub fn relative_rank_bits(mut self, bits: usize) -> Self {
        self.config.relative_rank_bits = bits;
        self
    }

    /// Set the storage layout strategy
    pub fn storage_layout(mut self, layout: StorageLayout) -> Self {
        self.config.storage_layout = layout;
        self
    }

    /// Set the memory optimization strategy
    pub fn memory_strategy(mut self, strategy: MemoryStrategy) -> Self {
        self.config.memory_strategy = strategy;
        self
    }

    /// Set the cache alignment strategy
    pub fn cache_alignment(mut self, alignment: CacheAlignment) -> Self {
        self.config.cache_alignment = alignment;
        self
    }

    /// Configure for multi-dimensional data with specified arity
    pub fn multi_dimensional_arity(mut self, arity: usize) -> Self {
        let hints = vec![DimensionHint::default(); arity];
        self.config.multi_dimensional = Some(MultiDimensionalConfig {
            arity,
            dimension_hints: hints,
            enable_correlation_analysis: false,
            cache_sharing_strategy: CacheSharingStrategy::Separate,
        });
        self
    }

    /// Configure multi-dimensional with detailed settings
    pub fn multi_dimensional_config(mut self, config: MultiDimensionalConfig) -> Self {
        self.config.multi_dimensional = Some(config);
        self
    }

    /// Set hardware optimization settings
    pub fn hardware_optimizations(mut self, opts: HardwareOptimizations) -> Self {
        self.config.hardware_optimizations = opts;
        self
    }

    /// Set performance tuning parameters
    pub fn performance_tuning(mut self, tuning: PerformanceTuning) -> Self {
        self.config.performance_tuning = tuning;
        self
    }

    /// Optimize configuration for space efficiency
    pub fn optimize_for_space(mut self) -> Self {
        self.config.memory_strategy = MemoryStrategy::MinimizeSpace;
        self.config.enable_select_acceleration = false;
        self.config.select_sample_rate = 1024;
        self.config.enable_bit_packed_ranks = true;
        self.config.superblock_size = 64;
        self
    }

    /// Optimize configuration for maximum performance
    pub fn optimize_for_performance(mut self) -> Self {
        self.config.memory_strategy = MemoryStrategy::MaximizePerformance;
        self.config.enable_select_acceleration = true;
        self.config.select_sample_rate = 256;
        self.config.enable_hardware_acceleration = true;
        self.config.superblock_size = 32;
        self.config.performance_tuning.enable_loop_unrolling = true;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> SeparatedStorageConfig {
        self.config
    }

    /// Build and validate the configuration
    pub fn build_validated(self) -> Result<SeparatedStorageConfig> {
        let config = self.config;
        config.validate()?;
        Ok(config)
    }
}

impl Default for SeparatedStorageConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for config analysis and optimization
impl SeparatedStorageConfig {
    /// Analyze bit vector characteristics and suggest optimal configuration
    pub fn analyze_and_optimize(
        bit_vector: &crate::succinct::BitVector,
    ) -> SeparatedStorageConfigBuilder {
        let len = bit_vector.len();
        let ones = bit_vector.count_ones();
        let density = ones as f64 / len as f64;

        let mut builder = SeparatedStorageConfigBuilder::new();

        // Size-based optimizations
        if len < 1_000_000 {
            // Small datasets: optimize for cache locality
            builder = builder.block_size(256).cache_alignment(CacheAlignment::CacheLine64);
        } else if len < 100_000_000 {
            // Medium datasets: balance performance and space
            builder = builder.block_size(512).cache_alignment(CacheAlignment::CacheLine64);
        } else {
            // Large datasets: optimize for space efficiency
            builder = builder
                .block_size(1024)
                .enable_bit_packed_ranks(true)
                .superblock_size(64);
        }

        // Density-based optimizations
        if density < 0.1 || density > 0.9 {
            // Very sparse or very dense: optimize for space
            builder = builder
                .memory_strategy(MemoryStrategy::MinimizeSpace)
                .enable_bit_packed_ranks(true);
        } else {
            // Balanced density: optimize for performance
            builder = builder
                .memory_strategy(MemoryStrategy::Balanced)
                .enable_select_acceleration(true);
        }

        builder
    }

    /// Get configuration summary for debugging
    pub fn summary(&self) -> ConfigSummary {
        ConfigSummary {
            block_size: self.block_size,
            storage_layout: self.storage_layout,
            memory_strategy: self.memory_strategy,
            estimated_overhead: self.estimated_memory_overhead(),
            has_select_cache: self.enable_select_acceleration,
            uses_bit_packing: self.enable_bit_packed_ranks,
            multi_dimensional_arity: self.multi_dimensional.as_ref().map(|m| m.arity),
            hardware_acceleration_enabled: self.enable_hardware_acceleration,
        }
    }
}

/// Summary of configuration settings for debugging and monitoring
#[derive(Debug, Clone)]
pub struct ConfigSummary {
    pub block_size: usize,
    pub storage_layout: StorageLayout,
    pub memory_strategy: MemoryStrategy,
    pub estimated_overhead: f64,
    pub has_select_cache: bool,
    pub uses_bit_packing: bool,
    pub multi_dimensional_arity: Option<usize>,
    pub hardware_acceleration_enabled: bool,
}

impl fmt::Display for ConfigSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SeparatedStorageConfig: {} blocks, {:?} layout, {:?} memory, {:.1}% overhead, select={}, bit_packing={}, hw_accel={}",
            self.block_size,
            self.storage_layout,
            self.memory_strategy,
            self.estimated_overhead,
            self.has_select_cache,
            self.uses_bit_packing,
            self.hardware_acceleration_enabled
        )?;

        if let Some(arity) = self.multi_dimensional_arity {
            write!(f, ", multi_dim={}", arity)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SeparatedStorageConfig::default();
        assert_eq!(config.block_size, 256);
        assert!(config.enable_select_acceleration);
        assert_eq!(config.select_sample_rate, 512);
        assert!(config.enable_hardware_acceleration);
    }

    #[test]
    fn test_config_builder() {
        let config = SeparatedStorageConfig::new()
            .block_size(512)
            .enable_select_acceleration(false)
            .superblock_size(32)
            .build();

        assert_eq!(config.block_size, 512);
        assert!(!config.enable_select_acceleration);
        assert_eq!(config.superblock_size, 32);
    }

    #[test]
    fn test_high_performance_preset() {
        let config = SeparatedStorageConfig::high_performance().build();
        assert_eq!(config.block_size, 512);
        assert!(config.enable_select_acceleration);
        assert_eq!(config.memory_strategy, MemoryStrategy::MaximizePerformance);
    }

    #[test]
    fn test_low_memory_preset() {
        let config = SeparatedStorageConfig::low_memory().build();
        assert_eq!(config.block_size, 1024);
        assert!(!config.enable_select_acceleration);
        assert_eq!(config.memory_strategy, MemoryStrategy::MinimizeSpace);
    }

    #[test]
    fn test_multi_dimensional_config() {
        let config = SeparatedStorageConfig::multi_dimensional(3).build();
        assert!(config.multi_dimensional.is_some());
        let multi = config.multi_dimensional.unwrap();
        assert_eq!(multi.arity, 3);
        assert_eq!(multi.dimension_hints.len(), 3);
    }

    #[test]
    fn test_config_validation() {
        // Valid config should pass
        let valid_config = SeparatedStorageConfig::default();
        assert!(valid_config.validate().is_ok());

        // Invalid block size should fail
        let mut invalid_config = SeparatedStorageConfig::default();
        invalid_config.block_size = 123;
        assert!(invalid_config.validate().is_err());

        // Invalid relative rank bits should fail
        let mut invalid_config = SeparatedStorageConfig::default();
        invalid_config.enable_bit_packed_ranks = true;
        invalid_config.relative_rank_bits = 15;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_memory_overhead_estimation() {
        let base_config = SeparatedStorageConfig::default();
        let base_overhead = base_config.estimated_memory_overhead();

        // Larger blocks should have lower overhead
        let large_block_config = SeparatedStorageConfig::new().block_size(1024).build();
        assert!(large_block_config.estimated_memory_overhead() < base_overhead);

        // Bit-packed ranks should reduce overhead
        let bit_packed_config = SeparatedStorageConfig::new()
            .enable_bit_packed_ranks(true)
            .build();
        assert!(bit_packed_config.estimated_memory_overhead() < base_overhead);
    }
}