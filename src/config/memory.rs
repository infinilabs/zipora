//! Memory management configuration for Zipora.
//! 
//! This module provides comprehensive configuration for memory allocation,
//! caching, and optimization strategies throughout the system.

use super::{Config, ValidationError, parse_env_var, parse_env_bool};
use crate::error::{Result, ZiporaError};
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Memory allocation strategy for different scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Default system allocator
    System,
    /// Secure memory pool with protection
    SecurePool,
    /// Lock-free allocator for high concurrency
    LockFree,
    /// Thread-local allocator for per-thread optimization
    ThreadLocal,
    /// Fixed-capacity allocator for real-time systems
    FixedCapacity,
    /// Memory-mapped allocator for large datasets
    MemoryMapped,
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        Self::SecurePool
    }
}

/// Cache optimization level for different performance requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheOptimizationLevel {
    /// No cache optimization
    None,
    /// Basic cache-line alignment
    Basic,
    /// Advanced optimization with prefetching
    Advanced,
    /// Maximum optimization with NUMA awareness
    Maximum,
}

impl Default for CacheOptimizationLevel {
    fn default() -> Self {
        Self::Advanced
    }
}

/// NUMA (Non-Uniform Memory Access) configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumaConfig {
    /// Enable NUMA-aware allocation
    pub enable_numa_awareness: bool,
    /// Preferred NUMA node (-1 for auto)
    pub preferred_node: i32,
    /// Enable local allocation preference
    pub prefer_local_allocation: bool,
    /// Enable NUMA balancing
    pub enable_balancing: bool,
}

impl Default for NumaConfig {
    fn default() -> Self {
        Self {
            enable_numa_awareness: true,
            preferred_node: -1, // Auto-detect
            prefer_local_allocation: true,
            enable_balancing: false,
        }
    }
}

/// Huge page configuration for large memory operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HugePageConfig {
    /// Enable huge page allocation
    pub enable_huge_pages: bool,
    /// Huge page size in bytes (0 = auto-detect)
    pub page_size: usize,
    /// Minimum allocation size for huge pages
    pub min_allocation_size: usize,
    /// Enable transparent huge pages
    pub enable_transparent: bool,
}

impl Default for HugePageConfig {
    fn default() -> Self {
        Self {
            enable_huge_pages: false, // Conservative default
            page_size: 0, // Auto-detect (typically 2MB or 1GB)
            min_allocation_size: 2 * 1024 * 1024, // 2MB
            enable_transparent: true,
        }
    }
}

/// Comprehensive memory management configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Primary allocation strategy
    pub allocation_strategy: AllocationStrategy,
    
    /// Initial memory pool size in bytes
    pub initial_pool_size: usize,
    
    /// Maximum memory pool size in bytes (0 = unlimited)
    pub max_pool_size: usize,
    
    /// Memory pool growth factor (1.0-4.0)
    pub growth_factor: f64,
    
    /// Cache optimization level
    pub cache_optimization: CacheOptimizationLevel,
    
    /// NUMA configuration
    pub numa_config: NumaConfig,
    
    /// Huge page configuration
    pub huge_page_config: HugePageConfig,
    
    /// Enable memory debugging and tracking
    pub enable_debug_tracking: bool,
    
    /// Enable memory statistics collection
    pub enable_statistics: bool,
    
    /// Memory allocation alignment in bytes
    pub alignment: usize,
    
    /// Cache line size in bytes (0 = auto-detect)
    pub cache_line_size: usize,
    
    /// Number of memory pools for lock-free allocation
    pub num_pools: usize,
    
    /// Enable memory prefetching hints
    pub enable_prefetching: bool,
    
    /// Prefetch distance for sequential access
    pub prefetch_distance: usize,
    
    /// Enable hot/cold data separation
    pub enable_hot_cold_separation: bool,
    
    /// Hot data access threshold
    pub hot_access_threshold: u32,
    
    /// Enable memory compaction
    pub enable_compaction: bool,
    
    /// Compaction trigger threshold (0.0-1.0)
    pub compaction_threshold: f64,
    
    /// Maximum memory fragmentation ratio (0.0-1.0)
    pub max_fragmentation_ratio: f64,
    
    /// Enable memory protection features
    pub enable_memory_protection: bool,
    
    /// Guard page size for buffer overflow protection
    pub guard_page_size: usize,
    
    /// Enable use-after-free detection
    pub enable_use_after_free_detection: bool,
    
    /// Enable double-free detection
    pub enable_double_free_detection: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            allocation_strategy: AllocationStrategy::default(),
            initial_pool_size: 64 * 1024 * 1024, // 64MB
            max_pool_size: 0, // Unlimited
            growth_factor: 1.618, // Golden ratio
            cache_optimization: CacheOptimizationLevel::default(),
            numa_config: NumaConfig::default(),
            huge_page_config: HugePageConfig::default(),
            enable_debug_tracking: false,
            enable_statistics: true,
            alignment: 64, // Cache line alignment
            cache_line_size: 0, // Auto-detect
            num_pools: 8, // Good for most systems
            enable_prefetching: true,
            prefetch_distance: 2, // Cache lines ahead
            enable_hot_cold_separation: true,
            hot_access_threshold: 100,
            enable_compaction: false, // Conservative default
            compaction_threshold: 0.8,
            max_fragmentation_ratio: 0.3,
            enable_memory_protection: true,
            guard_page_size: 4096, // One page
            enable_use_after_free_detection: true,
            enable_double_free_detection: true,
        }
    }
}

impl Config for MemoryConfig {
    fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();
        
        // Validate pool sizes
        if self.initial_pool_size == 0 {
            errors.push(ValidationError::new(
                "initial_pool_size",
                &self.initial_pool_size.to_string(),
                "initial pool size must be greater than 0"
            ).with_suggestion("typical values: 16MB-1GB"));
        }
        
        if self.max_pool_size != 0 && self.max_pool_size < self.initial_pool_size {
            errors.push(ValidationError::new(
                "max_pool_size",
                &self.max_pool_size.to_string(),
                "maximum pool size must be greater than initial pool size"
            ));
        }
        
        // Validate growth factor
        if self.growth_factor < 1.0 || self.growth_factor > 4.0 {
            errors.push(ValidationError::new(
                "growth_factor",
                &self.growth_factor.to_string(),
                "growth factor must be between 1.0 and 4.0"
            ).with_suggestion("typical values: 1.5-2.0, golden ratio: 1.618"));
        }
        
        // Validate alignment
        if self.alignment == 0 || (self.alignment & (self.alignment - 1)) != 0 {
            errors.push(ValidationError::new(
                "alignment",
                &self.alignment.to_string(),
                "alignment must be a power of 2"
            ).with_suggestion("typical values: 8, 16, 32, 64, 128"));
        }
        
        // Validate compaction threshold
        if self.compaction_threshold < 0.0 || self.compaction_threshold > 1.0 {
            errors.push(ValidationError::new(
                "compaction_threshold",
                &self.compaction_threshold.to_string(),
                "compaction threshold must be between 0.0 and 1.0"
            ));
        }
        
        // Validate fragmentation ratio
        if self.max_fragmentation_ratio < 0.0 || self.max_fragmentation_ratio > 1.0 {
            errors.push(ValidationError::new(
                "max_fragmentation_ratio",
                &self.max_fragmentation_ratio.to_string(),
                "fragmentation ratio must be between 0.0 and 1.0"
            ));
        }
        
        // Validate num_pools
        if self.num_pools == 0 {
            errors.push(ValidationError::new(
                "num_pools",
                &self.num_pools.to_string(),
                "number of pools must be at least 1"
            ).with_suggestion("typical values: 4-16 based on CPU cores"));
        }
        
        // Return first error if any
        if !errors.is_empty() {
            return Err(ZiporaError::configuration(format!(
                "Memory configuration validation failed: {}",
                errors.into_iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }
        
        Ok(())
    }
    
    fn from_env_with_prefix(prefix: &str) -> Result<Self> {
        let mut config = Self::default();
        
        // Basic memory settings
        config.initial_pool_size = parse_env_var(&format!("{}MEMORY_INITIAL_POOL_SIZE", prefix), config.initial_pool_size);
        config.max_pool_size = parse_env_var(&format!("{}MEMORY_MAX_POOL_SIZE", prefix), config.max_pool_size);
        config.growth_factor = parse_env_var(&format!("{}MEMORY_GROWTH_FACTOR", prefix), config.growth_factor);
        config.alignment = parse_env_var(&format!("{}MEMORY_ALIGNMENT", prefix), config.alignment);
        config.cache_line_size = parse_env_var(&format!("{}MEMORY_CACHE_LINE_SIZE", prefix), config.cache_line_size);
        config.num_pools = parse_env_var(&format!("{}MEMORY_NUM_POOLS", prefix), config.num_pools);
        
        // Feature flags
        config.enable_debug_tracking = parse_env_bool(&format!("{}MEMORY_DEBUG_TRACKING", prefix), config.enable_debug_tracking);
        config.enable_statistics = parse_env_bool(&format!("{}MEMORY_STATISTICS", prefix), config.enable_statistics);
        config.enable_prefetching = parse_env_bool(&format!("{}MEMORY_PREFETCHING", prefix), config.enable_prefetching);
        config.enable_hot_cold_separation = parse_env_bool(&format!("{}MEMORY_HOT_COLD_SEPARATION", prefix), config.enable_hot_cold_separation);
        config.enable_compaction = parse_env_bool(&format!("{}MEMORY_COMPACTION", prefix), config.enable_compaction);
        config.enable_memory_protection = parse_env_bool(&format!("{}MEMORY_PROTECTION", prefix), config.enable_memory_protection);
        
        // Advanced settings
        config.prefetch_distance = parse_env_var(&format!("{}MEMORY_PREFETCH_DISTANCE", prefix), config.prefetch_distance);
        config.hot_access_threshold = parse_env_var(&format!("{}MEMORY_HOT_ACCESS_THRESHOLD", prefix), config.hot_access_threshold);
        config.compaction_threshold = parse_env_var(&format!("{}MEMORY_COMPACTION_THRESHOLD", prefix), config.compaction_threshold);
        config.max_fragmentation_ratio = parse_env_var(&format!("{}MEMORY_MAX_FRAGMENTATION_RATIO", prefix), config.max_fragmentation_ratio);
        config.guard_page_size = parse_env_var(&format!("{}MEMORY_GUARD_PAGE_SIZE", prefix), config.guard_page_size);
        
        // NUMA settings
        config.numa_config.enable_numa_awareness = parse_env_bool(&format!("{}MEMORY_NUMA_ENABLE", prefix), config.numa_config.enable_numa_awareness);
        config.numa_config.preferred_node = parse_env_var(&format!("{}MEMORY_NUMA_PREFERRED_NODE", prefix), config.numa_config.preferred_node);
        config.numa_config.prefer_local_allocation = parse_env_bool(&format!("{}MEMORY_NUMA_LOCAL_ALLOCATION", prefix), config.numa_config.prefer_local_allocation);
        config.numa_config.enable_balancing = parse_env_bool(&format!("{}MEMORY_NUMA_BALANCING", prefix), config.numa_config.enable_balancing);
        
        // Huge page settings
        config.huge_page_config.enable_huge_pages = parse_env_bool(&format!("{}MEMORY_HUGE_PAGES_ENABLE", prefix), config.huge_page_config.enable_huge_pages);
        config.huge_page_config.page_size = parse_env_var(&format!("{}MEMORY_HUGE_PAGE_SIZE", prefix), config.huge_page_config.page_size);
        config.huge_page_config.min_allocation_size = parse_env_var(&format!("{}MEMORY_HUGE_PAGE_MIN_SIZE", prefix), config.huge_page_config.min_allocation_size);
        config.huge_page_config.enable_transparent = parse_env_bool(&format!("{}MEMORY_HUGE_PAGES_TRANSPARENT", prefix), config.huge_page_config.enable_transparent);
        
        config.validate()?;
        Ok(config)
    }
    
    fn performance_preset() -> Self {
        let mut config = Self::default();
        
        // Optimize for maximum performance
        config.allocation_strategy = AllocationStrategy::LockFree;
        config.initial_pool_size = 256 * 1024 * 1024; // 256MB
        config.growth_factor = 2.0; // Fast growth
        config.cache_optimization = CacheOptimizationLevel::Maximum;
        config.alignment = 64; // Cache line aligned
        config.num_pools = 16; // High concurrency
        config.enable_prefetching = true;
        config.prefetch_distance = 4; // Aggressive prefetching
        config.enable_hot_cold_separation = true;
        config.hot_access_threshold = 50; // Lower threshold for hot data
        
        // Enable huge pages for performance
        config.huge_page_config.enable_huge_pages = true;
        config.huge_page_config.min_allocation_size = 1024 * 1024; // 1MB
        
        // NUMA optimization
        config.numa_config.enable_numa_awareness = true;
        config.numa_config.prefer_local_allocation = true;
        
        // Reduce protection overhead for performance
        config.enable_memory_protection = false;
        config.enable_debug_tracking = false;
        
        config
    }
    
    fn memory_preset() -> Self {
        let mut config = Self::default();
        
        // Optimize for minimal memory usage
        config.allocation_strategy = AllocationStrategy::FixedCapacity;
        config.initial_pool_size = 16 * 1024 * 1024; // 16MB
        config.max_pool_size = 128 * 1024 * 1024; // 128MB limit
        config.growth_factor = 1.2; // Slow growth
        config.cache_optimization = CacheOptimizationLevel::Basic;
        config.alignment = 16; // Smaller alignment
        config.num_pools = 2; // Minimal pools
        config.enable_prefetching = false; // Save memory
        config.enable_hot_cold_separation = false;
        config.enable_compaction = true; // Reduce fragmentation
        config.compaction_threshold = 0.6; // Aggressive compaction
        config.max_fragmentation_ratio = 0.2; // Low fragmentation tolerance
        
        // Disable huge pages to save memory
        config.huge_page_config.enable_huge_pages = false;
        
        // Minimal NUMA features
        config.numa_config.enable_numa_awareness = false;
        
        // Enable protection for safety
        config.enable_memory_protection = true;
        config.enable_debug_tracking = false; // Save memory
        
        config
    }
    
    fn realtime_preset() -> Self {
        let mut config = Self::default();
        
        // Optimize for predictable real-time performance
        config.allocation_strategy = AllocationStrategy::FixedCapacity;
        config.initial_pool_size = 128 * 1024 * 1024; // 128MB pre-allocated
        config.max_pool_size = 128 * 1024 * 1024; // Fixed size
        config.growth_factor = 1.0; // No growth after initial
        config.cache_optimization = CacheOptimizationLevel::Advanced;
        config.alignment = 64; // Cache line aligned
        config.num_pools = 4; // Moderate concurrency
        config.enable_prefetching = true;
        config.prefetch_distance = 2; // Moderate prefetching
        config.enable_hot_cold_separation = false; // Avoid dynamic behavior
        config.enable_compaction = false; // Avoid unpredictable latency
        
        // Pre-allocate huge pages for predictability
        config.huge_page_config.enable_huge_pages = true;
        config.huge_page_config.enable_transparent = false; // Explicit allocation
        
        // NUMA awareness for consistent performance
        config.numa_config.enable_numa_awareness = true;
        config.numa_config.prefer_local_allocation = true;
        config.numa_config.enable_balancing = false; // Avoid migration
        
        // Enable protection but minimize overhead
        config.enable_memory_protection = true;
        config.enable_use_after_free_detection = false; // Reduce overhead
        config.enable_double_free_detection = true; // Keep critical protection
        config.enable_debug_tracking = false;
        
        config
    }
    
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| ZiporaError::configuration(format!("Failed to serialize memory config: {}", e)))?;
        
        std::fs::write(path, serialized)
            .map_err(|e| ZiporaError::configuration(format!("Failed to write memory config file: {}", e)))?;
        
        Ok(())
    }
    
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ZiporaError::configuration(format!("Failed to read memory config file: {}", e)))?;
        
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| ZiporaError::configuration(format!("Failed to parse memory config file: {}", e)))?;
        
        config.validate()?;
        Ok(config)
    }
}

impl MemoryConfig {
    /// Create a new memory configuration builder.
    pub fn builder() -> MemoryConfigBuilder {
        MemoryConfigBuilder::new()
    }
    
    /// Get the effective cache line size.
    /// 
    /// Returns the configured cache line size, or the detected system cache line size if auto-detect is enabled.
    pub fn effective_cache_line_size(&self) -> usize {
        if self.cache_line_size == 0 {
            // Auto-detect cache line size
            #[cfg(target_arch = "x86_64")]
            {
                64 // Typical x86_64 cache line size
            }
            #[cfg(target_arch = "aarch64")]
            {
                128 // Typical ARM64 cache line size
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                64 // Conservative default
            }
        } else {
            self.cache_line_size
        }
    }
    
    /// Get the effective number of memory pools.
    /// 
    /// Returns an appropriate number of pools based on the allocation strategy and system capabilities.
    pub fn effective_num_pools(&self) -> usize {
        match self.allocation_strategy {
            AllocationStrategy::System => 1,
            AllocationStrategy::SecurePool => std::cmp::min(self.num_pools, 8),
            AllocationStrategy::LockFree => self.num_pools,
            AllocationStrategy::ThreadLocal => num_cpus::get(),
            AllocationStrategy::FixedCapacity => 1,
            AllocationStrategy::MemoryMapped => 1,
        }
    }
}

/// Builder for constructing memory configurations.
#[derive(Debug, Clone)]
pub struct MemoryConfigBuilder {
    config: MemoryConfig,
}

impl MemoryConfigBuilder {
    /// Create a new memory configuration builder.
    pub fn new() -> Self {
        Self {
            config: MemoryConfig::default(),
        }
    }
    
    /// Set the allocation strategy.
    pub fn allocation_strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.config.allocation_strategy = strategy;
        self
    }
    
    /// Set the initial pool size.
    pub fn initial_pool_size(mut self, size: usize) -> Self {
        self.config.initial_pool_size = size;
        self
    }
    
    /// Set the maximum pool size.
    pub fn max_pool_size(mut self, size: usize) -> Self {
        self.config.max_pool_size = size;
        self
    }
    
    /// Set the cache optimization level.
    pub fn cache_optimization(mut self, level: CacheOptimizationLevel) -> Self {
        self.config.cache_optimization = level;
        self
    }
    
    /// Enable NUMA awareness.
    pub fn enable_numa(mut self, enabled: bool) -> Self {
        self.config.numa_config.enable_numa_awareness = enabled;
        self
    }
    
    /// Enable huge pages.
    pub fn enable_huge_pages(mut self, enabled: bool) -> Self {
        self.config.huge_page_config.enable_huge_pages = enabled;
        self
    }
    
    /// Set memory alignment.
    pub fn alignment(mut self, alignment: usize) -> Self {
        self.config.alignment = alignment;
        self
    }
    
    /// Set the number of memory pools.
    pub fn num_pools(mut self, pools: usize) -> Self {
        self.config.num_pools = pools;
        self
    }
    
    /// Enable memory protection features.
    pub fn enable_protection(mut self, enabled: bool) -> Self {
        self.config.enable_memory_protection = enabled;
        self
    }
    
    /// Build the configuration.
    pub fn build(self) -> Result<MemoryConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for MemoryConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = MemoryConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_builder_pattern() {
        let config = MemoryConfig::builder()
            .allocation_strategy(AllocationStrategy::LockFree)
            .initial_pool_size(128 * 1024 * 1024)
            .enable_numa(true)
            .build()
            .expect("Failed to build memory config");
        
        assert_eq!(config.allocation_strategy, AllocationStrategy::LockFree);
        assert_eq!(config.initial_pool_size, 128 * 1024 * 1024);
        assert!(config.numa_config.enable_numa_awareness);
    }
    
    #[test]
    fn test_presets() {
        let perf_config = MemoryConfig::performance_preset();
        assert!(perf_config.validate().is_ok());
        assert_eq!(perf_config.allocation_strategy, AllocationStrategy::LockFree);
        
        let mem_config = MemoryConfig::memory_preset();
        assert!(mem_config.validate().is_ok());
        assert_eq!(mem_config.allocation_strategy, AllocationStrategy::FixedCapacity);
        
        let rt_config = MemoryConfig::realtime_preset();
        assert!(rt_config.validate().is_ok());
        assert_eq!(rt_config.allocation_strategy, AllocationStrategy::FixedCapacity);
        assert!(!rt_config.enable_compaction);
    }
    
    #[test]
    fn test_validation() {
        let mut config = MemoryConfig::default();
        
        // Test invalid pool size
        config.initial_pool_size = 0;
        assert!(config.validate().is_err());
        
        // Test invalid growth factor
        config = MemoryConfig::default();
        config.growth_factor = 0.5;
        assert!(config.validate().is_err());
        
        config.growth_factor = 5.0;
        assert!(config.validate().is_err());
        
        // Test invalid alignment
        config = MemoryConfig::default();
        config.alignment = 3; // Not a power of 2
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_effective_values() {
        let config = MemoryConfig::default();
        
        // Test effective cache line size
        let cache_line_size = config.effective_cache_line_size();
        assert!(cache_line_size >= 32 && cache_line_size <= 128);
        
        // Test effective num pools
        let num_pools = config.effective_num_pools();
        assert!(num_pools >= 1);
    }
    
    #[test]
    fn test_serialization() {
        let config = MemoryConfig::default();
        
        // Test JSON serialization
        let json = serde_json::to_string(&config).expect("Failed to serialize");
        let deserialized: MemoryConfig = serde_json::from_str(&json).expect("Failed to deserialize");
        
        assert_eq!(config.allocation_strategy, deserialized.allocation_strategy);
        assert_eq!(config.initial_pool_size, deserialized.initial_pool_size);
        assert_eq!(config.numa_config.enable_numa_awareness, deserialized.numa_config.enable_numa_awareness);
    }
}