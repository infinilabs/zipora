//! Cache configuration and tuning parameters

use crate::error::Result;
use super::{CacheError, PAGE_SIZE, HUGE_PAGE_SIZE, MAX_SHARDS};

/// Page cache configuration with various optimization profiles
#[derive(Debug, Clone)]
pub struct PageCacheConfig {
    /// Total cache capacity in bytes
    pub capacity: usize,
    
    /// Number of shards for reduced contention
    pub num_shards: u32,
    
    /// Page size (must be power of 2)
    pub page_size: usize,
    
    /// Use huge pages for large allocations
    pub use_huge_pages: bool,
    
    /// Enable hardware prefetching
    pub enable_prefetch: bool,
    
    /// Hash table load factor (0.5 - 0.9)
    pub load_factor: f64,
    
    /// Maximum conflict chain length before rehashing
    pub max_conflict_length: usize,
    
    /// Enable detailed statistics collection
    pub enable_statistics: bool,
    
    /// Locking strategy configuration
    pub locking: LockingConfig,
    
    /// Memory allocation configuration
    pub memory: MemoryConfig,
    
    /// Performance tuning parameters
    pub performance: PerformanceConfig,
}

/// Locking strategy configuration
#[derive(Debug, Clone)]
pub struct LockingConfig {
    /// Use futex-based locks when available
    pub use_futex: bool,
    
    /// Enable individual file vector locking
    pub individual_file_locks: bool,
    
    /// Lock timeout in milliseconds
    pub lock_timeout_ms: u64,
    
    /// Enable lock-free fast paths
    pub enable_lock_free: bool,
}

/// Memory allocation configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Memory alignment for cache blocks
    pub alignment: usize,
    
    /// Enable NUMA awareness
    pub numa_aware: bool,
    
    /// Pre-allocate all memory on startup
    pub pre_allocate: bool,
    
    /// Use secure memory pools
    pub use_secure_pools: bool,
    
    /// Memory advice for OS kernel
    pub kernel_advice: KernelAdvice,
}

/// Kernel memory advice configuration
#[derive(Debug, Clone)]
pub struct KernelAdvice {
    /// Use MADV_HUGEPAGE for large allocations
    pub huge_pages: bool,
    
    /// Use MADV_WILLNEED for cache memory
    pub will_need: bool,
    
    /// Use MADV_SEQUENTIAL for sequential access patterns
    pub sequential: bool,
    
    /// Use MADV_DONTDUMP to exclude from core dumps
    pub dont_dump: bool,
}

/// Performance tuning configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Batch size for multi-page operations
    pub batch_size: usize,
    
    /// Prefetch distance for sequential access
    pub prefetch_distance: usize,
    
    /// Cache warming strategy
    pub warming_strategy: WarmingStrategy,
    
    /// Eviction policy configuration
    pub eviction: EvictionConfig,
    
    /// Background maintenance configuration
    pub maintenance: MaintenanceConfig,
}

/// Cache warming strategy
#[derive(Debug, Clone, PartialEq)]
pub enum WarmingStrategy {
    /// No cache warming
    None,
    /// Warm cache on first access
    OnDemand,
    /// Warm cache in background
    Background,
    /// Aggressive warming with prefetching
    Aggressive,
}

/// Eviction policy configuration
#[derive(Debug, Clone)]
pub struct EvictionConfig {
    /// Eviction algorithm
    pub algorithm: EvictionAlgorithm,
    
    /// High water mark for eviction (percentage)
    pub high_water_mark: f64,
    
    /// Low water mark for eviction (percentage)
    pub low_water_mark: f64,
    
    /// Priority boost for recently accessed pages
    pub recency_boost: f64,
    
    /// Priority boost for frequently accessed pages
    pub frequency_boost: f64,
}

/// Eviction algorithm options
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionAlgorithm {
    /// Least Recently Used
    Lru,
    /// Least Recently Used with aging
    LruAging,
    /// Adaptive Replacement Cache
    Arc,
    /// Clock algorithm
    Clock,
    /// Two-Queue algorithm
    TwoQ,
}

/// Background maintenance configuration
#[derive(Debug, Clone)]
pub struct MaintenanceConfig {
    /// Enable background defragmentation
    pub enable_defrag: bool,
    
    /// Defragmentation threshold (fragmentation percentage)
    pub defrag_threshold: f64,
    
    /// Enable background statistics collection
    pub enable_background_stats: bool,
    
    /// Maintenance interval in milliseconds
    pub maintenance_interval_ms: u64,
}

impl Default for PageCacheConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

impl PageCacheConfig {
    /// Create a balanced configuration suitable for most workloads
    pub fn balanced() -> Self {
        Self {
            capacity: 64 * 1024 * 1024, // 64MB default
            num_shards: 4,
            page_size: PAGE_SIZE,
            use_huge_pages: false,
            enable_prefetch: true,
            load_factor: 0.75,
            max_conflict_length: 8,
            enable_statistics: true,
            locking: LockingConfig::balanced(),
            memory: MemoryConfig::balanced(),
            performance: PerformanceConfig::balanced(),
        }
    }
    
    /// Create a performance-optimized configuration
    pub fn performance_optimized() -> Self {
        Self {
            capacity: 256 * 1024 * 1024, // 256MB for performance
            num_shards: 8,
            page_size: PAGE_SIZE,
            use_huge_pages: true,
            enable_prefetch: true,
            load_factor: 0.7, // Lower for better performance
            max_conflict_length: 6,
            enable_statistics: false, // Disable for max performance
            locking: LockingConfig::performance_optimized(),
            memory: MemoryConfig::performance_optimized(),
            performance: PerformanceConfig::performance_optimized(),
        }
    }
    
    /// Create a memory-efficient configuration
    pub fn memory_optimized() -> Self {
        Self {
            capacity: 32 * 1024 * 1024, // 32MB minimal
            num_shards: 2,
            page_size: PAGE_SIZE,
            use_huge_pages: false,
            enable_prefetch: false,
            load_factor: 0.85, // Higher for memory efficiency
            max_conflict_length: 12,
            enable_statistics: false,
            locking: LockingConfig::memory_optimized(),
            memory: MemoryConfig::memory_optimized(),
            performance: PerformanceConfig::memory_optimized(),
        }
    }
    
    /// Create a high-security configuration
    pub fn security_optimized() -> Self {
        Self {
            capacity: 64 * 1024 * 1024,
            num_shards: 4,
            page_size: PAGE_SIZE,
            use_huge_pages: false,
            enable_prefetch: true,
            load_factor: 0.75,
            max_conflict_length: 8,
            enable_statistics: true,
            locking: LockingConfig::security_optimized(),
            memory: MemoryConfig::security_optimized(),
            performance: PerformanceConfig::security_optimized(),
        }
    }
    
    /// Builder pattern: Set cache capacity
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }
    
    /// Builder pattern: Set number of shards
    pub fn with_shards(mut self, shards: u32) -> Self {
        self.num_shards = shards;
        self
    }
    
    /// Builder pattern: Enable/disable huge pages
    pub fn with_huge_pages(mut self, enable: bool) -> Self {
        self.use_huge_pages = enable;
        self
    }
    
    /// Builder pattern: Enable/disable prefetching
    pub fn with_prefetch(mut self, enable: bool) -> Self {
        self.enable_prefetch = enable;
        self
    }
    
    /// Builder pattern: Set load factor
    pub fn with_load_factor(mut self, factor: f64) -> Self {
        self.load_factor = factor;
        self
    }
    
    /// Builder pattern: Enable/disable statistics
    pub fn with_statistics(mut self, enable: bool) -> Self {
        self.enable_statistics = enable;
        self
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Validate capacity
        if self.capacity == 0 {
            return Err(CacheError::InvalidPageSize.into());
        }
        
        // Validate page size (must be power of 2)
        if !self.page_size.is_power_of_two() || self.page_size < 1024 {
            return Err(CacheError::InvalidPageSize.into());
        }
        
        // Validate number of shards
        if self.num_shards == 0 || self.num_shards > MAX_SHARDS as u32 {
            return Err(CacheError::InvalidShardConfig.into());
        }
        
        // Validate load factor
        if self.load_factor <= 0.0 || self.load_factor >= 1.0 {
            return Err(CacheError::InvalidShardConfig.into());
        }
        
        // Validate huge page usage
        if self.use_huge_pages && self.capacity < HUGE_PAGE_SIZE {
            return Err(CacheError::InvalidPageSize.into());
        }
        
        Ok(())
    }
    
    /// Calculate number of pages based on configuration
    pub fn calculate_page_count(&self) -> usize {
        self.capacity / self.page_size
    }
    
    /// Calculate hash table size for optimal performance
    pub fn calculate_hash_table_size(&self) -> usize {
        let pages = self.calculate_page_count();
        let target_size = (pages as f64 / self.load_factor) as usize;
        target_size.next_power_of_two()
    }
}

impl LockingConfig {
    pub fn balanced() -> Self {
        Self {
            use_futex: true,
            individual_file_locks: false,
            lock_timeout_ms: 1000,
            enable_lock_free: true,
        }
    }
    
    pub fn performance_optimized() -> Self {
        Self {
            use_futex: true,
            individual_file_locks: true,
            lock_timeout_ms: 100,
            enable_lock_free: true,
        }
    }
    
    pub fn memory_optimized() -> Self {
        Self {
            use_futex: false,
            individual_file_locks: false,
            lock_timeout_ms: 5000,
            enable_lock_free: false,
        }
    }
    
    pub fn security_optimized() -> Self {
        Self {
            use_futex: false,
            individual_file_locks: false,
            lock_timeout_ms: 2000,
            enable_lock_free: false,
        }
    }
}

impl MemoryConfig {
    pub fn balanced() -> Self {
        Self {
            alignment: PAGE_SIZE,
            numa_aware: false,
            pre_allocate: false,
            use_secure_pools: true,
            kernel_advice: KernelAdvice::balanced(),
        }
    }
    
    pub fn performance_optimized() -> Self {
        Self {
            alignment: HUGE_PAGE_SIZE,
            numa_aware: true,
            pre_allocate: true,
            use_secure_pools: false, // Disable for max performance
            kernel_advice: KernelAdvice::performance_optimized(),
        }
    }
    
    pub fn memory_optimized() -> Self {
        Self {
            alignment: PAGE_SIZE,
            numa_aware: false,
            pre_allocate: false,
            use_secure_pools: true,
            kernel_advice: KernelAdvice::memory_optimized(),
        }
    }
    
    pub fn security_optimized() -> Self {
        Self {
            alignment: PAGE_SIZE,
            numa_aware: false,
            pre_allocate: true,
            use_secure_pools: true,
            kernel_advice: KernelAdvice::security_optimized(),
        }
    }
}

impl KernelAdvice {
    pub fn balanced() -> Self {
        Self {
            huge_pages: false,
            will_need: true,
            sequential: false,
            dont_dump: false,
        }
    }
    
    pub fn performance_optimized() -> Self {
        Self {
            huge_pages: true,
            will_need: true,
            sequential: true,
            dont_dump: false,
        }
    }
    
    pub fn memory_optimized() -> Self {
        Self {
            huge_pages: false,
            will_need: false,
            sequential: false,
            dont_dump: true,
        }
    }
    
    pub fn security_optimized() -> Self {
        Self {
            huge_pages: false,
            will_need: true,
            sequential: false,
            dont_dump: true,
        }
    }
}

impl PerformanceConfig {
    pub fn balanced() -> Self {
        Self {
            batch_size: 8,
            prefetch_distance: 2,
            warming_strategy: WarmingStrategy::OnDemand,
            eviction: EvictionConfig::balanced(),
            maintenance: MaintenanceConfig::balanced(),
        }
    }
    
    pub fn performance_optimized() -> Self {
        Self {
            batch_size: 16,
            prefetch_distance: 4,
            warming_strategy: WarmingStrategy::Aggressive,
            eviction: EvictionConfig::performance_optimized(),
            maintenance: MaintenanceConfig::performance_optimized(),
        }
    }
    
    pub fn memory_optimized() -> Self {
        Self {
            batch_size: 4,
            prefetch_distance: 1,
            warming_strategy: WarmingStrategy::None,
            eviction: EvictionConfig::memory_optimized(),
            maintenance: MaintenanceConfig::memory_optimized(),
        }
    }
    
    pub fn security_optimized() -> Self {
        Self {
            batch_size: 4,
            prefetch_distance: 1,
            warming_strategy: WarmingStrategy::OnDemand,
            eviction: EvictionConfig::security_optimized(),
            maintenance: MaintenanceConfig::security_optimized(),
        }
    }
}

impl EvictionConfig {
    pub fn balanced() -> Self {
        Self {
            algorithm: EvictionAlgorithm::Lru,
            high_water_mark: 0.9,
            low_water_mark: 0.7,
            recency_boost: 1.2,
            frequency_boost: 1.1,
        }
    }
    
    pub fn performance_optimized() -> Self {
        Self {
            algorithm: EvictionAlgorithm::Arc,
            high_water_mark: 0.85,
            low_water_mark: 0.65,
            recency_boost: 1.5,
            frequency_boost: 1.3,
        }
    }
    
    pub fn memory_optimized() -> Self {
        Self {
            algorithm: EvictionAlgorithm::Clock,
            high_water_mark: 0.95,
            low_water_mark: 0.8,
            recency_boost: 1.0,
            frequency_boost: 1.0,
        }
    }
    
    pub fn security_optimized() -> Self {
        Self {
            algorithm: EvictionAlgorithm::Lru,
            high_water_mark: 0.9,
            low_water_mark: 0.7,
            recency_boost: 1.1,
            frequency_boost: 1.05,
        }
    }
}

impl MaintenanceConfig {
    pub fn balanced() -> Self {
        Self {
            enable_defrag: true,
            defrag_threshold: 0.3,
            enable_background_stats: true,
            maintenance_interval_ms: 5000,
        }
    }
    
    pub fn performance_optimized() -> Self {
        Self {
            enable_defrag: true,
            defrag_threshold: 0.2,
            enable_background_stats: false,
            maintenance_interval_ms: 1000,
        }
    }
    
    pub fn memory_optimized() -> Self {
        Self {
            enable_defrag: false,
            defrag_threshold: 0.5,
            enable_background_stats: false,
            maintenance_interval_ms: 30000,
        }
    }
    
    pub fn security_optimized() -> Self {
        Self {
            enable_defrag: true,
            defrag_threshold: 0.25,
            enable_background_stats: true,
            maintenance_interval_ms: 10000,
        }
    }
}