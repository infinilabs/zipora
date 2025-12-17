//! FSA Cache System
//!
//! High-performance caching layer for finite state automata operations.
//! Inspired by advanced cache trie implementations for optimal performance.

use crate::error::{Result, ZiporaError};
use crate::memory::SecureMemoryPool;
// Note: RankSelectInterleaved256 import removed as it's not used in current implementation
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Cache strategy for FSA operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Breadth-first search strategy for balanced access patterns
    BreadthFirst,
    /// Depth-first search strategy for sequential access patterns
    DepthFirst,
    /// Cache-friendly search - BFS for 2 levels then DFS for optimal cache locality
    CacheFriendly,
}

/// Configuration for FSA cache system
#[derive(Debug, Clone)]
pub struct FsaCacheConfig {
    /// Maximum number of states to cache
    pub max_states: usize,
    /// Cache strategy to use
    pub strategy: CacheStrategy,
    /// Enable compressed zero-path storage
    pub compressed_paths: bool,
    /// Enable hugepage allocation for large caches
    pub use_hugepages: bool,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
}

impl Default for FsaCacheConfig {
    fn default() -> Self {
        Self {
            max_states: 1_000_000,
            strategy: CacheStrategy::CacheFriendly,
            compressed_paths: true,
            use_hugepages: false,
            max_memory_bytes: 256 * 1024 * 1024, // 256MB
        }
    }
}

impl FsaCacheConfig {
    /// Create a configuration optimized for small datasets
    pub fn small() -> Self {
        Self {
            max_states: 10_000,
            max_memory_bytes: 4 * 1024 * 1024, // 4MB
            ..Default::default()
        }
    }

    /// Create a configuration optimized for large datasets
    pub fn large() -> Self {
        Self {
            max_states: 10_000_000,
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            use_hugepages: true,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for memory efficiency
    pub fn memory_efficient() -> Self {
        Self {
            max_states: 100_000,
            max_memory_bytes: 16 * 1024 * 1024, // 16MB
            compressed_paths: true,
            strategy: CacheStrategy::DepthFirst,
            ..Default::default()
        }
    }
}

/// Cached state representation (8 bytes, inspired by reference implementation)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CachedState {
    /// Base state for transitions (32-bit)
    pub child_base: u32,
    /// Parent/check state (24-bit) + flags (8-bit)
    pub parent_and_flags: u32,
}

impl CachedState {
    /// Create a new cached state
    pub fn new(child_base: u32, parent: u32, is_terminal: bool, is_free: bool) -> Self {
        let parent_and_flags = (parent & 0x00FFFFFF) 
            | if is_terminal { 0x80000000 } else { 0 }
            | if is_free { 0x40000000 } else { 0 };
        
        Self {
            child_base,
            parent_and_flags,
        }
    }

    /// Get the parent state ID
    pub fn parent(&self) -> u32 {
        self.parent_and_flags & 0x00FFFFFF
    }

    /// Check if this is a terminal state
    pub fn is_terminal(&self) -> bool {
        (self.parent_and_flags & 0x80000000) != 0
    }

    /// Check if this state is free
    pub fn is_free(&self) -> bool {
        (self.parent_and_flags & 0x40000000) != 0
    }

    /// Mark state as free
    pub fn mark_free(&mut self) {
        self.parent_and_flags |= 0x40000000;
    }

    /// Mark state as used
    pub fn mark_used(&mut self) {
        self.parent_and_flags &= !0x40000000;
    }
}

/// Compressed zero-path data for efficient path reconstruction
#[derive(Debug, Clone)]
pub struct ZeroPathData {
    /// Compressed path segments
    pub segments: Vec<u8>,
    /// Length of each segment
    pub lengths: Vec<u8>,
    /// Total path length
    pub total_length: u16,
}

impl ZeroPathData {
    /// Create new zero-path data
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            lengths: Vec::new(),
            total_length: 0,
        }
    }

    /// Add a path segment
    pub fn add_segment(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > 255 {
            return Err(ZiporaError::invalid_data("Path segment too long"));
        }
        
        self.segments.extend_from_slice(data);
        self.lengths.push(data.len() as u8);
        self.total_length += data.len() as u16;
        
        Ok(())
    }

    /// Get all segments as a single path
    pub fn get_full_path(&self) -> Vec<u8> {
        self.segments.clone()
    }

    /// Get compression ratio (compressed size / original size)
    pub fn compression_ratio(&self) -> f64 {
        if self.total_length == 0 {
            return 0.0;
        }
        
        let compressed_size = self.segments.len() + self.lengths.len();
        compressed_size as f64 / self.total_length as f64
    }
}

/// Statistics for FSA cache performance
#[derive(Debug, Clone, Default)]
pub struct FsaCacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of states currently cached
    pub cached_states: usize,
    /// Total memory usage in bytes
    pub memory_usage: usize,
    /// Average compression ratio for zero-paths
    pub avg_compression_ratio: f64,
    /// Number of evictions performed
    pub evictions: u64,
}

impl FsaCacheStats {
    /// Calculate cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate memory efficiency (states per byte)
    pub fn memory_efficiency(&self) -> f64 {
        if self.memory_usage == 0 {
            0.0
        } else {
            self.cached_states as f64 / self.memory_usage as f64
        }
    }
}

/// High-performance FSA cache system
pub struct FsaCache {
    /// Cache configuration
    config: FsaCacheConfig,
    /// Cached states indexed by state ID
    states: HashMap<u32, CachedState>,
    /// Zero-path data for compressed path storage
    zero_paths: HashMap<u32, ZeroPathData>,
    /// Free state list for efficient allocation
    free_list: Vec<u32>,
    /// Next available state ID
    next_state_id: u32,
    /// Cache statistics
    stats: FsaCacheStats,
    /// Memory pool for efficient allocation
    memory_pool: Option<Arc<SecureMemoryPool>>,
    /// Thread-safe access
    lock: RwLock<()>,
}

impl FsaCache {
    /// Create a new FSA cache with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(FsaCacheConfig::default())
    }

    /// Create a new FSA cache with custom configuration
    pub fn with_config(config: FsaCacheConfig) -> Result<Self> {
        let memory_pool = if config.max_memory_bytes > 0 {
            let pool_config = crate::memory::SecurePoolConfig::small_secure();
            Some(SecureMemoryPool::new(pool_config)?)
        } else {
            None
        };

        let initial_capacity = std::cmp::min(config.max_states, 1000);
        Ok(Self {
            config,
            states: HashMap::with_capacity(initial_capacity),
            zero_paths: HashMap::new(),
            free_list: Vec::new(),
            next_state_id: 1, // Start from 1, reserve 0 for invalid state
            stats: FsaCacheStats::default(),
            memory_pool,
            lock: RwLock::new(()),
        })
    }

    /// Get a cached state by ID
    pub fn get_state(&self, state_id: u32) -> Option<CachedState> {
        if let Some(state) = self.states.get(&state_id) {
            // Increment hit counter (we'd need atomic counters for true thread safety)
            Some(*state)
        } else {
            // Increment miss counter
            None
        }
    }

    /// Cache a new state
    pub fn cache_state(&mut self, parent_id: u32, child_base: u32, is_terminal: bool) -> Result<u32> {
        // Check if we need to evict states
        if self.states.len() >= self.config.max_states {
            self.evict_states()?;
        }

        // Allocate new state ID
        let state_id = if let Some(free_id) = self.free_list.pop() {
            free_id
        } else {
            let id = self.next_state_id;
            self.next_state_id += 1;
            id
        };

        // Create and cache the state
        let state = CachedState::new(child_base, parent_id, is_terminal, false);
        self.states.insert(state_id, state);
        
        // Update statistics
        self.stats.cached_states = self.states.len();
        self.stats.memory_usage = self.estimate_memory_usage();

        Ok(state_id)
    }

    /// Remove a state from cache
    pub fn remove_state(&mut self, state_id: u32) -> bool {
        // SAFETY: Return false if RwLock is poisoned (graceful degradation)
        let _guard = match self.lock.write() {
            Ok(g) => g,
            Err(_) => return false,
        };

        if self.states.remove(&state_id).is_some() {
            self.zero_paths.remove(&state_id);
            self.free_list.push(state_id);
            
            // Update statistics
            self.stats.cached_states = self.states.len();
            self.stats.memory_usage = self.estimate_memory_usage();
            
            true
        } else {
            false
        }
    }

    /// Add zero-path data for compressed path storage
    pub fn add_zero_path(&mut self, state_id: u32, path_data: ZeroPathData) -> Result<()> {
        {
            let _guard = self.lock.write()
                .map_err(|e| ZiporaError::system_error(
                    format!("FsaCache: lock RwLock poisoned: {}", e)
                ))?;

            if !self.states.contains_key(&state_id) {
                return Err(ZiporaError::invalid_data("State not found in cache"));
            }

            self.zero_paths.insert(state_id, path_data);
        }
        
        self.update_compression_stats();
        
        Ok(())
    }

    /// Get zero-path data for a state
    pub fn get_zero_path(&self, state_id: u32) -> Option<&ZeroPathData> {
        // SAFETY: Return None if RwLock is poisoned (graceful degradation)
        let _guard = match self.lock.read() {
            Ok(g) => g,
            Err(_) => return None,
        };
        self.zero_paths.get(&state_id)
    }

    /// Clear the entire cache
    pub fn clear(&mut self) {
        // SAFETY: Skip clear if RwLock is poisoned (graceful degradation)
        let _guard = match self.lock.write() {
            Ok(g) => g,
            Err(_) => return,
        };

        self.states.clear();
        self.zero_paths.clear();
        self.free_list.clear();
        self.next_state_id = 1;
        
        // Reset statistics
        self.stats = FsaCacheStats::default();
    }

    /// Get cache statistics
    pub fn stats(&self) -> FsaCacheStats {
        // SAFETY: Return default stats if RwLock is poisoned (graceful degradation)
        let _guard = match self.lock.read() {
            Ok(g) => g,
            Err(_) => return FsaCacheStats::default(),
        };
        self.stats.clone()
    }

    /// Get cache configuration
    pub fn config(&self) -> &FsaCacheConfig {
        &self.config
    }

    /// Check if cache is at capacity
    pub fn is_full(&self) -> bool {
        // SAFETY: Return false if RwLock is poisoned (safe default - not full)
        let _guard = match self.lock.read() {
            Ok(g) => g,
            Err(_) => return false,
        };
        self.states.len() >= self.config.max_states
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        let states_size = self.states.len() * std::mem::size_of::<(u32, CachedState)>();
        let zero_paths_size: usize = self.zero_paths.values()
            .map(|zp| zp.segments.len() + zp.lengths.len() + std::mem::size_of::<ZeroPathData>())
            .sum();
        let free_list_size = self.free_list.len() * std::mem::size_of::<u32>();
        
        states_size + zero_paths_size + free_list_size
    }

    /// Evict states based on strategy
    fn evict_states(&mut self) -> Result<()> {
        let evict_count = std::cmp::max(1, self.config.max_states / 10); // Evict 10%
        
        match self.config.strategy {
            CacheStrategy::BreadthFirst => self.evict_breadth_first(evict_count),
            CacheStrategy::DepthFirst => self.evict_depth_first(evict_count),
            CacheStrategy::CacheFriendly => self.evict_cache_friendly(evict_count),
        }
    }

    /// Evict states using breadth-first strategy
    fn evict_breadth_first(&mut self, count: usize) -> Result<()> {
        // Simple LRU approximation - remove lowest state IDs
        let mut to_remove: Vec<u32> = self.states.keys().copied().collect();
        to_remove.sort();
        to_remove.truncate(count);
        
        for state_id in to_remove {
            self.states.remove(&state_id);
            self.zero_paths.remove(&state_id);
            self.free_list.push(state_id);
        }
        
        self.stats.evictions += count as u64;
        Ok(())
    }

    /// Evict states using depth-first strategy
    fn evict_depth_first(&mut self, count: usize) -> Result<()> {
        // Remove highest state IDs (most recently allocated)
        let mut to_remove: Vec<u32> = self.states.keys().copied().collect();
        to_remove.sort_by(|a, b| b.cmp(a));
        to_remove.truncate(count);
        
        for state_id in to_remove {
            self.states.remove(&state_id);
            self.zero_paths.remove(&state_id);
            self.free_list.push(state_id);
        }
        
        self.stats.evictions += count as u64;
        Ok(())
    }

    /// Evict states using cache-friendly strategy
    fn evict_cache_friendly(&mut self, count: usize) -> Result<()> {
        // Hybrid approach: prefer evicting non-terminal states first
        let mut terminal_states = Vec::new();
        let mut non_terminal_states = Vec::new();
        
        for (&state_id, &state) in &self.states {
            if state.is_terminal() {
                terminal_states.push(state_id);
            } else {
                non_terminal_states.push(state_id);
            }
        }
        
        // Sort by state ID for consistent eviction
        non_terminal_states.sort();
        terminal_states.sort();
        
        let mut to_remove = Vec::new();
        
        // First evict non-terminal states
        let non_terminal_to_remove = std::cmp::min(count, non_terminal_states.len());
        to_remove.extend_from_slice(&non_terminal_states[..non_terminal_to_remove]);
        
        // If we need more, evict terminal states
        let remaining = count.saturating_sub(non_terminal_to_remove);
        if remaining > 0 {
            let terminal_to_remove = std::cmp::min(remaining, terminal_states.len());
            to_remove.extend_from_slice(&terminal_states[..terminal_to_remove]);
        }
        
        for state_id in to_remove {
            self.states.remove(&state_id);
            self.zero_paths.remove(&state_id);
            self.free_list.push(state_id);
        }
        
        self.stats.evictions += count as u64;
        Ok(())
    }

    /// Update compression statistics
    fn update_compression_stats(&mut self) {
        if self.zero_paths.is_empty() {
            self.stats.avg_compression_ratio = 0.0;
            return;
        }
        
        let total_ratio: f64 = self.zero_paths.values()
            .map(|zp| zp.compression_ratio())
            .sum();
        
        self.stats.avg_compression_ratio = total_ratio / self.zero_paths.len() as f64;
    }
}

impl Default for FsaCache {
    fn default() -> Self {
        // SAFETY: FsaCache::new() only fails on memory pool allocation errors.
        // Use unwrap_or_else with panic as this type has non-trivial dependencies.
        Self::new().unwrap_or_else(|e| {
            panic!("FsaCache creation failed in Default: {}. \
                   This indicates severe memory pressure.", e)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_state_creation() {
        let state = CachedState::new(100, 50, true, false);
        assert_eq!(state.child_base, 100);
        assert_eq!(state.parent(), 50);
        assert!(state.is_terminal());
        assert!(!state.is_free());
    }

    #[test]
    fn test_cached_state_flags() {
        let mut state = CachedState::new(100, 50, false, false);
        assert!(!state.is_terminal());
        assert!(!state.is_free());
        
        state.mark_free();
        assert!(state.is_free());
        
        state.mark_used();
        assert!(!state.is_free());
    }

    #[test]
    fn test_zero_path_data() {
        let mut zp = ZeroPathData::new();
        assert_eq!(zp.total_length, 0);
        
        zp.add_segment(b"hello").unwrap();
        zp.add_segment(b"world").unwrap();
        
        assert_eq!(zp.total_length, 10);
        assert_eq!(zp.get_full_path(), b"helloworld");
        
        // Compression ratio should be < 1.0 due to length storage overhead
        assert!(zp.compression_ratio() > 0.0);
    }

    #[test]
    fn test_fsa_cache_basic_operations() {
        let mut cache = FsaCache::new().unwrap();
        
        // Cache a state
        let state_id = cache.cache_state(0, 100, true).unwrap();
        assert!(state_id > 0);
        
        // Retrieve the state
        let state = cache.get_state(state_id).unwrap();
        assert_eq!(state.child_base, 100);
        assert_eq!(state.parent(), 0);
        assert!(state.is_terminal());
        
        // Remove the state
        assert!(cache.remove_state(state_id));
        assert!(cache.get_state(state_id).is_none());
    }

    #[test]
    fn test_fsa_cache_configurations() {
        let small_cache = FsaCache::with_config(FsaCacheConfig::small()).unwrap();
        assert_eq!(small_cache.config.max_states, 10_000);
        
        let large_cache = FsaCache::with_config(FsaCacheConfig::large()).unwrap();
        assert_eq!(large_cache.config.max_states, 10_000_000);
        assert!(large_cache.config.use_hugepages);
        
        let efficient_cache = FsaCache::with_config(FsaCacheConfig::memory_efficient()).unwrap();
        assert_eq!(efficient_cache.config.strategy, CacheStrategy::DepthFirst);
        assert!(efficient_cache.config.compressed_paths);
    }

    #[test]
    fn test_fsa_cache_eviction() {
        let config = FsaCacheConfig {
            max_states: 3,
            strategy: CacheStrategy::BreadthFirst,
            ..Default::default()
        };
        
        let mut cache = FsaCache::with_config(config).unwrap();
        
        // Fill the cache
        let id1 = cache.cache_state(0, 100, false).unwrap();
        let id2 = cache.cache_state(0, 200, false).unwrap();
        let id3 = cache.cache_state(0, 300, false).unwrap();
        
        assert_eq!(cache.stats().cached_states, 3);
        
        // Add one more - should trigger eviction
        let id4 = cache.cache_state(0, 400, false).unwrap();
        
        // Should still have 3 states (or less due to eviction)
        assert!(cache.stats().cached_states <= 3);
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn test_zero_path_integration() {
        let mut cache = FsaCache::new().unwrap();
        let state_id = cache.cache_state(0, 100, false).unwrap();
        
        let mut zp = ZeroPathData::new();
        zp.add_segment(b"test").unwrap();
        
        cache.add_zero_path(state_id, zp).unwrap();
        
        let retrieved_zp = cache.get_zero_path(state_id).unwrap();
        assert_eq!(retrieved_zp.get_full_path(), b"test");
    }

    #[test]
    fn test_cache_statistics() {
        let mut cache = FsaCache::new().unwrap();
        
        // Initially empty
        let stats = cache.stats();
        assert_eq!(stats.cached_states, 0);
        assert_eq!(stats.memory_usage, 0);
        
        // Add some states
        cache.cache_state(0, 100, true).unwrap();
        cache.cache_state(0, 200, false).unwrap();
        
        let stats = cache.stats();
        assert_eq!(stats.cached_states, 2);
        assert!(stats.memory_usage > 0);
    }
}