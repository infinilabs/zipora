//! High-performance container types
//!
//! This module provides optimized container types that prioritize performance
//! while maintaining Rust's safety guarantees.
//!
//! ## Core Containers
//!
//! - **`FastVec<T>`** - High-performance vector using realloc for growth
//!
//! ## Specialized Containers - Phase 1
//!
//! - **`ValVec32<T>`** - 32-bit indexed vectors for memory efficiency
//! - **`SmallMap<K,V>`** - Memory-efficient containers for small collections
//! - **`FixedCircularQueue<T, N>`** - Fixed-size circular buffer
//! - **`AutoGrowCircularQueue<T>`** - Dynamically resizing circular buffer
//!
//! ## Specialized Containers - Phase 2
//!
//! - **`UintVector`** - Compressed integer storage with 60-80% space reduction
//! - **`FixedLenStrVec<N>`** - Fixed-length string vector with SIMD optimizations
//! - **`SortableStrVec`** - Arena-based string storage with high-performance sorting
//!
//! ## Advanced Containers - Phase 3
//!
//! - **`ZoSortedStrVec`** - Zero-overhead sorted string collections with succinct structures
//! - **`GoldHashIdx<K,V>`** - Hash index for large value indirection and memory efficiency
//! - **`HashStrMap<V>`** - String-optimized hash map with automatic interning
//! - **`EasyHashMap<K,V>`** - Simplified hash map interface with builder pattern
//!
//! ## LRU Cache Containers - Phase 4
//!
//! - **`LruMap<K,V>`** - High-performance LRU cache with O(1) operations
//! - **`ConcurrentLruMap<K,V>`** - Thread-safe LRU cache with sharding for reduced contention

mod fast_vec;
pub mod specialized;

pub use fast_vec::FastVec;
pub use specialized::{
    AutoGrowCircularQueue,
    EasyHashMap,
    EasyHashMapBuilder,
    EasyHashMapStats,
    FixedCircularQueue,
    FixedLenStrVec,
    FixedStr4Vec,
    FixedStr8Vec,
    FixedStr16Vec,
    FixedStr32Vec,
    FixedStr64Vec,
    GoldHashIdx,
    HashStrMap,
    HashStrMapStats,
    SmallMap,
    SortableStrIter,
    SortableStrSortedIter,
    SortableStrVec,
    // Phase 2 containers
    IntVec,
    PackedInt,
    CompressionStrategy,
    BlockSize,
    UintVector,
    // Phase 1 containers
    ValVec32,
    // Phase 3 advanced containers
    ZoSortedStrVec,
    ZoSortedStrVecIter,
    ZoSortedStrVecRange,
    // Phase 4 LRU cache containers
    LruMap,
    LruMapConfig,
    LruMapStatistics,
    EvictionCallback,
    NoOpEvictionCallback,
    ConcurrentLruMap,
    ConcurrentLruMapConfig,
    ConcurrentLruMapStatistics,
    LoadBalancingStrategy,
};
