//! ZiporaTrie - High-performance trie with strategy-based configuration
//!
//! This module provides the core trie implementation for Zipora, designed for
//! extreme performance following referenced project's focused implementation philosophy.
//!
//! # Performance-First Design
//!
//! **"One excellent implementation per data structure"** - referenced project approach
//!
//! ZiporaTrie achieves high performance through configurable strategies:
//! - **TrieStrategy**: Optimized algorithms (Patricia, CritBit, DoubleArray, LOUDS, CompressedSparse)
//! - **TrieStorageStrategy**: Memory layout optimization and succinct data structures
//! - **TrieCompressionStrategy**: Advanced compression techniques (path, fragment, hierarchical)
//! - **RankSelectStrategy**: High-performance rank/select backend selection
//!
//! # Hardware Acceleration Features
//!
//! - **SIMD Framework**: BMI2/AVX2/POPCNT acceleration with runtime detection
//! - **Cache Optimization**: Prefetching, alignment, and NUMA awareness
//! - **Succinct Structures**: Space-efficient rank/select with hardware acceleration
//! - **Memory Pool Integration**: SecureMemoryPool for high-performance allocation
//! - **Concurrent Access**: Lock-free and token-based synchronization

mod config;
mod map;
mod storage;
#[cfg(test)]
mod tests;
mod trie;
mod builder;
mod search;

pub use config::{
    BitVectorType, RankSelectType, TrieCompressionStrategy, TrieStorageStrategy, TrieStrategy,
    ZiporaTrieConfig,
};
pub use map::ZiporaTrieMap;
pub use trie::{MemoryStats, TrieIterator, ZiporaTrie};
