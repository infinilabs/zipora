//! # Zipora: High-Performance Data Structures and Compression
//!
//! This crate provides a comprehensive Rust implementation of advanced data structures and compression algorithms,
//! offering high-performance solutions with modern Rust design.
//!
//! ## Key Features
//!
//! - **Fast Containers**: Optimized vector and string types with zero-copy semantics
//! - **Succinct Data Structures**: Rank-select operations with SIMD optimizations  
//! - **Advanced Tries**: LOUDS, Critical-Bit, and Patricia tries with full FSA support
//! - **Blob Storage**: Memory-mapped and compressed blob storage systems
//! - **Entropy Coding**: Huffman, rANS, and dictionary-based compression algorithms
//! - **Memory Management**: Advanced allocators including memory pools and bump allocators
//! - **Specialized Algorithms**: Suffix arrays, radix sort, and multi-way merge
//! - **Fiber-based Concurrency**: High-performance async/await with work-stealing execution
//! - **Real-time Compression**: Adaptive algorithms with strict latency guarantees
//! - **C FFI Support**: Complete C API compatibility layer for gradual migration
//! - **Memory Safety**: All the performance of C++ with Rust's memory safety guarantees
//!
//! ## Quick Start
//!
//! ```rust
//! use zipora::{
//!     FastVec, ValVec32, SmallMap, FixedCircularQueue, AutoGrowCircularQueue,
//!     FastStr, MemoryBlobStore, BlobStore, LoudsTrie, Trie, GoldHashMap,
//!     HuffmanEncoder, MemoryPool, PoolConfig, SuffixArray, FiberPool
//! };
//!
//! // High-performance vector with realloc optimization
//! let mut vec = FastVec::new();
//! vec.push(42).unwrap();
//!
//! // Memory-efficient 32-bit indexed vector
//! let mut vec32 = ValVec32::new();
//! vec32.push(42).unwrap();
//! println!("ValVec32 uses u32 indices vs usize for Vec, saving space on large collections");
//!
//! // Small map optimized for ≤8 elements
//! let mut small_map = SmallMap::new();
//! small_map.insert("key", "value").unwrap();
//!
//! // Fixed-size circular queue with lock-free operations
//! let mut fixed_queue: FixedCircularQueue<i32, 16> = FixedCircularQueue::new();
//! fixed_queue.push_back(1).unwrap();
//! assert_eq!(fixed_queue.pop_front(), Some(1));
//!
//! // Auto-growing circular queue
//! let mut auto_queue = AutoGrowCircularQueue::new();
//! for i in 0..100 { auto_queue.push_back(i).unwrap(); }
//!
//! // Zero-copy string operations
//! let s = FastStr::from_string("hello world");
//! println!("Hash: {:x}", s.hash_fast());
//!
//! // Advanced trie operations
//! let mut trie = LoudsTrie::new();
//! trie.insert(b"hello").unwrap();
//! assert!(trie.contains(b"hello"));
//!
//! // High-performance hash map
//! let mut map = GoldHashMap::new();
//! map.insert("key", "value").unwrap();
//! ```

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod algorithms;
pub mod blob_store;
pub mod compression;
pub mod concurrency;
pub mod containers;
pub mod entropy;
pub mod error;
pub mod fsa;
pub mod hash_map;
pub mod io;
pub mod memory;
pub mod string;
pub mod succinct;

// Re-export core types
pub use containers::{
    AutoGrowCircularQueue,
    EasyHashMap,
    EasyHashMapBuilder,
    EasyHashMapStats,
    // Core containers
    FastVec,
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
    // Phase 2 specialized containers
    UintVector,
    // Phase 1 specialized containers
    ValVec32,
    // Phase 3 advanced containers
    ZoSortedStrVec,
    ZoSortedStrVecIter,
    ZoSortedStrVecRange,
};
pub use error::{Result, ZiporaError};
pub use string::FastStr;
pub use succinct::{
    BitVector,
    BitwiseOp,
    BuilderOptions,
    CpuFeatures,
    MixedDimensionView,
    PerformanceStats,
    RankSelect256,
    RankSelectBuilder,
    RankSelectFew,
    RankSelectFewBuilder,
    RankSelectInterleaved256,
    RankSelectMixedIL256,
    RankSelectMixedSE512,
    RankSelectMixedXL256,
    RankSelectMultiDimensional,
    // Advanced rank/select variants (Phase 7A)
    RankSelectOps,
    RankSelectPerformanceOps,
    RankSelectSe256,
    RankSelectSeparated256,
    RankSelectSeparated512,
    RankSelectSimple,
    RankSelectSparse,
    SimdCapabilities,
    // SIMD operations
    SimdOps,
    bulk_popcount_simd,
    bulk_rank1_simd,
    bulk_select1_simd,
};

// Re-export Phase 1 implementations
pub use blob_store::{BlobStore, MemoryBlobStore, PlainBlobStore};
pub use fsa::{
    CritBitTrie, DoubleArrayTrie, DoubleArrayTrieBuilder, DoubleArrayTrieConfig,
    FiniteStateAutomaton, LoudsTrie, PatriciaTrie, Trie,
};
pub use io::{DataInput, DataOutput, VarInt};

// Re-export Phase 2 implementations
pub use hash_map::GoldHashMap;

// Re-export Phase 2.5 implementations (memory mapping)
#[cfg(feature = "mmap")]
pub use io::{MemoryMappedInput, MemoryMappedOutput};

// Re-export Phase 3 implementations (entropy coding)
pub use blob_store::{
    DictionaryBlobStore, EntropyAlgorithm, EntropyCompressionStats, HuffmanBlobStore, RansBlobStore,
};
pub use entropy::dictionary::Dictionary;
pub use entropy::rans::RansSymbol;
pub use entropy::{
    DictionaryBuilder, DictionaryCompressor, EntropyStats, HuffmanDecoder, HuffmanEncoder,
    HuffmanTree, OptimizedDictionaryCompressor, RansDecoder, RansEncoder, RansState,
};

// Re-export Phase 4 implementations (memory management)
pub use memory::{
    BumpAllocator,
    BumpArena,
    CACHE_LINE_SIZE,
    CacheAlignedVec,
    MemoryConfig,
    MemoryPool,
    MemoryStats,
    NumaPoolStats,
    NumaStats,
    PoolConfig,
    PooledBuffer,
    PooledVec,
    // Secure memory management
    SecureMemoryPool,
    SecurePoolConfig,
    SecurePoolStats,
    SecurePooledPtr,
    clear_numa_pools,
    get_global_pool_for_size,
    get_global_secure_pool_stats,
    get_numa_stats,
    get_optimal_numa_node,
    init_numa_pools,
    numa_alloc_aligned,
    numa_dealloc,
    set_current_numa_node,
    size_to_class,
};

#[cfg(target_os = "linux")]
pub use memory::{HugePage, HugePageAllocator};

// Re-export Phase 4 implementations (algorithms)
pub use algorithms::{
    AlgorithmConfig, ExternalSort, LcpArray, LoserTree, LoserTreeConfig, MergeSource, MultiWayMerge,
    RadixSort, RadixSortConfig, ReplaceSelectSort, ReplaceSelectSortConfig, SuffixArray,
    SuffixArrayBuilder, TournamentNode,
};

// Re-export Phase 5 implementations (concurrency)
pub use concurrency::{
    AsyncBlobStore, AsyncFileStore, AsyncMemoryBlobStore, ConcurrencyConfig, Fiber, FiberHandle,
    FiberId, FiberPool, FiberPoolConfig, FiberStats, ParallelLoudsTrie, ParallelTrieBuilder,
    Pipeline, PipelineBuilder, PipelineStage, PipelineStats, Task, WorkStealingExecutor,
    WorkStealingQueue,
};

// Re-export Phase 5 implementations (compression)
pub use compression::{
    AdaptiveCompressor, AdaptiveConfig, Algorithm, CompressionMode, CompressionProfile,
    CompressionStats, Compressor, CompressorFactory, PerformanceRequirements, RealtimeCompressor,
    RealtimeConfig,
};

#[cfg(feature = "zstd")]
pub use blob_store::ZstdBlobStore;

// Type aliases for compatibility
/// State identifier type for FSA operations
pub type StateId = u32;
/// Record identifier type for blob store operations  
pub type RecordId = u32;

#[cfg(feature = "ffi")]
pub mod ffi;

/// Library version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if SIMD optimizations are available
pub fn has_simd_support() -> bool {
    #[cfg(target_feature = "avx2")]
    {
        true
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        false
    }
}

/// Initialize the library (currently no-op, for future use)
pub fn init() {
    log::debug!("Initializing zipora v{}", VERSION);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        init();
        assert!(VERSION.len() > 0);
    }

    #[test]
    fn test_version_info() {
        assert!(VERSION.len() > 0);
        assert!(VERSION.contains('.'));
        // Version should be semver format like "0.1.0"
        let parts: Vec<&str> = VERSION.split('.').collect();
        assert!(parts.len() >= 2);
    }

    #[test]
    fn test_simd_support() {
        // Test that function doesn't panic
        let has_simd = has_simd_support();

        // On most platforms, this will be false unless specifically compiled with AVX2
        #[cfg(target_feature = "avx2")]
        assert!(has_simd);

        #[cfg(not(target_feature = "avx2"))]
        assert!(!has_simd);
    }

    #[test]
    fn test_type_aliases() {
        // Test that type aliases are properly defined
        let _state_id: StateId = 42;
        let _record_id: RecordId = 123;

        // Verify they're u32 as expected
        assert_eq!(std::mem::size_of::<StateId>(), 4);
        assert_eq!(std::mem::size_of::<RecordId>(), 4);
    }

    #[test]
    fn test_re_exports() {
        // Test that main types are properly re-exported
        let _vec = FastVec::<i32>::new();
        let _str = FastStr::from_string("test");
        let _bv = BitVector::new();

        // Test error types
        let _err = ZiporaError::invalid_data("test");
        assert!(std::any::type_name::<Result<()>>().contains("ZiporaError"));
    }

    #[test]
    fn test_multiple_init_calls() {
        // Calling init multiple times should be safe
        init();
        init();
        init();
        // Should not panic or cause issues
    }
}
