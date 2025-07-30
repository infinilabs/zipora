//! # Infini-Zip: High-Performance Data Structures and Compression
//!
//! This crate provides a Rust implementation of advanced data structures and compression algorithms,
//! offering high-performance solutions originally inspired by the topling-zip C++ library.
//!
//! ## Key Features
//!
//! - **Fast Containers**: Optimized vector and string types with zero-copy semantics
//! - **Succinct Data Structures**: Rank-select operations with SIMD optimizations  
//! - **Advanced Tries**: LOUDS tries, compressed sparse Patricia tries
//! - **Blob Storage**: Memory-mapped and compressed blob storage systems
//! - **Memory Safety**: All the performance of C++ with Rust's memory safety guarantees
//!
//! ## Quick Start
//!
//! ```rust
//! use infini_zip::{FastVec, FastStr};
//!
//! // High-performance vector with realloc optimization
//! let mut vec = FastVec::new();
//! vec.push(42);
//! 
//! // Zero-copy string operations
//! let s = FastStr::from_string("hello world");
//! println!("Length: {}", s.len());
//! ```

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod containers;
pub mod string;
pub mod succinct;
pub mod io;
pub mod blob_store;
pub mod fsa;
pub mod hash_map;
pub mod entropy;
pub mod error;
pub mod memory;
pub mod algorithms;
pub mod concurrency;
pub mod compression;

#[cfg(debug_assertions)]
pub mod debug_crit_bit;

// Re-export core types
pub use containers::FastVec;
pub use string::FastStr;
pub use succinct::{BitVector, RankSelect256, RankSelectSe256};
pub use error::{ToplingError, Result};

// Re-export Phase 1 implementations
pub use blob_store::{BlobStore, MemoryBlobStore, PlainBlobStore};
pub use io::{DataInput, DataOutput, VarInt};
pub use fsa::{LoudsTrie, CritBitTrie, PatriciaTrie, Trie, FiniteStateAutomaton};

// Re-export Phase 2 implementations
pub use hash_map::GoldHashMap;

// Re-export Phase 2.5 implementations (memory mapping)
#[cfg(feature = "mmap")]
pub use io::{MemoryMappedInput, MemoryMappedOutput};

// Re-export Phase 3 implementations (entropy coding)
pub use entropy::{
    EntropyStats, HuffmanEncoder, HuffmanDecoder, HuffmanTree,
    RansEncoder, RansDecoder, RansState,
    DictionaryCompressor, DictionaryBuilder
};
pub use entropy::rans::RansSymbol;
pub use entropy::dictionary::Dictionary;
pub use blob_store::{
    HuffmanBlobStore, RansBlobStore, DictionaryBlobStore,
    EntropyAlgorithm, EntropyCompressionStats
};

// Re-export Phase 4 implementations (memory management)
pub use memory::{
    MemoryPool, PoolConfig, PooledVec, PooledBuffer,
    BumpAllocator, BumpArena, MemoryConfig, MemoryStats
};

#[cfg(target_os = "linux")]
pub use memory::{HugePage, HugePageAllocator};

// Re-export Phase 4 implementations (algorithms)
pub use algorithms::{
    SuffixArray, SuffixArrayBuilder, LcpArray,
    RadixSort, RadixSortConfig,
    MultiWayMerge, MergeSource, AlgorithmConfig
};

// Re-export Phase 5 implementations (concurrency)
pub use concurrency::{
    FiberPool, FiberPoolConfig, FiberHandle, FiberStats,
    Pipeline, PipelineStage, PipelineBuilder, PipelineStats,
    ParallelTrieBuilder, ParallelLoudsTrie,
    AsyncBlobStore, AsyncMemoryBlobStore, AsyncFileStore,
    WorkStealingQueue, WorkStealingExecutor, Task,
    Fiber, FiberId, ConcurrencyConfig
};

// Re-export Phase 5 implementations (compression)
pub use compression::{
    AdaptiveCompressor, CompressionProfile, AdaptiveConfig,
    RealtimeCompressor, RealtimeConfig, CompressionMode,
    Algorithm, Compressor, CompressorFactory, PerformanceRequirements, CompressionStats
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
    log::debug!("Initializing infini-zip v{}", VERSION);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        init();
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
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
        let _err = ToplingError::invalid_data("test");
        assert!(std::any::type_name::<Result<()>>().contains("ToplingError"));
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