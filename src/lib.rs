// P0.1: Suppress warning categories that represent future cleanup work, not bugs.
// These will be re-enabled as modules are cleaned up (P1-P3 tasks in plan.md).
#![allow(missing_docs)]
// P3.7: documentation task
// #![allow(dead_code)] // P3.3/P0.6: dead code removal is a separate task
// #![allow(unused_variables)] // Stub implementations have unused params
#![allow(unused_imports)] // Will be cleaned with dead code removal
#![allow(unused_mut)] // Compiler can determine mutability needs
#![allow(unused_doc_comments)] // Orphan doc comments from refactoring
#![allow(redundant_semicolons)] // Style nit
#![allow(clippy::needless_return)] // Style nit
#![allow(clippy::incompatible_msrv)]
// AVX-512 intrinsics require newer toolchain than 1.88.0
// #![allow(unused_unsafe)] // Nested unsafe blocks from SIMD dispatch macros
#![allow(unused_parens)] // Style nit

//! # Zipora: High-Performance Data Structures and Compression
//!
//! This crate provides a comprehensive Rust implementation of advanced data structures and compression algorithms,
//! offering high-performance solutions with modern Rust design.
//!
//! ## Key Features
//!
//! - **Fast Containers**: Optimized vector and string types with zero-copy semantics
//! - **Specialized Hash Maps**: Golden ratio optimized, string-optimized, and small inline maps
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
//!     FastStr, MemoryBlobStore, BlobStore, ZiporaTrie, ZiporaTrieConfig, Trie,
//!     ZiporaHashMap, ZiporaHashMapConfig,
//!     HuffmanEncoder, MemoryPool, PoolConfig, SuffixArray, FiberPool,
//!     RankSelectInterleaved256,
//! };
//! use std::collections::hash_map::RandomState;
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
//! // Advanced trie operations (unified ZiporaTrie)
//! let mut trie: ZiporaTrie<RankSelectInterleaved256> =
//!     ZiporaTrie::with_config(ZiporaTrieConfig::default());
//! trie.insert(b"hello").unwrap();
//! assert!(trie.contains(b"hello"));
//!
//! // High-performance hash maps (unified ZiporaHashMap)
//! let mut map: ZiporaHashMap<&str, &str, RandomState> = ZiporaHashMap::new().unwrap();
//! map.insert("key", "value").unwrap();
//!
//! // Cache-optimized hash map
//! let mut cache_map: ZiporaHashMap<&str, &str, RandomState> = ZiporaHashMap::with_config(
//!     ZiporaHashMapConfig::cache_optimized()
//! ).unwrap();
//! cache_map.insert("optimal", "growth").unwrap();
//!
//! // String-optimized hash map with interning (memory efficient for string keys)
//! let mut string_map: ZiporaHashMap<&str, i32, RandomState> = ZiporaHashMap::with_config(
//!     ZiporaHashMapConfig::string_optimized()
//! ).unwrap();
//! string_map.insert("interned", 42).unwrap();
//!
//! // Small hash map with inline storage (zero allocations for ≤N elements)
//! let mut small_hash_map: ZiporaHashMap<&str, i32, RandomState> = ZiporaHashMap::with_config(
//!     ZiporaHashMapConfig::small_inline(4)
//! ).unwrap();
//! small_hash_map.insert("inline", 1).unwrap();
//! ```

// #![warn(missing_docs)] -- suppressed at top of file (P3.7 task)
#![deny(unsafe_op_in_unsafe_fn)]

pub mod algorithms;
pub mod blob_store;
pub mod cache;
pub mod compression;
pub mod concurrency;
pub mod config;
pub mod containers;
pub mod dev_infrastructure;
pub mod entropy;
pub mod error;
pub mod error_recovery;
pub mod fsa;
pub mod hash_map;
pub mod io;
pub mod memory;
pub mod scoring;
pub mod simd;
pub mod statistics;
pub mod string;
pub mod succinct;
pub mod system;
pub mod thread;

// Re-export core types
pub use containers::{
    AutoGrowCircularQueue,
    BlockSize,
    CompressionStrategy,
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
    // Phase 2 specialized containers
    IntVec,
    PackedInt,
    SmallMap,
    SortableStrIter,
    SortableStrSortedIter,
    SortableStrVec,
    UintVector,
    // Phase 1 specialized containers
    ValVec32,
    // Phase 3 advanced containers
    ZoSortedStrVec,
    ZoSortedStrVecIter,
    ZoSortedStrVecRange,
};
pub use error::{Result, ZiporaError};
pub use error_recovery::{
    verify_alignment, verify_allocation_success, verify_bounds_check, verify_power_of_2,
    verify_range_check,
};
pub use string::{
    FastStr, LexIteratorBuilder, LexicographicIterator, LineProcessor, LineProcessorConfig,
    LineProcessorStats, LineSplitter, SortedVecLexIterator, StreamingLexIterator, UnicodeAnalysis,
    UnicodeProcessor, Utf8ToUtf32Iterator, utf8_byte_count, validate_utf8_and_count_chars,
};
pub use succinct::{
    AdaptiveMultiDimensional,
    // Advanced optimization variants
    AdaptiveRankSelect,
    BitVector,
    BitwiseOp,
    // BMI2 acceleration
    Bmi2Accelerator,
    Bmi2BitOps,
    Bmi2BlockOps,
    Bmi2Capabilities,
    Bmi2PrefetchOps,
    Bmi2RangeOps,
    Bmi2RankOps,
    Bmi2SelectOps,
    Bmi2SequenceOps,
    Bmi2Stats,
    BuilderOptions,
    DataProfile,
    PerformanceStats,
    RankSelect256,
    RankSelectBuilder,
    RankSelectInterleaved256,
    // Advanced rank/select variants (Phase 7A)
    RankSelectOps,
    RankSelectPerformanceOps,
    SelectionCriteria,
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
    BitVectorType,
    CompressedSparseTrie,
    CompressionStrategy as TrieCompressionStrategy,
    ConcurrencyLevel,
    CritBitTrie,
    // Primary trie implementation — 8 bytes/state, faithful C++ reference port
    DoubleArrayTrie,
    DoubleArrayTrieMap,
    FiniteStateAutomaton,
    FragmentStats,
    MapValue,
    // Other trie strategies (available via explicit config)
    NestedLoudsTrie,
    NestedTrieStats,
    NestingConfig,
    PatriciaTrie,
    RankSelectType,
    ReaderToken,
    StorageStrategy,
    Trie,
    TrieStrategy,
    WriterToken,
    ZiporaTrie,
    ZiporaTrieConfig,
};
pub use io::{DataInput, DataOutput, VarInt};

// Re-export Phase 2 implementations
pub use hash_map::{
    CacheMetrics,
    CombineStrategy,
    GOLDEN_LOAD_FACTOR,
    GOLDEN_RATIO_FRAC_DEN,
    GOLDEN_RATIO_FRAC_NUM,
    HashCombinable,
    HashFunctionBuilder,
    HashMapStats,
    HashStrategy,
    OptimizationStrategy,
    Prefetcher,
    // SIMD and cache optimization utilities
    SimdStringOps,
    SimdTier,
    StorageStrategy as HashStorageStrategy,
    // Core unified hash map implementation
    ZiporaHashMap,
    ZiporaHashMapConfig,
    advanced_hash_combine,
    // Hash function utilities
    fabo_hash_combine_u32,
    fabo_hash_combine_u64,
    golden_ratio_next_size,
    optimal_bucket_count,
};

// Re-export Phase 2.5 implementations (memory mapping)
#[cfg(feature = "mmap")]
pub use io::{MemoryMappedInput, MemoryMappedOutput};

// Re-export Phase 3 implementations (entropy coding)
pub use blob_store::{
    DictionaryBlobStore, EntropyAlgorithm, EntropyCompressionStats, HuffmanBlobStore, RansBlobStore,
};
pub use entropy::dictionary::Dictionary;
pub use entropy::rans::Rans64Symbol;
pub use entropy::{
    DictionaryBuilder, DictionaryCompressor, EntropyStats, HuffmanDecoder, HuffmanEncoder,
    HuffmanTree, OptimizedDictionaryCompressor, Rans64Encoder, RansDecoder, RansState,
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
    AlgorithmConfig, EnhancedLoserTree, LcpArray, LoserTreeConfig, MergeSource, MultiWayMerge,
    RadixSort, RadixSortConfig, SuffixArray, SuffixArrayBuilder, TournamentNode, simd_block_filter,
    simd_gallop_to,
};

// Re-export Phase 5 implementations (concurrency)
#[cfg(feature = "async")]
pub use concurrency::{
    FiberHandle, FiberPool, FiberPoolBuilder, FiberPoolConfig, FiberStats, ParallelLoudsTrie,
    ParallelTrieBuilder, Pipeline, PipelineBuilder, PipelineStage, PipelineStats, Task,
    WorkStealingExecutor, WorkStealingQueue,
};

// Re-export compression implementations
pub use compression::{
    AdaptiveCompressor, AdaptiveConfig, Algorithm, CompressionProfile, CompressionStats,
    Compressor, CompressorFactory, PerformanceRequirements,
};

// Re-export System Utilities (Phase 10A)
pub use system::{
    // Base64 SIMD
    AdaptiveBase64,
    BenchmarkSuite,
    HighPrecisionTimer,
    KernelInfo,
    PageAlignedAlloc,
    // Performance profiling
    PerfTimer,
    ProfiledFunction,
    // CPU feature detection
    RuntimeCpuFeatures,
    SimdBase64Decoder,
    SimdBase64Encoder,
    // Virtual memory management
    VmManager,
    base64_decode_simd,
    base64_encode_simd,
    get_cpu_features,
    get_kernel_info,
    has_cpu_feature,
    vm_prefetch,
};
#[cfg(feature = "async")]
pub use system::{BidirectionalPipe, ProcessExecutor, ProcessManager, ProcessPool};

// Re-export Development Infrastructure (Phase 10B)
pub use dev_infrastructure::{
    AccumulatorStats,
    AutoRegister,
    BenchmarkResult,
    BenchmarkSuite as DevBenchmarkSuite,
    FactoryBuilder,
    // Factory Pattern
    FactoryRegistry,
    Factoryable,
    GlobalFactory,
    GlobalStatsRegistry,
    // Debugging Framework
    HighPrecisionTimer as DevHighPrecisionTimer,
    // Statistical Analysis
    Histogram,
    HistogramStats,
    MemoryDebugger,
    MemoryStats as DevMemoryStats,
    MultiDimensionalStats,
    PerformanceProfiler,
    ScopedTimer,
    StatAccumulator,
    StatIndex,
    U32Histogram,
    U64Histogram,
    format_duration,
    global_factory,
    global_memory_debugger,
    global_profiler,
    global_stats,
};

// Re-export Advanced Statistics and Monitoring Framework
pub use statistics::{
    BufferMetadata,
    BufferPoolConfig,
    BufferPoolManager,
    BufferPriority,
    CompressionEstimates,
    CompressionStats as StatsCompressionStats,
    // Buffer management
    ContextBuffer,
    DefaultStatisticsContext,
    DistributionInfo,
    DistributionStats,
    // Entropy analysis
    EntropyAnalyzer,
    EntropyAnalyzerCollection,
    EntropyConfig,
    EntropyResults,
    ErrorStats,
    ErrorType,
    FragmentationAnalysis,
    // Histogram framework
    FreqHist,
    FreqHistO1,
    FreqHistO2,
    GlobalMemoryTracker,
    HistogramCollection,
    HistogramData,
    HistogramDataO1,
    HistogramDataO2,
    LocalMemoryTracker,
    // Memory tracking
    MemoryBreakdown,
    MemoryCategory,
    MemoryStats as StatsMemoryStats,
    OperationProfile,
    PerfTimer as StatsPerfTimer,
    PerformanceStats as StatsPerformanceStats,
    PoolStatistics,
    ProfiledOperation,
    // Profiling
    Profiler,
    ProfilerConfig,
    // High-precision timing
    Profiling,
    QDuration,
    QTime,
    ScopedBuffer,
    ScopedTimer as StatsScopedTimer,
    StatisticsContext,
    TimerCollection,
    TimingStats,
    TrackedObject,
    // Core statistics
    TrieStatistics,
    global_profiler as stats_global_profiler,
    init_global_profiler,
    str_date_time_now,
};

// Re-export Low-Level Synchronization (Phase 11A)
pub use thread::{
    AsAtomic,
    AtomicBitOps,
    // Atomic Operations Framework
    AtomicExt,
    DefaultPlatformSync,
    // Instance-Specific Thread-Local Storage
    InstanceTls,
    OwnerTls,
    // Linux Futex Integration
    PlatformSync,
    TlsPool,
    memory_ordering,
    spin_loop_hint,
};

// Re-export LRU Page Cache (New Feature)
pub use cache::{
    BufferPool,
    BufferPoolStats,
    CACHE_LINE_SIZE as CACHE_CACHE_LINE_SIZE,
    CacheBuffer,
    // Cache operation types
    CacheError,
    CacheHitType,
    // Statistics and monitoring
    CacheStatistics,
    CacheStatsSnapshot,
    EvictionAlgorithm,
    EvictionConfig,
    FileId,
    HUGE_PAGE_SIZE,
    KernelAdvice,
    // Configuration types
    LockingConfig,
    // Core cache types
    LruPageCache,
    MAX_SHARDS,
    MaintenanceConfig,
    MemoryConfig as CacheMemoryConfig,
    NodeIndex,
    PAGE_BITS,
    // Constants
    PAGE_SIZE,
    PageCacheConfig,
    PageId,
    PerformanceConfig,
    SingleLruPageCache,
    WarmingStrategy,
    get_shard_id,
    // Utility functions
    hash_file_page,
    prefetch_hint,
};

// Platform-specific re-exports
#[cfg(target_os = "linux")]
pub use thread::{
    FutexCondvar, FutexGuard, FutexMutex, FutexReadGuard, FutexRwLock, FutexWriteGuard, LinuxFutex,
};

#[cfg(target_arch = "x86_64")]
pub use thread::x86_64_optimized;

#[cfg(target_arch = "aarch64")]
pub use thread::aarch64_optimized;

// Macros are re-exported automatically from dev_infrastructure module

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
