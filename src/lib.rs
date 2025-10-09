#![cfg_attr(feature = "nightly", feature(core_intrinsics))]

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

#![warn(missing_docs)]
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
pub mod simd;
pub mod string;
pub mod succinct;
pub mod statistics;
pub mod system;
pub mod thread;

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
    IntVec,
    PackedInt,
    CompressionStrategy,
    BlockSize,
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
    verify_alignment, verify_power_of_2, verify_allocation_success, 
    verify_bounds_check, verify_range_check
};
pub use string::{
    FastStr, LexicographicIterator, SortedVecLexIterator, StreamingLexIterator,
    LexIteratorBuilder, UnicodeProcessor, UnicodeAnalysis, Utf8ToUtf32Iterator,
    LineProcessor, LineProcessorConfig, LineProcessorStats, LineSplitter,
    utf8_byte_count, validate_utf8_and_count_chars,
};
pub use succinct::{
    BitVector,
    BitwiseOp,
    BuilderOptions,
    PerformanceStats,
    RankSelect256,
    RankSelectBuilder,
    RankSelectInterleaved256,
    // Advanced rank/select variants (Phase 7A)
    RankSelectOps,
    RankSelectPerformanceOps,
    // Advanced optimization variants
    AdaptiveRankSelect,
    AdaptiveMultiDimensional,
    DataProfile,
    SelectionCriteria,
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
    ZiporaTrie, ZiporaTrieConfig, TrieStrategy, StorageStrategy, CompressionStrategy as TrieCompressionStrategy,
    RankSelectType, BitVectorType, FiniteStateAutomaton, Trie,
    // Legacy compatibility exports
    DoubleArrayTrie, DoubleArrayTrieConfig, DoubleArrayTrieBuilder,
    NestedLoudsTrie, NestingConfig, NestingConfigBuilder, NestedTrieStats, FragmentStats,
    CompressedSparseTrie, ConcurrencyLevel, ReaderToken, WriterToken,
    PatriciaTrie, CritBitTrie,
};
pub use io::{DataInput, DataOutput, VarInt};

// Re-export Phase 2 implementations
pub use hash_map::{
    // Core unified hash map implementation
    ZiporaHashMap, ZiporaHashMapConfig, HashMapStats,
    HashStrategy, StorageStrategy as HashStorageStrategy, OptimizationStrategy,
    // Hash function utilities
    fabo_hash_combine_u32, fabo_hash_combine_u64, golden_ratio_next_size, optimal_bucket_count,
    advanced_hash_combine, HashFunctionBuilder, CombineStrategy, HashCombinable,
    GOLDEN_RATIO_FRAC_NUM, GOLDEN_RATIO_FRAC_DEN, GOLDEN_LOAD_FACTOR,
    // SIMD and cache optimization utilities
    SimdStringOps, SimdTier, CacheMetrics, Prefetcher,
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
    HuffmanTree, OptimizedDictionaryCompressor, RansDecoder, Rans64Encoder, RansState,
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

// Re-export System Utilities (Phase 10A)
pub use system::{
    // CPU feature detection
    RuntimeCpuFeatures, get_cpu_features, has_cpu_feature,
    // Performance profiling
    PerfTimer, BenchmarkSuite, HighPrecisionTimer, ProfiledFunction,
    // Process management
    ProcessManager, ProcessPool, BidirectionalPipe, ProcessExecutor,
    // Base64 SIMD
    AdaptiveBase64, SimdBase64Encoder, SimdBase64Decoder, base64_encode_simd, base64_decode_simd,
    // Virtual memory management
    VmManager, PageAlignedAlloc, KernelInfo, vm_prefetch, get_kernel_info,
};

// Re-export Development Infrastructure (Phase 10B)
pub use dev_infrastructure::{
    // Factory Pattern
    FactoryRegistry, GlobalFactory, AutoRegister, Factoryable, FactoryBuilder,
    global_factory,
    // Debugging Framework
    HighPrecisionTimer as DevHighPrecisionTimer, ScopedTimer, BenchmarkSuite as DevBenchmarkSuite,
    BenchmarkResult, MemoryDebugger, MemoryStats as DevMemoryStats, PerformanceProfiler,
    global_profiler, global_memory_debugger, format_duration,
    // Statistical Analysis
    Histogram, U32Histogram, U64Histogram, HistogramStats,
    StatAccumulator, AccumulatorStats, MultiDimensionalStats, GlobalStatsRegistry,
    global_stats, StatIndex,
};

// Re-export Advanced Statistics and Monitoring Framework  
pub use statistics::{
    // Core statistics
    TrieStatistics, MemoryStats as StatsMemoryStats, PerformanceStats as StatsPerformanceStats, 
    CompressionStats as StatsCompressionStats, DistributionStats, ErrorStats, TimingStats, 
    MemoryCategory, ErrorType,
    // Memory tracking
    MemorySize, MemoryBreakdown, GlobalMemoryTracker, TrackedObject, LocalMemoryTracker,
    FragmentationAnalysis,
    // High-precision timing
    Profiling, QTime, QDuration, PerfTimer as StatsPerfTimer, TimerCollection, TimerStats, 
    ScopedTimer as StatsScopedTimer, str_date_time_now,
    // Histogram framework
    FreqHist, FreqHistO1, FreqHistO2, HistogramData, HistogramDataO1, HistogramDataO2,
    HistogramCollection, GlobalHistogramStats,
    // Entropy analysis
    EntropyAnalyzer, EntropyConfig, EntropyResults, CompressionEstimates, DistributionInfo,
    SampleStats, EntropyAnalyzerCollection, GlobalEntropyStats,
    // Buffer management
    ContextBuffer, BufferMetadata, BufferPriority, StatisticsContext, DefaultStatisticsContext,
    BufferPoolManager, BufferPoolConfig, PoolStatistics, ScopedBuffer,
    // Profiling
    Profiler, ProfilerConfig, OperationProfile, GlobalProfilingStats, ProfiledOperation,
    global_profiler as stats_global_profiler, init_global_profiler,
};

// Re-export Low-Level Synchronization (Phase 11A)
pub use thread::{
    // Linux Futex Integration
    PlatformSync, DefaultPlatformSync,
    // Instance-Specific Thread-Local Storage
    InstanceTls, OwnerTls, TlsPool,
    // Atomic Operations Framework
    AtomicExt, AsAtomic, AtomicNode, AtomicStack, AtomicBitOps, spin_loop_hint,
    memory_ordering,
};

// Re-export LRU Page Cache (New Feature)
pub use cache::{
    // Core cache types
    LruPageCache, SingleLruPageCache, PageCacheConfig, CacheBuffer,
    // Configuration types
    LockingConfig, MemoryConfig as CacheMemoryConfig, KernelAdvice, PerformanceConfig,
    EvictionConfig, EvictionAlgorithm, WarmingStrategy, MaintenanceConfig,
    // Statistics and monitoring
    CacheStatistics, CacheStatsSnapshot, BufferPool, BufferPoolStats,
    // Cache operation types
    CacheError, CacheHitType, FileId, PageId, NodeIndex,
    // Utility functions
    hash_file_page, get_shard_id, prefetch_hint,
    // Constants
    PAGE_SIZE, PAGE_BITS, HUGE_PAGE_SIZE, MAX_SHARDS, CACHE_LINE_SIZE as CACHE_CACHE_LINE_SIZE,
};

// Platform-specific re-exports
#[cfg(target_os = "linux")]
pub use thread::{LinuxFutex, FutexMutex, FutexCondvar, FutexRwLock, FutexGuard, FutexReadGuard, FutexWriteGuard};

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
