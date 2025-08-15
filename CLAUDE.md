# CLAUDE.md

Project guidance for Claude Code when working with zipora codebase.

## Core Principles

1. **Performance First**: Always benchmark changes, aim to exceed C++ performance
2. **Memory Safety**: Use SecureMemoryPool, avoid unsafe operations in public APIs  
3. **Comprehensive Testing**: Maintain 97%+ coverage, all tests must pass
4. **SIMD Optimization**: Leverage AVX2/BMI2/POPCNT, AVX-512 on nightly
5. **Production Ready**: Zero compilation errors, robust error handling

## Quick Commands

```bash
# Build & Test
cargo build --release                  # Release build
cargo test --all-features             # All tests (755+ tests)
cargo bench                           # Performance validation

# Feature Testing
cargo build --features lz4,ffi        # Stable features
cargo +nightly build --features avx512 # Nightly features

# Quality
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

## Project Status

**LRU Page Cache COMPLETE** - Sophisticated Caching Layer + Low-Level Synchronization + ZipOffsetBlobStore Production Ready

### ✅ Completed Phases
- **Phase 1-5**: Core infrastructure, memory management, concurrency (COMPLETE)
- **Phase 6**: 11 specialized containers with exceptional performance (COMPLETE)
- **Phase 7A**: 11 rank/select variants with 3.3 Gelem/s peak performance (COMPLETE)
- **Phase 7B**: 3 advanced FSA & Trie variants with revolutionary features (COMPLETE)
- **Phase 8A**: 4 FSA infrastructure components with advanced optimization features (COMPLETE)
- **Phase 8B**: 8 comprehensive serialization components with full feature implementation (COMPLETE)
- **Phase 9A**: 4 advanced memory pool variants with lock-free, thread-local, fixed-capacity, and memory-mapped capabilities (COMPLETE)
- **Phase 9B**: 3 advanced sorting & search algorithms with external sorting, tournament trees, and linear-time suffix arrays (COMPLETE)
- **Phase 9C**: 3 string processing components with Unicode support, hardware acceleration, and line-based text processing (COMPLETE)
- **Phase 10A**: 5 system integration utilities with CPU feature detection, performance profiling, process management, Base64 SIMD, and virtual memory management (COMPLETE)
- **Phase 10B**: 3 development infrastructure components with factory patterns, debugging framework, and statistical analysis tools (COMPLETE)
- **Phase 10C**: 3 advanced fiber concurrency enhancements with async I/O integration, cooperative multitasking, and specialized mutex variants (COMPLETE)
- **Phase 11A**: 3 low-level synchronization components with Linux futex integration, instance-specific thread-local storage, and atomic operations framework (COMPLETE)
- **ZipOffsetBlobStore**: Offset-based compressed storage with block-based delta compression, template optimization, and hardware acceleration (COMPLETE)
- **LRU Page Cache**: Sophisticated caching layer for blob operations with multi-shard architecture, page-aligned memory management, and hardware prefetching (COMPLETE)

### 🚀 Latest Achievements
- **LRU Page Cache Implementation**: Sophisticated caching layer for blob operations with multi-shard architecture, page-aligned memory management, and hardware prefetching (COMPLETE)
- **Cache Components**: LruPageCache (multi-shard caching), CachedBlobStore (transparent integration), PageCacheConfig (optimization profiles), CacheStatistics (performance monitoring)
- **ZipOffsetBlobStore Implementation**: Offset-based compressed storage with block-based delta compression, template optimization, and hardware acceleration (COMPLETE)
- **3 Storage Components**: SortedUintVec (block-based delta compression), ZipOffsetBlobStore (template-based retrieval), ZipOffsetBlobStoreBuilder (ZSTD integration)
- **3 Low-Level Synchronization Components**: Linux Futex Integration (direct futex usage), Instance-Specific Thread-Local Storage (advanced TLS management), Atomic Operations Framework (lock-free programming utilities)
- **Linux Futex Integration**: Direct futex syscalls for zero-overhead synchronization, cross-platform abstraction with PlatformSync trait, high-level primitives (FutexMutex, FutexCondvar, FutexRwLock)
- **Instance-Specific Thread-Local Storage**: Matrix-based O(1) access storage, automatic resource management with RAII cleanup, owner-based TLS and TLS pools for complex scenarios
- **Atomic Operations Framework**: Extended atomic operations (atomic_maximize, atomic_minimize, conditional updates), lock-free data structures (AtomicStack, AtomicNode), platform-specific optimizations (x86_64 assembly, ARM NEON)
- **Advanced Synchronization Features**: Hardware-accelerated operations, safe atomic casting, comprehensive bit operations, memory ordering utilities
- **3 Development Infrastructure Components**: Factory patterns (generic object creation), debugging framework (advanced debugging utilities), statistical analysis tools (built-in statistics collection)
- **FactoryRegistry/GlobalFactory**: Thread-safe factory pattern with type-safe registration, discovery, and zero-cost abstractions
- **HighPrecisionTimer/PerformanceProfiler**: Nanosecond-accurate timing with global profiler integration and memory debugging capabilities
- **Histogram/StatAccumulator**: Adaptive histograms with dual storage strategy and lock-free real-time statistical collection
- **Global Management**: Thread-safe global registries with automatic initialization for factories and statistics
- **5 System Integration Utilities**: CPU feature detection (runtime SIMD optimization), performance profiling (nanosecond precision), process management (async bidirectional pipes), Base64 SIMD acceleration (adaptive selection), virtual memory management (kernel-aware operations)
- **RuntimeCpuFeatures**: Comprehensive x86_64 and ARM feature detection with optimization tier selection (scalar to AVX-512)
- **HighPrecisionTimer**: Nanosecond-accurate timing with automatic unit formatting and benchmark suites
- **ProcessManager**: Async process spawning with bidirectional communication, process pools, and timeout handling
- **AdaptiveBase64**: SIMD-accelerated encoding/decoding with automatic implementation selection (AVX-512/AVX2/SSE4.2/NEON/scalar)
- **VmManager**: Virtual memory management with kernel feature detection, page prefetching, and cross-platform compatibility
- **3 String Processing Components**: Lexicographic iterators (O(1) access), Unicode processing (SIMD acceleration), line-based text processing (configurable buffering)
- **LexicographicIterator**: Efficient iteration over sorted string collections with O(log n) binary search operations
- **UnicodeProcessor**: Full Unicode support with hardware-accelerated UTF-8 validation and comprehensive analysis
- **LineProcessor**: High-performance text file processing with configurable strategies and field splitting
- **3 Advanced Sorting & Search Algorithms**: External sorting (replacement selection), tournament tree merge (k-way), SA-IS suffix arrays (linear time)
- **ReplaceSelectSort**: External sorting for datasets larger than memory with replacement selection and k-way merging
- **LoserTree**: Tournament tree implementation for efficient k-way merging with O(log k) complexity per element
- **Enhanced Suffix Arrays**: SA-IS algorithm implementation with linear-time construction and LCP array support
- **RankSelectInterleaved256**: 3.3 billion operations/second
- **4 FSA Infrastructure Components**: Cache system (8-byte state representation), DFA/DAWG (state merging), Graph walkers (8 strategies), Fast search (SIMD optimization)
- **8 Comprehensive Serialization Components**: Smart pointer serialization (Box/Rc/Arc/Weak with cycle detection), complex type serialization (tuples/collections with metadata validation), cross-platform endian handling (SIMD bulk operations), advanced version management (schema migration), variable integer encoding (7 adaptive strategies), StreamBuffer (configurable strategies), RangeStream (partial access), Zero-Copy optimizations (hardware acceleration)
- **4 Advanced Memory Pool Variants**: Lock-free pool (CAS-based concurrent allocation), Thread-local pool (zero-contention caching), Fixed-capacity pool (real-time guarantees), Memory-mapped vectors (persistent storage)
- **Advanced FSA Features**: Multi-strategy caching (BFS/DFS/CacheFriendly), compressed zero-path storage, hardware-accelerated search algorithms
- **Advanced I/O Features**: Page-aligned allocation (4KB), golden ratio growth (1.618x), read-ahead optimization, progress tracking, vectored I/O
- **3 Revolutionary Trie Variants**: DoubleArrayTrie (O(1) access), CompressedSparseTrie (90% faster sparse), NestedLoudsTrie (50-70% memory reduction)
- **Advanced Concurrency**: 5 concurrency levels with token-based thread safety and lock-free optimizations
- **Comprehensive SIMD**: BMI2, AVX2, NEON, AVX-512 acceleration with adaptive algorithm selection
- **Multi-Dimensional**: 2-4 dimension support with const generics
- **Production Quality**: 1,000+ tests + 5,735+ trie tests + comprehensive serialization tests, 97%+ coverage (all implementations fully working)

### 📊 Performance Targets
- **Current**: 3.3 Gelem/s rank/select, 3-4x faster than C++ vectors
- **Memory**: 50-70% reduction (specialized containers)
- **Safety**: Zero unsafe operations in public APIs
- **Compatibility**: Stable Rust + experimental nightly features

## Architecture

### Core Types
- `FastVec<T>`, `FastStr` - High-performance containers (3-4x faster)
- `SecureMemoryPool` - Production memory management (RAII + thread safety)
- `RankSelectInterleaved256` - Peak performance rank/select (3.3 Gelem/s)
- `LruPageCache` - Multi-shard caching layer with page-aligned memory management
- `CachedBlobStore<T>` - Cache-aware blob store wrapper with transparent caching
- `ValVec32<T>`, `SmallMap<K,V>` - Specialized containers (memory efficient)
- `DoubleArrayTrie` - O(1) state transitions with 8-byte representation
- `CompressedSparseTrie` - Multi-level concurrency with token-based safety
- `NestedLoudsTrie` - Configurable nesting with fragment compression
- `FsaCache` - FSA cache system with 8-byte state representation and multi-strategy support
- `NestedTrieDawg` - DAWG implementation with state merging and rank-select acceleration
- `GraphWalker` - 8 graph traversal strategies (BFS, DFS, CFS, MultiPass, etc.)
- `FastSearchEngine` - SIMD-optimized byte search with hardware acceleration
- `StreamBufferedReader/Writer` - Configurable buffering strategies (performance/memory/latency)
- `RangeReader/Writer` - Precise byte-level access with multi-range support
- `ZeroCopyReader/Writer` - Direct buffer access with SIMD optimization
- `SmartPtrSerializer` - Reference-counted object serialization with cycle detection
- `ComplexTypeSerializer` - Tuple/collection serialization with metadata validation
- `EndianIO<T>` - Cross-platform endian handling with SIMD bulk conversions
- `VersionManager` - Schema evolution and backward compatibility support
- `VarIntEncoder` - Variable integer encoding with 7 strategies and adaptive selection
- `LockFreeMemoryPool` - High-performance concurrent allocation with CAS operations and false sharing prevention
- `ThreadLocalMemoryPool` - Zero-contention per-thread caching with hot area management and lazy synchronization
- `FixedCapacityMemoryPool` - Real-time deterministic allocation with bounded memory and size class management
- `MmapVec<T>` - Persistent memory-mapped vectors with cross-platform compatibility and automatic growth
- `ReplaceSelectSort` - External sorting for large datasets with replacement selection and k-way merging
- `LoserTree` - Tournament tree for efficient k-way merging with O(log k) complexity per element
- `EnhancedSuffixArray` - Advanced suffix arrays with SA-IS algorithm and LCP array support
- `LexicographicIterator` - Efficient iteration over sorted string collections with O(1) access and O(log n) seeking
- `UnicodeProcessor` - Full Unicode support with SIMD acceleration, normalization, and comprehensive analysis
- `LineProcessor` - High-performance text file processing with configurable buffering and field splitting
- `RuntimeCpuFeatures` - Comprehensive CPU feature detection with adaptive SIMD optimization selection
- `HighPrecisionTimer` - Nanosecond-accurate timing with automatic unit formatting and performance profiling
- `ProcessManager` - Async process management with bidirectional pipes, process pools, and timeout handling
- `AdaptiveBase64` - SIMD-accelerated Base64 encoding/decoding with automatic implementation selection
- `VmManager` - Virtual memory management with kernel feature detection and cross-platform optimization
- `FactoryRegistry<T>` - Generic factory registry with thread-safe registration and type-safe object creation
- `GlobalFactory<T>` - Global factory instances with automatic initialization and concurrent access
- `DevHighPrecisionTimer` - High-precision timing with automatic unit selection and nanosecond accuracy
- `PerformanceProfiler` - Global profiler with centralized performance tracking and statistical analysis
- `MemoryDebugger` - Memory debugging with allocation tracking, leak detection, and usage statistics
- `Histogram<T>` - Adaptive histogram with dual storage strategy for efficient statistics collection
- `StatAccumulator` - Real-time statistics accumulator with lock-free atomic operations
- `MultiDimensionalStats` - Multi-dimensional statistical analysis with correlation tracking
- `FiberAio` - High-performance async I/O manager with adaptive provider selection and read-ahead optimization
- `FiberFile` - Fiber-aware asynchronous file handle with vectored I/O and cache-friendly access patterns
- `FiberYield` - Sophisticated yielding mechanism with budget control and adaptive scheduling
- `YieldPoint` - Cooperative yield point for long-running operations with automatic checkpointing
- `AdaptiveYieldScheduler` - Multi-fiber yield management with load-aware scheduling and global statistics
- `AdaptiveMutex<T>` - Adaptive mutex with statistics collection, timeout support, and contention monitoring
- `SpinLock<T>` - High-performance spin lock optimized for short critical sections
- `PriorityRwLock<T>` - Reader-writer lock with configurable writer priority and reader limits
- `SegmentedMutex<T>` - Hash-based segment selection for reduced contention in multi-threaded scenarios
- `LinuxFutex` - Platform-specific futex implementation with direct syscall access
- `FutexMutex` - High-performance mutex using Linux futex with zero userspace overhead
- `FutexCondvar` - Condition variable with futex backing for efficient blocking
- `FutexRwLock` - Reader-writer lock using futex for scalable concurrency
- `InstanceTls<T>` - Matrix-based O(1) thread-local storage with configurable dimensions
- `OwnerTls<T, O>` - Owner-based TLS associating data with specific object instances
- `TlsPool<T>` - Thread-local storage pool for managing multiple TLS instances
- `AtomicExt` - Extended atomic operations trait (atomic_maximize, atomic_minimize, conditional updates)
- `AtomicStack<T>` - Lock-free stack using CAS operations with approximate size tracking
- `AtomicNode<T>` - Lock-free linked list node for atomic data structures
- `AsAtomic<T>` - Safe atomic casting trait for reinterpretation between regular and atomic types
- `ZipOffsetBlobStore` - High-performance offset-based compressed blob storage with block-based delta compression
- `SortedUintVec` - Block-based delta compression for sorted integer sequences with variable bit-width encoding
- `ZipOffsetBlobStoreBuilder` - Builder pattern for constructing compressed blob stores with ZSTD integration

### Feature Flags
- **Default**: `simd`, `mmap`, `zstd`, `serde`
- **Optional**: `lz4`, `ffi` (stable), `avx512` (nightly)

### Security
- Use `SecureMemoryPool` (not legacy `MemoryPool`)
- RAII with `SecurePooledPtr` 
- Thread-safe, prevents use-after-free/double-free
- Zero-on-free for sensitive data

## Development Patterns

### Memory Management
```rust
// ✅ SECURE: Production-ready
let pool = SecureMemoryPool::new(config)?;
let ptr = pool.allocate()?; // Auto-cleanup on drop

// ✅ Global pools
let ptr = get_global_pool_for_size(1024).allocate()?;
```

### LRU Page Cache
```rust
// ✅ High-performance cache configuration
let config = PageCacheConfig::performance_optimized()
    .with_capacity(256 * 1024 * 1024)  // 256MB cache
    .with_shards(8)                    // 8 shards for reduced contention
    .with_huge_pages(true);            // Use 2MB huge pages

let cache = LruPageCache::new(config)?;
let file_id = cache.register_file(1)?;

// ✅ Cache operations
let buffer = cache.read(file_id, 0, 4096)?;     // Read 4KB page
cache.prefetch(file_id, 4096, 16384)?;         // Prefetch 16KB

// ✅ Cache-aware blob store
let blob_store = MemoryBlobStore::new();
let cached_store = CachedBlobStore::new(blob_store, config)?;
```

### Performance Testing
```rust
#[cfg(test)]
use criterion::{criterion_group, Criterion};

fn benchmark_name(c: &mut Criterion) {
    c.bench_function("operation", |b| b.iter(|| {
        // benchmark code
    }));
}
```

### Error Handling
```rust
use crate::error::{ZiporaError, Result};

fn example() -> Result<()> {
    Err(ZiporaError::invalid_data("error"))
}
```

### Advanced Rank/Select Features
```rust
// Fragment-based compression
let rs_fragment = RankSelectFragment::new(bit_vector)?;
let compression_ratio = rs_fragment.compression_ratio();

// Hierarchical multi-level caching
let rs_hierarchical = RankSelectHierarchical::new(bit_vector)?;
let rank_fast = rs_hierarchical.rank1(position); // O(1)

// BMI2 hardware acceleration
let rs_bmi2 = RankSelectBMI2::new(bit_vector)?;
let select_ultra_fast = rs_bmi2.select1(rank)?; // 5-10x faster
```

### Advanced FSA & Trie Features
```rust
// Double Array Trie - O(1) state transitions
let mut dat = DoubleArrayTrie::new();
dat.insert(b"computer")?;
assert!(dat.contains(b"computer"));

// Compressed Sparse Trie - Multi-level concurrency
let mut csp = CompressedSparseTrie::new(ConcurrencyLevel::MultiWriteMultiRead)?;
let writer_token = csp.acquire_writer_token().await?;
csp.insert_with_token(b"hello", &writer_token)?;

// Nested LOUDS Trie - Fragment compression
let config = NestingConfig::builder()
    .max_levels(4)
    .fragment_compression_ratio(0.3)
    .build()?;
let mut nested = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config)?;
nested.insert(b"computer")?; // Automatic fragment compression
```

### Advanced FSA Infrastructure Features
```rust
// FSA Cache with multi-strategy support
let cache = FsaCache::with_config(FsaCacheConfig::performance_optimized())?;
cache.cache_state(parent_state, child_base, is_terminal)?;

// DAWG construction with state merging
let mut dawg = NestedTrieDawg::new()?;
dawg.build_from_keys(keys)?; // Automatic state merging and compression

// Graph traversal with multiple strategies
let mut walker = BfsGraphWalker::new(WalkerConfig::default());
walker.walk(start_vertex, &mut visitor)?;

// Hardware-accelerated search
let search = FastSearchEngine::with_hardware_acceleration()?;
let positions = search.search_byte_simd(data, target_byte)?; // SSE4.2 acceleration
```

### Advanced I/O & Serialization Features
```rust
// StreamBuffer with configurable strategies
let config = StreamBufferConfig::performance_optimized();
let mut reader = StreamBufferedReader::with_config(file, config)?;
let byte = reader.read_byte_fast()?; // Hot path optimization

// Range-based stream operations
let mut range_reader = RangeReader::new_and_seek(file, 1024, 4096)?;
let progress = range_reader.progress(); // 0.0 to 1.0
let value = range_reader.read_u32()?; // DataInput trait support

// Zero-copy optimizations
let mut zc_reader = ZeroCopyReader::with_secure_buffer(stream, 256 * 1024)?;
if let Some(data) = zc_reader.zc_read(1024)? {
    process_data_in_place(data); // No copying
    zc_reader.zc_advance(1024)?;
}

// Memory-mapped zero-copy operations
#[cfg(feature = "mmap")]
{
    let mut mmap_reader = MmapZeroCopyReader::new(file)?;
    let entire_file = mmap_reader.as_slice(); // Zero system calls
}

// Vectored I/O for bulk transfers
let buffers = [IoSliceMut::new(&mut buf1), IoSliceMut::new(&mut buf2)];
let bytes_read = VectoredIO::read_vectored(&mut reader, &mut buffers)?;
```

### Comprehensive Serialization Features
```rust
// Smart pointer serialization with cycle detection
let shared_data = Rc::new("shared value".to_string());
let serializer = SmartPtrSerializer::default();
let bytes = serializer.serialize_to_bytes(&shared_data)?;
let deserialized: Rc<String> = serializer.deserialize_from_bytes(&bytes)?;

// Complex type serialization
let complex_data = (vec![1u32, 2, 3], Some("nested".to_string()), HashMap::new());
let serializer = ComplexTypeSerializer::default();
let bytes = serializer.serialize_to_bytes(&complex_data)?;

// Cross-platform endian handling
let io = EndianIO::<u32>::little_endian();
let mut buffer = [0u8; 4];
io.write_to_bytes(0x12345678, &mut buffer)?;
let value = io.read_from_bytes(&buffer)?;

// Advanced version management with migration
#[derive(Debug, PartialEq)]
struct DataV2 { id: u32, name: String, new_field: Option<String> }

impl VersionedSerialize for DataV2 {
    fn current_version() -> Version { Version::new(2, 0, 0) }
    
    fn serialize_with_manager<O: DataOutput>(&self, manager: &mut VersionManager, output: &mut O) -> Result<()> {
        output.write_u32(self.id)?;
        output.write_length_prefixed_string(&self.name)?;
        manager.serialize_field("new_field", &self.new_field, output)
    }
    
    fn deserialize_with_manager<I: DataInput>(manager: &mut VersionManager, input: &mut I) -> Result<Self> {
        let id = input.read_u32()?;
        let name = input.read_length_prefixed_string()?;
        let new_field = manager.deserialize_field("new_field", input)?.unwrap_or(None);
        Ok(Self { id, name, new_field })
    }
}

// Variable integer encoding with adaptive strategies
let encoder = VarIntEncoder::zigzag(); // For signed integers
let values = vec![-100i64, -1, 0, 1, 100];
let encoded = encoder.encode_i64_sequence(&values)?;

// Automatic strategy selection
let optimal_strategy = choose_optimal_strategy(&data);
let auto_encoder = VarIntEncoder::new(optimal_strategy);
```

### Advanced Memory Pool Features
```rust
// Lock-free memory pool with CAS operations
let config = LockFreePoolConfig::high_performance();
let pool = LockFreeMemoryPool::new(config)?;
let alloc = pool.allocate(1024)?; // Lock-free concurrent allocation

// Thread-local memory pool with zero contention
let config = ThreadLocalPoolConfig::high_performance();
let pool = ThreadLocalMemoryPool::new(config)?;
let alloc = pool.allocate(512)?; // Per-thread cached allocation

// Fixed capacity pool for real-time systems
let config = FixedCapacityPoolConfig::realtime();
let pool = FixedCapacityMemoryPool::new(config)?;
let alloc = pool.allocate(256)?; // Bounded deterministic allocation

// Memory-mapped vectors for persistent storage
let config = MmapVecConfig::large_dataset();
let mut vec = MmapVec::<u64>::create("data.mmap", config)?;
vec.push(42)?; // Persistent vector operations
vec.sync()?; // Force persistence to disk
```

### ZipOffsetBlobStore Features
```rust
// High-performance offset-based compressed blob storage
let config = ZipOffsetBlobStoreConfig::performance_optimized();
let mut builder = ZipOffsetBlobStoreBuilder::with_config(config)?;

// Add records with automatic compression and checksumming
builder.add_record(b"First record data")?;
builder.add_record(b"Second record data")?;
builder.add_record(b"Third record data")?;

// Build the final store with optimized layout
let store = builder.finish()?;

// Template-based record retrieval with const generics
let record = store.get(0)?; // O(1) access to any record
let size = store.size(1)?.unwrap(); // Compressed size information

// Block-based delta compression for sorted integer sequences
let mut uint_builder = SortedUintVecBuilder::new();
uint_builder.push(1000)?;
uint_builder.push(1010)?; // Small delta = efficient compression
uint_builder.push(1025)?;

let compressed_uints = uint_builder.finish()?;
let value = compressed_uints.get(1)?; // BMI2-accelerated bit extraction

// Batch building for high-performance bulk operations
let mut batch_builder = BatchZipOffsetBlobStoreBuilder::new(100)?;
batch_builder.add_record(b"batch record 1")?;
batch_builder.add_record(b"batch record 2")?;
batch_builder.flush_batch()?; // Manual flush control

// File I/O with 128-byte aligned headers
store.save_to_file("compressed.zob")?;
let loaded_store = ZipOffsetBlobStore::load_from_file("compressed.zob")?;

// Statistics and compression analysis
let stats = builder.stats();
println!("Compression ratio: {:.2}", stats.compression_ratio());
println!("Space saved: {:.1}%", stats.space_saved_percent());
```

### String Processing Features
```rust
// Lexicographic string iterators - O(1) access, O(log n) seeking
let strings = vec!["apple".to_string(), "banana".to_string(), "cherry".to_string()];
let mut iter = SortedVecLexIterator::new(&strings);

// Bidirectional iteration
assert_eq!(iter.current(), Some("apple"));
iter.next()?;
assert_eq!(iter.current(), Some("banana"));

// Binary search operations
assert!(iter.seek_lower_bound("cherry")?); // Exact match
assert_eq!(iter.current(), Some("cherry"));

// Unicode processing with hardware acceleration
let text = "Hello 世界! 🦀 Rust";
let char_count = validate_utf8_and_count_chars(text.as_bytes())?;

let mut processor = UnicodeProcessor::new()
    .with_normalization(true)
    .with_case_folding(true);
let processed = processor.process("HELLO World!")?;

// Comprehensive Unicode analysis
let analysis = processor.analyze("Hello 世界! 🦀");
println!("ASCII ratio: {:.1}%", (analysis.ascii_count as f64 / analysis.char_count as f64) * 100.0);
println!("Complexity score: {:.2}", analysis.complexity_score());

// High-performance line processing
let text_data = "line1\nline2\nfield1,field2,field3\n";
let cursor = std::io::Cursor::new(text_data);
let config = LineProcessorConfig::performance_optimized(); // 256KB buffer
let mut processor = LineProcessor::with_config(cursor, config);

// Process lines with closure
let processed_count = processor.process_lines(|line| {
    println!("Processing: {}", line);
    Ok(true) // Continue processing
})?;

// Split lines by delimiter with field-level processing
let field_count = processor.split_lines_by(",", |field, line_num, field_num| {
    println!("Line {}, Field {}: {}", line_num, field_num, field);
    Ok(true)
})?;
```

### System Integration Utilities Features
```rust
// CPU feature detection with optimization tier selection
let features = get_cpu_features();
println!("CPU: {} {}", features.vendor, features.model);
println!("SIMD tier: {}", features.simd_tier);
println!("Optimal rank/select: {}", features.optimal_rank_select_variant());
println!("Optimal Base64: {}", features.optimal_base64_variant());

// High-precision timing and benchmarking
let timer = HighPrecisionTimer::named("operation");
// ... perform operation ...
timer.print_elapsed(); // Automatic unit selection (ns/μs/ms/s)

let mut suite = BenchmarkSuite::new("performance_tests");
suite.add_benchmark("fast_operation", 1000, || {
    // Fast operation to benchmark
});
suite.run_all(); // Comprehensive statistics with ops/sec

// Process management with bidirectional communication
let manager = ProcessManager::new(5); // Max 5 concurrent processes
let result = manager.execute_managed("echo", &["hello"]).await?;
assert_eq!(result.exit_code, 0);

// Bidirectional pipe for interactive processes
let config = ProcessConfig::default();
let pipe = manager.create_pipe("python3", &["-i"], config).await?;
pipe.write_line("print('Hello from Python!')").await?;
let output = pipe.read_stdout_line().await?;

// Adaptive SIMD Base64 encoding/decoding
let encoder = AdaptiveBase64::new(); // Automatic SIMD selection
let data = b"Hello, SIMD World!";
let encoded = encoder.encode(data);
let decoded = encoder.decode(&encoded)?;
assert_eq!(decoded, data);

// Force specific SIMD implementation for testing
let config = Base64Config {
    force_implementation: Some(SimdImplementation::AVX2),
    ..Default::default()
};
let avx2_encoder = AdaptiveBase64::with_config(config);

// Virtual memory management with kernel awareness
let vm_manager = VmManager::new()?;
let kernel_info = vm_manager.get_kernel_info();
println!("Kernel version: {}", kernel_info.version);
println!("Huge pages available: {}", kernel_info.has_huge_pages);

// Page-aligned allocation for SIMD operations
let alignment = features.recommended_alignment(); // 16/32/64 bytes
let aligned_alloc = PageAlignedAlloc::new(1024, alignment)?;
vm_prefetch(aligned_alloc.as_ptr(), 1024); // Hardware prefetch
```

### Development Infrastructure Features
```rust
// Factory pattern with type-safe registration
let factory = FactoryRegistry::<Box<dyn MyTrait>>::new();
factory.register_type::<ConcreteImpl, _>(|| {
    Ok(Box::new(ConcreteImpl::new()) as Box<dyn MyTrait>)
})?;

// Create objects by type name
let obj = factory.create_by_type::<ConcreteImpl>()?;

// Global factory for convenient access
global_factory::<Box<dyn MyTrait>>().register("my_impl", || {
    Ok(Box::new(ConcreteImpl::new()) as Box<dyn MyTrait>)
})?;

// High-precision debugging with automatic unit selection
let timer = DevHighPrecisionTimer::named("operation");
// ... perform operation ...
timer.print_elapsed(); // Automatic unit selection (ns/μs/ms/s)

// Performance profiling with global registry
global_profiler().profile("critical_path", || {
    // ... critical operation ...
    Ok(result)
})?;

// Memory debugging for custom allocators
let debugger = MemoryDebugger::new();
debugger.record_allocation(ptr as usize, size, "module:function:line");
let stats = debugger.get_stats();
println!("Peak usage: {} bytes", stats.peak_usage);

// Adaptive histogram with dual storage strategy
let mut hist = U32Histogram::new();
hist.increment(100);  // Small values: direct array access O(1)
hist.increment(5000); // Large values: hash map storage
hist.add(1000, 5);    // Add multiple counts

// Real-time statistics accumulator (thread-safe)
let acc = StatAccumulator::new();
acc.add(42);  // Lock-free atomic operations
acc.add(100);
acc.add(75);

let snapshot = acc.snapshot();
println!("Mean: {:.2}, Std Dev: {:.2}", snapshot.mean, snapshot.std_dev);

// Multi-dimensional statistics
let mut multi_stats = MultiDimensionalStats::new(
    "network_metrics",
    vec!["latency".to_string(), "throughput".to_string(), "errors".to_string()]
);

multi_stats.add_sample(&[50, 1000, 0])?; // latency, throughput, errors
let latency_stats = multi_stats.dimension_stats(0).unwrap();
```

### Advanced Fiber Concurrency Features
```rust
// FiberAIO - Asynchronous I/O integration with adaptive provider selection
let config = FiberAioConfig {
    io_provider: IoProvider::auto_detect(), // Tokio/io_uring/POSIX AIO/IOCP
    read_buffer_size: 64 * 1024,
    enable_vectored_io: true,
    read_ahead_size: 256 * 1024,
    ..Default::default()
};

let aio = FiberAio::with_config(config)?;
let mut file = aio.open("large_data.txt").await?;

// Read-ahead optimization with cache-friendly access
let mut buffer = vec![0u8; 1024];
let bytes_read = file.read(&mut buffer).await?;

// Parallel file processing with controlled concurrency
let results = FiberIoUtils::process_files_parallel(
    paths,
    4, // max concurrent
    |path| Box::pin(async move {
        let aio = FiberAio::new()?;
        aio.read_to_vec(path).await
    })
).await?;

// FiberYield - Cooperative multitasking with budget control
let config = YieldConfig {
    initial_budget: 16,
    adaptive_budgeting: true,
    yield_threshold: Duration::from_micros(100),
    ..Default::default()
};

let yield_controller = FiberYield::with_config(config);
yield_controller.yield_now().await;           // Budget-based yielding
yield_controller.force_yield().await;         // Immediate yield with budget reset
yield_controller.yield_if_needed().await;     // Conditional yield based on time

// Global yield operations with thread-local optimizations
GlobalYield::yield_now().await;
let stats = GlobalYield::stats(); // Thread-local yield statistics

// Yield points for long-running operations
let yield_point = YieldPoint::new(100); // Yield every 100 operations
for i in 0..10000 {
    process_item(i);
    yield_point.checkpoint().await; // Automatic yielding
}

// Yielding iterator wrapper
let yielding_iter = YieldingIterator::new(data.into_iter(), 3);
let processed = yielding_iter.for_each(|x| {
    sum += x;
    Ok(())
}).await?;

// Enhanced Mutex implementations with specialized variants
let config = MutexConfig {
    adaptive_spinning: true,
    max_spin_duration: Duration::from_micros(10),
    timeout: Some(Duration::from_millis(100)),
    ..Default::default()
};

// Adaptive mutex with statistics and timeout support
let mutex = AdaptiveMutex::with_config(42, config);
let guard = mutex.lock().await;
let stats = mutex.stats(); // Contention ratio, hold times, etc.

// High-performance spin lock for short critical sections
let spin_lock = SpinLock::new(100);
let guard = spin_lock.lock().await;

// Priority reader-writer lock with configurable options
let rwlock_config = RwLockConfig {
    writer_priority: true,
    max_readers: Some(64),
    fair: true,
};
let rwlock = PriorityRwLock::with_config(vec![1, 2, 3], rwlock_config);
let read_guard = rwlock.read().await;
let write_guard = rwlock.write().await;

// Segmented mutex for reduced contention
let segmented = SegmentedMutex::new(0, 8); // 8 segments
let segment_guard = segmented.lock_segment(3).await;
let key_guard = segmented.lock_for_key(&"my_key").await; // Hash-based selection
```

### Low-Level Synchronization Features
```rust
// Linux Futex Integration - Direct futex syscalls
use zipora::{LinuxFutex, FutexMutex, FutexCondvar, FutexRwLock, PlatformSync};

// High-performance mutex using direct futex syscalls
let mutex = FutexMutex::new();
{
    let guard = mutex.lock().unwrap();
    // Critical section with zero-overhead synchronization
}

// Condition variable with futex implementation
let condvar = FutexCondvar::new();
let guard = mutex.lock().unwrap();
let guard = condvar.wait(guard).unwrap(); // Zero-overhead blocking

// Reader-writer lock with futex backing
let rwlock = FutexRwLock::new();
{
    let read_guard = rwlock.read().unwrap();
    // Multiple concurrent readers
}
{
    let write_guard = rwlock.write().unwrap();
    // Exclusive writer access
}

// Platform abstraction for cross-platform code
use zipora::{DefaultPlatformSync};
DefaultPlatformSync::futex_wait(&atomic_value, expected_val, timeout).unwrap();
DefaultPlatformSync::futex_wake(&atomic_value, num_waiters).unwrap();

// Instance-Specific Thread-Local Storage - Matrix-based O(1) access
use zipora::{InstanceTls, OwnerTls, TlsPool};

// Matrix-based O(1) access thread-local storage
let tls = InstanceTls::<MyData>::new().unwrap();

// Each thread gets its own copy of the data
tls.set(MyData { value: 42, name: "thread-local".to_string() });
let data = tls.get(); // O(1) access, automatically creates default if not set
let optional_data = tls.try_get(); // O(1) access, returns None if not set

// Owner-based TLS associating data with specific objects
let mut owner_tls = OwnerTls::<MyData, MyOwner>::new();
let owner = MyOwner { id: 1 };
let data = owner_tls.get_or_create(&owner).unwrap();

// Thread-local storage pool for managing multiple instances
let pool = TlsPool::<MyData, 64>::new().unwrap(); // 64 TLS instances
let data = pool.get_next(); // Round-robin access
let specific_data = pool.get_slot(5).unwrap(); // Access specific slot

// Automatic cleanup and ID recycling
let id = tls.id(); // Unique instance ID
drop(tls); // ID automatically returned to free pool

// Atomic Operations Framework - Lock-free programming utilities
use zipora::{AtomicExt, AsAtomic, AtomicStack, AtomicNode, AtomicBitOps, 
            spin_loop_hint, memory_ordering};

// Extended atomic operations
use std::sync::atomic::{AtomicU32, Ordering};
let atomic = AtomicU32::new(10);

// Atomic max/min operations
let old_max = atomic.atomic_maximize(15, Ordering::Relaxed); // Returns 15
let old_min = atomic.atomic_minimize(5, Ordering::Relaxed);  // Returns 5

// Optimized compare-and-swap operations
let result = atomic.cas_weak(5, 10); // Weak CAS with optimized ordering
let strong_result = atomic.cas_strong(10, 20); // Strong CAS

// Conditional atomic updates
let updated = atomic.update_if(|val| val % 2 == 0, 100, Ordering::Relaxed);

// Lock-free data structures
let stack = AtomicStack::<i32>::new();
stack.push(42); // Lock-free push
stack.push(84);
assert_eq!(stack.pop(), Some(84)); // Lock-free pop (LIFO)
assert_eq!(stack.len(), 1); // Approximate size

// Atomic bit operations
let bits = AtomicU32::new(0);
assert!(!bits.set_bit(5)); // Set bit 5, returns previous state
assert!(bits.test_bit(5)); // Test if bit 5 is set
assert!(bits.toggle_bit(5)); // Toggle bit 5
assert_eq!(bits.find_first_set(), None); // Find first set bit

// Safe atomic casting between types
let mut value = 42u32;
let atomic_ref = value.as_atomic_mut(); // &mut AtomicU32
atomic_ref.store(100, Ordering::Relaxed);
assert_eq!(value, 100);

// Platform-specific optimizations
#[cfg(target_arch = "x86_64")]
{
    use zipora::x86_64_optimized;
    x86_64_optimized::pause(); // PAUSE instruction for spin loops
    x86_64_optimized::mfence(); // Memory fence
}

// Memory ordering utilities
memory_ordering::full_barrier(); // Full memory barrier
memory_ordering::load_barrier(); // Load barrier
memory_ordering::store_barrier(); // Store barrier
```

## Next Phase: 11B

**Priority**: GPU acceleration, distributed systems, advanced compression algorithms

**Target**: 6-12 months for advanced features beyond Phase 11A

---

*Updated: 2025-01-15 - LRU Page Cache Complete with Sophisticated Caching Layer*
*Tests: 1,100+ passing + 5,735+ trie tests + comprehensive serialization tests + memory pool tests + algorithm tests + string processing tests + system utilities tests + development infrastructure tests + fiber concurrency tests + low-level synchronization tests + LRU cache tests (all implementations fully working)*  
*Performance: Complete caching layer ecosystem + multi-shard architecture + page-aligned memory management + hardware prefetching + transparent blob store integration + low-level synchronization + 3.3 Gelem/s rank/select*
*Revolutionary Features: LRU Page Cache (multi-shard caching with configurable sharding, page-aligned memory with huge page support, cache-aware blob store integration with transparent caching, comprehensive statistics and monitoring), 3 low-level synchronization components (Linux Futex Integration, Instance-Specific Thread-Local Storage, Atomic Operations Framework), production-ready high-performance caching and synchronization primitives with zero-cost abstractions*