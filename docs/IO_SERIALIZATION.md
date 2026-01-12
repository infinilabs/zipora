# I/O & Serialization

Zipora provides 8 comprehensive serialization components with cutting-edge optimizations and cross-platform compatibility.

## Advanced Serialization System

```rust
use zipora::io::{
    // Smart Pointer Serialization
    SmartPtrSerializer, SerializationContext, Box, Rc, Arc, Weak,

    // Complex Type Serialization
    ComplexTypeSerializer, ComplexSerialize, VersionProxy,

    // Endian Handling
    EndianIO, Endianness, EndianConvert, EndianConfig,

    // Version Management
    VersionManager, VersionedSerialize, Version, MigrationRegistry,

    // Variable Integer Encoding
    VarIntEncoder, VarIntStrategy, choose_optimal_strategy,
};

// Smart Pointer Serialization - Reference-counted objects
let shared_data = Rc::new("shared value".to_string());
let clone1 = shared_data.clone();
let clone2 = shared_data.clone();

let serializer = SmartPtrSerializer::default();
let bytes = serializer.serialize_to_bytes(&clone1).unwrap();
let deserialized: Rc<String> = serializer.deserialize_from_bytes(&bytes).unwrap();

// Cycle detection and shared object optimization
let mut context = SerializationContext::new();
clone1.serialize_with_context(&mut output, &mut context).unwrap();
clone2.serialize_with_context(&mut output, &mut context).unwrap(); // References first object

// Complex Type Serialization - Tuples, collections, nested types
let complex_data = (
    vec![1u32, 2, 3],
    Some("nested".to_string()),
    HashMap::from([("key".to_string(), 42u32)]),
);

let serializer = ComplexTypeSerializer::default();
let bytes = serializer.serialize_to_bytes(&complex_data).unwrap();
let deserialized = serializer.deserialize_from_bytes(&bytes).unwrap();

// Batch operations for efficiency
let tuples = vec![(1u32, "first"), (2u32, "second"), (3u32, "third")];
let batch_bytes = serializer.serialize_batch(&tuples).unwrap();
let batch_result = serializer.deserialize_batch(&batch_bytes).unwrap();
```

## Endian Handling

```rust
// Comprehensive Endian Handling - Cross-platform compatibility
let io = EndianIO::<u32>::little_endian();
let value = 0x12345678u32;

// Safe endian conversion with bounds checking
let mut buffer = [0u8; 4];
io.write_to_bytes(value, &mut buffer).unwrap();
let read_value = io.read_from_bytes(&buffer).unwrap();

// SIMD-accelerated bulk conversions
#[cfg(target_arch = "x86_64")]
{
    use zipora::io::endian::simd::convert_u32_slice_simd;
    let mut values = vec![0x1234u32, 0x5678u32, 0x9abcu32];
    convert_u32_slice_simd(&mut values, false);
}

// Cross-platform configuration
let config = EndianConfig::cross_platform(); // Little endian + auto-detection
let optimized = EndianConfig::performance_optimized(); // Native + SIMD acceleration
```

## Variable Integer Encoding

```rust
// Variable Integer Encoding - Multiple strategies
let encoder = VarIntEncoder::zigzag(); // For signed integers
let signed_values = vec![-100i64, -1, 0, 1, 100];
let encoded = encoder.encode_i64_sequence(&signed_values).unwrap();
let decoded = encoder.decode_i64_sequence(&encoded).unwrap();

// Delta encoding for sorted sequences
let delta_encoder = VarIntEncoder::delta();
let sorted_values = vec![10u64, 12, 15, 20, 22, 25];
let delta_encoded = delta_encoder.encode_u64_sequence(&sorted_values).unwrap();

// Group varint for bulk operations
let group_encoder = VarIntEncoder::group_varint();
let bulk_values = vec![1u64, 256, 65536, 16777216];
let group_encoded = group_encoder.encode_u64_sequence(&bulk_values).unwrap();

// Automatic strategy selection based on data characteristics
let optimal_strategy = choose_optimal_strategy(&values);
let auto_encoder = VarIntEncoder::new(optimal_strategy);
```

## Stream Processing

```rust
use zipora::io::{
    StreamBufferedReader, StreamBufferedWriter, StreamBufferConfig,
    RangeReader, RangeWriter, MultiRangeReader,
    ZeroCopyReader, ZeroCopyWriter, ZeroCopyBuffer, VectoredIO
};

// Advanced Stream Buffering - Configurable strategies
let config = StreamBufferConfig::performance_optimized();
let mut reader = StreamBufferedReader::with_config(cursor, config).unwrap();

// Fast byte reading with hot path optimization
let byte = reader.read_byte_fast().unwrap();

// Bulk read optimization for large data transfers
let mut large_buffer = vec![0u8; 1024 * 1024];
let bytes_read = reader.read_bulk(&mut large_buffer).unwrap();

// Prefetch optimization for sequential access
reader.prefetch_ahead(64 * 1024).unwrap();

// Stream buffered writer with configurable flush strategies
let mut writer = StreamBufferedWriter::with_config(file, config).unwrap();
writer.write_all(b"High-performance writing").unwrap();
writer.flush_with_strategy(FlushStrategy::Async).unwrap();
```

## Range-Based I/O

```rust
// Range-based reading for random access patterns
let mut range_reader = RangeReader::new(file).unwrap();
let data = range_reader.read_range(1000, 2000).unwrap();

// Multi-range reading for scattered access
let ranges = vec![(0, 100), (500, 600), (1000, 1100)];
let mut multi_reader = MultiRangeReader::new(file, ranges).unwrap();
let chunks = multi_reader.read_all().unwrap();

// Range-based writing
let mut range_writer = RangeWriter::new(file).unwrap();
range_writer.write_at(1000, b"positioned data").unwrap();
```

## Zero-Copy I/O

```rust
// Zero-copy buffer management
let buffer = ZeroCopyBuffer::new(4096).unwrap();
let slice = buffer.as_slice();

// Zero-copy reader with memory mapping
let mut reader = ZeroCopyReader::open(file_path).unwrap();
let mapped_slice = reader.map_range(0, 1024).unwrap();

// Zero-copy writer
let mut writer = ZeroCopyWriter::create(file_path).unwrap();
writer.write_zero_copy(&data).unwrap();

// Vectored I/O for scatter-gather operations
let buffers = vec![
    IoSlice::new(b"header"),
    IoSlice::new(b"body"),
    IoSlice::new(b"footer"),
];
let vectored = VectoredIO::new(file).unwrap();
vectored.write_vectored(&buffers).unwrap();
```

## Version Management

```rust
use zipora::io::{VersionManager, VersionedSerialize, Version, MigrationRegistry};

// Version management for backward compatibility
let version_manager = VersionManager::new();
let current_version = Version::new(2, 1, 1);

// Register migration functions
let mut registry = MigrationRegistry::new();
registry.register_migration(
    Version::new(1, 0, 0),
    Version::new(2, 0, 0),
    |old_data| migrate_v1_to_v2(old_data)
);

// Serialize with version information
let data = MyData { field: 42 };
let versioned_bytes = version_manager.serialize_versioned(&data, current_version).unwrap();

// Deserialize with automatic migration
let deserialized: MyData = version_manager.deserialize_versioned(&versioned_bytes).unwrap();
```

## Performance Characteristics

| Component | Throughput | Latency | Use Case |
|-----------|------------|---------|----------|
| **SmartPtrSerializer** | Moderate | Low | Reference-counted objects |
| **EndianIO** | High (SIMD) | Minimal | Cross-platform binary I/O |
| **VarIntEncoder** | Very High | Minimal | Compact integer encoding |
| **StreamBufferedReader** | Very High | Low | Sequential file access |
| **ZeroCopyReader** | Maximum | Minimal | Memory-mapped access |
| **VectoredIO** | High | Low | Scatter-gather operations |
