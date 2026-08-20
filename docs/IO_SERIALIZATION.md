# I/O & Serialization

Zipora provides comprehensive serialization components with cutting-edge optimizations and cross-platform compatibility.

## Advanced Serialization System

```rust
use std::collections::HashMap;
use std::rc::Rc;

use zipora::io::{
    // Smart Pointer Serialization
    SmartPtrSerializer, SerializationContext, SmartPtrConfig, SmartPtrSerialize,

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
use zipora::io::VecDataOutput;
let mut output = VecDataOutput::new();
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

// Stream buffered writer (implements std::io::Write)
let mut writer = StreamBufferedWriter::with_config(file, config).unwrap();
writer.write_all(b"High-performance writing").unwrap();
writer.flush().unwrap();
```

## Range-Based I/O

```rust
use std::io::{Read, Write};

// Range-based reading for random access patterns:
// construct with (inner, start, length), then use std::io::Read.
// `new_and_seek` also seeks the underlying reader to `start`.
let mut range_reader = RangeReader::new_and_seek(file, 1000, 2000).unwrap();
let mut data = Vec::new();
range_reader.read_to_end(&mut data).unwrap(); // reads bytes 1000..3000

// Multi-range reading for scattered access: (start, end) byte ranges
let ranges = vec![(0, 100), (500, 600), (1000, 1100)];
let mut multi_reader = MultiRangeReader::new(file, ranges);
let mut chunks = Vec::new();
multi_reader.read_to_end(&mut chunks).unwrap(); // concatenates all ranges

// Range-based writing: construct with (inner, start, length), then use std::io::Write
let mut range_writer = RangeWriter::new_and_seek(file, 1000, 15).unwrap();
range_writer.write_all(b"positioned data").unwrap();
range_writer.flush().unwrap();
```

## Zero-Copy I/O

```rust
use std::io::IoSlice;
use zipora::io::{ZeroCopyBuffer, ZeroCopyReader, ZeroCopyWriter,
                 ZeroCopyRead, ZeroCopyWrite, VectoredIO};

// Zero-copy buffer management
let mut buffer = ZeroCopyBuffer::new(4096).unwrap();
let readable = buffer.readable_slice();   // direct view of buffered data, no copy

// Zero-copy reader: wraps any std::io::Read, borrows bytes from its internal buffer
let mut reader = ZeroCopyReader::new(file).unwrap(); // or with_capacity(file, 256 * 1024)
let consumed = match reader.zc_read(1024).unwrap() {
    Some(bytes) => {
        // process `bytes` without copying
        bytes.len()
    }
    None => 0, // not enough buffered data for zero-copy access
};
reader.zc_advance(consumed).unwrap();

// Zero-copy writer: wraps any std::io::Write, exposes its buffer for direct writes
let mut writer = ZeroCopyWriter::new(file).unwrap();
if let Some(buf) = writer.zc_write(5).unwrap() {
    buf.copy_from_slice(b"hello");
}
writer.zc_commit(5).unwrap();

// Memory-mapped zero-copy reading (requires the `mmap` feature)
#[cfg(feature = "mmap")]
{
    use zipora::io::MmapZeroCopyReader;
    let file = std::fs::File::open("data.bin").unwrap();
    let mmap_reader = MmapZeroCopyReader::new(file).unwrap();
    let all_bytes = mmap_reader.as_slice(); // entire file as one zero-copy slice
}

// Vectored I/O for scatter-gather operations
// (VectoredIO is a unit struct with associated functions — no constructor)
let buffers = [
    IoSlice::new(b"header"),
    IoSlice::new(b"body"),
    IoSlice::new(b"footer"),
];
let mut output: Vec<u8> = Vec::new();
let written = VectoredIO::write_vectored(&mut output, &buffers).unwrap();
```

## Version Management

```rust
use zipora::error::Result;
use zipora::io::{
    DataInput, DataOutput, MigrationRegistry, Version, VersionManager,
    VersionedSerialize, VecDataOutput,
};

// Implement VersionedSerialize for your type; serialize_versioned /
// deserialize_versioned are provided trait methods over DataOutput / DataInput.
struct MyData {
    field: u32,
}

impl VersionedSerialize for MyData {
    fn current_version() -> Version {
        Version::new(2, 1, 1)
    }

    fn serialize_with_manager<O: DataOutput>(
        &self,
        _manager: &mut VersionManager,
        output: &mut O,
    ) -> Result<()> {
        output.write_u32(self.field)
    }

    fn deserialize_with_manager<I: DataInput>(
        _manager: &mut VersionManager,
        input: &mut I,
    ) -> Result<Self> {
        Ok(MyData { field: input.read_u32()? })
    }
}

// Serialize with version information
let data = MyData { field: 42 };
let mut output = VecDataOutput::new();
data.serialize_versioned(&mut output).unwrap();

// A VersionManager tracks the version being read/written
// (constructed from the current version)
let mut manager = VersionManager::new(MyData::current_version());
manager.set_reading_version(Version::new(1, 0, 0));

// Register migration functions between on-disk formats (byte-level transforms)
let mut registry = MigrationRegistry::new();
registry.register_migration(
    Version::new(1, 0, 0),
    Version::new(2, 0, 0),
    |old_bytes| Ok(old_bytes.to_vec()), // transform old layout into new layout
);
```

## Performance Characteristics

For measured performance numbers, see [PERFORMANCE.md](PERFORMANCE.md).

## Related

For SIMD-decoded posting lists and sorted integer sequences, see StreamVByte in [COMPRESSION.md](COMPRESSION.md).
