//! I/O operations and streaming
//!
//! This module provides high-performance I/O operations including memory-mapped files,
//! streaming, zero-copy operations, and advanced buffering strategies.

pub mod data_input;
pub mod data_output;
pub mod mmap;
pub mod range_stream;
pub mod stream_buffer;
pub mod var_int;
pub mod zero_copy;

// New advanced serialization features
pub mod complex_types;
pub mod endian;
pub mod smart_ptr;
pub mod var_int_variants;
pub mod versioning;

// SIMD-accelerated operations
pub mod simd_encoding;
pub mod simd_memory;
pub mod simd_validation;

// Re-export core types
pub use data_input::{DataInput, ReaderDataInput, SliceDataInput};
pub use data_output::{DataOutput, FileDataOutput, VecDataOutput, WriterDataOutput};
pub use var_int::{SignedVarInt, VarInt};

// Re-export new advanced serialization features
pub use complex_types::{
    ComplexSerialize, ComplexTypeConfig, ComplexTypeSerializer, NestedSerialize,
};
pub use endian::{EndianConfig, EndianConvert, EndianIO, Endianness};
pub use smart_ptr::{
    DeserializationContext, SerializableType, SerializationContext, SmartPtrConfig,
    SmartPtrSerialize, SmartPtrSerializer,
};
pub use var_int_variants::{
    VarIntEncoder, VarIntStrategy, choose_optimal_strategy, choose_optimal_strategy_signed,
};
pub use versioning::{
    MigrationRegistry, Version, VersionConfig, VersionManager, VersionProxy, VersionedSerialize,
    VersionedSerializer,
};

// Re-export new I/O & Serialization features
pub use range_stream::{MultiRangeReader, RangeReader, RangeWriter};
pub use stream_buffer::{StreamBufferConfig, StreamBufferedReader, StreamBufferedWriter};
pub use zero_copy::{
    VectoredIO, ZeroCopyBuffer, ZeroCopyRead, ZeroCopyReader, ZeroCopyWrite, ZeroCopyWriter,
};

#[cfg(feature = "mmap")]
pub use data_input::MmapDataInput;
#[cfg(feature = "mmap")]
pub use mmap::{AccessPattern, InputStrategy, MemoryMappedInput, MemoryMappedOutput};
#[cfg(feature = "mmap")]
pub use zero_copy::mmap::MmapZeroCopyReader;

// Convenience functions
pub use data_input::{from_reader, from_slice};
pub use data_output::{to_file, to_file_append, to_vec, to_vec_with_capacity, to_writer};
pub use range_stream::range;

#[cfg(feature = "mmap")]
pub use data_input::from_file;
