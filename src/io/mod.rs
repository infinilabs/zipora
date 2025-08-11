//! I/O operations and streaming
//!
//! This module provides high-performance I/O operations including memory-mapped files,
//! streaming, zero-copy operations, and advanced buffering strategies.

pub mod data_input;
pub mod data_output;
pub mod mmap;
pub mod var_int;
pub mod stream_buffer;
pub mod range_stream;
pub mod zero_copy;

// New advanced serialization features
pub mod endian;
pub mod smart_ptr;
pub mod complex_types;
pub mod versioning;
pub mod var_int_variants;

// Re-export core types
pub use data_input::{DataInput, ReaderDataInput, SliceDataInput};
pub use data_output::{DataOutput, FileDataOutput, VecDataOutput, WriterDataOutput};
pub use var_int::{SignedVarInt, VarInt};

// Re-export new advanced serialization features
pub use endian::{Endianness, EndianConvert, EndianIO, EndianConfig};
pub use smart_ptr::{SmartPtrSerialize, SerializableType, SerializationContext, DeserializationContext, SmartPtrConfig, SmartPtrSerializer};
pub use complex_types::{ComplexSerialize, ComplexTypeConfig, ComplexTypeSerializer, NestedSerialize};
pub use versioning::{Version, VersionProxy, VersionManager, VersionedSerialize, VersionMigration, MigrationRegistry, VersionConfig, VersionedSerializer};
pub use var_int_variants::{VarIntStrategy, VarIntEncoder, choose_optimal_strategy, choose_optimal_strategy_signed};

// Re-export new I/O & Serialization features
pub use stream_buffer::{StreamBufferConfig, StreamBufferedReader, StreamBufferedWriter};
pub use range_stream::{RangeReader, RangeWriter, MultiRangeReader};
pub use zero_copy::{ZeroCopyRead, ZeroCopyWrite, ZeroCopyBuffer, ZeroCopyReader, ZeroCopyWriter, VectoredIO};

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
