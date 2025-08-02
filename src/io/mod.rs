//! I/O operations and streaming
//!
//! This module provides high-performance I/O operations including memory-mapped files,
//! streaming, and zero-copy operations.

pub mod data_input;
pub mod data_output;
pub mod mmap;
pub mod var_int;

// Re-export core types
pub use data_input::{DataInput, ReaderDataInput, SliceDataInput};
pub use data_output::{DataOutput, FileDataOutput, VecDataOutput, WriterDataOutput};
pub use var_int::{SignedVarInt, VarInt};

#[cfg(feature = "mmap")]
pub use data_input::MmapDataInput;
#[cfg(feature = "mmap")]
pub use mmap::{MemoryMappedInput, MemoryMappedOutput};

// Convenience functions
pub use data_input::{from_reader, from_slice};
pub use data_output::{to_file, to_file_append, to_vec, to_vec_with_capacity, to_writer};

#[cfg(feature = "mmap")]
pub use data_input::from_file;
