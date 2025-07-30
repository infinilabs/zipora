//! I/O operations and streaming
//!
//! This module provides high-performance I/O operations including memory-mapped files,
//! streaming, and zero-copy operations.

pub mod data_input;
pub mod data_output;
pub mod var_int;

// Re-export core types
pub use data_input::{DataInput, SliceDataInput, ReaderDataInput};
pub use data_output::{DataOutput, VecDataOutput, WriterDataOutput, FileDataOutput};
pub use var_int::{VarInt, SignedVarInt};

#[cfg(feature = "mmap")]
pub use data_input::MmapDataInput;

// Convenience functions
pub use data_input::{from_slice, from_reader};
pub use data_output::{to_vec, to_vec_with_capacity, to_writer, to_file, to_file_append};

#[cfg(feature = "mmap")]
pub use data_input::from_file;