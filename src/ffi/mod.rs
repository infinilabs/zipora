//! C FFI compatibility layer
//!
//! This module provides C-compatible APIs for existing users who need to migrate
//! gradually from the C++ implementation.

#[cfg(feature = "ffi")]
pub mod c_api;

#[cfg(feature = "ffi")]
pub mod types;

#[cfg(feature = "ffi")]
pub mod containers;

#[cfg(feature = "ffi")]
pub mod blob_store;

#[cfg(feature = "ffi")]
pub mod algorithms;

// Re-export main C API when FFI feature is enabled
#[cfg(feature = "ffi")]
pub use c_api::*;

// Basic error handling for C FFI
#[cfg(feature = "ffi")]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CResult {
    Success = 0,
    InvalidInput = -1,
    MemoryError = -2,
    IoError = -3,
    UnsupportedOperation = -4,
    InternalError = -5,
    NotFound = -6,
}

#[cfg(feature = "ffi")]
impl From<crate::Result<()>> for CResult {
    fn from(result: crate::Result<()>) -> Self {
        match result {
            Ok(_) => CResult::Success,
            Err(crate::ZiporaError::InvalidData { message: _ }) => CResult::InvalidInput,
            Err(crate::ZiporaError::OutOfMemory { .. }) => CResult::MemoryError,
            Err(crate::ZiporaError::Io(_)) => CResult::IoError,
            Err(crate::ZiporaError::NotSupported { .. }) => CResult::UnsupportedOperation,
            _ => CResult::InternalError,
        }
    }
}
