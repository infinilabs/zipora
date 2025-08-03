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

/// C-compatible result codes for FFI operations
#[cfg(feature = "ffi")]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CResult {
    /// Operation completed successfully
    Success = 0,
    /// Invalid input parameters provided
    InvalidInput = -1,
    /// Memory allocation or access error
    MemoryError = -2,
    /// Input/output operation failed
    IoError = -3,
    /// Operation not supported in current context
    UnsupportedOperation = -4,
    /// Internal library error occurred
    InternalError = -5,
    /// Requested item was not found
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
