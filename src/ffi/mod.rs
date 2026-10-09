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
impl CResult {
    /// Convert a [`crate::ZiporaError`] reference into its corresponding [`CResult`] code.
    pub(crate) fn from_zipora_error(err: &crate::ZiporaError) -> Self {
        match err {
            crate::ZiporaError::Io(io_err)
                if io_err.kind() == std::io::ErrorKind::NotFound =>
            {
                CResult::NotFound
            }
            crate::ZiporaError::Io(_) => CResult::IoError,
            crate::ZiporaError::InvalidData { .. }
            | crate::ZiporaError::InvalidParameter { .. }
            | crate::ZiporaError::OutOfBounds { .. }
            | crate::ZiporaError::Misaligned { .. }
            | crate::ZiporaError::Configuration { .. } => CResult::InvalidInput,
            crate::ZiporaError::OutOfMemory { .. }
            | crate::ZiporaError::ResourceExhausted { .. } => CResult::MemoryError,
            crate::ZiporaError::NotSupported { .. } => CResult::UnsupportedOperation,
            _ => CResult::InternalError,
        }
    }
}

#[cfg(feature = "ffi")]
impl From<crate::Result<()>> for CResult {
    fn from(result: crate::Result<()>) -> Self {
        match result {
            Ok(_) => CResult::Success,
            Err(ref err) => Self::from_zipora_error(err),
        }
    }
}
