//! Error handling for the infini-zip library
//!
//! This module provides comprehensive error handling with detailed error information
//! for all library operations.

use thiserror::Error;

/// Main error type for the infini-zip library
#[derive(Error, Debug)]
pub enum ToplingError {
    /// I/O related errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Invalid data format or corruption
    #[error("Invalid data: {message}")]
    InvalidData { 
        /// Error message describing the issue
        message: String 
    },
    
    /// Index out of bounds access
    #[error("Out of bounds: index {index}, size {size}")]
    OutOfBounds { 
        /// The invalid index
        index: usize, 
        /// The valid size/length
        size: usize 
    },
    
    /// Memory allocation failures
    #[error("Memory allocation failed: requested {size} bytes")]
    OutOfMemory { 
        /// Number of bytes requested
        size: usize 
    },
    
    /// Compression/decompression errors
    #[error("Compression error: {message}")]
    Compression { 
        /// Error message from compression library
        message: String 
    },
    
    /// Blob store related errors
    #[error("Blob store error: {message}")]
    BlobStore { 
        /// Error message describing the blob store issue
        message: String 
    },
    
    /// Trie/FSA related errors
    #[error("Trie error: {message}")]
    Trie { 
        /// Error message describing the trie issue
        message: String 
    },
    
    /// Checksum validation failures
    #[error("Checksum mismatch: expected {expected:x}, got {actual:x}")]
    ChecksumMismatch { 
        /// Expected checksum value
        expected: u32, 
        /// Actual checksum value
        actual: u32 
    },
    
    /// Feature not supported or not implemented
    #[error("Not supported: {feature}")]
    NotSupported { 
        /// Description of the unsupported feature
        feature: String 
    },
    
    /// Configuration or parameter errors
    #[error("Invalid configuration: {message}")]
    Configuration { 
        /// Configuration error message
        message: String 
    },
    
    /// Resource already in use or locked
    #[error("Resource busy: {resource}")]
    ResourceBusy { 
        /// Description of the busy resource
        resource: String 
    },
}

impl ToplingError {
    /// Create an invalid data error
    pub fn invalid_data<S: Into<String>>(message: S) -> Self {
        Self::InvalidData { message: message.into() }
    }
    
    /// Create an out of bounds error
    pub fn out_of_bounds(index: usize, size: usize) -> Self {
        Self::OutOfBounds { index, size }
    }
    
    /// Create an out of memory error
    pub fn out_of_memory(size: usize) -> Self {
        Self::OutOfMemory { size }
    }
    
    /// Create a compression error
    pub fn compression<S: Into<String>>(message: S) -> Self {
        Self::Compression { message: message.into() }
    }
    
    /// Create a blob store error
    pub fn blob_store<S: Into<String>>(message: S) -> Self {
        Self::BlobStore { message: message.into() }
    }
    
    /// Create a trie error
    pub fn trie<S: Into<String>>(message: S) -> Self {
        Self::Trie { message: message.into() }
    }
    
    /// Create a checksum mismatch error
    pub fn checksum_mismatch(expected: u32, actual: u32) -> Self {
        Self::ChecksumMismatch { expected, actual }
    }
    
    /// Create a not supported error
    pub fn not_supported<S: Into<String>>(feature: S) -> Self {
        Self::NotSupported { feature: feature.into() }
    }
    
    /// Create an I/O error from a message
    pub fn io_error<S: Into<String>>(message: S) -> Self {
        Self::Io(std::io::Error::new(std::io::ErrorKind::Other, message.into()))
    }
    
    /// Create a not found error (convenience method for I/O errors)
    pub fn not_found<S: Into<String>>(message: S) -> Self {
        Self::Io(std::io::Error::new(std::io::ErrorKind::NotFound, message.into()))
    }
    
    /// Create a configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration { message: message.into() }
    }
    
    /// Create a resource busy error
    pub fn resource_busy<S: Into<String>>(resource: S) -> Self {
        Self::ResourceBusy { resource: resource.into() }
    }
    
    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Io(_) => true,
            Self::OutOfMemory { .. } => true,
            Self::ResourceBusy { .. } => true,
            Self::Compression { .. } => false,
            Self::InvalidData { .. } => false,
            Self::OutOfBounds { .. } => false,
            Self::BlobStore { .. } => false,
            Self::Trie { .. } => false,
            Self::ChecksumMismatch { .. } => false,
            Self::NotSupported { .. } => false,
            Self::Configuration { .. } => false,
        }
    }
    
    /// Get the error category for logging/metrics
    pub fn category(&self) -> &'static str {
        match self {
            Self::Io(_) => "io",
            Self::InvalidData { .. } => "data",
            Self::OutOfBounds { .. } => "bounds",
            Self::OutOfMemory { .. } => "memory",
            Self::Compression { .. } => "compression",
            Self::BlobStore { .. } => "blob_store",
            Self::Trie { .. } => "trie",
            Self::ChecksumMismatch { .. } => "checksum",
            Self::NotSupported { .. } => "unsupported",
            Self::Configuration { .. } => "config",
            Self::ResourceBusy { .. } => "resource",
        }
    }
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, ToplingError>;

/// Assert that an index is within bounds
#[inline]
pub fn check_bounds(index: usize, size: usize) -> Result<()> {
    if index >= size {
        Err(ToplingError::out_of_bounds(index, size))
    } else {
        Ok(())
    }
}

/// Assert that a range is within bounds
#[inline]
pub fn check_range(start: usize, end: usize, size: usize) -> Result<()> {
    if start > end {
        return Err(ToplingError::invalid_data(
            format!("Invalid range: start {} > end {}", start, end)
        ));
    }
    if end > size {
        return Err(ToplingError::out_of_bounds(end, size));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = ToplingError::invalid_data("test message");
        assert_eq!(err.category(), "data");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_bounds_checking() {
        assert!(check_bounds(5, 10).is_ok());
        assert!(check_bounds(10, 10).is_err());
        assert!(check_bounds(15, 10).is_err());
    }

    #[test]
    fn test_range_checking() {
        assert!(check_range(2, 8, 10).is_ok());
        assert!(check_range(8, 2, 10).is_err()); // start > end
        assert!(check_range(2, 15, 10).is_err()); // end > size
    }

    #[test]
    fn test_error_categories() {
        let io_err = ToplingError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert_eq!(io_err.category(), "io");
        assert!(io_err.is_recoverable());

        let data_err = ToplingError::invalid_data("corrupt");
        assert_eq!(data_err.category(), "data");
        assert!(!data_err.is_recoverable());
    }

    #[test]
    fn test_all_error_types() {
        // Test all error constructor functions
        let invalid_data = ToplingError::invalid_data("test data error");
        assert_eq!(invalid_data.category(), "data");
        assert!(!invalid_data.is_recoverable());

        let bounds_err = ToplingError::out_of_bounds(5, 3);
        assert_eq!(bounds_err.category(), "bounds");
        assert!(!bounds_err.is_recoverable());

        let memory_err = ToplingError::out_of_memory(1024);
        assert_eq!(memory_err.category(), "memory");
        assert!(memory_err.is_recoverable());

        let compression_err = ToplingError::compression("ZSTD error");
        assert_eq!(compression_err.category(), "compression");
        assert!(!compression_err.is_recoverable());

        let blob_err = ToplingError::blob_store("blob not found");
        assert_eq!(blob_err.category(), "blob_store");
        assert!(!blob_err.is_recoverable());

        let trie_err = ToplingError::trie("invalid trie structure");
        assert_eq!(trie_err.category(), "trie");
        assert!(!trie_err.is_recoverable());

        let checksum_err = ToplingError::checksum_mismatch(0x12345678, 0x87654321);
        assert_eq!(checksum_err.category(), "checksum");
        assert!(!checksum_err.is_recoverable());

        let unsupported_err = ToplingError::not_supported("GPU acceleration");
        assert_eq!(unsupported_err.category(), "unsupported");
        assert!(!unsupported_err.is_recoverable());

        let config_err = ToplingError::configuration("invalid block size");
        assert_eq!(config_err.category(), "config");
        assert!(!config_err.is_recoverable());

        let busy_err = ToplingError::resource_busy("file lock");
        assert_eq!(busy_err.category(), "resource");
        assert!(busy_err.is_recoverable());
    }

    #[test]
    fn test_error_display() {
        let err = ToplingError::invalid_data("test message");
        let display = format!("{}", err);
        assert!(display.contains("Invalid data"));
        assert!(display.contains("test message"));

        let bounds_err = ToplingError::out_of_bounds(10, 5);
        let bounds_display = format!("{}", bounds_err);
        assert!(bounds_display.contains("Out of bounds"));
        assert!(bounds_display.contains("10"));
        assert!(bounds_display.contains("5"));

        let checksum_err = ToplingError::checksum_mismatch(0xDEADBEEF, 0xCAFEBABE);
        let checksum_display = format!("{}", checksum_err);
        assert!(checksum_display.contains("Checksum mismatch"));
        assert!(checksum_display.contains("deadbeef"));
        assert!(checksum_display.contains("cafebabe"));
    }

    #[test]
    fn test_from_io_error() {
        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let topling_error: ToplingError = io_error.into();
        
        assert_eq!(topling_error.category(), "io");
        assert!(topling_error.is_recoverable());
        
        let display = format!("{}", topling_error);
        assert!(display.contains("I/O error"));
    }

    #[test]
    fn test_error_debug() {
        let err = ToplingError::invalid_data("debug test");
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidData"));
        assert!(debug_str.contains("debug test"));
    }

    #[test]
    fn test_recoverable_errors() {
        // Test recoverable errors
        let io_err = ToplingError::Io(std::io::Error::new(std::io::ErrorKind::Interrupted, "test"));
        assert!(io_err.is_recoverable());

        let memory_err = ToplingError::out_of_memory(1000);
        assert!(memory_err.is_recoverable());

        let busy_err = ToplingError::resource_busy("test resource");
        assert!(busy_err.is_recoverable());
    }

    #[test]
    fn test_non_recoverable_errors() {
        // Test non-recoverable errors
        let compression_err = ToplingError::compression("test");
        assert!(!compression_err.is_recoverable());

        let data_err = ToplingError::invalid_data("test");
        assert!(!data_err.is_recoverable());

        let bounds_err = ToplingError::out_of_bounds(1, 0);
        assert!(!bounds_err.is_recoverable());

        let blob_err = ToplingError::blob_store("test");
        assert!(!blob_err.is_recoverable());

        let trie_err = ToplingError::trie("test");
        assert!(!trie_err.is_recoverable());

        let checksum_err = ToplingError::checksum_mismatch(1, 2);
        assert!(!checksum_err.is_recoverable());

        let unsupported_err = ToplingError::not_supported("test");
        assert!(!unsupported_err.is_recoverable());

        let config_err = ToplingError::configuration("test");
        assert!(!config_err.is_recoverable());
    }

    #[test]
    fn test_edge_case_bounds_checking() {
        // Edge cases for bounds checking
        assert!(check_bounds(0, 1).is_ok());
        assert!(check_bounds(0, 0).is_err());
        assert!(check_bounds(usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn test_edge_case_range_checking() {
        // Edge cases for range checking
        assert!(check_range(0, 0, 0).is_ok());
        assert!(check_range(0, 0, 1).is_ok());
        assert!(check_range(5, 5, 5).is_ok());
        assert!(check_range(5, 5, 10).is_ok());
        
        // Invalid ranges
        assert!(check_range(5, 4, 10).is_err()); // start > end
        assert!(check_range(0, 11, 10).is_err()); // end > size
        assert!(check_range(usize::MAX, 0, 10).is_err()); // start > end
    }
}