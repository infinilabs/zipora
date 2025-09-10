//! Rich Configuration APIs for Zipora
//! 
//! This module provides comprehensive configuration APIs that enable fine-grained
//! control over all aspects of Zipora's behavior, including data structures,
//! algorithms, memory management, compression, and performance optimization.
//! 
//! # Overview
//! 
//! The configuration system is designed around the following principles:
//! 
//! - **Comprehensive**: Cover all configurable aspects of the system
//! - **Type-safe**: Use Rust's type system to prevent invalid configurations
//! - **Performance-oriented**: Enable optimal performance through configuration
//! - **Environment-aware**: Support initialization from environment variables and files
//! - **Preset-friendly**: Provide common configuration presets for typical use cases
//! - **Validation**: Comprehensive validation to ensure configuration correctness
//! 
//! # Configuration Traits
//! 
//! The [`Config`] trait provides common functionality for all configuration types,
//! including validation, environment initialization, and preset management.
//! 
//! # Core Configuration Types
//! 
//! - [`NestLoudsTrieConfig`]: Configuration for Nested LOUDS Trie data structures
//! - [`BlobStoreConfig`]: Configuration for blob storage systems
//! - [`MemoryConfig`]: Configuration for memory management and allocation
//! - [`CompressionConfig`]: Configuration for compression algorithms
//! - [`CacheConfig`]: Configuration for caching and locality optimization
//! - [`SIMDConfig`]: Configuration for SIMD acceleration and hardware features
//! 
//! # Builder Patterns
//! 
//! Complex configurations support builder patterns for easy construction:
//! 
//! ```rust
//! use zipora::config::NestLoudsTrieConfig;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = NestLoudsTrieConfig::builder()
//!     .nest_level(3)
//!     .max_fragment_length(1024)
//!     .enable_queue_compression(true)
//!     .compression_level(6)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//! 
//! # Preset Configurations
//! 
//! Common configuration presets are provided for typical use cases:
//! 
//! ```rust
//! use zipora::config::{NestLoudsTrieConfig, Config};
//! 
//! // High-performance configuration optimized for speed
//! let config = NestLoudsTrieConfig::performance_preset();
//! 
//! // Memory-efficient configuration optimized for space
//! let config = NestLoudsTrieConfig::memory_preset();
//! 
//! // Real-time configuration with latency guarantees
//! let config = NestLoudsTrieConfig::realtime_preset();
//! ```
//! 
//! # Environment Initialization
//! 
//! Configurations can be initialized from environment variables:
//! 
//! ```rust
//! use zipora::config::{NestLoudsTrieConfig, Config};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize from environment variables with ZIPORA_ prefix
//! let config = NestLoudsTrieConfig::from_env()?;
//! 
//! // Initialize from environment with custom prefix
//! let config = NestLoudsTrieConfig::from_env_with_prefix("MYAPP_")?;
//! # Ok(())
//! # }
//! ```

use crate::error::Result;
use std::env;
use std::fmt;
use std::path::Path;

pub mod nest_louds_trie;
pub mod blob_store;
pub mod memory;
pub mod compression;
pub mod cache;
pub mod simd;

#[cfg(test)]
mod tests;

pub use nest_louds_trie::NestLoudsTrieConfig;
pub use blob_store::BlobStoreConfig;
pub use memory::MemoryConfig;
pub use compression::CompressionConfig;
pub use cache::CacheConfig;
pub use simd::SIMDConfig;

/// Common configuration trait providing validation, environment initialization,
/// and preset management functionality.
pub trait Config: Clone + fmt::Debug {
    /// Validate the configuration for correctness and consistency.
    /// 
    /// # Returns
    /// 
    /// `Ok(())` if the configuration is valid, `Err` with details if invalid.
    fn validate(&self) -> Result<()>;
    
    /// Initialize configuration from environment variables.
    /// 
    /// Environment variables should use the format `ZIPORA_{COMPONENT}_{FIELD}`.
    /// For example, `ZIPORA_TRIE_NEST_LEVEL=3` sets the nest level for trie configuration.
    /// 
    /// # Returns
    /// 
    /// A configuration instance initialized from environment variables, or default
    /// values if environment variables are not set.
    fn from_env() -> Result<Self>
    where
        Self: Default,
    {
        Self::from_env_with_prefix("ZIPORA_")
    }
    
    /// Initialize configuration from environment variables with a custom prefix.
    /// 
    /// # Arguments
    /// 
    /// * `prefix` - The environment variable prefix to use
    /// 
    /// # Returns
    /// 
    /// A configuration instance initialized from environment variables.
    fn from_env_with_prefix(prefix: &str) -> Result<Self>
    where
        Self: Default;
    
    /// Get a performance-optimized preset configuration.
    /// 
    /// This preset is optimized for maximum performance and throughput,
    /// potentially at the cost of increased memory usage.
    fn performance_preset() -> Self;
    
    /// Get a memory-optimized preset configuration.
    /// 
    /// This preset is optimized for minimal memory usage, potentially
    /// at the cost of reduced performance.
    fn memory_preset() -> Self;
    
    /// Get a real-time preset configuration.
    /// 
    /// This preset is optimized for low latency and predictable performance,
    /// suitable for real-time applications.
    fn realtime_preset() -> Self;
    
    /// Get a balanced preset configuration.
    /// 
    /// This preset provides a good balance between performance, memory usage,
    /// and features. It's suitable for most general-purpose applications.
    fn balanced_preset() -> Self
    where
        Self: Default,
    {
        Self::default()
    }
    
    /// Save configuration to a file.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The file path to save the configuration to
    /// 
    /// # Returns
    /// 
    /// `Ok(())` if the configuration was saved successfully.
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()>;
    
    /// Load configuration from a file.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The file path to load the configuration from
    /// 
    /// # Returns
    /// 
    /// The loaded configuration instance.
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self>;
}

/// Configuration validation error details.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ValidationError {
    /// The field that failed validation
    pub field: String,
    /// The invalid value
    pub value: String,
    /// Description of why the value is invalid
    pub reason: String,
    /// Suggested valid values or ranges
    pub suggestion: Option<String>,
}

impl ValidationError {
    /// Create a new validation error.
    /// 
    /// # Arguments
    /// 
    /// * `field` - The field name that failed validation
    /// * `value` - The invalid value
    /// * `reason` - Description of why the value is invalid
    pub fn new(field: &str, value: &str, reason: &str) -> Self {
        Self {
            field: field.to_string(),
            value: value.to_string(),
            reason: reason.to_string(),
            suggestion: None,
        }
    }
    
    /// Add a suggestion for valid values.
    /// 
    /// # Arguments
    /// 
    /// * `suggestion` - Suggested valid values or ranges
    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.suggestion = Some(suggestion.to_string());
        self
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid configuration for field '{}': value '{}' is invalid ({})",
               self.field, self.value, self.reason)?;
        
        if let Some(ref suggestion) = self.suggestion {
            write!(f, ". Suggested values: {}", suggestion)?;
        }
        
        Ok(())
    }
}

impl std::error::Error for ValidationError {}

/// Utility function to parse environment variable with fallback to default.
/// 
/// # Arguments
/// 
/// * `var_name` - The environment variable name
/// * `default` - The default value if the environment variable is not set
/// 
/// # Returns
/// 
/// The parsed value or the default value.
pub fn parse_env_var<T>(var_name: &str, default: T) -> T
where
    T: std::str::FromStr + Clone,
{
    env::var(var_name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Utility function to parse boolean environment variable.
/// 
/// Accepts: "true", "1", "yes", "on" (case-insensitive) as true,
/// everything else as false.
/// 
/// # Arguments
/// 
/// * `var_name` - The environment variable name
/// * `default` - The default value if the environment variable is not set
/// 
/// # Returns
/// 
/// The parsed boolean value or the default value.
pub fn parse_env_bool(var_name: &str, default: bool) -> bool {
    env::var(var_name)
        .ok()
        .map(|s| {
            let s = s.to_lowercase();
            matches!(s.as_str(), "true" | "1" | "yes" | "on")
        })
        .unwrap_or(default)
}

/// Common configuration presets for different use cases.
pub mod presets {
    //! Pre-configured settings for common use cases.
    //! 
    //! This module provides configuration presets that are optimized for
    //! specific scenarios and workloads.
    
    /// Performance-oriented configuration presets.
    pub mod performance {
        //! Configurations optimized for maximum performance and throughput.
        
        /// High-performance configuration for maximum throughput.
        /// 
        /// - Enables all SIMD optimizations
        /// - Uses aggressive caching
        /// - Optimizes for CPU performance over memory usage
        /// - Enables hardware acceleration features
        pub const THROUGHPUT: &str = "throughput";
        
        /// Low-latency configuration for real-time applications.
        /// 
        /// - Minimizes allocation overhead
        /// - Uses thread-local caching
        /// - Optimizes for predictable performance
        /// - Enables deadline-based scheduling
        pub const LOW_LATENCY: &str = "low_latency";
    }
    
    /// Memory-oriented configuration presets.
    pub mod memory {
        //! Configurations optimized for minimal memory usage.
        
        /// Memory-efficient configuration for constrained environments.
        /// 
        /// - Minimizes memory allocation
        /// - Uses compact data structures
        /// - Enables aggressive compression
        /// - Reduces cache usage
        pub const EFFICIENT: &str = "efficient";
        
        /// Configuration for embedded or resource-constrained systems.
        /// 
        /// - Fixed memory limits
        /// - No dynamic allocation after initialization
        /// - Minimal feature set
        /// - Optimized for small memory footprint
        pub const EMBEDDED: &str = "embedded";
    }
    
    /// Application-specific configuration presets.
    pub mod application {
        //! Configurations optimized for specific application types.
        
        /// Configuration optimized for text processing applications.
        /// 
        /// - Optimized string handling
        /// - UTF-8 acceleration
        /// - Text-specific compression
        /// - Line-based processing optimizations
        pub const TEXT_PROCESSING: &str = "text_processing";
        
        /// Configuration optimized for large dataset processing.
        /// 
        /// - External sorting support
        /// - Memory-mapped file operations
        /// - Streaming algorithms
        /// - Batch processing optimizations
        pub const BIG_DATA: &str = "big_data";
        
        /// Configuration optimized for interactive applications.
        /// 
        /// - Low-latency operations
        /// - Responsive design
        /// - Progressive processing
        /// - User experience optimizations
        pub const INTERACTIVE: &str = "interactive";
    }
}

