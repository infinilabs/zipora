//! Blob store configuration for different storage backends.

use super::{Config, parse_env_var, parse_env_bool};
use crate::error::{Result, ZiporaError};
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Blob store configuration placeholder.
/// 
/// This is a minimal implementation showing the configuration pattern.
/// Future implementations will expand this with comprehensive blob store settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlobStoreConfig {
    /// Default compression level
    pub compression_level: u8,
    /// Enable checksums
    pub enable_checksums: bool,
    /// Block size for storage
    pub block_size: usize,
}

impl Default for BlobStoreConfig {
    fn default() -> Self {
        Self {
            compression_level: 6,
            enable_checksums: true,
            block_size: 64 * 1024,
        }
    }
}

impl Config for BlobStoreConfig {
    fn validate(&self) -> Result<()> {
        if self.compression_level > 22 {
            return Err(ZiporaError::configuration(
                "compression_level must be between 0 and 22".to_string()
            ));
        }
        if self.block_size == 0 {
            return Err(ZiporaError::configuration(
                "block_size must be greater than 0".to_string()
            ));
        }
        Ok(())
    }
    
    fn from_env_with_prefix(prefix: &str) -> Result<Self> {
        let mut config = Self::default();
        config.compression_level = parse_env_var(&format!("{}BLOB_COMPRESSION_LEVEL", prefix), config.compression_level);
        config.enable_checksums = parse_env_bool(&format!("{}BLOB_CHECKSUMS", prefix), config.enable_checksums);
        config.block_size = parse_env_var(&format!("{}BLOB_BLOCK_SIZE", prefix), config.block_size);
        config.validate()?;
        Ok(config)
    }
    
    fn performance_preset() -> Self {
        Self {
            compression_level: 1,
            enable_checksums: false,
            block_size: 128 * 1024,
        }
    }
    
    fn memory_preset() -> Self {
        Self {
            compression_level: 15,
            enable_checksums: true,
            block_size: 16 * 1024,
        }
    }
    
    fn realtime_preset() -> Self {
        Self {
            compression_level: 0,
            enable_checksums: false,
            block_size: 64 * 1024,
        }
    }
    
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| ZiporaError::configuration(format!("Failed to serialize blob store config: {}", e)))?;
        std::fs::write(path, serialized)
            .map_err(|e| ZiporaError::configuration(format!("Failed to write blob store config file: {}", e)))?;
        Ok(())
    }
    
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ZiporaError::configuration(format!("Failed to read blob store config file: {}", e)))?;
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| ZiporaError::configuration(format!("Failed to parse blob store config file: {}", e)))?;
        config.validate()?;
        Ok(config)
    }
}