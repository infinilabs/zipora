//! Compression configuration for various algorithms.

use super::{Config, parse_env_var, parse_env_bool};
use crate::error::{Result, ZiporaError};
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Compression algorithm configuration placeholder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Default compression level
    pub level: u8,
    /// Enable dictionary compression
    pub enable_dictionary: bool,
    /// Dictionary size
    pub dictionary_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            level: 6,
            enable_dictionary: true,
            dictionary_size: 1024 * 1024,
        }
    }
}

impl Config for CompressionConfig {
    fn validate(&self) -> Result<()> {
        if self.level > 22 {
            return Err(ZiporaError::configuration("level must be between 0 and 22".to_string()));
        }
        Ok(())
    }
    
    fn from_env_with_prefix(prefix: &str) -> Result<Self> {
        let mut config = Self::default();
        config.level = parse_env_var(&format!("{}COMPRESSION_LEVEL", prefix), config.level);
        config.enable_dictionary = parse_env_bool(&format!("{}COMPRESSION_DICTIONARY", prefix), config.enable_dictionary);
        config.dictionary_size = parse_env_var(&format!("{}COMPRESSION_DICT_SIZE", prefix), config.dictionary_size);
        config.validate()?;
        Ok(config)
    }
    
    fn performance_preset() -> Self {
        Self { level: 1, enable_dictionary: false, dictionary_size: 0 }
    }
    
    fn memory_preset() -> Self {
        Self { level: 15, enable_dictionary: true, dictionary_size: 512 * 1024 }
    }
    
    fn realtime_preset() -> Self {
        Self { level: 0, enable_dictionary: false, dictionary_size: 0 }
    }
    
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| ZiporaError::configuration(format!("Failed to serialize compression config: {}", e)))?;
        std::fs::write(path, serialized)
            .map_err(|e| ZiporaError::configuration(format!("Failed to write compression config file: {}", e)))?;
        Ok(())
    }
    
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ZiporaError::configuration(format!("Failed to read compression config file: {}", e)))?;
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| ZiporaError::configuration(format!("Failed to parse compression config file: {}", e)))?;
        config.validate()?;
        Ok(config)
    }
}