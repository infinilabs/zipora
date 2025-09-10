//! Cache configuration for performance optimization.

use super::{Config, parse_env_var, parse_env_bool};
use crate::error::{Result, ZiporaError};
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Cache configuration placeholder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache size in bytes
    pub size: usize,
    /// Enable cache prefetching
    pub enable_prefetching: bool,
    /// Cache line size
    pub line_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            size: 16 * 1024 * 1024,
            enable_prefetching: true,
            line_size: 64,
        }
    }
}

impl Config for CacheConfig {
    fn validate(&self) -> Result<()> {
        if self.size == 0 {
            return Err(ZiporaError::configuration("size must be greater than 0".to_string()));
        }
        Ok(())
    }
    
    fn from_env_with_prefix(prefix: &str) -> Result<Self> {
        let mut config = Self::default();
        config.size = parse_env_var(&format!("{}CACHE_SIZE", prefix), config.size);
        config.enable_prefetching = parse_env_bool(&format!("{}CACHE_PREFETCHING", prefix), config.enable_prefetching);
        config.line_size = parse_env_var(&format!("{}CACHE_LINE_SIZE", prefix), config.line_size);
        config.validate()?;
        Ok(config)
    }
    
    fn performance_preset() -> Self {
        Self { size: 64 * 1024 * 1024, enable_prefetching: true, line_size: 64 }
    }
    
    fn memory_preset() -> Self {
        Self { size: 4 * 1024 * 1024, enable_prefetching: false, line_size: 64 }
    }
    
    fn realtime_preset() -> Self {
        Self { size: 32 * 1024 * 1024, enable_prefetching: true, line_size: 64 }
    }
    
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| ZiporaError::configuration(format!("Failed to serialize cache config: {}", e)))?;
        std::fs::write(path, serialized)
            .map_err(|e| ZiporaError::configuration(format!("Failed to write cache config file: {}", e)))?;
        Ok(())
    }
    
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ZiporaError::configuration(format!("Failed to read cache config file: {}", e)))?;
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| ZiporaError::configuration(format!("Failed to parse cache config file: {}", e)))?;
        config.validate()?;
        Ok(config)
    }
}