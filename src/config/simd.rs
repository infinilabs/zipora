//! SIMD acceleration configuration.

use super::{Config, parse_env_bool};
use crate::error::{Result, ZiporaError};
use std::path::Path;
use serde::{Deserialize, Serialize};

/// SIMD configuration placeholder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SIMDConfig {
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    /// Enable AVX2 instructions
    pub enable_avx2: bool,
    /// Enable BMI2 instructions
    pub enable_bmi2: bool,
}

impl Default for SIMDConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_avx2: true,
            enable_bmi2: true,
        }
    }
}

impl Config for SIMDConfig {
    fn validate(&self) -> Result<()> {
        // SIMD configuration is always valid
        Ok(())
    }
    
    fn from_env_with_prefix(prefix: &str) -> Result<Self> {
        let mut config = Self::default();
        config.enable_simd = parse_env_bool(&format!("{}SIMD_ENABLE", prefix), config.enable_simd);
        config.enable_avx2 = parse_env_bool(&format!("{}SIMD_AVX2", prefix), config.enable_avx2);
        config.enable_bmi2 = parse_env_bool(&format!("{}SIMD_BMI2", prefix), config.enable_bmi2);
        config.validate()?;
        Ok(config)
    }
    
    fn performance_preset() -> Self {
        Self { enable_simd: true, enable_avx2: true, enable_bmi2: true }
    }
    
    fn memory_preset() -> Self {
        Self { enable_simd: false, enable_avx2: false, enable_bmi2: false }
    }
    
    fn realtime_preset() -> Self {
        Self { enable_simd: true, enable_avx2: true, enable_bmi2: true }
    }
    
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| ZiporaError::configuration(format!("Failed to serialize SIMD config: {}", e)))?;
        std::fs::write(path, serialized)
            .map_err(|e| ZiporaError::configuration(format!("Failed to write SIMD config file: {}", e)))?;
        Ok(())
    }
    
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ZiporaError::configuration(format!("Failed to read SIMD config file: {}", e)))?;
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| ZiporaError::configuration(format!("Failed to parse SIMD config file: {}", e)))?;
        config.validate()?;
        Ok(config)
    }
}