//! Real-time compression with adaptive algorithms
//!
//! This module provides adaptive compression algorithms that automatically
//! choose the best compression method based on data characteristics and
//! performance requirements.

pub mod adaptive;
pub mod realtime;

pub use adaptive::{AdaptiveCompressor, CompressionProfile, AdaptiveConfig};
pub use realtime::{RealtimeCompressor, RealtimeConfig, CompressionMode};

use crate::error::{ToplingError, Result};
use std::time::Duration;

/// Compression algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Algorithm {
    /// No compression
    None,
    /// Fast LZ4 compression
    Lz4,
    /// ZSTD compression with configurable level
    Zstd(i32),
    /// Huffman coding (for entropy-heavy data)
    Huffman,
    /// rANS encoding (for statistical compression)
    Rans,
    /// Dictionary-based compression
    Dictionary,
    /// Hybrid approach using multiple algorithms
    Hybrid,
}

impl Algorithm {
    /// Get the expected compression speed (operations per second)
    pub fn compression_speed(&self) -> f64 {
        match self {
            Algorithm::None => f64::INFINITY,
            Algorithm::Lz4 => 500_000_000.0,      // Very fast
            Algorithm::Zstd(level) => match level {
                1..=3 => 200_000_000.0,           // Fast levels
                4..=9 => 50_000_000.0,            // Medium levels
                _ => 10_000_000.0,                 // Slow levels
            },
            Algorithm::Huffman => 100_000_000.0,   // Fast entropy coding
            Algorithm::Rans => 80_000_000.0,      // Good entropy coding
            Algorithm::Dictionary => 150_000_000.0, // Fast pattern matching
            Algorithm::Hybrid => 50_000_000.0,     // Depends on mix
        }
    }
    
    /// Get the expected compression ratio (0.0 to 1.0, lower = better)
    pub fn compression_ratio(&self) -> f64 {
        match self {
            Algorithm::None => 1.0,
            Algorithm::Lz4 => 0.6,
            Algorithm::Zstd(level) => match level {
                1..=3 => 0.5,
                4..=9 => 0.4,
                _ => 0.3,
            },
            Algorithm::Huffman => 0.65,
            Algorithm::Rans => 0.55,
            Algorithm::Dictionary => 0.45,
            Algorithm::Hybrid => 0.35,
        }
    }
    
    /// Get the memory usage in bytes per input byte
    pub fn memory_usage(&self) -> f64 {
        match self {
            Algorithm::None => 0.0,
            Algorithm::Lz4 => 0.1,
            Algorithm::Zstd(level) => match level {
                1..=3 => 0.5,
                4..=9 => 2.0,
                _ => 8.0,
            },
            Algorithm::Huffman => 1.0,
            Algorithm::Rans => 1.5,
            Algorithm::Dictionary => 3.0,
            Algorithm::Hybrid => 4.0,
        }
    }
}

/// Performance requirements for compression
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Minimum required throughput (bytes per second)
    pub min_throughput: u64,
    /// Maximum memory usage (bytes)
    pub max_memory: usize,
    /// Target compression ratio (0.0 to 1.0)
    pub target_ratio: f64,
    /// Priority: speed vs compression quality (0.0 = speed, 1.0 = quality)
    pub speed_vs_quality: f64,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_latency: Duration::from_millis(100),
            min_throughput: 100_000_000, // 100 MB/s
            max_memory: 64 * 1024 * 1024, // 64 MB
            target_ratio: 0.5,
            speed_vs_quality: 0.5,
        }
    }
}

/// Statistics for compression operations
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Total operations performed
    pub operations: u64,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Total bytes after compression
    pub bytes_compressed: u64,
    /// Total compression time
    pub total_time: Duration,
    /// Algorithm usage statistics
    pub algorithm_usage: std::collections::HashMap<Algorithm, u64>,
    /// Average compression ratio
    pub avg_ratio: f64,
    /// Average throughput (bytes/sec)
    pub avg_throughput: f64,
}

impl CompressionStats {
    /// Calculate overall compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.bytes_processed == 0 {
            0.0
        } else {
            self.bytes_compressed as f64 / self.bytes_processed as f64
        }
    }
    
    /// Calculate average throughput
    pub fn throughput(&self) -> f64 {
        if self.total_time.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.bytes_processed as f64 / self.total_time.as_secs_f64()
        }
    }
    
    /// Update statistics with a new operation
    pub fn update(&mut self, input_size: usize, output_size: usize, duration: Duration, algorithm: Algorithm) {
        self.operations += 1;
        self.bytes_processed += input_size as u64;
        self.bytes_compressed += output_size as u64;
        self.total_time += duration;
        
        *self.algorithm_usage.entry(algorithm).or_insert(0) += 1;
        
        // Update rolling averages
        let ratio = output_size as f64 / input_size as f64;
        self.avg_ratio = (self.avg_ratio * (self.operations - 1) as f64 + ratio) / self.operations as f64;
        
        let throughput = input_size as f64 / duration.as_secs_f64();
        self.avg_throughput = (self.avg_throughput * (self.operations - 1) as f64 + throughput) / self.operations as f64;
    }
}

/// Base trait for compression algorithms
pub trait Compressor: Send + Sync {
    /// Compress data
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>>;
    
    /// Decompress data
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>>;
    
    /// Get the algorithm type
    fn algorithm(&self) -> Algorithm;
    
    /// Estimate compression ratio for given data
    fn estimate_ratio(&self, data: &[u8]) -> f64 {
        // Default implementation: try compression on a sample
        if data.len() > 1024 {
            let sample = &data[..1024];
            if let Ok(compressed) = self.compress(sample) {
                return compressed.len() as f64 / sample.len() as f64;
            }
        }
        self.algorithm().compression_ratio()
    }
    
    /// Check if this compressor is suitable for the given requirements
    fn is_suitable(&self, requirements: &PerformanceRequirements, data_size: usize) -> bool {
        let algo = self.algorithm();
        let expected_time = data_size as f64 / algo.compression_speed();
        let expected_memory = (data_size as f64 * algo.memory_usage()) as usize;
        
        Duration::from_secs_f64(expected_time) <= requirements.max_latency &&
        expected_memory <= requirements.max_memory &&
        algo.compression_ratio() <= requirements.target_ratio
    }
}

/// No-op compressor (pass-through)
pub struct NoCompressor;

impl Compressor for NoCompressor {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(data.to_vec())
    }
    
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(data.to_vec())
    }
    
    fn algorithm(&self) -> Algorithm {
        Algorithm::None
    }
}

/// LZ4 compressor wrapper
pub struct Lz4Compressor;

impl Compressor for Lz4Compressor {
    fn compress(&self, #[cfg_attr(not(feature = "lz4"), allow(unused_variables))] data: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "lz4")]
        {
            lz4_flex::compress_prepend_size(data)
                .map_err(|e| ToplingError::compression(&format!("LZ4 compression failed: {}", e)))
        }
        #[cfg(not(feature = "lz4"))]
        {
            Err(ToplingError::not_supported("LZ4 compression not enabled"))
        }
    }
    
    fn decompress(&self, #[cfg_attr(not(feature = "lz4"), allow(unused_variables))] data: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "lz4")]
        {
            lz4_flex::decompress_size_prepended(data)
                .map_err(|e| ToplingError::compression(&format!("LZ4 decompression failed: {}", e)))
        }
        #[cfg(not(feature = "lz4"))]
        {
            Err(ToplingError::not_supported("LZ4 decompression not enabled"))
        }
    }
    
    fn algorithm(&self) -> Algorithm {
        Algorithm::Lz4
    }
}

/// ZSTD compressor wrapper
pub struct ZstdCompressor {
    level: i32,
}

impl ZstdCompressor {
    pub fn new(level: i32) -> Self {
        Self { level }
    }
}

impl Compressor for ZstdCompressor {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::bulk::compress(data, self.level)
            .map_err(|e| ToplingError::compression(&format!("ZSTD compression failed: {}", e)))
    }
    
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::bulk::decompress(data, 100 * 1024 * 1024) // 100MB limit
            .map_err(|e| ToplingError::compression(&format!("ZSTD decompression failed: {}", e)))
    }
    
    fn algorithm(&self) -> Algorithm {
        Algorithm::Zstd(self.level)
    }
}

/// Huffman compressor wrapper
pub struct HuffmanCompressor {
    // For now, we'll use a placeholder since HuffmanEncoder requires different API
    _phantom: std::marker::PhantomData<()>,
}

impl HuffmanCompressor {
    pub fn new(_data: &[u8]) -> Result<Self> {
        // Placeholder implementation until HuffmanEncoder is available
        Ok(Self { _phantom: std::marker::PhantomData })
    }
}

impl Compressor for HuffmanCompressor {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Placeholder implementation
        Ok(data.to_vec())
    }
    
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Placeholder implementation
        Ok(data.to_vec())
    }
    
    fn algorithm(&self) -> Algorithm {
        Algorithm::Huffman
    }
}

/// Factory for creating compressors
pub struct CompressorFactory;

impl CompressorFactory {
    /// Create a compressor for the given algorithm
    pub fn create(algorithm: Algorithm, training_data: Option<&[u8]>) -> Result<Box<dyn Compressor>> {
        match algorithm {
            Algorithm::None => Ok(Box::new(NoCompressor)),
            Algorithm::Lz4 => Ok(Box::new(Lz4Compressor)),
            Algorithm::Zstd(level) => Ok(Box::new(ZstdCompressor::new(level))),
            Algorithm::Huffman => {
                if let Some(data) = training_data {
                    Ok(Box::new(HuffmanCompressor::new(data)?))
                } else {
                    Err(ToplingError::invalid_data("Huffman compressor requires training data"))
                }
            }
            _ => Err(ToplingError::not_supported(&format!("algorithm not implemented: {:?}", algorithm))),
        }
    }
    
    /// Get all available algorithms
    pub fn available_algorithms() -> Vec<Algorithm> {
        vec![
            Algorithm::None,
            Algorithm::Lz4,
            Algorithm::Zstd(1),
            Algorithm::Zstd(3),
            Algorithm::Zstd(6),
            Algorithm::Zstd(9),
            Algorithm::Huffman,
        ]
    }
    
    /// Select the best algorithm for given requirements and data
    pub fn select_best(
        requirements: &PerformanceRequirements,
        data: &[u8],
    ) -> Algorithm {
        let available = Self::available_algorithms();
        let mut best_algorithm = Algorithm::None;
        let mut best_score = f64::NEG_INFINITY;
        
        for algorithm in available {
            // Skip algorithms that require training data if not available
            if matches!(algorithm, Algorithm::Huffman | Algorithm::Rans | Algorithm::Dictionary) {
                continue;
            }
            
            let speed = algorithm.compression_speed();
            let ratio = algorithm.compression_ratio();
            let memory = algorithm.memory_usage() * data.len() as f64;
            
            // Check hard constraints
            if memory > requirements.max_memory as f64 {
                continue;
            }
            
            let expected_time = data.len() as f64 / speed;
            if Duration::from_secs_f64(expected_time) > requirements.max_latency {
                continue;
            }
            
            // Calculate score based on requirements
            let speed_score = speed / 1_000_000_000.0; // Normalize to GB/s
            let ratio_score = 1.0 - ratio; // Better ratio = higher score
            let memory_score = 1.0 - (memory / requirements.max_memory as f64);
            
            let weighted_score = 
                requirements.speed_vs_quality * ratio_score +
                (1.0 - requirements.speed_vs_quality) * speed_score +
                0.1 * memory_score;
            
            if weighted_score > best_score {
                best_score = weighted_score;
                best_algorithm = algorithm;
            }
        }
        
        best_algorithm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_algorithm_properties() {
        assert_eq!(Algorithm::None.compression_ratio(), 1.0);
        assert!(Algorithm::Lz4.compression_speed() > Algorithm::Zstd(9).compression_speed());
        assert!(Algorithm::Zstd(9).compression_ratio() < Algorithm::Lz4.compression_ratio());
    }
    
    #[test]
    fn test_performance_requirements() {
        let req = PerformanceRequirements::default();
        assert_eq!(req.speed_vs_quality, 0.5);
        assert!(req.max_latency > Duration::ZERO);
    }
    
    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::default();
        
        stats.update(1000, 500, Duration::from_millis(10), Algorithm::Lz4);
        assert_eq!(stats.operations, 1);
        assert_eq!(stats.compression_ratio(), 0.5);
        
        stats.update(2000, 800, Duration::from_millis(20), Algorithm::Zstd(3));
        assert_eq!(stats.operations, 2);
        assert!(stats.compression_ratio() < 0.7);
    }
    
    #[test]
    fn test_no_compressor() {
        let compressor = NoCompressor;
        let data = b"test data";
        
        let compressed = compressor.compress(data).unwrap();
        assert_eq!(compressed, data);
        
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
        
        assert_eq!(compressor.algorithm(), Algorithm::None);
    }
    
    #[cfg(feature = "lz4")]
    #[test]
    fn test_lz4_compressor() {
        let compressor = Lz4Compressor;
        let data = b"test data that should compress well with repeated patterns";
        
        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        assert_eq!(compressor.algorithm(), Algorithm::Lz4);
    }
    
    #[test]
    fn test_zstd_compressor() {
        let compressor = ZstdCompressor::new(3);
        let data = b"test data that should compress well with repeated patterns and more text";
        
        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        
        assert_eq!(decompressed, data);
        assert_eq!(compressor.algorithm(), Algorithm::Zstd(3));
        assert!(compressed.len() < data.len());
    }
    
    #[test]
    fn test_compressor_factory() {
        let algorithms = CompressorFactory::available_algorithms();
        assert!(!algorithms.is_empty());
        assert!(algorithms.contains(&Algorithm::None));
        assert!(algorithms.contains(&Algorithm::Lz4));
    }
    
    #[test]
    fn test_algorithm_selection() {
        let req = PerformanceRequirements {
            max_latency: Duration::from_millis(1),
            speed_vs_quality: 0.9, // Prioritize speed
            ..Default::default()
        };
        
        let data = vec![0u8; 1000];
        let algorithm = CompressorFactory::select_best(&req, &data);
        
        // Should select a fast algorithm
        assert!(matches!(algorithm, Algorithm::None | Algorithm::Lz4));
    }
    
    #[test]
    #[cfg(feature = "lz4")]
    fn test_compressor_suitability() {
        let compressor = Lz4Compressor;
        let req = PerformanceRequirements::default();
        
        assert!(compressor.is_suitable(&req, 1024));
        
        let strict_req = PerformanceRequirements {
            max_latency: Duration::from_nanos(1),
            ..Default::default()
        };
        
        assert!(!compressor.is_suitable(&strict_req, 1024 * 1024));
    }
}