//! Entropy coding and compression algorithms
//!
//! This module provides various entropy coding algorithms for advanced compression,
//! including Huffman coding, rANS (range Asymmetric Numeral Systems), and dictionary-based compression.

pub mod bit_ops;
pub mod context;
pub mod dictionary;
pub mod fse;
pub mod huffman;
pub mod parallel;
pub mod rans;
pub mod simd_huffman;

// Re-export main types
pub use bit_ops::{BitOps, BitOpsConfig, EntropyBitOps, BitOpsStats};
pub use context::{EntropyContext, EntropyContextConfig, ContextBuffer, EntropyResult, ContextStats};
pub use dictionary::{DictionaryBuilder, DictionaryCompressor, OptimizedDictionaryCompressor};
pub use fse::{
    FseEncoder, FseDecoder, FseConfig, FseTable, 
    fse_compress, fse_decompress, fse_zip, fse_unzip,
    fse_compress_with_config, fse_decompress_with_config,
    HardwareCapabilities, FastDivision, EntropyNormalizer
};

// Type aliases for benchmark compatibility
pub type EnhancedFseEncoder = FseEncoder;
pub type EnhancedFseConfig = FseConfig;
pub use huffman::{
    HuffmanDecoder, HuffmanEncoder, HuffmanTree,
    ContextualHuffmanEncoder, ContextualHuffmanDecoder, HuffmanOrder
};
pub use simd_huffman::{
    SimdHuffmanEncoder, SimdHuffmanConfig, HuffmanSimdTier
};
pub use rans::{
    Rans64Decoder as RansDecoder, Rans64Encoder, Rans64State as RansState, 
    Rans64Symbol as RansSymbol, AdaptiveRans64Encoder as AdaptiveRansEncoder,
    ParallelX1, ParallelX2, ParallelX4, ParallelX8
};
pub use parallel::{
    ParallelVariant, ParallelX2Variant, ParallelX4Variant, ParallelX8Variant,
    ParallelConfig, ParallelHuffmanEncoder, ParallelHuffmanDecoder,
    AdaptiveParallelEncoder, ParallelBenchmark, BenchmarkResult
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// std::collections::HashMap is used in method implementations

/// Entropy coding algorithm selection
///
/// Supports different entropy coding algorithms with different trade-offs
/// between compression ratio, speed, and memory usage.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EntropyAlgorithm {
    /// Huffman coding - classical entropy coding, good for simple distributions
    Huffman,
    /// rANS (range Asymmetric Numeral Systems) - modern, high-performance entropy coding
    Rans,
    /// FSE (Finite State Entropy) - high compression ratio, part of zstd family
    #[cfg(feature = "zstd")]
    Fse,
    /// Legacy kFSE option for compatibility with reference implementations
    #[cfg(feature = "zstd")]
    KFse,
    /// Dictionary-based compression for repetitive data
    Dictionary,
    /// Automatic selection based on data characteristics
    Auto,
}

impl Default for EntropyAlgorithm {
    fn default() -> Self {
        Self::Auto
    }
}

impl EntropyAlgorithm {
    /// Get the name of the algorithm
    pub fn name(self) -> &'static str {
        match self {
            EntropyAlgorithm::Huffman => "Huffman",
            EntropyAlgorithm::Rans => "rANS",
            #[cfg(feature = "zstd")]
            EntropyAlgorithm::Fse => "FSE",
            #[cfg(feature = "zstd")]
            EntropyAlgorithm::KFse => "kFSE",
            EntropyAlgorithm::Dictionary => "Dictionary",
            EntropyAlgorithm::Auto => "Auto",
        }
    }
    
    /// Check if the algorithm is available in the current build
    pub fn is_available(self) -> bool {
        match self {
            EntropyAlgorithm::Huffman => true,
            EntropyAlgorithm::Rans => true,
            #[cfg(feature = "zstd")]
            EntropyAlgorithm::Fse => true,
            #[cfg(feature = "zstd")]
            EntropyAlgorithm::KFse => true,
            #[cfg(not(feature = "zstd"))]
            EntropyAlgorithm::Fse => false,
            #[cfg(not(feature = "zstd"))]
            EntropyAlgorithm::KFse => false,
            EntropyAlgorithm::Dictionary => true,
            EntropyAlgorithm::Auto => true,
        }
    }
    
    /// Get all available algorithms for the current build
    pub fn available_algorithms() -> Vec<Self> {
        let mut algorithms = vec![
            Self::Huffman,
            Self::Rans,
            Self::Dictionary,
            Self::Auto,
        ];
        
        #[cfg(feature = "zstd")]
        {
            algorithms.push(Self::Fse);
            algorithms.push(Self::KFse);
        }
        
        algorithms
    }
    
    /// Select optimal algorithm based on data characteristics
    pub fn select_for_data(data: &[u8]) -> Self {
        if data.is_empty() {
            return Self::Huffman; // Default for empty data
        }
        
        let entropy = EntropyStats::calculate_entropy(data);
        let size = data.len();
        
        // Calculate repetitiveness
        let mut char_counts = std::collections::HashMap::new();
        for &byte in data {
            *char_counts.entry(byte).or_insert(0) += 1;
        }
        let unique_chars = char_counts.len();
        let repetitiveness = 1.0 - (unique_chars as f64 / 256.0);
        
        match (entropy, size, repetitiveness) {
            // High repetitiveness - use dictionary compression
            (_, _, r) if r > 0.8 => Self::Dictionary,
            
            // Low entropy, larger data - FSE is excellent
            #[cfg(feature = "zstd")]
            (e, s, _) if e < 4.0 && s > 1024 => Self::Fse,
            
            // Medium entropy, medium size - rANS is balanced
            (e, s, _) if e >= 4.0 && e <= 6.0 && s > 256 => Self::Rans,
            
            // High entropy or small data - Huffman is simple and effective
            _ => Self::Huffman,
        }
    }
}

/// Entropy codec configuration
///
/// Unified configuration for all entropy coding algorithms with algorithm-specific options.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EntropyConfig {
    /// Selected entropy algorithm
    pub algorithm: EntropyAlgorithm,
    
    /// FSE-specific configuration
    #[cfg(feature = "zstd")]
    pub fse_config: Option<FseConfig>,
    
    /// Compression level (1-22, where higher = better compression but slower)
    pub compression_level: i32,
    
    /// Enable adaptive mode for dynamic algorithm selection
    pub adaptive: bool,
    
    /// Dictionary size for dictionary-based algorithms
    pub dict_size: usize,
    
    /// Enable fast decode mode (may reduce compression ratio)
    pub fast_decode: bool,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            algorithm: EntropyAlgorithm::Auto,
            #[cfg(feature = "zstd")]
            fse_config: Some(FseConfig::default()),
            compression_level: 3,
            adaptive: true,
            dict_size: 0,
            fast_decode: false,
        }
    }
}

impl EntropyConfig {
    /// Configuration optimized for fast compression
    pub fn fast() -> Self {
        Self {
            algorithm: EntropyAlgorithm::Huffman,
            compression_level: 1,
            fast_decode: true,
            #[cfg(feature = "zstd")]
            fse_config: Some(FseConfig::fast_compression()),
            ..Default::default()
        }
    }
    
    /// Configuration optimized for high compression ratio
    pub fn high_compression() -> Self {
        Self {
            #[cfg(feature = "zstd")]
            algorithm: EntropyAlgorithm::Fse,
            #[cfg(not(feature = "zstd"))]
            algorithm: EntropyAlgorithm::Rans,
            compression_level: 19,
            adaptive: true,
            dict_size: 32 * 1024,
            #[cfg(feature = "zstd")]
            fse_config: Some(FseConfig::high_compression()),
            ..Default::default()
        }
    }
    
    /// Configuration optimized for balanced performance
    pub fn balanced() -> Self {
        Self::default()
    }
}

// Note: Universal entropy encoder/decoder implementation is complex due to different
// constructor signatures for each algorithm. For now, use the individual encoders directly.
// A full implementation would require more sophisticated state management.

/// Statistics for entropy coding operations
#[derive(Debug, Clone, PartialEq)]
pub struct EntropyStats {
    /// Original size in bytes
    pub input_size: usize,
    /// Compressed size in bytes
    pub output_size: usize,
    /// Compression ratio (output/input)
    pub compression_ratio: f64,
    /// Bits per symbol achieved
    pub bits_per_symbol: f64,
    /// Theoretical entropy of the input
    pub entropy: f64,
    /// Encoding efficiency (theoretical / actual)
    pub efficiency: f64,
}

impl EntropyStats {
    /// Create new entropy statistics
    pub fn new(input_size: usize, output_size: usize, entropy: f64) -> Self {
        let compression_ratio = if input_size > 0 {
            output_size as f64 / input_size as f64
        } else {
            0.0
        };

        let bits_per_symbol = if input_size > 0 {
            (output_size * 8) as f64 / input_size as f64
        } else {
            0.0
        };

        let efficiency = if bits_per_symbol > 0.0 {
            entropy / bits_per_symbol
        } else {
            0.0
        };

        Self {
            input_size,
            output_size,
            compression_ratio,
            bits_per_symbol,
            entropy,
            efficiency,
        }
    }

    /// Calculate space savings as a percentage
    pub fn space_savings(&self) -> f64 {
        (1.0 - self.compression_ratio) * 100.0
    }

    /// Calculate theoretical entropy from byte frequencies
    pub fn calculate_entropy(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        // Count byte frequencies
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        // Calculate entropy
        let total = data.len() as f64;
        let mut entropy = 0.0;

        for &freq in &frequencies {
            if freq > 0 {
                let p = freq as f64 / total;
                entropy -= p * p.log2();
            }
        }

        entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_stats_calculation() {
        let stats = EntropyStats::new(1000, 600, 4.5);

        assert_eq!(stats.input_size, 1000);
        assert_eq!(stats.output_size, 600);
        assert!((stats.compression_ratio - 0.6).abs() < 0.001);
        assert!((stats.bits_per_symbol - 4.8).abs() < 0.001);
        assert!((stats.efficiency - 0.9375).abs() < 0.001);
        assert!((stats.space_savings() - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_entropy_calculation() {
        // Test uniform distribution (maximum entropy)
        let uniform_data = (0..=255).collect::<Vec<u8>>();
        let entropy = EntropyStats::calculate_entropy(&uniform_data);
        assert!((entropy - 8.0).abs() < 0.001); // Should be close to 8 bits

        // Test single symbol (minimum entropy)
        let single_symbol = vec![42u8; 100];
        let entropy = EntropyStats::calculate_entropy(&single_symbol);
        assert!(entropy < 0.001); // Should be close to 0

        // Test empty data
        let empty: Vec<u8> = vec![];
        let entropy = EntropyStats::calculate_entropy(&empty);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_entropy_stats_edge_cases() {
        // Test zero input size
        let stats = EntropyStats::new(0, 0, 0.0);
        assert_eq!(stats.compression_ratio, 0.0);
        assert_eq!(stats.bits_per_symbol, 0.0);
        assert_eq!(stats.efficiency, 0.0);

        // Test zero output size
        let stats = EntropyStats::new(100, 0, 4.0);
        assert_eq!(stats.compression_ratio, 0.0);
        assert_eq!(stats.space_savings(), 100.0);
    }
}
