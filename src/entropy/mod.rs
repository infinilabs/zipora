//! Entropy coding and compression algorithms
//!
//! This module provides various entropy coding algorithms for advanced compression,
//! including Huffman coding, rANS (range Asymmetric Numeral Systems), and dictionary-based compression.

pub mod dictionary;
pub mod huffman;
pub mod rans;
pub mod rans_fixed;
pub mod rans_minimal;
pub mod rans_reference;
pub mod rans_simple;
pub mod rans_simple_debug;

// Re-export main types
pub use dictionary::{DictionaryBuilder, DictionaryCompressor, OptimizedDictionaryCompressor};
pub use huffman::{HuffmanDecoder, HuffmanEncoder, HuffmanTree};
pub use rans::{RansDecoder, RansEncoder, RansState};

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
