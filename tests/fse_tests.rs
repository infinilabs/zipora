//! Comprehensive tests for FSE (Finite State Entropy) implementation
//!
//! This test suite covers all aspects of the FSE implementation including:
//! - Basic compression/decompression functionality
//! - Configuration and presets
//! - Integration with PA-Zip compression pipeline
//! - Reference implementation compatibility
//! - Performance benchmarks
//! - Edge cases and error handling

use zipora::compression::dict_zip::compression_types::{
    FseCompressor, FseConfig as PaZipFseConfig, apply_fse_compression, fse_unzip_reference,
    fse_zip_reference, remove_fse_compression,
};
use zipora::entropy::{
    FseConfig, FseDecoder, FseEncoder, FseTable, fse_compress, fse_compress_with_config,
    fse_decompress, fse_decompress_with_config,
};

/// Test basic FSE functionality
#[test]
fn test_fse_basic_functionality() {
    let config = FseConfig::default();

    // Test configuration validation
    assert!(config.validate().is_ok());

    // Test encoder creation
    let encoder_result = FseEncoder::new(config.clone());

    #[cfg(feature = "zstd")]
    {
        let mut encoder = encoder_result.unwrap();
        let _decoder = FseDecoder::new();

        // Test empty data
        let empty_compressed = encoder.compress(b"").unwrap();
        assert!(empty_compressed.is_empty());

        // Test small data
        let small_data = b"hello";
        let small_compressed = encoder.compress(small_data).unwrap();
        assert!(!small_compressed.is_empty());
    }

    #[cfg(not(feature = "zstd"))]
    {
        // Without zstd, should still create encoder without errors
        assert!(encoder_result.is_ok());
    }
}

/// Test FSE configuration presets
#[test]
fn test_fse_config_presets() {
    let fast = FseConfig::fast_compression();
    let high = FseConfig::high_compression();
    let balanced = FseConfig::balanced();
    let realtime = FseConfig::realtime();

    // Validate all presets
    assert!(fast.validate().is_ok());
    assert!(high.validate().is_ok());
    assert!(balanced.validate().is_ok());
    assert!(realtime.validate().is_ok());

    // Test characteristics
    assert!(fast.table_log <= balanced.table_log);
    assert!(balanced.table_log <= high.table_log);
    assert!(realtime.table_log <= fast.table_log);

    assert!(fast.compression_level <= balanced.compression_level);
    assert!(balanced.compression_level <= high.compression_level);
    assert!(realtime.compression_level <= fast.compression_level);

    assert!(high.dict_size >= balanced.dict_size);
    assert!(balanced.dict_size >= fast.dict_size);

    assert!(!realtime.adaptive);
    assert!(realtime.fast_decode);
}

/// Test FSE table creation and validation
#[test]
#[cfg(feature = "zstd")]
fn test_fse_table_creation() {
    let mut frequencies = [0u32; 256];

    // Set up frequency distribution
    frequencies[b'a' as usize] = 1000;
    frequencies[b'b' as usize] = 500;
    frequencies[b'c' as usize] = 250;
    frequencies[b'd' as usize] = 125;
    frequencies[b'e' as usize] = 125;

    let config = FseConfig::default();
    let table_result = FseTable::new(&frequencies, &config);

    assert!(table_result.is_ok());
    let table = table_result.unwrap();

    assert_eq!(table.table_log, config.table_log);
    assert_eq!(table.max_symbol, b'e');
    assert!(!table.states.is_empty());
    assert_eq!(table.states.len(), 1 << config.table_log);

    // Test symbol encoding/decoding
    for symbol in [b'a', b'b', b'c', b'd', b'e'] {
        if let Some((new_state, nb_bits)) = table.encode_symbol(symbol, 1024) {
            assert!(nb_bits <= 16); // Reasonable range for number of bits
            assert!(new_state > 0);
        }
    }

    // Test invalid table (all zeros)
    let zero_frequencies = [0u32; 256];
    let invalid_table = FseTable::new(&zero_frequencies, &config);
    assert!(invalid_table.is_err());
}

/// Test FSE compression with different data types
#[test]
#[cfg(feature = "zstd")]
fn test_fse_compression_data_types() {
    let mut encoder = FseEncoder::new(FseConfig::balanced()).unwrap();
    let mut decoder = FseDecoder::new();

    let test_cases = [
        // Text with patterns
        b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.".to_vec(),
        
        // Binary data with repetition
        [0x00, 0x01, 0x02, 0x03].repeat(50),
        
        // Single character repetition
        vec![b'A'; 200],
        
        // Mixed patterns
        b"AAABBBCCCDDDEEEFFFGGGHHHIIIJJJKKKLLLMMMNNNOOOPPPQQQRRRSSSTTTUUUVVVWWWXXXYYYZZZ".to_vec(),
        
        // Random-like data (harder to compress)
        (0..=255).collect::<Vec<u8>>(),
    ];

    for (i, data) in test_cases.iter().enumerate() {
        println!("Testing data type {}: {} bytes", i, data.len());

        let compressed = encoder.compress(data).unwrap();
        let decompressed = decoder.decompress(&compressed).unwrap();

        assert_eq!(data, &decompressed, "Roundtrip failed for test case {}", i);

        // Check compression stats
        let stats = encoder.stats();
        assert_eq!(stats.input_size, data.len());
        assert_eq!(stats.output_size, compressed.len());
        assert!(stats.entropy >= 0.0);

        println!("  Compression ratio: {:.3}", stats.compression_ratio);
        println!("  Entropy: {:.3} bits", stats.entropy);
        println!("  Efficiency: {:.3}", stats.efficiency);
    }
}

/// Test FSE with dictionary
#[test]
#[cfg(feature = "zstd")]
fn test_fse_with_dictionary() {
    let dictionary = b"The quick brown fox jumps over the lazy dog".to_vec();
    let config = FseConfig::high_compression();

    let mut encoder = FseEncoder::with_dictionary(config.clone(), dictionary.clone()).unwrap();
    let mut decoder = FseDecoder::with_config(config).unwrap();

    // Test data similar to dictionary
    let test_data = b"The quick brown fox runs fast over the lazy cat";

    let compressed = encoder.compress(test_data).unwrap();
    let decompressed = decoder.decompress(&compressed).unwrap();

    assert_eq!(test_data, &decompressed[..]);

    // Dictionary should help with compression ratio
    let stats = encoder.stats();
    println!(
        "Dictionary compression ratio: {:.3}",
        stats.compression_ratio
    );
    println!("Dictionary efficiency: {:.3}", stats.efficiency);
}

/// Test FSE convenience functions
#[test]
fn test_fse_convenience_functions() {
    let test_data = b"FSE convenience function test data with some patterns and repetition";

    let compressed = fse_compress(test_data);

    #[cfg(feature = "zstd")]
    {
        let compressed = compressed.unwrap();
        let decompressed = fse_decompress(&compressed).unwrap();

        assert_eq!(test_data, &decompressed[..]);

        // Test with custom config
        let config = FseConfig::fast_compression();
        let custom_compressed = fse_compress_with_config(test_data, config).unwrap();
        let custom_decompressed =
            fse_decompress_with_config(&custom_compressed, FseConfig::fast_compression()).unwrap();

        assert_eq!(test_data, &custom_decompressed[..]);
    }

    #[cfg(not(feature = "zstd"))]
    {
        // Should handle gracefully without zstd
        assert!(compressed.is_ok());
    }
}

/// Test entropy algorithm selection (commented out - EntropyAlgorithm not implemented)
#[test]
fn test_entropy_algorithm_selection() {
    // Note: EntropyAlgorithm enum and related functionality is not yet implemented
    // This test validates FSE availability instead

    #[cfg(feature = "zstd")]
    {
        // Test that FSE encoder can be created (indicates FSE is available)
        let config = FseConfig::default();
        let encoder_result = FseEncoder::new(config);
        assert!(encoder_result.is_ok());
    }

    #[cfg(not(feature = "zstd"))]
    {
        // Without zstd, FSE is not available but should handle gracefully
        let config = FseConfig::default();
        let encoder_result = FseEncoder::new(config);
        assert!(encoder_result.is_ok()); // Should still create encoder, may use fallback
    }
}

/// Test automatic algorithm selection (commented out - EntropyAlgorithm not implemented)
#[test]
fn test_auto_algorithm_selection() {
    // Note: Automatic algorithm selection is not yet implemented
    // This test validates different FSE configurations for different data types instead

    // High repetitiveness data - test with FSE
    let repetitive_data = b"AAAAAAAAAA".repeat(100);

    #[cfg(feature = "zstd")]
    {
        let mut encoder = FseEncoder::new(FseConfig::balanced()).unwrap();
        let compressed = encoder.compress(&repetitive_data).unwrap();
        assert!(!compressed.is_empty());

        let stats = encoder.stats();
        // Repetitive data should achieve good compression
        assert!(stats.compression_ratio < 0.5); // Should achieve significant compression
    }

    // Random data - test with FSE
    let random_data: Vec<u8> = (0..=255).collect();

    #[cfg(feature = "zstd")]
    {
        let mut encoder = FseEncoder::new(FseConfig::balanced()).unwrap();
        let compressed = encoder.compress(&random_data).unwrap();
        assert!(!compressed.is_empty());

        let stats = encoder.stats();
        // Random data should be harder to compress
        assert!(stats.compression_ratio > 0.8); // Should not compress much
    }
}

/// Test universal entropy encoder/decoder (commented out - complex implementation removed)
#[test]
fn test_universal_entropy_encoder() {
    // Note: Universal entropy encoder/decoder implementation was complex due to different
    // constructor signatures for each algorithm. This test validates individual FSE functionality instead.

    let test_data = b"Universal entropy encoder test data with various patterns";

    // Test FSE directly
    #[cfg(feature = "zstd")]
    {
        let mut encoder = FseEncoder::new(FseConfig::balanced()).unwrap();
        let mut decoder = FseDecoder::new();

        let compressed = encoder.compress(test_data).unwrap();
        let decompressed = decoder.decompress(&compressed).unwrap();

        assert_eq!(test_data, &decompressed[..]);
    }

    #[cfg(not(feature = "zstd"))]
    {
        // Test that basic FSE functions work without zstd
        let compressed = fse_compress(test_data).unwrap_or_default();
        assert!(!compressed.is_empty() || test_data.is_empty());
    }
}

/// Test PA-Zip FSE integration
#[test]
fn test_pa_zip_fse_integration() {
    let test_data = b"PA-Zip FSE integration test with encoded bit stream simulation";

    let config = PaZipFseConfig::for_pa_zip();
    assert_eq!(config.table_log, 11);
    assert_eq!(config.compression_level, 6);

    let compressed = apply_fse_compression(test_data, &config).unwrap();
    let decompressed = remove_fse_compression(&compressed, &config).unwrap();

    #[cfg(feature = "zstd")]
    {
        assert_eq!(test_data, &decompressed[..]);

        // For repetitive data, should achieve compression (allowing for 2-byte magic prefix overhead)
        if test_data.len() > 32 {
            assert!(compressed.len() <= test_data.len() + 2);
        }
    }

    #[cfg(not(feature = "zstd"))]
    {
        // Should handle gracefully without errors
        assert!(!compressed.is_empty());
    }

    // Test fast PA-Zip config
    let fast_config = PaZipFseConfig::fast_pa_zip();
    assert_eq!(fast_config.table_log, 9);
    assert_eq!(fast_config.compression_level, 1);
    assert!(!fast_config.adaptive);
    assert!(fast_config.fast_decode);

    let fast_compressed = apply_fse_compression(test_data, &fast_config).unwrap();
    let fast_decompressed = remove_fse_compression(&fast_compressed, &fast_config).unwrap();

    #[cfg(feature = "zstd")]
    {
        assert_eq!(test_data, &fast_decompressed[..]);
    }
}

/// Test reference implementation compatibility
#[test]
fn test_fse_reference_compatibility() {
    let test_data = b"Reference implementation compatibility test data with patterns";
    let mut compressed_buffer = vec![0u8; test_data.len() * 2];
    let mut compressed_size = 0;

    // Test FSE_zip reference function
    let compress_result =
        fse_zip_reference(test_data, &mut compressed_buffer, &mut compressed_size);

    assert!(compress_result.is_ok());

    #[cfg(feature = "zstd")]
    {
        if let Ok(true) = compress_result {
            assert!(compressed_size > 0);
            assert!(compressed_size <= test_data.len());

            // Test FSE_unzip reference function
            let mut decompressed_buffer = vec![0u8; test_data.len() * 2];
            let decompressed_size = fse_unzip_reference(
                &compressed_buffer[..compressed_size],
                &mut decompressed_buffer,
            )
            .unwrap();

            assert_eq!(decompressed_size, test_data.len());
            assert_eq!(&decompressed_buffer[..decompressed_size], test_data);
        }
    }

    // Test with small data (should return false)
    let small_data = b"ab";
    let small_result = fse_zip_reference(small_data, &mut compressed_buffer, &mut compressed_size);
    assert!(small_result.is_ok());
    assert!(!small_result.unwrap());

    // Test with single byte (should return false)
    let tiny_data = b"a";
    let tiny_result = fse_zip_reference(tiny_data, &mut compressed_buffer, &mut compressed_size);
    assert!(tiny_result.is_ok());
    assert!(!tiny_result.unwrap());
}

/// Test FSE compressor state management
#[test]
#[cfg(feature = "zstd")]
fn test_fse_compressor_state() {
    let config = PaZipFseConfig::for_pa_zip();
    let result = FseCompressor::with_config(config);

    #[cfg(feature = "zstd")]
    {
        let mut compressor = result.unwrap();
        let test_data = b"FSE compressor state management test data";

        // Test initial compression
        let compressed1 = compressor.compress(test_data).unwrap();
        let decompressed1 = compressor.decompress(&compressed1).unwrap();
        assert_eq!(&decompressed1, test_data);

        // Test statistics
        if let Some(stats) = compressor.stats() {
            assert_eq!(stats.input_size, test_data.len());
            assert!(stats.entropy >= 0.0);
        }

        // Test reset
        compressor.reset().unwrap();

        // Test compression after reset
        let compressed2 = compressor.compress(test_data).unwrap();
        let decompressed2 = compressor.decompress(&compressed2).unwrap();
        assert_eq!(&decompressed2, test_data);

        // Test empty data handling
        let empty_compressed = compressor.compress(b"").unwrap();
        assert!(empty_compressed.is_empty());

        let empty_decompressed = compressor.decompress(&empty_compressed).unwrap();
        assert!(empty_decompressed.is_empty());
    }

    #[cfg(not(feature = "zstd"))]
    {
        // Should handle gracefully even without zstd
        assert!(result.is_ok());
    }
}

/// Test FSE error handling
#[test]
fn test_fse_error_handling() {
    // Test invalid configuration
    let invalid_config = FseConfig {
        table_log: 25,         // Too large
        max_symbol: 70000,     // Too large
        compression_level: 30, // Too large
        ..Default::default()
    };

    assert!(invalid_config.validate().is_err());

    let encoder_result = FseEncoder::new(invalid_config);
    assert!(encoder_result.is_err());

    // Test buffer overflow scenarios
    let test_data = b"test data for buffer overflow testing";
    let mut small_buffer = vec![0u8; 5]; // Too small
    let mut compressed_size = 0;

    let result = fse_zip_reference(test_data, &mut small_buffer, &mut compressed_size);

    #[cfg(feature = "zstd")]
    {
        // Should handle buffer size gracefully
        if result.is_err() {
            // Error should be about buffer size
            assert!(format!("{:?}", result).contains("buffer"));
        }
    }
}

/// Test FSE with different data sizes
#[test]
#[cfg(feature = "zstd")]
fn test_fse_data_sizes() {
    let mut encoder = FseEncoder::new(FseConfig::balanced()).unwrap();
    let mut decoder = FseDecoder::new();

    let test_sizes = vec![
        0,     // Empty
        1,     // Single byte
        2,     // Two bytes (reference minimum)
        10,    // Small
        100,   // Medium
        1000,  // Large
        10000, // Very large
    ];

    for size in test_sizes {
        println!("Testing size: {} bytes", size);

        let test_data = if size == 0 {
            Vec::new()
        } else if size <= 2 {
            vec![b'A'; size]
        } else {
            // Create data with some patterns for better compression
            "ABCDEF".repeat(size.div_ceil(6)).as_bytes()[..size].to_vec()
        };

        let compressed = encoder.compress(&test_data).unwrap();

        if size == 0 {
            assert!(compressed.is_empty());
        } else {
            let decompressed = decoder.decompress(&compressed).unwrap();
            assert_eq!(test_data, decompressed);

            let stats = encoder.stats();
            println!(
                "  Ratio: {:.3}, Entropy: {:.3}",
                stats.compression_ratio, stats.entropy
            );
        }
    }
}

/// Performance benchmark for FSE
#[test]
#[cfg(feature = "zstd")]
fn bench_fse_performance() {
    use std::time::Instant;

    let mut encoder = FseEncoder::new(FseConfig::balanced()).unwrap();
    let mut decoder = FseDecoder::new();

    // Create test data with realistic patterns
    let test_data = "The quick brown fox jumps over the lazy dog. ".repeat(2000);
    let data = test_data.as_bytes();

    println!("FSE Performance Benchmark");
    println!("Data size: {} bytes", data.len());

    // Benchmark compression
    let start = Instant::now();
    let compressed = encoder.compress(data).unwrap();
    let compress_time = start.elapsed();

    let compress_speed = (data.len() as f64 / 1024.0 / 1024.0) / compress_time.as_secs_f64();

    // Benchmark decompression
    let start = Instant::now();
    let decompressed = decoder.decompress(&compressed).unwrap();
    let decompress_time = start.elapsed();

    let decompress_speed =
        (decompressed.len() as f64 / 1024.0 / 1024.0) / decompress_time.as_secs_f64();

    // Verify correctness
    assert_eq!(data, &decompressed[..]);

    let stats = encoder.stats();

    println!("Results:");
    println!("  Compression ratio: {:.3}", stats.compression_ratio);
    println!("  Entropy: {:.3} bits", stats.entropy);
    println!("  Efficiency: {:.3}", stats.efficiency);
    println!("  Compression speed: {:.2} MB/s", compress_speed);
    println!("  Decompression speed: {:.2} MB/s", decompress_speed);
    println!("  Compressed size: {} bytes", compressed.len());

    // Performance assertions
    assert!(compress_speed > 1.0); // Should compress at least 1 MB/s
    assert!(decompress_speed > 1.0); // Should decompress at least 1 MB/s
    assert!(stats.compression_ratio < 1.0); // Should achieve compression
    assert!(stats.efficiency > 0.5); // Should be reasonably efficient
}

/// Test FSE integration across multiple modules
#[test]
fn test_fse_cross_module_integration() {
    // Test that FSE can be used across different modules in the system

    // Test entropy module FSE
    let entropy_compressed = fse_compress(b"entropy module test").unwrap_or_default();

    // Test PA-Zip FSE integration
    let pa_zip_config = PaZipFseConfig::for_pa_zip();
    let pa_zip_compressed = apply_fse_compression(b"pa-zip module test", &pa_zip_config).unwrap();

    // Both should work without conflicts
    assert!(!entropy_compressed.is_empty() || cfg!(not(feature = "zstd")));
    assert!(!pa_zip_compressed.is_empty());

    // Test that configurations are compatible
    let entropy_config = FseConfig::balanced();
    let _pa_zip_config_default = PaZipFseConfig::default();

    assert!(entropy_config.validate().is_ok());

    // Verify FSE is properly exported
    #[cfg(feature = "zstd")]
    {
        // Test that FSE can be used directly
        let config = FseConfig::balanced();
        let encoder_result = FseEncoder::new(config);
        assert!(encoder_result.is_ok());
    }
}

// ============================================================================
// Strict lossless round-trip tests (plan.md Phase 3, C4)
//
// FSE is an entropy codec: decompress(compress(x)) == x must hold exactly for
// every input. These tests cover the failure modes of the frequency
// normalization and header pipeline that previously made the codec lossy.
// ============================================================================

/// Helper: assert exact round-trip through fresh encoder/decoder with a config.
fn assert_roundtrip_with_config(data: &[u8], config: FseConfig) {
    let mut encoder = FseEncoder::new(config.clone()).unwrap();
    let mut decoder = FseDecoder::with_config(config).unwrap();
    let compressed = encoder.compress(data).unwrap();
    let decompressed = decoder.decompress(&compressed).unwrap();
    assert_eq!(
        data,
        &decompressed[..],
        "FSE round-trip mismatch for input of {} bytes",
        data.len()
    );
}

fn assert_roundtrip(data: &[u8]) {
    assert_roundtrip_with_config(data, FseConfig::default());
}

#[test]
fn test_fse_strict_roundtrip_edge_sizes() {
    // Empty and tiny inputs (uncompressed path) plus the 99/100/101-byte
    // boundary where the codec switches from the uncompressed marker to the
    // real entropy-coded path.
    assert_roundtrip(b"");
    assert_roundtrip(b"a");
    assert_roundtrip(&[0xFFu8]); // input equal to the uncompressed marker byte
    assert_roundtrip(&[b'x'; 99]);
    assert_roundtrip(&[b'x'; 100]);
    assert_roundtrip(&[b'x'; 101]);
}

#[test]
fn test_fse_strict_roundtrip_all_same() {
    // Single-symbol input: normalized frequency == full table size.
    assert_roundtrip(&vec![0u8; 4096]);
    assert_roundtrip(&vec![255u8; 500]);
}

#[test]
fn test_fse_strict_roundtrip_three_symbol_uniform() {
    // 3 equal symbols: 4096/3 does not divide evenly, exercising the
    // remainder-distribution path of normalization.
    let data: Vec<u8> = b"abc".iter().cycle().take(999).copied().collect();
    assert_roundtrip(&data);
}

#[test]
fn test_fse_strict_roundtrip_adversarial_skew() {
    // Dominant symbol at a LOW byte index followed by many rare symbols at
    // higher indices. The old sequential `.min(remaining)` normalization
    // exhausted the frequency budget on the dominant symbol and assigned
    // rare symbols a normalized frequency of zero, corrupting the stream
    // with un-decodable escape markers.
    let mut data = vec![0u8; 4000];
    for b in 1..=150u8 {
        data.push(b);
    }
    assert_roundtrip(&data);

    // Same shape but dominant symbol at a HIGH index (reverse iteration order).
    let mut data = vec![255u8; 4000];
    for b in 1..=150u8 {
        data.push(b);
    }
    assert_roundtrip(&data);
}

#[test]
fn test_fse_strict_roundtrip_moderate_skew_entropy_path() {
    // Balanced-but-skewed distribution (entropy > 1 bit) that used to take the
    // entropy-weighted normalization path, which allocated by -p*log2(p)
    // instead of p and could zero out or misallocate symbols.
    let mut data = Vec::new();
    data.extend(std::iter::repeat_n(b'a', 700));
    data.extend(std::iter::repeat_n(b'b', 150));
    data.extend(std::iter::repeat_n(b'c', 150));
    for b in 0..=99u8 {
        data.push(b);
    }
    assert_roundtrip(&data);
}

#[test]
fn test_fse_strict_roundtrip_all_256_symbols() {
    // Every byte value present, uniform.
    let data: Vec<u8> = (0..=255u8).cycle().take(2048).collect();
    assert_roundtrip(&data);
    // Every byte value present, highly non-uniform (byte value == frequency).
    let mut data = Vec::new();
    for b in 0..=255u8 {
        data.extend(std::iter::repeat_n(b, b as usize + 1));
    }
    assert_roundtrip(&data);
}

#[test]
fn test_fse_strict_roundtrip_deterministic_random() {
    // Deterministic xorshift-generated pseudo-random data at several sizes.
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 32) as u8
    };
    for size in [128usize, 1000, 4096, 70_000] {
        let data: Vec<u8> = (0..size).map(|_| next()).collect();
        assert_roundtrip(&data);
    }
}

#[test]
fn test_fse_strict_roundtrip_config_presets() {
    let data: Vec<u8> = b"The quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(5000)
        .copied()
        .collect();
    assert_roundtrip_with_config(&data, FseConfig::fast_compression());
    assert_roundtrip_with_config(&data, FseConfig::balanced());
    assert_roundtrip_with_config(&data, FseConfig::high_compression());
    assert_roundtrip_with_config(&data, FseConfig::realtime());
}

#[test]
fn test_fse_strict_roundtrip_decoder_config_independent() {
    // A compressed stream carries its own table in the header, so any decoder
    // must reproduce it regardless of its local config. Previously the header
    // stored RAW frequencies and the decoder re-normalized them with its own
    // config, so an encoder/decoder config mismatch silently corrupted data.
    let mut data: Vec<u8> = b"mismatch test data with skewed frequencies! "
        .iter()
        .cycle()
        .take(3000)
        .copied()
        .collect();
    data.extend(std::iter::repeat_n(b'z', 2000));

    for enc_config in [
        FseConfig::fast_compression(),
        FseConfig::high_compression(),
        FseConfig::realtime(),
    ] {
        let mut encoder = FseEncoder::new(enc_config).unwrap();
        let compressed = encoder.compress(&data).unwrap();
        // Decode with the DEFAULT decoder (fse_decompress convenience path).
        let decompressed = fse_decompress(&compressed).unwrap();
        assert_eq!(data, decompressed, "decoder must not depend on its config");
    }
}

#[test]
fn test_fse_unseen_symbol_with_static_table_errors() {
    // With adaptive mode off, a reused table cannot encode a symbol that was
    // absent from the training data. That must be a hard error, not silent
    // stream corruption via escape bytes.
    let config = FseConfig {
        adaptive: false,
        ..FseConfig::default()
    };
    let mut encoder = FseEncoder::new(config).unwrap();
    let training: Vec<u8> = std::iter::repeat_n(b'a', 200).collect();
    encoder.compress(&training).unwrap();

    let unseen: Vec<u8> = std::iter::repeat_n(b'Z', 200).collect();
    assert!(
        encoder.compress(&unseen).is_err(),
        "encoding a symbol missing from the static table must fail loudly"
    );
}

mod fse_proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Arbitrary bytes must round-trip exactly.
        #[test]
        fn proptest_fse_roundtrip_arbitrary(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
            let mut encoder = FseEncoder::new(FseConfig::default()).unwrap();
            let mut decoder = FseDecoder::new();
            let compressed = encoder.compress(&data).unwrap();
            let decompressed = decoder.decompress(&compressed).unwrap();
            prop_assert_eq!(data, decompressed);
        }

        /// Skewed two-symbol distributions with varying dominance ratios.
        #[test]
        fn proptest_fse_roundtrip_skewed(
            dominant in any::<u8>(),
            rare in any::<u8>(),
            dominant_count in 100usize..5000,
            rare_count in 1usize..100,
        ) {
            let mut data = vec![dominant; dominant_count];
            data.extend(std::iter::repeat_n(rare, rare_count));
            let mut encoder = FseEncoder::new(FseConfig::default()).unwrap();
            let mut decoder = FseDecoder::new();
            let compressed = encoder.compress(&data).unwrap();
            let decompressed = decoder.decompress(&compressed).unwrap();
            prop_assert_eq!(data, decompressed);
        }
    }
}
