use zipora::entropy::*;
use zipora::error::Result;
use std::time::Instant;

/// Comprehensive performance test suite for entropy algorithms
/// Only runs in release mode to ensure accurate performance measurements

#[cfg(not(debug_assertions))]
mod performance_tests {
    use super::*;

    fn generate_test_datasets() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            // Low entropy data
            ("low_entropy_1kb", vec![42u8; 1024]),
            ("low_entropy_64kb", vec![123u8; 65536]),
            
            // Medium entropy data
            ("medium_entropy_1kb", (0..=255u8).cycle().take(1024).collect()),
            ("medium_entropy_64kb", (0..=255u8).cycle().take(65536).collect()),
            
            // High entropy data (pseudo-random)
            ("high_entropy_1kb", {
                let mut data = Vec::new();
                for i in 0..1024 {
                    data.push(((i * 31 + 17) % 256) as u8);
                }
                data
            }),
            ("high_entropy_64kb", {
                let mut data = Vec::new();
                for i in 0..65536 {
                    data.push(((i * 31 + 17) % 256) as u8);
                }
                data
            }),
            
            // Textual data (realistic compression scenario)
            ("text_data_1kb", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(19).into_bytes()),
            ("text_data_64kb", "The quick brown fox jumps over the lazy dog. ".repeat(1456).into_bytes()),
            
            // Binary patterns
            ("binary_pattern_1kb", {
                let mut data = Vec::new();
                for i in 0..256 {
                    data.extend_from_slice(&[i as u8, !i as u8, i as u8, !i as u8]);
                }
                data
            }),
            ("binary_pattern_64kb", {
                let mut data = Vec::new();
                for _ in 0..64 {
                    for i in 0..256 {
                        data.extend_from_slice(&[i as u8, !i as u8, i as u8, !i as u8]);
                    }
                }
                data
            }),
        ]
    }

    #[test]
    fn test_huffman_performance_comprehensive() -> Result<()> {
        println!("\n=== Huffman Algorithms Performance Test ===");
        
        let datasets = generate_test_datasets();
        
        for (name, data) in datasets {
            println!("\nDataset: {} ({} bytes)", name, data.len());
            
            // Basic Huffman
            let start = Instant::now();
            let basic_encoder = HuffmanEncoder::new(&data)?;
            let basic_compressed = basic_encoder.encode(&data)?;
            let basic_time = start.elapsed();
            let basic_ratio = data.len() as f64 / basic_compressed.len() as f64;
            
            println!("  Basic Huffman: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                     basic_ratio, basic_time.as_millis(), 
                     (data.len() as f64 / 1024.0 / 1024.0) / basic_time.as_secs_f64());
            
            // Contextual Huffman Order-1
            let start = Instant::now();
            let ctx1_encoder = ContextualHuffmanEncoder::new(&data, HuffmanOrder::Order1)?;
            let ctx1_compressed = ctx1_encoder.encode(&data)?;
            let ctx1_time = start.elapsed();
            let ctx1_ratio = data.len() as f64 / ctx1_compressed.len() as f64;
            
            println!("  Order-1 Huffman: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                     ctx1_ratio, ctx1_time.as_millis(),
                     (data.len() as f64 / 1024.0 / 1024.0) / ctx1_time.as_secs_f64());
            
            // Contextual Huffman Order-2
            if data.len() >= 3 {
                let start = Instant::now();
                let ctx2_encoder = ContextualHuffmanEncoder::new(&data, HuffmanOrder::Order2)?;
                let ctx2_compressed = ctx2_encoder.encode(&data)?;
                let ctx2_time = start.elapsed();
                let ctx2_ratio = data.len() as f64 / ctx2_compressed.len() as f64;
                
                println!("  Order-2 Huffman: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                         ctx2_ratio, ctx2_time.as_millis(),
                         (data.len() as f64 / 1024.0 / 1024.0) / ctx2_time.as_secs_f64());
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_rans_performance_comprehensive() -> Result<()> {
        println!("\n=== rANS Algorithms Performance Test ===");
        
        let datasets = generate_test_datasets();
        
        for (name, data) in datasets {
            println!("\nDataset: {} ({} bytes)", name, data.len());
            
            // Calculate frequencies
            let mut frequencies = [1u32; 256];
            for &byte in &data {
                frequencies[byte as usize] += 1;
            }
            
            // Enhanced rANS 64-bit single stream
            let start = Instant::now();
            let rans_encoder = Rans64Encoder::<ParallelX1>::new(&frequencies)?;
            let rans_compressed = rans_encoder.encode(&data)?;
            let rans_time = start.elapsed();
            let rans_ratio = data.len() as f64 / rans_compressed.len() as f64;
            
            println!("  rANS 64-bit: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                     rans_ratio, rans_time.as_millis(),
                     (data.len() as f64 / 1024.0 / 1024.0) / rans_time.as_secs_f64());
            
            // Parallel rANS x2
            let start = Instant::now();
            let rans_x2_encoder = Rans64Encoder::<ParallelX2>::new(&frequencies)?;
            let rans_x2_compressed = rans_x2_encoder.encode(&data)?;
            let rans_x2_time = start.elapsed();
            let rans_x2_ratio = data.len() as f64 / rans_x2_compressed.len() as f64;
            
            println!("  rANS x2: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                     rans_x2_ratio, rans_x2_time.as_millis(),
                     (data.len() as f64 / 1024.0 / 1024.0) / rans_x2_time.as_secs_f64());
            
            // Parallel rANS x4
            let start = Instant::now();
            let rans_x4_encoder = Rans64Encoder::<ParallelX4>::new(&frequencies)?;
            let rans_x4_compressed = rans_x4_encoder.encode(&data)?;
            let rans_x4_time = start.elapsed();
            let rans_x4_ratio = data.len() as f64 / rans_x4_compressed.len() as f64;
            
            println!("  rANS x4: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                     rans_x4_ratio, rans_x4_time.as_millis(),
                     (data.len() as f64 / 1024.0 / 1024.0) / rans_x4_time.as_secs_f64());
        }
        
        Ok(())
    }

    #[test]
    fn test_fse_performance_comprehensive() -> Result<()> {
        println!("\n=== FSE Algorithms Performance Test ===");
        
        let datasets = generate_test_datasets();
        
        for (name, data) in datasets {
            println!("\nDataset: {} ({} bytes)", name, data.len());
            
            // Enhanced FSE Default
            let start = Instant::now();
            let mut fse_encoder = FseEncoder::new(FseConfig::default())?;
            let fse_compressed = fse_encoder.compress(&data)?;
            let fse_time = start.elapsed();
            let fse_ratio = data.len() as f64 / fse_compressed.len() as f64;
            
            println!("  FSE Default: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                     fse_ratio, fse_time.as_millis(),
                     (data.len() as f64 / 1024.0 / 1024.0) / fse_time.as_secs_f64());
            
            // Enhanced FSE Fast
            let start = Instant::now();
            let mut fse_fast_encoder = FseEncoder::new(FseConfig::fast_compression())?;
            let fse_fast_compressed = fse_fast_encoder.compress(&data)?;
            let fse_fast_time = start.elapsed();
            let fse_fast_ratio = data.len() as f64 / fse_fast_compressed.len() as f64;
            
            println!("  FSE Fast: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                     fse_fast_ratio, fse_fast_time.as_millis(),
                     (data.len() as f64 / 1024.0 / 1024.0) / fse_fast_time.as_secs_f64());
            
            // Enhanced FSE High Compression
            let start = Instant::now();
            let mut fse_high_encoder = FseEncoder::new(FseConfig::high_compression())?;
            let fse_high_compressed = fse_high_encoder.compress(&data)?;
            let fse_high_time = start.elapsed();
            let fse_high_ratio = data.len() as f64 / fse_high_compressed.len() as f64;
            
            println!("  FSE High: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                     fse_high_ratio, fse_high_time.as_millis(),
                     (data.len() as f64 / 1024.0 / 1024.0) / fse_high_time.as_secs_f64());
        }
        
        Ok(())
    }

    #[test]
    fn test_parallel_encoding_performance() -> Result<()> {
        println!("\n=== Parallel Encoding Performance Test ===");
        
        let large_datasets = vec![
            ("large_low_entropy", vec![77u8; 262144]),
            ("large_medium_entropy", (0..=255u8).cycle().take(262144).collect()),
            ("large_high_entropy", {
                let mut data = Vec::new();
                for i in 0..262144 {
                    data.push(((i * 37 + 23) % 256) as u8);
                }
                data
            }),
        ];
        
        for (name, data) in large_datasets {
            println!("\nDataset: {} ({} KB)", name, data.len() / 1024);
            
            // Test different parallel configurations
            let configs = vec![
                ("x2", ParallelConfig::low_latency()),
                ("x4", ParallelConfig::balanced()),
                ("x8", ParallelConfig::high_throughput()),
            ];
            
            for (config_name, config) in configs {
                let start = Instant::now();
                let mut encoder = ParallelHuffmanEncoder::<ParallelX4Variant>::new(config)?;
                encoder.train(&data)?;
                let compressed = encoder.encode(&data)?;
                let elapsed = start.elapsed();
                let ratio = data.len() as f64 / compressed.len() as f64;
                
                println!("  Parallel {} Config: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                         config_name, ratio, elapsed.as_millis(),
                         (data.len() as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64());
            }
            
            // Test adaptive encoding
            let start = Instant::now();
            let mut adaptive_encoder = AdaptiveParallelEncoder::new()?;
            let adaptive_compressed = adaptive_encoder.encode_adaptive(&data)?;
            let adaptive_elapsed = start.elapsed();
            let adaptive_ratio = data.len() as f64 / adaptive_compressed.len() as f64;
            
            println!("  Adaptive Parallel: {:.2}x compression, {:.2}ms, {:.2} MB/s", 
                     adaptive_ratio, adaptive_elapsed.as_millis(),
                     (data.len() as f64 / 1024.0 / 1024.0) / adaptive_elapsed.as_secs_f64());
        }
        
        Ok(())
    }

    #[test]
    fn test_bit_operations_performance() -> Result<()> {
        println!("\n=== Bit Operations Performance Test ===");
        
        let bit_ops = BitOps::new();
        let entropy_bit_ops = EntropyBitOps::new();
        let test_values: Vec<u64> = (0..10000).map(|i| i * 0x123456789ABCDEF).collect();
        
        // Test popcount performance
        let start = Instant::now();
        let mut sum = 0u64;
        for &value in &test_values {
            sum += bit_ops.popcount64(value) as u64;
        }
        let popcount_time = start.elapsed();
        println!("  Popcount (10k values): {} bits total, {:.2}ms, {:.2} M ops/s", 
                 sum, popcount_time.as_millis(),
                 (test_values.len() as f64 / 1_000_000.0) / popcount_time.as_secs_f64());
        
        // Test reverse bits performance (using 32-bit since 64-bit is not available)
        let start = Instant::now();
        for &value in &test_values {
            let _ = entropy_bit_ops.reverse_bits32(value as u32);
        }
        let reverse_time = start.elapsed();
        println!("  Reverse bits 32 (10k values): {:.2}ms, {:.2} M ops/s", 
                 reverse_time.as_millis(),
                 (test_values.len() as f64 / 1_000_000.0) / reverse_time.as_secs_f64());
        
        // Test BMI2 operations if available
        if bit_ops.features().has_bmi2 {
            println!("  BMI2 Instructions Available");
            
            let start = Instant::now();
            for &value in &test_values {
                let _ = bit_ops.parallel_deposit64(value, 0xAAAAAAAAAAAAAAAA);
            }
            let pdep_time = start.elapsed();
            println!("  PDEP (10k values): {:.2}ms, {:.2} M ops/s", 
                     pdep_time.as_millis(),
                     (test_values.len() as f64 / 1_000_000.0) / pdep_time.as_secs_f64());
            
            let start = Instant::now();
            for &value in &test_values {
                let _ = bit_ops.parallel_extract64(value, 0xAAAAAAAAAAAAAAAA);
            }
            let pext_time = start.elapsed();
            println!("  PEXT (10k values): {:.2}ms, {:.2} M ops/s", 
                     pext_time.as_millis(),
                     (test_values.len() as f64 / 1_000_000.0) / pext_time.as_secs_f64());
        } else {
            println!("  BMI2 Instructions Not Available");
        }
        
        Ok(())
    }

    #[test]
    fn test_entropy_context_performance() -> Result<()> {
        println!("\n=== Entropy Context Performance Test ===");
        
        let context = EntropyContext::new();
        
        let buffer_sizes = vec![1024, 8192, 65536, 262144];
        
        for &size in &buffer_sizes {
            // Test buffer allocation performance
            let start = Instant::now();
            for _ in 0..1000 {
                let _buffer = context.alloc(size)?;
            }
            let alloc_time = start.elapsed();
            
            println!("  Buffer allocation ({} bytes, 1k times): {:.2}ms, {:.2} M allocs/s", 
                     size, alloc_time.as_millis(),
                     (1000.0 / 1_000_000.0) / alloc_time.as_secs_f64());
            
            // Test zeroed buffer performance
            let start = Instant::now();
            for _ in 0..1000 {
                let _buffer = context.alloc_zeroed(size)?;
            }
            let temp_time = start.elapsed();
            
            println!("  Zeroed buffer allocation ({} bytes, 1k times): {:.2}ms, {:.2} M allocs/s", 
                     size, temp_time.as_millis(),
                     (1000.0 / 1_000_000.0) / temp_time.as_secs_f64());
        }
        
        let stats = context.stats()?;
        println!("  Context Stats: {} cached buffers, {} total capacity, {} max capacity", 
                 stats.cached_buffers, stats.total_capacity, stats.max_capacity);
        
        Ok(())
    }

    #[test]
    fn test_algorithm_comparison_performance() -> Result<()> {
        println!("\n=== Algorithm Comparison Performance Test ===");
        
        let test_data = "This is a comprehensive test of various entropy encoding algorithms. ".repeat(1000);
        let data = test_data.as_bytes();
        
        println!("Test data: {} bytes", data.len());
        
        // Huffman
        let start = Instant::now();
        let huffman_encoder = HuffmanEncoder::new(data)?;
        let huffman_compressed = huffman_encoder.encode(data)?;
        let huffman_time = start.elapsed();
        let huffman_ratio = data.len() as f64 / huffman_compressed.len() as f64;
        
        // rANS
        let mut frequencies = [1u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }
        let start = Instant::now();
        let rans_encoder = Rans64Encoder::<ParallelX1>::new(&frequencies)?;
        let rans_compressed = rans_encoder.encode(data)?;
        let rans_time = start.elapsed();
        let rans_ratio = data.len() as f64 / rans_compressed.len() as f64;
        
        // FSE
        let start = Instant::now();
        let mut fse_encoder = FseEncoder::new(FseConfig::default())?;
        let fse_compressed = fse_encoder.compress(data)?;
        let fse_time = start.elapsed();
        let fse_ratio = data.len() as f64 / fse_compressed.len() as f64;
        
        // Adaptive Parallel
        let start = Instant::now();
        let mut adaptive_encoder = AdaptiveParallelEncoder::new()?;
        let adaptive_compressed = adaptive_encoder.encode_adaptive(data)?;
        let adaptive_time = start.elapsed();
        let adaptive_ratio = data.len() as f64 / adaptive_compressed.len() as f64;
        
        println!("\nComparison Results:");
        println!("Algorithm        | Ratio | Time (ms) | Speed (MB/s)");
        println!("-----------------|-------|-----------|-------------");
        println!("Huffman          | {:.2}x  | {:7.2} | {:10.2}", 
                 huffman_ratio, huffman_time.as_millis(),
                 (data.len() as f64 / 1024.0 / 1024.0) / huffman_time.as_secs_f64());
        println!("rANS 64          | {:.2}x  | {:7.2} | {:10.2}", 
                 rans_ratio, rans_time.as_millis(),
                 (data.len() as f64 / 1024.0 / 1024.0) / rans_time.as_secs_f64());
        println!("Enhanced FSE     | {:.2}x  | {:7.2} | {:10.2}", 
                 fse_ratio, fse_time.as_millis(),
                 (data.len() as f64 / 1024.0 / 1024.0) / fse_time.as_secs_f64());
        println!("Adaptive Parallel| {:.2}x  | {:7.2} | {:10.2}", 
                 adaptive_ratio, adaptive_time.as_millis(),
                 (data.len() as f64 / 1024.0 / 1024.0) / adaptive_time.as_secs_f64());
        
        Ok(())
    }
}

// Stub tests for debug builds
#[cfg(debug_assertions)]
mod debug_build_stubs {
    use super::*;
    
    #[test]
    fn debug_build_notice() {
        println!("Performance tests are only available in release builds.");
        println!("Run with: cargo test --release entropy_performance");
    }
}