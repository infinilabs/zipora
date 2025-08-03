use std::time::Instant;
use zipora::{DictionaryBuilder, DictionaryCompressor, OptimizedDictionaryCompressor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Dictionary Compression Performance Comparison Demo");
    println!("{}", "=".repeat(60));

    // Create test data with various characteristics
    let test_cases = vec![
        ("Short Repeated Pattern", generate_repeated_data(b"hello", 100)),
        ("Medium Repeated Pattern", generate_repeated_data(b"this is a test pattern", 200)),
        ("Long Repeated Pattern", generate_repeated_data(b"this is a very long pattern that should compress very well when repeated many times", 100)),
        ("Biased Data", generate_biased_data(5000)),
        ("Mixed Data", generate_mixed_data(10000)),
    ];

    for (test_name, data) in test_cases {
        println!("\n{}", test_name);
        println!("{}", "-".repeat(test_name.len()));
        println!("Data size: {} bytes", data.len());

        // Test original implementation
        let start = Instant::now();
        let builder = DictionaryBuilder::new();
        let dict = builder.build(&data);
        let original_compressor = DictionaryCompressor::new(dict);
        let original_compressed = original_compressor.compress(&data)?;
        let original_time = start.elapsed();

        // Test optimized implementation  
        let start = Instant::now();
        let optimized_compressor = OptimizedDictionaryCompressor::new(&data)?;
        let optimized_compressed = optimized_compressor.compress(&data)?;
        let optimized_time = start.elapsed();

        // Calculate performance metrics
        let original_ratio = original_compressed.len() as f64 / data.len() as f64;
        let optimized_ratio = optimized_compressed.len() as f64 / data.len() as f64;
        let speedup = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

        println!("Original:  {:8.2}ms, ratio: {:.3}", original_time.as_secs_f64() * 1000.0, original_ratio);
        println!("Optimized: {:8.2}ms, ratio: {:.3}", optimized_time.as_secs_f64() * 1000.0, optimized_ratio);
        println!("Speedup:   {:.1}x faster", speedup);

        // Verify correctness
        let original_decompressed = original_compressor.decompress(&original_compressed)?;
        let optimized_decompressed = optimized_compressor.decompress(&optimized_compressed)?;
        
        assert_eq!(data, original_decompressed, "Original decompression failed");
        assert_eq!(data, optimized_decompressed, "Optimized decompression failed");
        println!("✓ Correctness verified");
    }

    println!("\n{}", "=".repeat(60));
    println!("Performance Summary:");
    println!("• The optimized implementation uses suffix arrays for O(log n) pattern search");
    println!("• Bloom filters provide quick rejection of non-existent patterns");
    println!("• Rolling hash reduces expensive byte-by-byte comparisons");
    println!("• Target: Reduce 7,556x performance gap to <10x slower than optimal");

    Ok(())
}

fn generate_repeated_data(pattern: &[u8], repetitions: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for _ in 0..repetitions {
        data.extend_from_slice(pattern);
    }
    data
}

fn generate_biased_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let byte = match i % 100 {
            0..=79 => b'A',  // 80% A's
            80..=94 => b'B', // 15% B's  
            _ => b'C' + (i % 3) as u8, // 5% C, D, E
        };
        data.push(byte);
    }
    data
}

fn generate_mixed_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let patterns = [
        b"pattern1".as_slice(),
        b"another_pattern".as_slice(), 
        b"third_pattern_longer".as_slice(),
        b"short".as_slice(),
    ];
    
    for i in 0..size {
        let pattern = patterns[i % patterns.len()];
        data.push(pattern[i % pattern.len()]);
    }
    data
}