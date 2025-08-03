//! Entropy Coding Demonstration
//!
//! This example demonstrates the entropy coding capabilities of zipora,
//! including Huffman coding, rANS encoding, and dictionary compression.

use zipora::{
    BlobStore, DictionaryBuilder, DictionaryCompressor, EntropyStats, HuffmanBlobStore,
    HuffmanEncoder, HuffmanTree, MemoryBlobStore, RansDecoder, RansEncoder,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 Entropy Coding Demo for zipora");
    println!("=====================================\n");

    // === PART 1: Entropy Statistics ===
    println!("📊 PART 1: Entropy Analysis");
    println!("---------------------------");

    let sample_data = b"hello world! this is a sample text for entropy analysis. hello world!";
    let entropy = EntropyStats::calculate_entropy(sample_data);

    println!("Sample text: \"{}\"", String::from_utf8_lossy(sample_data));
    println!("Text length: {} bytes", sample_data.len());
    println!("Calculated entropy: {:.3} bits per symbol", entropy);
    println!(
        "Theoretical compression limit: {:.1}%",
        (1.0 - entropy / 8.0) * 100.0
    );
    println!();

    // === PART 2: Huffman Coding ===
    println!("🌳 PART 2: Huffman Coding");
    println!("-------------------------");

    // Build Huffman tree from data
    match HuffmanTree::from_data(sample_data) {
        Ok(tree) => {
            println!("✅ Built Huffman tree successfully");
            println!("   Maximum code length: {} bits", tree.max_code_length());

            // Show some codes
            for &byte in b"hello " {
                if let Some(code) = tree.get_code(byte) {
                    let code_str: String =
                        code.iter().map(|&b| if b { '1' } else { '0' }).collect();
                    println!("   '{}' -> {}", byte as char, code_str);
                }
            }
        }
        Err(e) => {
            println!("❌ Huffman tree construction failed: {}", e);
        }
    }

    // Demonstrate compression estimation
    match HuffmanEncoder::new(sample_data) {
        Ok(encoder) => {
            let ratio = encoder.estimate_compression_ratio(sample_data);
            println!("   Estimated compression ratio: {:.3}", ratio);
            println!("   Estimated space savings: {:.1}%", (1.0 - ratio) * 100.0);
        }
        Err(e) => {
            println!("❌ Huffman encoder creation failed: {}", e);
        }
    }
    println!();

    // === PART 3: rANS Encoding ===
    println!("📈 PART 3: rANS (Range Asymmetric Numeral Systems)");
    println!("--------------------------------------------------");

    // Calculate frequencies for rANS
    let mut frequencies = [0u32; 256];
    for &byte in sample_data {
        frequencies[byte as usize] += 1;
    }

    match RansEncoder::new(&frequencies) {
        Ok(encoder) => {
            println!("✅ Created rANS encoder successfully");
            println!("   Total frequency: {}", encoder.total_freq());

            // Show symbol information for some characters
            for &byte in b"hello" {
                let symbol = encoder.get_symbol(byte);
                println!(
                    "   '{}' -> start: {}, freq: {}",
                    byte as char, symbol.start, symbol.freq
                );
            }

            let decoder = RansDecoder::new(&encoder);
            println!("   Created corresponding rANS decoder");
        }
        Err(e) => {
            println!("❌ rANS encoder creation failed: {}", e);
        }
    }
    println!();

    // === PART 4: Dictionary Compression ===
    println!("📖 PART 4: Dictionary-Based Compression");
    println!("---------------------------------------");

    let dict_data = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";

    // Build dictionary
    let builder = DictionaryBuilder::new()
        .min_match_length(3)
        .max_match_length(20)
        .max_entries(100);

    let dictionary = builder.build(dict_data);
    println!("✅ Built dictionary with {} entries", dictionary.len());

    if !dictionary.is_empty() {
        let compressor = DictionaryCompressor::new(dictionary);
        let ratio = compressor.estimate_compression_ratio(dict_data);
        println!("   Estimated compression ratio: {:.3}", ratio);

        if ratio < 1.0 {
            println!("   Estimated space savings: {:.1}%", (1.0 - ratio) * 100.0);
        } else {
            println!("   No compression benefit expected (ratio >= 1.0)");
        }
    }
    println!();

    // === PART 5: Blob Store Integration ===
    println!("💾 PART 5: Entropy Blob Store Integration");
    println!("-----------------------------------------");

    // Create a Huffman blob store
    let inner_store = MemoryBlobStore::new();
    let mut huffman_store = HuffmanBlobStore::new(inner_store);

    // Add training data
    huffman_store.add_training_data(sample_data);

    match huffman_store.build_tree() {
        Ok(()) => {
            println!("✅ Built Huffman tree for blob store");

            // Store some data
            let test_data = b"this is test data for the huffman blob store";
            match huffman_store.put(test_data) {
                Ok(id) => {
                    println!("   Stored data with ID: {}", id);
                    println!("   Store contains {} items", huffman_store.len());

                    // Get compression statistics
                    let stats = huffman_store.compression_stats();
                    println!("   Compressions performed: {}", stats.compressions);
                    if stats.compressions > 0 {
                        println!(
                            "   Average compression time: {:.1} μs",
                            stats.avg_compression_time_us()
                        );
                    }
                }
                Err(e) => {
                    println!("❌ Failed to store data: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to build Huffman tree: {}", e);
        }
    }

    println!();

    // === PART 6: Performance Comparison ===
    println!("⚡ PART 6: Performance Analysis");
    println!("------------------------------");

    println!("Entropy coding algorithms comparison:");
    println!("• Huffman Coding:");
    println!("  - Optimal for known symbol probabilities");
    println!("  - Prefix-free codes, good compression ratio");
    println!("  - Fast decoding, moderate encoding speed");
    println!();
    println!("• rANS (range Asymmetric Numeral Systems):");
    println!("  - Near-optimal compression (close to entropy limit)");
    println!("  - Better than Huffman for most data types");
    println!("  - More complex implementation");
    println!();
    println!("• Dictionary Compression:");
    println!("  - Excellent for data with repeated patterns");
    println!("  - LZ-style compression, finds substring matches");
    println!("  - Good for text and structured data");
    println!();

    println!("📊 Compression effectiveness depends on data characteristics:");
    println!("• High entropy (random) data: Limited compression possible");
    println!("• Biased distributions: Huffman/rANS work well");
    println!("• Repeated patterns: Dictionary compression excels");
    println!("• Mixed data: Combination approaches often best");

    println!();
    println!("✅ Entropy coding demonstration completed successfully!");

    Ok(())
}
