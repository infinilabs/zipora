//! Debug example for critical-bit trie implementation
//!
//! This example demonstrates the critical-bit trie functionality
//! and provides detailed debugging information.

use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, TrieStrategy, StorageStrategy, CompressionStrategy, RankSelectType, Trie};
use zipora::succinct::RankSelectInterleaved256;
use zipora::Result;

fn analyze_key_bits(key: &[u8]) {
    println!(
        "Key: {:?}",
        std::str::from_utf8(key).unwrap_or("<invalid utf8>")
    );
    print!("Bits: ");
    for &byte in key {
        print!("{:08b} ", byte);
    }
    println!();
    print!("Hex:  ");
    for &byte in key {
        print!("{:02x}       ", byte);
    }
    println!();
}

fn compare_keys_bitwise(key1: &[u8], key2: &[u8]) {
    println!(
        "Comparing {:?} vs {:?}:",
        std::str::from_utf8(key1).unwrap_or("<invalid>"),
        std::str::from_utf8(key2).unwrap_or("<invalid>")
    );

    let max_len = key1.len().max(key2.len());
    for i in 0..max_len {
        let byte1 = key1.get(i).copied().unwrap_or(0);
        let byte2 = key2.get(i).copied().unwrap_or(0);

        if byte1 != byte2 {
            println!(
                "  First difference at byte {}: {:02x} vs {:02x}",
                i, byte1, byte2
            );
            println!("  Bit pattern: {:08b} vs {:08b}", byte1, byte2);

            // Find first differing bit
            let diff = byte1 ^ byte2;
            for bit in 0..8 {
                if (diff >> bit) & 1 != 0 {
                    println!("  First differing bit: {} (from right)", bit);
                    break;
                }
            }
            break;
        }
    }
}

fn main() -> Result<()> {
    println!("🔍 Critical-Bit Trie Debug Session");
    println!("=====================================\n");

    // Create ZiporaTrie with Patricia strategy for critical-bit-like behavior
    let config = ZiporaTrieConfig {
        trie_strategy: TrieStrategy::Patricia {
            max_path_length: 64,
            compression_threshold: 4,
            adaptive_compression: true,
        },
        storage_strategy: StorageStrategy::Standard {
            initial_capacity: 256,
            growth_factor: 1.5,
        },
        compression_strategy: CompressionStrategy::None,
        rank_select_type: RankSelectType::Interleaved256,
        enable_simd: true,
        enable_concurrency: false,
        cache_optimization: false,
    };
    let mut trie: ZiporaTrie<RankSelectInterleaved256> = ZiporaTrie::with_config(config);

    println!("🌳 Building trie with test keys...");
    let keys = [b"cat".as_slice(), b"car".as_slice(), b"card".as_slice()];

    for key in &keys {
        println!("Inserting: {:?}", std::str::from_utf8(key).unwrap());
        trie.insert(key)?;
    }

    println!("\n🔍 Testing lookups...");
    for key in &keys {
        let found = trie.contains(key);
        println!(
            "Looking up {:?}: {}",
            std::str::from_utf8(key).unwrap(),
            found
        );
    }

    println!("\n🔬 DETAILED BIT ANALYSIS");
    println!("========================");

    for key in &keys {
        analyze_key_bits(key);
        println!();
    }

    println!("🔬 PAIRWISE COMPARISONS");
    println!("=======================");

    compare_keys_bitwise(b"cat", b"car");
    println!();
    compare_keys_bitwise(b"car", b"card");
    println!();
    compare_keys_bitwise(b"cat", b"card");

    println!("\n📊 Trie Statistics");
    println!("==================");
    println!("Total keys: {}", trie.len());

    println!("\n✅ Debug session completed");

    Ok(())
}
