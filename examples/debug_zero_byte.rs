//! Debug the zero byte issue in critical-bit trie

use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, TrieStrategy, StorageStrategy, CompressionStrategy, RankSelectType, Trie};
use zipora::succinct::RankSelectInterleaved256;
use zipora::Result;

fn analyze_key_bits(key: &[u8]) {
    println!(
        "Key: {:?}",
        std::str::from_utf8(key).unwrap_or("<invalid utf8>")
    );
    print!("Bytes: ");
    for &byte in key {
        print!("{:02x} ", byte);
    }
    println!();
    print!("Bits:  ");
    for &byte in key {
        print!("{:08b} ", byte);
    }
    println!();
    println!("Length: {} bytes", key.len());
}

fn compare_keys_bitwise(key1: &[u8], key2: &[u8]) {
    println!("Comparing keys bitwise:");
    println!(
        "Key1: {:?} (len={})",
        std::str::from_utf8(key1).unwrap_or("<invalid>"),
        key1.len()
    );
    println!(
        "Key2: {:?} (len={})",
        std::str::from_utf8(key2).unwrap_or("<invalid>"),
        key2.len()
    );

    let max_len = key1.len().max(key2.len());
    let min_len = key1.len().min(key2.len());

    for i in 0..max_len {
        let byte1 = key1.get(i).copied().unwrap_or(0);
        let byte2 = key2.get(i).copied().unwrap_or(0);

        if i < min_len {
            println!(
                "  Byte {}: {:02x} vs {:02x} -> {:08b} vs {:08b}",
                i, byte1, byte2, byte1, byte2
            );
            if byte1 != byte2 {
                println!("    ^ First difference at byte {}", i);
                let diff = byte1 ^ byte2;
                for bit in 0..8 {
                    if (diff >> bit) & 1 != 0 {
                        println!("    ^ First differing bit: {} (from right)", bit);
                        break;
                    }
                }
                return;
            }
        } else {
            println!(
                "  Byte {}: {} vs {} (length difference)",
                i,
                if i < key1.len() {
                    format!("{:02x}", byte1)
                } else {
                    "N/A".to_string()
                },
                if i < key2.len() {
                    format!("{:02x}", byte2)
                } else {
                    "N/A".to_string()
                }
            );
        }
    }

    if key1.len() != key2.len() {
        println!("  Keys differ in length: {} vs {}", key1.len(), key2.len());
    } else {
        println!("  Keys are identical");
    }
}

fn main() -> Result<()> {
    println!("🔍 Zero Byte Debug Session");
    println!("=========================\n");

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

    let key1 = b"test";
    let key2 = b"test\x00"; // Same as key1 but with a zero byte appended

    println!("Testing keys with potential zero byte issue:");
    println!("============================================");

    analyze_key_bits(key1);
    println!();
    analyze_key_bits(key2);

    println!("\nComparing keys:");
    compare_keys_bitwise(key1, key2);

    println!("\n🌳 Inserting keys into trie:");
    println!("=============================");

    println!("Inserting key1: {:?}", std::str::from_utf8(key1).unwrap());
    trie.insert(key1)?;
    println!("  -> Trie now has {} keys", trie.len());

    println!(
        "Inserting key2: {:?} (with zero byte)",
        std::str::from_utf8(&key2[..key2.len() - 1]).unwrap()
    );
    trie.insert(key2)?;
    println!("  -> Trie now has {} keys", trie.len());

    println!("\n🔍 Looking up keys:");
    println!("===================");

    let found1 = trie.contains(key1);
    println!(
        "Looking up key1 {:?}: {}",
        std::str::from_utf8(key1).unwrap(),
        found1
    );

    let found2 = trie.contains(key2);
    println!(
        "Looking up key2 {:?} (with zero): {}",
        std::str::from_utf8(&key2[..key2.len() - 1]).unwrap(),
        found2
    );

    println!("\n📊 Final trie state:");
    println!("====================");
    println!("Total keys stored: {}", trie.len());
    println!("Expected: 2 (if zero byte handling works correctly)");

    // Test some edge cases
    println!("\n🧪 Additional zero byte tests:");
    println!("==============================");

    let test_keys = [
        b"".as_slice(),         // empty
        b"\x00".as_slice(),     // single zero
        b"a\x00".as_slice(),    // zero in middle
        b"\x00a".as_slice(),    // zero at start
        b"ab\x00cd".as_slice(), // zero in middle
    ];

    for key in &test_keys {
        let result = trie.insert(key);
        match result {
            Ok(_) => println!("Successfully inserted key with {} bytes", key.len()),
            Err(e) => println!("Failed to insert key: {}", e),
        }
    }

    println!("\nFinal trie size: {}", trie.len());

    Ok(())
}
