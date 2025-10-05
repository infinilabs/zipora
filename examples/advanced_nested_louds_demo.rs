//! Advanced ZiporaTrie Demonstration
//!
//! This example demonstrates the unified ZiporaTrie implementation
//! with advanced configuration strategies.

use zipora::{
    fsa::{ZiporaTrie, ZiporaTrieConfig, TrieStrategy, CompressionStrategy, StorageStrategy, RankSelectType},
    succinct::RankSelectInterleaved256,
    Result,
};

fn main() -> Result<()> {
    println!("🚀 Advanced ZiporaTrie Demonstration");
    println!("=====================================");

    // Create an advanced configuration with compressed sparse strategy
    let config = ZiporaTrieConfig {
        trie_strategy: TrieStrategy::CompressedSparse {
            sparse_threshold: 0.4,
            compression_level: 5,
            adaptive_sparse: true,
        },
        compression_strategy: CompressionStrategy::FragmentCompression {
            fragment_size: 8,
            frequency_threshold: 0.3,
            dictionary_size: 128,
        },
        storage_strategy: StorageStrategy::CacheOptimized {
            cache_line_size: 64,
            numa_aware: true,
            prefetch_enabled: true,
        },
        rank_select_type: RankSelectType::Interleaved256,
        enable_simd: true,
        enable_concurrency: false,
        cache_optimization: true,
    };

    println!("✅ Configuration:");
    println!("   - Trie strategy: CompressedSparse");
    println!("   - Compression: FragmentCompression");
    println!("   - Storage: CacheOptimized");
    println!("   - SIMD enabled: {}", config.enable_simd);
    println!("   - Cache optimization: {}", config.cache_optimization);

    // Create the trie with advanced configuration
    let mut trie: ZiporaTrie<RankSelectInterleaved256> = ZiporaTrie::with_config(config);

    // Test data with common patterns for compression
    let test_urls = vec![
        "https://www.example.com/api/v1/users",
        "https://www.example.com/api/v1/products",
        "https://www.example.com/api/v2/users",
        "https://api.github.com/repos/user/project",
        "https://api.github.com/repos/user/zipora",
        "http://files.download.com/archive.zip",
        "http://files.download.com/backup.zip",
        "ftp://ftp.server.com/path/to/file.txt",
        "ftp://ftp.server.com/path/to/data.csv",
    ];

    println!("\n🔧 Building Trie:");
    for url in &test_urls {
        trie.insert(url.as_bytes())?;
        println!("   + {}", url);
    }

    println!("\n📊 Trie Statistics:");
    let stats = trie.stats();
    println!("   - Total keys: {}", stats.num_keys);
    println!("   - Total states: {}", stats.num_states);
    println!("   - Memory usage: {} bytes", stats.memory_usage);
    println!("   - Bits per key: {:.2}", stats.bits_per_key);

    println!("\n🔍 Prefix Search:");
    let prefixes = ["https://www.example.com", "https://api.github.com", "ftp://"];

    for prefix in &prefixes {
        let matches: Vec<_> = trie.iter_prefix(prefix.as_bytes()).collect();
        println!("   '{}' -> {} matches:", prefix, matches.len());
        for m in matches {
            println!("      {}", String::from_utf8_lossy(&m));
        }
    }

    println!("\n✅ Demonstration Complete!");
    Ok(())
}