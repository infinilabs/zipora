//! Advanced NestedLoudsTrie Demonstration
//!
//! This example demonstrates the enhanced NestedLoudsTrie implementation
//! with topling-zip-style advanced nesting strategies.

use zipora::{
    fsa::nested_louds_trie::{NestedLoudsTrie, NestingConfig},
    succinct::RankSelectSimple,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced NestedLoudsTrie Demonstration");
    println!("==========================================");
    
    // Create an advanced configuration with mixed storage strategies
    let config = NestingConfig::builder()
        .max_levels(4)
        .fragment_compression_ratio(0.3)
        .enable_mixed_storage(true)
        .nest_scale_factor(1.2)
        .fragment_delimiters(vec![b'/', b'.', b'-', b'_'])
        .min_fragment_refs(2)
        .min_fragment_size(3)
        .max_fragment_size(20)
        .cache_optimization(true)
        .build()?;

    println!("✅ Configuration:");
    println!("   - Max levels: {}", config.max_levels);
    println!("   - Mixed storage: {}", config.enable_mixed_storage);
    println!("   - Nest scale factor: {}", config.nest_scale_factor);
    println!("   - Fragment delimiters: {:?}", String::from_utf8_lossy(&config.fragment_delimiters));

    // Create the trie with advanced configuration
    let mut trie = NestedLoudsTrie::<RankSelectSimple>::with_config(config)?;

    // Test data with common patterns for fragment detection
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

    println!("\n📝 Inserting {} URLs with common patterns...", test_urls.len());
    
    for url in &test_urls {
        trie.insert(url.as_bytes())?;
    }

    println!("✅ Insertion complete!");
    
    // Demonstrate advanced features
    println!("\n📊 Advanced Compression Metrics:");
    let metrics = trie.compression_metrics();
    println!("   - Original size: {} bytes", metrics.original_size);
    println!("   - Compressed size: {} bytes", metrics.compressed_size);
    println!("   - Efficiency ratio: {:.3}", metrics.efficiency_ratio);
    println!("   - Nest scale: {:.3}", metrics.nest_scale);
    println!("   - Levels processed: {}", metrics.level_breakdown.len());

    // Fragment analysis
    println!("\n🧩 Fragment Analysis:");
    let fragment_stats = trie.fragment_analyzer_stats();
    println!("   - Detected fragments: {}", fragment_stats.len());
    
    // Show some interesting fragments
    if !fragment_stats.is_empty() {
        println!("   - Sample fragments:");
        for (fragment, count) in fragment_stats.iter().take(5) {
            let fragment_str = String::from_utf8_lossy(fragment);
            println!("     '{}' (used {} times)", fragment_str, count);
        }
    }

    // Memory efficiency
    println!("\n💾 Memory Usage:");
    println!("   - Total memory: {} bytes", trie.total_memory_usage());
    println!("   - Keys stored: {}", trie.len());
    println!("   - Bytes per key: {:.1}", 
        trie.total_memory_usage() as f64 / trie.len() as f64);

    // Performance statistics
    let perf_stats = trie.performance_stats();
    println!("   - Fragment compression count: {}", perf_stats.fragment_stats.fragment_count);
    println!("   - Active levels: {}", trie.active_levels());

    // Verify functionality
    println!("\n🔍 Functionality Verification:");
    let mut found_count = 0;
    for url in &test_urls {
        if trie.contains(url.as_bytes()) {
            found_count += 1;
        }
    }
    println!("   - URLs found: {}/{}", found_count, test_urls.len());
    
    // Test prefix operations
    println!("\n🔎 Prefix Operations:");
    let https_count: Vec<_> = trie.iter_prefix(b"https://").collect();
    let api_count: Vec<_> = trie.iter_prefix(b"https://api").collect();
    println!("   - URLs starting with 'https://': {}", https_count.len());
    println!("   - URLs starting with 'https://api': {}", api_count.len());

    // Demonstrate termination algorithm
    println!("\n⚡ Termination Algorithm:");
    println!("   - Compression efficiency: {:.3}", trie.compression_efficiency());
    println!("   - Current nest scale: {:.3}", trie.current_nest_scale());

    println!("\n🎉 Advanced NestedLoudsTrie demonstration complete!");
    println!("    The implementation successfully provides:");
    println!("    ✓ Mixed storage strategies (core + nested)");
    println!("    ✓ Smart termination based on compression efficiency");
    println!("    ✓ Recursive nesting with fragment detection");
    println!("    ✓ Delimiter-aware fragment analysis");
    println!("    ✓ Real compression metrics and space savings");

    Ok(())
}