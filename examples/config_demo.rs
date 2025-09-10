//! Configuration demonstration example.
//! 
//! This example shows how to use the rich configuration APIs in Zipora.

use zipora::config::{
    NestLoudsTrieConfig, BlobStoreConfig, MemoryConfig, CompressionConfig, CacheConfig, SIMDConfig,
    Config,
};
use zipora::error::Result;

fn main() -> Result<()> {
    println!("Zipora Configuration API Demo");
    println!("==============================");

    // Create configurations using default values
    println!("\n1. Default Configurations");
    let nest_config = NestLoudsTrieConfig::default();
    println!("   NestLoudsTrieConfig: nest_level={}, compression_level={}", 
             nest_config.nest_level, nest_config.core_str_compression_level);

    let blob_config = BlobStoreConfig::default();
    println!("   BlobStoreConfig: compression_level={}, block_size={}", 
             blob_config.compression_level, blob_config.block_size);

    let memory_config = MemoryConfig::default();
    println!("   MemoryConfig: pool_size={}, alignment={}", 
             memory_config.initial_pool_size, memory_config.alignment);

    // Create configurations using presets
    println!("\n2. Preset Configurations");
    let perf_config = NestLoudsTrieConfig::performance_preset();
    println!("   Performance preset: nest_level={}, parallel_threads={}", 
             perf_config.nest_level, perf_config.parallel_threads);

    let mem_config = NestLoudsTrieConfig::memory_preset();
    println!("   Memory preset: nest_level={}, compression_level={}", 
             mem_config.nest_level, mem_config.core_str_compression_level);

    let rt_config = NestLoudsTrieConfig::realtime_preset();
    println!("   Realtime preset: nest_level={}, compression_level={}", 
             rt_config.nest_level, rt_config.core_str_compression_level);

    // Create configuration using builder pattern
    println!("\n3. Builder Pattern");
    let custom_config = NestLoudsTrieConfig::builder()
        .nest_level(4)
        .compression_level(8)
        .enable_statistics(true)
        .parallel_threads(6)
        .build()?;
    println!("   Custom config: nest_level={}, compression_level={}, statistics={}, threads={}", 
             custom_config.nest_level, 
             custom_config.core_str_compression_level,
             custom_config.enable_statistics,
             custom_config.parallel_threads);

    // Validate configurations
    println!("\n4. Configuration Validation");
    let configs = [
        ("Default", NestLoudsTrieConfig::default()),
        ("Performance", NestLoudsTrieConfig::performance_preset()),
        ("Memory", NestLoudsTrieConfig::memory_preset()),
        ("Realtime", NestLoudsTrieConfig::realtime_preset()),
    ];

    for (name, config) in configs.iter() {
        match config.validate() {
            Ok(()) => println!("   {} config: ✓ Valid", name),
            Err(e) => println!("   {} config: ✗ Invalid - {}", name, e),
        }
    }

    println!("\n5. Environment Variable Integration");
    println!("   Set environment variables like ZIPORA_TRIE_NEST_LEVEL=5 to configure from environment");
    
    // Try to load from environment (will use defaults if not set)
    let env_config = NestLoudsTrieConfig::from_env()?;
    println!("   Environment config: nest_level={}", env_config.nest_level);

    println!("\nConfiguration API demonstration completed successfully! ✅");
    Ok(())
}