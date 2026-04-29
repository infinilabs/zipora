//! Simple tests for LRU page cache implementation

use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_page_cache_config_basic() {
    let config = PageCacheConfig::balanced();
    assert!(config.capacity > 0);
    assert!(config.num_shards > 0);
    assert!(config.page_size == PAGE_SIZE);
}

#[test]
fn test_cache_creation() {
    let config = PageCacheConfig::memory_optimized();
    let cache = LruPageCache::new(config);
    assert!(cache.is_ok());
}

#[test]
fn test_real_file_io_basic() -> Result<()> {
    // Create a simple test file
    let mut temp_file = NamedTempFile::new()?;
    let test_data = b"Hello, LRU Cache!";
    temp_file.write_all(test_data)?;
    temp_file.flush()?;

    // Create cache and test file operations
    let config = PageCacheConfig::balanced();
    let cache = LruPageCache::new(config)?;
    let file_id = cache.open_file(temp_file.path())?;

    // Test basic read
    let buffer = cache.read(file_id, 0, test_data.len())?;
    assert_eq!(buffer.len(), test_data.len());
    assert_eq!(buffer.data(), test_data);

    // Test file size
    let file_size = cache.file_size(file_id)?;
    assert_eq!(file_size, test_data.len() as u64);

    // Close file
    cache.close_file(file_id)?;

    Ok(())
}

#[test]
fn test_cache_statistics() {
    let mut stats = CacheStatistics::new();

    stats.record_hit(CacheHitType::Hit);
    stats.record_miss();

    assert_eq!(stats.hit_ratio(), 0.5);
    assert_eq!(stats.miss_ratio(), 0.5);
}

#[test]
fn test_buffer_operations() {
    let mut buffer = CacheBuffer::new();
    assert!(buffer.is_empty());

    let test_data = b"Test data";
    buffer.copy_from_slice(test_data);

    assert!(!buffer.is_empty());
    assert_eq!(buffer.data(), test_data);
    assert_eq!(buffer.len(), test_data.len());

    buffer.clear();
    assert!(buffer.is_empty());
}

#[test]
fn test_cache_read_same_page_twice() -> Result<()> {
    let mut temp_file = NamedTempFile::new()?;
    let test_data = b"Cached page data";
    temp_file.write_all(test_data)?;
    temp_file.flush()?;

    let config = PageCacheConfig::balanced();
    let cache = LruPageCache::new(config)?;
    let file_id = cache.open_file(temp_file.path())?;

    // First read - cache miss (which loads into cache, triggering an InitialFree hit)
    let buffer1 = cache.read(file_id, 0, test_data.len())?;
    assert_eq!(buffer1.data(), test_data);
    assert_eq!(cache.stats().total_misses, 1);
    assert_eq!(cache.stats().total_hits, 1);

    // Second read - should be cache hit
    let buffer2 = cache.read(file_id, 0, test_data.len())?;
    assert_eq!(buffer2.data(), test_data);
    assert_eq!(cache.stats().total_misses, 1);
    assert_eq!(cache.stats().total_hits, 2);

    cache.close_file(file_id)?;
    Ok(())
}

#[test]
fn test_cache_read_different_offsets() -> Result<()> {
    let mut temp_file = NamedTempFile::new()?;
    let test_data = b"0123456789ABCDEF";
    temp_file.write_all(test_data)?;
    temp_file.flush()?;

    let config = PageCacheConfig::balanced();
    let cache = LruPageCache::new(config)?;
    let file_id = cache.open_file(temp_file.path())?;

    // Read from offset 0
    let buffer1 = cache.read(file_id, 0, 5)?;
    assert_eq!(buffer1.data(), b"01234");

    // Read from offset 5
    let buffer2 = cache.read(file_id, 5, 5)?;
    assert_eq!(buffer2.data(), b"56789");

    cache.close_file(file_id)?;
    Ok(())
}

#[test]
fn test_cache_miss_then_hit() -> Result<()> {
    let mut temp_file = NamedTempFile::new()?;
    temp_file.write_all(b"test")?;
    temp_file.flush()?;

    let config = PageCacheConfig::balanced();
    let cache = LruPageCache::new(config)?;
    let file_id = cache.open_file(temp_file.path())?;

    // Read to populate cache
    let _buffer1 = cache.read(file_id, 0, 4)?;
    let _buffer2 = cache.read(file_id, 0, 4)?;

    // Verify statistics are being tracked by the cache itself
    let stats = cache.stats();
    // The first read triggers a Miss + InitialFree (Hit). The second read triggers a Hit.
    assert_eq!(stats.total_hits, 2);
    assert_eq!(stats.total_misses, 1);

    // Test the ratios
    // total operations = misses + hits = 1 + 2 = 3.
    // Hit ratio = 2/3, Miss ratio = 1/3
    assert!((stats.hit_ratio - 0.666).abs() < 0.01);
    assert!((stats.miss_ratio - 0.333).abs() < 0.01);

    cache.close_file(file_id)?;
    Ok(())
}

#[test]
fn test_cache_file_size_query() -> Result<()> {
    let mut temp_file = NamedTempFile::new()?;
    let test_data = b"File size test data";
    temp_file.write_all(test_data)?;
    temp_file.flush()?;

    let config = PageCacheConfig::balanced();
    let cache = LruPageCache::new(config)?;
    let file_id = cache.open_file(temp_file.path())?;

    let file_size = cache.file_size(file_id)?;
    assert_eq!(file_size, test_data.len() as u64);

    cache.close_file(file_id)?;
    Ok(())
}

#[test]
fn test_cache_large_file_multipage() -> Result<()> {
    let mut temp_file = NamedTempFile::new()?;
    // Write 8KB of data (2 pages at 4KB each)
    let test_data = vec![0x42u8; 8192];
    temp_file.write_all(&test_data)?;
    temp_file.flush()?;

    let config = PageCacheConfig::balanced();
    let cache = LruPageCache::new(config)?;
    let file_id = cache.open_file(temp_file.path())?;

    // Read across page boundary (from offset 3000, length 3000)
    // This spans from page 0 into page 1
    let buffer = cache.read(file_id, 3000, 3000)?;
    assert_eq!(buffer.len(), 3000);
    assert!(buffer.data().iter().all(|&b| b == 0x42));

    cache.close_file(file_id)?;
    Ok(())
}

#[test]
fn test_cache_open_close_file() -> Result<()> {
    let mut temp_file = NamedTempFile::new()?;
    temp_file.write_all(b"lifecycle test")?;
    temp_file.flush()?;

    let config = PageCacheConfig::balanced();
    let cache = LruPageCache::new(config)?;

    // Open file
    let file_id = cache.open_file(temp_file.path())?;
    assert!(file_id > 0);

    // Verify we can read
    let buffer = cache.read(file_id, 0, 4)?;
    assert_eq!(buffer.data(), b"life");

    // Close file
    cache.close_file(file_id)?;

    Ok(())
}

#[test]
fn test_cache_memory_optimized_config() {
    let config = PageCacheConfig::memory_optimized();

    assert!(config.capacity > 0);
    assert!(config.num_shards > 0);
    assert_eq!(config.page_size, PAGE_SIZE);

    // Memory optimized should have smaller capacity than balanced
    let balanced_config = PageCacheConfig::balanced();
    assert!(config.capacity <= balanced_config.capacity);
}

#[test]
fn test_cache_buffer_extend_and_read() -> Result<()> {
    let mut temp_file = NamedTempFile::new()?;
    temp_file.write_all(b"part1part2")?;
    temp_file.flush()?;

    let config = PageCacheConfig::balanced();
    let cache = LruPageCache::new(config)?;
    let file_id = cache.open_file(temp_file.path())?;

    let mut buffer = cache.read(file_id, 0, 5)?;
    assert_eq!(buffer.data(), b"part1");

    // Extend buffer with more data
    buffer.extend_from_slice(b"part2");
    assert_eq!(buffer.data(), b"part1part2");
    assert_eq!(buffer.len(), 10);

    cache.close_file(file_id)?;
    Ok(())
}
