//! Simple tests for LRU page cache implementation

use super::*;
use tempfile::NamedTempFile;
use std::io::Write;

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