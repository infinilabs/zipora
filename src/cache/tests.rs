//! Comprehensive tests for LRU page cache implementation

use super::*;
use crate::error::Result;
use std::io::Write;
use std::sync::Arc;
use std::thread;
use tempfile::NamedTempFile;

#[test]
fn test_page_cache_config_creation() {
    let config = PageCacheConfig::balanced();
    assert!(config.capacity > 0);
    assert!(config.num_shards > 0);
    assert!(config.page_size == PAGE_SIZE);
    
    let performance_config = PageCacheConfig::performance_optimized();
    assert!(performance_config.capacity >= config.capacity);
    
    let memory_config = PageCacheConfig::memory_optimized();
    assert!(memory_config.capacity <= config.capacity);
    
    let security_config = PageCacheConfig::security_optimized();
    assert!(security_config.locking.secure_cleanup);
}

#[test]
fn test_page_cache_config_validation() {
    let mut config = PageCacheConfig::balanced();
    assert!(config.validate().is_ok());
    
    config.capacity = 0;
    assert!(config.validate().is_err());
    
    config.capacity = 1024 * 1024; // Reset
    config.num_shards = 0;
    assert!(config.validate().is_err());
    
    config.num_shards = 4; // Reset
    config.page_size = 1; // Invalid page size
    assert!(config.validate().is_err());
}

#[test]
fn test_single_lru_page_cache_creation() {
    let config = PageCacheConfig::memory_optimized();
    let cache = SingleLruPageCache::new(config);
    assert!(cache.is_ok());
    
    let cache = cache.unwrap();
    assert!(cache.capacity() > 0);
    assert_eq!(cache.size(), 0);
    
    let stats = cache.stats().snapshot();
    assert_eq!(stats.total_hits, 0);
    assert_eq!(stats.total_misses, 0);
}

#[test]
fn test_file_registration() {
    let config = PageCacheConfig::balanced();
    let cache = SingleLruPageCache::new(config).unwrap();
    
    let file_id1 = cache.register_file(1).unwrap();
    let file_id2 = cache.register_file(2).unwrap();
    
    assert_ne!(file_id1, file_id2);
    assert!(file_id1 > 0);
    assert!(file_id2 > 0);
}

#[test]
fn test_basic_cache_operations() {
    let config = PageCacheConfig::memory_optimized();
    let cache = SingleLruPageCache::new(config).unwrap();
    let file_id = cache.register_file(1).unwrap();
    
    // Create test data
    let test_data = vec![42u8; PAGE_SIZE];
    let mut buffer = CacheBuffer::new();
    
    // First read should be a miss (no actual I/O in test)
    let result = cache.read(file_id, 0, PAGE_SIZE, &mut buffer);
    assert!(result.is_ok());
    
    let stats = cache.stats().snapshot();
    assert!(stats.total_hits + stats.total_misses > 0);
}

#[test]
fn test_multi_shard_cache() {
    let config = PageCacheConfig::performance_optimized();
    let cache = LruPageCache::new(config).unwrap();
    
    assert!(cache.shard_count() > 1);
    
    let file_id = cache.register_file(1).unwrap();
    
    // Test read operations across shards
    let result1 = cache.read(file_id, 0, PAGE_SIZE);
    let result2 = cache.read(file_id, PAGE_SIZE as u64, PAGE_SIZE);
    
    assert!(result1.is_ok());
    assert!(result2.is_ok());
    
    let stats = cache.stats();
    assert!(stats.total_hits + stats.total_misses >= 2);
}

#[test]
fn test_cache_buffer_operations() {
    let mut buffer = CacheBuffer::new();
    assert!(buffer.is_empty());
    assert!(!buffer.has_data());
    
    let test_data = b"Hello, cache world!";
    buffer.copy_from_slice(test_data);
    
    assert!(!buffer.is_empty());
    assert!(buffer.has_data());
    assert_eq!(buffer.data(), test_data);
    assert_eq!(buffer.len(), test_data.len());
    
    buffer.clear();
    assert!(buffer.is_empty());
    assert!(!buffer.has_data());
}

#[test]
fn test_buffer_pool() {
    let pool = BufferPool::new(10);
    
    let buffer1 = pool.get();
    let buffer2 = pool.get();
    
    pool.put(buffer1);
    pool.put(buffer2);
    
    let stats = pool.stats();
    assert_eq!(stats.max_size, 10);
    assert!(stats.allocations >= 2);
    
    // Get buffers again - should reuse
    let _buffer3 = pool.get();
    let _buffer4 = pool.get();
    
    let stats_after = pool.stats();
    assert!(stats_after.reuses >= 2);
}

#[test]
fn test_cache_statistics() {
    let mut stats = CacheStatistics::new();
    
    stats.record_hit(CacheHitType::Hit);
    stats.record_hit(CacheHitType::EvictedOthers);
    stats.record_miss();
    stats.record_bytes_read(1024);
    stats.record_bytes_cached(512);
    
    assert_eq!(stats.hit_ratio(), 2.0 / 3.0);
    assert_eq!(stats.miss_ratio(), 1.0 / 3.0);
    
    let snapshot = stats.snapshot();
    assert_eq!(snapshot.total_hits, 2);
    assert_eq!(snapshot.total_misses, 1);
    assert_eq!(snapshot.bytes_read, 1024);
    assert_eq!(snapshot.bytes_cached, 512);
    
    let formatted = snapshot.format();
    assert!(formatted.contains("Hit Ratio"));
    assert!(formatted.contains("66.67%"));
}

#[test]
fn test_concurrent_cache_access() {
    let config = PageCacheConfig::performance_optimized();
    let cache = Arc::new(LruPageCache::new(config).unwrap());
    let file_id = cache.register_file(1).unwrap();
    
    let mut handles = vec![];
    
    // Spawn multiple threads to access cache concurrently
    for thread_id in 0..8 {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let offset = (thread_id * 100 + i) as u64 * PAGE_SIZE as u64;
                let result = cache_clone.read(file_id, offset, PAGE_SIZE);
                assert!(result.is_ok());
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    let stats = cache.stats();
    assert!(stats.total_hits + stats.total_misses >= 800);
}

#[test]
fn test_batch_operations() {
    let config = PageCacheConfig::balanced();
    let cache = LruPageCache::new(config).unwrap();
    let file_id = cache.register_file(1).unwrap();
    
    // Create batch read requests
    let requests = vec![
        (file_id, 0, PAGE_SIZE),
        (file_id, PAGE_SIZE as u64, PAGE_SIZE),
        (file_id, 2 * PAGE_SIZE as u64, PAGE_SIZE),
        (file_id, 3 * PAGE_SIZE as u64, PAGE_SIZE),
    ];
    
    let results = cache.read_batch(requests);
    assert!(results.is_ok());
    
    let buffers = results.unwrap();
    assert_eq!(buffers.len(), 4);
    
    for buffer in buffers {
        assert!(buffer.has_data());
    }
}

#[test]
fn test_prefetch_operations() {
    let config = PageCacheConfig::performance_optimized();
    let cache = LruPageCache::new(config).unwrap();
    let file_id = cache.register_file(1).unwrap();
    
    // Test prefetch
    let result = cache.prefetch(file_id, 0, 4 * PAGE_SIZE);
    assert!(result.is_ok());
    
    // Subsequent reads should potentially hit cache
    let buffer1 = cache.read(file_id, 0, PAGE_SIZE);
    let buffer2 = cache.read(file_id, PAGE_SIZE as u64, PAGE_SIZE);
    
    assert!(buffer1.is_ok());
    assert!(buffer2.is_ok());
}

// Note: CachedBlobStore integration tests are pending implementation
// of write-through/write-back caching functionality

#[test]
fn test_cache_error_handling() {
    // Test with invalid configuration
    let mut config = PageCacheConfig::balanced();
    config.capacity = 0;
    
    let result = LruPageCache::new(config);
    assert!(result.is_err());
    
    // Test with valid cache but invalid operations
    let config = PageCacheConfig::memory_optimized();
    let cache = LruPageCache::new(config).unwrap();
    
    // Test resize (should fail as not supported)
    let mut cache_mut = cache;
    let resize_result = cache_mut.resize(1024 * 1024);
    assert!(resize_result.is_err());
}

#[test]
fn test_hash_functions() {
    let file_id = 123;
    let page_id = 456;
    
    let hash1 = hash_file_page(file_id, page_id);
    let hash2 = hash_file_page(file_id, page_id);
    assert_eq!(hash1, hash2); // Same inputs should produce same hash
    
    let hash3 = hash_file_page(file_id + 1, page_id);
    assert_ne!(hash1, hash3); // Different inputs should produce different hash
    
    let shard_id = get_shard_id(file_id, page_id, 8);
    assert!(shard_id < 8);
}

#[test]
fn test_memory_alignment() {
    let config = PageCacheConfig::performance_optimized();
    let cache = SingleLruPageCache::new(config).unwrap();
    
    // Basic cache alignment test - detailed node testing pending
    // implementation of advanced cache node structures
    assert!(cache.capacity() > 0);
}

#[test]
fn test_hit_type_classification() {
    assert_eq!(CacheHitType::Hit.as_index(), 0);
    assert_eq!(CacheHitType::EvictedOthers.as_index(), 1);
    assert_eq!(CacheHitType::InitialFree.as_index(), 2);
    assert_eq!(CacheHitType::DroppedFree.as_index(), 3);
    assert_eq!(CacheHitType::HitOthersLoad.as_index(), 4);
    assert_eq!(CacheHitType::Mix.as_index(), 5);
    assert_eq!(CacheHitType::Miss.as_index(), 6);
}

#[test]
fn test_shard_strategy() {
    let strategy = ShardStrategy::Hash;
    let shard_id = strategy.select_shard(123, 456, 8);
    assert!(shard_id < 8);
    
    let round_robin = ShardStrategy::RoundRobin;
    let shard_id2 = round_robin.select_shard(123, 456, 8);
    assert!(shard_id2 < 8);
    assert_eq!(shard_id2, 456 % 8);
    
    let custom = ShardStrategy::Custom(|_file_id, page_id, num_shards| page_id % num_shards);
    let shard_id3 = custom.select_shard(123, 456, 8);
    assert_eq!(shard_id3, 456 % 8);
}

#[test]
fn test_cache_operation_context() {
    let mut ctx = CacheOpContext::new(1, 1024);
    assert_eq!(ctx.op_id, 1);
    assert_eq!(ctx.bytes_requested, 1024);
    assert!(ctx.shards_used.is_empty());
    
    ctx.add_shard(0, CacheHitType::Hit);
    ctx.add_shard(1, CacheHitType::Miss);
    
    assert_eq!(ctx.shards_used.len(), 2);
    assert_eq!(ctx.overall_hit_type(), CacheHitType::Mix);
    
    let duration = ctx.duration();
    assert!(duration.as_nanos() > 0);
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn bench_single_page_reads() {
        let config = PageCacheConfig::performance_optimized();
        let cache = LruPageCache::new(config).unwrap();
        let file_id = cache.register_file(1).unwrap();
        
        let iterations = 10000;
        let start = Instant::now();
        
        for i in 0..iterations {
            let offset = (i % 1000) as u64 * PAGE_SIZE as u64;
            let _result = cache.read(file_id, offset, PAGE_SIZE).unwrap();
        }
        
        let elapsed = start.elapsed();
        let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
        
        println!("Single page read performance: {:.0} ops/sec", ops_per_sec);
        assert!(ops_per_sec > 1000.0); // Should be at least 1K ops/sec
    }
    
    #[test]
    fn bench_multi_threaded_access() {
        let config = PageCacheConfig::performance_optimized();
        let cache = Arc::new(LruPageCache::new(config).unwrap());
        let file_id = cache.register_file(1).unwrap();
        
        let num_threads = 8;
        let iterations_per_thread = 1000;
        
        let start = Instant::now();
        let mut handles = vec![];
        
        for thread_id in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..iterations_per_thread {
                    let offset = ((thread_id * iterations_per_thread + i) % 1000) as u64 * PAGE_SIZE as u64;
                    let _result = cache_clone.read(file_id, offset, PAGE_SIZE).unwrap();
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let elapsed = start.elapsed();
        let total_ops = num_threads * iterations_per_thread;
        let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();
        
        println!("Multi-threaded performance: {:.0} ops/sec ({} threads)", ops_per_sec, num_threads);
        assert!(ops_per_sec > 5000.0); // Should be at least 5K ops/sec with multiple threads
    }
    
    #[test]
    fn bench_buffer_pool_performance() {
        let pool = BufferPool::new(100);
        let iterations = 100000;
        
        let start = Instant::now();
        
        for _ in 0..iterations {
            let buffer = pool.get();
            pool.put(buffer);
        }
        
        let elapsed = start.elapsed();
        let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
        
        println!("Buffer pool performance: {:.0} ops/sec", ops_per_sec);
        assert!(ops_per_sec > 10000.0); // Should be at least 10K ops/sec
        
        let stats = pool.stats();
        println!("Buffer pool reuse ratio: {:.2}%", stats.reuse_ratio() * 100.0);
    }
    
    #[test]
    fn bench_cache_statistics() {
        let stats = CacheStatistics::new();
        let iterations = 1000000;
        
        let start = Instant::now();
        
        for i in 0..iterations {
            if i % 2 == 0 {
                stats.record_hit(CacheHitType::Hit);
            } else {
                stats.record_miss();
            }
            stats.record_bytes_read(1024);
        }
        
        let elapsed = start.elapsed();
        let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
        
        println!("Statistics recording performance: {:.0} ops/sec", ops_per_sec);
        assert!(ops_per_sec > 1000000.0); // Should be at least 1M ops/sec for atomic operations
    }
}

// Integration tests with real I/O (if available)
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;
    
    #[test]
    fn test_with_real_file() -> Result<()> {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_cache_file");
        
        // Create test file
        let mut file = File::create(&file_path).unwrap();
        let test_data = vec![42u8; PAGE_SIZE * 4];
        file.write_all(&test_data).unwrap();
        file.sync_all().unwrap();
        drop(file);
        
        // Test cache with real file
        let config = PageCacheConfig::balanced();
        let cache = LruPageCache::new(config).unwrap();
        
        let file = File::open(&file_path).unwrap();
        let fd = 1; // Placeholder - would need actual file descriptor
        let file_id = cache.register_file(fd)?;
        
        // Test reading different pages
        let buffer1 = cache.read(file_id, 0, PAGE_SIZE)?;
        let buffer2 = cache.read(file_id, PAGE_SIZE as u64, PAGE_SIZE)?;
        
        assert!(buffer1.has_data());
        assert!(buffer2.has_data());
        
        // Test prefetch
        cache.prefetch(file_id, 2 * PAGE_SIZE as u64, 2 * PAGE_SIZE)?;
        
        let stats = cache.stats();
        println!("Cache stats: {}", stats.format());
        
        Ok(())
    }
}

#[test]
fn test_real_file_io_cache() -> Result<()> {
    // Create a test file with known content
    let mut temp_file = NamedTempFile::new()
        .map_err(|e| ZiporaError::invalid_data(format!("Failed to create temp file: {}", e)))?;
    
    // Create test data that spans multiple pages
    let test_data_size = PAGE_SIZE * 3 + 512; // 3.5 pages
    let mut test_data = Vec::with_capacity(test_data_size);
    
    // Fill with pattern: each byte contains its position modulo 256
    for i in 0..test_data_size {
        test_data.push((i % 256) as u8);
    }
    
    temp_file.write_all(&test_data)
        .map_err(|e| ZiporaError::invalid_data(format!("Failed to write test data: {}", e)))?;
    
    temp_file.flush()
        .map_err(|e| ZiporaError::invalid_data(format!("Failed to flush test data: {}", e)))?;
    
    // Create cache and open the file
    let config = PageCacheConfig::balanced()
        .with_capacity(512 * 1024); // 512KB cache
    let cache = LruPageCache::new(config)?;
    let file_id = cache.open_file(temp_file.path())?;
    
    // Test 1: Read first page
    println!("Testing first page read...");
    let first_page = cache.read(file_id, 0, PAGE_SIZE)?;
    assert_eq!(first_page.len(), PAGE_SIZE);
    
    // Verify content of first page
    let first_page_data = first_page.data();
    for i in 0..PAGE_SIZE {
        assert_eq!(first_page_data[i], (i % 256) as u8, "Mismatch at offset {}", i);
    }
    
    // Test 2: Read partial data from middle of file
    println!("Testing partial read...");
    let offset = PAGE_SIZE + 256;
    let length = 1024;
    let partial_data = cache.read(file_id, offset as u64, length)?;
    assert_eq!(partial_data.len(), length);
    
    // Verify partial data content
    let partial_data_slice = partial_data.data();
    for i in 0..length {
        let expected = ((offset + i) % 256) as u8;
        assert_eq!(partial_data_slice[i], expected, "Mismatch at partial offset {}", i);
    }
    
    // Test 3: Read data spanning multiple pages
    println!("Testing multi-page read...");
    let multi_page_offset = PAGE_SIZE - 512;
    let multi_page_length = PAGE_SIZE + 1024; // Spans 2 pages
    let multi_page_data = cache.read(file_id, multi_page_offset as u64, multi_page_length)?;
    assert_eq!(multi_page_data.len(), multi_page_length);
    
    // Verify multi-page content
    let multi_page_slice = multi_page_data.data();
    for i in 0..multi_page_length {
        let expected = ((multi_page_offset + i) % 256) as u8;
        assert_eq!(multi_page_slice[i], expected, "Mismatch at multi-page offset {}", i);
    }
    
    // Test 4: Test caching efficiency (should hit cache on second read)
    println!("Testing cache hits...");
    let stats_before = cache.stats();
    let cached_read = cache.read(file_id, 0, PAGE_SIZE)?;
    let stats_after = cache.stats();
    
    assert_eq!(cached_read.len(), PAGE_SIZE);
    assert!(stats_after.hits > stats_before.hits, "Expected cache hit");
    
    // Test 5: Test prefetching
    println!("Testing prefetching...");
    let prefetch_offset = 2 * PAGE_SIZE as u64;
    let prefetch_length = PAGE_SIZE;
    cache.prefetch(file_id, prefetch_offset, prefetch_length)?;
    
    // Reading prefetched data should be fast (from cache)
    let prefetched_data = cache.read(file_id, prefetch_offset, prefetch_length)?;
    assert_eq!(prefetched_data.len(), prefetch_length);
    
    // Test 6: Test reading beyond file end
    println!("Testing read beyond file...");
    let beyond_file_offset = test_data_size as u64 + PAGE_SIZE as u64;
    let beyond_file_data = cache.read(file_id, beyond_file_offset, PAGE_SIZE)?;
    assert_eq!(beyond_file_data.len(), 0, "Should return empty data when reading beyond file");
    
    // Test 7: Test reading at exact file end
    println!("Testing read at file boundary...");
    let boundary_offset = test_data_size as u64 - 100;
    let boundary_data = cache.read(file_id, boundary_offset, 200)?; // Request more than available
    assert_eq!(boundary_data.len(), 100, "Should return only available data");
    
    // Verify boundary data
    let boundary_slice = boundary_data.data();
    for i in 0..100 {
        let expected = ((boundary_offset as usize + i) % 256) as u8;
        assert_eq!(boundary_slice[i], expected, "Mismatch at boundary offset {}", i);
    }
    
    // Test 8: Test file size
    println!("Testing file size query...");
    let file_size = cache.file_size(file_id)?;
    assert_eq!(file_size, test_data_size as u64, "File size should match written data");
    
    println!("All real file I/O tests passed!");
    println!("Final cache stats: {}", cache.stats().format());
    
    // Clean up
    cache.close_file(file_id)?;
    
    Ok(())
}

#[test]
fn test_file_manager_operations() -> Result<()> {
    // Test creating FileManager and opening files
    let file_manager = FileManager::new();
    
    // Create a small test file
    let mut temp_file = NamedTempFile::new()
        .map_err(|e| ZiporaError::invalid_data(format!("Failed to create temp file: {}", e)))?;
    
    let test_content = b"Hello, FileManager test!";
    temp_file.write_all(test_content)
        .map_err(|e| ZiporaError::invalid_data(format!("Failed to write: {}", e)))?;
    temp_file.flush()
        .map_err(|e| ZiporaError::invalid_data(format!("Failed to flush: {}", e)))?;
    
    // Test opening file
    let file_id = file_manager.open_file(temp_file.path())?;
    assert!(file_id > 0, "File ID should be positive");
    
    // Test file size
    let size = file_manager.file_size(file_id)?;
    assert_eq!(size, test_content.len() as u64, "File size should match written content");
    
    // Test page calculations
    let page_id = FileManager::offset_to_page_id(5000);
    assert_eq!(page_id, 1, "Offset 5000 should be in page 1");
    
    let offset_in_page = FileManager::offset_within_page(5000);
    assert_eq!(offset_in_page, 5000 - PAGE_SIZE, "Offset within page should be 5000 - 4096");
    
    let aligned_offset = FileManager::page_aligned_offset(page_id);
    assert_eq!(aligned_offset, PAGE_SIZE as u64, "Page 1 should start at 4096");
    
    // Test reading a page
    let mut buffer = vec![0u8; PAGE_SIZE];
    let bytes_read = file_manager.read_page(file_id, 0, &mut buffer)?;
    assert_eq!(bytes_read, test_content.len(), "Should read all test content");
    assert_eq!(&buffer[..test_content.len()], test_content, "Content should match");
    
    // Test reading partial data
    let mut small_buffer = vec![0u8; 10];
    let bytes_read = file_manager.read_data(file_id, 7, 5, &mut small_buffer)?;
    assert_eq!(bytes_read, 5, "Should read 5 bytes");
    assert_eq!(&small_buffer[..5], &test_content[7..12], "Partial content should match");
    
    // Test closing file
    file_manager.close_file(file_id)?;
    
    // Attempting to read from closed file should fail
    let result = file_manager.file_size(file_id);
    assert!(result.is_err(), "Reading from closed file should fail");
    
    println!("FileManager operations test passed!");
    
    Ok(())
}