
use zipora::hash_map::ZiporaHashMap;
use zipora::hash_map::ZiporaHashMapConfig;
use zipora::hash_map::StorageStrategy;
use zipora::memory::{SecureMemoryPool, SecurePoolConfig};

#[test]
fn test_inline_storage_initialization() {
    // This triggers StorageStrategy::SmallInline initialization
    let config = ZiporaHashMapConfig {
        storage_strategy: StorageStrategy::SmallInline { 
            inline_capacity: 16,
            fallback_threshold: 16,
        },
        initial_capacity: 16,
        ..Default::default()
    };
    
    let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_config(config).unwrap();
    assert_eq!(map.len(), 0);

    // Test usage
    for i in 0..16 {
        map.insert(i, i * 10).unwrap();
    }
    assert_eq!(map.len(), 16);
    for i in 0..16 {
        assert_eq!(map.get(&i), Some(&(i * 10)));
    }
}

#[test]
fn test_cache_optimized_bucket_initialization() {
    // Cache-optimized config
    let config = ZiporaHashMapConfig::cache_optimized();
    let map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_config(config).unwrap();
    assert_eq!(map.len(), 0);
}

#[test]
fn test_secure_pool_corruption_leak_regression() {
    // We want to test that deallocate_internal correctly frees the memory even when corrupted.
    // If it leaked, Miri would complain about memory leaks at the end of this test.
    let config = SecurePoolConfig::small_secure();
    let pool = SecureMemoryPool::new(config).unwrap();

    let ptr = pool.allocate().unwrap();
    let raw_ptr = ptr.as_ptr();

    // The header is located just before the data pointer.
    // We know from SecureChunk::new that it subtracts std::mem::size_of::<ChunkHeader>()
    // We can't access ChunkHeader type easily, but it's typically 24-32 bytes.
    // By writing 0 to the bytes immediately before raw_ptr, we corrupt the header's magic/generation/canary.
    unsafe {
        // Corrupt 8 bytes before the pointer (which overlaps with canary or magic or pool_id)
        let header_ptr = raw_ptr.sub(8) as *mut u64;
        std::ptr::write(header_ptr, 0xDEADBEEF);
    }

    // Now when ptr drops, it will call deallocate_internal.
    // Because the header is corrupted, validation will fail.
    // If the bug is present, the memory will leak (caught by Miri).
    // If the bug is fixed, the memory is deallocated to the system despite corruption.
    drop(ptr);

    // Verify that corruption was actually detected
    let stats = pool.stats();
    assert_eq!(stats.corruption_detected, 1);
    // dealloc_count should still increment even on corruption path
    assert!(stats.dealloc_count >= 1, "dealloc_count should be >= 1, got {}", stats.dealloc_count);
}

#[test]
fn test_local_cache_overflow_no_leak() {
    // Path C regression: when the thread-local cache is full and try_push returns
    // Err(chunk), the rejected chunk must be pushed to the global stack — not dropped.
    // Use local_cache_size=2 so the cache fills quickly.
    let config = SecurePoolConfig::small_secure()
        .with_local_cache_size(2);
    let pool = SecureMemoryPool::new(config).unwrap();

    // Allocate and free 2 chunks to fill the local cache
    for _ in 0..2 {
        let ptr = pool.allocate().unwrap();
        drop(ptr);
    }

    // The 3rd deallocation should trigger the cache overflow path.
    // If the bug were present, this chunk would leak.
    let ptr = pool.allocate().unwrap();
    drop(ptr);

    let stats = pool.stats();
    assert_eq!(stats.dealloc_count, 3, "all 3 deallocations should complete");

    // Allocate again — this should succeed by reusing a cached/stacked chunk,
    // proving nothing was leaked to limbo.
    let ptr = pool.allocate().unwrap();
    assert!(!ptr.as_ptr().is_null());
    drop(ptr);
}

#[test]
fn test_pool_dropped_chunk_deallocated_directly() {
    // When the pool Arc is dropped before the SecurePooledPtr, the ptr's Drop impl
    // calls chunk.deallocate() directly (line 1374). This must not leak.
    let config = SecurePoolConfig::small_secure();
    let pool = SecureMemoryPool::new(config).unwrap();
    let ptr = pool.allocate().unwrap();

    // Drop the pool first (only our ptr holds a Weak ref now)
    drop(pool);

    // Now drop the ptr — it should deallocate directly since pool is gone
    // If this leaked, Miri would catch it.
    drop(ptr);
}
