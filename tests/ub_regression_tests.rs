
use zipora::hash_map::ZiporaHashMap;
use zipora::hash_map::ZiporaHashMapConfig;
use zipora::hash_map::StorageStrategy;

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
