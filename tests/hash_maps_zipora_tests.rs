use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};

#[test]
fn test_sentinel_hash_collisions() {
    // Let's create a custom hasher that always returns 0
    #[derive(Clone, Default)]
    struct ZeroHasher;
    impl Hasher for ZeroHasher {
        fn finish(&self) -> u64 {
            0
        }
        fn write(&mut self, _: &[u8]) {}
    }
    #[derive(Clone, Default)]
    struct ZeroBuildHasher;
    impl BuildHasher for ZeroBuildHasher {
        type Hasher = ZeroHasher;
        fn build_hasher(&self) -> Self::Hasher {
            ZeroHasher
        }
    }

    let mut map_zero: ZiporaHashMap<i32, &str, ZeroBuildHasher> =
        ZiporaHashMap::with_config_and_hasher(ZiporaHashMapConfig::default(), ZeroBuildHasher)
            .unwrap();
    map_zero.insert(1, "one").unwrap();
    assert_eq!(map_zero.get(&1), Some(&"one"));
    assert_eq!(map_zero.remove(&1), Some("one"));
    assert_eq!(map_zero.get(&1), None);

    // Let's create a custom hasher that always returns MAX
    #[derive(Clone, Default)]
    struct MaxHasher;
    impl Hasher for MaxHasher {
        fn finish(&self) -> u64 {
            u64::MAX
        }
        fn write(&mut self, _: &[u8]) {}
    }
    #[derive(Clone, Default)]
    struct MaxBuildHasher;
    impl BuildHasher for MaxBuildHasher {
        type Hasher = MaxHasher;
        fn build_hasher(&self) -> Self::Hasher {
            MaxHasher
        }
    }

    let mut map_max: ZiporaHashMap<i32, &str, MaxBuildHasher> =
        ZiporaHashMap::with_config_and_hasher(ZiporaHashMapConfig::default(), MaxBuildHasher)
            .unwrap();
    map_max.insert(1, "one").unwrap();
    assert_eq!(map_max.get(&1), Some(&"one"));
    assert_eq!(map_max.remove(&1), Some("one"));
    assert_eq!(map_max.get(&1), None);
}

#[test]
fn test_tombstone_resize() {
    let mut map: ZiporaHashMap<i32, i32, RandomState> =
        ZiporaHashMap::with_config(ZiporaHashMapConfig::default()).unwrap();

    // Insert many elements to trigger resize later
    for i in 0..100 {
        map.insert(i, i * 10).unwrap();
    }

    // Remove some to create tombstones
    for i in 0..50 {
        map.remove(&i);
    }

    assert_eq!(map.len(), 50);

    // Insert more to trigger resize. When it resizes, it should ignore the tombstones.
    for i in 100..200 {
        map.insert(i, i * 10).unwrap();
    }

    // Verify values
    for i in 0..50 {
        assert_eq!(map.get(&i), None);
    }
    for i in 50..100 {
        assert_eq!(map.get(&i), Some(&(i * 10)));
    }
    for i in 100..200 {
        assert_eq!(map.get(&i), Some(&(i * 10)));
    }

    assert_eq!(map.len(), 150);
}
