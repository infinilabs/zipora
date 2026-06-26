use super::*;
use std::hash::{BuildHasher, Hasher};

// --- Custom hasher that returns a fixed u64, for testing sentinel edge cases ---

#[derive(Clone)]
struct FixedHashBuilder(u64);

impl BuildHasher for FixedHashBuilder {
    type Hasher = FixedHasher;
    fn build_hasher(&self) -> FixedHasher {
        FixedHasher(self.0)
    }
}

struct FixedHasher(u64);

impl Hasher for FixedHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _bytes: &[u8]) {}
}

// ==================== Config creation tests ====================

#[test]
fn test_unified_hash_map_creation() {
    let map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
}

#[test]
fn test_cache_optimized_config() {
    let result = ZiporaHashMap::<String, i32>::with_config(ZiporaHashMapConfig::cache_optimized());
    assert!(
        result.is_err(),
        "CacheOptimized strategy should be rejected at construction"
    );
}

#[test]
fn test_string_optimized_config() {
    let result = ZiporaHashMap::<String, i32>::with_config(ZiporaHashMapConfig::string_optimized());
    assert!(
        result.is_err(),
        "StringOptimized strategy should be rejected at construction"
    );
}

#[test]
fn test_small_inline_config() {
    let mut map: ZiporaHashMap<i32, String> =
        ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(4)).expect("invariant broken");
    assert_eq!(map.len(), 0);

    // Fill up to inline capacity
    for i in 0..16 {
        map.insert(i, i.to_string()).expect("invariant broken");
    }
    assert_eq!(map.len(), 16);

    // Verify all entries
    for i in 0..16 {
        assert_eq!(map.get(&i), Some(&i.to_string()));
    }
}

// ==================== Core operations ====================

#[test]
fn test_insert_and_get() {
    let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    assert_eq!(
        map.insert("hello".to_string(), 42)
            .expect("invariant broken"),
        None
    );
    assert_eq!(map.get("hello"), Some(&42));
    assert_eq!(map.len(), 1);
    assert!(!map.is_empty());
}

#[test]
fn test_insert_overwrite_returns_old_value() {
    let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    assert_eq!(
        map.insert("key".to_string(), 1).expect("invariant broken"),
        None
    );
    assert_eq!(
        map.insert("key".to_string(), 2).expect("invariant broken"),
        Some(1)
    );
    assert_eq!(map.get("key"), Some(&2));
    assert_eq!(map.len(), 1);
}

#[test]
fn test_get_nonexistent() {
    let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert("a".to_string(), 1).expect("invariant broken");
    assert_eq!(map.get("b"), None);
    assert_eq!(map.get(""), None);
}

#[test]
fn test_remove_existing() {
    let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert("key".to_string(), 99).expect("invariant broken");
    assert_eq!(map.remove("key"), Some(99));
    assert_eq!(map.get("key"), None);
    assert_eq!(map.len(), 0);
}

#[test]
fn test_remove_nonexistent() {
    let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert("a".to_string(), 1).expect("invariant broken");
    assert_eq!(map.remove("b"), None);
    assert_eq!(map.len(), 1);
}

#[test]
fn test_get_mut() {
    let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert("key".to_string(), 10).expect("invariant broken");
    if let Some(val) = map.get_mut("key") {
        *val = 20;
    }
    assert_eq!(map.get("key"), Some(&20));
}

#[test]
fn test_get_mut_nonexistent() {
    let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    assert!(map.get_mut("nope").is_none());
}

#[test]
fn test_contains_key() {
    let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert("yes".to_string(), 1).expect("invariant broken");
    assert!(map.contains_key("yes"));
    assert!(!map.contains_key("no"));
}

#[test]
fn test_len_tracking() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    assert_eq!(map.len(), 0);

    map.insert(1, 10).expect("invariant broken");
    assert_eq!(map.len(), 1);

    map.insert(2, 20).expect("invariant broken");
    assert_eq!(map.len(), 2);

    // Overwrite doesn't increase len
    map.insert(1, 11).expect("invariant broken");
    assert_eq!(map.len(), 2);

    map.remove(&1);
    assert_eq!(map.len(), 1);

    map.remove(&2);
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
}

#[test]
fn test_single_element() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert(42, 100).expect("invariant broken");
    assert_eq!(map.get(&42), Some(&100));
    assert_eq!(map.len(), 1);
    assert_eq!(map.remove(&42), Some(100));
    assert!(map.is_empty());
}

// ==================== Resize / rehash ====================

#[test]
fn test_resize_preserves_all_entries() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    let n = 200;
    for i in 0..n {
        map.insert(i, i * 10).expect("invariant broken");
    }
    assert_eq!(map.len(), n as usize);
    for i in 0..n {
        assert_eq!(
            map.get(&i),
            Some(&(i * 10)),
            "key {} missing after resize",
            i
        );
    }
}

#[test]
fn test_resize_compacts_tombstones() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    // Fill to just under resize threshold
    for i in 0..15 {
        map.insert(i, i).expect("invariant broken");
    }
    // Remove half to create tombstones
    for i in 0..8 {
        map.remove(&i);
    }
    // The 7 remaining entries should still be accessible
    for i in 8..15 {
        assert_eq!(map.get(&i), Some(&i));
    }
    // Insert enough to force resize — tombstones should be compacted
    for i in 100..120 {
        map.insert(i, i).expect("invariant broken");
    }
    // All surviving entries must be present
    for i in 8..15 {
        assert_eq!(map.get(&i), Some(&i), "key {} lost after resize", i);
    }
    for i in 100..120 {
        assert_eq!(map.get(&i), Some(&i), "key {} lost after resize", i);
    }
    // Removed entries must stay gone
    for i in 0..8 {
        assert_eq!(map.get(&i), None, "removed key {} reappeared", i);
    }
}

// ==================== Tombstone behavior ====================

#[test]
fn test_tombstone_get_returns_none() {
    let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert("gone".to_string(), 1).expect("invariant broken");
    map.remove("gone");
    assert_eq!(map.get("gone"), None);
}

#[test]
fn test_tombstone_slot_reuse_same_key() {
    let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert("reuse".to_string(), 1)
        .expect("invariant broken");
    map.remove("reuse");
    // Re-inserting same key should work (tombstone slot eligible)
    assert_eq!(
        map.insert("reuse".to_string(), 2)
            .expect("invariant broken"),
        None
    );
    assert_eq!(map.get("reuse"), Some(&2));
}

#[test]
fn test_tombstone_does_not_break_probe_chain() {
    // Use a fixed hasher so all keys hash to the same value,
    // guaranteeing a linear probe chain: slot 1, 2, 3, ...
    let config = ZiporaHashMapConfig::default();
    let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
        ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(42))
            .expect("invariant broken");

    // All three keys hash to 42 → same initial slot → probe chain
    map.insert(1, 10).expect("invariant broken");
    map.insert(2, 20).expect("invariant broken");
    map.insert(3, 30).expect("invariant broken");

    // Remove the middle of the chain
    assert_eq!(map.remove(&2), Some(20));

    // Key before tombstone
    assert_eq!(map.get(&1), Some(&10));
    // Key after tombstone — probe must skip the tombstone
    assert_eq!(map.get(&3), Some(&30));
    // Removed key
    assert_eq!(map.get(&2), None);
}

#[test]
fn test_multiple_tombstones_in_chain() {
    let config = ZiporaHashMapConfig::default();
    let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
        ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(7))
            .expect("invariant broken");

    for i in 0..6 {
        map.insert(i, i * 100).expect("invariant broken");
    }
    // Remove alternating: 0, 2, 4
    map.remove(&0);
    map.remove(&2);
    map.remove(&4);

    assert_eq!(map.get(&1), Some(&100));
    assert_eq!(map.get(&3), Some(&300));
    assert_eq!(map.get(&5), Some(&500));
    assert_eq!(map.get(&0), None);
    assert_eq!(map.get(&2), None);
    assert_eq!(map.get(&4), None);
    assert_eq!(map.len(), 3);
}

// ==================== Hash sentinel collision fix ====================

#[test]
fn test_hash_sentinel_zero() {
    // All keys hash to raw 0 → sanitized to 1
    let config = ZiporaHashMapConfig::default();
    let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
        ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(0))
            .expect("invariant broken");

    map.insert(10, 100).expect("invariant broken");
    map.insert(20, 200).expect("invariant broken");
    assert_eq!(map.get(&10), Some(&100));
    assert_eq!(map.get(&20), Some(&200));
    assert_eq!(map.remove(&10), Some(100));
    assert_eq!(map.get(&10), None);
    assert_eq!(map.get(&20), Some(&200));
    assert_eq!(map.len(), 1);
}

#[test]
fn test_hash_sentinel_max() {
    // All keys hash to raw u64::MAX → sanitized to u64::MAX - 1
    let config = ZiporaHashMapConfig::default();
    let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
        ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(u64::MAX))
            .expect("invariant broken");

    map.insert(10, 100).expect("invariant broken");
    map.insert(20, 200).expect("invariant broken");
    assert_eq!(map.get(&10), Some(&100));
    assert_eq!(map.get(&20), Some(&200));
    assert_eq!(map.remove(&10), Some(100));
    assert_eq!(map.get(&10), None);
    assert_eq!(map.get(&20), Some(&200));
    assert_eq!(map.len(), 1);
}

#[test]
fn test_hash_sentinel_max_with_get_mut() {
    let config = ZiporaHashMapConfig::default();
    let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
        ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(u64::MAX))
            .expect("invariant broken");

    map.insert(1, 10).expect("invariant broken");
    if let Some(v) = map.get_mut(&1) {
        *v = 99;
    }
    assert_eq!(map.get(&1), Some(&99));
}

// ==================== Iterator ====================

#[test]
fn test_iterator_basic() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert(1, 10).expect("invariant broken");
    map.insert(2, 20).expect("invariant broken");
    map.insert(3, 30).expect("invariant broken");

    let mut collected: Vec<(i32, i32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
    collected.sort();
    assert_eq!(collected, vec![(1, 10), (2, 20), (3, 30)]);
}

#[test]
fn test_iterator_skips_tombstones() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert(1, 10).expect("invariant broken");
    map.insert(2, 20).expect("invariant broken");
    map.insert(3, 30).expect("invariant broken");
    map.remove(&2);

    let mut collected: Vec<(i32, i32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
    collected.sort();
    assert_eq!(collected, vec![(1, 10), (3, 30)]);
}

#[test]
fn test_iterator_empty() {
    let map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    assert_eq!(map.iter().count(), 0);
}

#[test]
fn test_iterator_all_removed() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert(1, 10).expect("invariant broken");
    map.insert(2, 20).expect("invariant broken");
    map.remove(&1);
    map.remove(&2);
    assert_eq!(map.iter().count(), 0);
}

// ==================== Backward shift deletion ====================

#[test]
fn test_backward_shift_delete() {
    // Directly exercise the backward_shift_delete method
    let config = ZiporaHashMapConfig::default();
    let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
        ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(0))
            .expect("invariant broken");

    // Insert 4 keys — all collide, forming a probe chain
    map.insert(1, 10).expect("invariant broken");
    map.insert(2, 20).expect("invariant broken");
    map.insert(3, 30).expect("invariant broken");
    map.insert(4, 40).expect("invariant broken");

    // Manually invoke backward_shift_delete on the first slot
    if let HashMapStorage::Standard { entries, mask, .. } = &mut map.storage {
        // Find the slot containing key 1 (the sanitized hash of 0 is 1 → slot 1 & mask)
        let target_slot = 1usize & *mask;
        entries[target_slot].hash = 0;
        entries[target_slot].key = None;
        entries[target_slot].value = None;
        let m = *mask;
        ZiporaHashMap::<i32, i32, FixedHashBuilder>::backward_shift_delete(entries, m, target_slot);
    }

    // After backward shift: keys 2, 3, 4 should be shifted backward
    // and remain findable (the chain is compacted, no tombstone needed)
    assert_eq!(map.get(&2), Some(&20));
    assert_eq!(map.get(&3), Some(&30));
    assert_eq!(map.get(&4), Some(&40));
}

// ==================== Clear ====================

#[test]
fn test_clear() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    for i in 0..10 {
        map.insert(i, i).expect("invariant broken");
    }
    assert_eq!(map.len(), 10);
    map.clear();
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
    // After clear, can insert again
    map.insert(42, 99).expect("invariant broken");
    assert_eq!(map.get(&42), Some(&99));
}

// ==================== Stress / edge cases ====================

#[test]
fn test_many_inserts() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    let n = 500;
    for i in 0..n {
        map.insert(i, i * 3).expect("invariant broken");
    }
    assert_eq!(map.len(), n as usize);
    for i in 0..n {
        assert_eq!(map.get(&i), Some(&(i * 3)));
    }
}

#[test]
fn test_insert_remove_cycle() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    for round in 0..5 {
        let base = round * 20;
        for i in base..base + 20 {
            map.insert(i, i).expect("invariant broken");
        }
        for i in base..base + 10 {
            assert_eq!(map.remove(&i), Some(i));
        }
    }
    // 5 rounds × 10 surviving = 50 entries
    assert_eq!(map.len(), 50);
    for round in 0..5 {
        let base = round * 20;
        for i in base..base + 10 {
            assert_eq!(map.get(&i), None);
        }
        for i in base + 10..base + 20 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }
}

#[test]
fn test_string_keys() {
    let mut map: ZiporaHashMap<String, String> = ZiporaHashMap::new().expect("invariant broken");
    map.insert("hello".to_string(), "world".to_string())
        .expect("invariant broken");
    map.insert("foo".to_string(), "bar".to_string())
        .expect("invariant broken");
    assert_eq!(map.get("hello"), Some(&"world".to_string()));
    assert_eq!(map.get("foo"), Some(&"bar".to_string()));
    map.remove("hello");
    assert_eq!(map.get("hello"), None);
    assert_eq!(map.get("foo"), Some(&"bar".to_string()));
}

#[test]
fn test_with_capacity() {
    let mut map: ZiporaHashMap<i32, i32> =
        ZiporaHashMap::with_capacity(64).expect("invariant broken");
    assert!(map.capacity() >= 64);
    for i in 0..64 {
        map.insert(i, i).expect("invariant broken");
    }
    assert_eq!(map.len(), 64);
}

#[test]
fn test_overwrite_many_times() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    for v in 0..100 {
        let old = map.insert(0, v).expect("invariant broken");
        if v == 0 {
            assert_eq!(old, None);
        } else {
            assert_eq!(old, Some(v - 1));
        }
    }
    assert_eq!(map.get(&0), Some(&99));
    assert_eq!(map.len(), 1);
}

#[test]
fn test_remove_then_reinsert_different_value() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().expect("invariant broken");
    map.insert(1, 100).expect("invariant broken");
    map.remove(&1);
    map.insert(1, 200).expect("invariant broken");
    assert_eq!(map.get(&1), Some(&200));
    assert_eq!(map.len(), 1);
}

#[test]
fn test_all_collisions_stress() {
    // All keys hash identically — worst-case linear probing
    let config = ZiporaHashMapConfig::default();
    let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
        ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(99))
            .expect("invariant broken");

    let n = 50;
    for i in 0..n {
        map.insert(i, i * 7).expect("invariant broken");
    }
    assert_eq!(map.len(), n as usize);
    for i in 0..n {
        assert_eq!(map.get(&i), Some(&(i * 7)));
    }
    // Remove every other key
    for i in (0..n).step_by(2) {
        assert_eq!(map.remove(&i), Some(i * 7));
    }
    assert_eq!(map.len(), (n / 2) as usize);
    for i in (1..n).step_by(2) {
        assert_eq!(map.get(&i), Some(&(i * 7)));
    }
}

// ==================== Strategy gating tests ====================

#[test]
fn test_cache_optimized_returns_not_supported() {
    let config = ZiporaHashMapConfig::cache_optimized();
    let result = ZiporaHashMap::<i32, i32>::with_config(config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not yet implemented"));
}

#[test]
fn test_string_optimized_returns_not_supported() {
    let config = ZiporaHashMapConfig::string_optimized();
    let result = ZiporaHashMap::<String, i32>::with_config(config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not yet implemented"));
}

// ==================== SmallInline fallback tests ====================

#[test]
fn test_small_inline_basic() {
    let config = ZiporaHashMapConfig::small_inline(16);
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::with_config(config).unwrap();
    for i in 0..16 {
        map.insert(i, i * 10).unwrap();
    }
    assert_eq!(map.len(), 16);
    for i in 0..16 {
        assert_eq!(map.get(&i), Some(&(i * 10)));
    }
}

#[test]
fn test_small_inline_fallback_to_standard() {
    let config = ZiporaHashMapConfig::small_inline(16);
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::with_config(config).unwrap();

    // Fill all 16 inline slots
    for i in 0..16 {
        map.insert(i, i * 10).unwrap();
    }
    assert_eq!(map.len(), 16);

    // Insert 17th element — triggers migration to Standard
    map.insert(16, 160).unwrap();
    assert_eq!(map.len(), 17);

    // All 17 entries must be retrievable
    for i in 0..=16 {
        assert_eq!(
            map.get(&i),
            Some(&(i * 10)),
            "key {} missing after fallback",
            i
        );
    }

    // Further inserts into fallback storage should work
    map.insert(17, 170).unwrap();
    assert_eq!(map.len(), 18);
    assert_eq!(map.get(&17), Some(&170));
}

#[test]
fn test_small_inline_overwrite_before_and_after_fallback() {
    let config = ZiporaHashMapConfig::small_inline(16);
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::with_config(config).unwrap();

    // Insert and overwrite within inline storage
    map.insert(1, 10).unwrap();
    assert_eq!(map.insert(1, 20).unwrap(), Some(10));
    assert_eq!(map.get(&1), Some(&20));

    // Fill rest and trigger fallback
    for i in 2..=16 {
        map.insert(i, i * 10).unwrap();
    }
    map.insert(17, 170).unwrap();

    // Overwrite within fallback storage
    assert_eq!(map.len(), 17);
    assert_eq!(map.get(&1), Some(&20));
}
