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
    if let HashMapStorage::Standard(table) = &mut map.storage {
        let target_slot = 0usize;
        ZiporaHashMap::<i32, i32, FixedHashBuilder>::backward_shift_delete(
            table,
            &map.hash_builder,
            target_slot,
        );
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

#[test]
fn test_tombstone_duplicate_key_regression() {
    let config = ZiporaHashMapConfig::default();
    let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
        ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(42))
            .expect("invariant broken");

    // All three keys collide and form a probe chain:
    // Slot 1: Key 1
    // Slot 2: Key 2
    // Slot 3: Key 3
    map.insert(1, 10).unwrap();
    map.insert(2, 20).unwrap();
    map.insert(3, 30).unwrap();

    // Remove key 2, creating a tombstone at Slot 2.
    // The probe chain now has a tombstone in the middle:
    // Slot 1: Key 1
    // Slot 2: Tombstone
    // Slot 3: Key 3
    assert_eq!(map.remove(&2), Some(20));

    // Re-insert key 3 with a new value.
    // Previously, the bug would cause the insertion to stop at Slot 2 (Tombstone)
    // and insert key 3 there, creating a duplicate.
    let old_value = map.insert(3, 300).unwrap();

    // We expect the old value (30) to be returned since key 3 already existed in the map.
    assert_eq!(old_value, Some(30));

    // The length of the map should still be 2 (keys 1 and 3).
    assert_eq!(map.len(), 2);

    // We should be able to retrieve the updated value for key 3.
    assert_eq!(map.get(&3), Some(&300));
}

#[test]
fn test_clone_preserves_all_entries() {
    // Test Standard storage clone
    {
        let mut map: ZiporaHashMap<i32, String> = ZiporaHashMap::new().unwrap();
        for i in 0..20 {
            map.insert(i, i.to_string()).unwrap();
        }
        let cloned = map.clone();
        assert_eq!(cloned.len(), map.len());
        for i in 0..20 {
            assert_eq!(cloned.get(&i), map.get(&i));
        }
    }

    // Test SmallInline storage clone (without migration)
    {
        let config = ZiporaHashMapConfig::small_inline(16);
        let mut map: ZiporaHashMap<i32, String> = ZiporaHashMap::with_config(config).unwrap();
        for i in 0..10 {
            map.insert(i, i.to_string()).unwrap();
        }
        let cloned = map.clone();
        assert_eq!(cloned.len(), map.len());
        for i in 0..10 {
            assert_eq!(cloned.get(&i), map.get(&i));
        }
    }

    // Test SmallInline storage clone (with migration/fallback)
    {
        let config = ZiporaHashMapConfig::small_inline(16);
        let mut map: ZiporaHashMap<i32, String> = ZiporaHashMap::with_config(config).unwrap();
        for i in 0..20 {
            map.insert(i, i.to_string()).unwrap();
        }
        let cloned = map.clone();
        assert_eq!(cloned.len(), map.len());
        for i in 0..20 {
            assert_eq!(cloned.get(&i), map.get(&i));
        }
    }
}

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Debug)]
struct DropTracker(Arc<AtomicUsize>);

impl Drop for DropTracker {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn test_small_inline_remove_and_clear_drops_values() {
    let drop_counter = Arc::new(AtomicUsize::new(0));
    let config = ZiporaHashMapConfig::small_inline(16);
    let mut map: ZiporaHashMap<i32, DropTracker> = ZiporaHashMap::with_config(config).unwrap();

    for i in 0..5 {
        map.insert(i, DropTracker(Arc::clone(&drop_counter)))
            .unwrap();
    }
    assert_eq!(map.len(), 5);

    // Remove entry 2
    let removed = map.remove(&2);
    assert!(removed.is_some());
    assert_eq!(map.len(), 4);
    assert!(!map.contains_key(&2));
    assert!(map.get(&2).is_none());

    // Drop the removed value
    drop(removed);
    assert_eq!(drop_counter.load(Ordering::SeqCst), 1);

    // Clear map
    map.clear();
    assert_eq!(map.len(), 0);
    assert_eq!(map.iter().count(), 0);
    assert_eq!(drop_counter.load(Ordering::SeqCst), 5); // 1 from remove + 4 from clear
}

#[test]
fn test_get_mut_after_small_inline_migration() {
    let config = ZiporaHashMapConfig::small_inline(16);
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::with_config(config).unwrap();

    // Insert 20 entries to force migration from SmallInline to Standard fallback
    for i in 0..20 {
        map.insert(i, i * 10).unwrap();
    }

    // get_mut on early key (0) should work
    let val_mut = map.get_mut(&0);
    assert!(val_mut.is_some());
    *val_mut.unwrap() = 999;

    assert_eq!(map.get(&0), Some(&999));
}

#[test]
fn test_len_tracking_consistency() {
    let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
    assert_eq!(map.len(), 0);

    for i in 0..50 {
        map.insert(i, i).unwrap();
        assert_eq!(map.len(), i as usize + 1);
        assert_eq!(map.len(), map.iter().count());
    }

    for i in 0..20 {
        map.remove(&i);
        assert_eq!(map.len(), 50 - (i as usize + 1));
        assert_eq!(map.len(), map.iter().count());
    }

    map.clear();
    assert_eq!(map.len(), 0);
    assert_eq!(map.len(), map.iter().count());
}

#[test]
fn test_migration_panic_safety() {
    use std::hash::Hasher;

    #[derive(Clone, Debug)]
    struct PanicHashKey {
        id: usize,
        counter: Arc<AtomicUsize>,
        panic_on_hash: Arc<AtomicUsize>,
        hash_count: Arc<AtomicUsize>,
    }

    impl PartialEq for PanicHashKey {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }

    impl Eq for PanicHashKey {}

    impl Hash for PanicHashKey {
        fn hash<H: Hasher>(&self, state: &mut H) {
            let count = self.hash_count.fetch_add(1, Ordering::SeqCst) + 1;
            if count == self.panic_on_hash.load(Ordering::SeqCst) {
                panic!("Simulated hash panic during migration");
            }
            self.id.hash(state);
        }
    }

    impl Drop for PanicHashKey {
        fn drop(&mut self) {
            self.counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drop_counter = Arc::new(AtomicUsize::new(0));
    let hash_counter = Arc::new(AtomicUsize::new(0));
    let panic_on_hash = Arc::new(AtomicUsize::new(9999));

    let config = ZiporaHashMapConfig::small_inline(16);
    let mut map: ZiporaHashMap<PanicHashKey, usize> = ZiporaHashMap::with_config(config).unwrap();

    // Insert 16 elements into SmallInline
    for i in 0..16 {
        map.insert(
            PanicHashKey {
                id: i,
                counter: Arc::clone(&drop_counter),
                panic_on_hash: Arc::clone(&panic_on_hash),
                hash_count: Arc::clone(&hash_counter),
            },
            i,
        )
        .unwrap();
    }

    // Now set panic_on_hash to trigger during migration loop
    panic_on_hash.store(hash_counter.load(Ordering::SeqCst) + 2, Ordering::SeqCst);

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        map.insert(
            PanicHashKey {
                id: 16,
                counter: Arc::clone(&drop_counter),
                panic_on_hash: Arc::clone(&panic_on_hash),
                hash_count: Arc::clone(&hash_counter),
            },
            16,
        )
    }));

    assert!(res.is_err());
    // Dropping map after panic unwinding must not cause double-free/double-drop
    drop(map);
}

#[test]
fn test_non_power_of_two_capacity_rounds_up() {
    // Regression (plan.md 7.2): with_capacity(100) used to produce mask 99,
    // whose probe sequence `(index + i) & 99` misses slots and could fail
    // resize with "Failed to reinsert during resize".
    let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_capacity(100).unwrap();
    for i in 0..300u32 {
        map.insert(i, i * 2).unwrap();
    }
    assert_eq!(map.len(), 300);
    for i in 0..300u32 {
        assert_eq!(map.get(&i), Some(&(i * 2)), "key {} lost", i);
    }
}

#[test]
fn test_clear_then_reuse_standard() {
    // Verifies clear_standard semantics (plan.md 7.2): after clear() the map
    // is empty and fully usable again (insert/get/len).
    let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::new().unwrap();
    for i in 0..100u32 {
        map.insert(i, i).unwrap();
    }
    map.clear();
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
    assert_eq!(map.get(&5), None);

    for i in 0..100u32 {
        map.insert(i, i + 1).unwrap();
    }
    assert_eq!(map.len(), 100);
    for i in 0..100u32 {
        assert_eq!(map.get(&i), Some(&(i + 1)));
    }
}

// ==================== Load factor / growth regression tests ====================

#[test]
fn test_table_grows_at_the_configured_load_factor() {
    // Regression: resize was driven only by insertion failure, so the table
    // ran to 100% occupancy before growing. config.load_factor (0.75 by
    // default) was never read on the insert path. At full occupancy linear
    // probing costs O(capacity) per operation and a miss scans the entire
    // table, because get_standard only stops on an empty slot.
    let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_capacity(32).expect("create");
    let initial = map.capacity();
    assert_eq!(initial, 32);

    for i in 0..1000u32 {
        map.insert(i, i).expect("insert");
        let capacity = map.capacity();
        assert!(
            map.len() as f64 <= capacity as f64 * 0.75 + 1.0,
            "load {}/{} exceeds the configured load factor",
            map.len(),
            capacity
        );
    }

    assert!(map.capacity() > initial);
    for i in 0..1000u32 {
        assert_eq!(map.get(&i), Some(&i), "entry lost across rehashes");
    }
}

#[test]
fn test_insert_remove_churn_does_not_grow_the_table_forever() {
    // The load check counts tombstones (remove leaves one behind and they
    // still lengthen probes), so churn does trigger rehashes. Those must
    // rehash in place while the live set stays small, or a bounded working
    // set would double the table indefinitely.
    let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_capacity(64).expect("create");
    let iters = if cfg!(miri) { 500 } else { 100_000 };
    for i in 0..iters {
        map.insert(i, i).expect("insert");
        map.remove(&i);
    }

    assert_eq!(map.len(), 0);
    assert!(
        map.capacity() <= 128,
        "churn grew the table to {}",
        map.capacity()
    );
}

#[test]
fn test_c7_1_small_inline_fallback_grows_past_32_and_compacts_tombstones() {
    let config = ZiporaHashMapConfig::small_inline(4);
    let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_config(config).unwrap();
    assert_eq!(map.capacity(), 16);

    // Insert 200 elements (previously failed at i = 32 with "Hash table full in SmallInline fallback storage")
    for i in 0..200u32 {
        map.insert(i, i * 10).unwrap();
    }
    assert_eq!(map.len(), 200);
    assert!(
        map.capacity().is_power_of_two(),
        "capacity after SmallInline migration must be the active fallback table size, got {}",
        map.capacity()
    );

    for i in 0..200u32 {
        assert_eq!(map.get(&i), Some(&(i * 10)));
    }

    // Remove all but 4 elements and verify insert/remove churn on migrated SmallInline compacts tombstones
    for i in 4..200u32 {
        assert_eq!(map.remove(&i), Some(i * 10));
    }
    let cap_before_churn = map.capacity();
    let rehashes_before_churn = map.stats().rehashes;
    let iters = if cfg!(miri) { 200 } else { 5_000 };
    for i in 1000..(1000 + iters) {
        map.insert(i, i).unwrap();
        assert_eq!(map.remove(&i), Some(i));
    }
    assert_eq!(map.len(), 4);
    assert!(
        map.capacity() <= cap_before_churn,
        "churn on migrated SmallInline grew table from {} to {}",
        cap_before_churn,
        map.capacity()
    );
    // Tombstones count toward the fallback's load check, so the churn has to
    // purge them with at least one in-place rehash.
    assert!(
        map.stats().rehashes > rehashes_before_churn,
        "churn never purged the fallback table's tombstones"
    );
    for i in 0..4u32 {
        assert_eq!(map.get(&i), Some(&(i * 10)));
    }

    // Above 1.0 the load check lets the fallback table fill up completely,
    // so a full table has to be grown on demand before the insert is retried.
    let config = ZiporaHashMapConfig {
        load_factor: 1.5,
        ..ZiporaHashMapConfig::small_inline(4)
    };
    let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_config(config).unwrap();
    for i in 0..100u32 {
        map.insert(i, i).unwrap();
    }
    assert_eq!(map.len(), 100);
    for i in 0..100u32 {
        assert_eq!(map.get(&i), Some(&i));
    }
}

#[test]
fn test_c7_1_inline_churn_does_not_force_a_resize_after_migration() {
    // Guards the `occupied` count that the migrated fallback's load check now
    // reads: inline removals free their slot outright and do not subtract, so
    // the migration must reset the count to the entries it moves, or 100
    // inline insert/remove pairs would double the fallback table right after
    // migration.
    let config = ZiporaHashMapConfig::small_inline(4);
    let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_config(config).unwrap();
    for i in 0..100u32 {
        map.insert(i, i).unwrap();
        assert_eq!(map.remove(&i), Some(i));
    }
    for i in 0..18u32 {
        map.insert(i, i).unwrap();
    }
    // The 17th entry migrates into a 32-slot table; 18 entries fit in it.
    assert_eq!(map.capacity(), 32);
    assert_eq!(map.stats().rehashes, 0);
    for i in 0..18u32 {
        assert_eq!(map.get(&i), Some(&i));
    }
}

#[test]
fn test_c7_1_migrated_fallback_load_check_counts_the_moved_entries() {
    // The other side of the migration's reset: the count must include the 16
    // entries moved into the fallback table. With load factor 0.5 the 32-slot
    // table holds 17 entries right after migration, past its limit of 16, so
    // the next insert has to grow it; a count restarted from 0 would let the
    // table run 16 inserts past its load factor. (The `small_inline` preset
    // uses 1.0, where that growth coincides with the full-table retry.)
    let config = ZiporaHashMapConfig {
        load_factor: 0.5,
        ..ZiporaHashMapConfig::small_inline(4)
    };
    let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_config(config).unwrap();
    for i in 0..17u32 {
        map.insert(i, i).unwrap();
    }
    assert_eq!(map.capacity(), 32);
    assert_eq!(map.stats().rehashes, 0);
    map.insert(17, 17).unwrap();
    assert_eq!(map.capacity(), 64);
    assert_eq!(map.stats().rehashes, 1);
    for i in 0..18u32 {
        assert_eq!(map.get(&i), Some(&i));
    }
}

#[test]
fn test_c7_2_load_factor_that_is_not_positive_and_finite_is_replaced() {
    // `(capacity as f64 * load_factor) as usize` is 0 for 0.0, a negative value
    // or NaN, so the load check fired on every insert and each resize doubled
    // the table until an allocation failed.
    for bad_lf in [0.0, -0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let config = ZiporaHashMapConfig {
            load_factor: bad_lf,
            ..ZiporaHashMapConfig::default()
        };
        let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_config(config).unwrap();
        for i in 0..32u32 {
            map.insert(i, i + 100).unwrap();
        }
        assert_eq!(map.len(), 32);
        // Replaced by 0.75: 32 entries need a 64-slot table.
        assert!(
            map.capacity() <= 64,
            "load_factor {} grew the table to {}",
            bad_lf,
            map.capacity()
        );
        for i in 0..32u32 {
            assert_eq!(map.get(&i), Some(&(i + 100)));
        }
    }

    // Values >= 1.0 are kept: the table grows only once every slot is claimed.
    for lf in [1.0, 1.5] {
        let config = ZiporaHashMapConfig {
            load_factor: lf,
            ..ZiporaHashMapConfig::default()
        };
        let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::with_config(config).unwrap();
        for i in 0..100u32 {
            map.insert(i, i).unwrap();
        }
        assert_eq!(map.len(), 100);
        for i in 0..100u32 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }
}

#[test]
fn test_c7_2_initial_capacity_with_no_power_of_two_is_an_error() {
    // `next_power_of_two()` overflows above `1 << (usize::BITS - 1)`: it
    // panicked under overflow checks (the test profile) and returned 0
    // without them, silently building an empty table.
    for cap in [usize::MAX, (1usize << (usize::BITS - 1)) + 1] {
        let config = ZiporaHashMapConfig {
            storage_strategy: HashStorageStrategy::Standard {
                initial_capacity: cap,
                growth_factor: 2.0,
            },
            ..ZiporaHashMapConfig::default()
        };
        assert!(
            ZiporaHashMap::<u32, u32>::with_config(config).is_err(),
            "initial_capacity {} must be rejected",
            cap
        );
    }
    // `with_capacity` goes through the same path.
    assert!(ZiporaHashMap::<u32, u32>::with_capacity(usize::MAX).is_err());
}

/// Checks that `iter`, `keys`, `values`, `iter_mut` and `values_mut` visit
/// exactly the live entries of `map`, each once, and that the mutable
/// iterators write through.
fn check_iterators_see_exactly(map: &mut ZiporaHashMap<u32, u32>, expected: &[(u32, u32)]) {
    let mut seen: Vec<(u32, u32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
    seen.sort_unstable();
    assert_eq!(seen, expected, "iter");
    assert_eq!(map.keys().count(), expected.len(), "keys");
    assert_eq!(map.values().count(), expected.len(), "values");

    let mut mut_keys: Vec<u32> = Vec::new();
    for (&k, v) in map.iter_mut() {
        mut_keys.push(k);
        *v += 1_000;
    }
    mut_keys.sort_unstable();
    let expected_keys: Vec<u32> = expected.iter().map(|&(k, _)| k).collect();
    assert_eq!(mut_keys, expected_keys, "iter_mut");

    let mut visited = 0;
    for v in map.values_mut() {
        *v -= 1_000;
        visited += 1;
    }
    assert_eq!(visited, expected.len(), "values_mut");
    for &(k, v) in expected {
        assert_eq!(map.get(&k), Some(&v), "key {k} after iter_mut/values_mut");
    }
    assert_eq!(map.len(), expected.len());
}

#[test]
fn test_c7_iteration_skips_tombstones_in_every_storage_state() {
    // Coverage for the D4 iterators under tombstones. (Changing the map while
    // one of these iterators is alive does not compile: each one borrows the
    // map, shared or exclusive.)
    let live = |keys: std::ops::Range<u32>, removed: fn(u32) -> bool| -> Vec<(u32, u32)> {
        keys.filter(|&k| !removed(k)).map(|k| (k, k * 3)).collect()
    };

    // Standard table with tombstones.
    let mut map: ZiporaHashMap<u32, u32> = ZiporaHashMap::new().unwrap();
    for k in 0..100u32 {
        map.insert(k, k * 3).unwrap();
    }
    for k in (0..100u32).filter(|k| k % 3 == 0) {
        assert_eq!(map.remove(&k), Some(k * 3));
    }
    check_iterators_see_exactly(&mut map, &live(0..100, |k| k % 3 == 0));

    // SmallInline before migration, with freed inline slots.
    let mut map: ZiporaHashMap<u32, u32> =
        ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(4)).unwrap();
    for k in 0..10u32 {
        map.insert(k, k * 3).unwrap();
    }
    assert_eq!(map.remove(&3), Some(9));
    assert_eq!(map.remove(&7), Some(21));
    check_iterators_see_exactly(&mut map, &live(0..10, |k| k == 3 || k == 7));

    // The same map after migration, with tombstones in the fallback table.
    for k in 10..60u32 {
        map.insert(k, k * 3).unwrap();
    }
    assert!(map.capacity() > 16, "the map should have migrated");
    for k in (10..60u32).filter(|k| k % 4 == 0) {
        assert_eq!(map.remove(&k), Some(k * 3));
    }
    check_iterators_see_exactly(
        &mut map,
        &live(0..60, |k| k == 3 || k == 7 || (k >= 10 && k % 4 == 0)),
    );
}

#[test]
fn test_s47_r1_clear_panic_in_drop_no_double_free() {
    #[derive(Clone, Debug)]
    struct Bomb {
        _payload: String,
        drops: Arc<AtomicUsize>,
        panic_at: usize,
    }
    impl Drop for Bomb {
        fn drop(&mut self) {
            let n = self.drops.fetch_add(1, Ordering::SeqCst) + 1;
            if n == self.panic_at {
                panic!("boom on drop {}", n);
            }
        }
    }

    // Standard storage
    let drops = Arc::new(AtomicUsize::new(0));
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut m: ZiporaHashMap<u64, Bomb> = ZiporaHashMap::new().unwrap();
        for i in 0..8u64 {
            m.insert(
                i,
                Bomb {
                    _payload: format!("hello-world-{}", i),
                    drops: Arc::clone(&drops),
                    panic_at: 3,
                },
            )
            .unwrap();
        }
        m.clear();
    }));
    assert!(res.is_err());
    assert_eq!(
        drops.load(Ordering::SeqCst),
        8,
        "every Bomb must be dropped exactly once across clear() panic + unwind drop"
    );

    // SmallInline storage
    let drops_inline = Arc::new(AtomicUsize::new(0));
    let res_inline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut m: ZiporaHashMap<u64, Bomb> =
            ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(4)).unwrap();
        for i in 0..8u64 {
            m.insert(
                i,
                Bomb {
                    _payload: format!("inline-world-{}", i),
                    drops: Arc::clone(&drops_inline),
                    panic_at: 3,
                },
            )
            .unwrap();
        }
        m.clear();
    }));
    assert!(res_inline.is_err());
    assert_eq!(
        drops_inline.load(Ordering::SeqCst),
        8,
        "every inline Bomb must be dropped exactly once across clear() panic + unwind drop"
    );
}

#[test]
fn test_s47_r6_resize_panic_in_hash_preserves_entries() {
    use std::hash::Hasher;

    #[derive(Clone, Debug)]
    struct PKey {
        k: u64,
        calls: Arc<AtomicUsize>,
        panic_on: Arc<AtomicUsize>,
    }
    impl PartialEq for PKey {
        fn eq(&self, other: &Self) -> bool {
            self.k == other.k
        }
    }
    impl Eq for PKey {}
    impl Hash for PKey {
        fn hash<H: Hasher>(&self, state: &mut H) {
            let c = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if c == self.panic_on.load(Ordering::SeqCst) {
                panic!("hash panic at {}", c);
            }
            self.k.hash(state);
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let panic_on = Arc::new(AtomicUsize::new(usize::MAX));
    let mut m: ZiporaHashMap<PKey, String> = ZiporaHashMap::new().unwrap();
    for i in 0..24u64 {
        m.insert(
            PKey {
                k: i,
                calls: Arc::clone(&calls),
                panic_on: Arc::clone(&panic_on),
            },
            format!("val-{}", i),
        )
        .unwrap();
    }
    let cur = calls.load(Ordering::SeqCst);
    panic_on.store(cur + 8, Ordering::SeqCst);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        m.insert(
            PKey {
                k: 99,
                calls: Arc::clone(&calls),
                panic_on: Arc::clone(&panic_on),
            },
            "val-99".to_string(),
        )
        .unwrap();
    }));
    panic_on.store(usize::MAX, Ordering::SeqCst);
    assert!(res.is_err());
    assert_eq!(m.len(), 24);
    assert_eq!(m.iter().count(), 24);
    for i in 0..24u64 {
        let probe = PKey {
            k: i,
            calls: Arc::clone(&calls),
            panic_on: Arc::clone(&panic_on),
        };
        assert_eq!(
            m.get(&probe).map(|s| s.as_str()),
            Some(format!("val-{}", i).as_str()),
            "entry {} lost after panicking resize",
            i
        );
    }
}

#[test]
fn test_s47_r11_custom_link_type_with_zero_delmark() {
    use crate::hash_map::{GoldHashMap, LinkType};

    #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Default, Debug)]
    struct L0(u32);

    impl LinkType for L0 {
        const MAX: Self = L0(u32::MAX - 1);
        const DELMARK: Self = L0(0);
        const TAIL: Self = L0(u32::MAX);

        fn as_usize(self) -> usize {
            (self.0 - 1) as usize
        }

        fn from_usize(v: usize) -> Option<Self> {
            u32::try_from(v + 1).ok().filter(|&x| x < u32::MAX).map(L0)
        }
    }

    assert!(!L0::DELMARK.is_valid());
    assert!(!L0::TAIL.is_valid());
    assert!(L0::MAX.is_valid());

    let mut gm: GoldHashMap<u64, u64, L0> = GoldHashMap::new();
    for i in 0..20u64 {
        gm.insert(i, i * 10).unwrap();
    }
    for i in 0..10u64 {
        assert_eq!(gm.remove(&i).unwrap(), Some(i * 10));
    }
    for i in 0..10u64 {
        gm.insert(i, i * 100).unwrap();
    }
    let mut pairs: Vec<(u64, u64)> = gm.iter().map(|(&k, &v)| (k, v)).collect();
    pairs.sort_unstable();
    assert_eq!(pairs.len(), 20);
    for i in 0..10u64 {
        assert_eq!(gm.get(&i), Some(&(i * 100)));
    }
    for i in 10..20u64 {
        assert_eq!(gm.get(&i), Some(&(i * 10)));
    }
}

#[test]
fn test_s48_r1_small_inline_update_key_drop_panic() {
    use std::hash::Hasher;

    #[derive(Clone, Debug)]
    struct KeyBomb {
        k: u64,
        armed: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }
    impl PartialEq for KeyBomb {
        fn eq(&self, other: &Self) -> bool {
            self.k == other.k
        }
    }
    impl Eq for KeyBomb {}
    impl Hash for KeyBomb {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.k.hash(state);
        }
    }
    impl Drop for KeyBomb {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
            if self.armed.swap(0, Ordering::SeqCst) == 1 && !std::thread::panicking() {
                panic!("key drop panic for {}", self.k);
            }
        }
    }

    let armed = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut m: ZiporaHashMap<KeyBomb, String> =
        ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(16)).unwrap();
    for i in 0..4u64 {
        m.insert(
            KeyBomb {
                k: i,
                armed: Arc::clone(&armed),
                drops: Arc::clone(&drops),
            },
            format!("v{i}"),
        )
        .unwrap();
    }
    armed.store(1, Ordering::SeqCst);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = m.insert(
            KeyBomb {
                k: 2,
                armed: Arc::clone(&armed),
                drops: Arc::clone(&drops),
            },
            "updated".to_string(),
        );
    }));
    assert!(res.is_err());
    assert_eq!(m.len(), 4);
    let probe = KeyBomb {
        k: 2,
        armed: Arc::clone(&armed),
        drops: Arc::clone(&drops),
    };
    assert_eq!(m.get(&probe).map(|s| s.as_str()), Some("updated"));
    drop(probe);
    drop(m);
    assert_eq!(drops.load(Ordering::SeqCst), 6);
}

#[test]
fn test_s48_r3_clear_panic_then_use() {
    #[derive(Clone, Debug)]
    struct Bomb {
        id: usize,
        panic_id: usize,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Bomb {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
            if self.id == self.panic_id && !std::thread::panicking() {
                panic!("drop panic on {}", self.id);
            }
        }
    }

    // 1. Standard table: clear() panics, then map is reused for new inserts
    let drops = Arc::new(AtomicUsize::new(0));
    let mut m: ZiporaHashMap<u64, Bomb> = ZiporaHashMap::with_capacity(32).unwrap();
    for i in 0..20 {
        m.insert(
            i as u64,
            Bomb {
                id: i,
                panic_id: 5,
                drops: Arc::clone(&drops),
            },
        )
        .unwrap();
    }
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        m.clear();
    }));
    assert!(res.is_err());
    assert_eq!(m.len(), 0);
    assert_eq!(m.iter().count(), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 20);

    for i in 0..40 {
        m.insert(
            i as u64 + 100,
            Bomb {
                id: 999,
                panic_id: usize::MAX,
                drops: Arc::clone(&drops),
            },
        )
        .unwrap();
    }
    assert_eq!(m.len(), 40);
    assert_eq!(m.iter().count(), 40);
    drop(m);
    assert_eq!(drops.load(Ordering::SeqCst), 60);

    // 2. SmallInline table (unmigrated and migrated): clear() panics, then reused
    for count in [8usize, 24usize] {
        let drops_si = Arc::new(AtomicUsize::new(0));
        let mut m_si: ZiporaHashMap<u64, Bomb> =
            ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(16)).unwrap();
        for i in 0..count {
            m_si.insert(
                i as u64,
                Bomb {
                    id: i,
                    panic_id: 3,
                    drops: Arc::clone(&drops_si),
                },
            )
            .unwrap();
        }
        let res_si = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            m_si.clear();
        }));
        assert!(res_si.is_err());
        assert_eq!(m_si.len(), 0);
        assert_eq!(m_si.iter().count(), 0);
        assert_eq!(drops_si.load(Ordering::SeqCst), count);

        for i in 0..10 {
            m_si.insert(
                i as u64 + 500,
                Bomb {
                    id: 999,
                    panic_id: usize::MAX,
                    drops: Arc::clone(&drops_si),
                },
            )
            .unwrap();
        }
        assert_eq!(m_si.len(), 10);
        drop(m_si);
        assert_eq!(drops_si.load(Ordering::SeqCst), count + 10);
    }
}

#[test]
fn test_s48_r9_large_load_factor_growth() {
    let cfg = ZiporaHashMapConfig {
        initial_capacity: 32,
        load_factor: 3.0,
        ..Default::default()
    };
    let mut m: ZiporaHashMap<u64, u64> = ZiporaHashMap::with_config(cfg).unwrap();
    for i in 0..100u64 {
        m.insert(i, i * 10).unwrap();
    }
    assert_eq!(m.len(), 100);
    for i in 0..100u64 {
        assert_eq!(m.get(&i), Some(&(i * 10)));
    }
}
