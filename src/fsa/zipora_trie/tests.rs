use super::*;
use crate::succinct::RankSelectInterleaved256;

#[test]
fn test_unified_trie_creation() {
    let trie: ZiporaTrie = ZiporaTrie::new();
    assert_eq!(trie.len(), 0);
    assert!(trie.is_empty());
}

#[test]
fn test_cache_optimized_config() {
    let trie: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::cache_optimized());
    assert!(trie.is_cache_optimized());
}

#[test]
fn test_space_optimized_insert_returns_not_supported() {
    let mut trie: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized());
    assert_eq!(trie.len(), 0);
    let err = trie.insert(b"hello").unwrap_err();
    assert!(
        matches!(err, crate::error::ZiporaError::NotSupported { .. }),
        "LOUDS insert must return NotSupported, got: {err}",
    );
    assert!(!trie.contains(b"hello"));
}

#[test]
fn test_string_specialized_insert_returns_not_supported() {
    let mut trie: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::string_specialized());
    assert_eq!(trie.len(), 0);
    let err = trie.insert(b"hello").unwrap_err();
    assert!(
        matches!(err, crate::error::ZiporaError::NotSupported { .. }),
        "CriticalBit insert must return NotSupported, got: {err}",
    );
    assert!(!trie.contains(b"hello"));
}

#[test]
fn test_implemented_strategies_still_work() {
    // DoubleArray (default)
    let mut da: ZiporaTrie = ZiporaTrie::new();
    da.insert(b"hello").unwrap();
    assert!(da.contains(b"hello"));

    // Patricia
    let mut pat: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::cache_optimized());
    pat.insert(b"hello").unwrap();
    assert!(pat.contains(b"hello"));
}

#[test]
fn test_double_array_insert_contains() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    // Default is now DoubleArray
    trie.insert(b"hello").unwrap();
    assert_eq!(trie.len(), 1);
    assert!(trie.contains(b"hello"));
    assert!(!trie.contains(b"world"));

    trie.insert(b"world").unwrap();
    assert_eq!(trie.len(), 2);
    assert!(trie.contains(b"hello"));
    assert!(trie.contains(b"world"));

    trie.insert(b"help").unwrap();
    assert_eq!(trie.len(), 3);
    assert!(trie.contains(b"help"));
    assert!(trie.contains(b"hello"));

    // Duplicate insert should not increase len
    trie.insert(b"hello").unwrap();
    assert_eq!(trie.len(), 3);
}

#[test]
fn test_double_array_keys() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    trie.insert(b"apple").unwrap();
    trie.insert(b"app").unwrap();
    trie.insert(b"banana").unwrap();

    let mut keys = trie.keys();
    keys.sort();
    assert_eq!(keys.len(), 3);
    assert_eq!(keys[0], b"app");
    assert_eq!(keys[1], b"apple");
    assert_eq!(keys[2], b"banana");
}

#[test]
fn test_double_array_prefix_with_empty_key() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    trie.insert(b"").unwrap();
    trie.insert(b"a").unwrap();
    trie.insert(b"ab").unwrap();
    trie.insert(b"abc").unwrap();
    trie.insert(b"abd").unwrap();
    trie.insert(b"b").unwrap();

    let all = trie.keys_with_prefix(b"");
    assert_eq!(
        all.len(),
        6,
        "keys_with_prefix('') should return all 6 keys"
    );
}

#[test]
fn test_double_array_empty_key() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    trie.insert(b"").unwrap();
    trie.insert(b"a").unwrap();
    trie.insert(b"ab").unwrap();

    assert_eq!(trie.len(), 3);
    assert!(trie.contains(b""));
    assert!(trie.contains(b"a"));
    assert!(trie.contains(b"ab"));

    let mut keys = trie.keys();
    keys.sort();
    assert_eq!(keys.len(), 3, "Should have 3 keys including empty");
    assert_eq!(keys[0], b"");
    assert_eq!(keys[1], b"a");
    assert_eq!(keys[2], b"ab");
}

// --- Coverage tests for each improvement ---

/// Issue #1: Lazy stats — verify stats() works correctly after inserts
#[test]
fn test_lazy_stats() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    for i in 0..100 {
        trie.insert(format!("key{:03}", i).as_bytes()).unwrap();
    }
    assert_eq!(trie.len(), 100);
    let stats = trie.stats();
    assert_eq!(stats.num_keys, 100);
    assert!(stats.memory_usage > 0);
    assert!(stats.num_states > 0);
}

/// Issue #2: No double traversal — duplicate insert does not increase len
#[test]
fn test_no_double_traversal_duplicate() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    trie.insert(b"abc").unwrap();
    trie.insert(b"abc").unwrap();
    trie.insert(b"abc").unwrap();
    assert_eq!(trie.len(), 1);

    trie.insert(b"def").unwrap();
    trie.insert(b"def").unwrap();
    assert_eq!(trie.len(), 2);
}

/// Issue #3: Compact PatriciaNode — Patricia still works with compact children
#[test]
fn test_patricia_compact_node() {
    let config = ZiporaTrieConfig {
        trie_strategy: crate::fsa::TrieStrategy::Patricia {
            max_path_length: 64,
            compression_threshold: 4,
            adaptive_compression: true,
        },
        ..ZiporaTrieConfig::default()
    };
    let mut trie: ZiporaTrie = ZiporaTrie::with_config(config);
    trie.insert(b"hello").unwrap();
    trie.insert(b"help").unwrap();
    trie.insert(b"world").unwrap();

    assert_eq!(trie.len(), 3);
    assert!(trie.contains(b"hello"));
    assert!(trie.contains(b"help"));
    assert!(trie.contains(b"world"));
    assert!(!trie.contains(b"hel"));
}

/// Issue #4/#5: find_free_base + relocate — many inserts don't panic
#[test]
fn test_find_free_base_many_inserts() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    // Insert many keys to stress find_free_base and relocation
    for i in 0..500 {
        trie.insert(format!("key_{:04}", i).as_bytes()).unwrap();
    }
    assert_eq!(trie.len(), 500);
    // Verify random lookups
    assert!(trie.contains(b"key_0000"));
    assert!(trie.contains(b"key_0250"));
    assert!(trie.contains(b"key_0499"));
    assert!(!trie.contains(b"key_0500"));
}

/// The relocation probe is anchored on the lowest symbol being placed and
/// walks free slots, rather than incrementing the base one at a time.
///
/// A four-letter alphabet makes almost every interior node branch, so states
/// keep outgrowing their base and have to be relocated — which is what
/// exercises that arithmetic. The sequential `key_%04d` set above barely
/// relocates at all.
#[test]
fn test_double_array_relocation_with_branching_keys() {
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    let mut keys: Vec<Vec<u8>> = (0..3000)
        .map(|_| {
            (0..12)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    b'a' + (x % 4) as u8
                })
                .collect()
        })
        .collect();
    keys.sort();
    keys.dedup();

    let mut trie: ZiporaTrie = ZiporaTrie::new();
    for key in &keys {
        trie.insert(key).unwrap();
    }

    for key in &keys {
        assert!(
            trie.contains(key),
            "key {:?} lost during relocation",
            String::from_utf8_lossy(key)
        );
    }
    assert!(!trie.contains(b"ZZZZZZZZ"));
}

/// Issue #6: Amortized growth — large insert doesn't OOM or take forever
#[test]
fn test_amortized_growth() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    // 1000 inserts should complete quickly with 1.5x growth
    for i in 0..1000 {
        trie.insert(format!("{:04}", i).as_bytes()).unwrap();
    }
    assert_eq!(trie.len(), 1000);
}

/// Issue #8: TrieMap — key-value storage
#[test]
fn test_trie_map() {
    let mut map = ZiporaTrieMap::<u32, RankSelectInterleaved256>::new();
    map.insert(b"hello", 42).unwrap();
    map.insert(b"world", 100).unwrap();
    map.insert(b"help", 7).unwrap();

    assert_eq!(map.get(b"hello"), Some(42));
    assert_eq!(map.get(b"world"), Some(100));
    assert_eq!(map.get(b"help"), Some(7));
    assert_eq!(map.get(b"missing"), None);
    assert_eq!(map.len(), 3);

    // Update existing key
    let prev = map.insert(b"hello", 99).unwrap();
    assert_eq!(prev, Some(42));
    assert_eq!(map.get(b"hello"), Some(99));
    assert_eq!(map.len(), 3); // len unchanged
}

/// TrieMap values must survive double-array state relocation: inserting enough
/// keys forces base/check collisions that move existing terminal states, and
/// the state-indexed value slots must be remapped along with them.
#[test]
fn test_trie_map_values_survive_relocation() {
    let mut map = ZiporaTrieMap::<u32, RankSelectInterleaved256>::new();
    const N: u32 = 300;
    for i in 0..N {
        map.insert(format!("key_{:04}", i).as_bytes(), i).unwrap();
    }
    assert_eq!(map.len(), N as usize);
    for i in 0..N {
        let key = format!("key_{:04}", i);
        assert_eq!(map.get(key.as_bytes()), Some(i), "value lost for {}", key);
    }

    // Overwrites must still return the (relocated) previous value
    let prev = map.insert(b"key_0000", 9999).unwrap();
    assert_eq!(prev, Some(0));
    assert_eq!(map.get(b"key_0000"), Some(9999));
}

/// Issue #9: Bulk construction
#[test]
fn test_build_from_sorted() {
    let keys: Vec<&[u8]> = vec![b"apple", b"application", b"apply", b"banana", b"band"];
    let trie: ZiporaTrie =
        ZiporaTrie::build_from_sorted(&keys, ZiporaTrieConfig::default()).unwrap();

    assert_eq!(trie.len(), 5);
    assert!(trie.contains(b"apple"));
    assert!(trie.contains(b"application"));
    assert!(trie.contains(b"apply"));
    assert!(trie.contains(b"banana"));
    assert!(trie.contains(b"band"));
    assert!(!trie.contains(b"ban"));
}

/// Issue #10: Default is DoubleArray
#[test]
fn test_default_is_double_array() {
    let config = ZiporaTrieConfig::default();
    assert!(matches!(
        config.trie_strategy,
        crate::fsa::TrieStrategy::DoubleArray { .. }
    ));
}

/// DoubleArray remove support
#[test]
fn test_double_array_remove() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    trie.insert(b"hello").unwrap();
    trie.insert(b"world").unwrap();
    assert_eq!(trie.len(), 2);

    assert!(trie.remove(b"hello").unwrap());
    assert_eq!(trie.len(), 1);
    assert!(!trie.contains(b"hello"));
    assert!(trie.contains(b"world"));

    // Remove non-existent key
    assert!(!trie.remove(b"missing").unwrap());
    assert_eq!(trie.len(), 1);
}

/// DoubleArray lookup_node_id + restore_string roundtrip
#[test]
fn test_double_array_node_id_roundtrip() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    trie.insert(b"hello").unwrap();
    trie.insert(b"world").unwrap();

    let node_id = trie.lookup_node_id(b"hello").expect("should find hello");
    let restored = trie.restore_string(node_id).expect("should restore");
    assert_eq!(restored, b"hello");

    let node_id2 = trie.lookup_node_id(b"world").expect("should find world");
    let restored2 = trie.restore_string(node_id2).expect("should restore");
    assert_eq!(restored2, b"world");

    assert!(trie.lookup_node_id(b"missing").is_none());
}

/// Regression: Patricia `remove` unlinked nodes but never recycled their ids,
/// so insert/remove churn of a single key grew `state_count()` without bound
/// while `len()` stayed constant (1000 rounds of one 13-byte key: 5 -> 13,005).
/// Removed node ids must go on a free list and be reused by later inserts.
#[test]
fn test_patricia_remove_recycles_nodes() {
    let config = ZiporaTrieConfig {
        trie_strategy: crate::fsa::TrieStrategy::Patricia {
            max_path_length: 64,
            compression_threshold: 4,
            adaptive_compression: true,
        },
        ..ZiporaTrieConfig::default()
    };
    let mut trie: ZiporaTrie = ZiporaTrie::with_config(config);

    let key = b"churn-key-abc"; // 13 bytes
    trie.insert(key).unwrap();
    let after_first_insert = trie.state_count();

    for _ in 0..1000 {
        assert!(trie.remove(key).unwrap());
        assert!(!trie.contains(key));
        trie.insert(key).unwrap();
        assert!(trie.contains(key));
    }

    assert_eq!(trie.len(), 1);
    assert!(
        trie.state_count() <= after_first_insert + 1,
        "state_count leaked: {} after churn vs {} after first insert",
        trie.state_count(),
        after_first_insert
    );

    // Recycled nodes must not carry stale children/finality into new keys.
    assert!(trie.remove(key).unwrap());
    trie.insert(b"other").unwrap();
    assert!(trie.contains(b"other"));
    assert!(!trie.contains(key));
    assert!(!trie.contains(b"churn"));
    assert!(!trie.contains(b"oth"));
    assert_eq!(trie.len(), 1);
}

/// Keys with no shared structure, so nearly every insert allocates a fresh
/// state and the free-slot search is exercised on every step.
fn pseudo_random_keys(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            let mut x = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let len = 8 + (x % 9) as usize;
            (0..len)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    b'a' + (x % 26) as u8
                })
                .collect()
        })
        .collect()
}

/// Construction must scale (near-)linearly in the number of keys. The old
/// free-slot probe walked the densely occupied prefix on every relocation, so
/// random keys cost ~4x per doubling (20k keys took 5.9 s in release). A
/// linear build finishes this well inside the bound even in debug.
#[test]
fn test_double_array_random_key_build_is_not_quadratic() {
    let keys = pseudo_random_keys(20_000);
    let unique: std::collections::BTreeSet<&Vec<u8>> = keys.iter().collect();

    let start = std::time::Instant::now();
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    for k in &keys {
        trie.insert(k).unwrap();
    }
    let elapsed = start.elapsed();

    assert_eq!(trie.len(), unique.len());
    for k in &keys {
        assert!(trie.contains(k));
    }
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "double-array build degraded to quadratic probing: 20k keys took {elapsed:?}"
    );
}

/// The free-list allocator relocates far more states than a handful of
/// hand-picked keys exercise; every value must still follow its state.
#[test]
fn test_trie_map_values_survive_random_key_relocations() {
    let keys = pseudo_random_keys(5_000);
    let mut map = ZiporaTrieMap::<u32, RankSelectInterleaved256>::new();
    for (i, k) in keys.iter().enumerate() {
        map.insert(k, i as u32).unwrap();
    }
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(map.get(k), Some(i as u32), "value lost for key #{i}");
    }
}

/// A double array has exactly one state per claimed cell and one incoming
/// transition per non-root state. Free cells carry non-zero link words and
/// must not be counted as either.
#[test]
fn test_double_array_stats_count_states_not_free_cells() {
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    for k in [b"a".as_slice(), b"b", b"c"] {
        trie.insert(k).unwrap();
    }
    let stats = trie.stats();
    assert_eq!(stats.num_states, 4, "root + 3 children");
    assert_eq!(stats.num_transitions, 3);
}

/// `shrink_to_fit` must drop the free tail the 1.5x growth policy leaves
/// behind, and the trie must stay insertable afterwards: the dropped cells
/// have to come off the free list, not just off the array.
#[test]
fn test_double_array_shrink_to_fit_truncates_free_tail_and_stays_insertable() {
    fn cells(t: &ZiporaTrie) -> usize {
        match &t.storage {
            super::storage::TrieStorage::DoubleArray { base, .. } => base.len(),
            _ => unreachable!(),
        }
    }
    let keys = pseudo_random_keys(2_000);
    let mut trie: ZiporaTrie = ZiporaTrie::new();
    for k in &keys[..1_000] {
        trie.insert(k).unwrap();
    }

    let before = cells(&trie);
    trie.shrink_to_fit();
    let after = cells(&trie);
    assert!(after < before, "free tail not dropped: {before} -> {after}");
    assert!(!trie.is_free_double_array((after - 1) as u32), "last cell must be occupied");

    for k in &keys[1_000..] {
        trie.insert(k).unwrap();
    }
    for k in &keys {
        assert!(trie.contains(k));
    }
}

#[test]
fn test_double_array_transitions_on_terminal_and_free_states() {
    use crate::fsa::traits::FiniteStateAutomaton;

    let mut trie: ZiporaTrie = ZiporaTrie::new();
    trie.insert(b"").unwrap();
    trie.insert(b"app").unwrap();
    trie.insert(b"apple").unwrap();

    // Root 0 is terminal (empty key) AND has child 'a'
    assert!(trie.is_final(0));
    let root_transitions = trie.transitions(0);
    assert_eq!(
        root_transitions,
        vec![(b'a', trie.transition(0, b'a').unwrap())],
        "terminal root must still report its outgoing transition on 'a'"
    );

    // State for "app" is terminal AND has child 'l' ("apple")
    let s_app = trie.lookup_node_id(b"app").unwrap();
    assert!(trie.is_final(s_app));
    let app_transitions = trie.transitions(s_app);
    assert_eq!(
        app_transitions,
        vec![(b'l', trie.transition(s_app, b'l').unwrap())],
        "terminal prefix state 'app' must report its child transition 'l'"
    );

    // Free states and out-of-bounds states must report no transitions and non-final
    for s in 1..256u32 {
        if trie.is_free_double_array(s) {
            assert!(!trie.is_final(s));
            assert_eq!(trie.transition(s, b'a'), None);
            assert!(trie.transitions(s).is_empty());
            assert_eq!(trie.restore_string(s), None);
        }
    }
    assert!(!trie.is_final(u32::MAX));
    assert_eq!(trie.transition(u32::MAX, b'a'), None);
    assert!(trie.transitions(u32::MAX).is_empty());
    assert_eq!(trie.restore_string(u32::MAX), None);
}

#[test]
fn test_compressed_sparse_strategy_full_fsa_and_map_consistency() {
    use crate::fsa::traits::FiniteStateAutomaton;

    let mut trie: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::sparse_optimized());
    for word in [b"app".as_slice(), b"apple", b"application", b"banana"] {
        trie.insert(word).unwrap();
    }
    assert_eq!(trie.len(), 4);

    let mut keys = trie.keys();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            b"app".to_vec(),
            b"apple".to_vec(),
            b"application".to_vec(),
            b"banana".to_vec()
        ]
    );

    let mut app_keys = trie.keys_with_prefix(b"app");
    app_keys.sort();
    assert_eq!(
        app_keys,
        vec![
            b"app".to_vec(),
            b"apple".to_vec(),
            b"application".to_vec()
        ]
    );

    let stats = trie.stats();
    assert!(stats.num_states > 0);
    assert!(stats.num_transitions > 0);

    let cloned = trie.clone();
    assert_eq!(cloned.len(), 4);
    assert!(cloned.contains(b"application"));

    for word in [b"app".as_slice(), b"apple", b"application", b"banana"] {
        let mut st = trie.root();
        for &b in word {
            let next = trie.transition(st, b).expect("missing FSA transition");
            assert!(
                trie.transitions(st).contains(&(b, next)),
                "transitions(st) must include ({b}, {next})"
            );
            st = next;
        }
        assert!(trie.is_final(st));
        assert_eq!(trie.lookup_node_id(word), Some(st));
        assert_eq!(trie.restore_string(st).as_deref(), Some(word));
    }

    assert!(!trie.is_final(u32::MAX));
    assert_eq!(trie.transition(u32::MAX, b'a'), None);
    assert!(trie.transitions(u32::MAX).is_empty());
    assert_eq!(trie.restore_string(u32::MAX), None);

    // ZiporaTrieMap with CompressedSparse must survive node splits/relocations
    let mut map: ZiporaTrieMap<u64, RankSelectInterleaved256> =
        ZiporaTrieMap::with_config(ZiporaTrieConfig::sparse_optimized());
    for i in 0..200u64 {
        let key = format!("sparse_key_{:04}", i);
        assert_eq!(map.insert(key.as_bytes(), i * 10).unwrap(), None);
    }
    assert_eq!(map.len(), 200);
    for i in 0..200u64 {
        let key = format!("sparse_key_{:04}", i);
        assert_eq!(map.get(key.as_bytes()), Some(i * 10));
    }
    assert_eq!(map.insert(b"sparse_key_0042", 999_999).unwrap(), Some(420));
    assert_eq!(map.get(b"sparse_key_0042"), Some(999_999));
}

#[test]
fn test_unsupported_strategies_reject_insert_and_get_node_id_and_remove() {
    for config in [
        ZiporaTrieConfig::space_optimized(),
        ZiporaTrieConfig::string_specialized(),
    ] {
        let mut trie: ZiporaTrie = ZiporaTrie::with_config(config);
        assert!(matches!(
            trie.insert_and_get_node_id(b"hello"),
            Err(crate::error::ZiporaError::NotSupported { .. })
        ));
        assert_eq!(trie.len(), 0);
        assert!(matches!(
            trie.remove(b"hello"),
            Err(crate::error::ZiporaError::NotSupported { .. })
        ));
    }
}

#[test]
fn test_compressed_sparse_transition_is_constant_time_and_rejects_overflow_slots() {
    use crate::fsa::traits::FiniteStateAutomaton;
    // Regression test:
    // 1. Walking 200 keys × 12 steps across a 20,000-key CompressedSparse trie
    //    must complete in milliseconds (O(1) per step via `NodeView::is_well_formed`),
    //    not seconds (O(n) whole-trie DFS per step).
    // 2. State IDs with `slot > CSPP_MAX_SLOT` must be rejected cleanly without wrapping.
    const N_KEYS: usize = if cfg!(miri) { 100 } else { 20_000 };
    let mut trie: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::sparse_optimized());
    for i in 0..N_KEYS {
        let k = format!("key_{:08}", i);
        trie.insert(k.as_bytes()).unwrap();
    }

    let sample = N_KEYS.min(200);
    let start = std::time::Instant::now();
    for i in 0..sample {
        let k = format!("key_{:08}", i);
        let mut s = trie.root();
        for &b in k.as_bytes() {
            s = trie.transition(s, b).expect("valid transition");
        }
        assert!(trie.is_final(s));
        assert_eq!(trie.lookup_node_id(k.as_bytes()), Some(s));
    }
    if !cfg!(miri) {
        assert!(
            start.elapsed() < std::time::Duration::from_millis(50),
            "200 × 12 CompressedSparse transitions took {:?} (expected < 50ms O(1) steps)",
            start.elapsed()
        );
    }

    // Out-of-range state IDs must return false/None
    assert!(!trie.is_final(u32::MAX));
    assert_eq!(trie.transition(u32::MAX, b'k'), None);
    assert!(trie.transitions(u32::MAX).is_empty());
}


