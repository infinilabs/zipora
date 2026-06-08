use super::iterators::*;
use super::map::*;
use super::state::*;
use super::trie::*;

// test module mirrors the file's own name by convention
#[allow(clippy::module_inception)]
mod tests {
    use super::super::iterators::*;
    use super::super::map::*;
    use super::super::state::*;
    use super::super::trie::*;
    use super::*;

    #[test]
    fn test_basic_insert_contains() {
        let mut t = DoubleArrayTrie::new();
        assert!(t.insert(b"hello").unwrap());
        assert!(t.insert(b"help").unwrap());
        assert!(t.insert(b"world").unwrap());
        assert_eq!(t.len(), 3);

        assert!(t.contains(b"hello"));
        assert!(t.contains(b"help"));
        assert!(t.contains(b"world"));
        assert!(!t.contains(b"hel"));
        assert!(!t.contains(b"hell"));
        assert!(!t.contains(b"worlds"));
    }

    #[test]
    fn test_duplicate_insert() {
        let mut t = DoubleArrayTrie::new();
        assert!(t.insert(b"abc").unwrap());
        assert!(!t.insert(b"abc").unwrap());
        assert!(!t.insert(b"abc").unwrap());
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_empty_key() {
        let mut t = DoubleArrayTrie::new();
        assert!(t.insert(b"").unwrap());
        assert!(t.contains(b""));
        assert!(t.insert(b"a").unwrap());
        assert_eq!(t.len(), 2);

        let mut keys = t.keys();
        keys.sort();
        assert_eq!(keys, vec![vec![], vec![b'a']]);
    }

    #[test]
    fn test_remove() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"hello").unwrap();
        t.insert(b"world").unwrap();
        assert_eq!(t.len(), 2);

        assert!(t.remove(b"hello"));
        assert_eq!(t.len(), 1);
        assert!(!t.contains(b"hello"));
        assert!(t.contains(b"world"));

        assert!(!t.remove(b"missing"));
    }

    #[test]
    fn test_restore_key() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"hello").unwrap();
        t.insert(b"world").unwrap();

        let state = t.lookup_state(b"hello").unwrap();
        assert_eq!(t.restore_key(state).unwrap(), b"hello");

        let state2 = t.lookup_state(b"world").unwrap();
        assert_eq!(t.restore_key(state2).unwrap(), b"world");
    }

    #[test]
    fn test_keys() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"apple").unwrap();
        t.insert(b"app").unwrap();
        t.insert(b"banana").unwrap();

        let mut keys = t.keys();
        keys.sort();
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], b"app");
        assert_eq!(keys[1], b"apple");
        assert_eq!(keys[2], b"banana");
    }

    #[test]
    fn test_keys_with_prefix() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"").unwrap();
        t.insert(b"a").unwrap();
        t.insert(b"ab").unwrap();
        t.insert(b"abc").unwrap();
        t.insert(b"abd").unwrap();
        t.insert(b"b").unwrap();

        let all = t.keys_with_prefix(b"");
        assert_eq!(all.len(), 6);

        let a = t.keys_with_prefix(b"a");
        assert_eq!(a.len(), 4); // a, ab, abc, abd

        let ab = t.keys_with_prefix(b"ab");
        assert_eq!(ab.len(), 3); // ab, abc, abd

        let none = t.keys_with_prefix(b"xyz");
        assert_eq!(none.len(), 0);
    }

    #[test]
    fn test_many_inserts() {
        let mut t = DoubleArrayTrie::new();
        for i in 0..1000 {
            t.insert(format!("key_{:04}", i).as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 1000);
        assert!(t.contains(b"key_0000"));
        assert!(t.contains(b"key_0500"));
        assert!(t.contains(b"key_0999"));
        assert!(!t.contains(b"key_1000"));
    }

    #[test]
    fn test_state_move() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"abc").unwrap();

        let s1 = t.state_move(0, b'a');
        assert_ne!(s1, NIL_STATE);
        let s2 = t.state_move(s1, b'b');
        assert_ne!(s2, NIL_STATE);
        let s3 = t.state_move(s2, b'c');
        assert_ne!(s3, NIL_STATE);
        assert!(t.is_term(s3));

        assert_eq!(t.state_move(0, b'z'), NIL_STATE);
    }

    #[test]
    fn test_build_from_sorted() {
        let keys: Vec<&[u8]> = vec![b"apple", b"application", b"apply", b"banana", b"band"];
        let t = DoubleArrayTrie::build_from_sorted(&keys).unwrap();

        assert_eq!(t.len(), 5);
        for key in &keys {
            assert!(t.contains(key), "missing: {:?}", std::str::from_utf8(key));
        }
    }

    #[test]
    fn test_for_each_child() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"ab").unwrap();
        t.insert(b"ac").unwrap();
        t.insert(b"ad").unwrap();

        // Root should have child 'a'
        let mut root_children = Vec::new();
        t.for_each_child(0, |ch, _| root_children.push(ch));
        assert_eq!(root_children, vec![b'a']);

        // State for 'a' should have children 'b', 'c', 'd'
        let a_state = t.state_move(0, b'a');
        let mut a_children = Vec::new();
        t.for_each_child(a_state, |ch, _| a_children.push(ch));
        a_children.sort();
        assert_eq!(a_children, vec![b'b', b'c', b'd']);
    }

    #[test]
    fn test_da_trie_map() {
        let mut map = DoubleArrayTrieMap::<u32>::new();
        map.insert(b"hello", 42).unwrap();
        map.insert(b"world", 100).unwrap();
        map.insert(b"help", 7).unwrap();

        assert_eq!(map.get(b"hello"), Some(42));
        assert_eq!(map.get(b"world"), Some(100));
        assert_eq!(map.get(b"help"), Some(7));
        assert_eq!(map.get(b"missing"), None);

        // Update
        let prev = map.insert(b"hello", 99).unwrap();
        assert_eq!(prev, Some(42));
        assert_eq!(map.get(b"hello"), Some(99));

        // Remove
        let removed = map.remove(b"world");
        assert_eq!(removed, Some(100));
        assert_eq!(map.get(b"world"), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_mem_size() {
        let t = DoubleArrayTrie::new();
        // 256 states * (8 bytes DaState + 4 bytes NInfo) = 256 * 12 = 3072
        assert_eq!(t.mem_size(), 256 * 12);
        assert_eq!(std::mem::size_of::<DaState>(), 8);
        assert_eq!(std::mem::size_of::<NInfo>(), 4);
    }

    #[test]
    fn test_shared_prefixes() {
        let mut t = DoubleArrayTrie::new();
        // Many keys sharing common prefixes — stress the base allocation
        t.insert(b"test").unwrap();
        t.insert(b"testing").unwrap();
        t.insert(b"tested").unwrap();
        t.insert(b"tester").unwrap();
        t.insert(b"tests").unwrap();
        t.insert(b"tea").unwrap();
        t.insert(b"team").unwrap();
        t.insert(b"tear").unwrap();

        assert_eq!(t.len(), 8);
        assert!(t.contains(b"test"));
        assert!(t.contains(b"testing"));
        assert!(t.contains(b"tea"));
        assert!(t.contains(b"team"));
        assert!(!t.contains(b"te")); // prefix only, not inserted
        assert!(!t.contains(b"testi")); // prefix only
    }

    #[test]
    fn test_long_keys() {
        let mut t = DoubleArrayTrie::new();
        let long_key = "a".repeat(1000);
        t.insert(long_key.as_bytes()).unwrap();
        assert!(t.contains(long_key.as_bytes()));
        assert_eq!(t.len(), 1);

        let state = t.lookup_state(long_key.as_bytes()).unwrap();
        let restored = t.restore_key(state).unwrap();
        assert_eq!(restored, long_key.as_bytes());
    }

    #[test]
    fn test_binary_keys() {
        let mut t = DoubleArrayTrie::new();
        // Keys with all byte values including 0x00 and 0xFF
        t.insert(&[0x00, 0xFF, 0x80]).unwrap();
        t.insert(&[0x00, 0xFF, 0x81]).unwrap();
        t.insert(&[0xFF, 0x00, 0x01]).unwrap();

        assert_eq!(t.len(), 3);
        assert!(t.contains(&[0x00, 0xFF, 0x80]));
        assert!(t.contains(&[0xFF, 0x00, 0x01]));
        assert!(!t.contains(&[0x00, 0xFF]));
    }

    #[test]
    fn test_relocation_stress() {
        let mut t = DoubleArrayTrie::new();
        // Insert keys that force many relocations
        // Single-char keys use same base offset, forcing conflicts
        for ch in 0u8..=127u8 {
            let key = [ch];
            t.insert(&key).unwrap();
        }
        assert_eq!(t.len(), 128);
        for ch in 0u8..=127u8 {
            assert!(t.contains(&[ch]), "missing single-byte key {}", ch);
        }
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut t = DoubleArrayTrie::with_capacity(10000);
        assert!(t.total_states() >= 10000);
        t.insert(b"hello").unwrap();
        t.insert(b"world").unwrap();
        t.shrink_to_fit();
        // After shrink, should be much smaller
        assert!(t.total_states() < 1000);
        // But keys must still work
        assert!(t.contains(b"hello"));
        assert!(t.contains(b"world"));
    }

    /// Regression test: shrink_to_fit on an empty trie must not collapse below
    /// 256 states. Otherwise contains() with a non-empty key does an OOB
    /// get_unchecked(base ^ ch) where base=0 and ch can be up to 255.
    #[test]
    fn test_shrink_to_fit_empty() {
        let mut t = DoubleArrayTrie::new();
        t.shrink_to_fit();
        assert!(
            t.total_states() >= 256,
            "empty trie after shrink_to_fit must retain >= 256 states, got {}",
            t.total_states()
        );
        // Must not OOB / UB on a non-empty lookup.
        assert!(!t.contains(b"a"));
        assert!(!t.contains(b"\xff"));
    }

    /// Regression test: with_capacity < 256 must not cause UB.
    /// Bug 1.1.1: base=0 + unsafe get_unchecked(ch) with ch >= cap is UB.
    #[test]
    fn test_small_capacity_no_oob() {
        // with_capacity(2) must be clamped to 256 minimum
        let t = DoubleArrayTrie::with_capacity(2);
        assert!(t.total_states() >= 256, "minimum capacity must be 256");

        // contains on empty trie must not crash (base=0, next=ch)
        assert!(!t.contains(b"hello")); // 'h' = 104, needs states[104]
        assert!(!t.contains(b"\xff")); // 0xFF = 255, needs states[255]
        assert!(!t.contains(b"\x00")); // 0x00 = 0, needs states[0]

        // insert + lookup must work
        let mut t = DoubleArrayTrie::with_capacity(1);
        t.insert(b"test").unwrap();
        assert!(t.contains(b"test"));
        assert!(!t.contains(b"other"));

        // with_capacity(0) must also be safe
        let t = DoubleArrayTrie::with_capacity(0);
        assert!(t.total_states() >= 256);
        assert!(!t.contains(b"anything"));
    }

    #[test]
    fn test_remove_and_reinsert() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"abc").unwrap();
        assert!(t.contains(b"abc"));

        t.remove(b"abc");
        assert!(!t.contains(b"abc"));
        assert_eq!(t.len(), 0);

        // Reinsert same key
        assert!(t.insert(b"abc").unwrap());
        assert!(t.contains(b"abc"));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_lookup_state_consistency() {
        let mut t = DoubleArrayTrie::new();
        let keys: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma", b"delta"];
        for &key in &keys {
            t.insert(key).unwrap();
        }

        // Each key should have a unique state
        let states: Vec<u32> = keys.iter().map(|k| t.lookup_state(k).unwrap()).collect();
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert_ne!(
                    states[i],
                    states[j],
                    "states for {:?} and {:?} should differ",
                    std::str::from_utf8(keys[i]).unwrap(),
                    std::str::from_utf8(keys[j]).unwrap()
                );
            }
        }
    }

    #[test]
    fn test_state_move_free_state() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"a").unwrap();

        // Transition with a byte that was never inserted — the target state
        // is free (has FREE_BIT set in parent). The optimized state_move
        // must return NIL_STATE because free parent can never equal curr.
        for ch in 0u8..=255 {
            if ch == b'a' {
                continue;
            }
            assert_eq!(
                t.state_move(0, ch),
                NIL_STATE,
                "state_move(0, {}) should be NIL for free state",
                ch
            );
        }
    }

    #[test]
    fn test_state_move_after_relocations() {
        let mut t = DoubleArrayTrie::new();
        // Insert many keys sharing prefix to force child array relocations.
        // After relocations, state_move must still return correct results.
        let keys: Vec<Vec<u8>> = (0u8..=127).map(|ch| vec![b'x', ch]).collect();
        for k in &keys {
            t.insert(k).unwrap();
        }

        let x_state = t.state_move(0, b'x');
        assert_ne!(x_state, NIL_STATE);

        for ch in 0u8..=127 {
            let child = t.state_move(x_state, ch);
            assert_ne!(
                child, NIL_STATE,
                "child for byte {} missing after relocations",
                ch
            );
            assert!(t.is_term(child));
        }
        // Bytes not inserted should still return NIL
        for ch in 128u8..=255 {
            assert_eq!(t.state_move(x_state, ch), NIL_STATE);
        }
    }

    #[test]
    fn test_state_move_byte_zero() {
        let mut t = DoubleArrayTrie::new();
        // byte 0 is special: base ^ 0 = base, so next == base(curr)
        t.insert(&[0u8]).unwrap();
        let s = t.state_move(0, 0);
        assert_ne!(s, NIL_STATE);
        assert!(t.is_term(s));
    }

    #[test]
    fn test_state_move_after_remove() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"ab").unwrap();
        t.insert(b"ac").unwrap();

        // Remove "ab", its leaf state becomes free
        t.remove(b"ab");
        assert_eq!(t.state_move(t.state_move(0, b'a'), b'b'), NIL_STATE);
        // "ac" must still work
        let a = t.state_move(0, b'a');
        assert_ne!(a, NIL_STATE);
        let ac = t.state_move(a, b'c');
        assert_ne!(ac, NIL_STATE);
        assert!(t.is_term(ac));
    }

    /// Performance test — only meaningful in release mode.
    /// Verifies O(key_length) insert/lookup, not O(n).
    #[test]
    fn test_performance_5000_terms() {
        // Generate 5000 realistic terms
        let terms: Vec<String> = (0..5000)
            .map(|i| {
                format!(
                    "term_{:06}_{}",
                    i,
                    ["alpha", "beta", "gamma", "delta"][i % 4]
                )
            })
            .collect();

        // Insert
        let start = std::time::Instant::now();
        let mut t = DoubleArrayTrie::new();
        for term in &terms {
            t.insert(term.as_bytes()).unwrap();
        }
        let _insert_time = start.elapsed();

        assert_eq!(t.len(), 5000);

        // Lookup (all hits)
        let start = std::time::Instant::now();
        for term in &terms {
            assert!(t.contains(term.as_bytes()));
        }
        let _lookup_time = start.elapsed();

        // Lookup (all misses)
        let start = std::time::Instant::now();
        for i in 0..5000 {
            let miss = format!("miss_{:06}", i);
            assert!(!t.contains(miss.as_bytes()));
        }
        let _miss_time = start.elapsed();

        // In release mode, all three should complete in well under 100ms
        // (Cedar does 5000 inserts in ~876µs)
        #[cfg(not(debug_assertions))]
        {
            eprintln!(
                "DoubleArrayTrie 5000 terms: insert={:?}, lookup_hit={:?}, lookup_miss={:?}",
                _insert_time, _lookup_time, _miss_time
            );
            eprintln!(
                "Memory: {} bytes ({} bytes/key), {} states",
                t.mem_size(),
                t.mem_size() / 5000,
                t.total_states()
            );
            // Sanity: insert should be under 50ms in release
            assert!(
                _insert_time.as_millis() < 50,
                "Insert too slow: {:?}",
                _insert_time
            );
            // Lookup should be under 10ms
            assert!(
                _lookup_time.as_millis() < 10,
                "Lookup too slow: {:?}",
                _lookup_time
            );
        }
    }

    #[test]
    fn test_entries_with_prefix() {
        let mut map = DoubleArrayTrieMap::<u32>::new();
        map.insert(b"apple", 1).unwrap();
        map.insert(b"app", 2).unwrap();
        map.insert(b"application", 3).unwrap();
        map.insert(b"banana", 4).unwrap();

        let entries = map.entries_with_prefix(b"app");
        assert_eq!(entries.len(), 3);
        // Verify all app* entries are present
        let values: Vec<u32> = entries.iter().map(|(_, v)| *v).collect();
        assert!(values.contains(&1)); // apple
        assert!(values.contains(&2)); // app
        assert!(values.contains(&3)); // application

        let banana_entries = map.entries_with_prefix(b"ban");
        assert_eq!(banana_entries.len(), 1);
        assert_eq!(banana_entries[0].1, 4);

        let none = map.entries_with_prefix(b"xyz");
        assert_eq!(none.len(), 0);
    }

    #[test]
    fn test_values_with_prefix() {
        let mut map = DoubleArrayTrieMap::<u32>::new();
        map.insert(b"test_a", 10).unwrap();
        map.insert(b"test_b", 20).unwrap();
        map.insert(b"other", 30).unwrap();

        let vals = map.values_with_prefix(b"test_");
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&10));
        assert!(vals.contains(&20));
    }

    #[test]
    fn test_for_each_key_with_prefix() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"hello").unwrap();
        t.insert(b"help").unwrap();
        t.insert(b"world").unwrap();

        let mut found = Vec::new();
        t.for_each_key_with_prefix(b"hel", |key| {
            found.push(key.to_vec());
        });
        found.sort();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0], b"hello");
        assert_eq!(found[1], b"help");

        let mut all = Vec::new();
        t.for_each_key_with_prefix(b"", |key| all.push(key.to_vec()));
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_prefix_iterator_matches_entries_with_prefix() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        let words: &[&[u8]] = &[b"apple", b"app", b"application", b"banana", b"band", b"bar"];
        for (i, w) in words.iter().enumerate() {
            trie.insert(w, i as i32 + 1).unwrap();
        }

        // Compare iter_prefix with entries_with_prefix for various prefixes
        let prefixes: Vec<&[u8]> = vec![b"app", b"ban", b"b", b"apple", b"z", b""];
        for prefix in &prefixes {
            let mut lazy: Vec<_> = trie.iter_prefix(prefix).collect();
            let mut eager = trie.entries_with_prefix(prefix);
            lazy.sort_by(|a, b| a.0.cmp(&b.0));
            eager.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(
                lazy,
                eager,
                "mismatch for prefix {:?}",
                std::str::from_utf8(prefix)
            );
        }
    }

    #[test]
    fn test_prefix_iterator_empty_trie() {
        let trie = DoubleArrayTrieMap::<i32>::new();
        assert_eq!(trie.iter_prefix(b"any").count(), 0);
    }

    #[test]
    fn test_prefix_iterator_nonexistent_prefix() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"hello", 1).unwrap();
        assert_eq!(trie.iter_prefix(b"xyz").count(), 0);
    }

    #[test]
    fn test_prefix_iterator_empty_prefix_yields_all() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"a", 1).unwrap();
        trie.insert(b"b", 2).unwrap();
        trie.insert(b"c", 3).unwrap();
        assert_eq!(trie.iter_prefix(b"").count(), 3);
    }

    #[test]
    fn test_prefix_iterator_drop_early() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        for i in 0..100 {
            let key = format!("key{:03}", i);
            trie.insert(key.as_bytes(), i).unwrap();
        }
        // Just take 5 — should not panic or leak
        let first5: Vec<_> = trie.iter_prefix(b"key").take(5).collect();
        assert_eq!(first5.len(), 5);
    }

    #[test]
    fn test_fuzzy_iterator_exact_match() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"cat", 1).unwrap();
        trie.insert(b"car", 2).unwrap();
        trie.insert(b"cap", 3).unwrap();
        trie.insert(b"dog", 4).unwrap();

        // max_dist=0 → exact match only
        let results: Vec<_> = trie.iter_fuzzy(b"cat", 0).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, b"cat");
        assert_eq!(results[0].1, 1);
    }

    #[test]
    fn test_fuzzy_iterator_distance_1() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"cat", 1).unwrap();
        trie.insert(b"car", 2).unwrap();
        trie.insert(b"cap", 3).unwrap();
        trie.insert(b"bat", 4).unwrap();
        trie.insert(b"dog", 5).unwrap();

        // Distance 1 from "cat": cat(0), car(1), cap(1), bat(1)
        let mut results: Vec<_> = trie.iter_fuzzy(b"cat", 1).collect();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        let keys: Vec<&[u8]> = results.iter().map(|(k, _)| k.as_slice()).collect();
        assert!(keys.contains(&b"cat".as_slice()));
        assert!(keys.contains(&b"car".as_slice()));
        assert!(keys.contains(&b"cap".as_slice()));
        assert!(keys.contains(&b"bat".as_slice()));
        assert!(!keys.contains(&b"dog".as_slice())); // distance 3
    }

    #[test]
    fn test_fuzzy_iterator_empty_query() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"a", 1).unwrap();
        trie.insert(b"ab", 2).unwrap();
        trie.insert(b"abc", 3).unwrap();

        // Empty query, max_dist=1 → only keys of length ≤ 1
        let results: Vec<_> = trie.iter_fuzzy(b"", 1).collect();
        assert!(results.iter().any(|(k, _)| k == b"a"));
        assert!(!results.iter().any(|(k, _)| k == b"abc")); // distance 3
    }

    #[test]
    fn test_fuzzy_iterator_empty_trie() {
        let trie = DoubleArrayTrieMap::<i32>::new();
        assert_eq!(trie.iter_fuzzy(b"test", 2).count(), 0);
    }

    #[test]
    fn test_fuzzy_iterator_all_within_distance() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"cat", 1).unwrap();

        // Large max_dist should find the single key
        let results: Vec<_> = trie.iter_fuzzy(b"xyz", 10).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, b"cat");
    }

    #[test]
    fn test_prefix_iterator_prefix_longer_than_keys() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"app", 1).unwrap();
        trie.insert(b"apple", 2).unwrap();
        // Prefix longer than any key — should yield nothing
        assert_eq!(trie.iter_prefix(b"application123").count(), 0);
    }

    #[test]
    fn test_prefix_iterator_single_key_trie() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"hello", 42).unwrap();

        // Exact key as prefix
        let results: Vec<_> = trie.iter_prefix(b"hello").collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (b"hello".to_vec(), 42));

        // Prefix of the key
        let results: Vec<_> = trie.iter_prefix(b"hel").collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (b"hello".to_vec(), 42));

        // Non-matching prefix
        assert_eq!(trie.iter_prefix(b"world").count(), 0);
    }

    #[test]
    fn test_prefix_iterator_deeply_nested() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        // Create deeply nested prefix chain
        trie.insert(b"a", 1).unwrap();
        trie.insert(b"ab", 2).unwrap();
        trie.insert(b"abc", 3).unwrap();
        trie.insert(b"abcd", 4).unwrap();
        trie.insert(b"abcde", 5).unwrap();

        // Each prefix should include itself and all extensions
        assert_eq!(trie.iter_prefix(b"").count(), 5);
        assert_eq!(trie.iter_prefix(b"a").count(), 5);
        assert_eq!(trie.iter_prefix(b"ab").count(), 4);
        assert_eq!(trie.iter_prefix(b"abc").count(), 3);
        assert_eq!(trie.iter_prefix(b"abcd").count(), 2);
        assert_eq!(trie.iter_prefix(b"abcde").count(), 1);
        assert_eq!(trie.iter_prefix(b"abcdef").count(), 0);
    }

    #[test]
    fn test_prefix_iterator_large_trie() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        for i in 0..1000 {
            let key = format!("key{:04}", i);
            trie.insert(key.as_bytes(), i).unwrap();
        }

        // Compare against entries_with_prefix for several prefixes
        for prefix in [
            b"key" as &[u8],
            b"key0",
            b"key00",
            b"key000",
            b"key0001",
            b"",
        ] {
            let mut lazy: Vec<_> = trie.iter_prefix(prefix).collect();
            let mut eager = trie.entries_with_prefix(prefix);
            lazy.sort_by(|a, b| a.0.cmp(&b.0));
            eager.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(
                lazy,
                eager,
                "mismatch for prefix {:?}",
                std::str::from_utf8(prefix)
            );
        }
    }

    #[test]
    fn test_prefix_iterator_binary_keys() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(&[0x00, 0x01, 0x02], 1).unwrap();
        trie.insert(&[0x00, 0x01, 0xFF], 2).unwrap();
        trie.insert(&[0x00, 0xFF], 3).unwrap();
        trie.insert(&[0xFF, 0x00], 4).unwrap();

        let results: Vec<_> = trie.iter_prefix(&[0x00]).collect();
        assert_eq!(results.len(), 3); // all keys starting with 0x00

        let results2: Vec<_> = trie.iter_prefix(&[0x00, 0x01]).collect();
        assert_eq!(results2.len(), 2);
    }

    #[test]
    fn test_fuzzy_iterator_edit_distance_verification() {
        // Helper: compute Levenshtein distance
        fn edit_distance(a: &[u8], b: &[u8]) -> usize {
            let m = a.len();
            let n = b.len();
            let mut dp = vec![vec![0usize; n + 1]; m + 1];
            for i in 0..=m {
                dp[i][0] = i;
            }
            for j in 0..=n {
                dp[0][j] = j;
            }
            for i in 1..=m {
                for j in 1..=n {
                    let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                    dp[i][j] = (dp[i - 1][j] + 1)
                        .min(dp[i][j - 1] + 1)
                        .min(dp[i - 1][j - 1] + cost);
                }
            }
            dp[m][n]
        }

        let mut trie = DoubleArrayTrieMap::<i32>::new();
        let words = [
            b"cat" as &[u8],
            b"car",
            b"cap",
            b"bat",
            b"hat",
            b"cart",
            b"ca",
            b"c",
            b"cats",
            b"dog",
        ];
        for (i, w) in words.iter().enumerate() {
            trie.insert(w, i as i32).unwrap();
        }

        let query = b"cat";
        for max_dist in 0..=3 {
            let results: Vec<_> = trie.iter_fuzzy(query, max_dist).collect();
            for (key, _) in &results {
                let dist = edit_distance(key, query);
                assert!(
                    dist <= max_dist,
                    "key {:?} has distance {} from {:?}, exceeds max_dist {}",
                    std::str::from_utf8(key),
                    dist,
                    std::str::from_utf8(query),
                    max_dist
                );
            }
            // Verify completeness: check all trie keys within distance
            for w in &words {
                let dist = edit_distance(w, query);
                if dist <= max_dist {
                    assert!(
                        results.iter().any(|(k, _)| k == *w),
                        "key {:?} at distance {} missing from results (max_dist={})",
                        std::str::from_utf8(w),
                        dist,
                        max_dist
                    );
                }
            }
        }
    }

    #[test]
    fn test_fuzzy_iterator_insertion_deletion_substitution() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"cat", 1).unwrap(); // exact
        trie.insert(b"at", 2).unwrap(); // deletion of 'c'
        trie.insert(b"ct", 3).unwrap(); // deletion of 'a'
        trie.insert(b"ca", 4).unwrap(); // deletion of 't'
        trie.insert(b"cats", 5).unwrap(); // insertion of 's'
        trie.insert(b"scat", 6).unwrap(); // insertion of 's' at start
        trie.insert(b"caat", 7).unwrap(); // insertion of 'a'
        trie.insert(b"bat", 8).unwrap(); // substitution
        trie.insert(b"cot", 9).unwrap(); // substitution
        trie.insert(b"cab", 10).unwrap(); // substitution

        let results: Vec<_> = trie.iter_fuzzy(b"cat", 1).collect();
        let keys: Vec<Vec<u8>> = results.iter().map(|(k, _)| k.clone()).collect();

        // All distance-1 keys should be found
        assert!(keys.contains(&b"cat".to_vec()), "exact match");
        assert!(keys.contains(&b"at".to_vec()), "deletion of c");
        assert!(keys.contains(&b"ct".to_vec()), "deletion of a");
        assert!(keys.contains(&b"ca".to_vec()), "deletion of t");
        assert!(keys.contains(&b"cats".to_vec()), "insertion of s");
        assert!(keys.contains(&b"bat".to_vec()), "substitution c->b");
        assert!(keys.contains(&b"cot".to_vec()), "substitution a->o");
        assert!(keys.contains(&b"cab".to_vec()), "substitution t->b");
    }

    #[test]
    fn test_fuzzy_iterator_root_terminal() {
        // Empty string as a key
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"", 1).unwrap();
        trie.insert(b"a", 2).unwrap();
        trie.insert(b"ab", 3).unwrap();

        // max_dist=0 with empty query — should find only ""
        let results: Vec<_> = trie.iter_fuzzy(b"", 0).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, b"".to_vec());

        // max_dist=1 with empty query — should find "" and "a"
        let results: Vec<_> = trie.iter_fuzzy(b"", 1).collect();
        assert!(results.iter().any(|(k, _)| k.is_empty()));
        assert!(results.iter().any(|(k, _)| k == b"a"));
        assert!(!results.iter().any(|(k, _)| k == b"ab")); // distance 2
    }

    #[test]
    fn test_fuzzy_iterator_long_query_short_keys() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(b"a", 1).unwrap();
        trie.insert(b"ab", 2).unwrap();

        // Long query, max_dist=1 — too far from short keys
        let results: Vec<_> = trie.iter_fuzzy(b"abcdefgh", 1).collect();
        assert_eq!(results.len(), 0);

        // Long query, large max_dist — should find them
        let results: Vec<_> = trie.iter_fuzzy(b"abcdefgh", 7).collect();
        assert!(results.iter().any(|(k, _)| k == b"a"));
    }

    #[test]
    fn test_fuzzy_iterator_binary_keys() {
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        trie.insert(&[0x00, 0x01], 1).unwrap();
        trie.insert(&[0x00, 0x02], 2).unwrap();
        trie.insert(&[0xFF, 0xFE], 3).unwrap();

        // Distance 1 from [0x00, 0x01]: substitution gives [0x00, 0x02]
        let results: Vec<_> = trie.iter_fuzzy(&[0x00, 0x01], 1).collect();
        assert!(results.iter().any(|(k, _)| k == &[0x00, 0x01]));
        assert!(results.iter().any(|(k, _)| k == &[0x00, 0x02]));
        assert!(!results.iter().any(|(k, _)| k == &[0xFF, 0xFE])); // too far
    }

    #[test]
    fn test_fuzzy_iterator_pruning_completeness() {
        // Stress test: ensure pruning doesn't miss valid results
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        let words: Vec<String> = (0..100).map(|i| format!("word{:02}", i)).collect();
        for (i, w) in words.iter().enumerate() {
            trie.insert(w.as_bytes(), i as i32).unwrap();
        }

        fn edit_distance(a: &[u8], b: &[u8]) -> usize {
            let m = a.len();
            let n = b.len();
            let mut dp = vec![vec![0usize; n + 1]; m + 1];
            for i in 0..=m {
                dp[i][0] = i;
            }
            for j in 0..=n {
                dp[0][j] = j;
            }
            for i in 1..=m {
                for j in 1..=n {
                    let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                    dp[i][j] = (dp[i - 1][j] + 1)
                        .min(dp[i][j - 1] + 1)
                        .min(dp[i - 1][j - 1] + cost);
                }
            }
            dp[m][n]
        }

        let query = b"word50";
        for max_dist in 0..=2 {
            let results: Vec<_> = trie.iter_fuzzy(query, max_dist).collect();

            // Every result should be within distance
            for (key, _) in &results {
                let d = edit_distance(key, query);
                assert!(
                    d <= max_dist,
                    "spurious result {:?} at distance {}",
                    std::str::from_utf8(key),
                    d
                );
            }

            // Every word within distance should appear
            for w in &words {
                let d = edit_distance(w.as_bytes(), query);
                if d <= max_dist {
                    assert!(
                        results.iter().any(|(k, _)| k == w.as_bytes()),
                        "missing {:?} at distance {} (max_dist={})",
                        w,
                        d,
                        max_dist
                    );
                }
            }
        }
    }

    // --- Cursor / Range tests ---

    #[test]
    fn test_cursor_seek_begin_end() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"apple").unwrap();
        t.insert(b"banana").unwrap();
        t.insert(b"cherry").unwrap();

        let mut c = t.cursor();
        assert!(c.seek_begin());
        assert_eq!(c.key(), b"apple");

        assert!(c.seek_end());
        assert_eq!(c.key(), b"cherry");
    }

    #[test]
    fn test_cursor_next_prev() {
        let mut t = DoubleArrayTrie::new();
        for w in &["apple", "banana", "cherry", "date", "elderberry"] {
            t.insert(w.as_bytes()).unwrap();
        }

        let mut c = t.cursor();
        c.seek_begin();
        assert_eq!(c.key(), b"apple");

        assert!(c.next());
        assert_eq!(c.key(), b"banana");
        assert!(c.next());
        assert_eq!(c.key(), b"cherry");
        assert!(c.next());
        assert_eq!(c.key(), b"date");
        assert!(c.next());
        assert_eq!(c.key(), b"elderberry");
        assert!(!c.next()); // Past end

        // Walk backward
        c.seek_end();
        assert_eq!(c.key(), b"elderberry");
        assert!(c.prev());
        assert_eq!(c.key(), b"date");
        assert!(c.prev());
        assert_eq!(c.key(), b"cherry");
        assert!(c.prev());
        assert_eq!(c.key(), b"banana");
        assert!(c.prev());
        assert_eq!(c.key(), b"apple");
        assert!(!c.prev()); // Past begin
    }

    #[test]
    fn test_cursor_seek_lower_bound() {
        let mut t = DoubleArrayTrie::new();
        for w in &["apple", "banana", "cherry", "date", "elderberry"] {
            t.insert(w.as_bytes()).unwrap();
        }

        let mut c = t.cursor();

        // Exact match
        assert!(c.seek_lower_bound(b"cherry"));
        assert_eq!(c.key(), b"cherry");

        // Between keys
        assert!(c.seek_lower_bound(b"c"));
        assert_eq!(c.key(), b"cherry");

        // Before all keys
        assert!(c.seek_lower_bound(b"a"));
        assert_eq!(c.key(), b"apple");

        // After all keys
        assert!(!c.seek_lower_bound(b"z"));

        // Between banana and cherry
        assert!(c.seek_lower_bound(b"cat"));
        assert_eq!(c.key(), b"cherry");
    }

    #[test]
    fn test_range() {
        let mut t = DoubleArrayTrie::new();
        for w in &["apple", "banana", "cherry", "date", "elderberry", "fig"] {
            t.insert(w.as_bytes()).unwrap();
        }

        // [b, e) => banana, cherry, date
        let range: Vec<Vec<u8>> = t.range(b"b", b"e").collect();
        assert_eq!(range.len(), 3);
        assert_eq!(range[0], b"banana");
        assert_eq!(range[1], b"cherry");
        assert_eq!(range[2], b"date");

        // [a, z) => all 6
        let all: Vec<Vec<u8>> = t.range(b"a", b"z").collect();
        assert_eq!(all.len(), 6);

        // [d, d) => empty (from == to)
        let empty: Vec<Vec<u8>> = t.range(b"d", b"d").collect();
        assert_eq!(empty.len(), 0);

        // [cherry, elderberry) => cherry, date
        let mid: Vec<Vec<u8>> = t.range(b"cherry", b"elderberry").collect();
        assert_eq!(mid.len(), 2);
        assert_eq!(mid[0], b"cherry");
        assert_eq!(mid[1], b"date");
    }

    #[test]
    fn test_cursor_empty_trie() {
        let t = DoubleArrayTrie::new();
        let mut c = t.cursor();
        assert!(!c.seek_begin());
        assert!(!c.seek_end());
        assert!(!c.seek_lower_bound(b"anything"));
    }

    #[test]
    fn test_cursor_single_key() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"only").unwrap();

        let mut c = t.cursor();
        assert!(c.seek_begin());
        assert_eq!(c.key(), b"only");
        assert!(!c.next());

        assert!(c.seek_end());
        assert_eq!(c.key(), b"only");
        assert!(!c.prev());
    }

    #[test]
    fn test_cursor_with_empty_key() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"").unwrap();
        t.insert(b"a").unwrap();
        t.insert(b"b").unwrap();

        let mut c = t.cursor();
        assert!(c.seek_begin());
        assert_eq!(c.key(), b""); // Empty key is the smallest

        assert!(c.next());
        assert_eq!(c.key(), b"a");

        assert!(c.next());
        assert_eq!(c.key(), b"b");
    }

    #[test]
    fn test_cursor_full_traversal_matches_keys() {
        let mut t = DoubleArrayTrie::new();
        let words = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        ];
        for w in &words {
            t.insert(w.as_bytes()).unwrap();
        }

        // Forward traversal via cursor should match sorted keys()
        let mut cursor_keys = Vec::new();
        let mut c = t.cursor();
        if c.seek_begin() {
            cursor_keys.push(c.key().to_vec());
            while c.next() {
                cursor_keys.push(c.key().to_vec());
            }
        }

        let mut trie_keys = t.keys();
        trie_keys.sort();

        assert_eq!(
            cursor_keys, trie_keys,
            "Cursor traversal must match sorted keys()"
        );
    }

    #[test]
    fn test_range_empty_bounds() {
        let mut t = DoubleArrayTrie::new();
        for w in &["a", "b", "c", "d"] {
            t.insert(w.as_bytes()).unwrap();
        }

        // Range where from > to should return nothing
        let r: Vec<_> = t.range(b"z", b"a").collect();
        assert_eq!(r.len(), 0);

        // Range covering everything
        let r: Vec<_> = t.range(b"", b"\xff").collect();
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_cursor_seek_lower_bound_exact_last() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"aaa").unwrap();
        t.insert(b"zzz").unwrap();

        let mut c = t.cursor();

        // Seek to exact last key
        assert!(c.seek_lower_bound(b"zzz"));
        assert_eq!(c.key(), b"zzz");
        assert!(!c.next()); // No key after zzz

        // Seek past last key
        assert!(!c.seek_lower_bound(b"zzzz"));
    }

    #[test]
    fn test_cursor_prev_from_begin() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"first").unwrap();
        t.insert(b"second").unwrap();

        let mut c = t.cursor();
        c.seek_begin();
        assert_eq!(c.key(), b"first");
        assert!(!c.prev()); // Can't go before first
    }

    #[test]
    fn test_cursor_interleaved_next_prev() {
        let mut t = DoubleArrayTrie::new();
        for w in &["a", "b", "c", "d", "e"] {
            t.insert(w.as_bytes()).unwrap();
        }

        let mut c = t.cursor();
        c.seek_begin();
        assert_eq!(c.key(), b"a");

        c.next();
        assert_eq!(c.key(), b"b");
        c.next();
        assert_eq!(c.key(), b"c");

        // Go back
        c.prev();
        assert_eq!(c.key(), b"b");

        // Forward again
        c.next();
        assert_eq!(c.key(), b"c");
        c.next();
        assert_eq!(c.key(), b"d");
    }

    #[test]
    fn test_cursor_many_keys_sorted() {
        let mut t = DoubleArrayTrie::new();
        for i in 0..200u32 {
            t.insert(format!("k{:04}", i).as_bytes()).unwrap();
        }

        // Forward traversal should produce sorted order
        let mut c = t.cursor();
        let mut keys = Vec::new();
        if c.seek_begin() {
            keys.push(c.key().to_vec());
            while c.next() {
                keys.push(c.key().to_vec());
            }
        }
        assert_eq!(keys.len(), 200);
        for i in 1..keys.len() {
            assert!(
                keys[i - 1] < keys[i],
                "Not sorted at {}: {:?} >= {:?}",
                i,
                String::from_utf8_lossy(&keys[i - 1]),
                String::from_utf8_lossy(&keys[i])
            );
        }

        // Backward traversal should produce reverse sorted order
        let mut c = t.cursor();
        let mut rkeys = Vec::new();
        if c.seek_end() {
            rkeys.push(c.key().to_vec());
            while c.prev() {
                rkeys.push(c.key().to_vec());
            }
        }
        assert_eq!(rkeys.len(), 200);
        rkeys.reverse();
        assert_eq!(keys, rkeys, "Forward and backward traversals must match");
    }

    #[test]
    fn test_range_single_element() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"hello").unwrap();
        t.insert(b"world").unwrap();

        // Range containing exactly one key
        let r: Vec<_> = t.range(b"hello", b"world").collect();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], b"hello");
    }

    #[test]
    fn test_seek_lower_bound_between_shared_prefix() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"abc").unwrap();
        t.insert(b"abd").unwrap();
        t.insert(b"abe").unwrap();

        let mut c = t.cursor();
        // Seek to "abd" exactly
        assert!(c.seek_lower_bound(b"abd"));
        assert_eq!(c.key(), b"abd");

        // Seek between "abc" and "abd"
        assert!(c.seek_lower_bound(b"abcc"));
        assert_eq!(c.key(), b"abd");

        // Seek before all
        assert!(c.seek_lower_bound(b"ab"));
        assert_eq!(c.key(), b"abc");
    }

    /// Reproduction of bug #1.1: keys_with_prefix returns incomplete results
    #[test]
    fn test_keys_with_prefix_1000_terms() {
        let mut t = DoubleArrayTrie::new();
        for i in 0..1000u32 {
            t.insert(format!("term_{:04}", i).as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 1000);

        let results = t.keys_with_prefix(b"term_00");
        assert_eq!(
            results.len(),
            100,
            "keys_with_prefix('term_00') should return 100 (term_0000..term_0099), got {}",
            results.len()
        );

        let results2 = t.keys_with_prefix(b"term_01");
        assert_eq!(
            results2.len(),
            100,
            "keys_with_prefix('term_01') should return 100, got {}",
            results2.len()
        );

        let all = t.keys_with_prefix(b"term_");
        assert_eq!(
            all.len(),
            1000,
            "keys_with_prefix('term_') should return 1000, got {}",
            all.len()
        );

        let all_keys = t.keys();
        assert_eq!(
            all_keys.len(),
            1000,
            "keys() should return 1000, got {}",
            all_keys.len()
        );
    }

    // --- Value tests (via DoubleArrayTrieMap) ---

    #[test]
    fn test_map_values_basic() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        let prev = m.insert(b"hello", 42).unwrap();
        assert_eq!(prev, None);

        assert_eq!(m.get(b"hello"), Some(42));
        assert_eq!(m.get(b"world"), None);

        m.insert(b"world", 100).unwrap();
        assert_eq!(m.get(b"world"), Some(100));

        let prev = m.insert(b"hello", 99).unwrap();
        assert_eq!(prev, Some(42));
        assert_eq!(m.get(b"hello"), Some(99));
    }

    #[test]
    fn test_map_values_many() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        for i in 0..500i32 {
            m.insert(format!("key_{:04}", i).as_bytes(), i).unwrap();
        }
        for i in 0..500i32 {
            assert_eq!(
                m.get(format!("key_{:04}", i).as_bytes()),
                Some(i),
                "value mismatch for key_{:04}",
                i
            );
        }
        assert_eq!(m.len(), 500);
    }

    #[test]
    fn test_map_values_with_contains() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        m.insert(b"abc", 1).unwrap();
        m.insert(b"abd", 2).unwrap();

        assert!(m.contains(b"abc"));
        assert!(m.contains(b"abd"));
        assert!(!m.contains(b"ab"));

        assert_eq!(m.get(b"abc"), Some(1));
        assert_eq!(m.get(b"abd"), Some(2));
    }

    // --- Consult tests ---

    #[test]
    fn test_consult_many_inserts() {
        // Consult (relocate-smaller) should not break correctness
        let mut t = DoubleArrayTrie::new();
        for i in 0..1000u32 {
            t.insert(format!("term_{:04}", i).as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 1000);
        for i in 0..1000u32 {
            assert!(
                t.contains(format!("term_{:04}", i).as_bytes()),
                "missing term_{:04}",
                i
            );
        }
    }

    // --- Prefix value iteration tests (via DoubleArrayTrieMap) ---

    #[test]
    fn test_map_prefix_value_iteration() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        m.insert(b"app", 1).unwrap();
        m.insert(b"apple", 2).unwrap();
        m.insert(b"application", 3).unwrap();
        m.insert(b"banana", 4).unwrap();

        let mut vals: Vec<i32> = m.values_with_prefix(b"app");
        vals.sort();
        assert_eq!(vals, vec![1, 2, 3]);

        let mut all_vals: Vec<i32> = m.values_with_prefix(b"");
        all_vals.sort();
        assert_eq!(all_vals, vec![1, 2, 3, 4]);

        let none: Vec<i32> = m.values_with_prefix(b"xyz");
        assert!(none.is_empty());
    }

    #[test]
    fn test_map_for_each_value_with_prefix() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        m.insert(b"app", 1).unwrap();
        m.insert(b"apple", 2).unwrap();
        m.insert(b"application", 3).unwrap();
        m.insert(b"banana", 4).unwrap();

        // Zero-alloc callback must match values_with_prefix
        let mut callback_vals = Vec::new();
        m.for_each_value_with_prefix(b"app", |v| callback_vals.push(v));
        callback_vals.sort();
        assert_eq!(callback_vals, vec![1, 2, 3]);

        let mut all = Vec::new();
        m.for_each_value_with_prefix(b"", |v| all.push(v));
        all.sort();
        assert_eq!(all, vec![1, 2, 3, 4]);

        let mut none = Vec::new();
        m.for_each_value_with_prefix(b"xyz", |v| none.push(v));
        assert!(none.is_empty());
    }

    /// Performance test with map values
    #[test]
    fn test_map_value_performance() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        for i in 0..5000i32 {
            m.insert(format!("term_{:06}", i).as_bytes(), i).unwrap();
        }
        assert_eq!(m.len(), 5000);

        for i in 0..5000i32 {
            assert_eq!(m.get(format!("term_{:06}", i).as_bytes()), Some(i));
        }

        let prefix_vals = m.values_with_prefix(b"term_001");
        assert_eq!(
            prefix_vals.len(),
            1000,
            "prefix 'term_001' should yield 1000 values, got {}",
            prefix_vals.len()
        );
    }
}

#[cfg(test)]
mod prefix_regression_tests {
    use super::super::iterators::*;
    use super::super::map::*;
    use super::super::state::*;
    use super::super::trie::*;
    use super::*;

    #[test]
    fn test_1000_terms_prefix() {
        let mut t = DoubleArrayTrie::new();
        for i in 0..1000u32 {
            let term = format!("term_{:04}", i);
            t.insert(term.as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 1000, "expected 1000 keys, got {}", t.len());

        // Verify all terms exist
        let mut missing = Vec::new();
        for i in 0..1000u32 {
            let term = format!("term_{:04}", i);
            if !t.contains(term.as_bytes()) {
                missing.push(i);
            }
        }
        assert!(
            missing.is_empty(),
            "missing {} terms: {:?}",
            missing.len(),
            &missing[..missing.len().min(20)]
        );

        let result = t.keys_with_prefix(b"term_00");
        assert_eq!(
            result.len(),
            100,
            "prefix 'term_00' returned {} (expected 100)",
            result.len()
        );
    }
}

#[cfg(test)]
mod map_prefix_regression_tests {
    use super::super::iterators::*;
    use super::super::map::*;
    use super::super::state::*;
    use super::super::trie::*;
    use super::*;

    #[test]
    fn test_map_1000_terms_prefix_fresh_trie() {
        // Simulate the engine's flush_to_trie: build fresh trie from sorted entries
        let mut entries: Vec<(String, u32)> = (0..1000u32)
            .map(|i| (format!("term_{:04}", i), i))
            .collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut trie = DoubleArrayTrieMap::with_capacity(entries.len());
        for (term, id) in &entries {
            trie.insert(term.as_bytes(), *id).expect("insert failed");
        }

        assert_eq!(trie.len(), 1000);

        // Verify all lookups
        for i in 0..1000u32 {
            let term = format!("term_{:04}", i);
            assert_eq!(
                trie.get(term.as_bytes()),
                Some(i),
                "get failed for {}",
                term
            );
        }

        // Verify prefix
        let result = trie.values_with_prefix(b"term_00");
        assert_eq!(
            result.len(),
            100,
            "values_with_prefix 'term_00' returned {} (expected 100)",
            result.len()
        );
    }

    // --- Additional corner case tests ---

    #[test]
    fn test_all_256_single_byte_keys() {
        let mut t = DoubleArrayTrie::new();
        for b in 0u8..=255 {
            t.insert(&[b]).unwrap();
        }
        assert_eq!(t.len(), 256);
        for b in 0u8..=255 {
            assert!(t.contains(&[b]), "missing single-byte key 0x{:02x}", b);
        }
    }

    #[test]
    fn test_insert_after_shrink() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"hello").unwrap();
        t.insert(b"world").unwrap();
        t.shrink_to_fit();
        // Insert more after shrinking
        t.insert(b"foo").unwrap();
        t.insert(b"bar").unwrap();
        assert_eq!(t.len(), 4);
        assert!(t.contains(b"hello"));
        assert!(t.contains(b"world"));
        assert!(t.contains(b"foo"));
        assert!(t.contains(b"bar"));
    }

    #[test]
    fn test_map_values_empty_key() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        m.insert(b"", 42).unwrap();
        assert_eq!(m.get(b""), Some(42));
        m.insert(b"a", 1).unwrap();
        assert_eq!(m.get(b""), Some(42));
        assert_eq!(m.get(b"a"), Some(1));
    }

    #[test]
    fn test_cursor_after_remove() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"a").unwrap();
        t.insert(b"b").unwrap();
        t.insert(b"c").unwrap();
        t.remove(b"b");

        let mut c = t.cursor();
        c.seek_begin();
        assert_eq!(c.key(), b"a");
        assert!(c.next());
        assert_eq!(c.key(), b"c");
        assert!(!c.next());
    }

    #[test]
    fn test_keys_with_prefix_empty_trie() {
        let t = DoubleArrayTrie::new();
        assert_eq!(t.keys_with_prefix(b"").len(), 0);
        assert_eq!(t.keys_with_prefix(b"anything").len(), 0);
        assert_eq!(t.keys().len(), 0);
    }

    #[test]
    fn test_map_empty_key() {
        let mut map = DoubleArrayTrieMap::<u32>::new();
        map.insert(b"", 99).unwrap();
        assert_eq!(map.get(b""), Some(99));
        assert_eq!(map.len(), 1);
        map.insert(b"x", 1).unwrap();
        assert_eq!(map.get(b""), Some(99));
        assert_eq!(map.get(b"x"), Some(1));
    }

    #[test]
    fn test_remove_all_then_reinsert() {
        let mut t = DoubleArrayTrie::new();
        for i in 0..50u32 {
            t.insert(format!("k{}", i).as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 50);
        // Remove all
        for i in 0..50u32 {
            assert!(t.remove(format!("k{}", i).as_bytes()));
        }
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        // Reinsert
        for i in 0..50u32 {
            t.insert(format!("k{}", i).as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 50);
        for i in 0..50u32 {
            assert!(t.contains(format!("k{}", i).as_bytes()));
        }
    }

    #[test]
    fn test_range_after_remove() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"a").unwrap();
        t.insert(b"b").unwrap();
        t.insert(b"c").unwrap();
        t.insert(b"d").unwrap();
        t.remove(b"b");
        t.remove(b"c");

        let range: Vec<Vec<u8>> = t.range(b"a", b"z").collect();
        assert_eq!(range.len(), 2);
        assert_eq!(range[0], b"a");
        assert_eq!(range[1], b"d");
    }

    #[test]
    fn test_map_values_empty_prefix_empty() {
        let m = DoubleArrayTrieMap::<i32>::new();
        let vals = m.values_with_prefix(b"");
        assert_eq!(vals.len(), 0);
    }

    // --- FuzzyIterator structural / performance regression tests ---

    #[test]
    fn test_fuzzy_compute_row_inplace_matches_reference() {
        fn reference_row(prev: &[usize], query: &[u8], c: u8) -> Vec<usize> {
            let mut row = vec![0; query.len() + 1];
            row[0] = prev[0] + 1;
            for j in 1..=query.len() {
                let cost = if query[j - 1] == c { 0 } else { 1 };
                row[j] = (prev[j] + 1).min(row[j - 1] + 1).min(prev[j - 1] + cost);
            }
            row
        }

        let queries: &[&[u8]] = &[b"cat", b"", b"a", b"abcdefghij", b"\x00\xff"];
        let labels: &[u8] = &[b'a', b'c', b'z', 0x00, 0xFF];

        for query in queries {
            let prev: Vec<usize> = (0..=query.len()).collect();
            for &label in labels {
                let expected = reference_row(&prev, query, label);
                let mut reused_buf = Vec::new();
                let min_val =
                    FuzzyIterator::<i32>::compute_row_inplace(&prev, query, label, &mut reused_buf);
                assert_eq!(
                    reused_buf, expected,
                    "mismatch for query={:?} label={}",
                    query, label
                );
                assert_eq!(min_val, *expected.iter().min().unwrap());
            }
        }

        // Multi-depth chain: row0 → row1 → row2, verifying chained computation
        let query = b"cat";
        let row0: Vec<usize> = (0..=query.len()).collect();
        let mut buf = Vec::new();
        FuzzyIterator::<i32>::compute_row_inplace(&row0, query, b'c', &mut buf);
        let row1 = buf.clone();
        FuzzyIterator::<i32>::compute_row_inplace(&row1, query, b'a', &mut buf);
        let row2 = buf.clone();
        FuzzyIterator::<i32>::compute_row_inplace(&row2, query, b't', &mut buf);
        // "cat" vs "cat" at depth 3 → last element should be 0
        assert_eq!(buf[query.len()], 0);
    }

    #[test]
    fn test_fuzzy_compute_row_inplace_reuses_buffer() {
        let query = b"hello";
        let row0: Vec<usize> = (0..=query.len()).collect();

        let mut buf = Vec::with_capacity(query.len() + 1);
        let ptr_before = buf.as_ptr();
        FuzzyIterator::<i32>::compute_row_inplace(&row0, query, b'h', &mut buf);
        let ptr_after = buf.as_ptr();
        // Vec::resize on a buffer with sufficient capacity does not reallocate
        assert_eq!(ptr_before, ptr_after, "buffer should reuse its allocation");

        // Second call reuses same buffer
        FuzzyIterator::<i32>::compute_row_inplace(&buf.clone(), query, b'e', &mut buf);
        // capacity was already sufficient, no realloc
        assert_eq!(buf.len(), query.len() + 1);
    }

    #[test]
    fn test_fuzzy_spare_rows_recycling() {
        // Build a trie with enough branching to force DFS pop+push cycles
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        let words = [
            "abc", "abd", "abe", "acd", "ace", "bcd", "bce", "bde", "cde", "xyz", "xyw", "xyv",
        ];
        for (i, w) in words.iter().enumerate() {
            trie.insert(w.as_bytes(), i as i32).unwrap();
        }

        let mut iter = trie.iter_fuzzy(b"abc", 2);

        // Exhaust the iterator
        let results: Vec<_> = iter.by_ref().collect();

        // After full traversal, spare_rows should have recycled buffers
        // (DFS pops recycle rows instead of dropping them)
        assert!(
            !iter.spare_rows.is_empty(),
            "spare_rows should contain recycled buffers after traversal, got 0"
        );

        // Every spare row should have the correct length (query.len() + 1 = 4)
        for row in &iter.spare_rows {
            assert_eq!(row.len(), 4, "recycled row has wrong length");
        }

        // Results should still be correct despite recycling
        fn edit_distance(a: &[u8], b: &[u8]) -> usize {
            let (m, n) = (a.len(), b.len());
            let mut dp = vec![vec![0usize; n + 1]; m + 1];
            for i in 0..=m {
                dp[i][0] = i;
            }
            for j in 0..=n {
                dp[0][j] = j;
            }
            for i in 1..=m {
                for j in 1..=n {
                    let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                    dp[i][j] = (dp[i - 1][j] + 1)
                        .min(dp[i][j - 1] + 1)
                        .min(dp[i - 1][j - 1] + cost);
                }
            }
            dp[m][n]
        }
        for (key, _) in &results {
            assert!(edit_distance(key, b"abc") <= 2);
        }
        for w in &words {
            if edit_distance(w.as_bytes(), b"abc") <= 2 {
                assert!(
                    results.iter().any(|(k, _)| k == w.as_bytes()),
                    "missing {:?}",
                    w
                );
            }
        }
    }

    #[test]
    fn test_fuzzy_recycling_correctness_large_trie() {
        // Stress test: large trie ensures heavy recycling, then verify
        // results match brute-force reference exactly.
        let mut trie = DoubleArrayTrieMap::<i32>::new();
        let mut words: Vec<String> = Vec::new();
        for a in b'a'..=b'e' {
            for b in b'a'..=b'e' {
                for c in b'a'..=b'e' {
                    let w = format!("{}{}{}", a as char, b as char, c as char);
                    words.push(w);
                }
            }
        }
        // 125 three-letter words from {a..e}^3
        for (i, w) in words.iter().enumerate() {
            trie.insert(w.as_bytes(), i as i32).unwrap();
        }

        fn edit_distance(a: &[u8], b: &[u8]) -> usize {
            let (m, n) = (a.len(), b.len());
            let mut dp = vec![vec![0usize; n + 1]; m + 1];
            for i in 0..=m {
                dp[i][0] = i;
            }
            for j in 0..=n {
                dp[0][j] = j;
            }
            for i in 1..=m {
                for j in 1..=n {
                    let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                    dp[i][j] = (dp[i - 1][j] + 1)
                        .min(dp[i][j - 1] + 1)
                        .min(dp[i - 1][j - 1] + cost);
                }
            }
            dp[m][n]
        }

        let queries: &[&[u8]] = &[b"abc", b"eee", b"cba", b"aaa"];
        for &query in queries {
            for max_dist in 0..=3 {
                let mut iter = trie.iter_fuzzy(query, max_dist);
                let results: Vec<_> = iter.by_ref().collect();

                // Verify completeness and soundness against brute-force
                let mut expected: Vec<&str> = words
                    .iter()
                    .filter(|w| edit_distance(w.as_bytes(), query) <= max_dist)
                    .map(|w| w.as_str())
                    .collect();
                expected.sort();

                let mut actual_keys: Vec<String> = results
                    .iter()
                    .map(|(k, _)| String::from_utf8(k.clone()).unwrap())
                    .collect();
                actual_keys.sort();

                assert_eq!(
                    actual_keys,
                    expected,
                    "mismatch for query={:?} max_dist={}",
                    std::str::from_utf8(query).unwrap(),
                    max_dist
                );

                // Verify recycling happened on non-trivial traversals
                if max_dist >= 1 {
                    assert!(
                        !iter.spare_rows.is_empty(),
                        "expected recycling for query={:?} max_dist={}",
                        std::str::from_utf8(query).unwrap(),
                        max_dist
                    );
                }
            }
        }
    }
}
