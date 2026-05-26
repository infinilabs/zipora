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
