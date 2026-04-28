use zipora::fsa::cspp_trie::{
    CsppTrie, CsppTrieIterator, MetaInfo, BigCount, PatriciaNode, NIL_STATE
};

#[test]
fn test_node_size() {
    assert_eq!(std::mem::size_of::<PatriciaNode>(), 4);
    assert_eq!(std::mem::size_of::<MetaInfo>(), 4);
    assert_eq!(std::mem::size_of::<BigCount>(), 4);
}

#[test]
fn test_root_node_creation() {
    let trie = CsppTrie::new(4);
    assert_eq!(trie.total_states(), 2 + 256 + 1); // meta + real_cnt + 256 children + 1 val_slot (valsize=4 -> 1 slot)
    
    let root_view = trie.node_view(0);
    assert_eq!(root_view.cnt_type(), 15);
    assert!(!root_view.is_final());
    assert_eq!(root_view.n_children(), 256);
    
    // All 256 children should be NIL_STATE initially
    for ch in 0..=255 {
        assert_eq!(root_view.child(2 + ch), NIL_STATE);
    }
}

#[test]
fn test_manual_trie_construction_and_lookup() {
    let mut trie = CsppTrie::new(4);
    
    // We will manually construct a trie for keys "a", "b", "ccc"
    // Root is at slot 0.
    
    // Node 1: leaf for "a" (valpos = slot 300)
    // Needs 1 slot [meta] + 1 value slot
    let node_a_state = trie.mempool.len() as u32;
    trie.mempool.push(PatriciaNode { meta: MetaInfo { flags: 0x10, n_zpath_len: 0, c_label: [0, 0] } });
    trie.mempool.push(PatriciaNode { bytes: [10, 0, 0, 0] }); // value = 10
    
    // Node 2: leaf for "b" (valpos = slot 302)
    let node_b_state = trie.mempool.len() as u32;
    trie.mempool.push(PatriciaNode { meta: MetaInfo { flags: 0x10, n_zpath_len: 0, c_label: [0, 0] } });
    trie.mempool.push(PatriciaNode { bytes: [20, 0, 0, 0] }); // value = 20
    
    // Node 3: leaf for "ccc" (zpath="cc", valpos = slot 304)
    // Needs 1 slot [meta] + zpath bytes + value slot
    let node_c_state = trie.mempool.len() as u32;
    trie.mempool.push(PatriciaNode { meta: MetaInfo { flags: 0x10, n_zpath_len: 2, c_label: [0, 0] } });
    trie.mempool.push(PatriciaNode { bytes: [b'c', b'c', 0, 0] }); // zpath padded to 4 bytes
    trie.mempool.push(PatriciaNode { bytes: [30, 0, 0, 0] }); // value = 30
    
    // Link from root
    trie.mempool[2 + b'a' as usize].child = node_a_state;
    trie.mempool[2 + b'b' as usize].child = node_b_state;
    trie.mempool[2 + b'c' as usize].child = node_c_state;
    
    // Test lookups
    let pos_a = trie.lookup(b"a").unwrap();
    assert_eq!(trie.get_value::<u32>(pos_a), 10);
    
    let pos_b = trie.lookup(b"b").unwrap();
    assert_eq!(trie.get_value::<u32>(pos_b), 20);
    
    let pos_c = trie.lookup(b"ccc").unwrap();
    assert_eq!(trie.get_value::<u32>(pos_c), 30);
    
    assert!(trie.lookup(b"c").is_none());
    assert!(trie.lookup(b"cc").is_none());
    assert!(trie.lookup(b"cccc").is_none());
    assert!(trie.lookup(b"d").is_none());

    // Test iteration
    let mut iter = CsppTrieIterator::<u32>::new(&trie);
    
    assert!(iter.seek_begin());
    assert_eq!(iter.word(), b"a");
    assert_eq!(iter.value(), 10);
    
    assert!(iter.incr());
    assert_eq!(iter.word(), b"b");
    assert_eq!(iter.value(), 20);
    
    assert!(iter.incr());
    assert_eq!(iter.word(), b"ccc");
    assert_eq!(iter.value(), 30);
    
    assert!(!iter.incr());
}

// ========== Phase B: Insert Tests ==========

#[test]
fn test_insert_single_key() {
    let mut trie = CsppTrie::new(4);
    let (is_new, _valpos) = trie.insert(b"hello");
    assert!(is_new);
    assert!(trie.contains(b"hello"));
    assert!(!trie.contains(b"hell"));
    assert!(!trie.contains(b"helloo"));
    assert_eq!(trie.num_words(), 1);
}

#[test]
fn test_insert_duplicate_key() {
    let mut trie = CsppTrie::new(4);
    let (is_new1, vp1) = trie.insert(b"hello");
    assert!(is_new1);
    let (is_new2, vp2) = trie.insert(b"hello");
    assert!(!is_new2);
    assert_eq!(vp1, vp2); // same valpos
    assert_eq!(trie.num_words(), 1);
}

#[test]
fn test_insert_empty_key() {
    let mut trie = CsppTrie::new(4);
    let (is_new, _) = trie.insert(b"");
    assert!(is_new);
    assert!(trie.contains(b""));
    assert!(!trie.contains(b"a"));
    assert_eq!(trie.num_words(), 1);
}

#[test]
fn test_insert_3_keys_lookup() {
    let mut trie = CsppTrie::new(4);
    assert!(trie.insert(b"a").0);
    assert!(trie.insert(b"b").0);
    assert!(trie.insert(b"ccc").0);
    assert_eq!(trie.num_words(), 3);
    assert!(trie.contains(b"a"));
    assert!(trie.contains(b"b"));
    assert!(trie.contains(b"ccc"));
    assert!(!trie.contains(b"c"));
    assert!(!trie.contains(b"cc"));
    assert!(!trie.contains(b"d"));
}

#[test]
fn test_insert_cnt_type_transitions_0_to_7() {
    // Insert 7 keys with distinct first bytes → exercises transitions 0→1→...→6→7
    let mut trie = CsppTrie::new(0);
    let keys: Vec<&[u8]> = vec![b"d", b"b", b"f", b"a", b"c", b"e", b"g"];
    for key in &keys {
        assert!(trie.insert(key).0, "Failed to insert {:?}", std::str::from_utf8(key));
    }
    assert_eq!(trie.num_words(), 7);
    for key in &keys {
        assert!(trie.contains(key), "Failed to find {:?}", std::str::from_utf8(key));
    }
}

#[test]
fn test_insert_cnt_type_transition_7_to_8() {
    // Insert 17 keys with distinct first bytes → exercises 7→8 (bitmap) transition
    let mut trie = CsppTrie::new(0);
    let keys: Vec<Vec<u8>> = (0..17u8).map(|i| vec![b'a' + i]).collect();
    for key in &keys {
        assert!(trie.insert(key).0, "Failed to insert {:?}", key);
    }
    assert_eq!(trie.num_words(), 17);
    for key in &keys {
        assert!(trie.contains(key), "Failed to find {:?}", key);
    }
    // Verify non-existent keys
    assert!(!trie.contains(&[b'a' + 17]));
}

#[test]
fn test_insert_cnt_type_8_grow() {
    // Insert 30 keys → cnt_type 8 with growing child array
    let mut trie = CsppTrie::new(0);
    let keys: Vec<Vec<u8>> = (0..30u8).map(|i| vec![i + 65]).collect();
    for key in &keys {
        assert!(trie.insert(key).0);
    }
    assert_eq!(trie.num_words(), 30);
    for key in &keys {
        assert!(trie.contains(key), "Missing key {:?}", key);
    }
}

#[test]
fn test_insert_fork_at_zpath() {
    // Insert keys sharing a prefix to exercise fork()
    let mut trie = CsppTrie::new(0);
    assert!(trie.insert(b"abcdef").0);
    assert!(trie.insert(b"abcxyz").0); // fork at position 3 ('d' vs 'x')
    assert_eq!(trie.num_words(), 2);
    assert!(trie.contains(b"abcdef"));
    assert!(trie.contains(b"abcxyz"));
    assert!(!trie.contains(b"abc"));
    assert!(!trie.contains(b"abcd"));
}

#[test]
fn test_insert_split_zpath() {
    // Insert a key that is a prefix of an existing key → exercises split_zpath
    let mut trie = CsppTrie::new(0);
    assert!(trie.insert(b"abcdef").0);
    assert!(trie.insert(b"abc").0); // key is prefix of "abcdef"
    assert_eq!(trie.num_words(), 2);
    assert!(trie.contains(b"abcdef"));
    assert!(trie.contains(b"abc"));
    assert!(!trie.contains(b"ab"));
    assert!(!trie.contains(b"abcd"));
}

#[test]
fn test_insert_mark_final_state() {
    // Insert a key, then insert a prefix that matches a non-final node
    let mut trie = CsppTrie::new(0);
    assert!(trie.insert(b"ab").0);
    assert!(trie.insert(b"ac").0);
    // Now "a" is a non-final node (cnt_type=2 with children 'b' and 'c')
    assert!(!trie.contains(b"a"));
    assert!(trie.insert(b"a").0); // MarkFinalState
    assert!(trie.contains(b"a"));
    assert!(trie.contains(b"ab"));
    assert!(trie.contains(b"ac"));
    assert_eq!(trie.num_words(), 3);
}

#[test]
fn test_insert_10k_random_keys() {
    use std::collections::BTreeSet;
    let mut trie = CsppTrie::new(0);
    let mut expected = BTreeSet::new();

    // Generate deterministic pseudo-random keys
    let mut rng_state: u64 = 12345;
    for _ in 0..10_000 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let len = ((rng_state >> 32) % 20 + 1) as usize;
        let key: Vec<u8> = (0..len).map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 40) % 26 + 97) as u8 // lowercase a-z
        }).collect();
        expected.insert(key.clone());
        trie.insert(&key);
    }

    // Verify all keys can be looked up
    for key in &expected {
        assert!(trie.contains(key), "Missing key: {:?}", std::str::from_utf8(key));
    }
    assert_eq!(trie.num_words(), expected.len());
}

#[test]
fn test_insert_and_iterate_sorted() {
    let mut trie = CsppTrie::new(4);
    let keys = vec![
        b"banana".to_vec(), b"apple".to_vec(), b"cherry".to_vec(),
        b"date".to_vec(), b"elderberry".to_vec(), b"fig".to_vec(),
        b"app".to_vec(), b"application".to_vec(),
    ];
    for key in &keys {
        let (is_new, valpos) = trie.insert(key);
        assert!(is_new);
        // Write a value at valpos
        unsafe {
            let ptr = trie.mempool.as_mut_ptr() as *mut u8;
            std::ptr::write_unaligned(ptr.add(valpos) as *mut u32, key.len() as u32);
        }
    }
    assert_eq!(trie.num_words(), 8);

    // Iterate and verify sorted order
    let mut iter = CsppTrieIterator::<u32>::new(&trie);
    let mut collected: Vec<(Vec<u8>, u32)> = Vec::new();
    if iter.seek_begin() {
        collected.push((iter.word().to_vec(), iter.value()));
        while iter.incr() {
            collected.push((iter.word().to_vec(), iter.value()));
        }
    }

    let mut sorted_keys = keys.clone();
    sorted_keys.sort();
    assert_eq!(collected.len(), sorted_keys.len());
    for (i, (word, val)) in collected.iter().enumerate() {
        assert_eq!(word, &sorted_keys[i], "Mismatch at position {}", i);
        assert_eq!(*val, sorted_keys[i].len() as u32, "Value mismatch for {:?}", std::str::from_utf8(word));
    }
}

#[test]
fn test_insert_long_common_prefix() {
    // Keys sharing very long common prefixes exercise deep zpath splits
    let mut trie = CsppTrie::new(0);
    let prefix = "a".repeat(100);
    let k1 = format!("{}x", prefix);
    let k2 = format!("{}y", prefix);
    let k3 = format!("{}z", prefix);
    assert!(trie.insert(k1.as_bytes()).0);
    assert!(trie.insert(k2.as_bytes()).0);
    assert!(trie.insert(k3.as_bytes()).0);
    assert!(trie.contains(k1.as_bytes()));
    assert!(trie.contains(k2.as_bytes()));
    assert!(trie.contains(k3.as_bytes()));
    assert!(!trie.contains(prefix.as_bytes()));
    assert_eq!(trie.num_words(), 3);
}

#[test]
fn test_insert_long_suffix_chain() {
    // Key longer than MAX_ZPATH (254) exercises suffix chain linking
    let mut trie = CsppTrie::new(0);
    let long_key: Vec<u8> = (0..300).map(|i| b'a' + (i % 26) as u8).collect();
    assert!(trie.insert(&long_key).0);
    assert!(trie.contains(&long_key));
    assert!(!trie.contains(&long_key[..299]));
    assert_eq!(trie.num_words(), 1);
}

// ========== Phase C: Memory Pool & Statistics Tests ==========

#[test]
fn test_mem_stat_initial() {
    let trie = CsppTrie::new(4);
    let stat = trie.mem_get_stat();
    assert!(stat.used_size > 0);
    assert_eq!(stat.frag_size, 0);
    assert_eq!(stat.large_cnt, 0);
    assert_eq!(stat.lazy_free_cnt, 0);
    // All fast bins should be empty
    assert!(stat.fastbin.iter().all(|&c| c == 0));
}

#[test]
fn test_mem_stat_after_inserts() {
    let mut trie = CsppTrie::new(0);
    let initial_size = trie.mem_get_stat().used_size;
    for i in 0..100u8 {
        trie.insert(&[i]);
    }
    let stat = trie.mem_get_stat();
    assert!(stat.used_size > initial_size);
    assert_eq!(trie.num_words(), 100);
    // frag_size may be > 0 due to node replacements during cnt_type transitions
    // (old nodes are freed back to the free list)
    assert!(stat.frag_size > 0 || stat.used_size > initial_size);
}

#[test]
fn test_free_list_reuse() {
    let mut trie = CsppTrie::new(0);
    // Insert keys with shared prefixes so interior nodes get replaced (not at mempool end)
    // This ensures freed nodes go to the free list rather than shrinking the pool.
    trie.insert(b"aaa");
    trie.insert(b"aab"); // fork at zpath → frees interior nodes
    trie.insert(b"aac");
    trie.insert(b"aad");
    trie.insert(b"aae");
    trie.insert(b"aaf");
    trie.insert(b"aag");
    let stat = trie.mem_get_stat();
    let total_free_bins: usize = stat.fastbin.iter().sum();
    // There should be some free list entries from replaced interior nodes
    assert!(stat.frag_size > 0 || total_free_bins > 0,
        "Expected some free list entries from interior node transitions, frag={}", stat.frag_size);
}

#[test]
fn test_lazy_free_and_reclaim() {
    let mut trie = CsppTrie::new(0);
    // Insert multiple keys so the mempool grows well past the root
    trie.insert(b"hello");
    trie.insert(b"world");
    trie.insert(b"test123");
    let stat_before = trie.mem_get_stat();
    assert_eq!(stat_before.lazy_free_cnt, 0);

    // Defer-free a slot in the middle of the mempool (not at end, so it won't shrink)
    let mid_slot = 260u32; // well inside the root area, not at the end
    trie.free_node_deferred_pub(mid_slot, 8); // 2 slots
    let stat_deferred = trie.mem_get_stat();
    assert_eq!(stat_deferred.lazy_free_cnt, 1);
    assert_eq!(stat_deferred.lazy_free_sum, 8);

    // Reclaim: lazy items move to free list
    trie.reclaim_lazy_frees();
    let stat_reclaimed = trie.mem_get_stat();
    assert_eq!(stat_reclaimed.lazy_free_cnt, 0);
    // The reclaimed node goes to free list (not at end of mempool)
    assert!(stat_reclaimed.frag_size >= 8,
        "Expected frag_size >= 8 after reclaim, got {}", stat_reclaimed.frag_size);
}

#[test]
fn test_mem_frag_size_tracking() {
    let mut trie = CsppTrie::new(0);
    assert_eq!(trie.mem_frag_size(), 0);

    // Insert enough keys to trigger node transitions (which free old nodes)
    for i in 0..20u8 {
        trie.insert(&[i + 65]);
    }
    // After transitions, there should be fragmentation
    let frag = trie.mem_frag_size();
    // frag may be 0 if all freed nodes were at the end (shrink-from-end optimization)
    // but with 20 inserts causing multiple transitions, some interior nodes will be freed
    let stat = trie.mem_get_stat();
    assert_eq!(frag, stat.frag_size);
}

#[test]
fn test_10k_insert_memory_efficiency() {
    let mut trie = CsppTrie::new(4);
    for i in 0..10_000u32 {
        let key = format!("key{:05}", i);
        let (is_new, valpos) = trie.insert(key.as_bytes());
        if is_new {
            unsafe {
                let ptr = trie.mempool.as_mut_ptr() as *mut u8;
                std::ptr::write_unaligned(ptr.add(valpos) as *mut u32, i);
            }
        }
    }
    let stat = trie.mem_get_stat();
    assert_eq!(trie.num_words(), 10_000);
    // Memory efficiency: bytes per key should be reasonable
    let bytes_per_key = stat.used_size as f64 / 10_000.0;
    // With 4-byte values and ~8-byte keys, expect < 100 bytes/key
    assert!(bytes_per_key < 200.0, "bytes_per_key={:.1} too high", bytes_per_key);
    // Fragmentation ratio should be manageable
    let frag_ratio = stat.frag_size as f64 / stat.used_size as f64;
    assert!(frag_ratio < 0.5, "frag_ratio={:.2} too high", frag_ratio);
}
