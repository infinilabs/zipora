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
    assert_eq!(root_view.is_final(), false);
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
    trie.mempool.push(PatriciaNode { meta: MetaInfo { flags: 0 | 0x10, n_zpath_len: 0, c_label: [0, 0] } });
    trie.mempool.push(PatriciaNode { bytes: [10, 0, 0, 0] }); // value = 10
    
    // Node 2: leaf for "b" (valpos = slot 302)
    let node_b_state = trie.mempool.len() as u32;
    trie.mempool.push(PatriciaNode { meta: MetaInfo { flags: 0 | 0x10, n_zpath_len: 0, c_label: [0, 0] } });
    trie.mempool.push(PatriciaNode { bytes: [20, 0, 0, 0] }); // value = 20
    
    // Node 3: leaf for "ccc" (zpath="cc", valpos = slot 304)
    // Needs 1 slot [meta] + zpath bytes + value slot
    let node_c_state = trie.mempool.len() as u32;
    trie.mempool.push(PatriciaNode { meta: MetaInfo { flags: 0 | 0x10, n_zpath_len: 2, c_label: [0, 0] } });
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
