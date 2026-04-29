//! Comprehensive integration tests for CsppTrie

use zipora::fsa::cspp_trie::CsppTrie;
use rand::Rng;

#[test]
fn test_stress_1000_random_keys() {
    let mut trie = CsppTrie::new(8);
    let mut rng = rand::rng();
    let mut keys = Vec::new();

    // Generate 1000 random keys
    for _ in 0..1000 {
        let len = rng.random_range(1..=50);
        let key: Vec<u8> = (0..len).map(|_| rng.random()).collect();
        keys.push(key);
    }

    // Insert all keys
    for key in &keys {
        let (is_new, valpos) = trie.insert(key);
        if is_new {
            trie.set_value(valpos, key.len() as u64);
        }
    }

    // Verify all keys are found
    for key in &keys {
        assert!(trie.contains(key), "key should be found: {:?}", key);
        let valpos = trie.lookup(key).expect("lookup should succeed");
        let stored_len: u64 = trie.get_value(valpos);
        assert_eq!(stored_len, key.len() as u64);
    }

    // Verify count (may be less than 1000 if duplicates)
    assert!(trie.num_words() <= 1000);
    assert!(trie.num_words() > 0);
}

#[test]
fn test_binary_keys_all_bytes() {
    let mut trie = CsppTrie::new(4);
    let mut keys = Vec::new();

    // Create keys containing all 256 byte values
    for i in 0u8..=255 {
        let key = vec![b'k', i, b'e', b'y'];
        keys.push(key);
    }

    // Insert all keys
    for key in &keys {
        let (is_new, valpos) = trie.insert(key);
        assert!(is_new);
        trie.set_value(valpos, key[1] as u32);
    }

    assert_eq!(trie.num_words(), 256);

    // Verify all keys
    for key in &keys {
        assert!(trie.contains(key));
        let valpos = trie.lookup(key).unwrap();
        let val: u32 = trie.get_value(valpos);
        assert_eq!(val, key[1] as u32);
    }
}

#[test]
fn test_sequential_pattern_keys() {
    let mut trie = CsppTrie::new(8);
    let n = 1000;

    // Insert sequential keys
    for i in 0..n {
        let key = format!("key{:06}", i);
        let (is_new, valpos) = trie.insert(key.as_bytes());
        assert!(is_new);
        trie.set_value(valpos, i as u64);
    }

    assert_eq!(trie.num_words(), n);

    // Verify all keys
    for i in 0..n {
        let key = format!("key{:06}", i);
        assert!(trie.contains(key.as_bytes()));
        let valpos = trie.lookup(key.as_bytes()).unwrap();
        let val: u64 = trie.get_value(valpos);
        assert_eq!(val, i as u64);
    }

    // Verify non-existent keys
    assert!(!trie.contains(format!("key{:06}", n).as_bytes()));
    assert!(!trie.contains(b"key"));
    assert!(!trie.contains(b"key0000000"));
}

#[test]
fn test_pathological_single_char_nesting() {
    let mut trie = CsppTrie::new(4);

    // Insert "a", "aa", "aaa", ..., "a"×100
    for i in 1..=100 {
        let key = vec![b'a'; i];
        let (is_new, valpos) = trie.insert(&key);
        assert!(is_new);
        trie.set_value(valpos, i as u32);
    }

    assert_eq!(trie.num_words(), 100);

    // Verify all lengths
    for i in 1..=100 {
        let key = vec![b'a'; i];
        assert!(trie.contains(&key));
        let valpos = trie.lookup(&key).unwrap();
        let val: u32 = trie.get_value(valpos);
        assert_eq!(val, i as u32);
    }

    // Verify non-existent
    assert!(!trie.contains(&vec![b'a'; 101]));
    assert!(!trie.contains(b"b"));
}

#[test]
fn test_interleaved_insert_lookup() {
    let mut trie = CsppTrie::new(8);
    let n = 500;

    for i in 0..n {
        let key = format!("item{:04}", i);

        // Insert
        let (is_new, valpos) = trie.insert(key.as_bytes());
        assert!(is_new);
        trie.set_value(valpos, (i * 2) as u64);

        // Immediately lookup
        assert!(trie.contains(key.as_bytes()));
        let retrieved_valpos = trie.lookup(key.as_bytes()).unwrap();
        assert_eq!(valpos, retrieved_valpos);
        let val: u64 = trie.get_value(retrieved_valpos);
        assert_eq!(val, (i * 2) as u64);

        // Lookup all previously inserted keys
        for j in 0..=i {
            let prev_key = format!("item{:04}", j);
            assert!(trie.contains(prev_key.as_bytes()));
        }
    }

    assert_eq!(trie.num_words(), n);
}

#[test]
fn test_memory_efficiency_tracking() {
    let mut trie = CsppTrie::new(8);

    let stat_initial = trie.mem_get_stat();
    let initial_used = stat_initial.used_size;

    // Batch insert
    for i in 0..200 {
        let key = format!("data{:05}", i);
        trie.insert(key.as_bytes());
    }

    let stat_after = trie.mem_get_stat();

    assert!(stat_after.used_size > initial_used, "used size should increase");
    assert!(stat_after.capacity >= stat_after.used_size, "capacity should be sufficient");
    assert_eq!(trie.num_words(), 200);

    // Check fragmentation is reasonable
    let frag = trie.mem_frag_size();
    assert!(frag <= stat_after.used_size, "fragmentation should not exceed used size");
}

#[test]
fn test_large_values_u128() {
    let mut trie = CsppTrie::new(16);

    let keys = [
        b"alpha".as_slice(),
        b"beta".as_slice(),
        b"gamma".as_slice(),
        b"delta".as_slice(),
    ];

    let values = [
        0x0123456789ABCDEFu128,
        0xFEDCBA9876543210u128,
        u128::MAX,
        u128::MIN,
    ];

    // Insert and set values
    for (key, &value) in keys.iter().zip(values.iter()) {
        let (is_new, valpos) = trie.insert(key);
        assert!(is_new);
        trie.set_value(valpos, value);
    }

    assert_eq!(trie.num_words(), 4);

    // Verify all values
    for (key, &expected) in keys.iter().zip(values.iter()) {
        let valpos = trie.lookup(key).unwrap();
        let retrieved: u128 = trie.get_value(valpos);
        assert_eq!(retrieved, expected);
    }
}

#[test]
fn test_keys_with_null_bytes() {
    let mut trie = CsppTrie::new(4);

    let keys = [
        b"before\x00after".to_vec(),
        b"\x00start".to_vec(),
        b"end\x00".to_vec(),
        b"\x00\x00\x00".to_vec(),
        b"mid\x00\x00dle".to_vec(),
    ];

    // Insert all keys
    for (i, key) in keys.iter().enumerate() {
        let (is_new, valpos) = trie.insert(key);
        assert!(is_new);
        trie.set_value(valpos, i as u32);
    }

    assert_eq!(trie.num_words(), keys.len());

    // Verify all keys with null bytes
    for (i, key) in keys.iter().enumerate() {
        assert!(trie.contains(key), "should contain key with null bytes");
        let valpos = trie.lookup(key).unwrap();
        let val: u32 = trie.get_value(valpos);
        assert_eq!(val, i as u32);
    }

    // Verify similar keys without null bytes are not found
    assert!(!trie.contains(b"beforeafter"));
    assert!(!trie.contains(b"start"));
    assert!(!trie.contains(b"end"));
}
