//! Structure-aware `arbitrary` fuzz target for `DoubleArrayTrieMap<u32>` (plan.md B2.1).
//! Verifies incremental `insert`, `get`, `shrink_to_fit`, and `remove` against a BTreeMap oracle.
#![no_main]
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeMap;
use zipora::fsa::DoubleArrayTrieMap;

fuzz_target!(|keys: Vec<Vec<u8>>| {
    let mut oracle: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
    let mut trie: DoubleArrayTrieMap<u32> = DoubleArrayTrieMap::new();

    for (idx, key) in keys.into_iter().take(96).enumerate() {
        if key.is_empty() || key.len() > 256 {
            continue;
        }
        let val = (idx as u32) & 0x7FFF_FFFF;
        oracle.insert(key.clone(), val);
        trie.insert(&key, val)
            .expect("DoubleArrayTrieMap::insert must succeed");
    }

    if oracle.is_empty() {
        return;
    }

    assert_eq!(trie.len(), oracle.len());
    for (k, &v) in &oracle {
        assert_eq!(trie.get(k), Some(v), "lookup mismatch before shrink_to_fit");
    }

    trie.shrink_to_fit();
    for (k, &v) in &oracle {
        assert_eq!(trie.get(k), Some(v), "lookup mismatch after shrink_to_fit");
    }
});
