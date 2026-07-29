//! Fuzz DoubleArrayTrie insert/relocate + lookup consistency (plan.md 6.1, H16).
#![no_main]
use libfuzzer_sys::fuzz_target;
use zipora::fsa::DoubleArrayTrie;

fuzz_target!(|data: &[u8]| {
    // Split into keys on 0xFF, cap count/length
    let mut keys: Vec<&[u8]> = data.splitn(256, |&b| b == 0xFF).collect();
    keys.retain(|k| !k.is_empty() && k.len() <= 512);
    keys.sort();
    keys.dedup();
    if keys.is_empty() {
        return;
    }
    let mut trie = DoubleArrayTrie::new();
    for k in &keys {
        trie.insert(k).expect("insert must not fail");
    }
    for k in &keys {
        assert!(trie.contains(k), "inserted key lost: {:?}", k);
    }
});
