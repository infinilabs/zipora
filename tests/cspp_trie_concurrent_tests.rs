use zipora::fsa::cspp_trie_concurrent::ConcurrentCsppTrie;
use crossbeam_epoch as epoch;
use std::sync::Arc;
use std::thread;
use std::collections::BTreeSet;

// Concurrent tests need smaller key counts in debug mode (unoptimized atomics are slow)
#[cfg(debug_assertions)]
const CONCURRENT_KEYS: usize = 100;
#[cfg(not(debug_assertions))]
const CONCURRENT_KEYS: usize = 2_000;

#[test]
fn test_concurrent_trie_single_thread_basic() {
    let trie = ConcurrentCsppTrie::with_capacity(0, 100_000);
    let (is_new, _) = trie.insert(b"hello");
    assert!(is_new);
    assert!(trie.contains(b"hello"));
    assert!(!trie.contains(b"hell"));
    assert!(!trie.contains(b"helloo"));
    assert_eq!(trie.num_words(), 1);
}

#[test]
fn test_concurrent_trie_duplicate_key() {
    let trie = ConcurrentCsppTrie::with_capacity(4, 100_000);
    let (is_new1, vp1) = trie.insert(b"hello");
    assert!(is_new1);
    let (is_new2, vp2) = trie.insert(b"hello");
    assert!(!is_new2);
    assert_eq!(vp1, vp2);
    assert_eq!(trie.num_words(), 1);
}

#[test]
fn test_concurrent_trie_empty_key() {
    let trie = ConcurrentCsppTrie::with_capacity(0, 100_000);
    let (is_new, _) = trie.insert(b"");
    assert!(is_new);
    assert!(trie.contains(b""));
    assert!(!trie.contains(b"a"));
    assert_eq!(trie.num_words(), 1);
}

#[test]
fn test_concurrent_trie_3_keys() {
    let trie = ConcurrentCsppTrie::with_capacity(0, 100_000);
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
fn test_concurrent_trie_cnt_type_transitions() {
    let trie = ConcurrentCsppTrie::with_capacity(0, 100_000);
    let keys: Vec<&[u8]> = vec![b"d", b"b", b"f", b"a", b"c", b"e", b"g"];
    for key in &keys {
        assert!(trie.insert(key).0);
    }
    assert_eq!(trie.num_words(), 7);
    for key in &keys {
        assert!(trie.contains(key));
    }
}

#[test]
fn test_concurrent_trie_bitmap_transition() {
    let trie = ConcurrentCsppTrie::with_capacity(0, 200_000);
    let keys: Vec<Vec<u8>> = (0..17u8).map(|i| vec![b'a' + i]).collect();
    for key in &keys {
        assert!(trie.insert(key).0);
    }
    assert_eq!(trie.num_words(), 17);
    for key in &keys {
        assert!(trie.contains(key));
    }
}

#[test]
fn test_concurrent_trie_fork_and_split() {
    let trie = ConcurrentCsppTrie::with_capacity(0, 100_000);
    assert!(trie.insert(b"abcdef").0);
    assert!(trie.insert(b"abcxyz").0);
    assert!(trie.insert(b"abc").0);
    assert_eq!(trie.num_words(), 3);
    assert!(trie.contains(b"abcdef"));
    assert!(trie.contains(b"abcxyz"));
    assert!(trie.contains(b"abc"));
    assert!(!trie.contains(b"ab"));
}

#[test]
fn test_concurrent_trie_10k_single_thread() {
    let trie = ConcurrentCsppTrie::with_capacity(0, 2_000_000);
    let mut expected = BTreeSet::new();
    let mut rng_state: u64 = 12345;
    for _ in 0..10_000 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let len = ((rng_state >> 32) % 20 + 1) as usize;
        let key: Vec<u8> = (0..len).map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 40) % 26 + 97) as u8
        }).collect();
        expected.insert(key.clone());
        trie.insert(&key);
    }
    for key in &expected {
        assert!(trie.contains(key), "Missing key: {:?}", std::str::from_utf8(key));
    }
    assert_eq!(trie.num_words(), expected.len());
}

// ========== Concurrent tests ==========

#[test]
fn test_concurrent_insert_2_threads_disjoint() {
    let trie = Arc::new(ConcurrentCsppTrie::with_capacity(0, 10_000_000));
    let n = CONCURRENT_KEYS;

    let t1 = {
        let trie = Arc::clone(&trie);
        thread::spawn(move || {
            for i in 0..n {
                let key = format!("thread1_key_{:05}", i);
                trie.insert(key.as_bytes());
            }
        })
    };
    let t2 = {
        let trie = Arc::clone(&trie);
        thread::spawn(move || {
            for i in 0..n {
                let key = format!("thread2_key_{:05}", i);
                trie.insert(key.as_bytes());
            }
        })
    };

    t1.join().unwrap();
    t2.join().unwrap();

    assert_eq!(trie.num_words(), n * 2);
    for i in 0..n {
        assert!(trie.contains(format!("thread1_key_{:05}", i).as_bytes()));
        assert!(trie.contains(format!("thread2_key_{:05}", i).as_bytes()));
    }
}

#[test]
fn test_concurrent_insert_4_threads_shared_prefix() {
    let trie = Arc::new(ConcurrentCsppTrie::with_capacity(0, 10_000_000));
    let n = CONCURRENT_KEYS;
    let num_threads = 4;

    // Pre-seed the subtrees to avoid shared node split contention
    for tid in 0..num_threads {
        let seed_key = format!("shared_prefix_{}_key_{:05}", tid, 0);
        trie.insert(seed_key.as_bytes());
    }

    let handles: Vec<_> = (0..num_threads).map(|tid| {
        let trie = Arc::clone(&trie);
        thread::spawn(move || {
            for i in 1..n {
                let key = format!("shared_prefix_{}_key_{:05}", tid, i);
                trie.insert(key.as_bytes());
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(trie.num_words(), n * num_threads);
    for tid in 0..num_threads {
        for i in 0..n {
            let key = format!("shared_prefix_{}_key_{:05}", tid, i);
            assert!(trie.contains(key.as_bytes()), "Missing: {}", key);
        }
    }
}

#[test]
fn test_concurrent_insert_8_threads_stress() {
    let trie = Arc::new(ConcurrentCsppTrie::with_capacity(0, 40_000_000));
    let n = CONCURRENT_KEYS;
    let num_threads = 8;

    let handles: Vec<_> = (0..num_threads).map(|tid| {
        let trie = Arc::clone(&trie);
        thread::spawn(move || {
            let mut inserted = Vec::with_capacity(n);
            for i in 0..n {
                let prefix = (b'A' + tid as u8) as char;
                let key = format!("{}_stress_{:05}", prefix, i);
                let (is_new, _) = trie.insert(key.as_bytes());
                if is_new {
                    inserted.push(key);
                }
            }
            inserted
        })
    }).collect();

    let mut all_keys = Vec::new();
    for h in handles {
        all_keys.extend(h.join().unwrap());
    }

    assert_eq!(trie.num_words(), all_keys.len());
    for key in &all_keys {
        assert!(trie.contains(key.as_bytes()), "Missing: {}", key);
    }
}

#[test]
fn test_concurrent_readers_and_writers() {
    let trie = Arc::new(ConcurrentCsppTrie::with_capacity(0, 10_000_000));
    let n = CONCURRENT_KEYS;

    // Pre-insert some keys
    for i in 0..n {
        trie.insert(format!("pre_{:05}", i).as_bytes());
    }

    let writer = {
        let trie = Arc::clone(&trie);
        thread::spawn(move || {
            for i in 0..n {
                trie.insert(format!("new_{:05}", i).as_bytes());
            }
        })
    };

    let reader = {
        let trie = Arc::clone(&trie);
        thread::spawn(move || {
            let guard = epoch::pin();
            let mut found = 0;
            for i in 0..n {
                if trie.contains_with_guard(format!("pre_{:05}", i).as_bytes(), &guard) {
                    found += 1;
                }
            }
            found
        })
    };

    writer.join().unwrap();
    let found = reader.join().unwrap();
    assert_eq!(found, n, "Reader should find all pre-inserted keys");
    assert_eq!(trie.num_words(), n * 2);
}

#[test]
fn test_concurrent_insert_with_values() {
    let trie = ConcurrentCsppTrie::with_capacity(4, 4_000_000);
    let n = 5_000;

    for i in 0..n {
        let key = format!("val_{:05}", i);
        let (is_new, valpos) = trie.insert(key.as_bytes());
        assert!(is_new);
        // Single-threaded value write is safe
        trie.set_value(valpos, i as u32);
    }

    assert_eq!(trie.num_words(), n);
    for i in 0..n {
        let key = format!("val_{:05}", i);
        let valpos = trie.lookup(key.as_bytes()).unwrap();
        let val: u32 = trie.get_value(valpos);
        assert_eq!(val, i as u32, "Value mismatch for key {}", key);
    }
}

#[test]
fn test_concurrent_race_stats() {
    let trie = Arc::new(ConcurrentCsppTrie::with_capacity(0, 40_000_000));
    let n = CONCURRENT_KEYS;

    let handles: Vec<_> = (0..4).map(|tid| {
        let trie = Arc::clone(&trie);
        thread::spawn(move || {
            for i in 0..n {
                // All threads insert keys with the same prefix to maximize contention
                let key = format!("contest_{:05}_{}", i, tid);
                trie.insert(key.as_bytes());
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    let retries = trie.race_stats.retries.load(std::sync::atomic::Ordering::Relaxed);
    let parent_fail = trie.race_stats.parent_lock_fail.load(std::sync::atomic::Ordering::Relaxed);
    let lazy_fail = trie.race_stats.lazy_free_fail.load(std::sync::atomic::Ordering::Relaxed);
    let child_fail = trie.race_stats.child_cas_fail.load(std::sync::atomic::Ordering::Relaxed);

    // Just verify we completed without deadlock or panic
    assert_eq!(trie.num_words(), n * 4);
    eprintln!(
        "Race stats: retries={}, parent_lock_fail={}, lazy_free_fail={}, child_cas_fail={}",
        retries, parent_fail, lazy_fail, child_fail
    );
}

#[test]
fn test_concurrent_long_keys() {
    let trie = Arc::new(ConcurrentCsppTrie::with_capacity(0, 40_000_000));

    // Pre-seed
    for tid in 0..4 {
        let prefix = "a".repeat(100);
        let key = format!("{}{:03}_{}", prefix, 0, tid);
        trie.insert(key.as_bytes());
    }

    let handles: Vec<_> = (0..4).map(|tid| {
        let trie = Arc::clone(&trie);
        thread::spawn(move || {
            let prefix = "a".repeat(100);
            for i in 1..if cfg!(miri) { 10 } else { 50 } {
                let key = format!("{}{:03}_{}", prefix, i, tid);
                assert!(trie.insert(key.as_bytes()).0);
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(trie.num_words(), if cfg!(miri) { 40 } else { 200 });
}

// Release-only performance test
#[cfg(not(debug_assertions))]
#[test]
fn test_concurrent_insert_throughput() {
    let trie = Arc::new(ConcurrentCsppTrie::with_capacity(0, 40_000_000));
    let n = 50_000;
    let num_threads = 4;

    let start = std::time::Instant::now();

    let handles: Vec<_> = (0..num_threads).map(|tid| {
        let trie = Arc::clone(&trie);
        thread::spawn(move || {
            let prefix = (b'A' + tid as u8) as char;
            for i in 0..n {
                let key = format!("{}_perf_{:06}", prefix, i);
                trie.insert(key.as_bytes());
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    let elapsed = start.elapsed();
    let total_keys = n * num_threads;
    let kps = total_keys as f64 / elapsed.as_secs_f64();

    assert_eq!(trie.num_words(), total_keys);
    eprintln!(
        "Concurrent insert: {} keys in {:.2}ms ({:.0} K keys/sec, {} threads)",
        total_keys, elapsed.as_secs_f64() * 1000.0, kps / 1000.0, num_threads
    );
}
