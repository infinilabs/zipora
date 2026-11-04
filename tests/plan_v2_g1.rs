//! Plan v2 (reliability-first) regression tests, verifier group G1.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};

use zipora::RankSelectInterleaved256;
use zipora::RecordId;
use zipora::blob_store::traits::{BlobStore, CompressedBlobStore};
use zipora::blob_store::{NestLoudsTrieBlobStore, TrieBlobStoreConfig};
use zipora::fsa::cspp_trie::CsppTrie;
use zipora::fsa::cspp_trie_concurrent::ConcurrentCsppTrie;
use zipora::fsa::token::{TokenManager, with_reader_token, with_writer_token};
use zipora::fsa::version_sync::{ConcurrencyLevel, VersionManager};

// ---------------------------------------------------------------------------
// S1: VersionManager / TokenManager lifetime, TLS cache and writer exclusivity
// ---------------------------------------------------------------------------

/// Clears the calling thread's token cache when dropped. Declared after the
/// managers it guards, so it runs first - also while unwinding from a failed
/// assertion - and v5.0.0's cached tokens never outlive their manager here.
struct ClearTokenCacheOnDrop<'a>(&'a TokenManager);

impl Drop for ClearTokenCacheOnDrop<'_> {
    fn drop(&mut self) {
        self.0.clear_thread_cache();
    }
}

/// S1(a): a reader token that outlives its `VersionManager` must not touch
/// the dead manager when dropped. v5.0.0 keeps a raw `*const VersionManager`
/// in `TokenReleaseCallback`, so Miri reports a use-after-free in
/// `TokenReleaseCallback::release`.
#[test]
fn test_v2_s1_reader_token_outlives_manager() {
    let token = {
        let manager = VersionManager::new(ConcurrencyLevel::OneWriteMultiRead);
        manager.acquire_reader_token().unwrap()
    };
    assert!(token.is_valid());
    drop(token);
}

/// S1(b): moving a `VersionManager` while one of its tokens is live must not
/// leave the token pointing at the old location. At v5.0.0 `reserve`
/// reallocates the Vec, the token keeps the freed address, and Miri reports a
/// use-after-free when it is dropped.
#[test]
fn test_v2_s1_reader_token_survives_manager_move() {
    let mut managers = vec![VersionManager::new(ConcurrencyLevel::OneWriteMultiRead)];
    let token = managers[0].acquire_reader_token().unwrap();
    managers.reserve(1024);
    drop(token);
    assert_eq!(
        managers[0].active_readers(),
        0,
        "dropping the token must release it on the moved manager"
    );
}

/// S1: a reader token cached in TLS must not outlive its manager either. At
/// v5.0.0 the thread-exit destructor of `TOKEN_CACHE` drops a cached reader
/// whose manager is already gone; Miri reports a use-after-free.
#[test]
fn test_v2_s1_tls_cached_reader_outlives_manager() {
    std::thread::spawn(|| {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        with_reader_token(&manager, |_| Ok(())).unwrap();
        drop(manager);
        // Thread exit drops TOKEN_CACHE and the reader cached above.
    })
    .join()
    .unwrap();
}

/// S1(c): the TLS token cache is shared by every manager on the thread. At
/// v5.0.0 a writer cached by a OneWriteMultiRead manager is handed out by a
/// NoWriteReadOnly manager, which must never grant a writer.
#[test]
fn test_v2_s1_cached_writer_not_reused_by_readonly_manager() {
    let owmr = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
    let read_only = TokenManager::new(ConcurrencyLevel::NoWriteReadOnly);
    let _clear = ClearTokenCacheOnDrop(&owmr);

    with_writer_token(&owmr, |_| Ok(())).unwrap();
    let granted = with_writer_token(&read_only, |token| Ok(token.concurrency_level()));
    assert!(
        granted.is_err(),
        "NoWriteReadOnly manager granted a writer token cached by another manager: {granted:?}"
    );
}

/// S1(c'): a cached reader must only be reused by the manager that issued
/// it. At v5.0.0 manager `b` hands out `a`'s cached reader, so the token
/// reports `a`'s level and `b` counts no active reader.
#[test]
fn test_v2_s1_cached_reader_not_shared_across_managers() {
    let a = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
    let b = TokenManager::new(ConcurrencyLevel::MultiWriteMultiRead);
    let _clear = ClearTokenCacheOnDrop(&a);

    with_reader_token(&a, |_| Ok(())).unwrap();
    let (level, b_readers) = with_reader_token(&b, |token| {
        Ok((
            token.concurrency_level(),
            b.version_manager().active_readers(),
        ))
    })
    .unwrap();
    assert_eq!(
        level,
        ConcurrencyLevel::MultiWriteMultiRead,
        "manager b handed out manager a's cached reader token"
    );
    assert_eq!(b_readers, 1, "b must account for the reader it handed out");
}

/// S1(d): a writer token returned to the TLS cache keeps the single
/// OneWriteMultiRead writer slot claimed forever. At v5.0.0 a second thread
/// therefore gets `resource_busy` although no writer is in use.
#[test]
fn test_v2_s1_cached_writer_does_not_block_other_threads() {
    let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
    let _clear = ClearTokenCacheOnDrop(&manager);

    with_writer_token(&manager, |_| Ok(())).unwrap();
    let other = std::thread::scope(|s| {
        s.spawn(|| with_writer_token(&manager, |_| Ok(())).map_err(|e| e.to_string()))
            .join()
            .unwrap()
    });
    assert!(
        other.is_ok(),
        "idle writer slot is still claimed by this thread's token cache: {other:?}"
    );
}

/// S1(e): `acquire_writer_token` checks `active_writers == 0` before taking
/// the lock and increments it afterwards, so in OneWriteMultiRead mode several
/// threads can pass the check together and all receive a writer token.
#[test]
fn test_v2_s1_concurrent_owmr_writer_claim_is_exclusive() {
    const THREADS: usize = 8;
    let rounds = if cfg!(miri) { 20 } else { 2_000 };
    let manager = VersionManager::new(ConcurrencyLevel::OneWriteMultiRead);
    let barrier = Barrier::new(THREADS);
    let holders = AtomicUsize::new(0);
    let max_holders = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..THREADS {
            s.spawn(|| {
                barrier.wait();
                for _ in 0..rounds {
                    if let Ok(token) = manager.acquire_writer_token() {
                        let now = holders.fetch_add(1, Ordering::SeqCst) + 1;
                        max_holders.fetch_max(now, Ordering::SeqCst);
                        std::thread::yield_now();
                        holders.fetch_sub(1, Ordering::SeqCst);
                        drop(token);
                    }
                }
            });
        }
    });

    assert_eq!(
        max_holders.load(Ordering::SeqCst),
        1,
        "OneWriteMultiRead handed out concurrent writer tokens"
    );
    assert_eq!(manager.active_writers(), 0);
}

// ---------------------------------------------------------------------------
// S7: CsppTrie / ConcurrentCsppTrie value accessors
// ---------------------------------------------------------------------------

const S7_NEIGHBOUR: u32 = 0xB0B0_B0B0;

/// S7(a): `CsppTrie::set_value::<T>` does not check `size_of::<T>()` against
/// `valsize`. At v5.0.0 an 8-byte write into a 4-byte slot silently
/// overwrites the next key's value.
#[test]
fn test_v2_s7_cspp_oversized_set_value_must_not_corrupt_neighbour() {
    let mut trie = CsppTrie::new(4);
    let (_, vp_a) = trie.insert(b"alpha");
    let (_, vp_b) = trie.insert(b"bravo");
    trie.set_value::<u32>(vp_b, S7_NEIGHBOUR);

    let write = catch_unwind(AssertUnwindSafe(|| trie.set_value::<u64>(vp_a, u64::MAX)));
    assert_eq!(
        trie.get_value::<u32>(vp_b),
        S7_NEIGHBOUR,
        "8-byte set_value on a 4-byte slot overwrote the neighbouring key's value"
    );
    assert!(
        write.is_err(),
        "set_value must reject size_of::<T>() > valsize"
    );
}

/// S7(a): the read side - at v5.0.0 an 8-byte `get_value` on a 4-byte slot
/// returns the neighbouring key's bytes instead of panicking.
#[test]
fn test_v2_s7_cspp_oversized_get_value_must_not_read_neighbour() {
    let mut trie = CsppTrie::new(4);
    let (_, vp_a) = trie.insert(b"alpha");
    let (_, vp_b) = trie.insert(b"bravo");
    trie.set_value::<u32>(vp_b, S7_NEIGHBOUR);

    let read = catch_unwind(AssertUnwindSafe(|| trie.get_value::<u64>(vp_a)));
    assert!(
        read.is_err(),
        "get_value must reject size_of::<T>() > valsize, got {:x?}",
        read
    );
}

/// S7(a): `valpos` is not checked against value-slot boundaries either. At
/// v5.0.0 a 4-byte write starting two bytes into a 4-byte slot spills into
/// the next key's value.
#[test]
fn test_v2_s7_cspp_mid_slot_set_value_must_not_straddle_neighbour() {
    let mut trie = CsppTrie::new(4);
    let (_, vp_a) = trie.insert(b"alpha");
    let (_, vp_b) = trie.insert(b"bravo");
    trie.set_value::<u32>(vp_b, S7_NEIGHBOUR);

    let write = catch_unwind(AssertUnwindSafe(|| {
        trie.set_value::<u32>(vp_a + 2, u32::MAX)
    }));
    assert_eq!(
        trie.get_value::<u32>(vp_b),
        S7_NEIGHBOUR,
        "set_value at a mid-slot offset overwrote the neighbouring key's value"
    );
    assert!(
        write.is_err(),
        "set_value must reject a valpos that is not a value-slot offset"
    );
}

/// S7(a), concurrent trie: values live in the shared node pool, so at v5.0.0
/// an 8-byte write into a 4-byte value slot overwrites whatever slot follows
/// it - here the next key's trie node or value.
#[test]
fn test_v2_s7_concurrent_oversized_set_value_must_not_corrupt_neighbour() {
    let trie = ConcurrentCsppTrie::with_capacity(4, 4096);
    let (_, vp_a) = trie.insert(b"alpha");
    let (_, vp_b) = trie.insert(b"bravo");
    trie.set_value::<u32>(vp_b, S7_NEIGHBOUR);

    let write = catch_unwind(AssertUnwindSafe(|| trie.set_value::<u64>(vp_a, 0)));
    let found = catch_unwind(AssertUnwindSafe(|| trie.lookup(b"bravo")));
    assert!(
        matches!(found, Ok(Some(vp)) if vp == vp_b),
        "8-byte set_value on a 4-byte slot corrupted the neighbouring key's node: lookup(bravo) = {found:?}, expected Ok(Some({vp_b}))"
    );
    assert_eq!(
        trie.get_value::<u32>(vp_b),
        S7_NEIGHBOUR,
        "8-byte set_value on a 4-byte slot overwrote the neighbouring key's value"
    );
    assert!(
        write.is_err(),
        "set_value must reject size_of::<T>() > valsize"
    );
}

/// S7(a), concurrent trie, read side: at v5.0.0 an 8-byte `get_value` on a
/// 4-byte slot returns the following pool word instead of panicking.
#[test]
fn test_v2_s7_concurrent_oversized_get_value_must_not_read_neighbour() {
    let trie = ConcurrentCsppTrie::with_capacity(4, 4096);
    let (_, vp_a) = trie.insert(b"alpha");
    trie.insert(b"bravo");

    let read = catch_unwind(AssertUnwindSafe(|| trie.get_value::<u64>(vp_a)));
    assert!(
        read.is_err(),
        "get_value must reject size_of::<T>() > valsize, got {:x?}",
        read
    );
}

/// S7(a), concurrent trie, bounds check: `get_value`/`set_value` compare
/// `valpos + size` against the pool size with a plain add. `[profile.release]`
/// sets `overflow-checks = false`, so a `valpos` near `usize::MAX` wraps past
/// the check there (and panics with "attempt to add with overflow" in debug).
/// Both must report "valpos out of bounds", like `CsppTrie::get_value`.
#[test]
fn test_v2_s7_concurrent_valpos_overflow_is_reported_as_out_of_bounds() {
    let trie = ConcurrentCsppTrie::with_capacity(4, 4096);
    trie.insert(b"alpha");
    let huge = usize::MAX - 3; // 4-byte aligned; `huge + 4` overflows
    let get = catch_unwind(AssertUnwindSafe(|| {
        let _ = trie.get_value::<u32>(huge);
    }));
    let set = catch_unwind(AssertUnwindSafe(|| trie.set_value::<u32>(huge, 1)));
    for (op, outcome) in [("get_value", get), ("set_value", set)] {
        let payload = match outcome {
            Ok(()) => panic!("{op}: valpos {huge:#x} was accepted"),
            Err(payload) => payload,
        };
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert!(
            msg.contains("valpos out of bounds"),
            "{op}: expected \"valpos out of bounds\", got {msg:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// C3: NestLoudsTrieBlobStore node-id bookkeeping and len()
// ---------------------------------------------------------------------------

type TrieStore = NestLoudsTrieBlobStore<RankSelectInterleaved256>;

const C3_KEYS: usize = 500;

fn c3_key(i: usize) -> Vec<u8> {
    format!("prefix/{:04}/leaf_{}", i % 17, i).into_bytes()
}

fn c3_value(i: usize) -> Vec<u8> {
    format!("value_{i}").into_bytes()
}

/// `memory_optimized()` uses the default `ZiporaTrieConfig` (DoubleArray),
/// whose collision handling relocates states. Statistics are already off
/// there; the key cache is disabled so every lookup goes through the trie.
fn c3_double_array_store() -> (TrieStore, Vec<RecordId>) {
    let mut cfg = TrieBlobStoreConfig::memory_optimized();
    cfg.enable_statistics = false;
    cfg.key_cache_size = 0;
    let mut store = TrieStore::new(cfg).unwrap();
    let ids = (0..C3_KEYS)
        .map(|i| store.put_with_key(&c3_key(i), &c3_value(i)).unwrap())
        .collect();
    (store, ids)
}

/// C3: `put_with_key` maps the key's DoubleArray state id to its blob, but
/// later inserts relocate states. At v5.0.0 about 10% of the keys become
/// unreachable (or resolve to another key's blob) through `get_by_key`.
#[test]
fn test_v2_c3_double_array_relocation_keeps_every_key_reachable() {
    let (mut store, _) = c3_double_array_store();
    let mut missing = Vec::new();
    let mut wrong = Vec::new();
    for i in 0..C3_KEYS {
        match store.get_by_key(&c3_key(i)) {
            Ok(v) if v == c3_value(i) => {}
            Ok(_) => wrong.push(i),
            Err(_) => missing.push(i),
        }
    }
    assert!(
        missing.is_empty() && wrong.is_empty(),
        "after DoubleArray relocations {} of {C3_KEYS} keys are unreachable and {} return another key's blob; first unreachable: {:?}, first wrong: {:?}",
        missing.len(),
        wrong.len(),
        &missing[..missing.len().min(5)],
        &wrong[..wrong.len().min(5)]
    );
}

/// C3: `BlobStore::len` returns `stats.blob_stats.blob_count`, which is only
/// maintained when `enable_statistics` is set; at v5.0.0 it reports 0.
#[test]
fn test_v2_c3_len_counts_records_without_statistics() {
    let (store, ids) = c3_double_array_store();
    assert_eq!(ids.len(), C3_KEYS);
    assert_eq!(
        BlobStore::len(&store),
        C3_KEYS,
        "len() must not depend on enable_statistics"
    );
}

/// C3: `remove(id)` rebuilds the key from the record's stored state id. After
/// DoubleArray relocations that id is stale, so at v5.0.0 removals fail or
/// delete a different key.
#[test]
fn test_v2_c3_remove_by_id_after_relocation_removes_only_that_key() {
    let (mut store, ids) = c3_double_array_store();
    let mut errors = Vec::new();
    for i in (0..C3_KEYS).step_by(5) {
        if let Err(e) = store.remove(ids[i]) {
            errors.push((i, e.to_string()));
        }
    }
    let still_present: Vec<usize> = (0..C3_KEYS)
        .step_by(5)
        .filter(|&i| store.contains_key(&c3_key(i)))
        .collect();
    let wrongly_removed: Vec<usize> = (0..C3_KEYS)
        .filter(|i| i % 5 != 0)
        .filter(|&i| !store.contains_key(&c3_key(i)))
        .collect();
    assert!(
        errors.is_empty() && still_present.is_empty() && wrongly_removed.is_empty(),
        "remove(id) used stale state ids: {} errors (first {:?}), {} removed keys still present (first {:?}), {} other keys deleted (first {:?})",
        errors.len(),
        errors.first(),
        still_present.len(),
        &still_present[..still_present.len().min(5)],
        wrongly_removed.len(),
        &wrongly_removed[..wrongly_removed.len().min(5)]
    );
}

/// C3: with the default (Patricia) trie, removing a key recycles its nodes,
/// but an older record of the same key keeps the old node id. At v5.0.0 a
/// later `remove` of that record rebuilds and deletes whichever key reused
/// the node.
#[test]
fn test_v2_c3_remove_stale_duplicate_record_keeps_node_reusing_key() {
    let cfg = TrieBlobStoreConfig::default().key_cache_size(0);
    let mut store = TrieStore::new(cfg).unwrap();
    let older = store.put_with_key(b"dup", b"v1").unwrap();
    let latest = store.put_with_key(b"dup", b"v2").unwrap();
    store.remove(latest).unwrap();
    assert!(!store.contains_key(b"dup"));

    store.put_with_key(b"zed", b"v3").unwrap(); // reuses the recycled nodes
    let _ = store.remove(older);
    assert!(
        store.contains_key(b"zed"),
        "removing a stale record of `dup` deleted the unrelated key `zed`"
    );
    assert_eq!(store.get_by_key(b"zed").unwrap(), b"v3");
}

/// C3: `compressed_size(id)` resolves the record through its stored state id
/// and the key's latest blob. At v5.0.0 an older duplicate reports the newest
/// blob's size, and DoubleArray relocations make some records unresolvable.
#[test]
fn test_v2_c3_compressed_size_reports_the_records_own_blob() {
    let cfg = TrieBlobStoreConfig::default().key_cache_size(0);
    let mut store = TrieStore::new(cfg).unwrap();
    let short = store.put_with_key(b"k", b"a").unwrap();
    store.put_with_key(b"k", b"bbbb").unwrap();
    assert_eq!(
        store.compressed_size(short).unwrap(),
        Some(1),
        "compressed_size of an older duplicate must describe its own blob"
    );

    let (store, ids) = c3_double_array_store();
    let wrong: Vec<usize> = (0..C3_KEYS)
        .filter(|&i| store.compressed_size(ids[i]).ok().flatten() != Some(c3_value(i).len()))
        .collect();
    assert!(
        wrong.is_empty(),
        "compressed_size is wrong or fails for {} of {C3_KEYS} DoubleArray records, first {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(5)]
    );
}
