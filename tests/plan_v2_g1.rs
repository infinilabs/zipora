//! Plan v2 (reliability-first) regression tests, verifier group G1.

use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};

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
