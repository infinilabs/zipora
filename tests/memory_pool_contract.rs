//! What `MemoryPool` actually promises, and what it does not.
//!
//! This file replaces `tests/security_memory_pool.rs` and
//! `tests/security_memory_pool_simple.rs` (plan.md row D8). Between them those
//! two held twelve tests, near-duplicates of each other, and **not one of them
//! asserted the thing it was named after**. They printed
//! `"VULNERABILITY CONFIRMED: Use-after-free detected!"`,
//! `"CRITICAL: Same memory allocated multiple times!"` and
//! `"BUG: Pool exceeded configured capacity!"` to stdout — which no one reads
//! on a green run — and then returned `Ok`. A suite that reports success
//! whether or not the vulnerability is present is worse than no suite: it
//! occupies the slot where a real test would go.
//!
//! Two of them were additionally `#[ignore]`d as "UB by design". The plan's D8
//! row called for moving those to a Miri or AddressSanitizer negative job.
//! **That is not achievable and they are deleted instead.** Both demonstrate
//! use *after return to the pool*, and a pool by definition does not return
//! recycled memory to the system: the chunk stays live in `free_chunks`, there
//! is no `free()` for AddressSanitizer to flag and no dead allocation for Miri
//! to flag. A sanitizer job built on them would pass unconditionally — the
//! same defect as the tests it replaced, one directory further away. Catching
//! use-after-recycle would need the pool itself to call
//! `__asan_poison_memory_region` on every free, which is not worth adding to a
//! type whose own documentation recommends using a different allocator.
//!
//! Use-after-recycle is a *caller* obligation, and since C3.12 it is stated as
//! one, in the `# Safety` section of `MemoryPool::deallocate`. No test can
//! check a caller's obligation; that is what the `unsafe` keyword is for.
//!
//! What remains below are the properties the pool really does guarantee, each
//! asserted.

use std::sync::Arc;
use std::thread;
use zipora::memory::{MemoryPool, PoolConfig};

/// C3.12. Freeing the same chunk twice used to park one address on the free
/// list twice, and the next two allocations both got it back — two live
/// callers owning the same 64 bytes, reachable without the caller writing a
/// single `unsafe` block. Once both were freed the pool passed that address to
/// the global allocator twice, and glibc aborted the process.
#[test]
fn test_double_free_is_refused() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 10, 8)).unwrap());

    let ptr = pool.allocate().unwrap();

    // SAFETY: `ptr` came from this pool's `allocate` and is live.
    unsafe { pool.deallocate(ptr) }.expect("the first free must succeed");

    // SAFETY (test only): this deliberately breaks the contract in order to
    // check that the pool catches it rather than corrupting its free list.
    let second = unsafe { pool.deallocate(ptr) };
    assert!(
        second.is_err(),
        "the second free of {ptr:?} was accepted, so the address is parked twice"
    );

    let first = pool.allocate().unwrap();
    let other = pool.allocate().unwrap();
    assert_ne!(
        first.as_ptr(),
        other.as_ptr(),
        "the pool handed the same address to two live callers"
    );

    // SAFETY: both came from this pool and are live and distinct.
    unsafe {
        pool.deallocate(first).unwrap();
        pool.deallocate(other).unwrap();
    }
}

/// The allocation counters are exact under contention, and the free list never
/// grows past `max_chunks`.
///
/// This is the property the three "lost updates" tests were gesturing at. It
/// only became true with C3.12: `deallocate` used to take the free-list lock
/// with `try_lock` and, on failure, hand the chunk straight to the global
/// allocator, so under contention deallocations really were lost from the
/// pool's point of view. It now blocks on the lock.
#[test]
fn test_counters_are_exact_under_contention() {
    const THREADS: u64 = 10;
    const CYCLES: u64 = 100;
    const MAX_CHUNKS: usize = 100;

    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, MAX_CHUNKS, 8)).unwrap());

    let handles: Vec<_> = (0..THREADS)
        .map(|thread_id| {
            let pool = Arc::clone(&pool);
            thread::spawn(move || {
                for _ in 0..CYCLES {
                    let ptr = pool.allocate().unwrap();
                    // SAFETY: the pool just handed this chunk over, it is
                    // `chunk_size` = 64 bytes wide, and nothing else holds it.
                    unsafe { ptr.as_ptr().write(thread_id as u8) };
                    thread::yield_now();
                    // SAFETY: `ptr` came from this pool and is freed once.
                    unsafe { pool.deallocate(ptr) }.unwrap();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let stats = pool.stats();
    assert_eq!(stats.alloc_count, THREADS * CYCLES);
    assert_eq!(stats.dealloc_count, THREADS * CYCLES);
    assert_eq!(
        stats.pool_hits + stats.pool_misses,
        stats.alloc_count,
        "every allocation is either a hit or a miss, never both and never neither"
    );
    assert!(
        stats.chunks <= MAX_CHUNKS,
        "the free list holds {} chunks, above the configured maximum of {MAX_CHUNKS}",
        stats.chunks
    );
}

/// Concurrent allocations are distinct addresses.
///
/// The test this replaces was called `demonstrate_toctou_vulnerability` and
/// treated "both threads got an allocation from a single-chunk pool" as a bug.
/// It is not: `max_chunks` bounds the *cache* of recycled chunks, not how many
/// chunks may be live, and `allocate` falls back to the global allocator when
/// the cache is empty. The property that does matter is that no two live
/// allocations share an address.
#[test]
fn test_concurrent_allocations_are_distinct() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 16;

    // A cache of one, so almost every allocation takes the fallback path.
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 1, 8)).unwrap());

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let pool = Arc::clone(&pool);
            // Hold everything at once: addresses may only be reused after a
            // free. `NonNull` is not `Send`, so the addresses travel back as
            // `usize` and are rebuilt below.
            thread::spawn(move || {
                (0..PER_THREAD)
                    .map(|_| pool.allocate().unwrap().as_ptr() as usize)
                    .collect::<Vec<usize>>()
            })
        })
        .collect();

    let mut addresses: Vec<usize> = handles
        .into_iter()
        .flat_map(|handle| handle.join().unwrap())
        .collect();
    addresses.sort_unstable();
    let unique = {
        let mut deduped = addresses.clone();
        deduped.dedup();
        deduped.len()
    };
    assert_eq!(
        unique,
        addresses.len(),
        "{} of {} live allocations shared an address",
        addresses.len() - unique,
        addresses.len()
    );

    for address in addresses {
        let ptr = std::ptr::NonNull::new(address as *mut u8).unwrap();
        // SAFETY: each address came from this pool's `allocate`, is live, and
        // is freed once -- the assertion above proves there are no duplicates.
        unsafe { pool.deallocate(ptr) }.unwrap();
    }
}

/// The free list caches at most `max_chunks`; the surplus goes back to the
/// global allocator rather than growing the pool without bound.
#[test]
fn test_pool_caches_at_most_max_chunks() {
    const MAX_CHUNKS: usize = 2;
    let pool = MemoryPool::new(PoolConfig::new(64, MAX_CHUNKS, 8)).unwrap();

    let chunks = [
        pool.allocate().unwrap(),
        pool.allocate().unwrap(),
        pool.allocate().unwrap(),
    ];

    for chunk in chunks {
        // SAFETY: each came from this pool, is live, and is freed once.
        unsafe { pool.deallocate(chunk) }.unwrap();
    }

    assert_eq!(
        pool.stats().chunks,
        MAX_CHUNKS,
        "the third chunk should have gone back to the global allocator"
    );
}

/// `clear` empties the free list and leaves chunks that are still out on loan
/// alone.
#[test]
fn test_clear_releases_only_pooled_chunks() {
    let pool = MemoryPool::new(PoolConfig::new(64, 10, 8)).unwrap();

    let mut chunks = Vec::new();
    for canary in 0..5u8 {
        let ptr = pool.allocate().unwrap();
        // SAFETY: the pool just handed this chunk over and it is 64 bytes wide.
        unsafe { ptr.as_ptr().write(canary) };
        chunks.push(ptr);
    }

    let still_held = chunks.split_off(3);
    for chunk in chunks {
        // SAFETY: each came from this pool, is live, and is freed once.
        unsafe { pool.deallocate(chunk) }.unwrap();
    }
    assert_eq!(pool.stats().chunks, 3);

    pool.clear().unwrap();
    assert_eq!(pool.stats().chunks, 0);

    for (offset, chunk) in still_held.iter().enumerate() {
        // SAFETY: these three were never returned to the pool, so `clear`
        // cannot have touched them; they are live and 64 bytes wide.
        let canary = unsafe { chunk.as_ptr().read() };
        assert_eq!(
            canary,
            (offset + 3) as u8,
            "clear() disturbed a chunk that was still on loan"
        );
    }

    for chunk in still_held {
        // SAFETY: as above -- live, from this pool, freed once.
        unsafe { pool.deallocate(chunk) }.unwrap();
    }
}

/// `MemoryPool` is `Send` and `Sync`, and that is a deliberate, documented
/// choice rather than an oversight.
///
/// The test this replaces printed *"WARNING: MemoryPool implements Send/Sync
/// despite containing raw pointers! This violates Rust's memory safety
/// guarantees."* It does not. The raw pointers live in the free list behind a
/// `Mutex` and are never dereferenced by the pool; what a caller does with a
/// chunk after receiving it is governed by the `# Safety` contract on
/// `deallocate`, which is why that method is `unsafe`.
#[test]
fn test_memory_pool_is_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<MemoryPool>();
    assert_sync::<MemoryPool>();
}
