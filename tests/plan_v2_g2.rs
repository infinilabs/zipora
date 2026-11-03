//! Reliability-first plan (v2), group G2: regression tests that use only the
//! public API.
//!
//! Every test compiles against v5.0.0 and fails there for the reason given in
//! its doc comment; line numbers refer to v5.0.0. A test marked "needs Miri"
//! fails at v5.0.0 only under
//! `MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly miri test --test plan_v2_g2`;
//! natively it is either skipped (its v5.0.0 failure mode is a process abort or
//! silent heap corruption) or a probabilistic stress test.
//!
//! The one G2 test that needs a private API (`FixedCapacityMemoryPool::deallocate`)
//! lives in `src/memory/fixed_capacity_pool.rs`, module `plan_v2_fcp`.

use std::sync::atomic::Ordering;

use zipora::memory::{ThreadLocalMemoryPool, ThreadLocalPoolConfig};

// ============================================================================
// S2: ThreadLocalMemoryPool (src/memory/threadlocal_pool.rs)
// ============================================================================

/// Runs `f` on a fresh thread, so the pool's thread-local cache starts empty
/// no matter which test ran earlier on the harness thread, and re-raises a
/// panic from `f` with its original payload.
fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    if let Err(payload) = std::thread::spawn(f).join() {
        std::panic::resume_unwind(payload);
    }
}

/// A pool backed by the system allocator (no `SecureMemoryPool`), with stats.
fn tlp_config(arena_size: usize) -> ThreadLocalPoolConfig {
    ThreadLocalPoolConfig {
        arena_size,
        use_secure_memory: false,
        ..ThreadLocalPoolConfig::default()
    }
}

fn ranges_overlap(a: (*mut u8, usize), b: (*mut u8, usize)) -> bool {
    let (a0, a1) = (a.0 as usize, a.0 as usize + a.1);
    let (b0, b1) = (b.0 as usize, b.0 as usize + b.1);
    a0 < b1 && b0 < a1
}

/// S2(a). When the hot area is exhausted, `allocate_new_area_or_fallback`
/// replaces it (`self.hot_area = Some(hot_area)`, threadlocal_pool.rs:292) and
/// `HotArea::drop` (197-204) frees the old slab while every allocation carved
/// from it is still live. The freed slabs are then reused by the allocator
/// (glibc metadata, later slabs), so live allocations are overwritten.
/// Miri: use-after-free on the first read of allocation 0.
#[test]
fn test_v2_s2_hot_area_rollover_keeps_live_allocations_intact() {
    on_fresh_thread(|| {
        let pool = ThreadLocalMemoryPool::new(tlp_config(4096)).unwrap();
        // 64 x 256 B = four 4 KiB slabs, so three rollovers.
        let mut live = Vec::with_capacity(64);
        for i in 0..64u8 {
            let mut a = pool.allocate(256).unwrap();
            a.as_mut_slice().fill(i);
            live.push(a);
        }
        for (i, a) in live.iter().enumerate() {
            assert!(
                a.as_slice().iter().all(|&b| b == i as u8),
                "allocation {i} was overwritten after its hot-area slab was replaced"
            );
        }
        for (i, a) in live.iter().enumerate() {
            for (j, b) in live.iter().enumerate().skip(i + 1) {
                assert!(
                    !ranges_overlap((a.as_ptr(), a.size()), (b.as_ptr(), b.size())),
                    "live allocations {i} and {j} overlap"
                );
            }
        }
    });
}

/// S2(b). A request above `arena_size / 4` goes to `allocate_from_global`
/// (283-284), which calls `ThreadLocalMemoryPool::allocate` on the same pool
/// (318-322) while `CURRENT_CACHE` is still mutably borrowed (407): the thread
/// panics with "already borrowed: BorrowMutError". (Without the panic the
/// pointer would come from a temporary allocation dropped inside `and_then`.)
#[test]
fn test_v2_s2_oversize_allocation_succeeds() {
    on_fresh_thread(|| {
        // high_performance(): arena_size 8 MiB, so anything above 2 MiB is oversize.
        let pool = ThreadLocalMemoryPool::new(ThreadLocalPoolConfig::high_performance()).unwrap();
        const SIZE: usize = 3 * 1024 * 1024;
        let mut big = pool
            .allocate(SIZE)
            .unwrap_or_else(|e| panic!("3 MiB allocation failed: {e}"));
        assert_eq!(big.size(), SIZE);
        let bytes = big.as_mut_slice();
        bytes[0] = 1;
        bytes[SIZE - 1] = 2;
        assert_eq!((bytes[0], bytes[SIZE - 1]), (1, 2));
        // The pool stays usable afterwards.
        let small = pool.allocate(64).unwrap();
        drop(small);
        drop(big);
    });
}

/// S2(c), needs Miri. `clear_caches` (476-481) drops the thread cache and with
/// it the slab under every live allocation. Dropping a survivor then finds no
/// cache and `deallocate_bypass_cache` (443-459) calls `dealloc` on a pointer
/// into the already-freed slab with a made-up `Layout(size, 8)`. Miri reports
/// the `dealloc` at threadlocal_pool.rs:455; natively glibc aborts the process
/// (`free(): invalid pointer` or similar), so the test is skipped there.
#[test]
fn test_v2_s2_drop_after_clear_caches_is_sound() {
    on_fresh_thread(|| {
        let pool = ThreadLocalMemoryPool::new(tlp_config(4096)).unwrap();
        let _head = pool.allocate(64).unwrap(); // offset 0 of the slab
        let survivor = pool.allocate(64).unwrap(); // offset 64: an interior pointer
        pool.clear_caches();
        drop(survivor);
    });
}

/// S2(c'). Same root cause as (c), observable natively: after `clear_caches`
/// freed the slab under `survivor`, the next slab is carved from the same
/// address, so a fresh allocation aliases the live one.
/// Miri: use-after-free reading `survivor` (if Miri did not reuse the address).
#[test]
fn test_v2_s2_clear_caches_does_not_recycle_live_memory() {
    on_fresh_thread(|| {
        let pool = ThreadLocalMemoryPool::new(tlp_config(4096)).unwrap();
        let _head = pool.allocate(64).unwrap();
        let mut survivor = pool.allocate(64).unwrap();
        survivor.as_mut_slice().fill(0xA5);
        pool.clear_caches();
        let fresh: Vec<_> = (0..4)
            .map(|_| {
                let mut a = pool.allocate(64).unwrap();
                a.as_mut_slice().fill(0x5A);
                a
            })
            .collect();
        for a in &fresh {
            assert!(
                !ranges_overlap((a.as_ptr(), a.size()), (survivor.as_ptr(), survivor.size())),
                "allocation at {:p} aliases the live allocation at {:p} after clear_caches",
                a.as_ptr(),
                survivor.as_ptr()
            );
        }
        assert!(
            survivor.as_slice().iter().all(|&b| b == 0xA5),
            "live allocation was overwritten after clear_caches"
        );
    });
}

/// S2(d). `CURRENT_CACHE` (372-374) is one untagged thread-local shared by
/// every pool. After pool A is gone, pool B on the same thread inherits A's
/// cache: B's oversize request resolves A's dead `Weak` and fails with
/// "Global pool unavailable" (318, 324), and B's small request is served from
/// A's slab and counted in A's statistics, not B's.
#[test]
fn test_v2_s2_second_pool_on_same_thread_is_independent() {
    on_fresh_thread(|| {
        let a = ThreadLocalMemoryPool::new(tlp_config(4096)).unwrap();
        drop(a.allocate(64).unwrap());
        drop(a);

        // Default arena 2 MiB: 1 MiB is above arena_size / 4.
        let b = ThreadLocalMemoryPool::new(tlp_config(2 * 1024 * 1024)).unwrap();
        let big = b.allocate(1024 * 1024);
        let small = b.allocate(64);
        let b_stats = b.stats().expect("stats are enabled");
        let b_served = b_stats.hot_allocations.load(Ordering::Relaxed)
            + b_stats.cache_hits.load(Ordering::Relaxed);
        assert!(
            big.is_ok() && small.is_ok() && b_served >= 1,
            "pool B on a thread that used pool A: 1 MiB -> {:?}, 64 B -> {:?}, \
             allocations served by B's own cache = {b_served}",
            big.as_ref().err().map(|e| e.to_string()),
            small.as_ref().err().map(|e| e.to_string()),
        );
    });
}

/// S2(e), found while writing S2. The hot area carves `round_up(size, 8)`
/// bytes (184), but `deallocate` files a freed block under the size *class* of
/// the request (255-260, 354-358). A 17-byte request carves 24 bytes, is filed
/// under the 32-byte class when freed, and the next 32-byte request gets it:
/// its last 8 bytes are the live neighbour's.
#[test]
fn test_v2_s2_recycled_block_is_as_large_as_its_size_class() {
    on_fresh_thread(|| {
        let pool = ThreadLocalMemoryPool::new(tlp_config(4096)).unwrap();
        let first = pool.allocate(17).unwrap(); // carves 24 bytes at offset 0
        let mut neighbour = pool.allocate(8).unwrap(); // carves 8 bytes at offset 24
        neighbour.as_mut_slice().fill(0xEE);
        drop(first); // filed under the 32-byte class
        let mut reused = pool.allocate(32).unwrap();
        reused.as_mut_slice().fill(0x11);
        assert!(
            !ranges_overlap(
                (reused.as_ptr(), reused.size()),
                (neighbour.as_ptr(), neighbour.size())
            ),
            "32-byte allocation at {:p} overlaps the live 8-byte allocation at {:p}",
            reused.as_ptr(),
            neighbour.as_ptr()
        );
        assert!(neighbour.as_slice().iter().all(|&b| b == 0xEE));
    });
}
