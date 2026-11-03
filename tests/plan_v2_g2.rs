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

use std::cell::{Cell, RefCell};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Barrier;
use std::sync::atomic::Ordering;

use zipora::containers::{GoldHashIdx, MinimalSso};
use zipora::memory::{
    FixedCapacityAllocation, FixedCapacityMemoryPool, FixedCapacityPoolConfig, SecureMemoryPool,
    SecurePoolConfig, ThreadLocalMemoryPool, ThreadLocalPoolConfig,
};
use zipora::{AutoGrowCircularQueue, SmallMap};

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

// ============================================================================
// S3: FixedCapacityMemoryPool (src/memory/fixed_capacity_pool.rs)
// ============================================================================

fn fcp_config(max_block_size: usize, total_blocks: usize) -> FixedCapacityPoolConfig {
    FixedCapacityPoolConfig {
        max_block_size,
        total_blocks,
        ..FixedCapacityPoolConfig::default()
    }
}

/// S3(a). With `eager_allocation: false`, `ensure_memory_allocated` reads
/// `memory` without synchronization (fixed_capacity_pool.rs:503-507) while the
/// initializing thread publishes it under the mutex (489-492) *before* it
/// builds the free lists (494-495, 579-600; the head is stored `Relaxed` at
/// 597). A thread that sees `Some` early finds every list empty and gets
/// "Memory allocation failed" from a pool with 20 000 free blocks. Natively
/// this is timing-dependent, so it is tried 32 times; nothing is freed or
/// written, so v5.0.0 cannot crash here. Miri: "Data race detected" on the
/// first trial.
#[test]
fn test_v2_s3_lazy_init_first_allocations_never_fail() {
    let (trials, blocks, threads) = if cfg!(miri) {
        (1, 16, 3)
    } else {
        (32, 20_000, 8)
    };
    for trial in 0..trials {
        let pool = FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
            max_block_size: 64,
            total_blocks: blocks,
            alignment: 8,
            enable_stats: false,
            eager_allocation: false,
            secure_clear: false,
        })
        .unwrap();
        let start = Barrier::new(threads);
        let done = Barrier::new(threads);
        let failures: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (pool, start, done) = (&pool, &start, &done);
                    s.spawn(move || {
                        start.wait();
                        let result = pool.allocate(64);
                        // Hold the block until every thread has allocated: no
                        // frees, so the free list never sees reuse.
                        done.wait();
                        result.err().map(|e| format!("thread {t}: {e}"))
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| h.join().unwrap())
                .collect()
        });
        assert!(
            failures.is_empty(),
            "trial {trial}: first allocations from a lazily initialized pool with {blocks} \
             free blocks failed: {failures:?}"
        );
    }
}

/// S3(a'), needs Miri. Concurrent allocate/write/free on a lazily initialized
/// pool. Besides the initialization race above, a pop reads the inline
/// `BlockHeader` through `&` (633-640) while the thread that won the block may
/// already be writing user data into it, and the untagged `u32` head (156)
/// admits ABA. Natively v5.0.0 eventually installs a garbage head and dies
/// with "misaligned pointer dereference" at 633 (a non-unwinding panic that
/// aborts the test binary), hence skipped there; Miri reports the data race.
#[test]
fn test_v2_s3_lazy_init_concurrent_alloc_free_is_race_free() {
    let (blocks, threads, rounds) = if cfg!(miri) {
        (16, 3, 4)
    } else {
        (20_000, 8, 2_000)
    };
    let pool = FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
        max_block_size: 64,
        total_blocks: blocks,
        alignment: 8,
        enable_stats: true,
        eager_allocation: false,
        secure_clear: false,
    })
    .unwrap();
    let barrier = Barrier::new(threads);
    std::thread::scope(|s| {
        for t in 0..threads {
            let (pool, barrier) = (&pool, &barrier);
            s.spawn(move || {
                barrier.wait();
                for r in 0..rounds {
                    let mut a = pool
                        .allocate(64)
                        .unwrap_or_else(|e| panic!("thread {t}, round {r}: {e}"));
                    let tag = (t * 31 + r) as u8;
                    a.as_mut_slice().fill(tag);
                    assert!(
                        a.as_slice().iter().all(|&b| b == tag),
                        "thread {t}, round {r}: block handed to two threads"
                    );
                }
            });
        }
    });
}

/// S3(c), needs Miri. `FixedCapacityAllocation` keeps
/// `pool: *const FixedCapacityMemoryPool` (805, 820) and dereferences it in
/// `Drop` (851-859); nothing ties the allocation to the pool's lifetime, so
/// this *safe* code writes into the freed arena and then calls `deallocate`
/// through a dangling pointer. Miri: use-after-free on the first write.
#[test]
fn test_v2_s3_allocation_may_outlive_its_pool() {
    let mut alloc = {
        let pool = FixedCapacityMemoryPool::new(fcp_config(64, 4)).unwrap();
        pool.allocate(64).unwrap()
    };
    alloc.as_mut_slice().fill(0x5A);
    assert!(alloc.as_slice().iter().all(|&b| b == 0x5A));
    drop(alloc);
}

/// S3(c'), needs Miri. The same raw back-pointer breaks when the pool merely
/// *moves*: `make` boxes the pool, so the allocation's `pool` pointer refers
/// to `make`'s dead stack slot and `Drop` reads it. Miri: use-after-free of
/// the stack allocation.
#[test]
fn test_v2_s3_allocation_survives_pool_move() {
    fn make() -> (Box<FixedCapacityMemoryPool>, FixedCapacityAllocation) {
        let pool = FixedCapacityMemoryPool::new(fcp_config(64, 4)).unwrap();
        let alloc = pool.allocate(64).unwrap();
        (Box::new(pool), alloc)
    }
    let (pool, alloc) = make();
    drop(alloc);
    let stats = pool.stats().expect("stats are enabled");
    assert_eq!(stats.active_blocks.load(Ordering::Relaxed), 0);
}

/// S3(d). Free-list links and heads are `u32` byte offsets (`as u32` at 549,
/// 590 and 779; `LIST_TAIL = u32::MAX`) and the block count is stored
/// `as u32` (559, 600), yet `validate_config` (240-283) only rejects `usize`
/// overflow, so this config is accepted and its free list would silently
/// wrap. `eager_allocation: false` keeps v5.0.0 from touching memory.
/// (`u32::MAX` blocks are unrepresentable with byte offsets *and* with `u32`
/// block indices, so this holds whichever encoding the fix picks.)
#[test]
fn test_v2_s3_config_beyond_u32_offsets_is_rejected() {
    let config = FixedCapacityPoolConfig {
        max_block_size: 16,
        total_blocks: u32::MAX as usize, // 64 GiB of 16-byte blocks
        alignment: 8,
        enable_stats: false,
        eager_allocation: false,
        secure_clear: false,
    };
    assert!(
        FixedCapacityMemoryPool::new(config).is_err(),
        "a pool with u32::MAX blocks (offsets up to ~64 GiB) cannot be addressed with u32 links \
         and must be rejected by new()"
    );
}

// ============================================================================
// S4: SecureMemoryPool alignment (src/memory/secure_pool.rs) and GoldHashIdx
// ============================================================================

/// S4(a). `SecureChunk::new` allocates with `Layout::from_size_align(total, 8)`
/// (secure_pool.rs:505) and returns `raw + size_of::<ChunkHeader>()` =
/// `raw + 40` (541-543); `config.alignment` is only validated (899-903). With
/// glibc's 16-byte-aligned `raw`, medium_secure (alignment 16) yields
/// `addr % 16 == 8` and large_secure (32) yields `addr % 32` in {8, 24}.
#[test]
fn test_v2_s4_preset_pools_honor_configured_alignment() {
    for (name, config) in [
        ("small_secure", SecurePoolConfig::small_secure()),
        ("medium_secure", SecurePoolConfig::medium_secure()),
        ("large_secure", SecurePoolConfig::large_secure()),
    ] {
        let align = config.alignment;
        let pool = SecureMemoryPool::new(config).unwrap();
        let held: Vec<_> = (0..4).map(|_| pool.allocate().unwrap()).collect();
        for p in &held {
            assert_eq!(
                p.as_ptr() as usize % align,
                0,
                "{name}: chunk at {:p} is not {align}-byte aligned",
                p.as_ptr()
            );
            p.validate().unwrap();
        }
    }
}

/// S4(b). Same defect with a caller-chosen alignment: `SecurePoolConfig::new`
/// accepts 64 and the chunks come back 8 (mod 16) aligned. Recycled chunks
/// (thread-local cache) must keep the alignment, too.
#[test]
fn test_v2_s4_custom_alignment_64_is_honored() {
    let pool = SecureMemoryPool::new(SecurePoolConfig::new(256, 8, 64)).unwrap();
    let held: Vec<_> = (0..8).map(|_| pool.allocate().unwrap()).collect();
    for p in &held {
        assert_eq!(
            p.as_ptr() as usize % 64,
            0,
            "chunk at {:p} is not 64-byte aligned",
            p.as_ptr()
        );
        p.validate().unwrap();
    }
    drop(held);
    let recycled = pool.allocate().unwrap();
    assert_eq!(recycled.as_ptr() as usize % 64, 0);
    recycled.validate().unwrap();
}

/// S4(c). `GoldHashIdx::allocate_pooled_value` (gold_hash_idx.rs:396-419)
/// rejects a pooled chunk whose address is not a multiple of
/// `align_of::<V>()`. A 16-byte `V` goes to the global small_secure pool
/// (`get_global_pool_for_size`, secure_pool.rs:1595-1597: alignment 8, chunks
/// at `raw + 40` = 8 mod 16), so with `V = u128` (align 16) every insert fails
/// with "pooled chunk size (1024) / alignment insufficient". Fixing (a) alone
/// is not enough: the small preset must provide at least 16-byte alignment.
#[test]
fn test_v2_s4_gold_hash_idx_stores_u128_values() {
    assert_eq!(
        std::mem::align_of::<u128>(),
        16,
        "precondition (x86_64, Rust >= 1.77)"
    );
    let value = |k: u64| (u128::from(k) << 64) | 0xDEAD_BEEF;
    let mut idx: GoldHashIdx<u64, u128> = GoldHashIdx::new();
    for k in 0..64u64 {
        idx.insert(k, value(k))
            .unwrap_or_else(|e| panic!("insert {k}: {e}"));
    }
    for k in 0..64u64 {
        assert_eq!(idx.get(&k), Some(&value(k)));
    }
}

// ============================================================================
// S7: panic safety of SmallMap::clear and AutoGrowCircularQueue::pop_bulk
// ============================================================================

/// Per-test drop ledger. `DropProbe` holds only a `&DropLog`, so dropping a
/// probe twice (the bug under test) re-runs `Drop::drop` without freeing
/// anything twice: the test stays deterministic and memory-safe at v5.0.0.
#[derive(Default)]
struct DropLog {
    drops: RefCell<Vec<usize>>,
    armed: Cell<bool>,
}

impl DropLog {
    fn armed() -> Self {
        let log = Self::default();
        log.armed.set(true);
        log
    }

    fn count(&self, id: usize) -> usize {
        self.drops.borrow().iter().filter(|&&d| d == id).count()
    }

    fn counts(&self, ids: &[usize]) -> Vec<(usize, usize)> {
        ids.iter().map(|&id| (id, self.count(id))).collect()
    }
}

/// Records each drop; the probe marked `boom` panics on the first drop only.
#[derive(Clone)]
struct DropProbe<'a> {
    id: usize,
    boom: bool,
    log: &'a DropLog,
}

impl<'a> DropProbe<'a> {
    fn new(id: usize, boom: bool, log: &'a DropLog) -> Self {
        Self { id, boom, log }
    }
}

impl Drop for DropProbe<'_> {
    fn drop(&mut self) {
        self.log.drops.borrow_mut().push(self.id);
        if self.boom && self.log.armed.replace(false) {
            panic!("DropProbe {} panics on purpose", self.id);
        }
    }
}

/// S7. `SmallMap::clear` (small_map.rs:566-588) drops elements `0..len` and
/// only then sets `len = 0`. When a destructor panics, `len` keeps its old
/// value and `Drop for SmallMap` (669-687) drops the already-dropped elements
/// a second time. v5.0.0: values 0 and 1 are dropped twice.
#[test]
fn test_v2_s7_small_map_clear_with_panicking_drop_drops_each_value_once() {
    let log = DropLog::armed();
    let mut map: SmallMap<u32, DropProbe<'_>> = SmallMap::new();
    for id in 0..4 {
        map.insert(id as u32, DropProbe::new(id, id == 1, &log))
            .unwrap();
    }
    let result = catch_unwind(AssertUnwindSafe(|| map.clear()));
    assert!(result.is_err(), "the marked destructor must panic");
    drop(map);
    assert_eq!(
        log.counts(&[0, 1, 2, 3]),
        vec![(0, 1), (1, 1), (2, 1), (3, 1)],
        "(value id, times dropped)"
    );
}

/// S7. `AutoGrowCircularQueue::pop_bulk` (circular_queue.rs:1129-1166) moves
/// elements out with `output[i] = read()` -- which first drops the old
/// `output[i]` -- and advances `head`/`len` only after the loop (1160-1162).
/// When that drop panics, the elements already moved into `output` are still
/// owned by the queue and are dropped a second time by `Drop`.
/// v5.0.0: queue elements 0 and 1 are dropped twice.
#[test]
fn test_v2_s7_auto_grow_queue_pop_bulk_with_panicking_drop_drops_each_value_once() {
    let log = DropLog::armed();
    let mut queue = AutoGrowCircularQueue::new();
    for id in 0..4 {
        queue.push_back(DropProbe::new(id, false, &log)).unwrap();
    }
    let mut output: Vec<DropProbe<'_>> = (100..104)
        .map(|id| DropProbe::new(id, id == 101, &log))
        .collect();
    let result = catch_unwind(AssertUnwindSafe(|| queue.pop_bulk(&mut output)));
    assert!(result.is_err(), "the marked destructor must panic");
    drop(output);
    drop(queue);
    let ids = [0, 1, 2, 3, 100, 101, 102, 103];
    let expected: Vec<_> = ids.iter().map(|&id| (id, 1)).collect();
    assert_eq!(log.counts(&ids), expected, "(id, times dropped)");
}

// ============================================================================
// MinimalSso (src/containers/specialized/minimal_sso.rs)
// ============================================================================

/// Needs Miri. `MinimalSso` stores its heap pointer as
/// `(ptr as usize).to_ne_bytes()` (minimal_sso.rs:258-264) and rebuilds it
/// with `usize as *mut u8` (237-241). Under `-Zmiri-strict-provenance` the
/// rebuilt pointer has no provenance, so the first heap access (`realloc` in
/// `heap_grow`, then reads, then `dealloc` in `Drop`) is rejected. Natively
/// the test passes at v5.0.0. The `Send + Sync` check guards the fix: storing
/// a raw pointer must not silently drop the auto traits.
#[test]
fn test_v2_minimal_sso_heap_string_keeps_pointer_provenance() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MinimalSso>();

    let mut s = MinimalSso::from_str("hello");
    s.push_str(" world, this is a long string!!!"); // 37 bytes: spills to the heap
    assert!(!s.is_local());
    s.push_str(&"x".repeat(64)); // 101 bytes: grows the heap buffer
    let expected = format!("hello world, this is a long string!!!{}", "x".repeat(64));
    assert_eq!(s.as_str(), expected);
    let moved = s;
    let cloned = moved.clone();
    assert_eq!(cloned.as_bytes(), expected.as_bytes());
    assert_eq!(moved.len(), expected.len());
}
