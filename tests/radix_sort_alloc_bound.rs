//! Regression test for the radix-sort over-allocation bug (32 GiB OOM).
//!
//! `RadixSort::sort_u32` selected counting sort purely on element *count*
//! (`data.len() <= use_counting_sort_threshold`), then allocated `(max_val + 1)`
//! buckets. A 5-element slice containing `u32::MAX` therefore requested
//! `(2^32) * 8 bytes = 32 GiB`. On a memory-constrained machine (CI runner) the
//! allocation failed and aborted the process; on a large dev box Linux
//! overcommit + lazy zero pages let it silently succeed, so a *correctness-only*
//! assertion did NOT catch the bug.
//!
//! This test instead asserts the *peak single allocation* stays bounded, which
//! fails on the buggy code on ANY machine — independent of installed RAM.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use zipora::algorithms::radix_sort::RadixSort;

/// Global allocator that records the largest single allocation request.
struct TrackingAllocator;

static MAX_SINGLE_ALLOC: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        MAX_SINGLE_ALLOC.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Counting sort uses `vec![0usize; n]`, which routes through
        // `alloc_zeroed`. Record the size, then delegate to `System` (calloc →
        // lazy zero pages) so that even a buggy 32 GiB request is *recorded*
        // without committing physical memory or OOMing the machine running the
        // test. This lets the regression fail via an assertion, not an abort.
        MAX_SINGLE_ALLOC.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator;

#[test]
fn sort_u32_small_slice_with_large_values_does_not_overallocate() {
    // 5 elements, but the max value is u32::MAX — the exact shape that drove the
    // buggy counting-sort path to request (u32::MAX + 1) * 8 = 32 GiB.
    let mut data = vec![u32::MAX, 1_000_000, 500_000, 0, 999_999];

    MAX_SINGLE_ALLOC.store(0, Ordering::Relaxed);
    let mut sorter = RadixSort::new();
    sorter.sort_u32(&mut data).expect("sort_u32 should succeed");
    let peak = MAX_SINGLE_ALLOC.load(Ordering::Relaxed);

    // Correctness is still required.
    assert_eq!(data, vec![0, 500_000, 999_999, 1_000_000, u32::MAX]);

    // The real regression guard: bound the largest single allocation. The fixed
    // path uses LSD radix sort with O(len) memory; the bug requested 32 GiB.
    // 64 MiB sits far above any legitimate allocation for 5 elements and far
    // below the 32 GiB bug, so it cleanly separates fixed from regressed on any
    // machine, with or without overcommit.
    const LIMIT: usize = 64 * 1024 * 1024;
    assert!(
        peak < LIMIT,
        "sort_u32 made a single {peak}-byte allocation for a 5-element slice \
         (counting-sort range guard regressed); expected < {LIMIT} bytes"
    );
}
