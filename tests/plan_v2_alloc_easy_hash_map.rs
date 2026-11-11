//! `EasyHashMap`: an insert must survive one failed allocation of the doubled table. `grow()`
//! is all-or-nothing, and the underlying `ZiporaHashMap::insert` grows the table by itself, so
//! `put` must still store the entry and `get_or_insert` must still return it (as in 5.0.1).
//!
//! The `#[global_allocator]` fails exactly one allocation of at least `2 * capacity` bytes: the
//! control-byte array of the doubled table. It is process-global, so this file holds a single
//! `#[test]`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use zipora::containers::EasyHashMap;

struct FailOnce;

static ARMED: AtomicBool = AtomicBool::new(false);
static THRESHOLD: AtomicUsize = AtomicUsize::new(usize::MAX);
static INJECTED: AtomicUsize = AtomicUsize::new(0);

fn inject(size: usize) -> bool {
    if size >= THRESHOLD.load(Ordering::Relaxed) && ARMED.swap(false, Ordering::Relaxed) {
        INJECTED.fetch_add(1, Ordering::Relaxed);
        return true;
    }
    false
}

unsafe impl GlobalAlloc for FailOnce {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if inject(layout.size()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if inject(layout.size()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if inject(new_size) {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: FailOnce = FailOnce;

const TARGET_CAP: usize = 1 << 17;

/// A map whose next fresh-key insert makes `EasyHashMap` grow a
/// `TARGET_CAP`-slot table (its `should_grow`: `len / capacity >= 0.75`).
fn map_about_to_grow() -> (EasyHashMap<u64, u64>, u64) {
    let mut map = EasyHashMap::new();
    let mut next = 0u64;
    loop {
        let cap = map.capacity();
        if cap >= TARGET_CAP && map.len() as f64 / cap as f64 >= 0.75 {
            return (map, next);
        }
        map.put(next, next);
        next += 1;
    }
}

fn fail_next_table_allocation(cap: usize) {
    THRESHOLD.store(2 * cap, Ordering::Relaxed);
    INJECTED.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
}

/// One `#[test]` only: the fault-injection state is process-global.
#[test]
fn test_v2_insert_survives_one_failed_growth_allocation() {
    let mut failures = Vec::new();

    let (mut map, key) = map_about_to_grow();
    let len = map.len();
    fail_next_table_allocation(map.capacity());
    map.put(key, 7);
    ARMED.store(false, Ordering::Relaxed);
    assert_eq!(
        INJECTED.load(Ordering::Relaxed),
        1,
        "fault was not injected (put)"
    );
    if map.get(&key) != Some(&7) || map.len() != len + 1 {
        failures.push(format!(
            "put dropped the entry after one failed growth allocation (len {} -> {})",
            len,
            map.len()
        ));
    }

    let (mut map, key) = map_about_to_grow();
    fail_next_table_allocation(map.capacity());
    let got = map.get_or_insert(key, 7).map(|v| *v);
    ARMED.store(false, Ordering::Relaxed);
    assert_eq!(
        INJECTED.load(Ordering::Relaxed),
        1,
        "fault was not injected (get_or_insert)"
    );
    if got.as_ref().ok() != Some(&7) {
        failures.push(format!(
            "get_or_insert returned {got:?} although the table could still grow"
        ));
    }

    assert!(failures.is_empty(), "{failures:#?}");
}
