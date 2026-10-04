//! C9: lengths and element counts read from untrusted input must not size an
//! allocation before the input has delivered the bytes they describe.
//!
//! The global allocator below records, per thread, the largest single
//! allocation request (`alloc`, `alloc_zeroed` or the new size of a
//! `realloc`) made while tracking is switched on. Each test bounds what a
//! hostile prefix can make the reader allocate. Requests go to `System`
//! unchanged, so on the unfixed code a huge zeroed request is recorded
//! without committing memory, and the test fails on its assertion instead of
//! exhausting the machine.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::Cursor;

use zipora::io::{DataInput, ReaderDataInput, VarInt};

struct CountingAlloc;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static MAX_ALLOC: Cell<usize> = const { Cell::new(0) };
}

fn record(size: usize) {
    // `try_with`: thread-local storage is unavailable while a thread is torn
    // down, and an allocator must not panic.
    let _ = TRACKING.try_with(|tracking| {
        if tracking.get() {
            let _ = MAX_ALLOC.try_with(|max| max.set(max.get().max(size)));
        }
    });
}

// SAFETY: every method forwards its arguments unchanged to `System`, so the
// `GlobalAlloc` guarantees are `System`'s. `record` neither allocates nor
// unwinds.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract for `layout`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        // SAFETY: the caller guarantees that `ptr` came from this allocator,
        // which means from `System`, with `layout`, and that `new_size` is valid.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller guarantees that `ptr` came from `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Runs `f` and returns its result together with the largest single
/// allocation it requested on this thread.
fn max_alloc_during<R>(f: impl FnOnce() -> R) -> (R, usize) {
    MAX_ALLOC.with(|max| max.set(0));
    TRACKING.with(|tracking| tracking.set(true));
    let result = f();
    TRACKING.with(|tracking| tracking.set(false));
    (result, MAX_ALLOC.with(Cell::get))
}

/// The read-ahead budget, `MAX_READ_AHEAD_ALLOC` in `src/io/data_input.rs`.
const READ_AHEAD: usize = 64 * 1024;

#[test]
fn test_c9_di1_stream_length_prefix_allocates_at_most_the_read_ahead() {
    // A 2^30 length prefix followed by three bytes, from a source that
    // cannot report its remaining length.
    let mut data = Vec::new();
    VarInt::write_to(&mut data, 1 << 30).unwrap();
    data.extend_from_slice(b"abc");
    let (result, max_alloc) = max_alloc_during(|| {
        ReaderDataInput::new(Cursor::new(data)).read_length_prefixed_bytes()
    });
    assert!(result.is_err());
    assert!(
        max_alloc <= READ_AHEAD,
        "largest single allocation was {} bytes",
        max_alloc
    );
}
