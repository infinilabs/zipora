//! `Rans64Decoder`: the largest single allocation request made while decoding a stream of more
//! than 1 MiB must not exceed `output_length` (5.0.1 doubled its buffer past it and asked for
//! 4 MiB to decode 2 MiB + 5 bytes).
//!
//! The in-crate test `test_v2_m1_windowed_decode_caps_reservation_at_output_length` only checks
//! the final capacity, which `shrink_to_fit` makes exact either way. This file installs a
//! counting `#[global_allocator]`, so it holds a single `#[test]`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use zipora::entropy::rans::ParallelVariant;
use zipora::entropy::{ParallelX1, ParallelX2, ParallelX4, ParallelX8, Rans64Encoder, RansDecoder};

/// Records the largest single `alloc`/`realloc` request made by the measuring thread.
struct PeakRequest;

static MAX_REQUEST: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

fn note(size: usize) {
    if MEASURING.try_with(|m| m.get()).unwrap_or(false) {
        MAX_REQUEST.fetch_max(size, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for PeakRequest {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: PeakRequest = PeakRequest;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Returns (largest allocation request during `decode`, FNV-1a hash of the encoded stream).
fn max_request_during_decode<P: ParallelVariant>(data: &[u8], freqs: &[u32; 256]) -> (usize, u64) {
    let encoder = Rans64Encoder::<P>::new(freqs).expect("encoder");
    let encoded = encoder.encode(data).expect("encode");
    let decoder = RansDecoder::<P>::new(&encoder);

    MAX_REQUEST.store(0, Ordering::SeqCst);
    MEASURING.with(|m| m.set(true));
    let decoded = decoder.decode(&encoded, data.len());
    MEASURING.with(|m| m.set(false));

    let decoded = decoded.expect("decode");
    assert!(decoded == data, "{}: round trip mismatch", P::NAME);
    (MAX_REQUEST.load(Ordering::SeqCst), fnv1a64(&encoded))
}

#[test]
fn test_v2_m1_rans_decode_never_requests_more_than_output_length() {
    let len = (2usize << 20) + 5;
    let data: Vec<u8> = (0..len).map(|i| (i * 13 % 7) as u8 + b'a').collect();
    let mut freqs = [0u32; 256];
    for &b in &data {
        freqs[b as usize] += 1;
    }

    let results = [
        ("x1", max_request_during_decode::<ParallelX1>(&data, &freqs)),
        ("x2", max_request_during_decode::<ParallelX2>(&data, &freqs)),
        ("x4", max_request_during_decode::<ParallelX4>(&data, &freqs)),
        ("x8", max_request_during_decode::<ParallelX8>(&data, &freqs)),
    ];

    let mut failures = Vec::new();
    for (name, (max_req, enc_hash)) in results {
        println!(
            "{name}: max single allocation request during decode = {max_req} \
             (output_length = {len}); encoded fnv1a64 = {enc_hash:016x}"
        );
        if max_req > len {
            failures.push(format!(
                "{name}: requested {max_req} bytes > output_length {len}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
