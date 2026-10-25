//! Fuzz ZipOffsetBlobStore::load_from_reader with arbitrary bytes (plan.md 6.1 / B2.1).
//! Must return Err (never panic, OOM, or UB) on malformed input, and exercise record
//! extraction when header + index + checksum validation passes.
#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use zipora::blob_store::{BlobStore, ZipOffsetBlobStore};

fuzz_target!(|data: &[u8]| {
    let mut cursor = Cursor::new(data);
    if let Ok(store) = ZipOffsetBlobStore::load_from_reader(&mut cursor) {
        let n = store.len().min(16);
        for i in 0..n {
            let _ = store.get(i as u32);
        }
    }
});
