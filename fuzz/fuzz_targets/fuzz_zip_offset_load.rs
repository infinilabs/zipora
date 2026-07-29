//! Fuzz ZipOffsetBlobStore::load_from_reader with arbitrary bytes (plan.md 6.1).
//! Must return Err (never panic, OOM, or UB) on malformed input.
#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use zipora::blob_store::ZipOffsetBlobStore;

fuzz_target!(|data: &[u8]| {
    let mut cursor = Cursor::new(data);
    let _ = ZipOffsetBlobStore::load_from_reader(&mut cursor);
});
