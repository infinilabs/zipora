//! Fuzz SimpleZipBlobStore build + record extraction round-trip (plan.md 6.1).
//! There is no untrusted deserializer for this store; the fuzz surface is
//! build_from over arbitrary record sets + get() consistency (C3 regression).
#![no_main]
use libfuzzer_sys::fuzz_target;
use zipora::blob_store::{BlobStore, SimpleZipBlobStore, SimpleZipConfig};

fuzz_target!(|data: &[u8]| {
    // Parse data into length-prefixed records (cap count and total size)
    let mut records: Vec<Vec<u8>> = Vec::new();
    let mut pos = 0usize;
    while pos + 1 < data.len() && records.len() < 64 {
        let len = (data[pos] as usize).min(data.len() - pos - 1);
        records.push(data[pos + 1..pos + 1 + len].to_vec());
        pos += 1 + len;
    }
    if records.is_empty() {
        return;
    }
    let config = SimpleZipConfig::default();
    if let Ok(store) = SimpleZipBlobStore::build_from(&records, &config) {
        for (i, rec) in records.iter().enumerate() {
            let got = store.get(i as u32).expect("built record must be readable");
            assert_eq!(&got, rec, "record {} corrupted by round-trip", i);
        }
    }
});
