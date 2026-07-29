//! Fuzz MixedLenBlobStore build + get_ref round-trip (plan.md 6.1, H10 regression).
#![no_main]
use libfuzzer_sys::fuzz_target;
use zipora::blob_store::MixedLenBlobStore;

fuzz_target!(|data: &[u8]| {
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
    if let Ok(store) = MixedLenBlobStore::build_from(&records) {
        for (i, rec) in records.iter().enumerate() {
            let got = store
                .get_ref(i as u32)
                .expect("built record must be readable");
            assert_eq!(got, &rec[..], "record {} corrupted by round-trip", i);
        }
    }
});
