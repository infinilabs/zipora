//! Structure-aware `arbitrary` fuzz target for blob store build -> save -> load -> get (plan.md B2.1).
#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use zipora::blob_store::{
    BlobStore, MixedLenBlobStore, SimpleZipBlobStore, SimpleZipConfig, ZipOffsetBlobStore,
    ZipOffsetBlobStoreBuilder,
};

fuzz_target!(|pairs: Vec<(Vec<u8>, Vec<u8>)>| {
    let mut records: Vec<Vec<u8>> = Vec::new();
    for (k, v) in pairs.into_iter().take(48) {
        let mut combined = Vec::with_capacity((k.len() + v.len()).min(512));
        let take_k = k.len().min(256);
        let take_v = v.len().min(256);
        combined.extend_from_slice(&k[..take_k]);
        combined.extend_from_slice(&v[..take_v]);
        records.push(combined);
    }
    if records.is_empty() {
        return;
    }

    // 1. ZipOffsetBlobStore build -> save_to_writer -> load_from_reader -> get
    let mut builder = ZipOffsetBlobStoreBuilder::new().expect("builder init");
    for rec in &records {
        builder.add_record(rec).expect("add_record must succeed");
    }
    let built = builder.finish().expect("builder finish must succeed");
    let mut wire = Vec::new();
    built
        .save_to_writer(&mut wire)
        .expect("save_to_writer must succeed");

    let mut cursor = Cursor::new(&wire);
    let loaded =
        ZipOffsetBlobStore::load_from_reader(&mut cursor).expect("load_from_reader must succeed");
    assert_eq!(loaded.len(), records.len());
    for (i, expected) in records.iter().enumerate() {
        let got = loaded
            .get(i as u32)
            .expect("loaded record must be readable");
        assert_eq!(&got, expected, "ZipOffsetBlobStore record {i} mismatch");
    }

    // 2. SimpleZipBlobStore build_from -> get
    if let Ok(simple) = SimpleZipBlobStore::build_from(&records, &SimpleZipConfig::default()) {
        for (i, expected) in records.iter().enumerate() {
            let got = simple
                .get(i as u32)
                .expect("SimpleZipBlobStore record readable");
            assert_eq!(&got, expected, "SimpleZipBlobStore record {i} mismatch");
        }
    }

    // 3. MixedLenBlobStore build_from -> get_ref
    if let Ok(mixed) = MixedLenBlobStore::build_from(&records) {
        for (i, expected) in records.iter().enumerate() {
            let got = mixed
                .get_ref(i as u32)
                .expect("MixedLenBlobStore record readable");
            assert_eq!(got, &expected[..], "MixedLenBlobStore record {i} mismatch");
        }
    }
});
