use std::time::Instant;
use zipora::blob_store::traits::BlobStore;
use zipora::compression::dict_zip::blob_store::*;

#[test]
fn test_clone_performance() {
    let mut config = DictZipConfig::default();
    config.min_compression_size = 10;
    let mut builder = DictZipBlobStoreBuilder::with_config(config).unwrap();

    // Train with 256KB of data to build a sizable DFA (10MB takes >100s in
    // builder.finish() — linear ~10s/MB — without changing what this test checks).
    let big_data = vec![42u8; 256 * 1024];
    builder.add_training_sample(&big_data).unwrap();
    let mut store = builder.finish().unwrap();

    let data = vec![b"hello world, this is a longer string".to_vec(); 1000];
    let start = Instant::now();
    for d in data {
        store.put(&d).unwrap();
    }
    println!("Elapsed for 1000 puts: {:?}", start.elapsed());
}
