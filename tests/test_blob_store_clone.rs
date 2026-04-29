use std::time::Instant;
use zipora::blob_store::traits::BlobStore;
use zipora::compression::dict_zip::blob_store::*;

#[test]
fn test_clone_performance() {
    let mut config = DictZipConfig::default();
    config.min_compression_size = 10;
    let mut builder = DictZipBlobStoreBuilder::with_config(config).unwrap();

    // Train with 10MB of data to make the DFA huge
    let big_data = vec![42u8; 10_000_000];
    builder.add_training_sample(&big_data).unwrap();
    let mut store = builder.finish().unwrap();

    let data = vec![b"hello world, this is a longer string".to_vec(); 1000];
    let start = Instant::now();
    for d in data {
        store.put(&d).unwrap();
    }
    println!("Elapsed for 1000 puts: {:?}", start.elapsed());
}
