# Compression Framework

Zipora provides dictionary compression (`zipora::compression`), entropy coders (`zipora::entropy`), and compressed blob stores (`zipora::blob_store`).

## PA-Zip Dictionary Compression (DictZip)

Dictionary compression is exposed as a blob store: train a dictionary from samples with `DictZipBlobStoreBuilder`, then store/retrieve records through the standard `BlobStore` trait.

```rust
use zipora::blob_store::BlobStore;
use zipora::compression::dict_zip::{DictZipBlobStoreBuilder, DictZipConfig};

// Train a dictionary from sample data
let mut builder = DictZipBlobStoreBuilder::with_config(DictZipConfig::text_compression()).unwrap();
builder.add_training_sample(b"GET /api/users HTTP/1.1").unwrap();
builder.add_training_sample(b"GET /api/posts HTTP/1.1").unwrap();
builder.add_training_sample(b"POST /api/users HTTP/1.1").unwrap();

// Build the compressed blob store
let mut store = builder.finish().unwrap();

// Records are compressed against the trained dictionary
let id = store.put(b"GET /api/users HTTP/1.1").unwrap();
let data = store.get(id).unwrap();
assert_eq!(data, b"GET /api/users HTTP/1.1");
```

Configuration presets: `DictZipConfig::text_compression()`, `binary_compression()`, `log_compression()`, `realtime_compression()`. `DictZipBlobStoreBuilder::new()` uses the default config.

Lower-level building blocks are also exported from `zipora::compression`: `PaZipDictionaryBuilder` (dictionary training with `DictionaryBuilderConfig`), `SuffixArrayDictionary`, and `PatternMatcher`.

## Huffman Coding

Huffman coding lives in `zipora::entropy`. Order-0 uses `HuffmanEncoder`/`HuffmanDecoder`; Order-1/Order-2 context models use `ContextualHuffmanEncoder` with a `HuffmanOrder`.

```rust
use zipora::entropy::{
    ContextualHuffmanDecoder, ContextualHuffmanEncoder, HuffmanDecoder, HuffmanEncoder,
    HuffmanOrder,
};

let data = b"sample data with repeated symbols";

// Order-0 Huffman (symbols are independent)
let encoder = HuffmanEncoder::new(data).unwrap();
let compressed = encoder.encode(data).unwrap();

let decoder = HuffmanDecoder::new(encoder.tree().clone());
let decompressed = decoder.decode(&compressed, data.len()).unwrap();
assert_eq!(decompressed, data);

// Order-1 Huffman (frequencies conditioned on the previous byte)
let o1 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
let o1_compressed = o1.encode(data).unwrap();

let o1_decoder = ContextualHuffmanDecoder::new(o1); // consumes the encoder
let o1_decoded = o1_decoder.decode(&o1_compressed, data.len()).unwrap();
assert_eq!(o1_decoded, data);

// Order-2 uses the previous two bytes
let o2 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();
let o2_compressed = o2.encode(data).unwrap();
```

## FSE (Finite State Entropy)

```rust
use zipora::entropy::{FseConfig, FseDecoder, FseEncoder};

let data = b"data with varying symbol frequencies";

// Presets: fast_compression(), high_compression(), realtime(), balanced()
let mut encoder = FseEncoder::new(FseConfig::balanced()).unwrap();
let compressed = encoder.compress(data).unwrap();

let mut decoder = FseDecoder::new(); // or FseDecoder::with_config(config)
let decompressed = decoder.decompress(&compressed).unwrap();
assert_eq!(decompressed, data);
```

Notes:

- **Stream format**: every FSE stream starts with a mode byte — `0xF5` (single stream) or `0xF6` (parallel multi-block) — so the decoder never sniffs the layout. This format is **not compatible with pre-7.4 streams**.
- Histogram construction uses a 4-lane unrolled scalar loop (not AVX2).

## rANS (Range Asymmetric Numeral Systems)

`Rans64Encoder` is parameterized by a parallelism variant: `ParallelX1` (single stream), `ParallelX2`, `ParallelX4`, or `ParallelX8` (interleaved streams).

```rust
use zipora::entropy::rans::{ParallelX1, ParallelX4, Rans64Decoder, Rans64Encoder};

let data = b"data for rANS compression";

// Build a frequency table (one count per byte value)
let mut frequencies = [0u32; 256];
for &byte in data.iter() {
    frequencies[byte as usize] += 1;
}

let encoder = Rans64Encoder::<ParallelX1>::new(&frequencies).unwrap();
let compressed = encoder.encode(data).unwrap();

let decoder = Rans64Decoder::new(&encoder);
let decompressed = decoder.decode(&compressed, data.len()).unwrap();
assert_eq!(decompressed, data);

// Interleaved 4-stream variant for larger inputs
let parallel = Rans64Encoder::<ParallelX4>::new(&frequencies).unwrap();
let parallel_compressed = parallel.encode(data).unwrap();
```

## ZSTD Integration

ZSTD is exposed as a compressed blob store wrapper (requires the default `zstd` feature) and as `Algorithm::Zstd(i32)` in the compression framework.

```rust
use zipora::blob_store::{BlobStore, MemoryBlobStore, ZstdBlobStore};

// Wrap any BlobStore with transparent ZSTD compression (level 1-22)
let mut store = ZstdBlobStore::new(MemoryBlobStore::new(), 3);

let data = b"data for ZSTD compression";
let id = store.put(data).unwrap();
let retrieved = store.get(id).unwrap();
assert_eq!(retrieved, data);
```

## Adaptive Compression

`AdaptiveCompressor` selects an algorithm (`Algorithm::None`/`Lz4`/`Zstd`/`Huffman`/`Rans`/...) based on data characteristics and performance requirements.

```rust
use std::time::Duration;
use zipora::compression::{AdaptiveCompressor, AdaptiveConfig, PerformanceRequirements};

let requirements = PerformanceRequirements {
    max_latency: Duration::from_millis(100),
    min_throughput: 100_000_000, // 100 MB/s
    max_memory: 64 * 1024 * 1024,
    target_ratio: 0.5,
    speed_vs_quality: 0.5, // 0.0 = speed, 1.0 = quality
};

let compressor = AdaptiveCompressor::new(AdaptiveConfig::default(), requirements).unwrap();

let data = b"adaptive compression input";
let compressed = compressor.compress(data).unwrap();
let decompressed = compressor.decompress(&compressed).unwrap();
assert_eq!(decompressed, data);

println!("Selected algorithm: {:?}", compressor.current_algorithm());
```

`CompressionProfile` records per-data-type performance; `compressor.train(&samples)` builds profiles from labeled samples.

## Compression Algorithm Selection

| Algorithm | Ratio | Speed | Best Use Case |
|-----------|-------|-------|---------------|
| **DictZip (PA-Zip)** | Data-dependent | Fast | Structured data, logs, repeated patterns |
| **Huffman O0/O1/O2** | Data-dependent | Fast | Text, structured data |
| **FSE** | Data-dependent | Fast | Variable symbol frequencies |
| **rANS** | Data-dependent | Fast | High precision entropy coding |
| **ZSTD** | Data-dependent | Moderate | General purpose, best ratio |
| **LZ4** | Data-dependent | Very Fast | Speed-critical applications |
| **StreamVByte** | 1.5-3x | **Ultra Fast (SSSE3)** | Inverted index delta lists, integer sequences |

## StreamVByte (SIMD-Accelerated Variable-Byte)

StreamVByte separates control bytes (2 bits per integer indicating length 1..=4 bytes) from data bytes. Decoding uses Lemire's SSSE3 shuffle-table algorithm (`_mm_shuffle_epi8` / `pshufb`) with a 256-entry precomputed mask table, achieving **8.7x faster decoding** than scalar varint.

```rust
use zipora::compression::stream_vbyte::StreamVByte;

// Encode sorted doc IDs with delta + stream vbyte
let doc_ids = vec![1, 5, 100, 300, 1000, 70000];
let encoded = StreamVByte::encode_deltas(&doc_ids);

// Fast SIMD decoding (SSSE3 shuffle table + scalar tail fallback)
let decoded = StreamVByte::decode_deltas(&encoded, doc_ids.len());
assert_eq!(decoded, doc_ids);

// Direct decode into pre-allocated buffer (raw values, no delta coding)
let raw = StreamVByte::encode_raw(&doc_ids);
let mut output = vec![0u32; doc_ids.len()];
let count = StreamVByte::decode_into(&raw, doc_ids.len(), &mut output);
assert_eq!(&output[..count], &doc_ids[..]);
```

## Hardware Acceleration

SIMD acceleration (histogram computation, bit manipulation, shuffle-table decoding) is dispatched automatically at runtime. See [docs/SIMD.md](SIMD.md) for the tier list and dispatch framework.

## Verified Performance

Measured numbers (see [docs/PERFORMANCE.md](PERFORMANCE.md) for methodology):

- **Huffman O1**: 173-188 µs for 65 KB
- **rANS**: 351-426 µs for 65 KB
- **StreamVByte decode**: 8.7x vs scalar varint
