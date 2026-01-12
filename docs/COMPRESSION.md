# Compression Framework

Zipora provides a comprehensive compression framework with multiple algorithms and real-time capabilities.

## PA-Zip Dictionary Compression

PA-Zip is a high-performance dictionary compression system optimized for structured data.

```rust
use zipora::compression::{
    PaZipEncoder, PaZipDecoder, PaZipConfig,
    DictionaryBuilder, DictionaryConfig
};

// Build dictionary from sample data
let samples = vec![
    b"GET /api/users HTTP/1.1".to_vec(),
    b"GET /api/posts HTTP/1.1".to_vec(),
    b"POST /api/users HTTP/1.1".to_vec(),
];

let dict_config = DictionaryConfig::performance_optimized();
let dictionary = DictionaryBuilder::build_from_samples(&samples, dict_config).unwrap();

// Create encoder with dictionary
let config = PaZipConfig::balanced();
let mut encoder = PaZipEncoder::with_dictionary(config, dictionary.clone()).unwrap();

// Compress data
let data = b"GET /api/users HTTP/1.1";
let compressed = encoder.encode(data).unwrap();
println!("Compression ratio: {:.2}x", data.len() as f64 / compressed.len() as f64);

// Decompress
let mut decoder = PaZipDecoder::with_dictionary(dictionary).unwrap();
let decompressed = decoder.decode(&compressed).unwrap();
assert_eq!(decompressed, data);

// Streaming compression
let mut stream_encoder = PaZipEncoder::streaming(config).unwrap();
stream_encoder.write_chunk(b"first chunk").unwrap();
stream_encoder.write_chunk(b"second chunk").unwrap();
let final_compressed = stream_encoder.finish().unwrap();
```

## Huffman Coding

```rust
use zipora::compression::{
    HuffmanEncoder, HuffmanDecoder,
    HuffmanO1Encoder, HuffmanO2Encoder, // Contextual variants
    ContextualHuffmanEncoder
};

// Basic Huffman encoding (Order-0)
let encoder = HuffmanEncoder::new(b"sample data").unwrap();
let compressed = encoder.encode(b"sample data").unwrap();

let decoder = HuffmanDecoder::from_encoder(&encoder).unwrap();
let decompressed = decoder.decode(&compressed).unwrap();

// Order-1 Huffman (context-aware, uses previous byte)
let o1_encoder = HuffmanO1Encoder::new(b"training data").unwrap();
let o1_compressed = o1_encoder.encode(b"similar data").unwrap();

// Order-2 Huffman (uses two previous bytes)
let o2_encoder = HuffmanO2Encoder::new(b"training data").unwrap();
let o2_compressed = o2_encoder.encode(b"similar data").unwrap();

// Contextual Huffman with fast symbol table
let config = ContextualHuffmanConfig {
    order: 1,
    use_fast_table: true,
    table_size: 256,
};
let contextual = ContextualHuffmanEncoder::with_config(b"training", config).unwrap();
let result = contextual.encode(b"test data").unwrap();
println!("Speedup with fast table: 2.1-2.6x");
```

## FSE (Finite State Entropy)

```rust
use zipora::compression::{FseEncoder, FseDecoder, FseConfig};

// FSE encoding with ZSTD optimizations
let config = FseConfig::zstd_compatible();
let mut encoder = FseEncoder::with_config(config).unwrap();

let data = b"data with varying symbol frequencies";
let compressed = encoder.encode(data).unwrap();

let mut decoder = FseDecoder::from_encoder(&encoder).unwrap();
let decompressed = decoder.decode(&compressed).unwrap();

// Statistics
let stats = encoder.stats();
println!("Entropy: {:.3} bits/symbol", stats.entropy);
println!("Compression ratio: {:.2}x", stats.compression_ratio);
```

## rANS (Range Asymmetric Numeral Systems)

```rust
use zipora::compression::{
    Rans64Encoder, Rans64Decoder,
    RansConfig, ParallelRansEncoder
};

// 64-bit rANS encoding
let config = RansConfig::high_precision();
let mut encoder = Rans64Encoder::with_config(config).unwrap();

let data = b"data for rANS compression";
let compressed = encoder.encode(data).unwrap();

let mut decoder = Rans64Decoder::from_encoder(&encoder).unwrap();
let decompressed = decoder.decode(&compressed).unwrap();

// Parallel rANS for large data
let parallel_config = RansConfig::parallel(8); // 8 streams
let mut parallel_encoder = ParallelRansEncoder::with_config(parallel_config).unwrap();
let large_data = vec![0u8; 10_000_000];
let parallel_compressed = parallel_encoder.encode(&large_data).unwrap();
```

## ZSTD Integration

```rust
use zipora::compression::{ZstdEncoder, ZstdDecoder, ZstdConfig};

// ZSTD compression with configurable level
let config = ZstdConfig {
    level: 10,
    window_log: 22,
    enable_checksums: true,
};
let mut encoder = ZstdEncoder::with_config(config).unwrap();

let data = b"data for ZSTD compression";
let compressed = encoder.encode(data).unwrap();

let mut decoder = ZstdDecoder::new().unwrap();
let decompressed = decoder.decode(&compressed).unwrap();

// Streaming ZSTD
let mut stream = ZstdEncoder::streaming(config).unwrap();
stream.write_chunk(b"chunk 1").unwrap();
stream.write_chunk(b"chunk 2").unwrap();
let final_data = stream.finish().unwrap();

// Dictionary-based ZSTD
let dict = ZstdDictionary::train(&samples, 64 * 1024).unwrap();
let dict_encoder = ZstdEncoder::with_dictionary(config, dict).unwrap();
```

## Real-Time Compression

```rust
use zipora::compression::{
    RealTimeCompressor, RealTimeConfig, LatencyBudget
};

// Real-time compression with strict latency guarantees
let config = RealTimeConfig {
    max_latency_us: 100,          // 100 microsecond max latency
    target_ratio: 2.0,             // Target 2x compression
    adaptive_level: true,          // Adjust level based on latency
    budget: LatencyBudget::Strict,
};

let mut compressor = RealTimeCompressor::with_config(config).unwrap();

// Compress with latency monitoring
let data = b"real-time data stream";
let result = compressor.compress_with_deadline(data).unwrap();

println!("Achieved latency: {}us", result.latency_us);
println!("Compression ratio: {:.2}x", result.ratio);

// Adaptive compression for varying workloads
let mut adaptive = RealTimeCompressor::adaptive(config).unwrap();
for chunk in data_stream {
    let compressed = adaptive.compress_adaptive(chunk).unwrap();
    // Automatically adjusts compression level to meet latency budget
}
```

## Compression Algorithm Selection

| Algorithm | Ratio | Speed | Best Use Case |
|-----------|-------|-------|---------------|
| **PA-Zip** | 3-10x | Fast | Structured data, logs |
| **Huffman O0** | 1.5-3x | Very Fast | General purpose |
| **Huffman O1** | 2-4x | Fast | Text, structured data |
| **Huffman O2** | 2.5-5x | Moderate | Highly structured data |
| **FSE** | 2-4x | Fast | Variable symbol frequencies |
| **rANS** | 2-4x | Fast | High precision entropy coding |
| **ZSTD** | 3-10x | Moderate | General purpose, best ratio |
| **LZ4** | 2-3x | Very Fast | Speed-critical applications |

## Hardware Acceleration

```rust
use zipora::compression::HardwareAcceleration;

// Check available hardware features
let hw = HardwareAcceleration::detect();
println!("BMI2: {}", hw.has_bmi2);
println!("AVX2: {}", hw.has_avx2);
println!("POPCNT: {}", hw.has_popcnt);

// Automatically uses hardware acceleration when available
// - BMI2: Bit manipulation for entropy coding
// - AVX2: Parallel histogram computation
// - POPCNT: Fast bit counting for symbol statistics
```

## Performance Targets

- **Huffman O1**: 2.1-2.6x speedup with fast symbol table
- **Radix Sort in compression**: 4-8x vs comparison sorts
- **SIMD histogram**: 4-8x faster frequency counting
- **Parallel rANS**: Near-linear scaling to 8+ threads
