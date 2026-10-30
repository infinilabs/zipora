# Zipora

[English](README.md)

[![Build Status](https://github.com/infinilabs/zipora/workflows/CI/badge.svg)](https://github.com/infinilabs/zipora/actions)
[![License](https://img.shields.io/badge/license-BDL--1.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.95+-orange.svg)](https://www.rust-lang.org)

高性能 Rust 数据结构与压缩算法库，提供内存安全保证。

## 核心特性

<!-- BEGIN GENERATED KEY FEATURES (scripts/perf_table.py) -->
- **高性能**：零拷贝操作、SIMD 优化（AVX2、AVX-512）、缓存友好的内存布局、Block-Max WAND SIMD 游标原语
- **内存安全**：所有生产环境 `unsafe` 位置均附有 `// SAFETY:` 或 `# Safety` 注释（`scripts/unsafe_audit.py`）
- **安全内存管理**：线程安全的生产级内存池（`SecureMemoryPool`、`LockFreeMemoryPool`、`FixedCapacityPool`、`AdaptiveFiveLevelPool`、`BumpArena`），支持 RAII（`tests/memory_pool_contract.rs`）
- **Blob 存储**：专用存储引擎（`MemoryBlobStore`、`PlainBlobStore`、`SimpleZipBlobStore`、`MixedLenBlobStore`、`ZipOffsetBlobStore`、`NestLoudsTrieBlobStore`、`DictZipBlobStore`、`ZstdBlobStore`、`Lz4BlobStore`），支持 Trie 索引和压缩（`tests/format_spec_golden_tests.rs`）
- **简洁数据结构**：`Rank9`（Vigna 2008，`25.00%` 额外开销）、`RankSelectInterleaved256`（仅 rank `25.00%` / 含 select 缓存 `31.25%`）、`RankSelectSE256`、`EliasFano` / `PartitionedEliasFano` / `OptimalPartitionedEliasFano` / `ClusteredEliasFano`（支持游标 `advance_to_index`）、`HybridPostingList`（自动选择编码策略）、AMD 安全 PDEP（`has_fast_bmi2` 检测）（`tests/memory_footprint_e6.rs`）
- **BM25 评分**：`FieldnormEncoder`（Lucene SmallFloat，`1.0 B/doc` 字段长度） + `Bm25BatchScorer`（AVX2/AVX-512 SIMD 批量评分、预取）（`tests/memory_footprint_e6.rs`、`benches/core_bench.rs`）
- **专用容器**：`FastVec`、`ValVec32`、`IntVec`、`UintVecMin0`、`ZipIntVec`、`FixedLenStrVec`、`SortableStrVec`、`ZoSortedStrVec`、`VecTrbSet`/`VecTrbMap`、`MinimalSso`、`LruMap`、`ConcurrentLruMap`、`FixedCircularQueue`
- **哈希表**：SwissTable 16 路 SSE2 组探测 `ZiporaHashMap`（`1M` `u64` 下 `35.65 B/entry`）、`GoldHashMap`（`benches/competitor_bench.rs`）
- **高级 Trie**：双数组（`DoubleArrayTrie`，`12.0 B/state`；`DoubleArrayTrieMap<u32>`，`12.0–16.0 B/state`）、LOUDS、Critical-Bit（BMI2）、Patricia Trie、惰性前缀/模糊迭代器、`CsppTrie`（压缩稀疏并行 Patricia，共享前缀键 `9.60 B/key`）、`ConcurrentCsppTrie`（多写多读，基于 epoch 的回收）（`tests/memory_footprint_e6.rs`）
- **压缩与整数编码**：PA-Zip、Huffman O0/O1/O2、FSE、rANS（64 位 X1/X2/X4/X8）、`StreamVByte`（SSSE3）、`GroupVarint`（SSSE3）、ZSTD 与 LZ4 集成（`benches/competitor_bench.rs`）
- **C FFI 支持**：完整的 C11/C++17 API（`--features ffi`，`tests/ffi_c_smoke.rs`）
<!-- END GENERATED KEY FEATURES (scripts/perf_table.py) -->

## 快速开始

```toml
[dependencies]
zipora = "5.0.0"

# 启用 C FFI 绑定
zipora = { version = "5.0.0", features = ["ffi"] }

# AVX-512
zipora = { version = "5.0.0", features = ["avx512"] }
```

### 基本用法

```rust
use zipora::*;

// 高性能向量
let mut vec = FastVec::new();
vec.push(42).unwrap();

// 零拷贝字符串 + SIMD 哈希
let s = FastStr::from_string("hello world");
println!("Hash: {:x}", s.hash_fast());

// 智能 rank/select，自动选择最优实现
let mut bv = BitVector::new();
for i in 0..1000 { bv.push(i % 7 == 0).unwrap(); }
let adaptive_rs = AdaptiveRankSelect::new(bv).unwrap();
let rank = adaptive_rs.rank1(500);

// 统一 Trie —— 基于策略模式的配置
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, Trie};

let mut trie: ZiporaTrie = ZiporaTrie::new();
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));

// 统一哈希表 —— 基于策略模式的配置
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};

let mut map: ZiporaHashMap<&str, &str> = ZiporaHashMap::new().unwrap();
map.insert("key", "value").unwrap();

// 带压缩的 Blob 存储
use zipora::blob_store::{ZipOffsetBlobStoreBuilder, ZipOffsetBlobStoreConfig};

let config = ZipOffsetBlobStoreConfig::performance_optimized();
let mut builder = ZipOffsetBlobStoreBuilder::with_config(config).unwrap();
builder.add_record(b"Compressed data").unwrap();
let store = builder.finish().unwrap();

// 熵编码
let encoder = HuffmanEncoder::new(b"sample data").unwrap();
let compressed = encoder.encode(b"sample data").unwrap();

// 字符串工具
use zipora::string::{join_str, hex_encode, hex_decode, words, decimal_strcmp};
let joined = join_str(", ", &["hello", "world"]);
assert_eq!(joined, "hello, world");
```

## 文档

### 核心组件
- **[容器](docs/CONTAINERS.md)** - 专用容器（FastVec、ValVec32、IntVec、LruMap 等）
- **[哈希表](docs/HASH_MAPS.md)** - ZiporaHashMap、GoldHashMap，基于策略模式配置
- **[Blob 存储](docs/BLOB_STORAGE.md)** - 专用 Blob 存储变体，支持 Trie 索引和压缩
- **[内存管理](docs/MEMORY_MANAGEMENT.md)** - SecureMemoryPool、MmapVec、五级内存池

### 算法与处理
- **[算法](docs/ALGORITHMS.md)** - 基数排序、后缀数组、集合运算、Cache-oblivious 算法、SIMD 位计数、SIMD 跳跃搜索、SIMD 块过滤
- **[压缩](docs/COMPRESSION.md)** - Huffman、FSE、rANS、字典压缩（PA-Zip）、StreamVByte、自适应选择
- **[字符串处理](docs/STRING_PROCESSING.md)** - SIMD 字符串搜索、拼接、数值比较、词边界、十六进制

### 系统架构
- **[并发](docs/CONCURRENCY.md)** - 流水线处理、工作窃取、并行 Trie 构建
- **[错误处理](docs/ERROR_HANDLING.md)** - ZiporaError/Result 模式、错误分类、边界检查
- **[配置](docs/CONFIGURATION.md)** - 丰富的配置 API、预设、校验
- **[SIMD 框架](docs/SIMD.md)** - 6 级 SIMD，支持 AVX2/BMI2/POPCNT

### 集成
- **[I/O 与序列化](docs/IO_SERIALIZATION.md)** - 流处理、字节序处理、VarInt 编码
- **[C FFI](docs/FFI.md)** - C 互操作 API

### 指南
- **[搜索引擎指南](docs/SEARCH_ENGINE_GUIDE.md)** - 使用 Zipora 构建端到端搜索引擎架构
- **[性能基准测试](docs/PERFORMANCE.md)** - 所有组件的验证基准测试结果

### 参考
- **[二进制格式规范](docs/format_spec.md)** - 持久化容器与 Blob 存储的字节布局规范
- **[移植状态](docs/PORTING_STATUS.md)** - 功能实现状态

## 功能特性

| 特性 | 默认启用 | 说明 |
|------|----------|------|
| `simd` | 是 | SIMD 优化（AVX2、SSE4.2） |
| `mmap` | 是 | 内存映射文件支持 |
| `zstd` | 是 | ZSTD 压缩 |
| `serde` | 是 | 序列化支持（serde、serde_json、bincode） |
| `lz4` | 是 | LZ4 压缩 |
| `async` | 是 | 异步运行时（tokio），用于并发、流水线、实时压缩 |
| `ffi` | 否 | C FFI 绑定 |
| `avx512` | 是 | AVX-512 优化 (自 1.89 起稳定) |

## 构建与测试

```bash
# 构建（默认特性）
cargo build --release

# 构建所有特性（含 FFI）
cargo build --release --all-features

# 测试
cargo test --lib

# 完整检查（所有特性组合，debug + release）
make sanity

# 基准测试（仅 release）
cargo bench

# 代码检查
cargo clippy --all-targets --all-features -- -D warnings
```

## 性能验证

详见 **[Stage 14 核心基准测试](docs/review/e1.md)**、**[Stage 15 竞品对比矩阵](docs/review/e2_e3.md)**、**[Stage 16 优化台账](docs/review/e4.md)** 以及 **[性能基准测试](docs/PERFORMANCE.md)**。

<!-- BEGIN GENERATED PERF TABLE (scripts/perf_table.py) -->
Measured on reference host `bindiego` (AMD EPYC 7B13, 64 logical cores, pinned core 23, `THP-off` via `PR_SET_THP_DISABLE`, `-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C codegen-units=1 -C llvm-args=-align-loops=32`).
Across all **197** head-to-head comparisons in [`docs/review/e2_e3.md`](docs/review/e2_e3.md) and [`docs/review/e4.md`](docs/review/e4.md), Zipora is **`LEADING` on 68/197** (primary within `-3.0%` and secondary space `<= +10.0%`) and **faster or tied on primary on 107/197 (54.3%)**.

### Head-to-Head Competitor Highlights (`benches/baselines/bindiego/competitor_bench.json` & `e2e_search.json`)

| Family | Zipora Operation | Zipora (`min`) | Space / Secondary | Competitor | Competitor (`min`) | Speedup |
|---|---|---:|---:|---|---:|---:|
| Rank/Select (`1 Gbit`) | `RankSelectInterleaved256::rank1` | `35.23 ns/op` | `131.25% of raw` | `sucds::Rank9Sel` | `42.92 ns/op` | **`1.22x`** |
| Rank/Select (`1 Gbit`) | `RankSelectSE256::select1` | `149.64 ns/op` | `137.50% of raw` | `sucds::Rank9Sel` | `203.58 ns/op` | **`1.36x`** |
| Elias-Fano (`100k`) | `EliasFano::next_geq` (uniform) | `22.17 ns/op` | `8.62 bits/elem` | `sucds::EliasFano` | `47.69 ns/op` | **`2.15x`** |
| Elias-Fano (`100k`) | `OptimalPartitionedEliasFano::next_geq` | `26.55 ns/op` | `5.42 bits/elem` | `vers-vecs::EliasFanoVec` | `61.44 ns/op` | **`2.31x`** |
| Elias-Fano (`100k`) | `ClusteredEliasFano::get` (dense) | `3.39 ns/op` | `2.51 bits/elem` | `sucds::EliasFano` | `8.03 ns/op` | **`2.37x`** |
| Tries (`1M` keys) | `DoubleArrayTrie::contains` (shared prefix) | `247.11 ns/op` | `25.60 B/key` | `qp-trie::Trie` | `521.72 ns/op` | **`2.11x`** |
| Tries (`1M` keys) | `CsppTrie::contains` (shared prefix) | `210.57 ns/op` | `9.60 B/key` | `radix_trie::Trie` | `1027.99 ns/op` | **`4.88x`** |
| Hash Maps (`1M` keys) | `ZiporaHashMap<[u8; 16], u64>::get` | `83.74 ns/op` | `52.43 B/entry` | `hashbrown::HashMap` (`ahash`) | `191.06 ns/op` | **`2.28x`** |
| Hash Maps (`1M` keys) | `ZiporaHashMap<u64, u64>::get` | `66.96 ns/op` | `35.65 B/entry` | `std::collections::HashMap` | `85.55 ns/op` | **`1.28x`** |
| Entropy (`64 KiB`) | `HuffmanDecoder::decode` (O0) | `3.50 ns/byte` | `4.01 bits/B` | `huffman-compress` | `9.43 ns/byte` | **`2.69x`** |
| Entropy (`64 KiB`) | `Rans64Decoder<ParallelX4>::decode` | `2.50 ns/byte` | `4.05 bits/B` | `constriction` ANS | `6.15 ns/byte` | **`2.47x`** |
| Integer Codecs (`100k`) | `StreamVByte::encode_deltas` (SSSE3) | `0.48 ns/int` | `10.00 bits/int` | `stream-vbyte` SIMD | `1.29 ns/int` | **`2.68x`** |
| Integer Codecs (`100k`) | `StreamVByte::decode_into` (SSSE3) | `0.19 ns/int` | `10.00 bits/int` | `stream-vbyte` SIMD | `0.25 ns/int` | **`1.36x`** |
| Radix Sort (`10M`) | `RadixSort::sort_u32` (hybrid MSB+LSD) | `5.68 ns/elem` | `8.00 B/elem` | `voracious_radix_sort` | `10.11 ns/elem` | **`1.78x`** |
| String Search (`64 KiB`) | `simd_strchr` (128B AVX2) | `644.84 ns/call` | `0 B aux` | `memchr::memchr` | `636.33 ns/call` | **`0.99x`** |
| String Search (`64 KiB`) | `simd_strstr` (128B AVX2) | `649.14 ns/call` | `0 B aux` | `memchr::memmem` | `1921.17 ns/call` | **`2.96x`** |
| Systemic Search (`100k` docs) | `Zipora BMW` top-10 (`p50 = 21.14 us`) | `19,917.6 QPS` | `1.219 B/posting` (postings) | `tantivy` v0.26.2 (`p50 = 55.04 us`) | `12,036.3 QPS` | **`1.65x`** (`2.34x` vs uncompressed) |

### Additional Core Suite Baselines (`benches/baselines/bindiego/core_bench.json`)

- **BM25 & Fieldnorm Scoring**: `Bm25BatchScorer` AVX2 batch `0.4784 ns/doc`, `FieldnormEncoder::build_score_table` `0.1699 ns/cell` (`1.0 B/doc` fieldnorms).
- **Memory Pools**: `LockFreeMemoryPool` 64B alloc+free `9.106 ns/op` (1T) / `9.110 ns/op` (4T).
- **Rank9 (`1 Gbit`)**: `Rank9::rank1` `43.23 ns/op` at `25.00%` space overhead.
- **Suffix Array (`10M` bytes, SA-IS)**: construction `75.28 ns/byte` (`9.05 B/byte`), `search_range` query `1330.43 ns/op`.
<!-- END GENERATED PERF TABLE (scripts/perf_table.py) -->

## 依赖

精简的依赖设计：
- **核心**：`bytemuck`、`thiserror`、`log`、`ahash`、`rayon`、`libc`、`raw-cpuid`、`crossbeam-epoch`、`thread_local`、`fastrand`、`bitflags`、`base64`
- **默认**：`memmap2`（mmap）、`zstd`、`lz4_flex`、`serde`/`serde_json`/`bincode`、`tokio`（async）
- **可选**：`cbindgen`（ffi）
- **已移除**：`crossbeam-utils`、`parking_lot`、`uuid`、`num_cpus`、`async-trait`、`futures`、`once_cell`、`pkg-config`（全部替换为标准库或删除）

## 使用 Zipora 构建搜索引擎

详见 **[搜索引擎指南](docs/SEARCH_ENGINE_GUIDE.md)**，涵盖完整的 11 个组件代码示例：词典（DoubleArrayTrie + 惰性前缀/模糊迭代器）、倒排列表（HybridPostingList + Elias-Fano 游标）、SIMD 查询原语（simd_gallop_to、simd_block_filter、advance_to_index）、BM25 评分（FieldnormEncoder + Bm25BatchScorer）、文档存储、压缩、多线程索引构建及组件选型指南。

## 许可证

The Bindiego License (BDL), Version 1.0 - 详见 [LICENSE](LICENSE)。
