# Zipora

[English](README.md)

[![Build Status](https://github.com/infinilabs/zipora/workflows/CI/badge.svg)](https://github.com/infinilabs/zipora/actions)
[![License](https://img.shields.io/badge/license-BDL--1.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.88+-orange.svg)](https://www.rust-lang.org)

高性能 Rust 数据结构与压缩算法库，提供内存安全保证。

## 核心特性

- **高性能**：零拷贝操作、SIMD 优化（AVX2、AVX-512）、缓存友好的内存布局、Block-Max WAND SIMD 游标原语
- **内存安全**：99.8% 的 unsafe 代码块文档覆盖率，所有生产环境中的 unsafe 代码块均附有 `// SAFETY:` 注释
- **安全内存管理**：线程安全的生产级内存池，支持 RAII
- **Blob 存储**：8 种专用存储引擎，支持 Trie 索引和压缩
- **简洁数据结构**：12 种 rank/select 变体、Rank9（Vigna 2008）、Elias-Fano / Partitioned / DP-Optimal Partitioned Elias-Fano（支持游标 `advance_to_index`）、HybridPostingList（自动选择编码策略）、AMD 安全 PDEP（`has_fast_bmi2` 检测）
- **BM25 评分**：FieldnormEncoder（Lucene SmallFloat，1 字节字段长度） + Bm25BatchScorer（AVX2 SIMD 批量评分、预取）
- **专用容器**：13+ 种容器（VecTrbSet/Map、MinimalSso、SortedUintVec、LruMap 等）
- **哈希表**：黄金比例优化、字符串优化、缓存优化的多种实现
- **高级 Trie**：双数组（DoubleArrayTrie，XOR 转移）、LOUDS、Critical-Bit（BMI2）、Patricia Trie + rank/select、NestTrieDawg、惰性前缀/模糊迭代器、CsppTrie（压缩稀疏并行 Patricia，10 种节点编码，10.7 字节/键）、ConcurrentCsppTrie（多写多读，基于 epoch 的回收，线程本地分配）
- **压缩算法**：PA-Zip、Huffman O0/O1/O2、FSE、rANS、ZSTD 集成
- **C FFI 支持**：完整的 C API（`--features ffi`）

## 快速开始

```toml
[dependencies]
zipora = "3.1.6"

# 启用 C FFI 绑定
zipora = { version = "3.1.6", features = ["ffi"] }

# AVX-512（仅 nightly）
zipora = { version = "3.1.6", features = ["avx512"] }
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

let mut trie = ZiporaTrie::new();
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));

// 统一哈希表 —— 基于策略模式的配置
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};

let mut map = ZiporaHashMap::new();
map.insert("key", "value").unwrap();

// 带压缩的 Blob 存储
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
- **[Blob 存储](docs/BLOB_STORAGE.md)** - 8 种 Blob 存储变体，支持 Trie 索引和压缩
- **[内存管理](docs/MEMORY_MANAGEMENT.md)** - SecureMemoryPool、MmapVec、五级内存池

### 算法与处理
- **[算法](docs/ALGORITHMS.md)** - 基数排序、后缀数组、集合运算、Cache-oblivious 算法、SIMD 位计数、SIMD 跳跃搜索、SIMD 块过滤
- **[压缩](docs/COMPRESSION.md)** - PA-Zip、Huffman、FSE、rANS、实时压缩
- **[字符串处理](docs/STRING_PROCESSING.md)** - SIMD 字符串操作、模式匹配

### 系统架构
- **[并发](docs/CONCURRENCY.md)** - 流水线处理、工作窃取、并行 Trie 构建
- **[错误处理](docs/ERROR_HANDLING.md)** - 错误分类、自动恢复策略
- **[配置](docs/CONFIGURATION.md)** - 丰富的配置 API、预设、校验
- **[SIMD 框架](docs/SIMD.md)** - 6 级 SIMD，支持 AVX2/BMI2/POPCNT

### 集成
- **[I/O 与序列化](docs/IO_SERIALIZATION.md)** - 流处理、字节序处理、VarInt 编码
- **[C FFI](docs/FFI.md)** - C 互操作 API

### 指南
- **[搜索引擎指南](docs/SEARCH_ENGINE_GUIDE.md)** - 使用 Zipora 构建端到端搜索引擎架构
- **[性能基准测试](docs/PERFORMANCE.md)** - 所有组件的验证基准测试结果

### 参考
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
| `avx512` | 否 | AVX-512（仅 nightly） |
| `nightly` | 否 | Nightly 专属优化 |

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

详见 **[性能基准测试](docs/PERFORMANCE.md)**，涵盖所有组件的详细结果（Trie、BitVector、位计数、rank/select、容器、熵编码、LRU 缓存、BM25 评分）。

**亮点**：DoubleArrayTrie 20.6 ns/次查找，CsppTrie 690 万次插入/秒 + 800 万次查找/秒（10.7 字节/键），ConcurrentCsppTrie 1000 万+ 键/秒（16 线程），SIMD 位计数 5.2 Gwords/s，批量位运算快 41 倍，BM25 SIMD 快 13.5 倍，LRU 热读取快 26 倍。

## 依赖

精简的依赖设计：
- **核心**：`bytemuck`、`thiserror`、`log`、`ahash`、`rayon`、`libc`、`once_cell`、`raw-cpuid`
- **默认**：`memmap2`（mmap）、`zstd`、`lz4_flex`、`serde`/`serde_json`/`bincode`、`tokio`（async）
- **可选**：`cbindgen`（ffi）
- **已移除**：`crossbeam-utils`、`parking_lot`、`uuid`、`num_cpus`、`async-trait`、`futures`（全部替换为标准库或删除）

## 使用 Zipora 构建搜索引擎

详见 **[搜索引擎指南](docs/SEARCH_ENGINE_GUIDE.md)**，涵盖完整的 11 个组件代码示例：词典（DoubleArrayTrie + 惰性前缀/模糊迭代器）、倒排列表（HybridPostingList + Elias-Fano 游标）、SIMD 查询原语（simd_gallop_to、simd_block_filter、advance_to_index）、BM25 评分（FieldnormEncoder + Bm25BatchScorer）、文档存储、压缩、多线程索引构建及组件选型指南。

## 许可证

Business Source License 1.0 - 详见 [LICENSE](LICENSE)。
