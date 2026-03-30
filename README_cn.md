# Zipora

[English](README.md)

[![Build Status](https://github.com/infinilabs/zipora/workflows/CI/badge.svg)](https://github.com/infinilabs/zipora/actions)
[![License](https://img.shields.io/badge/license-BDL--1.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.88+-orange.svg)](https://www.rust-lang.org)

高性能 Rust 数据结构与压缩算法库，提供内存安全保证。

## 核心特性

- **高性能**：零拷贝操作、SIMD 优化（AVX2、AVX-512）、缓存友好的内存布局
- **内存安全**：99.8% 的 unsafe 代码块文档覆盖率，所有生产环境中的 unsafe 代码块均附有 `// SAFETY:` 注释
- **安全内存管理**：线程安全的生产级内存池，支持 RAII
- **Blob 存储**：8 种专用存储引擎，支持 Trie 索引和压缩
- **简洁数据结构**：12 种 rank/select 变体
- **专用容器**：13+ 种容器（VecTrbSet/Map、MinimalSso、SortedUintVec、LruMap 等）
- **哈希表**：黄金比例优化、字符串优化、缓存优化的多种实现
- **高级 Trie**：双数组（DoubleArrayTrie）、LOUDS、Critical-Bit（BMI2）、Patricia Trie + rank/select、NestTrieDawg
- **压缩算法**：PA-Zip、Huffman O0/O1/O2、FSE、rANS、ZSTD 集成
- **C FFI 支持**：完整的 C API，便于从 C++ 迁移（`--features ffi`）

## 快速开始

```toml
[dependencies]
zipora = "3.0.1"

# 启用 C FFI 绑定
zipora = { version = "3.0.1", features = ["ffi"] }

# AVX-512（仅 nightly）
zipora = { version = "3.0.1", features = ["avx512"] }
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
- **[算法](docs/ALGORITHMS.md)** - 基数排序、后缀数组、集合运算、Cache-oblivious 算法
- **[压缩](docs/COMPRESSION.md)** - PA-Zip、Huffman、FSE、rANS、实时压缩
- **[字符串处理](docs/STRING_PROCESSING.md)** - SIMD 字符串操作、模式匹配

### 系统架构
- **[并发](docs/CONCURRENCY.md)** - 流水线处理、工作窃取、并行 Trie 构建
- **[错误处理](docs/ERROR_HANDLING.md)** - 错误分类、自动恢复策略
- **[配置](docs/CONFIGURATION.md)** - 丰富的配置 API、预设、校验
- **[SIMD 框架](docs/SIMD.md)** - 6 级 SIMD，支持 AVX2/BMI2/POPCNT

### 集成
- **[I/O 与序列化](docs/IO_SERIALIZATION.md)** - 流处理、字节序处理、VarInt 编码
- **[C FFI](docs/FFI.md)** - 用于从 C++ 迁移的 C API

### 性能报告
- **[性能对比 C++](docs/PERF_VS_CPP.md)** - 基准测试对比
- **[移植状态](docs/PORTING_STATUS.md)** - 功能对等状态

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

> **测试环境**：AMD EPYC 7B13（Zen 3），64 vCPU，117 GB 内存，AVX2/BMI2/POPCNT，rustc 1.91.1，Linux 6.17。
> 不同硬件上的结果可能有所不同 —— Intel 的 BMI2 实现方式不同（原生 vs 微码），ARM 不支持 x86 SIMD 指令。
> 运行 `cargo bench` 在你自己的硬件上复现。

### 简洁数据结构

| 操作 | Zipora | 基准 | 加速比 |
|------|--------|------|--------|
| Rank1 查询（100K 位） | 192 ns | — | ~5.2 Gops/s |
| Select1 查询（100K 位） | 5.4 ms / 100K 次查询 | — | ~18.5 Mops/s |
| 批量 rank（SIMD，50K） | 8.4 µs | 84.1 µs（逐个） | **10x** |
| 批量位运算（SIMD，50K） | 3.1 µs | 128.4 µs（逐个） | **41x** |
| 范围设置（SIMD，50K） | 3.2 µs | 17.9 µs（逐个） | **5.6x** |

### 容器 vs 标准库

| 操作 | Zipora | std | 比率 |
|------|--------|-----|------|
| ValVec32 push（100K） | 119 µs | 120 µs | 1.0x |
| ValVec32 随机访问（100K） | 706 ns | 729 ns | **0.97x** |
| ValVec32 迭代（10K） | 778 ns | 783 ns | 1.0x |
| ValVec32 批量扩展（100K） | 21.8 µs | 28.7 µs | **0.76x** |
| SmallMap 插入+查找（8 键） | 444 ns | 805 ns（HashMap） | **1.8x** |
| SmallMap 查找密集型 | 36.9 µs | 141.7 µs（HashMap） | **3.8x** |
| CircularQueue 入队+出队（100K） | 326 µs | 381 µs（VecDeque） | **0.86x** |
| FixedStr16Vec push（100K） | 755 µs | 5,906 µs（Vec\<String\>） | **7.8x** |
| SortableStrVec 排序（5K） | 390 µs | 448 µs（Vec\<String\>） | **1.15x** |

### 熵编码（65KB 输入）

| 算法 | 熵值 0.5 | 熵值 2.0 | 熵值 6.0 |
|------|----------|----------|----------|
| Huffman O0 | 1,124 µs | 1,235 µs | 1,720 µs |
| Huffman O1（x1 流） | 188 µs | 173 µs | 188 µs |
| rANS64 | 405 µs | 351 µs | 426 µs |

### 缓存（LRU vs HashMap）

| 操作 | LruMap | HashMap | 说明 |
|------|--------|---------|------|
| 热数据读取（容量=64，10K 次） | 5.7 µs | 152 µs | **26x** 更快（热集合在缓存中） |
| 热数据读取（容量=1024，10K 次） | 94.6 µs | 152 µs | **1.6x** 更快 |
| 插入（容量=64，10K 次） | 1,897 µs | 1,177 µs | 0.62x（淘汰开销） |

## 依赖

精简的依赖设计：
- **核心**：`bytemuck`、`thiserror`、`log`、`ahash`、`rayon`、`libc`、`once_cell`、`raw-cpuid`
- **默认**：`memmap2`（mmap）、`zstd`、`lz4_flex`、`serde`/`serde_json`/`bincode`、`tokio`（async）
- **可选**：`cbindgen`（ffi）
- **已移除**：`crossbeam-utils`、`parking_lot`、`uuid`、`num_cpus`、`async-trait`、`futures`（全部替换为标准库或删除）

## 使用 Zipora 构建搜索引擎

Zipora 提供了构建高性能搜索引擎的核心组件：简洁倒排列表、压缩文档存储、基于 Trie 的词典、SIMD 加速的查询处理，以及多线程索引构建流水线。

### 架构概览

```
 文档                          查询
  |                              |
  v                              v
 [分词器]                    [查询解析器]
  |                              |
  v                              v
 [词典]          --->        [词项查找]          ZiporaTrie / DoubleArrayTrie
  |                              |
  v                              v
 [倒排索引]      --->        [倒排列表]          UintVecMin0 / SortedUintVec / BitVector
  |                              |
  v                              v
 [文档存储]      --->        [文档检索]          DictZipBlobStore / MixedLenBlobStore
  |                              |
  v                              v
 [压缩]                      [排序打分]          HuffmanEncoder / Rans64Encoder
```

### 1. 词典（基于 Trie）

使用 `DoubleArrayTrie`（双数组 Trie）实现最高性能 —— 每状态 8 字节，每字节 O(1) 转移。对大型词汇表，比 `HashMap<String, u32>` 节省 3-5 倍内存，且查找更快。

```rust
use zipora::DoubleArrayTrie;

// 索引时构建词典
let mut dict = DoubleArrayTrie::new();

for term in terms.iter() {
    dict.insert(term.as_bytes()).unwrap();
}

// 查询时查找：O(|key|)，每字节 O(1) 转移
assert!(dict.contains(b"search"));

// 键值存储（词项 → 词项 ID）
use zipora::DoubleArrayTrieMap;
let mut term_ids: DoubleArrayTrieMap<u32> = DoubleArrayTrieMap::new();
for (term_id, term) in terms.iter().enumerate() {
    term_ids.insert(term.as_bytes(), term_id as u32).unwrap();
}
let id = term_ids.get(b"search");
```

需要其他 Trie 策略（LOUDS、Patricia、CritBit），可通过显式配置使用 `ZiporaTrie`。需要前缀压缩的词项存储，可使用 `NestLoudsTrieBlobStore`。

### 2. 倒排索引（倒排列表）

根据倒排列表的特征选择合适的容器：

```rust
use zipora::containers::{UintVecMin0, ZipIntVec};
use zipora::blob_store::SortedUintVec;
use zipora::BitVector;

// 方案 A：UintVecMin0 —— 变宽紧凑整数（每值 2-58 位）
// 适用于：中等长度的倒排列表，文档 ID 有界
let mut postings = UintVecMin0::new();
for doc_id in matching_docs {
    postings.push(doc_id);
}
// 访问：postings.get(i) —— O(1)，缓存友好的顺序布局

// 方案 B：SortedUintVec —— 排序文档 ID 的差值 + 分块压缩
// 适用于：长倒排列表（相比原始 u32 节省 60-80% 空间）

// 方案 C：BitVector + RankSelect —— 位图表示
// 适用于：高频词项（>10% 文档），布尔查询
let mut bitmap = BitVector::new();
for i in 0..num_docs {
    bitmap.push(doc_ids.contains(&i)).unwrap();
}
```

### 3. 布尔查询处理（集合运算）

SIMD 加速的倒排列表集合运算 —— 位运算最高可达 **41 倍**加速。

```rust
use zipora::algorithms::set_ops::{
    multiset_intersection,   // AND 查询
    multiset_union,          // OR 查询
    multiset_difference,     // NOT 查询
    multiset_fast_intersection, // 自适应：根据大小比率自动选择算法
};

// AND 查询："rust" AND "search"
let result = multiset_intersection(&postings_rust, &postings_search);

// 大小悬殊时（一个稀有词，一个常见词），使用自适应交集
// 根据 |A|/|B| 比率自动选择线性归并或二分查找
let result = multiset_fast_intersection(&rare_term, &common_term);

// 在 rank/select 位向量上的批量位运算（SIMD 加速 41 倍）
use zipora::AdaptiveRankSelect;
let rs = AdaptiveRankSelect::new(bitmap).unwrap();
let rank = rs.rank1(doc_id);   // 统计该 ID 之前的文档数 —— O(1)
let pos = rs.select1(rank);    // 查找第 N 个匹配文档 —— O(log n)
```

### 4. 文档存储（压缩 Blob 存储）

使用字典压缩（PA-Zip）存储和检索文档：

```rust
use zipora::DictZipBlobStore;
use zipora::blob_store::{MixedLenBlobStore, PlainBlobStore, BlobStore};

// DictZipBlobStore：最适合相似文档的压缩（网页、日志）
// 从训练数据学习共享字典，然后压缩每条记录
let store = DictZipBlobStore::builder()
    .build_from_records(&documents)
    .unwrap();

// 检索：通过 mmap 实现零拷贝访问
let doc = store.get(doc_id).unwrap();

// MixedLenBlobStore：适合混合定长/变长记录
// 根据记录大小分布自动选择存储策略

// PlainBlobStore：无压缩，热数据最快检索
```

### 5. 熵编码（倒排列表压缩）

使用 Huffman 或 rANS 压缩倒排列表的差值：

```rust
use zipora::HuffmanEncoder;
use zipora::Rans64Encoder;

// Huffman O0：简单、快速编码（65KB 仅需 1.1 µs）
let encoder = HuffmanEncoder::new(&training_data).unwrap();
let compressed = encoder.encode(&delta_encoded_postings).unwrap();

// Huffman O1：上下文感知，对结构化数据压缩效果更好
// 对分布偏斜的倒排列表差值特别有效

// rANS：最高压缩比，速度略慢
let rans = Rans64Encoder::new(&training_data).unwrap();
let compressed = rans.encode(&data).unwrap();
```

### 6. 多线程索引构建

使用 rayon 和 zipora 的流水线处理实现并行索引构建：

```rust
use rayon::prelude::*;
use zipora::algorithms::MultiWayMerge;

// 并行文档处理：每个线程构建一个段
let segments: Vec<_> = document_batches
    .par_iter()
    .map(|batch| {
        let mut segment_index = SegmentIndex::new();
        for doc in batch {
            let terms = tokenize(doc);
            for term in terms {
                segment_index.add(term, doc.id);
            }
        }
        segment_index
    })
    .collect();

// 使用 K 路归并合并段（败者树）
use zipora::EnhancedLoserTree;
// EnhancedLoserTree 每元素 O(log k)，用于 K 路归并
// 非常适合合并并行索引段的排序倒排列表
```

异步流水线处理（需要 `async` 特性）：

```rust
use zipora::Pipeline;
// 流水线阶段：解析 → 分词 → 索引 → 压缩 → 刷盘
// 每个阶段并发执行，支持工作窃取负载均衡
```

### 7. 内存映射索引文件

直接从磁盘提供大型索引服务，无需加载到内存：

```rust
use zipora::memory::MmapVec;

// 内存映射索引文件 —— 由操作系统管理分页
let index: MmapVec<u32> = MmapVec::open("postings.idx").unwrap();

// 随机访问由页缓存支撑
let doc_id = index[position];

// Blob 存储也支持 mmap 后端
// DictZipBlobStore 和 NestLoudsTrieBlobStore 原生支持 mmap
```

### 8. 查询结果缓存

LRU 缓存用于频繁访问的倒排列表 —— 热数据检索比 HashMap **快 26 倍**：

```rust
use zipora::containers::specialized::LruMap;

// 缓存热倒排列表
let mut cache: LruMap<String, Vec<u32>> = LruMap::new(1024);

fn get_postings(term: &str, cache: &mut LruMap<String, Vec<u32>>) -> Vec<u32> {
    if let Some(cached) = cache.get(term) {
        return cached.clone(); // 热键查找比 HashMap 快 26 倍
    }
    let postings = load_from_disk(term);
    cache.insert(term.to_string(), postings.clone());
    postings
}
```

### 9. 分词相关的字符串处理

```rust
use zipora::SortableStrVec;
use zipora::string::{decimal_strcmp, words};

// Arena 式字符串存储：push 100K 个字符串比 Vec<String> 快 7.8 倍
let mut terms = SortableStrVec::new();
for token in document.split_whitespace() {
    terms.push(token);
}
terms.sort(); // 原地排序，比 Vec<String>::sort 快 1.15 倍

// 小型查找表（字段名、停用词），SmallMap 快 3.8 倍
use zipora::SmallMap;
let mut stop_words = SmallMap::new();
stop_words.insert("the", true);
stop_words.insert("and", true);
```

### 组件选型指南

| 搜索引擎组件 | Zipora 类型 | 适用场景 |
|-------------|-------------|----------|
| 词典 | `DoubleArrayTrie` | 默认选择，8 字节/状态，O(1) 转移 |
| 词典（其他策略） | `ZiporaTrie` | 通过配置选择 LOUDS/Patricia/CritBit |
| 短倒排列表 | `UintVecMin0` | 变宽，<1M 文档 ID |
| 长倒排列表 | `SortedUintVec` | 差值压缩的排序 ID |
| 布尔倒排列表 | `BitVector` + `AdaptiveRankSelect` | 高频词，位运算 |
| AND/OR/NOT 查询 | `set_ops::multiset_*` | 排序倒排列表交集 |
| 批量位运算查询 | SIMD rank/select | 比标量快 10-41 倍 |
| 文档存储 | `DictZipBlobStore` | 相似文档最佳压缩 |
| 文档存储（快速） | `PlainBlobStore` | 无压缩，最快检索 |
| 倒排列表压缩 | `HuffmanEncoder` | 快速编解码 |
| 倒排列表压缩 | `Rans64Encoder` | 最高压缩比 |
| 查询缓存 | `LruMap` | 热数据访问快 26 倍 |
| 小型查找 | `SmallMap` | 8 键以内快 3.8 倍 |
| 字符串存储 | `SortableStrVec` / `FixedStr16Vec` | Arena 式，比 Vec\<String\> 快 7.8 倍 |
| 索引文件 | `MmapVec` | 磁盘后端，OS 管理分页 |
| 段合并 | `MultiWayMerge` / `EnhancedLoserTree` | 排序列表的 K 路归并 |
| 并行索引 | `rayon` + `Pipeline` | 多线程段构建 |

## 许可证

Business Source License 1.0 - 详见 [LICENSE](LICENSE)。
