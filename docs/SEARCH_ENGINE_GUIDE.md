# Building a Search Engine with Zipora

Zipora provides the core building blocks for high-performance search engines: succinct posting lists, compressed document storage, trie-based term dictionaries, SIMD-accelerated query processing, and multi-threaded indexing pipelines.

## Architecture Overview

```text
 Documents                    Query
     |                          |
     v                          v
 [Tokenizer]              [Query Parser]
     |                          |
     v                          v
 [Term Dictionary]  --->  [Term Lookup]        DoubleArrayTrie + iter_prefix/iter_fuzzy
     |                          |
     v                          v
 [Inverted Index]  --->  [Posting Lists]       HybridPostingList + advance_to_index
     |                          |
     v                          v
 [SIMD Query]       --->  [BMW / Top-K]        simd_gallop_to + simd_block_filter
     |                          |
     v                          v
 [Doc Lengths]      --->  [BM25 Scoring]       FieldnormEncoder + Bm25BatchScorer
     |                          |
     v                          v
 [Document Store]  --->  [Doc Retrieval]       DictZipBlobStore
     |                          |
     v                          v
 [Compression]            [Encoding]           StreamVByte
```

## 1. Term Dictionary (Trie-based)

Use `DoubleArrayTrie` (double-array trie with XOR transitions) for high-throughput lookups — compact per-state `base` + `check` + `tail` arrays with O(1) transitions per byte. Supports arbitrary binary keys including `\x00` bytes. See [PERFORMANCE.md](PERFORMANCE.md) for measured memory footprint and lookup latency against `fst`, `qp-trie`, and `radix_trie`.

```rust
use zipora::{DoubleArrayTrie, DoubleArrayTrieMap};

let terms = ["engine", "index", "rust", "search"];

// Build term dictionary during indexing
let mut dict = DoubleArrayTrie::new();
for term in terms.iter() {
    dict.insert(term.as_bytes()).unwrap();
}

// Query-time lookup: O(|key|) with O(1) per-byte transitions
assert!(dict.contains(b"search"));

// For key-value storage (term → term_id)
// DoubleArrayTrieMap<V> requires V: MapValue (configurable sentinel for zero-cost Option<V> elimination)
// Built-in impls: i32 (MIN), u32 (MAX), i64 (MIN), u64 (MAX), usize (MAX)
let mut term_ids: DoubleArrayTrieMap<u32> = DoubleArrayTrieMap::new();
for (term_id, term) in terms.iter().enumerate() {
    term_ids.insert(term.as_bytes(), term_id as u32).unwrap();
}
assert_eq!(term_ids.get(b"search"), Some(3));
```

`DoubleArrayTrieMap<V>` uses the `MapValue` trait with a compile-time sentinel constant instead of `Option<V>`, avoiding `Option<V>` discriminant padding in the value array for primitive types. The sentinel is monomorphized to a single `cmp` instruction — zero runtime cost.

### Lazy Prefix & Fuzzy Iterators

`DoubleArrayTrieMap` provides lazy iterators that yield results one at a time without collecting into a `Vec`, preventing memory spikes on broad queries:

```rust
use zipora::DoubleArrayTrieMap;

let mut trie: DoubleArrayTrieMap<u32> = DoubleArrayTrieMap::new();
trie.insert(b"search", 1).unwrap();
trie.insert(b"searching", 2).unwrap();

// Lazy prefix iteration — DFS with explicit stack, no Vec allocation
// Yields all key-value pairs whose key starts with the given prefix
for (key, term_id) in trie.iter_prefix(b"search") {
    println!("term: {:?}, id: {}", key, term_id);
}

// Early termination — only materializes what you consume
let top5: Vec<_> = trie.iter_prefix(b"s").take(5).collect();
assert_eq!(top5.len(), 2);

// Lazy fuzzy iteration — incremental Levenshtein with subtree pruning
// Yields all keys within edit distance `max_dist` of the query
for (key, term_id) in trie.iter_fuzzy(b"serch", 1) {
    // Finds "search" (1 insertion) and other terms within distance 1
    println!("fuzzy match: {:?}, id: {}", key, term_id);
}

// Zero-copy visitor counterparts — keys passed as borrowed &[u8], zero Vec allocations
trie.for_each_prefix(b"search", |key, term_id| {
    println!("zero-copy term: {:?}, id: {}", key, term_id);
});

trie.for_each_fuzzy(b"serch", 1, |key, term_id| {
    println!("zero-copy fuzzy: {:?}, id: {}", key, term_id);
});
```

For alternative trie strategies (LOUDS, Patricia, Critical-Bit), use `ZiporaTrie` with explicit config. For compressed term storage with prefix sharing, use `NestLoudsTrieBlobStore`.

## 2. Inverted Index (Posting Lists)

Choose the right container based on posting list characteristics:

```rust
use zipora::containers::UintVecMin0;
use zipora::BitVector;

let matching_docs = [1usize, 5, 10, 42];
let num_docs = 64usize;

// Option A: UintVecMin0 — variable-width packed integers (0-58 bits per value)
// Best for: medium-length posting lists with bounded doc IDs
let mut postings = UintVecMin0::new(matching_docs.len(), 42);
for (i, &doc_id) in matching_docs.iter().enumerate() {
    postings.set(i, doc_id);
}
assert_eq!(postings.get(2), 10);

// Option B: SortedUintVec — delta + block compression for sorted doc IDs
// Best for: long posting lists with monotonic offsets

// Option C: BitVector + RankSelect — bitmap representation
// Best for: high-frequency terms (>10% of docs), boolean queries
let mut bitmap = BitVector::new();
for i in 0..num_docs {
    bitmap.push(matching_docs.contains(&i)).unwrap();
}
assert_eq!(bitmap.len(), num_docs);
```

## 3. Boolean Query Processing (Set Operations)

Scalar merge-based multiset operations on sorted posting lists, with a caller-supplied comparator. (SIMD acceleration lives elsewhere: bulk bitwise operations on rank/select bitvectors, below, and the dedicated SIMD set kernels in `simd_set_intersect`/`simd_set_union`.)

```rust
use zipora::algorithms::set_ops::{
    multiset_intersection,   // AND queries
    multiset_fast_intersection, // adaptive: picks best algo by size ratio
};
use zipora::{AdaptiveRankSelect, BitVector, RankSelectOps};

let postings_rust = [1u32, 3, 5, 10];
let postings_search = [3u32, 5, 8, 10];

// AND query: "rust" AND "search"
let result = multiset_intersection(&postings_rust, &postings_search, |a, b| a.cmp(b));
assert_eq!(result, vec![3, 5, 10]);

// For skewed sizes (one term rare, one common), use adaptive intersection.
// Switches from linear merge to binary search when |A| * threshold < |B|
let rare_term = [5u32];
let common_term: Vec<u32> = (0..100).collect();
let result = multiset_fast_intersection(&rare_term, &common_term, |a, b| a.cmp(b), 32);
assert_eq!(result, vec![5]);

// Rank/select on bitvectors
let mut bitmap = BitVector::new();
for i in 0..64 {
    bitmap.push(i % 2 == 0).unwrap();
}
let rs = AdaptiveRankSelect::new(bitmap).unwrap();
let rank = rs.rank1(10);              // count docs before this ID — O(1)
let pos = rs.select1(rank).unwrap();  // find N-th matching doc — O(log n)
assert_eq!(pos, 10);
```

## 4. Document Storage (Compressed Blob Stores)

Store and retrieve documents with dictionary compression (PA-Zip):

```rust
use zipora::compression::dict_zip::DictZipBlobStoreBuilder;
use zipora::blob_store::BlobStore;

let documents: [&[u8]; 3] = [
    b"zipora search engine posting list compression",
    b"zipora search engine succinct data structures",
    b"zipora search engine double array trie lookup",
];

// DictZipBlobStore: best compression for similar documents (web pages, logs)
// Learns a shared dictionary from training samples, then compresses each record
let mut builder = DictZipBlobStoreBuilder::new().unwrap();
for doc in &documents {
    builder.add_training_sample(doc).unwrap();
}
let mut store = builder.finish().unwrap();

// Store: each record is compressed against the trained dictionary
let doc_id = store.put(documents[0]).unwrap();

// Retrieve
let doc = store.get(doc_id).unwrap();
assert_eq!(doc.as_slice(), documents[0]);

// MixedLenBlobStore: optimal for mixed fixed/variable-length records
// Automatically selects storage strategy based on record size distribution

// PlainBlobStore: uncompressed, fastest retrieval for hot data
```

## 5. Entropy Coding (Posting List Compression)

Compress posting list deltas with Huffman or rANS:

```rust
use zipora::HuffmanEncoder;
use zipora::Rans64Encoder;
use zipora::entropy::ParallelX1;

let training_data = b"aaaaabbbbcccdde";
let delta_encoded_postings = b"aaabbc";

// Huffman O0: simple, fast encoding
let encoder = HuffmanEncoder::new(training_data).unwrap();
let compressed = encoder.encode(delta_encoded_postings).unwrap();
assert!(!compressed.is_empty());

// Huffman O1: context-aware, better compression for structured data
// Particularly effective for posting list deltas with skewed distributions

// rANS: high compression ratio with 4-way/8-way interleaved decoding
// Rans64Encoder is built from a 256-entry byte-frequency table
let mut frequencies = [0u32; 256];
for &byte in training_data.iter() {
    frequencies[byte as usize] += 1;
}
let rans = Rans64Encoder::<ParallelX1>::new(&frequencies).unwrap();
let compressed = rans.encode(delta_encoded_postings).unwrap();
assert!(!compressed.is_empty());
```

## 6. Multi-threaded Indexing

Parallelize index building across threads and merge sorted posting lists with `MultiWayMerge` or `EnhancedLoserTree`:

```rust
use std::thread;
use zipora::algorithms::tournament_tree::{EnhancedLoserTree, LoserTreeConfig};

let document_batches: Vec<Vec<(u32, &'static str)>> = vec![
    vec![(1, "rust search"), (2, "succinct trie")],
    vec![(3, "rust index"), (4, "simd search")],
];

// Parallel document processing: each thread builds a sorted segment of doc IDs
let segments: Vec<Vec<u32>> = thread::scope(|s| {
    let handles: Vec<_> = document_batches
        .iter()
        .map(|batch| {
            s.spawn(move || {
                batch
                    .iter()
                    .filter(|(_, text)| text.contains("rust"))
                    .map(|(doc_id, _)| *doc_id)
                    .collect::<Vec<u32>>()
            })
        })
        .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
});

// Merge segments using k-way merge (loser tree)
let mut tree = EnhancedLoserTree::new(LoserTreeConfig::default());
for seg in segments {
    tree.add_way(seg.into_iter()).unwrap();
}
tree.initialize().unwrap();
assert_eq!(tree.peek(), Some(&1));
```

For async pipeline processing (requires `async` feature):

```rust
use zipora::Pipeline;
let _ = std::any::type_name::<Pipeline>();
// Pipeline stages: parse → tokenize → index → compress → flush
// Each stage runs concurrently with work-stealing load balancing
```

## 7. Memory-Mapped Index Files

Serve large indices directly from disk without loading into RAM:

```rust
use zipora::memory::{MmapVec, MmapVecConfig};

let path = std::env::temp_dir().join("zipora_postings.idx");
{
    let mut writer = MmapVec::<u32>::create(&path, MmapVecConfig::default()).unwrap();
    writer.extend([10u32, 20, 30, 40]).unwrap();
    writer.sync().unwrap();
}

// Memory-map an index file — OS manages paging
let index: MmapVec<u32> = MmapVec::open(&path, MmapVecConfig::read_only()).unwrap();

// Random access is backed by the page cache
let doc_id = *index.get(2).unwrap();
assert_eq!(doc_id, 30);

// Bulk access via slice
let postings: &[u32] = index.as_slice();
assert_eq!(postings, &[10, 20, 30, 40]);

drop(index);
let _ = std::fs::remove_file(&path);
```

## 8. Query Result Caching

LRU cache for frequently accessed posting lists:

```rust
use zipora::containers::specialized::LruMap;

fn load_from_disk(_term: &str) -> Vec<u32> {
    vec![1, 5, 10]
}

fn get_postings(term: &str, cache: &mut LruMap<String, Vec<u32>>) -> Vec<u32> {
    if let Some(cached) = cache.get(&term.to_string()) {
        return cached.clone();
    }
    let postings = load_from_disk(term);
    cache.put(term.to_string(), postings.clone()).unwrap();
    postings
}

// Cache hot posting lists
let mut cache: LruMap<String, Vec<u32>> = LruMap::new(1024).unwrap();
assert_eq!(get_postings("rust", &mut cache), vec![1, 5, 10]);
assert_eq!(get_postings("rust", &mut cache), vec![1, 5, 10]);
```

## 9. String Processing for Tokenization

```rust
use zipora::SortableStrVec;
use zipora::SmallMap;

let document = "the quick brown fox jumps over the lazy dog";

// Arena-based string storage
let mut terms = SortableStrVec::new();
for token in document.split_whitespace() {
    terms.push_str(token).unwrap(); // &str variant; push(String) also available
}
terms.sort().unwrap(); // In-place offset sort

// For small lookup tables (field names, stop words)
let mut stop_words = SmallMap::new();
stop_words.insert("the", true).unwrap();
stop_words.insert("and", true).unwrap();
assert_eq!(stop_words.get(&"the"), Some(&true));
```

## 10. BM25 Scoring (Document Length Normalization)

Compact single-byte doc-length storage with pre-computed BM25 scoring (see [PERFORMANCE.md](PERFORMANCE.md) for measured `Bm25BatchScorer`, `FieldnormEncoder::build_score_table`, and end-to-end 100k-doc Block-Max WAND search throughput on `bindiego`).

`FieldnormEncoder` uses Lucene-compatible SmallFloat encoding to compress document lengths into a single byte (3-bit mantissa + 5-bit exponent, same as Lucene/Tantivy). A 256-entry `[f32; 256]` norm table eliminates per-posting float division entirely. `Bm25BatchScorer` processes 8 postings per iteration with AVX2 SIMD.

```rust
use zipora::scoring::{FieldnormEncoder, Bm25BatchScorer};

// Index time: encode doc lengths to single bytes (1 byte/doc vs 2+ bytes for raw u16)
let doc_lengths = vec![50u32, 100, 150, 200, 300];
let fieldnorm_bytes: Vec<u8> = doc_lengths.iter()
    .map(|&l| FieldnormEncoder::encode(l))
    .collect();

// Build time: pre-compute BM25 norm table (256 floats, done once per segment)
let avg_dl = doc_lengths.iter().sum::<u32>() as f32 / doc_lengths.len() as f32;
let norm_table = FieldnormEncoder::build_norm_table(avg_dl, /*k1=*/1.2, /*b=*/0.75);

// Query time: batch-score a posting list (AVX2 SIMD, 8 scores per iteration)
let idf = 3.5f32;
let scorer = Bm25BatchScorer::new(&norm_table, idf, /*k1=*/1.2);
let tfs = vec![2u16, 3, 1, 5, 2];
let mut scores = vec![0.0f32; tfs.len()];
scorer.batch_score(&fieldnorm_bytes, &tfs, &mut scores);

// Phrase queries: single-doc scoring with next-doc prefetch
let score = scorer.score_with_prefetch(&fieldnorm_bytes, /*doc_id=*/2, /*tf=*/3, Some(4));
assert!(score > 0.0);

// Full 2D score table for quantized TF values (eliminates ALL query-time math)
let score_table = FieldnormEncoder::build_score_table(avg_dl, 1.2, 0.75, idf, /*max_tf=*/255);
let precomputed = score_table[3][fieldnorm_bytes[0] as usize]; // score(tf=3, doc=0)
assert!(precomputed > 0.0);
```

See [PERFORMANCE.md](PERFORMANCE.md) and [docs/review/e1.md](review/e1.md) for measured `FieldnormEncoder`, `Bm25BatchScorer`, and systemic `e2e_search_bench` benchmarks on `bindiego`.

## 11. SIMD Cursor Primitives (Block-Max WAND)

SIMD-accelerated primitives designed for Block-Max WAND (BMW) query execution — the core algorithm behind top-K ranked retrieval in modern search engines.

```rust
use zipora::algorithms::{simd_gallop_to, simd_block_filter};

// SIMD galloping search — advances cursor to first position where arr[cursor] >= target
// 3-phase: exponential search → AVX2/SSE2 SIMD scan → scalar tail
// Replaces scalar binary_search/while loops in cursor advance operations
let sorted_doc_ids: Vec<u32> = (0..10000).step_by(3).collect();
let mut cursor = 0usize;
let found = simd_gallop_to(&sorted_doc_ids, &mut cursor, 500);
assert!(found);
assert!(sorted_doc_ids[cursor] >= 500);
// Cursor maintains position — subsequent calls resume from current position
let found = simd_gallop_to(&sorted_doc_ids, &mut cursor, 1000);

// SIMD block filter — threshold filter for BMW block scoring
// Returns (bitmask, count) where bit i is set if scores[i] > theta
// Processes 8 f32s per iteration with AVX2, up to 64 elements per block
let doc_ids: Vec<u32> = (0..64).collect();
let scores: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
let theta = 3.0f32;
let (bitmask, count) = simd_block_filter(&doc_ids, &scores, theta);

// Iterate only over qualifying documents (set bits in bitmask)
for bit_pos in 0..64 {
    if bitmask & (1u64 << bit_pos) != 0 {
        // doc_ids[bit_pos] has score > theta — push to Top-K heap
    }
}
```

### Elias-Fano Cursor `advance_to_index`

All three EF cursor types (`EliasFanoCursor`, `PartitionedEliasFanoCursor`, `OptimalPefCursor`) support `advance_to_index(idx)` for O(log n) random repositioning — essential for BMW's `skip_to_qualifying_block` operation:

```rust
use zipora::succinct::EliasFano;

let values = vec![10, 20, 50, 100, 200, 500, 1000];
let ef = EliasFano::from_sorted(&values);
let mut cursor = ef.cursor();

// Jump directly to index 4 (value 200) — O(log n) via select1
cursor.advance_to_index(4);
assert_eq!(cursor.current(), Some(200));

// Continue forward iteration from new position
cursor.advance();
assert_eq!(cursor.current(), Some(500));

// Backward jumps also supported (reposition to any index)
cursor.advance_to_index(1);
assert_eq!(cursor.current(), Some(20));
```

## Component Selection Guide

| Search Engine Component | Zipora Type | When to Use |
|------------------------|-------------|-------------|
| Term dictionary | `DoubleArrayTrie` | Default choice, compact `base`/`check`/`tail` states, XOR transitions |
| Term dictionary (alternatives) | `ZiporaTrie` | LOUDS/Patricia/Critical-Bit via config |
| Short posting lists | `UintVecMin0` | Variable-width, <1M doc IDs |
| Long posting lists | `SortedUintVec` | Delta-compressed sorted IDs |
| Compressed posting lists | `HybridPostingList` | Auto-selects: Dense/EF/Partitioned/Optimal/Clustered by size and density |
| Dense/run-heavy posting lists | `ClusteredEliasFano` | Per-chunk Run/Bitmap/EF containers; auto-selected by `HybridPostingList` on dense spans; O(1) block `intersect_count` |
| Rank/Select (large bitvecs) | `Rank9` | O(1) rank, O(log n) select, hardware-independent |
| Boolean posting lists | `BitVector` + `AdaptiveRankSelect` | High-frequency terms, bitwise ops |
| AND/OR/NOT queries | `set_ops::multiset_*` | Sorted posting list intersection |
| BMW cursor advance | `simd_gallop_to` | SIMD exponential search, AVX2/SSE2, replaces scalar galloping |
| BMW block scoring | `simd_block_filter` | SIMD threshold filter, up to 64 docs/block, AVX2 |
| EF cursor reposition | `advance_to_index` | O(log n) random jump on EF/PEF/OPEF cursors |
| Prefix autocomplete | `DoubleArrayTrieMap::iter_prefix` | Lazy DFS, no Vec allocation, supports early termination |
| Zero-copy prefix visitor | `DoubleArrayTrieMap::for_each_prefix` | Borrowed `&[u8]` keys, zero per-result allocation |
| Fuzzy term lookup | `DoubleArrayTrieMap::iter_fuzzy` | Lazy Levenshtein, subtree pruning, configurable max distance |
| Zero-copy fuzzy visitor | `DoubleArrayTrieMap::for_each_fuzzy` | Borrowed `&[u8]` keys, zero per-result allocation |
| Integer delta encoding | `StreamVByte` | SIMD SSSE3 shuffle decoding, delta compression |
| Bulk bitwise queries | SIMD rank/select | Vectorized bulk rank/select and bitwise operations |
| Doc-length storage | `FieldnormEncoder` | 1-byte Lucene SmallFloat fieldnorms, replaces `UintVecMin0` |
| BM25 scoring | `Bm25BatchScorer` | AVX2 batch API + prefetch |
| BM25 score table | `FieldnormEncoder::build_score_table` | Full 2D precomputed scores, zero query-time math |
| Document storage | `DictZipBlobStore` | Dictionary compression for similar docs |
| Document storage (fast) | `PlainBlobStore` | Uncompressed, fastest retrieval |
| Posting compression | `HuffmanEncoder` | Fast encode/decode |
| Posting compression | `Rans64Encoder` | Interleaved rANS entropy coding |
| Query cache | `LruMap` | Intrusive LRU cache for hot posting lists |
| Small lookups | `SmallMap` | Inline linear search for ≤8 keys |
| String storage | `SortableStrVec` / `FixedLenStrVec` | Arena-based contiguous string storage |
| Index files | `MmapVec` | Disk-backed, OS-managed paging |
| Segment merge | `MultiWayMerge` / `EnhancedLoserTree` | K-way merge of sorted lists |
| Parallel indexing | `rayon` + `Pipeline` | Multi-threaded segment building |
