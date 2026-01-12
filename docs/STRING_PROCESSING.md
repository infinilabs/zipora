# String Processing

Zipora provides SIMD-accelerated string operations and pattern matching capabilities.

## SIMD String Operations

```rust
use zipora::string::{
    SimdStringOps, StringSearch, PatternMatcher,
    FastStr, StringArena, StringPool
};

// SIMD-accelerated string comparison
let s1 = "hello world";
let s2 = "hello earth";
let result = SimdStringOps::compare(s1, s2);
println!("Comparison result: {:?}", result);

// SIMD string search
let haystack = "the quick brown fox jumps over the lazy dog";
let needle = "fox";
let position = SimdStringOps::find(haystack, needle);
assert_eq!(position, Some(16));

// Bulk string operations
let strings = vec!["apple", "banana", "cherry", "date"];
let results = SimdStringOps::bulk_compare(&strings, "banana");

// SIMD memory comparison
let data1 = vec![1u8; 1024];
let data2 = vec![1u8; 1024];
let equal = SimdStringOps::memory_equal(&data1, &data2);
assert!(equal);
```

## FastStr - Zero-Copy Strings

```rust
use zipora::FastStr;

// Zero-copy string with SIMD hashing
let s = FastStr::from_string("hello world");
println!("Hash: {:x}", s.hash_fast());

// Small string optimization (inline storage for short strings)
let short = FastStr::from_str("hi");
assert!(short.is_inline());

// String interning for deduplication
let interned1 = FastStr::intern("shared string");
let interned2 = FastStr::intern("shared string");
assert!(interned1.ptr_eq(&interned2)); // Same memory location

// Efficient concatenation
let combined = FastStr::concat(&[s, FastStr::from_str(" - appended")]);
```

## Pattern Matching

```rust
use zipora::string::{PatternMatcher, PatternConfig, MatchResult};

// High-performance pattern matching
let matcher = PatternMatcher::new("pattern");
let text = "text with pattern inside";
let matches: Vec<MatchResult> = matcher.find_all(text);

// Multiple pattern matching (Aho-Corasick style)
let patterns = vec!["cat", "car", "card"];
let multi_matcher = PatternMatcher::multi(patterns);
let results = multi_matcher.find_all_patterns("the cat sat on the car");

// Regex-like patterns with SIMD acceleration
let config = PatternConfig {
    case_sensitive: false,
    use_simd: true,
    max_matches: 100,
};
let matcher = PatternMatcher::with_config("pattern", config);

// Wildcard matching
let wildcard = PatternMatcher::wildcard("*.txt");
assert!(wildcard.matches("document.txt"));
```

## String Arena

```rust
use zipora::string::{StringArena, StringArenaConfig};

// Memory-efficient string storage with deduplication
let config = StringArenaConfig::performance_optimized();
let mut arena = StringArena::with_config(config);

// Add strings with automatic deduplication
let id1 = arena.insert("shared string").unwrap();
let id2 = arena.insert("shared string").unwrap();
assert_eq!(id1, id2); // Same ID for duplicate strings

// Efficient retrieval
let s = arena.get(id1).unwrap();
assert_eq!(s, "shared string");

// Bulk insertion
let strings = vec!["one", "two", "three"];
let ids = arena.insert_batch(&strings).unwrap();

// Statistics
let stats = arena.stats();
println!("Unique strings: {}", stats.unique_count);
println!("Total bytes: {}", stats.total_bytes);
println!("Deduplication ratio: {:.2}%", stats.dedup_ratio * 100.0);
```

## String Pool

```rust
use zipora::string::{StringPool, StringPoolConfig};

// Thread-safe string pool with interning
let config = StringPoolConfig::default();
let pool = StringPool::with_config(config);

// Intern strings across threads
let interned = pool.intern("shared across threads");

// Access from multiple threads safely
let handle = interned.clone();
std::thread::spawn(move || {
    println!("String: {}", handle.as_str());
});

// Pool statistics
let stats = pool.stats();
println!("Interned count: {}", stats.interned_count);
println!("Memory usage: {} bytes", stats.memory_bytes);
```

## Advanced String Containers

```rust
use zipora::{AdvancedStringVec, AdvancedStringConfig, BitPackedStringVec32};

// 3-level compression strategy
let config = AdvancedStringConfig::performance_optimized();
let mut advanced_vec = AdvancedStringVec::with_config(config);
advanced_vec.push("hello world").unwrap();
advanced_vec.push("hello rust").unwrap();   // Prefix deduplication
advanced_vec.push("hello").unwrap();        // Overlap detection

let stats = advanced_vec.stats();
println!("Compression ratio: {:.1}%", stats.compression_ratio * 100.0);

// Bit-packed string vectors with BMI2 acceleration
let mut bit_packed: BitPackedStringVec32 = BitPackedStringVec::new();
bit_packed.push("memory efficient").unwrap();

// SIMD-accelerated search
#[cfg(feature = "simd")]
{
    if let Some(index) = bit_packed.find_simd("memory efficient") {
        println!("Found at index: {}", index);
    }
}
```

## String Sorting

```rust
use zipora::SortableStrVec;

// Arena-based string sorting with algorithm selection
let mut sortable = SortableStrVec::new();
sortable.push_str("cherry").unwrap();
sortable.push_str("apple").unwrap();
sortable.push_str("banana").unwrap();

// Intelligent algorithm selection (comparison vs radix)
sortable.sort_lexicographic().unwrap();

// Custom comparison
sortable.sort_by(|a, b| {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}).unwrap();

// Stable sort preserving equal element order
sortable.stable_sort_lexicographic().unwrap();
```

## Performance Characteristics

| Operation | SIMD Speedup | Use Case |
|-----------|--------------|----------|
| **String Compare** | 4-8x | Sorting, searching |
| **String Search** | 2-4x | Pattern matching |
| **Memory Compare** | 8-16x | Deduplication |
| **Hash Computation** | 2-3x | Hash maps, caching |
| **Bulk Operations** | 4-12x | Batch processing |

## SIMD Hardware Support

- **AVX2**: 256-bit operations (32 bytes per cycle)
- **SSE4.2**: String-specific instructions (PCMPESTRI, PCMPISTRM)
- **BMI2**: Bit manipulation for packed strings
- **NEON**: ARM SIMD for cross-platform support
- **Scalar fallback**: Automatic when SIMD unavailable
