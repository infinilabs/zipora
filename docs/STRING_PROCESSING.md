# String Processing

Zipora provides SIMD-accelerated string operations, pattern matching, and text processing utilities.

## Table of Contents

- [SIMD String Operations](#simd-string-operations)
- [FastStr - Zero-Copy Strings](#faststr---zero-copy-strings)
- [Pattern Matching](#pattern-matching)
- [String Arena](#string-arena)
- [String Pool](#string-pool)
- [Advanced String Containers](#advanced-string-containers)
- [String Sorting](#string-sorting)
- [Performance Characteristics](#performance-characteristics)
- [String Join Utilities](#string-join-utilities)
- [Numeric String Comparison](#numeric-string-comparison)
- [Word Boundary Detection](#word-boundary-detection)
- [Hex Encoding/Decoding](#hex-encodingdecoding)
- [SIMD Hardware Support](#simd-hardware-support)

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

## String Join Utilities

```rust
use zipora::string::{join, join_str, join_fast_str, JoinBuilder};

// Join byte slices
let parts: [&[u8]; 3] = [b"hello", b"world", b"test"];
let result = join(b", ", &parts);
assert_eq!(result, b"hello, world, test");

// Join string slices
let strings = ["a", "b", "c"];
let result = join_str("-", &strings);
assert_eq!(result, "a-b-c");

// Join FastStr values
use zipora::FastStr;
let fast_parts = [FastStr::from_string("hello"), FastStr::from_string("world")];
let result = join_fast_str(" ", &fast_parts);
assert_eq!(result, "hello world");

// Builder pattern with pre-calculated capacity
let mut builder = JoinBuilder::with_capacity(", ", 10);
builder.push("one").push("two").push("three");
let result = builder.finish();
assert_eq!(result, "one, two, three");
```

## Numeric String Comparison

Compare strings as numeric values, handling signs and decimal points correctly.

```rust
use zipora::string::{decimal_strcmp, realnum_strcmp};
use std::cmp::Ordering;

// Decimal integer comparison
assert_eq!(decimal_strcmp("123", "456"), Some(Ordering::Less));
assert_eq!(decimal_strcmp("-10", "5"), Some(Ordering::Less));
assert_eq!(decimal_strcmp("100", "99"), Some(Ordering::Greater));
assert_eq!(decimal_strcmp("-5", "-10"), Some(Ordering::Greater)); // -5 > -10

// Real number comparison (with decimal points)
assert_eq!(realnum_strcmp("3.14", "2.71"), Some(Ordering::Greater));
assert_eq!(realnum_strcmp("10", "9.99"), Some(Ordering::Greater));
assert_eq!(realnum_strcmp("-1.5", "1.5"), Some(Ordering::Less));

// Invalid inputs return None
assert_eq!(decimal_strcmp("abc", "123"), None);
assert_eq!(realnum_strcmp("1.2.3", "1.0"), None);
```

## Word Boundary Detection

Utilities for text tokenization and word-level operations.

```rust
use zipora::string::{
    is_word_boundary, is_word_char, words, word_count,
    find_word_boundaries, word_at_position
};

// Check word characters [a-zA-Z0-9_]
assert!(is_word_char(b'a'));
assert!(is_word_char(b'_'));
assert!(!is_word_char(b' '));

// Detect word boundaries
let text = b"hello world";
assert!(is_word_boundary(text, 0));  // Start of "hello"
assert!(is_word_boundary(text, 5));  // End of "hello"
assert!(is_word_boundary(text, 6));  // Start of "world"

// Find all word boundaries
let boundaries = find_word_boundaries(b"hello world");
assert_eq!(boundaries, vec![0, 5, 6, 11]);

// Iterate over words
let word_list: Vec<_> = words(b"hello, world! test_123").collect();
assert_eq!(word_list.len(), 3);
assert_eq!(word_list[0], b"hello");
assert_eq!(word_list[1], b"world");
assert_eq!(word_list[2], b"test_123");

// Count words
assert_eq!(word_count(b"hello world"), 2);
assert_eq!(word_count(b"one-two-three"), 3);

// Find word at position
assert_eq!(word_at_position(b"hello world", 2), Some((0, 5)));  // "hello"
assert_eq!(word_at_position(b"hello world", 8), Some((6, 11))); // "world"
```

## Hex Encoding/Decoding

Fast hexadecimal encoding and decoding utilities.

```rust
use zipora::string::{
    hex_decode, hex_encode, hex_encode_upper,
    hex_decode_to_slice, hex_encode_to_slice,
    is_valid_hex, parse_hex_byte
};

// Basic encoding/decoding
let encoded = hex_encode(b"Hello");
assert_eq!(encoded, "48656c6c6f");

let decoded = hex_decode("48656c6c6f").unwrap();
assert_eq!(decoded, b"Hello");

// Uppercase encoding
let upper = hex_encode_upper(b"\xDE\xAD\xBE\xEF");
assert_eq!(upper, "DEADBEEF");

// Decode to existing buffer (zero-allocation)
let mut buf = [0u8; 5];
let len = hex_decode_to_slice(b"48656c6c6f", &mut buf).unwrap();
assert_eq!(&buf[..len], b"Hello");

// Encode to existing buffer
let mut hex_buf = [0u8; 10];
let len = hex_encode_to_slice(b"Hello", &mut hex_buf).unwrap();
assert_eq!(&hex_buf[..len], b"48656c6c6f");

// Validation
assert!(is_valid_hex("DEADBEEF"));
assert!(!is_valid_hex("hello"));   // Invalid chars
assert!(!is_valid_hex("123"));     // Odd length

// Parse single hex byte
assert_eq!(parse_hex_byte(b'4', b'8'), Some(0x48));
```

## SIMD Hardware Support

- **AVX2**: 256-bit operations (32 bytes per cycle)
- **SSE4.2**: String-specific instructions (PCMPESTRI, PCMPISTRM)
- **BMI2**: Bit manipulation for packed strings
- **NEON**: ARM SIMD for cross-platform support
- **Scalar fallback**: Automatic when SIMD unavailable
