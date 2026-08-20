# String Processing

Zipora provides SIMD-accelerated string search, zero-copy string views, and text processing utilities.

## Table of Contents

- [SIMD String Search](#simd-string-search)
- [FastStr - Zero-Copy Strings](#faststr---zero-copy-strings)
- [BMI2 String Operations](#bmi2-string-operations)
- [String Sorting](#string-sorting)
- [String Join Utilities](#string-join-utilities)
- [Numeric String Comparison](#numeric-string-comparison)
- [Word Boundary Detection](#word-boundary-detection)
- [Hex Encoding/Decoding](#hex-encodingdecoding)

## SIMD String Search

SSE4.2 PCMPESTRI-based search with runtime feature detection and automatic
tier selection (AVX-512 → AVX2 → SSE4.2 → scalar fallback).

```rust
use zipora::string::{
    SimdStringSearch, SearchTier, get_global_simd_search,
    sse42_strstr, sse42_strchr, sse42_strcmp, sse42_multi_search,
};
use std::cmp::Ordering;

// Reusable instance with runtime feature detection
let search = SimdStringSearch::new();
println!("Selected tier: {:?}", search.tier());

let haystack = b"the quick brown fox jumps over the lazy dog";

// Substring search (strstr equivalent)
assert_eq!(search.sse42_strstr(haystack, b"fox"), Some(16));

// Single-character search (strchr equivalent)
assert_eq!(search.sse42_strchr(haystack, b'q'), Some(4));

// Comparison with early-exit mismatch detection.
// Note: compares lengths first (shorter sorts first), then bytes.
assert_eq!(search.sse42_strcmp(b"abc", b"abd"), Ordering::Less);

// Multi-character search: all positions of any needle byte
let result = search.sse42_multi_search(b"a,b;c", b",;");
assert_eq!(result.positions, vec![1, 3]);

// Module-level convenience functions use a cached global instance
assert_eq!(sse42_strstr(haystack, b"lazy"), Some(35));
let global = get_global_simd_search();
```

## FastStr - Zero-Copy Strings

`FastStr<'a>` is a zero-copy view over a borrowed byte slice with
SIMD-accelerated hashing and search.

```rust
use zipora::FastStr;

// Zero-copy views over borrowed bytes or &str
let s = FastStr::from_string("hello world");
let b = FastStr::new(b"hello world");
assert_eq!(s, b);

// SIMD-accelerated hashing (AVX2/SSE2 with scalar fallback)
println!("Hash: {:x}", s.hash_fast());

// Zero-copy slicing and search
assert!(s.starts_with(FastStr::from_string("hello")));
assert_eq!(s.find_byte(b'w'), Some(6));
assert_eq!(s.find(FastStr::from_string("world")), Some(6));
assert_eq!(s.substring(6, 5).as_str(), Some("world"));
assert_eq!(s.prefix(5).as_str(), Some("hello"));

// Comparison and common-prefix utilities
assert_eq!(s.common_prefix_len(FastStr::from_string("hello rust")), 6);
```

## BMI2 String Operations

Hardware-accelerated string processing using BMI2 instructions (PEXT/BEXTR)
with scalar fallbacks. Available as free functions (using a cached global
processor) or via `Bmi2StringProcessor`.

```rust
use zipora::string::{
    Bmi2StringProcessor, get_global_bmi2_processor,
    validate_utf8_bmi2, wildcard_match_bmi2, search_string_bmi2,
};

// UTF-8 validation
assert!(validate_utf8_bmi2("héllo".as_bytes()));

// Substring search
assert_eq!(search_string_bmi2("hello world", "world"), Some(6));

// Glob-style wildcard matching (* and ?)
assert!(wildcard_match_bmi2("document.txt", "*.txt"));

// Reusable processor with capability inspection
let processor = get_global_bmi2_processor();
println!("BMI2 available: {}", processor.is_bmi2_available());
```

Additional operations on `Bmi2StringProcessor` include
`count_utf8_chars_bmi2`, `hash_string_bmi2`, `to_lowercase_ascii_bmi2` /
`to_uppercase_ascii_bmi2`, and `detect_runs_bmi2`.

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
// `sort()` is a convenience alias for sort_lexicographic()

// Sort by string length
sortable.sort_by_length().unwrap();

// Custom comparison
sortable.sort_by(|a, b| {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}).unwrap();
```

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

> **Note**: `realnum_strcmp` expects normalized inputs (no leading zeros in the
> integer part, no trailing zeros in the fraction, no `-0`), matching the
> topling-zip C++ contract. Non-normalized inputs like `"1.50"` or `"01.5"`
> compare lexicographically, not numerically — see the `# Preconditions`
> section in the API docs.

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

For details on the SIMD tier framework and hardware feature detection, see [SIMD.md](SIMD.md).
