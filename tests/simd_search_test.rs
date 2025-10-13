//! Integration tests for SSE4.2 PCMPESTRI string search

use zipora::io::simd_memory::search::*;
use std::cmp::Ordering;

#[test]
fn test_find_char_basic() {
    let searcher = SimdStringSearch::new();

    // Found cases
    assert_eq!(searcher.find_char(b"hello", b'h'), Some(0));
    assert_eq!(searcher.find_char(b"hello", b'e'), Some(1));
    assert_eq!(searcher.find_char(b"hello", b'o'), Some(4));

    // Not found
    assert_eq!(searcher.find_char(b"hello", b'x'), None);

    // Empty haystack
    assert_eq!(searcher.find_char(b"", b'a'), None);
}

#[test]
fn test_find_char_long() {
    let searcher = SimdStringSearch::new();

    // Create long string to trigger SIMD path
    let mut haystack = vec![b'a'; 100];
    haystack.push(b'x');
    assert_eq!(searcher.find_char(&haystack, b'x'), Some(100));
}

#[test]
fn test_find_pattern_basic() {
    let searcher = SimdStringSearch::new();

    // Basic matches
    assert_eq!(searcher.find_pattern(b"hello world", b"world"), Some(6));
    assert_eq!(searcher.find_pattern(b"hello world", b"hello"), Some(0));
    assert_eq!(searcher.find_pattern(b"hello world", b"lo wo"), Some(3));

    // Not found
    assert_eq!(searcher.find_pattern(b"hello world", b"xyz"), None);

    // Empty needle
    assert_eq!(searcher.find_pattern(b"hello", b""), Some(0));
}

#[test]
fn test_find_pattern_short() {
    let searcher = SimdStringSearch::new();

    // Short patterns (≤16 bytes)
    assert_eq!(searcher.find_pattern(b"abcdefgh", b"cde"), Some(2));
    assert_eq!(
        searcher.find_pattern(b"0123456789abcdef", b"789a"),
        Some(7)
    );
}

#[test]
fn test_find_pattern_medium() {
    let searcher = SimdStringSearch::new();

    // Medium patterns (17-32 bytes)
    let needle = b"0123456789abcdefghij"; // 20 bytes
    let haystack = b"prefix_0123456789abcdefghij_suffix";
    assert_eq!(searcher.find_pattern(haystack, needle), Some(7));
}

#[test]
fn test_find_pattern_long() {
    let searcher = SimdStringSearch::new();

    // Long patterns (>32 bytes)
    let needle = b"0123456789abcdefghijklmnopqrstuvwxyz"; // 36 bytes
    let haystack = b"prefix_0123456789abcdefghijklmnopqrstuvwxyz_suffix";
    assert_eq!(searcher.find_pattern(haystack, needle), Some(7));
}

#[test]
fn test_find_any_of() {
    let searcher = SimdStringSearch::new();

    // Find vowels
    assert_eq!(searcher.find_any_of(b"hello", b"aeiou"), Some(1)); // 'e'
    assert_eq!(searcher.find_any_of(b"world", b"aeiou"), Some(1)); // 'o'

    // Not found
    assert_eq!(searcher.find_any_of(b"xyz", b"aeiou"), None);

    // Empty char set
    assert_eq!(searcher.find_any_of(b"hello", b""), None);
}

#[test]
fn test_compare_strings() {
    let searcher = SimdStringSearch::new();

    // Equal
    assert_eq!(
        searcher.compare_strings(b"hello", b"hello"),
        Ordering::Equal
    );

    // Less than
    assert_eq!(searcher.compare_strings(b"abc", b"abd"), Ordering::Less);

    // Greater than
    assert_eq!(
        searcher.compare_strings(b"xyz", b"abc"),
        Ordering::Greater
    );

    // Different lengths
    assert_eq!(searcher.compare_strings(b"abc", b"abcd"), Ordering::Less);
}

#[test]
fn test_convenience_functions() {
    // Test public convenience functions
    assert_eq!(find_char(b"hello", b'e'), Some(1));
    assert_eq!(find_pattern(b"hello world", b"world"), Some(6));
    assert_eq!(find_any_of(b"hello", b"aeiou"), Some(1));
    assert_eq!(compare_strings(b"abc", b"abc"), Ordering::Equal);
}

#[test]
fn test_simd_tier_detection() {
    let searcher = SimdStringSearch::new();
    let tier = searcher.tier();

    // Should detect some SIMD tier on modern hardware
    println!("Detected SIMD tier: {:?}", tier);

    // Tier should be valid
    assert!(matches!(
        tier,
        SearchTier::Scalar
            | SearchTier::SSE42
            | SearchTier::AVX2
            | SearchTier::AVX512
            | SearchTier::NEON
    ));
}

#[test]
fn test_edge_cases() {
    let searcher = SimdStringSearch::new();

    // Single character haystack
    assert_eq!(searcher.find_char(b"a", b'a'), Some(0));
    assert_eq!(searcher.find_char(b"a", b'b'), None);

    // Single character pattern
    assert_eq!(searcher.find_pattern(b"hello", b"h"), Some(0));

    // Pattern at end
    assert_eq!(searcher.find_pattern(b"hello", b"lo"), Some(3));
}

#[test]
fn test_performance_comparison() {
    let searcher = SimdStringSearch::new();

    // Create test data
    let mut haystack = vec![b'a'; 10000];
    haystack.extend_from_slice(b"target_pattern");
    haystack.extend_from_slice(&vec![b'a'; 1000]);

    // Test character search
    let result = searcher.find_char(&haystack, b't');
    assert_eq!(result, Some(10000));

    // Test pattern search
    let result = searcher.find_pattern(&haystack, b"target_pattern");
    assert_eq!(result, Some(10000));
}
