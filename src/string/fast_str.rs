//! FastStr: Zero-copy string slice with SIMD-optimized operations
//!
//! This is a Rust port of the C++ `fstring` providing zero-copy string operations
//! with SIMD optimizations for hashing and comparison.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str;

/// Zero-copy string slice equivalent to C++ fstring
/// 
/// FastStr provides a view into string data without owning it, similar to &str
/// but with additional optimizations for high-performance operations.
/// 
/// # Examples
/// 
/// ```rust
/// use infini_zip::FastStr;
/// 
/// let s = FastStr::from_string("hello world");
/// assert_eq!(s.len(), 11);
/// assert!(s.starts_with(FastStr::from_string("hello")));
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FastStr<'a> {
    data: &'a [u8],
}

impl<'a> FastStr<'a> {
    /// Create a FastStr from a byte slice
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Create a FastStr from a string slice
    #[inline]
    pub fn from_string(s: &'a str) -> Self {
        Self { data: s.as_bytes() }
    }

    /// Create a FastStr from a raw pointer and length
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that:
    /// - `ptr` is valid for reads of `len` bytes
    /// - The memory referenced by `ptr` is not mutated during the lifetime `'a`
    /// - `len` does not exceed the allocated size
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Self {
        Self {
            data: unsafe { std::slice::from_raw_parts(ptr, len) }
        }
    }

    /// Get the length of the string in bytes
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the string is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the underlying byte slice
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

    /// Convert to a string slice if the data is valid UTF-8
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        str::from_utf8(self.data).ok()
    }

    /// Convert to a string slice, assuming the data is valid UTF-8
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that the data contains valid UTF-8
    #[inline]
    pub unsafe fn as_str_unchecked(&self) -> &str {
        unsafe { str::from_utf8_unchecked(self.data) }
    }

    /// Get a pointer to the beginning of the data
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Get a byte at the specified index
    #[inline]
    pub fn get_byte(&self, index: usize) -> Option<u8> {
        self.data.get(index).copied()
    }

    /// Get a byte at the specified index without bounds checking
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that `index < self.len()`
    #[inline]
    pub unsafe fn get_byte_unchecked(&self, index: usize) -> u8 {
        debug_assert!(index < self.len());
        unsafe { *self.data.get_unchecked(index) }
    }

    /// Get a substring starting at `start` with length `len`
    pub fn substring(&self, start: usize, len: usize) -> FastStr<'a> {
        let end = start.saturating_add(len).min(self.data.len());
        FastStr::new(&self.data[start..end])
    }

    /// Get a substring from `start` to the end
    pub fn substring_from(&self, start: usize) -> FastStr<'a> {
        let start = start.min(self.data.len());
        FastStr::new(&self.data[start..])
    }

    /// Get a prefix of length `len`
    pub fn prefix(&self, len: usize) -> FastStr<'a> {
        let len = len.min(self.data.len());
        FastStr::new(&self.data[..len])
    }

    /// Get a suffix of length `len`
    pub fn suffix(&self, len: usize) -> FastStr<'a> {
        let len = len.min(self.data.len());
        let start = self.data.len() - len;
        FastStr::new(&self.data[start..])
    }

    /// Check if this string starts with another
    #[inline]
    pub fn starts_with(&self, prefix: FastStr) -> bool {
        self.data.starts_with(prefix.data)
    }

    /// Check if this string ends with another
    #[inline]
    pub fn ends_with(&self, suffix: FastStr) -> bool {
        self.data.ends_with(suffix.data)
    }

    /// Find the position of the first occurrence of a byte
    #[inline]
    pub fn find_byte(&self, byte: u8) -> Option<usize> {
        self.data.iter().position(|&b| b == byte)
    }

    /// Find the position of the first occurrence of a substring
    pub fn find(&self, needle: FastStr) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if needle.len() > self.len() {
            return None;
        }

        // Use optimized search for single byte
        if needle.len() == 1 {
            return self.find_byte(needle.data[0]);
        }

        // Simple naive search - could be optimized with Boyer-Moore or similar
        for i in 0..=(self.len() - needle.len()) {
            if &self.data[i..i + needle.len()] == needle.data {
                return Some(i);
            }
        }
        None
    }

    /// Get the common prefix length with another string
    pub fn common_prefix_len(&self, other: FastStr) -> usize {
        let min_len = self.len().min(other.len());
        for i in 0..min_len {
            if self.data[i] != other.data[i] {
                return i;
            }
        }
        min_len
    }

    /// Compare with another FastStr
    #[inline]
    pub fn compare(&self, other: FastStr) -> Ordering {
        self.data.cmp(other.data)
    }

    /// High-performance hash function with SIMD optimization where available
    pub fn hash_fast(&self) -> u64 {
        #[cfg(target_feature = "avx2")]
        {
            self.hash_avx2()
        }
        #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
        {
            self.hash_sse2()
        }
        #[cfg(not(any(target_feature = "avx2", target_feature = "sse2")))]
        {
            self.hash_fallback()
        }
    }

    #[cfg(target_feature = "avx2")]
    fn hash_avx2(&self) -> u64 {
        // TODO: Implement AVX2 optimized hash
        // For now, fall back to the portable version
        self.hash_fallback()
    }

    #[cfg(target_feature = "sse2")]
    fn hash_sse2(&self) -> u64 {
        // TODO: Implement SSE2 optimized hash
        // For now, fall back to the portable version
        self.hash_fallback()
    }

    fn hash_fallback(&self) -> u64 {
        // Fast hash similar to the C++ version
        let mut h = 2134173u64.wrapping_add(self.data.len() as u64 * 31);
        
        // Process 8-byte chunks for better performance
        let chunks = self.data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            h = h.rotate_left(5).wrapping_add(h).wrapping_add(word);
        }
        
        // Process remaining bytes
        for &byte in remainder {
            h = h.rotate_left(5).wrapping_add(h).wrapping_add(byte as u64);
        }
        
        h
    }

    /// Split the string by a delimiter
    pub fn split(&self, delimiter: u8) -> SplitIter<'a> {
        SplitIter {
            remainder: *self,
            delimiter,
        }
    }

    /// Convert to an owned String
    pub fn into_string(&self) -> String {
        String::from_utf8_lossy(self.data).into_owned()
    }

    /// Convert to a Cow<str>
    pub fn to_cow_str(&self) -> Cow<'a, str> {
        String::from_utf8_lossy(self.data)
    }
}

impl<'a> fmt::Debug for FastStr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_str() {
            Some(s) => write!(f, "FastStr({:?})", s),
            None => write!(f, "FastStr({:?})", self.data),
        }
    }
}

impl<'a> fmt::Display for FastStr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_str() {
            Some(s) => write!(f, "{}", s),
            None => write!(f, "{:?}", self.data),
        }
    }
}

impl<'a> PartialOrd for FastStr<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.compare(*other))
    }
}

impl<'a> Ord for FastStr<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.compare(*other)
    }
}

impl<'a> Hash for FastStr<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use our optimized hash for better performance
        state.write_u64(self.hash_fast());
    }
}

impl<'a> From<&'a str> for FastStr<'a> {
    fn from(s: &'a str) -> Self {
        Self::from_string(s)
    }
}

impl<'a> From<&'a [u8]> for FastStr<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Self::new(bytes)
    }
}

impl<'a> AsRef<[u8]> for FastStr<'a> {
    fn as_ref(&self) -> &[u8] {
        self.data
    }
}

impl<'a> PartialEq<str> for FastStr<'a> {
    fn eq(&self, other: &str) -> bool {
        self.data == other.as_bytes()
    }
}

impl<'a> PartialEq<&str> for FastStr<'a> {
    fn eq(&self, other: &&str) -> bool {
        self.data == other.as_bytes()
    }
}

impl<'a> PartialEq<String> for FastStr<'a> {
    fn eq(&self, other: &String) -> bool {
        self.data == other.as_bytes()
    }
}

impl<'a> PartialEq<[u8]> for FastStr<'a> {
    fn eq(&self, other: &[u8]) -> bool {
        self.data == other
    }
}

impl<'a> PartialEq<&[u8]> for FastStr<'a> {
    fn eq(&self, other: &&[u8]) -> bool {
        self.data == *other
    }
}

/// Iterator over the parts of a FastStr split by a delimiter
pub struct SplitIter<'a> {
    remainder: FastStr<'a>,
    delimiter: u8,
}

impl<'a> Iterator for SplitIter<'a> {
    type Item = FastStr<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remainder.is_empty() {
            return None;
        }

        match self.remainder.find_byte(self.delimiter) {
            Some(pos) => {
                let part = self.remainder.prefix(pos);
                self.remainder = self.remainder.substring_from(pos + 1);
                Some(part)
            }
            None => {
                let part = self.remainder;
                self.remainder = FastStr::new(&[]);
                Some(part)
            }
        }
    }
}

/// Hash function optimized for FastStr
pub struct FastStrHash;

impl FastStrHash {
    /// Hash a FastStr using the optimized hash function
    #[inline]
    #[allow(dead_code)]
    pub fn hash(s: FastStr) -> u64 {
        s.hash_fast()
    }
}

impl std::hash::BuildHasher for FastStrHash {
    type Hasher = FastStrHasher;

    fn build_hasher(&self) -> Self::Hasher {
        FastStrHasher::new()
    }
}

/// Hasher implementation using the FastStr hash algorithm
pub struct FastStrHasher {
    hash: u64,
}

impl FastStrHasher {
    fn new() -> Self {
        Self { hash: 0 }
    }
}

impl Hasher for FastStrHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        self.hash = FastStr::new(bytes).hash_fast();
    }

    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let s = FastStr::from_string("hello world");
        assert_eq!(s.len(), 11);
        assert!(!s.is_empty());
        assert_eq!(s.as_str().unwrap(), "hello world");
    }

    #[test]
    fn test_substring() {
        let s = FastStr::from_string("hello world");
        assert_eq!(s.substring(0, 5).as_str().unwrap(), "hello");
        assert_eq!(s.substring(6, 5).as_str().unwrap(), "world");
        assert_eq!(s.prefix(5).as_str().unwrap(), "hello");
        assert_eq!(s.suffix(5).as_str().unwrap(), "world");
    }

    #[test]
    fn test_starts_ends_with() {
        let s = FastStr::from_string("hello world");
        assert!(s.starts_with(FastStr::from_string("hello")));
        assert!(s.ends_with(FastStr::from_string("world")));
        assert!(!s.starts_with(FastStr::from_string("world")));
        assert!(!s.ends_with(FastStr::from_string("hello")));
    }

    #[test]
    fn test_find() {
        let s = FastStr::from_string("hello world");
        assert_eq!(s.find(FastStr::from_string("world")), Some(6));
        assert_eq!(s.find(FastStr::from_string("xyz")), None);
        assert_eq!(s.find_byte(b'o'), Some(4));
        assert_eq!(s.find_byte(b'z'), None);
    }

    #[test]
    fn test_common_prefix() {
        let s1 = FastStr::from_string("hello");
        let s2 = FastStr::from_string("help");
        assert_eq!(s1.common_prefix_len(s2), 3);
    }

    #[test]
    fn test_split() {
        let s = FastStr::from_string("a,b,c");
        let parts: Vec<_> = s.split(b',').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].as_str().unwrap(), "a");
        assert_eq!(parts[1].as_str().unwrap(), "b");
        assert_eq!(parts[2].as_str().unwrap(), "c");
    }

    #[test]
    fn test_comparison() {
        let s1 = FastStr::from_string("abc");
        let s2 = FastStr::from_string("abd");
        let s3 = FastStr::from_string("abc");
        
        assert!(s1 < s2);
        assert!(s1 == s3);
        assert!(s2 > s1);
    }

    #[test]
    fn test_hash() {
        let s1 = FastStr::from_string("test");
        let s2 = FastStr::from_string("test");
        let s3 = FastStr::from_string("different");
        
        assert_eq!(s1.hash_fast(), s2.hash_fast());
        assert_ne!(s1.hash_fast(), s3.hash_fast());
    }

    #[test]
    fn test_equality_with_string_types() {
        let fs = FastStr::from_string("test");
        assert_eq!(fs, "test");
        assert_eq!(fs, String::from("test"));
        assert_eq!(fs, b"test".as_slice());
    }

    #[test]
    fn test_unsafe_operations() {
        let s = FastStr::from_string("hello world");
        
        unsafe {
            assert_eq!(s.get_byte_unchecked(0), b'h');
            assert_eq!(s.get_byte_unchecked(6), b'w');
            
            let raw_str = s.as_str_unchecked();
            assert_eq!(raw_str, "hello world");
        }
    }

    #[test]
    fn test_byte_operations() {
        let s = FastStr::from_string("hello");
        
        assert_eq!(s.get_byte(0), Some(b'h'));
        assert_eq!(s.get_byte(4), Some(b'o'));
        assert_eq!(s.get_byte(5), None);
        
        assert_eq!(s.as_ptr(), s.as_bytes().as_ptr());
    }

    #[test]
    fn test_substring_operations() {
        let s = FastStr::from_string("hello world");
        
        // substring_from
        let from_6 = s.substring_from(6);
        assert_eq!(from_6.as_str().unwrap(), "world");
        
        let from_20 = s.substring_from(20); // Beyond length
        assert_eq!(from_20.len(), 0);
        
        // suffix
        let suffix_5 = s.suffix(5);
        assert_eq!(suffix_5.as_str().unwrap(), "world");
        
        let suffix_20 = s.suffix(20); // Longer than string
        assert_eq!(suffix_20.as_str().unwrap(), "hello world");
    }

    #[test]
    fn test_string_conversions() {
        let s = FastStr::from_string("hello world");
        
        let owned = s.into_string();
        assert_eq!(owned, "hello world");
        
        let cow = s.to_cow_str();
        assert_eq!(cow, "hello world");
        
        // Test with invalid UTF-8
        let invalid_bytes = &[0xFF, 0xFE, 0xFD];
        let s_invalid = FastStr::new(invalid_bytes);
        assert!(s_invalid.as_str().is_none());
        
        let cow_invalid = s_invalid.to_cow_str();
        assert!(cow_invalid.contains('�')); // Replacement character
    }

    #[test]
    fn test_from_raw_parts() {
        let data = b"test data";
        let s = unsafe { FastStr::from_raw_parts(data.as_ptr(), data.len()) };
        assert_eq!(s.as_str().unwrap(), "test data");
        assert_eq!(s.len(), 9);
    }

    #[test]
    fn test_display_and_debug() {
        let s = FastStr::from_string("hello");
        
        let display = format!("{}", s);
        assert_eq!(display, "hello");
        
        let debug = format!("{:?}", s);
        assert!(debug.contains("FastStr"));
        assert!(debug.contains("hello"));
        
        // Test with invalid UTF-8
        let invalid = FastStr::new(&[0xFF, 0xFE]);
        let debug_invalid = format!("{:?}", invalid);
        assert!(debug_invalid.contains("FastStr"));
        
        let display_invalid = format!("{}", invalid);
        assert!(display_invalid.contains("255") || display_invalid.contains("�"));
    }

    #[test]
    fn test_ordering() {
        let s1 = FastStr::from_string("abc");
        let s2 = FastStr::from_string("abd");
        let s3 = FastStr::from_string("abc");
        
        assert!(s1 < s2);
        assert!(s1 <= s2);
        assert!(s1 <= s3);
        assert!(s2 > s1);
        assert!(s2 >= s1);
        assert!(s1 >= s3);
        
        use std::cmp::Ordering;
        assert_eq!(s1.compare(s2), Ordering::Less);
        assert_eq!(s2.compare(s1), Ordering::Greater);
        assert_eq!(s1.compare(s3), Ordering::Equal);
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let s1 = FastStr::from_string("test string");
        let s2 = FastStr::from_string("test string");
        let s3 = FastStr::from_string("different");
        
        // Test hash_fast consistency
        assert_eq!(s1.hash_fast(), s2.hash_fast());
        assert_ne!(s1.hash_fast(), s3.hash_fast());
        
        // Test Hash trait implementation
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        s1.hash(&mut hasher1);
        s2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_empty_string() {
        let empty = FastStr::from_string("");
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.as_str().unwrap(), "");
        assert_eq!(empty.find_byte(b'a'), None);
        assert_eq!(empty.find(FastStr::from_string("a")), None);
        assert_eq!(empty.common_prefix_len(FastStr::from_string("abc")), 0);
    }

    #[test]
    fn test_find_edge_cases() {
        let s = FastStr::from_string("abcdefg");
        
        // Find empty string
        assert_eq!(s.find(FastStr::from_string("")), Some(0));
        
        // Find string longer than haystack
        assert_eq!(s.find(FastStr::from_string("abcdefghijk")), None);
        
        // Find at beginning
        assert_eq!(s.find(FastStr::from_string("abc")), Some(0));
        
        // Find at end
        assert_eq!(s.find(FastStr::from_string("efg")), Some(4));
        
        // Single character find
        assert_eq!(s.find(FastStr::from_string("d")), Some(3));
    }

    #[test]
    fn test_split_edge_cases() {
        let s = FastStr::from_string("a,,b,c,");
        let parts: Vec<_> = s.split(b',').collect();
        assert_eq!(parts.len(), 4); // "a", "", "b", "c" (trailing comma results in empty remainder that is consumed)
        assert_eq!(parts[0].as_str().unwrap(), "a");
        assert_eq!(parts[1].as_str().unwrap(), "");
        assert_eq!(parts[2].as_str().unwrap(), "b");
        assert_eq!(parts[3].as_str().unwrap(), "c");
        
        // Split empty string
        let empty = FastStr::from_string("");
        let empty_parts: Vec<_> = empty.split(b',').collect();
        assert_eq!(empty_parts.len(), 0);
        
        // Split with no delimiter
        let no_delim = FastStr::from_string("abcdef");
        let no_delim_parts: Vec<_> = no_delim.split(b',').collect();
        assert_eq!(no_delim_parts.len(), 1);
        assert_eq!(no_delim_parts[0].as_str().unwrap(), "abcdef");
    }

    #[test]
    fn test_equality_edge_cases() {
        let s = FastStr::from_string("test");
        
        // Test equality with &str
        assert_eq!(s, "test");
        assert_ne!(s, "different");
        
        // Test equality with &str (already covered above)
        // The PartialEq<&str> implementation covers this case
        
        // Test equality with String
        assert_eq!(s, String::from("test"));
        assert_ne!(s, String::from("different"));
        
        // Test equality with [u8]
        assert_eq!(s, b"test"[..]);
        assert_ne!(s, b"different"[..]);
        
        // Test equality with &[u8] (direct implementation available)
        let bytes: &[u8] = b"test";
        assert_eq!(s, bytes);
    }

    #[test]
    fn test_as_ref() {
        let s = FastStr::from_string("test");
        let bytes: &[u8] = s.as_ref();
        assert_eq!(bytes, b"test");
    }

    #[test]
    fn test_from_implementations() {
        // Test From<&str>
        let from_str: FastStr = "test".into();
        assert_eq!(from_str.as_str().unwrap(), "test");
        
        // Test From<&[u8]>
        let bytes: &[u8] = b"test";
        let from_bytes: FastStr = bytes.into();
        assert_eq!(from_bytes.as_bytes(), b"test");
    }

    #[test] 
    fn test_fast_str_hasher() {
        use crate::string::fast_str::{FastStrHash, FastStrHasher};
        use std::hash::{BuildHasher, Hasher};
        
        let build_hasher = FastStrHash;
        let mut hasher = build_hasher.build_hasher();
        
        hasher.write(b"test");
        let hash1 = hasher.finish();
        
        let mut hasher2 = FastStrHasher::new();
        hasher2.write(b"test");  
        let hash2 = hasher2.finish();
        
        assert_eq!(hash1, hash2);
        
        // Test write_u64
        let mut hasher3 = FastStrHasher::new();
        hasher3.write_u64(12345);
        assert_eq!(hasher3.finish(), 12345);
        
        // Test static hash function
        let s = FastStr::from_string("test");
        let static_hash = FastStrHash::hash(s);
        assert_eq!(static_hash, s.hash_fast());
    }
}