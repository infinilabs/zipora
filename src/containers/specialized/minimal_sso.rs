//! Small String Optimization.
//!
//! Strings ≤31 bytes are stored inline (no heap allocation).
//! Longer strings are heap-allocated with standard growth strategy.
//! Discriminant: last byte = 255 → heap, else → inline (unused_len).
//!
//! Memory layout (32 bytes total):
//! ```text
//! Local:  [u8; 31] data  |  u8 unused_len  (unused_len = 31 - len)
//! Heap:   *mut u8 ptr    |  usize len  |  usize cap  |  pad  |  u8 flag=255
//! ```

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

/// Maximum inline capacity (SIZE - 1 bytes).
const INLINE_CAP: usize = 31;
/// Flag value indicating heap allocation.
const HEAP_FLAG: u8 = 255;

/// A 32-byte string with small string optimization.
/// Strings ≤31 bytes are stored inline without heap allocation.
#[repr(C)]
pub struct MinimalSso {
    bytes: [u8; 32],
}

// Check that the struct is exactly 32 bytes
const _: () = assert!(std::mem::size_of::<MinimalSso>() == 32);

impl MinimalSso {
    /// Create an empty string (inline, zero length).
    #[inline]
    pub fn new() -> Self {
        let mut s = Self { bytes: [0u8; 32] };
        s.bytes[31] = INLINE_CAP as u8; // unused_len = 31 (empty)
        s
    }

    /// Create from a byte slice.
    #[inline]
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut s = Self::new();
        if data.len() <= INLINE_CAP {
            s.bytes[..data.len()].copy_from_slice(data);
            s.bytes[31] = (INLINE_CAP - data.len()) as u8;
        } else {
            s.heap_init(data);
        }
        s
    }

    /// Create from a string slice.
    #[inline]
    pub fn from_str(data: &str) -> Self {
        Self::from_bytes(data.as_bytes())
    }

    // -- accessors --

    /// Returns `true` if the string is stored inline (no heap allocation).
    #[inline]
    pub fn is_local(&self) -> bool {
        self.bytes[31] != HEAP_FLAG
    }

    /// Returns the length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        if self.is_local() {
            INLINE_CAP - self.bytes[31] as usize
        } else {
            self.heap_len()
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the current capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        if self.is_local() {
            INLINE_CAP
        } else {
            self.heap_cap()
        }
    }

    /// Returns the string contents as a byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        if self.is_local() {
            let len = INLINE_CAP - self.bytes[31] as usize;
            &self.bytes[..len]
        } else {
            // SAFETY: Heap mode guarantees valid ptr/len from successful allocation in heap_init/write_heap
            unsafe {
                let ptr = self.heap_ptr();
                let len = self.heap_len();
                std::slice::from_raw_parts(ptr, len)
            }
        }
    }

    /// Returns the string contents as `&str`.
    /// # Safety
    /// The caller must ensure the contents are valid UTF-8.
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: Caller ensures contents are valid UTF-8 per function documentation
        unsafe { std::str::from_utf8_unchecked(self.as_bytes()) }
    }

    /// Try to return as `&str`, validating UTF-8.
    #[inline]
    pub fn to_str(&self) -> Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(self.as_bytes())
    }

    // -- mutation --

    /// Append bytes, potentially spilling to heap.
    pub fn push_bytes(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let old_len = self.len();
        let new_len = old_len + data.len();

        if self.is_local() {
            if new_len <= INLINE_CAP {
                // Still fits inline
                self.bytes[old_len..old_len + data.len()].copy_from_slice(data);
                self.bytes[31] = (INLINE_CAP - new_len) as u8;
            } else {
                // Spill to heap
                self.spill_to_heap(old_len, data);
            }
        } else {
            // Already on heap
            if new_len > self.heap_cap() {
                self.heap_grow(new_len);
            }
            // SAFETY: heap_grow ensures capacity >= new_len, ptr valid from allocation
            unsafe {
                let ptr = self.heap_ptr();
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(old_len), data.len());
            }
            self.set_heap_len(new_len);
        }
    }

    /// Append a string slice.
    #[inline]
    pub fn push_str(&mut self, s: &str) {
        self.push_bytes(s.as_bytes());
    }

    /// Clear contents (keeps heap allocation if present).
    pub fn clear(&mut self) {
        if self.is_local() {
            self.bytes[31] = INLINE_CAP as u8;
        } else {
            self.set_heap_len(0);
        }
    }

    // -- private heap helpers --

    #[cold]
    fn heap_init(&mut self, data: &[u8]) {
        let cap = data.len().next_power_of_two().max(64);
        let layout = std::alloc::Layout::from_size_align(cap, 1)
            .expect("layout creation: non-zero size, power-of-two alignment");
        // SAFETY: Layout is valid (cap > 0, align = 1)
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        // SAFETY: ptr valid from allocation, data.len() <= cap
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        self.write_heap(ptr, data.len(), cap);
    }

    #[cold]
    fn spill_to_heap(&mut self, old_len: usize, extra: &[u8]) {
        let new_len = old_len + extra.len();
        let cap = new_len.next_power_of_two().max(64);
        let layout = std::alloc::Layout::from_size_align(cap, 1)
            .expect("layout creation: non-zero size, power-of-two alignment");
        // SAFETY: Layout is valid (cap > 0, align = 1)
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        // SAFETY: ptr valid from allocation, old_len + extra.len() <= cap
        unsafe {
            // Copy existing inline data
            std::ptr::copy_nonoverlapping(self.bytes.as_ptr(), ptr, old_len);
            // Copy new data
            std::ptr::copy_nonoverlapping(extra.as_ptr(), ptr.add(old_len), extra.len());
        }
        self.write_heap(ptr, new_len, cap);
    }

    fn heap_grow(&mut self, min_cap: usize) {
        let old_cap = self.heap_cap();
        let new_cap = (old_cap * 2).max(min_cap.next_power_of_two());
        let old_ptr = self.heap_ptr();
        let old_layout = std::alloc::Layout::from_size_align(old_cap, 1)
            .expect("layout creation: non-zero size, power-of-two alignment");
        // SAFETY: old_ptr from valid allocation, old_layout matches original alloc, new_cap > 0
        let ptr = unsafe { std::alloc::realloc(old_ptr, old_layout, new_cap) };
        if ptr.is_null() {
            let new_layout = std::alloc::Layout::from_size_align(new_cap, 1)
                .expect("layout creation: non-zero size, power-of-two alignment");
            std::alloc::handle_alloc_error(new_layout);
        }
        let len = self.heap_len();
        self.write_heap(ptr, len, new_cap);
    }

    // Read/write heap metadata from the 32-byte blob.
    // Layout: [ptr:8][len:8][cap:8][pad][flag:1]

    #[inline]
    fn heap_ptr(&self) -> *mut u8 {
        let p: usize = usize::from_ne_bytes(self.bytes[0..8].try_into().expect("slice is 8 bytes"));
        p as *mut u8
    }

    #[inline]
    fn heap_len(&self) -> usize {
        usize::from_ne_bytes(self.bytes[8..16].try_into().expect("slice is 8 bytes"))
    }

    #[inline]
    fn heap_cap(&self) -> usize {
        usize::from_ne_bytes(self.bytes[16..24].try_into().expect("slice is 8 bytes"))
    }

    #[inline]
    fn set_heap_len(&mut self, len: usize) {
        self.bytes[8..16].copy_from_slice(&len.to_ne_bytes());
    }

    fn write_heap(&mut self, ptr: *mut u8, len: usize, cap: usize) {
        self.bytes[0..8].copy_from_slice(&(ptr as usize).to_ne_bytes());
        self.bytes[8..16].copy_from_slice(&len.to_ne_bytes());
        self.bytes[16..24].copy_from_slice(&cap.to_ne_bytes());
        // pad bytes 24..31 are don't-care
        self.bytes[31] = HEAP_FLAG;
    }
}

impl Drop for MinimalSso {
    fn drop(&mut self) {
        if !self.is_local() {
            let ptr = self.heap_ptr();
            let cap = self.heap_cap();
            if cap > 0 && !ptr.is_null() {
                let layout = std::alloc::Layout::from_size_align(cap, 1)
                    .expect("layout creation: non-zero size, power-of-two alignment");
                // SAFETY: ptr from valid allocation, layout matches original alloc
                unsafe {
                    std::alloc::dealloc(ptr, layout);
                }
            }
        }
    }
}

impl Clone for MinimalSso {
    fn clone(&self) -> Self {
        Self::from_bytes(self.as_bytes())
    }
}

impl Default for MinimalSso {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for MinimalSso {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl Hash for MinimalSso {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
    }
}

impl PartialEq for MinimalSso {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}
impl Eq for MinimalSso {}

impl PartialOrd for MinimalSso {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinimalSso {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl fmt::Debug for MinimalSso {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_str() {
            Ok(s) => write!(f, "MinimalSso({:?})", s),
            Err(_) => write!(f, "MinimalSso({:?})", self.as_bytes()),
        }
    }
}

impl fmt::Display for MinimalSso {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_str() {
            Ok(s) => f.write_str(s),
            Err(_) => write!(f, "{:?}", self.as_bytes()),
        }
    }
}

impl From<&str> for MinimalSso {
    fn from(s: &str) -> Self {
        Self::from_str(s)
    }
}

impl From<String> for MinimalSso {
    fn from(s: String) -> Self {
        Self::from_str(&s)
    }
}

impl From<&[u8]> for MinimalSso {
    fn from(s: &[u8]) -> Self {
        Self::from_bytes(s)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of(s: &MinimalSso) -> u64 {
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        h.finish()
    }

    #[test]
    fn test_size() {
        assert_eq!(std::mem::size_of::<MinimalSso>(), 32);
    }

    #[test]
    fn test_empty() {
        let s = MinimalSso::new();
        assert!(s.is_local());
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.as_bytes(), &[] as &[u8]);
    }

    #[test]
    fn test_short_string_stays_local() {
        let s = MinimalSso::from_str("hello");
        assert!(s.is_local());
        assert_eq!(s.len(), 5);
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn test_exact_boundary_stays_local() {
        // 31 bytes = maximum inline
        let data = "abcdefghijklmnopqrstuvwxyz12345"; // 31 chars
        assert_eq!(data.len(), 31);
        let s = MinimalSso::from_str(data);
        assert!(s.is_local());
        assert_eq!(s.len(), 31);
        assert_eq!(s.as_str(), data);
    }

    #[test]
    fn test_long_string_goes_to_heap() {
        let data = "abcdefghijklmnopqrstuvwxyz123456"; // 32 chars
        assert_eq!(data.len(), 32);
        let s = MinimalSso::from_str(data);
        assert!(!s.is_local());
        assert_eq!(s.len(), 32);
        assert_eq!(s.as_str(), data);
    }

    #[test]
    fn test_clone_local() {
        let s = MinimalSso::from_str("hello");
        let c = s.clone();
        assert!(c.is_local());
        assert_eq!(c.as_str(), "hello");
    }

    #[test]
    fn test_clone_heap() {
        let data = "a]".repeat(20); // 40 bytes
        let s = MinimalSso::from_str(&data);
        let c = s.clone();
        assert!(!c.is_local());
        assert_eq!(c.as_str(), data);
    }

    #[test]
    fn test_push_stays_local() {
        let mut s = MinimalSso::from_str("hello");
        s.push_str(" world");
        assert!(s.is_local()); // 11 bytes, still fits
        assert_eq!(s.as_str(), "hello world");
    }

    #[test]
    fn test_push_spills_to_heap() {
        let mut s = MinimalSso::from_str("hello");
        s.push_str(" world, this is a long string!!!"); // total > 31
        assert!(!s.is_local());
        assert_eq!(s.as_str(), "hello world, this is a long string!!!");
    }

    #[test]
    fn test_push_on_heap() {
        let mut s = MinimalSso::from_str(&"x".repeat(40));
        assert!(!s.is_local());
        s.push_str("more");
        assert_eq!(s.len(), 44);
        assert!(s.as_str().starts_with("xxxx"));
        assert!(s.as_str().ends_with("more"));
    }

    #[test]
    fn test_hash_consistency() {
        let a = MinimalSso::from_str("hello");
        let b = MinimalSso::from_str("hello");
        assert_eq!(hash_of(&a), hash_of(&b));

        // Same content, one local one heap
        let short = MinimalSso::from_str("hi");
        let long = MinimalSso::from_bytes(b"hi"); // also local
        assert_eq!(hash_of(&short), hash_of(&long));
    }

    #[test]
    fn test_eq_ord() {
        let a = MinimalSso::from_str("abc");
        let b = MinimalSso::from_str("abc");
        let c = MinimalSso::from_str("abd");
        assert_eq!(a, b);
        assert!(a < c);
    }

    #[test]
    fn test_clear() {
        let mut s = MinimalSso::from_str("hello");
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);

        let mut h = MinimalSso::from_str(&"x".repeat(50));
        h.clear();
        assert!(h.is_empty());
        assert!(!h.is_local()); // still heap-allocated, just empty
    }

    #[test]
    fn test_from_conversions() {
        let a: MinimalSso = "hello".into();
        let b: MinimalSso = String::from("hello").into();
        let c: MinimalSso = b"hello".as_slice().into();
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn test_display_debug() {
        let s = MinimalSso::from_str("test");
        assert_eq!(format!("{}", s), "test");
        assert_eq!(format!("{:?}", s), "MinimalSso(\"test\")");
    }

    #[test]
    fn test_capacity() {
        let local = MinimalSso::from_str("hi");
        assert_eq!(local.capacity(), 31);

        let heap = MinimalSso::from_str(&"x".repeat(50));
        assert!(heap.capacity() >= 50);
    }
}
