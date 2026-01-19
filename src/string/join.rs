//! String join utilities
//!
//! Provides efficient string joining operations with SIMD-optimized memory allocation.
//!
//! Ported from topling-zip's fstring join utilities.

use super::FastStr;

/// Join multiple byte slices with a separator
///
/// # Examples
///
/// ```rust
/// use zipora::string::join;
///
/// let parts = [b"hello".as_slice(), b"world".as_slice()];
/// let result = join(b", ", &parts);
/// assert_eq!(result, b"hello, world");
/// ```
pub fn join(separator: &[u8], parts: &[&[u8]]) -> Vec<u8> {
    if parts.is_empty() {
        return Vec::new();
    }

    if parts.len() == 1 {
        return parts[0].to_vec();
    }

    // Calculate total capacity needed
    let total_len: usize = parts.iter().map(|p| p.len()).sum();
    let sep_len = separator.len() * (parts.len() - 1);
    let capacity = total_len + sep_len;

    let mut result = Vec::with_capacity(capacity);

    // Append first part
    result.extend_from_slice(parts[0]);

    // Append remaining parts with separator
    for part in &parts[1..] {
        result.extend_from_slice(separator);
        result.extend_from_slice(part);
    }

    result
}

/// Join multiple strings with a separator
///
/// # Examples
///
/// ```rust
/// use zipora::string::join_str;
///
/// let parts = ["hello", "world"];
/// let result = join_str(", ", &parts);
/// assert_eq!(result, "hello, world");
/// ```
pub fn join_str(separator: &str, parts: &[&str]) -> String {
    if parts.is_empty() {
        return String::new();
    }

    if parts.len() == 1 {
        return parts[0].to_string();
    }

    // Calculate total capacity needed
    let total_len: usize = parts.iter().map(|p| p.len()).sum();
    let sep_len = separator.len() * (parts.len() - 1);
    let capacity = total_len + sep_len;

    let mut result = String::with_capacity(capacity);

    // Append first part
    result.push_str(parts[0]);

    // Append remaining parts with separator
    for part in &parts[1..] {
        result.push_str(separator);
        result.push_str(part);
    }

    result
}

/// Join multiple FastStr with a separator into owned String
///
/// # Examples
///
/// ```rust
/// use zipora::FastStr;
/// use zipora::string::join_fast_str;
///
/// let parts = [FastStr::from_string("hello"), FastStr::from_string("world")];
/// let result = join_fast_str(", ", &parts);
/// assert_eq!(result, "hello, world");
/// ```
pub fn join_fast_str(separator: &str, parts: &[FastStr<'_>]) -> String {
    if parts.is_empty() {
        return String::new();
    }

    if parts.len() == 1 {
        return parts[0].into_string();
    }

    // Calculate total capacity needed
    let total_len: usize = parts.iter().map(|p| p.len()).sum();
    let sep_len = separator.len() * (parts.len() - 1);
    let capacity = total_len + sep_len;

    let mut result = String::with_capacity(capacity);

    // Append first part
    if let Some(s) = parts[0].as_str() {
        result.push_str(s);
    } else {
        result.push_str(&parts[0].into_string());
    }

    // Append remaining parts with separator
    for part in &parts[1..] {
        result.push_str(separator);
        if let Some(s) = part.as_str() {
            result.push_str(s);
        } else {
            result.push_str(&part.into_string());
        }
    }

    result
}

/// Join iterator of strings with a separator
///
/// More efficient for iterators as it avoids intermediate collection.
///
/// # Examples
///
/// ```rust
/// use zipora::string::join_iter;
///
/// let parts = vec!["a", "b", "c"];
/// let result = join_iter(", ", parts.into_iter());
/// assert_eq!(result, "a, b, c");
/// ```
pub fn join_iter<I, S>(separator: &str, iter: I) -> String
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    let mut result = String::new();
    let mut first = true;

    for item in iter {
        if first {
            first = false;
        } else {
            result.push_str(separator);
        }
        result.push_str(item.as_ref());
    }

    result
}

/// Join iterator of byte slices with a separator
///
/// # Examples
///
/// ```rust
/// use zipora::string::join_bytes_iter;
///
/// let parts = vec![b"a".as_slice(), b"b".as_slice()];
/// let result = join_bytes_iter(b"-", parts.into_iter());
/// assert_eq!(result, b"a-b");
/// ```
pub fn join_bytes_iter<I>(separator: &[u8], iter: I) -> Vec<u8>
where
    I: Iterator<Item = &'static [u8]>,
{
    let mut result = Vec::new();
    let mut first = true;

    for item in iter {
        if first {
            first = false;
        } else {
            result.extend_from_slice(separator);
        }
        result.extend_from_slice(item);
    }

    result
}

/// Builder for efficient string joining with pre-calculated capacity
pub struct JoinBuilder<'a> {
    separator: &'a str,
    parts: Vec<&'a str>,
    total_len: usize,
}

impl<'a> JoinBuilder<'a> {
    /// Create a new JoinBuilder with the given separator
    pub fn new(separator: &'a str) -> Self {
        Self {
            separator,
            parts: Vec::new(),
            total_len: 0,
        }
    }

    /// Create a new JoinBuilder with pre-allocated capacity for parts
    pub fn with_capacity(separator: &'a str, capacity: usize) -> Self {
        Self {
            separator,
            parts: Vec::with_capacity(capacity),
            total_len: 0,
        }
    }

    /// Add a part to be joined
    pub fn push(&mut self, part: &'a str) -> &mut Self {
        self.total_len += part.len();
        self.parts.push(part);
        self
    }

    /// Get the number of parts added
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Build the joined string
    pub fn build(&self) -> String {
        if self.parts.is_empty() {
            return String::new();
        }

        if self.parts.len() == 1 {
            return self.parts[0].to_string();
        }

        let sep_len = self.separator.len() * (self.parts.len() - 1);
        let capacity = self.total_len + sep_len;

        let mut result = String::with_capacity(capacity);

        result.push_str(self.parts[0]);
        for part in &self.parts[1..] {
            result.push_str(self.separator);
            result.push_str(part);
        }

        result
    }

    /// Consume the builder and return the joined string
    pub fn finish(self) -> String {
        self.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_bytes() {
        let parts: [&[u8]; 3] = [b"hello", b"world", b"test"];
        let result = join(b", ", &parts);
        assert_eq!(result, b"hello, world, test");
    }

    #[test]
    fn test_join_bytes_empty() {
        let parts: [&[u8]; 0] = [];
        let result = join(b", ", &parts);
        assert!(result.is_empty());
    }

    #[test]
    fn test_join_bytes_single() {
        let parts: [&[u8]; 1] = [b"single"];
        let result = join(b", ", &parts);
        assert_eq!(result, b"single");
    }

    #[test]
    fn test_join_str() {
        let parts = ["a", "b", "c"];
        let result = join_str("-", &parts);
        assert_eq!(result, "a-b-c");
    }

    #[test]
    fn test_join_str_empty_separator() {
        let parts = ["a", "b", "c"];
        let result = join_str("", &parts);
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_join_fast_str() {
        let parts = [
            FastStr::from_string("hello"),
            FastStr::from_string("world"),
        ];
        let result = join_fast_str(" ", &parts);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_join_iter() {
        let parts = vec!["x", "y", "z"];
        let result = join_iter(", ", parts.into_iter());
        assert_eq!(result, "x, y, z");
    }

    #[test]
    fn test_join_iter_empty() {
        let parts: Vec<&str> = vec![];
        let result = join_iter(", ", parts.into_iter());
        assert!(result.is_empty());
    }

    #[test]
    fn test_join_builder() {
        let mut builder = JoinBuilder::new(", ");
        builder.push("a").push("b").push("c");

        assert_eq!(builder.len(), 3);
        assert!(!builder.is_empty());

        let result = builder.build();
        assert_eq!(result, "a, b, c");
    }

    #[test]
    fn test_join_builder_with_capacity() {
        let mut builder = JoinBuilder::with_capacity("|", 10);
        builder.push("1").push("2").push("3");

        let result = builder.finish();
        assert_eq!(result, "1|2|3");
    }

    #[test]
    fn test_join_builder_empty() {
        let builder = JoinBuilder::new(", ");
        assert!(builder.is_empty());
        assert_eq!(builder.build(), "");
    }

    #[test]
    fn test_join_builder_single() {
        let mut builder = JoinBuilder::new(", ");
        builder.push("only");

        let result = builder.build();
        assert_eq!(result, "only");
    }
}
