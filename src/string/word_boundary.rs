//! Word boundary detection utilities
//!
//! Provides functions for detecting word boundaries in text, useful for
//! tokenization, search indexing, and text processing.
//!
//! Ported from topling-zip's fstring word boundary utilities.

/// Check if a byte is at a word boundary
///
/// A word boundary exists:
/// - At the start of the string
/// - At the end of the string
/// - Between a word character and a non-word character
///
/// Word characters are: [a-zA-Z0-9_]
///
/// # Examples
///
/// ```rust
/// use zipora::string::is_word_boundary;
///
/// let text = b"hello world";
/// assert!(is_word_boundary(text, 0));     // Start of "hello"
/// assert!(is_word_boundary(text, 5));     // Between "hello" and space
/// assert!(is_word_boundary(text, 6));     // Between space and "world"
/// assert!(is_word_boundary(text, 11));    // End of string
/// ```
#[inline]
pub fn is_word_boundary(text: &[u8], pos: usize) -> bool {
    if text.is_empty() {
        return true;
    }

    if pos == 0 || pos >= text.len() {
        return true;
    }

    let prev = text[pos - 1];
    let curr = text[pos];

    is_word_char(prev) != is_word_char(curr)
}

/// Check if a byte is a word character [a-zA-Z0-9_]
///
/// # Examples
///
/// ```rust
/// use zipora::string::is_word_char;
///
/// assert!(is_word_char(b'a'));
/// assert!(is_word_char(b'Z'));
/// assert!(is_word_char(b'5'));
/// assert!(is_word_char(b'_'));
/// assert!(!is_word_char(b' '));
/// assert!(!is_word_char(b'-'));
/// ```
#[inline]
pub const fn is_word_char(c: u8) -> bool {
    matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
}

/// Check if a byte is a whitespace character
///
/// Matches: space, tab, newline, carriage return, form feed, vertical tab
#[inline]
pub const fn is_whitespace(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0C | 0x0B)
}

/// Check if a byte is an ASCII punctuation character
#[inline]
pub const fn is_punctuation(c: u8) -> bool {
    matches!(c,
        b'!' | b'"' | b'#' | b'$' | b'%' | b'&' | b'\'' | b'(' | b')' |
        b'*' | b'+' | b',' | b'-' | b'.' | b'/' | b':' | b';' | b'<' |
        b'=' | b'>' | b'?' | b'@' | b'[' | b'\\' | b']' | b'^' | b'`' |
        b'{' | b'|' | b'}' | b'~'
    )
}

/// Find all word boundary positions in text
///
/// Returns a vector of positions where word boundaries occur.
///
/// # Examples
///
/// ```rust
/// use zipora::string::find_word_boundaries;
///
/// let text = b"hello world";
/// let boundaries = find_word_boundaries(text);
/// assert_eq!(boundaries, vec![0, 5, 6, 11]);
/// ```
pub fn find_word_boundaries(text: &[u8]) -> Vec<usize> {
    if text.is_empty() {
        return vec![0];
    }

    let mut boundaries = Vec::new();

    // Always include start
    boundaries.push(0);

    // Find internal boundaries
    for i in 1..text.len() {
        if is_word_char(text[i - 1]) != is_word_char(text[i]) {
            boundaries.push(i);
        }
    }

    // Always include end
    boundaries.push(text.len());

    boundaries
}

/// Iterator over words in text
///
/// Splits text by word boundaries, yielding only word sequences (not delimiters).
pub struct WordIterator<'a> {
    text: &'a [u8],
    pos: usize,
}

impl<'a> WordIterator<'a> {
    /// Create a new word iterator
    pub fn new(text: &'a [u8]) -> Self {
        Self { text, pos: 0 }
    }
}

impl<'a> Iterator for WordIterator<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        // Skip non-word characters
        while self.pos < self.text.len() && !is_word_char(self.text[self.pos]) {
            self.pos += 1;
        }

        if self.pos >= self.text.len() {
            return None;
        }

        let start = self.pos;

        // Collect word characters
        while self.pos < self.text.len() && is_word_char(self.text[self.pos]) {
            self.pos += 1;
        }

        Some(&self.text[start..self.pos])
    }
}

/// Iterate over words in text
///
/// # Examples
///
/// ```rust
/// use zipora::string::words;
///
/// let text = b"hello, world! test_123";
/// let word_list: Vec<_> = words(text).collect();
/// assert_eq!(word_list.len(), 3);
/// assert_eq!(word_list[0], b"hello");
/// assert_eq!(word_list[1], b"world");
/// assert_eq!(word_list[2], b"test_123");
/// ```
pub fn words(text: &[u8]) -> WordIterator<'_> {
    WordIterator::new(text)
}

/// Count words in text
///
/// # Examples
///
/// ```rust
/// use zipora::string::word_count;
///
/// assert_eq!(word_count(b"hello world"), 2);
/// assert_eq!(word_count(b"one-two-three"), 3);
/// assert_eq!(word_count(b""), 0);
/// ```
pub fn word_count(text: &[u8]) -> usize {
    words(text).count()
}

/// Find the word at a given position
///
/// Returns the byte range of the word containing the position,
/// or None if the position is not within a word.
///
/// # Examples
///
/// ```rust
/// use zipora::string::word_at_position;
///
/// let text = b"hello world";
/// assert_eq!(word_at_position(text, 2), Some((0, 5)));  // "hello"
/// assert_eq!(word_at_position(text, 8), Some((6, 11))); // "world"
/// assert_eq!(word_at_position(text, 5), None);          // space
/// ```
pub fn word_at_position(text: &[u8], pos: usize) -> Option<(usize, usize)> {
    if pos >= text.len() || !is_word_char(text[pos]) {
        return None;
    }

    // Find start of word
    let mut start = pos;
    while start > 0 && is_word_char(text[start - 1]) {
        start -= 1;
    }

    // Find end of word
    let mut end = pos;
    while end < text.len() && is_word_char(text[end]) {
        end += 1;
    }

    Some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_word_char() {
        // Letters
        assert!(is_word_char(b'a'));
        assert!(is_word_char(b'z'));
        assert!(is_word_char(b'A'));
        assert!(is_word_char(b'Z'));

        // Digits
        assert!(is_word_char(b'0'));
        assert!(is_word_char(b'9'));

        // Underscore
        assert!(is_word_char(b'_'));

        // Non-word characters
        assert!(!is_word_char(b' '));
        assert!(!is_word_char(b'-'));
        assert!(!is_word_char(b'.'));
        assert!(!is_word_char(b','));
        assert!(!is_word_char(b'!'));
    }

    #[test]
    fn test_is_whitespace() {
        assert!(is_whitespace(b' '));
        assert!(is_whitespace(b'\t'));
        assert!(is_whitespace(b'\n'));
        assert!(is_whitespace(b'\r'));

        assert!(!is_whitespace(b'a'));
        assert!(!is_whitespace(b'0'));
    }

    #[test]
    fn test_is_punctuation() {
        assert!(is_punctuation(b'.'));
        assert!(is_punctuation(b','));
        assert!(is_punctuation(b'!'));
        assert!(is_punctuation(b'?'));
        assert!(is_punctuation(b'"'));

        assert!(!is_punctuation(b'a'));
        assert!(!is_punctuation(b'0'));
        assert!(!is_punctuation(b' '));
    }

    #[test]
    fn test_is_word_boundary() {
        let text = b"hello world";

        // Start and end are boundaries
        assert!(is_word_boundary(text, 0));
        assert!(is_word_boundary(text, 11));

        // Boundary between word and space
        assert!(is_word_boundary(text, 5));  // After "hello"
        assert!(is_word_boundary(text, 6));  // Before "world"

        // Middle of words - not boundaries
        assert!(!is_word_boundary(text, 1));  // Inside "hello"
        assert!(!is_word_boundary(text, 7));  // Inside "world"
    }

    #[test]
    fn test_is_word_boundary_empty() {
        assert!(is_word_boundary(b"", 0));
    }

    #[test]
    fn test_find_word_boundaries() {
        let text = b"hello world";
        let boundaries = find_word_boundaries(text);
        assert_eq!(boundaries, vec![0, 5, 6, 11]);
    }

    #[test]
    fn test_find_word_boundaries_multiple() {
        let text = b"a-b-c";
        let boundaries = find_word_boundaries(text);
        assert_eq!(boundaries, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_find_word_boundaries_empty() {
        let boundaries = find_word_boundaries(b"");
        assert_eq!(boundaries, vec![0]);
    }

    #[test]
    fn test_words_iterator() {
        let text = b"hello, world! test_123";
        let word_list: Vec<_> = words(text).collect();

        assert_eq!(word_list.len(), 3);
        assert_eq!(word_list[0], b"hello");
        assert_eq!(word_list[1], b"world");
        assert_eq!(word_list[2], b"test_123");
    }

    #[test]
    fn test_words_empty() {
        let word_list: Vec<_> = words(b"").collect();
        assert!(word_list.is_empty());
    }

    #[test]
    fn test_words_only_delimiters() {
        let word_list: Vec<_> = words(b"   ,,,   ").collect();
        assert!(word_list.is_empty());
    }

    #[test]
    fn test_word_count() {
        assert_eq!(word_count(b"hello world"), 2);
        assert_eq!(word_count(b"one-two-three"), 3);
        assert_eq!(word_count(b"  spaced  out  "), 2);
        assert_eq!(word_count(b""), 0);
        assert_eq!(word_count(b"single"), 1);
    }

    #[test]
    fn test_word_at_position() {
        let text = b"hello world";

        // In first word
        assert_eq!(word_at_position(text, 0), Some((0, 5)));
        assert_eq!(word_at_position(text, 2), Some((0, 5)));
        assert_eq!(word_at_position(text, 4), Some((0, 5)));

        // In second word
        assert_eq!(word_at_position(text, 6), Some((6, 11)));
        assert_eq!(word_at_position(text, 8), Some((6, 11)));
        assert_eq!(word_at_position(text, 10), Some((6, 11)));

        // On space
        assert_eq!(word_at_position(text, 5), None);

        // Beyond end
        assert_eq!(word_at_position(text, 20), None);
    }

    #[test]
    fn test_word_at_position_underscore() {
        let text = b"test_word";
        assert_eq!(word_at_position(text, 4), Some((0, 9)));
    }
}
