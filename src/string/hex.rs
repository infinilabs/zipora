//! Hexadecimal encoding and decoding utilities
//!
//! Provides efficient hex encoding and decoding functions.
//!
//! Ported from topling-zip's hex utilities.

use crate::error::{Result, ZiporaError};

/// Decode a hex character to its nibble value (0-15)
///
/// # Examples
///
/// ```rust
/// use zipora::string::hex_char_to_nibble;
///
/// assert_eq!(hex_char_to_nibble(b'0'), Some(0));
/// assert_eq!(hex_char_to_nibble(b'9'), Some(9));
/// assert_eq!(hex_char_to_nibble(b'a'), Some(10));
/// assert_eq!(hex_char_to_nibble(b'F'), Some(15));
/// assert_eq!(hex_char_to_nibble(b'g'), None);
/// ```
#[inline]
pub const fn hex_char_to_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Encode a nibble value (0-15) to a hex character (lowercase)
///
/// # Panics
///
/// Panics if value >= 16
#[inline]
pub const fn nibble_to_hex_lower(value: u8) -> u8 {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    HEX_CHARS[value as usize]
}

/// Encode a nibble value (0-15) to a hex character (uppercase)
///
/// # Panics
///
/// Panics if value >= 16
#[inline]
pub const fn nibble_to_hex_upper(value: u8) -> u8 {
    const HEX_CHARS: &[u8; 16] = b"0123456789ABCDEF";
    HEX_CHARS[value as usize]
}

/// Decode a hexadecimal string to bytes
///
/// The input must have an even number of characters.
/// Supports both uppercase and lowercase hex digits.
///
/// # Examples
///
/// ```rust
/// use zipora::string::hex_decode;
///
/// let result = hex_decode("48656c6c6f").unwrap();
/// assert_eq!(result, b"Hello");
///
/// let result = hex_decode("DEADBEEF").unwrap();
/// assert_eq!(result, vec![0xDE, 0xAD, 0xBE, 0xEF]);
/// ```
pub fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    hex_decode_bytes(hex.as_bytes())
}

/// Decode a hexadecimal byte slice to bytes
///
/// The input must have an even number of bytes.
/// Supports both uppercase and lowercase hex digits.
///
/// # Examples
///
/// ```rust
/// use zipora::string::hex_decode_bytes;
///
/// let result = hex_decode_bytes(b"48656c6c6f").unwrap();
/// assert_eq!(result, b"Hello");
/// ```
pub fn hex_decode_bytes(hex: &[u8]) -> Result<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return Err(ZiporaError::InvalidData {
            message: "hex string must have even length".to_string(),
        });
    }

    let mut result = Vec::with_capacity(hex.len() / 2);

    for chunk in hex.chunks_exact(2) {
        let high = hex_char_to_nibble(chunk[0]).ok_or_else(|| ZiporaError::InvalidData {
            message: format!("invalid hex character: {:?}", chunk[0] as char),
        })?;
        let low = hex_char_to_nibble(chunk[1]).ok_or_else(|| ZiporaError::InvalidData {
            message: format!("invalid hex character: {:?}", chunk[1] as char),
        })?;

        result.push((high << 4) | low);
    }

    Ok(result)
}

/// Decode hex string into an existing buffer
///
/// Returns the number of bytes written. The buffer must be at least `hex.len() / 2` bytes.
///
/// # Examples
///
/// ```rust
/// use zipora::string::hex_decode_to_slice;
///
/// let mut buf = [0u8; 5];
/// let len = hex_decode_to_slice(b"48656c6c6f", &mut buf).unwrap();
/// assert_eq!(len, 5);
/// assert_eq!(&buf, b"Hello");
/// ```
pub fn hex_decode_to_slice(hex: &[u8], output: &mut [u8]) -> Result<usize> {
    if hex.len() % 2 != 0 {
        return Err(ZiporaError::InvalidData {
            message: "hex string must have even length".to_string(),
        });
    }

    let output_len = hex.len() / 2;
    if output.len() < output_len {
        return Err(ZiporaError::OutOfBounds {
            index: output_len,
            size: output.len(),
        });
    }

    for (i, chunk) in hex.chunks_exact(2).enumerate() {
        let high = hex_char_to_nibble(chunk[0]).ok_or_else(|| ZiporaError::InvalidData {
            message: format!("invalid hex character: {:?}", chunk[0] as char),
        })?;
        let low = hex_char_to_nibble(chunk[1]).ok_or_else(|| ZiporaError::InvalidData {
            message: format!("invalid hex character: {:?}", chunk[1] as char),
        })?;

        output[i] = (high << 4) | low;
    }

    Ok(output_len)
}

/// Encode bytes to hexadecimal string (lowercase)
///
/// # Examples
///
/// ```rust
/// use zipora::string::hex_encode;
///
/// let result = hex_encode(b"Hello");
/// assert_eq!(result, "48656c6c6f");
/// ```
pub fn hex_encode(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(bytes.len() * 2);

    for &byte in bytes {
        result.push(nibble_to_hex_lower(byte >> 4) as char);
        result.push(nibble_to_hex_lower(byte & 0x0f) as char);
    }

    result
}

/// Encode bytes to hexadecimal string (uppercase)
///
/// # Examples
///
/// ```rust
/// use zipora::string::hex_encode_upper;
///
/// let result = hex_encode_upper(b"Hello");
/// assert_eq!(result, "48656C6C6F");
/// ```
pub fn hex_encode_upper(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(bytes.len() * 2);

    for &byte in bytes {
        result.push(nibble_to_hex_upper(byte >> 4) as char);
        result.push(nibble_to_hex_upper(byte & 0x0f) as char);
    }

    result
}

/// Encode bytes to hex bytes (lowercase)
///
/// # Examples
///
/// ```rust
/// use zipora::string::hex_encode_to_bytes;
///
/// let result = hex_encode_to_bytes(b"\xDE\xAD");
/// assert_eq!(result, b"dead");
/// ```
pub fn hex_encode_to_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(bytes.len() * 2);

    for &byte in bytes {
        result.push(nibble_to_hex_lower(byte >> 4));
        result.push(nibble_to_hex_lower(byte & 0x0f));
    }

    result
}

/// Encode bytes into an existing buffer
///
/// Returns the number of bytes written. The buffer must be at least `bytes.len() * 2` bytes.
///
/// # Examples
///
/// ```rust
/// use zipora::string::hex_encode_to_slice;
///
/// let mut buf = [0u8; 10];
/// let len = hex_encode_to_slice(b"Hello", &mut buf).unwrap();
/// assert_eq!(len, 10);
/// assert_eq!(&buf, b"48656c6c6f");
/// ```
pub fn hex_encode_to_slice(bytes: &[u8], output: &mut [u8]) -> Result<usize> {
    let output_len = bytes.len() * 2;
    if output.len() < output_len {
        return Err(ZiporaError::OutOfBounds {
            index: output_len,
            size: output.len(),
        });
    }

    for (i, &byte) in bytes.iter().enumerate() {
        output[i * 2] = nibble_to_hex_lower(byte >> 4);
        output[i * 2 + 1] = nibble_to_hex_lower(byte & 0x0f);
    }

    Ok(output_len)
}

/// Check if a string is valid hexadecimal
///
/// # Examples
///
/// ```rust
/// use zipora::string::is_valid_hex;
///
/// assert!(is_valid_hex("48656c6c6f"));
/// assert!(is_valid_hex("DEADBEEF"));
/// assert!(!is_valid_hex("hello"));
/// assert!(!is_valid_hex("123"));  // Odd length
/// ```
pub fn is_valid_hex(s: &str) -> bool {
    if s.len() % 2 != 0 {
        return false;
    }
    s.bytes().all(|c| hex_char_to_nibble(c).is_some())
}

/// Parse a single hex byte from two characters
///
/// # Examples
///
/// ```rust
/// use zipora::string::parse_hex_byte;
///
/// assert_eq!(parse_hex_byte(b'4', b'8'), Some(0x48));
/// assert_eq!(parse_hex_byte(b'f', b'f'), Some(0xff));
/// assert_eq!(parse_hex_byte(b'g', b'0'), None);
/// ```
#[inline]
pub fn parse_hex_byte(high: u8, low: u8) -> Option<u8> {
    let h = hex_char_to_nibble(high)?;
    let l = hex_char_to_nibble(low)?;
    Some((h << 4) | l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_char_to_nibble() {
        assert_eq!(hex_char_to_nibble(b'0'), Some(0));
        assert_eq!(hex_char_to_nibble(b'9'), Some(9));
        assert_eq!(hex_char_to_nibble(b'a'), Some(10));
        assert_eq!(hex_char_to_nibble(b'f'), Some(15));
        assert_eq!(hex_char_to_nibble(b'A'), Some(10));
        assert_eq!(hex_char_to_nibble(b'F'), Some(15));
        assert_eq!(hex_char_to_nibble(b'g'), None);
        assert_eq!(hex_char_to_nibble(b' '), None);
    }

    #[test]
    fn test_nibble_to_hex() {
        assert_eq!(nibble_to_hex_lower(0), b'0');
        assert_eq!(nibble_to_hex_lower(9), b'9');
        assert_eq!(nibble_to_hex_lower(10), b'a');
        assert_eq!(nibble_to_hex_lower(15), b'f');

        assert_eq!(nibble_to_hex_upper(0), b'0');
        assert_eq!(nibble_to_hex_upper(10), b'A');
        assert_eq!(nibble_to_hex_upper(15), b'F');
    }

    #[test]
    fn test_hex_decode() {
        let result = hex_decode("48656c6c6f").unwrap();
        assert_eq!(result, b"Hello");

        let result = hex_decode("DEADBEEF").unwrap();
        assert_eq!(result, vec![0xDE, 0xAD, 0xBE, 0xEF]);

        let result = hex_decode("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_hex_decode_mixed_case() {
        let result = hex_decode("DeAdBeEf").unwrap();
        assert_eq!(result, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_hex_decode_error_odd_length() {
        let result = hex_decode("123");
        assert!(result.is_err());
    }

    #[test]
    fn test_hex_decode_error_invalid_char() {
        let result = hex_decode("gg");
        assert!(result.is_err());
    }

    #[test]
    fn test_hex_decode_to_slice() {
        let mut buf = [0u8; 5];
        let len = hex_decode_to_slice(b"48656c6c6f", &mut buf).unwrap();
        assert_eq!(len, 5);
        assert_eq!(&buf, b"Hello");
    }

    #[test]
    fn test_hex_decode_to_slice_error_small_buffer() {
        let mut buf = [0u8; 2];
        let result = hex_decode_to_slice(b"48656c6c6f", &mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_hex_encode() {
        let result = hex_encode(b"Hello");
        assert_eq!(result, "48656c6c6f");

        let result = hex_encode(b"\xDE\xAD\xBE\xEF");
        assert_eq!(result, "deadbeef");

        let result = hex_encode(b"");
        assert!(result.is_empty());
    }

    #[test]
    fn test_hex_encode_upper() {
        let result = hex_encode_upper(b"Hello");
        assert_eq!(result, "48656C6C6F");

        let result = hex_encode_upper(b"\xDE\xAD\xBE\xEF");
        assert_eq!(result, "DEADBEEF");
    }

    #[test]
    fn test_hex_encode_to_bytes() {
        let result = hex_encode_to_bytes(b"\xDE\xAD");
        assert_eq!(result, b"dead");
    }

    #[test]
    fn test_hex_encode_to_slice() {
        let mut buf = [0u8; 10];
        let len = hex_encode_to_slice(b"Hello", &mut buf).unwrap();
        assert_eq!(len, 10);
        assert_eq!(&buf, b"48656c6c6f");
    }

    #[test]
    fn test_hex_encode_to_slice_error_small_buffer() {
        let mut buf = [0u8; 4];
        let result = hex_encode_to_slice(b"Hello", &mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_valid_hex() {
        assert!(is_valid_hex("48656c6c6f"));
        assert!(is_valid_hex("DEADBEEF"));
        assert!(is_valid_hex(""));

        assert!(!is_valid_hex("hello"));
        assert!(!is_valid_hex("123")); // Odd length
        assert!(!is_valid_hex("12gh"));
    }

    #[test]
    fn test_parse_hex_byte() {
        assert_eq!(parse_hex_byte(b'4', b'8'), Some(0x48));
        assert_eq!(parse_hex_byte(b'f', b'f'), Some(0xff));
        assert_eq!(parse_hex_byte(b'0', b'0'), Some(0x00));
        assert_eq!(parse_hex_byte(b'g', b'0'), None);
        assert_eq!(parse_hex_byte(b'0', b'g'), None);
    }

    #[test]
    fn test_roundtrip() {
        let original = b"Hello, World! \x00\xFF\x12\x34";
        let encoded = hex_encode(original);
        let decoded = hex_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }
}
