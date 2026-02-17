//! Base64 encoding/decoding for the IO layer.
//!
//! Thin wrapper around the `base64` crate. Previously 872 LOC of hand-rolled
//! SIMD base64, now delegates to the well-tested `base64` crate.

use crate::error::{Result, ZiporaError};
use base64::engine::general_purpose;
use base64::Engine;

/// Encodes binary data to Base64 string
pub fn encode_base64(data: &[u8]) -> Result<String> {
    Ok(general_purpose::STANDARD.encode(data))
}

/// Decodes Base64 string to binary data
pub fn decode_base64(data: &str) -> Result<Vec<u8>> {
    general_purpose::STANDARD
        .decode(data)
        .map_err(|e| ZiporaError::invalid_data(format!("base64 decode error: {}", e)))
}

/// Encode base64 into a pre-allocated buffer. Returns bytes written.
pub fn encode_base64_to_buffer(data: &[u8], output: &mut [u8]) -> Result<usize> {
    let encoded = general_purpose::STANDARD.encode(data);
    let bytes = encoded.as_bytes();
    if bytes.len() > output.len() {
        return Err(ZiporaError::invalid_data("output buffer too small for base64 encoding"));
    }
    output[..bytes.len()].copy_from_slice(bytes);
    Ok(bytes.len())
}

/// Decode base64 from a buffer into a pre-allocated output. Returns bytes written.
pub fn decode_base64_from_buffer(data: &[u8], output: &mut [u8]) -> Result<usize> {
    let input_str = std::str::from_utf8(data)
        .map_err(|e| ZiporaError::invalid_data(format!("invalid UTF-8 in base64 input: {}", e)))?;
    let decoded = general_purpose::STANDARD
        .decode(input_str)
        .map_err(|e| ZiporaError::invalid_data(format!("base64 decode error: {}", e)))?;
    if decoded.len() > output.len() {
        return Err(ZiporaError::invalid_data("output buffer too small for base64 decoding"));
    }
    output[..decoded.len()].copy_from_slice(&decoded);
    Ok(decoded.len())
}

/// Calculate the encoded length for a given input length
pub fn calculate_encoded_len(input_len: usize) -> usize {
    // Standard base64: 4 output chars per 3 input bytes, rounded up, with padding
    ((input_len + 2) / 3) * 4
}

/// Calculate the maximum decoded length for a given encoded length
pub fn calculate_decoded_len(encoded_len: usize) -> usize {
    (encoded_len / 4) * 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let data = b"Hello, World!";
        let encoded = encode_base64(data).unwrap();
        assert_eq!(encoded, "SGVsbG8sIFdvcmxkIQ==");
        let decoded = decode_base64(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_buffer_operations() {
        let data = b"test";
        let mut enc_buf = vec![0u8; calculate_encoded_len(data.len())];
        let written = encode_base64_to_buffer(data, &mut enc_buf).unwrap();
        enc_buf.truncate(written);

        let mut dec_buf = vec![0u8; calculate_decoded_len(written)];
        let dec_written = decode_base64_from_buffer(&enc_buf, &mut dec_buf).unwrap();
        assert_eq!(&dec_buf[..dec_written], data);
    }

    #[test]
    fn test_empty() {
        assert_eq!(encode_base64(b"").unwrap(), "");
        assert_eq!(decode_base64("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_length_calculations() {
        assert_eq!(calculate_encoded_len(0), 0);
        assert_eq!(calculate_encoded_len(1), 4);
        assert_eq!(calculate_encoded_len(3), 4);
        assert_eq!(calculate_encoded_len(4), 8);
    }
}
