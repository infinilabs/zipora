//! Base64 encoding/decoding - thin wrapper around the `base64` crate.
//!
//! Previously 2,239 LOC of hand-rolled SIMD base64. Replaced with the
//! well-tested `base64` crate (already a dependency). The `base64` crate
//! is heavily optimized and widely used in production.

use crate::error::{Result, ZiporaError};
use base64::engine::general_purpose;
use base64::Engine;

/// Base64 encoding/decoding configuration
#[derive(Debug, Clone)]
pub struct Base64Config {
    /// Use URL-safe alphabet (RFC 4648 Section 5)
    pub url_safe: bool,
    /// Add padding characters
    pub padding: bool,
    /// Force specific SIMD implementation for testing (ignored, kept for API compat)
    pub force_implementation: Option<SimdImplementation>,
}

impl Default for Base64Config {
    fn default() -> Self {
        Self {
            url_safe: false,
            padding: true,
            force_implementation: None,
        }
    }
}

/// Available SIMD implementations (kept for API compatibility)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdImplementation {
    /// Scalar fallback
    Scalar,
    /// SSE 4.2
    SSE42,
    /// AVX2
    AVX2,
    /// AVX-512
    AVX512,
    /// ARM NEON
    NEON,
}

/// Adaptive Base64 encoder/decoder.
///
/// Wraps the `base64` crate with the same public API as the previous
/// hand-rolled implementation.
pub struct AdaptiveBase64 {
    config: Base64Config,
}

impl AdaptiveBase64 {
    /// Create a new adaptive Base64 codec with default configuration
    pub fn new() -> Self {
        Self::with_config(Base64Config::default())
    }

    /// Create a new adaptive Base64 codec with custom configuration
    pub fn with_config(config: Base64Config) -> Self {
        Self { config }
    }

    fn engine(&self) -> impl Engine + '_ {
        use base64::engine::general_purpose::*;
        match (self.config.url_safe, self.config.padding) {
            (false, true) => STANDARD,
            (false, false) => STANDARD_NO_PAD,
            (true, true) => URL_SAFE,
            (true, false) => URL_SAFE_NO_PAD,
        }
    }

    /// Encode binary data to Base64 string
    pub fn encode(&self, input: &[u8]) -> String {
        self.engine().encode(input)
    }

    /// Decode Base64 string to binary data
    pub fn decode(&self, input: &str) -> Result<Vec<u8>> {
        self.engine()
            .decode(input)
            .map_err(|e| ZiporaError::invalid_data(format!("base64 decode error: {}", e)))
    }

    /// Get the selected SIMD implementation (always Scalar for crate-based impl)
    pub fn selected_implementation(&self) -> SimdImplementation {
        SimdImplementation::Scalar
    }
}

impl Default for AdaptiveBase64 {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD Base64 encoder (wraps AdaptiveBase64)
pub struct SimdBase64Encoder {
    codec: AdaptiveBase64,
}

impl SimdBase64Encoder {
    /// Create a new encoder
    pub fn new() -> Self {
        Self { codec: AdaptiveBase64::new() }
    }

    /// Create a new encoder with custom configuration
    pub fn with_config(config: Base64Config) -> Self {
        Self { codec: AdaptiveBase64::with_config(config) }
    }

    /// Encode binary data to Base64 string
    pub fn encode(&self, input: &[u8]) -> String {
        self.codec.encode(input)
    }
}

impl Default for SimdBase64Encoder {
    fn default() -> Self { Self::new() }
}

/// SIMD Base64 decoder (wraps AdaptiveBase64)
pub struct SimdBase64Decoder {
    codec: AdaptiveBase64,
}

impl SimdBase64Decoder {
    /// Create a new decoder
    pub fn new() -> Self {
        Self { codec: AdaptiveBase64::new() }
    }

    /// Create a new decoder with custom configuration
    pub fn with_config(config: Base64Config) -> Self {
        Self { codec: AdaptiveBase64::with_config(config) }
    }

    /// Decode Base64 string to binary data
    pub fn decode(&self, input: &str) -> Result<Vec<u8>> {
        self.codec.decode(input)
    }
}

impl Default for SimdBase64Decoder {
    fn default() -> Self { Self::new() }
}

/// Convenience function: encode binary data to Base64
pub fn base64_encode_simd(input: &[u8]) -> String {
    general_purpose::STANDARD.encode(input)
}

/// Convenience function: decode Base64 to binary data
pub fn base64_decode_simd(input: &str) -> Result<Vec<u8>> {
    general_purpose::STANDARD
        .decode(input)
        .map_err(|e| ZiporaError::invalid_data(format!("base64 decode error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let codec = AdaptiveBase64::new();
        let data = b"Hello, World!";
        let encoded = codec.encode(data);
        assert_eq!(encoded, "SGVsbG8sIFdvcmxkIQ==");
        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_url_safe() {
        let config = Base64Config {
            url_safe: true,
            padding: true,
            force_implementation: None,
        };
        let codec = AdaptiveBase64::with_config(config);
        let data = b"\xfb\xff\xfe";
        let encoded = codec.encode(data);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
    }

    #[test]
    fn test_no_padding() {
        let config = Base64Config {
            url_safe: false,
            padding: false,
            force_implementation: None,
        };
        let codec = AdaptiveBase64::with_config(config);
        let encoded = codec.encode(b"f");
        assert!(!encoded.contains('='));
    }

    #[test]
    fn test_convenience_functions() {
        let input = b"foobar";
        let encoded = base64_encode_simd(input);
        assert_eq!(encoded, "Zm9vYmFy");
        let decoded = base64_decode_simd(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encoder_decoder_types() {
        let encoder = SimdBase64Encoder::new();
        let decoder = SimdBase64Decoder::new();
        let data = b"test data 12345";
        let encoded = encoder.encode(data);
        let decoded = decoder.decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_empty_input() {
        let codec = AdaptiveBase64::new();
        assert_eq!(codec.encode(b""), "");
        assert_eq!(codec.decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_invalid_input() {
        let codec = AdaptiveBase64::new();
        assert!(codec.decode("!!!invalid!!!").is_err());
    }
}
