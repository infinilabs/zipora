//! Base64 encoding/decoding tests
//!
//! Tests the base64 wrapper API using standard test vectors.

use zipora::system::base64::{AdaptiveBase64, Base64Config, SimdImplementation};

/// Test vectors for Base64 encoding/decoding (RFC 4648)
const TEST_VECTORS: &[(&[u8], &str)] = &[
    (b"", ""),
    (b"f", "Zg=="),
    (b"fo", "Zm8="),
    (b"foo", "Zm9v"),
    (b"foob", "Zm9vYg=="),
    (b"fooba", "Zm9vYmE="),
    (b"foobar", "Zm9vYmFy"),
    (b"Hello, World!", "SGVsbG8sIFdvcmxkIQ=="),
];

#[test]
fn test_standard_encode_decode() {
    let codec = AdaptiveBase64::new();
    for (input, expected) in TEST_VECTORS {
        let encoded = codec.encode(input);
        assert_eq!(&encoded, expected, "encoding {:?}", input);
        let decoded = codec.decode(expected).unwrap();
        assert_eq!(&decoded, input, "decoding {:?}", expected);
    }
}

#[test]
fn test_url_safe_alphabet() {
    let config = Base64Config {
        url_safe: true,
        padding: true,
        force_implementation: None,
    };
    let codec = AdaptiveBase64::with_config(config);
    // Data that produces + and / in standard encoding
    let data = b"\xfb\xff\xfe";
    let encoded = codec.encode(data);
    assert!(!encoded.contains('+'), "URL-safe should not contain +");
    assert!(!encoded.contains('/'), "URL-safe should not contain /");
    let decoded = codec.decode(&encoded).unwrap();
    assert_eq!(decoded, data);
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
    assert!(!encoded.contains('='), "no-pad mode should not contain =");
}

#[test]
fn test_roundtrip_binary_data() {
    let codec = AdaptiveBase64::new();
    let data: Vec<u8> = (0..=255).collect();
    let encoded = codec.encode(&data);
    let decoded = codec.decode(&encoded).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn test_invalid_input() {
    let codec = AdaptiveBase64::new();
    assert!(codec.decode("!!!not-base64!!!").is_err());
}

#[test]
fn test_selected_implementation() {
    let codec = AdaptiveBase64::new();
    // Crate-based impl always reports Scalar
    assert_eq!(codec.selected_implementation(), SimdImplementation::Scalar);
}
