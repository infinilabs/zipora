//! Comprehensive SIMD Base64 Tests
//!
//! Tests to verify that all SIMD implementations work correctly and produce
//! identical results to the scalar implementation.

use zipora::system::base64::{AdaptiveBase64, Base64Config, SimdImplementation};

/// Test vectors for Base64 encoding/decoding
const TEST_VECTORS: &[(&[u8], &str)] = &[
    (b"", ""),
    (b"f", "Zg=="),
    (b"fo", "Zm8="),
    (b"foo", "Zm9v"),
    (b"foob", "Zm9vYg=="),
    (b"fooba", "Zm9vYmE="),
    (b"foobar", "Zm9vYmFy"),
    (b"Hello, World!", "SGVsbG8sIFdvcmxkIQ=="),
    (b"The quick brown fox jumps over the lazy dog", "VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw=="),
    // Test cases with special characters
    (b"Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure.",
     "TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4="),
    // Binary data with all byte values  
    (b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x20\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\x30\x31\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x3e\x3f\x40\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f\x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x5b\x5c\x5d\x5e\x5f\x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f\x70\x71\x72\x73\x74\x75\x76\x77\x78\x79\x7a\x7b\x7c\x7d\x7e\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff", 
     "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0BBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6e3x9fn+AgYKDhIWGh4iJiouMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/wMHCw8TFxsfIycrLzM3Oz9DR0tPU1dbX2Nna29zd3t/g4eLj5OXm5+jp6uvs7e7v8PHy8/T19vf4+fr7/P3+/w=="),
];

/// Test the scalar implementation as a reference
#[test]
fn test_scalar_implementation() {
    let config = Base64Config {
        force_implementation: Some(SimdImplementation::Scalar),
        ..Default::default()
    };
    let codec = AdaptiveBase64::with_config(config);

    for (input, expected) in TEST_VECTORS {
        let encoded = codec.encode(*input);
        assert_eq!(encoded, *expected, "Scalar encoding mismatch for input: {:?}", input);

        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, *input, "Scalar decoding mismatch for input: {:?}", input);
    }
}

/// Test SSE4.2 implementation
#[cfg(target_arch = "x86_64")]
#[test]
fn test_sse42_implementation() {
    if !is_x86_feature_detected!("sse4.2") {
        println!("SSE4.2 not available, skipping test");
        return;
    }

    let config = Base64Config {
        force_implementation: Some(SimdImplementation::SSE42),
        ..Default::default()
    };
    let codec = AdaptiveBase64::with_config(config);

    for (input, expected) in TEST_VECTORS {
        let encoded = codec.encode(*input);
        assert_eq!(encoded, *expected, "SSE4.2 encoding mismatch for input: {:?}", input);

        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, *input, "SSE4.2 decoding mismatch for input: {:?}", input);
    }
}

/// Test AVX2 implementation
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_implementation() {
    if !is_x86_feature_detected!("avx2") {
        println!("AVX2 not available, skipping test");
        return;
    }

    let config = Base64Config {
        force_implementation: Some(SimdImplementation::AVX2),
        ..Default::default()
    };
    let codec = AdaptiveBase64::with_config(config);

    for (input, expected) in TEST_VECTORS {
        let encoded = codec.encode(*input);
        assert_eq!(encoded, *expected, "AVX2 encoding mismatch for input: {:?}", input);

        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, *input, "AVX2 decoding mismatch for input: {:?}", input);
    }
}

/// Test AVX-512 implementation
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512_implementation() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
        println!("AVX-512 not available, skipping test");
        return;
    }

    let config = Base64Config {
        force_implementation: Some(SimdImplementation::AVX512),
        ..Default::default()
    };
    let codec = AdaptiveBase64::with_config(config);

    for (input, expected) in TEST_VECTORS {
        let encoded = codec.encode(*input);
        assert_eq!(encoded, *expected, "AVX-512 encoding mismatch for input: {:?}", input);

        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, *input, "AVX-512 decoding mismatch for input: {:?}", input);
    }
}

/// Test NEON implementation
#[cfg(target_arch = "aarch64")]
#[test]
fn test_neon_implementation() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        println!("NEON not available, skipping test");
        return;
    }

    let config = Base64Config {
        force_implementation: Some(SimdImplementation::NEON),
        ..Default::default()
    };
    let codec = AdaptiveBase64::with_config(config);

    for (input, expected) in TEST_VECTORS {
        let encoded = codec.encode(*input);
        assert_eq!(encoded, *expected, "NEON encoding mismatch for input: {:?}", input);

        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, *input, "NEON decoding mismatch for input: {:?}", input);
    }
}

/// Test adaptive selection works correctly
#[test]
fn test_adaptive_selection() {
    let codec = AdaptiveBase64::new();
    
    // Test that the codec selects an appropriate implementation
    let implementation = codec.get_implementation();
    
    // Should not be scalar unless no SIMD is available
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        // On modern x86_64 or ARM64, we should get some form of SIMD
        match implementation {
            SimdImplementation::Scalar => {
                // This is okay if the system doesn't support any SIMD
                println!("No SIMD support detected, using scalar implementation");
            }
            _ => {
                println!("Using SIMD implementation: {:?}", implementation);
            }
        }
    }
    
    // Test encoding/decoding works regardless of selection
    for (input, expected) in TEST_VECTORS.iter().take(5) {
        let encoded = codec.encode(*input);
        assert_eq!(encoded, *expected);

        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, *input);
    }
}

/// Test URL-safe encoding
#[test]
fn test_url_safe_encoding() {
    let config = Base64Config {
        url_safe: true,
        padding: false,
        force_implementation: None,
    };
    let codec = AdaptiveBase64::with_config(config);

    // Test data that normally produces + and / characters
    let test_cases: &[(&[u8], &str)] = &[
        (b"?>", "Pz4"),
        (b"??", "Pz8"),
        (b">?>", "Pj8-"),
        (b">>>", "Pj4-"),
    ];

    for (input, expected) in test_cases {
        let encoded = codec.encode(*input);
        assert_eq!(encoded, *expected);
        
        // Should not contain + or / or =
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        assert!(!encoded.contains('='));

        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, *input);
    }
}

/// Test error handling
#[test]
fn test_error_handling() {
    let codec = AdaptiveBase64::new();

    // Test invalid characters
    let invalid_inputs = &[
        "Invalid!@#$%",
        "ABC@",
        "Hello World!", // Spaces are invalid
        "ABC\n",        // Newlines are invalid (in strict mode)
    ];

    for invalid_input in invalid_inputs {
        let result = codec.decode(invalid_input);
        assert!(result.is_err(), "Should have failed for input: {}", invalid_input);
    }
}

/// Test large data to ensure all SIMD paths are exercised
#[test]
fn test_large_data() {
    let codec = AdaptiveBase64::new();
    
    // Create test data that will exercise all SIMD chunk sizes
    let sizes = &[
        12,   // SSE4.2 chunk size
        24,   // AVX2 chunk size  
        48,   // AVX-512 chunk size
        100,  // Mixed chunk processing
        1000, // Multiple chunks
        5000, // Large data
    ];

    for &size in sizes {
        // Create test data with known pattern
        let input: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        
        let encoded = codec.encode(&input);
        let decoded = codec.decode(&encoded).unwrap();
        
        assert_eq!(decoded, input, "Large data test failed for size: {}", size);
    }
}

/// Compare all implementations produce identical results
#[test]
fn test_implementation_consistency() {
    let implementations = vec![
        SimdImplementation::Scalar,
        #[cfg(target_arch = "x86_64")]
        SimdImplementation::SSE42,
        #[cfg(target_arch = "x86_64")]
        SimdImplementation::AVX2,
        #[cfg(target_arch = "x86_64")]
        SimdImplementation::AVX512,
        #[cfg(target_arch = "aarch64")]
        SimdImplementation::NEON,
    ];

    // Test data that exercises different scenarios
    let test_inputs = &[
        &b""[..],
        &b"A"[..],
        b"AB",
        b"ABC",
        b"ABCD",
        b"ABCDE",
        b"ABCDEF",
        b"The quick brown fox jumps over the lazy dog",
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
    ];

    for input in test_inputs {
        let mut results = Vec::new();
        
        for &impl_type in &implementations {
            let config = Base64Config {
                force_implementation: Some(impl_type),
                ..Default::default()
            };
            let codec = AdaptiveBase64::with_config(config);
            
            // Skip if implementation is not available
            match impl_type {
                #[cfg(target_arch = "x86_64")]
                SimdImplementation::SSE42 if !is_x86_feature_detected!("sse4.2") => continue,
                #[cfg(target_arch = "x86_64")]
                SimdImplementation::AVX2 if !is_x86_feature_detected!("avx2") => continue,
                #[cfg(target_arch = "x86_64")]
                SimdImplementation::AVX512 if !is_x86_feature_detected!("avx512f") => continue,
                #[cfg(target_arch = "aarch64")]
                SimdImplementation::NEON if !std::arch::is_aarch64_feature_detected!("neon") => continue,
                _ => {}
            }
            
            let encoded = codec.encode(*input);
            let decoded = codec.decode(&encoded).unwrap();
            
            results.push((impl_type, encoded, decoded));
        }

        // Ensure all implementations produce the same results
        if let Some((first_impl, first_encoded, first_decoded)) = results.first() {
            for (impl_type, encoded, decoded) in &results[1..] {
                assert_eq!(
                    encoded, first_encoded,
                    "Encoding mismatch between {:?} and {:?} for input: {:?}",
                    impl_type, first_impl, input
                );
                assert_eq!(
                    decoded, first_decoded,
                    "Decoding mismatch between {:?} and {:?} for input: {:?}",
                    impl_type, first_impl, input
                );
            }
        }
    }
}

/// Test convenience functions
#[test]
fn test_convenience_functions() {
    use zipora::system::base64::{base64_encode_simd, base64_decode_simd, 
                        base64_encode_url_safe_simd, base64_decode_url_safe_simd};

    let input = b"Hello, SIMD World!";
    
    // Test standard convenience functions
    let encoded = base64_encode_simd(input);
    let decoded = base64_decode_simd(&encoded).unwrap();
    assert_eq!(decoded, input);
    
    // Test URL-safe convenience functions
    let encoded_url = base64_encode_url_safe_simd(input);
    let decoded_url = base64_decode_url_safe_simd(&encoded_url).unwrap();
    assert_eq!(decoded_url, input);
    
    // URL-safe should not have padding
    assert!(!encoded_url.contains('='));
}

/// Benchmark comparison between implementations
#[test]
fn test_performance_comparison() {
    use std::time::Instant;
    
    let mut test_data = Vec::with_capacity(100000);
    for i in 0..100000 {
        test_data.push((i % 256) as u8);
    }
    let iterations = 100;
    
    let implementations = vec![
        ("Scalar", SimdImplementation::Scalar),
        #[cfg(target_arch = "x86_64")]
        ("SSE4.2", SimdImplementation::SSE42),
        #[cfg(target_arch = "x86_64")]
        ("AVX2", SimdImplementation::AVX2),
        #[cfg(target_arch = "x86_64")]
        ("AVX512", SimdImplementation::AVX512),
        #[cfg(target_arch = "aarch64")]
        ("NEON", SimdImplementation::NEON),
    ];
    
    println!("Performance comparison (encoding {} bytes, {} iterations):", test_data.len(), iterations);
    
    for (name, impl_type) in implementations {
        // Skip unavailable implementations
        match impl_type {
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::SSE42 if !is_x86_feature_detected!("sse4.2") => continue,
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::AVX2 if !is_x86_feature_detected!("avx2") => continue,
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::AVX512 if !is_x86_feature_detected!("avx512f") => continue,
            #[cfg(target_arch = "aarch64")]
            SimdImplementation::NEON if !std::arch::is_aarch64_feature_detected!("neon") => continue,
            _ => {}
        }
        
        let config = Base64Config {
            force_implementation: Some(impl_type),
            ..Default::default()
        };
        let codec = AdaptiveBase64::with_config(config);
        
        // Warmup
        for _ in 0..10 {
            let _ = codec.encode(&test_data);
        }
        
        // Measure encoding performance
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = codec.encode(&test_data);
        }
        let encode_duration = start.elapsed();
        
        // Test decoding performance with encoded data
        let encoded_data = codec.encode(&test_data);
        
        // Warmup
        for _ in 0..10 {
            let _ = codec.decode(&encoded_data);
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = codec.decode(&encoded_data).unwrap();
        }
        let decode_duration = start.elapsed();
        
        println!("{}: Encode: {:.3}ms, Decode: {:.3}ms", 
                name, 
                encode_duration.as_secs_f64() * 1000.0,
                decode_duration.as_secs_f64() * 1000.0);
    }
}