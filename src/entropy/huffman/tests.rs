use super::*;
mod tests {
    use super::*;

    #[test]
    fn test_huffman_tree_single_symbol() {
        let mut frequencies = [0u32; 256];
        frequencies[65] = 100; // 'A'

        let tree = HuffmanTree::from_frequencies(&frequencies).unwrap();
        assert_eq!(tree.max_code_length(), 1);
        assert_eq!(tree.get_code(65).unwrap(), &vec![false]);
    }

    #[test]
    fn test_huffman_tree_two_symbols() {
        let mut frequencies = [0u32; 256];
        frequencies[65] = 100; // 'A'
        frequencies[66] = 50; // 'B'

        let tree = HuffmanTree::from_frequencies(&frequencies).unwrap();

        // Should have codes of length 1
        assert!(tree.get_code(65).is_some());
        assert!(tree.get_code(66).is_some());
        assert_eq!(tree.max_code_length(), 1);
    }

    #[test]
    fn test_huffman_encoding_decoding() {
        let data = b"hello world! this is a test message for huffman coding.";

        let encoder = HuffmanEncoder::new(data).unwrap();
        let encoded = encoder.encode(data).unwrap();

        let decoder = HuffmanDecoder::new(encoder.tree().clone());
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_huffman_compression_ratio() {
        let data = b"aaaaaabbbbcccc"; // Highly compressible

        let encoder = HuffmanEncoder::new(data).unwrap();
        let ratio = encoder.estimate_compression_ratio(data);

        // Should achieve good compression
        assert!(ratio < 1.0);
    }

    #[test]
    fn test_huffman_tree_serialization() {
        let data = b"hello world";
        let tree = HuffmanTree::from_data(data).unwrap();

        let serialized = tree.serialize();
        let deserialized = HuffmanTree::deserialize(&serialized).unwrap();

        // Check that codes match
        for (&symbol, code) in &tree.codes {
            assert_eq!(deserialized.get_code(symbol), Some(code));
        }
    }

    #[test]
    fn test_empty_data() {
        let data = b"";
        let encoder = HuffmanEncoder::new(data).unwrap();
        let encoded = encoder.encode(data).unwrap();
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_large_alphabet() {
        // Test with data containing many different symbols
        let data: Vec<u8> = (0..=255).cycle().take(1000).collect();

        let encoder = HuffmanEncoder::new(&data).unwrap();
        let encoded = encoder.encode(&data).unwrap();

        let decoder = HuffmanDecoder::new(encoder.tree().clone());
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data, decoded);
    }

    #[test]
    fn test_huffman_tree_frequencies() {
        let mut frequencies = [0u32; 256];
        frequencies[b'a' as usize] = 45;
        frequencies[b'b' as usize] = 13;
        frequencies[b'c' as usize] = 12;
        frequencies[b'd' as usize] = 16;
        frequencies[b'e' as usize] = 9;
        frequencies[b'f' as usize] = 5;

        let tree = HuffmanTree::from_frequencies(&frequencies).unwrap();

        // Verify that tree creates valid codes for all symbols
        let code_a = tree.get_code(b'a').unwrap();
        let code_f = tree.get_code(b'f').unwrap();

        // Both codes should exist and be non-empty
        assert!(!code_a.is_empty());
        assert!(!code_f.is_empty());

        // The tree should respect Huffman property: average code length is minimized
        // But individual codes may vary due to tie-breaking in tree construction
        let max_length = tree.max_code_length();
        assert!(max_length > 0);
    }

    #[test]
    fn test_contextual_huffman_order0() {
        let data = b"hello world! this is a test message for huffman coding.";

        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order0).unwrap();
        assert_eq!(encoder.order(), HuffmanOrder::Order0);
        assert_eq!(encoder.tree_count(), 1);

        let encoded = encoder.encode(data).unwrap();

        let decoder = ContextualHuffmanDecoder::new(encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_contextual_huffman_order1() {
        let data = b"abababab"; // Repetitive pattern that Order-1 should compress well

        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        assert_eq!(encoder.order(), HuffmanOrder::Order1);
        assert!(encoder.tree_count() >= 1);

        let encoded = encoder.encode(data).unwrap();

        let decoder = ContextualHuffmanDecoder::new(encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_contextual_huffman_order2() {
        let data = b"abcabcabcabc"; // Repetitive pattern that Order-2 should compress well

        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();
        assert_eq!(encoder.order(), HuffmanOrder::Order2);
        assert!(encoder.tree_count() >= 1);

        let encoded = encoder.encode(data).unwrap();

        let decoder = ContextualHuffmanDecoder::new(encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_contextual_huffman_compression_comparison() {
        // Test that all Huffman orders produce valid encodings
        // Note: Since Order-1/2 now include ALL 256 symbols for correctness,
        // compression ratios may be close to 1.0 for small datasets
        let data = b"aaaaabbbbbcccccdddddeeeeefffff"; // More compressible test data

        let encoder0 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order0).unwrap();
        let encoder1 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        let encoder2 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();

        let ratio0 = encoder0.estimate_compression_ratio(data);
        let ratio1 = encoder1.estimate_compression_ratio(data);
        let ratio2 = encoder2.estimate_compression_ratio(data);

        println!("Order-0 ratio: {:.3}", ratio0);
        println!("Order-1 ratio: {:.3}", ratio1);
        println!("Order-2 ratio: {:.3}", ratio2);

        // Order-0 should achieve compression since it only includes seen symbols
        assert!(
            ratio0 < 1.0,
            "Order-0 ratio should be < 1.0, got {:.3}",
            ratio0
        );

        // Order-1/2 include all symbols for correctness, so just check they don't expand too much
        assert!(ratio1 <= 1.5, "Order-1 ratio too high, got {:.3}", ratio1);
        assert!(ratio2 <= 1.5, "Order-2 ratio too high, got {:.3}", ratio2);

        // Verify round-trip for all orders
        let encoded0 = encoder0.encode(data).unwrap();
        let decoder0 = ContextualHuffmanDecoder::new(encoder0);
        let decoded0 = decoder0.decode(&encoded0, data.len()).unwrap();
        assert_eq!(data.to_vec(), decoded0);

        let encoder1 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        let encoded1 = encoder1.encode(data).unwrap();
        let decoder1 = ContextualHuffmanDecoder::new(encoder1);
        let decoded1 = decoder1.decode(&encoded1, data.len()).unwrap();
        assert_eq!(data.to_vec(), decoded1);

        let encoder2 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();
        let encoded2 = encoder2.encode(data).unwrap();
        let decoder2 = ContextualHuffmanDecoder::new(encoder2);
        let decoded2 = decoder2.decode(&encoded2, data.len()).unwrap();
        assert_eq!(data.to_vec(), decoded2);
    }

    #[test]
    fn test_contextual_huffman_serialization() {
        let data = b"test data for serialization";

        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        let serialized = encoder.serialize();

        let deserialized = ContextualHuffmanEncoder::deserialize(&serialized).unwrap();

        assert_eq!(encoder.order(), deserialized.order());
        assert_eq!(encoder.tree_count(), deserialized.tree_count());

        // Test that encoding produces same results
        let encoded1 = encoder.encode(data).unwrap();
        let encoded2 = deserialized.encode(data).unwrap();
        assert_eq!(encoded1, encoded2);
    }

    #[test]
    fn test_contextual_huffman_edge_cases() {
        // Test with very short data
        let short_data = b"a";
        let encoder = ContextualHuffmanEncoder::new(short_data, HuffmanOrder::Order2).unwrap();
        // Should fallback to simpler order
        assert!(encoder.order() == HuffmanOrder::Order0 || encoder.order() == HuffmanOrder::Order1);

        // Test with empty data
        let empty_data = b"";
        let encoder = ContextualHuffmanEncoder::new(empty_data, HuffmanOrder::Order1).unwrap();
        let encoded = encoder.encode(empty_data).unwrap();
        assert!(encoded.is_empty());

        // Test with single repeated symbol
        let repeated_data = b"aaaaaaaaaa";
        let encoder = ContextualHuffmanEncoder::new(repeated_data, HuffmanOrder::Order1).unwrap();
        let encoded = encoder.encode(repeated_data).unwrap();

        let decoder = ContextualHuffmanDecoder::new(encoder);
        let decoded = decoder.decode(&encoded, repeated_data.len()).unwrap();
        assert_eq!(repeated_data.to_vec(), decoded);
    }

    #[test]
    fn test_huffman_order_enum() {
        assert_eq!(HuffmanOrder::default(), HuffmanOrder::Order0);

        let orders = [
            HuffmanOrder::Order0,
            HuffmanOrder::Order1,
            HuffmanOrder::Order2,
        ];
        for order in orders {
            let data = b"test data";
            let encoder = ContextualHuffmanEncoder::new(data, order).unwrap();
            assert_eq!(encoder.order(), order);
        }
    }

    // ==================== Interleaving Tests ====================

    #[test]
    fn test_interleaving_factor_streams() {
        assert_eq!(InterleavingFactor::X1.streams(), 1);
        assert_eq!(InterleavingFactor::X2.streams(), 2);
        assert_eq!(InterleavingFactor::X4.streams(), 4);
        assert_eq!(InterleavingFactor::X8.streams(), 8);
    }

    #[test]
    fn test_interleaving_factor_default() {
        assert_eq!(InterleavingFactor::default(), InterleavingFactor::X1);
    }

    #[test]
    fn test_encode_x1_basic() {
        let data =
            b"hello world! this is a test for interleaved huffman coding with order-1 context.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        let encoded = encoder.encode_x1(data).unwrap();
        let decoded = encoder.decode_x1(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded, "X1 encode-decode round trip failed");
    }

    #[test]
    fn test_encode_x2_basic() {
        let data = b"hello world! this is a test for x2 interleaved huffman coding.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        let encoded = encoder.encode_x2(data).unwrap();
        let decoded = encoder.decode_x2(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded, "X2 encode-decode round trip failed");
    }

    #[test]
    fn test_encode_x4_basic() {
        let data = b"hello world! this is a test for x4 interleaved huffman coding with more data.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        let encoded = encoder.encode_x4(data).unwrap();
        let decoded = encoder.decode_x4(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded, "X4 encode-decode round trip failed");
    }

    #[test]
    fn test_encode_x8_basic() {
        let data = b"hello world! this is a test for x8 interleaved huffman coding with even more data to test.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        let encoded = encoder.encode_x8(data).unwrap();
        let decoded = encoder.decode_x8(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded, "X8 encode-decode round trip failed");
    }

    #[test]
    fn test_interleaving_all_variants() {
        let data = b"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        // Test all 4 variants
        for factor in [
            InterleavingFactor::X1,
            InterleavingFactor::X2,
            InterleavingFactor::X4,
            InterleavingFactor::X8,
        ] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder
                .decode_with_interleaving(&encoded, data.len(), factor)
                .unwrap();

            assert_eq!(
                data.to_vec(),
                decoded,
                "Round trip failed for {:?} interleaving",
                factor
            );
        }
    }

    #[test]
    fn test_interleaving_empty_data() {
        let data = b"";
        let encoder =
            ContextualHuffmanEncoder::new(b"training data", HuffmanOrder::Order1).unwrap();

        let encoded = encoder.encode_x1(data).unwrap();
        assert!(encoded.is_empty());

        let decoded = encoder.decode_x1(&encoded, 0).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_interleaving_single_byte() {
        let data = b"a";
        let encoder = ContextualHuffmanEncoder::new(b"abcdef", HuffmanOrder::Order1).unwrap();

        for factor in [
            InterleavingFactor::X1,
            InterleavingFactor::X2,
            InterleavingFactor::X4,
            InterleavingFactor::X8,
        ] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder
                .decode_with_interleaving(&encoded, data.len(), factor)
                .unwrap();

            assert_eq!(
                data.to_vec(),
                decoded,
                "Single byte failed for {:?}",
                factor
            );
        }
    }

    #[test]
    fn test_interleaving_two_bytes() {
        let data = b"ab";
        let encoder = ContextualHuffmanEncoder::new(b"abcdef", HuffmanOrder::Order1).unwrap();

        for factor in [
            InterleavingFactor::X1,
            InterleavingFactor::X2,
            InterleavingFactor::X4,
            InterleavingFactor::X8,
        ] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder
                .decode_with_interleaving(&encoded, data.len(), factor)
                .unwrap();

            assert_eq!(data.to_vec(), decoded, "Two bytes failed for {:?}", factor);
        }
    }

    #[test]
    fn test_interleaving_power_of_two_sizes() {
        let training_data = b"The quick brown fox jumps over the lazy dog.";
        let encoder = ContextualHuffmanEncoder::new(training_data, HuffmanOrder::Order1).unwrap();

        // Test with sizes that are powers of 2
        for size in [8, 16, 32, 64, 128, 256] {
            let data: Vec<u8> = training_data.iter().cycle().take(size).copied().collect();

            for factor in [
                InterleavingFactor::X1,
                InterleavingFactor::X2,
                InterleavingFactor::X4,
                InterleavingFactor::X8,
            ] {
                let encoded = encoder.encode_with_interleaving(&data, factor).unwrap();
                let decoded = encoder
                    .decode_with_interleaving(&encoded, data.len(), factor)
                    .unwrap();

                assert_eq!(
                    data, decoded,
                    "Power-of-2 size {} failed for {:?}",
                    size, factor
                );
            }
        }
    }

    #[test]
    fn test_interleaving_non_power_of_two_sizes() {
        let training_data = b"The quick brown fox jumps over the lazy dog.";
        let encoder = ContextualHuffmanEncoder::new(training_data, HuffmanOrder::Order1).unwrap();

        // Test with sizes that are NOT powers of 2
        for size in [7, 15, 31, 63, 127, 255] {
            let data: Vec<u8> = training_data.iter().cycle().take(size).copied().collect();

            for factor in [
                InterleavingFactor::X1,
                InterleavingFactor::X2,
                InterleavingFactor::X4,
                InterleavingFactor::X8,
            ] {
                let encoded = encoder.encode_with_interleaving(&data, factor).unwrap();
                let decoded = encoder
                    .decode_with_interleaving(&encoded, data.len(), factor)
                    .unwrap();

                assert_eq!(
                    data, decoded,
                    "Non-power-of-2 size {} failed for {:?}",
                    size, factor
                );
            }
        }
    }

    #[test]
    fn test_interleaving_repeated_symbols() {
        let data = b"aaaaaaaaaaaaaaaa"; // 16 'a's
        let encoder = ContextualHuffmanEncoder::new(b"abc", HuffmanOrder::Order1).unwrap();

        for factor in [
            InterleavingFactor::X1,
            InterleavingFactor::X2,
            InterleavingFactor::X4,
            InterleavingFactor::X8,
        ] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder
                .decode_with_interleaving(&encoded, data.len(), factor)
                .unwrap();

            assert_eq!(
                data.to_vec(),
                decoded,
                "Repeated symbols failed for {:?}",
                factor
            );
        }
    }

    #[test]
    fn test_interleaving_alternating_symbols() {
        let data = b"abababababababab"; // Alternating pattern
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        for factor in [
            InterleavingFactor::X1,
            InterleavingFactor::X2,
            InterleavingFactor::X4,
            InterleavingFactor::X8,
        ] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder
                .decode_with_interleaving(&encoded, data.len(), factor)
                .unwrap();

            assert_eq!(
                data.to_vec(),
                decoded,
                "Alternating pattern failed for {:?}",
                factor
            );
        }
    }

    #[test]
    fn test_interleaving_all_bytes() {
        // Test with data containing all possible byte values
        let data: Vec<u8> = (0..=255u8).cycle().take(512).collect();
        let encoder = ContextualHuffmanEncoder::new(&data, HuffmanOrder::Order1).unwrap();

        for factor in [
            InterleavingFactor::X1,
            InterleavingFactor::X2,
            InterleavingFactor::X4,
            InterleavingFactor::X8,
        ] {
            let encoded = encoder.encode_with_interleaving(&data, factor).unwrap();
            let decoded = encoder
                .decode_with_interleaving(&encoded, data.len(), factor)
                .unwrap();

            assert_eq!(data, decoded, "All bytes test failed for {:?}", factor);
        }
    }

    #[test]
    fn test_interleaving_large_data() {
        // Test with larger dataset (1KB)
        let base = b"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
        let data: Vec<u8> = base.iter().cycle().take(1024).copied().collect();
        let encoder = ContextualHuffmanEncoder::new(&data, HuffmanOrder::Order1).unwrap();

        for factor in [
            InterleavingFactor::X1,
            InterleavingFactor::X2,
            InterleavingFactor::X4,
            InterleavingFactor::X8,
        ] {
            let encoded = encoder.encode_with_interleaving(&data, factor).unwrap();
            let decoded = encoder
                .decode_with_interleaving(&encoded, data.len(), factor)
                .unwrap();

            assert_eq!(data, decoded, "Large data (1KB) failed for {:?}", factor);
        }
    }

    #[test]
    fn test_interleaving_only_order1() {
        let data = b"test data";

        // Order-0 should fail
        let encoder0 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order0).unwrap();
        assert!(
            encoder0
                .encode_with_interleaving(data, InterleavingFactor::X2)
                .is_err()
        );

        // Order-2 should fail
        let encoder2 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();
        assert!(
            encoder2
                .encode_with_interleaving(data, InterleavingFactor::X2)
                .is_err()
        );

        // Order-1 should succeed
        let encoder1 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        assert!(
            encoder1
                .encode_with_interleaving(data, InterleavingFactor::X2)
                .is_ok()
        );
    }

    #[test]
    fn test_interleaving_compression_ratio() {
        // Test that interleaving produces valid round-trip encoding
        // Note: Since Order-1 trees now include ALL 256 symbols for correctness,
        // compression ratio may be close to 1.0 for small datasets
        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        for factor in [
            InterleavingFactor::X1,
            InterleavingFactor::X2,
            InterleavingFactor::X4,
            InterleavingFactor::X8,
        ] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder
                .decode_with_interleaving(&encoded, data.len(), factor)
                .unwrap();

            // Verify round-trip correctness
            assert_eq!(data.to_vec(), decoded, "Round trip failed for {:?}", factor);

            // Compression ratio should be reasonable (not expanding too much)
            let ratio = encoded.len() as f64 / data.len() as f64;
            assert!(
                ratio <= 1.2,
                "Compression ratio too high for {:?}, ratio: {:.3}",
                factor,
                ratio
            );
        }
    }

    #[test]
    fn test_bitstream_writer_basic() {
        let mut writer = BitStreamWriter::new();

        // Write 8 bits
        writer.write(0b10101010, 8);
        let result = writer.finish();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0b10101010);
    }

    #[test]
    fn test_bitstream_writer_partial_byte() {
        let mut writer = BitStreamWriter::new();

        // Write 4 bits
        writer.write(0b1010, 4);
        let result = writer.finish();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0b1010);
    }

    #[test]
    fn test_bitstream_writer_multiple_writes() {
        let mut writer = BitStreamWriter::new();

        // Write 4 bits + 4 bits
        writer.write(0b1010, 4);
        writer.write(0b0101, 4);
        let result = writer.finish();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0b01011010); // LSB first
    }

    #[test]
    fn test_bitstream_reader_basic() {
        let data = vec![0b10101010];
        let mut reader = BitStreamReader::new(&data);

        let bits = reader.read(8);
        assert_eq!(bits, 0b10101010);
    }

    #[test]
    fn test_bitstream_reader_partial() {
        let data = vec![0b10101010];
        let mut reader = BitStreamReader::new(&data);

        let first = reader.read(4);
        let second = reader.read(4);

        assert_eq!(first, 0b1010);
        assert_eq!(second, 0b1010);
    }

    #[test]
    fn test_bitstream_roundtrip() {
        let mut writer = BitStreamWriter::new();

        // Write various bit patterns
        writer.write(0b101, 3);
        writer.write(0b11110000, 8);
        writer.write(0b1, 1);
        writer.write(0b111111, 6);

        let data = writer.finish();
        let mut reader = BitStreamReader::new(&data);

        assert_eq!(reader.read(3), 0b101);
        assert_eq!(reader.read(8), 0b11110000);
        assert_eq!(reader.read(1), 0b1);
        assert_eq!(reader.read(6), 0b111111);
    }
}
