use super::*;
// test module mirrors the file's own name by convention
#[allow(clippy::module_inception)]
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

    /// Regression: the tree must be built from a *min*-heap.
    ///
    /// `HuffmanNode::cmp` used to reverse the frequency comparison while
    /// `from_frequencies` also wrapped every node in `Reverse`. The two
    /// cancelled out, so the heap popped the two *most* frequent nodes first
    /// and produced a degenerate chain in which the most frequent symbol got
    /// the longest code. With freqs 1000/100/10/1 that gave a=3, b=3, c=2,
    /// d=1 bits instead of the correct a=1, b=2, c=3, d=3.
    #[test]
    fn test_huffman_more_frequent_symbols_get_shorter_codes() {
        let mut frequencies = [0u32; 256];
        frequencies[b'a' as usize] = 1000;
        frequencies[b'b' as usize] = 100;
        frequencies[b'c' as usize] = 10;
        frequencies[b'd' as usize] = 1;

        let tree = HuffmanTree::from_frequencies(&frequencies).unwrap();
        let len = |s: u8| tree.get_code(s).expect("symbol present").len();

        assert_eq!(
            len(b'a'),
            1,
            "most frequent symbol must get the shortest code"
        );
        assert_eq!(len(b'b'), 2);
        assert_eq!(len(b'c'), 3);
        assert_eq!(len(b'd'), 3);

        // Code length must be monotonically non-increasing in frequency.
        assert!(len(b'a') <= len(b'b'));
        assert!(len(b'b') <= len(b'c'));
        assert!(len(b'c') <= len(b'd'));
    }

    /// Regression: a skewed input must compress to near its entropy bound.
    ///
    /// The inverted heap emitted 416 bytes for this 1111-byte input (weighted
    /// length 3321 bits); correct Huffman emits 1233 bits = 155 bytes.
    #[test]
    fn test_huffman_skewed_input_compresses_near_entropy_bound() {
        let mut data = Vec::new();
        data.extend(std::iter::repeat_n(b'a', 1000));
        data.extend(std::iter::repeat_n(b'b', 100));
        data.extend(std::iter::repeat_n(b'c', 10));
        data.push(b'd');

        let encoder = HuffmanEncoder::new(&data).unwrap();
        let encoded = encoder.encode(&data).unwrap();

        // Optimal is ceil(1233 / 8) = 155 bytes; allow a little slack for the
        // packing tail but stay far below the 416 the inverted tree produced.
        assert!(
            encoded.len() <= 160,
            "expected ~155 bytes for a skewed 1111-byte input, got {}",
            encoded.len()
        );

        let decoder = HuffmanDecoder::new(encoder.tree().clone());
        assert_eq!(decoder.decode(&encoded, data.len()).unwrap(), data);
    }

    /// Training corpus whose Order-1 context tree is maximally unbalanced.
    ///
    /// Fibonacci frequencies are the classic worst case for Huffman: every
    /// merge produces a node that is just smaller than the next leaf, so the
    /// tree degenerates into a chain and code lengths grow with the number of
    /// distinct symbols. Each symbol is preceded by `CTX` so all of the depth
    /// lands in a single Order-1 context.
    fn fibonacci_ladder_corpus(num_symbols: usize) -> Vec<u8> {
        const CTX: u8 = b'Z';

        let mut fibs = vec![1u32, 1];
        while fibs.len() < num_symbols {
            let next = fibs[fibs.len() - 1] + fibs[fibs.len() - 2];
            fibs.push(next);
        }

        let mut data = Vec::new();
        for (i, &count) in fibs.iter().enumerate() {
            // skip CTX itself so it keeps its role as the context byte
            let symbol = if (i as u8) >= CTX {
                i as u8 + 1
            } else {
                i as u8
            };
            for _ in 0..count {
                data.push(CTX);
                data.push(symbol);
            }
        }
        data
    }

    /// Regression: interleaved encoding must not truncate wide codes.
    ///
    /// `build_fast_symbol_table_inner` used to clamp every code to the low 16
    /// bits while the decoder kept walking the full tree, so any symbol whose
    /// code was wider than 16 bits was written as a *prefix* of its real code.
    /// That prefix is not prefix-free against the decoder's tree, so the whole
    /// remainder of the stream decoded to garbage - silently, with no error.
    ///
    /// The bug was masked until the tree was built from a real min-heap: the
    /// inverted heap made every Order-1 tree deeper than 64 levels, which
    /// tripped the fixed-length 8-bit fallback in `HuffmanTree`.
    #[test]
    fn test_interleaved_round_trip_with_codes_wider_than_16_bits() {
        const CTX: u8 = b'Z';

        let training = fibonacci_ladder_corpus(20);
        let encoder = ContextualHuffmanEncoder::new(&training, HuffmanOrder::Order1).unwrap();

        // Guard the premise: without a wide code this test proves nothing.
        let max_len = encoder.max_code_length_all_trees();
        assert!(
            max_len > 16,
            "training corpus should produce codes wider than 16 bits, got {max_len}"
        );

        // Exercise every (CTX, symbol) pair, including the deep, rare symbols
        // that carry the wide codes. A single stream is enough: one truncated
        // code desynchronises everything after it.
        let mut message = Vec::with_capacity(512);
        for symbol in 0..=255u8 {
            message.push(CTX);
            message.push(symbol);
        }

        let round_trips: [(usize, Vec<u8>); 3] = [
            (1, encoder.encode_x1(&message).unwrap()),
            (2, encoder.encode_x2(&message).unwrap()),
            (4, encoder.encode_x4(&message).unwrap()),
        ];
        for (factor, bytes) in round_trips {
            let decoded = match factor {
                1 => encoder.decode_x1(&bytes, message.len()).unwrap(),
                2 => encoder.decode_x2(&bytes, message.len()).unwrap(),
                _ => encoder.decode_x4(&bytes, message.len()).unwrap(),
            };
            assert_eq!(decoded, message, "x{factor} round trip corrupted the data");
        }
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

    #[test]
    fn test_encode_matches_the_bit_by_bit_reference() {
        // `HuffmanEncoder::encode` now shifts whole codes into a 128-bit
        // accumulator using a flat `(bits, len)` table instead of appending one
        // `bool` per bit and packing in a second pass. Pin the result against
        // the bit-at-a-time reference so neither the LSB-first bit order nor
        // the table build can drift.
        fn reference(tree: &HuffmanTree, data: &[u8]) -> Vec<u8> {
            let mut out = Vec::new();
            let mut current = 0u8;
            let mut bit_count = 0;
            for &symbol in data {
                for &bit in tree.get_code(symbol).unwrap() {
                    if bit {
                        current |= 1 << bit_count;
                    }
                    bit_count += 1;
                    if bit_count == 8 {
                        out.push(current);
                        current = 0;
                        bit_count = 0;
                    }
                }
            }
            if bit_count > 0 {
                out.push(current);
            }
            out
        }

        // A skewed alphabet gives codes of several different lengths; the
        // lengths below straddle byte boundaries in every phase.
        let alphabet: Vec<u8> = b"eeeeeeeeeettttttaaaaoooiiuuxyz".to_vec();
        for len in [1usize, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1000] {
            let data: Vec<u8> = (0..len).map(|i| alphabet[i % alphabet.len()]).collect();

            let encoder = HuffmanEncoder::new(&data).unwrap();
            let encoded = encoder.encode(&data).unwrap();
            assert_eq!(
                encoded,
                reference(encoder.tree(), &data),
                "encoded bytes diverge from the reference packing at len {len}"
            );

            let decoder = HuffmanDecoder::new(encoder.tree().clone());
            assert_eq!(decoder.decode(&encoded, data.len()).unwrap(), data);
        }
    }

    #[test]
    fn test_encode_rejects_a_symbol_outside_the_tree() {
        // The flat table uses a length of 0 for absent symbols; make sure that
        // is still an error and not a zero-length code silently written out.
        let mut frequencies = [0u32; 256];
        frequencies[b'a' as usize] = 10;
        frequencies[b'b' as usize] = 5;

        let encoder = HuffmanEncoder::from_frequencies(&frequencies).unwrap();
        assert!(encoder.encode(b"ab").is_ok());
        assert!(encoder.encode(b"abc").is_err());
    }
}
