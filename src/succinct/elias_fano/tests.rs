use super::*;
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let ef = EliasFano::from_sorted(&[]);
        assert_eq!(ef.len(), 0);
        assert!(ef.is_empty());
        assert_eq!(ef.get(0), None);
        assert_eq!(ef.next_geq(0), None);
    }

    #[test]
    fn test_single() {
        let ef = EliasFano::from_sorted(&[42]);
        assert_eq!(ef.len(), 1);
        assert_eq!(ef.get(0), Some(42));
        assert_eq!(ef.next_geq(0), Some((0, 42)));
        assert_eq!(ef.next_geq(42), Some((0, 42)));
        assert_eq!(ef.next_geq(43), None);
    }

    #[test]
    fn test_basic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), 8);

        // Random access
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(ef.get(i), Some(v as u64), "get({}) failed", i);
        }
        assert_eq!(ef.get(8), None);
    }

    #[test]
    fn test_next_geq() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);

        // Exact matches
        assert_eq!(ef.next_geq(3), Some((0, 3)));
        assert_eq!(ef.next_geq(42), Some((5, 42)));
        assert_eq!(ef.next_geq(63), Some((7, 63)));

        // Between values
        assert_eq!(ef.next_geq(0), Some((0, 3)));
        assert_eq!(ef.next_geq(4), Some((1, 5)));
        assert_eq!(ef.next_geq(10), Some((2, 11)));
        assert_eq!(ef.next_geq(28), Some((4, 31)));
        assert_eq!(ef.next_geq(59), Some((7, 63)));

        // Past end
        assert_eq!(ef.next_geq(64), None);
        assert_eq!(ef.next_geq(1000), None);
    }

    #[test]
    fn test_iterator() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);

        let collected: Vec<u64> = ef.iter().collect();
        let expected: Vec<u64> = docs.iter().map(|&v| v as u64).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_consecutive() {
        let docs: Vec<u32> = (0..100).collect();
        let ef = EliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), 100);
        for i in 0..100 {
            assert_eq!(ef.get(i), Some(i as u64));
        }
        assert_eq!(ef.next_geq(50), Some((50, 50)));
    }

    #[test]
    fn test_sparse() {
        let docs: Vec<u32> = (0..100).map(|i| i * 1000).collect();
        let ef = EliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), 100);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(ef.get(i), Some(v as u64), "get({}) failed", i);
        }

        assert_eq!(ef.next_geq(500), Some((1, 1000)));
        assert_eq!(ef.next_geq(1000), Some((1, 1000)));
        assert_eq!(ef.next_geq(1001), Some((2, 2000)));
    }

    #[test]
    fn test_large_posting_list() {
        // Simulate a posting list: 10K doc IDs in universe of 1M
        let docs: Vec<u32> = (0..10000).map(|i| i * 100 + i % 7).collect();
        let ef = EliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), 10000);

        // Verify all elements
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(
                ef.get(i),
                Some(v as u64),
                "get({}) = {:?}, expected {}",
                i,
                ef.get(i),
                v
            );
        }

        // Verify iterator matches
        let from_iter: Vec<u64> = ef.iter().collect();
        assert_eq!(from_iter.len(), 10000);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(from_iter[i], v as u64);
        }

        // Space efficiency
        let bits_per_elem = ef.bits_per_element();
        assert!(
            bits_per_elem < 32.0,
            "Should be much less than 32 bits/elem, got {:.1}",
            bits_per_elem
        );
    }

    #[test]
    fn test_next_geq_scan() {
        // Simulate posting list intersection via next_geq
        let docs: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let ef = EliasFano::from_sorted(&docs);

        // Find all multiples of 30 via next_geq
        let mut target = 0u64;
        let mut found = Vec::new();
        while let Some((_, val)) = ef.next_geq(target) {
            if val % 30 == 0 {
                found.push(val as u32);
            }
            target = val + 1;
        }

        let expected: Vec<u32> = (0..1000).map(|i| i * 10).filter(|v| v % 30 == 0).collect();
        assert_eq!(found, expected);
    }

    #[test]
    fn test_space_efficiency() {
        // 10K elements in universe of 1M should use ~12 bits/elem
        let docs: Vec<u32> = (0..10000).map(|i| i * 100).collect();
        let ef = EliasFano::from_sorted(&docs);

        let bpe = ef.bits_per_element();
        eprintln!(
            "Elias-Fano: {} elements, {:.1} bits/elem, {} bytes total",
            ef.len(),
            bpe,
            ef.size_bytes()
        );

        // Theoretical: 2 + log2(1M/10K) ≈ 2 + 6.6 ≈ 8.6 bits
        // Practical overhead pushes to ~10-15 bits
        assert!(bpe < 20.0, "Too many bits per element: {:.1}", bpe);
    }

    #[test]
    fn test_max_values() {
        let docs = vec![0, u32::MAX / 2, u32::MAX];
        let ef = EliasFano::from_sorted(&docs);
        assert_eq!(ef.get(0), Some(0));
        assert_eq!(ef.get(1), Some(u32::MAX as u64 / 2));
        assert_eq!(ef.get(2), Some(u32::MAX as u64));
    }

    /// Performance — release only.
    #[test]
    fn test_performance_next_geq() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        let ef = EliasFano::from_sorted(&docs);

        let targets: Vec<u64> = (0..10000).map(|i| (i * 100) as u64).collect();

        let start = std::time::Instant::now();
        let mut found = 0usize;
        for _ in 0..100 {
            for &t in &targets {
                if ef.next_geq(t).is_some() {
                    found += 1;
                }
            }
        }
        #[allow(unused_variables)]
        let elapsed = start.elapsed();

        #[cfg(not(debug_assertions))]
        {
            let per_call = elapsed / (100 * targets.len() as u32);
            eprintln!("Elias-Fano next_geq: {:?}/call, {} found", per_call, found);
        }
    }

    // --- Cursor tests ---

    #[test]
    fn test_cursor_basic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);
        let mut cursor = ef.cursor();

        // Sequential access matches get()
        for (i, &v) in docs.iter().enumerate() {
            assert!(!cursor.is_exhausted());
            assert_eq!(cursor.index(), i);
            assert_eq!(cursor.current(), Some(v as u64), "cursor at {} failed", i);
            if i < docs.len() - 1 {
                assert!(cursor.advance());
            }
        }
        assert!(!cursor.advance()); // past end
        assert!(cursor.is_exhausted());
    }

    #[test]
    fn test_cursor_advance_to_geq() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);
        let mut cursor = ef.cursor();

        // Jump to >= 30
        assert!(cursor.advance_to_geq(30));
        assert_eq!(cursor.current(), Some(31));

        // Already >= 42 from current pos? No, 31 < 42
        assert!(cursor.advance_to_geq(42));
        assert_eq!(cursor.current(), Some(42));

        // Jump past end
        assert!(!cursor.advance_to_geq(100));
        assert!(cursor.is_exhausted());
    }

    #[test]
    fn test_cursor_matches_iterator() {
        let docs: Vec<u32> = (0..10000).map(|i| i * 10 + i % 7).collect();
        let ef = EliasFano::from_sorted(&docs);

        // Cursor should produce same values as iterator
        let from_iter: Vec<u64> = ef.iter().collect();
        let mut from_cursor = Vec::new();
        let mut cursor = ef.cursor();
        if let Some(v) = cursor.current() {
            from_cursor.push(v);
            while cursor.advance() {
                from_cursor.push(cursor.current().unwrap());
            }
        }

        assert_eq!(from_cursor.len(), from_iter.len());
        for (i, (&a, &b)) in from_cursor.iter().zip(from_iter.iter()).enumerate() {
            assert_eq!(a, b, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_cursor_reset() {
        let docs = vec![10, 20, 30];
        let ef = EliasFano::from_sorted(&docs);
        let mut cursor = ef.cursor();

        cursor.advance();
        cursor.advance();
        assert_eq!(cursor.current(), Some(30));

        cursor.reset();
        assert_eq!(cursor.current(), Some(10));
        assert_eq!(cursor.index(), 0);
    }

    /// Performance: cursor sequential vs get() sequential
    #[test]
    fn test_cursor_performance() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        #[allow(unused_variables)]
        let ef = EliasFano::from_sorted(&docs);

        #[cfg(not(debug_assertions))]
        {
            // get(i) sequential
            let start = std::time::Instant::now();
            let mut sum1 = 0u64;
            for _ in 0..10 {
                for i in 0..ef.len() {
                    sum1 += ef.get(i).unwrap();
                }
            }
            let get_time = start.elapsed();

            // cursor sequential
            let start = std::time::Instant::now();
            let mut sum2 = 0u64;
            for _ in 0..10 {
                let mut cursor = ef.cursor();
                sum2 += cursor.current().unwrap();
                while cursor.advance() {
                    sum2 += cursor.current().unwrap();
                }
            }
            let cursor_time = start.elapsed();

            assert_eq!(sum1, sum2, "cursor and get must produce same sum");

            let speedup = get_time.as_nanos() as f64 / cursor_time.as_nanos() as f64;
            eprintln!(
                "Sequential 100K: get={:?}, cursor={:?}, speedup={:.1}×",
                get_time, cursor_time, speedup
            );

            assert!(
                speedup > 2.0,
                "cursor should be at least 2× faster than get(), got {:.1}×",
                speedup
            );
        }
    }

    // ========================================================================
    // PartitionedEliasFano tests
    // ========================================================================

    #[test]
    fn test_pef_empty() {
        let pef = PartitionedEliasFano::from_sorted(&[]);
        assert_eq!(pef.len(), 0);
        assert!(pef.is_empty());
        assert_eq!(pef.get(0), None);
        assert_eq!(pef.next_geq(0), None);
    }

    #[test]
    fn test_pef_single() {
        let pef = PartitionedEliasFano::from_sorted(&[42]);
        assert_eq!(pef.len(), 1);
        assert_eq!(pef.get(0), Some(42));
        assert_eq!(pef.next_geq(0), Some((0, 42)));
        assert_eq!(pef.next_geq(42), Some((0, 42)));
        assert_eq!(pef.next_geq(43), None);
    }

    #[test]
    fn test_pef_basic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let pef = PartitionedEliasFano::from_sorted(&docs);
        assert_eq!(pef.len(), 8);

        // Random access
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(pef.get(i), Some(v as u64), "get({}) failed", i);
        }
        assert_eq!(pef.get(8), None);
    }

    #[test]
    fn test_pef_next_geq() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let pef = PartitionedEliasFano::from_sorted(&docs);

        // Exact matches
        assert_eq!(pef.next_geq(3), Some((0, 3)));
        assert_eq!(pef.next_geq(42), Some((5, 42)));
        assert_eq!(pef.next_geq(63), Some((7, 63)));

        // Between values
        assert_eq!(pef.next_geq(0), Some((0, 3)));
        assert_eq!(pef.next_geq(4), Some((1, 5)));
        assert_eq!(pef.next_geq(10), Some((2, 11)));
        assert_eq!(pef.next_geq(28), Some((4, 31)));
        assert_eq!(pef.next_geq(59), Some((7, 63)));

        // Past end
        assert_eq!(pef.next_geq(64), None);
        assert_eq!(pef.next_geq(1000), None);
    }

    #[test]
    fn test_pef_iterator() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let pef = PartitionedEliasFano::from_sorted(&docs);

        let collected: Vec<u64> = pef.iter().collect();
        let expected: Vec<u64> = docs.iter().map(|&v| v as u64).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_pef_consecutive() {
        let docs: Vec<u32> = (0..100).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        assert_eq!(pef.len(), 100);
        for i in 0..100 {
            assert_eq!(pef.get(i), Some(i as u64));
        }
        assert_eq!(pef.next_geq(50), Some((50, 50)));
    }

    #[test]
    fn test_pef_sparse() {
        let docs: Vec<u32> = (0..100).map(|i| i * 1000).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        assert_eq!(pef.len(), 100);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(pef.get(i), Some(v as u64), "get({}) failed", i);
        }

        assert_eq!(pef.next_geq(500), Some((1, 1000)));
        assert_eq!(pef.next_geq(1000), Some((1, 1000)));
        assert_eq!(pef.next_geq(1001), Some((2, 2000)));
    }

    #[test]
    fn test_pef_multi_chunk() {
        // Force multiple chunks: 300 elements > 128 chunk size
        let docs: Vec<u32> = (0..300).map(|i| i * 10 + i % 7).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        assert_eq!(pef.len(), 300);

        // Verify all elements via get()
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(pef.get(i), Some(v as u64), "get({}) failed", i);
        }

        // Verify iterator matches
        let from_iter: Vec<u64> = pef.iter().collect();
        assert_eq!(from_iter.len(), 300);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(from_iter[i], v as u64, "iter mismatch at {}", i);
        }

        // Verify next_geq across chunk boundaries
        for &v in &docs {
            let (idx, found) = pef.next_geq(v as u64).unwrap();
            assert_eq!(found, v as u64, "next_geq({}) returned {}", v, found);
            assert_eq!(pef.get(idx), Some(found));
        }
    }

    #[test]
    fn test_pef_matches_ef() {
        // Verify PEF produces same results as plain EF for all operations
        let docs: Vec<u32> = (0..500).map(|i| i * 10 + i % 7).collect();
        let ef = EliasFano::from_sorted(&docs);
        let pef = PartitionedEliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), pef.len());

        // get() must match
        for i in 0..docs.len() {
            assert_eq!(ef.get(i), pef.get(i), "get({}) mismatch", i);
        }

        // next_geq() must match
        for target in (0..5010).step_by(7) {
            assert_eq!(
                ef.next_geq(target),
                pef.next_geq(target),
                "next_geq({}) mismatch",
                target
            );
        }

        // iter() must match
        let ef_iter: Vec<u64> = ef.iter().collect();
        let pef_iter: Vec<u64> = pef.iter().collect();
        assert_eq!(ef_iter, pef_iter);
    }

    #[test]
    fn test_pef_large_posting_list() {
        // 10K doc IDs in universe of 1M — realistic posting list
        let docs: Vec<u32> = (0..10000).map(|i| i * 100 + i % 7).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        assert_eq!(pef.len(), 10000);

        // Sample verification
        for i in (0..10000).step_by(100) {
            assert_eq!(pef.get(i), Some(docs[i] as u64), "get({}) failed", i);
        }

        // Iterator must produce all elements
        let from_iter: Vec<u64> = pef.iter().collect();
        assert_eq!(from_iter.len(), 10000);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(from_iter[i], v as u64, "iter[{}] mismatch", i);
        }
    }

    #[test]
    fn test_pef_cursor_basic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let mut cursor = pef.cursor();

        for (i, &v) in docs.iter().enumerate() {
            assert!(!cursor.is_exhausted());
            assert_eq!(cursor.index(), i);
            assert_eq!(cursor.current(), Some(v as u64), "cursor at {} failed", i);
            if i < docs.len() - 1 {
                assert!(cursor.advance());
            }
        }
        assert!(!cursor.advance());
        assert!(cursor.is_exhausted());
    }

    #[test]
    fn test_pef_cursor_multi_chunk() {
        let docs: Vec<u32> = (0..300).map(|i| i * 10).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let mut cursor = pef.cursor();

        let mut collected = Vec::new();
        if let Some(v) = cursor.current() {
            collected.push(v);
            while cursor.advance() {
                collected.push(cursor.current().unwrap());
            }
        }

        let expected: Vec<u64> = docs.iter().map(|&v| v as u64).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_pef_cursor_advance_to_geq() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let mut cursor = pef.cursor();

        // Jump across chunks
        assert!(cursor.advance_to_geq(1500));
        assert_eq!(cursor.current(), Some(1500));

        assert!(cursor.advance_to_geq(3005));
        assert_eq!(cursor.current(), Some(3010));

        assert!(!cursor.advance_to_geq(5000));
        assert!(cursor.is_exhausted());
    }

    #[test]
    fn test_pef_cursor_reset() {
        let docs = vec![10, 20, 30];
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let mut cursor = pef.cursor();

        cursor.advance();
        cursor.advance();
        assert_eq!(cursor.current(), Some(30));

        cursor.reset();
        assert_eq!(cursor.current(), Some(10));
        assert_eq!(cursor.index(), 0);
    }

    #[test]
    fn test_pef_space_efficiency() {
        let docs: Vec<u32> = (0..10000).map(|i| i * 100).collect();
        let ef = EliasFano::from_sorted(&docs);
        let pef = PartitionedEliasFano::from_sorted(&docs);

        eprintln!(
            "EF:  {} elements, {:.1} bits/elem, {} bytes",
            ef.len(),
            ef.bits_per_element(),
            ef.size_bytes()
        );
        eprintln!(
            "PEF: {} elements, {:.1} bits/elem, {} bytes",
            pef.len(),
            pef.bits_per_element(),
            pef.size_bytes()
        );

        // PEF should be within 2x of EF space (chunk overhead)
        assert!(
            pef.bits_per_element() < ef.bits_per_element() * 2.5,
            "PEF too large: {:.1} vs EF {:.1}",
            pef.bits_per_element(),
            ef.bits_per_element()
        );
    }

    #[test]
    fn test_pef_max_values() {
        let docs = vec![0, u32::MAX / 2, u32::MAX];
        let pef = PartitionedEliasFano::from_sorted(&docs);
        assert_eq!(pef.get(0), Some(0));
        assert_eq!(pef.get(1), Some(u32::MAX as u64 / 2));
        assert_eq!(pef.get(2), Some(u32::MAX as u64));
    }

    #[test]
    fn test_pef_next_geq_scan() {
        let docs: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        let mut target = 0u64;
        let mut found = Vec::new();
        while let Some((_, val)) = pef.next_geq(target) {
            if val % 30 == 0 {
                found.push(val as u32);
            }
            target = val + 1;
        }

        let expected: Vec<u32> = (0..1000).map(|i| i * 10).filter(|v| v % 30 == 0).collect();
        assert_eq!(found, expected);
    }

    /// Performance: PEF next_geq vs plain EF next_geq — release only.
    #[test]
    fn test_pef_performance_next_geq() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        #[allow(unused_variables)]
        let ef = EliasFano::from_sorted(&docs);
        #[allow(unused_variables)]
        let pef = PartitionedEliasFano::from_sorted(&docs);

        #[allow(unused_variables)]
        let targets: Vec<u64> = (0..10000).map(|i| (i * 100) as u64).collect();

        #[cfg(not(debug_assertions))]
        {
            let mut sink = 0usize;

            // Warmup
            for &t in &targets {
                if ef.next_geq(t).is_some() {
                    sink += 1;
                }
                if pef.next_geq(t).is_some() {
                    sink += 1;
                }
            }

            let iterations = 100;

            let start = std::time::Instant::now();
            for _ in 0..iterations {
                for &t in &targets {
                    if ef.next_geq(t).is_some() {
                        sink += 1;
                    }
                }
            }
            let ef_time = start.elapsed();

            let start = std::time::Instant::now();
            for _ in 0..iterations {
                for &t in &targets {
                    if pef.next_geq(t).is_some() {
                        sink += 1;
                    }
                }
            }
            let pef_time = start.elapsed();

            let ef_ns = ef_time.as_nanos() as f64 / (iterations as f64 * targets.len() as f64);
            let pef_ns = pef_time.as_nanos() as f64 / (iterations as f64 * targets.len() as f64);
            let ratio = ef_ns / pef_ns;

            eprintln!(
                "next_geq 100K elements: EF={ef_ns:.1}ns, PEF={pef_ns:.1}ns, \
                 PEF is {ratio:.2}× (>1 = PEF faster) [sink={sink}]"
            );
        }
    }

    /// Performance: PEF cursor vs plain EF cursor — release only.
    #[test]
    fn test_pef_performance_cursor() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        #[allow(unused_variables)]
        let ef = EliasFano::from_sorted(&docs);
        #[allow(unused_variables)]
        let pef = PartitionedEliasFano::from_sorted(&docs);

        #[cfg(not(debug_assertions))]
        {
            let iterations = 10;

            // EF cursor
            let start = std::time::Instant::now();
            let mut sum1 = 0u64;
            for _ in 0..iterations {
                let mut cursor = ef.cursor();
                sum1 += cursor.current().unwrap();
                while cursor.advance() {
                    sum1 += cursor.current().unwrap();
                }
            }
            let ef_time = start.elapsed();

            // PEF cursor
            let start = std::time::Instant::now();
            let mut sum2 = 0u64;
            for _ in 0..iterations {
                let mut cursor = pef.cursor();
                sum2 += cursor.current().unwrap();
                while cursor.advance() {
                    sum2 += cursor.current().unwrap();
                }
            }
            let pef_time = start.elapsed();

            assert_eq!(sum1, sum2, "cursor sums must match");

            let ratio = ef_time.as_nanos() as f64 / pef_time.as_nanos() as f64;
            eprintln!(
                "Cursor 100K: EF={:?}, PEF={:?}, PEF is {ratio:.2}×",
                ef_time, pef_time
            );
        }
    }

    // ========================================================================
    // Batch Cursor tests
    // ========================================================================

    #[test]
    fn test_batch_cursor_ef_basic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);
        let mut bc = ef.batch_cursor();

        let mut collected = Vec::new();
        if let Some(v) = bc.current() {
            collected.push(v);
            while bc.advance() {
                collected.push(bc.current().unwrap());
            }
        }

        let expected: Vec<u64> = docs.iter().map(|&v| v as u64).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_batch_cursor_ef_matches_iter() {
        let docs: Vec<u32> = (0..10000).map(|i| i * 10 + i % 7).collect();
        let ef = EliasFano::from_sorted(&docs);

        let from_iter: Vec<u64> = ef.iter().collect();
        let mut from_batch = Vec::new();
        let mut bc = ef.batch_cursor();
        if let Some(v) = bc.current() {
            from_batch.push(v);
            while bc.advance() {
                from_batch.push(bc.current().unwrap());
            }
        }

        assert_eq!(from_batch.len(), from_iter.len());
        assert_eq!(from_batch, from_iter);
    }

    #[test]
    fn test_batch_cursor_ef_index() {
        let docs: Vec<u32> = (0..100).map(|i| i * 5).collect();
        let ef = EliasFano::from_sorted(&docs);
        let mut bc = ef.batch_cursor();

        for i in 0..100 {
            assert_eq!(bc.index(), i, "index mismatch at {}", i);
            assert_eq!(bc.current(), Some(docs[i] as u64));
            if i < 99 {
                bc.advance();
            }
        }
    }

    #[test]
    fn test_batch_cursor_ef_reset() {
        let docs = vec![10, 20, 30, 40, 50];
        let ef = EliasFano::from_sorted(&docs);
        let mut bc = ef.batch_cursor();

        bc.advance();
        bc.advance();
        assert_eq!(bc.current(), Some(30));

        bc.reset();
        assert_eq!(bc.current(), Some(10));
        assert_eq!(bc.index(), 0);
    }

    #[test]
    fn test_batch_cursor_pef_matches_iter() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10 + i % 7).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        let from_iter: Vec<u64> = pef.iter().collect();
        let mut from_batch = Vec::new();
        let mut bc = pef.batch_cursor();
        if let Some(v) = bc.current() {
            from_batch.push(v);
            while bc.advance() {
                from_batch.push(bc.current().unwrap());
            }
        }

        assert_eq!(from_batch.len(), from_iter.len());
        assert_eq!(from_batch, from_iter);
    }

    #[test]
    fn test_batch_cursor_pef_cross_chunk() {
        // 300 elements spans 3 chunks of 128
        let docs: Vec<u32> = (0..300).map(|i| i * 10).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);

        let mut bc = pef.batch_cursor();
        let mut count = 0;
        if bc.current().is_some() {
            count += 1;
            while bc.advance() {
                count += 1;
            }
        }
        assert_eq!(count, 300);
    }

    /// Performance: batch cursor vs regular cursor — release only.
    #[test]
    fn test_batch_cursor_performance() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        #[allow(unused_variables)]
        let ef = EliasFano::from_sorted(&docs);

        #[cfg(not(debug_assertions))]
        {
            let iterations = 10;

            // Regular cursor
            let start = std::time::Instant::now();
            let mut sum1 = 0u64;
            for _ in 0..iterations {
                let mut cursor = ef.cursor();
                sum1 += cursor.current().unwrap();
                while cursor.advance() {
                    sum1 += cursor.current().unwrap();
                }
            }
            let cursor_time = start.elapsed();

            // Batch cursor
            let start = std::time::Instant::now();
            let mut sum2 = 0u64;
            for _ in 0..iterations {
                let mut bc = ef.batch_cursor();
                sum2 += bc.current().unwrap();
                while bc.advance() {
                    sum2 += bc.current().unwrap();
                }
            }
            let batch_time = start.elapsed();

            assert_eq!(sum1, sum2, "batch cursor and cursor must produce same sum");

            eprintln!(
                "Sequential 100K: cursor={:?}, batch={:?}, ratio={:.2}×",
                cursor_time,
                batch_time,
                cursor_time.as_nanos() as f64 / batch_time.as_nanos() as f64
            );
        }
    }

    // ========================================================================
    // DP-Optimal PEF tests
    // ========================================================================

    #[test]
    fn test_opef_empty() {
        let opef = OptimalPartitionedEliasFano::from_sorted(&[]);
        assert_eq!(opef.len(), 0);
        assert!(opef.is_empty());
        assert_eq!(opef.get(0), None);
        assert_eq!(opef.next_geq(0), None);
    }

    #[test]
    fn test_opef_small() {
        // Below MIN_CHUNK_SIZE — should still work (last chunk exception)
        let docs = vec![3, 5, 11, 27, 31];
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);
        assert_eq!(opef.len(), 5);

        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(opef.get(i), Some(v as u64), "get({}) failed", i);
        }
    }

    #[test]
    fn test_opef_matches_ef() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10 + i % 7).collect();
        let ef = EliasFano::from_sorted(&docs);
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        assert_eq!(ef.len(), opef.len());

        // get() must match
        for i in 0..docs.len() {
            assert_eq!(
                ef.get(i),
                opef.get(i),
                "get({}) mismatch: ef={:?} opef={:?}",
                i,
                ef.get(i),
                opef.get(i)
            );
        }

        // next_geq() must match
        for target in (0..5010).step_by(7) {
            assert_eq!(
                ef.next_geq(target),
                opef.next_geq(target),
                "next_geq({}) mismatch",
                target
            );
        }

        // iter() must match
        let ef_vals: Vec<u64> = ef.iter().collect();
        let opef_vals: Vec<u64> = opef.iter().collect();
        assert_eq!(ef_vals, opef_vals);
    }

    #[test]
    fn test_opef_large() {
        let docs: Vec<u32> = (0..10000).map(|i| i * 100 + i % 7).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        assert_eq!(opef.len(), 10000);

        // Sample verification
        for i in (0..10000).step_by(100) {
            assert_eq!(opef.get(i), Some(docs[i] as u64), "get({}) failed", i);
        }

        // Iterator
        let from_iter: Vec<u64> = opef.iter().collect();
        assert_eq!(from_iter.len(), 10000);
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(from_iter[i], v as u64, "iter[{}] mismatch", i);
        }
    }

    #[test]
    fn test_opef_space_vs_uniform() {
        let docs: Vec<u32> = (0..10000).map(|i| i * 100).collect();
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        eprintln!(
            "Uniform PEF: {:.1} bits/elem, {} bytes",
            pef.bits_per_element(),
            pef.size_bytes()
        );
        eprintln!(
            "Optimal PEF: {:.1} bits/elem, {} bytes",
            opef.bits_per_element(),
            opef.size_bytes()
        );

        // Optimal should be no worse than 1.5× uniform (usually better)
        assert!(
            opef.bits_per_element() < pef.bits_per_element() * 1.5,
            "Optimal PEF too large: {:.1} vs uniform {:.1}",
            opef.bits_per_element(),
            pef.bits_per_element()
        );
    }

    #[test]
    fn test_opef_next_geq() {
        let docs: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        assert_eq!(opef.next_geq(0), Some((0, 0)));
        assert_eq!(opef.next_geq(55), Some((6, 60)));
        assert_eq!(opef.next_geq(9990), Some((999, 9990)));
        assert_eq!(opef.next_geq(9991), None);
    }

    #[test]
    fn test_opef_cursor_sequential() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10 + i % 7).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        // Cursor sequential should match iterator
        let from_iter: Vec<u64> = opef.iter().collect();
        let mut from_cursor = Vec::new();
        let mut cursor = opef.cursor();
        if let Some(v) = cursor.current() {
            from_cursor.push(v);
            while cursor.advance() {
                from_cursor.push(cursor.current().unwrap());
            }
        }
        assert_eq!(from_cursor, from_iter);
    }

    #[test]
    fn test_opef_cursor_advance_to_geq() {
        let docs: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);
        let mut cursor = opef.cursor();

        // Same-chunk fast path
        assert!(cursor.advance_to_geq(50));
        assert_eq!(cursor.current(), Some(50));

        // Advance further
        assert!(cursor.advance_to_geq(100));
        assert_eq!(cursor.current(), Some(100));

        // Already at target
        assert!(cursor.advance_to_geq(100));
        assert_eq!(cursor.current(), Some(100));

        // Cross-chunk jump
        assert!(cursor.advance_to_geq(5000));
        assert_eq!(cursor.current(), Some(5000));

        // Near end
        assert!(cursor.advance_to_geq(9990));
        assert_eq!(cursor.current(), Some(9990));

        // Past end
        assert!(!cursor.advance_to_geq(10000));
    }

    #[test]
    fn test_opef_cursor_advance_to_geq_sorted_targets() {
        // Simulates posting list intersection: sorted targets
        let docs: Vec<u32> = (0..1000).map(|i| i * 7 + i % 3).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);
        let targets: Vec<u64> = (0..200).map(|i| (i * 35) as u64).collect();

        let mut cursor = opef.cursor();
        let mut cursor_results = Vec::new();
        for &t in &targets {
            if cursor.advance_to_geq(t) {
                cursor_results.push(cursor.current().unwrap());
            }
        }

        // Verify against stateless next_geq
        let mut stateless_results = Vec::new();
        for &t in &targets {
            if let Some((_, v)) = opef.next_geq(t as u64) {
                stateless_results.push(v);
            }
        }

        // Cursor should give same or later results (since it maintains position)
        // For sorted targets, results should match
        assert_eq!(cursor_results.len(), stateless_results.len());
        for (c, s) in cursor_results.iter().zip(stateless_results.iter()) {
            assert!(*c >= *s, "cursor {} should be >= stateless {}", c, s);
        }
    }

    #[test]
    fn test_opef_batch_cursor() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10 + i % 7).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&docs);

        let from_iter: Vec<u64> = opef.iter().collect();
        let mut from_batch = Vec::new();
        let mut bc = opef.batch_cursor();
        if let Some(v) = bc.current() {
            from_batch.push(v);
            while bc.advance() {
                from_batch.push(bc.current().unwrap());
            }
        }

        assert_eq!(from_batch, from_iter);
    }

    // ========================================================================
    // Hybrid Posting List tests
    // ========================================================================

    #[test]
    fn test_hybrid_dense() {
        let docs: Vec<u32> = vec![1, 5, 10, 20, 50];
        let h = HybridPostingList::from_sorted(&docs);
        assert_eq!(h.encoding(), PostingEncoding::Dense);
        assert_eq!(h.len(), 5);
        assert_eq!(h.get(0), Some(1));
        assert_eq!(h.get(4), Some(50));
        assert_eq!(h.next_geq(6), Some((2, 10)));
    }

    #[test]
    fn test_hybrid_ef() {
        let docs: Vec<u32> = (0..100).map(|i| i * 10).collect();
        let h = HybridPostingList::from_sorted(&docs);
        assert_eq!(h.encoding(), PostingEncoding::EliasFano);
        assert_eq!(h.len(), 100);
        assert_eq!(h.get(50), Some(500));
        assert_eq!(h.next_geq(55), Some((6, 60)));
    }

    #[test]
    fn test_hybrid_partitioned() {
        let docs: Vec<u32> = (0..500).map(|i| i * 10).collect();
        let h = HybridPostingList::from_sorted(&docs);
        assert_eq!(h.encoding(), PostingEncoding::Partitioned);
        assert_eq!(h.len(), 500);
        assert_eq!(h.get(0), Some(0));
        assert_eq!(h.next_geq(4990), Some((499, 4990)));
        assert_eq!(h.next_geq(4991), None);
    }

    #[test]
    fn test_hybrid_optimal() {
        let docs: Vec<u32> = (0..5000).map(|i| i * 10).collect();
        let h = HybridPostingList::from_sorted(&docs);
        assert_eq!(h.encoding(), PostingEncoding::Optimal);
        assert_eq!(h.len(), 5000);

        // Verify correctness matches plain EF
        let ef = EliasFano::from_sorted(&docs);
        for i in (0..5000).step_by(100) {
            assert_eq!(h.get(i), ef.get(i), "get({}) mismatch", i);
        }
        for t in (0..50000).step_by(77) {
            assert_eq!(h.next_geq(t), ef.next_geq(t), "next_geq({}) mismatch", t);
        }
    }

    #[test]
    fn test_hybrid_force_encoding() {
        let docs: Vec<u32> = (0..100).map(|i| i * 10).collect();

        let dense = HybridPostingList::with_encoding(&docs, PostingEncoding::Dense);
        assert_eq!(dense.encoding(), PostingEncoding::Dense);

        let ef = HybridPostingList::with_encoding(&docs, PostingEncoding::EliasFano);
        assert_eq!(ef.encoding(), PostingEncoding::EliasFano);

        // Both should give same results
        for i in 0..100 {
            assert_eq!(dense.get(i), ef.get(i));
        }
    }

    #[test]
    fn test_hybrid_empty() {
        let h = HybridPostingList::from_sorted(&[]);
        assert_eq!(h.encoding(), PostingEncoding::Dense);
        assert_eq!(h.len(), 0);
        assert_eq!(h.get(0), None);
        assert_eq!(h.next_geq(0), None);
    }

    /// Performance comparison across encodings — release only.
    #[test]
    fn test_hybrid_performance_comparison() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        #[allow(unused_variables)]
        let targets: Vec<u64> = (0..10000).map(|i| (i * 100) as u64).collect();

        #[allow(unused_variables)]
        let dense = HybridPostingList::with_encoding(&docs, PostingEncoding::Dense);
        #[allow(unused_variables)]
        let ef = HybridPostingList::with_encoding(&docs, PostingEncoding::EliasFano);
        #[allow(unused_variables)]
        let pef = HybridPostingList::with_encoding(&docs, PostingEncoding::Partitioned);
        #[allow(unused_variables)]
        let opef = HybridPostingList::with_encoding(&docs, PostingEncoding::Optimal);

        #[cfg(not(debug_assertions))]
        {
            let iters = 50;
            for (name, h) in [
                ("Dense", &dense),
                ("EF", &ef),
                ("PEF", &pef),
                ("OPEF", &opef),
            ] {
                let start = std::time::Instant::now();
                let mut sink = 0usize;
                for _ in 0..iters {
                    for &t in &targets {
                        if h.next_geq(t).is_some() {
                            sink += 1;
                        }
                    }
                }
                let elapsed = start.elapsed();
                let per_call = elapsed.as_nanos() as f64 / (iters as f64 * targets.len() as f64);
                eprintln!(
                    "{name}: {per_call:.1}ns/call, {:.1} bits/elem [sink={sink}]",
                    h.bits_per_element()
                );
            }
        }
    }

    #[test]
    fn test_cursor_advance_to_index_basic() {
        let vals = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // advance_to_index(0) matches get(0)
        assert!(cursor.advance_to_index(0));
        assert_eq!(cursor.current(), Some(3));
        assert_eq!(cursor.index(), 0);

        // advance_to_index(7) matches get(7) — last element
        assert!(cursor.advance_to_index(7));
        assert_eq!(cursor.current(), Some(63));
        assert_eq!(cursor.index(), 7);

        // advance_to_index(8) — past end
        assert!(!cursor.advance_to_index(8));
        // Cursor should be unchanged
        assert_eq!(cursor.index(), 7);
        assert_eq!(cursor.current(), Some(63));
    }

    #[test]
    fn test_cursor_advance_to_index_then_advance() {
        let vals = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Jump to index 5, then advance should give index 6
        assert!(cursor.advance_to_index(5));
        assert_eq!(cursor.current(), Some(42));
        assert!(cursor.advance());
        assert_eq!(cursor.current(), Some(58));
        assert_eq!(cursor.index(), 6);
    }

    #[test]
    fn test_cursor_advance_to_index_then_geq() {
        let vals = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Jump to index 2, then advance_to_geq(40) should find 42 at index 5
        assert!(cursor.advance_to_index(2));
        assert_eq!(cursor.current(), Some(11));
        assert!(cursor.advance_to_geq(40));
        assert_eq!(cursor.current(), Some(42));
        assert_eq!(cursor.index(), 5);
    }

    #[test]
    fn test_cursor_advance_to_index_backward() {
        let vals = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Forward to 5, then backward to 2
        assert!(cursor.advance_to_index(5));
        assert_eq!(cursor.current(), Some(42));
        assert!(cursor.advance_to_index(2));
        assert_eq!(cursor.current(), Some(11));
        assert_eq!(cursor.index(), 2);
    }

    #[test]
    fn test_cursor_advance_to_index_roundtrip() {
        let vals = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Visit every index, verify matches get()
        for i in 0..vals.len() {
            assert!(cursor.advance_to_index(i));
            assert_eq!(cursor.current(), ef.get(i));
            assert_eq!(cursor.index(), i);
        }
    }

    #[test]
    fn test_cursor_advance_to_index_same_position() {
        let vals = vec![10, 20, 30];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        assert!(cursor.advance_to_index(1));
        assert_eq!(cursor.current(), Some(20));
        // Same position — should be no-op
        assert!(cursor.advance_to_index(1));
        assert_eq!(cursor.current(), Some(20));
    }

    #[test]
    fn test_cursor_advance_to_index_empty() {
        let ef = EliasFano::from_sorted(&[]);
        let mut cursor = ef.cursor();
        assert!(!cursor.advance_to_index(0));
    }

    #[test]
    fn test_pef_cursor_advance_to_index() {
        // Need 200+ elements to exercise multiple chunks
        let vals: Vec<u32> = (0..500).map(|i| i * 3 + 1).collect();
        let pef = PartitionedEliasFano::from_sorted(&vals);
        let mut cursor = pef.cursor();

        // Test various indices across chunks
        for &idx in &[0, 1, 127, 128, 129, 255, 256, 300, 499] {
            assert!(
                cursor.advance_to_index(idx),
                "advance_to_index({idx}) failed"
            );
            assert_eq!(
                cursor.current(),
                Some(vals[idx] as u64),
                "wrong value at {idx}"
            );
            assert_eq!(cursor.index(), idx);
        }

        // Past end
        assert!(!cursor.advance_to_index(500));

        // Backward jump
        assert!(cursor.advance_to_index(300));
        assert!(cursor.advance_to_index(50));
        assert_eq!(cursor.current(), Some(vals[50] as u64));
    }

    #[test]
    fn test_opef_cursor_advance_to_index() {
        let vals: Vec<u32> = (0..500).map(|i| i * 3 + 1).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&vals);
        let mut cursor = opef.cursor();

        for &idx in &[0, 1, 50, 100, 200, 300, 499] {
            assert!(
                cursor.advance_to_index(idx),
                "advance_to_index({idx}) failed"
            );
            assert_eq!(
                cursor.current(),
                Some(vals[idx] as u64),
                "wrong value at {idx}"
            );
            assert_eq!(cursor.index(), idx);
        }

        assert!(!cursor.advance_to_index(500));

        // Backward jump
        assert!(cursor.advance_to_index(400));
        assert!(cursor.advance_to_index(100));
        assert_eq!(cursor.current(), Some(vals[100] as u64));
    }

    #[test]
    fn test_cursor_advance_to_index_large_values() {
        // Test with values near u32::MAX to verify select1 with large universes
        let vals = vec![0, 1000, u32::MAX / 2, u32::MAX - 1000, u32::MAX];
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        for i in 0..vals.len() {
            assert!(cursor.advance_to_index(i));
            assert_eq!(cursor.current(), Some(vals[i] as u64));
            assert_eq!(cursor.index(), i);
        }
    }

    #[test]
    fn test_cursor_advance_to_index_large_sequence() {
        // 1500 elements exercises select1 sampling (every 256 elements)
        let vals: Vec<u32> = (0..1500).map(|i| i * 7 + 3).collect();
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Every 10th index
        for i in (0..1500).step_by(10) {
            assert!(cursor.advance_to_index(i));
            assert_eq!(cursor.current(), Some(vals[i] as u64));
            assert_eq!(cursor.index(), i);
        }

        // Backward jumps across the sequence
        assert!(cursor.advance_to_index(1000));
        assert!(cursor.advance_to_index(500));
        assert_eq!(cursor.current(), Some(vals[500] as u64));
        assert!(cursor.advance_to_index(1499));
        assert_eq!(cursor.current(), Some(vals[1499] as u64));
        assert!(cursor.advance_to_index(0));
        assert_eq!(cursor.current(), Some(vals[0] as u64));
    }

    #[test]
    fn test_cursor_advance_to_index_multiple_backward_jumps() {
        let vals: Vec<u32> = (0..200).map(|i| i * 5).collect();
        let ef = EliasFano::from_sorted(&vals);
        let mut cursor = ef.cursor();

        // Zigzag: forward, backward, forward, backward
        assert!(cursor.advance_to_index(100));
        assert_eq!(cursor.current(), Some(500));
        assert!(cursor.advance_to_index(50));
        assert_eq!(cursor.current(), Some(250));
        assert!(cursor.advance_to_index(150));
        assert_eq!(cursor.current(), Some(750));
        assert!(cursor.advance_to_index(25));
        assert_eq!(cursor.current(), Some(125));

        // Verify advance() still works after backward jump
        assert!(cursor.advance());
        assert_eq!(cursor.index(), 26);
        assert_eq!(cursor.current(), Some(130));
    }

    #[test]
    fn test_empty_pef_advance_to_index() {
        let pef = PartitionedEliasFano::from_sorted(&[]);
        let mut cursor = pef.cursor();
        assert!(!cursor.advance_to_index(0));
    }

    #[test]
    fn test_empty_opef_advance_to_index() {
        let opef = OptimalPartitionedEliasFano::from_sorted(&[]);
        let mut cursor = opef.cursor();
        assert!(!cursor.advance_to_index(0));
    }

    #[test]
    fn test_pef_cursor_within_chunk_jumps() {
        let vals: Vec<u32> = (0..500).map(|i| i * 3).collect();
        let pef = PartitionedEliasFano::from_sorted(&vals);
        let mut cursor = pef.cursor();

        // Forward within chunk 0
        assert!(cursor.advance_to_index(10));
        assert_eq!(cursor.current(), Some(30));
        assert!(cursor.advance_to_index(20));
        assert_eq!(cursor.current(), Some(60));

        // Backward within chunk 0
        assert!(cursor.advance_to_index(5));
        assert_eq!(cursor.current(), Some(15));

        // Last element of chunk 0
        assert!(cursor.advance_to_index(127));
        assert_eq!(cursor.current(), Some(vals[127] as u64));
        assert_eq!(cursor.index(), 127);

        // First element of chunk 1, then advance()
        assert!(cursor.advance_to_index(128));
        assert_eq!(cursor.current(), Some(vals[128] as u64));
        assert_eq!(cursor.index(), 128);
        assert!(cursor.advance());
        assert_eq!(cursor.index(), 129);
        assert_eq!(cursor.current(), Some(vals[129] as u64));
    }

    #[test]
    fn test_pef_cursor_chunk_boundary_roundtrip() {
        let vals: Vec<u32> = (0..500).map(|i| i * 3).collect();
        let pef = PartitionedEliasFano::from_sorted(&vals);
        let mut cursor = pef.cursor();

        // Sequential calls crossing chunk boundary
        for i in 125..132 {
            assert!(cursor.advance_to_index(i));
            assert_eq!(cursor.current(), Some(vals[i] as u64));
            assert_eq!(cursor.index(), i);
        }
    }

    #[test]
    fn test_pef_cursor_advance_to_index_large_values() {
        let vals = vec![0, 1000, u32::MAX / 2, u32::MAX - 1000, u32::MAX];
        let pef = PartitionedEliasFano::from_sorted(&vals);
        let mut cursor = pef.cursor();

        for i in 0..vals.len() {
            assert!(cursor.advance_to_index(i));
            assert_eq!(cursor.current(), Some(vals[i] as u64));
        }
    }

    #[test]
    fn test_opef_cursor_advance_to_index_within_chunk() {
        let vals: Vec<u32> = (0..1000).map(|i| i * 5).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&vals);
        let mut cursor = opef.cursor();

        // Forward within a chunk
        assert!(cursor.advance_to_index(10));
        assert_eq!(cursor.current(), Some(50));
        assert!(cursor.advance_to_index(20));
        assert_eq!(cursor.current(), Some(100));

        // Backward within the same chunk
        assert!(cursor.advance_to_index(5));
        assert_eq!(cursor.current(), Some(25));

        // Jump far forward then far backward
        assert!(cursor.advance_to_index(900));
        assert_eq!(cursor.current(), Some(vals[900] as u64));
        assert!(cursor.advance_to_index(50));
        assert_eq!(cursor.current(), Some(vals[50] as u64));
    }

    #[test]
    fn test_opef_cursor_advance_to_index_then_advance() {
        let vals: Vec<u32> = (0..500).map(|i| i * 3 + 1).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&vals);
        let mut cursor = opef.cursor();

        // Jump to an index, then call advance()
        assert!(cursor.advance_to_index(250));
        assert_eq!(cursor.current(), Some(vals[250] as u64));
        assert!(cursor.advance());
        assert_eq!(cursor.index(), 251);
        assert_eq!(cursor.current(), Some(vals[251] as u64));

        // Jump to an index, then call advance_to_geq()
        assert!(cursor.advance_to_index(100));
        assert_eq!(cursor.current(), Some(vals[100] as u64));
        let target = vals[200] as u64;
        assert!(cursor.advance_to_geq(target));
        assert_eq!(cursor.index(), 200);
    }
}
