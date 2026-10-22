use super::*;
// test module mirrors the file's own name by convention
#[allow(clippy::module_inception)]
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
        let _ = found;
        let elapsed = start.elapsed();
        #[cfg(debug_assertions)]
        let _ = elapsed;

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
        let ef = EliasFano::from_sorted(&docs);
        #[cfg(debug_assertions)]
        let _ = ef;

        #[cfg(not(debug_assertions))]
        {
            let mut best_get = std::time::Duration::MAX;
            let mut best_cursor = std::time::Duration::MAX;
            for _ in 0..5 {
                // get(i) sequential
                let start = std::time::Instant::now();
                let mut sum1 = 0u64;
                for _ in 0..4 {
                    for i in 0..ef.len() {
                        sum1 += std::hint::black_box(ef.get(i).unwrap());
                    }
                }
                best_get = best_get.min(start.elapsed());

                // cursor sequential
                let start = std::time::Instant::now();
                let mut sum2 = 0u64;
                for _ in 0..4 {
                    let mut cursor = ef.cursor();
                    sum2 += std::hint::black_box(cursor.current().unwrap());
                    while cursor.advance() {
                        sum2 += std::hint::black_box(cursor.current().unwrap());
                    }
                }
                best_cursor = best_cursor.min(start.elapsed());

                assert_eq!(sum1, sum2, "cursor and get must produce same sum");
            }

            let speedup = best_get.as_nanos() as f64 / best_cursor.as_nanos().max(1) as f64;
            eprintln!(
                "Sequential 100K: get={:?}, cursor={:?}, speedup={:.1}×",
                best_get, best_cursor, speedup
            );

            assert!(
                speedup > 1.2,
                "cursor should be faster than get(), got {:.2}×",
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
        let ef = EliasFano::from_sorted(&docs);
        let pef = PartitionedEliasFano::from_sorted(&docs);
        let targets: Vec<u64> = (0..10000).map(|i| (i * 100) as u64).collect();
        #[cfg(debug_assertions)]
        {
            let _ = ef;
            let _ = pef;
            let _ = targets;
        }

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
        let ef = EliasFano::from_sorted(&docs);
        let pef = PartitionedEliasFano::from_sorted(&docs);
        #[cfg(debug_assertions)]
        {
            let _ = ef;
            let _ = pef;
        }

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
        let ef = EliasFano::from_sorted(&docs);
        #[cfg(debug_assertions)]
        let _ = ef;

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
            if let Some((_, v)) = opef.next_geq(t) {
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
    fn test_hybrid_clustered_selected_for_dense() {
        // Dense / run-heavy (span == len) → auto-selects Clustered.
        let docs: Vec<u32> = (0..5000).collect();
        let h = HybridPostingList::from_sorted(&docs);
        assert_eq!(h.encoding(), PostingEncoding::Clustered);
        // Correctness must match plain EF on the same data.
        let ef = EliasFano::from_sorted(&docs);
        for i in (0..5000).step_by(50) {
            assert_eq!(h.get(i), ef.get(i), "get({i})");
        }
        for t in (0..5200).step_by(37) {
            assert_eq!(h.next_geq(t), ef.next_geq(t), "next_geq({t})");
        }
        // Sparse data of the same length must NOT pick Clustered.
        let sparse: Vec<u32> = (0..5000).map(|i| i * 10).collect();
        assert_ne!(
            HybridPostingList::from_sorted(&sparse).encoding(),
            PostingEncoding::Clustered
        );
    }

    #[test]
    fn test_hybrid_intersect_count() {
        let a: Vec<u32> = (0..3000).collect();
        let b: Vec<u32> = (1000..4000).collect();
        let ha = HybridPostingList::from_sorted(&a);
        let hb = HybridPostingList::from_sorted(&b);
        // Both dense → Clustered → block path. Overlap 1000..3000 = 2000.
        assert_eq!(ha.encoding(), PostingEncoding::Clustered);
        assert_eq!(ha.intersect_count(&hb), 2000);
        assert_eq!(hb.intersect_count(&ha), 2000); // symmetry

        // Mixed encodings exercise the leapfrog fallback.
        let sparse: Vec<u32> = (0..3000).map(|i| i * 10).collect();
        let hs = HybridPostingList::from_sorted(&sparse);
        assert_ne!(hs.encoding(), PostingEncoding::Clustered);
        // sparse ∩ a (0..3000): common values are multiples of 10 in [0,2999] → 300.
        assert_eq!(hs.intersect_count(&ha), 300);
        assert_eq!(ha.intersect_count(&hs), 300);
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

    #[test]
    fn test_hybrid_from_sorted_u64_no_truncation() {
        // Regression: short lists (≤ DENSE_THRESHOLD) with values > u32::MAX
        // used to be cast to u32 by the Dense variant, silently truncating
        // the upper 32 bits.
        let docs: Vec<u64> = (0..10).map(|i| (1u64 << 33) + i * (1u64 << 20)).collect();
        let h = HybridPostingList::from_sorted_u64(&docs);
        assert_ne!(h.encoding(), PostingEncoding::Dense);
        assert_eq!(h.len(), docs.len());
        for (i, &v) in docs.iter().enumerate() {
            assert_eq!(h.get(i), Some(v), "get({i})");
        }
        assert_eq!(h.next_geq(1u64 << 33), Some((0, 1u64 << 33)));
        assert_eq!(
            h.next_geq((1u64 << 33) + 1),
            Some((1, (1u64 << 33) + (1u64 << 20)))
        );
        assert_eq!(h.next_geq(docs[9] + 1), None);

        // Short lists that DO fit in u32 must still pick Dense.
        let small: Vec<u64> = vec![1, 5, 10, u32::MAX as u64];
        let hs = HybridPostingList::from_sorted_u64(&small);
        assert_eq!(hs.encoding(), PostingEncoding::Dense);
        assert_eq!(hs.get(3), Some(u32::MAX as u64));

        // Empty input stays valid.
        let he = HybridPostingList::from_sorted_u64(&[]);
        assert_eq!(he.len(), 0);
        assert_eq!(he.get(0), None);
    }

    /// Performance comparison across encodings — release only.
    #[test]
    fn test_hybrid_performance_comparison() {
        let docs: Vec<u32> = (0..100000).map(|i| i * 10).collect();
        let targets: Vec<u64> = (0..10000).map(|i| (i * 100) as u64).collect();
        let dense = HybridPostingList::with_encoding(&docs, PostingEncoding::Dense);
        let ef = HybridPostingList::with_encoding(&docs, PostingEncoding::EliasFano);
        let pef = HybridPostingList::with_encoding(&docs, PostingEncoding::Partitioned);
        let opef = HybridPostingList::with_encoding(&docs, PostingEncoding::Optimal);
        #[cfg(debug_assertions)]
        {
            let _ = targets;
            let _ = dense;
            let _ = ef;
            let _ = pef;
            let _ = opef;
        }

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

    #[test]
    #[cfg(feature = "serde")]
    fn test_elias_fano_batch_cursor_corruption_oob_panic() {
        let docs = vec![3, 5, 11, 27, 31, 42, 58, 63];
        let ef = EliasFano::from_sorted(&docs);

        let json_str = serde_json::to_string(&ef).unwrap();
        let mut val: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Corrupt the len field to be larger than actual element count
        if let Some(len) = val.get_mut("len") {
            *len = serde_json::json!(100);
        }

        let corrupt_json = serde_json::to_string(&val).unwrap();
        let corrupt_ef: EliasFano = serde_json::from_str(&corrupt_json).unwrap();

        let mut cursor = corrupt_ef.batch_cursor();

        // The first element is already decoded and available.
        assert_eq!(cursor.current(), Some(docs[0] as u64));

        // Advance to read the next 7 elements of the first batch
        for i in 1..8 {
            assert!(cursor.advance(), "Failed to advance at index {}", i);
            assert_eq!(cursor.current(), Some(docs[i] as u64));
        }

        // The next advance will trigger refill() since the buffer of 8 elements is exhausted.
        // It should detect the corruption and safely return false instead of panicking.
        assert!(
            !cursor.advance(),
            "Cursor should be exhausted after batch depletion on corrupt structure"
        );
        assert!(cursor.is_exhausted());
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_optimal_pef_batch_cursor_corruption_oob() {
        let vals: Vec<u32> = (0..20).map(|i| i * 5).collect();
        let opef = OptimalPartitionedEliasFano::from_sorted(&vals);

        let json_str = serde_json::to_string(&opef).unwrap();
        let mut val: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Corrupt the count field in the first chunk meta to be larger
        if let Some(count) = val
            .get_mut("meta")
            .and_then(|m| m.get_mut(0))
            .and_then(|c0| c0.get_mut("count"))
        {
            *count = serde_json::json!(100);
        }
        // Also corrupt the root len field
        if let Some(len) = val.get_mut("len") {
            *len = serde_json::json!(100);
        }

        let corrupt_json = serde_json::to_string(&val).unwrap();
        let corrupt_opef: OptimalPartitionedEliasFano =
            serde_json::from_str(&corrupt_json).unwrap();

        let mut cursor = corrupt_opef.batch_cursor();

        // The first element is already decoded and available.
        assert_eq!(cursor.current(), Some(vals[0] as u64));

        // We should be able to safely read 20 elements (indices 0..19)
        for i in 1..20 {
            assert!(cursor.advance(), "Failed to advance at index {}", i);
            assert_eq!(cursor.current(), Some(vals[i] as u64));
        }

        // The next advance (to 21st element) will trigger refill() which will detect the corruption and return false.
        assert!(
            !cursor.advance(),
            "Cursor should be exhausted after chunk depletion on corrupt structure"
        );
        assert!(cursor.is_exhausted());
    }

    #[test]
    fn test_u64_max_value_universe_saturation() {
        // Regression (plan.md 7.3): universe = last + 1 wrapped to 0 for
        // u64::MAX values, forcing low_bit_width = 0 and a ~u64::MAX-bit
        // high array (overflow panic / OOM). All variants must round-trip.
        let values: Vec<u64> = vec![0, 1, 1000, u64::MAX - 1, u64::MAX];

        let ef = EliasFano::from_sorted_u64(&values);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(ef.get(i), Some(v), "EliasFano idx {}", i);
        }

        let pef = PartitionedEliasFano::from_sorted_u64(&values);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(pef.get(i), Some(v), "PEF idx {}", i);
        }

        let opef = OptimalPartitionedEliasFano::from_sorted_u64(&values);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(opef.get(i), Some(v), "OPEF idx {}", i);
        }

        let cef = ClusteredEliasFano::from_sorted_u64(&values);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(cef.get(i), Some(v), "CEF idx {}", i);
        }
    }

    #[test]
    fn test_next_geq_reaches_u64_max_element() {
        // `universe` is `last + 1` computed with `saturating_add`, so a list
        // containing u64::MAX gets `universe == u64::MAX` and the half-open
        // `target >= universe` guard used to reject the largest element -
        // `get` returned it but `next_geq` claimed it did not exist.
        let values: Vec<u64> = vec![0, 1, 1000, u64::MAX - 1, u64::MAX];
        let last = values.len() - 1;

        let ef = EliasFano::from_sorted_u64(&values);
        assert_eq!(ef.next_geq(u64::MAX), Some((last, u64::MAX)));
        assert_eq!(ef.next_geq(u64::MAX - 1), Some((last - 1, u64::MAX - 1)));

        let pef = PartitionedEliasFano::from_sorted_u64(&values);
        assert_eq!(pef.next_geq(u64::MAX), Some((last, u64::MAX)));

        let opef = OptimalPartitionedEliasFano::from_sorted_u64(&values);
        assert_eq!(opef.next_geq(u64::MAX), Some((last, u64::MAX)));

        // A list that stops just short of u64::MAX must still report None.
        let short = EliasFano::from_sorted_u64(&[0, 1, u64::MAX - 1]);
        assert_eq!(short.next_geq(u64::MAX), None);
    }

    #[test]
    fn test_hybrid_dense_rejects_targets_above_u32_max() {
        // Dense stores u32; `target as u32` wrapped, so a target beyond
        // u32::MAX was answered with an element *below* it.
        let dense = HybridPostingList::from_sorted(&[5u32, 10, 20]);
        assert_eq!(dense.encoding(), PostingEncoding::Dense);
        assert_eq!(dense.next_geq((1u64 << 32) + 3), None);
        assert_eq!(dense.next_geq(u64::MAX), None);
        // Still correct below the boundary.
        assert_eq!(dense.next_geq(6), Some((1, 10)));
    }

    #[test]
    fn test_hybrid_intersect_count_at_u64_max() {
        // Two defects met here: `probe = va + 1` wrapped to 0 at u64::MAX and
        // restarted the leapfrog forever, and the `next_geq` universe guard
        // dropped the u64::MAX element so the count came out short.
        let values: Vec<u64> = vec![10, 1000, u64::MAX];
        let a = HybridPostingList::from_sorted_u64(&values);
        let b = HybridPostingList::from_sorted_u64(&values);
        assert_eq!(a.intersect_count(&b), 3);

        let partial = HybridPostingList::from_sorted_u64(&[1000, u64::MAX]);
        assert_eq!(a.intersect_count(&partial), 2);
    }

    #[test]
    fn test_hybrid_run_heavy_does_not_overflow_on_full_range() {
        // `last - first + 1` overflowed for a list spanning the whole u64
        // range: a panic in debug, and in release a span of 0, which made the
        // sparsest possible list look maximally run-heavy.
        let h = HybridPostingList::from_sorted_u64(&[0, u64::MAX]);
        assert_ne!(h.encoding(), PostingEncoding::Clustered);
        assert_eq!(h.get(0), Some(0));
        assert_eq!(h.get(1), Some(u64::MAX));

        // A genuinely run-heavy list must still pick Clustered.
        let runs: Vec<u64> = (0..200u64).collect();
        assert_eq!(
            HybridPostingList::from_sorted_u64(&runs).encoding(),
            PostingEncoding::Clustered
        );
    }

    #[test]
    fn test_monotone_duplicates_and_next_geq_first_index_all_variants() {
        // 1. Constant sequence (all duplicates): ClusteredEliasFano and HybridPostingList
        //    must not collapse duplicates into a 1-bit Bitmap.
        let const_vals = vec![42u32; 200];
        let cef_const = ClusteredEliasFano::from_sorted(&const_vals);
        let pef_const = PartitionedEliasFano::from_sorted(&const_vals);
        let opef_const = OptimalPartitionedEliasFano::from_sorted(&const_vals);
        let ef_const = EliasFano::from_sorted(&const_vals);
        let hybrid_const = HybridPostingList::from_sorted(&const_vals);

        for i in 0..const_vals.len() {
            assert_eq!(ef_const.get(i), Some(42), "EF const idx {i}");
            assert_eq!(pef_const.get(i), Some(42), "PEF const idx {i}");
            assert_eq!(opef_const.get(i), Some(42), "OPEF const idx {i}");
            assert_eq!(cef_const.get(i), Some(42), "CEF const idx {i}");
            assert_eq!(hybrid_const.get(i), Some(42), "Hybrid const idx {i}");
        }
        // next_geq(42) must return index 0 (the FIRST element >= 42), even when
        // multiple chunks share upper_bound == 42.
        assert_eq!(ef_const.next_geq(42), Some((0, 42)));
        assert_eq!(pef_const.next_geq(42), Some((0, 42)));
        assert_eq!(opef_const.next_geq(42), Some((0, 42)));
        assert_eq!(cef_const.next_geq(42), Some((0, 42)));
        assert_eq!(hybrid_const.next_geq(42), Some((0, 42)));
        assert_eq!(
            HybridPostingList::from_sorted(&[42u32; 20]).next_geq(42),
            Some((0, 42))
        );

        // 2. 128-element chunk where local_universe == count (min=0, max=127, count=128)
        //    but has a duplicate (0, 0) and a gap (missing 126): must NOT use ChunkKind::Run.
        let mut fake_run: Vec<u32> = (0..126).collect();
        fake_run.insert(0, 0); // duplicate 0 -> 127 elements
        fake_run.push(127); // max=127, count=128, missing 126!
        assert_eq!(fake_run.len(), 128);
        let cef_fake = ClusteredEliasFano::from_sorted(&fake_run);
        let hybrid_fake = HybridPostingList::from_sorted(&fake_run);
        for (i, &expected) in fake_run.iter().enumerate() {
            assert_eq!(
                cef_fake.get(i),
                Some(expected as u64),
                "CEF fake_run idx {i}"
            );
            assert_eq!(
                hybrid_fake.get(i),
                Some(expected as u64),
                "Hybrid fake_run idx {i}"
            );
        }
        assert_eq!(cef_fake.next_geq(126), Some((127, 127)));
        assert_eq!(hybrid_fake.next_geq(126), Some((127, 127)));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_corrupt_serde_get_and_pef_batch_cursor_no_ub_or_panic() {
        // S17-R1: Exact corrupt deserialized input from /tmp/s17_hang_mutant.json
        // where chunk_upper_bounds is non-monotone ([312658, 8992919731896089034, 945839, 1133629]).
        // On 1ebdb1d this deserialized as Ok(h) and hung forever in h.intersect_count(&h).
        const S17_HANG_MUTANT_JSON: &str = r#"{"Partitioned":{"all_high_bits":[6725181367942554973,13027900335803917482,7580238445390931346,5558931001735469238,5974605,5921470780086571305,3723512153954886802,10562221667770510157,7607552466545449364,304764198,12218920774162098595,5281254741531765427,7604725093897947794,13207273554404231588,469349,8296648021782408531,6001373803218024937,40221682080330],"all_low_bits":[12851844896141176832,1747619668229033507,3051461650804711616,8733184653346221470,6330244851109875561,16533579935420713827,8085855438957212373,12640437754101315176,9026650630548971559,9769956701692931094,13865749852306623835,11322230192252730439,14923981729729068413,15815287167444744868,15955321782605204040,2012681119491669362,2226082541167939114,7794642711579268982,13098460096358694397,6179250497981018554,12749992742622085523,15346411821965875922,2738486428396136448,11875809012465281303,7354962045982413476,14096981271316938724,5753935690476285897,2665190416921433079,3197037135779573076,12181302780187847644,12283425208823800018,18272739115147417741,17933082139700408081,17499675853826281980,1931397724306628666,5168798816267415105,14820382321381105827,13955550951456101639,1182560758183374432,11068748747576536429,2494028952667343416,11429483860507832673,2171935809508206484,1928810607386358295,1116111505330450432,3977276696118718726,4384276098328312913,2316195880163270958,14246993388819357537,9610585945641472143,5914939160258710180,4263375972322160625,17626283628502267607,11459747412332906484,7040670553475505827,12936584850481146202,715188996089964060,13450465804200164482,15949284612660665513,12513735145025821782,9199658272035294831,15268081292960773409,457019345648958343,7204600462001397391,11712063501967314904,13805380378376055111,1460434542348648448,5687307254615725982,1970061880274699905,8304794971901529479,3149089098433080533,748863998593645088,3916932566079088895,11868786322954594371,17103721347371264594,11784190889517095067,13032062508719966152,10520906129741768736,12020459586070546565,15005179582605110268,94868439,0],"chunk_upper_bounds":[312658,8992919731896089034,945839,1133629],"len":468,"meta":[{"count":128,"high_len_bits":280,"high_offset":0,"high_words":5,"low_bit_width":11,"low_offset":0,"low_words":22,"min_value":1707},{"count":128,"high_len_bits":286,"high_offset":5,"high_words":5,"low_bit_width":11,"low_offset":22,"low_words":22,"min_value":316831},{"count":128,"high_len_bits":276,"high_offset":10,"high_words":5,"low_bit_width":11,"low_offset":44,"low_words":22,"min_value":643251},{"count":84,"high_len_bits":175,"high_offset":15,"high_words":3,"low_bit_width":11,"low_offset":66,"low_words":63,"min_value":948586}],"universe":1133630}}"#;
        if let Ok(h) = serde_json::from_str::<HybridPostingList>(S17_HANG_MUTANT_JSON) {
            assert!(h.intersect_count(&h) <= h.len());
        }
        assert!(
            serde_json::from_str::<HybridPostingList>(S17_HANG_MUTANT_JSON).is_err(),
            "S17-R2: corrupt non-monotone HybridPostingList::Partitioned must be rejected at deserialize time"
        );

        let vals: Vec<u32> = (0..20).map(|i| i * 5).collect();

        // Corrupt EliasFano: len=100 -> get(50) must not read out-of-bounds low_bits
        let ef = EliasFano::from_sorted(&vals);
        let mut v_ef: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&ef).unwrap()).unwrap();
        v_ef["len"] = serde_json::json!(100);
        let corrupt_ef: EliasFano = serde_json::from_value(v_ef).unwrap();
        assert_eq!(corrupt_ef.get(50), None);

        // S17-R2: Structural corruption on PartitionedEliasFano (len=100, meta[0].count=100)
        // must be rejected on deserialize; content-only corruption (zeroed high_bits) must not panic.
        let pef = PartitionedEliasFano::from_sorted(&vals);
        let mut v_pef: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&pef).unwrap()).unwrap();
        v_pef["len"] = serde_json::json!(100);
        v_pef["meta"][0]["count"] = serde_json::json!(100);
        assert!(serde_json::from_value::<PartitionedEliasFano>(v_pef).is_err());

        let mut v_pef_zero_high: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&pef).unwrap()).unwrap();
        v_pef_zero_high["all_high_bits"] = serde_json::json!([0u64]);
        let corrupt_pef: PartitionedEliasFano = serde_json::from_value(v_pef_zero_high).unwrap();
        let _ = corrupt_pef.get(0);
        let mut bc = corrupt_pef.batch_cursor();
        assert!(
            !bc.advance(),
            "PEF batch_cursor must stop on zeroed high_bits"
        );
        assert!(bc.is_exhausted());

        // S17-R2: Structural corruption on OptimalPartitionedEliasFano, ClusteredEliasFano,
        // and HybridPostingList::Dense must be rejected on deserialize (not panic on query).
        let opef = OptimalPartitionedEliasFano::from_sorted(&vals);
        let mut v_opef: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&opef).unwrap()).unwrap();
        v_opef["meta"][0]["low_offset"] = serde_json::json!(999_999);
        assert!(serde_json::from_value::<OptimalPartitionedEliasFano>(v_opef).is_err());

        let cef = ClusteredEliasFano::from_sorted(&vals);
        let mut v_cef: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&cef).unwrap()).unwrap();
        v_cef["meta"][0]["low_offset"] = serde_json::json!(999_999);
        assert!(serde_json::from_value::<ClusteredEliasFano>(v_cef).is_err());

        assert!(
            serde_json::from_str::<HybridPostingList>(r#"{"Dense":[10,5,20]}"#).is_err(),
            "Unsorted HybridPostingList::Dense must be rejected on deserialize"
        );
    }

    #[test]
    #[should_panic(expected = "non-decreasing")]
    fn test_unsorted_input_panics_cleanly() {
        let _ = EliasFano::from_sorted(&[10, 5, 20]);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_s47_r5_legacy_wire_format_and_forged_samples_roundtrip() {
        let n = if cfg!(miri) { 150u64 } else { 1_000u64 };
        let vals: Vec<u64> = (0..n).map(|i| i * 37 + (i % 5)).collect();
        let ef = EliasFano::from_sorted_u64(&vals);

        // Simulate legacy b4148fc JSON with word-aligned select1/select0 samples and rank_samples.
        let mut legacy_json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&ef).unwrap()).unwrap();
        assert!(legacy_json.get("rank_samples").is_some());
        if let Some(arr) = legacy_json["select1_samples"].as_array_mut() {
            for v in arr.iter_mut() {
                *v = serde_json::json!(v.as_u64().unwrap() & !63);
            }
        }
        let decoded: EliasFano = serde_json::from_value(legacy_json).unwrap();
        for (i, &want) in vals.iter().enumerate() {
            assert_eq!(decoded.get(i), Some(want), "mismatch at {i}");
        }
        let mid = (n as usize) / 2;
        assert_eq!(decoded.next_geq(vals[mid]), Some((mid, vals[mid])));

        // Forged low_bit_width >= 64 and forged high_len_bits must not panic on get/next_geq/cursor.
        let mut forged: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&ef).unwrap()).unwrap();
        forged["low_bit_width"] = serde_json::json!(68);
        forged["high_len_bits"] = serde_json::json!(1);
        let f_ef: EliasFano = serde_json::from_value(forged).unwrap();
        let _ = f_ef.get(0);
        let _ = f_ef.next_geq(100);
        let mut cur = f_ef.cursor();
        let _ = cur.advance();

        // S48-R9: Forged len = 2^40 or usize::MAX must be clamped against high_bits popcount
        // and low_bits capacity so `.iter().collect()` and `get_low` never OOM-abort or overflow.
        let mut forged_len: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&ef).unwrap()).unwrap();
        forged_len["len"] = serde_json::json!(1u64 << 40);
        forged_len["low_bit_width"] = serde_json::json!(63);
        let f_len_ef: EliasFano = serde_json::from_value(forged_len).unwrap();
        assert!(f_len_ef.len() <= n as usize);
        let collected: Vec<u64> = f_len_ef.iter().collect();
        assert_eq!(collected.len(), f_len_ef.len());
    }
}
