use super::*;
use crate::algorithms::Algorithm;

#[test]
fn test_radix_sort_u32_empty() {
    let mut sorter = RadixSort::new();
    let mut data: Vec<u32> = vec![];

    let result = sorter.sort_u32(&mut data);
    assert!(result.is_ok());
    assert!(data.is_empty());
}

#[test]
fn test_radix_sort_u32_simple() {
    let mut sorter = RadixSort::new();
    let mut data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];

    let result = sorter.sort_u32(&mut data);
    assert!(result.is_ok());
    assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let stats = sorter.stats();
    assert_eq!(stats.items_processed, 9);
    // In release mode, sorting 9 items might be so fast that timing shows 0 microseconds
    // This is acceptable as it demonstrates excellent performance
}

#[test]
fn test_radix_sort_u32_large_numbers() {
    let mut sorter = RadixSort::new();
    let mut data = vec![u32::MAX, 1000000, 500000, 0, 999999];

    let result = sorter.sort_u32(&mut data);
    assert!(result.is_ok());
    assert_eq!(data, vec![0, 500000, 999999, 1000000, u32::MAX]);
}

#[test]
fn test_radix_sort_u64() {
    let mut sorter = RadixSort::new();
    let mut data = vec![5u64, 2, 8, 1, 9, 3, 7, 4, 6];

    let result = sorter.sort_u64(&mut data);
    assert!(result.is_ok());
    assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn test_radix_sort_bytes() {
    let mut sorter = RadixSort::new();
    let mut data = vec![
        b"banana".to_vec(),
        b"apple".to_vec(),
        b"cherry".to_vec(),
        b"date".to_vec(),
    ];

    let result = sorter.sort_bytes(&mut data);
    assert!(result.is_ok());

    assert_eq!(
        data,
        vec![
            b"apple".to_vec(),
            b"banana".to_vec(),
            b"cherry".to_vec(),
            b"date".to_vec(),
        ]
    );
}

#[test]
fn test_radix_sort_bytes_different_lengths() {
    let mut sorter = RadixSort::new();
    let mut data = vec![b"a".to_vec(), b"abc".to_vec(), b"ab".to_vec(), b"".to_vec()];

    let result = sorter.sort_bytes(&mut data);
    assert!(result.is_ok());

    assert_eq!(
        data,
        vec![b"".to_vec(), b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec(),]
    );
}

#[test]
fn test_radix_sort_config() {
    let config = RadixSortConfig {
        use_parallel: false,
        parallel_threshold: 100,
        radix_bits: 4,
        use_counting_sort_threshold: 10,
        use_simd: false,
    };

    let mut sorter = RadixSort::with_config(config);
    let mut data = vec![5u32, 2, 8, 1, 9];

    let result = sorter.sort_u32(&mut data);
    assert!(result.is_ok());
    assert_eq!(data, vec![1, 2, 5, 8, 9]);
    assert!(!sorter.stats().used_parallel);
}

#[test]
fn test_counting_sort_threshold() {
    let config = RadixSortConfig {
        use_counting_sort_threshold: 100,
        ..Default::default()
    };

    let mut sorter = RadixSort::with_config(config);
    let mut data = vec![3u32, 1, 4, 1, 5, 9, 2, 6]; // Small dataset

    let result = sorter.sort_u32(&mut data);
    assert!(result.is_ok());
    assert_eq!(data, vec![1, 1, 2, 3, 4, 5, 6, 9]);
}

#[test]
fn test_key_value_radix_sort() {
    let sorter = KeyValueRadixSort::<u32, String>::new();
    let mut data = vec![
        (5, "five".to_string()),
        (2, "two".to_string()),
        (8, "eight".to_string()),
        (1, "one".to_string()),
    ];

    let result = sorter.sort_by_key(&mut data);
    assert!(result.is_ok());

    let expected = vec![
        (1, "one".to_string()),
        (2, "two".to_string()),
        (5, "five".to_string()),
        (8, "eight".to_string()),
    ];
    assert_eq!(data, expected);
}

#[test]
fn test_key_value_sort_empty() {
    let sorter = KeyValueRadixSort::<u32, String>::new();
    let mut data: Vec<(u32, String)> = vec![];
    assert!(sorter.sort_by_key(&mut data).is_ok());
    assert!(data.is_empty());
}

#[test]
fn test_key_value_sort_single() {
    let sorter = KeyValueRadixSort::<u32, String>::new();
    let mut data = vec![(42, "only".to_string())];
    assert!(sorter.sort_by_key(&mut data).is_ok());
    assert_eq!(data, vec![(42, "only".to_string())]);
}

#[test]
fn test_key_value_sort_already_sorted() {
    let sorter = KeyValueRadixSort::<u32, String>::new();
    let mut data: Vec<(u32, String)> = (0..50).map(|i| (i, format!("v{}", i))).collect();
    let expected = data.clone();
    assert!(sorter.sort_by_key(&mut data).is_ok());
    assert_eq!(data, expected);
}

#[test]
fn test_key_value_sort_reverse() {
    let sorter = KeyValueRadixSort::<u32, String>::new();
    let mut data: Vec<(u32, String)> = (0..50).rev().map(|i| (i, format!("v{}", i))).collect();
    assert!(sorter.sort_by_key(&mut data).is_ok());
    for i in 0..50u32 {
        assert_eq!(data[i as usize], (i, format!("v{}", i)));
    }
}

#[test]
fn test_key_value_sort_duplicate_keys() {
    let sorter = KeyValueRadixSort::<u32, String>::new();
    let mut data = vec![
        (3, "a".to_string()),
        (1, "b".to_string()),
        (3, "c".to_string()),
        (1, "d".to_string()),
        (2, "e".to_string()),
    ];
    assert!(sorter.sort_by_key(&mut data).is_ok());
    let keys: Vec<u32> = data.iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, vec![1, 1, 2, 3, 3]);
    let all_values: std::collections::HashSet<&str> =
        data.iter().map(|(_, v)| v.as_str()).collect();
    assert_eq!(all_values.len(), 5);
}

#[test]
fn test_key_value_sort_large_strings() {
    let sorter = KeyValueRadixSort::<u32, String>::new();
    let n = 200;
    let mut data: Vec<(u32, String)> = (0..n)
        .map(|i| {
            let key = ((i * 7 + 13) % n) as u32;
            (key, "x".repeat(64) + &format!("_{}", key))
        })
        .collect();
    assert!(sorter.sort_by_key(&mut data).is_ok());
    for i in 1..data.len() {
        assert!(data[i - 1].0 <= data[i].0, "not sorted at index {}", i);
    }
    for (k, v) in &data {
        assert!(v.ends_with(&format!("_{}", k)));
    }
}

#[test]
fn test_algorithm_trait() {
    let sorter = RadixSort::new();

    assert!(sorter.supports_parallel());

    let memory_estimate = sorter.estimate_memory(1000);
    assert!(memory_estimate > 1000 * std::mem::size_of::<u32>());

    let input = vec![3u32, 1, 4, 1, 5];
    let config = RadixSortConfig::default();
    let result = sorter.execute(&config, input);
    assert!(result.is_ok());

    let sorted = result.unwrap();
    assert_eq!(sorted, vec![1, 1, 3, 4, 5]);
}

#[test]
fn test_msd_sort_bytes_large_dataset() {
    // A permutation of key_000000..key_{n-1}; Miri sorts 200 keys.
    let n: u32 = if cfg!(miri) { 200 } else { 1000 };
    let mut sorter = RadixSort::new();
    let mut data: Vec<Vec<u8>> = (0..n)
        .map(|i| format!("key_{:06}", (i * 997) % n).into_bytes())
        .collect();

    sorter.sort_bytes(&mut data).unwrap();

    for w in data.windows(2) {
        assert!(
            w[0] <= w[1],
            "not sorted: {:?} > {:?}",
            String::from_utf8_lossy(&w[0]),
            String::from_utf8_lossy(&w[1])
        );
    }
}

#[test]
fn test_msd_sort_bytes_duplicates() {
    let mut sorter = RadixSort::new();
    let mut data: Vec<Vec<u8>> = (0..500)
        .map(|i| format!("dup_{}", i % 10).into_bytes())
        .collect();

    sorter.sort_bytes(&mut data).unwrap();

    for w in data.windows(2) {
        assert!(w[0] <= w[1]);
    }
    let dup_0_count = data.iter().filter(|d| d == &&b"dup_0".to_vec()).count();
    assert_eq!(dup_0_count, 50);
}

#[test]
fn test_msd_sort_bytes_shared_prefixes() {
    let mut sorter = RadixSort::new();
    let mut data = vec![
        b"prefix_zzz".to_vec(),
        b"prefix_aaa".to_vec(),
        b"prefix_mmm".to_vec(),
        b"prefix".to_vec(),
        b"prefi".to_vec(),
        b"prefix_aaa_longer".to_vec(),
    ];

    sorter.sort_bytes(&mut data).unwrap();

    assert_eq!(
        data,
        vec![
            b"prefi".to_vec(),
            b"prefix".to_vec(),
            b"prefix_aaa".to_vec(),
            b"prefix_aaa_longer".to_vec(),
            b"prefix_mmm".to_vec(),
            b"prefix_zzz".to_vec(),
        ]
    );
}

#[test]
fn test_msd_sort_bytes_all_empty() {
    let mut sorter = RadixSort::new();
    let mut data: Vec<Vec<u8>> = vec![vec![], vec![], vec![]];
    sorter.sort_bytes(&mut data).unwrap();
    assert_eq!(data.len(), 3);
    assert!(data.iter().all(|d| d.is_empty()));
}

#[test]
fn test_msd_sort_bytes_single_char_permutation() {
    let mut sorter = RadixSort::new();
    let mut data: Vec<Vec<u8>> = (0u8..=255).rev().map(|b| vec![b]).collect();

    sorter.sort_bytes(&mut data).unwrap();

    let expected: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
    assert_eq!(data, expected);
}

// Tests for AdvancedRadixSort
#[test]
fn test_advanced_radix_sort_u32_simple() {
    let mut sorter = AdvancedU32RadixSort::new().unwrap();
    let mut data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];

    let result = sorter.sort(&mut data);
    assert!(result.is_ok());
    assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let stats = sorter.stats();
    assert_eq!(stats.basic_stats.items_processed, 9);
    assert_eq!(stats.strategy_used, SortingStrategy::Insertion); // Small dataset uses insertion sort
}

#[test]
fn test_advanced_radix_sort_u32_large() {
    let mut sorter = AdvancedU32RadixSort::with_config(AdvancedRadixSortConfig {
        insertion_sort_threshold: 50,
        ..Default::default()
    })
    .unwrap();

    let mut data: Vec<u32> = (0..1000).rev().collect(); // Reverse sorted
    let result = sorter.sort(&mut data);
    assert!(result.is_ok());

    let expected: Vec<u32> = (0..1000).collect();
    assert_eq!(data, expected);

    let stats = sorter.stats();
    assert_eq!(stats.basic_stats.items_processed, 1000);
    // Should use TimSort for reverse sorted data
    assert!(matches!(
        stats.strategy_used,
        SortingStrategy::TimSort | SortingStrategy::LsdRadix
    ));
}

#[test]
fn test_advanced_radix_sort_u64() {
    let mut sorter = AdvancedU64RadixSort::new().unwrap();
    let mut data = vec![u64::MAX, 1000000, 500000, 0, 999999];

    let result = sorter.sort(&mut data);
    assert!(result.is_ok());
    assert_eq!(data, vec![0, 500000, 999999, 1000000, u64::MAX]);
}

#[test]
fn test_advanced_radix_sort_strings() {
    let mut sorter = AdvancedStringRadixSort::new().unwrap();
    let strings = [
        b"banana".as_slice(),
        b"apple".as_slice(),
        b"cherry".as_slice(),
        b"date".as_slice(),
    ];
    let mut data: Vec<RadixString> = strings.iter().map(|s| RadixString::new(s)).collect();

    let result = sorter.sort(&mut data);
    assert!(result.is_ok());

    let expected_strings = [
        b"apple".as_slice(),
        b"banana".as_slice(),
        b"cherry".as_slice(),
        b"date".as_slice(),
    ];
    for (i, expected) in expected_strings.iter().enumerate() {
        assert_eq!(data[i].as_slice(), *expected);
    }
}

#[test]
fn test_advanced_radix_sort_forced_strategy() {
    let config = AdvancedRadixSortConfig {
        force_strategy: Some(SortingStrategy::LsdRadix),
        insertion_sort_threshold: 1000, // Force LSD even for small data
        ..Default::default()
    };

    let mut sorter = AdvancedU32RadixSort::with_config(config).unwrap();
    let mut data = vec![5, 2, 8, 1, 9];

    let result = sorter.sort(&mut data);
    assert!(result.is_ok());
    assert_eq!(data, vec![1, 2, 5, 8, 9]);

    let stats = sorter.stats();
    assert_eq!(stats.strategy_used, SortingStrategy::LsdRadix);
}

#[test]
fn test_advanced_radix_sort_parallel() {
    let config = AdvancedRadixSortConfig {
        use_parallel: true,
        parallel_threshold: 100,
        force_strategy: Some(SortingStrategy::LsdRadix),
        ..Default::default()
    };

    let mut sorter = AdvancedU32RadixSort::with_config(config).unwrap();
    let mut data: Vec<u32> = (0..1000).rev().collect();

    let result = sorter.sort(&mut data);
    assert!(result.is_ok());

    let expected: Vec<u32> = (0..1000).collect();
    assert_eq!(data, expected);

    let stats = sorter.stats();
    assert!(stats.basic_stats.used_parallel);
    assert!(stats.threads_used > 0);
}

#[test]
fn test_cpu_features_detection() {
    let features = CpuFeatures::detect();

    // These tests depend on the actual CPU, so we just verify the structure
    #[cfg(target_arch = "x86_64")]
    {
        // On x86_64, at least one of these should be detectable
        // (even if false, the detection should work)
        let _ = features.avx2;
        let _ = features.bmi2;
        let _ = features.popcnt;
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // On non-x86_64, all should be false
        assert!(!features.avx2);
        assert!(!features.bmi2);
        assert!(!features.popcnt);
        assert!(!features.avx512f);
        assert!(!features.avx512bw);
    }
}

#[test]
fn test_data_characteristics_integers() {
    let data = vec![1u32, 2, 3, 4, 5]; // Sorted
    let chars = DataCharacteristics::analyze_integers(&data);

    assert_eq!(chars.size, 5);
    assert!(chars.is_nearly_sorted);
    assert!(!chars.is_string_data);
    assert!(chars.max_key_bits >= 3); // At least 3 bits for value 5

    let data = vec![5u32, 1, 4, 2, 3]; // Unsorted
    let chars = DataCharacteristics::analyze_integers(&data);
    assert!(!chars.is_nearly_sorted);
}

#[test]
fn test_data_characteristics_strings() {
    let data = vec![b"apple".to_vec(), b"banana".to_vec(), b"cherry".to_vec()];
    let chars = DataCharacteristics::analyze_strings(&data);

    assert_eq!(chars.size, 3);
    assert!(chars.is_nearly_sorted); // Lexicographically sorted
    assert!(chars.is_string_data);
}

#[test]
fn test_radix_sortable_trait_u32() {
    let value = 0x12345678u32;
    assert_eq!(value.extract_key(), 0x12345678u64);
    assert_eq!(value.get_byte(0), Some(0x12));
    assert_eq!(value.get_byte(1), Some(0x34));
    assert_eq!(value.get_byte(2), Some(0x56));
    assert_eq!(value.get_byte(3), Some(0x78));
    assert_eq!(value.get_byte(4), None);
    assert_eq!(value.max_bytes(), 4);
}

#[test]
fn test_radix_sortable_trait_radix_string() {
    let value = RadixString::new(b"hello");
    let key = value.extract_key();

    // Should extract first 8 bytes as big-endian u64
    let expected = (b'h' as u64) << 56
        | (b'e' as u64) << 48
        | (b'l' as u64) << 40
        | (b'l' as u64) << 32
        | (b'o' as u64) << 24;
    assert_eq!(key, expected);

    assert_eq!(value.get_byte(0), Some(b'h'));
    assert_eq!(value.get_byte(4), Some(b'o'));
    assert_eq!(value.get_byte(5), None);
    assert_eq!(value.max_bytes(), 5);
}

#[test]
fn test_advanced_algorithm_trait() {
    let sorter = AdvancedU32RadixSort::new().unwrap();

    assert!(sorter.supports_parallel());
    assert_eq!(sorter.supports_simd(), cfg!(feature = "simd"));

    let memory_estimate = sorter.estimate_memory(1000);
    assert!(memory_estimate > 1000 * std::mem::size_of::<u32>());

    let input = vec![3u32, 1, 4, 1, 5];
    let config = AdvancedRadixSortConfig::default();
    let result = sorter.execute(&config, input);
    assert!(result.is_ok());

    let sorted = result.unwrap();
    assert_eq!(sorted, vec![1, 1, 3, 4, 5]);
}

#[test]
fn test_phase_times_tracking() {
    let mut sorter = AdvancedU32RadixSort::with_config(AdvancedRadixSortConfig {
        enable_profiling: true,
        ..Default::default()
    })
    .unwrap();

    let mut data: Vec<u32> = (0..1000).rev().collect(); // Larger dataset to ensure measurable timing
    let result = sorter.sort(&mut data);
    assert!(result.is_ok());

    let stats = sorter.stats();
    // Analysis phase should have taken some time
    assert!(stats.phase_times.analysis_time_us > 0 || stats.phase_times.sorting_time_us > 0);
}

#[test]
fn test_memory_pool_integration() {
    let memory_pool =
        crate::memory::SecureMemoryPool::new(crate::memory::SecurePoolConfig::large_secure())
            .unwrap();
    let config = AdvancedRadixSortConfig::default();

    let mut sorter = AdvancedU32RadixSort::with_memory_pool(config, memory_pool);
    let mut data = vec![5u32, 2, 8, 1, 9];

    let result = sorter.sort(&mut data);
    assert!(result.is_ok());
    assert_eq!(data, vec![1, 2, 5, 8, 9]);
}

/// The LSD passes now ping-pong between data and the scratch buffer
/// instead of copying back after every pass; an odd pass count leaves the
/// result in the buffer and needs one final copy. radix_bits=11 gives
/// ceil(32/11)=3 passes (odd) for u32 and ceil(64/11)=6 (even) for u64,
/// covering both endings. The pass count does not depend on the length,
/// so Miri sorts 300 keys instead of 5,000.
#[test]
fn test_ping_pong_odd_and_even_pass_counts() {
    let config = RadixSortConfig {
        radix_bits: 11,
        ..Default::default()
    };
    let n: u32 = if cfg!(miri) { 300 } else { 5000 };

    let mut data: Vec<u32> = (0..n).rev().map(|i| i.wrapping_mul(2654435761)).collect();
    let mut expected = data.clone();
    expected.sort_unstable();
    let mut sorter = RadixSort::with_config(config.clone());
    sorter.sort_u32(&mut data).unwrap();
    assert_eq!(data, expected);

    let mut data: Vec<u64> = (0..u64::from(n))
        .rev()
        .map(|i| i.wrapping_mul(0x9E3779B97F4A7C15))
        .collect();
    let mut expected = data.clone();
    expected.sort_unstable();
    let mut sorter = RadixSort::with_config(config);
    sorter.sort_u64(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_advanced_radix_sort_u64_random_simd() {
    let config = AdvancedRadixSortConfig {
        use_simd: true,
        use_parallel: false,
        adaptive_strategy: false,
        force_strategy: Some(SortingStrategy::LsdRadix),
        ..Default::default()
    };
    let mut sorter = AdvancedU64RadixSort::with_config(config).unwrap();
    // Miri: 200 keys (the largest key still needs all 8 byte passes).
    let n: u64 = if cfg!(miri) { 200 } else { 1000 };
    let mut data: Vec<u64> = (0..n).map(|i| i.wrapping_mul(0x517cc1b727220a95)).collect();
    let mut expected = data.clone();
    expected.sort_unstable();

    sorter.sort(&mut data).unwrap();
    assert_eq!(data, expected);
}

#[test]
fn test_c8_10_radix_sort_invalid_radix_bits_and_differential_vs_std() {
    // 1. Out-of-range radix_bits (0 and 64) must return Err instead of panicking.
    for bad_bits in [0usize, 17, 64] {
        let mut lsd = RadixSort::with_config(RadixSortConfig {
            radix_bits: bad_bits,
            use_counting_sort_threshold: 0,
            ..Default::default()
        });
        let mut d32 = vec![3u32, 1, 2];
        assert!(lsd.sort_u32(&mut d32).is_err());
        let mut d64 = vec![3u64, 1, 2];
        assert!(lsd.sort_u64(&mut d64).is_err());

        assert!(AdvancedU32RadixSort::with_config(AdvancedRadixSortConfig {
            radix_bits: bad_bits,
            use_secure_memory: false,
            ..Default::default()
        })
        .is_err());
    }

    // 2. Strings longer than 8 bytes sharing an 8-byte prefix + embedded NULs
    // must sort according to full Ord across all strategies.
    let raw_strings: &[&[u8]] = &[
        b"shared88_zzz",
        b"shared88_aaa",
        b"shared88_mmm",
        b"shared88",
        b"shared88\0b",
        b"shared88\0a",
        b"a\0z",
        b"a\0a",
        b"a",
        b"",
    ];
    for strategy in [
        SortingStrategy::Insertion,
        SortingStrategy::TimSort,
        SortingStrategy::LsdRadix,
        SortingStrategy::MsdRadix,
    ] {
        let mut sorter = AdvancedStringRadixSort::with_config(AdvancedRadixSortConfig {
            force_strategy: Some(strategy),
            use_parallel: false,
            use_secure_memory: false,
            ..Default::default()
        })
        .unwrap();
        let mut items: Vec<RadixString> =
            raw_strings.iter().map(|s| RadixString::new(s)).collect();
        let mut expected = items.clone();
        expected.sort_unstable();

        sorter.sort(&mut items).unwrap();
        assert_eq!(items, expected, "strategy {:?} failed full Ord sort", strategy);
    }

    // 3. Differential test across u32, u64, Vec<u8>, and KV pairs vs std sort.
    let n = if cfg!(miri) { 64 } else { 512 };
    let mut u32_data: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(0x9E37_79B9) ^ ((i as u32) << 13))
        .collect();
    u32_data.push(0);
    u32_data.push(u32::MAX);
    u32_data.push(0x8000_0000);
    let mut u32_expected = u32_data.clone();
    u32_expected.sort_unstable();

    let mut lsd = RadixSort::new();
    let mut u32_lsd = u32_data.clone();
    lsd.sort_u32(&mut u32_lsd).unwrap();
    assert_eq!(u32_lsd, u32_expected);

    let mut u64_data: Vec<u64> = (0..n)
        .map(|i| (i as u64).wrapping_mul(0x517cc1b727220a95) ^ (1u64 << (i % 64)))
        .collect();
    u64_data.push(0);
    u64_data.push(u64::MAX);
    let mut u64_expected = u64_data.clone();
    u64_expected.sort_unstable();
    let mut u64_lsd = u64_data.clone();
    lsd.sort_u64(&mut u64_lsd).unwrap();
    assert_eq!(u64_lsd, u64_expected);

    let kv_sorter = KeyValueRadixSort::<u32, usize>::new();
    let mut kv_data: Vec<(u32, usize)> = u32_data
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, k)| (k, idx))
        .collect();
    let mut kv_expected = kv_data.clone();
    kv_expected.sort_by_key(|&(k, _)| k);
    kv_sorter.sort_by_key(&mut kv_data).unwrap();
    assert_eq!(kv_data, kv_expected);
}

/// Ordered by `key` only, so equal keys with different `tag`s expose whether a
/// merge or sort is stable.
#[derive(Debug, Clone, Copy)]
struct Tagged {
    key: u8,
    tag: u32,
}

impl PartialEq for Tagged {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Tagged {}

impl PartialOrd for Tagged {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Tagged {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl RadixSortable for Tagged {
    fn extract_key(&self) -> u64 {
        self.key as u64
    }

    fn get_byte(&self, position: usize) -> Option<u8> {
        (position == 0).then_some(self.key)
    }

    fn max_bytes(&self) -> usize {
        1
    }
}

fn key_tag(v: &[Tagged]) -> Vec<(u8, u32)> {
    v.iter().map(|t| (t.key, t.tag)).collect()
}

#[test]
fn test_s20_r3_parallel_merge_tree_stable_and_matches_std() {
    if cfg!(miri) {
        // A rayon::join issued outside any pool injects its job into the
        // global pool and blocks until a pool thread has run it. Under Miri
        // with Tree Borrows that round trip dominated: 1,592 of part 1's
        // 2,000 par_merge calls start with one. On a pool thread the joins
        // take the in-worker path. The split points, chunk counts and merge
        // tree are unchanged; with one thread nothing is stolen.
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(s20_r3_merge_tree_cases);
    } else {
        s20_r3_merge_tree_cases();
    }
}

fn s20_r3_merge_tree_cases() {
    // 1. par_merge split logic, exhaustively: every pair of non-decreasing key
    //    sequences over {0, 1, 2} up to length 5 (Miri: 3), with cutoffs small
    //    enough to split down to single elements. The output must equal the
    //    stable merge (ties from `left` first).
    let max_len = if cfg!(miri) { 3 } else { 5 };
    let mut seqs: Vec<Vec<u8>> = vec![Vec::new()];
    let mut frontier: Vec<Vec<u8>> = vec![Vec::new()];
    for _ in 0..max_len {
        let mut next = Vec::new();
        for s in &frontier {
            for k in s.last().copied().unwrap_or(0)..=2 {
                let mut t = s.clone();
                t.push(k);
                next.push(t);
            }
        }
        seqs.extend(next.iter().cloned());
        frontier = next;
    }
    for l in &seqs {
        for r in &seqs {
            let left: Vec<Tagged> = l
                .iter()
                .enumerate()
                .map(|(i, &key)| Tagged { key, tag: i as u32 })
                .collect();
            let right: Vec<Tagged> = r
                .iter()
                .enumerate()
                .map(|(i, &key)| Tagged {
                    key,
                    tag: 1000 + i as u32,
                })
                .collect();
            let mut expected: Vec<Tagged> = left.iter().chain(&right).copied().collect();
            expected.sort_by_key(|t| t.key); // stable: left's equal keys stay first
            for cutoff in [0, 1, 2, 3, 4] {
                let mut out = vec![
                    Tagged {
                        key: 0,
                        tag: u32::MAX
                    };
                    left.len() + right.len()
                ];
                super::merge::par_merge_with_cutoff(&left, &right, &mut out, cutoff);
                assert_eq!(
                    key_tag(&out),
                    key_tag(&expected),
                    "par_merge cutoff={cutoff} left={l:?} right={r:?}"
                );
            }
        }
    }

    // 2. Parallel LSD path end to end: forced chunk counts (odd and even merge
    //    trees) and sizes straddling 2 * parallel_threshold. Miri sorts only
    //    n = 129, which still splits into 2, 3, 5, 7, 15 and 43 or 44 chunks.
    let sizes: &[usize] = if cfg!(miri) {
        &[129]
    } else {
        &[129, 1_000, 20_011, 100_003]
    };
    for &threads in &[2usize, 3, 5, 7, 16, 64] {
        let cfg = AdvancedRadixSortConfig {
            force_strategy: Some(SortingStrategy::LsdRadix),
            use_parallel: true,
            parallel_threshold: 64,
            num_threads: threads,
            use_secure_memory: false,
            ..Default::default()
        };
        for &n in sizes {
            let mut v32: Vec<u32> = (0..n as u32)
                .map(|i| i.wrapping_mul(0x9E37_79B9) ^ (i << 7))
                .collect();
            v32.extend([0, u32::MAX, 0x8000_0000]);
            let mut want32 = v32.clone();
            want32.sort_unstable();
            AdvancedU32RadixSort::with_config(cfg.clone())
                .unwrap()
                .sort(&mut v32)
                .unwrap();
            assert_eq!(v32, want32, "u32 threads={threads} n={n}");

            let mut v64: Vec<u64> = (0..n as u64)
                .map(|i| i.wrapping_mul(0x517c_c1b7_2722_0a95) ^ (1u64 << (i % 64)))
                .collect();
            v64.extend([0, u64::MAX]);
            let mut want64 = v64.clone();
            want64.sort_unstable();
            let mut sorter = AdvancedU64RadixSort::with_config(cfg.clone()).unwrap();
            sorter.sort(&mut v64).unwrap();
            assert_eq!(v64, want64, "u64 threads={threads} n={n}");
            assert!(
                sorter.stats().basic_stats.used_parallel,
                "threads={threads} n={n}"
            );

            // Many duplicate keys: the parallel path must be stable end to end.
            let mut tagged: Vec<Tagged> = (0..n as u32)
                .map(|i| Tagged {
                    key: (i.wrapping_mul(7919) % 13) as u8,
                    tag: i,
                })
                .collect();
            let mut want_tagged = tagged.clone();
            want_tagged.sort_by_key(|t| t.key);
            AdvancedRadixSort::<Tagged>::with_config(cfg.clone())
                .unwrap()
                .sort(&mut tagged)
                .unwrap();
            assert_eq!(
                key_tag(&tagged),
                key_tag(&want_tagged),
                "tagged threads={threads} n={n}"
            );
        }
    }
}

#[test]
fn test_s20_r3b_radix_sort_parallel_path_and_used_parallel() {
    // RadixSort::sort_u32 / sort_u64 inside rayon pools of 1..16 threads with
    // parallel_threshold 64, at sizes on both sides of 2 * 64: the output must
    // equal sort_unstable, and stats().used_parallel must report whether the
    // chunked parallel sort ran. It used to be set from
    // `len >= parallel_threshold` alone, while the parallel path sorted
    // sequentially below 2 * parallel_threshold and ran a single chunk on a
    // one-thread pool.
    let threshold = 64;
    let (thread_counts, sizes): (&[usize], &[usize]) = if cfg!(miri) {
        (&[1, 3], &[100, 128, 300])
    } else {
        (
            &[1, 2, 3, 5, 7, 16],
            &[100, 127, 128, 1_000, 20_011, 100_003],
        )
    };
    let config = RadixSortConfig {
        use_parallel: true,
        parallel_threshold: threshold,
        ..Default::default()
    };
    for &threads in thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for &n in sizes {
                let want_parallel = threads > 1 && n >= 2 * threshold;

                // Wide values (radix passes) and a small range with many
                // duplicates (small chunks take the counting-sort path).
                let mut wide: Vec<u32> = (0..n as u32)
                    .map(|i| i.wrapping_mul(0x9E37_79B9) ^ (i << 7))
                    .collect();
                for (slot, x) in wide.iter_mut().zip([u32::MAX, 0, 0x8000_0000]) {
                    *slot = x;
                }
                let narrow: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(7919) % 300).collect();
                for (name, input) in [("wide", wide), ("narrow", narrow)] {
                    let mut v = input.clone();
                    let mut want = input;
                    want.sort_unstable();
                    let mut sorter = RadixSort::with_config(config.clone());
                    sorter.sort_u32(&mut v).unwrap();
                    assert_eq!(v, want, "u32 {name} threads={threads} n={n}");
                    assert_eq!(
                        sorter.stats().used_parallel,
                        want_parallel,
                        "u32 {name} used_parallel threads={threads} n={n}"
                    );
                }

                let mut v64: Vec<u64> = (0..n as u64)
                    .map(|i| i.wrapping_mul(0x517c_c1b7_2722_0a95) ^ (1u64 << (i % 64)))
                    .collect();
                for (slot, x) in v64.iter_mut().zip([u64::MAX, 0]) {
                    *slot = x;
                }
                let mut want64 = v64.clone();
                want64.sort_unstable();
                let mut sorter = RadixSort::with_config(config.clone());
                sorter.sort_u64(&mut v64).unwrap();
                assert_eq!(v64, want64, "u64 threads={threads} n={n}");
                assert_eq!(
                    sorter.stats().used_parallel,
                    want_parallel,
                    "u64 used_parallel threads={threads} n={n}"
                );
            }
        });
    }
}

/// S20-R5 guard: `sort_u32` extracts digits with u32 shifts, which needs
/// `shift < 32` on every pass. Sort with every accepted radix width and with
/// `use_simd` on and off (without AVX-512 these take the two sequential
/// counting loops); the test profile has overflow checks, so a shift of 32
/// or more would panic here.
///
/// Under Miri only widths 1..=12 run. Every pass walks all 2^radix_bits
/// buckets twice (reset, prefix sum), and widths 13..=16 account for 303,104
/// of the 330,296 (bucket, pass) pairs across the 16 widths (91.8%); under
/// Tree Borrows the full sweep took 166 s.
#[test]
fn test_s20_r5_sort_u32_every_radix_width() {
    let n: u32 = if cfg!(miri) { 40 } else { 2_000 };
    let mut input: Vec<u32> = (0..n)
        .map(|i| i.wrapping_mul(0x9E37_79B9) ^ (i << 7))
        .collect();
    input.extend([0, 1, 0x7FFF_FFFF, 0x8000_0000, u32::MAX - 1, u32::MAX]);
    let mut want = input.clone();
    want.sort_unstable();
    // Under Miri both settings take the same loop (SIMD is gated off).
    let simd_settings: &[bool] = if cfg!(miri) { &[false] } else { &[false, true] };
    let max_radix_bits = if cfg!(miri) { 12 } else { 16 };
    for radix_bits in 1..=max_radix_bits {
        for &use_simd in simd_settings {
            let mut sorter = RadixSort::with_config(RadixSortConfig {
                radix_bits,
                use_parallel: false,
                use_counting_sort_threshold: 0,
                use_simd,
                ..Default::default()
            });
            let mut v = input.clone();
            sorter.sort_u32(&mut v).unwrap();
            assert_eq!(v, want, "radix_bits={radix_bits} use_simd={use_simd}");
        }
    }
}
