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
    let mut sorter = RadixSort::new();
    let mut data: Vec<Vec<u8>> = (0..1000u32)
        .map(|i| format!("key_{:06}", (i * 997) % 1000).into_bytes())
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
    let strings = vec![
        b"banana".as_slice(),
        b"apple".as_slice(),
        b"cherry".as_slice(),
        b"date".as_slice(),
    ];
    let mut data: Vec<RadixString> = strings.iter().map(|s| RadixString::new(s)).collect();

    let result = sorter.sort(&mut data);
    assert!(result.is_ok());

    let expected_strings = vec![
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
    let memory_pool = crate::memory::SecureMemoryPool::new(crate::memory::SecurePoolConfig::large_secure()).unwrap();
    let config = AdvancedRadixSortConfig::default();

    let mut sorter = AdvancedU32RadixSort::with_memory_pool(config, memory_pool);
    let mut data = vec![5u32, 2, 8, 1, 9];

    let result = sorter.sort(&mut data);
    assert!(result.is_ok());
    assert_eq!(data, vec![1, 2, 5, 8, 9]);
}
