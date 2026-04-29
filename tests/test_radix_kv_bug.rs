use zipora::algorithms::radix_sort::KeyValueRadixSort;

#[test]
fn test_radix_kv_duplicate_keys() {
    let mut data: Vec<(u64, String)> = vec![
        (5, "A".to_string()),
        (2, "B".to_string()),
        (5, "C".to_string()),
        (1, "D".to_string()),
    ];
    let sorter = KeyValueRadixSort::new();
    sorter.sort_by_key(&mut data).unwrap();

    // Expected: [(1, "D"), (2, "B"), (5, "A"), (5, "C")]
    // If it finds the first match for 5, it'll output "A" twice instead of "A" and "C"
    let values: Vec<String> = data.into_iter().map(|(_, v)| v).collect();
    assert_eq!(
        values,
        vec![
            "D".to_string(),
            "B".to_string(),
            "A".to_string(),
            "C".to_string()
        ]
    );
}
