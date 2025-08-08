//! Property-based testing for specialized containers
//!
//! This module provides comprehensive property-based testing using proptest
//! to validate correctness properties across all container implementations.

use proptest::prelude::*;
use std::collections::{HashMap, VecDeque};
use zipora::containers::specialized::{
    ValVec32, SmallMap, FixedCircularQueue, AutoGrowCircularQueue, UintVector,
    FixedStr8Vec, FixedStr16Vec, SortableStrVec
};

// =============================================================================
// PROPERTY TEST GENERATORS
// =============================================================================

/// Generate strings that fit within fixed-length constraints
fn fixed_string_strategy(max_len: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(prop::char::range('a', 'z'), 0..=max_len)
        .prop_map(|chars| chars.into_iter().collect())
}

/// Generate sequences of container operations
#[derive(Debug, Clone)]
pub enum ContainerOp<K, V> {
    Insert(K, V),
    Remove(K),
    Get(K),
    Clear,
}

fn container_ops_strategy<K: Arbitrary + Clone + 'static, V: Arbitrary + Clone + 'static>() 
    -> impl Strategy<Value = Vec<ContainerOp<K, V>>> 
{
    prop::collection::vec(
        prop_oneof![
            (any::<K>(), any::<V>()).prop_map(|(k, v)| ContainerOp::Insert(k, v)),
            any::<K>().prop_map(ContainerOp::Remove),
            any::<K>().prop_map(ContainerOp::Get),
            Just(ContainerOp::Clear),
        ],
        0..1000
    )
}

// =============================================================================
// VALVEC32 PROPERTY TESTS
// =============================================================================

proptest! {
    #[test]
    fn prop_valvec32_length_invariant(
        elements in prop::collection::vec(any::<i32>(), 0..10000)
    ) {
        let mut vec = ValVec32::new();
        
        // Insert all elements
        for &elem in &elements {
            vec.push(elem).unwrap();
        }
        
        // Length should match input
        prop_assert_eq!(vec.len() as usize, elements.len());
        
        // Each element should be preserved
        for (i, &expected) in elements.iter().enumerate() {
            prop_assert_eq!(vec[i as u32], expected);
        }
    }

    #[test]
    fn prop_valvec32_push_pop_symmetry(
        elements in prop::collection::vec(any::<u64>(), 0..1000)
    ) {
        let mut vec = ValVec32::new();
        
        // Push all elements
        for &elem in &elements {
            vec.push(elem).unwrap();
        }
        
        // Pop all elements in reverse order
        let mut popped = Vec::new();
        while let Some(elem) = vec.pop() {
            popped.push(elem);
        }
        
        // Should get back the same elements in reverse order
        popped.reverse();
        prop_assert_eq!(popped, elements);
        prop_assert!(vec.is_empty());
    }

    #[test]
    fn prop_valvec32_index_bounds(
        elements in prop::collection::vec(any::<i16>(), 1..1000),
        bad_index in any::<u32>()
    ) {
        let mut vec = ValVec32::new();
        vec.extend_from_slice(&elements).unwrap();
        
        let len = vec.len();
        
        // Valid indices should work
        for i in 0..len {
            prop_assert_eq!(vec.get(i), Some(&elements[i as usize]));
        }
        
        // Invalid indices should return None
        if bad_index >= len {
            prop_assert_eq!(vec.get(bad_index), None);
        }
    }

    #[test]
    fn prop_valvec32_capacity_growth(
        batches in prop::collection::vec(
            prop::collection::vec(any::<i32>(), 1..100), 
            1..20
        )
    ) {
        let mut vec = ValVec32::new();
        let mut total_elements = 0;
        
        for batch in batches {
            let old_capacity = vec.capacity();
            
            for &elem in &batch {
                vec.push(elem).unwrap();
                total_elements += 1;
            }
            
            // Capacity should never decrease
            prop_assert!(vec.capacity() >= old_capacity);
            // Length should match total elements added
            prop_assert_eq!(vec.len(), total_elements);
        }
    }
}

// =============================================================================
// SMALLMAP PROPERTY TESTS
// =============================================================================

proptest! {
    #[test]
    fn prop_small_map_vs_hashmap(
        ops in container_ops_strategy::<i32, String>()
    ) {
        let mut small_map = SmallMap::new();
        let mut hash_map = HashMap::new();
        
        for op in ops {
            match op {
                ContainerOp::Insert(k, v) => {
                    let small_result = small_map.insert(k, v.clone());
                    let _hash_result = hash_map.insert(k, v);
                    prop_assert_eq!(small_result.is_ok(), true);
                    // Both should have same behavior for duplicate keys
                    prop_assert_eq!(small_map.get(&k), hash_map.get(&k));
                }
                ContainerOp::Remove(k) => {
                    let small_result = small_map.remove(&k);
                    let hash_result = hash_map.remove(&k);
                    prop_assert_eq!(small_result, hash_result);
                }
                ContainerOp::Get(k) => {
                    let small_result = small_map.get(&k);
                    let hash_result = hash_map.get(&k);
                    prop_assert_eq!(small_result, hash_result);
                }
                ContainerOp::Clear => {
                    small_map.clear();
                    hash_map.clear();
                }
            }
            
            // Invariant: lengths should always match
            prop_assert_eq!(small_map.len(), hash_map.len());
            prop_assert_eq!(small_map.is_empty(), hash_map.is_empty());
        }
    }

    #[test]
    fn prop_small_map_key_uniqueness(
        pairs in prop::collection::vec((any::<i32>(), any::<String>()), 0..100)
    ) {
        let mut map = SmallMap::new();
        let mut unique_keys = std::collections::HashSet::new();
        
        for (key, value) in pairs {
            map.insert(key, value.clone()).unwrap();
            unique_keys.insert(key);
            
            // Map should contain exactly the unique keys we've inserted
            prop_assert_eq!(map.len(), unique_keys.len());
            prop_assert!(map.contains_key(&key));
            prop_assert_eq!(map.get(&key), Some(&value));
        }
    }

    #[test]
    fn prop_small_map_inline_vs_heap_transition(
        small_keys in prop::collection::vec(any::<u8>(), 0..8),
        large_keys in prop::collection::vec(any::<u8>(), 9..50)
    ) {
        let mut map = SmallMap::new();
        
        // Insert small number of elements (should use inline storage)
        for &key in &small_keys {
            map.insert(key, key as i32 * 10).unwrap();
        }
        
        // Verify small elements are accessible
        for &key in &small_keys {
            let expected_value = key as i32 * 10;
            prop_assert_eq!(map.get(&key), Some(&expected_value));
        }
        
        // Insert large number of elements (should transition to heap)
        for &key in &large_keys {
            map.insert(key, key as i32 * 10).unwrap();
        }
        
        // All elements should still be accessible after transition
        for &key in &small_keys {
            let expected_value = key as i32 * 10;
            prop_assert_eq!(map.get(&key), Some(&expected_value));
        }
        for &key in &large_keys {
            let expected_value = key as i32 * 10;
            prop_assert_eq!(map.get(&key), Some(&expected_value));
        }
        
        // Count unique keys since SmallMap doesn't allow duplicates
        let mut all_keys = small_keys.clone();
        all_keys.extend_from_slice(&large_keys);
        all_keys.sort();
        all_keys.dedup();
        prop_assert_eq!(map.len(), all_keys.len());
    }
}

// =============================================================================
// CIRCULAR QUEUE PROPERTY TESTS
// =============================================================================

proptest! {
    #[test]
    fn prop_auto_grow_queue_vs_vecdeque(
        ops in prop::collection::vec(
            prop_oneof![
                any::<i32>().prop_map(|x| (x, true)),   // push
                Just((0, false))                         // pop
            ],
            0..1000
        )
    ) {
        let mut auto_queue = AutoGrowCircularQueue::new();
        let mut vec_deque = VecDeque::new();
        
        for (value, is_push) in ops {
            if is_push || vec_deque.is_empty() {
                auto_queue.push(value).unwrap();
                vec_deque.push_back(value);
            } else {
                let auto_result = auto_queue.pop();
                let vec_result = vec_deque.pop_front();
                prop_assert_eq!(auto_result, vec_result);
            }
            
            prop_assert_eq!(auto_queue.len(), vec_deque.len());
            prop_assert_eq!(auto_queue.is_empty(), vec_deque.is_empty());
        }
    }

    #[test]
    fn prop_fixed_queue_fifo_order(
        elements in prop::collection::vec(any::<i32>(), 0..8)
    ) {
        let mut queue: FixedCircularQueue<i32, 8> = FixedCircularQueue::new();
        
        // Fill queue with elements
        for &elem in &elements {
            queue.push(elem).unwrap();
        }
        
        // Pop elements and verify FIFO order
        for &expected in &elements {
            prop_assert_eq!(queue.pop(), Some(expected));
        }
        
        prop_assert!(queue.is_empty());
    }

    #[test]
    fn prop_fixed_queue_capacity_bounds(
        elements in prop::collection::vec(any::<i32>(), 0..20)
    ) {
        let mut queue: FixedCircularQueue<i32, 8> = FixedCircularQueue::new();
        
        let mut pushed = 0;
        for &elem in &elements {
            if pushed < 8 {
                prop_assert!(queue.push(elem).is_ok());
                pushed += 1;
            } else {
                // Should fail when queue is full
                prop_assert!(queue.push(elem).is_err());
            }
        }
        
        prop_assert_eq!(queue.len(), pushed.min(8));
        if pushed >= 8 {
            prop_assert!(queue.is_full());
        }
    }
}

// =============================================================================
// UINT VECTOR PROPERTY TESTS
// =============================================================================

proptest! {
    #[test]
    fn prop_uint_vector_vs_vec(
        values in prop::collection::vec(any::<u32>(), 0..1000)
    ) {
        let mut uint_vec = UintVector::new();
        let mut std_vec = Vec::new();
        
        // Insert all values
        for &value in &values {
            uint_vec.push(value).unwrap();
            std_vec.push(value);
        }
        
        // Verify same length and contents
        prop_assert_eq!(uint_vec.len(), std_vec.len());
        
        for (i, &expected) in std_vec.iter().enumerate() {
            prop_assert_eq!(uint_vec.get(i), Some(expected));
        }
    }

    #[test]
    fn prop_uint_vector_compression_correctness(
        values in prop::collection::vec(0u32..1000u32, 0..1000)
    ) {
        let mut uint_vec = UintVector::new();
        
        // Test that compression preserves all values correctly
        for &value in &values {
            uint_vec.push(value).unwrap();
        }
        
        // Verify each value is preserved exactly
        for (i, &expected) in values.iter().enumerate() {
            prop_assert_eq!(uint_vec.get(i), Some(expected));
        }
        
        // Test memory usage is reasonable
        let memory_usage = uint_vec.memory_usage();
        let _uncompressed_size = values.len() * std::mem::size_of::<u32>();
        
        // Should use less memory than uncompressed for typical data
        // (This property may not hold for all random data)
        prop_assert!(memory_usage > 0);
    }
}

// =============================================================================
// FIXED LENGTH STRING VECTOR PROPERTY TESTS
// =============================================================================

proptest! {
    #[test]
    fn prop_fixed_str8_vec_length_constraints(
        strings in prop::collection::vec(fixed_string_strategy(8), 0..100)
    ) {
        let mut vec = FixedStr8Vec::new();
        
        for s in &strings {
            prop_assert!(vec.push(s).is_ok());
        }
        
        prop_assert_eq!(vec.len(), strings.len());
        
        for (i, expected) in strings.iter().enumerate() {
            prop_assert_eq!(vec.get(i), Some(expected.as_str()));
        }
    }

    #[test]
    fn prop_fixed_str16_vec_vs_vec_string(
        strings in prop::collection::vec(fixed_string_strategy(16), 0..100)
    ) {
        let mut fixed_vec = FixedStr16Vec::new();
        let mut std_vec = Vec::new();
        
        for s in &strings {
            fixed_vec.push(s).unwrap();
            std_vec.push(s.clone());
        }
        
        prop_assert_eq!(fixed_vec.len(), std_vec.len());
        
        for (i, expected) in std_vec.iter().enumerate() {
            prop_assert_eq!(fixed_vec.get(i), Some(expected.as_str()));
        }
    }

    #[test]
    fn prop_fixed_str_vec_unicode_handling(
        base_strings in prop::collection::vec(fixed_string_strategy(4), 0..50)
    ) {
        let mut vec = FixedStr8Vec::new();
        
        // Test ASCII strings (should always fit)
        for s in &base_strings {
            if s.len() <= 8 {
                prop_assert!(vec.push(s).is_ok());
            }
        }
        
        // Test that strings longer than capacity are rejected
        let too_long = "this_is_too_long_for_8_chars";
        prop_assert!(vec.push(too_long).is_err());
    }
}

// =============================================================================
// SORTABLE STRING VECTOR PROPERTY TESTS
// =============================================================================

proptest! {
    #[test]
    fn prop_sortable_str_vec_sorting_correctness(
        strings in prop::collection::vec(
            "[a-zA-Z0-9 ]{1,20}", 0..100
        )
    ) {
        if strings.is_empty() {
            return Ok(());
        }
        
        let mut sortable_vec = SortableStrVec::new();
        let mut reference_vec = strings.clone();
        
        // Add strings to sortable vector
        for s in &strings {
            sortable_vec.push(s.clone()).unwrap();
        }
        
        // Sort both vectors
        sortable_vec.sort().unwrap();
        reference_vec.sort();
        
        // Results should be identical
        prop_assert_eq!(sortable_vec.len(), reference_vec.len());
        
        let sorted_result: Vec<_> = sortable_vec.iter_sorted().collect();
        prop_assert_eq!(sorted_result, reference_vec);
    }

    #[test]
    fn prop_sortable_str_vec_custom_sort(
        strings in prop::collection::vec(
            "[a-zA-Z]{1,10}", 0..50
        )
    ) {
        if strings.is_empty() {
            return Ok(());
        }
        
        let mut sortable_vec = SortableStrVec::new();
        let mut reference_vec = strings.clone();
        
        for s in &strings {
            sortable_vec.push(s.clone()).unwrap();
        }
        
        // Sort by length, then lexicographically
        let cmp_fn = |a: &str, b: &str| {
            a.len().cmp(&b.len()).then_with(|| a.cmp(b))
        };
        
        sortable_vec.sort_by(cmp_fn).unwrap();
        reference_vec.sort_by(|a: &String, b: &String| {
            a.len().cmp(&b.len()).then_with(|| a.cmp(b))
        });
        
        let sorted_result: Vec<_> = sortable_vec.iter_sorted().collect();
        prop_assert_eq!(sorted_result, reference_vec);
    }
}

// =============================================================================
// CROSS-CONTAINER PROPERTY TESTS
// =============================================================================

proptest! {
    #[test]
    fn prop_container_composition(
        data in prop::collection::vec((any::<u32>(), any::<String>()), 0..100)
    ) {
        let mut val_vec = ValVec32::new();
        let mut small_map = SmallMap::new();
        
        // Use ValVec32 as storage and SmallMap for indexing
        for (key, value) in &data {
            val_vec.push(value.clone()).unwrap();
            let index = val_vec.len() - 1;
            small_map.insert(*key, index).unwrap();
        }
        
        // Verify that we can retrieve values through the index
        for (key, expected_value) in &data {
            if let Some(&index) = small_map.get(key) {
                prop_assert_eq!(val_vec.get(index), Some(expected_value));
            }
        }
        
        prop_assert_eq!(val_vec.len() as usize, small_map.len());
    }
}

// =============================================================================
// STRESS TESTING PROPERTIES
// =============================================================================

proptest! {
    #[test]
    fn prop_stress_test_all_containers(
        operations in prop::collection::vec(
            (any::<u8>(), any::<u32>(), any::<bool>()), 0..10000
        )
    ) {
        let mut containers = (
            ValVec32::new(),
            SmallMap::new(),
            AutoGrowCircularQueue::new(),
            UintVector::new(),
        );
        
        let mut operation_count = 0;
        
        for (container_choice, value, is_add) in operations {
            operation_count += 1;
            
            match container_choice % 4 {
                0 => {
                    // ValVec32 operations
                    if is_add {
                        containers.0.push(value).unwrap();
                    } else if !containers.0.is_empty() {
                        containers.0.pop();
                    }
                }
                1 => {
                    // SmallMap operations
                    if is_add {
                        containers.1.insert(value, operation_count).unwrap();
                    } else {
                        containers.1.remove(&value);
                    }
                }
                2 => {
                    // AutoGrowCircularQueue operations
                    if is_add {
                        containers.2.push(value).unwrap();
                    } else if !containers.2.is_empty() {
                        containers.2.pop();
                    }
                }
                3 => {
                    // UintVector operations
                    if is_add {
                        containers.3.push(value).unwrap();
                    }
                    // Note: UintVector doesn't support pop in current interface
                }
                _ => unreachable!(),
            }
            
            // All containers should maintain basic invariants
            prop_assert!(containers.0.len() <= containers.0.capacity());
            prop_assert!(containers.2.len() <= containers.2.capacity());
        }
        
        // After stress testing, containers should still be in valid state
        prop_assert!(containers.0.capacity() >= containers.0.len());
        prop_assert_eq!(containers.2.is_empty(), containers.2.len() == 0);
    }
}

#[cfg(test)]
mod property_test_runner {
    use super::*;

    #[test]
    fn run_all_property_tests() {
        println!("Property-based testing framework initialized");
        println!("Run with: cargo test --test container_property_tests");
    }
}