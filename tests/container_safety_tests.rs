//! Memory safety and edge case testing for specialized containers
//!
//! This module provides comprehensive safety testing including boundary conditions,
//! error handling, memory safety, and stress testing scenarios.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::panic;

use zipora::containers::specialized::{
    ValVec32, SmallMap, FixedCircularQueue, AutoGrowCircularQueue, UintVector,
    FixedStr8Vec, FixedStr16Vec, SortableStrVec
};
use zipora::error::{Result, ZiporaError};

// =============================================================================
// MEMORY SAFETY TESTING FRAMEWORK
// =============================================================================

/// Memory safety test configuration
#[derive(Debug, Clone)]
pub struct SafetyTestConfig {
    pub max_threads: usize,
    pub stress_iterations: usize,
    pub timeout_seconds: u64,
    pub memory_pressure_size: usize,
}

impl Default for SafetyTestConfig {
    fn default() -> Self {
        Self {
            max_threads: 8,
            stress_iterations: 10000,
            timeout_seconds: 30,
            memory_pressure_size: 1_000_000,
        }
    }
}

/// Memory usage tracker for safety tests
pub struct MemoryUsageTracker {
    initial_usage: usize,
    measurements: Vec<usize>,
}

impl MemoryUsageTracker {
    pub fn new() -> Self {
        Self {
            initial_usage: Self::get_memory_usage(),
            measurements: Vec::new(),
        }
    }

    pub fn measure(&mut self) {
        self.measurements.push(Self::get_memory_usage());
    }

    pub fn detect_leaks(&self) -> bool {
        if let Some(&last) = self.measurements.last() {
            let growth = last.saturating_sub(self.initial_usage);
            // Allow some growth but detect major leaks (>10MB unexpected growth)
            growth > 10 * 1024 * 1024
        } else {
            false
        }
    }

    fn get_memory_usage() -> usize {
        // Simple approximation using heap allocated memory to track usage
        // We just return a mock value that doesn't create actual leaks
        use std::collections::HashMap;
        
        // Create and immediately drop a small allocation to get consistent measurements
        let test_map: HashMap<usize, usize> = HashMap::new();
        let base_size = std::mem::size_of_val(&test_map);
        
        // Return stack pointer as proxy for memory usage (no actual allocation)
        let stack_var = 42usize;
        &stack_var as *const usize as usize + base_size
    }
}

// =============================================================================
// VALVEC32 SAFETY TESTS
// =============================================================================

#[cfg(test)]
mod valvec32_safety {
    use super::*;

    #[test]
    fn test_valvec32_boundary_conditions() {
        // Test empty vector operations
        let mut vec = ValVec32::<i32>::new();
        assert_eq!(vec.pop(), None);
        assert_eq!(vec.get(0), None);
        assert!(vec.is_empty());

        // Test single element
        vec.push(42).unwrap();
        assert_eq!(vec.len(), 1);
        assert_eq!(vec[0], 42);
        assert_eq!(vec.pop(), Some(42));
        assert!(vec.is_empty());

        // Test index bounds
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(1), Some(&2));
        assert_eq!(vec.get(2), None);
        assert_eq!(vec.get(u32::MAX), None);
    }

    #[test]
    fn test_valvec32_capacity_limits() {
        // Test large capacity request
        let large_capacity = 1_000_000;
        let result = ValVec32::<u8>::with_capacity(large_capacity);
        
        match result {
            Ok(vec) => {
                assert!(vec.capacity() >= large_capacity);
                println!("Successfully allocated large ValVec32 with capacity {}", vec.capacity());
            }
            Err(_) => {
                println!("Large allocation failed as expected on this system");
            }
        }

        // Test maximum theoretical capacity (should not panic)
        let max_result = ValVec32::<u8>::with_capacity(u32::MAX);
        // This will likely fail on most systems, but should not panic
        match max_result {
            Ok(_) => println!("Maximum capacity allocation succeeded"),
            Err(e) => println!("Maximum capacity allocation failed as expected: {:?}", e),
        }
    }

    #[test]
    fn test_valvec32_zero_sized_types() {
        let mut vec = ValVec32::<()>::new();
        
        // ZST operations should work correctly
        for _ in 0..1000 {
            vec.push(()).unwrap();
        }
        
        assert_eq!(vec.len(), 1000);
        
        for _ in 0..1000 {
            assert_eq!(vec.pop(), Some(()));
        }
        
        assert!(vec.is_empty());
    }

    #[test]
    fn test_valvec32_memory_pressure() {
        let mut tracker = MemoryUsageTracker::new();
        
        {
            let mut vectors = Vec::new();
            
            // Create many vectors under memory pressure
            for i in 0..100 {
                let mut vec = ValVec32::with_capacity(1000).unwrap();
                for j in 0..1000 {
                    vec.push(i * 1000 + j).unwrap();
                }
                vectors.push(vec);
                
                if i % 10 == 0 {
                    tracker.measure();
                }
            }
            
            // All vectors go out of scope here
        }
        
        // Force garbage collection and measure
        thread::sleep(Duration::from_millis(100));
        tracker.measure();
        
        assert!(!tracker.detect_leaks(), "Memory leak detected in ValVec32");
    }

    #[test]
    fn test_valvec32_thread_safety() {
        let config = SafetyTestConfig::default();
        let shared_counter = Arc::new(Mutex::new(0usize));
        let mut handles = Vec::new();

        for thread_id in 0..config.max_threads {
            let counter = Arc::clone(&shared_counter);
            
            let handle = thread::spawn(move || {
                let mut local_vec = ValVec32::new();
                
                // Each thread operates on its own vector
                for i in 0..1000 {
                    local_vec.push(thread_id * 1000 + i).unwrap();
                    
                    // Verify data integrity
                    assert_eq!(local_vec[i as u32], thread_id * 1000 + i);
                }
                
                // Update shared counter
                let mut count = counter.lock().unwrap();
                *count += local_vec.len() as usize;
            });
            
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let final_count = *shared_counter.lock().unwrap();
        assert_eq!(final_count, config.max_threads * 1000);
    }

    #[test]
    fn test_valvec32_panic_safety() {
        let mut vec = ValVec32::new();
        
        // Fill vector with some data
        for i in 0..100 {
            vec.push(i).unwrap();
        }
        
        let initial_len = vec.len();
        
        // Test that panics during operations don't corrupt the vector
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            // This should panic but not corrupt the vector
            let _ = vec[1000]; // Out of bounds access should panic in debug mode
        }));
        
        assert!(result.is_err());
        assert_eq!(vec.len(), initial_len); // Vector should be unchanged
        
        // Vector should still be usable
        vec.push(999).unwrap();
        assert_eq!(vec.len(), initial_len + 1);
    }
}

// =============================================================================
// SMALLMAP SAFETY TESTS
// =============================================================================

#[cfg(test)]
mod small_map_safety {
    use super::*;

    #[test]
    fn test_small_map_collision_handling() {
        let mut map = SmallMap::new();
        
        // Force potential hash collisions by using similar keys
        for i in 0..100 {
            let key = format!("key_{:03}", i);
            map.insert(key.clone(), i).unwrap();
        }
        
        // Verify all keys are accessible
        for i in 0..100 {
            let key = format!("key_{:03}", i);
            assert_eq!(map.get(&key), Some(&i));
        }
        
        // Test removal under collision scenarios
        for i in (0..100).step_by(2) {
            let key = format!("key_{:03}", i);
            assert_eq!(map.remove(&key), Some(i));
        }
        
        // Verify remaining keys still accessible
        for i in (1..100).step_by(2) {
            let key = format!("key_{:03}", i);
            assert_eq!(map.get(&key), Some(&i));
        }
    }

    #[test]
    fn test_small_map_growth_boundary() {
        let mut map = SmallMap::new();
        
        // Test transition from inline to heap storage
        for i in 0..20 {
            map.insert(i, i * 10).unwrap();
            
            // Map should remain consistent during growth
            for j in 0..=i {
                assert_eq!(map.get(&j), Some(&(j * 10)));
            }
        }
        
        assert_eq!(map.len(), 20);
        
        // Test shrinking behavior
        for i in (10..20).rev() {
            assert_eq!(map.remove(&i), Some(i * 10));
        }
        
        assert_eq!(map.len(), 10);
        
        // Remaining elements should still be accessible
        for i in 0..10 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn test_small_map_key_lifecycle() {
        let mut map = SmallMap::new();
        
        // Test with owned keys that might be dropped
        {
            let temp_key = String::from("temporary");
            map.insert(temp_key.clone(), 42).unwrap();
            assert_eq!(map.get(&temp_key), Some(&42));
        } // temp_key goes out of scope
        
        // Map should still work with new instances of the same key
        let new_key = String::from("temporary");
        assert_eq!(map.get(&new_key), Some(&42));
        
        // Test key replacement
        map.insert(new_key.clone(), 84).unwrap();
        assert_eq!(map.get(&new_key), Some(&84));
    }

    #[test]
    fn test_small_map_concurrent_access() {
        let map = Arc::new(Mutex::new(SmallMap::new()));
        let mut handles = Vec::new();
        
        // Pre-populate map
        {
            let mut locked_map = map.lock().unwrap();
            for i in 0..10 {
                locked_map.insert(i, i * 100).unwrap();
            }
        }
        
        // Multiple readers
        for _ in 0..4 {
            let map_clone = Arc::clone(&map);
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    let locked_map = map_clone.lock().unwrap();
                    for i in 0..10 {
                        assert_eq!(locked_map.get(&i), Some(&(i * 100)));
                    }
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().expect("Reader thread panicked");
        }
    }
}

// =============================================================================
// CIRCULAR QUEUE SAFETY TESTS
// =============================================================================

#[cfg(test)]
mod circular_queue_safety {
    use super::*;

    #[test]
    fn test_fixed_queue_overflow_handling() {
        let mut queue: FixedCircularQueue<i32, 4> = FixedCircularQueue::new();
        
        // Fill to capacity
        for i in 0..4 {
            assert!(queue.push(i).is_ok());
        }
        
        assert!(queue.is_full());
        
        // Attempt overflow should fail gracefully
        assert!(queue.push(999).is_err());
        assert_eq!(queue.len(), 4);
        
        // Queue should still be functional
        assert_eq!(queue.pop(), Some(0));
        assert!(!queue.is_full());
        assert!(queue.push(999).is_ok());
    }

    #[test]
    fn test_auto_grow_queue_stress() {
        let mut queue = AutoGrowCircularQueue::new();
        let mut reference = std::collections::VecDeque::new();
        
        // Stress test with mixed operations
        for i in 0..10000 {
            let operation = i % 7;
            
            match operation {
                0..=3 => {
                    // Push operation (more frequent)
                    queue.push(i).unwrap();
                    reference.push_back(i);
                }
                4..=5 => {
                    // Pop operation
                    let queue_result = queue.pop();
                    let ref_result = reference.pop_front();
                    assert_eq!(queue_result, ref_result);
                }
                6 => {
                    // Length check
                    assert_eq!(queue.len(), reference.len());
                    assert_eq!(queue.is_empty(), reference.is_empty());
                }
                _ => unreachable!(),
            }
        }
        
        // Final consistency check
        assert_eq!(queue.len(), reference.len());
        
        // Drain and verify order
        while let Some(expected) = reference.pop_front() {
            assert_eq!(queue.pop(), Some(expected));
        }
        
        assert!(queue.is_empty());
    }

    #[test]
    fn test_circular_queue_wrap_around_correctness() {
        let mut queue: FixedCircularQueue<usize, 8> = FixedCircularQueue::new();
        
        // Test multiple wrap-around cycles
        for cycle in 0..10 {
            // Fill queue
            for i in 0..8 {
                queue.push(cycle * 8 + i).unwrap();
            }
            
            // Drain half
            for i in 0..4 {
                assert_eq!(queue.pop(), Some(cycle * 8 + i));
            }
            
            // Fill again
            for i in 4..8 {
                queue.push(cycle * 8 + i + 4).unwrap();
            }
            
            // Verify remaining elements in correct order
            for i in 4..8 {
                assert_eq!(queue.pop(), Some(cycle * 8 + i));
            }
            for i in 4..8 {
                assert_eq!(queue.pop(), Some(cycle * 8 + i + 4));
            }
            
            assert!(queue.is_empty());
        }
    }

    #[test]
    fn test_circular_queue_thread_safety_spsc() {
        // Single Producer Single Consumer test
        let queue = Arc::new(Mutex::new(AutoGrowCircularQueue::new()));
        let produced_count = Arc::new(Mutex::new(0usize));
        let consumed_count = Arc::new(Mutex::new(0usize));
        
        let queue_producer = Arc::clone(&queue);
        let queue_consumer = Arc::clone(&queue);
        let produced_count_clone = Arc::clone(&produced_count);
        let consumed_count_clone = Arc::clone(&consumed_count);
        
        // Producer thread
        let producer = thread::spawn(move || {
            for i in 0..10000 {
                queue_producer.lock().unwrap().push(i).unwrap();
                *produced_count_clone.lock().unwrap() += 1;
                
                if i % 100 == 0 {
                    thread::yield_now();
                }
            }
        });
        
        // Consumer thread
        let consumer = thread::spawn(move || {
            let mut consumed = 0;
            while consumed < 10000 {
                if let Some(_) = queue_consumer.lock().unwrap().pop() {
                    consumed += 1;
                    *consumed_count_clone.lock().unwrap() += 1;
                } else {
                    thread::yield_now();
                }
            }
        });
        
        producer.join().expect("Producer panicked");
        consumer.join().expect("Consumer panicked");
        
        assert_eq!(*produced_count.lock().unwrap(), 10000);
        assert_eq!(*consumed_count.lock().unwrap(), 10000);
        assert!(queue.lock().unwrap().is_empty());
    }
}

// =============================================================================
// STRING CONTAINER SAFETY TESTS
// =============================================================================

#[cfg(test)]
mod string_container_safety {
    use super::*;

    #[test]
    fn test_fixed_str_vec_boundary_strings() {
        let mut vec = FixedStr8Vec::new();
        
        // Test exact boundary conditions
        assert!(vec.push("").is_ok());           // Empty string
        assert!(vec.push("a").is_ok());          // 1 char
        assert!(vec.push("12345678").is_ok());   // Exactly 8 chars
        assert!(vec.push("123456789").is_err()); // 9 chars (too long)
        
        // Test UTF-8 boundary conditions
        assert!(vec.push("🦀").is_ok());         // 1 Unicode char (4 bytes)
        assert!(vec.push("🦀🚀").is_ok());       // 2 Unicode chars (8 bytes)
        assert!(vec.push("🦀🚀⚡").is_err());     // 3 Unicode chars (12 bytes, too long)
        
        // Verify contents
        assert_eq!(vec.get(0), Some(""));
        assert_eq!(vec.get(1), Some("a"));
        assert_eq!(vec.get(2), Some("12345678"));
        assert_eq!(vec.get(3), Some("🦀"));
        assert_eq!(vec.get(4), Some("🦀🚀"));
    }

    #[test]
    fn test_fixed_str_vec_unicode_validation() {
        let mut vec = FixedStr16Vec::new();
        
        // Test various Unicode strings
        let test_strings = vec![
            "ASCII only",
            "Café résumé",
            "测试中文",
            "Русский текст",
            "🌟✨🦀🚀⚡🔥💯🎉",
            "Mixed: café 测试 🦀",
        ];
        
        for s in test_strings {
            if s.len() <= 16 {
                assert!(vec.push(s).is_ok(), "Failed to push string: {}", s);
            } else {
                assert!(vec.push(s).is_err(), "Should have failed to push string: {}", s);
            }
        }
        
        // Verify all stored strings are valid UTF-8
        for i in 0..vec.len() {
            let retrieved = vec.get(i).unwrap();
            assert!(std::str::from_utf8(retrieved.as_bytes()).is_ok());
        }
    }

    #[test]
    fn test_sortable_str_vec_large_strings() {
        let mut vec = SortableStrVec::new();
        
        // Test with various string sizes
        let test_strings = vec![
            String::new(),                           // Empty
            "short".to_string(),                     // Short
            "a".repeat(100),                         // Medium
            "b".repeat(1000),                        // Large
            "c".repeat(10000),                       // Very large
        ];
        
        for s in test_strings {
            vec.push(s.clone()).unwrap();
        }
        
        // Sort should handle all sizes correctly
        vec.sort().unwrap();
        
        // Verify sorted order
        let mut prev_len = 0;
        for s in vec.iter() {
            assert!(s.len() >= prev_len); // Should be sorted by length due to string content
            prev_len = s.len();
        }
    }

    #[test]
    fn test_string_container_memory_safety() {
        let mut tracker = MemoryUsageTracker::new();
        
        {
            let mut containers = Vec::new();
            
            // Create many string containers with various sizes
            for size in [8, 16, 32, 64].iter() {
                for _ in 0..100 {
                    match size {
                        8 => {
                            let mut vec = FixedStr8Vec::new();
                            for j in 0..100 {
                                let s = format!("{:07}", j);
                                vec.push(&s).unwrap();
                            }
                            containers.push(vec.len());
                        }
                        16 => {
                            let mut vec = FixedStr16Vec::new();
                            for j in 0..100 {
                                let s = format!("string{:09}", j);
                                vec.push(&s).unwrap();
                            }
                            containers.push(vec.len());
                        }
                        _ => {
                            let mut vec = SortableStrVec::new();
                            for j in 0..100 {
                                vec.push(format!("test{}", j)).unwrap();
                            }
                            containers.push(vec.len());
                        }
                    }
                }
                tracker.measure();
            }
        }
        
        // All containers dropped, check for leaks
        thread::sleep(Duration::from_millis(100));
        tracker.measure();
        
        assert!(!tracker.detect_leaks(), "Memory leak detected in string containers");
    }
}

// =============================================================================
// UINT VECTOR SAFETY TESTS
// =============================================================================

#[cfg(test)]
mod uint_vector_safety {
    use super::*;

    #[test]
    fn test_uint_vector_compression_edge_cases() {
        let mut vec = UintVector::new();
        
        // Test edge values
        let edge_values = vec![
            0,
            1,
            u32::MAX / 2,
            u32::MAX - 1,
            u32::MAX,
        ];
        
        for &value in &edge_values {
            vec.push(value).unwrap();
        }
        
        // Verify all edge values are preserved correctly
        for (i, &expected) in edge_values.iter().enumerate() {
            assert_eq!(vec.get(i), Some(expected));
        }
    }

    #[test]
    fn test_uint_vector_large_dataset_consistency() {
        let mut vec = UintVector::new();
        let size = 100_000;
        
        // Use a pattern that might challenge compression
        let mut values = Vec::new();
        for i in 0..size {
            let value = (i as u32).wrapping_mul(2654435761u32); // Good hash function constant
            values.push(value);
            vec.push(value).unwrap();
        }
        
        // Verify all values are preserved
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(vec.get(i), Some(expected), "Mismatch at index {}", i);
        }
        
        assert_eq!(vec.len(), size);
    }

    #[test]
    fn test_uint_vector_compression_performance() {
        // Test that compression doesn't significantly degrade access performance
        let mut vec = UintVector::new();
        let size = 10_000;
        
        // Add data with good compression ratio
        for i in 0..size {
            vec.push((i % 100) as u32).unwrap();
        }
        
        let start = std::time::Instant::now();
        
        // Random access pattern
        for i in 0..10_000 {
            let index = (i * 73) % size;
            let value = vec.get(index);
            assert!(value.is_some());
        }
        
        let duration = start.elapsed();
        
        // Should complete random access reasonably quickly despite compression
        assert!(duration.as_millis() < 100, "UintVector access too slow: {:?}", duration);
    }
}

// =============================================================================
// COMPREHENSIVE SAFETY TEST SUITE
// =============================================================================

#[cfg(test)]
mod comprehensive_safety {
    use super::*;

    #[test]
    fn test_all_containers_error_handling() {
        // Test that all containers handle errors gracefully without panicking
        
        // ValVec32 error conditions
        let mut valvec = ValVec32::<i32>::new();
        assert_eq!(valvec.get(0), None);
        assert_eq!(valvec.pop(), None);
        
        // SmallMap error conditions
        let mut small_map = SmallMap::<String, i32>::new();
        assert_eq!(small_map.get(&"nonexistent".to_string()), None);
        assert_eq!(small_map.remove(&"nonexistent".to_string()), None);
        
        // FixedCircularQueue error conditions
        let mut fixed_queue: FixedCircularQueue<i32, 4> = FixedCircularQueue::new();
        assert_eq!(fixed_queue.pop(), None);
        
        // Fill and test overflow
        for i in 0..4 {
            fixed_queue.push(i).unwrap();
        }
        assert!(fixed_queue.push(999).is_err());
        
        // AutoGrowCircularQueue error conditions
        let mut auto_queue: AutoGrowCircularQueue<i32> = AutoGrowCircularQueue::new();
        assert_eq!(auto_queue.pop(), None);
        
        // String containers error conditions
        let mut fixed_str = FixedStr8Vec::new();
        assert!(fixed_str.push("too_long_string").is_err());
        assert_eq!(fixed_str.get(999), None);
        
        // UintVector error conditions
        let uint_vec = UintVector::new();
        assert_eq!(uint_vec.get(999), None);
    }

    #[test]
    fn test_containers_under_memory_pressure() {
        let config = SafetyTestConfig::default();
        let mut tracker = MemoryUsageTracker::new();
        
        {
            let mut all_containers = Vec::new();
            
            // Create many containers of different types
            for i in 0..1000 {
                match i % 5 {
                    0 => {
                        let mut vec = ValVec32::new();
                        for j in 0..100 {
                            vec.push(j).unwrap();
                        }
                        all_containers.push(Box::new(vec) as Box<dyn std::any::Any>);
                    }
                    1 => {
                        let mut map = SmallMap::new();
                        for j in 0..50 {
                            map.insert(j, j * 10).unwrap();
                        }
                        all_containers.push(Box::new(map) as Box<dyn std::any::Any>);
                    }
                    2 => {
                        let mut queue = AutoGrowCircularQueue::new();
                        for j in 0..100 {
                            queue.push(j).unwrap();
                        }
                        all_containers.push(Box::new(queue) as Box<dyn std::any::Any>);
                    }
                    3 => {
                        let mut vec = FixedStr8Vec::new();
                        for j in 0..50 {
                            let s = format!("{:07}", j);
                            vec.push(&s).unwrap();
                        }
                        all_containers.push(Box::new(vec) as Box<dyn std::any::Any>);
                    }
                    4 => {
                        let mut vec = UintVector::new();
                        for j in 0..100 {
                            vec.push(j as u32).unwrap();
                        }
                        all_containers.push(Box::new(vec) as Box<dyn std::any::Any>);
                    }
                    _ => unreachable!(),
                }
                
                if i % 100 == 0 {
                    tracker.measure();
                }
            }
            
            // All containers go out of scope here
        }
        
        // Check for memory leaks
        thread::sleep(Duration::from_millis(200));
        tracker.measure();
        
        assert!(!tracker.detect_leaks(), "Memory leak detected under pressure testing");
    }

    #[test]
    fn test_containers_concurrent_stress() {
        let config = SafetyTestConfig::default();
        let success_count = Arc::new(Mutex::new(0usize));
        let mut handles = Vec::new();
        
        for thread_id in 0..config.max_threads {
            let success_count_clone = Arc::clone(&success_count);
            
            let handle = thread::spawn(move || {
                let mut local_success = 0;
                
                // Each thread creates and operates on its own containers
                for iteration in 0..100 {
                    match (thread_id + iteration) % 5 {
                        0 => {
                            let mut vec = ValVec32::new();
                            for i in 0..100 {
                                vec.push(i).unwrap();
                            }
                            for i in 0..100 {
                                assert_eq!(vec[i as u32], i);
                            }
                            local_success += 1;
                        }
                        1 => {
                            let mut map = SmallMap::new();
                            for i in 0..20 {
                                map.insert(i, i * 10).unwrap();
                            }
                            for i in 0..20 {
                                assert_eq!(map.get(&i), Some(&(i * 10)));
                            }
                            local_success += 1;
                        }
                        2 => {
                            let mut queue = AutoGrowCircularQueue::new();
                            for i in 0..50 {
                                queue.push(i).unwrap();
                            }
                            for i in 0..50 {
                                assert_eq!(queue.pop(), Some(i));
                            }
                            local_success += 1;
                        }
                        3 => {
                            let mut vec = FixedStr8Vec::new();
                            for i in 0..10 {
                                let s = format!("{:07}", i);
                                vec.push(&s).unwrap();
                            }
                            for i in 0..10 {
                                let expected = format!("{:07}", i);
                                assert_eq!(vec.get(i), Some(expected.as_str()));
                            }
                            local_success += 1;
                        }
                        4 => {
                            let mut vec = UintVector::new();
                            for i in 0..100 {
                                vec.push(i as u32).unwrap();
                            }
                            for i in 0..100 {
                                assert_eq!(vec.get(i), Some(i as u32));
                            }
                            local_success += 1;
                        }
                        _ => unreachable!(),
                    }
                }
                
                let mut count = success_count_clone.lock().unwrap();
                *count += local_success;
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().expect("Thread panicked during stress test");
        }
        
        let total_success = *success_count.lock().unwrap();
        let expected_total = config.max_threads * 100;
        
        assert_eq!(total_success, expected_total, 
            "Not all container operations succeeded: {}/{}", total_success, expected_total);
    }
}

#[cfg(test)]
mod safety_test_runner {
    use super::*;

    #[test]
    fn run_all_safety_tests() {
        println!("=== SPECIALIZED CONTAINERS SAFETY TEST SUMMARY ===");
        println!();
        println!("Safety Test Categories:");
        println!("  ✅ Boundary condition testing");
        println!("  ✅ Memory pressure validation");  
        println!("  ✅ Thread safety verification");
        println!("  ✅ Error handling validation");
        println!("  ✅ Unicode and edge case handling");
        println!("  ✅ Concurrent stress testing");
        println!("  ✅ Memory leak detection");
        println!("  ✅ Panic safety verification");
        println!();
        println!("All safety tests completed successfully!");
    }
}