#![cfg(feature = "ffi")]

//! Comprehensive FFI integration tests
//!
//! These tests verify the complete FFI API surface including:
//! - Library initialization
//! - FastVec operations
//! - Memory pool lifecycle
//! - Blob store operations
//! - Suffix array operations
//! - Sorting algorithms
//! - Error handling integration
//!
//! All tests must pass in both debug and release mode.

use std::ffi::CStr;
use std::ptr;
use zipora::ffi::*;

// ==================== Library Initialization (2 tests) ====================

#[test]
fn test_library_init() {
    unsafe {
        let result = zipora_init();
        assert_eq!(result, CResult::Success);

        // Should be safe to call multiple times
        let result = zipora_init();
        assert_eq!(result, CResult::Success);
    }
}

#[test]
fn test_version_and_simd() {
    unsafe {
        // Test version string
        let version_ptr = zipora_version();
        assert!(!version_ptr.is_null());

        let version_cstr = CStr::from_ptr(version_ptr);
        let version_str = version_cstr.to_str().unwrap();
        assert!(!version_str.is_empty());
        println!("Zipora version: {}", version_str);

        // Test SIMD support detection (should return 0 or 1)
        let has_simd = zipora_has_simd();
        assert!(has_simd == 0 || has_simd == 1);
        println!("SIMD support: {}", has_simd);
    }
}

// ==================== FastVec (3 tests) ====================

#[test]
fn test_ffi_fast_vec_full_lifecycle() {
    unsafe {
        // Create new FastVec
        let vec = fast_vec_new();
        assert!(!vec.is_null());

        // Verify empty state
        assert_eq!(fast_vec_len(vec), 0);

        // Push 5 bytes
        let test_data = [10u8, 20, 30, 40, 50];
        for &byte in &test_data {
            let result = fast_vec_push(vec, byte);
            assert_eq!(result, CResult::Success);
        }

        // Verify length
        assert_eq!(fast_vec_len(vec), 5);

        // Verify data integrity
        let data_ptr = fast_vec_data(vec);
        assert!(!data_ptr.is_null());

        let data_slice = std::slice::from_raw_parts(data_ptr, 5);
        assert_eq!(data_slice, &test_data);

        // Clean up
        fast_vec_free(vec);
    }
}

#[test]
fn test_ffi_fast_vec_empty() {
    unsafe {
        let vec = fast_vec_new();
        assert!(!vec.is_null());

        // Empty vec should have length 0
        assert_eq!(fast_vec_len(vec), 0);

        // Data pointer behavior when empty (may or may not be null)
        let _data_ptr = fast_vec_data(vec);
        // Don't assert on data_ptr - implementation defined

        fast_vec_free(vec);
    }
}

#[test]
fn test_ffi_fast_vec_null_safety() {
    unsafe {
        // Free null pointer should be safe (no-op)
        fast_vec_free(ptr::null_mut());

        // Length of null vec should be 0
        assert_eq!(fast_vec_len(ptr::null()), 0);

        // Push to null vec should return error
        let result = fast_vec_push(ptr::null_mut(), 42);
        assert_eq!(result, CResult::InvalidInput);

        // Data of null vec should be null
        let data_ptr = fast_vec_data(ptr::null());
        assert!(data_ptr.is_null());
    }
}

// ==================== Memory Pool (3 tests) ====================

#[test]
fn test_ffi_memory_pool_lifecycle() {
    unsafe {
        // Create pool with 1024-byte chunks, max 10 chunks
        let pool = memory_pool_new(1024, 10);
        assert!(!pool.is_null());

        // Allocate a chunk
        let chunk = memory_pool_allocate(pool);
        assert!(!chunk.is_null());

        // Deallocate the chunk
        let result = memory_pool_deallocate(pool, chunk);
        assert_eq!(result, CResult::Success);

        // Free the pool
        memory_pool_free(pool);
    }
}

#[test]
fn test_ffi_memory_pool_multiple_allocs() {
    unsafe {
        let pool = memory_pool_new(512, 10);
        assert!(!pool.is_null());

        // Allocate 5 chunks
        let mut chunks = Vec::new();
        for _ in 0..5 {
            let chunk = memory_pool_allocate(pool);
            assert!(!chunk.is_null());
            chunks.push(chunk);
        }

        // Verify all chunks are different
        for i in 0..chunks.len() {
            for j in i+1..chunks.len() {
                assert_ne!(chunks[i], chunks[j]);
            }
        }

        // Deallocate all chunks
        for chunk in chunks {
            let result = memory_pool_deallocate(pool, chunk);
            assert_eq!(result, CResult::Success);
        }

        memory_pool_free(pool);
    }
}

#[test]
fn test_ffi_memory_pool_null_safety() {
    unsafe {
        // Free null pool should be safe
        memory_pool_free(ptr::null_mut());

        // Allocate from null pool should return null
        let chunk = memory_pool_allocate(ptr::null_mut());
        assert!(chunk.is_null());

        // Deallocate with null pool should return error
        let result = memory_pool_deallocate(ptr::null_mut(), 0x1000 as *mut _);
        assert_eq!(result, CResult::InvalidInput);

        // Create valid pool for testing null pointer deallocation
        let pool = memory_pool_new(1024, 10);
        assert!(!pool.is_null());

        // Deallocate null pointer should return error
        let result = memory_pool_deallocate(pool, ptr::null_mut());
        assert_eq!(result, CResult::InvalidInput);

        memory_pool_free(pool);
    }
}

// ==================== Blob Store (3 tests) ====================

#[test]
fn test_ffi_blob_store_put_get() {
    unsafe {
        // Create blob store
        let store = blob_store_new();
        assert!(!store.is_null());

        // Put data into store
        let test_data = b"Hello, Zipora FFI!";
        let mut record_id: u32 = 0;

        let result = blob_store_put(
            store,
            test_data.as_ptr(),
            test_data.len(),
            &mut record_id
        );
        assert_eq!(result, CResult::Success);

        // Get data back from store
        let mut data_ptr: *const u8 = ptr::null();
        let mut size: usize = 0;

        let result = blob_store_get(store, record_id, &mut data_ptr, &mut size);
        assert_eq!(result, CResult::Success);
        assert!(!data_ptr.is_null());
        assert_eq!(size, test_data.len());

        // Verify data integrity
        let retrieved_data = std::slice::from_raw_parts(data_ptr, size);
        assert_eq!(retrieved_data, test_data);

        // Free blob data
        zipora_free_blob_data(data_ptr as *mut u8, size);

        // Free blob store
        blob_store_free(store);
    }
}

#[test]
fn test_ffi_blob_store_get_missing() {
    unsafe {
        let store = blob_store_new();
        assert!(!store.is_null());

        // Try to get a record that doesn't exist (ID 9999)
        let mut data_ptr: *const u8 = ptr::null();
        let mut size: usize = 0;

        let result = blob_store_get(store, 9999, &mut data_ptr, &mut size);
        // Should return error (InvalidInput based on c_api.rs line 372)
        assert_eq!(result, CResult::InvalidInput);

        blob_store_free(store);
    }
}

#[test]
fn test_ffi_blob_store_null_safety() {
    unsafe {
        // Free null store should be safe
        blob_store_free(ptr::null_mut());

        // Put to null store should return error
        let test_data = b"test";
        let mut record_id: u32 = 0;
        let result = blob_store_put(
            ptr::null_mut(),
            test_data.as_ptr(),
            test_data.len(),
            &mut record_id
        );
        assert_eq!(result, CResult::InvalidInput);

        // Get from null store should return error
        let mut data_ptr: *const u8 = ptr::null();
        let mut size: usize = 0;
        let result = blob_store_get(ptr::null(), 0, &mut data_ptr, &mut size);
        assert_eq!(result, CResult::InvalidInput);

        // Test null data pointer
        let store = blob_store_new();
        let result = blob_store_put(
            store,
            ptr::null(),
            10,
            &mut record_id
        );
        assert_eq!(result, CResult::InvalidInput);
        blob_store_free(store);
    }
}

// ==================== Suffix Array (2 tests) ====================

#[test]
fn test_ffi_suffix_array_lifecycle() {
    unsafe {
        let text = b"banana";

        // Create suffix array
        let sa = suffix_array_new(text.as_ptr(), text.len());
        assert!(!sa.is_null());

        // Verify length
        let len = suffix_array_len(sa);
        assert_eq!(len, text.len());

        // Search for pattern "ana"
        let pattern = b"ana";
        let mut start: usize = 0;
        let mut count: usize = 0;

        let result = suffix_array_search(
            sa,
            text.as_ptr(),
            text.len(),
            pattern.as_ptr(),
            pattern.len(),
            &mut start,
            &mut count
        );
        assert_eq!(result, CResult::Success);

        // "ana" appears twice in "banana" (at positions 1 and 3)
        assert_eq!(count, 2);

        // Free suffix array
        suffix_array_free(sa);
    }
}

#[test]
fn test_ffi_suffix_array_search_patterns() {
    unsafe {
        let text = b"the quick brown fox jumps over the lazy dog";
        let sa = suffix_array_new(text.as_ptr(), text.len());
        assert!(!sa.is_null());

        // Search for existing pattern "the"
        let pattern1 = b"the";
        let mut start: usize = 0;
        let mut count: usize = 0;

        let result = suffix_array_search(
            sa,
            text.as_ptr(),
            text.len(),
            pattern1.as_ptr(),
            pattern1.len(),
            &mut start,
            &mut count
        );
        assert_eq!(result, CResult::Success);
        // "the" appears 2 times
        assert_eq!(count, 2);

        // Search for non-existing pattern "cat"
        let pattern2 = b"cat";
        let result = suffix_array_search(
            sa,
            text.as_ptr(),
            text.len(),
            pattern2.as_ptr(),
            pattern2.len(),
            &mut start,
            &mut count
        );
        assert_eq!(result, CResult::Success);
        assert_eq!(count, 0); // "cat" doesn't exist

        // Search for single character "o"
        let pattern3 = b"o";
        let result = suffix_array_search(
            sa,
            text.as_ptr(),
            text.len(),
            pattern3.as_ptr(),
            pattern3.len(),
            &mut start,
            &mut count
        );
        assert_eq!(result, CResult::Success);
        assert!(count >= 4); // "o" appears in "brown", "fox", "over", "dog"

        suffix_array_free(sa);
    }
}

// ==================== Algorithms (1 test) ====================

#[test]
fn test_ffi_radix_sort() {
    unsafe {
        // Create an array of 100 values in reverse order
        let mut data: Vec<u32> = (0..100).rev().collect();

        // Sort using radix sort
        let result = radix_sort_u32(data.as_mut_ptr(), data.len());
        assert_eq!(result, CResult::Success);

        // Verify sorted
        for i in 0..100 {
            assert_eq!(data[i], i as u32);
        }

        // Test with random-ish data
        let mut data2 = vec![
            42, 17, 89, 3, 56, 91, 23, 67, 8, 45,
            71, 34, 95, 12, 58, 84, 29, 63, 6, 39,
            77, 21, 88, 14, 52, 96, 31, 69, 5, 48,
            82, 25, 93, 11, 61, 87, 36, 74, 19, 55,
        ];
        let mut expected = data2.clone();
        expected.sort();

        let result = radix_sort_u32(data2.as_mut_ptr(), data2.len());
        assert_eq!(result, CResult::Success);
        assert_eq!(data2, expected);
    }
}

// ==================== Error Handling Integration (1 test) ====================

#[test]
fn test_ffi_error_message_after_null_op() {
    unsafe {
        // Clear any previous errors by doing a successful operation
        let _init = zipora_init();

        // Trigger an error by pushing to null vec
        let result = fast_vec_push(ptr::null_mut(), 42);
        assert_eq!(result, CResult::InvalidInput);

        // Check that error message was set
        let error_ptr = zipora_last_error();
        assert!(!error_ptr.is_null());

        let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
        assert!(error_msg.contains("null") || error_msg.contains("FastVec"));
        println!("Error message: {}", error_msg);

        // Try another null operation
        let result = memory_pool_deallocate(ptr::null_mut(), 0x1000 as *mut _);
        assert_eq!(result, CResult::InvalidInput);

        let error_ptr = zipora_last_error();
        let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
        assert!(error_msg.contains("null") || error_msg.contains("pool"));
        println!("Error message: {}", error_msg);
    }
}

// ==================== Additional Edge Cases ====================

#[test]
fn test_ffi_blob_store_multiple_records() {
    unsafe {
        let store = blob_store_new();
        assert!(!store.is_null());

        // Put multiple different records
        let data1 = b"First record";
        let data2 = b"Second record with more data";
        let data3 = b"Third";

        let mut id1: u32 = 0;
        let mut id2: u32 = 0;
        let mut id3: u32 = 0;

        assert_eq!(
            blob_store_put(store, data1.as_ptr(), data1.len(), &mut id1),
            CResult::Success
        );
        assert_eq!(
            blob_store_put(store, data2.as_ptr(), data2.len(), &mut id2),
            CResult::Success
        );
        assert_eq!(
            blob_store_put(store, data3.as_ptr(), data3.len(), &mut id3),
            CResult::Success
        );

        // Verify all records can be retrieved
        let mut ptr1: *const u8 = ptr::null();
        let mut size1: usize = 0;
        assert_eq!(
            blob_store_get(store, id1, &mut ptr1, &mut size1),
            CResult::Success
        );
        assert_eq!(std::slice::from_raw_parts(ptr1, size1), data1);

        let mut ptr2: *const u8 = ptr::null();
        let mut size2: usize = 0;
        assert_eq!(
            blob_store_get(store, id2, &mut ptr2, &mut size2),
            CResult::Success
        );
        assert_eq!(std::slice::from_raw_parts(ptr2, size2), data2);

        let mut ptr3: *const u8 = ptr::null();
        let mut size3: usize = 0;
        assert_eq!(
            blob_store_get(store, id3, &mut ptr3, &mut size3),
            CResult::Success
        );
        assert_eq!(std::slice::from_raw_parts(ptr3, size3), data3);

        // Clean up
        zipora_free_blob_data(ptr1 as *mut u8, size1);
        zipora_free_blob_data(ptr2 as *mut u8, size2);
        zipora_free_blob_data(ptr3 as *mut u8, size3);
        blob_store_free(store);
    }
}

#[test]
fn test_ffi_suffix_array_null_safety() {
    unsafe {
        // Create with null text should return null
        let sa = suffix_array_new(ptr::null(), 10);
        assert!(sa.is_null());

        // Create with zero size should return null
        let text = b"test";
        let sa = suffix_array_new(text.as_ptr(), 0);
        assert!(sa.is_null());

        // Free null suffix array should be safe
        suffix_array_free(ptr::null_mut());

        // Length of null suffix array should be 0
        assert_eq!(suffix_array_len(ptr::null()), 0);

        // Search on null suffix array should return error
        let mut start: usize = 0;
        let mut count: usize = 0;
        let pattern = b"test";
        let result = suffix_array_search(
            ptr::null(),
            text.as_ptr(),
            text.len(),
            pattern.as_ptr(),
            pattern.len(),
            &mut start,
            &mut count
        );
        assert_eq!(result, CResult::InvalidInput);
    }
}

#[test]
fn test_ffi_radix_sort_edge_cases() {
    unsafe {
        // Sort empty array should return error
        let mut empty: Vec<u32> = vec![];
        let result = radix_sort_u32(empty.as_mut_ptr(), 0);
        assert_eq!(result, CResult::InvalidInput);

        // Sort null pointer should return error
        let result = radix_sort_u32(ptr::null_mut(), 10);
        assert_eq!(result, CResult::InvalidInput);

        // Sort single element
        let mut single = vec![42u32];
        let result = radix_sort_u32(single.as_mut_ptr(), 1);
        assert_eq!(result, CResult::Success);
        assert_eq!(single[0], 42);

        // Sort already sorted array
        let mut sorted = vec![1u32, 2, 3, 4, 5];
        let result = radix_sort_u32(sorted.as_mut_ptr(), 5);
        assert_eq!(result, CResult::Success);
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);

        // Sort all same values
        let mut same = vec![7u32; 20];
        let result = radix_sort_u32(same.as_mut_ptr(), 20);
        assert_eq!(result, CResult::Success);
        assert_eq!(same, vec![7u32; 20]);
    }
}
