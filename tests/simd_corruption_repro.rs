//! Minimal reproduction test for SIMD memory corruption issue
//!
//! This test isolates the SIMD streaming store corruption that causes SIGSEGV

use zipora::io::simd_memory::copy::{copy_large_simd, copy_small_simd};

#[test]
fn test_simd_streaming_store_boundary() {
    // Test edge case: buffer size that triggers tail handling
    let sizes = vec![1040, 1056, 1072]; // Just above 1024 threshold

    for size in sizes {
        println!("Testing size: {}", size);

        // Allocate slightly larger to detect overrun
        let mut src_guard = vec![0xDEu8; size + 64];
        let mut dst_guard = vec![0xBEu8; size + 64];

        // Fill source with test pattern
        for i in 0..size {
            src_guard[i] = (i % 256) as u8;
        }

        // Get slices of exact size
        let src = &src_guard[0..size];

        // Perform the copy operation
        {
            let mut dst = &mut dst_guard[0..size];
            let result = copy_large_simd(&mut dst, &src);
            assert!(result.is_ok(), "Copy failed for size {}", size);
        }

        // Verify no overrun - guard bytes should be unchanged
        for i in size..size+64 {
            assert_eq!(dst_guard[i], 0xBE,
                "Buffer overrun detected at offset {} for size {}", i, size);
        }

        // Verify correct copy
        for i in 0..size {
            assert_eq!(dst_guard[i], src[i], "Mismatch at offset {} for size {}", i, size);
        }
    }
}

#[test]
fn test_simd_overlapping_tail_bounds() {
    // Test the specific tail handling that may cause corruption
    // When len > 32 but not aligned to 32
    let size = 47; // Will trigger overlapping tail logic

    let src = vec![42u8; size];
    let mut dst = vec![0u8; size];

    // This triggers the problematic path:
    // - Process 32-byte chunk
    // - Remaining 15 bytes handled with overlapping load at offset 31
    let result = copy_large_simd(&mut dst, &src);
    assert!(result.is_ok());
    assert_eq!(src, dst);

    // Now test with exact boundary conditions
    let boundary_sizes = vec![33, 48, 63, 65, 129];
    for size in boundary_sizes {
        let src = vec![(size % 256) as u8; size];
        let mut dst = vec![0u8; size];

        let result = if size <= 256 {
            copy_small_simd(&mut dst, &src)
        } else {
            copy_large_simd(&mut dst, &src)
        };

        assert!(result.is_ok(), "Failed at size {}", size);
        assert_eq!(src, dst, "Mismatch at size {}", size);
    }
}

#[test]
fn test_multiple_simd_operations_cleanup() {
    // Simulate the test suite scenario - multiple SIMD ops followed by cleanup
    for iteration in 0..10 {
        println!("Iteration {}", iteration);

        // Allocate and use multiple buffers
        let sizes = vec![1024, 2048, 4096];
        let mut allocations = Vec::new();

        for size in sizes {
            let src = vec![0x42u8; size];
            let mut dst = vec![0u8; size];

            copy_large_simd(&mut dst, &src).expect("Copy failed");

            // Keep allocations alive to stress memory
            allocations.push((src, dst));
        }

        // Cleanup phase - this is where SIGSEGV occurs
        drop(allocations);
    }
}

fn main() {
    println!("Running SIMD corruption reproduction tests...");
    test_simd_streaming_store_boundary();
    test_simd_overlapping_tail_bounds();
    test_multiple_simd_operations_cleanup();
    println!("All tests passed!");
}