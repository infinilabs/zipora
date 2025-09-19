//! Test file for RankSelectInterleaved256 implementation
//!
//! This test suite validates the correctness and basic functionality of the
//! RankSelectInterleaved256 implementation, which is the best-performing
//! rank/select implementation (121-302 Mops/s).

use zipora::{
    BitVector,
    succinct::rank_select::{RankSelectOps, RankSelectInterleaved256},
};

/// Test data generator for different bit patterns
struct TestDataGenerator;

impl TestDataGenerator {
    /// Generate ultra-sparse data (0.1% density)
    fn ultra_sparse(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(i % 1000 == 0).unwrap();
        }
        bv
    }

    /// Generate very sparse data (0.5% density)
    fn very_sparse(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(i % 200 == 0).unwrap();
        }
        bv
    }

    /// Generate clustered sparse data
    fn clustered_sparse(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            // Create clusters: groups of 100 bits with 5 bits set, then 900 bits clear
            let cluster_pos = i % 1000;
            bv.push(cluster_pos < 100 && cluster_pos % 20 == 0).unwrap();
        }
        bv
    }

    /// Generate random sparse data
    fn random_sparse(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            let hash = (i.wrapping_mul(31337).wrapping_add(17)) % 1000;
            bv.push(hash < 5).unwrap(); // 0.5% random density
        }
        bv
    }

    /// Generate alternating block pattern
    fn alternating_blocks(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            let block = i / 64;
            let in_block = i % 64;
            bv.push(block % 8 < 2 && in_block % 8 == 0).unwrap();
        }
        bv
    }

    /// Generate power-of-two positions
    fn power_of_two_positions(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            let is_power_of_two = i > 0 && (i & (i - 1)) == 0;
            bv.push(is_power_of_two).unwrap();
        }
        bv
    }

    /// Generate all test patterns
    fn all_patterns() -> Vec<(&'static str, BitVector)> {
        let size = 50000; // Medium size for comprehensive testing
        vec![
            ("ultra_sparse", Self::ultra_sparse(size)),
            ("very_sparse", Self::very_sparse(size)),
            ("clustered_sparse", Self::clustered_sparse(size)),
            ("random_sparse", Self::random_sparse(size)),
            ("alternating_blocks", Self::alternating_blocks(size)),
            ("power_of_two_positions", Self::power_of_two_positions(size)),
        ]
    }
}

/// Test correctness of RankSelectInterleaved256 implementation
#[test]
fn test_rank_select_interleaved256_correctness() {
    for (pattern_name, bv) in TestDataGenerator::all_patterns() {
        println!("Testing pattern: {} (length: {}, ones: {})",
                 pattern_name, bv.len(), bv.count_ones());

        // Create RankSelectInterleaved256 implementation
        let rank_select = RankSelectInterleaved256::new(bv.clone()).unwrap();

        // Verify basic properties
        assert_eq!(rank_select.len(), bv.len());
        assert_eq!(rank_select.count_ones(), bv.count_ones());
        assert_eq!(rank_select.count_zeros(), bv.count_zeros());

        // Test rank operations
        let test_positions = (0..bv.len()).step_by((bv.len() / 100).max(1)).collect::<Vec<_>>();
        for &pos in &test_positions {
            // Calculate expected rank by counting bits in original BitVector
            let expected_rank1 = (0..pos).map(|i| if bv.get(i).unwrap() { 1 } else { 0 }).sum::<usize>();
            let expected_rank0 = pos - expected_rank1;

            assert_eq!(rank_select.rank1(pos), expected_rank1,
                      "Pattern {}: rank1 mismatch at pos {}", pattern_name, pos);
            assert_eq!(rank_select.rank0(pos), expected_rank0,
                      "Pattern {}: rank0 mismatch at pos {}", pattern_name, pos);
        }

        // Test select operations
        let ones_count = rank_select.count_ones();
        let zeros_count = rank_select.count_zeros();

        if ones_count > 0 {
            let test_ks = (0..ones_count).step_by((ones_count / 20).max(1)).collect::<Vec<_>>();
            for &k in &test_ks {
                if k < ones_count {
                    let selected_pos = rank_select.select1(k).unwrap();
                    // Verify the selected position actually contains a 1
                    assert!(bv.get(selected_pos).unwrap(),
                           "Pattern {}: select1({}) returned position {} which should contain 1", pattern_name, k, selected_pos);
                    // Verify this is the k-th occurrence
                    let rank_at_selected = rank_select.rank1(selected_pos);
                    assert_eq!(rank_at_selected, k,
                              "Pattern {}: select1({}) returned position {} but rank1({}) = {}", pattern_name, k, selected_pos, selected_pos, rank_at_selected);
                }
            }
        }

        if zeros_count > 0 {
            let test_ks = (0..zeros_count).step_by((zeros_count / 20).max(1)).collect::<Vec<_>>();
            for &k in &test_ks {
                if k < zeros_count {
                    let selected_pos = rank_select.select0(k).unwrap();
                    // Verify the selected position actually contains a 0
                    assert!(!bv.get(selected_pos).unwrap(),
                           "Pattern {}: select0({}) returned position {} which should contain 0", pattern_name, k, selected_pos);
                    // Verify this is the k-th occurrence
                    let rank_at_selected = rank_select.rank0(selected_pos);
                    assert_eq!(rank_at_selected, k,
                              "Pattern {}: select0({}) returned position {} but rank0({}) = {}", pattern_name, k, selected_pos, selected_pos, rank_at_selected);
                }
            }
        }

        // Test get operations
        for &pos in &test_positions {
            if pos < bv.len() {
                let expected_bit = bv.get(pos).unwrap();
                assert_eq!(rank_select.get(pos).unwrap(), expected_bit,
                          "Pattern {}: get mismatch at pos {}", pattern_name, pos);
            }
        }
    }
}

/// Test RankSelectInterleaved256 performance characteristics
#[test]
fn test_rank_select_performance_characteristics() {
    let patterns = TestDataGenerator::all_patterns();

    for (pattern_name, bv) in patterns {
        println!("Analyzing performance characteristics for pattern: {}", pattern_name);

        // Create RankSelectInterleaved256 implementation
        let rank_select = RankSelectInterleaved256::new(bv.clone()).unwrap();

        // Measure memory usage
        let original_bytes = (bv.len() + 7) / 8;
        let overhead = rank_select.space_overhead_percent();

        // Verify reasonableness of metrics
        assert!(overhead >= 0.0,
               "Pattern {}: space overhead should be non-negative, got {:.2}%", pattern_name, overhead);

        // Test basic functionality across different access patterns
        let density = bv.count_ones() as f64 / bv.len() as f64;

        // Test a sample of operations to ensure they work
        let test_positions = (0..bv.len()).step_by((bv.len() / 10).max(1)).collect::<Vec<_>>();
        for &pos in &test_positions {
            let _ = rank_select.rank1(pos);
            let _ = rank_select.rank0(pos);
            if let Some(bit) = rank_select.get(pos) {
                assert_eq!(bit, bv.get(pos).unwrap());
            }
        }

        // Test select operations if there are 1s and 0s
        if rank_select.count_ones() > 0 {
            let _ = rank_select.select1(0);
        }
        if rank_select.count_zeros() > 0 {
            let _ = rank_select.select0(0);
        }

        println!("  Original: {} bytes", original_bytes);
        println!("  RankSelectInterleaved256 overhead: {:.2}%", overhead);
        println!("  Pattern density: {:.3}%", density * 100.0);
        println!("  Total bits: {}, ones: {}, zeros: {}", bv.len(), rank_select.count_ones(), rank_select.count_zeros());
    }
}

/// Test error handling and edge cases
#[test]
fn test_error_handling_and_edge_cases() {
    // Test empty bit vector
    let empty_bv = BitVector::new();
    let empty_rank_select = RankSelectInterleaved256::new(empty_bv).unwrap();

    assert_eq!(empty_rank_select.len(), 0);
    assert_eq!(empty_rank_select.count_ones(), 0);
    assert_eq!(empty_rank_select.rank1(0), 0);
    assert!(empty_rank_select.select1(0).is_err());

    // Test single bit
    let mut single_bv = BitVector::new();
    single_bv.push(true).unwrap();
    let single_rank_select = RankSelectInterleaved256::new(single_bv).unwrap();

    assert_eq!(single_rank_select.len(), 1);
    assert_eq!(single_rank_select.count_ones(), 1);
    assert_eq!(single_rank_select.rank1(0), 0);
    assert_eq!(single_rank_select.rank1(1), 1);
    assert_eq!(single_rank_select.select1(0).unwrap(), 0);
    assert!(single_rank_select.select1(1).is_err());

    // Test all zeros
    let all_zeros = BitVector::with_size(1000, false).unwrap();
    let zeros_rank_select = RankSelectInterleaved256::new(all_zeros).unwrap();

    assert_eq!(zeros_rank_select.count_ones(), 0);
    assert_eq!(zeros_rank_select.count_zeros(), 1000);
    assert_eq!(zeros_rank_select.rank1(500), 0);
    assert_eq!(zeros_rank_select.rank0(500), 500);
    assert!(zeros_rank_select.select1(0).is_err());
    assert_eq!(zeros_rank_select.select0(499).unwrap(), 499);

    // Test all ones
    let all_ones = BitVector::with_size(1000, true).unwrap();
    let ones_rank_select = RankSelectInterleaved256::new(all_ones).unwrap();

    assert_eq!(ones_rank_select.count_ones(), 1000);
    assert_eq!(ones_rank_select.count_zeros(), 0);
    assert_eq!(ones_rank_select.rank1(500), 500);
    assert_eq!(ones_rank_select.rank0(500), 0);
    assert_eq!(ones_rank_select.select1(499).unwrap(), 499);
    assert!(ones_rank_select.select0(0).is_err());

    // Test out of bounds access
    let bv = TestDataGenerator::very_sparse(1000);
    let rank_select = RankSelectInterleaved256::new(bv).unwrap();

    assert!(rank_select.get(1000).is_none()); // Out of bounds get
    assert!(rank_select.select1(rank_select.count_ones()).is_err()); // Out of bounds select
}

/// Integration test for RankSelectInterleaved256 comprehensive functionality
#[test]
fn test_full_integration() {
    println!("Running full integration test for RankSelectInterleaved256");

    // Create a complex test pattern
    let mut complex_bv = BitVector::new();
    let size = 20000;

    for i in 0..size {
        // Complex pattern combining multiple characteristics
        let bit = match i % 1000 {
            0..=50 => i % 10 == 0,           // Clustered sparse in first 5%
            51..=100 => true,                // Dense cluster
            101..=950 => false,              // Sparse gap
            951..=999 => (i * 31) % 7 == 0,  // Random sparse pattern
            _ => unreachable!(),
        };
        complex_bv.push(bit).unwrap();
    }

    // Test RankSelectInterleaved256 on this complex pattern
    let rank_select = RankSelectInterleaved256::new(complex_bv.clone()).unwrap();

    // Verify basic properties
    assert_eq!(rank_select.len(), size);
    let ones_count = rank_select.count_ones();
    let zeros_count = rank_select.count_zeros();
    assert_eq!(ones_count + zeros_count, size);

    // Comprehensive correctness testing
    let test_positions = (0..size).step_by(size / 200).collect::<Vec<_>>();

    for &pos in &test_positions {
        // Calculate expected rank from original bit vector
        let expected_rank1 = (0..pos).map(|i| if complex_bv.get(i).unwrap() { 1 } else { 0 }).sum::<usize>();
        let expected_rank0 = pos - expected_rank1;

        assert_eq!(rank_select.rank1(pos), expected_rank1, "rank1 mismatch at {}", pos);
        assert_eq!(rank_select.rank0(pos), expected_rank0, "rank0 mismatch at {}", pos);

        // Test get operation
        if pos < size {
            let expected_bit = complex_bv.get(pos).unwrap();
            assert_eq!(rank_select.get(pos).unwrap(), expected_bit, "get mismatch at {}", pos);
        }
    }

    // Test select operations
    if ones_count > 0 {
        let test_ks = (0..ones_count).step_by((ones_count / 50).max(1)).collect::<Vec<_>>();

        for &k in &test_ks {
            if k < ones_count {
                let selected_pos = rank_select.select1(k).unwrap();
                // Verify the selected position contains a 1
                assert!(complex_bv.get(selected_pos).unwrap(), "select1({}) returned position {} which should contain 1", k, selected_pos);
                // Verify this is the k-th occurrence
                assert_eq!(rank_select.rank1(selected_pos), k, "select1({}) returned position {} but rank at that position is {}", k, selected_pos, rank_select.rank1(selected_pos));
            }
        }
    }

    if zeros_count > 0 {
        let test_ks = (0..zeros_count).step_by((zeros_count / 50).max(1)).collect::<Vec<_>>();

        for &k in &test_ks {
            if k < zeros_count {
                let selected_pos = rank_select.select0(k).unwrap();
                // Verify the selected position contains a 0
                assert!(!complex_bv.get(selected_pos).unwrap(), "select0({}) returned position {} which should contain 0", k, selected_pos);
                // Verify this is the k-th occurrence
                assert_eq!(rank_select.rank0(selected_pos), k, "select0({}) returned position {} but rank at that position is {}", k, selected_pos, rank_select.rank0(selected_pos));
            }
        }
    }

    // Performance analysis
    println!("Integration test results:");
    println!("  Pattern density: {:.3}%", (ones_count as f64 / size as f64) * 100.0);
    println!("  RankSelectInterleaved256 overhead: {:.2}%", rank_select.space_overhead_percent());
    println!("  Total bits: {}, ones: {}, zeros: {}", size, ones_count, zeros_count);

    // Test rank-select consistency
    for _ in 0..100 {
        let pos = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as usize) % size;
        let rank1 = rank_select.rank1(pos);
        let rank0 = rank_select.rank0(pos);
        assert_eq!(rank1 + rank0, pos, "rank1 + rank0 should equal position");

        if rank1 > 0 {
            let select1_result = rank_select.select1(rank1 - 1);
            assert!(select1_result.is_ok(), "select1 should succeed for valid rank");
        }

        if rank0 > 0 {
            let select0_result = rank_select.select0(rank0 - 1);
            assert!(select0_result.is_ok(), "select0 should succeed for valid rank");
        }
    }

    println!("Full integration test completed successfully!");
}