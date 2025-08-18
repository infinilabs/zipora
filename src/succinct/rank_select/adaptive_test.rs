//! Comprehensive tests for adaptive rank/select strategy selection

#[cfg(test)]
mod tests {
    use super::super::adaptive::*;
    use super::super::*;
    use crate::succinct::BitVector;

    fn create_test_bitvector(size: usize, pattern: fn(usize) -> bool) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(pattern(i)).unwrap();
        }
        bv
    }

    #[test]
    fn test_adaptive_sparse_selection() {
        // Very sparse data (0.1% density) - should select RankSelectFew
        let sparse_bv = create_test_bitvector(10000, |i| i % 1000 == 0);
        let adaptive = AdaptiveRankSelect::new(sparse_bv).unwrap();
        
        assert!(adaptive.implementation_name().contains("RankSelectFew"));
        assert!(adaptive.data_profile().density < 0.01);
        assert_eq!(adaptive.data_profile().size_category, SizeCategory::Medium);
        
        // Verify correctness
        assert_eq!(adaptive.count_ones(), 10);
        assert_eq!(adaptive.rank1(5000), 5);
        assert_eq!(adaptive.select1(0).unwrap(), 0);
        assert_eq!(adaptive.select1(4).unwrap(), 4000);
    }

    #[test]
    fn test_adaptive_dense_selection() {
        // Dense data (95% density) - should select dense-optimized implementation
        let dense_bv = create_test_bitvector(1000, |i| i % 20 != 0);
        let adaptive = AdaptiveRankSelect::new(dense_bv).unwrap();
        
        assert!(adaptive.implementation_name().contains("RankSelectSeparated256"));
        assert!(adaptive.data_profile().density > 0.9);
        
        // Verify correctness
        assert_eq!(adaptive.count_ones(), 950);
        assert_eq!(adaptive.rank1(100), 95);
    }

    #[test]
    fn test_adaptive_balanced_selection() {
        // Balanced data (50% density) - should select based on size
        let balanced_bv = create_test_bitvector(50000, |i| i % 2 == 0);
        let adaptive = AdaptiveRankSelect::new(balanced_bv).unwrap();
        
        // Medium dataset should select appropriate implementation
        assert!(adaptive.implementation_name().contains("RankSelect"));
        assert!((adaptive.data_profile().density - 0.5).abs() < 0.1);
        
        // Verify correctness
        assert_eq!(adaptive.count_ones(), 25000);
        assert_eq!(adaptive.rank1(10000), 5000);
        assert_eq!(adaptive.select1(4999).unwrap(), 9998);
    }

    #[test]
    fn test_adaptive_small_dataset() {
        // Small dataset should prefer cache-efficient implementation
        let small_bv = create_test_bitvector(5000, |i| i % 3 == 0);
        let adaptive = AdaptiveRankSelect::new(small_bv).unwrap();
        
        assert!(adaptive.implementation_name().contains("Interleaved256"));
        assert_eq!(adaptive.data_profile().size_category, SizeCategory::Small);
        
        // Verify correctness
        let expected_ones = 5000 / 3 + 1; // 0, 3, 6, 9, ..., 4998
        assert_eq!(adaptive.count_ones(), expected_ones);
    }

    #[test]
    fn test_adaptive_large_dataset() {
        // Large dataset should prefer separated implementation
        let large_bv = create_test_bitvector(2_000_000, |i| i % 4 == 0);
        let adaptive = AdaptiveRankSelect::new(large_bv).unwrap();
        
        assert!(adaptive.implementation_name().contains("Separated"));
        assert_eq!(adaptive.data_profile().size_category, SizeCategory::Large);
        
        // Verify correctness for a few operations
        assert_eq!(adaptive.count_ones(), 500000);
        assert_eq!(adaptive.rank1(1000), 250);
    }

    #[test]
    fn test_custom_selection_criteria() {
        let bv = create_test_bitvector(10000, |i| i % 50 == 0); // 2% density
        
        // Custom criteria with lower sparse threshold
        let criteria = SelectionCriteria {
            sparse_threshold: 0.01, // 1% threshold
            ..Default::default()
        };
        
        let adaptive = AdaptiveRankSelect::with_criteria(bv, criteria).unwrap();
        assert!(adaptive.implementation_name().contains("RankSelectFew"));
    }

    #[test]
    fn test_access_pattern_optimization() {
        let bv = create_test_bitvector(50000, |i| i % 7 == 0);
        
        // Sequential access pattern preference
        let criteria = SelectionCriteria {
            access_pattern: AccessPattern::Sequential,
            ..Default::default()
        };
        
        let adaptive = AdaptiveRankSelect::with_criteria(bv, criteria).unwrap();
        
        // Should prefer implementation suitable for sequential access
        assert!(adaptive.implementation_name().contains("RankSelect"));
        assert_eq!(adaptive.data_profile().access_pattern, AccessPattern::Sequential);
    }

    #[test]
    fn test_edge_cases() {
        // Empty bit vector
        let empty_bv = BitVector::new();
        let adaptive = AdaptiveRankSelect::new(empty_bv).unwrap();
        assert_eq!(adaptive.count_ones(), 0);
        assert_eq!(adaptive.len(), 0);
        
        // All zeros
        let all_zeros = BitVector::with_size(1000, false).unwrap();
        let adaptive = AdaptiveRankSelect::new(all_zeros).unwrap();
        assert!(adaptive.implementation_name().contains("RankSelectSimple"));
        assert_eq!(adaptive.count_ones(), 0);
        assert_eq!(adaptive.rank1(500), 0);
        
        // All ones
        let all_ones = BitVector::with_size(1000, true).unwrap();
        let adaptive = AdaptiveRankSelect::new(all_ones).unwrap();
        assert!(adaptive.implementation_name().contains("RankSelectSimple"));
        assert_eq!(adaptive.count_ones(), 1000);
        assert_eq!(adaptive.rank1(500), 500);
    }

    #[test]
    fn test_optimization_stats() {
        let sparse_bv = create_test_bitvector(10000, |i| i % 200 == 0);
        let adaptive = AdaptiveRankSelect::new(sparse_bv).unwrap();
        
        let stats = adaptive.optimization_stats();
        assert_eq!(stats.total_bits, 10000);
        assert_eq!(stats.ones_count, 50);
        assert!((stats.density - 0.005).abs() < 0.001);
        assert!(stats.implementation.contains("RankSelectFew"));
        assert!(stats.space_overhead_percent >= 0.0);
        
        // Test display functionality
        let display_str = format!("{}", stats);
        assert!(display_str.contains("AdaptiveRankSelect Stats"));
        assert!(display_str.contains("Total bits: 10000"));
        assert!(display_str.contains("RankSelectFew"));
    }

    #[test]
    fn test_multi_dimensional_adaptive() {
        let bv1 = create_test_bitvector(1000, |i| i % 2 == 0);  // 50% density
        let bv2 = create_test_bitvector(1000, |i| i % 5 == 0);  // 20% density
        
        let multi = AdaptiveMultiDimensional::new_dual(bv1, bv2).unwrap();
        
        assert!(multi.implementation_name().contains("RankSelectMixedIL256"));
        assert_eq!(multi.dimensions(), 2);
        assert_eq!(multi.len(), 1000);
        
        // Test basic operations
        let rank = multi.rank1(100);
        assert!(rank > 0);
        
        let pos = multi.select1(10);
        assert!(pos.is_ok());
    }

    #[test]
    fn test_multi_dimensional_length_validation() {
        let bv1 = create_test_bitvector(1000, |i| i % 2 == 0);
        let bv2 = create_test_bitvector(500, |i| i % 2 == 0);  // Different length
        
        let result = AdaptiveMultiDimensional::new_dual(bv1, bv2);
        assert!(result.is_err());
    }

    #[test]
    fn test_performance_tier_classification() {
        // Sparse data -> SpaceOptimized
        let sparse_bv = create_test_bitvector(10000, |i| i % 500 == 0);
        let adaptive = AdaptiveRankSelect::new(sparse_bv).unwrap();
        let stats = adaptive.optimization_stats();
        assert_eq!(stats.estimated_performance_tier, PerformanceTier::SpaceOptimized);
        
        // Small balanced data -> HighPerformance (Interleaved)
        let small_bv = create_test_bitvector(5000, |i| i % 3 == 0);
        let adaptive = AdaptiveRankSelect::new(small_bv).unwrap();
        let stats = adaptive.optimization_stats();
        assert_eq!(stats.estimated_performance_tier, PerformanceTier::HighPerformance);
        
        // Large sequential data -> Sequential (512-bit blocks)
        let large_bv = create_test_bitvector(2_000_000, |i| i % 4 == 0);
        let criteria = SelectionCriteria {
            access_pattern: AccessPattern::Sequential,
            ..Default::default()
        };
        let adaptive = AdaptiveRankSelect::with_criteria(large_bv, criteria).unwrap();
        let stats = adaptive.optimization_stats();
        assert_eq!(stats.estimated_performance_tier, PerformanceTier::Sequential);
    }

    #[test]
    fn test_data_profile_analysis() {
        let bv = create_test_bitvector(50000, |i| i % 100 == 0); // 1% density
        let criteria = SelectionCriteria::default();
        let profile = AdaptiveRankSelect::analyze_data(&bv, &criteria);
        
        assert_eq!(profile.total_bits, 50000);
        assert_eq!(profile.ones_count, 500);
        assert!((profile.density - 0.01).abs() < 0.001);
        assert_eq!(profile.size_category, SizeCategory::Medium);
        assert_eq!(profile.access_pattern, AccessPattern::Mixed);
    }

    #[test]
    fn test_size_category_classification() {
        // Small dataset
        let small_bv = create_test_bitvector(5000, |i| i % 2 == 0);
        let profile = AdaptiveRankSelect::analyze_data(&small_bv, &SelectionCriteria::default());
        assert_eq!(profile.size_category, SizeCategory::Small);
        
        // Medium dataset
        let medium_bv = create_test_bitvector(50000, |i| i % 2 == 0);
        let profile = AdaptiveRankSelect::analyze_data(&medium_bv, &SelectionCriteria::default());
        assert_eq!(profile.size_category, SizeCategory::Medium);
        
        // Large dataset
        let large_bv = create_test_bitvector(5_000_000, |i| i % 2 == 0);
        let profile = AdaptiveRankSelect::analyze_data(&large_bv, &SelectionCriteria::default());
        assert_eq!(profile.size_category, SizeCategory::Large);
    }

    #[test]
    fn test_correctness_vs_reference() {
        // Ensure adaptive selection maintains correctness
        let test_bv = create_test_bitvector(10000, |i| (i * 17 + 7) % 23 == 0);
        
        let adaptive = AdaptiveRankSelect::new(test_bv.clone()).unwrap();
        let reference = RankSelectSimple::new(test_bv).unwrap();
        
        // Test rank operations
        for pos in [0, 1000, 5000, 9999] {
            assert_eq!(adaptive.rank1(pos), reference.rank1(pos), "Rank mismatch at {}", pos);
            assert_eq!(adaptive.rank0(pos), reference.rank0(pos), "Rank0 mismatch at {}", pos);
        }
        
        // Test select operations
        let ones_count = adaptive.count_ones();
        for k in [0, ones_count / 4, ones_count / 2, ones_count - 1] {
            if k < ones_count {
                assert_eq!(
                    adaptive.select1(k).unwrap(),
                    reference.select1(k).unwrap(),
                    "Select mismatch at k={}", k
                );
            }
        }
        
        // Test basic properties
        assert_eq!(adaptive.len(), reference.len());
        assert_eq!(adaptive.count_ones(), reference.count_ones());
        assert_eq!(adaptive.count_zeros(), reference.count_zeros());
    }

    #[test]
    fn test_selection_criteria_thresholds() {
        let bv = create_test_bitvector(10000, |i| i % 25 == 0); // 4% density
        
        // Default criteria (5% threshold) - should NOT trigger sparse
        let default_criteria = SelectionCriteria::default();
        let adaptive1 = AdaptiveRankSelect::with_criteria(bv.clone(), default_criteria).unwrap();
        assert!(!adaptive1.implementation_name().contains("RankSelectFew"));
        
        // Lower threshold (3%) - should trigger sparse
        let strict_criteria = SelectionCriteria {
            sparse_threshold: 0.03,
            ..Default::default()
        };
        let adaptive2 = AdaptiveRankSelect::with_criteria(bv, strict_criteria).unwrap();
        assert!(adaptive2.implementation_name().contains("RankSelectFew"));
    }

    #[test]
    fn test_prefer_space_option() {
        let bv = create_test_bitvector(50000, |i| i % 3 == 0); // ~33% density
        
        // Space-preferring criteria
        let space_criteria = SelectionCriteria {
            prefer_space: true,
            ..Default::default()
        };
        
        let adaptive = AdaptiveRankSelect::with_criteria(bv, space_criteria).unwrap();
        
        // Should still work correctly regardless of space preference
        assert!(adaptive.len() > 0);
        assert!(adaptive.count_ones() > 0);
        
        let stats = adaptive.optimization_stats();
        assert!(stats.space_overhead_percent >= 0.0);
    }

    #[test] 
    fn test_benchmark_adaptive_vs_manual() {
        use std::time::Instant;
        
        let test_sizes = [1000, 10000, 100000];
        let test_patterns = [
            ("sparse", |i: usize| i % 100 == 0),
            ("balanced", |i: usize| i % 2 == 0),
            ("dense", |i: usize| i % 10 != 0),
        ];
        
        for &size in &test_sizes {
            for &(pattern_name, pattern_fn) in &test_patterns {
                let bv = create_test_bitvector(size, pattern_fn);
                
                // Time adaptive selection
                let start = Instant::now();
                let adaptive = AdaptiveRankSelect::new(bv.clone()).unwrap();
                let adaptive_time = start.elapsed();
                
                // Time manual selection (use simple as baseline)
                let start = Instant::now();
                let manual = RankSelectSimple::new(bv).unwrap();
                let manual_time = start.elapsed();
                
                // Verify they produce same results
                for test_pos in [0, size / 4, size / 2, size * 3 / 4].iter().cloned() {
                    if test_pos < size {
                        assert_eq!(adaptive.rank1(test_pos), manual.rank1(test_pos));
                    }
                }
                
                println!(
                    "Size: {}, Pattern: {}, Adaptive: {:?}, Manual: {:?}, Selected: {}",
                    size, pattern_name, adaptive_time, manual_time, adaptive.implementation_name()
                );
            }
        }
    }
}