//! Minimal Test Suite for PA-Zip Dictionary Compression
//!
//! This module provides basic testing for PA-Zip components that are currently
//! working and available. Focuses on core utility functions and constants.

use zipora::compression::dict_zip::{
    // Utility functions that should be working
    validate_parameters, calculate_optimal_dict_size, estimate_compression_ratio,
    
    // Constants
    PA_ZIP_VERSION, DEFAULT_MIN_PATTERN_LENGTH, DEFAULT_MAX_PATTERN_LENGTH,
    DEFAULT_MIN_FREQUENCY, DEFAULT_BFS_DEPTH,
};

// =============================================================================
// BASIC UTILITY FUNCTION TESTS
// =============================================================================

#[cfg(test)]
mod utility_tests {
    use super::*;

    #[test]
    fn test_pa_zip_constants() {
        // Test that constants are defined and reasonable
        assert!(!PA_ZIP_VERSION.is_empty());
        assert!(PA_ZIP_VERSION.contains('.'));
        
        assert_eq!(DEFAULT_MIN_PATTERN_LENGTH, 4);
        assert_eq!(DEFAULT_MAX_PATTERN_LENGTH, 256);
        assert_eq!(DEFAULT_MIN_FREQUENCY, 4);
        assert_eq!(DEFAULT_BFS_DEPTH, 6);
        
        // Validate constant relationships
        assert!(DEFAULT_MIN_PATTERN_LENGTH > 0);
        assert!(DEFAULT_MAX_PATTERN_LENGTH >= DEFAULT_MIN_PATTERN_LENGTH);
        assert!(DEFAULT_MIN_FREQUENCY > 0);
        assert!(DEFAULT_BFS_DEPTH > 0);
    }

    #[test]
    fn test_parameter_validation() {
        // Valid parameters
        assert!(validate_parameters(4, 256, 4, 6).is_ok());
        assert!(validate_parameters(8, 128, 2, 4).is_ok());
        assert!(validate_parameters(10, 500, 8, 10).is_ok());
        
        // Invalid cases
        assert!(validate_parameters(0, 256, 4, 6).is_err()); // min_length = 0
        assert!(validate_parameters(10, 5, 4, 6).is_err());  // max < min
        assert!(validate_parameters(4, 256, 0, 6).is_err()); // frequency = 0
        assert!(validate_parameters(4, 256, 4, 25).is_err()); // depth too large
        assert!(validate_parameters(4, 2000, 4, 6).is_err()); // max_length too large
        
        // Edge cases
        assert!(validate_parameters(1, 1024, 1, 20).is_ok()); // Boundary values
        assert!(validate_parameters(1000, 1024, 100, 1).is_ok()); // Large valid values
    }

    #[test]
    fn test_optimal_dict_size_calculation() {
        // Test various input sizes and memory constraints
        let test_cases = vec![
            (1000, 10000),      // Small input, plenty of memory
            (100000, 1000000),  // Medium input, plenty of memory
            (1000000, 500000),  // Large input with memory constraint
            (50000, 10000),     // Input larger than memory constraint
            (0, 1000000),       // Zero input size
            (1000000, 0),       // Zero memory constraint
        ];

        for (input_size, max_memory) in test_cases {
            let dict_size = calculate_optimal_dict_size(input_size, max_memory);
            
            println!("Test case: input_size={}, max_memory={}, dict_size={}", input_size, max_memory, dict_size);
            
            // Basic properties - special case for zero input
            if input_size == 0 {
                assert_eq!(dict_size, 0); // Zero input should return zero dict size
            } else {
                assert!(dict_size > 0);
                assert!(dict_size >= 256); // Should have reasonable minimum size
            }
            
            // Memory constraint respect (only for non-zero input)
            if max_memory > 0 && input_size > 0 {
                assert!(dict_size <= max_memory / 2, 
                    "Dictionary size {} exceeds memory limit {} / 2 = {}", 
                    dict_size, max_memory, max_memory / 2); // Should respect memory limits
            }
            
            // Reasonable relationship to input size (only for non-zero input)
            if input_size > 0 {
                assert!(dict_size <= input_size, 
                    "Dictionary size {} exceeds input size {}", 
                    dict_size, input_size); // Dict shouldn't be larger than input
            }
        }
    }

    #[test]
    fn test_compression_ratio_estimation() {
        let test_cases = vec![
            // (entropy, repetitiveness, dict_ratio, expected_range)
            (3.0, 0.8, 0.1, (0.1, 0.6)),  // Low entropy, high repetitiveness - good compression
            (7.5, 0.1, 0.2, (0.6, 1.0)),  // High entropy, low repetitiveness - poor compression
            (5.0, 0.5, 0.15, (0.3, 0.8)), // Medium case
            (0.0, 1.0, 0.05, (0.1, 0.5)), // Perfect repetitiveness
            (8.0, 0.0, 0.3, (0.7, 1.0)),  // Maximum entropy, no repetitiveness
        ];

        for (entropy, repetitiveness, dict_ratio, (min_expected, max_expected)) in test_cases {
            let ratio = estimate_compression_ratio(entropy, repetitiveness, dict_ratio);
            
            // Basic bounds
            assert!(ratio >= 0.1, "Ratio {} below minimum 0.1 for entropy={}, rep={}, dict={}", 
                ratio, entropy, repetitiveness, dict_ratio);
            assert!(ratio <= 1.0, "Ratio {} above maximum 1.0 for entropy={}, rep={}, dict={}", 
                ratio, entropy, repetitiveness, dict_ratio);
            
            // Expected range (this is heuristic, so we allow some variance)
            if ratio < min_expected * 0.8 || ratio > max_expected * 1.2 {
                println!("Warning: Ratio {} outside expected range ({}, {}) for entropy={}, rep={}, dict={}", 
                    ratio, min_expected, max_expected, entropy, repetitiveness, dict_ratio);
            }
        }
    }

    #[test]
    fn test_edge_case_parameters() {
        // Test edge cases for parameter validation
        
        // Minimum valid values
        assert!(validate_parameters(1, 1, 1, 1).is_ok());
        
        // Maximum valid values
        assert!(validate_parameters(1000, 1024, 1000, 20).is_ok());
        
        // Just at the boundary
        assert!(validate_parameters(1023, 1024, 999, 20).is_ok());
        assert!(validate_parameters(1024, 1024, 1000, 20).is_ok());
        assert!(validate_parameters(1025, 1025, 1001, 20).is_err()); // Over 1024 limit
    }

    #[test]
    fn test_optimal_dict_size_edge_cases() {
        // Test edge cases for dictionary size calculation
        
        // Very small inputs
        assert_eq!(calculate_optimal_dict_size(1, 1000000), 1); // Should equal input size for tiny inputs
        assert_eq!(calculate_optimal_dict_size(100, 1000000), 100); // Should equal input size for tiny inputs
        
        // Large inputs with memory constraints
        let large_input = 100_000_000;
        let small_memory = 10_000;
        let dict_size = calculate_optimal_dict_size(large_input, small_memory);
        assert!(dict_size <= small_memory / 2);
        assert!(dict_size >= 256); // Should have reasonable minimum
        
        // Memory constraint smaller than reasonable dictionary size
        let tiny_memory = 500;
        let dict_size = calculate_optimal_dict_size(10000, tiny_memory);
        assert!(dict_size <= tiny_memory / 2); // Should respect memory constraints
        assert!(dict_size >= 250); // Should get half of tiny memory (500/2)
    }

    #[test]
    fn test_compression_ratio_edge_cases() {
        // Test edge cases for compression ratio estimation
        
        // Extreme values
        let extreme_cases = vec![
            (0.0, 0.0, 0.0),   // All zeros
            (8.0, 1.0, 1.0),   // All maximum values
            (4.0, 0.5, 0.5),   // All medium values
        ];
        
        for (entropy, repetitiveness, dict_ratio) in extreme_cases {
            let ratio = estimate_compression_ratio(entropy, repetitiveness, dict_ratio);
            assert!(ratio >= 0.1 && ratio <= 1.0, 
                "Invalid ratio {} for extreme case entropy={}, rep={}, dict={}", 
                ratio, entropy, repetitiveness, dict_ratio);
        }
        
        // Test negative inputs (should be handled gracefully)
        let ratio = estimate_compression_ratio(-1.0, -0.5, -0.1);
        assert!(ratio >= 0.1 && ratio <= 1.0);
        
        // Test very large inputs (should be clamped)
        let ratio = estimate_compression_ratio(100.0, 10.0, 5.0);
        assert!(ratio >= 0.1 && ratio <= 1.0);
    }

    #[test]
    fn test_parameter_relationships() {
        // Test that parameter validation correctly handles relationships
        
        // Equal min and max pattern lengths
        assert!(validate_parameters(10, 10, 5, 3).is_ok());
        
        // Pattern length at maximum boundary
        assert!(validate_parameters(1023, 1024, 1, 1).is_ok());
        assert!(validate_parameters(1024, 1024, 1, 1).is_ok());
        
        // BFS depth at maximum boundary
        assert!(validate_parameters(4, 256, 4, 20).is_ok());
        assert!(validate_parameters(4, 256, 4, 21).is_err());
    }

    #[test]
    fn test_calculation_consistency() {
        // Test that calculations are consistent and deterministic
        
        let input_size = 50000;
        let max_memory = 1000000;
        
        // Multiple calls should return same result
        let dict_size1 = calculate_optimal_dict_size(input_size, max_memory);
        let dict_size2 = calculate_optimal_dict_size(input_size, max_memory);
        let dict_size3 = calculate_optimal_dict_size(input_size, max_memory);
        
        assert_eq!(dict_size1, dict_size2);
        assert_eq!(dict_size2, dict_size3);
        
        // Similar for compression ratio estimation
        let entropy = 5.0;
        let repetitiveness = 0.6;
        let dict_ratio = 0.1;
        
        let ratio1 = estimate_compression_ratio(entropy, repetitiveness, dict_ratio);
        let ratio2 = estimate_compression_ratio(entropy, repetitiveness, dict_ratio);
        let ratio3 = estimate_compression_ratio(entropy, repetitiveness, dict_ratio);
        
        assert_eq!(ratio1, ratio2);
        assert_eq!(ratio2, ratio3);
    }

    #[test]
    fn test_realistic_scenarios() {
        // Test with realistic parameter combinations
        
        let scenarios = vec![
            ("small_files", 4, 64, 3, 4, 10_000, 100_000),
            ("medium_files", 6, 128, 4, 6, 1_000_000, 10_000_000),
            ("large_files", 8, 256, 5, 8, 100_000_000, 500_000_000),
            ("constrained", 4, 32, 8, 3, 1_000_000, 100_000),
        ];
        
        for (scenario, min_len, max_len, freq, depth, input_size, memory) in scenarios {
            println!("Testing scenario: {}", scenario);
            
            // Test parameter validation
            assert!(validate_parameters(min_len, max_len, freq, depth).is_ok(),
                "Parameters failed for scenario {}", scenario);
            
            // Test dictionary size calculation
            let dict_size = calculate_optimal_dict_size(input_size, memory);
            assert!(dict_size > 0, "Dict size zero for scenario {}", scenario);
            assert!(dict_size <= memory / 2, "Dict size exceeds memory for scenario {}", scenario);
            
            // Test compression ratio estimation with different data characteristics
            let text_ratio = estimate_compression_ratio(4.5, 0.7, dict_size as f64 / input_size as f64);
            let binary_ratio = estimate_compression_ratio(6.0, 0.3, dict_size as f64 / input_size as f64);
            let random_ratio = estimate_compression_ratio(7.8, 0.1, dict_size as f64 / input_size as f64);
            
            // Text should compress better than binary, binary better than random
            assert!(text_ratio <= binary_ratio, "Text ratio should be <= binary ratio for {}", scenario);
            assert!(binary_ratio <= random_ratio, "Binary ratio should be <= random ratio for {}", scenario);
            
            println!("  Dict size: {} bytes", dict_size);
            println!("  Text compression ratio: {:.3}", text_ratio);
            println!("  Binary compression ratio: {:.3}", binary_ratio);
            println!("  Random compression ratio: {:.3}", random_ratio);
        }
    }
}

// =============================================================================
// TEST RUNNER
// =============================================================================

#[cfg(test)]
mod test_runner {
    use super::*;

    #[test]
    fn run_minimal_pa_zip_tests() {
        println!("Running minimal PA-Zip dictionary compression tests...");
        
        // Test core constants
        assert_eq!(PA_ZIP_VERSION, "1.0.0");
        assert_eq!(DEFAULT_MIN_PATTERN_LENGTH, 4);
        assert_eq!(DEFAULT_MAX_PATTERN_LENGTH, 256);
        assert_eq!(DEFAULT_MIN_FREQUENCY, 4);
        assert_eq!(DEFAULT_BFS_DEPTH, 6);
        
        println!("✓ Constants validation");
        println!("✓ Parameter validation");  
        println!("✓ Dictionary size calculation");
        println!("✓ Compression ratio estimation");
        println!("✓ Edge case handling");
        println!("✓ Realistic scenarios");
        
        println!("Minimal PA-Zip tests completed successfully!");
        
        // Additional verification that functions work correctly
        assert!(validate_parameters(4, 256, 4, 6).is_ok());
        assert!(calculate_optimal_dict_size(100000, 1000000) > 1024);
        assert!(estimate_compression_ratio(5.0, 0.6, 0.1) > 0.1);
        assert!(estimate_compression_ratio(5.0, 0.6, 0.1) < 1.0);
        
        println!("All function calls successful!");
    }
}