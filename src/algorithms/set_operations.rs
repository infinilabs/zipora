//! Advanced Set Operations for Multi-way Algorithms
//!
//! This module provides sophisticated set operations optimized for multi-way merge
//! scenarios, with specialized implementations for different numbers of input streams:
//!
//! - **Bit mask optimization**: For ≤32 ways using efficient bit manipulation
//! - **General algorithms**: For larger numbers of ways
//! - **Memory-efficient operations**: Streaming intersection and union
//! - **Hardware acceleration**: SIMD optimizations where applicable

use crate::error::{Result, ZiporaError};
use crate::algorithms::tournament_tree::EnhancedLoserTree;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Configuration for set operations
#[derive(Debug, Clone)]
pub struct SetOperationsConfig {
    /// Use bit mask optimization for small numbers of ways
    pub use_bit_mask_optimization: bool,
    /// Threshold for switching to bit mask optimization
    pub bit_mask_threshold: usize,
    /// Enable frequency counting during operations
    pub count_frequencies: bool,
    /// Use SIMD acceleration when available
    pub use_simd: bool,
}

impl Default for SetOperationsConfig {
    fn default() -> Self {
        Self {
            use_bit_mask_optimization: true,
            bit_mask_threshold: 32,
            count_frequencies: false,
            use_simd: cfg!(feature = "simd"),
        }
    }
}

/// Statistics from set operations
#[derive(Debug, Clone)]
pub struct SetOperationStats {
    /// Number of input ways processed
    pub ways_processed: usize,
    /// Total input elements examined
    pub elements_examined: usize,
    /// Number of output elements produced
    pub output_elements: usize,
    /// Whether bit mask optimization was used
    pub used_bit_mask: bool,
    /// Processing time in microseconds
    pub processing_time_us: u64,
}

/// Advanced set operations for multi-way algorithms
pub struct SetOperations {
    config: SetOperationsConfig,
    stats: SetOperationStats,
}

impl SetOperations {
    /// Create a new set operations instance
    pub fn new() -> Self {
        Self::with_config(SetOperationsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SetOperationsConfig) -> Self {
        Self {
            config,
            stats: SetOperationStats {
                ways_processed: 0,
                elements_examined: 0,
                output_elements: 0,
                used_bit_mask: false,
                processing_time_us: 0,
            },
        }
    }

    /// Compute intersection of multiple sorted sequences
    ///
    /// Returns elements that appear in ALL input sequences.
    /// For ≤32 ways, uses bit mask optimization for O(1) membership testing.
    pub fn intersection<T, I>(&mut self, iterators: Vec<I>) -> Result<Vec<T>>
    where
        T: Ord + Clone + std::fmt::Debug,
        I: Iterator<Item = T> + 'static,
    {
        if iterators.is_empty() {
            return Ok(Vec::new());
        }

        let num_ways = iterators.len();
        self.stats.ways_processed = num_ways;

        let start = std::time::Instant::now();

        let result = if self.config.use_bit_mask_optimization && num_ways <= self.config.bit_mask_threshold {
            self.stats.used_bit_mask = true;
            self.intersection_bit_mask(iterators)?
        } else {
            self.stats.used_bit_mask = false;
            self.intersection_general(iterators)?
        };

        self.stats.processing_time_us = start.elapsed().as_micros() as u64;
        self.stats.output_elements = result.len();

        Ok(result)
    }

    /// Intersection using bit mask optimization for ≤32 ways
    fn intersection_bit_mask<T, I>(&mut self, mut iterators: Vec<I>) -> Result<Vec<T>>
    where
        T: Ord + Clone + std::fmt::Debug,
        I: Iterator<Item = T> + 'static,
    {
        let num_ways = iterators.len();
        let full_mask = if num_ways >= 32 {
            0xFFFFFFFF
        } else {
            (1u32 << num_ways) - 1
        };

        let mut result = Vec::new();
        let mut current_values: Vec<Option<T>> = iterators.iter_mut().map(|it| it.next()).collect();
        
        loop {
            // Find the minimum value among all current values
            let mut min_value: Option<&T> = None;
            let mut min_indices = Vec::new();

            for (idx, value) in current_values.iter().enumerate() {
                if let Some(val) = value {
                    match min_value {
                        None => {
                            min_value = Some(val);
                            min_indices = vec![idx];
                        }
                        Some(min_val) => match val.cmp(min_val) {
                            Ordering::Less => {
                                min_value = Some(val);
                                min_indices = vec![idx];
                            }
                            Ordering::Equal => {
                                min_indices.push(idx);
                            }
                            Ordering::Greater => {}
                        }
                    }
                }
            }

            // If no minimum found, we're done
            let min_val = match min_value {
                Some(val) => val.clone(),
                None => break,
            };

            // Create bit mask for ways that have this minimum value
            let mut current_mask = 0u32;
            for &idx in &min_indices {
                current_mask |= 1u32 << idx;
            }

            // If all ways have this value, add to intersection
            if current_mask == full_mask {
                result.push(min_val.clone());
            }

            // Advance iterators that had the minimum value
            for &idx in &min_indices {
                current_values[idx] = iterators[idx].next();
            }

            self.stats.elements_examined += min_indices.len();
        }

        Ok(result)
    }

    /// General intersection algorithm for larger numbers of ways
    fn intersection_general<T, I>(&mut self, iterators: Vec<I>) -> Result<Vec<T>>
    where
        T: Ord + Clone + std::fmt::Debug,
        I: Iterator<Item = T> + 'static,
    {
        let num_ways = iterators.len();
        let mut result = Vec::new();
        
        // Convert iterators to way iterators for the tournament tree
        let mut tree = EnhancedLoserTree::new(crate::algorithms::LoserTreeConfig::default());
        
        for iterator in iterators {
            tree.add_way(iterator)?;
        }
        
        tree.initialize()?;

        // Process elements using the tournament tree
        let mut current_key: Option<T> = None;
        let mut count = 0;

        while !tree.is_empty() {
            if let Some(value) = tree.pop()? {
                match &current_key {
                    None => {
                        current_key = Some(value.clone());
                        count = 1;
                    }
                    Some(key) => {
                        match value.cmp(key) {
                            Ordering::Equal => {
                                count += 1;
                            }
                            Ordering::Greater => {
                                // Check if previous key appeared in all ways
                                if count == num_ways {
                                    result.push(key.clone());
                                }
                                current_key = Some(value.clone());
                                count = 1;
                            }
                            Ordering::Less => {
                                return Err(ZiporaError::invalid_data("Input sequences not properly sorted"));
                            }
                        }
                    }
                }
                self.stats.elements_examined += 1;
            }
        }

        // Check the last key
        if let Some(key) = current_key {
            if count == num_ways {
                result.push(key);
            }
        }

        Ok(result)
    }

    /// Compute union of multiple sorted sequences
    ///
    /// Returns all unique elements from input sequences in sorted order.
    pub fn union<T, I>(&mut self, iterators: Vec<I>) -> Result<Vec<T>>
    where
        T: Ord + Clone + std::fmt::Debug,
        I: Iterator<Item = T> + 'static,
    {
        if iterators.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();
        self.stats.ways_processed = iterators.len();

        let mut tree = EnhancedLoserTree::new(crate::algorithms::LoserTreeConfig::default());
        
        for iterator in iterators {
            tree.add_way(iterator)?;
        }
        
        tree.initialize()?;

        let mut result = Vec::new();
        let mut last_value: Option<T> = None;

        while !tree.is_empty() {
            if let Some(value) = tree.pop()? {
                // Only add if different from last value (deduplication)
                let should_add = match &last_value {
                    None => true,
                    Some(last) => value.cmp(last) != Ordering::Equal,
                };

                if should_add {
                    result.push(value.clone());
                    last_value = Some(value);
                }

                self.stats.elements_examined += 1;
            }
        }

        self.stats.processing_time_us = start.elapsed().as_micros() as u64;
        self.stats.output_elements = result.len();

        Ok(result)
    }

    /// Count frequency of elements across multiple sorted sequences
    pub fn count_frequencies<T, I>(&mut self, iterators: Vec<I>) -> Result<HashMap<T, usize>>
    where
        T: Ord + Clone + std::hash::Hash + std::fmt::Debug,
        I: Iterator<Item = T> + 'static,
    {
        let start = std::time::Instant::now();
        self.stats.ways_processed = iterators.len();

        let mut tree = EnhancedLoserTree::new(crate::algorithms::LoserTreeConfig::default());
        
        for iterator in iterators {
            tree.add_way(iterator)?;
        }
        
        tree.initialize()?;

        let mut frequencies = HashMap::new();

        while !tree.is_empty() {
            if let Some(value) = tree.pop()? {
                *frequencies.entry(value).or_insert(0) += 1;
                self.stats.elements_examined += 1;
            }
        }

        self.stats.processing_time_us = start.elapsed().as_micros() as u64;
        self.stats.output_elements = frequencies.len();

        Ok(frequencies)
    }

    /// Filter elements using a custom predicate during merge
    pub fn filter_merge<T, I, P>(&mut self, iterators: Vec<I>, predicate: P) -> Result<Vec<T>>
    where
        T: Ord + Clone + std::fmt::Debug,
        I: Iterator<Item = T> + 'static,
        P: Fn(&T) -> bool,
    {
        let start = std::time::Instant::now();
        self.stats.ways_processed = iterators.len();

        let mut tree = EnhancedLoserTree::new(crate::algorithms::LoserTreeConfig::default());
        
        for iterator in iterators {
            tree.add_way(iterator)?;
        }
        
        tree.initialize()?;

        let mut result = Vec::new();

        while !tree.is_empty() {
            if let Some(value) = tree.pop()? {
                if predicate(&value) {
                    result.push(value);
                }
                self.stats.elements_examined += 1;
            }
        }

        self.stats.processing_time_us = start.elapsed().as_micros() as u64;
        self.stats.output_elements = result.len();

        Ok(result)
    }

    /// Get performance statistics from the last operation
    pub fn stats(&self) -> &SetOperationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SetOperationStats {
            ways_processed: 0,
            elements_examined: 0,
            output_elements: 0,
            used_bit_mask: false,
            processing_time_us: 0,
        };
    }
}

impl Default for SetOperations {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersection_bit_mask() {
        let mut ops = SetOperations::new();
        
        let sequences = vec![
            vec![1, 3, 5, 7, 9].into_iter(),
            vec![1, 2, 3, 8, 9].into_iter(),
            vec![1, 3, 4, 6, 9].into_iter(),
        ];

        let result = ops.intersection(sequences).unwrap();
        assert_eq!(result, vec![1, 3, 9]);
        
        let stats = ops.stats();
        assert_eq!(stats.ways_processed, 3);
        assert!(stats.used_bit_mask);
    }

    #[test]
    fn test_intersection_empty() {
        let mut ops = SetOperations::new();
        
        let sequences = vec![
            vec![1, 3, 5].into_iter(),
            vec![2, 4, 6].into_iter(),
        ];

        let result = ops.intersection(sequences).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_union_operation() {
        let mut ops = SetOperations::new();
        
        let sequences = vec![
            vec![1, 3, 5].into_iter(),
            vec![2, 4, 6].into_iter(),
            vec![1, 2, 7].into_iter(),
        ];

        let result = ops.union(sequences).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_frequency_counting() {
        let mut ops = SetOperations::new();
        
        let sequences = vec![
            vec![1, 2, 3].into_iter(),
            vec![1, 2, 4].into_iter(),
            vec![1, 3, 5].into_iter(),
        ];

        let frequencies = ops.count_frequencies(sequences).unwrap();
        
        assert_eq!(frequencies[&1], 3);
        assert_eq!(frequencies[&2], 2);
        assert_eq!(frequencies[&3], 2);
        assert_eq!(frequencies[&4], 1);
        assert_eq!(frequencies[&5], 1);
    }

    #[test]
    fn test_filter_merge() {
        let mut ops = SetOperations::new();
        
        let sequences = vec![
            vec![1, 2, 3, 4, 5].into_iter(),
            vec![2, 4, 6, 8].into_iter(),
        ];

        // Filter only even numbers
        let result = ops.filter_merge(sequences, |x| x % 2 == 0).unwrap();
        assert_eq!(result, vec![2, 2, 4, 4, 6, 8]);
    }

    #[test]
    fn test_large_intersection_general_algorithm() {
        let mut config = SetOperationsConfig::default();
        config.bit_mask_threshold = 2; // Force general algorithm
        
        let mut ops = SetOperations::with_config(config);
        
        let sequences = vec![
            vec![1, 3, 5, 7, 9].into_iter(),
            vec![1, 2, 3, 8, 9].into_iter(),
            vec![1, 3, 4, 6, 9].into_iter(),
        ];

        let result = ops.intersection(sequences).unwrap();
        assert_eq!(result, vec![1, 3, 9]);
        
        let stats = ops.stats();
        assert!(!stats.used_bit_mask);
    }

    #[test]
    fn test_stats_reset() {
        let mut ops = SetOperations::new();
        
        let sequences = vec![
            vec![1, 2, 3].into_iter(),
            vec![1, 2, 4].into_iter(),
        ];

        ops.intersection(sequences).unwrap();
        assert!(ops.stats().elements_examined > 0);
        
        ops.reset_stats();
        assert_eq!(ops.stats().elements_examined, 0);
    }

    #[test]
    fn test_performance_stats() {
        let mut ops = SetOperations::new();
        
        let sequences = vec![
            vec![1, 2, 3, 4, 5].into_iter(),
            vec![1, 3, 5, 7, 9].into_iter(),
        ];

        let result = ops.intersection(sequences).unwrap();
        
        let stats = ops.stats();
        assert!(stats.processing_time_us > 0);
        assert_eq!(stats.output_elements, result.len());
        assert_eq!(stats.ways_processed, 2);
    }
}