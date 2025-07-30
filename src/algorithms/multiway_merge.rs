//! Multi-way merge algorithms for combining sorted sequences
//!
//! This module provides efficient implementations for merging multiple sorted
//! sequences, which is essential for external sorting and distributed processing.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;
use crate::error::{ToplingError, Result};
use crate::algorithms::{Algorithm, AlgorithmStats};

/// Configuration for multi-way merge operations
#[derive(Debug, Clone)]
pub struct MultiWayMergeConfig {
    /// Use parallel processing for large merges
    pub use_parallel: bool,
    /// Buffer size for each input source
    pub buffer_size: usize,
    /// Maximum number of sources to merge at once
    pub max_merge_ways: usize,
    /// Use tournament tree instead of heap for many sources
    pub use_tournament_tree: bool,
}

impl Default for MultiWayMergeConfig {
    fn default() -> Self {
        Self {
            use_parallel: true,
            buffer_size: 64 * 1024, // 64KB buffers
            max_merge_ways: 1024,
            use_tournament_tree: false,
        }
    }
}

/// A source of sorted data for multi-way merge
pub trait MergeSource<T> {
    /// Get the next item from this source
    fn next(&mut self) -> Option<T>;
    
    /// Peek at the next item without consuming it
    fn peek(&self) -> Option<&T>;
    
    /// Check if this source is exhausted
    fn is_empty(&self) -> bool;
    
    /// Get an estimate of remaining items (for optimization)
    fn remaining_hint(&self) -> Option<usize> {
        None
    }
}

/// A simple vector-based merge source
pub struct VectorSource<T> {
    data: Vec<T>,
    index: usize,
}

impl<T> VectorSource<T> {
    /// Create a new vector source
    pub fn new(data: Vec<T>) -> Self {
        Self { data, index: 0 }
    }
    
    /// Get the remaining items as a slice
    pub fn remaining(&self) -> &[T] {
        &self.data[self.index..]
    }
}

impl<T> MergeSource<T> for VectorSource<T>
where
    T: Clone,
{
    fn next(&mut self) -> Option<T> {
        if self.index < self.data.len() {
            let item = self.data[self.index].clone();
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
    
    fn peek(&self) -> Option<&T> {
        self.data.get(self.index)
    }
    
    fn is_empty(&self) -> bool {
        self.index >= self.data.len()
    }
    
    fn remaining_hint(&self) -> Option<usize> {
        Some(self.data.len() - self.index)
    }
}

/// Entry in the merge heap
#[derive(Debug)]
struct HeapEntry<T> {
    item: T,
    source_id: usize,
}

impl<T> PartialEq for HeapEntry<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.item.eq(&other.item)
    }
}

impl<T> Eq for HeapEntry<T> where T: Eq {}

impl<T> PartialOrd for HeapEntry<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse order for min-heap behavior
        other.item.partial_cmp(&self.item)
    }
}

impl<T> Ord for HeapEntry<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior
        other.item.cmp(&self.item)
    }
}

/// Multi-way merge implementation
pub struct MultiWayMerge {
    config: MultiWayMergeConfig,
    stats: AlgorithmStats,
}

impl MultiWayMerge {
    /// Create a new multi-way merge instance
    pub fn new() -> Self {
        Self::with_config(MultiWayMergeConfig::default())
    }
    
    /// Create a multi-way merge instance with custom configuration
    pub fn with_config(config: MultiWayMergeConfig) -> Self {
        Self {
            config,
            stats: AlgorithmStats {
                items_processed: 0,
                processing_time_us: 0,
                memory_used: 0,
                used_parallel: false,
                used_simd: false,
            },
        }
    }
    
    /// Merge multiple sorted sources into a single sorted output
    pub fn merge<T, S>(&mut self, sources: Vec<S>) -> Result<Vec<T>>
    where
        T: Ord + Clone,
        S: MergeSource<T>,
    {
        let start_time = Instant::now();
        
        if sources.is_empty() {
            return Ok(Vec::new());
        }
        
        if sources.len() == 1 {
            // Single source - just collect all items
            let mut source = sources.into_iter().next().unwrap();
            let mut result = Vec::new();
            while let Some(item) = source.next() {
                result.push(item);
            }
            return Ok(result);
        }
        
        let result = if sources.len() > self.config.max_merge_ways {
            self.merge_hierarchical(sources)?
        } else if self.config.use_tournament_tree && sources.len() > 8 {
            self.merge_tournament(sources)?
        } else {
            self.merge_heap(sources)?
        };
        
        let elapsed = start_time.elapsed();
        self.stats = AlgorithmStats {
            items_processed: result.len(),
            processing_time_us: elapsed.as_micros() as u64,
            memory_used: result.len() * std::mem::size_of::<T>(),
            used_parallel: false, // Simple implementation doesn't use parallelism yet
            used_simd: false,
        };
        
        Ok(result)
    }
    
    /// Merge using a binary heap (good for moderate number of sources)
    fn merge_heap<T, S>(&self, mut sources: Vec<S>) -> Result<Vec<T>>
    where
        T: Ord + Clone,
        S: MergeSource<T>,
    {
        let mut heap = BinaryHeap::new();
        let mut result = Vec::new();
        
        // Initialize heap with first item from each source
        for (id, source) in sources.iter_mut().enumerate() {
            if let Some(item) = source.next() {
                heap.push(HeapEntry {
                    item,
                    source_id: id,
                });
            }
        }
        
        // Merge process
        while let Some(entry) = heap.pop() {
            result.push(entry.item);
            
            // Get next item from the same source
            if let Some(next_item) = sources[entry.source_id].next() {
                heap.push(HeapEntry {
                    item: next_item,
                    source_id: entry.source_id,
                });
            }
        }
        
        Ok(result)
    }
    
    /// Merge using tournament tree (better for many sources)
    fn merge_tournament<T, S>(&self, mut sources: Vec<S>) -> Result<Vec<T>>
    where
        T: Ord + Clone,
        S: MergeSource<T>,
    {
        // Simplified tournament tree - in practice, this would be more optimized
        let mut active_sources: Vec<usize> = (0..sources.len()).collect();
        let mut result = Vec::new();
        
        while !active_sources.is_empty() {
            // Find minimum among active sources
            let mut min_source = 0;
            let mut min_item: Option<&T> = None;
            
            for &source_id in &active_sources {
                if let Some(item) = sources[source_id].peek() {
                    if min_item.is_none() || item < min_item.unwrap() {
                        min_item = Some(item);
                        min_source = source_id;
                    }
                }
            }
            
            if let Some(_) = min_item {
                // Take the minimum item
                if let Some(item) = sources[min_source].next() {
                    result.push(item);
                }
                
                // Remove exhausted sources
                if sources[min_source].is_empty() {
                    active_sources.retain(|&id| id != min_source);
                }
            } else {
                break;
            }
        }
        
        Ok(result)
    }
    
    /// Hierarchical merge for very large number of sources
    fn merge_hierarchical<T, S>(&self, sources: Vec<S>) -> Result<Vec<T>>
    where
        T: Ord + Clone,
        S: MergeSource<T>,
    {
        // For simplicity, fall back to direct heap merge for now
        // A full hierarchical implementation would need dynamic typing
        self.merge_heap(sources)
    }
    
    /// Get performance statistics
    pub fn stats(&self) -> &AlgorithmStats {
        &self.stats
    }
}

impl Default for MultiWayMerge {
    fn default() -> Self {
        Self::new()
    }
}

impl Algorithm for MultiWayMerge {
    type Config = MultiWayMergeConfig;
    type Input = Vec<Vec<i32>>; // Simplified input type for the trait
    type Output = Vec<i32>;
    
    fn execute(&self, config: &Self::Config, input: Self::Input) -> Result<Self::Output> {
        let mut merger = Self::with_config(config.clone());
        let sources: Vec<VectorSource<i32>> = input
            .into_iter()
            .map(VectorSource::new)
            .collect();
        merger.merge(sources)
    }
    
    fn stats(&self) -> AlgorithmStats {
        self.stats.clone()
    }
    
    fn estimate_memory(&self, input_size: usize) -> usize {
        // Estimate based on heap size and output buffer
        let heap_size = self.config.max_merge_ways * std::mem::size_of::<HeapEntry<i32>>();
        let output_size = input_size * std::mem::size_of::<i32>();
        heap_size + output_size
    }
    
    fn supports_parallel(&self) -> bool {
        true // Could be enhanced with parallel processing
    }
}

/// Specialized merge operations
pub struct MergeOperations;

impl MergeOperations {
    /// Merge two sorted vectors
    pub fn merge_two<T>(left: Vec<T>, right: Vec<T>) -> Vec<T>
    where
        T: Ord,
    {
        let mut result = Vec::with_capacity(left.len() + right.len());
        let mut left_iter = left.into_iter();
        let mut right_iter = right.into_iter();
        
        let mut left_current = left_iter.next();
        let mut right_current = right_iter.next();
        
        loop {
            match (&left_current, &right_current) {
                (Some(l), Some(r)) => {
                    if l <= r {
                        result.push(left_current.take().unwrap());
                        left_current = left_iter.next();
                    } else {
                        result.push(right_current.take().unwrap());
                        right_current = right_iter.next();
                    }
                }
                (Some(_), None) => {
                    result.push(left_current.take().unwrap());
                    result.extend(left_iter);
                    break;
                }
                (None, Some(_)) => {
                    result.push(right_current.take().unwrap());
                    result.extend(right_iter);
                    break;
                }
                (None, None) => break,
            }
        }
        
        result
    }
    
    /// Merge sorted slices in-place (for external sorting)
    pub fn merge_in_place<T>(data: &mut [T], mid: usize)
    where
        T: Ord + Clone,
    {
        if mid == 0 || mid >= data.len() {
            return;
        }
        
        let mut temp = Vec::with_capacity(data.len());
        let (left, right) = data.split_at(mid);
        
        let mut left_iter = left.iter();
        let mut right_iter = right.iter();
        
        let mut left_current = left_iter.next();
        let mut right_current = right_iter.next();
        
        loop {
            match (left_current, right_current) {
                (Some(l), Some(r)) => {
                    if l <= r {
                        temp.push(l.clone());
                        left_current = left_iter.next();
                    } else {
                        temp.push(r.clone());
                        right_current = right_iter.next();
                    }
                }
                (Some(l), None) => {
                    temp.push(l.clone());
                    temp.extend(left_iter.cloned());
                    break;
                }
                (None, Some(r)) => {
                    temp.push(r.clone());
                    temp.extend(right_iter.cloned());
                    break;
                }
                (None, None) => break,
            }
        }
        
        // Move elements back using individual assignments to avoid Copy requirement
        for (i, item) in temp.into_iter().enumerate() {
            data[i] = item;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_source() {
        let mut source = VectorSource::new(vec![1, 2, 3, 4, 5]);
        
        assert_eq!(source.peek(), Some(&1));
        assert_eq!(source.next(), Some(1));
        assert_eq!(source.peek(), Some(&2));
        assert!(!source.is_empty());
        assert_eq!(source.remaining_hint(), Some(4));
        
        // Consume all
        source.next();
        source.next();
        source.next();
        source.next();
        
        assert!(source.is_empty());
        assert_eq!(source.next(), None);
    }

    #[test]
    fn test_multiway_merge_empty() {
        let mut merger = MultiWayMerge::new();
        let sources: Vec<VectorSource<i32>> = vec![];
        
        let result = merger.merge(sources).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_multiway_merge_single_source() {
        let mut merger = MultiWayMerge::new();
        let sources = vec![VectorSource::new(vec![1, 3, 5, 7, 9])];
        
        let result = merger.merge(sources).unwrap();
        assert_eq!(result, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn test_multiway_merge_two_sources() {
        let mut merger = MultiWayMerge::new();
        let sources = vec![
            VectorSource::new(vec![1, 3, 5, 7, 9]),
            VectorSource::new(vec![2, 4, 6, 8, 10]),
        ];
        
        let result = merger.merge(sources).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_multiway_merge_multiple_sources() {
        let mut merger = MultiWayMerge::new();
        let sources = vec![
            VectorSource::new(vec![1, 4, 7]),
            VectorSource::new(vec![2, 5, 8]),
            VectorSource::new(vec![3, 6, 9]),
        ];
        
        let result = merger.merge(sources).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_multiway_merge_uneven_sources() {
        let mut merger = MultiWayMerge::new();
        let sources = vec![
            VectorSource::new(vec![1, 2, 3, 4, 5]),
            VectorSource::new(vec![6]),
            VectorSource::new(vec![]),
            VectorSource::new(vec![7, 8]),
        ];
        
        let result = merger.merge(sources).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_multiway_merge_tournament() {
        let config = MultiWayMergeConfig {
            use_tournament_tree: true,
            ..Default::default()
        };
        
        let mut merger = MultiWayMerge::with_config(config);
        let sources = vec![
            VectorSource::new(vec![1, 5, 9]),
            VectorSource::new(vec![2, 6, 10]),
            VectorSource::new(vec![3, 7, 11]),
            VectorSource::new(vec![4, 8, 12]),
        ];
        
        let result = merger.merge(sources).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn test_merge_two_vectors() {
        let left = vec![1, 3, 5, 7];
        let right = vec![2, 4, 6, 8];
        
        let result = MergeOperations::merge_two(left, right);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_merge_two_uneven() {
        let left = vec![1, 3, 5];
        let right = vec![2, 4, 6, 7, 8, 9];
        
        let result = MergeOperations::merge_two(left, right);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_merge_in_place() {
        let mut data = vec![1, 3, 5, 7, 2, 4, 6, 8];
        MergeOperations::merge_in_place(&mut data, 4);
        
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_heap_entry_ordering() {
        let mut heap = BinaryHeap::new();
        
        heap.push(HeapEntry { item: 5, source_id: 0 });
        heap.push(HeapEntry { item: 2, source_id: 1 });
        heap.push(HeapEntry { item: 8, source_id: 2 });
        heap.push(HeapEntry { item: 1, source_id: 3 });
        
        // Should pop in ascending order (min-heap behavior)
        assert_eq!(heap.pop().unwrap().item, 1);
        assert_eq!(heap.pop().unwrap().item, 2);
        assert_eq!(heap.pop().unwrap().item, 5);
        assert_eq!(heap.pop().unwrap().item, 8);
    }

    #[test]
    fn test_algorithm_trait() {
        let merger = MultiWayMerge::new();
        
        assert!(merger.supports_parallel());
        
        let memory_estimate = merger.estimate_memory(1000);
        assert!(memory_estimate > 0);
        
        let input = vec![
            vec![1, 3, 5],
            vec![2, 4, 6],
            vec![7, 8, 9],
        ];
        let config = MultiWayMergeConfig::default();
        let result = merger.execute(&config, input);
        assert!(result.is_ok());
        
        let merged = result.unwrap();
        assert_eq!(merged, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_multiway_merge_performance() {
        let mut merger = MultiWayMerge::new();
        
        // Create many small sorted sequences
        let sources: Vec<VectorSource<i32>> = (0..10)
            .map(|i| {
                let data: Vec<i32> = (i * 10..(i + 1) * 10).collect();
                VectorSource::new(data)
            })
            .collect();
        
        let result = merger.merge(sources).unwrap();
        
        // Should be a sorted sequence of 0..99
        let expected: Vec<i32> = (0..100).collect();
        assert_eq!(result, expected);
        
        let stats = merger.stats();
        assert_eq!(stats.items_processed, 100);
        assert!(stats.processing_time_us > 0);
    }
}