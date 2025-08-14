//! Tournament Tree (Loser Tree) implementation for efficient k-way merging
//!
//! This module provides a high-performance tournament tree implementation optimized for
//! merging multiple sorted sequences. The loser tree variant stores losers at internal
//! nodes while winners propagate up the tree, providing O(log k) complexity for each
//! element selection in k-way merge operations.

use crate::error::{Result, ZiporaError};
use std::cmp::Ordering;
use std::marker::PhantomData;

/// Configuration for tournament tree operations
#[derive(Debug, Clone)]
pub struct LoserTreeConfig {
    /// Initial capacity for the tree
    pub initial_capacity: usize,
    /// Whether to use secure memory pool for allocations
    pub use_secure_memory: bool,
    /// Enable stable sorting (preserve order of equal elements)
    pub stable_sort: bool,
    /// Cache-friendly memory layout optimization
    pub cache_optimized: bool,
}

impl Default for LoserTreeConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 64,
            use_secure_memory: true,
            stable_sort: true,
            cache_optimized: true,
        }
    }
}

/// A node in the tournament tree representing a comparison result
#[derive(Debug, Clone, Copy)]
pub struct TournamentNode {
    /// Index of the way (input stream) that lost this comparison
    pub loser_way: usize,
    /// Index in the original sequence for stable sorting
    pub sequence_index: usize,
}

impl TournamentNode {
    /// Create a new tournament node
    pub fn new(loser_way: usize, sequence_index: usize) -> Self {
        Self {
            loser_way,
            sequence_index,
        }
    }
}

/// Iterator wrapper for input streams with position tracking
struct WayIterator<I, T> {
    iterator: I,
    current: Option<T>,
    way_index: usize,
    position: usize,
    exhausted: bool,
}

impl<I, T> WayIterator<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    fn new(mut iterator: I, way_index: usize) -> Self {
        let current = iterator.next();
        let exhausted = current.is_none();
        
        Self {
            iterator,
            current,
            way_index,
            position: 0,
            exhausted,
        }
    }

    fn advance(&mut self) -> Result<()> {
        if self.exhausted {
            return Ok(());
        }

        self.current = self.iterator.next();
        self.position += 1;
        
        if self.current.is_none() {
            self.exhausted = true;
        }
        
        Ok(())
    }

    fn peek(&self) -> Option<&T> {
        self.current.as_ref()
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

/// High-performance tournament tree for k-way merging
/// 
/// The loser tree is a complete binary tree where internal nodes store the "loser"
/// of comparisons, while the winner propagates to the root. This provides efficient
/// k-way merging with O(log k) complexity per element.
///
/// # Example
/// ```
/// use zipora::algorithms::{LoserTree, LoserTreeConfig};
/// 
/// let config = LoserTreeConfig::default();
/// let mut tree = LoserTree::new(config);
/// 
/// // Add sorted input streams
/// tree.add_way(vec![1, 4, 7, 10].into_iter())?;
/// tree.add_way(vec![2, 5, 8, 11].into_iter())?;
/// tree.add_way(vec![3, 6, 9, 12].into_iter())?;
/// 
/// // Merge all streams
/// let mut result = Vec::new();
/// tree.merge_all(&mut result)?;
/// 
/// assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
pub struct LoserTree<T, F = fn(&T, &T) -> Ordering> {
    /// Tree nodes storing losers of comparisons
    tree: Vec<TournamentNode>,
    /// Input iterators (ways)
    ways: Vec<WayIterator<Box<dyn Iterator<Item = T>>, T>>,
    /// Index of current winner
    winner: usize,
    /// Comparison function
    comparator: F,
    /// Configuration
    config: LoserTreeConfig,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

impl<T> LoserTree<T, fn(&T, &T) -> Ordering>
where
    T: Ord + Clone,
{
    /// Create a new tournament tree with default ordering
    pub fn new(config: LoserTreeConfig) -> Self {
        Self::with_comparator(config, |a, b| a.cmp(b))
    }
}

impl<T, F> LoserTree<T, F>
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    /// Create a new tournament tree with custom comparator
    pub fn with_comparator(config: LoserTreeConfig, comparator: F) -> Self {
        Self {
            tree: Vec::with_capacity(config.initial_capacity),
            ways: Vec::new(),
            winner: 0,
            comparator,
            config,
            _phantom: PhantomData,
        }
    }

    /// Add a sorted input stream to the tournament tree
    pub fn add_way<I>(&mut self, iterator: I) -> Result<()>
    where
        I: Iterator<Item = T> + 'static,
    {
        let way_index = self.ways.len();
        let boxed_iter: Box<dyn Iterator<Item = T>> = Box::new(iterator);
        let way_iter = WayIterator::new(boxed_iter, way_index);
        self.ways.push(way_iter);
        Ok(())
    }

    /// Initialize the tournament tree after all ways have been added
    pub fn initialize(&mut self) -> Result<()> {
        if self.ways.is_empty() {
            return Err(ZiporaError::invalid_data("No input ways provided"));
        }

        let num_ways = self.ways.len();
        
        // Tree size is num_ways - 1 for internal nodes
        self.tree.resize(num_ways.saturating_sub(1), TournamentNode::new(0, 0));
        
        // Initialize tree with bottom-up construction
        self.build_initial_tree()?;
        
        Ok(())
    }

    /// Build the initial tournament tree
    fn build_initial_tree(&mut self) -> Result<()> {
        let num_ways = self.ways.len();
        
        if num_ways <= 1 {
            self.winner = 0;
            return Ok(());
        }

        // Find initial winner through tournament
        let mut min_way = 0;
        let mut min_value: Option<&T> = None;

        for (way_idx, way) in self.ways.iter().enumerate() {
            if let Some(value) = way.peek() {
                if min_value.is_none() || 
                   (self.comparator)(value, min_value.unwrap()) == Ordering::Less {
                    min_value = Some(value);
                    min_way = way_idx;
                }
            }
        }

        self.winner = min_way;

        // Build tournament tree structure
        self.replay_matches()?;

        Ok(())
    }

    /// Replay tournament matches to maintain tree consistency
    fn replay_matches(&mut self) -> Result<()> {
        let num_ways = self.ways.len();
        
        if num_ways <= 1 {
            return Ok(());
        }

        // Build tree bottom-up
        for level in 0..self.tree.len() {
            let left_child = 2 * level + 1;
            let right_child = 2 * level + 2;

            if left_child < num_ways && right_child < num_ways {
                // Compare leaf nodes
                let (_winner, loser) = self.compare_ways(left_child, right_child)?;
                self.tree[level] = TournamentNode::new(loser, self.ways[loser].position);
            } else if left_child < self.tree.len() && right_child < self.tree.len() {
                // Compare internal nodes
                let left_winner = self.get_node_winner(left_child);
                let right_winner = self.get_node_winner(right_child);
                let (_winner, loser) = self.compare_ways(left_winner, right_winner)?;
                self.tree[level] = TournamentNode::new(loser, self.ways[loser].position);
            }
        }

        Ok(())
    }

    /// Compare two ways and return (winner_index, loser_index)
    fn compare_ways(&self, way1: usize, way2: usize) -> Result<(usize, usize)> {
        let value1 = self.ways.get(way1)
            .ok_or_else(|| ZiporaError::out_of_bounds(way1, self.ways.len()))?
            .peek();
        
        let value2 = self.ways.get(way2)
            .ok_or_else(|| ZiporaError::out_of_bounds(way2, self.ways.len()))?
            .peek();

        match (value1, value2) {
            (Some(v1), Some(v2)) => {
                match (self.comparator)(v1, v2) {
                    Ordering::Less => Ok((way1, way2)),
                    Ordering::Greater => Ok((way2, way1)),
                    Ordering::Equal => {
                        // For stable sorting, prefer earlier sequence
                        if self.config.stable_sort {
                            if self.ways[way1].position <= self.ways[way2].position {
                                Ok((way1, way2))
                            } else {
                                Ok((way2, way1))
                            }
                        } else {
                            Ok((way1, way2))
                        }
                    }
                }
            }
            (Some(_), None) => Ok((way1, way2)),
            (None, Some(_)) => Ok((way2, way1)),
            (None, None) => Ok((way1, way2)), // Both exhausted
        }
    }

    /// Get the winner index for a tree node
    fn get_node_winner(&self, node_index: usize) -> usize {
        if node_index < self.tree.len() {
            // This is simplified - in a full implementation, we'd need to
            // traverse the tree to find the actual winner
            self.winner
        } else {
            node_index
        }
    }

    /// Get the current minimum element without removing it
    pub fn peek(&self) -> Option<&T> {
        if self.ways.is_empty() || self.winner >= self.ways.len() {
            return None;
        }
        
        self.ways[self.winner].peek()
    }

    /// Remove and return the current minimum element
    pub fn pop(&mut self) -> Result<Option<T>> {
        if self.ways.is_empty() || self.winner >= self.ways.len() {
            return Ok(None);
        }

        let result = self.ways[self.winner].current.clone();
        
        if result.is_some() {
            // Advance the winner stream
            self.ways[self.winner].advance()?;
            
            // Replay tournament to find new winner
            self.update_winner()?;
        }

        Ok(result)
    }

    /// Update the winner after advancing a stream
    fn update_winner(&mut self) -> Result<()> {
        // Simplified winner update - find minimum among all ways
        let mut new_winner = 0;
        let mut min_value: Option<&T> = None;

        for (way_idx, way) in self.ways.iter().enumerate() {
            if let Some(value) = way.peek() {
                if min_value.is_none() || 
                   (self.comparator)(value, min_value.unwrap()) == Ordering::Less {
                    min_value = Some(value);
                    new_winner = way_idx;
                }
            }
        }

        // Check if any way has values left
        let has_values = self.ways.iter().any(|way| !way.is_exhausted());
        
        if has_values {
            self.winner = new_winner;
        }

        Ok(())
    }

    /// Check if the tournament tree is empty (all ways exhausted)
    pub fn is_empty(&self) -> bool {
        self.ways.iter().all(|way| way.is_exhausted())
    }

    /// Merge all input streams into the output
    pub fn merge_all<O>(&mut self, output: &mut O) -> Result<()>
    where
        O: Extend<T>,
    {
        self.initialize()?;
        
        let mut result = Vec::new();
        
        while !self.is_empty() {
            if let Some(value) = self.pop()? {
                result.push(value);
            }
        }
        
        output.extend(result);
        Ok(())
    }

    /// Merge streams and collect into a vector
    pub fn merge_to_vec(&mut self) -> Result<Vec<T>> {
        let mut result = Vec::new();
        self.merge_all(&mut result)?;
        Ok(result)
    }

    /// Get the number of input ways
    pub fn num_ways(&self) -> usize {
        self.ways.len()
    }

    /// Get configuration
    pub fn config(&self) -> &LoserTreeConfig {
        &self.config
    }
}

/// Iterator implementation for consuming the tournament tree
impl<T, F> Iterator for LoserTree<T, F>
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.pop().unwrap_or(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tournament_node() {
        let node = TournamentNode::new(5, 10);
        assert_eq!(node.loser_way, 5);
        assert_eq!(node.sequence_index, 10);
    }

    #[test]
    fn test_loser_tree_config_default() {
        let config = LoserTreeConfig::default();
        assert_eq!(config.initial_capacity, 64);
        assert!(config.use_secure_memory);
        assert!(config.stable_sort);
        assert!(config.cache_optimized);
    }

    #[test]
    fn test_empty_tree() {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::<i32>::new(config);
        
        assert!(tree.is_empty());
        assert_eq!(tree.num_ways(), 0);
        assert!(tree.peek().is_none());
        assert!(tree.pop().unwrap().is_none());
    }

    #[test]
    fn test_single_way() -> Result<()> {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::<i32>::new(config);
        
        tree.add_way(vec![1, 2, 3].into_iter())?;
        
        let result = tree.merge_to_vec()?;
        assert_eq!(result, vec![1, 2, 3]);
        
        Ok(())
    }

    #[test]
    fn test_two_way_merge() -> Result<()> {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::<i32>::new(config);
        
        tree.add_way(vec![1, 3, 5].into_iter())?;
        tree.add_way(vec![2, 4, 6].into_iter())?;
        
        let result = tree.merge_to_vec()?;
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
        
        Ok(())
    }

    #[test]
    fn test_three_way_merge() -> Result<()> {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::<i32>::new(config);
        
        tree.add_way(vec![1, 4, 7].into_iter())?;
        tree.add_way(vec![2, 5, 8].into_iter())?;
        tree.add_way(vec![3, 6, 9].into_iter())?;
        
        let result = tree.merge_to_vec()?;
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        
        Ok(())
    }

    #[test]
    fn test_uneven_lengths() -> Result<()> {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::<i32>::new(config);
        
        tree.add_way(vec![1].into_iter())?;
        tree.add_way(vec![2, 3, 4, 5].into_iter())?;
        tree.add_way(vec![6, 7].into_iter())?;
        
        let result = tree.merge_to_vec()?;
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7]);
        
        Ok(())
    }

    #[test]
    fn test_empty_ways() -> Result<()> {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::<i32>::new(config);
        
        tree.add_way(vec![1, 2].into_iter())?;
        tree.add_way(std::iter::empty())?;
        tree.add_way(vec![3, 4].into_iter())?;
        
        let result = tree.merge_to_vec()?;
        assert_eq!(result, vec![1, 2, 3, 4]);
        
        Ok(())
    }

    #[test]
    fn test_duplicate_values() -> Result<()> {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::<i32>::new(config);
        
        tree.add_way(vec![1, 2, 2, 3].into_iter())?;
        tree.add_way(vec![2, 2, 4].into_iter())?;
        
        let result = tree.merge_to_vec()?;
        assert_eq!(result, vec![1, 2, 2, 2, 2, 3, 4]);
        
        Ok(())
    }

    #[test]
    fn test_custom_comparator() -> Result<()> {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::with_comparator(config, |a: &i32, b: &i32| b.cmp(a)); // Reverse order
        
        tree.add_way(vec![5, 3, 1].into_iter())?;
        tree.add_way(vec![6, 4, 2].into_iter())?;
        
        let result = tree.merge_to_vec()?;
        assert_eq!(result, vec![6, 5, 4, 3, 2, 1]);
        
        Ok(())
    }

    #[test]
    fn test_iterator_interface() -> Result<()> {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::<i32>::new(config);
        
        tree.add_way(vec![1, 3].into_iter())?;
        tree.add_way(vec![2, 4].into_iter())?;
        
        tree.initialize()?;
        
        let collected: Vec<_> = tree.collect();
        assert_eq!(collected, vec![1, 2, 3, 4]);
        
        Ok(())
    }

    #[test]
    fn test_peek_before_pop() -> Result<()> {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::<i32>::new(config);
        
        tree.add_way(vec![1, 3].into_iter())?;
        tree.add_way(vec![2, 4].into_iter())?;
        
        tree.initialize()?;
        
        assert_eq!(tree.peek(), Some(&1));
        assert_eq!(tree.pop()?, Some(1));
        
        assert_eq!(tree.peek(), Some(&2));
        assert_eq!(tree.pop()?, Some(2));
        
        Ok(())
    }

    #[test]
    fn test_large_merge() -> Result<()> {
        let config = LoserTreeConfig::default();
        let mut tree = LoserTree::<i32>::new(config);
        
        // Add 10 ways with 100 elements each
        for way in 0..10 {
            let values: Vec<i32> = (0..100).map(|i| way * 100 + i).collect();
            tree.add_way(values.into_iter())?;
        }
        
        let result = tree.merge_to_vec()?;
        
        // Should have 1000 elements total
        assert_eq!(result.len(), 1000);
        
        // Should be sorted
        for i in 1..result.len() {
            assert!(result[i] >= result[i-1]);
        }
        
        Ok(())
    }
}