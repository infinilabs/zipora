//! Simplified FSA infrastructure implementations for initial Phase 8A completion
//!
//! This module provides basic working implementations of the FSA infrastructure
//! components to establish the foundation for Phase 8A.

use crate::error::Result;
use std::collections::HashMap;

/// Simple FSA cache for basic operations
#[derive(Debug)]
pub struct SimpleFsaCache {
    cache: HashMap<u32, u32>,
    max_size: usize,
}

impl SimpleFsaCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }

    pub fn get(&self, key: u32) -> Option<u32> {
        self.cache.get(&key).copied()
    }

    pub fn insert(&mut self, key: u32, value: u32) -> Result<()> {
        if self.cache.len() >= self.max_size {
            // Simple eviction - remove first entry
            if let Some(&first_key) = self.cache.keys().next() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, value);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

/// Simple directed acyclic word graph
#[derive(Debug)]
pub struct SimpleDawg {
    /// States mapped to their terminal status
    states: HashMap<u32, bool>,
    /// Transitions: (from_state, symbol) -> to_state
    transitions: HashMap<(u32, u8), u32>,
    /// Root state
    root: u32,
    /// Next available state ID
    next_state: u32,
    /// Number of keys
    num_keys: usize,
}

impl SimpleDawg {
    pub fn new() -> Self {
        let mut dawg = Self {
            states: HashMap::new(),
            transitions: HashMap::new(),
            root: 0,
            next_state: 1,
            num_keys: 0,
        };
        
        // Add root state
        dawg.states.insert(0, false);
        dawg
    }

    pub fn insert(&mut self, key: &[u8]) -> Result<()> {
        let mut current_state = self.root;

        // Follow existing path as far as possible
        for &symbol in key {
            if let Some(&next_state) = self.transitions.get(&(current_state, symbol)) {
                current_state = next_state;
            } else {
                // Create new state and transition
                let new_state = self.next_state;
                self.next_state += 1;
                
                self.states.insert(new_state, false);
                self.transitions.insert((current_state, symbol), new_state);
                current_state = new_state;
            }
        }

        // Mark final state as terminal
        self.states.insert(current_state, true);
        self.num_keys += 1;
        Ok(())
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        let mut current_state = self.root;

        for &symbol in key {
            if let Some(&next_state) = self.transitions.get(&(current_state, symbol)) {
                current_state = next_state;
            } else {
                return false;
            }
        }

        self.states.get(&current_state).copied().unwrap_or(false)
    }

    pub fn num_keys(&self) -> usize {
        self.num_keys
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn memory_usage(&self) -> usize {
        self.states.len() * std::mem::size_of::<(u32, bool)>() +
        self.transitions.len() * std::mem::size_of::<((u32, u8), u32)>()
    }
}

/// Simple graph walker for basic traversal
#[derive(Debug)]
pub struct SimpleGraphWalker {
    visited: HashMap<u32, bool>,
}

impl SimpleGraphWalker {
    pub fn new() -> Self {
        Self {
            visited: HashMap::new(),
        }
    }

    pub fn walk_bfs<F>(&mut self, start: u32, mut visit_fn: F) -> Result<()>
    where
        F: FnMut(u32) -> Result<Vec<u32>>,
    {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        self.visited.insert(start, true);

        while let Some(current) = queue.pop_front() {
            let neighbors = visit_fn(current)?;
            
            for neighbor in neighbors {
                if !self.visited.contains_key(&neighbor) {
                    self.visited.insert(neighbor, true);
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.visited.clear();
    }

    pub fn visited_count(&self) -> usize {
        self.visited.len()
    }
}

/// Simple fast search using linear methods
#[derive(Debug)]
pub struct SimpleFastSearch;

impl SimpleFastSearch {
    pub fn new() -> Self {
        Self
    }

    pub fn search_byte(&self, data: &[u8], target: u8) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter_map(|(i, &b)| if b == target { Some(i) } else { None })
            .collect()
    }

    pub fn find_first(&self, data: &[u8], target: u8) -> Option<usize> {
        data.iter().position(|&b| b == target)
    }

    pub fn find_last(&self, data: &[u8], target: u8) -> Option<usize> {
        data.iter().rposition(|&b| b == target)
    }

    pub fn count(&self, data: &[u8], target: u8) -> usize {
        data.iter().filter(|&&b| b == target).count()
    }

    pub fn search_pattern(&self, data: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut positions = Vec::new();
        
        if pattern.is_empty() || pattern.len() > data.len() {
            return positions;
        }
        
        for i in 0..=(data.len() - pattern.len()) {
            if data[i..i + pattern.len()] == *pattern {
                positions.push(i);
            }
        }
        
        positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_fsa_cache() {
        let mut cache = SimpleFsaCache::new(3);
        
        cache.insert(1, 10).unwrap();
        cache.insert(2, 20).unwrap();
        cache.insert(3, 30).unwrap();
        
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(1), Some(10));
        assert_eq!(cache.get(2), Some(20));
        assert_eq!(cache.get(3), Some(30));
        
        // Test eviction
        cache.insert(4, 40).unwrap();
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_simple_dawg() {
        let mut dawg = SimpleDawg::new();
        
        dawg.insert(b"cat").unwrap();
        dawg.insert(b"car").unwrap();
        dawg.insert(b"card").unwrap();
        
        assert!(dawg.contains(b"cat"));
        assert!(dawg.contains(b"car"));
        assert!(dawg.contains(b"card"));
        assert!(!dawg.contains(b"dog"));
        assert!(!dawg.contains(b"ca"));
        
        assert_eq!(dawg.num_keys(), 3);
        assert!(dawg.num_states() > 0);
        assert!(dawg.memory_usage() > 0);
    }

    #[test]
    fn test_simple_graph_walker() {
        let mut walker = SimpleGraphWalker::new();
        
        // Simple graph: 0 -> [1, 2], 1 -> [3], 2 -> [3], 3 -> []
        let graph = |node: u32| -> Result<Vec<u32>> {
            match node {
                0 => Ok(vec![1, 2]),
                1 => Ok(vec![3]),
                2 => Ok(vec![3]),
                3 => Ok(vec![]),
                _ => Ok(vec![]),
            }
        };
        
        walker.walk_bfs(0, graph).unwrap();
        assert_eq!(walker.visited_count(), 4);
        
        walker.reset();
        assert_eq!(walker.visited_count(), 0);
    }

    #[test]
    fn test_simple_fast_search() {
        let search = SimpleFastSearch::new();
        let data = b"hello world hello";
        
        // Test byte search
        let positions = search.search_byte(data, b'l');
        assert_eq!(positions, vec![2, 3, 9, 14, 15]);
        
        // Test find operations
        assert_eq!(search.find_first(data, b'l'), Some(2));
        assert_eq!(search.find_last(data, b'l'), Some(15));
        assert_eq!(search.find_first(data, b'z'), None);
        
        // Test count
        assert_eq!(search.count(data, b'l'), 5);
        assert_eq!(search.count(data, b'o'), 3);
        
        // Test pattern search
        let positions = search.search_pattern(data, b"hello");
        assert_eq!(positions, vec![0, 12]);
        
        let positions = search.search_pattern(data, b"xyz");
        assert_eq!(positions, vec![]);
    }
}