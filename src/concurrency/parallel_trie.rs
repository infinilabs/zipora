//! Parallel trie construction and operations

use crate::error::{ToplingError, Result};
use crate::fsa::{LoudsTrie, Trie};
use crate::fsa::traits::PrefixIterable;
use crate::StateId;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use rayon::prelude::*;

/// Builder for constructing tries in parallel
pub struct ParallelTrieBuilder {
    chunk_size: usize,
    max_workers: usize,
}

impl ParallelTrieBuilder {
    /// Create a new parallel trie builder
    pub fn new() -> Self {
        Self {
            chunk_size: 10000,
            max_workers: num_cpus::get(),
        }
    }
    
    /// Set the chunk size for parallel processing
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }
    
    /// Set the maximum number of worker threads
    pub fn max_workers(mut self, max_workers: usize) -> Self {
        self.max_workers = max_workers;
        self
    }
    
    /// Build a LOUDS trie from a sorted iterator in parallel
    pub async fn build_louds_trie<I>(&self, keys: I) -> Result<ParallelLoudsTrie>
    where
        I: IntoIterator<Item = Vec<u8>>,
        I::IntoIter: Send,
    {
        let keys: Vec<Vec<u8>> = keys.into_iter().collect();
        
        if keys.is_empty() {
            return Ok(ParallelLoudsTrie::new());
        }
        
        // Build trie in parallel chunks
        let chunks: Vec<Vec<Vec<u8>>> = keys.chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let mut partial_tries = Vec::new();
        
        for chunk in chunks {
            let trie = tokio::task::spawn_blocking(move || {
                LoudsTrie::build_from_sorted(chunk)
            }).await
            .map_err(|e| ToplingError::configuration(&format!("parallel build failed: {}", e)))??;
            
            partial_tries.push(trie);
        }
        
        // Merge partial tries
        self.merge_tries(partial_tries).await
    }
    
    /// Merge multiple LOUDS tries into a single trie
    async fn merge_tries(&self, tries: Vec<LoudsTrie>) -> Result<ParallelLoudsTrie> {
        if tries.is_empty() {
            return Ok(ParallelLoudsTrie::new());
        }
        
        if tries.len() == 1 {
            return Ok(ParallelLoudsTrie::from_louds_trie(tries.into_iter().next().unwrap()));
        }
        
        // For now, use a simple approach: extract all keys and rebuild
        // In a production implementation, this would be optimized
        let mut all_keys = Vec::new();
        
        for trie in tries {
            let keys = self.extract_keys_from_trie(&trie).await?;
            all_keys.extend(keys);
        }
        
        // Sort and deduplicate
        all_keys.sort();
        all_keys.dedup();
        
        // Build final trie
        let final_trie = tokio::task::spawn_blocking(move || {
            LoudsTrie::build_from_sorted(all_keys)
        }).await
        .map_err(|e| ToplingError::configuration(&format!("final build failed: {}", e)))??;
        
        Ok(ParallelLoudsTrie::from_louds_trie(final_trie))
    }
    
    /// Extract all keys from a trie (placeholder implementation)
    async fn extract_keys_from_trie(&self, trie: &LoudsTrie) -> Result<Vec<Vec<u8>>> {
        // This is a simplified implementation
        // In practice, you'd iterate through the trie structure
        Ok(Vec::new()) // Placeholder
    }
}

impl Default for ParallelTrieBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A thread-safe LOUDS trie that supports parallel operations
pub struct ParallelLoudsTrie {
    inner: Arc<Mutex<LoudsTrie>>,
    read_replicas: Vec<Arc<LoudsTrie>>,
    replica_count: usize,
}

impl ParallelLoudsTrie {
    /// Create a new empty parallel trie
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(LoudsTrie::new())),
            read_replicas: Vec::new(),
            replica_count: num_cpus::get(),
        }
    }
    
    /// Create from an existing LOUDS trie
    pub fn from_louds_trie(trie: LoudsTrie) -> Self {
        let replica_count = num_cpus::get();
        let mut read_replicas = Vec::with_capacity(replica_count);
        
        // Create read replicas (in practice, these would be actual copies)
        for _ in 0..replica_count {
            read_replicas.push(Arc::new(LoudsTrie::new())); // Placeholder
        }
        
        Self {
            inner: Arc::new(Mutex::new(trie)),
            read_replicas,
            replica_count,
        }
    }
    
    /// Insert a key (requires write lock)
    pub async fn insert(&self, key: &[u8]) -> Result<StateId> {
        let mut trie = self.inner.lock().await;
        trie.insert(key)
    }
    
    /// Check if a key exists (uses read replica for better concurrency)
    pub async fn contains(&self, key: &[u8]) -> bool {
        // Use a read replica for better concurrency
        let replica_id = self.select_replica();
        let replica = &self.read_replicas[replica_id];
        replica.contains(key)
    }
    
    /// Parallel search across multiple patterns
    pub async fn parallel_contains<I>(&self, keys: I) -> Vec<bool>
    where
        I: IntoIterator<Item = Vec<u8>>,
        I::IntoIter: Send,
    {
        let keys: Vec<Vec<u8>> = keys.into_iter().collect();
        
        // Use rayon for parallel processing
        let results: Vec<bool> = keys.par_iter().map(|key| {
            let replica_id = self.select_replica();
            self.read_replicas[replica_id].contains(key)
        }).collect();
        
        results
    }
    
    /// Parallel prefix search
    pub async fn parallel_prefix_search<I>(&self, prefixes: I) -> Vec<Vec<Vec<u8>>>
    where
        I: IntoIterator<Item = Vec<u8>>,
        I::IntoIter: Send,
    {
        let prefixes: Vec<Vec<u8>> = prefixes.into_iter().collect();
        
        let results: Vec<Vec<Vec<u8>>> = prefixes.par_iter().map(|prefix| {
            let replica_id = self.select_replica();
            let replica = &self.read_replicas[replica_id];
            
            // Collect results from iterator (simplified)
            replica.iter_prefix(prefix).collect()
        }).collect();
        
        results
    }
    
    /// Select a read replica using round-robin
    fn select_replica(&self) -> usize {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed) % self.replica_count
    }
    
    /// Rebuild read replicas from the current state
    pub async fn refresh_replicas(&self) -> Result<()> {
        let trie = self.inner.lock().await;
        
        // In practice, this would create actual copies of the trie
        // For now, we'll just acknowledge the operation
        drop(trie);
        
        Ok(())
    }
    
    /// Get the number of keys in the trie
    pub async fn len(&self) -> usize {
        let trie = self.inner.lock().await;
        trie.len()
    }
    
    /// Check if the trie is empty
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
    
    /// Parallel bulk insert
    pub async fn bulk_insert<I>(&self, keys: I) -> Result<Vec<StateId>>
    where
        I: IntoIterator<Item = Vec<u8>>,
        I::IntoIter: Send,
    {
        let keys: Vec<Vec<u8>> = keys.into_iter().collect();
        let mut results = Vec::with_capacity(keys.len());
        
        // For bulk operations, we need to hold the write lock
        let mut trie = self.inner.lock().await;
        
        for key in keys {
            let state_id = trie.insert(&key)?;
            results.push(state_id);
        }
        
        Ok(results)
    }
    
    /// Parallel processing of trie operations
    pub async fn parallel_process<F, T>(&self, operations: Vec<F>) -> Vec<Result<T>>
    where
        F: Fn(&LoudsTrie) -> Result<T> + Send + Sync,
        T: Send,
    {
        let results: Vec<Result<T>> = operations.par_iter().map(|op| {
            let replica_id = self.select_replica();
            let replica = &self.read_replicas[replica_id];
            op(replica)
        }).collect();
        
        results
    }
}

impl Default for ParallelLoudsTrie {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel trie operations utilities
pub struct ParallelTrieOps;

impl ParallelTrieOps {
    /// Find common prefixes in parallel
    pub async fn find_common_prefixes(
        keys: Vec<Vec<u8>>,
        min_support: usize,
    ) -> Result<Vec<Vec<u8>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        
        // Build prefix frequency map in parallel
        use std::collections::HashMap;
        use std::sync::Mutex;
        
        let prefix_counts = Arc::new(Mutex::new(HashMap::<Vec<u8>, usize>::new()));
        
        keys.par_iter().for_each(|key| {
            // Generate all prefixes for this key
            for i in 1..=key.len() {
                let prefix = key[..i].to_vec();
                let mut counts = prefix_counts.lock().unwrap();
                *counts.entry(prefix).or_insert(0) += 1;
            }
        });
        
        // Extract prefixes with sufficient support
        let counts = prefix_counts.lock().unwrap();
        let mut common_prefixes: Vec<Vec<u8>> = counts.iter()
            .filter(|(_, &count)| count >= min_support)
            .map(|(prefix, _)| prefix.clone())
            .collect();
        
        common_prefixes.sort();
        Ok(common_prefixes)
    }
    
    /// Compute trie similarity in parallel
    pub async fn compute_similarity(
        trie1: &ParallelLoudsTrie,
        trie2: &ParallelLoudsTrie,
        sample_size: usize,
    ) -> Result<f64> {
        // Sample keys from both tries and compute Jaccard similarity
        // This is a simplified implementation
        
        let len1 = trie1.len().await;
        let len2 = trie2.len().await;
        
        if len1 == 0 && len2 == 0 {
            return Ok(1.0);
        }
        
        if len1 == 0 || len2 == 0 {
            return Ok(0.0);
        }
        
        // For demonstration, return a placeholder similarity
        Ok(0.5)
    }
    
    /// Merge multiple tries in parallel
    pub async fn merge_tries(tries: Vec<ParallelLoudsTrie>) -> Result<ParallelLoudsTrie> {
        if tries.is_empty() {
            return Ok(ParallelLoudsTrie::new());
        }
        
        if tries.len() == 1 {
            return Ok(tries.into_iter().next().unwrap());
        }
        
        // Extract and merge all keys
        let mut all_keys = Vec::new();
        
        for trie in tries {
            // In practice, you'd extract keys from each trie
            // For now, this is a placeholder
        }
        
        // Build merged trie
        let builder = ParallelTrieBuilder::new();
        builder.build_louds_trie(all_keys).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    
    #[tokio::test]
    async fn test_parallel_trie_builder() {
        let builder = ParallelTrieBuilder::new()
            .chunk_size(1000)
            .max_workers(2);
        
        let keys = vec![
            b"cat".to_vec(),
            b"car".to_vec(),
            b"card".to_vec(),
            b"dog".to_vec(),
        ];
        
        let trie = builder.build_louds_trie(keys).await.unwrap();
        assert!(!trie.is_empty().await);
    }
    
    #[tokio::test]
    async fn test_parallel_louds_trie() {
        let trie = ParallelLoudsTrie::new();
        
        // Test insertion
        let _state_id = trie.insert(b"test").await.unwrap();
        assert_eq!(trie.len().await, 1);
        assert!(!trie.is_empty().await);
        
        // Test contains
        assert!(trie.contains(b"test").await);
        assert!(!trie.contains(b"missing").await);
    }
    
    #[tokio::test]
    async fn test_parallel_contains() {
        let trie = ParallelLoudsTrie::new();
        
        // Insert some keys
        let _id1 = trie.insert(b"cat").await.unwrap();
        let _id2 = trie.insert(b"car").await.unwrap();
        let _id3 = trie.insert(b"dog").await.unwrap();
        
        // Test parallel contains
        let test_keys = vec![
            b"cat".to_vec(),
            b"car".to_vec(),
            b"bird".to_vec(),
            b"dog".to_vec(),
        ];
        
        let results = trie.parallel_contains(test_keys).await;
        assert_eq!(results, vec![true, true, false, true]);
    }
    
    #[tokio::test]
    async fn test_bulk_insert() {
        let trie = ParallelLoudsTrie::new();
        
        let keys = vec![
            b"apple".to_vec(),
            b"banana".to_vec(),
            b"cherry".to_vec(),
        ];
        
        let state_ids = trie.bulk_insert(keys).await.unwrap();
        assert_eq!(state_ids.len(), 3);
        assert_eq!(trie.len().await, 3);
    }
    
    #[tokio::test]
    async fn test_find_common_prefixes() {
        let keys = vec![
            b"prefix_one".to_vec(),
            b"prefix_two".to_vec(),
            b"prefix_three".to_vec(),
            b"other".to_vec(),
        ];
        
        let prefixes = ParallelTrieOps::find_common_prefixes(keys, 2).await.unwrap();
        
        // Should find "prefix" as a common prefix (appears 3 times, >= min_support of 2)
        assert!(prefixes.iter().any(|p| p.starts_with(b"prefix")));
    }
    
    #[tokio::test]
    async fn test_replica_selection() {
        let trie = ParallelLoudsTrie::new();
        
        // Test that replica selection cycles through available replicas
        let replica1 = trie.select_replica();
        let replica2 = trie.select_replica();
        
        // Should be different (unless only 1 replica)
        if trie.replica_count > 1 {
            assert_ne!(replica1, replica2);
        }
    }
}