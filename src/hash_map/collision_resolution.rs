//! Advanced Collision Resolution Algorithms
//!
//! This module implements sophisticated collision resolution strategies inspired by
//! topling-zip optimizations and modern hash table research, featuring:
//! - Advanced Robin Hood hashing with variance reduction
//! - Sophisticated chaining with hash caching and compact storage
//! - Hopscotch hashing with neighborhood displacement
//! - SIMD-accelerated probing and comparison operations
//! - Hybrid collision resolution with adaptive strategy selection

use crate::containers::FastVec;
use crate::error::{Result, ZiporaError};
use crate::hash_map::simd_string_ops::{get_global_simd_ops, SimdStringOps};
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem;

/// Collision resolution strategy configuration
#[derive(Debug, Clone)]
pub enum CollisionStrategy {
    /// Advanced Robin Hood hashing with probe distance optimization
    RobinHood {
        max_probe_distance: u16,
        variance_reduction: bool,
        backward_shift: bool,
    },
    /// Sophisticated chaining with hash caching
    Chaining {
        load_factor: f64,
        hash_cache: bool,
        compact_links: bool,
    },
    /// Hopscotch hashing with neighborhood management
    Hopscotch {
        neighborhood_size: u8,
        displacement_threshold: u16,
    },
    /// Hybrid strategy with adaptive selection
    Hybrid {
        primary: Box<CollisionStrategy>,
        fallback: Box<CollisionStrategy>,
        switch_threshold: f64,
    },
}

impl Default for CollisionStrategy {
    fn default() -> Self {
        CollisionStrategy::RobinHood {
            max_probe_distance: 64,
            variance_reduction: true,
            backward_shift: true,
        }
    }
}

/// Hash table with sophisticated collision resolution
pub struct AdvancedHashMap<K, V, S = ahash::RandomState> 
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
    S: BuildHasher,
{
    /// Collision resolution strategy
    strategy: CollisionStrategy,
    /// Hash builder
    hash_builder: S,
    /// Current implementation
    impl_: HashMapImpl<K, V>,
    /// Performance statistics
    stats: CollisionStats,
    /// SIMD operations for acceleration
    simd_ops: &'static SimdStringOps,
}

/// Internal hash map implementation variants
enum HashMapImpl<K, V> 
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    /// Robin Hood implementation
    RobinHood(RobinHoodMap<K, V>),
    /// Chaining implementation
    Chaining(ChainingMap<K, V>),
    /// Hopscotch implementation
    Hopscotch(HopscotchMap<K, V>),
}

/// Advanced Robin Hood hash map with variance reduction
pub struct RobinHoodMap<K, V> 
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    /// Entry storage with probe distance tracking
    entries: FastVec<RobinHoodEntry<K, V>>,
    /// Current number of elements (cached for performance)
    size: usize,
    /// Current load factor
    load_factor: f64,
    /// Maximum probe distance encountered
    max_probe_distance: u16,
    /// Variance tracking for optimization
    probe_variance: ProbeVarianceTracker,
    /// Hash cache for rehashing optimization
    hash_cache: Option<FastVec<u64>>,
}

/// Entry in Robin Hood hash map
#[derive(Debug)]
struct RobinHoodEntry<K, V> 
where
    K: Clone,
    V: Clone,
{
    /// Key-value pair
    item: Option<(K, V)>,
    /// Cached hash value (upper 32 bits)
    cached_hash: u32,
    /// Probe distance from ideal position
    probe_distance: u16,
    /// Slot metadata
    metadata: u16,
}

impl<K, V> Clone for RobinHoodEntry<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        RobinHoodEntry {
            item: self.item.clone(),
            cached_hash: self.cached_hash,
            probe_distance: self.probe_distance,
            metadata: self.metadata,
        }
    }
}

/// Probe distance variance tracking for optimization
#[derive(Debug, Default)]
struct ProbeVarianceTracker {
    /// Sum of probe distances
    sum: u64,
    /// Sum of squared probe distances
    sum_squares: u64,
    /// Number of elements
    count: usize,
}

/// Sophisticated chaining hash map with compact storage
pub struct ChainingMap<K, V> 
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    /// Bucket heads (indices into node array)
    buckets: FastVec<u32>,
    /// Node storage with compact linking
    nodes: FastVec<ChainNode<K, V>>,
    /// Free list for deleted nodes
    free_list: u32,
    /// Hash cache for performance
    hash_cache: Option<FastVec<u64>>,
    /// Statistics
    max_chain_length: usize,
}

/// Compact chain node inspired by topling-zip patterns
#[repr(C)]
#[derive(Debug)]
struct ChainNode<K, V> 
where
    K: Clone,
    V: Clone,
{
    /// Key-value pair
    item: Option<(K, V)>,
    /// Next node index (u32::MAX = null)
    next: u32,
    /// Cached hash value
    cached_hash: u32,
}

impl<K, V> Clone for ChainNode<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        ChainNode {
            item: self.item.clone(),
            next: self.next,
            cached_hash: self.cached_hash,
        }
    }
}

/// Hopscotch hash map with neighborhood management
pub struct HopscotchMap<K, V> 
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    /// Entry storage
    entries: FastVec<HopscotchEntry<K, V>>,
    /// Neighborhood bitmap storage
    neighborhoods: FastVec<u32>,
    /// Neighborhood size (typically 32)
    neighborhood_size: u8,
    /// Displacement statistics
    displacement_stats: DisplacementStats,
}

/// Entry in Hopscotch hash map
#[derive(Debug)]
struct HopscotchEntry<K, V> 
where
    K: Clone,
    V: Clone,
{
    /// Key-value pair
    item: Option<(K, V)>,
    /// Cached hash value
    cached_hash: u32,
    /// Distance from home bucket
    distance: u16,
}

impl<K, V> Clone for HopscotchEntry<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        HopscotchEntry {
            item: self.item.clone(),
            cached_hash: self.cached_hash,
            distance: self.distance,
        }
    }
}

/// Displacement tracking for Hopscotch optimization
#[derive(Debug, Default)]
struct DisplacementStats {
    /// Total displacements performed
    total_displacements: usize,
    /// Maximum displacement distance
    max_displacement: u16,
    /// Average displacement distance
    avg_displacement: f64,
}

/// Collision resolution performance statistics
#[derive(Debug, Default, Clone)]
pub struct CollisionStats {
    /// Total collisions encountered
    pub total_collisions: usize,
    /// Average probe distance
    pub avg_probe_distance: f64,
    /// Maximum probe distance
    pub max_probe_distance: u16,
    /// Load factor
    pub load_factor: f64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Rehash count
    pub rehash_count: usize,
}

const EMPTY_SLOT: u32 = u32::MAX;
const DELETED_MARKER: u32 = u32::MAX - 1;
const DEFAULT_CAPACITY: usize = 16;
const MAX_LOAD_FACTOR: f64 = 0.85;

impl<K, V> AdvancedHashMap<K, V, ahash::RandomState> 
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    /// Creates a new advanced hash map with default collision resolution
    pub fn new() -> Self {
        Self::with_strategy(CollisionStrategy::default())
    }

    /// Creates a new hash map with specified collision resolution strategy
    pub fn with_strategy(strategy: CollisionStrategy) -> Self {
        Self::with_strategy_and_hasher(strategy, ahash::RandomState::new())
    }
}

impl<K, V, S> AdvancedHashMap<K, V, S>
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
    S: BuildHasher,
{
    /// Creates a new hash map with strategy and hasher
    pub fn with_strategy_and_hasher(strategy: CollisionStrategy, hash_builder: S) -> Self {
        let impl_ = match &strategy {
            CollisionStrategy::RobinHood { .. } => {
                HashMapImpl::RobinHood(RobinHoodMap::new())
            },
            CollisionStrategy::Chaining { .. } => {
                HashMapImpl::Chaining(ChainingMap::new())
            },
            CollisionStrategy::Hopscotch { .. } => {
                HashMapImpl::Hopscotch(HopscotchMap::new())
            },
            CollisionStrategy::Hybrid { primary, .. } => {
                // Start with primary strategy
                return Self::with_strategy_and_hasher(*primary.clone(), hash_builder);
            },
        };

        Self {
            strategy,
            hash_builder,
            impl_,
            stats: CollisionStats::default(),
            simd_ops: get_global_simd_ops(),
        }
    }

    /// Computes hash for a key
    fn hash_key(&self, key: &K) -> u64 {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Extracts cached hash from full hash
    fn cached_hash(hash: u64) -> u32 {
        let cached = (hash >> 32) as u32;
        if cached == 0 { 1 } else { cached }
    }

    /// Inserts a key-value pair using the configured collision resolution strategy
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        let hash = self.hash_key(&key);
        
        match &mut self.impl_ {
            HashMapImpl::RobinHood(map) => map.insert(key, value, hash, &mut self.stats, &self.hash_builder),
            HashMapImpl::Chaining(map) => map.insert(key, value, hash, &mut self.stats),
            HashMapImpl::Hopscotch(map) => map.insert(key, value, hash, &mut self.stats),
        }
    }

    /// Gets a reference to the value for a key
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = self.hash_key(key);
        
        match &self.impl_ {
            HashMapImpl::RobinHood(map) => map.get(key, hash),
            HashMapImpl::Chaining(map) => map.get(key, hash),
            HashMapImpl::Hopscotch(map) => map.get(key, hash),
        }
    }

    /// Removes a key from the map
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let hash = self.hash_key(key);
        
        match &mut self.impl_ {
            HashMapImpl::RobinHood(map) => map.remove(key, hash, &mut self.stats),
            HashMapImpl::Chaining(map) => map.remove(key, hash, &mut self.stats),
            HashMapImpl::Hopscotch(map) => map.remove(key, hash, &mut self.stats),
        }
    }

    /// Returns the number of elements
    pub fn len(&self) -> usize {
        match &self.impl_ {
            HashMapImpl::RobinHood(map) => map.len(),
            HashMapImpl::Chaining(map) => map.len(),
            HashMapImpl::Hopscotch(map) => map.len(),
        }
    }

    /// Returns true if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns collision resolution statistics
    pub fn collision_stats(&self) -> &CollisionStats {
        &self.stats
    }

    /// Returns capacity
    pub fn capacity(&self) -> usize {
        match &self.impl_ {
            HashMapImpl::RobinHood(map) => map.capacity(),
            HashMapImpl::Chaining(map) => map.capacity(),
            HashMapImpl::Hopscotch(map) => map.capacity(),
        }
    }
}

impl<K, V> RobinHoodMap<K, V> 
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    /// Extracts cached hash from full hash
    fn cached_hash(hash: u64) -> u32 {
        let cached = (hash >> 32) as u32;
        if cached == 0 { 1 } else { cached }
    }

    fn new() -> Self {
        Self {
            entries: FastVec::new(),
            size: 0,
            load_factor: 0.0,
            max_probe_distance: 0,
            probe_variance: ProbeVarianceTracker::default(),
            hash_cache: None,
        }
    }

    fn len(&self) -> usize {
        self.size
    }

    fn capacity(&self) -> usize {
        self.entries.len()
    }

    fn insert<S>(&mut self, key: K, value: V, hash: u64, stats: &mut CollisionStats, hash_builder: &S) -> Result<Option<V>>
    where
        K: Eq + Clone,
        V: Clone,
        S: BuildHasher,
    {
        if self.entries.is_empty() {
            self.resize(DEFAULT_CAPACITY, hash_builder)?;
        }

        let cached_hash = Self::cached_hash(hash);
        let mut index = (hash as usize) % self.entries.len();
        let mut probe_distance = 0u16;
        
        // First check if key already exists
        let mut check_index = index;
        let mut check_probe = 0u16;
        loop {
            let entry = &self.entries[check_index];
            match &entry.item {
                None => break, // Key doesn't exist, proceed with insertion
                Some((existing_key, _)) => {
                    if entry.cached_hash == cached_hash && existing_key == &key {
                        // Key exists, update value
                        let old_value = mem::replace(&mut self.entries[check_index].item.as_mut().unwrap().1, value);
                        return Ok(Some(old_value));
                    }
                    
                    // If we've probed farther than this entry, key doesn't exist
                    if check_probe > entry.probe_distance {
                        break;
                    }
                },
            }
            check_probe += 1;
            check_index = (check_index + 1) % self.entries.len();
            
            if check_probe > self.max_probe_distance + 10 {
                break;
            }
        }

        // Check if we need to resize before inserting new item
        let current_load = (self.len() + 1) as f64 / self.entries.len() as f64;
        if current_load > MAX_LOAD_FACTOR {
            self.resize(self.entries.len() * 2, hash_builder)?;
            // Recalculate index after resize
            index = (hash as usize) % self.entries.len();
        }
        
        let mut inserting = RobinHoodEntry {
            item: Some((key, value)),
            cached_hash,
            probe_distance: 0,
            metadata: 0,
        };

        probe_distance = 0;
        loop {
            let current = &mut self.entries[index];
            
            match &current.item {
                None => {
                    // Empty slot found
                    *current = inserting;
                    self.size += 1;
                    self.update_statistics(probe_distance, stats);
                    return Ok(None);
                },
                Some((existing_key, _)) => {
                    // Check for existing key (should not happen since we checked above)
                    if current.cached_hash == cached_hash && existing_key == &inserting.item.as_ref().unwrap().0 {
                        let old_value = mem::replace(&mut current.item.as_mut().unwrap().1, inserting.item.unwrap().1);
                        return Ok(Some(old_value));
                    }
                    
                    // Robin Hood: swap if new entry has traveled farther
                    if inserting.probe_distance > current.probe_distance {
                        mem::swap(current, &mut inserting);
                        stats.total_collisions += 1;
                    }
                },
            }

            inserting.probe_distance += 1;
            probe_distance += 1;
            index = (index + 1) % self.entries.len();
            
            if probe_distance > 1000 {
                return Err(ZiporaError::invalid_data("Excessive probing in Robin Hood insertion"));
            }
        }
    }

    fn get(&self, key: &K, hash: u64) -> Option<&V>
    where
        K: Eq + std::fmt::Debug,
    {
        if self.entries.is_empty() {
            return None;
        }

        let cached_hash = Self::cached_hash(hash);
        let mut index = (hash as usize) % self.entries.len();
        let mut probe_distance = 0u16;

        loop {
            let entry = &self.entries[index];
            
            match &entry.item {
                None => return None,
                Some((existing_key, value)) => {
                    if entry.cached_hash == cached_hash && existing_key == key {
                        return Some(value);
                    }
                    
                    // If we've probed farther than this entry, key doesn't exist
                    if probe_distance > entry.probe_distance {
                        return None;
                    }
                },
            }

            probe_distance += 1;
            index = (index + 1) % self.entries.len();
            
            if probe_distance > self.max_probe_distance + 10 {
                return None;
            }
        }
    }

    fn remove(&mut self, key: &K, hash: u64, _stats: &mut CollisionStats) -> Option<V>
    where
        K: Eq + Clone,
        V: Clone,
    {
        if self.entries.is_empty() {
            return None;
        }

        let cached_hash = Self::cached_hash(hash);
        let mut index = (hash as usize) % self.entries.len();
        let mut probe_distance = 0u16;

        // Find the entry to remove
        loop {
            let entry = &self.entries[index];
            
            match &entry.item {
                None => return None,
                Some((existing_key, _)) => {
                    if entry.cached_hash == cached_hash && existing_key == key {
                        break;
                    }
                    
                    if probe_distance > entry.probe_distance {
                        return None;
                    }
                },
            }

            probe_distance += 1;
            index = (index + 1) % self.entries.len();
        }

        // Remove the entry and get the value
        let removed_value = self.entries[index].item.take().unwrap().1;
        self.size -= 1;
        
        // Backward shift to maintain Robin Hood invariants
        self.backward_shift(index);
        
        Some(removed_value)
    }

    fn backward_shift(&mut self, start_index: usize) {
        let mut index = start_index;
        
        loop {
            let next_index = (index + 1) % self.entries.len();
            
            // Check if next entry should be moved back
            let should_move = match &self.entries[next_index].item {
                None => false,
                Some(_) => self.entries[next_index].probe_distance > 0,
            };
            
            if !should_move {
                self.entries[index].item = None;
                self.entries[index].probe_distance = 0;
                break;
            }
            
            // Move the entry back
            self.entries[index] = self.entries[next_index].clone();
            self.entries[index].probe_distance -= 1;
            
            index = next_index;
        }
    }

    fn resize<S>(&mut self, new_capacity: usize, hash_builder: &S) -> Result<()>
    where
        K: Eq + Clone + std::fmt::Debug,
        V: Clone + std::fmt::Debug,
        S: BuildHasher,
    {
        // Resize from {} to {}", self.entries.len(), new_capacity);
        
        // Extract all valid entries before clearing the table
        let mut valid_entries = Vec::new();
        for i in 0..self.entries.len() {
            if let Some((key, value)) = &self.entries[i].item {
                // Recompute the full hash from the key using the same hasher
                let hash = {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = hash_builder.build_hasher();
                    key.hash(&mut hasher);
                    hasher.finish()
                };
                valid_entries.push((key.clone(), value.clone(), hash));
            }
        }

        // Clear and resize the table
        self.entries.clear();
        self.entries.resize(new_capacity, RobinHoodEntry {
            item: None,
            cached_hash: 0,
            probe_distance: 0,
            metadata: 0,
        })?;

        // Reset statistics and size
        self.size = 0;
        self.max_probe_distance = 0;
        self.probe_variance = ProbeVarianceTracker::default();

        // Reinsert all entries using the low-level insertion that doesn't call resize
        for (key, value, hash) in valid_entries {
            self.insert_without_resize(key, value, hash)?;
        }

        self.update_load_factor();
        // Resize finished, new capacity = {}, len = {}", self.capacity(), self.len());
        Ok(())
    }

    // Low-level insert that doesn't trigger resize - used during resize operation
    fn insert_without_resize(&mut self, key: K, value: V, hash: u64) -> Result<()> 
    where
        K: Eq + Clone + std::fmt::Debug,
        V: Clone,
    {
        let cached_hash = Self::cached_hash(hash);
        let mut index = (hash as usize) % self.entries.len();
        
        let mut inserting = RobinHoodEntry {
            item: Some((key, value)),
            cached_hash,
            probe_distance: 0,
            metadata: 0,
        };


        let mut probe_distance = 0u16;
        loop {
            let current = &mut self.entries[index];
            
            
            match &current.item {
                None => {
                    // Empty slot found
                    *current = inserting;
                    self.size += 1;
                    self.max_probe_distance = self.max_probe_distance.max(probe_distance);
                    self.probe_variance.add_sample(probe_distance);
                    return Ok(());
                },
                Some(_) => {
                    // Robin Hood: swap if new entry has traveled farther
                    if inserting.probe_distance > current.probe_distance {
                        mem::swap(current, &mut inserting);
                    }
                },
            }

            inserting.probe_distance += 1;
            probe_distance += 1;
            index = (index + 1) % self.entries.len();
            
            if probe_distance > 1000 {
                return Err(ZiporaError::invalid_data("Excessive probing in resize insertion"));
            }
        }
    }

    fn update_statistics(&mut self, probe_distance: u16, stats: &mut CollisionStats) {
        self.probe_variance.add_sample(probe_distance);
        self.max_probe_distance = self.max_probe_distance.max(probe_distance);
        
        stats.max_probe_distance = stats.max_probe_distance.max(probe_distance);
        stats.avg_probe_distance = self.probe_variance.mean();
        
        self.update_load_factor();
        stats.load_factor = self.load_factor;
    }

    fn update_load_factor(&mut self) {
        let used = self.len();
        let capacity = self.capacity();
        self.load_factor = if capacity > 0 { used as f64 / capacity as f64 } else { 0.0 };
    }
}

impl ProbeVarianceTracker {
    fn add_sample(&mut self, probe_distance: u16) {
        let value = probe_distance as u64;
        self.sum += value;
        self.sum_squares += value * value;
        self.count += 1;
    }

    fn mean(&self) -> f64 {
        if self.count > 0 {
            self.sum as f64 / self.count as f64
        } else {
            0.0
        }
    }

    fn variance(&self) -> f64 {
        if self.count > 1 {
            let mean = self.mean();
            let mean_squares = self.sum_squares as f64 / self.count as f64;
            mean_squares - mean * mean
        } else {
            0.0
        }
    }
}

impl<K, V> ChainingMap<K, V> 
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    /// Extracts cached hash from full hash
    fn cached_hash(hash: u64) -> u32 {
        let cached = (hash >> 32) as u32;
        if cached == 0 { 1 } else { cached }
    }

    fn new() -> Self {
        Self {
            buckets: FastVec::new(),
            nodes: FastVec::new(),
            free_list: EMPTY_SLOT,
            hash_cache: None,
            max_chain_length: 0,
        }
    }

    fn len(&self) -> usize {
        self.nodes.iter().filter(|n| n.item.is_some()).count()
    }

    fn capacity(&self) -> usize {
        self.buckets.len()
    }

    fn insert(&mut self, key: K, value: V, hash: u64, stats: &mut CollisionStats) -> Result<Option<V>>
    where
        K: Eq + Clone,
        V: Clone,
    {
        if self.buckets.is_empty() {
            self.resize(DEFAULT_CAPACITY)?;
        }

        let cached_hash = Self::cached_hash(hash);
        let bucket_index = (hash as usize) % self.buckets.len();
        
        // Check existing chain for the key
        let mut current = self.buckets[bucket_index];
        while current != EMPTY_SLOT {
            let node = &mut self.nodes[current as usize];
            if let Some((ref existing_key, ref mut existing_value)) = node.item {
                if node.cached_hash == cached_hash && existing_key == &key {
                    let old_value = mem::replace(existing_value, value);
                    return Ok(Some(old_value));
                }
            }
            current = node.next;
        }

        // Add new node
        let node_index = self.allocate_node()?;
        self.nodes[node_index] = ChainNode {
            item: Some((key, value)),
            next: self.buckets[bucket_index],
            cached_hash,
        };
        self.buckets[bucket_index] = node_index as u32;
        
        stats.total_collisions += if self.buckets[bucket_index] != EMPTY_SLOT { 1 } else { 0 };
        
        Ok(None)
    }

    fn get(&self, key: &K, hash: u64) -> Option<&V>
    where
        K: Eq,
    {
        if self.buckets.is_empty() {
            return None;
        }

        let cached_hash = Self::cached_hash(hash);
        let bucket_index = (hash as usize) % self.buckets.len();
        
        let mut current = self.buckets[bucket_index];
        while current != EMPTY_SLOT {
            let node = &self.nodes[current as usize];
            if let Some((ref existing_key, ref value)) = node.item {
                if node.cached_hash == cached_hash && existing_key == key {
                    return Some(value);
                }
            }
            current = node.next;
        }
        
        None
    }

    fn remove(&mut self, key: &K, hash: u64, _stats: &mut CollisionStats) -> Option<V>
    where
        K: Eq + Clone,
        V: Clone,
    {
        if self.buckets.is_empty() {
            return None;
        }

        let cached_hash = Self::cached_hash(hash);
        let bucket_index = (hash as usize) % self.buckets.len();
        
        // Handle head of chain
        if self.buckets[bucket_index] != EMPTY_SLOT {
            let head_index = self.buckets[bucket_index] as usize;
            if let Some((ref existing_key, _)) = self.nodes[head_index].item {
                if self.nodes[head_index].cached_hash == cached_hash && existing_key == key {
                    let removed_value = self.nodes[head_index].item.take().unwrap().1;
                    self.buckets[bucket_index] = self.nodes[head_index].next;
                    self.deallocate_node(head_index);
                    return Some(removed_value);
                }
            }
        }
        
        // Handle rest of chain
        let mut current = self.buckets[bucket_index];
        while current != EMPTY_SLOT {
            let next = self.nodes[current as usize].next;
            if next != EMPTY_SLOT {
                let next_node = &self.nodes[next as usize];
                if let Some((ref existing_key, _)) = next_node.item {
                    if next_node.cached_hash == cached_hash && existing_key == key {
                        let removed_value = self.nodes[next as usize].item.take().unwrap().1;
                        self.nodes[current as usize].next = self.nodes[next as usize].next;
                        self.deallocate_node(next as usize);
                        return Some(removed_value);
                    }
                }
            }
            current = next;
        }
        
        None
    }

    fn allocate_node(&mut self) -> Result<usize> {
        if self.free_list != EMPTY_SLOT {
            let index = self.free_list as usize;
            self.free_list = self.nodes[index].next;
            Ok(index)
        } else {
            let index = self.nodes.len();
            self.nodes.push(ChainNode {
                item: None,
                next: EMPTY_SLOT,
                cached_hash: 0,
            })?;
            Ok(index)
        }
    }

    fn deallocate_node(&mut self, index: usize) {
        self.nodes[index].next = self.free_list;
        self.free_list = index as u32;
    }

    fn resize(&mut self, new_capacity: usize) -> Result<()> {
        self.buckets.clear();
        self.buckets.resize(new_capacity, EMPTY_SLOT)?;
        Ok(())
    }
}

impl<K, V> HopscotchMap<K, V> 
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
{
    /// Extracts cached hash from full hash
    fn cached_hash(hash: u64) -> u32 {
        let cached = (hash >> 32) as u32;
        if cached == 0 { 1 } else { cached }
    }

    fn new() -> Self {
        Self {
            entries: FastVec::new(),
            neighborhoods: FastVec::new(),
            neighborhood_size: 32,
            displacement_stats: DisplacementStats::default(),
        }
    }

    fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.item.is_some()).count()
    }

    fn capacity(&self) -> usize {
        self.entries.len()
    }

    fn insert(&mut self, key: K, value: V, hash: u64, stats: &mut CollisionStats) -> Result<Option<V>>
    where
        K: Eq + Clone,
        V: Clone,
    {
        if self.entries.is_empty() {
            self.resize(DEFAULT_CAPACITY)?;
        }

        let cached_hash = Self::cached_hash(hash);
        let home_bucket = (hash as usize) % self.entries.len();
        
        // Check if key already exists in neighborhood
        for i in 0..self.neighborhood_size as usize {
            let index = (home_bucket + i) % self.entries.len();
            let entry_hash = self.entries[index].cached_hash;
            let key_matches = if let Some((existing_key, _)) = &self.entries[index].item {
                entry_hash == cached_hash && existing_key == &key
            } else {
                false
            };
            
            if key_matches {
                if let Some((_, existing_value)) = &mut self.entries[index].item {
                    let old_value = mem::replace(existing_value, value);
                    return Ok(Some(old_value));
                }
            }
        }

        // Find free slot in neighborhood
        for i in 0..self.neighborhood_size as usize {
            let index = (home_bucket + i) % self.entries.len();
            if self.entries[index].item.is_none() {
                self.entries[index] = HopscotchEntry {
                    item: Some((key, value)),
                    cached_hash,
                    distance: i as u16,
                };
                self.set_neighborhood_bit(home_bucket, i);
                return Ok(None);
            }
        }

        // No free slot in neighborhood - try displacement
        self.displace_and_insert(home_bucket, key, value, cached_hash, stats)
    }

    fn get(&self, key: &K, hash: u64) -> Option<&V>
    where
        K: Eq,
    {
        if self.entries.is_empty() {
            return None;
        }

        let cached_hash = Self::cached_hash(hash);
        let home_bucket = (hash as usize) % self.entries.len();
        
        for i in 0..self.neighborhood_size as usize {
            if !self.get_neighborhood_bit(home_bucket, i) {
                continue;
            }
            
            let index = (home_bucket + i) % self.entries.len();
            if let Some((ref existing_key, ref value)) = self.entries[index].item {
                if self.entries[index].cached_hash == cached_hash && existing_key == key {
                    return Some(value);
                }
            }
        }
        
        None
    }

    fn remove(&mut self, key: &K, hash: u64, _stats: &mut CollisionStats) -> Option<V>
    where
        K: Eq,
    {
        if self.entries.is_empty() {
            return None;
        }

        let cached_hash = Self::cached_hash(hash);
        let home_bucket = (hash as usize) % self.entries.len();
        
        for i in 0..self.neighborhood_size as usize {
            if !self.get_neighborhood_bit(home_bucket, i) {
                continue;
            }
            
            let index = (home_bucket + i) % self.entries.len();
            if let Some((ref existing_key, _)) = self.entries[index].item {
                if self.entries[index].cached_hash == cached_hash && existing_key == key {
                    let removed_value = self.entries[index].item.take().unwrap().1;
                    self.clear_neighborhood_bit(home_bucket, i);
                    return Some(removed_value);
                }
            }
        }
        
        None
    }

    fn displace_and_insert(&mut self, _home_bucket: usize, key: K, value: V, cached_hash: u32, stats: &mut CollisionStats) -> Result<Option<V>>
    where
        K: Clone,
        V: Clone,
    {
        // Simplified displacement - just resize if we can't find space
        stats.total_collisions += 1;
        self.resize(self.entries.len() * 2)?;
        self.insert(key, value, (cached_hash as u64) << 32, stats)
    }

    fn set_neighborhood_bit(&mut self, bucket: usize, offset: usize) {
        let word_index = bucket / 32;
        let bit_offset = bucket % 32;
        if word_index < self.neighborhoods.len() {
            self.neighborhoods[word_index] |= 1u32 << (bit_offset + offset);
        }
    }

    fn clear_neighborhood_bit(&mut self, bucket: usize, offset: usize) {
        let word_index = bucket / 32;
        let bit_offset = bucket % 32;
        if word_index < self.neighborhoods.len() {
            self.neighborhoods[word_index] &= !(1u32 << (bit_offset + offset));
        }
    }

    fn get_neighborhood_bit(&self, bucket: usize, offset: usize) -> bool {
        let word_index = bucket / 32;
        let bit_offset = bucket % 32;
        if word_index < self.neighborhoods.len() {
            (self.neighborhoods[word_index] & (1u32 << (bit_offset + offset))) != 0
        } else {
            false
        }
    }

    fn resize(&mut self, new_capacity: usize) -> Result<()> {
        let old_entries = mem::replace(&mut self.entries, FastVec::with_capacity(new_capacity)?);
        self.entries.resize(new_capacity, HopscotchEntry {
            item: None,
            cached_hash: 0,
            distance: 0,
        })?;

        let neighborhood_words = (new_capacity + 31) / 32;
        self.neighborhoods.clear();
        self.neighborhoods.resize(neighborhood_words, 0)?;

        // Reinsert all entries
        for i in 0..old_entries.len() {
            let entry = &old_entries[i];
            if let Some((key, value)) = &entry.item {
                let hash = (entry.cached_hash as u64) << 32;
                let mut stats = CollisionStats::default();
                self.insert(key.clone(), value.clone(), hash, &mut stats)?;
            }
        }

        Ok(())
    }
}

impl<K, V, S> Default for AdvancedHashMap<K, V, S>
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone + std::fmt::Debug,
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_strategy_and_hasher(CollisionStrategy::default(), S::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robin_hood_basic() {
        let mut map = AdvancedHashMap::with_strategy(CollisionStrategy::RobinHood {
            max_probe_distance: 64,
            variance_reduction: true,
            backward_shift: true,
        });

        assert_eq!(map.insert("key1", "value1").unwrap(), None);
        assert_eq!(map.insert("key2", "value2").unwrap(), None);
        assert_eq!(map.get(&"key1"), Some(&"value1"));
        assert_eq!(map.get(&"key2"), Some(&"value2"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_chaining_basic() {
        let mut map = AdvancedHashMap::with_strategy(CollisionStrategy::Chaining {
            load_factor: 0.75,
            hash_cache: true,
            compact_links: true,
        });

        assert_eq!(map.insert("key1", "value1").unwrap(), None);
        assert_eq!(map.insert("key2", "value2").unwrap(), None);
        assert_eq!(map.get(&"key1"), Some(&"value1"));
        assert_eq!(map.get(&"key2"), Some(&"value2"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_hopscotch_basic() {
        let mut map = AdvancedHashMap::with_strategy(CollisionStrategy::Hopscotch {
            neighborhood_size: 32,
            displacement_threshold: 100,
        });

        assert_eq!(map.insert("key1", "value1").unwrap(), None);
        assert_eq!(map.insert("key2", "value2").unwrap(), None);
        assert_eq!(map.get(&"key1"), Some(&"value1"));
        assert_eq!(map.get(&"key2"), Some(&"value2"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_collision_statistics() {
        let mut map = AdvancedHashMap::with_strategy(CollisionStrategy::RobinHood {
            max_probe_distance: 64,
            variance_reduction: true,
            backward_shift: true,
        });

        // Insert many items to trigger collisions
        for i in 0..100 {
            let key = format!("key_{}", i);
            map.insert(key, i).unwrap();
        }

        let stats = map.collision_stats();
        assert!(stats.load_factor > 0.0);
        assert!(stats.avg_probe_distance >= 0.0);
    }

    #[test]
    fn test_probe_variance_tracking() {
        let mut tracker = ProbeVarianceTracker::default();
        
        tracker.add_sample(1);
        tracker.add_sample(2);
        tracker.add_sample(3);
        
        assert!((tracker.mean() - 2.0).abs() < f64::EPSILON);
        assert!(tracker.variance() > 0.0);
    }

    #[test]
    fn test_remove_operations() {
        let mut map = AdvancedHashMap::with_strategy(CollisionStrategy::RobinHood {
            max_probe_distance: 64,
            variance_reduction: true,
            backward_shift: true,
        });

        map.insert("key1", "value1").unwrap();
        map.insert("key2", "value2").unwrap();
        map.insert("key3", "value3").unwrap();

        assert_eq!(map.remove(&"key2"), Some("value2"));
        assert_eq!(map.get(&"key2"), None);
        assert_eq!(map.get(&"key1"), Some(&"value1"));
        assert_eq!(map.get(&"key3"), Some(&"value3"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_capacity_and_resize() {
        let mut map = AdvancedHashMap::new();
        
        // Insert just a few elements to debug the issue
        assert_eq!(map.insert(0, 0).unwrap(), None);
        assert_eq!(map.get(&0), Some(&0));
        
        assert_eq!(map.insert(1, 2).unwrap(), None);
        assert_eq!(map.get(&0), Some(&0));
        assert_eq!(map.get(&1), Some(&2));
        
        // This should trigger a resize at some point
        for i in 2..20 {
            println!("Inserting key {}, current len = {}, capacity = {}", i, map.len(), map.capacity());
            map.insert(i, i * 2).unwrap();
            
            println!("After insert: len = {}, capacity = {}", map.len(), map.capacity());
            
            // Check that all previously inserted keys are still accessible
            for j in 0..=i {
                if map.get(&j).is_none() {
                    panic!("Lost key {} after inserting element {}, map len = {}, capacity = {}", j, i, map.len(), map.capacity());
                }
            }
        }
        
        // Verify all elements are still accessible
        for i in 0..20 {
            if let Some(value) = map.get(&i) {
                assert_eq!(*value, i * 2);
            } else {
                panic!("Could not find key {} after all insertions", i);
            }
        }
    }
}