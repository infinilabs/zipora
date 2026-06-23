use super::config::{
    TrieStrategy,
    ZiporaTrieConfig,
};
use super::storage::{CritBitNode, PatriciaNode, TrieStorage};
use crate::StateId;
use crate::containers::FastVec;
use crate::containers::specialized::UintVector;
use crate::error::{Result, ZiporaError};
use crate::fsa::traits::{
    FiniteStateAutomaton, PrefixIterable, Trie, TrieStats,
};
use crate::memory::SecureMemoryPool;
use crate::memory::cache_layout::{CacheLayoutConfig, CacheOptimizedAllocator};
use crate::succinct::RankSelectOps;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Unified trie implementation with strategy-based configuration
///
/// ZiporaTrie consolidates all Zipora trie variants into a single,
/// highly configurable implementation. Different behaviors are achieved
/// through strategy configuration rather than separate implementations.
///
/// # Examples
///
/// ```rust
/// use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig};
/// use zipora::fsa::traits::Trie;
/// use zipora::succinct::RankSelectInterleaved256;
///
/// // Cache-optimized trie (Patricia, with explicit type parameter)
/// let mut trie: ZiporaTrie<RankSelectInterleaved256> =
///     ZiporaTrie::with_config(ZiporaTrieConfig::cache_optimized());
/// trie.insert(b"hello").unwrap();
/// trie.insert(b"world").unwrap();
///
/// // Default trie (DoubleArray)
/// let mut da_trie: ZiporaTrie = ZiporaTrie::new();
/// da_trie.insert(b"fast").unwrap();
/// assert!(da_trie.contains(b"fast"));
///
/// // Space-optimized (LOUDS) and string-specialized (CriticalBit) are
/// // not yet implemented — insert returns NotSupported.
/// let mut space_trie: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized());
/// assert!(space_trie.insert(b"key").is_err());
/// ```
#[derive(Debug)]
pub struct ZiporaTrie<R = crate::succinct::RankSelectInterleaved256>
where
    R: RankSelectOps,
{
    /// Configuration strategy
    config: ZiporaTrieConfig,
    /// Internal storage implementation
    storage: TrieStorage<R>,
    /// Performance statistics
    stats: TrieStats,
    /// Track whether stats need recomputation
    stats_dirty: bool,
    /// Cache optimization components
    cache_allocator: Option<CacheOptimizedAllocator>,
    /// Memory pool for allocation
    _memory_pool: Option<Arc<SecureMemoryPool>>,
    /// Root state for traversal
    root_state: StateId,
}

impl<R> ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    /// Create a new trie with default configuration
    pub fn new() -> Self {
        Self::with_config(ZiporaTrieConfig::default())
    }

    /// Create a new trie with custom configuration
    pub fn with_config(config: ZiporaTrieConfig) -> Self {
        let cache_allocator = if config.cache_optimization {
            Some(CacheOptimizedAllocator::new(CacheLayoutConfig::default()))
        } else {
            None
        };

        let storage = Self::create_storage(&config);

        Self {
            config,
            storage,
            stats: TrieStats::new(),
            stats_dirty: false,
            cache_allocator,
            _memory_pool: None,
            root_state: 0,
        }
    }

    /// Create storage based on strategy configuration
    fn create_storage(config: &ZiporaTrieConfig) -> TrieStorage<R> {
        match &config.trie_strategy {
            TrieStrategy::Patricia { .. } => TrieStorage::Patricia {
                nodes: FastVec::new(),
                edge_data: FastVec::new(),
                compressed_paths: HashMap::new(),
            },
            TrieStrategy::CriticalBit { .. } => TrieStorage::CriticalBit {
                nodes: FastVec::new(),
                keys: FastVec::new(),
                critical_cache: HashMap::new(),
            },
            TrieStrategy::DoubleArray {
                initial_capacity, ..
            } => {
                // Referenced project pattern: start minimal SIZE, but respect CAPACITY hint
                // Referenced C++ implementation line 70: states.resize(1) - minimal size
                // Our approach: reserve capacity but only allocate 1 state (minimal memory)

                // Create vectors with capacity - these operations can fail on OOM
                let mut base = match FastVec::with_capacity(*initial_capacity) {
                    Ok(vec) => vec,
                    Err(_) => {
                        // Fallback to minimal capacity if requested capacity fails
                        FastVec::with_capacity(1).unwrap_or_else(|_| FastVec::new())
                    }
                };

                let mut check = match FastVec::with_capacity(*initial_capacity) {
                    Ok(vec) => vec,
                    Err(_) => {
                        // Fallback to minimal capacity if requested capacity fails
                        FastVec::with_capacity(1).unwrap_or_else(|_| FastVec::new())
                    }
                };

                // Initialize with just root state (referenced project: line 70)
                // CRITICAL: Root base must be non-zero to allow transitions
                // Using 1 as the base means child states will be at base+symbol = 1+symbol
                // SAFETY: These push operations on empty vectors cannot fail unless we're completely OOM
                // In that case, the program cannot continue anyway
                let _ = base.push(1); // Ignore error - if this fails, we're out of memory
                let _ = check.push(0); // Ignore error - if this fails, we're out of memory

                TrieStorage::DoubleArray {
                    base,
                    check,
                    free_list: VecDeque::new(),
                    state_count: 1, // Start with root state
                }
            }
            TrieStrategy::Louds { .. } => TrieStorage::Louds {
                louds: R::default(),
                is_link: R::default(),
                next_link: UintVector::new(),
                label_data: FastVec::new(),
                core_data: FastVec::new(),
                next_trie: None,
            },
            TrieStrategy::CompressedSparse { .. } => {
                TrieStorage::CompressedSparse(crate::fsa::cspp_trie::CsppTrie::new(4))
            }
        }
    }

    /// Get the root state
    #[inline]
    pub fn root(&self) -> StateId {
        self.root_state
    }

    /// Get performance statistics
    pub fn stats(&self) -> TrieStats {
        // Return a copy with updated statistics
        let mut stats = self.stats.clone();

        // Update memory usage
        stats.memory_usage = self.memory_usage();

        // Update bits per key
        if stats.num_keys > 0 {
            stats.bits_per_key = (stats.memory_usage as f64 * 8.0) / stats.num_keys as f64;
        } else {
            stats.bits_per_key = 0.0;
        }

        // Update number of states based on storage type
        // Special case: empty trie should report 0 states
        stats.num_states = if stats.num_keys == 0 {
            0
        } else {
            match &self.storage {
                TrieStorage::Patricia { nodes, .. } => nodes.len(),
                TrieStorage::CriticalBit { nodes, .. } => nodes.len(),
                TrieStorage::DoubleArray { check, .. } => {
                    // Count non-zero check values as active states
                    // But also count state 0 (root) which has check[0] = 0
                    1 + check.iter().skip(1).filter(|&&c| c != 0).count()
                }
                TrieStorage::Louds { .. } => 1, // TODO: implement for LOUDS
                TrieStorage::CompressedSparse(cspp) => cspp.total_states(),
            }
        };

        // Update number of transitions
        stats.num_transitions = match &self.storage {
            TrieStorage::Patricia { nodes, .. } => nodes.iter().map(|n| n.children.len()).sum(),
            TrieStorage::CriticalBit { .. } => 0, // TODO: implement
            TrieStorage::DoubleArray { base, check, .. } => {
                const STATE_MASK: u32 = 0x3FFF_FFFF;
                const TERMINAL_FLAG: u32 = 0x4000_0000;

                // Count transitions more efficiently:
                // Each non-zero check value represents a transition TO that state
                // (except for root which has check[0] = 0)
                let mut transition_count = 0;

                for i in 1..check.len() {
                    let check_val = check[i];
                    // If check is non-zero, this state has a parent (there's a transition to it)
                    if check_val != 0 {
                        // Special handling for root's children
                        if (check_val & STATE_MASK) == 0 {
                            // This is a child of root - only count if it's properly initialized
                            if (check_val & TERMINAL_FLAG) != 0 || (i < base.len() && base[i] != 0)
                            {
                                transition_count += 1;
                            }
                        } else {
                            // Regular transition
                            transition_count += 1;
                        }
                    }
                }

                transition_count
            }
            TrieStorage::Louds { .. } => 0, // TODO: implement
            TrieStorage::CompressedSparse(_cspp) => 0, /* TODO: implement num_transitions */
        };

        stats
    }

    /// Get the current configuration
    pub fn config(&self) -> &ZiporaTrieConfig {
        &self.config
    }

    /// Check if the trie is using cache optimization
    pub fn is_cache_optimized(&self) -> bool {
        self.cache_allocator.is_some()
    }

    /// Get number of states in the trie
    pub fn state_count(&self) -> usize {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => nodes.len(),
            TrieStorage::CriticalBit { nodes, .. } => nodes.len(),
            TrieStorage::DoubleArray { state_count, .. } => *state_count,
            TrieStorage::Louds { label_data, .. } => label_data.len(),
            TrieStorage::CompressedSparse(cspp) => cspp.total_states(),
        }
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Special case: empty trie should report 0 memory usage
        // even though it has a root state (structural overhead)
        if self.stats.num_keys == 0 {
            return 0;
        }

        match &self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data,
                compressed_paths,
            } => {
                nodes.capacity() * std::mem::size_of::<PatriciaNode>()
                    + edge_data.capacity()
                    + compressed_paths.capacity() * 64 // Rough estimate
            }
            TrieStorage::CriticalBit {
                nodes,
                keys,
                critical_cache,
            } => {
                nodes.capacity() * std::mem::size_of::<CritBitNode>()
                    + keys.capacity() * 32 // Rough estimate per key
                    + critical_cache.capacity() * 9 // usize + u8
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                // Use actual length instead of capacity for more accurate memory usage
                // Each element is 4 bytes (u32)
                base.len() * 4 + check.len() * 4
            }
            TrieStorage::Louds {
                label_data,
                core_data,
                ..
            } => {
                label_data.capacity() + core_data.capacity() + 1024 // Rank/select overhead
            }
            TrieStorage::CompressedSparse(cspp) => cspp.total_states() * 4,
        }
    }

    /// Insert a key into the trie
    pub fn insert(&mut self, key: &[u8]) -> Result<()> {
        // Delegate to the trait method which has complete implementation for all storage types
        let _state_id = <Self as Trie>::insert(self, key)?;
        // Mark stats as dirty - lazy update on next stats() call
        self.stats_dirty = true;
        Ok(())
    }

    /// Check if the trie contains a key
    #[inline]
    pub fn contains(&self, key: &[u8]) -> bool {
        // Delegate to the trait method which has complete implementation for all storage types
        <Self as Trie>::contains(self, key)
    }

    /// Remove a key from the trie
    pub fn remove(&mut self, key: &[u8]) -> Result<bool> {
        match &mut self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data,
                compressed_paths,
            } => {
                let removed =
                    Self::remove_patricia_actual(nodes, edge_data, compressed_paths, key)?;
                if removed {
                    self.stats.num_keys = self.stats.num_keys.saturating_sub(1);
                    self.stats_dirty = true;
                }
                Ok(removed)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                // Remove by clearing TERMINAL_BIT on the final state
                let state = Self::lookup_node_id_double_array(base, check, key);
                if let Some(state_id) = state {
                    const TERMINAL_BIT: u32 = 0x8000_0000;
                    base[state_id as usize] &= !TERMINAL_BIT;
                    self.stats.num_keys = self.stats.num_keys.saturating_sub(1);
                    self.stats_dirty = true;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            _ => Ok(false),
        }
    }

    /// Get the number of keys in the trie
    #[inline]
    pub fn len(&self) -> usize {
        self.stats.num_keys
    }

    /// Check if the trie is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all keys in the trie
    pub fn keys(&self) -> Vec<Vec<u8>> {
        match &self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data: _,
                compressed_paths,
            } => Self::keys_patricia_actual(nodes, compressed_paths),
            TrieStorage::Louds { label_data, .. } => Self::keys_louds_actual(label_data),
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::keys_double_array_actual(base, check)
            }
            TrieStorage::CompressedSparse(_cspp) => Vec::new(), // Handled by _cspp.iter
            _ => {
                // TODO: Implement for other storage types
                Vec::new()
            }
        }
    }

    /// Get all keys with a given prefix
    pub fn keys_with_prefix(&self, prefix: &[u8]) -> Vec<Vec<u8>> {
        match &self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data: _,
                compressed_paths,
            } => Self::keys_with_prefix_patricia_actual(nodes, compressed_paths, prefix),
            TrieStorage::Louds { label_data, .. } => {
                Self::keys_with_prefix_louds_actual(label_data, prefix)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::keys_with_prefix_double_array_actual(base, check, prefix)
            }
            TrieStorage::CompressedSparse(_cspp) => Vec::new(), // Handled by _cspp.iter
            _ => {
                // TODO: Implement for other storage types
                Vec::new()
            }
        }
    }

    /// Iterate over all keys in the trie
    pub fn iter_all(&self) -> TrieIterator {
        let keys = self.keys();
        TrieIterator::with_keys(keys)
    }

    /// Iterate over keys with a given prefix
    pub fn iter_prefix(&self, prefix: &[u8]) -> TrieIterator {
        let keys = self.keys_with_prefix(prefix);
        TrieIterator::with_keys(keys)
    }

    /// Get capacity (maximum number of states)
    pub fn capacity(&self) -> usize {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                // Patricia trie capacity is number of nodes * growth headroom
                nodes.capacity().max(nodes.len() * 2)
            }
            TrieStorage::CriticalBit { nodes, .. } => nodes.capacity().max(nodes.len() * 2),
            TrieStorage::DoubleArray { base, .. } => {
                // Double array capacity is the size of the base array
                base.capacity().max(base.len())
            }
            TrieStorage::Louds { label_data, .. } => {
                // LOUDS capacity based on label data size
                label_data.capacity().max(label_data.len() * 2)
            }
            TrieStorage::CompressedSparse(cspp) => cspp.total_states() * 4,
        }
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        match &self.storage {
            TrieStorage::DoubleArray { base, check, .. } => {
                let base_memory = base.capacity() * std::mem::size_of::<u32>();
                let check_memory = check.capacity() * std::mem::size_of::<u32>();
                (base_memory, check_memory, 0)
            }
            _ => {
                let total_memory = self.memory_usage();
                (total_memory / 2, total_memory / 2, 0)
            }
        }
    }

    /// Insert and get node ID
    pub fn insert_and_get_node_id(&mut self, key: &[u8]) -> Result<StateId> {
        match &mut self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data,
                compressed_paths,
            } => {
                let node_id = Self::insert_patricia_actual(
                    nodes,
                    edge_data,
                    compressed_paths,
                    key,
                    &mut self.stats.num_keys,
                )?;
                Ok(node_id)
            }
            TrieStorage::Louds {
                louds,
                is_link,
                next_link,
                label_data,
                core_data,
                next_trie,
            } => {
                let node_id = Self::insert_louds(
                    louds, is_link, next_link, label_data, core_data, next_trie, key,
                )?;
                self.stats.num_keys += 1;
                Ok(node_id)
            }
            TrieStorage::DoubleArray {
                base,
                check,
                free_list,
                state_count,
            } => {
                // insert_double_array handles num_keys internally (checks was_new)
                let node_id = Self::insert_double_array(
                    base,
                    check,
                    free_list,
                    state_count,
                    key,
                    &mut self.stats.num_keys,
                )?;
                self.stats_dirty = true;
                Ok(node_id)
            }
            _ => {
                self.stats.num_keys += 1;
                Ok(0)
            }
        }
    }

    /// Lookup node ID for a key
    pub fn lookup_node_id(&self, key: &[u8]) -> Option<StateId> {
        match &self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data,
                compressed_paths,
            } => Self::lookup_node_id_patricia_actual(nodes, edge_data, compressed_paths, key),
            TrieStorage::Louds { .. } => None,
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::lookup_node_id_double_array(base, check, key)
            }
            _ => None,
        }
    }

    /// Lookup node ID in DoubleArray storage
    fn lookup_node_id_double_array(
        base: &FastVec<u32>,
        check: &FastVec<u32>,
        key: &[u8],
    ) -> Option<StateId> {
        const TERMINAL_BIT: u32 = 0x8000_0000;
        const VALUE_MASK: u32 = 0x7FFF_FFFF;
        const FREE_BIT: u32 = 0x8000_0000;

        if base.is_empty() {
            return None;
        }

        let mut current_state = 0u32;

        if key.is_empty() {
            let base_val = base[0];
            return if (base_val & TERMINAL_BIT) != 0 {
                Some(0)
            } else {
                None
            };
        }

        for &symbol in key {
            let base_value = base[current_state as usize] & VALUE_MASK;
            let next_state = base_value.saturating_add(symbol as u32);

            if next_state as usize >= check.len() {
                return None;
            }

            let check_val = check[next_state as usize];
            let is_free = (check_val & FREE_BIT) != 0;
            if is_free || check_val != current_state {
                return None;
            }

            current_state = next_state;
        }

        // Only return state if it's marked terminal
        let base_val = base[current_state as usize];
        if (base_val & TERMINAL_BIT) != 0 {
            Some(current_state)
        } else {
            None
        }
    }

    /// Restore string from state ID
    pub fn restore_string(&self, state_id: StateId) -> Option<Vec<u8>> {
        match &self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data,
                compressed_paths,
            } => Self::restore_string_patricia_actual(nodes, edge_data, compressed_paths, state_id),
            TrieStorage::Louds { label_data, .. } => {
                Self::restore_string_louds(label_data, state_id)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::restore_string_double_array(base, check, state_id)
            }
            _ => None,
        }
    }

    /// Restore string from DoubleArray state by walking parent chain
    fn restore_string_double_array(
        base: &FastVec<u32>,
        check: &FastVec<u32>,
        state_id: StateId,
    ) -> Option<Vec<u8>> {
        const VALUE_MASK: u32 = 0x7FFF_FFFF;
        const FREE_BIT: u32 = 0x8000_0000;

        if state_id as usize >= check.len() {
            return None;
        }

        // Walk parent chain from state_id back to root, collecting symbols
        let mut symbols = Vec::new();
        let mut current = state_id;

        while current != 0 {
            let check_val = check[current as usize];
            if (check_val & FREE_BIT) != 0 {
                return None; // Free state, invalid
            }
            let parent = check_val; // parent state
            let parent_base = base[parent as usize] & VALUE_MASK;

            // The symbol is: current - parent_base
            if current < parent_base {
                return None; // Invalid state
            }
            let symbol = (current - parent_base) as u8;
            symbols.push(symbol);
            current = parent;
        }

        symbols.reverse();
        Some(symbols)
    }

    /// Check if a state is free (for DoubleArray)
    pub fn is_free_double_array(&self, state: StateId) -> bool {
        match &self.storage {
            TrieStorage::DoubleArray { check, .. } => {
                const FREE_BIT: u32 = 0x8000_0000; // Bit 31 in check for free states (referenced project)

                // Special case: root (state 0) is never free
                if state == 0 {
                    return false;
                }

                // A state is free if it's out of bounds or has FREE_BIT set
                if (state as usize) >= check.len() {
                    return true; // Out of bounds states are considered free
                }

                // Check the FREE_BIT (referenced project line 33: is_free)
                (check[state as usize] & FREE_BIT) != 0
            }
            _ => false,
        }
    }

    /// Get parent state (for DoubleArray)
    pub fn get_parent_double_array(&self, state: StateId) -> StateId {
        match &self.storage {
            TrieStorage::DoubleArray { check, .. } => {
                const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for parent value
                if (state as usize) < check.len() {
                    check[state as usize] & VALUE_MASK
                } else {
                    0 // Default to root
                }
            }
            _ => 0,
        }
    }

    /// Get base value (for DoubleArray)
    pub fn get_base_double_array(&self, state: StateId) -> u32 {
        match &self.storage {
            TrieStorage::DoubleArray { base, .. } => {
                const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for base value
                if (state as usize) < base.len() {
                    base[state as usize] & VALUE_MASK
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    /// Get check value (for DoubleArray)
    pub fn get_check_double_array(&self, state: StateId) -> u32 {
        match &self.storage {
            TrieStorage::DoubleArray { check, .. } => {
                const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for parent value
                if (state as usize) < check.len() {
                    check[state as usize] & VALUE_MASK
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    /// Shrink arrays to fit (for DoubleArray)
    pub fn shrink_to_fit(&mut self) {
        if let TrieStorage::DoubleArray { base, check, .. } = &mut self.storage {
            // Find the actual used length by scanning from the end
            // Skip trailing unused entries (check == 0 and base == 0)
            let mut actual_len = base.len();

            // Find the last used position
            while actual_len > 1 {
                let idx = actual_len - 1;
                // A state is used if either check is non-zero or base is non-zero
                // (state 0 is always used as root)
                if check[idx] != 0 || base[idx] != 0 {
                    break;
                }
                actual_len -= 1;
            }

            // Set unused bases to 1 (referenced project line 354-355)
            const NIL_STATE: u32 = 0x7FFF_FFFF;
            const VALUE_MASK: u32 = 0x7FFF_FFFF;
            for i in 0..actual_len {
                let base_val = base[i] & VALUE_MASK;
                if base_val == NIL_STATE {
                    base[i] = (base[i] & !VALUE_MASK) | 1; // Keep terminal bit, set base to 1
                }
            }

            // Truncate to exact used length (referenced project: exact sizing)
            if actual_len < base.len() {
                let _ = base.resize(actual_len, 0).ok();
                let _ = check.resize(actual_len, 0).ok();
            }

            // Shrink capacity to size (referenced project: minimal memory)
            let _ = base.shrink_to_fit();
            let _ = check.shrink_to_fit();
        }
    }

    // Helper method to restore string from LOUDS storage
    fn restore_string_louds(label_data: &FastVec<u8>, state_id: StateId) -> Option<Vec<u8>> {
        let start_pos = state_id as usize;
        if start_pos >= label_data.len() {
            return None;
        }

        // Read until we hit a null terminator
        let mut key = Vec::new();
        for i in start_pos..label_data.len() {
            if label_data[i] == 0 {
                break;
            }
            key.push(label_data[i]);
        }

        if key.is_empty() { None } else { Some(key) }
    }
}

/// Iterator for trie keys
pub struct TrieIterator {
    keys: Vec<Vec<u8>>,
    index: usize,
}

impl Default for TrieIterator {
    fn default() -> Self {
        Self::new()
    }
}

impl TrieIterator {
    pub fn new() -> Self {
        TrieIterator {
            keys: Vec::new(),
            index: 0,
        }
    }

    pub fn with_keys(keys: Vec<Vec<u8>>) -> Self {
        TrieIterator { keys, index: 0 }
    }
}

impl Iterator for TrieIterator {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.keys.len() {
            let key = self.keys[self.index].clone();
            self.index += 1;
            Some(key)
        } else {
            None
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_bytes: usize,
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
}

// Add Clone implementation for ZiporaTrie
impl<R> Clone for ZiporaTrie<R>
where
    R: RankSelectOps + Default + Clone,
{
    fn clone(&self) -> Self {
        // Create a new trie with the same config
        let mut new_trie = Self::with_config(self.config.clone());

        // Copy all keys from the original trie
        let keys = self.keys();
        for key in keys {
            let _ = new_trie.insert(&key);
        }

        // Copy statistics
        new_trie.stats = self.stats.clone();

        new_trie
    }
}

impl<R> Trie for ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    fn insert(&mut self, key: &[u8]) -> Result<StateId> {
        // Track if this was a new key insertion
        let result = match &mut self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data,
                compressed_paths,
            } => Self::insert_patricia(
                nodes,
                edge_data,
                compressed_paths,
                key,
                &mut self.stats.num_keys,
            ),
            TrieStorage::CriticalBit {
                nodes,
                keys,
                critical_cache,
            } => Self::insert_critical_bit(nodes, keys, critical_cache, key),
            TrieStorage::DoubleArray {
                base,
                check,
                free_list,
                state_count,
            } => Self::insert_double_array(
                base,
                check,
                free_list,
                state_count,
                key,
                &mut self.stats.num_keys,
            ),
            TrieStorage::Louds {
                louds,
                is_link,
                next_link,
                label_data,
                core_data,
                next_trie,
            } => Self::insert_louds(
                louds, is_link, next_link, label_data, core_data, next_trie, key,
            ),
            TrieStorage::CompressedSparse(cspp) => {
                let (is_new, _) = cspp.insert(key);
                if is_new {
                    self.stats.num_keys += 1;
                }
                Ok(0)
            }
        }?;

        Ok(result)
    }

    fn contains(&self, key: &[u8]) -> bool {
        match &self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data,
                compressed_paths,
            } => self.contains_patricia(nodes, edge_data, compressed_paths, key),
            TrieStorage::CriticalBit {
                nodes,
                keys,
                critical_cache,
            } => self.contains_critical_bit(nodes, keys, critical_cache, key),
            TrieStorage::DoubleArray { base, check, .. } => {
                self.contains_double_array(base, check, key)
            }
            TrieStorage::Louds {
                louds,
                is_link,
                next_link,
                label_data,
                core_data,
                next_trie,
            } => self.contains_louds(
                louds, is_link, next_link, label_data, core_data, next_trie, key,
            ),
            TrieStorage::CompressedSparse(cspp) => cspp.contains(key),
        }
    }

    fn len(&self) -> usize {
        self.stats.num_keys
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<R> FiniteStateAutomaton for ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    fn root(&self) -> StateId {
        self.root_state
    }

    fn is_final(&self, state: StateId) -> bool {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => nodes
                .get(state as usize)
                .map(|n| n.is_final)
                .unwrap_or(false),
            TrieStorage::CriticalBit { nodes, .. } => nodes
                .get(state as usize)
                .map(|n| n.is_final)
                .unwrap_or(false),
            TrieStorage::DoubleArray { base, .. } => {
                // Check the terminal bit in the BASE array (referenced project line 32: is_term)
                const TERMINAL_BIT: u32 = 0x8000_0000;
                base.get(state as usize)
                    .map(|b| (b & TERMINAL_BIT) != 0)
                    .unwrap_or(false)
            }
            TrieStorage::Louds { .. } => {
                // TODO: Implement LOUDS final state check
                false
            }
            TrieStorage::CompressedSparse(_cspp) => false, // Stub for legacy method
        }
    }

    fn transition(&self, state: StateId, symbol: u8) -> Option<StateId> {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                let node = nodes.get(state as usize)?;
                node.children
                    .binary_search_by_key(&symbol, |(s, _)| *s)
                    .ok()
                    .map(|idx| node.children[idx].1)
            }
            TrieStorage::CriticalBit { nodes: _, .. } => {
                // TODO: Implement critical bit transition
                None
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                // Double array trie transition: next = (base[state] & VALUE_MASK) + symbol
                // Validate with: check[next] == state (referenced project line 100-110)
                const VALUE_MASK: u32 = 0x7FFF_FFFF;

                let base_value = base.get(state as usize)? & VALUE_MASK;
                let next_state = base_value.saturating_add(symbol as u32);
                if let Some(check_value) = check.get(next_state as usize) {
                    if *check_value == state {
                        Some(next_state)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            TrieStorage::Louds { .. } => {
                // TODO: Implement LOUDS transition
                None
            }
            TrieStorage::CompressedSparse(_cspp) => None, // Stub for legacy method
        }
    }

    fn transitions(&self, state: StateId) -> Vec<(u8, StateId)> {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                if let Some(node) = nodes.get(state as usize) {
                    // Compact children representation - already in the right format
                    node.children.clone()
                } else {
                    Vec::new()
                }
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                let Some(&base_val) = base.get(state as usize) else {
                    return Vec::new();
                };
                if base_val == 0 {
                    return Vec::new();
                }

                const STATE_MASK: u32 = 0x3FFF_FFFF;
                const TERMINAL_FLAG: u32 = 0x4000_0000;

                (0u8..=255u8)
                    .filter_map(|symbol| {
                        let next_state = base_val.saturating_add(symbol as u32);
                        if (next_state as usize) >= check.len() {
                            return None;
                        }
                        let check_val = check[next_state as usize];
                        let is_valid_child = if state == 0 {
                            (check_val & STATE_MASK) == 0
                                && ((check_val & TERMINAL_FLAG) != 0
                                    || ((next_state as usize) < base.len()
                                        && base[next_state as usize] != 0))
                        } else {
                            check_val != 0 && (check_val & STATE_MASK) == state
                        };
                        if is_valid_child {
                            Some((symbol, next_state))
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    }
}

impl<R> PrefixIterable for ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    fn iter_prefix(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = Vec<u8>> + '_> {
        Box::new(self.iter_prefix(prefix))
    }

    fn iter_all(&self) -> Box<dyn Iterator<Item = Vec<u8>> + '_> {
        Box::new(self.iter_all())
    }
}

impl<R> Default for ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

// Implementation methods for different strategies
impl<R> ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    // Patricia trie implementation methods
    fn insert_patricia(
        nodes: &mut FastVec<PatriciaNode>,
        edge_data: &mut FastVec<u8>,
        compressed_paths: &mut HashMap<StateId, Vec<u8>>,
        key: &[u8],
        num_keys: &mut usize,
    ) -> Result<StateId> {
        Self::insert_patricia_actual(nodes, edge_data, compressed_paths, key, num_keys)
    }

    fn contains_patricia(
        &self,
        nodes: &FastVec<PatriciaNode>,
        edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> bool {
        Self::contains_patricia_actual(nodes, edge_data, compressed_paths, key)
    }

    // Critical-bit trie implementation methods
    //  TODO: port from C++ reference `src/terark/fsa/crit_bit_trie.hpp`
    fn insert_critical_bit(
        _nodes: &mut FastVec<CritBitNode>,
        _keys: &mut FastVec<Vec<u8>>,
        _critical_cache: &mut HashMap<usize, u8>,
        _key: &[u8],
    ) -> Result<StateId> {
        Err(ZiporaError::not_supported(
            "CriticalBit trie strategy is not yet implemented",
        ))
    }

    fn contains_critical_bit(
        &self,
        _nodes: &FastVec<CritBitNode>,
        _keys: &FastVec<Vec<u8>>,
        _critical_cache: &HashMap<usize, u8>,
        _key: &[u8],
    ) -> bool {
        false
    }

    // Double array trie implementation methods
    fn insert_double_array(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        _free_list: &mut VecDeque<StateId>,
        state_count: &mut usize,
        key: &[u8],
        num_keys: &mut usize,
    ) -> Result<StateId> {
        // Following referenced project's double array trie implementation EXACTLY
        // Base array (m_child0): bits 0-30 = base value, bit 31 = terminal bit
        // Check array (m_parent): bits 0-30 = parent state, bit 31 = free bit

        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal states (referenced project)
        const FREE_BIT: u32 = 0x8000_0000; // Bit 31 in check for free states (referenced project)
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for actual values (referenced project)
        const MAX_STATE: u32 = 0x7FFF_FFFE; // Maximum valid state value (referenced project)
        const NIL_STATE: u32 = 0x7FFF_FFFF; // Nil state marker (referenced project)

        // Ensure we have at least the root state
        // Referenced project starts with 1 state (line 70: states.resize(1))
        // We initialize in storage creation, but check here for safety
        if base.is_empty() {
            let _ = base.resize(1, NIL_STATE); // Just root state
            let _ = check.resize(1, 0); // Root check is 0 (itself), no free bit
            // Use compact base allocation like referenced project
            base[0] = Self::find_free_base(base, check, 0)?;
            *state_count = 1;
        }

        // Special case for empty key - mark root as terminal
        if key.is_empty() {
            let was_new = (base[0] & TERMINAL_BIT) == 0;
            base[0] |= TERMINAL_BIT;
            if was_new {
                *num_keys += 1;
            }
            return Ok(0);
        }

        let mut current_state = 0u32;

        #[cfg(debug_assertions)]
        eprintln!(
            "DEBUG insert: Starting insertion of key: {:?}",
            std::str::from_utf8(key).unwrap_or("<non-utf8>")
        );

        // Traverse the trie for each symbol in the key
        for (_pos, &symbol) in key.iter().enumerate() {
            #[cfg(debug_assertions)]
            let pos = _pos;
            // Calculate next state position using base value (bits 0-30)
            let mut base_value = base[current_state as usize] & VALUE_MASK;

            // If base is NIL_STATE, we need to find a good base for this state's children
            // Referenced project does this during build (lines 309-327)
            if base_value == NIL_STATE {
                base_value = Self::find_free_base(base, check, current_state)?;
                // CRITICAL: Preserve terminal bit when setting new base
                let old_val = base[current_state as usize];
                base[current_state as usize] = base_value | (old_val & TERMINAL_BIT);
            }

            let next_state = base_value.saturating_add(symbol as u32);

            // Expand arrays if needed - use amortized growth
            let required = next_state as usize + 1;
            if required > base.len() {
                let new_size = required.max(base.len() * 3 / 2).max(256);
                let _ = base.resize(new_size, NIL_STATE);
                let _ = check.resize(new_size, NIL_STATE | FREE_BIT);
            }

            // Check if this transition already exists (referenced project style at line 106)
            // A transition exists if check[next] == current_state (without free bit)
            // Free states have FREE_BIT set, so won't match
            let check_val = check[next_state as usize];
            let is_free = (check_val & FREE_BIT) != 0;
            let transition_exists = !is_free && check_val == current_state;

            if transition_exists {
                // Transition exists, follow it
                #[cfg(debug_assertions)]
                eprintln!(
                    "  [{}] '{:02x}' state {} -> {} (existing)",
                    pos, symbol, current_state, next_state
                );
                current_state = next_state;
            } else {
                // Need to create new transition
                // CRITICAL: Never allow transitions to state 0 (reserved for root)
                if next_state == 0 {
                    // State 0 is reserved, need to relocate
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "  [{}] '{:02x}' conflict: next_state would be 0 (reserved for root)",
                        pos, symbol
                    );

                    // We must relocate ALL children of current_state to maintain consistency
                    let new_base =
                        Self::relocate_state(base, check, current_state, symbol, state_count)?;

                    // Now the transition should be available at the new location
                    let new_next = new_base.saturating_add(symbol as u32);

                    // Expand if needed - use amortized growth
                    let required = new_next as usize + 1;
                    if required > base.len() {
                        let new_size = required.max(base.len() * 3 / 2).max(256);
                        let _ = base.resize(new_size, NIL_STATE);
                        let _ = check.resize(new_size, NIL_STATE | FREE_BIT);
                    }

                    // Allocate the state (referenced project: set_parent clears free bit)
                    check[new_next as usize] = current_state; // No free bit
                    // Initialize base to NIL_STATE - will be set when children are added
                    // (referenced project line 354-355: set to 1 for unused states)
                    base[new_next as usize] = NIL_STATE;
                    current_state = new_next;
                    *state_count += 1;
                } else if is_free {
                    // Position is free and not state 0, use it directly
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "  [{}] '{:02x}' state {} -> {} (new, free)",
                        pos, symbol, current_state, next_state
                    );

                    // Ensure the parent state fits within VALUE_MASK
                    if current_state > MAX_STATE {
                        return Err(ZiporaError::invalid_data("State value exceeds maximum"));
                    }
                    // Allocate the state (referenced project: set_parent clears free bit)
                    check[next_state as usize] = current_state; // Clear free bit by assignment
                    // Initialize base to NIL_STATE - will be set when children are added
                    // (referenced project line 354-355: set to 1 for unused states)
                    base[next_state as usize] = NIL_STATE;
                    current_state = next_state;
                    *state_count += 1;
                } else {
                    // Position is occupied - need to relocate
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "  [{}] '{:02x}' conflict at state {}, next_state {} already has check={:08x}",
                        pos, symbol, current_state, next_state, check[next_state as usize]
                    );
                    // We must relocate ALL children of current_state to maintain consistency
                    let new_base =
                        Self::relocate_state(base, check, current_state, symbol, state_count)?;

                    // Now the transition should be available
                    let new_next = new_base.saturating_add(symbol as u32);

                    #[cfg(debug_assertions)]
                    eprintln!(
                        "  Relocated state {} to new_base {}, new transition {} -> {}",
                        current_state, new_base, current_state, new_next
                    );

                    // Expand if needed - use amortized growth
                    let required = new_next as usize + 1;
                    if required > base.len() {
                        let new_size = required.max(base.len() * 3 / 2).max(256);
                        let _ = base.resize(new_size, NIL_STATE);
                        let _ = check.resize(new_size, NIL_STATE | FREE_BIT);
                    }

                    // Ensure the parent state fits within VALUE_MASK
                    if current_state > MAX_STATE {
                        return Err(ZiporaError::invalid_data(
                            "State value exceeds maximum during relocation",
                        ));
                    }
                    // Allocate the state (referenced project: set_parent clears free bit)
                    check[new_next as usize] = current_state; // No free bit
                    // Initialize base to NIL_STATE - will be set when children are added
                    // (referenced project line 354-355: set to 1 for unused states)
                    base[new_next as usize] = NIL_STATE;
                    current_state = new_next;
                    *state_count += 1;
                }
            }

        }

        // Mark the final state as terminal (referenced project: set_term_bit on base at line 27)
        // Check if this is a new key or duplicate
        let was_new = (base[current_state as usize] & TERMINAL_BIT) == 0;
        base[current_state as usize] |= TERMINAL_BIT;

        // Only increment key count if this was a new key
        if was_new {
            *num_keys += 1;
        }

        // Debug: Verify what we just inserted
        #[cfg(debug_assertions)]
        {
            eprintln!(
                "DEBUG insert_double_array: Inserted key, final state={}, base[{}]={:08x}, check[{}]={:08x}, was_new={}",
                current_state,
                current_state,
                base[current_state as usize],
                current_state,
                check[current_state as usize],
                was_new
            );
        }

        Ok(current_state)
    }

    // Helper: Find a free base value for a state that doesn't conflict
    // For incremental insert, use a proper heuristic matching referenced project's approach
    fn find_free_base(_base: &FastVec<u32>, check: &FastVec<u32>, _state: u32) -> Result<u32> {
        const FREE_BIT: u32 = 0x8000_0000;
        const NIL_STATE: u32 = 0x7FFF_FFFF;

        // Start search from position 1 (0 is root)
        let mut candidate = 1u32;
        let len = check.len();

        // Linear probe for a free position (matching C++ reference heuristic)
        while (candidate as usize) < len {
            let check_val = check[candidate as usize];
            let is_free = check_val == (NIL_STATE | FREE_BIT) || (check_val & FREE_BIT) != 0;
            if is_free {
                return Ok(candidate);
            }
            candidate += 1;
        }

        // Past the end of array — return the next position (will trigger array growth)
        Ok(candidate)
    }

    // Helper: Relocate a state and all its children to use a new base value
    fn relocate_state(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        state: u32,
        new_symbol: u8,
        _state_count: &mut usize,
    ) -> Result<u32> {
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for values (referenced project)
        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal (referenced project)
        const FREE_BIT: u32 = 0x8000_0000; // Bit 31 in check for free (referenced project)
        const NIL_STATE: u32 = 0x7FFF_FFFF; // Match referenced project's nil_state

        // Special handling for root state - try to avoid relocating it
        if state == 0 {
            // For root, try to find a different base that works
            // This is critical because relocating root affects the entire trie
            #[cfg(debug_assertions)]
            eprintln!("  WARNING: Attempting to relocate root state - this may cause issues");
        }

        let old_base = base[state as usize] & VALUE_MASK;

        // Collect all existing children of this state with their base and terminal info
        let mut children = Vec::new();
        for symbol in 0u8..=255u8 {
            let child_pos = old_base.saturating_add(symbol as u32);
            if (child_pos as usize) < check.len() {
                let check_val = check[child_pos as usize];
                // Check if this is an allocated child (not free, parent matches)
                if (check_val & FREE_BIT) == 0 && check_val == state {
                    // This is a child of our state - save its info
                    let child_base = if (child_pos as usize) < base.len() {
                        base[child_pos as usize]
                    } else {
                        NIL_STATE
                    };
                    let is_terminal = (child_base & TERMINAL_BIT) != 0;
                    children.push((symbol, child_pos, child_base, is_terminal));
                }
            }
        }

        // Find a new base where we can place all children plus the new symbol
        // Use find_free_base to get a better starting point that spreads states out
        let initial_base = Self::find_free_base(base, check, state)?;
        let mut new_base = initial_base;
        let mut attempts = 0;
        const MAX_BASE: u32 = u32::MAX - 256; // Leave room for 256 symbols

        'search: loop {
            if attempts > 1_000_000 || new_base > MAX_BASE {
                return Err(ZiporaError::invalid_data(
                    "Cannot relocate state in double array",
                ));
            }
            attempts += 1;

            // Check if new_base works for the new symbol
            let new_pos = new_base.saturating_add(new_symbol as u32);

            // Ensure arrays are large enough
            let max_pos = children
                .iter()
                .map(|(sym, _, _, _)| new_base.saturating_add(*sym as u32))
                .chain(std::iter::once(new_pos))
                .max()
                .unwrap_or(new_pos);

            // Expand arrays if needed - use amortized growth
            let required = max_pos as usize + 1;
            if required > base.len() {
                let new_size = required.max(base.len() * 3 / 2).max(256);
                let _ = base.resize(new_size, NIL_STATE);
                let _ = check.resize(new_size, NIL_STATE | FREE_BIT);
            }

            // CRITICAL: Never allow any child to be relocated to state 0
            // Check if new position for new_symbol is free and not state 0
            let new_pos_check = check[new_pos as usize];
            let new_pos_is_free = (new_pos_check & FREE_BIT) != 0;
            if new_pos == 0 || !new_pos_is_free {
                // State 0 is reserved or position is occupied
                // Use smaller increment for denser packing
                new_base = new_base.saturating_add(1);
                continue 'search;
            }

            // Check if all children can be relocated (and none would go to state 0)
            for (symbol, _, _, _) in &children {
                let test_pos = new_base.saturating_add(*symbol as u32);
                let test_check = check[test_pos as usize];
                let test_is_free = (test_check & FREE_BIT) != 0;
                // CRITICAL: Reject if any child would be relocated to state 0
                if test_pos == 0 || !test_is_free {
                    // State 0 is reserved or position is occupied
                    new_base = new_base.saturating_add(1);
                    continue 'search;
                }
            }

            // Found a suitable new base - relocate all children
            // First, mark old positions as free
            for (_, old_pos, _, _) in &children {
                check[*old_pos as usize] = NIL_STATE | FREE_BIT;
                // Mark base as NIL to indicate it's free
                base[*old_pos as usize] = NIL_STATE;
            }

            // Then, set new positions with both check and base values
            for (symbol, old_pos, child_base_val, is_terminal) in &children {
                let new_child_pos = new_base.saturating_add(*symbol as u32);
                // Allocate the new position (referenced project: set_parent clears free bit)
                check[new_child_pos as usize] = state; // Parent state, no free bit
                // Set base value, preserving terminal bit if needed
                let base_value = child_base_val & VALUE_MASK;
                base[new_child_pos as usize] = if *is_terminal {
                    base_value | TERMINAL_BIT
                } else {
                    base_value
                };

                // CRITICAL: Update any grandchildren that point to the old child position
                // to point to the new child position
                if base_value != 0 && base_value != NIL_STATE {
                    Self::update_grandchildren_check_values(base, check, *old_pos, new_child_pos);
                }
            }

            // Update the base for this state (preserve terminal bit if state is terminal)
            let state_base = base[state as usize];
            let state_is_terminal = (state_base & TERMINAL_BIT) != 0;
            base[state as usize] = if state_is_terminal {
                new_base | TERMINAL_BIT
            } else {
                new_base
            };

            return Ok(new_base);
        }
    }

    // Helper function to update grandchildren when a child state is relocated
    fn update_grandchildren_check_values(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        old_parent_pos: u32,
        new_parent_pos: u32,
    ) {
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for values (referenced project)
        const FREE_BIT: u32 = 0x8000_0000; // Bit 31 in check for free (referenced project)

        // Get the base value of the relocated child to find its children
        if let Some(&child_base_raw) = base.get(new_parent_pos as usize) {
            let child_base = child_base_raw & VALUE_MASK;
            if child_base != 0 && child_base != 0x7FFF_FFFF {
                // Find all grandchildren that were pointing to the old parent position
                for symbol in 0u8..=255u8 {
                    let grandchild_pos = child_base.saturating_add(symbol as u32);
                    if (grandchild_pos as usize) < check.len() {
                        let check_val = check[grandchild_pos as usize];
                        // Check if it's allocated (not free) and points to old parent
                        if (check_val & FREE_BIT) == 0 && check_val == old_parent_pos {
                            // This grandchild was pointing to the old parent position
                            // Update it to point to the new parent position
                            check[grandchild_pos as usize] = new_parent_pos;
                        }
                    }
                }
            }
        }
    }

    fn contains_double_array(&self, base: &FastVec<u32>, check: &FastVec<u32>, key: &[u8]) -> bool {
        // Following referenced project's double array trie lookup (line 100-110)
        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal states
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for actual values

        if base.is_empty() {
            return false;
        }

        // Special case for empty key - check if root is terminal (check terminal bit in base)
        if key.is_empty() {
            return base
                .first()
                .map(|b| (b & TERMINAL_BIT) != 0)
                .unwrap_or(false);
        }

        let mut current_state = 0u32;

        // Traverse the trie for each symbol (referenced project line 100-110: state_move)
        for (_i, &symbol) in key.iter().enumerate() {
            #[cfg(debug_assertions)]
            let i = _i;
            // SAFETY: We check if base_val exists, then use it
            let base_val = match base.get(current_state as usize) {
                Some(val) => val,
                None => {
                    #[cfg(debug_assertions)]
                    eprintln!("DEBUG contains: No base for state {}", current_state);
                    return false;
                }
            };

            // Calculate next state using base value (bits 0-30)
            let next_state = (base_val & VALUE_MASK).saturating_add(symbol as u32);

            // Check if the transition is valid (referenced project line 106: states[next].parent() == curr)
            if next_state as usize >= check.len() {
                #[cfg(debug_assertions)]
                eprintln!(
                    "DEBUG contains: next_state {} >= check.len() {}",
                    next_state,
                    check.len()
                );
                return false;
            }

            let check_val = check[next_state as usize];
            // Direct comparison like referenced project: check[next] == current_state
            // Free states have FREE_BIT set, so won't match
            if check_val != current_state {
                // Invalid transition
                #[cfg(debug_assertions)]
                eprintln!(
                    "DEBUG contains: Invalid transition at pos {}, symbol {:02x}, state {}->{}, check[{}]={:08x}, expected parent {}",
                    i, symbol, current_state, next_state, next_state, check_val, current_state
                );
                return false;
            }

            current_state = next_state;

        }

        // Check if the final state is marked as terminal (check terminal bit in base)
        let is_terminal = base
            .get(current_state as usize)
            .map(|b| (b & TERMINAL_BIT) != 0)
            .unwrap_or(false);

        #[cfg(debug_assertions)]
        {
            let base_val = base.get(current_state as usize).unwrap_or(&0);
            let check_val = check.get(current_state as usize).unwrap_or(&0);
            eprintln!(
                "DEBUG contains: Final state={}, base[{}]={:08x}, check[{}]={:08x}, is_terminal={}",
                current_state, current_state, base_val, current_state, check_val, is_terminal
            );
        }

        is_terminal
    }

    // LOUDS trie implementation methods
    //  TODO: port from C++ reference `src/terark/fsa/nest_louds_trie.hpp`
    fn insert_louds(
        _louds: &mut R,
        _is_link: &mut R,
        _next_link: &mut UintVector,
        _label_data: &mut FastVec<u8>,
        _core_data: &mut FastVec<u8>,
        _next_trie: &mut Option<Box<ZiporaTrie<R>>>,
        _key: &[u8],
    ) -> Result<StateId> {
        Err(ZiporaError::not_supported(
            "LOUDS trie strategy is not yet implemented",
        ))
    }

    #[allow(clippy::too_many_arguments)] // internal helper; arg bundle would add indirection
    fn contains_louds(
        &self,
        _louds: &R,
        _is_link: &R,
        _next_link: &UintVector,
        _label_data: &FastVec<u8>,
        _core_data: &FastVec<u8>,
        _next_trie: &Option<Box<ZiporaTrie<R>>>,
        _key: &[u8],
    ) -> bool {
        false
    }

    // Actual implementation methods for Patricia trie
    fn insert_patricia_actual(
        nodes: &mut FastVec<PatriciaNode>,
        _edge_data: &mut FastVec<u8>,
        _compressed_paths: &mut HashMap<StateId, Vec<u8>>,
        key: &[u8],
        num_keys: &mut usize,
    ) -> Result<StateId> {
        if nodes.is_empty() {
            // Initialize with root node
            let _ = nodes.push(PatriciaNode::default());
        }

        let mut current = 0;
        let mut key_pos = 0;

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                // Follow existing path
                let child_id = node.children[idx].1;
                current = child_id as usize;
                key_pos += 1;
            } else {
                // Create new child node
                let new_node_id = nodes.len();
                let _ = nodes.push(PatriciaNode::default());

                // Insert into sorted children Vec
                let insert_pos = nodes[current]
                    .children
                    .binary_search_by_key(&symbol, |(s, _)| *s)
                    .unwrap_err();
                nodes[current]
                    .children
                    .insert(insert_pos, (symbol, new_node_id as StateId));

                current = new_node_id;
                key_pos += 1;
            }
        }

        // Mark current node as final (check if was_new)
        let was_new = !nodes[current].is_final;
        nodes[current].is_final = true;
        if was_new {
            *num_keys += 1;
        }
        Ok(current as StateId)
    }

    fn contains_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        _edge_data: &FastVec<u8>,
        _compressed_paths: &HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> bool {
        if nodes.is_empty() {
            return false;
        }

        let mut current = 0;
        let mut key_pos = 0;

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                current = child_id as usize;
                key_pos += 1;
            } else {
                return false;
            }
        }

        // Check if we've consumed the entire key and reached a final state
        key_pos == key.len() && nodes[current].is_final
    }

    fn remove_patricia_actual(
        nodes: &mut FastVec<PatriciaNode>,
        _edge_data: &mut FastVec<u8>,
        _compressed_paths: &mut HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> Result<bool> {
        if nodes.is_empty() {
            return Ok(false);
        }

        // First, check if the key exists and find the path to it
        let mut current = 0;
        let mut key_pos = 0;
        let mut path = Vec::new(); // Track the path for potential cleanup

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                path.push((current, symbol)); // Store parent and symbol for path
                current = child_id as usize;
                key_pos += 1;
            } else {
                // Key doesn't exist
                return Ok(false);
            }
        }

        // Check if we found a complete key at a final state
        if key_pos != key.len() || !nodes[current].is_final {
            return Ok(false);
        }

        // Mark the node as non-final (remove the key)
        nodes[current].is_final = false;

        // Check if this node has any children
        let has_children = !nodes[current].children.is_empty();

        // If the node has no children and is not final, we can potentially clean it up
        if !has_children {
            // Walk back up the path and remove unnecessary nodes
            for &(parent_idx, symbol) in path.iter().rev() {
                // Remove the child pointer from parent
                if let Ok(idx) = nodes[parent_idx]
                    .children
                    .binary_search_by_key(&symbol, |(s, _)| *s)
                {
                    nodes[parent_idx].children.remove(idx);
                }

                // Check if parent node should also be cleaned up
                let parent_has_children = !nodes[parent_idx].children.is_empty();
                let parent_is_final = nodes[parent_idx].is_final;

                // If parent has other children or is final, stop cleanup
                if parent_has_children || parent_is_final {
                    break;
                }
            }
        }

        Ok(true)
    }

    fn restore_string_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        _edge_data: &FastVec<u8>,
        _compressed_paths: &HashMap<StateId, Vec<u8>>,
        state_id: StateId,
    ) -> Option<Vec<u8>> {
        if nodes.is_empty() || state_id as usize >= nodes.len() {
            return None;
        }

        // Check if the target state is final
        if !nodes[state_id as usize].is_final {
            return None;
        }

        // Perform DFS to find the path from root to the target state
        let mut path = Vec::new();
        if Self::find_path_to_state(nodes, 0, state_id as usize, &mut path) {
            Some(path)
        } else {
            None
        }
    }

    fn find_path_to_state(
        nodes: &FastVec<PatriciaNode>,
        current: usize,
        target: usize,
        path: &mut Vec<u8>,
    ) -> bool {
        if current == target {
            return true;
        }

        if current >= nodes.len() {
            return false;
        }

        let node = &nodes[current];

        // Try each child (compact representation)
        for &(symbol, child_id) in node.children.iter() {
            let child_id = child_id as usize;

            // Add this symbol to the path
            path.push(symbol);

            // Recursively search in child
            if Self::find_path_to_state(nodes, child_id, target, path) {
                return true;
            }

            // Backtrack if not found
            path.pop();
        }

        false
    }

    fn lookup_node_id_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        _edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> Option<StateId> {
        if nodes.is_empty() {
            return None;
        }

        let mut current = 0;
        let mut key_pos = 0;

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                current = child_id as usize;
                key_pos += 1;

                // Check compressed path
                if let Some(path) = compressed_paths.get(&child_id) {
                    if key_pos + path.len() > key.len() {
                        return None; // Not enough key left
                    }
                    if &key[key_pos..key_pos + path.len()] != path.as_slice() {
                        return None; // Path doesn't match
                    }
                    key_pos += path.len();
                }
            } else {
                return None;
            }
        }

        // Check if we've consumed the entire key and reached a final state
        if key_pos == key.len() && nodes[current].is_final {
            Some(current as StateId)
        } else {
            None
        }
    }

    /// Get all keys from Patricia trie
    fn keys_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
    ) -> Vec<Vec<u8>> {
        if nodes.is_empty() {
            return Vec::new();
        }

        let mut keys = Vec::new();
        let mut current_path = Vec::new();
        Self::collect_keys_patricia_recursive(
            nodes,
            compressed_paths,
            0,
            &mut current_path,
            &mut keys,
        );
        keys
    }

    /// Get all keys with prefix from Patricia trie
    fn keys_with_prefix_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        prefix: &[u8],
    ) -> Vec<Vec<u8>> {
        if nodes.is_empty() {
            return Vec::new();
        }

        // Navigate to the prefix position first
        let mut current = 0;
        let mut key_pos = 0;
        let mut path_to_prefix = Vec::new();

        while key_pos < prefix.len() {
            let symbol = prefix[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                path_to_prefix.push(symbol);
                current = child_id as usize;
                key_pos += 1;

                // Check compressed path
                if let Some(path) = compressed_paths.get(&child_id) {
                    if key_pos + path.len() > prefix.len() {
                        // Prefix doesn't fully match this path
                        let remaining_prefix = &prefix[key_pos..];
                        if path.starts_with(remaining_prefix) {
                            // Prefix is a partial match of this compressed path
                            // Continue from this node with the partial prefix included
                            path_to_prefix.extend_from_slice(remaining_prefix);
                            break;
                        } else {
                            // Prefix doesn't match - no keys with this prefix
                            return Vec::new();
                        }
                    } else if &prefix[key_pos..key_pos + path.len()] != path.as_slice() {
                        // Path doesn't match prefix
                        return Vec::new();
                    } else {
                        // Path matches, continue
                        path_to_prefix.extend_from_slice(path);
                        key_pos += path.len();
                    }
                }
            } else {
                // No child for this symbol - no keys with this prefix
                return Vec::new();
            }
        }

        // Now collect all keys from this point
        let mut keys = Vec::new();
        let mut current_path = path_to_prefix;
        Self::collect_keys_patricia_recursive(
            nodes,
            compressed_paths,
            current,
            &mut current_path,
            &mut keys,
        );

        // Filter to only include keys that actually start with the prefix
        keys.into_iter()
            .filter(|key| key.starts_with(prefix))
            .collect()
    }

    /// Recursively collect all keys from Patricia trie
    fn collect_keys_patricia_recursive(
        nodes: &FastVec<PatriciaNode>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        node_id: usize,
        current_path: &mut Vec<u8>,
        keys: &mut Vec<Vec<u8>>,
    ) {
        if node_id >= nodes.len() {
            return;
        }

        let node = &nodes[node_id];

        // If this is a final node, add the current path as a key
        if node.is_final {
            keys.push(current_path.clone());
        }

        // Explore all children (compact representation)
        for &(symbol, child_id) in node.children.iter() {
            let child_id_usize = child_id as usize;

            // Add this symbol to the path
            current_path.push(symbol);

            // Add compressed path if it exists
            let path_start_len = current_path.len();
            if let Some(path) = compressed_paths.get(&child_id) {
                current_path.extend_from_slice(path);
            }

            // Recursively collect from child
            Self::collect_keys_patricia_recursive(
                nodes,
                compressed_paths,
                child_id_usize,
                current_path,
                keys,
            );

            // Backtrack: remove the path we added
            current_path.truncate(path_start_len - 1);
        }
    }

    /// Get all keys from LOUDS trie storage
    fn keys_louds_actual(label_data: &FastVec<u8>) -> Vec<Vec<u8>> {
        let mut keys = Vec::new();

        if label_data.is_empty() {
            return keys;
        }

        let mut current_key = Vec::new();

        for &byte in label_data.iter() {
            if byte == 0u8 {
                // Found separator, this completes a key
                if !current_key.is_empty() {
                    keys.push(current_key.clone());
                    current_key.clear();
                }
            } else {
                // Add byte to current key
                current_key.push(byte);
            }
        }

        // Handle last key if there's no trailing separator
        if !current_key.is_empty() {
            keys.push(current_key);
        }

        // Remove duplicates and sort
        keys.sort();
        keys.dedup();

        keys
    }

    /// Get all keys with a given prefix from LOUDS trie storage
    fn keys_with_prefix_louds_actual(label_data: &FastVec<u8>, prefix: &[u8]) -> Vec<Vec<u8>> {
        let all_keys = Self::keys_louds_actual(label_data);

        // Filter keys that start with the given prefix
        all_keys
            .into_iter()
            .filter(|key| key.starts_with(prefix))
            .collect()
    }

    /// Get all keys from DoubleArray trie storage
    fn keys_double_array_actual(base: &FastVec<u32>, check: &FastVec<u32>) -> Vec<Vec<u8>> {
        if base.is_empty() {
            return Vec::new();
        }

        let mut keys = Vec::new();
        let mut current_path = Vec::new();

        #[cfg(debug_assertions)]
        eprintln!(
            "DEBUG keys_double_array: Starting from root state 0, base[0]={:?}",
            base.first()
        );

        Self::collect_keys_double_array_recursive(base, check, 0, &mut current_path, &mut keys);
        keys
    }

    /// Get all keys with prefix from DoubleArray trie storage
    fn keys_with_prefix_double_array_actual(
        base: &FastVec<u32>,
        check: &FastVec<u32>,
        prefix: &[u8],
    ) -> Vec<Vec<u8>> {
        if base.is_empty() {
            return Vec::new();
        }

        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for values (referenced project)

        // Navigate to the prefix position first
        let mut current_state = 0u32;
        for &symbol in prefix {
            // SAFETY: We check if base_val exists, then use it
            let base_value = match base.get(current_state as usize) {
                Some(val) => val & VALUE_MASK,
                None => return Vec::new(),
            };
            let next_state = base_value.saturating_add(symbol as u32);
            if next_state as usize >= check.len() {
                return Vec::new();
            }

            let check_val = check[next_state as usize];
            // Direct comparison like referenced project (line 106)
            if check_val != current_state {
                return Vec::new();
            }

            current_state = next_state;
        }

        // Now collect all keys from this point
        let mut keys = Vec::new();
        let mut current_path = prefix.to_vec();
        Self::collect_keys_double_array_recursive(
            base,
            check,
            current_state,
            &mut current_path,
            &mut keys,
        );
        keys
    }

    /// Recursively collect all keys from DoubleArray trie
    fn collect_keys_double_array_recursive(
        base: &FastVec<u32>,
        check: &FastVec<u32>,
        state: u32,
        current_path: &mut Vec<u8>,
        keys: &mut Vec<Vec<u8>>,
    ) {
        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal (referenced project)
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for values (referenced project)

        #[cfg(debug_assertions)]
        if state == 0 && current_path.is_empty() {
            eprintln!("DEBUG collect_keys: At root, checking for children...");
        }

        // If this is a terminal state, add the current path as a key (check base array)
        if (state as usize) < base.len() && (base[state as usize] & TERMINAL_BIT) != 0 {
            #[cfg(debug_assertions)]
            eprintln!(
                "DEBUG collect_keys: Found terminal state {} with path {:?}",
                state,
                std::str::from_utf8(current_path).unwrap_or("<non-utf8>")
            );
            keys.push(current_path.clone());
        }

        // Get the base value for this state
        if let Some(&base_raw) = base.get(state as usize) {
            let base_val = base_raw & VALUE_MASK;
            if base_val == 0 || base_val == 0x7FFF_FFFF {
                #[cfg(debug_assertions)]
                eprintln!(
                    "DEBUG collect_keys: State {} has base={}, no children",
                    state, base_val
                );
                return; // No children
            }

            #[cfg(debug_assertions)]
            if state == 0 {
                eprintln!(
                    "DEBUG collect_keys: Root state 0 has base={}, checking all 256 symbols...",
                    base_val
                );
            }

            // Try all possible symbols
            for symbol in 0u8..=255u8 {
                let next_state = base_val.saturating_add(symbol as u32);

                // Check if this is a valid transition (referenced project line 106)
                if (next_state as usize) < check.len() {
                    let check_val = check[next_state as usize];
                    // Direct comparison: check[next] == current_state
                    let is_valid_child = check_val == state;

                    if is_valid_child {
                        // Valid transition found
                        #[cfg(debug_assertions)]
                        if state == 0 {
                            eprintln!(
                                "DEBUG collect_keys: Found valid transition from root: symbol={:02x} ('{}'), next_state={}",
                                symbol, symbol as char, next_state
                            );
                        }
                        current_path.push(symbol);
                        Self::collect_keys_double_array_recursive(
                            base,
                            check,
                            next_state,
                            current_path,
                            keys,
                        );
                        current_path.pop();
                    }
                }
            }
        }
    }

    /// Build a trie from sorted keys using BFS construction
    ///
    /// This is more efficient than incremental insertion for sorted input because:
    /// 1. Pre-allocates arrays based on estimated size
    /// 2. Processes keys in sorted order to minimize relocations
    /// 3. Uses improved find_free_base for better packing
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig};
    /// use zipora::succinct::RankSelectInterleaved256;
    ///
    /// let keys: Vec<&[u8]> = vec![b"apple", b"application", b"apply", b"banana", b"band"];
    /// let trie: ZiporaTrie<RankSelectInterleaved256> =
    ///     ZiporaTrie::build_from_sorted(&keys, ZiporaTrieConfig::default()).unwrap();
    ///
    /// assert_eq!(trie.len(), 5);
    /// assert!(trie.contains(b"apple"));
    /// assert!(trie.contains(b"banana"));
    /// ```
    pub fn build_from_sorted(keys: &[&[u8]], config: ZiporaTrieConfig) -> Result<Self> {
        // Create trie with config
        let mut trie = Self::with_config(config);

        // Estimate size and pre-allocate for DoubleArray strategy
        if let TrieStrategy::DoubleArray { .. } = &trie.config.trie_strategy
            && let TrieStorage::DoubleArray { base, check, .. } = &mut trie.storage
        {
            // Estimate: each key adds ~key_length states on average
            let estimated_states = keys.iter().map(|k| k.len()).sum::<usize>() / 2;
            let initial_size = estimated_states.max(256);

            const NIL_STATE: u32 = 0x7FFF_FFFF;
            const FREE_BIT: u32 = 0x8000_0000;

            let _ = base.resize(initial_size, NIL_STATE);
            let _ = check.resize(initial_size, NIL_STATE | FREE_BIT);
        }

        // Insert keys in sorted order
        // Sorted order tends to result in fewer relocations
        for &key in keys {
            trie.insert(key)?;
        }

        Ok(trie)
    }
}
