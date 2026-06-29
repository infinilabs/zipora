use super::config::{
    TrieStrategy,
    ZiporaTrieConfig,
};
use super::storage::{CritBitNode, PatriciaNode, TrieStorage};
use crate::StateId;
use crate::containers::FastVec;
use crate::containers::specialized::UintVector;
use crate::error::Result;
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
    pub(super) config: ZiporaTrieConfig,
    /// Internal storage implementation
    pub(super) storage: TrieStorage<R>,
    /// Performance statistics
    pub(super) stats: TrieStats,
    /// Track whether stats need recomputation
    pub(super) stats_dirty: bool,
    /// Cache optimization components
    pub(super) cache_allocator: Option<CacheOptimizedAllocator>,
    /// Memory pool for allocation
    pub(super) _memory_pool: Option<Arc<SecureMemoryPool>>,
    /// Root state for traversal
    pub(super) root_state: StateId,
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

