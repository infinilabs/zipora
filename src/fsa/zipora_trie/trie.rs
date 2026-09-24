use super::config::{
    TrieStrategy,
    ZiporaTrieConfig,
};
use super::storage::{CritBitNode, DaFreeList, PatriciaNode, TrieStorage};
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
use std::collections::HashMap;
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
    /// State moves (old_state, new_state) recorded by the most recent insert's
    /// double-array collision relocations; consumed by ZiporaTrieMap to keep
    /// state-indexed value slots in sync. Cleared at the start of each insert.
    pub(super) relocations: Vec<(u32, u32)>,
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
            relocations: Vec::new(),
        }
    }

    /// Create storage based on strategy configuration
    fn create_storage(config: &ZiporaTrieConfig) -> TrieStorage<R> {
        match &config.trie_strategy {
            TrieStrategy::Patricia { .. } => TrieStorage::Patricia {
                nodes: FastVec::new(),
                edge_data: FastVec::new(),
                compressed_paths: HashMap::new(),
                free_list: Vec::new(),
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
                    free_list: DaFreeList::new(),
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
        let cspp_counts = if stats.num_keys == 0 {
            None
        } else if let TrieStorage::CompressedSparse(cspp) = &self.storage {
            Some(Self::count_cspp_states_and_transitions(cspp))
        } else {
            None
        };

        stats.num_states = if stats.num_keys == 0 {
            0
        } else {
            match &self.storage {
                TrieStorage::Patricia { nodes, .. } => nodes.len(),
                TrieStorage::CriticalBit { nodes, .. } => nodes.len(),
                // Free cells hold non-zero link words, so counting cells by
                // value would count the whole array; the allocator's count is exact.
                TrieStorage::DoubleArray { state_count, .. } => *state_count,
                TrieStorage::Louds { label_data, .. } => label_data.len(),
                TrieStorage::CompressedSparse(_) => cspp_counts.map_or(0, |(s, _)| s),
            }
        };

        // Update number of transitions
        stats.num_transitions = match &self.storage {
            TrieStorage::Patricia { nodes, .. } => nodes.iter().map(|n| n.children.len()).sum(),
            TrieStorage::CriticalBit { nodes, .. } => nodes.len().saturating_sub(1),
            // Every state except the root has exactly one incoming transition.
            TrieStorage::DoubleArray { state_count, .. } => state_count.saturating_sub(1),
            TrieStorage::Louds { label_data, .. } => label_data.len().saturating_sub(1),
            TrieStorage::CompressedSparse(_) => cspp_counts.map_or(0, |(_, t)| t),
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
                ..
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
                free_list,
            } => {
                let removed = Self::remove_patricia_actual(
                    nodes,
                    edge_data,
                    compressed_paths,
                    free_list,
                    key,
                )?;
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
            TrieStorage::CompressedSparse(_)
            | TrieStorage::CriticalBit { .. }
            | TrieStorage::Louds { .. } => Err(crate::error::ZiporaError::not_supported(
                "remove is not supported for this trie strategy",
            )),
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
                ..
            } => Self::keys_patricia_actual(nodes, compressed_paths),
            TrieStorage::Louds { label_data, .. } => Self::keys_louds_actual(label_data),
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::keys_double_array_actual(base, check)
            }
            TrieStorage::CompressedSparse(cspp) => {
                let mut iter = crate::fsa::cspp_trie::CsppTrieIterator::<()>::new(cspp);
                let mut out = Vec::with_capacity(self.stats.num_keys);
                if iter.seek_begin() {
                    loop {
                        out.push(iter.word().to_vec());
                        if !iter.incr() {
                            break;
                        }
                    }
                }
                out
            }
            TrieStorage::CriticalBit { .. } => Vec::new(),
        }
    }

    /// Get all keys with a given prefix
    pub fn keys_with_prefix(&self, prefix: &[u8]) -> Vec<Vec<u8>> {
        match &self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data: _,
                compressed_paths,
                ..
            } => Self::keys_with_prefix_patricia_actual(nodes, compressed_paths, prefix),
            TrieStorage::Louds { label_data, .. } => {
                Self::keys_with_prefix_louds_actual(label_data, prefix)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::keys_with_prefix_double_array_actual(base, check, prefix)
            }
            TrieStorage::CompressedSparse(cspp) => {
                let mut iter = crate::fsa::cspp_trie::CsppTrieIterator::<()>::new(cspp);
                let mut out = Vec::new();
                if iter.seek_begin() {
                    loop {
                        let w = iter.word();
                        if w.starts_with(prefix) {
                            out.push(w.to_vec());
                        } else if !prefix.is_empty() && w > prefix {
                            break;
                        }
                        if !iter.incr() {
                            break;
                        }
                    }
                }
                out
            }
            TrieStorage::CriticalBit { .. } => Vec::new(),
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
        let node_id = <Self as Trie>::insert(self, key)?;
        self.stats_dirty = true;
        Ok(node_id)
    }

    /// Lookup node ID for a key
    pub fn lookup_node_id(&self, key: &[u8]) -> Option<StateId> {
        match &self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data,
                compressed_paths,
                ..
            } => Self::lookup_node_id_patricia_actual(nodes, edge_data, compressed_paths, key),
            TrieStorage::Louds { .. } => None,
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::lookup_node_id_double_array(base, check, key)
            }
            TrieStorage::CompressedSparse(cspp) => Self::lookup_node_id_cspp(cspp, key),
            TrieStorage::CriticalBit { .. } => None,
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
                ..
            } => Self::restore_string_patricia_actual(nodes, edge_data, compressed_paths, state_id),
            TrieStorage::Louds { label_data, .. } => {
                Self::restore_string_louds(label_data, state_id)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::restore_string_double_array(base, check, state_id)
            }
            TrieStorage::CompressedSparse(cspp) => Self::restore_string_cspp(cspp, state_id),
            TrieStorage::CriticalBit { .. } => None,
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

        if state_id as usize >= check.len() || state_id as usize >= base.len() {
            return None;
        }
        if state_id != 0 && (check[state_id as usize] & FREE_BIT) != 0 {
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
            if parent as usize >= base.len() || symbols.len() >= check.len() {
                return None;
            }
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
        if let TrieStorage::DoubleArray {
            base,
            check,
            free_list,
            ..
        } = &mut self.storage
        {
            const FREE_BIT: u32 = 0x8000_0000;

            // Keep everything up to the last occupied cell (root is always occupied).
            let mut actual_len = base.len();
            while actual_len > 1 && (check[actual_len - 1] & FREE_BIT) != 0 {
                actual_len -= 1;
            }

            // The dropped cells are all free; take them off the free list first
            // or the next allocation would follow a link past the end.
            for cell in actual_len..base.len() {
                free_list.unlink(base, check, cell as u32);
            }
            free_list.high_water = free_list.high_water.min(actual_len as u32);

            // Set unused bases to 1 (referenced project line 354-355).
            // Free cells are skipped: their `base` holds a free-list link.
            const NIL_STATE: u32 = 0x7FFF_FFFF;
            const VALUE_MASK: u32 = 0x7FFF_FFFF;
            for i in 0..actual_len {
                if (check[i] & FREE_BIT) != 0 {
                    continue;
                }
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

    pub(crate) const CSPP_MAX_SLOT: u32 = (1 << 24) - 1;

    #[inline(always)]
    pub(crate) fn encode_cspp_state(slot: u32, zprog: usize) -> Option<StateId> {
        if slot <= Self::CSPP_MAX_SLOT && zprog <= 255 {
            Some((slot << 8) | (zprog as u32))
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(crate) fn set_cspp_total_states_for_test(&mut self, slots: usize) {
        if let TrieStorage::CompressedSparse(cspp) = &mut self.storage {
            cspp.mempool.resize(
                slots,
                crate::fsa::cspp_trie::PatriciaNode { child: u32::MAX },
            );
        }
    }

    #[inline(always)]
    fn is_valid_cspp_node(cspp: &crate::fsa::cspp_trie::CsppTrie, target_slot: u32) -> bool {
        if target_slot > Self::CSPP_MAX_SLOT {
            return false;
        }
        if target_slot == crate::fsa::cspp_trie::INITIAL_STATE {
            return cspp.total_states() > 0;
        }
        if target_slot < 258 {
            return false;
        }
        cspp.node_view(target_slot).is_well_formed()
    }

    fn count_cspp_states_and_transitions(
        cspp: &crate::fsa::cspp_trie::CsppTrie,
    ) -> (usize, usize) {
        if cspp.total_states() == 0 {
            return (0, 0);
        }
        let mut states = 0usize;
        let mut transitions = 0usize;
        let mut stack = vec![crate::fsa::cspp_trie::INITIAL_STATE];
        while let Some(curr) = stack.pop() {
            if (curr as usize) >= cspp.total_states() {
                continue;
            }
            let view = cspp.node_view(curr);
            if !view.is_well_formed() {
                continue;
            }
            let zlen = view.zpath_len();
            states += 1 + zlen;
            transitions += zlen + view.n_children();
            view.for_each_child(|_, child_slot| {
                stack.push(child_slot);
            });
        }
        (states, transitions)
    }

    fn lookup_node_id_cspp(
        cspp: &crate::fsa::cspp_trie::CsppTrie,
        key: &[u8],
    ) -> Option<StateId> {
        let mut curr = crate::fsa::cspp_trie::INITIAL_STATE;
        let mut pos = 0usize;
        loop {
            if (curr as usize) >= cspp.total_states() {
                return None;
            }
            let view = cspp.node_view(curr);
            let zlen = view.zpath_len();
            if zlen > 0 {
                let zpath = view.zpath_slice();
                let rem = key.len() - pos;
                if rem < zlen || &key[pos..pos + zlen] != zpath {
                    return None;
                }
                pos += zlen;
            }
            if pos == key.len() {
                return if view.is_final() {
                    Self::encode_cspp_state(curr, zlen)
                } else {
                    None
                };
            }
            let next = view.state_move(key[pos]);
            if next == crate::fsa::cspp_trie::NIL_STATE {
                return None;
            }
            curr = next;
            pos += 1;
        }
    }

    fn restore_string_cspp(
        cspp: &crate::fsa::cspp_trie::CsppTrie,
        state_id: StateId,
    ) -> Option<Vec<u8>> {
        let target_slot = state_id >> 8;
        let target_zprog = (state_id & 0xFF) as usize;
        if !Self::is_valid_cspp_node(cspp, target_slot) {
            return None;
        }
        let target_view = cspp.node_view(target_slot);
        if target_zprog > target_view.zpath_len() {
            return None;
        }
        if target_slot == crate::fsa::cspp_trie::INITIAL_STATE {
            return Some(target_view.zpath_slice()[..target_zprog].to_vec());
        }

        // Iterative DFS with explicit path-truncation frames: `(curr, base_len, edge_byte)`.
        let mut path = Vec::new();
        let mut stack: Vec<(u32, usize, Option<u8>)> =
            vec![(crate::fsa::cspp_trie::INITIAL_STATE, 0, None)];
        while let Some((curr, base_len, edge_byte)) = stack.pop() {
            path.truncate(base_len);
            if let Some(ch) = edge_byte {
                path.push(ch);
            }
            if (curr as usize) >= cspp.total_states() {
                continue;
            }
            let view = cspp.node_view(curr);
            if !view.is_well_formed() {
                continue;
            }
            if curr == target_slot {
                path.extend_from_slice(&view.zpath_slice()[..target_zprog]);
                return Some(path);
            }
            if view.zpath_len() > 0 {
                path.extend_from_slice(view.zpath_slice());
            }
            let next_base = path.len();
            view.for_each_child(|ch, child_slot| {
                stack.push((child_slot, next_base, Some(ch)));
            });
        }
        None
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
        self.relocations.clear();
        // Track if this was a new key insertion
        let result = match &mut self.storage {
            TrieStorage::Patricia {
                nodes,
                edge_data,
                compressed_paths,
                free_list,
            } => Self::insert_patricia(
                nodes,
                edge_data,
                compressed_paths,
                free_list,
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
                &mut self.relocations,
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
                if let Some(existing_id) = Self::lookup_node_id_cspp(cspp, key) {
                    return Ok(existing_id);
                }
                let max_new_slots = 512usize
                    .saturating_add(key.len() / 4)
                    .saturating_add(16);
                if cspp.total_states().saturating_add(max_new_slots)
                    > (Self::CSPP_MAX_SLOT as usize) + 1
                {
                    return Err(crate::error::ZiporaError::resource_exhausted(
                        "CompressedSparse ZiporaTrie state space (2^24 slots / 64 MiB) exceeded",
                    ));
                }
                let (is_new, valpos) = cspp.insert(key);
                if is_new {
                    self.stats.num_keys += 1;
                    let value_id = self.stats.num_keys as u32;
                    cspp.set_value::<u32>(valpos, value_id);
                }
                Self::lookup_node_id_cspp(cspp, key).ok_or_else(|| {
                    crate::error::ZiporaError::resource_exhausted(
                        "CompressedSparse ZiporaTrie state ID exceeds 24-bit slot encoding",
                    )
                })
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
                ..
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
            TrieStorage::DoubleArray { base, check, .. } => {
                const FREE_BIT: u32 = 0x8000_0000;
                const TERMINAL_BIT: u32 = 0x8000_0000;
                let idx = state as usize;
                if idx >= base.len() || (state != 0 && (check[idx] & FREE_BIT) != 0) {
                    return false;
                }
                (base[idx] & TERMINAL_BIT) != 0
            }
            TrieStorage::Louds { .. } => false,
            TrieStorage::CompressedSparse(cspp) => {
                let node_slot = state >> 8;
                let zprog = (state & 0xFF) as usize;
                if !Self::is_valid_cspp_node(cspp, node_slot) {
                    return false;
                }
                let view = cspp.node_view(node_slot);
                zprog == view.zpath_len() && view.is_final()
            }
        }
    }

    #[inline(always)]
    fn transition(&self, state: StateId, symbol: u8) -> Option<StateId> {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                let node = nodes.get(state as usize)?;
                node.children
                    .binary_search_by_key(&symbol, |(s, _)| *s)
                    .ok()
                    .map(|idx| node.children[idx].1)
            }
            TrieStorage::CriticalBit { .. } => None,
            TrieStorage::DoubleArray { base, check, .. } => {
                // Double array trie transition: next = (base[state] & VALUE_MASK) + symbol
                // Validate with: check[next] == state (referenced project line 100-110).
                // Folding the free-state check into the `check[next] == state` hit path avoids
                // an extra `check[state]` cache miss on every step while still rejecting free states.
                const VALUE_MASK: u32 = 0x7FFF_FFFF;
                const FREE_BIT: u32 = 0x8000_0000;
                const DA_NIL_STATE: u32 = 0x7FFF_FFFF;

                let base_value = base.get(state as usize)? & VALUE_MASK;
                if base_value == 0 || base_value == DA_NIL_STATE {
                    return None;
                }
                let next_state = base_value.checked_add(symbol as u32)?;
                if next_state == 0 {
                    return None;
                }
                if let Some(&check_value) = check.get(next_state as usize)
                    && check_value == state
                    && (state == 0 || (check[state as usize] & FREE_BIT) == 0)
                {
                    return Some(next_state);
                }
                None
            }
            TrieStorage::Louds { .. } => None,
            TrieStorage::CompressedSparse(cspp) => {
                let node_slot = state >> 8;
                let zprog = (state & 0xFF) as usize;
                if !Self::is_valid_cspp_node(cspp, node_slot) {
                    return None;
                }
                let view = cspp.node_view(node_slot);
                let zlen = view.zpath_len();
                if zprog < zlen {
                    if view.zpath_slice()[zprog] == symbol {
                        Self::encode_cspp_state(node_slot, zprog + 1)
                    } else {
                        None
                    }
                } else if zprog == zlen {
                    let next_slot = view.state_move(symbol);
                    if next_slot == crate::fsa::cspp_trie::NIL_STATE {
                        None
                    } else {
                        Self::encode_cspp_state(next_slot, 0)
                    }
                } else {
                    None
                }
            }
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
                if self.is_free_double_array(state) {
                    return Vec::new();
                }
                const VALUE_MASK: u32 = 0x7FFF_FFFF;
                const DA_NIL_STATE: u32 = 0x7FFF_FFFF;

                let Some(&base_raw) = base.get(state as usize) else {
                    return Vec::new();
                };
                let base_val = base_raw & VALUE_MASK;
                if base_val == 0 || base_val == DA_NIL_STATE {
                    return Vec::new();
                }

                (0u8..=255u8)
                    .filter_map(|symbol| {
                        let next_state = base_val.checked_add(symbol as u32)?;
                        if next_state == 0 || (next_state as usize) >= check.len() {
                            return None;
                        }
                        if check[next_state as usize] == state {
                            Some((symbol, next_state))
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            TrieStorage::CompressedSparse(cspp) => {
                let node_slot = state >> 8;
                let zprog = (state & 0xFF) as usize;
                if !Self::is_valid_cspp_node(cspp, node_slot) {
                    return Vec::new();
                }
                let view = cspp.node_view(node_slot);
                let zlen = view.zpath_len();
                if zprog < zlen {
                    if let Some(next_id) = Self::encode_cspp_state(node_slot, zprog + 1) {
                        vec![(view.zpath_slice()[zprog], next_id)]
                    } else {
                        Vec::new()
                    }
                } else if zprog == zlen {
                    let mut out = Vec::with_capacity(view.n_children());
                    view.for_each_child(|ch, child_slot| {
                        if let Some(next_id) = Self::encode_cspp_state(child_slot, 0) {
                            out.push((ch, next_id));
                        }
                    });
                    out
                } else {
                    Vec::new()
                }
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

