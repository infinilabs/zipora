//! Fixed LOUDS Trie implementation with proper dynamic insertion

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::error::Result;
use crate::fsa::traits::{
    FiniteStateAutomaton, PrefixIterable, StateInspectable, StatisticsProvider, Trie, TrieBuilder,
    TrieStats,
};
use crate::statistics::{TrieStatistics, MemorySize, MemoryBreakdown};
use crate::succinct::{BitVector, RankSelect256};
use crate::{FastVec, StateId};

/// Internal node representation for building the trie
#[derive(Debug, Clone)]
struct TrieNode {
    children: HashMap<u8, usize>, // label -> child node index
    is_final: bool,
}

/// LOUDS Trie implementation using succinct data structures
///
/// This implementation uses a hybrid approach:
/// - Internal tree representation for dynamic construction
/// - LOUDS representation for efficient querying
#[derive(Debug, Clone)]
pub struct LoudsTrie {
    /// LOUDS bit sequence representing the tree structure
    louds_bits: BitVector,
    /// Rank-select structure for efficient navigation
    rank_select: RankSelect256,
    /// Edge labels stored in level order
    labels: FastVec<u8>,
    /// Bit vector marking final (accepting) states
    is_final: BitVector,
    /// Number of keys stored in the trie
    num_keys: usize,
    /// Internal tree representation for dynamic construction
    nodes: Vec<TrieNode>,
    /// Next available node index
    next_node_id: usize,
    /// Simple statistics  
    statistics: std::sync::Arc<std::sync::Mutex<crate::statistics::TrieStat>>,
}

impl LoudsTrie {
    /// Create a new empty LOUDS trie
    pub fn new() -> Self {
        let louds_bits = BitVector::new();
        let rank_select = RankSelect256::new(louds_bits.clone()).unwrap();

        // Initialize with root node
        let mut nodes = Vec::new();
        nodes.push(TrieNode {
            children: HashMap::new(),
            is_final: false,
        });

        Self {
            louds_bits,
            rank_select,
            labels: FastVec::new(),
            is_final: BitVector::new(),
            num_keys: 0,
            nodes,
            next_node_id: 1,
            statistics: std::sync::Arc::new(std::sync::Mutex::new(crate::statistics::TrieStat::new())),
        }
    }

    /// Get simple statistics
    pub fn get_stats(&self) -> crate::statistics::TrieStat {
        self.statistics.lock().unwrap().clone()
    }
    
    /// Print statistics to stderr
    pub fn print_stats(&self) {
        let mut stderr = std::io::stderr();
        self.statistics.lock().unwrap().print(&mut stderr).ok();
    }
    
    /// Simple memory usage calculation
    pub fn mem_size(&self) -> usize {
        // Calculate LOUDS bits memory (approximate as bit vector length / 8)
        let louds_bits_memory = (self.louds_bits.len() + 7) / 8;
        
        // Calculate rank-select memory (estimated)
        let rank_select_memory = 256; // Fixed overhead approximation
        
        // Calculate labels memory (FastVec length)
        let labels_memory = self.labels.len();
        
        // Calculate is_final memory (approximate as bit vector length / 8)
        let is_final_memory = (self.is_final.len() + 7) / 8;
        
        // Calculate nodes memory
        let nodes_memory = std::mem::size_of_val(&*self.nodes) + 
            self.nodes.iter().map(|node| {
                std::mem::size_of_val(node) + 
                node.children.capacity() * (std::mem::size_of::<u8>() + std::mem::size_of::<usize>())
            }).sum::<usize>();
        
        let total_size = louds_bits_memory + rank_select_memory + labels_memory + is_final_memory + nodes_memory;
        
        // Update total_bytes in statistics
        self.statistics.lock().unwrap().total_bytes = total_size as u64;
        
        total_size
    }
    
    /// Check if debug output is enabled via environment variable
    fn debug_enabled() -> bool {
        std::env::var("LOUDS_TRIE_DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    }
    
    /// Conditionally print statistics if debug is enabled
    pub fn debug_print_statistics(&self) {
        if Self::debug_enabled() {
            eprintln!("LoudsTrie Debug Output:");
            self.print_stats();
        }
    }

    /// Rebuild the LOUDS representation from the tree structure
    fn rebuild_louds(&mut self) -> Result<()> {
        // Clear existing LOUDS structures
        self.louds_bits = BitVector::new();
        self.labels = FastVec::new();
        self.is_final = BitVector::new();

        if self.nodes.is_empty() {
            return Ok(());
        }

        // Build level-order traversal with correct state ID assignment
        let mut queue = VecDeque::new();
        let mut nodes_in_order = Vec::new(); // nodes in the order they should appear in LOUDS

        // Start with root
        queue.push_back(0usize); // root node_id
        nodes_in_order.push(0usize);

        while !queue.is_empty() {
            let level_size = queue.len();

            // Process all nodes at current level
            for _ in 0..level_size {
                if let Some(node_id) = queue.pop_front() {
                    if let Some(node) = self.nodes.get(node_id) {
                        // Get children in sorted order
                        let mut children: Vec<_> = node.children.iter().collect();
                        children.sort_by_key(|(label, _)| *label);

                        // Add children to the order and queue
                        for &(ref _label, &child_node_id) in &children {
                            nodes_in_order.push(child_node_id);
                            queue.push_back(child_node_id);
                        }
                    }
                }
            }
        }

        // Now build LOUDS structures based on the correct ordering
        for (_state_id, &node_id) in nodes_in_order.iter().enumerate() {
            if let Some(node) = self.nodes.get(node_id) {
                // Add final state info
                self.is_final.push(node.is_final)?;

                // Get children in sorted order for LOUDS bits and labels
                let mut children: Vec<_> = node.children.iter().collect();
                children.sort_by_key(|(label, _)| *label);

                // Add LOUDS bits and labels for children
                for (label, _child_node_id) in &children {
                    self.louds_bits.push(true)?; // 1 for each child
                    self.labels.push(**label)?;
                }

                // Add terminating 0 if node has children
                if !children.is_empty() {
                    self.louds_bits.push(false)?;
                }
            }
        }

        // Rebuild rank-select structure
        self.rank_select = RankSelect256::new(self.louds_bits.clone())?;

        Ok(())
    }

    /// Get the position in LOUDS sequence for a state
    fn state_to_louds_pos(&self, state: StateId) -> usize {
        if state == 0 {
            0 // Root starts at position 0
        } else {
            // For state > 0, find the position where this state's children start
            // This is the position after the (state)th '0' bit in LOUDS sequence
            let mut zeros_seen = 0;
            for i in 0..self.louds_bits.len() {
                if let Some(bit) = self.louds_bits.get(i) {
                    if !bit {
                        zeros_seen += 1;
                        if zeros_seen == state as usize {
                            return i + 1;
                        }
                    }
                }
            }
            self.louds_bits.len()
        }
    }

    /// Get the first child position in the labels array
    fn first_child_label_pos(&self, state: StateId) -> usize {
        // Count all '1' bits before this state's children start position
        let pos = self.state_to_louds_pos(state);
        self.rank_select.rank1(pos)
    }

    /// Get the number of children for a state
    fn child_count(&self, state: StateId) -> usize {
        let start_pos = self.state_to_louds_pos(state);
        let mut count = 0;

        // Count consecutive 1s starting from start_pos
        if start_pos >= self.louds_bits.len() {
            return 0;
        }

        for i in start_pos..self.louds_bits.len() {
            if let Some(bit) = self.louds_bits.get(i) {
                if bit {
                    count += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        count
    }

    /// Navigate to a child state with the given label
    fn goto_child(&self, state: StateId, label: u8) -> Option<StateId> {
        let start_time = std::time::Instant::now();
        
        let child_count = self.child_count(state);
        if child_count == 0 {
            // Update simple timing statistics
            self.statistics.lock().unwrap().lookup_time += start_time.elapsed().as_secs_f64();
            return None;
        }

        let first_label_pos = self.first_child_label_pos(state);

        // Linear search through children
        for i in 0..child_count {
            if let Some(&child_label) = self.labels.get(first_label_pos + i) {
                if child_label == label {
                    // Calculate child state ID based on the position of its '1' bit
                    let parent_pos = self.state_to_louds_pos(state);
                    let child_1bit_pos = parent_pos + i;

                    // Count total number of '1' bits up to and including this position
                    let ones_up_to = self.rank_select.rank1(child_1bit_pos + 1);
                    // Update simple timing statistics
                    self.statistics.lock().unwrap().lookup_time += start_time.elapsed().as_secs_f64();
                    return Some(ones_up_to as StateId);
                }
            }
        }

        // Update simple timing statistics
        self.statistics.lock().unwrap().lookup_time += start_time.elapsed().as_secs_f64();
        None
    }
}

impl Default for LoudsTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl FiniteStateAutomaton for LoudsTrie {
    fn root(&self) -> StateId {
        0
    }

    fn is_final(&self, state: StateId) -> bool {
        self.is_final.get(state as usize).unwrap_or(false)
    }

    fn transition(&self, state: StateId, symbol: u8) -> Option<StateId> {
        self.goto_child(state, symbol)
    }

    fn transitions(&self, state: StateId) -> Box<dyn Iterator<Item = (u8, StateId)> + '_> {
        let child_count = self.child_count(state);
        let first_label_pos = self.first_child_label_pos(state);

        let iter = (0..child_count).filter_map(move |i| {
            if let Some(&label) = self.labels.get(first_label_pos + i) {
                let child_state = self.goto_child(state, label)?;
                Some((label, child_state))
            } else {
                None
            }
        });

        Box::new(iter)
    }

    fn longest_prefix(&self, input: &[u8]) -> Option<usize> {
        let mut state = self.root();
        let mut last_final = None;
        let mut consumed = 0;

        for (i, &symbol) in input.iter().enumerate() {
            if self.is_final(state) {
                last_final = Some(i);
            }

            match self.transition(state, symbol) {
                Some(next_state) => {
                    state = next_state;
                    consumed = i + 1;
                }
                None => break,
            }
        }

        // Check if final state after consuming input is final
        // Only if we consumed ALL input characters
        if consumed == input.len() && self.is_final(state) {
            Some(input.len())
        } else {
            last_final
        }
    }
}

impl MemorySize for LoudsTrie {
    fn mem_size(&self) -> usize {
        // Calculate LOUDS bits memory (approximate as bit vector length / 8)
        let louds_bits_memory = (self.louds_bits.len() + 7) / 8;
        
        // Calculate rank-select memory (estimated)
        let rank_select_memory = 256; // Fixed overhead approximation
        
        // Calculate labels memory (FastVec length)
        let labels_memory = self.labels.len();
        
        // Calculate is_final memory (approximate as bit vector length / 8)
        let is_final_memory = (self.is_final.len() + 7) / 8;
        
        // Calculate nodes memory
        let nodes_memory = std::mem::size_of_val(&*self.nodes) + 
            self.nodes.iter().map(|node| {
                std::mem::size_of_val(node) + 
                node.children.capacity() * (std::mem::size_of::<u8>() + std::mem::size_of::<usize>())
            }).sum::<usize>();
        
        // Calculate struct overhead
        let struct_overhead = std::mem::size_of::<Self>();
        
        let total_size = louds_bits_memory + rank_select_memory + labels_memory + is_final_memory + nodes_memory + struct_overhead;
        
        // Update total_bytes in statistics
        self.statistics.lock().unwrap().total_bytes = total_size as u64;
        
        total_size
    }
    
    fn detailed_mem_size(&self) -> MemoryBreakdown {
        let mut breakdown = MemoryBreakdown::new();
        
        // Calculate LOUDS bits memory (approximate as bit vector length / 8)
        let louds_bits_memory = (self.louds_bits.len() + 7) / 8;
        breakdown.add_component("louds_bits", louds_bits_memory);
        
        // Calculate rank-select memory (estimated)
        let rank_select_memory = 256; // Fixed overhead approximation
        breakdown.add_component("rank_select", rank_select_memory);
        
        // Calculate labels memory (FastVec length)
        let labels_memory = self.labels.len();
        breakdown.add_component("labels", labels_memory);
        
        // Calculate is_final memory (approximate as bit vector length / 8)
        let is_final_memory = (self.is_final.len() + 7) / 8;
        breakdown.add_component("is_final", is_final_memory);
        
        // Calculate nodes memory
        let nodes_memory = std::mem::size_of_val(&*self.nodes) + 
            self.nodes.iter().map(|node| {
                std::mem::size_of_val(node) + 
                node.children.capacity() * (std::mem::size_of::<u8>() + std::mem::size_of::<usize>())
            }).sum::<usize>();
        breakdown.add_component("nodes", nodes_memory);
        
        // Calculate struct overhead
        breakdown.add_component("struct_overhead", std::mem::size_of::<Self>());
        
        breakdown
    }
}

impl Trie for LoudsTrie {
    fn insert(&mut self, key: &[u8]) -> Result<StateId> {
        use std::sync::atomic::Ordering;
        use std::time::Instant;
        
        let start_time = Instant::now();
        let mut node_id = 0usize; // Start at root in tree representation

        // Traverse as far as possible in tree representation
        let mut i = 0;
        while i < key.len() {
            if let Some(node) = self.nodes.get(node_id) {
                if let Some(&child_node_id) = node.children.get(&key[i]) {
                    node_id = child_node_id;
                    i += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Add remaining suffix
        while i < key.len() {
            let is_final = i == key.len() - 1;
            let new_node_id = self.next_node_id;
            self.next_node_id += 1;

            self.nodes.push(TrieNode {
                children: HashMap::new(),
                is_final,
            });

            // Add child to current node
            if let Some(current_node) = self.nodes.get_mut(node_id) {
                current_node.children.insert(key[i], new_node_id);
            }

            if is_final {
                self.num_keys += 1;
            }

            node_id = new_node_id;
            i += 1;
        }

        // Mark final state if we traversed the entire key
        if i == key.len() {
            if let Some(node) = self.nodes.get_mut(node_id) {
                if !node.is_final {
                    node.is_final = true;
                    self.num_keys += 1;
                }
            }
        }

        // Rebuild LOUDS representation
        self.rebuild_louds()?;

        // Update simple timing statistics
        self.statistics.lock().unwrap().insert_time += start_time.elapsed().as_secs_f64();

        Ok(node_id as StateId)
    }

    fn len(&self) -> usize {
        self.num_keys
    }
}

impl StateInspectable for LoudsTrie {
    fn out_degree(&self, state: StateId) -> usize {
        self.child_count(state)
    }

    fn out_symbols(&self, state: StateId) -> Vec<u8> {
        let child_count = self.child_count(state);
        let first_label_pos = self.first_child_label_pos(state);

        (0..child_count)
            .filter_map(|i| self.labels.get(first_label_pos + i).copied())
            .collect()
    }
}

impl StatisticsProvider for LoudsTrie {
    fn stats(&self) -> TrieStats {
        let louds_memory = self.louds_bits.len() / 8 + 1;
        let labels_memory = self.labels.len();
        let final_memory = self.is_final.len() / 8 + 1;
        let rank_select_memory = 256;

        let memory_usage = louds_memory + labels_memory + final_memory + rank_select_memory;

        let mut stats = TrieStats {
            num_states: self.is_final.len(),
            num_keys: self.num_keys,
            num_transitions: self.labels.len(),
            max_depth: 0,
            avg_depth: 0.0,
            memory_usage,
            bits_per_key: 0.0,
        };

        stats.calculate_bits_per_key();
        stats
    }
}

/// Builder for constructing LOUDS tries from sorted key sequences
pub struct LoudsTrieBuilder;

impl TrieBuilder<LoudsTrie> for LoudsTrieBuilder {
    fn build_from_sorted<I>(keys: I) -> Result<LoudsTrie>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        let mut trie = LoudsTrie::new();

        for key in keys {
            trie.insert(&key)?;
        }

        Ok(trie)
    }
}

/// Iterator for prefix enumeration in LOUDS tries
pub struct LoudsTriePrefixIterator<'a> {
    trie: &'a LoudsTrie,
    stack: VecDeque<(StateId, Vec<u8>)>,
}

impl<'a> LoudsTriePrefixIterator<'a> {
    fn new(trie: &'a LoudsTrie, prefix: &[u8]) -> Option<Self> {
        let mut state = trie.root();

        // Navigate to prefix state
        for &symbol in prefix {
            state = trie.transition(state, symbol)?;
        }

        let mut stack = VecDeque::new();
        stack.push_back((state, prefix.to_vec()));

        Some(Self { trie, stack })
    }
}

impl<'a> Iterator for LoudsTriePrefixIterator<'a> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((state, path)) = self.stack.pop_front() {
            // If this is a final state, yield the path
            let is_final = self.trie.is_final(state);

            // Add children to stack for future exploration
            for (symbol, child_state) in self.trie.transitions(state) {
                let mut child_path = path.clone();
                child_path.push(symbol);
                self.stack.push_back((child_state, child_path));
            }

            if is_final {
                return Some(path);
            }
        }

        None
    }
}

impl PrefixIterable for LoudsTrie {
    fn iter_prefix(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = Vec<u8>> + '_> {
        match LoudsTriePrefixIterator::new(self, prefix) {
            Some(iter) => Box::new(iter),
            None => Box::new(std::iter::empty()),
        }
    }
}

// Implement builder as associated function
impl LoudsTrie {
    /// Build a LOUDS trie from a sorted iterator of keys
    pub fn build_from_sorted<I>(keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        LoudsTrieBuilder::build_from_sorted(keys)
    }

    /// Build a LOUDS trie from an unsorted iterator of keys
    pub fn build_from_unsorted<I>(keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        LoudsTrieBuilder::build_from_unsorted(keys)
    }

    /// Build LOUDS trie from SortableStrVec with configuration (matches C++ pattern)
    /// This follows the C++ pattern: build_from(SortableStrVec& strVec, const NestLoudsTrieConfig& conf)
    pub fn build_from_sortable_str_vec(
        keys: &crate::containers::specialized::SortableStrVec,
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<Self> {
        // Convert SortableStrVec to Vec<&[u8]> for processing
        let key_refs: Vec<&[u8]> = (0..keys.len())
            .filter_map(|i| keys.get(i).map(|s| s.as_bytes()))
            .collect();
        Self::build_from_str_vec_impl(&key_refs, config)
    }

    /// Build LOUDS trie from FixedLenStrVec with configuration (matches C++ pattern)
    /// This follows the C++ pattern: build_from(FixedLenStrVec& strVec, const NestLoudsTrieConfig& conf)
    pub fn build_from_fixed_len_str_vec<const N: usize>(
        keys: &crate::containers::specialized::FixedLenStrVec<N>,
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<Self> {
        // Convert FixedLenStrVec to Vec<&[u8]> for processing
        let key_refs: Vec<&[u8]> = (0..keys.len())
            .filter_map(|i| keys.get(i).map(|s| s.as_bytes()))
            .collect();
        Self::build_from_str_vec_impl(&key_refs, config)
    }

    /// Build LOUDS trie from ZoSortedStrVec with configuration (matches C++ pattern)
    /// This follows the C++ pattern: build_from(ZoSortedStrVec& strVec, const NestLoudsTrieConfig& conf)
    pub fn build_from_zo_sorted_str_vec(
        keys: &crate::containers::specialized::ZoSortedStrVec,
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<Self> {
        // Convert ZoSortedStrVec to Vec<&[u8]> for processing
        let key_refs: Vec<&[u8]> = (0..keys.len())
            .filter_map(|i| keys.get(i).map(|s| s.as_bytes()))
            .collect();
        Self::build_from_str_vec_impl(&key_refs, config)
    }

    /// Build LOUDS trie from vector of byte slices with configuration (matches C++ pattern)
    /// This follows the C++ pattern: build_from(Vec<Vec<u8>>& strVec, const NestLoudsTrieConfig& conf)
    pub fn build_from_vec_u8(
        keys: &[Vec<u8>],
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<Self> {
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        Self::build_from_str_vec_impl(&key_refs, config)
    }

    /// Build LOUDS trie from slice of byte slices with configuration (matches C++ pattern)
    /// This follows the C++ pattern: build_from(&[&[u8]], const NestLoudsTrieConfig& conf)
    pub fn build_from_slice_u8(
        keys: &[&[u8]],
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<Self> {
        Self::build_from_str_vec_impl(keys, config)
    }

    /// Internal implementation for build_from methods
    /// Converts NestLoudsTrieConfig to LoudsTrie construction parameters
    fn build_from_str_vec_impl(keys: &[&[u8]], config: &crate::config::nest_louds_trie::NestLoudsTrieConfig) -> Result<Self> {
        // For LoudsTrie, we focus on the basic configuration parameters
        // and ignore the advanced nesting features that are specific to NestedLoudsTrie
        
        let mut trie = Self::new();
        
        // Insert all keys - LoudsTrie doesn't have advanced configuration support,
        // so we use the basic insertion mechanism
        for key in keys {
            trie.insert(key)?;
        }

        Ok(trie)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsa::traits::{FiniteStateAutomaton, StateInspectable, Trie};

    #[test]
    fn test_louds_trie_basic_operations() {
        let mut trie = LoudsTrie::new();

        assert!(trie.is_empty());
        assert_eq!(trie.root(), 0);

        // Insert some keys
        trie.insert(b"cat").unwrap();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();

        assert_eq!(trie.len(), 3);
        assert!(!trie.is_empty());

        // Test lookups
        assert!(trie.contains(b"cat"));
        assert!(trie.contains(b"car"));
        assert!(trie.contains(b"card"));
        assert!(!trie.contains(b"ca"));
        assert!(!trie.contains(b"care"));
        assert!(!trie.contains(b"dog"));
    }

    #[test]
    fn test_louds_trie_transitions() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"abc").unwrap();
        trie.insert(b"abd").unwrap();

        let root = trie.root();
        let a_state = trie.transition(root, b'a').unwrap();
        let b_state = trie.transition(a_state, b'b').unwrap();

        // Should have two children: 'c' and 'd'
        assert_eq!(trie.out_degree(b_state), 2);
        let symbols = trie.out_symbols(b_state);
        assert!(symbols.contains(&b'c'));
        assert!(symbols.contains(&b'd'));

        let c_state = trie.transition(b_state, b'c').unwrap();
        let d_state = trie.transition(b_state, b'd').unwrap();

        assert!(trie.is_final(c_state));
        assert!(trie.is_final(d_state));
    }

    #[test]
    fn test_louds_trie_accepts() {
        let mut trie = LoudsTrie::new();

        // Debug: Insert first key
        println!("=== Inserting 'hello' ===");
        trie.insert(b"hello").unwrap();
        println!(
            "After 'hello': LOUDS bits={:?}",
            (0..trie.louds_bits.len())
                .map(|i| if trie.louds_bits.get(i).unwrap_or(false) {
                    '1'
                } else {
                    '0'
                })
                .collect::<String>()
        );
        println!(
            "After 'hello': labels={:?}",
            (0..trie.labels.len())
                .map(|i| trie.labels.get(i).copied().unwrap_or(0) as char)
                .collect::<String>()
        );
        assert!(trie.accepts(b"hello"));
        println!("'hello' lookup: OK");

        // Debug: Insert second key
        println!("=== Inserting 'world' ===");
        trie.insert(b"world").unwrap();
        println!(
            "After 'world': LOUDS bits={:?}",
            (0..trie.louds_bits.len())
                .map(|i| if trie.louds_bits.get(i).unwrap_or(false) {
                    '1'
                } else {
                    '0'
                })
                .collect::<String>()
        );
        println!(
            "After 'world': labels={:?}",
            (0..trie.labels.len())
                .map(|i| trie.labels.get(i).copied().unwrap_or(0) as char)
                .collect::<String>()
        );

        println!("Testing 'hello' after inserting 'world'...");

        // Debug the lookup process for 'hello'
        let mut state = trie.root();
        println!("Root state: {}", state);
        for (_i, &byte) in b"hello".iter().enumerate() {
            let ch = byte as char;
            println!("Looking for '{}' from state {}", ch, state);
            if let Some(next_state) = trie.transition(state, byte) {
                println!("  Found transition to state {}", next_state);
                state = next_state;
            } else {
                println!("  No transition found for '{}'", ch);
                break;
            }
        }
        println!("Final state: {}, is_final: {}", state, trie.is_final(state));

        assert!(trie.accepts(b"hello"));
        assert!(trie.accepts(b"world"));
        assert!(!trie.accepts(b"hell"));
        assert!(!trie.accepts(b"worlds"));
        assert!(!trie.accepts(b"foo"));
    }

    #[test]
    fn test_louds_trie_longest_prefix() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();
        trie.insert(b"care").unwrap();

        assert_eq!(trie.longest_prefix(b"car"), Some(3));
        assert_eq!(trie.longest_prefix(b"card"), Some(4));
        assert_eq!(trie.longest_prefix(b"cards"), Some(4));
        assert_eq!(trie.longest_prefix(b"careless"), Some(4)); // "care" is a 4-char prefix
        assert_eq!(trie.longest_prefix(b"cat"), None);
    }

    #[test]
    fn test_louds_trie_builder() {
        let keys = vec![
            b"cat".to_vec(),
            b"car".to_vec(),
            b"card".to_vec(),
            b"care".to_vec(),
        ];

        let trie = LoudsTrie::build_from_sorted(keys.clone()).unwrap();
        assert_eq!(trie.len(), 4);

        for key in &keys {
            assert!(trie.contains(key));
        }

        // Test with unsorted keys
        let mut unsorted_keys = keys.clone();
        unsorted_keys.reverse();

        let trie2 = LoudsTrie::build_from_unsorted(unsorted_keys).unwrap();
        assert_eq!(trie2.len(), 4);

        for key in &keys {
            assert!(trie2.contains(key));
        }
    }

    #[test]
    fn test_louds_trie_prefix_iteration() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();
        trie.insert(b"care").unwrap();
        trie.insert(b"cat").unwrap();

        // Test prefix "car"
        let mut car_results: Vec<Vec<u8>> = trie.iter_prefix(b"car").collect();
        car_results.sort();

        let expected = vec![b"car".to_vec(), b"card".to_vec(), b"care".to_vec()];
        assert_eq!(car_results, expected);

        // Test prefix "ca"
        let mut ca_results: Vec<Vec<u8>> = trie.iter_prefix(b"ca").collect();
        ca_results.sort();

        let expected = vec![
            b"car".to_vec(),
            b"card".to_vec(),
            b"care".to_vec(),
            b"cat".to_vec(),
        ];
        assert_eq!(ca_results, expected);

        // Test non-existent prefix (should return empty iterator)
        let dog_results: Vec<Vec<u8>> = trie.iter_prefix(b"dog").collect();
        assert!(dog_results.is_empty());
    }

    #[test]
    fn test_louds_trie_empty_key() {
        let mut trie = LoudsTrie::new();

        // Insert empty key
        trie.insert(b"").unwrap();
        assert_eq!(trie.len(), 1);
        assert!(trie.contains(b""));

        // Root should now be final
        assert!(trie.is_final(trie.root()));
    }

    #[test]
    fn test_louds_trie_duplicate_keys() {
        let mut trie = LoudsTrie::new();

        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 1);

        // Insert the same key again
        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 1); // Should not increase

        assert!(trie.contains(b"hello"));
    }

    #[test]
    fn test_louds_trie_statistics() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"cat").unwrap();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();

        let stats = trie.stats();
        assert_eq!(stats.num_keys, 3);
        assert!(stats.memory_usage > 0);
        assert!(stats.bits_per_key > 0.0);
    }

    #[test]
    fn test_louds_trie_large_keys() {
        let mut trie = LoudsTrie::new();

        // Test with longer keys
        let long_key = b"this_is_a_very_long_key_for_testing_purposes";
        trie.insert(long_key).unwrap();

        assert!(trie.contains(long_key));
        assert_eq!(trie.len(), 1);

        // Test prefix of long key
        let prefix = b"this_is_a_very";
        assert_eq!(trie.longest_prefix(long_key), Some(long_key.len()));
        assert_eq!(trie.longest_prefix(prefix), None);
    }

    #[test]
    fn test_louds_trie_transitions_iterator() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"abc").unwrap();
        trie.insert(b"abd").unwrap();
        trie.insert(b"ac").unwrap();

        let root = trie.root();
        let a_state = trie.transition(root, b'a').unwrap();

        // Should have transitions to 'b' and 'c'
        let transitions: Vec<(u8, StateId)> = trie.transitions(a_state).collect();
        assert_eq!(transitions.len(), 2);

        let symbols: Vec<u8> = transitions.iter().map(|(s, _)| *s).collect();
        assert!(symbols.contains(&b'b'));
        assert!(symbols.contains(&b'c'));
    }
}
