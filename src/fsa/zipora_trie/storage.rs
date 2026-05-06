use crate::containers::FastVec;
use crate::succinct::RankSelectOps;
use crate::StateId;
use crate::containers::specialized::UintVector;
use std::collections::{HashMap, VecDeque};

/// Internal storage implementations for different strategies
#[derive(Debug)]
pub(super) enum TrieStorage<R>
where
    R: RankSelectOps,
{
    /// Patricia trie storage with path compression
    Patricia {
        nodes: FastVec<PatriciaNode>,
        edge_data: FastVec<u8>,
        compressed_paths: HashMap<StateId, Vec<u8>>,
    },
    /// Critical-bit trie storage
    CriticalBit {
        nodes: FastVec<CritBitNode>,
        keys: FastVec<Vec<u8>>,
        critical_cache: HashMap<usize, u8>,
    },
    /// Double array trie storage
    DoubleArray {
        base: FastVec<u32>,
        check: FastVec<u32>,
        free_list: VecDeque<StateId>,
        state_count: usize,
    },
    /// LOUDS trie storage with succinct structures
    Louds {
        louds: R,
        is_link: R,
        next_link: UintVector,
        label_data: FastVec<u8>,
        core_data: FastVec<u8>,
        next_trie: Option<Box<super::trie::ZiporaTrie<R>>>,
    },
    /// Compressed sparse trie storage
    CompressedSparse(crate::fsa::cspp_trie::CsppTrie),
}

/// Patricia trie node with path compression (compact representation)
#[derive(Debug, Clone, Default)]
pub(super) struct PatriciaNode {
    /// Compact children storage: sorted Vec of (symbol, StateId) pairs
    pub(super) children: Vec<(u8, StateId)>,
    /// Compressed path data offset
    pub(super) _path_offset: u32,
    /// Compressed path length
    pub(super) _path_length: u16,
    /// Whether this node represents a complete key
    pub(super) is_final: bool,
    /// Node flags for optimization
    pub(super) _flags: u8,
}

/// Critical-bit trie node
#[repr(align(64))]
#[derive(Debug, Clone)]
pub(super) struct CritBitNode {
    /// Critical byte position
    pub(super) _crit_byte: usize,
    /// Critical bit position (0-7)
    pub(super) _crit_bit: u8,
    /// Left child (bit = 0)
    pub(super) _left_child: Option<StateId>,
    /// Right child (bit = 1)
    pub(super) _right_child: Option<StateId>,
    /// Key stored at this node (for leaves)
    pub(super) _key_index: Option<u32>,
    /// Whether this is a final state
    pub(super) is_final: bool,
}

/// Sparse trie node for compressed sparse storage
#[derive(Debug, Clone)]
pub(super) struct SparseNode {
    /// Sparse children map
    pub(super) children: HashMap<u8, StateId>,
    /// Compressed edge label
    pub(super) _edge_label: Option<u32>,
    /// Final state flag
    pub(super) is_final: bool,
}
