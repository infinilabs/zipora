use crate::StateId;
use crate::containers::FastVec;
use crate::containers::specialized::UintVector;
use crate::succinct::RankSelectOps;
use std::collections::HashMap;

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
        /// Node ids unlinked by `remove`, reused by the next `insert` so
        /// insert/remove churn does not grow `nodes` without bound.
        free_list: Vec<StateId>,
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
        /// Every free cell, threaded through the cells themselves; the only
        /// thing `find_free_base` walks, so allocation never rescans the
        /// occupied prefix.
        free_list: DaFreeList,
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

/// Double-array cell layout (referenced project): `base` bits 0-30 = base
/// value (`DA_NIL_STATE` = no children yet), bit 31 = terminal; `check` bits
/// 0-30 = parent state, bit 31 = free.
pub(super) const DA_TERMINAL_BIT: u32 = 0x8000_0000;
pub(super) const DA_FREE_BIT: u32 = 0x8000_0000;
pub(super) const DA_VALUE_MASK: u32 = 0x7FFF_FFFF;
pub(super) const DA_NIL_STATE: u32 = 0x7FFF_FFFF;
/// Highest index a state may occupy (`DA_NIL_STATE` is reserved as a marker).
pub(super) const DA_MAX_STATE: u32 = 0x7FFF_FFFE;

/// Free-cell bookkeeping for the double array.
///
/// Free cells form a doubly-linked list threaded through the cells themselves:
/// a free cell stores its successor in the low 31 bits of `check` (under
/// `DA_FREE_BIT`) and its predecessor in `base`; `DA_NIL_STATE` ends the list
/// in both directions. Lookups never read the low bits of a free cell (the
/// free bit already fails the parent comparison), so the links cost no extra
/// memory. Walking this list visits only free cells, which is what keeps base
/// allocation linear: the previous linear probe crossed the whole occupied
/// prefix on every relocation.
#[derive(Debug, Clone)]
pub(super) struct DaFreeList {
    pub(super) head: u32,
    pub(super) tail: u32,
    /// One past the highest cell ever claimed. Every cell at or above it is
    /// free, so a multi-symbol search that gives up on the scattered holes can
    /// settle there without growing the array.
    pub(super) high_water: u32,
}

impl DaFreeList {
    /// Root cell 0 is always occupied.
    pub(super) fn new() -> Self {
        Self {
            head: DA_NIL_STATE,
            tail: DA_NIL_STATE,
            high_water: 1,
        }
    }

    /// Take the free cell `pos` off the list and attach it to `parent` with no
    /// children yet.
    pub(super) fn claim(&mut self, base: &mut FastVec<u32>, check: &mut FastVec<u32>, pos: u32, parent: u32) {
        self.unlink(base, check, pos);
        check[pos as usize] = parent;
        base[pos as usize] = DA_NIL_STATE;
        self.high_water = self.high_water.max(pos + 1);
    }

    /// Append `cell` as the last free cell (freshly grown cells, ascending).
    pub(super) fn push_back(&mut self, base: &mut FastVec<u32>, check: &mut FastVec<u32>, cell: u32) {
        base[cell as usize] = self.tail;
        check[cell as usize] = DA_FREE_BIT | DA_NIL_STATE;
        if self.tail == DA_NIL_STATE {
            self.head = cell;
        } else {
            check[self.tail as usize] = DA_FREE_BIT | cell;
        }
        self.tail = cell;
    }

    /// Insert `cell` as the first free cell, so holes released by a relocation
    /// are retried before the fresh tail and the array stays compact.
    pub(super) fn push_front(&mut self, base: &mut FastVec<u32>, check: &mut FastVec<u32>, cell: u32) {
        base[cell as usize] = DA_NIL_STATE;
        check[cell as usize] = DA_FREE_BIT | self.head;
        if self.head == DA_NIL_STATE {
            self.tail = cell;
        } else {
            base[self.head as usize] = cell;
        }
        self.head = cell;
    }

    /// Remove free `cell` from the list; the caller then overwrites its
    /// `base`/`check` with real state data.
    pub(super) fn unlink(&mut self, base: &mut FastVec<u32>, check: &mut FastVec<u32>, cell: u32) {
        debug_assert!(check[cell as usize] & DA_FREE_BIT != 0, "unlink of an occupied cell");
        let prev = base[cell as usize];
        let next = check[cell as usize] & DA_VALUE_MASK;
        if prev == DA_NIL_STATE {
            self.head = next;
        } else {
            check[prev as usize] = DA_FREE_BIT | next;
        }
        if next == DA_NIL_STATE {
            self.tail = prev;
        } else {
            base[next as usize] = prev;
        }
    }
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
