use super::config::ZiporaTrieConfig;
use super::trie::ZiporaTrie;
use crate::error::Result;
use crate::fsa::traits::Trie;
use crate::succinct::RankSelectOps;

/// Map wrapper for ZiporaTrie that associates values with keys
///
/// This is a separate type that wraps a ZiporaTrie and adds value storage.
/// Values are stored in a parallel Vec indexed by the state ID returned from insert.
///
/// # Examples
///
/// ```rust
/// use zipora::fsa::ZiporaTrieMap;
///
/// let mut map = ZiporaTrieMap::<u32>::new();
/// map.insert(b"hello", 42).unwrap();
/// map.insert(b"world", 100).unwrap();
///
/// assert_eq!(map.get(b"hello"), Some(42));
/// assert_eq!(map.get(b"world"), Some(100));
/// assert_eq!(map.get(b"missing"), None);
/// ```
#[derive(Debug)]
pub struct ZiporaTrieMap<V: Copy, R = crate::succinct::RankSelectInterleaved256>
where
    R: RankSelectOps,
{
    trie: ZiporaTrie<R>,
    values: Vec<Option<V>>,
}

impl<V: Copy, R> ZiporaTrieMap<V, R>
where
    R: RankSelectOps + Default,
{
    /// Create a new empty trie map
    pub fn new() -> Self {
        Self {
            trie: ZiporaTrie::new(),
            values: Vec::new(),
        }
    }

    /// Create a new trie map with custom configuration
    pub fn with_config(config: ZiporaTrieConfig) -> Self {
        Self {
            trie: ZiporaTrie::with_config(config),
            values: Vec::new(),
        }
    }

    /// Insert a key-value pair, returning the previous value if the key existed
    pub fn insert(&mut self, key: &[u8], value: V) -> Result<Option<V>> {
        // Get the state ID for this key
        let state_id = <ZiporaTrie<R> as Trie>::insert(&mut self.trie, key)?;

        // Double-array collision resolution may have relocated existing states;
        // move their values to the new slots (in recorded order) before storing
        // ours, or lookups on previously inserted keys would read stale slots.
        for &(old, new) in &self.trie.relocations {
            let (old, new) = (old as usize, new as usize);
            let old_val = self.values.get_mut(old).and_then(Option::take);
            if new >= self.values.len() {
                self.values.resize(new + 1, None);
            }
            self.values[new] = old_val;
        }

        // Determine value slot: CompressedSparse stores a stable 1-based u32
        // value_id inside the CSPP node's trailing bytes (preserved across
        // fork/split_zpath/add_state_move/realloc_node), whereas DoubleArray
        // and Patricia index `values` by `state_id`.
        let idx = match &self.trie.storage {
            super::storage::TrieStorage::CompressedSparse(cspp) => {
                let valpos = cspp.lookup(key).ok_or_else(|| {
                    crate::error::ZiporaError::invalid_data(
                        "CSPP lookup failed immediately after insert",
                    )
                })?;
                cspp.get_value::<u32>(valpos) as usize
            }
            _ => state_id as usize,
        };

        // Ensure values vec is large enough
        if idx >= self.values.len() {
            self.values.resize(idx + 1, None);
        }

        // Store the value and return the previous one
        let prev = self.values[idx];
        self.values[idx] = Some(value);

        Ok(prev)
    }

    /// Get the value associated with a key
    pub fn get(&self, key: &[u8]) -> Option<V> {
        let idx = match &self.trie.storage {
            super::storage::TrieStorage::CompressedSparse(cspp) => {
                let valpos = cspp.lookup(key)?;
                cspp.get_value::<u32>(valpos) as usize
            }
            _ => self.trie.lookup_node_id(key)? as usize,
        };

        self.values.get(idx).and_then(|&v| v)
    }

    /// Check if a key exists in the map
    pub fn contains(&self, key: &[u8]) -> bool {
        self.trie.contains(key)
    }

    /// Get the number of key-value pairs
    pub fn len(&self) -> usize {
        self.trie.len()
    }

    /// Check if the map is empty
    pub fn is_empty(&self) -> bool {
        self.trie.is_empty()
    }

    /// Get all keys in the map
    pub fn keys(&self) -> Vec<Vec<u8>> {
        self.trie.keys()
    }
}

impl<V: Copy, R> Default for ZiporaTrieMap<V, R>
where
    R: RankSelectOps + Default,
{
    fn default() -> Self {
        Self::new()
    }
}
