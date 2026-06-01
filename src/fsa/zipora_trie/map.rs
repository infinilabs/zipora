use super::config::ZiporaTrieConfig;
use super::trie::ZiporaTrie;
use crate::StateId;
use crate::error::Result;
use crate::fsa::traits::{FiniteStateAutomaton, Trie};
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

        // Ensure values vec is large enough
        let idx = state_id as usize;
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
        // First check if the key exists in the trie
        if !self.trie.contains(key) {
            return None;
        }

        // Find the state ID for this key by traversing
        // For now, we need to traverse to find the state ID
        // This is a simple O(key_length) traversal
        let state_id = self.find_state_for_key(key)?;

        // Return the value at that state
        self.values.get(state_id as usize).and_then(|&v| v)
    }

    /// Helper to find the state ID for a key
    fn find_state_for_key(&self, key: &[u8]) -> Option<StateId> {
        let mut state = self.trie.root();
        for &symbol in key {
            state = self.trie.transition(state, symbol)?;
        }
        Some(state)
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
