//! Internal storage layout types for [`ZiporaHashMap`](super::ZiporaHashMap).

use crate::containers::FastVec;
use crate::hash_map::cache_locality::{CacheOptimizedBucket, Prefetcher};
use std::mem::MaybeUninit;

/// Internal storage implementations for different strategies
#[allow(dead_code)] // strategy-placeholder variants, matched exhaustively in 9 arms
pub(super) enum HashMapStorage<K, V>
where
    K: Clone,
    V: Clone,
{
    /// Standard FastVec-based storage
    Standard {
        buckets: FastVec<StandardBucket<K, V>>,
        entries: FastVec<HashEntry<K, V>>,
        mask: usize,
    },
    /// Inline storage for small collections
    SmallInline {
        inline_data: InlineStorage<K, V>,
        fallback: Option<Box<HashMapStorage<K, V>>>,
        len: usize,
    },
    /// Cache-optimized storage with alignment
    CacheOptimized {
        buckets: FastVec<CacheOptimizedBucket<K, V>>,
        hot_data: FastVec<K>,
        cold_data: FastVec<V>,
        prefetcher: Prefetcher,
    },
    /// String-specialized storage with arena
    StringOptimized {
        arena: StringArena,
        buckets: FastVec<StringBucket>,
        entries: FastVec<StringEntry<V>>,
        prefix_cache: FastVec<PrefixCacheEntry>,
    },
}

/// Standard hash table bucket
#[repr(align(64))]
pub(super) struct StandardBucket<K, V> {
    pub(super) _hash: u64,
    pub(super) _key: K,
    pub(super) _value: V,
    pub(super) _probe_distance: u16,
    pub(super) _is_occupied: bool,
}

/// Inline storage for small hash maps
pub(super) struct InlineStorage<K, V> {
    pub(super) _data: [MaybeUninit<(K, V)>; 16], // Fixed size for simplicity
    pub(super) occupied: u16,                    // Bit mask for occupied slots
}

impl<K, V> InlineStorage<K, V> {
}

/// String arena for interned strings
pub(super) struct StringArena {
    pub(super) _data: FastVec<u8>,
    pub(super) _offsets: FastVec<u32>,
    pub(super) _interned: std::collections::HashMap<Vec<u8>, u32>,
}

/// String bucket with prefix caching
pub(super) struct StringBucket {
    pub(super) _hash: u64,
    pub(super) _string_id: u32,
    pub(super) _probe_distance: u16,
    pub(super) _prefix_cache: u32,
}

/// String entry with value
pub(super) struct StringEntry<V> {
    pub(super) _value: V,
    pub(super) _next: Option<u32>,
}

/// Prefix cache entry for fast string matching
pub(super) struct PrefixCacheEntry {
    pub(super) _prefix: u64, // First 8 bytes of string
    pub(super) _string_id: u32,
}

/// Hash entry for standard storage
pub(super) struct HashEntry<K, V> {
    pub(super) key: Option<K>,
    pub(super) value: Option<V>,
    pub(super) hash: u64,
    pub(super) _next: Option<u32>,
}
