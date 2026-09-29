//! Internal storage layout types for [`ZiporaHashMap`](super::ZiporaHashMap).

use crate::containers::FastVec;
use std::mem::MaybeUninit;

/// Internal storage implementations for different strategies
pub(super) enum HashMapStorage<K, V>
where
    K: Clone,
    V: Clone,
{
    /// Standard FastVec-based storage
    Standard {
        entries: FastVec<HashEntry<K, V>>,
        mask: usize,
    },
    /// Inline storage for small collections
    SmallInline {
        inline_data: InlineStorage<K, V>,
        fallback: Option<Box<HashMapStorage<K, V>>>,
        len: usize,
    },
}

/// Inline storage for small hash maps
pub(super) struct InlineStorage<K, V> {
    pub(super) _data: [MaybeUninit<(K, V)>; 16], // Fixed size for simplicity
    pub(super) occupied: u16,                    // Bit mask for occupied slots
}

impl<K, V> InlineStorage<K, V> {
}

impl<K, V> Drop for InlineStorage<K, V> {
    fn drop(&mut self) {
        for i in 0..16 {
            if (self.occupied >> i) & 1 == 1 {
                // SAFETY: Bit i is set in occupied, so slot i is initialized
                unsafe {
                    std::ptr::drop_in_place(self._data[i].as_mut_ptr());
                }
            }
        }
    }
}

/// Hash entry for standard storage
pub(super) struct HashEntry<K, V> {
    pub(super) key: Option<K>,
    pub(super) value: Option<V>,
    pub(super) hash: u64,
    pub(super) _next: Option<u32>,
}
