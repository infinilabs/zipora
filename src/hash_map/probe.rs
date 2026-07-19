//! Probe and CRUD internals for [`ZiporaHashMap`](super::ZiporaHashMap).

use crate::containers::FastVec;
use crate::error::Result;
use crate::hash_map::storage::*;
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};
use super::ZiporaHashMap;

impl<K, V, S> ZiporaHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    /// Hash a key using the configured hasher
    pub(super) fn hash_key(&self, key: &K) -> u64 {
        let h = self.hash_builder.hash_one(key);
        if h == 0 {
            1
        } else if h == u64::MAX {
            u64::MAX - 1
        } else {
            h
        }
    }

    /// Hash a borrowed key using the configured hasher
    pub(super) fn hash_key_borrowed<Q>(&self, key: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        let h = self.hash_builder.hash_one(key);
        if h == 0 {
            1
        } else if h == u64::MAX {
            u64::MAX - 1
        } else {
            h
        }
    }

    /// Resize the storage to accommodate more elements
    pub(super) fn resize_storage(&mut self) -> Result<()> {
        match &mut self.storage {
            HashMapStorage::Standard {
                buckets: _,
                entries,
                mask,
            } => {
                let old_capacity = entries.len();
                let new_capacity = (old_capacity * 2).max(32); // At least double the size

                // Create new larger storage
                let mut new_entries: FastVec<HashEntry<K, V>> =
                    FastVec::with_capacity(new_capacity)?;

                // Initialize new empty entries
                // SAFETY: `new_capacity` is within bounds as it was just successfully allocated.
                // All elements 0..new_capacity are immediately initialized via `ptr::write`.
                unsafe {
                    new_entries.set_len(new_capacity);
                }
                for i in 0..new_capacity {
                    // SAFETY: `new_entries` has capacity `new_capacity`. `i` < `new_capacity`.
                    // It is safe to write to this uninitialized memory.
                    unsafe {
                        std::ptr::write(
                            new_entries.as_mut_ptr().add(i),
                            HashEntry {
                                key: None,
                                value: None,
                                hash: 0,
                                _next: None,
                            },
                        );
                    }
                }

                let new_mask = new_capacity - 1;

                // Move existing entries
                let mut old_entries = std::mem::replace(entries, new_entries);

                for entry in old_entries.iter_mut() {
                    // Skip empty slots AND tombstones (u64::MAX)
                    if entry.hash != 0 && entry.hash != u64::MAX {
                        let index = (entry.hash as usize) & new_mask;

                        // Find empty slot with linear probing
                        let mut inserted = false;
                        for i in 0..new_capacity {
                            let probe_index = (index + i) & new_mask;
                            let new_entry = &mut entries[probe_index];

                            if new_entry.hash == 0 {
                                // Empty slot, insert here
                                new_entry.key = entry.key.take();
                                new_entry.value = entry.value.take();
                                new_entry.hash = entry.hash;
                                inserted = true;
                                break;
                            }
                        }

                        if !inserted {
                            return Err(crate::error::ZiporaError::invalid_state(
                                "Failed to reinsert during resize",
                            ));
                        }
                    }
                }

                // old_entries is dropped here, which will drop the None keys/values harmlessly.

                *mask = new_mask;
                self.stats.rehashes += 1;

                Ok(())
            }
            _ => {
                // Other storage types don't support resizing yet
                Err(crate::error::ZiporaError::invalid_state(
                    "Resize not supported for this storage type",
                ))
            }
        }
    }

    // Implementation methods for different storage strategies
    pub(super) fn insert_standard(
        _hash_builder: &S,
        _buckets: &mut FastVec<StandardBucket<K, V>>,
        entries: &mut FastVec<HashEntry<K, V>>,
        mask: &mut usize,
        key: K,
        value: V,
        hash: u64,
    ) -> std::result::Result<Option<V>, (K, V)> {
        // Initialize entries if empty
        if entries.is_empty() {
            let capacity = entries.capacity();
            if capacity == 0 {
                return Err((key, value)); // Trigger resize to allocate
            }
            // SAFETY: `capacity` is the actual allocated capacity.
            // Elements 0..capacity will be immediately initialized by `ptr::write`.
            unsafe {
                entries.set_len(capacity);
            }
            for i in 0..capacity {
                // SAFETY: `entries` capacity is `capacity`. `i` < `capacity`.
                // Thus `as_mut_ptr().add(i)` is valid and within bounds.
                unsafe {
                    std::ptr::write(
                        entries.as_mut_ptr().add(i),
                        HashEntry {
                            key: None,
                            value: None,
                            hash: 0,
                            _next: None,
                        },
                    );
                }
            }
            *mask = capacity - 1;
        }

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        let mut first_tombstone_idx = None;
        let mut found_existing_idx = None;
        let mut empty_slot_idx = None;

        // Linear probing to find slot status
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &entries[probe_index];

            if entry.hash == 0 {
                empty_slot_idx = Some(probe_index);
                break;
            } else if entry.hash == u64::MAX {
                if first_tombstone_idx.is_none() {
                    first_tombstone_idx = Some(probe_index);
                }
            } else if entry.hash == hash
                && entry.key.as_ref().expect("occupied entry must have key") == &key
            {
                found_existing_idx = Some(probe_index);
                break;
            }
        }

        if let Some(idx) = found_existing_idx {
            // Key exists, update value
            let entry = &mut entries[idx];
            let old_value = entry
                .value
                .replace(value)
                .expect("occupied entry must have previous value");
            Ok(Some(old_value))
        } else if let Some(insert_idx) = first_tombstone_idx.or(empty_slot_idx) {
            // Insert at first tombstone or empty slot
            let entry = &mut entries[insert_idx];
            entry.key = Some(key);
            entry.value = Some(value);
            entry.hash = hash;
            Ok(None)
        } else {
            // Table is full, need to resize
            Err((key, value))
        }
    }

    pub(super) fn insert_small_inline(
        inline_data: &mut InlineStorage<K, V>,
        _fallback: &mut Option<Box<HashMapStorage<K, V>>>,
        len: &mut usize,
        key: K,
        value: V,
        hash: u64,
        hash_builder: &S,
    ) -> Result<Option<V>> {
        // If already migrated to fallback, delegate to Standard storage
        if let Some(fb) = _fallback.as_mut()
            && let HashMapStorage::Standard {
                buckets,
                entries,
                mask,
                ..
            } = fb.as_mut()
        {
                let result =
                    Self::insert_standard(hash_builder, buckets, entries, mask, key, value, hash)
                        .map_err(|_| {
                        crate::error::ZiporaError::invalid_state(
                            "Hash table full in SmallInline fallback storage",
                        )
                    })?;
                if result.is_none() {
                    *len += 1;
                }
                return Ok(result);
            }

        // Check if key already exists in inline storage
        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                let matches = unsafe {
                    let (k, _) = inline_data._data[i].assume_init_ref();
                    k == &key
                };
                if matches {
                    // SAFETY: Slot i is initialized and no active references exist.
                    let (old_k, old_v) = unsafe { std::ptr::read(inline_data._data[i].as_ptr()) };
                    drop(old_k);
                    unsafe {
                        std::ptr::write(inline_data._data[i].as_mut_ptr(), (key, value));
                    }
                    return Ok(Some(old_v));
                }
            }
        }

        // Try to find an empty slot
        if inline_data.occupied != 0xFFFF {
            let slot = inline_data.occupied.trailing_ones() as usize;
            // SAFETY: slot < 16 and is currently uninitialized (bit is 0)
            unsafe {
                std::ptr::write(inline_data._data[slot].as_mut_ptr(), (key, value));
            }
            inline_data.occupied |= 1 << slot;
            *len += 1;
            return Ok(None);
        }

        // Inline storage full — migrate all 16 entries to Standard storage.
        let std_cap = 32; // 16 existing + room to grow
        let mut buckets = FastVec::with_capacity(std_cap)?;
        let mut entries = FastVec::with_capacity(std_cap)?;
        let mut mask = std_cap - 1;

        // Initialize entries
        for _ in 0..std_cap {
            entries.push(HashEntry {
                key: None,
                value: None,
                hash: 0,
                _next: None,
            })?;
        }
        // buckets are allocated but not initialized — Standard path uses entries for probing

        // Re-insert all 16 inline entries into standard storage
        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit i is set in occupied, so slot i is initialized
                let (k, v) = unsafe { std::ptr::read(inline_data._data[i].as_ptr()) };
                inline_data.occupied &= !(1 << i);
                let raw_h = hash_builder.hash_one(&k);
                let h = if raw_h == 0 {
                    1
                } else if raw_h == u64::MAX {
                    u64::MAX - 1
                } else {
                    raw_h
                };
                let _ = Self::insert_standard(
                    hash_builder,
                    &mut buckets,
                    &mut entries,
                    &mut mask,
                    k,
                    v,
                    h,
                );
            }
        }
        inline_data.occupied = 0;

        // Insert the new key-value pair
        let result = Self::insert_standard(
            hash_builder,
            &mut buckets,
            &mut entries,
            &mut mask,
            key,
            value,
            hash,
        );
        let result = result.map_err(|_| {
            crate::error::ZiporaError::invalid_state("Hash table full after SmallInline migration")
        })?;

        // Store the migrated storage as fallback
        *_fallback = Some(Box::new(HashMapStorage::Standard {
            buckets,
            entries,
            mask,
        }));
        *len += 1;

        Ok(result)
    }

    pub(super) fn get_standard<'a, Q>(
        &self,
        _buckets: &FastVec<StandardBucket<K, V>>,
        entries: &'a FastVec<HashEntry<K, V>>,
        mask: &usize,
        key: &Q,
        hash: u64,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if entries.is_empty() {
            return None;
        }

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Linear probing to find key
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &entries[probe_index];

            if entry.hash == 0 {
                // Empty slot, key not found
                return None;
            } else if entry.hash == u64::MAX {
                // Tombstone, skip and continue searching
                continue;
            } else if entry.hash == hash
                && entry
                    .key
                    .as_ref()
                    .expect("occupied entry must have key")
                    .borrow()
                    == key
            {
                // Found the key
                return Some(
                    entry
                        .value
                        .as_ref()
                        .expect("occupied entry must have value"),
                );
            }
        }

        None
    }

    pub(super) fn get_small_inline<'a, Q>(
        &self,
        inline_data: &'a InlineStorage<K, V>,
        fallback: &'a Option<Box<HashMapStorage<K, V>>>,
        _len: &usize,
        key: &Q,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // Check fallback first (migrated data)
        if let Some(fb) = fallback
            && let HashMapStorage::Standard {
                buckets,
                entries,
                mask,
            } = fb.as_ref()
        {
            let hash = self.hash_key_borrowed(key);
            return self.get_standard(buckets, entries, mask, key, hash);
        }

        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit i is set in occupied, so slot i is initialized
                let (k, v) = unsafe { inline_data._data[i].assume_init_ref() };
                if k.borrow() == key {
                    return Some(v);
                }
            }
        }
        None
    }

    // get_mut implementation methods
    pub(super) fn get_mut_standard<'a, Q>(
        hash_builder: &S,
        _buckets: &'a mut FastVec<StandardBucket<K, V>>,
        entries: &'a mut FastVec<HashEntry<K, V>>,
        mask: &mut usize,
        key: &Q,
    ) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if entries.is_empty() {
            return None;
        }

        let h = hash_builder.hash_one(key);
        let hash = if h == 0 {
            1
        } else if h == u64::MAX {
            u64::MAX - 1
        } else {
            h
        };

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Find the index first
        let mut found_index = None;
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &entries[probe_index]; // Immutable borrow for checking

            if entry.hash == 0 {
                // Empty slot, key not found
                break;
            } else if entry.hash == u64::MAX {
                // Tombstone, skip and continue searching
                continue;
            } else if entry.hash == hash
                && entry
                    .key
                    .as_ref()
                    .expect("occupied entry must have key")
                    .borrow()
                    == key
            {
                // Found the key
                found_index = Some(probe_index);
                break;
            }
        }

        // Return mutable reference if found
        if let Some(idx) = found_index {
            Some(
                entries[idx]
                    .value
                    .as_mut()
                    .expect("occupied entry must have value"),
            )
        } else {
            None
        }
    }

    pub(super) fn get_mut_small_inline<'a, Q>(
        hash_builder: &S,
        inline_data: &'a mut InlineStorage<K, V>,
        fallback: &'a mut Option<Box<HashMapStorage<K, V>>>,
        _len: &mut usize,
        key: &Q,
    ) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(fb) = fallback.as_mut()
            && let HashMapStorage::Standard {
                buckets,
                entries,
                mask,
            } = fb.as_mut()
        {
            return Self::get_mut_standard(hash_builder, buckets, entries, mask, key);
        }

        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                let matches = unsafe {
                    let (k, _) = inline_data._data[i].assume_init_ref();
                    k.borrow() == key
                };
                if matches {
                    // SAFETY: Slot i is initialized and no active references exist
                    let (_, v) = unsafe { inline_data._data[i].assume_init_mut() };
                    return Some(v);
                }
            }
        }
        None
    }

    // remove implementation methods
    pub(super) fn remove_standard<Q>(
        hash_builder: &S,
        _buckets: &mut FastVec<StandardBucket<K, V>>,
        entries: &mut FastVec<HashEntry<K, V>>,
        mask: &mut usize,
        key: &Q,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if entries.is_empty() {
            return None;
        }

        let h = hash_builder.hash_one(key);
        let hash = if h == 0 {
            1
        } else if h == u64::MAX {
            u64::MAX - 1
        } else {
            h
        };

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Linear probing to find key
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &mut entries[probe_index];

            if entry.hash == 0 {
                // Empty slot, key not found
                return None;
            } else if entry.hash == hash
                && entry
                    .key
                    .as_ref()
                    .expect("occupied entry must have key")
                    .borrow()
                    == key
            {
                // Found the key, remove it
                let old_value = entry.value.take().expect("occupied entry must have value");
                entry.key.take(); // free the key

                // Use tombstone approach: mark as deleted but don't create holes
                entry.hash = u64::MAX; // Special tombstone marker

                return Some(old_value);
            }
        }

        None
    }

    /// Backward shift deletion to maintain linear probing invariant
    #[cfg(test)]
    pub(super) fn backward_shift_delete(entries: &mut FastVec<HashEntry<K, V>>, mask: usize, mut pos: usize)
    where
        K: Clone,
        V: Clone,
    {
        // Clear the removed entry
        entries[pos].hash = 0;

        loop {
            let next_pos = (pos + 1) & mask;
            let next_entry = &entries[next_pos];

            // Stop if next entry is empty
            if next_entry.hash == 0 {
                break;
            }

            // Calculate the ideal position for the next entry
            let ideal_pos = (next_entry.hash as usize) & mask;

            // Check if we can move this entry backward
            // We can move it if its ideal position would still allow it to be found
            // after the move. This happens when:
            // - The ideal position is at or before the empty slot, OR
            // - The entry is displaced and moving it backward doesn't break the probe sequence

            let can_move = if ideal_pos <= pos {
                // Ideal position is before the empty slot - safe to move
                true
            } else {
                // Entry is displaced. Check if moving backward maintains findability.
                // In a wrapping hash table, we need to consider wrap-around cases.
                // The entry can be moved if the ideal position is between the current
                // empty position and the entry's current position (considering wrap-around).

                if pos < next_pos {
                    // No wrap-around case: ideal should be between pos and next_pos
                    ideal_pos > pos && ideal_pos <= next_pos
                } else {
                    // Wrap-around case: ideal can be after pos or before next_pos
                    ideal_pos > pos || ideal_pos <= next_pos
                }
            };

            if !can_move {
                break;
            }

            // Move the entry backward using swap (no cloning needed!)
            entries.swap(pos, next_pos);
            entries[next_pos].hash = 0; // Mark the old position as empty
            entries[next_pos].key = None;
            entries[next_pos].value = None;

            pos = next_pos;
        }
    }

    pub(super) fn remove_small_inline<Q>(
        hash_builder: &S,
        inline_data: &mut InlineStorage<K, V>,
        fallback: &mut Option<Box<HashMapStorage<K, V>>>,
        len: &mut usize,
        key: &Q,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(fb) = fallback.as_mut()
            && let HashMapStorage::Standard {
                buckets,
                entries,
                mask,
            } = fb.as_mut()
        {
            let result = Self::remove_standard(hash_builder, buckets, entries, mask, key);
            if result.is_some() {
                *len -= 1;
            }
            return result;
        }

        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                let matches = unsafe {
                    let (k, _) = inline_data._data[i].assume_init_ref();
                    k.borrow() == key
                };
                if matches {
                    // SAFETY: Slot i is initialized and no active references exist. Read out key and value tuple.
                    let (_k_val, v_val) = unsafe { std::ptr::read(inline_data._data[i].as_ptr()) };
                    inline_data.occupied &= !(1 << i);
                    *len -= 1;
                    return Some(v_val);
                }
            }
        }

        None
    }

    // clear implementation methods
    pub(super) fn clear_standard(
        buckets: &mut FastVec<StandardBucket<K, V>>,
        entries: &mut FastVec<HashEntry<K, V>>,
        mask: &mut usize,
    ) {
        buckets.clear();
        entries.clear();
        *mask = 0;
    }

    pub(super) fn clear_small_inline(
        inline_data: &mut InlineStorage<K, V>,
        fallback: &mut Option<Box<HashMapStorage<K, V>>>,
        len: &mut usize,
    ) {
        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit i is set in occupied, so slot i is initialized
                unsafe {
                    std::ptr::drop_in_place(inline_data._data[i].as_mut_ptr());
                }
            }
        }
        inline_data.occupied = 0;
        *fallback = None;
        *len = 0;
    }
}
