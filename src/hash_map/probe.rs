//! Probe and CRUD internals for [`ZiporaHashMap`](super::ZiporaHashMap).

use super::ZiporaHashMap;
use crate::error::Result;
use crate::hash_map::storage::*;
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};
use std::mem::MaybeUninit;

/// Return a 16-bit mask of lanes in `ctrl[pos..pos + 16]` equal to `byte`.
#[inline(always)]
fn group_match_byte(ctrl_ptr: *const u8, pos: usize, byte: u8) -> u16 {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        use std::arch::x86_64::{_mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8, __m128i};
        // SAFETY: `pos < capacity` and `ctrl` allocates `capacity + 16` initialized bytes.
        unsafe {
            let group = _mm_loadu_si128(ctrl_ptr.add(pos) as *const __m128i);
            let cmp = _mm_cmpeq_epi8(group, _mm_set1_epi8(byte as i8));
            _mm_movemask_epi8(cmp) as u16
        }
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    {
        let mut mask = 0u16;
        for lane in 0..GROUP_WIDTH {
            // SAFETY: `pos + lane < capacity + 16 == ctrl.len()`.
            let c = unsafe { *ctrl_ptr.add(pos + lane) };
            if c == byte {
                mask |= 1u16 << lane;
            }
        }
        mask
    }
}

/// Return a 16-bit mask of lanes in `ctrl[pos..pos + 16]` equal to `CTRL_EMPTY`.
#[inline(always)]
fn group_match_empty(ctrl_ptr: *const u8, pos: usize) -> u16 {
    group_match_byte(ctrl_ptr, pos, CTRL_EMPTY)
}

/// Return a 16-bit mask of lanes in `ctrl[pos..pos + 16]` that are `CTRL_EMPTY` or `CTRL_DELETED` (MSB == 1).
#[inline(always)]
fn group_match_empty_or_deleted(ctrl_ptr: *const u8, pos: usize) -> u16 {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        use std::arch::x86_64::{_mm_loadu_si128, _mm_movemask_epi8, __m128i};
        // SAFETY: `pos < capacity` and `ctrl` allocates `capacity + 16` initialized bytes.
        unsafe {
            let group = _mm_loadu_si128(ctrl_ptr.add(pos) as *const __m128i);
            _mm_movemask_epi8(group) as u16
        }
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    {
        let mut mask = 0u16;
        for lane in 0..GROUP_WIDTH {
            // SAFETY: `pos + lane < capacity + 16 == ctrl.len()`.
            let c = unsafe { *ctrl_ptr.add(pos + lane) };
            if (c & 0x80) != 0 {
                mask |= 1u16 << lane;
            }
        }
        mask
    }
}

impl<K, V, S> ZiporaHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    /// Hash a key using the configured hasher
    #[inline(always)]
    pub(super) fn hash_key(&self, key: &K) -> u64 {
        self.hash_builder.hash_one(key)
    }

    /// Hash a borrowed key using the configured hasher
    #[inline(always)]
    pub(super) fn hash_key_borrowed<Q>(&self, key: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }

    /// Insert a unique `(key, value)` with precomputed `hash` into an allocated table with no duplicate check.
    #[inline(always)]
    fn insert_unique_unchecked(table: &mut StandardTable<K, V>, key: K, value: V, hash: u64) -> bool {
        let capacity = table.capacity();
        let mask = table.mask;
        let h2 = h2_tag(hash);
        let ctrl_ptr = table.ctrl.as_ptr();
        let mut pos = (hash as usize) & mask;
        let groups = (capacity >> 4).max(1);
        for _ in 0..groups {
            let avail = group_match_empty_or_deleted(ctrl_ptr, pos);
            if avail != 0 {
                let lane = avail.trailing_zeros() as usize;
                let idx = (pos + lane) & mask;
                table.set_ctrl(idx, h2);
                // SAFETY: `idx < capacity` and slot `idx` is currently empty/deleted.
                unsafe {
                    std::ptr::write(table.slots.as_mut_ptr().add(idx), MaybeUninit::new((key, value)));
                }
                return true;
            }
            pos = (pos + GROUP_WIDTH) & mask;
        }
        false
    }

    /// Helper to resize/rehash a `Standard` table in place.
    pub(super) fn resize_standard_entries(
        hash_builder: &S,
        table: &mut StandardTable<K, V>,
        live: usize,
        load_factor: f64,
        occupied: &mut usize,
        rehashes: &mut u64,
    ) -> Result<()> {
        let old_capacity = table.capacity();
        let load_limit = (old_capacity as f64 * load_factor) as usize;
        let new_capacity = if old_capacity >= 32 && live < load_limit / 2 {
            old_capacity
        } else {
            (old_capacity * 2).max(32)
        };
        assert!(
            new_capacity.is_power_of_two(),
            "hash map capacity must be a power of two for mask-based probing (got {})",
            new_capacity
        );

        let new_table = StandardTable::with_capacity_and_load_factor(new_capacity, load_factor)?;
        let mut old_table = std::mem::replace(table, new_table);

        for i in 0..old_capacity {
            // SAFETY: `i < old_capacity <= old_table.ctrl.len()`.
            let c = unsafe { *old_table.ctrl.as_ptr().add(i) };
            if is_ctrl_full(c) {
                // Mark empty before reading out `(k, v)` so unwinding on a panicking
                // `K::hash` cannot double-drop slot `i` when `old_table` drops.
                // SAFETY: `i < old_capacity` and `is_ctrl_full(c)` guarantees `slots[i]` is initialized.
                let (k, v) = unsafe {
                    *old_table.ctrl.as_mut_ptr().add(i) = CTRL_EMPTY;
                    std::ptr::read(old_table.slots.as_ptr().add(i)).assume_init()
                };
                let hash = hash_builder.hash_one(&k);
                if !Self::insert_unique_unchecked(table, k, v, hash) {
                    return Err(crate::error::ZiporaError::invalid_state(
                        "Failed to reinsert during resize",
                    ));
                }
            }
        }

        // `old_table` has all moved slots marked `CTRL_EMPTY`, so its `Drop` is a no-op.
        drop(old_table);

        *occupied = live;
        *rehashes += 1;
        Ok(())
    }

    /// Resize the storage to accommodate more elements
    pub(super) fn resize_storage(&mut self) -> Result<()> {
        let live = self.len;
        let load_factor = self.config.load_factor;
        match &mut self.storage {
            HashMapStorage::Standard(table) => Self::resize_standard_entries(
                &self.hash_builder,
                table,
                live,
                load_factor,
                &mut self.occupied,
                &mut self.stats.rehashes,
            ),
            HashMapStorage::SmallInline {
                fallback: Some(fb), ..
            } => match fb.as_mut() {
                HashMapStorage::Standard(table) => Self::resize_standard_entries(
                    &self.hash_builder,
                    table,
                    live,
                    load_factor,
                    &mut self.occupied,
                    &mut self.stats.rehashes,
                ),
                HashMapStorage::SmallInline { .. } => {
                    Err(crate::error::ZiporaError::invalid_state(
                        "Nested SmallInline fallback storage is invalid",
                    ))
                }
            },
            HashMapStorage::SmallInline { fallback: None, .. } => {
                Err(crate::error::ZiporaError::invalid_state(
                    "Resize not supported before SmallInline fallback migration",
                ))
            }
        }
    }

    #[inline]
    pub(super) fn insert_standard(
        _hash_builder: &S,
        table: &mut StandardTable<K, V>,
        key: K,
        value: V,
        hash: u64,
    ) -> std::result::Result<Option<V>, (K, V)> {
        let capacity = table.capacity();
        if capacity == 0 {
            return Err((key, value));
        }

        let mask = table.mask;
        let h2 = h2_tag(hash);
        let ctrl_ptr = table.ctrl.as_ptr();
        let slots_ptr = table.slots.as_mut_ptr();
        let mut pos = (hash as usize) & mask;
        let mut first_avail_idx: Option<usize> = None;
        let groups = (capacity >> 4).max(1);

        for _ in 0..groups {
            let mut matches = group_match_byte(ctrl_ptr, pos, h2);
            while matches != 0 {
                let lane = matches.trailing_zeros() as usize;
                let idx = (pos + lane) & mask;
                // SAFETY: `idx < capacity` and `ctrl[idx] == h2` (`< 0x80`), so `slots[idx]` is initialized.
                let (existing_k, existing_v) = unsafe { (*slots_ptr.add(idx)).assume_init_mut() };
                if existing_k == &key {
                    let old_v = std::mem::replace(existing_v, value);
                    return Ok(Some(old_v));
                }
                matches &= matches - 1;
            }

            if first_avail_idx.is_none() {
                let avail = group_match_empty_or_deleted(ctrl_ptr, pos);
                if avail != 0 {
                    let lane = avail.trailing_zeros() as usize;
                    first_avail_idx = Some((pos + lane) & mask);
                }
            }

            if group_match_empty(ctrl_ptr, pos) != 0 {
                break;
            }
            pos = (pos + GROUP_WIDTH) & mask;
        }

        if let Some(insert_idx) = first_avail_idx {
            table.set_ctrl(insert_idx, h2);
            // SAFETY: `insert_idx < capacity` and slot `insert_idx` is empty or deleted (uninitialized).
            unsafe {
                std::ptr::write(slots_ptr.add(insert_idx), MaybeUninit::new((key, value)));
            }
            Ok(None)
        } else {
            Err((key, value))
        }
    }

    #[allow(clippy::too_many_arguments)] // internal inline-storage probe helper; destructured fields avoid self borrow split
    pub(super) fn insert_small_inline(
        inline_data: &mut InlineStorage<K, V>,
        fallback: &mut Option<Box<HashMapStorage<K, V>>>,
        len: &mut usize,
        key: K,
        value: V,
        hash: u64,
        hash_builder: &S,
        load_factor: f64,
        occupied: &mut usize,
        rehashes: &mut u64,
    ) -> Result<Option<V>> {
        if let Some(fb) = fallback.as_mut()
            && let HashMapStorage::Standard(table) = fb.as_mut()
        {
            if table.capacity() > 0 && *occupied >= table.load_limit {
                Self::resize_standard_entries(
                    hash_builder,
                    table,
                    *len,
                    load_factor,
                    occupied,
                    rehashes,
                )?;
            }
            let result = match Self::insert_standard(hash_builder, table, key, value, hash) {
                Ok(res) => res,
                Err((retry_key, retry_val)) => {
                    Self::resize_standard_entries(
                        hash_builder,
                        table,
                        *len,
                        load_factor,
                        occupied,
                        rehashes,
                    )?;
                    Self::insert_standard(hash_builder, table, retry_key, retry_val, hash).map_err(
                        |_| {
                            crate::error::ZiporaError::invalid_state(
                                "Hash table full in SmallInline fallback storage after resize",
                            )
                        },
                    )?
                }
            };
            if result.is_none() {
                *len += 1;
            }
            return Ok(result);
        }

        // Check if key already exists in inline storage
        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit `i` is set in `inline_data.occupied`, so `_data[i]` is initialized.
                let matches = unsafe {
                    let (k, _) = inline_data._data[i].assume_init_ref();
                    k == &key
                };
                if matches {
                    // SAFETY: Slot i is initialized and no active references exist.
                    let (old_k, old_v) = unsafe { std::ptr::read(inline_data._data[i].as_ptr()) };
                    drop(old_k);
                    // SAFETY: Slot i is valid in `_data[0..16]` and was just read out above.
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
        let std_cap = 32;
        let mut table = StandardTable::with_capacity_and_load_factor(std_cap, load_factor)?;

        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit i is set in occupied, so slot i is initialized.
                // Clear the bit before hashing so panics in `hash_one` cannot double-drop.
                let (k, v) = unsafe { std::ptr::read(inline_data._data[i].as_ptr()) };
                inline_data.occupied &= !(1 << i);
                let h = hash_builder.hash_one(&k);
                let _ = Self::insert_standard(hash_builder, &mut table, k, v, h);
            }
        }
        inline_data.occupied = 0;

        let result = Self::insert_standard(hash_builder, &mut table, key, value, hash).map_err(
            |_| {
                crate::error::ZiporaError::invalid_state(
                    "Hash table full after SmallInline migration",
                )
            },
        )?;

        *fallback = Some(Box::new(HashMapStorage::Standard(table)));
        *occupied = *len;
        *len += 1;

        Ok(result)
    }

    #[inline]
    pub(super) fn get_standard<'a, Q>(
        &self,
        table: &'a StandardTable<K, V>,
        key: &Q,
        hash: u64,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let capacity = table.capacity();
        if capacity == 0 {
            return None;
        }

        let mask = table.mask;
        let h2 = h2_tag(hash);
        let ctrl_ptr = table.ctrl.as_ptr();
        let slots_ptr = table.slots.as_ptr();
        let mut pos = (hash as usize) & mask;
        let groups = (capacity >> 4).max(1);

        for _ in 0..groups {
            let mut matches = group_match_byte(ctrl_ptr, pos, h2);
            while matches != 0 {
                let lane = matches.trailing_zeros() as usize;
                let idx = (pos + lane) & mask;
                // SAFETY: `idx < capacity` and `ctrl[idx] == h2` (`< 0x80`), so `slots[idx]` is initialized.
                let (existing_k, existing_v) = unsafe { (*slots_ptr.add(idx)).assume_init_ref() };
                if existing_k.borrow() == key {
                    return Some(existing_v);
                }
                matches &= matches - 1;
            }
            if group_match_empty(ctrl_ptr, pos) != 0 {
                return None;
            }
            pos = (pos + GROUP_WIDTH) & mask;
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
        if let Some(fb) = fallback
            && let HashMapStorage::Standard(table) = fb.as_ref()
        {
            let hash = self.hash_key_borrowed(key);
            return self.get_standard(table, key, hash);
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

    pub(super) fn get_mut_standard<'a, Q>(
        hash_builder: &S,
        table: &'a mut StandardTable<K, V>,
        key: &Q,
    ) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let capacity = table.capacity();
        if capacity == 0 {
            return None;
        }

        let hash = hash_builder.hash_one(key);
        let mask = table.mask;
        let h2 = h2_tag(hash);
        let ctrl_ptr = table.ctrl.as_ptr();
        let slots_ptr = table.slots.as_mut_ptr();
        let mut pos = (hash as usize) & mask;
        let groups = (capacity >> 4).max(1);

        for _ in 0..groups {
            let mut matches = group_match_byte(ctrl_ptr, pos, h2);
            while matches != 0 {
                let lane = matches.trailing_zeros() as usize;
                let idx = (pos + lane) & mask;
                // SAFETY: `idx < capacity` and `ctrl[idx] == h2` (`< 0x80`), so `slots[idx]` is initialized.
                let k_matches = unsafe {
                    let (existing_k, _) = (*slots_ptr.add(idx)).assume_init_ref();
                    existing_k.borrow() == key
                };
                if k_matches {
                    // SAFETY: `idx < capacity` is initialized and we return immediately with `'a` borrow.
                    let (_, existing_v) = unsafe { (*slots_ptr.add(idx)).assume_init_mut() };
                    return Some(existing_v);
                }
                matches &= matches - 1;
            }
            if group_match_empty(ctrl_ptr, pos) != 0 {
                return None;
            }
            pos = (pos + GROUP_WIDTH) & mask;
        }

        None
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
            && let HashMapStorage::Standard(table) = fb.as_mut()
        {
            return Self::get_mut_standard(hash_builder, table, key);
        }

        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit `i` is set in `inline_data.occupied`, so `_data[i]` is initialized.
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

    pub(super) fn remove_standard<Q>(
        hash_builder: &S,
        table: &mut StandardTable<K, V>,
        key: &Q,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let capacity = table.capacity();
        if capacity == 0 {
            return None;
        }

        let hash = hash_builder.hash_one(key);
        let mask = table.mask;
        let h2 = h2_tag(hash);
        let ctrl_ptr = table.ctrl.as_ptr();
        let slots_ptr = table.slots.as_mut_ptr();
        let mut pos = (hash as usize) & mask;
        let groups = (capacity >> 4).max(1);

        for _ in 0..groups {
            let mut matches = group_match_byte(ctrl_ptr, pos, h2);
            while matches != 0 {
                let lane = matches.trailing_zeros() as usize;
                let idx = (pos + lane) & mask;
                // SAFETY: `idx < capacity` and `ctrl[idx] == h2` (`< 0x80`), so `slots[idx]` is initialized.
                let k_matches = unsafe {
                    let (existing_k, _) = (*slots_ptr.add(idx)).assume_init_ref();
                    existing_k.borrow() == key
                };
                if k_matches {
                    // SAFETY: `idx < capacity` is initialized; we mark it deleted and read out `(k, v)`.
                    let (_old_k, old_v) = unsafe { std::ptr::read(slots_ptr.add(idx)).assume_init() };
                    table.set_ctrl(idx, CTRL_DELETED);
                    return Some(old_v);
                }
                matches &= matches - 1;
            }
            if group_match_empty(ctrl_ptr, pos) != 0 {
                return None;
            }
            pos = (pos + GROUP_WIDTH) & mask;
        }

        None
    }

    /// Backward shift deletion to compact a contiguous cluster in `StandardTable`.
    #[cfg(test)]
    pub(super) fn backward_shift_delete(
        table: &mut StandardTable<K, V>,
        hash_builder: &S,
        mut pos: usize,
    ) where
        K: Clone,
        V: Clone,
    {
        let mask = table.mask;
        if is_ctrl_full(table.ctrl[pos]) {
            // SAFETY: `pos < table.capacity()` and `is_ctrl_full` guarantees `slots[pos]` is initialized.
            unsafe {
                std::ptr::drop_in_place(table.slots[pos].as_mut_ptr());
            }
        }
        table.set_ctrl(pos, CTRL_EMPTY);

        loop {
            let next_pos = (pos + 1) & mask;
            let next_ctrl = table.ctrl[next_pos];
            if !is_ctrl_full(next_ctrl) {
                break;
            }
            // SAFETY: `next_pos < table.capacity()` and `is_ctrl_full(next_ctrl)` guarantees initialization.
            let ideal_pos = unsafe {
                let (next_k, _) = table.slots[next_pos].assume_init_ref();
                (hash_builder.hash_one(next_k) as usize) & mask
            };
            let can_move = if pos < next_pos {
                ideal_pos <= pos || ideal_pos > next_pos
            } else {
                ideal_pos <= pos && ideal_pos > next_pos
            };
            if !can_move {
                break;
            }
            // SAFETY: `next_pos` is initialized and `pos` is currently empty (`CTRL_EMPTY`).
            unsafe {
                let moved = std::ptr::read(table.slots.as_ptr().add(next_pos));
                std::ptr::write(table.slots.as_mut_ptr().add(pos), moved);
            }
            table.set_ctrl(pos, next_ctrl);
            table.set_ctrl(next_pos, CTRL_EMPTY);
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
            && let HashMapStorage::Standard(table) = fb.as_mut()
        {
            let result = Self::remove_standard(hash_builder, table, key);
            if result.is_some() {
                *len -= 1;
            }
            return result;
        }

        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit `i` is set in `inline_data.occupied`, so `_data[i]` is initialized.
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

    pub(super) fn clear_standard(table: &mut StandardTable<K, V>) {
        table.clear_slots();
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
