//! Internal storage layout types for [`ZiporaHashMap`](super::ZiporaHashMap).

use crate::containers::FastVec;
use crate::error::Result;
use std::mem::MaybeUninit;

/// Control byte for an empty slot (`0b1000_0000`).
pub(super) const CTRL_EMPTY: u8 = 0x80;
/// Control byte for a deleted/tombstone slot (`0b1111_1110`).
pub(super) const CTRL_DELETED: u8 = 0xFE;
/// Number of control bytes probed in parallel per SSE2 group.
pub(super) const GROUP_WIDTH: usize = 16;

/// Extract the 7-bit secondary hash tag (`0x00..=0x7F`, MSB always 0).
#[inline(always)]
pub(super) const fn h2_tag(hash: u64) -> u8 {
    ((hash >> 57) as u8) & 0x7F
}

/// Return `true` if `ctrl` marks an occupied slot (`0x00..=0x7F`).
#[inline(always)]
pub(super) const fn is_ctrl_full(ctrl: u8) -> bool {
    (ctrl & 0x80) == 0
}

/// SwissTable-style flat control-byte + uninitialized-slot storage.
///
/// Memory layout: `1` control byte + `size_of::<(K, V)>()` per slot (`17 B/slot`
/// for `(u64, u64)`, `25 B/slot` for `([u8; 16], u64)`), plus `16` mirrored tail
/// control bytes so 16-byte SIMD group loads never wrap past the allocation.
pub(super) struct StandardTable<K, V> {
    pub(super) ctrl: FastVec<u8>,
    pub(super) slots: FastVec<MaybeUninit<(K, V)>>,
    pub(super) mask: usize,
    pub(super) load_limit: usize,
}

impl<K, V> StandardTable<K, V> {
    /// Allocate a flat SwissTable with `capacity` slots (`0` or a power of two `>= 16`).
    pub(super) fn with_capacity_and_load_factor(capacity: usize, load_factor: f64) -> Result<Self> {
        if capacity == 0 {
            return Ok(Self {
                ctrl: FastVec::new(),
                slots: FastVec::new(),
                mask: 0,
                load_limit: 0,
            });
        }
        let cap = capacity.max(GROUP_WIDTH);
        debug_assert!(cap.is_power_of_two());
        let ctrl_len = cap + GROUP_WIDTH;
        let mut ctrl = FastVec::with_capacity(ctrl_len)?;
        ctrl.resize(ctrl_len, CTRL_EMPTY)?;
        let mut slots = FastVec::with_capacity(cap)?;
        // SAFETY: `MaybeUninit<(K, V)>` requires no initialization; `slots` has capacity `cap`.
        unsafe {
            slots.set_len(cap);
        }
        let load_limit = (cap as f64 * load_factor) as usize;
        Ok(Self {
            ctrl,
            slots,
            mask: cap - 1,
            load_limit,
        })
    }

    #[inline(always)]
    pub(super) fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Write `byte` to `ctrl[idx]` and update the mirrored tail byte when `idx < GROUP_WIDTH`.
    #[inline(always)]
    pub(super) fn set_ctrl(&mut self, idx: usize, byte: u8) {
        let mask = self.mask;
        let mirror_idx = ((idx.wrapping_sub(GROUP_WIDTH)) & mask) + GROUP_WIDTH;
        // SAFETY: `idx < self.slots.len()` and `mirror_idx < self.slots.len() + GROUP_WIDTH == self.ctrl.len()`.
        unsafe {
            let ptr = self.ctrl.as_mut_ptr();
            *ptr.add(idx) = byte;
            *ptr.add(mirror_idx) = byte;
        }
    }

    /// Drop all live entries and reset all control bytes to `CTRL_EMPTY` in place.
    pub(super) fn clear_slots(&mut self) {
        let cap = self.slots.len();
        if cap == 0 {
            return;
        }
        if std::mem::needs_drop::<(K, V)>() {
            // Clear mirrored tail bytes upfront and mark each slot CTRL_EMPTY before
            // dropping its value so that if `K::drop` or `V::drop` panics, unwinding
            // will not drop already-dropped slots a second time while still dropping
            // all remaining live slots (`i + 1..cap`).
            self.ctrl.as_mut_slice()[cap..].fill(CTRL_EMPTY);
            let ctrl_ptr = self.ctrl.as_mut_ptr();
            let slots_ptr = self.slots.as_mut_ptr();
            for i in 0..cap {
                // SAFETY: `i < cap` and `is_ctrl_full` guarantees `slots[i]` is initialized.
                unsafe {
                    let c = *ctrl_ptr.add(i);
                    *ctrl_ptr.add(i) = CTRL_EMPTY;
                    if is_ctrl_full(c) {
                        std::ptr::drop_in_place((*slots_ptr.add(i)).as_mut_ptr());
                    }
                }
            }
        } else {
            self.ctrl.as_mut_slice().fill(CTRL_EMPTY);
        }
    }
}

impl<K, V> Drop for StandardTable<K, V> {
    fn drop(&mut self) {
        let cap = self.slots.len();
        if cap > 0 && std::mem::needs_drop::<(K, V)>() {
            let ctrl_ptr = self.ctrl.as_mut_ptr();
            let slots_ptr = self.slots.as_mut_ptr();
            for i in 0..cap {
                // SAFETY: `i < cap` and `is_ctrl_full` guarantees `slots[i]` is initialized.
                unsafe {
                    let c = *ctrl_ptr.add(i);
                    *ctrl_ptr.add(i) = CTRL_EMPTY;
                    if is_ctrl_full(c) {
                        std::ptr::drop_in_place((*slots_ptr.add(i)).as_mut_ptr());
                    }
                }
            }
        }
    }
}

/// Internal storage implementations for different strategies
pub(super) enum HashMapStorage<K, V>
where
    K: Clone,
    V: Clone,
{
    /// Standard SwissTable flat control-byte storage
    Standard(StandardTable<K, V>),
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

impl<K, V> InlineStorage<K, V> {}

impl<K, V> Drop for InlineStorage<K, V> {
    fn drop(&mut self) {
        for i in 0..16 {
            if (self.occupied >> i) & 1 == 1 {
                self.occupied &= !(1 << i);
                // SAFETY: Bit i was set in occupied, so slot i is initialized
                unsafe {
                    std::ptr::drop_in_place(self._data[i].as_mut_ptr());
                }
            }
        }
    }
}
