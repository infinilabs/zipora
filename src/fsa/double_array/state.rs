use crate::error::{Result, ZiporaError};
use std::cmp::Ordering;

/// 8-byte state matching C++ DA_State8B exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct DaState {
    /// Base for XOR transitions (terminal flag stored in NInfo, not here).
    pub(crate) child0: u32,
    /// Check/parent: bits 0-30 = parent state, bit 31 = free bit
    pub(crate) parent: u32,
}

// Bit constants
pub(crate) const FREE_BIT: u32 = 0x8000_0000;
pub(crate) const VALUE_MASK: u32 = 0x7FFF_FFFF;
pub(crate) const NIL_STATE: u32 = 0x7FFF_FFFF;
pub(crate) const MAX_STATE: u32 = 0x7FFF_FFFE;

impl DaState {
    /// New free state (matching C++ constructor)
    #[inline(always)]
    pub(crate) const fn new_free() -> Self {
        Self {
            child0: NIL_STATE,            // No children, no terminal
            parent: NIL_STATE | FREE_BIT, // Free
        }
    }

    /// New root state. child0=0 is a safe "no children" base: 0 ^ ch = ch,
    /// always in-bounds (array >= 256). Parent = NIL_STATE prevents false
    /// positive when child0=0 and ch=0 (0 ^ 0 = 0 → states[0].parent check).
    #[inline(always)]
    pub(crate) const fn new_root() -> Self {
        Self {
            child0: 0,         // Safe leaf base: 0 ^ ch always in [0, 255]
            parent: NIL_STATE, // Sentinel — never matches any valid curr
        }
    }

    /// Base value for XOR transitions. Raw read — no masking needed
    /// because the terminal flag is stored in NInfo, not here.
    #[inline(always)]
    pub(crate) fn child0(&self) -> u32 {
        self.child0
    }

    #[inline(always)]
    pub(crate) fn parent(&self) -> u32 {
        self.parent & VALUE_MASK
    }

    #[inline(always)]
    pub(crate) fn is_free(&self) -> bool {
        (self.parent & FREE_BIT) != 0
    }

    /// Set child0/base (raw write — terminal flag is in NInfo).
    #[inline(always)]
    pub(crate) fn set_child0(&mut self, val: u32) {
        self.child0 = val;
    }

    /// Set parent/check, clears free bit (allocates the state)
    #[inline(always)]
    pub(crate) fn set_parent(&mut self, val: u32) {
        self.parent = val & VALUE_MASK; // No free bit
    }

    /// Mark as free with next/prev pointers for doubly-linked free list.
    #[inline(always)]
    fn set_free_linked(&mut self, next: u32, prev: u32) {
        self.child0 = next; // next free
        self.parent = FREE_BIT | prev; // prev free + free marker
    }

    /// Mark as free (standalone, not linked).
    #[inline(always)]
    pub(crate) fn set_free(&mut self) {
        self.set_free_linked(NIL_STATE, NIL_STATE);
    }

    /// Get next free pointer (only valid when is_free()).
    #[inline(always)]
    #[allow(dead_code)]
    fn free_next(&self) -> u32 {
        self.child0
    }

    /// Get prev free pointer (only valid when is_free()).
    #[inline(always)]
    #[allow(dead_code)]
    fn free_prev(&self) -> u32 {
        self.parent & VALUE_MASK
    }
}

/// Node info for O(k) child enumeration.
/// Labels stored as label+1 (u16) so 0 means "none" while all 256 byte values are valid.
/// Bit 15 of `child` stores the terminal flag (not in DaState.child0, so child0 is mask-free).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct NInfo {
    pub(crate) sibling: u16, // next sibling: label+1 (0 = end)
    pub(crate) child: u16,   // bits 0-8: first child label+1 (0 = no children), bit 15: terminal flag
}

pub(crate) const NINFO_NONE: u16 = 0;
pub(crate) const NINFO_TERM: u16 = 0x8000;

impl NInfo {
    /// Whether this state is terminal (has a complete key ending here).
    #[inline(always)]
    pub(crate) fn is_term(&self) -> bool {
        (self.child & NINFO_TERM) != 0
    }

    /// Mark this state as terminal.
    #[inline(always)]
    pub(crate) fn set_term(&mut self) {
        self.child |= NINFO_TERM;
    }

    /// Clear the terminal flag.
    #[inline(always)]
    pub(crate) fn clear_term(&mut self) {
        self.child &= !NINFO_TERM;
    }

    /// Get the first child pointer (masking out the terminal flag).
    #[inline(always)]
    pub(crate) fn first_child(&self) -> u16 {
        self.child & !NINFO_TERM
    }

    /// Set the first child pointer, preserving the terminal flag.
    #[inline(always)]
    pub(crate) fn set_first_child(&mut self, val: u16) {
        self.child = (self.child & NINFO_TERM) | val;
    }
}

#[inline(always)]
#[allow(dead_code)]
fn ninfo_to_label(v: u16) -> Option<u8> {
    if v == 0 { None } else { Some((v - 1) as u8) }
}

#[inline(always)]
pub(crate) fn label_to_ninfo(label: u8) -> u16 {
    label as u16 + 1
}

