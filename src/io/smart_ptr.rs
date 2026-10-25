//! Smart pointer serialization for reference-counted objects
//!
//! This module provides automatic serialization and deserialization support for
//! Rust's smart pointer types including `Box<T>`, `Arc<T>`, `Rc<T>`, and custom smart pointers.
//! Supports null handling, reference counting, and circular reference detection.

use crate::error::{Result, ZiporaError};
use crate::io::data_input::prealloc_capacity;
use crate::io::{DataInput, DataOutput};
use std::collections::HashMap;
// Hash import removed as it's not used in this module
use std::rc::{Rc, Weak as RcWeak};
use std::sync::{Arc, Weak as ArcWeak};

use std::any::Any;
use std::cell::{Cell, RefCell};

/// Default maximum recursion depth across nested smart pointers and containers.
///
/// Chosen so `max_depth` recursive deserialization frames for pointer and typical
/// struct nodes fit within a standard 2 MiB thread stack in both debug
/// (`opt-level=0`) and release builds. Types with very large inline stack
/// payloads consume additional stack proportional to `size_of::<T>()` per frame
/// in unoptimized builds.
pub(crate) const DEFAULT_MAX_DEPTH: usize = 256;

const MAX_DEPTH_MASK: u64 = 0x0FFF_FFFF;
const STATE_INSTALLED_BIT: u64 = 1 << 60;
const STATE_DIRTY_CTX_BIT: u64 = 1 << 61;
const STATE_COMPRESS_IDS_BIT: u64 = 1 << 62;
const SER_DETECT_CYCLES_BIT: u64 = 1 << 63;

const INITIAL_SER_STATE: u64 = ((DEFAULT_MAX_DEPTH as u64) << 32) | SER_DETECT_CYCLES_BIT;
const INITIAL_DE_STATE: u64 = (DEFAULT_MAX_DEPTH as u64) << 32;
const DIRTY_OUTER_SER_STATE: u64 = INITIAL_SER_STATE | STATE_DIRTY_CTX_BIT;
const DIRTY_OUTER_DE_STATE: u64 = INITIAL_DE_STATE | STATE_DIRTY_CTX_BIT;

thread_local! {
    static SER_DEPTH: Cell<u64> = const { Cell::new(INITIAL_SER_STATE) };
    static DE_DEPTH: Cell<u64> = const { Cell::new(INITIAL_DE_STATE) };
    static ACTIVE_SER_CTX: RefCell<SerializationContext> = RefCell::new(SerializationContext::new());
    static ACTIVE_DE_CTX: RefCell<DeserializationContext> = RefCell::new(DeserializationContext::new());
}

#[cold]
#[inline(never)]
fn max_depth_exceeded_err() -> ZiporaError {
    ZiporaError::invalid_data("Maximum nesting depth exceeded")
}

#[cold]
#[inline(never)]
fn clear_active_ser_ctx() {
    ACTIVE_SER_CTX.with(|ctx| ctx.borrow_mut().clear());
}

#[cold]
#[inline(never)]
fn clear_active_de_ctx() {
    ACTIVE_DE_CTX.with(|ctx| ctx.borrow_mut().clear());
}

#[cold]
#[inline(never)]
fn finish_outer_ser_depth(cell: &Cell<u64>) {
    clear_active_ser_ctx();
    cell.set(INITIAL_SER_STATE);
}

#[cold]
#[inline(never)]
fn finish_outer_de_depth(cell: &Cell<u64>) {
    clear_active_de_ctx();
    cell.set(INITIAL_DE_STATE);
}

pub(crate) struct SerDepthGuard;

impl Drop for SerDepthGuard {
    #[inline(always)]
    fn drop(&mut self) {
        SER_DEPTH.with(|cell| {
            let next = cell.get() - 1;
            cell.set(next);
            if next == DIRTY_OUTER_SER_STATE {
                finish_outer_ser_depth(cell);
            }
        });
    }
}

pub(crate) struct DeDepthGuard;

impl Drop for DeDepthGuard {
    #[inline(always)]
    fn drop(&mut self) {
        DE_DEPTH.with(|cell| {
            let next = cell.get() - 1;
            cell.set(next);
            if next == DIRTY_OUTER_DE_STATE {
                finish_outer_de_depth(cell);
            }
        });
    }
}

#[inline(always)]
pub(crate) fn enter_ser_depth() -> Result<SerDepthGuard> {
    SER_DEPTH.with(|cell| {
        let cur = cell.get();
        let depth = cur as u32;
        let max_depth = ((cur >> 32) & MAX_DEPTH_MASK) as u32;
        if depth >= max_depth {
            return Err(max_depth_exceeded_err());
        }
        cell.set(cur + 1);
        Ok(SerDepthGuard)
    })
}

#[inline(always)]
pub(crate) fn enter_de_depth() -> Result<DeDepthGuard> {
    DE_DEPTH.with(|cell| {
        let cur = cell.get();
        let depth = cur as u32;
        let max_depth = ((cur >> 32) & MAX_DEPTH_MASK) as u32;
        if depth >= max_depth {
            return Err(max_depth_exceeded_err());
        }
        cell.set(cur + 1);
        Ok(DeDepthGuard)
    })
}

pub(crate) struct NestedSerDepthGuard {
    prev: u64,
}

impl Drop for NestedSerDepthGuard {
    #[inline]
    fn drop(&mut self) {
        SER_DEPTH.with(|cell| {
            let cur = cell.get();
            let prev_active = (self.prev as u32) != 0 || (self.prev & STATE_INSTALLED_BIT) != 0;
            if !prev_active {
                if (cur & STATE_DIRTY_CTX_BIT) != 0 {
                    clear_active_ser_ctx();
                }
                cell.set(INITIAL_SER_STATE);
            } else {
                cell.set((self.prev & !STATE_DIRTY_CTX_BIT) | (cur & STATE_DIRTY_CTX_BIT));
            }
        });
    }
}

pub(crate) struct NestedDeDepthGuard {
    prev: u64,
}

impl Drop for NestedDeDepthGuard {
    #[inline]
    fn drop(&mut self) {
        DE_DEPTH.with(|cell| {
            let cur = cell.get();
            let prev_active = (self.prev as u32) != 0 || (self.prev & STATE_INSTALLED_BIT) != 0;
            if !prev_active {
                if (cur & STATE_DIRTY_CTX_BIT) != 0 {
                    clear_active_de_ctx();
                }
                cell.set(INITIAL_DE_STATE);
            } else {
                cell.set((self.prev & !STATE_DIRTY_CTX_BIT) | (cur & STATE_DIRTY_CTX_BIT));
            }
        });
    }
}

#[inline]
pub(crate) fn enter_nested_ser_depth(
    depth: usize,
    max_depth: usize,
) -> Result<NestedSerDepthGuard> {
    if depth > max_depth {
        return Err(max_depth_exceeded_err());
    }
    SER_DEPTH.with(|cell| {
        let cur = cell.get();
        let cur_depth = (cur as u32) as usize;
        let cur_max = ((cur >> 32) & MAX_DEPTH_MASK) as usize;
        let cur_active = cur_depth != 0 || (cur & STATE_INSTALLED_BIT) != 0;
        let effective_depth = cur_depth.max(depth);
        let effective_max = if cur_active {
            cur_max.min(max_depth)
        } else {
            max_depth
        };
        if effective_depth > effective_max {
            return Err(max_depth_exceeded_err());
        }
        let d = effective_depth.min(MAX_DEPTH_MASK as usize) as u64;
        let m = effective_max.min(MAX_DEPTH_MASK as usize) as u64;
        let flags = (cur & (SER_DETECT_CYCLES_BIT | STATE_COMPRESS_IDS_BIT | STATE_DIRTY_CTX_BIT))
            | STATE_INSTALLED_BIT;
        cell.set(d | (m << 32) | flags);
        Ok(NestedSerDepthGuard { prev: cur })
    })
}

#[inline]
pub(crate) fn enter_nested_de_depth(depth: usize, max_depth: usize) -> Result<NestedDeDepthGuard> {
    if depth > max_depth {
        return Err(max_depth_exceeded_err());
    }
    DE_DEPTH.with(|cell| {
        let cur = cell.get();
        let cur_depth = (cur as u32) as usize;
        let cur_max = ((cur >> 32) & MAX_DEPTH_MASK) as usize;
        let cur_active = cur_depth != 0 || (cur & STATE_INSTALLED_BIT) != 0;
        let effective_depth = cur_depth.max(depth);
        let effective_max = if cur_active {
            cur_max.min(max_depth)
        } else {
            max_depth
        };
        if effective_depth > effective_max {
            return Err(max_depth_exceeded_err());
        }
        let d = effective_depth.min(MAX_DEPTH_MASK as usize) as u64;
        let m = effective_max.min(MAX_DEPTH_MASK as usize) as u64;
        let flags = (cur & (STATE_COMPRESS_IDS_BIT | STATE_DIRTY_CTX_BIT)) | STATE_INSTALLED_BIT;
        cell.set(d | (m << 32) | flags);
        Ok(NestedDeDepthGuard { prev: cur })
    })
}

/// Convert a collection length to its 32-bit wire prefix, rejecting lengths above `u32::MAX`.
pub(crate) fn checked_u32_len(len: usize, what: &str) -> Result<u32> {
    u32::try_from(len).map_err(|_| {
        ZiporaError::invalid_data(format!("{what} length {len} exceeds u32::MAX wire limit"))
    })
}

/// Fast single-multiply pointer hasher for `SerializationContext::object_ids`.
#[derive(Default)]
struct PtrHasher(u64);

impl std::hash::Hasher for PtrHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ (b as u64)).wrapping_mul(0x517cc1b727220a95);
        }
    }

    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.0 = (i as u64).wrapping_mul(0x517cc1b727220a95);
    }
}

type PtrBuildHasher = std::hash::BuildHasherDefault<PtrHasher>;

/// Serialization context for tracking shared objects and preventing cycles
#[derive(Debug)]
pub struct SerializationContext {
    /// Object ID counter for reference tracking
    next_id: u32,
    /// Map from object pointer to assigned ID
    object_ids: HashMap<usize, u32, PtrBuildHasher>,
    /// Primary retained smart-pointer vector (`Box<Vec<P>>`) avoiding per-pointer
    /// `Box<dyn Any>` allocations in homogeneous pointer collections.
    retained_primary: Option<Box<dyn Any>>,
    /// Cloned smart pointers of secondary types kept alive so their addresses
    /// cannot be reused by the allocator while this context is active.
    retained_objects: Vec<Box<dyn Any>>,
    /// Enable circular reference detection
    detect_cycles: bool,
    /// Encode object IDs with variable-length integers
    compress_ids: bool,
    /// Current recursion depth across nested containers and smart pointers
    depth: usize,
    /// Maximum allowed recursion depth
    max_depth: usize,
}

impl SerializationContext {
    /// Create a new serialization context
    pub fn new() -> Self {
        Self {
            next_id: 1, // Start from 1, reserve 0 for null
            object_ids: HashMap::default(),
            retained_primary: None,
            retained_objects: Vec::new(),
            detect_cycles: true,
            compress_ids: false,
            depth: 0,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Create context with cycle detection disabled (for performance)
    pub fn without_cycle_detection() -> Self {
        Self {
            next_id: 1,
            object_ids: HashMap::default(),
            retained_primary: None,
            retained_objects: Vec::new(),
            detect_cycles: false,
            compress_ids: false,
            depth: 0,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Look up or assign an ID for an object in a single hash probe, returning `(seen, id)`.
    #[inline]
    fn register_or_assign_id<T>(&mut self, ptr: &T) -> (bool, u32) {
        if !self.detect_cycles {
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            return (false, id);
        }
        let addr = ptr as *const T as usize;
        match self.object_ids.entry(addr) {
            std::collections::hash_map::Entry::Occupied(entry) => (true, *entry.get()),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let id = self.next_id;
                self.next_id = self.next_id.saturating_add(1);
                entry.insert(id);
                (false, id)
            }
        }
    }

    /// Get or assign an ID for an object
    #[cfg(test)]
    #[inline]
    fn get_or_assign_id<T>(&mut self, ptr: &T) -> u32 {
        self.register_or_assign_id(ptr).1
    }

    /// Keep a clone of a serialized smart pointer alive for the lifetime of
    /// this context so its heap address cannot be recycled for an unrelated
    /// allocation.
    #[inline]
    fn retain_object<P: 'static>(&mut self, ptr: P) {
        if !self.detect_cycles {
            return;
        }
        if self.retained_primary.is_none() {
            let mut vec = Vec::<P>::with_capacity(16);
            vec.push(ptr);
            self.retained_primary = Some(Box::new(vec));
            return;
        }
        if let Some(vec) = self
            .retained_primary
            .as_mut()
            .and_then(|b| b.downcast_mut::<Vec<P>>())
        {
            vec.push(ptr);
            return;
        }
        self.retained_objects.push(Box::new(ptr));
    }

    /// Check if an object has been seen before (for cycle detection)
    #[cfg(test)]
    #[inline]
    fn has_seen<T>(&self, ptr: &T) -> bool {
        if !self.detect_cycles {
            return false;
        }

        let addr = ptr as *const T as usize;
        self.object_ids.contains_key(&addr)
    }

    /// Clear the context for reuse
    pub fn clear(&mut self) {
        self.next_id = 1;
        self.object_ids.clear();
        self.retained_primary = None;
        self.retained_objects.clear();
        self.depth = 0;
    }
}

impl Default for SerializationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Deserialization context for reconstructing shared objects.
///
/// Note: `DeserializationContext` is non-generic and stores reconstructed
/// shared pointers in a dense sequential table (`Vec<T>`) for sequential
/// 1-based IDs (`1, 2, 3, ...`) of the primary smart-pointer type and
/// type-erased `Box<dyn Any>` slots (`store_object<T: 'static>` /
/// `get_object<T: 'static>`) for non-sequential IDs or secondary types, so a
/// single context can reconstruct both homogeneous vectors of smart pointers
/// without per-node `Box<dyn Any>` allocations and heterogeneous object graphs
/// containing multiple `Rc<T>` and `Arc<U>` pointee types without allocating
/// gap slots for non-sequential object IDs.
#[derive(Debug)]
pub struct DeserializationContext {
    /// Dense 1-based table (`Box<Vec<T>>`) for strictly sequential IDs (`1..=n`) of the primary type
    dense_objects: Option<Box<dyn Any>>,
    /// Map from object ID to reconstructed smart pointer for non-sequential IDs or secondary types
    objects: HashMap<u32, Box<dyn Any>>,
    /// Decode object IDs with variable-length integers
    compress_ids: bool,
    /// Current recursion depth across nested containers and smart pointers
    depth: usize,
    /// Maximum allowed recursion depth
    max_depth: usize,
}

impl DeserializationContext {
    /// Create a new deserialization context
    pub fn new() -> Self {
        Self {
            dense_objects: None,
            objects: HashMap::new(),
            compress_ids: false,
            depth: 0,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Store an object with its ID
    #[inline]
    pub fn store_object<T: 'static>(&mut self, id: u32, object: T) {
        if id >= 1 {
            let idx = (id - 1) as usize;
            if self.dense_objects.is_none() && idx == 0 {
                let mut vec = Vec::<T>::with_capacity(16);
                vec.push(object);
                self.dense_objects = Some(Box::new(vec));
                return;
            }
            if let Some(vec) = self
                .dense_objects
                .as_mut()
                .and_then(|b| b.downcast_mut::<Vec<T>>())
            {
                if idx == vec.len() {
                    vec.push(object);
                    return;
                } else if idx < vec.len() {
                    vec[idx] = object;
                    return;
                }
            }
        }
        self.objects.insert(id, Box::new(object));
    }

    /// Get an object by its ID
    #[inline]
    pub fn get_object<T: 'static>(&self, id: u32) -> Option<&T> {
        if id >= 1
            && let Some(vec) = self
                .dense_objects
                .as_ref()
                .and_then(|b| b.downcast_ref::<Vec<T>>())
            && let Some(obj) = vec.get((id - 1) as usize)
        {
            return Some(obj);
        }
        self.objects.get(&id)?.downcast_ref::<T>()
    }

    /// Clear the context for reuse
    pub fn clear(&mut self) {
        self.dense_objects = None;
        self.objects.clear();
        self.depth = 0;
    }
}

impl Default for DeserializationContext {
    fn default() -> Self {
        Self::new()
    }
}

fn with_installed_ser_ctx<R>(
    context: &mut SerializationContext,
    f: impl FnOnce() -> Result<R>,
) -> Result<R> {
    if context.depth > context.max_depth {
        return Err(max_depth_exceeded_err());
    }
    struct InstalledSerGuard<'a> {
        caller: &'a mut SerializationContext,
        prev_ctx: SerializationContext,
        prev_state: u64,
        orig_depth: usize,
    }
    impl Drop for InstalledSerGuard<'_> {
        fn drop(&mut self) {
            ACTIVE_SER_CTX.with(|cell| {
                let mut active =
                    std::mem::replace(&mut *cell.borrow_mut(), std::mem::take(&mut self.prev_ctx));
                active.depth = self.orig_depth;
                *self.caller = active;
            });
            SER_DEPTH.with(|cell| cell.set(self.prev_state));
        }
    }
    let orig_depth = context.depth;
    let d = context.depth.min(MAX_DEPTH_MASK as usize) as u64;
    let m = context.max_depth.min(MAX_DEPTH_MASK as usize) as u64;
    let flags = STATE_INSTALLED_BIT
        | if context.detect_cycles {
            SER_DETECT_CYCLES_BIT
        } else {
            0
        }
        | if context.compress_ids {
            STATE_COMPRESS_IDS_BIT
        } else {
            0
        };
    let new_state = d | (m << 32) | flags;
    let prev_state = SER_DEPTH.with(|cell| cell.replace(new_state));
    let prev_ctx = ACTIVE_SER_CTX
        .with(|cell| std::mem::replace(&mut *cell.borrow_mut(), std::mem::take(context)));
    let _guard = InstalledSerGuard {
        caller: context,
        prev_ctx,
        prev_state,
        orig_depth,
    };
    f()
}

fn with_installed_de_ctx<R>(
    context: &mut DeserializationContext,
    f: impl FnOnce() -> Result<R>,
) -> Result<R> {
    if context.depth > context.max_depth {
        return Err(max_depth_exceeded_err());
    }
    struct InstalledDeGuard<'a> {
        caller: &'a mut DeserializationContext,
        prev_ctx: DeserializationContext,
        prev_state: u64,
        orig_depth: usize,
    }
    impl Drop for InstalledDeGuard<'_> {
        fn drop(&mut self) {
            ACTIVE_DE_CTX.with(|cell| {
                let mut active =
                    std::mem::replace(&mut *cell.borrow_mut(), std::mem::take(&mut self.prev_ctx));
                active.depth = self.orig_depth;
                *self.caller = active;
            });
            DE_DEPTH.with(|cell| cell.set(self.prev_state));
        }
    }
    let orig_depth = context.depth;
    let d = context.depth.min(MAX_DEPTH_MASK as usize) as u64;
    let m = context.max_depth.min(MAX_DEPTH_MASK as usize) as u64;
    let flags = STATE_INSTALLED_BIT
        | if context.compress_ids {
            STATE_COMPRESS_IDS_BIT
        } else {
            0
        };
    let new_state = d | (m << 32) | flags;
    let prev_state = DE_DEPTH.with(|cell| cell.replace(new_state));
    let prev_ctx = ACTIVE_DE_CTX
        .with(|cell| std::mem::replace(&mut *cell.borrow_mut(), std::mem::take(context)));
    let _guard = InstalledDeGuard {
        caller: context,
        prev_ctx,
        prev_state,
        orig_depth,
    };
    f()
}

pub(crate) fn with_active_ser_ctx<R>(
    f: impl FnOnce(&mut SerializationContext) -> Result<R>,
) -> Result<R> {
    let mut ctx = ACTIVE_SER_CTX.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    let state = SER_DEPTH.with(|cell| cell.get());
    let active = (state as u32) != 0 || (state & STATE_INSTALLED_BIT) != 0;
    ctx.depth = (state as u32) as usize;
    ctx.max_depth = ((state >> 32) & MAX_DEPTH_MASK) as usize;
    ctx.detect_cycles = (state & SER_DETECT_CYCLES_BIT) != 0;
    ctx.compress_ids = (state & STATE_COMPRESS_IDS_BIT) != 0;
    struct RestoreGuard {
        ctx: Option<SerializationContext>,
        clear_on_drop: bool,
    }
    impl Drop for RestoreGuard {
        fn drop(&mut self) {
            if let Some(mut ctx) = self.ctx.take() {
                if self.clear_on_drop {
                    ctx.clear();
                }
                ACTIVE_SER_CTX.with(|cell| {
                    *cell.borrow_mut() = ctx;
                });
            }
        }
    }
    let mut guard = RestoreGuard {
        ctx: Some(ctx),
        clear_on_drop: !active,
    };
    let ctx_ref = match guard.ctx.as_mut() {
        Some(c) => c,
        None => unreachable!(),
    };
    f(ctx_ref)
}

pub(crate) fn with_active_de_ctx<R>(
    f: impl FnOnce(&mut DeserializationContext) -> Result<R>,
) -> Result<R> {
    let mut ctx = ACTIVE_DE_CTX.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    let state = DE_DEPTH.with(|cell| cell.get());
    let active = (state as u32) != 0 || (state & STATE_INSTALLED_BIT) != 0;
    ctx.depth = (state as u32) as usize;
    ctx.max_depth = ((state >> 32) & MAX_DEPTH_MASK) as usize;
    ctx.compress_ids = (state & STATE_COMPRESS_IDS_BIT) != 0;
    struct RestoreGuard {
        ctx: Option<DeserializationContext>,
        clear_on_drop: bool,
    }
    impl Drop for RestoreGuard {
        fn drop(&mut self) {
            if let Some(mut ctx) = self.ctx.take() {
                if self.clear_on_drop {
                    ctx.clear();
                }
                ACTIVE_DE_CTX.with(|cell| {
                    *cell.borrow_mut() = ctx;
                });
            }
        }
    }
    let mut guard = RestoreGuard {
        ctx: Some(ctx),
        clear_on_drop: !active,
    };
    let ctx_ref = match guard.ctx.as_mut() {
        Some(c) => c,
        None => unreachable!(),
    };
    f(ctx_ref)
}

#[inline]
fn write_active_ser_id<O: DataOutput>(output: &mut O, id: u32, compress_ids: bool) -> Result<()> {
    if compress_ids {
        output.write_var_int(id as u64)
    } else {
        output.write_u32(id)
    }
}

#[inline]
fn read_active_de_id<I: DataInput>(input: &mut I) -> Result<u32> {
    let compress_ids = DE_DEPTH.with(|cell| (cell.get() & STATE_COMPRESS_IDS_BIT) != 0);
    if compress_ids {
        u32::try_from(input.read_var_int()?)
            .map_err(|_| ZiporaError::invalid_data("Compressed object ID overflows u32"))
    } else {
        input.read_u32()
    }
}

#[inline]
fn store_active_de_object<T: 'static>(id: u32, object: T) {
    DE_DEPTH.with(|cell| {
        let cur = cell.get();
        if (cur & STATE_DIRTY_CTX_BIT) == 0 {
            cell.set(cur | STATE_DIRTY_CTX_BIT);
        }
    });
    ACTIVE_DE_CTX.with(|cell| {
        cell.borrow_mut().store_object(id, object);
    });
}

#[inline]
fn get_active_de_object<T: Clone + 'static>(id: u32) -> Option<T> {
    ACTIVE_DE_CTX.with(|cell| cell.borrow().get_object::<T>(id).cloned())
}

#[inline]
fn register_active_ser_ptr<T, P: 'static>(
    obj_ref: &T,
    retain_fn: impl FnOnce() -> P,
) -> (bool, u32, bool) {
    let state = SER_DEPTH.with(|cell| {
        let cur = cell.get();
        if (cur & STATE_DIRTY_CTX_BIT) == 0 {
            cell.set(cur | STATE_DIRTY_CTX_BIT);
        }
        cur
    });
    let detect_cycles = (state & SER_DETECT_CYCLES_BIT) != 0;
    let compress_ids = (state & STATE_COMPRESS_IDS_BIT) != 0;
    if !detect_cycles {
        let id = ACTIVE_SER_CTX.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let id = ctx.next_id;
            ctx.next_id = ctx.next_id.saturating_add(1);
            id
        });
        return (false, id, compress_ids);
    }
    let (seen, id) = ACTIVE_SER_CTX.with(|cell| {
        let mut ctx = cell.borrow_mut();
        ctx.detect_cycles = true;
        let (seen, id) = ctx.register_or_assign_id(obj_ref);
        if !seen {
            ctx.retain_object(retain_fn());
        }
        (seen, id)
    });
    (seen, id, compress_ids)
}

/// Trait for types that can be serialized as smart pointers
pub trait SmartPtrSerialize<T> {
    /// Serialize the smart pointer with context
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()>;

    /// Deserialize the smart pointer with context
    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Serialize without an explicit context, reusing any active outer
    /// [`SerializationContext`] on the current thread.
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        with_active_ser_ctx(|context| self.serialize_with_context(output, context))
    }

    /// Deserialize without an explicit context, reusing any active outer
    /// [`DeserializationContext`] on the current thread.
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self>
    where
        Self: Sized,
    {
        with_active_de_ctx(|context| Self::deserialize_with_context(input, context))
    }
}

/// Marker trait for types that are serializable.
///
/// # Limitations
///
/// - **Shared-pointer context scope on plain structs**: [`SerializableType::serialize`]
///   and [`SerializableType::deserialize`] do not take an explicit context argument.
///   Top-level calls on a smart pointer (`Rc<T>`, `Arc<T>`, `RcWeak<T>`, `ArcWeak<T>`,
///   `Box<T>`), a collection (`Vec<T>`, `HashMap`, `BTreeMap`, `HashSet`, `BTreeSet`),
///   or [`crate::io::NestedSerialize`] install a thread-local context for the duration
///   of that call so nested smart pointers share object IDs and recursion depth.
///   Calling `SerializableType::serialize` directly on a plain user-defined struct
///   outside an enclosing smart-pointer/container scope opens a fresh context per
///   smart-pointer field; to share `Rc`/`Arc` identity across sibling fields of a
///   top-level struct, wrap the struct in a smart pointer or drive serialization
///   through [`SmartPtrSerialize::serialize_with_context`] /
///   [`SmartPtrSerialize::deserialize_with_context`].
/// - **32-bit collection length prefix**: collection implementations (`Vec<T>`,
///   `HashMap`, `BTreeMap`, `HashSet`, `BTreeSet`) encode the element count as a
///   `u32` wire prefix and reject lengths above `u32::MAX` (`4,294,967,295`).
pub trait SerializableType: 'static {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()>;
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self>
    where
        Self: Sized;

    /// Optional contiguous-slice fast path for `Vec<Self>::serialize`.
    ///
    /// # Contract
    ///
    /// - Returning `Ok(false)` (the default) must not write any bytes to `output`;
    ///   `Vec<Self>::serialize` will then serialize each element individually via
    ///   [`SerializableType::serialize`].
    /// - Returning `Ok(true)` indicates that the entire `slice` has been written
    ///   directly to `output` (for example via a single [`DataOutput::write_bytes`]
    ///   call for little-endian `Pod` primitives `u8..u64` / `i8..i64`, bypassing
    ///   per-element `write_u16`/`write_u32`/`write_u64` calls in accordance with
    ///   [`DataOutput`]'s little-endian wire contract).
    #[inline(always)]
    fn serialize_slice<O: DataOutput>(_slice: &[Self], _output: &mut O) -> Result<bool>
    where
        Self: Sized,
    {
        Ok(false)
    }
}

// Implementation for Box<T>
impl<T: SerializableType> SmartPtrSerialize<T> for Box<T> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        with_installed_ser_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::serialize(self, output)
        })
    }

    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        with_installed_de_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::deserialize(input)
        })
    }

    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        // Box is always unique, no need for reference tracking
        output.write_u8(1)?; // Non-null marker
        let _guard = enter_ser_depth()?;
        self.as_ref().serialize(output)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Err(ZiporaError::invalid_data("Box cannot be null")),
            1 => {
                let _guard = enter_de_depth()?;
                let value = T::deserialize(input)?;
                Ok(Box::new(value))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Box marker")),
        }
    }
}

// Implementation for Option<Box<T>>
impl<T: SerializableType> SmartPtrSerialize<T> for Option<Box<T>> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        with_installed_ser_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::serialize(self, output)
        })
    }

    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        with_installed_de_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::deserialize(input)
        })
    }

    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        match self {
            Some(boxed) => {
                output.write_u8(1)?; // Non-null marker
                <Box<T> as SmartPtrSerialize<T>>::serialize(boxed, output)
            }
            None => {
                output.write_u8(0)?; // Null marker
                Ok(())
            }
        }
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(None),
            1 => {
                let boxed = <Box<T> as SmartPtrSerialize<T>>::deserialize(input)?;
                Ok(Some(boxed))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Option<Box> marker")),
        }
    }
}

// Implementation for Rc<T>
impl<T: SerializableType> SmartPtrSerialize<T> for Rc<T> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        with_installed_ser_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::serialize(self, output)
        })
    }

    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        with_installed_de_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::deserialize(input)
        })
    }

    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        let obj_ref = self.as_ref();
        let (seen, id, compress_ids) = register_active_ser_ptr(obj_ref, || self.clone());
        if seen {
            output.write_u8(2)?; // Reference marker
            write_active_ser_id(output, id, compress_ids)
        } else {
            let _guard = enter_ser_depth()?;
            output.write_u8(1)?; // New object marker
            write_active_ser_id(output, id, compress_ids)?;
            obj_ref.serialize(output)
        }
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Err(ZiporaError::invalid_data("Rc cannot be null")),
            1 => {
                let _guard = enter_de_depth()?;
                let id = read_active_de_id(input)?;
                let value = T::deserialize(input)?;
                let rc = Rc::new(value);
                store_active_de_object(id, rc.clone());
                Ok(rc)
            }
            2 => {
                let id = read_active_de_id(input)?;
                get_active_de_object::<Self>(id)
                    .ok_or_else(|| ZiporaError::invalid_data("Referenced object not found"))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Rc marker")),
        }
    }
}

// Implementation for Arc<T>
impl<T: SerializableType + Send + Sync> SmartPtrSerialize<T> for Arc<T> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        with_installed_ser_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::serialize(self, output)
        })
    }

    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        with_installed_de_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::deserialize(input)
        })
    }

    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        let obj_ref = self.as_ref();
        let (seen, id, compress_ids) = register_active_ser_ptr(obj_ref, || self.clone());
        if seen {
            output.write_u8(2)?; // Reference marker
            write_active_ser_id(output, id, compress_ids)
        } else {
            let _guard = enter_ser_depth()?;
            output.write_u8(1)?; // New object marker
            write_active_ser_id(output, id, compress_ids)?;
            obj_ref.serialize(output)
        }
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Err(ZiporaError::invalid_data("Arc cannot be null")),
            1 => {
                let _guard = enter_de_depth()?;
                let id = read_active_de_id(input)?;
                let value = T::deserialize(input)?;
                let arc = Arc::new(value);
                store_active_de_object(id, arc.clone());
                Ok(arc)
            }
            2 => {
                let id = read_active_de_id(input)?;
                get_active_de_object::<Self>(id)
                    .ok_or_else(|| ZiporaError::invalid_data("Referenced object not found"))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Arc marker")),
        }
    }
}

// Weak pointer support for Rc<T>
impl<T: SerializableType> SmartPtrSerialize<T> for RcWeak<T> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        with_installed_ser_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::serialize(self, output)
        })
    }

    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        with_installed_de_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::deserialize(input)
        })
    }

    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        match self.upgrade() {
            Some(rc) => {
                output.write_u8(1)?; // Valid weak reference
                <Rc<T> as SmartPtrSerialize<T>>::serialize(&rc, output)
            }
            None => {
                output.write_u8(0)?; // Dangling weak reference
                Ok(())
            }
        }
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(RcWeak::new()),
            1 => {
                let rc = <Rc<T> as SmartPtrSerialize<T>>::deserialize(input)?;
                Ok(Rc::downgrade(&rc))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Weak<Rc> marker")),
        }
    }
}

// Weak pointer support for Arc<T>
impl<T: SerializableType + Send + Sync> SmartPtrSerialize<T> for ArcWeak<T> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        with_installed_ser_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::serialize(self, output)
        })
    }

    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        with_installed_de_ctx(context, || {
            <Self as SmartPtrSerialize<T>>::deserialize(input)
        })
    }

    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        match self.upgrade() {
            Some(arc) => {
                output.write_u8(1)?; // Valid weak reference
                <Arc<T> as SmartPtrSerialize<T>>::serialize(&arc, output)
            }
            None => {
                output.write_u8(0)?; // Dangling weak reference
                Ok(())
            }
        }
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(ArcWeak::new()),
            1 => {
                let arc = <Arc<T> as SmartPtrSerialize<T>>::deserialize(input)?;
                Ok(Arc::downgrade(&arc))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Weak<Arc> marker")),
        }
    }
}

/// Configuration for smart pointer serialization
#[derive(Debug, Clone)]
pub struct SmartPtrConfig {
    /// Enable cycle detection and shared object optimization
    pub cycle_detection: bool,
    /// Maximum recursion depth to prevent stack overflow
    pub max_depth: usize,
    /// Compress object IDs for smaller serialized size
    pub compress_ids: bool,
}

impl SmartPtrConfig {
    /// Create a new configuration with default settings (`max_depth: 256`)
    pub fn new() -> Self {
        Self {
            cycle_detection: true,
            max_depth: DEFAULT_MAX_DEPTH,
            compress_ids: false,
        }
    }

    /// Configuration optimized for performance (`max_depth: 128`)
    pub fn performance_optimized() -> Self {
        Self {
            cycle_detection: false, // Faster but no cycle detection
            max_depth: 128,
            compress_ids: false,
        }
    }

    /// Configuration optimized for space efficiency (`max_depth: 256`)
    pub fn space_optimized() -> Self {
        Self {
            cycle_detection: true,
            max_depth: DEFAULT_MAX_DEPTH,
            compress_ids: true, // Smaller serialized size
        }
    }

    /// Configuration for robust handling of complex graphs (`max_depth: 256`,
    /// sized so pointer and typical struct nodes stay within a 2 MiB thread
    /// stack even in unoptimized `opt-level=0` builds)
    pub fn robust() -> Self {
        Self {
            cycle_detection: true,
            max_depth: DEFAULT_MAX_DEPTH,
            compress_ids: false,
        }
    }
}

impl Default for SmartPtrConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level smart pointer serialization utilities
pub struct SmartPtrSerializer {
    config: SmartPtrConfig,
}

impl Default for SmartPtrSerializer {
    fn default() -> Self {
        Self::new(SmartPtrConfig::default())
    }
}

impl SmartPtrSerializer {
    /// Create a new serializer with the given configuration
    pub fn new(config: SmartPtrConfig) -> Self {
        Self { config }
    }

    /// Serialize a smart pointer to bytes
    pub fn serialize_to_bytes<T, P>(&self, ptr: &P) -> Result<Vec<u8>>
    where
        P: SmartPtrSerialize<T>,
        T: SerializableType,
    {
        let mut output = crate::io::VecDataOutput::new();
        let mut context = if self.config.cycle_detection {
            SerializationContext::new()
        } else {
            SerializationContext::without_cycle_detection()
        };
        context.compress_ids = self.config.compress_ids;
        context.max_depth = self.config.max_depth;

        ptr.serialize_with_context(&mut output, &mut context)?;
        Ok(output.into_vec())
    }

    /// Deserialize a smart pointer from bytes
    pub fn deserialize_from_bytes<T, P>(&self, bytes: &[u8]) -> Result<P>
    where
        P: SmartPtrSerialize<T>,
        T: SerializableType,
    {
        let mut input = crate::io::SliceDataInput::new(bytes);
        let mut context = DeserializationContext::new();
        context.compress_ids = self.config.compress_ids;
        context.max_depth = self.config.max_depth;

        P::deserialize_with_context(&mut input, &mut context)
    }
}

// Implementations for basic types to make them SerializableType
impl SerializableType for i32 {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(*self as u32)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        Ok(input.read_u32()? as i32)
    }

    #[cfg(target_endian = "little")]
    #[inline(always)]
    fn serialize_slice<O: DataOutput>(slice: &[Self], output: &mut O) -> Result<bool> {
        output.write_bytes(bytemuck::cast_slice(slice))?;
        Ok(true)
    }
}

impl SerializableType for u32 {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(*self)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        input.read_u32()
    }

    #[cfg(target_endian = "little")]
    #[inline(always)]
    fn serialize_slice<O: DataOutput>(slice: &[Self], output: &mut O) -> Result<bool> {
        output.write_bytes(bytemuck::cast_slice(slice))?;
        Ok(true)
    }
}

impl SerializableType for String {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_length_prefixed_string(self)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        input.read_length_prefixed_string()
    }
}

#[cold]
#[inline(never)]
fn deserialize_vec_tail<T: SerializableType, I: DataInput>(
    vec: &mut Vec<T>,
    input: &mut I,
    remaining_elems: usize,
) -> Result<()> {
    for _ in 0..remaining_elems {
        vec.push(T::deserialize(input)?);
    }
    Ok(())
}

impl<T: SerializableType> SerializableType for Vec<T> {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(checked_u32_len(self.len(), "Vec")?)?;
        if self.is_empty() {
            return Ok(());
        }
        let _guard = enter_ser_depth()?;
        if T::serialize_slice(self.as_slice(), output)? {
            return Ok(());
        }
        for item in self {
            item.serialize(output)?;
        }
        Ok(())
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let len = input.read_u32()? as usize;
        if len == 0 {
            return Ok(Vec::new());
        }
        let _guard = enter_de_depth()?;
        // `len` is untrusted: reserve within the read-ahead budget and let
        // the vector grow as elements arrive.
        let cap = prealloc_capacity(len, std::mem::size_of::<T>(), input.remaining_bytes());
        let mut vec = Vec::with_capacity(cap);
        if std::mem::size_of::<T>() == 1 {
            for _ in 0..len {
                vec.push(T::deserialize(input)?);
            }
            return Ok(vec);
        }
        for _ in 0..cap {
            let val = T::deserialize(input)?;
            if vec.len() < vec.capacity() {
                vec.push(val);
            }
        }
        if cap < len {
            deserialize_vec_tail(&mut vec, input, len - cap)?;
        }
        Ok(vec)
    }
}

// Bridge implementation for Rc<T>
impl<T: SerializableType> SerializableType for Rc<T> {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as SmartPtrSerialize<T>>::serialize(self, output)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as SmartPtrSerialize<T>>::deserialize(input)
    }
}

// Bridge implementation for Arc<T> - requires Send + Sync for thread safety
impl<T: SerializableType + Send + Sync> SerializableType for Arc<T> {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as SmartPtrSerialize<T>>::serialize(self, output)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as SmartPtrSerialize<T>>::deserialize(input)
    }
}

// Bridge implementation for RcWeak<T>
impl<T: SerializableType> SerializableType for RcWeak<T> {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as SmartPtrSerialize<T>>::serialize(self, output)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as SmartPtrSerialize<T>>::deserialize(input)
    }
}

// Bridge implementation for ArcWeak<T>
impl<T: SerializableType + Send + Sync> SerializableType for ArcWeak<T> {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as SmartPtrSerialize<T>>::serialize(self, output)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as SmartPtrSerialize<T>>::deserialize(input)
    }
}

// Bridge implementation for Box<T>
impl<T: SerializableType> SerializableType for Box<T> {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as SmartPtrSerialize<T>>::serialize(self, output)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as SmartPtrSerialize<T>>::deserialize(input)
    }
}

// Implement for primitive types using available DataInput/DataOutput methods
impl SerializableType for u8 {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u8(*self)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        input.read_u8()
    }

    #[inline(always)]
    fn serialize_slice<O: DataOutput>(slice: &[Self], output: &mut O) -> Result<bool> {
        output.write_bytes(slice)?;
        Ok(true)
    }
}

impl SerializableType for u16 {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u16(*self)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        input.read_u16()
    }

    #[cfg(target_endian = "little")]
    #[inline(always)]
    fn serialize_slice<O: DataOutput>(slice: &[Self], output: &mut O) -> Result<bool> {
        output.write_bytes(bytemuck::cast_slice(slice))?;
        Ok(true)
    }
}

impl SerializableType for u64 {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u64(*self)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        input.read_u64()
    }

    #[cfg(target_endian = "little")]
    #[inline(always)]
    fn serialize_slice<O: DataOutput>(slice: &[Self], output: &mut O) -> Result<bool> {
        output.write_bytes(bytemuck::cast_slice(slice))?;
        Ok(true)
    }
}

impl SerializableType for i8 {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u8(*self as u8)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        Ok(input.read_u8()? as i8)
    }

    #[inline(always)]
    fn serialize_slice<O: DataOutput>(slice: &[Self], output: &mut O) -> Result<bool> {
        output.write_bytes(bytemuck::cast_slice(slice))?;
        Ok(true)
    }
}

impl SerializableType for i16 {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u16(*self as u16)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        Ok(input.read_u16()? as i16)
    }

    #[cfg(target_endian = "little")]
    #[inline(always)]
    fn serialize_slice<O: DataOutput>(slice: &[Self], output: &mut O) -> Result<bool> {
        output.write_bytes(bytemuck::cast_slice(slice))?;
        Ok(true)
    }
}

impl SerializableType for i64 {
    #[inline]
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u64(*self as u64)
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        Ok(input.read_u64()? as i64)
    }

    #[cfg(target_endian = "little")]
    #[inline(always)]
    fn serialize_slice<O: DataOutput>(slice: &[Self], output: &mut O) -> Result<bool> {
        output.write_bytes(bytemuck::cast_slice(slice))?;
        Ok(true)
    }
}

impl SerializableType for bool {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u8(if *self { 1 } else { 0 })
    }

    #[inline]
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        // `serialize` writes only 0 or 1; any other byte is corrupt input.
        match input.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(ZiporaError::invalid_data("Invalid bool byte")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{SliceDataInput, VecDataOutput};

    #[test]
    fn test_box_serialization() {
        let boxed_value = Box::new(42i32);
        let mut output = VecDataOutput::new();

        <Box<i32> as SerializableType>::serialize(&boxed_value, &mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Box<i32> as SerializableType>::deserialize(&mut input).unwrap();

        assert_eq!(*deserialized, 42);
    }

    #[test]
    fn test_option_box_serialization() {
        // Test Some case
        let some_value = Some(Box::new(42i32));
        let mut output = VecDataOutput::new();

        <Option<Box<i32>> as SmartPtrSerialize<i32>>::serialize(&some_value, &mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized =
            <Option<Box<i32>> as SmartPtrSerialize<i32>>::deserialize(&mut input).unwrap();

        assert_eq!(deserialized.map(|b| *b), Some(42));

        // Test None case
        let none_value: Option<Box<i32>> = None;
        let mut output = VecDataOutput::new();

        <Option<Box<i32>> as SmartPtrSerialize<i32>>::serialize(&none_value, &mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized =
            <Option<Box<i32>> as SmartPtrSerialize<i32>>::deserialize(&mut input).unwrap();

        assert!(deserialized.is_none());
    }

    #[test]
    fn test_rc_serialization() {
        let rc_value = Rc::new(42i32);
        let mut output = VecDataOutput::new();

        <Rc<i32> as SerializableType>::serialize(&rc_value, &mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Rc<i32> as SerializableType>::deserialize(&mut input).unwrap();

        assert_eq!(*deserialized, 42);
    }

    #[test]
    fn test_arc_serialization() {
        let arc_value = Arc::new(42i32);
        let mut output = VecDataOutput::new();

        <Arc<i32> as SerializableType>::serialize(&arc_value, &mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Arc<i32> as SerializableType>::deserialize(&mut input).unwrap();

        assert_eq!(*deserialized, 42);
    }

    #[test]
    fn test_shared_object_serialization() {
        let shared_value = Rc::new(42i32);
        let clone1 = shared_value.clone();
        let clone2 = shared_value.clone();

        let mut context = SerializationContext::new();
        let mut output = VecDataOutput::new();

        // Serialize multiple references to the same object
        <Rc<i32> as SmartPtrSerialize<i32>>::serialize_with_context(
            &clone1,
            &mut output,
            &mut context,
        )
        .unwrap();
        <Rc<i32> as SmartPtrSerialize<i32>>::serialize_with_context(
            &clone2,
            &mut output,
            &mut context,
        )
        .unwrap();

        let bytes = output.into_vec();
        let mut input = SliceDataInput::new(&bytes);
        let mut deserialize_context = DeserializationContext::new();

        let deserialized1 = <Rc<i32> as SmartPtrSerialize<i32>>::deserialize_with_context(
            &mut input,
            &mut deserialize_context,
        )
        .unwrap();
        let deserialized2 = <Rc<i32> as SmartPtrSerialize<i32>>::deserialize_with_context(
            &mut input,
            &mut deserialize_context,
        )
        .unwrap();

        assert_eq!(*deserialized1, 42);
        assert_eq!(*deserialized2, 42);

        // They should be the same object (same allocation)
        assert!(Rc::ptr_eq(&deserialized1, &deserialized2));
    }

    #[test]
    fn test_serialization_context() {
        let mut context = SerializationContext::new();

        let value1 = 42i32;
        let value2 = 43i32;

        // Same object should get the same ID
        let id1a = context.get_or_assign_id(&value1);
        let id1b = context.get_or_assign_id(&value1);
        assert_eq!(id1a, id1b);

        // Different object should get different ID
        let id2 = context.get_or_assign_id(&value2);
        assert_ne!(id1a, id2);

        // Should detect already seen objects
        assert!(context.has_seen(&value1));
        assert!(context.has_seen(&value2));
    }

    #[test]
    fn test_smart_ptr_serializer() {
        let serializer = SmartPtrSerializer::default();
        let boxed_value = Box::new(42i32);

        let bytes = serializer.serialize_to_bytes(&boxed_value).unwrap();
        let deserialized: Box<i32> = serializer.deserialize_from_bytes(&bytes).unwrap();

        assert_eq!(*deserialized, 42);
    }

    #[test]
    fn test_performance_config() {
        let config = SmartPtrConfig::performance_optimized();
        assert!(!config.cycle_detection);
        assert_eq!(config.max_depth, 128);

        let config = SmartPtrConfig::space_optimized();
        assert!(config.cycle_detection);
        assert!(config.compress_ids);
        assert_eq!(config.max_depth, 256);

        let config = SmartPtrConfig::robust();
        assert!(config.cycle_detection);
        assert_eq!(config.max_depth, 256);
    }

    #[test]
    fn test_vec_serialization() {
        let vec_value = vec![1i32, 2, 3, 4, 5];
        let mut output = VecDataOutput::new();

        <Vec<i32> as SerializableType>::serialize(&vec_value, &mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Vec<i32> as SerializableType>::deserialize(&mut input).unwrap();

        assert_eq!(deserialized, vec_value);
    }

    #[test]
    fn test_string_serialization() {
        let string_value = "Hello, World!".to_string();
        let mut output = VecDataOutput::new();

        <String as SerializableType>::serialize(&string_value, &mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <String as SerializableType>::deserialize(&mut input).unwrap();

        assert_eq!(deserialized, string_value);
    }

    #[test]
    fn test_complex_nested_structure() {
        let nested = Box::new(vec![
            Rc::new("Hello".to_string()),
            Rc::new("World".to_string()),
        ]);

        let mut output = VecDataOutput::new();
        <Box<Vec<Rc<String>>> as SerializableType>::serialize(&nested, &mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized =
            <Box<Vec<Rc<String>>> as SerializableType>::deserialize(&mut input).unwrap();

        assert_eq!(deserialized.len(), 2);
        assert_eq!(*deserialized[0], "Hello");
        assert_eq!(*deserialized[1], "World");
    }

    #[test]
    fn test_c9_sp6_bool_rejects_bytes_above_one() {
        for byte in [2u8, 0xFF] {
            let bytes = [byte];
            let mut input = SliceDataInput::new(&bytes);
            assert!(
                <bool as SerializableType>::deserialize(&mut input).is_err(),
                "byte {byte:#04x} must not decode as a bool"
            );
        }
        for (byte, expected) in [(0u8, false), (1u8, true)] {
            let bytes = [byte];
            let mut input = SliceDataInput::new(&bytes);
            assert_eq!(
                <bool as SerializableType>::deserialize(&mut input).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn test_c9_sp2_dangling_weak_round_trips() {
        let mut out = VecDataOutput::new();
        let dangling_rc: RcWeak<u32> = RcWeak::new();
        <RcWeak<u32> as SmartPtrSerialize<u32>>::serialize(&dangling_rc, &mut out).unwrap();
        let dangling_arc: ArcWeak<u32> = ArcWeak::new();
        <ArcWeak<u32> as SmartPtrSerialize<u32>>::serialize(&dangling_arc, &mut out).unwrap();
        out.write_u32(99).unwrap();

        let bytes = out.into_vec();
        assert_eq!(bytes, [0, 0, 99, 0, 0, 0]);

        let mut input = SliceDataInput::new(&bytes);
        let back_rc = <RcWeak<u32> as SmartPtrSerialize<u32>>::deserialize(&mut input).unwrap();
        assert!(back_rc.upgrade().is_none());
        let back_arc = <ArcWeak<u32> as SmartPtrSerialize<u32>>::deserialize(&mut input).unwrap();
        assert!(back_arc.upgrade().is_none());
        assert_eq!(input.read_u32().unwrap(), 99);
    }

    struct StrongAndWeak {
        strong: Rc<u32>,
        weak: RcWeak<u32>,
    }

    impl SerializableType for StrongAndWeak {
        fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
            <Rc<u32> as SmartPtrSerialize<u32>>::serialize(&self.strong, output)?;
            <RcWeak<u32> as SmartPtrSerialize<u32>>::serialize(&self.weak, output)
        }

        fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
            let strong = <Rc<u32> as SmartPtrSerialize<u32>>::deserialize(input)?;
            let weak = <RcWeak<u32> as SmartPtrSerialize<u32>>::deserialize(input)?;
            Ok(Self { strong, weak })
        }
    }

    #[test]
    fn test_c9_sp3_live_weak_and_shared_context_round_trip() {
        let strong = Rc::new(5u32);
        let weak = Rc::downgrade(&strong);
        let mut out = VecDataOutput::new();
        <RcWeak<u32> as SmartPtrSerialize<u32>>::serialize(&weak, &mut out).unwrap();
        let bytes = out.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let mut ctx = DeserializationContext::new();
        let back =
            <RcWeak<u32> as SmartPtrSerialize<u32>>::deserialize_with_context(&mut input, &mut ctx)
                .unwrap();
        assert_eq!(back.upgrade().map(|r| *r), Some(5));

        let pair = Box::new(StrongAndWeak {
            strong: strong.clone(),
            weak: Rc::downgrade(&strong),
        });
        let mut pair_out = VecDataOutput::new();
        <Box<StrongAndWeak> as SmartPtrSerialize<StrongAndWeak>>::serialize(&pair, &mut pair_out)
            .unwrap();
        let pair_bytes = pair_out.into_vec();
        let mut pair_in = SliceDataInput::new(&pair_bytes);
        let pair_back =
            <Box<StrongAndWeak> as SmartPtrSerialize<StrongAndWeak>>::deserialize(&mut pair_in)
                .unwrap();
        let upgraded = pair_back
            .weak
            .upgrade()
            .expect("weak reference must upgrade");
        assert_eq!(*upgraded, 5);
        assert!(Rc::ptr_eq(&pair_back.strong, &upgraded));
    }

    #[test]
    fn test_c9_sp4_temporary_rc_addresses_are_not_confused() {
        let mut ctx = SerializationContext::new();
        let mut out = VecDataOutput::new();
        for i in 0..4u32 {
            <Rc<u32> as SmartPtrSerialize<u32>>::serialize_with_context(
                &Rc::new(i),
                &mut out,
                &mut ctx,
            )
            .unwrap();
        }

        let bytes = out.into_vec();
        let mut input = SliceDataInput::new(&bytes);
        let mut de_ctx = DeserializationContext::new();
        let mut decoded = Vec::new();
        for _ in 0..4 {
            let rc = <Rc<u32> as SmartPtrSerialize<u32>>::deserialize_with_context(
                &mut input,
                &mut de_ctx,
            )
            .unwrap();
            decoded.push(*rc);
        }
        assert_eq!(decoded, [0, 1, 2, 3]);
    }

    static CYCLE_DEPTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    struct CyclicNode {
        val: u32,
        next: std::cell::RefCell<Option<Rc<CyclicNode>>>,
    }

    impl SerializableType for CyclicNode {
        fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
            use std::sync::atomic::Ordering;
            let d = CYCLE_DEPTH.fetch_add(1, Ordering::Relaxed) + 1;
            struct Reset;
            impl Drop for Reset {
                fn drop(&mut self) {
                    CYCLE_DEPTH.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            let _reset = Reset;
            if d >= 16 {
                return Err(ZiporaError::invalid_data("cycle was not detected"));
            }
            self.val.serialize(output)?;
            match self.next.borrow().as_ref() {
                Some(rc) => {
                    output.write_u8(1)?;
                    <Rc<CyclicNode> as SerializableType>::serialize(rc, output)
                }
                None => output.write_u8(0),
            }
        }

        fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
            let val = u32::deserialize(input)?;
            let next = match input.read_u8()? {
                0 => None,
                1 => Some(<Rc<CyclicNode> as SerializableType>::deserialize(input)?),
                _ => return Err(ZiporaError::invalid_data("Invalid CyclicNode marker")),
            };
            Ok(Self {
                val,
                next: std::cell::RefCell::new(next),
            })
        }
    }

    #[test]
    fn test_c9_sp5_nested_sharing_and_cycle_detection() {
        let shared = Rc::new("shared-payload".to_string());
        let vec = vec![shared.clone(), shared.clone(), shared];
        let mut out = VecDataOutput::new();
        <Vec<Rc<String>> as SerializableType>::serialize(&vec, &mut out).unwrap();
        let bytes = out.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let back = <Vec<Rc<String>> as SerializableType>::deserialize(&mut input).unwrap();
        assert_eq!(back.len(), 3);
        assert!(Rc::ptr_eq(&back[0], &back[1]));
        assert!(Rc::ptr_eq(&back[1], &back[2]));

        // Self-referential node must emit a reference marker (2) rather than recursing.
        CYCLE_DEPTH.store(0, std::sync::atomic::Ordering::Relaxed);
        let node = Rc::new(CyclicNode {
            val: 7,
            next: std::cell::RefCell::new(None),
        });
        *node.next.borrow_mut() = Some(node.clone());
        let mut cyc_out = VecDataOutput::new();
        let res = <Rc<CyclicNode> as SmartPtrSerialize<CyclicNode>>::serialize(&node, &mut cyc_out);
        *node.next.borrow_mut() = None;
        res.expect("cycle detection must stop self-referential recursion");
        let cyc_bytes = cyc_out.into_vec();
        // [marker=1, id=1(le u32), val=7(le u32), Some=1, marker=2, id=1(le u32)]
        assert_eq!(cyc_bytes, [1, 1, 0, 0, 0, 7, 0, 0, 0, 1, 2, 1, 0, 0, 0]);
    }

    #[test]
    fn test_c9_ct5_smart_ptr_serializer_enforces_max_depth() {
        #[derive(Debug, PartialEq)]
        struct ChainNode {
            next: Option<Box<ChainNode>>,
        }

        impl SerializableType for ChainNode {
            fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
                <Option<Box<ChainNode>> as SmartPtrSerialize<ChainNode>>::serialize(
                    &self.next, output,
                )
            }

            fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
                Ok(Self {
                    next: <Option<Box<ChainNode>> as SmartPtrSerialize<ChainNode>>::deserialize(
                        input,
                    )?,
                })
            }
        }

        let build_chain = |depth: usize| {
            let mut node = Box::new(ChainNode { next: None });
            for _ in 0..depth {
                node = Box::new(ChainNode { next: Some(node) });
            }
            node
        };

        let shallow = build_chain(3);
        let deep = build_chain(12);

        let bounded = SmartPtrSerializer::new(SmartPtrConfig {
            max_depth: 8,
            ..SmartPtrConfig::default()
        });
        let shallow_bytes = bounded.serialize_to_bytes(&shallow).unwrap();
        let shallow_back: Box<ChainNode> = bounded.deserialize_from_bytes(&shallow_bytes).unwrap();
        assert_eq!(shallow_back, shallow);

        assert!(
            bounded.serialize_to_bytes(&deep).is_err(),
            "depth 12 must exceed SmartPtrConfig::max_depth = 8 on serialize"
        );

        let default_bytes = SmartPtrSerializer::default()
            .serialize_to_bytes(&deep)
            .unwrap();
        assert!(
            bounded
                .deserialize_from_bytes::<ChainNode, Box<ChainNode>>(&default_bytes)
                .is_err(),
            "depth 12 must exceed SmartPtrConfig::max_depth = 8 on deserialize"
        );
    }

    #[test]
    fn test_s22_r1_hostile_deep_chains_rejected_on_2mib_stack() {
        struct BoxChain {
            next: Option<Box<BoxChain>>,
        }
        impl SerializableType for BoxChain {
            fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
                <Option<Box<BoxChain>> as SmartPtrSerialize<BoxChain>>::serialize(
                    &self.next, output,
                )
            }
            fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
                Ok(Self {
                    next: <Option<Box<BoxChain>> as SmartPtrSerialize<BoxChain>>::deserialize(
                        input,
                    )?,
                })
            }
        }

        struct RcChain {
            next: Option<Rc<RcChain>>,
        }
        impl SerializableType for RcChain {
            fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
                <Option<Rc<RcChain>> as SerializableType>::serialize(&self.next, output)
            }
            fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
                Ok(Self {
                    next: <Option<Rc<RcChain>> as SerializableType>::deserialize(input)?,
                })
            }
        }

        let depth = if cfg!(miri) { 2_000usize } else { 100_000usize };
        let mut box_wire = vec![1u8; depth + 1];
        box_wire[depth] = 0;

        let mut rc_wire = Vec::with_capacity(depth * 6 + 1);
        for i in 1..=depth {
            if i > 1 {
                rc_wire.push(1u8); // Option::Some
            }
            rc_wire.push(1u8); // Rc new-object marker
            rc_wire.extend_from_slice(&(i as u32).to_le_bytes());
        }
        rc_wire.push(0u8); // Option::None

        let mut rc_compressed_wire = Vec::with_capacity(depth * 4 + 1);
        let mut tmp_out = VecDataOutput::new();
        for i in 1..=depth {
            if i > 1 {
                tmp_out.write_u8(1).unwrap(); // Option::Some
            }
            tmp_out.write_u8(1).unwrap(); // Rc new-object marker
            tmp_out.write_var_int(i as u64).unwrap();
        }
        tmp_out.write_u8(0).unwrap(); // Option::None
        rc_compressed_wire.extend_from_slice(&tmp_out.into_vec());

        std::thread::Builder::new()
            .name("s22-r1-2mib".into())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                for config in [
                    SmartPtrConfig::default(),
                    SmartPtrConfig::performance_optimized(),
                    SmartPtrConfig::space_optimized(),
                    SmartPtrConfig::robust(),
                ] {
                    let ser = SmartPtrSerializer::new(config.clone());
                    assert!(
                        ser.deserialize_from_bytes::<BoxChain, Box<BoxChain>>(&box_wire)
                            .is_err(),
                        "BoxChain must return Err within 2 MiB stack for {:?}",
                        config
                    );
                    let rc_payload = if config.compress_ids {
                        &rc_compressed_wire
                    } else {
                        &rc_wire
                    };
                    assert!(
                        ser.deserialize_from_bytes::<RcChain, Rc<RcChain>>(rc_payload)
                            .is_err(),
                        "RcChain must return Err within 2 MiB stack for {:?}",
                        config
                    );
                }

                let mut box_in = SliceDataInput::new(&box_wire);
                assert!(<Box<BoxChain> as SerializableType>::deserialize(&mut box_in).is_err());
                let mut rc_in = SliceDataInput::new(&rc_wire);
                assert!(<Rc<RcChain> as SerializableType>::deserialize(&mut rc_in).is_err());
            })
            .expect("spawn 2 MiB thread")
            .join()
            .expect("2 MiB thread must not overflow stack or panic");
    }

    #[test]
    fn test_s23_r1_nested_and_vec_deserialize_and_big_chain_2mib_stack() {
        struct BigChain {
            payload: [u8; 512],
            next: Option<Box<BigChain>>,
        }
        impl Drop for BigChain {
            fn drop(&mut self) {
                let mut curr = self.next.take();
                while let Some(mut node) = curr {
                    curr = node.next.take();
                }
            }
        }
        impl SerializableType for BigChain {
            fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
                output.write_bytes(&self.payload)?;
                <Option<Box<BigChain>> as SmartPtrSerialize<BigChain>>::serialize(
                    &self.next, output,
                )
            }
            fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
                let mut payload = [0u8; 512];
                input.read_bytes(&mut payload)?;
                let next =
                    <Option<Box<BigChain>> as SmartPtrSerialize<BigChain>>::deserialize(input)?;
                Ok(Self { payload, next })
            }
        }

        let make_big_chain_wire = |depth: usize| {
            let mut wire = Vec::with_capacity(depth * 514 + 1);
            for i in 0..depth {
                if i > 0 {
                    wire.push(1u8); // Option::Some
                }
                wire.push(1u8); // Box non-null marker
                wire.extend_from_slice(&[0xABu8; 512]);
            }
            wire.push(0u8); // Option::None
            wire
        };

        std::thread::Builder::new()
            .name("s23-r5-bigchain-2mib".into())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                for config in [
                    SmartPtrConfig::default(),
                    SmartPtrConfig::performance_optimized(),
                    SmartPtrConfig::space_optimized(),
                    SmartPtrConfig::robust(),
                ] {
                    let limit = if cfg!(miri) {
                        16usize
                    } else {
                        config.max_depth
                    };
                    let eff_config = SmartPtrConfig {
                        max_depth: limit,
                        ..config
                    };
                    let ser = SmartPtrSerializer::new(eff_config);

                    let ok_wire = make_big_chain_wire(limit);
                    let decoded = ser
                        .deserialize_from_bytes::<BigChain, Box<BigChain>>(&ok_wire)
                        .expect("BigChain at max_depth must succeed within 2 MiB stack");
                    assert_eq!(decoded.payload[0], 0xAB);

                    let err_wire = make_big_chain_wire(limit + 1);
                    assert!(
                        ser.deserialize_from_bytes::<BigChain, Box<BigChain>>(&err_wire)
                            .is_err(),
                        "BigChain at max_depth + 1 must return Err"
                    );
                }
            })
            .expect("spawn 2 MiB thread")
            .join()
            .expect("BigChain 2 MiB thread must not overflow stack");

        if !cfg!(miri) {
            // Round-trip correctness check for Vec<u64>, Vec<Rc<u64>>, and Vec<Vec<u32>>
            let vec_u64: Vec<u64> = (0..1_000_000u64).collect();
            let mut out_u64 = VecDataOutput::with_capacity(4 + 1_000_000 * 8);
            vec_u64.serialize(&mut out_u64).unwrap();
            let bytes_u64 = out_u64.into_vec();

            let vec_rc: Vec<Rc<u64>> = (0..100_000u64).map(Rc::new).collect();
            let mut out_rc = VecDataOutput::with_capacity(4 + 100_000 * 13);
            vec_rc.serialize(&mut out_rc).unwrap();
            let bytes_rc = out_rc.into_vec();

            let vec_vec: Vec<Vec<u32>> = (0..1_000u32)
                .map(|i| (0..100u32).map(|j| i * 100 + j).collect())
                .collect();
            let mut out_vv = VecDataOutput::with_capacity(4 + 1_000 * 404);
            vec_vec.serialize(&mut out_vv).unwrap();
            let bytes_vv = out_vv.into_vec();

            // Warmup & correctness
            let mut in_u64 = SliceDataInput::new(&bytes_u64);
            let dec_u64 = Vec::<u64>::deserialize(&mut in_u64).unwrap();
            assert_eq!(dec_u64.len(), 1_000_000);

            let mut in_rc = SliceDataInput::new(&bytes_rc);
            let dec_rc = Vec::<Rc<u64>>::deserialize(&mut in_rc).unwrap();
            assert_eq!(dec_rc.len(), 100_000);
            assert_eq!(*dec_rc[99_999], 99_999);

            let mut in_vv = SliceDataInput::new(&bytes_vv);
            let dec_vv = Vec::<Vec<u32>>::deserialize(&mut in_vv).unwrap();
            assert_eq!(dec_vv.len(), 1_000);
            assert_eq!(dec_vv[999][99], 99_999);
        }
    }

    #[test]
    fn test_s24_r1_stride_65_object_ids_do_not_amplify_dense_table_memory() {
        let count: usize = if cfg!(miri) { 64 } else { 10_000 };
        let mut ctx = DeserializationContext::new();
        for i in 0..count {
            let id = 1u32 + (i as u32) * 65;
            ctx.store_object(id, Rc::new(i as u64));
        }
        for i in 0..count {
            let id = 1u32 + (i as u32) * 65;
            let got = ctx
                .get_object::<Rc<u64>>(id)
                .expect("stored stride-65 object");
            assert_eq!(**got, i as u64);
        }

        // Only id == 1 belongs in the sequential 1-based dense table; non-sequential
        // IDs (66, 131, ...) must fall back to `objects` without padding 64 gap slots each.
        let dense_len_and_cap = ctx.dense_objects.as_ref().and_then(|b| {
            b.downcast_ref::<Vec<Rc<u64>>>()
                .map(|v| (v.len(), v.capacity()))
                .or_else(|| {
                    b.downcast_ref::<Vec<Option<Rc<u64>>>>()
                        .map(|v| (v.len(), v.capacity()))
                })
        });
        assert_eq!(
            dense_len_and_cap.map(|(len, _)| len),
            Some(1),
            "stride-65 object IDs must not grow dense_objects beyond id == 1 (got {dense_len_and_cap:?})"
        );
        assert!(
            dense_len_and_cap.is_some_and(|(_, cap)| cap <= 16),
            "stride-65 object IDs must not amplify dense_objects capacity (got {dense_len_and_cap:?})"
        );
        assert_eq!(ctx.objects.len(), count - 1);

        // Also verify end-to-end wire deserialization with stride-65 object IDs and back-references.
        let wire_count: u32 = if cfg!(miri) { 32 } else { 2_000 };
        let mut out = VecDataOutput::with_capacity(4 + (wire_count as usize) * 2 * 13);
        out.write_u32(wire_count * 2).unwrap();
        for i in 0..wire_count {
            let id = 1 + i * 65;
            out.write_u8(1).unwrap(); // New Rc<u64>
            out.write_u32(id).unwrap();
            out.write_u64(i as u64 + 1000).unwrap();
        }
        for i in 0..wire_count {
            let id = 1 + i * 65;
            out.write_u8(2).unwrap(); // Reference to existing Rc<u64>
            out.write_u32(id).unwrap();
        }
        let wire = out.into_vec();
        let mut input = SliceDataInput::new(&wire);
        let decoded = Vec::<Rc<u64>>::deserialize(&mut input).unwrap();
        assert_eq!(decoded.len(), (wire_count as usize) * 2);
        for i in 0..(wire_count as usize) {
            assert_eq!(*decoded[i], i as u64 + 1000);
            assert!(Rc::ptr_eq(&decoded[i], &decoded[i + wire_count as usize]));
        }
    }

    #[test]
    #[ignore = "Timing benchmark harness for S23-R1 / S24-R3; run with `cargo test --release --lib bench_s24_r3_io_deserialize_timings -- --ignored --nocapture`"]
    fn bench_s24_r3_io_deserialize_timings() {
        use std::hint::black_box;
        use std::time::Instant;

        fn median_ns_per_op<T>(iters: usize, elems: usize, mut f: impl FnMut() -> T) -> f64 {
            for _ in 0..3 {
                black_box(f());
            }
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let val = black_box(f());
                let elapsed = t0.elapsed().as_nanos() as f64;
                drop(val);
                samples.push(elapsed / elems as f64);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            samples[iters / 2]
        }

        // 1. Vec<u64> (1,000,000 elements)
        let vec_u64: Vec<u64> = (0..1_000_000u64).collect();
        let mut out_u64 = VecDataOutput::with_capacity(4 + 1_000_000 * 8);
        vec_u64.serialize(&mut out_u64).unwrap();
        let bytes_u64 = out_u64.into_vec();
        let ns_u64 = median_ns_per_op(21, 1_000_000, || {
            let mut input = SliceDataInput::new(black_box(&bytes_u64));
            Vec::<u64>::deserialize(&mut input).unwrap()
        });

        // 2. Vec<Rc<u64>> (100,000 elements)
        let vec_rc: Vec<Rc<u64>> = (0..100_000u64).map(Rc::new).collect();
        let mut out_rc = VecDataOutput::with_capacity(4 + 100_000 * 13);
        vec_rc.serialize(&mut out_rc).unwrap();
        let bytes_rc = out_rc.into_vec();
        let ns_rc = median_ns_per_op(21, 100_000, || {
            let mut input = SliceDataInput::new(black_box(&bytes_rc));
            Vec::<Rc<u64>>::deserialize(&mut input).unwrap()
        });
        let ns_rc_ser = median_ns_per_op(21, 100_000, || {
            let mut out = VecDataOutput::with_capacity(4 + 100_000 * 13);
            black_box(&vec_rc).serialize(&mut out).unwrap();
            out
        });

        // 3. Vec<Vec<u32>> (1,000 x 100)
        let vec_vec_u32: Vec<Vec<u32>> = (0..1_000u32)
            .map(|i| (0..100u32).map(|j| i * 100 + j).collect())
            .collect();
        let mut out_vv32 = VecDataOutput::with_capacity(4 + 1_000 * 404);
        vec_vec_u32.serialize(&mut out_vv32).unwrap();
        let bytes_vv32 = out_vv32.into_vec();
        let ns_vv32 = median_ns_per_op(21, 1_000, || {
            let mut input = SliceDataInput::new(black_box(&bytes_vv32));
            Vec::<Vec<u32>>::deserialize(&mut input).unwrap()
        });
        let ns_vv32_ser = median_ns_per_op(21, 1_000, || {
            let mut out = VecDataOutput::with_capacity(4 + 1_000 * 404);
            black_box(&vec_vec_u32).serialize(&mut out).unwrap();
            out
        });

        // 4. Vec<Vec<u8>> (100,000 x 3)
        let vec_vec_u8: Vec<Vec<u8>> = (0..100_000u32)
            .map(|i| vec![(i & 0xFF) as u8, ((i >> 8) & 0xFF) as u8, 0x5Au8])
            .collect();
        let mut out_vv8 = VecDataOutput::with_capacity(4 + 100_000 * 7);
        vec_vec_u8.serialize(&mut out_vv8).unwrap();
        let bytes_vv8 = out_vv8.into_vec();
        let ns_vv8 = median_ns_per_op(21, 100_000, || {
            let mut input = SliceDataInput::new(black_box(&bytes_vv8));
            Vec::<Vec<u8>>::deserialize(&mut input).unwrap()
        });

        // 5. Vec<Box<u64>> (100,000 elements)
        let vec_box: Vec<Box<u64>> = (0..100_000u64).map(Box::new).collect();
        let mut out_box = VecDataOutput::with_capacity(4 + 100_000 * 9);
        vec_box.serialize(&mut out_box).unwrap();
        let bytes_box = out_box.into_vec();
        let ns_box = median_ns_per_op(21, 100_000, || {
            let mut input = SliceDataInput::new(black_box(&bytes_box));
            Vec::<Box<u64>>::deserialize(&mut input).unwrap()
        });

        println!(
            "S24-R3 timings: Vec<u64>={ns_u64:.3} ns/elem | Vec<Rc<u64>> de={ns_rc:.2} ns/elem ser={ns_rc_ser:.2} ns/elem | Vec<Vec<u32>> de={ns_vv32:.1} ns/inner ser={ns_vv32_ser:.1} ns/inner | Vec<Vec<u8>>(100kx3)={ns_vv8:.2} ns/inner | Vec<Box<u64>>={ns_box:.2} ns/elem"
        );
    }

    #[test]
    fn test_s48_r9_serialization_context_clear_drops_retained_primary() {
        let rc = Rc::new(7u64);
        let mut ctx = SerializationContext::new();
        let mut out = VecDataOutput::new();
        <Rc<u64> as SmartPtrSerialize<u64>>::serialize_with_context(&rc, &mut out, &mut ctx)
            .unwrap();
        assert_eq!(
            Rc::strong_count(&rc),
            2,
            "SerializationContext must retain the first serialized Rc while active"
        );
        ctx.clear();
        assert_eq!(
            Rc::strong_count(&rc),
            1,
            "SerializationContext::clear() must drop retained_primary so Rc strong_count returns to 1"
        );
    }
}
