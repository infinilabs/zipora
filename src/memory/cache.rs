//! Cache-conscious memory management
//!
//! This module provides cache-aligned allocations and NUMA-aware memory management
//! to optimize performance on modern multi-core systems.

use crate::error::{Result, ZiporaError};
use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashMap;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};

/// Cache line size on most modern processors (64 bytes)
pub const CACHE_LINE_SIZE: usize = 64;

/// NUMA node identifier
pub type NumaNode = usize;

/// Cache-aligned vector that ensures data starts at cache line boundaries
#[repr(align(64))] // Cache line alignment
pub struct CacheAlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    numa_node: Option<NumaNode>,
}

impl<T> CacheAlignedVec<T> {
    /// Create a new cache-aligned vector
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            numa_node: get_current_numa_node(),
        }
    }

    /// Create a new cache-aligned vector with specified capacity
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        let mut vec = Self::new();
        vec.reserve(capacity)?;
        Ok(vec)
    }

    /// Create a cache-aligned vector on a specific NUMA node
    pub fn with_numa_node(numa_node: NumaNode) -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            numa_node: Some(numa_node),
        }
    }

    /// Get the current length
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the current capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the NUMA node this vector is allocated on
    pub fn numa_node(&self) -> Option<NumaNode> {
        self.numa_node
    }

    /// The `Layout` backing `capacity` elements of `T`.
    ///
    /// The alignment is `max(CACHE_LINE_SIZE, align_of::<T>())`. Every layout
    /// in this type used to be built with `CACHE_LINE_SIZE` alone, which
    /// silently under-aligns any `T` aligned more strictly than a cache line
    /// (`#[repr(align(128))]`, an AVX-512 register wrapper); the `ptr::write`
    /// in `push` and the `slice::from_raw_parts` in `as_slice` are then
    /// undefined behaviour, reachable from safe public API.
    ///
    /// # Errors
    ///
    /// Returns an error if `capacity * size_of::<T>()` overflows, or exceeds
    /// what `Layout` permits.
    fn layout_for(capacity: usize) -> Result<Layout> {
        let bytes = capacity
            .checked_mul(mem::size_of::<T>())
            .ok_or_else(|| ZiporaError::invalid_data("Capacity overflow"))?;
        Layout::from_size_align(bytes, CACHE_LINE_SIZE.max(mem::align_of::<T>()))
            .map_err(|_| ZiporaError::invalid_data("Invalid layout for cache-aligned allocation"))
    }

    /// Reserve capacity for at least `additional` more elements
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        let required_cap = self
            .len
            .checked_add(additional)
            .ok_or_else(|| ZiporaError::invalid_data("Capacity overflow"))?;

        if required_cap <= self.capacity {
            return Ok(());
        }

        // Grow by at least 2x to amortize allocations. `saturating_mul`
        // rather than `*`: `self.capacity` is `usize::MAX` for a zero-sized
        // element type, which never reaches here but must not overflow if it
        // ever did.
        let new_cap = required_cap.max(self.capacity.saturating_mul(2)).max(4);
        self.reallocate(new_cap)
    }

    /// Push an element onto the end of the vector
    #[inline]
    pub fn push(&mut self, value: T) -> Result<()> {
        if self.len == self.capacity {
            self.reserve(1)?;
        }

        // SAFETY: pointer valid from alloc, offset < capacity ensured by reserve above
        unsafe {
            ptr::write(self.ptr.as_ptr().add(self.len), value);
        }
        self.len += 1;
        Ok(())
    }

    /// Pop an element from the end of the vector
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        // SAFETY: pointer valid from alloc, offset < len checked by decrement above
        unsafe { Some(ptr::read(self.ptr.as_ptr().add(self.len))) }
    }

    /// Get a reference to an element by index
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY: pointer valid from alloc, index < len checked above
            unsafe { Some(&*self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Get a mutable reference to an element by index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            // SAFETY: pointer valid from alloc, index < len checked above
            unsafe { Some(&mut *self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Get a slice of all elements
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: pointer valid from alloc, len <= capacity
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice of all elements
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: pointer valid from alloc, len <= capacity
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        // Drop all elements
        for i in 0..self.len {
            // SAFETY: pointer valid from alloc, i < len checked by loop bounds
            unsafe {
                ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }
        self.len = 0;
    }

    /// Shrink the vector to the given size
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }

        // Drop the excess elements
        for i in len..self.len {
            // SAFETY: pointer valid from alloc, i < self.len checked by loop bounds
            unsafe {
                ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }
        self.len = len;
    }

    /// Reallocate the vector with cache-aligned memory
    fn reallocate(&mut self, new_capacity: usize) -> Result<()> {
        if new_capacity == 0 {
            return Ok(());
        }

        if mem::size_of::<T>() == 0 {
            // A zero-sized element needs no storage, and `NonNull::dangling()`
            // is already a valid, correctly aligned address for any number of
            // them. Falling through would divide by `size_of::<T>()` below.
            self.capacity = usize::MAX;
            return Ok(());
        }

        // Round the byte count up to a whole number of cache lines, then back
        // to whatever element count that buys.
        let aligned_capacity = align_to_cache_line(
            new_capacity
                .checked_mul(mem::size_of::<T>())
                .ok_or_else(|| ZiporaError::invalid_data("Capacity overflow"))?,
        ) / mem::size_of::<T>();

        let layout = Self::layout_for(aligned_capacity)?;

        let new_ptr = if self.capacity == 0 {
            // First allocation - use NUMA-aware allocation if possible
            numa_alloc(layout, self.numa_node)?
        } else {
            // Reallocation - try to preserve NUMA locality
            let old_layout = Self::layout_for(self.capacity)?;

            let new_ptr = numa_alloc(layout, self.numa_node)?;

            // SAFETY: both pointers valid from alloc, non-overlapping, len <= old capacity
            unsafe {
                ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
            }

            // SAFETY: ptr from matching alloc, layout matches allocation
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
            }

            new_ptr
        };

        self.ptr = new_ptr;
        self.capacity = aligned_capacity;
        Ok(())
    }
}

impl<T> Drop for CacheAlignedVec<T> {
    fn drop(&mut self) {
        // Drop all elements first
        self.clear();

        // Deallocate memory. A zero-sized element type never allocated
        // (`reallocate` short-circuits and parks `capacity` at `usize::MAX`),
        // so there is nothing to give back.
        if self.capacity > 0 && mem::size_of::<T>() != 0 {
            let layout = Self::layout_for(self.capacity)
                .expect("layout was valid when this capacity was allocated");

            // SAFETY: `ptr` came from `numa_alloc` with exactly this layout --
            // `Self::layout_for(self.capacity)`, the same call `reallocate`
            // made -- and is deallocated here once.
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

// SAFETY: CacheAlignedVec<T> is Send if T is Send (same as Vec<T>)
unsafe impl<T: Send> Send for CacheAlignedVec<T> {}
// SAFETY: CacheAlignedVec<T> is Sync if T is Sync (same as Vec<T>)
unsafe impl<T: Sync> Sync for CacheAlignedVec<T> {}

impl<T> Default for CacheAlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Align a size to cache line boundaries
fn align_to_cache_line(size: usize) -> usize {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// NUMA-aware memory allocation.
///
/// `NumaMemoryPool` does not cache blocks. An earlier version filed freed
/// blocks into three size-category caches keyed only by `layout.size()`,
/// discarding each block's real size and alignment, and its `Drop` then freed
/// every cached pointer with a hardcoded `Layout` — 1 KiB/8, 64 KiB/16,
/// 1 MiB/32 — that the allocation never had. Deallocating with a layout other
/// than the allocating one is undefined behaviour (C3.3). Nothing ever read
/// those caches back, so they were pure leak plus time bomb; the pool is now
/// an accounting hook only, and every block is released with its own layout.
fn numa_alloc<T>(layout: Layout, preferred_node: Option<NumaNode>) -> Result<NonNull<T>> {
    if layout.size() == 0 {
        // `std::alloc::alloc` requires a non-zero size; `numa_alloc_aligned(0, ..)`
        // is reachable from safe code and used to hand it a zero-sized layout.
        return Err(ZiporaError::invalid_data(
            "NUMA allocation size must be non-zero",
        ));
    }

    // SAFETY: `layout.size() > 0` is checked immediately above and
    // `Layout::from_size_align` already guaranteed a power-of-two alignment.
    let ptr = unsafe { alloc(layout) };

    if ptr.is_null() {
        return Err(ZiporaError::out_of_memory(layout.size()));
    }

    // Try to bind to NUMA node if specified
    if let Some(node) = preferred_node {
        bind_to_numa_node(ptr, layout.size(), node);
        if let Ok(pools) = NUMA_MANAGER.node_pools.lock()
            && let Some(pool) = pools.get(&node)
        {
            pool.record_alloc(layout.size());
        }
    }

    // SAFETY: null check performed above
    Ok(unsafe { NonNull::new_unchecked(ptr as *mut T) })
}

/// NUMA node management
struct NumaNodeManager {
    node_count: AtomicUsize,
    thread_nodes: RwLock<HashMap<std::thread::ThreadId, NumaNode>>,
    node_pools: Mutex<HashMap<NumaNode, NumaMemoryPool>>,
}

/// Per-node accounting for NUMA allocations.
///
/// This deliberately holds no block cache. See `numa_alloc` for why: a cache
/// that stores bare addresses cannot free them with the layout they were
/// allocated with, and every reuse path that could have made the cache
/// worthwhile was already removed.
struct NumaMemoryPool {
    allocated_bytes: AtomicUsize,
}

impl NumaMemoryPool {
    fn new() -> Self {
        Self {
            allocated_bytes: AtomicUsize::new(0),
        }
    }

    fn record_alloc(&self, size: usize) {
        self.allocated_bytes.fetch_add(size, Ordering::Relaxed);
    }

    /// Release `ptr` with the layout it was allocated with.
    ///
    /// # Safety
    ///
    /// `ptr` must have been allocated by `numa_alloc` with exactly `layout`.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the caller guarantees `ptr` came from `alloc(layout)` and has
        // not been freed; `numa_dealloc` reconstructs `layout` with the same
        // `size`/`align.max(CACHE_LINE_SIZE)` expression `numa_alloc_aligned`
        // used, so the two layouts are identical by construction.
        unsafe {
            dealloc(ptr.as_ptr(), layout);
        }
        // Saturating: a block freed on a node that never recorded it (the pool
        // was created after the allocation) must not wrap the counter.
        self.allocated_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |b| {
                Some(b.saturating_sub(layout.size()))
            })
            .ok();
    }

    fn stats(&self) -> NumaPoolStats {
        NumaPoolStats {
            allocated_bytes: self.allocated_bytes.load(Ordering::Relaxed),
        }
    }
}

static NUMA_MANAGER: std::sync::LazyLock<NumaNodeManager> =
    std::sync::LazyLock::new(|| NumaNodeManager {
        node_count: AtomicUsize::new(detect_numa_nodes()),
        thread_nodes: RwLock::new(HashMap::new()),
        node_pools: Mutex::new(HashMap::new()),
    });

/// Detect the number of NUMA nodes on the system
fn detect_numa_nodes() -> usize {
    // Try to detect NUMA nodes - fallback to 1 if not available
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/sys/devices/system/node/online") {
            // Parse format like "0-3" or "0,2,4"
            if let Some(hyphen_pos) = contents.find('-')
                && let Ok(max_node) = contents[hyphen_pos + 1..].trim().parse::<usize>()
            {
                return max_node + 1;
            }
            // Count comma-separated nodes
            return contents.split(',').count();
        }
    }

    // Fallback: assume single NUMA node
    1
}

/// Get the current thread's preferred NUMA node
fn get_current_numa_node() -> Option<NumaNode> {
    let thread_id = std::thread::current().id();

    // Check if thread already has a preferred node
    if let Ok(nodes) = NUMA_MANAGER.thread_nodes.read()
        && let Some(&node) = nodes.get(&thread_id)
    {
        return Some(node);
    }

    // Assign a node using round-robin
    let node_count = NUMA_MANAGER.node_count.load(Ordering::Relaxed);
    if node_count > 1 {
        // Use thread ID hash for consistent assignment - simplified hash since as_u64 is unstable
        let thread_hash = format!("{:?}", thread_id).len(); // Simple hash based on debug string
        let node = (thread_hash.wrapping_mul(0x9e3779b9)) % node_count;

        // Store the assignment
        if let Ok(mut nodes) = NUMA_MANAGER.thread_nodes.write() {
            nodes.insert(thread_id, node);
        }

        Some(node)
    } else {
        None
    }
}

/// Bind memory to a specific NUMA node (platform-specific)
fn bind_to_numa_node(ptr: *mut u8, size: usize, node: NumaNode) {
    // For now, this is a no-op as we don't want to depend on libnuma
    // In a real implementation, this would use libnuma or syscalls
    // The allocation strategy still provides NUMA awareness through thread-local allocation
    let _ = (ptr, size, node);
}

/// Get NUMA statistics
#[derive(Debug, Clone)]
pub struct NumaStats {
    pub node_count: usize,
    pub current_node: Option<NumaNode>,
    pub thread_assignments: usize,
    pub pools: HashMap<NumaNode, NumaPoolStats>,
}

/// Statistics for a NUMA memory pool.
///
/// The `hit_count` / `miss_count` / `cached_*` fields and the `hit_rate()` and
/// `total_cached()` accessors were removed in C3.3: the pool has no block
/// cache, so those five numbers were structurally always zero — `hit_rate()`
/// could only ever return `0.0`. `allocated_bytes` is now real: it is charged
/// in `numa_alloc` when a node is named and credited back in `numa_dealloc`.
#[derive(Debug, Clone)]
pub struct NumaPoolStats {
    /// Live bytes allocated through `numa_alloc_aligned` on this node.
    pub allocated_bytes: usize,
}

/// Get current NUMA statistics
pub fn get_numa_stats() -> NumaStats {
    let node_count = NUMA_MANAGER.node_count.load(Ordering::Relaxed);
    let current_node = get_current_numa_node();
    let thread_assignments = NUMA_MANAGER
        .thread_nodes
        .read()
        .map(|nodes| nodes.len())
        .unwrap_or(0);

    // Collect pool statistics
    let mut pools = HashMap::new();
    if let Ok(node_pools) = NUMA_MANAGER.node_pools.lock() {
        for (&node, pool) in node_pools.iter() {
            pools.insert(node, pool.stats());
        }
    }

    NumaStats {
        node_count,
        current_node,
        thread_assignments,
        pools,
    }
}

/// Set the preferred NUMA node for the current thread
pub fn set_current_numa_node(node: NumaNode) -> Result<()> {
    let node_count = NUMA_MANAGER.node_count.load(Ordering::Relaxed);
    if node >= node_count {
        return Err(ZiporaError::invalid_data(format!(
            "NUMA node {} is invalid (max: {})",
            node,
            node_count - 1
        )));
    }

    let thread_id = std::thread::current().id();
    if let Ok(mut nodes) = NUMA_MANAGER.thread_nodes.write() {
        nodes.insert(thread_id, node);
    }

    Ok(())
}

/// Allocate memory on a specific NUMA node with cache alignment
pub fn numa_alloc_aligned(size: usize, align: usize, node: NumaNode) -> Result<NonNull<u8>> {
    let layout = Layout::from_size_align(size, align.max(CACHE_LINE_SIZE))
        .map_err(|_| ZiporaError::invalid_data("Invalid layout for NUMA allocation"))?;

    numa_alloc::<u8>(layout, Some(node))
}

/// Deallocate a block previously obtained from [`numa_alloc_aligned`].
///
/// # Safety
///
/// * `ptr` must have been returned by [`numa_alloc_aligned`] with the same
///   `(size, align, node)` arguments and must not yet have been deallocated.
/// * Neither [`numa_alloc_aligned`] nor `NumaMemoryPool` records live
///   allocations, so a foreign pointer, a mismatched `(size, align)` pair, or
///   a double free is handed straight to [`std::alloc::dealloc`].
///
/// # Errors
///
/// Returns [`ZiporaError::InvalidData`] if `(size, align)` cannot form a valid
/// [`Layout`], in which case `ptr` is not touched.
///
/// ```compile_fail
/// use zipora::memory::{numa_alloc_aligned, numa_dealloc};
///
/// let ptr = numa_alloc_aligned(64, 64, 0).unwrap();
/// // `numa_dealloc` is `unsafe fn`: calling it outside `unsafe` must not compile.
/// numa_dealloc(ptr, 64, 64, 0).unwrap();
/// ```
pub unsafe fn numa_dealloc(
    ptr: NonNull<u8>,
    size: usize,
    align: usize,
    node: NumaNode,
) -> Result<()> {
    let layout = Layout::from_size_align(size, align.max(CACHE_LINE_SIZE))
        .map_err(|_| ZiporaError::invalid_data("Invalid layout for NUMA deallocation"))?;

    if let Ok(pools) = NUMA_MANAGER.node_pools.lock()
        && let Some(pool) = pools.get(&node)
    {
        // SAFETY: the caller contract of `numa_dealloc` is that `ptr` came from
        // `numa_alloc_aligned(size, align, node)`, and `layout` above is built
        // with the identical `Layout::from_size_align(size, align.max(
        // CACHE_LINE_SIZE))` expression, so it is the allocating layout.
        unsafe {
            pool.deallocate(ptr, layout);
        }
        return Ok(());
    }

    // SAFETY: as above — `layout` is reconstructed from the same `size`/`align`
    // the caller passed to `numa_alloc_aligned`.
    unsafe {
        dealloc(ptr.as_ptr(), layout);
    }
    Ok(())
}

/// Get the optimal NUMA node for the current thread
pub fn get_optimal_numa_node() -> NumaNode {
    get_current_numa_node().unwrap_or(0)
}

/// Initialize NUMA pools for all detected nodes
pub fn init_numa_pools() -> Result<()> {
    let node_count = NUMA_MANAGER.node_count.load(Ordering::Relaxed);

    if let Ok(mut pools) = NUMA_MANAGER.node_pools.lock() {
        for node in 0..node_count {
            pools.entry(node).or_insert_with(NumaMemoryPool::new);
        }
    }

    Ok(())
}

/// Clear all NUMA pools and reset statistics
pub fn clear_numa_pools() -> Result<()> {
    if let Ok(mut pools) = NUMA_MANAGER.node_pools.lock() {
        pools.clear();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An element whose alignment exceeds a cache line. `__m512i` wrappers and
    /// anything carrying `#[repr(align(128))]` land here.
    #[repr(align(128))]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct OverAligned(u64);

    /// C3.9. Every `Layout` in `CacheAlignedVec` was built with
    /// `CACHE_LINE_SIZE` as the alignment and never consulted
    /// `align_of::<T>()`, in `reallocate` (twice, for the new and the old
    /// layout) and again in `Drop`. For any `T` aligned more strictly than a
    /// cache line the backing store is under-aligned, and the `ptr::write` in
    /// `push` is then undefined behaviour -- from a safe public API, for a
    /// perfectly ordinary type.
    ///
    /// This is deterministic in a debug build without any sanitizer: the
    /// standard library's own `slice::from_raw_parts` precondition check fires
    /// inside `as_slice` and aborts the process. In release that check is
    /// compiled out and the misalignment is silent, so `make miri_pool` is the
    /// release-mode oracle.
    #[test]
    fn test_cache_aligned_vec_honours_the_element_alignment() {
        const VECTORS: usize = 32;

        let mut vectors = Vec::with_capacity(VECTORS);
        for index in 0..VECTORS {
            let mut vector = CacheAlignedVec::<OverAligned>::new();
            vector.push(OverAligned(index as u64)).unwrap();
            vectors.push(vector);
        }

        for (index, vector) in vectors.iter().enumerate() {
            let address = vector.as_slice().as_ptr() as usize;
            assert_eq!(
                address % mem::align_of::<OverAligned>(),
                0,
                "vector {index}: base {address:#x} is not aligned to \
                 align_of::<OverAligned>() = {}",
                mem::align_of::<OverAligned>()
            );
        }
    }

    /// C3.9. `reallocate` computes
    /// `align_to_cache_line(new_capacity * size_of::<T>()) / size_of::<T>()`.
    /// For a zero-sized element that divides by zero, so pushing onto a
    /// `CacheAlignedVec<()>` panics from safe code.
    #[test]
    fn test_cache_aligned_vec_supports_zero_sized_elements() {
        let mut vector = CacheAlignedVec::<()>::new();
        for _ in 0..4 {
            vector.push(()).unwrap();
        }
        assert_eq!(vector.len(), 4);
        assert_eq!(vector.as_slice().len(), 4);
        assert_eq!(vector.pop(), Some(()));
        assert_eq!(vector.len(), 3);
        vector.clear();
        assert!(vector.is_empty());
    }

    /// C3.9. A zero-sized element type that also carries a `Drop` must have
    /// that `Drop` run exactly once per element, not once per allocated byte.
    #[test]
    fn test_zero_sized_elements_are_dropped_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct CountsItsDrop;
        impl Drop for CountsItsDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }

        DROPS.store(0, Ordering::Relaxed);
        {
            let mut vector = CacheAlignedVec::<CountsItsDrop>::new();
            for _ in 0..5 {
                vector.push(CountsItsDrop).unwrap();
            }
            assert_eq!(DROPS.load(Ordering::Relaxed), 0);
        }
        assert_eq!(DROPS.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_cache_aligned_vec_basic() {
        let mut vec = CacheAlignedVec::<i32>::new();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());

        vec.push(42).unwrap();
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.get(0), Some(&42));

        vec.push(24).unwrap();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.as_slice(), &[42, 24]);
    }

    #[test]
    fn test_cache_aligned_vec_capacity() {
        let vec = CacheAlignedVec::<u64>::with_capacity(10).unwrap();
        assert!(vec.capacity() >= 10);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_cache_aligned_vec_pop() {
        let mut vec = CacheAlignedVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.len(), 1);
    }

    #[test]
    fn test_cache_aligned_vec_clear() {
        let mut vec = CacheAlignedVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_cache_aligned_vec_truncate() {
        let mut vec = CacheAlignedVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();
        vec.push(4).unwrap();

        vec.truncate(2);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.as_slice(), &[1, 2]);
    }

    #[test]
    fn test_cache_alignment() {
        let vec = CacheAlignedVec::<u8>::new();
        let ptr = &vec as *const _ as usize;
        assert_eq!(
            ptr % CACHE_LINE_SIZE,
            0,
            "CacheAlignedVec should be cache-line aligned"
        );
    }

    #[test]
    fn test_align_to_cache_line() {
        assert_eq!(align_to_cache_line(0), 0);
        assert_eq!(align_to_cache_line(1), CACHE_LINE_SIZE);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE), CACHE_LINE_SIZE);
        assert_eq!(
            align_to_cache_line(CACHE_LINE_SIZE + 1),
            CACHE_LINE_SIZE * 2
        );
    }

    #[test]
    fn test_numa_detection() {
        let stats = get_numa_stats();
        assert!(stats.node_count >= 1);
    }

    #[test]
    fn test_numa_node_assignment() {
        let node = get_current_numa_node();
        // Should get consistent assignment for the same thread
        let node2 = get_current_numa_node();
        assert_eq!(node, node2);
    }

    #[test]
    fn test_numa_vec_creation() {
        let vec = CacheAlignedVec::<i32>::with_numa_node(0);
        assert_eq!(vec.numa_node(), Some(0));
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_set_numa_node() {
        // Should work for valid node 0
        assert!(set_current_numa_node(0).is_ok());

        // Should get the node we just set
        assert_eq!(get_current_numa_node(), Some(0));
    }

    #[test]
    fn test_large_allocation() {
        let mut vec = CacheAlignedVec::<u64>::with_capacity(1000).unwrap();
        for i in 0..1000u64 {
            vec.push(i).unwrap();
        }
        assert_eq!(vec.len(), 1000);

        // Verify data integrity
        for i in 0..1000u64 {
            assert_eq!(vec.get(i as usize), Some(&i));
        }
    }

    #[test]
    fn test_numa_alloc_dealloc() {
        let node = 0;
        let ptr = numa_alloc_aligned(1024, 64, node).unwrap();

        // Verify alignment
        assert_eq!(ptr.as_ptr() as usize % CACHE_LINE_SIZE, 0);

        // Test deallocation
        // SAFETY: `ptr` came from `numa_alloc_aligned(1024, 64, node)`.
        assert!(unsafe { numa_dealloc(ptr, 1024, 64, node) }.is_ok());
    }

    #[test]
    fn test_numa_pool_initialization() {
        clear_numa_pools().unwrap();
        assert!(init_numa_pools().is_ok());

        let stats = get_numa_stats();
        assert!(stats.node_count >= 1);
    }

    #[test]
    fn test_numa_pool_stats() {
        clear_numa_pools().unwrap();
        init_numa_pools().unwrap();

        // Allocate some memory to test allocation still works
        // Note: Pool caching is disabled due to heap corruption fix,
        // so we just verify allocations succeed
        let ptr1 = numa_alloc_aligned(1024, 64, 0).unwrap();
        let ptr2 = numa_alloc_aligned(512, 32, 0).unwrap();

        // Verify allocations are properly aligned
        assert_eq!(ptr1.as_ptr() as usize % 64, 0);
        assert_eq!(ptr2.as_ptr() as usize % 32, 0);

        // Stats structure should exist
        let stats = get_numa_stats();
        assert!(stats.node_count >= 1);

        // SAFETY: each pointer is freed with its allocating `(size, align, node)`.
        unsafe {
            numa_dealloc(ptr1, 1024, 64, 0).unwrap();
            numa_dealloc(ptr2, 512, 32, 0).unwrap();
        }
    }

    /// C3.3 (CRITICAL, Miri-confirmed). `NumaMemoryPool::deallocate` parked the
    /// pointer in a size-category cache keyed on `layout.size()` alone and
    /// discarded its real size and alignment; nothing ever read the cache back,
    /// and `Drop` then freed every cached pointer with a hardcoded `Layout`
    /// (1 KiB/8, 64 KiB/16, 1 MiB/32) the allocation never had. Deallocating
    /// with a layout other than the allocating one is UB.
    ///
    /// The existing suite already walked this: `test_numa_alloc_dealloc` frees
    /// a `Layout(1024, 64)` block, which landed in `medium_chunks`, which
    /// `test_numa_pool_stats`'s `clear_numa_pools()` then freed as
    /// `Layout(65536, 16)`.
    ///
    /// Oracle: `make miri_pool`. Under Miri at the parent commit this reports
    /// "incorrect layout on deallocation". The assertions below are the
    /// default-suite half: `allocated_bytes` must actually track the block.
    #[test]
    fn test_numa_dealloc_releases_the_block_with_its_own_layout() {
        let node = 0;
        init_numa_pools().unwrap();

        let before = get_numa_stats()
            .pools
            .get(&node)
            .map(|p| p.allocated_bytes)
            .unwrap_or(0);

        // An over-aligned block: its layout is Layout(64, 64), which the old
        // cache would have filed as "small" and freed as Layout(1024, 8).
        let ptr = numa_alloc_aligned(64, 64, node).unwrap();
        assert_eq!(ptr.as_ptr() as usize % CACHE_LINE_SIZE, 0);

        let during = get_numa_stats()
            .pools
            .get(&node)
            .map(|p| p.allocated_bytes)
            .unwrap_or(0);
        assert!(
            during >= before + 64,
            "numa_alloc must charge the block to its node: {before} -> {during}"
        );

        // SAFETY: `ptr` came from `numa_alloc_aligned(64, 64, node)`.
        unsafe {
            numa_dealloc(ptr, 64, 64, node).unwrap();
        }

        let after = get_numa_stats()
            .pools
            .get(&node)
            .map(|p| p.allocated_bytes)
            .unwrap_or(0);
        assert!(
            after <= during - 64,
            "numa_dealloc must credit the block back: {during} -> {after}"
        );
    }

    /// C3.3. `numa_alloc_aligned(0, ..)` built a valid zero-sized `Layout` and
    /// handed it to `std::alloc::alloc`, whose contract forbids that.
    #[test]
    fn test_numa_alloc_rejects_zero_size() {
        let err =
            numa_alloc_aligned(0, 64, 0).expect_err("zero-sized NUMA allocation must be rejected");
        assert!(err.to_string().contains("non-zero"), "got: {err}");
    }

    #[test]
    fn test_optimal_numa_node() {
        let node = get_optimal_numa_node();
        let stats = get_numa_stats();
        assert!(node < stats.node_count);
    }

    #[test]
    fn test_cache_aligned_vec_with_numa() {
        let node = 0;
        let mut vec = CacheAlignedVec::<i32>::with_numa_node(node);
        assert_eq!(vec.numa_node(), Some(node));

        vec.push(42).unwrap();
        assert_eq!(vec.get(0), Some(&42));
    }

    #[test]
    fn test_cache_aligned_vec_mutation() {
        let mut vec = CacheAlignedVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        if let Some(val) = vec.get_mut(1) {
            *val = 42;
        }

        assert_eq!(vec.as_slice(), &[1, 42, 3]);
    }

    #[test]
    fn test_cache_aligned_vec_large_capacity() {
        let capacity = 100000;
        let vec = CacheAlignedVec::<u8>::with_capacity(capacity).unwrap();
        assert!(vec.capacity() >= capacity);

        // Verify the allocation is cache-aligned
        let ptr = vec.as_slice().as_ptr() as usize;
        assert_eq!(ptr % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn test_error_handling() {
        let node_count = get_numa_stats().node_count;

        // Test invalid NUMA node
        let result = set_current_numa_node(node_count + 100);
        assert!(result.is_err());
    }
}
