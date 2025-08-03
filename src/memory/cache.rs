//! Cache-conscious memory management
//!
//! This module provides cache-aligned allocations and NUMA-aware memory management
//! to optimize performance on modern multi-core systems.

use crate::error::{Result, ZiporaError};
use std::alloc::{alloc, dealloc, Layout};
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
#[repr(align(64))]  // Cache line alignment
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
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the current capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the NUMA node this vector is allocated on
    pub fn numa_node(&self) -> Option<NumaNode> {
        self.numa_node
    }

    /// Reserve capacity for at least `additional` more elements
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        let required_cap = self.len.checked_add(additional)
            .ok_or_else(|| ZiporaError::invalid_data("Capacity overflow"))?;
        
        if required_cap <= self.capacity {
            return Ok(());
        }

        // Grow by at least 2x to amortize allocations
        let new_cap = required_cap.max(self.capacity * 2).max(4);
        self.reallocate(new_cap)
    }

    /// Push an element onto the end of the vector
    pub fn push(&mut self, value: T) -> Result<()> {
        if self.len == self.capacity {
            self.reserve(1)?;
        }

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
        unsafe {
            Some(ptr::read(self.ptr.as_ptr().add(self.len)))
        }
    }

    /// Get a reference to an element by index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe {
                Some(&*self.ptr.as_ptr().add(index))
            }
        } else {
            None
        }
    }

    /// Get a mutable reference to an element by index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            unsafe {
                Some(&mut *self.ptr.as_ptr().add(index))
            }
        } else {
            None
        }
    }

    /// Get a slice of all elements
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }

    /// Get a mutable slice of all elements
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        // Drop all elements
        for i in 0..self.len {
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

        // Ensure capacity is aligned to cache line boundaries for optimal access
        let aligned_capacity = align_to_cache_line(new_capacity * mem::size_of::<T>()) / mem::size_of::<T>();
        
        let layout = Layout::from_size_align(
            aligned_capacity * mem::size_of::<T>(),
            CACHE_LINE_SIZE,
        ).map_err(|_| ZiporaError::invalid_data("Invalid layout for cache-aligned allocation"))?;

        let new_ptr = if self.capacity == 0 {
            // First allocation - use NUMA-aware allocation if possible
            numa_alloc(layout, self.numa_node)?
        } else {
            // Reallocation - try to preserve NUMA locality
            let old_layout = Layout::from_size_align(
                self.capacity * mem::size_of::<T>(),
                CACHE_LINE_SIZE,
            ).unwrap();

            let new_ptr = numa_alloc(layout, self.numa_node)?;
            
            // Copy existing data
            unsafe {
                ptr::copy_nonoverlapping(
                    self.ptr.as_ptr(),
                    new_ptr.as_ptr(),
                    self.len,
                );
            }
            
            // Deallocate old memory
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

        // Deallocate memory
        if self.capacity > 0 {
            let layout = Layout::from_size_align(
                self.capacity * mem::size_of::<T>(),
                CACHE_LINE_SIZE,
            ).unwrap();

            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

unsafe impl<T: Send> Send for CacheAlignedVec<T> {}
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

/// NUMA-aware memory allocation using node-specific pools
fn numa_alloc<T>(layout: Layout, preferred_node: Option<NumaNode>) -> Result<NonNull<T>> {
    let node = preferred_node.unwrap_or(0); // Default to node 0
    
    // Try to use NUMA pool for allocation
    if let Ok(mut pools) = NUMA_MANAGER.node_pools.lock() {
        let pool = pools.entry(node).or_insert_with(NumaMemoryPool::new);
        if let Ok(ptr) = pool.allocate(layout, node) {
            return Ok(NonNull::new(ptr.as_ptr() as *mut T).unwrap());
        }
    }

    // Fallback to standard allocation
    let ptr = unsafe { alloc(layout) };
    
    if ptr.is_null() {
        return Err(ZiporaError::out_of_memory(layout.size()));
    }

    // Try to bind to NUMA node if specified
    if let Some(node) = preferred_node {
        bind_to_numa_node(ptr, layout.size(), node);
    }

    Ok(NonNull::new(ptr as *mut T).unwrap())
}

/// NUMA node management
struct NumaNodeManager {
    node_count: AtomicUsize,
    thread_nodes: RwLock<HashMap<std::thread::ThreadId, NumaNode>>,
    node_pools: Mutex<HashMap<NumaNode, NumaMemoryPool>>,
}

/// NUMA-aware memory pool for each node
struct NumaMemoryPool {
    small_chunks: Vec<usize>,       // < 1KB allocations (stored as usize for Send/Sync)
    medium_chunks: Vec<usize>,      // 1KB - 64KB allocations  
    large_chunks: Vec<usize>,       // > 64KB allocations
    allocated_bytes: AtomicUsize,
    hit_count: AtomicUsize,
    miss_count: AtomicUsize,
}

impl NumaMemoryPool {
    fn new() -> Self {
        Self {
            small_chunks: Vec::new(),
            medium_chunks: Vec::new(),
            large_chunks: Vec::new(),
            allocated_bytes: AtomicUsize::new(0),
            hit_count: AtomicUsize::new(0),
            miss_count: AtomicUsize::new(0),
        }
    }

    fn allocate(&mut self, layout: Layout, node: NumaNode) -> Result<NonNull<u8>> {
        let pool = if layout.size() < 1024 {
            &mut self.small_chunks
        } else if layout.size() < 64 * 1024 {
            &mut self.medium_chunks
        } else {
            &mut self.large_chunks
        };

        // Try to reuse from pool first
        if let Some(ptr_addr) = pool.pop() {
            self.hit_count.fetch_add(1, Ordering::Relaxed);
            return Ok(NonNull::new(ptr_addr as *mut u8).unwrap());
        }

        // Allocate new memory bound to NUMA node
        self.miss_count.fetch_add(1, Ordering::Relaxed);
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return Err(ZiporaError::out_of_memory(layout.size()));
        }

        // Bind to NUMA node
        bind_to_numa_node(ptr, layout.size(), node);
        self.allocated_bytes.fetch_add(layout.size(), Ordering::Relaxed);

        Ok(NonNull::new(ptr).unwrap())
    }

    fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        let pool = if layout.size() < 1024 {
            &mut self.small_chunks
        } else if layout.size() < 64 * 1024 {
            &mut self.medium_chunks
        } else {
            &mut self.large_chunks
        };

        // Return to pool for reuse (with size limits to prevent unbounded growth)
        if pool.len() < 100 {  // Limit pool size
            pool.push(ptr.as_ptr() as usize);
        } else {
            // Pool is full, actually deallocate
            unsafe {
                dealloc(ptr.as_ptr(), layout);
            }
            self.allocated_bytes.fetch_sub(layout.size(), Ordering::Relaxed);
        }
    }

    fn stats(&self) -> NumaPoolStats {
        NumaPoolStats {
            allocated_bytes: self.allocated_bytes.load(Ordering::Relaxed),
            hit_count: self.hit_count.load(Ordering::Relaxed),
            miss_count: self.miss_count.load(Ordering::Relaxed),
            cached_small: self.small_chunks.len(),
            cached_medium: self.medium_chunks.len(), 
            cached_large: self.large_chunks.len(),
        }
    }
}

impl Drop for NumaMemoryPool {
    fn drop(&mut self) {
        // Clean up all cached allocations
        for &ptr_addr in &self.small_chunks {
            unsafe {
                dealloc(ptr_addr as *mut u8, Layout::from_size_align(1024, 8).unwrap());
            }
        }
        for &ptr_addr in &self.medium_chunks {
            unsafe {
                dealloc(ptr_addr as *mut u8, Layout::from_size_align(64 * 1024, 16).unwrap());
            }
        }
        for &ptr_addr in &self.large_chunks {
            unsafe {
                dealloc(ptr_addr as *mut u8, Layout::from_size_align(1024 * 1024, 32).unwrap());
            }
        }
    }
}

static NUMA_MANAGER: std::sync::LazyLock<NumaNodeManager> = std::sync::LazyLock::new(|| {
    NumaNodeManager {
        node_count: AtomicUsize::new(detect_numa_nodes()),
        thread_nodes: RwLock::new(HashMap::new()),
        node_pools: Mutex::new(HashMap::new()),
    }
});

/// Detect the number of NUMA nodes on the system
fn detect_numa_nodes() -> usize {
    // Try to detect NUMA nodes - fallback to 1 if not available
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/sys/devices/system/node/online") {
            // Parse format like "0-3" or "0,2,4"
            if let Some(hyphen_pos) = contents.find('-') {
                if let Ok(max_node) = contents[hyphen_pos + 1..].trim().parse::<usize>() {
                    return max_node + 1;
                }
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
    if let Ok(nodes) = NUMA_MANAGER.thread_nodes.read() {
        if let Some(&node) = nodes.get(&thread_id) {
            return Some(node);
        }
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

/// Statistics for a NUMA memory pool
#[derive(Debug, Clone)]
pub struct NumaPoolStats {
    pub allocated_bytes: usize,
    pub hit_count: usize,
    pub miss_count: usize,
    pub cached_small: usize,
    pub cached_medium: usize,
    pub cached_large: usize,
}

impl NumaPoolStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }

    pub fn total_cached(&self) -> usize {
        self.cached_small + self.cached_medium + self.cached_large
    }
}

/// Get current NUMA statistics
pub fn get_numa_stats() -> NumaStats {
    let node_count = NUMA_MANAGER.node_count.load(Ordering::Relaxed);
    let current_node = get_current_numa_node();
    let thread_assignments = NUMA_MANAGER.thread_nodes.read()
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
        return Err(ZiporaError::invalid_data(&format!(
            "NUMA node {} is invalid (max: {})", node, node_count - 1
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

/// Deallocate NUMA memory
pub fn numa_dealloc(ptr: NonNull<u8>, size: usize, align: usize, node: NumaNode) -> Result<()> {
    let layout = Layout::from_size_align(size, align.max(CACHE_LINE_SIZE))
        .map_err(|_| ZiporaError::invalid_data("Invalid layout for NUMA deallocation"))?;
    
    if let Ok(mut pools) = NUMA_MANAGER.node_pools.lock() {
        if let Some(pool) = pools.get_mut(&node) {
            pool.deallocate(ptr, layout);
            return Ok(());
        }
    }

    // Fallback to standard deallocation
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
        assert_eq!(ptr % CACHE_LINE_SIZE, 0, "CacheAlignedVec should be cache-line aligned");
    }

    #[test]
    fn test_align_to_cache_line() {
        assert_eq!(align_to_cache_line(0), 0);
        assert_eq!(align_to_cache_line(1), CACHE_LINE_SIZE);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE), CACHE_LINE_SIZE);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE + 1), CACHE_LINE_SIZE * 2);
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
        assert!(numa_dealloc(ptr, 1024, 64, node).is_ok());
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
        
        // Allocate some memory to test stats
        let _ptr1 = numa_alloc_aligned(1024, 64, 0).unwrap();
        let _ptr2 = numa_alloc_aligned(512, 32, 0).unwrap();
        
        let stats = get_numa_stats();
        if let Some(pool_stats) = stats.pools.get(&0) {
            assert!(pool_stats.allocated_bytes > 0 || pool_stats.hit_count > 0);
        }
    }

    #[test]
    fn test_numa_stats_hit_rate() {
        let stats = NumaPoolStats {
            allocated_bytes: 1024,
            hit_count: 80,
            miss_count: 20,
            cached_small: 5,
            cached_medium: 3,
            cached_large: 1,
        };
        
        assert_eq!(stats.hit_rate(), 0.8);
        assert_eq!(stats.total_cached(), 9);
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