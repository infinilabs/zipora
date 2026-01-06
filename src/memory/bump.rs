//! Bump allocator for very fast sequential allocations
//!
//! Bump allocators are extremely fast for allocation-heavy workloads where
//! objects have similar lifetimes and can be freed all at once.

use crate::error::{Result, ZiporaError};
use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// A bump allocator that allocates memory sequentially from a large buffer
///
/// # Thread Safety
///
/// This allocator is both `Send` and `Sync`:
/// - **Send**: The allocator can be transferred between threads safely because
///   it owns its memory buffer via `NonNull<u8>` and all mutable state uses atomics.
/// - **Sync**: The allocator can be shared between threads safely because:
///   - `buffer` is a raw pointer that is only dereferenced in `alloc_bytes` after
///     atomic coordination via `current`
///   - `current` uses `AtomicUsize` with proper memory ordering to coordinate
///     concurrent allocations (CAS loop prevents data races)
///   - `allocated_bytes` is `AtomicU64` for thread-safe statistics
///   - `capacity` is immutable after construction
///
/// Note: While thread-safe, concurrent allocations may experience contention
/// in the CAS loop. For high-contention scenarios, consider using per-thread
/// bump allocators or the thread-local pool variants.
pub struct BumpAllocator {
    buffer: NonNull<u8>,
    capacity: usize,
    /// Current allocation offset, atomically updated for thread-safe bump allocation
    current: AtomicUsize,
    allocated_bytes: AtomicU64,
}

impl BumpAllocator {
    /// Create a new bump allocator with the specified capacity
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(ZiporaError::invalid_data("capacity cannot be zero"));
        }

        let layout = Layout::from_size_align(capacity, 8)
            .map_err(|_| ZiporaError::invalid_data("invalid layout for bump allocator"))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(ZiporaError::out_of_memory(capacity));
        }

        Ok(Self {
            buffer: unsafe { NonNull::new_unchecked(ptr) },
            capacity,
            current: AtomicUsize::new(0),
            allocated_bytes: AtomicU64::new(0),
        })
    }

    /// Allocate memory for an object of type T
    pub fn alloc<T>(&self) -> Result<NonNull<T>> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        self.alloc_bytes(size, align).map(|ptr| ptr.cast())
    }

    /// Allocate a slice of objects of type T
    pub fn alloc_slice<T>(&self, count: usize) -> Result<NonNull<[T]>> {
        let size = std::mem::size_of::<T>() * count;
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc_bytes(size, align)?;

        // Create a fat pointer for the slice
        let slice_ptr = std::ptr::slice_from_raw_parts_mut(ptr.as_ptr() as *mut T, count);
        Ok(unsafe { NonNull::new_unchecked(slice_ptr) })
    }

    /// Allocate raw bytes with specified alignment
    ///
    /// This method is thread-safe and uses compare-and-swap to atomically
    /// reserve space in the buffer. Under high contention, the CAS loop
    /// will retry until successful or until the buffer is exhausted.
    pub fn alloc_bytes(&self, size: usize, align: usize) -> Result<NonNull<u8>> {
        if size == 0 {
            return Err(ZiporaError::invalid_data("allocation size cannot be zero"));
        }

        if !align.is_power_of_two() {
            return Err(ZiporaError::invalid_data(
                "alignment must be a power of two",
            ));
        }

        // Use CAS loop for thread-safe bump allocation
        loop {
            let current = self.current.load(Ordering::Acquire);

            // Calculate aligned offset
            let aligned_offset = (current + align - 1) & !(align - 1);
            let new_offset = aligned_offset + size;

            if new_offset > self.capacity {
                return Err(ZiporaError::out_of_memory(size));
            }

            // Try to atomically update the current offset
            match self.current.compare_exchange_weak(
                current,
                new_offset,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.allocated_bytes
                        .fetch_add(size as u64, Ordering::Relaxed);

                    // SAFETY: aligned_offset is within bounds (checked above),
                    // and we successfully reserved this range via CAS
                    let ptr = unsafe { self.buffer.as_ptr().add(aligned_offset) };
                    return Ok(unsafe { NonNull::new_unchecked(ptr) });
                }
                Err(_) => {
                    // CAS failed, another thread allocated - retry
                    std::hint::spin_loop();
                    continue;
                }
            }
        }
    }

    /// Reset the allocator, making all memory available again
    ///
    /// # Safety
    ///
    /// This invalidates all previously allocated pointers. The caller must ensure
    /// that no allocated objects are accessed after calling this function.
    /// Additionally, the caller must ensure no other threads are concurrently
    /// allocating from this allocator during reset.
    pub unsafe fn reset(&self) {
        self.current.store(0, Ordering::Release);
        self.allocated_bytes.store(0, Ordering::Relaxed);
    }

    /// Get the number of bytes currently allocated
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Get the total capacity of the allocator
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the number of bytes remaining
    pub fn remaining_bytes(&self) -> usize {
        self.capacity - self.current.load(Ordering::Relaxed)
    }

    /// Check if the allocator can satisfy an allocation of the given size and alignment
    ///
    /// Note: This is a best-effort check in a concurrent context. Another thread
    /// may allocate between this check and the actual allocation.
    pub fn can_allocate(&self, size: usize, align: usize) -> bool {
        let current = self.current.load(Ordering::Relaxed);
        let aligned_offset = (current + align - 1) & !(align - 1);
        aligned_offset + size <= self.capacity
    }
}

// SAFETY: BumpAllocator is Send because:
// 1. `buffer: NonNull<u8>` - Raw pointer to heap-allocated memory owned by this struct.
//    The memory is allocated in `new()` and deallocated in `Drop`. No thread-local state.
// 2. `capacity: usize` - Immutable after construction, trivially Send.
// 3. `current: AtomicUsize` - AtomicUsize is Send.
// 4. `allocated_bytes: AtomicU64` - AtomicU64 is Send.
unsafe impl Send for BumpAllocator {}

// SAFETY: BumpAllocator is Sync because:
// 1. `buffer: NonNull<u8>` - Only accessed after successful CAS on `current`, which
//    provides synchronization. Each thread gets a unique, non-overlapping region.
// 2. `capacity: usize` - Immutable after construction, safe to read concurrently.
// 3. `current: AtomicUsize` - Atomic operations with Acquire/Release ordering ensure
//    proper synchronization for the bump allocation algorithm.
// 4. `allocated_bytes: AtomicU64` - Atomic updates are inherently thread-safe.
//
// The allocation algorithm uses compare_exchange_weak to atomically reserve space,
// ensuring no two threads receive overlapping memory regions.
unsafe impl Sync for BumpAllocator {}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        // Safety check: Only deallocate if we have a valid buffer
        if self.capacity > 0 {
            if let Ok(layout) = Layout::from_size_align(self.capacity, 8) {
                unsafe {
                    dealloc(self.buffer.as_ptr(), layout);
                }
            }
        }
    }
}

/// A scoped arena that automatically resets when dropped
pub struct BumpArena {
    allocator: BumpAllocator,
    initial_offset: usize,
}

impl BumpArena {
    /// Create a new arena with the specified capacity
    pub fn new(capacity: usize) -> Result<Self> {
        let allocator = BumpAllocator::new(capacity)?;
        Ok(Self {
            allocator,
            initial_offset: 0,
        })
    }

    /// Create a nested arena that resets to the current position when dropped
    pub fn scope(&self) -> BumpScope<'_> {
        BumpScope {
            allocator: &self.allocator,
            initial_offset: self.allocator.current.load(Ordering::Relaxed),
            initial_allocated_bytes: self.allocator.allocated_bytes(),
        }
    }

    /// Allocate memory for an object of type T
    pub fn alloc<T>(&self) -> Result<NonNull<T>> {
        self.allocator.alloc()
    }

    /// Allocate a slice of objects of type T
    pub fn alloc_slice<T>(&self, count: usize) -> Result<NonNull<[T]>> {
        self.allocator.alloc_slice(count)
    }

    /// Allocate raw bytes with specified alignment
    pub fn alloc_bytes(&self, size: usize, align: usize) -> Result<NonNull<u8>> {
        self.allocator.alloc_bytes(size, align)
    }

    /// Get allocation statistics
    pub fn stats(&self) -> BumpStats {
        BumpStats {
            allocated_bytes: self.allocator.allocated_bytes(),
            capacity: self.allocator.capacity(),
            remaining_bytes: self.allocator.remaining_bytes(),
        }
    }
}

impl Drop for BumpArena {
    fn drop(&mut self) {
        // Reset to initial position
        unsafe {
            self.allocator.reset();
        }
        self.allocator.current.store(self.initial_offset, Ordering::Relaxed);
    }
}

/// A scoped bump allocator that resets when dropped
pub struct BumpScope<'a> {
    allocator: &'a BumpAllocator,
    initial_offset: usize,
    initial_allocated_bytes: u64,
}

impl<'a> BumpScope<'a> {
    /// Allocate memory for an object of type T
    pub fn alloc<T>(&self) -> Result<NonNull<T>> {
        self.allocator.alloc()
    }

    /// Allocate a slice of objects of type T  
    pub fn alloc_slice<T>(&self, count: usize) -> Result<NonNull<[T]>> {
        self.allocator.alloc_slice(count)
    }

    /// Allocate raw bytes with specified alignment
    pub fn alloc_bytes(&self, size: usize, align: usize) -> Result<NonNull<u8>> {
        self.allocator.alloc_bytes(size, align)
    }

    /// Get allocation statistics
    pub fn stats(&self) -> BumpStats {
        BumpStats {
            allocated_bytes: self.allocator.allocated_bytes(),
            capacity: self.allocator.capacity(),
            remaining_bytes: self.allocator.remaining_bytes(),
        }
    }
}

impl<'a> Drop for BumpScope<'a> {
    fn drop(&mut self) {
        // Reset to initial position
        self.allocator.current.store(self.initial_offset, Ordering::Relaxed);

        // Reset allocated bytes counter to what it was when scope was created
        self.allocator
            .allocated_bytes
            .store(self.initial_allocated_bytes, Ordering::Relaxed);
    }
}

/// Statistics for bump allocator usage
#[derive(Debug, Clone)]
pub struct BumpStats {
    /// Number of bytes currently allocated
    pub allocated_bytes: u64,
    /// Total capacity of the allocator
    pub capacity: usize,
    /// Number of bytes remaining
    pub remaining_bytes: usize,
}

impl BumpStats {
    /// Get the utilization percentage (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        self.allocated_bytes as f64 / self.capacity as f64
    }

    /// Check if the allocator is nearly full (> 90% utilized)
    pub fn is_nearly_full(&self) -> bool {
        self.utilization() > 0.9
    }
}

/// A bump-allocated vector that can grow within the allocator
pub struct BumpVec<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    #[allow(dead_code)]
    allocator: &'a BumpAllocator,
}

impl<'a, T> BumpVec<'a, T> {
    /// Create a new bump vector with the specified initial capacity
    pub fn new_in(allocator: &'a BumpAllocator, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(ZiporaError::invalid_data("capacity cannot be zero"));
        }

        let ptr = allocator.alloc_slice::<T>(capacity)?;

        Ok(Self {
            ptr: ptr.cast(),
            len: 0,
            capacity,
            allocator,
        })
    }

    /// Push an element to the vector
    pub fn push(&mut self, item: T) -> Result<()> {
        if self.len >= self.capacity {
            return Err(ZiporaError::invalid_data("bump vector capacity exceeded"));
        }

        unsafe {
            self.ptr.as_ptr().add(self.len).write(item);
        }
        self.len += 1;
        Ok(())
    }

    /// Pop an element from the vector
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        Some(unsafe { self.ptr.as_ptr().add(self.len).read() })
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the vector
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get a slice of the vector's contents
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice of the vector's contents
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<'a, T> Drop for BumpVec<'a, T> {
    fn drop(&mut self) {
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                self.ptr.as_ptr().add(i).drop_in_place();
            }
        }
        // Memory is owned by the bump allocator, no need to deallocate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bump_allocator_creation() {
        let allocator = BumpAllocator::new(4096).unwrap();
        assert_eq!(allocator.capacity(), 4096);
        assert_eq!(allocator.allocated_bytes(), 0);
        assert_eq!(allocator.remaining_bytes(), 4096);
    }

    #[test]
    fn test_bump_allocation() {
        let allocator = BumpAllocator::new(4096).unwrap();

        let ptr1 = allocator.alloc::<u64>().unwrap();
        let ptr2 = allocator.alloc::<u64>().unwrap();

        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
        assert!(allocator.allocated_bytes() >= 16); // At least 2 * sizeof(u64)
        assert!(allocator.remaining_bytes() < 4096);
    }

    #[test]
    fn test_bump_slice_allocation() {
        let allocator = BumpAllocator::new(4096).unwrap();

        let mut slice_ptr = allocator.alloc_slice::<u32>(10).unwrap();
        let slice = unsafe { slice_ptr.as_mut() };

        assert_eq!(slice.len(), 10);

        // Initialize and verify the slice
        for (i, item) in slice.iter_mut().enumerate() {
            *item = i as u32;
        }

        for (i, item) in slice.iter().enumerate() {
            assert_eq!(*item, i as u32);
        }
    }

    #[test]
    fn test_bump_alignment() {
        let allocator = BumpAllocator::new(4096).unwrap();

        // Allocate a u8 to misalign the allocator
        let _ptr1 = allocator.alloc::<u8>().unwrap();

        // Allocate a u64, which requires 8-byte alignment
        let ptr2 = allocator.alloc::<u64>().unwrap();

        // Check that the pointer is properly aligned
        assert_eq!(ptr2.as_ptr() as usize % 8, 0);
    }

    #[test]
    fn test_bump_exhaustion() {
        // FIX: Increase capacity to account for alignment requirements
        // u64 requires 8-byte alignment, so we need extra space for alignment
        let allocator = BumpAllocator::new(24).unwrap();

        // Allocate until exhausted
        let _ptr1 = allocator.alloc::<u64>().unwrap(); // Uses 8 bytes
        let _ptr2 = allocator.alloc::<u64>().unwrap(); // Uses 8 bytes
        let _ptr3 = allocator.alloc::<u64>().unwrap(); // Uses 8 bytes (24 total)

        // Now this should fail - no space for another u64
        let result = allocator.alloc::<u64>();
        assert!(result.is_err(), "Should fail to allocate when exhausted");
    }

    #[test]
    fn test_bump_reset() {
        let allocator = BumpAllocator::new(4096).unwrap();

        let _ptr1 = allocator.alloc::<u64>().unwrap();
        let _ptr2 = allocator.alloc::<u64>().unwrap();

        assert!(allocator.allocated_bytes() > 0);
        assert!(allocator.remaining_bytes() < 4096);

        unsafe {
            allocator.reset();
        }

        assert_eq!(allocator.allocated_bytes(), 0);
        assert_eq!(allocator.remaining_bytes(), 4096);
    }

    #[test]
    fn test_bump_arena() {
        let arena = BumpArena::new(4096).unwrap();

        let ptr1 = arena.alloc::<u64>().unwrap();
        let ptr2 = arena.alloc::<u64>().unwrap();

        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());

        let stats = arena.stats();
        assert!(stats.allocated_bytes >= 16);
        assert!(stats.utilization() > 0.0);
        assert!(!stats.is_nearly_full());
    }

    #[test]
    fn test_bump_scope() {
        let allocator = BumpAllocator::new(4096).unwrap();

        let initial_allocated = allocator.allocated_bytes();

        {
            let scope = BumpScope {
                allocator: &allocator,
                initial_offset: allocator.current.load(Ordering::Relaxed),
                initial_allocated_bytes: allocator.allocated_bytes(),
            };

            let _ptr1 = scope.alloc::<u64>().unwrap();
            let _ptr2 = scope.alloc::<u64>().unwrap();

            assert!(allocator.allocated_bytes() > initial_allocated);
        }

        // After scope ends, allocation should be reset
        assert_eq!(allocator.current.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_bump_vec() {
        let allocator = BumpAllocator::new(4096).unwrap();
        let mut vec = BumpVec::new_in(&allocator, 10).unwrap();

        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert_eq!(vec.capacity(), 10);

        vec.push(42).unwrap();
        vec.push(84).unwrap();

        assert_eq!(vec.len(), 2);
        assert!(!vec.is_empty());

        let slice = vec.as_slice();
        assert_eq!(slice[0], 42);
        assert_eq!(slice[1], 84);

        let popped = vec.pop().unwrap();
        assert_eq!(popped, 84);
        assert_eq!(vec.len(), 1);
    }

    #[test]
    fn test_can_allocate() {
        let allocator = BumpAllocator::new(64).unwrap();

        assert!(allocator.can_allocate(8, 8));
        assert!(allocator.can_allocate(64, 1));
        assert!(!allocator.can_allocate(65, 1));

        // Allocate some memory
        let _ptr = allocator.alloc::<u64>().unwrap();

        assert!(allocator.can_allocate(8, 8));
        assert!(!allocator.can_allocate(64, 1));
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(BumpAllocator::new(0).is_err());

        let allocator = BumpAllocator::new(1024).unwrap();
        assert!(allocator.alloc_bytes(0, 8).is_err());
        assert!(allocator.alloc_bytes(8, 3).is_err()); // Non-power-of-2 alignment
    }
}
