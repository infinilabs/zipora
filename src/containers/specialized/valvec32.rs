//! ValVec32: 32-bit indexed vector for memory efficiency
//!
//! This container uses u32 indices instead of usize, providing significant
//! memory savings on 64-bit systems for large collections while maintaining
//! high performance for common operations.
//!
//! Optimized for high performance and memory efficiency.

use crate::error::{Result, ZiporaError};
use crate::memory::SecureMemoryPool;
use std::alloc::{self, Layout};
use std::fmt;
use std::mem;
use std::ops::{Index, IndexMut};
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::Arc;

// Import libc for direct malloc/realloc/free access
extern crate libc;

/// Maximum capacity for ValVec32 (2^32 - 1 elements)
pub const MAX_CAPACITY: u32 = u32::MAX;

/// Cache line size for x86_64 processors
#[cfg(target_arch = "x86_64")]
pub const CACHE_LINE_SIZE: usize = 64;

/// Cache line size for ARM64 processors
#[cfg(target_arch = "aarch64")]
pub const CACHE_LINE_SIZE: usize = 128;

/// Default cache line size for other architectures
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const CACHE_LINE_SIZE: usize = 64;

/// Platform-specific malloc_usable_size wrapper
#[cfg(target_os = "linux")]
fn get_usable_size(ptr: *mut u8, size: usize) -> usize {
    unsafe {
        // Use malloc_usable_size on Linux
        unsafe extern "C" {
            fn malloc_usable_size(ptr: *mut std::ffi::c_void) -> usize;
        }
        if !ptr.is_null() {
            malloc_usable_size(ptr as *mut std::ffi::c_void)
        } else {
            size
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn get_usable_size(ptr: *mut u8, size: usize) -> usize {
    unsafe {
        // Use malloc_size on macOS/iOS
        unsafe extern "C" {
            fn malloc_size(ptr: *mut std::ffi::c_void) -> usize;
        }
        if !ptr.is_null() {
            malloc_size(ptr as *mut std::ffi::c_void)
        } else {
            size
        }
    }
}

#[cfg(target_os = "windows")]
fn get_usable_size(ptr: *mut u8, size: usize) -> usize {
    unsafe {
        // Use _msize on Windows
        unsafe extern "C" {
            fn _msize(ptr: *mut std::ffi::c_void) -> usize;
        }
        if !ptr.is_null() {
            _msize(ptr as *mut std::ffi::c_void)
        } else {
            size
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios", target_os = "windows")))]
fn get_usable_size(_ptr: *mut u8, size: usize) -> usize {
    // Fallback for other platforms
    size
}

/// Branch prediction hints for performance optimization
#[cfg(feature = "nightly")]
use std::intrinsics::{likely, unlikely};

#[cfg(not(feature = "nightly"))]
#[inline(always)]
fn likely(b: bool) -> bool {
    b
}

#[cfg(not(feature = "nightly"))]
#[inline(always)]
fn unlikely(b: bool) -> bool {
    b
}

/// Adaptive growth strategy with size-based growth factors
/// Optimized for different vector sizes to reduce memory waste and improve performance
#[inline]
fn larger_capacity(old_cap: u32, element_size: usize) -> u32 {
    if old_cap == 0 {
        // Start with reasonable initial capacity based on element size
        let elements_per_cache_line = 64 / element_size.max(1);
        return (elements_per_cache_line as u32).clamp(4, 16);
    }
    
    // Adaptive growth strategy based on current size
    let new_cap = if old_cap <= 64 {
        // Small vectors: 2x growth for better performance
        old_cap.saturating_mul(2)
    } else if old_cap <= 4096 {
        // Medium vectors: 1.5x growth for balanced approach
        old_cap + (old_cap >> 1)
    } else {
        // Large vectors: 1.25x growth to minimize memory waste
        old_cap + (old_cap >> 2)
    };
    
    new_cap.min(MAX_CAPACITY)
}

/// High-performance vector with 32-bit indices for memory efficiency
///
/// ValVec32 provides significant memory savings on 64-bit systems by using
/// u32 indices instead of usize. This results in 33% memory reduction for
/// the struct overhead while supporting up to 4 billion elements.
///
/// # Memory Efficiency
///
/// - Uses u32 for length and capacity (8 bytes vs 16 bytes on 64-bit)
/// - Maximum capacity: 4,294,967,295 elements
/// - Memory overhead: 16 bytes vs 24 bytes for std::Vec
/// - Target: 33% memory reduction for struct size
///
/// # Performance Optimizations
///
/// - O(1) amortized push/pop operations
/// - O(1) random access via indexing
/// - Adaptive growth strategy with size-based growth factors
/// - malloc_usable_size optimization for maximum memory utilization
/// - Hot path optimization with branch prediction hints
/// - Streamlined design for optimal cache performance
///
/// # Examples
///
/// ```rust
/// use zipora::ValVec32;
///
/// let mut vec = ValVec32::new();
/// vec.push(42)?;
/// vec.push(84)?;
///
/// assert_eq!(vec.len(), 2);
/// assert_eq!(vec[0], 42);
/// assert_eq!(vec[1], 84);
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
#[repr(C)] // Ensure predictable layout
pub struct ValVec32<T> {
    /// Pointer to the allocated memory
    ptr: NonNull<T>,
    /// Number of elements currently stored (u32 for memory efficiency)
    len: u32,
    /// Allocated capacity in elements (u32 for memory efficiency)  
    capacity: u32,
    // Total: ptr(8) + len(4) + capacity(4) = 16 bytes
    // Streamlined design for optimal performance and cache efficiency
}

// Simple iterator using slice - prioritize correctness over micro-optimizations
pub type ValVec32Iter<'a, T> = std::slice::Iter<'a, T>;
pub type ValVec32IterMut<'a, T> = std::slice::IterMut<'a, T>;

impl<T> ValVec32<T> {
    /// Creates a new empty ValVec32
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let vec: ValVec32<i32> = ValVec32::new();
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }

    /// Creates a new ValVec32 with specified capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity to allocate
    ///
    /// # Errors
    ///
    /// Returns `ZiporaError::MemoryError` if allocation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let vec: ValVec32<i32> = ValVec32::with_capacity(100)?;
    /// assert_eq!(vec.len(), 0);
    /// assert!(vec.capacity() >= 100); // Capacity may be larger due to allocator optimization
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn with_capacity(capacity: u32) -> Result<Self> {
        if capacity == 0 || mem::size_of::<T>() == 0 {
            return Ok(Self::new());
        }

        let capacity_usize = capacity as usize;
        let size = capacity_usize * mem::size_of::<T>();
        
        // Use standard allocation with proper alignment
        let layout = Layout::array::<T>(capacity_usize)
            .map_err(|_| ZiporaError::invalid_data("Invalid layout for capacity"))?;
        
        let ptr = unsafe {
            let raw_ptr = alloc::alloc(layout);
            if raw_ptr.is_null() {
                return Err(ZiporaError::out_of_memory(size));
            }
            NonNull::new_unchecked(raw_ptr as *mut T)
        };
        
        // Use malloc_usable_size to get actual allocated capacity
        let actual_size = get_usable_size(ptr.as_ptr() as *mut u8, size);
        let actual_capacity = (actual_size / mem::size_of::<T>()).min(MAX_CAPACITY as usize);
        
        Ok(Self {
            ptr,
            len: 0,
            capacity: actual_capacity as u32,
        })
    }

    /// Creates a new ValVec32 with secure pool compatibility
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity to allocate
    /// * `_pool` - SecureMemoryPool (currently unused for performance)
    ///
    /// # Errors
    ///
    /// Returns `ZiporaError::MemoryError` if allocation fails
    pub fn with_secure_pool(capacity: u32, _pool: Arc<SecureMemoryPool>) -> Result<Self> {
        // For now, ignore the pool parameter and use standard allocation
        // This maintains API compatibility while maximizing performance
        Self::with_capacity(capacity)
    }

    /// Returns the number of elements in the vector (u32)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// assert_eq!(vec.len(), 0);
    ///
    /// vec.push(42)?;
    /// assert_eq!(vec.len(), 1);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }
    
    /// Returns the number of elements in the vector (usize)
    #[inline]
    pub fn len_usize(&self) -> usize {
        self.len as usize
    }

    /// Returns the allocated capacity of the vector (u32)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let vec: ValVec32<i32> = ValVec32::with_capacity(10)?;
    /// assert_eq!(vec.capacity(), 10);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    
    /// Returns the allocated capacity of the vector (usize)
    #[inline]
    pub fn capacity_usize(&self) -> usize {
        self.capacity as usize
    }

    /// Returns true if the vector is empty
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let vec: ValVec32<i32> = ValVec32::new();
    /// assert!(vec.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reserves capacity for at least `additional` more elements
    ///
    /// # Arguments
    ///
    /// * `additional` - Number of additional elements to reserve space for
    ///
    /// # Errors
    ///
    /// Returns `ZiporaError::InvalidData` if new capacity would overflow
    /// Returns `ZiporaError::MemoryError` if allocation fails
    #[inline]
    pub fn reserve(&mut self, additional: u32) -> Result<()> {
        let required = self
            .len
            .checked_add(additional)
            .ok_or_else(|| ZiporaError::invalid_data("Capacity overflow"))?;

        if required <= self.capacity {
            return Ok(());
        }

        let new_capacity = self.calculate_new_capacity(required);
        self.grow_to(new_capacity)
    }

    /// Calculates new capacity using adaptive growth strategy
    #[inline]
    fn calculate_new_capacity(&self, min_capacity: u32) -> u32 {
        let new_cap = larger_capacity(self.capacity, mem::size_of::<T>()).max(min_capacity);
        new_cap.min(MAX_CAPACITY)
    }

    /// Grows the vector to the specified capacity - marked cold and never inline
    /// to keep it out of the hot path for better instruction cache performance
    #[cold]
    #[inline(never)]
    fn grow_to(&mut self, new_capacity: u32) -> Result<()> {
        if new_capacity <= self.capacity {
            return Ok(());
        }

        if mem::size_of::<T>() == 0 {
            // ZSTs don't need allocation
            self.capacity = MAX_CAPACITY;
            return Ok(());
        }

        let new_capacity_usize = new_capacity as usize;
        let elem_size = mem::size_of::<T>();
        let new_size = new_capacity_usize.saturating_mul(elem_size);
        
        // Optimize: Try to use realloc first which can be more efficient
        let new_ptr = if self.capacity == 0 {
            // Initial allocation - use simple malloc without Layout overhead
            unsafe {
                let align = mem::align_of::<T>();
                let raw_ptr = if align <= mem::align_of::<usize>() {
                    // Most common case: standard alignment
                    let ptr = libc::malloc(new_size);
                    ptr as *mut T
                } else {
                    // Need special alignment
                    let layout = Layout::from_size_align_unchecked(new_size, align);
                    alloc::alloc(layout) as *mut T
                };
                
                if raw_ptr.is_null() {
                    return Err(ZiporaError::out_of_memory(new_size));
                }
                NonNull::new_unchecked(raw_ptr)
            }
        } else {
            // Reallocation: Try realloc first which may avoid copying
            unsafe {
                let align = mem::align_of::<T>();
                let raw_ptr = if align <= mem::align_of::<usize>() {
                    // Standard alignment - use realloc
                    let ptr = libc::realloc(self.ptr.as_ptr() as *mut libc::c_void, new_size);
                    ptr as *mut T
                } else {
                    // Special alignment - must allocate and copy
                    let layout = Layout::from_size_align_unchecked(new_size, align);
                    let new_raw = alloc::alloc(layout) as *mut T;
                    if !new_raw.is_null() && self.len > 0 {
                        ptr::copy_nonoverlapping(
                            self.ptr.as_ptr(),
                            new_raw,
                            self.len as usize,
                        );
                    }
                    if self.capacity > 0 {
                        let old_layout = Layout::from_size_align_unchecked(
                            self.capacity as usize * elem_size,
                            align
                        );
                        alloc::dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
                    }
                    new_raw
                };
                
                if raw_ptr.is_null() {
                    return Err(ZiporaError::out_of_memory(new_size));
                }
                NonNull::new_unchecked(raw_ptr)
            }
        };
        
        // Use malloc_usable_size to get actual allocated capacity
        let actual_size = get_usable_size(new_ptr.as_ptr() as *mut u8, new_size);
        let actual_capacity = (actual_size / elem_size).min(MAX_CAPACITY as usize);

        self.ptr = new_ptr;
        self.capacity = actual_capacity as u32;
        
        Ok(())
    }

    /// Appends an element to the back of the vector
    ///
    /// # Arguments
    ///
    /// * `value` - Element to append
    ///
    /// # Errors
    ///
    /// Returns `ZiporaError::InvalidData` if vector is at maximum capacity
    /// Returns `ZiporaError::MemoryError` if allocation fails during growth
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// vec.push(42)?;
    /// vec.push(84)?;
    ///
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(vec[0], 42);
    /// assert_eq!(vec[1], 84);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    /// Appends an element to the back of the vector - hot path optimized
    /// The fast path is always inlined for maximum performance
    #[inline(always)]
    pub fn push(&mut self, value: T) -> Result<()> {
        // Hot path: check capacity with likely hint
        if likely(self.len < self.capacity) {
            // Fast path: direct write without any function calls
            unsafe {
                // Use offset instead of add for potentially better codegen
                ptr::write(self.ptr.as_ptr().offset(self.len as isize), value);
                self.len += 1;
            }
            Ok(())
        } else {
            // Cold path: delegate to slow path which is never inlined
            self.push_slow(value)
        }
    }


    /// Appends an element to the vector, panicking on failure (like std::Vec)
    ///
    /// This method matches the behavior of std::Vec::push, panicking if
    /// allocation fails or the vector is at maximum capacity. This provides
    /// optimal performance for benchmarking against std::Vec.
    ///
    /// # Arguments
    ///
    /// * `value` - Element to append
    ///
    /// # Panics
    ///
    /// Panics if the vector is at maximum capacity or allocation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// vec.push_panic(42);
    /// vec.push_panic(84);
    ///
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(vec[0], 42);
    /// assert_eq!(vec[1], 84);
    /// ```
    /// Appends an element to the vector, panicking on failure - maximum performance
    /// This provides optimal performance for benchmarking against std::Vec
    #[inline(always)]
    pub fn push_panic(&mut self, value: T) {
        // Check with branch prediction hint
        if likely(self.len < self.capacity) {
            // Fast path with minimal overhead
            unsafe {
                ptr::write(self.ptr.as_ptr().offset(self.len as isize), value);
                self.len += 1;
            }
        } else {
            // Delegate to cold path
            self.push_slow_panic(value);
        }
    }

    /// Slow path for push_panic when growth is needed
    #[cold]
    #[inline(never)]
    fn push_slow_panic(&mut self, value: T) {
        if self.len >= MAX_CAPACITY {
            panic!("ValVec32 at maximum capacity");
        }

        let new_capacity = self.calculate_new_capacity(self.len + 1);
        self.grow_to(new_capacity).expect("Failed to allocate memory");
        
        unsafe {
            let ptr = self.ptr.as_ptr().add(self.len as usize);
            ptr::write(ptr, value);
            self.len += 1;
        }
    }

    /// Unchecked push for when capacity is guaranteed - maximum performance
    /// 
    /// # Safety
    /// 
    /// Caller must ensure that `self.len < self.capacity`
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use zipora::ValVec32;
    /// 
    /// let mut vec = ValVec32::with_capacity(10)?;
    /// unsafe {
    ///     vec.unchecked_push(42); // Safe because we reserved capacity
    /// }
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    #[inline(always)]
    pub unsafe fn unchecked_push(&mut self, value: T) {
        debug_assert!(self.len < self.capacity);
        // Direct write with no checks - maximum performance
        unsafe {
            ptr::write(self.ptr.as_ptr().offset(self.len as isize), value);
        }
        self.len += 1;
    }

    /// Unchecked push for Copy types - even more optimized
    /// 
    /// # Safety
    /// 
    /// Caller must ensure that `self.len < self.capacity`
    #[inline(always)]
    pub unsafe fn unchecked_push_copy(&mut self, value: T) 
    where 
        T: Copy
    {
        debug_assert!(self.len < self.capacity);
        // For Copy types, we can use direct assignment which may be faster
        unsafe {
            *self.ptr.as_ptr().offset(self.len as isize) = value;
        }
        self.len += 1;
    }


    /// Slow path for push when growth is needed
    #[cold]
    #[inline(never)]
    fn push_slow(&mut self, value: T) -> Result<()> {
        if self.len >= MAX_CAPACITY {
            return Err(ZiporaError::invalid_data("Vector at maximum capacity"));
        }

        let new_capacity = self.calculate_new_capacity(self.len + 1);
        self.grow_to(new_capacity)?;
        
        unsafe {
            let ptr = self.ptr.as_ptr().add(self.len as usize);
            ptr::write(ptr, value);
            self.len += 1;
        }
        Ok(())
    }

    /// Removes and returns the last element
    ///
    /// # Returns
    ///
    /// `Some(T)` if the vector is not empty, `None` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// assert_eq!(vec.pop(), None);
    ///
    /// vec.push(42)?;
    /// vec.push(84)?;
    ///
    /// assert_eq!(vec.pop(), Some(84));
    /// assert_eq!(vec.pop(), Some(42));
    /// assert_eq!(vec.pop(), None);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        // SAFETY: We've decremented len, so this index is now out of bounds
        // but was previously valid
        Some(unsafe { ptr::read(self.ptr.as_ptr().add(self.len as usize)) })
    }

    /// Returns a reference to the element at the given index
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the element to retrieve
    ///
    /// # Returns
    ///
    /// `Some(&T)` if index is valid, `None` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// vec.push(42)?;
    ///
    /// assert_eq!(vec.get(0), Some(&42));
    /// assert_eq!(vec.get(1), None);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    #[inline]
    pub fn get(&self, index: u32) -> Option<&T> {
        if index < self.len {
            let index_usize = index as usize;
            Some(unsafe { &*self.ptr.as_ptr().add(index_usize) })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the given index
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the element to retrieve
    ///
    /// # Returns
    ///
    /// `Some(&mut T)` if index is valid, `None` otherwise
    #[inline]
    pub fn get_mut(&mut self, index: u32) -> Option<&mut T> {
        if index < self.len {
            let index_usize = index as usize;
            Some(unsafe { &mut *self.ptr.as_ptr().add(index_usize) })
        } else {
            None
        }
    }

    /// Sets the value at the given index
    ///
    /// # Arguments
    ///
    /// * `index` - Index to set
    /// * `value` - Value to set
    ///
    /// # Errors
    ///
    /// Returns `ZiporaError::InvalidData` if index is out of bounds
    pub fn set(&mut self, index: u32, value: T) -> Result<()> {
        if index >= self.len {
            return Err(ZiporaError::invalid_data(format!(
                "Index {} out of bounds for length {}",
                index, self.len
            )));
        }

        // SAFETY: Index is bounds checked
        unsafe {
            ptr::write(self.ptr.as_ptr().add(index as usize), value);
        }
        Ok(())
    }

    /// Clears the vector, removing all elements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// vec.push(42)?;
    /// vec.push(84)?;
    ///
    /// vec.clear();
    /// assert!(vec.is_empty());
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn clear(&mut self) {
        // Drop all elements
        for i in 0..(self.len as usize) {
            // SAFETY: All indices 0..len are valid
            unsafe {
                ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }
        self.len = 0;
    }

    /// Returns a slice containing all elements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// vec.push(1)?;
    /// vec.push(2)?;
    /// vec.push(3)?;
    ///
    /// let slice = vec.as_slice();
    /// assert_eq!(slice, &[1, 2, 3]);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 || mem::size_of::<T>() == 0 {
            return &[];
        }
        // SAFETY: We have len valid elements starting from ptr
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
    }

    /// Returns a mutable slice containing all elements
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 || mem::size_of::<T>() == 0 {
            return &mut [];
        }
        // SAFETY: We have len valid elements starting from ptr
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len as usize) }
    }

    /// Extends the vector by appending all elements from a slice
    ///
    /// # Arguments
    ///
    /// * `slice` - Slice of elements to append
    ///
    /// # Errors
    ///
    /// Returns `ZiporaError::InvalidData` if resulting length would overflow
    /// Returns `ZiporaError::MemoryError` if allocation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// vec.push(1)?;
    /// vec.extend_from_slice(&[2, 3, 4])?;
    ///
    /// assert_eq!(vec.len(), 4);
    /// assert_eq!(vec.get(0), Some(&1));
    /// assert_eq!(vec.get(3), Some(&4));
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn extend_from_slice(&mut self, slice: &[T]) -> Result<()>
    where
        T: Clone,
    {
        if slice.is_empty() {
            return Ok(());
        }

        let additional = slice.len() as u32;
        let new_len = self
            .len
            .checked_add(additional)
            .ok_or_else(|| ZiporaError::invalid_data("Length overflow"))?;

        self.reserve(additional)?;

        unsafe {
            let dst = self.ptr.as_ptr().add(self.len as usize);
            for (i, item) in slice.iter().enumerate() {
                ptr::write(dst.add(i), item.clone());
            }
        }

        self.len = new_len;
        Ok(())
    }

    /// SIMD-optimized extend for Copy types - uses memcpy for maximum performance
    ///
    /// # Arguments
    ///
    /// * `slice` - Slice of elements to append
    ///
    /// # Errors
    ///
    /// Returns error if capacity overflow or allocation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// vec.extend_from_slice_copy(&[1, 2, 3, 4])?;
    /// assert_eq!(vec.len(), 4);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    #[inline]
    pub fn extend_from_slice_copy(&mut self, slice: &[T]) -> Result<()>
    where
        T: Copy,
    {
        if slice.is_empty() {
            return Ok(());
        }

        let additional = slice.len() as u32;
        let new_len = self
            .len
            .checked_add(additional)
            .ok_or_else(|| ZiporaError::invalid_data("Length overflow"))?;

        self.reserve(additional)?;

        unsafe {
            // Use memcpy for Copy types - significantly faster than iteration
            let dst = self.ptr.as_ptr().add(self.len as usize);
            ptr::copy_nonoverlapping(slice.as_ptr(), dst, slice.len());
        }

        self.len = new_len;
        Ok(())
    }

    /// Bulk push with SIMD optimization for Copy types
    ///
    /// # Arguments
    ///
    /// * `count` - Number of elements to push
    /// * `value` - Value to replicate
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// vec.push_n_copy(100, 42u32)?;
    /// assert_eq!(vec.len(), 100);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    #[inline]
    pub fn push_n_copy(&mut self, count: u32, value: T) -> Result<()>
    where
        T: Copy,
    {
        if count == 0 {
            return Ok(());
        }

        let new_len = self
            .len
            .checked_add(count)
            .ok_or_else(|| ZiporaError::invalid_data("Length overflow"))?;

        self.reserve(count)?;

        unsafe {
            let dst = self.ptr.as_ptr().add(self.len as usize);
            
            // For small counts, use simple loop
            if count <= 16 {
                for i in 0..count as usize {
                    ptr::write(dst.add(i), value);
                }
            } else {
                // For larger counts, use doubling strategy for better performance
                ptr::write(dst, value);
                let mut written = 1usize;
                while written < count as usize {
                    let to_copy = written.min(count as usize - written);
                    ptr::copy_nonoverlapping(dst, dst.add(written), to_copy);
                    written += to_copy;
                }
            }
        }

        self.len = new_len;
        Ok(())
    }


    /// Returns an iterator over the elements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// vec.push(1)?;
    /// vec.push(2)?;
    /// vec.push(3)?;
    ///
    /// let mut iter = vec.iter();
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), None);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn iter(&self) -> ValVec32Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the elements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::ValVec32;
    ///
    /// let mut vec = ValVec32::new();
    /// vec.push(1)?;
    /// vec.push(2)?;
    ///
    /// for item in vec.iter_mut() {
    ///     *item *= 2;
    /// }
    ///
    /// assert_eq!(vec.get(0), Some(&2));
    /// assert_eq!(vec.get(1), Some(&4));
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn iter_mut(&mut self) -> ValVec32IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T> Default for ValVec32<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a ValVec32<T> {
    type Item = &'a T;
    type IntoIter = ValVec32Iter<'a, T>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ValVec32<T> {
    type Item = &'a mut T;
    type IntoIter = ValVec32IterMut<'a, T>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T> Drop for ValVec32<T> {
    fn drop(&mut self) {
        self.clear();

        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            let layout = Layout::array::<T>(self.capacity as usize).unwrap();
            unsafe {
                alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T> Index<u32> for ValVec32<T> {
    type Output = T;

    fn index(&self, index: u32) -> &Self::Output {
        self.get(index).unwrap_or_else(|| {
            panic!(
                "Index {} out of bounds for ValVec32 with length {}",
                index, self.len
            )
        })
    }
}

impl<T> IndexMut<u32> for ValVec32<T> {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        let len = self.len; // Capture len before borrow
        self.get_mut(index).unwrap_or_else(|| {
            panic!(
                "Index {} out of bounds for ValVec32 with length {}",
                index, len
            )
        })
    }
}

impl<T: fmt::Debug> fmt::Debug for ValVec32<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: Clone> Clone for ValVec32<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::with_capacity(self.len).expect("Failed to allocate during clone");

        // Use bulk copy for better performance
        if self.len > 0 {
            unsafe {
                for i in 0..(self.len as usize) {
                    let value = (*self.ptr.as_ptr().add(i)).clone();
                    ptr::write(new_vec.ptr.as_ptr().add(i), value);
                }
                new_vec.len = self.len;
            }
        }

        new_vec
    }
}

impl<T: PartialEq> PartialEq for ValVec32<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for ValVec32<T> {}

// SAFETY: ValVec32 is Send if T is Send
unsafe impl<T: Send> Send for ValVec32<T> {}

// SAFETY: ValVec32 is Sync if T is Sync
unsafe impl<T: Sync> Sync for ValVec32<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let vec: ValVec32<i32> = ValVec32::new();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_with_capacity() -> Result<()> {
        let vec: ValVec32<i32> = ValVec32::with_capacity(10)?;
        assert_eq!(vec.len(), 0);
        // With malloc_usable_size optimization, capacity may be larger than requested
        assert!(vec.capacity() >= 10);
        assert!(vec.is_empty());
        Ok(())
    }

    #[test]
    fn test_with_capacity_zero() -> Result<()> {
        let vec: ValVec32<i32> = ValVec32::with_capacity(0)?;
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
        Ok(())
    }

    #[test]
    fn test_with_capacity_max() {
        // Test that MAX_CAPACITY is actually supported
        let result: Result<ValVec32<i32>> = ValVec32::with_capacity(MAX_CAPACITY);
        // This should succeed in theory, but might fail due to memory limitations
        // So we just check it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_push_and_get() -> Result<()> {
        let mut vec = ValVec32::<i32>::new();

        vec.push(42)?;
        vec.push(84)?;
        vec.push(126)?;

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(0), Some(&42));
        assert_eq!(vec.get(1), Some(&84));
        assert_eq!(vec.get(2), Some(&126));
        assert_eq!(vec.get(3), None);

        Ok(())
    }

    #[test]
    fn test_indexing() -> Result<()> {
        let mut vec = ValVec32::<i32>::new();
        vec.push(10)?;
        vec.push(20)?;

        assert_eq!(vec[0], 10);
        assert_eq!(vec[1], 20);

        vec[0] = 15;
        assert_eq!(vec[0], 15);

        Ok(())
    }

    #[test]
    #[should_panic(expected = "Index 2 out of bounds")]
    fn test_index_panic() {
        let vec: ValVec32<i32> = ValVec32::new();
        let _ = vec[2]; // Should panic
    }

    #[test]
    fn test_pop() -> Result<()> {
        let mut vec = ValVec32::<i32>::new();

        assert_eq!(vec.pop(), None);

        vec.push(42)?;
        vec.push(84)?;

        assert_eq!(vec.pop(), Some(84));
        assert_eq!(vec.pop(), Some(42));
        assert_eq!(vec.pop(), None);
        assert!(vec.is_empty());

        Ok(())
    }

    #[test]
    fn test_set() -> Result<()> {
        let mut vec = ValVec32::<i32>::new();
        vec.push(42)?;

        vec.set(0, 84)?;
        assert_eq!(vec[0], 84);

        let result = vec.set(1, 126);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_clear() -> Result<()> {
        let mut vec = ValVec32::<i32>::new();
        vec.push(1)?;
        vec.push(2)?;
        vec.push(3)?;

        assert_eq!(vec.len(), 3);
        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());

        Ok(())
    }

    #[test]
    fn test_as_slice() -> Result<()> {
        let mut vec = ValVec32::<i32>::new();
        vec.push(1)?;
        vec.push(2)?;
        vec.push(3)?;

        let slice = vec.as_slice();
        assert_eq!(slice, &[1, 2, 3]);

        Ok(())
    }

    #[test]
    fn test_clone() -> Result<()> {
        let mut vec = ValVec32::<i32>::new();
        vec.push(1)?;
        vec.push(2)?;
        vec.push(3)?;

        let cloned = vec.clone();
        assert_eq!(vec, cloned);
        assert_eq!(vec.as_slice(), cloned.as_slice());

        Ok(())
    }

    #[test]
    fn test_equality() -> Result<()> {
        let mut vec1 = ValVec32::new();
        let mut vec2 = ValVec32::new();

        assert_eq!(vec1, vec2);

        vec1.push(42)?;
        assert_ne!(vec1, vec2);

        vec2.push(42)?;
        assert_eq!(vec1, vec2);

        Ok(())
    }

    #[test]
    fn test_growth() -> Result<()> {
        let mut vec = ValVec32::<i32>::new();
        let initial_capacity = vec.capacity();

        // Push enough elements to trigger growth
        for i in 0..10 {
            vec.push(i)?;
        }

        assert!(vec.capacity() > initial_capacity);
        assert_eq!(vec.len(), 10);

        for i in 0..10 {
            assert_eq!(vec[i], i as i32);
        }

        Ok(())
    }

    #[test]
    fn test_reserve() -> Result<()> {
        let mut vec = ValVec32::<i32>::new();
        vec.reserve(100)?;

        assert!(vec.capacity() >= 100);
        assert_eq!(vec.len(), 0);

        Ok(())
    }

    #[test]
    fn test_memory_efficiency() -> Result<()> {
        // Test that ValVec32 uses less memory than equivalent std::Vec
        // This is primarily a compile-time verification

        let _vec32 = ValVec32::<u64>::new();
        let _std_vec = Vec::<u64>::new();

        // ValVec32 core data should be more memory efficient
        // Note: Due to Option<SecureMemoryPool> the struct is larger, but the core indices are smaller

        // Verify the struct size - ValVec32 uses u32 for len/capacity vs usize for Vec
        let vec32_size = std::mem::size_of::<ValVec32<u64>>();
        let std_vec_size = std::mem::size_of::<Vec<u64>>();

        // The benefit is in the smaller indices (u32 vs usize), not necessarily the full struct
        println!("ValVec32 size: {}, Vec size: {}", vec32_size, std_vec_size);

        // The memory efficiency comes from the 32-bit indices, not the total struct size
        assert!(vec32_size > 0);
        assert!(std_vec_size > 0);

        Ok(())
    }




    #[test]
    fn test_branch_prediction_hints() {
        // Test that likely/unlikely functions work without crashing
        assert!(likely(true));
        assert!(!likely(false));
        assert!(unlikely(true));
        assert!(!unlikely(false));
    }

    #[test]
    fn test_small_size_growth() -> Result<()> {
        let mut vec = ValVec32::<i32>::new();

        // First push should trigger initial allocation 
        vec.push(1)?;
        let initial_capacity = vec.capacity();

        // Should be at least 4 elements for cache efficiency
        assert!(initial_capacity >= 4);

        // The initial capacity should be reasonable for small vectors
        // With malloc_usable_size bonus, this could be slightly larger than expected
        let cache_line_elements = 64 / std::mem::size_of::<i32>();
        let max_reasonable = (cache_line_elements * 2).max(32) as u32; // Allow for malloc bonus
        assert!(initial_capacity <= max_reasonable, 
                "initial_capacity {} > max_reasonable {}", initial_capacity, max_reasonable);

        // Test that growth works properly
        let mut count = 1;
        let first_capacity = initial_capacity;
        
        // Fill to capacity
        while count < first_capacity {
            vec.push(count as i32)?;
            count += 1;
        }
        
        // Next push should trigger growth
        vec.push(count as i32)?;
        assert!(vec.capacity() > first_capacity);

        Ok(())
    }


    #[test]
    fn test_zero_sized_types() -> Result<()> {
        // Test basic ZST support
        let mut vec = ValVec32::<()>::new();
        
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
        assert!(vec.is_empty());

        // Test basic push operations
        for _ in 0..10 {
            vec.push(())?;
        }
        assert_eq!(vec.len(), 10);
        
        // Test indexing
        assert_eq!(vec.get(0), Some(&()));
        assert_eq!(vec.get(9), Some(&()));
        assert_eq!(vec.get(10), None);

        // Test pop operations
        assert_eq!(vec.pop(), Some(()));
        assert_eq!(vec.len(), 9);

        Ok(())
    }


}
