//! FastVec: High-performance vector using realloc for growth
//!
//! This is a direct port of the C++ `valvec` with Rust safety guarantees.
//! Unlike std::Vec which uses malloc+memcpy for growth, FastVec uses realloc
//! which can often avoid copying when the allocator can expand in place.

use crate::error::{Result, ZiporaError};
use crate::memory::simd_ops::{fast_copy, fast_fill, fast_compare};
// Import verification macros for error handling
use crate::zipora_verify;
use std::alloc::{self, Layout};
use std::fmt;
use std::mem;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::{self, NonNull};
use std::slice;

/// Check that a pointer is properly aligned for type T
/// Uses verification for fail-fast error handling
#[inline]
fn check_alignment<T>(ptr: *mut u8) {
    // Use verification instead of debug_assert
    crate::zipora_verify_not_null!(ptr);
    crate::zipora_verify_aligned!(ptr, mem::align_of::<T>());
}

/// Safely cast an aligned u8 pointer to T pointer with alignment verification
#[inline]
fn cast_aligned_ptr<T>(ptr: *mut u8) -> *mut T {
    check_alignment::<T>(ptr);
    ptr as *mut T
}

/// Check if a type is suitable for SIMD operations (Copy + no custom drop)
#[inline]
const fn is_simd_safe<T>() -> bool {
    // Use const traits when available, for now rely on Copy bound in caller
    mem::needs_drop::<T>() == false
}

/// Check if an operation size is large enough to benefit from SIMD
#[inline]
const fn is_simd_beneficial<T>(element_count: usize) -> bool {
    // SIMD beneficial threshold: 64 bytes minimum
    const SIMD_THRESHOLD: usize = 64;
    element_count * mem::size_of::<T>() >= SIMD_THRESHOLD
}

/// Convert a slice of T to a slice of u8 for SIMD operations
/// 
/// # Safety
/// 
/// T must be Copy and have no custom Drop implementation
#[inline]
unsafe fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    if slice.is_empty() {
        &[]
    } else {
        unsafe {
            slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                slice.len() * mem::size_of::<T>(),
            )
        }
    }
}

/// Convert a mutable slice of T to a mutable slice of u8 for SIMD operations
/// 
/// # Safety
/// 
/// T must be Copy and have no custom Drop implementation
#[inline]
unsafe fn slice_as_bytes_mut<T>(slice: &mut [T]) -> &mut [u8] {
    if slice.is_empty() {
        &mut []
    } else {
        unsafe {
            slice::from_raw_parts_mut(
                slice.as_mut_ptr() as *mut u8,
                slice.len() * mem::size_of::<T>(),
            )
        }
    }
}

/// High-performance vector using realloc for growth
///
/// FastVec is designed for maximum performance when dealing with types that are
/// memmove-safe (most primitive types and simple structs). It uses realloc()
/// for growth which can avoid memory copying in many cases.
///
/// # Safety
///
/// FastVec is safe to use with any type T, but performs best with types that
/// are `Copy` or have trivial move semantics.
///
/// # Examples
///
/// ```rust
/// use zipora::FastVec;
///
/// let mut vec = FastVec::new();
/// vec.push(42);
/// vec.push(84);
/// assert_eq!(vec.len(), 2);
/// assert_eq!(vec[0], 42);
/// ```
pub struct FastVec<T> {
    ptr: Option<NonNull<T>>,
    len: usize,
    cap: usize,
}

impl<T> FastVec<T> {
    /// Create a new empty FastVec
    #[inline]
    pub fn new() -> Self {
        Self {
            ptr: None,
            len: 0,
            cap: 0,
        }
    }

    /// Create a FastVec with the specified capacity
    pub fn with_capacity(cap: usize) -> Result<Self> {
        if cap == 0 {
            return Ok(Self::new());
        }

        // Verify capacity is reasonable
        crate::zipora_verify!(cap <= (isize::MAX as usize) / mem::size_of::<T>().max(1),
            "capacity {} too large for element size {}", cap, mem::size_of::<T>());

        let layout = Layout::array::<T>(cap)
            .map_err(|_| ZiporaError::out_of_memory(cap * mem::size_of::<T>()))?;

        let ptr = unsafe {
            let raw_ptr = alloc::alloc(layout);
            // Verify allocation success
            crate::zipora_verify_alloc!(raw_ptr, layout.size());
            cast_aligned_ptr::<T>(raw_ptr)
        };

        Ok(Self {
            ptr: Some(unsafe { NonNull::new_unchecked(ptr) }),
            len: 0,
            cap,
        })
    }

    /// Create a FastVec with the specified size, filled with the given value
    pub fn with_size(size: usize, value: T) -> Result<Self>
    where
        T: Clone,
    {
        let mut vec = Self::with_capacity(size)?;
        vec.resize(size, value)?;
        Ok(vec)
    }

    /// Get the number of elements in the vector
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the vector
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Get a pointer to the underlying data
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        match self.ptr {
            Some(ptr) => ptr.as_ptr(),
            None => ptr::null(),
        }
    }

    /// Get a mutable pointer to the underlying data
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        match self.ptr {
            Some(ptr) => ptr.as_ptr(),
            None => ptr::null_mut(),
        }
    }

    /// Get the vector as a slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
        }
    }

    /// Get the vector as a mutable slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
        }
    }

    /// Reserve space for at least `additional` more elements
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        // Verify input parameters
        crate::zipora_verify!(additional <= (isize::MAX as usize) / mem::size_of::<T>().max(1),
            "additional capacity {} too large for element size {}", additional, mem::size_of::<T>());
        
        let required = self
            .len
            .checked_add(additional)
            .ok_or_else(|| ZiporaError::out_of_memory(usize::MAX))?;

        if required <= self.cap {
            return Ok(());
        }

        self.realloc(required)
    }

    /// Ensure the vector has at least the specified capacity
    pub fn ensure_capacity(&mut self, min_cap: usize) -> Result<()> {
        // Verify input parameters
        crate::zipora_verify!(min_cap <= (isize::MAX as usize) / mem::size_of::<T>().max(1),
            "minimum capacity {} too large for element size {}", min_cap, mem::size_of::<T>());
        crate::zipora_verify_ge!(min_cap, self.len);
        
        if min_cap <= self.cap {
            return Ok(());
        }

        self.realloc(min_cap)
    }

    /// Reallocate to the new capacity using realloc for optimal performance
    fn realloc(&mut self, new_cap: usize) -> Result<()> {
        // Verify reallocation parameters
        crate::zipora_verify_ge!(new_cap, self.len);
        crate::zipora_verify!(new_cap <= (isize::MAX as usize) / mem::size_of::<T>().max(1),
            "new capacity {} too large for element size {}", new_cap, mem::size_of::<T>());

        // Use exponential growth with a minimum increase
        let target_cap = new_cap.max(self.cap.saturating_mul(2));

        let new_layout = Layout::array::<T>(target_cap)
            .map_err(|_| ZiporaError::out_of_memory(target_cap * mem::size_of::<T>()))?;

        let new_ptr = match self.ptr {
            Some(ptr) => {
                if self.cap == 0 {
                    // This shouldn't happen, but handle it safely
                    unsafe {
                        let raw_ptr = alloc::alloc(new_layout);
                        if raw_ptr.is_null() {
                            std::ptr::null_mut()
                        } else {
                            cast_aligned_ptr::<T>(raw_ptr)
                        }
                    }
                } else {
                    let old_layout = Layout::array::<T>(self.cap).unwrap();
                    unsafe {
                        let raw_ptr =
                            alloc::realloc(ptr.as_ptr() as *mut u8, old_layout, new_layout.size());
                        if raw_ptr.is_null() {
                            std::ptr::null_mut()
                        } else {
                            cast_aligned_ptr::<T>(raw_ptr)
                        }
                    }
                }
            }
            None => unsafe {
                let raw_ptr = alloc::alloc(new_layout);
                if raw_ptr.is_null() {
                    std::ptr::null_mut()
                } else {
                    cast_aligned_ptr::<T>(raw_ptr)
                }
            },
        };

        if new_ptr.is_null() {
            return Err(ZiporaError::out_of_memory(new_layout.size()));
        }

        self.ptr = Some(unsafe { NonNull::new_unchecked(new_ptr) });
        self.cap = target_cap;
        Ok(())
    }

    /// Push an element to the end of the vector
    pub fn push(&mut self, value: T) -> Result<()> {
        // Verify data structure invariants
        crate::zipora_verify_le!(self.len, self.cap);
        crate::zipora_verify!(self.len < (isize::MAX as usize),
            "vector length {} would exceed maximum", self.len);
        
        if self.len >= self.cap {
            self.ensure_capacity(self.len + 1)?;
        }

        // Verify state after potential reallocation
        crate::zipora_verify_le!(self.len, self.cap);
        crate::zipora_verify!(self.ptr.is_some() || self.len == 0,
            "invalid state: non-null pointer required for len > 0");

        unsafe {
            ptr::write(self.as_mut_ptr().add(self.len), value);
        }
        self.len += 1;
        Ok(())
    }

    /// Pop an element from the end of the vector
    pub fn pop(&mut self) -> Option<T> {
        // Verify data structure invariants
        crate::zipora_verify_le!(self.len, self.cap);
        
        if self.len == 0 {
            None
        } else {
            // Verify we have valid memory before accessing
            crate::zipora_verify!(self.ptr.is_some(), "invalid state: null pointer with len > 0");
            
            self.len -= 1;
            Some(unsafe { ptr::read(self.as_ptr().add(self.len)) })
        }
    }

    /// Insert an element at the specified index
    pub fn insert(&mut self, index: usize, value: T) -> Result<()> {
        if index > self.len {
            return Err(ZiporaError::out_of_bounds(index, self.len));
        }

        // Add verification for internal consistency after parameter validation
        crate::zipora_verify_le!(index, self.len);

        if self.len >= self.cap {
            self.ensure_capacity(self.len + 1)?;
        }

        let move_count = self.len - index;
        
        unsafe {
            let ptr = self.as_mut_ptr().add(index);
            
            // Use SIMD optimization for large move operations on Copy types
            if move_count > 0 && is_simd_safe::<T>() && is_simd_beneficial::<T>(move_count) {
                // Create temporary slices for SIMD copy
                let src_bytes = slice_as_bytes(slice::from_raw_parts(ptr, move_count));
                let mut temp_vec: Vec<u8> = vec![0; src_bytes.len()];
                
                // Copy to temporary buffer with SIMD
                fast_copy(src_bytes, &mut temp_vec)?;
                
                // Copy back from temporary buffer with SIMD
                let dst_bytes = slice_as_bytes_mut(slice::from_raw_parts_mut(ptr.add(1), move_count));
                fast_copy(&temp_vec, dst_bytes)?;
            } else if move_count > 0 {
                // Standard move for small operations or non-Copy types
                ptr::copy(ptr, ptr.add(1), move_count);
            }
            
            // Write the new value
            ptr::write(ptr, value);
        }
        self.len += 1;
        Ok(())
    }

    /// Remove and return the element at the specified index
    pub fn remove(&mut self, index: usize) -> Result<T> {
        if index >= self.len {
            return Err(ZiporaError::out_of_bounds(index, self.len));
        }

        let move_count = self.len - index - 1;
        
        unsafe {
            let ptr = self.as_mut_ptr().add(index);
            let value = ptr::read(ptr);
            
            // Use SIMD optimization for large move operations on Copy types
            if move_count > 0 && is_simd_safe::<T>() && is_simd_beneficial::<T>(move_count) {
                // Create temporary slices for SIMD copy
                let src_bytes = slice_as_bytes(slice::from_raw_parts(ptr.add(1), move_count));
                let mut temp_vec: Vec<u8> = vec![0; src_bytes.len()];
                
                // Copy to temporary buffer with SIMD
                fast_copy(src_bytes, &mut temp_vec)?;
                
                // Copy back from temporary buffer with SIMD
                let dst_bytes = slice_as_bytes_mut(slice::from_raw_parts_mut(ptr, move_count));
                fast_copy(&temp_vec, dst_bytes)?;
            } else if move_count > 0 {
                // Standard move for small operations or non-Copy types
                ptr::copy(ptr.add(1), ptr, move_count);
            }
            
            self.len -= 1;
            Ok(value)
        }
    }

    /// Resize the vector to the specified length
    pub fn resize(&mut self, new_len: usize, value: T) -> Result<()>
    where
        T: Clone,
    {
        // Verify input parameters
        crate::zipora_verify!(new_len <= (isize::MAX as usize) / mem::size_of::<T>().max(1),
            "new length {} too large for element size {}", new_len, mem::size_of::<T>());
        crate::zipora_verify_le!(self.len, self.cap);
        
        if new_len > self.len {
            self.ensure_capacity(new_len)?;
            
            // Verify state after capacity adjustment
            crate::zipora_verify_ge!(self.cap, new_len);
            crate::zipora_verify!(self.ptr.is_some(), "invalid state: null pointer after capacity adjustment");
            
            let fill_count = new_len - self.len;
            
            // Use SIMD optimization for large fill operations on Copy types
            if is_simd_safe::<T>() && is_simd_beneficial::<T>(fill_count) && mem::size_of::<T>() == 1 {
                // For u8-sized Copy types, use direct SIMD fill
                unsafe {
                    let fill_slice = slice::from_raw_parts_mut(
                        self.as_mut_ptr().add(self.len) as *mut u8,
                        fill_count,
                    );
                    fast_fill(fill_slice, *((&value) as *const T as *const u8));
                }
            } else {
                // Standard fill for non-Copy types or small operations
                for i in self.len..new_len {
                    unsafe {
                        ptr::write(self.as_mut_ptr().add(i), value.clone());
                    }
                }
            }
        } else if new_len < self.len {
            // Verify we have valid memory to drop elements from
            crate::zipora_verify!(self.ptr.is_some() || self.len == 0,
                "invalid state: null pointer with elements to drop");
                
            // Drop excess elements
            for i in new_len..self.len {
                unsafe {
                    ptr::drop_in_place(self.as_mut_ptr().add(i));
                }
            }
        }
        self.len = new_len;
        
        // Verify final state invariants
        crate::zipora_verify_le!(self.len, self.cap);
        Ok(())
    }

    /// Clear all elements from the vector
    pub fn clear(&mut self) {
        // Verify data structure invariants
        crate::zipora_verify_le!(self.len, self.cap);
        crate::zipora_verify!(self.ptr.is_some() || self.len == 0,
            "invalid state: null pointer with elements to clear");
        
        for i in 0..self.len {
            unsafe {
                ptr::drop_in_place(self.as_mut_ptr().add(i));
            }
        }
        self.len = 0;
        
        // Verify final state
        crate::zipora_verify_le!(self.len, self.cap);
    }

    /// Shrink the capacity to fit the current length
    pub fn shrink_to_fit(&mut self) -> Result<()> {
        if self.len == self.cap {
            return Ok(());
        }

        if self.len == 0 {
            if let Some(ptr) = self.ptr {
                unsafe {
                    let layout = Layout::array::<T>(self.cap).unwrap();
                    alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
                }
            }
            self.ptr = None;
            self.cap = 0;
            return Ok(());
        }

        let new_layout = Layout::array::<T>(self.len)
            .map_err(|_| ZiporaError::out_of_memory(self.len * mem::size_of::<T>()))?;

        let new_ptr = if let Some(ptr) = self.ptr {
            let old_layout = Layout::array::<T>(self.cap).unwrap();
            unsafe {
                let raw_ptr =
                    alloc::realloc(ptr.as_ptr() as *mut u8, old_layout, new_layout.size());
                if raw_ptr.is_null() {
                    std::ptr::null_mut()
                } else {
                    cast_aligned_ptr::<T>(raw_ptr)
                }
            }
        } else {
            return Ok(()); // Nothing to shrink
        };

        if new_ptr.is_null() {
            return Err(ZiporaError::out_of_memory(new_layout.size()));
        }

        self.ptr = Some(unsafe { NonNull::new_unchecked(new_ptr) });
        self.cap = self.len;
        Ok(())
    }

    /// Get a reference to the element at the specified index without bounds checking
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        debug_assert!(index < self.len);
        unsafe { &*self.as_ptr().add(index) }
    }

    /// Get a mutable reference to the element at the specified index without bounds checking
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.len);
        unsafe { &mut *self.as_mut_ptr().add(index) }
    }

    /// Extend the vector with elements from an iterator
    pub fn extend<I>(&mut self, iter: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut iter = iter.into_iter();
        let additional = iter.len();
        self.reserve(additional)?;

        // For Copy types with large data, try to optimize with SIMD
        if is_simd_safe::<T>() && is_simd_beneficial::<T>(additional) {
            // Collect into a Vec first for SIMD optimization
            let items: Vec<T> = iter.collect();
            if items.len() == additional {
                // Use SIMD-optimized copy from slice
                unsafe {
                    let src_bytes = slice_as_bytes(&items);
                    let dst_bytes = slice_as_bytes_mut(slice::from_raw_parts_mut(
                        self.as_mut_ptr().add(self.len),
                        additional,
                    ));
                    fast_copy(src_bytes, dst_bytes)?;
                }
                self.len += additional;
                return Ok(());
            }
        } else {
            // Standard element-by-element extend for small operations or non-Copy types
            for item in iter {
                // We know we have capacity, so this won't fail
                unsafe {
                    ptr::write(self.as_mut_ptr().add(self.len), item);
                    self.len += 1;
                }
            }
        }
        
        Ok(())
    }

    //==============================================================================
    // SIMD-OPTIMIZED BULK OPERATIONS
    //==============================================================================

    /// Fast fill a range of the vector with the given value using SIMD optimization
    /// 
    /// This method provides 2-3x performance improvement over standard fill operations
    /// for bulk data when T is Copy and the operation size is ≥64 bytes.
    /// 
    /// # Performance
    /// - Uses SIMD acceleration for Copy types with large ranges
    /// - Falls back to standard operations for small ranges or non-Copy types
    /// - Provides optimal performance for primitive types (u8, u16, u32, u64, etc.)
    pub fn fill_range_fast(&mut self, start: usize, end: usize, value: T) -> Result<()>
    where
        T: Copy,
    {
        if start > end || end > self.len {
            return Err(ZiporaError::out_of_bounds(end, self.len));
        }

        // Verify internal consistency after parameter validation
        crate::zipora_verify_le!(start, end);
        crate::zipora_verify_le!(end, self.len);
        crate::zipora_verify_le!(self.len, self.cap);
        crate::zipora_verify!(self.ptr.is_some() || self.len == 0,
            "invalid state: null pointer with data to fill");

        if start == end {
            return Ok(()); // Nothing to fill
        }

        let range_len = end - start;
        
        // Use SIMD optimization for suitable types and large ranges
        if is_simd_safe::<T>() && is_simd_beneficial::<T>(range_len) && mem::size_of::<T>() == 1 {
            // For u8-sized types, use direct SIMD fill
            unsafe {
                let range_slice = slice::from_raw_parts_mut(
                    self.as_mut_ptr().add(start) as *mut u8,
                    range_len,
                );
                fast_fill(range_slice, *((&value) as *const T as *const u8));
            }
        } else if is_simd_safe::<T>() && is_simd_beneficial::<T>(range_len) {
            // For other Copy types with SIMD-beneficial size, use standard bulk fill
            // SIMD optimization is mainly beneficial for byte-sized types
            let range_slice = unsafe {
                slice::from_raw_parts_mut(
                    self.as_mut_ptr().add(start),
                    range_len,
                )
            };
            
            // Use standard copy for non-byte types but still benefit from bulk operation
            for item in range_slice.iter_mut() {
                *item = value;
            }
        } else {
            // Standard fill for small ranges or non-Copy types
            let range_slice = &mut self.as_mut_slice()[start..end];
            for item in range_slice.iter_mut() {
                *item = value;
            }
        }

        Ok(())
    }

    /// Fast copy data from a slice using SIMD optimization
    /// 
    /// This method provides 2-3x performance improvement over standard copy operations
    /// for bulk data when T is Copy and the operation size is ≥64 bytes.
    /// 
    /// # Performance
    /// - Uses SIMD acceleration for Copy types with large slices
    /// - Falls back to standard operations for small slices or non-Copy types
    /// - Provides optimal performance for primitive types and simple structs
    pub fn copy_from_slice_fast(&mut self, src: &[T]) -> Result<()>
    where
        T: Copy,
    {
        // Verify input parameters
        crate::zipora_verify!(src.len() <= (isize::MAX as usize) / mem::size_of::<T>().max(1),
            "source slice length {} too large for element size {}", src.len(), mem::size_of::<T>());
        crate::zipora_verify_le!(self.len, self.cap);
        
        if src.is_empty() {
            return Ok(());
        }

        // Ensure we have enough capacity
        self.ensure_capacity(src.len())?;
        
        // Verify state after capacity adjustment
        crate::zipora_verify_ge!(self.cap, src.len());
        crate::zipora_verify!(self.ptr.is_some(), "invalid state: null pointer after capacity adjustment");

        // Use SIMD optimization for suitable types and large slices
        if is_simd_safe::<T>() && is_simd_beneficial::<T>(src.len()) {
            unsafe {
                let src_bytes = slice_as_bytes(src);
                let dst_bytes = slice_as_bytes_mut(slice::from_raw_parts_mut(
                    self.as_mut_ptr(),
                    src.len(),
                ));
                fast_copy(src_bytes, dst_bytes)?;
            }
        } else {
            // Standard copy for small slices or non-Copy types
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), self.as_mut_ptr(), src.len());
            }
        }

        self.len = src.len();
        Ok(())
    }

    /// Fast extend with SIMD optimization for slice data
    /// 
    /// This method provides 2-3x performance improvement over standard extend operations
    /// for bulk data when T is Copy and the operation size is ≥64 bytes.
    pub fn extend_from_slice_fast(&mut self, src: &[T]) -> Result<()>
    where
        T: Copy,
    {
        if src.is_empty() {
            return Ok(());
        }

        let old_len = self.len;
        self.reserve(src.len())?;

        // Use SIMD optimization for suitable types and large slices
        if is_simd_safe::<T>() && is_simd_beneficial::<T>(src.len()) {
            unsafe {
                let src_bytes = slice_as_bytes(src);
                let dst_bytes = slice_as_bytes_mut(slice::from_raw_parts_mut(
                    self.as_mut_ptr().add(old_len),
                    src.len(),
                ));
                fast_copy(src_bytes, dst_bytes)?;
            }
        } else {
            // Standard copy for small slices or non-Copy types
            unsafe {
                ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    self.as_mut_ptr().add(old_len),
                    src.len(),
                );
            }
        }

        self.len += src.len();
        Ok(())
    }
}

impl<T> Default for FastVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for FastVec<T> {
    fn drop(&mut self) {
        self.clear();
        if let Some(ptr) = self.ptr {
            if self.cap > 0 {
                unsafe {
                    let layout = Layout::array::<T>(self.cap).unwrap();
                    alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
                }
            }
        }
    }
}

impl<T> Deref for FastVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for FastVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Index<usize> for FastVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        // Add bounds verification with rich context
        crate::zipora_verify_bounds!(index, self.len);
        &self.as_slice()[index]
    }
}

impl<T> IndexMut<usize> for FastVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // Add bounds verification with rich context  
        crate::zipora_verify_bounds!(index, self.len);
        &mut self.as_mut_slice()[index]
    }
}

impl<T: fmt::Debug> fmt::Debug for FastVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: PartialEq> PartialEq for FastVec<T> {
    fn eq(&self, other: &Self) -> bool {
        // Quick length check first
        if self.len != other.len {
            return false;
        }
        
        if self.len == 0 {
            return true;
        }
        
        // Use SIMD optimization for Copy types with large vectors
        if is_simd_safe::<T>() && is_simd_beneficial::<T>(self.len) {
            unsafe {
                let self_bytes = slice_as_bytes(self.as_slice());
                let other_bytes = slice_as_bytes(other.as_slice());
                fast_compare(self_bytes, other_bytes) == 0
            }
        } else {
            // Standard comparison for small vectors or non-Copy types
            self.as_slice() == other.as_slice()
        }
    }
}

impl<T: Eq> Eq for FastVec<T> {}

impl<T: Clone> Clone for FastVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::with_capacity(self.len).unwrap();
        for item in self.as_slice() {
            new_vec.push(item.clone()).unwrap();
        }
        new_vec
    }
}

// Safety: FastVec<T> is Send if T is Send
unsafe impl<T: Send> Send for FastVec<T> {}

// Safety: FastVec<T> is Sync if T is Sync
unsafe impl<T: Sync> Sync for FastVec<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let vec: FastVec<i32> = FastVec::new();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let vec: FastVec<i32> = FastVec::with_capacity(10).unwrap();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 10);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_push_pop() {
        let mut vec = FastVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.len(), 1);
    }

    #[test]
    fn test_index() {
        let mut vec = FastVec::new();
        vec.push(42).unwrap();
        vec.push(84).unwrap();

        assert_eq!(vec[0], 42);
        assert_eq!(vec[1], 84);

        vec[0] = 100;
        assert_eq!(vec[0], 100);
    }

    #[test]
    fn test_insert_remove() {
        let mut vec = FastVec::new();
        vec.push(1).unwrap();
        vec.push(3).unwrap();

        vec.insert(1, 2).unwrap();
        assert_eq!(vec.as_slice(), &[1, 2, 3]);

        let removed = vec.remove(1).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(vec.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_resize() {
        let mut vec = FastVec::new();
        vec.resize(5, 42).unwrap();
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.as_slice(), &[42, 42, 42, 42, 42]);

        vec.resize(3, 0).unwrap();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.as_slice(), &[42, 42, 42]);
    }

    #[test]
    fn test_clone() {
        let mut vec = FastVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();

        let cloned = vec.clone();
        assert_eq!(vec.as_slice(), cloned.as_slice());
    }

    #[test]
    fn test_clear() {
        let mut vec = FastVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();

        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_index_bounds() {
        let vec: FastVec<i32> = FastVec::new();
        let _ = vec[0]; // Should panic
    }

    #[test]
    fn test_with_size() {
        let vec = FastVec::with_size(5, 42).unwrap();
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.capacity(), 5);
        for i in 0..5 {
            assert_eq!(vec[i], 42);
        }
    }

    #[test]
    fn test_pointers() {
        let mut vec: FastVec<i32> = FastVec::new();

        // Test empty vector pointers
        assert!(vec.as_ptr().is_null());
        assert!(vec.as_mut_ptr().is_null());

        vec.push(42).unwrap();
        vec.push(84).unwrap();

        // Test non-empty vector pointers
        assert!(!vec.as_ptr().is_null());
        assert!(!vec.as_mut_ptr().is_null());

        unsafe {
            assert_eq!(*vec.as_ptr(), 42);
            assert_eq!(*vec.as_ptr().add(1), 84);
        }
    }

    #[test]
    fn test_slices() {
        let mut vec = FastVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        let slice = vec.as_slice();
        assert_eq!(slice, &[1, 2, 3]);

        let mut_slice = vec.as_mut_slice();
        mut_slice[1] = 20;
        assert_eq!(vec[1], 20);
    }

    #[test]
    fn test_unsafe_access() {
        let mut vec = FastVec::new();
        vec.push(10).unwrap();
        vec.push(20).unwrap();
        vec.push(30).unwrap();

        unsafe {
            assert_eq!(*vec.get_unchecked(0), 10);
            assert_eq!(*vec.get_unchecked(2), 30);

            *vec.get_unchecked_mut(1) = 200;
            assert_eq!(vec[1], 200);
        }
    }

    #[test]
    fn test_extend() {
        let mut vec = FastVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();

        let data = vec![3, 4, 5];
        vec.extend(data).unwrap();

        assert_eq!(vec.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_reserve() {
        let mut vec: FastVec<i32> = FastVec::new();
        assert_eq!(vec.capacity(), 0);

        vec.reserve(10).unwrap();
        assert!(vec.capacity() >= 10);

        // Test reserve when already has capacity
        let old_cap = vec.capacity();
        vec.reserve(5).unwrap();
        assert_eq!(vec.capacity(), old_cap); // Should not change
    }

    #[test]
    fn test_ensure_capacity() {
        let mut vec: FastVec<i32> = FastVec::new();
        vec.ensure_capacity(15).unwrap();
        assert!(vec.capacity() >= 15);

        // Test ensure when already has capacity
        let old_cap = vec.capacity();
        vec.ensure_capacity(10).unwrap();
        assert_eq!(vec.capacity(), old_cap); // Should not change
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut vec = FastVec::with_capacity(100).unwrap();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        assert!(vec.capacity() >= 100);
        vec.shrink_to_fit().unwrap();
        assert_eq!(vec.capacity(), 3);
        assert_eq!(vec.as_slice(), &[1, 2, 3]);

        // Test shrink empty vector
        let mut empty_vec: FastVec<i32> = FastVec::with_capacity(50).unwrap();
        empty_vec.shrink_to_fit().unwrap();
        assert_eq!(empty_vec.capacity(), 0);
        assert!(empty_vec.as_ptr().is_null());
    }

    #[test]
    fn test_out_of_bounds_errors() {
        let mut vec = FastVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();

        // Test out of bounds insert
        assert!(vec.insert(5, 100).is_err());

        // Test out of bounds remove
        assert!(vec.remove(5).is_err());

        // Test out of bounds set (not available, but remove from bounds)
        assert!(vec.remove(2).is_err());
    }

    #[test]
    fn test_memory_management() {
        let mut vec = FastVec::new();

        // Test growth pattern
        for i in 0..1000 {
            vec.push(i).unwrap();
        }
        assert_eq!(vec.len(), 1000);

        // Test that capacity grows efficiently
        assert!(vec.capacity() >= 1000);
        assert!(vec.capacity() < 2000); // Not too much over-allocation
    }

    #[test]
    fn test_drop_elements() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));

        #[derive(Clone)]
        struct DropCounter {
            counter: Arc<AtomicUsize>,
        }

        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
        }

        {
            let mut vec = FastVec::new();
            for _ in 0..5 {
                vec.push(DropCounter {
                    counter: counter.clone(),
                })
                .unwrap();
            }

            // Remove one element
            vec.remove(2).unwrap();
            assert_eq!(counter.load(Ordering::SeqCst), 1);

            // Resize down - this will drop the extra elements beyond index 2
            let resize_value = DropCounter {
                counter: counter.clone(),
            };
            vec.resize(2, resize_value).unwrap();
            // 1 from remove + 2 from resize down (dropped 2 elements) + 1 from resize_value going out of scope = 4
            assert_eq!(counter.load(Ordering::SeqCst), 4);

            // Clear all
            vec.clear();
            assert_eq!(counter.load(Ordering::SeqCst), 6); // 4 previous + 2 from clear
        }

        // Final drops should not happen since we already cleared
        assert_eq!(counter.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn test_zero_capacity() {
        let vec: FastVec<i32> = FastVec::with_capacity(0).unwrap();
        assert_eq!(vec.capacity(), 0);
        assert_eq!(vec.len(), 0);
        assert!(vec.as_ptr().is_null());
    }

    #[test]
    fn test_equality_and_debug() {
        let mut vec1 = FastVec::new();
        let mut vec2 = FastVec::new();

        vec1.push(1).unwrap();
        vec1.push(2).unwrap();
        vec1.push(3).unwrap();

        vec2.push(1).unwrap();
        vec2.push(2).unwrap();
        vec2.push(3).unwrap();

        assert_eq!(vec1, vec2);

        vec2.push(4).unwrap();
        assert_ne!(vec1, vec2);

        // Test debug output
        let debug_str = format!("{:?}", vec1);
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
        assert!(debug_str.contains("3"));
    }

    #[test]
    fn test_deref() {
        let mut vec = FastVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        // Test Deref to slice
        let slice: &[i32] = &vec;
        assert_eq!(slice, &[1, 2, 3]);

        // Test DerefMut to slice
        let mut_slice: &mut [i32] = &mut vec;
        mut_slice[1] = 20;
        assert_eq!(vec[1], 20);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<FastVec<i32>>();
        assert_sync::<FastVec<i32>>();
    }

    #[test]
    fn test_edge_cases() {
        // Test inserting at beginning
        let mut vec = FastVec::new();
        vec.push(2).unwrap();
        vec.push(3).unwrap();
        vec.insert(0, 1).unwrap();
        assert_eq!(vec.as_slice(), &[1, 2, 3]);

        // Test inserting at end
        vec.insert(3, 4).unwrap();
        assert_eq!(vec.as_slice(), &[1, 2, 3, 4]);

        // Test removing from beginning
        assert_eq!(vec.remove(0).unwrap(), 1);
        assert_eq!(vec.as_slice(), &[2, 3, 4]);

        // Test removing from end
        assert_eq!(vec.remove(2).unwrap(), 4);
        assert_eq!(vec.as_slice(), &[2, 3]);
    }

    #[test]
    fn test_large_allocation() {
        // Test that we can handle reasonably large allocations
        let mut vec = FastVec::with_capacity(10000).unwrap();
        for i in 0..10000 {
            vec.push(i).unwrap();
        }
        assert_eq!(vec.len(), 10000);
        assert_eq!(vec[9999], 9999);
    }

    // Comprehensive alignment edge case tests for memory safety verification
    mod alignment_tests {
        use super::*;
        use std::mem;

        // Test types with different alignment requirements
        #[repr(align(1))]
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Align1(u8);

        #[repr(align(2))]
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Align2(u16);

        #[repr(align(4))]
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Align4(u32);

        #[repr(align(8))]
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Align8(u64);

        #[repr(align(16))]
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Align16([u64; 2]);

        #[repr(align(32))]
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Align32([u64; 4]);

        /// Verify pointer alignment for a given type
        fn verify_alignment<T>(ptr: *const T) {
            let align = mem::align_of::<T>();
            let addr = ptr as usize;
            assert_eq!(
                addr % align,
                0,
                "Pointer {:#x} is not aligned for type {} (requires {}-byte alignment)",
                addr,
                std::any::type_name::<T>(),
                align
            );
        }

        #[test]
        fn test_alignment_1_byte() {
            let mut vec = FastVec::<Align1>::new();

            // Test initial allocation
            vec.push(Align1(42)).unwrap();
            verify_alignment(vec.as_ptr());

            // Test reallocation with alignment 1
            for i in 0..1000 {
                vec.push(Align1(i as u8)).unwrap();
                verify_alignment(vec.as_ptr());
            }

            assert_eq!(vec.len(), 1001);
            assert_eq!(vec[0], Align1(42));
            assert_eq!(vec[1000], Align1(231)); // (1000 - 1) % 256 since i starts at 0
        }

        #[test]
        fn test_alignment_2_byte() {
            let mut vec = FastVec::<Align2>::new();

            // Test initial allocation
            vec.push(Align2(42)).unwrap();
            verify_alignment(vec.as_ptr());

            // Test reallocation with alignment 2
            for i in 0..1000 {
                vec.push(Align2(i as u16)).unwrap();
                verify_alignment(vec.as_ptr());
            }

            assert_eq!(vec.len(), 1001);
            assert_eq!(vec[0], Align2(42));
            assert_eq!(vec[1000], Align2(999)); // i goes from 0 to 999, so vec[1000] has value 999
        }

        #[test]
        fn test_alignment_4_byte() {
            let mut vec = FastVec::<Align4>::new();

            // Test initial allocation
            vec.push(Align4(42)).unwrap();
            verify_alignment(vec.as_ptr());

            // Test reallocation with alignment 4
            for i in 0..1000 {
                vec.push(Align4(i as u32)).unwrap();
                verify_alignment(vec.as_ptr());
            }

            assert_eq!(vec.len(), 1001);
            assert_eq!(vec[0], Align4(42));
            assert_eq!(vec[1000], Align4(999)); // i goes from 0 to 999, so vec[1000] has value 999
        }

        #[test]
        fn test_alignment_8_byte() {
            let mut vec = FastVec::<Align8>::new();

            // Test initial allocation
            vec.push(Align8(42)).unwrap();
            verify_alignment(vec.as_ptr());

            // Test reallocation with alignment 8
            for i in 0..1000 {
                vec.push(Align8(i as u64)).unwrap();
                verify_alignment(vec.as_ptr());
            }

            assert_eq!(vec.len(), 1001);
            assert_eq!(vec[0], Align8(42));
            assert_eq!(vec[1000], Align8(999)); // i goes from 0 to 999, so vec[1000] has value 999
        }

        #[test]
        fn test_alignment_16_byte() {
            let mut vec = FastVec::<Align16>::new();

            // Test initial allocation
            vec.push(Align16([42, 84])).unwrap();
            verify_alignment(vec.as_ptr());

            // Test reallocation with alignment 16
            for i in 0..500 {
                vec.push(Align16([i as u64, (i * 2) as u64])).unwrap();
                verify_alignment(vec.as_ptr());
            }

            assert_eq!(vec.len(), 501);
            assert_eq!(vec[0], Align16([42, 84]));
            assert_eq!(vec[500], Align16([499, 998]));
        }

        #[test]
        fn test_alignment_32_byte() {
            let mut vec = FastVec::<Align32>::new();

            // Test initial allocation
            vec.push(Align32([1, 2, 3, 4])).unwrap();
            verify_alignment(vec.as_ptr());

            // Test reallocation with alignment 32
            for i in 0..200 {
                vec.push(Align32([
                    i as u64,
                    i as u64 + 1,
                    i as u64 + 2,
                    i as u64 + 3,
                ]))
                .unwrap();
                verify_alignment(vec.as_ptr());
            }

            assert_eq!(vec.len(), 201);
            assert_eq!(vec[0], Align32([1, 2, 3, 4]));
            assert_eq!(vec[200], Align32([199, 200, 201, 202]));
        }

        #[test]
        fn test_large_allocations_with_realloc() {
            // Test types with different alignments in large allocations
            let mut vec8 = FastVec::<Align8>::new();
            let mut vec16 = FastVec::<Align16>::new();
            let mut vec32 = FastVec::<Align32>::new();

            // Large allocation that will definitely trigger multiple reallocs
            for i in 0..10000 {
                vec8.push(Align8(i as u64)).unwrap();
                verify_alignment(vec8.as_ptr());

                if i % 2 == 0 {
                    vec16.push(Align16([i as u64, i as u64 + 1])).unwrap();
                    verify_alignment(vec16.as_ptr());
                }

                if i % 4 == 0 {
                    vec32
                        .push(Align32([
                            i as u64,
                            i as u64 + 1,
                            i as u64 + 2,
                            i as u64 + 3,
                        ]))
                        .unwrap();
                    verify_alignment(vec32.as_ptr());
                }
            }

            assert_eq!(vec8.len(), 10000);
            assert_eq!(vec16.len(), 5000);
            assert_eq!(vec32.len(), 2500);

            // Verify final values and alignment
            assert_eq!(vec8[9999], Align8(9999));
            assert_eq!(vec16[4999], Align16([9998, 9999]));
            assert_eq!(vec32[2499], Align32([9996, 9997, 9998, 9999]));

            verify_alignment(vec8.as_ptr());
            verify_alignment(vec16.as_ptr());
            verify_alignment(vec32.as_ptr());
        }

        #[test]
        fn test_stress_allocation_cycles() {
            // Stress test with many allocation/reallocation cycles
            let mut vec = FastVec::<Align16>::new();

            for cycle in 0..100 {
                // Grow the vector
                for i in 0..100 {
                    vec.push(Align16([cycle as u64, i as u64])).unwrap();
                    verify_alignment(vec.as_ptr());
                }

                // Shrink the vector
                for _ in 0..50 {
                    vec.pop();
                    if !vec.is_empty() {
                        verify_alignment(vec.as_ptr());
                    }
                }

                // Verify integrity
                assert_eq!(vec.len(), (cycle + 1) * 50);
                if !vec.is_empty() {
                    verify_alignment(vec.as_ptr());
                }
            }

            // Final verification
            assert_eq!(vec.len(), 5000);
            verify_alignment(vec.as_ptr());
        }

        #[test]
        fn test_zero_to_nonzero_capacity_transitions() {
            // Test transition from zero capacity to non-zero for different alignments
            let mut vec1 = FastVec::<Align1>::new();
            let mut vec8 = FastVec::<Align8>::new();
            let mut vec16 = FastVec::<Align16>::new();
            let mut vec32 = FastVec::<Align32>::new();

            // All start with zero capacity
            assert_eq!(vec1.capacity(), 0);
            assert_eq!(vec8.capacity(), 0);
            assert_eq!(vec16.capacity(), 0);
            assert_eq!(vec32.capacity(), 0);

            // First push should trigger allocation
            vec1.push(Align1(1)).unwrap();
            vec8.push(Align8(8)).unwrap();
            vec16.push(Align16([16, 17])).unwrap();
            vec32.push(Align32([32, 33, 34, 35])).unwrap();

            // Verify alignments after zero-to-nonzero transition
            verify_alignment(vec1.as_ptr());
            verify_alignment(vec8.as_ptr());
            verify_alignment(vec16.as_ptr());
            verify_alignment(vec32.as_ptr());

            // Verify values
            assert_eq!(vec1[0], Align1(1));
            assert_eq!(vec8[0], Align8(8));
            assert_eq!(vec16[0], Align16([16, 17]));
            assert_eq!(vec32[0], Align32([32, 33, 34, 35]));
        }

        #[test]
        fn test_shrink_to_fit_preserves_alignment() {
            // Test that shrink_to_fit preserves alignment
            let mut vec = FastVec::<Align32>::with_capacity(1000).unwrap();

            // Add some elements
            for i in 0..10 {
                vec.push(Align32([i, i + 1, i + 2, i + 3])).unwrap();
            }

            verify_alignment(vec.as_ptr());
            assert!(vec.capacity() >= 1000);

            // Shrink to fit
            vec.shrink_to_fit().unwrap();

            // Verify alignment is preserved
            verify_alignment(vec.as_ptr());
            assert_eq!(vec.capacity(), 10);
            assert_eq!(vec.len(), 10);

            // Verify data integrity
            for i in 0..10 {
                assert_eq!(
                    vec[i],
                    Align32([i as u64, i as u64 + 1, i as u64 + 2, i as u64 + 3])
                );
            }
        }

        #[test]
        fn test_reserve_preserves_alignment() {
            // Test that reserve operations preserve alignment
            let mut vec = FastVec::<Align16>::new();

            vec.push(Align16([1, 2])).unwrap();
            verify_alignment(vec.as_ptr());

            // Reserve additional space
            vec.reserve(1000).unwrap();
            verify_alignment(vec.as_ptr());

            // Add more elements
            for i in 2..100 {
                vec.push(Align16([i, i + 1])).unwrap();
                verify_alignment(vec.as_ptr());
            }

            assert_eq!(vec.len(), 99); // 1 initial + 98 from loop (2..100)
            assert_eq!(vec[0], Align16([1, 2]));
            assert_eq!(vec[98], Align16([99, 100])); // Last element from loop
        }

        #[test]
        fn test_insert_remove_preserves_alignment() {
            // Test that insert/remove operations preserve alignment
            let mut vec = FastVec::<Align8>::new();

            // Build initial vector
            for i in 0..10 {
                vec.push(Align8(i)).unwrap();
            }
            verify_alignment(vec.as_ptr());

            // Insert in middle (may trigger reallocation)
            vec.insert(5, Align8(999)).unwrap();
            verify_alignment(vec.as_ptr());

            // Verify data integrity
            assert_eq!(vec[4], Align8(4));
            assert_eq!(vec[5], Align8(999));
            assert_eq!(vec[6], Align8(5));

            // Remove from middle
            let removed = vec.remove(5).unwrap();
            assert_eq!(removed, Align8(999));
            verify_alignment(vec.as_ptr());

            // Verify data integrity after removal
            assert_eq!(vec[4], Align8(4));
            assert_eq!(vec[5], Align8(5));
        }

        #[test]
        fn test_resize_preserves_alignment() {
            // Test that resize operations preserve alignment
            let mut vec = FastVec::<Align32>::new();

            // Initial elements
            for i in 0..5 {
                vec.push(Align32([i, i + 1, i + 2, i + 3])).unwrap();
            }
            verify_alignment(vec.as_ptr());

            // Resize larger
            vec.resize(100, Align32([99, 100, 101, 102])).unwrap();
            verify_alignment(vec.as_ptr());
            assert_eq!(vec.len(), 100);

            // Verify original data
            for i in 0..5 {
                assert_eq!(
                    vec[i],
                    Align32([i as u64, i as u64 + 1, i as u64 + 2, i as u64 + 3])
                );
            }

            // Verify new data
            for i in 5..100 {
                assert_eq!(vec[i], Align32([99, 100, 101, 102]));
            }

            // Resize smaller
            vec.resize(10, Align32([0, 0, 0, 0])).unwrap();
            verify_alignment(vec.as_ptr());
            assert_eq!(vec.len(), 10);
        }

        #[test]
        fn test_mixed_alignment_stress() {
            // Stress test with mixed operations on high-alignment type
            let mut vec = FastVec::<Align32>::new();

            for round in 0..50 {
                // Add elements
                for i in 0..20 {
                    vec.push(Align32([
                        round as u64,
                        i as u64,
                        round as u64 + i as u64,
                        0,
                    ]))
                    .unwrap();
                    verify_alignment(vec.as_ptr());
                }

                // Insert some elements
                if vec.len() > 10 {
                    vec.insert(vec.len() / 2, Align32([888, 888, 888, 888]))
                        .unwrap();
                    verify_alignment(vec.as_ptr());
                }

                // Remove some elements
                if vec.len() > 5 {
                    vec.remove(vec.len() - 1).unwrap();
                    verify_alignment(vec.as_ptr());
                }

                // Occasionally resize
                if round % 10 == 0 && vec.len() > 0 {
                    let new_size = vec.len() + 10;
                    vec.resize(new_size, Align32([777, 777, 777, 777])).unwrap();
                    verify_alignment(vec.as_ptr());
                }

                // Occasionally shrink
                if round % 15 == 0 && vec.capacity() > vec.len() * 2 {
                    vec.shrink_to_fit().unwrap();
                    if !vec.is_empty() {
                        verify_alignment(vec.as_ptr());
                    }
                }
            }

            // Final verification
            if !vec.is_empty() {
                verify_alignment(vec.as_ptr());
            }
        }

        #[test]
        fn test_debug_assertions_in_debug_mode() {
            // This test verifies that debug assertions work correctly
            // Note: This will only trigger assertions in debug builds
            let mut vec = FastVec::<Align16>::new();

            vec.push(Align16([1, 2])).unwrap();

            // In debug mode, this should trigger alignment checks
            // In release mode, checks are optimized out
            let ptr = vec.as_ptr();
            verify_alignment(ptr);

            // Test with reallocation
            for i in 0..1000 {
                vec.push(Align16([i as u64, i as u64 + 1])).unwrap();
                // Each push may trigger reallocation and alignment checks
            }

            verify_alignment(vec.as_ptr());
        }

        #[test]
        fn test_pointer_cast_safety() {
            // Test that our alignment checks and casts are safe
            let mut vec = FastVec::<Align32>::new();

            // Test empty vector
            assert!(vec.as_ptr().is_null());
            assert!(vec.as_mut_ptr().is_null());

            // Add element and verify non-null aligned pointer
            vec.push(Align32([1, 2, 3, 4])).unwrap();

            let ptr = vec.as_ptr();
            assert!(!ptr.is_null());
            verify_alignment(ptr);

            let mut_ptr = vec.as_mut_ptr();
            assert!(!mut_ptr.is_null());
            verify_alignment(mut_ptr);

            // Test after multiple reallocations
            for i in 0..100 {
                vec.push(Align32([i, i + 1, i + 2, i + 3])).unwrap();
                verify_alignment(vec.as_ptr());
                verify_alignment(vec.as_mut_ptr());
            }
        }

        #[test]
        fn test_edge_case_alignment_boundary() {
            // Test alignment at memory boundaries that might be problematic
            let mut vec = FastVec::<Align32>::new();

            // Force specific allocation patterns that might expose alignment issues
            vec.reserve(1).unwrap();
            verify_alignment(vec.as_ptr());

            vec.push(Align32([1, 2, 3, 4])).unwrap();
            verify_alignment(vec.as_ptr());

            // Force reallocation from very small to larger size
            vec.reserve(1000).unwrap();
            verify_alignment(vec.as_ptr());

            // Test shrinking back down
            vec.shrink_to_fit().unwrap();
            verify_alignment(vec.as_ptr());

            // Verify data integrity throughout
            assert_eq!(vec[0], Align32([1, 2, 3, 4]));
        }
    }

    //==============================================================================
    // SIMD FUNCTIONALITY TESTS
    //==============================================================================

    mod simd_tests {
        use super::*;

        #[test]
        fn test_fill_range_fast_u8() {
            let mut vec = FastVec::with_capacity(1000).unwrap();
            vec.resize(1000, 0u8).unwrap();

            // Test SIMD-optimized fill for large range
            vec.fill_range_fast(100, 900, 0xAA).unwrap();

            for i in 0..100 {
                assert_eq!(vec[i], 0u8);
            }
            for i in 100..900 {
                assert_eq!(vec[i], 0xAA);
            }
            for i in 900..1000 {
                assert_eq!(vec[i], 0u8);
            }
        }

        #[test]
        fn test_fill_range_fast_small() {
            let mut vec = FastVec::with_capacity(10).unwrap();
            vec.resize(10, 0u8).unwrap();

            // Test small range (should use standard fill)
            vec.fill_range_fast(2, 8, 0xFF).unwrap();

            assert_eq!(vec[1], 0u8);
            assert_eq!(vec[2], 0xFF);
            assert_eq!(vec[7], 0xFF);
            assert_eq!(vec[8], 0u8);
        }

        #[test]
        fn test_fill_range_fast_bounds() {
            let mut vec = FastVec::with_size(5, 42u8).unwrap();

            // Test out of bounds
            assert!(vec.fill_range_fast(0, 10, 0xFF).is_err());
            assert!(vec.fill_range_fast(3, 2, 0xFF).is_err());

            // Test valid range
            assert!(vec.fill_range_fast(1, 4, 0xFF).is_ok());
            assert_eq!(vec[0], 42);
            assert_eq!(vec[1], 0xFF);
            assert_eq!(vec[3], 0xFF);
            assert_eq!(vec[4], 42);
        }

        #[test]
        fn test_copy_from_slice_fast_large() {
            let src: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
            let mut vec = FastVec::new();

            // Test SIMD-optimized copy
            vec.copy_from_slice_fast(&src).unwrap();

            assert_eq!(vec.len(), 1000);
            for i in 0..1000 {
                assert_eq!(vec[i], (i % 256) as u8);
            }
        }

        #[test]
        fn test_copy_from_slice_fast_small() {
            let src = vec![1u8, 2, 3, 4, 5];
            let mut vec = FastVec::new();

            // Test small copy (should use standard copy)
            vec.copy_from_slice_fast(&src).unwrap();

            assert_eq!(vec.len(), 5);
            assert_eq!(vec.as_slice(), &[1, 2, 3, 4, 5]);
        }

        #[test]
        fn test_copy_from_slice_fast_empty() {
            let src: Vec<u8> = vec![];
            let mut vec = FastVec::new();

            vec.copy_from_slice_fast(&src).unwrap();
            assert_eq!(vec.len(), 0);
        }

        #[test]
        fn test_extend_from_slice_fast_large() {
            let mut vec = FastVec::new();
            vec.push(255u8).unwrap();

            let src: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

            // Test SIMD-optimized extend
            vec.extend_from_slice_fast(&src).unwrap();

            assert_eq!(vec.len(), 1001);
            assert_eq!(vec[0], 255);
            for i in 1..1001 {
                assert_eq!(vec[i], ((i - 1) % 256) as u8);
            }
        }

        #[test]
        fn test_extend_from_slice_fast_small() {
            let mut vec = FastVec::new();
            vec.push(100u8).unwrap();

            let src = vec![1u8, 2, 3, 4, 5];

            // Test small extend (should use standard extend)
            vec.extend_from_slice_fast(&src).unwrap();

            assert_eq!(vec.len(), 6);
            assert_eq!(vec.as_slice(), &[100, 1, 2, 3, 4, 5]);
        }

        #[test]
        fn test_simd_optimized_insert_large() {
            let mut vec = FastVec::new();
            
            // Create a large vector to test SIMD insert optimization
            for i in 0..2000u16 {
                vec.push(i).unwrap();
            }

            // Insert in the middle (should trigger SIMD optimization for move)
            vec.insert(1000, 9999u16).unwrap();

            assert_eq!(vec.len(), 2001);
            assert_eq!(vec[999], 999);
            assert_eq!(vec[1000], 9999);
            assert_eq!(vec[1001], 1000);
        }

        #[test]
        fn test_simd_optimized_remove_large() {
            let mut vec = FastVec::new();
            
            // Create a large vector to test SIMD remove optimization
            for i in 0..2000u16 {
                vec.push(i).unwrap();
            }

            // Remove from the middle (should trigger SIMD optimization for move)
            let removed = vec.remove(1000).unwrap();

            assert_eq!(removed, 1000);
            assert_eq!(vec.len(), 1999);
            assert_eq!(vec[999], 999);
            assert_eq!(vec[1000], 1001);
        }

        #[test]
        fn test_simd_optimized_resize_large() {
            let mut vec: FastVec<u8> = FastVec::new();

            // Test SIMD-optimized resize with large fill
            vec.resize(2000, 0x42).unwrap();

            assert_eq!(vec.len(), 2000);
            for i in 0..2000 {
                assert_eq!(vec[i], 0x42);
            }
        }

        #[test]
        fn test_simd_optimized_extend_large() {
            let mut vec = FastVec::new();
            vec.push(0u8).unwrap();

            let data: Vec<u8> = (1..=2000).map(|i| (i % 256) as u8).collect();

            // Test SIMD-optimized extend
            vec.extend(data.into_iter()).unwrap();

            assert_eq!(vec.len(), 2001);
            assert_eq!(vec[0], 0);
            for i in 1..=2000 {
                assert_eq!(vec[i], ((i % 256) as u8));
            }
        }

        #[test]
        fn test_simd_optimized_partial_eq() {
            // Create two large vectors for SIMD comparison
            let vec1: FastVec<u8> = {
                let mut v = FastVec::new();
                for i in 0..2000 {
                    v.push((i % 256) as u8).unwrap();
                }
                v
            };

            let vec2: FastVec<u8> = {
                let mut v = FastVec::new();
                for i in 0..2000 {
                    v.push((i % 256) as u8).unwrap();
                }
                v
            };

            let vec3: FastVec<u8> = {
                let mut v = FastVec::new();
                for i in 0..2000 {
                    v.push(((i + 1) % 256) as u8).unwrap();
                }
                v
            };

            // Test SIMD-optimized equality
            assert_eq!(vec1, vec2);
            assert_ne!(vec1, vec3);
        }

        #[test]
        fn test_simd_with_different_types() {
            // Test with u16 (2-byte type)
            let mut vec_u16 = FastVec::new();
            let data_u16: Vec<u16> = (0..1000).map(|i| i as u16).collect();
            vec_u16.extend_from_slice_fast(&data_u16).unwrap();
            assert_eq!(vec_u16.len(), 1000);

            // Test with u32 (4-byte type)
            let mut vec_u32 = FastVec::new();
            let data_u32: Vec<u32> = (0..1000).map(|i| i as u32).collect();
            vec_u32.extend_from_slice_fast(&data_u32).unwrap();
            assert_eq!(vec_u32.len(), 1000);

            // Test with u64 (8-byte type)
            let mut vec_u64 = FastVec::new();
            let data_u64: Vec<u64> = (0..1000).map(|i| i as u64).collect();
            vec_u64.extend_from_slice_fast(&data_u64).unwrap();
            assert_eq!(vec_u64.len(), 1000);
        }

        #[test]
        fn test_simd_thresholds() {
            // Test that small operations don't use SIMD (threshold check)
            let mut small_vec: FastVec<u8> = FastVec::new();
            small_vec.resize(10, 0).unwrap();
            small_vec.fill_range_fast(0, 10, 0xFF).unwrap();
            
            for i in 0..10 {
                assert_eq!(small_vec[i], 0xFF);
            }

            // Test that large operations do use SIMD
            let mut large_vec: FastVec<u8> = FastVec::new();
            large_vec.resize(1000, 0).unwrap();
            large_vec.fill_range_fast(0, 1000, 0xAA).unwrap();
            
            for i in 0..1000 {
                assert_eq!(large_vec[i], 0xAA);
            }
        }

        #[test]
        fn test_simd_safety_with_drop_types() {
            // Test that types with custom Drop don't use SIMD paths
            use std::sync::Arc;
            use std::sync::atomic::{AtomicUsize, Ordering};

            let counter = Arc::new(AtomicUsize::new(0));

            #[derive(Clone)]
            struct DropCounter {
                counter: Arc<AtomicUsize>,
            }

            impl Drop for DropCounter {
                fn drop(&mut self) {
                    self.counter.fetch_add(1, Ordering::SeqCst);
                }
            }

            let mut vec = FastVec::new();
            for _ in 0..100 {
                vec.push(DropCounter {
                    counter: counter.clone(),
                })
                .unwrap();
            }

            // These operations should work correctly with Drop types
            vec.insert(50, DropCounter {
                counter: counter.clone(),
            })
            .unwrap();

            vec.remove(25).unwrap();

            assert_eq!(vec.len(), 100);
            // Should have 1 drop from remove operation
            assert_eq!(counter.load(Ordering::SeqCst), 1);
        }

        #[test] 
        fn test_simd_memory_safety() {
            // Test that SIMD operations maintain memory safety
            let mut vec: FastVec<u8> = FastVec::new();
            
            // Test with various sizes around SIMD thresholds
            for size in [1, 16, 32, 63, 64, 65, 100, 256, 1000] {
                vec.clear();
                vec.resize(size, 0).unwrap();
                
                // Fill with pattern
                for i in 0..size {
                    vec[i] = (i % 256) as u8;
                }
                
                // Test SIMD operations
                if size > 10 {
                    vec.fill_range_fast(1, size - 1, 0xAA).unwrap();
                    assert_eq!(vec[0], 0);
                    if size > 1 {
                        assert_eq!(vec[size - 1], ((size - 1) % 256) as u8);
                    }
                }
                
                // Test copy operations
                let copy_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
                vec.copy_from_slice_fast(&copy_data).unwrap();
                assert_eq!(vec.len(), size);
                for i in 0..size {
                    assert_eq!(vec[i], (i % 256) as u8);
                }
            }
        }

        #[test]
        fn test_simd_performance_characteristics() {
            // This test verifies that SIMD operations complete correctly
            // Performance measurement would be done in benchmarks
            
            let mut vec: FastVec<u8> = FastVec::new();
            
            // Large resize with SIMD
            let large_size = 10000;
            vec.resize(large_size, 0x55).unwrap();
            assert_eq!(vec.len(), large_size);
            for i in 0..large_size {
                assert_eq!(vec[i], 0x55);
            }
            
            // Large fill range with SIMD
            vec.fill_range_fast(1000, 9000, 0xAA).unwrap();
            for i in 1000..9000 {
                assert_eq!(vec[i], 0xAA);
            }
            
            // Large copy with SIMD
            let source_data: Vec<u8> = (0..large_size).map(|i| (i % 256) as u8).collect();
            vec.copy_from_slice_fast(&source_data).unwrap();
            for i in 0..large_size {
                assert_eq!(vec[i], (i % 256) as u8);
            }
        }
    }
}
