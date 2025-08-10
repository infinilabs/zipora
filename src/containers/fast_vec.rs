//! FastVec: High-performance vector using realloc for growth
//!
//! This is a direct port of the C++ `valvec` with Rust safety guarantees.
//! Unlike std::Vec which uses malloc+memcpy for growth, FastVec uses realloc
//! which can often avoid copying when the allocator can expand in place.

use crate::error::{Result, ZiporaError};
use std::alloc::{self, Layout};
use std::fmt;
use std::mem;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::{self, NonNull};
use std::slice;

/// Check that a pointer is properly aligned for type T
/// This is a debug-only assertion to verify allocator guarantees
#[inline]
fn check_alignment<T>(ptr: *mut u8) {
    debug_assert!(
        !ptr.is_null(),
        "Pointer should not be null when checking alignment"
    );
    debug_assert!(
        (ptr as usize) % mem::align_of::<T>() == 0,
        "Pointer {:#x} is not aligned for type {} (requires {}-byte alignment)",
        ptr as usize,
        std::any::type_name::<T>(),
        mem::align_of::<T>()
    );
}

/// Safely cast an aligned u8 pointer to T pointer with alignment verification
#[inline]
fn cast_aligned_ptr<T>(ptr: *mut u8) -> *mut T {
    check_alignment::<T>(ptr);
    ptr as *mut T
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

        let layout = Layout::array::<T>(cap)
            .map_err(|_| ZiporaError::out_of_memory(cap * mem::size_of::<T>()))?;

        let ptr = unsafe {
            let raw_ptr = alloc::alloc(layout);
            if raw_ptr.is_null() {
                return Err(ZiporaError::out_of_memory(layout.size()));
            }
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
        if min_cap <= self.cap {
            return Ok(());
        }

        self.realloc(min_cap)
    }

    /// Reallocate to the new capacity using realloc for optimal performance
    fn realloc(&mut self, new_cap: usize) -> Result<()> {
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
        if self.len >= self.cap {
            self.ensure_capacity(self.len + 1)?;
        }

        unsafe {
            ptr::write(self.as_mut_ptr().add(self.len), value);
        }
        self.len += 1;
        Ok(())
    }

    /// Pop an element from the end of the vector
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(unsafe { ptr::read(self.as_ptr().add(self.len)) })
        }
    }

    /// Insert an element at the specified index
    pub fn insert(&mut self, index: usize, value: T) -> Result<()> {
        if index > self.len {
            return Err(ZiporaError::out_of_bounds(index, self.len));
        }

        if self.len >= self.cap {
            self.ensure_capacity(self.len + 1)?;
        }

        unsafe {
            let ptr = self.as_mut_ptr().add(index);
            // Move existing elements one position to the right
            ptr::copy(ptr, ptr.add(1), self.len - index);
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

        unsafe {
            let ptr = self.as_mut_ptr().add(index);
            let value = ptr::read(ptr);
            // Move remaining elements one position to the left
            ptr::copy(ptr.add(1), ptr, self.len - index - 1);
            self.len -= 1;
            Ok(value)
        }
    }

    /// Resize the vector to the specified length
    pub fn resize(&mut self, new_len: usize, value: T) -> Result<()>
    where
        T: Clone,
    {
        if new_len > self.len {
            self.ensure_capacity(new_len)?;
            for i in self.len..new_len {
                unsafe {
                    ptr::write(self.as_mut_ptr().add(i), value.clone());
                }
            }
        } else if new_len < self.len {
            // Drop excess elements
            for i in new_len..self.len {
                unsafe {
                    ptr::drop_in_place(self.as_mut_ptr().add(i));
                }
            }
        }
        self.len = new_len;
        Ok(())
    }

    /// Clear all elements from the vector
    pub fn clear(&mut self) {
        for i in 0..self.len {
            unsafe {
                ptr::drop_in_place(self.as_mut_ptr().add(i));
            }
        }
        self.len = 0;
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
        let iter = iter.into_iter();
        let additional = iter.len();
        self.reserve(additional)?;

        for item in iter {
            // We know we have capacity, so this won't fail
            unsafe {
                ptr::write(self.as_mut_ptr().add(self.len), item);
                self.len += 1;
            }
        }
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
        &self.as_slice()[index]
    }
}

impl<T> IndexMut<usize> for FastVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
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
        self.as_slice() == other.as_slice()
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
}
