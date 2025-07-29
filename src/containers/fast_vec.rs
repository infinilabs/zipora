//! FastVec: High-performance vector using realloc for growth
//!
//! This is a direct port of the C++ `valvec` with Rust safety guarantees.
//! Unlike std::Vec which uses malloc+memcpy for growth, FastVec uses realloc
//! which can often avoid copying when the allocator can expand in place.

use crate::error::{Result, ToplingError};
use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};
use std::mem;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice;
use std::fmt;

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
/// use infini_zip::FastVec;
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
            .map_err(|_| ToplingError::out_of_memory(cap * mem::size_of::<T>()))?;
        
        let ptr = unsafe { alloc::alloc(layout) as *mut T };
        if ptr.is_null() {
            return Err(ToplingError::out_of_memory(layout.size()));
        }

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
        let required = self.len.checked_add(additional)
            .ok_or_else(|| ToplingError::out_of_memory(usize::MAX))?;
        
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
            .map_err(|_| ToplingError::out_of_memory(target_cap * mem::size_of::<T>()))?;

        let new_ptr = match self.ptr {
            Some(ptr) => {
                if self.cap == 0 {
                    // This shouldn't happen, but handle it safely
                    unsafe { alloc::alloc(new_layout) as *mut T }
                } else {
                    let old_layout = Layout::array::<T>(self.cap).unwrap();
                    unsafe { 
                        alloc::realloc(
                            ptr.as_ptr() as *mut u8, 
                            old_layout, 
                            new_layout.size()
                        ) as *mut T 
                    }
                }
            }
            None => unsafe { alloc::alloc(new_layout) as *mut T }
        };

        if new_ptr.is_null() {
            return Err(ToplingError::out_of_memory(new_layout.size()));
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
            return Err(ToplingError::out_of_bounds(index, self.len));
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
            return Err(ToplingError::out_of_bounds(index, self.len));
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
            .map_err(|_| ToplingError::out_of_memory(self.len * mem::size_of::<T>()))?;

        let new_ptr = if let Some(ptr) = self.ptr {
            let old_layout = Layout::array::<T>(self.cap).unwrap();
            unsafe { 
                alloc::realloc(
                    ptr.as_ptr() as *mut u8, 
                    old_layout, 
                    new_layout.size()
                ) as *mut T 
            }
        } else {
            return Ok(()); // Nothing to shrink
        };

        if new_ptr.is_null() {
            return Err(ToplingError::out_of_memory(new_layout.size()));
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
                vec.push(DropCounter { counter: counter.clone() }).unwrap();
            }
            
            // Remove one element
            vec.remove(2).unwrap();
            assert_eq!(counter.load(Ordering::SeqCst), 1);
            
            // Resize down - this will drop the extra elements beyond index 2
            let resize_value = DropCounter { counter: counter.clone() };
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
}