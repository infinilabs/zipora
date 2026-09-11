//! Standard library trait implementations for `FastVec`
//!
//! `Default`, `Drop`, `Deref`/`DerefMut`, conversions (`From`, `FromIterator`,
//! `Extend`, `IntoIterator` + `FastVecIntoIter`), indexing, comparison,
//! `Clone`, `Debug`, and the `Send`/`Sync` markers.

use super::FastVec;
use std::alloc::{self, Layout};
use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::{self, NonNull};

impl<T> Default for FastVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for FastVec<T> {
    fn drop(&mut self) {
        self.clear();
        // ZST: ptr is dangling and nothing was allocated — dealloc with a
        // zero-size Layout would be UB per the GlobalAlloc contract.
        if let Some(ptr) = self.ptr
            && self.cap > 0
            && std::mem::size_of::<T>() > 0
        {
            // SAFETY: ptr and cap are valid from allocation, layout matches allocation layout
            unsafe {
                let layout = Layout::array::<T>(self.cap).expect(
                    "Layout::array succeeded during allocation, must succeed during deallocation",
                );
                alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
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

impl<T> From<Vec<T>> for FastVec<T> {
    fn from(v: Vec<T>) -> Self {
        Self::from_vec(v)
    }
}

impl<T> FromIterator<T> for FastVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut fv =
            Self::with_capacity(lower).expect("FastVec: allocation failed in FromIterator");
        for item in iter {
            fv.push(item).expect("FastVec: push failed in FromIterator");
        }
        fv
    }
}

impl<T> Extend<T> for FastVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        if lower > 0 {
            let _ = self.reserve(lower);
        }
        for item in iter {
            let _ = self.push(item);
        }
    }
}

impl<T> IntoIterator for FastVec<T> {
    type Item = T;
    type IntoIter = FastVecIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.len;
        let cap = self.cap;
        let ptr = self.ptr;
        std::mem::forget(self);
        FastVecIntoIter {
            ptr,
            cap,
            len,
            index: 0,
        }
    }
}

/// Iterator over the elements of a FastVec
pub struct FastVecIntoIter<T> {
    ptr: Option<NonNull<T>>,
    cap: usize,
    len: usize,
    index: usize,
}

impl<T> Iterator for FastVecIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let ptr = self.ptr.expect("valid ptr for non-empty iter");
            // SAFETY: index < len, ptr valid from FastVec allocation, add(index) within allocated range
            let item = unsafe { ptr::read(ptr.as_ptr().add(self.index)) };
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<T> Drop for FastVecIntoIter<T> {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr {
            // Drop remaining elements
            for i in self.index..self.len {
                // SAFETY: i in [index, len), ptr valid from FastVec allocation, add(i) within allocated range
                unsafe {
                    ptr::drop_in_place(ptr.as_ptr().add(i));
                }
            }
            // Deallocate memory (ZST: nothing was allocated, dealloc with a
            // zero-size Layout would be UB)
            if self.cap > 0 && std::mem::size_of::<T>() > 0 {
                let layout = Layout::array::<T>(self.cap).expect(
                    "Layout::array succeeded during allocation, must succeed during deallocation",
                );
                // SAFETY: ptr allocated with same layout during FastVec creation
                unsafe {
                    alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
                }
            }
        }
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
        // Quick length check first
        if self.len != other.len {
            return false;
        }

        if self.len == 0 {
            return true;
        }

        // Byte equality is not value equality: `T::eq` is the only thing that
        // may decide this. Padding bytes, `-0.0 == 0.0`, `NaN != NaN` and any
        // hand-written `PartialEq` all disagree with memcmp, and the previous
        // SIMD path silently took over past 64 bytes, so the same two vectors
        // compared differently depending on their length.
        //
        // Nothing is lost by deferring: the standard slice comparison already
        // specialises to memcmp for the types where that is sound.
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for FastVec<T> {}

impl<T: Clone> Clone for FastVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec =
            Self::with_capacity(self.len).expect("FastVec: allocation failed in Clone");

        for item in self.as_slice() {
            let _ = new_vec.push(item.clone());
        }
        new_vec
    }
}

// Safety: FastVec<T> is Send if T is Send
unsafe impl<T: Send> Send for FastVec<T> {}

// Safety: FastVec<T> is Sync if T is Sync
unsafe impl<T: Sync> Sync for FastVec<T> {}
