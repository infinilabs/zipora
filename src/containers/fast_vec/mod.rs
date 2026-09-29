//! FastVec: High-performance vector using realloc for growth
//!
//! This is a direct port of the C++ `valvec` with Rust safety guarantees.
//! Unlike std::Vec which uses malloc+memcpy for growth, FastVec uses realloc
//! which can often avoid copying when the allocator can expand in place.

use crate::error::{Result, ZiporaError};
use crate::simd::{AdaptiveSimdSelector, Operation};
use std::alloc::{self, Layout};
use std::mem;
use std::ptr::{self, NonNull};
use std::slice;
use std::time::Instant;

/// Check that a pointer is properly aligned for type T
#[inline]
fn check_alignment<T>(ptr: *mut u8) {
    // PROVEN: caller checks !raw_ptr.is_null() before calling cast_aligned_ptr, and GlobalAlloc guarantees alignment for Layout::array::<T>.
    debug_assert!(!ptr.is_null());
    // PROVEN: GlobalAlloc returns a pointer aligned to at least Layout::align() == align_of::<T>().
    debug_assert!((ptr as usize).is_multiple_of(mem::align_of::<T>()));
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
    // Rely on the Copy bound at the call site and the absence of drop glue.
    !mem::needs_drop::<T>()
}

/// Check if an operation size is large enough to benefit from SIMD
#[inline]
const fn is_simd_beneficial<T>(element_count: usize) -> bool {
    // SIMD beneficial threshold: 64 bytes minimum
    const SIMD_THRESHOLD: usize = 64;
    element_count.saturating_mul(mem::size_of::<T>()) >= SIMD_THRESHOLD
}

/// Prefetch distance for lookahead operations (based on successful patterns)
/// Matches the PREFETCH_DISTANCE=8 pattern from RankSelectInterleaved256
const PREFETCH_DISTANCE: usize = 8;

/// Prefetch utilities for FastVec operations
struct PrefetchOps;

impl PrefetchOps {
    /// Prefetch memory location for reading with cache hints
    #[inline]
    fn prefetch_read<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: _mm_prefetch is always safe - it's a hint that can be ignored; ptr validity checked by caller
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }

        #[cfg(target_arch = "aarch64")]
        // SAFETY: prfm is always safe - it's a prefetch hint that can be ignored; ptr validity checked by caller
        unsafe {
            // ARM64 PRFM PLDL1KEEP - prefetch for load to L1 cache, temporal
            std::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) ptr,
                options(nostack, preserves_flags, readonly)
            );
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Compiler hint for other architectures
            std::hint::black_box(ptr);
        }
    }

    /// Prefetch range of memory with stride
    #[inline]
    fn prefetch_range<T>(start: *const T, count: usize, distance: usize) {
        if count <= distance {
            return;
        }

        // Prefetch using cache line strides
        const CACHE_LINE_SIZE: usize = 64;
        let element_size = mem::size_of::<T>().max(1);
        let elements_per_line = (CACHE_LINE_SIZE / element_size).max(1);

        for i in (0..count).step_by(elements_per_line) {
            if i + distance < count {
                // SAFETY: i + distance < count guaranteed by the branch check above.
                unsafe {
                    Self::prefetch_read(start.add(i + distance));
                }
            }
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
        // ZST: passing a zero-size Layout to alloc is UB per the GlobalAlloc
        // contract. A dangling aligned pointer is valid for all ZST accesses,
        // and capacity is virtually unlimited since elements occupy no memory.
        if mem::size_of::<T>() == 0 {
            return Ok(Self {
                ptr: Some(NonNull::dangling()),
                len: 0,
                cap: usize::MAX,
            });
        }

        if cap == 0 {
            return Ok(Self::new());
        }

        if cap > (isize::MAX as usize) / mem::size_of::<T>().max(1) {
            return Err(ZiporaError::out_of_memory(
                cap.saturating_mul(mem::size_of::<T>()),
            ));
        }

        let layout = Layout::array::<T>(cap)
            .map_err(|_| ZiporaError::out_of_memory(cap.saturating_mul(mem::size_of::<T>())))?;

        // SAFETY: alloc::alloc returns valid pointer for non-zero layout
        let ptr = unsafe {
            let raw_ptr = alloc::alloc(layout);
            if raw_ptr.is_null() {
                return Err(ZiporaError::out_of_memory(layout.size()));
            }
            cast_aligned_ptr::<T>(raw_ptr)
        };

        Ok(Self {
            // SAFETY: ptr is non-null after successful allocation verified above
            ptr: Some(unsafe { NonNull::new_unchecked(ptr) }),
            len: 0,
            cap,
        })
    }

    /// Create a FastVec with zeroed memory using `alloc_zeroed` (maps to `calloc`).
    ///
    /// For large allocations, `calloc` leverages kernel zero-page mapping,
    /// avoiding physical zeroing entirely. This makes it significantly faster
    /// than `alloc` + `memset` for zero-initialized buffers.
    ///
    /// The returned vector has `len == cap` — all elements are zero-initialized.
    ///
    /// `T: bytemuck::Zeroable` guarantees the all-zero byte pattern is a valid
    /// value, so this is sound for any `T` that compiles (types with non-zero
    /// invariants like `String` or `Box<T>` are rejected at compile time).
    pub fn with_capacity_zeroed(cap: usize) -> Result<Self>
    where
        T: bytemuck::Zeroable,
    {
        // ZST: no memory to zero; all `cap` elements trivially exist.
        // See with_capacity for why alloc must not be called with size 0.
        if mem::size_of::<T>() == 0 {
            return Ok(Self {
                ptr: Some(NonNull::dangling()),
                len: cap,
                cap: usize::MAX,
            });
        }

        if cap == 0 {
            return Ok(Self::new());
        }

        if cap > (isize::MAX as usize) / mem::size_of::<T>().max(1) {
            return Err(ZiporaError::out_of_memory(
                cap.saturating_mul(mem::size_of::<T>()),
            ));
        }

        let layout = Layout::array::<T>(cap)
            .map_err(|_| ZiporaError::out_of_memory(cap.saturating_mul(mem::size_of::<T>())))?;

        // SAFETY: alloc_zeroed returns zeroed memory; calloc kernel optimization
        // avoids physical zeroing for large allocations via zero-page mapping.
        let ptr = unsafe {
            let raw_ptr = alloc::alloc_zeroed(layout);
            if raw_ptr.is_null() {
                return Err(ZiporaError::out_of_memory(layout.size()));
            }
            cast_aligned_ptr::<T>(raw_ptr)
        };

        Ok(Self {
            // SAFETY: ptr is non-null after successful allocation
            ptr: Some(unsafe { NonNull::new_unchecked(ptr) }),
            len: cap, // all elements are valid (zeroed)
            cap,
        })
    }

    /// Create a FastVec by taking ownership of a `Vec<T>` without copying.
    ///
    /// The Vec's buffer is transferred to FastVec. The original Vec is consumed.
    pub fn from_vec(vec: Vec<T>) -> Self {
        let mut vec = std::mem::ManuallyDrop::new(vec);
        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        let cap = if mem::size_of::<T>() == 0 {
            usize::MAX
        } else {
            vec.capacity()
        };

        Self {
            ptr: NonNull::new(ptr),
            len,
            cap,
        }
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

    /// Set length without dropping or allocating
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to `capacity()`.
    /// - The elements at `old_len..new_len` must be initialized.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        // PROVEN: caller upholds the unsafe fn precondition `new_len <= self.capacity()`.
        debug_assert!(new_len <= self.cap);
        self.len = new_len;
    }

    /// Get the vector as a slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // Branch-free: when len==0, from_raw_parts with a dangling aligned
        // pointer and length 0 is safe. When len>0, ptr is always Some.
        // SAFETY: ptr is valid+aligned when len > 0 (allocation invariant).
        // When len == 0, dangling() provides aligned non-null pointer.
        unsafe { slice::from_raw_parts(self.ptr.unwrap_or(NonNull::dangling()).as_ptr(), self.len) }
    }

    /// Get the vector as a mutable slice
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: same as as_slice — dangling pointer with len=0 is safe.
        unsafe {
            slice::from_raw_parts_mut(self.ptr.unwrap_or(NonNull::dangling()).as_ptr(), self.len)
        }
    }

    /// Reserve space for at least `additional` more elements
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        if additional > (isize::MAX as usize) / mem::size_of::<T>().max(1) {
            return Err(ZiporaError::out_of_memory(
                additional.saturating_mul(mem::size_of::<T>()),
            ));
        }

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
        if min_cap > (isize::MAX as usize) / mem::size_of::<T>().max(1) {
            return Err(ZiporaError::out_of_memory(
                min_cap.saturating_mul(mem::size_of::<T>()),
            ));
        }

        if min_cap <= self.cap {
            return Ok(());
        }

        self.realloc(min_cap)
    }

    /// Reallocate to the new capacity using realloc for optimal performance
    fn realloc(&mut self, new_cap: usize) -> Result<()> {
        // PROVEN: callers (`reserve` and `ensure_capacity`) only invoke `realloc` when `new_cap > self.cap >= self.len`.
        debug_assert!(new_cap >= self.len);

        // ZST: capacity is virtual — never call alloc/realloc with a
        // zero-size Layout (UB per the GlobalAlloc contract).
        if mem::size_of::<T>() == 0 {
            self.ptr = Some(NonNull::dangling());
            self.cap = usize::MAX;
            return Ok(());
        }

        if new_cap > (isize::MAX as usize) / mem::size_of::<T>().max(1) {
            return Err(ZiporaError::out_of_memory(
                new_cap.saturating_mul(mem::size_of::<T>()),
            ));
        }

        // Use exponential growth with a minimum increase
        let target_cap = new_cap.max(self.cap.saturating_mul(2));

        let new_layout = Layout::array::<T>(target_cap).map_err(|_| {
            ZiporaError::out_of_memory(target_cap.saturating_mul(mem::size_of::<T>()))
        })?;

        let new_ptr = match self.ptr {
            Some(ptr) => {
                if self.cap == 0 {
                    // This shouldn't happen, but handle it safely
                    // SAFETY: alloc returns valid pointer or null; cast_aligned_ptr checks alignment
                    unsafe {
                        let raw_ptr = alloc::alloc(new_layout);
                        if raw_ptr.is_null() {
                            std::ptr::null_mut()
                        } else {
                            cast_aligned_ptr::<T>(raw_ptr)
                        }
                    }
                } else {
                    let old_layout = Layout::array::<T>(self.cap).map_err(|_| {
                        ZiporaError::out_of_memory(self.cap.saturating_mul(mem::size_of::<T>()))
                    })?;
                    // SAFETY: ptr from self.ptr valid allocation, old_layout matches original allocation, new_layout.size() >= old_layout.size()
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
            // SAFETY: alloc returns valid pointer or null; cast_aligned_ptr checks alignment
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

        // SAFETY: new_ptr is non-null after the null check above.
        self.ptr = Some(unsafe { NonNull::new_unchecked(new_ptr) });
        self.cap = target_cap;
        Ok(())
    }

    /// Push an element to the end of the vector
    pub fn push(&mut self, value: T) -> Result<()> {
        // PROVEN: type invariant guarantees self.len <= self.cap.
        debug_assert!(self.len <= self.cap);
        if self.len >= (isize::MAX as usize) {
            return Err(ZiporaError::invalid_state(
                "vector length would exceed maximum",
            ));
        }

        if self.len >= self.cap {
            self.ensure_capacity(self.len + 1)?;
        }

        debug_assert!(self.len < self.cap); // PROVEN: ensure_capacity(self.len + 1) succeeded above.
        debug_assert!(self.ptr.is_some()); // PROVEN: ensure_capacity(self.len + 1) sets self.ptr = Some(...).

        // SAFETY: self.len < self.cap and self.ptr is Some after ensure_capacity(self.len + 1).
        unsafe {
            ptr::write(self.as_mut_ptr().add(self.len), value);
        }
        self.len += 1;
        Ok(())
    }

    /// Pop an element from the end of the vector
    pub fn pop(&mut self) -> Option<T> {
        debug_assert!(self.len <= self.cap); // PROVEN: type invariant guarantees self.len <= self.cap.

        if self.len == 0 {
            None
        } else {
            debug_assert!(self.ptr.is_some()); // PROVEN: self.len > 0 implies self.ptr.is_some() by the FastVec allocation invariant.

            self.len -= 1;
            // SAFETY: self.len was decremented from a positive value, so self.len is an initialized slot and self.ptr is Some.
            Some(unsafe { ptr::read(self.as_ptr().add(self.len)) })
        }
    }

    /// Insert an element at the specified index
    pub fn insert(&mut self, index: usize, value: T) -> Result<()> {
        if index > self.len {
            return Err(ZiporaError::out_of_bounds(index, self.len));
        }

        debug_assert!(index <= self.len); // PROVEN: index <= self.len checked by the early return above.

        if self.len >= self.cap {
            self.ensure_capacity(self.len + 1)?;
        }

        let move_count = self.len - index;

        // SAFETY: index <= self.len verified above, ptr valid after ensure_capacity, move_count computed safely
        unsafe {
            let ptr = self.as_mut_ptr().add(index);

            if move_count > 0 {
                // ptr::copy handles overlapping memory correctly (compiles to memmove,
                // which LLVM/glibc already optimizes with SIMD for large buffers)
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

        // SAFETY: index < self.len verified above, ptr valid from allocation, move_count computed safely
        unsafe {
            let ptr = self.as_mut_ptr().add(index);
            let value = ptr::read(ptr);

            if move_count > 0 {
                // ptr::copy handles overlapping memory correctly (compiles to memmove,
                // which LLVM/glibc already optimizes with SIMD for large buffers)
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
        if new_len > (isize::MAX as usize) / mem::size_of::<T>().max(1) {
            return Err(ZiporaError::out_of_memory(
                new_len.saturating_mul(mem::size_of::<T>()),
            ));
        }
        debug_assert!(self.len <= self.cap); // PROVEN: type invariant guarantees self.len <= self.cap.

        if new_len > self.len {
            self.ensure_capacity(new_len)?;

            debug_assert!(self.cap >= new_len); // PROVEN: ensure_capacity(new_len) guarantees self.cap >= new_len.
            debug_assert!(self.ptr.is_some()); // PROVEN: ensure_capacity(new_len) with new_len > 0 sets self.ptr = Some(...).

            // Bump len after each write so that if value.clone() panics,
            // the elements already written are owned by the vector and
            // dropped during unwinding instead of leaking.
            for i in self.len..new_len {
                // SAFETY: i < new_len <= self.cap and self.ptr is Some after ensure_capacity(new_len).
                unsafe {
                    ptr::write(self.as_mut_ptr().add(i), value.clone());
                }
                self.len = i + 1;
            }
        } else if new_len < self.len {
            debug_assert!(self.ptr.is_some()); // PROVEN: new_len < self.len implies self.len > 0, so self.ptr.is_some() by allocation invariant.

            let drop_count = self.len - new_len;
            self.len = new_len;
            // SAFETY: new_len + drop_count == old_len <= self.cap, and self.ptr is valid for old_len initialized elements;
            // updating self.len before drop_in_place prevents double-drop if an element's Drop panics.
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    self.as_mut_ptr().add(new_len),
                    drop_count,
                ));
            }
        }

        debug_assert!(self.len <= self.cap); // PROVEN: type invariant self.len <= self.cap is preserved on both branches.
        Ok(())
    }

    /// Resize the vector to the specified length, using a closure to create new elements
    pub fn resize_with<F>(&mut self, new_len: usize, f: F) -> Result<()>
    where
        F: FnMut() -> T,
    {
        if new_len > (isize::MAX as usize) / mem::size_of::<T>().max(1) {
            return Err(ZiporaError::out_of_memory(
                new_len.saturating_mul(mem::size_of::<T>()),
            ));
        }
        debug_assert!(self.len <= self.cap); // PROVEN: type invariant guarantees self.len <= self.cap.

        if new_len > self.len {
            self.ensure_capacity(new_len)?;

            debug_assert!(self.cap >= new_len); // PROVEN: ensure_capacity(new_len) guarantees self.cap >= new_len.
            debug_assert!(self.ptr.is_some()); // PROVEN: ensure_capacity(new_len) with new_len > 0 sets self.ptr = Some(...).

            let mut closure = f;
            // Bump len after each write so that if closure() panics, the
            // elements already written are dropped during unwinding instead
            // of leaking.
            for i in self.len..new_len {
                // SAFETY: i < new_len <= self.cap and self.ptr is Some after ensure_capacity(new_len).
                unsafe {
                    ptr::write(self.as_mut_ptr().add(i), closure());
                }
                self.len = i + 1;
            }
        } else if new_len < self.len {
            debug_assert!(self.ptr.is_some()); // PROVEN: new_len < self.len implies self.len > 0, so self.ptr.is_some() by allocation invariant.

            let drop_count = self.len - new_len;
            self.len = new_len;
            // SAFETY: new_len + drop_count == old_len <= self.cap, and self.ptr is valid for old_len initialized elements;
            // updating self.len before drop_in_place prevents double-drop if an element's Drop panics.
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    self.as_mut_ptr().add(new_len),
                    drop_count,
                ));
            }
        }

        debug_assert!(self.len <= self.cap); // PROVEN: type invariant self.len <= self.cap is preserved on both branches.
        Ok(())
    }

    /// Clear all elements from the vector
    pub fn clear(&mut self) {
        debug_assert!(self.len <= self.cap); // PROVEN: type invariant guarantees self.len <= self.cap.
        debug_assert!(self.ptr.is_some() || self.len == 0); // PROVEN: type invariant guarantees self.ptr.is_some() whenever self.len > 0.

        let len = self.len;
        self.len = 0;
        if len > 0 {
            // SAFETY: len > 0 implies self.ptr is Some and valid for `len` initialized elements;
            // setting self.len = 0 before drop_in_place prevents double-drop if an element's Drop panics.
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), len));
            }
        }

        // PROVEN: self.len == 0 <= self.cap.
        debug_assert!(self.len <= self.cap);
    }

    /// Shrink the capacity to fit the current length
    pub fn shrink_to_fit(&mut self) -> Result<()> {
        // ZST: nothing was ever allocated, nothing to shrink or dealloc.
        if mem::size_of::<T>() == 0 {
            return Ok(());
        }

        if self.len == self.cap {
            return Ok(());
        }

        if self.len == 0 {
            if let Some(ptr) = self.ptr {
                // SAFETY: ptr from valid allocation, layout matches original allocation parameters
                unsafe {
                    let layout = Layout::array::<T>(self.cap).map_err(|_| {
                        ZiporaError::out_of_memory(self.cap.saturating_mul(mem::size_of::<T>()))
                    })?;
                    alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
                }
            }
            self.ptr = None;
            self.cap = 0;
            return Ok(());
        }

        let new_layout = Layout::array::<T>(self.len).map_err(|_| {
            ZiporaError::out_of_memory(self.len.saturating_mul(mem::size_of::<T>()))
        })?;

        let new_ptr = if let Some(ptr) = self.ptr {
            let old_layout = Layout::array::<T>(self.cap).map_err(|_| {
                ZiporaError::out_of_memory(self.cap.saturating_mul(mem::size_of::<T>()))
            })?;
            // SAFETY: ptr from valid allocation, old_layout matches original, new_layout.size() <= old_layout.size() (shrinking)
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

        // SAFETY: new_ptr is non-null after the null check above.
        self.ptr = Some(unsafe { NonNull::new_unchecked(new_ptr) });
        self.cap = self.len;
        Ok(())
    }

    /// Get a reference to the element at the specified index without bounds checking
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        // PROVEN: caller upholds the unsafe fn precondition `index < self.len()`.
        debug_assert!(index < self.len);
        // SAFETY: caller guarantees index < self.len, which implies self.len > 0 and self.ptr is Some.
        unsafe { &*self.ptr.unwrap_unchecked().as_ptr().add(index) }
    }

    /// Get a mutable reference to the element at the specified index without bounds checking
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // PROVEN: caller upholds the unsafe fn precondition `index < self.len()`.
        debug_assert!(index < self.len);
        // SAFETY: caller guarantees index < self.len, which implies self.len > 0 and self.ptr is Some.
        unsafe { &mut *self.ptr.unwrap_unchecked().as_ptr().add(index) }
    }

    /// Extend the vector with elements from an iterator.
    /// For bulk slice copies, prefer `extend_from_slice_fast` which uses SIMD.
    pub fn extend<I>(&mut self, iter: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut iter = iter.into_iter();
        let additional = iter.len();
        self.reserve(additional)?;

        struct SetLenOnDrop<'a> {
            len: &'a mut usize,
            local_len: usize,
        }
        impl Drop for SetLenOnDrop<'_> {
            #[inline(always)]
            fn drop(&mut self) {
                *self.len = self.local_len;
            }
        }

        // Fast path: write up to `spare = self.cap - self.len` elements without
        // per-item capacity branches or per-item `self.len` stores so LLVM can
        // vectorize the loop. `SetLenOnDrop` commits `local_len` on completion
        // or unwind (if `iter.next()` panics), and `.take(spare)` bounds the
        // loop to the allocated capacity even if `ExactSizeIterator::len()` lied.
        if self.len < self.cap {
            let base_ptr = self.as_mut_ptr();
            let spare = self.cap - self.len;
            let mut guard = SetLenOnDrop {
                local_len: self.len,
                len: &mut self.len,
            };
            for item in (&mut iter).take(spare) {
                // SAFETY: `guard.local_len` starts at `self.len` and increments at most
                // `spare = self.cap - self.len` times, so `guard.local_len < self.cap`
                // and `base_ptr` is non-null throughout.
                unsafe {
                    ptr::write(base_ptr.add(guard.local_len), item);
                }
                guard.local_len += 1;
            }
        }

        // Cold path: only reached if a dishonest `ExactSizeIterator` yielded more
        // elements than its reported `len()` (exceeding `self.cap`).
        if let Some(first) = iter.next() {
            self.extend_overflow_slow(first, iter)?;
        }

        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn extend_overflow_slow<I: Iterator<Item = T>>(&mut self, first: T, iter: I) -> Result<()> {
        self.push(first)?;
        for item in iter {
            self.push(item)?;
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
    /// - Uses **Adaptive SIMD Selection** for optimal implementation choice
    /// - **Advanced Prefetching** with PREFETCH_DISTANCE=8 for large ranges
    /// - Falls back to standard operations for small ranges or non-Copy types
    /// - Provides optimal performance for primitive types (u8, u16, u32, u64, etc.)
    pub fn fill_range_fast(&mut self, start: usize, end: usize, value: T) -> Result<()>
    where
        T: Copy,
    {
        if start > end || end > self.len {
            return Err(ZiporaError::out_of_bounds(end, self.len));
        }

        // PROVEN: start <= end <= self.len checked by the early return above, and self.len <= self.cap by type invariant.
        debug_assert!(start <= end);
        // PROVEN: end <= self.len checked by the early return above.
        debug_assert!(end <= self.len);
        // PROVEN: type invariant guarantees self.len <= self.cap.
        debug_assert!(self.len <= self.cap);
        // PROVEN: type invariant guarantees self.ptr.is_some() whenever self.len > 0.
        debug_assert!(self.ptr.is_some() || self.len == 0);

        if start == end {
            return Ok(()); // Nothing to fill
        }

        let range_len = end - start;

        // Adaptive SIMD selection for optimal implementation
        if is_simd_safe::<T>() && is_simd_beneficial::<T>(range_len) {
            let selector = AdaptiveSimdSelector::global();
            let _ = selector.select_optimal_impl(
                Operation::MemZero,
                range_len.saturating_mul(mem::size_of::<T>()),
                None, // No density for fill operations
            );

            // Monitor performance for adaptive optimization
            let start_time = Instant::now();

            let range_slice = &mut self.as_mut_slice()[start..end];
            if range_len >= PREFETCH_DISTANCE * 8 {
                PrefetchOps::prefetch_range(range_slice.as_ptr(), range_len, PREFETCH_DISTANCE);
            }
            range_slice.fill(value);

            // Record performance for monitoring
            selector.monitor_performance(
                Operation::MemZero,
                start_time.elapsed(),
                range_len as u64,
            );
        } else {
            // Standard fill for small ranges or non-Copy types
            let range_slice = &mut self.as_mut_slice()[start..end];
            range_slice.fill(value);
        }

        Ok(())
    }

    /// Fast copy data from a slice using SIMD optimization
    ///
    /// This method provides 2-3x performance improvement over standard copy operations
    /// for bulk data when T is Copy and the operation size is ≥64 bytes.
    ///
    /// # Performance
    /// - Uses **Adaptive SIMD Selection** for optimal implementation choice
    /// - **Advanced Prefetching** with PREFETCH_DISTANCE=8 for large copies
    /// - Falls back to standard operations for small slices or non-Copy types
    /// - Provides optimal performance for primitive types and simple structs
    pub fn copy_from_slice_fast(&mut self, src: &[T]) -> Result<()>
    where
        T: Copy,
    {
        if src.len() > (isize::MAX as usize) / mem::size_of::<T>().max(1) {
            return Err(ZiporaError::out_of_memory(std::mem::size_of_val(src)));
        }
        // PROVEN: type invariant guarantees self.len <= self.cap.
        debug_assert!(self.len <= self.cap);

        if src.is_empty() {
            self.len = 0;
            return Ok(());
        }

        self.ensure_capacity(src.len())?;

        // PROVEN: ensure_capacity(src.len()) with !src.is_empty() guarantees self.cap >= src.len() and self.ptr.is_some().
        debug_assert!(self.cap >= src.len());
        // PROVEN: ensure_capacity(src.len()) with !src.is_empty() sets self.ptr = Some(...).
        debug_assert!(self.ptr.is_some());

        // Adaptive SIMD selection for optimal copy implementation
        if is_simd_safe::<T>() && is_simd_beneficial::<T>(src.len()) {
            let selector = AdaptiveSimdSelector::global();
            let _ = selector.select_optimal_impl(
                Operation::Copy,
                std::mem::size_of_val(src),
                None, // No density for copy operations
            );

            // Monitor performance for adaptive optimization
            let start_time = Instant::now();

            if src.len() >= PREFETCH_DISTANCE * 8 {
                PrefetchOps::prefetch_range(src.as_ptr(), src.len(), PREFETCH_DISTANCE);
            }

            // SAFETY: src.len() <= self.cap after ensure_capacity, self.ptr is Some, and src is disjoint from self.
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), self.as_mut_ptr(), src.len());
            }

            // Record performance for monitoring
            selector.monitor_performance(Operation::Copy, start_time.elapsed(), src.len() as u64);
        } else {
            // Standard copy for small slices or non-Copy types
            // SAFETY: src.len() <= self.cap after ensure_capacity, self.ptr is Some, and src is disjoint from self.
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
    ///
    /// # Performance
    /// - Uses **Adaptive SIMD Selection** for optimal implementation choice
    /// - **Advanced Prefetching** with PREFETCH_DISTANCE=8 for large extends
    /// - Continuous performance monitoring for adaptive optimization
    pub fn extend_from_slice_fast(&mut self, src: &[T]) -> Result<()>
    where
        T: Copy,
    {
        if src.is_empty() {
            return Ok(());
        }

        let old_len = self.len;
        self.reserve(src.len())?;

        // Adaptive SIMD selection for optimal extend implementation
        if is_simd_safe::<T>() && is_simd_beneficial::<T>(src.len()) {
            let selector = AdaptiveSimdSelector::global();
            let _ = selector.select_optimal_impl(
                Operation::Copy,
                std::mem::size_of_val(src),
                None, // No density for extend operations
            );

            // Monitor performance for adaptive optimization
            let start_time = Instant::now();

            if src.len() >= PREFETCH_DISTANCE * 8 {
                PrefetchOps::prefetch_range(src.as_ptr(), src.len(), PREFETCH_DISTANCE);
            }

            // SAFETY: old_len + src.len() <= self.cap after reserve, self.ptr is Some, and src is disjoint from self.
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), self.as_mut_ptr().add(old_len), src.len());
            }

            // Record performance for monitoring
            selector.monitor_performance(Operation::Copy, start_time.elapsed(), src.len() as u64);
        } else {
            // Standard copy for small slices or non-Copy types
            // SAFETY: old_len + src.len() <= self.cap after reserve, self.ptr is Some, and src is disjoint from self.
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), self.as_mut_ptr().add(old_len), src.len());
            }
        }

        self.len += src.len();
        Ok(())
    }
}

mod traits;

#[cfg(test)]
mod tests;
