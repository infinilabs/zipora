//! Main C API interface
//!
//! This module provides the primary C-compatible interface for the zipora library,
//! allowing seamless integration with existing C/C++ codebases.

use super::{CResult, types::*};
use crate::blob_store::traits::BlobStore;
use std::cell::RefCell;
#[cfg(test)]
use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::Mutex;

// Thread-local storage for error messages
thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

// Global error callback storage
static ERROR_CALLBACK: Mutex<Option<unsafe extern "C" fn(*const c_char)>> = Mutex::new(None);

/// Set the last error message for the current thread
fn set_last_error(msg: &str) {
    let cstring = match CString::new(msg) {
        Ok(c) => c,
        Err(_) => CString::new(msg.replace('\0', "\\0")).unwrap_or_default(),
    };

    let _ = LAST_ERROR.try_with(|error| {
        *error.borrow_mut() = Some(cstring.clone());
    });

    // Copy the callback function pointer and drop the mutex guard BEFORE invoking it
    // so a re-entrant callback calling `zipora_set_error_callback` or generating an
    // error cannot deadlock on `ERROR_CALLBACK`.
    let callback = ERROR_CALLBACK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .copied();
    if let Some(cb) = callback {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: callback function pointer was registered by the caller via
            // `zipora_set_error_callback`, and `cstring` outlives this call.
            unsafe {
                cb(cstring.as_ptr());
            }
        }));
    }
}

/// Clear the last error message for the current thread (for testing)
#[cfg(test)]
fn clear_last_error() {
    let _ = LAST_ERROR.try_with(|error| {
        *error.borrow_mut() = None;
    });
}

/// Execute an FFI body, catching any panic so it cannot unwind into the C
/// caller. Unwinding out of an `extern "C"` function aborts the whole process
/// (Rust >= 1.71); this converts a panic into `fallback` plus a last-error
/// message the caller can retrieve via `zipora_last_error`.
///
/// `AssertUnwindSafe` is acceptable here: every entry point re-validates its
/// raw-pointer inputs from scratch, and objects reachable through those
/// pointers are owned by the C caller, so no Rust-side invariant is silently
/// carried across the catch.
fn ffi_guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                s
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.as_str()
            } else {
                "unknown panic payload"
            };
            set_last_error(&format!("panic caught at FFI boundary: {}", msg));
            fallback
        }
    }
}


/// Initialize the zipora library
///
/// # Safety
///
/// This function is safe to call multiple times.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_init() -> CResult {
    ffi_guard(CResult::InternalError, || {
        crate::init();
        CResult::Success
    })
}

/// Get library version string
///
/// # Safety
///
/// The returned null-terminated string has static lifetime and is valid for the
/// lifetime of the program. The caller must not free the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_version() -> *const c_char {
    ffi_guard(ptr::null(), || {
        concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
    })
}

/// Check if SIMD optimizations are available
///
/// # Safety
/// C FFI entry point. Takes no arguments and only reads CPU feature flags, so
/// it is safe to call from any thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_has_simd() -> c_int {
    ffi_guard(0, || {
        if crate::has_simd_support() { 1 } else { 0 }
    })
}

/// Create a new FastVec instance
///
/// # Safety
///
/// The returned pointer must be freed with `fast_vec_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fast_vec_new() -> *mut CFastVec {
    ffi_guard(ptr::null_mut(), || {
        let fast_vec = Box::new(crate::FastVec::<u8>::new());
        Box::into_raw(fast_vec) as *mut CFastVec
    })
}

/// Free a FastVec instance
///
/// # Safety
///
/// The pointer must be a valid CFastVec pointer returned from `fast_vec_new`.
/// The pointer becomes invalid after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fast_vec_free(vec: *mut CFastVec) {
    ffi_guard((), || {
        let raw = vec as *mut crate::FastVec<u8>;
        if !raw.is_null() {
            if !raw.is_aligned() {
                set_last_error("FastVec pointer is misaligned");
                return;
            }
            // SAFETY: pointer is non-null and aligned (checked above), created by fast_vec_new via Box::into_raw, not previously freed
            let _vec = unsafe { Box::from_raw(raw) };
            // Automatic cleanup when Box is dropped
        }
    })
}

/// Push a byte to a FastVec
///
/// # Safety
///
/// The vec pointer must be a valid CFastVec pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fast_vec_push(vec: *mut CFastVec, value: u8) -> CResult {
    ffi_guard(CResult::InternalError, || {
        let raw = vec as *mut crate::FastVec<u8>;
        if raw.is_null() || !raw.is_aligned() {
            set_last_error("FastVec pointer is null or misaligned");
            return CResult::InvalidInput;
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized FastVec created by fast_vec_new
        let fast_vec = unsafe { &mut *raw };
        match fast_vec.push(value) {
            Ok(_) => CResult::Success,
            Err(e) => {
                set_last_error(&format!("Failed to push to FastVec: {}", e));
                CResult::MemoryError
            }
        }
    })
}

/// Get the length of a FastVec
///
/// # Safety
///
/// The vec pointer must be a valid CFastVec pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fast_vec_len(vec: *const CFastVec) -> usize {
    ffi_guard(0, || {
        let raw = vec as *const crate::FastVec<u8>;
        if raw.is_null() || !raw.is_aligned() {
            return 0;
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized FastVec
        let fast_vec = unsafe { &*raw };
        fast_vec.len()
    })
}

/// Get a pointer to the FastVec's data
///
/// # Safety
///
/// The vec pointer must be a valid CFastVec pointer.
/// The returned pointer is valid until the FastVec is modified or freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fast_vec_data(vec: *const CFastVec) -> *const u8 {
    ffi_guard(ptr::null(), || {
        let raw = vec as *const crate::FastVec<u8>;
        if raw.is_null() || !raw.is_aligned() {
            return ptr::null();
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized FastVec
        let fast_vec = unsafe { &*raw };
        fast_vec.as_ptr()
    })
}

/// Create a memory pool
///
/// # Safety
///
/// The returned pointer must be freed with `memory_pool_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_pool_new(chunk_size: usize, max_chunks: usize) -> *mut CMemoryPool {
    ffi_guard(ptr::null_mut(), || {
        let config = crate::memory::PoolConfig::new(chunk_size, max_chunks, 8);
        match crate::memory::MemoryPool::new(config) {
            Ok(pool) => Box::into_raw(Box::new(pool)) as *mut CMemoryPool,
            Err(_) => ptr::null_mut(),
        }
    })
}

/// Free a memory pool
///
/// # Safety
///
/// The pointer must be a valid CMemoryPool pointer returned from `memory_pool_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_pool_free(pool: *mut CMemoryPool) {
    ffi_guard((), || {
        let raw = pool as *mut crate::memory::MemoryPool;
        if !raw.is_null() {
            if !raw.is_aligned() {
                set_last_error("MemoryPool pointer is misaligned");
                return;
            }
            // SAFETY: pointer is non-null and aligned (checked above), created by memory_pool_new via Box::into_raw, not previously freed
            let _pool = unsafe { Box::from_raw(raw) };
            // Automatic cleanup when Box is dropped
        }
    })
}

/// Allocate memory from a pool
///
/// # Safety
///
/// The pool pointer must be a valid CMemoryPool pointer.
/// The returned pointer must be freed with `memory_pool_deallocate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_pool_allocate(pool: *mut CMemoryPool) -> *mut c_void {
    ffi_guard(ptr::null_mut(), || {
        let raw = pool as *const crate::memory::MemoryPool;
        if raw.is_null() || !raw.is_aligned() {
            set_last_error("MemoryPool pointer is null or misaligned");
            return ptr::null_mut();
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized MemoryPool
        let memory_pool = unsafe { &*raw };
        match memory_pool.allocate() {
            Ok(ptr) => ptr.as_ptr() as *mut c_void,
            Err(e) => {
                set_last_error(&format!("Failed to allocate from memory pool: {}", e));
                ptr::null_mut()
            }
        }
    })
}

/// Deallocate memory back to a pool
///
/// # Safety
///
/// Both pointers must be valid, and the memory pointer must have been
/// allocated from the specified pool.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_pool_deallocate(
    pool: *mut CMemoryPool,
    ptr: *mut c_void,
) -> CResult {
    ffi_guard(CResult::InternalError, || {
        let raw = pool as *const crate::memory::MemoryPool;
        if raw.is_null() || !raw.is_aligned() || ptr.is_null() {
            set_last_error("Memory pool or pointer is null or misaligned");
            return CResult::InvalidInput;
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized MemoryPool
        let memory_pool = unsafe { &*raw };
        let non_null_ptr = match std::ptr::NonNull::new(ptr as *mut u8) {
            Some(p) => p,
            None => {
                set_last_error("Failed to create NonNull pointer from raw pointer");
                return CResult::InvalidInput;
            }
        };

        // SAFETY: this function's own contract requires `ptr` to have come
        // from `memory_pool_allocate` on this same pool and not to have been
        // freed yet, which is exactly what `MemoryPool::deallocate` asks for.
        match unsafe { memory_pool.deallocate(non_null_ptr) } {
            Ok(_) => CResult::Success,
            Err(e) => {
                set_last_error(&format!("Failed to deallocate memory: {}", e));
                CResult::from_zipora_error(&e)
            }
        }
    })
}

/// Create a new blob store
///
/// # Safety
///
/// The returned pointer must be freed with `blob_store_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blob_store_new() -> *mut CBlobStore {
    ffi_guard(ptr::null_mut(), || {
        let store = Box::new(crate::blob_store::MemoryBlobStore::new());
        Box::into_raw(store) as *mut CBlobStore
    })
}

/// Free a blob store
///
/// # Safety
///
/// The pointer must be a valid CBlobStore pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blob_store_free(store: *mut CBlobStore) {
    ffi_guard((), || {
        let raw = store as *mut crate::blob_store::MemoryBlobStore;
        if !raw.is_null() {
            if !raw.is_aligned() {
                set_last_error("BlobStore pointer is misaligned");
                return;
            }
            // SAFETY: pointer is non-null and aligned (checked above), created by blob_store_new via Box::into_raw, not previously freed
            let _store = unsafe { Box::from_raw(raw) };
            // Automatic cleanup when Box is dropped
        }
    })
}

/// Put data into a blob store
///
/// # Safety
///
/// The store pointer must be valid, and data must point to a valid buffer of the specified size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blob_store_put(
    store: *mut CBlobStore,
    data: *const u8,
    size: usize,
    record_id: *mut u32,
) -> CResult {
    ffi_guard(CResult::InternalError, || {
        let raw_store = store as *mut crate::blob_store::MemoryBlobStore;
        if raw_store.is_null()
            || !raw_store.is_aligned()
            || record_id.is_null()
            || !record_id.is_aligned()
            || (data.is_null() && size > 0)
            || size > isize::MAX as usize
        {
            set_last_error("Blob store, data, or record_id parameter is null, misaligned, or exceeds isize::MAX");
            return CResult::InvalidInput;
        }

        // SAFETY: raw_store is non-null and aligned (checked above), points to initialized MemoryBlobStore
        let blob_store = unsafe { &mut *raw_store };
        let data_slice = if size == 0 {
            &[][..]
        } else {
            // SAFETY: data pointer is non-null and size is in 1..=isize::MAX (checked above)
            unsafe { std::slice::from_raw_parts(data, size) }
        };

        match blob_store.put(data_slice) {
            Ok(id) => {
                // SAFETY: record_id pointer is non-null and aligned (checked above), points to valid u32
                unsafe {
                    *record_id = id;
                }
                CResult::Success
            }
            Err(e) => {
                set_last_error(&format!("Failed to put data in blob store: {}", e));
                CResult::from_zipora_error(&e)
            }
        }
    })
}

/// Get data from a blob store
///
/// # Safety
///
/// The store pointer must be valid. The data and size pointers will be set to
/// point to allocated memory that must be freed with `zipora_free_blob_data`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blob_store_get(
    store: *const CBlobStore,
    record_id: u32,
    data: *mut *const u8,
    size: *mut usize,
) -> CResult {
    ffi_guard(CResult::InternalError, || {
        let raw_store = store as *const crate::blob_store::MemoryBlobStore;
        if raw_store.is_null()
            || !raw_store.is_aligned()
            || data.is_null()
            || !data.is_aligned()
            || size.is_null()
            || !size.is_aligned()
        {
            set_last_error("Blob store or output parameter is null or misaligned");
            return CResult::InvalidInput;
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized MemoryBlobStore
        let blob_store = unsafe { &*raw_store };

        match blob_store.get(record_id) {
            Ok(blob_data) => {
                let data_len = blob_data.len();
                let data_ptr = if data_len == 0 {
                    std::ptr::NonNull::<u8>::dangling().as_ptr()
                } else {
                    // H14: validate the layout (rejects size > isize::MAX) instead
                    // of from_size_align_unchecked.
                    let layout = match std::alloc::Layout::from_size_align(data_len, 1) {
                        Ok(layout) => layout,
                        Err(_) => {
                            set_last_error("blob size exceeds isize::MAX; cannot allocate");
                            return CResult::MemoryError;
                        }
                    };
                    // SAFETY: layout validated above, size is non-zero
                    unsafe { std::alloc::alloc(layout) }
                };

                if data_len > 0 && data_ptr.is_null() {
                    set_last_error("Failed to allocate memory for retrieved blob data");
                    return CResult::MemoryError;
                }

                if data_len > 0 {
                    // SAFETY: data_ptr is valid, blob_data is valid slice, non-overlapping
                    unsafe {
                        std::ptr::copy_nonoverlapping(blob_data.as_ptr(), data_ptr, data_len);
                    }
                }

                // SAFETY: data and size pointers are non-null and aligned (checked above), point to valid output parameters
                unsafe {
                    *data = data_ptr;
                    *size = data_len;
                }
                CResult::Success
            }
            Err(e) => {
                set_last_error(&format!("Failed to get data from blob store: {}", e));
                CResult::from_zipora_error(&e)
            }
        }
    })
}

/// Create a suffix array from text
///
/// # Safety
///
/// The text pointer must point to a valid buffer of the specified size.
/// The returned pointer must be freed with `suffix_array_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn suffix_array_new(text: *const u8, size: usize) -> *mut CSuffixArray {
    ffi_guard(ptr::null_mut(), || {
        if text.is_null() || size == 0 || size > isize::MAX as usize {
            set_last_error("SuffixArray input text is null, empty, or exceeds isize::MAX");
            return ptr::null_mut();
        }

        // SAFETY: pointer is non-null and size is in 1..=isize::MAX (checked above)
        let text_slice = unsafe { std::slice::from_raw_parts(text, size) };
        match crate::algorithms::SuffixArray::new(text_slice) {
            Ok(sa) => Box::into_raw(Box::new(sa)) as *mut CSuffixArray,
            Err(e) => {
                set_last_error(&format!("Failed to construct suffix array: {}", e));
                ptr::null_mut()
            }
        }
    })
}

/// Free a suffix array
///
/// # Safety
///
/// The pointer must be a valid CSuffixArray pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn suffix_array_free(sa: *mut CSuffixArray) {
    ffi_guard((), || {
        let raw = sa as *mut crate::algorithms::SuffixArray;
        if !raw.is_null() {
            if !raw.is_aligned() {
                set_last_error("SuffixArray pointer is misaligned");
                return;
            }
            // SAFETY: pointer is non-null and aligned (checked above), created by suffix_array_new via Box::into_raw, not previously freed
            let _sa = unsafe { Box::from_raw(raw) };
            // Automatic cleanup when Box is dropped
        }
    })
}

/// Get the length of a suffix array
///
/// # Safety
///
/// The sa pointer must be a valid CSuffixArray pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn suffix_array_len(sa: *const CSuffixArray) -> usize {
    ffi_guard(0, || {
        let raw = sa as *const crate::algorithms::SuffixArray;
        if raw.is_null() || !raw.is_aligned() {
            return 0;
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized SuffixArray
        let suffix_array = unsafe { &*raw };
        suffix_array.text_len()
    })
}

/// Search for a pattern in the suffix array
///
/// # Safety
///
/// All pointers must be valid, and pattern must point to a buffer of the specified size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn suffix_array_search(
    sa: *const CSuffixArray,
    text: *const u8,
    text_size: usize,
    pattern: *const u8,
    pattern_size: usize,
    start: *mut usize,
    count: *mut usize,
) -> CResult {
    ffi_guard(CResult::InternalError, || {
        let raw_sa = sa as *const crate::algorithms::SuffixArray;
        if raw_sa.is_null()
            || !raw_sa.is_aligned()
            || (text.is_null() && text_size > 0)
            || text_size > isize::MAX as usize
            || (pattern.is_null() && pattern_size > 0)
            || pattern_size > isize::MAX as usize
            || start.is_null()
            || !start.is_aligned()
            || count.is_null()
            || !count.is_aligned()
        {
            set_last_error("SuffixArray search parameter is null, misaligned, or exceeds isize::MAX");
            return CResult::InvalidInput;
        }

        // SAFETY: raw_sa is non-null and aligned (checked above), points to initialized SuffixArray
        let suffix_array = unsafe { &*raw_sa };
        let text_slice = if text_size == 0 {
            &[][..]
        } else {
            // SAFETY: text pointer is non-null and text_size is in 1..=isize::MAX (checked above)
            unsafe { std::slice::from_raw_parts(text, text_size) }
        };
        let pattern_slice = if pattern_size == 0 {
            &[][..]
        } else {
            // SAFETY: pattern pointer is non-null and pattern_size is in 1..=isize::MAX (checked above)
            unsafe { std::slice::from_raw_parts(pattern, pattern_size) }
        };

        let (search_start, search_count) = suffix_array.search(text_slice, pattern_slice);
        // SAFETY: start and count pointers are non-null and aligned (checked above), point to valid output parameters
        unsafe {
            *start = search_start;
            *count = search_count;
        }

        CResult::Success
    })
}

/// Sort an array of 32-bit unsigned integers using radix sort
///
/// # Safety
///
/// The data pointer must point to a valid array of the specified size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn radix_sort_u32(data: *mut u32, size: usize) -> CResult {
    ffi_guard(CResult::InternalError, || {
        const MAX_U32_LEN: usize = (isize::MAX as usize) / std::mem::size_of::<u32>();
        if data.is_null() || !data.is_aligned() || size == 0 || size > MAX_U32_LEN {
            set_last_error("radix_sort_u32 data pointer is null, misaligned, empty, or exceeds isize::MAX");
            return CResult::InvalidInput;
        }

        // SAFETY: pointer is non-null and aligned (checked above), size is in 1..=MAX_U32_LEN (checked above)
        let data_slice = unsafe { std::slice::from_raw_parts_mut(data, size) };
        let mut sorter = crate::algorithms::RadixSort::new();

        match sorter.sort_u32(data_slice) {
            Ok(_) => CResult::Success,
            Err(e) => {
                set_last_error(&format!("radix_sort_u32 failed: {}", e));
                CResult::from_zipora_error(&e)
            }
        }
    })
}

/// Get last error message (thread-local)
///
/// # Safety
///
/// The returned string is valid until the next error occurs on this thread.
/// The caller should not free the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_last_error() -> *const c_char {
    static PANIC_MSG: &[u8] = b"panic caught at FFI boundary\0";
    static NO_ERROR_MSG: &[u8] = b"No error information available\0";
    ffi_guard(PANIC_MSG.as_ptr() as *const c_char, || {
        LAST_ERROR
            .try_with(|error| match error.borrow().as_ref() {
                Some(cstring) => cstring.as_ptr(),
                None => NO_ERROR_MSG.as_ptr() as *const c_char,
            })
            .unwrap_or(NO_ERROR_MSG.as_ptr() as *const c_char)
    })
}

/// Set a custom error callback
///
/// # Safety
///
/// The callback function pointer must be valid for the lifetime of the library usage.
/// The callback will be called from the thread that generates the error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_set_error_callback(
    callback: Option<unsafe extern "C" fn(*const c_char)>,
) {
    ffi_guard((), || {
        let mut callback_guard = ERROR_CALLBACK.lock().unwrap_or_else(|e| e.into_inner());
        *callback_guard = callback;
    })
}

/// Free blob data returned by blob_store_get
///
/// # Safety
///
/// - The data pointer must be a pointer returned by `blob_store_get`, and
///   `size` must be the exact size returned alongside it.
/// - This function must be called **exactly once** per pointer returned by
///   `blob_store_get`. Calling it a second time is a double-free; calling it
///   with a pointer not obtained from `blob_store_get` is undefined behavior.
/// - The pointer becomes invalid after this call; any further read or write
///   through it is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_free_blob_data(data: *mut u8, size: usize) {
    ffi_guard((), || {
        if !data.is_null() && size > 0 {
            // H14: a size that fails Layout validation (> isize::MAX) can never
            // correspond to a live allocation from blob_store_get — refuse to
            // dealloc with a bogus layout rather than invoke UB.
            let Ok(layout) = std::alloc::Layout::from_size_align(size, 1) else {
                return;
            };
            // SAFETY: layout validated above; per the contract, (data, size) came
            // from blob_store_get's std::alloc::alloc with the identical layout
            // and has not been freed yet.
            unsafe {
                std::alloc::dealloc(data, layout);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_version_api() {
        // SAFETY: test calls C API functions with valid parameters, CStr::from_ptr receives valid static string pointer
        unsafe {
            let version_ptr = zipora_version();
            assert!(!version_ptr.is_null());

            let version_cstr = CStr::from_ptr(version_ptr);
            let version_str = version_cstr.to_str().unwrap();
            assert!(!version_str.is_empty());
        }
    }

    #[test]
    fn test_fast_vec_api() {
        // SAFETY: test calls C API functions with valid parameters, slice created from valid pointer returned by fast_vec_data
        unsafe {
            let vec = fast_vec_new();
            assert!(!vec.is_null());

            assert_eq!(fast_vec_len(vec), 0);

            assert_eq!(fast_vec_push(vec, 42), CResult::Success);
            assert_eq!(fast_vec_push(vec, 84), CResult::Success);

            assert_eq!(fast_vec_len(vec), 2);

            let data_ptr = fast_vec_data(vec);
            assert!(!data_ptr.is_null());

            let data_slice = std::slice::from_raw_parts(data_ptr, 2);
            assert_eq!(data_slice, &[42, 84]);

            fast_vec_free(vec);
        }
    }

    #[test]
    fn test_memory_pool_api() {
        // SAFETY: test calls C API functions with valid parameters, allocated pointers are deallocated before pool is freed
        unsafe {
            let pool = memory_pool_new(1024, 10);
            assert!(!pool.is_null());

            let ptr1 = memory_pool_allocate(pool);
            assert!(!ptr1.is_null());

            let ptr2 = memory_pool_allocate(pool);
            assert!(!ptr2.is_null());
            assert_ne!(ptr1, ptr2);

            assert_eq!(memory_pool_deallocate(pool, ptr1), CResult::Success);
            assert_eq!(memory_pool_deallocate(pool, ptr2), CResult::Success);

            memory_pool_free(pool);
        }
    }

    #[test]
    fn test_blob_store_zero_len() {
        unsafe {
            let store = blob_store_new();
            assert!(!store.is_null());

            let test_data: [u8; 0] = [];
            let mut record_id: u32 = 0;

            let result = blob_store_put(store, test_data.as_ptr(), 0, &mut record_id);
            assert_eq!(result, CResult::Success);

            let mut out_data: *const u8 = std::ptr::null();
            let mut out_size: usize = 0;

            let result = blob_store_get(store, record_id, &mut out_data, &mut out_size);
            assert_eq!(result, CResult::Success);
            assert_eq!(out_size, 0);
            assert!(!out_data.is_null()); // Dangling ptr but not null

            zipora_free_blob_data(out_data as *mut u8, out_size);
            blob_store_free(store);
        }
    }
    #[test]
    fn test_blob_store_api() {
        // SAFETY: test calls C API functions with valid parameters, slice created from valid pointer, blob data freed before store
        unsafe {
            let store = blob_store_new();
            assert!(!store.is_null());

            let test_data = b"Hello, world!";
            let mut record_id: u32 = 0;

            let result = blob_store_put(store, test_data.as_ptr(), test_data.len(), &mut record_id);
            assert_eq!(result, CResult::Success);

            let mut data_ptr: *const u8 = ptr::null();
            let mut size: usize = 0;

            let result = blob_store_get(store, record_id, &mut data_ptr, &mut size);
            assert_eq!(result, CResult::Success);
            assert!(!data_ptr.is_null());
            assert_eq!(size, test_data.len());

            // Verify the data content
            let retrieved_data = std::slice::from_raw_parts(data_ptr, size);
            assert_eq!(retrieved_data, test_data);

            // Free the blob data
            zipora_free_blob_data(data_ptr as *mut u8, size);

            blob_store_free(store);
        }
    }

    #[test]
    fn test_radix_sort_api() {
        // SAFETY: test calls C API function with valid pointer to mutable array and correct length
        unsafe {
            let mut data = [5u32, 2, 8, 1, 9, 3, 7, 4, 6];
            let result = radix_sort_u32(data.as_mut_ptr(), data.len());

            assert_eq!(result, CResult::Success);
            assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        }
    }

    #[test]
    fn test_init_and_simd() {
        // SAFETY: test calls C API functions with no parameters, both are safe operations
        unsafe {
            assert_eq!(zipora_init(), CResult::Success);

            let has_simd = zipora_has_simd();
            // Should not panic, return 0 or 1
            assert!(has_simd == 0 || has_simd == 1);
        }
    }

    #[test]
    fn test_error_handling() {
        // SAFETY: test calls C API functions, CStr::from_ptr receives valid static/thread-local string pointers
        unsafe {
            // Test initial state - no error
            let error_ptr = zipora_last_error();
            assert!(!error_ptr.is_null());
            let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert_eq!(error_msg, "No error information available");

            // Test setting an error message directly
            set_last_error("Test error message");

            // Check that error message was set
            let error_ptr = zipora_last_error();
            assert!(!error_ptr.is_null());
            let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert_eq!(error_msg, "Test error message");
        }
    }

    static CALLBACK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_error_callback() {
        let _callback_serial = CALLBACK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicBool, Ordering};

        static CALLBACK_CALLED: AtomicBool = AtomicBool::new(false);
        static CALLBACK_MESSAGE: Mutex<Option<String>> = Mutex::new(None);

        unsafe extern "C" fn test_callback(msg: *const c_char) {
            CALLBACK_CALLED.store(true, Ordering::SeqCst);
            // SAFETY: msg pointer is valid, passed from set_last_error via CString::as_ptr
            let c_str = unsafe { CStr::from_ptr(msg) };
            if let Ok(str_slice) = c_str.to_str()
                && let Ok(mut callback_msg) = CALLBACK_MESSAGE.lock() {
                    *callback_msg = Some(str_slice.to_string());
                }
        }

        // SAFETY: test calls C API functions with valid callback function pointer
        unsafe {
            // Clear any previous error state
            clear_last_error();

            // Set the error callback
            zipora_set_error_callback(Some(test_callback));

            // Reset callback state
            CALLBACK_CALLED.store(false, Ordering::SeqCst);
            if let Ok(mut callback_msg) = CALLBACK_MESSAGE.lock() {
                *callback_msg = None;
            }

            // Trigger an error by setting one directly
            set_last_error("Test callback message");

            // Check that callback was called
            assert!(CALLBACK_CALLED.load(Ordering::SeqCst));
            if let Ok(callback_msg) = CALLBACK_MESSAGE.lock() {
                assert!(callback_msg.is_some());
                assert_eq!(callback_msg.as_ref().unwrap(), "Test callback message");
            }

            // Clear the callback
            zipora_set_error_callback(None);

            // Reset state and trigger another error
            CALLBACK_CALLED.store(false, Ordering::SeqCst);
            if let Ok(mut callback_msg) = CALLBACK_MESSAGE.lock() {
                *callback_msg = None;
            }

            set_last_error("Another test message");

            // Callback should not have been called this time
            assert!(!CALLBACK_CALLED.load(Ordering::SeqCst));
            if let Ok(callback_msg) = CALLBACK_MESSAGE.lock() {
                assert!(callback_msg.is_none());
            }
        }
    }

    #[test]
    fn test_ffi_guard_catches_panics() {
        clear_last_error();

        // A panic inside an FFI body must be converted into the fallback
        // value instead of unwinding (which would abort a real C caller).
        let result = ffi_guard(CResult::InternalError, || -> CResult {
            panic!("boom in ffi body");
        });
        assert_eq!(result, CResult::InternalError);

        // SAFETY: zipora_last_error returns a valid thread-local or static string
        unsafe {
            let error_ptr = zipora_last_error();
            assert!(!error_ptr.is_null());
            let msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert!(
                msg.contains("boom in ffi body"),
                "unexpected last error: {msg}"
            );
        }
    }

    #[test]
    fn test_thread_local_errors() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        // Create multiple threads that each generate different errors
        for i in 0..3 {
            let results_clone = Arc::clone(&results);
            let handle = thread::spawn(move || {
                // SAFETY: test calls C API functions from spawned thread, CStr::from_ptr receives valid thread-local pointer
                unsafe {
                    // Each thread sets a different error message
                    let error_msg = match i {
                        0 => "Thread 0 error",
                        1 => "Thread 1 error",
                        2 => "Thread 2 error",
                        _ => "Unknown thread error",
                    };

                    set_last_error(error_msg);

                    // Get the error message from this thread
                    let error_ptr = zipora_last_error();
                    let retrieved_msg = if !error_ptr.is_null() {
                        CStr::from_ptr(error_ptr)
                            .to_str()
                            .unwrap_or("Invalid UTF-8")
                            .to_string()
                    } else {
                        "Null pointer".to_string()
                    };

                    // Store the result
                    let mut results_guard = results_clone.lock().unwrap_or_else(|e| e.into_inner());
                    results_guard.push((i, retrieved_msg));
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Check that each thread got its own error message
        let results_guard = results.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(results_guard.len(), 3);

        // Check that we got the expected error messages (order may vary due to threading)
        let mut found_thread0 = false;
        let mut found_thread1 = false;
        let mut found_thread2 = false;

        for (thread_id, error_msg) in results_guard.iter() {
            match *thread_id {
                0 => {
                    assert_eq!(error_msg, "Thread 0 error");
                    found_thread0 = true;
                }
                1 => {
                    assert_eq!(error_msg, "Thread 1 error");
                    found_thread1 = true;
                }
                2 => {
                    assert_eq!(error_msg, "Thread 2 error");
                    found_thread2 = true;
                }
                _ => {}
            }
        }

        assert!(found_thread0 && found_thread1 && found_thread2);
    }

    #[test]
    fn test_c11_1_misaligned_and_overflow_pointers_rejected_without_ub() {
        let mut raw = [0u64; 8];
        // SAFETY: `raw` is 8-byte aligned (`[u64; 8]`), so `(raw.as_mut_ptr() as *mut u8).add(1)`
        // has `addr % 8 == 1` and is strictly misaligned for all types with align >= 2.
        unsafe {
            let unaligned = (raw.as_mut_ptr() as *mut u8).add(1);

            // 1. Misaligned u32 buffer and overflow length in radix_sort_u32
            assert_eq!(
                radix_sort_u32(unaligned as *mut u32, 4),
                CResult::InvalidInput
            );
            let mut aligned_u32 = [3u32, 1, 2];
            assert_eq!(
                radix_sort_u32(aligned_u32.as_mut_ptr(), usize::MAX),
                CResult::InvalidInput
            );

            // 2. Misaligned opaque handles across FFI entry points
            fast_vec_free(unaligned as *mut CFastVec);
            assert_eq!(
                fast_vec_push(unaligned as *mut CFastVec, 1),
                CResult::InvalidInput
            );
            assert_eq!(fast_vec_len(unaligned as *const CFastVec), 0);
            assert!(fast_vec_data(unaligned as *const CFastVec).is_null());

            memory_pool_free(unaligned as *mut CMemoryPool);
            assert!(memory_pool_allocate(unaligned as *mut CMemoryPool).is_null());
            assert_eq!(
                memory_pool_deallocate(unaligned as *mut CMemoryPool, raw.as_mut_ptr().cast()),
                CResult::InvalidInput
            );

            // 3. BlobStore: misaligned handles/out-pointers, size > isize::MAX, and (NULL, 0) empty put
            blob_store_free(unaligned as *mut CBlobStore);
            let store = blob_store_new();
            assert!(!store.is_null());

            let mut valid_id: u32 = 0;
            // Misaligned store or record_id out-pointer
            assert_eq!(
                blob_store_put(
                    unaligned as *mut CBlobStore,
                    b"abc".as_ptr(),
                    3,
                    &mut valid_id
                ),
                CResult::InvalidInput
            );
            assert_eq!(
                blob_store_put(store, b"abc".as_ptr(), 3, unaligned as *mut u32),
                CResult::InvalidInput
            );
            // Overflow byte size (> isize::MAX)
            assert_eq!(
                blob_store_put(store, b"abc".as_ptr(), usize::MAX, &mut valid_id),
                CResult::InvalidInput
            );
            // (NULL, 0) is a valid C zero-length buffer (e.g. malloc(0) == NULL)
            assert_eq!(
                blob_store_put(store, ptr::null(), 0, &mut valid_id),
                CResult::Success
            );
            // (NULL, non-zero) must still fail
            assert_eq!(
                blob_store_put(store, ptr::null(), 4, &mut valid_id),
                CResult::InvalidInput
            );

            // Misaligned out-pointers in blob_store_get
            let mut valid_data: *const u8 = ptr::null();
            let mut valid_size: usize = 0;
            assert_eq!(
                blob_store_get(
                    store,
                    valid_id,
                    unaligned as *mut *const u8,
                    &mut valid_size
                ),
                CResult::InvalidInput
            );
            assert_eq!(
                blob_store_get(
                    store,
                    valid_id,
                    &mut valid_data,
                    unaligned as *mut usize
                ),
                CResult::InvalidInput
            );
            blob_store_free(store);

            // 4. SuffixArray: misaligned handle/out-pointers and overflow lengths
            suffix_array_free(unaligned as *mut CSuffixArray);
            assert_eq!(suffix_array_len(unaligned as *const CSuffixArray), 0);
            assert!(suffix_array_new(b"abc".as_ptr(), usize::MAX).is_null());

            let sa = suffix_array_new(b"banana".as_ptr(), 6);
            assert!(!sa.is_null());
            let mut start = 0usize;
            let mut count = 0usize;
            assert_eq!(
                suffix_array_search(
                    sa,
                    b"banana".as_ptr(),
                    6,
                    b"ana".as_ptr(),
                    3,
                    unaligned as *mut usize,
                    &mut count
                ),
                CResult::InvalidInput
            );
            assert_eq!(
                suffix_array_search(
                    sa,
                    b"banana".as_ptr(),
                    usize::MAX,
                    b"ana".as_ptr(),
                    3,
                    &mut start,
                    &mut count
                ),
                CResult::InvalidInput
            );
            suffix_array_free(sa);

            // 5. CBuffer: zero-capacity dangling/arbitrary pointer into_vec and size > isize::MAX slice
            let empty_buf = CBuffer {
                data: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                size: 4,
                capacity: 0,
            };
            assert!(empty_buf.as_slice().is_empty());
            assert!(empty_buf.into_vec().is_empty());
        }
    }

    #[test]
    fn test_c11_2a_cresult_not_found_mapping_and_blob_store_get() {
        assert_eq!(
            CResult::from(Err(crate::ZiporaError::not_found("missing item"))),
            CResult::NotFound
        );
        assert_eq!(
            CResult::from(Err(crate::ZiporaError::invalid_parameter("bad param"))),
            CResult::InvalidInput
        );

        // SAFETY: store is freshly allocated and freed once; out pointers are stack-aligned.
        unsafe {
            let store = blob_store_new();
            assert!(!store.is_null());
            let mut out_ptr: *const u8 = ptr::null();
            let mut out_len: usize = 0;
            clear_last_error();
            assert_eq!(
                blob_store_get(store, 999_999, &mut out_ptr, &mut out_len),
                CResult::NotFound
            );
            blob_store_free(store);
        }
    }

    #[test]
    fn test_c11_2b_set_last_error_preserves_embedded_nul_bytes() {
        clear_last_error();
        set_last_error("prefix\0suffix");
        // SAFETY: zipora_last_error returns a valid NUL-terminated C string pointer on this thread.
        let err_msg = unsafe { CStr::from_ptr(zipora_last_error()).to_str().unwrap() };
        assert_eq!(err_msg, "prefix\\0suffix");
    }

    #[test]
    fn test_c11_2c_error_callback_reentrant_unregister_does_not_deadlock() {
        let _callback_serial = CALLBACK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        unsafe extern "C" fn oneshot_unsubscribing_callback(_msg: *const c_char) {
            // SAFETY: re-entrant unregistration inside callback must not deadlock on ERROR_CALLBACK.
            unsafe {
                zipora_set_error_callback(None);
            }
        }

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // SAFETY: registers valid C callback and passes null handle to trigger set_last_error.
            unsafe {
                zipora_set_error_callback(Some(oneshot_unsubscribing_callback));
                let _ = fast_vec_push(ptr::null_mut(), 7);
            }
            let _ = tx.send(());
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(500)).is_ok(),
            "C11.2: re-entrant zipora_set_error_callback inside error callback deadlocked on ERROR_CALLBACK"
        );
    }
}
