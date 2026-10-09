//! C FFI bindings for container and memory pool types

use super::CResult;
use super::c_api::{ffi_guard, set_last_error};
use super::types::{CFastVec, CMemoryPool};
use std::os::raw::c_void;
use std::ptr;

/// Create a new FastVec instance
///
/// # Safety
///
/// The returned pointer must be freed with `fast_vec_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fast_vec_new() -> *mut CFastVec {
    ffi_guard(ptr::null_mut(), || Box::into_raw(Box::new(CFastVec::new())))
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
        if !vec.is_null() {
            if !vec.is_aligned() {
                set_last_error("FastVec pointer is misaligned");
                return;
            }
            // SAFETY: pointer is non-null and aligned (checked above), created by fast_vec_new via Box::into_raw, not previously freed
            let _vec = unsafe { Box::from_raw(vec) };
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
        if vec.is_null() || !vec.is_aligned() {
            set_last_error("FastVec pointer is null or misaligned");
            return CResult::InvalidInput;
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized CFastVec created by fast_vec_new
        let fast_vec = unsafe { &mut *vec };
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
        if vec.is_null() || !vec.is_aligned() {
            return 0;
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized CFastVec
        let fast_vec = unsafe { &*vec };
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
        if vec.is_null() || !vec.is_aligned() {
            return ptr::null();
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized CFastVec
        let fast_vec = unsafe { &*vec };
        fast_vec.as_ptr()
    })
}

/// Create a memory pool
///
/// `chunk_size` must be non-zero and fit within a valid 8-byte aligned layout
/// (`chunk_size <= INTPTR_MAX - 7`); otherwise `NULL` is returned and the error
/// details are stored in `zipora_last_error`.
///
/// `max_chunks` limits only the cache of free chunks retained when blocks are
/// returned via `memory_pool_deallocate`; it does not limit the number of
/// concurrently live blocks returned by `memory_pool_allocate`.
///
/// # Safety
///
/// The returned pointer must be freed with `memory_pool_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memory_pool_new(chunk_size: usize, max_chunks: usize) -> *mut CMemoryPool {
    ffi_guard(ptr::null_mut(), || {
        let config = crate::memory::PoolConfig::new(chunk_size, max_chunks, 8);
        match crate::memory::MemoryPool::new(config) {
            Ok(pool) => Box::into_raw(Box::new(CMemoryPool { inner: pool })),
            Err(e) => {
                set_last_error(&format!("Failed to create memory pool: {}", e));
                ptr::null_mut()
            }
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
        if !pool.is_null() {
            if !pool.is_aligned() {
                set_last_error("MemoryPool pointer is misaligned");
                return;
            }
            // SAFETY: pointer is non-null and aligned (checked above), created by memory_pool_new via Box::into_raw, not previously freed
            let _pool = unsafe { Box::from_raw(pool) };
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
        if pool.is_null() || !pool.is_aligned() {
            set_last_error("MemoryPool pointer is null or misaligned");
            return ptr::null_mut();
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized CMemoryPool
        let memory_pool = unsafe { &*pool };
        match memory_pool.inner.allocate() {
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
        if pool.is_null() || !pool.is_aligned() || ptr.is_null() {
            set_last_error("Memory pool or pointer is null or misaligned");
            return CResult::InvalidInput;
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized CMemoryPool
        let memory_pool = unsafe { &*pool };
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
        match unsafe { memory_pool.inner.deallocate(non_null_ptr) } {
            Ok(_) => CResult::Success,
            Err(e) => {
                set_last_error(&format!("Failed to deallocate memory: {}", e));
                CResult::from_zipora_error(&e)
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::ffi::CResult;
    use crate::ffi::c_api::*;

    #[test]
    fn test_fast_vec_lifecycle() {
        unsafe {
            let vec = fast_vec_new();
            assert!(!vec.is_null());

            assert_eq!(fast_vec_push(vec, 42), CResult::Success);
            assert_eq!(fast_vec_push(vec, 100), CResult::Success);
            assert_eq!(fast_vec_push(vec, 255), CResult::Success);

            assert_eq!(fast_vec_len(vec), 3);

            let data_ptr = fast_vec_data(vec);
            assert!(!data_ptr.is_null());
            let data = std::slice::from_raw_parts(data_ptr, 3);
            assert_eq!(data[0], 42);
            assert_eq!(data[1], 100);
            assert_eq!(data[2], 255);

            fast_vec_free(vec);
        }
    }

    #[test]
    fn test_fast_vec_empty() {
        unsafe {
            let vec = fast_vec_new();
            assert!(!vec.is_null());
            assert_eq!(fast_vec_len(vec), 0);
            fast_vec_free(vec);
        }
    }

    #[test]
    fn test_fast_vec_null_safety() {
        unsafe {
            fast_vec_free(std::ptr::null_mut());
            assert_eq!(fast_vec_len(std::ptr::null()), 0);
            assert!(fast_vec_data(std::ptr::null()).is_null());
            assert_eq!(
                fast_vec_push(std::ptr::null_mut(), 0),
                CResult::InvalidInput
            );
        }
    }

    #[test]
    fn test_fast_vec_many_pushes() {
        unsafe {
            let vec = fast_vec_new();
            for i in 0..=255u8 {
                assert_eq!(fast_vec_push(vec, i), CResult::Success);
            }
            assert_eq!(fast_vec_len(vec), 256);

            let data_ptr = fast_vec_data(vec);
            let data = std::slice::from_raw_parts(data_ptr, 256);
            for i in 0..=255u8 {
                assert_eq!(data[i as usize], i);
            }
            fast_vec_free(vec);
        }
    }

    #[test]
    fn test_s31_r5_memory_pool_new_rejects_invalid_layout_and_documents_max_chunks() {
        // SAFETY: calls C FFI memory_pool functions with valid handles and frees every allocation.
        unsafe {
            assert!(
                memory_pool_new(usize::MAX, 1).is_null(),
                "memory_pool_new(usize::MAX, 1) must fail upfront on invalid Layout"
            );
            assert!(
                memory_pool_new((isize::MAX as usize) - 6, 1).is_null(),
                "memory_pool_new((isize::MAX as usize) - 6, 1) exceeds 8-byte aligned Layout limit"
            );

            let pool = memory_pool_new(64, 2);
            assert!(!pool.is_null());
            let mut blocks = Vec::new();
            for _ in 0..8 {
                let p = memory_pool_allocate(pool);
                assert!(!p.is_null());
                blocks.push(p);
            }
            for p in blocks {
                assert_eq!(memory_pool_deallocate(pool, p), CResult::Success);
            }
            memory_pool_free(pool);
        }
    }
}
