//! C FFI bindings for blob store types

use super::CResult;
use super::c_api::{ffi_guard, set_last_error};
use super::types::CBlobStore;
use std::ptr;

/// Create a new blob store
///
/// # Safety
///
/// The returned pointer must be freed with `blob_store_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blob_store_new() -> *mut CBlobStore {
    ffi_guard(ptr::null_mut(), || Box::into_raw(Box::new(CBlobStore::new())))
}

/// Free a blob store
///
/// # Safety
///
/// The pointer must be a valid CBlobStore pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blob_store_free(store: *mut CBlobStore) {
    ffi_guard((), || {
        if !store.is_null() {
            if !store.is_aligned() {
                set_last_error("BlobStore pointer is misaligned");
                return;
            }
            // SAFETY: pointer is non-null and aligned (checked above), created by blob_store_new via Box::into_raw, not previously freed
            let _store = unsafe { Box::from_raw(store) };
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
        if store.is_null()
            || !store.is_aligned()
            || record_id.is_null()
            || !record_id.is_aligned()
            || (data.is_null() && size > 0)
            || size > isize::MAX as usize
        {
            set_last_error(
                "Blob store, data, or record_id parameter is null, misaligned, or exceeds isize::MAX",
            );
            return CResult::InvalidInput;
        }

        // SAFETY: store is non-null and aligned (checked above), points to initialized CBlobStore
        let blob_store = unsafe { &mut *store };
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
        if store.is_null()
            || !store.is_aligned()
            || data.is_null()
            || !data.is_aligned()
            || size.is_null()
            || !size.is_aligned()
        {
            set_last_error("Blob store or output parameter is null or misaligned");
            return CResult::InvalidInput;
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized CBlobStore
        let blob_store = unsafe { &*store };

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
