//! Main C API interface
//!
//! This module provides the primary C-compatible interface for the infini-zip library,
//! allowing seamless integration with existing C/C++ codebases.

use super::{types::*, CResult};
use crate::blob_store::traits::BlobStore;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::Mutex;

// Thread-local storage for error messages
thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

// Global error callback storage
static ERROR_CALLBACK: Mutex<Option<unsafe extern "C" fn(*const c_char)>> = Mutex::new(None);

/// Set the last error message for the current thread
fn set_last_error(msg: &str) {
    LAST_ERROR.with(|error| {
        if let Ok(cstring) = CString::new(msg) {
            // Store a clone for thread-local storage
            *error.borrow_mut() = Some(cstring.clone());

            // Also call the error callback if one is set
            if let Ok(callback_guard) = ERROR_CALLBACK.lock() {
                if let Some(callback) = *callback_guard {
                    unsafe {
                        callback(cstring.as_ptr());
                    }
                }
            }
        }
    });
}

/// Convert CResult to error message
fn cresult_to_error_msg(result: CResult) -> &'static str {
    match result {
        CResult::Success => "Success",
        CResult::InvalidInput => "Invalid input parameter",
        CResult::MemoryError => "Memory allocation error",
        CResult::IoError => "I/O operation failed",
        CResult::InternalError => "Internal library error",
        CResult::UnsupportedOperation => "Unsupported operation",
        CResult::NotFound => "Resource not found",
    }
}

/// Initialize the infini-zip library
///
/// # Safety
///
/// This function is safe to call multiple times.
#[no_mangle]
pub unsafe extern "C" fn infini_zip_init() -> CResult {
    crate::init();
    CResult::Success
}

/// Get library version string
///
/// # Safety
///
/// The returned string is valid until the next call to this function.
/// The caller should not free the returned pointer.
#[no_mangle]
pub unsafe extern "C" fn infini_zip_version() -> *const c_char {
    static mut VERSION_CSTRING: Option<CString> = None;

    unsafe {
        if VERSION_CSTRING.is_none() {
            if let Ok(cstring) = CString::new(crate::VERSION) {
                VERSION_CSTRING = Some(cstring);
            } else {
                return ptr::null();
            }
        }

        VERSION_CSTRING.as_ref().unwrap().as_ptr()
    }
}

/// Check if SIMD optimizations are available
#[no_mangle]
pub unsafe extern "C" fn infini_zip_has_simd() -> c_int {
    if crate::has_simd_support() {
        1
    } else {
        0
    }
}

/// Create a new FastVec instance
///
/// # Safety
///
/// The returned pointer must be freed with `fast_vec_free`.
#[no_mangle]
pub unsafe extern "C" fn fast_vec_new() -> *mut CFastVec {
    let fast_vec = Box::new(crate::FastVec::<u8>::new());
    Box::into_raw(fast_vec) as *mut CFastVec
}

/// Free a FastVec instance
///
/// # Safety
///
/// The pointer must be a valid CFastVec pointer returned from `fast_vec_new`.
/// The pointer becomes invalid after this call.
#[no_mangle]
pub unsafe extern "C" fn fast_vec_free(vec: *mut CFastVec) {
    if !vec.is_null() {
        let _vec = unsafe { Box::from_raw(vec as *mut crate::FastVec<u8>) };
        // Automatic cleanup when Box is dropped
    }
}

/// Push a byte to a FastVec
///
/// # Safety
///
/// The vec pointer must be a valid CFastVec pointer.
#[no_mangle]
pub unsafe extern "C" fn fast_vec_push(vec: *mut CFastVec, value: u8) -> CResult {
    if vec.is_null() {
        set_last_error("FastVec pointer is null");
        return CResult::InvalidInput;
    }

    let fast_vec = unsafe { &mut *(vec as *mut crate::FastVec<u8>) };
    match fast_vec.push(value) {
        Ok(_) => CResult::Success,
        Err(e) => {
            set_last_error(&format!("Failed to push to FastVec: {}", e));
            CResult::MemoryError
        }
    }
}

/// Get the length of a FastVec
///
/// # Safety
///
/// The vec pointer must be a valid CFastVec pointer.
#[no_mangle]
pub unsafe extern "C" fn fast_vec_len(vec: *const CFastVec) -> usize {
    if vec.is_null() {
        return 0;
    }

    let fast_vec = unsafe { &*(vec as *const crate::FastVec<u8>) };
    fast_vec.len()
}

/// Get a pointer to the FastVec's data
///
/// # Safety
///
/// The vec pointer must be a valid CFastVec pointer.
/// The returned pointer is valid until the FastVec is modified or freed.
#[no_mangle]
pub unsafe extern "C" fn fast_vec_data(vec: *const CFastVec) -> *const u8 {
    if vec.is_null() {
        return ptr::null();
    }

    let fast_vec = unsafe { &*(vec as *const crate::FastVec<u8>) };
    fast_vec.as_ptr()
}

/// Create a memory pool
///
/// # Safety
///
/// The returned pointer must be freed with `memory_pool_free`.
#[no_mangle]
pub unsafe extern "C" fn memory_pool_new(chunk_size: usize, max_chunks: usize) -> *mut CMemoryPool {
    let config = crate::memory::PoolConfig::new(chunk_size, max_chunks, 8);
    match crate::memory::MemoryPool::new(config) {
        Ok(pool) => Box::into_raw(Box::new(pool)) as *mut CMemoryPool,
        Err(_) => ptr::null_mut(),
    }
}

/// Free a memory pool
///
/// # Safety
///
/// The pointer must be a valid CMemoryPool pointer returned from `memory_pool_new`.
#[no_mangle]
pub unsafe extern "C" fn memory_pool_free(pool: *mut CMemoryPool) {
    if !pool.is_null() {
        let _pool = unsafe { Box::from_raw(pool as *mut crate::memory::MemoryPool) };
        // Automatic cleanup when Box is dropped
    }
}

/// Allocate memory from a pool
///
/// # Safety
///
/// The pool pointer must be a valid CMemoryPool pointer.
/// The returned pointer must be freed with `memory_pool_deallocate`.
#[no_mangle]
pub unsafe extern "C" fn memory_pool_allocate(pool: *mut CMemoryPool) -> *mut c_void {
    if pool.is_null() {
        return ptr::null_mut();
    }

    let memory_pool = unsafe { &*(pool as *const crate::memory::MemoryPool) };
    match memory_pool.allocate() {
        Ok(ptr) => ptr.as_ptr() as *mut c_void,
        Err(_) => ptr::null_mut(),
    }
}

/// Deallocate memory back to a pool
///
/// # Safety
///
/// Both pointers must be valid, and the memory pointer must have been
/// allocated from the specified pool.
#[no_mangle]
pub unsafe extern "C" fn memory_pool_deallocate(
    pool: *mut CMemoryPool,
    ptr: *mut c_void,
) -> CResult {
    if pool.is_null() || ptr.is_null() {
        set_last_error("Memory pool or pointer is null");
        return CResult::InvalidInput;
    }

    let memory_pool = unsafe { &*(pool as *const crate::memory::MemoryPool) };
    let non_null_ptr = match std::ptr::NonNull::new(ptr as *mut u8) {
        Some(p) => p,
        None => {
            set_last_error("Failed to create NonNull pointer from raw pointer");
            return CResult::InvalidInput;
        }
    };

    match memory_pool.deallocate(non_null_ptr) {
        Ok(_) => CResult::Success,
        Err(e) => {
            set_last_error(&format!("Failed to deallocate memory: {}", e));
            CResult::InternalError
        }
    }
}

/// Create a new blob store
///
/// # Safety
///
/// The returned pointer must be freed with `blob_store_free`.
#[no_mangle]
pub unsafe extern "C" fn blob_store_new() -> *mut CBlobStore {
    let store = Box::new(crate::blob_store::MemoryBlobStore::new());
    Box::into_raw(store) as *mut CBlobStore
}

/// Free a blob store
///
/// # Safety
///
/// The pointer must be a valid CBlobStore pointer.
#[no_mangle]
pub unsafe extern "C" fn blob_store_free(store: *mut CBlobStore) {
    if !store.is_null() {
        let _store = unsafe { Box::from_raw(store as *mut crate::blob_store::MemoryBlobStore) };
        // Automatic cleanup when Box is dropped
    }
}

/// Put data into a blob store
///
/// # Safety
///
/// The store pointer must be valid, and data must point to a valid buffer of the specified size.
#[no_mangle]
pub unsafe extern "C" fn blob_store_put(
    store: *mut CBlobStore,
    data: *const u8,
    size: usize,
    record_id: *mut u32,
) -> CResult {
    if store.is_null() || data.is_null() || record_id.is_null() {
        set_last_error("Blob store, data, or record_id pointer is null");
        return CResult::InvalidInput;
    }

    let blob_store = unsafe { &mut *(store as *mut crate::blob_store::MemoryBlobStore) };
    let data_slice = unsafe { std::slice::from_raw_parts(data, size) };

    match blob_store.put(data_slice) {
        Ok(id) => {
            unsafe {
                *record_id = id;
            }
            CResult::Success
        }
        Err(e) => {
            set_last_error(&format!("Failed to put data in blob store: {}", e));
            CResult::InternalError
        }
    }
}

/// Get data from a blob store
///
/// # Safety
///
/// The store pointer must be valid. The data and size pointers will be set to
/// point to allocated memory that must be freed with `infini_zip_free_blob_data`.
#[no_mangle]
pub unsafe extern "C" fn blob_store_get(
    store: *const CBlobStore,
    record_id: u32,
    data: *mut *const u8,
    size: *mut usize,
) -> CResult {
    if store.is_null() || data.is_null() || size.is_null() {
        return CResult::InvalidInput;
    }

    let blob_store = unsafe { &*(store as *const crate::blob_store::MemoryBlobStore) };

    match blob_store.get(record_id) {
        Ok(blob_data) => {
            // Allocate memory for the data that can be properly freed later
            let data_len = blob_data.len();
            let data_ptr = unsafe {
                std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(data_len, 1))
            };
            if data_ptr.is_null() {
                return CResult::MemoryError;
            }

            // Copy the data into the allocated memory
            unsafe {
                std::ptr::copy_nonoverlapping(blob_data.as_ptr(), data_ptr, data_len);
            }

            unsafe {
                *data = data_ptr;
                *size = data_len;
            }
            CResult::Success
        }
        Err(_) => CResult::InvalidInput,
    }
}

/// Create a suffix array from text
///
/// # Safety
///
/// The text pointer must point to a valid buffer of the specified size.
/// The returned pointer must be freed with `suffix_array_free`.
#[no_mangle]
pub unsafe extern "C" fn suffix_array_new(text: *const u8, size: usize) -> *mut CSuffixArray {
    if text.is_null() || size == 0 {
        return ptr::null_mut();
    }

    let text_slice = unsafe { std::slice::from_raw_parts(text, size) };
    match crate::algorithms::SuffixArray::new(text_slice) {
        Ok(sa) => Box::into_raw(Box::new(sa)) as *mut CSuffixArray,
        Err(_) => ptr::null_mut(),
    }
}

/// Free a suffix array
///
/// # Safety
///
/// The pointer must be a valid CSuffixArray pointer.
#[no_mangle]
pub unsafe extern "C" fn suffix_array_free(sa: *mut CSuffixArray) {
    if !sa.is_null() {
        let _sa = unsafe { Box::from_raw(sa as *mut crate::algorithms::SuffixArray) };
        // Automatic cleanup when Box is dropped
    }
}

/// Get the length of a suffix array
///
/// # Safety
///
/// The sa pointer must be a valid CSuffixArray pointer.
#[no_mangle]
pub unsafe extern "C" fn suffix_array_len(sa: *const CSuffixArray) -> usize {
    if sa.is_null() {
        return 0;
    }

    let suffix_array = unsafe { &*(sa as *const crate::algorithms::SuffixArray) };
    suffix_array.text_len()
}

/// Search for a pattern in the suffix array
///
/// # Safety
///
/// All pointers must be valid, and pattern must point to a buffer of the specified size.
#[no_mangle]
pub unsafe extern "C" fn suffix_array_search(
    sa: *const CSuffixArray,
    text: *const u8,
    text_size: usize,
    pattern: *const u8,
    pattern_size: usize,
    start: *mut usize,
    count: *mut usize,
) -> CResult {
    if sa.is_null() || text.is_null() || pattern.is_null() || start.is_null() || count.is_null() {
        return CResult::InvalidInput;
    }

    let suffix_array = unsafe { &*(sa as *const crate::algorithms::SuffixArray) };
    let text_slice = unsafe { std::slice::from_raw_parts(text, text_size) };
    let pattern_slice = unsafe { std::slice::from_raw_parts(pattern, pattern_size) };

    let (search_start, search_count) = suffix_array.search(text_slice, pattern_slice);
    unsafe {
        *start = search_start;
        *count = search_count;
    }

    CResult::Success
}

/// Sort an array of 32-bit unsigned integers using radix sort
///
/// # Safety
///
/// The data pointer must point to a valid array of the specified size.
#[no_mangle]
pub unsafe extern "C" fn radix_sort_u32(data: *mut u32, size: usize) -> CResult {
    if data.is_null() || size == 0 {
        return CResult::InvalidInput;
    }

    let data_slice = unsafe { std::slice::from_raw_parts_mut(data, size) };
    let mut sorter = crate::algorithms::RadixSort::new();

    match sorter.sort_u32(data_slice) {
        Ok(_) => CResult::Success,
        Err(_) => CResult::InternalError,
    }
}

/// Get last error message (thread-local)
///
/// # Safety
///
/// The returned string is valid until the next error occurs on this thread.
/// The caller should not free the returned pointer.
#[no_mangle]
pub unsafe extern "C" fn infini_zip_last_error() -> *const c_char {
    LAST_ERROR.with(|error| match error.borrow().as_ref() {
        Some(cstring) => cstring.as_ptr(),
        None => {
            static NO_ERROR_MSG: &[u8] = b"No error information available\0";
            NO_ERROR_MSG.as_ptr() as *const c_char
        }
    })
}

/// Set a custom error callback
///
/// # Safety
///
/// The callback function pointer must be valid for the lifetime of the library usage.
/// The callback will be called from the thread that generates the error.
#[no_mangle]
pub unsafe extern "C" fn infini_zip_set_error_callback(
    callback: Option<unsafe extern "C" fn(*const c_char)>,
) {
    if let Ok(mut callback_guard) = ERROR_CALLBACK.lock() {
        *callback_guard = callback;
    }
}

/// Free blob data returned by blob_store_get
///
/// # Safety
///
/// The data pointer must be a pointer returned by `blob_store_get`.
/// The size must match the size returned by `blob_store_get`.
/// The pointer becomes invalid after this call.
#[no_mangle]
pub unsafe extern "C" fn infini_zip_free_blob_data(data: *mut u8, size: usize) {
    if !data.is_null() && size > 0 {
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(size, 1);
            std::alloc::dealloc(data, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_version_api() {
        unsafe {
            let version_ptr = infini_zip_version();
            assert!(!version_ptr.is_null());

            let version_cstr = CStr::from_ptr(version_ptr);
            let version_str = version_cstr.to_str().unwrap();
            assert!(!version_str.is_empty());
        }
    }

    #[test]
    fn test_fast_vec_api() {
        unsafe {
            let vec = fast_vec_new();
            assert!(!vec.is_null());

            assert_eq!(fast_vec_len(vec), 0);

            assert_eq!(fast_vec_push(vec, 42), CResult::Success);
            assert_eq!(fast_vec_push(vec, 84), CResult::Success);

            assert_eq!(fast_vec_len(vec), 2);

            let data_ptr = fast_vec_data(vec);
            assert!(!data_ptr.is_null());

            let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, 2) };
            assert_eq!(data_slice, &[42, 84]);

            fast_vec_free(vec);
        }
    }

    #[test]
    fn test_memory_pool_api() {
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
    fn test_blob_store_api() {
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
            let retrieved_data = unsafe { std::slice::from_raw_parts(data_ptr, size) };
            assert_eq!(retrieved_data, test_data);

            // Free the blob data
            infini_zip_free_blob_data(data_ptr as *mut u8, size);

            blob_store_free(store);
        }
    }

    #[test]
    fn test_radix_sort_api() {
        unsafe {
            let mut data = [5u32, 2, 8, 1, 9, 3, 7, 4, 6];
            let result = radix_sort_u32(data.as_mut_ptr(), data.len());

            assert_eq!(result, CResult::Success);
            assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        }
    }

    #[test]
    fn test_init_and_simd() {
        unsafe {
            assert_eq!(infini_zip_init(), CResult::Success);

            let has_simd = infini_zip_has_simd();
            // Should not panic, return 0 or 1
            assert!(has_simd == 0 || has_simd == 1);
        }
    }

    #[test]
    fn test_error_handling() {
        unsafe {
            // Test initial state - no error
            let error_ptr = infini_zip_last_error();
            assert!(!error_ptr.is_null());
            let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert_eq!(error_msg, "No error information available");

            // Test setting an error message directly
            set_last_error("Test error message");

            // Check that error message was set
            let error_ptr = infini_zip_last_error();
            assert!(!error_ptr.is_null());
            let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert_eq!(error_msg, "Test error message");
        }
    }

    #[test]
    fn test_error_callback() {
        use std::sync::atomic::{AtomicBool, Ordering};

        static CALLBACK_CALLED: AtomicBool = AtomicBool::new(false);
        static mut CALLBACK_MESSAGE: Option<String> = None;

        unsafe extern "C" fn test_callback(msg: *const c_char) {
            CALLBACK_CALLED.store(true, Ordering::SeqCst);
            let c_str = unsafe { CStr::from_ptr(msg) };
            if let Ok(str_slice) = c_str.to_str() {
                unsafe {
                    CALLBACK_MESSAGE = Some(str_slice.to_string());
                }
            }
        }

        unsafe {
            // Set the error callback
            infini_zip_set_error_callback(Some(test_callback));

            // Reset callback state
            CALLBACK_CALLED.store(false, Ordering::SeqCst);
            CALLBACK_MESSAGE = None;

            // Trigger an error by setting one directly
            set_last_error("Test callback message");

            // Check that callback was called
            assert!(CALLBACK_CALLED.load(Ordering::SeqCst));
            assert!(CALLBACK_MESSAGE.is_some());
            assert_eq!(CALLBACK_MESSAGE.as_ref().unwrap(), "Test callback message");

            // Clear the callback
            infini_zip_set_error_callback(None);

            // Reset state and trigger another error
            CALLBACK_CALLED.store(false, Ordering::SeqCst);
            CALLBACK_MESSAGE = None;

            set_last_error("Another test message");

            // Callback should not have been called this time
            assert!(!CALLBACK_CALLED.load(Ordering::SeqCst));
            assert!(CALLBACK_MESSAGE.is_none());
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
                    let error_ptr = infini_zip_last_error();
                    let retrieved_msg = if !error_ptr.is_null() {
                        CStr::from_ptr(error_ptr)
                            .to_str()
                            .unwrap_or("Invalid UTF-8")
                            .to_string()
                    } else {
                        "Null pointer".to_string()
                    };

                    // Store the result
                    let mut results_guard = results_clone.lock().unwrap();
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
        let results_guard = results.lock().unwrap();
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
}
