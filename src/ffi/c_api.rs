//! Main C API interface
//!
//! This module provides the primary C-compatible interface for the infini-zip library,
//! allowing seamless integration with existing C/C++ codebases.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use super::{CResult, types::*};

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
    
    if VERSION_CSTRING.is_none() {
        if let Ok(cstring) = CString::new(crate::VERSION) {
            VERSION_CSTRING = Some(cstring);
        } else {
            return ptr::null();
        }
    }
    
    VERSION_CSTRING.as_ref().unwrap().as_ptr()
}

/// Check if SIMD optimizations are available
#[no_mangle]
pub unsafe extern "C" fn infini_zip_has_simd() -> c_int {
    if crate::has_simd_support() { 1 } else { 0 }
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
        let _vec = Box::from_raw(vec as *mut crate::FastVec<u8>);
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
        return CResult::InvalidInput;
    }
    
    let fast_vec = &mut *(vec as *mut crate::FastVec<u8>);
    match fast_vec.push(value) {
        Ok(_) => CResult::Success,
        Err(_) => CResult::MemoryError,
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
    
    let fast_vec = &*(vec as *const crate::FastVec<u8>);
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
    
    let fast_vec = &*(vec as *const crate::FastVec<u8>);
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
        let _pool = Box::from_raw(pool as *mut crate::memory::MemoryPool);
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
    
    let memory_pool = &*(pool as *const crate::memory::MemoryPool);
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
pub unsafe extern "C" fn memory_pool_deallocate(pool: *mut CMemoryPool, ptr: *mut c_void) -> CResult {
    if pool.is_null() || ptr.is_null() {
        return CResult::InvalidInput;
    }
    
    let memory_pool = &*(pool as *const crate::memory::MemoryPool);
    let non_null_ptr = match std::ptr::NonNull::new(ptr as *mut u8) {
        Some(p) => p,
        None => return CResult::InvalidInput,
    };
    
    match memory_pool.deallocate(non_null_ptr) {
        Ok(_) => CResult::Success,
        Err(_) => CResult::InternalError,
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
        let _store = Box::from_raw(store as *mut crate::blob_store::MemoryBlobStore);
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
        return CResult::InvalidInput;
    }
    
    let blob_store = &mut *(store as *mut crate::blob_store::MemoryBlobStore);
    let data_slice = std::slice::from_raw_parts(data, size);
    
    match blob_store.put(data_slice) {
        Ok(id) => {
            *record_id = id;
            CResult::Success
        }
        Err(_) => CResult::InternalError,
    }
}

/// Get data from a blob store
/// 
/// # Safety
/// 
/// The store pointer must be valid. The data and size pointers will be set to
/// point to internal storage that is valid until the blob store is modified or freed.
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
    
    let blob_store = &*(store as *const crate::blob_store::MemoryBlobStore);
    
    match blob_store.get(record_id) {
        Ok(blob_data) => {
            // This is a potential memory safety issue - we're returning a pointer
            // to data that might be deallocated. In a real implementation, we'd need
            // a different approach, perhaps with reference counting or copying.
            *data = blob_data.as_ptr();
            *size = blob_data.len();
            // We're leaking the Vec here to keep the data alive
            // In practice, you'd want a better memory management strategy
            std::mem::forget(blob_data);
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
    
    let text_slice = std::slice::from_raw_parts(text, size);
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
        let _sa = Box::from_raw(sa as *mut crate::algorithms::SuffixArray);
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
    
    let suffix_array = &*(sa as *const crate::algorithms::SuffixArray);
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
    
    let suffix_array = &*(sa as *const crate::algorithms::SuffixArray);
    let text_slice = std::slice::from_raw_parts(text, text_size);
    let pattern_slice = std::slice::from_raw_parts(pattern, pattern_size);
    
    let (search_start, search_count) = suffix_array.search(text_slice, pattern_slice);
    *start = search_start;
    *count = search_count;
    
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
    
    let data_slice = std::slice::from_raw_parts_mut(data, size);
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
    // This would require a thread-local error storage system
    // For now, return a placeholder
    static ERROR_MSG: &[u8] = b"No error information available\0";
    ERROR_MSG.as_ptr() as *const c_char
}

/// Set a custom error callback
/// 
/// # Safety
/// 
/// The callback function pointer must be valid for the lifetime of the library usage.
#[no_mangle]
pub unsafe extern "C" fn infini_zip_set_error_callback(
    callback: Option<unsafe extern "C" fn(*const c_char)>,
) {
    // Store the callback for future error reporting
    // This would be implemented with thread-local or global state
    let _ = callback; // Placeholder
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
            
            let data_slice = std::slice::from_raw_parts(data_ptr, 2);
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
            
            let result = blob_store_put(
                store,
                test_data.as_ptr(),
                test_data.len(),
                &mut record_id,
            );
            assert_eq!(result, CResult::Success);
            
            let mut data_ptr: *const u8 = ptr::null();
            let mut size: usize = 0;
            
            let result = blob_store_get(store, record_id, &mut data_ptr, &mut size);
            assert_eq!(result, CResult::Success);
            assert!(!data_ptr.is_null());
            assert_eq!(size, test_data.len());
            
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
}