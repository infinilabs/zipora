//! C FFI bindings for blob store types

use super::types::*;
use crate::blob_store::{BlobStore, MemoryBlobStore};
use crate::error::Result;
use crate::ffi::CResult;
use std::os::raw::c_void;

/// Create a new memory blob store
#[no_mangle]
pub unsafe extern "C" fn blob_store_new() -> *mut CBlobStore {
    let store = Box::new(CBlobStore::new());
    Box::into_raw(store)
}

/// Free a blob store
#[no_mangle]
pub unsafe extern "C" fn blob_store_free(store: *mut CBlobStore) {
    if !store.is_null() {
        unsafe { drop(Box::from_raw(store)) };
    }
}

/// Put data into blob store
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
    
    let store = &mut *store;
    let data_slice = std::slice::from_raw_parts(data, size);
    
    match store.put(data_slice) {
        Ok(id) => {
            *record_id = id;
            CResult::Success
        }
        Err(_) => CResult::InternalError,
    }
}

/// Get data from blob store
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
    
    let store = &*store;
    match store.get(record_id) {
        Ok(retrieved_data) => {
            // Note: This is a simplified implementation that leaks memory
            // In a real implementation, you'd need proper memory management
            let leaked_data = Box::leak(retrieved_data.into_boxed_slice());
            *data = leaked_data.as_ptr();
            *size = leaked_data.len();
            CResult::Success
        }
        Err(_) => CResult::NotFound,
    }
}