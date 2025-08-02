//! C FFI bindings for container types

use super::types::*;
use crate::ffi::CResult;

/// Create a new FastVec
#[no_mangle]
pub unsafe extern "C" fn fast_vec_new() -> *mut CFastVec {
    let vec = Box::new(CFastVec::new());
    Box::into_raw(vec)
}

/// Free a FastVec
#[no_mangle]
pub unsafe extern "C" fn fast_vec_free(vec: *mut CFastVec) {
    if !vec.is_null() {
        unsafe { drop(Box::from_raw(vec)) };
    }
}

/// Push an element to FastVec
#[no_mangle]
pub unsafe extern "C" fn fast_vec_push(vec: *mut CFastVec, value: u32) -> CResult {
    if vec.is_null() {
        return CResult::InvalidInput;
    }
    
    let vec = &mut *vec;
    match vec.push(value) {
        Ok(_) => CResult::Success,
        Err(_) => CResult::InternalError,
    }
}

/// Get the length of FastVec
#[no_mangle]
pub unsafe extern "C" fn fast_vec_len(vec: *const CFastVec) -> usize {
    if vec.is_null() {
        return 0;
    }
    
    let vec = &*vec;
    vec.len()
}

/// Get element at index
#[no_mangle]
pub unsafe extern "C" fn fast_vec_get(vec: *const CFastVec, index: usize, value: *mut u32) -> CResult {
    if vec.is_null() || value.is_null() {
        return CResult::InvalidInput;
    }
    
    let vec = &*vec;
    if let Some(val) = vec.get(index) {
        *value = *val;
        CResult::Success
    } else {
        CResult::InvalidInput
    }
}