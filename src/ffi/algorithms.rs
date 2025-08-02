//! C FFI bindings for algorithm types

use super::types::*;
use crate::algorithms::{SuffixArray, RadixSort};
use crate::ffi::CResult;

/// Create a new suffix array
#[no_mangle]
pub unsafe extern "C" fn suffix_array_new(data: *const u8, size: usize) -> *mut CSuffixArray {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    
    let data_slice = std::slice::from_raw_parts(data, size);
    match SuffixArray::new(data_slice) {
        Ok(sa) => {
            let c_sa = Box::new(CSuffixArray::new(sa));
            Box::into_raw(c_sa)
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a suffix array
#[no_mangle]
pub unsafe extern "C" fn suffix_array_free(sa: *mut CSuffixArray) {
    if !sa.is_null() {
        unsafe { drop(Box::from_raw(sa)) };
    }
}

/// Search pattern in suffix array
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
    
    let sa = &*sa;
    let text_slice = std::slice::from_raw_parts(text, text_size);
    let pattern_slice = std::slice::from_raw_parts(pattern, pattern_size);
    
    let (search_start, search_count) = sa.search(text_slice, pattern_slice);
    *start = search_start;
    *count = search_count;
    
    CResult::Success
}

/// Sort u32 array using radix sort
#[no_mangle]
pub unsafe extern "C" fn radix_sort_u32(data: *mut u32, size: usize) -> CResult {
    if data.is_null() {
        return CResult::InvalidInput;
    }
    
    let data_slice = unsafe { std::slice::from_raw_parts_mut(data, size) };
    let mut sorter = RadixSort::new();
    match sorter.sort_u32(data_slice) {
        Ok(_) => CResult::Success,
        Err(_) => CResult::InternalError,
    }
}