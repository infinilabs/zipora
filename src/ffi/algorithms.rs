//! C FFI bindings for algorithm types

use super::CResult;
use super::c_api::{ffi_guard, set_last_error};
use super::types::CSuffixArray;
use std::ptr;

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
            Ok(sa) => Box::into_raw(Box::new(CSuffixArray::new(sa))),
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
        if !sa.is_null() {
            if !sa.is_aligned() {
                set_last_error("SuffixArray pointer is misaligned");
                return;
            }
            // SAFETY: pointer is non-null and aligned (checked above), created by suffix_array_new via Box::into_raw, not previously freed
            let _sa = unsafe { Box::from_raw(sa) };
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
        if sa.is_null() || !sa.is_aligned() {
            return 0;
        }

        // SAFETY: pointer is non-null and aligned (checked above), points to initialized CSuffixArray
        let suffix_array = unsafe { &*sa };
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
        if sa.is_null()
            || !sa.is_aligned()
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

        // SAFETY: sa is non-null and aligned (checked above), points to initialized CSuffixArray
        let suffix_array = unsafe { &*sa };
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
            set_last_error(
                "radix_sort_u32 data pointer is null, misaligned, empty, or exceeds isize::MAX",
            );
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

#[cfg(test)]
mod tests {
    use crate::ffi::CResult;
    use crate::ffi::c_api::*;

    #[test]
    fn test_radix_sort_basic() {
        unsafe {
            let mut data = vec![9u32, 3, 7, 1, 5, 8, 2, 6, 4, 0];
            let result = radix_sort_u32(data.as_mut_ptr(), data.len());
            assert_eq!(result, CResult::Success);
            assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        }
    }

    #[test]
    fn test_radix_sort_already_sorted() {
        unsafe {
            let mut data = vec![1u32, 2, 3, 4, 5];
            let result = radix_sort_u32(data.as_mut_ptr(), data.len());
            assert_eq!(result, CResult::Success);
            assert_eq!(data, vec![1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn test_radix_sort_null_safety() {
        unsafe {
            assert_eq!(
                radix_sort_u32(std::ptr::null_mut(), 5),
                CResult::InvalidInput
            );
            let mut data = vec![1u32];
            assert_eq!(radix_sort_u32(data.as_mut_ptr(), 0), CResult::InvalidInput);
        }
    }

    #[test]
    fn test_suffix_array_lifecycle() {
        let text = b"banana";
        unsafe {
            let sa = suffix_array_new(text.as_ptr(), text.len());
            assert!(!sa.is_null());
            assert_eq!(suffix_array_len(sa), text.len());

            let pattern = b"ana";
            let mut start = 0usize;
            let mut count = 0usize;
            let result = suffix_array_search(
                sa,
                text.as_ptr(),
                text.len(),
                pattern.as_ptr(),
                pattern.len(),
                &mut start,
                &mut count,
            );
            assert_eq!(result, CResult::Success);
            assert!(count > 0, "\"ana\" should be found in \"banana\"");

            suffix_array_free(sa);
        }
    }

    #[test]
    fn test_suffix_array_search_not_found() {
        let text = b"hello world";
        unsafe {
            let sa = suffix_array_new(text.as_ptr(), text.len());
            assert!(!sa.is_null());

            let pattern = b"xyz";
            let mut start = 0usize;
            let mut count = 0usize;
            let result = suffix_array_search(
                sa,
                text.as_ptr(),
                text.len(),
                pattern.as_ptr(),
                pattern.len(),
                &mut start,
                &mut count,
            );
            assert_eq!(result, CResult::Success);
            assert_eq!(count, 0);

            suffix_array_free(sa);
        }
    }

    #[test]
    fn test_suffix_array_null_safety() {
        unsafe {
            assert!(suffix_array_new(std::ptr::null(), 5).is_null());
            assert!(suffix_array_new(b"hello".as_ptr(), 0).is_null());
            suffix_array_free(std::ptr::null_mut());
            assert_eq!(suffix_array_len(std::ptr::null()), 0);
        }
    }
}
