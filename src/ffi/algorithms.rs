//! C FFI bindings for algorithm types

// Algorithm FFI functions are defined in c_api.rs to avoid duplication

#[cfg(test)]
mod tests {
    use crate::ffi::c_api::*;
    use crate::ffi::CResult;

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
            assert_eq!(radix_sort_u32(std::ptr::null_mut(), 5), CResult::InvalidInput);
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
