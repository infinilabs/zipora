//! C FFI bindings for container types

// FFI functions for containers are defined in c_api.rs to avoid duplication

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
}
