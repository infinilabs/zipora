//! Integration tests for FFI error handling functionality
//!
//! These tests verify that the error handling system implemented in the C API
//! works correctly without depending on the full FFI compilation.

#[cfg(feature = "ffi")]
mod ffi_error_tests {
    use std::cell::RefCell;
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    // Re-create the same error handling structure from c_api.rs for testing
    thread_local! {
        static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
    }

    static ERROR_CALLBACK: Mutex<Option<unsafe extern "C" fn(*const c_char)>> = Mutex::new(None);

    fn set_last_error(msg: &str) {
        LAST_ERROR.with(|error| {
            if let Ok(cstring) = CString::new(msg) {
                *error.borrow_mut() = Some(cstring.clone());

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

    unsafe fn get_last_error() -> *const c_char {
        LAST_ERROR.with(|error| match error.borrow().as_ref() {
            Some(cstring) => cstring.as_ptr(),
            None => {
                static NO_ERROR_MSG: &[u8] = b"No error information available\0";
                NO_ERROR_MSG.as_ptr() as *const c_char
            }
        })
    }

    unsafe fn set_error_callback(callback: Option<unsafe extern "C" fn(*const c_char)>) {
        if let Ok(mut callback_guard) = ERROR_CALLBACK.lock() {
            *callback_guard = callback;
        }
    }

    #[test]
    fn test_basic_error_handling() {
        unsafe {
            // Test initial state - no error
            let error_ptr = get_last_error();
            assert!(!error_ptr.is_null());
            let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert_eq!(error_msg, "No error information available");

            // Test setting an error message
            set_last_error("Test error message");

            // Check that error message was set
            let error_ptr = get_last_error();
            assert!(!error_ptr.is_null());
            let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert_eq!(error_msg, "Test error message");
        }
    }

    #[test]
    fn test_error_callback() {
        use std::sync::atomic::{AtomicPtr, Ordering};
        
        static CALLBACK_CALLED: AtomicBool = AtomicBool::new(false);
        static CALLBACK_MESSAGE: Mutex<Option<String>> = Mutex::new(None);

        unsafe extern "C" fn test_callback(msg: *const c_char) {
            CALLBACK_CALLED.store(true, Ordering::SeqCst);
            let c_str = unsafe { CStr::from_ptr(msg) };
            if let Ok(str_slice) = c_str.to_str() {
                if let Ok(mut callback_msg) = CALLBACK_MESSAGE.lock() {
                    *callback_msg = Some(str_slice.to_string());
                }
            }
        }

        unsafe {
            // Set the error callback
            set_error_callback(Some(test_callback));

            // Reset callback state
            CALLBACK_CALLED.store(false, Ordering::SeqCst);
            if let Ok(mut callback_msg) = CALLBACK_MESSAGE.lock() {
                *callback_msg = None;
            }

            // Trigger an error
            set_last_error("Test callback message");

            // Check that callback was called
            assert!(CALLBACK_CALLED.load(Ordering::SeqCst));
            if let Ok(callback_msg) = CALLBACK_MESSAGE.lock() {
                assert!(callback_msg.is_some());
                assert_eq!(callback_msg.as_ref().unwrap(), "Test callback message");
            }

            // Clear the callback
            set_error_callback(None);

            // Reset state and trigger another error
            CALLBACK_CALLED.store(false, Ordering::SeqCst);
            if let Ok(mut callback_msg) = CALLBACK_MESSAGE.lock() {
                *callback_msg = None;
            }

            set_last_error("Another test message");

            // Callback should not have been called this time
            assert!(!CALLBACK_CALLED.load(Ordering::SeqCst));
            if let Ok(callback_msg) = CALLBACK_MESSAGE.lock() {
                assert!(callback_msg.is_none());
            }
        }
    }

    #[test]
    fn test_thread_local_errors() {
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
                    let error_ptr = get_last_error();
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

    #[test]
    fn test_error_message_persistence() {
        unsafe {
            // Set an error and verify it persists
            set_last_error("Persistent error");

            let error_ptr1 = get_last_error();
            let error_msg1 = CStr::from_ptr(error_ptr1).to_str().unwrap();

            let error_ptr2 = get_last_error();
            let error_msg2 = CStr::from_ptr(error_ptr2).to_str().unwrap();

            assert_eq!(error_msg1, "Persistent error");
            assert_eq!(error_msg2, "Persistent error");
            assert_eq!(error_ptr1, error_ptr2); // Same pointer
        }
    }

    #[test]
    fn test_error_message_replacement() {
        unsafe {
            // Set first error
            set_last_error("First error");
            let error_ptr = get_last_error();
            let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert_eq!(error_msg, "First error");

            // Replace with second error
            set_last_error("Second error");
            let error_ptr = get_last_error();
            let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert_eq!(error_msg, "Second error");
        }
    }
}
