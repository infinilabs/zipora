//! Main C API interface
//!
//! This module provides the primary C-compatible interface for the zipora library,
//! allowing seamless integration with existing C/C++ codebases.

use super::CResult;
#[cfg(test)]
use super::types::*;
use std::cell::{Cell, RefCell};
#[cfg(test)]
use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Mutex;

pub use super::{algorithms::*, blob_store::*, containers::*};

// Thread-local storage for error messages and callback re-entrancy guard
thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
    static IN_ERROR_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

struct CallbackReentrancyGuard;

impl Drop for CallbackReentrancyGuard {
    fn drop(&mut self) {
        let _ = IN_ERROR_CALLBACK.try_with(|in_cb| in_cb.set(false));
    }
}

// Global error callback storage
static ERROR_CALLBACK: Mutex<Option<unsafe extern "C" fn(*const c_char)>> = Mutex::new(None);

/// Set the last error message for the current thread
pub(crate) fn set_last_error(msg: &str) {
    let cstring = match CString::new(msg) {
        Ok(c) => c,
        Err(_) => CString::new(msg.replace('\0', "\\0")).unwrap_or_default(),
    };

    let _ = LAST_ERROR.try_with(|error| {
        *error.borrow_mut() = Some(cstring.clone());
    });

    // Skip invoking the registered error callback if this thread is already executing one
    // so an error triggered inside the callback cannot infinitely recurse on the stack.
    let already_in_callback = IN_ERROR_CALLBACK
        .try_with(|in_cb| in_cb.replace(true))
        .unwrap_or(true);
    if already_in_callback {
        return;
    }
    let _reentrancy_guard = CallbackReentrancyGuard;

    // Copy the callback function pointer and drop the mutex guard BEFORE invoking it
    // so a re-entrant callback calling `zipora_set_error_callback` cannot deadlock on `ERROR_CALLBACK`.
    let callback = ERROR_CALLBACK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .copied();
    if let Some(cb) = callback {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: callback function pointer was registered by the caller via
            // `zipora_set_error_callback`, and `cstring` outlives this call.
            unsafe {
                cb(cstring.as_ptr());
            }
        }));
    }
}

/// Clear the last error message for the current thread (for testing)
#[cfg(test)]
pub(crate) fn clear_last_error() {
    let _ = LAST_ERROR.try_with(|error| {
        *error.borrow_mut() = None;
    });
}

/// Execute an FFI body, catching any panic so it cannot unwind into the C
/// caller. Unwinding out of an `extern "C"` function aborts the whole process
/// (Rust >= 1.71); this converts a panic into `fallback` plus a last-error
/// message the caller can retrieve via `zipora_last_error`.
///
/// `AssertUnwindSafe` is acceptable here: every entry point re-validates its
/// raw-pointer inputs from scratch, and objects reachable through those
/// pointers are owned by the C caller, so no Rust-side invariant is silently
/// carried across the catch.
pub(crate) fn ffi_guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                s
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.as_str()
            } else {
                "unknown panic payload"
            };
            set_last_error(&format!("panic caught at FFI boundary: {}", msg));
            fallback
        }
    }
}

/// Initialize the zipora library
///
/// # Safety
///
/// This function is safe to call multiple times.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_init() -> CResult {
    ffi_guard(CResult::InternalError, || {
        crate::init();
        CResult::Success
    })
}

/// Get library version string
///
/// # Safety
///
/// The returned null-terminated string has static lifetime and is valid for the
/// lifetime of the program. The caller must not free the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_version() -> *const c_char {
    ffi_guard(ptr::null(), || {
        concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
    })
}

/// Check if SIMD optimizations are available
///
/// # Safety
/// C FFI entry point. Takes no arguments and only reads CPU feature flags, so
/// it is safe to call from any thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_has_simd() -> c_int {
    ffi_guard(0, || {
        if crate::has_simd_support() { 1 } else { 0 }
    })
}

/// Get last error message (thread-local)
///
/// # Safety
///
/// The returned string is valid until the next error occurs on this thread.
/// The caller should not free the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_last_error() -> *const c_char {
    static PANIC_MSG: &[u8] = b"panic caught at FFI boundary\0";
    static NO_ERROR_MSG: &[u8] = b"No error information available\0";
    ffi_guard(PANIC_MSG.as_ptr() as *const c_char, || {
        LAST_ERROR
            .try_with(|error| match error.borrow().as_ref() {
                Some(cstring) => cstring.as_ptr(),
                None => NO_ERROR_MSG.as_ptr() as *const c_char,
            })
            .unwrap_or(NO_ERROR_MSG.as_ptr() as *const c_char)
    })
}

/// Set a custom error callback
///
/// # Safety
///
/// The callback function pointer must be valid for the lifetime of the library usage.
/// The callback will be called from the thread that generates the error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipora_set_error_callback(
    callback: Option<unsafe extern "C" fn(*const c_char)>,
) {
    ffi_guard((), || {
        let mut callback_guard = ERROR_CALLBACK.lock().unwrap_or_else(|e| e.into_inner());
        *callback_guard = callback;
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_version_api() {
        // SAFETY: test calls C API functions with valid parameters, CStr::from_ptr receives valid static string pointer
        unsafe {
            let version_ptr = zipora_version();
            assert!(!version_ptr.is_null());

            let version_cstr = CStr::from_ptr(version_ptr);
            let version_str = version_cstr.to_str().unwrap();
            assert!(!version_str.is_empty());
        }
    }

    #[test]
    fn test_fast_vec_api() {
        // SAFETY: test calls C API functions with valid parameters, slice created from valid pointer returned by fast_vec_data
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
        // SAFETY: test calls C API functions with valid parameters, allocated pointers are deallocated before pool is freed
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
    fn test_blob_store_zero_len() {
        unsafe {
            let store = blob_store_new();
            assert!(!store.is_null());

            let test_data: [u8; 0] = [];
            let mut record_id: u32 = 0;

            let result = blob_store_put(store, test_data.as_ptr(), 0, &mut record_id);
            assert_eq!(result, CResult::Success);

            let mut out_data: *const u8 = std::ptr::null();
            let mut out_size: usize = 0;

            let result = blob_store_get(store, record_id, &mut out_data, &mut out_size);
            assert_eq!(result, CResult::Success);
            assert_eq!(out_size, 0);
            assert!(!out_data.is_null()); // Dangling ptr but not null

            zipora_free_blob_data(out_data as *mut u8, out_size);
            blob_store_free(store);
        }
    }
    #[test]
    fn test_blob_store_api() {
        // SAFETY: test calls C API functions with valid parameters, slice created from valid pointer, blob data freed before store
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
            let retrieved_data = std::slice::from_raw_parts(data_ptr, size);
            assert_eq!(retrieved_data, test_data);

            // Free the blob data
            zipora_free_blob_data(data_ptr as *mut u8, size);

            blob_store_free(store);
        }
    }

    #[test]
    fn test_radix_sort_api() {
        // SAFETY: test calls C API function with valid pointer to mutable array and correct length
        unsafe {
            let mut data = [5u32, 2, 8, 1, 9, 3, 7, 4, 6];
            let result = radix_sort_u32(data.as_mut_ptr(), data.len());

            assert_eq!(result, CResult::Success);
            assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        }
    }

    #[test]
    fn test_init_and_simd() {
        // SAFETY: test calls C API functions with no parameters, both are safe operations
        unsafe {
            assert_eq!(zipora_init(), CResult::Success);

            let has_simd = zipora_has_simd();
            // Should not panic, return 0 or 1
            assert!(has_simd == 0 || has_simd == 1);
        }
    }

    #[test]
    fn test_error_handling() {
        // SAFETY: test calls C API functions, CStr::from_ptr receives valid static/thread-local string pointers
        unsafe {
            // Test initial state - no error
            let error_ptr = zipora_last_error();
            assert!(!error_ptr.is_null());
            let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert_eq!(error_msg, "No error information available");

            // Test setting an error message directly
            set_last_error("Test error message");

            // Check that error message was set
            let error_ptr = zipora_last_error();
            assert!(!error_ptr.is_null());
            let error_msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert_eq!(error_msg, "Test error message");
        }
    }

    static CALLBACK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_error_callback() {
        let _callback_serial = CALLBACK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicBool, Ordering};

        static CALLBACK_CALLED: AtomicBool = AtomicBool::new(false);
        static CALLBACK_MESSAGE: Mutex<Option<String>> = Mutex::new(None);

        unsafe extern "C" fn test_callback(msg: *const c_char) {
            CALLBACK_CALLED.store(true, Ordering::SeqCst);
            // SAFETY: msg pointer is valid, passed from set_last_error via CString::as_ptr
            let c_str = unsafe { CStr::from_ptr(msg) };
            if let Ok(str_slice) = c_str.to_str()
                && let Ok(mut callback_msg) = CALLBACK_MESSAGE.lock() {
                    *callback_msg = Some(str_slice.to_string());
                }
        }

        // SAFETY: test calls C API functions with valid callback function pointer
        unsafe {
            // Clear any previous error state
            clear_last_error();

            // Set the error callback
            zipora_set_error_callback(Some(test_callback));

            // Reset callback state
            CALLBACK_CALLED.store(false, Ordering::SeqCst);
            if let Ok(mut callback_msg) = CALLBACK_MESSAGE.lock() {
                *callback_msg = None;
            }

            // Trigger an error by setting one directly
            set_last_error("Test callback message");

            // Check that callback was called
            assert!(CALLBACK_CALLED.load(Ordering::SeqCst));
            if let Ok(callback_msg) = CALLBACK_MESSAGE.lock() {
                assert!(callback_msg.is_some());
                assert_eq!(callback_msg.as_ref().unwrap(), "Test callback message");
            }

            // Clear the callback
            zipora_set_error_callback(None);

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
    fn test_ffi_guard_catches_panics() {
        clear_last_error();

        // A panic inside an FFI body must be converted into the fallback
        // value instead of unwinding (which would abort a real C caller).
        let result = ffi_guard(CResult::InternalError, || -> CResult {
            panic!("boom in ffi body");
        });
        assert_eq!(result, CResult::InternalError);

        // SAFETY: zipora_last_error returns a valid thread-local or static string
        unsafe {
            let error_ptr = zipora_last_error();
            assert!(!error_ptr.is_null());
            let msg = CStr::from_ptr(error_ptr).to_str().unwrap();
            assert!(
                msg.contains("boom in ffi body"),
                "unexpected last error: {msg}"
            );
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
                // SAFETY: test calls C API functions from spawned thread, CStr::from_ptr receives valid thread-local pointer
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
                    let error_ptr = zipora_last_error();
                    let retrieved_msg = if !error_ptr.is_null() {
                        CStr::from_ptr(error_ptr)
                            .to_str()
                            .unwrap_or("Invalid UTF-8")
                            .to_string()
                    } else {
                        "Null pointer".to_string()
                    };

                    // Store the result
                    let mut results_guard = results_clone.lock().unwrap_or_else(|e| e.into_inner());
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
        let results_guard = results.lock().unwrap_or_else(|e| e.into_inner());
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
    fn test_c11_1_misaligned_and_overflow_pointers_rejected_without_ub() {
        let mut raw = [0u64; 8];
        // SAFETY: `raw` is 8-byte aligned (`[u64; 8]`), so `(raw.as_mut_ptr() as *mut u8).add(1)`
        // has `addr % 8 == 1` and is strictly misaligned for all types with align >= 2.
        unsafe {
            let unaligned = (raw.as_mut_ptr() as *mut u8).add(1);

            // 1. Misaligned u32 buffer and overflow length in radix_sort_u32
            assert_eq!(
                radix_sort_u32(unaligned as *mut u32, 4),
                CResult::InvalidInput
            );
            let mut aligned_u32 = [3u32, 1, 2];
            assert_eq!(
                radix_sort_u32(aligned_u32.as_mut_ptr(), usize::MAX),
                CResult::InvalidInput
            );

            // 2. Misaligned opaque handles across FFI entry points
            fast_vec_free(unaligned as *mut CFastVec);
            assert_eq!(
                fast_vec_push(unaligned as *mut CFastVec, 1),
                CResult::InvalidInput
            );
            assert_eq!(fast_vec_len(unaligned as *const CFastVec), 0);
            assert!(fast_vec_data(unaligned as *const CFastVec).is_null());

            memory_pool_free(unaligned as *mut CMemoryPool);
            assert!(memory_pool_allocate(unaligned as *mut CMemoryPool).is_null());
            assert_eq!(
                memory_pool_deallocate(unaligned as *mut CMemoryPool, raw.as_mut_ptr().cast()),
                CResult::InvalidInput
            );

            // 3. BlobStore: misaligned handles/out-pointers, size > isize::MAX, and (NULL, 0) empty put
            blob_store_free(unaligned as *mut CBlobStore);
            let store = blob_store_new();
            assert!(!store.is_null());

            let mut valid_id: u32 = 0;
            // Misaligned store or record_id out-pointer
            assert_eq!(
                blob_store_put(
                    unaligned as *mut CBlobStore,
                    b"abc".as_ptr(),
                    3,
                    &mut valid_id
                ),
                CResult::InvalidInput
            );
            assert_eq!(
                blob_store_put(store, b"abc".as_ptr(), 3, unaligned as *mut u32),
                CResult::InvalidInput
            );
            // Overflow byte size (> isize::MAX)
            assert_eq!(
                blob_store_put(store, b"abc".as_ptr(), usize::MAX, &mut valid_id),
                CResult::InvalidInput
            );
            // (NULL, 0) is a valid C zero-length buffer (e.g. malloc(0) == NULL)
            assert_eq!(
                blob_store_put(store, ptr::null(), 0, &mut valid_id),
                CResult::Success
            );
            // (NULL, non-zero) must still fail
            assert_eq!(
                blob_store_put(store, ptr::null(), 4, &mut valid_id),
                CResult::InvalidInput
            );

            // Misaligned out-pointers in blob_store_get
            let mut valid_data: *const u8 = ptr::null();
            let mut valid_size: usize = 0;
            assert_eq!(
                blob_store_get(
                    store,
                    valid_id,
                    unaligned as *mut *const u8,
                    &mut valid_size
                ),
                CResult::InvalidInput
            );
            assert_eq!(
                blob_store_get(
                    store,
                    valid_id,
                    &mut valid_data,
                    unaligned as *mut usize
                ),
                CResult::InvalidInput
            );
            blob_store_free(store);

            // 4. SuffixArray: misaligned handle/out-pointers and overflow lengths
            suffix_array_free(unaligned as *mut CSuffixArray);
            assert_eq!(suffix_array_len(unaligned as *const CSuffixArray), 0);
            assert!(suffix_array_new(b"abc".as_ptr(), usize::MAX).is_null());

            let sa = suffix_array_new(b"banana".as_ptr(), 6);
            assert!(!sa.is_null());
            let mut start = 0usize;
            let mut count = 0usize;
            assert_eq!(
                suffix_array_search(
                    sa,
                    b"banana".as_ptr(),
                    6,
                    b"ana".as_ptr(),
                    3,
                    unaligned as *mut usize,
                    &mut count
                ),
                CResult::InvalidInput
            );
            assert_eq!(
                suffix_array_search(
                    sa,
                    b"banana".as_ptr(),
                    usize::MAX,
                    b"ana".as_ptr(),
                    3,
                    &mut start,
                    &mut count
                ),
                CResult::InvalidInput
            );
            suffix_array_free(sa);

            // 5. CBuffer: zero-capacity dangling/arbitrary pointer into_vec and size > isize::MAX slice
            let empty_buf = CBuffer {
                data: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                size: 4,
                capacity: 0,
            };
            assert!(empty_buf.as_slice().is_empty());
            assert!(empty_buf.into_vec().is_empty());
        }
    }

    #[test]
    fn test_c11_2a_cresult_not_found_mapping_and_blob_store_get() {
        assert_eq!(
            CResult::from(Err(crate::ZiporaError::not_found("missing item"))),
            CResult::NotFound
        );
        assert_eq!(
            CResult::from(Err(crate::ZiporaError::invalid_parameter("bad param"))),
            CResult::InvalidInput
        );

        // SAFETY: store is freshly allocated and freed once; out pointers are stack-aligned.
        unsafe {
            let store = blob_store_new();
            assert!(!store.is_null());
            let mut out_ptr: *const u8 = ptr::null();
            let mut out_len: usize = 0;
            clear_last_error();
            assert_eq!(
                blob_store_get(store, 999_999, &mut out_ptr, &mut out_len),
                CResult::NotFound
            );
            blob_store_free(store);
        }
    }

    #[test]
    fn test_c11_2b_set_last_error_preserves_embedded_nul_bytes() {
        clear_last_error();
        set_last_error("prefix\0suffix");
        // SAFETY: zipora_last_error returns a valid NUL-terminated C string pointer on this thread.
        let err_msg = unsafe { CStr::from_ptr(zipora_last_error()).to_str().unwrap() };
        assert_eq!(err_msg, "prefix\\0suffix");
    }

    #[test]
    fn test_c11_2c_error_callback_reentrant_unregister_does_not_deadlock() {
        let _callback_serial = CALLBACK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        unsafe extern "C" fn oneshot_unsubscribing_callback(_msg: *const c_char) {
            // SAFETY: re-entrant unregistration inside callback must not deadlock on ERROR_CALLBACK.
            unsafe {
                zipora_set_error_callback(None);
            }
        }

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // SAFETY: registers valid C callback and passes null handle to trigger set_last_error.
            unsafe {
                zipora_set_error_callback(Some(oneshot_unsubscribing_callback));
                let _ = fast_vec_push(ptr::null_mut(), 7);
            }
            let _ = tx.send(());
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(500)).is_ok(),
            "C11.2: re-entrant zipora_set_error_callback inside error callback deadlocked on ERROR_CALLBACK"
        );
    }

    #[test]
    fn test_s31_r4_error_callback_triggering_error_does_not_recurse() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let _callback_serial = CALLBACK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
        CALL_COUNT.store(0, Ordering::SeqCst);

        unsafe extern "C" fn error_generating_callback(_msg: *const c_char) {
            CALL_COUNT.fetch_add(1, Ordering::SeqCst);
            // SAFETY: intentionally triggers nested FFI error inside callback; must not recurse.
            unsafe {
                let _ = fast_vec_push(ptr::null_mut(), 9);
            }
        }

        // SAFETY: registers valid callback, triggers two top-level errors, and unregisters callback.
        unsafe {
            zipora_set_error_callback(Some(error_generating_callback));
            assert_eq!(fast_vec_push(ptr::null_mut(), 1), CResult::InvalidInput);
            assert_eq!(CALL_COUNT.load(Ordering::SeqCst), 1);
            assert_eq!(fast_vec_push(ptr::null_mut(), 2), CResult::InvalidInput);
            assert_eq!(CALL_COUNT.load(Ordering::SeqCst), 2);
            zipora_set_error_callback(None);
        }
    }
}
