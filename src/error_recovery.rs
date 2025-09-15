//! Error Handling System for Zipora
//!
//! This module provides distributed verification macros and error handling patterns
//! using fail-fast philosophy with rich contextual error reporting throughout data structures.


/// Verification macros providing fail-fast error handling with rich contextual information

/// Fatal error macro - equivalent to TERARK_DIE
/// Prints error context and terminates the program immediately
#[macro_export]
macro_rules! zipora_die {
    ($fmt:expr $(, $args:expr)*) => {
        {
            eprintln!("{}:{}: in {}: die: {} !",
                file!(), line!(),
                std::any::type_name::<()>(),
                format!($fmt $(, $args)*));
            std::process::abort();
        }
    };
}

/// Basic verification macro - equivalent to TERARK_VERIFY
/// Checks condition and aborts with context if false
/// In test mode, panics instead of aborting to allow test recovery
#[macro_export]
macro_rules! zipora_verify {
    ($expr:expr) => {
        if !($expr) {
            let msg = format!("{}:{}: verify({}) failed !",
                file!(), line!(), stringify!($expr));
            eprintln!("{}", msg);
            #[cfg(test)]
            panic!("{}", msg);
            #[cfg(not(test))]
            std::process::abort();
        }
    };
    ($expr:expr, $fmt:expr $(, $args:expr)*) => {
        if !($expr) {
            let msg = format!("{}:{}: verify({}) failed: {} !",
                file!(), line!(), stringify!($expr),
                format!($fmt $(, $args)*));
            eprintln!("{}", msg);
            #[cfg(test)]
            panic!("{}", msg);
            #[cfg(not(test))]
            std::process::abort();
        }
    };
}

/// Memory allocation verification with size context
#[macro_export]
macro_rules! zipora_verify_alloc {
    ($ptr:expr, $size:expr) => {
        zipora_verify!(!$ptr.is_null(), "allocation of {} bytes failed", $size);
    };
}

/// Alignment verification for memory operations
#[macro_export]
macro_rules! zipora_verify_aligned {
    ($ptr:expr, $align:expr) => {
        zipora_verify!(($ptr as usize) % $align == 0, 
            "pointer {:p} not aligned to {} bytes", $ptr, $align);
    };
    ($size:expr, $align:expr) => {
        zipora_verify!($size % $align == 0,
            "size {} not aligned to {} bytes", $size, $align);
    };
}

/// Power-of-2 verification for sizes and alignments
#[macro_export]
macro_rules! zipora_verify_pow2 {
    ($val:expr) => {
        zipora_verify!(($val & ($val - 1)) == 0,
            "value {} (0x{:X}) is not a power of 2", $val, $val);
    };
}

/// Comparison verification macros with value display
#[macro_export]
macro_rules! zipora_verify_eq {
    ($x:expr, $y:expr) => {
        {
            let x_val = $x;
            let y_val = $y;
            zipora_verify!(x_val == y_val, "{} != {}", x_val, y_val);
        }
    };
}

#[macro_export]
macro_rules! zipora_verify_ne {
    ($x:expr, $y:expr) => {
        {
            let x_val = $x;
            let y_val = $y;
            zipora_verify!(x_val != y_val, "{} == {}", x_val, y_val);
        }
    };
}

#[macro_export]
macro_rules! zipora_verify_lt {
    ($x:expr, $y:expr) => {
        {
            let x_val = $x;
            let y_val = $y;
            zipora_verify!(x_val < y_val, "{} >= {}", x_val, y_val);
        }
    };
}

#[macro_export]
macro_rules! zipora_verify_le {
    ($x:expr, $y:expr) => {
        {
            let x_val = $x;
            let y_val = $y;
            zipora_verify!(x_val <= y_val, "{} > {}", x_val, y_val);
        }
    };
}

#[macro_export]
macro_rules! zipora_verify_gt {
    ($x:expr, $y:expr) => {
        {
            let x_val = $x;
            let y_val = $y;
            zipora_verify!(x_val > y_val, "{} <= {}", x_val, y_val);
        }
    };
}

#[macro_export]
macro_rules! zipora_verify_ge {
    ($x:expr, $y:expr) => {
        {
            let x_val = $x;
            let y_val = $y;
            zipora_verify!(x_val >= y_val, "{} < {}", x_val, y_val);
        }
    };
}

/// Zero verification - common pattern
#[macro_export]
macro_rules! zipora_verify_ez {
    ($x:expr) => {
        {
            let x_val = $x;
            zipora_verify!(x_val == 0, "expected 0, got {}", x_val);
        }
    };
}

/// Non-null pointer verification
#[macro_export]
macro_rules! zipora_verify_not_null {
    ($ptr:expr) => {
        zipora_verify!(!$ptr.is_null(), "pointer is null");
    };
}

/// Bounds checking with context
#[macro_export]
macro_rules! zipora_verify_bounds {
    ($index:expr, $size:expr) => {
        {
            let idx = $index;
            let sz = $size;
            zipora_verify!(idx < sz, "index {} out of bounds for size {}", idx, sz);
        }
    };
}

/// Range verification 
#[macro_export]
macro_rules! zipora_verify_range {
    ($start:expr, $end:expr, $size:expr) => {
        {
            let s = $start;
            let e = $end;
            let sz = $size;
            zipora_verify!(s <= e, "invalid range: start {} > end {}", s, e);
            zipora_verify!(e <= sz, "range end {} exceeds size {}", e, sz);
        }
    };
}

/// Capacity verification for container operations
#[macro_export]
macro_rules! zipora_verify_capacity {
    ($current:expr, $required:expr, $max:expr) => {
        {
            let curr = $current;
            let req = $required;
            let max_cap = $max;
            zipora_verify!(req <= max_cap, 
                "required capacity {} exceeds maximum {}", req, max_cap);
            zipora_verify!(curr <= req,
                "current size {} exceeds required capacity {}", curr, req);
        }
    };
}

/// System call result verification
#[macro_export]
macro_rules! zipora_verify_syscall {
    ($result:expr, $syscall:expr) => {
        {
            let res = $result;
            zipora_verify!(res == 0, "syscall {} failed with error {}: {}", 
                $syscall, res, std::io::Error::last_os_error());
        }
    };
}

/// Export convenience functions for use in generic contexts
pub fn verify_alignment(ptr: *const u8, align: usize) {
    zipora_verify_aligned!(ptr, align);
}

pub fn verify_power_of_2(val: usize) {
    zipora_verify_pow2!(val);
}

pub fn verify_allocation_success(ptr: *const u8, size: usize) {
    zipora_verify_alloc!(ptr, size);
}

pub fn verify_bounds_check(index: usize, size: usize) {
    zipora_verify_bounds!(index, size);
}

pub fn verify_range_check(start: usize, end: usize, size: usize) {
    zipora_verify_range!(start, end, size);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_macros_success() {
        // These should not panic
        zipora_verify!(true);
        zipora_verify!(1 == 1, "should be equal");
        zipora_verify_eq!(42, 42);
        zipora_verify_ne!(1, 2);
        zipora_verify_lt!(1, 2);
        zipora_verify_le!(1, 1);
        zipora_verify_gt!(2, 1);
        zipora_verify_ge!(2, 2);
        zipora_verify_ez!(0);
        zipora_verify_bounds!(5, 10);
        zipora_verify_range!(2, 8, 10);
        zipora_verify_pow2!(16);
    }

    #[test]
    fn test_verify_alignment() {
        let aligned_ptr = Box::into_raw(Box::new(42u64)) as *const u8;
        verify_alignment(aligned_ptr, 8); // Should not panic for u64 alignment
        unsafe { Box::from_raw(aligned_ptr as *mut u64) }; // Cleanup
    }

    #[test]
    fn test_verify_power_of_2() {
        verify_power_of_2(1);
        verify_power_of_2(2);
        verify_power_of_2(4);
        verify_power_of_2(8);
        verify_power_of_2(1024);
    }

    #[test]
    fn test_verify_bounds() {
        verify_bounds_check(0, 10);
        verify_bounds_check(5, 10);
        verify_bounds_check(9, 10);
    }

    #[test]
    fn test_verify_range() {
        verify_range_check(0, 5, 10);
        verify_range_check(2, 8, 10);
        verify_range_check(0, 10, 10);
    }

    // Note: Tests that would cause abort() are not included as they would terminate the test process
    // These macros are designed to fail-fast in production
}