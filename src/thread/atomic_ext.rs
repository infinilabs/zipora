//! Atomic Operations Framework
//!
//! Comprehensive lock-free programming utilities providing enhanced atomic
//! operations, platform-specific optimizations, and safe atomic casting.

use std::ptr;
use std::sync::atomic::{
    AtomicBool, AtomicI8, AtomicI16, AtomicI32, AtomicI64, AtomicIsize, AtomicPtr, AtomicU8,
    AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};

/// Extended atomic operations trait
pub trait AtomicExt<T> {
    /// Atomic maximum operation - updates value to max(current, val)
    fn atomic_maximize(&self, val: T, order: Ordering) -> T;

    /// Atomic minimum operation - updates value to min(current, val)
    fn atomic_minimize(&self, val: T, order: Ordering) -> T;

    /// Weak compare-and-swap with optimized ordering
    fn cas_weak(&self, expected: T, desired: T) -> std::result::Result<T, T>;

    /// Strong compare-and-swap with optimized ordering
    fn cas_strong(&self, expected: T, desired: T) -> std::result::Result<T, T>;

    /// Atomic add and return old value
    fn fetch_add_acq_rel(&self, val: T) -> T;

    /// Atomic subtract and return old value
    fn fetch_sub_acq_rel(&self, val: T) -> T;

    /// Conditional atomic update with predicate
    fn update_if<F>(&self, condition: F, new_val: T, order: Ordering) -> bool
    where
        F: Fn(T) -> bool;
}

macro_rules! impl_atomic_ext {
    ($atomic_type:ty, $value_type:ty) => {
        impl AtomicExt<$value_type> for $atomic_type {
            #[inline]
            fn atomic_maximize(&self, val: $value_type, order: Ordering) -> $value_type {
                let mut current = self.load(order);
                loop {
                    let new_val = current.max(val);
                    if new_val == current {
                        return current; // No change needed
                    }

                    match self.compare_exchange_weak(current, new_val, order, order) {
                        Ok(_) => return new_val,
                        Err(actual) => current = actual,
                    }
                }
            }

            #[inline]
            fn atomic_minimize(&self, val: $value_type, order: Ordering) -> $value_type {
                let mut current = self.load(order);
                loop {
                    let new_val = current.min(val);
                    if new_val == current {
                        return current; // No change needed
                    }

                    match self.compare_exchange_weak(current, new_val, order, order) {
                        Ok(_) => return new_val,
                        Err(actual) => current = actual,
                    }
                }
            }

            #[inline]
            fn cas_weak(
                &self,
                expected: $value_type,
                desired: $value_type,
            ) -> std::result::Result<$value_type, $value_type> {
                self.compare_exchange_weak(expected, desired, Ordering::AcqRel, Ordering::Acquire)
            }

            #[inline]
            fn cas_strong(
                &self,
                expected: $value_type,
                desired: $value_type,
            ) -> std::result::Result<$value_type, $value_type> {
                self.compare_exchange(expected, desired, Ordering::AcqRel, Ordering::Acquire)
            }

            #[inline]
            fn fetch_add_acq_rel(&self, val: $value_type) -> $value_type {
                self.fetch_add(val, Ordering::AcqRel)
            }

            #[inline]
            fn fetch_sub_acq_rel(&self, val: $value_type) -> $value_type {
                self.fetch_sub(val, Ordering::AcqRel)
            }

            #[inline]
            fn update_if<F>(&self, condition: F, new_val: $value_type, order: Ordering) -> bool
            where
                F: Fn($value_type) -> bool,
            {
                let current = self.load(order);
                if condition(current) {
                    self.compare_exchange_weak(current, new_val, order, order)
                        .is_ok()
                } else {
                    false
                }
            }
        }
    };
}

// Implement for all atomic integer types
impl_atomic_ext!(AtomicU8, u8);
impl_atomic_ext!(AtomicU16, u16);
impl_atomic_ext!(AtomicU32, u32);
impl_atomic_ext!(AtomicU64, u64);
impl_atomic_ext!(AtomicUsize, usize);
impl_atomic_ext!(AtomicI8, i8);
impl_atomic_ext!(AtomicI16, i16);
impl_atomic_ext!(AtomicI32, i32);
impl_atomic_ext!(AtomicI64, i64);
impl_atomic_ext!(AtomicIsize, isize);

/// Platform-specific atomic optimizations
#[cfg(target_arch = "x86_64")]
pub mod x86_64_optimized {
    // x86_64 optimizations (currently unused but available for future implementations)
    // use super::*;
    // use std::arch::x86_64::*;

    /// x86_64-specific optimized CAS for u64
    #[inline]
    pub unsafe fn cas_weak_u64_asm(ptr: *mut u64, expected: u64, desired: u64) -> bool {
        let result: u8;
        // SAFETY: asm: atomic compare-exchange u64, rax/result clobbered, ptr must be aligned and valid u64
        unsafe {
            std::arch::asm!(
                "lock cmpxchg {desired:r}, ({ptr})",
                "sete {result}",
                ptr = in(reg) ptr,
                desired = in(reg) desired,
                result = out(reg_byte) result,
                in("rax") expected,
                options(nostack, preserves_flags)
            );
        }
        result != 0
    }

    /// 128-bit CAS using cmpxchg16b instruction
    #[target_feature(enable = "cmpxchg16b")]
    pub unsafe fn cas_weak_u128(ptr: *mut u128, expected: u128, desired: u128) -> bool {
        let result: u8;
        let expected_lo = expected as u64;
        let expected_hi = (expected >> 64) as u64;
        let desired_lo = desired as u64;
        let desired_hi = (desired >> 64) as u64;

        // SAFETY: asm: 128-bit atomic compare-exchange, rax/rbx/rcx/rdx/result clobbered, ptr must be 16-byte aligned and valid u128
        unsafe {
            std::arch::asm!(
                "mov {desired_lo}, %rbx",
                "lock cmpxchg16b ({ptr})",
                "sete {result}",
                ptr = in(reg) ptr,
                result = out(reg_byte) result,
                desired_lo = in(reg) desired_lo,
                in("rax") expected_lo,
                in("rdx") expected_hi,
                in("rcx") desired_hi,
                options(nostack, preserves_flags, att_syntax)
            );
        }
        result != 0
    }

    /// Optimized atomic increment using xadd
    #[inline]
    pub unsafe fn atomic_increment_u64(ptr: *mut u64) -> u64 {
        let result: u64;
        // SAFETY: asm: atomic fetch-add u64, result clobbered, ptr must be aligned and valid u64
        unsafe {
            std::arch::asm!(
                "mov {result:r}, 1",
                "lock xadd ({ptr}), {result:r}",
                ptr = in(reg) ptr,
                result = out(reg) result,
                options(nostack, preserves_flags)
            );
        }
        result
    }

    /// Optimized atomic exchange
    #[inline]
    pub unsafe fn atomic_exchange_u64(ptr: *mut u64, val: u64) -> u64 {
        let result: u64;
        // SAFETY: asm: atomic exchange u64, result clobbered, ptr must be aligned and valid u64
        unsafe {
            std::arch::asm!(
                "xchg ({ptr}), {val:r}",
                ptr = in(reg) ptr,
                val = inout(reg) val => result,
                options(nostack, preserves_flags)
            );
        }
        result
    }

    /// Pause instruction for spin loops
    #[inline]
    pub fn pause() {
        // SAFETY: asm: pause instruction (spin-loop hint), no registers clobbered, no memory effects
        unsafe {
            std::arch::asm!("pause", options(nomem, nostack, preserves_flags));
        }
    }

    /// Memory fence operations
    #[inline]
    pub fn mfence() {
        // SAFETY: asm: full memory fence, no registers clobbered, orders all loads/stores
        unsafe {
            std::arch::asm!("mfence", options(nostack, preserves_flags));
        }
    }

    #[inline]
    pub fn lfence() {
        // SAFETY: asm: load fence, no registers clobbered, orders all loads
        unsafe {
            std::arch::asm!("lfence", options(nostack, preserves_flags));
        }
    }

    #[inline]
    pub fn sfence() {
        // SAFETY: asm: store fence, no registers clobbered, orders all stores
        unsafe {
            std::arch::asm!("sfence", options(nostack, preserves_flags));
        }
    }
}

/// ARM-specific optimizations
#[cfg(target_arch = "aarch64")]
pub mod aarch64_optimized {
    use super::*;

    /// ARM yield instruction for spin loops
    #[inline]
    pub fn yield_now() {
        // SAFETY: asm: yield instruction (spin-loop hint), no registers clobbered, no memory effects
        unsafe {
            std::arch::asm!("yield", options(nomem, nostack, preserves_flags));
        }
    }

    /// Data memory barrier
    #[inline]
    pub fn dmb() {
        // SAFETY: asm: data memory barrier, no registers clobbered, orders all memory operations
        unsafe {
            std::arch::asm!("dmb sy", options(nostack, preserves_flags));
        }
    }

    /// Data synchronization barrier
    #[inline]
    pub fn dsb() {
        // SAFETY: asm: data synchronization barrier, no registers clobbered, completes all memory operations
        unsafe {
            std::arch::asm!("dsb sy", options(nostack, preserves_flags));
        }
    }

    /// Instruction synchronization barrier
    #[inline]
    pub fn isb() {
        // SAFETY: asm: instruction synchronization barrier, no registers clobbered, flushes pipeline
        unsafe {
            std::arch::asm!("isb", options(nostack, preserves_flags));
        }
    }
}

/// Safe atomic reinterpret casting
pub trait AsAtomic<T> {
    type Atomic;

    /// Get atomic reference to the value
    fn as_atomic(&self) -> &Self::Atomic;

    /// Get mutable atomic reference to the value
    fn as_atomic_mut(&mut self) -> &mut Self::Atomic;
}

macro_rules! impl_as_atomic {
    ($type:ty, $atomic:ty) => {
        impl AsAtomic<$type> for $type {
            type Atomic = $atomic;

            #[inline]
            fn as_atomic(&self) -> &Self::Atomic {
                // SAFETY: atomic types have identical representation to their primitive types, pointer is properly aligned and valid
                unsafe { &*(self as *const $type as *const $atomic) }
            }

            #[inline]
            fn as_atomic_mut(&mut self) -> &mut Self::Atomic {
                // SAFETY: atomic types have identical representation to their primitive types, pointer is properly aligned and valid
                unsafe { &mut *(self as *mut $type as *mut $atomic) }
            }
        }
    };
}

impl_as_atomic!(u8, AtomicU8);
impl_as_atomic!(u16, AtomicU16);
impl_as_atomic!(u32, AtomicU32);
impl_as_atomic!(u64, AtomicU64);
impl_as_atomic!(usize, AtomicUsize);
impl_as_atomic!(i8, AtomicI8);
impl_as_atomic!(i16, AtomicI16);
impl_as_atomic!(i32, AtomicI32);
impl_as_atomic!(i64, AtomicI64);
impl_as_atomic!(isize, AtomicIsize);
impl_as_atomic!(bool, AtomicBool);

/// Atomic pointer operations extensions
impl<T> AsAtomic<*mut T> for *mut T {
    type Atomic = AtomicPtr<T>;

    #[inline]
    fn as_atomic(&self) -> &Self::Atomic {
        // SAFETY: AtomicPtr<T> has identical representation to *mut T, pointer is properly aligned and valid
        unsafe { &*(self as *const *mut T as *const AtomicPtr<T>) }
    }

    #[inline]
    fn as_atomic_mut(&mut self) -> &mut Self::Atomic {
        // SAFETY: AtomicPtr<T> has identical representation to *mut T, pointer is properly aligned and valid
        unsafe { &mut *(self as *mut *mut T as *mut AtomicPtr<T>) }
    }
}

/// Atomic bit operations
pub trait AtomicBitOps {
    /// Set bit atomically
    fn set_bit(&self, bit: usize) -> bool;

    /// Clear bit atomically
    fn clear_bit(&self, bit: usize) -> bool;

    /// Toggle bit atomically
    fn toggle_bit(&self, bit: usize) -> bool;

    /// Test bit
    fn test_bit(&self, bit: usize) -> bool;

    /// Find first set bit
    fn find_first_set(&self) -> Option<usize>;
}

macro_rules! impl_atomic_bit_ops {
    ($atomic_type:ty, $value_type:ty, $bits:expr) => {
        impl AtomicBitOps for $atomic_type {
            fn set_bit(&self, bit: usize) -> bool {
                if bit >= $bits {
                    return false;
                }
                let mask = 1 << bit;
                let old = self.fetch_or(mask, Ordering::AcqRel);
                (old & mask) != 0
            }

            fn clear_bit(&self, bit: usize) -> bool {
                if bit >= $bits {
                    return false;
                }
                let mask = 1 << bit;
                let old = self.fetch_and(!mask, Ordering::AcqRel);
                (old & mask) != 0
            }

            fn toggle_bit(&self, bit: usize) -> bool {
                if bit >= $bits {
                    return false;
                }
                let mask = 1 << bit;
                let old = self.fetch_xor(mask, Ordering::AcqRel);
                (old & mask) != 0
            }

            fn test_bit(&self, bit: usize) -> bool {
                if bit >= $bits {
                    return false;
                }
                let mask = 1 << bit;
                (self.load(Ordering::Acquire) & mask) != 0
            }

            fn find_first_set(&self) -> Option<usize> {
                let val = self.load(Ordering::Acquire);
                if val == 0 {
                    None
                } else {
                    Some(val.trailing_zeros() as usize)
                }
            }
        }
    };
}

impl_atomic_bit_ops!(AtomicU8, u8, 8);
impl_atomic_bit_ops!(AtomicU16, u16, 16);
impl_atomic_bit_ops!(AtomicU32, u32, 32);
impl_atomic_bit_ops!(AtomicU64, u64, 64);
impl_atomic_bit_ops!(AtomicUsize, usize, std::mem::size_of::<usize>() * 8);

/// Optimized spin loop with platform-specific hints
#[inline]
pub fn spin_loop_hint() {
    #[cfg(target_arch = "x86_64")]
    x86_64_optimized::pause();

    #[cfg(target_arch = "aarch64")]
    aarch64_optimized::yield_now();

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    std::hint::spin_loop();
}

/// Memory ordering utilities
pub mod memory_ordering {
    use super::*;

    /// Full memory barrier
    #[inline]
    pub fn full_barrier() {
        #[cfg(target_arch = "x86_64")]
        x86_64_optimized::mfence();

        #[cfg(target_arch = "aarch64")]
        aarch64_optimized::dmb();

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Load barrier
    #[inline]
    pub fn load_barrier() {
        #[cfg(target_arch = "x86_64")]
        x86_64_optimized::lfence();

        #[cfg(target_arch = "aarch64")]
        aarch64_optimized::dmb();

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        std::sync::atomic::fence(Ordering::Acquire);
    }

    /// Store barrier
    #[inline]
    pub fn store_barrier() {
        #[cfg(target_arch = "x86_64")]
        x86_64_optimized::sfence();

        #[cfg(target_arch = "aarch64")]
        aarch64_optimized::dmb();

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        std::sync::atomic::fence(Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_atomic_maximize() {
        let atomic = AtomicU32::new(10);

        assert_eq!(atomic.atomic_maximize(5, Ordering::Relaxed), 10);
        assert_eq!(atomic.load(Ordering::Relaxed), 10);

        assert_eq!(atomic.atomic_maximize(15, Ordering::Relaxed), 15);
        assert_eq!(atomic.load(Ordering::Relaxed), 15);
    }

    #[test]
    fn test_atomic_minimize() {
        let atomic = AtomicU32::new(10);

        assert_eq!(atomic.atomic_minimize(15, Ordering::Relaxed), 10);
        assert_eq!(atomic.load(Ordering::Relaxed), 10);

        assert_eq!(atomic.atomic_minimize(5, Ordering::Relaxed), 5);
        assert_eq!(atomic.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_as_atomic() {
        let mut value = 42u32;
        let atomic = value.as_atomic_mut();

        assert_eq!(atomic.load(Ordering::Relaxed), 42);
        atomic.store(100, Ordering::Relaxed);
        assert_eq!(value, 100);
    }

    #[test]
    fn test_atomic_bit_ops() {
        let atomic = AtomicU32::new(0);

        // Set bits
        assert!(!atomic.set_bit(0)); // Was 0
        assert!(atomic.test_bit(0));
        assert!(atomic.set_bit(0)); // Now 1

        assert!(!atomic.set_bit(5));
        assert!(atomic.test_bit(5));

        // Clear bits
        assert!(atomic.clear_bit(0)); // Was 1
        assert!(!atomic.test_bit(0));
        assert!(!atomic.clear_bit(0)); // Now 0

        // Toggle bits
        assert!(!atomic.toggle_bit(3)); // Was 0
        assert!(atomic.test_bit(3));
        assert!(atomic.toggle_bit(3)); // Now 0

        // Find first set
        atomic.store(0b1010, Ordering::Relaxed);
        assert_eq!(atomic.find_first_set(), Some(1));

        atomic.store(0, Ordering::Relaxed);
        assert_eq!(atomic.find_first_set(), None);
    }

    #[test]
    fn test_update_if() {
        let atomic = AtomicU32::new(10);

        // Update if even
        assert!(atomic.update_if(|x| x % 2 == 0, 20, Ordering::Relaxed));
        assert_eq!(atomic.load(Ordering::Relaxed), 20);

        // Don't update if odd
        assert!(!atomic.update_if(|x| x % 2 == 1, 30, Ordering::Relaxed));
        assert_eq!(atomic.load(Ordering::Relaxed), 20);
    }
}
