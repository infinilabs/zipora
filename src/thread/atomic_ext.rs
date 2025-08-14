//! Atomic Operations Framework
//!
//! Comprehensive lock-free programming utilities providing enhanced atomic
//! operations, platform-specific optimizations, and safe atomic casting.

use std::sync::atomic::{
    AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize,
    AtomicI8, AtomicI16, AtomicI32, AtomicI64, AtomicIsize,
    AtomicBool, AtomicPtr, Ordering
};
use std::ptr;
use crate::error::Result;

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
            fn cas_weak(&self, expected: $value_type, desired: $value_type) -> std::result::Result<$value_type, $value_type> {
                self.compare_exchange_weak(expected, desired, Ordering::AcqRel, Ordering::Acquire)
            }
            
            #[inline]
            fn cas_strong(&self, expected: $value_type, desired: $value_type) -> std::result::Result<$value_type, $value_type> {
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
                    self.compare_exchange_weak(current, new_val, order, order).is_ok()
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
    use super::*;
    use std::arch::x86_64::*;

    /// x86_64-specific optimized CAS for u64
    #[inline]
    pub unsafe fn cas_weak_u64_asm(ptr: *mut u64, expected: u64, desired: u64) -> bool {
        let result: u8;
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
        unsafe {
            std::arch::asm!("pause", options(nomem, nostack, preserves_flags));
        }
    }

    /// Memory fence operations
    #[inline]
    pub fn mfence() {
        unsafe {
            std::arch::asm!("mfence", options(nostack, preserves_flags));
        }
    }

    #[inline]
    pub fn lfence() {
        unsafe {
            std::arch::asm!("lfence", options(nostack, preserves_flags));
        }
    }

    #[inline]
    pub fn sfence() {
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
        unsafe {
            std::arch::asm!("yield", options(nomem, nostack, preserves_flags));
        }
    }

    /// Data memory barrier
    #[inline]
    pub fn dmb() {
        unsafe {
            std::arch::asm!("dmb sy", options(nostack, preserves_flags));
        }
    }

    /// Data synchronization barrier
    #[inline]
    pub fn dsb() {
        unsafe {
            std::arch::asm!("dsb sy", options(nostack, preserves_flags));
        }
    }

    /// Instruction synchronization barrier
    #[inline]
    pub fn isb() {
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
                // Safe because atomic types have same representation
                unsafe { &*(self as *const $type as *const $atomic) }
            }
            
            #[inline]
            fn as_atomic_mut(&mut self) -> &mut Self::Atomic {
                // Safe because atomic types have same representation
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
        unsafe { &*(self as *const *mut T as *const AtomicPtr<T>) }
    }
    
    #[inline]
    fn as_atomic_mut(&mut self) -> &mut Self::Atomic {
        unsafe { &mut *(self as *mut *mut T as *mut AtomicPtr<T>) }
    }
}

/// Lock-free linked list node
#[repr(C)]
pub struct AtomicNode<T> {
    data: T,
    next: AtomicPtr<AtomicNode<T>>,
}

impl<T> AtomicNode<T> {
    /// Create a new atomic node
    pub fn new(data: T) -> Self {
        Self {
            data,
            next: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Get the data
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Get mutable data
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Get next node pointer
    pub fn next(&self) -> *mut AtomicNode<T> {
        self.next.load(Ordering::Acquire)
    }

    /// Set next node pointer
    pub fn set_next(&self, next: *mut AtomicNode<T>) {
        self.next.store(next, Ordering::Release);
    }

    /// Compare-and-swap next pointer
    pub fn cas_next(
        &self,
        expected: *mut AtomicNode<T>,
        desired: *mut AtomicNode<T>,
    ) -> std::result::Result<*mut AtomicNode<T>, *mut AtomicNode<T>> {
        self.next.compare_exchange_weak(expected, desired, Ordering::AcqRel, Ordering::Acquire)
    }
}

unsafe impl<T: Send> Send for AtomicNode<T> {}
unsafe impl<T: Send + Sync> Sync for AtomicNode<T> {}

/// Lock-free stack using atomic operations
pub struct AtomicStack<T> {
    head: AtomicPtr<AtomicNode<T>>,
    size: AtomicUsize,
}

impl<T> AtomicStack<T> {
    /// Create a new atomic stack
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            size: AtomicUsize::new(0),
        }
    }

    /// Push an item onto the stack
    pub fn push(&self, data: T) {
        let new_node = Box::into_raw(Box::new(AtomicNode::new(data)));
        
        loop {
            let head = self.head.load(Ordering::Acquire);
            unsafe { (*new_node).set_next(head) };
            
            if self.head.compare_exchange_weak(head, new_node, Ordering::Release, Ordering::Relaxed).is_ok() {
                self.size.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    /// Pop an item from the stack
    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            if head.is_null() {
                return None;
            }
            
            let next = unsafe { (*head).next() };
            
            if self.head.compare_exchange_weak(head, next, Ordering::Release, Ordering::Relaxed).is_ok() {
                self.size.fetch_sub(1, Ordering::Relaxed);
                let data = unsafe { Box::from_raw(head).data };
                return Some(data);
            }
        }
    }

    /// Check if stack is empty
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire).is_null()
    }

    /// Get approximate size
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }
}

impl<T> Default for AtomicStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Send> Send for AtomicStack<T> {}
unsafe impl<T: Send> Sync for AtomicStack<T> {}

impl<T> Drop for AtomicStack<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
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
    fn test_atomic_stack() {
        let stack = Arc::new(AtomicStack::new());
        
        // Test single-threaded operations
        stack.push(1);
        stack.push(2);
        stack.push(3);
        
        assert_eq!(stack.len(), 3);
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
        assert!(stack.is_empty());
    }

    #[test]
    fn test_atomic_stack_concurrent() {
        let stack = Arc::new(AtomicStack::new());
        let num_threads = 4;
        let items_per_thread = 1000;
        
        // Push items concurrently
        let push_handles: Vec<_> = (0..num_threads).map(|t| {
            let stack = Arc::clone(&stack);
            thread::spawn(move || {
                for i in 0..items_per_thread {
                    stack.push(t * items_per_thread + i);
                }
            })
        }).collect();
        
        for handle in push_handles {
            handle.join().unwrap();
        }
        
        assert_eq!(stack.len(), num_threads * items_per_thread);
        
        // Pop items concurrently
        let pop_handles: Vec<_> = (0..num_threads).map(|_| {
            let stack = Arc::clone(&stack);
            thread::spawn(move || {
                let mut count = 0;
                while stack.pop().is_some() {
                    count += 1;
                }
                count
            })
        }).collect();
        
        let total_popped: usize = pop_handles.into_iter()
            .map(|h| h.join().unwrap())
            .sum();
        
        assert_eq!(total_popped, num_threads * items_per_thread);
        assert!(stack.is_empty());
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