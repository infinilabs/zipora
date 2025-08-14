//! Thread and synchronization utilities
//!
//! This module provides advanced synchronization primitives including:
//! - Linux Futex Integration for high-performance synchronization
//! - Instance-Specific Thread-Local Storage management
//! - Atomic Operations Framework for lock-free programming

#[cfg(target_os = "linux")]
pub mod linux_futex;

pub mod instance_tls;
pub mod atomic_ext;

#[cfg(target_os = "linux")]
pub use linux_futex::*;
pub use instance_tls::*;
pub use atomic_ext::*;

use crate::error::Result;

/// Cross-platform synchronization trait
pub trait PlatformSync {
    /// Wait on a futex (or equivalent)
    fn futex_wait(addr: &std::sync::atomic::AtomicU32, val: u32, timeout: Option<std::time::Duration>) -> Result<()>;
    
    /// Wake waiters on a futex (or equivalent)
    fn futex_wake(addr: &std::sync::atomic::AtomicU32, count: u32) -> Result<usize>;
}

/// Platform-specific implementation selector
#[cfg(target_os = "linux")]
pub type DefaultPlatformSync = crate::thread::linux_futex::LinuxFutex;

#[cfg(not(target_os = "linux"))]
pub type DefaultPlatformSync = crate::thread::FallbackSync;

#[cfg(not(target_os = "linux"))]
pub struct FallbackSync;

#[cfg(not(target_os = "linux"))]
impl PlatformSync for FallbackSync {
    fn futex_wait(_addr: &std::sync::atomic::AtomicU32, _val: u32, _timeout: Option<std::time::Duration>) -> Result<()> {
        // Fallback to parking_lot or standard library
        std::thread::yield_now();
        Ok(())
    }
    
    fn futex_wake(_addr: &std::sync::atomic::AtomicU32, _count: u32) -> Result<usize> {
        Ok(0)
    }
}