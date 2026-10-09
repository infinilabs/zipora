//! Thread and synchronization utilities
//!
//! This module provides advanced synchronization primitives including:
//! - Linux Futex Integration for high-performance synchronization
//! - Instance-Specific Thread-Local Storage management
//! - Atomic Operations Framework for lock-free programming

#[cfg(target_os = "linux")]
pub mod linux_futex;

pub mod atomic_ext;
pub mod instance_tls;

pub use atomic_ext::*;
pub use instance_tls::*;
#[cfg(target_os = "linux")]
pub use linux_futex::*;

use crate::error::Result;

/// Cross-platform synchronization trait
pub trait PlatformSync {
    /// Wait on a futex (or equivalent)
    fn futex_wait(
        addr: &std::sync::atomic::AtomicU32,
        val: u32,
        timeout: Option<std::time::Duration>,
    ) -> Result<()>;

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
    fn futex_wait(
        _addr: &std::sync::atomic::AtomicU32,
        _val: u32,
        _timeout: Option<std::time::Duration>,
    ) -> Result<()> {
        // Fallback to parking_lot or standard library
        std::thread::yield_now();
        Ok(())
    }

    fn futex_wake(_addr: &std::sync::atomic::AtomicU32, _count: u32) -> Result<usize> {
        Ok(0)
    }
}

/// Returns a pointer-sized identifier for a [`tokio::runtime::Handle`].
///
/// Under `--cfg tokio_unstable`, delegates to [`tokio::runtime::Handle::id`].
/// Otherwise reads word 1 of the 2-word `scheduler::Handle` `ScalarPair` representation
/// (`Arc` scheduler pointer). Because the non-`tokio_unstable` fallback returns a heap
/// allocation address, a newly allocated runtime that reuses the exact scheduler address of
/// an already-dropped runtime could share the same value on that thread until `submit`
/// self-heals via `ensure_workers_running`.
#[cfg(all(feature = "async", tokio_unstable))]
#[inline]
pub(crate) fn tokio_runtime_handle_id(handle: &tokio::runtime::Handle) -> usize {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    handle.id().hash(&mut hasher);
    hasher.finish() as usize
}

/// Returns a pointer-sized identifier for the underlying scheduler allocation
/// of a [`tokio::runtime::Handle`].
///
/// Because this fallback returns a heap allocation address when `tokio_unstable`
/// is not enabled, a newly allocated runtime that reuses the exact scheduler
/// address of an already-dropped runtime could match a cached key on that thread
/// until `submit` self-heals via `ensure_workers_running`.
#[cfg(all(feature = "async", not(tokio_unstable)))]
#[inline]
pub(crate) fn tokio_runtime_handle_id(handle: &tokio::runtime::Handle) -> usize {
    const _: () = {
        assert!(
            std::mem::size_of::<tokio::runtime::Handle>() == 2 * std::mem::size_of::<usize>()
        );
        assert!(std::mem::align_of::<tokio::runtime::Handle>() == std::mem::align_of::<usize>());
    };
    let base = std::ptr::from_ref(handle).cast::<usize>();
    // SAFETY: `tokio::runtime::Handle` is a 2-word (`ScalarPair`) wrapper around
    // `scheduler::Handle::{CurrentThread(Arc<_>), MultiThread(Arc<_>)}` whose second
    // pointer-aligned word (offset `size_of::<usize>()`) holds the initialized non-null
    // `Arc` allocation address without padding bytes.
    unsafe { base.add(1).read() }
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::tokio_runtime_handle_id;

    #[test]
    fn test_s30_r1_tokio_runtime_handle_id_clone_identity_and_distinctness() {
        let ct1 = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let ct2 = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let mt1 = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .unwrap();
        let mt2 = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .unwrap();

        let h_ct1 = ct1.handle();
        let h_ct1_clone = h_ct1.clone();
        assert_eq!(
            tokio_runtime_handle_id(h_ct1),
            tokio_runtime_handle_id(&h_ct1_clone),
            "cloned current_thread handle must report identical runtime id"
        );

        let h_mt1 = mt1.handle();
        let h_mt1_clone = h_mt1.clone();
        assert_eq!(
            tokio_runtime_handle_id(h_mt1),
            tokio_runtime_handle_id(&h_mt1_clone),
            "cloned multi_thread handle must report identical runtime id"
        );

        assert_ne!(
            tokio_runtime_handle_id(ct1.handle()),
            tokio_runtime_handle_id(ct2.handle()),
            "distinct concurrent current_thread runtimes must report distinct ids"
        );
        assert_ne!(
            tokio_runtime_handle_id(mt1.handle()),
            tokio_runtime_handle_id(mt2.handle()),
            "distinct concurrent multi_thread runtimes must report distinct ids"
        );
        assert_ne!(
            tokio_runtime_handle_id(ct1.handle()),
            tokio_runtime_handle_id(mt1.handle()),
            "current_thread and multi_thread runtimes must report distinct ids"
        );
    }
}

