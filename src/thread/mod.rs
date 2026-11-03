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

/// Returns the unique [`tokio::runtime::Id`] for a [`tokio::runtime::Handle`].
#[cfg(feature = "async")]
#[inline]
pub(crate) fn tokio_runtime_handle_id(handle: &tokio::runtime::Handle) -> tokio::runtime::Id {
    handle.id()
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

#[cfg(all(test, feature = "async"))]
mod plan_v2_s6 {
    use super::tokio_runtime_handle_id;

    /// S6: the runtime key must identify the runtime, not the address of its
    /// scheduler allocation. A runtime built right after another one was
    /// dropped on the same thread usually reuses that allocation, so at
    /// v5.0.0 (outside `--cfg tokio_unstable`) the key repeats; the stable
    /// `Handle::id()` (tokio >= 1.49) never does.
    #[test]
    fn test_v2_s6_runtime_key_not_reused_after_runtime_drop() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..16 {
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            let key = tokio_runtime_handle_id(rt.handle());
            assert!(
                seen.insert(key),
                "runtime #{i} got the key {key:?} of an already-dropped runtime"
            );
            drop(rt);
        }
    }
}
