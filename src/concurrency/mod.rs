//! Concurrency primitives and pipeline processing
//!
//! Provides fiber pool, pipeline stages, work-stealing executor,
//! and parallel trie building.

#[cfg(feature = "async")]
pub mod fiber_pool;
#[cfg(feature = "async")]
pub mod parallel_trie;
#[cfg(feature = "async")]
pub mod pipeline;
#[cfg(feature = "async")]
pub mod work_stealing;

#[cfg(feature = "async")]
pub use fiber_pool::{FiberHandle, FiberPool, FiberPoolConfig, FiberPoolBuilder, FiberStats};
#[cfg(feature = "async")]
pub use parallel_trie::{ParallelLoudsTrie, ParallelTrieBuilder};
#[cfg(feature = "async")]
pub use pipeline::{Pipeline, PipelineBuilder, PipelineStage, PipelineStats};
#[cfg(feature = "async")]
pub use work_stealing::{Task, WorkStealingExecutor, WorkStealingQueue};

use crate::error::Result;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A lightweight fiber for concurrent execution.
/// Used internally by WorkStealingExecutor.
pub struct Fiber<T> {
    future: Pin<Box<dyn Future<Output = Result<T>> + Send + 'static>>,
    id: FiberId,
}

/// Unique identifier for a fiber
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FiberId(u64);

impl FiberId {
    /// Generate a new unique fiber identifier
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl<T> Fiber<T> {
    /// Create a new fiber from a future
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = Result<T>> + Send + 'static,
    {
        Self {
            future: Box::pin(future),
            id: FiberId::new(),
        }
    }

    /// Get the fiber's unique ID
    #[inline]
    pub fn id(&self) -> FiberId {
        self.id
    }
}

impl<T> Future for Fiber<T> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.future.as_mut().poll(cx)
    }
}

/// Configuration for concurrency parameters.
/// Used by WorkStealingExecutor::init().
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    /// Maximum number of concurrent fibers
    pub max_fibers: usize,
    /// Size of work-stealing queues
    pub queue_size: usize,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_fibers: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) * 2,
            queue_size: 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_fiber_creation() {
        let fiber = Fiber::new(async { Ok(42i32) });
        let id = fiber.id();

        let result = fiber.await.unwrap();
        assert_eq!(result, 42);

        // IDs should be unique
        let fiber2 = Fiber::new(async { Ok(24i32) });
        assert_ne!(id, fiber2.id());
    }

    #[test]
    fn test_concurrency_config() {
        let config = ConcurrencyConfig::default();
        assert!(config.max_fibers > 0);
        assert!(config.queue_size > 0);
    }
}
