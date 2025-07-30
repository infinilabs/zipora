//! Fiber-based concurrency and pipeline processing
//!
//! This module provides high-performance async/await based concurrency primitives
//! optimized for data processing pipelines and parallel algorithms.

pub mod fiber_pool;
pub mod pipeline;
pub mod parallel_trie;
pub mod async_blob_store;
pub mod work_stealing;

pub use fiber_pool::{FiberPool, FiberPoolConfig, FiberHandle, FiberStats};
pub use pipeline::{Pipeline, PipelineStage, PipelineBuilder, PipelineStats};
pub use parallel_trie::{ParallelTrieBuilder, ParallelLoudsTrie};
pub use async_blob_store::{AsyncBlobStore, AsyncMemoryBlobStore, AsyncFileStore};
pub use work_stealing::{WorkStealingQueue, WorkStealingExecutor, Task};

use crate::error::{ToplingError, Result};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A lightweight fiber for concurrent execution
pub struct Fiber<T> {
    future: Pin<Box<dyn Future<Output = Result<T>> + Send + 'static>>,
    id: FiberId,
}

/// Unique identifier for a fiber
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FiberId(u64);

impl FiberId {
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

/// Configuration for concurrency parameters
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    /// Maximum number of concurrent fibers
    pub max_fibers: usize,
    /// Size of work-stealing queues
    pub queue_size: usize,
    /// Enable NUMA-aware scheduling
    pub numa_aware: bool,
    /// Fiber stack size
    pub stack_size: usize,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_fibers: num_cpus::get() * 2,
            queue_size: 1024,
            numa_aware: false,
            stack_size: 64 * 1024, // 64KB
        }
    }
}

/// Initialize the concurrency runtime
pub async fn init_concurrency(config: ConcurrencyConfig) -> Result<()> {
    // Verify configuration
    if config.max_fibers == 0 {
        return Err(ToplingError::invalid_data("max_fibers cannot be zero"));
    }
    
    if config.queue_size == 0 {
        return Err(ToplingError::invalid_data("queue_size cannot be zero"));
    }
    
    // Initialize the global executor
    WorkStealingExecutor::init(config).await?;
    
    Ok(())
}

/// Spawn a new fiber for execution
pub fn spawn<F, T>(future: F) -> FiberHandle<T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    let fiber = Fiber::new(future);
    let id = fiber.id();
    
    // Submit to the global executor
    WorkStealingExecutor::spawn(fiber)
}

/// Join multiple fibers and collect their results
pub async fn join_all<T>(handles: Vec<FiberHandle<T>>) -> Result<Vec<T>>
where
    T: Send + 'static,
{
    let mut results = Vec::with_capacity(handles.len());
    
    for handle in handles {
        results.push(handle.await?);
    }
    
    Ok(results)
}

/// Execute a function on a separate thread pool
pub async fn spawn_blocking<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ToplingError::configuration(&format!("spawn_blocking failed: {}", e)))?
}

/// Parallel map operation over an iterator
pub async fn parallel_map<I, F, T, R>(
    iter: I,
    f: F,
) -> Result<Vec<R>>
where
    I: IntoIterator<Item = T> + Send,
    I::IntoIter: Send,
    F: Fn(T) -> Result<R> + Send + Sync + Clone + 'static,
    T: Send + 'static,
    R: Send + 'static,
{
    let items: Vec<T> = iter.into_iter().collect();
    let mut handles = Vec::with_capacity(items.len());
    
    for item in items {
        let f = f.clone();
        let handle = spawn(async move { f(item) });
        handles.push(handle);
    }
    
    join_all(handles).await
}

/// Parallel reduce operation 
pub async fn parallel_reduce<I, F, T>(
    iter: I,
    identity: T,
    f: F,
) -> Result<T>
where
    I: IntoIterator<Item = T> + Send,
    I::IntoIter: Send,
    F: Fn(T, T) -> Result<T> + Send + Sync + Clone + 'static,
    T: Send + Clone + 'static,
{
    let items: Vec<T> = iter.into_iter().collect();
    
    if items.is_empty() {
        return Ok(identity);
    }
    
    // Divide into chunks for parallel processing
    let chunk_size = (items.len() + num_cpus::get() - 1) / num_cpus::get();
    let chunks: Vec<Vec<T>> = items.chunks(chunk_size).map(|c| c.to_vec()).collect();
    
    let mut handles = Vec::with_capacity(chunks.len());
    
    for chunk in chunks {
        let f = f.clone();
        let identity = identity.clone();
        
        let handle = spawn(async move {
            let mut acc = identity;
            for item in chunk {
                acc = f(acc, item)?;
            }
            Ok(acc)
        });
        
        handles.push(handle);
    }
    
    let partial_results = join_all(handles).await?;
    
    // Reduce the partial results
    let mut final_result = identity;
    for partial in partial_results {
        final_result = f(final_result, partial)?;
    }
    
    Ok(final_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    
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
    
    #[tokio::test]
    async fn test_concurrency_config() {
        let config = ConcurrencyConfig::default();
        assert!(config.max_fibers > 0);
        assert!(config.queue_size > 0);
        assert!(config.stack_size > 0);
    }
    
    #[tokio::test]
    async fn test_parallel_map() {
        let input = vec![1, 2, 3, 4, 5];
        let f = |x: i32| -> Result<i32> { Ok(x * 2) };
        
        let result = parallel_map(input, f).await.unwrap();
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }
    
    #[tokio::test]
    async fn test_parallel_reduce() {
        let input = vec![1, 2, 3, 4, 5];
        let f = |acc: i32, x: i32| -> Result<i32> { Ok(acc + x) };
        
        let result = parallel_reduce(input, 0, f).await.unwrap();
        assert_eq!(result, 15);
    }
    
    #[tokio::test]
    async fn test_spawn_blocking() {
        let result = spawn_blocking(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            Ok(42)
        }).await.unwrap();
        
        assert_eq!(result, 42);
    }
}