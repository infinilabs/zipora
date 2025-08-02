//! High-performance fiber pool for concurrent execution

use crate::error::{Result, ToplingError};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

/// Configuration for the fiber pool
#[derive(Debug, Clone)]
pub struct FiberPoolConfig {
    /// Maximum number of concurrent fibers
    pub max_fibers: usize,
    /// Initial number of worker threads
    pub initial_workers: usize,
    /// Maximum number of worker threads
    pub max_workers: usize,
    /// Queue capacity for pending tasks
    pub queue_capacity: usize,
    /// Idle timeout for worker threads
    pub idle_timeout: Duration,
}

impl Default for FiberPoolConfig {
    fn default() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            max_fibers: cpu_count * 4,
            initial_workers: cpu_count,
            max_workers: cpu_count * 2,
            queue_capacity: 10000,
            idle_timeout: Duration::from_secs(60),
        }
    }
}

/// Statistics for fiber pool performance monitoring
#[derive(Debug, Clone)]
pub struct FiberStats {
    /// Total number of fibers spawned
    pub total_spawned: u64,
    /// Number of fibers currently running
    pub active_fibers: usize,
    /// Number of fibers completed successfully
    pub completed: u64,
    /// Number of fibers that failed
    pub failed: u64,
    /// Average execution time in microseconds
    pub avg_execution_time_us: u64,
    /// Number of active worker threads
    pub active_workers: usize,
    /// Queue utilization (0.0 to 1.0)
    pub queue_utilization: f64,
}

/// A handle to a spawned fiber
pub struct FiberHandle<T> {
    inner: JoinHandle<Result<T>>,
    id: u64,
    start_time: Instant,
}

impl<T> FiberHandle<T> {
    pub fn new(handle: JoinHandle<Result<T>>, id: u64) -> Self {
        Self {
            inner: handle,
            id,
            start_time: Instant::now(),
        }
    }

    /// Get the fiber's unique ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the elapsed time since the fiber was spawned
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Check if the fiber is finished
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Abort the fiber execution
    pub fn abort(&self) {
        self.inner.abort();
    }
}

impl<T> Future for FiberHandle<T> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.inner).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(e)) => Poll::Ready(Err(ToplingError::configuration(&format!(
                "fiber join error: {}",
                e
            )))),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// High-performance fiber pool for concurrent execution
pub struct FiberPool {
    config: FiberPoolConfig,
    semaphore: Arc<Semaphore>,
    stats: Arc<FiberPoolStats>,
    _runtime: tokio::runtime::Handle,
}

struct FiberPoolStats {
    total_spawned: AtomicU64,
    active_fibers: AtomicUsize,
    completed: AtomicU64,
    failed: AtomicU64,
    total_execution_time_us: AtomicU64,
    active_workers: AtomicUsize,
}

impl FiberPoolStats {
    fn new() -> Self {
        Self {
            total_spawned: AtomicU64::new(0),
            active_fibers: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            total_execution_time_us: AtomicU64::new(0),
            active_workers: AtomicUsize::new(0),
        }
    }
}

impl FiberPool {
    /// Create a new fiber pool with the given configuration
    pub fn new(config: FiberPoolConfig) -> Result<Self> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| ToplingError::configuration("no tokio runtime found"))?;

        let semaphore = Arc::new(Semaphore::new(config.max_fibers));
        let stats = Arc::new(FiberPoolStats::new());

        // Initialize with minimum number of workers
        stats
            .active_workers
            .store(config.initial_workers, Ordering::Relaxed);

        Ok(Self {
            config,
            semaphore,
            stats,
            _runtime: runtime,
        })
    }

    /// Create a fiber pool with default configuration
    pub fn default() -> Result<Self> {
        Self::new(FiberPoolConfig::default())
    }

    /// Spawn a new fiber for execution
    pub fn spawn<F, T>(&self, future: F) -> FiberHandle<T>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let id = self.stats.total_spawned.fetch_add(1, Ordering::Relaxed);
        let semaphore = self.semaphore.clone();
        let stats = self.stats.clone();

        let handle = tokio::task::spawn(async move {
            // Acquire semaphore permit
            let _permit = semaphore
                .acquire()
                .await
                .map_err(|_| ToplingError::configuration("semaphore acquire failed"))?;

            stats.active_fibers.fetch_add(1, Ordering::Relaxed);
            let start_time = Instant::now();

            let result = future.await;

            let execution_time = start_time.elapsed().as_micros() as u64;
            stats
                .total_execution_time_us
                .fetch_add(execution_time, Ordering::Relaxed);
            stats.active_fibers.fetch_sub(1, Ordering::Relaxed);

            match &result {
                Ok(_) => {
                    stats.completed.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    stats.failed.fetch_add(1, Ordering::Relaxed);
                }
            }

            result
        });

        FiberHandle::new(handle, id)
    }

    /// Spawn multiple fibers and return their handles
    pub fn spawn_batch<F, T, I>(&self, futures: I) -> Vec<FiberHandle<T>>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
        I: IntoIterator<Item = F>,
    {
        futures.into_iter().map(|f| self.spawn(f)).collect()
    }

    /// Execute a parallel map operation
    pub async fn parallel_map<I, F, T, R>(&self, iter: I, f: F) -> Result<Vec<R>>
    where
        I: IntoIterator<Item = T>,
        F: Fn(T) -> Result<R> + Send + Sync + Clone + 'static,
        T: Send + 'static,
        R: Send + 'static,
    {
        let handles: Vec<_> = iter
            .into_iter()
            .map(|item| {
                let f = f.clone();
                self.spawn(async move { f(item) })
            })
            .collect();

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            results.push(handle.await?);
        }

        Ok(results)
    }

    /// Execute a parallel for-each operation
    pub async fn parallel_for_each<I, F, T>(&self, iter: I, f: F) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        F: Fn(T) -> Result<()> + Send + Sync + Clone + 'static,
        T: Send + 'static,
    {
        let handles: Vec<_> = iter
            .into_iter()
            .map(|item| {
                let f = f.clone();
                self.spawn(async move { f(item) })
            })
            .collect();

        for handle in handles {
            handle.await?;
        }

        Ok(())
    }

    /// Execute a parallel reduce operation
    pub async fn parallel_reduce<I, F, T>(&self, iter: I, identity: T, f: F) -> Result<T>
    where
        I: IntoIterator<Item = T>,
        F: Fn(T, T) -> Result<T> + Send + Sync + Clone + 'static,
        T: Send + Clone + 'static,
    {
        let items: Vec<T> = iter.into_iter().collect();

        if items.is_empty() {
            return Ok(identity);
        }

        // Divide into chunks for parallel processing
        let chunk_size = std::cmp::max(1, items.len() / self.config.max_workers);
        let chunks: Vec<Vec<T>> = items.chunks(chunk_size).map(|c| c.to_vec()).collect();

        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let f = f.clone();
                let identity = identity.clone();
                self.spawn(async move {
                    let mut acc = identity;
                    for item in chunk {
                        acc = f(acc, item)?;
                    }
                    Ok(acc)
                })
            })
            .collect();

        let partial_results: Vec<T> = futures::future::try_join_all(handles).await?;

        // Reduce the partial results sequentially
        let mut final_result = identity;
        for partial in partial_results {
            final_result = f(final_result, partial)?;
        }

        Ok(final_result)
    }

    /// Get current pool statistics
    pub fn stats(&self) -> FiberStats {
        let total_spawned = self.stats.total_spawned.load(Ordering::Relaxed);
        let completed = self.stats.completed.load(Ordering::Relaxed);
        let total_time = self.stats.total_execution_time_us.load(Ordering::Relaxed);

        let avg_execution_time_us = if completed > 0 {
            total_time / completed
        } else {
            0
        };

        let active_fibers = self.stats.active_fibers.load(Ordering::Relaxed);
        let queue_utilization = active_fibers as f64 / self.config.max_fibers as f64;

        FiberStats {
            total_spawned,
            active_fibers,
            completed,
            failed: self.stats.failed.load(Ordering::Relaxed),
            avg_execution_time_us,
            active_workers: self.stats.active_workers.load(Ordering::Relaxed),
            queue_utilization,
        }
    }

    /// Wait for all active fibers to complete
    pub async fn shutdown(&self) -> Result<()> {
        // Wait for all permits to be available (no active fibers)
        let semaphore = self.semaphore.clone();
        let _permits = semaphore
            .acquire_many(self.config.max_fibers as u32)
            .await
            .map_err(|_| ToplingError::configuration("shutdown acquire failed"))?;

        Ok(())
    }

    /// Get the current load factor (0.0 to 1.0)
    pub fn load_factor(&self) -> f64 {
        let active = self.stats.active_fibers.load(Ordering::Relaxed);
        active as f64 / self.config.max_fibers as f64
    }

    /// Check if the pool is at capacity
    pub fn is_at_capacity(&self) -> bool {
        self.load_factor() >= 1.0
    }
}

/// Builder for configuring fiber pools
pub struct FiberPoolBuilder {
    config: FiberPoolConfig,
}

impl FiberPoolBuilder {
    pub fn new() -> Self {
        Self {
            config: FiberPoolConfig::default(),
        }
    }

    pub fn max_fibers(mut self, max_fibers: usize) -> Self {
        self.config.max_fibers = max_fibers;
        self
    }

    pub fn initial_workers(mut self, initial_workers: usize) -> Self {
        self.config.initial_workers = initial_workers;
        self
    }

    pub fn max_workers(mut self, max_workers: usize) -> Self {
        self.config.max_workers = max_workers;
        self
    }

    pub fn queue_capacity(mut self, queue_capacity: usize) -> Self {
        self.config.queue_capacity = queue_capacity;
        self
    }

    pub fn idle_timeout(mut self, idle_timeout: Duration) -> Self {
        self.config.idle_timeout = idle_timeout;
        self
    }

    pub fn build(self) -> Result<FiberPool> {
        FiberPool::new(self.config)
    }
}

impl Default for FiberPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_fiber_pool_creation() {
        let pool = FiberPool::default().unwrap();
        let stats = pool.stats();

        assert_eq!(stats.active_fibers, 0);
        assert_eq!(stats.total_spawned, 0);
        assert_eq!(stats.completed, 0);
    }

    #[tokio::test]
    async fn test_fiber_spawning() {
        let pool = FiberPool::default().unwrap();

        let handle = pool.spawn(async { Ok(42i32) });
        let result = handle.await.unwrap();

        assert_eq!(result, 42);

        let stats = pool.stats();
        assert_eq!(stats.total_spawned, 1);
        assert_eq!(stats.completed, 1);
    }

    #[tokio::test]
    async fn test_parallel_map() {
        let pool = FiberPool::default().unwrap();
        let input = vec![1, 2, 3, 4, 5];

        let result = pool.parallel_map(input, |x| Ok(x * 2)).await.unwrap();
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[tokio::test]
    async fn test_parallel_reduce() {
        let pool = FiberPool::default().unwrap();
        let input = vec![1, 2, 3, 4, 5];

        let result = pool
            .parallel_reduce(input, 0, |acc, x| Ok(acc + x))
            .await
            .unwrap();
        assert_eq!(result, 15);
    }

    #[tokio::test]
    async fn test_fiber_pool_builder() {
        let pool = FiberPoolBuilder::new()
            .max_fibers(100)
            .initial_workers(4)
            .max_workers(8)
            .build()
            .unwrap();

        assert_eq!(pool.config.max_fibers, 100);
        assert_eq!(pool.config.initial_workers, 4);
        assert_eq!(pool.config.max_workers, 8);
    }

    #[tokio::test]
    async fn test_load_factor() {
        let pool = FiberPool::default().unwrap();

        assert_eq!(pool.load_factor(), 0.0);
        assert!(!pool.is_at_capacity());

        // Spawn some fibers to increase load
        let handles: Vec<_> = (0..5)
            .map(|i| {
                pool.spawn(async move {
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    Ok(i)
                })
            })
            .collect();

        // Load factor should be > 0 while fibers are running
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert!(pool.load_factor() > 0.0);

        // Wait for completion
        for handle in handles {
            handle.await.unwrap();
        }
    }
}
