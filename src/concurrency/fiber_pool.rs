//! High-performance fiber pool for concurrent execution

use crate::error::{Result, ZiporaError};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
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
    /// Create a new fiber handle with the given join handle and ID
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
            Poll::Ready(Err(e)) => Poll::Ready(Err(ZiporaError::configuration(format!(
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
    queue_semaphore: Arc<Semaphore>,
    stats: Arc<FiberPoolStats>,
    shutdown: std::sync::atomic::AtomicBool,
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
    fn new(initial_workers: usize) -> Self {
        Self {
            total_spawned: AtomicU64::new(0),
            active_fibers: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            total_execution_time_us: AtomicU64::new(0),
            active_workers: AtomicUsize::new(initial_workers),
        }
    }
}

impl FiberPool {
    /// Create a new fiber pool with the given configuration
    pub fn new(config: FiberPoolConfig) -> Result<Self> {
        if config.max_fibers == 0 {
            return Err(ZiporaError::invalid_parameter(
                "FiberPool max_fibers cannot be zero",
            ));
        }
        if config.initial_workers == 0 {
            return Err(ZiporaError::invalid_parameter(
                "FiberPool initial_workers cannot be zero",
            ));
        }
        if config.initial_workers > config.max_workers {
            return Err(ZiporaError::invalid_parameter(
                "FiberPool initial_workers cannot exceed max_workers",
            ));
        }
        let total_permits = config
            .max_fibers
            .checked_add(config.queue_capacity)
            .ok_or_else(|| {
                ZiporaError::invalid_parameter(
                    "FiberPool max_fibers + queue_capacity overflows usize",
                )
            })?;
        let max_permits = Semaphore::MAX_PERMITS.min(u32::MAX as usize);
        if total_permits > max_permits {
            return Err(ZiporaError::invalid_parameter(
                "FiberPool max_fibers + queue_capacity exceeds maximum semaphore permits",
            ));
        }

        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| ZiporaError::configuration("no tokio runtime found"))?;

        let semaphore = Arc::new(Semaphore::new(config.max_fibers));
        let queue_semaphore = Arc::new(Semaphore::new(total_permits));
        let stats = Arc::new(FiberPoolStats::new(config.initial_workers));

        Ok(Self {
            config,
            semaphore,
            queue_semaphore,
            stats,
            shutdown: std::sync::atomic::AtomicBool::new(false),
            _runtime: runtime,
        })
    }

    /// Create a fiber pool with default configuration
    #[allow(clippy::should_implement_trait)] // inherent default() returns Result; cannot implement Default trait
    pub fn default() -> Result<Self> {
        Self::new(FiberPoolConfig::default())
    }

    /// Spawn a new fiber for execution
    pub fn spawn<F, T>(&self, future: F) -> Result<FiberHandle<T>>
    where
        F: Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(ZiporaError::invalid_state("Fiber pool is shut down"));
        }

        // Enforce queue capacity limit before spawning so `shutdown()` can wait on
        // `queue_semaphore` for all spawned (running and queued) fibers to finish.
        let queue_permit = self
            .queue_semaphore
            .clone()
            .try_acquire_owned()
            .map_err(|_| ZiporaError::invalid_state("Fiber pool queue capacity exceeded"))?;

        // ORDERING: Relaxed is sufficient because total_spawned is a monotonic ID/telemetry
        // counter; task synchronization is provided by `queue_semaphore` and `JoinHandle`.
        let id = self.stats.total_spawned.fetch_add(1, Ordering::Relaxed);
        let semaphore = self.semaphore.clone();
        let stats = self.stats.clone();

        let handle = tokio::task::spawn(async move {
            // Keep queue permit across the entire lifecycle of the task.
            let _queue_permit = queue_permit;

            // Acquire running semaphore permit
            let _run_permit = semaphore
                .acquire()
                .await
                .map_err(|_| ZiporaError::configuration("semaphore acquire failed"))?;

            // Drop guard: a panic unwinding out of `future.await` skips any
            // straight-line bookkeeping after it, which would permanently
            // over-count active_fibers and count the fiber in neither the
            // completed nor the failed bucket.
            struct ActiveFiberGuard {
                stats: Arc<FiberPoolStats>,
                finished: bool,
            }
            impl Drop for ActiveFiberGuard {
                fn drop(&mut self) {
                    self.stats.active_fibers.fetch_sub(1, Ordering::AcqRel);
                    if !self.finished {
                        // ORDERING: Relaxed is sufficient for cumulative failure telemetry.
                        self.stats.failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            stats.active_fibers.fetch_add(1, Ordering::AcqRel);
            let mut guard = ActiveFiberGuard {
                stats: stats.clone(),
                finished: false,
            };
            let start_time = Instant::now();

            let result = future.await;

            let execution_time = start_time.elapsed().as_micros() as u64;
            // ORDERING: Relaxed is sufficient for aggregate execution-time telemetry.
            stats
                .total_execution_time_us
                .fetch_add(execution_time, Ordering::Relaxed);
            guard.finished = true;

            match &result {
                Ok(_) => {
                    // ORDERING: Relaxed is sufficient for cumulative completion telemetry.
                    stats.completed.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    // ORDERING: Relaxed is sufficient for cumulative failure telemetry.
                    stats.failed.fetch_add(1, Ordering::Relaxed);
                }
            }

            result
        });

        Ok(FiberHandle::new(handle, id))
    }

    /// Get current pool statistics
    pub fn stats(&self) -> FiberStats {
        // ORDERING: Relaxed is sufficient for point-in-time telemetry snapshots.
        let total_spawned = self.stats.total_spawned.load(Ordering::Relaxed);
        // ORDERING: Relaxed is sufficient for point-in-time telemetry snapshots.
        let completed = self.stats.completed.load(Ordering::Relaxed);
        // ORDERING: Relaxed is sufficient for point-in-time telemetry snapshots.
        let total_time = self.stats.total_execution_time_us.load(Ordering::Relaxed);

        let avg_execution_time_us = total_time.checked_div(completed).unwrap_or(0);

        let active_fibers = self.stats.active_fibers.load(Ordering::Acquire);

        let total_queue_slots = self.config.max_fibers + self.config.queue_capacity;
        let available_permits = self.queue_semaphore.available_permits();
        let acquired = total_queue_slots.saturating_sub(available_permits);
        let queued = acquired
            .saturating_sub(active_fibers)
            .min(self.config.queue_capacity);

        let queue_utilization = if self.config.queue_capacity > 0 {
            queued as f64 / self.config.queue_capacity as f64
        } else {
            0.0
        };

        FiberStats {
            total_spawned,
            active_fibers,
            completed,
            // ORDERING: Relaxed is sufficient for point-in-time telemetry snapshots.
            failed: self.stats.failed.load(Ordering::Relaxed),
            avg_execution_time_us,
            // ORDERING: Relaxed is sufficient for point-in-time telemetry snapshots.
            active_workers: self.stats.active_workers.load(Ordering::Relaxed),
            queue_utilization,
        }
    }

    /// Wait for all spawned (active and queued) fibers to complete
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Release);
        let total_permits = (self.config.max_fibers + self.config.queue_capacity) as u32;
        let _permits = self
            .queue_semaphore
            .acquire_many(total_permits)
            .await
            .map_err(|_| ZiporaError::configuration("shutdown acquire failed"))?;

        Ok(())
    }

    /// Get the current load factor (0.0 to 1.0)
    pub fn load_factor(&self) -> f64 {
        let active = self.stats.active_fibers.load(Ordering::Acquire);
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
    /// Create a new builder for configuring a fiber pool
    pub fn new() -> Self {
        Self {
            config: FiberPoolConfig::default(),
        }
    }

    /// Set the maximum number of concurrent fibers
    pub fn max_fibers(mut self, max_fibers: usize) -> Self {
        self.config.max_fibers = max_fibers;
        self
    }

    /// Set the number of worker threads to start initially
    pub fn initial_workers(mut self, initial_workers: usize) -> Self {
        self.config.initial_workers = initial_workers;
        self
    }

    /// Set the maximum number of worker threads
    pub fn max_workers(mut self, max_workers: usize) -> Self {
        self.config.max_workers = max_workers;
        self
    }

    /// Set the capacity of the work queue
    pub fn queue_capacity(mut self, queue_capacity: usize) -> Self {
        self.config.queue_capacity = queue_capacity;
        self
    }

    /// Set the timeout after which idle workers are terminated
    pub fn idle_timeout(mut self, idle_timeout: Duration) -> Self {
        self.config.idle_timeout = idle_timeout;
        self
    }

    /// Build the configured fiber pool
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_fiber_pool_creation() {
        let pool = FiberPool::default().unwrap();
        let stats = pool.stats();

        assert_eq!(stats.active_fibers, 0);
        assert_eq!(stats.total_spawned, 0);
        assert_eq!(stats.completed, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_fiber_spawning() {
        let pool = FiberPool::default().unwrap();

        let handle = pool.spawn(async { Ok(42i32) }).unwrap();
        let result = handle.await.unwrap();

        assert_eq!(result, 42);

        let stats = pool.stats();
        assert_eq!(stats.total_spawned, 1);
        assert_eq!(stats.completed, 1);
    }

    /// A fiber that panics must still decrement active_fibers (Drop guard)
    /// and be accounted as failed; otherwise the counter leaks permanently
    /// and derived stats (queued, queue_utilization) skew forever after.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_fiber_panic_does_not_leak_active_counter() {
        async fn panicking() -> Result<i32> {
            panic!("intentional fiber panic");
        }

        let pool = FiberPool::default().unwrap();

        let handle = pool.spawn(panicking()).unwrap();
        let result = handle.await;
        assert!(result.is_err(), "panicked fiber must surface a join error");

        let stats = pool.stats();
        assert_eq!(stats.active_fibers, 0, "active_fibers leaked after panic");
        assert_eq!(stats.failed, 1, "panicked fiber must count as failed");
        assert_eq!(stats.completed, 0);

        // The pool must remain fully usable afterwards
        let handle = pool.spawn(async { Ok(7i32) }).unwrap();
        assert_eq!(handle.await.unwrap(), 7);
        assert_eq!(pool.stats().active_fibers, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
                .unwrap()
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_fiber_pool_queue_capacity_limit() {
        let config = FiberPoolConfig {
            max_fibers: 1,
            queue_capacity: 2,
            initial_workers: 1,
            max_workers: 1,
            idle_timeout: Duration::from_secs(60),
        };
        let pool = FiberPool::new(config).unwrap();

        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let started_cl = Arc::clone(&started);
        let release_cl = Arc::clone(&release);

        // Spawn 1st task to block the only running slot
        let _h1 = pool
            .spawn(async move {
                started_cl.notify_one();
                release_cl.notified().await;
                Ok(())
            })
            .unwrap();

        // Wait until the 1st task has acquired the run permit and started
        started.notified().await;

        // Spawn 2nd and 3rd tasks to fill the queue capacity (queue_capacity = 2)
        let _h2 = pool.spawn(async { Ok(()) }).unwrap();

        let stats_half = pool.stats();
        assert_eq!(stats_half.active_fibers, 1);
        // queue_utilization should be 0.5 (1 task queued out of 2)
        assert_eq!(stats_half.queue_utilization, 0.5);

        let _h3 = pool.spawn(async { Ok(()) }).unwrap();

        let stats_full = pool.stats();
        // queue_utilization should be 1.0 (2 tasks queued out of 2)
        assert_eq!(stats_full.queue_utilization, 1.0);

        // Hitting the queue capacity limit on the 4th spawn
        let result = pool.spawn(async { Ok(()) });
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("queue capacity exceeded"));
        }

        // Release the running task and wait for all tasks to complete
        release.notify_one();
        let _ = _h1.await;
        let _ = _h2.await;
        let _ = _h3.await;

        // Now we should be able to spawn again
        let result_again = pool.spawn(async { Ok(100) });
        assert!(result_again.is_ok());
        assert_eq!(result_again.unwrap().await.unwrap(), 100);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_c10_fp1_shutdown_waits_for_spawned_and_queued_fibers() {
        let config = FiberPoolConfig {
            max_fibers: 1,
            queue_capacity: 4,
            initial_workers: 1,
            max_workers: 1,
            idle_timeout: Duration::from_secs(60),
        };
        let pool = FiberPool::new(config).unwrap();
        let done = Arc::new(AtomicUsize::new(0));

        for _ in 0..3 {
            let done = Arc::clone(&done);
            drop(
                pool.spawn(async move {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    done.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
                .unwrap(),
            );
        }

        // Immediately shut down without yielding first: shutdown() must wait for both
        // active and queued fibers to complete before returning.
        pool.shutdown().await.unwrap();
        assert_eq!(
            done.load(Ordering::SeqCst),
            3,
            "FiberPool::shutdown() returned before all spawned/queued fibers completed"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_c10_fp2_invalid_config_returns_err_instead_of_panic_or_nan() {
        let zero_fibers = FiberPoolConfig {
            max_fibers: 0,
            ..FiberPoolConfig::default()
        };
        assert!(
            FiberPool::new(zero_fibers).is_err(),
            "FiberPool::new with max_fibers == 0 must return Err"
        );

        let zero_workers = FiberPoolConfig {
            initial_workers: 0,
            ..FiberPoolConfig::default()
        };
        assert!(
            FiberPool::new(zero_workers).is_err(),
            "FiberPool::new with initial_workers == 0 must return Err"
        );

        let overflow_queue = FiberPoolConfig {
            max_fibers: 16,
            queue_capacity: usize::MAX,
            ..FiberPoolConfig::default()
        };
        assert!(
            FiberPool::new(overflow_queue).is_err(),
            "FiberPool::new with overflowing max_fibers + queue_capacity must return Err"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_s25_r7_multithreaded_fiber_pool_payload_writes_and_shutdown() {
        let pool = Arc::new(
            FiberPool::new(FiberPoolConfig {
                max_fibers: 8,
                initial_workers: 4,
                max_workers: 4,
                queue_capacity: 256,
                idle_timeout: Duration::from_secs(10),
            })
            .unwrap(),
        );
        let completed = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for i in 0..64usize {
            let completed = Arc::clone(&completed);
            let mut payload = Box::new([0u64; 8]);
            for (k, slot) in payload.iter_mut().enumerate() {
                *slot = (i * 17 + k) as u64;
            }
            let expected: u64 = payload.iter().map(|v| v.wrapping_mul(3)).sum();
            handles.push(
                pool.spawn(async move {
                    for slot in payload.iter_mut() {
                        *slot = slot.wrapping_mul(3);
                    }
                    let actual: u64 = payload.iter().sum();
                    assert_eq!(actual, expected);
                    completed.fetch_add(1, Ordering::AcqRel);
                    Ok(actual)
                })
                .unwrap(),
            );
        }
        pool.shutdown().await.unwrap();
        assert_eq!(completed.load(Ordering::Acquire), 64);
        for h in handles {
            assert!(h.await.is_ok());
        }
    }
}
