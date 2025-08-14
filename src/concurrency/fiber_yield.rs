//! Cooperative multitasking utilities for fine-grained control
//!
//! This module provides sophisticated yielding mechanisms for fiber-based concurrency,
//! enabling fine-grained control over task scheduling and preventing fiber starvation.

use crate::error::Result;
use std::cell::Cell;
use std::future::Future;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for fiber yielding behavior
#[derive(Debug, Clone)]
pub struct YieldConfig {
    /// Initial yield budget per fiber
    pub initial_budget: u8,
    /// Maximum yield budget
    pub max_budget: u8,
    /// Minimum yield budget
    pub min_budget: u8,
    /// Budget decay rate (per second)
    pub decay_rate: f64,
    /// Yield threshold for automatic yielding
    pub yield_threshold: Duration,
    /// Enable adaptive yield budgeting
    pub adaptive_budgeting: bool,
}

impl Default for YieldConfig {
    fn default() -> Self {
        Self {
            initial_budget: 16,
            max_budget: 32,
            min_budget: 1,
            decay_rate: 0.1,
            yield_threshold: Duration::from_micros(100),
            adaptive_budgeting: true,
        }
    }
}

/// High-performance yielding mechanism with budget control
pub struct FiberYield {
    config: YieldConfig,
    yield_budget: Cell<u8>,
    last_yield: Cell<Instant>,
    execution_time: Cell<Duration>,
    total_yields: Cell<u64>,
    runtime_handle: Option<tokio::runtime::Handle>,
}

impl FiberYield {
    /// Create a new fiber yield controller
    pub fn new() -> Self {
        Self::with_config(YieldConfig::default())
    }

    /// Create a new fiber yield controller with custom configuration
    pub fn with_config(config: YieldConfig) -> Self {
        let runtime_handle = tokio::runtime::Handle::try_current().ok();

        Self {
            yield_budget: Cell::new(config.initial_budget),
            last_yield: Cell::new(Instant::now()),
            execution_time: Cell::new(Duration::ZERO),
            total_yields: Cell::new(0),
            config,
            runtime_handle,
        }
    }

    /// Perform a lightweight yield operation
    pub async fn yield_now(&self) {
        // Update execution time
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_yield.get());
        self.execution_time.set(self.execution_time.get() + elapsed);

        // Check if we should yield based on budget
        let current_budget = self.yield_budget.get();
        if current_budget > 0 {
            self.yield_budget.set(current_budget - 1);
            self.last_yield.set(now);
            self.total_yields.set(self.total_yields.get() + 1);

            // Perform the actual yield
            tokio::task::yield_now().await;
        } else {
            // Force yield and reset budget
            self.force_yield().await;
        }
    }

    /// Force an immediate yield regardless of budget
    pub async fn force_yield(&self) {
        self.yield_budget.set(self.config.initial_budget);
        self.last_yield.set(Instant::now());
        self.total_yields.set(self.total_yields.get() + 1);

        tokio::task::yield_now().await;
    }

    /// Conditional yield based on execution time
    pub async fn yield_if_needed(&self) {
        let elapsed = self.last_yield.get().elapsed();
        if elapsed >= self.config.yield_threshold {
            self.yield_now().await;
        }
    }

    /// Yield with a custom sleep duration
    pub async fn yield_for(&self, duration: Duration) {
        self.last_yield.set(Instant::now());
        self.total_yields.set(self.total_yields.get() + 1);

        tokio::time::sleep(duration).await;
    }

    /// Check if a yield is recommended
    pub fn should_yield(&self) -> bool {
        let elapsed = self.last_yield.get().elapsed();
        elapsed >= self.config.yield_threshold || self.yield_budget.get() == 0
    }

    /// Get current yield budget
    pub fn budget(&self) -> u8 {
        self.yield_budget.get()
    }

    /// Get total number of yields performed
    pub fn total_yields(&self) -> u64 {
        self.total_yields.get()
    }

    /// Get total execution time
    pub fn execution_time(&self) -> Duration {
        self.execution_time.get()
    }

    /// Reset the yield controller
    pub fn reset(&self) {
        self.yield_budget.set(self.config.initial_budget);
        self.last_yield.set(Instant::now());
        self.execution_time.set(Duration::ZERO);
        self.total_yields.set(0);
    }

    /// Update yield budget based on adaptive algorithm
    pub fn update_budget(&self, load_factor: f64) {
        if !self.config.adaptive_budgeting {
            return;
        }

        let current_budget = self.yield_budget.get();
        let new_budget = if load_factor > 0.8 {
            // High load: reduce budget to yield more frequently
            (current_budget.saturating_sub(1)).max(self.config.min_budget)
        } else if load_factor < 0.2 {
            // Low load: increase budget to yield less frequently
            (current_budget.saturating_add(1)).min(self.config.max_budget)
        } else {
            current_budget
        };

        self.yield_budget.set(new_budget);
    }
}

/// Thread-local yield controller for optimal performance
thread_local! {
    static THREAD_LOCAL_YIELD: FiberYield = FiberYield::new();
}

/// Global fiber yield operations using thread-local optimizations
pub struct GlobalYield;

impl GlobalYield {
    /// Perform a yield using the thread-local controller
    pub async fn yield_now() {
        // Update the thread-local yield controller
        THREAD_LOCAL_YIELD.with(|y| {
            let current_budget = y.yield_budget.get();
            if current_budget > 0 {
                y.yield_budget.set(current_budget - 1);
                y.total_yields.set(y.total_yields.get() + 1);
            } else {
                y.yield_budget.set(y.config.initial_budget);
                y.total_yields.set(y.total_yields.get() + 1);
            }
            y.last_yield.set(Instant::now());
        });
        
        // Perform the actual yield
        tokio::task::yield_now().await;
    }

    /// Force yield using the thread-local controller
    pub async fn force_yield() {
        THREAD_LOCAL_YIELD.with(|y| {
            y.yield_budget.set(y.config.initial_budget);
            y.total_yields.set(y.total_yields.get() + 1);
            y.last_yield.set(Instant::now());
        });
        
        tokio::task::yield_now().await;
    }

    /// Conditional yield using the thread-local controller
    pub async fn yield_if_needed() {
        let should_yield = THREAD_LOCAL_YIELD.with(|y| {
            let elapsed = y.last_yield.get().elapsed();
            elapsed >= y.config.yield_threshold || y.yield_budget.get() == 0
        });
        
        if should_yield {
            Self::yield_now().await;
        }
    }

    /// Check if yield is recommended
    pub fn should_yield() -> bool {
        THREAD_LOCAL_YIELD.with(|y| y.should_yield())
    }

    /// Get current thread's yield statistics
    pub fn stats() -> YieldStats {
        THREAD_LOCAL_YIELD.with(|y| YieldStats {
            budget: y.budget(),
            total_yields: y.total_yields(),
            execution_time: y.execution_time(),
            last_yield: y.last_yield.get(),
        })
    }

    /// Reset the thread-local yield controller
    pub fn reset() {
        THREAD_LOCAL_YIELD.with(|y| y.reset());
    }
}

/// Statistics for yield operations
#[derive(Debug, Clone)]
pub struct YieldStats {
    pub budget: u8,
    pub total_yields: u64,
    pub execution_time: Duration,
    pub last_yield: Instant,
}

/// Cooperative yield point that can be inserted into long-running operations
pub struct YieldPoint {
    yield_controller: Arc<FiberYield>,
    operation_count: AtomicUsize,
    yield_interval: usize,
}

impl YieldPoint {
    /// Create a new yield point
    pub fn new(yield_interval: usize) -> Self {
        Self {
            yield_controller: Arc::new(FiberYield::new()),
            operation_count: AtomicUsize::new(0),
            yield_interval,
        }
    }

    /// Mark an operation and potentially yield
    pub async fn checkpoint(&self) {
        let count = self.operation_count.fetch_add(1, Ordering::Relaxed);
        if count % self.yield_interval == 0 {
            self.yield_controller.yield_now().await;
        }
    }

    /// Force a yield at this point
    pub async fn yield_now(&self) {
        self.yield_controller.force_yield().await;
    }

    /// Get operation count
    pub fn operation_count(&self) -> usize {
        self.operation_count.load(Ordering::Relaxed)
    }

    /// Reset the yield point
    pub fn reset(&self) {
        self.operation_count.store(0, Ordering::Relaxed);
        self.yield_controller.reset();
    }
}

/// Yielding wrapper for iterators
pub struct YieldingIterator<I> {
    inner: I,
    yield_point: YieldPoint,
    processed: usize,
}

impl<I> YieldingIterator<I>
where
    I: Iterator,
{
    /// Create a new yielding iterator
    pub fn new(iterator: I, yield_interval: usize) -> Self {
        Self {
            inner: iterator,
            yield_point: YieldPoint::new(yield_interval),
            processed: 0,
        }
    }

    /// Process the iterator with automatic yielding
    pub async fn for_each<F>(mut self, mut f: F) -> Result<usize>
    where
        F: FnMut(I::Item) -> Result<()>,
    {
        while let Some(item) = self.inner.next() {
            f(item)?;
            self.processed += 1;
            self.yield_point.checkpoint().await;
        }

        Ok(self.processed)
    }

    /// Collect items with automatic yielding
    pub async fn collect<C>(mut self) -> C
    where
        C: Default + Extend<I::Item>,
    {
        let mut collection = C::default();

        while let Some(item) = self.inner.next() {
            collection.extend(std::iter::once(item));
            self.processed += 1;

            // Yield periodically
            if self.processed % self.yield_point.yield_interval == 0 {
                self.yield_point.yield_now().await;
            }
        }

        collection
    }

    /// Get number of processed items
    pub fn processed_count(&self) -> usize {
        self.processed
    }
}

/// Adaptive yield scheduler for managing multiple fibers
pub struct AdaptiveYieldScheduler {
    global_stats: Arc<GlobalYieldStats>,
    config: YieldConfig,
}

struct GlobalYieldStats {
    total_fibers: AtomicUsize,
    active_fibers: AtomicUsize,
    total_yields: AtomicU64,
    avg_execution_time: AtomicU64, // In microseconds
}

impl AdaptiveYieldScheduler {
    /// Create a new adaptive yield scheduler
    pub fn new() -> Self {
        Self::with_config(YieldConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: YieldConfig) -> Self {
        Self {
            global_stats: Arc::new(GlobalYieldStats {
                total_fibers: AtomicUsize::new(0),
                active_fibers: AtomicUsize::new(0),
                total_yields: AtomicU64::new(0),
                avg_execution_time: AtomicU64::new(0),
            }),
            config,
        }
    }

    /// Register a new fiber with the scheduler
    pub fn register_fiber(&self) -> FiberYieldHandle {
        self.global_stats.total_fibers.fetch_add(1, Ordering::Relaxed);
        self.global_stats.active_fibers.fetch_add(1, Ordering::Relaxed);

        FiberYieldHandle {
            scheduler: self.global_stats.clone(),
            yield_controller: FiberYield::with_config(self.config.clone()),
            registered: true,
        }
    }

    /// Get current load factor
    pub fn load_factor(&self) -> f64 {
        let active = self.global_stats.active_fibers.load(Ordering::Relaxed);
        let total = self.global_stats.total_fibers.load(Ordering::Relaxed);

        if total == 0 {
            0.0
        } else {
            active as f64 / total as f64
        }
    }

    /// Get global statistics
    pub fn stats(&self) -> GlobalYieldStats {
        GlobalYieldStats {
            total_fibers: AtomicUsize::new(self.global_stats.total_fibers.load(Ordering::Relaxed)),
            active_fibers: AtomicUsize::new(self.global_stats.active_fibers.load(Ordering::Relaxed)),
            total_yields: AtomicU64::new(self.global_stats.total_yields.load(Ordering::Relaxed)),
            avg_execution_time: AtomicU64::new(
                self.global_stats.avg_execution_time.load(Ordering::Relaxed),
            ),
        }
    }
}

/// Handle for individual fiber yield management
pub struct FiberYieldHandle {
    scheduler: Arc<GlobalYieldStats>,
    yield_controller: FiberYield,
    registered: bool,
}

impl FiberYieldHandle {
    /// Perform a scheduled yield
    pub async fn yield_now(&self) {
        // Update global statistics
        self.scheduler.total_yields.fetch_add(1, Ordering::Relaxed);

        // Update adaptive budget based on global load
        let load_factor = {
            let active = self.scheduler.active_fibers.load(Ordering::Relaxed);
            let total = self.scheduler.total_fibers.load(Ordering::Relaxed);
            if total == 0 { 0.0 } else { active as f64 / total as f64 }
        };

        self.yield_controller.update_budget(load_factor);
        self.yield_controller.yield_now().await;
    }

    /// Get yield statistics for this fiber
    pub fn stats(&self) -> YieldStats {
        YieldStats {
            budget: self.yield_controller.budget(),
            total_yields: self.yield_controller.total_yields(),
            execution_time: self.yield_controller.execution_time(),
            last_yield: self.yield_controller.last_yield.get(),
        }
    }
}

impl Drop for FiberYieldHandle {
    fn drop(&mut self) {
        if self.registered {
            self.scheduler.active_fibers.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Utility functions for cooperative multitasking
pub struct CooperativeUtils;

impl CooperativeUtils {
    /// Run a closure with automatic yielding every N iterations
    pub async fn run_with_yield<F, R>(
        iterations: usize,
        yield_interval: usize,
        mut f: F,
    ) -> Result<Vec<R>>
    where
        F: FnMut(usize) -> Result<R>,
    {
        let mut results = Vec::with_capacity(iterations);
        let yield_point = YieldPoint::new(yield_interval);

        for i in 0..iterations {
            results.push(f(i)?);
            yield_point.checkpoint().await;
        }

        Ok(results)
    }

    /// Process a vector with automatic yielding
    pub async fn process_vec_yielding<T, F, R>(
        items: Vec<T>,
        yield_interval: usize,
        mut processor: F,
    ) -> Result<Vec<R>>
    where
        F: FnMut(T) -> Result<R>,
    {
        let mut results = Vec::with_capacity(items.len());
        let yield_point = YieldPoint::new(yield_interval);

        for (i, item) in items.into_iter().enumerate() {
            results.push(processor(item)?);

            if i % yield_interval == 0 {
                yield_point.yield_now().await;
            }
        }

        Ok(results)
    }

    /// Run multiple operations concurrently with yield control
    pub async fn concurrent_with_yield<F, R>(
        operations: Vec<F>,
        max_concurrent: usize,
    ) -> Result<Vec<R>>
    where
        F: Future<Output = Result<R>> + Send + 'static,
        R: Send + 'static,
    {
        use futures::stream::{self, StreamExt};

        let results = stream::iter(operations)
            .map(|op| async move {
                // Add yield points during execution
                let yield_controller = FiberYield::new();
                
                tokio::select! {
                    result = op => result,
                    _ = async {
                        loop {
                            yield_controller.yield_if_needed().await;
                            tokio::time::sleep(Duration::from_micros(100)).await;
                        }
                    } => unreachable!(),
                }
            })
            .buffer_unordered(max_concurrent)
            .collect::<Vec<_>>()
            .await;

        // Collect results, propagating errors
        let mut output = Vec::with_capacity(results.len());
        for result in results {
            output.push(result?);
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_fiber_yield_creation() {
        let yield_controller = FiberYield::new();
        assert!(yield_controller.budget() > 0);
        assert_eq!(yield_controller.total_yields(), 0);
    }

    #[tokio::test]
    async fn test_yield_now() {
        let yield_controller = FiberYield::new();
        let initial_budget = yield_controller.budget();

        yield_controller.yield_now().await;

        assert_eq!(yield_controller.budget(), initial_budget - 1);
        assert_eq!(yield_controller.total_yields(), 1);
    }

    #[tokio::test]
    async fn test_force_yield() {
        let yield_controller = FiberYield::new();
        
        // Exhaust budget
        while yield_controller.budget() > 0 {
            yield_controller.yield_now().await;
        }

        let yields_before = yield_controller.total_yields();
        yield_controller.force_yield().await;

        assert!(yield_controller.budget() > 0);
        assert_eq!(yield_controller.total_yields(), yields_before + 1);
    }

    #[tokio::test]
    async fn test_global_yield() {
        GlobalYield::reset();
        assert!(GlobalYield::stats().budget > 0);

        GlobalYield::yield_now().await;
        assert_eq!(GlobalYield::stats().total_yields, 1);
    }

    #[tokio::test]
    async fn test_yield_point() {
        let yield_point = YieldPoint::new(5);

        // Checkpoint several times
        for i in 0..12 {
            yield_point.checkpoint().await;
            if i == 4 || i == 9 {
                // Should have yielded at these points
                assert!(yield_point.yield_controller.total_yields() > 0);
            }
        }

        assert_eq!(yield_point.operation_count(), 12);
    }

    #[tokio::test]
    async fn test_yielding_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let yielding_iter = YieldingIterator::new(data.into_iter(), 3);

        let mut sum = 0;
        let processed = yielding_iter
            .for_each(|x| {
                sum += x;
                Ok(())
            })
            .await
            .unwrap();

        assert_eq!(processed, 10);
        assert_eq!(sum, 55);
    }

    #[tokio::test]
    async fn test_adaptive_scheduler() {
        let scheduler = AdaptiveYieldScheduler::new();
        let handle = scheduler.register_fiber();

        assert_eq!(scheduler.load_factor(), 1.0);

        handle.yield_now().await;
        assert!(handle.stats().total_yields > 0);
    }

    #[tokio::test]
    async fn test_cooperative_utils() {
        let result = CooperativeUtils::run_with_yield(10, 3, |i| Ok(i * 2))
            .await
            .unwrap();

        assert_eq!(result.len(), 10);
        assert_eq!(result[0], 0);
        assert_eq!(result[9], 18);
    }

    #[tokio::test]
    async fn test_yield_for_duration() {
        let yield_controller = FiberYield::new();
        let start = Instant::now();

        yield_controller.yield_for(Duration::from_millis(10)).await;

        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
        assert_eq!(yield_controller.total_yields(), 1);
    }

    #[tokio::test]
    async fn test_should_yield() {
        let config = YieldConfig {
            yield_threshold: Duration::from_millis(1),
            ..Default::default()
        };

        let yield_controller = FiberYield::with_config(config);
        
        // Initially should not need to yield
        assert!(!yield_controller.should_yield());

        // After some time, should recommend yielding
        tokio::time::sleep(Duration::from_millis(2)).await;
        assert!(yield_controller.should_yield());
    }
}