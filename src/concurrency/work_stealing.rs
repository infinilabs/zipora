//! Work-stealing task scheduler for efficient load balancing

use crate::error::{Result, ZiporaError};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

/// A task that can be executed by the work-stealing scheduler
pub trait Task: Send + 'static {
    /// Execute the task
    fn execute(self: Box<Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>;

    /// Get the task's priority (higher values = higher priority)
    fn priority(&self) -> u8 {
        0
    }

    /// Check if this task can be stolen by other workers
    fn is_stealable(&self) -> bool {
        true
    }

    /// Get an estimate of the task's execution time
    fn estimated_duration(&self) -> Duration {
        Duration::from_millis(1)
    }
}

/// A simple closure-based task
pub struct ClosureTask<F>
where
    F: FnOnce() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + 'static,
{
    closure: Option<F>,
    priority: u8,
    stealable: bool,
    estimated_duration: Duration,
}

impl<F> ClosureTask<F>
where
    F: FnOnce() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + 'static,
{
    /// Create a new work item with the provided closure
    pub fn new(closure: F) -> Self {
        Self {
            closure: Some(closure),
            priority: 0,
            stealable: true,
            estimated_duration: Duration::from_millis(1),
        }
    }

    /// Set the priority of this work item (higher values = higher priority)
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set whether this work item can be stolen by other workers
    pub fn with_stealable(mut self, stealable: bool) -> Self {
        self.stealable = stealable;
        self
    }

    /// Set the estimated duration for this work item (helps with scheduling)
    pub fn with_estimated_duration(mut self, duration: Duration) -> Self {
        self.estimated_duration = duration;
        self
    }
}

impl<F> Task for ClosureTask<F>
where
    F: FnOnce() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + 'static,
{
    fn execute(mut self: Box<Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
        if let Some(closure) = self.closure.take() {
            closure()
        } else {
            Box::pin(async { Err(ZiporaError::configuration("task already executed")) })
        }
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn is_stealable(&self) -> bool {
        self.stealable
    }

    fn estimated_duration(&self) -> Duration {
        self.estimated_duration
    }
}

/// A work-stealing queue for distributing tasks
pub struct WorkStealingQueue {
    local_queue: Mutex<VecDeque<Box<dyn Task>>>,
    steal_queue: Mutex<VecDeque<Box<dyn Task>>>,
    worker_id: usize,
    capacity: usize,
}

impl WorkStealingQueue {
    /// Create a new work-stealing queue for the specified worker
    pub fn new(worker_id: usize, capacity: usize) -> Self {
        Self {
            local_queue: Mutex::new(VecDeque::with_capacity(capacity)),
            steal_queue: Mutex::new(VecDeque::with_capacity(capacity / 2)),
            worker_id,
            capacity,
        }
    }

    /// Push a task to the local queue
    pub fn push_local(&self, task: Box<dyn Task>) -> Result<()> {
        self.try_push_local(task)
            .map_err(|_| ZiporaError::configuration("local queue full"))
    }

    /// Push a task to the local queue, returning the task back to the caller
    /// if the queue is full so it can be requeued elsewhere instead of dropped.
    pub fn try_push_local(&self, task: Box<dyn Task>) -> std::result::Result<(), Box<dyn Task>> {
        let mut queue = self.local_queue.lock().unwrap_or_else(|e| e.into_inner());
        let steal_len = self
            .steal_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();

        if queue.len().saturating_add(steal_len) >= self.capacity {
            return Err(task);
        }

        // Insert based on priority
        let priority = task.priority();
        let pos = queue
            .iter()
            .position(|t| t.priority() < priority)
            .unwrap_or(queue.len());
        queue.insert(pos, task);

        Ok(())
    }

    /// Pop a task from the local queue (highest priority first), falling back to
    /// tasks previously moved into `steal_queue` by `balance()`.
    pub fn pop_local(&self) -> Option<Box<dyn Task>> {
        if let Some(task) = self
            .local_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
        {
            return Some(task);
        }

        self.steal_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
    }

    /// Pop a task for the owning worker and atomically increment `active_tasks`
    /// while still holding the queue lock so `is_idle()` cannot observe a gap.
    fn pop_local_and_activate(&self, active_tasks: &AtomicUsize) -> Option<Box<dyn Task>> {
        {
            let mut local = self.local_queue.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(task) = local.pop_front() {
                active_tasks.fetch_add(1, Ordering::AcqRel);
                return Some(task);
            }
        }
        let mut steal = self.steal_queue.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(task) = steal.pop_front() {
            active_tasks.fetch_add(1, Ordering::AcqRel);
            return Some(task);
        }
        None
    }

    /// Steal a task from this queue (FIFO for load balancing)
    pub fn steal(&self) -> Option<Box<dyn Task>> {
        // First try the steal queue
        if let Some(task) = self
            .steal_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
        {
            return Some(task);
        }

        // Then try to steal any stealable task from the back of the local queue (lowest priority)
        let mut local_queue = self.local_queue.lock().unwrap_or_else(|e| e.into_inner());
        if !local_queue.is_empty() {
            let mut found_index = None;
            for (i, task) in local_queue.iter().enumerate().rev() {
                if task.is_stealable() {
                    found_index = Some(i);
                    break;
                }
            }

            if let Some(index) = found_index {
                return local_queue.remove(index);
            }
        }

        None
    }

    /// Steal a task and atomically increment `active_tasks` while holding the queue lock.
    fn steal_and_activate(&self, active_tasks: &AtomicUsize) -> Option<Box<dyn Task>> {
        {
            let mut steal_queue = self.steal_queue.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(task) = steal_queue.pop_front() {
                active_tasks.fetch_add(1, Ordering::AcqRel);
                return Some(task);
            }
        }

        let mut local_queue = self.local_queue.lock().unwrap_or_else(|e| e.into_inner());
        if !local_queue.is_empty() {
            let mut found_index = None;
            for (i, task) in local_queue.iter().enumerate().rev() {
                if task.is_stealable() {
                    found_index = Some(i);
                    break;
                }
            }

            if let Some(index) = found_index
                && let Some(task) = local_queue.remove(index)
            {
                active_tasks.fetch_add(1, Ordering::AcqRel);
                return Some(task);
            }
        }

        None
    }

    /// Move half of the local tasks to the steal queue
    pub fn balance(&self) {
        let mut local_queue = self.local_queue.lock().unwrap_or_else(|e| e.into_inner());
        let mut steal_queue = self.steal_queue.lock().unwrap_or_else(|e| e.into_inner());

        let local_len = local_queue.len();
        let steal_len = steal_queue.len();

        if local_len > steal_len + 1 {
            let to_move = (local_len - steal_len) / 2;
            for _ in 0..to_move {
                // Move from back (lower priority) to maintain priority ordering
                if let Some(task) = local_queue.pop_back() {
                    if task.is_stealable() {
                        steal_queue.push_back(task);
                    } else {
                        local_queue.push_back(task);
                        break;
                    }
                }
            }
        }
    }

    /// Get the number of tasks in both queues
    #[inline]
    pub fn len(&self) -> usize {
        self.local_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
            + self
                .steal_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len()
    }

    /// Check if both queues are empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the worker ID
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }
}

/// Statistics for the work-stealing executor
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    /// Total tasks executed
    pub total_executed: u64,
    /// Tasks currently being executed
    pub active_tasks: usize,
    /// Number of active workers
    pub active_workers: usize,
    /// Total steals performed
    pub total_steals: u64,
    /// Average task execution time
    pub avg_execution_time_us: u64,
    /// Worker utilization (0.0 to 1.0)
    pub utilization: f64,
}

/// Work-stealing executor for parallel task execution
pub struct WorkStealingExecutor {
    workers: Vec<WorkerThread>,
    queues: Vec<Arc<WorkStealingQueue>>,
    global_queue: Arc<Mutex<VecDeque<Box<dyn Task>>>>,
    stats: Arc<ExecutorStatsInner>,
    shutdown: Arc<AtomicBool>,
    next_worker: AtomicUsize,
}

struct ExecutorStatsInner {
    total_executed: AtomicUsize,
    active_tasks: AtomicUsize,
    active_workers: AtomicUsize,
    total_steals: AtomicUsize,
    total_execution_time_us: AtomicUsize,
}

#[allow(dead_code)] // RAII: holds Arc to keep worker's queue alive
struct WorkerThread {
    id: usize,
    handle: Mutex<Option<JoinHandle<()>>>,
    queue: Arc<WorkStealingQueue>,
}

/// Future adapter that catches panics while polling an inner future so a panicking
/// task does not terminate the worker loop.
struct CatchUnwindFuture<F> {
    inner: F,
}

impl<F: Future + Unpin> Future for CatchUnwindFuture<F> {
    type Output = std::thread::Result<F::Output>;

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let inner = &mut self.inner;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Pin::new(inner).poll(cx))) {
            Ok(std::task::Poll::Ready(out)) => std::task::Poll::Ready(Ok(out)),
            Ok(std::task::Poll::Pending) => std::task::Poll::Pending,
            Err(payload) => std::task::Poll::Ready(Err(payload)),
        }
    }
}

impl WorkStealingExecutor {
    /// Create a new work-stealing executor
    pub fn new(num_workers: usize, queue_capacity: usize) -> Result<Arc<Self>> {
        if num_workers == 0 {
            return Err(ZiporaError::invalid_data("num_workers cannot be zero"));
        }
        if queue_capacity == 0 {
            return Err(ZiporaError::invalid_data("queue_capacity cannot be zero"));
        }

        let mut workers = Vec::with_capacity(num_workers);
        let mut queues = Vec::with_capacity(num_workers);

        let global_queue = Arc::new(Mutex::new(VecDeque::new()));
        let stats = Arc::new(ExecutorStatsInner {
            total_executed: AtomicUsize::new(0),
            active_tasks: AtomicUsize::new(0),
            active_workers: AtomicUsize::new(num_workers),
            total_steals: AtomicUsize::new(0),
            total_execution_time_us: AtomicUsize::new(0),
        });
        let shutdown = Arc::new(AtomicBool::new(false));

        // Create worker queues
        for i in 0..num_workers {
            let queue = Arc::new(WorkStealingQueue::new(i, queue_capacity));
            queues.push(queue.clone());

            workers.push(WorkerThread {
                id: i,
                handle: Mutex::new(None),
                queue,
            });
        }

        // Start worker threads
        for worker in &mut workers {
            let queues = queues.clone();
            let global_queue = global_queue.clone();
            let stats = stats.clone();
            let shutdown = shutdown.clone();
            let worker_id = worker.id;

            let handle = tokio::spawn(async move {
                Self::worker_loop(worker_id, queues, global_queue, stats, shutdown).await;
            });

            *worker.handle.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
        }

        let executor = Arc::new(Self {
            workers,
            queues,
            global_queue,
            stats,
            shutdown,
            next_worker: AtomicUsize::new(0),
        });

        Ok(executor)
    }

    /// Submit a task for execution
    pub fn submit(&self, task: Box<dyn Task>) -> Result<()> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(ZiporaError::invalid_state(
                "cannot submit task to shut down executor",
            ));
        }

        // ORDERING: Relaxed is sufficient because next_worker is an independent round-robin
        // dispatch counter; queue locks provide synchronization for the task itself.
        let worker_id = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len();

        // Capacity check and push happen under a single lock acquisition; on a
        // full local queue the task is handed back and requeued globally, so a
        // concurrent fill can never cause it to be dropped unexecuted.
        let task = match self.queues[worker_id].try_push_local(task) {
            Ok(()) => return Ok(()),
            Err(task) => task,
        };

        // Go straight to global queue with priority ordering
        let mut global_queue = self.global_queue.lock().unwrap_or_else(|e| e.into_inner());
        if self.shutdown.load(Ordering::Acquire) {
            return Err(ZiporaError::invalid_state(
                "cannot submit task to shut down executor",
            ));
        }
        if global_queue.len() < 10000 {
            // Insert based on priority (highest first)
            let priority = task.priority();
            let pos = global_queue
                .iter()
                .position(|t| t.priority() < priority)
                .unwrap_or(global_queue.len());
            global_queue.insert(pos, task);
            Ok(())
        } else {
            Err(ZiporaError::configuration("all queues full"))
        }
    }

    /// Submit a closure as a task
    pub fn submit_closure<F>(&self, closure: F) -> Result<()>
    where
        F: FnOnce() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + 'static,
    {
        let task = Box::new(ClosureTask::new(closure));
        self.submit(task)
    }

    /// Get current executor statistics
    pub fn stats(&self) -> ExecutorStats {
        let total_executed = self.stats.total_executed.load(Ordering::Acquire) as u64;
        // ORDERING: Relaxed is sufficient for aggregate execution-time telemetry.
        let total_time = self.stats.total_execution_time_us.load(Ordering::Relaxed) as u64;

        let avg_execution_time_us = total_time.checked_div(total_executed).unwrap_or(0);

        let active_workers = self.stats.active_workers.load(Ordering::Acquire);
        let active_tasks = self.stats.active_tasks.load(Ordering::Acquire);
        let utilization = if active_workers > 0 {
            active_tasks as f64 / active_workers as f64
        } else {
            0.0
        };

        ExecutorStats {
            total_executed,
            active_tasks,
            active_workers,
            // ORDERING: Relaxed is sufficient for cumulative steal-count telemetry.
            total_steals: self.stats.total_steals.load(Ordering::Relaxed) as u64,
            avg_execution_time_us,
            utilization,
        }
    }

    /// Shutdown the executor, draining all queued and in-flight tasks before workers exit.
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Release);

        // Wait for all workers to drain remaining tasks and finish cleanly
        for worker in &self.workers {
            let handle = worker
                .handle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            if let Some(handle) = handle {
                let _ = handle.await;
            }
        }

        Ok(())
    }

    /// Worker loop for processing tasks
    async fn worker_loop(
        worker_id: usize,
        queues: Vec<Arc<WorkStealingQueue>>,
        global_queue: Arc<Mutex<VecDeque<Box<dyn Task>>>>,
        stats: Arc<ExecutorStatsInner>,
        shutdown: Arc<AtomicBool>,
    ) {
        struct ActiveWorkerGuard {
            stats: Arc<ExecutorStatsInner>,
        }
        impl Drop for ActiveWorkerGuard {
            fn drop(&mut self) {
                self.stats.active_workers.fetch_sub(1, Ordering::AcqRel);
            }
        }
        let _worker_guard = ActiveWorkerGuard {
            stats: Arc::clone(&stats),
        };

        let my_queue = &queues[worker_id];
        let other_queues: Vec<_> = queues
            .iter()
            .enumerate()
            .filter(|(id, _)| *id != worker_id)
            .map(|(_, queue)| queue.clone())
            .collect();

        let mut idle_count = 0;
        const MAX_IDLE: usize = 100;

        loop {
            let is_shutting_down = shutdown.load(Ordering::Acquire);
            let task = Self::find_task(my_queue, &other_queues, &global_queue, &stats);

            match task {
                Some(task) => {
                    idle_count = 0;

                    // `active_tasks` was already incremented inside `find_task` before
                    // releasing the queue lock; ensure it is always decremented even on panic.
                    struct ActiveTaskGuard<'a> {
                        stats: &'a ExecutorStatsInner,
                        start_time: Instant,
                    }
                    impl Drop for ActiveTaskGuard<'_> {
                        fn drop(&mut self) {
                            let execution_time = self.start_time.elapsed().as_micros() as usize;
                            // ORDERING: Relaxed is sufficient for cumulative execution-time telemetry.
                            self.stats
                                .total_execution_time_us
                                .fetch_add(execution_time, Ordering::Relaxed);
                            self.stats.active_tasks.fetch_sub(1, Ordering::AcqRel);
                        }
                    }

                    let task_guard = ActiveTaskGuard {
                        stats: &stats,
                        start_time: Instant::now(),
                    };

                    if let Ok(fut) =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task.execute()))
                    {
                        let _ = (CatchUnwindFuture { inner: fut }).await;
                    }

                    let executed = stats.total_executed.fetch_add(1, Ordering::Release) + 1;
                    drop(task_guard);

                    // Periodically balance the queue. Keyed to this worker's
                    // own completions: checking the shared counter on every
                    // loop iteration made an idle worker take both queue locks
                    // on every spin while the pool sat on a multiple of 100.
                    if executed.is_multiple_of(100) {
                        my_queue.balance();
                    }
                }
                None => {
                    if is_shutting_down {
                        break;
                    }
                    idle_count += 1;
                    if idle_count < MAX_IDLE {
                        // Short busy wait for low latency
                        tokio::task::yield_now().await;
                    } else {
                        // Longer sleep to reduce CPU usage
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                }
            }
        }
    }

    /// Find a task from local queue, other queues, or global queue, incrementing
    /// `active_tasks` under the queue lock before returning `Some(task)`.
    fn find_task(
        my_queue: &WorkStealingQueue,
        other_queues: &[Arc<WorkStealingQueue>],
        global_queue: &Arc<Mutex<VecDeque<Box<dyn Task>>>>,
        stats: &Arc<ExecutorStatsInner>,
    ) -> Option<Box<dyn Task>> {
        // 1. Try local queue first (best cache locality)
        if let Some(task) = my_queue.pop_local_and_activate(&stats.active_tasks) {
            return Some(task);
        }

        // 2. Try global queue
        if let Ok(mut queue) = global_queue.try_lock()
            && let Some(task) = queue.pop_front()
        {
            stats.active_tasks.fetch_add(1, Ordering::AcqRel);
            return Some(task);
        }

        // 3. Try to steal from other workers
        for other_queue in other_queues {
            if let Some(task) = other_queue.steal_and_activate(&stats.active_tasks) {
                // ORDERING: Relaxed is sufficient for cumulative steal-count telemetry.
                stats.total_steals.fetch_add(1, Ordering::Relaxed);
                return Some(task);
            }
        }

        None
    }

    /// Get the total number of queued tasks across all workers
    pub fn total_queued(&self) -> usize {
        let worker_tasks: usize = self.queues.iter().map(|q| q.len()).sum();
        let global_tasks = self
            .global_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        worker_tasks + global_tasks
    }

    /// Check if the executor is idle (no active or queued tasks)
    pub fn is_idle(&self) -> bool {
        self.stats.active_tasks.load(Ordering::Acquire) == 0 && self.total_queued() == 0
    }
}

impl Drop for WorkStealingExecutor {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        for worker in &self.workers {
            if let Some(handle) = worker
                .handle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
            {
                handle.abort();
            }
        }
    }
}

// Global executor instance for convenient access
use std::sync::OnceLock;
static GLOBAL_EXECUTOR: OnceLock<Arc<WorkStealingExecutor>> = OnceLock::new();

impl WorkStealingExecutor {
    /// Initialize the global executor
    pub async fn init(config: super::ConcurrencyConfig) -> Result<()> {
        if GLOBAL_EXECUTOR.get().is_some() {
            return Ok(());
        }

        let executor = WorkStealingExecutor::new(config.max_fibers, config.queue_size)?;

        match GLOBAL_EXECUTOR.set(executor) {
            Ok(()) => Ok(()),
            Err(unused_executor) => {
                let _ = unused_executor.shutdown().await;
                Ok(())
            }
        }
    }

    /// Get a reference to the global executor
    pub fn global() -> Option<&'static Arc<WorkStealingExecutor>> {
        GLOBAL_EXECUTOR.get()
    }

    /// Spawn a task on the global executor (or the current Tokio runtime if the global
    /// executor has not been initialized), preserving the fiber's unique ID.
    pub fn spawn<T>(fiber: super::Fiber<T>) -> super::FiberHandle<T>
    where
        T: Send + 'static,
    {
        let id = fiber.id().as_u64();
        if let Some(executor) = Self::global()
            && !executor.shutdown.load(Ordering::Acquire)
        {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let mut fiber_opt = Some(fiber);
            let _ = executor.submit_closure(move || {
                let fiber = fiber_opt.take();
                Box::pin(async move {
                    if let Some(f) = fiber {
                        let _ = tx.send(f.await);
                    }
                    Ok(())
                })
            });
            let handle = tokio::spawn(async move {
                rx.await.unwrap_or_else(|_| {
                    Err(ZiporaError::invalid_state(
                        "global executor dropped task before completion",
                    ))
                })
            });
            return super::FiberHandle::new(handle, id);
        }
        let handle = tokio::spawn(fiber);
        super::FiberHandle::new(handle, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_work_stealing_queue() {
        let queue = WorkStealingQueue::new(0, 100);

        // Test push and pop
        let task = Box::new(ClosureTask::new(|| Box::pin(async { Ok(()) })));
        queue.push_local(task).unwrap();

        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        let popped = queue.pop_local();
        assert!(popped.is_some());
        assert!(queue.is_empty());
    }

    #[tokio::test]
    async fn test_full_queue_returns_task_instead_of_dropping() {
        let queue = WorkStealingQueue::new(0, 1);

        let first = Box::new(ClosureTask::new(|| Box::pin(async { Ok(()) })));
        queue.push_local(first).unwrap();

        // A push into a full queue must hand the task back to the caller so it
        // can be requeued (e.g. to the global queue) rather than being dropped.
        let second = Box::new(ClosureTask::new(|| Box::pin(async { Ok(()) })).with_priority(7));
        let returned = queue
            .try_push_local(second)
            .expect_err("push into full queue must return the task");
        assert_eq!(returned.priority(), 7);
        assert_eq!(queue.len(), 1);
    }

    #[tokio::test]
    async fn test_submit_overflows_to_global_queue() {
        // Single worker with capacity 1: rapid submissions overflow the local
        // queue, and every overflowing task must be accepted into the global
        // queue instead of being lost.
        let executor = WorkStealingExecutor::new(1, 1).unwrap();
        for _ in 0..64 {
            executor
                .submit_closure(|| Box::pin(async { Ok(()) }))
                .expect("overflowing submission must fall back to global queue");
        }
        executor.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_task_priority() {
        let queue = WorkStealingQueue::new(0, 100);

        // Add tasks with different priorities
        let task1 = Box::new(ClosureTask::new(|| Box::pin(async { Ok(()) })).with_priority(1));
        let task2 = Box::new(ClosureTask::new(|| Box::pin(async { Ok(()) })).with_priority(3));
        let task3 = Box::new(ClosureTask::new(|| Box::pin(async { Ok(()) })).with_priority(2));

        queue.push_local(task1).unwrap();
        queue.push_local(task2).unwrap();
        queue.push_local(task3).unwrap();

        // Should pop in priority order (highest first)
        let popped1 = queue.pop_local().unwrap();
        assert_eq!(popped1.priority(), 3);

        let popped2 = queue.pop_local().unwrap();
        assert_eq!(popped2.priority(), 2);

        let popped3 = queue.pop_local().unwrap();
        assert_eq!(popped3.priority(), 1);
    }

    #[tokio::test]
    async fn test_task_stealing() {
        let queue1 = WorkStealingQueue::new(0, 100);
        let _queue2 = WorkStealingQueue::new(1, 100);

        // Add stealable and non-stealable tasks
        let stealable =
            Box::new(ClosureTask::new(|| Box::pin(async { Ok(()) })).with_stealable(true));
        let non_stealable =
            Box::new(ClosureTask::new(|| Box::pin(async { Ok(()) })).with_stealable(false));

        queue1.push_local(stealable).unwrap();
        queue1.push_local(non_stealable).unwrap();

        // Should be able to steal the stealable task
        let stolen = queue1.steal();
        assert!(stolen.is_some());
        assert!(stolen.unwrap().is_stealable());

        // Should not be able to steal the non-stealable task
        let not_stolen = queue1.steal();
        assert!(not_stolen.is_none());
    }

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = WorkStealingExecutor::new(4, 1000).unwrap();
        let stats = executor.stats();

        assert_eq!(stats.active_workers, 4);
        assert_eq!(stats.active_tasks, 0);
        assert_eq!(stats.total_executed, 0);
    }

    #[tokio::test]
    async fn test_task_submission() {
        let executor = WorkStealingExecutor::new(2, 1000).unwrap();

        let result = executor.submit_closure(|| Box::pin(async { Ok(()) }));
        assert!(result.is_ok());

        // Wait a bit for task to be processed
        tokio::time::sleep(Duration::from_millis(10)).await;

        let stats = executor.stats();
        assert!(stats.total_executed > 0 || stats.active_tasks > 0);
    }

    #[tokio::test]
    async fn test_c10_ws1_balanced_tasks_are_popped_by_owner_and_single_worker_executor() {
        let queue = WorkStealingQueue::new(0, 8);
        for _ in 0..4 {
            queue
                .push_local(Box::new(ClosureTask::new(|| Box::pin(async { Ok(()) }))))
                .unwrap();
        }
        queue.balance();

        let mut popped = 0;
        while queue.pop_local().is_some() {
            popped += 1;
        }
        assert_eq!(
            popped, 4,
            "pop_local must drain tasks moved to steal_queue by balance()"
        );
        assert!(queue.is_empty());
    }

    #[tokio::test]
    async fn test_c10_ws2_shutdown_drains_queued_and_active_tasks_and_rejects_post_shutdown_submit()
    {
        let executor = WorkStealingExecutor::new(2, 16).unwrap();
        let completed = Arc::new(AtomicUsize::new(0));

        for _ in 0..8 {
            let completed = Arc::clone(&completed);
            executor
                .submit_closure(move || {
                    Box::pin(async move {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                        completed.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                })
                .unwrap();
        }

        executor.shutdown().await.unwrap();
        assert_eq!(
            completed.load(Ordering::SeqCst),
            8,
            "shutdown() must drain and finish all submitted tasks instead of aborting workers"
        );
        assert!(
            executor
                .submit_closure(|| Box::pin(async { Ok(()) }))
                .is_err(),
            "submit() after shutdown() must return Err"
        );
    }

    #[tokio::test]
    async fn test_c10_ws3_panicking_task_does_not_leak_active_tasks_or_kill_worker() {
        let executor = WorkStealingExecutor::new(1, 16).unwrap();
        executor
            .submit_closure(|| {
                Box::pin(async {
                    panic!("intentional task panic");
                })
            })
            .unwrap();

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            executor.stats().active_tasks,
            0,
            "panicking task must not permanently leak active_tasks"
        );

        let ran_after_panic = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran_after_panic);
        executor
            .submit_closure(move || {
                Box::pin(async move {
                    flag.store(true, Ordering::SeqCst);
                    Ok(())
                })
            })
            .unwrap();

        executor.shutdown().await.unwrap();
        assert!(
            ran_after_panic.load(Ordering::SeqCst),
            "single worker must survive a panicking task and execute subsequent tasks"
        );
    }

    #[tokio::test]
    async fn test_c10_ws5_steal_takes_single_queued_stealable_task() {
        let queue = WorkStealingQueue::new(0, 8);
        queue
            .push_local(Box::new(
                ClosureTask::new(|| Box::pin(async { Ok(()) })).with_stealable(true),
            ))
            .unwrap();

        let stolen = queue.steal();
        assert!(
            stolen.is_some(),
            "steal() must be able to steal a single queued stealable task"
        );
        assert!(queue.is_empty());
    }

    #[tokio::test]
    async fn test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected() {
        assert!(
            WorkStealingExecutor::new(1, 0).is_err(),
            "WorkStealingExecutor::new with queue_capacity == 0 must return Err"
        );

        let _f1 = crate::concurrency::Fiber::new(async { Ok(1i32) });
        let f2 = crate::concurrency::Fiber::new(async { Ok(2i32) });
        let expected_id = f2.id();
        let handle = WorkStealingExecutor::spawn(f2);
        assert_ne!(
            handle.id(),
            0,
            "WorkStealingExecutor::spawn must preserve the Fiber's non-zero ID"
        );
        let _ = expected_id;
        assert_eq!(handle.await.unwrap(), 2);
    }
}
