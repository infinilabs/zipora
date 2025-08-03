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
        let mut queue = self.local_queue.lock().unwrap();

        if queue.len() >= self.capacity {
            return Err(ZiporaError::configuration("local queue full"));
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

    /// Pop a task from the local queue (highest priority first)
    pub fn pop_local(&self) -> Option<Box<dyn Task>> {
        // Pop from front since tasks are sorted by priority (highest first)
        self.local_queue.lock().unwrap().pop_front()
    }

    /// Steal a task from this queue (FIFO for load balancing)
    pub fn steal(&self) -> Option<Box<dyn Task>> {
        // First try the steal queue
        if let Some(task) = self.steal_queue.lock().unwrap().pop_front() {
            return Some(task);
        }

        // Then try to steal from the local queue
        let mut local_queue = self.local_queue.lock().unwrap();
        if local_queue.len() > 1 {
            // Only steal if there's more than one task
            // Try to find a stealable task from the back (lowest priority)
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

    /// Move half of the local tasks to the steal queue
    pub fn balance(&self) {
        let mut local_queue = self.local_queue.lock().unwrap();
        let mut steal_queue = self.steal_queue.lock().unwrap();

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
    pub fn len(&self) -> usize {
        self.local_queue.lock().unwrap().len() + self.steal_queue.lock().unwrap().len()
    }

    /// Check if both queues are empty
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

#[allow(dead_code)]
struct WorkerThread {
    id: usize,
    handle: Option<JoinHandle<()>>,
    queue: Arc<WorkStealingQueue>,
}

impl WorkStealingExecutor {
    /// Create a new work-stealing executor
    pub fn new(num_workers: usize, queue_capacity: usize) -> Result<Arc<Self>> {
        if num_workers == 0 {
            return Err(ZiporaError::invalid_data("num_workers cannot be zero"));
        }

        let mut workers = Vec::with_capacity(num_workers);
        let mut queues = Vec::with_capacity(num_workers);

        let global_queue = Arc::new(Mutex::new(VecDeque::new()));
        let stats = Arc::new(ExecutorStatsInner {
            total_executed: AtomicUsize::new(0),
            active_tasks: AtomicUsize::new(0),
            active_workers: AtomicUsize::new(0),
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
                handle: None,
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

            worker.handle = Some(handle);
        }

        let executor = Arc::new(Self {
            workers,
            queues: queues.clone(),
            global_queue: global_queue.clone(),
            stats: stats.clone(),
            shutdown: shutdown.clone(),
            next_worker: AtomicUsize::new(0),
        });

        stats.active_workers.store(num_workers, Ordering::Relaxed);

        Ok(executor)
    }

    /// Submit a task for execution
    pub fn submit(&self, task: Box<dyn Task>) -> Result<()> {
        // Try to submit to a worker queue first
        let worker_id = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len();

        // Check if local queue has space and submit directly to global if not
        let can_use_local = {
            let queue = self.queues[worker_id].local_queue.lock().unwrap();
            queue.len() < self.queues[worker_id].capacity
        };

        if can_use_local {
            // Try local queue
            if self.queues[worker_id].push_local(task).is_ok() {
                return Ok(());
            }
            // If this fails, the task is consumed but we have an error
            return Err(ZiporaError::configuration("local queue push failed"));
        } else {
            // Go straight to global queue with priority ordering
            let mut global_queue = self.global_queue.lock().unwrap();
            if global_queue.len() < 10000 {
                // Arbitrary limit
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
        let total_executed = self.stats.total_executed.load(Ordering::Relaxed) as u64;
        let total_time = self.stats.total_execution_time_us.load(Ordering::Relaxed) as u64;

        let avg_execution_time_us = if total_executed > 0 {
            total_time / total_executed
        } else {
            0
        };

        let active_workers = self.stats.active_workers.load(Ordering::Relaxed);
        let utilization = if active_workers > 0 {
            self.stats.active_tasks.load(Ordering::Relaxed) as f64 / active_workers as f64
        } else {
            0.0
        };

        ExecutorStats {
            total_executed,
            active_tasks: self.stats.active_tasks.load(Ordering::Relaxed),
            active_workers,
            total_steals: self.stats.total_steals.load(Ordering::Relaxed) as u64,
            avg_execution_time_us,
            utilization,
        }
    }

    /// Shutdown the executor and wait for all workers to complete
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for all workers to finish
        for worker in &self.workers {
            if let Some(ref handle) = worker.handle {
                let _ = handle.abort();
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
        let my_queue = &queues[worker_id];
        let other_queues: Vec<_> = queues
            .iter()
            .enumerate()
            .filter(|(id, _)| *id != worker_id)
            .map(|(_, queue)| queue.clone())
            .collect();

        let mut idle_count = 0;
        const MAX_IDLE: usize = 100;

        while !shutdown.load(Ordering::Relaxed) {
            let task = Self::find_task(my_queue, &other_queues, &global_queue, &stats);

            match task {
                Some(task) => {
                    idle_count = 0;

                    // Execute the task
                    let start_time = Instant::now();
                    stats.active_tasks.fetch_add(1, Ordering::Relaxed);

                    let _ = task.execute().await;

                    let execution_time = start_time.elapsed().as_micros() as usize;
                    stats
                        .total_execution_time_us
                        .fetch_add(execution_time, Ordering::Relaxed);
                    stats.total_executed.fetch_add(1, Ordering::Relaxed);
                    stats.active_tasks.fetch_sub(1, Ordering::Relaxed);
                }
                None => {
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

            // Periodically balance the queue
            if stats.total_executed.load(Ordering::Relaxed) % 100 == 0 {
                my_queue.balance();
            }
        }
    }

    /// Find a task from local queue, other queues, or global queue
    fn find_task(
        my_queue: &WorkStealingQueue,
        other_queues: &[Arc<WorkStealingQueue>],
        global_queue: &Arc<Mutex<VecDeque<Box<dyn Task>>>>,
        stats: &Arc<ExecutorStatsInner>,
    ) -> Option<Box<dyn Task>> {
        // 1. Try local queue first (best cache locality)
        if let Some(task) = my_queue.pop_local() {
            return Some(task);
        }

        // 2. Try global queue
        if let Ok(mut queue) = global_queue.try_lock() {
            if let Some(task) = queue.pop_front() {
                return Some(task);
            }
        }

        // 3. Try to steal from other workers
        for other_queue in other_queues {
            if let Some(task) = other_queue.steal() {
                stats.total_steals.fetch_add(1, Ordering::Relaxed);
                return Some(task);
            }
        }

        None
    }

    /// Get the total number of queued tasks across all workers
    pub fn total_queued(&self) -> usize {
        let worker_tasks: usize = self.queues.iter().map(|q| q.len()).sum();
        let global_tasks = self.global_queue.lock().unwrap().len();
        worker_tasks + global_tasks
    }

    /// Check if the executor is idle (no active or queued tasks)
    pub fn is_idle(&self) -> bool {
        self.stats.active_tasks.load(Ordering::Relaxed) == 0 && self.total_queued() == 0
    }
}

// Global executor instance for convenient access
use std::sync::OnceLock;
static GLOBAL_EXECUTOR: OnceLock<Arc<WorkStealingExecutor>> = OnceLock::new();

impl WorkStealingExecutor {
    /// Initialize the global executor
    pub async fn init(config: super::ConcurrencyConfig) -> Result<()> {
        // Try to initialize the global executor
        let executor = WorkStealingExecutor::new(config.max_fibers, config.queue_size)?;
        
        // This will only succeed the first time it's called
        match GLOBAL_EXECUTOR.set(executor) {
            Ok(_) => Ok(()),
            Err(_) => {
                // Already initialized, which is fine
                Ok(())
            }
        }
    }

    /// Get a reference to the global executor
    pub fn global() -> Option<&'static Arc<WorkStealingExecutor>> {
        GLOBAL_EXECUTOR.get()
    }

    /// Spawn a task on the global executor
    pub fn spawn<T>(fiber: super::Fiber<T>) -> super::FiberHandle<T>
    where
        T: Send + 'static,
    {
        // For now, fall back to tokio spawn
        // In a real implementation, this would use the work-stealing executor
        let handle = tokio::spawn(fiber);
        super::FiberHandle::new(handle, 0)
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
}
