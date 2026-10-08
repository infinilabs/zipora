//! Work-stealing task scheduler for efficient load balancing

use crate::error::{Result, ZiporaError};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
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
    steal_len: AtomicUsize,
    worker_id: usize,
    capacity: usize,
}

impl WorkStealingQueue {
    /// Create a new work-stealing queue for the specified worker
    pub fn new(worker_id: usize, capacity: usize) -> Self {
        Self {
            local_queue: Mutex::new(VecDeque::with_capacity(capacity)),
            steal_queue: Mutex::new(VecDeque::with_capacity(capacity / 2)),
            steal_len: AtomicUsize::new(0),
            worker_id,
            capacity,
        }
    }

    #[inline]
    fn insert_by_priority(queue: &mut VecDeque<Box<dyn Task>>, task: Box<dyn Task>) {
        let priority = task.priority();
        if queue.back().is_none_or(|t| t.priority() >= priority) {
            queue.push_back(task);
        } else {
            let pos = queue
                .iter()
                .position(|t| t.priority() < priority)
                .unwrap_or(queue.len());
            queue.insert(pos, task);
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
        // ORDERING: Relaxed is sufficient for capacity estimation alongside the `local_queue` lock.
        let steal_len = self.steal_len.load(Ordering::Relaxed);

        if queue.len().saturating_add(steal_len) >= self.capacity {
            return Err(task);
        }

        Self::insert_by_priority(&mut queue, task);
        Ok(())
    }

    /// Push a task to the local queue unless `shutdown` has been set.
    /// Checking `shutdown` while holding `local_queue` serializes against `shutdown()`'s
    /// lock-barrier sweep so an accepted task can never be stranded after workers exit.
    fn try_push_local_unless_shutdown(
        &self,
        shutdown: &AtomicBool,
        task: Box<dyn Task>,
    ) -> std::result::Result<(), (Box<dyn Task>, bool)> {
        let mut queue = self.local_queue.lock().unwrap_or_else(|e| e.into_inner());
        if shutdown.load(Ordering::Acquire) {
            return Err((task, true));
        }
        // ORDERING: Relaxed is sufficient for capacity estimation alongside the `local_queue` lock.
        let steal_len = self.steal_len.load(Ordering::Relaxed);

        if queue.len().saturating_add(steal_len) >= self.capacity {
            return Err((task, false));
        }

        Self::insert_by_priority(&mut queue, task);
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

        if self.steal_len.load(Ordering::Acquire) == 0 {
            return None;
        }

        let mut steal = self.steal_queue.lock().unwrap_or_else(|e| e.into_inner());
        let task = steal.pop_front();
        if task.is_some() {
            self.steal_len.fetch_sub(1, Ordering::Release);
        }
        task
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
        if self.steal_len.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut steal = self.steal_queue.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(task) = steal.pop_front() {
            active_tasks.fetch_add(1, Ordering::AcqRel);
            self.steal_len.fetch_sub(1, Ordering::Release);
            return Some(task);
        }
        None
    }

    /// Steal a task from this queue (FIFO for load balancing)
    pub fn steal(&self) -> Option<Box<dyn Task>> {
        // First try the steal queue if non-empty
        if self.steal_len.load(Ordering::Acquire) > 0 {
            let mut steal_queue = self.steal_queue.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(task) = steal_queue.pop_front() {
                self.steal_len.fetch_sub(1, Ordering::Release);
                return Some(task);
            }
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
        if self.steal_len.load(Ordering::Acquire) > 0 {
            let mut steal_queue = self.steal_queue.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(task) = steal_queue.pop_front() {
                active_tasks.fetch_add(1, Ordering::AcqRel);
                self.steal_len.fetch_sub(1, Ordering::Release);
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
            let mut moved = 0usize;
            for _ in 0..to_move {
                // Move from back (lower priority) to maintain priority ordering
                if let Some(task) = local_queue.pop_back() {
                    if task.is_stealable() {
                        steal_queue.push_back(task);
                        moved += 1;
                    } else {
                        local_queue.push_back(task);
                        break;
                    }
                }
            }
            if moved > 0 {
                self.steal_len.fetch_add(moved, Ordering::Release);
            }
        }
    }

    /// Get the number of tasks in both queues
    #[inline]
    pub fn len(&self) -> usize {
        let local_len = self
            .local_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        let steal_len = if self.steal_len.load(Ordering::Acquire) == 0 {
            0
        } else {
            self.steal_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len()
        };
        local_len + steal_len
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

const WORKER_DEAD: u8 = 0;
const WORKER_SPAWNING: u8 = 1;
const WORKER_LIVE: u8 = 2;

static NEXT_EXECUTOR_ID: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    static LAST_RECORDED_FALLBACK: std::cell::Cell<Option<(usize, usize)>> =
        const { std::cell::Cell::new(None) };
}

/// Work-stealing executor for parallel task execution
pub struct WorkStealingExecutor {
    executor_id: usize,
    workers: Vec<WorkerThread>,
    queues: Vec<Arc<WorkStealingQueue>>,
    global_queue: Arc<Mutex<VecDeque<Box<dyn Task>>>>,
    stats: Arc<ExecutorStatsInner>,
    shutdown: Arc<AtomicBool>,
    work_notify: Arc<tokio::sync::Notify>,
    respawn_lock: Mutex<()>,
    fallback_runtime: Mutex<Option<tokio::runtime::Handle>>,
    self_weak: Weak<Self>,
    next_worker: AtomicUsize,
}

struct ExecutorStatsInner {
    total_executed: AtomicUsize,
    active_tasks: AtomicUsize,
    active_workers: AtomicUsize,
    parked_workers: AtomicUsize,
    total_steals: AtomicUsize,
    total_execution_time_us: AtomicUsize,
}

#[allow(dead_code)] // RAII: holds Arc to keep worker's queue alive
struct WorkerThread {
    id: usize,
    state: Arc<AtomicU8>,
    handle: Mutex<Option<JoinHandle<()>>>,
    queue: Arc<WorkStealingQueue>,
}

/// RAII guard constructed before spawning a worker task so that even if the
/// hosting Tokio runtime is dropped before the worker future is ever polled once,
/// dropping the task marks the worker slot dead and decrements `active_workers`
/// without acquiring any worker-handle mutex.
struct ActiveWorkerGuard {
    state: Arc<AtomicU8>,
    stats: Arc<ExecutorStatsInner>,
    shutdown: Arc<AtomicBool>,
    work_notify: Arc<tokio::sync::Notify>,
    executor: Weak<WorkStealingExecutor>,
}

impl Drop for ActiveWorkerGuard {
    fn drop(&mut self) {
        self.stats.active_workers.fetch_sub(1, Ordering::AcqRel);
        let prev_state = self.state.swap(WORKER_DEAD, Ordering::AcqRel);
        if !self.shutdown.load(Ordering::Acquire) {
            self.work_notify.notify_waiters();
            if prev_state == WORKER_LIVE
                && let Some(exec) = self.executor.upgrade()
            {
                let rt_opt = exec
                    .fallback_runtime
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
                drop(exec);
                if let Some(rt_handle) = rt_opt {
                    let weak_exec = self.executor.clone();
                    drop(rt_handle.spawn(async move {
                        if let Some(exec) = weak_exec.upgrade() {
                            let _ = exec.ensure_workers_running();
                            if exec.stats.parked_workers.load(Ordering::SeqCst) > 0 {
                                exec.work_notify.notify_waiters();
                            }
                        }
                    }));
                }
            }
        }
    }
}

/// RAII guard tracking workers parked on `work_notify` so `submit()` only calls
/// `notify_waiters()` when at least one worker is preparing to park or parked.
struct ParkedWorkerGuard<'a> {
    stats: &'a ExecutorStatsInner,
}

impl Drop for ParkedWorkerGuard<'_> {
    fn drop(&mut self) {
        // ORDERING: SeqCst pairs with SeqCst load in `submit()`.
        self.stats.parked_workers.fetch_sub(1, Ordering::SeqCst);
    }
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
            active_workers: AtomicUsize::new(0),
            parked_workers: AtomicUsize::new(0),
            total_steals: AtomicUsize::new(0),
            total_execution_time_us: AtomicUsize::new(0),
        });
        let shutdown = Arc::new(AtomicBool::new(false));
        let work_notify = Arc::new(tokio::sync::Notify::new());

        // Create worker queues
        for i in 0..num_workers {
            let queue = Arc::new(WorkStealingQueue::new(i, queue_capacity));
            queues.push(queue.clone());

            workers.push(WorkerThread {
                id: i,
                state: Arc::new(AtomicU8::new(WORKER_DEAD)),
                handle: Mutex::new(None),
                queue,
            });
        }

        // ORDERING: Relaxed is sufficient for a monotonic unique executor identifier.
        let executor_id = NEXT_EXECUTOR_ID.fetch_add(1, Ordering::Relaxed);
        let executor = Arc::new_cyclic(|weak| Self {
            executor_id,
            workers,
            queues,
            global_queue,
            stats,
            shutdown,
            work_notify,
            respawn_lock: Mutex::new(()),
            fallback_runtime: Mutex::new(None),
            self_weak: weak.clone(),
            next_worker: AtomicUsize::new(0),
        });

        // Start worker tasks without holding `worker.handle` across `tokio::spawn`, so an
        // immediate synchronous drop (closed runtime or panic unwind) cannot self-deadlock.
        for worker in &executor.workers {
            worker.state.store(WORKER_SPAWNING, Ordering::Release);
            executor.stats.active_workers.fetch_add(1, Ordering::AcqRel);
            let worker_guard = ActiveWorkerGuard {
                state: Arc::clone(&worker.state),
                stats: Arc::clone(&executor.stats),
                shutdown: Arc::clone(&executor.shutdown),
                work_notify: Arc::clone(&executor.work_notify),
                executor: executor.self_weak.clone(),
            };
            let queues = executor.queues.clone();
            let global_queue = executor.global_queue.clone();
            let stats = executor.stats.clone();
            let shutdown = executor.shutdown.clone();
            let work_notify = executor.work_notify.clone();
            let worker_id = worker.id;

            let handle = tokio::spawn(async move {
                Self::worker_loop(
                    worker_id,
                    queues,
                    global_queue,
                    stats,
                    shutdown,
                    work_notify,
                    worker_guard,
                )
                .await;
            });

            match worker.state.compare_exchange(
                WORKER_SPAWNING,
                WORKER_LIVE,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) | Err(WORKER_LIVE) => {
                    *worker.handle.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
                }
                Err(_) => {}
            }
        }

        Ok(executor)
    }

    /// Ensure worker tasks are running on an active Tokio runtime, respawning any
    /// workers whose original runtime was dropped while `self.shutdown` is still false.
    fn ensure_workers_running(&self) -> Result<()> {
        if self.stats.active_workers.load(Ordering::Acquire) == self.workers.len() {
            return Ok(());
        }
        let rt_handle = tokio::runtime::Handle::try_current().map_err(|_| {
            ZiporaError::invalid_state(
                "cannot submit task: executor workers terminated with previous Tokio runtime and no current Tokio runtime is active",
            )
        })?;
        let _respawn_guard = self
            .respawn_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while self.stats.active_workers.load(Ordering::Acquire) < self.workers.len() {
            if self.shutdown.load(Ordering::Acquire) {
                return Err(ZiporaError::invalid_state(
                    "cannot submit task to shut down executor",
                ));
            }
            let mut progressed = false;
            for worker in &self.workers {
                if worker
                    .state
                    .compare_exchange(
                        WORKER_DEAD,
                        WORKER_SPAWNING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_err()
                {
                    continue;
                }
                if self.shutdown.load(Ordering::Acquire) {
                    worker.state.store(WORKER_DEAD, Ordering::Release);
                    return Err(ZiporaError::invalid_state(
                        "cannot submit task to shut down executor",
                    ));
                }
                self.stats.active_workers.fetch_add(1, Ordering::AcqRel);
                let worker_guard = ActiveWorkerGuard {
                    state: Arc::clone(&worker.state),
                    stats: Arc::clone(&self.stats),
                    shutdown: Arc::clone(&self.shutdown),
                    work_notify: Arc::clone(&self.work_notify),
                    executor: self.self_weak.clone(),
                };
                let queues = self.queues.clone();
                let global_queue = self.global_queue.clone();
                let stats = self.stats.clone();
                let shutdown = self.shutdown.clone();
                let work_notify = self.work_notify.clone();
                let worker_id = worker.id;
                let new_handle = rt_handle.spawn(async move {
                    Self::worker_loop(
                        worker_id,
                        queues,
                        global_queue,
                        stats,
                        shutdown,
                        work_notify,
                        worker_guard,
                    )
                    .await;
                });
                match worker.state.compare_exchange(
                    WORKER_SPAWNING,
                    WORKER_LIVE,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) | Err(WORKER_LIVE) => {
                        *worker.handle.lock().unwrap_or_else(|e| e.into_inner()) = Some(new_handle);
                        progressed = true;
                    }
                    Err(_) => {
                        if self.shutdown.load(Ordering::Acquire) {
                            return Err(ZiporaError::invalid_state(
                                "cannot submit task to shut down executor",
                            ));
                        }
                        return Err(ZiporaError::invalid_state(
                            "cannot submit task: current Tokio runtime is shutting down",
                        ));
                    }
                }
            }
            if !progressed {
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    #[inline]
    fn notify_if_parked(&self) {
        if self.stats.active_workers.load(Ordering::Acquire) < self.workers.len()
            && !self.shutdown.load(Ordering::Acquire)
        {
            let _ = self.ensure_workers_running();
        }
        // ORDERING: SeqCst pairs with `parked_workers.fetch_add(1, Ordering::SeqCst)` in
        // `worker_loop` before its pre-park `find_task` re-check so either the worker
        // sees the newly queued task or the submitter sees `parked_workers > 0`.
        if self.stats.parked_workers.load(Ordering::SeqCst) > 0 {
            self.work_notify.notify_waiters();
        }
    }

    /// Submit a task for execution.
    ///
    /// # Errors
    ///
    /// Returns [`ZiporaError::InvalidState`] if the executor has been shut down, if its
    /// worker tasks terminated when a previous Tokio runtime was dropped and no active
    /// Tokio runtime is available on the calling thread to respawn them, or if the
    /// current Tokio runtime is shutting down. Returns [`ZiporaError::Configuration`]
    /// if both the target worker's local queue and the bounded global fallback queue
    /// (10,000 tasks) are full.
    pub fn submit(&self, task: Box<dyn Task>) -> Result<()> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(ZiporaError::invalid_state(
                "cannot submit task to shut down executor",
            ));
        }
        if let Ok(rt_handle) = tokio::runtime::Handle::try_current() {
            let key = (self.executor_id, crate::thread::tokio_runtime_handle_id(&rt_handle));
            LAST_RECORDED_FALLBACK.with(|last| {
                if last.get() != Some(key) {
                    *self
                        .fallback_runtime
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = Some(rt_handle);
                    last.set(Some(key));
                }
            });
        }
        self.ensure_workers_running()?;

        // ORDERING: Relaxed is sufficient because next_worker is an independent round-robin
        // dispatch counter; queue locks provide synchronization for the task itself.
        let worker_id = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len();

        // Capacity + shutdown check and push happen under a single lock acquisition; on a
        // full local queue the task is handed back and requeued globally, so a
        // concurrent fill can never cause it to be dropped unexecuted.
        let task = match self.queues[worker_id]
            .try_push_local_unless_shutdown(&self.shutdown, task)
        {
            Ok(()) => {
                self.notify_if_parked();
                return Ok(());
            }
            Err((_task, true)) => {
                return Err(ZiporaError::invalid_state(
                    "cannot submit task to shut down executor",
                ));
            }
            Err((task, false)) => task,
        };

        // Go straight to global queue with priority ordering
        let mut global_queue = self.global_queue.lock().unwrap_or_else(|e| e.into_inner());
        if self.shutdown.load(Ordering::Acquire) {
            return Err(ZiporaError::invalid_state(
                "cannot submit task to shut down executor",
            ));
        }
        if global_queue.len() < 10000 {
            WorkStealingQueue::insert_by_priority(&mut global_queue, task);
            drop(global_queue);
            self.notify_if_parked();
            Ok(())
        } else {
            drop(global_queue);
            self.notify_if_parked();
            Err(ZiporaError::configuration("all queues full"))
        }
    }

    /// Submit a closure as a task.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Self::submit`]: if the executor
    /// is shut down, if workers cannot be respawned on an active Tokio runtime, or if
    /// both the target local queue and the global queue are full.
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
        *self
            .fallback_runtime
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;

        // Acquire and release each queue lock after storing `shutdown = true` so no
        // concurrent `submit()` can insert a task after this barrier.
        for q in &self.queues {
            drop(q.local_queue.lock().unwrap_or_else(|e| e.into_inner()));
        }
        drop(
            self.global_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        );
        self.work_notify.notify_waiters();

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

        // Drain any residual tasks in case all workers had already parked/exited.
        for q in &self.queues {
            while let Some(task) = q.pop_local_and_activate(&self.stats.active_tasks) {
                Self::run_task_with_stats(task, &self.stats).await;
            }
        }
        loop {
            let task_opt = {
                let mut gq = self.global_queue.lock().unwrap_or_else(|e| e.into_inner());
                let t = gq.pop_front();
                if t.is_some() {
                    self.stats.active_tasks.fetch_add(1, Ordering::AcqRel);
                }
                t
            };
            let Some(task) = task_opt else {
                break;
            };
            Self::run_task_with_stats(task, &self.stats).await;
        }

        Ok(())
    }

    async fn run_task_with_stats(task: Box<dyn Task>, stats: &ExecutorStatsInner) -> usize {
        // `active_tasks` was already incremented before releasing the queue lock;
        // ensure it is always decremented even on panic.
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
            stats,
            start_time: Instant::now(),
        };

        if let Ok(fut) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task.execute()))
        {
            let _ = (CatchUnwindFuture { inner: fut }).await;
        }

        let executed = stats.total_executed.fetch_add(1, Ordering::Release) + 1;
        drop(task_guard);
        executed
    }

    /// Worker loop for processing tasks
    async fn worker_loop(
        worker_id: usize,
        queues: Vec<Arc<WorkStealingQueue>>,
        global_queue: Arc<Mutex<VecDeque<Box<dyn Task>>>>,
        stats: Arc<ExecutorStatsInner>,
        shutdown: Arc<AtomicBool>,
        work_notify: Arc<tokio::sync::Notify>,
        _worker_guard: ActiveWorkerGuard,
    ) {
        let _ = _worker_guard.state.compare_exchange(
            WORKER_SPAWNING,
            WORKER_LIVE,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let my_queue = &queues[worker_id];
        let other_queues: Vec<_> = queues
            .iter()
            .enumerate()
            .filter(|(id, _)| *id != worker_id)
            .map(|(_, queue)| queue.clone())
            .collect();

        let mut idle_count = 0;
        const MAX_YIELD: usize = 4;

        loop {
            let is_shutting_down = shutdown.load(Ordering::Acquire);
            let task = Self::find_task(
                my_queue,
                &other_queues,
                &global_queue,
                &stats,
                is_shutting_down,
            );

            match task {
                Some(task) => {
                    idle_count = 0;
                    let executed = Self::run_task_with_stats(task, &stats).await;

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
                    if idle_count <= MAX_YIELD {
                        tokio::task::yield_now().await;
                    } else {
                        // Register our waiter and increment `parked_workers` before re-checking
                        // the queues and `shutdown` so no concurrent `submit()` or `shutdown()`
                        // notification can be missed between the empty check and parking.
                        let notified = work_notify.notified();
                        tokio::pin!(notified);
                        notified.as_mut().enable();
                        // ORDERING: SeqCst pairs with `parked_workers.load(Ordering::SeqCst)` in `submit()`.
                        stats.parked_workers.fetch_add(1, Ordering::SeqCst);
                        let parked_guard = ParkedWorkerGuard { stats: &stats };
                        let shutting_down_now = shutdown.load(Ordering::Acquire);
                        if let Some(task) = Self::find_task(
                            my_queue,
                            &other_queues,
                            &global_queue,
                            &stats,
                            true,
                        ) {
                            drop(parked_guard);
                            idle_count = 0;
                            let executed = Self::run_task_with_stats(task, &stats).await;
                            if executed.is_multiple_of(100) {
                                my_queue.balance();
                            }
                        } else if shutting_down_now {
                            break;
                        } else {
                            notified.await;
                        }
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
        is_shutting_down: bool,
    ) -> Option<Box<dyn Task>> {
        // 1. Try local queue first (best cache locality)
        if let Some(task) = my_queue.pop_local_and_activate(&stats.active_tasks) {
            return Some(task);
        }

        // 2. Try global queue (lock unconditionally during shutdown drain so contended
        //    submissions are never skipped before a worker exits)
        if is_shutting_down {
            let mut queue = global_queue.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(task) = queue.pop_front() {
                stats.active_tasks.fetch_add(1, Ordering::AcqRel);
                return Some(task);
            }
        } else if let Ok(mut queue) = global_queue.try_lock()
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

    /// Check if the executor is idle (no queued or active tasks).
    /// Scanning queues before reading `active_tasks` pairs with `pop_local_and_activate`
    /// and `steal_and_activate` (which increment `active_tasks` before releasing the queue
    /// lock), preventing a false-positive `true` when a task transitions from queued to active.
    pub fn is_idle(&self) -> bool {
        self.total_queued() == 0 && self.stats.active_tasks.load(Ordering::Acquire) == 0
    }
}

impl Drop for WorkStealingExecutor {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.work_notify.notify_waiters();
        for worker in &self.workers {
            let handle = worker
                .handle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            if let Some(handle) = handle {
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

    /// Spawn a fiber as an independent asynchronous task on the current Tokio runtime,
    /// preserving the fiber's unique ID.
    pub fn spawn<T>(fiber: super::Fiber<T>) -> super::FiberHandle<T>
    where
        T: Send + 'static,
    {
        let id = fiber.id().as_u64();
        let handle = tokio::spawn(fiber);
        super::FiberHandle::new(handle, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_executor_creation() {
        let executor = WorkStealingExecutor::new(4, 1000).unwrap();
        let stats = executor.stats();

        assert_eq!(stats.active_workers, 4);
        assert_eq!(stats.active_tasks, 0);
        assert_eq!(stats.total_executed, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_task_submission() {
        let executor = WorkStealingExecutor::new(2, 1000).unwrap();

        let result = executor.submit_closure(|| Box::pin(async { Ok(()) }));
        assert!(result.is_ok());

        // Wait a bit for task to be processed
        tokio::time::sleep(Duration::from_millis(10)).await;

        let stats = executor.stats();
        assert!(stats.total_executed > 0 || stats.active_tasks > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected() {
        assert!(
            WorkStealingExecutor::new(1, 0).is_err(),
            "WorkStealingExecutor::new with queue_capacity == 0 must return Err"
        );

        let _f1 = crate::concurrency::Fiber::new(async { Ok(1i32) });
        let f2 = crate::concurrency::Fiber::new(async { Ok(2i32) });
        let expected_id = f2.id();
        let handle = WorkStealingExecutor::spawn(f2);
        assert_eq!(
            handle.id(),
            expected_id.as_u64(),
            "WorkStealingExecutor::spawn must preserve the exact FiberId"
        );
        assert_eq!(handle.await.unwrap(), 2);
    }

    #[test]
    fn test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers() {
        // (a) Calling WorkStealingExecutor::init() on a temporary current_thread runtime
        //     whose worker futures are never polled before drop (S27-R2) must decrement
        //     active_workers, reject no-runtime submit_closure, and respawn on a new runtime.
        {
            let rt0 = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            rt0.block_on(async {
                let cfg = crate::concurrency::ConcurrencyConfig { max_fibers: 2, queue_size: 64 };
                WorkStealingExecutor::init(cfg).await.unwrap();
            });
        }

        let global_exec =
            WorkStealingExecutor::global().expect("global executor must be initialized");
        assert_eq!(
            global_exec.stats().active_workers,
            0,
            "dropping a runtime whose workers were never polled must decrement active_workers to 0"
        );
        assert!(
            global_exec.submit_closure(|| Box::pin(async { Ok(()) })).is_err(),
            "submit_closure with 0 live workers and no active Tokio runtime must return Err"
        );

        {
            let rt1 = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            rt1.block_on(async {
                let h = WorkStealingExecutor::spawn(crate::concurrency::Fiber::new(async {
                    Ok(1i32)
                }));
                assert_eq!(h.await.unwrap(), 1);
            });
        }

        let unpolled_ct_exec = {
            let temp_ct = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            temp_ct.block_on(async { WorkStealingExecutor::new(2, 64).unwrap() })
        };
        assert_eq!(unpolled_ct_exec.stats().active_workers, 0);
        assert!(unpolled_ct_exec.submit_closure(|| Box::pin(async { Ok(()) })).is_err());

        // S28-R1: Synchronous future drop inside `tokio::spawn` / `Handle::spawn` must never
        // self-deadlock on `worker.handle`:
        // (D3) `new()` outside any runtime panics cleanly instead of hanging during unwind;
        // (D2) `new()` with a shut-down runtime's handle entered returns `Ok` with 0 active workers;
        // (D1) `submit()` with a shut-down runtime's handle entered returns `Err` without hanging;
        // (D4) a task whose `Drop` submits while its runtime shuts down completes without hanging.
        {
            let (wd_tx, wd_rx) = std::sync::mpsc::channel::<()>();
            let wd_join = std::thread::spawn(move || {
                let no_rt = std::panic::catch_unwind(|| WorkStealingExecutor::new(2, 64));
                assert!(no_rt.is_err(), "WorkStealingExecutor::new outside runtime must panic");

                let closed_rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .unwrap();
                let closed_handle = closed_rt.handle().clone();
                let dead_exec = closed_rt.block_on(async { WorkStealingExecutor::new(2, 64).unwrap() });
                drop(closed_rt);
                assert_eq!(dead_exec.stats().active_workers, 0);
                {
                    let _enter = closed_handle.enter();
                    let d2_exec = WorkStealingExecutor::new(2, 64).unwrap();
                    assert_eq!(d2_exec.stats().active_workers, 0);
                    assert!(
                        dead_exec
                            .submit_closure(|| Box::pin(async { Ok(()) }))
                            .is_err(),
                        "submit with shut-down runtime handle entered must return Err"
                    );
                }

                struct SubmitOnDrop(Arc<WorkStealingExecutor>);
                impl Drop for SubmitOnDrop {
                    fn drop(&mut self) {
                        let _ = self.0.submit_closure(|| Box::pin(async { Ok(()) }));
                    }
                }
                for _ in 0..8 {
                    let drop_rt = tokio::runtime::Builder::new_multi_thread()
                        .worker_threads(2)
                        .enable_all()
                        .build()
                        .unwrap();
                    drop_rt.block_on(async {
                        let ex = WorkStealingExecutor::new(2, 64).unwrap();
                        let sod = SubmitOnDrop(ex);
                        tokio::spawn(async move {
                            let _keep = sod;
                            std::future::pending::<()>().await;
                        });
                        tokio::task::yield_now().await;
                    });
                    drop(drop_rt);
                }
                let _ = wd_tx.send(());
            });
            wd_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("S28-R1 self-deadlock when Tokio synchronously drops worker future during spawn");
            wd_join.join().unwrap();
        }

        let mut dropped_mt_execs = Vec::with_capacity(8);
        for _ in 0..8 {
            let temp_mt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            dropped_mt_execs.push(temp_mt.block_on(async { WorkStealingExecutor::new(2, 64).unwrap() }));
        }

        let rt2 = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt2.block_on(async {
            let h = tokio::time::timeout(
                Duration::from_secs(2),
                WorkStealingExecutor::spawn(crate::concurrency::Fiber::new(async { Ok(2i32) })),
            )
            .await
            .expect("spawn hung after the runtime that called WorkStealingExecutor::init() was dropped");
            assert_eq!(h.unwrap(), 2);

            let (done_tx, done_rx) = tokio::sync::oneshot::channel::<i32>();
            global_exec
                .submit_closure(move || {
                    Box::pin(async move {
                        let _ = done_tx.send(77);
                        Ok(())
                    })
                })
                .expect("global submit_closure on active runtime must succeed");
            let submitted_val = tokio::time::timeout(Duration::from_secs(2), done_rx)
                .await
                .expect("global submit_closure task never ran after init runtime was dropped")
                .unwrap();
            assert_eq!(submitted_val, 77);

            for ex in std::iter::once(unpolled_ct_exec).chain(dropped_mt_execs) {
                let (tx, rx) = tokio::sync::oneshot::channel::<i32>();
                ex.submit_closure(move || {
                    Box::pin(async move {
                        let _ = tx.send(99);
                        Ok(())
                    })
                })
                .expect("submit_closure after runtime drop must respawn workers");
                let v = tokio::time::timeout(Duration::from_secs(2), rx)
                    .await
                    .expect("task stranded after runtime drop")
                    .unwrap();
                assert_eq!(v, 99);
                ex.shutdown().await.unwrap();
            }

            let mut parents = Vec::new();
            for i in 0..4i32 {
                parents.push(WorkStealingExecutor::spawn(crate::concurrency::Fiber::new(
                    async move {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        let child = WorkStealingExecutor::spawn(crate::concurrency::Fiber::new(
                            async move { Ok(i * 10) },
                        ));
                        child.await
                    },
                )));
            }
            for (i, p) in parents.into_iter().enumerate() {
                let v = tokio::time::timeout(Duration::from_secs(2), p)
                    .await
                    .expect("parent fiber awaiting child fiber deadlocked")
                    .unwrap();
                assert_eq!(v, (i as i32) * 10);
            }

            let start = Instant::now();
            let mut sleepers = Vec::new();
            for i in 0..8i32 {
                sleepers.push(WorkStealingExecutor::spawn(crate::concurrency::Fiber::new(
                    async move {
                        tokio::time::sleep(Duration::from_millis(80)).await;
                        Ok(i)
                    },
                )));
            }
            for s in sleepers {
                s.await.unwrap();
            }
            assert!(
                start.elapsed() < Duration::from_millis(280),
                "8 sleeping fibers took {:?}; expected concurrent timer execution",
                start.elapsed()
            );
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_s25_r6_is_idle_ordering_and_submit_shutdown_race() {
        // 1. When is_idle() returns true, every previously submitted task must have finished.
        let exec = Arc::new(WorkStealingExecutor::new(2, 64).unwrap());
        for _ in 0..256 {
            let finished = Arc::new(AtomicBool::new(false));
            let f2 = Arc::clone(&finished);
            exec.submit_closure(move || {
                Box::pin(async move {
                    for _ in 0..64 {
                        std::hint::spin_loop();
                    }
                    f2.store(true, Ordering::Release);
                    Ok(())
                })
            })
            .unwrap();
            while !exec.is_idle() {
                std::hint::spin_loop();
            }
            assert!(
                finished.load(Ordering::Acquire),
                "is_idle() returned true before submitted task finished"
            );
        }
        exec.shutdown().await.unwrap();

        // 2. Concurrent submit() from 8 OS threads racing shutdown() on a small-capacity
        //    executor must execute every accepted task without loss (S25-R6 / S27-R4).
        //    Note: this 256-iteration concurrent stress check detects single-line reverts of
        //    `try_push_local_unless_shutdown` -> `try_push_local` in ~50-63% of runs.
        for it in 0..256usize {
            let exec = Arc::new(WorkStealingExecutor::new(2, 4).unwrap());
            let accepted = Arc::new(AtomicUsize::new(0));
            let executed = Arc::new(AtomicUsize::new(0));
            let ready = Arc::new(AtomicUsize::new(0));
            let go = Arc::new(AtomicBool::new(false));
            let mut submitters = Vec::with_capacity(8);
            for s in 0..8usize {
                let (ex, acc, run, rdy, go) = (
                    Arc::clone(&exec),
                    Arc::clone(&accepted),
                    Arc::clone(&executed),
                    Arc::clone(&ready),
                    Arc::clone(&go),
                );
                submitters.push(std::thread::spawn(move || {
                    rdy.fetch_add(1, Ordering::AcqRel);
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    let mut i = 0usize;
                    loop {
                        let run = Arc::clone(&run);
                        let task = ClosureTask::new(move || {
                            run.fetch_add(1, Ordering::AcqRel);
                            Box::pin(async { Ok(()) })
                        })
                        .with_priority(((s + i) % 3) as u8)
                        .with_stealable(!i.is_multiple_of(5));
                        i = i.wrapping_add(1);
                        match ex.submit(Box::new(task)) {
                            Ok(()) => {
                                acc.fetch_add(1, Ordering::AcqRel);
                            }
                            Err(_) => {
                                if ex.shutdown.load(Ordering::Acquire) {
                                    break;
                                }
                                std::thread::yield_now();
                            }
                        }
                    }
                }));
            }
            while ready.load(Ordering::Acquire) < 8 {
                std::hint::spin_loop();
            }
            go.store(true, Ordering::Release);
            let target_accepted = 16 + (it % 64);
            while accepted.load(Ordering::Acquire) < target_accepted {
                std::hint::spin_loop();
            }
            for _ in 0..((it % 8) * 64) {
                std::hint::spin_loop();
            }
            exec.shutdown().await.unwrap();
            for h in submitters {
                h.join().unwrap();
            }
            assert_eq!(
                accepted.load(Ordering::Acquire),
                executed.load(Ordering::Acquire),
                "every task accepted before shutdown() must execute (iter {it})"
            );
        }
    }

    fn run_s29_r1_fallback_respawn_mode(mode: char) {
        let (wd_tx, wd_rx) = std::sync::mpsc::channel::<()>();
        let wd_join = std::thread::spawn(move || {
            let rt2 = Arc::new(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .unwrap(),
            );
            let rt1 = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            let ex = rt1.block_on(async {
                let ex = WorkStealingExecutor::new(1, 1024).unwrap();
                tokio::time::sleep(Duration::from_millis(5)).await;
                ex
            });

            let ran = Arc::new(AtomicUsize::new(0));
            let accepted = Arc::new(AtomicUsize::new(0));
            {
                let (ex, ran, accepted, rt2) = (
                    Arc::clone(&ex),
                    Arc::clone(&ran),
                    Arc::clone(&accepted),
                    Arc::clone(&rt2),
                );
                std::thread::spawn(move || {
                    let pre = Arc::new(AtomicUsize::new(0));
                    match mode {
                        'b' => {
                            let pre_c = Arc::clone(&pre);
                            ex.submit_closure(move || {
                                pre_c.fetch_add(1, Ordering::AcqRel);
                                Box::pin(async { Ok(()) })
                            })
                            .unwrap();
                        }
                        'c' => {
                            let rt3 = tokio::runtime::Builder::new_multi_thread()
                                .worker_threads(1)
                                .enable_all()
                                .build()
                                .unwrap();
                            {
                                let _enter = rt3.enter();
                                let pre_c = Arc::clone(&pre);
                                ex.submit_closure(move || {
                                    pre_c.fetch_add(1, Ordering::AcqRel);
                                    Box::pin(async { Ok(()) })
                                })
                                .unwrap();
                            }
                            drop(rt3);
                        }
                        _ => unreachable!(),
                    }
                    let deadline = std::time::Instant::now() + Duration::from_millis(500);
                    while pre.load(Ordering::Acquire) < 1
                        && std::time::Instant::now() < deadline
                    {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    assert_eq!(pre.load(Ordering::Acquire), 1);

                    let _enter = rt2.enter();
                    ex.submit_closure(|| {
                        Box::pin(async {
                            tokio::time::sleep(Duration::from_millis(200)).await;
                            Ok(())
                        })
                    })
                    .unwrap();
                    for _ in 0..100usize {
                        let ran_c = Arc::clone(&ran);
                        if ex
                            .submit_closure(move || {
                                ran_c.fetch_add(1, Ordering::AcqRel);
                                Box::pin(async { Ok(()) })
                            })
                            .is_ok()
                        {
                            accepted.fetch_add(1, Ordering::AcqRel);
                        }
                    }
                })
                .join()
                .unwrap();
            }

            std::thread::sleep(Duration::from_millis(10));
            drop(rt1);

            let deadline = std::time::Instant::now() + Duration::from_millis(1000);
            while ran.load(Ordering::Acquire) < accepted.load(Ordering::Acquire)
                && std::time::Instant::now() < deadline
            {
                std::thread::sleep(Duration::from_millis(2));
            }
            let executed = ran.load(Ordering::Acquire);
            let total_accepted = accepted.load(Ordering::Acquire);
            rt2.block_on(async {
                let _ = ex.shutdown().await;
            });
            assert_eq!(
                executed, total_accepted,
                "S29-R1 mode '{mode}': queued backlog stranded after worker runtime drop (executed {executed}/{total_accepted})"
            );
            let _ = wd_tx.send(());
        });
        match wd_rx.recv_timeout(Duration::from_secs(15)) {
            Ok(()) => wd_join.join().unwrap(),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                std::panic::resume_unwind(wd_join.join().unwrap_err());
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                panic!("S29-R1 watchdog timed out (>15s)");
            }
        }
    }

    #[test]
    fn test_s29_r1_fallback_runtime_mode_b_plain_then_live_runtime() {
        run_s29_r1_fallback_respawn_mode('b');
    }

    #[test]
    fn test_s29_r1_fallback_runtime_mode_c_dropped_temp_then_live_runtime() {
        run_s29_r1_fallback_respawn_mode('c');
    }

    #[test]
    #[ignore]
    fn bench_s29_r1_executor_timings() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(8)
            .enable_all()
            .build()
            .unwrap();
        let ex = rt.block_on(async {
            let ex = WorkStealingExecutor::new(8, 1 << 16).unwrap();
            tokio::time::sleep(Duration::from_millis(5)).await;
            ex
        });

        let mut best_1t = Duration::MAX;
        for _ in 0..7 {
            let done = Arc::new(AtomicUsize::new(0));
            let t = std::time::Instant::now();
            for _ in 0..100_000usize {
                loop {
                    let d = Arc::clone(&done);
                    if ex
                        .submit_closure(move || {
                            d.fetch_add(1, Ordering::Relaxed);
                            Box::pin(async { Ok(()) })
                        })
                        .is_ok()
                    {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
            while done.load(Ordering::Relaxed) < 100_000 {
                std::hint::spin_loop();
            }
            best_1t = best_1t.min(t.elapsed());
        }

        let mut best_8t = Duration::MAX;
        for _ in 0..7 {
            let done = Arc::new(AtomicUsize::new(0));
            let total = 8 * 50_000usize;
            let t = std::time::Instant::now();
            let hs: Vec<_> = (0..8)
                .map(|_| {
                    let (ex, done) = (Arc::clone(&ex), Arc::clone(&done));
                    std::thread::spawn(move || {
                        let mut i = 0usize;
                        while i < 50_000 {
                            let d = Arc::clone(&done);
                            if ex
                                .submit_closure(move || {
                                    d.fetch_add(1, Ordering::Relaxed);
                                    Box::pin(async { Ok(()) })
                                })
                                .is_ok()
                            {
                                i += 1;
                            } else {
                                std::thread::yield_now();
                            }
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            while done.load(Ordering::Relaxed) < total {
                std::hint::spin_loop();
            }
            best_8t = best_8t.min(t.elapsed());
        }

        rt.block_on(async {
            ex.shutdown().await.unwrap();
        });
        println!(
            "S29-R1 executor bench: 1t_100k={:.1} ns/task | mp8_400k={:.1} ns/task",
            best_1t.as_nanos() as f64 / 100_000.0,
            best_8t.as_nanos() as f64 / 400_000.0,
        );
    }
}


