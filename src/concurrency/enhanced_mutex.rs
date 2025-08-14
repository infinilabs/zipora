//! Enhanced Mutex implementations - Specialized mutex variants for different use cases
//!
//! This module provides high-performance synchronization primitives optimized for
//! different concurrency patterns and workload characteristics.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex as TokioMutex, RwLock as TokioRwLock};

/// Configuration for enhanced mutex behavior
#[derive(Debug, Clone)]
pub struct MutexConfig {
    /// Enable fairness (FIFO ordering)
    pub fair: bool,
    /// Enable adaptive spinning
    pub adaptive_spinning: bool,
    /// Maximum spin duration before blocking
    pub max_spin_duration: Duration,
    /// Enable priority inheritance
    pub priority_inheritance: bool,
    /// Timeout for lock acquisition
    pub timeout: Option<Duration>,
}

impl Default for MutexConfig {
    fn default() -> Self {
        Self {
            fair: false,
            adaptive_spinning: true,
            max_spin_duration: Duration::from_micros(10),
            priority_inheritance: false,
            timeout: None,
        }
    }
}

/// Statistics for mutex performance monitoring
#[derive(Debug, Clone)]
pub struct MutexStats {
    /// Total number of lock acquisitions
    pub total_acquisitions: u64,
    /// Total number of lock contentions
    pub contentions: u64,
    /// Average lock hold time in microseconds
    pub avg_hold_time_us: u64,
    /// Maximum lock hold time in microseconds
    pub max_hold_time_us: u64,
    /// Number of timeouts
    pub timeouts: u64,
    /// Contention ratio (0.0 to 1.0)
    pub contention_ratio: f64,
}

/// Adaptive mutex using tokio's async mutex with statistics
pub struct AdaptiveMutex<T> {
    inner: TokioMutex<T>,
    stats: Arc<AdaptiveMutexStats>,
    config: MutexConfig,
}

struct AdaptiveMutexStats {
    acquisitions: AtomicU64,
    contentions: AtomicU64,
    total_hold_time_us: AtomicU64,
    max_hold_time_us: AtomicU64,
    timeouts: AtomicU64,
}

impl AdaptiveMutexStats {
    fn new() -> Self {
        Self {
            acquisitions: AtomicU64::new(0),
            contentions: AtomicU64::new(0),
            total_hold_time_us: AtomicU64::new(0),
            max_hold_time_us: AtomicU64::new(0),
            timeouts: AtomicU64::new(0),
        }
    }
}

impl<T> AdaptiveMutex<T> {
    /// Create a new adaptive mutex with default configuration
    pub fn new(value: T) -> Self {
        Self::with_config(value, MutexConfig::default())
    }

    /// Create a new adaptive mutex with custom configuration
    pub fn with_config(value: T, config: MutexConfig) -> Self {
        Self {
            inner: TokioMutex::new(value),
            stats: Arc::new(AdaptiveMutexStats::new()),
            config,
        }
    }

    /// Lock the mutex asynchronously
    pub async fn lock(&self) -> AdaptiveMutexGuard<'_, T> {
        let start_time = Instant::now();
        self.stats.acquisitions.fetch_add(1, Ordering::Relaxed);

        let guard = if let Some(timeout) = self.config.timeout {
            match tokio::time::timeout(timeout, self.inner.lock()).await {
                Ok(guard) => guard,
                Err(_) => {
                    self.stats.timeouts.fetch_add(1, Ordering::Relaxed);
                    panic!("mutex lock timeout");
                }
            }
        } else {
            self.inner.lock().await
        };

        let acquisition_time = start_time.elapsed();
        if acquisition_time > Duration::from_micros(1) {
            self.stats.contentions.fetch_add(1, Ordering::Relaxed);
        }

        AdaptiveMutexGuard {
            inner: guard,
            stats: self.stats.clone(),
            start_time,
        }
    }

    /// Try to lock the mutex without blocking
    pub fn try_lock(&self) -> Option<AdaptiveMutexGuard<'_, T>> {
        let start_time = Instant::now();
        self.stats.acquisitions.fetch_add(1, Ordering::Relaxed);

        let guard = self.inner.try_lock().ok()?;

        Some(AdaptiveMutexGuard {
            inner: guard,
            stats: self.stats.clone(),
            start_time,
        })
    }

    /// Get mutex performance statistics
    pub fn stats(&self) -> MutexStats {
        let acquisitions = self.stats.acquisitions.load(Ordering::Relaxed);
        let contentions = self.stats.contentions.load(Ordering::Relaxed);
        let total_hold_time = self.stats.total_hold_time_us.load(Ordering::Relaxed);

        let avg_hold_time_us = if acquisitions > 0 {
            total_hold_time / acquisitions
        } else {
            0
        };

        let contention_ratio = if acquisitions > 0 {
            contentions as f64 / acquisitions as f64
        } else {
            0.0
        };

        MutexStats {
            total_acquisitions: acquisitions,
            contentions,
            avg_hold_time_us,
            max_hold_time_us: self.stats.max_hold_time_us.load(Ordering::Relaxed),
            timeouts: self.stats.timeouts.load(Ordering::Relaxed),
            contention_ratio,
        }
    }
}

/// Guard for adaptive mutex
pub struct AdaptiveMutexGuard<'a, T> {
    inner: tokio::sync::MutexGuard<'a, T>,
    stats: Arc<AdaptiveMutexStats>,
    start_time: Instant,
}

impl<T> std::ops::Deref for AdaptiveMutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl<T> std::ops::DerefMut for AdaptiveMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.inner
    }
}

impl<T> Drop for AdaptiveMutexGuard<'_, T> {
    fn drop(&mut self) {
        let hold_time = self.start_time.elapsed().as_micros() as u64;
        self.stats
            .total_hold_time_us
            .fetch_add(hold_time, Ordering::Relaxed);

        // Update max hold time
        let mut current_max = self.stats.max_hold_time_us.load(Ordering::Relaxed);
        while hold_time > current_max {
            match self.stats.max_hold_time_us.compare_exchange_weak(
                current_max,
                hold_time,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }
}

/// High-performance spin lock for short critical sections
pub struct SpinLock<T> {
    locked: AtomicBool,
    data: std::cell::UnsafeCell<T>,
    waiters: AtomicUsize,
}

unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    /// Create a new spin lock
    pub fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: std::cell::UnsafeCell::new(value),
            waiters: AtomicUsize::new(0),
        }
    }

    /// Lock the spin lock asynchronously with yielding
    pub async fn lock(&self) -> SpinLockGuard<T> {
        self.waiters.fetch_add(1, Ordering::Relaxed);

        let mut spin_count = 0;
        const MAX_SPINS: usize = 100;

        loop {
            // Try to acquire the lock
            if self.locked.compare_exchange_weak(
                false,
                true,
                Ordering::Acquire,
                Ordering::Relaxed,
            ).is_ok() {
                self.waiters.fetch_sub(1, Ordering::Relaxed);
                return SpinLockGuard { lock: self };
            }

            spin_count += 1;

            if spin_count < MAX_SPINS {
                // Spin for a short time
                std::hint::spin_loop();
            } else {
                // Yield to the scheduler to avoid wasting CPU
                tokio::task::yield_now().await;
                spin_count = 0;
            }
        }
    }

    /// Try to lock the spin lock without spinning
    pub fn try_lock(&self) -> Option<SpinLockGuard<T>> {
        if self.locked.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ).is_ok() {
            Some(SpinLockGuard { lock: self })
        } else {
            None
        }
    }

    /// Get the number of waiting threads
    pub fn waiter_count(&self) -> usize {
        self.waiters.load(Ordering::Relaxed)
    }

    /// Check if the lock is currently held
    pub fn is_locked(&self) -> bool {
        self.locked.load(Ordering::Relaxed)
    }
}

/// Guard for spin lock
pub struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> std::ops::Deref for SpinLockGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> std::ops::DerefMut for SpinLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for SpinLockGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

/// Reader-writer lock with priority options
pub struct PriorityRwLock<T> {
    inner: TokioRwLock<T>,
    read_count: AtomicUsize,
    write_waiters: AtomicUsize,
    stats: Arc<RwLockStats>,
    config: RwLockConfig,
}

/// Configuration for reader-writer locks
#[derive(Debug, Clone)]
pub struct RwLockConfig {
    /// Prefer writers over readers
    pub writer_priority: bool,
    /// Maximum number of concurrent readers
    pub max_readers: Option<usize>,
    /// Enable fair scheduling
    pub fair: bool,
}

impl Default for RwLockConfig {
    fn default() -> Self {
        Self {
            writer_priority: false,
            max_readers: None,
            fair: true,
        }
    }
}

struct RwLockStats {
    read_acquisitions: AtomicU64,
    write_acquisitions: AtomicU64,
    read_contentions: AtomicU64,
    write_contentions: AtomicU64,
}

impl RwLockStats {
    fn new() -> Self {
        Self {
            read_acquisitions: AtomicU64::new(0),
            write_acquisitions: AtomicU64::new(0),
            read_contentions: AtomicU64::new(0),
            write_contentions: AtomicU64::new(0),
        }
    }
}

impl<T> PriorityRwLock<T> {
    /// Create a new priority reader-writer lock
    pub fn new(value: T) -> Self {
        Self::with_config(value, RwLockConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(value: T, config: RwLockConfig) -> Self {
        Self {
            inner: TokioRwLock::new(value),
            read_count: AtomicUsize::new(0),
            write_waiters: AtomicUsize::new(0),
            stats: Arc::new(RwLockStats::new()),
            config,
        }
    }

    /// Acquire a read lock
    pub async fn read(&self) -> tokio::sync::RwLockReadGuard<'_, T> {
        let start_time = Instant::now();
        self.stats.read_acquisitions.fetch_add(1, Ordering::Relaxed);

        // Check writer priority
        if self.config.writer_priority && self.write_waiters.load(Ordering::Relaxed) > 0 {
            // Wait for writers to finish
            while self.write_waiters.load(Ordering::Relaxed) > 0 {
                tokio::task::yield_now().await;
            }
        }

        // Check reader limit
        if let Some(max_readers) = self.config.max_readers {
            while self.read_count.load(Ordering::Relaxed) >= max_readers {
                tokio::task::yield_now().await;
            }
        }

        let guard = self.inner.read().await;
        self.read_count.fetch_add(1, Ordering::Relaxed);

        let acquisition_time = start_time.elapsed();
        if acquisition_time > Duration::from_micros(1) {
            self.stats.read_contentions.fetch_add(1, Ordering::Relaxed);
        }

        guard
    }

    /// Acquire a write lock
    pub async fn write(&self) -> tokio::sync::RwLockWriteGuard<'_, T> {
        let start_time = Instant::now();
        self.stats.write_acquisitions.fetch_add(1, Ordering::Relaxed);
        self.write_waiters.fetch_add(1, Ordering::Relaxed);

        let guard = self.inner.write().await;
        self.write_waiters.fetch_sub(1, Ordering::Relaxed);

        let acquisition_time = start_time.elapsed();
        if acquisition_time > Duration::from_micros(1) {
            self.stats.write_contentions.fetch_add(1, Ordering::Relaxed);
        }

        guard
    }

    /// Try to acquire a read lock without blocking
    pub fn try_read(&self) -> Option<tokio::sync::RwLockReadGuard<'_, T>> {
        if self.config.writer_priority && self.write_waiters.load(Ordering::Relaxed) > 0 {
            return None;
        }

        if let Some(max_readers) = self.config.max_readers {
            if self.read_count.load(Ordering::Relaxed) >= max_readers {
                return None;
            }
        }

        let guard = self.inner.try_read().ok()?;
        self.read_count.fetch_add(1, Ordering::Relaxed);
        self.stats.read_acquisitions.fetch_add(1, Ordering::Relaxed);
        Some(guard)
    }

    /// Try to acquire a write lock without blocking
    pub fn try_write(&self) -> Option<tokio::sync::RwLockWriteGuard<'_, T>> {
        let guard = self.inner.try_write().ok()?;
        self.stats.write_acquisitions.fetch_add(1, Ordering::Relaxed);
        Some(guard)
    }

    /// Get current reader count
    pub fn reader_count(&self) -> usize {
        self.read_count.load(Ordering::Relaxed)
    }

    /// Get current writer waiter count
    pub fn writer_waiters(&self) -> usize {
        self.write_waiters.load(Ordering::Relaxed)
    }
}

/// Segmented mutex for reducing contention in high-concurrency scenarios
pub struct SegmentedMutex<T> {
    segments: Vec<AdaptiveMutex<T>>,
    segment_count: usize,
}

impl<T: Clone> SegmentedMutex<T> {
    /// Create a new segmented mutex
    pub fn new(value: T, segment_count: usize) -> Self {
        let segments = (0..segment_count)
            .map(|_| AdaptiveMutex::new(value.clone()))
            .collect();

        Self {
            segments,
            segment_count,
        }
    }

    /// Lock a specific segment
    pub async fn lock_segment(&self, segment: usize) -> AdaptiveMutexGuard<T> {
        self.segments[segment % self.segment_count].lock().await
    }

    /// Lock segment based on hash
    pub async fn lock_for_key<K: std::hash::Hash>(&self, key: &K) -> AdaptiveMutexGuard<T> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        let segment = (hash as usize) % self.segment_count;

        self.lock_segment(segment).await
    }

    /// Get the number of segments
    pub fn segment_count(&self) -> usize {
        self.segment_count
    }

    /// Get aggregated statistics across all segments
    pub fn aggregate_stats(&self) -> MutexStats {
        let mut total_acquisitions = 0;
        let mut total_contentions = 0;
        let mut total_hold_time = 0;
        let mut max_hold_time = 0;
        let mut total_timeouts = 0;

        for segment in &self.segments {
            let stats = segment.stats();
            total_acquisitions += stats.total_acquisitions;
            total_contentions += stats.contentions;
            total_hold_time += stats.avg_hold_time_us * stats.total_acquisitions;
            max_hold_time = max_hold_time.max(stats.max_hold_time_us);
            total_timeouts += stats.timeouts;
        }

        let avg_hold_time_us = if total_acquisitions > 0 {
            total_hold_time / total_acquisitions
        } else {
            0
        };

        let contention_ratio = if total_acquisitions > 0 {
            total_contentions as f64 / total_acquisitions as f64
        } else {
            0.0
        };

        MutexStats {
            total_acquisitions,
            contentions: total_contentions,
            avg_hold_time_us,
            max_hold_time_us: max_hold_time,
            timeouts: total_timeouts,
            contention_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn test_adaptive_mutex() {
        let mutex = AdaptiveMutex::new(42);
        
        {
            let guard = mutex.lock().await;
            assert_eq!(*guard, 42);
        }

        let stats = mutex.stats();
        assert_eq!(stats.total_acquisitions, 1);
    }

    #[tokio::test]
    async fn test_adaptive_mutex_try_lock() {
        let mutex = Arc::new(AdaptiveMutex::new(42));
        
        let guard = mutex.try_lock().unwrap();
        assert_eq!(*guard, 42);

        // Should fail while lock is held
        assert!(mutex.try_lock().is_none());
        
        drop(guard);
        
        // Should succeed after lock is released
        assert!(mutex.try_lock().is_some());
    }

    #[tokio::test]
    async fn test_spin_lock() {
        let spin_lock = SpinLock::new(100);
        
        {
            let guard = spin_lock.lock().await;
            assert_eq!(*guard, 100);
        }

        assert!(!spin_lock.is_locked());
        assert_eq!(spin_lock.waiter_count(), 0);
    }

    #[tokio::test]
    async fn test_spin_lock_contention() {
        let spin_lock = Arc::new(SpinLock::new(0));
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let lock = spin_lock.clone();
                tokio::spawn(async move {
                    let mut guard = lock.lock().await;
                    *guard += i;
                    tokio::time::sleep(Duration::from_millis(1)).await;
                })
            })
            .collect();

        for handle in handles {
            handle.await.unwrap();
        }

        let guard = spin_lock.lock().await;
        assert_eq!(*guard, 45); // Sum of 0..10
    }

    #[tokio::test]
    async fn test_priority_rwlock() {
        let rwlock = PriorityRwLock::new(vec![1, 2, 3]);
        
        // Multiple readers should work
        let read1 = rwlock.read().await;
        let read2 = rwlock.read().await;
        
        assert_eq!(read1[0], 1);
        assert_eq!(read2[0], 1);
        assert_eq!(rwlock.reader_count(), 2);
        
        drop(read1);
        drop(read2);

        // Writer should work
        {
            let mut write = rwlock.write().await;
            write.push(4);
        }

        let read = rwlock.read().await;
        assert_eq!(read.len(), 4);
    }

    #[tokio::test]
    async fn test_segmented_mutex() {
        let segmented = Arc::new(SegmentedMutex::new(0, 4));
        
        // Lock different segments concurrently
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let segmented = segmented.clone();
                let seg = segmented.segment_count();
                tokio::spawn(async move {
                    let mut guard = segmented.lock_segment(i % seg).await;
                    *guard += 1;
                })
            })
            .collect();

        for handle in handles {
            handle.await.unwrap();
        }

        let stats = segmented.aggregate_stats();
        assert_eq!(stats.total_acquisitions, 8);
    }

    #[tokio::test]
    async fn test_segmented_mutex_key_based() {
        let segmented = SegmentedMutex::new(String::new(), 2);
        
        {
            let mut guard = segmented.lock_for_key(&"test_key").await;
            guard.push_str("hello");
        }

        {
            let guard = segmented.lock_for_key(&"test_key").await;
            assert_eq!(*guard, "hello");
        }
    }

    #[tokio::test]
    async fn test_mutex_config() {
        let config = MutexConfig {
            fair: true,
            adaptive_spinning: false,
            timeout: Some(Duration::from_millis(100)),
            ..Default::default()
        };

        let mutex = AdaptiveMutex::with_config(42, config);
        let guard = mutex.lock().await;
        assert_eq!(*guard, 42);
    }

    #[tokio::test]
    async fn test_rwlock_writer_priority() {
        let config = RwLockConfig {
            writer_priority: true,
            ..Default::default()
        };

        let rwlock = Arc::new(PriorityRwLock::with_config(0, config));
        
        // Start a long-running reader
        let _read_guard = rwlock.read().await;
        
        // Writer should be prioritized
        let rwlock_clone = rwlock.clone();
        let write_task = tokio::spawn(async move {
            let mut guard = rwlock_clone.write().await;
            *guard = 42;
        });

        // Give writer time to register as waiting
        tokio::time::sleep(Duration::from_millis(1)).await;
        
        // New readers should wait for writer
        assert!(rwlock.try_read().is_none());
        
        drop(_read_guard);
        write_task.await.unwrap();
    }
}