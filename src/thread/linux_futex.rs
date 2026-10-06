//! Linux Futex Integration
//!
//! High-performance synchronization primitives using direct futex syscalls.
//! This provides zero-overhead synchronization for Linux platforms.

use super::PlatformSync;
use crate::error::{Result, ZiporaError};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

#[cfg(target_os = "linux")]
mod sys {
    use libc::{SYS_futex, syscall, timespec};
    use std::ffi::c_int;
    use std::ptr;
    use std::time::Duration;

    // Futex operation constants

    pub const FUTEX_WAIT_PRIVATE: c_int = 128;
    pub const FUTEX_WAKE_PRIVATE: c_int = 129;

    /// Direct futex syscall wrapper
    ///
    /// # Safety
    /// `uaddr` (and `uaddr2` when non-null) must point to a valid, 4-byte-aligned
    /// `u32` (typically an `AtomicU32`), and `timeout` must either be null or point
    /// to a valid, initialized `libc::timespec` for the duration of the syscall.
    #[inline]
    pub unsafe fn futex(
        uaddr: *const u32,
        op: c_int,
        val: u32,
        timeout: *const timespec,
        uaddr2: *const u32,
        val3: u32,
    ) -> c_int {
        // SAFETY: Caller guarantees `uaddr`, `timeout`, and `uaddr2` satisfy the kernel
        // `SYS_futex` pointer validity and alignment preconditions. Passing pointer arguments
        // directly preserves pointer provenance, and passing `op: c_int`, `val: u32`, and
        // `val3: u32` matches the variadic `futex(2)` ABI expected by libc/Miri.
        unsafe { syscall(SYS_futex, uaddr, op, val, timeout, uaddr2, val3) as c_int }
    }

    /// Wait on futex with optional timeout
    ///
    /// # Safety
    /// `uaddr` must point to a valid, 4-byte-aligned `u32` (such as `AtomicU32::as_ptr`).
    #[inline]
    pub unsafe fn futex_wait(uaddr: *const u32, val: u32, timeout: Option<Duration>) -> c_int {
        let ts: timespec;
        let timeout_ptr = match timeout {
            Some(d) => {
                ts = timespec {
                    tv_sec: d.as_secs().min(i64::MAX as u64) as i64,
                    tv_nsec: d.subsec_nanos() as i64,
                };
                &raw const ts
            }
            None => ptr::null(),
        };

        // SAFETY: `uaddr` is a valid atomic `u32` pointer and `timeout_ptr` is either null or
        // points to `ts`, which lives in the enclosing stack frame across the `futex` call.
        unsafe {
            futex(
                uaddr,
                FUTEX_WAIT_PRIVATE,
                val,
                timeout_ptr,
                std::ptr::null(),
                0,
            )
        }
    }

    /// Wake waiters on futex
    ///
    /// # Safety
    /// `uaddr` must point to a valid, 4-byte-aligned `u32` (such as `AtomicU32::as_ptr`).
    #[inline]
    pub unsafe fn futex_wake(uaddr: *const u32, count: u32) -> c_int {
        // Linux `futex_wake` interprets `val` as a signed 32-bit `int nr_wake` and stops
        // as soon as `++woken >= nr_wake`; clamping to `i32::MAX as u32` prevents `u32::MAX`
        // (`-1`) from waking only a single waiter.
        let clamped = count.min(i32::MAX as u32);
        // SAFETY: `uaddr` is a valid atomic `u32` pointer.
        unsafe {
            futex(
                uaddr,
                FUTEX_WAKE_PRIVATE,
                clamped,
                std::ptr::null(),
                std::ptr::null(),
                0,
            )
        }
    }
}

/// Linux-specific futex implementation
pub struct LinuxFutex;

impl PlatformSync for LinuxFutex {
    fn futex_wait(addr: &AtomicU32, val: u32, timeout: Option<Duration>) -> Result<()> {
        // SAFETY: addr is valid AtomicU32, as_ptr() returns valid pointer
        unsafe {
            let result = sys::futex_wait(addr.as_ptr(), val, timeout);
            if result == -1 {
                let errno = *libc::__errno_location();
                match errno {
                    libc::EAGAIN => Ok(()), // Value changed before wait
                    libc::ETIMEDOUT => Err(ZiporaError::timeout("Futex wait timed out")),
                    libc::EINTR => Ok(()), // Interrupted by signal
                    _ => Err(ZiporaError::system_error(format!(
                        "Futex wait failed: errno {}",
                        errno
                    ))),
                }
            } else {
                Ok(())
            }
        }
    }

    fn futex_wake(addr: &AtomicU32, count: u32) -> Result<usize> {
        // SAFETY: addr is valid AtomicU32, as_ptr() returns valid pointer
        unsafe {
            let result = sys::futex_wake(addr.as_ptr(), count);
            if result == -1 {
                let errno = *libc::__errno_location();
                Err(ZiporaError::system_error(format!(
                    "Futex wake failed: errno {}",
                    errno
                )))
            } else {
                Ok(result as usize)
            }
        }
    }
}

/// High-performance mutex using futex
pub struct FutexMutex {
    /// State: 0 = unlocked, 1 = locked (no waiters), 2 = locked (with waiters)
    state: AtomicU32,
}

impl FutexMutex {
    /// Create a new futex mutex
    pub fn new() -> Self {
        Self {
            state: AtomicU32::new(0),
        }
    }

    /// Lock the mutex
    pub fn lock(&self) -> Result<FutexGuard<'_>> {
        // Fast path: try to acquire lock immediately.
        // ORDERING: Acquire on success synchronizes with the Release swap in `unlock`;
        // Relaxed on failure is sufficient because `lock_slow` immediately performs an
        // Acquire swap before inspecting or waiting on `self.state`.
        if self
            .state
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return Ok(FutexGuard { mutex: self });
        }

        // Slow path: contended lock
        self.lock_slow()?;
        Ok(FutexGuard { mutex: self })
    }

    /// Try to lock the mutex without blocking
    pub fn try_lock(&self) -> Result<Option<FutexGuard<'_>>> {
        // ORDERING: Strong CAS with Acquire on success synchronizes with `unlock`;
        // Relaxed on failure is sufficient because no lock-protected state is accessed when
        // `try_lock` returns `Ok(None)`.
        match self
            .state
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => Ok(Some(FutexGuard { mutex: self })),
            Err(_) => Ok(None),
        }
    }

    /// Slow path for contended lock
    fn lock_slow(&self) -> Result<()> {
        loop {
            // Set state to 2 (locked with waiters)
            let state = self.state.swap(2, Ordering::Acquire);
            if state == 0 {
                // Lock was released between our attempts
                return Ok(());
            }

            // Wait for the lock to be released
            LinuxFutex::futex_wait(&self.state, 2, None)?;
        }
    }

    /// Unlock the mutex
    fn unlock(&self) {
        let old_state = self.state.swap(0, Ordering::Release);
        if old_state == 2 {
            // There were waiters, wake one
            let _ = LinuxFutex::futex_wake(&self.state, 1);
        }
    }
}

impl Default for FutexMutex {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: FutexMutex is Send because:
// 1. `state: AtomicU32` - AtomicU32 is Send, no thread-local state.
// The mutex can safely be transferred between threads.
unsafe impl Send for FutexMutex {}

// SAFETY: FutexMutex is Sync because:
// 1. `state: AtomicU32` - All state transitions use atomic operations.
// 2. Lock acquisition uses CAS with Acquire ordering.
// 3. Lock release uses swap with Release ordering.
// 4. Futex syscalls provide kernel-level synchronization for waiters.
// The mutex protocol ensures mutual exclusion across threads.
unsafe impl Sync for FutexMutex {}

/// RAII guard for FutexMutex
pub struct FutexGuard<'a> {
    mutex: &'a FutexMutex,
}

impl Drop for FutexGuard<'_> {
    fn drop(&mut self) {
        self.mutex.unlock();
    }
}

/// High-performance condition variable using futex
pub struct FutexCondvar {
    futex: AtomicU32,
}

impl FutexCondvar {
    /// Create a new futex condition variable
    pub fn new() -> Self {
        Self {
            futex: AtomicU32::new(0),
        }
    }

    /// Wait on the condition variable
    pub fn wait<'a>(&self, guard: FutexGuard<'a>) -> Result<FutexGuard<'a>> {
        // ORDERING: Acquire pairs with the Release `fetch_add` in `notify_one`/`notify_all`
        // and prevents the sequence read from being reordered after the mutex unlock in `drop(guard)`.
        let futex_val = self.futex.load(Ordering::Acquire);

        // Release the mutex
        let mutex = guard.mutex;
        drop(guard);

        // Wait on the condition variable
        let wait_res = LinuxFutex::futex_wait(&self.futex, futex_val, None);

        // Reacquire the mutex before propagating any wait error
        let guard = mutex.lock()?;
        wait_res?;
        Ok(guard)
    }

    /// Wait on the condition variable with timeout
    pub fn wait_timeout<'a>(
        &self,
        guard: FutexGuard<'a>,
        timeout: Duration,
    ) -> Result<(FutexGuard<'a>, bool)> {
        // ORDERING: Acquire pairs with the Release `fetch_add` in `notify_one`/`notify_all`
        // and prevents the sequence read from being reordered after the mutex unlock in `drop(guard)`.
        let futex_val = self.futex.load(Ordering::Acquire);

        // Release the mutex
        let mutex = guard.mutex;
        drop(guard);

        // Wait on the condition variable with timeout
        let wait_res = LinuxFutex::futex_wait(&self.futex, futex_val, Some(timeout));

        // Reacquire the mutex
        let guard = mutex.lock()?;
        match wait_res {
            Ok(()) => Ok((guard, false)),
            Err(ZiporaError::Timeout { .. }) => Ok((guard, true)),
            Err(e) => Err(e),
        }
    }

    /// Notify one waiter
    pub fn notify_one(&self) -> Result<()> {
        // ORDERING: Release publishes prior predicate modifications before advancing the
        // sequence counter observed by `wait`/`wait_timeout` and `FUTEX_WAIT_PRIVATE`.
        self.futex.fetch_add(1, Ordering::Release);
        LinuxFutex::futex_wake(&self.futex, 1)?;
        Ok(())
    }

    /// Notify all waiters
    pub fn notify_all(&self) -> Result<()> {
        // ORDERING: Release publishes prior predicate modifications before advancing the
        // sequence counter observed by `wait`/`wait_timeout` and `FUTEX_WAIT_PRIVATE`.
        self.futex.fetch_add(1, Ordering::Release);
        LinuxFutex::futex_wake(&self.futex, i32::MAX as u32)?;
        Ok(())
    }
}

impl Default for FutexCondvar {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: FutexCondvar is Send because:
// 1. `futex: AtomicU32` - AtomicU32 is Send, no thread-local state.
unsafe impl Send for FutexCondvar {}

// SAFETY: FutexCondvar is Sync because:
// 1. `futex: AtomicU32` - All operations use atomic updates.
// 2. wait() releases the mutex and blocks atomically via futex syscall.
// 3. notify_one()/notify_all() use futex_wake for kernel-level wakeup.
// The condvar protocol is designed for cross-thread signaling.
unsafe impl Sync for FutexCondvar {}

/// High-performance reader-writer lock using futex
pub struct FutexRwLock {
    /// Bit layout: [31-16: writer_id] [15-0: reader_count]
    /// Special values: 0 = unlocked, 0x80000000 = write-locked
    state: AtomicU32,
}

impl FutexRwLock {
    /// Create a new futex reader-writer lock
    pub fn new() -> Self {
        Self {
            state: AtomicU32::new(0),
        }
    }

    /// Acquire a read lock
    pub fn read(&self) -> Result<FutexReadGuard<'_>> {
        loop {
            let state = self.state.load(Ordering::Acquire);

            // Check if write-locked
            if state & 0x80000000 != 0 {
                LinuxFutex::futex_wait(&self.state, state, None)?;
                continue;
            }

            // Try to increment reader count
            let reader_count = state & 0xFFFF;
            if reader_count == 0xFFFF {
                return Err(ZiporaError::resource_exhausted("Too many readers"));
            }

            let new_state = state + 1;
            // ORDERING: Acquire on success synchronizes with `unlock_write`; Relaxed on
            // failure is sufficient because the loop immediately reloads `self.state` with Acquire.
            if self
                .state
                .compare_exchange_weak(state, new_state, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(FutexReadGuard { lock: self });
            }
        }
    }

    /// Acquire a write lock
    pub fn write(&self) -> Result<FutexWriteGuard<'_>> {
        loop {
            let state = self.state.load(Ordering::Acquire);

            // Try to acquire write lock (state must be 0)
            if state == 0 {
                // ORDERING: Acquire on success synchronizes with prior `unlock_read`/`unlock_write`;
                // Relaxed on failure is sufficient because the loop immediately reloads `self.state` with Acquire.
                if self
                    .state
                    .compare_exchange_weak(0, 0x80000000, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    return Ok(FutexWriteGuard { lock: self });
                }
            } else {
                LinuxFutex::futex_wait(&self.state, state, None)?;
            }
        }
    }

    /// Release a read lock
    fn unlock_read(&self) {
        let old_state = self.state.fetch_sub(1, Ordering::Release);
        let reader_count = old_state & 0xFFFF;

        // If this was the last reader, wake writers
        if reader_count == 1 {
            let _ = LinuxFutex::futex_wake(&self.state, i32::MAX as u32);
        }
    }

    /// Release a write lock
    fn unlock_write(&self) {
        self.state.store(0, Ordering::Release);
        let _ = LinuxFutex::futex_wake(&self.state, i32::MAX as u32);
    }
}

impl Default for FutexRwLock {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: FutexRwLock is Send because:
// 1. `state: AtomicU32` - AtomicU32 is Send, no thread-local state.
// The lock can safely be transferred between threads.
unsafe impl Send for FutexRwLock {}

// SAFETY: FutexRwLock is Sync because:
// 1. `state: AtomicU32` - All state transitions use atomic operations.
// 2. Reader count tracked in lower bits, writer flag in upper bit.
// 3. Acquire/Release ordering ensures proper happens-before relationships.
// 4. Futex syscalls provide kernel-level synchronization for waiters.
// The RwLock protocol ensures reader-writer exclusion across threads.
unsafe impl Sync for FutexRwLock {}

/// RAII guard for read access
pub struct FutexReadGuard<'a> {
    lock: &'a FutexRwLock,
}

impl Drop for FutexReadGuard<'_> {
    fn drop(&mut self) {
        self.lock.unlock_read();
    }
}

/// RAII guard for write access
pub struct FutexWriteGuard<'a> {
    lock: &'a FutexRwLock,
}

impl Drop for FutexWriteGuard<'_> {
    fn drop(&mut self) {
        self.lock.unlock_write();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_futex_mutex_basic() {
        let mutex = FutexMutex::new();
        let _guard = mutex.lock().unwrap();
        assert!(mutex.try_lock().unwrap().is_none());
        // Guard should automatically unlock on drop
    }

    #[test]
    fn test_futex_mutex_contention() {
        let mutex = Arc::new(FutexMutex::new());
        let counter = Arc::new(AtomicU32::new(0));
        let num_threads = if cfg!(miri) { 3 } else { 10 };
        let iters = if cfg!(miri) { 10 } else { 100 };

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let mutex = Arc::clone(&mutex);
                let counter = Arc::clone(&counter);

                thread::spawn(move || {
                    for _ in 0..iters {
                        let _guard = mutex.lock().unwrap();
                        let old = counter.load(Ordering::Relaxed);
                        if !cfg!(miri) {
                            thread::sleep(Duration::from_nanos(1));
                        }
                        counter.store(old + 1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.load(Ordering::Relaxed), num_threads * iters);
    }

    #[test]
    fn test_futex_condvar() {
        let mutex = Arc::new(FutexMutex::new());
        let condvar = Arc::new(FutexCondvar::new());
        let started = Arc::new(AtomicU32::new(0));

        let mutex_clone = Arc::clone(&mutex);
        let condvar_clone = Arc::clone(&condvar);
        let started_clone = Arc::clone(&started);

        let handle = thread::spawn(move || {
            let guard = mutex_clone.lock().unwrap();
            started_clone.store(1, Ordering::Release);
            let _guard = condvar_clone.wait(guard).unwrap();
        });

        // Wait for thread to start
        while started.load(Ordering::Acquire) == 0 {
            thread::yield_now();
        }

        thread::sleep(Duration::from_millis(10));
        condvar.notify_one().unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn test_futex_rwlock() {
        let lock = Arc::new(FutexRwLock::new());
        let counter = Arc::new(AtomicU32::new(0));
        let num_readers = if cfg!(miri) { 2 } else { 5 };
        let read_iters = if cfg!(miri) { 10 } else { 100 };
        let write_iters = if cfg!(miri) { 10 } else { 50 };

        // Spawn multiple readers
        let read_handles: Vec<_> = (0..num_readers)
            .map(|_| {
                let lock = Arc::clone(&lock);
                let counter = Arc::clone(&counter);

                thread::spawn(move || {
                    for _ in 0..read_iters {
                        let _guard = lock.read().unwrap();
                        let _val = counter.load(Ordering::Relaxed);
                        if !cfg!(miri) {
                            thread::sleep(Duration::from_nanos(10));
                        }
                    }
                })
            })
            .collect();

        // Spawn a writer
        let write_handle = {
            let lock = Arc::clone(&lock);
            let counter = Arc::clone(&counter);

            thread::spawn(move || {
                for _ in 0..write_iters {
                    let _guard = lock.write().unwrap();
                    let old = counter.load(Ordering::Relaxed);
                    if !cfg!(miri) {
                        thread::sleep(Duration::from_nanos(100));
                    }
                    counter.store(old + 1, Ordering::Relaxed);
                }
            })
        };

        for handle in read_handles {
            handle.join().unwrap();
        }
        write_handle.join().unwrap();

        assert_eq!(counter.load(Ordering::Relaxed), write_iters);
    }

    #[test]
    fn test_c10_d_notify_all_wakes_all_waiters() {
        let futex_word = Arc::new(AtomicU32::new(0));
        let ready = Arc::new(AtomicU32::new(0));
        let woken = Arc::new(AtomicU32::new(0));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let futex_word = Arc::clone(&futex_word);
            let ready = Arc::clone(&ready);
            let woken = Arc::clone(&woken);
            handles.push(thread::spawn(move || {
                ready.fetch_add(1, Ordering::SeqCst);
                while futex_word.load(Ordering::SeqCst) == 0 {
                    let _ =
                        LinuxFutex::futex_wait(&futex_word, 0, Some(Duration::from_millis(500)));
                }
                woken.fetch_add(1, Ordering::SeqCst);
            }));
        }

        while ready.load(Ordering::SeqCst) < 4 {
            thread::yield_now();
        }
        thread::sleep(Duration::from_millis(30));

        futex_word.store(1, Ordering::SeqCst);
        let wake_count = LinuxFutex::futex_wake(&futex_word, u32::MAX).unwrap();
        assert_eq!(
            wake_count, 4,
            "FUTEX_WAKE_PRIVATE with u32::MAX must clamp to i32::MAX and wake all 4 waiters, got {wake_count}"
        );

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(woken.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_c10_e_futex_wait_duration_max_does_not_einval() {
        let word = AtomicU32::new(1);
        // Value mismatch (1 != 0) returns EAGAIN -> Ok(()), provided the kernel
        // does not reject the timespec with EINVAL before inspecting `word`.
        let res = LinuxFutex::futex_wait(&word, 0, Some(Duration::MAX));
        assert!(
            res.is_ok(),
            "futex_wait with Duration::MAX must clamp tv_sec to i64::MAX instead of failing with EINVAL: {res:?}"
        );
    }
}
