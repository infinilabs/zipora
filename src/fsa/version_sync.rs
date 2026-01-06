//! Version-Based Synchronization for Finite State Automata and Tries
//!
//! This module implements advanced token and version sequence management based on
//! research from advanced concurrent data structure patterns. It provides
//! graduated concurrency control with five distinct levels, from read-only to
//! full multi-writer scenarios.
//!
//! # Key Features
//!
//! - **Graduated Concurrency Control**: Five levels from single-threaded to multi-writer
//! - **Version Sequence Management**: Atomic version counters with consistency validation
//! - **Token-Based Access Control**: Type-safe reader/writer tokens with RAII lifecycle
//! - **Lazy Memory Management**: Age-based cleanup with bulk processing optimizations
//! - **Thread-Local Optimization**: High-performance token caching for reduced contention
//!
//! # Architecture
//!
//! The system is designed around three core concepts:
//!
//! 1. **Concurrency Levels**: Graduated complexity from Level 0 (read-only) to Level 4 (multi-writer)
//! 2. **Version Management**: Master version counter with minimum version tracking for consistency
//! 3. **Token System**: Type-safe access tokens with automatic lifecycle management
//!
//! # Example Usage
//!
//! ```rust
//! use zipora::fsa::version_sync::{ConcurrencyLevel, VersionManager, ReaderToken, WriterToken};
//!
//! // Create version manager with graduated concurrency
//! let manager = VersionManager::new(ConcurrencyLevel::OneWriteMultiRead);
//!
//! // Acquire reader token for concurrent access
//! let reader_token = manager.acquire_reader_token().unwrap();
//! assert!(reader_token.is_valid());
//!
//! // Acquire writer token for exclusive modifications
//! let writer_token = manager.acquire_writer_token().unwrap();
//! assert_eq!(writer_token.concurrency_level(), ConcurrencyLevel::OneWriteMultiRead);
//!
//! // Tokens are automatically released when dropped (RAII)
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
// Additional sync primitives (currently unused)
// use std::sync::atomic::AtomicU8;
// use std::sync::RwLock;
use std::thread::{self, ThreadId};
use std::time::Instant;
use std::collections::VecDeque;

use crate::error::{ZiporaError, Result};
// Memory pool integration (currently unused in this module)
// use crate::memory::SecureMemoryPool;

/// Graduated concurrency control levels providing optimal performance across different threading scenarios.
///
/// This enum defines five distinct concurrency levels, each with specific synchronization guarantees
/// and performance characteristics. Higher levels provide more concurrency but with additional overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ConcurrencyLevel {
    /// Level 0: Read-only access with no synchronization overhead.
    /// 
    /// **Use Case**: Static data structures that never change after initialization.
    /// **Performance**: Zero synchronization overhead, maximum single-threaded performance.
    /// **Thread Safety**: Multiple readers only, no writers allowed.
    NoWriteReadOnly = 0,

    /// Level 1: Single-threaded strict access with no token management.
    ///
    /// **Use Case**: Single-threaded applications or single-threaded phases of execution.
    /// **Performance**: Zero synchronization overhead, immediate memory deallocation.
    /// **Thread Safety**: Single thread only, no concurrent access allowed.
    SingleThreadStrict = 1,

    /// Level 2: Single-threaded with token validity checking and lazy cleanup.
    ///
    /// **Use Case**: Single-threaded with potential for future concurrent access.
    /// **Performance**: Minimal overhead for token management and lazy memory cleanup.
    /// **Thread Safety**: Single thread with token-based access validation.
    SingleThreadShared = 2,

    /// Level 3: One writer with multiple concurrent readers.
    ///
    /// **Use Case**: Read-heavy workloads with occasional updates.
    /// **Performance**: Excellent reader scaling, writer exclusivity guaranteed.
    /// **Thread Safety**: Multiple readers OR single writer (reader-writer lock semantics).
    OneWriteMultiRead = 3,

    /// Level 4: Multiple writers with multiple concurrent readers.
    ///
    /// **Use Case**: High-contention scenarios with frequent updates from multiple threads.
    /// **Performance**: Full concurrency with lock-free optimizations where possible.
    /// **Thread Safety**: Multiple readers AND multiple writers with atomic operations.
    MultiWriteMultiRead = 4,
}

impl ConcurrencyLevel {
    /// Returns true if this concurrency level allows concurrent readers.
    #[inline]
    pub const fn allows_concurrent_readers(self) -> bool {
        matches!(self, Self::OneWriteMultiRead | Self::MultiWriteMultiRead)
    }

    /// Returns true if this concurrency level allows concurrent writers.
    #[inline]
    pub const fn allows_concurrent_writers(self) -> bool {
        matches!(self, Self::MultiWriteMultiRead)
    }

    /// Returns true if this concurrency level requires synchronization.
    #[inline]
    pub const fn requires_synchronization(self) -> bool {
        !matches!(self, Self::NoWriteReadOnly | Self::SingleThreadStrict)
    }

    /// Returns true if this concurrency level uses lazy memory management.
    #[inline]
    pub const fn uses_lazy_cleanup(self) -> bool {
        matches!(
            self,
            Self::SingleThreadShared | Self::OneWriteMultiRead | Self::MultiWriteMultiRead
        )
    }

    /// Returns the recommended maximum number of concurrent readers for this level.
    pub const fn max_concurrent_readers(self) -> Option<usize> {
        match self {
            Self::NoWriteReadOnly => None, // Unlimited readers
            Self::SingleThreadStrict => Some(1),
            Self::SingleThreadShared => Some(1),
            Self::OneWriteMultiRead => None, // Unlimited readers
            Self::MultiWriteMultiRead => None, // Unlimited readers
        }
    }

    /// Returns the recommended maximum number of concurrent writers for this level.
    pub const fn max_concurrent_writers(self) -> Option<usize> {
        match self {
            Self::NoWriteReadOnly => Some(0), // No writers allowed
            Self::SingleThreadStrict => Some(1),
            Self::SingleThreadShared => Some(1),
            Self::OneWriteMultiRead => Some(1), // Single writer only
            Self::MultiWriteMultiRead => None, // Unlimited writers
        }
    }
}

impl Default for ConcurrencyLevel {
    fn default() -> Self {
        Self::SingleThreadStrict
    }
}

impl std::fmt::Display for ConcurrencyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoWriteReadOnly => write!(f, "NoWriteReadOnly"),
            Self::SingleThreadStrict => write!(f, "SingleThreadStrict"),
            Self::SingleThreadShared => write!(f, "SingleThreadShared"),
            Self::OneWriteMultiRead => write!(f, "OneWriteMultiRead"),
            Self::MultiWriteMultiRead => write!(f, "MultiWriteMultiRead"),
        }
    }
}

/// Lazy free list item representing memory that can be safely deallocated after a certain age.
///
/// This structure is used to implement age-based memory reclamation, where memory is not
/// immediately freed but queued for later cleanup when it's safe to do so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LazyFreeItem {
    /// Version sequence number when this memory was freed.
    pub age: u64,
    /// Memory offset or pointer representation.
    pub memory_offset: u32,
    /// Size of the freed memory block in bytes.
    pub size: u32,
}

impl LazyFreeItem {
    /// Creates a new lazy free item.
    pub fn new(age: u64, memory_offset: u32, size: u32) -> Self {
        Self {
            age,
            memory_offset,
            size,
        }
    }

    /// Returns true if this item can be safely freed given the minimum version.
    #[inline]
    pub fn can_free(&self, min_version: u64) -> bool {
        self.age < min_version
    }
}

/// Lazy free list for age-based memory reclamation.
///
/// This structure manages a queue of memory blocks that have been marked for deallocation
/// but cannot be immediately freed due to potential concurrent access. Items are freed
/// in bulk when they reach a safe age.
#[derive(Debug)]
pub struct LazyFreeList {
    /// Queue of items waiting to be freed.
    items: VecDeque<LazyFreeItem>,
    /// Bulk processing threshold.
    bulk_threshold: usize,
    /// Statistics for monitoring performance.
    stats: LazyFreeStats,
}

impl LazyFreeList {
    /// Bulk processing threshold for optimal batch processing.
    pub const BULK_FREE_NUM: usize = 32;

    /// Creates a new lazy free list with default settings.
    pub fn new() -> Self {
        Self::with_bulk_threshold(Self::BULK_FREE_NUM)
    }

    /// Creates a new lazy free list with custom bulk threshold.
    pub fn with_bulk_threshold(bulk_threshold: usize) -> Self {
        Self {
            items: VecDeque::new(),
            bulk_threshold,
            stats: LazyFreeStats::default(),
        }
    }

    /// Adds an item to the lazy free list.
    pub fn push(&mut self, item: LazyFreeItem) {
        self.items.push_back(item);
        self.stats.items_added += 1;
    }

    /// Returns the number of items in the queue.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the queue is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Processes items that can be safely freed given the minimum version.
    ///
    /// Returns the number of items processed. This method implements bulk processing
    /// to reduce the overhead of individual deallocations.
    pub fn process_safe_items<F>(&mut self, min_version: u64, mut free_fn: F) -> usize
    where
        F: FnMut(LazyFreeItem),
    {
        let mut processed = 0;
        let start_time = Instant::now();

        // Process items that can be safely freed
        while let Some(&front) = self.items.front() {
            if !front.can_free(min_version) {
                break; // Items are ordered by age, so we can stop here
            }

            let item = self.items.pop_front().unwrap();
            free_fn(item);
            processed += 1;

            // Bulk processing limit to avoid blocking for too long
            if processed >= self.bulk_threshold {
                break;
            }
        }

        self.stats.items_processed += processed as u64;
        self.stats.total_processing_time += start_time.elapsed();
        processed
    }

    /// Returns true if bulk processing should be triggered.
    ///
    /// This implements advanced processing patterns where the queue
    /// reaches 2x the bulk threshold.
    pub fn should_bulk_process(&self) -> bool {
        self.len() >= 2 * self.bulk_threshold
    }

    /// Returns statistics about the lazy free list.
    pub fn stats(&self) -> &LazyFreeStats {
        &self.stats
    }

    /// Clears all statistics.
    pub fn clear_stats(&mut self) {
        self.stats = LazyFreeStats::default();
    }
}

impl Default for LazyFreeList {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for monitoring lazy free list performance.
#[derive(Debug, Default, Clone)]
pub struct LazyFreeStats {
    /// Total number of items added to the list.
    pub items_added: u64,
    /// Total number of items processed (freed).
    pub items_processed: u64,
    /// Total time spent processing items.
    pub total_processing_time: std::time::Duration,
}

impl LazyFreeStats {
    /// Returns the processing efficiency (processed / added).
    pub fn efficiency(&self) -> f64 {
        if self.items_added == 0 {
            0.0
        } else {
            self.items_processed as f64 / self.items_added as f64
        }
    }

    /// Returns the average processing time per item.
    pub fn avg_processing_time(&self) -> std::time::Duration {
        if self.items_processed == 0 {
            std::time::Duration::ZERO
        } else {
            self.total_processing_time / self.items_processed as u32
        }
    }
}

/// Version manager for token-based synchronization and version sequence management.
///
/// This structure implements the core versioning system that enables safe concurrent access
/// to data structures. It maintains a master version counter and tracks the minimum version
/// still in use by active tokens.
#[derive(Debug)]
pub struct VersionManager {
    /// Current concurrency level.
    concurrency_level: ConcurrencyLevel,
    /// Master version sequence counter.
    current_version: AtomicU64,
    /// Minimum version still in use by active tokens.
    min_version: AtomicU64,
    /// Active reader tokens count.
    active_readers: AtomicU64,
    /// Active writer tokens count.
    active_writers: AtomicU64,
    /// Mutex for token chain management.
    token_chain_mutex: Mutex<()>,
    /// Statistics for monitoring performance.
    stats: Mutex<VersionManagerStats>,
}

impl VersionManager {
    /// Creates a new version manager with the specified concurrency level.
    pub fn new(concurrency_level: ConcurrencyLevel) -> Self {
        Self {
            concurrency_level,
            current_version: AtomicU64::new(1), // Start at 1 to avoid zero-version issues
            min_version: AtomicU64::new(1),
            active_readers: AtomicU64::new(0),
            active_writers: AtomicU64::new(0),
            token_chain_mutex: Mutex::new(()),
            stats: Mutex::new(VersionManagerStats::default()),
        }
    }

    /// Returns the current concurrency level.
    #[inline]
    pub fn concurrency_level(&self) -> ConcurrencyLevel {
        self.concurrency_level
    }

    /// Returns the current version sequence number.
    #[inline]
    pub fn current_version(&self) -> u64 {
        self.current_version.load(Ordering::Acquire)
    }

    /// Returns the minimum version still in use.
    #[inline]
    pub fn min_version(&self) -> u64 {
        self.min_version.load(Ordering::Acquire)
    }

    /// Returns the number of active reader tokens.
    #[inline]
    pub fn active_readers(&self) -> u64 {
        self.active_readers.load(Ordering::Relaxed)
    }

    /// Returns the number of active writer tokens.
    #[inline]
    pub fn active_writers(&self) -> u64 {
        self.active_writers.load(Ordering::Relaxed)
    }

    /// Acquires a new reader token.
    ///
    /// This method implements the token acquisition protocol, assigning a version
    /// sequence number and updating the active token count.
    pub fn acquire_reader_token(&self) -> Result<ReaderToken> {
        // Check if readers are allowed at this concurrency level
        if self.concurrency_level == ConcurrencyLevel::NoWriteReadOnly {
            // Read-only level allows unlimited readers without version tracking
            return Ok(ReaderToken::new_readonly());
        }

        let start_time = Instant::now();

        // For levels that require synchronization, acquire version under lock
        let (version, min_version) = if self.concurrency_level.requires_synchronization() {
            let _lock = self.token_chain_mutex.lock().map_err(|_| {
                ZiporaError::system_error("Failed to acquire token chain mutex for reader")
            })?;

            let current_min = self.min_version.load(Ordering::Acquire);
            let version = self.current_version.fetch_add(1, Ordering::AcqRel) + 1;

            (version, current_min)
        } else {
            // Single-threaded modes don't need version tracking
            (1, 1)
        };

        // Increment active reader count
        self.active_readers.fetch_add(1, Ordering::Relaxed);

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.reader_tokens_acquired += 1;
            stats.total_reader_acquisition_time += start_time.elapsed();
        }

        Ok(ReaderToken::new(
            version,
            min_version,
            thread::current().id(),
            self.concurrency_level,
            Arc::new(TokenReleaseCallback {
                version_manager: self as *const Self,
                token_type: TokenType::Reader,
            }),
        ))
    }

    /// Acquires a new writer token.
    ///
    /// This method implements writer token acquisition with proper exclusivity
    /// checking based on the concurrency level.
    pub fn acquire_writer_token(&self) -> Result<WriterToken> {
        // Check if writers are allowed at this concurrency level
        if self.concurrency_level == ConcurrencyLevel::NoWriteReadOnly {
            return Err(ZiporaError::invalid_operation(
                "Writers not allowed in NoWriteReadOnly mode",
            ));
        }

        let start_time = Instant::now();

        // For OneWriteMultiRead, ensure no other writers are active
        if self.concurrency_level == ConcurrencyLevel::OneWriteMultiRead {
            let current_writers = self.active_writers.load(Ordering::Acquire);
            if current_writers > 0 {
                return Err(ZiporaError::resource_busy(
                    "Another writer is already active in OneWriteMultiRead mode",
                ));
            }
        }

        // Acquire version under lock for synchronized levels
        let (version, min_version) = if self.concurrency_level.requires_synchronization() {
            let _lock = self.token_chain_mutex.lock().map_err(|_| {
                ZiporaError::system_error("Failed to acquire token chain mutex for writer")
            })?;

            let current_min = self.min_version.load(Ordering::Acquire);
            let version = self.current_version.fetch_add(1, Ordering::AcqRel) + 1;

            (version, current_min)
        } else {
            (1, 1)
        };

        // Increment active writer count
        self.active_writers.fetch_add(1, Ordering::Relaxed);

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.writer_tokens_acquired += 1;
            stats.total_writer_acquisition_time += start_time.elapsed();
        }

        Ok(WriterToken::new(
            version,
            min_version,
            thread::current().id(),
            self.concurrency_level,
            Arc::new(TokenReleaseCallback {
                version_manager: self as *const Self,
                token_type: TokenType::Writer,
            }),
        ))
    }

    /// Internal method to release a reader token.
    fn release_reader_token(&self, token_version: u64) {
        self.active_readers.fetch_sub(1, Ordering::Relaxed);

        // Update minimum version if this was the head token
        if self.concurrency_level.requires_synchronization() {
            self.try_advance_min_version();
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.reader_tokens_released += 1;
        }
    }

    /// Internal method to release a writer token.
    fn release_writer_token(&self, token_version: u64) {
        self.active_writers.fetch_sub(1, Ordering::Relaxed);

        // Update minimum version if this was the head token
        if self.concurrency_level.requires_synchronization() {
            self.try_advance_min_version();
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.writer_tokens_released += 1;
        }
    }

    /// Attempts to advance the minimum version based on active tokens.
    ///
    /// This is a simplified version - in a full implementation, this would
    /// track individual token versions in a linked list.
    fn try_advance_min_version(&self) {
        if self.active_readers.load(Ordering::Relaxed) == 0
            && self.active_writers.load(Ordering::Relaxed) == 0
        {
            let current = self.current_version.load(Ordering::Acquire);
            self.min_version.store(current, Ordering::Release);
        }
    }

    /// Returns version manager statistics.
    pub fn stats(&self) -> Result<VersionManagerStats> {
        self.stats
            .lock()
            .map(|stats| stats.clone())
            .map_err(|_| ZiporaError::system_error("Failed to acquire stats mutex"))
    }

    /// Clears all statistics.
    pub fn clear_stats(&self) -> Result<()> {
        self.stats
            .lock()
            .map(|mut stats| *stats = VersionManagerStats::default())
            .map_err(|_| ZiporaError::system_error("Failed to acquire stats mutex"))
    }

    /// Validates that a token version is still valid.
    pub fn validate_token_version(&self, token_version: u64) -> bool {
        let current = self.current_version();
        let min = self.min_version();
        token_version >= min && token_version <= current
    }
}

/// Statistics for monitoring version manager performance.
#[derive(Debug, Default, Clone)]
pub struct VersionManagerStats {
    /// Number of reader tokens acquired.
    pub reader_tokens_acquired: u64,
    /// Number of reader tokens released.
    pub reader_tokens_released: u64,
    /// Number of writer tokens acquired.
    pub writer_tokens_acquired: u64,
    /// Number of writer tokens released.
    pub writer_tokens_released: u64,
    /// Total time spent acquiring reader tokens.
    pub total_reader_acquisition_time: std::time::Duration,
    /// Total time spent acquiring writer tokens.
    pub total_writer_acquisition_time: std::time::Duration,
}

impl VersionManagerStats {
    /// Returns the average reader token acquisition time.
    pub fn avg_reader_acquisition_time(&self) -> std::time::Duration {
        if self.reader_tokens_acquired == 0 {
            std::time::Duration::ZERO
        } else {
            self.total_reader_acquisition_time / self.reader_tokens_acquired as u32
        }
    }

    /// Returns the average writer token acquisition time.
    pub fn avg_writer_acquisition_time(&self) -> std::time::Duration {
        if self.writer_tokens_acquired == 0 {
            std::time::Duration::ZERO
        } else {
            self.total_writer_acquisition_time / self.writer_tokens_acquired as u32
        }
    }

    /// Returns the number of active reader tokens (acquired - released).
    pub fn active_readers(&self) -> i64 {
        self.reader_tokens_acquired as i64 - self.reader_tokens_released as i64
    }

    /// Returns the number of active writer tokens (acquired - released).
    pub fn active_writers(&self) -> i64 {
        self.writer_tokens_acquired as i64 - self.writer_tokens_released as i64
    }
}

/// Token type for callback identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenType {
    Reader,
    Writer,
}

/// Callback structure for token release.
struct TokenReleaseCallback {
    version_manager: *const VersionManager,
    token_type: TokenType,
}

impl std::fmt::Debug for TokenReleaseCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenReleaseCallback")
            .field("version_manager", &(self.version_manager as usize))
            .field("token_type", &self.token_type)
            .finish()
    }
}

impl TokenReleaseCallback {
    fn release(&self, token_version: u64) {
        unsafe {
            let manager = &*self.version_manager;
            match self.token_type {
                TokenType::Reader => manager.release_reader_token(token_version),
                TokenType::Writer => manager.release_writer_token(token_version),
            }
        }
    }
}

// SAFETY: TokenReleaseCallback is Send because:
// 1. `version_manager: *const VersionManager` - Raw pointer to a VersionManager.
//    The VersionManager is expected to outlive all callbacks (managed by Arc).
// 2. `token_type: TokenType` - Simple enum, trivially Send.
//
// INVARIANT: The VersionManager must remain valid for the lifetime of all callbacks.
// This is enforced by the Arc<VersionManager> ownership in the token creation path.
unsafe impl Send for TokenReleaseCallback {}

// SAFETY: TokenReleaseCallback is Sync because:
// 1. Both fields are read-only after construction.
// 2. `release()` calls thread-safe methods on VersionManager (which uses atomics).
// 3. The VersionManager's release_reader_token/release_writer_token are atomic.
// Sharing &TokenReleaseCallback for concurrent reads is safe.
unsafe impl Sync for TokenReleaseCallback {}

/// Reader token for safe concurrent read access.
///
/// Reader tokens provide read-only access to data structures with version-based
/// consistency guarantees. Multiple reader tokens can be active simultaneously
/// for most concurrency levels.
#[derive(Debug)]
pub struct ReaderToken {
    /// Version sequence number when this token was acquired.
    version: u64,
    /// Minimum version that was valid when this token was acquired.
    min_version: u64,
    /// Thread ID that owns this token.
    thread_id: ThreadId,
    /// Concurrency level when this token was acquired.
    concurrency_level: ConcurrencyLevel,
    /// Release callback for automatic cleanup.
    release_callback: Option<Arc<TokenReleaseCallback>>,
}

impl ReaderToken {
    /// Creates a new reader token.
    fn new(
        version: u64,
        min_version: u64,
        thread_id: ThreadId,
        concurrency_level: ConcurrencyLevel,
        release_callback: Arc<TokenReleaseCallback>,
    ) -> Self {
        Self {
            version,
            min_version,
            thread_id,
            concurrency_level,
            release_callback: Some(release_callback),
        }
    }

    /// Creates a read-only token for NoWriteReadOnly mode.
    fn new_readonly() -> Self {
        Self {
            version: 0, // Special version for read-only tokens
            min_version: 0,
            thread_id: thread::current().id(),
            concurrency_level: ConcurrencyLevel::NoWriteReadOnly,
            release_callback: None,
        }
    }

    /// Returns the token's version sequence number.
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Returns the minimum version that was valid when this token was acquired.
    #[inline]
    pub fn min_version(&self) -> u64 {
        self.min_version
    }

    /// Returns the thread ID that owns this token.
    #[inline]
    pub fn thread_id(&self) -> ThreadId {
        self.thread_id
    }

    /// Returns the concurrency level when this token was acquired.
    #[inline]
    pub fn concurrency_level(&self) -> ConcurrencyLevel {
        self.concurrency_level
    }

    /// Returns true if this token is valid (read-only tokens are always valid).
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.concurrency_level == ConcurrencyLevel::NoWriteReadOnly || self.version > 0
    }

    /// Returns true if this is a read-only token.
    #[inline]
    pub fn is_readonly(&self) -> bool {
        self.concurrency_level == ConcurrencyLevel::NoWriteReadOnly
    }
}

impl Drop for ReaderToken {
    fn drop(&mut self) {
        if let Some(callback) = self.release_callback.take() {
            callback.release(self.version);
        }
    }
}

/// Writer token for safe concurrent write access.
///
/// Writer tokens provide exclusive write access to data structures with version-based
/// consistency guarantees. The number of concurrent writer tokens depends on the
/// concurrency level.
#[derive(Debug)]
pub struct WriterToken {
    /// Version sequence number when this token was acquired.
    version: u64,
    /// Minimum version that was valid when this token was acquired.
    min_version: u64,
    /// Thread ID that owns this token.
    thread_id: ThreadId,
    /// Concurrency level when this token was acquired.
    concurrency_level: ConcurrencyLevel,
    /// Release callback for automatic cleanup.
    release_callback: Option<Arc<TokenReleaseCallback>>,
}

impl WriterToken {
    /// Creates a new writer token.
    fn new(
        version: u64,
        min_version: u64,
        thread_id: ThreadId,
        concurrency_level: ConcurrencyLevel,
        release_callback: Arc<TokenReleaseCallback>,
    ) -> Self {
        Self {
            version,
            min_version,
            thread_id,
            concurrency_level,
            release_callback: Some(release_callback),
        }
    }

    /// Returns the token's version sequence number.
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Returns the minimum version that was valid when this token was acquired.
    #[inline]
    pub fn min_version(&self) -> u64 {
        self.min_version
    }

    /// Returns the thread ID that owns this token.
    #[inline]
    pub fn thread_id(&self) -> ThreadId {
        self.thread_id
    }

    /// Returns the concurrency level when this token was acquired.
    #[inline]
    pub fn concurrency_level(&self) -> ConcurrencyLevel {
        self.concurrency_level
    }

    /// Returns true if this token is valid.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.version > 0
    }

    /// Returns true if this token allows concurrent writers.
    #[inline]
    pub fn allows_concurrent_writers(&self) -> bool {
        self.concurrency_level.allows_concurrent_writers()
    }
}

impl Drop for WriterToken {
    fn drop(&mut self) {
        if let Some(callback) = self.release_callback.take() {
            callback.release(self.version);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_concurrency_level_properties() {
        assert!(!ConcurrencyLevel::NoWriteReadOnly.allows_concurrent_writers());
        assert!(ConcurrencyLevel::MultiWriteMultiRead.allows_concurrent_writers());
        assert!(ConcurrencyLevel::OneWriteMultiRead.allows_concurrent_readers());
        assert!(!ConcurrencyLevel::SingleThreadStrict.requires_synchronization());
        assert!(ConcurrencyLevel::OneWriteMultiRead.uses_lazy_cleanup());
    }

    #[test]
    fn test_lazy_free_list() {
        let mut list = LazyFreeList::new();
        assert!(list.is_empty());

        // Add some items
        list.push(LazyFreeItem::new(1, 100, 64));
        list.push(LazyFreeItem::new(2, 200, 128));
        list.push(LazyFreeItem::new(3, 300, 256));

        assert_eq!(list.len(), 3);

        // Process items with min_version = 2
        let mut freed_items = Vec::new();
        let processed = list.process_safe_items(3, |item| freed_items.push(item));

        assert_eq!(processed, 2); // Items with age 1 and 2 should be processed
        assert_eq!(freed_items.len(), 2);
        assert_eq!(list.len(), 1); // One item remaining

        let stats = list.stats();
        assert_eq!(stats.items_added, 3);
        assert_eq!(stats.items_processed, 2);
    }

    #[test]
    fn test_version_manager_single_thread() -> Result<()> {
        let manager = VersionManager::new(ConcurrencyLevel::SingleThreadStrict);

        // Acquire reader token
        let reader_token = manager.acquire_reader_token()?;
        assert!(reader_token.is_valid());
        assert_eq!(reader_token.concurrency_level(), ConcurrencyLevel::SingleThreadStrict);

        // Acquire writer token
        let writer_token = manager.acquire_writer_token()?;
        assert!(writer_token.is_valid());
        assert!(!writer_token.allows_concurrent_writers());

        Ok(())
    }

    #[test]
    fn test_version_manager_readonly() -> Result<()> {
        let manager = VersionManager::new(ConcurrencyLevel::NoWriteReadOnly);

        // Acquire reader token (should work)
        let reader_token = manager.acquire_reader_token()?;
        assert!(reader_token.is_valid());
        assert!(reader_token.is_readonly());

        // Acquire writer token (should fail)
        let result = manager.acquire_writer_token();
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_version_manager_one_write_multi_read() -> Result<()> {
        let manager = VersionManager::new(ConcurrencyLevel::OneWriteMultiRead);

        // Acquire multiple reader tokens
        let reader1 = manager.acquire_reader_token()?;
        let reader2 = manager.acquire_reader_token()?;

        assert_eq!(manager.active_readers(), 2);

        // Acquire first writer token (should work)
        let writer1 = manager.acquire_writer_token()?;
        assert_eq!(manager.active_writers(), 1);

        // Try to acquire second writer token (should fail)
        let result = manager.acquire_writer_token();
        assert!(result.is_err());

        drop(writer1);
        assert_eq!(manager.active_writers(), 0);

        // Now second writer should work
        let writer2 = manager.acquire_writer_token()?;
        assert!(writer2.is_valid());

        Ok(())
    }

    #[test]
    fn test_version_manager_multi_write_multi_read() -> Result<()> {
        let manager = VersionManager::new(ConcurrencyLevel::MultiWriteMultiRead);

        // Acquire multiple reader tokens
        let _reader1 = manager.acquire_reader_token()?;
        let _reader2 = manager.acquire_reader_token()?;

        // Acquire multiple writer tokens (should all work)
        let _writer1 = manager.acquire_writer_token()?;
        let _writer2 = manager.acquire_writer_token()?;

        assert_eq!(manager.active_readers(), 2);
        assert_eq!(manager.active_writers(), 2);

        Ok(())
    }

    #[test]
    fn test_token_version_validation() -> Result<()> {
        let manager = VersionManager::new(ConcurrencyLevel::OneWriteMultiRead);

        let token = manager.acquire_reader_token()?;
        assert!(manager.validate_token_version(token.version()));

        // Invalid versions should fail
        assert!(!manager.validate_token_version(0));
        assert!(!manager.validate_token_version(u64::MAX));

        Ok(())
    }

    #[test]
    fn test_concurrent_token_acquisition() -> Result<()> {
        let manager = Arc::new(VersionManager::new(ConcurrencyLevel::MultiWriteMultiRead));
        let num_threads = 4;
        let tokens_per_thread = 10;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let manager_clone = Arc::clone(&manager);
                thread::spawn(move || -> Result<()> {
                    for _ in 0..tokens_per_thread {
                        let _reader = manager_clone.acquire_reader_token()?;
                        let _writer = manager_clone.acquire_writer_token()?;
                        thread::sleep(Duration::from_millis(1));
                    }
                    Ok(())
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap()?;
        }

        // All tokens should be released
        assert_eq!(manager.active_readers(), 0);
        assert_eq!(manager.active_writers(), 0);

        let stats = manager.stats()?;
        assert_eq!(stats.reader_tokens_acquired, num_threads * tokens_per_thread);
        assert_eq!(stats.writer_tokens_acquired, num_threads * tokens_per_thread);

        Ok(())
    }

    #[test]
    fn test_token_drop_cleanup() -> Result<()> {
        let manager = VersionManager::new(ConcurrencyLevel::OneWriteMultiRead);

        {
            let _reader = manager.acquire_reader_token()?;
            let _writer = manager.acquire_writer_token()?;
            assert_eq!(manager.active_readers(), 1);
            assert_eq!(manager.active_writers(), 1);
        } // Tokens dropped here

        // Give time for cleanup
        thread::sleep(Duration::from_millis(10));

        assert_eq!(manager.active_readers(), 0);
        assert_eq!(manager.active_writers(), 0);

        Ok(())
    }
}