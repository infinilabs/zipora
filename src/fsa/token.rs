//! Token System for Thread-Safe FSA Access
//!
//! This module implements the token-based access control system that enables
//! safe concurrent access to Finite State Automata and Tries. It provides
//! thread-local token caching, automatic lifecycle management, and integration
//! with the version-based synchronization system.
//!
//! # Features
//!
//! - **Thread-Local Token Caching**: High-performance token reuse with zero allocation
//! - **Automatic Lifecycle Management**: RAII-based token cleanup with leak detection
//! - **Type-Safe Access Control**: Compile-time guarantees for reader/writer access patterns
//! - **Performance Monitoring**: Built-in statistics for cache hit rates and performance
//! - **Memory Safety**: Zero unsafe operations in public APIs
//!
//! # Example Usage
//!
//! ```rust
//! use zipora::fsa::token::{TokenManager, with_reader_token, with_writer_token};
//! use zipora::fsa::version_sync::ConcurrencyLevel;
//!
//! // Initialize token manager for concurrent access
//! let token_manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
//!
//! // Use reader token with automatic caching
//! with_reader_token(&token_manager, |token| {
//!     // Perform read operations with the token
//!     assert!(token.is_valid());
//!     Ok(())
//! }).unwrap();
//!
//! // Use writer token with exclusive access
//! with_writer_token(&token_manager, |token| {
//!     // Perform write operations with the token
//!     assert!(token.is_valid());
//!     Ok(())
//! }).unwrap();
//! ```

use std::cell::RefCell;
use std::sync::Arc;
// Thread utilities for token management (self and ThreadId currently unused)
// use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use crate::error::{Result, ZiporaError};
use crate::fsa::version_sync::{ConcurrencyLevel, ReaderToken, VersionManager, WriterToken};

/// Thread-local token cache for high-performance token reuse.
///
/// This structure maintains thread-local caches of tokens to avoid the overhead
/// of repeated token acquisition and release. While a synchronized
/// [`ReaderToken`] remains cached, it stays live (`active_readers > 0` on its
/// [`VersionManager`]), so [`VersionManager::min_version`] cannot advance past
/// its snapshot version while it stays cached; call [`TokenCache::clear`] or
/// [`TokenManager::clear_thread_cache`] (or replace/exit the thread) to release
/// it and allow version reclamation.
#[derive(Debug)]
pub struct TokenCache {
    /// Cached reader token for reuse.
    cached_reader: Option<ReaderToken>,
    /// Cached writer token for reuse.
    cached_writer: Option<WriterToken>,
    /// Cache statistics for monitoring performance.
    stats: TokenCacheStats,
}

impl TokenCache {
    /// Creates a new empty token cache.
    pub fn new() -> Self {
        Self {
            cached_reader: None,
            cached_writer: None,
            stats: TokenCacheStats::default(),
        }
    }

    /// Attempts to get a cached reader token.
    ///
    /// Returns the cached token if available and valid, otherwise None.
    pub fn get_reader_token(&mut self) -> Option<ReaderToken> {
        if let Some(token) = self.cached_reader.take() {
            if token.is_valid() {
                self.stats.reader_cache_hits += 1;
                return Some(token);
            } else {
                self.stats.reader_cache_invalidations += 1;
            }
        }
        self.stats.reader_cache_misses += 1;
        None
    }

    /// Attempts to get a cached reader token belonging to `manager_id`.
    ///
    /// If a reader token from another manager is cached, leaves it in place,
    /// records a reader cache miss, and returns `None`.
    pub(crate) fn get_reader_token_for(&mut self, manager_id: u64) -> Option<ReaderToken> {
        if matches!(
            self.cached_reader.as_ref(),
            Some(token) if token.manager_id() != manager_id
        ) {
            self.stats.reader_cache_misses += 1;
            return None;
        }
        self.get_reader_token()
    }

    /// Attempts to get a cached writer token.
    ///
    /// Returns the cached token if available and valid, otherwise None.
    pub fn get_writer_token(&mut self) -> Option<WriterToken> {
        if let Some(token) = self.cached_writer.take() {
            if token.is_valid() {
                self.stats.writer_cache_hits += 1;
                return Some(token);
            } else {
                self.stats.writer_cache_invalidations += 1;
            }
        }
        self.stats.writer_cache_misses += 1;
        None
    }

    /// Caches a reader token for future reuse.
    pub fn cache_reader_token(&mut self, token: ReaderToken) {
        self.cached_reader = Some(token);
        self.stats.reader_tokens_cached += 1;
    }

    /// Caches a writer token for future reuse.
    pub fn cache_writer_token(&mut self, token: WriterToken) {
        self.cached_writer = Some(token);
        self.stats.writer_tokens_cached += 1;
    }

    /// Clears all cached tokens.
    pub fn clear(&mut self) {
        self.cached_reader = None;
        self.cached_writer = None;
        self.stats.cache_clears += 1;
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> &TokenCacheStats {
        &self.stats
    }

    /// Clears cache statistics.
    pub fn clear_stats(&mut self) {
        self.stats = TokenCacheStats::default();
    }
}

impl Default for TokenCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for monitoring token cache performance.
#[derive(Debug, Default, Clone)]
pub struct TokenCacheStats {
    /// Number of successful reader token cache hits.
    pub reader_cache_hits: u64,
    /// Number of reader token cache misses.
    pub reader_cache_misses: u64,
    /// Number of reader token cache invalidations (stale tokens).
    pub reader_cache_invalidations: u64,
    /// Number of reader tokens cached.
    pub reader_tokens_cached: u64,

    /// Number of successful writer token cache hits.
    pub writer_cache_hits: u64,
    /// Number of writer token cache misses.
    pub writer_cache_misses: u64,
    /// Number of writer token cache invalidations (stale tokens).
    pub writer_cache_invalidations: u64,
    /// Number of writer tokens cached.
    pub writer_tokens_cached: u64,

    /// Number of cache clear operations.
    pub cache_clears: u64,
}

impl TokenCacheStats {
    /// Returns the reader token cache hit rate (0.0 to 1.0).
    pub fn reader_hit_rate(&self) -> f64 {
        let total = self.reader_cache_hits + self.reader_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.reader_cache_hits as f64 / total as f64
        }
    }

    /// Returns the writer token cache hit rate (0.0 to 1.0).
    pub fn writer_hit_rate(&self) -> f64 {
        let total = self.writer_cache_hits + self.writer_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.writer_cache_hits as f64 / total as f64
        }
    }

    /// Returns the overall cache hit rate (0.0 to 1.0).
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.reader_cache_hits + self.writer_cache_hits;
        let total_requests = total_hits + self.reader_cache_misses + self.writer_cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            total_hits as f64 / total_requests as f64
        }
    }

    /// Returns the reader token invalidation rate (0.0 to 1.0).
    pub fn reader_invalidation_rate(&self) -> f64 {
        let total = self.reader_cache_hits + self.reader_cache_invalidations;
        if total == 0 {
            0.0
        } else {
            self.reader_cache_invalidations as f64 / total as f64
        }
    }

    /// Returns the writer token invalidation rate (0.0 to 1.0).
    pub fn writer_invalidation_rate(&self) -> f64 {
        let total = self.writer_cache_hits + self.writer_cache_invalidations;
        if total == 0 {
            0.0
        } else {
            self.writer_cache_invalidations as f64 / total as f64
        }
    }
}

// Thread-local storage for token caches
thread_local! {
    /// Thread-local token cache for high-performance token reuse.
    static TOKEN_CACHE: RefCell<TokenCache> = RefCell::new(TokenCache::new());
}

/// Token manager for coordinating token acquisition and caching across threads.
///
/// This structure provides a high-level interface for managing tokens with
/// automatic caching, performance monitoring, and lifecycle management.
#[derive(Debug)]
pub struct TokenManager {
    /// Version manager for token coordination.
    version_manager: Arc<VersionManager>,
    /// Global statistics for all threads.
    global_stats: Arc<std::sync::Mutex<GlobalTokenStats>>,
}

impl TokenManager {
    /// Creates a new token manager with the specified concurrency level.
    pub fn new(concurrency_level: ConcurrencyLevel) -> Self {
        Self {
            version_manager: Arc::new(VersionManager::new(concurrency_level)),
            global_stats: Arc::new(std::sync::Mutex::new(GlobalTokenStats::default())),
        }
    }

    /// Creates a new token manager with a shared version manager.
    pub fn with_version_manager(version_manager: Arc<VersionManager>) -> Self {
        Self {
            version_manager,
            global_stats: Arc::new(std::sync::Mutex::new(GlobalTokenStats::default())),
        }
    }

    /// Returns the underlying version manager.
    pub fn version_manager(&self) -> &Arc<VersionManager> {
        &self.version_manager
    }

    /// Acquires a reader token with caching optimization.
    ///
    /// This method first checks the thread-local cache for a valid token issued
    /// by this manager. If no matching cached token is available, it acquires a
    /// new one from the version manager.
    pub fn acquire_reader_token(&self) -> Result<ReaderToken> {
        let start_time = Instant::now();
        let manager_id = self.version_manager.id();

        // Try to get a cached token belonging to this manager first
        let token = TOKEN_CACHE.with(|cache| cache.borrow_mut().get_reader_token_for(manager_id));

        let token = if let Some(cached_token) = token {
            // Update global cache hit statistics
            if let Ok(mut stats) = self.global_stats.lock() {
                stats.total_reader_cache_hits += 1;
            }
            cached_token
        } else {
            // Acquire new token from version manager
            let new_token = self.version_manager.acquire_reader_token()?;

            // Update global cache miss statistics
            if let Ok(mut stats) = self.global_stats.lock() {
                stats.total_reader_cache_misses += 1;
                stats.total_reader_acquisition_time += start_time.elapsed();
            }

            new_token
        };

        Ok(token)
    }

    /// Acquires a writer token directly from the version manager.
    ///
    /// Writer tokens are never cached in thread-local storage so exclusive
    /// writer slots are released promptly and cannot cross manager boundaries.
    pub fn acquire_writer_token(&self) -> Result<WriterToken> {
        let start_time = Instant::now();
        let new_token = self.version_manager.acquire_writer_token()?;

        if let Ok(mut stats) = self.global_stats.lock() {
            stats.total_writer_cache_misses += 1;
            stats.total_writer_acquisition_time += start_time.elapsed();
        }

        Ok(new_token)
    }

    /// Returns a reader token to the cache for reuse.
    ///
    /// This method caches the token in thread-local storage for future reuse,
    /// avoiding the overhead of repeated token acquisition.
    ///
    /// # Version Retention
    ///
    /// A cached synchronized [`ReaderToken`] remains live and keeps
    /// [`VersionManager::active_readers`] `> 0`, so [`VersionManager::min_version`]
    /// cannot advance past its snapshot version while it stays cached. Call
    /// [`Self::clear_thread_cache`] (or let a subsequent reader replacement or
    /// thread exit drop the cached token) when idle threads must release pinned
    /// versions.
    pub fn return_reader_token(&self, token: ReaderToken) {
        if token.manager_id() == self.version_manager.id() {
            TOKEN_CACHE.with(|cache| {
                cache.borrow_mut().cache_reader_token(token);
            });

            // Update global statistics
            if let Ok(mut stats) = self.global_stats.lock() {
                stats.total_reader_tokens_returned += 1;
            }
        } else {
            drop(token);
        }
    }

    /// Releases a writer token immediately without caching it.
    ///
    /// Dropping the writer token frees exclusive writer slots (such as
    /// `OneWriteMultiRead`) so other threads can acquire a writer token.
    pub fn return_writer_token(&self, token: WriterToken) {
        drop(token);

        // Update global statistics
        if let Ok(mut stats) = self.global_stats.lock() {
            stats.total_writer_tokens_returned += 1;
        }
    }

    /// Clears the thread-local token cache.
    pub fn clear_thread_cache(&self) {
        TOKEN_CACHE.with(|cache| {
            cache.borrow_mut().clear();
        });
    }

    /// Returns the current concurrency level.
    pub fn concurrency_level(&self) -> ConcurrencyLevel {
        self.version_manager.concurrency_level()
    }

    /// Returns global token manager statistics.
    pub fn global_stats(&self) -> Result<GlobalTokenStats> {
        self.global_stats
            .lock()
            .map(|stats| stats.clone())
            .map_err(|_| ZiporaError::system_error("Failed to acquire global stats mutex"))
    }

    /// Returns thread-local cache statistics.
    pub fn thread_cache_stats(&self) -> TokenCacheStats {
        TOKEN_CACHE.with(|cache| cache.borrow().stats().clone())
    }

    /// Clears all statistics (global and thread-local).
    pub fn clear_all_stats(&self) -> Result<()> {
        // Clear global statistics
        self.global_stats
            .lock()
            .map(|mut stats| *stats = GlobalTokenStats::default())
            .map_err(|_| ZiporaError::system_error("Failed to acquire global stats mutex"))?;

        // Clear thread-local statistics
        TOKEN_CACHE.with(|cache| {
            cache.borrow_mut().clear_stats();
        });

        // Clear version manager statistics
        self.version_manager.clear_stats()
    }
}

/// Global statistics for token manager performance across all threads.
#[derive(Debug, Default, Clone)]
pub struct GlobalTokenStats {
    /// Total reader token cache hits across all threads.
    pub total_reader_cache_hits: u64,
    /// Total reader token cache misses across all threads.
    pub total_reader_cache_misses: u64,
    /// Total reader tokens returned to cache across all threads.
    pub total_reader_tokens_returned: u64,
    /// Total time spent acquiring reader tokens across all threads.
    pub total_reader_acquisition_time: Duration,

    /// Total writer token cache hits across all threads.
    pub total_writer_cache_hits: u64,
    /// Total writer token cache misses across all threads.
    pub total_writer_cache_misses: u64,
    /// Total writer tokens returned to cache across all threads.
    pub total_writer_tokens_returned: u64,
    /// Total time spent acquiring writer tokens across all threads.
    pub total_writer_acquisition_time: Duration,
}

impl GlobalTokenStats {
    /// Returns the global reader token cache hit rate (0.0 to 1.0).
    pub fn reader_hit_rate(&self) -> f64 {
        let total = self.total_reader_cache_hits + self.total_reader_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.total_reader_cache_hits as f64 / total as f64
        }
    }

    /// Returns the global writer token cache hit rate (0.0 to 1.0).
    pub fn writer_hit_rate(&self) -> f64 {
        let total = self.total_writer_cache_hits + self.total_writer_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.total_writer_cache_hits as f64 / total as f64
        }
    }

    /// Returns the overall global cache hit rate (0.0 to 1.0).
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.total_reader_cache_hits + self.total_writer_cache_hits;
        let total_requests =
            total_hits + self.total_reader_cache_misses + self.total_writer_cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            total_hits as f64 / total_requests as f64
        }
    }

    /// Returns the average reader token acquisition time.
    pub fn avg_reader_acquisition_time(&self) -> Duration {
        if self.total_reader_cache_misses == 0 {
            Duration::ZERO
        } else {
            self.total_reader_acquisition_time / self.total_reader_cache_misses as u32
        }
    }

    /// Returns the average writer token acquisition time.
    pub fn avg_writer_acquisition_time(&self) -> Duration {
        if self.total_writer_cache_misses == 0 {
            Duration::ZERO
        } else {
            self.total_writer_acquisition_time / self.total_writer_cache_misses as u32
        }
    }
}

/// Convenience function for executing a closure with a reader token.
///
/// This function automatically acquires a reader token, executes the closure,
/// and returns the token to the thread-local cache via
/// [`TokenManager::return_reader_token`]. Because a cached synchronized reader
/// keeps `active_readers > 0`, [`VersionManager::min_version`] cannot advance
/// past it while it stays cached; call [`TokenManager::clear_thread_cache`] to
/// release it on idle threads.
///
/// # Example
///
/// ```rust
/// use zipora::fsa::token::{with_reader_token, TokenManager};
/// use zipora::fsa::version_sync::ConcurrencyLevel;
///
/// let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
///
/// let result = with_reader_token(&manager, |token| {
///     // Use the token for read operations
///     assert!(token.is_valid());
///     Ok(42)
/// }).unwrap();
///
/// assert_eq!(result, 42);
/// ```
pub fn with_reader_token<F, R>(token_manager: &TokenManager, f: F) -> Result<R>
where
    F: FnOnce(&ReaderToken) -> Result<R>,
{
    let token = token_manager.acquire_reader_token()?;
    let result = f(&token)?;
    token_manager.return_reader_token(token);
    Ok(result)
}

/// Convenience function for executing a closure with a writer token.
///
/// This function acquires a writer token, executes the closure, and releases
/// the token immediately when finished.
///
/// # Example
///
/// ```rust
/// use zipora::fsa::token::{with_writer_token, TokenManager};
/// use zipora::fsa::version_sync::ConcurrencyLevel;
///
/// let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
///
/// let result = with_writer_token(&manager, |token| {
///     // Use the token for write operations
///     assert!(token.is_valid());
///     Ok(42)
/// }).unwrap();
///
/// assert_eq!(result, 42);
/// ```
pub fn with_writer_token<F, R>(token_manager: &TokenManager, f: F) -> Result<R>
where
    F: FnOnce(&WriterToken) -> Result<R>,
{
    let token = token_manager.acquire_writer_token()?;
    let result = f(&token)?;
    token_manager.return_writer_token(token);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_token_cache_basic() {
        let mut cache = TokenCache::new();

        // Cache should be empty initially
        assert!(cache.get_reader_token().is_none());
        assert!(cache.get_writer_token().is_none());

        // Create a token manager for testing
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);

        // Acquire tokens
        let reader_token = manager.acquire_reader_token().unwrap();
        let writer_token = manager.acquire_writer_token().unwrap();

        // Cache tokens
        cache.cache_reader_token(reader_token);
        cache.cache_writer_token(writer_token);

        // Should be able to retrieve cached tokens
        let cached_reader = cache.get_reader_token();
        let cached_writer = cache.get_writer_token();

        assert!(cached_reader.is_some());
        assert!(cached_writer.is_some());

        // Cache should be empty after retrieval
        assert!(cache.get_reader_token().is_none());
        assert!(cache.get_writer_token().is_none());

        let stats = cache.stats();
        assert_eq!(stats.reader_cache_hits, 1);
        assert_eq!(stats.writer_cache_hits, 1);
    }

    #[test]
    fn test_token_manager_basic() -> Result<()> {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);

        // Test reader token acquisition and return
        let reader_token = manager.acquire_reader_token()?;
        assert!(reader_token.is_valid());
        manager.return_reader_token(reader_token);

        // Test writer token acquisition and return
        let writer_token = manager.acquire_writer_token()?;
        assert!(writer_token.is_valid());
        manager.return_writer_token(writer_token);

        let global_stats = manager.global_stats()?;
        assert_eq!(global_stats.total_reader_tokens_returned, 1);
        assert_eq!(global_stats.total_writer_tokens_returned, 1);

        Ok(())
    }

    #[test]
    fn test_token_caching_performance() -> Result<()> {
        let manager = TokenManager::new(ConcurrencyLevel::MultiWriteMultiRead);

        // First acquisition should be a cache miss
        let reader1 = manager.acquire_reader_token()?;
        manager.return_reader_token(reader1);

        // Second acquisition should be a cache hit
        let reader2 = manager.acquire_reader_token()?;
        manager.return_reader_token(reader2);

        let global_stats = manager.global_stats()?;
        assert_eq!(global_stats.total_reader_cache_misses, 1);
        assert_eq!(global_stats.total_reader_cache_hits, 1);
        assert_eq!(global_stats.reader_hit_rate(), 0.5);

        Ok(())
    }

    #[test]
    fn test_with_reader_token_convenience() -> Result<()> {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);

        let result = with_reader_token(&manager, |token| {
            assert!(token.is_valid());
            Ok(42)
        })?;

        assert_eq!(result, 42);

        let global_stats = manager.global_stats()?;
        assert_eq!(global_stats.total_reader_tokens_returned, 1);

        Ok(())
    }

    #[test]
    fn test_with_writer_token_convenience() -> Result<()> {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);

        let result = with_writer_token(&manager, |token| {
            assert!(token.is_valid());
            Ok(84)
        })?;

        assert_eq!(result, 84);

        let global_stats = manager.global_stats()?;
        assert_eq!(global_stats.total_writer_tokens_returned, 1);

        Ok(())
    }

    #[test]
    fn test_concurrent_token_caching() -> Result<()> {
        let manager = Arc::new(TokenManager::new(ConcurrencyLevel::MultiWriteMultiRead));
        let num_threads = 4;
        let operations_per_thread = 20;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let manager_clone = Arc::clone(&manager);
                thread::spawn(move || -> Result<()> {
                    for _ in 0..operations_per_thread {
                        // Acquire and return reader token
                        let reader = manager_clone.acquire_reader_token()?;
                        manager_clone.return_reader_token(reader);

                        // Acquire and return writer token
                        let writer = manager_clone.acquire_writer_token()?;
                        manager_clone.return_writer_token(writer);

                        thread::sleep(Duration::from_millis(1));
                    }
                    Ok(())
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap()?;
        }

        let global_stats = manager.global_stats()?;
        assert_eq!(
            global_stats.total_reader_tokens_returned,
            num_threads * operations_per_thread
        );
        assert_eq!(
            global_stats.total_writer_tokens_returned,
            num_threads * operations_per_thread
        );

        // Reader cache hit rate should be high after the first operation per thread;
        // writer tokens are never cached.
        assert!(global_stats.reader_hit_rate() > 0.5);

        Ok(())
    }

    #[test]
    fn test_thread_cache_isolation() -> Result<()> {
        let manager = Arc::new(TokenManager::new(ConcurrencyLevel::MultiWriteMultiRead));

        // Test that thread-local caches are isolated
        let handles: Vec<_> = (0..2)
            .map(|_thread_id| {
                let manager_clone = Arc::clone(&manager);
                thread::spawn(move || -> Result<()> {
                    // Each thread should have its own cache
                    let reader = manager_clone.acquire_reader_token()?;
                    manager_clone.return_reader_token(reader);

                    // Get thread-local stats
                    let thread_stats = manager_clone.thread_cache_stats();

                    // Each thread should have exactly one cache miss and one token cached
                    assert_eq!(thread_stats.reader_cache_misses, 1);
                    assert_eq!(thread_stats.reader_tokens_cached, 1);

                    Ok(())
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap()?;
        }

        let global_stats = manager.global_stats()?;
        assert_eq!(global_stats.total_reader_cache_misses, 2); // One per thread
        assert_eq!(global_stats.total_reader_tokens_returned, 2); // One per thread

        Ok(())
    }

    #[test]
    fn test_statistics_accuracy() -> Result<()> {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        manager.clear_thread_cache();
        manager.clear_all_stats()?;

        // Perform multiple operations
        for _ in 0..5 {
            with_reader_token(&manager, |_| Ok(()))?;
            with_writer_token(&manager, |_| Ok(()))?;
        }

        let global_stats = manager.global_stats()?;
        let thread_stats = manager.thread_cache_stats();

        // Verify global statistics (readers cached; writers never cached)
        assert_eq!(global_stats.total_reader_cache_misses, 1); // First acquisition
        assert_eq!(global_stats.total_reader_cache_hits, 4); // Subsequent acquisitions
        assert_eq!(global_stats.total_writer_cache_misses, 5); // Always acquired fresh
        assert_eq!(global_stats.total_writer_cache_hits, 0);

        // Verify thread-local statistics
        assert_eq!(thread_stats.reader_cache_misses, 1);
        assert_eq!(thread_stats.reader_cache_hits, 4);
        assert_eq!(thread_stats.writer_cache_misses, 0);
        assert_eq!(thread_stats.writer_cache_hits, 0);
        assert_eq!(thread_stats.writer_tokens_cached, 0);

        // Verify hit rates
        assert_eq!(global_stats.reader_hit_rate(), 0.8); // 4/5
        assert_eq!(global_stats.writer_hit_rate(), 0.0); // 0/5
        assert_eq!(global_stats.overall_hit_rate(), 0.4); // 4/10

        Ok(())
    }

    #[test]
    fn test_cache_clearing() -> Result<()> {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        manager.clear_thread_cache();
        manager.clear_all_stats()?;

        // Cache reader token (writers are not cached)
        with_reader_token(&manager, |_| Ok(()))?;
        with_writer_token(&manager, |_| Ok(()))?;

        // Clear thread cache
        manager.clear_thread_cache();

        // Next reader operation should be a cache miss
        with_reader_token(&manager, |_| Ok(()))?;
        with_writer_token(&manager, |_| Ok(()))?;

        let thread_stats = manager.thread_cache_stats();
        assert_eq!(thread_stats.reader_cache_misses, 2);
        assert_eq!(thread_stats.writer_cache_misses, 0);
        assert_eq!(thread_stats.writer_tokens_cached, 0);
        assert_eq!(thread_stats.cache_clears, 1);

        Ok(())
    }

    #[test]
    fn test_cross_manager_reader_cache_isolation_and_eviction() -> Result<()> {
        let readonly = TokenManager::new(ConcurrencyLevel::NoWriteReadOnly);
        let strict = TokenManager::new(ConcurrencyLevel::SingleThreadStrict);
        let owmr = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        readonly.clear_thread_cache();
        readonly.clear_all_stats()?;

        // Read-only manager caches and reuses its own reader token.
        with_reader_token(&readonly, |t| {
            assert_eq!(t.concurrency_level(), ConcurrencyLevel::NoWriteReadOnly);
            Ok(())
        })?;
        with_reader_token(&readonly, |t| {
            assert_eq!(t.concurrency_level(), ConcurrencyLevel::NoWriteReadOnly);
            Ok(())
        })?;
        assert_eq!(readonly.global_stats()?.total_reader_cache_hits, 1);

        // OneWriteMultiRead does not reuse ReadOnly's cached token, and caches its own.
        with_reader_token(&owmr, |t| {
            assert_eq!(t.concurrency_level(), ConcurrencyLevel::OneWriteMultiRead);
            assert_eq!(owmr.version_manager().active_readers(), 1);
            Ok(())
        })?;
        // Cached OWMR reader remains live while cached.
        assert_eq!(owmr.version_manager().active_readers(), 1);

        // SingleThreadStrict does not reuse OWMR's cached reader, and returning Strict's
        // reader evicts OWMR's cached reader so OWMR's active_readers drops to 0.
        with_reader_token(&strict, |t| {
            assert_eq!(t.concurrency_level(), ConcurrencyLevel::SingleThreadStrict);
            assert_eq!(strict.version_manager().active_readers(), 1);
            Ok(())
        })?;
        assert_eq!(owmr.version_manager().active_readers(), 0);
        assert_eq!(strict.version_manager().active_readers(), 1);

        strict.clear_thread_cache();
        assert_eq!(strict.version_manager().active_readers(), 0);
        Ok(())
    }

    #[test]
    fn test_foreign_return_reader_token_rejected() -> Result<()> {
        let a = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        let b = TokenManager::new(ConcurrencyLevel::MultiWriteMultiRead);
        a.clear_thread_cache();
        a.clear_all_stats()?;
        b.clear_all_stats()?;

        let token_a = a.acquire_reader_token()?;
        assert_eq!(a.version_manager().active_readers(), 1);

        // Returning `a`'s token to `b` drops it immediately without caching or
        // inflating `b`'s returned-token counter.
        b.return_reader_token(token_a);
        assert_eq!(a.version_manager().active_readers(), 0);
        assert_eq!(b.global_stats()?.total_reader_tokens_returned, 0);
        assert_eq!(b.thread_cache_stats().reader_tokens_cached, 0);

        Ok(())
    }
}
