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

use crate::error::{ZiporaError, Result};
use crate::fsa::version_sync::{ConcurrencyLevel, VersionManager, ReaderToken, WriterToken};

/// Thread-local token cache for high-performance token reuse.
///
/// This structure maintains thread-local caches of reader and writer tokens
/// to avoid the overhead of repeated token acquisition and release.
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
    /// This method first checks the thread-local cache for a valid token.
    /// If no cached token is available, it acquires a new one from the version manager.
    pub fn acquire_reader_token(&self) -> Result<ReaderToken> {
        let start_time = Instant::now();

        // Try to get a cached token first
        let token = TOKEN_CACHE.with(|cache| {
            cache.borrow_mut().get_reader_token()
        });

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

    /// Acquires a writer token with caching optimization.
    ///
    /// This method first checks the thread-local cache for a valid token.
    /// If no cached token is available, it acquires a new one from the version manager.
    pub fn acquire_writer_token(&self) -> Result<WriterToken> {
        let start_time = Instant::now();

        // Try to get a cached token first
        let token = TOKEN_CACHE.with(|cache| {
            cache.borrow_mut().get_writer_token()
        });

        let token = if let Some(cached_token) = token {
            // Update global cache hit statistics
            if let Ok(mut stats) = self.global_stats.lock() {
                stats.total_writer_cache_hits += 1;
            }
            cached_token
        } else {
            // Acquire new token from version manager
            let new_token = self.version_manager.acquire_writer_token()?;
            
            // Update global cache miss statistics
            if let Ok(mut stats) = self.global_stats.lock() {
                stats.total_writer_cache_misses += 1;
                stats.total_writer_acquisition_time += start_time.elapsed();
            }
            
            new_token
        };

        Ok(token)
    }

    /// Returns a reader token to the cache for reuse.
    ///
    /// This method caches the token in thread-local storage for future reuse,
    /// avoiding the overhead of repeated token acquisition.
    pub fn return_reader_token(&self, token: ReaderToken) {
        TOKEN_CACHE.with(|cache| {
            cache.borrow_mut().cache_reader_token(token);
        });

        // Update global statistics
        if let Ok(mut stats) = self.global_stats.lock() {
            stats.total_reader_tokens_returned += 1;
        }
    }

    /// Returns a writer token to the cache for reuse.
    ///
    /// This method caches the token in thread-local storage for future reuse,
    /// avoiding the overhead of repeated token acquisition.
    pub fn return_writer_token(&self, token: WriterToken) {
        TOKEN_CACHE.with(|cache| {
            cache.borrow_mut().cache_writer_token(token);
        });

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
        TOKEN_CACHE.with(|cache| {
            cache.borrow().stats().clone()
        })
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
        let total_requests = total_hits + self.total_reader_cache_misses + self.total_writer_cache_misses;
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
/// and returns the token to the cache for reuse. It provides a clean RAII
/// pattern for token usage.
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
/// This function automatically acquires a writer token, executes the closure,
/// and returns the token to the cache for reuse. It provides a clean RAII
/// pattern for token usage.
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

/// Trait for types that can be accessed with reader tokens.
pub trait ReaderTokenAccess {
    /// The type returned when reading with a token.
    type ReadResult;

    /// Performs a read operation with the given reader token.
    fn read_with_token(&self, token: &ReaderToken) -> Result<Self::ReadResult>;
}

/// Trait for types that can be accessed with writer tokens.
pub trait WriterTokenAccess {
    /// The type returned when writing with a token.
    type WriteResult;

    /// Performs a write operation with the given writer token.
    fn write_with_token(&mut self, token: &WriterToken) -> Result<Self::WriteResult>;
}

/// Trait for types that support both reader and writer token access.
pub trait TokenAccess: ReaderTokenAccess + WriterTokenAccess {
    /// Performs a read operation using the token manager.
    fn read_with_manager<F, R>(&self, token_manager: &TokenManager, f: F) -> Result<R>
    where
        F: FnOnce(&Self::ReadResult) -> Result<R>,
    {
        with_reader_token(token_manager, |token| {
            let result = self.read_with_token(token)?;
            f(&result)
        })
    }

    /// Performs a write operation using the token manager.
    fn write_with_manager<F, R>(&mut self, token_manager: &TokenManager, f: F) -> Result<R>
    where
        F: FnOnce(&mut Self::WriteResult) -> Result<R>,
    {
        with_writer_token(token_manager, |token| {
            let mut result = self.write_with_token(token)?;
            f(&mut result)
        })
    }
}

// Automatically implement TokenAccess for types that implement both traits
impl<T> TokenAccess for T where T: ReaderTokenAccess + WriterTokenAccess {}

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

        // Cache hit rate should be high after the first operation per thread
        assert!(global_stats.overall_hit_rate() > 0.5);

        Ok(())
    }

    #[test]
    fn test_thread_cache_isolation() -> Result<()> {
        let manager = Arc::new(TokenManager::new(ConcurrencyLevel::MultiWriteMultiRead));

        // Test that thread-local caches are isolated
        let handles: Vec<_> = (0..2)
            .map(|thread_id| {
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

        // Perform multiple operations
        for _ in 0..5 {
            with_reader_token(&manager, |_| Ok(()))?;
            with_writer_token(&manager, |_| Ok(()))?;
        }

        let global_stats = manager.global_stats()?;
        let thread_stats = manager.thread_cache_stats();

        // Verify global statistics
        assert_eq!(global_stats.total_reader_cache_misses, 1); // First acquisition
        assert_eq!(global_stats.total_reader_cache_hits, 4); // Subsequent acquisitions
        assert_eq!(global_stats.total_writer_cache_misses, 1); // First acquisition
        assert_eq!(global_stats.total_writer_cache_hits, 4); // Subsequent acquisitions

        // Verify thread-local statistics
        assert_eq!(thread_stats.reader_cache_misses, 1);
        assert_eq!(thread_stats.reader_cache_hits, 4);
        assert_eq!(thread_stats.writer_cache_misses, 1);
        assert_eq!(thread_stats.writer_cache_hits, 4);

        // Verify hit rates
        assert_eq!(global_stats.reader_hit_rate(), 0.8); // 4/5
        assert_eq!(global_stats.writer_hit_rate(), 0.8); // 4/5
        assert_eq!(global_stats.overall_hit_rate(), 0.8); // 8/10

        Ok(())
    }

    #[test]
    fn test_cache_clearing() -> Result<()> {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);

        // Cache some tokens
        with_reader_token(&manager, |_| Ok(()))?;
        with_writer_token(&manager, |_| Ok(()))?;

        // Clear thread cache
        manager.clear_thread_cache();

        // Next operations should be cache misses
        with_reader_token(&manager, |_| Ok(()))?;
        with_writer_token(&manager, |_| Ok(()))?;

        let thread_stats = manager.thread_cache_stats();
        assert_eq!(thread_stats.reader_cache_misses, 2);
        assert_eq!(thread_stats.writer_cache_misses, 2);
        assert_eq!(thread_stats.cache_clears, 1);

        Ok(())
    }
}