//! # Dynamic SIMD Selection Module
//!
//! Runtime adaptive SIMD selection surpassing compile-time approaches with:
//! - Runtime hardware detection and micro-benchmarking
//! - Data-aware selection based on size, density, and access patterns
//! - Continuous performance monitoring with adaptive adjustment
//! - Seamless integration with zipora's 6-tier SIMD framework
//!
//! ## Example
//!
//! ```
//! use zipora::simd::{AdaptiveSimdSelector, Operation};
//!
//! // Get global selector instance
//! let selector = AdaptiveSimdSelector::global();
//!
//! // Select optimal SIMD implementation
//! let impl_type = selector.select_optimal_impl(
//!     Operation::Rank,
//!     4096,  // data size
//!     None,  // data density (optional)
//! );
//! ```

pub mod adaptive;
pub mod benchmarks;
pub mod performance;

pub use adaptive::{AdaptiveSimdSelector, SelectionKey, SimdImpl, SimdTier};
pub use benchmarks::BenchmarkResults;
pub use performance::PerformanceHistory;

/// Operation types for SIMD selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Rank operation (count 1s up to position)
    Rank,
    /// Select operation (find position of nth 1)
    Select,
    /// Popcount (count total 1s)
    Popcount,
    /// String/byte search
    Search,
    /// Sorting operations
    Sort,
    /// Compression
    Compress,
    /// Decompression
    Decompress,
    /// Hash computation
    Hash,
    /// String search (SSE4.2 PCMPESTRI)
    StringSearch,
    /// Bit manipulation
    BitManip,
    /// Memory zeroing/filling
    MemZero,
    /// Memory copy operations
    Copy,
}
