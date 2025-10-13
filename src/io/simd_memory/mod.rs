//! # SIMD Memory Operations
//!
//! Hardware-accelerated memory operations including high-performance copy and search operations.
//!
//! ## Modules
//!
//! - **copy**: High-performance SIMD memory copy with non-temporal stores
//! - **search**: SSE4.2 PCMPESTRI-based string search operations
//!
//! ## Performance Targets
//!
//! - **Large copies** (>1KB): 35-50 GB/s with streaming stores
//! - **Small copies** (16-256B): 15-25 GB/s with minimal overhead
//! - **Aligned copies**: Maximum throughput with cache-line alignment
//! - **Character search**: 2-4x faster than memchr
//! - **Pattern search**: 2-8x faster than naive search
//!
//! ## Architecture
//!
//! - **6-Tier SIMD Framework**: AVX-512 → AVX2 → SSE4.2 → SSE2 → NEON → Scalar
//! - **Runtime CPU Detection**: Optimal implementation selection
//! - **Non-Temporal Stores**: Bypass cache for large buffers
//! - **PCMPESTRI Instructions**: Hardware string comparison with early exit
//! - **Zero Unsafe in Public APIs**: Memory safety guaranteed

pub mod copy;
pub mod search;

pub use copy::{copy_large_simd, copy_small_simd, copy_aligned_simd, SimdCopy, SimdCopyTier};
pub use search::{
    find_char, find_pattern, find_any_of, compare_strings,
    SimdStringSearch, SearchTier, SearchConfig,
};
