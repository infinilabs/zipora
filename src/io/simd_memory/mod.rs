//! # SIMD Memory Operations
//!
//! Hardware-accelerated search operations using SSE4.2 PCMPESTRI instructions.
//!
//! ## Modules
//!
//! - **search**: SSE4.2 PCMPESTRI-based string search operations
//!
//! ## Performance
//!
//! - **Character search**: 2-4x faster than memchr
//! - **Pattern search**: 2-8x faster than naive search
//!
//! ## Architecture
//!
//! - **6-Tier SIMD Framework**: AVX-512 → AVX2 → SSE4.2 → SSE2 → NEON → Scalar
//! - **Runtime CPU Detection**: Optimal implementation selection
//! - **PCMPESTRI Instructions**: Hardware string comparison with early exit
//! - **Zero Unsafe in Public APIs**: Memory safety guaranteed

pub mod search;

pub use search::{
    SearchConfig, SearchTier, SimdStringSearch, compare_strings, find_any_of, find_char,
    find_pattern,
};
