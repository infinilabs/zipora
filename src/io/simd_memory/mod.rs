//! # SIMD Memory Operations
//!
//! Search operations with SSE4.2 PCMPESTRI kernels on x86_64 and scalar
//! code everywhere else.
//!
//! ## Modules
//!
//! - **search**: SSE4.2 PCMPESTRI-based string search operations
//!
//! ## Performance
//!
//! Limitation: no speedup over `memchr` or a naive search is claimed.
//!
//! ## Architecture
//!
//! - **Two tiers**: SSE4.2 and scalar; the AVX2, AVX-512 and NEON tier
//!   names exist in [`SearchTier`] but are never selected
//! - **Runtime CPU Detection**: the SSE4.2 tier is used when the CPU has
//!   SSE4.2 and [`SearchConfig::enable_sse42`] is set
//! - **PCMPESTRI Instructions**: character, pattern and byte-set search;
//!   string comparison uses SSE2 PCMPEQB
//! - **In-bounds loads**: no SIMD load reads outside the caller's slices

pub mod search;

pub use search::{
    SearchConfig, SearchTier, SimdStringSearch, compare_strings, find_any_of, find_char,
    find_pattern,
};
