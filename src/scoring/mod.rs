//! Search engine scoring utilities
//!
//! This module provides fieldnorm encoding and BM25 scoring infrastructure
//! for compact document-length storage and fast scoring.
//!
//! - [`FieldnormEncoder`] — Lucene-compatible SmallFloat encoding (Tier 1)
//! - [`Bm25BatchScorer`] — SIMD-accelerated batch scoring (Tier 2) + prefetch (Tier 3)

pub mod bm25;
pub mod fieldnorm;

pub use bm25::{Bm25BatchScorer, prefetch_fieldnorm};
pub use fieldnorm::FieldnormEncoder;
