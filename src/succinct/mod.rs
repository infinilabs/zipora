//! Succinct data structures with constant-time rank and select operations
//!
//! This module provides space-efficient bit vectors and arrays with support for
//! rank (count of set bits up to position) and select (find position of nth set bit)
//! operations in constant time with ~3% space overhead.

pub mod bit_vector;
pub mod rank_select;

pub use bit_vector::{BitVector, BitwiseOp};
pub use rank_select::{
    RankSelect256, RankSelectSe256, CpuFeatures,
    // New rank/select variants
    RankSelectOps, RankSelectPerformanceOps, RankSelectMultiDimensional, RankSelectSparse,
    RankSelectBuilder, BuilderOptions, PerformanceStats,
    RankSelectSimple, RankSelectSeparated256, RankSelectSeparated512,
    RankSelectInterleaved256, RankSelectFew, RankSelectFewBuilder,
    RankSelectMixedIL256, RankSelectMixedSE512, RankSelectMixedXL256,
    MixedDimensionView,
    // SIMD operations
    SimdOps, bulk_rank1_simd, bulk_select1_simd, bulk_popcount_simd, SimdCapabilities,
};
