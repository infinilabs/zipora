//! Succinct data structures with constant-time rank and select operations
//!
//! This module provides space-efficient bit vectors and arrays with support for
//! rank (count of set bits up to position) and select (find position of nth set bit)
//! operations in constant time with ~3% space overhead.

pub mod bit_vector;
pub mod rank_select;

pub use bit_vector::{BitVector, BitwiseOp};
pub use rank_select::{
    BuilderOptions,
    CpuFeatures,
    MixedDimensionView,
    PerformanceStats,
    RankSelect256,
    RankSelectBuilder,
    RankSelectFew,
    RankSelectFewBuilder,
    RankSelectInterleaved256,
    RankSelectMixedIL256,
    RankSelectMixed_IL_256,
    RankSelectMixedSE512,
    RankSelectMixedXL256,
    RankSelectMixedXLBitPacked,
    RankSelectMultiDimensional,
    // New rank/select variants
    RankSelectOps,
    RankSelectPerformanceOps,
    RankSelectSe256,
    RankSelectSeparated256,
    RankSelectSeparated512,
    RankSelectSimple,
    RankSelectSparse,
    SimdCapabilities,
    // SIMD operations
    SimdOps,
    bulk_popcount_simd,
    bulk_rank1_simd,
    bulk_select1_simd,
};
