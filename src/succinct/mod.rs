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
    PerformanceStats,
    RankSelect256,
    RankSelectBuilder,
    RankSelectInterleaved256,
    // New rank/select variants
    RankSelectOps,
    RankSelectPerformanceOps,
    // Advanced optimization variants
    AdaptiveRankSelect,
    AdaptiveMultiDimensional,
    DataProfile,
    SelectionCriteria,
    // BMI2 acceleration
    Bmi2Accelerator,
    Bmi2BitOps,
    Bmi2BlockOps,
    Bmi2Capabilities,
    Bmi2PrefetchOps,
    Bmi2RangeOps,
    Bmi2RankOps,
    Bmi2SelectOps,
    Bmi2SequenceOps,
    Bmi2Stats,
    SimdCapabilities,
    // SIMD operations
    SimdOps,
    bulk_popcount_simd,
    bulk_rank1_simd,
    bulk_select1_simd,
};
