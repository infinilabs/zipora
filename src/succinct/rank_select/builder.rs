//! Builder Utilities for Rank/Select Structures
//!
//! This module provides utilities and helper functions for constructing
//! rank/select structures with various optimizations and configurations.

use super::{BuilderOptions, RankSelectInterleaved256};
use crate::error::Result;
use crate::succinct::BitVector;

/// Factory for creating optimal rank/select implementations.
///
/// Currently always returns `RankSelectInterleaved256` — the verified
/// best performer across all data profiles.
pub struct RankSelectFactory;

impl RankSelectFactory {
    /// Create the best rank/select implementation for given data characteristics.
    pub fn create_optimal(
        bit_vector: BitVector,
        opts: BuilderOptions,
    ) -> Result<RankSelectInterleaved256> {
        <RankSelectInterleaved256 as super::RankSelectBuilder<RankSelectInterleaved256>>::with_optimizations(
            bit_vector, opts,
        )
    }
}

/// Analyze bit vector characteristics to suggest optimal configuration
pub fn analyze_bit_vector(bit_vector: &BitVector) -> BuilderOptions {
    let total_bits = bit_vector.len();
    let total_ones = bit_vector.count_ones();
    let density = if total_bits == 0 {
        0.0
    } else {
        total_ones as f64 / total_bits as f64
    };

    BuilderOptions {
        optimize_select: total_ones > 0,
        block_size: super::utils::optimal_block_size(total_bits, density),
        select_sample_rate: super::utils::optimal_select_sample_rate(total_ones),
        enable_simd: true,
        prefer_space: !(0.05..=0.95).contains(&density),
    }
}
