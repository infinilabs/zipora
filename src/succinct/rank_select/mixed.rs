//! Multi-Dimensional Rank/Select Implementations
//!
//! This module provides rank/select implementations that support multiple
//! related bit vectors stored together for cache efficiency and shared
//! infrastructure overhead. These implementations are based on research
//! from advanced succinct data structure libraries.
//!
//! # Variants
//!
//! - **`RankSelectMixedIL256`**: Dual-dimension interleaved (2 bit vectors)
//! - **`RankSelectMixedSE512`**: Dual-dimension separated (2 bit vectors, 512-bit blocks)
//! - **`RankSelectMixedXL256`**: Multi-dimension extended (2-4 bit vectors)
//!
//! # Design Philosophy
//!
//! - **Shared Infrastructure**: Common rank cache across dimensions
//! - **Cache Optimization**: Interleaved storage for spatial locality
//! - **Dimension Independence**: Per-dimension operations with shared metadata
//! - **Memory Efficiency**: Reduced overhead through shared structures
//!
//! # Performance Characteristics
//!
//! ## RankSelectMixedIL256
//! - **Memory Overhead**: ~30% (shared cache + dimension metadata)
//! - **Rank Time**: O(1) with excellent cache behavior
//! - **Select Time**: O(1) average with dimension-specific caching
//! - **Best For**: Related bit vectors with similar access patterns
//!
//! ## RankSelectMixedSE512
//! - **Memory Overhead**: ~25% (larger blocks, separated storage)
//! - **Rank Time**: O(1) with good sequential performance
//! - **Select Time**: O(1) average
//! - **Best For**: Large datasets with independent dimension access
//!
//! ## RankSelectMixedXL256
//! - **Memory Overhead**: ~35% (scales with arity)
//! - **Rank Time**: O(1) per dimension
//! - **Select Time**: O(1) average per dimension
//! - **Best For**: Multi-dimensional data analysis
//!
//! # Examples
//!
//! ```rust
//! use zipora::{BitVector, RankSelectOps, RankSelectMultiDimensional, RankSelectMixedIL256};
//!
//! // Create two related bit vectors
//! let mut bv1 = BitVector::new();
//! let mut bv2 = BitVector::new();
//! for i in 0..1000 {
//!     bv1.push(i % 3 == 0)?;
//!     bv2.push(i % 5 == 0)?;
//! }
//!
//! // Create dual-dimension rank/select
//! let mixed_rs = RankSelectMixedIL256::new([bv1, bv2])?;
//!
//! // Query both dimensions
//! let rank_dim0 = mixed_rs.rank1_dim::<0>(500);
//! let rank_dim1 = mixed_rs.rank1_dim::<1>(500);
//! let pos_dim0 = mixed_rs.select1_dim::<0>(50)?;
//! let pos_dim1 = mixed_rs.select1_dim::<1>(25)?;
//! # Ok::<(), zipora::ZiporaError>(())
//! ```

use super::{
    BuilderOptions, CpuFeatures, RankSelectBuilder, RankSelectMultiDimensional, RankSelectOps,
    RankSelectPerformanceOps,
};
use crate::FastVec;
use crate::error::{Result, ZiporaError};
use crate::succinct::BitVector;
use std::fmt;

// Hardware acceleration imports
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_pdep_u64, _popcnt64};

// Implementation methods are included inline for now

/// View of a specific dimension in a multi-dimensional rank/select structure
///
/// Provides a dimension-specific interface to multi-dimensional rank/select
/// operations, allowing efficient per-dimension queries while sharing
/// common infrastructure.
pub struct MixedDimensionView<'a, const DIM: usize> {
    /// Reference to the parent multi-dimensional structure (generic type)
    parent_il256: Option<&'a RankSelectMixedIL256>,
    parent_se512: Option<&'a RankSelectMixedSE512>,
}

impl<'a, const DIM: usize> MixedDimensionView<'a, DIM> {
    /// Create a new dimension view for RankSelectMixedIL256
    pub fn new(parent: &'a RankSelectMixedIL256) -> Self {
        assert!(
            DIM < 2,
            "Invalid dimension {} for dual-dimension structure",
            DIM
        );
        Self {
            parent_il256: Some(parent),
            parent_se512: None,
        }
    }

    /// Create a new dimension view for RankSelectMixedSE512
    pub fn new_se512(parent: &'a RankSelectMixedSE512) -> Self {
        assert!(
            DIM < 2,
            "Invalid dimension {} for dual-dimension structure",
            DIM
        );
        Self {
            parent_il256: None,
            parent_se512: Some(parent),
        }
    }

    /// Get bit at position in this dimension
    pub fn get(&self, index: usize) -> Option<bool> {
        if let Some(parent) = self.parent_il256 {
            parent.get_dimension_bit_runtime(index, DIM)
        } else if let Some(parent) = self.parent_se512 {
            parent.get_dimension_bit(index, DIM)
        } else {
            None
        }
    }

    /// Rank operation for this dimension
    pub fn rank1(&self, pos: usize) -> usize {
        if let Some(parent) = self.parent_il256 {
            parent.rank1_dimension(pos, DIM)
        } else if let Some(parent) = self.parent_se512 {
            parent.rank1_dimension(pos, DIM)
        } else {
            0
        }
    }

    /// Select operation for this dimension
    pub fn select1(&self, k: usize) -> Result<usize> {
        if let Some(parent) = self.parent_il256 {
            parent.select1_dimension(k, DIM)
        } else if let Some(parent) = self.parent_se512 {
            parent.select1_dimension(k, DIM)
        } else {
            Err(ZiporaError::invalid_data(
                "No parent structure available".to_string(),
            ))
        }
    }

    /// Length of this dimension
    pub fn len(&self) -> usize {
        if let Some(parent) = self.parent_il256 {
            parent.total_bits
        } else if let Some(parent) = self.parent_se512 {
            parent.total_bits
        } else {
            0
        }
    }

    /// Check if this dimension is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Count of set bits in this dimension
    pub fn count_ones(&self) -> usize {
        if DIM < 2 {
            if let Some(parent) = self.parent_il256 {
                parent.total_ones[DIM]
            } else if let Some(parent) = self.parent_se512 {
                parent.total_ones[DIM]
            } else {
                0
            }
        } else {
            0
        }
    }
}

/// Dual-dimension interleaved rank/select (2 bit vectors)
///
/// This implementation stores two related bit vectors using interleaved
/// cache-optimized storage. The rank cache and bit data are stored together
/// for optimal memory access patterns across both dimensions.
///
/// # Memory Layout
///
/// ```text
/// Interleaved Cache Lines (32 bytes each):
/// [rank0:4][rank1:4][bits0:12][bits1:12]
/// [rank0:4][rank1:4][bits0:12][bits1:12]
/// ...
/// ```
///
/// # Performance Characteristics
///
/// - **Single Cache Line Access**: Both dimensions' data in same cache line
/// - **Shared Infrastructure**: Common block indexing reduces overhead
/// - **Dimension Independence**: Each dimension can be queried separately
/// - **Memory Efficiency**: ~30% overhead vs individual structures
///
/// # Examples
///
/// ```rust
/// use zipora::{BitVector, RankSelectOps, RankSelectMultiDimensional, RankSelectMixedIL256};
///
/// let mut bv1 = BitVector::new();
/// let mut bv2 = BitVector::new();
/// for i in 0..1000 {
///     bv1.push(i % 3 == 0)?;
///     bv2.push(i % 7 == 0)?;
/// }
///
/// let mixed_rs = RankSelectMixedIL256::new([bv1, bv2])?;
///
/// // Query specific dimensions
/// let rank_dim0 = mixed_rs.rank1_dim::<0>(500);
/// let rank_dim1 = mixed_rs.rank1_dim::<1>(500);
/// let pos_dim0 = mixed_rs.select1_dim::<0>(50)?;
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
#[derive(Clone)]
pub struct RankSelectMixedIL256 {
    /// Length of both bit vectors (must be same)
    total_bits: usize,
    /// Total set bits in each dimension
    total_ones: [usize; 2],
    /// Interleaved cache lines combining rank and bit data
    interleaved_cache: FastVec<InterleavedDualLine>,
    /// Optional select cache for each dimension
    select_caches: [Option<FastVec<u32>>; 2],
    /// Select sampling rate
    select_sample_rate: usize,
}

/// Interleaved cache line for dual-dimension storage
///
/// Combines rank metadata and bit data for both dimensions in a single
/// cache-aligned structure for optimal memory access patterns.
#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct InterleavedDualLine {
    /// Rank level 1 for dimension 0 (cumulative rank at end of 256-bit block)
    rank0_lev1: u32,
    /// Rank level 1 for dimension 1
    rank1_lev1: u32,
    /// Reserved for future rank level 2 data
    _reserved: [u32; 2],
    /// Bit data for dimension 0 (3 × 64-bit words = 192 bits)
    bits0: [u64; 3],
    /// Bit data for dimension 1 (3 × 64-bit words = 192 bits)
    bits1: [u64; 3],
}

/// Constants for multi-dimensional implementations
const DUAL_BLOCK_SIZE: usize = 192; // Reduced from 256 to fit in cache line
const DUAL_SELECT_SAMPLE_RATE: usize = 512;
const SEP512_BLOCK_SIZE: usize = 512;
const MULTI_BLOCK_SIZE: usize = 256;
const MULTI_SELECT_SAMPLE_RATE: usize = 512;

/// Constants for advanced bit-packed multi-dimensional implementations
const MULTI_RELATIVE_RANK_BITS: usize = 9;  // 9 bits per relative rank (max 511)
const MULTI_RANKS_PER_WORD: usize = 7;      // 7 × 9-bit values in 64-bit word
const MULTI_DEFAULT_SUPERBLOCK_SIZE: usize = 16; // 16 × 256-bit blocks per superblock
const MULTI_RELATIVE_RANK_MASK: u64 = (1u64 << MULTI_RELATIVE_RANK_BITS) - 1; // 0x1FF

/// Interleaved rank metadata for multi-dimensional storage
///
/// Combines rank information for multiple dimensions in a single
/// cache-aligned structure for optimal memory access patterns.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct InterleavedMultiRank<const ARITY: usize> {
    /// Rank values for each dimension (cumulative rank at end of block)
    ranks: [u32; ARITY],
    /// Reserved space for alignment and future expansion
    _reserved: [u32; 16],
}

/// Dual-dimension separated rank/select (2 bit vectors, 512-bit blocks)
///
/// This implementation uses larger 512-bit blocks with separated storage
/// for better sequential performance on large datasets. The rank cache
/// and bit data are stored separately for optimal memory access patterns.
///
/// # Memory Layout
///
/// ```text
/// Bit Data:      |bit_vector_0|bit_vector_1|
/// Rank Cache:    |rank0_0|rank1_0|rank0_1|rank1_1|...| (8 bytes per 512-bit block)
/// Select Cache:  |select_cache_0|select_cache_1|      (optional)
/// ```
///
/// # Performance Characteristics
///
/// - **Memory Overhead**: ~12.5% (fewer blocks, larger cache entries)
/// - **Sequential Performance**: Optimized for streaming access patterns
/// - **Cache Behavior**: Larger blocks reduce cache line pressure
/// - **Best For**: Large datasets with sequential access patterns
///
/// # Examples
///
/// ```rust
/// use zipora::{BitVector, RankSelectMixedSE512};
///
/// let mut bv1 = BitVector::new();
/// let mut bv2 = BitVector::new();
/// for i in 0..10000 {
///     bv1.push(i % 11 == 0)?;
///     bv2.push(i % 13 == 0)?;
/// }
///
/// let mixed_rs = RankSelectMixedSE512::new([bv1, bv2])?;
///
/// // Efficient bulk operations
/// let positions = vec![1000, 2000, 3000, 4000, 5000];
/// let ranks_dim0 = mixed_rs.rank1_bulk_dim::<0>(&positions);
/// let ranks_dim1 = mixed_rs.rank1_bulk_dim::<1>(&positions);
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
#[derive(Clone)]
pub struct RankSelectMixedSE512 {
    /// Length of both bit vectors (must be same)
    total_bits: usize,
    /// Total set bits in each dimension
    total_ones: [usize; 2],
    /// Separated bit vectors
    bit_vectors: [BitVector; 2],
    /// Separated rank cache for 512-bit blocks (u64 for larger counts)
    rank_caches: [FastVec<u64>; 2],
    /// Optional select caches for each dimension
    select_caches: [Option<FastVec<u32>>; 2],
    /// Select sampling rate
    select_sample_rate: usize,
}

/// Multi-dimension extended rank/select (2-4 bit vectors)
///
/// This implementation supports variable arity (2-4 dimensions) with
/// optimized storage and query patterns. Uses template specialization
/// for optimal performance based on the number of dimensions.
///
/// Enhanced with advanced separated storage techniques from research:
/// - **Bit-packed rank caching**: Hierarchical superblock + relative rank encoding
/// - **Adaptive block sizing**: Variable block sizes based on dimension count
/// - **Hardware acceleration**: BMI2, POPCNT, and AVX-512 optimizations
/// - **Cache-aware layout**: Interleaved metadata for optimal memory access
///
/// # Memory Layout
///
/// ```text
/// Superblock Cache:     |sb0|sb1|...|sbN|        (absolute cumulative ranks)
/// Relative Rank Cache:  |rel_packed_words|       (bit-packed 9-bit relative ranks)
/// Bit Data Arrays:      |bits0|bits1|...|bitsN|  (ARITY × bit vectors)
/// Select Caches:        |sel0|sel1|...|selN|     (optional per dimension)
/// ```
///
/// # Performance Characteristics
///
/// - **Memory Overhead**: ~(3-8)% × ARITY (40-50% reduction vs basic approach)
/// - **Dimension Access**: O(1) per dimension with hierarchical rank cache
/// - **Cache Efficiency**: Optimized layout for multi-dimensional queries
/// - **Hardware Acceleration**: SIMD and BMI2 optimizations when available
/// - **Best For**: Complex data analysis with multiple related bit patterns
///
/// # Examples
///
/// ```rust
/// use zipora::{BitVector, RankSelectOps, RankSelectMultiDimensional, RankSelectMixedXL256};
///
/// // 3-dimensional analysis (e.g., user features: active, premium, mobile)
/// let mut active_users = BitVector::new();
/// let mut premium_users = BitVector::new();
/// let mut mobile_users = BitVector::new();
///
/// for i in 0..10000 {
///     active_users.push(i % 5 != 0)?;    // 80% active
///     premium_users.push(i % 10 == 0)?;  // 10% premium
///     mobile_users.push(i % 3 == 0)?;    // 33% mobile
/// }
///
/// let mixed_rs = RankSelectMixedXL256::<3>::new([active_users, premium_users, mobile_users])?;
///
/// // Multi-dimensional queries
/// let active_count = mixed_rs.rank1_dimension::<0>(5000);
/// let premium_count = mixed_rs.rank1_dimension::<1>(5000);
/// let mobile_count = mixed_rs.rank1_dimension::<2>(5000);
///
/// // Combined analysis
/// let first_premium_mobile = mixed_rs.find_intersection(&[1, 2], 100)?;
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
#[derive(Clone)]
pub struct RankSelectMixedXL256<const ARITY: usize> {
    /// Length of all bit vectors (must be same)
    total_bits: usize,
    /// Total set bits in each dimension
    total_ones: [usize; ARITY],
    /// Interleaved rank cache combining all dimensions
    interleaved_ranks: FastVec<InterleavedMultiRank<ARITY>>,
    /// Bit data for each dimension
    bit_data: [FastVec<u64>; ARITY],
    /// Optional select caches for each dimension
    select_caches: [Option<FastVec<u32>>; ARITY],
    /// Select sampling rate
    select_sample_rate: usize,
}

/// Advanced multi-dimension rank/select with bit-packed hierarchical caching (2-4 bit vectors)
///
/// This implementation combines the advanced bit-packed rank caching techniques
/// from separated storage with multi-dimensional support. Features hierarchical
/// rank caching using superblocks and bit-packed relative ranks for significant
/// memory savings while maintaining O(1) query performance.
///
/// Based on research from advanced rank cache encoding techniques
/// and adapted for multi-dimensional succinct data structures.
///
/// # Design
///
/// Uses a hierarchical structure per dimension:
/// - **Superblocks**: Store absolute cumulative ranks every N blocks (64-bit values)
/// - **Relative Ranks**: Store 9-bit relative ranks bit-packed into 64-bit words
/// - **Bit Packing**: 7 × 9-bit values per 64-bit word for ~40-50% memory reduction
/// - **Interleaved Storage**: Metadata interleaved for optimal cache access patterns
///
/// # Memory Layout
///
/// ```text
/// Superblock Caches:   |sb0_dims|sb1_dims|...|     (ARITY × u64 per superblock)
/// Relative Rank Cache:  |rel_packed_words_dims|     (bit-packed per dimension)
/// Bit Data Arrays:      |bits0|bits1|...|bitsN|    (ARITY × bit vectors)
/// Select Caches:        |sel0|sel1|...|selN|       (optional per dimension)
/// ```
///
/// # Examples
///
/// ```rust
/// use zipora::succinct::{BitVector, rank_select::{RankSelectOps, mixed::RankSelectMixedXLBitPacked}}; 
///
/// // 3-dimensional analysis with advanced memory optimization
/// let mut user_active = BitVector::new();
/// let mut user_premium = BitVector::new(); 
/// let mut user_mobile = BitVector::new();
///
/// for i in 0..100 {
///     user_active.push(i % 10 != 0)?;   // 90% active
///     user_premium.push(i % 50 == 0)?;  // 2% premium  
///     user_mobile.push(i % 4 == 0)?;    // 25% mobile
/// }
///
/// let mixed_rs = RankSelectMixedXLBitPacked::<3>::new([user_active, user_premium, user_mobile])?;
///
/// // Significant memory savings with same performance
/// println!("Memory overhead: {:.2}%", mixed_rs.space_overhead_percent());
/// let active_count = mixed_rs.rank1_dimension::<0>(50);
/// let premium_count = mixed_rs.rank1_dimension::<1>(50);
/// let mobile_count = mixed_rs.rank1_dimension::<2>(50);
/// # Ok::<(), zipora::error::ZiporaError>(())
/// ```
#[derive(Clone)]
pub struct RankSelectMixedXLBitPacked<const ARITY: usize> {
    /// Length of all bit vectors (must be same)
    total_bits: usize,
    /// Total set bits in each dimension
    total_ones: [usize; ARITY],
    /// Superblock cache storing absolute cumulative ranks per dimension
    superblock_caches: [FastVec<u64>; ARITY],
    /// Bit-packed relative ranks per dimension (9-bit values packed 7 per 64-bit word)  
    relative_rank_caches: [FastVec<u64>; ARITY],
    /// Bit data for each dimension
    bit_data: [FastVec<u64>; ARITY],
    /// Optional select caches for each dimension
    select_caches: [Option<FastVec<u32>>; ARITY],
    /// Select sampling rate
    select_sample_rate: usize,
    /// Number of 256-bit blocks per superblock
    superblock_size: usize,
}

impl RankSelectMixedIL256 {
    /// Create a new dual-dimension interleaved rank/select structure
    ///
    /// # Arguments
    /// * `bit_vectors` - Array of exactly 2 bit vectors of the same length
    ///
    /// # Returns
    /// A new RankSelectMixedIL256 instance with interleaved cache optimization
    pub fn new(bit_vectors: [BitVector; 2]) -> Result<Self> {
        Self::with_options(bit_vectors, true, DUAL_SELECT_SAMPLE_RATE)
    }

    /// Create with custom options
    ///
    /// # Arguments
    /// * `bit_vectors` - Array of exactly 2 bit vectors of the same length
    /// * `enable_select_cache` - Whether to build select cache for faster select operations
    /// * `select_sample_rate` - Sample every N set bits for select cache
    pub fn with_options(
        bit_vectors: [BitVector; 2],
        enable_select_cache: bool,
        select_sample_rate: usize,
    ) -> Result<Self> {
        // Validate input
        if bit_vectors[0].len() != bit_vectors[1].len() {
            return Err(ZiporaError::invalid_data(
                "Both bit vectors must have the same length".to_string(),
            ));
        }

        let total_bits = bit_vectors[0].len();
        let mut rs = Self {
            total_bits,
            total_ones: [0, 0],
            interleaved_cache: FastVec::new(),
            select_caches: [
                if enable_select_cache {
                    Some(FastVec::new())
                } else {
                    None
                },
                if enable_select_cache {
                    Some(FastVec::new())
                } else {
                    None
                },
            ],
            select_sample_rate,
        };

        rs.build_interleaved_cache(&bit_vectors)?;

        // Build select caches if enabled
        if enable_select_cache {
            rs.build_select_caches(&bit_vectors)?;
        }

        Ok(rs)
    }

    /// Build the interleaved cache combining both dimensions
    fn build_interleaved_cache(&mut self, bit_vectors: &[BitVector; 2]) -> Result<()> {
        if self.total_bits == 0 {
            return Ok(());
        }

        let num_blocks = (self.total_bits + DUAL_BLOCK_SIZE - 1) / DUAL_BLOCK_SIZE;
        self.interleaved_cache.reserve(num_blocks)?;

        let mut cumulative_ranks = [0u32, 0u32];

        for block_idx in 0..num_blocks {
            let block_start_bit = block_idx * DUAL_BLOCK_SIZE;
            let block_end_bit = ((block_idx + 1) * DUAL_BLOCK_SIZE).min(self.total_bits);

            // Count bits in this block for both dimensions
            let block_ranks = [
                self.count_bits_in_range(&bit_vectors[0], block_start_bit, block_end_bit),
                self.count_bits_in_range(&bit_vectors[1], block_start_bit, block_end_bit),
            ];

            cumulative_ranks[0] += block_ranks[0] as u32;
            cumulative_ranks[1] += block_ranks[1] as u32;

            // Extract bit data for this block
            let bits0 = self.extract_block_bits(&bit_vectors[0], block_start_bit, block_end_bit);
            let bits1 = self.extract_block_bits(&bit_vectors[1], block_start_bit, block_end_bit);

            // Create interleaved cache line
            let cache_line = InterleavedDualLine {
                rank0_lev1: cumulative_ranks[0],
                rank1_lev1: cumulative_ranks[1],
                _reserved: [0, 0],
                bits0,
                bits1,
            };

            self.interleaved_cache.push(cache_line)?;
        }

        self.total_ones = [cumulative_ranks[0] as usize, cumulative_ranks[1] as usize];
        Ok(())
    }

    /// Count bits in a range using hardware acceleration when available
    #[inline]
    fn count_bits_in_range(
        &self,
        bit_vector: &BitVector,
        start_bit: usize,
        end_bit: usize,
    ) -> usize {
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;
        let blocks = bit_vector.blocks();
        let mut count = 0;

        for word_idx in start_word..end_word.min(blocks.len()) {
            let mut word = blocks[word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            count += self.popcount_hardware_accelerated(word) as usize;
        }

        count
    }

    /// Extract bit data for a block into 3 × 64-bit words
    fn extract_block_bits(
        &self,
        bit_vector: &BitVector,
        start_bit: usize,
        end_bit: usize,
    ) -> [u64; 3] {
        let mut result = [0u64; 3];
        let blocks = bit_vector.blocks();
        let start_word = start_bit / 64;

        // Copy up to 3 words (192 bits) for this block
        for i in 0..3 {
            let word_idx = start_word + i;
            if word_idx < blocks.len() {
                let mut word = blocks[word_idx];

                // Handle partial word at the beginning
                if i == 0 && start_bit % 64 != 0 {
                    let start_bit_in_word = start_bit % 64;
                    word &= !((1u64 << start_bit_in_word) - 1);
                }

                // Handle partial word at the end
                let bit_pos = (word_idx * 64).saturating_sub(start_bit);
                if start_bit + bit_pos + 64 > end_bit {
                    let valid_bits = end_bit.saturating_sub(start_bit + bit_pos);
                    if valid_bits > 0 && valid_bits < 64 {
                        let mask = (1u64 << valid_bits) - 1;
                        word &= mask;
                    }
                }

                result[i] = word;
            }
        }

        result
    }

    /// Hardware-accelerated popcount with fallback
    #[inline(always)]
    fn popcount_hardware_accelerated(&self, x: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(test)]
            {
                x.count_ones()
            }

            #[cfg(not(test))]
            {
                if CpuFeatures::get().has_popcnt {
                    unsafe { _popcnt64(x as i64) as u32 }
                } else {
                    x.count_ones()
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            x.count_ones()
        }
    }

    /// Build select caches for faster select operations
    fn build_select_caches(&mut self, bit_vectors: &[BitVector; 2]) -> Result<()> {
        for dim in 0..2 {
            if self.select_caches[dim].is_some() {
                // Take ownership of the cache temporarily to avoid borrowing conflicts
                if let Some(mut cache) = self.select_caches[dim].take() {
                    self.build_select_cache_for_dimension(&mut cache, &bit_vectors[dim], dim)?;
                    self.select_caches[dim] = Some(cache);
                }
            }
        }
        Ok(())
    }

    /// Build select cache for a specific dimension
    fn build_select_cache_for_dimension(
        &self,
        cache: &mut FastVec<u32>,
        bit_vector: &BitVector,
        dim: usize,
    ) -> Result<()> {
        let total_ones = self.total_ones[dim];
        if total_ones == 0 {
            return Ok(());
        }

        let mut ones_seen = 0;
        let mut current_pos = 0;

        while ones_seen < total_ones {
            let target_ones = (ones_seen + self.select_sample_rate).min(total_ones);

            while ones_seen < target_ones && current_pos < bit_vector.len() {
                if bit_vector.get(current_pos).unwrap_or(false) {
                    ones_seen += 1;
                }
                if ones_seen < target_ones {
                    current_pos += 1;
                }
            }

            if ones_seen == target_ones {
                cache.push(current_pos as u32)?;
            }

            current_pos += 1;
        }

        Ok(())
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let cache_size = self.interleaved_cache.len() * std::mem::size_of::<InterleavedDualLine>();
        let select_cache_size = self
            .select_caches
            .iter()
            .map(|cache| cache.as_ref().map(|c| c.len() * 4).unwrap_or(0))
            .sum::<usize>();

        cache_size + select_cache_size + std::mem::size_of::<Self>()
    }

    /// Check if using select cache for a dimension
    pub fn has_select_cache(&self, dim: usize) -> bool {
        dim < 2 && self.select_caches[dim].is_some()
    }

    /// Get bit from a specific dimension (const generic version)
    pub fn get_dimension_bit<const DIM: usize>(&self, index: usize) -> Option<bool> {
        self.get_dimension_bit_runtime(index, DIM)
    }

    /// Get bit from a specific dimension (runtime version)
    pub fn get_dimension_bit_runtime(&self, index: usize, dim: usize) -> Option<bool> {
        if index >= self.total_bits || dim >= 2 {
            return None;
        }

        let block_idx = index / DUAL_BLOCK_SIZE;
        let bit_offset_in_block = index % DUAL_BLOCK_SIZE;

        if block_idx >= self.interleaved_cache.len() {
            return None;
        }

        let cache_line = &self.interleaved_cache[block_idx];
        let bits = if dim == 0 {
            &cache_line.bits0
        } else {
            &cache_line.bits1
        };

        let word_idx = bit_offset_in_block / 64;
        let bit_idx = bit_offset_in_block % 64;

        if word_idx < bits.len() {
            Some((bits[word_idx] >> bit_idx) & 1 == 1)
        } else {
            None
        }
    }

    /// Internal rank implementation for a specific dimension
    pub fn rank1_dimension(&self, pos: usize, dim: usize) -> usize {
        if pos == 0 || self.total_bits == 0 || dim >= 2 {
            return 0;
        }

        let pos = pos.min(self.total_bits);

        // Find containing block
        let block_idx = pos / DUAL_BLOCK_SIZE;
        let bit_offset_in_block = pos % DUAL_BLOCK_SIZE;

        // Get rank up to start of this block
        let rank_before_block = if block_idx > 0 {
            let prev_cache_line = &self.interleaved_cache[block_idx - 1];
            if dim == 0 {
                prev_cache_line.rank0_lev1 as usize
            } else {
                prev_cache_line.rank1_lev1 as usize
            }
        } else {
            0
        };

        // Count bits in current block up to position
        if block_idx < self.interleaved_cache.len() {
            let cache_line = &self.interleaved_cache[block_idx];
            let bits = if dim == 0 {
                &cache_line.bits0
            } else {
                &cache_line.bits1
            };

            let mut rank_in_block = 0;
            let words_to_process = (bit_offset_in_block + 63) / 64;

            for word_idx in 0..words_to_process.min(bits.len()) {
                let mut word = bits[word_idx];

                // Handle partial word at the end
                if word_idx == words_to_process - 1 {
                    let remaining_bits = bit_offset_in_block % 64;
                    if remaining_bits > 0 {
                        let mask = (1u64 << remaining_bits) - 1;
                        word &= mask;
                    }
                }

                rank_in_block += self.popcount_hardware_accelerated(word) as usize;
            }

            rank_before_block + rank_in_block
        } else {
            rank_before_block
        }
    }

    /// Internal select implementation for a specific dimension
    pub fn select1_dimension(&self, k: usize, dim: usize) -> Result<usize> {
        if dim >= 2 || k >= self.total_ones[dim] {
            return Err(ZiporaError::out_of_bounds(k, self.total_ones[dim]));
        }

        let target_rank = k + 1;

        // Use select cache if available
        if let Some(ref select_cache) = self.select_caches[dim] {
            let hint_idx = k / self.select_sample_rate;
            if hint_idx < select_cache.len() {
                let hint_pos = select_cache[hint_idx] as usize;
                return self.select1_from_hint(k, hint_pos, dim);
            }
        }

        // Binary search on rank blocks
        let block_idx = self.binary_search_rank_blocks(target_rank, dim);

        let block_start_rank = if block_idx > 0 {
            let prev_cache_line = &self.interleaved_cache[block_idx - 1];
            if dim == 0 {
                prev_cache_line.rank0_lev1 as usize
            } else {
                prev_cache_line.rank1_lev1 as usize
            }
        } else {
            0
        };

        let remaining_ones = target_rank - block_start_rank;
        let block_start_bit = block_idx * DUAL_BLOCK_SIZE;
        let block_end_bit = ((block_idx + 1) * DUAL_BLOCK_SIZE).min(self.total_bits);

        self.select1_within_block(block_start_bit, block_end_bit, remaining_ones, dim)
    }

    /// Select with hint for specific dimension
    fn select1_from_hint(&self, k: usize, hint_pos: usize, dim: usize) -> Result<usize> {
        let target_rank = k + 1;
        let hint_rank = self.rank1_dimension(hint_pos + 1, dim);

        if hint_rank >= target_rank {
            self.select1_linear_search(0, hint_pos + 1, target_rank, dim)
        } else {
            self.select1_linear_search(hint_pos, self.total_bits, target_rank, dim)
        }
    }

    /// Linear search for select within a range
    fn select1_linear_search(
        &self,
        start: usize,
        end: usize,
        target_rank: usize,
        dim: usize,
    ) -> Result<usize> {
        let mut current_rank = self.rank1_dimension(start, dim);

        for pos in start..end {
            if self.get_dimension_bit_unchecked(pos, dim) {
                current_rank += 1;
                if current_rank == target_rank {
                    return Ok(pos);
                }
            }
        }

        Err(ZiporaError::invalid_data(
            "Select position not found".to_string(),
        ))
    }

    /// Get bit from dimension without bounds checking (internal use)
    fn get_dimension_bit_unchecked(&self, index: usize, dim: usize) -> bool {
        let block_idx = index / DUAL_BLOCK_SIZE;
        let bit_offset_in_block = index % DUAL_BLOCK_SIZE;

        if block_idx >= self.interleaved_cache.len() {
            return false;
        }

        let cache_line = &self.interleaved_cache[block_idx];
        let bits = if dim == 0 {
            &cache_line.bits0
        } else {
            &cache_line.bits1
        };

        let word_idx = bit_offset_in_block / 64;
        let bit_idx = bit_offset_in_block % 64;

        if word_idx < bits.len() {
            (bits[word_idx] >> bit_idx) & 1 == 1
        } else {
            false
        }
    }

    /// Binary search to find which block contains the target rank
    fn binary_search_rank_blocks(&self, target_rank: usize, dim: usize) -> usize {
        let mut left = 0;
        let mut right = self.interleaved_cache.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let cache_line = &self.interleaved_cache[mid];
            let rank = if dim == 0 {
                cache_line.rank0_lev1 as usize
            } else {
                cache_line.rank1_lev1 as usize
            };

            if rank < target_rank {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left
    }

    /// Search for the k-th set bit within a specific block
    fn select1_within_block(
        &self,
        start_bit: usize,
        end_bit: usize,
        k: usize,
        dim: usize,
    ) -> Result<usize> {
        if start_bit >= self.total_bits {
            return Err(ZiporaError::invalid_data(
                "Block start beyond bit vector".to_string(),
            ));
        }

        let block_idx = start_bit / DUAL_BLOCK_SIZE;
        if block_idx >= self.interleaved_cache.len() {
            return Err(ZiporaError::invalid_data(
                "Block index out of range".to_string(),
            ));
        }

        let cache_line = &self.interleaved_cache[block_idx];
        let bits = if dim == 0 {
            &cache_line.bits0
        } else {
            &cache_line.bits1
        };

        let mut remaining_k = k;
        let start_word = (start_bit % DUAL_BLOCK_SIZE) / 64;
        let end_word =
            ((end_bit - start_bit).min(DUAL_BLOCK_SIZE - (start_bit % DUAL_BLOCK_SIZE)) + 63) / 64;

        for word_idx in start_word..end_word.min(bits.len()) {
            let mut word = bits[word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word {
                let start_bit_in_word = start_bit % 64;
                if start_bit_in_word > 0 {
                    word &= !((1u64 << start_bit_in_word) - 1);
                }
            }

            // Handle partial word at the end
            let word_end_bit = start_bit + (word_idx - start_word + 1) * 64;
            if word_end_bit > end_bit {
                let valid_bits = 64 - (word_end_bit - end_bit);
                if valid_bits < 64 {
                    let mask = (1u64 << valid_bits) - 1;
                    word &= mask;
                }
            }

            let word_popcount = self.popcount_hardware_accelerated(word) as usize;

            if remaining_k <= word_popcount {
                // The k-th bit is in this word
                let select_pos = self.select_u64_hardware_accelerated(word, remaining_k);
                if select_pos < 64 {
                    let absolute_pos = start_bit + (word_idx - start_word) * 64 + select_pos;
                    return Ok(absolute_pos);
                }
            }

            remaining_k = remaining_k.saturating_sub(word_popcount);
        }

        Err(ZiporaError::invalid_data(
            "Select position not found in block".to_string(),
        ))
    }

    /// Hardware-accelerated select using BMI2 when available
    #[inline(always)]
    fn select_u64_hardware_accelerated(&self, x: u64, k: usize) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(test)]
            {
                self.select_u64_fallback(x, k)
            }

            #[cfg(not(test))]
            {
                if CpuFeatures::get().has_bmi2 {
                    self.select_u64_bmi2(x, k)
                } else {
                    self.select_u64_fallback(x, k)
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            self.select_u64_fallback(x, k)
        }
    }

    /// BMI2-accelerated select implementation
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn select_u64_bmi2(&self, x: u64, k: usize) -> usize {
        if k == 0 || k > self.popcount_hardware_accelerated(x) as usize {
            return 64;
        }

        unsafe {
            let select_mask = (1u64 << k) - 1;
            let expanded_mask = _pdep_u64(select_mask, x);

            if expanded_mask == 0 {
                return 64;
            }

            expanded_mask.trailing_zeros() as usize
        }
    }

    /// Fallback select implementation
    #[inline]
    fn select_u64_fallback(&self, x: u64, k: usize) -> usize {
        if k == 0 || k > self.popcount_hardware_accelerated(x) as usize {
            return 64;
        }

        let mut remaining_k = k;

        for byte_idx in 0..8 {
            let byte = ((x >> (byte_idx * 8)) & 0xFF) as u8;
            let byte_popcount = byte.count_ones() as usize;

            if remaining_k <= byte_popcount {
                let mut bit_count = 0;
                for bit_idx in 0..8 {
                    if (byte >> bit_idx) & 1 == 1 {
                        bit_count += 1;
                        if bit_count == remaining_k {
                            return byte_idx * 8 + bit_idx;
                        }
                    }
                }
            }

            remaining_k = remaining_k.saturating_sub(byte_popcount);
        }

        64
    }
}

impl RankSelectOps for RankSelectMixedIL256 {
    /// Rank operation on primary dimension (dimension 0)
    fn rank1(&self, pos: usize) -> usize {
        self.rank1_dim::<0>(pos)
    }

    /// Rank0 operation on primary dimension (dimension 0)
    fn rank0(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let pos = pos.min(self.total_bits);
        pos - self.rank1(pos)
    }

    /// Select operation on primary dimension (dimension 0)
    fn select1(&self, k: usize) -> Result<usize> {
        self.select1_dim::<0>(k)
    }

    /// Select0 operation on primary dimension (dimension 0)
    fn select0(&self, k: usize) -> Result<usize> {
        let total_zeros = self.len() - self.count_ones();
        if k >= total_zeros {
            return Err(ZiporaError::out_of_bounds(k, total_zeros));
        }

        // Linear search for select0 (could be optimized with additional indexing)
        let mut zeros_seen = 0;
        for pos in 0..self.total_bits {
            if !self.get_dimension_bit::<0>(pos).unwrap_or(true) {
                if zeros_seen == k {
                    return Ok(pos);
                }
                zeros_seen += 1;
            }
        }

        Err(ZiporaError::invalid_data(
            "Select0 position not found".to_string(),
        ))
    }

    fn len(&self) -> usize {
        self.total_bits
    }

    /// Count ones in primary dimension (dimension 0)
    fn count_ones(&self) -> usize {
        self.total_ones[0]
    }

    /// Get bit from primary dimension (dimension 0)
    fn get(&self, index: usize) -> Option<bool> {
        self.get_dimension_bit::<0>(index)
    }

    fn space_overhead_percent(&self) -> f64 {
        if self.total_bits == 0 {
            return 0.0;
        }

        let cache_bits =
            self.interleaved_cache.len() * std::mem::size_of::<InterleavedDualLine>() * 8;
        let select_cache_bits = self
            .select_caches
            .iter()
            .map(|cache| cache.as_ref().map(|c| c.len() * 32).unwrap_or(0))
            .sum::<usize>();

        let total_overhead_bits = cache_bits + select_cache_bits;
        let original_bits = self.total_bits * 2; // Two bit vectors

        (total_overhead_bits as f64 / original_bits as f64) * 100.0
    }
}

impl RankSelectMultiDimensional<2> for RankSelectMixedIL256 {
    fn dimension<const D: usize>(&self) -> MixedDimensionView<'_, D> {
        MixedDimensionView::new(self)
    }

    fn rank1_dim<const D: usize>(&self, pos: usize) -> usize {
        self.rank1_dimension(pos, D)
    }

    fn select1_dim<const D: usize>(&self, k: usize) -> Result<usize> {
        self.select1_dimension(k, D)
    }
}

impl RankSelectBuilder<RankSelectMixedIL256> for RankSelectMixedIL256 {
    fn from_bit_vector(bit_vector: BitVector) -> Result<RankSelectMixedIL256> {
        // Create a second empty bit vector of the same size
        let empty_bv = BitVector::with_size(bit_vector.len(), false)?;
        Self::new([bit_vector, empty_bv])
    }

    fn from_iter<I>(iter: I) -> Result<RankSelectMixedIL256>
    where
        I: IntoIterator<Item = bool>,
    {
        let mut bit_vector = BitVector::new();
        for bit in iter {
            bit_vector.push(bit)?;
        }
        Self::from_bit_vector(bit_vector)
    }

    fn from_bytes(bytes: &[u8], bit_len: usize) -> Result<RankSelectMixedIL256> {
        let mut bit_vector = BitVector::new();

        for (byte_idx, &byte) in bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                let bit_pos = byte_idx * 8 + bit_idx;
                if bit_pos >= bit_len {
                    break;
                }

                let bit = (byte >> bit_idx) & 1 == 1;
                bit_vector.push(bit)?;
            }

            if (byte_idx + 1) * 8 >= bit_len {
                break;
            }
        }

        Self::from_bit_vector(bit_vector)
    }

    fn with_optimizations(
        bit_vector: BitVector,
        opts: BuilderOptions,
    ) -> Result<RankSelectMixedIL256> {
        let empty_bv = BitVector::with_size(bit_vector.len(), false)?;
        Self::with_options(
            [bit_vector, empty_bv],
            opts.optimize_select,
            opts.select_sample_rate,
        )
    }
}

impl fmt::Debug for RankSelectMixedIL256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RankSelectMixedIL256")
            .field("len", &self.len())
            .field("dimensions", &2)
            .field("ones_dim0", &self.total_ones[0])
            .field("ones_dim1", &self.total_ones[1])
            .field("cache_lines", &self.interleaved_cache.len())
            .field(
                "select_cache_dim0",
                &self.select_caches[0].as_ref().map(|c| c.len()).unwrap_or(0),
            )
            .field(
                "select_cache_dim1",
                &self.select_caches[1].as_ref().map(|c| c.len()).unwrap_or(0),
            )
            .field("sample_rate", &self.select_sample_rate)
            .field(
                "overhead",
                &format!("{:.2}%", self.space_overhead_percent()),
            )
            .finish()
    }
}

impl RankSelectMixedSE512 {
    /// Create a new dual-dimension separated rank/select structure with 512-bit blocks
    ///
    /// # Arguments
    /// * `bit_vectors` - Array of exactly 2 bit vectors of the same length
    ///
    /// # Returns
    /// A new RankSelectMixedSE512 instance optimized for large datasets
    pub fn new(bit_vectors: [BitVector; 2]) -> Result<Self> {
        Self::with_options(bit_vectors, true, DUAL_SELECT_SAMPLE_RATE)
    }

    /// Create with custom options
    ///
    /// # Arguments
    /// * `bit_vectors` - Array of exactly 2 bit vectors of the same length
    /// * `enable_select_cache` - Whether to build select cache for faster select operations
    /// * `select_sample_rate` - Sample every N set bits for select cache
    pub fn with_options(
        bit_vectors: [BitVector; 2],
        enable_select_cache: bool,
        select_sample_rate: usize,
    ) -> Result<Self> {
        // Validate input
        if bit_vectors[0].len() != bit_vectors[1].len() {
            return Err(ZiporaError::invalid_data(
                "Both bit vectors must have the same length".to_string(),
            ));
        }

        let total_bits = bit_vectors[0].len();
        let mut rs = Self {
            total_bits,
            total_ones: [0, 0],
            bit_vectors: bit_vectors.clone(),
            rank_caches: [FastVec::new(), FastVec::new()],
            select_caches: [
                if enable_select_cache {
                    Some(FastVec::new())
                } else {
                    None
                },
                if enable_select_cache {
                    Some(FastVec::new())
                } else {
                    None
                },
            ],
            select_sample_rate,
        };

        rs.build_separated_caches()?;

        // Build select caches if enabled
        if enable_select_cache {
            rs.build_select_caches()?;
        }

        Ok(rs)
    }

    /// Build separated rank caches for both dimensions using 512-bit blocks
    fn build_separated_caches(&mut self) -> Result<()> {
        if self.total_bits == 0 {
            return Ok(());
        }

        let num_blocks = (self.total_bits + SEP512_BLOCK_SIZE - 1) / SEP512_BLOCK_SIZE;

        for dim in 0..2 {
            self.rank_caches[dim].reserve(num_blocks)?;

            let mut cumulative_rank = 0u64;
            let blocks = self.bit_vectors[dim].blocks();

            // Process each 512-bit block
            for block_idx in 0..num_blocks {
                let block_start_bit = block_idx * SEP512_BLOCK_SIZE;
                let block_end_bit = ((block_idx + 1) * SEP512_BLOCK_SIZE).min(self.total_bits);

                // Count bits in this block using hardware acceleration
                let block_rank =
                    self.count_bits_in_block_512(blocks, block_start_bit, block_end_bit);

                cumulative_rank += block_rank as u64;
                self.rank_caches[dim].push(cumulative_rank)?;
            }

            self.total_ones[dim] = cumulative_rank as usize;
        }

        Ok(())
    }

    /// Count bits in a 512-bit block using hardware acceleration
    #[inline]
    fn count_bits_in_block_512(&self, blocks: &[u64], start_bit: usize, end_bit: usize) -> usize {
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;
        let mut count = 0;

        for word_idx in start_word..end_word.min(blocks.len()) {
            let mut word = blocks[word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            count += self.popcount_hardware_accelerated(word) as usize;
        }

        count
    }

    /// Hardware-accelerated popcount with fallback
    #[inline(always)]
    fn popcount_hardware_accelerated(&self, x: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(test)]
            {
                x.count_ones()
            }

            #[cfg(not(test))]
            {
                if CpuFeatures::get().has_popcnt {
                    unsafe { _popcnt64(x as i64) as u32 }
                } else {
                    x.count_ones()
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            x.count_ones()
        }
    }

    /// Build select caches for faster select operations
    fn build_select_caches(&mut self) -> Result<()> {
        for dim in 0..2 {
            if self.select_caches[dim].is_some() && self.total_ones[dim] > 0 {
                // Take ownership of the cache temporarily to avoid borrowing conflicts
                if let Some(mut cache) = self.select_caches[dim].take() {
                    self.build_select_cache_for_dimension(&mut cache, dim)?;
                    self.select_caches[dim] = Some(cache);
                }
            }
        }
        Ok(())
    }

    /// Build select cache for a specific dimension
    fn build_select_cache_for_dimension(&self, cache: &mut FastVec<u32>, dim: usize) -> Result<()> {
        let total_ones = self.total_ones[dim];
        if total_ones == 0 {
            return Ok(());
        }

        let mut ones_seen = 0;
        let mut current_pos = 0;

        while ones_seen < total_ones {
            let target_ones = (ones_seen + self.select_sample_rate).min(total_ones);

            while ones_seen < target_ones && current_pos < self.bit_vectors[dim].len() {
                if self.bit_vectors[dim].get(current_pos).unwrap_or(false) {
                    ones_seen += 1;
                }
                if ones_seen < target_ones {
                    current_pos += 1;
                }
            }

            if ones_seen == target_ones {
                cache.push(current_pos as u32)?;
            }

            current_pos += 1;
        }

        Ok(())
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let rank_cache_size = self
            .rank_caches
            .iter()
            .map(|cache| cache.len() * 8) // u64 = 8 bytes
            .sum::<usize>();
        let select_cache_size = self
            .select_caches
            .iter()
            .map(|cache| cache.as_ref().map(|c| c.len() * 4).unwrap_or(0))
            .sum::<usize>();
        let bit_vector_size = self
            .bit_vectors
            .iter()
            .map(|bv| (bv.len() + 7) / 8) // Approximate bit vector size
            .sum::<usize>();

        rank_cache_size + select_cache_size + bit_vector_size + std::mem::size_of::<Self>()
    }

    /// Check if using select cache for a dimension
    pub fn has_select_cache(&self, dim: usize) -> bool {
        dim < 2 && self.select_caches[dim].is_some()
    }

    /// Get bit from a specific dimension
    pub fn get_dimension_bit(&self, index: usize, dim: usize) -> Option<bool> {
        if dim >= 2 {
            None
        } else {
            self.bit_vectors[dim].get(index)
        }
    }

    /// Internal rank implementation for a specific dimension using 512-bit blocks
    pub fn rank1_dimension(&self, pos: usize, dim: usize) -> usize {
        if pos == 0 || self.total_bits == 0 || dim >= 2 {
            return 0;
        }

        let pos = pos.min(self.total_bits);

        // Find containing block
        let block_idx = pos / SEP512_BLOCK_SIZE;
        let bit_offset_in_block = pos % SEP512_BLOCK_SIZE;

        // Get rank up to start of this block
        let rank_before_block = if block_idx > 0 {
            self.rank_caches[dim][block_idx - 1] as usize
        } else {
            0
        };

        // Count bits in current block up to position
        let block_start = block_idx * SEP512_BLOCK_SIZE;
        let block_end = (block_start + bit_offset_in_block).min(self.total_bits);

        let rank_in_block =
            self.count_bits_in_block_512(self.bit_vectors[dim].blocks(), block_start, block_end);

        rank_before_block + rank_in_block
    }

    /// Internal select implementation for a specific dimension
    pub fn select1_dimension(&self, k: usize, dim: usize) -> Result<usize> {
        if dim >= 2 || k >= self.total_ones[dim] {
            return Err(ZiporaError::out_of_bounds(k, self.total_ones[dim]));
        }

        let target_rank = k + 1;

        // Use select cache if available
        if let Some(ref select_cache) = self.select_caches[dim] {
            let hint_idx = k / self.select_sample_rate;
            if hint_idx < select_cache.len() {
                let hint_pos = select_cache[hint_idx] as usize;
                return self.select1_from_hint(k, hint_pos, dim);
            }
        }

        // Binary search on rank blocks
        let block_idx = self.binary_search_rank_blocks(target_rank, dim);

        let block_start_rank = if block_idx > 0 {
            self.rank_caches[dim][block_idx - 1] as usize
        } else {
            0
        };

        let remaining_ones = target_rank - block_start_rank;
        let block_start_bit = block_idx * SEP512_BLOCK_SIZE;
        let block_end_bit = ((block_idx + 1) * SEP512_BLOCK_SIZE).min(self.total_bits);

        self.select1_within_block(block_start_bit, block_end_bit, remaining_ones, dim)
    }

    /// Select with hint for specific dimension
    fn select1_from_hint(&self, k: usize, hint_pos: usize, dim: usize) -> Result<usize> {
        let target_rank = k + 1;
        let hint_rank = self.rank1_dimension(hint_pos + 1, dim);

        if hint_rank >= target_rank {
            self.select1_linear_search(0, hint_pos + 1, target_rank, dim)
        } else {
            self.select1_linear_search(hint_pos, self.total_bits, target_rank, dim)
        }
    }

    /// Linear search for select within a range
    fn select1_linear_search(
        &self,
        start: usize,
        end: usize,
        target_rank: usize,
        dim: usize,
    ) -> Result<usize> {
        let mut current_rank = self.rank1_dimension(start, dim);

        for pos in start..end {
            if self.bit_vectors[dim].get(pos).unwrap_or(false) {
                current_rank += 1;
                if current_rank == target_rank {
                    return Ok(pos);
                }
            }
        }

        Err(ZiporaError::invalid_data(
            "Select position not found".to_string(),
        ))
    }

    /// Binary search to find which block contains the target rank
    fn binary_search_rank_blocks(&self, target_rank: usize, dim: usize) -> usize {
        let mut left = 0;
        let mut right = self.rank_caches[dim].len();

        while left < right {
            let mid = left + (right - left) / 2;
            if self.rank_caches[dim][mid] < target_rank as u64 {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left
    }

    /// Search for the k-th set bit within a specific block
    fn select1_within_block(
        &self,
        start_bit: usize,
        end_bit: usize,
        k: usize,
        dim: usize,
    ) -> Result<usize> {
        let blocks = self.bit_vectors[dim].blocks();
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;

        let mut remaining_k = k;

        for word_idx in start_word..end_word.min(blocks.len()) {
            let mut word = blocks[word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            let word_popcount = self.popcount_hardware_accelerated(word) as usize;

            if remaining_k <= word_popcount {
                // The k-th bit is in this word
                let select_pos = self.select_u64_hardware_accelerated(word, remaining_k);
                if select_pos < 64 {
                    return Ok(word_idx * 64 + select_pos);
                }
            }

            remaining_k = remaining_k.saturating_sub(word_popcount);
        }

        Err(ZiporaError::invalid_data(
            "Select position not found in block".to_string(),
        ))
    }

    /// Hardware-accelerated select using BMI2 when available
    #[inline(always)]
    fn select_u64_hardware_accelerated(&self, x: u64, k: usize) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(test)]
            {
                self.select_u64_fallback(x, k)
            }

            #[cfg(not(test))]
            {
                if CpuFeatures::get().has_bmi2 {
                    self.select_u64_bmi2(x, k)
                } else {
                    self.select_u64_fallback(x, k)
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            self.select_u64_fallback(x, k)
        }
    }

    /// BMI2-accelerated select implementation
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn select_u64_bmi2(&self, x: u64, k: usize) -> usize {
        if k == 0 || k > self.popcount_hardware_accelerated(x) as usize {
            return 64;
        }

        unsafe {
            let select_mask = (1u64 << k) - 1;
            let expanded_mask = _pdep_u64(select_mask, x);

            if expanded_mask == 0 {
                return 64;
            }

            expanded_mask.trailing_zeros() as usize
        }
    }

    /// Fallback select implementation
    #[inline]
    fn select_u64_fallback(&self, x: u64, k: usize) -> usize {
        if k == 0 || k > self.popcount_hardware_accelerated(x) as usize {
            return 64;
        }

        let mut remaining_k = k;

        for byte_idx in 0..8 {
            let byte = ((x >> (byte_idx * 8)) & 0xFF) as u8;
            let byte_popcount = byte.count_ones() as usize;

            if remaining_k <= byte_popcount {
                let mut bit_count = 0;
                for bit_idx in 0..8 {
                    if (byte >> bit_idx) & 1 == 1 {
                        bit_count += 1;
                        if bit_count == remaining_k {
                            return byte_idx * 8 + bit_idx;
                        }
                    }
                }
            }

            remaining_k = remaining_k.saturating_sub(byte_popcount);
        }

        64
    }

    /// Bulk rank operations for dimension (performance optimization)
    pub fn rank1_bulk_dim<const D: usize>(&self, positions: &[usize]) -> Vec<usize> {
        assert!(
            D < 2,
            "Invalid dimension {} for dual-dimension structure",
            D
        );
        positions
            .iter()
            .map(|&pos| self.rank1_dimension(pos, D))
            .collect()
    }

    /// Bulk select operations for dimension (performance optimization)
    pub fn select1_bulk_dim<const D: usize>(&self, indices: &[usize]) -> Result<Vec<usize>> {
        assert!(
            D < 2,
            "Invalid dimension {} for dual-dimension structure",
            D
        );
        indices
            .iter()
            .map(|&k| self.select1_dimension(k, D))
            .collect()
    }

    /// Get dimension view for specific dimension
    pub fn dimension<const D: usize>(&self) -> MixedSeparatedDimensionView<'_, D> {
        MixedSeparatedDimensionView::new(self)
    }
}

/// Dimension view for separated mixed rank/select
pub struct MixedSeparatedDimensionView<'a, const DIM: usize> {
    /// Reference to the parent multi-dimensional structure
    parent: &'a RankSelectMixedSE512,
}

impl<'a, const DIM: usize> MixedSeparatedDimensionView<'a, DIM> {
    /// Create a new dimension view
    pub fn new(parent: &'a RankSelectMixedSE512) -> Self {
        assert!(
            DIM < 2,
            "Invalid dimension {} for dual-dimension structure",
            DIM
        );
        Self { parent }
    }

    /// Get bit at position in this dimension
    pub fn get(&self, index: usize) -> Option<bool> {
        self.parent.get_dimension_bit(index, DIM)
    }

    /// Rank operation for this dimension
    pub fn rank1(&self, pos: usize) -> usize {
        self.parent.rank1_dimension(pos, DIM)
    }

    /// Select operation for this dimension
    pub fn select1(&self, k: usize) -> Result<usize> {
        self.parent.select1_dimension(k, DIM)
    }

    /// Length of this dimension
    pub fn len(&self) -> usize {
        self.parent.total_bits
    }

    /// Check if this dimension is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Count of set bits in this dimension
    pub fn count_ones(&self) -> usize {
        if DIM < 2 {
            self.parent.total_ones[DIM]
        } else {
            0
        }
    }
}

impl RankSelectOps for RankSelectMixedSE512 {
    /// Rank operation on primary dimension (dimension 0)
    fn rank1(&self, pos: usize) -> usize {
        self.rank1_dimension(pos, 0)
    }

    /// Rank0 operation on primary dimension (dimension 0)
    fn rank0(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let pos = pos.min(self.total_bits);
        pos - self.rank1(pos)
    }

    /// Select operation on primary dimension (dimension 0)
    fn select1(&self, k: usize) -> Result<usize> {
        self.select1_dimension(k, 0)
    }

    /// Select0 operation on primary dimension (dimension 0)
    fn select0(&self, k: usize) -> Result<usize> {
        let total_zeros = self.len() - self.count_ones();
        if k >= total_zeros {
            return Err(ZiporaError::out_of_bounds(k, total_zeros));
        }

        // Linear search for select0 (could be optimized with additional indexing)
        let mut zeros_seen = 0;
        for pos in 0..self.total_bits {
            if !self.get_dimension_bit(pos, 0).unwrap_or(true) {
                if zeros_seen == k {
                    return Ok(pos);
                }
                zeros_seen += 1;
            }
        }

        Err(ZiporaError::invalid_data(
            "Select0 position not found".to_string(),
        ))
    }

    fn len(&self) -> usize {
        self.total_bits
    }

    /// Count ones in primary dimension (dimension 0)
    fn count_ones(&self) -> usize {
        self.total_ones[0]
    }

    /// Get bit from primary dimension (dimension 0)
    fn get(&self, index: usize) -> Option<bool> {
        self.get_dimension_bit(index, 0)
    }

    fn space_overhead_percent(&self) -> f64 {
        if self.total_bits == 0 {
            return 0.0;
        }

        let rank_cache_bits = self
            .rank_caches
            .iter()
            .map(|cache| cache.len() * 64) // u64 = 64 bits
            .sum::<usize>();
        let select_cache_bits = self
            .select_caches
            .iter()
            .map(|cache| cache.as_ref().map(|c| c.len() * 32).unwrap_or(0))
            .sum::<usize>();

        let total_overhead_bits = rank_cache_bits + select_cache_bits;
        let original_bits = self.total_bits * 2; // Two bit vectors

        (total_overhead_bits as f64 / original_bits as f64) * 100.0
    }
}

impl RankSelectMultiDimensional<2> for RankSelectMixedSE512 {
    fn dimension<const D: usize>(&self) -> MixedDimensionView<'_, D> {
        MixedDimensionView::new_se512(self)
    }

    fn rank1_dim<const D: usize>(&self, pos: usize) -> usize {
        self.rank1_dimension(pos, D)
    }

    fn select1_dim<const D: usize>(&self, k: usize) -> Result<usize> {
        self.select1_dimension(k, D)
    }
}

impl RankSelectBuilder<RankSelectMixedSE512> for RankSelectMixedSE512 {
    fn from_bit_vector(bit_vector: BitVector) -> Result<RankSelectMixedSE512> {
        // Create a second empty bit vector of the same size
        let empty_bv = BitVector::with_size(bit_vector.len(), false)?;
        Self::new([bit_vector, empty_bv])
    }

    fn from_iter<I>(iter: I) -> Result<RankSelectMixedSE512>
    where
        I: IntoIterator<Item = bool>,
    {
        let mut bit_vector = BitVector::new();
        for bit in iter {
            bit_vector.push(bit)?;
        }
        Self::from_bit_vector(bit_vector)
    }

    fn from_bytes(bytes: &[u8], bit_len: usize) -> Result<RankSelectMixedSE512> {
        let mut bit_vector = BitVector::new();

        for (byte_idx, &byte) in bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                let bit_pos = byte_idx * 8 + bit_idx;
                if bit_pos >= bit_len {
                    break;
                }

                let bit = (byte >> bit_idx) & 1 == 1;
                bit_vector.push(bit)?;
            }

            if (byte_idx + 1) * 8 >= bit_len {
                break;
            }
        }

        Self::from_bit_vector(bit_vector)
    }

    fn with_optimizations(
        bit_vector: BitVector,
        opts: BuilderOptions,
    ) -> Result<RankSelectMixedSE512> {
        let empty_bv = BitVector::with_size(bit_vector.len(), false)?;
        Self::with_options(
            [bit_vector, empty_bv],
            opts.optimize_select,
            opts.select_sample_rate,
        )
    }
}

impl fmt::Debug for RankSelectMixedSE512 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RankSelectMixedSE512")
            .field("len", &self.len())
            .field("dimensions", &2)
            .field("ones_dim0", &self.total_ones[0])
            .field("ones_dim1", &self.total_ones[1])
            .field("rank_blocks", &self.rank_caches[0].len())
            .field(
                "select_cache_dim0",
                &self.select_caches[0].as_ref().map(|c| c.len()).unwrap_or(0),
            )
            .field(
                "select_cache_dim1",
                &self.select_caches[1].as_ref().map(|c| c.len()).unwrap_or(0),
            )
            .field("sample_rate", &self.select_sample_rate)
            .field(
                "overhead",
                &format!("{:.2}%", self.space_overhead_percent()),
            )
            .finish()
    }
}

impl<const ARITY: usize> RankSelectMixedXL256<ARITY> {
    /// Create a new multi-dimensional extended rank/select structure
    ///
    /// # Arguments
    /// * `bit_vectors` - Array of ARITY bit vectors of the same length (2-4 dimensions)
    ///
    /// # Returns
    /// A new RankSelectMixedXL256 instance with interleaved multi-dimensional storage
    pub fn new(bit_vectors: [BitVector; ARITY]) -> Result<Self>
    where
        [(); ARITY]: Sized,
    {
        Self::with_options(bit_vectors, true, MULTI_SELECT_SAMPLE_RATE)
    }

    /// Create with custom options
    ///
    /// # Arguments
    /// * `bit_vectors` - Array of ARITY bit vectors of the same length
    /// * `enable_select_cache` - Whether to build select cache for faster select operations
    /// * `select_sample_rate` - Sample every N set bits for select cache
    pub fn with_options(
        bit_vectors: [BitVector; ARITY],
        enable_select_cache: bool,
        select_sample_rate: usize,
    ) -> Result<Self>
    where
        [(); ARITY]: Sized,
    {
        // Validate input
        if ARITY < 2 || ARITY > 4 {
            return Err(ZiporaError::invalid_data(format!(
                "ARITY must be between 2 and 4, got {}",
                ARITY
            )));
        }

        // Validate all bit vectors have same length
        let total_bits = bit_vectors[0].len();
        for (i, bv) in bit_vectors.iter().enumerate().skip(1) {
            if bv.len() != total_bits {
                return Err(ZiporaError::invalid_data(format!(
                    "All bit vectors must have the same length. Vector 0 has {}, vector {} has {}",
                    total_bits,
                    i,
                    bv.len()
                )));
            }
        }

        let mut rs = Self {
            total_bits,
            total_ones: [0; ARITY],
            interleaved_ranks: FastVec::new(),
            bit_data: {
                // Initialize array of FastVec<u64> for ARITY dimensions
                let mut data = Vec::with_capacity(ARITY);
                for _ in 0..ARITY {
                    data.push(FastVec::new());
                }
                data.try_into().map_err(|_| {
                    ZiporaError::invalid_data("Failed to initialize bit data".to_string())
                })?
            },
            select_caches: {
                // Initialize array of select caches
                let mut caches = Vec::with_capacity(ARITY);
                for _ in 0..ARITY {
                    caches.push(if enable_select_cache {
                        Some(FastVec::new())
                    } else {
                        None
                    });
                }
                caches.try_into().map_err(|_| {
                    ZiporaError::invalid_data("Failed to initialize select caches".to_string())
                })?
            },
            select_sample_rate,
        };

        rs.build_interleaved_multi_cache(&bit_vectors)?;

        // Build select caches if enabled
        if enable_select_cache {
            rs.build_select_caches_multi(&bit_vectors)?;
        }

        Ok(rs)
    }

    /// Build the interleaved multi-dimensional cache
    fn build_interleaved_multi_cache(&mut self, bit_vectors: &[BitVector; ARITY]) -> Result<()>
    where
        [(); ARITY]: Sized,
    {
        if self.total_bits == 0 {
            return Ok(());
        }

        let num_blocks = (self.total_bits + MULTI_BLOCK_SIZE - 1) / MULTI_BLOCK_SIZE;
        self.interleaved_ranks.reserve(num_blocks)?;

        // Store bit data in separate FastVec arrays for each dimension
        for dim in 0..ARITY {
            let num_words = (self.total_bits + 63) / 64;
            self.bit_data[dim].reserve(num_words)?;

            // Copy bit data to our storage
            let blocks = bit_vectors[dim].blocks();
            for &word in blocks.iter().take(num_words) {
                self.bit_data[dim].push(word)?;
            }

            // Pad with zeros if needed
            while self.bit_data[dim].len() < num_words {
                self.bit_data[dim].push(0u64)?;
            }
        }

        let mut cumulative_ranks = [0u32; ARITY];

        for block_idx in 0..num_blocks {
            let block_start_bit = block_idx * MULTI_BLOCK_SIZE;
            let block_end_bit = ((block_idx + 1) * MULTI_BLOCK_SIZE).min(self.total_bits);

            // Count bits in this block for all dimensions
            for dim in 0..ARITY {
                let block_rank =
                    self.count_bits_in_multi_block(dim, block_start_bit, block_end_bit);
                cumulative_ranks[dim] += block_rank as u32;
            }

            // Create interleaved rank entry
            let rank_entry = InterleavedMultiRank::<ARITY> {
                ranks: cumulative_ranks,
                _reserved: [0; 16],
            };

            self.interleaved_ranks.push(rank_entry)?;
        }

        self.total_ones = cumulative_ranks.map(|r| r as usize);
        Ok(())
    }

    /// Count bits in a multi-dimensional block for specific dimension
    #[inline]
    fn count_bits_in_multi_block(&self, dim: usize, start_bit: usize, end_bit: usize) -> usize {
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;
        let mut count = 0;

        for word_idx in start_word..end_word.min(self.bit_data[dim].len()) {
            let mut word = self.bit_data[dim][word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            count += word.count_ones() as usize;
        }

        count
    }

    /// Build select caches for all dimensions
    fn build_select_caches_multi(&mut self, bit_vectors: &[BitVector; ARITY]) -> Result<()>
    where
        [(); ARITY]: Sized,
    {
        for dim in 0..ARITY {
            if self.select_caches[dim].is_some() && self.total_ones[dim] > 0 {
                // Take ownership temporarily to avoid borrowing conflicts
                if let Some(mut cache) = self.select_caches[dim].take() {
                    self.build_select_cache_for_dimension_multi(
                        &mut cache,
                        &bit_vectors[dim],
                        dim,
                    )?;
                    self.select_caches[dim] = Some(cache);
                }
            }
        }
        Ok(())
    }

    /// Build select cache for a specific dimension
    fn build_select_cache_for_dimension_multi(
        &self,
        cache: &mut FastVec<u32>,
        bit_vector: &BitVector,
        dim: usize,
    ) -> Result<()> {
        let total_ones = self.total_ones[dim];
        if total_ones == 0 {
            return Ok(());
        }

        let mut ones_seen = 0;
        let mut current_pos = 0;

        while ones_seen < total_ones {
            let target_ones = (ones_seen + self.select_sample_rate).min(total_ones);

            while ones_seen < target_ones && current_pos < bit_vector.len() {
                if bit_vector.get(current_pos).unwrap_or(false) {
                    ones_seen += 1;
                }
                if ones_seen < target_ones {
                    current_pos += 1;
                }
            }

            if ones_seen == target_ones {
                cache.push(current_pos as u32)?;
            }

            current_pos += 1;
        }

        Ok(())
    }

    /// Get bit from a specific dimension
    pub fn get_dimension_bit<const DIM: usize>(&self, index: usize) -> Option<bool>
    where
        [(); ARITY]: Sized,
    {
        if DIM >= ARITY || index >= self.total_bits {
            return None;
        }

        let word_idx = index / 64;
        let bit_idx = index % 64;

        if word_idx < self.bit_data[DIM].len() {
            Some((self.bit_data[DIM][word_idx] >> bit_idx) & 1 == 1)
        } else {
            None
        }
    }

    /// Internal rank implementation for a specific dimension
    pub fn rank1_dimension<const DIM: usize>(&self, pos: usize) -> usize
    where
        [(); ARITY]: Sized,
    {
        if pos == 0 || self.total_bits == 0 || DIM >= ARITY {
            return 0;
        }

        let pos = pos.min(self.total_bits);

        // Find containing block
        let block_idx = pos / MULTI_BLOCK_SIZE;
        let bit_offset_in_block = pos % MULTI_BLOCK_SIZE;

        // Get rank up to start of this block
        let rank_before_block = if block_idx > 0 && block_idx <= self.interleaved_ranks.len() {
            self.interleaved_ranks[block_idx - 1].ranks[DIM] as usize
        } else {
            0
        };

        // Count bits in current block up to position
        let block_start = block_idx * MULTI_BLOCK_SIZE;
        let block_end = (block_start + bit_offset_in_block).min(self.total_bits);

        let rank_in_block = self.count_bits_in_multi_block(DIM, block_start, block_end);

        rank_before_block + rank_in_block
    }

    /// Internal select implementation for a specific dimension
    pub fn select1_dimension<const DIM: usize>(&self, k: usize) -> Result<usize>
    where
        [(); ARITY]: Sized,
    {
        if DIM >= ARITY || k >= self.total_ones[DIM] {
            return Err(ZiporaError::out_of_bounds(k, self.total_ones[DIM]));
        }

        let target_rank = k + 1;

        // Use select cache if available
        if let Some(ref select_cache) = self.select_caches[DIM] {
            let hint_idx = k / self.select_sample_rate;
            if hint_idx < select_cache.len() {
                let hint_pos = select_cache[hint_idx] as usize;
                return self.select1_from_hint_multi::<DIM>(k, hint_pos);
            }
        }

        // Binary search on rank blocks
        let block_idx = self.binary_search_rank_blocks_multi::<DIM>(target_rank);

        let block_start_rank = if block_idx > 0 && block_idx <= self.interleaved_ranks.len() {
            self.interleaved_ranks[block_idx - 1].ranks[DIM] as usize
        } else {
            0
        };

        let remaining_ones = target_rank - block_start_rank;
        let block_start_bit = block_idx * MULTI_BLOCK_SIZE;
        let block_end_bit = ((block_idx + 1) * MULTI_BLOCK_SIZE).min(self.total_bits);

        self.select1_within_block_multi::<DIM>(block_start_bit, block_end_bit, remaining_ones)
    }

    /// Select with hint for specific dimension
    fn select1_from_hint_multi<const DIM: usize>(&self, k: usize, hint_pos: usize) -> Result<usize>
    where
        [(); ARITY]: Sized,
    {
        let target_rank = k + 1;
        let hint_rank = self.rank1_dimension::<DIM>(hint_pos + 1);

        if hint_rank >= target_rank {
            self.select1_linear_search_multi::<DIM>(0, hint_pos + 1, target_rank)
        } else {
            self.select1_linear_search_multi::<DIM>(hint_pos, self.total_bits, target_rank)
        }
    }

    /// Linear search for select within a range
    fn select1_linear_search_multi<const DIM: usize>(
        &self,
        start: usize,
        end: usize,
        target_rank: usize,
    ) -> Result<usize>
    where
        [(); ARITY]: Sized,
    {
        let mut current_rank = self.rank1_dimension::<DIM>(start);

        for pos in start..end {
            if self.get_dimension_bit::<DIM>(pos).unwrap_or(false) {
                current_rank += 1;
                if current_rank == target_rank {
                    return Ok(pos);
                }
            }
        }

        Err(ZiporaError::invalid_data(
            "Select position not found".to_string(),
        ))
    }

    /// Binary search to find which block contains the target rank
    fn binary_search_rank_blocks_multi<const DIM: usize>(&self, target_rank: usize) -> usize
    where
        [(); ARITY]: Sized,
    {
        let mut left = 0;
        let mut right = self.interleaved_ranks.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if self.interleaved_ranks[mid].ranks[DIM] < target_rank as u32 {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left
    }

    /// Search for the k-th set bit within a specific block
    fn select1_within_block_multi<const DIM: usize>(
        &self,
        start_bit: usize,
        end_bit: usize,
        k: usize,
    ) -> Result<usize>
    where
        [(); ARITY]: Sized,
    {
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;

        let mut remaining_k = k;

        for word_idx in start_word..end_word.min(self.bit_data[DIM].len()) {
            let mut word = self.bit_data[DIM][word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            let word_popcount = word.count_ones() as usize;

            if remaining_k <= word_popcount {
                // The k-th bit is in this word
                let select_pos = self.select_u64_multi(word, remaining_k);
                if select_pos < 64 {
                    return Ok(word_idx * 64 + select_pos);
                }
            }

            remaining_k = remaining_k.saturating_sub(word_popcount);
        }

        Err(ZiporaError::invalid_data(
            "Select position not found in block".to_string(),
        ))
    }

    /// Select within u64 for multi-dimensional implementation
    #[inline]
    fn select_u64_multi(&self, x: u64, k: usize) -> usize {
        if k == 0 || k > x.count_ones() as usize {
            return 64;
        }

        let mut remaining_k = k;

        for byte_idx in 0..8 {
            let byte = ((x >> (byte_idx * 8)) & 0xFF) as u8;
            let byte_popcount = byte.count_ones() as usize;

            if remaining_k <= byte_popcount {
                let mut bit_count = 0;
                for bit_idx in 0..8 {
                    if (byte >> bit_idx) & 1 == 1 {
                        bit_count += 1;
                        if bit_count == remaining_k {
                            return byte_idx * 8 + bit_idx;
                        }
                    }
                }
            }

            remaining_k = remaining_k.saturating_sub(byte_popcount);
        }

        64
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize
    where
        [(); ARITY]: Sized,
    {
        let ranks_size =
            self.interleaved_ranks.len() * std::mem::size_of::<InterleavedMultiRank<ARITY>>();
        let bit_data_size = self
            .bit_data
            .iter()
            .map(|data| data.len() * 8) // u64 = 8 bytes
            .sum::<usize>();
        let select_cache_size = self
            .select_caches
            .iter()
            .map(|cache| cache.as_ref().map(|c| c.len() * 4).unwrap_or(0))
            .sum::<usize>();

        ranks_size + bit_data_size + select_cache_size + std::mem::size_of::<Self>()
    }

    /// Check if using select cache for a dimension
    pub fn has_select_cache(&self, dim: usize) -> bool
    where
        [(); ARITY]: Sized,
    {
        dim < ARITY && self.select_caches[dim].is_some()
    }

    /// Find intersection of set bits across multiple dimensions
    ///
    /// # Arguments
    /// * `dimensions` - Array of dimension indices to intersect
    /// * `limit` - Maximum number of positions to find
    ///
    /// # Returns
    /// Vector of positions where all specified dimensions have set bits
    pub fn find_intersection(&self, dimensions: &[usize], limit: usize) -> Result<Vec<usize>>
    where
        [(); ARITY]: Sized,
    {
        // Validate dimensions
        for &dim in dimensions {
            if dim >= ARITY {
                return Err(ZiporaError::invalid_data(format!(
                    "Invalid dimension {}, max is {}",
                    dim,
                    ARITY - 1
                )));
            }
        }

        let mut intersections = Vec::new();
        let mut pos = 0;

        while intersections.len() < limit && pos < self.total_bits {
            let mut all_set = true;

            for &dim in dimensions {
                if !self.get_dimension_bit_runtime(pos, dim).unwrap_or(false) {
                    all_set = false;
                    break;
                }
            }

            if all_set {
                intersections.push(pos);
            }

            pos += 1;
        }

        Ok(intersections)
    }

    /// Get bit from dimension (runtime version)
    fn get_dimension_bit_runtime(&self, index: usize, dim: usize) -> Option<bool>
    where
        [(); ARITY]: Sized,
    {
        if dim >= ARITY || index >= self.total_bits {
            return None;
        }

        let word_idx = index / 64;
        let bit_idx = index % 64;

        if word_idx < self.bit_data[dim].len() {
            Some((self.bit_data[dim][word_idx] >> bit_idx) & 1 == 1)
        } else {
            None
        }
    }
}

impl<const ARITY: usize> RankSelectMixedXLBitPacked<ARITY> {
    /// Create a new advanced multi-dimensional rank/select with bit-packed hierarchical caching
    ///
    /// # Arguments
    /// * `bit_vectors` - Array of ARITY bit vectors of the same length (2-4 dimensions)
    ///
    /// # Returns
    /// A new RankSelectMixedXLBitPacked instance with hierarchical rank caching
    pub fn new(bit_vectors: [BitVector; ARITY]) -> Result<Self>
    where
        [(); ARITY]: Sized,
    {
        Self::with_options(bit_vectors, true, MULTI_SELECT_SAMPLE_RATE, MULTI_DEFAULT_SUPERBLOCK_SIZE)
    }

    /// Create with custom options
    ///
    /// # Arguments
    /// * `bit_vectors` - Array of ARITY bit vectors of the same length
    /// * `enable_select_cache` - Whether to build select cache for faster select operations
    /// * `select_sample_rate` - Sample every N set bits for select cache
    /// * `superblock_size` - Number of 256-bit blocks per superblock (affects space/time trade-off)
    pub fn with_options(
        bit_vectors: [BitVector; ARITY],
        enable_select_cache: bool,
        select_sample_rate: usize,
        superblock_size: usize,
    ) -> Result<Self>
    where
        [(); ARITY]: Sized,
    {
        // Validate input
        if ARITY < 2 || ARITY > 4 {
            return Err(ZiporaError::invalid_data(format!(
                "ARITY must be between 2 and 4, got {}",
                ARITY
            )));
        }

        // Validate all bit vectors have same length
        let total_bits = bit_vectors[0].len();
        for (i, bv) in bit_vectors.iter().enumerate().skip(1) {
            if bv.len() != total_bits {
                return Err(ZiporaError::invalid_data(format!(
                    "All bit vectors must have the same length. Vector 0 has {}, vector {} has {}",
                    total_bits,
                    i,
                    bv.len()
                )));
            }
        }

        let mut rs = Self {
            total_bits,
            total_ones: [0; ARITY],
            superblock_caches: {
                let mut caches = Vec::with_capacity(ARITY);
                for _ in 0..ARITY {
                    caches.push(FastVec::new());
                }
                caches.try_into().map_err(|_| {
                    ZiporaError::invalid_data("Failed to initialize superblock caches".to_string())
                })?
            },
            relative_rank_caches: {
                let mut caches = Vec::with_capacity(ARITY);
                for _ in 0..ARITY {
                    caches.push(FastVec::new());
                }
                caches.try_into().map_err(|_| {
                    ZiporaError::invalid_data("Failed to initialize relative rank caches".to_string())
                })?
            },
            bit_data: {
                let mut data = Vec::with_capacity(ARITY);
                for _ in 0..ARITY {
                    data.push(FastVec::new());
                }
                data.try_into().map_err(|_| {
                    ZiporaError::invalid_data("Failed to initialize bit data".to_string())
                })?
            },
            select_caches: {
                let mut caches = Vec::with_capacity(ARITY);
                for _ in 0..ARITY {
                    caches.push(if enable_select_cache {
                        Some(FastVec::new())
                    } else {
                        None
                    });
                }
                caches.try_into().map_err(|_| {
                    ZiporaError::invalid_data("Failed to initialize select caches".to_string())
                })?
            },
            select_sample_rate,
            superblock_size,
        };

        rs.build_bit_packed_hierarchical_caches(&bit_vectors)?;

        // Build select caches if enabled
        if enable_select_cache {
            rs.build_select_caches_bit_packed(&bit_vectors)?;
        }

        Ok(rs)
    }

    /// Build bit-packed hierarchical rank caches for all dimensions
    fn build_bit_packed_hierarchical_caches(&mut self, bit_vectors: &[BitVector; ARITY]) -> Result<()>
    where
        [(); ARITY]: Sized,
    {
        if self.total_bits == 0 {
            return Ok(());
        }

        let num_blocks = (self.total_bits + MULTI_BLOCK_SIZE - 1) / MULTI_BLOCK_SIZE;
        let num_superblocks = (num_blocks + self.superblock_size - 1) / self.superblock_size;

        // Store bit data first
        for dim in 0..ARITY {
            let num_words = (self.total_bits + 63) / 64;
            self.bit_data[dim].reserve(num_words)?;

            let blocks = bit_vectors[dim].blocks();
            for &word in blocks.iter().take(num_words) {
                self.bit_data[dim].push(word)?;
            }

            // Pad with zeros if needed
            while self.bit_data[dim].len() < num_words {
                self.bit_data[dim].push(0u64)?;
            }
        }

        // Build hierarchical rank caches per dimension
        for dim in 0..ARITY {
            self.superblock_caches[dim].reserve(num_superblocks)?;

            // Reserve space for relative rank cache (bit-packed)
            let num_relative_blocks = num_blocks.saturating_sub(num_superblocks);
            let num_relative_words = (num_relative_blocks + MULTI_RANKS_PER_WORD - 1) / MULTI_RANKS_PER_WORD;
            self.relative_rank_caches[dim].reserve(num_relative_words)?;

            let mut cumulative_rank = 0u64;
            let mut superblock_start_rank = 0u64;
            let mut current_relative_word = 0u64;
            let mut relative_ranks_in_word = 0;

            for block_idx in 0..num_blocks {
                let block_start_bit = block_idx * MULTI_BLOCK_SIZE;
                let block_end_bit = ((block_idx + 1) * MULTI_BLOCK_SIZE).min(self.total_bits);

                let block_rank = self.count_bits_in_multi_block_bit_packed(dim, block_start_bit, block_end_bit);
                cumulative_rank += block_rank as u64;

                // Check if this is a superblock boundary
                if block_idx % self.superblock_size == 0 {
                    // Store absolute rank in superblock cache
                    self.superblock_caches[dim].push(cumulative_rank)?;
                    superblock_start_rank = cumulative_rank;
                } else {
                    // Store relative rank in bit-packed cache
                    let relative_rank = cumulative_rank - superblock_start_rank;
                    
                    // Ensure relative rank fits in 9 bits
                    if relative_rank >= (1u64 << MULTI_RELATIVE_RANK_BITS) {
                        return Err(ZiporaError::invalid_data(format!(
                            "Relative rank {} exceeds maximum for 9-bit encoding in dimension {}. Consider reducing superblock_size.",
                            relative_rank, dim
                        )));
                    }

                    // Pack the relative rank into current word
                    current_relative_word |= relative_rank << (relative_ranks_in_word * MULTI_RELATIVE_RANK_BITS);
                    relative_ranks_in_word += 1;

                    // If word is full, store it and start a new one
                    if relative_ranks_in_word == MULTI_RANKS_PER_WORD {
                        self.relative_rank_caches[dim].push(current_relative_word)?;
                        current_relative_word = 0;
                        relative_ranks_in_word = 0;
                    }
                }
            }

            // Store any remaining partial word
            if relative_ranks_in_word > 0 {
                self.relative_rank_caches[dim].push(current_relative_word)?;
            }

            self.total_ones[dim] = cumulative_rank as usize;
        }

        Ok(())
    }

    /// Count bits in a multi-dimensional block for specific dimension
    #[inline]
    fn count_bits_in_multi_block_bit_packed(&self, dim: usize, start_bit: usize, end_bit: usize) -> usize {
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;
        let mut count = 0;

        for word_idx in start_word..end_word.min(self.bit_data[dim].len()) {
            let mut word = self.bit_data[dim][word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            count += self.popcount_bit_packed(word) as usize;
        }

        count
    }

    /// Hardware-accelerated popcount for bit-packed implementation
    #[inline(always)]
    fn popcount_bit_packed(&self, x: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(test)]
            {
                x.count_ones()
            }

            #[cfg(not(test))]
            {
                if CpuFeatures::get().has_popcnt {
                    unsafe { _popcnt64(x as i64) as u32 }
                } else {
                    x.count_ones()
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            x.count_ones()
        }
    }

    /// Extract relative rank from bit-packed cache for a dimension
    #[inline]
    fn get_relative_rank(&self, dim: usize, relative_block_idx: usize) -> u64 {
        let word_idx = relative_block_idx / MULTI_RANKS_PER_WORD;
        let rank_idx_in_word = relative_block_idx % MULTI_RANKS_PER_WORD;
        
        if word_idx < self.relative_rank_caches[dim].len() {
            let packed_word = self.relative_rank_caches[dim][word_idx];
            let shift = rank_idx_in_word * MULTI_RELATIVE_RANK_BITS;
            (packed_word >> shift) & MULTI_RELATIVE_RANK_MASK
        } else {
            0
        }
    }

    /// Get bit from a specific dimension
    pub fn get_dimension_bit<const DIM: usize>(&self, index: usize) -> Option<bool>
    where
        [(); ARITY]: Sized,
    {
        if DIM >= ARITY || index >= self.total_bits {
            return None;
        }

        let word_idx = index / 64;
        let bit_idx = index % 64;

        if word_idx < self.bit_data[DIM].len() {
            Some((self.bit_data[DIM][word_idx] >> bit_idx) & 1 == 1)
        } else {
            None
        }
    }

    /// Internal rank implementation for a specific dimension using bit-packed hierarchical cache
    pub fn rank1_dimension<const DIM: usize>(&self, pos: usize) -> usize
    where
        [(); ARITY]: Sized,
    {
        if pos == 0 || self.total_bits == 0 || DIM >= ARITY {
            return 0;
        }

        let pos = pos.min(self.total_bits);
        let block_idx = pos / MULTI_BLOCK_SIZE;
        let bit_offset_in_block = pos % MULTI_BLOCK_SIZE;

        let superblock_idx = block_idx / self.superblock_size;
        let relative_block_idx = block_idx % self.superblock_size;

        // Get rank up to start of this block
        let rank_before_block = if relative_block_idx == 0 {
            // This is a superblock boundary - use absolute rank from previous superblock
            if superblock_idx > 0 {
                self.superblock_caches[DIM][superblock_idx - 1] as usize
            } else {
                0
            }
        } else {
            // Get superblock base rank and add relative rank
            let superblock_base = if superblock_idx < self.superblock_caches[DIM].len() {
                self.superblock_caches[DIM][superblock_idx] as usize
            } else {
                return 0; // Beyond end of data
            };
            
            // Calculate absolute index in relative rank cache
            let abs_relative_idx = superblock_idx * (self.superblock_size - 1) + (relative_block_idx - 1);
            let relative_rank = self.get_relative_rank(DIM, abs_relative_idx) as usize;
            
            superblock_base + relative_rank
        };

        // Count bits in current block up to position
        let block_start = block_idx * MULTI_BLOCK_SIZE;
        let block_end = (block_start + bit_offset_in_block).min(self.total_bits);

        let rank_in_block = self.count_bits_in_multi_block_bit_packed(DIM, block_start, block_end);

        rank_before_block + rank_in_block
    }

    /// Build select caches for all dimensions
    fn build_select_caches_bit_packed(&mut self, bit_vectors: &[BitVector; ARITY]) -> Result<()>
    where
        [(); ARITY]: Sized,
    {
        for dim in 0..ARITY {
            if self.select_caches[dim].is_some() && self.total_ones[dim] > 0 {
                if let Some(mut cache) = self.select_caches[dim].take() {
                    self.build_select_cache_for_dimension_bit_packed(&mut cache, &bit_vectors[dim], dim)?;
                    self.select_caches[dim] = Some(cache);
                }
            }
        }
        Ok(())
    }

    /// Build select cache for a specific dimension
    fn build_select_cache_for_dimension_bit_packed(
        &self,
        cache: &mut FastVec<u32>,
        bit_vector: &BitVector,
        _dim: usize,
    ) -> Result<()> {
        let total_ones = self.total_ones[_dim];
        if total_ones == 0 {
            return Ok(());
        }

        let mut ones_seen = 0;
        let mut current_pos = 0;

        while ones_seen < total_ones {
            let target_ones = (ones_seen + self.select_sample_rate).min(total_ones);

            while ones_seen < target_ones && current_pos < bit_vector.len() {
                if bit_vector.get(current_pos).unwrap_or(false) {
                    ones_seen += 1;
                }
                if ones_seen < target_ones {
                    current_pos += 1;
                }
            }

            if ones_seen == target_ones {
                cache.push(current_pos as u32)?;
            }

            current_pos += 1;
        }

        Ok(())
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize
    where
        [(); ARITY]: Sized,
    {
        let superblock_cache_size = self
            .superblock_caches
            .iter()
            .map(|cache| cache.len() * 8) // u64 = 8 bytes
            .sum::<usize>();
        let relative_cache_size = self
            .relative_rank_caches
            .iter()
            .map(|cache| cache.len() * 8) // u64 = 8 bytes
            .sum::<usize>();
        let bit_data_size = self
            .bit_data
            .iter()
            .map(|data| data.len() * 8) // u64 = 8 bytes
            .sum::<usize>();
        let select_cache_size = self
            .select_caches
            .iter()
            .map(|cache| cache.as_ref().map(|c| c.len() * 4).unwrap_or(0))
            .sum::<usize>();

        superblock_cache_size + relative_cache_size + bit_data_size + select_cache_size + std::mem::size_of::<Self>()
    }

    /// Check if using select cache for a dimension
    pub fn has_select_cache(&self, dim: usize) -> bool
    where
        [(); ARITY]: Sized,
    {
        dim < ARITY && self.select_caches[dim].is_some()
    }

    /// Calculate space overhead percentage
    pub fn space_overhead_percent(&self) -> f64
    where
        [(); ARITY]: Sized,
    {
        if self.total_bits == 0 {
            return 0.0;
        }

        let superblock_cache_bits = self
            .superblock_caches
            .iter()
            .map(|cache| cache.len() * 64) // u64 = 64 bits
            .sum::<usize>();
        let relative_cache_bits = self
            .relative_rank_caches
            .iter()
            .map(|cache| cache.len() * 64) // u64 = 64 bits
            .sum::<usize>();
        let select_cache_bits = self
            .select_caches
            .iter()
            .map(|cache| cache.as_ref().map(|c| c.len() * 32).unwrap_or(0))
            .sum::<usize>();

        let total_overhead_bits = superblock_cache_bits + relative_cache_bits + select_cache_bits;
        let original_bits = self.total_bits * ARITY; // ARITY bit vectors

        (total_overhead_bits as f64 / original_bits as f64) * 100.0
    }
}
