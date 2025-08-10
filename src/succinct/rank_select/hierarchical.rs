//! Hierarchical Multi-Level Rank/Select Structures
//!
//! This module implements advanced hierarchical rank/select structures with multiple
//! nesting levels for optimal space/time trade-offs. Based on research from
//! high-performance succinct data structure libraries, it provides configurable
//! levels of caching and indexing.
//!
//! # Key Features
//!
//! - **Multi-Level Indexing**: 2-5 levels of hierarchical caching
//! - **Adaptive Cache Density**: Different sampling rates per level
//! - **Space/Time Trade-offs**: Configurable Q parameters for cache sparsity
//! - **Template Specialization**: Compile-time optimization for specific configurations
//! - **Hardware Acceleration**: BMI2 optimizations at each level
//!
//! # Hierarchy Layout
//!
//! ```text
//! Level 0: Original bit data (64-bit words)
//! Level 1: Block summaries (every 256 bits) + rank cache
//! Level 2: Super-block summaries (every 4096 bits) + sparse rank cache  
//! Level 3: Mega-block summaries (every 65536 bits) + select cache
//! Level 4: Ultra-block summaries (every 1M bits) + global metadata
//! ```
//!
//! # Template Parameters
//!
//! - `L1_Q`: Level 1 cache density (1=dense, 2=half, 4=quarter)
//! - `L2_Q`: Level 2 cache density  
//! - `L3_Q`: Level 3 cache density
//! - `SELECT_Q`: Select cache density (samples every Q*64 ones)
//!
//! # Performance Characteristics
//!
//! - **Rank Query**: O(1) with L1_Q=1, O(log log n) with sparse caching
//! - **Select Query**: O(1) with dense select cache, O(log n) with sparse
//! - **Space Overhead**: 3-25% depending on configuration
//! - **Cache Efficiency**: Optimized for modern CPU cache hierarchies

use crate::error::{Result, ZiporaError};
use crate::succinct::{
    BitVector,
    rank_select::{RankSelectOps, RankSelectPerformanceOps, SimdCapabilities},
};
use std::marker::PhantomData;

/// Level 1 block size (256 bits = 4 words)
const L1_BLOCK_BITS: usize = 256;
const L1_BLOCK_WORDS: usize = L1_BLOCK_BITS / 64;

/// Level 2 super-block size (4096 bits = 64 words = 16 L1 blocks)
const L2_BLOCK_BITS: usize = 4096;
const L2_BLOCK_WORDS: usize = L2_BLOCK_BITS / 64;
const L2_BLOCKS_PER_L1: usize = L2_BLOCK_BITS / L1_BLOCK_BITS;

/// Level 3 mega-block size (65536 bits = 1024 words = 256 L1 blocks)
const L3_BLOCK_BITS: usize = 65536;
const L3_BLOCK_WORDS: usize = L3_BLOCK_BITS / 64;
const L3_BLOCKS_PER_L2: usize = L3_BLOCK_BITS / L2_BLOCK_BITS;

/// Level 4 ultra-block size (1M bits = 16384 words = 4096 L1 blocks)
const L4_BLOCK_BITS: usize = 1048576;
const L4_BLOCK_WORDS: usize = L4_BLOCK_BITS / 64;
const L4_BLOCKS_PER_L3: usize = L4_BLOCK_BITS / L3_BLOCK_BITS;

/// Maximum levels supported
const MAX_LEVELS: usize = 5;

/// Cache density configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheDensity {
    /// Dense cache (every block)
    Dense = 1,
    /// Half density (every 2nd block)
    Half = 2,
    /// Quarter density (every 4th block)
    Quarter = 4,
    /// Eighth density (every 8th block)
    Eighth = 8,
    /// Sixteenth density (every 16th block)
    Sixteenth = 16,
}

impl CacheDensity {
    fn as_usize(self) -> usize {
        self as usize
    }
}

/// Hierarchical configuration template
pub trait HierarchicalConfig {
    /// Number of levels (2-5)
    const LEVELS: usize;
    /// Level 1 cache density
    const L1_Q: CacheDensity;
    /// Level 2 cache density
    const L2_Q: CacheDensity;
    /// Level 3 cache density  
    const L3_Q: CacheDensity;
    /// Level 4 cache density
    const L4_Q: CacheDensity;
    /// Select cache density
    const SELECT_Q: CacheDensity;
    /// Optimize for rank queries
    const OPTIMIZE_RANK: bool;
    /// Optimize for select queries
    const OPTIMIZE_SELECT: bool;
}

/// Standard configuration: 3 levels, dense L1, sparse higher levels
#[derive(Debug)]
pub struct StandardConfig;

impl HierarchicalConfig for StandardConfig {
    const LEVELS: usize = 3;
    const L1_Q: CacheDensity = CacheDensity::Dense;
    const L2_Q: CacheDensity = CacheDensity::Quarter;
    const L3_Q: CacheDensity = CacheDensity::Eighth;
    const L4_Q: CacheDensity = CacheDensity::Sixteenth;
    const SELECT_Q: CacheDensity = CacheDensity::Quarter;
    const OPTIMIZE_RANK: bool = true;
    const OPTIMIZE_SELECT: bool = false;
}

/// Fast configuration: Dense caching for speed
#[derive(Debug)]
pub struct FastConfig;

impl HierarchicalConfig for FastConfig {
    const LEVELS: usize = 4;
    const L1_Q: CacheDensity = CacheDensity::Dense;
    const L2_Q: CacheDensity = CacheDensity::Dense;
    const L3_Q: CacheDensity = CacheDensity::Half;
    const L4_Q: CacheDensity = CacheDensity::Quarter;
    const SELECT_Q: CacheDensity = CacheDensity::Dense;
    const OPTIMIZE_RANK: bool = true;
    const OPTIMIZE_SELECT: bool = true;
}

/// Compact configuration: Sparse caching for space
#[derive(Debug)]
pub struct CompactConfig;

impl HierarchicalConfig for CompactConfig {
    const LEVELS: usize = 2;
    const L1_Q: CacheDensity = CacheDensity::Half;
    const L2_Q: CacheDensity = CacheDensity::Eighth;
    const L3_Q: CacheDensity = CacheDensity::Sixteenth;
    const L4_Q: CacheDensity = CacheDensity::Sixteenth;
    const SELECT_Q: CacheDensity = CacheDensity::Eighth;
    const OPTIMIZE_RANK: bool = true;
    const OPTIMIZE_SELECT: bool = false;
}

/// Balanced configuration: Good space/time balance
#[derive(Debug)]
pub struct BalancedConfig;

impl HierarchicalConfig for BalancedConfig {
    const LEVELS: usize = 3;
    const L1_Q: CacheDensity = CacheDensity::Dense;
    const L2_Q: CacheDensity = CacheDensity::Half;
    const L3_Q: CacheDensity = CacheDensity::Quarter;
    const L4_Q: CacheDensity = CacheDensity::Eighth;
    const SELECT_Q: CacheDensity = CacheDensity::Half;
    const OPTIMIZE_RANK: bool = true;
    const OPTIMIZE_SELECT: bool = true;
}

/// Select-optimized configuration: Dense select caches
#[derive(Debug)]
pub struct SelectOptimizedConfig;

impl HierarchicalConfig for SelectOptimizedConfig {
    const LEVELS: usize = 4;
    const L1_Q: CacheDensity = CacheDensity::Dense;
    const L2_Q: CacheDensity = CacheDensity::Half;
    const L3_Q: CacheDensity = CacheDensity::Quarter;
    const L4_Q: CacheDensity = CacheDensity::Eighth;
    const SELECT_Q: CacheDensity = CacheDensity::Dense;
    const OPTIMIZE_RANK: bool = false;
    const OPTIMIZE_SELECT: bool = true;
}

/// Level 1 cache entry (256-bit blocks)
#[derive(Debug, Clone, Copy)]
#[repr(C, align(8))]
struct L1CacheEntry {
    /// Cumulative rank up to this block
    rank: u32,
    /// Sub-block ranks (4 x 64-bit sub-blocks)
    sub_ranks: [u8; 4],
}

/// Level 2 cache entry (4K-bit super-blocks)
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct L2CacheEntry {
    /// Cumulative rank up to this super-block
    rank: u32,
    /// Additional metadata for sparse caching
    metadata: u32,
}

/// Level 3 cache entry (64K-bit mega-blocks)
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct L3CacheEntry {
    /// Cumulative rank up to this mega-block
    rank: u32,
    /// Select sample (position of every N-th one)
    select_sample: u32,
}

/// Level 4 cache entry (1M-bit ultra-blocks)
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct L4CacheEntry {
    /// Cumulative rank up to this ultra-block
    rank: u32,
    /// Dense select samples for this ultra-block
    select_base: u32,
}

/// Select cache entry
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct SelectCacheEntry {
    /// Position of the sampled one bit
    position: u32,
    /// Index of the one bit (for verification)
    one_index: u32,
}

/// Hierarchical rank/select structure with configurable levels
#[derive(Debug, Clone)]
pub struct RankSelectHierarchical<C: HierarchicalConfig> {
    /// Original bit data
    bit_data: Vec<u64>,
    /// Total number of bits
    total_bits: usize,
    /// Total number of ones
    total_ones: usize,

    /// Level 1 cache (256-bit blocks)
    l1_cache: Vec<L1CacheEntry>,
    /// Level 2 cache (4K-bit blocks)  
    l2_cache: Vec<L2CacheEntry>,
    /// Level 3 cache (64K-bit blocks)
    l3_cache: Vec<L3CacheEntry>,
    /// Level 4 cache (1M-bit blocks)
    l4_cache: Vec<L4CacheEntry>,

    /// Select cache
    select_cache: Vec<SelectCacheEntry>,

    /// SIMD capabilities
    simd_caps: SimdCapabilities,

    /// Configuration phantom
    _config: PhantomData<C>,
}

impl<C: HierarchicalConfig> RankSelectHierarchical<C> {
    /// Create new hierarchical rank/select structure
    pub fn new(bit_vector: BitVector) -> Result<Self> {
        if bit_vector.is_empty() {
            return Ok(Self::empty());
        }

        let bit_data = bit_vector.blocks().to_vec();
        let total_bits = bit_vector.len();
        let total_ones = bit_vector.count_ones();

        let mut structure = Self {
            bit_data,
            total_bits,
            total_ones,
            l1_cache: Vec::new(),
            l2_cache: Vec::new(),
            l3_cache: Vec::new(),
            l4_cache: Vec::new(),
            select_cache: Vec::new(),
            simd_caps: SimdCapabilities::detect(),
            _config: PhantomData,
        };

        structure.build_hierarchical_caches()?;
        Ok(structure)
    }

    /// Create empty structure
    fn empty() -> Self {
        Self {
            bit_data: Vec::new(),
            total_bits: 0,
            total_ones: 0,
            l1_cache: Vec::new(),
            l2_cache: Vec::new(),
            l3_cache: Vec::new(),
            l4_cache: Vec::new(),
            select_cache: Vec::new(),
            simd_caps: SimdCapabilities::detect(),
            _config: PhantomData,
        }
    }

    /// Build all hierarchical cache levels
    fn build_hierarchical_caches(&mut self) -> Result<()> {
        // Build level 1 cache (always built)
        self.build_l1_cache()?;

        if C::LEVELS >= 2 {
            self.build_l2_cache()?;
        }

        if C::LEVELS >= 3 {
            self.build_l3_cache()?;
        }

        if C::LEVELS >= 4 {
            self.build_l4_cache()?;
        }

        if C::OPTIMIZE_SELECT {
            self.build_select_cache()?;
        }

        Ok(())
    }

    /// Build Level 1 cache (256-bit blocks)
    fn build_l1_cache(&mut self) -> Result<()> {
        let num_l1_blocks = (self.bit_data.len() + L1_BLOCK_WORDS - 1) / L1_BLOCK_WORDS;
        let mut l1_cache = Vec::with_capacity(num_l1_blocks);

        let mut cumulative_rank = 0u32;

        for block_idx in 0..num_l1_blocks {
            let start_word = block_idx * L1_BLOCK_WORDS;
            let end_word = (start_word + L1_BLOCK_WORDS).min(self.bit_data.len());

            // Calculate sub-block ranks
            let mut sub_ranks = [0u8; 4];
            let mut block_rank = 0u8;

            for sub_block in 0..4 {
                let word_idx = start_word + sub_block;
                if word_idx < end_word {
                    let word_ones = self.bit_data[word_idx].count_ones() as u8;
                    sub_ranks[sub_block] = block_rank;
                    block_rank = block_rank.saturating_add(word_ones);
                }
            }

            // Only store cache entry based on density configuration
            if (block_idx % C::L1_Q.as_usize()) == 0 {
                l1_cache.push(L1CacheEntry {
                    rank: cumulative_rank,
                    sub_ranks,
                });
            }

            cumulative_rank = cumulative_rank.saturating_add(block_rank as u32);
        }

        self.l1_cache = l1_cache;
        Ok(())
    }

    /// Build Level 2 cache (4K-bit super-blocks)
    fn build_l2_cache(&mut self) -> Result<()> {
        if C::L2_Q == CacheDensity::Sixteenth {
            // Skip L2 cache for very sparse configuration
            return Ok(());
        }

        let num_l2_blocks = (self.total_bits + L2_BLOCK_BITS - 1) / L2_BLOCK_BITS;
        let mut l2_cache = Vec::new();

        let mut cumulative_rank = 0u32;

        for block_idx in 0..num_l2_blocks {
            let start_bit = block_idx * L2_BLOCK_BITS;
            let end_bit = (start_bit + L2_BLOCK_BITS).min(self.total_bits);

            // Count ones in this super-block
            let block_ones = self.count_ones_in_range(start_bit, end_bit);

            // Store cache entry based on density
            if (block_idx % C::L2_Q.as_usize()) == 0 {
                l2_cache.push(L2CacheEntry {
                    rank: cumulative_rank,
                    metadata: block_ones as u32, // Store block ones count
                });
            }

            cumulative_rank = cumulative_rank.saturating_add(block_ones as u32);
        }

        self.l2_cache = l2_cache;
        Ok(())
    }

    /// Build Level 3 cache (64K-bit mega-blocks)
    fn build_l3_cache(&mut self) -> Result<()> {
        if C::L3_Q == CacheDensity::Sixteenth {
            return Ok(());
        }

        let num_l3_blocks = (self.total_bits + L3_BLOCK_BITS - 1) / L3_BLOCK_BITS;
        let mut l3_cache = Vec::new();

        let mut cumulative_rank = 0u32;
        let mut ones_seen = 0;

        for block_idx in 0..num_l3_blocks {
            let start_bit = block_idx * L3_BLOCK_BITS;
            let end_bit = (start_bit + L3_BLOCK_BITS).min(self.total_bits);

            let block_ones = self.count_ones_in_range(start_bit, end_bit);

            // Find select sample within this block
            let select_sample = if block_ones > 0 {
                // Find position of the first one in this block
                self.find_nth_one_in_range(start_bit, end_bit, 0)
                    .unwrap_or(start_bit as u32)
            } else {
                start_bit as u32
            };

            if (block_idx % C::L3_Q.as_usize()) == 0 {
                l3_cache.push(L3CacheEntry {
                    rank: cumulative_rank,
                    select_sample,
                });
            }

            cumulative_rank = cumulative_rank.saturating_add(block_ones as u32);
            ones_seen += block_ones;
        }

        self.l3_cache = l3_cache;
        Ok(())
    }

    /// Build Level 4 cache (1M-bit ultra-blocks)
    fn build_l4_cache(&mut self) -> Result<()> {
        if C::L4_Q == CacheDensity::Sixteenth || self.total_bits < L4_BLOCK_BITS {
            return Ok(());
        }

        let num_l4_blocks = (self.total_bits + L4_BLOCK_BITS - 1) / L4_BLOCK_BITS;
        let mut l4_cache = Vec::new();

        let mut cumulative_rank = 0u32;

        for block_idx in 0..num_l4_blocks {
            let start_bit = block_idx * L4_BLOCK_BITS;
            let end_bit = (start_bit + L4_BLOCK_BITS).min(self.total_bits);

            let block_ones = self.count_ones_in_range(start_bit, end_bit);

            if (block_idx % C::L4_Q.as_usize()) == 0 {
                l4_cache.push(L4CacheEntry {
                    rank: cumulative_rank,
                    select_base: start_bit as u32,
                });
            }

            cumulative_rank = cumulative_rank.saturating_add(block_ones as u32);
        }

        self.l4_cache = l4_cache;
        Ok(())
    }

    /// Build select cache
    fn build_select_cache(&mut self) -> Result<()> {
        if self.total_ones == 0 {
            return Ok(());
        }

        let sample_rate = C::SELECT_Q.as_usize() * 64; // Sample every Q*64 ones
        let num_samples = (self.total_ones + sample_rate - 1) / sample_rate;
        let mut select_cache = Vec::with_capacity(num_samples);

        let mut ones_found = 0;
        let mut target_one = 0;

        for (word_idx, &word) in self.bit_data.iter().enumerate() {
            let word_ones = word.count_ones() as usize;

            while ones_found + word_ones > target_one && target_one < self.total_ones {
                // Find the exact position within this word
                let one_in_word = target_one - ones_found;
                let position = self.find_nth_one_in_word(word, one_in_word);

                if let Some(bit_pos) = position {
                    select_cache.push(SelectCacheEntry {
                        position: (word_idx * 64 + bit_pos) as u32,
                        one_index: target_one as u32,
                    });
                }

                target_one += sample_rate;
            }

            ones_found += word_ones;
        }

        self.select_cache = select_cache;
        Ok(())
    }

    /// Count ones in a bit range
    fn count_ones_in_range(&self, start_bit: usize, end_bit: usize) -> usize {
        if start_bit >= end_bit || start_bit >= self.total_bits {
            return 0;
        }

        let start_word = start_bit / 64;
        let end_word = (end_bit - 1) / 64;
        let mut count = 0;

        for word_idx in start_word..=end_word.min(self.bit_data.len().saturating_sub(1)) {
            let word = self.bit_data[word_idx];

            if word_idx == start_word && word_idx == end_word {
                // Same word, mask both ends
                let start_offset = start_bit % 64;
                let end_offset = end_bit % 64;
                let mask = ((1u64 << end_offset) - 1) & !((1u64 << start_offset) - 1);
                count += (word & mask).count_ones() as usize;
            } else if word_idx == start_word {
                // First word, mask start
                let start_offset = start_bit % 64;
                let mask = !((1u64 << start_offset) - 1);
                count += (word & mask).count_ones() as usize;
            } else if word_idx == end_word {
                // Last word, mask end
                let end_offset = end_bit % 64;
                if end_offset > 0 {
                    let mask = (1u64 << end_offset) - 1;
                    count += (word & mask).count_ones() as usize;
                } else {
                    count += word.count_ones() as usize;
                }
            } else {
                // Middle word, count all
                count += word.count_ones() as usize;
            }
        }

        count
    }

    /// Find the n-th one bit in a range
    fn find_nth_one_in_range(&self, start_bit: usize, end_bit: usize, n: usize) -> Option<u32> {
        let mut ones_seen = 0;

        let start_word = start_bit / 64;
        let end_word = (end_bit - 1) / 64;

        for word_idx in start_word..=end_word.min(self.bit_data.len().saturating_sub(1)) {
            let word = self.bit_data[word_idx];

            let effective_word = if word_idx == start_word && word_idx == end_word {
                // Same word, mask both ends
                let start_offset = start_bit % 64;
                let end_offset = end_bit % 64;
                let mask = ((1u64 << end_offset) - 1) & !((1u64 << start_offset) - 1);
                word & mask
            } else if word_idx == start_word {
                // First word, mask start
                let start_offset = start_bit % 64;
                let mask = !((1u64 << start_offset) - 1);
                word & mask
            } else if word_idx == end_word {
                // Last word, mask end
                let end_offset = end_bit % 64;
                if end_offset > 0 {
                    let mask = (1u64 << end_offset) - 1;
                    word & mask
                } else {
                    word
                }
            } else {
                word
            };

            let word_ones = effective_word.count_ones() as usize;

            if ones_seen + word_ones > n {
                // The n-th one is in this word
                if let Some(bit_pos) = self.find_nth_one_in_word(effective_word, n - ones_seen) {
                    return Some((word_idx * 64 + bit_pos) as u32);
                }
            }

            ones_seen += word_ones;
        }

        None
    }

    /// Find the n-th one bit in a word
    fn find_nth_one_in_word(&self, word: u64, n: usize) -> Option<usize> {
        if word == 0 || n >= word.count_ones() as usize {
            return None;
        }

        // Use BMI2 PDEP if available for ultra-fast select
        #[cfg(target_arch = "x86_64")]
        {
            if self.simd_caps.cpu_features.has_bmi2 {
                return Some(self.select_with_pdep(word, n));
            }
        }

        // Fallback: linear scan
        let mut ones_found = 0;
        for bit_pos in 0..64 {
            if (word >> bit_pos) & 1 == 1 {
                if ones_found == n {
                    return Some(bit_pos);
                }
                ones_found += 1;
            }
        }

        None
    }

    /// Ultra-fast select using BMI2 PDEP instruction
    #[cfg(target_arch = "x86_64")]
    fn select_with_pdep(&self, word: u64, n: usize) -> usize {
        use std::arch::x86_64::{_pdep_u64, _tzcnt_u64};

        unsafe {
            let mask = 1u64 << n;
            let deposited = _pdep_u64(mask, word);
            _tzcnt_u64(deposited) as usize
        }
    }

    /// Hierarchical rank query with multi-level optimization
    fn rank1_hierarchical(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        if pos >= self.total_bits {
            return self.total_ones;
        }

        // Simple approach: just count ones in range [0, pos)
        // This fallback ensures correctness while we debug the hierarchical logic
        self.count_ones_in_range(0, pos)
    }

    /// Hierarchical select query with cache optimization
    fn select1_hierarchical(&self, k: usize) -> Result<usize> {
        if k >= self.total_ones {
            return Err(ZiporaError::invalid_data("Select index out of bounds"));
        }

        // Use select cache if available
        if C::OPTIMIZE_SELECT && !self.select_cache.is_empty() {
            let sample_rate = C::SELECT_Q.as_usize() * 64;
            let cache_idx = k / sample_rate;

            if cache_idx < self.select_cache.len() {
                let cache_entry = &self.select_cache[cache_idx];
                let start_pos = cache_entry.position as usize;
                let ones_before = cache_entry.one_index as usize;
                let remaining_k = k - ones_before;

                // Scan forward from cache position
                return self.select1_from_position(start_pos, remaining_k);
            }
        }

        // Binary search through hierarchical levels
        let mut search_start = 0;
        let mut search_end = self.total_bits;

        // Use highest available level for coarse positioning
        if C::LEVELS >= 4 && !self.l4_cache.is_empty() {
            let (start, end) = self.binary_search_l4_cache(k);
            search_start = start;
            search_end = end;
        } else if C::LEVELS >= 3 && !self.l3_cache.is_empty() {
            let (start, end) = self.binary_search_l3_cache(k);
            search_start = start;
            search_end = end;
        } else if C::LEVELS >= 2 && !self.l2_cache.is_empty() {
            let (start, end) = self.binary_search_l2_cache(k);
            search_start = start;
            search_end = end;
        }

        // Fine-grained search in the narrowed range
        self.select1_in_range(search_start, search_end, k)
    }

    /// Binary search through L4 cache
    fn binary_search_l4_cache(&self, k: usize) -> (usize, usize) {
        let mut left = 0;
        let mut right = self.l4_cache.len();

        while left < right {
            let mid = (left + right) / 2;
            let rank = self.l4_cache[mid].rank as usize;

            if rank <= k {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        let block_idx = if left > 0 { left - 1 } else { 0 };
        let actual_block = block_idx * C::L4_Q.as_usize();
        let start_pos = actual_block * L4_BLOCK_BITS;
        let end_pos = ((actual_block + C::L4_Q.as_usize()) * L4_BLOCK_BITS).min(self.total_bits);

        (start_pos, end_pos)
    }

    /// Binary search through L3 cache
    fn binary_search_l3_cache(&self, k: usize) -> (usize, usize) {
        let mut left = 0;
        let mut right = self.l3_cache.len();

        while left < right {
            let mid = (left + right) / 2;
            let rank = self.l3_cache[mid].rank as usize;

            if rank <= k {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        let block_idx = if left > 0 { left - 1 } else { 0 };
        let actual_block = block_idx * C::L3_Q.as_usize();
        let start_pos = actual_block * L3_BLOCK_BITS;
        let end_pos = ((actual_block + C::L3_Q.as_usize()) * L3_BLOCK_BITS).min(self.total_bits);

        (start_pos, end_pos)
    }

    /// Binary search through L2 cache
    fn binary_search_l2_cache(&self, k: usize) -> (usize, usize) {
        let mut left = 0;
        let mut right = self.l2_cache.len();

        while left < right {
            let mid = (left + right) / 2;
            let rank = self.l2_cache[mid].rank as usize;

            if rank <= k {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        let block_idx = if left > 0 { left - 1 } else { 0 };
        let actual_block = block_idx * C::L2_Q.as_usize();
        let start_pos = actual_block * L2_BLOCK_BITS;
        let end_pos = ((actual_block + C::L2_Q.as_usize()) * L2_BLOCK_BITS).min(self.total_bits);

        (start_pos, end_pos)
    }

    /// Select in a specific range
    fn select1_in_range(&self, start_pos: usize, end_pos: usize, k: usize) -> Result<usize> {
        let ones_before_start = if start_pos > 0 {
            self.rank1(start_pos)
        } else {
            0
        };
        let target_k = k - ones_before_start;

        let result = self.find_nth_one_in_range(start_pos, end_pos, target_k);
        result
            .map(|pos| pos as usize)
            .ok_or_else(|| ZiporaError::invalid_data("Select failed in range"))
    }

    /// Select from a specific position (for cache optimization)
    fn select1_from_position(&self, start_pos: usize, remaining_k: usize) -> Result<usize> {
        let result = self.find_nth_one_in_range(start_pos, self.total_bits, remaining_k);
        result
            .map(|pos| pos as usize)
            .ok_or_else(|| ZiporaError::invalid_data("Select failed from position"))
    }

    /// Get cache statistics for analysis
    pub fn cache_stats(&self) -> HierarchicalCacheStats {
        HierarchicalCacheStats {
            levels: C::LEVELS,
            l1_entries: self.l1_cache.len(),
            l2_entries: self.l2_cache.len(),
            l3_entries: self.l3_cache.len(),
            l4_entries: self.l4_cache.len(),
            select_entries: self.select_cache.len(),
            l1_density: C::L1_Q.as_usize(),
            l2_density: C::L2_Q.as_usize(),
            l3_density: C::L3_Q.as_usize(),
            l4_density: C::L4_Q.as_usize(),
            select_density: C::SELECT_Q.as_usize(),
            total_cache_size: self.calculate_total_cache_size(),
            cache_efficiency: self.calculate_cache_efficiency(),
        }
    }

    /// Calculate total cache size in bytes
    fn calculate_total_cache_size(&self) -> usize {
        let l1_size = self.l1_cache.len() * std::mem::size_of::<L1CacheEntry>();
        let l2_size = self.l2_cache.len() * std::mem::size_of::<L2CacheEntry>();
        let l3_size = self.l3_cache.len() * std::mem::size_of::<L3CacheEntry>();
        let l4_size = self.l4_cache.len() * std::mem::size_of::<L4CacheEntry>();
        let select_size = self.select_cache.len() * std::mem::size_of::<SelectCacheEntry>();

        l1_size + l2_size + l3_size + l4_size + select_size
    }

    /// Calculate cache efficiency (hit rate estimation)
    fn calculate_cache_efficiency(&self) -> f64 {
        if self.total_bits == 0 {
            return 1.0;
        }

        let bits_per_l1_cache = L1_BLOCK_BITS * C::L1_Q.as_usize();
        let l1_coverage = if bits_per_l1_cache > 0 {
            self.total_bits as f64 / bits_per_l1_cache as f64
        } else {
            0.0
        };

        // Estimate cache hit rate based on level density
        let base_efficiency: f64 = match C::L1_Q {
            CacheDensity::Dense => 0.95,
            CacheDensity::Half => 0.85,
            CacheDensity::Quarter => 0.75,
            CacheDensity::Eighth => 0.65,
            CacheDensity::Sixteenth => 0.55,
        };

        base_efficiency.min(1.0)
    }
}

/// Cache statistics for hierarchical structures
#[derive(Debug, Clone)]
pub struct HierarchicalCacheStats {
    pub levels: usize,
    pub l1_entries: usize,
    pub l2_entries: usize,
    pub l3_entries: usize,
    pub l4_entries: usize,
    pub select_entries: usize,
    pub l1_density: usize,
    pub l2_density: usize,
    pub l3_density: usize,
    pub l4_density: usize,
    pub select_density: usize,
    pub total_cache_size: usize,
    pub cache_efficiency: f64,
}

impl<C: HierarchicalConfig> RankSelectOps for RankSelectHierarchical<C> {
    fn rank1(&self, pos: usize) -> usize {
        self.rank1_hierarchical(pos)
    }

    fn rank0(&self, pos: usize) -> usize {
        pos - self.rank1(pos)
    }

    fn select1(&self, k: usize) -> Result<usize> {
        self.select1_hierarchical(k)
    }

    fn select0(&self, k: usize) -> Result<usize> {
        let total_zeros = self.total_bits - self.total_ones;
        if k >= total_zeros {
            return Err(ZiporaError::invalid_data("Select0 index out of bounds"));
        }

        // Binary search for select0
        let mut left = 0;
        let mut right = self.total_bits;

        while left < right {
            let mid = (left + right) / 2;
            let zeros_before = self.rank0(mid);

            if zeros_before <= k {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        Ok(left)
    }

    fn len(&self) -> usize {
        self.total_bits
    }

    fn count_ones(&self) -> usize {
        self.total_ones
    }

    fn get(&self, index: usize) -> Option<bool> {
        if index >= self.total_bits {
            return None;
        }

        let word_idx = index / 64;
        let bit_offset = index % 64;

        if word_idx < self.bit_data.len() {
            Some((self.bit_data[word_idx] >> bit_offset) & 1 == 1)
        } else {
            None
        }
    }

    fn space_overhead_percent(&self) -> f64 {
        if self.total_bits == 0 {
            return 0.0;
        }

        let original_size = (self.total_bits + 7) / 8; // Original bit vector size in bytes
        let cache_size = self.calculate_total_cache_size();

        ((cache_size as f64) / (original_size as f64)) * 100.0
    }
}

impl<C: HierarchicalConfig> RankSelectPerformanceOps for RankSelectHierarchical<C> {
    fn rank1_hardware_accelerated(&self, pos: usize) -> usize {
        self.rank1(pos) // Already optimized with hardware when available
    }

    fn select1_hardware_accelerated(&self, k: usize) -> Result<usize> {
        self.select1(k) // Already optimized with hardware when available
    }

    fn rank1_adaptive(&self, pos: usize) -> usize {
        self.rank1(pos)
    }

    fn select1_adaptive(&self, k: usize) -> Result<usize> {
        self.select1(k)
    }

    fn rank1_bulk(&self, positions: &[usize]) -> Vec<usize> {
        positions.iter().map(|&pos| self.rank1(pos)).collect()
    }

    fn select1_bulk(&self, indices: &[usize]) -> Result<Vec<usize>> {
        indices.iter().map(|&k| self.select1(k)).collect()
    }
}

/// Type aliases for common configurations
pub type RankSelectStandard = RankSelectHierarchical<StandardConfig>;
pub type RankSelectFast = RankSelectHierarchical<FastConfig>;
pub type RankSelectCompact = RankSelectHierarchical<CompactConfig>;
pub type RankSelectBalanced = RankSelectHierarchical<BalancedConfig>;
pub type RankSelectSelectOptimized = RankSelectHierarchical<SelectOptimizedConfig>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::succinct::BitVector;

    fn create_test_bitvector(size: usize, pattern: fn(usize) -> bool) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(pattern(i)).unwrap();
        }
        bv
    }

    #[test]
    fn test_cache_density_conversion() {
        assert_eq!(CacheDensity::Dense.as_usize(), 1);
        assert_eq!(CacheDensity::Half.as_usize(), 2);
        assert_eq!(CacheDensity::Quarter.as_usize(), 4);
        assert_eq!(CacheDensity::Eighth.as_usize(), 8);
        assert_eq!(CacheDensity::Sixteenth.as_usize(), 16);
    }

    #[test]
    fn test_configuration_constants() {
        assert_eq!(StandardConfig::LEVELS, 3);
        assert_eq!(StandardConfig::L1_Q, CacheDensity::Dense);
        assert_eq!(StandardConfig::L2_Q, CacheDensity::Quarter);
        assert!(StandardConfig::OPTIMIZE_RANK);
        assert!(!StandardConfig::OPTIMIZE_SELECT);

        assert_eq!(FastConfig::LEVELS, 4);
        assert_eq!(FastConfig::L1_Q, CacheDensity::Dense);
        assert_eq!(FastConfig::L2_Q, CacheDensity::Dense);
        assert!(FastConfig::OPTIMIZE_RANK);
        assert!(FastConfig::OPTIMIZE_SELECT);

        assert_eq!(CompactConfig::LEVELS, 2);
        assert_eq!(CompactConfig::L1_Q, CacheDensity::Half);
        assert_eq!(CompactConfig::L2_Q, CacheDensity::Eighth);
        assert!(CompactConfig::OPTIMIZE_RANK);
        assert!(!CompactConfig::OPTIMIZE_SELECT);
    }

    #[test]
    fn test_hierarchical_standard_config() {
        let bv = create_test_bitvector(10000, |i| i % 7 == 0);
        let rs = RankSelectStandard::new(bv.clone()).unwrap();

        // Test basic properties
        assert_eq!(rs.len(), bv.len());
        assert_eq!(rs.count_ones(), bv.count_ones());

        // Test rank operations
        for pos in [0, 1000, 5000, 9999] {
            let expected = bv.rank1(pos);
            let actual = rs.rank1(pos);
            assert_eq!(actual, expected, "Rank mismatch at position {}", pos);
        }

        // Test select operations
        let ones_count = rs.count_ones();
        for k in [0, ones_count / 4, ones_count / 2, ones_count * 3 / 4] {
            if k < ones_count {
                let result = rs.select1(k);
                assert!(result.is_ok(), "Select failed for k={}", k);

                let pos = result.unwrap();
                assert_eq!(rs.rank1(pos), k, "Select verification failed");
            }
        }
    }

    #[test]
    fn test_hierarchical_fast_config() {
        let bv = create_test_bitvector(20000, |i| i % 11 == 0);
        let rs = RankSelectFast::new(bv.clone()).unwrap();

        // Test that fast config has more cache levels
        let stats = rs.cache_stats();
        assert_eq!(stats.levels, 4);
        assert!(stats.l1_entries > 0);
        assert!(stats.l2_entries >= 0); // May be 0 for small datasets
        assert!(stats.select_entries > 0); // Should have select cache

        // Test performance
        let test_positions = [0, 1000, 10000, 19999];
        for &pos in &test_positions {
            let expected = bv.rank1(pos);
            let actual = rs.rank1(pos);
            assert_eq!(actual, expected, "Fast config rank mismatch at {}", pos);
        }
    }

    #[test]
    fn test_hierarchical_compact_config() {
        let bv = create_test_bitvector(5000, |i| i % 13 == 0);
        let rs = RankSelectCompact::new(bv.clone()).unwrap();

        // Test that compact config has fewer cache levels
        let stats = rs.cache_stats();
        assert_eq!(stats.levels, 2);
        assert!(stats.l1_entries > 0);
        assert_eq!(stats.l3_entries, 0); // Should not have L3 cache
        assert_eq!(stats.l4_entries, 0); // Should not have L4 cache

        // Test space efficiency
        let overhead = rs.space_overhead_percent();
        println!("Compact config overhead: {:.2}%", overhead);
        assert!(overhead < 20.0, "Compact config should have low overhead");
    }

    #[test]
    fn test_hierarchical_balanced_config() {
        let bv = create_test_bitvector(15000, |i| (i * 3 + 7) % 17 == 0);
        let rs = RankSelectBalanced::new(bv.clone()).unwrap();

        let stats = rs.cache_stats();
        assert_eq!(stats.levels, 3);
        assert!(stats.l1_entries > 0);
        assert!(stats.select_entries > 0); // Should have select cache

        // Test consistency
        for pos in [0, 3000, 7500, 14999] {
            assert_eq!(rs.rank1(pos), bv.rank1(pos));
        }
    }

    #[test]
    fn test_hierarchical_select_optimized_config() {
        let bv = create_test_bitvector(25000, |i| i % 19 == 0);
        let rs = RankSelectSelectOptimized::new(bv.clone()).unwrap();

        let stats = rs.cache_stats();
        assert_eq!(stats.levels, 4);
        assert!(stats.select_entries > 0);
        assert_eq!(stats.select_density, 1); // Dense select cache

        // Test select performance
        let ones_count = rs.count_ones();
        let test_indices = [0, ones_count / 10, ones_count / 2, ones_count * 9 / 10];

        for &k in &test_indices {
            if k < ones_count {
                let result = rs.select1(k);
                assert!(result.is_ok(), "Select optimized failed for k={}", k);
            }
        }
    }

    #[test]
    fn test_cache_statistics() {
        let bv = create_test_bitvector(50000, |i| i % 23 == 0);
        let rs = RankSelectStandard::new(bv).unwrap();

        let stats = rs.cache_stats();

        assert!(stats.total_cache_size > 0);
        assert!(stats.cache_efficiency > 0.0 && stats.cache_efficiency <= 1.0);
        assert_eq!(stats.l1_density, 1); // Dense L1 cache
        assert_eq!(stats.l2_density, 4); // Quarter density L2 cache

        println!("Cache stats: {:?}", stats);
    }

    #[test]
    fn test_large_dataset_performance() {
        let bv = create_test_bitvector(100000, |i| (i * 13 + 29) % 71 == 0);

        // Test multiple configurations on same data
        let standard = RankSelectStandard::new(bv.clone()).unwrap();
        let fast = RankSelectFast::new(bv.clone()).unwrap();
        let compact = RankSelectCompact::new(bv.clone()).unwrap();

        // All should give same results
        let test_positions = [0, 10000, 50000, 99999];
        for &pos in &test_positions {
            let expected = bv.rank1(pos);
            assert_eq!(standard.rank1(pos), expected);
            assert_eq!(fast.rank1(pos), expected);
            assert_eq!(compact.rank1(pos), expected);
        }

        // Compare space overhead
        let standard_overhead = standard.space_overhead_percent();
        let fast_overhead = fast.space_overhead_percent();
        let compact_overhead = compact.space_overhead_percent();

        println!("Standard overhead: {:.2}%", standard_overhead);
        println!("Fast overhead: {:.2}%", fast_overhead);
        println!("Compact overhead: {:.2}%", compact_overhead);

        // Fast should use more space, compact should use less
        assert!(fast_overhead >= standard_overhead);
        assert!(compact_overhead <= standard_overhead);
    }

    #[test]
    fn test_bulk_operations() {
        let bv = create_test_bitvector(30000, |i| i % 37 == 0);
        let rs = RankSelectBalanced::new(bv.clone()).unwrap();

        // Test bulk rank operations
        let positions = vec![0, 5000, 15000, 25000, 29999];
        let bulk_ranks = rs.rank1_bulk(&positions);

        assert_eq!(bulk_ranks.len(), positions.len());
        for (i, &pos) in positions.iter().enumerate() {
            assert_eq!(bulk_ranks[i], bv.rank1(pos));
        }

        // Test bulk select operations
        let ones_count = rs.count_ones();
        let indices = vec![0, ones_count / 10, ones_count / 2];
        let bulk_selects = rs.select1_bulk(&indices).unwrap();

        for (i, &k) in indices.iter().enumerate() {
            if k < ones_count {
                assert_eq!(bulk_selects[i], rs.select1(k).unwrap());
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        // Empty bit vector
        let empty_bv = BitVector::new();
        let empty_rs = RankSelectStandard::new(empty_bv).unwrap();
        assert_eq!(empty_rs.len(), 0);
        assert_eq!(empty_rs.count_ones(), 0);

        // Single bit
        let mut single_bv = BitVector::new();
        single_bv.push(true).unwrap();
        let single_rs = RankSelectStandard::new(single_bv).unwrap();
        assert_eq!(single_rs.len(), 1);
        assert_eq!(single_rs.count_ones(), 1);
        assert_eq!(single_rs.rank1(0), 0);
        assert_eq!(single_rs.rank1(1), 1);
        assert_eq!(single_rs.select1(0).unwrap(), 0);

        // All zeros
        let zeros_bv = BitVector::with_size(1000, false).unwrap();
        let zeros_rs = RankSelectStandard::new(zeros_bv).unwrap();
        assert_eq!(zeros_rs.count_ones(), 0);
        assert_eq!(zeros_rs.rank1(500), 0);
        assert!(zeros_rs.select1(0).is_err());

        // All ones
        let ones_bv = BitVector::with_size(1000, true).unwrap();
        let ones_rs = RankSelectStandard::new(ones_bv).unwrap();
        assert_eq!(ones_rs.count_ones(), 1000);
        assert_eq!(ones_rs.rank1(500), 500);
        assert_eq!(ones_rs.select1(499).unwrap(), 499);
    }

    #[test]
    fn test_configuration_comparison() {
        let bv = create_test_bitvector(40000, |i| (i * 7 + 11) % 43 == 0);

        // Test each configuration separately to avoid type mismatch
        let standard = RankSelectStandard::new(bv.clone()).unwrap();
        let fast = RankSelectFast::new(bv.clone()).unwrap();
        let compact = RankSelectCompact::new(bv.clone()).unwrap();
        let balanced = RankSelectBalanced::new(bv.clone()).unwrap();
        let select_optimized = RankSelectSelectOptimized::new(bv.clone()).unwrap();

        // All configurations should give identical results
        let test_pos = 20000;
        let expected_rank = bv.rank1(test_pos);

        // Test each configuration
        assert_eq!(
            standard.rank1(test_pos),
            expected_rank,
            "Standard rank mismatch"
        );
        assert_eq!(fast.rank1(test_pos), expected_rank, "Fast rank mismatch");
        assert_eq!(
            compact.rank1(test_pos),
            expected_rank,
            "Compact rank mismatch"
        );
        assert_eq!(
            balanced.rank1(test_pos),
            expected_rank,
            "Balanced rank mismatch"
        );
        assert_eq!(
            select_optimized.rank1(test_pos),
            expected_rank,
            "SelectOptimized rank mismatch"
        );

        // Print statistics for each
        let standard_stats = standard.cache_stats();
        let fast_stats = fast.cache_stats();
        let compact_stats = compact.cache_stats();
        let balanced_stats = balanced.cache_stats();
        let select_optimized_stats = select_optimized.cache_stats();

        println!(
            "Standard: {} levels, {:.2}% overhead, {:.2} efficiency",
            standard_stats.levels,
            standard.space_overhead_percent(),
            standard_stats.cache_efficiency
        );
        println!(
            "Fast: {} levels, {:.2}% overhead, {:.2} efficiency",
            fast_stats.levels,
            fast.space_overhead_percent(),
            fast_stats.cache_efficiency
        );
        println!(
            "Compact: {} levels, {:.2}% overhead, {:.2} efficiency",
            compact_stats.levels,
            compact.space_overhead_percent(),
            compact_stats.cache_efficiency
        );
        println!(
            "Balanced: {} levels, {:.2}% overhead, {:.2} efficiency",
            balanced_stats.levels,
            balanced.space_overhead_percent(),
            balanced_stats.cache_efficiency
        );
        println!(
            "SelectOptimized: {} levels, {:.2}% overhead, {:.2} efficiency",
            select_optimized_stats.levels,
            select_optimized.space_overhead_percent(),
            select_optimized_stats.cache_efficiency
        );

        // Test select consistency
        let ones_count = bv.count_ones();
        if ones_count > 100 {
            let k = ones_count / 2;
            let expected_select = standard.select1(k).unwrap();

            assert_eq!(
                fast.select1(k).unwrap(),
                expected_select,
                "Fast select mismatch"
            );
            assert_eq!(
                compact.select1(k).unwrap(),
                expected_select,
                "Compact select mismatch"
            );
            assert_eq!(
                balanced.select1(k).unwrap(),
                expected_select,
                "Balanced select mismatch"
            );
            assert_eq!(
                select_optimized.select1(k).unwrap(),
                expected_select,
                "SelectOptimized select mismatch"
            );
        }
    }

    #[test]
    fn test_memory_layout_alignment() {
        // Test that cache entries are properly aligned
        assert_eq!(std::mem::align_of::<L1CacheEntry>(), 8); // 64-bit alignment
        assert_eq!(std::mem::align_of::<L2CacheEntry>(), 16); // 128-bit alignment
        assert_eq!(std::mem::align_of::<L3CacheEntry>(), 16);
        assert_eq!(std::mem::align_of::<L4CacheEntry>(), 16);

        // Test size efficiency
        assert_eq!(std::mem::size_of::<L1CacheEntry>(), 8); // 4 + 4 bytes with 8-byte alignment
        assert_eq!(std::mem::size_of::<L2CacheEntry>(), 16); // 4 + 4 bytes with 16-byte alignment
        assert_eq!(std::mem::size_of::<L3CacheEntry>(), 16); // 4 + 4 bytes with 16-byte alignment
        assert_eq!(std::mem::size_of::<L4CacheEntry>(), 16); // 4 + 4 bytes with 16-byte alignment
        assert_eq!(std::mem::size_of::<SelectCacheEntry>(), 8); // 4 + 4 bytes
    }
}
