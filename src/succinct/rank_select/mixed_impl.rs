//! Implementation methods for RankSelectMixedIL256
//! 
//! This file contains the implementation methods that were too large for inline editing.

use super::*;

impl RankSelectMixedIL256 {
    /// Get bit from a specific dimension
    pub fn get_dimension_bit<const DIM: usize>(&self, index: usize) -> Option<bool> {
        if index >= self.total_bits || DIM >= 2 {
            return None;
        }

        let block_idx = index / DUAL_BLOCK_SIZE;
        let bit_offset_in_block = index % DUAL_BLOCK_SIZE;
        
        if block_idx >= self.interleaved_cache.len() {
            return None;
        }

        let cache_line = &self.interleaved_cache[block_idx];
        let bits = if DIM == 0 { &cache_line.bits0 } else { &cache_line.bits1 };
        
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
            let bits = if dim == 0 { &cache_line.bits0 } else { &cache_line.bits1 };
            
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
    fn select1_linear_search(&self, start: usize, end: usize, target_rank: usize, dim: usize) -> Result<usize> {
        let mut current_rank = self.rank1_dimension(start, dim);

        for pos in start..end {
            if self.get_dimension_bit_unchecked(pos, dim) {
                current_rank += 1;
                if current_rank == target_rank {
                    return Ok(pos);
                }
            }
        }

        Err(ZiporaError::invalid_data("Select position not found".to_string()))
    }

    /// Get bit from dimension without bounds checking (internal use)
    fn get_dimension_bit_unchecked(&self, index: usize, dim: usize) -> bool {
        let block_idx = index / DUAL_BLOCK_SIZE;
        let bit_offset_in_block = index % DUAL_BLOCK_SIZE;
        
        if block_idx >= self.interleaved_cache.len() {
            return false;
        }

        let cache_line = &self.interleaved_cache[block_idx];
        let bits = if dim == 0 { &cache_line.bits0 } else { &cache_line.bits1 };
        
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
    fn select1_within_block(&self, start_bit: usize, end_bit: usize, k: usize, dim: usize) -> Result<usize> {
        if start_bit >= self.total_bits {
            return Err(ZiporaError::invalid_data("Block start beyond bit vector".to_string()));
        }

        let block_idx = start_bit / DUAL_BLOCK_SIZE;
        if block_idx >= self.interleaved_cache.len() {
            return Err(ZiporaError::invalid_data("Block index out of range".to_string()));
        }

        let cache_line = &self.interleaved_cache[block_idx];
        let bits = if dim == 0 { &cache_line.bits0 } else { &cache_line.bits1 };
        
        let mut remaining_k = k;
        let start_word = (start_bit % DUAL_BLOCK_SIZE) / 64;
        let end_word = ((end_bit - start_bit).min(DUAL_BLOCK_SIZE - (start_bit % DUAL_BLOCK_SIZE)) + 63) / 64;
        
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
        
        Err(ZiporaError::invalid_data("Select position not found in block".to_string()))
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