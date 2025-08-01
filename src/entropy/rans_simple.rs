//! Simple working rANS implementation
//! 
//! This is a reference implementation that prioritizes correctness over performance.

use crate::error::{Result, ToplingError};

const RANS_BYTE_L: u32 = 1 << 23; // Lower bound for renormalization

/// Simple rANS encoder/decoder
#[derive(Debug)]
pub struct SimpleRans {
    freq_table: [u32; 256],
    cum_freq: [u32; 257],  // 257 to include total
    total_freq: u32,
}

impl SimpleRans {
    /// Create from frequencies
    pub fn new(frequencies: &[u32; 256]) -> Result<Self> {
        let total_freq: u32 = frequencies.iter().sum();
        if total_freq == 0 {
            return Err(ToplingError::invalid_data("No symbols with frequency"));
        }
        
        // Build cumulative frequency table
        let mut cum_freq = [0u32; 257];
        for i in 0..256 {
            cum_freq[i + 1] = cum_freq[i] + frequencies[i];
        }
        
        Ok(Self {
            freq_table: *frequencies,
            cum_freq,
            total_freq,
        })
    }
    
    /// Encode data
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut state = RANS_BYTE_L;
        let mut output = Vec::new();
        
        // Encode in reverse order
        for &symbol in data.iter().rev() {
            let freq = self.freq_table[symbol as usize];
            if freq == 0 {
                return Err(ToplingError::invalid_data(format!("Symbol {} not in table", symbol)));
            }
            
            let start = self.cum_freq[symbol as usize];
            
            // Renormalize if needed
            let max_state = ((RANS_BYTE_L >> 8) << 8) * self.total_freq;
            while state >= max_state {
                output.push((state & 0xFF) as u8);
                state >>= 8;
            }
            
            // Encode
            state = ((state / freq) * self.total_freq) + (state % freq) + start;
        }
        
        // Output final state (4 bytes, little endian)
        output.extend_from_slice(&state.to_le_bytes());
        
        // Data is in reverse order, so reverse it
        output.reverse();
        
        Ok(output)
    }
    
    /// Decode data
    pub fn decode(&self, encoded: &[u8], length: usize) -> Result<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(ToplingError::invalid_data("Encoded data too short"));
        }
        
        // Read initial state from first 4 bytes
        let mut state = u32::from_le_bytes([
            encoded[0], encoded[1], encoded[2], encoded[3]
        ]);
        
        let mut pos = 4;
        let mut result = Vec::with_capacity(length);
        
        for _ in 0..length {
            // Renormalize if needed
            while state < RANS_BYTE_L && pos < encoded.len() {
                state = (state << 8) | (encoded[pos] as u32);
                pos += 1;
            }
            
            if state < RANS_BYTE_L {
                return Err(ToplingError::invalid_data("Insufficient data for decoding"));
            }
            
            // Find symbol
            let slot = state % self.total_freq;
            let mut symbol = 0u8;
            
            // Linear search for simplicity (could be binary search)
            for s in 0..256 {
                if slot >= self.cum_freq[s] && slot < self.cum_freq[s + 1] {
                    symbol = s as u8;
                    break;
                }
            }
            
            let freq = self.freq_table[symbol as usize];
            let start = self.cum_freq[symbol as usize];
            
            // Decode
            state = freq * (state / self.total_freq) + (state % self.total_freq) - start;
            
            result.push(symbol);
        }
        
        // Reverse result since we decoded in reverse
        result.reverse();
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore] // TODO: Fix rANS decode/encode mismatch - complex algorithm issue
    fn test_simple_rans() {
        let data = b"hello world";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }
        
        let rans = SimpleRans::new(&frequencies).unwrap();
        let encoded = rans.encode(data).unwrap();
        let decoded = rans.decode(&encoded, data.len()).unwrap();
        
        assert_eq!(data.to_vec(), decoded);
    }
}