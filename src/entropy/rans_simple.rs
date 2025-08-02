//! Simple working rANS implementation
//!
//! This is a reference implementation that prioritizes correctness over performance.

use crate::error::{Result, ToplingError};

const RANS_BYTE_L: u32 = 1 << 23; // Lower bound for renormalization

/// Simple rANS encoder/decoder
#[derive(Debug)]
pub struct SimpleRans {
    freq_table: [u32; 256],
    cum_freq: [u32; 257], // 257 to include total
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
                return Err(ToplingError::invalid_data(format!(
                    "Symbol {} not in table",
                    symbol
                )));
            }

            let start = self.cum_freq[symbol as usize];

            // Renormalize if needed - frequency-aware to prevent overflow
            let max_state = (((RANS_BYTE_L as u64) << 8) / self.total_freq as u64) * freq as u64;
            while state as u64 >= max_state {
                output.push((state & 0xFF) as u8);
                state >>= 8;
            }

            // Encode
            state = ((state / freq) * self.total_freq) + (state % freq) + start;
        }

        // Output final state (4 bytes, little endian)
        output.extend_from_slice(&state.to_le_bytes());

        Ok(output)
    }

    /// Decode data
    pub fn decode(&self, encoded: &[u8], length: usize) -> Result<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(ToplingError::invalid_data("Encoded data too short"));
        }

        // Read initial state from last 4 bytes (like other working implementations)
        let data_len = encoded.len();
        let mut state = u32::from_le_bytes([
            encoded[data_len - 4],
            encoded[data_len - 3],
            encoded[data_len - 2],
            encoded[data_len - 1],
        ]);

        let mut pos = data_len - 4; // Start reading backwards
        let mut result = Vec::with_capacity(length);

        for _ in 0..length {
            // Renormalize if needed - read bytes backwards
            while state < RANS_BYTE_L {
                if pos == 0 {
                    return Err(ToplingError::invalid_data("Insufficient data for decoding"));
                }
                pos -= 1;
                state = (state << 8) | (encoded[pos] as u32);
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

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
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
