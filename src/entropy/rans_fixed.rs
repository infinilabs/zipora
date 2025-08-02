//! Fixed rANS implementation following canonical specification
//!
//! This implements the correct rANS algorithm based on Jarek Duda's specification

use crate::error::{Result, ToplingError};

const RANS_BYTE_L: u32 = 1 << 23; // 8388608

/// Fixed rANS implementation
#[derive(Debug)]
pub struct FixedRans {
    frequencies: [u32; 256],
    cumulative: [u32; 257], // cumulative[i] = sum of frequencies[0..i]
    total_freq: u32,
}

impl FixedRans {
    /// Create from frequencies
    pub fn new(frequencies: &[u32; 256]) -> Result<Self> {
        let total_freq: u32 = frequencies.iter().sum();
        if total_freq == 0 {
            return Err(ToplingError::invalid_data("No symbols with frequency"));
        }

        // Build cumulative frequency table: cumulative[i] = sum of freq[0..i]
        let mut cumulative = [0u32; 257];
        for i in 0..256 {
            cumulative[i + 1] = cumulative[i] + frequencies[i];
        }

        Ok(Self {
            frequencies: *frequencies,
            cumulative,
            total_freq,
        })
    }

    /// Encode data
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut state = RANS_BYTE_L;
        let mut output = Vec::new();

        // Process symbols in reverse order (this is key to rANS)
        for &symbol in data.iter().rev() {
            let freq = self.frequencies[symbol as usize];
            let cumfreq = self.cumulative[symbol as usize];

            if freq == 0 {
                return Err(ToplingError::invalid_data(format!(
                    "Symbol {} not in table",
                    symbol
                )));
            }

            // Renormalize before encoding
            // Frequency-aware renormalization to prevent overflow
            let max_state = (((RANS_BYTE_L as u64) << 8) / self.total_freq as u64) * freq as u64;
            while state as u64 >= max_state {
                output.push((state & 0xFF) as u8);
                state >>= 8;
            }

            // rANS encoding formula: x_new = (x // freq) * M + (x % freq) + cumfreq
            state = ((state / freq) * self.total_freq) + (state % freq) + cumfreq;
        }

        // Output final state
        output.extend_from_slice(&state.to_le_bytes());

        Ok(output)
    }

    /// Decode data
    pub fn decode(&self, encoded: &[u8], length: usize) -> Result<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(ToplingError::invalid_data("Encoded data too short"));
        }

        // Read initial state from last 4 bytes
        let data_len = encoded.len();
        let mut state = u32::from_le_bytes([
            encoded[data_len - 4],
            encoded[data_len - 3],
            encoded[data_len - 2],
            encoded[data_len - 1],
        ]);

        let mut pos = data_len - 4; // Position for reading bytes backwards
        let mut result = Vec::with_capacity(length);

        // Decode symbols
        for _ in 0..length {
            // Renormalize: read bytes when state gets too small
            while state < RANS_BYTE_L && pos > 0 {
                pos -= 1;
                state = (state << 8) | (encoded[pos] as u32);
            }

            if state < RANS_BYTE_L {
                return Err(ToplingError::invalid_data("Insufficient data"));
            }

            // Find symbol: determine slot and look up symbol
            let slot = state % self.total_freq;

            // Find which symbol this slot belongs to
            let mut symbol = 0u8;
            for s in 0..256 {
                if slot >= self.cumulative[s] && slot < self.cumulative[s + 1] {
                    symbol = s as u8;
                    break;
                }
            }

            let freq = self.frequencies[symbol as usize];
            let cumfreq = self.cumulative[symbol as usize];

            // rANS decoding formula: x_new = freq * (x // M) + (x % M) - cumfreq
            state = freq * (state / self.total_freq) + (state % self.total_freq) - cumfreq;

            result.push(symbol);
        }

        // No need to reverse - rANS naturally decodes in reverse order
        // and we encoded in reverse order, so they cancel out

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_rans_simple() {
        let data = b"aa";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        let rans = FixedRans::new(&frequencies).unwrap();
        let encoded = rans.encode(data).unwrap();
        let decoded = rans.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_fixed_rans_he() {
        let data = b"he";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        let rans = FixedRans::new(&frequencies).unwrap();
        let encoded = rans.encode(data).unwrap();
        let decoded = rans.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_fixed_rans_hello() {
        let data = b"hello";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        let rans = FixedRans::new(&frequencies).unwrap();
        let encoded = rans.encode(data).unwrap();
        let decoded = rans.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }
}
