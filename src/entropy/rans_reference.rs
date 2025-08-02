//! Reference rANS implementation based on canonical algorithm
//! 
//! This implements the exact canonical rANS algorithm for debugging purposes

use crate::error::{Result, ToplingError};

const RANS_BYTE_L: u32 = 1 << 23; // Lower bound for renormalization (16M)

/// Reference rANS implementation
#[derive(Debug)]
pub struct ReferenceRans {
    /// Symbol frequencies
    frequencies: [u32; 256],
    /// Cumulative frequency table - cumulative[i] = sum of frequencies[0..i]
    cumulative: [u32; 257],
    /// Total frequency (sum of all symbol frequencies)
    total_freq: u32,
}

impl ReferenceRans {
    /// Create from frequencies
    pub fn new(frequencies: &[u32; 256]) -> Result<Self> {
        let total_freq: u32 = frequencies.iter().sum();
        if total_freq == 0 {
            return Err(ToplingError::invalid_data("No symbols with frequency"));
        }
        
        // Build cumulative frequency table
        let mut cumulative = [0u32; 257];
        for i in 0..256 {
            cumulative[i + 1] = cumulative[i] + frequencies[i];
        }
        
        // Print frequency table for debugging (only when needed)
        if total_freq <= 50 {
            println!("Building frequency table:");
            for i in 0..256 {
                if frequencies[i] > 0 {
                    println!("  Symbol {} ('{}'): freq={}, cumfreq_start={}", 
                        i, i as u8 as char, frequencies[i], cumulative[i]);
                }
            }
            println!("Total frequency: {}", total_freq);
        }
        
        Ok(Self {
            frequencies: *frequencies,
            cumulative,
            total_freq,
        })
    }
    
    /// Encode data using canonical rANS
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut state = RANS_BYTE_L;
        let mut output = Vec::new();
        
        println!("\n=== REFERENCE ENCODING ===");
        println!("Initial state: {}", state);
        
        // Process symbols in reverse order
        for (i, &symbol) in data.iter().rev().enumerate() {
            println!("\nEncoding symbol {} ('{}') at reverse position {}", 
                symbol, symbol as char, i);
            
            let freq = self.frequencies[symbol as usize];
            let cumfreq = self.cumulative[symbol as usize];
            
            if freq == 0 {
                return Err(ToplingError::invalid_data(format!("Symbol {} not in table", symbol)));
            }
            
            println!("  Symbol freq: {}, cumfreq: {}", freq, cumfreq);
            
            // Renormalize: output bytes when state gets too large
            // Proper frequency-aware condition: state >= ((L >> 8) << 8) * freq
            // But simplified to: state >= (L << 8) / total_freq * freq
            let max_state = ((RANS_BYTE_L as u64) << 8) / (self.total_freq as u64) * (freq as u64);
            println!("  Max state for freq {}: {}, current state: {}", freq, max_state, state);
            
            while (state as u64) >= max_state {
                let byte_out = (state & 0xFF) as u8;
                println!("  Renormalizing: outputting byte {}, state {} -> {}", 
                    byte_out, state, state >> 8);
                output.push(byte_out);
                state >>= 8;
            }
            
            // rANS encoding step: x_new = (x // freq) * M + (x % freq) + cumfreq
            // Use u64 arithmetic to prevent overflow
            let state_u64 = state as u64;
            let freq_u64 = freq as u64;
            let total_freq_u64 = self.total_freq as u64;
            let cumfreq_u64 = cumfreq as u64;
            
            let new_state_u64 = ((state_u64 / freq_u64) * total_freq_u64) + (state_u64 % freq_u64) + cumfreq_u64;
            let new_state = new_state_u64 as u32;
            println!("  Encoding: state {} -> {}", state, new_state);
            println!("    Formula: (({} / {}) * {}) + ({} % {}) + {} = {}",
                state, freq, self.total_freq, state, freq, cumfreq, new_state);
            state = new_state;
        }
        
        // Output final state (4 bytes, little endian)
        println!("\nFinal state: {}", state);
        output.extend_from_slice(&state.to_le_bytes());
        
        println!("Output bytes: {:?} (length: {})", output, output.len());
        Ok(output)
    }
    
    /// Decode data using canonical rANS
    pub fn decode(&self, encoded: &[u8], length: usize) -> Result<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(ToplingError::invalid_data("Encoded data too short"));
        }
        
        println!("\n=== REFERENCE DECODING ===");
        println!("Encoded data: {:?}", encoded);
        
        // Read final state from last 4 bytes
        let data_len = encoded.len();
        let mut state = u32::from_le_bytes([
            encoded[data_len - 4], encoded[data_len - 3], encoded[data_len - 2], encoded[data_len - 1]
        ]);
        
        println!("Initial state from bytes: {}", state);
        
        let mut pos = data_len - 4; // Position for reading renorm bytes (backwards)
        let mut result = Vec::with_capacity(length);
        
        // Decode symbols
        for i in 0..length {
            println!("\nDecoding symbol at position {}", i);
            
            // Renormalize: read bytes when state gets too small
            while state < RANS_BYTE_L {
                if pos == 0 {
                    return Err(ToplingError::invalid_data("Insufficient data for decoding"));
                }
                pos -= 1;
                let byte_in = encoded[pos];
                let new_state = (state << 8) | (byte_in as u32);
                println!("  Renormalizing: reading byte {} at pos {}, state {} -> {}", 
                    byte_in, pos, state, new_state);
                state = new_state;
            }
            
            // Find symbol: which symbol does this state represent?
            let slot = state % self.total_freq;
            println!("  State: {}, slot: {}", state, slot);
            
            // Linear search to find the symbol (simple but correct)
            let mut symbol = 0u8;
            for s in 0..256 {
                if slot >= self.cumulative[s] && slot < self.cumulative[s + 1] {
                    symbol = s as u8;
                    break;
                }
            }
            
            let freq = self.frequencies[symbol as usize];
            let cumfreq = self.cumulative[symbol as usize];
            
            println!("  Found symbol {} ('{}'), freq: {}, cumfreq: {}", 
                symbol, symbol as char, freq, cumfreq);
            
            // rANS decoding step: x_new = freq * (x // M) + (x % M) - cumfreq
            let new_state = freq * (state / self.total_freq) + (state % self.total_freq) - cumfreq;
            println!("  Decoding: state {} -> {}", state, new_state);
            println!("    Formula: {} * ({} / {}) + ({} % {}) - {} = {}",
                freq, state, self.total_freq, state, self.total_freq, cumfreq, new_state);
            state = new_state;
            
            result.push(symbol);
        }
        
        println!("\nDecoded result: {:?}", result);
        
        // No need to reverse - the decoding already produces the correct order
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reference_rans_simple() {
        let data = b"aa";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }
        
        let rans = ReferenceRans::new(&frequencies).unwrap();
        let encoded = rans.encode(data).unwrap();
        let decoded = rans.decode(&encoded, data.len()).unwrap();
        
        assert_eq!(data.to_vec(), decoded);
    }
    
    #[test]
    fn test_reference_rans_hello() {
        let data = b"hello";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }
        
        let rans = ReferenceRans::new(&frequencies).unwrap();
        let encoded = rans.encode(data).unwrap();
        let decoded = rans.decode(&encoded, data.len()).unwrap();
        
        assert_eq!(data.to_vec(), decoded);
    }
    
    #[test]
    fn test_reference_rans_longer() {
        let data = b"hello world this is a test";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }
        
        let rans = ReferenceRans::new(&frequencies).unwrap();
        let encoded = rans.encode(data).unwrap();
        let decoded = rans.decode(&encoded, data.len()).unwrap();
        
        assert_eq!(data.to_vec(), decoded);
    }
}