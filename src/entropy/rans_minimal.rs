//! Minimal rANS implementation for debugging
//! 
//! This is a reference implementation that follows the exact canonical algorithm.

use crate::error::{Result, ToplingError};

const RANS_BYTE_L: u32 = 1 << 23; // Lower bound for renormalization

/// Minimal rANS encoder/decoder
#[derive(Debug)]
pub struct MinimalRans {
    frequencies: [u32; 256],
    cumulative: [u32; 257], // 257 to include total at end
    total_freq: u32,
    decode_table: Vec<u8>, // For fast symbol lookup during decoding
}

impl MinimalRans {
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
        
        // Build decode table for fast symbol lookup
        let mut decode_table = vec![0u8; total_freq as usize];
        for symbol in 0..256 {
            for i in 0..frequencies[symbol] {
                decode_table[(cumulative[symbol] + i) as usize] = symbol as u8;
            }
        }
        
        println!("Frequency table:");
        for i in 0..256 {
            if frequencies[i] > 0 {
                println!("  Symbol {} ('{}'): freq={}, cumulative={}", 
                    i, i as u8 as char, frequencies[i], cumulative[i]);
            }
        }
        println!("Total frequency: {}", total_freq);
        
        Ok(Self {
            frequencies: *frequencies,
            cumulative,
            total_freq,
            decode_table,
        })
    }
    
    /// Encode data
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut state = RANS_BYTE_L;
        let mut output = Vec::new();
        
        println!("\n=== ENCODING ===");
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
            // Use u64 to prevent overflow
            let max_state = ((RANS_BYTE_L as u64 >> 8) << 8) * self.total_freq as u64;
            println!("  Max state: {}, current state: {}", max_state, state);
            
            while state as u64 >= max_state {
                let byte_out = (state & 0xFF) as u8;
                println!("  Renormalizing: outputting byte {}, state {} -> {}", 
                    byte_out, state, state >> 8);
                output.push(byte_out);
                state >>= 8;
            }
            
            // Apply rANS encoding formula
            let new_state = ((state / freq) * self.total_freq) + (state % freq) + cumfreq;
            println!("  Encoding: state {} -> {}", state, new_state);
            println!("    Formula: (({} / {}) * {}) + ({} % {}) + {} = {}",
                state, freq, self.total_freq, state, freq, cumfreq, new_state);
            state = new_state;
        }
        
        // Output final state (4 bytes, little endian)
        println!("\nFinal state: {}", state);
        let state_bytes = state.to_le_bytes();
        output.extend_from_slice(&state_bytes);
        
        println!("Output bytes: {:?} (length: {})", output, output.len());
        Ok(output)
    }
    
    /// Decode data
    pub fn decode(&self, encoded: &[u8], length: usize) -> Result<Vec<u8>> {
        if encoded.len() < 4 {
            return Err(ToplingError::invalid_data("Encoded data too short"));
        }
        
        println!("\n=== DECODING ===");
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
            
            // Renormalize: read bytes when state gets too small (backwards)
            while state < RANS_BYTE_L && pos > 0 {
                pos -= 1;
                let byte_in = encoded[pos];
                let new_state = (state << 8) | (byte_in as u32);
                println!("  Renormalizing: reading byte {} at pos {}, state {} -> {}", 
                    byte_in, pos, state, new_state);
                state = new_state;
            }
            
            if state < RANS_BYTE_L {
                return Err(ToplingError::invalid_data("Insufficient data for decoding"));
            }
            
            // Find symbol
            let slot = state % self.total_freq;
            println!("  State: {}, slot: {}", state, slot);
            
            // Use decode table for fast lookup
            let symbol = if slot < self.decode_table.len() as u32 {
                self.decode_table[slot as usize]
            } else {
                return Err(ToplingError::invalid_data("Invalid slot value"));
            };
            
            let freq = self.frequencies[symbol as usize];
            let cumfreq = self.cumulative[symbol as usize];
            
            println!("  Found symbol {} ('{}'), freq: {}, cumfreq: {}", 
                symbol, symbol as char, freq, cumfreq);
            
            // Apply rANS decoding formula
            let new_state = freq * (state / self.total_freq) + (state % self.total_freq) - cumfreq;
            println!("  Decoding: state {} -> {}", state, new_state);
            println!("    Formula: {} * ({} / {}) + ({} % {}) - {} = {}",
                freq, state, self.total_freq, state, self.total_freq, cumfreq, new_state);
            state = new_state;
            
            result.push(symbol);
        }
        
        println!("\nDecoded result before reverse: {:?}", result);
        
        // Since we encoded in reverse order, we need to reverse the result
        result.reverse();
        
        println!("Decoded result after reverse: {:?}", result);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_minimal_rans_simple() {
        // Test with just two symbols for easier debugging
        let data = b"aa";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }
        
        let rans = MinimalRans::new(&frequencies).unwrap();
        let encoded = rans.encode(data).unwrap();
        let decoded = rans.decode(&encoded, data.len()).unwrap();
        
        assert_eq!(data.to_vec(), decoded);
    }
    
    #[test]
    fn test_minimal_rans_hello() {
        let data = b"hello";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }
        
        let rans = MinimalRans::new(&frequencies).unwrap();
        let encoded = rans.encode(data).unwrap();
        let decoded = rans.decode(&encoded, data.len()).unwrap();
        
        assert_eq!(data.to_vec(), decoded);
    }
}