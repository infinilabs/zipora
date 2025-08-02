//! rANS (range Asymmetric Numeral Systems) implementation
//!
//! This module provides rANS encoding, a modern entropy coding technique that
//! achieves better compression than Huffman coding for many data types.

use crate::error::{Result, ToplingError};

// Standard rANS constants for byte-based implementation
const RANS_BYTE_L: u32 = 1 << 23; // Lower bound: 8,388,608

/// rANS state for encoding/decoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RansState {
    state: u32,
}

impl RansState {
    /// Create new rANS state
    pub fn new() -> Self {
        Self { state: RANS_BYTE_L } // Initial state
    }

    /// Create from raw state value
    pub fn from_state(state: u32) -> Self {
        Self { state }
    }

    /// Get raw state value
    pub fn state(&self) -> u32 {
        self.state
    }

    /// Set raw state value
    pub fn set_state(&mut self, state: u32) {
        self.state = state;
    }
}

impl Default for RansState {
    fn default() -> Self {
        Self::new()
    }
}

/// Symbol information for rANS
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RansSymbol {
    /// Start of the symbol's range
    pub start: u32,
    /// Frequency of the symbol
    pub freq: u32,
}

impl RansSymbol {
    /// Create new rANS symbol
    pub fn new(start: u32, freq: u32) -> Self {
        Self { start, freq }
    }
}

/// rANS encoder
#[derive(Debug)]
pub struct RansEncoder {
    symbols: [RansSymbol; 256],
    total_freq: u32,
}

impl RansEncoder {
    /// Create encoder from symbol frequencies  
    pub fn new(frequencies: &[u32; 256]) -> Result<Self> {
        let total_freq: u32 = frequencies.iter().sum();
        // Allow zero frequency for empty data handling
        if total_freq == 0 {
            // Create a dummy encoder for empty data
            return Ok(Self {
                symbols: [RansSymbol::new(0, 0); 256],
                total_freq: 0,
            });
        }

        let mut symbols = [RansSymbol::new(0, 0); 256];
        let mut cumulative = 0u32;

        // Build cumulative frequency table
        for (i, &freq) in frequencies.iter().enumerate() {
            symbols[i] = RansSymbol::new(cumulative, freq);
            cumulative += freq;
        }

        Ok(Self {
            symbols,
            total_freq,
        })
    }

    /// Encode a symbol
    pub fn encode_symbol(
        &self,
        state: &mut RansState,
        symbol: u8,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        let sym = &self.symbols[symbol as usize];

        if sym.freq == 0 {
            return Err(ToplingError::invalid_data(format!(
                "Symbol {} not in frequency table",
                symbol
            )));
        }

        let mut s = state.state;
        let freq = sym.freq;
        let total_freq = self.total_freq;

        // Renormalize: output bytes when state gets too large
        // Proper frequency-aware condition: state >= ((L >> 8) << 8) * freq
        let max_state = ((RANS_BYTE_L as u64) << 8) / (total_freq as u64) * (freq as u64);

        while (s as u64) >= max_state {
            output.push((s & 0xFF) as u8);
            s >>= 8;
        }

        // Encode using canonical rANS formula
        // Use u64 arithmetic to prevent overflow
        let s_u64 = s as u64;
        let freq_u64 = freq as u64;
        let total_freq_u64 = total_freq as u64;
        let start_u64 = sym.start as u64;

        let new_state_u64 = ((s_u64 / freq_u64) * total_freq_u64) + (s_u64 % freq_u64) + start_u64;
        let new_state = new_state_u64 as u32;
        state.state = new_state;

        Ok(())
    }

    /// Encode data
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Handle empty data
        if data.is_empty() {
            let mut output = Vec::new();
            let state = RansState::new();
            output.extend_from_slice(&state.state.to_le_bytes());
            return Ok(output);
        }

        let mut state = RansState::new();
        let mut output = Vec::new();

        // Encode symbols in reverse order
        for &symbol in data.iter().rev() {
            self.encode_symbol(&mut state, symbol, &mut output)?;
        }

        // Flush final state as 4 bytes
        output.extend_from_slice(&state.state.to_le_bytes());

        Ok(output)
    }

    /// Get symbol information
    pub fn get_symbol(&self, symbol: u8) -> &RansSymbol {
        &self.symbols[symbol as usize]
    }

    /// Get total frequency
    pub fn total_freq(&self) -> u32 {
        self.total_freq
    }
}

/// rANS decoder
#[derive(Debug)]
pub struct RansDecoder {
    symbols: [RansSymbol; 256],
    #[allow(dead_code)]
    cumulative: [u32; 257],
    total_freq: u32,
}

impl RansDecoder {
    /// Create decoder from encoder
    pub fn new(encoder: &RansEncoder) -> Self {
        Self {
            symbols: encoder.symbols,
            cumulative: [0u32; 257], // We won't use this, will use symbols directly
            total_freq: encoder.total_freq,
        }
    }

    /// Decode a symbol reading bytes in reverse from encoded stream  
    pub fn decode_symbol_reverse(
        &self,
        state: &mut RansState,
        input: &[u8],
        pos: &mut usize,
    ) -> Result<u8> {
        // Renormalize first: read bytes when state gets too small
        while state.state < RANS_BYTE_L {
            if *pos == 0 {
                return Err(ToplingError::invalid_data("Insufficient data for decoding"));
            }
            *pos -= 1;
            state.state = (state.state << 8) | (input[*pos] as u32);
        }

        // Find symbol using linear search through symbols
        let slot = state.state % self.total_freq;
        let mut symbol = 0u8;

        // Search for the symbol whose range contains the slot
        for s in 0..256 {
            let sym = &self.symbols[s];
            if sym.freq > 0 && slot >= sym.start && slot < sym.start + sym.freq {
                symbol = s as u8;
                break;
            }
        }

        let sym_info = &self.symbols[symbol as usize];
        let freq = sym_info.freq;
        let total_freq = self.total_freq;

        // Decode using canonical rANS formula (inverse of encoding)
        let cumfreq = sym_info.start;
        let new_state = freq * (state.state / total_freq) + (state.state % total_freq) - cumfreq;
        state.state = new_state;

        Ok(symbol)
    }

    /// Decode data
    pub fn decode(&self, encoded_data: &[u8], output_length: usize) -> Result<Vec<u8>> {
        if output_length == 0 {
            return Ok(Vec::new());
        }

        if encoded_data.len() < 4 {
            return Err(ToplingError::invalid_data("rANS data too short"));
        }

        // Read initial state from last 4 bytes (little endian)
        let data_len = encoded_data.len();
        let state_bytes = &encoded_data[data_len - 4..];
        let initial_state = u32::from_le_bytes([
            state_bytes[0],
            state_bytes[1],
            state_bytes[2],
            state_bytes[3],
        ]);

        let mut state = RansState::from_state(initial_state);
        let mut pos = data_len - 4;
        let mut result = Vec::with_capacity(output_length);

        for _ in 0..output_length {
            let symbol = self.decode_symbol_reverse(&mut state, encoded_data, &mut pos)?;
            result.push(symbol);
        }

        // No need to reverse - rANS naturally decodes in reverse order
        // and we encoded in reverse order, so they cancel out

        Ok(result)
    }
}

/// Calculate frequencies from data.
pub fn calculate_frequencies(data: &[u8]) -> [u32; 256] {
    let mut frequencies = [0u32; 256];
    for &byte in data {
        frequencies[byte as usize] += 1;
    }
    frequencies
}

/// Estimate compression ratio for rANS
pub fn estimate_compression_ratio(frequencies: &[u32; 256]) -> f64 {
    let total: u64 = frequencies.iter().map(|&f| f as u64).sum();
    if total == 0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &freq in frequencies {
        if freq > 0 {
            let p = freq as f64 / total as f64;
            entropy -= p * p.log2();
        }
    }

    entropy / 8.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rans_state() {
        let mut state = RansState::new();
        assert_eq!(state.state(), RANS_BYTE_L);

        state.set_state(12345);
        assert_eq!(state.state(), 12345);

        let state2 = RansState::from_state(67890);
        assert_eq!(state2.state(), 67890);
    }

    #[test]
    fn test_rans_symbol() {
        let symbol = RansSymbol::new(10, 5);
        assert_eq!(symbol.start, 10);
        assert_eq!(symbol.freq, 5);
    }

    #[test]
    fn test_rans_encoding_decoding() {
        let data = b"hello world, this is a test of rANS encoding and decoding";
        let frequencies = calculate_frequencies(data);

        let encoder = RansEncoder::new(&frequencies).unwrap();
        let encoded = encoder.encode(data).unwrap();

        let decoder = RansDecoder::new(&encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_rans_simple() {
        let data = b"aa";
        let frequencies = calculate_frequencies(data);

        let encoder = RansEncoder::new(&frequencies).unwrap();
        let encoded = encoder.encode(data).unwrap();

        let decoder = RansDecoder::new(&encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_rans_empty_data() {
        let data = b"";
        let frequencies = calculate_frequencies(data);

        let encoder = RansEncoder::new(&frequencies).unwrap();
        let encoded = encoder.encode(data).unwrap();

        let decoder = RansDecoder::new(&encoder);
        let decoded = decoder.decode(&encoded, 0).unwrap();

        assert!(decoded.is_empty());
    }

    #[test]
    fn test_rans_single_symbol() {
        let data = b"aaaaaaaaaa";
        let frequencies = calculate_frequencies(data);

        let encoder = RansEncoder::new(&frequencies).unwrap();
        let encoded = encoder.encode(data).unwrap();

        let decoder = RansDecoder::new(&encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_rans_compression_ratio() {
        let data = b"aaaaaabbbbccccdddd";
        let frequencies = calculate_frequencies(data);

        let ratio = estimate_compression_ratio(&frequencies);
        assert!(ratio < 1.0);
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_rans_uniform_distribution() {
        let data: Vec<u8> = (0..=255).collect();
        let frequencies = calculate_frequencies(&data);

        let ratio = estimate_compression_ratio(&frequencies);
        assert!((ratio - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_frequency_calculation() {
        let data = b"hello";
        let frequencies = calculate_frequencies(data);

        assert_eq!(frequencies[b'h' as usize], 1);
        assert_eq!(frequencies[b'e' as usize], 1);
        assert_eq!(frequencies[b'l' as usize], 2);
        assert_eq!(frequencies[b'o' as usize], 1);
        assert_eq!(frequencies[b'x' as usize], 0); // Changed from 1 to 0
    }

    #[test]
    fn test_rans_encoder_creation() {
        let mut frequencies = [1u32; 256];
        frequencies[65] = 100;
        frequencies[66] = 50;
        frequencies[67] = 25;

        let encoder = RansEncoder::new(&frequencies).unwrap();
        assert!(encoder.total_freq() > 0);

        let sym_a = encoder.get_symbol(65);
        assert!(sym_a.freq > 0);
    }

    #[test]
    fn test_rans_error_handling() {
        let frequencies = [0u32; 256]; // All zero frequencies
        let result = RansEncoder::new(&frequencies);
        assert!(result.is_ok()); // Should succeed for empty data handling
    }

    #[test]
    fn test_large_data() {
        let mut data: Vec<u8> = Vec::with_capacity(2000);
        for i in 0..2000 {
            data.push(((i * 123 + 45) % 256) as u8);
        }
        let frequencies = calculate_frequencies(&data);

        let encoder = RansEncoder::new(&frequencies).unwrap();
        let encoded = encoder.encode(&data).unwrap();

        let decoder = RansDecoder::new(&encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data, decoded);
    }

    #[test]
    fn test_all_byte_values() {
        let data: Vec<u8> = (0..=255).collect();
        let frequencies = calculate_frequencies(&data);

        let encoder = RansEncoder::new(&frequencies).unwrap();
        let encoded = encoder.encode(&data).unwrap();

        let decoder = RansDecoder::new(&encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data, decoded);
    }
}
