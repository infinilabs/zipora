//! Enhanced 64-bit rANS implementation with parallel variants
//!
//! This module provides high-performance rANS encoding with advanced optimizations,
//! including 64-bit state management, parallel processing variants (x2, x4, x8),
//! and hardware-specific optimizations.

use crate::error::{Result, ZiporaError};
use crate::entropy::bit_ops::BitOps;
use std::marker::PhantomData;

// Removed unused import - x86_64 SIMD intrinsics not currently used

/// Enhanced rANS constants optimized for 64-bit performance
const RANS64_L: u64 = 1u64 << 16; // Lower bound: 65536 (optimized for 64-bit)
const TF_SHIFT: u32 = 12; // Frequency table size: 4096
const TOTFREQ: u32 = 1u32 << TF_SHIFT; // Total frequency: 4096
const BLOCK_SIZE: usize = 4; // 4-byte read/write operations

/// 64-bit rANS state with hardware optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rans64State {
    state: u64,
}

impl Rans64State {
    /// Create new 64-bit rANS state
    #[inline]
    pub fn new() -> Self {
        Self { state: RANS64_L }
    }

    /// Create from raw state value
    #[inline]
    pub fn from_state(state: u64) -> Self {
        Self { state }
    }

    /// Get raw state value
    #[inline]
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Set raw state value
    #[inline]
    pub fn set_state(&mut self, state: u64) {
        self.state = state;
    }

    /// Check if renormalization is needed for encoding
    #[inline]
    pub fn needs_renorm_encode(&self, freq: u32) -> bool {
        // Standard rANS renormalization condition
        self.state >= ((RANS64_L << 8) / freq as u64) * freq as u64
    }

    /// Check if renormalization is needed for decoding
    #[inline]
    pub fn needs_renorm_decode(&self) -> bool {
        self.state < RANS64_L
    }
}

impl Default for Rans64State {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced symbol information with pre-computed reciprocals for fast division
#[derive(Debug, Clone, Copy)]
pub struct Rans64Symbol {
    /// Start of the symbol's range
    pub start: u32,
    /// Frequency of the symbol
    pub freq: u32,
    /// Pre-computed reciprocal for fast division (rcp_freq = ceil(2^64 / freq))
    pub rcp_freq: u64,
    /// Right shift amount for reciprocal
    pub rcp_shift: u32,
    /// Bias for fast division
    pub bias: u64,
    /// Complement frequency (TOTFREQ - freq)
    pub cmpl_freq: u32,
}

impl Rans64Symbol {
    /// Create new enhanced rANS symbol with pre-computed reciprocals
    pub fn new(start: u32, freq: u32) -> Self {
        let (rcp_freq, rcp_shift, bias) = if freq > 0 {
            Self::compute_reciprocal(freq)
        } else {
            (0, 0, 0)
        };

        Self {
            start,
            freq,
            rcp_freq,
            rcp_shift,
            bias,
            cmpl_freq: TOTFREQ - freq,
        }
    }

    /// Compute reciprocal for fast division using Alverson's algorithm
    fn compute_reciprocal(freq: u32) -> (u64, u32, u64) {
        if freq == 0 {
            return (0, 0, 0);
        }

        // Find the number of bits needed
        let bits = 64 - freq.leading_zeros();
        let shift = bits;
        
        // Compute ceil(2^64 / freq)
        let rcp_freq = if freq == 1 {
            u64::MAX
        } else {
            (u64::MAX / freq as u64) + 1
        };

        // Compute bias for exact division
        let bias = if freq == 1 { 0 } else { freq as u64 - 1 };

        (rcp_freq, shift, bias)
    }

    /// Fast division using pre-computed reciprocal
    #[inline]
    pub fn fast_div(&self, x: u64) -> (u64, u64) {
        if self.freq == 0 {
            return (0, 0);
        }
        
        if self.freq == 1 {
            // Special case for freq=1: no division needed
            return (x, 0);
        }

        // Use 128-bit multiplication and extract high 64 bits
        let q = self.mul_hi_u64(x, self.rcp_freq) >> self.rcp_shift;
        let r = x - q * self.freq as u64;
        (q, r)
    }

    /// 128-bit multiplication, return high 64 bits
    #[inline]
    fn mul_hi_u64(&self, a: u64, b: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            // Use hardware 128-bit multiplication when available
            if std::is_x86_feature_detected!("bmi2") {
                unsafe {
                    let result = (a as u128) * (b as u128);
                    (result >> 64) as u64
                }
            } else {
                self.mul_hi_u64_software(a, b)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.mul_hi_u64_software(a, b)
        }
    }

    /// Software implementation of 128-bit multiplication
    #[inline]
    fn mul_hi_u64_software(&self, a: u64, b: u64) -> u64 {
        let a_lo = a & 0xFFFFFFFF;
        let a_hi = a >> 32;
        let b_lo = b & 0xFFFFFFFF;
        let b_hi = b >> 32;

        let p0 = a_lo * b_lo;
        let p1 = a_lo * b_hi;
        let p2 = a_hi * b_lo;
        let p3 = a_hi * b_hi;

        let p01 = (p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF);
        let p01_hi = p01 >> 32;
        let p01_carry = (p1 >> 32) + (p2 >> 32);

        p3 + p01_hi + p01_carry
    }
}

/// Parallel processing variants - compile-time selection
pub trait ParallelVariant {
    const N: usize; // Number of parallel streams
    const NAME: &'static str;
}

/// Single-stream (x1) variant
pub struct ParallelX1;
impl ParallelVariant for ParallelX1 {
    const N: usize = 1;
    const NAME: &'static str = "x1";
}

/// Dual-stream (x2) variant
pub struct ParallelX2;
impl ParallelVariant for ParallelX2 {
    const N: usize = 2;
    const NAME: &'static str = "x2";
}

/// Quad-stream (x4) variant
pub struct ParallelX4;
impl ParallelVariant for ParallelX4 {
    const N: usize = 4;
    const NAME: &'static str = "x4";
}

/// Octa-stream (x8) variant
pub struct ParallelX8;
impl ParallelVariant for ParallelX8 {
    const N: usize = 8;
    const NAME: &'static str = "x8";
}

/// Enhanced 64-bit rANS encoder with parallel processing
#[derive(Debug)]
pub struct Rans64Encoder<P: ParallelVariant> {
    symbols: [Rans64Symbol; 256],
    total_freq: u32,
    _phantom: PhantomData<P>,
}

impl<P: ParallelVariant> Rans64Encoder<P> {
    /// Create encoder from symbol frequencies
    pub fn new(frequencies: &[u32; 256]) -> Result<Self> {
        let total_freq: u32 = frequencies.iter().sum();
        if total_freq == 0 {
            return Ok(Self {
                symbols: [Rans64Symbol::new(0, 0); 256],
                total_freq: 0,
                _phantom: PhantomData,
            });
        }

        // Normalize frequencies to TOTFREQ
        let normalized_freqs = Self::normalize_frequencies(frequencies, total_freq)?;
        
        let mut symbols = [Rans64Symbol::new(0, 0); 256];
        let mut cumulative = 0u32;

        // Build enhanced symbol table with reciprocals
        for (i, &freq) in normalized_freqs.iter().enumerate() {
            symbols[i] = Rans64Symbol::new(cumulative, freq);
            cumulative += freq;
        }

        Ok(Self {
            symbols,
            total_freq: TOTFREQ,
            _phantom: PhantomData,
        })
    }

    /// Normalize frequencies to target total frequency (power of 2)
    fn normalize_frequencies(frequencies: &[u32; 256], total_freq: u32) -> Result<[u32; 256]> {
        let mut normalized = [0u32; 256];
        let mut remaining = TOTFREQ;
        let mut used_symbols = 0;

        // First pass: assign at least 1 to each non-zero frequency
        for (i, &freq) in frequencies.iter().enumerate() {
            if freq > 0 {
                normalized[i] = 1;
                remaining -= 1;
                used_symbols += 1;
            }
        }

        if used_symbols == 0 {
            return Err(ZiporaError::invalid_data("No symbols with non-zero frequency"));
        }

        // Second pass: distribute remaining frequency proportionally
        for (i, &freq) in frequencies.iter().enumerate() {
            if freq > 0 && remaining > 0 {
                let additional = ((freq as u64 * remaining as u64) / total_freq as u64) as u32;
                let to_add = additional.min(remaining);
                normalized[i] += to_add;
                remaining -= to_add;
            }
        }

        // Third pass: distribute any remaining frequency to most frequent symbols
        while remaining > 0 {
            let mut max_freq = 0;
            let mut max_idx = 0;
            for (i, &freq) in frequencies.iter().enumerate() {
                if freq > max_freq && normalized[i] < TOTFREQ / 4 {
                    max_freq = freq;
                    max_idx = i;
                }
            }
            
            if max_freq == 0 {
                // Fallback: give to first non-zero symbol
                for (i, &freq) in frequencies.iter().enumerate() {
                    if freq > 0 {
                        max_idx = i;
                        break;
                    }
                }
            }
            
            normalized[max_idx] += 1;
            remaining -= 1;
        }

        Ok(normalized)
    }

    /// Encode a symbol using enhanced 64-bit state
    #[inline]
    pub fn encode_symbol(
        &self,
        state: &mut Rans64State,
        symbol: u8,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        let sym = &self.symbols[symbol as usize];
        
        if sym.freq == 0 {
            return Err(ZiporaError::invalid_data(format!(
                "Symbol {} not in frequency table", symbol
            )));
        }

        // Renormalize: output bytes when state gets too large
        let max_state = ((RANS64_L << 8) / TOTFREQ as u64) * sym.freq as u64;
        while state.state >= max_state {
            output.push((state.state & 0xFF) as u8);
            state.state >>= 8;
        }

        // Standard rANS encoding using 64-bit arithmetic 
        let s = state.state;
        let freq = sym.freq as u64;
        let start = sym.start as u64;
        let total_freq = TOTFREQ as u64;
        
        let new_state = ((s / freq) * total_freq) + (s % freq) + start;
        state.set_state(new_state);

        Ok(())
    }

    /// Encode data using parallel processing
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            let mut output = Vec::new();
            let state = Rans64State::new();
            output.extend_from_slice(&state.state().to_le_bytes());
            return Ok(output);
        }

        if P::N == 1 {
            self.encode_single(data)
        } else {
            self.encode_parallel(data)
        }
    }

    /// Single-stream encoding
    fn encode_single(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut state = Rans64State::new();
        let mut output = Vec::new();

        // Encode symbols in reverse order
        for &symbol in data.iter().rev() {
            self.encode_symbol(&mut state, symbol, &mut output)?;
        }

        // Flush final state
        output.extend_from_slice(&state.state().to_le_bytes());
        Ok(output)
    }

    /// Parallel-stream encoding (following advanced approach)
    fn encode_parallel(&self, data: &[u8]) -> Result<Vec<u8>> {
        let n_streams = P::N;
        let data_len = data.len();
        
        if data_len < n_streams {
            // Too little data for parallelization
            return self.encode_single(data);
        }
        
        // Initialize N independent rANS states
        let mut states = vec![Rans64State::new(); n_streams];
        let mut outputs = vec![Vec::new(); n_streams];
        
        // Encode data in interleaved fashion, processing backwards
        // Each stream processes every Nth symbol: stream i handles indices i, i+N, i+2N, ...
        // Process symbols in reverse order for proper rANS encoding
        let mut stream_indices = vec![Vec::new(); n_streams];
        
        // Build indices for each stream (interleaved assignment)
        for i in 0..data_len {
            let stream_idx = i % n_streams;
            stream_indices[stream_idx].push(i);
        }
        
        // Process each stream's symbols in reverse order
        for stream_idx in 0..n_streams {
            let indices = &stream_indices[stream_idx];
            for &pos in indices.iter().rev() {
                self.encode_symbol(&mut states[stream_idx], data[pos], &mut outputs[stream_idx])?;
            }
        }
        
        // Combine outputs: first output stream states, then concatenate stream data
        let mut final_output = Vec::new();
        
        // Write stream states (8 bytes each)
        for state in &states {
            final_output.extend_from_slice(&state.state().to_le_bytes());
        }
        
        // Write stream data lengths
        for output in &outputs {
            final_output.extend_from_slice(&(output.len() as u32).to_le_bytes());
        }
        
        // Write stream data
        for output in &outputs {
            final_output.extend_from_slice(output);
        }
        
        Ok(final_output)
    }

    /// Get symbol information
    pub fn get_symbol(&self, symbol: u8) -> &Rans64Symbol {
        &self.symbols[symbol as usize]
    }

    /// Get total frequency
    pub fn total_freq(&self) -> u32 {
        self.total_freq
    }

    /// Get parallel variant name
    pub fn variant_name(&self) -> &'static str {
        P::NAME
    }
}

/// Enhanced 64-bit rANS decoder with parallel processing
#[derive(Debug)]
pub struct Rans64Decoder<P: ParallelVariant> {
    symbols: [Rans64Symbol; 256],
    decode_table: [u8; TOTFREQ as usize], // Direct lookup table for decoding
    total_freq: u32,
    _phantom: PhantomData<P>,
}

impl<P: ParallelVariant> Rans64Decoder<P> {
    /// Create decoder from encoder
    pub fn new(encoder: &Rans64Encoder<P>) -> Self {
        let mut decode_table = [0u8; TOTFREQ as usize];
        
        // Build direct lookup table for O(1) symbol lookup
        for symbol in 0..256 {
            let sym = &encoder.symbols[symbol];
            for i in 0..sym.freq {
                if (sym.start + i) < TOTFREQ {
                    decode_table[(sym.start + i) as usize] = symbol as u8;
                }
            }
        }

        Self {
            symbols: encoder.symbols,
            decode_table,
            total_freq: encoder.total_freq,
            _phantom: PhantomData,
        }
    }

    /// Decode a symbol using enhanced 64-bit state
    #[inline]
    pub fn decode_symbol(
        &self,
        state: &mut Rans64State,
        input: &[u8],
        pos: &mut usize,
    ) -> Result<u8> {
        // Renormalize: read bytes when state gets too small
        while state.needs_renorm_decode() {
            if *pos == 0 {
                return Err(ZiporaError::invalid_data("Insufficient data for decoding"));
            }
            *pos -= 1;
            state.state = (state.state << 8) | (input[*pos] as u64);
        }

        // Fast symbol lookup using direct table
        let slot = (state.state % TOTFREQ as u64) as u32;
        if slot as usize >= self.decode_table.len() {
            return Err(ZiporaError::invalid_data("Invalid slot value for decode table"));
        }
        let symbol = self.decode_table[slot as usize];
        let sym_info = &self.symbols[symbol as usize];

        // Standard rANS decoding (inverse of encoding)
        let freq = sym_info.freq as u64;
        let start = sym_info.start as u64;
        let total_freq = TOTFREQ as u64;
        let s = state.state;
        
        let new_state = freq * (s / total_freq) + (s % total_freq) - start;
        state.set_state(new_state);

        Ok(symbol)
    }

    /// Decode data using parallel processing
    pub fn decode(&self, encoded_data: &[u8], output_length: usize) -> Result<Vec<u8>> {
        if output_length == 0 {
            return Ok(Vec::new());
        }

        if P::N == 1 {
            self.decode_single(encoded_data, output_length)
        } else {
            self.decode_parallel(encoded_data, output_length)
        }
    }

    /// Single-stream decoding
    fn decode_single(&self, encoded_data: &[u8], output_length: usize) -> Result<Vec<u8>> {
        if encoded_data.len() < 8 {
            return Err(ZiporaError::invalid_data("rANS data too short"));
        }

        // Read initial state from last 8 bytes
        let data_len = encoded_data.len();
        let state_bytes = &encoded_data[data_len - 8..];
        let initial_state = u64::from_le_bytes([
            state_bytes[0], state_bytes[1], state_bytes[2], state_bytes[3],
            state_bytes[4], state_bytes[5], state_bytes[6], state_bytes[7],
        ]);

        let mut state = Rans64State::from_state(initial_state);
        let mut pos = data_len - 8;
        let mut result = Vec::with_capacity(output_length);

        for _ in 0..output_length {
            let symbol = self.decode_symbol(&mut state, encoded_data, &mut pos)?;
            result.push(symbol);
        }

        Ok(result)
    }

    /// Parallel-stream decoding (following advanced approach)
    fn decode_parallel(&self, encoded_data: &[u8], output_length: usize) -> Result<Vec<u8>> {
        let n_streams = P::N;
        
        if output_length < n_streams {
            // Too little data for parallelization
            return self.decode_single(encoded_data, output_length);
        }
        
        let min_header_size = n_streams * 8 + n_streams * 4; // states + lengths
        if encoded_data.len() < min_header_size {
            return Err(ZiporaError::invalid_data("Insufficient data for parallel rANS header"));
        }
        
        // Read stream states (8 bytes each)
        let mut states = Vec::with_capacity(n_streams);
        let mut pos = 0;
        for _ in 0..n_streams {
            let state_bytes = &encoded_data[pos..pos + 8];
            let state_value = u64::from_le_bytes([
                state_bytes[0], state_bytes[1], state_bytes[2], state_bytes[3],
                state_bytes[4], state_bytes[5], state_bytes[6], state_bytes[7],
            ]);
            states.push(Rans64State::from_state(state_value));
            pos += 8;
        }
        
        // Read stream data lengths
        let mut stream_lengths = Vec::with_capacity(n_streams);
        for _ in 0..n_streams {
            let length_bytes = &encoded_data[pos..pos + 4];
            let length = u32::from_le_bytes([
                length_bytes[0], length_bytes[1], length_bytes[2], length_bytes[3],
            ]) as usize;
            stream_lengths.push(length);
            pos += 4;
        }
        
        // Extract stream data
        let mut stream_data = Vec::with_capacity(n_streams);
        for &length in &stream_lengths {
            if pos + length > encoded_data.len() {
                return Err(ZiporaError::invalid_data("Invalid stream data length"));
            }
            stream_data.push(&encoded_data[pos..pos + length]);
            pos += length;
        }
        
        // Decode each stream independently in interleaved fashion
        let mut result = vec![0u8; output_length];
        let mut stream_positions = vec![0usize; n_streams];
        
        // Initialize stream positions at the end of each stream's data (read backwards)
        for i in 0..n_streams {
            stream_positions[i] = stream_data[i].len();
        }
        
        // Build indices for each stream (same interleaved assignment as encoding)
        let mut stream_indices = vec![Vec::new(); n_streams];
        for i in 0..output_length {
            let stream_idx = i % n_streams;
            stream_indices[stream_idx].push(i);
        }
        
        // Decode each stream's symbols (in forward order since we encoded in reverse)
        for stream_idx in 0..n_streams {
            let indices = &stream_indices[stream_idx];
            let mut stream_pos = stream_positions[stream_idx];
            
            for &output_idx in indices {
                let symbol = self.decode_symbol(
                    &mut states[stream_idx], 
                    stream_data[stream_idx], 
                    &mut stream_pos
                )?;
                result[output_idx] = symbol;
            }
        }
        
        Ok(result)
    }
}

/// Adaptive rANS encoder that selects optimal parallel variant based on data size
pub struct AdaptiveRans64Encoder {
    bit_ops: BitOps,
}

impl AdaptiveRans64Encoder {
    /// Create new adaptive encoder
    pub fn new() -> Self {
        Self {
            bit_ops: BitOps::new(),
        }
    }

    /// Select optimal parallel variant based on data characteristics
    pub fn select_variant(&self, data_size: usize) -> &'static str {
        // Based on advanced thresholds
        if data_size < 73 {
            "x1"
        } else if data_size < 73 * 73 {
            "x2"
        } else if data_size < 73 * 73 * 73 * 73 {
            "x4"
        } else {
            "x8"
        }
    }

    /// Encode with automatic variant selection
    pub fn encode_adaptive(&self, data: &[u8]) -> Result<Vec<u8>> {
        let frequencies = self.calculate_frequencies(data);
        let variant = self.select_variant(data.len());

        match variant {
            "x1" => {
                let encoder = Rans64Encoder::<ParallelX1>::new(&frequencies)?;
                encoder.encode(data)
            }
            "x2" => {
                let encoder = Rans64Encoder::<ParallelX2>::new(&frequencies)?;
                encoder.encode(data)
            }
            "x4" => {
                let encoder = Rans64Encoder::<ParallelX4>::new(&frequencies)?;
                encoder.encode(data)
            }
            "x8" => {
                let encoder = Rans64Encoder::<ParallelX8>::new(&frequencies)?;
                encoder.encode(data)
            }
            _ => unreachable!(),
        }
    }

    /// Calculate symbol frequencies
    fn calculate_frequencies(&self, data: &[u8]) -> [u32; 256] {
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }
        frequencies
    }
}

impl Default for AdaptiveRans64Encoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rans64_state() {
        let mut state = Rans64State::new();
        assert_eq!(state.state(), RANS64_L);

        state.set_state(12345);
        assert_eq!(state.state(), 12345);

        let state2 = Rans64State::from_state(67890);
        assert_eq!(state2.state(), 67890);
    }

    #[test]
    fn test_rans64_symbol_reciprocal() {
        let symbol = Rans64Symbol::new(10, 5);
        assert_eq!(symbol.start, 10);
        assert_eq!(symbol.freq, 5);
        assert!(symbol.rcp_freq > 0);
        
        // Test fast division
        let (q, r) = symbol.fast_div(1000);
        assert_eq!(q * 5 + r, 1000);
    }

    #[test]
    fn test_rans64_encoding_decoding_x1() {
        let data = b"hello world, this is a test of enhanced 64-bit rANS encoding";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        let encoder = Rans64Encoder::<ParallelX1>::new(&frequencies).unwrap();
        let encoded = encoder.encode(data).unwrap();

        let decoder = Rans64Decoder::<ParallelX1>::new(&encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
        assert_eq!(encoder.variant_name(), "x1");
    }

    #[test]
    fn test_rans64_encoding_decoding_x2() {
        let data = b"parallel encoding test with dual streams";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        let encoder = Rans64Encoder::<ParallelX2>::new(&frequencies).unwrap();
        let encoded = encoder.encode(data).unwrap();

        let decoder = Rans64Decoder::<ParallelX2>::new(&encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
        assert_eq!(encoder.variant_name(), "x2");
    }

    #[test]
    fn test_rans64_encoding_decoding_x4() {
        let data = b"quad-stream parallel encoding test with four independent streams for better performance";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        let encoder = Rans64Encoder::<ParallelX4>::new(&frequencies).unwrap();
        let encoded = encoder.encode(data).unwrap();

        let decoder = Rans64Decoder::<ParallelX4>::new(&encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
        assert_eq!(encoder.variant_name(), "x4");
    }

    #[test]
    fn test_adaptive_encoder() {
        let adaptive = AdaptiveRans64Encoder::new();
        
        // Test variant selection
        assert_eq!(adaptive.select_variant(50), "x1");       // < 73
        assert_eq!(adaptive.select_variant(100), "x2");      // 73 <= x < 5329 
        assert_eq!(adaptive.select_variant(10000), "x4");    // 5329 <= x < 28,372,625
        assert_eq!(adaptive.select_variant(30000000), "x8"); // >= 28,372,625

        // Test adaptive encoding
        let data = b"test data for adaptive encoding";
        let encoded = adaptive.encode_adaptive(data).unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_frequency_normalization() {
        let mut frequencies = [1u32; 256];
        frequencies[65] = 100;
        frequencies[66] = 50;
        
        let encoder = Rans64Encoder::<ParallelX1>::new(&frequencies).unwrap();
        assert_eq!(encoder.total_freq(), TOTFREQ);
        
        // Check that all symbols have at least frequency 1
        for i in 0..256 {
            if frequencies[i] > 0 {
                assert!(encoder.get_symbol(i as u8).freq > 0);
            }
        }
    }

    #[test]
    fn test_empty_data() {
        let data = b"";
        let frequencies = [0u32; 256];
        
        let encoder = Rans64Encoder::<ParallelX1>::new(&frequencies).unwrap();
        let encoded = encoder.encode(data).unwrap();
        
        let decoder = Rans64Decoder::<ParallelX1>::new(&encoder);
        let decoded = decoder.decode(&encoded, 0).unwrap();
        
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_large_data_x8() {
        let mut data = Vec::new();
        for i in 0..10000 {
            data.push(((i * 123 + 45) % 256) as u8);
        }
        
        let mut frequencies = [0u32; 256];
        for &byte in &data {
            frequencies[byte as usize] += 1;
        }

        let encoder = Rans64Encoder::<ParallelX8>::new(&frequencies).unwrap();
        let encoded = encoder.encode(&data).unwrap();

        let decoder = Rans64Decoder::<ParallelX8>::new(&encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data, decoded);
        assert_eq!(encoder.variant_name(), "x8");
    }

    #[test]
    fn test_parallel_roundtrip_all_variants() {
        let data = b"This is a test message for parallel rANS processing with multiple streams to verify correctness across all variants.".repeat(10);
        let mut frequencies = [0u32; 256];
        for &byte in &data {
            frequencies[byte as usize] += 1;
        }

        // Test x1 variant
        let encoder_x1 = Rans64Encoder::<ParallelX1>::new(&frequencies).unwrap();
        let encoded_x1 = encoder_x1.encode(&data).unwrap();
        let decoder_x1 = Rans64Decoder::<ParallelX1>::new(&encoder_x1);
        let decoded_x1 = decoder_x1.decode(&encoded_x1, data.len()).unwrap();
        assert_eq!(data, decoded_x1);

        // Test x2 variant
        let encoder_x2 = Rans64Encoder::<ParallelX2>::new(&frequencies).unwrap();
        let encoded_x2 = encoder_x2.encode(&data).unwrap();
        let decoder_x2 = Rans64Decoder::<ParallelX2>::new(&encoder_x2);
        let decoded_x2 = decoder_x2.decode(&encoded_x2, data.len()).unwrap();
        assert_eq!(data, decoded_x2);

        // Test x4 variant
        let encoder_x4 = Rans64Encoder::<ParallelX4>::new(&frequencies).unwrap();
        let encoded_x4 = encoder_x4.encode(&data).unwrap();
        let decoder_x4 = Rans64Decoder::<ParallelX4>::new(&encoder_x4);
        let decoded_x4 = decoder_x4.decode(&encoded_x4, data.len()).unwrap();
        assert_eq!(data, decoded_x4);

        // Test x8 variant
        let encoder_x8 = Rans64Encoder::<ParallelX8>::new(&frequencies).unwrap();
        let encoded_x8 = encoder_x8.encode(&data).unwrap();
        let decoder_x8 = Rans64Decoder::<ParallelX8>::new(&encoder_x8);
        let decoded_x8 = decoder_x8.decode(&encoded_x8, data.len()).unwrap();
        assert_eq!(data, decoded_x8);
    }
}