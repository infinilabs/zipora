//! Enhanced 64-bit rANS implementation with parallel variants
//!
//! This module provides high-performance rANS encoding with advanced optimizations,
//! including 64-bit state management, parallel processing variants (x2, x4, x8),
//! and hardware-specific optimizations.

use crate::entropy::bit_ops::BitOps;
use crate::error::{Result, ZiporaError};
use std::marker::PhantomData;

// Removed unused import - x86_64 SIMD intrinsics not currently used

/// Enhanced rANS constants optimized for 64-bit performance
const RANS64_L: u64 = 1u64 << 16; // Lower bound: 65536 (optimized for 64-bit)
const TF_SHIFT: u32 = 12; // Frequency table size: 4096
const TOTFREQ: u32 = 1u32 << TF_SHIFT; // Total frequency: 4096

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
    /// Pre-computed reciprocal for fast division
    /// (rcp_freq = ceil(2^(rcp_shift+64) / freq), ryg rans64 scheme)
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

    /// Compute reciprocal for fast division (ryg rans64 scheme).
    ///
    /// For freq >= 2: `shift = ceil(log2(freq))`, `rcp_freq =
    /// ceil(2^(shift+63) / freq)`, and the quotient is recovered as
    /// `mul_hi(x, rcp_freq) >> (shift - 1)`, exact for all 64-bit x.
    /// (A plain 64-bit round-up reciprocal without the shift is NOT exact
    /// for all divisors — the extra `shift` bits of precision are required.)
    fn compute_reciprocal(freq: u32) -> (u64, u32, u64) {
        if freq == 0 {
            return (0, 0, 0);
        }

        if freq == 1 {
            // Division by 1 is handled by an early return in fast_div
            return (u64::MAX, 0, 0);
        }

        // shift = ceil(log2(freq))
        let shift = 32 - (freq - 1).leading_zeros();

        // rcp_freq = ceil(2^(shift+63) / freq), needs 128-bit intermediate
        let rcp_freq = (1u128 << (shift + 63)).div_ceil(freq as u128) as u64;

        // Compute bias for exact division
        let bias = freq as u64 - 1;

        (rcp_freq, shift - 1, bias)
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

        // Exact: rcp_freq = ceil(2^(rcp_shift+64) / freq), so the high
        // 64 multiply bits shifted down by rcp_shift yield floor(x / freq)
        let q = self.mul_hi_u64(x, self.rcp_freq) >> self.rcp_shift;
        let r = x - q * self.freq as u64;
        (q, r)
    }

    /// 128-bit multiplication, return high 64 bits.
    ///
    /// Compiles to a single hardware instruction on all 64-bit platforms
    /// (`mul` on x86-64, `umulh` on aarch64) with no runtime branching.
    #[inline(always)]
    fn mul_hi_u64(&self, a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) >> 64) as u64
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

/// Compact 16-byte encoding entry for `s < (1 << 24)` (`max_state = freq << 12 <= 1 << 24`).
///
/// For any `1 <= freq <= 4096` and `0 <= s < (1 << 24)`, `rcp40 = ceil(2^40 / freq)` satisfies
/// `s * rcp40 < 2^64` and `(s * rcp40) >> 40 == s / freq` with a constant 40-bit shift and
/// zero branches for `freq == 1`.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct RansEncEntry {
    rcp40: u64,
    max_state: u32,
    cmpl_freq: u16,
    start: u16,
}

/// Enhanced 64-bit rANS encoder with parallel processing
#[derive(Debug)]
pub struct Rans64Encoder<P: ParallelVariant> {
    symbols: [Rans64Symbol; 256],
    enc_entries: [RansEncEntry; 256],
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
                enc_entries: [RansEncEntry::default(); 256],
                total_freq: 0,
                _phantom: PhantomData,
            });
        }

        // Normalize frequencies to TOTFREQ
        let normalized_freqs = Self::normalize_frequencies(frequencies, total_freq)?;

        let mut symbols = [Rans64Symbol::new(0, 0); 256];
        let mut enc_entries = [RansEncEntry::default(); 256];
        let mut cumulative = 0u32;

        // Build enhanced symbol table with reciprocals
        for (i, &freq) in normalized_freqs.iter().enumerate() {
            symbols[i] = Rans64Symbol::new(cumulative, freq);
            if freq > 0 {
                let rcp40 = (1u64 << 40).div_ceil(freq as u64);
                enc_entries[i] = RansEncEntry {
                    rcp40,
                    max_state: freq << TF_SHIFT,
                    cmpl_freq: (TOTFREQ - freq) as u16,
                    start: cumulative as u16,
                };
            }
            cumulative += freq;
        }

        Ok(Self {
            symbols,
            enc_entries,
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
            return Err(ZiporaError::invalid_data(
                "No symbols with non-zero frequency",
            ));
        }

        // Second pass: distribute remaining frequency proportionally.
        // Use the budget captured *before* the loop: `remaining` shrinks as
        // slices are handed out, so scaling by the live value would give
        // earlier symbols a geometrically larger share than later symbols
        // with the same input frequency.
        let initial_remaining = remaining as u64;
        for (i, &freq) in frequencies.iter().enumerate() {
            if freq > 0 && remaining > 0 {
                let additional = ((freq as u64 * initial_remaining) / total_freq as u64) as u32;
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
    #[inline(always)]
    pub fn encode_symbol(
        &self,
        state: &mut Rans64State,
        symbol: u8,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        let entry = &self.enc_entries[symbol as usize];

        if entry.max_state == 0 {
            return Err(ZiporaError::invalid_data(format!(
                "Symbol {} not in frequency table",
                symbol
            )));
        }

        // Renormalize: output bytes when state gets too large
        let max_state = entry.max_state as u64;
        while state.state >= max_state {
            output.push((state.state & 0xFF) as u8);
            state.state >>= 8;
        }

        // Standard rANS encoding using pre-computed 40-bit reciprocal [E4(d)]
        // After renormalization `state.state < max_state <= 1 << 24`, so `s * rcp40 < 2^64`
        // and `(s * rcp40) >> 40 == s / freq` with a constant shift.
        let s = state.state;
        let q = (s * entry.rcp40) >> 40;
        let new_state = s + (entry.start as u64) + q * (entry.cmpl_freq as u64);
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
        let mut output = Vec::with_capacity(data.len() + 16);

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
        let per_stream_cap = data_len / n_streams + 16;
        let mut outputs: Vec<Vec<u8>> = (0..n_streams)
            .map(|_| Vec::with_capacity(per_stream_cap))
            .collect();

        if n_streams == 4 {
            let full_quads = data_len / 4;
            for pos in (full_quads * 4..data_len).rev() {
                let sym = *data
                    .get(pos)
                    .ok_or_else(|| ZiporaError::out_of_bounds(pos, data_len))?;
                let stream_idx = pos & 3;
                self.encode_symbol(&mut states[stream_idx], sym, &mut outputs[stream_idx])?;
            }

            let [s0, s1, s2, s3] = &mut states[..4] else {
                unreachable!()
            };
            let [out0, out1, out2, out3] = &mut outputs[..4] else {
                unreachable!()
            };
            let head = data.get(..full_quads * 4).unwrap_or(&[]);
            for quad in head.chunks_exact(4).rev() {
                let &[b0, b1, b2, b3] = quad else {
                    unreachable!()
                };
                self.encode_symbol(s3, b3, out3)?;
                self.encode_symbol(s2, b2, out2)?;
                self.encode_symbol(s1, b1, out1)?;
                self.encode_symbol(s0, b0, out0)?;
            }
        } else {
            // Encode data in interleaved fashion, processing backwards.
            // Each stream processes every Nth symbol: stream i handles indices
            // i, i+N, i+2N, ..., in reverse order for proper rANS encoding.
            for stream_idx in 0..n_streams {
                for pos in (stream_idx..data_len).step_by(n_streams).rev() {
                    let sym = *data
                        .get(pos)
                        .ok_or_else(|| ZiporaError::out_of_bounds(pos, data_len))?;
                    self.encode_symbol(&mut states[stream_idx], sym, &mut outputs[stream_idx])?;
                }
            }
        }

        // Combine outputs: first output stream states, then concatenate stream data
        let total_out_bytes: usize = outputs.iter().map(Vec::len).sum();
        let mut final_output = Vec::with_capacity(n_streams * 12 + total_out_bytes);

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

/// Combined slot entry for single-lookup rANS decoding (`bias = slot - sym.start`).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct RansDecEntry {
    freq: u16,
    bias: u16,
    symbol: u8,
    _pad: u8,
}

/// Enhanced 64-bit rANS decoder with parallel processing
#[derive(Debug)]
pub struct Rans64Decoder<P: ParallelVariant> {
    symbols: [Rans64Symbol; 256],
    decode_table: [u8; TOTFREQ as usize], // Direct lookup table for decoding
    decode_entries: Box<[RansDecEntry; TOTFREQ as usize]>,
    _total_freq: u32,
    _phantom: PhantomData<P>,
}

impl<P: ParallelVariant> Rans64Decoder<P> {
    /// Create decoder from encoder
    pub fn new(encoder: &Rans64Encoder<P>) -> Self {
        let mut decode_table = [0u8; TOTFREQ as usize];
        let mut decode_entries = Box::new([RansDecEntry::default(); TOTFREQ as usize]);

        // Build direct lookup table for O(1) symbol lookup
        for symbol in 0..256 {
            let sym = &encoder.symbols[symbol];
            let freq16 = sym.freq as u16;
            let sym_u8 = symbol as u8;
            for i in 0..sym.freq {
                if (sym.start + i) < TOTFREQ {
                    let slot = (sym.start + i) as usize;
                    decode_table[slot] = sym_u8; // D10.1: in-bounds
                    decode_entries[slot] = RansDecEntry {
                        freq: freq16,
                        bias: i as u16,
                        symbol: sym_u8,
                        _pad: 0,
                    };
                }
            }
        }

        let _ = encoder.symbols;
        Self {
            symbols: encoder.symbols,
            decode_table,
            decode_entries,
            _total_freq: encoder.total_freq,
            _phantom: PhantomData,
        }
    }

    /// Decode a symbol using enhanced 64-bit state
    #[inline(always)]
    pub fn decode_symbol(
        &self,
        state: &mut Rans64State,
        input: &[u8],
        pos: &mut usize,
    ) -> Result<u8> {
        let _ = &self.symbols;
        // Renormalize: read bytes when state gets too small
        while state.needs_renorm_decode() {
            if *pos == 0 {
                return Err(ZiporaError::invalid_data("Insufficient data for decoding"));
            }
            *pos -= 1;
            let byte = *input
                .get(*pos)
                .ok_or_else(|| ZiporaError::invalid_data("Insufficient data for decoding"))?;
            state.state = (state.state << 8) | (byte as u64);
        }

        // Fast symbol lookup using direct table
        let slot = (state.state & (TOTFREQ as u64 - 1)) as usize;
        let symbol = self.decode_table[slot]; // D10.1: in-bounds
        let entry = self.decode_entries[slot];

        // Standard rANS decoding (inverse of encoding)
        let new_state = (entry.freq as u64) * (state.state >> TF_SHIFT) + (entry.bias as u64);
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
        let data_len = encoded_data.len();
        if data_len < 8 {
            return Err(ZiporaError::invalid_data("rANS data too short"));
        }

        // Read initial state from last 8 bytes
        let state_bytes = encoded_data
            .get(data_len - 8..data_len)
            .ok_or_else(|| ZiporaError::invalid_data("rANS data too short"))?;
        let initial_state = u64::from_le_bytes(
            state_bytes
                .try_into()
                .map_err(|_| ZiporaError::invalid_data("rANS state bytes malformed"))?,
        );

        let mut state = Rans64State::from_state(initial_state);
        let mut pos = data_len - 8;
        if output_length <= (1 << 20) {
            let mut result = vec![0u8; output_length];
            for out_slot in &mut result {
                *out_slot = self.decode_symbol(&mut state, encoded_data, &mut pos)?;
            }
            return Ok(result);
        }
        let mut result = Vec::with_capacity(output_length.min(1 << 20));

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
            return Err(ZiporaError::invalid_data(
                "Insufficient data for parallel rANS header",
            ));
        }

        // Read stream states (8 bytes each)
        let mut states = Vec::with_capacity(n_streams);
        let mut pos = 0;
        for _ in 0..n_streams {
            let state_bytes = encoded_data.get(pos..pos + 8).ok_or_else(|| {
                ZiporaError::invalid_data("Insufficient data for parallel rANS header")
            })?;
            let state_value = u64::from_le_bytes(
                state_bytes
                    .try_into()
                    .map_err(|_| ZiporaError::invalid_data("Invalid rANS state bytes"))?,
            );
            states.push(Rans64State::from_state(state_value));
            pos += 8;
        }

        // Read stream data lengths (bounds guaranteed by the min_header_size check above:
        // pos = n_streams * 8 here, and len >= n_streams * 8 + n_streams * 4)
        let mut stream_lengths = Vec::with_capacity(n_streams);
        let mut total_stream_len: usize = 0;
        for _ in 0..n_streams {
            let length_bytes = encoded_data.get(pos..pos + 4).ok_or_else(|| {
                ZiporaError::invalid_data("Insufficient data for parallel rANS header")
            })?;
            let length = u32::from_le_bytes(
                length_bytes
                    .try_into()
                    .map_err(|_| ZiporaError::invalid_data("Invalid rANS length bytes"))?,
            ) as usize;
            total_stream_len = total_stream_len.checked_add(length).ok_or_else(|| {
                ZiporaError::invalid_data("Stream length sum overflow in rANS data")
            })?;
            stream_lengths.push(length);
            pos += 4;
        }

        if pos
            .checked_add(total_stream_len)
            .is_none_or(|end| end > encoded_data.len())
        {
            return Err(ZiporaError::invalid_data("Invalid stream data length"));
        }

        // Extract stream data
        let mut stream_data = Vec::with_capacity(n_streams);
        for &length in &stream_lengths {
            let chunk = encoded_data
                .get(pos..pos + length)
                .ok_or_else(|| ZiporaError::invalid_data("Invalid stream data length"))?;
            stream_data.push(chunk);
            pos += length;
        }

        // Decode each stream independently in interleaved fashion
        let mut result = vec![0u8; output_length];

        if n_streams == 4 {
            let [sd0, sd1, sd2, sd3] = stream_data.as_slice() else {
                unreachable!()
            };
            let [s0, s1, s2, s3] = &mut states[..4] else {
                unreachable!()
            };
            let mut p0 = sd0.len();
            let mut p1 = sd1.len();
            let mut p2 = sd2.len();
            let mut p3 = sd3.len();

            for quad in result.chunks_exact_mut(4) {
                let [o0, o1, o2, o3] = quad else {
                    unreachable!()
                };
                *o0 = self.decode_symbol(s0, sd0, &mut p0)?;
                *o1 = self.decode_symbol(s1, sd1, &mut p1)?;
                *o2 = self.decode_symbol(s2, sd2, &mut p2)?;
                *o3 = self.decode_symbol(s3, sd3, &mut p3)?;
            }

            let rem_start = (output_length / 4) * 4;
            if let Some(tail) = result.get_mut(rem_start..) {
                if let Some(slot) = tail.get_mut(0) {
                    *slot = self.decode_symbol(s0, sd0, &mut p0)?;
                }
                if let Some(slot) = tail.get_mut(1) {
                    *slot = self.decode_symbol(s1, sd1, &mut p1)?;
                }
                if let Some(slot) = tail.get_mut(2) {
                    *slot = self.decode_symbol(s2, sd2, &mut p2)?;
                }
            }
            return Ok(result);
        }

        let mut stream_positions = vec![0usize; n_streams];

        // Initialize stream positions at the end of each stream's data (read backwards)
        for i in 0..n_streams {
            stream_positions[i] = stream_data[i].len();
        }

        // Decode each stream's symbols (in forward order since we encoded in
        // reverse), walking the same interleaved index sequence the encoder
        // used. As on the encode side, the sequence is a strided range and
        // needs no index vector.
        for stream_idx in 0..n_streams {
            let mut stream_pos = stream_positions[stream_idx];

            for output_idx in (stream_idx..output_length).step_by(n_streams) {
                let symbol = self.decode_symbol(
                    &mut states[stream_idx],
                    stream_data[stream_idx],
                    &mut stream_pos,
                )?;
                result[output_idx] = symbol;
            }
        }

        Ok(result)
    }
}

/// Adaptive rANS encoder that selects optimal parallel variant based on data size
pub struct AdaptiveRans64Encoder {
    _bit_ops: BitOps,
}

impl AdaptiveRans64Encoder {
    /// Create new adaptive encoder
    pub fn new() -> Self {
        Self {
            _bit_ops: BitOps::new(),
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

    /// Regression (codereview.md 2026-08 #7): the proportional pass of
    /// normalize_frequencies used the *shrinking* `remaining` budget in
    /// `freq * remaining / total_freq`, so earlier symbols received a
    /// geometrically larger share than later symbols with identical input
    /// frequency. 16 equal-frequency symbols must normalize to (near-)equal
    /// slices of TOTFREQ.
    #[test]
    fn test_normalize_frequencies_uniform_input_stays_uniform() {
        let mut frequencies = [0u32; 256];
        for f in frequencies.iter_mut().take(16) {
            *f = 100;
        }
        let normalized =
            Rans64Encoder::<ParallelX1>::normalize_frequencies(&frequencies, 1600).unwrap();

        let used: Vec<u32> = normalized[..16].to_vec();
        let sum: u32 = normalized.iter().sum();
        assert_eq!(sum, TOTFREQ, "normalized frequencies must sum to TOTFREQ");
        let min = *used.iter().min().unwrap();
        let max = *used.iter().max().unwrap();
        assert!(
            max - min <= 1,
            "equal input frequencies must get equal normalized frequencies, got {:?}",
            used
        );
    }

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

        // Regression: the old assert `q * 5 + r == 1000` was a tautology
        // (r was computed as 1000 - q*5), passing for ANY q. Assert the
        // exact quotient and remainder instead.
        let (q, r) = symbol.fast_div(1000);
        assert_eq!(q, 200);
        assert_eq!(r, 0);
    }

    /// Regression: fast_div applied rcp_shift to an already-final quotient
    /// (a 64-bit reciprocal double-shift), returning ~0 for most inputs.
    /// Cross-check the ryg-scheme reciprocal against hardware division for
    /// every valid freq and a spread of x values including u64::MAX.
    #[test]
    fn test_fast_div_matches_hardware_division() {
        for freq in 1..=TOTFREQ {
            let sym = Rans64Symbol::new(0, freq);
            let f = freq as u64;
            for x in [
                0u64,
                1,
                f - 1,
                f,
                f + 1,
                1000,
                RANS64_L,
                RANS64_L * f,
                u32::MAX as u64,
                u64::MAX / f,
                u64::MAX - 1,
                u64::MAX,
            ] {
                let (q, r) = sym.fast_div(x);
                assert_eq!(q, x / f, "quotient mismatch: freq={} x={}", freq, x);
                assert_eq!(r, x % f, "remainder mismatch: freq={} x={}", freq, x);
            }
        }
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
        assert_eq!(adaptive.select_variant(50), "x1"); // < 73
        assert_eq!(adaptive.select_variant(100), "x2"); // 73 <= x < 5329 
        assert_eq!(adaptive.select_variant(10000), "x4"); // 5329 <= x < 28,372,625
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

    #[test]
    fn test_parallel_rans_truncated_stream_data() {
        let data = b"hello world 1234567890 parallel test data string for testing truncated stream lengths";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        let encoder = Rans64Encoder::<ParallelX4>::new(&frequencies).unwrap();
        let mut encoded = encoder.encode(data).unwrap();

        // Truncate encoded data by removing last 10 bytes
        encoded.truncate(encoded.len() - 10);

        let decoder = Rans64Decoder::<ParallelX4>::new(&encoder);
        let res = decoder.decode(&encoded, data.len());
        assert!(res.is_err());
    }

    #[test]
    fn test_parallel_rans_truncated_header_lengths() {
        let data = b"hello world 1234567890 parallel test data string for testing truncated stream lengths";
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        let encoder = Rans64Encoder::<ParallelX4>::new(&frequencies).unwrap();
        let encoded = encoder.encode(data).unwrap();

        // Truncate inside the header (40 < min_header_size = n_streams * 8 + n_streams * 4 = 48),
        // exercising the min_header_size guard in decode_parallel
        let truncated = &encoded[..40];

        let decoder = Rans64Decoder::<ParallelX4>::new(&encoder);
        let res = decoder.decode(truncated, data.len());
        assert!(res.is_err());
    }

    #[test]
    fn test_normalization_edge_cases_roundtrip() {
        // plan.md 7.4: degenerate frequency distributions must normalize to
        // exactly TOTFREQ and round-trip losslessly.
        // (a) single symbol: normalized freq == TOTFREQ
        // (b) all 256 symbols present: every symbol gets >= 1 slot
        // (c) extreme skew: dominant symbol must not starve the rare one
        let cases: Vec<Vec<u8>> = vec![
            vec![b'a'; 10_000],
            (0..=255u8).cycle().take(4096).collect(),
            {
                let mut v = vec![0u8; 100_000];
                v[50_000] = 255; // 99_999 : 1 skew
                v
            },
        ];

        for (case_idx, data) in cases.iter().enumerate() {
            let mut frequencies = [0u32; 256];
            for &byte in data.iter() {
                frequencies[byte as usize] += 1;
            }

            let encoder = Rans64Encoder::<ParallelX1>::new(&frequencies).unwrap();
            let encoded = encoder.encode(data).unwrap();
            let decoder = Rans64Decoder::<ParallelX1>::new(&encoder);
            let decoded = decoder.decode(&encoded, data.len()).unwrap();
            assert_eq!(data, &decoded, "case {} failed round-trip", case_idx);
        }
    }

    #[test]
    fn test_parallel_round_trip_at_every_stream_offset() {
        // The interleaved index sequences are now computed as strided ranges
        // rather than materialised into per-stream index vectors. Pin the
        // mapping at every remainder of `len % n_streams`, which is where an
        // off-by-one in the stride would show up.
        fn round_trip<P: ParallelVariant>(data: &[u8]) {
            let mut frequencies = [0u32; 256];
            for &b in data {
                frequencies[b as usize] += 1;
            }

            let encoder = Rans64Encoder::<P>::new(&frequencies).unwrap();
            let decoder = Rans64Decoder::<P>::new(&encoder);

            let encoded = encoder.encode(data).unwrap();
            let decoded = decoder.decode(&encoded, data.len()).unwrap();
            assert_eq!(
                decoded,
                data,
                "{} round trip failed at len {}",
                P::NAME,
                data.len()
            );
        }

        // 8..=23 covers every residue class for x1/x2/x4/x8, including the
        // lengths just above and just below a whole number of rounds.
        for len in 8..=23usize {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 % 5) as u8 + b'a').collect();
            round_trip::<ParallelX1>(&data);
            round_trip::<ParallelX2>(&data);
            round_trip::<ParallelX4>(&data);
            round_trip::<ParallelX8>(&data);
        }
    }

    #[test]
    fn test_rans_decode_truncated_data() {
        let mut frequencies = [0u32; 256];
        frequencies[b'a' as usize] = 100;

        let encoder1 = Rans64Encoder::<ParallelX1>::new(&frequencies).unwrap();
        let decoder1 = Rans64Decoder::<ParallelX1>::new(&encoder1);
        assert!(decoder1.decode(&[], 10).is_err());
        assert!(decoder1.decode(&[0u8; 7], 10).is_err());

        let encoder4 = Rans64Encoder::<ParallelX4>::new(&frequencies).unwrap();
        let decoder4 = Rans64Decoder::<ParallelX4>::new(&encoder4);
        assert!(decoder4.decode(&[], 10).is_err());
        assert!(decoder4.decode(&[0u8; 12], 10).is_err());
    }
}
