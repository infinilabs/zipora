use super::tree::HuffmanTree;
use crate::error::{Result, ZiporaError};

/// Huffman encoding symbol - compact representation for fast lookup
///
/// `bits` is 32 bits wide so that every code a `HuffmanTree` can emit in
/// practice fits without truncation. A `bit_count` of 0 is the sentinel for
/// "this code is too long to represent here"; the encoder rejects such a
/// symbol rather than silently writing a truncated, non-prefix-free code.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct HuffmanEncSymbol {
    /// Packed bit pattern for this symbol (LSB-first)
    pub bits: u32,
    /// Number of bits in the code, or 0 if the code is unrepresentable
    pub bit_count: u16,
}

impl HuffmanEncSymbol {
    /// Widest code `bits` can hold.
    pub const MAX_BITS: usize = 32;

    /// Create a new encoding symbol
    #[inline(always)]
    pub const fn new(bits: u32, bit_count: u16) -> Self {
        Self { bits, bit_count }
    }
}

/// Bit stream writer for Huffman encoding
///
/// Writes bits in reverse order (most significant bit first) to match
/// the C++ reference implementation's behavior.
#[derive(Debug)]
pub(crate) struct BitStreamWriter {
    pub(crate) buffer: Vec<u8>,
    pub(crate) current: u64,
    pub(crate) bit_count: usize,
}

impl BitStreamWriter {
    pub(crate) fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current: 0,
            bit_count: 0,
        }
    }

    /// Write bits to the stream
    #[inline]
    pub(crate) fn write(&mut self, bits: u64, count: usize) {
        // `bit_count` is always in 0..=7 here (the flush loop below drains
        // whole bytes), so `bits << bit_count` silently drops high bits once
        // `count` passes 57. Callers must stay within that budget.
        debug_assert!(
            count + self.bit_count <= 64,
            "code of {count} bits does not fit above {} buffered bits",
            self.bit_count
        );

        self.current |= bits << self.bit_count;
        self.bit_count += count;

        // Flush complete bytes
        while self.bit_count >= 8 {
            self.buffer.push(self.current as u8);
            self.current >>= 8;
            self.bit_count -= 8;
        }
    }

    /// Flush remaining bits and return the buffer
    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.buffer.push(self.current as u8);
        }
        self.buffer
    }
}

/// Huffman encoder
#[derive(Debug)]
pub struct HuffmanEncoder {
    tree: HuffmanTree,
}

impl HuffmanEncoder {
    /// Create encoder from data
    pub fn new(data: &[u8]) -> Result<Self> {
        let tree = HuffmanTree::from_data(data)?;
        Ok(Self { tree })
    }

    /// Create encoder from frequencies
    pub fn from_frequencies(frequencies: &[u32; 256]) -> Result<Self> {
        let tree = HuffmanTree::from_frequencies(frequencies)?;
        Ok(Self { tree })
    }

    /// Encode data using Huffman coding
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut bits = Vec::new();

        // Encode each symbol
        for &symbol in data {
            if let Some(code) = self.tree.get_code(symbol) {
                bits.extend_from_slice(code);
            } else {
                return Err(ZiporaError::invalid_data(format!(
                    "Symbol {} not in Huffman tree",
                    symbol
                )));
            }
        }

        // Pack bits into bytes
        let mut result = Vec::new();
        let mut current_byte = 0u8;
        let mut bit_count = 0;

        for bit in bits {
            if bit {
                current_byte |= 1 << bit_count;
            }
            bit_count += 1;

            if bit_count == 8 {
                result.push(current_byte);
                current_byte = 0;
                bit_count = 0;
            }
        }

        // Add remaining bits if any
        if bit_count > 0 {
            result.push(current_byte);
        }

        Ok(result)
    }

    /// Get the Huffman tree
    pub fn tree(&self) -> &HuffmanTree {
        &self.tree
    }

    /// Estimate compression ratio
    pub fn estimate_compression_ratio(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut total_bits = 0;
        for &symbol in data {
            if let Some(code) = self.tree.get_code(symbol) {
                total_bits += code.len();
            }
        }

        let compressed_bytes = total_bits.div_ceil(8);
        compressed_bytes as f64 / data.len() as f64
    }
}
