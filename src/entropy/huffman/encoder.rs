use super::tree::HuffmanTree;
use crate::error::{Result, ZiporaError};

/// Huffman encoding symbol - compact representation for fast lookup
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct HuffmanEncSymbol {
    /// Packed bit pattern for this symbol
    pub bits: u16,
    /// Number of bits in the code
    pub bit_count: u16,
}

impl HuffmanEncSymbol {
    /// Create a new encoding symbol
    #[inline(always)]
    pub const fn new(bits: u16, bit_count: u16) -> Self {
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
        debug_assert!(count <= 64);

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

    /// Get current buffer size in bits
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn len_bits(&self) -> usize {
        self.buffer.len() * 8 + self.bit_count
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
