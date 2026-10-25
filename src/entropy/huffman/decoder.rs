use super::tree::{HuffmanNode, HuffmanTree};
use crate::error::{Result, ZiporaError};

/// Bit stream reader for Huffman decoding
#[derive(Debug)]
pub(crate) struct BitStreamReader<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) current: u64,
    pub(crate) bit_count: usize,
    pub(crate) byte_pos: usize,
}

impl<'a> BitStreamReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            current: 0,
            bit_count: 0,
            byte_pos: 0,
        };
        reader.refill();
        reader
    }

    /// Refill the bit buffer
    #[inline]
    pub(crate) fn refill(&mut self) {
        while self.bit_count <= 56 {
            if let Some(&byte) = self.data.get(self.byte_pos) {
                self.current |= (byte as u64) << self.bit_count;
                self.bit_count += 8;
                self.byte_pos += 1;
            } else {
                break;
            }
        }
    }

    /// Peek at the next `count` bits without consuming them
    #[inline]
    pub(crate) fn peek(&self, count: usize) -> u64 {
        debug_assert!(count <= 64);
        self.current & ((1u64 << count) - 1)
    }

    /// Consume `count` bits
    #[inline]
    pub(crate) fn consume(&mut self, count: usize) {
        debug_assert!(count <= self.bit_count);
        self.current >>= count;
        self.bit_count -= count;
    }

    /// Read `count` bits
    #[inline]
    #[cfg(test)]
    pub(crate) fn read(&mut self, count: usize) -> u64 {
        if count > self.bit_count {
            self.refill();
        }
        let result = self.peek(count);
        self.consume(count);
        result
    }

    /// Check if there are more bits available
    #[inline]
    #[cfg(test)]
    pub(crate) fn has_bits(&self) -> bool {
        self.bit_count > 0 || self.byte_pos < self.data.len()
    }

    /// Get remaining bits
    #[inline]
    #[cfg(test)]
    pub(crate) fn remaining_bits(&self) -> usize {
        self.bit_count + (self.data.len() - self.byte_pos) * 8
    }
}

const FAST_LUT_BITS: u8 = 11;
const FAST_LUT_SIZE: usize = 1 << FAST_LUT_BITS;
const FAST_LUT_MASK: u64 = (FAST_LUT_SIZE as u64) - 1;

/// Flat lookup entry for single-cycle Huffman decoding of codes up to `FAST_LUT_BITS`.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct FastDecEntry {
    pub(crate) symbol: u8,
    pub(crate) bit_len: u8,
}

pub(crate) fn fill_lut_dfs<const N: usize>(
    node: &HuffmanNode,
    code_bits: usize,
    depth: u8,
    lut_bits: u8,
    table: &mut [FastDecEntry; N],
) {
    match node {
        HuffmanNode::Leaf { symbol, .. } => {
            let bit_len = depth.max(1);
            let step = 1usize << (depth as usize);
            let mut idx = code_bits;
            while idx < N {
                table[idx] = FastDecEntry {
                    symbol: *symbol,
                    bit_len,
                };
                idx += step;
            }
        }
        HuffmanNode::Internal { left, right, .. } => {
            if depth < lut_bits {
                fill_lut_dfs(left, code_bits, depth + 1, lut_bits, table);
                fill_lut_dfs(
                    right,
                    code_bits | (1usize << (depth as usize)),
                    depth + 1,
                    lut_bits,
                    table,
                );
            }
        }
    }
}

/// Huffman decoder
#[derive(Debug)]
pub struct HuffmanDecoder {
    tree: HuffmanTree,
    fast_lut: Box<[FastDecEntry; FAST_LUT_SIZE]>,
}

impl HuffmanDecoder {
    /// Create decoder from tree
    pub fn new(tree: HuffmanTree) -> Self {
        let mut fast_lut = Box::new([FastDecEntry::default(); FAST_LUT_SIZE]);
        if let Some(root) = tree.root() {
            fill_lut_dfs(root, 0, 0, FAST_LUT_BITS, &mut fast_lut);
        }
        Self { tree, fast_lut }
    }

    /// Decode Huffman-encoded data
    pub fn decode(&self, encoded_data: &[u8], output_length: usize) -> Result<Vec<u8>> {
        if output_length == 0 {
            return Ok(Vec::new());
        }
        if encoded_data.is_empty() {
            return Err(ZiporaError::invalid_data(
                "Empty encoded data for non-zero output_length",
            ));
        }

        let root = match self.tree.root() {
            Some(root) => root,
            None => return Err(ZiporaError::invalid_data("Empty Huffman tree")),
        };

        let max_plausible = encoded_data.len().saturating_mul(64);
        if output_length > max_plausible {
            return Err(ZiporaError::invalid_data(format!(
                "Implausible output length {} for input length {}",
                output_length,
                encoded_data.len()
            )));
        }

        if let HuffmanNode::Leaf { symbol, .. } = root {
            let code = self
                .tree
                .get_code(*symbol)
                .ok_or_else(|| ZiporaError::invalid_data("Missing leaf code"))?;
            if code.is_empty() {
                return Err(ZiporaError::invalid_data(
                    "Empty code for single-symbol Huffman tree",
                ));
            }
            let mut bit_buf: u64 = 0;
            let mut bits_in_buf: u32 = 0;
            let mut byte_idx: usize = 0;
            for _ in 0..output_length {
                for &expected_bit in code {
                    if bits_in_buf == 0 {
                        if let Some(&b) = encoded_data.get(byte_idx) {
                            bit_buf = b as u64;
                            bits_in_buf = 8;
                            byte_idx += 1;
                        } else {
                            return Err(ZiporaError::invalid_data(format!(
                                "Decoded length < expected {}",
                                output_length
                            )));
                        }
                    }
                    let bit = (bit_buf & 1) != 0;
                    bit_buf >>= 1;
                    bits_in_buf -= 1;
                    if bit != expected_bit {
                        return Err(ZiporaError::invalid_data(
                            "Invalid bit for single-symbol Huffman tree",
                        ));
                    }
                }
            }
            return Ok(vec![*symbol; output_length]);
        }

        let mut result = Vec::with_capacity(output_length);
        let mut bit_buf: u64 = 0;
        let mut bits_in_buf: u32 = 0;
        let mut byte_idx: usize = 0;

        while result.len() < output_length {
            if bits_in_buf <= 32 {
                if let Some(chunk) = encoded_data
                    .get(byte_idx..byte_idx + 4)
                    .and_then(|s| s.first_chunk::<4>())
                {
                    bit_buf |= (u32::from_le_bytes(*chunk) as u64) << bits_in_buf;
                    bits_in_buf += 32;
                    byte_idx += 4;
                } else {
                    while bits_in_buf <= 56 {
                        if let Some(&b) = encoded_data.get(byte_idx) {
                            bit_buf |= (b as u64) << bits_in_buf;
                            bits_in_buf += 8;
                            byte_idx += 1;
                        } else {
                            break;
                        }
                    }
                }
            }

            let entry = self.fast_lut[(bit_buf & FAST_LUT_MASK) as usize];
            let len = entry.bit_len as u32;
            if len > 0 && bits_in_buf >= len {
                result.push(entry.symbol);
                bit_buf >>= len;
                bits_in_buf -= len;
                continue;
            }

            // Fallback to tree walk for codes > FAST_LUT_BITS or end-of-stream partial byte
            let mut current_node = root;
            let mut found = false;
            loop {
                match current_node {
                    HuffmanNode::Leaf { symbol, .. } => {
                        result.push(*symbol);
                        found = true;
                        break;
                    }
                    HuffmanNode::Internal { left, right, .. } => {
                        if bits_in_buf == 0 {
                            if let Some(&b) = encoded_data.get(byte_idx) {
                                bit_buf = b as u64;
                                bits_in_buf = 8;
                                byte_idx += 1;
                            } else {
                                break;
                            }
                        }
                        let bit = (bit_buf & 1) != 0;
                        bit_buf >>= 1;
                        bits_in_buf -= 1;
                        current_node = if bit { right } else { left };
                    }
                }
            }
            if !found {
                break;
            }
        }

        if result.len() != output_length {
            return Err(ZiporaError::invalid_data(format!(
                "Decoded length {} != expected {}",
                result.len(),
                output_length
            )));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::huffman::HuffmanEncoder;

    #[test]
    fn test_huffman_decoder_oversized_claim_rejection() {
        let encoder = HuffmanEncoder::new(b"hello world").unwrap();
        let encoded = encoder.encode(b"hello world").unwrap();
        let tree = encoder.tree().clone();
        let decoder = HuffmanDecoder::new(tree);

        // Claiming 1_000_000 output_length for an 11-byte input should be rejected by plausibility check
        let res = decoder.decode(&encoded, 1_000_000);
        assert!(res.is_err());
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("Implausible output length")
        );
    }
}
