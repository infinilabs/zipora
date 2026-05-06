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
        while self.bit_count <= 56 && self.byte_pos < self.data.len() {
            self.current |= (self.data[self.byte_pos] as u64) << self.bit_count;
            self.bit_count += 8;
            self.byte_pos += 1;
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub(crate) fn has_bits(&self) -> bool {
        self.bit_count > 0 || self.byte_pos < self.data.len()
    }

    /// Get remaining bits
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn remaining_bits(&self) -> usize {
        self.bit_count + (self.data.len() - self.byte_pos) * 8
    }
}

/// Huffman decoder
#[derive(Debug)]
pub struct HuffmanDecoder {
    tree: HuffmanTree,
}

impl HuffmanDecoder {
    /// Create decoder from tree
    pub fn new(tree: HuffmanTree) -> Self {
        Self { tree }
    }

    /// Decode Huffman-encoded data
    pub fn decode(&self, encoded_data: &[u8], output_length: usize) -> Result<Vec<u8>> {
        if encoded_data.is_empty() || output_length == 0 {
            return Ok(Vec::new());
        }

        let root = match self.tree.root() {
            Some(root) => root,
            None => return Err(ZiporaError::invalid_data("Empty Huffman tree")),
        };

        let mut result = Vec::with_capacity(output_length);
        let mut current_node = root;

        for &byte in encoded_data {
            for bit_pos in 0..8 {
                if result.len() >= output_length {
                    break;
                }

                let bit = (byte >> bit_pos) & 1 == 1;

                match current_node {
                    HuffmanNode::Leaf { symbol, .. } => {
                        result.push(*symbol);
                        current_node = root;
                        // Process the current bit with the reset node
                        current_node = match current_node {
                            HuffmanNode::Leaf { .. } => {
                                // Single symbol tree case
                                current_node
                            }
                            HuffmanNode::Internal { left, right, .. } => {
                                if bit {
                                    right
                                } else {
                                    left
                                }
                            }
                        };
                    }
                    HuffmanNode::Internal { left, right, .. } => {
                        current_node = if bit { right } else { left };
                    }
                }
            }

            if result.len() >= output_length {
                break;
            }
        }

        // Handle final symbol if we're at a leaf
        if let HuffmanNode::Leaf { symbol, .. } = current_node
            && result.len() < output_length
        {
            result.push(*symbol);
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
