//! Huffman coding implementation
//!
//! This module provides classical Huffman coding for entropy compression, including:
//! - Order-0: Classic Huffman (independent symbols)
//! - Order-1: Context-based Huffman (depends on previous symbol)
//! - Order-2: Context-based Huffman (depends on previous two symbols)
//!
//! Order-1 and Order-2 models provide better compression for data with local dependencies.
//!
//! ## Interleaving Support
//!
//! The module includes explicit interleaving variants (x1/x2/x4/x8) for parallel
//! Huffman Order-1 encoding/decoding. Interleaving splits the input into N independent
//! streams that can be processed in parallel, improving throughput on modern CPUs.

use crate::error::{Result, ZiporaError};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Interleaving factor for parallel Huffman encoding/decoding
///
/// Interleaving splits input data into N independent streams that can be
/// processed in parallel, improving throughput on modern CPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterleavingFactor {
    /// Single stream (no interleaving) - baseline performance
    X1,
    /// 2-way interleaving - modest parallelism
    X2,
    /// 4-way interleaving - good parallelism with AVX2
    X4,
    /// 8-way interleaving - maximum parallelism
    X8,
}

impl InterleavingFactor {
    /// Get the number of parallel streams for this factor
    #[inline]
    pub const fn streams(&self) -> usize {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }

    /// Check if SIMD optimizations are available for this factor
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn has_simd_support(&self) -> bool {
        match self {
            Self::X4 | Self::X8 => is_x86_feature_detected!("avx2"),
            _ => false,
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline]
    pub fn has_simd_support(&self) -> bool {
        false
    }
}

impl Default for InterleavingFactor {
    fn default() -> Self {
        Self::X1
    }
}

/// Bit stream writer for Huffman encoding
///
/// Writes bits in reverse order (most significant bit first) to match
/// the C++ reference implementation's behavior.
#[derive(Debug)]
struct BitStreamWriter {
    buffer: Vec<u8>,
    current: u64,
    bit_count: usize,
}

impl BitStreamWriter {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current: 0,
            bit_count: 0,
        }
    }

    /// Write bits to the stream
    #[inline]
    fn write(&mut self, bits: u64, count: usize) {
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
    fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.buffer.push(self.current as u8);
        }
        self.buffer
    }

    /// Get current buffer size in bits
    #[inline]
    fn len_bits(&self) -> usize {
        self.buffer.len() * 8 + self.bit_count
    }
}

/// Bit stream reader for Huffman decoding
#[derive(Debug)]
struct BitStreamReader<'a> {
    data: &'a [u8],
    current: u64,
    bit_count: usize,
    byte_pos: usize,
}

impl<'a> BitStreamReader<'a> {
    fn new(data: &'a [u8]) -> Self {
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
    fn refill(&mut self) {
        while self.bit_count <= 56 && self.byte_pos < self.data.len() {
            self.current |= (self.data[self.byte_pos] as u64) << self.bit_count;
            self.bit_count += 8;
            self.byte_pos += 1;
        }
    }

    /// Peek at the next `count` bits without consuming them
    #[inline]
    fn peek(&self, count: usize) -> u64 {
        debug_assert!(count <= 64);
        self.current & ((1u64 << count) - 1)
    }

    /// Consume `count` bits
    #[inline]
    fn consume(&mut self, count: usize) {
        debug_assert!(count <= self.bit_count);
        self.current >>= count;
        self.bit_count -= count;
    }

    /// Read `count` bits
    #[inline]
    fn read(&mut self, count: usize) -> u64 {
        if count > self.bit_count {
            self.refill();
        }
        let result = self.peek(count);
        self.consume(count);
        result
    }

    /// Check if there are more bits available
    #[inline]
    fn has_bits(&self) -> bool {
        self.bit_count > 0 || self.byte_pos < self.data.len()
    }

    /// Get remaining bits
    #[inline]
    fn remaining_bits(&self) -> usize {
        self.bit_count + (self.data.len() - self.byte_pos) * 8
    }
}

/// Huffman symbol with code information
#[derive(Debug, Clone, Copy)]
struct HuffmanSymbol {
    bits: u64,  // Changed from u16 to u64 to support longer codes
    bit_count: u8,
}

impl HuffmanSymbol {
    #[inline]
    fn new(bits: u64, bit_count: u8) -> Self {
        Self { bits, bit_count }
    }
}

/// Node in the Huffman tree
#[derive(Debug, Clone, PartialEq, Eq)]
enum HuffmanNode {
    Leaf {
        symbol: u8,
        frequency: u32,
    },
    Internal {
        frequency: u32,
        left: Box<HuffmanNode>,
        right: Box<HuffmanNode>,
    },
}

impl HuffmanNode {
    fn frequency(&self) -> u32 {
        match self {
            HuffmanNode::Leaf { frequency, .. } => *frequency,
            HuffmanNode::Internal { frequency, .. } => *frequency,
        }
    }
}

impl PartialOrd for HuffmanNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HuffmanNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap behavior
        other.frequency().cmp(&self.frequency())
    }
}

/// Huffman tree for encoding and decoding
#[derive(Debug, Clone)]
pub struct HuffmanTree {
    root: Option<HuffmanNode>,
    codes: HashMap<u8, Vec<bool>>,
    max_code_length: usize,
}

impl HuffmanTree {
    /// Build Huffman tree from symbol frequencies
    pub fn from_frequencies(frequencies: &[u32; 256]) -> Result<Self> {
        const MAX_CODE_LENGTH: usize = 64; // Maximum bits that fit in u64

        // Collect symbols with non-zero frequencies
        let mut heap = BinaryHeap::new();
        let mut symbol_count = 0;

        for (symbol, &freq) in frequencies.iter().enumerate() {
            if freq > 0 {
                heap.push(Reverse(HuffmanNode::Leaf {
                    symbol: symbol as u8,
                    frequency: freq,
                }));
                symbol_count += 1;
            }
        }

        if symbol_count == 0 {
            return Ok(Self {
                root: None,
                codes: HashMap::new(),
                max_code_length: 0,
            });
        }

        // Special case: only one symbol
        if symbol_count == 1 {
            let node = heap.pop().unwrap().0;
            if let HuffmanNode::Leaf { symbol, .. } = node {
                let mut codes = HashMap::new();
                codes.insert(symbol, vec![false]); // Use single bit
                return Ok(Self {
                    root: Some(node),
                    codes,
                    max_code_length: 1,
                });
            }
        }

        // Build Huffman tree
        while heap.len() > 1 {
            let left = heap.pop().unwrap().0;
            let right = heap.pop().unwrap().0;

            let merged = HuffmanNode::Internal {
                frequency: left.frequency() + right.frequency(),
                left: Box::new(left),
                right: Box::new(right),
            };

            heap.push(Reverse(merged));
        }

        let root = heap.pop().unwrap().0;
        let mut codes = HashMap::new();
        let mut max_code_length = 0;

        // Generate codes
        Self::generate_codes(&root, Vec::new(), &mut codes, &mut max_code_length);

        // Check if tree is too deep (codes too long for u64)
        if max_code_length > MAX_CODE_LENGTH {
            // For very deep trees with many symbols, use a simpler encoding
            // Assign fixed-length codes instead
            return Self::from_frequencies_fixed_length(frequencies);
        }

        Ok(Self {
            root: Some(root),
            codes,
            max_code_length,
        })
    }

    /// Build tree with fixed-length codes for pathological cases
    fn from_frequencies_fixed_length(frequencies: &[u32; 256]) -> Result<Self> {
        // Count symbols
        let symbols: Vec<u8> = frequencies.iter()
            .enumerate()
            .filter(|(_, f)| **f > 0)
            .map(|(s, _)| s as u8)
            .collect();

        if symbols.is_empty() {
            return Ok(Self {
                root: None,
                codes: HashMap::new(),
                max_code_length: 0,
            });
        }

        // Use 8-bit fixed codes (we have at most 256 symbols)
        let code_length = 8;
        let mut codes = HashMap::new();

        for (i, &symbol) in symbols.iter().enumerate() {
            let mut code = Vec::with_capacity(code_length);
            let mut val = i;
            for _ in 0..code_length {
                code.push((val & 1) != 0);
                val >>= 1;
            }
            codes.insert(symbol, code);
        }

        // Build decoding tree from codes
        let root = Self::build_decoding_tree_from_codes(&codes)?;

        Ok(Self {
            root,
            codes,
            max_code_length: code_length,
        })
    }

    /// Build Huffman tree from data
    pub fn from_data(data: &[u8]) -> Result<Self> {
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }
        Self::from_frequencies(&frequencies)
    }

    /// Generate Huffman codes recursively
    fn generate_codes(
        node: &HuffmanNode,
        code: Vec<bool>,
        codes: &mut HashMap<u8, Vec<bool>>,
        max_length: &mut usize,
    ) {
        match node {
            HuffmanNode::Leaf { symbol, .. } => {
                *max_length = (*max_length).max(code.len());
                codes.insert(*symbol, code);
            }
            HuffmanNode::Internal { left, right, .. } => {
                let mut left_code = code.clone();
                left_code.push(false);
                Self::generate_codes(left, left_code, codes, max_length);

                let mut right_code = code;
                right_code.push(true);
                Self::generate_codes(right, right_code, codes, max_length);
            }
        }
    }

    /// Get the code for a symbol
    pub fn get_code(&self, symbol: u8) -> Option<&Vec<bool>> {
        self.codes.get(&symbol)
    }

    /// Get maximum code length
    pub fn max_code_length(&self) -> usize {
        self.max_code_length
    }

    /// Get the root node for decoding
    fn root(&self) -> Option<&HuffmanNode> {
        self.root.as_ref()
    }

    /// Serialize the tree for storage
    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::new();

        // Store number of symbols
        let symbol_count = self.codes.len() as u16;
        result.extend_from_slice(&symbol_count.to_le_bytes());

        // Store symbol -> code mappings
        for (&symbol, code) in &self.codes {
            result.push(symbol);
            result.push(code.len() as u8);

            // Pack bits into bytes
            let mut bit_index = 0;
            let mut current_byte = 0u8;

            for &bit in code {
                if bit {
                    current_byte |= 1 << bit_index;
                }
                bit_index += 1;

                if bit_index == 8 {
                    result.push(current_byte);
                    current_byte = 0;
                    bit_index = 0;
                }
            }

            // Push remaining bits if any
            if bit_index > 0 {
                result.push(current_byte);
            }
        }

        result
    }

    /// Deserialize the tree from storage
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(ZiporaError::invalid_data("Huffman tree data too short"));
        }

        let symbol_count = u16::from_le_bytes([data[0], data[1]]) as usize;
        let mut codes = HashMap::new();
        let mut max_code_length = 0;
        let mut offset = 2;

        for _ in 0..symbol_count {
            if offset + 2 > data.len() {
                return Err(ZiporaError::invalid_data("Truncated Huffman tree data"));
            }

            let symbol = data[offset];
            let code_length = data[offset + 1] as usize;
            offset += 2;

            max_code_length = max_code_length.max(code_length);

            // Read code bits
            let byte_count = (code_length + 7) / 8;
            if offset + byte_count > data.len() {
                return Err(ZiporaError::invalid_data("Truncated Huffman code data"));
            }

            let mut code = Vec::with_capacity(code_length);

            for i in 0..code_length {
                let byte_offset = i / 8;
                let bit_offset = i % 8;
                let byte_value = data[offset + byte_offset];
                let bit = (byte_value >> bit_offset) & 1 == 1;
                code.push(bit);
            }

            codes.insert(symbol, code);
            offset += byte_count;
        }

        // Build decoding tree directly from codes
        let root = Self::build_decoding_tree_from_codes(&codes)?;

        Ok(Self {
            root,
            codes,
            max_code_length,
        })
    }

    /// Build a decoding tree directly from symbol->code mappings
    fn build_decoding_tree_from_codes(
        codes: &HashMap<u8, Vec<bool>>,
    ) -> Result<Option<HuffmanNode>> {
        if codes.is_empty() {
            return Ok(None);
        }

        // Special case: single symbol
        if codes.len() == 1 {
            let (&symbol, _) = codes.iter().next().unwrap();
            return Ok(Some(HuffmanNode::Leaf {
                symbol,
                frequency: 1, // Dummy frequency for leaf node
            }));
        }

        // Create root as internal node to start
        let mut root = HuffmanNode::Internal {
            frequency: 0,
            left: Box::new(HuffmanNode::Leaf {
                symbol: 0,
                frequency: 0,
            }),
            right: Box::new(HuffmanNode::Leaf {
                symbol: 0,
                frequency: 0,
            }),
        };

        // Insert each symbol->code mapping into the tree
        for (&symbol, code) in codes {
            Self::insert_code_into_tree(&mut root, symbol, code)?;
        }

        Ok(Some(root))
    }

    /// Insert a symbol and its code into the decoding tree
    fn insert_code_into_tree(node: &mut HuffmanNode, symbol: u8, code: &[bool]) -> Result<()> {
        if code.is_empty() {
            // Replace this node with a leaf
            *node = HuffmanNode::Leaf {
                symbol,
                frequency: 1, // Dummy frequency
            };
            return Ok(());
        }

        // Ensure this is an internal node
        match node {
            HuffmanNode::Leaf { frequency: 0, .. } => {
                // This is a placeholder leaf, convert to internal node
                let next_bit = code[0];
                let remaining_code = &code[1..];

                if remaining_code.is_empty() {
                    // Final bit, create leaf and keep placeholder
                    let leaf = HuffmanNode::Leaf {
                        symbol,
                        frequency: 1,
                    };
                    let placeholder = HuffmanNode::Leaf {
                        symbol: 0,
                        frequency: 0,
                    };

                    if next_bit {
                        *node = HuffmanNode::Internal {
                            frequency: 0,
                            left: Box::new(placeholder),
                            right: Box::new(leaf),
                        };
                    } else {
                        *node = HuffmanNode::Internal {
                            frequency: 0,
                            left: Box::new(leaf),
                            right: Box::new(placeholder),
                        };
                    }
                } else {
                    // More bits, create internal structure and continue insertion
                    let placeholder = HuffmanNode::Leaf {
                        symbol: 0,
                        frequency: 0,
                    };

                    if next_bit {
                        // Create internal node with placeholder on left, continue on right
                        let mut right_child = HuffmanNode::Leaf {
                            symbol: 0,
                            frequency: 0,
                        };
                        Self::insert_code_into_tree(&mut right_child, symbol, remaining_code)?;

                        *node = HuffmanNode::Internal {
                            frequency: 0,
                            left: Box::new(placeholder),
                            right: Box::new(right_child),
                        };
                    } else {
                        // Create internal node with placeholder on right, continue on left
                        let mut left_child = HuffmanNode::Leaf {
                            symbol: 0,
                            frequency: 0,
                        };
                        Self::insert_code_into_tree(&mut left_child, symbol, remaining_code)?;

                        *node = HuffmanNode::Internal {
                            frequency: 0,
                            left: Box::new(left_child),
                            right: Box::new(placeholder),
                        };
                    }
                }
                return Ok(());
            }
            HuffmanNode::Leaf { .. } => {
                return Err(ZiporaError::invalid_data(
                    "Code collision: trying to overwrite existing symbol",
                ));
            }
            HuffmanNode::Internal { .. } => {
                // Already internal, continue
            }
        }

        // Navigate to the correct child
        match node {
            HuffmanNode::Internal { left, right, .. } => {
                let next_bit = code[0];
                let remaining_code = &code[1..];

                if next_bit {
                    // Go right
                    Self::insert_code_into_tree(right, symbol, remaining_code)?;
                } else {
                    // Go left
                    Self::insert_code_into_tree(left, symbol, remaining_code)?;
                }
            }
            HuffmanNode::Leaf { .. } => {
                return Err(ZiporaError::invalid_data(
                    "Unexpected leaf node during tree construction",
                ));
            }
        }

        Ok(())
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

        let compressed_bytes = (total_bits + 7) / 8;
        compressed_bytes as f64 / data.len() as f64
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
        if let HuffmanNode::Leaf { symbol, .. } = current_node {
            if result.len() < output_length {
                result.push(*symbol);
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

/// Context-based Huffman encoding models for improved compression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuffmanOrder {
    /// Order-0: Classic Huffman coding (symbols are independent)
    Order0,
    /// Order-1: Symbol frequencies depend on the previous symbol
    Order1,
    /// Order-2: Symbol frequencies depend on the previous two symbols  
    Order2,
}

impl Default for HuffmanOrder {
    fn default() -> Self {
        Self::Order0
    }
}

/// Enhanced Huffman encoder with context-based models
#[derive(Debug)]
pub struct ContextualHuffmanEncoder {
    order: HuffmanOrder,
    /// For Order-0: single tree
    /// For Order-1: 256 trees (one per previous symbol)
    /// For Order-2: 65536 trees (one per previous two symbols)
    trees: Vec<HuffmanTree>,
    /// Context-to-tree index mapping
    context_map: HashMap<u32, usize>,
}

impl ContextualHuffmanEncoder {
    /// Create encoder with specified order from training data
    pub fn new(data: &[u8], order: HuffmanOrder) -> Result<Self> {
        match order {
            HuffmanOrder::Order0 => Self::new_order0(data),
            HuffmanOrder::Order1 => Self::new_order1(data),
            HuffmanOrder::Order2 => Self::new_order2(data),
        }
    }

    /// Create Order-0 encoder (classic Huffman)
    fn new_order0(data: &[u8]) -> Result<Self> {
        let tree = HuffmanTree::from_data(data)?;
        Ok(Self {
            order: HuffmanOrder::Order0,
            trees: vec![tree],
            context_map: {
                let mut map = HashMap::new();
                map.insert(0, 0);
                map
            },
        })
    }

    /// Create Order-1 encoder (depends on previous symbol)
    fn new_order1(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Self::new_order0(data);
        }

        // Get Order-0 frequencies from training data
        let mut order0_freqs = [0u32; 256];
        for &byte in data {
            order0_freqs[byte as usize] += 1;
        }

        // Ensure ALL symbols have at least frequency 1 in Order-0 tree
        // This is critical because Order-0 is used as fallback for unseen contexts
        for freq in &mut order0_freqs {
            if *freq == 0 {
                *freq = 1;
            }
        }

        // Build universal Order-0 tree that can encode ALL symbols
        let order0_tree = HuffmanTree::from_frequencies(&order0_freqs)?;

        let mut trees = vec![order0_tree];
        let mut context_map = HashMap::new();

        // Collect context-dependent frequencies
        let mut context_frequencies: HashMap<u8, [u32; 256]> = HashMap::new();

        for i in 1..data.len() {
            let context = data[i - 1];
            let symbol = data[i];

            let freqs = context_frequencies.entry(context).or_insert([0u32; 256]);
            freqs[symbol as usize] += 1;
        }

        // Build one tree per context that has data
        // IMPORTANT: Merge with Order-0 frequencies so every tree can encode ALL symbols
        for (&context, context_freqs) in &context_frequencies {
            // Merge context-specific frequencies with Order-0 baseline
            // Use context-specific when available, fall back to Order-0 scaled down
            let mut merged_freqs = [0u32; 256];
            let context_total: u32 = context_freqs.iter().sum();

            for symbol in 0..256 {
                if context_freqs[symbol] > 0 {
                    // Use context-specific frequency (much higher weight)
                    merged_freqs[symbol] = context_freqs[symbol] * 100;
                } else if order0_freqs[symbol] > 0 {
                    // Use Order-0 frequency with minimal weight
                    merged_freqs[symbol] = order0_freqs[symbol];
                } else {
                    // Symbol never seen in data - use frequency 1 to ensure it exists in tree
                    merged_freqs[symbol] = 1;
                }
            }

            let symbol_count = merged_freqs.iter().filter(|&&f| f > 0).count();
            if symbol_count > 0 {
                let tree = HuffmanTree::from_frequencies(&merged_freqs)?;
                context_map.insert(context as u32, trees.len());
                trees.push(tree);
            }
        }

        Ok(Self {
            order: HuffmanOrder::Order1,
            trees,
            context_map,
        })
    }

    /// Create Order-2 encoder (depends on previous two symbols)
    fn new_order2(data: &[u8]) -> Result<Self> {
        if data.len() < 3 {
            return Self::new_order1(data);
        }

        // Get Order-0 frequencies from training data
        let mut order0_freqs = [0u32; 256];
        for &byte in data {
            order0_freqs[byte as usize] += 1;
        }

        // Ensure ALL symbols have at least frequency 1 in Order-0 tree
        // This is critical because Order-0 is used as fallback for unseen contexts
        for freq in &mut order0_freqs {
            if *freq == 0 {
                *freq = 1;
            }
        }

        // Build universal Order-0 tree that can encode ALL symbols
        let order0_tree = HuffmanTree::from_frequencies(&order0_freqs)?;
        let mut trees = vec![order0_tree];
        let mut context_map = HashMap::new();

        // Collect context-dependent frequencies
        let mut context_frequencies: HashMap<u16, [u32; 256]> = HashMap::new();

        for i in 2..data.len() {
            let context = ((data[i - 2] as u16) << 8) | (data[i - 1] as u16);
            let symbol = data[i];

            let freqs = context_frequencies.entry(context).or_insert([0u32; 256]);
            freqs[symbol as usize] += 1;
        }

        // Build one tree per context that has data
        // Limit to most frequent contexts to avoid memory explosion
        let mut contexts: Vec<_> = context_frequencies.into_iter().collect();
        contexts.sort_by_key(|(_, freqs)| freqs.iter().sum::<u32>());
        contexts.reverse();

        // Take top 1024 most frequent contexts
        let max_contexts = 1024.min(contexts.len());

        for (context, context_freqs) in contexts.into_iter().take(max_contexts) {
            // Merge context-specific frequencies with Order-0 baseline
            // Use context-specific when available, fall back to Order-0 scaled down
            let mut merged_freqs = [0u32; 256];

            for symbol in 0..256 {
                if context_freqs[symbol] > 0 {
                    // Use context-specific frequency (much higher weight)
                    merged_freqs[symbol] = context_freqs[symbol] * 100;
                } else if order0_freqs[symbol] > 0 {
                    // Use Order-0 frequency with minimal weight
                    merged_freqs[symbol] = order0_freqs[symbol];
                } else {
                    // Symbol never seen in data - use frequency 1 to ensure it exists in tree
                    merged_freqs[symbol] = 1;
                }
            }

            let symbol_count = merged_freqs.iter().filter(|&&f| f > 0).count();
            if symbol_count > 0 {
                let tree = HuffmanTree::from_frequencies(&merged_freqs)?;
                context_map.insert(context as u32, trees.len());
                trees.push(tree);
            }
        }

        Ok(Self {
            order: HuffmanOrder::Order2,
            trees,
            context_map,
        })
    }

    /// Encode data using context-based Huffman coding
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut bits = Vec::new();

        match self.order {
            HuffmanOrder::Order0 => {
                let tree = &self.trees[0];
                for &symbol in data {
                    if let Some(code) = tree.get_code(symbol) {
                        bits.extend_from_slice(code);
                    } else {
                        return Err(ZiporaError::invalid_data(format!(
                            "Symbol {} not in Huffman tree", symbol
                        )));
                    }
                }
            }
            HuffmanOrder::Order1 => {
                // First symbol uses Order-0 tree (trees[0])
                let order0_tree = &self.trees[0];
                if let Some(code) = order0_tree.get_code(data[0]) {
                    bits.extend_from_slice(code);
                } else {
                    return Err(ZiporaError::invalid_data(format!(
                        "First symbol {} not in Order-0 tree", data[0]
                    )));
                }

                // Subsequent symbols use context-dependent trees
                for i in 1..data.len() {
                    let context = data[i - 1] as u32;
                    let symbol = data[i];
                    
                    // Try context-specific tree first
                    if let Some(&tree_idx) = self.context_map.get(&context) {
                        let tree = &self.trees[tree_idx];
                        if let Some(code) = tree.get_code(symbol) {
                            bits.extend_from_slice(code);
                            continue;
                        }
                    }
                    
                    // Fallback to Order-0 tree for unknown contexts/symbols
                    if let Some(code) = self.trees[0].get_code(symbol) {
                        bits.extend_from_slice(code);
                    } else {
                        return Err(ZiporaError::invalid_data(format!(
                            "Symbol {} not found in any tree", symbol
                        )));
                    }
                }
            }
            HuffmanOrder::Order2 => {
                // First two symbols use Order-0 tree (trees[0])
                let order0_tree = &self.trees[0];
                for i in 0..2.min(data.len()) {
                    if let Some(code) = order0_tree.get_code(data[i]) {
                        bits.extend_from_slice(code);
                    } else {
                        return Err(ZiporaError::invalid_data(format!(
                            "Symbol {} not in Order-0 tree", data[i]
                        )));
                    }
                }

                // Subsequent symbols use 2-symbol context
                for i in 2..data.len() {
                    let context = ((data[i - 2] as u32) << 8) | (data[i - 1] as u32);
                    let symbol = data[i];
                    
                    // Try context-specific tree first
                    if let Some(&tree_idx) = self.context_map.get(&context) {
                        let tree = &self.trees[tree_idx];
                        if let Some(code) = tree.get_code(symbol) {
                            bits.extend_from_slice(code);
                            continue;
                        }
                    }
                    
                    // Fallback to Order-0 tree for unknown contexts/symbols
                    if let Some(code) = self.trees[0].get_code(symbol) {
                        bits.extend_from_slice(code);
                    } else {
                        return Err(ZiporaError::invalid_data(format!(
                            "Symbol {} not found in any tree", symbol
                        )));
                    }
                }
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

    /// Get the order of this encoder
    pub fn order(&self) -> HuffmanOrder {
        self.order
    }

    /// Get number of context trees
    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    /// Estimate compression ratio for the given data
    pub fn estimate_compression_ratio(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut total_bits = 0;

        match self.order {
            HuffmanOrder::Order0 => {
                let tree = &self.trees[0];
                for &symbol in data {
                    if let Some(code) = tree.get_code(symbol) {
                        total_bits += code.len();
                    }
                }
            }
            HuffmanOrder::Order1 => {
                // First symbol
                if let Some(tree) = self.trees.first() {
                    if let Some(code) = tree.get_code(data[0]) {
                        total_bits += code.len();
                    }
                }

                // Context-dependent symbols
                for i in 1..data.len() {
                    let context = data[i - 1] as u32;
                    let symbol = data[i];
                    
                    let tree_idx = self.context_map.get(&context).copied().unwrap_or(0);
                    let tree = &self.trees[tree_idx];
                    
                    if let Some(code) = tree.get_code(symbol) {
                        total_bits += code.len();
                    } else if let Some(code) = self.trees[0].get_code(symbol) {
                        total_bits += code.len();
                    }
                }
            }
            HuffmanOrder::Order2 => {
                // First two symbols
                for i in 0..2.min(data.len()) {
                    if let Some(tree) = self.trees.first() {
                        if let Some(code) = tree.get_code(data[i]) {
                            total_bits += code.len();
                        }
                    }
                }

                // Context-dependent symbols
                for i in 2..data.len() {
                    let context = ((data[i - 2] as u32) << 8) | (data[i - 1] as u32);
                    let symbol = data[i];
                    
                    let tree_idx = self.context_map.get(&context).copied().unwrap_or(0);
                    let tree = &self.trees[tree_idx];
                    
                    if let Some(code) = tree.get_code(symbol) {
                        total_bits += code.len();
                    } else if let Some(code) = self.trees[0].get_code(symbol) {
                        total_bits += code.len();
                    }
                }
            }
        }

        let compressed_bytes = (total_bits + 7) / 8;
        compressed_bytes as f64 / data.len() as f64
    }

    /// Serialize the encoder for storage
    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::new();

        // Store order
        result.push(self.order as u8);

        // Store number of trees
        let tree_count = self.trees.len() as u32;
        result.extend_from_slice(&tree_count.to_le_bytes());

        // Store context map
        let context_count = self.context_map.len() as u32;
        result.extend_from_slice(&context_count.to_le_bytes());

        for (&context, &tree_idx) in &self.context_map {
            result.extend_from_slice(&context.to_le_bytes());
            result.extend_from_slice(&(tree_idx as u32).to_le_bytes());
        }

        // Store trees
        for tree in &self.trees {
            let tree_data = tree.serialize();
            result.extend_from_slice(&(tree_data.len() as u32).to_le_bytes());
            result.extend_from_slice(&tree_data);
        }

        result
    }

    /// Deserialize an encoder from storage  
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(ZiporaError::invalid_data("Empty contextual Huffman data"));
        }

        let mut offset = 0;

        // Read order
        let order = match data[offset] {
            0 => HuffmanOrder::Order0,
            1 => HuffmanOrder::Order1,
            2 => HuffmanOrder::Order2,
            _ => return Err(ZiporaError::invalid_data("Invalid Huffman order")),
        };
        offset += 1;

        // Read tree count
        if offset + 4 > data.len() {
            return Err(ZiporaError::invalid_data("Truncated tree count"));
        }
        let tree_count = u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
        offset += 4;

        // Read context map
        if offset + 4 > data.len() {
            return Err(ZiporaError::invalid_data("Truncated context count"));
        }
        let context_count = u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
        offset += 4;

        let mut context_map = HashMap::new();
        for _ in 0..context_count {
            if offset + 8 > data.len() {
                return Err(ZiporaError::invalid_data("Truncated context map"));
            }
            let context = u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
            offset += 4;
            let tree_idx = u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
            offset += 4;
            context_map.insert(context, tree_idx);
        }

        // Read trees
        let mut trees = Vec::with_capacity(tree_count);
        for _ in 0..tree_count {
            if offset + 4 > data.len() {
                return Err(ZiporaError::invalid_data("Truncated tree size"));
            }
            let tree_size = u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
            offset += 4;

            if offset + tree_size > data.len() {
                return Err(ZiporaError::invalid_data("Truncated tree data"));
            }
            let tree_data = &data[offset..offset + tree_size];
            offset += tree_size;

            let tree = HuffmanTree::deserialize(tree_data)?;
            trees.push(tree);
        }

        Ok(Self {
            order,
            trees,
            context_map,
        })
    }

    /// Encode with specified interleaving factor (Order-1 only)
    ///
    /// This splits the input data into N independent streams for parallel processing.
    /// Only works with Order-1 encoding; returns error for other orders.
    pub fn encode_with_interleaving(
        &self,
        data: &[u8],
        factor: InterleavingFactor,
    ) -> Result<Vec<u8>> {
        if self.order != HuffmanOrder::Order1 {
            return Err(ZiporaError::invalid_operation(
                "Interleaving only supported for Order-1 Huffman encoding",
            ));
        }

        if data.is_empty() {
            return Ok(Vec::new());
        }

        match factor {
            InterleavingFactor::X1 => self.encode_xn::<1>(data),
            InterleavingFactor::X2 => self.encode_xn::<2>(data),
            InterleavingFactor::X4 => self.encode_xn::<4>(data),
            InterleavingFactor::X8 => self.encode_xn::<8>(data),
        }
    }

    /// Decode with specified interleaving factor (Order-1 only)
    pub fn decode_with_interleaving(
        &self,
        data: &[u8],
        output_size: usize,
        factor: InterleavingFactor,
    ) -> Result<Vec<u8>> {
        if self.order != HuffmanOrder::Order1 {
            return Err(ZiporaError::invalid_operation(
                "Interleaving only supported for Order-1 Huffman decoding",
            ));
        }

        match factor {
            InterleavingFactor::X1 => self.decode_xn::<1>(data, output_size),
            InterleavingFactor::X2 => self.decode_xn::<2>(data, output_size),
            InterleavingFactor::X4 => self.decode_xn::<4>(data, output_size),
            InterleavingFactor::X8 => self.decode_xn::<8>(data, output_size),
        }
    }

    /// X1 variant - single stream (no interleaving)
    pub fn encode_x1(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.encode_with_interleaving(data, InterleavingFactor::X1)
    }

    /// X2 variant - 2-way interleaving
    pub fn encode_x2(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.encode_with_interleaving(data, InterleavingFactor::X2)
    }

    /// X4 variant - 4-way interleaving
    pub fn encode_x4(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.encode_with_interleaving(data, InterleavingFactor::X4)
    }

    /// X8 variant - 8-way interleaving
    pub fn encode_x8(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.encode_with_interleaving(data, InterleavingFactor::X8)
    }

    /// X1 decode variant
    pub fn decode_x1(&self, data: &[u8], output_size: usize) -> Result<Vec<u8>> {
        self.decode_with_interleaving(data, output_size, InterleavingFactor::X1)
    }

    /// X2 decode variant
    pub fn decode_x2(&self, data: &[u8], output_size: usize) -> Result<Vec<u8>> {
        self.decode_with_interleaving(data, output_size, InterleavingFactor::X2)
    }

    /// X4 decode variant
    pub fn decode_x4(&self, data: &[u8], output_size: usize) -> Result<Vec<u8>> {
        self.decode_with_interleaving(data, output_size, InterleavingFactor::X4)
    }

    /// X8 decode variant
    pub fn decode_x8(&self, data: &[u8], output_size: usize) -> Result<Vec<u8>> {
        self.decode_with_interleaving(data, output_size, InterleavingFactor::X8)
    }

    /// Generic N-way interleaved encoding
    ///
    /// Following the C++ reference implementation algorithm:
    /// 1. Split data into N CONSECUTIVE chunks (not interleaved positions!)
    /// 2. Encode symbols from each chunk in round-robin fashion
    /// 3. Use context from previous symbol in the original data
    ///
    /// NOTE: The C++ uses backward iteration + reverse writer to get forward bitstream.
    /// We use forward iteration + forward writer to achieve the same result.
    fn encode_xn<const N: usize>(&self, data: &[u8]) -> Result<Vec<u8>> {
        debug_assert!(N > 0 && N <= 8);

        if data.is_empty() {
            return Ok(Vec::new());
        }

        // Build symbol table for fast lookups (context -> symbol -> HuffmanSymbol)
        let symbol_table = self.build_symbol_table()?;

        let mut writer = BitStreamWriter::new();
        let record_size = data.len();

        // Calculate stream boundaries (C++ lines 682-700)
        // Each stream gets a consecutive chunk of the input
        let mut stream_starts = [0usize; 8];
        let mut stream_ends = [0usize; 8];

        for n in 0..N {
            let stream_size = record_size / N + if n < record_size % N { 1 } else { 0 };
            stream_starts[n] = if n == 0 { 0 } else { stream_ends[n - 1] };
            stream_ends[n] = stream_starts[n] + stream_size;
        }

        // Verify last stream ends at record_size
        debug_assert_eq!(stream_ends[N - 1], record_size);

        let mut stream_positions = stream_starts;

        // Main encoding loop - process in round-robin FORWARD order
        let mut total_encoded = 0;
        while total_encoded < record_size {
            for n in 0..N {
                if stream_positions[n] >= stream_ends[n] {
                    continue;
                }

                let pos = stream_positions[n];
                let symbol = data[pos];

                // Context is the previous symbol in the ORIGINAL DATA
                // If at the start of this stream, use context 256
                let context = if stream_positions[n] == stream_starts[n] {
                    256u16
                } else {
                    data[pos - 1] as u16
                };

                // Get Huffman code for this symbol in this context
                let code = symbol_table.get(&(context, symbol))
                    .ok_or_else(|| ZiporaError::invalid_data(
                        format!("Symbol {} not found in context {}", symbol, context)
                    ))?;

                writer.write(code.bits as u64, code.bit_count as usize);

                stream_positions[n] += 1;
                total_encoded += 1;
            }
        }

        Ok(writer.finish())
    }

    /// Generic N-way interleaved decoding
    fn decode_xn<const N: usize>(&self, data: &[u8], output_size: usize) -> Result<Vec<u8>> {
        debug_assert!(N > 0 && N <= 8);

        if data.is_empty() {
            return Ok(Vec::new());
        }

        // Build decode table (block-based lookup)
        let decode_table = self.build_decode_table()?;

        let mut reader = BitStreamReader::new(data);

        // Calculate stream boundaries (following C++ algorithm)
        let mut stream_starts = [0usize; 8];
        let mut stream_ends = [0usize; 8];

        for n in 0..N {
            let stream_size = output_size / N + if n < output_size % N { 1 } else { 0 };
            stream_starts[n] = if n == 0 { 0 } else { stream_ends[n - 1] };
            stream_ends[n] = stream_starts[n] + stream_size;
        }

        // Create output buffer with correct size
        let mut output = vec![0u8; output_size];

        // Track context and position for each stream
        let mut contexts = [256u16; 8]; // 256 = initial context
        let mut stream_positions = stream_starts;

        // Decode symbols in round-robin fashion, writing to correct positions
        let mut total_decoded = 0;
        while total_decoded < output_size {
            for n in 0..N {
                if stream_positions[n] >= stream_ends[n] {
                    continue;
                }

                let pos = stream_positions[n];
                let context = contexts[n];

                // Decode one symbol from this stream
                let symbol = self.decode_one_symbol(&mut reader, context, &decode_table)?;

                // Write directly to the correct output position
                output[pos] = symbol;

                // Update context to the decoded symbol
                contexts[n] = symbol as u16;
                stream_positions[n] += 1;
                total_decoded += 1;

                if total_decoded >= output_size {
                    break;
                }
            }
        }

        Ok(output)
    }

    /// Build fast lookup table for encoding: (context, symbol) -> HuffmanSymbol
    fn build_symbol_table(&self) -> Result<HashMap<(u16, u8), HuffmanSymbol>> {
        let mut table = HashMap::new();

        // For each context, build codes for all symbols
        for context in 0..=256u16 {
            // Get the tree for this context
            let tree_idx = if context == 256 {
                0 // Initial context uses Order-0 tree
            } else {
                *self.context_map.get(&(context as u32)).unwrap_or(&0)
            };

            let tree = &self.trees[tree_idx];

            // Build codes for all symbols in this tree
            // NOTE: Since new_order1 now ensures all trees contain all symbols,
            // every symbol should have a code in every tree
            for symbol in 0..=255u8 {
                if let Some(code) = tree.get_code(symbol) {
                    // Convert Vec<bool> to packed bits
                    let mut bits = 0u64;
                    let bit_count = code.len() as u8;

                    if bit_count > 64 {
                        return Err(ZiporaError::invalid_data(
                            format!("Huffman code too long: {} bits", bit_count)
                        ));
                    }

                    for (i, &bit) in code.iter().enumerate() {
                        if bit && i < 64 {
                            bits |= 1u64 << i;
                        }
                    }

                    table.insert((context, symbol), HuffmanSymbol::new(bits, bit_count));
                } else {
                    // This should not happen if new_order1 works correctly
                    return Err(ZiporaError::invalid_data(
                        format!("Symbol {} not found in tree for context {}", symbol, context)
                    ));
                }
            }
        }

        Ok(table)
    }

    /// Build decode table for fast symbol lookup
    /// Uses tree-walking to populate ALL 4096 entries for each context
    ///
    /// IMPORTANT: Mirrors the encoding logic - uses context-specific tree where available,
    /// falls back to Order-0 tree for codes not in context-specific tree
    fn build_decode_table(&self) -> Result<HashMap<u16, Vec<(u64, u8, u8)>>> {
        const BLOCK_BITS: usize = 12;
        let mut table: HashMap<u16, Vec<(u64, u8, u8)>> = HashMap::new();

        // Helper function to build table for a single tree
        let build_tree_table = |tree: &HuffmanTree| -> Vec<(u64, u8, u8)> {
            let mut tree_table = vec![(0u64, 0u8, 0u8); 1 << BLOCK_BITS];

            if let Some(root) = tree.root() {
                for peek_value in 0..(1 << BLOCK_BITS) {
                    let mut current = root;
                    let mut bits_used = 0;
                    let mut code_bits = 0u64;

                    // Walk the tree following the bits in peek_value
                    for bit_pos in 0..BLOCK_BITS {
                        match current {
                            HuffmanNode::Leaf { symbol, .. } => {
                                tree_table[peek_value] = (code_bits, *symbol, bits_used);
                                break;
                            }
                            HuffmanNode::Internal { left, right, .. } => {
                                let bit = (peek_value >> bit_pos) & 1;
                                if bit == 1 {
                                    code_bits |= 1u64 << bit_pos;
                                    current = right;
                                } else {
                                    current = left;
                                }
                                bits_used += 1;
                            }
                        }
                    }

                    // If we ended on a leaf after using all BLOCK_BITS, record it
                    if let HuffmanNode::Leaf { symbol, .. } = current {
                        if tree_table[peek_value].2 == 0 {
                            tree_table[peek_value] = (code_bits, *symbol, bits_used);
                        }
                    }
                }
            }

            tree_table
        };

        for context in 0..=256u16 {
            let tree_idx = if context == 256 {
                0
            } else {
                *self.context_map.get(&(context as u32)).unwrap_or(&0)
            };

            // Build table using the tree for this context
            let context_table = build_tree_table(&self.trees[tree_idx]);
            table.insert(context, context_table);
        }

        Ok(table)
    }

    /// Decode a single symbol from the bit stream
    fn decode_one_symbol(
        &self,
        reader: &mut BitStreamReader,
        context: u16,
        decode_table: &HashMap<u16, Vec<(u64, u8, u8)>>,
    ) -> Result<u8> {
        const BLOCK_BITS: usize = 12;

        // Ensure we have enough bits for table lookup
        if reader.bit_count < BLOCK_BITS {
            reader.refill();
        }

        // If we still don't have enough bits for table lookup, use tree-based decoding
        if reader.bit_count < BLOCK_BITS {
            return self.decode_one_symbol_tree(reader, context);
        }

        let context_table = decode_table.get(&context)
            .ok_or_else(|| ZiporaError::invalid_data(format!("Context {} not found in decode table", context)))?;

        // Peek at BLOCK_BITS
        let peek_bits = reader.peek(BLOCK_BITS);
        let (_, symbol, bit_count) = context_table[peek_bits as usize];

        if bit_count == 0 {
            // Code is longer than BLOCK_BITS, use tree-based decoding
            return self.decode_one_symbol_tree(reader, context);
        }

        // Check if we have enough bits for the symbol found in the table
        // This handles cases where zero-padding at end of stream matches wrong code
        if reader.bit_count < bit_count as usize {
            // Not enough bits for table result, use tree-based decoding instead
            return self.decode_one_symbol_tree(reader, context);
        }

        reader.consume(bit_count as usize);
        reader.refill();

        Ok(symbol)
    }

    /// Decode a single symbol using tree-based decoding (fallback for end-of-stream)
    fn decode_one_symbol_tree(
        &self,
        reader: &mut BitStreamReader,
        context: u16,
    ) -> Result<u8> {
        // Get the tree for this context
        let tree_idx = if context == 256 {
            0
        } else {
            *self.context_map.get(&(context as u32)).unwrap_or(&0)
        };

        let tree = &self.trees[tree_idx];
        let root = tree.root().ok_or_else(|| ZiporaError::invalid_data("Empty tree"))?;

        let mut current = root;

        loop {
            match current {
                HuffmanNode::Leaf { symbol, .. } => {
                    return Ok(*symbol);
                }
                HuffmanNode::Internal { left, right, .. } => {
                    // Need at least 1 bit
                    if reader.bit_count == 0 {
                        reader.refill();
                        if reader.bit_count == 0 {
                            return Err(ZiporaError::invalid_data("Unexpected end of stream"));
                        }
                    }

                    let bit = reader.peek(1) & 1;
                    reader.consume(1);

                    current = if bit == 1 { right } else { left };
                }
            }
        }
    }

}

/// Context-aware Huffman decoder
#[derive(Debug)]
pub struct ContextualHuffmanDecoder {
    encoder: ContextualHuffmanEncoder,
}

impl ContextualHuffmanDecoder {
    /// Create decoder from encoder
    pub fn new(encoder: ContextualHuffmanEncoder) -> Self {
        Self { encoder }
    }

    /// Decode context-based Huffman data  
    pub fn decode(&self, encoded_data: &[u8], output_length: usize) -> Result<Vec<u8>> {
        if encoded_data.is_empty() || output_length == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(output_length);

        match self.encoder.order {
            HuffmanOrder::Order0 => {
                let tree = &self.encoder.trees[0];
                result = self.decode_order0(encoded_data, tree, output_length)?;
            }
            HuffmanOrder::Order1 => {
                result = self.decode_order1(encoded_data, output_length)?;
            }
            HuffmanOrder::Order2 => {
                result = self.decode_order2(encoded_data, output_length)?;
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

    /// Decode Order-0 (classic Huffman)
    fn decode_order0(&self, encoded_data: &[u8], tree: &HuffmanTree, output_length: usize) -> Result<Vec<u8>> {
        let root = tree.root().ok_or_else(|| ZiporaError::invalid_data("Empty tree"))?;
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
                            HuffmanNode::Leaf { .. } => current_node,
                            HuffmanNode::Internal { left, right, .. } => {
                                if bit { right } else { left }
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
        if let HuffmanNode::Leaf { symbol, .. } = current_node {
            if result.len() < output_length {
                result.push(*symbol);
            }
        }

        Ok(result)
    }

    /// Decode Order-1 (context-based)
    fn decode_order1(&self, encoded_data: &[u8], output_length: usize) -> Result<Vec<u8>> {
        if self.encoder.trees.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(output_length);
        let mut byte_idx = 0;
        let mut bit_pos = 0;

        // Decode first symbol with first tree
        let first_tree = &self.encoder.trees[0];
        if let Ok(first_symbol) = self.decode_next_symbol(encoded_data, &mut byte_idx, &mut bit_pos, first_tree) {
            result.push(first_symbol);
        }

        // Decode remaining symbols with context
        while result.len() < output_length && byte_idx < encoded_data.len() {
            let context = *result.last().unwrap() as u32;
            let tree_idx = self.encoder.context_map.get(&context).copied().unwrap_or(0);
            let tree = &self.encoder.trees[tree_idx];
            
            if let Ok(symbol) = self.decode_next_symbol(encoded_data, &mut byte_idx, &mut bit_pos, tree) {
                result.push(symbol);
            } else {
                break;
            }
        }

        Ok(result)
    }

    /// Decode Order-2 (2-symbol context)
    fn decode_order2(&self, encoded_data: &[u8], output_length: usize) -> Result<Vec<u8>> {
        if self.encoder.trees.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(output_length);
        let mut byte_idx = 0;
        let mut bit_pos = 0;

        // Decode first two symbols with first tree
        let first_tree = &self.encoder.trees[0];
        for _ in 0..2.min(output_length) {
            if let Ok(symbol) = self.decode_next_symbol(encoded_data, &mut byte_idx, &mut bit_pos, first_tree) {
                result.push(symbol);
            } else {
                break;
            }
        }

        // Decode remaining symbols with 2-symbol context
        while result.len() < output_length && byte_idx < encoded_data.len() {
            let len = result.len();
            let context = ((result[len - 2] as u32) << 8) | (result[len - 1] as u32);
            let tree_idx = self.encoder.context_map.get(&context).copied().unwrap_or(0);
            let tree = &self.encoder.trees[tree_idx];
            
            if let Ok(symbol) = self.decode_next_symbol(encoded_data, &mut byte_idx, &mut bit_pos, tree) {
                result.push(symbol);
            } else {
                break;
            }
        }

        Ok(result)
    }

    /// Decode next symbol using the original bit processing logic
    fn decode_next_symbol(&self, encoded_data: &[u8], byte_idx: &mut usize, bit_pos: &mut usize, tree: &HuffmanTree) -> Result<u8> {
        let root = tree.root().ok_or_else(|| ZiporaError::invalid_data("Empty tree"))?;
        let mut current_node = root;

        while *byte_idx < encoded_data.len() {
            let byte = encoded_data[*byte_idx];
            
            while *bit_pos < 8 {
                let bit = (byte >> *bit_pos) & 1 == 1;

                match current_node {
                    HuffmanNode::Leaf { symbol, .. } => {
                        return Ok(*symbol);
                    }
                    HuffmanNode::Internal { left, right, .. } => {
                        current_node = if bit { right } else { left };
                        *bit_pos += 1;
                    }
                }
            }

            *bit_pos = 0;
            *byte_idx += 1;
        }

        // Handle final leaf
        if let HuffmanNode::Leaf { symbol, .. } = current_node {
            Ok(*symbol)
        } else {
            Err(ZiporaError::invalid_data("Incomplete symbol"))
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_tree_single_symbol() {
        let mut frequencies = [0u32; 256];
        frequencies[65] = 100; // 'A'

        let tree = HuffmanTree::from_frequencies(&frequencies).unwrap();
        assert_eq!(tree.max_code_length(), 1);
        assert_eq!(tree.get_code(65).unwrap(), &vec![false]);
    }

    #[test]
    fn test_huffman_tree_two_symbols() {
        let mut frequencies = [0u32; 256];
        frequencies[65] = 100; // 'A'
        frequencies[66] = 50; // 'B'

        let tree = HuffmanTree::from_frequencies(&frequencies).unwrap();

        // Should have codes of length 1
        assert!(tree.get_code(65).is_some());
        assert!(tree.get_code(66).is_some());
        assert_eq!(tree.max_code_length(), 1);
    }

    #[test]
    fn test_huffman_encoding_decoding() {
        let data = b"hello world! this is a test message for huffman coding.";

        let encoder = HuffmanEncoder::new(data).unwrap();
        let encoded = encoder.encode(data).unwrap();

        let decoder = HuffmanDecoder::new(encoder.tree().clone());
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_huffman_compression_ratio() {
        let data = b"aaaaaabbbbcccc"; // Highly compressible

        let encoder = HuffmanEncoder::new(data).unwrap();
        let ratio = encoder.estimate_compression_ratio(data);

        // Should achieve good compression
        assert!(ratio < 1.0);
    }

    #[test]
    fn test_huffman_tree_serialization() {
        let data = b"hello world";
        let tree = HuffmanTree::from_data(data).unwrap();

        let serialized = tree.serialize();
        let deserialized = HuffmanTree::deserialize(&serialized).unwrap();

        // Check that codes match
        for (&symbol, code) in &tree.codes {
            assert_eq!(deserialized.get_code(symbol), Some(code));
        }
    }

    #[test]
    fn test_empty_data() {
        let data = b"";
        let encoder = HuffmanEncoder::new(data).unwrap();
        let encoded = encoder.encode(data).unwrap();
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_large_alphabet() {
        // Test with data containing many different symbols
        let data: Vec<u8> = (0..=255).cycle().take(1000).collect();

        let encoder = HuffmanEncoder::new(&data).unwrap();
        let encoded = encoder.encode(&data).unwrap();

        let decoder = HuffmanDecoder::new(encoder.tree().clone());
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data, decoded);
    }

    #[test]
    fn test_huffman_tree_frequencies() {
        let mut frequencies = [0u32; 256];
        frequencies[b'a' as usize] = 45;
        frequencies[b'b' as usize] = 13;
        frequencies[b'c' as usize] = 12;
        frequencies[b'd' as usize] = 16;
        frequencies[b'e' as usize] = 9;
        frequencies[b'f' as usize] = 5;

        let tree = HuffmanTree::from_frequencies(&frequencies).unwrap();

        // Verify that tree creates valid codes for all symbols
        let code_a = tree.get_code(b'a').unwrap();
        let code_f = tree.get_code(b'f').unwrap();

        // Both codes should exist and be non-empty
        assert!(!code_a.is_empty());
        assert!(!code_f.is_empty());

        // The tree should respect Huffman property: average code length is minimized
        // But individual codes may vary due to tie-breaking in tree construction
        let max_length = tree.max_code_length();
        assert!(max_length > 0);
    }

    #[test]
    fn test_contextual_huffman_order0() {
        let data = b"hello world! this is a test message for huffman coding.";

        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order0).unwrap();
        assert_eq!(encoder.order(), HuffmanOrder::Order0);
        assert_eq!(encoder.tree_count(), 1);

        let encoded = encoder.encode(data).unwrap();
        
        let decoder = ContextualHuffmanDecoder::new(encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_contextual_huffman_order1() {
        let data = b"abababab"; // Repetitive pattern that Order-1 should compress well

        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        assert_eq!(encoder.order(), HuffmanOrder::Order1);
        assert!(encoder.tree_count() >= 1);

        let encoded = encoder.encode(data).unwrap();
        
        let decoder = ContextualHuffmanDecoder::new(encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_contextual_huffman_order2() {
        let data = b"abcabcabcabc"; // Repetitive pattern that Order-2 should compress well

        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();
        assert_eq!(encoder.order(), HuffmanOrder::Order2);
        assert!(encoder.tree_count() >= 1);

        let encoded = encoder.encode(data).unwrap();
        
        let decoder = ContextualHuffmanDecoder::new(encoder);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_contextual_huffman_compression_comparison() {
        // Test that all Huffman orders produce valid encodings
        // Note: Since Order-1/2 now include ALL 256 symbols for correctness,
        // compression ratios may be close to 1.0 for small datasets
        let data = b"aaaaabbbbbcccccdddddeeeeefffff"; // More compressible test data

        let encoder0 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order0).unwrap();
        let encoder1 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        let encoder2 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();

        let ratio0 = encoder0.estimate_compression_ratio(data);
        let ratio1 = encoder1.estimate_compression_ratio(data);
        let ratio2 = encoder2.estimate_compression_ratio(data);

        println!("Order-0 ratio: {:.3}", ratio0);
        println!("Order-1 ratio: {:.3}", ratio1);
        println!("Order-2 ratio: {:.3}", ratio2);

        // Order-0 should achieve compression since it only includes seen symbols
        assert!(ratio0 < 1.0, "Order-0 ratio should be < 1.0, got {:.3}", ratio0);

        // Order-1/2 include all symbols for correctness, so just check they don't expand too much
        assert!(ratio1 <= 1.5, "Order-1 ratio too high, got {:.3}", ratio1);
        assert!(ratio2 <= 1.5, "Order-2 ratio too high, got {:.3}", ratio2);

        // Verify round-trip for all orders
        let encoded0 = encoder0.encode(data).unwrap();
        let decoder0 = ContextualHuffmanDecoder::new(encoder0);
        let decoded0 = decoder0.decode(&encoded0, data.len()).unwrap();
        assert_eq!(data.to_vec(), decoded0);

        let encoder1 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        let encoded1 = encoder1.encode(data).unwrap();
        let decoder1 = ContextualHuffmanDecoder::new(encoder1);
        let decoded1 = decoder1.decode(&encoded1, data.len()).unwrap();
        assert_eq!(data.to_vec(), decoded1);

        let encoder2 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();
        let encoded2 = encoder2.encode(data).unwrap();
        let decoder2 = ContextualHuffmanDecoder::new(encoder2);
        let decoded2 = decoder2.decode(&encoded2, data.len()).unwrap();
        assert_eq!(data.to_vec(), decoded2);
    }

    #[test]
    fn test_contextual_huffman_serialization() {
        let data = b"test data for serialization";

        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        let serialized = encoder.serialize();
        
        let deserialized = ContextualHuffmanEncoder::deserialize(&serialized).unwrap();
        
        assert_eq!(encoder.order(), deserialized.order());
        assert_eq!(encoder.tree_count(), deserialized.tree_count());

        // Test that encoding produces same results
        let encoded1 = encoder.encode(data).unwrap();
        let encoded2 = deserialized.encode(data).unwrap();
        assert_eq!(encoded1, encoded2);
    }

    #[test]
    fn test_contextual_huffman_edge_cases() {
        // Test with very short data
        let short_data = b"a";
        let encoder = ContextualHuffmanEncoder::new(short_data, HuffmanOrder::Order2).unwrap();
        // Should fallback to simpler order
        assert!(encoder.order() == HuffmanOrder::Order0 || encoder.order() == HuffmanOrder::Order1);

        // Test with empty data
        let empty_data = b"";
        let encoder = ContextualHuffmanEncoder::new(empty_data, HuffmanOrder::Order1).unwrap();
        let encoded = encoder.encode(empty_data).unwrap();
        assert!(encoded.is_empty());

        // Test with single repeated symbol
        let repeated_data = b"aaaaaaaaaa";
        let encoder = ContextualHuffmanEncoder::new(repeated_data, HuffmanOrder::Order1).unwrap();
        let encoded = encoder.encode(repeated_data).unwrap();
        
        let decoder = ContextualHuffmanDecoder::new(encoder);
        let decoded = decoder.decode(&encoded, repeated_data.len()).unwrap();
        assert_eq!(repeated_data.to_vec(), decoded);
    }

    #[test]
    fn test_huffman_order_enum() {
        assert_eq!(HuffmanOrder::default(), HuffmanOrder::Order0);

        let orders = [HuffmanOrder::Order0, HuffmanOrder::Order1, HuffmanOrder::Order2];
        for order in orders {
            let data = b"test data";
            let encoder = ContextualHuffmanEncoder::new(data, order).unwrap();
            assert_eq!(encoder.order(), order);
        }
    }

    // ==================== Interleaving Tests ====================

    #[test]
    fn test_interleaving_factor_streams() {
        assert_eq!(InterleavingFactor::X1.streams(), 1);
        assert_eq!(InterleavingFactor::X2.streams(), 2);
        assert_eq!(InterleavingFactor::X4.streams(), 4);
        assert_eq!(InterleavingFactor::X8.streams(), 8);
    }

    #[test]
    fn test_interleaving_factor_default() {
        assert_eq!(InterleavingFactor::default(), InterleavingFactor::X1);
    }

    #[test]
    fn test_encode_x1_basic() {
        let data = b"hello world! this is a test for interleaved huffman coding with order-1 context.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        let encoded = encoder.encode_x1(data).unwrap();
        let decoded = encoder.decode_x1(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded, "X1 encode-decode round trip failed");
    }

    #[test]
    fn test_encode_x2_basic() {
        let data = b"hello world! this is a test for x2 interleaved huffman coding.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        let encoded = encoder.encode_x2(data).unwrap();
        let decoded = encoder.decode_x2(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded, "X2 encode-decode round trip failed");
    }

    #[test]
    fn test_encode_x4_basic() {
        let data = b"hello world! this is a test for x4 interleaved huffman coding with more data.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        let encoded = encoder.encode_x4(data).unwrap();
        let decoded = encoder.decode_x4(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded, "X4 encode-decode round trip failed");
    }

    #[test]
    fn test_encode_x8_basic() {
        let data = b"hello world! this is a test for x8 interleaved huffman coding with even more data to test.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        let encoded = encoder.encode_x8(data).unwrap();
        let decoded = encoder.decode_x8(&encoded, data.len()).unwrap();

        assert_eq!(data.to_vec(), decoded, "X8 encode-decode round trip failed");
    }

    #[test]
    fn test_interleaving_all_variants() {
        let data = b"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        // Test all 4 variants
        for factor in [InterleavingFactor::X1, InterleavingFactor::X2,
                      InterleavingFactor::X4, InterleavingFactor::X8] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder.decode_with_interleaving(&encoded, data.len(), factor).unwrap();

            assert_eq!(data.to_vec(), decoded,
                "Round trip failed for {:?} interleaving", factor);
        }
    }

    #[test]
    fn test_interleaving_empty_data() {
        let data = b"";
        let encoder = ContextualHuffmanEncoder::new(b"training data", HuffmanOrder::Order1).unwrap();

        let encoded = encoder.encode_x1(data).unwrap();
        assert!(encoded.is_empty());

        let decoded = encoder.decode_x1(&encoded, 0).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_interleaving_single_byte() {
        let data = b"a";
        let encoder = ContextualHuffmanEncoder::new(b"abcdef", HuffmanOrder::Order1).unwrap();

        for factor in [InterleavingFactor::X1, InterleavingFactor::X2,
                      InterleavingFactor::X4, InterleavingFactor::X8] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder.decode_with_interleaving(&encoded, data.len(), factor).unwrap();

            assert_eq!(data.to_vec(), decoded,
                "Single byte failed for {:?}", factor);
        }
    }

    #[test]
    fn test_interleaving_two_bytes() {
        let data = b"ab";
        let encoder = ContextualHuffmanEncoder::new(b"abcdef", HuffmanOrder::Order1).unwrap();

        for factor in [InterleavingFactor::X1, InterleavingFactor::X2,
                      InterleavingFactor::X4, InterleavingFactor::X8] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder.decode_with_interleaving(&encoded, data.len(), factor).unwrap();

            assert_eq!(data.to_vec(), decoded,
                "Two bytes failed for {:?}", factor);
        }
    }

    #[test]
    fn test_interleaving_power_of_two_sizes() {
        let training_data = b"The quick brown fox jumps over the lazy dog.";
        let encoder = ContextualHuffmanEncoder::new(training_data, HuffmanOrder::Order1).unwrap();

        // Test with sizes that are powers of 2
        for size in [8, 16, 32, 64, 128, 256] {
            let data: Vec<u8> = training_data.iter().cycle().take(size).copied().collect();

            for factor in [InterleavingFactor::X1, InterleavingFactor::X2,
                          InterleavingFactor::X4, InterleavingFactor::X8] {
                let encoded = encoder.encode_with_interleaving(&data, factor).unwrap();
                let decoded = encoder.decode_with_interleaving(&encoded, data.len(), factor).unwrap();

                assert_eq!(data, decoded,
                    "Power-of-2 size {} failed for {:?}", size, factor);
            }
        }
    }

    #[test]
    fn test_interleaving_non_power_of_two_sizes() {
        let training_data = b"The quick brown fox jumps over the lazy dog.";
        let encoder = ContextualHuffmanEncoder::new(training_data, HuffmanOrder::Order1).unwrap();

        // Test with sizes that are NOT powers of 2
        for size in [7, 15, 31, 63, 127, 255] {
            let data: Vec<u8> = training_data.iter().cycle().take(size).copied().collect();

            for factor in [InterleavingFactor::X1, InterleavingFactor::X2,
                          InterleavingFactor::X4, InterleavingFactor::X8] {
                let encoded = encoder.encode_with_interleaving(&data, factor).unwrap();
                let decoded = encoder.decode_with_interleaving(&encoded, data.len(), factor).unwrap();

                assert_eq!(data, decoded,
                    "Non-power-of-2 size {} failed for {:?}", size, factor);
            }
        }
    }

    #[test]
    fn test_interleaving_repeated_symbols() {
        let data = b"aaaaaaaaaaaaaaaa"; // 16 'a's
        let encoder = ContextualHuffmanEncoder::new(b"abc", HuffmanOrder::Order1).unwrap();

        for factor in [InterleavingFactor::X1, InterleavingFactor::X2,
                      InterleavingFactor::X4, InterleavingFactor::X8] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder.decode_with_interleaving(&encoded, data.len(), factor).unwrap();

            assert_eq!(data.to_vec(), decoded,
                "Repeated symbols failed for {:?}", factor);
        }
    }

    #[test]
    fn test_interleaving_alternating_symbols() {
        let data = b"abababababababab"; // Alternating pattern
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        for factor in [InterleavingFactor::X1, InterleavingFactor::X2,
                      InterleavingFactor::X4, InterleavingFactor::X8] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder.decode_with_interleaving(&encoded, data.len(), factor).unwrap();

            assert_eq!(data.to_vec(), decoded,
                "Alternating pattern failed for {:?}", factor);
        }
    }

    #[test]
    fn test_interleaving_all_bytes() {
        // Test with data containing all possible byte values
        let data: Vec<u8> = (0..=255u8).cycle().take(512).collect();
        let encoder = ContextualHuffmanEncoder::new(&data, HuffmanOrder::Order1).unwrap();

        for factor in [InterleavingFactor::X1, InterleavingFactor::X2,
                      InterleavingFactor::X4, InterleavingFactor::X8] {
            let encoded = encoder.encode_with_interleaving(&data, factor).unwrap();
            let decoded = encoder.decode_with_interleaving(&encoded, data.len(), factor).unwrap();

            assert_eq!(data, decoded,
                "All bytes test failed for {:?}", factor);
        }
    }

    #[test]
    fn test_interleaving_large_data() {
        // Test with larger dataset (1KB)
        let base = b"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
        let data: Vec<u8> = base.iter().cycle().take(1024).copied().collect();
        let encoder = ContextualHuffmanEncoder::new(&data, HuffmanOrder::Order1).unwrap();

        for factor in [InterleavingFactor::X1, InterleavingFactor::X2,
                      InterleavingFactor::X4, InterleavingFactor::X8] {
            let encoded = encoder.encode_with_interleaving(&data, factor).unwrap();
            let decoded = encoder.decode_with_interleaving(&encoded, data.len(), factor).unwrap();

            assert_eq!(data, decoded,
                "Large data (1KB) failed for {:?}", factor);
        }
    }

    #[test]
    fn test_interleaving_only_order1() {
        let data = b"test data";

        // Order-0 should fail
        let encoder0 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order0).unwrap();
        assert!(encoder0.encode_with_interleaving(data, InterleavingFactor::X2).is_err());

        // Order-2 should fail
        let encoder2 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();
        assert!(encoder2.encode_with_interleaving(data, InterleavingFactor::X2).is_err());

        // Order-1 should succeed
        let encoder1 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        assert!(encoder1.encode_with_interleaving(data, InterleavingFactor::X2).is_ok());
    }

    #[test]
    fn test_interleaving_compression_ratio() {
        // Test that interleaving produces valid round-trip encoding
        // Note: Since Order-1 trees now include ALL 256 symbols for correctness,
        // compression ratio may be close to 1.0 for small datasets
        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();

        for factor in [InterleavingFactor::X1, InterleavingFactor::X2,
                      InterleavingFactor::X4, InterleavingFactor::X8] {
            let encoded = encoder.encode_with_interleaving(data, factor).unwrap();
            let decoded = encoder.decode_with_interleaving(&encoded, data.len(), factor).unwrap();

            // Verify round-trip correctness
            assert_eq!(data.to_vec(), decoded,
                "Round trip failed for {:?}", factor);

            // Compression ratio should be reasonable (not expanding too much)
            let ratio = encoded.len() as f64 / data.len() as f64;
            assert!(ratio <= 1.2,
                "Compression ratio too high for {:?}, ratio: {:.3}", factor, ratio);
        }
    }

    #[test]
    fn test_bitstream_writer_basic() {
        let mut writer = BitStreamWriter::new();

        // Write 8 bits
        writer.write(0b10101010, 8);
        let result = writer.finish();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0b10101010);
    }

    #[test]
    fn test_bitstream_writer_partial_byte() {
        let mut writer = BitStreamWriter::new();

        // Write 4 bits
        writer.write(0b1010, 4);
        let result = writer.finish();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0b1010);
    }

    #[test]
    fn test_bitstream_writer_multiple_writes() {
        let mut writer = BitStreamWriter::new();

        // Write 4 bits + 4 bits
        writer.write(0b1010, 4);
        writer.write(0b0101, 4);
        let result = writer.finish();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0b01011010); // LSB first
    }

    #[test]
    fn test_bitstream_reader_basic() {
        let data = vec![0b10101010];
        let mut reader = BitStreamReader::new(&data);

        let bits = reader.read(8);
        assert_eq!(bits, 0b10101010);
    }

    #[test]
    fn test_bitstream_reader_partial() {
        let data = vec![0b10101010];
        let mut reader = BitStreamReader::new(&data);

        let first = reader.read(4);
        let second = reader.read(4);

        assert_eq!(first, 0b1010);
        assert_eq!(second, 0b1010);
    }

    #[test]
    fn test_bitstream_roundtrip() {
        let mut writer = BitStreamWriter::new();

        // Write various bit patterns
        writer.write(0b101, 3);
        writer.write(0b11110000, 8);
        writer.write(0b1, 1);
        writer.write(0b111111, 6);

        let data = writer.finish();
        let mut reader = BitStreamReader::new(&data);

        assert_eq!(reader.read(3), 0b101);
        assert_eq!(reader.read(8), 0b11110000);
        assert_eq!(reader.read(1), 0b1);
        assert_eq!(reader.read(6), 0b111111);
    }
}
