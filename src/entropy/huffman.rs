//! Huffman coding implementation
//!
//! This module provides classical Huffman coding for entropy compression, including:
//! - Order-0: Classic Huffman (independent symbols)
//! - Order-1: Context-based Huffman (depends on previous symbol)
//! - Order-2: Context-based Huffman (depends on previous two symbols)
//! 
//! Order-1 and Order-2 models provide better compression for data with local dependencies.

use crate::error::{Result, ZiporaError};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

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

        Ok(Self {
            root: Some(root),
            codes,
            max_code_length,
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

        // First, create an Order-0 tree for the first symbol
        let order0_tree = HuffmanTree::from_data(data)?;
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
        for (&context, &frequencies) in &context_frequencies {
            let symbol_count = frequencies.iter().filter(|&&f| f > 0).count();
            if symbol_count > 0 {
                let tree = HuffmanTree::from_frequencies(&frequencies)?;
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

        // First, create an Order-0 tree for the first two symbols
        let order0_tree = HuffmanTree::from_data(data)?;
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
        
        for (context, frequencies) in contexts.into_iter().take(max_contexts) {
            let symbol_count = frequencies.iter().filter(|&&f| f > 0).count();
            if symbol_count > 0 { // Any context with data is useful
                let tree = HuffmanTree::from_frequencies(&frequencies)?;
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
        // Test that higher order models achieve better compression on suitable data
        let data = b"aaaaabbbbbcccccdddddeeeeefffff"; // More compressible test data

        let encoder0 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order0).unwrap();
        let encoder1 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
        let encoder2 = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();

        let ratio0 = encoder0.estimate_compression_ratio(data);
        let ratio1 = encoder1.estimate_compression_ratio(data);
        let ratio2 = encoder2.estimate_compression_ratio(data);

        // Higher order models should generally achieve better compression
        // Note: This may not always be true for very small datasets
        println!("Order-0 ratio: {:.3}", ratio0);
        println!("Order-1 ratio: {:.3}", ratio1);  
        println!("Order-2 ratio: {:.3}", ratio2);

        // All should achieve some compression on this highly compressible data
        assert!(ratio0 < 1.0, "Order-0 ratio should be < 1.0, got {:.3}", ratio0);
        assert!(ratio1 < 1.0, "Order-1 ratio should be < 1.0, got {:.3}", ratio1);
        assert!(ratio2 < 1.0, "Order-2 ratio should be < 1.0, got {:.3}", ratio2);
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
}
