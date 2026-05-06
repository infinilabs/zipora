use crate::error::{Result, ZiporaError};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Huffman symbol with code information
#[derive(Debug, Clone, Copy)]
#[cfg_attr(test, allow(dead_code))]
pub(crate) struct HuffmanSymbol {
    _bits: u64, // Changed from u16 to u64 to support longer codes
    _bit_count: u8,
}

impl HuffmanSymbol {
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn new(bits: u64, bit_count: u8) -> Self {
        Self {
            _bits: bits,
            _bit_count: bit_count,
        }
    }
}

/// Node in the Huffman tree
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HuffmanNode {
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
    pub(crate) fn frequency(&self) -> u32 {
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
    pub(crate) root: Option<HuffmanNode>,
    pub(crate) codes: HashMap<u8, Vec<bool>>,
    pub(crate) max_code_length: usize,
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
            // SAFETY: symbol_count == 1 means we pushed exactly 1 item to heap, so pop() succeeds
            let node = heap.pop().expect("heap non-empty by loop invariant").0;
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
            // SAFETY: while loop condition guarantees heap.len() >= 2, so both pops succeed
            let left = heap.pop().expect("heap has >= 2 nodes").0;
            let right = heap.pop().expect("heap has >= 2 nodes").0;

            let merged = HuffmanNode::Internal {
                frequency: left.frequency() + right.frequency(),
                left: Box::new(left),
                right: Box::new(right),
            };

            heap.push(Reverse(merged));
        }

        // SAFETY: After while loop, exactly 1 element remains (started with >=2, each iteration removes 2, adds 1)
        let root = heap.pop().expect("heap has final root").0;
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
        let symbols: Vec<u8> = frequencies
            .iter()
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
    pub(crate) fn root(&self) -> Option<&HuffmanNode> {
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
            let byte_count = code_length.div_ceil(8);
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
            // SAFETY: codes.len() == 1 guarantees iter().next() returns Some
            let (&symbol, _) = codes.iter().next().expect("at least one symbol in codes");
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
