use super::decoder::BitStreamReader;
use super::encoder::{BitStreamWriter, HuffmanEncSymbol};
use super::tree::{HuffmanNode, HuffmanSymbol, HuffmanTree};
use crate::error::{Result, ZiporaError};
use std::sync::OnceLock;
use std::collections::HashMap;

/// Interleaving factor for parallel Huffman encoding/decoding
///
/// Interleaving splits input data into N independent streams that can be
/// processed in parallel, improving throughput on modern CPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterleavingFactor {
    /// Single stream (no interleaving) - baseline performance
    #[default]
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

/// Fast symbol table for O(1) lookup: syms[context][symbol]
/// - Context 0-255: previous byte value
/// - Context 256: initial context (no previous byte)
/// Total: 257 contexts × 256 symbols = 65,792 entries
/// This replaces the slow HashMap<(u16, u8), HuffmanSymbol>
type FastSymbolTable = Box<[[HuffmanEncSymbol; 256]; 257]>;

/// Context-based Huffman encoding models for improved compression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HuffmanOrder {
    /// Order-0: Classic Huffman coding (symbols are independent)
    #[default]
    Order0,
    /// Order-1: Symbol frequencies depend on the previous symbol
    Order1,
    /// Order-2: Symbol frequencies depend on the previous two symbols  
    Order2,
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
    /// Cached fast symbol table for O(1) lookup (built lazily on first use)
    /// 257 contexts × 256 symbols = 65,792 entries (~263KB)
    /// Uses OnceLock for thread-safe lazy initialization
    fast_symbol_table: OnceLock<FastSymbolTable>,
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
            fast_symbol_table: OnceLock::new(),
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
            fast_symbol_table: OnceLock::new(),
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
            fast_symbol_table: OnceLock::new(),
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
                            "Symbol {} not in Huffman tree",
                            symbol
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
                        "First symbol {} not in Order-0 tree",
                        data[0]
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
                            "Symbol {} not found in any tree",
                            symbol
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
                            "Symbol {} not in Order-0 tree",
                            data[i]
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
                            "Symbol {} not found in any tree",
                            symbol
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
                if let Some(tree) = self.trees.first()
                    && let Some(code) = tree.get_code(data[0])
                {
                    total_bits += code.len();
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
                    if let Some(tree) = self.trees.first()
                        && let Some(code) = tree.get_code(data[i])
                    {
                        total_bits += code.len();
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

        let compressed_bytes = total_bits.div_ceil(8);
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
        let tree_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        // Read context map
        if offset + 4 > data.len() {
            return Err(ZiporaError::invalid_data("Truncated context count"));
        }
        let context_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut context_map = HashMap::new();
        for _ in 0..context_count {
            if offset + 8 > data.len() {
                return Err(ZiporaError::invalid_data("Truncated context map"));
            }
            let context = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;
            let tree_idx = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            context_map.insert(context, tree_idx);
        }

        // Read trees
        let mut trees = Vec::with_capacity(tree_count);
        for _ in 0..tree_count {
            if offset + 4 > data.len() {
                return Err(ZiporaError::invalid_data("Truncated tree size"));
            }
            let tree_size = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
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
            fast_symbol_table: OnceLock::new(),
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
    ///
    /// ## Optimization: Fast Symbol Table
    ///
    /// Uses a 257×256 array for O(1) symbol lookup instead of HashMap.
    /// The batched loop structure enables some ILP benefits from:
    /// 1. Grouping symbol lookups (prefetchable memory accesses)
    /// 2. Grouping position updates (predictable branches)
    fn encode_xn<const N: usize>(&self, data: &[u8]) -> Result<Vec<u8>> {
        debug_assert!(N > 0 && N <= 8);

        if data.is_empty() {
            return Ok(Vec::new());
        }

        // Get cached fast symbol table for O(1) lookup (built lazily, reused across calls)
        let syms = self.get_or_init_fast_symbol_table();

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

        debug_assert_eq!(stream_ends[N - 1], record_size);

        // Track context and position for each stream
        let mut contexts = [256usize; 8]; // 256 = initial context
        let mut positions = stream_starts;

        // Main encoding loop - process in round-robin FORWARD order
        // Uses fast O(1) array lookup instead of HashMap
        let mut total_encoded = 0;
        while total_encoded < record_size {
            for n in 0..N {
                if positions[n] >= stream_ends[n] {
                    continue;
                }

                let pos = positions[n];
                let symbol = data[pos] as usize;
                let context = contexts[n];

                // O(1) array lookup instead of HashMap
                let code = syms[context][symbol];

                // Write bits using same format as original encoder
                writer.write(code.bits as u64, code.bit_count as usize);

                // Update context to current symbol
                contexts[n] = symbol;
                positions[n] += 1;
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
    #[allow(dead_code)]
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
                        return Err(ZiporaError::invalid_data(format!(
                            "Huffman code too long: {} bits",
                            bit_count
                        )));
                    }

                    for (i, &bit) in code.iter().enumerate() {
                        if bit && i < 64 {
                            bits |= 1u64 << i;
                        }
                    }

                    table.insert((context, symbol), HuffmanSymbol::new(bits, bit_count));
                } else {
                    // This should not happen if new_order1 works correctly
                    return Err(ZiporaError::invalid_data(format!(
                        "Symbol {} not found in tree for context {}",
                        symbol, context
                    )));
                }
            }
        }

        Ok(table)
    }

    /// Get or initialize the cached fast symbol table
    ///
    /// The table is built lazily on first access and cached for subsequent calls.
    /// This amortizes the ~263KB allocation cost across multiple encode calls.
    fn get_or_init_fast_symbol_table(&self) -> &FastSymbolTable {
        self.fast_symbol_table
            .get_or_init(|| self.build_fast_symbol_table_inner())
    }

    /// Build fast symbol table for ILP-optimized encoding
    ///
    /// Creates a 257×256 array for O(1) lookup: table[context][symbol]
    /// - Context 0-255: previous byte value
    /// - Context 256: initial context (no previous byte)
    ///
    /// This replaces the slow HashMap<(u16, u8), HuffmanSymbol> lookup with
    /// direct array indexing, enabling better ILP when batching operations.
    fn build_fast_symbol_table_inner(&self) -> FastSymbolTable {
        // Allocate 257 * 256 * 4 = ~263KB table
        let mut table: FastSymbolTable = Box::new([[HuffmanEncSymbol::default(); 256]; 257]);

        // For each context, build codes for all symbols
        for context in 0..=256usize {
            // Get the tree for this context
            let tree_idx = if context == 256 {
                0 // Initial context uses Order-0 tree
            } else {
                *self.context_map.get(&(context as u32)).unwrap_or(&0)
            };

            let tree = &self.trees[tree_idx];

            // Build codes for all symbols in this tree
            for symbol in 0..=255u8 {
                if let Some(code) = tree.get_code(symbol) {
                    // Convert Vec<bool> to packed bits
                    let mut bits = 0u16;
                    let bit_count = code.len() as u16;

                    // Safety: Huffman codes should not exceed 16 bits for byte alphabets
                    // If they do, we truncate (very rare edge case)
                    let safe_bit_count = bit_count.min(16);

                    for (i, &bit) in code.iter().take(16).enumerate() {
                        if bit {
                            bits |= 1u16 << i;
                        }
                    }

                    table[context][symbol as usize] = HuffmanEncSymbol::new(bits, safe_bit_count);
                } else {
                    // Symbol not in tree - use a default placeholder
                    // This should not happen with properly built trees
                    table[context][symbol as usize] = HuffmanEncSymbol::new(0, 1);
                }
            }
        }

        table
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
                    if let HuffmanNode::Leaf { symbol, .. } = current
                        && tree_table[peek_value].2 == 0
                    {
                        tree_table[peek_value] = (code_bits, *symbol, bits_used);
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

        let context_table = decode_table.get(&context).ok_or_else(|| {
            ZiporaError::invalid_data(format!("Context {} not found in decode table", context))
        })?;

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
    fn decode_one_symbol_tree(&self, reader: &mut BitStreamReader, context: u16) -> Result<u8> {
        // Get the tree for this context
        let tree_idx = if context == 256 {
            0
        } else {
            *self.context_map.get(&(context as u32)).unwrap_or(&0)
        };

        let tree = &self.trees[tree_idx];
        let root = tree
            .root()
            .ok_or_else(|| ZiporaError::invalid_data("Empty tree"))?;

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

        let result = match self.encoder.order {
            HuffmanOrder::Order0 => {
                let tree = &self.encoder.trees[0];
                self.decode_order0(encoded_data, tree, output_length)?
            }
            HuffmanOrder::Order1 => self.decode_order1(encoded_data, output_length)?,
            HuffmanOrder::Order2 => self.decode_order2(encoded_data, output_length)?,
        };

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
    fn decode_order0(
        &self,
        encoded_data: &[u8],
        tree: &HuffmanTree,
        output_length: usize,
    ) -> Result<Vec<u8>> {
        let root = tree
            .root()
            .ok_or_else(|| ZiporaError::invalid_data("Empty tree"))?;
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
        if let Ok(first_symbol) =
            self.decode_next_symbol(encoded_data, &mut byte_idx, &mut bit_pos, first_tree)
        {
            result.push(first_symbol);
        }

        // Decode remaining symbols with context
        while result.len() < output_length && byte_idx < encoded_data.len() {
            // SAFETY: First symbol pushed at line 1862 before loop, so result is always non-empty
            let context = *result.last().expect("result non-empty after push") as u32;
            let tree_idx = self.encoder.context_map.get(&context).copied().unwrap_or(0);
            let tree = &self.encoder.trees[tree_idx];

            if let Ok(symbol) =
                self.decode_next_symbol(encoded_data, &mut byte_idx, &mut bit_pos, tree)
            {
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
            if let Ok(symbol) =
                self.decode_next_symbol(encoded_data, &mut byte_idx, &mut bit_pos, first_tree)
            {
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

            if let Ok(symbol) =
                self.decode_next_symbol(encoded_data, &mut byte_idx, &mut bit_pos, tree)
            {
                result.push(symbol);
            } else {
                break;
            }
        }

        Ok(result)
    }

    /// Decode next symbol using the original bit processing logic
    fn decode_next_symbol(
        &self,
        encoded_data: &[u8],
        byte_idx: &mut usize,
        bit_pos: &mut usize,
        tree: &HuffmanTree,
    ) -> Result<u8> {
        let root = tree
            .root()
            .ok_or_else(|| ZiporaError::invalid_data("Empty tree"))?;
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

