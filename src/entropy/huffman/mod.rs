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

mod decoder;
mod encoder;
mod interleaved;
#[cfg(test)]
mod tests;
mod tree;

pub use decoder::HuffmanDecoder;
pub use encoder::{HuffmanEncSymbol, HuffmanEncoder};
pub use interleaved::{
    ContextualHuffmanDecoder, ContextualHuffmanEncoder, HuffmanOrder, InterleavingFactor,
};
pub use tree::HuffmanTree;

// Re-export internal types for tests
#[cfg(test)]
pub(crate) use decoder::BitStreamReader;
#[cfg(test)]
pub(crate) use encoder::BitStreamWriter;
