//! SIMD-accelerated encoding and decoding operations
//!
//! This module provides high-performance encoding/decoding using the 6-tier SIMD framework.

pub mod base64;
pub mod varint;

// Re-export commonly used items
pub use base64::{
    encode_base64,
    decode_base64,
    encode_base64_to_buffer,
    decode_base64_from_buffer,
    calculate_encoded_len,
    calculate_decoded_len,
};

pub use varint::{
    SimdVarintCodec,
    VarintSimdTier,
    encode_varint_batch,
    decode_varint_batch,
    encode_varint,
    decode_varint,
};
