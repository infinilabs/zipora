//! Encoding and decoding operations.

pub mod base64;

pub use base64::{
    encode_base64,
    decode_base64,
    encode_base64_to_buffer,
    decode_base64_from_buffer,
    calculate_encoded_len,
    calculate_decoded_len,
};
