//! Variable-length integer encoding
//!
//! This module provides efficient variable-length integer encoding using the LEB128
//! (Little Endian Base 128) format. This encoding is space-efficient for small integers
//! while still supporting the full range of 64-bit values.

use crate::error::{Result, ToplingError};
use crate::io::data_input::DataInput;
use std::io::Write;

/// Utility struct for variable-length integer encoding/decoding
pub struct VarInt;

impl VarInt {
    /// Maximum number of bytes needed to encode a u64 as a varint
    pub const MAX_ENCODED_LEN: usize = 10;

    /// Write a u64 value as a variable-length integer to a Write implementation
    ///
    /// # Arguments
    /// * `writer` - The writer to write to
    /// * `value` - The value to encode
    ///
    /// # Returns
    /// * `Ok(usize)` - Number of bytes written
    /// * `Err(ToplingError)` - If writing fails
    pub fn write_to<W: Write>(writer: &mut W, mut value: u64) -> Result<usize> {
        let mut bytes_written = 0;

        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;

            if value != 0 {
                byte |= 0x80; // Set continuation bit
            }

            writer.write_all(&[byte]).map_err(|e| {
                ToplingError::io_error(format!("Failed to write varint byte: {}", e))
            })?;

            bytes_written += 1;

            if value == 0 {
                break;
            }
        }

        Ok(bytes_written)
    }

    /// Write a u64 value as a variable-length integer to a Vec<u8>
    ///
    /// # Arguments
    /// * `buffer` - The buffer to write to
    /// * `value` - The value to encode
    ///
    /// # Returns
    /// * `Ok(usize)` - Number of bytes written
    /// * `Err(ToplingError)` - If writing fails
    pub fn write_to_vec(buffer: &mut Vec<u8>, mut value: u64) -> Result<usize> {
        let mut bytes_written = 0;

        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;

            if value != 0 {
                byte |= 0x80; // Set continuation bit
            }

            buffer.push(byte);
            bytes_written += 1;

            if value == 0 {
                break;
            }
        }

        Ok(bytes_written)
    }

    /// Encode a u64 value as a variable-length integer and return the bytes
    ///
    /// # Arguments
    /// * `value` - The value to encode
    ///
    /// # Returns
    /// * Vec<u8> containing the encoded bytes
    pub fn encode(value: u64) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(Self::MAX_ENCODED_LEN);
        Self::write_to_vec(&mut buffer, value).unwrap(); // Vec<u8> writes cannot fail
        buffer
    }

    /// Read a variable-length integer from a DataInput implementation
    ///
    /// # Arguments
    /// * `reader` - The reader to read from
    ///
    /// # Returns
    /// * `Ok(u64)` - The decoded value
    /// * `Err(ToplingError)` - If reading fails or encoding is invalid
    pub fn read_from<R: DataInput>(reader: &mut R) -> Result<u64> {
        let mut result = 0u64;
        let mut shift = 0;

        for _ in 0..Self::MAX_ENCODED_LEN {
            let byte = reader.read_u8()?;

            // Check for overflow
            if shift >= 64 {
                return Err(ToplingError::invalid_data("Varint too long"));
            }

            result |= ((byte & 0x7F) as u64) << shift;

            // If continuation bit is not set, we're done
            if (byte & 0x80) == 0 {
                return Ok(result);
            }

            shift += 7;
        }

        Err(ToplingError::invalid_data("Varint too long"))
    }

    /// Decode a variable-length integer from a byte slice
    ///
    /// # Arguments
    /// * `data` - The byte slice to decode from
    ///
    /// # Returns
    /// * `Ok((value, bytes_consumed))` - The decoded value and number of bytes consumed
    /// * `Err(ToplingError)` - If decoding fails or encoding is invalid
    pub fn decode(data: &[u8]) -> Result<(u64, usize)> {
        let mut result = 0u64;
        let mut shift = 0;

        for (i, &byte) in data.iter().enumerate() {
            if i >= Self::MAX_ENCODED_LEN {
                return Err(ToplingError::invalid_data("Varint too long"));
            }

            // Check for overflow
            if shift >= 64 {
                return Err(ToplingError::invalid_data("Varint too long"));
            }

            result |= ((byte & 0x7F) as u64) << shift;

            // If continuation bit is not set, we're done
            if (byte & 0x80) == 0 {
                return Ok((result, i + 1));
            }

            shift += 7;
        }

        Err(ToplingError::invalid_data("Incomplete varint"))
    }

    /// Calculate the number of bytes needed to encode a value
    ///
    /// # Arguments
    /// * `value` - The value to encode
    ///
    /// # Returns
    /// * usize - Number of bytes needed
    pub fn encoded_len(mut value: u64) -> usize {
        if value == 0 {
            return 1;
        }

        let mut len = 0;
        while value > 0 {
            len += 1;
            value >>= 7;
        }
        len
    }

    /// Check if a value can be encoded in a single byte
    ///
    /// # Arguments
    /// * `value` - The value to check
    ///
    /// # Returns
    /// * bool - True if the value fits in a single byte
    pub fn fits_in_one_byte(value: u64) -> bool {
        value < 128
    }

    /// Check if a value can be encoded in two bytes
    ///
    /// # Arguments
    /// * `value` - The value to check
    ///
    /// # Returns
    /// * bool - True if the value fits in two bytes
    pub fn fits_in_two_bytes(value: u64) -> bool {
        value < 16384 // 2^14
    }

    /// Encode multiple values efficiently
    ///
    /// # Arguments
    /// * `values` - Iterator over values to encode
    ///
    /// # Returns
    /// * Vec<u8> containing all encoded values
    pub fn encode_multiple<I>(values: I) -> Vec<u8>
    where
        I: IntoIterator<Item = u64>,
    {
        let mut buffer = Vec::new();
        for value in values {
            Self::write_to_vec(&mut buffer, value).unwrap();
        }
        buffer
    }

    /// Decode multiple values from a byte slice
    ///
    /// # Arguments
    /// * `data` - The byte slice to decode from
    ///
    /// # Returns
    /// * `Ok(Vec<u64>)` - All decoded values
    /// * `Err(ToplingError)` - If decoding fails
    pub fn decode_multiple(mut data: &[u8]) -> Result<Vec<u64>> {
        let mut values = Vec::new();

        while !data.is_empty() {
            let (value, consumed) = Self::decode(data)?;
            values.push(value);
            data = &data[consumed..];
        }

        Ok(values)
    }
}

/// Extension trait for encoding signed integers as varints
pub trait SignedVarInt {
    /// Encode a signed integer using zigzag encoding
    fn encode_signed(value: i64) -> Vec<u8>;

    /// Decode a signed integer using zigzag encoding
    fn decode_signed(data: &[u8]) -> Result<(i64, usize)>;
}

impl SignedVarInt for VarInt {
    fn encode_signed(value: i64) -> Vec<u8> {
        // Zigzag encoding: maps signed integers to unsigned integers
        // Positive: 0, 2, 4, 6, ...
        // Negative: 1, 3, 5, 7, ...
        let encoded = ((value << 1) ^ (value >> 63)) as u64;
        Self::encode(encoded)
    }

    fn decode_signed(data: &[u8]) -> Result<(i64, usize)> {
        let (encoded, consumed) = Self::decode(data)?;
        // Reverse zigzag encoding
        let value = ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64));
        Ok((value, consumed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::data_input::SliceDataInput;

    #[test]
    fn test_varint_basic_encoding() {
        let test_cases = [
            (0, vec![0]),
            (1, vec![1]),
            (127, vec![127]),
            (128, vec![128, 1]),
            (255, vec![255, 1]),
            (300, vec![172, 2]),
            (16384, vec![128, 128, 1]),
        ];

        for (value, expected) in test_cases {
            let encoded = VarInt::encode(value);
            assert_eq!(encoded, expected, "Failed encoding {}", value);

            let (decoded, consumed) = VarInt::decode(&encoded).unwrap();
            assert_eq!(decoded, value, "Failed decoding {}", value);
            assert_eq!(consumed, encoded.len(), "Wrong consumption for {}", value);
        }
    }

    #[test]
    fn test_varint_large_values() {
        let test_values = [
            u64::MAX,
            u64::MAX / 2,
            1u64 << 32,
            1u64 << 20,
            1u64 << 14,
            1u64 << 7,
        ];

        for &value in &test_values {
            let encoded = VarInt::encode(value);
            let (decoded, consumed) = VarInt::decode(&encoded).unwrap();
            assert_eq!(decoded, value, "Failed for value {}", value);
            assert_eq!(consumed, encoded.len());

            // Test round-trip through DataInput
            let mut input = SliceDataInput::new(&encoded);
            let decoded2 = VarInt::read_from(&mut input).unwrap();
            assert_eq!(decoded2, value);
        }
    }

    #[test]
    fn test_varint_encoded_len() {
        let test_cases = [
            (0, 1),
            (127, 1),
            (128, 2),
            (16383, 2),
            (16384, 3),
            (2097151, 3),
            (2097152, 4),
            (u64::MAX, 10),
        ];

        for (value, expected_len) in test_cases {
            let actual_len = VarInt::encoded_len(value);
            assert_eq!(actual_len, expected_len, "Wrong length for {}", value);

            let encoded = VarInt::encode(value);
            assert_eq!(
                encoded.len(),
                expected_len,
                "Actual encoding length differs for {}",
                value
            );
        }
    }

    #[test]
    fn test_varint_fits_checks() {
        assert!(VarInt::fits_in_one_byte(0));
        assert!(VarInt::fits_in_one_byte(127));
        assert!(!VarInt::fits_in_one_byte(128));

        assert!(VarInt::fits_in_two_bytes(0));
        assert!(VarInt::fits_in_two_bytes(16383));
        assert!(!VarInt::fits_in_two_bytes(16384));
    }

    #[test]
    fn test_varint_write_to_vec() {
        let mut buffer = Vec::new();

        let bytes_written = VarInt::write_to_vec(&mut buffer, 300).unwrap();
        assert_eq!(bytes_written, 2);
        assert_eq!(buffer, vec![172, 2]);
    }

    #[test]
    fn test_varint_write_to_writer() {
        let mut buffer = Vec::new();

        let bytes_written = VarInt::write_to(&mut buffer, 300).unwrap();
        assert_eq!(bytes_written, 2);
        assert_eq!(buffer, vec![172, 2]);
    }

    #[test]
    fn test_varint_multiple_encoding() {
        let values = vec![0, 127, 128, 16384, u64::MAX];
        let encoded = VarInt::encode_multiple(values.clone());
        let decoded = VarInt::decode_multiple(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_varint_invalid_encoding() {
        // Test varint that's too long (11 bytes with continuation bits)
        let invalid_data = vec![0x80; 11];
        let result = VarInt::decode(&invalid_data);
        assert!(result.is_err());

        // Test incomplete varint (ends with continuation bit)
        let incomplete_data = vec![0x80];
        let result = VarInt::decode(&incomplete_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_signed_varint_encoding() {
        let test_values = [0, 1, -1, 2, -2, 63, -64, 64, -65, i64::MAX, i64::MIN];

        for &value in &test_values {
            let encoded = VarInt::encode_signed(value);
            let (decoded, consumed) = VarInt::decode_signed(&encoded).unwrap();
            assert_eq!(decoded, value, "Failed for signed value {}", value);
            assert_eq!(consumed, encoded.len());
        }
    }

    #[test]
    fn test_signed_varint_zigzag() {
        // Test specific zigzag encoding cases
        let test_cases = [(0, 0), (-1, 1), (1, 2), (-2, 3), (2, 4), (-3, 5), (3, 6)];

        for (signed_val, expected_unsigned) in test_cases {
            let encoded = VarInt::encode_signed(signed_val);
            let unsigned_encoded = VarInt::encode(expected_unsigned);
            assert_eq!(
                encoded, unsigned_encoded,
                "Zigzag encoding failed for {}",
                signed_val
            );
        }
    }

    #[test]
    fn test_varint_edge_cases() {
        // Test boundary values
        let boundary_values = [
            127,     // Max 1-byte
            128,     // Min 2-byte
            16383,   // Max 2-byte
            16384,   // Min 3-byte
            2097151, // Max 3-byte
            2097152, // Min 4-byte
        ];

        for &value in &boundary_values {
            let encoded = VarInt::encode(value);
            let (decoded, _) = VarInt::decode(&encoded).unwrap();
            assert_eq!(decoded, value, "Boundary test failed for {}", value);
        }
    }

    #[test]
    fn test_varint_data_input_integration() {
        let values = vec![0, 127, 128, 16384, u64::MAX];
        let mut buffer = Vec::new();

        // Write all values
        for &value in &values {
            VarInt::write_to_vec(&mut buffer, value).unwrap();
        }

        // Read all values back
        let mut input = SliceDataInput::new(&buffer);
        for &expected in &values {
            let actual = VarInt::read_from(&mut input).unwrap();
            assert_eq!(actual, expected);
        }

        // Should be at end of input
        assert!(!input.has_more());
    }

    #[test]
    fn test_varint_performance_patterns() {
        // Test common patterns that should be efficient

        // Small integers (common case)
        for i in 0..256 {
            let encoded = VarInt::encode(i);
            if i < 128 {
                assert_eq!(encoded.len(), 1, "Small int {} should be 1 byte", i);
            } else {
                assert_eq!(encoded.len(), 2, "Int {} should be 2 bytes", i);
            }
        }

        // Powers of 2
        for shift in 0..64 {
            let value = 1u64 << shift;
            let encoded = VarInt::encode(value);
            let expected_len = VarInt::encoded_len(value);
            assert_eq!(encoded.len(), expected_len, "Power of 2: 2^{}", shift);
        }
    }

    #[test]
    fn test_varint_max_encoded_len() {
        // Verify that MAX_ENCODED_LEN is correct
        let max_value = u64::MAX;
        let encoded = VarInt::encode(max_value);
        assert_eq!(encoded.len(), VarInt::MAX_ENCODED_LEN);

        // All values should encode to at most MAX_ENCODED_LEN bytes
        let test_values = [0, 1, 127, 128, 16383, 16384, u64::MAX];
        for &value in &test_values {
            let encoded = VarInt::encode(value);
            assert!(
                encoded.len() <= VarInt::MAX_ENCODED_LEN,
                "Value {} encoded to {} bytes, exceeds max {}",
                value,
                encoded.len(),
                VarInt::MAX_ENCODED_LEN
            );
        }
    }
}
