//! Variable integer encoding variants with multiple strategies
//!
//! This module provides multiple variable integer encoding strategies for
//! different data distributions and use cases, including LEB128, zigzag encoding,
//! delta encoding, group varint encoding, and prefix-free encoding.
//!
//! # Limitation
//! [`VarIntStrategy::Compact`] and [`VarIntStrategy::Simd`] currently delegate to
//! scalar LEB128 (for unsigned integers) and zigzag + LEB128 (for signed integers).

use crate::error::{Result, ZiporaError};
use std::cmp;

/// Variable integer encoding strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarIntStrategy {
    /// LEB128 encoding (Little Endian Base 128)
    Leb128,
    /// Zigzag encoding for signed integers
    Zigzag,
    /// Delta encoding for sequences
    Delta,
    /// Group varint encoding for bulk operations.
    ///
    /// # Wire Format
    /// - Single values ([`VarIntEncoder::encode_u64`], [`VarIntEncoder::encode_i64`])
    ///   are encoded with LEB128 (after zigzag mapping `((v << 1) ^ (v >> 63)) as u64`
    ///   for signed `i64` values).
    /// - Sequences ([`VarIntEncoder::encode_u64_sequence`],
    ///   [`VarIntEncoder::encode_i64_sequence`]) write a LEB128 element count followed
    ///   by groups of up to 4 values. Each group begins with a 2-byte little-endian
    ///   `u16` selector header packing up to four 3-bit `(byte_len - 1)` tags (`0..=7`
    ///   for `1..=8` little-endian value bytes) in bits `0..12` (bits `12..16` are zero),
    ///   with signed `i64` values zigzag-mapped to `u64` before group encoding.
    /// - **Wire compatibility note**: this 2-byte 3-bit-per-tag selector + zigzag `i64`
    ///   wire format is incompatible with the pre-C9 1-byte 2-bit-per-tag selector
    ///   format (which could only represent 1–4 bytes per value and truncated 5–8 byte
    ///   `u64`s and negative `i64`s). There is no magic/version header, so pre-C9 bytes
    ///   are not reliably rejected with `Err`: for example, a pre-C9 single-value
    ///   `encode_i64(5)` payload (`[0x05]`, un-zigzagged LEB128) decodes as `Ok(-3)`
    ///   under zigzag, and a pre-C9 1-byte group header is misread by taking the first
    ///   value byte as the high byte of the 2-byte `u16` selector.
    GroupVarint,
    /// Prefix-free encoding with length prefix
    PrefixFree,
    /// Compact encoding for small ranges.
    ///
    /// # Limitation
    /// Currently delegates to scalar LEB128 (unsigned) and zigzag + LEB128 (signed).
    Compact,
    /// Bulk varint encoding strategy.
    ///
    /// # Limitation
    /// Currently delegates to scalar LEB128 (unsigned) and zigzag + LEB128 (signed).
    Simd,
}

/// Variable integer encoder with multiple strategies
pub struct VarIntEncoder {
    strategy: VarIntStrategy,
}

impl VarIntEncoder {
    /// Create a new encoder with the specified strategy
    pub fn new(strategy: VarIntStrategy) -> Self {
        Self { strategy }
    }

    /// Create encoder for LEB128 strategy
    pub fn leb128() -> Self {
        Self::new(VarIntStrategy::Leb128)
    }

    /// Create encoder for zigzag strategy
    pub fn zigzag() -> Self {
        Self::new(VarIntStrategy::Zigzag)
    }

    /// Create encoder for delta strategy
    pub fn delta() -> Self {
        Self::new(VarIntStrategy::Delta)
    }

    /// Create encoder for group varint strategy (see [`VarIntStrategy::GroupVarint`]
    /// for the 2-byte selector and zigzag `i64` wire format).
    pub fn group_varint() -> Self {
        Self::new(VarIntStrategy::GroupVarint)
    }

    /// Create encoder for prefix-free strategy
    pub fn prefix_free() -> Self {
        Self::new(VarIntStrategy::PrefixFree)
    }

    /// Create encoder for compact strategy
    pub fn compact() -> Self {
        Self::new(VarIntStrategy::Compact)
    }

    /// Create encoder for SIMD strategy
    pub fn simd() -> Self {
        Self::new(VarIntStrategy::Simd)
    }

    /// Get the current strategy
    pub fn strategy(&self) -> VarIntStrategy {
        self.strategy
    }

    /// Encode a single unsigned integer
    pub fn encode_u64(&self, value: u64) -> Result<Vec<u8>> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.encode_leb128_u64(value),
            VarIntStrategy::Zigzag => {
                Err(ZiporaError::invalid_data("Zigzag requires signed input"))
            }
            VarIntStrategy::Delta => Err(ZiporaError::invalid_data(
                "Delta encoding requires sequence",
            )),
            VarIntStrategy::GroupVarint => self.encode_group_varint_single(value),
            VarIntStrategy::PrefixFree => self.encode_prefix_free_u64(value),
            VarIntStrategy::Compact => self.encode_compact_u64(value),
            VarIntStrategy::Simd => self.encode_leb128_u64(value), // Fallback to LEB128 for single values
        }
    }

    /// Encode a single signed integer
    pub fn encode_i64(&self, value: i64) -> Result<Vec<u8>> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.encode_leb128_i64(value),
            VarIntStrategy::Zigzag => self.encode_zigzag_i64(value),
            VarIntStrategy::Delta => Err(ZiporaError::invalid_data(
                "Delta encoding requires sequence",
            )),
            VarIntStrategy::GroupVarint => {
                let unsigned = ((value << 1) ^ (value >> 63)) as u64;
                self.encode_group_varint_single(unsigned)
            }
            VarIntStrategy::PrefixFree => self.encode_prefix_free_i64(value),
            VarIntStrategy::Compact => self.encode_compact_i64(value),
            VarIntStrategy::Simd => self.encode_zigzag_i64(value), // Use zigzag for signed SIMD
        }
    }

    /// Encode a sequence of unsigned integers
    pub fn encode_u64_sequence(&self, values: &[u64]) -> Result<Vec<u8>> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.encode_leb128_sequence_u64(values),
            VarIntStrategy::Zigzag => {
                Err(ZiporaError::invalid_data("Zigzag requires signed input"))
            }
            VarIntStrategy::Delta => self.encode_delta_sequence_u64(values),
            VarIntStrategy::GroupVarint => self.encode_group_varint_u64(values),
            VarIntStrategy::PrefixFree => self.encode_prefix_free_sequence_u64(values),
            VarIntStrategy::Compact => self.encode_compact_sequence_u64(values),
            VarIntStrategy::Simd => self.encode_simd_sequence_u64(values),
        }
    }

    /// Encode a sequence of signed integers
    pub fn encode_i64_sequence(&self, values: &[i64]) -> Result<Vec<u8>> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.encode_leb128_sequence_i64(values),
            VarIntStrategy::Zigzag => self.encode_zigzag_sequence_i64(values),
            VarIntStrategy::Delta => self.encode_delta_sequence_i64(values),
            VarIntStrategy::GroupVarint => {
                let unsigned: Vec<u64> = values
                    .iter()
                    .map(|&v| ((v << 1) ^ (v >> 63)) as u64)
                    .collect();
                self.encode_group_varint_u64(&unsigned)
            }
            VarIntStrategy::PrefixFree => self.encode_prefix_free_sequence_i64(values),
            VarIntStrategy::Compact => self.encode_compact_sequence_i64(values),
            VarIntStrategy::Simd => self.encode_simd_sequence_i64(values),
        }
    }

    /// Decode a single unsigned integer
    pub fn decode_u64(&self, data: &[u8]) -> Result<(u64, usize)> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.decode_leb128_u64(data),
            VarIntStrategy::Zigzag => {
                Err(ZiporaError::invalid_data("Zigzag requires signed output"))
            }
            VarIntStrategy::Delta => Err(ZiporaError::invalid_data(
                "Delta decoding requires sequence context",
            )),
            VarIntStrategy::GroupVarint => self.decode_group_varint_single(data),
            VarIntStrategy::PrefixFree => self.decode_prefix_free_u64(data),
            VarIntStrategy::Compact => self.decode_compact_u64(data),
            VarIntStrategy::Simd => self.decode_leb128_u64(data),
        }
    }

    /// Decode a single signed integer
    pub fn decode_i64(&self, data: &[u8]) -> Result<(i64, usize)> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.decode_leb128_i64(data),
            VarIntStrategy::Zigzag => self.decode_zigzag_i64(data),
            VarIntStrategy::Delta => Err(ZiporaError::invalid_data(
                "Delta decoding requires sequence context",
            )),
            VarIntStrategy::GroupVarint => {
                let (unsigned, consumed) = self.decode_group_varint_single(data)?;
                let signed = ((unsigned >> 1) as i64) ^ (-((unsigned & 1) as i64));
                Ok((signed, consumed))
            }
            VarIntStrategy::PrefixFree => self.decode_prefix_free_i64(data),
            VarIntStrategy::Compact => self.decode_compact_i64(data),
            VarIntStrategy::Simd => self.decode_zigzag_i64(data),
        }
    }

    /// Decode a sequence of unsigned integers
    pub fn decode_u64_sequence(&self, data: &[u8]) -> Result<Vec<u64>> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.decode_leb128_sequence_u64(data),
            VarIntStrategy::Zigzag => {
                Err(ZiporaError::invalid_data("Zigzag requires signed output"))
            }
            VarIntStrategy::Delta => self.decode_delta_sequence_u64(data),
            VarIntStrategy::GroupVarint => self.decode_group_varint_u64(data),
            VarIntStrategy::PrefixFree => self.decode_prefix_free_sequence_u64(data),
            VarIntStrategy::Compact => self.decode_compact_sequence_u64(data),
            VarIntStrategy::Simd => self.decode_simd_sequence_u64(data),
        }
    }

    /// Decode a sequence of signed integers
    pub fn decode_i64_sequence(&self, data: &[u8]) -> Result<Vec<i64>> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.decode_leb128_sequence_i64(data),
            VarIntStrategy::Zigzag => self.decode_zigzag_sequence_i64(data),
            VarIntStrategy::Delta => self.decode_delta_sequence_i64(data),
            VarIntStrategy::GroupVarint => {
                let unsigned = self.decode_group_varint_u64(data)?;
                Ok(unsigned
                    .into_iter()
                    .map(|v| ((v >> 1) as i64) ^ (-((v & 1) as i64)))
                    .collect())
            }
            VarIntStrategy::PrefixFree => self.decode_prefix_free_sequence_i64(data),
            VarIntStrategy::Compact => self.decode_compact_sequence_i64(data),
            VarIntStrategy::Simd => self.decode_simd_sequence_i64(data),
        }
    }
}

// LEB128 implementations
impl VarIntEncoder {
    fn encode_leb128_u64(&self, mut value: u64) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;

            if value != 0 {
                byte |= 0x80;
            }

            result.push(byte);

            if value == 0 {
                break;
            }
        }

        Ok(result)
    }

    fn encode_leb128_i64(&self, value: i64) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        let mut value = value;
        let mut more = true;

        while more {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;

            if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
                more = false;
            } else {
                byte |= 0x80;
            }

            result.push(byte);
        }

        Ok(result)
    }

    fn decode_leb128_u64(&self, data: &[u8]) -> Result<(u64, usize)> {
        let mut result = 0u64;
        let mut shift = 0;
        let mut bytes_read = 0;

        for &byte in data {
            if shift >= 64 || (shift == 63 && byte > 1) {
                return Err(ZiporaError::invalid_data("LEB128 overflow"));
            }

            result |= ((byte & 0x7F) as u64) << shift;
            bytes_read += 1;

            if (byte & 0x80) == 0 {
                return Ok((result, bytes_read));
            }

            shift += 7;
        }

        Err(ZiporaError::invalid_data("Incomplete LEB128"))
    }

    fn decode_leb128_i64(&self, data: &[u8]) -> Result<(i64, usize)> {
        let mut result = 0i64;
        let mut shift = 0;
        let mut bytes_read = 0;

        for &byte in data {
            if shift >= 64 || (shift == 63 && byte != 0x00 && byte != 0x7F) {
                return Err(ZiporaError::invalid_data("LEB128 overflow"));
            }

            result |= ((byte & 0x7F) as i64) << shift;
            bytes_read += 1;
            shift += 7;

            if (byte & 0x80) == 0 {
                // Sign extend if needed
                if shift < 64 && (byte & 0x40) != 0 {
                    result |= !0i64 << shift;
                }
                return Ok((result, bytes_read));
            }
        }

        Err(ZiporaError::invalid_data("Incomplete LEB128"))
    }

    fn encode_leb128_u128(&self, mut value: u128, out: &mut Vec<u8>) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;

            if value != 0 {
                byte |= 0x80;
            }

            out.push(byte);

            if value == 0 {
                break;
            }
        }
    }

    fn decode_leb128_u128_65(&self, data: &[u8]) -> Result<(u128, usize)> {
        let mut result = 0u128;
        let mut shift = 0u32;
        let mut bytes_read = 0usize;

        for &byte in data {
            if shift >= 65 || (shift == 63 && byte > 3) {
                return Err(ZiporaError::invalid_data("LEB128 delta overflow"));
            }

            result |= ((byte & 0x7F) as u128) << shift;
            bytes_read += 1;

            if (byte & 0x80) == 0 {
                return Ok((result, bytes_read));
            }

            shift += 7;
        }

        Err(ZiporaError::invalid_data("Incomplete LEB128"))
    }

    fn encode_leb128_sequence_u64(&self, values: &[u64]) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        // Write count
        let count_bytes = self.encode_leb128_u64(values.len() as u64)?;
        result.extend_from_slice(&count_bytes);

        // Write values
        for &value in values {
            let value_bytes = self.encode_leb128_u64(value)?;
            result.extend_from_slice(&value_bytes);
        }

        Ok(result)
    }

    fn encode_leb128_sequence_i64(&self, values: &[i64]) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        // Write count
        let count_bytes = self.encode_leb128_u64(values.len() as u64)?;
        result.extend_from_slice(&count_bytes);

        // Write values
        for &value in values {
            let value_bytes = self.encode_leb128_i64(value)?;
            result.extend_from_slice(&value_bytes);
        }

        Ok(result)
    }

    fn decode_leb128_sequence_u64(&self, data: &[u8]) -> Result<Vec<u64>> {
        let mut offset = 0;

        // Read count
        let slice = data
            .get(offset..)
            .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
        let (count, count_bytes) = self.decode_leb128_u64(slice)?;
        offset += count_bytes;

        let remaining = data.len().saturating_sub(offset);
        if count > remaining as u64 {
            return Err(ZiporaError::invalid_data(format!(
                "leb128 sequence count {} exceeds remaining bytes {}",
                count, remaining
            )));
        }

        let mut result = Vec::with_capacity(count as usize);

        // Read values
        for _ in 0..count {
            let slice = data
                .get(offset..)
                .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
            let (value, value_bytes) = self.decode_leb128_u64(slice)?;
            result.push(value);
            offset += value_bytes;
        }

        Ok(result)
    }

    fn decode_leb128_sequence_i64(&self, data: &[u8]) -> Result<Vec<i64>> {
        let mut offset = 0;

        // Read count
        let slice = data
            .get(offset..)
            .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
        let (count, count_bytes) = self.decode_leb128_u64(slice)?;
        offset += count_bytes;

        let remaining = data.len().saturating_sub(offset);
        if count > remaining as u64 {
            return Err(ZiporaError::invalid_data(format!(
                "leb128 sequence count {} exceeds remaining bytes {}",
                count, remaining
            )));
        }

        let mut result = Vec::with_capacity(count as usize);

        // Read values
        for _ in 0..count {
            let slice = data
                .get(offset..)
                .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
            let (value, value_bytes) = self.decode_leb128_i64(slice)?;
            result.push(value);
            offset += value_bytes;
        }

        Ok(result)
    }

}

// Zigzag implementations
impl VarIntEncoder {
    fn encode_zigzag_i64(&self, value: i64) -> Result<Vec<u8>> {
        let encoded = ((value << 1) ^ (value >> 63)) as u64;
        self.encode_leb128_u64(encoded)
    }

    fn decode_zigzag_i64(&self, data: &[u8]) -> Result<(i64, usize)> {
        let (encoded, bytes_read) = self.decode_leb128_u64(data)?;
        let decoded = ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64));
        Ok((decoded, bytes_read))
    }

    fn encode_zigzag_sequence_i64(&self, values: &[i64]) -> Result<Vec<u8>> {
        let encoded: Vec<u64> = values
            .iter()
            .map(|&v| ((v << 1) ^ (v >> 63)) as u64)
            .collect();

        self.encode_leb128_sequence_u64(&encoded)
    }

    fn decode_zigzag_sequence_i64(&self, data: &[u8]) -> Result<Vec<i64>> {
        let encoded = self.decode_leb128_sequence_u64(data)?;
        let decoded: Vec<i64> = encoded
            .into_iter()
            .map(|v| ((v >> 1) as i64) ^ (-((v & 1) as i64)))
            .collect();
        Ok(decoded)
    }
}

// Delta encoding implementations
impl VarIntEncoder {
    fn encode_delta_sequence_u64(&self, values: &[u64]) -> Result<Vec<u8>> {
        if values.is_empty() {
            return self.encode_leb128_u64(0);
        }

        let mut result = Vec::new();

        // Write count and first value
        let count_bytes = self.encode_leb128_u64(values.len() as u64)?;
        result.extend_from_slice(&count_bytes);

        let first_bytes = self.encode_leb128_u64(values[0])?;
        result.extend_from_slice(&first_bytes);

        // Write deltas (64-bit magnitude + 1 sign bit = up to 65 bits)
        for i in 1..values.len() {
            let delta: u128 = if values[i] >= values[i - 1] {
                ((values[i] - values[i - 1]) as u128) << 1 // Positive delta, LSB = 0
            } else {
                (((values[i - 1] - values[i]) as u128) << 1) | 1 // Negative delta, LSB = 1
            };

            self.encode_leb128_u128(delta, &mut result);
        }

        Ok(result)
    }

    fn encode_delta_sequence_i64(&self, values: &[i64]) -> Result<Vec<u8>> {
        if values.is_empty() {
            return self.encode_leb128_u64(0);
        }

        let mut result = Vec::new();

        // Write count and first value
        let count_bytes = self.encode_leb128_u64(values.len() as u64)?;
        result.extend_from_slice(&count_bytes);

        let first_bytes = self.encode_leb128_i64(values[0])?;
        result.extend_from_slice(&first_bytes);

        // Write deltas using wrapping arithmetic + zigzag encoding
        for i in 1..values.len() {
            let delta = values[i].wrapping_sub(values[i - 1]);
            let delta_bytes = self.encode_zigzag_i64(delta)?;
            result.extend_from_slice(&delta_bytes);
        }

        Ok(result)
    }

    fn decode_delta_sequence_u64(&self, data: &[u8]) -> Result<Vec<u64>> {
        let mut offset = 0;

        // Read count
        let slice = data
            .get(offset..)
            .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
        let (count, count_bytes) = self.decode_leb128_u64(slice)?;
        offset += count_bytes;

        if count == 0 {
            return Ok(Vec::new());
        }

        let remaining = data.len().saturating_sub(offset);
        if count > remaining as u64 {
            return Err(ZiporaError::invalid_data(format!(
                "delta sequence count {} exceeds remaining bytes {}",
                count, remaining
            )));
        }

        let mut result = Vec::with_capacity(count as usize);

        // Read first value
        let slice = data
            .get(offset..)
            .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
        let (first_value, first_bytes) = self.decode_leb128_u64(slice)?;
        result.push(first_value);
        offset += first_bytes;

        // Read deltas
        let mut current_value = first_value;
        for _ in 1..count {
            let slice = data
                .get(offset..)
                .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
            let (encoded_delta, delta_bytes) = self.decode_leb128_u128_65(slice)?;
            let mag = (encoded_delta >> 1) as u64;

            current_value = if (encoded_delta & 1) == 0 {
                // Positive delta
                current_value.wrapping_add(mag)
            } else {
                // Negative delta
                current_value.wrapping_sub(mag)
            };

            result.push(current_value);
            offset += delta_bytes;
        }

        Ok(result)
    }

    fn decode_delta_sequence_i64(&self, data: &[u8]) -> Result<Vec<i64>> {
        let mut offset = 0;

        // Read count
        let slice = data
            .get(offset..)
            .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
        let (count, count_bytes) = self.decode_leb128_u64(slice)?;
        offset += count_bytes;

        if count == 0 {
            return Ok(Vec::new());
        }

        let remaining = data.len().saturating_sub(offset);
        if count > remaining as u64 {
            return Err(ZiporaError::invalid_data(format!(
                "delta sequence count {} exceeds remaining bytes {}",
                count, remaining
            )));
        }

        let mut result = Vec::with_capacity(count as usize);

        // Read first value
        let slice = data
            .get(offset..)
            .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
        let (first_value, first_bytes) = self.decode_leb128_i64(slice)?;
        result.push(first_value);
        offset += first_bytes;

        // Read deltas
        let mut current_value = first_value;
        for _ in 1..count {
            let slice = data
                .get(offset..)
                .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
            let (delta, delta_bytes) = self.decode_zigzag_i64(slice)?;
            current_value = current_value.wrapping_add(delta);
            result.push(current_value);
            offset += delta_bytes;
        }

        Ok(result)
    }

}

// Group varint implementations
impl VarIntEncoder {
    fn encode_group_varint_single(&self, value: u64) -> Result<Vec<u8>> {
        // For single values, fall back to LEB128
        self.encode_leb128_u64(value)
    }

    fn decode_group_varint_single(&self, data: &[u8]) -> Result<(u64, usize)> {
        // For single values, fall back to LEB128
        self.decode_leb128_u64(data)
    }

    fn encode_group_varint_u64(&self, values: &[u64]) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        // Write count
        let count_bytes = self.encode_leb128_u64(values.len() as u64)?;
        result.extend_from_slice(&count_bytes);

        // Process in groups of 4 using a 2-byte selector (3 bits per value for 1..=8 bytes)
        for chunk in values.chunks(4) {
            let mut selector = 0u16;
            let mut group_data = Vec::new();

            for (i, &value) in chunk.iter().enumerate() {
                let bytes_needed = if value == 0 {
                    1
                } else {
                    (64 - value.leading_zeros()).div_ceil(8) as usize
                };

                // Encode bytes needed in selector (3 bits per value: 1..=8 bytes -> 0..=7)
                selector |= ((bytes_needed - 1) as u16) << (i * 3);

                // Store the value in little-endian format
                let value_bytes = value.to_le_bytes();
                group_data.extend_from_slice(&value_bytes[..bytes_needed]);
            }

            result.extend_from_slice(&selector.to_le_bytes());
            result.extend_from_slice(&group_data);
        }

        Ok(result)
    }

    fn decode_group_varint_u64(&self, data: &[u8]) -> Result<Vec<u64>> {
        let mut offset = 0;

        // Read count
        let slice = data
            .get(offset..)
            .ok_or_else(|| ZiporaError::invalid_data("Empty group varint data"))?;
        let (count, count_bytes) = self.decode_leb128_u64(slice)?;
        offset += count_bytes;

        let remaining_bytes = data.len().saturating_sub(offset);
        if count > remaining_bytes as u64 {
            return Err(ZiporaError::invalid_data("Group varint count exceeds remaining bytes"));
        }

        let mut result = Vec::with_capacity(count as usize);
        let mut remaining = count;

        while remaining > 0 {
            let selector = match data.get(offset..offset + 2) {
                Some(&[s0, s1]) => u16::from_le_bytes([s0, s1]),
                _ => return Err(ZiporaError::invalid_data("Incomplete group varint")),
            };
            offset += 2;
            if (selector & !0x0FFF) != 0 {
                return Err(ZiporaError::invalid_data("Invalid group varint selector"));
            }

            let chunk_size = cmp::min(remaining, 4);

            for i in 0..chunk_size {
                let bytes_needed = ((selector >> (i * 3)) & 0x7) as usize + 1;

                let val_slice = data
                    .get(offset..offset + bytes_needed)
                    .ok_or_else(|| ZiporaError::invalid_data("Incomplete group varint value"))?;

                let mut raw = [0u8; 8];
                if let Some(dest) = raw.get_mut(..bytes_needed) {
                    dest.copy_from_slice(val_slice);
                }

                let value = u64::from_le_bytes(raw);
                result.push(value);
                offset += bytes_needed;
            }

            remaining -= chunk_size;
        }

        Ok(result)
    }
}

// Prefix-free implementations
impl VarIntEncoder {
    fn encode_prefix_free_u64(&self, value: u64) -> Result<Vec<u8>> {
        // Use length prefix followed by value
        let value_bytes = value.to_le_bytes();
        let significant_bytes = (64 - value.leading_zeros()).div_ceil(8) as usize;
        let significant_bytes = if significant_bytes == 0 {
            1
        } else {
            significant_bytes
        };

        let mut result = Vec::new();
        result.push(significant_bytes as u8);
        result.extend_from_slice(&value_bytes[..significant_bytes]);

        Ok(result)
    }

    fn encode_prefix_free_i64(&self, value: i64) -> Result<Vec<u8>> {
        // Convert to unsigned using zigzag
        let unsigned = ((value << 1) ^ (value >> 63)) as u64;
        self.encode_prefix_free_u64(unsigned)
    }

    fn decode_prefix_free_u64(&self, data: &[u8]) -> Result<(u64, usize)> {
        let length_byte = *data
            .first()
            .ok_or_else(|| ZiporaError::invalid_data("Empty prefix-free data"))?;
        let length = length_byte as usize;
        if length == 0 || length > 8 {
            return Err(ZiporaError::invalid_data("Invalid prefix-free length"));
        }

        let chunk = data
            .get(1..1 + length)
            .ok_or_else(|| ZiporaError::invalid_data("Incomplete prefix-free data"))?;

        let mut raw = [0u8; 8];
        if let Some(dest) = raw.get_mut(..length) {
            dest.copy_from_slice(chunk);
        }

        let value = u64::from_le_bytes(raw);
        Ok((value, 1 + length))
    }

    fn decode_prefix_free_i64(&self, data: &[u8]) -> Result<(i64, usize)> {
        let (unsigned, consumed) = self.decode_prefix_free_u64(data)?;
        let signed = ((unsigned >> 1) as i64) ^ (-((unsigned & 1) as i64));
        Ok((signed, consumed))
    }

    fn encode_prefix_free_sequence_u64(&self, values: &[u64]) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        // Write count
        let count_bytes = self.encode_leb128_u64(values.len() as u64)?;
        result.extend_from_slice(&count_bytes);

        // Write values
        for &value in values {
            let value_bytes = self.encode_prefix_free_u64(value)?;
            result.extend_from_slice(&value_bytes);
        }

        Ok(result)
    }

    fn encode_prefix_free_sequence_i64(&self, values: &[i64]) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        // Write count
        let count_bytes = self.encode_leb128_u64(values.len() as u64)?;
        result.extend_from_slice(&count_bytes);

        // Write values
        for &value in values {
            let value_bytes = self.encode_prefix_free_i64(value)?;
            result.extend_from_slice(&value_bytes);
        }

        Ok(result)
    }

    fn decode_prefix_free_sequence_u64(&self, data: &[u8]) -> Result<Vec<u64>> {
        let mut offset = 0;

        // Read count
        let slice = data
            .get(offset..)
            .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
        let (count, count_bytes) = self.decode_leb128_u64(slice)?;
        offset += count_bytes;

        let remaining = data.len().saturating_sub(offset);
        if count > remaining as u64 {
            return Err(ZiporaError::invalid_data(format!(
                "prefix-free sequence count {} exceeds remaining bytes {}",
                count, remaining
            )));
        }

        let mut result = Vec::with_capacity(count as usize);

        // Read values
        for _ in 0..count {
            let slice = data
                .get(offset..)
                .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
            let (value, value_bytes) = self.decode_prefix_free_u64(slice)?;
            result.push(value);
            offset += value_bytes;
        }

        Ok(result)
    }

    fn decode_prefix_free_sequence_i64(&self, data: &[u8]) -> Result<Vec<i64>> {
        let mut offset = 0;

        // Read count
        let slice = data
            .get(offset..)
            .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
        let (count, count_bytes) = self.decode_leb128_u64(slice)?;
        offset += count_bytes;

        let remaining = data.len().saturating_sub(offset);
        if count > remaining as u64 {
            return Err(ZiporaError::invalid_data(format!(
                "prefix-free sequence count {} exceeds remaining bytes {}",
                count, remaining
            )));
        }

        let mut result = Vec::with_capacity(count as usize);

        // Read values
        for _ in 0..count {
            let slice = data
                .get(offset..)
                .ok_or_else(|| ZiporaError::invalid_data("offset out of bounds"))?;
            let (value, value_bytes) = self.decode_prefix_free_i64(slice)?;
            result.push(value);
            offset += value_bytes;
        }

        Ok(result)
    }
}

// Compact and SIMD implementations
impl VarIntEncoder {
    fn encode_compact_u64(&self, value: u64) -> Result<Vec<u8>> {
        self.encode_leb128_u64(value)
    }

    fn encode_compact_i64(&self, value: i64) -> Result<Vec<u8>> {
        self.encode_zigzag_i64(value)
    }

    fn decode_compact_u64(&self, data: &[u8]) -> Result<(u64, usize)> {
        self.decode_leb128_u64(data)
    }

    fn decode_compact_i64(&self, data: &[u8]) -> Result<(i64, usize)> {
        self.decode_zigzag_i64(data)
    }

    fn encode_compact_sequence_u64(&self, values: &[u64]) -> Result<Vec<u8>> {
        self.encode_leb128_sequence_u64(values)
    }

    fn encode_compact_sequence_i64(&self, values: &[i64]) -> Result<Vec<u8>> {
        self.encode_zigzag_sequence_i64(values)
    }

    fn decode_compact_sequence_u64(&self, data: &[u8]) -> Result<Vec<u64>> {
        self.decode_leb128_sequence_u64(data)
    }

    fn decode_compact_sequence_i64(&self, data: &[u8]) -> Result<Vec<i64>> {
        self.decode_zigzag_sequence_i64(data)
    }

    fn encode_simd_sequence_u64(&self, values: &[u64]) -> Result<Vec<u8>> {
        // LEB128 encoding for bulk sequences
        self.encode_leb128_sequence_u64(values)
    }

    fn encode_simd_sequence_i64(&self, values: &[i64]) -> Result<Vec<u8>> {
        // Zigzag encoding for bulk sequences
        self.encode_zigzag_sequence_i64(values)
    }


    fn decode_simd_sequence_u64(&self, data: &[u8]) -> Result<Vec<u64>> {
        self.decode_leb128_sequence_u64(data)
    }

    fn decode_simd_sequence_i64(&self, data: &[u8]) -> Result<Vec<i64>> {
        self.decode_zigzag_sequence_i64(data)
    }
}

/// Choose optimal encoding strategy based on data characteristics
pub fn choose_optimal_strategy(values: &[u64]) -> VarIntStrategy {
    let Some(&max_value) = values.iter().max() else {
        return VarIntStrategy::Leb128;
    };

    // Analyze data characteristics
    let is_sorted = values.windows(2).all(|w| w[0] <= w[1]);
    let small_values = values.iter().all(|&v| v < 256);

    // Choose strategy based on characteristics with careful prioritization
    if values.len() >= 16 && max_value < (1u64 << 32) {
        VarIntStrategy::GroupVarint // Good for bulk operations with moderate values (prioritized for large sequences)
    } else if is_sorted && values.len() > 6 {
        // Delta for sorted sequences with sufficient length (but not too small)
        VarIntStrategy::Delta // Good for sorted sequences
    } else if small_values {
        VarIntStrategy::Compact // Good for small values
    } else {
        VarIntStrategy::Leb128 // Default fallback
    }
}

/// Choose optimal strategy for signed integers
pub fn choose_optimal_strategy_signed(values: &[i64]) -> VarIntStrategy {
    if values.is_empty() {
        return VarIntStrategy::Zigzag;
    }

    // Analyze data characteristics
    let has_negative = values.iter().any(|&v| v < 0);
    let is_sorted = values.windows(2).all(|w| w[0] <= w[1]);
    let small_range = values.iter().all(|&v| v.unsigned_abs() < 256);

    // Choose strategy based on characteristics - prioritize sorted sequences even with negative values
    if is_sorted && values.len() > 5 {
        // Delta for sorted sequences with sufficient length
        VarIntStrategy::Delta // Good for sorted sequences
    } else if has_negative {
        VarIntStrategy::Zigzag // Good for mixed positive/negative
    } else if small_range {
        VarIntStrategy::Compact // Good for small values
    } else {
        VarIntStrategy::Leb128 // Use unsigned encoding for all positive
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leb128_encoding() {
        let encoder = VarIntEncoder::leb128();

        let test_values = [0u64, 127, 128, 16383, 16384, u64::MAX];

        for &value in &test_values {
            let encoded = encoder.encode_u64(value).unwrap();
            let (decoded, _) = encoder.decode_u64(&encoded).unwrap();
            assert_eq!(decoded, value, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_zigzag_encoding() {
        let encoder = VarIntEncoder::zigzag();

        let test_values = [0i64, 1, -1, 127, -128, i64::MAX, i64::MIN];

        for &value in &test_values {
            let encoded = encoder.encode_i64(value).unwrap();
            let (decoded, _) = encoder.decode_i64(&encoded).unwrap();
            assert_eq!(decoded, value, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_delta_encoding() {
        let encoder = VarIntEncoder::delta();

        let values = vec![10u64, 12, 15, 20, 22, 25];
        let encoded = encoder.encode_u64_sequence(&values).unwrap();
        let decoded = encoder.decode_u64_sequence(&encoded).unwrap();

        assert_eq!(decoded, values);
    }

    #[test]
    fn test_group_varint_encoding() {
        let encoder = VarIntEncoder::group_varint();

        let values = vec![1u64, 256, 65536, 16777216];
        let encoded = encoder.encode_u64_sequence(&values).unwrap();
        let decoded = encoder.decode_group_varint_u64(&encoded).unwrap();

        assert_eq!(decoded, values);
    }

    #[test]
    fn test_prefix_free_encoding() {
        let encoder = VarIntEncoder::prefix_free();

        let test_values = [0u64, 255, 65535, 16777215, u64::MAX];

        for &value in &test_values {
            let encoded = encoder.encode_u64(value).unwrap();
            let (decoded, _) = encoder.decode_u64(&encoded).unwrap();
            assert_eq!(decoded, value, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_sequence_encoding() {
        let encoder = VarIntEncoder::leb128();

        let values = vec![1u64, 2, 3, 127, 128, 16383, 16384];
        let encoded = encoder.encode_u64_sequence(&values).unwrap();
        let decoded = encoder.decode_u64_sequence(&encoded).unwrap();

        assert_eq!(decoded, values);
    }

    #[test]
    fn test_signed_sequence_encoding() {
        let encoder = VarIntEncoder::zigzag();

        let values = vec![-100i64, -1, 0, 1, 100];
        let encoded = encoder.encode_i64_sequence(&values).unwrap();
        let decoded = encoder.decode_i64_sequence(&encoded).unwrap();

        assert_eq!(decoded, values);
    }

    #[test]
    fn test_strategy_selection() {
        // Test sorted sequence
        let sorted_values = vec![1u64, 5, 10, 15, 20, 25, 30];
        let strategy = choose_optimal_strategy(&sorted_values);
        assert_eq!(strategy, VarIntStrategy::Delta);

        // Test small values
        let small_values = vec![1u64, 2, 3, 100, 200];
        let strategy = choose_optimal_strategy(&small_values);
        assert_eq!(strategy, VarIntStrategy::Compact);

        // Test large sequence with moderate values
        let large_sequence: Vec<u64> = (0..20).map(|i| (i * 1000) as u64).collect();
        let strategy = choose_optimal_strategy(&large_sequence);
        assert_eq!(strategy, VarIntStrategy::GroupVarint);
    }

    #[test]
    fn test_signed_strategy_selection() {
        // Test mixed positive/negative
        let mixed_values = vec![-10i64, -5, 0, 5, 10];
        let strategy = choose_optimal_strategy_signed(&mixed_values);
        assert_eq!(strategy, VarIntStrategy::Zigzag);

        // Test sorted sequence
        let sorted_values = vec![-100i64, -50, 0, 50, 100, 150];
        let strategy = choose_optimal_strategy_signed(&sorted_values);
        assert_eq!(strategy, VarIntStrategy::Delta);
    }

    #[test]
    fn test_empty_sequences() {
        let encoder = VarIntEncoder::leb128();

        let empty_u64: Vec<u64> = vec![];
        let encoded = encoder.encode_u64_sequence(&empty_u64).unwrap();
        let decoded = encoder.decode_u64_sequence(&encoded).unwrap();
        assert_eq!(decoded, empty_u64);

        let empty_i64: Vec<i64> = vec![];
        let encoded = encoder.encode_i64_sequence(&empty_i64).unwrap();
        let decoded = encoder.decode_i64_sequence(&encoded).unwrap();
        assert_eq!(decoded, empty_i64);
    }

    #[test]
    fn test_single_value_sequences() {
        let encoder = VarIntEncoder::delta();

        let single_value = vec![42u64];
        let encoded = encoder.encode_u64_sequence(&single_value).unwrap();
        let decoded = encoder.decode_u64_sequence(&encoded).unwrap();
        assert_eq!(decoded, single_value);
    }

    #[test]
    fn test_large_values() {
        let encoder = VarIntEncoder::leb128();

        let large_values = vec![u64::MAX, u64::MAX - 1, 1u64 << 63, 1u64 << 32];
        let encoded = encoder.encode_u64_sequence(&large_values).unwrap();
        let decoded = encoder.decode_u64_sequence(&encoded).unwrap();
        assert_eq!(decoded, large_values);
    }

    #[test]
    fn test_edge_cases() {
        let encoder = VarIntEncoder::zigzag();

        // Test edge cases for signed integers
        let edge_values = vec![i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX];
        let encoded = encoder.encode_i64_sequence(&edge_values).unwrap();
        let decoded = encoder.decode_i64_sequence(&encoded).unwrap();
        assert_eq!(decoded, edge_values);
    }

    #[test]
    fn test_sequence_count_exceeds_buffer_rejected() {
        let encoder_leb = VarIntEncoder::leb128();
        let encoder_delta = VarIntEncoder::delta();
        let encoder_group = VarIntEncoder::group_varint();
        let encoder_prefix = VarIntEncoder::prefix_free();

        // Count = 1_000_000_000 encoded in LEB128: 0x80, 0x94, 0xeb, 0xdc, 0x03 followed by no data or just 1 byte
        let malformed = vec![0x80, 0x94, 0xeb, 0xdc, 0x03, 0x01];
        assert!(encoder_leb.decode_u64_sequence(&malformed).is_err());
        assert!(encoder_leb.decode_i64_sequence(&malformed).is_err());
        assert!(encoder_delta.decode_u64_sequence(&malformed).is_err());
        assert!(encoder_delta.decode_i64_sequence(&malformed).is_err());
        assert!(encoder_group.decode_u64_sequence(&malformed).is_err());
        assert!(encoder_prefix.decode_u64_sequence(&malformed).is_err());
        assert!(encoder_prefix.decode_i64_sequence(&malformed).is_err());

        // Count = u64::MAX
        let max_count_leb = vec![0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        assert!(encoder_leb.decode_u64_sequence(&max_count_leb).is_err());
        assert!(encoder_leb.decode_i64_sequence(&max_count_leb).is_err());
        assert!(encoder_delta.decode_u64_sequence(&max_count_leb).is_err());
        assert!(encoder_delta.decode_i64_sequence(&max_count_leb).is_err());
        assert!(encoder_group.decode_u64_sequence(&max_count_leb).is_err());
        assert!(encoder_prefix.decode_u64_sequence(&max_count_leb).is_err());
        assert!(encoder_prefix.decode_i64_sequence(&max_count_leb).is_err());
    }

    #[test]
    fn test_delta_sequence_wrapping_overflow() {
        let encoder_delta = VarIntEncoder::delta();
        // Input [2, 0, 3]: count = 2, first = 0, delta = negative 1.
        // In debug build with checked arithmetic this panicked with attempt to subtract with overflow.
        // With wrapping arithmetic, it decodes cleanly to [0, u64::MAX].
        let input = [2, 0, 3];
        let decoded = encoder_delta.decode_u64_sequence(&input).expect("wrapping subtraction does not panic");
        assert_eq!(decoded, vec![0, u64::MAX]);

        // Same for i64 delta sequence
        let input_i64 = [2, 0, 1];
        let decoded_i64 = encoder_delta.decode_i64_sequence(&input_i64).expect("wrapping delta does not panic");
        assert_eq!(decoded_i64, vec![0, -1]);
    }

    #[test]
    fn test_c9_vv1_group_varint_roundtrips_u64_above_32_bits_and_negative_i64() {
        let encoder = VarIntEncoder::group_varint();

        let u64_vals = vec![
            0u64,
            1,
            u32::MAX as u64,
            1u64 << 32,
            (1u64 << 40) + 7,
            (1u64 << 48) + 9,
            (1u64 << 56) + 11,
            u64::MAX,
        ];
        let encoded_u64 = encoder.encode_u64_sequence(&u64_vals).unwrap();
        let decoded_u64 = encoder.decode_u64_sequence(&encoded_u64).unwrap();
        assert_eq!(decoded_u64, u64_vals);

        let i64_vals = vec![-1i64, -128, -100_000, i64::MIN, i64::MAX, 0, 42];
        let encoded_i64 = encoder.encode_i64_sequence(&i64_vals).unwrap();
        let decoded_i64 = encoder.decode_i64_sequence(&encoded_i64).unwrap();
        assert_eq!(decoded_i64, i64_vals);

        let single_neg = encoder.encode_i64(-1).unwrap();
        assert_eq!(encoder.decode_i64(&single_neg).unwrap().0, -1);
    }

    #[test]
    fn test_c9_vv2_delta_u64_roundtrips_full_64_bit_span() {
        let encoder = VarIntEncoder::delta();

        let values = vec![0u64, 1u64 << 63, u64::MAX, 0, u64::MAX];
        let encoded = encoder.encode_u64_sequence(&values).unwrap();
        let decoded = encoder.decode_u64_sequence(&encoded).unwrap();
        assert_eq!(decoded, values);

        let sorted_span = vec![0u64, 1, 2, 3, 4, 5, u64::MAX];
        let strategy = choose_optimal_strategy(&sorted_span);
        assert_eq!(strategy, VarIntStrategy::Delta);
        let auto_encoder = VarIntEncoder::new(strategy);
        let auto_encoded = auto_encoder.encode_u64_sequence(&sorted_span).unwrap();
        let auto_decoded = auto_encoder.decode_u64_sequence(&auto_encoded).unwrap();
        assert_eq!(auto_decoded, sorted_span);
    }

    #[test]
    fn test_c9_vv3_delta_i64_roundtrips_i64_min_to_max_without_overflow() {
        let encoder = VarIntEncoder::delta();
        let values = vec![i64::MIN, i64::MAX, i64::MIN, 0];
        let res = std::panic::catch_unwind(|| {
            let encoded = encoder.encode_i64_sequence(&values).unwrap();
            encoder.decode_i64_sequence(&encoded).unwrap()
        });
        assert_eq!(
            res.expect("encode_delta_sequence_i64 must not panic on i64::MIN..i64::MAX"),
            values
        );
    }

    #[test]
    fn test_c9_vv4_leb128_decoders_reject_tenth_byte_overflow() {
        let encoder = VarIntEncoder::leb128();

        // 10th byte > 1 overflows u64
        let overflow_u64 = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02];
        assert!(
            encoder.decode_u64(&overflow_u64).is_err(),
            "u64 LEB128 with 10th byte 0x02 must fail"
        );

        // 10th byte for signed i64 must be 0x00 (positive sign bit 0) or 0x7F (negative sign bit 1)
        let overflow_pos_i64 = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
        assert!(
            encoder.decode_i64(&overflow_pos_i64).is_err(),
            "i64 LEB128 with 10th byte 0x01 (+2^63) must fail"
        );
        let overflow_neg_i64 = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7E];
        assert!(
            encoder.decode_i64(&overflow_neg_i64).is_err(),
            "i64 LEB128 with 10th byte 0x7E (-2^63 - 1) must fail"
        );
    }

    #[test]
    fn test_c9_vv5_choose_optimal_strategy_signed_handles_i64_min() {
        let res = std::panic::catch_unwind(|| choose_optimal_strategy_signed(&[i64::MIN]));
        assert_eq!(
            res.expect("choose_optimal_strategy_signed must not panic on i64::MIN"),
            VarIntStrategy::Zigzag
        );
    }
}

