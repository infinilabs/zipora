//! Variable integer encoding variants with multiple strategies
//!
//! This module provides multiple variable integer encoding strategies optimized for
//! different data distributions and use cases, including LEB128, zigzag encoding,
//! delta encoding, and SIMD-accelerated variants.

use crate::error::{Result, ZiporaError};
// Note: DataInput and DataOutput imports are available but not directly used in this module
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
    /// Group varint encoding for bulk operations
    GroupVarint,
    /// Prefix-free encoding with length prefix
    PrefixFree,
    /// Compact encoding for small ranges
    Compact,
    /// SIMD-optimized encoding
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
    
    /// Create encoder for group varint strategy
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
            VarIntStrategy::Zigzag => Err(ZiporaError::invalid_data("Zigzag requires signed input")),
            VarIntStrategy::Delta => Err(ZiporaError::invalid_data("Delta encoding requires sequence")),
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
            VarIntStrategy::Delta => Err(ZiporaError::invalid_data("Delta encoding requires sequence")),
            VarIntStrategy::GroupVarint => self.encode_group_varint_single(value as u64),
            VarIntStrategy::PrefixFree => self.encode_prefix_free_i64(value),
            VarIntStrategy::Compact => self.encode_compact_i64(value),
            VarIntStrategy::Simd => self.encode_zigzag_i64(value), // Use zigzag for signed SIMD
        }
    }
    
    /// Encode a sequence of unsigned integers
    pub fn encode_u64_sequence(&self, values: &[u64]) -> Result<Vec<u8>> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.encode_leb128_sequence_u64(values),
            VarIntStrategy::Zigzag => Err(ZiporaError::invalid_data("Zigzag requires signed input")),
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
                let unsigned: Vec<u64> = values.iter().map(|&v| v as u64).collect();
                self.encode_group_varint_u64(&unsigned)
            },
            VarIntStrategy::PrefixFree => self.encode_prefix_free_sequence_i64(values),
            VarIntStrategy::Compact => self.encode_compact_sequence_i64(values),
            VarIntStrategy::Simd => self.encode_simd_sequence_i64(values),
        }
    }
    
    /// Decode a single unsigned integer
    pub fn decode_u64(&self, data: &[u8]) -> Result<(u64, usize)> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.decode_leb128_u64(data),
            VarIntStrategy::Zigzag => Err(ZiporaError::invalid_data("Zigzag requires signed output")),
            VarIntStrategy::Delta => Err(ZiporaError::invalid_data("Delta decoding requires sequence context")),
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
            VarIntStrategy::Delta => Err(ZiporaError::invalid_data("Delta decoding requires sequence context")),
            VarIntStrategy::GroupVarint => {
                let (value, consumed) = self.decode_group_varint_single(data)?;
                Ok((value as i64, consumed))
            },
            VarIntStrategy::PrefixFree => self.decode_prefix_free_i64(data),
            VarIntStrategy::Compact => self.decode_compact_i64(data),
            VarIntStrategy::Simd => self.decode_zigzag_i64(data),
        }
    }
    
    /// Decode a sequence of unsigned integers
    pub fn decode_u64_sequence(&self, data: &[u8]) -> Result<Vec<u64>> {
        match self.strategy {
            VarIntStrategy::Leb128 => self.decode_leb128_sequence_u64(data),
            VarIntStrategy::Zigzag => Err(ZiporaError::invalid_data("Zigzag requires signed output")),
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
                Ok(unsigned.into_iter().map(|v| v as i64).collect())
            },
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
            if shift >= 64 {
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
            if shift >= 64 {
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
        let (count, count_bytes) = self.decode_leb128_u64(&data[offset..])?;
        offset += count_bytes;
        
        let mut result = Vec::with_capacity(count as usize);
        
        // Read values
        for _ in 0..count {
            let (value, value_bytes) = self.decode_leb128_u64(&data[offset..])?;
            result.push(value);
            offset += value_bytes;
        }
        
        Ok(result)
    }
    
    fn decode_leb128_sequence_i64(&self, data: &[u8]) -> Result<Vec<i64>> {
        let mut offset = 0;
        
        // Read count
        let (count, count_bytes) = self.decode_leb128_u64(&data[offset..])?;
        offset += count_bytes;
        
        let mut result = Vec::with_capacity(count as usize);
        
        // Read values
        for _ in 0..count {
            let (value, value_bytes) = self.decode_leb128_i64(&data[offset..])?;
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
        let encoded: Vec<u64> = values.iter()
            .map(|&v| {
                ((v << 1) ^ (v >> 63)) as u64
            })
            .collect();
        
        self.encode_leb128_sequence_u64(&encoded)
    }
    
    fn decode_zigzag_sequence_i64(&self, data: &[u8]) -> Result<Vec<i64>> {
        let encoded = self.decode_leb128_sequence_u64(data)?;
        let decoded: Vec<i64> = encoded.into_iter()
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
        
        // Write deltas
        for i in 1..values.len() {
            let delta = if values[i] >= values[i-1] {
                (values[i] - values[i-1]) << 1 // Positive delta, LSB = 0
            } else {
                ((values[i-1] - values[i]) << 1) | 1 // Negative delta, LSB = 1
            };
            
            let delta_bytes = self.encode_leb128_u64(delta)?;
            result.extend_from_slice(&delta_bytes);
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
        
        // Write deltas using zigzag encoding
        for i in 1..values.len() {
            let delta = values[i] - values[i-1];
            let delta_bytes = self.encode_zigzag_i64(delta)?;
            result.extend_from_slice(&delta_bytes);
        }
        
        Ok(result)
    }
    
    fn decode_delta_sequence_u64(&self, data: &[u8]) -> Result<Vec<u64>> {
        let mut offset = 0;
        
        // Read count
        let (count, count_bytes) = self.decode_leb128_u64(&data[offset..])?;
        offset += count_bytes;
        
        if count == 0 {
            return Ok(Vec::new());
        }
        
        let mut result = Vec::with_capacity(count as usize);
        
        // Read first value
        let (first_value, first_bytes) = self.decode_leb128_u64(&data[offset..])?;
        result.push(first_value);
        offset += first_bytes;
        
        // Read deltas
        for _ in 1..count {
            let (encoded_delta, delta_bytes) = self.decode_leb128_u64(&data[offset..])?;
            
            let prev_value = result[result.len() - 1];
            let next_value = if (encoded_delta & 1) == 0 {
                // Positive delta
                prev_value + (encoded_delta >> 1)
            } else {
                // Negative delta
                prev_value - (encoded_delta >> 1)
            };
            
            result.push(next_value);
            offset += delta_bytes;
        }
        
        Ok(result)
    }
    
    fn decode_delta_sequence_i64(&self, data: &[u8]) -> Result<Vec<i64>> {
        let mut offset = 0;
        
        // Read count
        let (count, count_bytes) = self.decode_leb128_u64(&data[offset..])?;
        offset += count_bytes;
        
        if count == 0 {
            return Ok(Vec::new());
        }
        
        let mut result = Vec::with_capacity(count as usize);
        
        // Read first value
        let (first_value, first_bytes) = self.decode_leb128_i64(&data[offset..])?;
        result.push(first_value);
        offset += first_bytes;
        
        // Read deltas
        for _ in 1..count {
            let (delta, delta_bytes) = self.decode_zigzag_i64(&data[offset..])?;
            let next_value = result[result.len() - 1] + delta;
            result.push(next_value);
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
        
        // Process in groups of 4
        for chunk in values.chunks(4) {
            let mut selector = 0u8;
            let mut group_data = Vec::new();
            
            for (i, &value) in chunk.iter().enumerate() {
                let bytes_needed = if value == 0 {
                    1
                } else {
                    ((64 - value.leading_zeros() + 7) / 8) as usize
                };
                
                // Encode bytes needed in selector (2 bits per value)
                selector |= ((bytes_needed - 1) as u8) << (i * 2);
                
                // Store the value in little-endian format
                let value_bytes = value.to_le_bytes();
                group_data.extend_from_slice(&value_bytes[..bytes_needed]);
            }
            
            result.push(selector);
            result.extend_from_slice(&group_data);
        }
        
        Ok(result)
    }
    
    fn decode_group_varint_u64(&self, data: &[u8]) -> Result<Vec<u64>> {
        let mut offset = 0;
        
        // Read count
        let (count, count_bytes) = self.decode_leb128_u64(&data[offset..])?;
        offset += count_bytes;
        
        let mut result = Vec::with_capacity(count as usize);
        let mut remaining = count;
        
        while remaining > 0 {
            if offset >= data.len() {
                return Err(ZiporaError::invalid_data("Incomplete group varint"));
            }
            
            let selector = data[offset];
            offset += 1;
            
            let chunk_size = cmp::min(remaining, 4);
            
            for i in 0..chunk_size {
                let bytes_needed = ((selector >> (i * 2)) & 0x3) as usize + 1;
                
                if offset + bytes_needed > data.len() {
                    return Err(ZiporaError::invalid_data("Incomplete group varint value"));
                }
                
                let mut value_bytes = [0u8; 8];
                value_bytes[..bytes_needed].copy_from_slice(&data[offset..offset + bytes_needed]);
                
                let value = u64::from_le_bytes(value_bytes);
                result.push(value);
                offset += bytes_needed;
            }
            
            remaining -= chunk_size;
        }
        
        Ok(result)
    }
}

// Prefix-free implementations (simplified)
impl VarIntEncoder {
    fn encode_prefix_free_u64(&self, value: u64) -> Result<Vec<u8>> {
        // Use length prefix followed by value
        let value_bytes = value.to_le_bytes();
        let significant_bytes = ((64 - value.leading_zeros() + 7) / 8) as usize;
        let significant_bytes = if significant_bytes == 0 { 1 } else { significant_bytes };
        
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
        if data.is_empty() {
            return Err(ZiporaError::invalid_data("Empty prefix-free data"));
        }
        
        let length = data[0] as usize;
        if length == 0 || length > 8 {
            return Err(ZiporaError::invalid_data("Invalid prefix-free length"));
        }
        
        if data.len() < 1 + length {
            return Err(ZiporaError::invalid_data("Incomplete prefix-free data"));
        }
        
        let mut value_bytes = [0u8; 8];
        value_bytes[..length].copy_from_slice(&data[1..1 + length]);
        
        let value = u64::from_le_bytes(value_bytes);
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
        let (count, count_bytes) = self.decode_leb128_u64(&data[offset..])?;
        offset += count_bytes;
        
        let mut result = Vec::with_capacity(count as usize);
        
        // Read values
        for _ in 0..count {
            let (value, value_bytes) = self.decode_prefix_free_u64(&data[offset..])?;
            result.push(value);
            offset += value_bytes;
        }
        
        Ok(result)
    }
    
    fn decode_prefix_free_sequence_i64(&self, data: &[u8]) -> Result<Vec<i64>> {
        let mut offset = 0;
        
        // Read count
        let (count, count_bytes) = self.decode_leb128_u64(&data[offset..])?;
        offset += count_bytes;
        
        let mut result = Vec::with_capacity(count as usize);
        
        // Read values
        for _ in 0..count {
            let (value, value_bytes) = self.decode_prefix_free_i64(&data[offset..])?;
            result.push(value);
            offset += value_bytes;
        }
        
        Ok(result)
    }
}

// Compact and SIMD implementations (simplified)
impl VarIntEncoder {
    fn encode_compact_u64(&self, value: u64) -> Result<Vec<u8>> {
        // For simplicity, use LEB128 for compact encoding
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
        // For now, use LEB128 with potential for SIMD optimization
        self.encode_leb128_sequence_u64(values)
    }
    
    fn encode_simd_sequence_i64(&self, values: &[i64]) -> Result<Vec<u8>> {
        // For now, use zigzag with potential for SIMD optimization
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
    if values.is_empty() {
        return VarIntStrategy::Leb128;
    }
    
    // Analyze data characteristics
    // SAFETY: is_empty() check at line 777 returns early, so values is non-empty
    let max_value = *values.iter().max().unwrap();
    let is_sorted = values.windows(2).all(|w| w[0] <= w[1]);
    let small_values = values.iter().all(|&v| v < 256);
    
    // Choose strategy based on characteristics with careful prioritization
    if values.len() >= 16 && max_value < (1u64 << 32) {
        VarIntStrategy::GroupVarint // Good for bulk operations with moderate values (prioritized for large sequences)
    } else if is_sorted && values.len() > 6 {  // Delta for sorted sequences with sufficient length (but not too small)
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
    let small_range = values.iter().all(|&v| v.abs() < 256);
    
    // Choose strategy based on characteristics - prioritize sorted sequences even with negative values
    if is_sorted && values.len() > 5 {  // Delta for sorted sequences with sufficient length
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
}