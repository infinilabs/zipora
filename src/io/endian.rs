//! Comprehensive endian handling for I/O operations
//!
//! This module provides type-safe, zero-overhead endian conversions with compile-time
//! optimization and hardware acceleration. Supports both big and little endian formats
//! with automatic detection and conversion.

use crate::error::{Result, ZiporaError};
use std::mem::size_of;

/// Endian format specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    /// Little endian byte order (least significant byte first)
    Little,
    /// Big endian byte order (most significant byte first)
    Big,
    /// Native endian byte order (platform default)
    Native,
}

impl Endianness {
    /// Get the native endianness of the current platform
    #[inline]
    pub const fn native() -> Self {
        #[cfg(target_endian = "little")]
        {
            Self::Little
        }
        #[cfg(target_endian = "big")]
        {
            Self::Big
        }
    }

    /// Check if this endianness matches the native endianness
    #[inline]
    pub const fn is_native(self) -> bool {
        matches!(
            (self, Self::native()),
            (Self::Native, _) | (Self::Little, Self::Little) | (Self::Big, Self::Big)
        )
    }

    /// Check if conversion is needed from this endianness to native
    #[inline]
    pub const fn needs_conversion(self) -> bool {
        !self.is_native()
    }
}

/// Trait for types that support endian conversion
pub trait EndianConvert: Sized + Copy {
    /// Convert from little endian to native endianness
    fn from_le(self) -> Self;
    
    /// Convert from big endian to native endianness
    fn from_be(self) -> Self;
    
    /// Convert from native endianness to little endian
    fn to_le(self) -> Self;
    
    /// Convert from native endianness to big endian
    fn to_be(self) -> Self;
    
    /// Convert from specified endianness to native
    #[inline]
    fn from_endian(self, endian: Endianness) -> Self {
        match endian {
            Endianness::Little => self.from_le(),
            Endianness::Big => self.from_be(),
            Endianness::Native => self,
        }
    }
    
    /// Convert from native to specified endianness
    #[inline]
    fn to_endian(self, endian: Endianness) -> Self {
        match endian {
            Endianness::Little => self.to_le(),
            Endianness::Big => self.to_be(),
            Endianness::Native => self,
        }
    }
    
    /// Check if this type needs byte swapping for the given endianness
    #[inline]
    fn needs_swap_for(endian: Endianness) -> bool {
        endian.needs_conversion()
    }
}

// Macro to implement EndianConvert for primitive integer types
macro_rules! impl_endian_convert {
    ($($t:ty),*) => {
        $(
            impl EndianConvert for $t {
                #[inline]
                fn from_le(self) -> Self {
                    <$t>::from_le(self)
                }
                
                #[inline]
                fn from_be(self) -> Self {
                    <$t>::from_be(self)
                }
                
                #[inline]
                fn to_le(self) -> Self {
                    <$t>::to_le(self)
                }
                
                #[inline]
                fn to_be(self) -> Self {
                    <$t>::to_be(self)
                }
            }
        )*
    };
}

// Implement for multi-byte integer types
impl_endian_convert!(u16, u32, u64, u128, usize);
impl_endian_convert!(i16, i32, i64, i128, isize);

// Special implementation for single-byte types (no conversion needed)
impl EndianConvert for u8 {
    #[inline]
    fn from_le(self) -> Self { self }
    #[inline]
    fn from_be(self) -> Self { self }
    #[inline]
    fn to_le(self) -> Self { self }
    #[inline]
    fn to_be(self) -> Self { self }
    #[inline]
    fn needs_swap_for(_endian: Endianness) -> bool { false }
}

impl EndianConvert for i8 {
    #[inline]
    fn from_le(self) -> Self { self }
    #[inline]
    fn from_be(self) -> Self { self }
    #[inline]
    fn to_le(self) -> Self { self }
    #[inline]
    fn to_be(self) -> Self { self }
    #[inline]
    fn needs_swap_for(_endian: Endianness) -> bool { false }
}

// Floating point implementations
impl EndianConvert for f32 {
    #[inline]
    fn from_le(self) -> Self {
        f32::from_bits(self.to_bits().from_le())
    }
    
    #[inline]
    fn from_be(self) -> Self {
        f32::from_bits(self.to_bits().from_be())
    }
    
    #[inline]
    fn to_le(self) -> Self {
        f32::from_bits(self.to_bits().to_le())
    }
    
    #[inline]
    fn to_be(self) -> Self {
        f32::from_bits(self.to_bits().to_be())
    }
}

impl EndianConvert for f64 {
    #[inline]
    fn from_le(self) -> Self {
        f64::from_bits(self.to_bits().from_le())
    }
    
    #[inline]
    fn from_be(self) -> Self {
        f64::from_bits(self.to_bits().from_be())
    }
    
    #[inline]
    fn to_le(self) -> Self {
        f64::from_bits(self.to_bits().to_le())
    }
    
    #[inline]
    fn to_be(self) -> Self {
        f64::from_bits(self.to_bits().to_be())
    }
}

/// High-performance endian-aware I/O operations
pub struct EndianIO<T> {
    endianness: Endianness,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: EndianConvert> EndianIO<T> {
    /// Create a new EndianIO with specified endianness
    #[inline]
    pub const fn new(endianness: Endianness) -> Self {
        Self {
            endianness,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Create EndianIO for little endian format
    #[inline]
    pub const fn little_endian() -> Self {
        Self::new(Endianness::Little)
    }
    
    /// Create EndianIO for big endian format
    #[inline]
    pub const fn big_endian() -> Self {
        Self::new(Endianness::Big)
    }
    
    /// Create EndianIO for native endian format
    #[inline]
    pub const fn native_endian() -> Self {
        Self::new(Endianness::Native)
    }
    
    /// Read a value from bytes with endian conversion
    #[inline]
    pub fn read_from_bytes(&self, bytes: &[u8]) -> Result<T> {
        if bytes.len() < size_of::<T>() {
            return Err(ZiporaError::invalid_data(
                "Insufficient bytes for type"
            ));
        }
        
        let value = unsafe {
            std::ptr::read_unaligned(bytes.as_ptr() as *const T)
        };
        
        Ok(value.from_endian(self.endianness))
    }
    
    /// Write a value to bytes with endian conversion
    #[inline]
    pub fn write_to_bytes(&self, value: T, bytes: &mut [u8]) -> Result<()> {
        if bytes.len() < size_of::<T>() {
            return Err(ZiporaError::invalid_data(
                "Insufficient buffer size for type"
            ));
        }
        
        let converted = value.to_endian(self.endianness);
        unsafe {
            std::ptr::write_unaligned(bytes.as_mut_ptr() as *mut T, converted);
        }
        
        Ok(())
    }
    
    /// Convert a slice of values to the specified endianness
    pub fn convert_slice_to_endian(&self, values: &mut [T]) {
        if !self.endianness.needs_conversion() {
            return; // No conversion needed for native endianness
        }
        
        for value in values.iter_mut() {
            *value = value.to_endian(self.endianness);
        }
    }
    
    /// Convert a slice of values from the specified endianness to native
    pub fn convert_slice_from_endian(&self, values: &mut [T]) {
        if !self.endianness.needs_conversion() {
            return; // No conversion needed for native endianness
        }
        
        for value in values.iter_mut() {
            *value = value.from_endian(self.endianness);
        }
    }
    
    /// Get the endianness
    #[inline]
    pub const fn endianness(&self) -> Endianness {
        self.endianness
    }
    
    /// Check if this EndianIO needs conversion
    #[inline]
    pub const fn needs_conversion(&self) -> bool {
        self.endianness.needs_conversion()
    }
}

/// SIMD-accelerated bulk endian conversion for supported types
#[cfg(target_arch = "x86_64")]
pub mod simd {
    use super::{Endianness, EndianConvert, EndianIO, EndianConfig, detect_endianness_from_magic, ENDIAN_MAGIC_LITTLE, ENDIAN_MAGIC_BIG};
    
    #[cfg(target_feature = "sse2")]
    /// SIMD-accelerated conversion for u16 arrays
    pub fn convert_u16_slice_simd(values: &mut [u16], from_little: bool) {
        if !from_little == cfg!(target_endian = "little") {
            return; // No conversion needed
        }
        
        // Use SIMD for bulk conversion when available
        #[cfg(target_feature = "sse2")]
        {
            use std::arch::x86_64::*;
            
            let mut chunks = values.chunks_exact_mut(8);
            let chunk_iter: Vec<_> = chunks.by_ref().collect();
            let remainder = chunks.into_remainder();
            
            for chunk in chunk_iter {
                unsafe {
                    let ptr = chunk.as_mut_ptr() as *mut __m128i;
                    let data = _mm_loadu_si128(ptr);
                    
                    // Byte swap using SIMD shuffle
                    let swapped = _mm_or_si128(
                        _mm_slli_epi16(data, 8),
                        _mm_srli_epi16(data, 8)
                    );
                    
                    _mm_storeu_si128(ptr, swapped);
                }
            }
            
            // Handle remainder with scalar operations
            for value in remainder {
                *value = value.swap_bytes();
            }
        }
    }
    
    #[cfg(target_feature = "sse2")]
    /// SIMD-accelerated conversion for u32 arrays
    pub fn convert_u32_slice_simd(values: &mut [u32], from_little: bool) {
        if !from_little == cfg!(target_endian = "little") {
            return; // No conversion needed
        }
        
        #[cfg(target_feature = "sse2")]
        {
            use std::arch::x86_64::*;
            
            let mut chunks = values.chunks_exact_mut(4);
            let chunk_iter: Vec<_> = chunks.by_ref().collect();
            let remainder = chunks.into_remainder();
            
            for chunk in chunk_iter {
                unsafe {
                    let ptr = chunk.as_mut_ptr() as *mut __m128i;
                    let data = _mm_loadu_si128(ptr);
                    
                    // 32-bit byte swap using SIMD
                    let swapped = _mm_shuffle_epi8(data, 
                        _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3));
                    
                    _mm_storeu_si128(ptr, swapped);
                }
            }
            
            // Handle remainder
            for value in remainder {
                *value = value.swap_bytes();
            }
        }
    }
}

/// Builder for configuring endian handling
pub struct EndianConfig {
    default_endianness: Endianness,
    auto_detect: bool,
    simd_acceleration: bool,
}

impl EndianConfig {
    /// Create a new endian configuration
    pub fn new() -> Self {
        Self {
            default_endianness: Endianness::Native,
            auto_detect: false,
            simd_acceleration: true,
        }
    }
    
    /// Set the default endianness
    pub fn with_default_endianness(mut self, endianness: Endianness) -> Self {
        self.default_endianness = endianness;
        self
    }
    
    /// Enable automatic endianness detection from data headers
    pub fn with_auto_detect(mut self, enable: bool) -> Self {
        self.auto_detect = enable;
        self
    }
    
    /// Enable SIMD acceleration for bulk conversions
    pub fn with_simd_acceleration(mut self, enable: bool) -> Self {
        self.simd_acceleration = enable;
        self
    }
    
    /// Create configuration optimized for performance
    pub fn performance_optimized() -> Self {
        Self {
            default_endianness: Endianness::Native,
            auto_detect: false,
            simd_acceleration: true,
        }
    }
    
    /// Create configuration for cross-platform compatibility
    pub fn cross_platform() -> Self {
        Self {
            default_endianness: Endianness::Little, // Most common
            auto_detect: true,
            simd_acceleration: true,
        }
    }
}

impl Default for EndianConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Magic number for endianness detection in file headers
pub const ENDIAN_MAGIC_LITTLE: u32 = 0x12345678;
pub const ENDIAN_MAGIC_BIG: u32 = 0x78563412;

/// Detect endianness from a magic number
pub fn detect_endianness_from_magic(magic: u32) -> Option<Endianness> {
    match magic {
        ENDIAN_MAGIC_LITTLE => Some(Endianness::Little),
        ENDIAN_MAGIC_BIG => Some(Endianness::Big),
        _ => None,
    }
}

/// Write endianness magic number to identify format
pub fn write_endianness_magic(endianness: Endianness) -> u32 {
    match endianness {
        Endianness::Little | Endianness::Native if Endianness::native() == Endianness::Little => {
            ENDIAN_MAGIC_LITTLE
        }
        Endianness::Big | Endianness::Native if Endianness::native() == Endianness::Big => {
            ENDIAN_MAGIC_BIG
        }
        _ => ENDIAN_MAGIC_LITTLE, // Default to little endian
    }
}

#[cfg(test)]
mod tests {
    use super::{Endianness, EndianConvert, EndianIO, EndianConfig, detect_endianness_from_magic, ENDIAN_MAGIC_LITTLE, ENDIAN_MAGIC_BIG};

    #[test]
    fn test_endianness_detection() {
        assert_eq!(Endianness::native().is_native(), true);
        
        #[cfg(target_endian = "little")]
        {
            assert_eq!(Endianness::native(), Endianness::Little);
            assert!(Endianness::Little.is_native());
            assert!(!Endianness::Big.is_native());
        }
        
        #[cfg(target_endian = "big")]
        {
            assert_eq!(Endianness::native(), Endianness::Big);
            assert!(Endianness::Big.is_native());
            assert!(!Endianness::Little.is_native());
        }
    }

    #[test]
    fn test_basic_endian_conversion() {
        let value: u32 = 0x12345678;
        
        // Test round-trip conversions
        assert_eq!(value.to_le().from_le(), value);
        assert_eq!(value.to_be().from_be(), value);
        
        // Test endianness-specific conversions
        let le_bytes = value.to_le_bytes();
        let be_bytes = value.to_be_bytes();
        
        assert_eq!(u32::from_le_bytes(le_bytes), value);
        assert_eq!(u32::from_be_bytes(be_bytes), value);
    }

    #[test]
    fn test_endian_io_operations() {
        let value: u32 = 0x12345678;
        let mut buffer = [0u8; 4];
        
        // Test little endian I/O
        let le_io = EndianIO::<u32>::little_endian();
        le_io.write_to_bytes(value, &mut buffer).unwrap();
        let read_value = le_io.read_from_bytes(&buffer).unwrap();
        assert_eq!(read_value, value);
        
        // Test big endian I/O
        let be_io = EndianIO::<u32>::big_endian();
        be_io.write_to_bytes(value, &mut buffer).unwrap();
        let read_value = be_io.read_from_bytes(&buffer).unwrap();
        assert_eq!(read_value, value);
        
        // Verify that little and big endian produce different byte patterns
        let mut le_buffer = [0u8; 4];
        let mut be_buffer = [0u8; 4];
        
        le_io.write_to_bytes(value, &mut le_buffer).unwrap();
        be_io.write_to_bytes(value, &mut be_buffer).unwrap();
        
        if cfg!(target_endian = "little") {
            assert_ne!(le_buffer, be_buffer);
        }
    }

    #[test]
    fn test_single_byte_types() {
        let value: u8 = 0x42;
        
        // Single byte types should not need conversion
        assert!(!u8::needs_swap_for(Endianness::Little));
        assert!(!u8::needs_swap_for(Endianness::Big));
        assert!(!i8::needs_swap_for(Endianness::Little));
        assert!(!i8::needs_swap_for(Endianness::Big));
        
        // All operations should be no-ops
        assert_eq!(value.to_le(), value);
        assert_eq!(value.to_be(), value);
        assert_eq!(value.from_le(), value);
        assert_eq!(value.from_be(), value);
    }

    #[test]
    fn test_floating_point_endian() {
        let value: f32 = 3.14159;
        
        // Test round-trip conversion
        assert_eq!(value.to_le().from_le(), value);
        assert_eq!(value.to_be().from_be(), value);
        
        let value: f64 = 2.718281828459045;
        assert_eq!(value.to_le().from_le(), value);
        assert_eq!(value.to_be().from_be(), value);
    }

    #[test]
    fn test_slice_conversion() {
        let mut values = vec![0x1234u16, 0x5678u16, 0x9abcu16, 0xdef0u16];
        let original = values.clone();
        
        let le_io = EndianIO::<u16>::little_endian();
        
        // Convert to endian and back
        le_io.convert_slice_to_endian(&mut values);
        le_io.convert_slice_from_endian(&mut values);
        
        assert_eq!(values, original);
    }

    #[test]
    fn test_magic_number_detection() {
        assert_eq!(
            detect_endianness_from_magic(ENDIAN_MAGIC_LITTLE),
            Some(Endianness::Little)
        );
        assert_eq!(
            detect_endianness_from_magic(ENDIAN_MAGIC_BIG),
            Some(Endianness::Big)
        );
        assert_eq!(detect_endianness_from_magic(0xdeadbeef), None);
    }

    #[test]
    fn test_endian_config() {
        let config = EndianConfig::performance_optimized();
        assert_eq!(config.default_endianness, Endianness::Native);
        assert!(!config.auto_detect);
        assert!(config.simd_acceleration);
        
        let config = EndianConfig::cross_platform();
        assert_eq!(config.default_endianness, Endianness::Little);
        assert!(config.auto_detect);
        assert!(config.simd_acceleration);
    }

    #[test]
    fn test_insufficient_buffer_error() {
        let value: u32 = 0x12345678;
        let mut small_buffer = [0u8; 2]; // Too small for u32
        
        let io = EndianIO::<u32>::native_endian();
        let result = io.write_to_bytes(value, &mut small_buffer);
        assert!(result.is_err());
        
        let result = io.read_from_bytes(&small_buffer);
        assert!(result.is_err());
    }

    #[test]
    fn test_needs_conversion() {
        let native_io = EndianIO::<u32>::native_endian();
        assert!(!native_io.needs_conversion());
        
        #[cfg(target_endian = "little")]
        {
            let be_io = EndianIO::<u32>::big_endian();
            assert!(be_io.needs_conversion());
            
            let le_io = EndianIO::<u32>::little_endian();
            assert!(!le_io.needs_conversion());
        }
        
        #[cfg(target_endian = "big")]
        {
            let le_io = EndianIO::<u32>::little_endian();
            assert!(le_io.needs_conversion());
            
            let be_io = EndianIO::<u32>::big_endian();
            assert!(!be_io.needs_conversion());
        }
    }

    #[test]
    fn test_endian_conversion_traits() {
        let value: u32 = 0x12345678;
        
        // Test from_endian and to_endian methods
        assert_eq!(value.from_endian(Endianness::Native), value);
        assert_eq!(value.to_endian(Endianness::Native), value);
        
        // Test that conversion is consistent
        let le_converted = value.to_endian(Endianness::Little);
        assert_eq!(le_converted.from_endian(Endianness::Little), value);
        
        let be_converted = value.to_endian(Endianness::Big);
        assert_eq!(be_converted.from_endian(Endianness::Big), value);
    }
}