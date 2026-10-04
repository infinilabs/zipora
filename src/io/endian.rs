//! Endian handling for I/O operations
//!
//! [`EndianConvert`] converts plain-old-data values between little-endian,
//! big-endian and host byte order, and [`EndianIO`] reads and writes them
//! from and to byte slices with bounds checks. On x86_64 the `simd` module
//! byte-swaps `u16`/`u32` slices with SSE2. Byte order is never detected
//! automatically: [`detect_endianness_from_magic`] maps a header value
//! written by [`write_endianness_magic`] back to an [`Endianness`].

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
///
/// The [`bytemuck::Pod`] supertrait is what makes [`EndianIO`] sound: it
/// copies raw bytes into and out of `T`, which is only defined behaviour
/// when every bit pattern is a valid `T` and `T` has no padding bytes.
/// Types such as `bool`, `char`, enums, references or `NonZero*`, and
/// structs with padding, cannot implement this trait:
///
/// ```compile_fail,E0277
/// use zipora::io::endian::EndianConvert;
///
/// #[derive(Clone, Copy)]
/// struct Flag(bool);
///
/// impl EndianConvert for Flag {
///     fn from_le(self) -> Self { self }
///     fn from_be(self) -> Self { self }
///     fn to_le(self) -> Self { self }
///     fn to_be(self) -> Self { self }
/// }
/// ```
///
/// A plain-old-data newtype can:
///
/// ```
/// use zipora::io::endian::{EndianConvert, EndianIO};
///
/// #[derive(Clone, Copy, Debug, PartialEq)]
/// #[repr(transparent)]
/// struct Id(u32);
///
/// // SAFETY: `Id` is a `repr(transparent)` wrapper around `u32`: it has no
/// // padding and every bit pattern is a valid value.
/// unsafe impl bytemuck::Zeroable for Id {}
/// // SAFETY: as above; `Id` is also `Copy` and `'static`.
/// unsafe impl bytemuck::Pod for Id {}
///
/// impl EndianConvert for Id {
///     fn from_le(self) -> Self { Id(u32::from_le(self.0)) }
///     fn from_be(self) -> Self { Id(u32::from_be(self.0)) }
///     fn to_le(self) -> Self { Id(self.0.to_le()) }
///     fn to_be(self) -> Self { Id(self.0.to_be()) }
/// }
///
/// let mut buf = [0u8; 4];
/// EndianIO::<Id>::big_endian().write_to_bytes(Id(0x0102_0304), &mut buf).unwrap();
/// assert_eq!(buf, [1, 2, 3, 4]);
/// ```
pub trait EndianConvert: bytemuck::Pod {
    /// Convert from little endian to native endianness
    #[allow(clippy::wrong_self_convention)] // conventional prefix `from_` is for constructors; here it is a method converting FROM LE
    fn from_le(self) -> Self;

    /// Convert from big endian to native endianness
    #[allow(clippy::wrong_self_convention)] // conventional prefix `from_` is for constructors; here it is a method converting FROM BE
    fn from_be(self) -> Self;

    /// Convert from native endianness to little endian
    fn to_le(self) -> Self;

    /// Convert from native endianness to big endian
    fn to_be(self) -> Self;

    /// Convert from specified endianness to native
    #[inline]
    #[allow(clippy::wrong_self_convention)] // conventional prefix `from_` is for constructors
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
    fn from_le(self) -> Self {
        self
    }
    #[inline]
    fn from_be(self) -> Self {
        self
    }
    #[inline]
    fn to_le(self) -> Self {
        self
    }
    #[inline]
    fn to_be(self) -> Self {
        self
    }
    #[inline]
    fn needs_swap_for(_endian: Endianness) -> bool {
        false
    }
}

impl EndianConvert for i8 {
    #[inline]
    fn from_le(self) -> Self {
        self
    }
    #[inline]
    fn from_be(self) -> Self {
        self
    }
    #[inline]
    fn to_le(self) -> Self {
        self
    }
    #[inline]
    fn to_be(self) -> Self {
        self
    }
    #[inline]
    fn needs_swap_for(_endian: Endianness) -> bool {
        false
    }
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
    ///
    /// Reads the first `size_of::<T>()` bytes; a shorter slice is an error.
    #[inline]
    pub fn read_from_bytes(&self, bytes: &[u8]) -> Result<T> {
        let Some(raw) = bytes.get(..size_of::<T>()) else {
            return Err(ZiporaError::invalid_data("Insufficient bytes for type"));
        };
        // `T: Pod` makes every byte pattern a valid `T`; `raw` has exactly
        // `size_of::<T>()` bytes, so the conversion cannot fail.
        let value: T = bytemuck::try_pod_read_unaligned(raw)
            .map_err(|_| ZiporaError::invalid_data("Insufficient bytes for type"))?;

        Ok(value.from_endian(self.endianness))
    }

    /// Write a value to bytes with endian conversion
    ///
    /// Writes the first `size_of::<T>()` bytes and leaves the rest alone; a
    /// shorter slice is an error.
    #[inline]
    pub fn write_to_bytes(&self, value: T, bytes: &mut [u8]) -> Result<()> {
        let Some(out) = bytes.get_mut(..size_of::<T>()) else {
            return Err(ZiporaError::invalid_data(
                "Insufficient buffer size for type",
            ));
        };

        let converted = value.to_endian(self.endianness);
        // `T: Pod` has no padding, so all of its bytes are initialised.
        out.copy_from_slice(bytemuck::bytes_of(&converted));

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
///
/// Both functions take values whose bytes are in little-endian order
/// (`from_little == true`) or big-endian order (`from_little == false`) and
/// turn them into host order: a no-op when that order is the host's, a
/// byte swap of every element otherwise. Whole 16-byte blocks are swapped
/// with SSE2, the remaining elements with `swap_bytes`.
#[cfg(target_arch = "x86_64")]
pub mod simd {
    #[cfg(target_feature = "sse2")]
    /// SIMD-accelerated conversion for u16 arrays
    pub fn convert_u16_slice_simd(values: &mut [u16], from_little: bool) {
        if from_little == cfg!(target_endian = "little") {
            return; // Already in host byte order
        }

        // Use SIMD for bulk conversion when available
        #[cfg(target_feature = "sse2")]
        {
            use std::arch::x86_64::*;

            let mut chunks = values.chunks_exact_mut(8);
            for chunk in chunks.by_ref() {
                let ptr = chunk.as_mut_ptr() as *mut __m128i;
                // SAFETY: `chunk` is exactly 8 `u16`s = 16 bytes, so the
                // unaligned 16-byte load and store touch only `chunk`, which
                // this loop borrows mutably. The intrinsics need SSE2 only,
                // which the build enables (`cfg(target_feature = "sse2")`).
                unsafe {
                    let data = _mm_loadu_si128(ptr);

                    // Swap the two bytes of every 16-bit lane
                    let swapped = _mm_or_si128(_mm_slli_epi16(data, 8), _mm_srli_epi16(data, 8));

                    _mm_storeu_si128(ptr, swapped);
                }
            }

            // Handle remainder with scalar operations
            for value in chunks.into_remainder() {
                *value = value.swap_bytes();
            }
        }
    }

    #[cfg(target_feature = "sse2")]
    /// SIMD-accelerated conversion for u32 arrays
    pub fn convert_u32_slice_simd(values: &mut [u32], from_little: bool) {
        if from_little == cfg!(target_endian = "little") {
            return; // Already in host byte order
        }

        #[cfg(target_feature = "sse2")]
        {
            use std::arch::x86_64::*;

            let mut chunks = values.chunks_exact_mut(4);
            for chunk in chunks.by_ref() {
                let ptr = chunk.as_mut_ptr() as *mut __m128i;
                // SAFETY: `chunk` is exactly 4 `u32`s = 16 bytes, so the
                // unaligned 16-byte load and store touch only `chunk`, which
                // this loop borrows mutably. The intrinsics need SSE2 only,
                // which the build enables (`cfg(target_feature = "sse2")`).
                unsafe {
                    let data = _mm_loadu_si128(ptr);

                    // 32-bit byte swap with SSE2 only: swap the bytes of each
                    // 16-bit lane, then the 16-bit halves of each 32-bit lane,
                    // [b0 b1 b2 b3] -> [b1 b0 b3 b2] -> [b3 b2 b1 b0].
                    let halves_swapped =
                        _mm_or_si128(_mm_slli_epi16(data, 8), _mm_srli_epi16(data, 8));
                    let swapped = _mm_or_si128(
                        _mm_slli_epi32(halves_swapped, 16),
                        _mm_srli_epi32(halves_swapped, 16),
                    );

                    _mm_storeu_si128(ptr, swapped);
                }
            }

            // Handle remainder
            for value in chunks.into_remainder() {
                *value = value.swap_bytes();
            }
        }
    }
}

/// Builder for configuring endian handling
///
/// Limitation: a configuration only records preferences, readable through
/// [`EndianConfig::default_endianness`], [`EndianConfig::auto_detect`] and
/// [`EndianConfig::simd_acceleration`]; nothing in zipora acts on them.
/// [`EndianIO`] takes its byte order from its constructor and never detects
/// one, and the x86_64 `simd` functions are called directly.
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

    /// Record whether to detect endianness from data headers. Limitation:
    /// nothing in zipora reads this flag; see [`EndianConfig`].
    pub fn with_auto_detect(mut self, enable: bool) -> Self {
        self.auto_detect = enable;
        self
    }

    /// Record whether to use SIMD for bulk conversions. Limitation: nothing
    /// in zipora reads this flag; see [`EndianConfig`].
    pub fn with_simd_acceleration(mut self, enable: bool) -> Self {
        self.simd_acceleration = enable;
        self
    }

    /// The default endianness recorded in this configuration
    pub const fn default_endianness(&self) -> Endianness {
        self.default_endianness
    }

    /// Whether this configuration asks for detection from data headers
    pub const fn auto_detect(&self) -> bool {
        self.auto_detect
    }

    /// Whether this configuration asks for SIMD bulk conversions
    pub const fn simd_acceleration(&self) -> bool {
        self.simd_acceleration
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
///
/// [`write_endianness_magic`] returns it for [`Endianness::Little`]. Store
/// a magic value in one fixed byte order and read it back in the same order
/// (for example `u32::to_le_bytes` and `u32::from_le_bytes`).
pub const ENDIAN_MAGIC_LITTLE: u32 = 0x12345678;
/// Magic number for big-endian data: [`ENDIAN_MAGIC_LITTLE`] with its bytes
/// reversed. [`write_endianness_magic`] returns it for [`Endianness::Big`].
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
///
/// Returns the magic number that [`detect_endianness_from_magic`] maps back
/// to `endianness`; [`Endianness::Native`] is resolved to the host's byte
/// order first.
pub fn write_endianness_magic(endianness: Endianness) -> u32 {
    match endianness {
        Endianness::Little => ENDIAN_MAGIC_LITTLE,
        Endianness::Big => ENDIAN_MAGIC_BIG,
        Endianness::Native => {
            if cfg!(target_endian = "big") {
                ENDIAN_MAGIC_BIG
            } else {
                ENDIAN_MAGIC_LITTLE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    use super::simd;
    use super::{
        ENDIAN_MAGIC_BIG, ENDIAN_MAGIC_LITTLE, EndianConfig, EndianConvert, EndianIO, Endianness,
        detect_endianness_from_magic, write_endianness_magic,
    };

    #[test]
    fn test_endianness_detection() {
        assert!(Endianness::native().is_native());

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
        let value: f32 = std::f32::consts::PI;

        // Test round-trip conversion
        assert_eq!(value.to_le().from_le(), value);
        assert_eq!(value.to_be().from_be(), value);

        let value: f64 = std::f64::consts::E;
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

    /// C9 E1: `read_from_bytes`/`write_to_bytes` for every primitive impl,
    /// checked against the std `to_le_bytes`/`to_be_bytes`/`to_ne_bytes`
    /// encodings; bytes past the value are left alone and a buffer one byte
    /// too short is rejected.
    #[test]
    fn test_c9_e1_every_primitive_matches_the_std_byte_encodings() {
        macro_rules! check {
            ($($t:ty => $v:expr),* $(,)?) => {$({
                let value: $t = $v;
                const N: usize = std::mem::size_of::<$t>();
                for (io, want) in [
                    (EndianIO::<$t>::little_endian(), value.to_le_bytes()),
                    (EndianIO::<$t>::big_endian(), value.to_be_bytes()),
                    (EndianIO::<$t>::native_endian(), value.to_ne_bytes()),
                ] {
                    let what = (stringify!($t), io.endianness());
                    let mut buf = [0xAAu8; N + 3];
                    io.write_to_bytes(value, &mut buf).unwrap();
                    assert_eq!(&buf[..N], &want[..], "write {what:?}");
                    assert_eq!(&buf[N..], &[0xAAu8; 3][..], "write past the value {what:?}");
                    let back: $t = io.read_from_bytes(&buf).unwrap();
                    assert_eq!(back.to_ne_bytes(), value.to_ne_bytes(), "read {what:?}");
                    assert!(io.read_from_bytes(&buf[..N - 1]).is_err(), "short read {what:?}");
                    assert!(io.write_to_bytes(value, &mut buf[..N - 1]).is_err(), "short write {what:?}");
                }
            })*};
        }
        check!(
            u8 => 0xA5,
            i8 => -91,
            u16 => 0x0102,
            i16 => -0x0102,
            u32 => 0x0102_0304,
            i32 => -0x0102_0304,
            u64 => 0x0102_0304_0506_0708,
            i64 => -0x0102_0304_0506_0708,
            u128 => 0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10,
            i128 => -0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10,
            usize => 0x0102_0304,
            isize => -0x0102_0304,
            f32 => -1.5e-3,
            f64 => 6.02214076e23,
        );
    }

    /// C9 E1: a user type implements `EndianConvert` on top of `bytemuck::Pod`
    /// and round-trips through `EndianIO`.
    #[test]
    fn test_c9_e1_pod_newtype_round_trips() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(transparent)]
        struct Id(u32);
        // SAFETY: `Id` is a `repr(transparent)` wrapper around `u32`: it has
        // no padding and every bit pattern is a valid value.
        unsafe impl bytemuck::Zeroable for Id {}
        // SAFETY: as above; `Id` is also `Copy` and `'static`.
        unsafe impl bytemuck::Pod for Id {}
        impl EndianConvert for Id {
            fn from_le(self) -> Self {
                Id(u32::from_le(self.0))
            }
            fn from_be(self) -> Self {
                Id(u32::from_be(self.0))
            }
            fn to_le(self) -> Self {
                Id(self.0.to_le())
            }
            fn to_be(self) -> Self {
                Id(self.0.to_be())
            }
        }

        let io = EndianIO::<Id>::big_endian();
        let mut buf = [0u8; 4];
        io.write_to_bytes(Id(0x0102_0304), &mut buf).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);
        assert_eq!(io.read_from_bytes(&buf).unwrap(), Id(0x0102_0304));
        let le = EndianIO::<Id>::little_endian();
        le.write_to_bytes(Id(0x0102_0304), &mut buf).unwrap();
        assert_eq!(buf, [4, 3, 2, 1]);
        assert_eq!(le.read_from_bytes(&buf).unwrap(), Id(0x0102_0304));
    }

    /// C9 E2: `convert_*_slice_simd(values, from_little)` turns values whose
    /// bytes are in the `from_little` order into host order: a no-op when
    /// that order is the host's, `swap_bytes` otherwise. Lengths 0..=40 cover
    /// whole 16-byte blocks and every remainder.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[test]
    fn test_c9_e2_simd_slice_conversions_match_swap_bytes_both_directions() {
        let host_little = cfg!(target_endian = "little");
        for len in 0..=40usize {
            let v16: Vec<u16> = (0..len)
                .map(|i| 0x0102u16.wrapping_add((i as u16).wrapping_mul(0x0303)))
                .collect();
            let v32: Vec<u32> = (0..len)
                .map(|i| 0x0102_0304u32.wrapping_add((i as u32).wrapping_mul(0x0505_0505)))
                .collect();
            for from_little in [true, false] {
                let swap = from_little != host_little;
                let mut got16 = v16.clone();
                simd::convert_u16_slice_simd(&mut got16, from_little);
                let want16: Vec<u16> =
                    v16.iter().map(|&x| if swap { x.swap_bytes() } else { x }).collect();
                assert_eq!(got16, want16, "u16 len {len} from_little {from_little}");

                let mut got32 = v32.clone();
                simd::convert_u32_slice_simd(&mut got32, from_little);
                let want32: Vec<u32> =
                    v32.iter().map(|&x| if swap { x.swap_bytes() } else { x }).collect();
                assert_eq!(got32, want32, "u32 len {len} from_little {from_little}");
            }
        }
    }

    /// C9 E3: the u32 block swap must not need more than SSE2, the only
    /// feature its `cfg` gate guarantees. Both directions run, so whichever
    /// one swaps goes through the 16-byte block path; the Miri build without
    /// extra target features reports any SSSE3 instruction.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[test]
    fn test_c9_e3_u32_simd_swap_needs_only_sse2() {
        for from_little in [true, false] {
            let orig = [0x0102_0304u32, 0x0506_0708, 0x090A_0B0C, 0x0D0E_0F10, 0x1112_1314];
            let mut values = orig;
            simd::convert_u32_slice_simd(&mut values, from_little);
            let swap = from_little != cfg!(target_endian = "little");
            let want = orig.map(|x| if swap { x.swap_bytes() } else { x });
            assert_eq!(values, want, "from_little {from_little}");
        }
    }

    /// C9 E4: `write_endianness_magic` is the inverse of
    /// `detect_endianness_from_magic`, with `Native` resolved to the host's
    /// byte order.
    #[test]
    fn test_c9_e4_endianness_magic_round_trips_for_every_variant() {
        assert_ne!(
            write_endianness_magic(Endianness::Big),
            write_endianness_magic(Endianness::Little)
        );
        assert_eq!(
            detect_endianness_from_magic(write_endianness_magic(Endianness::Little)),
            Some(Endianness::Little)
        );
        assert_eq!(
            detect_endianness_from_magic(write_endianness_magic(Endianness::Big)),
            Some(Endianness::Big)
        );
        assert_eq!(
            detect_endianness_from_magic(write_endianness_magic(Endianness::Native)),
            Some(Endianness::native())
        );
    }

    /// C9 E5: the values given to the `EndianConfig` builder can be read
    /// back; the defaults are Native, no detection, SIMD on.
    #[test]
    fn test_c9_e5_config_values_are_readable_through_getters() {
        let config = EndianConfig::new()
            .with_default_endianness(Endianness::Big)
            .with_auto_detect(true)
            .with_simd_acceleration(false);
        assert_eq!(config.default_endianness(), Endianness::Big);
        assert!(config.auto_detect());
        assert!(!config.simd_acceleration());

        let defaults = EndianConfig::default();
        assert_eq!(defaults.default_endianness(), Endianness::Native);
        assert!(!defaults.auto_detect());
        assert!(defaults.simd_acceleration());
    }
}
