//! Complex type serialization for tuples, arrays, and nested structures
//!
//! This module provides automatic serialization and deserialization for complex
//! Rust types including tuples, arrays, Option, Result, and custom structures.
//! Uses recursive template patterns and procedural macros for type-safe serialization.

use crate::error::{Result, ZiporaError};
use crate::io::data_input::prealloc_capacity;
use crate::io::smart_ptr::{DEFAULT_MAX_DEPTH, SerializableType, checked_u32_len};
use crate::io::{DataInput, DataOutput};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::Hash;

/// Trait for complex type serialization with metadata.
///
/// # Limitations
///
/// - **32-bit length prefixes**: the built-in collection and array implementations
///   (`HashMap`, `HashSet`, `BTreeMap`, `BTreeSet`, `[T; N]`) and
///   [`ComplexTypeSerializer::serialize_batch`] encode element counts as a `u32`
///   wire prefix and return [`ZiporaError::InvalidData`] if a length exceeds
///   `u32::MAX` (`4,294,967,295`).
pub trait ComplexSerialize: Sized {
    /// Get the type identifier for versioning
    ///
    /// Limitation: the built-in impls name the container kind and, for
    /// tuples, the arity (`tuple0` to `tuple12`). Element types are not part
    /// of the id, so metadata written for `(u32, u32)` also passes the type
    /// check of `(u64, u64)`.
    fn type_id() -> &'static str;

    /// Get the version of this type's serialization format
    fn version() -> u32 {
        1
    }

    /// Serialize with type metadata
    fn serialize_with_metadata<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        // Write type metadata
        output.write_length_prefixed_string(Self::type_id())?;
        output.write_u32(Self::version())?;

        // Serialize the actual data
        self.serialize_data(output)
    }

    /// Deserialize with type metadata validation
    ///
    /// The type id must equal [`type_id`](Self::type_id). The version is
    /// passed to [`deserialize_with_version`](Self::deserialize_with_version),
    /// which decides which versions it can read; the built-in impls read
    /// only version 1.
    fn deserialize_with_metadata<I: DataInput>(input: &mut I) -> Result<Self> {
        // Read and validate type metadata
        let type_id = input.read_length_prefixed_string()?;
        if type_id != Self::type_id() {
            return Err(ZiporaError::invalid_data(format!(
                "Type mismatch: expected {}, got {}",
                Self::type_id(),
                type_id
            )));
        }

        let version = input.read_u32()?;
        Self::deserialize_with_version(input, version)
    }

    /// Serialize just the data (without metadata)
    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()>;

    /// Deserialize with specific version handling
    fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self>;
}

/// The built-in impls have a single encoding, version 1, which is what
/// [`ComplexSerialize::version`] returns for them.
fn check_builtin_version(type_id: &str, version: u32) -> Result<()> {
    if version == 1 {
        Ok(())
    } else {
        Err(ZiporaError::invalid_data(format!(
            "{} has no serialization version {}; only version 1 exists",
            type_id, version
        )))
    }
}

// Tuple implementations using recursive patterns
macro_rules! impl_tuple_serialize {
    () => {
        impl ComplexSerialize for () {
            fn type_id() -> &'static str { "tuple0" }

            fn serialize_data<O: DataOutput>(&self, _output: &mut O) -> Result<()> {
                Ok(()) // Empty tuple, nothing to serialize
            }

            fn deserialize_with_version<I: DataInput>(_input: &mut I, version: u32) -> Result<Self> {
                check_builtin_version("tuple0", version)?;
                Ok(()) // Empty tuple, nothing to deserialize
            }
        }
    };

    ($id:literal; $($T:ident),*) => {
        impl<$($T: SerializableType),*> ComplexSerialize for ($($T,)*) {
            fn type_id() -> &'static str {
                $id
            }

            fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
                // Macro-generated variables use type names (which are CamelCase)
                #[allow(non_snake_case)]
                let ($($T,)*) = self;
                $(
                    $T.serialize(output)?;
                )*
                Ok(())
            }

            fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self> {
                check_builtin_version($id, version)?;
                Ok(($(
                    $T::deserialize(input)?,
                )*))
            }
        }
    };
}

// Implement for tuples of different arities
impl_tuple_serialize!();
impl_tuple_serialize!("tuple1"; T0);
impl_tuple_serialize!("tuple2"; T0, T1);
impl_tuple_serialize!("tuple3"; T0, T1, T2);
impl_tuple_serialize!("tuple4"; T0, T1, T2, T3);
impl_tuple_serialize!("tuple5"; T0, T1, T2, T3, T4);
impl_tuple_serialize!("tuple6"; T0, T1, T2, T3, T4, T5);
impl_tuple_serialize!("tuple7"; T0, T1, T2, T3, T4, T5, T6);
impl_tuple_serialize!("tuple8"; T0, T1, T2, T3, T4, T5, T6, T7);
impl_tuple_serialize!("tuple9"; T0, T1, T2, T3, T4, T5, T6, T7, T8);
impl_tuple_serialize!("tuple10"; T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_tuple_serialize!("tuple11"; T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_tuple_serialize!("tuple12"; T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);

// Array implementations with const generics
impl<T: SerializableType, const N: usize> ComplexSerialize for [T; N] {
    fn type_id() -> &'static str {
        "array"
    }

    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        // Write array length for validation
        output.write_u32(checked_u32_len(N, "array")?)?;

        for item in self.iter() {
            item.serialize(output)?;
        }
        Ok(())
    }

    fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self> {
        check_builtin_version("array", version)?;
        let length = input.read_u32()? as usize;
        if length != N {
            return Err(ZiporaError::invalid_data(format!(
                "Array length mismatch: expected {}, got {}",
                N, length
            )));
        }

        // Decode into a Vec: if an element fails, the elements decoded so
        // far are dropped with it instead of leaking. `N` is a compile-time
        // constant and was checked against the input above, so the
        // reservation is the size of the array being returned.
        let mut items = Vec::with_capacity(N);
        for _ in 0..N {
            items.push(T::deserialize(input)?);
        }
        <[T; N]>::try_from(items).map_err(|items| {
            ZiporaError::invalid_data(format!(
                "decoded {} elements for an array of {}",
                items.len(),
                N
            ))
        })
    }
}

// Option implementation
impl<T: SerializableType> ComplexSerialize for Option<T> {
    fn type_id() -> &'static str {
        "option"
    }

    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        match self {
            Some(value) => {
                output.write_u8(1)?; // Some marker
                value.serialize(output)
            }
            None => {
                output.write_u8(0)?; // None marker
                Ok(())
            }
        }
    }

    fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self> {
        check_builtin_version("option", version)?;
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(None),
            1 => Ok(Some(T::deserialize(input)?)),
            _ => Err(ZiporaError::invalid_data("Invalid Option marker")),
        }
    }
}

// Result implementation
impl<T: SerializableType, E: SerializableType> ComplexSerialize for std::result::Result<T, E> {
    fn type_id() -> &'static str {
        "result"
    }

    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        match self {
            Ok(value) => {
                output.write_u8(1)?; // Ok marker
                value.serialize(output)
            }
            Err(error) => {
                output.write_u8(0)?; // Err marker
                error.serialize(output)
            }
        }
    }

    fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self> {
        check_builtin_version("result", version)?;
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(Err(E::deserialize(input)?)),
            1 => Ok(Ok(T::deserialize(input)?)),
            _ => Err(ZiporaError::invalid_data("Invalid Result marker")),
        }
    }
}

// HashMap implementation
impl<K: SerializableType + Hash + Eq, V: SerializableType> ComplexSerialize for HashMap<K, V> {
    fn type_id() -> &'static str {
        "hashmap"
    }

    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(checked_u32_len(self.len(), "HashMap")?)?;
        if self.is_empty() {
            return Ok(());
        }

        crate::io::smart_ptr::with_active_ser_ctx(|context| {
            context.run_nested(|| {
                for (key, value) in self {
                    key.serialize(output)?;
                    value.serialize(output)?;
                }
                Ok(())
            })
        })
    }

    fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self> {
        check_builtin_version("hashmap", version)?;
        let len = input.read_u32()? as usize;
        if len == 0 {
            return Ok(HashMap::new());
        }
        // `len` is untrusted: reserve within the read-ahead budget and let
        // the map grow as entries arrive. hashbrown stores one control byte
        // per bucket next to each entry, hence the `+ 1`, and rounds the
        // bucket count up to a power of two at a 7/8 load factor, so the
        // allocation is at most about 2 * 8/7 of the budget.
        let mut map = HashMap::with_capacity(prealloc_capacity(
            len,
            std::mem::size_of::<(K, V)>() + 1,
            input.remaining_bytes(),
        ));

        crate::io::smart_ptr::with_active_de_ctx(|context| {
            context.run_nested(|| {
                for _ in 0..len {
                    let key = K::deserialize(input)?;
                    let value = V::deserialize(input)?;
                    map.insert(key, value);
                }
                Ok(map)
            })
        })
    }
}

// HashSet implementation
impl<T: SerializableType + Hash + Eq> ComplexSerialize for HashSet<T> {
    fn type_id() -> &'static str {
        "hashset"
    }

    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(checked_u32_len(self.len(), "HashSet")?)?;
        if self.is_empty() {
            return Ok(());
        }

        crate::io::smart_ptr::with_active_ser_ctx(|context| {
            context.run_nested(|| {
                for item in self {
                    item.serialize(output)?;
                }
                Ok(())
            })
        })
    }

    fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self> {
        check_builtin_version("hashset", version)?;
        let len = input.read_u32()? as usize;
        if len == 0 {
            return Ok(HashSet::new());
        }
        // Untrusted `len`: see the HashMap impl above.
        let mut set = HashSet::with_capacity(prealloc_capacity(
            len,
            std::mem::size_of::<T>() + 1,
            input.remaining_bytes(),
        ));

        crate::io::smart_ptr::with_active_de_ctx(|context| {
            context.run_nested(|| {
                for _ in 0..len {
                    set.insert(T::deserialize(input)?);
                }
                Ok(set)
            })
        })
    }
}

// BTreeMap implementation
impl<K: SerializableType + Ord, V: SerializableType> ComplexSerialize for BTreeMap<K, V> {
    fn type_id() -> &'static str {
        "btreemap"
    }

    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(checked_u32_len(self.len(), "BTreeMap")?)?;
        if self.is_empty() {
            return Ok(());
        }

        crate::io::smart_ptr::with_active_ser_ctx(|context| {
            context.run_nested(|| {
                for (key, value) in self {
                    key.serialize(output)?;
                    value.serialize(output)?;
                }
                Ok(())
            })
        })
    }

    fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self> {
        check_builtin_version("btreemap", version)?;
        let len = input.read_u32()? as usize;
        if len == 0 {
            return Ok(BTreeMap::new());
        }
        let mut map = BTreeMap::new();

        crate::io::smart_ptr::with_active_de_ctx(|context| {
            context.run_nested(|| {
                for _ in 0..len {
                    let key = K::deserialize(input)?;
                    let value = V::deserialize(input)?;
                    map.insert(key, value);
                }
                Ok(map)
            })
        })
    }
}

// BTreeSet implementation
impl<T: SerializableType + Ord> ComplexSerialize for BTreeSet<T> {
    fn type_id() -> &'static str {
        "btreeset"
    }

    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(checked_u32_len(self.len(), "BTreeSet")?)?;
        if self.is_empty() {
            return Ok(());
        }

        crate::io::smart_ptr::with_active_ser_ctx(|context| {
            context.run_nested(|| {
                for item in self {
                    item.serialize(output)?;
                }
                Ok(())
            })
        })
    }

    fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self> {
        check_builtin_version("btreeset", version)?;
        let len = input.read_u32()? as usize;
        if len == 0 {
            return Ok(BTreeSet::new());
        }
        let mut set = BTreeSet::new();

        crate::io::smart_ptr::with_active_de_ctx(|context| {
            context.run_nested(|| {
                for _ in 0..len {
                    set.insert(T::deserialize(input)?);
                }
                Ok(set)
            })
        })
    }
}

// Bridge implementations for SerializableType

// Bridge: Option<T> as SerializableType
impl<T: SerializableType> SerializableType for Option<T> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as ComplexSerialize>::serialize_data(self, output)
    }

    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as ComplexSerialize>::deserialize_with_version(input, 1)
    }
}

// Bridge: HashMap<K, V> as SerializableType
impl<K: SerializableType + Hash + Eq, V: SerializableType> SerializableType for HashMap<K, V> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as ComplexSerialize>::serialize_data(self, output)
    }

    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as ComplexSerialize>::deserialize_with_version(input, 1)
    }
}

// Bridge: HashSet<T> as SerializableType
impl<T: SerializableType + Hash + Eq> SerializableType for HashSet<T> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as ComplexSerialize>::serialize_data(self, output)
    }

    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as ComplexSerialize>::deserialize_with_version(input, 1)
    }
}

// Bridge: BTreeMap<K, V> as SerializableType
impl<K: SerializableType + Ord, V: SerializableType> SerializableType for BTreeMap<K, V> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as ComplexSerialize>::serialize_data(self, output)
    }

    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as ComplexSerialize>::deserialize_with_version(input, 1)
    }
}

// Bridge: BTreeSet<T> as SerializableType
impl<T: SerializableType + Ord> SerializableType for BTreeSet<T> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as ComplexSerialize>::serialize_data(self, output)
    }

    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as ComplexSerialize>::deserialize_with_version(input, 1)
    }
}

/// Configuration for complex type serialization
#[derive(Debug, Clone)]
pub struct ComplexTypeConfig {
    /// Include type metadata in serialization
    pub include_metadata: bool,
    /// Validate type compatibility on deserialization: the type id in the
    /// metadata must equal `T::type_id()`.
    ///
    /// Limitation: type ids do not encode element types; see
    /// [`ComplexSerialize::type_id`].
    pub validate_types: bool,
    /// Handle version mismatches gracefully: accept a version in the
    /// metadata that differs from `T::version()` and pass it to
    /// `T::deserialize_with_version`. When false, such data is rejected.
    /// The built-in impls read only version 1 either way.
    pub allow_version_skew: bool,
    /// Optimize for space efficiency
    ///
    /// Limitation: no serializer reads this flag; it has no effect on the
    /// encoding.
    pub space_optimized: bool,
}

impl ComplexTypeConfig {
    /// Create a new configuration with default settings
    pub fn new() -> Self {
        Self {
            include_metadata: true,
            validate_types: true,
            allow_version_skew: false,
            space_optimized: false,
        }
    }

    /// Configuration for maximum safety and validation
    pub fn safe() -> Self {
        Self {
            include_metadata: true,
            validate_types: true,
            allow_version_skew: false,
            space_optimized: false,
        }
    }

    /// Configuration optimized for performance
    pub fn fast() -> Self {
        Self {
            include_metadata: false,
            validate_types: false,
            allow_version_skew: true,
            space_optimized: false,
        }
    }

    /// Configuration optimized for space efficiency
    pub fn compact() -> Self {
        Self {
            include_metadata: false,
            validate_types: false,
            allow_version_skew: true,
            space_optimized: true,
        }
    }

    /// Configuration for cross-version compatibility
    pub fn compatible() -> Self {
        Self {
            include_metadata: true,
            validate_types: true,
            allow_version_skew: true,
            space_optimized: false,
        }
    }
}

impl Default for ComplexTypeConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level complex type serialization utilities
pub struct ComplexTypeSerializer {
    config: ComplexTypeConfig,
}

impl Default for ComplexTypeSerializer {
    fn default() -> Self {
        Self::new(ComplexTypeConfig::default())
    }
}

impl ComplexTypeSerializer {
    /// Create a new serializer with the given configuration
    pub fn new(config: ComplexTypeConfig) -> Self {
        Self { config }
    }

    /// Serialize a complex type to bytes
    pub fn serialize_to_bytes<T: ComplexSerialize>(&self, value: &T) -> Result<Vec<u8>> {
        let mut output = crate::io::VecDataOutput::new();

        if self.config.include_metadata {
            value.serialize_with_metadata(&mut output)?;
        } else {
            value.serialize_data(&mut output)?;
        }

        Ok(output.into_vec())
    }

    /// Deserialize a complex type from bytes
    ///
    /// With `include_metadata`, the type id and version are read here rather
    /// than through [`ComplexSerialize::deserialize_with_metadata`], so that
    /// the config applies: the type id is compared with `T::type_id()` only
    /// when `validate_types` is set, and a version other than `T::version()`
    /// is rejected unless `allow_version_skew` is set.
    pub fn deserialize_from_bytes<T: ComplexSerialize>(&self, bytes: &[u8]) -> Result<T> {
        let mut input = crate::io::SliceDataInput::new(bytes);

        if self.config.include_metadata {
            let type_id = input.read_length_prefixed_string()?;
            if self.config.validate_types && type_id != T::type_id() {
                return Err(ZiporaError::invalid_data(format!(
                    "Type mismatch: expected {}, got {}",
                    T::type_id(),
                    type_id
                )));
            }
            let version = input.read_u32()?;
            self.check_version::<T>(version)?;
            T::deserialize_with_version(&mut input, version)
        } else {
            T::deserialize_with_version(&mut input, T::version())
        }
    }

    /// Reject a version other than `T::version()` unless the config allows
    /// version skew.
    fn check_version<T: ComplexSerialize>(&self, version: u32) -> Result<()> {
        if !self.config.allow_version_skew && version != T::version() {
            return Err(ZiporaError::invalid_data(format!(
                "Version mismatch for {}: expected {}, got {}",
                T::type_id(),
                T::version(),
                version
            )));
        }
        Ok(())
    }

    /// Serialize multiple objects efficiently.
    ///
    /// # Errors
    /// Returns [`ZiporaError::InvalidData`] if `values.len() > u32::MAX as usize`
    /// or if serializing any element fails.
    pub fn serialize_batch<T: ComplexSerialize>(&self, values: &[T]) -> Result<Vec<u8>> {
        let mut output = crate::io::VecDataOutput::new();

        // Write batch header
        output.write_u32(checked_u32_len(values.len(), "batch")?)?;

        if self.config.include_metadata && !values.is_empty() {
            // Write type metadata once for the entire batch
            output.write_length_prefixed_string(T::type_id())?;
            output.write_u32(T::version())?;
        }

        // Serialize each value
        for value in values {
            value.serialize_data(&mut output)?;
        }

        Ok(output.into_vec())
    }

    /// Deserialize a batch of objects
    ///
    /// The element count is read from `bytes` and is not trusted: the
    /// metadata is checked first, and the result vector reserves at most the
    /// read-ahead budget (and at most one element per remaining byte) before
    /// it grows with the decoded values.
    ///
    /// Limitation: a type whose encoding is zero bytes long, such as `()`,
    /// consumes no input per element, so a 4-byte count makes this loop run
    /// up to `u32::MAX` times.
    pub fn deserialize_batch<T: ComplexSerialize>(&self, bytes: &[u8]) -> Result<Vec<T>> {
        let mut input = crate::io::SliceDataInput::new(bytes);

        let count = input.read_u32()? as usize;

        let version = if self.config.include_metadata && count > 0 {
            // Read type metadata once for the entire batch
            let type_id = input.read_length_prefixed_string()?;
            if self.config.validate_types && type_id != T::type_id() {
                return Err(ZiporaError::invalid_data(format!(
                    "Batch type mismatch: expected {}, got {}",
                    T::type_id(),
                    type_id
                )));
            }
            let version = input.read_u32()?;
            self.check_version::<T>(version)?;
            version
        } else {
            T::version()
        };

        let mut values = Vec::with_capacity(prealloc_capacity(
            count,
            std::mem::size_of::<T>(),
            input.remaining_bytes(),
        ));

        // Deserialize each value
        for _ in 0..count {
            values.push(T::deserialize_with_version(&mut input, version)?);
        }

        Ok(values)
    }
}

/// Recursive serialization support for nested structures
pub trait NestedSerialize {
    /// Serialize with depth tracking to prevent infinite recursion
    fn serialize_nested<O: DataOutput>(&self, output: &mut O, depth: usize) -> Result<()>;

    /// Deserialize with depth tracking
    fn deserialize_nested<I: DataInput>(input: &mut I, depth: usize) -> Result<Self>
    where
        Self: Sized;

    /// Maximum allowed nesting depth (`256`, sized so recursive deserialization
    /// frames fit a 2 MiB thread stack in both debug and release builds)
    fn max_depth() -> usize {
        DEFAULT_MAX_DEPTH
    }
}

// Default implementation using ComplexSerialize
impl<T: ComplexSerialize> NestedSerialize for T {
    fn serialize_nested<O: DataOutput>(&self, output: &mut O, depth: usize) -> Result<()> {
        crate::io::smart_ptr::with_nested_ser_depth(depth, Self::max_depth(), || {
            self.serialize_data(output)
        })
    }

    fn deserialize_nested<I: DataInput>(input: &mut I, depth: usize) -> Result<Self> {
        crate::io::smart_ptr::with_nested_de_depth(depth, Self::max_depth(), || {
            Self::deserialize_with_version(input, Self::version())
        })
    }
}

/// Macro for implementing [`ComplexSerialize`] for custom structs.
///
/// # Limitations
///
/// Each field is serialized/deserialized through [`SerializableType`]. When
/// `serialize_data` or `deserialize_with_version` is invoked at the top level
/// outside an enclosing smart-pointer, collection, or [`NestedSerialize`] scope,
/// sibling smart-pointer fields do not share a single
/// [`crate::io::SerializationContext`] / [`crate::io::DeserializationContext`];
/// see [`SerializableType#limitations`] for details.
#[macro_export]
macro_rules! impl_complex_serialize {
    ($struct_name:ident { $($field:ident : $field_type:ty),* $(,)? }) => {
        impl ComplexSerialize for $struct_name {
            fn type_id() -> &'static str {
                stringify!($struct_name)
            }

            fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
                $(
                    self.$field.serialize(output)?;
                )*
                Ok(())
            }

            fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self> {
                if version != <Self as ComplexSerialize>::version() {
                    return Err($crate::error::ZiporaError::invalid_data(format!(
                        "{} has no serialization version {}; only version {} exists",
                        stringify!($struct_name),
                        version,
                        <Self as ComplexSerialize>::version()
                    )));
                }
                Ok(Self {
                    $(
                        $field: <$field_type>::deserialize(input)?,
                    )*
                })
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{SliceDataInput, VecDataOutput};

    #[derive(Debug, PartialEq)]
    struct TestStruct {
        id: u32,
        name: String,
        active: bool,
    }

    impl_complex_serialize!(TestStruct {
        id: u32,
        name: String,
        active: bool,
    });

    #[test]
    fn test_tuple_serialization() {
        let tuple = (42u32, "hello".to_string(), true);
        let mut output = VecDataOutput::new();

        tuple.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <(u32, String, bool)>::deserialize_with_version(&mut input, 1).unwrap();

        assert_eq!(deserialized, tuple);
    }

    #[test]
    fn test_empty_tuple_serialization() {
        let tuple = ();
        let mut output = VecDataOutput::new();

        tuple.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        // Deserializing the unit type must succeed; the value itself is always ().
        <()>::deserialize_with_version(&mut input, 1).unwrap();
    }

    #[test]
    fn test_array_serialization() {
        let array = [1u32, 2, 3, 4, 5];
        let mut output = VecDataOutput::new();

        array.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <[u32; 5]>::deserialize_with_version(&mut input, 1).unwrap();

        assert_eq!(deserialized, array);
    }

    #[test]
    fn test_option_serialization() {
        // Test Some case
        let some_value = Some(42u32);
        let mut output = VecDataOutput::new();

        some_value.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = Option::<u32>::deserialize_with_version(&mut input, 1).unwrap();

        assert_eq!(deserialized, some_value);

        // Test None case
        let none_value: Option<u32> = None;
        let mut output = VecDataOutput::new();

        none_value.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = Option::<u32>::deserialize_with_version(&mut input, 1).unwrap();

        assert_eq!(deserialized, none_value);
    }

    #[test]
    fn test_result_serialization() {
        // Test Ok case
        let ok_value: std::result::Result<u32, String> = Ok(42);
        let mut output = VecDataOutput::new();

        ok_value.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized =
            std::result::Result::<u32, String>::deserialize_with_version(&mut input, 1).unwrap();

        assert_eq!(deserialized, ok_value);

        // Test Err case
        let err_value: std::result::Result<u32, String> = Err("error".to_string());
        let mut output = VecDataOutput::new();

        err_value.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized =
            std::result::Result::<u32, String>::deserialize_with_version(&mut input, 1).unwrap();

        assert_eq!(deserialized, err_value);
    }

    #[test]
    fn test_hashmap_serialization() {
        let mut map = HashMap::new();
        map.insert("key1".to_string(), 42u32);
        map.insert("key2".to_string(), 84u32);

        let mut output = VecDataOutput::new();
        map.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = HashMap::<String, u32>::deserialize_with_version(&mut input, 1).unwrap();

        assert_eq!(deserialized, map);
    }

    #[test]
    fn test_hashset_serialization() {
        let mut set = HashSet::new();
        set.insert("item1".to_string());
        set.insert("item2".to_string());

        let mut output = VecDataOutput::new();
        set.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = HashSet::<String>::deserialize_with_version(&mut input, 1).unwrap();

        assert_eq!(deserialized, set);
    }

    #[test]
    fn test_metadata_serialization() {
        let tuple = (42u32, "hello".to_string());
        let mut output = VecDataOutput::new();

        tuple.serialize_with_metadata(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <(u32, String)>::deserialize_with_metadata(&mut input).unwrap();

        assert_eq!(deserialized, tuple);
    }

    #[test]
    fn test_complex_type_serializer() {
        let serializer = ComplexTypeSerializer::default();
        let tuple = (42u32, "hello".to_string(), vec![1, 2, 3]);

        let bytes = serializer.serialize_to_bytes(&tuple).unwrap();
        let deserialized: (u32, String, Vec<i32>) =
            serializer.deserialize_from_bytes(&bytes).unwrap();

        assert_eq!(deserialized, tuple);
    }

    #[test]
    fn test_batch_serialization() {
        let serializer = ComplexTypeSerializer::default();
        let tuples = vec![
            (1u32, "first".to_string()),
            (2u32, "second".to_string()),
            (3u32, "third".to_string()),
        ];

        let bytes = serializer.serialize_batch(&tuples).unwrap();
        let deserialized: Vec<(u32, String)> = serializer.deserialize_batch(&bytes).unwrap();

        assert_eq!(deserialized, tuples);
    }

    #[test]
    fn test_custom_struct_serialization() {
        let test_struct = TestStruct {
            id: 42,
            name: "test".to_string(),
            active: true,
        };

        let mut output = VecDataOutput::new();
        test_struct.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized = TestStruct::deserialize_with_version(&mut input, 1).unwrap();

        assert_eq!(deserialized, test_struct);
    }

    #[test]
    fn test_nested_complex_types() {
        let nested = (
            vec![1u32, 2, 3],
            Some("nested".to_string()),
            HashMap::from([("key".to_string(), 42u32)]),
        );

        let mut output = VecDataOutput::new();
        nested.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized =
            <(Vec<u32>, Option<String>, HashMap<String, u32>)>::deserialize_with_version(
                &mut input, 1,
            )
            .unwrap();

        assert_eq!(deserialized, nested);
    }

    #[test]
    fn test_serializer_configs() {
        let safe_config = ComplexTypeConfig::safe();
        assert!(safe_config.include_metadata);
        assert!(safe_config.validate_types);

        let fast_config = ComplexTypeConfig::fast();
        assert!(!fast_config.include_metadata);
        assert!(!fast_config.validate_types);

        let compact_config = ComplexTypeConfig::compact();
        assert!(compact_config.space_optimized);
    }

    #[test]
    fn test_type_validation() {
        let tuple = (42u32, "hello".to_string());
        let mut output = VecDataOutput::new();

        // Serialize as one type
        tuple.serialize_with_metadata(&mut output).unwrap();
        let bytes = output.into_vec();

        // Try to deserialize as different type
        let mut input = SliceDataInput::new(&bytes);
        let result = <(u64, String)>::deserialize_with_metadata(&mut input);

        // Both types report "tuple2", so the type check passes. The error
        // comes from reading the String's length out of the bytes after the
        // u64 and running out of input. test_c9_ct3_tuple_arity_is_checked
        // covers type ids that differ.
        assert!(result.is_err());
    }

    #[test]
    fn test_array_length_validation() {
        let array = [1u32, 2, 3];
        let mut output = VecDataOutput::new();

        array.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();

        // Try to deserialize as different length array
        let mut input = SliceDataInput::new(&bytes);
        let result = <[u32; 5]>::deserialize_with_version(&mut input, 1);

        assert!(result.is_err());
    }

    thread_local! {
        static DROPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Element that counts its drops; it decodes from one non-zero byte and
    /// rejects a zero byte.
    struct DropCounter(u8);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            DROPS.with(|drops| drops.set(drops.get() + 1));
        }
    }

    impl SerializableType for DropCounter {
        fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
            output.write_u8(self.0)
        }

        fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
            match input.read_u8()? {
                0 => Err(ZiporaError::invalid_data("DropCounter: zero byte")),
                byte => Ok(DropCounter(byte)),
            }
        }
    }

    #[test]
    fn test_c9_ct4_array_error_drops_decoded_elements() {
        DROPS.with(|drops| drops.set(0));
        // Array length 3, one valid element, then a rejected one.
        let bytes = [3, 0, 0, 0, 7, 0];
        let result =
            <[DropCounter; 3]>::deserialize_with_version(&mut SliceDataInput::new(&bytes), 1);
        assert!(result.is_err());
        assert_eq!(DROPS.with(|drops| drops.get()), 1);
    }

    #[test]
    fn test_c9_ct3_tuple_arity_is_checked() {
        let mut output = VecDataOutput::new();
        (1u32, 2u32).serialize_with_metadata(&mut output).unwrap();
        let bytes = output.into_vec();
        let result = <(u64,)>::deserialize_with_metadata(&mut SliceDataInput::new(&bytes));
        assert!(
            result.is_err(),
            "a pair must not decode as a 1-tuple: {result:?}"
        );
    }

    /// Metadata (type id, version) followed by the data of `(1u32, 2u32)`.
    fn pair_with_header(type_id: &str, version: u32) -> Vec<u8> {
        let mut output = VecDataOutput::new();
        output.write_length_prefixed_string(type_id).unwrap();
        output.write_u32(version).unwrap();
        (1u32, 2u32).serialize_data(&mut output).unwrap();
        output.into_vec()
    }

    /// A type with two encodings: version 1 stores a u8, version 2 a u32.
    #[derive(Debug, PartialEq)]
    struct Versioned(u32);

    impl ComplexSerialize for Versioned {
        fn type_id() -> &'static str {
            "versioned"
        }

        fn version() -> u32 {
            2
        }

        fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
            output.write_u32(self.0)
        }

        fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self> {
            match version {
                1 => Ok(Versioned(u32::from(input.read_u8()?))),
                2 => Ok(Versioned(input.read_u32()?)),
                _ => Err(ZiporaError::invalid_data("Versioned: unknown version")),
            }
        }
    }

    /// Version-1 `Versioned` data (the value 5 as one byte) with metadata.
    fn versioned_v1_with_header() -> Vec<u8> {
        let mut output = VecDataOutput::new();
        output.write_length_prefixed_string("versioned").unwrap();
        output.write_u32(1).unwrap();
        output.write_u8(5).unwrap();
        output.into_vec()
    }

    #[test]
    fn test_c9_ct3_unknown_version_is_rejected() {
        // The built-in tuple impl has only version 1, so version 2 is
        // rejected even when the serializer allows version skew.
        let bytes = pair_with_header(<(u32, u32)>::type_id(), 2);
        let result = <(u32, u32)>::deserialize_with_metadata(&mut SliceDataInput::new(&bytes));
        assert!(result.is_err(), "version 2 must be rejected: {result:?}");
        let result = ComplexTypeSerializer::default().deserialize_from_bytes::<(u32, u32)>(&bytes);
        assert!(result.is_err(), "version 2 must be rejected: {result:?}");
        let result = ComplexTypeSerializer::new(ComplexTypeConfig::compatible())
            .deserialize_from_bytes::<(u32, u32)>(&bytes);
        assert!(result.is_err(), "version 2 must be rejected: {result:?}");
    }

    #[test]
    fn test_c9_ct3_impl_macro_rejects_unknown_version() {
        let test_struct = TestStruct {
            id: 7,
            name: "x".to_string(),
            active: true,
        };
        let mut output = VecDataOutput::new();
        test_struct.serialize_data(&mut output).unwrap();
        let data = output.into_vec();
        let result = TestStruct::deserialize_with_version(&mut SliceDataInput::new(&data), 2);
        assert!(result.is_err(), "version 2 must be rejected: {result:?}");
        let result = TestStruct::deserialize_with_version(&mut SliceDataInput::new(&data), 1);
        assert_eq!(result.unwrap(), test_struct);
    }

    #[test]
    fn test_c9_ct3_validate_types_false_accepts_other_type_id() {
        let legacy = pair_with_header("legacy", 1);
        let lenient = ComplexTypeSerializer::new(ComplexTypeConfig {
            validate_types: false,
            ..ComplexTypeConfig::default()
        });
        assert_eq!(
            lenient
                .deserialize_from_bytes::<(u32, u32)>(&legacy)
                .unwrap(),
            (1, 2)
        );
        let result = ComplexTypeSerializer::default().deserialize_from_bytes::<(u32, u32)>(&legacy);
        assert!(result.is_err(), "the type id must be checked: {result:?}");
    }

    #[test]
    fn test_c9_ct3_version_skew_requires_allow_version_skew() {
        let v1 = versioned_v1_with_header();
        let result = ComplexTypeSerializer::default().deserialize_from_bytes::<Versioned>(&v1);
        assert!(
            result.is_err(),
            "version 1 needs allow_version_skew: {result:?}"
        );
        let compatible = ComplexTypeSerializer::new(ComplexTypeConfig::compatible());
        assert_eq!(
            compatible.deserialize_from_bytes::<Versioned>(&v1).unwrap(),
            Versioned(5)
        );
    }

    #[test]
    fn test_c9_ct3_batch_version_skew_requires_allow_version_skew() {
        // Batch: count 1, metadata ("versioned", version 1), one byte of data.
        let mut output = VecDataOutput::new();
        output.write_u32(1).unwrap();
        output.write_bytes(&versioned_v1_with_header()).unwrap();
        let batch = output.into_vec();
        let result = ComplexTypeSerializer::default().deserialize_batch::<Versioned>(&batch);
        assert!(
            result.is_err(),
            "version 1 needs allow_version_skew: {result:?}"
        );
        let compatible = ComplexTypeSerializer::new(ComplexTypeConfig::compatible());
        assert_eq!(
            compatible.deserialize_batch::<Versioned>(&batch).unwrap(),
            vec![Versioned(5)]
        );
    }

    #[test]
    fn test_c9_ct5_nested_serialize_enforces_max_depth() {
        #[derive(Debug, PartialEq)]
        struct TreeNode {
            children: Vec<TreeNode>,
        }

        impl SerializableType for TreeNode {
            fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
                self.children.serialize(output)
            }

            fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
                Ok(Self {
                    children: Vec::<TreeNode>::deserialize(input)?,
                })
            }
        }

        impl_complex_serialize!(TreeNode {
            children: Vec<TreeNode>,
        });

        // Depth-1100 input: 1100 single-child nodes (`1u32` LE) followed by a leaf (`0u32` LE).
        let mut bytes = Vec::with_capacity(1101 * 4);
        for _ in 0..1100 {
            bytes.extend_from_slice(&1u32.to_le_bytes());
        }
        bytes.extend_from_slice(&0u32.to_le_bytes());

        let mut input = SliceDataInput::new(&bytes);
        let res = TreeNode::deserialize_nested(&mut input, 0);
        // Prevent recursive drop overflow if `deserialize_nested` erroneously succeeded on unpatched code.
        if let Ok(mut root) = res {
            while let Some(child) = root.children.pop() {
                root = child;
            }
            panic!("depth-1100 input must be rejected by NestedSerialize::deserialize_nested");
        }

        // Also verify `serialize_nested` rejects when starting depth + child depth exceeds `max_depth()`.
        let one_child = TreeNode {
            children: vec![TreeNode {
                children: Vec::new(),
            }],
        };
        let mut out = VecDataOutput::new();
        assert!(
            one_child
                .serialize_nested(&mut out, TreeNode::max_depth())
                .is_err(),
            "child at depth max_depth() + 1 must be rejected by serialize_nested"
        );
    }

    #[test]
    fn test_s22_r1_nested_containers_rejected_on_2mib_stack() {
        #[derive(Debug, PartialEq)]
        struct BTreeNode {
            children: BTreeMap<u32, BTreeNode>,
        }
        impl SerializableType for BTreeNode {
            fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
                self.children.serialize(output)
            }
            fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
                Ok(Self {
                    children: BTreeMap::<u32, BTreeNode>::deserialize(input)?,
                })
            }
        }
        impl_complex_serialize!(BTreeNode {
            children: BTreeMap<u32, BTreeNode>,
        });

        let depth = if cfg!(miri) { 2_000usize } else { 100_000usize };
        let mut btree_wire = Vec::with_capacity(depth * 8 + 4);
        for _ in 0..depth {
            btree_wire.extend_from_slice(&1u32.to_le_bytes()); // len = 1
            btree_wire.extend_from_slice(&0u32.to_le_bytes()); // key = 0
        }
        btree_wire.extend_from_slice(&0u32.to_le_bytes()); // leaf len = 0

        std::thread::Builder::new()
            .name("s22-r1-btree-2mib".into())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                let mut input = SliceDataInput::new(&btree_wire);
                assert!(
                    BTreeNode::deserialize_nested(&mut input, 0).is_err(),
                    "deep BTreeNode must return Err within 2 MiB stack"
                );
                let mut input2 = SliceDataInput::new(&btree_wire);
                assert!(
                    <BTreeMap<u32, BTreeNode> as ComplexSerialize>::deserialize_with_version(
                        &mut input2,
                        1
                    )
                    .is_err(),
                    "deep BTreeMap must return Err within 2 MiB stack"
                );
            })
            .expect("spawn 2 MiB thread")
            .join()
            .expect("2 MiB thread must not overflow stack or panic");
    }

    #[test]
    fn test_s22_r8_checked_u32_len_rejects_above_u32_max() {
        assert_eq!(checked_u32_len(0, "Vec").unwrap(), 0);
        assert_eq!(
            checked_u32_len(u32::MAX as usize, "Vec").unwrap(),
            u32::MAX
        );
        #[cfg(target_pointer_width = "64")]
        {
            assert!(checked_u32_len((u32::MAX as usize) + 1, "Vec").is_err());
        }
    }
}
