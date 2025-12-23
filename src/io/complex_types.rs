//! Complex type serialization for tuples, arrays, and nested structures
//!
//! This module provides automatic serialization and deserialization for complex
//! Rust types including tuples, arrays, Option, Result, and custom structures.
//! Uses recursive template patterns and procedural macros for type-safe serialization.

use crate::error::{Result, ZiporaError};
use crate::io::{DataInput, DataOutput};
use crate::io::smart_ptr::SerializableType;
use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet};
use std::hash::Hash;

/// Trait for complex type serialization with metadata
pub trait ComplexSerialize: Sized {
    /// Get the type identifier for versioning
    fn type_id() -> &'static str;
    
    /// Get the version of this type's serialization format
    fn version() -> u32 { 1 }
    
    /// Serialize with type metadata
    fn serialize_with_metadata<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        // Write type metadata
        output.write_length_prefixed_string(Self::type_id())?;
        output.write_u32(Self::version())?;
        
        // Serialize the actual data
        self.serialize_data(output)
    }
    
    /// Deserialize with type metadata validation
    fn deserialize_with_metadata<I: DataInput>(input: &mut I) -> Result<Self> {
        // Read and validate type metadata
        let type_id = input.read_length_prefixed_string()?;
        if type_id != Self::type_id() {
            return Err(ZiporaError::invalid_data(
                format!("Type mismatch: expected {}, got {}", Self::type_id(), type_id)
            ));
        }
        
        let version = input.read_u32()?;
        Self::deserialize_with_version(input, version)
    }
    
    /// Serialize just the data (without metadata)
    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()>;
    
    /// Deserialize with specific version handling
    fn deserialize_with_version<I: DataInput>(input: &mut I, version: u32) -> Result<Self>;
}

// Tuple implementations using recursive patterns
macro_rules! impl_tuple_serialize {
    () => {
        impl ComplexSerialize for () {
            fn type_id() -> &'static str { "tuple0" }
            
            fn serialize_data<O: DataOutput>(&self, _output: &mut O) -> Result<()> {
                Ok(()) // Empty tuple, nothing to serialize
            }
            
            fn deserialize_with_version<I: DataInput>(_input: &mut I, _version: u32) -> Result<Self> {
                Ok(()) // Empty tuple, nothing to deserialize
            }
        }
    };
    
    ($($T:ident),*) => {
        impl<$($T: SerializableType),*> ComplexSerialize for ($($T,)*) {
            fn type_id() -> &'static str { 
                "tuple" // Simplified for now
            }
            
            fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
                #[allow(non_snake_case)]
                let ($($T,)*) = self;
                $(
                    $T.serialize(output)?;
                )*
                Ok(())
            }
            
            fn deserialize_with_version<I: DataInput>(input: &mut I, _version: u32) -> Result<Self> {
                Ok(($(
                    $T::deserialize(input)?,
                )*))
            }
        }
    };
}

// Implement for tuples of different arities
impl_tuple_serialize!();
impl_tuple_serialize!(T0);
impl_tuple_serialize!(T0, T1);
impl_tuple_serialize!(T0, T1, T2);
impl_tuple_serialize!(T0, T1, T2, T3);
impl_tuple_serialize!(T0, T1, T2, T3, T4);
impl_tuple_serialize!(T0, T1, T2, T3, T4, T5);
impl_tuple_serialize!(T0, T1, T2, T3, T4, T5, T6);
impl_tuple_serialize!(T0, T1, T2, T3, T4, T5, T6, T7);
impl_tuple_serialize!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
impl_tuple_serialize!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_tuple_serialize!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_tuple_serialize!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);

// Array implementations with const generics
impl<T: SerializableType, const N: usize> ComplexSerialize for [T; N] {
    fn type_id() -> &'static str { "array" }
    
    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        // Write array length for validation
        output.write_u32(N as u32)?;
        
        for item in self.iter() {
            item.serialize(output)?;
        }
        Ok(())
    }
    
    fn deserialize_with_version<I: DataInput>(input: &mut I, _version: u32) -> Result<Self> {
        let length = input.read_u32()? as usize;
        if length != N {
            return Err(ZiporaError::invalid_data(
                format!("Array length mismatch: expected {}, got {}", N, length)
            ));
        }
        
        // Use MaybeUninit for safe initialization
        use std::mem::MaybeUninit;
        // SAFETY: This creates an array of MaybeUninit<T> values.
        // MaybeUninit<T> does not require initialization, so an array of
        // uninitialized MaybeUninit values is valid. All elements are
        // initialized in the loop below before the final transmute.
        let mut array: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        
        for i in 0..N {
            array[i] = MaybeUninit::new(T::deserialize(input)?);
        }
        
        // Safety: All elements have been initialized
        Ok(unsafe { std::mem::transmute_copy::<_, [T; N]>(&array) })
    }
}

// Option implementation
impl<T: SerializableType> ComplexSerialize for Option<T> {
    fn type_id() -> &'static str { "option" }
    
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
    
    fn deserialize_with_version<I: DataInput>(input: &mut I, _version: u32) -> Result<Self> {
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
    fn type_id() -> &'static str { "result" }
    
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
    
    fn deserialize_with_version<I: DataInput>(input: &mut I, _version: u32) -> Result<Self> {
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
    fn type_id() -> &'static str { "hashmap" }
    
    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(self.len() as u32)?;
        
        for (key, value) in self {
            key.serialize(output)?;
            value.serialize(output)?;
        }
        Ok(())
    }
    
    fn deserialize_with_version<I: DataInput>(input: &mut I, _version: u32) -> Result<Self> {
        let len = input.read_u32()? as usize;
        let mut map = HashMap::with_capacity(len);
        
        for _ in 0..len {
            let key = K::deserialize(input)?;
            let value = V::deserialize(input)?;
            map.insert(key, value);
        }
        
        Ok(map)
    }
}

// HashSet implementation
impl<T: SerializableType + Hash + Eq> ComplexSerialize for HashSet<T> {
    fn type_id() -> &'static str { "hashset" }
    
    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(self.len() as u32)?;
        
        for item in self {
            item.serialize(output)?;
        }
        Ok(())
    }
    
    fn deserialize_with_version<I: DataInput>(input: &mut I, _version: u32) -> Result<Self> {
        let len = input.read_u32()? as usize;
        let mut set = HashSet::with_capacity(len);
        
        for _ in 0..len {
            set.insert(T::deserialize(input)?);
        }
        
        Ok(set)
    }
}

// BTreeMap implementation
impl<K: SerializableType + Ord, V: SerializableType> ComplexSerialize for BTreeMap<K, V> {
    fn type_id() -> &'static str { "btreemap" }
    
    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(self.len() as u32)?;
        
        for (key, value) in self {
            key.serialize(output)?;
            value.serialize(output)?;
        }
        Ok(())
    }
    
    fn deserialize_with_version<I: DataInput>(input: &mut I, _version: u32) -> Result<Self> {
        let len = input.read_u32()? as usize;
        let mut map = BTreeMap::new();
        
        for _ in 0..len {
            let key = K::deserialize(input)?;
            let value = V::deserialize(input)?;
            map.insert(key, value);
        }
        
        Ok(map)
    }
}

// BTreeSet implementation
impl<T: SerializableType + Ord> ComplexSerialize for BTreeSet<T> {
    fn type_id() -> &'static str { "btreeset" }
    
    fn serialize_data<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(self.len() as u32)?;
        
        for item in self {
            item.serialize(output)?;
        }
        Ok(())
    }
    
    fn deserialize_with_version<I: DataInput>(input: &mut I, _version: u32) -> Result<Self> {
        let len = input.read_u32()? as usize;
        let mut set = BTreeSet::new();
        
        for _ in 0..len {
            set.insert(T::deserialize(input)?);
        }
        
        Ok(set)
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
    /// Validate type compatibility on deserialization
    pub validate_types: bool,
    /// Handle version mismatches gracefully
    pub allow_version_skew: bool,
    /// Optimize for space efficiency
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

impl ComplexTypeSerializer {
    /// Create a new serializer with the given configuration
    pub fn new(config: ComplexTypeConfig) -> Self {
        Self { config }
    }
    
    /// Create a serializer with default configuration
    pub fn default() -> Self {
        Self::new(ComplexTypeConfig::default())
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
    pub fn deserialize_from_bytes<T: ComplexSerialize>(&self, bytes: &[u8]) -> Result<T> {
        let mut input = crate::io::SliceDataInput::new(bytes);
        
        if self.config.include_metadata {
            T::deserialize_with_metadata(&mut input)
        } else {
            T::deserialize_with_version(&mut input, T::version())
        }
    }
    
    /// Serialize multiple objects efficiently
    pub fn serialize_batch<T: ComplexSerialize>(&self, values: &[T]) -> Result<Vec<u8>> {
        let mut output = crate::io::VecDataOutput::new();
        
        // Write batch header
        output.write_u32(values.len() as u32)?;
        
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
    pub fn deserialize_batch<T: ComplexSerialize>(&self, bytes: &[u8]) -> Result<Vec<T>> {
        let mut input = crate::io::SliceDataInput::new(bytes);
        
        let count = input.read_u32()? as usize;
        let mut values = Vec::with_capacity(count);
        
        let version = if self.config.include_metadata && count > 0 {
            // Read type metadata once for the entire batch
            let type_id = input.read_length_prefixed_string()?;
            if self.config.validate_types && type_id != T::type_id() {
                return Err(ZiporaError::invalid_data(
                    format!("Batch type mismatch: expected {}, got {}", T::type_id(), type_id)
                ));
            }
            input.read_u32()?
        } else {
            T::version()
        };
        
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
    
    /// Maximum allowed nesting depth
    fn max_depth() -> usize { 1000 }
}

// Default implementation using ComplexSerialize
impl<T: ComplexSerialize> NestedSerialize for T {
    fn serialize_nested<O: DataOutput>(&self, output: &mut O, depth: usize) -> Result<()> {
        if depth > Self::max_depth() {
            return Err(ZiporaError::invalid_data("Maximum nesting depth exceeded"));
        }
        
        self.serialize_data(output)
    }
    
    fn deserialize_nested<I: DataInput>(input: &mut I, depth: usize) -> Result<Self> {
        if depth > Self::max_depth() {
            return Err(ZiporaError::invalid_data("Maximum nesting depth exceeded"));
        }
        
        Self::deserialize_with_version(input, Self::version())
    }
}

/// Macro for implementing ComplexSerialize for custom structs
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
            
            fn deserialize_with_version<I: DataInput>(input: &mut I, _version: u32) -> Result<Self> {
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
        let deserialized = <()>::deserialize_with_version(&mut input, 1).unwrap();
        
        assert_eq!(deserialized, tuple);
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
        let deserialized = std::result::Result::<u32, String>::deserialize_with_version(&mut input, 1).unwrap();
        
        assert_eq!(deserialized, ok_value);
        
        // Test Err case
        let err_value: std::result::Result<u32, String> = Err("error".to_string());
        let mut output = VecDataOutput::new();
        
        err_value.serialize_data(&mut output).unwrap();
        let bytes = output.into_vec();
        
        let mut input = SliceDataInput::new(&bytes);
        let deserialized = std::result::Result::<u32, String>::deserialize_with_version(&mut input, 1).unwrap();
        
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
        let deserialized: (u32, String, Vec<i32>) = serializer.deserialize_from_bytes(&bytes).unwrap();
        
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
        let deserialized = <(Vec<u32>, Option<String>, HashMap<String, u32>)>::deserialize_with_version(&mut input, 1).unwrap();
        
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
}