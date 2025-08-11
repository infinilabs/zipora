//! Smart pointer serialization for reference-counted objects
//!
//! This module provides automatic serialization and deserialization support for
//! Rust's smart pointer types including Box<T>, Arc<T>, Rc<T>, and custom smart pointers.
//! Supports null handling, reference counting, and circular reference detection.

use crate::error::{Result, ZiporaError};
use crate::io::{DataInput, DataOutput};
use std::collections::HashMap;
// Hash import removed as it's not used in this module
use std::rc::{Rc, Weak as RcWeak};
use std::sync::{Arc, Weak as ArcWeak};

/// Serialization context for tracking shared objects and preventing cycles
#[derive(Debug)]
pub struct SerializationContext {
    /// Object ID counter for reference tracking
    next_id: u32,
    /// Map from object pointer to assigned ID
    object_ids: HashMap<usize, u32>,
    /// Enable circular reference detection
    detect_cycles: bool,
}

impl SerializationContext {
    /// Create a new serialization context
    pub fn new() -> Self {
        Self {
            next_id: 1, // Start from 1, reserve 0 for null
            object_ids: HashMap::new(),
            detect_cycles: true,
        }
    }
    
    /// Create context with cycle detection disabled (for performance)
    pub fn without_cycle_detection() -> Self {
        Self {
            next_id: 1,
            object_ids: HashMap::new(),
            detect_cycles: false,
        }
    }
    
    /// Get or assign an ID for an object
    fn get_or_assign_id<T>(&mut self, ptr: &T) -> u32 {
        let addr = ptr as *const T as usize;
        
        if let Some(&id) = self.object_ids.get(&addr) {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            if self.detect_cycles {
                self.object_ids.insert(addr, id);
            }
            id
        }
    }
    
    /// Check if an object has been seen before (for cycle detection)
    fn has_seen<T>(&self, ptr: &T) -> bool {
        if !self.detect_cycles {
            return false;
        }
        
        let addr = ptr as *const T as usize;
        self.object_ids.contains_key(&addr)
    }
    
    /// Clear the context for reuse
    pub fn clear(&mut self) {
        self.next_id = 1;
        self.object_ids.clear();
    }
}

impl Default for SerializationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Deserialization context for reconstructing shared objects
#[derive(Debug)]
pub struct DeserializationContext<T> {
    /// Map from object ID to reconstructed object
    objects: HashMap<u32, T>,
}

impl<T> DeserializationContext<T> {
    /// Create a new deserialization context
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
        }
    }
    
    /// Store an object with its ID
    pub fn store_object(&mut self, id: u32, object: T) {
        self.objects.insert(id, object);
    }
    
    /// Get an object by its ID
    pub fn get_object(&self, id: u32) -> Option<&T> {
        self.objects.get(&id)
    }
    
    /// Clear the context for reuse
    pub fn clear(&mut self) {
        self.objects.clear();
    }
}

impl<T> Default for DeserializationContext<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can be serialized as smart pointers
pub trait SmartPtrSerialize<T> {
    /// Serialize the smart pointer with context
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()>;
    
    /// Deserialize the smart pointer with context
    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        _context: &mut DeserializationContext<Self>,
    ) -> Result<Self>
    where
        Self: Sized;
    
    /// Serialize without context (simple case)
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        let mut context = SerializationContext::new();
        self.serialize_with_context(output, &mut context)
    }
    
    /// Deserialize without context (simple case)
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self>
    where
        Self: Sized,
    {
        let mut context = DeserializationContext::new();
        Self::deserialize_with_context(input, &mut context)
    }
}

/// Marker trait for types that are serializable
pub trait SerializableType {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()>;
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self>
    where
        Self: Sized;
}

// Implementation for Box<T>
impl<T: SerializableType> SmartPtrSerialize<T> for Box<T> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        _context: &mut SerializationContext,
    ) -> Result<()> {
        // Box is always unique, no need for reference tracking
        output.write_u8(1)?; // Non-null marker
        self.as_ref().serialize(output)
    }
    
    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        _context: &mut DeserializationContext<Self>,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Err(ZiporaError::invalid_data("Box cannot be null")),
            1 => {
                let value = T::deserialize(input)?;
                Ok(Box::new(value))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Box marker")),
        }
    }
}

// Implementation for Option<Box<T>>
impl<T: SerializableType> SmartPtrSerialize<T> for Option<Box<T>> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        match self {
            Some(boxed) => {
                output.write_u8(1)?; // Non-null marker
                boxed.serialize_with_context(output, context)
            }
            None => {
                output.write_u8(0)?; // Null marker
                Ok(())
            }
        }
    }
    
    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        _context: &mut DeserializationContext<Self>,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(None),
            1 => {
                let boxed = Box::<T>::deserialize_with_context(input, 
                    &mut DeserializationContext::new())?;
                Ok(Some(boxed))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Option<Box> marker")),
        }
    }
}

// Implementation for Rc<T>
impl<T: SerializableType> SmartPtrSerialize<T> for Rc<T> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        let obj_ref = self.as_ref();
        
        if context.has_seen(obj_ref) {
            // Object already serialized, write reference
            let id = context.get_or_assign_id(obj_ref);
            output.write_u8(2)?; // Reference marker
            output.write_u32(id)?;
        } else {
            // First time seeing this object, serialize it
            let id = context.get_or_assign_id(obj_ref);
            output.write_u8(1)?; // New object marker
            output.write_u32(id)?;
            obj_ref.serialize(output)?;
        }
        
        Ok(())
    }
    
    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext<Self>,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Err(ZiporaError::invalid_data("Rc cannot be null")),
            1 => {
                // New object
                let id = input.read_u32()?;
                let value = T::deserialize(input)?;
                let rc = Rc::new(value);
                context.store_object(id, rc.clone());
                Ok(rc)
            }
            2 => {
                // Reference to existing object
                let id = input.read_u32()?;
                context.get_object(id)
                    .cloned()
                    .ok_or_else(|| ZiporaError::invalid_data("Referenced object not found"))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Rc marker")),
        }
    }
}

// Implementation for Arc<T>
impl<T: SerializableType + Send + Sync> SmartPtrSerialize<T> for Arc<T> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        let obj_ref = self.as_ref();
        
        if context.has_seen(obj_ref) {
            // Object already serialized, write reference
            let id = context.get_or_assign_id(obj_ref);
            output.write_u8(2)?; // Reference marker
            output.write_u32(id)?;
        } else {
            // First time seeing this object, serialize it
            let id = context.get_or_assign_id(obj_ref);
            output.write_u8(1)?; // New object marker
            output.write_u32(id)?;
            obj_ref.serialize(output)?;
        }
        
        Ok(())
    }
    
    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext<Self>,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Err(ZiporaError::invalid_data("Arc cannot be null")),
            1 => {
                // New object
                let id = input.read_u32()?;
                let value = T::deserialize(input)?;
                let arc = Arc::new(value);
                context.store_object(id, arc.clone());
                Ok(arc)
            }
            2 => {
                // Reference to existing object
                let id = input.read_u32()?;
                context.get_object(id)
                    .cloned()
                    .ok_or_else(|| ZiporaError::invalid_data("Referenced object not found"))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Arc marker")),
        }
    }
}

// Weak pointer support for Rc<T>
impl<T: SerializableType> SmartPtrSerialize<T> for RcWeak<T> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        match self.upgrade() {
            Some(rc) => {
                output.write_u8(1)?; // Valid weak reference
                rc.serialize_with_context(output, context)
            }
            None => {
                output.write_u8(0)?; // Dangling weak reference
                Ok(())
            }
        }
    }
    
    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        _context: &mut DeserializationContext<Self>,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => {
                // Dangling weak reference - create a weak that will never upgrade
                let dummy = Rc::new(T::deserialize(input)?);
                Ok(Rc::downgrade(&dummy))
            }
            1 => {
                // Valid weak reference
                let mut rc_context = DeserializationContext::new();
                let rc = Rc::<T>::deserialize_with_context(input, &mut rc_context)?;
                Ok(Rc::downgrade(&rc))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Weak<Rc> marker")),
        }
    }
}

// Weak pointer support for Arc<T>
impl<T: SerializableType + Send + Sync> SmartPtrSerialize<T> for ArcWeak<T> {
    fn serialize_with_context<O: DataOutput>(
        &self,
        output: &mut O,
        context: &mut SerializationContext,
    ) -> Result<()> {
        match self.upgrade() {
            Some(arc) => {
                output.write_u8(1)?; // Valid weak reference
                arc.serialize_with_context(output, context)
            }
            None => {
                output.write_u8(0)?; // Dangling weak reference
                Ok(())
            }
        }
    }
    
    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        _context: &mut DeserializationContext<Self>,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => {
                // Dangling weak reference - create a weak that will never upgrade
                let dummy = Arc::new(T::deserialize(input)?);
                Ok(Arc::downgrade(&dummy))
            }
            1 => {
                // Valid weak reference
                let mut arc_context = DeserializationContext::new();
                let arc = Arc::<T>::deserialize_with_context(input, &mut arc_context)?;
                Ok(Arc::downgrade(&arc))
            }
            _ => Err(ZiporaError::invalid_data("Invalid Weak<Arc> marker")),
        }
    }
}

/// Configuration for smart pointer serialization
#[derive(Debug, Clone)]
pub struct SmartPtrConfig {
    /// Enable cycle detection and shared object optimization
    pub cycle_detection: bool,
    /// Maximum recursion depth to prevent stack overflow
    pub max_depth: usize,
    /// Compress object IDs for smaller serialized size
    pub compress_ids: bool,
}

impl SmartPtrConfig {
    /// Create a new configuration with default settings
    pub fn new() -> Self {
        Self {
            cycle_detection: true,
            max_depth: 1000,
            compress_ids: false,
        }
    }
    
    /// Configuration optimized for performance
    pub fn performance_optimized() -> Self {
        Self {
            cycle_detection: false, // Faster but no cycle detection
            max_depth: 500,
            compress_ids: false,
        }
    }
    
    /// Configuration optimized for space efficiency
    pub fn space_optimized() -> Self {
        Self {
            cycle_detection: true,
            max_depth: 2000,
            compress_ids: true, // Smaller serialized size
        }
    }
    
    /// Configuration for robust handling of complex graphs
    pub fn robust() -> Self {
        Self {
            cycle_detection: true,
            max_depth: 5000,
            compress_ids: false,
        }
    }
}

impl Default for SmartPtrConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level smart pointer serialization utilities
pub struct SmartPtrSerializer {
    config: SmartPtrConfig,
}

impl SmartPtrSerializer {
    /// Create a new serializer with the given configuration
    pub fn new(config: SmartPtrConfig) -> Self {
        Self { config }
    }
    
    /// Create a serializer with default configuration
    pub fn default() -> Self {
        Self::new(SmartPtrConfig::default())
    }
    
    /// Serialize a smart pointer to bytes
    pub fn serialize_to_bytes<T, P>(&self, ptr: &P) -> Result<Vec<u8>>
    where
        P: SmartPtrSerialize<T>,
        T: SerializableType,
    {
        let mut output = crate::io::VecDataOutput::new();
        let mut context = if self.config.cycle_detection {
            SerializationContext::new()
        } else {
            SerializationContext::without_cycle_detection()
        };
        
        ptr.serialize_with_context(&mut output, &mut context)?;
        Ok(output.into_vec())
    }
    
    /// Deserialize a smart pointer from bytes
    pub fn deserialize_from_bytes<T, P>(&self, bytes: &[u8]) -> Result<P>
    where
        P: SmartPtrSerialize<T>,
        T: SerializableType,
    {
        let mut input = crate::io::SliceDataInput::new(bytes);
        let mut context = DeserializationContext::new();
        
        P::deserialize_with_context(&mut input, &mut context)
    }
}

// Implementations for basic types to make them SerializableType
impl SerializableType for i32 {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(*self as u32)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        Ok(input.read_u32()? as i32)
    }
}

impl SerializableType for u32 {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(*self)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        input.read_u32()
    }
}

impl SerializableType for String {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_length_prefixed_string(self)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        input.read_length_prefixed_string()
    }
}

impl<T: SerializableType> SerializableType for Vec<T> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(self.len() as u32)?;
        for item in self {
            item.serialize(output)?;
        }
        Ok(())
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let len = input.read_u32()? as usize;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(T::deserialize(input)?);
        }
        Ok(vec)
    }
}

// Bridge implementation for Rc<T>
impl<T: SerializableType> SerializableType for Rc<T> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as SmartPtrSerialize<T>>::serialize(self, output)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as SmartPtrSerialize<T>>::deserialize(input)
    }
}

// Bridge implementation for Arc<T> - requires Send + Sync for thread safety
impl<T: SerializableType + Send + Sync> SerializableType for Arc<T> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as SmartPtrSerialize<T>>::serialize(self, output)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as SmartPtrSerialize<T>>::deserialize(input)
    }
}

// Bridge implementation for Box<T>
impl<T: SerializableType> SerializableType for Box<T> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as SmartPtrSerialize<T>>::serialize(self, output)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as SmartPtrSerialize<T>>::deserialize(input)
    }
}

// Implement for primitive types using available DataInput/DataOutput methods
impl SerializableType for u8 {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u8(*self)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        input.read_u8()
    }
}

impl SerializableType for u16 {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u16(*self)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        input.read_u16()
    }
}

impl SerializableType for u64 {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u64(*self)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        input.read_u64()
    }
}

impl SerializableType for i8 {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u8(*self as u8)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        Ok(input.read_u8()? as i8)
    }
}

impl SerializableType for i16 {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u16(*self as u16)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        Ok(input.read_u16()? as i16)
    }
}

impl SerializableType for i64 {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u64(*self as u64)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        Ok(input.read_u64()? as i64)
    }
}

impl SerializableType for bool {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u8(if *self { 1 } else { 0 })
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        Ok(input.read_u8()? != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{SliceDataInput, VecDataOutput};

    #[test]
    fn test_box_serialization() {
        let boxed_value = Box::new(42i32);
        let mut output = VecDataOutput::new();
        
        <Box<i32> as SerializableType>::serialize(&boxed_value, &mut output).unwrap();
        let bytes = output.into_vec();
        
        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Box<i32> as SerializableType>::deserialize(&mut input).unwrap();
        
        assert_eq!(*deserialized, 42);
    }

    #[test]
    fn test_option_box_serialization() {
        // Test Some case
        let some_value = Some(Box::new(42i32));
        let mut output = VecDataOutput::new();
        
        <Option<Box<i32>> as SmartPtrSerialize<i32>>::serialize(&some_value, &mut output).unwrap();
        let bytes = output.into_vec();
        
        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Option<Box<i32>> as SmartPtrSerialize<i32>>::deserialize(&mut input).unwrap();
        
        assert_eq!(deserialized.map(|b| *b), Some(42));
        
        // Test None case
        let none_value: Option<Box<i32>> = None;
        let mut output = VecDataOutput::new();
        
        <Option<Box<i32>> as SmartPtrSerialize<i32>>::serialize(&none_value, &mut output).unwrap();
        let bytes = output.into_vec();
        
        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Option<Box<i32>> as SmartPtrSerialize<i32>>::deserialize(&mut input).unwrap();
        
        assert!(deserialized.is_none());
    }

    #[test]
    fn test_rc_serialization() {
        let rc_value = Rc::new(42i32);
        let mut output = VecDataOutput::new();
        
        <Rc<i32> as SerializableType>::serialize(&rc_value, &mut output).unwrap();
        let bytes = output.into_vec();
        
        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Rc<i32> as SerializableType>::deserialize(&mut input).unwrap();
        
        assert_eq!(*deserialized, 42);
    }

    #[test]
    fn test_arc_serialization() {
        let arc_value = Arc::new(42i32);
        let mut output = VecDataOutput::new();
        
        <Arc<i32> as SerializableType>::serialize(&arc_value, &mut output).unwrap();
        let bytes = output.into_vec();
        
        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Arc<i32> as SerializableType>::deserialize(&mut input).unwrap();
        
        assert_eq!(*deserialized, 42);
    }

    #[test]
    fn test_shared_object_serialization() {
        let shared_value = Rc::new(42i32);
        let clone1 = shared_value.clone();
        let clone2 = shared_value.clone();
        
        let mut context = SerializationContext::new();
        let mut output = VecDataOutput::new();
        
        // Serialize multiple references to the same object
        <Rc<i32> as SmartPtrSerialize<i32>>::serialize_with_context(&clone1, &mut output, &mut context).unwrap();
        <Rc<i32> as SmartPtrSerialize<i32>>::serialize_with_context(&clone2, &mut output, &mut context).unwrap();
        
        let bytes = output.into_vec();
        let mut input = SliceDataInput::new(&bytes);
        let mut deserialize_context = DeserializationContext::new();
        
        let deserialized1 = <Rc<i32> as SmartPtrSerialize<i32>>::deserialize_with_context(&mut input, &mut deserialize_context).unwrap();
        let deserialized2 = <Rc<i32> as SmartPtrSerialize<i32>>::deserialize_with_context(&mut input, &mut deserialize_context).unwrap();
        
        assert_eq!(*deserialized1, 42);
        assert_eq!(*deserialized2, 42);
        
        // They should be the same object (same allocation)
        assert!(Rc::ptr_eq(&deserialized1, &deserialized2));
    }

    #[test]
    fn test_serialization_context() {
        let mut context = SerializationContext::new();
        
        let value1 = 42i32;
        let value2 = 43i32;
        
        // Same object should get the same ID
        let id1a = context.get_or_assign_id(&value1);
        let id1b = context.get_or_assign_id(&value1);
        assert_eq!(id1a, id1b);
        
        // Different object should get different ID
        let id2 = context.get_or_assign_id(&value2);
        assert_ne!(id1a, id2);
        
        // Should detect already seen objects
        assert!(context.has_seen(&value1));
        assert!(context.has_seen(&value2));
    }

    #[test]
    fn test_smart_ptr_serializer() {
        let serializer = SmartPtrSerializer::default();
        let boxed_value = Box::new(42i32);
        
        let bytes = serializer.serialize_to_bytes(&boxed_value).unwrap();
        let deserialized: Box<i32> = serializer.deserialize_from_bytes(&bytes).unwrap();
        
        assert_eq!(*deserialized, 42);
    }

    #[test]
    fn test_performance_config() {
        let config = SmartPtrConfig::performance_optimized();
        assert!(!config.cycle_detection);
        assert_eq!(config.max_depth, 500);
        
        let config = SmartPtrConfig::space_optimized();
        assert!(config.cycle_detection);
        assert!(config.compress_ids);
    }

    #[test]
    fn test_vec_serialization() {
        let vec_value = vec![1i32, 2, 3, 4, 5];
        let mut output = VecDataOutput::new();
        
        <Vec<i32> as SerializableType>::serialize(&vec_value, &mut output).unwrap();
        let bytes = output.into_vec();
        
        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Vec<i32> as SerializableType>::deserialize(&mut input).unwrap();
        
        assert_eq!(deserialized, vec_value);
    }

    #[test]
    fn test_string_serialization() {
        let string_value = "Hello, World!".to_string();
        let mut output = VecDataOutput::new();
        
        <String as SerializableType>::serialize(&string_value, &mut output).unwrap();
        let bytes = output.into_vec();
        
        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <String as SerializableType>::deserialize(&mut input).unwrap();
        
        assert_eq!(deserialized, string_value);
    }

    #[test]
    fn test_complex_nested_structure() {
        let nested = Box::new(vec![
            Rc::new("Hello".to_string()),
            Rc::new("World".to_string()),
        ]);
        
        let mut output = VecDataOutput::new();
        <Box<Vec<Rc<String>>> as SerializableType>::serialize(&nested, &mut output).unwrap();
        let bytes = output.into_vec();
        
        let mut input = SliceDataInput::new(&bytes);
        let deserialized = <Box<Vec<Rc<String>>> as SerializableType>::deserialize(&mut input).unwrap();
        
        assert_eq!(deserialized.len(), 2);
        assert_eq!(*deserialized[0], "Hello");
        assert_eq!(*deserialized[1], "World");
    }
}