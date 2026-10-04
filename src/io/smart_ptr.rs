//! Smart pointer serialization for reference-counted objects
//!
//! This module provides automatic serialization and deserialization support for
//! Rust's smart pointer types including Box<T>, Arc<T>, Rc<T>, and custom smart pointers.
//! Supports null handling, reference counting, and circular reference detection.

use crate::error::{Result, ZiporaError};
use crate::io::data_input::prealloc_capacity;
use crate::io::{DataInput, DataOutput};
use std::collections::HashMap;
// Hash import removed as it's not used in this module
use std::rc::{Rc, Weak as RcWeak};
use std::sync::{Arc, Weak as ArcWeak};

use std::any::Any;
use std::cell::RefCell;

thread_local! {
    static ACTIVE_SER_CTX: RefCell<Option<SerializationContext>> = const { RefCell::new(None) };
    static ACTIVE_DE_CTX: RefCell<Option<DeserializationContext>> = const { RefCell::new(None) };
}

/// Serialization context for tracking shared objects and preventing cycles
#[derive(Debug)]
pub struct SerializationContext {
    /// Object ID counter for reference tracking
    next_id: u32,
    /// Map from object pointer to assigned ID
    object_ids: HashMap<usize, u32>,
    /// Cloned smart pointers kept alive so their addresses cannot be reused
    /// by the allocator while this context is active.
    retained_objects: Vec<Box<dyn Any>>,
    /// Enable circular reference detection
    detect_cycles: bool,
    /// Encode object IDs with variable-length integers
    compress_ids: bool,
    /// Current recursion depth across nested containers and smart pointers
    depth: usize,
    /// Maximum allowed recursion depth
    max_depth: usize,
}

impl SerializationContext {
    /// Create a new serialization context
    pub fn new() -> Self {
        Self {
            next_id: 1, // Start from 1, reserve 0 for null
            object_ids: HashMap::new(),
            retained_objects: Vec::new(),
            detect_cycles: true,
            compress_ids: false,
            depth: 0,
            max_depth: 1000,
        }
    }

    /// Create context with cycle detection disabled (for performance)
    pub fn without_cycle_detection() -> Self {
        Self {
            next_id: 1,
            object_ids: HashMap::new(),
            retained_objects: Vec::new(),
            detect_cycles: false,
            compress_ids: false,
            depth: 0,
            max_depth: 1000,
        }
    }

    /// Get or assign an ID for an object
    fn get_or_assign_id<T>(&mut self, ptr: &T) -> u32 {
        let addr = ptr as *const T as usize;

        if let Some(&id) = self.object_ids.get(&addr) {
            id
        } else {
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            if self.detect_cycles {
                self.object_ids.insert(addr, id);
            }
            id
        }
    }

    /// Keep a clone of a serialized smart pointer alive for the lifetime of
    /// this context so its heap address cannot be recycled for an unrelated
    /// allocation.
    fn retain_object<P: 'static>(&mut self, ptr: P) {
        if self.detect_cycles {
            self.retained_objects.push(Box::new(ptr));
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

    fn write_id<O: DataOutput>(&self, output: &mut O, id: u32) -> Result<()> {
        if self.compress_ids {
            output.write_var_int(id as u64)
        } else {
            output.write_u32(id)
        }
    }

    pub(crate) fn run_nested<R>(&mut self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        if self.depth >= self.max_depth {
            return Err(ZiporaError::invalid_data("Maximum nesting depth exceeded"));
        }
        struct Guard<'a> {
            caller: &'a mut SerializationContext,
            prev: Option<SerializationContext>,
        }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                ACTIVE_SER_CTX.with(|cell| {
                    if let Some(mut active) = cell.borrow_mut().take() {
                        active.depth = active.depth.saturating_sub(1);
                        *self.caller = active;
                    }
                    *cell.borrow_mut() = self.prev.take();
                });
            }
        }
        self.depth += 1;
        let prev = ACTIVE_SER_CTX.with(|cell| {
            let taken = std::mem::take(self);
            cell.borrow_mut().replace(taken)
        });
        let _guard = Guard { caller: self, prev };
        f()
    }

    /// Clear the context for reuse
    pub fn clear(&mut self) {
        self.next_id = 1;
        self.object_ids.clear();
        self.retained_objects.clear();
        self.depth = 0;
    }
}

impl Default for SerializationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Deserialization context for reconstructing shared objects
#[derive(Debug)]
pub struct DeserializationContext {
    /// Map from object ID to reconstructed smart pointer
    objects: HashMap<u32, Box<dyn Any>>,
    /// Decode object IDs with variable-length integers
    compress_ids: bool,
    /// Current recursion depth across nested containers and smart pointers
    depth: usize,
    /// Maximum allowed recursion depth
    max_depth: usize,
}

impl DeserializationContext {
    /// Create a new deserialization context
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            compress_ids: false,
            depth: 0,
            max_depth: 1000,
        }
    }

    /// Store an object with its ID
    pub fn store_object<T: 'static>(&mut self, id: u32, object: T) {
        self.objects.insert(id, Box::new(object));
    }

    /// Get an object by its ID
    pub fn get_object<T: 'static>(&self, id: u32) -> Option<&T> {
        self.objects.get(&id)?.downcast_ref::<T>()
    }

    fn read_id<I: DataInput>(&self, input: &mut I) -> Result<u32> {
        if self.compress_ids {
            u32::try_from(input.read_var_int()?)
                .map_err(|_| ZiporaError::invalid_data("Compressed object ID overflows u32"))
        } else {
            input.read_u32()
        }
    }

    pub(crate) fn run_nested<R>(&mut self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        if self.depth >= self.max_depth {
            return Err(ZiporaError::invalid_data("Maximum nesting depth exceeded"));
        }
        struct Guard<'a> {
            caller: &'a mut DeserializationContext,
            prev: Option<DeserializationContext>,
        }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                ACTIVE_DE_CTX.with(|cell| {
                    if let Some(mut active) = cell.borrow_mut().take() {
                        active.depth = active.depth.saturating_sub(1);
                        *self.caller = active;
                    }
                    *cell.borrow_mut() = self.prev.take();
                });
            }
        }
        self.depth += 1;
        let prev = ACTIVE_DE_CTX.with(|cell| {
            let taken = std::mem::take(self);
            cell.borrow_mut().replace(taken)
        });
        let _guard = Guard { caller: self, prev };
        f()
    }

    /// Clear the context for reuse
    pub fn clear(&mut self) {
        self.objects.clear();
        self.depth = 0;
    }
}

impl Default for DeserializationContext {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn with_active_ser_ctx<R>(
    f: impl FnOnce(&mut SerializationContext) -> Result<R>,
) -> Result<R> {
    let existing = ACTIVE_SER_CTX.with(|cell| cell.borrow_mut().take());
    match existing {
        Some(ctx) => {
            struct RestoreGuard(Option<SerializationContext>);
            impl Drop for RestoreGuard {
                fn drop(&mut self) {
                    if let Some(ctx) = self.0.take() {
                        ACTIVE_SER_CTX.with(|cell| {
                            *cell.borrow_mut() = Some(ctx);
                        });
                    }
                }
            }
            let mut guard = RestoreGuard(Some(ctx));
            let ctx_ref = match guard.0.as_mut() {
                Some(c) => c,
                None => unreachable!(),
            };
            f(ctx_ref)
        }
        None => {
            let mut fresh = SerializationContext::new();
            f(&mut fresh)
        }
    }
}

pub(crate) fn with_active_de_ctx<R>(
    f: impl FnOnce(&mut DeserializationContext) -> Result<R>,
) -> Result<R> {
    let existing = ACTIVE_DE_CTX.with(|cell| cell.borrow_mut().take());
    match existing {
        Some(ctx) => {
            struct RestoreGuard(Option<DeserializationContext>);
            impl Drop for RestoreGuard {
                fn drop(&mut self) {
                    if let Some(ctx) = self.0.take() {
                        ACTIVE_DE_CTX.with(|cell| {
                            *cell.borrow_mut() = Some(ctx);
                        });
                    }
                }
            }
            let mut guard = RestoreGuard(Some(ctx));
            let ctx_ref = match guard.0.as_mut() {
                Some(c) => c,
                None => unreachable!(),
            };
            f(ctx_ref)
        }
        None => {
            let mut fresh = DeserializationContext::new();
            f(&mut fresh)
        }
    }
}

pub(crate) fn with_nested_ser_depth<R>(
    depth: usize,
    max_depth: usize,
    f: impl FnOnce() -> Result<R>,
) -> Result<R> {
    if depth > max_depth {
        return Err(ZiporaError::invalid_data("Maximum nesting depth exceeded"));
    }
    let existing = ACTIVE_SER_CTX.with(|cell| cell.borrow_mut().take());
    match existing {
        Some(mut ctx) => {
            let prev_depth = ctx.depth;
            let prev_max = ctx.max_depth;
            let effective_depth = prev_depth.max(depth);
            let effective_max = prev_max.min(max_depth);
            if effective_depth > effective_max {
                ACTIVE_SER_CTX.with(|cell| {
                    *cell.borrow_mut() = Some(ctx);
                });
                return Err(ZiporaError::invalid_data("Maximum nesting depth exceeded"));
            }
            ctx.depth = effective_depth;
            ctx.max_depth = effective_max;
            struct RestoreGuard {
                prev_depth: usize,
                prev_max: usize,
            }
            impl Drop for RestoreGuard {
                fn drop(&mut self) {
                    ACTIVE_SER_CTX.with(|cell| {
                        if let Some(active) = cell.borrow_mut().as_mut() {
                            active.depth = self.prev_depth;
                            active.max_depth = self.prev_max;
                        }
                    });
                }
            }
            ACTIVE_SER_CTX.with(|cell| {
                *cell.borrow_mut() = Some(ctx);
            });
            let _guard = RestoreGuard {
                prev_depth,
                prev_max,
            };
            f()
        }
        None => {
            let mut fresh = SerializationContext::new();
            fresh.depth = depth;
            fresh.max_depth = max_depth;
            struct ClearGuard;
            impl Drop for ClearGuard {
                fn drop(&mut self) {
                    ACTIVE_SER_CTX.with(|cell| {
                        let _ = cell.borrow_mut().take();
                    });
                }
            }
            ACTIVE_SER_CTX.with(|cell| {
                *cell.borrow_mut() = Some(fresh);
            });
            let _guard = ClearGuard;
            f()
        }
    }
}

pub(crate) fn with_nested_de_depth<R>(
    depth: usize,
    max_depth: usize,
    f: impl FnOnce() -> Result<R>,
) -> Result<R> {
    if depth > max_depth {
        return Err(ZiporaError::invalid_data("Maximum nesting depth exceeded"));
    }
    let existing = ACTIVE_DE_CTX.with(|cell| cell.borrow_mut().take());
    match existing {
        Some(mut ctx) => {
            let prev_depth = ctx.depth;
            let prev_max = ctx.max_depth;
            let effective_depth = prev_depth.max(depth);
            let effective_max = prev_max.min(max_depth);
            if effective_depth > effective_max {
                ACTIVE_DE_CTX.with(|cell| {
                    *cell.borrow_mut() = Some(ctx);
                });
                return Err(ZiporaError::invalid_data("Maximum nesting depth exceeded"));
            }
            ctx.depth = effective_depth;
            ctx.max_depth = effective_max;
            struct RestoreGuard {
                prev_depth: usize,
                prev_max: usize,
            }
            impl Drop for RestoreGuard {
                fn drop(&mut self) {
                    ACTIVE_DE_CTX.with(|cell| {
                        if let Some(active) = cell.borrow_mut().as_mut() {
                            active.depth = self.prev_depth;
                            active.max_depth = self.prev_max;
                        }
                    });
                }
            }
            ACTIVE_DE_CTX.with(|cell| {
                *cell.borrow_mut() = Some(ctx);
            });
            let _guard = RestoreGuard {
                prev_depth,
                prev_max,
            };
            f()
        }
        None => {
            let mut fresh = DeserializationContext::new();
            fresh.depth = depth;
            fresh.max_depth = max_depth;
            struct ClearGuard;
            impl Drop for ClearGuard {
                fn drop(&mut self) {
                    ACTIVE_DE_CTX.with(|cell| {
                        let _ = cell.borrow_mut().take();
                    });
                }
            }
            ACTIVE_DE_CTX.with(|cell| {
                *cell.borrow_mut() = Some(fresh);
            });
            let _guard = ClearGuard;
            f()
        }
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
        context: &mut DeserializationContext,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Serialize without an explicit context, reusing any active outer
    /// [`SerializationContext`] on the current thread.
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        with_active_ser_ctx(|context| self.serialize_with_context(output, context))
    }

    /// Deserialize without an explicit context, reusing any active outer
    /// [`DeserializationContext`] on the current thread.
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self>
    where
        Self: Sized,
    {
        with_active_de_ctx(|context| Self::deserialize_with_context(input, context))
    }
}

/// Marker trait for types that are serializable
pub trait SerializableType: 'static {
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
        context: &mut SerializationContext,
    ) -> Result<()> {
        // Box is always unique, no need for reference tracking
        output.write_u8(1)?; // Non-null marker
        context.run_nested(|| self.as_ref().serialize(output))
    }

    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Err(ZiporaError::invalid_data("Box cannot be null")),
            1 => {
                let value = context.run_nested(|| T::deserialize(input))?;
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
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(None),
            1 => {
                let boxed = Box::<T>::deserialize_with_context(input, context)?;
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
            context.write_id(output, id)?;
        } else {
            // First time seeing this object, serialize it
            let id = context.get_or_assign_id(obj_ref);
            context.retain_object(self.clone());
            output.write_u8(1)?; // New object marker
            context.write_id(output, id)?;
            context.run_nested(|| obj_ref.serialize(output))?;
        }

        Ok(())
    }

    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Err(ZiporaError::invalid_data("Rc cannot be null")),
            1 => {
                // New object
                let id = context.read_id(input)?;
                let value = context.run_nested(|| T::deserialize(input))?;
                let rc = Rc::new(value);
                context.store_object(id, rc.clone());
                Ok(rc)
            }
            2 => {
                // Reference to existing object
                let id = context.read_id(input)?;
                context
                    .get_object::<Self>(id)
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
            context.write_id(output, id)?;
        } else {
            // First time seeing this object, serialize it
            let id = context.get_or_assign_id(obj_ref);
            context.retain_object(self.clone());
            output.write_u8(1)?; // New object marker
            context.write_id(output, id)?;
            context.run_nested(|| obj_ref.serialize(output))?;
        }

        Ok(())
    }

    fn deserialize_with_context<I: DataInput>(
        input: &mut I,
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Err(ZiporaError::invalid_data("Arc cannot be null")),
            1 => {
                // New object
                let id = context.read_id(input)?;
                let value = context.run_nested(|| T::deserialize(input))?;
                let arc = Arc::new(value);
                context.store_object(id, arc.clone());
                Ok(arc)
            }
            2 => {
                // Reference to existing object
                let id = context.read_id(input)?;
                context
                    .get_object::<Self>(id)
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
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(RcWeak::new()),
            1 => {
                let rc = Rc::<T>::deserialize_with_context(input, context)?;
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
        context: &mut DeserializationContext,
    ) -> Result<Self> {
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(ArcWeak::new()),
            1 => {
                let arc = Arc::<T>::deserialize_with_context(input, context)?;
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

impl Default for SmartPtrSerializer {
    fn default() -> Self {
        Self::new(SmartPtrConfig::default())
    }
}

impl SmartPtrSerializer {
    /// Create a new serializer with the given configuration
    pub fn new(config: SmartPtrConfig) -> Self {
        Self { config }
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
        context.compress_ids = self.config.compress_ids;
        context.max_depth = self.config.max_depth;

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
        context.compress_ids = self.config.compress_ids;
        context.max_depth = self.config.max_depth;

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
        if self.is_empty() {
            return Ok(());
        }
        with_active_ser_ctx(|context| {
            context.run_nested(|| {
                for item in self {
                    item.serialize(output)?;
                }
                Ok(())
            })
        })
    }

    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let len = input.read_u32()? as usize;
        if len == 0 {
            return Ok(Vec::new());
        }
        // `len` is untrusted: reserve within the read-ahead budget and let
        // the vector grow as elements arrive.
        let mut vec = Vec::with_capacity(prealloc_capacity(
            len,
            std::mem::size_of::<T>(),
            input.remaining_bytes(),
        ));
        with_active_de_ctx(|context| {
            context.run_nested(|| {
                for _ in 0..len {
                    vec.push(T::deserialize(input)?);
                }
                Ok(vec)
            })
        })
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

// Bridge implementation for RcWeak<T>
impl<T: SerializableType> SerializableType for RcWeak<T> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        <Self as SmartPtrSerialize<T>>::serialize(self, output)
    }

    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        <Self as SmartPtrSerialize<T>>::deserialize(input)
    }
}

// Bridge implementation for ArcWeak<T>
impl<T: SerializableType + Send + Sync> SerializableType for ArcWeak<T> {
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
        // `serialize` writes only 0 or 1; any other byte is corrupt input.
        match input.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(ZiporaError::invalid_data("Invalid bool byte")),
        }
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
        let deserialized =
            <Option<Box<i32>> as SmartPtrSerialize<i32>>::deserialize(&mut input).unwrap();

        assert_eq!(deserialized.map(|b| *b), Some(42));

        // Test None case
        let none_value: Option<Box<i32>> = None;
        let mut output = VecDataOutput::new();

        <Option<Box<i32>> as SmartPtrSerialize<i32>>::serialize(&none_value, &mut output).unwrap();
        let bytes = output.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let deserialized =
            <Option<Box<i32>> as SmartPtrSerialize<i32>>::deserialize(&mut input).unwrap();

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
        <Rc<i32> as SmartPtrSerialize<i32>>::serialize_with_context(
            &clone1,
            &mut output,
            &mut context,
        )
        .unwrap();
        <Rc<i32> as SmartPtrSerialize<i32>>::serialize_with_context(
            &clone2,
            &mut output,
            &mut context,
        )
        .unwrap();

        let bytes = output.into_vec();
        let mut input = SliceDataInput::new(&bytes);
        let mut deserialize_context = DeserializationContext::new();

        let deserialized1 = <Rc<i32> as SmartPtrSerialize<i32>>::deserialize_with_context(
            &mut input,
            &mut deserialize_context,
        )
        .unwrap();
        let deserialized2 = <Rc<i32> as SmartPtrSerialize<i32>>::deserialize_with_context(
            &mut input,
            &mut deserialize_context,
        )
        .unwrap();

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
        let deserialized =
            <Box<Vec<Rc<String>>> as SerializableType>::deserialize(&mut input).unwrap();

        assert_eq!(deserialized.len(), 2);
        assert_eq!(*deserialized[0], "Hello");
        assert_eq!(*deserialized[1], "World");
    }

    #[test]
    fn test_c9_sp6_bool_rejects_bytes_above_one() {
        for byte in [2u8, 0xFF] {
            let bytes = [byte];
            let mut input = SliceDataInput::new(&bytes);
            assert!(
                <bool as SerializableType>::deserialize(&mut input).is_err(),
                "byte {byte:#04x} must not decode as a bool"
            );
        }
        for (byte, expected) in [(0u8, false), (1u8, true)] {
            let bytes = [byte];
            let mut input = SliceDataInput::new(&bytes);
            assert_eq!(
                <bool as SerializableType>::deserialize(&mut input).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn test_c9_sp2_dangling_weak_round_trips() {
        let mut out = VecDataOutput::new();
        let dangling_rc: RcWeak<u32> = RcWeak::new();
        <RcWeak<u32> as SmartPtrSerialize<u32>>::serialize(&dangling_rc, &mut out).unwrap();
        let dangling_arc: ArcWeak<u32> = ArcWeak::new();
        <ArcWeak<u32> as SmartPtrSerialize<u32>>::serialize(&dangling_arc, &mut out).unwrap();
        out.write_u32(99).unwrap();

        let bytes = out.into_vec();
        assert_eq!(bytes, [0, 0, 99, 0, 0, 0]);

        let mut input = SliceDataInput::new(&bytes);
        let back_rc = <RcWeak<u32> as SmartPtrSerialize<u32>>::deserialize(&mut input).unwrap();
        assert!(back_rc.upgrade().is_none());
        let back_arc = <ArcWeak<u32> as SmartPtrSerialize<u32>>::deserialize(&mut input).unwrap();
        assert!(back_arc.upgrade().is_none());
        assert_eq!(input.read_u32().unwrap(), 99);
    }

    struct StrongAndWeak {
        strong: Rc<u32>,
        weak: RcWeak<u32>,
    }

    impl SerializableType for StrongAndWeak {
        fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
            <Rc<u32> as SmartPtrSerialize<u32>>::serialize(&self.strong, output)?;
            <RcWeak<u32> as SmartPtrSerialize<u32>>::serialize(&self.weak, output)
        }

        fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
            let strong = <Rc<u32> as SmartPtrSerialize<u32>>::deserialize(input)?;
            let weak = <RcWeak<u32> as SmartPtrSerialize<u32>>::deserialize(input)?;
            Ok(Self { strong, weak })
        }
    }

    #[test]
    fn test_c9_sp3_live_weak_and_shared_context_round_trip() {
        let strong = Rc::new(5u32);
        let weak = Rc::downgrade(&strong);
        let mut out = VecDataOutput::new();
        <RcWeak<u32> as SmartPtrSerialize<u32>>::serialize(&weak, &mut out).unwrap();
        let bytes = out.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let mut ctx = DeserializationContext::new();
        let back =
            <RcWeak<u32> as SmartPtrSerialize<u32>>::deserialize_with_context(&mut input, &mut ctx)
                .unwrap();
        assert_eq!(back.upgrade().map(|r| *r), Some(5));

        let pair = Box::new(StrongAndWeak {
            strong: strong.clone(),
            weak: Rc::downgrade(&strong),
        });
        let mut pair_out = VecDataOutput::new();
        <Box<StrongAndWeak> as SmartPtrSerialize<StrongAndWeak>>::serialize(&pair, &mut pair_out)
            .unwrap();
        let pair_bytes = pair_out.into_vec();
        let mut pair_in = SliceDataInput::new(&pair_bytes);
        let pair_back =
            <Box<StrongAndWeak> as SmartPtrSerialize<StrongAndWeak>>::deserialize(&mut pair_in)
                .unwrap();
        let upgraded = pair_back.weak.upgrade().expect("weak reference must upgrade");
        assert_eq!(*upgraded, 5);
        assert!(Rc::ptr_eq(&pair_back.strong, &upgraded));
    }

    #[test]
    fn test_c9_sp4_temporary_rc_addresses_are_not_confused() {
        let mut ctx = SerializationContext::new();
        let mut out = VecDataOutput::new();
        for i in 0..4u32 {
            <Rc<u32> as SmartPtrSerialize<u32>>::serialize_with_context(
                &Rc::new(i),
                &mut out,
                &mut ctx,
            )
            .unwrap();
        }

        let bytes = out.into_vec();
        let mut input = SliceDataInput::new(&bytes);
        let mut de_ctx = DeserializationContext::new();
        let mut decoded = Vec::new();
        for _ in 0..4 {
            let rc = <Rc<u32> as SmartPtrSerialize<u32>>::deserialize_with_context(
                &mut input,
                &mut de_ctx,
            )
            .unwrap();
            decoded.push(*rc);
        }
        assert_eq!(decoded, [0, 1, 2, 3]);
    }

    static CYCLE_DEPTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    struct CyclicNode {
        val: u32,
        next: std::cell::RefCell<Option<Rc<CyclicNode>>>,
    }

    impl SerializableType for CyclicNode {
        fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
            use std::sync::atomic::Ordering;
            let d = CYCLE_DEPTH.fetch_add(1, Ordering::Relaxed) + 1;
            struct Reset;
            impl Drop for Reset {
                fn drop(&mut self) {
                    CYCLE_DEPTH.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            let _reset = Reset;
            if d >= 16 {
                return Err(ZiporaError::invalid_data("cycle was not detected"));
            }
            self.val.serialize(output)?;
            match self.next.borrow().as_ref() {
                Some(rc) => {
                    output.write_u8(1)?;
                    <Rc<CyclicNode> as SerializableType>::serialize(rc, output)
                }
                None => output.write_u8(0),
            }
        }

        fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
            let val = u32::deserialize(input)?;
            let next = match input.read_u8()? {
                0 => None,
                1 => Some(<Rc<CyclicNode> as SerializableType>::deserialize(input)?),
                _ => return Err(ZiporaError::invalid_data("Invalid CyclicNode marker")),
            };
            Ok(Self {
                val,
                next: std::cell::RefCell::new(next),
            })
        }
    }

    #[test]
    fn test_c9_sp5_nested_sharing_and_cycle_detection() {
        let shared = Rc::new("shared-payload".to_string());
        let vec = vec![shared.clone(), shared.clone(), shared];
        let mut out = VecDataOutput::new();
        <Vec<Rc<String>> as SerializableType>::serialize(&vec, &mut out).unwrap();
        let bytes = out.into_vec();

        let mut input = SliceDataInput::new(&bytes);
        let back = <Vec<Rc<String>> as SerializableType>::deserialize(&mut input).unwrap();
        assert_eq!(back.len(), 3);
        assert!(Rc::ptr_eq(&back[0], &back[1]));
        assert!(Rc::ptr_eq(&back[1], &back[2]));

        // Self-referential node must emit a reference marker (2) rather than recursing.
        CYCLE_DEPTH.store(0, std::sync::atomic::Ordering::Relaxed);
        let node = Rc::new(CyclicNode {
            val: 7,
            next: std::cell::RefCell::new(None),
        });
        *node.next.borrow_mut() = Some(node.clone());
        let mut cyc_out = VecDataOutput::new();
        let res = <Rc<CyclicNode> as SmartPtrSerialize<CyclicNode>>::serialize(&node, &mut cyc_out);
        *node.next.borrow_mut() = None;
        res.expect("cycle detection must stop self-referential recursion");
        let cyc_bytes = cyc_out.into_vec();
        // [marker=1, id=1(le u32), val=7(le u32), Some=1, marker=2, id=1(le u32)]
        assert_eq!(cyc_bytes, [1, 1, 0, 0, 0, 7, 0, 0, 0, 1, 2, 1, 0, 0, 0]);
    }

    #[test]
    fn test_c9_ct5_smart_ptr_serializer_enforces_max_depth() {
        #[derive(Debug, PartialEq)]
        struct ChainNode {
            next: Option<Box<ChainNode>>,
        }

        impl SerializableType for ChainNode {
            fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
                <Option<Box<ChainNode>> as SmartPtrSerialize<ChainNode>>::serialize(
                    &self.next, output,
                )
            }

            fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
                Ok(Self {
                    next: <Option<Box<ChainNode>> as SmartPtrSerialize<ChainNode>>::deserialize(
                        input,
                    )?,
                })
            }
        }

        let build_chain = |depth: usize| {
            let mut node = Box::new(ChainNode { next: None });
            for _ in 0..depth {
                node = Box::new(ChainNode { next: Some(node) });
            }
            node
        };

        let shallow = build_chain(3);
        let deep = build_chain(12);

        let bounded = SmartPtrSerializer::new(SmartPtrConfig {
            max_depth: 8,
            ..SmartPtrConfig::default()
        });
        let shallow_bytes = bounded.serialize_to_bytes(&shallow).unwrap();
        let shallow_back: Box<ChainNode> = bounded.deserialize_from_bytes(&shallow_bytes).unwrap();
        assert_eq!(shallow_back, shallow);

        assert!(
            bounded.serialize_to_bytes(&deep).is_err(),
            "depth 12 must exceed SmartPtrConfig::max_depth = 8 on serialize"
        );

        let default_bytes = SmartPtrSerializer::default()
            .serialize_to_bytes(&deep)
            .unwrap();
        assert!(
            bounded
                .deserialize_from_bytes::<ChainNode, Box<ChainNode>>(&default_bytes)
                .is_err(),
            "depth 12 must exceed SmartPtrConfig::max_depth = 8 on deserialize"
        );
    }
}
