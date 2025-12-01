//! Generic Factory Pattern Implementation
//!
//! Provides a flexible, type-safe factory system for object creation with automatic
//! registration, discovery, and efficient runtime creation. Inspired by production
//! infrastructure patterns while leveraging Rust's ownership and trait systems.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, OnceLock};
use std::marker::PhantomData;
use crate::error::{ZiporaError, Result};

/// Type-erased creator function that can create objects of type T
pub type Creator<T> = Box<dyn Fn() -> Result<T> + Send + Sync>;

/// Registry for factory creators, providing thread-safe registration and creation
pub struct FactoryRegistry<T> {
    creators: RwLock<HashMap<String, Creator<T>>>,
    type_map: RwLock<HashMap<TypeId, String>>,
    _phantom: PhantomData<T>,
}

impl<T> Default for FactoryRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> FactoryRegistry<T> {
    /// Create a new factory registry
    pub fn new() -> Self {
        Self {
            creators: RwLock::new(HashMap::new()),
            type_map: RwLock::new(HashMap::new()),
            _phantom: PhantomData,
        }
    }

    /// Register a creator function with a given name
    pub fn register<F>(&self, name: &str, creator: F) -> Result<()>
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        let mut creators = self.creators.write()
            .map_err(|_| ZiporaError::io_error("Failed to acquire write lock on creators"))?;
        
        if creators.contains_key(name) {
            return Err(ZiporaError::invalid_data(&format!("Creator '{}' already registered", name)));
        }
        
        creators.insert(name.to_string(), Box::new(creator));
        Ok(())
    }

    /// Register a creator function with automatic type name derivation
    pub fn register_type<U, F>(&self, creator: F) -> Result<()>
    where
        U: 'static,
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        let type_name = std::any::type_name::<U>();
        let type_id = TypeId::of::<U>();
        
        // Store type mapping
        {
            let mut type_map = self.type_map.write()
                .map_err(|_| ZiporaError::io_error("Failed to acquire write lock on type_map"))?;
            type_map.insert(type_id, type_name.to_string());
        }
        
        self.register(type_name, creator)
    }

    /// Create an object by name
    pub fn create(&self, name: &str) -> Result<T> {
        let creators = self.creators.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock on creators"))?;
        
        let creator = creators.get(name)
            .ok_or_else(|| ZiporaError::not_found(&format!("Creator '{}' not found", name)))?;
        
        creator()
    }

    /// Create an object by type
    pub fn create_by_type<U: 'static>(&self) -> Result<T> {
        let type_id = TypeId::of::<U>();
        
        let type_map = self.type_map.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock on type_map"))?;
        
        let type_name = type_map.get(&type_id)
            .ok_or_else(|| ZiporaError::not_found(&format!("Type '{}' not registered", std::any::type_name::<U>())))?.clone();
        
        drop(type_map); // Release lock before calling create
        self.create(&type_name)
    }

    /// List all registered creator names
    pub fn list_creators(&self) -> Result<Vec<String>> {
        let creators = self.creators.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock on creators"))?;
        
        Ok(creators.keys().cloned().collect())
    }

    /// Get the number of registered creators
    pub fn creator_count(&self) -> Result<usize> {
        let creators = self.creators.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock on creators"))?;
        
        Ok(creators.len())
    }

    /// Check if a creator is registered
    pub fn contains(&self, name: &str) -> Result<bool> {
        let creators = self.creators.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock on creators"))?;
        
        Ok(creators.contains_key(name))
    }
}

/// Global factory registry for a specific type
pub struct GlobalFactory<T> {
    registry: OnceLock<FactoryRegistry<T>>,
    _phantom: PhantomData<T>,
}

impl<T> Default for GlobalFactory<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GlobalFactory<T> {
    /// Create a new global factory
    pub const fn new() -> Self {
        Self {
            registry: OnceLock::new(),
            _phantom: PhantomData,
        }
    }

    /// Get the global registry, initializing it if necessary
    fn registry(&self) -> &FactoryRegistry<T> {
        self.registry.get_or_init(|| FactoryRegistry::new())
    }

    /// Register a creator function globally
    pub fn register<F>(&self, name: &str, creator: F) -> Result<()>
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        self.registry().register(name, creator)
    }

    /// Register a creator function with automatic type name derivation
    pub fn register_type<U, F>(&self, creator: F) -> Result<()>
    where
        U: 'static,
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        self.registry().register_type::<U, F>(creator)
    }

    /// Create an object by name
    pub fn create(&self, name: &str) -> Result<T> {
        self.registry().create(name)
    }

    /// Create an object by type
    pub fn create_by_type<U: 'static>(&self) -> Result<T> {
        self.registry().create_by_type::<U>()
    }

    /// List all registered creator names
    pub fn list_creators(&self) -> Result<Vec<String>> {
        self.registry().list_creators()
    }

    /// Get the number of registered creators
    pub fn creator_count(&self) -> Result<usize> {
        self.registry().creator_count()
    }

    /// Check if a creator is registered
    pub fn contains(&self, name: &str) -> Result<bool> {
        self.registry().contains(name)
    }
}

/// Auto-registration helper for static initialization
pub struct AutoRegister<T> {
    _phantom: PhantomData<T>,
}

impl<T: Send + Sync + 'static> AutoRegister<T> {
    /// Register a creator function with the global factory
    pub fn new<F>(name: &str, creator: F) -> Self
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        // This will panic if registration fails, which is appropriate for static initialization
        global_factory::<T>().register(name, creator)
            .expect(&format!("Failed to register creator '{}'", name));
        
        Self {
            _phantom: PhantomData,
        }
    }

    /// Register a creator function with automatic type name derivation
    pub fn new_type<U, F>(creator: F) -> Self
    where
        U: 'static,
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        // This will panic if registration fails, which is appropriate for static initialization
        global_factory::<T>().register_type::<U, F>(creator)
            .expect(&format!("Failed to register creator for type '{}'", std::any::type_name::<U>()));
        
        Self {
            _phantom: PhantomData,
        }
    }
}

/// Get the global factory instance for type T
pub fn global_factory<T: Send + Sync + 'static>() -> &'static GlobalFactory<T> {
    use std::sync::Mutex;
    static FACTORIES: std::sync::LazyLock<Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> = 
        std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));
    
    let type_id = TypeId::of::<T>();
    
    // First, try to get an existing factory
    {
        let factories = FACTORIES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(factory_any) = factories.get(&type_id) {
            let factory = factory_any.downcast_ref::<GlobalFactory<T>>().unwrap();
            // Safety: The factory is stored in a static HashMap and lives for the entire program duration
            return unsafe { std::mem::transmute::<&GlobalFactory<T>, &'static GlobalFactory<T>>(factory) };
        }
    }
    
    // If not found, create a new one
    {
        let mut factories = FACTORIES.lock().unwrap_or_else(|e| e.into_inner());
        // Double-check pattern in case another thread created it while we were waiting for the lock
        if let Some(factory_any) = factories.get(&type_id) {
            let factory = factory_any.downcast_ref::<GlobalFactory<T>>().unwrap();
            // Safety: The factory is stored in a static HashMap and lives for the entire program duration
            return unsafe { std::mem::transmute::<&GlobalFactory<T>, &'static GlobalFactory<T>>(factory) };
        }
        
        let factory = Box::new(GlobalFactory::<T>::new());
        factories.insert(type_id, factory);
        
        let factory_any = factories.get(&type_id).unwrap();
        let factory = factory_any.downcast_ref::<GlobalFactory<T>>().unwrap();
        
        // Safety: The factory is stored in a static HashMap and lives for the entire program duration
        unsafe { std::mem::transmute::<&GlobalFactory<T>, &'static GlobalFactory<T>>(factory) }
    }
}

/// Trait for types that can be created by factories
pub trait Factoryable: Sized {
    /// Create an instance by name using the global factory
    fn create(name: &str) -> Result<Self>;
    
    /// List all available creator names
    fn list_creators() -> Result<Vec<String>>;
    
    /// Check if a creator is available
    fn has_creator(name: &str) -> Result<bool>;
}

/// Implement Factoryable for any type
impl<T: Send + Sync + 'static> Factoryable for T {
    fn create(name: &str) -> Result<Self> {
        global_factory::<T>().create(name)
    }
    
    fn list_creators() -> Result<Vec<String>> {
        global_factory::<T>().list_creators()
    }
    
    fn has_creator(name: &str) -> Result<bool> {
        global_factory::<T>().contains(name)
    }
}

/// Macro for convenient factory registration
#[macro_export]
macro_rules! register_factory {
    ($type:ty, $name:expr, $creator:expr) => {
        static _FACTORY_REG: $crate::dev_infrastructure::factory::AutoRegister<$type> = 
            $crate::dev_infrastructure::factory::AutoRegister::new($name, $creator);
    };
}

/// Macro for factory registration with automatic type name
#[macro_export]
macro_rules! register_factory_type {
    ($impl_type:ty, $trait_type:ty, $creator:expr) => {
        static _FACTORY_REG: $crate::dev_infrastructure::factory::AutoRegister<$trait_type> = 
            $crate::dev_infrastructure::factory::AutoRegister::new_type::<$impl_type, _>($creator);
    };
}

/// Builder pattern for complex object creation
pub struct FactoryBuilder<T> {
    name: String,
    registry: Arc<FactoryRegistry<T>>,
}

impl<T> FactoryBuilder<T> {
    /// Create a new factory builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            registry: Arc::new(FactoryRegistry::new()),
        }
    }

    /// Add a creator to the builder
    pub fn with_creator<F>(self, name: &str, creator: F) -> Result<Self>
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        self.registry.register(name, creator)?;
        Ok(self)
    }

    /// Build the factory and return the registry
    pub fn build(self) -> Arc<FactoryRegistry<T>> {
        self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test types
    struct TestObject {
        value: i32,
    }

    impl TestObject {
        fn new(value: i32) -> Self {
            Self { value }
        }
    }

    trait TestTrait {
        fn get_value(&self) -> i32;
    }

    impl TestTrait for TestObject {
        fn get_value(&self) -> i32 {
            self.value
        }
    }

    #[test]
    fn test_factory_registry_basic() {
        let registry = FactoryRegistry::new();
        
        // Register a creator
        registry.register("test", || Ok(TestObject::new(42))).unwrap();
        
        // Create an object
        let obj = registry.create("test").unwrap();
        assert_eq!(obj.value, 42);
        
        // Check registry properties
        assert_eq!(registry.creator_count().unwrap(), 1);
        assert!(registry.contains("test").unwrap());
        assert!(!registry.contains("nonexistent").unwrap());
        
        let creators = registry.list_creators().unwrap();
        assert_eq!(creators.len(), 1);
        assert!(creators.contains(&"test".to_string()));
    }

    #[test]
    fn test_factory_registry_type_registration() {
        let registry = FactoryRegistry::new();
        
        // Register by type
        registry.register_type::<TestObject, _>(|| Ok(TestObject::new(99))).unwrap();
        
        // Create by type
        let obj = registry.create_by_type::<TestObject>().unwrap();
        assert_eq!(obj.value, 99);
    }

    #[test]
    fn test_global_factory() {
        let factory = global_factory::<TestObject>();
        
        // Register a creator
        factory.register("global_test", || Ok(TestObject::new(123))).unwrap();
        
        // Create an object
        let obj = factory.create("global_test").unwrap();
        assert_eq!(obj.value, 123);
        
        // Test Factoryable trait
        TestObject::create("global_test").unwrap();
        assert!(TestObject::has_creator("global_test").unwrap());
    }

    #[test]
    fn test_factory_builder() {
        let factory = FactoryBuilder::new("test_factory")
            .with_creator("builder_test", || Ok(TestObject::new(456))).unwrap()
            .build();
        
        let obj = factory.create("builder_test").unwrap();
        assert_eq!(obj.value, 456);
    }

    #[test]
    fn test_factory_errors() {
        let registry = FactoryRegistry::new();
        
        // Test creating non-existent object
        assert!(registry.create("nonexistent").is_err());
        
        // Test duplicate registration
        registry.register("duplicate", || Ok(TestObject::new(1))).unwrap();
        assert!(registry.register("duplicate", || Ok(TestObject::new(2))).is_err());
    }

    #[test]
    fn test_trait_objects() {
        let registry = FactoryRegistry::<Box<dyn TestTrait>>::new();
        
        registry.register("trait_obj", || {
            Ok(Box::new(TestObject::new(789)) as Box<dyn TestTrait>)
        }).unwrap();
        
        let obj = registry.create("trait_obj").unwrap();
        assert_eq!(obj.get_value(), 789);
    }
}