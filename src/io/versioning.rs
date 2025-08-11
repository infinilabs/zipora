//! Advanced version management for backward compatibility
//!
//! This module provides sophisticated versioning capabilities for serialized data,
//! enabling forward and backward compatibility, conditional field serialization,
//! and migration support between different schema versions.

use crate::error::{Result, ZiporaError};
use crate::io::{DataInput, DataOutput};
use crate::io::smart_ptr::SerializableType;
use std::collections::HashMap;

/// Version identifier for serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    major: u16,
    minor: u16,
    patch: u16,
}

impl Version {
    /// Create a new version
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self { major, minor, patch }
    }
    
    /// Create version from a single u32 (packed format)
    /// Format: 0xMMmmpppp where MM=major, mm=minor, pppp=patch
    pub const fn from_u32(version: u32) -> Self {
        Self {
            major: ((version >> 24) & 0xFF) as u16,
            minor: ((version >> 16) & 0xFF) as u16,
            patch: (version & 0xFFFF) as u16,
        }
    }
    
    /// Convert to u32 (packed format)
    /// Format: 0xMMmmpppp where MM=major, mm=minor, pppp=patch
    pub const fn to_u32(self) -> u32 {
        ((self.major as u32) << 24) | ((self.minor as u32) << 16) | (self.patch as u32)
    }
    
    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        // Major version must match, minor and patch can be different
        self.major == other.major && self >= other
    }
    
    /// Check if this version supports a feature introduced in a specific version
    pub fn supports_feature(&self, feature_version: &Version) -> bool {
        self >= feature_version
    }
    
    /// Get the major version
    pub const fn major(&self) -> u16 { self.major }
    
    /// Get the minor version
    pub const fn minor(&self) -> u16 { self.minor }
    
    /// Get the patch version
    pub const fn patch(&self) -> u16 { self.patch }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl SerializableType for Version {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        output.write_u32(self.to_u32())
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let packed = input.read_u32()?;
        Ok(Version::from_u32(packed))
    }
}

/// Conditional serialization based on version requirements
pub struct VersionProxy<T> {
    data: T,
    min_version: Version,
    max_version: Option<Version>,
}

impl<T> VersionProxy<T> {
    /// Create a new version proxy
    pub fn new(data: T, min_version: Version) -> Self {
        Self {
            data,
            min_version,
            max_version: None,
        }
    }
    
    /// Create a version proxy with version range
    pub fn with_range(data: T, min_version: Version, max_version: Version) -> Self {
        Self {
            data,
            min_version,
            max_version: Some(max_version),
        }
    }
    
    /// Check if the data should be serialized for the given version
    pub fn should_serialize(&self, version: &Version) -> bool {
        if version < &self.min_version {
            return false;
        }
        
        if let Some(max_version) = &self.max_version {
            if version > max_version {
                return false;
            }
        }
        
        true
    }
    
    /// Get the inner data
    pub fn data(&self) -> &T {
        &self.data
    }
    
    /// Get the inner data mutably
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
    
    /// Take the inner data
    pub fn into_data(self) -> T {
        self.data
    }
}

impl<T: SerializableType> SerializableType for VersionProxy<T> {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        // This will be handled by VersionManager during serialization
        self.data.serialize(output)
    }
    
    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let data = T::deserialize(input)?;
        Ok(VersionProxy::new(data, Version::new(0, 0, 0)))
    }
}

/// Version manager for handling serialization with version awareness
pub struct VersionManager {
    current_version: Version,
    reading_version: Option<Version>,
    field_versions: HashMap<String, Version>,
}

impl VersionManager {
    /// Create a new version manager
    pub fn new(current_version: Version) -> Self {
        Self {
            current_version,
            reading_version: None,
            field_versions: HashMap::new(),
        }
    }
    
    /// Set the version being read during deserialization
    pub fn set_reading_version(&mut self, version: Version) {
        self.reading_version = Some(version);
    }
    
    /// Get the current version for serialization
    pub fn current_version(&self) -> Version {
        self.current_version
    }
    
    /// Get the version being read during deserialization
    pub fn reading_version(&self) -> Version {
        self.reading_version.unwrap_or(self.current_version)
    }
    
    /// Register a field with its minimum version requirement
    pub fn register_field(&mut self, field_name: impl Into<String>, min_version: Version) {
        self.field_versions.insert(field_name.into(), min_version);
    }
    
    /// Check if a field should be serialized based on current version
    pub fn should_serialize_field(&self, field_name: &str) -> bool {
        if let Some(&min_version) = self.field_versions.get(field_name) {
            self.current_version.supports_feature(&min_version)
        } else {
            true // Unknown fields are serialized by default
        }
    }
    
    /// Check if a field should be deserialized based on reading version
    pub fn should_deserialize_field(&self, field_name: &str) -> bool {
        if let Some(&min_version) = self.field_versions.get(field_name) {
            self.reading_version().supports_feature(&min_version)
        } else {
            true // Unknown fields are deserialized by default
        }
    }
    
    /// Serialize a versioned field conditionally
    pub fn serialize_field<T: SerializableType, O: DataOutput>(
        &self,
        field_name: &str,
        value: &T,
        output: &mut O,
    ) -> Result<()> {
        if self.should_serialize_field(field_name) {
            output.write_u8(1)?; // Field present marker
            value.serialize(output)
        } else {
            output.write_u8(0)?; // Field absent marker
            Ok(())
        }
    }
    
    /// Deserialize a versioned field conditionally
    pub fn deserialize_field<T: SerializableType, I: DataInput>(
        &self,
        field_name: &str,
        input: &mut I,
    ) -> Result<Option<T>> {
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(None), // Field was not serialized
            1 => {
                if self.should_deserialize_field(field_name) {
                    Ok(Some(T::deserialize(input)?))
                } else {
                    // Skip the field data
                    let _ = T::deserialize(input)?;
                    Ok(None)
                }
            }
            _ => Err(ZiporaError::invalid_data("Invalid field marker")),
        }
    }
    
    /// Serialize a version proxy conditionally
    pub fn serialize_proxy<T: SerializableType, O: DataOutput>(
        &self,
        proxy: &VersionProxy<T>,
        output: &mut O,
    ) -> Result<()> {
        if proxy.should_serialize(&self.current_version) {
            output.write_u8(1)?; // Data present marker
            proxy.data().serialize(output)
        } else {
            output.write_u8(0)?; // Data absent marker
            Ok(())
        }
    }
    
    /// Deserialize a version proxy conditionally
    pub fn deserialize_proxy<T: SerializableType, I: DataInput>(
        &self,
        min_version: Version,
        input: &mut I,
    ) -> Result<Option<VersionProxy<T>>> {
        let marker = input.read_u8()?;
        match marker {
            0 => Ok(None), // Data was not serialized
            1 => {
                let data = T::deserialize(input)?;
                Ok(Some(VersionProxy::new(data, min_version)))
            }
            _ => Err(ZiporaError::invalid_data("Invalid proxy marker")),
        }
    }
}

/// Version-aware serialization trait
pub trait VersionedSerialize: Sized {
    /// Get the current version of this type
    fn current_version() -> Version;
    
    /// Get the minimum supported version for reading
    fn min_supported_version() -> Version {
        Version::new(1, 0, 0)
    }
    
    /// Serialize with version information
    fn serialize_versioned<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        let mut manager = VersionManager::new(Self::current_version());
        self.serialize_with_manager(&mut manager, output)
    }
    
    /// Deserialize with version information
    fn deserialize_versioned<I: DataInput>(input: &mut I) -> Result<Self> {
        let version = Version::deserialize(input)?;
        let mut manager = VersionManager::new(Self::current_version());
        manager.set_reading_version(version);
        Self::deserialize_with_manager(&mut manager, input)
    }
    
    /// Serialize using a version manager
    fn serialize_with_manager<O: DataOutput>(
        &self,
        manager: &mut VersionManager,
        output: &mut O,
    ) -> Result<()>;
    
    /// Deserialize using a version manager
    fn deserialize_with_manager<I: DataInput>(
        manager: &mut VersionManager,
        input: &mut I,
    ) -> Result<Self>;
    
    /// Check if a version is supported for reading
    fn supports_version(version: &Version) -> bool {
        version >= &Self::min_supported_version() && 
        version.is_compatible_with(&Self::current_version())
    }
}

/// Migration support for handling version upgrades
pub trait VersionMigration<From, To> {
    /// Migrate data from an older version to a newer version
    fn migrate(from: From) -> Result<To>;
}

/// Version migration registry
pub struct MigrationRegistry {
    migrations: HashMap<(Version, Version), Box<dyn Fn(&[u8]) -> Result<Vec<u8>>>>,
}

impl MigrationRegistry {
    /// Create a new migration registry
    pub fn new() -> Self {
        Self {
            migrations: HashMap::new(),
        }
    }
    
    /// Register a migration between two versions
    pub fn register_migration<F>(&mut self, from: Version, to: Version, migration: F)
    where
        F: Fn(&[u8]) -> Result<Vec<u8>> + 'static,
    {
        self.migrations.insert((from, to), Box::new(migration));
    }
    
    /// Apply migration to upgrade data
    pub fn migrate_data(&self, data: &[u8], from: Version, to: Version) -> Result<Vec<u8>> {
        if from == to {
            return Ok(data.to_vec());
        }
        
        if let Some(migration) = self.migrations.get(&(from, to)) {
            migration(data)
        } else {
            // Try to find a migration path
            self.find_migration_path(data, from, to)
        }
    }
    
    /// Find a migration path through intermediate versions
    fn find_migration_path(&self, data: &[u8], from: Version, to: Version) -> Result<Vec<u8>> {
        // Simple implementation - could be made more sophisticated
        for &(start, end) in self.migrations.keys() {
            if start == from && end <= to {
                let intermediate = self.migrations.get(&(start, end)).unwrap()(data)?;
                if end == to {
                    return Ok(intermediate);
                } else {
                    return self.find_migration_path(&intermediate, end, to);
                }
            }
        }
        
        Err(ZiporaError::invalid_data(
            format!("No migration path from {} to {}", from, to)
        ))
    }
}

impl Default for MigrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for version management
#[derive(Debug, Clone)]
pub struct VersionConfig {
    /// Strict version checking
    pub strict_version_checking: bool,
    /// Allow forward compatibility (reading newer versions)
    pub allow_forward_compatibility: bool,
    /// Maximum version skew allowed
    pub max_version_skew: u16,
    /// Enable automatic migrations
    pub enable_migrations: bool,
}

impl VersionConfig {
    /// Create a new version configuration
    pub fn new() -> Self {
        Self {
            strict_version_checking: true,
            allow_forward_compatibility: false,
            max_version_skew: 1,
            enable_migrations: true,
        }
    }
    
    /// Configuration for strict compatibility
    pub fn strict() -> Self {
        Self {
            strict_version_checking: true,
            allow_forward_compatibility: false,
            max_version_skew: 0,
            enable_migrations: false,
        }
    }
    
    /// Configuration for flexible compatibility
    pub fn flexible() -> Self {
        Self {
            strict_version_checking: false,
            allow_forward_compatibility: true,
            max_version_skew: 10,
            enable_migrations: true,
        }
    }
    
    /// Configuration optimized for development
    pub fn development() -> Self {
        Self {
            strict_version_checking: false,
            allow_forward_compatibility: true,
            max_version_skew: 100,
            enable_migrations: true,
        }
    }
}

impl Default for VersionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level versioned serialization utilities
pub struct VersionedSerializer {
    config: VersionConfig,
    migration_registry: MigrationRegistry,
}

impl VersionedSerializer {
    /// Create a new versioned serializer
    pub fn new(config: VersionConfig) -> Self {
        Self {
            config,
            migration_registry: MigrationRegistry::new(),
        }
    }
    
    /// Create a serializer with default configuration
    pub fn default() -> Self {
        Self::new(VersionConfig::default())
    }
    
    /// Register a migration
    pub fn register_migration<F>(&mut self, from: Version, to: Version, migration: F)
    where
        F: Fn(&[u8]) -> Result<Vec<u8>> + 'static,
    {
        self.migration_registry.register_migration(from, to, migration);
    }
    
    /// Serialize a versioned object to bytes
    pub fn serialize_to_bytes<T: VersionedSerialize>(&self, value: &T) -> Result<Vec<u8>> {
        let mut output = crate::io::VecDataOutput::new();
        
        // Write version header
        T::current_version().serialize(&mut output)?;
        
        // Serialize the object
        value.serialize_versioned(&mut output)?;
        
        Ok(output.into_vec())
    }
    
    /// Deserialize a versioned object from bytes
    pub fn deserialize_from_bytes<T: VersionedSerialize>(&self, bytes: &[u8]) -> Result<T> {
        let mut input = crate::io::SliceDataInput::new(bytes);
        
        // Read version header
        let stored_version = Version::deserialize(&mut input)?;
        let current_version = T::current_version();
        
        // Check version compatibility
        if self.config.strict_version_checking {
            if !T::supports_version(&stored_version) {
                return Err(ZiporaError::invalid_data(
                    format!("Unsupported version: {} (current: {})", stored_version, current_version)
                ));
            }
        }
        
        // Check version skew
        let version_diff = if stored_version > current_version {
            stored_version.minor() - current_version.minor()
        } else {
            current_version.minor() - stored_version.minor()
        };
        
        if version_diff > self.config.max_version_skew {
            return Err(ZiporaError::invalid_data(
                format!("Version skew too large: {}", version_diff)
            ));
        }
        
        // Apply migration if needed
        let data = if stored_version != current_version && self.config.enable_migrations {
            let remaining_bytes = if let Some(pos) = input.position() {
            &bytes[pos as usize..]
        } else {
            return Err(ZiporaError::invalid_data("Cannot get input position for migration"));
        };
            let migrated = self.migration_registry.migrate_data(
                remaining_bytes, 
                stored_version, 
                current_version
            )?;
            let mut migrated_input = crate::io::SliceDataInput::new(&migrated);
            return T::deserialize_with_manager(
                &mut VersionManager::new(current_version),
                &mut migrated_input
            );
        } else {
            // Deserialize with version awareness
            let mut manager = VersionManager::new(current_version);
            manager.set_reading_version(stored_version);
            T::deserialize_with_manager(&mut manager, &mut input)?
        };
        
        Ok(data)
    }
}

/// Convenience macros for version management
#[macro_export]
macro_rules! versioned_field {
    ($manager:expr, $field_name:expr, $value:expr, $output:expr) => {
        $manager.serialize_field($field_name, $value, $output)?;
    };
}

#[macro_export]
macro_rules! versioned_field_with_default {
    ($manager:expr, $field_name:expr, $input:expr, $default:expr) => {
        $manager.deserialize_field::<_, _>($field_name, $input)?
            .unwrap_or($default)
    };
}

#[macro_export]
macro_rules! since_version {
    ($version:expr, $data:expr) => {
        VersionProxy::new($data, $version)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{SliceDataInput, VecDataOutput};

    #[test]
    fn test_version_creation_and_comparison() {
        let v1 = Version::new(1, 2, 3);
        let v2 = Version::new(1, 2, 4);
        let v3 = Version::new(2, 0, 0);
        
        assert!(v2 > v1);
        assert!(v3 > v2);
        assert!(v1.is_compatible_with(&v1));
        assert!(v2.is_compatible_with(&v1));
        assert!(!v3.is_compatible_with(&v1));
    }

    #[test]
    fn test_version_serialization() {
        let version = Version::new(1, 2, 3);
        let mut output = VecDataOutput::new();
        
        version.serialize(&mut output).unwrap();
        let bytes = output.into_vec();
        
        let mut input = SliceDataInput::new(&bytes);
        let deserialized = Version::deserialize(&mut input).unwrap();
        
        assert_eq!(deserialized, version);
    }

    #[test]
    fn test_version_packed_format() {
        let version = Version::new(1, 2, 3);
        let packed = version.to_u32();
        let unpacked = Version::from_u32(packed);
        
        assert_eq!(unpacked, version);
        assert_eq!(packed, 0x01020003);
    }

    #[test]
    fn test_version_proxy() {
        let proxy = VersionProxy::new(42u32, Version::new(1, 1, 0));
        
        assert!(proxy.should_serialize(&Version::new(1, 1, 0)));
        assert!(proxy.should_serialize(&Version::new(1, 2, 0)));
        assert!(!proxy.should_serialize(&Version::new(1, 0, 0)));
        
        let range_proxy = VersionProxy::with_range(
            42u32, 
            Version::new(1, 1, 0), 
            Version::new(1, 3, 0)
        );
        
        assert!(range_proxy.should_serialize(&Version::new(1, 2, 0)));
        assert!(!range_proxy.should_serialize(&Version::new(1, 4, 0)));
    }

    #[test]
    fn test_version_manager() {
        let mut manager = VersionManager::new(Version::new(1, 2, 0));
        manager.register_field("new_field", Version::new(1, 1, 0));
        manager.register_field("newer_field", Version::new(1, 3, 0));
        
        assert!(manager.should_serialize_field("new_field"));
        assert!(!manager.should_serialize_field("newer_field"));
        
        manager.set_reading_version(Version::new(1, 3, 0));
        assert!(manager.should_deserialize_field("newer_field"));
    }

    #[test]
    fn test_version_manager_field_serialization() {
        let manager = VersionManager::new(Version::new(1, 2, 0));
        let mut output = VecDataOutput::new();
        
        let value = 42u32;
        manager.serialize_field("test_field", &value, &mut output).unwrap();
        
        let bytes = output.into_vec();
        let mut input = SliceDataInput::new(&bytes);
        
        let deserialized: Option<u32> = manager.deserialize_field("test_field", &mut input).unwrap();
        assert_eq!(deserialized, Some(42));
    }

    #[test]
    fn test_migration_registry() {
        let mut registry = MigrationRegistry::new();
        
        // Register a simple migration that doubles all u32 values
        registry.register_migration(
            Version::new(1, 0, 0),
            Version::new(1, 1, 0),
            |data| {
                let mut input = crate::io::SliceDataInput::new(data);
                let value = input.read_u32()?;
                
                let mut output = crate::io::VecDataOutput::new();
                output.write_u32(value * 2)?;
                Ok(output.into_vec())
            }
        );
        
        let old_data = {
            let mut output = crate::io::VecDataOutput::new();
            output.write_u32(21).unwrap();
            output.into_vec()
        };
        
        let migrated = registry.migrate_data(
            &old_data,
            Version::new(1, 0, 0),
            Version::new(1, 1, 0)
        ).unwrap();
        
        let mut input = crate::io::SliceDataInput::new(&migrated);
        let value = input.read_u32().unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_version_config() {
        let strict_config = VersionConfig::strict();
        assert!(strict_config.strict_version_checking);
        assert!(!strict_config.allow_forward_compatibility);
        assert_eq!(strict_config.max_version_skew, 0);
        
        let flexible_config = VersionConfig::flexible();
        assert!(!flexible_config.strict_version_checking);
        assert!(flexible_config.allow_forward_compatibility);
        assert_eq!(flexible_config.max_version_skew, 10);
    }

    #[test]
    fn test_versioned_serializer() {
        let serializer = VersionedSerializer::default();
        
        // Test basic serialization/deserialization
        #[derive(Debug, PartialEq)]
        struct TestStruct {
            value: u32,
        }
        
        impl VersionedSerialize for TestStruct {
            fn current_version() -> Version {
                Version::new(1, 0, 0)
            }
            
            fn serialize_with_manager<O: DataOutput>(
                &self,
                _manager: &mut VersionManager,
                output: &mut O,
            ) -> Result<()> {
                output.write_u32(self.value)
            }
            
            fn deserialize_with_manager<I: DataInput>(
                _manager: &mut VersionManager,
                input: &mut I,
            ) -> Result<Self> {
                Ok(Self {
                    value: input.read_u32()?,
                })
            }
        }
        
        let test_struct = TestStruct { value: 42 };
        let bytes = serializer.serialize_to_bytes(&test_struct).unwrap();
        let deserialized: TestStruct = serializer.deserialize_from_bytes(&bytes).unwrap();
        
        assert_eq!(deserialized, test_struct);
    }

    #[test]
    fn test_version_display() {
        let version = Version::new(1, 2, 3);
        assert_eq!(format!("{}", version), "1.2.3");
    }

    #[test]
    fn test_feature_support() {
        let current = Version::new(1, 5, 0);
        let feature_v1 = Version::new(1, 2, 0);
        let feature_v2 = Version::new(1, 8, 0);
        
        assert!(current.supports_feature(&feature_v1));
        assert!(!current.supports_feature(&feature_v2));
    }
}