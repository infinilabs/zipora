//! Advanced version management for backward compatibility
//!
//! This module provides sophisticated versioning capabilities for serialized data,
//! enabling forward and backward compatibility, conditional field serialization,
//! and migration support between different schema versions.

use crate::error::{Result, ZiporaError};
use crate::io::smart_ptr::SerializableType;
use crate::io::{DataInput, DataOutput};
use std::collections::HashMap;

/// Version identifier for serialization formats.
///
/// # Wire Format ([`SerializableType`])
/// - When `major < 255 && minor < 255`, [`SerializableType::serialize`] writes
///   the 4-byte packed `u32` `0xMMmmpppp` ([`Version::to_u32`]).
/// - When `major >= 255 || minor >= 255`, [`SerializableType::serialize`] writes
///   the 4-byte sentinel `0xFFFF_pppp` followed by two `u16` values (`major`,
///   `minor`) so the full `u16` range of all three fields round-trips losslessly.
/// - **Wire compatibility note**: because `0xFFFF_pppp` is reserved as the
///   extended-header sentinel, a 4-byte pre-C9 payload encoding `255.255.patch`
///   (or any `major >= 255, minor >= 255` saturated by `to_u32()`) without the
///   trailing 4-byte `(major, minor)` extension is incompatible with this
///   format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    major: u16,
    minor: u16,
    patch: u16,
}

impl Version {
    /// Create a new version
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
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
    /// Format: 0xMMmmpppp where MM=major, mm=minor, pppp=patch.
    ///
    /// Limitation: the 32-bit packed representation allocates 8 bits each to
    /// `major` and `minor`, so values above `255` saturate to `255` here
    /// rather than overflowing into adjacent fields. [`SerializableType`] for
    /// [`Version`] preserves the full `u16` range of all three fields using the
    /// `0xFFFF_pppp` extended wire format documented on [`Version`].
    pub const fn to_u32(self) -> u32 {
        let major = if self.major > 0xFF { 0xFF } else { self.major };
        let minor = if self.minor > 0xFF { 0xFF } else { self.minor };
        ((major as u32) << 24) | ((minor as u32) << 16) | (self.patch as u32)
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
    pub const fn major(&self) -> u16 {
        self.major
    }

    /// Get the minor version
    pub const fn minor(&self) -> u16 {
        self.minor
    }

    /// Get the patch version
    pub const fn patch(&self) -> u16 {
        self.patch
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl SerializableType for Version {
    fn serialize<O: DataOutput>(&self, output: &mut O) -> Result<()> {
        if self.major < 0xFF && self.minor < 0xFF {
            output.write_u32(self.to_u32())
        } else {
            let sentinel = 0xFFFF_0000u32 | (self.patch as u32);
            output.write_u32(sentinel)?;
            output.write_u16(self.major)?;
            output.write_u16(self.minor)
        }
    }

    fn deserialize<I: DataInput>(input: &mut I) -> Result<Self> {
        let packed = input.read_u32()?;
        if (packed >> 16) == 0xFFFF {
            let major = input.read_u16()?;
            let minor = input.read_u16()?;
            let patch = (packed & 0xFFFF) as u16;
            Ok(Version::new(major, minor, patch))
        } else {
            Ok(Version::from_u32(packed))
        }
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

        if let Some(max_version) = &self.max_version
            && version > max_version
        {
            return false;
        }

        true
    }

    /// Get the inner data
    #[inline]
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
        Self::current_version().serialize(output)?;
        let mut manager = VersionManager::new(Self::current_version());
        self.serialize_with_manager(&mut manager, output)
    }

    /// Deserialize with version information
    fn deserialize_versioned<I: DataInput>(input: &mut I) -> Result<Self> {
        let version = Version::deserialize(input)?;
        if !Self::supports_version(&version) {
            return Err(ZiporaError::invalid_data(format!(
                "Unsupported version: {} (current: {}, min supported: {})",
                version,
                Self::current_version(),
                Self::min_supported_version()
            )));
        }
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
        version >= &Self::min_supported_version()
            && Self::current_version().is_compatible_with(version)
    }
}
pub type MigrationFn = Box<dyn Fn(&[u8]) -> Result<Vec<u8>>>;

/// Version migration registry
pub struct MigrationRegistry {
    migrations: HashMap<(Version, Version), MigrationFn>,
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

    /// Return the shortest deterministic sequence of migration steps from `from` to `to`.
    fn shortest_path(&self, from: Version, to: Version) -> Option<Vec<(Version, Version)>> {
        if from == to {
            return Some(Vec::new());
        }
        if self.migrations.contains_key(&(from, to)) {
            return Some(vec![(from, to)]);
        }

        let mut queue = std::collections::VecDeque::new();
        let mut prev: HashMap<Version, Version> = HashMap::new();
        let mut visited = std::collections::HashSet::new();
        visited.insert(from);
        queue.push_back(from);

        while let Some(curr) = queue.pop_front() {
            let mut next_versions: Vec<Version> = self
                .migrations
                .keys()
                .filter_map(|&(start, end)| {
                    if start == curr && !visited.contains(&end) {
                        Some(end)
                    } else {
                        None
                    }
                })
                .collect();
            next_versions.sort_unstable();

            for next in next_versions {
                prev.insert(next, curr);
                if next == to {
                    let mut steps = Vec::new();
                    let mut node = to;
                    while let Some(&parent) = prev.get(&node) {
                        steps.push((parent, node));
                        if parent == from {
                            break;
                        }
                        node = parent;
                    }
                    steps.reverse();
                    return Some(steps);
                }
                visited.insert(next);
                queue.push_back(next);
            }
        }

        None
    }

    fn has_migration_path(&self, from: Version, to: Version) -> bool {
        self.shortest_path(from, to).is_some()
    }

    /// Apply migration to upgrade data
    pub fn migrate_data(&self, data: &[u8], from: Version, to: Version) -> Result<Vec<u8>> {
        if from == to {
            return Ok(data.to_vec());
        }

        let steps = self.shortest_path(from, to).ok_or_else(|| {
            ZiporaError::invalid_data(format!("No migration path from {} to {}", from, to))
        })?;

        let mut current = data.to_vec();
        for (start, end) in steps {
            let migrate_fn = self.migrations.get(&(start, end)).ok_or_else(|| {
                ZiporaError::invalid_data(format!("No migration path from {} to {}", start, end))
            })?;
            current = migrate_fn(&current)?;
        }
        Ok(current)
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

impl Default for VersionedSerializer {
    fn default() -> Self {
        Self::new(VersionConfig::default())
    }
}

impl VersionedSerializer {
    /// Create a new versioned serializer
    pub fn new(config: VersionConfig) -> Self {
        Self {
            config,
            migration_registry: MigrationRegistry::new(),
        }
    }

    /// Register a migration
    pub fn register_migration<F>(&mut self, from: Version, to: Version, migration: F)
    where
        F: Fn(&[u8]) -> Result<Vec<u8>> + 'static,
    {
        self.migration_registry
            .register_migration(from, to, migration);
    }

    /// Serialize a versioned object to bytes
    pub fn serialize_to_bytes<T: VersionedSerialize>(&self, value: &T) -> Result<Vec<u8>> {
        let mut output = crate::io::VecDataOutput::new();
        value.serialize_versioned(&mut output)?;
        Ok(output.into_vec())
    }

    /// Deserialize a versioned object from bytes
    pub fn deserialize_from_bytes<T: VersionedSerialize>(&self, bytes: &[u8]) -> Result<T> {
        let mut input = crate::io::SliceDataInput::new(bytes);

        // Read version header
        let stored_version = Version::deserialize(&mut input)?;
        let current_version = T::current_version();
        let can_migrate = stored_version != current_version
            && self.config.enable_migrations
            && self
                .migration_registry
                .has_migration_path(stored_version, current_version);

        if !can_migrate {
            if self.config.strict_version_checking && !T::supports_version(&stored_version) {
                return Err(ZiporaError::invalid_data(format!(
                    "Unsupported version: {} (current: {})",
                    stored_version, current_version
                )));
            }

            if stored_version < T::min_supported_version() {
                return Err(ZiporaError::invalid_data(format!(
                    "Version {} is below minimum supported version {}",
                    stored_version,
                    T::min_supported_version()
                )));
            }

            if stored_version.major() != current_version.major() {
                return Err(ZiporaError::invalid_data(format!(
                    "Incompatible major version: {} (current: {})",
                    stored_version, current_version
                )));
            }

            if stored_version > current_version && !self.config.allow_forward_compatibility {
                return Err(ZiporaError::invalid_data(format!(
                    "Forward version {} not allowed (current: {})",
                    stored_version, current_version
                )));
            }
        }

        if stored_version.major() == current_version.major() {
            let version_diff = stored_version.minor().abs_diff(current_version.minor());
            if version_diff > self.config.max_version_skew {
                return Err(ZiporaError::invalid_data(format!(
                    "Version skew too large: {}",
                    version_diff
                )));
            }
        }

        // Apply migration if needed
        let data = if can_migrate {
            let remaining_bytes = if let Some(pos) = input.position() {
                &bytes[pos as usize..]
            } else {
                return Err(ZiporaError::invalid_data(
                    "Cannot get input position for migration",
                ));
            };
            let migrated = self.migration_registry.migrate_data(
                remaining_bytes,
                stored_version,
                current_version,
            )?;
            let mut migrated_input = crate::io::SliceDataInput::new(&migrated);
            return T::deserialize_with_manager(
                &mut VersionManager::new(current_version),
                &mut migrated_input,
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
        $manager
            .deserialize_field::<_, _>($field_name, $input)?
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

        let range_proxy =
            VersionProxy::with_range(42u32, Version::new(1, 1, 0), Version::new(1, 3, 0));

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
        manager
            .serialize_field("test_field", &value, &mut output)
            .unwrap();

        let bytes = output.into_vec();
        let mut input = SliceDataInput::new(&bytes);

        let deserialized: Option<u32> =
            manager.deserialize_field("test_field", &mut input).unwrap();
        assert_eq!(deserialized, Some(42));
    }

    #[test]
    fn test_migration_registry() {
        let mut registry = MigrationRegistry::new();

        // Register a simple migration that doubles all u32 values
        registry.register_migration(Version::new(1, 0, 0), Version::new(1, 1, 0), |data| {
            let mut input = crate::io::SliceDataInput::new(data);
            let value = input.read_u32()?;

            let mut output = crate::io::VecDataOutput::new();
            output.write_u32(value * 2)?;
            Ok(output.into_vec())
        });

        let old_data = {
            let mut output = crate::io::VecDataOutput::new();
            output.write_u32(21).unwrap();
            output.into_vec()
        };

        let migrated = registry
            .migrate_data(&old_data, Version::new(1, 0, 0), Version::new(1, 1, 0))
            .unwrap();

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

    #[test]
    fn test_c9_vs1_supports_older_same_major_and_rejects_newer_without_forward_compat() {
        #[derive(Debug, PartialEq)]
        struct T11(u32);

        impl VersionedSerialize for T11 {
            fn current_version() -> Version {
                Version::new(1, 1, 0)
            }

            fn serialize_with_manager<O: DataOutput>(
                &self,
                _manager: &mut VersionManager,
                output: &mut O,
            ) -> Result<()> {
                output.write_u32(self.0)
            }

            fn deserialize_with_manager<I: DataInput>(
                _manager: &mut VersionManager,
                input: &mut I,
            ) -> Result<Self> {
                Ok(Self(input.read_u32()?))
            }
        }

        let mut ser = VersionedSerializer::default();
        ser.register_migration(Version::new(1, 0, 0), Version::new(1, 1, 0), |data| {
            let mut inp = SliceDataInput::new(data);
            let v = inp.read_u32()?;
            let mut out = VecDataOutput::new();
            out.write_u32(v + 10)?;
            Ok(out.into_vec())
        });

        // Older same-major 1.0.0 data (0x0100_0000 LE = [0, 0, 0, 1]) + payload 7u32 LE.
        let v10_bytes = [0u8, 0, 0, 1, 7, 0, 0, 0];
        let migrated = ser
            .deserialize_from_bytes::<T11>(&v10_bytes)
            .expect("1.0.0 data must be accepted and migrated to 1.1.0");
        assert_eq!(migrated, T11(17));

        // Newer 1.2.0 data (0x0102_0000 LE = [0, 0, 2, 1]) must be rejected when forward compat is false.
        let v12_bytes = [0u8, 0, 2, 1, 7, 0, 0, 0];
        assert!(
            ser.deserialize_from_bytes::<T11>(&v12_bytes).is_err(),
            "newer 1.2.0 data must be rejected when allow_forward_compatibility is false"
        );
    }

    #[test]
    fn test_c9_vs2_skew_check_rejects_underflow_and_major_mismatch() {
        #[derive(Debug, PartialEq)]
        struct T10(u32);

        impl VersionedSerialize for T10 {
            fn current_version() -> Version {
                Version::new(1, 0, 0)
            }

            fn serialize_with_manager<O: DataOutput>(
                &self,
                _manager: &mut VersionManager,
                output: &mut O,
            ) -> Result<()> {
                output.write_u32(self.0)
            }

            fn deserialize_with_manager<I: DataInput>(
                _manager: &mut VersionManager,
                input: &mut I,
            ) -> Result<Self> {
                Ok(Self(input.read_u32()?))
            }
        }

        let ser = VersionedSerializer::new(VersionConfig::flexible());

        // Header 0.9.0 ([0, 0, 9, 0]) vs current 1.0.0: previously `0 - 9` panicked in debug.
        let res_09 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ser.deserialize_from_bytes::<T10>(&[0, 0, 9, 0, 7, 0, 0, 0])
        }));
        assert!(
            matches!(res_09, Ok(Err(_))),
            "0.9.0 header must return Err without panicking: {res_09:?}"
        );

        // Header 9.0.0 ([0, 0, 0, 9]) vs current 1.0.0: different major version must be rejected.
        assert!(
            ser.deserialize_from_bytes::<T10>(&[0, 0, 0, 9, 7, 0, 0, 0])
                .is_err(),
            "9.0.0 header must be rejected when current is 1.0.0"
        );
    }

    #[test]
    fn test_c9_vs3_migration_path_handles_cycles_and_dead_ends() {
        use std::cell::Cell;
        use std::rc::Rc;

        // 1. Cycle 1.0 -> 2.0 -> 1.0 when searching for unreachable 3.0 must terminate cleanly.
        let calls = Rc::new(Cell::new(0usize));
        let mut cyc_reg = MigrationRegistry::new();
        let c1 = calls.clone();
        cyc_reg.register_migration(Version::new(1, 0, 0), Version::new(2, 0, 0), move |d| {
            let n = c1.get() + 1;
            c1.set(n);
            if n > 50 {
                return Err(ZiporaError::invalid_data("infinite migration cycle"));
            }
            Ok(d.to_vec())
        });
        let c2 = calls.clone();
        cyc_reg.register_migration(Version::new(2, 0, 0), Version::new(1, 0, 0), move |d| {
            let n = c2.get() + 1;
            c2.set(n);
            if n > 50 {
                return Err(ZiporaError::invalid_data("infinite migration cycle"));
            }
            Ok(d.to_vec())
        });
        assert!(
            cyc_reg
                .migrate_data(&[1, 2, 3], Version::new(1, 0, 0), Version::new(3, 0, 0))
                .is_err()
        );
        assert!(
            calls.get() <= 2,
            "cycle detection must not repeatedly run migrations (ran {} times)",
            calls.get()
        );

        // 2. Dead-end branch 1.0 -> 1.5 alongside valid path 1.0 -> 2.0 -> 3.0 must always succeed.
        for _ in 0..32 {
            let mut reg = MigrationRegistry::new();
            reg.register_migration(Version::new(1, 0, 0), Version::new(1, 5, 0), |d| {
                Ok(d.to_vec())
            });
            reg.register_migration(Version::new(1, 0, 0), Version::new(2, 0, 0), |d| {
                let mut v = d.to_vec();
                v.push(2);
                Ok(v)
            });
            reg.register_migration(Version::new(2, 0, 0), Version::new(3, 0, 0), |d| {
                let mut v = d.to_vec();
                v.push(3);
                Ok(v)
            });
            let out = reg
                .migrate_data(&[1], Version::new(1, 0, 0), Version::new(3, 0, 0))
                .expect("valid migration path 1.0 -> 2.0 -> 3.0 must be found");
            assert_eq!(out, vec![1, 2, 3]);
        }
    }

    #[test]
    fn test_c9_vs4_large_major_and_minor_versions_round_trip() {
        for v in [
            Version::new(2, 256, 0),
            Version::new(256, 0, 0),
            Version::new(255, 255, 65535),
            Version::new(u16::MAX, u16::MAX, u16::MAX),
        ] {
            let mut out = VecDataOutput::new();
            v.serialize(&mut out).unwrap();
            let bytes = out.into_vec();
            let mut inp = SliceDataInput::new(&bytes);
            let back = Version::deserialize(&mut inp).unwrap();
            assert_eq!(back, v);
        }

        // Packed u32 saturates minor at 255 rather than spilling into major.
        let packed = Version::new(2, 256, 0).to_u32();
        assert_eq!(Version::from_u32(packed).major(), 2);
    }

    #[test]
    fn test_c9_vs5_serialize_versioned_and_deserialize_versioned_round_trip_and_validate() {
        #[derive(Debug, PartialEq)]
        struct VItem {
            val: u32,
        }

        impl VersionedSerialize for VItem {
            fn current_version() -> Version {
                Version::new(1, 1, 0)
            }

            fn serialize_with_manager<O: DataOutput>(
                &self,
                _manager: &mut VersionManager,
                output: &mut O,
            ) -> Result<()> {
                output.write_u32(self.val)
            }

            fn deserialize_with_manager<I: DataInput>(
                _manager: &mut VersionManager,
                input: &mut I,
            ) -> Result<Self> {
                Ok(Self {
                    val: input.read_u32()?,
                })
            }
        }

        let item = VItem { val: 42 };
        let mut out = VecDataOutput::new();
        item.serialize_versioned(&mut out).unwrap();
        let bytes = out.into_vec();

        let mut inp = SliceDataInput::new(&bytes);
        let back = VItem::deserialize_versioned(&mut inp).unwrap();
        assert_eq!(back, item);

        // Unsupported major version 2.0.0 must be rejected by deserialize_versioned.
        let mut bad_out = VecDataOutput::new();
        Version::new(2, 0, 0).serialize(&mut bad_out).unwrap();
        bad_out.write_u32(42).unwrap();
        let bad_bytes = bad_out.into_vec();
        let mut bad_inp = SliceDataInput::new(&bad_bytes);
        assert!(
            VItem::deserialize_versioned(&mut bad_inp).is_err(),
            "deserialize_versioned must reject unsupported version 2.0.0"
        );
    }
}
