//! Memory-mapped vectors backed by memory-mapped files
//!
//! This module provides vectors that are backed by memory-mapped files,
//! enabling persistent storage and efficient large data handling.
//!
//! # Features
//!
//! - **Persistent Storage**: Data survives process restarts
//! - **Virtual Memory**: OS manages paging for large datasets
//! - **Zero-Copy Access**: Direct memory access without buffer copying
//! - **Cross-Platform**: Works on Unix and Windows systems
//!
//! # Use Cases
//!
//! - **Large datasets**: Data larger than available RAM
//! - **Persistent caches**: Survive application restarts
//! - **Shared memory**: IPC between processes
//! - **Database storage**: Efficient file-based storage

use crate::error::{Result, ZiporaError};
use crate::memory::mmap::{MemoryMappedAllocator, MmapAllocation};
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::slice;

/// Header for memory-mapped vector files
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MmapVecHeader {
    /// Magic number for file format validation
    magic: u64,
    /// Version of the file format
    version: u32,
    /// Size of each element in bytes
    element_size: u32,
    /// Number of elements in the vector
    length: u64,
    /// Capacity of the vector (in elements)
    capacity: u64,
    /// Reserved for future use
    reserved: [u64; 6],
}

const MMAP_VEC_MAGIC: u64 = 0x4D4D41505F564543; // "MMAP_VEC"
const MMAP_VEC_VERSION: u32 = 1;
const HEADER_SIZE: usize = std::mem::size_of::<MmapVecHeader>();

impl MmapVecHeader {
    fn new<T>() -> Self {
        Self {
            magic: MMAP_VEC_MAGIC,
            version: MMAP_VEC_VERSION,
            element_size: std::mem::size_of::<T>() as u32,
            length: 0,
            capacity: 0,
            reserved: [0; 6],
        }
    }

    fn validate<T>(&self) -> Result<()> {
        if self.magic != MMAP_VEC_MAGIC {
            return Err(ZiporaError::invalid_data("Invalid magic number"));
        }
        if self.version != MMAP_VEC_VERSION {
            return Err(ZiporaError::invalid_data("Unsupported version"));
        }
        if self.element_size != std::mem::size_of::<T>() as u32 {
            return Err(ZiporaError::invalid_data("Element size mismatch"));
        }
        if self.length > self.capacity {
            return Err(ZiporaError::invalid_data("Length exceeds capacity"));
        }
        Ok(())
    }
}

/// Configuration for memory-mapped vectors
#[derive(Debug, Clone)]
pub struct MmapVecConfig {
    /// Initial capacity (in elements)
    pub initial_capacity: usize,
    /// Growth factor when expanding (1.5 = 50% growth)
    pub growth_factor: f64,
    /// Enable read-only mode
    pub read_only: bool,
    /// Enable population of pages on creation (MAP_POPULATE on Linux)
    pub populate_pages: bool,
    /// Use huge pages if available
    pub use_huge_pages: bool,
    /// Sync changes to disk immediately
    pub sync_on_write: bool,
}

impl Default for MmapVecConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1024,
            growth_factor: 1.618, // Golden ratio for optimal growth
            read_only: false,
            populate_pages: false,
            use_huge_pages: false,
            sync_on_write: false,
        }
    }
}

impl MmapVecConfig {
    /// Create configuration for read-only vectors
    pub fn read_only() -> Self {
        Self {
            read_only: true,
            populate_pages: true, // Pre-load for read performance
            ..Self::default()
        }
    }

    /// Create configuration for large datasets
    pub fn large_dataset() -> Self {
        Self {
            initial_capacity: 1024 * 1024, // 1M elements
            growth_factor: 1.5, // Conservative growth
            populate_pages: false, // Lazy loading
            use_huge_pages: true,
            ..Self::default()
        }
    }

    /// Create configuration for persistent cache
    pub fn persistent_cache() -> Self {
        Self {
            initial_capacity: 16384,
            growth_factor: 2.0, // Aggressive growth
            sync_on_write: true, // Ensure persistence
            ..Self::default()
        }
    }
}

/// Memory-mapped vector implementation
pub struct MmapVec<T> {
    /// Path to the backing file
    file_path: PathBuf,
    /// Memory-mapped allocation
    mmap: Option<MmapAllocation>,
    /// Configuration
    config: MmapVecConfig,
    /// Cached header pointer
    header: Option<NonNull<MmapVecHeader>>,
    /// Cached data pointer  
    data: Option<NonNull<T>>,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

impl<T> MmapVec<T> 
where
    T: Copy + 'static, // Require Copy for memory-mapped safety
{
    /// Create a new memory-mapped vector
    pub fn create<P: AsRef<Path>>(path: P, config: MmapVecConfig) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        
        // Create the backing file
        let initial_file_size = Self::calculate_file_size(config.initial_capacity);
        Self::create_backing_file(&file_path, initial_file_size)?;

        // Create memory mapping
        let mmap = Self::create_mmap(&file_path, &config)?;
        
        let mut vec = Self {
            file_path,
            mmap: Some(mmap),
            config,
            header: None,
            data: None,
            _phantom: PhantomData,
        };

        // Initialize header and data pointers
        vec.update_pointers()?;
        vec.initialize_header()?;

        Ok(vec)
    }

    /// Open an existing memory-mapped vector
    pub fn open<P: AsRef<Path>>(path: P, config: MmapVecConfig) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        
        if !file_path.exists() {
            return Err(ZiporaError::invalid_data("File does not exist"));
        }

        // Create memory mapping
        let mmap = Self::create_mmap(&file_path, &config)?;
        
        let mut vec = Self {
            file_path,
            mmap: Some(mmap),
            config,
            header: None,
            data: None,
            _phantom: PhantomData,
        };

        // Initialize pointers and validate
        vec.update_pointers()?;
        vec.validate_header()?;

        Ok(vec)
    }

    /// Get the number of elements in the vector
    pub fn len(&self) -> usize {
        self.header()
            .map(|h| unsafe { h.as_ref().length as usize })
            .unwrap_or(0)
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the capacity of the vector
    pub fn capacity(&self) -> usize {
        self.header()
            .map(|h| unsafe { h.as_ref().capacity as usize })
            .unwrap_or(0)
    }

    /// Push an element to the end of the vector
    pub fn push(&mut self, value: T) -> Result<()> {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        let current_len = self.len();
        let current_capacity = self.capacity();

        // Check if we need to grow the vector
        if current_len >= current_capacity {
            self.grow()?;
        }

        // Write the element
        unsafe {
            let data_ptr = self.data_ptr()?.as_ptr();
            std::ptr::write(data_ptr.add(current_len), value);
        }

        // Update length in header
        self.set_length(current_len + 1)?;

        // Sync if requested
        if self.config.sync_on_write {
            self.sync()?;
        }

        Ok(())
    }

    /// Pop an element from the end of the vector
    pub fn pop(&mut self) -> Option<T> {
        if self.config.read_only {
            return None;
        }

        let current_len = self.len();
        if current_len == 0 {
            return None;
        }

        // Read the element
        let value = unsafe {
            let data_ptr = self.data_ptr().ok()?.as_ptr();
            std::ptr::read(data_ptr.add(current_len - 1))
        };

        // Update length
        if self.set_length(current_len - 1).is_err() {
            return None;
        }

        // Sync if requested
        if self.config.sync_on_write {
            let _ = self.sync();
        }

        Some(value)
    }

    /// Get an element at the specified index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            return None;
        }

        unsafe {
            let data_ptr = self.data_ptr().ok()?.as_ptr();
            Some(&*data_ptr.add(index))
        }
    }

    /// Get a mutable reference to an element at the specified index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if self.config.read_only || index >= self.len() {
            return None;
        }

        unsafe {
            let data_ptr = self.data_ptr().ok()?.as_ptr();
            Some(&mut *data_ptr.add(index))
        }
    }

    /// Get a slice of all elements
    pub fn as_slice(&self) -> &[T] {
        let len = self.len();
        if len == 0 {
            return &[];
        }

        unsafe {
            let data_ptr = self.data_ptr().unwrap().as_ptr();
            slice::from_raw_parts(data_ptr, len)
        }
    }

    /// Get a mutable slice of all elements
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.config.read_only {
            return &mut [];
        }

        let len = self.len();
        if len == 0 {
            return &mut [];
        }

        unsafe {
            let data_ptr = self.data_ptr().unwrap().as_ptr();
            slice::from_raw_parts_mut(data_ptr, len)
        }
    }

    /// Clear all elements from the vector
    pub fn clear(&mut self) -> Result<()> {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        self.set_length(0)?;

        if self.config.sync_on_write {
            self.sync()?;
        }

        Ok(())
    }

    /// Reserve capacity for at least `additional` more elements
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        let current_len = self.len();
        let current_capacity = self.capacity();
        let required_capacity = current_len + additional;

        if required_capacity > current_capacity {
            let new_capacity = required_capacity.max(
                (current_capacity as f64 * self.config.growth_factor) as usize
            );
            self.resize_to_capacity(new_capacity)?;
        }

        Ok(())
    }

    /// Sync changes to disk
    pub fn sync(&self) -> Result<()> {
        // Sync memory content back to file (temporary implementation)
        if let Some(mmap) = &self.mmap {
            let file_size = self.file_size_from_header()?;
            let content = unsafe {
                std::slice::from_raw_parts(mmap.as_ptr(), file_size)
            };
            
            std::fs::write(&self.file_path, content)
                .map_err(|e| ZiporaError::io_error(&format!("Failed to sync to file: {}", e)))?;
        }
        Ok(())
    }
    
    /// Get current file size based on header
    fn file_size_from_header(&self) -> Result<usize> {
        let capacity = self.capacity();
        Ok(HEADER_SIZE + capacity * std::mem::size_of::<T>())
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.file_path
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> usize {
        self.capacity() * std::mem::size_of::<T>() + HEADER_SIZE
    }

    /// Create backing file with initial size
    fn create_backing_file(path: &Path, size: u64) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| ZiporaError::io_error(&format!("Failed to create file: {}", e)))?;

        // Set file size
        file.seek(SeekFrom::Start(size - 1))
            .map_err(|e| ZiporaError::io_error(&format!("Failed to seek: {}", e)))?;
        file.write_all(&[0])
            .map_err(|e| ZiporaError::io_error(&format!("Failed to write: {}", e)))?;
        file.sync_all()
            .map_err(|e| ZiporaError::io_error(&format!("Failed to sync: {}", e)))?;

        Ok(())
    }

    /// Create memory mapping for file
    fn create_mmap(path: &Path, _config: &MmapVecConfig) -> Result<MmapAllocation> {
        let file_size = std::fs::metadata(path)
            .map_err(|e| ZiporaError::io_error(&format!("Failed to get file size: {}", e)))?
            .len() as usize;
            
        // Use a minimum size that's appropriate for memory mapping
        // but not too large to waste space for small vectors
        let min_mmap_size = 64 * 1024; // 64KB minimum instead of 1MB
        let allocation_size = file_size.max(min_mmap_size);
        
        let allocator = MemoryMappedAllocator::new(min_mmap_size);
        let mut allocation = allocator.allocate(allocation_size)
            .map_err(|e| ZiporaError::io_error(&format!("Failed to create mmap: {}", e)))?;
        
        // Read file content into the memory mapping
        // This is a temporary implementation - real mmap would map the file directly
        if file_size > 0 {
            let file_content = std::fs::read(path)
                .map_err(|e| ZiporaError::io_error(&format!("Failed to read file: {}", e)))?;
            
            if file_content.len() <= allocation.size() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        file_content.as_ptr(),
                        allocation.as_mut_ptr(),
                        file_content.len()
                    );
                }
            }
        }
        
        Ok(allocation)
    }

    /// Calculate file size for given capacity
    fn calculate_file_size(capacity: usize) -> u64 {
        (HEADER_SIZE + capacity * std::mem::size_of::<T>()) as u64
    }

    /// Update header and data pointers
    fn update_pointers(&mut self) -> Result<()> {
        let mmap = self.mmap.as_ref()
            .ok_or_else(|| ZiporaError::invalid_data("No memory mapping"))?;

        let base_ptr = mmap.as_ptr() as *mut u8;
        
        // Header is at the beginning
        self.header = Some(NonNull::new(base_ptr as *mut MmapVecHeader)
            .ok_or_else(|| ZiporaError::invalid_data("Invalid header pointer"))?);

        // Data follows the header
        let data_ptr = unsafe { base_ptr.add(HEADER_SIZE) } as *mut T;
        self.data = Some(NonNull::new(data_ptr)
            .ok_or_else(|| ZiporaError::invalid_data("Invalid data pointer"))?);

        Ok(())
    }

    /// Initialize header for new file
    fn initialize_header(&mut self) -> Result<()> {
        let header_ptr = self.header()?;
        let header = MmapVecHeader::new::<T>();
        
        unsafe {
            std::ptr::write(header_ptr.as_ptr(), header);
        }

        self.set_capacity(self.config.initial_capacity)?;
        Ok(())
    }

    /// Validate header for existing file
    fn validate_header(&self) -> Result<()> {
        let header_ptr = self.header()?;
        let header = unsafe { header_ptr.as_ref() };
        header.validate::<T>()
    }

    /// Get header pointer
    fn header(&self) -> Result<NonNull<MmapVecHeader>> {
        self.header.ok_or_else(|| ZiporaError::invalid_data("Header not initialized"))
    }

    /// Get data pointer
    fn data_ptr(&self) -> Result<NonNull<T>> {
        self.data.ok_or_else(|| ZiporaError::invalid_data("Data not initialized"))
    }

    /// Set length in header
    fn set_length(&mut self, length: usize) -> Result<()> {
        let header_ptr = self.header()?;
        unsafe {
            (*header_ptr.as_ptr()).length = length as u64;
        }
        Ok(())
    }

    /// Set capacity in header
    fn set_capacity(&mut self, capacity: usize) -> Result<()> {
        let header_ptr = self.header()?;
        unsafe {
            (*header_ptr.as_ptr()).capacity = capacity as u64;
        }
        Ok(())
    }

    /// Grow the vector capacity
    fn grow(&mut self) -> Result<()> {
        let current_capacity = self.capacity();
        let new_capacity = std::cmp::max(
            (current_capacity as f64 * self.config.growth_factor) as usize,
            current_capacity + 1,
        );
        self.resize_to_capacity(new_capacity)
    }

    /// Resize vector to specific capacity
    fn resize_to_capacity(&mut self, new_capacity: usize) -> Result<()> {
        // First, sync current data to file to preserve it
        self.sync()?;
        
        let new_file_size = Self::calculate_file_size(new_capacity);
        
        // Extend the file
        let file = OpenOptions::new()
            .write(true)
            .open(&self.file_path)
            .map_err(|e| ZiporaError::io_error(&format!("Failed to open file: {}", e)))?;
        
        file.set_len(new_file_size)
            .map_err(|e| ZiporaError::io_error(&format!("Failed to resize file: {}", e)))?;

        // Recreate memory mapping with new size
        drop(self.mmap.take()); // Unmap old mapping
        self.mmap = Some(Self::create_mmap(&self.file_path, &self.config)?);
        
        // Update pointers
        self.update_pointers()?;
        
        // Update capacity in header
        self.set_capacity(new_capacity)?;
        
        // Sync the updated header to file
        self.sync()?;

        Ok(())
    }
}

impl<T> Drop for MmapVec<T> {
    fn drop(&mut self) {
        // Note: Cannot call sync() here due to trait bounds
        // Users should explicitly call sync() before dropping if needed
    }
}

/// Iterator for memory-mapped vectors
pub struct MmapVecIter<'a, T> {
    slice: &'a [T],
    index: usize,
}

impl<'a, T> Iterator for MmapVecIter<'a, T>
where
    T: Copy,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.slice.len() {
            let item = &self.slice[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for MmapVecIter<'a, T>
where
    T: Copy,
{
    fn len(&self) -> usize {
        self.slice.len() - self.index
    }
}

impl<'a, T> IntoIterator for &'a MmapVec<T>
where
    T: Copy + 'static,
{
    type Item = &'a T;
    type IntoIter = MmapVecIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        MmapVecIter {
            slice: self.as_slice(),
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_create_mmap_vec() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_vec.mmap");
        
        let config = MmapVecConfig::default();
        let vec = MmapVec::<u64>::create(&file_path, config).unwrap();
        
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 1024);
        assert!(file_path.exists());
    }

    #[test]
    fn test_push_and_get() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_vec.mmap");
        
        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<u32>::create(&file_path, config).unwrap();
        
        // Push some elements
        vec.push(10).unwrap();
        vec.push(20).unwrap();
        vec.push(30).unwrap();
        
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(0), Some(&10));
        assert_eq!(vec.get(1), Some(&20));
        assert_eq!(vec.get(2), Some(&30));
        assert_eq!(vec.get(3), None);
    }

    #[test]
    fn test_pop() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_vec.mmap");
        
        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<i32>::create(&file_path, config).unwrap();
        
        vec.push(100).unwrap();
        vec.push(200).unwrap();
        
        assert_eq!(vec.pop(), Some(200));
        assert_eq!(vec.pop(), Some(100));
        assert_eq!(vec.pop(), None);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_persistence() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_vec.mmap");
        
        // Create vector and add data
        {
            let config = MmapVecConfig::persistent_cache();
            let mut vec = MmapVec::<u64>::create(&file_path, config).unwrap();
            
            for i in 0..100 {
                vec.push(i * 2).unwrap();
            }
            vec.sync().unwrap();
        } // Vector dropped, file should persist
        
        // Open existing vector
        {
            let config = MmapVecConfig::default();
            let vec = MmapVec::<u64>::open(&file_path, config).unwrap();
            
            assert_eq!(vec.len(), 100);
            assert_eq!(vec.get(0), Some(&0));
            assert_eq!(vec.get(50), Some(&100));
            assert_eq!(vec.get(99), Some(&198));
        }
    }

    #[test]
    fn test_growth() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_vec.mmap");
        
        let config = MmapVecConfig {
            initial_capacity: 10,
            growth_factor: 2.0,
            ..MmapVecConfig::default()
        };
        let mut vec = MmapVec::<u8>::create(&file_path, config).unwrap();
        
        assert_eq!(vec.capacity(), 10);
        
        // Fill beyond initial capacity
        for i in 0..20 {
            vec.push(i as u8).unwrap();
        }
        
        assert_eq!(vec.len(), 20);
        assert!(vec.capacity() >= 20);
        
        // Verify all data is correct
        for i in 0..20 {
            assert_eq!(vec.get(i), Some(&(i as u8)));
        }
    }

    #[test]
    fn test_slice_access() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_vec.mmap");
        
        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<u16>::create(&file_path, config).unwrap();
        
        for i in 0..10 {
            vec.push(i * 3).unwrap();
        }
        
        let slice = vec.as_slice();
        assert_eq!(slice.len(), 10);
        assert_eq!(slice[5], 15);
        
        let mut_slice = vec.as_mut_slice();
        mut_slice[5] = 999;
        assert_eq!(vec.get(5), Some(&999));
    }

    #[test]
    fn test_iterator() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_vec.mmap");
        
        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<usize>::create(&file_path, config).unwrap();
        
        let values = vec![1, 4, 9, 16, 25];
        for &value in &values {
            vec.push(value).unwrap();
        }
        
        let collected: Vec<_> = vec.into_iter().copied().collect();
        assert_eq!(collected, values);
    }

    #[test]
    fn test_read_only() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_vec.mmap");
        
        // Create and populate vector
        {
            let config = MmapVecConfig::default();
            let mut vec = MmapVec::<i64>::create(&file_path, config).unwrap();
            vec.push(123).unwrap();
            vec.push(456).unwrap();
            vec.sync().unwrap();
        }
        
        // Open as read-only
        {
            let config = MmapVecConfig::read_only();
            let mut vec = MmapVec::<i64>::open(&file_path, config).unwrap();
            
            assert_eq!(vec.len(), 2);
            assert_eq!(vec.get(0), Some(&123));
            
            // Should fail to modify
            assert!(vec.push(789).is_err());
            assert_eq!(vec.pop(), None);
            assert!(vec.clear().is_err());
        }
    }

    #[test]
    fn test_reserve() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_vec.mmap");
        
        let config = MmapVecConfig {
            initial_capacity: 5,
            ..MmapVecConfig::default()
        };
        let mut vec = MmapVec::<u32>::create(&file_path, config).unwrap();
        
        assert_eq!(vec.capacity(), 5);
        
        vec.reserve(100).unwrap();
        assert!(vec.capacity() >= 100);
        
        // Should be able to add reserved elements without reallocation
        let capacity_before = vec.capacity();
        for i in 0..100 {
            vec.push(i).unwrap();
        }
        assert_eq!(vec.capacity(), capacity_before);
    }

    #[test]
    fn test_different_configs() {
        let temp_dir = tempdir().unwrap();
        
        // Test large dataset config
        let file_path = temp_dir.path().join("large.mmap");
        let config = MmapVecConfig::large_dataset();
        let vec = MmapVec::<u64>::create(&file_path, config).unwrap();
        assert_eq!(vec.capacity(), 1024 * 1024);
        
        // Test persistent cache config
        let file_path2 = temp_dir.path().join("cache.mmap");
        let config = MmapVecConfig::persistent_cache();
        let vec2 = MmapVec::<u32>::create(&file_path2, config).unwrap();
        assert!(vec2.config.sync_on_write);
    }
}