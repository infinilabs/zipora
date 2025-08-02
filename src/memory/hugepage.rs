//! Hugepage support for improved memory performance
//!
//! This module provides support for allocating and managing hugepages on Linux systems.
//! Hugepages can significantly improve performance for memory-intensive applications
//! by reducing TLB misses and improving cache locality.

#[cfg(target_os = "linux")]
use std::fs;
#[cfg(target_os = "linux")]
use std::ptr::NonNull;
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(target_os = "linux")]
use std::sync::Mutex;

use crate::error::{Result, ToplingError};

/// Standard hugepage size on x86_64 Linux (2MB)
pub const HUGEPAGE_SIZE_2MB: usize = 2 * 1024 * 1024;

/// Large hugepage size on x86_64 Linux (1GB)
pub const HUGEPAGE_SIZE_1GB: usize = 1024 * 1024 * 1024;

/// Information about system hugepage availability
#[derive(Debug, Clone)]
pub struct HugePageInfo {
    /// Size of hugepages in bytes
    pub page_size: usize,
    /// Total number of hugepages configured
    pub total_pages: usize,
    /// Number of hugepages currently free
    pub free_pages: usize,
    /// Number of hugepages reserved
    pub reserved_pages: usize,
}

#[cfg(target_os = "linux")]
static HUGEPAGE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(target_os = "linux")]
static HUGEPAGE_ALLOCATIONS: Mutex<Vec<HugePageAllocation>> = Mutex::new(Vec::new());

#[cfg(target_os = "linux")]
#[allow(dead_code)]
struct HugePageAllocation {
    ptr: *mut u8,
    size: usize,
    page_size: usize,
}

#[cfg(target_os = "linux")]
unsafe impl Send for HugePageAllocation {}
#[cfg(target_os = "linux")]
unsafe impl Sync for HugePageAllocation {}

/// A hugepage-backed memory allocation
pub struct HugePage {
    #[cfg(target_os = "linux")]
    ptr: NonNull<u8>,
    #[cfg(target_os = "linux")]
    size: usize,
    #[cfg(target_os = "linux")]
    page_size: usize,
    #[cfg(not(target_os = "linux"))]
    _phantom: std::marker::PhantomData<u8>,
}

impl HugePage {
    /// Allocate memory using hugepages
    pub fn new(size: usize, page_size: usize) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::allocate_linux(size, page_size)
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = (size, page_size);
            Err(ToplingError::not_supported(
                "hugepages only supported on Linux",
            ))
        }
    }

    /// Allocate memory using 2MB hugepages
    pub fn new_2mb(size: usize) -> Result<Self> {
        Self::new(size, HUGEPAGE_SIZE_2MB)
    }

    /// Allocate memory using 1GB hugepages
    pub fn new_1gb(size: usize) -> Result<Self> {
        Self::new(size, HUGEPAGE_SIZE_1GB)
    }

    /// Get the memory as a slice
    pub fn as_slice(&self) -> &[u8] {
        #[cfg(target_os = "linux")]
        {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
        }

        #[cfg(not(target_os = "linux"))]
        {
            &[]
        }
    }

    /// Get the memory as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        #[cfg(target_os = "linux")]
        {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
        }

        #[cfg(not(target_os = "linux"))]
        {
            &mut []
        }
    }

    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        #[cfg(target_os = "linux")]
        {
            self.size
        }

        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }

    /// Get the hugepage size used for this allocation
    pub fn page_size(&self) -> usize {
        #[cfg(target_os = "linux")]
        {
            self.page_size
        }

        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }

    #[cfg(target_os = "linux")]
    fn allocate_linux(size: usize, page_size: usize) -> Result<Self> {
        if size == 0 {
            return Err(ToplingError::invalid_data("allocation size cannot be zero"));
        }

        if page_size != HUGEPAGE_SIZE_2MB && page_size != HUGEPAGE_SIZE_1GB {
            return Err(ToplingError::invalid_data("invalid hugepage size"));
        }

        // Round up size to multiple of page size
        let aligned_size = (size + page_size - 1) & !(page_size - 1);

        // Try to allocate using mmap with MAP_HUGETLB
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                aligned_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(ToplingError::out_of_memory(aligned_size));
        }

        let ptr = unsafe { NonNull::new_unchecked(ptr as *mut u8) };

        // Track the allocation
        let allocation = HugePageAllocation {
            ptr: ptr.as_ptr(),
            size: aligned_size,
            page_size,
        };

        HUGEPAGE_ALLOCATIONS.lock().unwrap().push(allocation);
        HUGEPAGE_COUNT.fetch_add(aligned_size / page_size, Ordering::Relaxed);

        Ok(Self {
            ptr,
            size,
            page_size,
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for HugePage {
    fn drop(&mut self) {
        // Unmap the memory
        let aligned_size = (self.size + self.page_size - 1) & !(self.page_size - 1);

        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, aligned_size);
        }

        // Remove from tracking
        let mut allocations = HUGEPAGE_ALLOCATIONS.lock().unwrap();
        allocations.retain(|alloc| alloc.ptr != self.ptr.as_ptr());

        HUGEPAGE_COUNT.fetch_sub(aligned_size / self.page_size, Ordering::Relaxed);
    }
}

/// A memory allocator that uses hugepages for large allocations
pub struct HugePageAllocator {
    #[cfg(target_os = "linux")]
    min_allocation_size: usize,
    #[cfg(target_os = "linux")]
    preferred_page_size: usize,
    #[cfg(not(target_os = "linux"))]
    _phantom: std::marker::PhantomData<u8>,
}

impl HugePageAllocator {
    /// Create a new hugepage allocator
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            Ok(Self {
                min_allocation_size: HUGEPAGE_SIZE_2MB,
                preferred_page_size: HUGEPAGE_SIZE_2MB,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(ToplingError::not_supported(
                "hugepages only supported on Linux",
            ))
        }
    }

    /// Create a new hugepage allocator with custom settings
    pub fn with_config(min_size: usize, page_size: usize) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            if page_size != HUGEPAGE_SIZE_2MB && page_size != HUGEPAGE_SIZE_1GB {
                return Err(ToplingError::invalid_data("invalid hugepage size"));
            }

            Ok(Self {
                min_allocation_size: min_size,
                preferred_page_size: page_size,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = (min_size, page_size);
            Err(ToplingError::not_supported(
                "hugepages only supported on Linux",
            ))
        }
    }

    /// Allocate memory, using hugepages for large allocations
    pub fn allocate(&self, size: usize) -> Result<HugePage> {
        #[cfg(target_os = "linux")]
        {
            if size >= self.min_allocation_size {
                HugePage::new(size, self.preferred_page_size)
            } else {
                Err(ToplingError::invalid_data(
                    "allocation too small for hugepages",
                ))
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = size;
            Err(ToplingError::not_supported(
                "hugepages only supported on Linux",
            ))
        }
    }

    /// Check if hugepages should be used for the given allocation size
    pub fn should_use_hugepages(&self, size: usize) -> bool {
        #[cfg(target_os = "linux")]
        {
            size >= self.min_allocation_size
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = size;
            false
        }
    }
}

impl Default for HugePageAllocator {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            #[cfg(target_os = "linux")]
            {
                Self {
                    min_allocation_size: HUGEPAGE_SIZE_2MB,
                    preferred_page_size: HUGEPAGE_SIZE_2MB,
                }
            }

            #[cfg(not(target_os = "linux"))]
            {
                Self {
                    _phantom: std::marker::PhantomData,
                }
            }
        })
    }
}

/// Get information about system hugepage configuration
pub fn get_hugepage_info(page_size: usize) -> Result<HugePageInfo> {
    #[cfg(target_os = "linux")]
    {
        let path = match page_size {
            HUGEPAGE_SIZE_2MB => "/sys/kernel/mm/hugepages/hugepages-2048kB",
            HUGEPAGE_SIZE_1GB => "/sys/kernel/mm/hugepages/hugepages-1048576kB",
            _ => return Err(ToplingError::invalid_data("unsupported hugepage size")),
        };

        let total_pages = read_hugepage_value(&format!("{}/nr_hugepages", path))?;
        let free_pages = read_hugepage_value(&format!("{}/free_hugepages", path))?;
        let reserved_pages = read_hugepage_value(&format!("{}/resv_hugepages", path))?;

        Ok(HugePageInfo {
            page_size,
            total_pages,
            free_pages,
            reserved_pages,
        })
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = page_size;
        Err(ToplingError::not_supported(
            "hugepages only supported on Linux",
        ))
    }
}

#[cfg(target_os = "linux")]
fn read_hugepage_value(path: &str) -> Result<usize> {
    let content = fs::read_to_string(path)
        .map_err(|_| ToplingError::io_error("failed to read hugepage information"))?;

    content
        .trim()
        .parse()
        .map_err(|_| ToplingError::invalid_data("invalid hugepage value"))
}

/// Initialize hugepage support
pub fn init_hugepage_support() -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        // Check if hugepages are available
        let info_2mb = get_hugepage_info(HUGEPAGE_SIZE_2MB);
        let info_1gb = get_hugepage_info(HUGEPAGE_SIZE_1GB);

        match (info_2mb, info_1gb) {
            (Ok(info), _) | (_, Ok(info)) => {
                log::debug!(
                    "Hugepage support initialized: {} pages of {} bytes",
                    info.total_pages,
                    info.page_size
                );
                Ok(())
            }
            (Err(_), Err(_)) => {
                log::warn!("Hugepages not available on this system");
                Err(ToplingError::not_supported("hugepages not available"))
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        Err(ToplingError::not_supported(
            "hugepages only supported on Linux",
        ))
    }
}

/// Get the current number of allocated hugepages
pub fn get_hugepage_count() -> usize {
    #[cfg(target_os = "linux")]
    {
        HUGEPAGE_COUNT.load(Ordering::Relaxed)
    }

    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Check if hugepages are available on the system
pub fn hugepages_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        get_hugepage_info(HUGEPAGE_SIZE_2MB).is_ok() || get_hugepage_info(HUGEPAGE_SIZE_1GB).is_ok()
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hugepage_constants() {
        assert_eq!(HUGEPAGE_SIZE_2MB, 2 * 1024 * 1024);
        assert_eq!(HUGEPAGE_SIZE_1GB, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_hugepage_availability() {
        // Should not panic
        let available = hugepages_available();

        #[cfg(target_os = "linux")]
        {
            // On Linux, this depends on system configuration
            println!("Hugepages available: {}", available);
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(!available);
        }
    }

    #[test]
    fn test_hugepage_info() {
        let result = get_hugepage_info(HUGEPAGE_SIZE_2MB);

        #[cfg(target_os = "linux")]
        {
            // On Linux, this might succeed or fail depending on system config
            match result {
                Ok(info) => {
                    assert_eq!(info.page_size, HUGEPAGE_SIZE_2MB);
                    println!("Hugepage info: {:?}", info);
                }
                Err(_) => {
                    println!("Hugepages not available");
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_hugepage_allocator_creation() {
        let result = HugePageAllocator::new();

        #[cfg(target_os = "linux")]
        {
            // Might succeed or fail depending on system
            match result {
                Ok(allocator) => {
                    assert!(allocator.should_use_hugepages(HUGEPAGE_SIZE_2MB));
                    assert!(!allocator.should_use_hugepages(1024));
                }
                Err(_) => {
                    println!("Hugepage allocator creation failed");
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_hugepage_allocator_config() {
        let result = HugePageAllocator::with_config(HUGEPAGE_SIZE_2MB, HUGEPAGE_SIZE_2MB);

        #[cfg(target_os = "linux")]
        {
            // Should succeed on Linux
            match result {
                Ok(allocator) => {
                    assert!(allocator.should_use_hugepages(HUGEPAGE_SIZE_2MB));
                    assert!(!allocator.should_use_hugepages(1024));
                }
                Err(e) => {
                    println!("Hugepage allocator config failed: {:?}", e);
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_init_hugepage_support() {
        let result = init_hugepage_support();

        #[cfg(target_os = "linux")]
        {
            // Might succeed or fail depending on system
            match result {
                Ok(_) => {
                    println!("Hugepage support initialized");
                }
                Err(_) => {
                    println!("Hugepage support initialization failed");
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_hugepage_count() {
        let _count = get_hugepage_count();
        // Function should not panic and returns valid usize
    }

    // Note: Actual hugepage allocation tests are not included here because
    // they require system-level hugepage configuration and may fail on
    // systems without sufficient hugepage availability.
}
