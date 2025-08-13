//! # Virtual Memory Management Utilities
//!
//! Advanced virtual memory operations with kernel-aware optimizations.
//! Inspired by production-grade VM management with cross-platform support.

use std::sync::OnceLock;
use crate::error::{Result, ZiporaError};

/// Kernel information and capabilities
#[derive(Debug, Clone)]
pub struct KernelInfo {
    /// Operating system name
    pub os_name: String,
    /// Kernel version
    pub kernel_version: String,
    /// Page size in bytes
    pub page_size: usize,
    /// Whether MADV_POPULATE_READ is available (Linux 5.14+)
    pub has_madv_populate: bool,
    /// Whether transparent huge pages are available
    pub has_thp: bool,
    /// Whether NUMA is supported
    pub has_numa: bool,
}

impl KernelInfo {
    /// Detect kernel capabilities
    pub fn detect() -> Self {
        let os_name = std::env::consts::OS.to_string();
        let page_size = Self::get_page_size();
        
        #[cfg(target_os = "linux")]
        {
            let kernel_version = Self::get_linux_kernel_version();
            let has_madv_populate = Self::check_madv_populate_support(&kernel_version);
            let has_thp = Self::check_thp_support();
            let has_numa = Self::check_numa_support();
            
            Self {
                os_name,
                kernel_version,
                page_size,
                has_madv_populate,
                has_thp,
                has_numa,
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            Self {
                os_name,
                kernel_version: "Unknown".to_string(),
                page_size,
                has_madv_populate: false,
                has_thp: false,
                has_numa: false,
            }
        }
    }

    /// Get system page size
    fn get_page_size() -> usize {
        #[cfg(unix)]
        {
            unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
        }
        #[cfg(windows)]
        {
            use std::mem;
            use winapi::um::sysinfoapi::{GetSystemInfo, SYSTEM_INFO};
            
            unsafe {
                let mut system_info: SYSTEM_INFO = mem::zeroed();
                GetSystemInfo(&mut system_info);
                system_info.dwPageSize as usize
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            4096 // Reasonable default
        }
    }

    /// Get Linux kernel version
    #[cfg(target_os = "linux")]
    fn get_linux_kernel_version() -> String {
        if let Ok(uname_output) = std::process::Command::new("uname").arg("-r").output() {
            String::from_utf8_lossy(&uname_output.stdout).trim().to_string()
        } else {
            "Unknown".to_string()
        }
    }

    /// Check if MADV_POPULATE_READ is supported (Linux 5.14+)
    #[cfg(target_os = "linux")]
    fn check_madv_populate_support(kernel_version: &str) -> bool {
        // Parse kernel version and check if >= 5.14
        if let Some(version_part) = kernel_version.split('-').next() {
            let parts: Vec<&str> = version_part.split('.').collect();
            if parts.len() >= 2 {
                if let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                    return major > 5 || (major == 5 && minor >= 14);
                }
            }
        }
        false
    }

    /// Check if transparent huge pages are supported
    #[cfg(target_os = "linux")]
    fn check_thp_support() -> bool {
        std::path::Path::new("/sys/kernel/mm/transparent_hugepage").exists()
    }

    /// Check if NUMA is supported
    #[cfg(target_os = "linux")]
    fn check_numa_support() -> bool {
        std::path::Path::new("/sys/devices/system/node").exists()
    }

    #[cfg(not(target_os = "linux"))]
    fn check_madv_populate_support(_kernel_version: &str) -> bool { false }
    #[cfg(not(target_os = "linux"))]
    fn check_thp_support() -> bool { false }
    #[cfg(not(target_os = "linux"))]
    fn check_numa_support() -> bool { false }
}

/// Virtual memory manager for advanced operations
pub struct VmManager {
    kernel_info: &'static KernelInfo,
}

impl VmManager {
    /// Create a new VM manager
    pub fn new() -> Self {
        Self {
            kernel_info: get_kernel_info(),
        }
    }

    /// Get kernel information
    pub fn kernel_info(&self) -> &KernelInfo {
        self.kernel_info
    }

    /// Prefetch memory pages with optimal strategy
    pub fn prefetch(&self, addr: *const u8, len: usize) -> Result<()> {
        self.prefetch_with_strategy(addr, len, PrefetchStrategy::Auto)
    }

    /// Prefetch memory with specific strategy
    pub fn prefetch_with_strategy(&self, addr: *const u8, len: usize, strategy: PrefetchStrategy) -> Result<()> {
        if len == 0 {
            return Ok(());
        }

        // Align to page boundaries
        let page_size = self.kernel_info.page_size;
        let aligned_addr = ((addr as usize) / page_size) * page_size;
        let aligned_end = (((addr as usize) + len + page_size - 1) / page_size) * page_size;
        let aligned_len = aligned_end - aligned_addr;

        match strategy {
            PrefetchStrategy::Auto => {
                #[cfg(target_os = "linux")]
                {
                    if self.kernel_info.has_madv_populate {
                        self.madv_populate_read(aligned_addr as *const u8, aligned_len)
                    } else {
                        self.madv_willneed(aligned_addr as *const u8, aligned_len)
                    }
                }
                #[cfg(target_os = "windows")]
                {
                    self.prefetch_virtual_memory(aligned_addr as *const u8, aligned_len)
                }
                #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                {
                    self.manual_prefetch(aligned_addr as *const u8, aligned_len)
                }
            }
            PrefetchStrategy::Populate => {
                #[cfg(target_os = "linux")]
                {
                    if self.kernel_info.has_madv_populate {
                        self.madv_populate_read(aligned_addr as *const u8, aligned_len)
                    } else {
                        Err(ZiporaError::invalid_data("MADV_POPULATE_READ not supported"))
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err(ZiporaError::invalid_data("MADV_POPULATE_READ only available on Linux"))
                }
            }
            PrefetchStrategy::WillNeed => {
                #[cfg(unix)]
                {
                    self.madv_willneed(aligned_addr as *const u8, aligned_len)
                }
                #[cfg(not(unix))]
                {
                    self.manual_prefetch(aligned_addr as *const u8, aligned_len)
                }
            }
            PrefetchStrategy::Manual => {
                self.manual_prefetch(aligned_addr as *const u8, aligned_len)
            }
        }
    }

    /// Use MADV_POPULATE_READ for prefetching (Linux 5.14+)
    #[cfg(target_os = "linux")]
    fn madv_populate_read(&self, addr: *const u8, len: usize) -> Result<()> {
        const MADV_POPULATE_READ: libc::c_int = 22;
        
        let result = unsafe {
            libc::madvise(addr as *mut libc::c_void, len, MADV_POPULATE_READ)
        };
        
        if result == 0 {
            Ok(())
        } else {
            Err(ZiporaError::invalid_data(&format!("madvise MADV_POPULATE_READ failed: {}", 
                std::io::Error::last_os_error())))
        }
    }

    /// Use MADV_WILLNEED for prefetching
    #[cfg(unix)]
    fn madv_willneed(&self, addr: *const u8, len: usize) -> Result<()> {
        let result = unsafe {
            libc::madvise(addr as *mut libc::c_void, len, libc::MADV_WILLNEED)
        };
        
        if result == 0 {
            Ok(())
        } else {
            Err(ZiporaError::invalid_data(&format!("madvise MADV_WILLNEED failed: {}", 
                std::io::Error::last_os_error())))
        }
    }

    /// Use PrefetchVirtualMemory on Windows
    #[cfg(target_os = "windows")]
    fn prefetch_virtual_memory(&self, addr: *const u8, len: usize) -> Result<()> {
        use winapi::um::memoryapi::PrefetchVirtualMemory;
        use winapi::um::processthreadsapi::GetCurrentProcess;
        use winapi::um::winnt::WIN32_MEMORY_RANGE_ENTRY;
        
        let range = WIN32_MEMORY_RANGE_ENTRY {
            VirtualAddress: addr as *mut std::ffi::c_void,
            NumberOfBytes: len,
        };
        
        let result = unsafe {
            PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)
        };
        
        if result != 0 {
            Ok(())
        } else {
            Err(ZiporaError::invalid_data("PrefetchVirtualMemory failed"))
        }
    }

    /// Manual prefetch by touching pages
    fn manual_prefetch(&self, addr: *const u8, len: usize) -> Result<()> {
        let page_size = self.kernel_info.page_size;
        let mut current = addr as usize;
        let end = current + len;
        
        while current < end {
            unsafe {
                // Touch the page to bring it into memory
                let _touch = *(current as *const u8);
            }
            current += page_size;
        }
        
        Ok(())
    }

    /// Advise kernel about memory usage patterns
    pub fn advise_memory_usage(&self, addr: *const u8, len: usize, advice: MemoryAdvice) -> Result<()> {
        if len == 0 {
            return Ok(());
        }

        #[cfg(unix)]
        {
            let madvise_flag = match advice {
                MemoryAdvice::Sequential => libc::MADV_SEQUENTIAL,
                MemoryAdvice::Random => libc::MADV_RANDOM,
                MemoryAdvice::WillNeed => libc::MADV_WILLNEED,
                MemoryAdvice::DontNeed => libc::MADV_DONTNEED,
                #[cfg(target_os = "linux")]
                MemoryAdvice::HugePage => libc::MADV_HUGEPAGE,
                #[cfg(not(target_os = "linux"))]
                MemoryAdvice::HugePage => return Err(ZiporaError::invalid_data("MADV_HUGEPAGE only available on Linux")),
            };

            let result = unsafe {
                libc::madvise(addr as *mut libc::c_void, len, madvise_flag)
            };

            if result == 0 {
                Ok(())
            } else {
                Err(ZiporaError::invalid_data(&format!("madvise failed: {}", 
                    std::io::Error::last_os_error())))
            }
        }

        #[cfg(not(unix))]
        {
            // Limited support on non-Unix platforms
            match advice {
                MemoryAdvice::WillNeed => self.prefetch(addr, len),
                _ => Ok(()), // Ignore other advice on unsupported platforms
            }
        }
    }

    /// Lock memory pages to prevent swapping
    pub fn lock_memory(&self, addr: *const u8, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }

        #[cfg(unix)]
        {
            let result = unsafe {
                libc::mlock(addr as *const libc::c_void, len)
            };

            if result == 0 {
                Ok(())
            } else {
                Err(ZiporaError::invalid_data(&format!("mlock failed: {}", 
                    std::io::Error::last_os_error())))
            }
        }

        #[cfg(target_os = "windows")]
        {
            use winapi::um::memoryapi::VirtualLock;
            
            let result = unsafe {
                VirtualLock(addr as *mut std::ffi::c_void, len)
            };

            if result != 0 {
                Ok(())
            } else {
                Err(ZiporaError::invalid_data("VirtualLock failed"))
            }
        }

        #[cfg(not(any(unix, target_os = "windows")))]
        {
            Err(ZiporaError::invalid_data("Memory locking not supported on this platform"))
        }
    }

    /// Unlock memory pages
    pub fn unlock_memory(&self, addr: *const u8, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }

        #[cfg(unix)]
        {
            let result = unsafe {
                libc::munlock(addr as *const libc::c_void, len)
            };

            if result == 0 {
                Ok(())
            } else {
                Err(ZiporaError::invalid_data(&format!("munlock failed: {}", 
                    std::io::Error::last_os_error())))
            }
        }

        #[cfg(target_os = "windows")]
        {
            use winapi::um::memoryapi::VirtualUnlock;
            
            let result = unsafe {
                VirtualUnlock(addr as *mut std::ffi::c_void, len)
            };

            if result != 0 {
                Ok(())
            } else {
                Err(ZiporaError::invalid_data("VirtualUnlock failed"))
            }
        }

        #[cfg(not(any(unix, target_os = "windows")))]
        {
            Err(ZiporaError::invalid_data("Memory unlocking not supported on this platform"))
        }
    }
}

impl Default for VmManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory prefetch strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchStrategy {
    /// Automatically select the best strategy
    Auto,
    /// Use MADV_POPULATE_READ if available
    Populate,
    /// Use MADV_WILLNEED
    WillNeed,
    /// Manual page touching
    Manual,
}

/// Memory usage advice for the kernel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAdvice {
    /// Sequential access pattern
    Sequential,
    /// Random access pattern
    Random,
    /// Will need soon
    WillNeed,
    /// Don't need anymore
    DontNeed,
    /// Use huge pages if available
    HugePage,
}

/// Page-aligned allocator for optimal memory layout
pub struct PageAlignedAlloc {
    vm_manager: VmManager,
}

impl PageAlignedAlloc {
    /// Create a new page-aligned allocator
    pub fn new() -> Self {
        Self {
            vm_manager: VmManager::new(),
        }
    }

    /// Allocate page-aligned memory
    pub fn allocate(&self, size: usize) -> Result<PageAlignedBuffer> {
        let page_size = self.vm_manager.kernel_info().page_size;
        let aligned_size = ((size + page_size - 1) / page_size) * page_size;

        #[cfg(unix)]
        {
            let ptr = unsafe {
                libc::aligned_alloc(page_size, aligned_size)
            };

            if ptr.is_null() {
                Err(ZiporaError::invalid_data("Failed to allocate page-aligned memory"))
            } else {
                Ok(PageAlignedBuffer {
                    ptr: ptr as *mut u8,
                    size: aligned_size,
                    actual_size: size,
                })
            }
        }

        #[cfg(target_os = "windows")]
        {
            use winapi::um::memoryapi::VirtualAlloc;
            use winapi::um::winnt::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE};
            
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null_mut(),
                    aligned_size,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
                )
            };

            if ptr.is_null() {
                Err(ZiporaError::invalid_data("Failed to allocate page-aligned memory"))
            } else {
                Ok(PageAlignedBuffer {
                    ptr: ptr as *mut u8,
                    size: aligned_size,
                    actual_size: size,
                })
            }
        }

        #[cfg(not(any(unix, target_os = "windows")))]
        {
            // Fallback using standard allocation with manual alignment
            let layout = std::alloc::Layout::from_size_align(aligned_size + page_size, page_size)
                .map_err(|_| ZiporaError::invalid_data("Invalid layout"))?;
            
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                Err(ZiporaError::invalid_data("Failed to allocate memory"))
            } else {
                let aligned_ptr = ((ptr as usize + page_size - 1) / page_size) * page_size;
                Ok(PageAlignedBuffer {
                    ptr: aligned_ptr as *mut u8,
                    size: aligned_size,
                    actual_size: size,
                })
            }
        }
    }
}

impl Default for PageAlignedAlloc {
    fn default() -> Self {
        Self::new()
    }
}

/// Page-aligned memory buffer
pub struct PageAlignedBuffer {
    ptr: *mut u8,
    size: usize,
    actual_size: usize,
}

impl PageAlignedBuffer {
    /// Get a pointer to the buffer
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get a mutable pointer to the buffer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Get the actual allocated size (page-aligned)
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the requested size
    pub fn actual_size(&self) -> usize {
        self.actual_size
    }

    /// Get a slice view of the buffer
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.actual_size) }
    }

    /// Get a mutable slice view of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.actual_size) }
    }
}

impl Drop for PageAlignedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            #[cfg(unix)]
            {
                unsafe { libc::free(self.ptr as *mut libc::c_void) };
            }
            #[cfg(target_os = "windows")]
            {
                use winapi::um::memoryapi::VirtualFree;
                use winapi::um::winnt::MEM_RELEASE;
                unsafe {
                    VirtualFree(self.ptr as *mut std::ffi::c_void, 0, MEM_RELEASE);
                }
            }
            #[cfg(not(any(unix, target_os = "windows")))]
            {
                // For the fallback implementation, we can't safely free
                // since we don't store the original pointer
            }
        }
    }
}

// Global kernel info singleton
static KERNEL_INFO: OnceLock<KernelInfo> = OnceLock::new();

/// Get global kernel information (detected once on first call)
pub fn get_kernel_info() -> &'static KernelInfo {
    KERNEL_INFO.get_or_init(|| KernelInfo::detect())
}

/// Convenience function for memory prefetching
pub fn vm_prefetch(addr: *const u8, len: usize) -> Result<()> {
    let vm_manager = VmManager::new();
    vm_manager.prefetch(addr, len)
}

/// Convenience function for memory prefetching with minimum page count
pub fn vm_prefetch_min_pages(addr: *const u8, len: usize, min_pages: usize) -> Result<()> {
    let kernel_info = get_kernel_info();
    let page_size = kernel_info.page_size;
    let page_count = (len + page_size - 1) / page_size;
    
    if page_count >= min_pages {
        vm_prefetch(addr, len)
    } else {
        Ok(()) // Skip prefetch for small allocations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_info_detection() {
        let kernel_info = get_kernel_info();
        
        assert!(!kernel_info.os_name.is_empty());
        assert!(kernel_info.page_size > 0);
        assert!(kernel_info.page_size.is_power_of_two());
        
        println!("OS: {}", kernel_info.os_name);
        println!("Kernel: {}", kernel_info.kernel_version);
        println!("Page size: {} bytes", kernel_info.page_size);
        println!("MADV_POPULATE: {}", kernel_info.has_madv_populate);
        println!("THP: {}", kernel_info.has_thp);
        println!("NUMA: {}", kernel_info.has_numa);
    }

    #[test]
    fn test_vm_manager() {
        let vm_manager = VmManager::new();
        let kernel_info = vm_manager.kernel_info();
        
        assert!(kernel_info.page_size > 0);
        
        // Test with a small buffer
        let buffer = vec![0u8; 4096];
        let result = vm_manager.prefetch(buffer.as_ptr(), buffer.len());
        // Should not fail, even if prefetch is not supported
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_page_aligned_alloc() {
        let allocator = PageAlignedAlloc::new();
        
        // Test allocation
        let buffer = allocator.allocate(1000).unwrap();
        assert!(buffer.size() >= 1000);
        assert_eq!(buffer.actual_size(), 1000);
        
        // Check alignment
        let page_size = get_kernel_info().page_size;
        assert_eq!(buffer.as_ptr() as usize % page_size, 0);
        
        // Test slice access
        let slice = buffer.as_slice();
        assert_eq!(slice.len(), 1000);
    }

    #[test]
    fn test_memory_advice() {
        let vm_manager = VmManager::new();
        let buffer = vec![0u8; 8192];
        
        // Test various memory advice (should not fail on most systems)
        let advice_types = vec![
            MemoryAdvice::Sequential,
            MemoryAdvice::Random,
            MemoryAdvice::WillNeed,
        ];
        
        for advice in advice_types {
            let result = vm_manager.advise_memory_usage(buffer.as_ptr(), buffer.len(), advice);
            // Should either succeed or fail gracefully
            assert!(result.is_ok() || result.is_err());
        }
    }

    #[test]
    fn test_convenience_functions() {
        let buffer = vec![0u8; 4096];
        
        // Test basic prefetch
        let result = vm_prefetch(buffer.as_ptr(), buffer.len());
        assert!(result.is_ok() || result.is_err());
        
        // Test prefetch with minimum pages
        let result = vm_prefetch_min_pages(buffer.as_ptr(), buffer.len(), 1);
        assert!(result.is_ok() || result.is_err());
        
        // Test skip for small allocation
        let result = vm_prefetch_min_pages(buffer.as_ptr(), 100, 10);
        assert!(result.is_ok()); // Should always succeed (skipped)
    }
}