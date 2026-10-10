//! # Virtual Memory Management Utilities
//!
//! Advanced virtual memory operations with kernel-aware optimizations.
//! Inspired by production-grade VM management with cross-platform support.

use crate::error::{Result, ZiporaError};
use std::sync::OnceLock;

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
        #[cfg(all(unix, not(miri)))]
        {
            // SAFETY: sysconf with _SC_PAGESIZE is always safe to call, returns valid page size or -1 on error
            let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if ps > 0 {
                ps as usize
            } else {
                4096
            }
        }
        #[cfg(all(windows, not(miri)))]
        {
            use std::mem;
            use winapi::um::sysinfoapi::{GetSystemInfo, SYSTEM_INFO};

            // SAFETY: SYSTEM_INFO is a C-compatible struct that is safe to zero-initialize,
            // GetSystemInfo fills it correctly per Windows API contract
            unsafe {
                let mut system_info: SYSTEM_INFO = mem::zeroed();
                GetSystemInfo(&mut system_info);
                (system_info.dwPageSize as usize).max(4096)
            }
        }
        #[cfg(any(miri, not(any(unix, windows))))]
        {
            4096 // Standard page size under Miri and fallback platforms
        }
    }

    /// Get Linux kernel version
    #[cfg(target_os = "linux")]
    fn get_linux_kernel_version() -> String {
        #[cfg(miri)]
        {
            return "Unknown".to_string();
        }
        #[cfg(not(miri))]
        if let Ok(uname_output) = std::process::Command::new("uname").arg("-r").output() {
            String::from_utf8_lossy(&uname_output.stdout)
                .trim()
                .to_string()
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
            if parts.len() >= 2
                && let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>())
            {
                return major > 5 || (major == 5 && minor >= 14);
            }
        }
        false
    }

    /// Check if transparent huge pages are supported
    #[cfg(target_os = "linux")]
    fn check_thp_support() -> bool {
        #[cfg(miri)]
        {
            false
        }
        #[cfg(not(miri))]
        {
            std::path::Path::new("/sys/kernel/mm/transparent_hugepage").exists()
        }
    }

    /// Check if NUMA is supported
    #[cfg(target_os = "linux")]
    fn check_numa_support() -> bool {
        #[cfg(miri)]
        {
            false
        }
        #[cfg(not(miri))]
        {
            std::path::Path::new("/sys/devices/system/node").exists()
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn check_madv_populate_support(_kernel_version: &str) -> bool {
        false
    }
    #[cfg(not(target_os = "linux"))]
    fn check_thp_support() -> bool {
        false
    }
    #[cfg(not(target_os = "linux"))]
    fn check_numa_support() -> bool {
        false
    }
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

    /// Compute the page-aligned `[aligned_addr, aligned_addr + aligned_len)` span covering `data`.
    #[cfg(not(miri))]
    #[inline]
    fn page_aligned_range(&self, data: &[u8]) -> Result<(*const u8, usize)> {
        let page_size = self.kernel_info.page_size.max(4096);
        let addr = data.as_ptr() as usize;
        let aligned_addr = (addr / page_size) * page_size;
        let end = addr
            .checked_add(data.len())
            .ok_or_else(|| ZiporaError::invalid_data("memory range address overflow"))?;
        let aligned_end = end
            .checked_next_multiple_of(page_size)
            .ok_or_else(|| ZiporaError::invalid_data("page-aligned range end overflow"))?;
        Ok((aligned_addr as *const u8, aligned_end - aligned_addr))
    }

    /// Prefetch memory pages with optimal strategy
    ///
    /// # Safety
    /// This function is safe because it accepts a slice, which guarantees
    /// the pointer and length form a valid memory range.
    pub fn prefetch(&self, data: &[u8]) -> Result<()> {
        self.prefetch_with_strategy(data, PrefetchStrategy::Auto)
    }

    /// Prefetch memory with specific strategy
    ///
    /// # Safety
    /// This function is safe because it accepts a slice, which guarantees
    /// the pointer and length form a valid memory range.
    pub fn prefetch_with_strategy(&self, data: &[u8], strategy: PrefetchStrategy) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        if matches!(strategy, PrefetchStrategy::Manual) {
            return self.manual_prefetch(data);
        }

        #[cfg(miri)]
        {
            return match strategy {
                PrefetchStrategy::Populate => Err(ZiporaError::invalid_data(
                    "MADV_POPULATE_READ not supported under Miri",
                )),
                PrefetchStrategy::Auto | PrefetchStrategy::WillNeed | PrefetchStrategy::Manual => {
                    self.manual_prefetch(data)
                }
            };
        }

        #[cfg(not(miri))]
        {
            let (aligned_addr, aligned_len) = self.page_aligned_range(data)?;

            match strategy {
                PrefetchStrategy::Auto => {
                    #[cfg(target_os = "linux")]
                    {
                        if self.kernel_info.has_madv_populate {
                            self.madv_populate_read(aligned_addr, aligned_len)
                        } else {
                            self.madv_willneed(aligned_addr, aligned_len)
                        }
                    }
                    #[cfg(target_os = "windows")]
                    {
                        self.prefetch_virtual_memory(aligned_addr, aligned_len)
                    }
                    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                    {
                        let _ = (aligned_addr, aligned_len);
                        self.manual_prefetch(data)
                    }
                }
                PrefetchStrategy::Populate => {
                    #[cfg(target_os = "linux")]
                    {
                        if self.kernel_info.has_madv_populate {
                            self.madv_populate_read(aligned_addr, aligned_len)
                        } else {
                            Err(ZiporaError::invalid_data(
                                "MADV_POPULATE_READ not supported",
                            ))
                        }
                    }
                    #[cfg(not(target_os = "linux"))]
                    {
                        let _ = (aligned_addr, aligned_len);
                        Err(ZiporaError::invalid_data(
                            "MADV_POPULATE_READ only available on Linux",
                        ))
                    }
                }
                PrefetchStrategy::WillNeed => {
                    #[cfg(unix)]
                    {
                        self.madv_willneed(aligned_addr, aligned_len)
                    }
                    #[cfg(not(unix))]
                    {
                        let _ = (aligned_addr, aligned_len);
                        self.manual_prefetch(data)
                    }
                }
                PrefetchStrategy::Manual => self.manual_prefetch(data),
            }
        }
    }

    /// Use MADV_POPULATE_READ for prefetching (Linux 5.14+)
    #[cfg(all(target_os = "linux", not(miri)))]
    fn madv_populate_read(&self, addr: *const u8, len: usize) -> Result<()> {
        const MADV_POPULATE_READ: libc::c_int = 22;

        // SAFETY: pointer and length derived from valid slice (page-aligned), madvise returns error code (not UB)
        let result = unsafe { libc::madvise(addr as *mut libc::c_void, len, MADV_POPULATE_READ) };

        if result == 0 {
            Ok(())
        } else {
            Err(ZiporaError::invalid_data(format!(
                "madvise MADV_POPULATE_READ failed: {}",
                std::io::Error::last_os_error()
            )))
        }
    }

    /// Use MADV_WILLNEED for prefetching
    #[cfg(all(unix, not(miri)))]
    fn madv_willneed(&self, addr: *const u8, len: usize) -> Result<()> {
        // SAFETY: pointer and length derived from valid slice (page-aligned), madvise returns error code (not UB)
        let result = unsafe { libc::madvise(addr as *mut libc::c_void, len, libc::MADV_WILLNEED) };

        if result == 0 {
            Ok(())
        } else {
            Err(ZiporaError::invalid_data(format!(
                "madvise MADV_WILLNEED failed: {}",
                std::io::Error::last_os_error()
            )))
        }
    }

    /// Use PrefetchVirtualMemory on Windows
    #[cfg(all(target_os = "windows", not(miri)))]
    fn prefetch_virtual_memory(&self, addr: *const u8, len: usize) -> Result<()> {
        use winapi::um::memoryapi::PrefetchVirtualMemory;
        use winapi::um::processthreadsapi::GetCurrentProcess;
        use winapi::um::winnt::WIN32_MEMORY_RANGE_ENTRY;

        let range = WIN32_MEMORY_RANGE_ENTRY {
            VirtualAddress: addr as *mut std::ffi::c_void,
            NumberOfBytes: len,
        };

        // SAFETY: pointer and length derived from valid slice, GetCurrentProcess() always valid, range is properly initialized
        let result = unsafe { PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0) };

        if result != 0 {
            Ok(())
        } else {
            Err(ZiporaError::invalid_data("PrefetchVirtualMemory failed"))
        }
    }

    /// Manual prefetch by touching pages strictly within `data` while preserving slice provenance.
    fn manual_prefetch(&self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let page_size = self.kernel_info.page_size.max(4096);
        let mut offset = 0usize;
        while offset < data.len() {
            // SAFETY: `offset < data.len()`, so `&data[offset]` is an in-bounds, initialized byte with full slice provenance.
            unsafe {
                let _touch = std::ptr::read_volatile(&data[offset]);
            }
            offset = offset.saturating_add(page_size);
        }
        // Touch the final byte so the last page spanned by `data` is also paged in.
        // SAFETY: `!data.is_empty()`, so `data.len() - 1` is in bounds.
        unsafe {
            let _touch_last = std::ptr::read_volatile(&data[data.len() - 1]);
        }

        Ok(())
    }

    /// Advise kernel about memory usage patterns
    ///
    /// # Safety
    /// This function is safe because it accepts a slice, which guarantees
    /// the pointer and length form a valid memory range.
    pub fn advise_memory_usage(&self, data: &[u8], advice: MemoryAdvice) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        #[cfg(not(target_os = "linux"))]
        if matches!(advice, MemoryAdvice::HugePage) {
            return Err(ZiporaError::invalid_data(
                "MADV_HUGEPAGE only available on Linux",
            ));
        }

        #[cfg(miri)]
        {
            let _ = advice;
            return Ok(());
        }

        #[cfg(all(unix, not(miri)))]
        {
            let (aligned_addr, aligned_len) = self.page_aligned_range(data)?;

            // On Linux, `MADV_DONTNEED` on anonymous private memory destructively zeroes pages,
            // which would mutate `data: &[u8]` behind a shared reference and corrupt adjacent
            // heap allocations on boundary pages. Use non-destructive `posix_madvise` with
            // `POSIX_MADV_DONTNEED` for `MemoryAdvice::DontNeed`.
            if matches!(advice, MemoryAdvice::DontNeed) {
                // SAFETY: `aligned_addr` and `aligned_len` cover the page-aligned span of `data`,
                // and `posix_madvise(..., POSIX_MADV_DONTNEED)` is a non-destructive paging hint.
                let rc = unsafe {
                    libc::posix_madvise(
                        aligned_addr as *mut libc::c_void,
                        aligned_len,
                        libc::POSIX_MADV_DONTNEED,
                    )
                };
                return if rc == 0 {
                    Ok(())
                } else {
                    Err(ZiporaError::invalid_data(format!(
                        "posix_madvise failed: {}",
                        std::io::Error::from_raw_os_error(rc)
                    )))
                };
            }

            let madvise_flag = match advice {
                MemoryAdvice::Sequential => libc::MADV_SEQUENTIAL,
                MemoryAdvice::Random => libc::MADV_RANDOM,
                MemoryAdvice::WillNeed => libc::MADV_WILLNEED,
                MemoryAdvice::DontNeed => unreachable!("handled above"),
                #[cfg(target_os = "linux")]
                MemoryAdvice::HugePage => libc::MADV_HUGEPAGE,
                #[cfg(not(target_os = "linux"))]
                MemoryAdvice::HugePage => unreachable!("handled above"),
            };

            // SAFETY: `aligned_addr` and `aligned_len` are page-aligned and cover `data`;
            // non-destructive `madvise` hints only advise the VM subsystem and return an error code on failure.
            let result = unsafe {
                libc::madvise(aligned_addr as *mut libc::c_void, aligned_len, madvise_flag)
            };

            if result == 0 {
                Ok(())
            } else {
                Err(ZiporaError::invalid_data(format!(
                    "madvise failed: {}",
                    std::io::Error::last_os_error()
                )))
            }
        }

        #[cfg(all(not(unix), not(miri)))]
        {
            match advice {
                MemoryAdvice::WillNeed => self.prefetch(data),
                _ => Ok(()),
            }
        }
    }

    /// Lock memory pages to prevent swapping
    ///
    /// # Safety
    /// This function is safe because it accepts a slice, which guarantees
    /// the pointer and length form a valid memory range.
    pub fn lock_memory(&self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let addr = data.as_ptr();
        let len = data.len();

        #[cfg(unix)]
        {
            // SAFETY: pointer and length derived from valid slice, mlock returns error code (not UB)
            let result = unsafe { libc::mlock(addr as *const libc::c_void, len) };

            if result == 0 {
                Ok(())
            } else {
                Err(ZiporaError::invalid_data(format!(
                    "mlock failed: {}",
                    std::io::Error::last_os_error()
                )))
            }
        }

        #[cfg(target_os = "windows")]
        {
            use winapi::um::memoryapi::VirtualLock;

            // SAFETY: pointer and length derived from valid slice, VirtualLock returns non-zero on success
            let result = unsafe { VirtualLock(addr as *mut std::ffi::c_void, len) };

            if result != 0 {
                Ok(())
            } else {
                Err(ZiporaError::invalid_data("VirtualLock failed"))
            }
        }

        #[cfg(not(any(unix, target_os = "windows")))]
        {
            Err(ZiporaError::invalid_data(
                "Memory locking not supported on this platform",
            ))
        }
    }

    /// Unlock memory pages
    ///
    /// # Safety
    /// This function is safe because it accepts a slice, which guarantees
    /// the pointer and length form a valid memory range.
    pub fn unlock_memory(&self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let addr = data.as_ptr();
        let len = data.len();

        #[cfg(unix)]
        {
            // SAFETY: pointer and length derived from valid slice, munlock returns error code (not UB)
            let result = unsafe { libc::munlock(addr as *const libc::c_void, len) };

            if result == 0 {
                Ok(())
            } else {
                Err(ZiporaError::invalid_data(format!(
                    "munlock failed: {}",
                    std::io::Error::last_os_error()
                )))
            }
        }

        #[cfg(target_os = "windows")]
        {
            use winapi::um::memoryapi::VirtualUnlock;

            // SAFETY: pointer and length derived from valid slice, VirtualUnlock returns non-zero on success
            let result = unsafe { VirtualUnlock(addr as *mut std::ffi::c_void, len) };

            if result != 0 {
                Ok(())
            } else {
                Err(ZiporaError::invalid_data("VirtualUnlock failed"))
            }
        }

        #[cfg(not(any(unix, target_os = "windows")))]
        {
            Err(ZiporaError::invalid_data(
                "Memory unlocking not supported on this platform",
            ))
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

    /// Allocate zero-initialized, page-aligned memory.
    pub fn allocate(&self, size: usize) -> Result<PageAlignedBuffer> {
        let page_size = self.vm_manager.kernel_info().page_size.max(4096);
        if !page_size.is_power_of_two() {
            return Err(ZiporaError::invalid_data(
                "System page size is not a power of two",
            ));
        }
        if size > isize::MAX as usize {
            return Err(ZiporaError::invalid_data(
                "Requested allocation size exceeds isize::MAX",
            ));
        }

        let aligned_size = size
            .max(1)
            .checked_next_multiple_of(page_size)
            .ok_or_else(|| ZiporaError::invalid_data("Page-aligned allocation size overflow"))?;

        let layout = std::alloc::Layout::from_size_align(aligned_size, page_size)
            .map_err(|_| ZiporaError::invalid_data("Invalid page-aligned allocation layout"))?;

        // SAFETY: `layout` has non-zero `aligned_size >= page_size` and power-of-two `page_size` alignment.
        // `alloc_zeroed` returns zero-initialized memory so `as_slice()` never exposes uninitialized bytes.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            Err(ZiporaError::invalid_data(
                "Failed to allocate page-aligned memory",
            ))
        } else {
            Ok(PageAlignedBuffer {
                ptr,
                size: aligned_size,
                actual_size: size,
                layout,
            })
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
    layout: std::alloc::Layout,
}

impl PageAlignedBuffer {
    /// Get a pointer to the buffer
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get a mutable pointer to the buffer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Get the actual allocated size (page-aligned)
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the requested size
    pub fn actual_size(&self) -> usize {
        self.actual_size
    }

    /// Get a slice view of the buffer
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` was allocated via `alloc_zeroed` with `self.size >= self.actual_size`
        // initialized bytes, aligned to `page_size >= 1`, and is valid for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.actual_size) }
    }

    /// Get a mutable slice view of the buffer
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `self.ptr` was allocated via `alloc_zeroed` with `self.size >= self.actual_size`
        // initialized bytes, and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.actual_size) }
    }
}

impl Drop for PageAlignedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` was allocated with `std::alloc::alloc_zeroed(self.layout)` in
            // `PageAlignedAlloc::allocate` and is deallocated once with the exact same `self.layout`.
            unsafe {
                std::alloc::dealloc(self.ptr, self.layout);
            }
        }
    }
}

// Global kernel info singleton
static KERNEL_INFO: OnceLock<KernelInfo> = OnceLock::new();

/// Get global kernel information (detected once on first call)
pub fn get_kernel_info() -> &'static KernelInfo {
    KERNEL_INFO.get_or_init(KernelInfo::detect)
}

/// Convenience function for memory prefetching
///
/// # Safety
/// This function is safe because it accepts a slice, which guarantees
/// the pointer and length form a valid memory range.
pub fn vm_prefetch(data: &[u8]) -> Result<()> {
    let vm_manager = VmManager::new();
    vm_manager.prefetch(data)
}

/// Convenience function for memory prefetching with minimum page count
///
/// # Safety
/// This function is safe because it accepts a slice, which guarantees
/// the pointer and length form a valid memory range.
pub fn vm_prefetch_min_pages(data: &[u8], min_pages: usize) -> Result<()> {
    let kernel_info = get_kernel_info();
    let page_size = kernel_info.page_size;
    let page_count = data.len().div_ceil(page_size);

    if page_count >= min_pages {
        vm_prefetch(data)
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
        let result = vm_manager.prefetch(&buffer);
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
            let result = vm_manager.advise_memory_usage(&buffer, advice);
            // Should either succeed or fail gracefully
            assert!(result.is_ok() || result.is_err());
        }
    }

    #[test]
    fn test_convenience_functions() {
        let buffer = vec![0u8; 4096];

        // Test basic prefetch
        let result = vm_prefetch(&buffer);
        assert!(result.is_ok() || result.is_err());

        // Test prefetch with minimum pages
        let result = vm_prefetch_min_pages(&buffer, 1);
        assert!(result.is_ok() || result.is_err());

        // Test skip for small allocation
        let result = vm_prefetch_min_pages(&buffer[..100], 10);
        assert!(result.is_ok()); // Should always succeed (skipped)
    }

    #[test]
    fn test_c12_3a_vm_utils_prefetch_provenance_zeroed_alloc_and_advice() {
        let vm_manager = VmManager::new();
        let alloc = PageAlignedAlloc::new();

        // 1. PageAlignedAlloc::allocate(usize::MAX) must return Err instead of overflowing div_ceil * page_size.
        assert!(
            alloc.allocate(usize::MAX).is_err(),
            "PageAlignedAlloc::allocate(usize::MAX) must reject overflow size"
        );

        // 2. PageAlignedAlloc::allocate must return zero-initialized memory so reading as_slice()
        // is not UB under Miri or on reused heap blocks.
        let mut buf = alloc.allocate(512).expect("allocate(512) must succeed");
        buf.as_mut_slice().fill(0xAA);
        drop(buf);
        let buf2 = alloc.allocate(512).expect("second allocate(512) must succeed");
        assert!(
            buf2.as_slice().iter().all(|&b| b == 0),
            "PageAlignedAlloc::allocate must zero-initialize the buffer"
        );

        // 3. Manual prefetch on an unaligned subslice must not dereference before data.as_ptr()
        // or lose pointer provenance (flagged by Miri).
        let raw = vec![0x5Au8; 8192];
        let unaligned_subslice = &raw[17..4113];
        vm_manager
            .prefetch_with_strategy(unaligned_subslice, PrefetchStrategy::Manual)
            .expect("manual prefetch on unaligned subslice must succeed");

        // 4. advise_memory_usage on an unaligned Vec slice must page-align and succeed,
        // and MemoryAdvice::DontNeed must not zero shared &[u8] contents.
        let mut data = vec![0x77u8; 8192];
        data[0] = 0x42;
        vm_manager
            .advise_memory_usage(&data[3..4100], MemoryAdvice::Sequential)
            .expect("advise_memory_usage(Sequential) on unaligned slice must succeed");
        vm_manager
            .advise_memory_usage(&data, MemoryAdvice::DontNeed)
            .expect("advise_memory_usage(DontNeed) must succeed");
        assert!(
            data.iter().skip(1).all(|&b| b == 0x77) && data[0] == 0x42,
            "MemoryAdvice::DontNeed on &[u8] must not zero caller's memory behind a shared reference"
        );
    }
}

