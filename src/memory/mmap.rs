//! Memory-mapped allocator for large objects
//!
//! This module provides memory-mapped allocation for large objects to achieve
//! C++-competitive performance for allocations >16KB.

use crate::error::{Result, ZiporaError};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Memory-mapped allocation for high-performance large object allocation
pub struct MemoryMappedAllocator {
    /// Minimum size for memory-mapped allocations
    min_mmap_size: usize,
    /// Cache of memory-mapped regions to avoid repeated mmap/munmap
    region_cache: Arc<Mutex<HashMap<usize, Vec<*mut u8>>>>,
    /// Statistics
    total_allocated: AtomicU64,
    total_freed: AtomicU64,
    mmap_calls: AtomicU64,
    munmap_calls: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

impl Default for MemoryMappedAllocator {
    fn default() -> Self {
        Self::new(16 * 1024)
    }
}

/// An owned `mmap` region.
///
/// Dropping one unmaps the region, so it is safe to let an allocation simply
/// go out of scope. Handing it back to
/// [`MemoryMappedAllocator::deallocate`] instead is the faster route: the
/// allocator keeps the region mapped and reuses it, which is the whole point
/// of the allocator, and it is the only route that updates the statistics.
///
/// The region is *not* tied to the allocator's lifetime: an allocation
/// outlives the allocator that produced it, and unmaps itself when dropped.
#[derive(Debug)]
pub struct MmapAllocation {
    ptr: NonNull<u8>,
    size: usize,
    actual_size: usize, // Rounded up to page size
}

impl Drop for MmapAllocation {
    fn drop(&mut self) {
        // SAFETY: `ptr` and `actual_size` are the address and length of a
        // successful `libc::mmap` in `MemoryMappedAllocator::allocate` (or of
        // a region that call cached, which was mapped the same way). The only
        // other disposal route, `MemoryMappedAllocator::deallocate`, takes the
        // region over inside a `ManuallyDrop`, so this cannot run for a region
        // the allocator has already cached or unmapped.
        let rc = unsafe {
            libc::munmap(
                self.ptr.as_ptr() as *mut libc::c_void,
                self.actual_size,
            )
        };
        if rc != 0 {
            log::warn!(
                "failed to unmap {} bytes at {:p}: {}",
                self.actual_size,
                self.ptr.as_ptr(),
                std::io::Error::last_os_error()
            );
        }
    }
}

/// Statistics for memory-mapped allocations
#[derive(Debug, Clone)]
pub struct MmapStats {
    /// Total bytes allocated via mmap
    pub total_allocated: u64,
    /// Total bytes freed via munmap
    pub total_freed: u64,
    /// Number of mmap system calls made
    pub mmap_calls: u64,
    /// Number of munmap system calls made
    pub munmap_calls: u64,
    /// Number of times a cached region was reused
    pub cache_hits: u64,
    /// Number of times a new region had to be allocated
    pub cache_misses: u64,
    /// Number of regions currently in cache
    pub cached_regions: usize,
}

impl MemoryMappedAllocator {
    /// Create a new memory-mapped allocator
    // clippy::arc_with_non_send_sync: intentional — `unsafe impl Send/Sync for
    // MemoryMappedAllocator` soundly asserts cross-thread safety of the pointer cache.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(min_mmap_size: usize) -> Self {
        Self {
            min_mmap_size,
            region_cache: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: AtomicU64::new(0),
            total_freed: AtomicU64::new(0),
            mmap_calls: AtomicU64::new(0),
            munmap_calls: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }



    /// Allocate memory using mmap for optimal large allocation performance
    pub fn allocate(&self, size: usize) -> Result<MmapAllocation> {
        if size < self.min_mmap_size {
            return Err(ZiporaError::invalid_data(
                "allocation too small for memory mapping",
            ));
        }

        // Round up to page size for optimal performance
        let page_size = Self::get_page_size();
        let actual_size = size
            .checked_add(page_size - 1)
            .map(|n| n & !(page_size - 1))
            .ok_or_else(|| {
                ZiporaError::invalid_data("mmap allocation size overflows page rounding")
            })?;

        // Check cache first — blocking lock is vastly cheaper than an mmap syscall.
        // Recover from poisoning: cache contents (pointer HashMap) remain valid.
        let cached_ptr = {
            let mut cache = self.region_cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.get_mut(&actual_size).and_then(Vec::pop)
        };
        if let Some(ptr) = cached_ptr {
            // C3.22 (S8-R2): fresh anonymous mappings are kernel-zeroed, so a
            // cached region must be wiped over `0..size` before `as_slice` can
            // expose the previous tenant's bytes. Done after releasing
            // `region_cache` so the mutex is not held across the memset.
            // SAFETY: `ptr` came from a live `mmap` of `actual_size >= size`
            // writable bytes and was just popped exclusively by this thread.
            unsafe {
                std::ptr::write_bytes(ptr, 0, size);
            }
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            self.total_allocated
                .fetch_add(size as u64, Ordering::Relaxed);

            // SAFETY: cached ptr was obtained from successful mmap, guaranteed non-null
            return Ok(MmapAllocation {
                ptr: unsafe { NonNull::new_unchecked(ptr) },
                size,
                actual_size,
            });
        }

        // Cache miss — no cached region for this size
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.mmap_calls.fetch_add(1, Ordering::Relaxed);

        // SAFETY: fd=-1 for anonymous mapping, size/offset are page-aligned, flags are valid
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                actual_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(ZiporaError::out_of_memory(size));
        }

        // SAFETY: ptr is valid from successful mmap, actual_size matches allocation, hints are advisory
        // Use madvise for better performance hints
        unsafe {
            // Hint that we'll access this memory soon
            libc::madvise(ptr, actual_size, libc::MADV_WILLNEED);
            // Hint for sequential access pattern (if applicable)
            libc::madvise(ptr, actual_size, libc::MADV_SEQUENTIAL);
        }

        self.total_allocated
            .fetch_add(size as u64, Ordering::Relaxed);

        // SAFETY: ptr != MAP_FAILED guarantees non-null
        Ok(MmapAllocation {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut u8) },
            size,
            actual_size,
        })
    }

    /// Deallocate memory, potentially caching for reuse
    ///
    /// The allocator takes the region over: it is cached for the next
    /// allocation of the same size, or unmapped if the cache for that size is
    /// full. Dropping an [`MmapAllocation`] instead always unmaps it.
    pub fn deallocate(&self, allocation: MmapAllocation) -> Result<()> {
        // The allocation unmaps itself on drop. From here the region belongs
        // to the allocator, which either caches it (still mapped) or unmaps it
        // below, so that drop must not run.
        let allocation = std::mem::ManuallyDrop::new(allocation);

        self.total_freed
            .fetch_add(allocation.size as u64, Ordering::Relaxed);

        // Return region to cache — blocking lock is vastly cheaper than a munmap syscall.
        {
            let mut cache = self.region_cache.lock().unwrap_or_else(|e| e.into_inner());
            let regions = cache.entry(allocation.actual_size).or_default();

            const MAX_CACHED_REGIONS_PER_SIZE: usize = 4;
            if regions.len() < MAX_CACHED_REGIONS_PER_SIZE {
                regions.push(allocation.ptr.as_ptr());
                return Ok(());
            }
        }

        // Cache full for this size class — release back to OS
        self.munmap_calls.fetch_add(1, Ordering::Relaxed);
        // SAFETY: ptr from allocation was obtained via mmap with matching size
        unsafe {
            if libc::munmap(
                allocation.ptr.as_ptr() as *mut libc::c_void,
                allocation.actual_size,
            ) != 0
            {
                return Err(ZiporaError::io_error("failed to unmap memory"));
            }
        }

        Ok(())
    }

    /// Check if this allocator should be used for the given size
    pub fn should_use_mmap(&self, size: usize) -> bool {
        size >= self.min_mmap_size
    }

    /// Get current statistics
    pub fn stats(&self) -> MmapStats {
        let cached_regions = if let Ok(cache) = self.region_cache.try_lock() {
            cache.values().map(|v| v.len()).sum()
        } else {
            0
        };

        MmapStats {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_freed: self.total_freed.load(Ordering::Relaxed),
            mmap_calls: self.mmap_calls.load(Ordering::Relaxed),
            munmap_calls: self.munmap_calls.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cached_regions,
        }
    }

    /// Clear the region cache, forcing all cached regions to be unmapped
    pub fn clear_cache(&self) -> Result<()> {
        if let Ok(mut cache) = self.region_cache.lock() {
            for (size, regions) in cache.drain() {
                for ptr in regions {
                    self.munmap_calls.fetch_add(1, Ordering::Relaxed);
                    // SAFETY: cached ptr was obtained via mmap with this size
                    unsafe {
                        if libc::munmap(ptr as *mut libc::c_void, size) != 0 {
                            log::warn!("Failed to unmap cached region of size {}", size);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Get system page size (falls back to 4096 if `sysconf` fails or returns a
    /// non-power-of-two value).
    fn get_page_size() -> usize {
        // SAFETY: sysconf with _SC_PAGESIZE is always safe to call
        let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if raw > 0 && (raw as usize).is_power_of_two() {
            raw as usize
        } else {
            4096
        }
    }
}

impl Drop for MemoryMappedAllocator {
    fn drop(&mut self) {
        // Clean up all cached regions
        let _ = self.clear_cache();
    }
}

// SAFETY: MemoryMappedAllocator is Send because:
// 1. `min_mmap_size: usize` - Immutable primitive, trivially Send.
// 2. `page_size: usize` - Immutable primitive, trivially Send.
// 3. `region_cache: Mutex<HashMap<usize, Vec<*mut u8>>>` - Mutex is Send.
//    Raw pointers in the cache point to mmap'd regions, not thread-local data.
// 4. `total_allocated/total_freed/...` - AtomicU64 counters are Send.
unsafe impl Send for MemoryMappedAllocator {}

// SAFETY: MemoryMappedAllocator is Sync because:
// 1. `region_cache` - Protected by Mutex for exclusive access.
// 2. All atomic counters are inherently thread-safe.
// 3. mmap/munmap syscalls are thread-safe.
// 4. Immutable fields (min_mmap_size, page_size) are safe to read concurrently.
// The Mutex ensures serialized access to the region cache.
unsafe impl Sync for MemoryMappedAllocator {}

impl MmapAllocation {
    /// Get the allocated memory as a slice
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for size bytes, obtained via mmap, mapping valid for lifetime of MmapAllocation
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get the allocated memory as a mutable slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for size bytes, obtained via mmap, mapping valid for lifetime of MmapAllocation
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Get the size of the allocation
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the actual allocated size (rounded to page size)
    pub fn actual_size(&self) -> usize {
        self.actual_size
    }

    /// Get the memory as a typed pointer
    pub fn as_ptr<T>(&self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }

    /// Get mutable pointer to the allocation as a raw byte pointer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Is `addr` inside any mapping of this process, per /proc/self/maps?
    ///
    /// Per-address rather than a process-wide number from /proc/self/statm:
    /// `cargo test` runs the suite in parallel threads, and a process-wide
    /// total moves under any other test that maps memory at the same moment.
    #[cfg(target_os = "linux")]
    fn address_is_mapped(addr: usize) -> bool {
        std::fs::read_to_string("/proc/self/maps")
            .unwrap()
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .filter_map(|range| range.split_once('-'))
            .filter_map(|(start, end)| {
                Some((
                    usize::from_str_radix(start, 16).ok()?,
                    usize::from_str_radix(end, 16).ok()?,
                ))
            })
            .any(|(start, end)| (start..end).contains(&addr))
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dropping_an_allocation_releases_the_mapping() {
        // `MmapAllocation` owns an mmap region and is handed out by a safe
        // public method, but used to have no Drop: every disposal other than
        // handing it back to `MemoryMappedAllocator::deallocate` leaked the
        // whole mapping.
        //
        // Retried: `cargo test` runs the suite in parallel threads, and the
        // kernel is free to hand the hole we just freed straight to another
        // thread's mmap, which would leave the address mapped by someone else.
        // A leak fails every attempt; a reuse has to happen three times in a
        // row to be mistaken for one.
        let allocator = MemoryMappedAllocator::new(16 * 1024);
        const SIZE: usize = 4 * 1024 * 1024;
        const ATTEMPTS: usize = 3;

        let mut still_mapped = Vec::with_capacity(ATTEMPTS);
        for _ in 0..ATTEMPTS {
            let mut allocation = allocator.allocate(SIZE).unwrap();
            allocation.as_mut_slice()[0] = 1;
            let addr = allocation.as_ptr::<u8>() as usize;
            assert!(
                address_is_mapped(addr),
                "vacuous unless the allocation is mapped to begin with"
            );

            drop(allocation);

            if !address_is_mapped(addr) {
                return;
            }
            still_mapped.push(addr);
        }

        panic!(
            "{still_mapped:#x?} were all still mapped after being dropped: \
             the mapping leaked"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_deallocate_does_not_unmap_a_cached_region() {
        // The mirror of the test above: `deallocate` parks the region in the
        // allocator's cache, so it must *not* be unmapped, and the next
        // allocation of that size must get it back.
        let allocator = MemoryMappedAllocator::new(16 * 1024);
        const SIZE: usize = 4 * 1024 * 1024;

        let allocation = allocator.allocate(SIZE).unwrap();
        let addr = allocation.as_ptr::<u8>() as usize;
        allocator.deallocate(allocation).unwrap();

        assert!(
            address_is_mapped(addr),
            "a cached region was unmapped: the cache now hands out dead pointers"
        );
        assert_eq!(allocator.stats().munmap_calls, 0);

        let again = allocator.allocate(SIZE).unwrap();
        assert_eq!(again.as_ptr::<u8>() as usize, addr, "cache hit expected");
        assert_eq!(allocator.stats().cache_hits, 1);
        allocator.deallocate(again).unwrap();
    }

    /// C3.22 (S8-R2, MEDIUM). A fresh `mmap(MAP_ANONYMOUS)` returns
    /// kernel-zeroed pages, while a cache hit in `MemoryMappedAllocator::allocate`
    /// handed the recycled mapping straight back through the safe
    /// `MmapAllocation::as_slice` with the previous tenant's bytes still in it.
    #[test]
    fn test_cached_mmap_region_does_not_leak_previous_contents() {
        let allocator = MemoryMappedAllocator::new(4096);
        let mut first = allocator.allocate(16 * 1024).unwrap();
        first.as_mut_slice().fill(0xAB);
        allocator.deallocate(first).unwrap();

        let second = allocator.allocate(16 * 1024).unwrap();
        assert_eq!(allocator.stats().cache_hits, 1, "expected a region-cache hit");
        assert!(
            second.as_slice().iter().all(|&b| b == 0),
            "recycled MmapAllocation leaked previous tenant's 0xAB bytes through as_slice()"
        );
    }

    /// C3.22 (M3). `(size + page_size - 1) & !(page_size - 1)` overflowed for
    /// `size` within a page of `usize::MAX` (debug panic; release wrap to 0).
    #[test]
    fn test_allocate_rejects_overflowing_size() {
        let allocator = MemoryMappedAllocator::new(4096);
        assert!(
            allocator.allocate(usize::MAX).is_err(),
            "overflowing size must return Err, not panic or wrap to 0"
        );
    }

    #[test]
    fn test_mmap_allocator_creation() {
        let allocator = MemoryMappedAllocator::new(16 * 1024);
        assert!(allocator.should_use_mmap(20 * 1024));
        assert!(!allocator.should_use_mmap(8 * 1024));
    }

    #[test]
    fn test_mmap_allocation() {
        let allocator = MemoryMappedAllocator::default();
        let size = 64 * 1024; // 64KB

        let mut allocation = allocator.allocate(size).unwrap();
        assert_eq!(allocation.size(), size);
        assert!(allocation.actual_size() >= size);

        // Test that we can write to the memory
        let slice = allocation.as_mut_slice();
        slice[0] = 42;
        slice[size - 1] = 84;

        let slice = allocation.as_slice();
        assert_eq!(slice[0], 42);
        assert_eq!(slice[size - 1], 84);

        allocator.deallocate(allocation).unwrap();

        let stats = allocator.stats();
        assert_eq!(stats.total_allocated, size as u64);
        assert_eq!(stats.total_freed, size as u64);
        assert_eq!(stats.mmap_calls, 1);
    }

    #[test]
    fn test_mmap_cache() {
        let allocator = MemoryMappedAllocator::default();
        let size = 64 * 1024;

        // Allocate and deallocate to populate cache
        let allocation1 = allocator.allocate(size).unwrap();
        allocator.deallocate(allocation1).unwrap();

        let stats_before = allocator.stats();

        // Allocate again, should hit cache
        let allocation2 = allocator.allocate(size).unwrap();
        allocator.deallocate(allocation2).unwrap();

        let stats_after = allocator.stats();

        // Should have one cache hit
        assert_eq!(stats_after.cache_hits, stats_before.cache_hits + 1);
        // Should not have made additional mmap calls
        assert_eq!(stats_after.mmap_calls, stats_before.mmap_calls);
    }

    #[test]
    fn test_mmap_different_sizes() {
        let allocator = MemoryMappedAllocator::default();

        let sizes = vec![16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024];
        let mut allocations = Vec::new();

        // Allocate different sizes
        for size in &sizes {
            let allocation = allocator.allocate(*size).unwrap();
            assert_eq!(allocation.size(), *size);
            allocations.push(allocation);
        }

        // Deallocate all
        for allocation in allocations {
            allocator.deallocate(allocation).unwrap();
        }

        let stats = allocator.stats();
        assert_eq!(stats.mmap_calls, sizes.len() as u64);
        assert_eq!(stats.total_allocated, sizes.iter().sum::<usize>() as u64);
        assert_eq!(stats.total_freed, sizes.iter().sum::<usize>() as u64);
    }

    #[test]
    fn test_mmap_cache_limit() {
        let allocator = MemoryMappedAllocator::default();
        let size = 64 * 1024;

        // Allocate and deallocate more than cache limit
        for _ in 0..10 {
            let allocation = allocator.allocate(size).unwrap();
            allocator.deallocate(allocation).unwrap();
        }

        let stats = allocator.stats();
        // Should have some cached regions, but not more than the limit
        assert!(stats.cached_regions <= 4); // MAX_CACHED_REGIONS_PER_SIZE
        // Note: munmap_calls might be 0 if all allocations fit in cache during this test
        // This is acceptable as the cache is working correctly
    }

    #[test]
    fn test_clear_cache() {
        let allocator = MemoryMappedAllocator::default();
        let size = 64 * 1024;

        // Populate cache
        let allocation = allocator.allocate(size).unwrap();
        allocator.deallocate(allocation).unwrap();

        let stats_before = allocator.stats();
        assert!(stats_before.cached_regions > 0);

        // Clear cache
        allocator.clear_cache().unwrap();

        let stats_after = allocator.stats();
        assert_eq!(stats_after.cached_regions, 0);
        assert!(stats_after.munmap_calls > stats_before.munmap_calls);
    }

    #[test]
    fn test_invalid_allocation_size() {
        let allocator = MemoryMappedAllocator::new(16 * 1024);

        // Too small for mmap
        let result = allocator.allocate(8 * 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_concurrent_cache_no_bypass() {
        use std::sync::Arc;
        use std::thread;

        let allocator = Arc::new(MemoryMappedAllocator::default());
        let size = 64 * 1024;

        // Seed the cache: allocate and deallocate 4 regions (fills one size class)
        let mut seed = Vec::new();
        for _ in 0..4 {
            seed.push(allocator.allocate(size).unwrap());
        }
        for alloc in seed {
            allocator.deallocate(alloc).unwrap();
        }

        let stats_before = allocator.stats();
        assert_eq!(stats_before.cached_regions, 4);
        let mmap_before = stats_before.mmap_calls;

        // Spawn threads that each take one cached region and return it.
        // With blocking lock, every allocate must hit the cache — no new mmap calls.
        let mut handles = Vec::new();
        for _ in 0..4 {
            let alloc = Arc::clone(&allocator);
            handles.push(thread::spawn(move || {
                let a = alloc.allocate(size).unwrap();
                // Brief hold to create contention window on deallocate
                thread::yield_now();
                alloc.deallocate(a).unwrap();
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let stats_after = allocator.stats();
        // All 4 allocations should have been served from cache — zero new mmap calls
        assert_eq!(
            stats_after.mmap_calls,
            mmap_before,
            "cache was bypassed under contention: {} new mmap calls",
            stats_after.mmap_calls - mmap_before
        );
    }
}
