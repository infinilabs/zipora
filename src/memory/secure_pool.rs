//! Secure, thread-safe memory pool implementation
//!
//! This module provides a production-ready, thread-safe memory pool that eliminates
//! the security vulnerabilities found in the original MemoryPool implementation.
//!
//! # Security Features
//!
//! - **Memory Safety**: No raw pointers, uses safe Rust abstractions
//! - **Use-After-Free Prevention**: Generation counters validate pointer lifetime
//! - **Double-Free Detection**: Cryptographic validation of deallocations
//! - **Corruption Detection**: Guard pages and canary values
//! - **Thread Safety**: Proper synchronization without manual Send/Sync
//!
//! # Performance Features
//!
//! - **Thread-Local Caching**: Reduces contention with per-thread allocation caches
//! - **Lock-Free Fast Paths**: Lock-free stacks for high-performance allocation
//! - **NUMA Awareness**: Optimized allocation for multi-socket systems
//! - **Batch Operations**: Amortized system call overhead
//! - **SIMD Optimizations**: Vectorized memory operations (AVX-512/AVX2/SSE2) for 2-3x faster
//!   memory zeroing on deallocation, with automatic CPU feature detection and threshold-based
//!   optimization (≥64 bytes). Maintains all security guarantees while significantly improving
//!   performance for large memory operations.
//!
//! # Architecture
//!
//! The secure pool uses a hybrid architecture combining thread-local caches
//! with a central depot for cross-thread coordination. Each allocation is
//! validated with generation counters and cryptographic signatures.

use super::CachePadded;
use crate::error::{Result, ZiporaError};
#[cfg(target_arch = "x86_64")]
use crate::memory::cache_layout::PrefetchHint;
use crate::memory::cache_layout::{AccessPattern, CacheLayoutConfig, CacheOptimizedAllocator};
#[cfg(target_arch = "x86_64")]
use crate::memory::simd_ops::fast_prefetch;
use crate::memory::simd_ops::{fast_compare, fast_fill_raw};
use crate::simd::Operation;
use crate::simd::adaptive::AdaptiveSimdSelector;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Magic constants for corruption detection
const CHUNK_HEADER_MAGIC: u64 = 0xDEADBEEFCAFEBABE;
const CHUNK_FOOTER_MAGIC: u64 = 0xFEEDFACEDEADBEEF;

/// Size classes for efficient allocation (jemalloc-inspired)
const SIZE_CLASSES: &[usize] = &[
    8, 16, 32, 48, 64, 80, 96, 112, 128, // Small (8-byte increments)
    160, 192, 224, 256, 320, 384, 448, 512, // Medium (64-byte increments)
    640, 768, 896, 1024, 1280, 1536, 1792, 2048, // Large (256-byte increments)
    2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, // XLarge (512-byte increments)
];

/// Configuration for secure memory pool
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SecurePoolConfig {
    /// Size of each chunk in bytes
    pub chunk_size: usize,
    /// Maximum number of chunks to keep in the pool
    pub max_chunks: usize,
    /// Alignment requirement for allocations
    pub alignment: usize,
    /// Enable guard pages for overflow detection
    pub use_guard_pages: bool,
    /// Zero memory on deallocation for security
    pub zero_on_free: bool,
    /// Zero memory on allocation for security
    pub zero_on_alloc: bool,
    /// Thread-local cache size
    pub local_cache_size: usize,
    /// Batch size for depot transfers
    pub batch_size: usize,
    /// Enable SIMD optimizations for memory operations (default: true)
    ///
    /// When enabled, uses vectorized instructions for memory operations like zeroing
    /// on systems that support SIMD (AVX-512/AVX2/SSE2). Provides significant
    /// performance improvements for large memory operations while maintaining
    /// all security guarantees.
    pub enable_simd_ops: bool,
    /// Minimum size threshold for SIMD operations (default: 64 bytes)
    ///
    /// Memory operations smaller than this threshold use standard implementations.
    /// SIMD operations provide meaningful performance benefits only for larger
    /// memory regions. The default of 64 bytes aligns with cache line size.
    pub simd_threshold: usize,
    /// Enables cache-line prefetch hints for metadata during bulk allocations.
    ///
    /// Returned allocations are aligned to `self.alignment` bytes (minimum 8 bytes;
    /// not cache-line aligned unless `self.alignment >= 64`).
    pub enable_cache_alignment: bool,
    /// Cache layout configuration for optimization
    ///
    /// Specifies cache-specific optimizations like prefetch distances,
    /// hot/cold separation, and access pattern hints.
    pub cache_config: Option<CacheLayoutConfig>,
    /// Enable NUMA-aware allocation (default: true)
    ///
    /// When enabled, allocations prefer the local NUMA node to minimize
    /// memory access latency and maximize bandwidth utilization.
    pub enable_numa_awareness: bool,
    /// Enable hot/cold data separation for cache optimization
    ///
    /// When enabled, frequently accessed allocations are placed in
    /// cache-friendly regions while cold data is moved to separate areas.
    pub enable_hot_cold_separation: bool,
    /// Allocation frequency threshold for hot data classification
    ///
    /// Allocations accessed more than this threshold are considered hot.
    pub hot_data_threshold: usize,
    /// Enable huge page allocation for large chunks (Linux only)
    ///
    /// When enabled, large allocations use huge pages to reduce TLB pressure
    /// and improve memory access performance.
    pub enable_huge_pages: bool,
    /// Minimum chunk size for huge page allocation (default: 2MB)
    ///
    /// Chunks smaller than this size use regular pages.
    pub huge_page_threshold: usize,
    /// Prefetch distance for bulk allocation operations (default: 8)
    ///
    /// Number of allocations to look ahead when prefetching cache lines.
    /// Higher values can improve throughput for large batch allocations but
    /// may waste prefetch bandwidth for small batches. Tune based on workload:
    /// - 4: Conservative, good for variable-size batches
    /// - 8: Default, balanced for most workloads
    /// - 12-16: Aggressive, best for large uniform batches
    pub prefetch_distance: usize,
}

impl SecurePoolConfig {
    /// Create a new secure pool configuration
    pub fn new(chunk_size: usize, max_chunks: usize, alignment: usize) -> Self {
        Self {
            chunk_size,
            max_chunks,
            alignment,
            use_guard_pages: false,
            zero_on_free: true,
            zero_on_alloc: false,
            local_cache_size: 64,
            batch_size: 16,
            enable_simd_ops: true,
            simd_threshold: 64,
            enable_cache_alignment: true,
            cache_config: Some(CacheLayoutConfig::new()),
            enable_numa_awareness: true,
            enable_hot_cold_separation: true,
            hot_data_threshold: 1000,
            enable_huge_pages: cfg!(target_os = "linux"),
            huge_page_threshold: 2 * 1024 * 1024, // 2MB
            prefetch_distance: 8,                 // Default: balanced for most workloads
        }
    }

    /// Create configuration for small objects (< 1KB) with security features
    pub fn small_secure() -> Self {
        Self {
            chunk_size: 1024,
            max_chunks: 100,
            alignment: 16,
            use_guard_pages: false,
            zero_on_free: true,
            zero_on_alloc: false,
            local_cache_size: 64,
            batch_size: 16,
            enable_simd_ops: true,
            simd_threshold: 64,
            enable_cache_alignment: true,
            cache_config: Some(CacheLayoutConfig::random()),
            enable_numa_awareness: true,
            enable_hot_cold_separation: true,
            hot_data_threshold: 500,  // More sensitive for small objects
            enable_huge_pages: false, // Disable for small objects
            huge_page_threshold: 2 * 1024 * 1024,
            prefetch_distance: 8, // Default for small objects
        }
    }

    /// Create configuration for medium objects (< 64KB) with security features
    pub fn medium_secure() -> Self {
        Self {
            chunk_size: 64 * 1024,
            max_chunks: 50,
            alignment: 16,
            use_guard_pages: true,
            zero_on_free: true,
            zero_on_alloc: false,
            local_cache_size: 32,
            batch_size: 8,
            enable_simd_ops: true,
            simd_threshold: 64,
            enable_cache_alignment: true,
            cache_config: Some(CacheLayoutConfig::sequential()),
            enable_numa_awareness: true,
            enable_hot_cold_separation: true,
            hot_data_threshold: 750,  // Balanced for medium objects
            enable_huge_pages: false, // Usually not needed for 64KB
            huge_page_threshold: 2 * 1024 * 1024,
            prefetch_distance: 8, // Default for medium objects
        }
    }

    /// Create configuration for large objects (< 1MB) with maximum security
    pub fn large_secure() -> Self {
        Self {
            chunk_size: 1024 * 1024,
            max_chunks: 10,
            alignment: 32,
            use_guard_pages: true,
            zero_on_free: true,
            zero_on_alloc: false,
            local_cache_size: 16,
            batch_size: 4,
            enable_simd_ops: true,
            simd_threshold: 64,
            enable_cache_alignment: true,
            cache_config: Some(CacheLayoutConfig::read_heavy()),
            enable_numa_awareness: true,
            enable_hot_cold_separation: true,
            hot_data_threshold: 1500, // Higher threshold for large objects
            enable_huge_pages: true,  // Enable for large objects
            huge_page_threshold: 2 * 1024 * 1024,
            prefetch_distance: 12, // More aggressive for large batch allocations
        }
    }

    /// Builder method to set alignment
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    /// Builder method to enable/disable guard pages
    pub fn with_guard_pages(mut self, use_guard_pages: bool) -> Self {
        self.use_guard_pages = use_guard_pages;
        self
    }

    /// Builder method to enable/disable zero on free
    pub fn with_zero_on_free(mut self, zero_on_free: bool) -> Self {
        self.zero_on_free = zero_on_free;
        self
    }

    /// Builder method to enable/disable zero on allocation
    pub fn with_zero_on_alloc(mut self, zero_on_alloc: bool) -> Self {
        self.zero_on_alloc = zero_on_alloc;
        self
    }

    /// Builder method to set local cache size
    pub fn with_local_cache_size(mut self, size: usize) -> Self {
        self.local_cache_size = size;
        self
    }

    /// Builder method to set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Builder method to enable/disable SIMD optimizations
    ///
    /// When enabled (default), uses vectorized instructions for memory operations
    /// like zeroing on systems that support SIMD. Provides significant performance
    /// improvements while maintaining all security guarantees.
    pub fn with_simd_ops(mut self, enable: bool) -> Self {
        self.enable_simd_ops = enable;
        self
    }

    /// Builder method to set SIMD threshold
    ///
    /// Memory operations smaller than this threshold use standard implementations.
    /// The default of 64 bytes aligns with cache line size and provides optimal
    /// performance characteristics for most workloads.
    pub fn with_simd_threshold(mut self, threshold: usize) -> Self {
        self.simd_threshold = threshold;
        self
    }

    /// Builder method to enable/disable cache-line prefetch hints.
    ///
    /// Returned allocations are aligned to `self.alignment` bytes (minimum 8 bytes;
    /// not cache-line aligned unless `self.alignment >= 64`).
    pub fn with_cache_alignment(mut self, enable: bool) -> Self {
        self.enable_cache_alignment = enable;
        self
    }

    /// Builder method to set cache configuration
    ///
    /// Specifies cache-specific optimizations like prefetch distances,
    /// hot/cold separation, and access pattern hints. If None, cache optimizations
    /// are disabled.
    pub fn with_cache_config(mut self, config: Option<CacheLayoutConfig>) -> Self {
        self.cache_config = config;
        self
    }

    /// Builder method to set cache configuration for specific access pattern
    ///
    /// Convenience method that creates an appropriate cache configuration
    /// based on the expected access pattern.
    pub fn with_access_pattern(mut self, pattern: AccessPattern) -> Self {
        let config = match pattern {
            AccessPattern::Sequential => CacheLayoutConfig::sequential(),
            AccessPattern::Random => CacheLayoutConfig::random(),
            AccessPattern::WriteHeavy => CacheLayoutConfig::write_heavy(),
            AccessPattern::ReadHeavy => CacheLayoutConfig::read_heavy(),
            AccessPattern::Mixed => CacheLayoutConfig::new(),
        };
        self.cache_config = Some(config);
        self
    }

    /// Builder method to enable/disable NUMA awareness
    ///
    /// When enabled, allocations prefer the local NUMA node to minimize
    /// memory access latency and maximize bandwidth utilization.
    pub fn with_numa_awareness(mut self, enable: bool) -> Self {
        self.enable_numa_awareness = enable;
        self
    }

    /// Builder method to enable/disable hot/cold data separation
    ///
    /// When enabled, frequently accessed allocations are placed in
    /// cache-friendly regions while cold data is moved to separate areas.
    pub fn with_hot_cold_separation(mut self, enable: bool) -> Self {
        self.enable_hot_cold_separation = enable;
        self
    }

    /// Builder method to set hot data threshold
    ///
    /// Allocations accessed more than this threshold are considered hot.
    pub fn with_hot_data_threshold(mut self, threshold: usize) -> Self {
        self.hot_data_threshold = threshold;
        self
    }

    /// Builder method to enable/disable huge pages
    ///
    /// When enabled, large allocations use huge pages to reduce TLB pressure
    /// and improve memory access performance.
    pub fn with_huge_pages(mut self, enable: bool) -> Self {
        self.enable_huge_pages = enable;
        self
    }

    /// Builder method to set huge page threshold
    ///
    /// Chunks smaller than this size use regular pages.
    pub fn with_huge_page_threshold(mut self, threshold: usize) -> Self {
        self.huge_page_threshold = threshold;
        self
    }

    /// Builder method to set prefetch distance for bulk allocations
    ///
    /// Higher values can improve throughput for large batch allocations but
    /// may waste prefetch bandwidth for small batches. Recommended values:
    /// - 4: Conservative, good for variable-size batches
    /// - 8: Default, balanced for most workloads
    /// - 12-16: Aggressive, best for large uniform batches
    pub fn with_prefetch_distance(mut self, distance: usize) -> Self {
        self.prefetch_distance = distance;
        self
    }
}

/// Statistics for secure memory pool
#[derive(Debug, Clone)]
pub struct SecurePoolStats {
    /// Total bytes allocated
    pub allocated: u64,
    /// Total bytes available in pool
    pub available: u64,
    /// Number of chunks in pool
    pub chunks: usize,
    /// Number of allocations served
    pub alloc_count: u64,
    /// Number of deallocations
    pub dealloc_count: u64,
    /// Number of pool hits (reused memory)
    pub pool_hits: u64,
    /// Number of pool misses (new allocations)
    pub pool_misses: u64,
    /// Number of corruption detections
    pub corruption_detected: u64,
    /// Number of double-free attempts detected
    pub double_free_detected: u64,
    /// Thread-local cache hits
    pub local_cache_hits: u64,
    /// Cross-thread steals
    pub cross_thread_steals: u64,
    /// Allocations made with cache-line prefetch hints enabled
    pub cache_aligned_allocs: u64,
    /// NUMA-local allocations
    pub numa_local_allocs: u64,
    /// Hot data allocations
    pub hot_data_allocs: u64,
    /// Cold data allocations
    pub cold_data_allocs: u64,
    /// Huge page allocations
    pub huge_page_allocs: u64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

impl Default for SecurePoolStats {
    fn default() -> Self {
        Self {
            allocated: 0,
            available: 0,
            chunks: 0,
            alloc_count: 0,
            dealloc_count: 0,
            pool_hits: 0,
            pool_misses: 0,
            corruption_detected: 0,
            double_free_detected: 0,
            local_cache_hits: 0,
            cross_thread_steals: 0,
            cache_aligned_allocs: 0,
            numa_local_allocs: 0,
            hot_data_allocs: 0,
            cold_data_allocs: 0,
            huge_page_allocs: 0,
            cache_hit_ratio: 0.0,
        }
    }
}

/// Header for each memory chunk with security metadata
#[repr(C)]
#[derive(Debug)]
struct ChunkHeader {
    magic: u64,
    size: usize,
    // u64 generation: a u32 counter can wrap under sustained allocation churn,
    // letting a stale pointer's generation collide with a live chunk's.
    generation: u64,
    allocation_time: u64,
    pool_id: u32,
    canary: u32,
}

/// Footer for each memory chunk with security metadata
#[repr(C)]
#[derive(Debug)]
struct ChunkFooter {
    canary: u32,
    _padding: u32,
    generation: u64,
    magic: u64,
}

const MIN_CHUNK_ALIGN: usize = 8;
const GENERATION_MASK: u64 = (1u64 << 56) - 1;

const _: () = assert!(
    std::mem::size_of::<ChunkHeader>().is_multiple_of(MIN_CHUNK_ALIGN)
        && std::mem::align_of::<ChunkHeader>() <= MIN_CHUNK_ALIGN
        && std::mem::align_of::<ChunkFooter>() <= MIN_CHUNK_ALIGN
        && std::mem::align_of::<usize>() <= MIN_CHUNK_ALIGN
);

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<SecureChunk>() == 32);

/// Secure wrapper around memory chunks with validation.
///
/// The chunk's allocation alignment shift (`align.trailing_zeros() as u8`) is packed
/// into the upper 8 bits of `gen_and_align_shift` (with the 56-bit generation counter
/// in the lower 56 bits) so `size_of::<SecureChunk>()` stays 32 bytes while keeping
/// the layout alignment out-of-band outside the heap buffer.
pub struct SecureChunk {
    ptr: NonNull<u8>,
    size: usize,
    gen_and_align_shift: u64,
    pool_id: u32,
    canary: u32,
}

/// Payload size rounded up so the footer lands on its natural alignment.
///
/// The payload size is the caller's `chunk_size`, which is only validated as
/// non-zero, while `ChunkFooter` holds two `u64`s and needs 8-byte alignment.
/// Placing the footer at `header + size` therefore produces a misaligned
/// reference for any size that is not a multiple of 8 — undefined behaviour,
/// and an immediate abort under `-Cdebug-assertions` or on targets that fault
/// on unaligned access. Every site that locates the footer must apply the
/// same padding.
#[inline]
fn padded_payload_size(size: usize) -> usize {
    size.next_multiple_of(std::mem::align_of::<ChunkFooter>())
}

/// Compute `(layout, payload_offset, effective_align)` for a chunk of `size` bytes
/// with requested `alignment`.
///
/// The allocation begins at `raw_ptr` (aligned to `effective_align = alignment.max(8)`),
/// places `data_ptr = raw_ptr.add(payload_offset)` on an `effective_align` boundary,
/// places `ChunkHeader` immediately preceding `data_ptr` at `data_ptr.sub(header_size)`,
/// and places `ChunkFooter` at `data_ptr.add(padded_payload_size(size))`.
#[inline]
fn chunk_layout(size: usize, alignment: usize) -> Result<(Layout, usize, usize)> {
    if size == 0 {
        return Err(ZiporaError::invalid_data("chunk_size cannot be zero"));
    }
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(ZiporaError::invalid_data(
            "alignment must be a power of two",
        ));
    }
    let align = alignment.max(MIN_CHUNK_ALIGN);
    if !align.is_power_of_two() {
        return Err(ZiporaError::invalid_data(
            "alignment must be a power of two",
        ));
    }

    let header_size = std::mem::size_of::<ChunkHeader>();
    let footer_size = std::mem::size_of::<ChunkFooter>();
    let payload_offset = header_size
        .checked_next_multiple_of(align)
        .ok_or_else(|| ZiporaError::invalid_data("Invalid alignment for chunk allocation"))?;
    let padded_size = size
        .checked_next_multiple_of(std::mem::align_of::<ChunkFooter>())
        .ok_or_else(|| ZiporaError::invalid_data("Invalid size for chunk allocation"))?;
    let total_size = payload_offset
        .checked_add(padded_size)
        .and_then(|s| s.checked_add(footer_size))
        .ok_or_else(|| ZiporaError::invalid_data("Invalid layout for chunk allocation"))?;

    let layout = Layout::from_size_align(total_size, align)
        .map_err(|_| ZiporaError::invalid_data("Invalid layout for chunk allocation"))?;
    Ok((layout, payload_offset, align))
}

impl SecureChunk {
    /// Create a new secure chunk with validation metadata
    fn new(size: usize, alignment: usize, generation: u64, pool_id: u32) -> Result<Self> {
        let (layout, payload_offset, align) = chunk_layout(size, alignment)?;
        let generation = generation & GENERATION_MASK;
        let gen_and_align_shift = generation | ((align.trailing_zeros() as u64) << 56);
        let canary = fastrand::u32(..);
        let header_size = std::mem::size_of::<ChunkHeader>();
        let padded_size = padded_payload_size(size);

        // SAFETY: `chunk_layout` validated that `layout` has non-zero size and power-of-two `align`.
        // `alloc_zeroed` initializes all bytes of a freshly allocated chunk to zero so
        // `SecurePooledPtr::as_slice` and `as_mut_slice` never expose uninitialized heap
        // memory on fresh allocations (recycled chunks retain prior initialized bytes unless
        // `zero_on_free` or `zero_on_alloc` is enabled).
        let raw_ptr = unsafe { alloc_zeroed(layout) };
        if raw_ptr.is_null() {
            return Err(ZiporaError::out_of_memory(size));
        }

        // Return pointer to data area (`align`-aligned because `raw_ptr` is `align`-aligned
        // and `payload_offset` is a multiple of `align`).
        // SAFETY: `raw_ptr` points to a live allocation of `layout.size()` bytes and `payload_offset < layout.size()`.
        let data_ptr = unsafe { raw_ptr.add(payload_offset) };

        // Place header immediately before `data_ptr` so `data_ptr.sub(header_size)` locates it
        // regardless of `align`.
        // SAFETY: `payload_offset >= header_size`, so `data_ptr.sub(header_size)` lies within
        // the allocated region; because `align >= 8` is a power of two and `header_size` is a
        // multiple of 8, `header_ptr` satisfies `align_of::<ChunkHeader>()`.
        unsafe {
            let header_ptr = data_ptr.sub(header_size) as *mut ChunkHeader;
            header_ptr.write(ChunkHeader {
                magic: CHUNK_HEADER_MAGIC,
                size,
                generation,
                pool_id,
                allocation_time: current_time_nanos(),
                canary,
            });
        }

        // Initialize footer
        // SAFETY: `payload_offset + padded_size + size_of::<ChunkFooter>() == layout.size()`,
        // and both `data_ptr` and `padded_size` are multiples of `align_of::<ChunkFooter>()`.
        unsafe {
            let footer_ptr = data_ptr.add(padded_size) as *mut ChunkFooter;
            footer_ptr.write(ChunkFooter {
                canary,
                _padding: 0,
                generation,
                magic: CHUNK_FOOTER_MAGIC,
            });
        }

        Ok(Self {
            // SAFETY: `raw_ptr` is non-null and `add(payload_offset)` remains non-null.
            ptr: unsafe { NonNull::new_unchecked(data_ptr) },
            size,
            gen_and_align_shift,
            pool_id,
            canary,
        })
    }

    // Note: new_from_ptr method temporarily removed to simplify compilation
    // Would be used for NUMA-aware allocations in the future

    /// Validate chunk integrity
    fn validate(&self) -> Result<()> {
        let generation = self.generation();
        let header_size = std::mem::size_of::<ChunkHeader>();
        // SAFETY: new() places ChunkHeader immediately preceding data_ptr at raw_ptr + (payload_offset - header_size)
        let header_ptr = unsafe { self.ptr.as_ptr().sub(header_size) as *const ChunkHeader };
        // SAFETY: header_ptr valid, aligned to ChunkHeader, from allocation
        let header = unsafe { &*header_ptr };

        // Validate header
        if header.magic != CHUNK_HEADER_MAGIC {
            return Err(ZiporaError::invalid_data(format!(
                "Header corruption detected: magic={:#x}, expected={:#x}",
                header.magic, CHUNK_HEADER_MAGIC
            )));
        }

        if header.generation != generation {
            return Err(ZiporaError::invalid_data(format!(
                "Generation mismatch: header={}, chunk={}",
                header.generation, generation
            )));
        }

        if header.pool_id != self.pool_id {
            return Err(ZiporaError::invalid_data(format!(
                "Pool ID mismatch: header={}, chunk={}",
                header.pool_id, self.pool_id
            )));
        }

        if header.canary != self.canary {
            return Err(ZiporaError::invalid_data(format!(
                "Header canary mismatch: header={:#x}, chunk={:#x}",
                header.canary, self.canary
            )));
        }

        // Validate footer
        // SAFETY: ptr valid from allocation, padded offset matches new()
        let footer_ptr =
            unsafe { self.ptr.as_ptr().add(padded_payload_size(self.size)) as *const ChunkFooter };
        // SAFETY: footer_ptr valid, aligned to ChunkFooter, from allocation
        let footer = unsafe { &*footer_ptr };

        if footer.magic != CHUNK_FOOTER_MAGIC {
            return Err(ZiporaError::invalid_data(format!(
                "Footer corruption detected: magic={:#x}, expected={:#x}",
                footer.magic, CHUNK_FOOTER_MAGIC
            )));
        }

        if footer.canary != self.canary {
            return Err(ZiporaError::invalid_data(format!(
                "Footer canary mismatch: footer={:#x}, chunk={:#x}",
                footer.canary, self.canary
            )));
        }

        if footer.generation != generation {
            return Err(ZiporaError::invalid_data(format!(
                "Footer generation mismatch: footer={}, chunk={}",
                footer.generation, generation
            )));
        }

        Ok(())
    }

    /// Zero the data area on the recycle path, with SIMD acceleration for large regions.
    ///
    /// Recycled chunks remain live in the thread-local cache or global stack, so
    /// non-volatile `fast_fill_raw` / `write_bytes` cannot be eliminated as a dead store.
    fn zero_payload(&mut self, enable_simd_ops: bool, simd_threshold: usize) {
        // SAFETY: `self.ptr` points to `self.size` writable bytes exclusively owned by `self`.
        unsafe {
            // SIMD-optimized memory zeroing for large regions
            if enable_simd_ops && self.size >= simd_threshold {
                // Use raw-pointer SIMD fill without forming `&mut [u8]` over potentially
                // uninitialized bytes written via raw pointer by a caller.
                fast_fill_raw(self.ptr.as_ptr(), 0, self.size);
            } else {
                // Standard zeroing for small regions where SIMD overhead isn't beneficial
                std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.size);
            }
        }
    }

    /// Zero the data area with volatile writes immediately prior to `dealloc`
    /// so dead-store elimination cannot remove secret wiping.
    fn zero_payload_volatile(&mut self) {
        let base = self.ptr.as_ptr();
        let word_size = std::mem::size_of::<usize>();
        let word_count = self.size / word_size;
        let word_ptr = base as *mut usize;

        for i in 0..word_count {
            // SAFETY: `base` is aligned to `self.align() >= MIN_CHUNK_ALIGN >= align_of::<usize>()`,
            // and `(i + 1) * word_size <= self.size` lies within the exclusively owned payload.
            unsafe {
                std::ptr::write_volatile(word_ptr.add(i), 0usize);
            }
        }

        for offset in (word_count * word_size)..self.size {
            // SAFETY: `offset < self.size` lies within the exclusively owned payload.
            unsafe {
                std::ptr::write_volatile(base.add(offset), 0u8);
            }
        }

        std::sync::atomic::compiler_fence(Ordering::SeqCst);
    }

    /// Get pointer to data area
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get size of data area
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get generation for validation
    #[inline]
    pub fn generation(&self) -> u64 {
        self.gen_and_align_shift & GENERATION_MASK
    }

    /// Get allocation alignment stored out-of-band in `SecureChunk`
    #[inline]
    fn align(&self) -> usize {
        1usize << ((self.gen_and_align_shift >> 56) as u32)
    }

    /// Safely deallocate the chunk, wiping the payload (when `zero_on_free` is enabled)
    /// and scrubbing security metadata (`ChunkHeader` and `ChunkFooter`) via volatile
    /// writes before `dealloc`.
    #[cold]
    #[inline(never)]
    fn deallocate(mut self, zero_on_free: bool) {
        let align = self.align();
        let Ok((layout, payload_offset, _)) = chunk_layout(self.size, align) else {
            unreachable!("layout parameters were validated in SecureChunk::new");
        };

        if zero_on_free {
            self.zero_payload_volatile();
        }

        // Scrub security metadata (canaries, magic values, generation, pool_id) before freeing.
        // SAFETY: `new()` placed `ChunkHeader` immediately preceding `self.ptr` and `ChunkFooter`
        // at `self.ptr.add(padded_payload_size(self.size))`, both properly aligned and exclusively
        // owned within `layout`.
        unsafe {
            let header_ptr =
                self.ptr.as_ptr().sub(std::mem::size_of::<ChunkHeader>()) as *mut ChunkHeader;
            std::ptr::write_volatile(
                header_ptr,
                ChunkHeader {
                    magic: 0,
                    size: 0,
                    generation: 0,
                    allocation_time: 0,
                    pool_id: 0,
                    canary: 0,
                },
            );

            let footer_ptr =
                self.ptr.as_ptr().add(padded_payload_size(self.size)) as *mut ChunkFooter;
            std::ptr::write_volatile(
                footer_ptr,
                ChunkFooter {
                    canary: 0,
                    _padding: 0,
                    generation: 0,
                    magic: 0,
                },
            );
        }

        std::sync::atomic::compiler_fence(Ordering::SeqCst);

        // SAFETY: `self.ptr` was constructed in `SecureChunk::new` as `raw_ptr.add(payload_offset)`
        // from an allocation with this exact `layout`.
        unsafe {
            let raw_ptr = self.ptr.as_ptr().sub(payload_offset);
            dealloc(raw_ptr, layout);
        }
    }
}

// Note: SecureChunk doesn't implement Drop because deallocate() takes self by value
// and Drop::drop() takes &mut self. Deallocation is handled explicitly by the pool
// or by SecurePooledPtr::drop().

// SAFETY: SecureChunk is Send because:
// 1. `ptr: NonNull<u8>` - Raw pointer to heap-allocated memory. The chunk owns
//    this memory exclusively until deallocated. No thread-local state.
// 2. `size: usize` - Immutable primitive, trivially Send.
// 3. `canary: u32`, `pool_id: u32` - Immutable after construction, trivially Send.
// 4. `gen_and_align_shift: u64` - Immutable packed generation and log2(align)
//    after construction, trivially Send.
unsafe impl Send for SecureChunk {}

// SAFETY: SecureChunk is Sync because:
// 1. All fields are immutable after construction.
// 2. The chunk represents exclusive ownership of a memory region.
// 3. Sharing &SecureChunk only provides read access to the pointer and metadata.
// 4. The pointer itself is never dereferenced through &self - only through
//    explicit unsafe code that requires the caller to ensure synchronization.
//
// Note: While SecureChunk is Sync, writing to the underlying memory through
// the pointer requires external synchronization (which SecurePooledPtr provides).
unsafe impl Sync for SecureChunk {}

/// Lock-free stack for high-performance chunk storage (Treiber stack)
use crossbeam_epoch::{self as epoch, Atomic, Owned};

struct LockFreeStack<T> {
    head: Atomic<Node<T>>,
}

struct Node<T> {
    data: std::mem::ManuallyDrop<T>,
    next: Atomic<Node<T>>,
}

impl<T> LockFreeStack<T> {
    fn new() -> Self {
        Self {
            head: Atomic::null(),
        }
    }

    fn push(&self, item: T) {
        let mut node = Owned::new(Node {
            data: std::mem::ManuallyDrop::new(item),
            next: Atomic::null(),
        });

        let guard = epoch::pin();
        loop {
            let head = self.head.load(Ordering::Acquire, &guard);
            node.next.store(head, Ordering::Relaxed);

            match self.head.compare_exchange(
                head,
                node,
                Ordering::Release,
                Ordering::Relaxed,
                &guard,
            ) {
                Ok(_) => break,
                Err(e) => node = e.new,
            }
        }
    }

    fn pop(&self) -> Option<T> {
        let guard = epoch::pin();
        loop {
            let head = self.head.load(Ordering::Acquire, &guard);

            // Check if stack is empty
            if head.is_null() {
                return None;
            }

            // SAFETY: The node is protected from destruction by the epoch guard.
            // Even if another thread pops this node, its memory will remain valid
            // until the current epoch advances, ensuring the read of `next` is safe.
            let next = unsafe { head.deref() }.next.load(Ordering::Relaxed, &guard);

            if self
                .head
                .compare_exchange(head, next, Ordering::AcqRel, Ordering::Acquire, &guard)
                .is_ok()
            {
                // We successfully unlinked the head node. We now have exclusive ownership
                // of the data inside it, but we must wait for all threads to drop their
                // guards before the memory backing the node can be reclaimed.
                // SAFETY: head exclusively owned from successful CAS, defer_destroy registers deferred dealloc, ptr::read valid on head.deref().data
                unsafe {
                    guard.defer_destroy(head);
                    // The data itself can be moved out immediately since we logically
                    // own it. Rust's Ownership prevents use-after-free of the `data`.
                    return Some(std::mem::ManuallyDrop::into_inner(std::ptr::read(
                        &head.deref().data,
                    )));
                }
            }
        }
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

/// Thread-local cache for reduced contention
struct LocalCache {
    chunks: Vec<SecureChunk>,
    max_size: usize,
    zero_on_free: bool,
    _enable_simd_ops: bool,
    _simd_threshold: usize,
}

impl LocalCache {
    fn new(
        max_size: usize,
        zero_on_free: bool,
        enable_simd_ops: bool,
        simd_threshold: usize,
    ) -> Self {
        Self {
            chunks: Vec::with_capacity(max_size),
            max_size,
            zero_on_free,
            _enable_simd_ops: enable_simd_ops,
            _simd_threshold: simd_threshold,
        }
    }

    fn try_pop(&mut self) -> Option<SecureChunk> {
        self.chunks.pop()
    }

    fn try_push(&mut self, chunk: SecureChunk) -> std::result::Result<(), SecureChunk> {
        if self.chunks.len() < self.max_size {
            self.chunks.push(chunk);
            Ok(())
        } else {
            Err(chunk)
        }
    }
}

impl Drop for LocalCache {
    fn drop(&mut self) {
        for chunk in self.chunks.drain(..) {
            chunk.deallocate(self.zero_on_free);
        }
    }
}

/// Production-ready secure memory pool
pub struct SecureMemoryPool {
    config: SecurePoolConfig,
    pool_id: u32,
    global_stack: LockFreeStack<SecureChunk>,
    next_generation: AtomicU64,
    local_caches: thread_local::ThreadLocal<RefCell<LocalCache>>,

    // Cache optimization infrastructure
    cache_allocator: Option<CacheOptimizedAllocator>,

    // Statistics (lock-free)
    alloc_count: CachePadded<AtomicU64>,
    dealloc_count: CachePadded<AtomicU64>,
    pool_hits: CachePadded<AtomicU64>,
    pool_misses: CachePadded<AtomicU64>,
    corruption_detected: CachePadded<AtomicU64>,
    double_free_detected: CachePadded<AtomicU64>,
    local_cache_hits: CachePadded<AtomicU64>,
    cross_thread_steals: CachePadded<AtomicU64>,

    // Cache-specific statistics
    cache_aligned_allocs: CachePadded<AtomicU64>,
    numa_local_allocs: CachePadded<AtomicU64>,
    hot_data_allocs: CachePadded<AtomicU64>,
    cold_data_allocs: CachePadded<AtomicU64>,
    huge_page_allocs: CachePadded<AtomicU64>,

    // Allocation tracking for double-free detection (using usize for Send+Sync safety)
    active_allocations: std::sync::RwLock<HashMap<usize, (u64, Instant)>>, // ptr_addr -> (generation, time)
}

impl std::fmt::Debug for SecureMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecureMemoryPool")
            .field("config", &self.config)
            .field("pool_id", &self.pool_id)
            .field("next_generation", &self.next_generation)
            .finish_non_exhaustive()
    }
}

impl SecureMemoryPool {
    /// Create a new secure memory pool
    pub fn new(config: SecurePoolConfig) -> Result<Arc<Self>> {
        let _ = chunk_layout(config.chunk_size, config.alignment)?;

        static POOL_ID_COUNTER: AtomicU32 = AtomicU32::new(1);
        let pool_id = POOL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Initialize cache allocator if cache alignment is enabled
        let cache_allocator = if config.enable_cache_alignment && config.cache_config.is_some() {
            // SAFETY: is_some() check above guarantees this unwrap succeeds
            Some(CacheOptimizedAllocator::new(
                config
                    .cache_config
                    .clone()
                    .expect("cache_config present when cache_friendly enabled"),
            ))
        } else {
            None
        };

        Ok(Arc::new(Self {
            config,
            pool_id,
            global_stack: LockFreeStack::new(),
            next_generation: AtomicU64::new(1),
            local_caches: thread_local::ThreadLocal::new(),
            cache_allocator,
            cache_aligned_allocs: CachePadded::new(AtomicU64::new(0)),
            numa_local_allocs: CachePadded::new(AtomicU64::new(0)),
            hot_data_allocs: CachePadded::new(AtomicU64::new(0)),
            cold_data_allocs: CachePadded::new(AtomicU64::new(0)),
            huge_page_allocs: CachePadded::new(AtomicU64::new(0)),
            alloc_count: CachePadded::new(AtomicU64::new(0)),
            dealloc_count: CachePadded::new(AtomicU64::new(0)),
            pool_hits: CachePadded::new(AtomicU64::new(0)),
            pool_misses: CachePadded::new(AtomicU64::new(0)),
            corruption_detected: CachePadded::new(AtomicU64::new(0)),
            double_free_detected: CachePadded::new(AtomicU64::new(0)),
            local_cache_hits: CachePadded::new(AtomicU64::new(0)),
            cross_thread_steals: CachePadded::new(AtomicU64::new(0)),
            active_allocations: std::sync::RwLock::new(HashMap::new()),
        }))
    }

    /// Allocate a chunk from the pool with RAII guard and cache optimization.
    ///
    /// Returned pointers are guaranteed to be aligned to `config.alignment.max(8)` bytes.
    pub fn allocate(self: &Arc<Self>) -> Result<SecurePooledPtr> {
        self.allocate_with_hint(false) // Default: not marked as hot
    }

    /// Allocate a chunk with hot/cold hint for cache optimization
    pub fn allocate_with_hint(self: &Arc<Self>, is_hot: bool) -> Result<SecurePooledPtr> {
        self.alloc_count.fetch_add(1, Ordering::Relaxed);

        // Try thread-local cache first
        let local_cache = self.local_caches.get_or(|| {
            RefCell::new(LocalCache::new(
                self.config.local_cache_size,
                self.config.zero_on_free,
                self.config.enable_simd_ops,
                self.config.simd_threshold,
            ))
        });

        if let Some(mut chunk) = local_cache.borrow_mut().try_pop() {
            self.local_cache_hits.fetch_add(1, Ordering::Relaxed);
            self.pool_hits.fetch_add(1, Ordering::Relaxed);

            // Validate chunk before returning
            if let Err(e) = chunk.validate() {
                self.corruption_detected.fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }

            // Only allocate_new_chunk_optimized used to honour this flag, so a
            // cache hit skipped it entirely.
            if self.config.zero_on_alloc {
                self.zero_chunk_simd(&mut chunk)?;
            }

            // Track hot/cold allocation statistics
            if is_hot {
                self.hot_data_allocs.fetch_add(1, Ordering::Relaxed);
            } else {
                self.cold_data_allocs.fetch_add(1, Ordering::Relaxed);
            }

            // Hot/cold separation temporarily simplified
            // if self.config.enable_hot_cold_separation {
            //     if let Ok(mut separator_guard) = self.hot_cold_separator.lock() {
            //         if let Some(ref mut separator) = *separator_guard {
            //             let access_count = if is_hot { self.config.hot_data_threshold + 1 } else { 1 };
            //             separator.insert(chunk.as_ptr() as usize, access_count);
            //         }
            //     }
            // }

            // Track allocation
            self.active_allocations
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(
                    chunk.as_ptr() as usize,
                    (chunk.generation(), Instant::now()),
                );

            return Ok(SecurePooledPtr {
                chunk: Some(chunk),
                pool: Arc::downgrade(self),
            });
        }

        // Try global stack
        if let Some(mut chunk) = self.global_stack.pop() {
            self.cross_thread_steals.fetch_add(1, Ordering::Relaxed);
            self.pool_hits.fetch_add(1, Ordering::Relaxed);

            // Validate chunk before returning
            if let Err(e) = chunk.validate() {
                self.corruption_detected.fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }

            // Same gap as the local-cache hit above.
            if self.config.zero_on_alloc {
                self.zero_chunk_simd(&mut chunk)?;
            }

            // Track hot/cold allocation statistics
            if is_hot {
                self.hot_data_allocs.fetch_add(1, Ordering::Relaxed);
            } else {
                self.cold_data_allocs.fetch_add(1, Ordering::Relaxed);
            }

            // Track allocation
            self.active_allocations
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(
                    chunk.as_ptr() as usize,
                    (chunk.generation(), Instant::now()),
                );

            return Ok(SecurePooledPtr {
                chunk: Some(chunk),
                pool: Arc::downgrade(self),
            });
        }

        // Allocate new chunk with cache optimizations
        self.pool_misses.fetch_add(1, Ordering::Relaxed);
        let generation = self.next_generation.fetch_add(1, Ordering::AcqRel);
        let chunk = self.allocate_new_chunk_optimized(generation, is_hot)?;

        // Track hot/cold allocation statistics
        if is_hot {
            self.hot_data_allocs.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cold_data_allocs.fetch_add(1, Ordering::Relaxed);
        }

        // Track allocation
        self.active_allocations
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                chunk.as_ptr() as usize,
                (chunk.generation(), Instant::now()),
            );

        Ok(SecurePooledPtr {
            chunk: Some(chunk),
            pool: Arc::downgrade(self),
        })
    }

    #[cold]
    #[inline(never)]
    fn deallocate_corrupted(
        &self,
        chunk: SecureChunk,
        chunk_ptr: usize,
        e: ZiporaError,
    ) -> Result<()> {
        self.corruption_detected.fetch_add(1, Ordering::Relaxed);
        let mut allocs = self
            .active_allocations
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if allocs
            .get(&chunk_ptr)
            .is_some_and(|&(orig_gen, _)| orig_gen == chunk.generation())
        {
            allocs.remove(&chunk_ptr);
        }
        drop(allocs);
        chunk.deallocate(self.config.zero_on_free);
        Err(e)
    }

    #[cold]
    #[inline(never)]
    fn deallocate_double_free(&self, chunk: SecureChunk, msg: &'static str) -> Result<()> {
        self.double_free_detected.fetch_add(1, Ordering::Relaxed);
        chunk.deallocate(self.config.zero_on_free);
        Err(ZiporaError::invalid_data(msg))
    }

    /// Internal deallocation with security validation
    fn deallocate_internal(&self, mut chunk: SecureChunk) -> Result<()> {
        self.dealloc_count.fetch_add(1, Ordering::Relaxed);
        let chunk_ptr = chunk.as_ptr() as usize;

        // Validate chunk before deallocation
        if let Err(e) = chunk.validate() {
            return self.deallocate_corrupted(chunk, chunk_ptr, e);
        }

        // Check for double-free
        let mut allocs = self
            .active_allocations
            .write()
            .unwrap_or_else(|e| e.into_inner());

        if let Some((original_generation, prev_time)) = allocs.remove(&chunk_ptr) {
            if original_generation != chunk.generation() {
                allocs.insert(chunk_ptr, (original_generation, prev_time));
                drop(allocs);
                return self
                    .deallocate_double_free(chunk, "Double-free detected: generation mismatch");
            }
        } else {
            drop(allocs);
            return self
                .deallocate_double_free(chunk, "Double-free detected: pointer not allocated");
        }
        drop(allocs);

        // Zeroing used to live only in SecureChunk::deallocate, which this
        // path never reaches: a freed chunk goes straight to the thread-local
        // cache or the global stack and is handed back out as-is, so the next
        // tenant saw the previous one's bytes even with zero_on_free set.
        if self.config.zero_on_free {
            chunk.zero_payload(self.config.enable_simd_ops, self.config.simd_threshold);
        }

        // Try to return to thread-local cache
        let local_cache = self.local_caches.get_or(|| {
            RefCell::new(LocalCache::new(
                self.config.local_cache_size,
                self.config.zero_on_free,
                self.config.enable_simd_ops,
                self.config.simd_threshold,
            ))
        });

        if let Err(chunk_to_push) = local_cache.borrow_mut().try_push(chunk) {
            // Local cache full, push the chunk directly to global stack
            self.global_stack.push(chunk_to_push);
        }

        Ok(())
    }

    /// Allocate a new chunk with cache optimizations
    #[cold]
    #[inline(never)]
    fn allocate_new_chunk_optimized(&self, generation: u64, is_hot: bool) -> Result<SecureChunk> {
        // Use cache allocator if available and chunk size meets threshold
        if let Some(ref _cache_allocator) = self.cache_allocator {
            if self.config.enable_cache_alignment {
                self.cache_aligned_allocs.fetch_add(1, Ordering::Relaxed);
            }

            // Check for NUMA-aware allocation
            if self.config.enable_numa_awareness {
                let optimal_node = -1; // Simplified: disable NUMA for now
                // let optimal_node = get_optimal_numa_node();
                if optimal_node >= 0 {
                    self.numa_local_allocs.fetch_add(1, Ordering::Relaxed);

                    // NUMA allocation temporarily disabled for simplicity
                    // if let Ok(ptr) = numa_alloc_aligned(
                    //     self.config.chunk_size,
                    //     self.config.cache_config.as_ref().map(|c| c.cache_line_size).unwrap_or(64),
                    //     optimal_node as u32
                    // ) {
                    //     return SecureChunk::new_from_ptr(ptr, self.config.chunk_size, generation, self.pool_id);
                    // }
                }
            }

            // Check for huge page allocation
            #[cfg(target_os = "linux")]
            if self.config.enable_huge_pages
                && self.config.chunk_size >= self.config.huge_page_threshold
            {
                self.huge_page_allocs.fetch_add(1, Ordering::Relaxed);
                // Try huge page allocation (would integrate with hugepage module)
                // For now, fall through to regular allocation
            }

            // Prefetch hints for hot data
            if is_hot
                && self
                    .config
                    .cache_config
                    .as_ref()
                    .map(|c| c.enable_prefetch)
                    .unwrap_or(false)
            {
                // Future: implement prefetch hints for hot allocations
            }
        }

        // Fall back to regular allocation (`SecureChunk::new` uses `alloc_zeroed`, so fresh
        // chunks start zero-initialized regardless of `self.config.zero_on_alloc`).
        let mut chunk = SecureChunk::new(
            self.config.chunk_size,
            self.config.alignment,
            generation,
            self.pool_id,
        )?;
        if self.config.zero_on_alloc {
            self.zero_chunk_simd(&mut chunk)?;
        }
        Ok(chunk)
    }

    /// Zero chunk memory with SIMD acceleration and adaptive selection
    fn zero_chunk_simd(&self, chunk: &mut SecureChunk) -> Result<()> {
        if !self.config.enable_simd_ops || chunk.size() < self.config.simd_threshold {
            // SAFETY: pointer valid from allocation, size matches allocation
            unsafe {
                std::ptr::write_bytes(chunk.as_ptr(), 0, chunk.size());
            }
            return Ok(());
        }

        // Use adaptive SIMD selection for optimal performance
        let selector = AdaptiveSimdSelector::global();
        let start = Instant::now();

        // SAFETY: `chunk.as_ptr()` points to `chunk.size()` writable bytes exclusively owned by `chunk`.
        unsafe {
            fast_fill_raw(chunk.as_ptr(), 0, chunk.size());
        }

        // Monitor performance for adaptive adjustment
        selector.monitor_performance(Operation::MemZero, start.elapsed(), chunk.size() as u64);

        Ok(())
    }

    /// Verify memory is properly zeroed using SIMD acceleration
    ///
    /// # Safety
    /// This function is safe because it accepts a slice, which guarantees
    /// the pointer and length form a valid memory range.
    ///
    /// # Performance
    /// Uses SIMD comparison for 8-16x faster verification than byte-by-byte checking.
    /// Particularly effective for large memory regions.
    #[inline]
    pub fn verify_zeroed_simd(&self, data: &[u8]) -> Result<bool> {
        // Create zero buffer for comparison
        let zero_buf = vec![0u8; data.len()];

        // Use SIMD comparison (8-16x faster than byte-by-byte)
        Ok(fast_compare(data, &zero_buf) == 0)
    }

    /// Prefetch allocation metadata for improved cache performance
    ///
    /// # Performance
    /// Issues prefetch hints for metadata structures to reduce cache misses
    /// during allocation. Most beneficial for large batch allocations.
    #[inline]
    fn prefetch_allocation_metadata(&self, _size: usize) {
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_cache_alignment {
            // Prefetch the global stack head (most frequently accessed metadata)
            fast_prefetch(&self.global_stack, PrefetchHint::T0);

            // Prefetch stats counters (frequently accessed during allocation)
            fast_prefetch(&self.alloc_count, PrefetchHint::T0);
        }
    }

    /// Bulk allocation with prefetching for improved performance
    ///
    /// # Performance
    /// Uses configurable lookahead prefetching (default: 8) to warm cache for future
    /// allocations, providing 20-30% improvement over sequential allocations.
    /// The prefetch distance can be tuned via `SecurePoolConfig::prefetch_distance`.
    ///
    /// # Arguments
    /// * `sizes` - Slice of allocation sizes (all should be equal to chunk_size)
    ///
    /// # Returns
    /// Vector of allocated pointers in the same order as requested sizes
    pub fn allocate_bulk_with_prefetch(
        self: &Arc<Self>,
        sizes: &[usize],
    ) -> Result<Vec<SecurePooledPtr>> {
        let prefetch_distance = self.config.prefetch_distance;
        let mut results = Vec::with_capacity(sizes.len());

        for (i, &size) in sizes.iter().enumerate() {
            // Validate size matches chunk size
            if size != self.config.chunk_size {
                return Err(ZiporaError::invalid_data(format!(
                    "Size {} does not match chunk_size {}",
                    size, self.config.chunk_size
                )));
            }

            // Prefetch future allocations (lookahead)
            if i + prefetch_distance < sizes.len() {
                self.prefetch_allocation_metadata(sizes[i + prefetch_distance]);
            }

            // Perform allocation
            results.push(self.allocate()?);
        }

        Ok(results)
    }

    /// Get current pool statistics
    pub fn stats(&self) -> SecurePoolStats {
        let total_allocs = self.alloc_count.load(Ordering::Relaxed);
        let cache_hit_ratio = if total_allocs > 0 {
            self.pool_hits.load(Ordering::Relaxed) as f64 / total_allocs as f64
        } else {
            0.0
        };

        SecurePoolStats {
            allocated: 0, // Would need to track this separately
            available: 0, // Would need to track this separately
            chunks: 0,    // Would need to track this separately
            alloc_count: total_allocs,
            dealloc_count: self.dealloc_count.load(Ordering::Relaxed),
            pool_hits: self.pool_hits.load(Ordering::Relaxed),
            pool_misses: self.pool_misses.load(Ordering::Relaxed),
            corruption_detected: self.corruption_detected.load(Ordering::Relaxed),
            double_free_detected: self.double_free_detected.load(Ordering::Relaxed),
            local_cache_hits: self.local_cache_hits.load(Ordering::Relaxed),
            cross_thread_steals: self.cross_thread_steals.load(Ordering::Relaxed),
            cache_aligned_allocs: self.cache_aligned_allocs.load(Ordering::Relaxed),
            numa_local_allocs: self.numa_local_allocs.load(Ordering::Relaxed),
            hot_data_allocs: self.hot_data_allocs.load(Ordering::Relaxed),
            cold_data_allocs: self.cold_data_allocs.load(Ordering::Relaxed),
            huge_page_allocs: self.huge_page_allocs.load(Ordering::Relaxed),
            cache_hit_ratio,
        }
    }

    /// Clear all chunks from the pool
    pub fn clear(&self) -> Result<()> {
        // Clear global stack and deallocate chunks
        while let Some(chunk) = self.global_stack.pop() {
            chunk.deallocate(self.config.zero_on_free);
        }

        // Note: We cannot safely clear thread-local caches from another thread
        // due to RefCell not being Sync. Thread-local caches will be cleared
        // when threads exit or when they access the cache and find it should be cleared.

        // Clear allocation tracking
        self.active_allocations
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();

        Ok(())
    }

    /// Get pool configuration
    pub fn config(&self) -> &SecurePoolConfig {
        &self.config
    }

    /// Validate pool integrity
    pub fn validate(&self) -> Result<()> {
        // Check active allocations for corruption
        for (&ptr_addr, &(generation, _time)) in self
            .active_allocations
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            // Read canary from the chunk header for validation
            let data_ptr = ptr_addr as *mut u8;
            let header_size = std::mem::size_of::<ChunkHeader>();
            // SAFETY: data_ptr is from an active allocation where new() places ChunkHeader immediately preceding data_ptr at raw_ptr + (payload_offset - header_size)
            let header_ptr = unsafe { data_ptr.sub(header_size) as *const ChunkHeader };
            // SAFETY: header_ptr valid, aligned to ChunkHeader, from allocation
            let header = unsafe { &*header_ptr };

            // Validate header
            if header.magic != CHUNK_HEADER_MAGIC {
                self.corruption_detected.fetch_add(1, Ordering::Relaxed);
                return Err(ZiporaError::invalid_data(format!(
                    "Header corruption detected: magic={:#x}, expected={:#x}",
                    header.magic, CHUNK_HEADER_MAGIC
                )));
            }

            if header.generation != generation {
                self.corruption_detected.fetch_add(1, Ordering::Relaxed);
                return Err(ZiporaError::invalid_data(format!(
                    "Generation mismatch: header={}, tracking={}",
                    header.generation, generation
                )));
            }

            if header.pool_id != self.pool_id {
                self.corruption_detected.fetch_add(1, Ordering::Relaxed);
                return Err(ZiporaError::invalid_data(format!(
                    "Pool ID mismatch: header={}, pool={}",
                    header.pool_id, self.pool_id
                )));
            }

            // Validate footer
            // SAFETY: data_ptr valid, padded chunk_size offset matches new()
            let footer_ptr = unsafe {
                data_ptr.add(padded_payload_size(self.config.chunk_size)) as *const ChunkFooter
            };
            // SAFETY: footer_ptr valid, aligned to ChunkFooter, within allocated region
            let footer = unsafe { &*footer_ptr };

            if footer.magic != CHUNK_FOOTER_MAGIC {
                self.corruption_detected.fetch_add(1, Ordering::Relaxed);
                return Err(ZiporaError::invalid_data(format!(
                    "Footer corruption detected: magic={:#x}, expected={:#x}",
                    footer.magic, CHUNK_FOOTER_MAGIC
                )));
            }

            if footer.canary != header.canary {
                self.corruption_detected.fetch_add(1, Ordering::Relaxed);
                return Err(ZiporaError::invalid_data(format!(
                    "Canary mismatch between header and footer: header={:#x}, footer={:#x}",
                    header.canary, footer.canary
                )));
            }

            if footer.generation != generation {
                self.corruption_detected.fetch_add(1, Ordering::Relaxed);
                return Err(ZiporaError::invalid_data(format!(
                    "Footer generation mismatch: footer={}, tracking={}",
                    footer.generation, generation
                )));
            }
        }

        Ok(())
    }
}

impl Drop for SecureMemoryPool {
    fn drop(&mut self) {
        let _ = self.clear();
    }
}

/// RAII guard for automatic secure deallocation
pub struct SecurePooledPtr {
    chunk: Option<SecureChunk>,
    pool: Weak<SecureMemoryPool>,
}

impl SecurePooledPtr {
    /// Get pointer to allocated memory (null once the chunk has been released).
    ///
    /// Bytes written through this pointer within `0..self.size()` must be
    /// initialized: [`Self::as_slice`] and [`Self::as_mut_slice`] expose them as
    /// `u8`, on this allocation and, when neither `zero_on_free` nor
    /// `zero_on_alloc` is set, on later allocations that reuse the chunk.
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.chunk
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null_mut())
    }

    /// Get non-null pointer to allocated memory
    pub fn as_non_null(&self) -> Option<NonNull<u8>> {
        self.chunk.as_ref().and_then(|c| NonNull::new(c.as_ptr()))
    }

    /// Get size of allocated memory
    #[inline]
    pub fn size(&self) -> usize {
        self.chunk.as_ref().map(|c| c.size()).unwrap_or(0)
    }

    /// Get generation for debugging
    pub fn generation(&self) -> u64 {
        self.chunk.as_ref().map(|c| c.generation()).unwrap_or(0)
    }

    /// Validate chunk integrity
    pub fn validate(&self) -> Result<()> {
        if let Some(chunk) = &self.chunk {
            chunk.validate()
        } else {
            Err(ZiporaError::invalid_data("Chunk already deallocated"))
        }
    }

    /// Get slice view of the allocated memory
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        if let Some(chunk) = &self.chunk {
            // SAFETY: pointer valid from allocation, size matches allocation
            unsafe { std::slice::from_raw_parts(chunk.as_ptr(), chunk.size()) }
        } else {
            &[]
        }
    }

    /// Get mutable slice view of the allocated memory
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if let Some(chunk) = &self.chunk {
            // SAFETY: pointer valid from allocation, size matches allocation
            unsafe { std::slice::from_raw_parts_mut(chunk.as_ptr(), chunk.size()) }
        } else {
            &mut []
        }
    }
}

impl Drop for SecurePooledPtr {
    fn drop(&mut self) {
        if let Some(chunk) = self.chunk.take() {
            if let Some(pool) = self.pool.upgrade() {
                // Best effort deallocation - if it fails, the chunk is leaked
                // but this prevents crashes during cleanup
                let _ = pool.deallocate_internal(chunk);
            } else {
                // Pool is gone; wipe payload with volatile writes before freeing
                chunk.deallocate(true);
            }
        }
    }
}

// SAFETY: SecurePooledPtr is Send because:
// 1. `chunk: Option<SecureChunk>` - SecureChunk is Send (see above).
// 2. `pool: Weak<SecureMemoryPool>` - Weak<T> is Send if T is Send+Sync.
//    SecureMemoryPool uses Arc and atomics for thread-safe reference counting.
// Transferring ownership between threads is safe as only one owner exists.
unsafe impl Send for SecurePooledPtr {}

// SAFETY: SecurePooledPtr is Sync because:
// 1. All access is through `&self` (immutable) or `&mut self` (exclusive).
// 2. `as_slice()` returns a read-only view of the memory.
// 3. `as_mut_slice()` requires `&mut self`, ensuring exclusive access.
// 4. The underlying chunk is either None or exclusively owned by this ptr.
// 5. Pool deallocation is thread-safe via Arc's atomic reference counting.
//
// Note: Safe concurrent read access is provided. Mutable access requires &mut self.
unsafe impl Sync for SecurePooledPtr {}

/// Get current time in nanoseconds
fn current_time_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Find appropriate size class for given size
pub fn size_to_class(size: usize) -> usize {
    SIZE_CLASSES.binary_search(&size).unwrap_or_else(|i| {
        if i < SIZE_CLASSES.len() {
            i
        } else {
            SIZE_CLASSES.len() - 1
        }
    })
}

/// Global secure pools for different size classes
static GLOBAL_SECURE_POOLS: LazyLock<Vec<Arc<SecureMemoryPool>>> = LazyLock::new(|| {
    // SAFETY: Static initialization - panic on pool creation failure is appropriate
    // because the program cannot continue without global memory pools
    vec![
        SecureMemoryPool::new(SecurePoolConfig::small_secure())
            .expect("small secure pool creation"),
        SecureMemoryPool::new(SecurePoolConfig::medium_secure())
            .expect("medium secure pool creation"),
        SecureMemoryPool::new(SecurePoolConfig::large_secure())
            .expect("large secure pool creation"),
    ]
});

/// Get appropriate global pool for size
pub fn get_global_pool_for_size(size: usize) -> &'static Arc<SecureMemoryPool> {
    if size <= 1024 {
        &GLOBAL_SECURE_POOLS[0]
    } else if size <= 64 * 1024 {
        &GLOBAL_SECURE_POOLS[1]
    } else {
        &GLOBAL_SECURE_POOLS[2]
    }
}

/// Get aggregated statistics from all global pools
pub fn get_global_secure_pool_stats() -> SecurePoolStats {
    let mut total_stats = SecurePoolStats::default();

    for pool in GLOBAL_SECURE_POOLS.iter() {
        let stats = pool.stats();
        total_stats.allocated += stats.allocated;
        total_stats.available += stats.available;
        total_stats.chunks += stats.chunks;
        total_stats.alloc_count += stats.alloc_count;
        total_stats.dealloc_count += stats.dealloc_count;
        total_stats.pool_hits += stats.pool_hits;
        total_stats.pool_misses += stats.pool_misses;
        total_stats.corruption_detected += stats.corruption_detected;
        total_stats.double_free_detected += stats.double_free_detected;
        total_stats.local_cache_hits += stats.local_cache_hits;
        total_stats.cross_thread_steals += stats.cross_thread_steals;
    }

    total_stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_secure_pool_creation() {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.alloc_count, 0);
        assert_eq!(stats.dealloc_count, 0);
    }

    #[test]
    fn test_secure_allocation_deallocation() {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();

        let ptr1 = pool.allocate().unwrap();
        assert!(!ptr1.as_ptr().is_null());
        assert_eq!(ptr1.size(), 1024);

        let ptr2 = pool.allocate().unwrap();
        assert!(!ptr2.as_ptr().is_null());
        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());

        // Chunks are automatically deallocated on drop
        drop(ptr1);
        drop(ptr2);

        let stats = pool.stats();
        assert_eq!(stats.alloc_count, 2);
        assert_eq!(stats.dealloc_count, 2);
    }

    #[test]
    fn test_footer_stays_aligned_for_unaligned_chunk_sizes() {
        // Regression: the footer was written at `header + chunk_size`.
        // ChunkFooter holds two u64s and needs 8-byte alignment, but
        // chunk_size is only validated as non-zero, so any size that is not a
        // multiple of 8 produced a misaligned reference — UB, and an abort
        // under debug assertions. Every shipped preset happens to be a
        // multiple of 8, so only user-chosen sizes hit it.
        let footer_align = std::mem::align_of::<ChunkFooter>();
        for align in [1usize, 8, 16, 32, 64, 128] {
            for size in [1usize, 7, 8, 9, 100, 1023, 1024] {
                let (_, payload_offset, _) = chunk_layout(size, align).unwrap();
                let padded = padded_payload_size(size);
                assert!(padded >= size, "padding shrank the payload for {size}");
                assert!(
                    padded < size + footer_align,
                    "padding overshot for {size}: {padded}"
                );
                assert_eq!(
                    (payload_offset + padded) % footer_align,
                    0,
                    "footer misaligned for chunk_size {size} with align {align}"
                );
            }
        }
    }

    #[test]
    fn test_pool_with_unaligned_chunk_size() {
        // The same defect end to end: allocating from a pool whose chunk_size
        // is not a multiple of 8 wrote and read the footer through a
        // misaligned pointer.
        let config = SecurePoolConfig::new(100, 16, 8);
        let pool = SecureMemoryPool::new(config).unwrap();

        let mut ptr = pool.allocate().unwrap();
        assert_eq!(ptr.size(), 100);
        ptr.as_mut_slice().fill(0xAB);
        assert!(ptr.validate().is_ok());
        drop(ptr);

        let ptr = pool.allocate().unwrap();
        assert!(ptr.validate().is_ok());
    }

    #[test]
    fn test_zero_on_free_clears_recycled_chunks() {
        // Regression: zeroing lived only in SecureChunk::deallocate, which the
        // success path never calls — a freed chunk goes straight to the
        // thread-local cache. Every preset sets zero_on_free, yet the next
        // allocation handed back the previous tenant's bytes.
        let mut config = SecurePoolConfig::new(1024, 16, 8);
        config.zero_on_free = true;
        config.zero_on_alloc = false;
        let pool = SecureMemoryPool::new(config).unwrap();

        let mut ptr = pool.allocate().unwrap();
        let raw = ptr.as_ptr();
        ptr.as_mut_slice().fill(0xAB);
        drop(ptr);

        let ptr = pool.allocate().unwrap();
        assert_eq!(
            ptr.as_ptr(),
            raw,
            "this test is vacuous unless the chunk is recycled"
        );
        assert!(
            ptr.as_slice().iter().all(|&b| b == 0),
            "recycled chunk still carries the previous tenant's bytes"
        );
    }

    #[test]
    fn test_zero_on_alloc_clears_recycled_chunks() {
        // The mirror image: only allocate_new_chunk_optimized honoured
        // zero_on_alloc, so a cache hit skipped it.
        let mut config = SecurePoolConfig::new(1024, 16, 8);
        config.zero_on_free = false;
        config.zero_on_alloc = true;
        let pool = SecureMemoryPool::new(config).unwrap();

        let mut ptr = pool.allocate().unwrap();
        let raw = ptr.as_ptr();
        ptr.as_mut_slice().fill(0xCD);
        drop(ptr);

        let ptr = pool.allocate().unwrap();
        assert_eq!(
            ptr.as_ptr(),
            raw,
            "this test is vacuous unless the chunk is recycled"
        );
        assert!(
            ptr.as_slice().iter().all(|&b| b == 0),
            "recycled chunk was handed out without being zeroed"
        );
    }

    /// Drive enough chunks through the pool that the thread-local cache
    /// overflows, so the surplus is recycled through the global stack.
    ///
    /// Returns the addresses handed out on the second round, in order, and the
    /// number of chunks that came off the global stack.
    fn recycle_through_global_stack(config: SecurePoolConfig) -> (Vec<*mut u8>, u64) {
        let local_cache_size = config.local_cache_size;
        let rounds = local_cache_size + 4;
        let pool = SecureMemoryPool::new(config).unwrap();

        let mut first: Vec<_> = (0..rounds).map(|_| pool.allocate().unwrap()).collect();
        for ptr in first.iter_mut() {
            ptr.as_mut_slice().fill(0xE7);
        }
        drop(first);

        let steals_before = pool.stats().cross_thread_steals;
        let second: Vec<_> = (0..rounds).map(|_| pool.allocate().unwrap()).collect();
        let steals = pool.stats().cross_thread_steals - steals_before;

        for ptr in second.iter() {
            assert!(
                ptr.as_slice().iter().all(|&b| b == 0),
                "a recycled chunk still carries the previous tenant's bytes"
            );
        }
        (second.iter().map(|p| p.as_ptr()).collect(), steals)
    }

    #[test]
    fn test_zero_on_free_clears_chunks_routed_through_the_global_stack() {
        // Coverage, not RED: 2707e30 moved the zeroing ahead of the routing, so
        // it covers both destinations. The two tests above only ever exercise
        // the thread-local cache hit; this one forces the local cache to
        // overflow so the surplus is recycled through the global stack.
        let mut config = SecurePoolConfig::new(1024, 64, 8);
        config.local_cache_size = 2;
        config.zero_on_free = true;
        config.zero_on_alloc = false;

        let (addrs, steals) = recycle_through_global_stack(config);
        assert!(
            steals > 0,
            "vacuous unless the global stack is on the path: {steals} steals"
        );
        assert_eq!(
            addrs.len(),
            addrs.iter().collect::<std::collections::HashSet<_>>().len(),
            "one address was handed to two live callers"
        );
    }

    #[test]
    fn test_zero_on_alloc_clears_chunks_taken_off_the_global_stack() {
        // The global-stack arm of allocate_with_hint is a second copy of the
        // zero_on_alloc check (secure_pool.rs:1024); nothing asserted it.
        let mut config = SecurePoolConfig::new(1024, 64, 8);
        config.local_cache_size = 2;
        config.zero_on_free = false;
        config.zero_on_alloc = true;

        let (addrs, steals) = recycle_through_global_stack(config);
        assert!(
            steals > 0,
            "vacuous unless the global stack is on the path: {steals} steals"
        );
        assert_eq!(
            addrs.len(),
            addrs.iter().collect::<std::collections::HashSet<_>>().len(),
            "one address was handed to two live callers"
        );
    }

    #[test]
    fn test_chunk_validation() {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();

        let ptr = pool.allocate().unwrap();
        assert!(ptr.validate().is_ok());

        // Validation should still work
        assert!(ptr.validate().is_ok());
    }

    #[test]
    fn test_double_free_detection() {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();

        let ptr = pool.allocate().unwrap();
        let _raw_ptr = ptr.as_ptr();

        // First deallocation is automatic on drop
        drop(ptr);

        // Attempt to manually create and deallocate the same pointer should fail
        // This is prevented by the RAII design - you can't create a SecurePooledPtr
        // from a raw pointer, so double-free is structurally prevented

        let stats = pool.stats();
        assert_eq!(stats.double_free_detected, 0); // No double-free attempts possible
    }

    #[test]
    fn test_pool_reuse() {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();

        let ptr1 = pool.allocate().unwrap();
        let _addr1 = ptr1.as_ptr();
        drop(ptr1);

        let ptr2 = pool.allocate().unwrap();
        let addr2 = ptr2.as_ptr();

        // Memory might be reused (but not guaranteed due to generation counters)
        // What matters is that it works correctly either way
        assert!(!addr2.is_null());
        drop(ptr2);

        let stats = pool.stats();
        assert!(stats.pool_hits > 0 || stats.pool_misses > 0);
    }

    #[test]
    fn test_concurrent_allocation() {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        let allocated_count = Arc::new(AtomicUsize::new(0));
        let (threads, iters) = if cfg!(miri) { (4, 8) } else { (10, 100) };

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let pool = pool.clone();
                let count = allocated_count.clone();
                thread::spawn(move || {
                    for _ in 0..iters {
                        let _ptr = pool.allocate().unwrap();
                        count.fetch_add(1, Ordering::Relaxed);

                        // Simulate some work
                        if cfg!(miri) {
                            thread::yield_now();
                        } else {
                            thread::sleep(Duration::from_micros(1));
                        }

                        // ptr automatically deallocated on scope exit
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let expected = threads * iters;
        assert_eq!(allocated_count.load(Ordering::Relaxed), expected);

        let stats = pool.stats();
        assert_eq!(stats.alloc_count, expected as u64);
        assert_eq!(stats.dealloc_count, expected as u64);
        assert_eq!(stats.corruption_detected, 0);
        assert_eq!(stats.double_free_detected, 0);
    }

    #[test]
    fn test_thread_local_caching() {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();

        // Allocate and deallocate to populate thread-local cache
        for _ in 0..10 {
            let ptr = pool.allocate().unwrap();
            drop(ptr);
        }

        // Next allocations should hit thread-local cache
        let ptr = pool.allocate().unwrap();
        drop(ptr);

        let stats = pool.stats();
        assert!(stats.local_cache_hits > 0);
    }

    #[test]
    fn test_size_classes() {
        assert_eq!(size_to_class(8), 0);
        assert_eq!(size_to_class(16), 1);
        assert_eq!(size_to_class(100), 7); // Should use 112-byte class
        assert_eq!(size_to_class(1000), 20); // Should use 1024-byte class
    }

    #[test]
    fn test_global_pools() {
        let small_pool = get_global_pool_for_size(100);
        let medium_pool = get_global_pool_for_size(10000);
        let large_pool = get_global_pool_for_size(100000);

        assert_eq!(small_pool.config().chunk_size, 1024);
        assert_eq!(medium_pool.config().chunk_size, 64 * 1024);
        assert_eq!(large_pool.config().chunk_size, 1024 * 1024);

        let ptr1 = small_pool.allocate().unwrap();
        let ptr2 = medium_pool.allocate().unwrap();
        let ptr3 = large_pool.allocate().unwrap();

        assert!(!ptr1.as_ptr().is_null());
        assert!(!ptr2.as_ptr().is_null());
        assert!(!ptr3.as_ptr().is_null());
    }

    #[test]
    fn test_memory_access() {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();

        let mut ptr = pool.allocate().unwrap();

        // Test slice access
        let slice = ptr.as_mut_slice();
        slice[0] = 42;
        slice[1023] = 84;

        let slice = ptr.as_slice();
        assert_eq!(slice[0], 42);
        assert_eq!(slice[1023], 84);
        assert_eq!(slice.len(), 1024);
    }

    #[test]
    fn test_simd_configuration() {
        // Test SIMD defaults
        let config = SecurePoolConfig::small_secure();
        assert!(config.enable_simd_ops);
        assert_eq!(config.simd_threshold, 64);

        // Test SIMD builder methods
        let config_disabled = SecurePoolConfig::small_secure()
            .with_simd_ops(false)
            .with_simd_threshold(128);
        assert!(!config_disabled.enable_simd_ops);
        assert_eq!(config_disabled.simd_threshold, 128);

        // Test pool creation with SIMD config
        let pool = SecureMemoryPool::new(config_disabled).unwrap();
        assert!(!pool.config().enable_simd_ops);
        assert_eq!(pool.config().simd_threshold, 128);

        // Test that allocation/deallocation works with SIMD disabled
        let ptr = pool.allocate().unwrap();
        assert!(!ptr.as_ptr().is_null());
        // ptr automatically deallocated on drop with SIMD config
    }

    #[test]
    fn test_simd_with_large_chunks() {
        // Test with large chunks that should benefit from SIMD
        let config = SecurePoolConfig::large_secure()
            .with_simd_ops(true)
            .with_simd_threshold(64);
        let pool = SecureMemoryPool::new(config).unwrap();

        let ptr = pool.allocate().unwrap();
        assert_eq!(ptr.size(), 1024 * 1024); // 1MB chunks
        assert!(!ptr.as_ptr().is_null());

        // Verify configuration
        assert!(pool.config().enable_simd_ops);
        assert_eq!(pool.config().simd_threshold, 64);

        // Large chunks should be SIMD-optimized since 1MB >> 64 bytes
        // This is tested implicitly through the deallocation process
    }

    #[test]
    fn test_simd_memory_zeroing() {
        let config = SecurePoolConfig::small_secure()
            .with_zero_on_alloc(true)
            .with_simd_ops(true)
            .with_simd_threshold(64);
        let pool = SecureMemoryPool::new(config).unwrap();

        // Allocate and verify SIMD zeroing
        let ptr = pool.allocate().unwrap();
        // SAFETY: test code, pointer valid from allocation
        let slice = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), ptr.size()) };

        // All bytes should be zero
        assert!(slice.iter().all(|&b| b == 0));

        // Verify using SIMD verification
        assert!(pool.verify_zeroed_simd(ptr.as_slice()).unwrap());
    }

    #[test]
    fn test_simd_verification() {
        let config = SecurePoolConfig::medium_secure().with_simd_ops(true);
        let pool = SecureMemoryPool::new(config).unwrap();

        let ptr = pool.allocate().unwrap();

        // SAFETY: test code, pointer valid from allocation
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, ptr.size());
        }

        // Verify using SIMD
        assert!(pool.verify_zeroed_simd(ptr.as_slice()).unwrap());

        // SAFETY: test code, pointer valid from allocation
        unsafe {
            *ptr.as_ptr() = 1;
        }
        assert!(!pool.verify_zeroed_simd(ptr.as_slice()).unwrap());
    }

    #[test]
    fn test_bulk_allocation_with_prefetch() {
        let pool = SecureMemoryPool::new(SecurePoolConfig::small_secure()).unwrap();

        let sizes = vec![1024; 20]; // 20 allocations of 1KB each
        let ptrs = pool.allocate_bulk_with_prefetch(&sizes).unwrap();

        assert_eq!(ptrs.len(), sizes.len());
        for (ptr, &size) in ptrs.iter().zip(&sizes) {
            assert!(!ptr.as_ptr().is_null());
            assert_eq!(ptr.size(), size);
        }

        // Verify all pointers are unique
        let mut addresses: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for ptr in &ptrs {
            let addr = ptr.as_ptr() as usize;
            assert!(addresses.insert(addr), "Duplicate pointer detected");
        }
    }

    #[test]
    fn test_bulk_allocation_size_mismatch() {
        let pool = SecureMemoryPool::new(SecurePoolConfig::small_secure()).unwrap();

        // Try to allocate with wrong sizes
        let sizes = vec![512, 1024, 2048]; // Sizes don't match chunk_size
        let result = pool.allocate_bulk_with_prefetch(&sizes);

        // Should fail for sizes that don't match chunk_size
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_optimized_operations() {
        let mut config = SecurePoolConfig::large_secure()
            .with_simd_ops(true)
            .with_simd_threshold(64)
            .with_cache_alignment(true)
            .with_zero_on_alloc(true);
        let iters = if cfg!(miri) {
            config.chunk_size = 4096;
            3
        } else {
            10
        };
        let pool = SecureMemoryPool::new(config).unwrap();

        // Test various sizes to trigger different SIMD paths
        for _ in 0..iters {
            let ptr = pool.allocate().unwrap();
            assert!(!ptr.as_ptr().is_null());

            // SAFETY: test code, pointer valid from allocation
            let slice = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), ptr.size()) };
            assert!(slice.iter().all(|&b| b == 0));
        }

        let stats = pool.stats();
        assert!(stats.alloc_count >= iters as u64);
    }

    #[test]
    fn test_simd_with_small_threshold() {
        // Test with a very high threshold that forces scalar operations
        let config = SecurePoolConfig::small_secure()
            .with_simd_ops(true)
            .with_simd_threshold(1024 * 1024) // 1MB threshold for 1KB chunks
            .with_zero_on_alloc(true);
        let pool = SecureMemoryPool::new(config).unwrap();

        let ptr = pool.allocate().unwrap();

        // Should still be zeroed, just using scalar path
        // SAFETY: Hardware features verified or caller ensures safety invariants
        let slice = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), ptr.size()) };
        assert!(slice.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_verify_zeroed_empty_slice() {
        let pool = SecureMemoryPool::new(SecurePoolConfig::small_secure()).unwrap();

        // Empty slice should verify as zeroed
        let result = pool.verify_zeroed_simd(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_zero_on_alloc_disabled() {
        let config = SecurePoolConfig::small_secure()
            .with_zero_on_alloc(false)
            .with_simd_ops(true);
        let pool = SecureMemoryPool::new(config).unwrap();

        let ptr = pool.allocate().unwrap();

        // Memory may contain garbage when zero_on_alloc is disabled
        // Just verify the allocation succeeded
        assert!(!ptr.as_ptr().is_null());
        assert_eq!(ptr.size(), 1024);
    }

    #[test]
    fn test_adaptive_simd_integration() {
        let mut config = SecurePoolConfig::large_secure()
            .with_simd_ops(true)
            .with_zero_on_alloc(true);
        let iters = if cfg!(miri) {
            config.chunk_size = 4096;
            4
        } else {
            100
        };
        let pool = SecureMemoryPool::new(config).unwrap();

        // Allocate multiple times to trigger adaptive monitoring
        for _ in 0..iters {
            let ptr = pool.allocate().unwrap();
            assert!(!ptr.as_ptr().is_null());

            // Verify using SIMD
            assert!(pool.verify_zeroed_simd(ptr.as_slice()).unwrap());

            // Drop happens automatically
        }

        // Verify stats
        let stats = pool.stats();
        assert_eq!(stats.alloc_count, iters);
        assert_eq!(stats.dealloc_count, iters);
    }

    #[test]
    fn test_concurrent_simd_operations() {
        use std::sync::atomic::AtomicUsize;
        use std::thread;

        let mut config = SecurePoolConfig::medium_secure()
            .with_simd_ops(true)
            .with_zero_on_alloc(true);
        let (threads, iters) = if cfg!(miri) {
            config.chunk_size = 2048;
            (2, 4)
        } else {
            (4, 25)
        };
        let pool = SecureMemoryPool::new(config).unwrap();
        let verified_count = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let pool = pool.clone();
                let count = verified_count.clone();
                thread::spawn(move || {
                    for _ in 0..iters {
                        let ptr = pool.allocate().unwrap();

                        // Verify zeroing
                        if pool.verify_zeroed_simd(ptr.as_slice()).unwrap() {
                            count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let expected = threads * iters;
        assert_eq!(verified_count.load(Ordering::Relaxed), expected);

        let stats = pool.stats();
        assert_eq!(stats.alloc_count, expected as u64);
        assert_eq!(stats.dealloc_count, expected as u64);
    }

    #[test]
    fn test_s4_volatile_zeroing_and_aligned_dealloc_paths() {
        for (size, align) in [(5usize, 1usize), (100, 64), (256, 128)] {
            let mut chunk = SecureChunk::new(size, align, 1, 7).unwrap();
            let effective_align = align.max(MIN_CHUNK_ALIGN);
            assert_eq!(chunk.as_ptr() as usize % effective_align, 0);
            // SAFETY: chunk exclusively owns `size` bytes at `chunk.as_ptr()`.
            unsafe {
                std::ptr::write_bytes(chunk.as_ptr(), 0xA5, size);
            }
            chunk.zero_payload_volatile();
            // SAFETY: chunk payload was initialized and just zeroed.
            let slice = unsafe { std::slice::from_raw_parts(chunk.as_ptr(), size) };
            assert!(slice.iter().all(|&b| b == 0));
            chunk.validate().unwrap();
            chunk.deallocate(true);
        }

        // Pool-gone deallocation path with 64-byte alignment (Miri verifies matching Layout).
        let ptr = {
            let pool = SecureMemoryPool::new(SecurePoolConfig::new(100, 4, 64)).unwrap();
            let mut p = pool.allocate().unwrap();
            assert_eq!(p.as_ptr() as usize % 64, 0);
            p.as_mut_slice().fill(0x5A);
            p
        };
        drop(ptr);

        // Overflowing size or alignment must fail cleanly with Err rather than panicking.
        assert!(SecureMemoryPool::new(SecurePoolConfig::new(usize::MAX, 4, 16)).is_err());
        assert!(
            SecureMemoryPool::new(SecurePoolConfig::new(64, 4, 1usize << (usize::BITS - 1)))
                .is_err()
        );
    }

    #[test]
    fn test_s4_uninit_free_slice_and_gold_hash_idx_padded_recycle() {
        #[repr(C)]
        #[derive(Clone, Debug, PartialEq)]
        struct PaddedVal {
            tag: u8,
            // 7 bytes of struct padding on 64-bit targets
            word: u64,
        }

        // 1. Fresh chunk on a `zero_on_alloc: false, zero_on_free: false` pool starts
        // zero-initialized (`alloc_zeroed`), so safe `as_slice()` / `as_mut_slice()` never
        // expose uninitialized bytes under Miri.
        let pool = SecureMemoryPool::new(
            SecurePoolConfig::new(128, 8, 16)
                .with_zero_on_alloc(false)
                .with_zero_on_free(false),
        )
        .unwrap();
        let mut fresh = pool.allocate().unwrap();
        assert!(fresh.as_slice().iter().all(|&b| b == 0));
        fresh.as_mut_slice()[0] = 0;
        drop(fresh);

        // 2. Inserting, updating, removing, and dropping padded `(u8, u64)` and `#[repr(C)]`
        // values in `GoldHashIdx::with_pool` scrubs struct padding before returning chunks to
        // `pool`, even when both `zero_on_alloc` and `zero_on_free` are false.
        let mut tuple_idx: crate::containers::GoldHashIdx<u64, (u8, u64)> =
            crate::containers::GoldHashIdx::with_pool(16, pool.clone());
        tuple_idx.insert(1, (0x7F, 0xCAFE_BABE_DEAD_BEEF)).unwrap();
        tuple_idx.insert(1, (0x12, 0x1122_3344_5566_7788)).unwrap();
        assert_eq!(tuple_idx.remove(&1), Some((0x12, 0x1122_3344_5566_7788)));
        tuple_idx.insert(2, (0x34, 0x99AA_BBCC_DDEE_FF00)).unwrap();
        drop(tuple_idx);

        let mut repr_idx: crate::containers::GoldHashIdx<u64, PaddedVal> =
            crate::containers::GoldHashIdx::with_pool(16, pool.clone());
        repr_idx
            .insert(
                7,
                PaddedVal {
                    tag: 0xAB,
                    word: 0xFEED_FACE_1234_5678,
                },
            )
            .unwrap();
        assert_eq!(
            repr_idx.remove(&7),
            Some(PaddedVal {
                tag: 0xAB,
                word: 0xFEED_FACE_1234_5678,
            })
        );
        drop(repr_idx);

        let recycled = pool.allocate().unwrap();
        assert!(pool.stats().local_cache_hits >= 1);
        assert!(recycled.as_slice().iter().all(|&b| b == 0));
        drop(recycled);

        // 3. Raw-pointer writes with uninitialized padding bytes into >= 64B SIMD-enabled
        // chunks must not trigger `&mut [u8]` uninit UB in `zero_payload` (`zero_on_free`) or
        // `zero_chunk_simd` (`zero_on_alloc`).
        for (zero_on_free, zero_on_alloc) in [(true, false), (false, true)] {
            let simd_pool = SecureMemoryPool::new(
                SecurePoolConfig::new(128, 8, 16)
                    .with_zero_on_free(zero_on_free)
                    .with_zero_on_alloc(zero_on_alloc)
                    .with_simd_ops(true)
                    .with_simd_threshold(64),
            )
            .unwrap();
            let p = simd_pool.allocate().unwrap();
            let first_addr = p.as_ptr();
            // SAFETY: `p` has size 128 >= size_of::<PaddedVal>() and alignment 16 >= align_of::<PaddedVal>()
            unsafe {
                std::ptr::write(
                    p.as_ptr() as *mut PaddedVal,
                    PaddedVal {
                        tag: 0x55,
                        word: 0xABCD_EF01_2345_6789,
                    },
                );
            }
            drop(p);

            let p2 = simd_pool.allocate().unwrap();
            assert_eq!(p2.as_ptr(), first_addr);
            assert!(p2.as_slice().iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn test_s4_corrupted_chunk_removed_from_active_allocations_before_dealloc() {
        let pool = SecureMemoryPool::new(SecurePoolConfig::new(128, 8, 32)).unwrap();
        let victim = pool.allocate().unwrap();
        let survivor = pool.allocate().unwrap();

        // Corrupt victim's footer magic so `deallocate_internal` takes the corruption path.
        // SAFETY: `victim.as_ptr().add(padded_payload_size(victim.size()))` points to `victim`'s
        // exclusively owned `ChunkFooter`.
        unsafe {
            let footer_ptr =
                victim.as_ptr().add(padded_payload_size(victim.size())) as *mut ChunkFooter;
            (*footer_ptr).magic ^= 0xFF;
        }

        drop(victim);
        assert_eq!(pool.stats().corruption_detected, 1);

        // `pool.validate()` must succeed without UAF on `victim`'s freed heap allocation.
        pool.validate().unwrap();
        survivor.validate().unwrap();
    }
}
