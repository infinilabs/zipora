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
use crate::memory::cache_layout::AccessPattern;
use crate::memory::simd_ops::{fast_copy_cache_optimized, fast_fill, fast_prefetch_range};
use crate::simd::{AdaptiveSimdSelector, Operation};
use bytemuck::Pod;
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::slice;

/// Underlying file mapping: writable `MAP_SHARED` (`MmapMut`) for mutable
/// vectors, or read-only `PROT_READ` (`Mmap`) for `config.read_only` vectors.
///
/// Using `Mmap` (`PROT_READ`) rather than `MmapMut::map_copy` (`MAP_PRIVATE`
/// with `PROT_READ | PROT_WRITE`) for read-only vectors ensures that
/// `populate_pages: true` (`MAP_POPULATE` on Linux) prefaults clean
/// file-backed page-cache pages (`RssFile`) rather than write-faulting every
/// page into private anonymous copy-on-write memory (`RssAnon`), and lets
/// hardware page protection reject any stray write.
enum MappingKind {
    ReadWrite(MmapMut),
    ReadOnly(Mmap),
}

impl MappingKind {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::ReadWrite(m) => m.len(),
            Self::ReadOnly(m) => m.len(),
        }
    }

    #[inline]
    fn get(&self, range: std::ops::RangeTo<usize>) -> Option<&[u8]> {
        match self {
            Self::ReadWrite(m) => m.get(range),
            Self::ReadOnly(m) => m.get(range),
        }
    }

    fn advise(&self, advice: memmap2::Advice) -> std::io::Result<()> {
        match self {
            Self::ReadWrite(m) => m.advise(advice),
            Self::ReadOnly(m) => m.advise(advice),
        }
    }

    fn flush(&self) -> std::io::Result<()> {
        match self {
            Self::ReadWrite(m) => m.flush(),
            Self::ReadOnly(_) => Ok(()),
        }
    }

    fn flush_range(&self, offset: usize, len: usize) -> std::io::Result<()> {
        match self {
            Self::ReadWrite(m) => m.flush_range(offset, len),
            Self::ReadOnly(_) => Ok(()),
        }
    }
}

/// A file-backed memory mapping (`MAP_SHARED` writable or `PROT_READ` read-only).
///
/// Writes through a writable `map` land in the OS page cache of the backing
/// file (`MAP_SHARED`), so data persists across drop/reopen even without an
/// explicit `flush()`; `flush()` only adds durability (`msync` to disk).
/// For read-only vectors the mapping is `PROT_READ` (`Mmap`), so page
/// protection prevents writes and `MAP_POPULATE` faults clean file-backed
/// pages without allocating private anonymous memory.
struct FileMapping {
    /// Kept open for `set_len()` during growth; the mapping itself holds its
    /// own reference to the file, so ordering of drops does not matter.
    file: File,
    map: MappingKind,
}

/// Header for memory-mapped vector files (80 bytes, little-endian header fields on disk)
///
/// Binary layout (C2.1):
/// - `0..8`: `magic: u64` (`0x4D4D41505F564543` LE)
/// - `8..10`: `version: u16` (`2` LE)
/// - `10..12`: `flags: u16` (LE: bit 0 = little-endian host payload, bit 4 = 64-bit header field width; `0x0011` on little-endian, `0x0010` on big-endian)
/// - `12..16`: `element_size: u32` (LE)
/// - `16..24`: `length: u64` (LE)
/// - `24..32`: `capacity: u64` (LE)
/// - `32..40`: `extended_align: u64` (LE: `align_of::<T>()` when `align_of::<T>() > 16`, else `0` so `32..80` stays all-zero for standard alignments `<= 16` where `data_offset() == 80`)
/// - `40..80`: `reserved: [u8; 40]` (zero)
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
struct MmapVecHeader {
    data: [u8; 80],
}

const MMAP_VEC_MAGIC: u64 = 0x4D4D41505F564543; // "MMAP_VEC"
const MMAP_VEC_VERSION: u16 = 2;
const MMAP_VEC_FLAGS: u16 = if cfg!(target_endian = "little") {
    0x0011
} else {
    0x0010
}; // bit 0 = little-endian host payload, bit 4 = 64-bit header field width
const HEADER_SIZE: usize = std::mem::size_of::<MmapVecHeader>();
// The header is accessed in place at the (page-aligned) start of a mapping
// and its fields are decoded byte-wise, so it needs no alignment.
const _: () = assert!(std::mem::align_of::<MmapVecHeader>() == 1);

/// The bytes of a slice of `Pod` values.
fn pod_bytes<T: Pod>(values: &[T]) -> Result<&[u8]> {
    bytemuck::try_cast_slice(values)
        .map_err(|e| ZiporaError::invalid_data(format!("MmapVec byte view: {:?}", e)))
}

/// The bytes of a mutable slice of `Pod` values.
fn pod_bytes_mut<T: Pod>(values: &mut [T]) -> Result<&mut [u8]> {
    bytemuck::try_cast_slice_mut(values)
        .map_err(|e| ZiporaError::invalid_data(format!("MmapVec byte view: {:?}", e)))
}

impl MmapVecHeader {
    /// Alignment marker stored in bytes `32..40`: `0` when `align_of::<T>() <= 16`
    /// (the 80-byte header is already a multiple of 16, so `data_offset() == 80`
    /// and bytes `32..80` remain all-zero for v2 wire compatibility), and
    /// `align_of::<T>() as u64` when `align_of::<T>() > 16` (where `data_offset()`
    /// is padded past 80).
    #[inline]
    const fn required_extended_align<T>() -> u64 {
        let align = std::mem::align_of::<T>();
        if align > 16 { align as u64 } else { 0 }
    }

    fn new<T>() -> Self {
        let mut data = [0u8; 80];
        data[0..8].copy_from_slice(&MMAP_VEC_MAGIC.to_le_bytes());
        data[8..10].copy_from_slice(&MMAP_VEC_VERSION.to_le_bytes());
        data[10..12].copy_from_slice(&MMAP_VEC_FLAGS.to_le_bytes());
        data[12..16].copy_from_slice(&(std::mem::size_of::<T>() as u32).to_le_bytes());
        data[32..40].copy_from_slice(&Self::required_extended_align::<T>().to_le_bytes());
        Self { data }
    }

    #[inline]
    fn magic(&self) -> u64 {
        u64::from_le_bytes([
            self.data[0],
            self.data[1],
            self.data[2],
            self.data[3],
            self.data[4],
            self.data[5],
            self.data[6],
            self.data[7],
        ])
    }

    #[inline]
    fn version(&self) -> u16 {
        u16::from_le_bytes([self.data[8], self.data[9]])
    }

    #[inline]
    fn flags(&self) -> u16 {
        u16::from_le_bytes([self.data[10], self.data[11]])
    }

    #[inline]
    fn element_size(&self) -> u32 {
        u32::from_le_bytes([self.data[12], self.data[13], self.data[14], self.data[15]])
    }

    #[inline]
    fn length(&self) -> u64 {
        u64::from_le_bytes([
            self.data[16],
            self.data[17],
            self.data[18],
            self.data[19],
            self.data[20],
            self.data[21],
            self.data[22],
            self.data[23],
        ])
    }

    #[inline]
    fn set_length(&mut self, len: u64) {
        self.data[16..24].copy_from_slice(&len.to_le_bytes());
    }

    #[inline]
    fn capacity(&self) -> u64 {
        u64::from_le_bytes([
            self.data[24],
            self.data[25],
            self.data[26],
            self.data[27],
            self.data[28],
            self.data[29],
            self.data[30],
            self.data[31],
        ])
    }

    #[inline]
    fn set_capacity(&mut self, cap: u64) {
        self.data[24..32].copy_from_slice(&cap.to_le_bytes());
    }

    #[inline]
    fn extended_align(&self) -> u64 {
        u64::from_le_bytes([
            self.data[32],
            self.data[33],
            self.data[34],
            self.data[35],
            self.data[36],
            self.data[37],
            self.data[38],
            self.data[39],
        ])
    }

    fn validate<T>(&self) -> Result<()> {
        if self.magic() != MMAP_VEC_MAGIC {
            return Err(ZiporaError::invalid_data("Invalid magic number"));
        }
        if self.version() != MMAP_VEC_VERSION {
            return Err(ZiporaError::invalid_data("Unsupported version"));
        }
        if self.flags() != MMAP_VEC_FLAGS {
            return Err(ZiporaError::invalid_data(
                "Incompatible endianness or word-size flags in MmapVecHeader",
            ));
        }
        if self.element_size() != std::mem::size_of::<T>() as u32 {
            return Err(ZiporaError::invalid_data("Element size mismatch"));
        }
        if self.extended_align() != Self::required_extended_align::<T>() {
            return Err(ZiporaError::invalid_data("Element alignment mismatch"));
        }
        if self.length() > self.capacity() {
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
    /// Enable read-only mode (valid only for [`MmapVec::open`]; rejected by
    /// [`MmapVec::create`])
    pub read_only: bool,
    /// Enable population of pages on creation (`MAP_POPULATE` on Linux)
    pub populate_pages: bool,
    /// Advise the kernel to back the mapping with transparent huge pages
    /// (`MADV_HUGEPAGE` on Linux).
    ///
    /// Limitation: `MADV_HUGEPAGE` is a best-effort kernel hint that takes
    /// effect on `tmpfs`/shared-memory mappings and on kernels with
    /// file-backed transparent huge pages enabled; on regular filesystems
    /// without file-THP support it is a silent no-op.
    pub use_huge_pages: bool,
    /// Flush (`msync`) modified ranges and the header on mutating methods
    /// that return `Result`.
    ///
    /// Limitation: direct writes through borrows returned by
    /// [`MmapVec::get_mut`] or [`MmapVec::as_mut_slice`] land in the
    /// `MAP_SHARED` OS page cache immediately but cannot trigger an automatic
    /// `msync` when the borrow ends (call [`MmapVec::sync`] explicitly if
    /// crash durability is required), and [`MmapVec::pop`] returns `Option<T>`
    /// so any `msync` error from its header flush is ignored.
    pub sync_on_write: bool,
    /// Expected access pattern, forwarded to the kernel via madvise
    /// (Sequential/ReadHeavy -> MADV_SEQUENTIAL, Random -> MADV_RANDOM)
    pub access_pattern: AccessPattern,
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
            access_pattern: AccessPattern::Mixed,
        }
    }
}

impl MmapVecConfig {
    /// Create a builder (returns Self with Default values for fluent configuration).
    pub fn builder() -> Self {
        Self::default()
    }
    /// Set initial capacity.
    pub fn with_initial_capacity(mut self, v: usize) -> Self {
        self.initial_capacity = v;
        self
    }
    /// Set growth factor.
    pub fn with_growth_factor(mut self, v: f64) -> Self {
        self.growth_factor = v;
        self
    }
    /// Set populate pages.
    pub fn with_populate_pages(mut self, v: bool) -> Self {
        self.populate_pages = v;
        self
    }
    /// Set sync on write.
    pub fn with_sync_on_write(mut self, v: bool) -> Self {
        self.sync_on_write = v;
        self
    }
    /// Finalize.
    pub fn build(self) -> Self {
        self
    }

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
            growth_factor: 1.5,            // Conservative growth
            populate_pages: false,         // Lazy loading
            use_huge_pages: true,
            ..Self::default()
        }
    }

    /// Create configuration for persistent cache
    pub fn persistent_cache() -> Self {
        Self {
            initial_capacity: 16384,
            growth_factor: 2.0,  // Aggressive growth
            sync_on_write: true, // Ensure persistence
            ..Self::default()
        }
    }

    /// Create configuration optimized for performance
    pub fn performance_optimized() -> Self {
        Self {
            initial_capacity: 8192,
            growth_factor: 1.618, // Golden ratio growth
            populate_pages: true,
            use_huge_pages: cfg!(target_os = "linux"),
            sync_on_write: false,
            ..Self::default()
        }
    }

    /// Create configuration optimized for memory usage
    pub fn memory_optimized() -> Self {
        Self {
            initial_capacity: 256,
            growth_factor: 1.4, // Conservative growth
            populate_pages: false,
            use_huge_pages: false,
            sync_on_write: false,
            ..Self::default()
        }
    }

    /// Create configuration for real-time systems
    pub fn realtime() -> Self {
        Self {
            initial_capacity: 1024,
            growth_factor: 1.5,   // Predictable growth
            populate_pages: true, // Avoid page faults
            use_huge_pages: true, // Reduce TLB misses
            sync_on_write: false, // Avoid I/O in real-time path
            ..Self::default()
        }
    }

    /// Validate configuration parameters.
    ///
    /// # Errors
    /// Returns [`ZiporaError::InvalidParameter`] if `growth_factor` is not
    /// finite or outside `(1.0, 1024.0]`.
    pub fn validate(&self) -> Result<()> {
        if !self.growth_factor.is_finite()
            || self.growth_factor <= 1.0
            || self.growth_factor > 1024.0
        {
            return Err(ZiporaError::invalid_parameter(format!(
                "MmapVecConfig::growth_factor must be finite and in (1.0, 1024.0], got {}",
                self.growth_factor
            )));
        }
        Ok(())
    }
}

/// Statistics for memory-mapped vectors
#[derive(Debug, Clone)]
pub struct MmapVecStats {
    /// Number of elements
    pub len: usize,
    /// Total capacity
    pub capacity: usize,
    /// Size of each element in bytes
    pub element_size: usize,
    /// Size of header in bytes
    pub header_size: usize,
    /// Size of data section in bytes
    pub data_size: usize,
    /// Total size in bytes
    pub total_size: usize,
    /// Utilization ratio (len / capacity)
    pub utilization: f64,
    /// Path to backing file
    pub file_path: PathBuf,
    /// Whether the vector is read-only
    pub read_only: bool,
    /// Growth factor
    pub growth_factor: f64,
}

impl MmapVecStats {
    /// Get memory efficiency as a percentage
    pub fn memory_efficiency(&self) -> f64 {
        self.utilization * 100.0
    }

    /// Get wasted space in bytes
    pub fn wasted_space(&self) -> usize {
        (self.capacity - self.len) * self.element_size
    }

    /// Check if the vector needs compaction
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.utilization < threshold
    }
}

/// Memory-mapped vector implementation
///
/// # Pointer-stability invariant (H13)
///
/// Any growth operation (`push` → `grow`, `reserve`, `resize`,
/// `resize_to_capacity`) may **remap the backing file to a different
/// address**: the cached `header`/`data` pointers are re-derived from a
/// new mapping, which is fully checked before the old mapping is unmapped.
/// Consequently,
/// raw pointers or slices derived from this vector (`as_slice`,
/// `as_mut_slice`, `get`, iterator items) must never be held across any
/// `&mut self` call — Rust's borrow rules enforce this for safe callers, and
/// every internal access via `checked_data` / `mapped_slice` reads the cached
/// `self.data` pointer *after* any potential remap. Unsafe code must not
/// stash raw pointers derived from `self.data` across growth operations.
///
/// # Length and capacity
///
/// Each handle owns its length and capacity. `open` reads them from the
/// file header once and checks them against the mapping; every change made
/// through this handle is written through to the header, but the header is
/// never read back. A header rewritten by another handle or process
/// therefore cannot widen this handle's views past its mapping; equally,
/// elements appended through another handle are not seen by this one.
///
/// # Limitations
///
/// - **External file truncation (`SIGBUS`) and concurrent mutation**:
///   [`MmapVec::create`], [`MmapVec::open`], and [`MmapVec::with_capacity_simd`]
///   wrap `memmap2`'s `unsafe` file mappings behind safe constructors (`T: Pod`).
///   The OS still requires the backing file to remain at least as long as the
///   mapped region and unmutated by external writers while references are live:
///   if another process (or a second [`MmapVec`] handle via [`MmapVec::create`]
///   or [`MmapVec::shrink_to_fit`] on the same path) truncates the file below
///   this handle's mapped length, reading or writing the unmapped tail raises
///   `SIGBUS` on Unix (process abort, not a recoverable Rust panic), and
///   concurrent writes to the mapped bytes while a `&[T]` or `&T` borrow is
///   live violate Rust's aliasing rules (undefined behavior even for `T: Pod`).
/// - **Single-writer handle model**: because each handle caches `len` and `cap`
///   and writes them back to the header on mutation without re-reading the
///   header, two writable handles open on the same file will overwrite each
///   other's header fields, and a read-only handle will not observe elements
///   appended after it was opened without reopening the file.
/// - **Host-native payload layout**: header fields are little-endian on disk
///   with `flags` bit 0 recording host endianness (`0x0011` on little-endian,
///   `0x0010` on big-endian) and bytes `32..40` recording extended element
///   alignment (`align_of::<T>()` when `> 16`, else `0`), while element bytes
///   are mapped directly in the host's `T: Pod` representation (`open` rejects
///   files with mismatched endianness flags, element size, or extended
///   alignment).
pub struct MmapVec<T> {
    /// Path to the backing file
    file_path: PathBuf,
    /// File-backed memory mapping
    mmap: Option<FileMapping>,
    /// Configuration
    config: MmapVecConfig,
    /// Cached header pointer (into `mmap`; `Some` for writable mappings, `None` for read-only)
    header: Option<NonNull<MmapVecHeader>>,
    /// Cached data pointer (into `mmap`, validated aligned for `T` by `view_of`)
    data: NonNull<T>,
    /// Number of elements; invariant `len <= cap`
    len: usize,
    /// Capacity in elements; invariant `cap <= mapped_cap`
    cap: usize,
    /// Whole elements of `T` the current mapping holds after the header and
    /// its padding (`usize::MAX` for zero-sized `T`); set by `view_of`
    /// together with `header` and `data`
    mapped_cap: usize,
    /// Whether this is a temporary file that should be deleted on drop
    is_temp_file: bool,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

impl<T> MmapVec<T>
where
    T: Pod + 'static, // Pod ensures every bit pattern is valid for T (prevents UB on mmap read)
{
    /// Create a new memory-mapped vector at `path`, truncating any existing file.
    ///
    /// # Errors
    /// Returns an error if `config.read_only` is `true`, if `config` fails
    /// [`MmapVecConfig::validate`], if the initial capacity overflows the
    /// file-size calculation, or if the file cannot be created, sized, or
    /// mapped.
    ///
    /// # Limitations
    /// Truncating `path` while another [`MmapVec`] handle or process has the
    /// same file mapped will cause that other mapping to raise `SIGBUS` on
    /// Unix when it accesses pages beyond the new file length; see the
    /// struct-level [Limitations](Self#limitations).
    pub fn create<P: AsRef<Path>>(path: P, config: MmapVecConfig) -> Result<Self> {
        if config.read_only {
            return Err(ZiporaError::invalid_parameter(
                "MmapVec::create cannot use a read-only configuration",
            ));
        }
        config.validate()?;

        let file_path = path.as_ref().to_path_buf();

        // Size first: an invalid capacity must fail before an existing file
        // at `path` is truncated.
        let file_len = Self::calculate_file_size(config.initial_capacity)?;

        // Create the backing file; it is sized and mapped through this
        // handle, never re-opened by path.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| ZiporaError::io_error(format!("Failed to create file: {}", e)))?;
        Self::from_new_file(file, file_path, config, file_len)
    }

    /// Open an existing memory-mapped vector at `path`.
    ///
    /// # Errors
    /// Returns an error if `config` fails [`MmapVecConfig::validate`], if
    /// `path` does not exist, cannot be opened or mapped, has an
    /// invalid/mismatched `MmapVecHeader` (magic, version, endianness or
    /// word-size flags, `size_of::<T>()`, extended `align_of::<T>()`), or is
    /// smaller than the header-declared capacity.
    ///
    /// # Limitations
    /// The returned handle caches `len` and `cap` from the header at open time
    /// and assumes no other process or handle truncates the file below the
    /// mapped length or mutates mapped bytes concurrently with live references;
    /// see the struct-level [Limitations](Self#limitations).
    pub fn open<P: AsRef<Path>>(path: P, config: MmapVecConfig) -> Result<Self> {
        config.validate()?;
        let file_path = path.as_ref().to_path_buf();

        if !file_path.exists() {
            return Err(ZiporaError::invalid_data("File does not exist"));
        }

        let file = if config.read_only {
            File::open(&file_path)
        } else {
            OpenOptions::new().read(true).write(true).open(&file_path)
        }
        .map_err(|e| ZiporaError::io_error(format!("Failed to open file: {}", e)))?;

        // Create memory mapping
        let mut map = Self::map_file(&file, &config, None)?;

        // Copy the header out once: from here on this handle's length and
        // capacity are authoritative (see "Length and capacity" above).
        let header = map
            .get(..HEADER_SIZE)
            .and_then(|bytes| <[u8; HEADER_SIZE]>::try_from(bytes).ok())
            .map(|data| MmapVecHeader { data })
            .ok_or_else(|| {
                ZiporaError::invalid_data("Backing file too small for MmapVec header")
            })?;
        let (header_ptr, data, mapped_cap) = Self::view_of(&mut map, Self::data_offset())?;

        // The mapping is exactly the file; a header declaring more capacity
        // than the file holds would make element access read past EOF.
        let (len, cap) = Self::counts_from_header(&header, mapped_cap)?;

        Ok(Self {
            file_path,
            mmap: Some(FileMapping { file, map }),
            config,
            header: header_ptr,
            data,
            len,
            cap,
            mapped_cap,
            is_temp_file: false,
            _phantom: PhantomData,
        })
    }

    /// Get the number of elements in the vector
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the capacity of the vector
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Push an element to the end of the vector
    ///
    /// # Errors
    /// Returns an error if the vector is read-only, if the length would
    /// overflow `usize`, or if growing the file or its mapping fails.
    #[inline]
    pub fn push(&mut self, value: T) -> Result<()> {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        let index = self.len;
        let new_len = index
            .checked_add(1)
            .ok_or_else(|| ZiporaError::invalid_data("MmapVec length overflows usize"))?;

        // Check if we need to grow the vector
        if new_len > self.cap {
            self.grow()?;
        }

        // Write the element directly into the validated mapped capacity
        if index >= self.cap || self.cap > self.mapped_cap {
            return Err(ZiporaError::out_of_bounds(index, self.cap));
        }
        // SAFETY: `!self.config.read_only` guarantees a writable `MmapMut`
        // mapping; `self.data` was validated by `view_of` to be aligned for `T`
        // and backed by `self.mapped_cap` elements in `self.mmap`; the check
        // above verified `index < self.cap <= self.mapped_cap`; and `&mut self`
        // guarantees exclusive access.
        unsafe { self.data.as_ptr().add(index).write(value) };

        // Update length in header
        self.store_len(new_len)?;

        // Sync if requested (only the written element + header)
        if self.config.sync_on_write {
            self.sync_range(index, 1)?;
        }

        Ok(())
    }

    /// Pop an element from the end of the vector.
    ///
    /// Limitation: because this method returns `Option<T>` rather than
    /// `Result<Option<T>>`, any `msync` error when `config.sync_on_write` is
    /// enabled is ignored (use [`MmapVec::pop_bulk_simd`]`(1)` or call
    /// [`MmapVec::sync`] if `msync` errors must be surfaced).
    pub fn pop(&mut self) -> Option<T> {
        if self.config.read_only {
            return None;
        }

        let new_len = self.len.checked_sub(1)?;

        // Read the element
        let value = *self.as_slice().get(new_len)?;

        // Update length
        self.store_len(new_len).ok()?;

        // Sync if requested (only the header changed)
        if self.config.sync_on_write {
            let _ = self.sync_header();
        }

        Some(value)
    }

    /// Get an element at the specified index
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    /// Get a mutable reference to an element at the specified index.
    ///
    /// Limitation: mutations through the returned `&mut T` are visible via
    /// `MAP_SHARED` immediately, but `config.sync_on_write` cannot trigger an
    /// automatic `msync` when the borrow ends; call [`MmapVec::sync`] explicitly
    /// if crash durability is required.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if self.config.read_only {
            return None;
        }
        self.as_mut_slice().get_mut(index)
    }

    /// Get a slice of all elements
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `self.data` was validated by `view_of` to be aligned for `T`
        // and backed by `self.mapped_cap` elements in the live mapping in
        // `self.mmap` (`self.data`, `self.mapped_cap`, and `self.mmap` are
        // only ever replaced together), and every path that sets `self.len`
        // (`open`, `from_new_file`, `store_counts`, `store_len`) maintains
        // `self.len <= self.cap <= self.mapped_cap`. `T: Pod` makes any mapped
        // bit pattern valid, and `&self` prevents remapping or unmapping while
        // the returned slice lives.
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    /// Get a mutable slice of all elements.
    ///
    /// Limitation: mutations through the returned `&mut [T]` are visible via
    /// `MAP_SHARED` immediately, but `config.sync_on_write` cannot trigger an
    /// automatic `msync` when the borrow ends; call [`MmapVec::sync`] explicitly
    /// if crash durability is required.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.config.read_only {
            return &mut [];
        }

        // SAFETY: as in `as_slice`; in addition, `&mut self` guarantees
        // exclusive access and `!self.config.read_only` guarantees `self.mmap`
        // holds a writable `MappingKind::ReadWrite` mapping.
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }

    /// Clear all elements from the vector
    pub fn clear(&mut self) -> Result<()> {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        self.store_len(0)?;

        if self.config.sync_on_write {
            self.sync_header()?;
        }

        Ok(())
    }

    /// Reserve capacity for at least `additional` more elements
    ///
    /// # Errors
    /// Returns an error, leaving the vector unchanged, if the vector is
    /// read-only, if `len() + additional` overflows `usize`, or if growing
    /// the file or its mapping fails.
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        let current_capacity = self.cap;
        let required_capacity = self.len.checked_add(additional).ok_or_else(|| {
            ZiporaError::invalid_parameter("MmapVec reserve: len + additional overflows usize")
        })?;

        if required_capacity > current_capacity {
            let grown = self.scaled_capacity(current_capacity)?;
            let new_capacity = required_capacity.max(grown);
            self.resize_to_capacity(new_capacity)?;
        }

        Ok(())
    }

    /// Sync changes to disk (msync the shared mapping)
    ///
    /// Writes are already visible to other readers of the file via the OS
    /// page cache; this only forces durability against a system crash.
    pub fn sync(&self) -> Result<()> {
        if let Some(mmap) = &self.mmap {
            mmap.map
                .flush()
                .map_err(|e| ZiporaError::io_error(format!("Failed to sync mapping: {}", e)))?;
        }
        Ok(())
    }

    /// Flush only the header (length/capacity updates).
    fn sync_header(&self) -> Result<()> {
        if let Some(mmap) = &self.mmap {
            mmap.map
                .flush_range(0, HEADER_SIZE)
                .map_err(|e| ZiporaError::io_error(format!("Failed to sync header: {}", e)))?;
        }
        Ok(())
    }

    /// Flush the header plus `count` elements starting at index `start`.
    ///
    /// sync_on_write used to msync the entire mapping on every write; only
    /// the touched pages need durability.
    fn sync_range(&self, start: usize, count: usize) -> Result<()> {
        if let Some(mmap) = &self.mmap {
            let len = count * std::mem::size_of::<T>();
            if len > 0 {
                let offset = Self::data_offset() + start * std::mem::size_of::<T>();
                mmap.map
                    .flush_range(offset, len)
                    .map_err(|e| ZiporaError::io_error(format!("Failed to sync range: {}", e)))?;
            }
        }
        self.sync_header()
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.file_path
    }

    /// Get memory usage statistics
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.capacity() * std::mem::size_of::<T>() + Self::data_offset()
    }

    /// Get detailed statistics about the vector
    pub fn stats(&self) -> MmapVecStats {
        let len = self.len();
        let capacity = self.capacity();
        let element_size = std::mem::size_of::<T>();
        let header_size = Self::data_offset(); // header + alignment padding
        let data_size = capacity * element_size;
        let total_size = header_size + data_size;
        let utilization = if capacity > 0 {
            len as f64 / capacity as f64
        } else {
            0.0
        };

        MmapVecStats {
            len,
            capacity,
            element_size,
            header_size,
            data_size,
            total_size,
            utilization,
            file_path: self.file_path.clone(),
            read_only: self.config.read_only,
            growth_factor: self.config.growth_factor,
        }
    }

    /// Shrink the vector to fit the current length (at least 1 element capacity).
    ///
    /// # Limitations
    /// Shrinking truncates the backing file after remapping this handle; if
    /// another [`MmapVec`] handle or process still has the old, larger range
    /// mapped, accessing pages beyond the new file length through that other
    /// mapping will raise `SIGBUS` on Unix (see [Limitations](Self#limitations)).
    pub fn shrink_to_fit(&mut self) -> Result<()> {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        let current_len = self.len();
        let current_capacity = self.capacity();

        if current_len < current_capacity {
            let new_capacity = current_len.max(1); // At least 1 element capacity
            self.resize_to_capacity(new_capacity)?;
        }

        Ok(())
    }

    /// Extend the vector with the contents of an iterator
    pub fn extend<I>(&mut self, iter: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
    {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        // Reserve space for at least the lower bound
        if lower > 0 {
            self.reserve(lower)?;
        }

        for item in iter {
            self.push(item)?;
        }

        Ok(())
    }

    /// Truncate the vector to the specified length
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        let current_len = self.len();
        if len < current_len {
            self.store_len(len)?;

            if self.config.sync_on_write {
                self.sync_header()?;
            }
        }

        Ok(())
    }

    /// Fill the vector with a specific value up to the given length
    pub fn resize(&mut self, len: usize, value: T) -> Result<()> {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        let current_len = self.len();

        if len > current_len {
            // Need to grow
            let additional = len - current_len;
            self.reserve(additional)?;

            // Fill with the specified value
            self.mapped_slice_mut(len)?
                .get_mut(current_len..)
                .ok_or_else(|| ZiporaError::out_of_bounds(current_len, len))?
                .fill(value);

            self.store_len(len)?;
        } else if len < current_len {
            // Need to shrink
            self.truncate(len)?;
        }

        if self.config.sync_on_write {
            if len > current_len {
                self.sync_range(current_len, len - current_len)?;
            } else {
                self.sync_header()?;
            }
        }

        Ok(())
    }

    /// Size, map and initialize a freshly created backing file so that it
    /// is `file_len` bytes long.
    fn from_new_file(
        file: File,
        file_path: PathBuf,
        config: MmapVecConfig,
        file_len: usize,
    ) -> Result<Self> {
        // Set file size
        let size = u64::try_from(file_len)
            .map_err(|_| ZiporaError::invalid_data("MmapVec total file size overflow"))?;
        file.set_len(size)
            .map_err(|e| ZiporaError::io_error(format!("Failed to resize file: {}", e)))?;
        file.sync_all()
            .map_err(|e| ZiporaError::io_error(format!("Failed to sync: {}", e)))?;

        // Create memory mapping
        let mut map = Self::map_file(&file, &config, None)?;
        let (header, data, mapped_cap) = Self::view_of(&mut map, file_len)?;
        let initial_capacity = config.initial_capacity;
        let mut vec = Self {
            file_path,
            mmap: Some(FileMapping { file, map }),
            config,
            header,
            data,
            len: 0,
            cap: 0,
            mapped_cap,
            is_temp_file: false,
            _phantom: PhantomData,
        };

        // Initialize header
        vec.initialize_header(initial_capacity)?;
        if vec.config.sync_on_write {
            vec.sync_header()?;
        }
        Ok(vec)
    }

    /// Apply the configured kernel advice to a fresh mapping.
    ///
    /// `madvise` is a non-semantic kernel performance hint; return values are
    /// intentionally ignored because `MADV_HUGEPAGE` and access-pattern hints
    /// can return `EINVAL` on filesystems or kernels without file-backed
    /// transparent huge-page support, and failing `create` or `open` on
    /// advisory hint rejection would break portability across filesystems.
    fn advise_mapping(map: &MappingKind, config: &MmapVecConfig) {
        use memmap2::Advice;

        match config.access_pattern {
            AccessPattern::Sequential | AccessPattern::ReadHeavy => {
                let _ = map.advise(Advice::Sequential);
            }
            AccessPattern::Random => {
                let _ = map.advise(Advice::Random);
            }
            AccessPattern::WriteHeavy | AccessPattern::Mixed => {}
        }

        #[cfg(target_os = "linux")]
        if config.use_huge_pages {
            let _ = map.advise(Advice::HugePage);
        }
    }

    /// Map the first `len` bytes of `file`, or the whole file when `len` is
    /// `None`: shared and writable (`MmapMut`), or read-only `PROT_READ`
    /// (`Mmap`) for read-only vectors.
    fn map_file(file: &File, config: &MmapVecConfig, len: Option<usize>) -> Result<MappingKind> {
        let mut options = MmapOptions::new();
        if let Some(len) = len {
            options.len(len);
        }
        #[cfg(target_os = "linux")]
        if config.populate_pages {
            options.populate();
        }

        let map = if config.read_only {
            // SAFETY: shared read-only mapping (`PROT_READ`) — hardware page
            // protection prevents any writes through the mapping, and
            // `MAP_POPULATE` faults clean file-backed page-cache pages
            // (`RssFile`) rather than private anonymous copy-on-write pages
            // (`RssAnon`). The `Pod` bound on `T` makes any file content a
            // valid bit pattern, and the map lives as long as the
            // `FileMapping` that stores it. As required by `memmap2`
            // (documented under `MmapVec` Limitations), the backing file must
            // not be concurrently truncated below the mapped length (which
            // raises `SIGBUS` on Unix) or mutated while references into the
            // mapping are live.
            unsafe { options.map(file) }.map(MappingKind::ReadOnly)
        } else {
            // SAFETY: shared mutable mapping (`MAP_SHARED`) of the vector's own
            // file handle. The `Pod` bound on `T` makes any file content a
            // valid bit pattern. As required by `memmap2` (documented under
            // `MmapVec` Limitations), callers must not let another process or
            // handle truncate the file below the mapped length (which raises
            // `SIGBUS` on Unix) or mutate mapped bytes concurrently with live
            // references.
            unsafe { options.map_mut(file) }.map(MappingKind::ReadWrite)
        }
        .map_err(|e| ZiporaError::io_error(format!("Failed to mmap file: {}", e)))?;
        Self::advise_mapping(&map, config);
        Ok(map)
    }

    /// Calculate file size for given capacity
    fn calculate_file_size(capacity: usize) -> Result<usize> {
        let payload_bytes = capacity
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| ZiporaError::invalid_data("MmapVec capacity byte size overflow"))?;
        Self::data_offset()
            .checked_add(payload_bytes)
            .ok_or_else(|| ZiporaError::invalid_data("MmapVec total file size overflow"))
    }

    /// Byte offset of the element payload within the mapping.
    ///
    /// The 80-byte header is padded up to `align_of::<T>()` so the data
    /// pointer is properly aligned for T (creating a `&[T]` from a misaligned
    /// pointer is UB). For `align_of::<T>() <= 16` this equals HEADER_SIZE,
    /// preserving the existing file layout; larger alignments (e.g. 32-byte
    /// AVX types) were previously misaligned and unusable.
    #[inline]
    fn data_offset() -> usize {
        HEADER_SIZE.next_multiple_of(std::mem::align_of::<T>())
    }

    /// Number of whole elements a mapping of `map_len` bytes starting at
    /// address `base_addr` holds after the header and its padding
    /// (`usize::MAX` for zero-sized `T`).
    ///
    /// # Errors
    /// `InvalidData` if the mapping cannot hold the header and its padding;
    /// `Misaligned` if the element data would not be aligned for `T`.
    /// Mappings start on a page boundary, so the latter only happens for
    /// alignments above the page size.
    fn mapped_capacity_for(base_addr: usize, map_len: usize) -> Result<usize> {
        let data_offset = Self::data_offset();
        let payload = map_len.checked_sub(data_offset).ok_or_else(|| {
            ZiporaError::invalid_data("Backing file too small for MmapVec header")
        })?;
        let data_addr = base_addr
            .checked_add(data_offset)
            .ok_or_else(|| ZiporaError::invalid_data("MmapVec data address overflows usize"))?;
        let align = std::mem::align_of::<T>();
        if !data_addr.is_multiple_of(align) {
            return Err(ZiporaError::misaligned(
                data_addr,
                align,
                "MmapVec element data",
            ));
        }
        Ok(payload
            .checked_div(std::mem::size_of::<T>())
            .unwrap_or(usize::MAX))
    }

    /// Header pointer (`Some` for writable mappings, `None` for read-only),
    /// data pointer and element capacity of `map`, which must be at least
    /// `required_len` bytes long.
    ///
    /// Both pointers derive from one base pointer of `map`. They stay valid
    /// until `map` is dropped or replaced, so callers store them together
    /// with `map` and the returned capacity.
    fn view_of(
        map: &mut MappingKind,
        required_len: usize,
    ) -> Result<(Option<NonNull<MmapVecHeader>>, NonNull<T>, usize)> {
        let map_len = map.len();
        match map {
            MappingKind::ReadWrite(m) => {
                let base = m.as_mut_ptr();
                let mapped_cap = Self::mapped_capacity_for(base.addr(), map_len)?;
                if map_len < required_len {
                    return Err(ZiporaError::invalid_data(format!(
                        "MmapVec mapping is {} bytes, {} required",
                        map_len, required_len
                    )));
                }
                let header = NonNull::new(base.cast::<MmapVecHeader>())
                    .ok_or_else(|| ZiporaError::invalid_data("Invalid header pointer"))?;
                // `wrapping_add` stays in bounds: `mapped_capacity_for` checked that
                // the mapping holds at least `data_offset()` bytes.
                let data = NonNull::new(base.wrapping_add(Self::data_offset()).cast::<T>())
                    .ok_or_else(|| ZiporaError::invalid_data("Invalid data pointer"))?;
                Ok((Some(header), data, mapped_cap))
            }
            MappingKind::ReadOnly(m) => {
                let base = m.as_ptr();
                let mapped_cap = Self::mapped_capacity_for(base.addr(), map_len)?;
                if map_len < required_len {
                    return Err(ZiporaError::invalid_data(format!(
                        "MmapVec mapping is {} bytes, {} required",
                        map_len, required_len
                    )));
                }
                let data = NonNull::new(
                    base.wrapping_add(Self::data_offset())
                        .cast::<T>()
                        .cast_mut(),
                )
                .ok_or_else(|| ZiporaError::invalid_data("Invalid data pointer"))?;
                Ok((None, data, mapped_cap))
            }
        }
    }

    /// Length and capacity declared by `header`, checked against a mapping
    /// that holds `mapped_cap` elements.
    fn counts_from_header(header: &MmapVecHeader, mapped_cap: usize) -> Result<(usize, usize)> {
        header.validate::<T>()?;
        let len = usize::try_from(header.length())
            .map_err(|_| ZiporaError::invalid_data("MmapVec header length overflows usize"))?;
        let cap = usize::try_from(header.capacity())
            .map_err(|_| ZiporaError::invalid_data("MmapVec header capacity overflows usize"))?;
        if cap > mapped_cap {
            return Err(ZiporaError::invalid_data(
                "Backing file smaller than header-declared capacity",
            ));
        }
        Ok((len, cap))
    }

    /// Initialize header for new file
    fn initialize_header(&mut self, capacity: usize) -> Result<()> {
        self.update_header(|header| *header = MmapVecHeader::new::<T>())?;
        self.store_counts(0, capacity)
    }

    /// Get header pointer
    #[inline]
    fn header(&self) -> Result<NonNull<MmapVecHeader>> {
        self.header
            .ok_or_else(|| ZiporaError::invalid_data("Header not initialized"))
    }

    /// Apply `f` to the header in the mapping.
    #[inline]
    fn update_header(&mut self, f: impl FnOnce(&mut MmapVecHeader)) -> Result<()> {
        let header = self.header()?;
        // SAFETY: `header` was derived by `view_of` from the base of the
        // live writable mapping in `self.mmap` (`header` and `mmap` are only
        // ever replaced together), which holds at least `data_offset() >=
        // HEADER_SIZE` bytes. `MmapVecHeader` is a byte array: alignment 1,
        // every bit pattern valid. `&mut self` makes this the only reference
        // into the mapping while `f` runs.
        f(unsafe { &mut *header.as_ptr() });
        Ok(())
    }

    /// Adopt `len` and `cap`, writing both through to the file header.
    fn store_counts(&mut self, len: usize, cap: usize) -> Result<()> {
        if len > cap || cap > self.mapped_cap {
            return Err(ZiporaError::invalid_state(format!(
                "MmapVec length {} / capacity {} do not fit the {} mapped elements",
                len, cap, self.mapped_cap
            )));
        }
        let len64 = u64::try_from(len)
            .map_err(|_| ZiporaError::invalid_data("MmapVec length overflows u64"))?;
        let cap64 = u64::try_from(cap)
            .map_err(|_| ZiporaError::invalid_data("MmapVec capacity overflows u64"))?;
        self.update_header(|header| {
            header.set_length(len64);
            header.set_capacity(cap64);
        })?;
        self.len = len;
        self.cap = cap;
        Ok(())
    }

    /// Adopt `len` (at most the capacity), writing it through to the header.
    #[inline]
    fn store_len(&mut self, len: usize) -> Result<()> {
        if len > self.cap {
            return Err(ZiporaError::invalid_state(format!(
                "MmapVec length {} / capacity {} do not fit the {} mapped elements",
                len, self.cap, self.mapped_cap
            )));
        }
        let len64 = u64::try_from(len)
            .map_err(|_| ZiporaError::invalid_data("MmapVec length overflows u64"))?;
        self.update_header(|header| {
            header.set_length(len64);
        })?;
        self.len = len;
        Ok(())
    }

    /// The data pointer, checked for a view of `count` elements within the
    /// elements the current mapping holds (`self.data` alignment is verified
    /// once by `view_of` when the mapping is created or remapped).
    #[inline]
    fn checked_data(&self, count: usize) -> Result<NonNull<T>> {
        if count > self.mapped_cap {
            return Err(ZiporaError::out_of_bounds(count, self.mapped_cap));
        }
        Ok(self.data)
    }

    /// The first `count` elements of the mapped data region.
    #[inline]
    fn mapped_slice(&self, count: usize) -> Result<&[T]> {
        let data = self.checked_data(count)?;
        // SAFETY: `view_of` verified that `data` is aligned for `T`, and
        // `checked_data` verified that `count <= mapped_cap`, the number of
        // whole elements `view_of` found between `data` and the end of the
        // live mapping in `self.mmap` (`data`, `mapped_cap` and `mmap` are
        // only ever replaced together), so the range is in bounds and at most
        // `isize::MAX` bytes. `T: Pod` makes any mapped bytes a valid `T`.
        // The slice borrows `self`, and every remap or unmap takes
        // `&mut self`, so the mapping outlives the slice.
        Ok(unsafe { slice::from_raw_parts(data.as_ptr(), count) })
    }

    /// The first `count` elements of the mapped data region, mutably.
    #[inline]
    fn mapped_slice_mut(&mut self, count: usize) -> Result<&mut [T]> {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }
        let data = self.checked_data(count)?;
        // SAFETY: as in `mapped_slice`; in addition, `!self.config.read_only`
        // guarantees a writable `MappingKind::ReadWrite` mapping and
        // `&mut self` makes this the only reference into the mapping while
        // the slice lives.
        Ok(unsafe { slice::from_raw_parts_mut(data.as_ptr(), count) })
    }

    /// Compute `current_capacity * growth_factor` with configuration validation
    /// and overflow checking.
    fn scaled_capacity(&self, current_capacity: usize) -> Result<usize> {
        self.config.validate()?;
        let scaled = (current_capacity as f64) * self.config.growth_factor;
        if !scaled.is_finite() || scaled >= (usize::MAX as f64) {
            return Err(ZiporaError::invalid_data(
                "MmapVec capacity overflows usize",
            ));
        }
        Ok(scaled as usize)
    }

    /// Grow the vector capacity
    fn grow(&mut self) -> Result<()> {
        let current_capacity = self.cap;
        let min_capacity = current_capacity
            .checked_add(1)
            .ok_or_else(|| ZiporaError::invalid_data("MmapVec capacity overflows usize"))?;
        let grown = self.scaled_capacity(current_capacity)?;
        let new_capacity = std::cmp::max(grown, min_capacity);
        self.resize_to_capacity(new_capacity)
    }

    /// Resize vector to specific capacity
    ///
    /// The mapping is shared with the file, so existing data needs no
    /// copy-out/copy-in. The file is resized and remapped through the
    /// vector's own handle, never re-opened by path. The new mapping must
    /// cover `new_capacity` elements with aligned data before it replaces
    /// the old one, so if resizing the file or mapping it fails the vector
    /// is unchanged. Both mappings exist for a moment, which needs address
    /// space for the old and the new size.
    ///
    /// Growth never shrinks the file: a file that is already large enough
    /// (for example, grown through another handle) is only remapped.
    /// Shrinking maps the shorter range first and truncates the file after
    /// the old mapping is gone; if that truncation fails the vector already
    /// uses the smaller capacity and the error is returned.
    fn resize_to_capacity(&mut self, new_capacity: usize) -> Result<()> {
        if new_capacity < self.len {
            return Err(ZiporaError::invalid_parameter(
                "MmapVec capacity below current length",
            ));
        }
        if new_capacity == self.cap {
            return Ok(());
        }
        let new_file_size = Self::calculate_file_size(new_capacity)?;
        let new_size = u64::try_from(new_file_size)
            .map_err(|_| ZiporaError::invalid_data("MmapVec total file size overflow"))?;
        let shrinking = new_capacity < self.cap;
        let mmap = self
            .mmap
            .as_mut()
            .ok_or_else(|| ZiporaError::invalid_data("No memory mapping"))?;

        let mut map = if shrinking {
            Self::map_file(&mmap.file, &self.config, Some(new_file_size))?
        } else {
            let file_size = mmap
                .file
                .metadata()
                .map_err(|e| ZiporaError::io_error(format!("Failed to stat file: {}", e)))?
                .len();
            if file_size < new_size {
                mmap.file
                    .set_len(new_size)
                    .map_err(|e| ZiporaError::io_error(format!("Failed to resize file: {}", e)))?;
            }
            Self::map_file(&mmap.file, &self.config, None)?
        };
        let (header, data, mapped_cap) = Self::view_of(&mut map, new_file_size)?;

        // The new mapping is valid: only now is the old one unmapped.
        mmap.map = map;
        self.header = header;
        self.data = data;
        self.mapped_cap = mapped_cap;

        // Update capacity in header
        self.store_counts(self.len, new_capacity)?;

        if shrinking && let Some(mmap) = &self.mmap {
            mmap.file
                .set_len(new_size)
                .map_err(|e| ZiporaError::io_error(format!("Failed to resize file: {}", e)))?;
        }
        if self.config.sync_on_write {
            self.sync_header()?;
        }
        Ok(())
    }

    //==========================================================================
    // SIMD-OPTIMIZED BULK OPERATIONS
    //==========================================================================

    /// Create a temporary vector in `dir` by trying candidate file names from
    /// `next_name` with `create_new(true)` (`O_EXCL`), retrying on
    /// `ErrorKind::AlreadyExists` (up to 256 attempts) and removing the file
    /// if initialization fails.
    fn create_temp_in(
        capacity: usize,
        dir: &Path,
        mut next_name: impl FnMut() -> String,
    ) -> Result<Self> {
        const MAX_ATTEMPTS: usize = 256;
        let file_len = Self::calculate_file_size(capacity)?;
        let config = MmapVecConfig::default().with_initial_capacity(capacity);

        let mut last_exists = None;
        let mut opened = None;
        for _ in 0..MAX_ATTEMPTS {
            let file_path = dir.join(next_name());
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&file_path)
            {
                Ok(file) => {
                    opened = Some((file, file_path));
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    last_exists = Some(e);
                }
                Err(e) => {
                    return Err(ZiporaError::io_error(format!(
                        "Failed to create temporary file: {}",
                        e
                    )));
                }
            }
        }
        let (file, file_path) = opened.ok_or_else(|| {
            ZiporaError::io_error(format!(
                "Failed to create unique temporary file after {} attempts: {}",
                MAX_ATTEMPTS,
                last_exists
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "path already exists".to_string())
            ))
        })?;

        match Self::from_new_file(file, file_path.clone(), config, file_len) {
            Ok(mut vec) => {
                vec.is_temp_file = true;
                Ok(vec)
            }
            Err(e) => {
                let _ = std::fs::remove_file(&file_path);
                Err(e)
            }
        }
    }

    /// Create a vector with room for `capacity` elements, backed by a new
    /// temporary file in `std::env::temp_dir()` that is deleted on drop.
    ///
    /// The backing file is created at its full size up front, and bytes
    /// added by extending a file read back as zeros, so no separate
    /// zero-fill pass (SIMD or otherwise) runs; the name is kept for API
    /// compatibility.
    ///
    /// # Errors
    /// Returns an error if `capacity` elements plus the header do not fit
    /// in `usize` bytes, or if the file cannot be created, sized or mapped.
    ///
    /// # Limitations
    /// The temporary file in `std::env::temp_dir()` is opened with
    /// `create_new(true)` (`O_EXCL`) and unlinked when `Self` is dropped, not
    /// immediately after mapping; another process with write access to the
    /// temporary directory that truncates or mutates the file while it is
    /// mapped is subject to the same OS `mmap` constraints documented under
    /// [Limitations](Self#limitations).
    pub fn with_capacity_simd(capacity: usize) -> Result<Self>
    where
        T: Copy + 'static,
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let temp_dir = std::env::temp_dir();
        let pid = std::process::id();
        let thread_id = std::thread::current().id();
        Self::create_temp_in(capacity, &temp_dir, || {
            let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
            let rand = fastrand::u64(..);
            format!("mmap_vec_simd_{pid}_{thread_id:?}_{counter}_{rand:016x}.dat")
        })
    }

    /// Push multiple elements with SIMD optimization
    ///
    /// # Performance
    /// 4-8x faster than pushing elements individually for bulk operations
    ///
    /// # Errors
    /// Returns an error, leaving the vector unchanged, if the vector is
    /// read-only, if the new length would overflow `usize`, or if growing
    /// the file or its mapping fails.
    pub fn push_bulk_simd(&mut self, items: &[T]) -> Result<()>
    where
        T: Copy,
    {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        if items.is_empty() {
            return Ok(());
        }

        let start = std::time::Instant::now();

        // Ensure capacity
        let old_len = self.len;
        let new_len = old_len.checked_add(items.len()).ok_or_else(|| {
            ZiporaError::invalid_parameter("MmapVec push_bulk_simd: length overflows usize")
        })?;
        if new_len > self.cap {
            self.reserve(items.len())?;
        }

        let dst = self
            .mapped_slice_mut(new_len)?
            .get_mut(old_len..)
            .ok_or_else(|| ZiporaError::out_of_bounds(old_len, new_len))?;
        let size_bytes = std::mem::size_of_val(items);

        if size_bytes >= 64 {
            // SIMD threshold
            // Cache-optimized SIMD copy
            fast_copy_cache_optimized(pod_bytes(items)?, pod_bytes_mut(dst)?)?;
        } else {
            // Small copy: use standard approach
            dst.copy_from_slice(items);
        }

        // Update length
        self.store_len(new_len)?;

        // Monitor performance for adaptive SIMD selection
        Self::monitor_simd_perf(Operation::Copy, start.elapsed(), items.len());

        if self.config.sync_on_write {
            self.sync_range(old_len, items.len())?;
        }

        Ok(())
    }

    /// Pop the last `count` elements, returned in their original order.
    ///
    /// The elements are copied out of the mapping with one slice copy.
    ///
    /// # Errors
    /// Returns an error if the vector is read-only or `count > len()`.
    pub fn pop_bulk_simd(&mut self, count: usize) -> Result<Vec<T>>
    where
        T: Copy,
    {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        if count == 0 {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();
        let old_len = self.len;
        let new_len = old_len
            .checked_sub(count)
            .ok_or_else(|| ZiporaError::invalid_data("Count exceeds length"))?;
        let result = self
            .mapped_slice(old_len)?
            .get(new_len..)
            .map(<[T]>::to_vec)
            .ok_or_else(|| ZiporaError::out_of_bounds(new_len, old_len))?;
        self.store_len(new_len)?;

        // Monitor performance
        Self::monitor_simd_perf(Operation::Copy, start.elapsed(), count);

        if self.config.sync_on_write {
            self.sync_header()?;
        }

        Ok(result)
    }

    /// Copy from another MmapVec with SIMD optimization
    ///
    /// # Performance
    /// 3-5x faster than element-by-element copy with prefetching for large vectors
    pub fn copy_from_simd(&mut self, other: &Self) -> Result<()>
    where
        T: Copy,
    {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        if other.is_empty() {
            self.clear()?;
            return Ok(());
        }

        let start = std::time::Instant::now();
        let src = other.mapped_slice(other.len)?;
        let count = src.len();

        // Resize to match source
        if self.capacity() < count {
            // Need to grow capacity - use reserve instead of creating new vector
            self.reserve(count.saturating_sub(self.len()))?;
        }

        // Now copy the data
        {
            // Have enough capacity, copy directly
            let size_bytes = std::mem::size_of_val(src);

            // Prefetch source data for large copies
            if size_bytes >= 4096 {
                fast_prefetch_range(pod_bytes(src)?);
            }

            let dst = self.mapped_slice_mut(count)?;
            // Use cache-optimized SIMD copy
            if size_bytes >= 64 {
                fast_copy_cache_optimized(pod_bytes(src)?, pod_bytes_mut(dst)?)?;
            } else {
                dst.copy_from_slice(src);
            }

            self.store_len(count)?;
        }

        // Monitor performance
        Self::monitor_simd_perf(Operation::Copy, start.elapsed(), count);

        if self.config.sync_on_write {
            self.sync_range(0, self.len())?;
        }

        Ok(())
    }

    /// Fill a range with a value using SIMD
    ///
    /// # Performance
    /// 4-6x faster for byte types with SIMD vectorization
    pub fn fill_range_simd(&mut self, range: std::ops::Range<usize>, value: T) -> Result<()>
    where
        T: Copy,
    {
        if self.config.read_only {
            return Err(ZiporaError::invalid_data("Vector is read-only"));
        }

        if range.end > self.len() {
            return Err(ZiporaError::invalid_data("Range exceeds length"));
        }

        if range.is_empty() {
            return Ok(());
        }

        let range_start = range.start;
        let count = range.end - range.start;
        let len = self.len;
        let slice = self
            .mapped_slice_mut(len)?
            .get_mut(range)
            .ok_or_else(|| ZiporaError::invalid_data("Range exceeds length"))?;

        match *bytemuck::bytes_of(&value) {
            // For byte-sized types, use direct SIMD fill
            [byte] if count >= 64 => fast_fill(pod_bytes_mut(slice)?, byte),
            // Standard fill for complex types
            _ => slice.fill(value),
        }

        if self.config.sync_on_write {
            self.sync_range(range_start, count)?;
        }

        Ok(())
    }

    /// Compare `self[range]` with `other[..range.len()]` using `T::eq`.
    ///
    /// Uses standard slice equality (`PartialEq`), which specializes to
    /// vectorized byte comparison for integer types while respecting `T::eq`
    /// for floats (`-0.0 == 0.0`, `NaN != NaN`) and custom `PartialEq`
    /// implementations regardless of range length.
    pub fn compare_range_simd(&self, range: std::ops::Range<usize>, other: &Self) -> Result<bool>
    where
        T: PartialEq + Copy,
    {
        if range.end > self.len() || range.len() > other.len() {
            return Err(ZiporaError::invalid_data("Invalid range"));
        }

        if range.is_empty() {
            return Ok(true);
        }

        let count = range.end - range.start;
        let mine = self
            .mapped_slice(self.len)?
            .get(range)
            .ok_or_else(|| ZiporaError::invalid_data("Invalid range"))?;
        let theirs = other
            .mapped_slice(other.len)?
            .get(..count)
            .ok_or_else(|| ZiporaError::invalid_data("Invalid range"))?;

        // Byte equality is not value equality (`-0.0 == 0.0`, `NaN != NaN`,
        // custom `PartialEq`), and standard slice equality already
        // specializes to `memcmp` for integer types where that is sound.
        Ok(mine == theirs)
    }

    /// Internal: Monitor SIMD operation performance
    #[inline]
    fn monitor_simd_perf(operation: Operation, elapsed: std::time::Duration, size: usize) {
        let selector = AdaptiveSimdSelector::global();
        selector.monitor_performance(operation, elapsed, size as u64);
    }
}

impl<T> Drop for MmapVec<T> {
    fn drop(&mut self) {
        // Drop the mmap first to ensure munmap happens before file deletion
        self.mmap.take();

        // Clean up temporary files
        if self.is_temp_file && self.file_path.exists() {
            // Best effort deletion - ignore errors
            let _ = std::fs::remove_file(&self.file_path);
        }
    }
}

/// Iterator for memory-mapped vectors
pub struct MmapVecIter<'a, T> {
    slice: &'a [T],
    index: usize,
}

impl<'a, T> Iterator for MmapVecIter<'a, T>
where
    T: Pod,
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
    T: Pod,
{
    fn len(&self) -> usize {
        self.slice.len() - self.index
    }
}

impl<'a, T> IntoIterator for &'a MmapVec<T>
where
    T: Pod + 'static,
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
    use tempfile::tempdir;

    /// Regression (codereview.md 2026-08 #5): MmapVec was not actually
    /// file-backed — it copied the file into an anonymous mapping and only
    /// wrote it back on an explicit sync(), so every unsynced modification
    /// was silently lost, contradicting the documented persistence
    /// guarantee. With a real shared file mapping, writes must survive
    /// drop + reopen without any sync() call.
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_data_persists_without_explicit_sync() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("persist_no_sync.mmap");

        {
            let mut vec = MmapVec::<u64>::create(&file_path, MmapVecConfig::default()).unwrap();
            vec.push(11).unwrap();
            vec.push(22).unwrap();
            vec.push(33).unwrap();
            // Intentionally no sync() before drop.
        }

        let vec = MmapVec::<u64>::open(&file_path, MmapVecConfig::default()).unwrap();
        assert_eq!(vec.as_slice(), &[11, 22, 33]);
    }

    /// Same regression, growth path: capacity growth remaps the file; data
    /// written before and after growth must persist without sync().
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_growth_persists_without_explicit_sync() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("persist_growth.mmap");

        {
            let config = MmapVecConfig::builder().with_initial_capacity(4).build();
            let mut vec = MmapVec::<u32>::create(&file_path, config).unwrap();
            for i in 0..100u32 {
                vec.push(i).unwrap();
            }
            // Intentionally no sync() before drop.
        }

        let vec = MmapVec::<u32>::open(&file_path, MmapVecConfig::default()).unwrap();
        assert_eq!(vec.len(), 100);
        for i in 0..100u32 {
            assert_eq!(vec.get(i as usize), Some(&i));
        }
    }

    /// The remaining config knobs must all be honored (madvise is a hint,
    /// so this verifies the advised mappings still behave correctly; the
    /// unused knobs — cache alignment, NUMA, prefetch distance — were
    /// removed rather than left as silent no-ops).
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_access_pattern_and_huge_page_configs_work() {
        for pattern in [
            AccessPattern::Sequential,
            AccessPattern::Random,
            AccessPattern::ReadHeavy,
            AccessPattern::WriteHeavy,
            AccessPattern::Mixed,
        ] {
            let temp_dir = tempdir().unwrap();
            let file_path = temp_dir.path().join("advised.mmap");
            let config = MmapVecConfig {
                access_pattern: pattern,
                use_huge_pages: true,
                ..Default::default()
            };
            let mut vec = MmapVec::<u64>::create(&file_path, config).unwrap();
            for i in 0..100 {
                vec.push(i).unwrap();
            }
            assert_eq!(vec.len(), 100);
            assert_eq!(vec.get(99), Some(&99));
        }
    }

    /// sync_on_write now flushes only the touched element range + header
    /// (was: full-mapping msync per write). Exercise every write path with
    /// the flag on — a wrong flush_range offset/len would return an error —
    /// and verify the result persists across reopen.
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_sync_on_write_range_flushes_cover_all_write_paths() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("sync_range.mmap");
        let config = MmapVecConfig {
            sync_on_write: true,
            initial_capacity: 4,
            ..Default::default()
        };

        {
            let mut vec = MmapVec::<u32>::create(&file_path, config).unwrap();
            for i in 0..10 {
                vec.push(i).unwrap(); // push + grow paths
            }
            vec.push_bulk_simd(&[100, 101, 102]).unwrap();
            vec.fill_range_simd(0..2, 7).unwrap();
            vec.resize(20, 9).unwrap();
            assert_eq!(vec.pop(), Some(9));
            vec.truncate(15).unwrap();
        }

        let vec = MmapVec::<u32>::open(&file_path, MmapVecConfig::default()).unwrap();
        assert_eq!(vec.len(), 15);
        assert_eq!(vec.get(0), Some(&7));
        assert_eq!(vec.get(1), Some(&7));
        assert_eq!(vec.get(10), Some(&100));
        assert_eq!(vec.get(14), Some(&9));
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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

    /// Regression: the payload used to start at the fixed 80-byte header
    /// offset, misaligning the data pointer for types with align > 16
    /// (80 % 32 == 16) — UB when creating &[T]. The offset is now padded
    /// to align_of::<T>(), including across a persist/reopen cycle.
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_alignment_over_16() {
        #[repr(C, align(32))]
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Avx32 {
            data: [u8; 32],
        }
        // SAFETY: 32 bytes of u8 with align(32) — size == field size, no
        // padding, any bit pattern valid. (derive(Pod) rejects repr(align).)
        unsafe impl bytemuck::Zeroable for Avx32 {}
        unsafe impl bytemuck::Pod for Avx32 {}

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_align32.mmap");

        {
            let config = MmapVecConfig::persistent_cache();
            let mut vec = MmapVec::<Avx32>::create(&file_path, config).unwrap();

            for i in 0..10u8 {
                vec.push(Avx32 { data: [i; 32] }).unwrap();
            }

            for i in 0..10u8 {
                let elem = vec.get(i as usize).unwrap();
                assert_eq!(elem.data, [i; 32]);
                assert!(
                    (elem as *const Avx32 as usize).is_multiple_of(std::mem::align_of::<Avx32>()),
                    "element {} misaligned",
                    i
                );
            }
            vec.sync().unwrap();
        }

        // Reopen: data offset must be computed identically on load
        {
            let config = MmapVecConfig::default();
            let vec = MmapVec::<Avx32>::open(&file_path, config).unwrap();
            assert_eq!(vec.len(), 10);
            for i in 0..10u8 {
                let elem = vec.get(i as usize).unwrap();
                assert_eq!(elem.data, [i; 32]);
                assert!((elem as *const Avx32 as usize).is_multiple_of(32));
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_builder_pattern() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("builder.mmap");

        let config = MmapVecConfig::builder()
            .with_initial_capacity(5000)
            .with_growth_factor(1.5)
            .with_populate_pages(true)
            .with_sync_on_write(false)
            .build();

        let vec = MmapVec::<i32>::create(&file_path, config).unwrap();
        assert_eq!(vec.capacity(), 5000);
        assert!(!vec.config.sync_on_write);
        assert_eq!(vec.config.growth_factor, 1.5);
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_stats() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("stats.mmap");

        let config = MmapVecConfig {
            initial_capacity: 100,
            ..MmapVecConfig::default()
        };
        let mut vec = MmapVec::<u64>::create(&file_path, config).unwrap();

        // Add some data
        for i in 0..50 {
            vec.push(i).unwrap();
        }

        let stats = vec.stats();
        assert_eq!(stats.len, 50);
        assert_eq!(stats.capacity, 100);
        assert_eq!(stats.element_size, 8); // u64 = 8 bytes
        assert_eq!(stats.utilization, 0.5); // 50/100
        assert_eq!(stats.memory_efficiency(), 50.0);
        assert_eq!(stats.wasted_space(), 50 * 8);
        assert!(stats.needs_compaction(0.7));
        assert!(!stats.needs_compaction(0.4));
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_extend() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("extend.mmap");

        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<u16>::create(&file_path, config).unwrap();

        // Extend with a range
        vec.extend(1..=10).unwrap();
        assert_eq!(vec.len(), 10);
        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(9), Some(&10));

        // Extend with an iterator
        let more_data = vec![100, 200, 300];
        vec.extend(more_data).unwrap();
        assert_eq!(vec.len(), 13);
        assert_eq!(vec.get(10), Some(&100));
        assert_eq!(vec.get(12), Some(&300));
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_truncate() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("truncate.mmap");

        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<i16>::create(&file_path, config).unwrap();

        // Fill with data
        for i in 0..20 {
            vec.push(i).unwrap();
        }
        assert_eq!(vec.len(), 20);

        // Truncate
        vec.truncate(10).unwrap();
        assert_eq!(vec.len(), 10);
        assert_eq!(vec.get(9), Some(&9));
        assert_eq!(vec.get(10), None);

        // Truncate to larger size should not change anything
        vec.truncate(15).unwrap();
        assert_eq!(vec.len(), 10);
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_resize_with_value() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("resize_val.mmap");

        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<i8>::create(&file_path, config).unwrap();

        // Start with some data
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        assert_eq!(vec.len(), 2);

        // Resize larger with specific value
        vec.resize(5, 99).unwrap();
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(1), Some(&2));
        assert_eq!(vec.get(2), Some(&99));
        assert_eq!(vec.get(3), Some(&99));
        assert_eq!(vec.get(4), Some(&99));

        // Resize smaller
        vec.resize(3, 0).unwrap();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(2), Some(&99));
        assert_eq!(vec.get(3), None);
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_shrink_to_fit() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("shrink.mmap");

        let config = MmapVecConfig {
            initial_capacity: 1000,
            ..MmapVecConfig::default()
        };
        let mut vec = MmapVec::<u32>::create(&file_path, config).unwrap();

        assert_eq!(vec.capacity(), 1000);

        // Add only a few elements
        for i in 0..10 {
            vec.push(i).unwrap();
        }
        assert_eq!(vec.len(), 10);
        assert_eq!(vec.capacity(), 1000);

        // Shrink to fit
        vec.shrink_to_fit().unwrap();
        assert_eq!(vec.len(), 10);
        assert!(vec.capacity() <= 10);

        // Verify data is still intact
        for i in 0..10 {
            assert_eq!(vec.get(i), Some(&(i as u32)));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_new_config_presets() {
        let temp_dir = tempdir().unwrap();

        // Performance optimized
        let file_path = temp_dir.path().join("perf.mmap");
        let config = MmapVecConfig::performance_optimized();
        let vec = MmapVec::<u64>::create(&file_path, config).unwrap();
        assert_eq!(vec.capacity(), 8192);
        assert_eq!(vec.config.growth_factor, 1.618);
        assert!(vec.config.populate_pages);

        // Memory optimized
        let file_path2 = temp_dir.path().join("mem.mmap");
        let config = MmapVecConfig::memory_optimized();
        let vec2 = MmapVec::<u64>::create(&file_path2, config).unwrap();
        assert_eq!(vec2.capacity(), 256);
        assert_eq!(vec2.config.growth_factor, 1.4);
        assert!(!vec2.config.populate_pages);

        // Real-time
        let file_path3 = temp_dir.path().join("rt.mmap");
        let config = MmapVecConfig::realtime();
        let vec3 = MmapVec::<u64>::create(&file_path3, config).unwrap();
        assert_eq!(vec3.capacity(), 1024);
        assert_eq!(vec3.config.growth_factor, 1.5);
        assert!(vec3.config.populate_pages);
        assert!(!vec3.config.sync_on_write);
    }

    //==========================================================================
    // SIMD-OPTIMIZED OPERATIONS TESTS
    //==========================================================================

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_bulk_initialization() {
        let vec: MmapVec<u64> = MmapVec::with_capacity_simd(1000).unwrap();
        assert!(vec.capacity() >= 1000);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_bulk_push() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("simd_push.mmap");

        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<u32>::create(&file_path, config).unwrap();
        let items: Vec<u32> = (0..1000).collect();

        vec.push_bulk_simd(&items).unwrap();

        assert_eq!(vec.len(), 1000);
        assert_eq!(vec.as_slice(), &items[..]);
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_bulk_pop() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("simd_pop.mmap");

        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<u32>::create(&file_path, config).unwrap();

        // Add test data
        let items: Vec<u32> = (0..1000).collect();
        vec.push_bulk_simd(&items).unwrap();

        let popped = vec.pop_bulk_simd(500).unwrap();

        assert_eq!(popped.len(), 500);
        assert_eq!(vec.len(), 500);
        assert_eq!(&popped[..], &items[500..1000]);
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_copy_from() {
        let temp_dir = tempdir().unwrap();
        let file_path1 = temp_dir.path().join("simd_src.mmap");
        let file_path2 = temp_dir.path().join("simd_dst.mmap");

        let config = MmapVecConfig::default();
        let mut source = MmapVec::<u64>::create(&file_path1, config.clone()).unwrap();
        let items: Vec<u64> = (0..1000).collect();
        source.push_bulk_simd(&items).unwrap();

        let mut dest = MmapVec::<u64>::create(&file_path2, config).unwrap();
        dest.copy_from_simd(&source).unwrap();

        assert_eq!(dest.len(), source.len());
        assert_eq!(dest.as_slice(), source.as_slice());
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_fill_range() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("simd_fill.mmap");

        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<u8>::create(&file_path, config).unwrap();

        // Initialize with zeros
        let zeros = vec![0u8; 1000];
        vec.push_bulk_simd(&zeros).unwrap();

        vec.fill_range_simd(100..500, 0xFF).unwrap();

        assert!(vec.as_slice()[100..500].iter().all(|&b| b == 0xFF));
        assert!(vec.as_slice()[0..100].iter().all(|&b| b == 0));
        assert!(vec.as_slice()[500..1000].iter().all(|&b| b == 0));
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_compare_range() {
        let temp_dir = tempdir().unwrap();
        let file_path1 = temp_dir.path().join("simd_cmp1.mmap");
        let file_path2 = temp_dir.path().join("simd_cmp2.mmap");
        let file_path3 = temp_dir.path().join("simd_cmp3.mmap");

        let config = MmapVecConfig::default();

        let mut vec1 = MmapVec::<u32>::create(&file_path1, config.clone()).unwrap();
        let items1: Vec<u32> = (0..1000).collect();
        vec1.push_bulk_simd(&items1).unwrap();

        let mut vec2 = MmapVec::<u32>::create(&file_path2, config.clone()).unwrap();
        let items2: Vec<u32> = (0..1000).collect();
        vec2.push_bulk_simd(&items2).unwrap();

        let mut vec3 = MmapVec::<u32>::create(&file_path3, config).unwrap();
        let items3: Vec<u32> = (1..1001).collect();
        vec3.push_bulk_simd(&items3).unwrap();

        assert!(vec1.compare_range_simd(0..1000, &vec2).unwrap());
        assert!(!vec1.compare_range_simd(0..1000, &vec3).unwrap());
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_performance_different_sizes() {
        let temp_dir = tempdir().unwrap();

        // Test SIMD paths with various sizes
        for size in &[10, 64, 256, 1024, 4096] {
            let file_path = temp_dir.path().join(format!("simd_perf_{}.mmap", size));
            let config = MmapVecConfig::default();
            let mut vec = MmapVec::<u64>::create(&file_path, config).unwrap();
            let items: Vec<u64> = (0..*size as u64).collect();

            vec.push_bulk_simd(&items).unwrap();
            assert_eq!(vec.len(), *size);

            let popped = vec.pop_bulk_simd(*size / 2).unwrap();
            assert_eq!(popped.len(), *size / 2);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_edge_cases() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("simd_edge.mmap");

        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<u32>::create(&file_path, config).unwrap();

        // Empty push
        vec.push_bulk_simd(&[]).unwrap();
        assert_eq!(vec.len(), 0);

        // Single element
        vec.push_bulk_simd(&[42]).unwrap();
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.as_slice()[0], 42);

        // Pop more than available (should error)
        assert!(vec.pop_bulk_simd(10).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_small_operations() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("simd_small.mmap");

        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<u8>::create(&file_path, config).unwrap();

        // Test operations below SIMD threshold (should still work correctly)
        let small_data = vec![1u8, 2, 3, 4, 5];
        vec.push_bulk_simd(&small_data).unwrap();
        assert_eq!(vec.as_slice(), &small_data[..]);

        let popped = vec.pop_bulk_simd(2).unwrap();
        assert_eq!(popped, vec![4, 5]);
        assert_eq!(vec.len(), 3);
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_read_only_error() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("simd_ro.mmap");

        // Create and populate
        {
            let config = MmapVecConfig::default();
            let mut vec = MmapVec::<u32>::create(&file_path, config).unwrap();
            vec.push_bulk_simd(&[1, 2, 3]).unwrap();
            vec.sync().unwrap();
        }

        // Open as read-only
        {
            let config = MmapVecConfig::read_only();
            let mut vec = MmapVec::<u32>::open(&file_path, config).unwrap();

            // Should fail to modify
            assert!(vec.push_bulk_simd(&[4, 5]).is_err());
            assert!(vec.pop_bulk_simd(1).is_err());
            assert!(vec.fill_range_simd(0..2, 99).is_err());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_large_bulk_operations() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("simd_large.mmap");

        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<u64>::create(&file_path, config).unwrap();

        // Large bulk push (should use SIMD path)
        let large_data: Vec<u64> = (0..10000).collect();
        vec.push_bulk_simd(&large_data).unwrap();
        assert_eq!(vec.len(), 10000);

        // Verify data integrity
        for (i, &val) in vec.as_slice().iter().enumerate() {
            assert_eq!(val, i as u64);
        }

        // Large bulk pop
        let popped = vec.pop_bulk_simd(5000).unwrap();
        assert_eq!(popped.len(), 5000);
        assert_eq!(vec.len(), 5000);
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_fill_range_edge_cases() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("simd_fill_edge.mmap");

        let config = MmapVecConfig::default();
        let mut vec = MmapVec::<u8>::create(&file_path, config).unwrap();

        let data = vec![0u8; 100];
        vec.push_bulk_simd(&data).unwrap();

        // Empty range
        vec.fill_range_simd(50..50, 0xFF).unwrap();
        assert_eq!(vec.as_slice()[49], 0);
        assert_eq!(vec.as_slice()[50], 0);

        // Range exceeds length (should error)
        assert!(vec.fill_range_simd(50..200, 0xFF).is_err());

        // Full range
        vec.fill_range_simd(0..100, 0xAA).unwrap();
        assert!(vec.as_slice().iter().all(|&b| b == 0xAA));
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_compare_range_edge_cases() {
        let temp_dir = tempdir().unwrap();
        let file_path1 = temp_dir.path().join("simd_cmp_e1.mmap");
        let file_path2 = temp_dir.path().join("simd_cmp_e2.mmap");

        let config = MmapVecConfig::default();

        let mut vec1 = MmapVec::<u32>::create(&file_path1, config.clone()).unwrap();
        vec1.push_bulk_simd(&[1, 2, 3, 4, 5]).unwrap();

        let mut vec2 = MmapVec::<u32>::create(&file_path2, config).unwrap();
        vec2.push_bulk_simd(&[1, 2, 3]).unwrap();

        // Empty range
        assert!(vec1.compare_range_simd(0..0, &vec2).unwrap());

        // Range exceeds length (should error)
        assert!(vec1.compare_range_simd(0..10, &vec2).is_err());

        // Valid comparison
        assert!(vec1.compare_range_simd(0..3, &vec2).unwrap());
        assert!(!vec1.compare_range_simd(2..5, &vec2).unwrap());
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_copy_from_empty() {
        let temp_dir = tempdir().unwrap();
        let file_path1 = temp_dir.path().join("simd_copy_empty1.mmap");
        let file_path2 = temp_dir.path().join("simd_copy_empty2.mmap");

        let config = MmapVecConfig::default();

        let source = MmapVec::<u64>::create(&file_path1, config.clone()).unwrap();
        let mut dest = MmapVec::<u64>::create(&file_path2, config).unwrap();

        dest.copy_from_simd(&source).unwrap();
        assert_eq!(dest.len(), 0);
        assert!(dest.is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_simd_copy_from_capacity_growth() {
        let temp_dir = tempdir().unwrap();
        let file_path1 = temp_dir.path().join("simd_copy_grow1.mmap");
        let file_path2 = temp_dir.path().join("simd_copy_grow2.mmap");

        let config = MmapVecConfig {
            initial_capacity: 10,
            ..MmapVecConfig::default()
        };

        let mut source = MmapVec::<u64>::create(&file_path1, config.clone()).unwrap();
        let large_data: Vec<u64> = (0..1000).collect();
        source.push_bulk_simd(&large_data).unwrap();

        // Destination with smaller capacity
        let mut dest = MmapVec::<u64>::create(&file_path2, config).unwrap();
        assert!(dest.capacity() < source.len());

        // Should grow to accommodate source
        dest.copy_from_simd(&source).unwrap();
        assert_eq!(dest.len(), source.len());
        assert_eq!(dest.as_slice(), source.as_slice());
    }

    /// C9.1 F1: `with_capacity_simd` ignored its argument. It always
    /// created the default 1024-element file and then zero-filled
    /// `capacity` elements through the data pointer, so any request above
    /// 1024 elements wrote past the end of the mapping. The requested
    /// capacity must size the (zero-filled) file.
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f1_with_capacity_simd_honours_capacity() {
        // Small request first: in bounds even on the unfixed code, which
        // reports the default capacity of 1024 here.
        let small = MmapVec::<u64>::with_capacity_simd(10).unwrap();
        assert_eq!(small.capacity(), 10, "requested capacity ignored");
        drop(small);

        let big = MmapVec::<u64>::with_capacity_simd(5000).unwrap();
        assert_eq!(big.capacity(), 5000);
        assert!(big.is_empty());
        let bytes = std::fs::read(big.path()).unwrap();
        assert_eq!(bytes.len(), HEADER_SIZE + 5000 * 8);
        assert!(bytes[HEADER_SIZE..].iter().all(|&b| b == 0));
    }

    /// C9.1 F1: a capacity whose byte size overflows `usize` must be an
    /// `Err`. The unfixed code multiplied without a check; a debug build
    /// panics there, which this test catches and reports.
    #[test]
    fn test_c9_f1_with_capacity_simd_rejects_overflowing_capacity() {
        let result = std::panic::catch_unwind(|| {
            MmapVec::<u64>::with_capacity_simd(usize::MAX).map(|v| v.capacity())
        });
        assert!(
            matches!(result, Ok(Err(_))),
            "expected Err, got {:?}",
            result
        );
    }

    /// Element type aligned beyond the 4 KiB page size: its data offset is
    /// 8192, so its data is misaligned whenever the kernel places a mapping
    /// on an odd 4 KiB page.
    #[repr(C, align(8192))]
    #[derive(Clone, Copy)]
    struct Page8k([u8; 8192]);
    // SAFETY: a single `[u8; 8192]` field with `align(8192)`: size 8192, no
    // padding, and the all-zero bit pattern is a valid value.
    unsafe impl bytemuck::Zeroable for Page8k {}
    // SAFETY: as above, every bit pattern is a valid value; `Page8k` is
    // `Copy + 'static` and holds no pointers or padding bytes.
    unsafe impl bytemuck::Pod for Page8k {}

    /// Message of a caught panic, for assertion output.
    fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
        payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".to_string())
    }

    /// C9.1 (alignment check): `create`, `open` and a growth/shrink remap
    /// must each report `ZiporaError::Misaligned` when the element data
    /// would be misaligned, and a failed remap must leave the vector
    /// unchanged. The unfixed code reported `InvalidData`, and a failed
    /// remap left the header pointer set but the data pointer unset, so
    /// the next `as_slice()` panicked in its `expect`.
    ///
    /// The kernel chooses mapping addresses, so each phase retries (up to
    /// 256 times) and holds a one-page anonymous mapping between attempts
    /// to move the next mapping by a page.
    #[test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_1_misaligned_mapping_reported_and_harmless() {
        // SAFETY: sysconf has no preconditions; _SC_PAGESIZE is a valid name.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = match usize::try_from(page) {
            Ok(page) if page > 0 && page < 8192 => page,
            // Pages of 8 KiB or more align every mapping for Page8k.
            _ => return,
        };
        const ATTEMPTS: usize = 256;
        let dir = tempdir().unwrap();
        let mut fillers = Vec::new();
        let mut keep: Vec<MmapVec<Page8k>> = Vec::new();

        let mut create_err = None;
        let mut aligned = None;
        for i in 0..ATTEMPTS {
            let path = dir.path().join(format!("create{i}.mmap"));
            let config = MmapVecConfig::default().with_initial_capacity(1);
            let mut unmapped_hole = false;
            match MmapVec::<Page8k>::create(path, config) {
                Ok(v) if aligned.is_none() => aligned = Some(v),
                Ok(v) => keep.push(v),
                Err(e) => {
                    unmapped_hole = true;
                    if create_err.is_none() {
                        create_err = Some(e);
                    }
                }
            }
            if create_err.is_some() && aligned.is_some() {
                break;
            }
            if unmapped_hole {
                fillers.push(MmapMut::map_anon(2 * 8192).unwrap());
            }
            fillers.push(MmapMut::map_anon(page).unwrap());
        }

        let path = dir.path().join("handmade.mmap");
        let mut header = MmapVecHeader::new::<Page8k>();
        header.set_length(1);
        header.set_capacity(1);
        let mut file = vec![0xABu8; 2 * 8192];
        file[..HEADER_SIZE].copy_from_slice(&header.data);
        file[HEADER_SIZE..8192].fill(0);
        std::fs::write(&path, &file).unwrap();
        let mut open_err = None;
        let mut opened = None;
        for _ in 0..ATTEMPTS {
            let mut unmapped_hole = false;
            match MmapVec::<Page8k>::open(&path, MmapVecConfig::default()) {
                Ok(v) => {
                    if opened.is_none() {
                        opened = Some((v.len(), v.get(0).map(|e| e.0[0])));
                    }
                    keep.push(v);
                }
                Err(e) => {
                    unmapped_hole = true;
                    if open_err.is_none() {
                        open_err = Some(e);
                    }
                }
            }
            if open_err.is_some() && opened.is_some() {
                break;
            }
            if unmapped_hole {
                fillers.push(MmapMut::map_anon(2 * 8192).unwrap());
            }
            fillers.push(MmapMut::map_anon(page).unwrap());
        }

        let mut remap_err = None;
        let mut after = None;
        if let Some(mut v) = aligned {
            v.push(Page8k([0xAB; 8192])).unwrap();
            for _ in 0..ATTEMPTS {
                let prev_bytes = (v.capacity() + 1) * 8192;
                let step = if v.capacity() > 1 {
                    v.shrink_to_fit()
                } else {
                    v.reserve(2)
                };
                if let Err(e) = step {
                    remap_err = Some(e);
                    break;
                }
                fillers.push(MmapMut::map_anon(prev_bytes).unwrap());
                fillers.push(MmapMut::map_anon(page).unwrap());
            }
            after = Some(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (v.len(), v.as_slice().len(), v.get(0).map(|e| e.0[0]))
                }))
                .map_err(|p| panic_message(&*p)),
            );
            keep.push(v);
        }

        let page_misaligned = |e: &Option<ZiporaError>| {
            matches!(e, Some(ZiporaError::Misaligned { required: 8192, .. }))
        };
        assert!(
            page_misaligned(&create_err)
                && page_misaligned(&open_err)
                && page_misaligned(&remap_err)
                && opened == Some((1, Some(0xAB)))
                && after == Some(Ok((1, 1, Some(0xAB)))),
            "create: {:?}\nopen: {:?} (aligned open: {:?})\nremap: {:?}\nafter the failed remap: {:?}",
            create_err,
            open_err,
            opened,
            remap_err,
            after
        );
    }

    /// C9.1 F2: `len()` and `capacity()` re-read the shared header on
    /// every call, so a foreign write to the header widened every later
    /// view past the mapping. Each handle must keep its own validated
    /// length and capacity.
    #[test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f2_header_rewrite_does_not_widen_views() {
        use std::os::unix::fs::FileExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("f2a.mmap");
        let config = MmapVecConfig::default().with_initial_capacity(4);
        let mut v = MmapVec::<u64>::create(&path, config).unwrap();
        v.push(7).unwrap();

        // Another writer scribbles over length and capacity in the header.
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.write_all_at(&500u64.to_le_bytes(), 16).unwrap();
        file.write_all_at(&500u64.to_le_bytes(), 24).unwrap();
        drop(file);

        // Checked before any element access: on the unfixed code len() and
        // capacity() now read 500, and as_slice() would read past the
        // 112-byte file.
        assert!(
            v.len() <= 4 && v.capacity() <= 4,
            "len {} / capacity {} widened by a foreign header write",
            v.len(),
            v.capacity()
        );
        assert_eq!(v.as_slice(), &[7]);
        assert_eq!(v.get(1), None);
        v.push(8).unwrap();
        assert_eq!(v.as_slice(), &[7, 8]);
    }

    /// C9.1 F2: growth through a second handle rewrote the shared header,
    /// and the first handle then reported the second handle's capacity
    /// over its own, smaller mapping. Growth through either handle must
    /// also never shrink a file the other one already grew.
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f2_second_handle_growth_does_not_widen_first_handle() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f2b.mmap");
        let config = MmapVecConfig::default().with_initial_capacity(4);
        let mut a = MmapVec::<u64>::create(&path, config).unwrap();
        a.push(1).unwrap();

        let mut b = MmapVec::<u64>::open(&path, MmapVecConfig::default()).unwrap();
        b.reserve(100_000).unwrap();
        assert!(
            a.capacity() <= 4 && a.len() <= a.capacity(),
            "first handle reports len {} / capacity {} over a 4-element mapping",
            a.len(),
            a.capacity()
        );
        drop(b);

        a.push(2).unwrap();
        assert_eq!(a.as_slice(), &[1, 2]);
        let grown_len = std::fs::metadata(&path).unwrap().len();
        a.reserve(10).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() >= grown_len);
        assert_eq!(a.as_slice(), &[1, 2]);
    }

    /// C9.1 F3: growth re-opened the backing file by path. Once the path
    /// had been replaced, it resized the vector's own (unlinked) file but
    /// then mapped, and wrote a header into, whatever file now lived at
    /// the path. Growth must keep using the vector's own file handle.
    #[test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f3_growth_never_reopens_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("f3.mmap");
        let config = MmapVecConfig::default().with_initial_capacity(4);
        let mut v = MmapVec::<u64>::create(&path, config).unwrap();
        v.push(42).unwrap();

        // Someone replaces the file at our path.
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, [0u8; 80]).unwrap();
        let grown = v.reserve(10_000);

        // Checked before any element access: on the unfixed code the vector
        // now maps the 80-byte stranger while claiming 10 001 elements.
        assert_eq!(
            std::fs::read(&path).unwrap(),
            vec![0u8; 80],
            "growth wrote into the file that now lives at the path"
        );
        grown.unwrap();
        assert!(v.capacity() >= 10_000);
        v.push(100).unwrap();
        assert_eq!(v.get(0), Some(&42));
        assert_eq!(v.get(1), Some(&100));
    }

    /// C9.1 F6: `reserve` added `len + additional` unchecked, so an
    /// oversized request (direct, or through `extend` trusting an
    /// iterator's `size_hint`) overflowed: a panic in debug builds, a
    /// wrapped and silently ignored request in release builds. It must be
    /// an `Err` that leaves the vector unchanged.
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f6_reserve_and_extend_overflow_are_errors() {
        /// Yields 1, 2, 3 but claims at least `usize::MAX` items.
        struct Liar(u64);
        impl Iterator for Liar {
            type Item = u64;
            fn next(&mut self) -> Option<u64> {
                self.0 += 1;
                (self.0 <= 3).then_some(self.0)
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (usize::MAX, None)
            }
        }

        let dir = tempdir().unwrap();
        let config = MmapVecConfig::default().with_initial_capacity(4);
        let mut v = MmapVec::<u64>::create(dir.path().join("f6.mmap"), config).unwrap();
        v.push(1).unwrap();
        let reserve =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.reserve(usize::MAX)));
        let extend = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.extend(Liar(0))));
        assert!(
            matches!(reserve, Ok(Err(_))) && matches!(extend, Ok(Err(_))),
            "reserve: {:?}\nextend: {:?}",
            reserve.map_err(|p| panic_message(&*p)),
            extend.map_err(|p| panic_message(&*p))
        );
        assert_eq!(v.as_slice(), &[1]);
    }

    /// C9.1 F9: for a zero-sized `T` the file never limits capacity, so a
    /// header may declare `u64::MAX` elements. `push`, `reserve` and
    /// `push_bulk_simd` then computed `len + n` unchecked (a panic in
    /// debug builds). They must return `Err`.
    #[test]
    #[cfg(all(unix, target_pointer_width = "64"))]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f9_zst_length_overflow_is_error() {
        use std::os::unix::fs::FileExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("zst.mmap");
        drop(MmapVec::<()>::create(&path, MmapVecConfig::default()).unwrap());
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.write_all_at(&u64::MAX.to_le_bytes(), 16).unwrap();
        file.write_all_at(&u64::MAX.to_le_bytes(), 24).unwrap();
        drop(file);

        let mut v = MmapVec::<()>::open(&path, MmapVecConfig::default()).unwrap();
        assert_eq!(v.len(), usize::MAX);
        let push = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.push(())));
        let reserve = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.reserve(1)));
        let bulk =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.push_bulk_simd(&[(); 2])));
        assert!(
            matches!(push, Ok(Err(_)))
                && matches!(reserve, Ok(Err(_)))
                && matches!(bulk, Ok(Err(_))),
            "push: {:?}\nreserve: {:?}\npush_bulk_simd: {:?}",
            push.map_err(|p| panic_message(&*p)),
            reserve.map_err(|p| panic_message(&*p)),
            bulk.map_err(|p| panic_message(&*p))
        );
        assert_eq!(v.len(), usize::MAX);
    }

    /// Guard for the C9.1 `open` rewrite (passes before and after): a
    /// zero-length file has no header and must be rejected.
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f2_guard_open_rejects_zero_length_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.mmap");
        std::fs::write(&path, b"").unwrap();
        assert!(MmapVec::<u64>::open(&path, MmapVecConfig::default()).is_err());
    }

    /// Guard for the C9.1 `open` rewrite (passes before and after): a
    /// header declaring more capacity than the file holds must be
    /// rejected.
    #[test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f2_guard_open_rejects_header_capacity_beyond_file() {
        use std::os::unix::fs::FileExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("short.mmap");
        let config = MmapVecConfig::default().with_initial_capacity(4);
        drop(MmapVec::<u64>::create(&path, config).unwrap());
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.write_all_at(&500u64.to_le_bytes(), 16).unwrap();
        file.write_all_at(&500u64.to_le_bytes(), 24).unwrap();
        drop(file);
        assert!(MmapVec::<u64>::open(&path, MmapVecConfig::default()).is_err());
    }

    /// C9.1 (alignment check), GREEN-only: `mapped_capacity_for` is new in
    /// this commit. It is pure address arithmetic, so it also runs under
    /// Miri, which cannot execute the real-file tests.
    #[test]
    fn test_c9_1_mapped_capacity_for_checks_layout() {
        // u64: data at offset 80; 47 payload bytes hold 5 whole elements.
        assert_eq!(
            MmapVec::<u64>::mapped_capacity_for(0x10000, 80 + 5 * 8 + 7).ok(),
            Some(5)
        );
        assert!(matches!(
            MmapVec::<u64>::mapped_capacity_for(0x10000, 79),
            Err(ZiporaError::InvalidData { .. })
        ));
        assert!(matches!(
            MmapVec::<u64>::mapped_capacity_for(0x10004, 200),
            Err(ZiporaError::Misaligned {
                address: 0x10054,
                required: 8,
                ..
            })
        ));
        // Page8k: data at offset 8192, aligned only on even 4 KiB pages.
        assert_eq!(MmapVec::<Page8k>::data_offset(), 8192);
        assert_eq!(
            MmapVec::<Page8k>::mapped_capacity_for(0x4000, 3 * 8192).ok(),
            Some(2)
        );
        assert!(matches!(
            MmapVec::<Page8k>::mapped_capacity_for(0x5000, 3 * 8192),
            Err(ZiporaError::Misaligned {
                address: 0x7000,
                required: 8192,
                ..
            })
        ));
        // Zero-sized elements: the mapping never limits the capacity.
        assert_eq!(
            MmapVec::<()>::mapped_capacity_for(0x1000, 80).ok(),
            Some(usize::MAX)
        );
        // The data address itself must not overflow.
        assert!(MmapVec::<u8>::mapped_capacity_for(usize::MAX - 10, 100).is_err());
    }

    /// C9.1 F5, GREEN-only: `counts_from_header` is new in this commit. It
    /// validates the header and checks the declared capacity against the
    /// mapping. Its `usize::try_from` conversions can only fail where
    /// `usize` is narrower than 64 bits; the zero-sized case below asserts
    /// that rejection on such targets and the full value on 64-bit ones.
    #[test]
    fn test_c9_f5_counts_from_header_validates() {
        let header = |len: u64, cap: u64| {
            let mut h = MmapVecHeader::new::<u64>();
            h.set_length(len);
            h.set_capacity(cap);
            h
        };
        let counts =
            |h: &MmapVecHeader, mapped_cap| MmapVec::<u64>::counts_from_header(h, mapped_cap).ok();
        assert_eq!(counts(&header(3, 5), 5), Some((3, 5)));
        assert_eq!(counts(&header(3, 5), 9), Some((3, 5)));
        assert_eq!(counts(&header(3, 5), 4), None, "capacity beyond mapping");
        assert_eq!(counts(&header(6, 5), 9), None, "length beyond capacity");
        let mut bad_magic = header(3, 5);
        bad_magic.data[0] ^= 1;
        assert_eq!(counts(&bad_magic, 9), None, "bad magic");
        assert!(
            MmapVec::<u32>::counts_from_header(&header(3, 5), 9).is_err(),
            "element size mismatch"
        );

        let mut zst = MmapVecHeader::new::<()>();
        zst.set_length(u64::MAX);
        zst.set_capacity(u64::MAX);
        let zst_counts = MmapVec::<()>::counts_from_header(&zst, usize::MAX).ok();
        #[cfg(target_pointer_width = "64")]
        assert_eq!(zst_counts, Some((usize::MAX, usize::MAX)));
        #[cfg(not(target_pointer_width = "64"))]
        assert_eq!(zst_counts, None);
    }

    /// Guard (passes before and after C9.1): the file size for a capacity
    /// is overflow-checked. Pure arithmetic, so it also runs under Miri.
    #[test]
    fn test_c9_f6_guard_calculate_file_size_is_checked() {
        assert_eq!(MmapVec::<u64>::calculate_file_size(3).ok(), Some(104));
        assert_eq!(MmapVec::<u8>::calculate_file_size(0).ok(), Some(80));
        assert!(MmapVec::<u64>::calculate_file_size(usize::MAX).is_err());
        assert!(MmapVec::<u8>::calculate_file_size(usize::MAX).is_err());
        assert_eq!(
            MmapVec::<()>::calculate_file_size(usize::MAX).ok(),
            Some(80)
        );
    }

    /// C9.1 F7: `compare_range_simd` used `PartialEq` below 64 bytes (`len =
    /// 7` `f64`s) and byte comparison (`fast_compare`) from 64 bytes (`len =
    /// 8` `f64`s) upward, so `0.0` vs `-0.0`, `NaN` vs `NaN`, and custom
    /// `PartialEq` types changed answer at the 64-byte boundary instead of
    /// matching `a.as_slice() == b.as_slice()`.
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f7_compare_range_simd_respects_partial_eq() {
        #[repr(transparent)]
        #[derive(Clone, Copy)]
        struct Mod2(u64);
        // SAFETY: `#[repr(transparent)]` over `u64`: size 8, no padding, and
        // the all-zero bit pattern is a valid value.
        unsafe impl bytemuck::Zeroable for Mod2 {}
        // SAFETY: as above, every bit pattern is a valid `u64`; `Mod2` is
        // `Copy + 'static` and holds no pointers or padding bytes.
        unsafe impl bytemuck::Pod for Mod2 {}
        impl PartialEq for Mod2 {
            fn eq(&self, other: &Self) -> bool {
                self.0 % 2 == other.0 % 2
            }
        }

        let dir = tempdir().unwrap();
        let mut mismatches = Vec::new();
        for len in [7usize, 8] {
            let mut pos = MmapVec::<f64>::create(
                dir.path().join(format!("pos_{len}.mmap")),
                MmapVecConfig::default(),
            )
            .unwrap();
            let mut neg = MmapVec::<f64>::create(
                dir.path().join(format!("neg_{len}.mmap")),
                MmapVecConfig::default(),
            )
            .unwrap();
            let mut nan_a = MmapVec::<f64>::create(
                dir.path().join(format!("nana_{len}.mmap")),
                MmapVecConfig::default(),
            )
            .unwrap();
            let mut nan_b = MmapVec::<f64>::create(
                dir.path().join(format!("nanb_{len}.mmap")),
                MmapVecConfig::default(),
            )
            .unwrap();
            let mut mod_a = MmapVec::<Mod2>::create(
                dir.path().join(format!("moda_{len}.mmap")),
                MmapVecConfig::default(),
            )
            .unwrap();
            let mut mod_b = MmapVec::<Mod2>::create(
                dir.path().join(format!("modb_{len}.mmap")),
                MmapVecConfig::default(),
            )
            .unwrap();
            for _ in 0..len {
                pos.push(0.0f64).unwrap();
                neg.push(-0.0f64).unwrap();
                nan_a.push(f64::NAN).unwrap();
                nan_b.push(f64::NAN).unwrap();
                mod_a.push(Mod2(0)).unwrap();
                mod_b.push(Mod2(2)).unwrap();
            }

            for (label, got, want) in [
                (
                    "0.0 vs -0.0",
                    pos.compare_range_simd(0..len, &neg).unwrap(),
                    pos.as_slice() == neg.as_slice(),
                ),
                (
                    "NaN vs NaN",
                    nan_a.compare_range_simd(0..len, &nan_b).unwrap(),
                    nan_a.as_slice() == nan_b.as_slice(),
                ),
                (
                    "Mod2(0) vs Mod2(2)",
                    mod_a.compare_range_simd(0..len, &mod_b).unwrap(),
                    mod_a.as_slice() == mod_b.as_slice(),
                ),
            ] {
                if got != want {
                    mismatches.push(format!("{label} at len {len}: got {got}, want {want}"));
                }
            }
        }
        assert!(mismatches.is_empty(), "{mismatches:#?}");
    }

    /// C9.1 F8: `with_capacity_simd` used a predictable path in
    /// `std::env::temp_dir()` without `create_new(true)` (`O_EXCL`), so a
    /// symlink planted at the next predicted path had its target truncated
    /// and overwritten; and if mapping initialization failed before
    /// `is_temp_file = true` (e.g. a misaligned `Page8k` mapping), the
    /// newly created temporary file was leaked on disk.
    #[test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f8_with_capacity_simd_refuses_symlinks_and_cleans_up_on_error() {
        let dir = tempdir().unwrap();
        let victim = dir.path().join("victim.dat");
        std::fs::write(&victim, b"precious").unwrap();

        let prefix = format!(
            "mmap_vec_simd_{}_{:?}_",
            std::process::id(),
            std::thread::current().id()
        );
        let first = MmapVec::<u64>::with_capacity_simd(4).unwrap();
        let first_name = first
            .path()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap()
            .to_string();
        let rest = first_name.strip_prefix(&prefix).unwrap();
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        let counter: u64 = digits.parse().unwrap();

        let predicted = std::env::temp_dir().join(format!("{prefix}{}.dat", counter + 1));
        let _ = std::fs::remove_file(&predicted);
        std::os::unix::fs::symlink(&victim, &predicted).unwrap();

        let second = MmapVec::<u64>::with_capacity_simd(4);
        let second_ok = second.is_ok();
        drop(second);
        drop(first);
        let _ = std::fs::remove_file(&predicted);
        let victim_bytes = std::fs::read(&victim).unwrap();

        // Check temp-file cleanup when `from_new_file` fails after creating
        // the file (here via a misaligned `Page8k` mapping on 4 KiB pages).
        // Track the exact thread-local `fastrand` suffixes and counter range
        // rather than calling `read_dir(temp_dir())`, which can stall for
        // seconds when `TMPDIR` contains hundreds of thousands of entries.
        let mut rng = fastrand::Rng::with_seed(fastrand::get_seed());
        let mut tried_rands = Vec::new();
        // SAFETY: sysconf has no preconditions; _SC_PAGESIZE is a valid name.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let mut saw_misaligned = false;
        if let Ok(page) = usize::try_from(page)
            && page > 0
            && page < 8192
        {
            let mut fillers = Vec::new();
            let mut keep = Vec::new();
            for _ in 0..256 {
                // Probe with an anonymous 16 KiB mapping (matching `Page8k`
                // capacity-1 file size without disk fsyncs) and 20 KiB steps
                // until the next 16 KiB VMA slot is misaligned for `Page8k`.
                for _ in 0..64 {
                    let probe = MmapMut::map_anon(2 * 8192).unwrap();
                    if !probe.as_ptr().addr().is_multiple_of(8192) {
                        drop(probe);
                        break;
                    }
                    fillers.push(probe);
                    fillers.push(MmapMut::map_anon(2 * 8192 + page).unwrap());
                }
                tried_rands.push(rng.u64(..));
                match MmapVec::<Page8k>::with_capacity_simd(1) {
                    Ok(v) => keep.push(v),
                    Err(ZiporaError::Misaligned { required: 8192, .. }) => {
                        saw_misaligned = true;
                        break;
                    }
                    Err(_) => {}
                }
                fillers.push(MmapMut::map_anon(2 * 8192 + page).unwrap());
            }
            drop(keep);
        } else {
            saw_misaligned = true;
        }
        let after_rand = rng.u64(..);
        let after_vec = MmapVec::<u64>::with_capacity_simd(4).unwrap();
        let after_name = after_vec
            .path()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap()
            .to_string();
        drop(after_vec);
        let after_rest = after_name.strip_prefix(&prefix).unwrap();
        let after_digits: String = after_rest
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        let end_counter: u64 = after_digits.parse().unwrap();
        assert!(
            after_name.ends_with(&format!("_{after_rand:016x}.dat")),
            "thread-local fastrand out of sync: {after_name} vs {after_rand:016x}"
        );
        let temp_dir = std::env::temp_dir();
        let mut leaked = Vec::new();
        for c in (counter + 1)..end_counter {
            for &rand in &tried_rands {
                let candidate = temp_dir.join(format!("{prefix}{c}_{rand:016x}.dat"));
                if candidate.exists() {
                    leaked.push(candidate);
                }
            }
        }
        for p in &leaked {
            let _ = std::fs::remove_file(p);
        }

        assert!(
            second_ok && victim_bytes == b"precious" && saw_misaligned && leaked.is_empty(),
            "second_ok: {second_ok}, victim_bytes len: {} (expected b\"precious\"), saw_misaligned: {saw_misaligned}, leaked temp files: {}",
            victim_bytes.len(),
            leaked.len()
        );
    }

    /// C9.1 F8, GREEN-only: `create_temp_in` is new in this commit. When a
    /// candidate path is an existing symlink (or regular file), `create_new`
    /// (`O_EXCL`) rejects it with `AlreadyExists` without following the
    /// symlink, retries the next candidate, and fails if every attempt
    /// collides.
    #[test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f8_create_temp_in_retries_past_existing_symlink() {
        let dir = tempdir().unwrap();
        let victim = dir.path().join("victim.dat");
        std::fs::write(&victim, b"precious").unwrap();
        let symlink = dir.path().join("slot0.dat");
        std::os::unix::fs::symlink(&victim, &symlink).unwrap();

        let mut step = 0usize;
        let vec = MmapVec::<u64>::create_temp_in(4, dir.path(), || {
            let name = format!("slot{step}.dat");
            step += 1;
            name
        })
        .unwrap();
        assert_eq!(step, 2);
        assert_eq!(vec.path(), dir.path().join("slot1.dat"));
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        drop(vec);
        assert!(!dir.path().join("slot1.dat").exists());

        // Exhausting all 256 attempts against the symlink must return Err
        // without touching the symlink target.
        assert!(MmapVec::<u64>::create_temp_in(4, dir.path(), || "slot0.dat".to_string()).is_err());
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
    }

    /// C9.1 F9: `MmapVec::create` with `MmapVecConfig::read_only()` truncated
    /// the path and mapped it with `map_copy` (`MAP_PRIVATE`), so the header
    /// was written only to a private copy-on-write page and the file on disk
    /// stayed all zeros (unreadable by `open`). `create` must reject a
    /// read-only configuration before touching the path.
    #[test]
    fn test_c9_f9_create_rejects_read_only_config() {
        let dir = tempdir().unwrap();
        let existing = dir.path().join("existing.mmap");
        std::fs::write(&existing, b"keep-me").unwrap();
        let fresh = dir.path().join("fresh.mmap");

        let err_existing = MmapVec::<u64>::create(&existing, MmapVecConfig::read_only());
        let err_fresh = MmapVec::<u64>::create(&fresh, MmapVecConfig::read_only());
        let existing_bytes = std::fs::read(&existing).unwrap();
        let fresh_exists = fresh.exists();
        assert!(
            err_existing.is_err()
                && err_fresh.is_err()
                && existing_bytes == b"keep-me"
                && !fresh_exists,
            "create(existing, read_only): {:?}, existing_bytes len: {}, create(fresh, read_only): {:?}, fresh_exists: {fresh_exists}",
            err_existing.map(|v| v.capacity()),
            existing_bytes.len(),
            err_fresh.map(|v| v.capacity()),
        );
    }

    /// C9.1 F9: `MmapVecHeader` checked `size_of::<T>()` but not whether `T`
    /// padded `data_offset()` past the 80-byte header (`align_of::<T>() >
    /// 16`), so a file written as `[u8; 32]` (`data_offset = 80`) could be
    /// opened as an `align(32)` type of the same size (`data_offset = 96`),
    /// reading elements shifted by 16 bytes.
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_f9_open_rejects_alignment_padding_mismatch() {
        #[repr(C, align(32))]
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Align32([u8; 32]);
        // SAFETY: a single `[u8; 32]` field with `align(32)`: size 32, no
        // padding, and the all-zero bit pattern is a valid value.
        unsafe impl bytemuck::Zeroable for Align32 {}
        // SAFETY: as above, every bit pattern is valid; `Align32` is
        // `Copy + 'static` and holds no pointers or padding bytes.
        unsafe impl bytemuck::Pod for Align32 {}

        let dir = tempdir().unwrap();
        let path_u8 = dir.path().join("u8_32.mmap");
        let path_a32 = dir.path().join("align32.mmap");
        let config = MmapVecConfig::default().with_initial_capacity(10);

        let mut v_u8 = MmapVec::<[u8; 32]>::create(&path_u8, config.clone()).unwrap();
        v_u8.push([0xAA; 32]).unwrap();
        v_u8.push([0xBB; 32]).unwrap();
        drop(v_u8);
        // Leave 16 bytes of slack in `path_u8` (e.g. from an earlier larger
        // file) so the unfixed `open` does not stop at the file-length check.
        OpenOptions::new()
            .write(true)
            .open(&path_u8)
            .unwrap()
            .set_len(96 + 10 * 32)
            .unwrap();

        let mut v_a32 = MmapVec::<Align32>::create(&path_a32, config).unwrap();
        v_a32.push(Align32([0xCC; 32])).unwrap();
        v_a32.push(Align32([0xDD; 32])).unwrap();
        drop(v_a32);

        let open_u8_as_a32 = MmapVec::<Align32>::open(&path_u8, MmapVecConfig::default())
            .map(|v| v.get(0).map(|e| (e.0[0], e.0[16])));
        let open_a32_as_u8 = MmapVec::<[u8; 32]>::open(&path_a32, MmapVecConfig::default())
            .map(|v| v.get(0).map(|e| (e[0], e[16])));
        assert!(
            open_u8_as_a32.is_err() && open_a32_as_u8.is_err(),
            "u8->Align32: {:?}\nAlign32->u8: {:?}",
            open_u8_as_a32,
            open_a32_as_u8
        );
    }

    /// C9.1 F9, GREEN-only: pure in-memory check (runs under Miri) that
    /// `MmapVecHeader::new` keeps bytes `32..80` all-zero for `align <= 16`
    /// while recording `extended_align` for `align > 16`, and `validate`
    /// rejects both extended-alignment mismatches and flipped endianness flags.
    #[test]
    fn test_c9_f9_header_validates_alignment_and_endianness_flags() {
        let h_u64 = MmapVecHeader::new::<u64>();
        assert_eq!(&h_u64.data[32..80], &[0u8; 48]);
        assert!(h_u64.validate::<u64>().is_ok());

        let h_8k = MmapVecHeader::new::<Page8k>();
        assert_eq!(h_8k.extended_align(), 8192);
        assert!(h_8k.validate::<Page8k>().is_ok());
        assert!(h_8k.validate::<[u8; 8192]>().is_err());

        let h_bytes_8k = MmapVecHeader::new::<[u8; 8192]>();
        assert_eq!(h_bytes_8k.extended_align(), 0);
        assert!(h_bytes_8k.validate::<Page8k>().is_err());

        let mut flipped_endian = h_u64;
        let bad_flags = MMAP_VEC_FLAGS ^ 0x0001;
        flipped_endian.data[10..12].copy_from_slice(&bad_flags.to_le_bytes());
        assert!(flipped_endian.validate::<u64>().is_err());
    }

    /// C9.1(d), GREEN-only: exercises every `AccessPattern` variant,
    /// `use_huge_pages`, `populate_pages`, and `sync_on_write` (`true`/`false`)
    /// across `create` -> `push`/grow remap -> `reserve` remap ->
    /// `shrink_to_fit` remap -> drop without `sync()` -> `open`, verifying
    /// `advise_mapping` and `MAP_SHARED` persistence without asserting page
    /// residency.
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_madvise_knobs_and_map_shared_survive_remap_and_reopen() {
        let dir = tempdir().unwrap();
        let patterns = [
            AccessPattern::Sequential,
            AccessPattern::ReadHeavy,
            AccessPattern::Random,
            AccessPattern::WriteHeavy,
            AccessPattern::Mixed,
        ];

        for (idx, pattern) in patterns.into_iter().enumerate() {
            for sync_on_write in [false, true] {
                let path = dir.path().join(format!("knobs_{idx}_{sync_on_write}.mmap"));
                let config = MmapVecConfig {
                    initial_capacity: 2,
                    growth_factor: 2.0,
                    read_only: false,
                    populate_pages: true,
                    use_huge_pages: true,
                    sync_on_write,
                    access_pattern: pattern,
                };

                let mut vec = MmapVec::<u64>::create(&path, config.clone()).unwrap();
                for v in 0..5u64 {
                    vec.push(v * 10 + idx as u64).unwrap();
                }
                vec.reserve(16).unwrap();
                vec.as_mut_slice()[0] = 999 + idx as u64;
                vec.shrink_to_fit().unwrap();
                assert_eq!(vec.len(), 5);
                assert_eq!(vec.capacity(), 5);
                // Drop without calling `sync()`: `MAP_SHARED` writes must still
                // be visible when reopened.
                drop(vec);

                let ro = MmapVec::<u64>::open(&path, MmapVecConfig::read_only()).unwrap();
                assert_eq!(ro.len(), 5);
                assert_eq!(ro.capacity(), 5);
                assert_eq!(
                    ro.as_slice(),
                    &[
                        999 + idx as u64,
                        10 + idx as u64,
                        20 + idx as u64,
                        30 + idx as u64,
                        40 + idx as u64,
                    ]
                );
            }
        }
    }

    /// S22-R5: `MmapVec::open` with `MmapVecConfig::read_only()` (`populate_pages: true`)
    /// previously mapped the file via `MmapOptions::map_copy` (`MAP_PRIVATE` with
    /// `PROT_READ | PROT_WRITE`). On Linux, combining `MAP_POPULATE` with a writable
    /// `MAP_PRIVATE` mapping write-faulted every page into private anonymous memory
    /// (`RssAnon`). With `MappingKind::ReadOnly(Mmap)` (`PROT_READ`), `MAP_POPULATE`
    /// faults clean file-backed page-cache pages without inflating `RssAnon`.
    #[test]
    #[cfg(target_os = "linux")]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_s22_r5_read_only_open_does_not_fault_into_rss_anon() {
        // S54-R3: `/proc/self/status` `RssAnon` is process-wide, so concurrent test
        // threads in the parent test runner can allocate anonymous heap pages during
        // `MmapVec::open` and inflate `RssAnon`. Run the measurement in an isolated
        // `--test-threads=1` child process when not already inside that subprocess.
        if std::env::var_os("ZIPORA_S22_R5_SUBPROCESS").is_none() {
            let exe = std::env::current_exe().expect("current test executable path");
            let output = std::process::Command::new(exe)
                .env("ZIPORA_S22_R5_SUBPROCESS", "1")
                .args([
                    "--exact",
                    "memory::mmap_vec::tests::test_s22_r5_read_only_open_does_not_fault_into_rss_anon",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .output()
                .expect("spawn subprocess for isolated RssAnon measurement");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "subprocess RssAnon test failed:\nstdout: {stdout}\nstderr: {stderr}"
            );
            return;
        }

        fn rss_anon_kib() -> Option<u64> {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("RssAnon:") {
                    let digits = rest.trim().strip_suffix("kB")?.trim();
                    return digits.parse::<u64>().ok();
                }
            }
            None
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("ro_populate.mmap");
        let n = 4 * 1024 * 1024usize; // 4M * u64 = 32 MiB
        {
            let config = MmapVecConfig::default().with_initial_capacity(n);
            let mut vec = MmapVec::<u64>::create(&path, config).unwrap();
            vec.resize(n, 7).unwrap();
            vec.sync().unwrap();
        }

        let before = rss_anon_kib();
        let ro = MmapVec::<u64>::open(&path, MmapVecConfig::read_only()).unwrap();
        assert_eq!(ro.len(), n);
        let sum: u64 = ro.as_slice().iter().step_by(512).copied().sum();
        assert_eq!(sum, (n as u64 / 512) * 7);
        if let (Some(b), Some(a)) = (before, rss_anon_kib()) {
            let delta_kib = a.saturating_sub(b);
            assert!(
                delta_kib < 4096,
                "opening 32 MiB read-only MmapVec with populate_pages inflated RssAnon by {delta_kib} KiB (before={b} KiB, after={a} KiB)"
            );
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_s54_r3_rss_anon_measurement_isolated_from_parent_threads() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = Arc::clone(&stop);
        let bg = std::thread::spawn(move || {
            let mut chunks = Vec::new();
            while !stop_bg.load(Ordering::Relaxed) {
                if chunks.len() >= 16 {
                    chunks.clear();
                }
                if let Ok(mut anon) = MmapMut::map_anon(8 * 1024 * 1024) {
                    for page in anon.chunks_mut(4096) {
                        page[0] = 1;
                    }
                    chunks.push(anon);
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            chunks.len()
        });

        let res = std::panic::catch_unwind(|| {
            for _ in 0..3 {
                test_s22_r5_read_only_open_does_not_fault_into_rss_anon();
            }
        });
        stop.store(true, Ordering::Relaxed);
        let _ = bg.join();
        if let Err(payload) = res {
            std::panic::resume_unwind(payload);
        }
    }

    #[test]
    fn test_s23_r3_mmap_vec_config_rejects_invalid_growth_factor() {
        let invalid_factors = [
            0.0,
            1.0,
            -2.0,
            1024.1,
            1e30,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];

        #[cfg(not(miri))]
        {
            let dir = tempdir().unwrap();
            let path = dir.path().join("valid.mmap");
            {
                let mut vec = MmapVec::<u64>::create(
                    &path,
                    MmapVecConfig::default().with_initial_capacity(4),
                )
                .unwrap();
                vec.push(42).unwrap();
                vec.sync().unwrap();
            }

            // Calling create() or open() with an invalid growth_factor must fail
            // before truncating or mapping `path`.
            for &bad in &invalid_factors {
                let cfg = MmapVecConfig::default().with_growth_factor(bad);
                assert!(
                    MmapVec::<u64>::create(&path, cfg.clone()).is_err(),
                    "growth_factor {bad} must be rejected by MmapVec::create"
                );
                assert!(
                    MmapVec::<u64>::open(&path, cfg).is_err(),
                    "growth_factor {bad} must be rejected by MmapVec::open"
                );
            }

            let reopened = MmapVec::<u64>::open(&path, MmapVecConfig::read_only()).unwrap();
            assert_eq!(reopened.as_slice(), &[42]);
        }

        for &bad in &invalid_factors {
            let cfg = MmapVecConfig::default().with_growth_factor(bad);
            assert!(
                matches!(cfg.validate(), Err(ZiporaError::InvalidParameter { .. })),
                "growth_factor {bad} must be rejected by cfg.validate() with InvalidParameter"
            );
            assert!(
                matches!(
                    MmapVec::<u64>::create("/nonexistent/s23_r3_never_opened.mmap", cfg),
                    Err(ZiporaError::InvalidParameter { .. })
                ),
                "growth_factor {bad} must be rejected with InvalidParameter before touching disk"
            );
        }
    }
}
