//! Dropping an `LruPageCache` (or a `CachedBlobStore`) must not write back dirty pages of
//! virtual (`register_file(-1)`) files. Nothing can read a virtual file once the cache is gone,
//! and `FileManager::write_page` grows its in-memory buffer to the page offset, so the write
//! could only allocate and zero-fill memory: 1 GiB here, for a single dirty byte.
//!
//! The `#[global_allocator]` records the largest request made while armed and refuses anything
//! above 64 MiB, so the test never commits that memory. It is process-global, so this file
//! holds a single `#[test]`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use zipora::RecordId;
use zipora::blob_store::{BlobStore, CachedBlobStore};
use zipora::cache::{LruPageCache, PageCacheConfig};
use zipora::error::Result;

struct Probe;

static ARMED: AtomicBool = AtomicBool::new(false);
static MAX_REQUEST: AtomicUsize = AtomicUsize::new(0);
const LIMIT: usize = 64 << 20;

fn note(size: usize) -> bool {
    if ARMED.load(Ordering::Relaxed) {
        MAX_REQUEST.fetch_max(size, Ordering::Relaxed);
        return size > LIMIT;
    }
    false
}

unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if note(layout.size()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if note(layout.size()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if note(new_size) {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Probe = Probe;

fn peak_during(f: impl FnOnce()) -> usize {
    MAX_REQUEST.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    f();
    ARMED.store(false, Ordering::Relaxed);
    MAX_REQUEST.load(Ordering::Relaxed)
}

/// `BlobStore` that keeps nothing, so the probe needs no blob memory.
#[derive(Default)]
struct NullStore {
    n: usize,
}

impl BlobStore for NullStore {
    fn get(&self, _id: RecordId) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
    fn put(&mut self, _data: &[u8]) -> Result<RecordId> {
        self.n += 1;
        Ok(self.n as RecordId)
    }
    fn remove(&mut self, _id: RecordId) -> Result<()> {
        Ok(())
    }
    fn contains(&self, _id: RecordId) -> bool {
        true
    }
    fn size(&self, _id: RecordId) -> Result<Option<usize>> {
        Ok(None)
    }
    fn len(&self) -> usize {
        self.n
    }
}

/// One `#[test]` only: the allocator flag is process-global, so no other
/// test may allocate while it is armed.
#[test]
fn test_v2_drop_does_not_grow_virtual_files() {
    let mut failures = Vec::new();

    // (1) Direct: one dirty byte of a virtual file at 1 GiB.
    let cache = LruPageCache::new(PageCacheConfig::balanced()).unwrap();
    let fid = cache.register_file(-1).unwrap();
    cache.write(fid, 1 << 30, b"x").unwrap();
    let peak = peak_during(move || drop(cache));
    if peak >= LIMIT {
        failures.push(format!(
            "LruPageCache::drop requested a {peak}-byte allocation to flush one byte of a virtual file"
        ));
    }

    // (2) Public blob-store API, ordinary calls only.
    let mut store =
        CachedBlobStore::new(NullStore::default(), PageCacheConfig::balanced()).unwrap();
    {
        // 1 GiB logical blob; `vec![0; n]` is calloc-backed and never touched.
        let big = vec![0u8; 1 << 30];
        store.put(&big).unwrap();
    }
    let id = store.put(b"x").unwrap(); // logical offset 1 GiB
    let _ = store.get(id).unwrap(); // caches that (empty, dirty) page
    let peak = peak_during(move || drop(store));
    if peak >= LIMIT {
        failures.push(format!(
            "dropping a CachedBlobStore requested a {peak}-byte allocation"
        ));
    }

    assert!(failures.is_empty(), "{failures:#?}");
}
