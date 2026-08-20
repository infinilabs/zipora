# C FFI

Zipora exposes a small, focused C API (feature-gated behind `ffi`) covering library
initialization, error handling, blob storage, `FastVec<u8>`, memory pools, and two
algorithms (radix sort, suffix array). All entry points live in `src/ffi/c_api.rs`.

## Enabling FFI

```toml
[dependencies]
zipora = { version = "4.0.2", features = ["ffi"] }
```

The `ffi` feature pulls in `cbindgen` as a build dependency (`ffi = ["dep:cbindgen"]`).

## Building and Locating the Header

```bash
cargo build --release --features ffi
```

The build script generates the C header with cbindgen and writes it to
**`$OUT_DIR/zipora.h`** (not `target/zipora.h`). `OUT_DIR` is a per-crate build
directory chosen by Cargo; locate the header with:

```bash
find target/release/build -name zipora.h
```

Copy it wherever your C build expects headers, or wire the path into your build system.

## Error Handling and Panic Safety

Most functions return `CResult` (a `#[repr(C)]` enum):

| Value | Meaning |
|-------|---------|
| `Success = 0` | Operation completed successfully |
| `InvalidInput = -1` | Invalid input parameters (including NULL pointers) |
| `MemoryError = -2` | Memory allocation or access error |
| `IoError = -3` | Input/output operation failed |
| `UnsupportedOperation = -4` | Operation not supported in current context |
| `InternalError = -5` | Internal library error |
| `NotFound = -6` | Requested item was not found |

Constructor-style functions return `NULL` on failure instead. A thread-local
last-error message is available, plus an optional global error callback:

```c
const char *zipora_last_error(void);                         // never NULL
void zipora_set_error_callback(void (*callback)(const char *msg)); // NULL clears
```

**Panic safety:** every exported function body runs inside `ffi_guard`
(`std::panic::catch_unwind`, `src/ffi/c_api.rs`). A Rust panic never unwinds across
the FFI boundary (which would abort the process); instead the function returns its
error fallback (`InternalError`, `NULL`, `0`, ...) and the panic message is surfaced
through the error callback / `zipora_last_error()`.

## Init / Version

```c
CResult zipora_init(void);          // initialize the library (call once)
const char *zipora_version(void);   // static version string, do not free
int zipora_has_simd(void);          // 1 if SIMD support is available, else 0
```

## Blob Store (in-memory)

`blob_store_new` creates an in-memory blob store (`MemoryBlobStore`).

```c
CBlobStore *blob_store_new(void);
void blob_store_free(CBlobStore *store);

CResult blob_store_put(CBlobStore *store, const uint8_t *data, size_t size,
                       uint32_t *record_id);
CResult blob_store_get(const CBlobStore *store, uint32_t record_id,
                       const uint8_t **data, size_t *size);

// blob_store_get returns a copy allocated by Rust — free it with:
void zipora_free_blob_data(uint8_t *data, size_t size);
```

Example:

```c
CBlobStore *store = blob_store_new();
uint32_t id;
if (blob_store_put(store, (const uint8_t *)"blob data", 9, &id) == Success) {
    const uint8_t *data;
    size_t size;
    if (blob_store_get(store, id, &data, &size) == Success) {
        /* use data[0..size] */
        zipora_free_blob_data((uint8_t *)data, size);
    }
}
blob_store_free(store);
```

## FastVec (byte vector)

A growable `FastVec<u8>` behind an opaque handle:

```c
CFastVec *fast_vec_new(void);
void fast_vec_free(CFastVec *vec);
CResult fast_vec_push(CFastVec *vec, uint8_t value);
size_t fast_vec_len(const CFastVec *vec);          // 0 for NULL
const uint8_t *fast_vec_data(const CFastVec *vec); // NULL for NULL/empty
```

The pointer returned by `fast_vec_data` is invalidated by the next `fast_vec_push`
(reallocation) and by `fast_vec_free`.

## Memory Pool

Fixed-chunk memory pool (`zipora::memory::MemoryPool`):

```c
CMemoryPool *memory_pool_new(size_t chunk_size, size_t max_chunks);
void memory_pool_free(CMemoryPool *pool);
void *memory_pool_allocate(CMemoryPool *pool);                  // one chunk, NULL on failure
CResult memory_pool_deallocate(CMemoryPool *pool, void *ptr);   // return chunk to pool
```

## Algorithms

```c
// In-place radix sort of a u32 array
CResult radix_sort_u32(uint32_t *data, size_t size);

// Suffix array construction + pattern search
CSuffixArray *suffix_array_new(const uint8_t *text, size_t size);
void suffix_array_free(CSuffixArray *sa);
size_t suffix_array_len(const CSuffixArray *sa);
CResult suffix_array_search(const CSuffixArray *sa,
                            const uint8_t *text, size_t text_size,
                            const uint8_t *pattern, size_t pattern_size,
                            size_t *start, size_t *count);
```

`suffix_array_search` writes the range of matching suffixes into `start`/`count`
(`count == 0` means no match); pass the same `text` the suffix array was built from.

## General Rules

- All handle types (`CBlobStore`, `CFastVec`, `CMemoryPool`, `CSuffixArray`) are
  opaque; only use the functions above.
- Every `*_free` function accepts `NULL` (no-op). Do not free a handle twice.
- NULL/invalid arguments are rejected with `InvalidInput` (or `NULL`/`0` returns) —
  they never crash.
- Handles are not internally synchronized; guard concurrent mutation with your own lock.
