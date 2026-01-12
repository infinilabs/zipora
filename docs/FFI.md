# C FFI Migration

Zipora provides a complete C API for migration from C++ codebases.

## Overview

The FFI layer provides C-compatible interfaces for all major Zipora components, enabling:

- **Gradual migration** from C++ to Rust
- **Interoperability** with existing C/C++ code
- **Memory safety** through safe wrappers
- **Zero-cost abstractions** where possible

## Enabling FFI

```toml
[dependencies]
zipora = { version = "2.1.1", features = ["ffi"] }
```

## Building the C Library

```bash
# Build with FFI support
cargo build --release --features ffi

# Generate C headers
cargo build --release --features ffi
# Headers are generated at: target/zipora.h
```

## C API Examples

### Memory Management

```c
#include "zipora.h"

// Create secure memory pool
ZiporaSecurePool* pool = zipora_secure_pool_new(ZIPORA_POOL_SMALL);
if (!pool) {
    fprintf(stderr, "Failed to create pool\n");
    return -1;
}

// Allocate memory
ZiporaAllocation* alloc = zipora_pool_allocate(pool, 1024);
if (alloc) {
    void* ptr = zipora_allocation_ptr(alloc);
    size_t size = zipora_allocation_size(alloc);

    // Use memory...
    memset(ptr, 0, size);

    // Free allocation (RAII-style)
    zipora_allocation_free(alloc);
}

// Cleanup pool
zipora_secure_pool_free(pool);
```

### Hash Maps

```c
// Create hash map
ZiporaHashMap* map = zipora_hashmap_new();

// Insert key-value pairs
zipora_hashmap_insert(map, "key1", 5, "value1", 6);
zipora_hashmap_insert(map, "key2", 5, "value2", 6);

// Lookup
size_t value_len;
const char* value = zipora_hashmap_get(map, "key1", 5, &value_len);
if (value) {
    printf("Found: %.*s\n", (int)value_len, value);
}

// Iterate
ZiporaHashMapIter* iter = zipora_hashmap_iter(map);
const char *key, *val;
size_t key_len, val_len;
while (zipora_hashmap_iter_next(iter, &key, &key_len, &val, &val_len)) {
    printf("%.*s => %.*s\n", (int)key_len, key, (int)val_len, val);
}
zipora_hashmap_iter_free(iter);

// Cleanup
zipora_hashmap_free(map);
```

### Blob Storage

```c
// Create blob store
ZiporaBlobStore* store = zipora_blobstore_new(ZIPORA_BLOB_MEMORY);

// Store data
uint64_t id = zipora_blobstore_put(store, "blob data", 9);

// Retrieve data
size_t data_len;
const uint8_t* data = zipora_blobstore_get(store, id, &data_len);
if (data) {
    printf("Retrieved %zu bytes\n", data_len);
}

// Cleanup
zipora_blobstore_free(store);
```

### Compression

```c
// Create compressor
ZiporaCompressor* comp = zipora_compressor_new(ZIPORA_COMPRESS_ZSTD);
zipora_compressor_set_level(comp, 10);

// Compress data
const uint8_t* input = (const uint8_t*)"data to compress";
size_t input_len = 17;
size_t output_capacity = zipora_compress_bound(input_len);
uint8_t* output = malloc(output_capacity);

size_t compressed_len = zipora_compress(comp, input, input_len, output, output_capacity);
if (compressed_len > 0) {
    printf("Compressed %zu -> %zu bytes\n", input_len, compressed_len);
}

// Decompress
ZiporaDecompressor* decomp = zipora_decompressor_new(ZIPORA_COMPRESS_ZSTD);
uint8_t* decompressed = malloc(input_len);
size_t decompressed_len = zipora_decompress(decomp, output, compressed_len,
                                            decompressed, input_len);

// Cleanup
free(output);
free(decompressed);
zipora_compressor_free(comp);
zipora_decompressor_free(decomp);
```

### Tries

```c
// Create trie
ZiporaTrie* trie = zipora_trie_new(ZIPORA_TRIE_PATRICIA);

// Insert keys
zipora_trie_insert(trie, "hello", 5);
zipora_trie_insert(trie, "help", 4);
zipora_trie_insert(trie, "world", 5);

// Lookup
if (zipora_trie_contains(trie, "hello", 5)) {
    printf("Found 'hello'\n");
}

// Prefix iteration
ZiporaTrieIter* iter = zipora_trie_iter_prefix(trie, "hel", 3);
const char* key;
size_t key_len;
while (zipora_trie_iter_next(iter, &key, &key_len)) {
    printf("Prefix match: %.*s\n", (int)key_len, key);
}
zipora_trie_iter_free(iter);

// Cleanup
zipora_trie_free(trie);
```

## Error Handling

```c
// All functions return error codes or NULL on failure
ZiporaResult result = zipora_operation(...);
if (result.error != ZIPORA_OK) {
    const char* msg = zipora_error_message(result.error);
    fprintf(stderr, "Error: %s\n", msg);
}

// Get last error for functions returning NULL
if (!ptr) {
    ZiporaError err = zipora_last_error();
    fprintf(stderr, "Error %d: %s\n", err, zipora_error_message(err));
}
```

## Thread Safety

```c
// Most Zipora types are thread-safe for reads
// Write operations require external synchronization or use concurrent variants

// Thread-safe concurrent map
ZiporaConcurrentMap* cmap = zipora_concurrent_map_new(8); // 8 shards
zipora_concurrent_map_insert(cmap, "key", 3, "value", 5);

// Safe from multiple threads
pthread_t threads[4];
for (int i = 0; i < 4; i++) {
    pthread_create(&threads[i], NULL, worker, cmap);
}
```

## Memory Safety Guarantees

- **No use-after-free**: All pointers are validated
- **No double-free**: Reference counting where needed
- **No buffer overflows**: Bounds checking on all operations
- **Thread safety**: Documented thread safety for each type

## Migration Guide from Terark C++

| Terark C++ | Zipora C FFI |
|------------|--------------|
| `NestLoudsTrieDAWG` | `zipora_trie_new(ZIPORA_TRIE_NESTED_LOUDS)` |
| `Patricia` | `zipora_trie_new(ZIPORA_TRIE_PATRICIA)` |
| `gold_hash_map` | `zipora_hashmap_new()` or `zipora_gold_hashmap_new()` |
| `SecureMemoryPool` | `zipora_secure_pool_new()` |
| `BlobStore` | `zipora_blobstore_new()` |
| `rank_select_il_256` | `zipora_rankselect_new(ZIPORA_RS_IL256)` |

## Performance Considerations

- FFI calls have minimal overhead (~1-2ns per call)
- Batch operations reduce FFI overhead
- Zero-copy where possible (data remains in Rust memory)
- Use streaming APIs for large data transfers
