# Error Handling

Zipora uses **`ZiporaError` / `Result<T>`** (`src/error.rs`) as the crate-wide error type returned by all fallible APIs.

## ZiporaError and Result

`ZiporaError` is the main error type, re-exported at the crate root along with
the `Result<T>` alias (`std::result::Result<T, ZiporaError>`).

### Error Variants and Constructors

| Variant | Constructor | Recoverable |
|---------|-------------|-------------|
| `Io` | `io_error(msg)`, `not_found(msg)`, or `From<std::io::Error>` | yes |
| `InvalidData` | `invalid_data(msg)` | no |
| `OutOfBounds` | `out_of_bounds(index, size)` | no |
| `OutOfMemory` | `out_of_memory(size)` | yes |
| `Compression` | `compression(msg)` | no |
| `BlobStore` | `blob_store(msg)` | no |
| `Trie` | `trie(msg)` | no |
| `ChecksumMismatch` | `checksum_mismatch(expected, actual)` | no |
| `NotSupported` | `not_supported(feature)` | no |
| `Configuration` | `configuration(msg)` | no |
| `ResourceBusy` | `resource_busy(resource)` | yes |
| `Timeout` | `timeout(msg)` | yes |
| `SystemError` | `system_error(msg)` | no |
| `ResourceExhausted` | `resource_exhausted(msg)` | yes |
| `InvalidParameter` | `invalid_parameter(msg)` | no |
| `InvalidOperation` | `invalid_operation(msg)` | no |
| `InvalidState` | `invalid_state(msg)` | no |
| `Serialization` | `From<serde_json::Error>` (`serde` feature) | no |

### Idiomatic Usage

```rust
use zipora::{Result, ZiporaError, FastVec};

fn build_vec(data: &[u8]) -> Result<FastVec<u8>> {
    if data.is_empty() {
        return Err(ZiporaError::invalid_data("input must not be empty"));
    }

    let mut vec = FastVec::new();
    for &byte in data {
        vec.push(byte)?; // ZiporaError propagates with `?`
    }
    Ok(vec)
}

// std::io::Error converts automatically
fn read_input(path: &str) -> Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}
```

### Error Classification

```rust
use zipora::ZiporaError;

let err = ZiporaError::out_of_memory(1024);
assert!(err.is_recoverable());        // retry may succeed
assert_eq!(err.category(), "memory"); // static str for logging/metrics
```

### Bounds Checking Helpers

`src/error.rs` also provides `Result`-returning bounds checks:

```rust
use zipora::Result;
use zipora::error::{check_bounds, check_range};

fn validate_slice(index: usize, start: usize, end: usize, size: usize) -> Result<()> {
    check_bounds(index, size)?;     // Err(OutOfBounds) if index >= size
    check_range(start, end, size)?; // validates start <= end && end <= size
    Ok(())
}

validate_slice(2, 1, 4, 10).unwrap();
assert!(validate_slice(10, 1, 4, 10).is_err());
```
