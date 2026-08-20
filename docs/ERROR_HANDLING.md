# Error Handling

Zipora provides two complementary error handling mechanisms:

- **`ZiporaError` / `Result<T>`** (`src/error.rs`): the crate-wide error type returned by all fallible APIs
- **Verification macros** (`src/error_recovery.rs`): fail-fast runtime assertions for invariants that must never be violated

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
use zipora::error::{check_bounds, check_range};

check_bounds(index, size)?;       // Err(OutOfBounds) if index >= size
check_range(start, end, size)?;   // validates start <= end && end <= size
```

## Verification Macros

`src/error_recovery.rs` provides fail-fast verification macros with rich
contextual messages (file, line, expression). On failure they print the
context and **abort the process** (`std::process::abort()`); under
`cfg(test)` they panic instead so tests can recover. Use them for internal
invariants where continuing would risk memory corruption — use
`ZiporaError` for recoverable, caller-facing failures.

```rust
use zipora::{zipora_verify, zipora_verify_eq, zipora_verify_lt, zipora_die};

// Basic verification, with optional formatted context
zipora_verify!(index < size);
zipora_verify!(index < size, "index {} out of bounds for size {}", index, size);

// Comparison macros (display both values on failure)
zipora_verify_eq!(actual, expected);  // also: zipora_verify_ne!
zipora_verify_lt!(value, limit);      // also: _le!, _gt!, _ge!
zipora_verify_ez!(status_code);       // verifies value == 0

// Fatal error macro for immediate termination
if critical_condition {
    zipora_die!("critical failure: {}", detail);
}
```

### Specialized Macros

| Macro | Checks |
|-------|--------|
| `zipora_verify_alloc!(ptr, size)` | allocation pointer is non-null |
| `zipora_verify_aligned!(ptr, align)` | pointer (or size) is aligned |
| `zipora_verify_pow2!(val)` | value is a power of 2 |
| `zipora_verify_not_null!(ptr)` | pointer is non-null |
| `zipora_verify_bounds!(index, size)` | `index < size` |
| `zipora_verify_range!(start, end, size)` | `start <= end && end <= size` |
| `zipora_verify_capacity!(current, required, max)` | capacity invariants |
| `zipora_verify_syscall!(result, name)` | syscall returned 0 (reports `last_os_error`) |

### Verification Helper Functions

For use in generic contexts, five function wrappers are re-exported at the
crate root:

```rust
use zipora::{
    verify_alignment,          // (ptr: *const u8, align: usize)
    verify_power_of_2,         // (val: usize)
    verify_allocation_success, // (ptr: *const u8, size: usize)
    verify_bounds_check,       // (index: usize, size: usize)
    verify_range_check,        // (start: usize, end: usize, size: usize)
};

verify_bounds_check(5, 10);      // ok
verify_power_of_2(1024);         // ok
verify_range_check(2, 8, 10);    // ok — aborts on violation
```
