//! C FFI bindings for algorithm types

use super::types::*;
use crate::algorithms::{RadixSort, SuffixArray};
use crate::ffi::CResult;

// suffix_array_new function is defined in c_api.rs to avoid duplication

// suffix_array_free function is defined in c_api.rs to avoid duplication

// suffix_array_search function is defined in c_api.rs to avoid duplication

// radix_sort_u32 function is defined in c_api.rs to avoid duplication
