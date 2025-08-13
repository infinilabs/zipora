//! # System Integration Utilities
//!
//! This module provides comprehensive system integration utilities for high-performance computing,
//! including CPU feature detection, performance profiling, process management, and hardware acceleration.
//!
//! ## Features
//!
//! - **CPU Feature Detection**: Runtime detection of SIMD capabilities (AVX2, BMI2, AVX-512, NEON)
//! - **Performance Profiling**: High-precision timing and benchmarking framework
//! - **Process Management**: Process spawning, control, and bidirectional communication
//! - **Base64 SIMD**: Hardware-accelerated encoding/decoding with adaptive selection
//! - **Virtual Memory Management**: Kernel-aware memory operations and prefetching
//!
//! ## Architecture
//!
//! This implementation is inspired by production-grade system utilities while leveraging Rust's
//! memory safety guarantees and modern language features.

pub mod cpu_features;
pub mod profiling;
pub mod process;
pub mod base64;
pub mod vm_utils;

// Re-export core functionality
pub use cpu_features::{CpuFeatureSet, RuntimeCpuFeatures, get_cpu_features, has_cpu_feature};
pub use profiling::{PerfTimer, BenchmarkSuite, HighPrecisionTimer, ProfiledFunction};
pub use process::{ProcessManager, ProcessPool, BidirectionalPipe, ProcessExecutor};
pub use base64::{AdaptiveBase64, SimdBase64Encoder, SimdBase64Decoder, base64_encode_simd, base64_decode_simd};
pub use vm_utils::{VmManager, PageAlignedAlloc, KernelInfo, vm_prefetch, get_kernel_info};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Test that module imports work
        let _features = get_cpu_features();
        let _timer = HighPrecisionTimer::new();
        let _kernel = get_kernel_info();
    }
}