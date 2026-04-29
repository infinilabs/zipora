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

pub mod base64;
pub mod cpu_features;
#[cfg(feature = "async")]
pub mod process;
pub mod profiling;
pub mod vm_utils;

// Re-export core functionality
pub use base64::{
    AdaptiveBase64, SimdBase64Decoder, SimdBase64Encoder, base64_decode_simd, base64_encode_simd,
};
pub use cpu_features::{CpuFeatures, RuntimeCpuFeatures, get_cpu_features, has_cpu_feature};
#[cfg(feature = "async")]
pub use process::{BidirectionalPipe, ProcessExecutor, ProcessManager, ProcessPool};
pub use profiling::{BenchmarkSuite, HighPrecisionTimer, PerfTimer, ProfiledFunction};
pub use vm_utils::{KernelInfo, PageAlignedAlloc, VmManager, get_kernel_info, vm_prefetch};

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
