//! High-performance radix sort implementation with SIMD optimizations
//!
//! This module provides radix sort implementations for various data types,
//! including optimizations for specific use cases and parallel processing.
//!
//! ## Advanced Radix Sort Variants
//!
//! The `AdvancedRadixSort` provides sophisticated algorithm variants:
//! - **LSD radix sort**: Enhanced with AVX2, BMI2, and POPCNT optimizations
//! - **MSD string radix sort**: Advanced string-specific optimizations
//! - **Adaptive hybrid approach**: Intelligent strategy selection based on data characteristics
//! - **Parallel processing**: Work-stealing implementation with optimal load balancing
//! - **Runtime feature detection**: Optimal SIMD usage based on CPU capabilities

mod config;
mod lsd;
mod advanced;
#[cfg(test)]
mod tests;

pub use config::{
    AdvancedRadixSortConfig, AdvancedAlgorithmStats, CpuFeatures, DataCharacteristics,
    PhaseTimes, RadixSortConfig, SortingStrategy,
};
pub use lsd::{RadixSort, KeyValueRadixSort};
pub use advanced::{
    AdvancedRadixSort, AdvancedStringRadixSort, AdvancedU32RadixSort, AdvancedU64RadixSort,
    RadixSortable, RadixString,
};
