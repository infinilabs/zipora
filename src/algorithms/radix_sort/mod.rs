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

mod advanced;
mod config;
mod lsd;
#[cfg(test)]
mod tests;

pub use advanced::{
    AdvancedRadixSort, AdvancedStringRadixSort, AdvancedU32RadixSort, AdvancedU64RadixSort,
    RadixSortable, RadixString,
};
pub use config::{
    AdvancedAlgorithmStats, AdvancedRadixSortConfig, CpuFeatures, DataCharacteristics, PhaseTimes,
    RadixSortConfig, SortingStrategy,
};
pub use lsd::{KeyValueRadixSort, RadixSort};
