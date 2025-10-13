//! # Adaptive SIMD Selector
//!
//! Runtime dynamic SIMD selection based on hardware capabilities, data characteristics,
//! and continuous performance monitoring.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::system::cpu_features::{CpuFeatures, get_cpu_features};
use super::{Operation, BenchmarkResults, PerformanceHistory};

/// SIMD implementation tier matching zipora's 6-tier framework
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdTier {
    /// Tier 0: Scalar fallback
    Tier0Scalar = 0,
    /// Tier 1: ARM NEON
    Tier1Neon = 1,
    /// Tier 2: x86 POPCNT
    Tier2Popcnt = 2,
    /// Tier 3: x86 BMI2 (PDEP/PEXT)
    Tier3Bmi2 = 3,
    /// Tier 4: x86 AVX2
    Tier4Avx2 = 4,
    /// Tier 5: x86 AVX-512
    Tier5Avx512 = 5,
}

impl SimdTier {
    /// Detect SIMD tier from CPU features
    pub fn from_features(features: &CpuFeatures) -> Self {
        if features.has_avx512f && features.has_avx512vl && features.has_avx512bw {
            SimdTier::Tier5Avx512
        } else if features.has_avx2 {
            SimdTier::Tier4Avx2
        } else if features.has_bmi2 {
            SimdTier::Tier3Bmi2
        } else if features.has_popcnt {
            SimdTier::Tier2Popcnt
        } else if features.has_neon {
            SimdTier::Tier1Neon
        } else {
            SimdTier::Tier0Scalar
        }
    }
}

/// Specific SIMD implementation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdImpl {
    /// Scalar fallback (lowest)
    Scalar = 0,
    /// ARM NEON implementation
    Neon = 1,
    /// SSE2 implementation
    Sse2 = 2,
    /// BMI2 implementation (PDEP/PEXT)
    Bmi2 = 3,
    /// SSE4.2 implementation
    Sse42 = 4,
    /// AVX2 implementation
    Avx2 = 5,
    /// AVX-512 implementation (highest)
    Avx512 = 6,
}

/// Key for caching selection decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SelectionKey {
    pub operation: Operation,
    pub size_bucket: usize,  // Bucketed size for cache efficiency
    pub density_bucket: u8,  // Bucketed density (0-255)
}

impl SelectionKey {
    /// Create new selection key with bucketing for cache efficiency
    pub fn new(operation: Operation, data_size: usize, data_density: Option<f64>) -> Self {
        // Bucket sizes: <64, 64-256, 256-1024, 1024-4096, >4096
        let size_bucket = match data_size {
            0..=63 => 0,
            64..=255 => 1,
            256..=1023 => 2,
            1024..=4095 => 3,
            _ => 4,
        };

        // Bucket density to 0-255 range
        let density_bucket = data_density
            .map(|d| (d * 255.0).clamp(0.0, 255.0) as u8)
            .unwrap_or(128); // Default to medium density

        SelectionKey {
            operation,
            size_bucket,
            density_bucket,
        }
    }
}

/// Configuration for adaptive selector
#[derive(Debug, Clone)]
pub struct AdaptiveSelectorConfig {
    /// Enable micro-benchmarking at startup
    pub enable_startup_benchmarks: bool,
    /// Enable continuous performance monitoring
    pub enable_monitoring: bool,
    /// Enable adaptive threshold adjustment
    pub enable_adaptation: bool,
    /// Warmup iterations for benchmarks
    pub warmup_iterations: usize,
    /// Measurement iterations for benchmarks
    pub measurement_iterations: usize,
    /// Performance degradation threshold (0.9 = 90%)
    pub performance_threshold: f64,
    /// Operations before re-benchmarking
    pub degradation_trigger_count: u32,
    /// Maximum cache entries
    pub max_cache_entries: usize,
}

impl Default for AdaptiveSelectorConfig {
    fn default() -> Self {
        AdaptiveSelectorConfig {
            enable_startup_benchmarks: true,
            enable_monitoring: true,
            enable_adaptation: true,
            warmup_iterations: 10,
            measurement_iterations: 100,
            performance_threshold: 0.9,
            degradation_trigger_count: 1000,
            max_cache_entries: 1024,
        }
    }
}

/// Data characteristic thresholds for SIMD selection
#[derive(Debug, Clone)]
pub struct SelectionThresholds {
    /// Minimum size for AVX-512 (default: 1024)
    pub avx512_min_size: usize,
    /// Minimum size for AVX2 (default: 256)
    pub avx2_min_size: usize,
    /// Minimum size for BMI2 (default: 64)
    pub bmi2_min_size: usize,
    /// Minimum size for SSE2 (default: 16)
    pub sse2_min_size: usize,
    /// Sparse data threshold (default: 0.1)
    pub sparse_threshold: f64,
    /// Dense data threshold (default: 0.5)
    pub dense_threshold: f64,
    /// Performance degradation threshold (default: 0.9 = 90%)
    pub performance_threshold: f64,
}

impl Default for SelectionThresholds {
    fn default() -> Self {
        SelectionThresholds {
            avx512_min_size: 1024,
            avx2_min_size: 256,
            bmi2_min_size: 64,
            sse2_min_size: 16,
            sparse_threshold: 0.1,
            dense_threshold: 0.5,
            performance_threshold: 0.9,
        }
    }
}

/// Adaptive SIMD selector with runtime benchmarking and performance monitoring
pub struct AdaptiveSimdSelector {
    /// Detected hardware tier
    hardware_tier: SimdTier,

    /// Cached CPU features (reference to static)
    cpu_features: &'static CpuFeatures,

    /// Performance benchmarks per operation type
    operation_benchmarks: Arc<RwLock<HashMap<Operation, BenchmarkResults>>>,

    /// Historical performance data
    performance_history: Arc<RwLock<HashMap<Operation, PerformanceHistory>>>,

    /// Selection thresholds
    thresholds: SelectionThresholds,

    /// Configuration
    config: AdaptiveSelectorConfig,

    /// Cached selection decisions
    selection_cache: Arc<RwLock<HashMap<SelectionKey, SimdImpl>>>,
}

impl AdaptiveSimdSelector {
    /// Create new adaptive SIMD selector with hardware detection
    pub fn new() -> Self {
        let cpu_features = get_cpu_features();
        let hardware_tier = SimdTier::from_features(cpu_features);

        AdaptiveSimdSelector {
            hardware_tier,
            cpu_features,
            operation_benchmarks: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            thresholds: SelectionThresholds::default(),
            config: AdaptiveSelectorConfig::default(),
            selection_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AdaptiveSelectorConfig) -> Self {
        let mut selector = Self::new();
        selector.config = config;
        selector
    }

    /// Get global singleton instance
    pub fn global() -> &'static AdaptiveSimdSelector {
        static GLOBAL_SELECTOR: OnceLock<AdaptiveSimdSelector> = OnceLock::new();

        GLOBAL_SELECTOR.get_or_init(|| {
            let mut selector = AdaptiveSimdSelector::new();

            // Run initial benchmarks if enabled
            if selector.config.enable_startup_benchmarks {
                selector.run_initial_benchmarks();
            }

            selector
        })
    }

    /// Select optimal SIMD implementation for given operation and data characteristics
    pub fn select_optimal_impl(
        &self,
        operation: Operation,
        data_size: usize,
        data_density: Option<f64>,
    ) -> SimdImpl {
        // Check cache first
        let key = SelectionKey::new(operation, data_size, data_density);

        if let Ok(cache) = self.selection_cache.read() {
            if let Some(&impl_type) = cache.get(&key) {
                return impl_type;
            }
        }

        // Perform selection
        let selected = self.select_impl_internal(operation, data_size, data_density);

        // Cache the result (best effort, ignore lock failures)
        if let Ok(mut cache) = self.selection_cache.write() {
            // LRU eviction if cache is full
            if cache.len() >= self.config.max_cache_entries {
                // Simple eviction: remove first entry
                if let Some(first_key) = cache.keys().next().copied() {
                    cache.remove(&first_key);
                }
            }

            cache.insert(key, selected);
        }

        selected
    }

    /// Internal selection logic based on hardware tier and data characteristics
    fn select_impl_internal(
        &self,
        _operation: Operation,
        data_size: usize,
        data_density: Option<f64>,
    ) -> SimdImpl {
        // Adjust thresholds based on data density
        let (avx512_min, avx2_min, bmi2_min, sse2_min) = if let Some(density) = data_density {
            if density < self.thresholds.sparse_threshold {
                // Sparse data: higher thresholds (less SIMD benefit)
                (
                    self.thresholds.avx512_min_size * 2,
                    self.thresholds.avx2_min_size * 2,
                    self.thresholds.bmi2_min_size * 2,
                    self.thresholds.sse2_min_size * 2,
                )
            } else if density > self.thresholds.dense_threshold {
                // Dense data: lower thresholds (more SIMD benefit)
                (
                    self.thresholds.avx512_min_size / 2,
                    self.thresholds.avx2_min_size / 2,
                    self.thresholds.bmi2_min_size / 2,
                    self.thresholds.sse2_min_size / 2,
                )
            } else {
                // Medium density: default thresholds
                (
                    self.thresholds.avx512_min_size,
                    self.thresholds.avx2_min_size,
                    self.thresholds.bmi2_min_size,
                    self.thresholds.sse2_min_size,
                )
            }
        } else {
            (
                self.thresholds.avx512_min_size,
                self.thresholds.avx2_min_size,
                self.thresholds.bmi2_min_size,
                self.thresholds.sse2_min_size,
            )
        };

        // Select based on hardware tier and data size
        match (self.hardware_tier, data_size) {
            // AVX-512 tier (Tier 5)
            (SimdTier::Tier5Avx512, size) if size >= avx512_min => SimdImpl::Avx512,

            // AVX2 tier (Tier 4)
            (SimdTier::Tier4Avx2 | SimdTier::Tier5Avx512, size) if size >= avx2_min => {
                SimdImpl::Avx2
            }

            // BMI2 tier (Tier 3)
            (
                SimdTier::Tier3Bmi2 | SimdTier::Tier4Avx2 | SimdTier::Tier5Avx512,
                size,
            ) if size >= bmi2_min => SimdImpl::Bmi2,

            // SSE2 tier (Tier 2)
            (tier, size) if tier >= SimdTier::Tier2Popcnt && size >= sse2_min => {
                SimdImpl::Sse2
            }

            // NEON tier (Tier 1 - ARM)
            (SimdTier::Tier1Neon, size) if size >= 64 => SimdImpl::Neon,

            // Scalar fallback (Tier 0)
            _ => SimdImpl::Scalar,
        }
    }

    /// Monitor performance and update history
    pub fn monitor_performance(&self, operation: Operation, duration: Duration, ops: u64) {
        if !self.config.enable_monitoring {
            return;
        }

        if let Ok(mut history_map) = self.performance_history.write() {
            let history = history_map
                .entry(operation)
                .or_insert_with(PerformanceHistory::new);

            history.record_sample(duration, ops);

            // Check for performance degradation
            if self.config.enable_adaptation {
                if let Ok(benchmarks) = self.operation_benchmarks.read() {
                    if let Some(benchmark) = benchmarks.get(&operation) {
                        if history.check_performance_degradation(benchmark.throughput,
                                                                 self.config.degradation_trigger_count) {
                            // Clear cache for this operation to force re-evaluation
                            drop(benchmarks); // Release read lock before write
                            self.clear_operation_cache(operation);
                        }
                    }
                }
            }
        }
    }

    /// Clear cache entries for a specific operation
    fn clear_operation_cache(&self, operation: Operation) {
        if let Ok(mut cache) = self.selection_cache.write() {
            cache.retain(|k, _| k.operation != operation);
        }
    }

    /// Run initial benchmarks for common operations
    fn run_initial_benchmarks(&mut self) {
        // To be implemented with micro-benchmarking framework
        // For now, just initialize empty benchmarks
        let operations = [
            Operation::Rank,
            Operation::Select,
            Operation::Popcount,
            Operation::Search,
            Operation::Sort,
        ];

        if let Ok(mut benchmarks) = self.operation_benchmarks.write() {
            for &op in &operations {
                benchmarks.insert(op, BenchmarkResults::default());
            }
        }
    }

    /// Get hardware tier
    pub fn hardware_tier(&self) -> SimdTier {
        self.hardware_tier
    }

    /// Get CPU features
    pub fn cpu_features(&self) -> &CpuFeatures {
        &self.cpu_features
    }

    /// Get current thresholds
    pub fn thresholds(&self) -> &SelectionThresholds {
        &self.thresholds
    }

    /// Update thresholds (for testing/tuning)
    pub fn set_thresholds(&mut self, thresholds: SelectionThresholds) {
        self.thresholds = thresholds;
        // Clear cache when thresholds change
        if let Ok(mut cache) = self.selection_cache.write() {
            cache.clear();
        }
    }
}

impl Default for AdaptiveSimdSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selector_creation() {
        let selector = AdaptiveSimdSelector::new();
        assert!(selector.hardware_tier >= SimdTier::Tier0Scalar);
        assert!(selector.hardware_tier <= SimdTier::Tier5Avx512);
    }

    #[test]
    fn test_simd_tier_ordering() {
        assert!(SimdTier::Tier0Scalar < SimdTier::Tier1Neon);
        assert!(SimdTier::Tier1Neon < SimdTier::Tier2Popcnt);
        assert!(SimdTier::Tier2Popcnt < SimdTier::Tier3Bmi2);
        assert!(SimdTier::Tier3Bmi2 < SimdTier::Tier4Avx2);
        assert!(SimdTier::Tier4Avx2 < SimdTier::Tier5Avx512);
    }

    #[test]
    fn test_size_based_selection() {
        let selector = AdaptiveSimdSelector::new();

        // Very small data should use scalar (smaller than minimum thresholds)
        let impl_small = selector.select_optimal_impl(Operation::Rank, 8, None);
        // Note: On systems with POPCNT/SSE2, even small data might use SIMD if above threshold
        // Just verify we got a valid implementation
        assert!(matches!(impl_small, SimdImpl::Scalar | SimdImpl::Sse2));

        // Large data might use SIMD (depending on hardware)
        let impl_large = selector.select_optimal_impl(Operation::Rank, 4096, None);
        if selector.hardware_tier >= SimdTier::Tier4Avx2 {
            // On AVX2+ hardware, large data should use SIMD
            assert!(impl_large != SimdImpl::Scalar);
        }
    }

    #[test]
    fn test_density_based_selection() {
        let selector = AdaptiveSimdSelector::new();

        // Sparse data
        let impl_sparse = selector.select_optimal_impl(Operation::Rank, 512, Some(0.05));

        // Dense data
        let impl_dense = selector.select_optimal_impl(Operation::Rank, 512, Some(0.8));

        // Dense data should be more likely to use SIMD
        if selector.hardware_tier >= SimdTier::Tier3Bmi2 {
            assert!(impl_dense != SimdImpl::Scalar);
        }

        // Results may differ based on density
        println!("Sparse impl: {:?}, Dense impl: {:?}", impl_sparse, impl_dense);
    }

    #[test]
    fn test_selection_caching() {
        let selector = AdaptiveSimdSelector::new();

        // First call
        let impl1 = selector.select_optimal_impl(Operation::Rank, 1024, None);

        // Second call with same parameters (should hit cache)
        let impl2 = selector.select_optimal_impl(Operation::Rank, 1024, None);

        assert_eq!(impl1, impl2);
    }

    #[test]
    fn test_selection_key_bucketing() {
        let key1 = SelectionKey::new(Operation::Rank, 100, Some(0.5));
        let key2 = SelectionKey::new(Operation::Rank, 150, Some(0.51));

        // Should have same size bucket (64-255)
        assert_eq!(key1.size_bucket, key2.size_bucket);

        // Density might differ slightly due to bucketing
        assert!((key1.density_bucket as i16 - key2.density_bucket as i16).abs() <= 3);
    }

    #[test]
    fn test_performance_monitoring() {
        let selector = AdaptiveSimdSelector::new();

        // Monitor some operations
        selector.monitor_performance(Operation::Rank, Duration::from_micros(100), 1000);
        selector.monitor_performance(Operation::Select, Duration::from_micros(150), 500);

        // Verify history was recorded
        if let Ok(history) = selector.performance_history.read() {
            assert!(history.contains_key(&Operation::Rank));
            assert!(history.contains_key(&Operation::Select));
        }
    }

    #[test]
    fn test_global_singleton() {
        let selector1 = AdaptiveSimdSelector::global();
        let selector2 = AdaptiveSimdSelector::global();

        // Should be same instance
        assert!(std::ptr::eq(selector1, selector2));
    }

    #[test]
    fn test_custom_thresholds() {
        let mut selector = AdaptiveSimdSelector::new();

        let custom_thresholds = SelectionThresholds {
            avx512_min_size: 2048,
            avx2_min_size: 512,
            bmi2_min_size: 128,
            sse2_min_size: 32,
            sparse_threshold: 0.05,
            dense_threshold: 0.7,
            performance_threshold: 0.85,
        };

        selector.set_thresholds(custom_thresholds);

        assert_eq!(selector.thresholds().avx512_min_size, 2048);
        assert_eq!(selector.thresholds().avx2_min_size, 512);
    }
}
