
/// Configuration for radix sort
#[derive(Debug, Clone)]
pub struct RadixSortConfig {
    /// Use parallel processing for large datasets
    pub use_parallel: bool,
    /// Threshold for switching to parallel processing
    pub parallel_threshold: usize,
    /// Radix size (typically 8 or 16 bits)
    pub radix_bits: usize,
    /// Use counting sort for small datasets
    pub use_counting_sort_threshold: usize,
    /// Enable SIMD optimizations when available
    pub use_simd: bool,
}

impl Default for RadixSortConfig {
    fn default() -> Self {
        Self {
            use_parallel: true,
            parallel_threshold: 10_000,
            radix_bits: 8,
            use_counting_sort_threshold: 256,
            use_simd: cfg!(feature = "simd"),
        }
    }
}

/// Runtime CPU feature detection for optimal SIMD usage
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub bmi2: bool,
    pub popcnt: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
}

impl CpuFeatures {
    /// Detect available CPU features at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                bmi2: std::arch::is_x86_feature_detected!("bmi2"),
                popcnt: std::arch::is_x86_feature_detected!("popcnt"),
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx2: false,
                bmi2: false,
                popcnt: false,
                avx512f: false,
                avx512bw: false,
            }
        }
    }

    /// Check if advanced SIMD optimizations are available
    pub fn has_advanced_simd(&self) -> bool {
        self.avx2 && self.bmi2
    }

    /// Check if AVX-512 optimizations are available
    pub fn has_avx512(&self) -> bool {
        self.avx512f && self.avx512bw
    }
}

/// Sorting algorithm strategy for adaptive selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortingStrategy {
    /// Insertion sort for small datasets
    Insertion,
    /// Tim sort for nearly sorted data
    TimSort,
    /// LSD radix sort for random integer data
    LsdRadix,
    /// MSD radix sort for string data
    MsdRadix,
    /// Hybrid approach with intelligent switching
    Adaptive,
}

/// Data characteristics for adaptive strategy selection
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    pub size: usize,
    pub is_nearly_sorted: bool,
    pub is_string_data: bool,
    pub estimated_entropy: f64,
    pub max_key_bits: usize,
}

impl DataCharacteristics {
    /// Analyze integer data characteristics
    pub fn analyze_integers<T>(data: &[T]) -> Self
    where
        T: Copy + Into<u64> + Ord,
    {
        let size = data.len();

        // Check if data is nearly sorted
        let mut inversions = 0usize;
        let mut sorted_runs = 0usize;
        let mut current_run_length = 1usize;

        for i in 1..data.len() {
            if data[i] >= data[i - 1] {
                current_run_length += 1;
            } else {
                inversions += 1;
                if current_run_length > 1 {
                    sorted_runs += 1;
                }
                current_run_length = 1;
            }
        }

        if current_run_length > 1 {
            sorted_runs += 1;
        }

        // Be more strict about what constitutes "nearly sorted"
        // For small datasets, require very few inversions AND good run structure
        let inversion_threshold = if size <= 10 {
            std::cmp::max(1, size / 5) // For small datasets, allow at most 20% inversions
        } else {
            size / 10 // For larger datasets, allow up to 10% inversions
        };

        // A single long run (for sorted data) OR multiple good runs should qualify
        let is_nearly_sorted =
            inversions < inversion_threshold && (sorted_runs >= 1 || inversions == 0);

        // Estimate entropy and max key bits
        let mut max_val = 0u64;
        for &item in data {
            max_val = max_val.max(item.into());
        }

        let max_key_bits = if max_val == 0 {
            1
        } else {
            64 - max_val.leading_zeros() as usize
        };
        let estimated_entropy = if size > 0 { (size as f64).log2() } else { 0.0 };

        Self {
            size,
            is_nearly_sorted,
            is_string_data: false,
            estimated_entropy,
            max_key_bits,
        }
    }

    /// Analyze string data characteristics
    pub fn analyze_strings(data: &[Vec<u8>]) -> Self {
        let size = data.len();

        // Check if strings are nearly sorted
        let mut inversions = 0usize;
        for i in 1..data.len() {
            if data[i] < data[i - 1] {
                inversions += 1;
            }
        }

        let is_nearly_sorted = inversions < std::cmp::max(1, size / 10);

        // Estimate maximum string length for optimization decisions
        let max_length = data.iter().map(|s| s.len()).max().unwrap_or(0);
        let estimated_entropy = if size > 0 { (size as f64).log2() } else { 0.0 };

        Self {
            size,
            is_nearly_sorted,
            is_string_data: true,
            estimated_entropy,
            max_key_bits: max_length * 8, // Rough estimate based on string length
        }
    }
}

/// Advanced configuration for sophisticated radix sort variants
#[derive(Debug, Clone)]
pub struct AdvancedRadixSortConfig {
    /// Whether to use secure memory pool for allocations
    pub use_secure_memory: bool,
    /// Enable adaptive strategy selection based on data characteristics
    pub adaptive_strategy: bool,
    /// Force a specific sorting strategy (overrides adaptive selection)
    pub force_strategy: Option<SortingStrategy>,
    /// Use parallel processing for large datasets
    pub use_parallel: bool,
    /// Threshold for switching to parallel processing
    pub parallel_threshold: usize,
    /// Number of worker threads for parallel processing (0 = auto-detect)
    pub num_threads: usize,
    /// Radix size for LSD radix sort (typically 8 or 16 bits)
    pub radix_bits: usize,
    /// Threshold for switching to insertion sort for small datasets
    pub insertion_sort_threshold: usize,
    /// Threshold for using counting sort for small ranges
    pub counting_sort_threshold: usize,
    /// Enable enhanced SIMD optimizations when available
    pub use_simd: bool,
    /// Enable work-stealing for better load balancing
    pub use_work_stealing: bool,
    /// Memory prefetching distance for cache optimization
    pub prefetch_distance: usize,
    /// Cache alignment requirement for optimal performance
    pub cache_alignment: usize,
    /// Maximum memory budget for temporary allocations (bytes)
    pub memory_budget: usize,
    /// Enable detailed performance monitoring
    pub enable_profiling: bool,
}

impl Default for AdvancedRadixSortConfig {
    fn default() -> Self {
        Self {
            use_secure_memory: true,
            adaptive_strategy: true,
            force_strategy: None,
            use_parallel: true,
            parallel_threshold: 10_000,
            num_threads: 0, // Auto-detect
            radix_bits: 8,
            insertion_sort_threshold: 100,
            counting_sort_threshold: 1024,
            use_simd: cfg!(feature = "simd"),
            use_work_stealing: true,
            prefetch_distance: 2,
            cache_alignment: 64,
            memory_budget: 64 * 1024 * 1024, // 64MB
            enable_profiling: false,
        }
    }
}

/// Enhanced performance statistics with detailed metrics
#[derive(Debug, Clone)]
pub struct AdvancedAlgorithmStats {
    /// Basic algorithm statistics
    pub basic_stats: crate::algorithms::AlgorithmStats,
    /// Strategy that was actually used
    pub strategy_used: SortingStrategy,
    /// CPU features that were utilized
    pub cpu_features_used: CpuFeatures,
    /// Number of cache misses (estimated)
    pub estimated_cache_misses: u64,
    /// Peak memory usage during sorting
    pub peak_memory_bytes: usize,
    /// Number of worker threads used
    pub threads_used: usize,
    /// Time spent in different phases (us)
    pub phase_times: PhaseTimes,
}

/// Detailed timing information for different sorting phases
#[derive(Debug, Clone)]
pub struct PhaseTimes {
    pub analysis_time_us: u64,
    pub allocation_time_us: u64,
    pub sorting_time_us: u64,
    pub merging_time_us: u64,
    pub cleanup_time_us: u64,
}
