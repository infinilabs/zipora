//! Statistics and Monitoring - consolidated minimal module.
//!
//! Previously 5,275 LOC across 7 sub-modules. Collapsed to type stubs that
//! preserve the public API for downstream compatibility. None of these types
//! are used by the core library; they exist only as exported API surface.
//!
//! For actual profiling/timing, use `dev_infrastructure::debug` (ScopedTimer,
//! HighPrecisionTimer, BenchmarkSuite).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use crate::error::ZiporaError;

// ============================================================================
// Core types (from mod.rs)
// ============================================================================

/// Universal memory size interface
pub trait MemorySize {
    fn mem_size(&self) -> usize;
    fn detailed_mem_size(&self) -> MemoryBreakdown {
        MemoryBreakdown { total: self.mem_size(), components: HashMap::new() }
    }
}

/// Simple statistics matching topling-zip's ZipStat
#[derive(Debug, Clone)]
pub struct TrieStat {
    pub insert_time: f64,
    pub lookup_time: f64,
    pub build_time: f64,
    pub total_bytes: u64,
}

impl TrieStat {
    pub fn new() -> Self {
        Self { insert_time: 0.0, lookup_time: 0.0, build_time: 0.0, total_bytes: 0 }
    }
}

/// Composite statistics
#[derive(Debug)]
pub struct TrieStatistics {
    pub memory: MemoryStats,
    pub performance: PerformanceStats,
    pub compression: CompressionStats,
    pub distribution: DistributionStats,
    pub errors: ErrorStats,
    pub timing: TimingStats,
}

impl Default for TrieStatistics { fn default() -> Self { Self::new() } }

impl TrieStatistics {
    pub fn new() -> Self {
        Self {
            memory: MemoryStats::new(), performance: PerformanceStats::new(),
            compression: CompressionStats::new(), distribution: DistributionStats::new(),
            errors: ErrorStats::new(), timing: TimingStats::new(),
        }
    }
    pub fn merge(&mut self, _other: &TrieStatistics) {}
    pub fn reset(&mut self) {}
    pub fn generate_report(&self) -> String { String::new() }
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryCategory { Nodes, Cache, Overhead }

#[derive(Debug, Clone, Copy)]
pub enum ErrorType { Memory, Io, Corruption, Timeout, Other }

// ============================================================================
// MemoryStats
// ============================================================================

#[derive(Debug)]
pub struct MemoryStats {
    pub total_allocated: AtomicUsize,
    pub nodes_memory: AtomicUsize,
    pub cache_memory: AtomicUsize,
    pub overhead_memory: AtomicUsize,
    pub peak_memory: AtomicUsize,
    pub allocation_count: AtomicU64,
    pub deallocation_count: AtomicU64,
}

impl MemoryStats {
    pub fn new() -> Self {
        Self {
            total_allocated: AtomicUsize::new(0), nodes_memory: AtomicUsize::new(0),
            cache_memory: AtomicUsize::new(0), overhead_memory: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0), allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
        }
    }
    pub fn record_allocation(&self, size: usize, _cat: MemoryCategory) {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
    }
    pub fn record_deallocation(&self, size: usize, _cat: MemoryCategory) {
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
        self.total_allocated.fetch_sub(size, Ordering::Relaxed);
    }
    pub fn merge(&mut self, _other: &MemoryStats) {}
    pub fn reset(&mut self) {}
    pub fn report(&self) -> String { String::new() }
}

// ============================================================================
// PerformanceStats
// ============================================================================

#[derive(Debug)]
pub struct PerformanceStats {
    pub insert_count: AtomicU64,
    pub lookup_count: AtomicU64,
    pub delete_count: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub total_operations: AtomicU64,
    pub failed_operations: AtomicU64,
    pub average_operation_time_ns: AtomicU64,
}

impl PerformanceStats {
    pub fn new() -> Self {
        Self {
            insert_count: AtomicU64::new(0), lookup_count: AtomicU64::new(0),
            delete_count: AtomicU64::new(0), cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0), total_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0), average_operation_time_ns: AtomicU64::new(0),
        }
    }
    pub fn record_insert(&self) { self.insert_count.fetch_add(1, Ordering::Relaxed); }
    pub fn record_lookup(&self, _hit: bool) { self.lookup_count.fetch_add(1, Ordering::Relaxed); }
    pub fn record_delete(&self) { self.delete_count.fetch_add(1, Ordering::Relaxed); }
    pub fn merge(&mut self, _other: &PerformanceStats) {}
    pub fn reset(&mut self) {}
    pub fn report(&self) -> String { String::new() }
}

// ============================================================================
// CompressionStats, DistributionStats, ErrorStats, TimingStats
// ============================================================================

#[derive(Debug)]
pub struct CompressionStats {
    pub original_size: AtomicUsize,
    pub compressed_size: AtomicUsize,
}

impl CompressionStats {
    pub fn new() -> Self {
        Self { original_size: AtomicUsize::new(0), compressed_size: AtomicUsize::new(0) }
    }
    pub fn merge(&mut self, _other: &CompressionStats) {}
    pub fn reset(&mut self) {}
    pub fn report(&self) -> String { String::new() }
}

#[derive(Debug)]
pub struct DistributionStats { pub total_samples: AtomicU64 }

impl DistributionStats {
    pub fn new() -> Self { Self { total_samples: AtomicU64::new(0) } }
    pub fn merge(&mut self, _other: &DistributionStats) {}
    pub fn reset(&mut self) {}
    pub fn report(&self) -> String { String::new() }
}

#[derive(Debug)]
pub struct ErrorStats { pub total_errors: AtomicU64 }

impl ErrorStats {
    pub fn new() -> Self { Self { total_errors: AtomicU64::new(0) } }
    pub fn record_error(&self, _et: ErrorType) { self.total_errors.fetch_add(1, Ordering::Relaxed); }
    pub fn merge(&mut self, _other: &ErrorStats) {}
    pub fn reset(&mut self) {}
    pub fn report(&self) -> String { String::new() }
}

#[derive(Debug)]
pub struct TimingStats { pub creation_time: Instant }

impl TimingStats {
    pub fn new() -> Self { Self { creation_time: Instant::now() } }
    pub fn uptime(&self) -> Duration { self.creation_time.elapsed() }
    pub fn merge(&mut self, _other: &TimingStats) {}
    pub fn reset(&mut self) {}
    pub fn report(&self) -> String { String::new() }
}

// ============================================================================
// Memory tracking stubs (from memory_tracking.rs)
// ============================================================================

#[derive(Debug, Clone)]
pub struct MemoryBreakdown { pub total: usize, pub components: HashMap<String, usize> }

impl MemoryBreakdown {
    pub fn new() -> Self { Self { total: 0, components: HashMap::new() } }
    pub fn add_component(&mut self, name: &str, size: usize) {
        self.total += size;
        self.components.insert(name.to_string(), size);
    }
}

pub struct GlobalMemoryTracker;
impl GlobalMemoryTracker { pub fn new() -> Self { Self } }

pub struct TrackedObject;
pub struct LocalMemoryTracker;
impl LocalMemoryTracker { pub fn new() -> Self { Self } }

#[derive(Debug, Clone)]
pub struct FragmentationAnalysis;

// ============================================================================
// Timing stubs (from timing.rs)
// ============================================================================

pub struct Profiling;
pub type QTime = Instant;
pub type QDuration = Duration;

pub struct PerfTimer { start: Instant }
impl PerfTimer {
    pub fn new() -> Self { Self { start: Instant::now() } }
    pub fn elapsed(&self) -> Duration { self.start.elapsed() }
}

pub struct TimerCollection;
impl TimerCollection { pub fn new() -> Self { Self } }

pub struct TimerStats;

pub struct ScopedTimer;
impl ScopedTimer { pub fn new(_name: &str) -> Self { Self } }

pub fn str_date_time_now() -> String {
    format!("{:?}", std::time::SystemTime::now())
}

// ============================================================================
// Histogram stubs (from histogram.rs)
// ============================================================================

#[derive(Debug, Clone)]
pub struct FreqHist { counts: Vec<u64> }
impl FreqHist {
    pub fn new() -> Self { Self { counts: vec![0u64; 256] } }
    pub fn add(&mut self, byte: u8) { self.counts[byte as usize] += 1; }
}

pub type FreqHistO1 = FreqHist;
pub type FreqHistO2 = FreqHist;
pub type HistogramData = FreqHist;
pub type HistogramDataO1 = FreqHist;
pub type HistogramDataO2 = FreqHist;

pub struct HistogramCollection;
impl HistogramCollection { pub fn new() -> Self { Self } }

pub struct GlobalHistogramStats;

// ============================================================================
// Entropy analysis stubs (from entropy_analysis.rs)
// ============================================================================

pub struct EntropyAnalyzer;
impl EntropyAnalyzer { pub fn new() -> Self { Self } }

#[derive(Debug, Clone)]
pub struct EntropyConfig;
impl Default for EntropyConfig { fn default() -> Self { Self } }

#[derive(Debug, Clone)]
pub struct EntropyResults;

#[derive(Debug, Clone)]
pub struct CompressionEstimates;

#[derive(Debug, Clone)]
pub struct DistributionInfo;

#[derive(Debug, Clone)]
pub struct SampleStats;

pub struct EntropyAnalyzerCollection;
pub struct GlobalEntropyStats;

// ============================================================================
// Buffer management stubs (from buffer_management.rs)
// ============================================================================

pub struct ContextBuffer { data: Vec<u8> }
impl ContextBuffer {
    pub fn new(cap: usize) -> Self { Self { data: Vec::with_capacity(cap) } }
}

#[derive(Debug, Clone)]
pub struct BufferMetadata;

#[derive(Debug, Clone, Copy)]
pub enum BufferPriority { Low, Normal, High, Critical }

pub trait StatisticsContext: Send + Sync {}

pub struct DefaultStatisticsContext;
impl StatisticsContext for DefaultStatisticsContext {}

pub struct BufferPoolManager;
impl BufferPoolManager { pub fn new(_config: BufferPoolConfig) -> Self { Self } }

#[derive(Debug, Clone)]
pub struct BufferPoolConfig;
impl Default for BufferPoolConfig { fn default() -> Self { Self } }

pub struct PoolStatistics;
pub struct ScopedBuffer;

// ============================================================================
// Profiling stubs (from profiling.rs — not dev_infrastructure profiling)
// ============================================================================

#[derive(Clone)]
pub struct Profiler;
impl Default for Profiler { fn default() -> Self { Self } }
impl Profiler {
    pub fn new(_config: ProfilerConfig) -> Self { Self }
}

#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    pub enabled: bool,
    pub sample_rate: f64,
}
impl Default for ProfilerConfig {
    fn default() -> Self { Self { enabled: false, sample_rate: 1.0 } }
}

pub struct OperationProfile;
pub struct GlobalProfilingStats;

pub struct ProfiledOperation;

static GLOBAL_PROFILER: std::sync::OnceLock<Profiler> = std::sync::OnceLock::new();

pub fn global_profiler() -> &'static Profiler {
    GLOBAL_PROFILER.get_or_init(Profiler::default)
}

pub fn init_global_profiler(config: ProfilerConfig) -> Result<(), ZiporaError> {
    GLOBAL_PROFILER.set(Profiler::new(config))
        .map_err(|_| ZiporaError::invalid_data("global profiler already initialized"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_statistics_creation() {
        let stats = TrieStatistics::new();
        assert_eq!(stats.memory.total_allocated.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_memory_breakdown() {
        let mut breakdown = MemoryBreakdown::new();
        breakdown.add_component("nodes", 1000);
        breakdown.add_component("cache", 500);
        assert_eq!(breakdown.total, 1500);
        assert_eq!(breakdown.components.len(), 2);
    }

    #[test]
    fn test_freq_hist() {
        let mut hist = FreqHist::new();
        hist.add(65); // 'A'
        hist.add(65);
        assert_eq!(hist.counts[65], 2);
    }
}
