//! Statistics and Monitoring - consolidated lightweight telemetry module.
//!
//! Provides atomic memory, performance, compression, distribution, error, and
//! timing accumulators alongside byte-frequency histograms and memory breakdown
//! helpers.
//!
//! For hardware-assisted profiling/timing utilities, see `system::profiling`
//! (`PerfTimer`, `HighPrecisionTimer`, `BenchmarkSuite`).

use crate::error::ZiporaError;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[inline]
fn atomic_saturating_sub(target: &AtomicUsize, delta: usize) {
    // ORDERING: Relaxed is sufficient for independent telemetry counters.
    let _ = target.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
        Some(cur.saturating_sub(delta))
    });
}

// ============================================================================
// Core types
// ============================================================================
/// Simple statistics
#[derive(Debug, Clone)]
pub struct TrieStat {
    pub insert_time: f64,
    pub lookup_time: f64,
    pub build_time: f64,
    pub total_bytes: u64,
}

impl Default for TrieStat {
    fn default() -> Self {
        Self::new()
    }
}

impl TrieStat {
    pub fn new() -> Self {
        Self {
            insert_time: 0.0,
            lookup_time: 0.0,
            build_time: 0.0,
            total_bytes: 0,
        }
    }
}

/// Composite statistics
#[derive(Debug)]
pub struct TrieStatistics {
    pub memory: StatsMemoryStats,
    pub performance: StatsPerformanceStats,
    pub compression: StatsCompressionStats,
    pub distribution: DistributionStats,
    pub errors: ErrorStats,
    pub timing: TimingStats,
}

impl Default for TrieStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl TrieStatistics {
    pub fn new() -> Self {
        Self {
            memory: StatsMemoryStats::new(),
            performance: StatsPerformanceStats::new(),
            compression: StatsCompressionStats::new(),
            distribution: DistributionStats::new(),
            errors: ErrorStats::new(),
            timing: TimingStats::new(),
        }
    }
    pub fn merge(&mut self, other: &TrieStatistics) {
        self.memory.merge(&other.memory);
        self.performance.merge(&other.performance);
        self.compression.merge(&other.compression);
        self.distribution.merge(&other.distribution);
        self.errors.merge(&other.errors);
        self.timing.merge(&other.timing);
    }
    pub fn reset(&mut self) {
        self.memory.reset();
        self.performance.reset();
        self.compression.reset();
        self.distribution.reset();
        self.errors.reset();
        self.timing.reset();
    }
    pub fn generate_report(&self) -> String {
        format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            self.memory.report(),
            self.performance.report(),
            self.compression.report(),
            self.distribution.report(),
            self.errors.report(),
            self.timing.report()
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryCategory {
    Nodes,
    Cache,
    Overhead,
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorType {
    Memory,
    Io,
    Corruption,
    Timeout,
    Other,
}

// ============================================================================
// StatsMemoryStats
// ============================================================================

#[derive(Debug)]
pub struct StatsMemoryStats {
    pub total_allocated: AtomicUsize,
    pub nodes_memory: AtomicUsize,
    pub cache_memory: AtomicUsize,
    pub overhead_memory: AtomicUsize,
    pub peak_memory: AtomicUsize,
    pub allocation_count: AtomicU64,
    pub deallocation_count: AtomicU64,
}

impl Default for StatsMemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsMemoryStats {
    pub fn new() -> Self {
        Self {
            total_allocated: AtomicUsize::new(0),
            nodes_memory: AtomicUsize::new(0),
            cache_memory: AtomicUsize::new(0),
            overhead_memory: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
        }
    }
    pub fn record_allocation(&self, size: usize, cat: MemoryCategory) {
        // ORDERING: Relaxed is sufficient for independent telemetry counters.
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        match cat {
            MemoryCategory::Nodes => {
                self.nodes_memory.fetch_add(size, Ordering::Relaxed);
            }
            MemoryCategory::Cache => {
                self.cache_memory.fetch_add(size, Ordering::Relaxed);
            }
            MemoryCategory::Overhead => {
                self.overhead_memory.fetch_add(size, Ordering::Relaxed);
            }
        }
        let new_total = self
            .total_allocated
            .fetch_add(size, Ordering::Relaxed)
            .saturating_add(size);
        self.peak_memory.fetch_max(new_total, Ordering::Relaxed);
    }
    pub fn record_deallocation(&self, size: usize, cat: MemoryCategory) {
        // ORDERING: Relaxed is sufficient for independent telemetry counters.
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
        match cat {
            MemoryCategory::Nodes => atomic_saturating_sub(&self.nodes_memory, size),
            MemoryCategory::Cache => atomic_saturating_sub(&self.cache_memory, size),
            MemoryCategory::Overhead => atomic_saturating_sub(&self.overhead_memory, size),
        }
        atomic_saturating_sub(&self.total_allocated, size);
    }
    pub fn merge(&mut self, other: &StatsMemoryStats) {
        // ORDERING: Relaxed is sufficient for telemetry aggregation.
        let other_total = other.total_allocated.load(Ordering::Relaxed);
        let new_total = self
            .total_allocated
            .fetch_add(other_total, Ordering::Relaxed)
            .saturating_add(other_total);
        self.nodes_memory.fetch_add(
            other.nodes_memory.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.cache_memory.fetch_add(
            other.cache_memory.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.overhead_memory.fetch_add(
            other.overhead_memory.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        let other_peak = other.peak_memory.load(Ordering::Relaxed).max(new_total);
        self.peak_memory.fetch_max(other_peak, Ordering::Relaxed);
        self.allocation_count.fetch_add(
            other.allocation_count.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.deallocation_count.fetch_add(
            other.deallocation_count.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
    pub fn reset(&mut self) {
        // ORDERING: Relaxed is sufficient for resetting telemetry counters.
        self.total_allocated.store(0, Ordering::Relaxed);
        self.nodes_memory.store(0, Ordering::Relaxed);
        self.cache_memory.store(0, Ordering::Relaxed);
        self.overhead_memory.store(0, Ordering::Relaxed);
        self.peak_memory.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.deallocation_count.store(0, Ordering::Relaxed);
    }
    pub fn report(&self) -> String {
        // ORDERING: Relaxed is sufficient for point-in-time telemetry formatting.
        format!(
            "Memory: total={}B (nodes={}B, cache={}B, overhead={}B), peak={}B, allocs={}, deallocs={}",
            self.total_allocated.load(Ordering::Relaxed),
            self.nodes_memory.load(Ordering::Relaxed),
            self.cache_memory.load(Ordering::Relaxed),
            self.overhead_memory.load(Ordering::Relaxed),
            self.peak_memory.load(Ordering::Relaxed),
            self.allocation_count.load(Ordering::Relaxed),
            self.deallocation_count.load(Ordering::Relaxed),
        )
    }
}

// ============================================================================
// StatsPerformanceStats
// ============================================================================

#[derive(Debug)]
pub struct StatsPerformanceStats {
    pub insert_count: AtomicU64,
    pub lookup_count: AtomicU64,
    pub delete_count: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub total_operations: AtomicU64,
    pub failed_operations: AtomicU64,
    pub average_operation_time_ns: AtomicU64,
}

impl Default for StatsPerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsPerformanceStats {
    pub fn new() -> Self {
        Self {
            insert_count: AtomicU64::new(0),
            lookup_count: AtomicU64::new(0),
            delete_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            total_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            average_operation_time_ns: AtomicU64::new(0),
        }
    }
    pub fn record_insert(&self) {
        // ORDERING: Relaxed is sufficient for independent telemetry counters.
        self.insert_count.fetch_add(1, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_lookup(&self, hit: bool) {
        // ORDERING: Relaxed is sufficient for independent telemetry counters.
        self.lookup_count.fetch_add(1, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        if hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn record_delete(&self) {
        // ORDERING: Relaxed is sufficient for independent telemetry counters.
        self.delete_count.fetch_add(1, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);
    }
    pub fn merge(&mut self, other: &StatsPerformanceStats) {
        // ORDERING: Relaxed is sufficient for telemetry aggregation.
        self.insert_count.fetch_add(
            other.insert_count.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.lookup_count.fetch_add(
            other.lookup_count.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.delete_count.fetch_add(
            other.delete_count.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.cache_hits
            .fetch_add(other.cache_hits.load(Ordering::Relaxed), Ordering::Relaxed);
        self.cache_misses.fetch_add(
            other.cache_misses.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.total_operations.fetch_add(
            other.total_operations.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.failed_operations.fetch_add(
            other.failed_operations.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
    pub fn reset(&mut self) {
        // ORDERING: Relaxed is sufficient for resetting telemetry counters.
        self.insert_count.store(0, Ordering::Relaxed);
        self.lookup_count.store(0, Ordering::Relaxed);
        self.delete_count.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.total_operations.store(0, Ordering::Relaxed);
        self.failed_operations.store(0, Ordering::Relaxed);
        self.average_operation_time_ns.store(0, Ordering::Relaxed);
    }
    pub fn report(&self) -> String {
        // ORDERING: Relaxed is sufficient for point-in-time telemetry formatting.
        format!(
            "Performance: ops={} (inserts={}, lookups={}, deletes={}, hits={}, misses={}, failed={})",
            self.total_operations.load(Ordering::Relaxed),
            self.insert_count.load(Ordering::Relaxed),
            self.lookup_count.load(Ordering::Relaxed),
            self.delete_count.load(Ordering::Relaxed),
            self.cache_hits.load(Ordering::Relaxed),
            self.cache_misses.load(Ordering::Relaxed),
            self.failed_operations.load(Ordering::Relaxed),
        )
    }
}

// ============================================================================
// StatsCompressionStats, DistributionStats, ErrorStats, TimingStats
// ============================================================================

#[derive(Debug)]
pub struct StatsCompressionStats {
    pub original_size: AtomicUsize,
    pub compressed_size: AtomicUsize,
}

impl Default for StatsCompressionStats {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsCompressionStats {
    pub fn new() -> Self {
        Self {
            original_size: AtomicUsize::new(0),
            compressed_size: AtomicUsize::new(0),
        }
    }
    pub fn merge(&mut self, other: &StatsCompressionStats) {
        // ORDERING: Relaxed is sufficient for telemetry aggregation.
        self.original_size.fetch_add(
            other.original_size.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.compressed_size.fetch_add(
            other.compressed_size.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
    pub fn reset(&mut self) {
        // ORDERING: Relaxed is sufficient for resetting telemetry counters.
        self.original_size.store(0, Ordering::Relaxed);
        self.compressed_size.store(0, Ordering::Relaxed);
    }
    pub fn report(&self) -> String {
        // ORDERING: Relaxed is sufficient for point-in-time telemetry formatting.
        format!(
            "Compression: original={}B, compressed={}B",
            self.original_size.load(Ordering::Relaxed),
            self.compressed_size.load(Ordering::Relaxed),
        )
    }
}

#[derive(Debug)]
pub struct DistributionStats {
    pub total_samples: AtomicU64,
}

impl Default for DistributionStats {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributionStats {
    pub fn new() -> Self {
        Self {
            total_samples: AtomicU64::new(0),
        }
    }
    pub fn merge(&mut self, other: &DistributionStats) {
        // ORDERING: Relaxed is sufficient for telemetry aggregation.
        self.total_samples.fetch_add(
            other.total_samples.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
    pub fn reset(&mut self) {
        // ORDERING: Relaxed is sufficient for resetting telemetry counters.
        self.total_samples.store(0, Ordering::Relaxed);
    }
    pub fn report(&self) -> String {
        // ORDERING: Relaxed is sufficient for point-in-time telemetry formatting.
        format!(
            "Distribution: samples={}",
            self.total_samples.load(Ordering::Relaxed)
        )
    }
}

#[derive(Debug)]
pub struct ErrorStats {
    pub total_errors: AtomicU64,
}

impl Default for ErrorStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorStats {
    pub fn new() -> Self {
        Self {
            total_errors: AtomicU64::new(0),
        }
    }
    pub fn record_error(&self, _et: ErrorType) {
        // ORDERING: Relaxed is sufficient for independent telemetry counters.
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }
    pub fn merge(&mut self, other: &ErrorStats) {
        // ORDERING: Relaxed is sufficient for telemetry aggregation.
        self.total_errors.fetch_add(
            other.total_errors.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
    pub fn reset(&mut self) {
        // ORDERING: Relaxed is sufficient for resetting telemetry counters.
        self.total_errors.store(0, Ordering::Relaxed);
    }
    pub fn report(&self) -> String {
        // ORDERING: Relaxed is sufficient for point-in-time telemetry formatting.
        format!("Errors: total={}", self.total_errors.load(Ordering::Relaxed))
    }
}

#[derive(Debug)]
pub struct TimingStats {
    pub creation_time: Instant,
}

impl Default for TimingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingStats {
    pub fn new() -> Self {
        Self {
            creation_time: Instant::now(),
        }
    }
    pub fn uptime(&self) -> Duration {
        self.creation_time.elapsed()
    }
    pub fn merge(&mut self, other: &TimingStats) {
        if other.creation_time < self.creation_time {
            self.creation_time = other.creation_time;
        }
    }
    pub fn reset(&mut self) {
        self.creation_time = Instant::now();
    }
    pub fn report(&self) -> String {
        format!("Timing: uptime={:.6}s", self.uptime().as_secs_f64())
    }
}

// ============================================================================
// Memory tracking helpers
// ============================================================================

#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    pub total: usize,
    pub components: HashMap<String, usize>,
}

impl Default for MemoryBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBreakdown {
    pub fn new() -> Self {
        Self {
            total: 0,
            components: HashMap::new(),
        }
    }
    pub fn add_component(&mut self, name: &str, size: usize) {
        if let Some(prev) = self.components.insert(name.to_string(), size) {
            self.total = self.total.saturating_sub(prev);
        }
        self.total = self.total.saturating_add(size);
    }
}

pub struct GlobalMemoryTracker;
impl Default for GlobalMemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalMemoryTracker {
    pub fn new() -> Self {
        Self
    }
}

pub struct TrackedObject;
pub struct LocalMemoryTracker;
impl Default for LocalMemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalMemoryTracker {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct FragmentationAnalysis;

// ============================================================================
// Timing helpers
// ============================================================================

pub struct Profiling;
pub type QTime = Instant;
pub type QDuration = Duration;

pub struct StatsPerfTimer {
    start: Instant,
}
impl Default for StatsPerfTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsPerfTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

pub struct TimerCollection;
impl Default for TimerCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl TimerCollection {
    pub fn new() -> Self {
        Self
    }
}

pub struct StatsScopedTimer;
impl StatsScopedTimer {
    pub fn new(_name: &str) -> Self {
        Self
    }
}

pub fn str_date_time_now() -> String {
    format!("{:?}", std::time::SystemTime::now())
}

// ============================================================================
// Histogram helpers
// ============================================================================

#[derive(Debug, Clone)]
pub struct FreqHist {
    counts: Vec<u64>,
}
impl Default for FreqHist {
    fn default() -> Self {
        Self::new()
    }
}

impl FreqHist {
    pub fn new() -> Self {
        Self {
            counts: vec![0u64; 256],
        }
    }
    pub fn add(&mut self, byte: u8) {
        self.counts[byte as usize] += 1;
    }
    pub fn count(&self, byte: u8) -> u64 {
        self.counts[byte as usize]
    }
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }
    pub fn reset(&mut self) {
        self.counts.fill(0);
    }
}

pub type FreqHistO1 = FreqHist;
pub type FreqHistO2 = FreqHist;
pub type HistogramData = FreqHist;
pub type HistogramDataO1 = FreqHist;
pub type HistogramDataO2 = FreqHist;

pub struct HistogramCollection;
impl Default for HistogramCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl HistogramCollection {
    pub fn new() -> Self {
        Self
    }
}

// ============================================================================
// Entropy analysis types
// ============================================================================

pub struct EntropyAnalyzer;
impl Default for EntropyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl EntropyAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct EntropyConfig;
impl Default for EntropyConfig {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct EntropyResults;

#[derive(Debug, Clone)]
pub struct CompressionEstimates;

#[derive(Debug, Clone)]
pub struct DistributionInfo;

pub struct EntropyAnalyzerCollection;

// ============================================================================
// Buffer management types
// ============================================================================

pub struct ContextBuffer {
    data: Vec<u8>,
}
impl ContextBuffer {
    pub fn new(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
        }
    }
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

#[derive(Debug, Clone)]
pub struct BufferMetadata;

#[derive(Debug, Clone, Copy)]
pub enum BufferPriority {
    Low,
    Normal,
    High,
    Critical,
}

pub trait StatisticsContext: Send + Sync {}

pub struct DefaultStatisticsContext;
impl StatisticsContext for DefaultStatisticsContext {}

pub struct BufferPoolManager;
impl BufferPoolManager {
    pub fn new(_config: BufferPoolConfig) -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct BufferPoolConfig;
impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self
    }
}

pub struct PoolStatistics;
pub struct ScopedBuffer;

// ============================================================================
// Profiling types
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct Profiler {
    config: ProfilerConfig,
}
impl Profiler {
    pub fn new(config: ProfilerConfig) -> Self {
        Self { config }
    }
    pub fn config(&self) -> &ProfilerConfig {
        &self.config
    }
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    pub enabled: bool,
    pub sample_rate: f64,
}
impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_rate: 1.0,
        }
    }
}

pub struct OperationProfile;

pub struct ProfiledOperation;

static GLOBAL_PROFILER: std::sync::OnceLock<Profiler> = std::sync::OnceLock::new();

pub fn stats_global_profiler() -> &'static Profiler {
    GLOBAL_PROFILER.get_or_init(Profiler::default)
}

pub fn init_global_profiler(config: ProfilerConfig) -> Result<(), ZiporaError> {
    GLOBAL_PROFILER
        .set(Profiler::new(config))
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

    #[test]
    fn test_c12_8_statistics_memory_breakdown_overwrite_and_functional_accumulators() {
        // 1. MemoryBreakdown::add_component must subtract overwritten component size from total
        let mut breakdown = MemoryBreakdown::new();
        breakdown.add_component("nodes", 1000);
        breakdown.add_component("cache", 500);
        breakdown.add_component("nodes", 1200);
        assert_eq!(
            breakdown.total, 1700,
            "MemoryBreakdown::total must equal sum of components after overwriting key"
        );

        // 2. StatsMemoryStats must track category memory, peak_memory, saturating deallocation, merge, reset, report
        let mut stats = TrieStatistics::new();
        stats.memory.record_allocation(1024, MemoryCategory::Nodes);
        stats.memory.record_allocation(512, MemoryCategory::Cache);
        stats
            .memory
            .record_allocation(256, MemoryCategory::Overhead);
        assert_eq!(stats.memory.nodes_memory.load(Ordering::Relaxed), 1024);
        assert_eq!(stats.memory.cache_memory.load(Ordering::Relaxed), 512);
        assert_eq!(stats.memory.overhead_memory.load(Ordering::Relaxed), 256);
        assert_eq!(stats.memory.total_allocated.load(Ordering::Relaxed), 1792);
        assert_eq!(stats.memory.peak_memory.load(Ordering::Relaxed), 1792);

        stats.memory.record_deallocation(512, MemoryCategory::Cache);
        assert_eq!(stats.memory.cache_memory.load(Ordering::Relaxed), 0);
        assert_eq!(stats.memory.total_allocated.load(Ordering::Relaxed), 1280);
        assert_eq!(stats.memory.peak_memory.load(Ordering::Relaxed), 1792);

        // Over-deallocation must saturate at 0 instead of wrapping to usize::MAX
        stats
            .memory
            .record_deallocation(10_000, MemoryCategory::Nodes);
        assert_eq!(stats.memory.nodes_memory.load(Ordering::Relaxed), 0);

        // 3. StatsPerformanceStats must track cache_hits, cache_misses, total_operations
        stats.performance.record_insert();
        stats.performance.record_lookup(true);
        stats.performance.record_lookup(false);
        stats.performance.record_delete();
        assert_eq!(stats.performance.cache_hits.load(Ordering::Relaxed), 1);
        assert_eq!(stats.performance.cache_misses.load(Ordering::Relaxed), 1);
        assert_eq!(stats.performance.total_operations.load(Ordering::Relaxed), 4);

        // 4. TrieStatistics::merge, generate_report, and reset must function
        let other = TrieStatistics::new();
        other.memory.record_allocation(4096, MemoryCategory::Nodes);
        other.performance.record_lookup(true);
        other.errors.record_error(ErrorType::Io);
        stats.merge(&other);
        assert_eq!(stats.memory.nodes_memory.load(Ordering::Relaxed), 4096);
        assert_eq!(stats.performance.cache_hits.load(Ordering::Relaxed), 2);
        assert_eq!(stats.errors.total_errors.load(Ordering::Relaxed), 1);
        assert!(!stats.generate_report().is_empty());

        stats.reset();
        assert_eq!(stats.memory.total_allocated.load(Ordering::Relaxed), 0);
        assert_eq!(stats.memory.nodes_memory.load(Ordering::Relaxed), 0);
        assert_eq!(stats.memory.peak_memory.load(Ordering::Relaxed), 0);
        assert_eq!(stats.performance.total_operations.load(Ordering::Relaxed), 0);
        assert_eq!(stats.errors.total_errors.load(Ordering::Relaxed), 0);
    }
}
