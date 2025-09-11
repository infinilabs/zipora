//! Advanced Statistics and Monitoring Framework
//!
//! Comprehensive operational monitoring capabilities including memory usage tracking,
//! performance counters, compression metrics, distribution analysis, and error tracking.
//! 
//! Based on sophisticated patterns from high-performance systems, this module provides
//! real-time monitoring with nanosecond precision timing and hierarchical statistics
//! aggregation.

pub mod core_stats;
pub mod memory_tracking;
pub mod timing;
pub mod histogram;
pub mod entropy_analysis;
pub mod buffer_management;
pub mod profiling;

pub use core_stats::*;
pub use memory_tracking::*;
pub use timing::*;
pub use histogram::*;
pub use entropy_analysis::*;
pub use buffer_management::*;
pub use profiling::*;

use crate::error::ZiporaError;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::collections::HashMap;

/// Universal memory size interface for comprehensive memory tracking
pub trait MemorySize {
    /// Returns the total memory footprint in bytes
    fn mem_size(&self) -> usize;
    
    /// Returns detailed memory breakdown by component
    fn detailed_mem_size(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            total: self.mem_size(),
            components: HashMap::new(),
        }
    }
}

/// Simple statistics structure matching topling-zip ZipStat
#[derive(Debug, Clone)]
pub struct TrieStat {
    // All times are in seconds (matching topling-zip pattern)
    pub insert_time: f64,
    pub lookup_time: f64,
    pub build_time: f64,
    pub total_bytes: u64, // matching pipelineThroughBytes pattern
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
    
    /// Simple print method matching topling-zip pattern
    pub fn print(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(writer, "=== Trie Statistics ===")?;
        writeln!(writer, "Insert Time: {:.6} seconds", self.insert_time)?;
        writeln!(writer, "Lookup Time: {:.6} seconds", self.lookup_time)?;
        writeln!(writer, "Build Time: {:.6} seconds", self.build_time)?;
        writeln!(writer, "Total Bytes: {} bytes", self.total_bytes)?;
        Ok(())
    }
}

/// Core statistics structure for comprehensive operational monitoring
#[derive(Debug)]
pub struct TrieStatistics {
    /// Memory usage statistics
    pub memory: MemoryStats,
    
    /// Performance counters
    pub performance: PerformanceStats,
    
    /// Compression metrics
    pub compression: CompressionStats,
    
    /// Distribution analysis
    pub distribution: DistributionStats,
    
    /// Error tracking
    pub errors: ErrorStats,
    
    /// Timing information
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
            memory: MemoryStats::new(),
            performance: PerformanceStats::new(),
            compression: CompressionStats::new(),
            distribution: DistributionStats::new(),
            errors: ErrorStats::new(),
            timing: TimingStats::new(),
        }
    }
    
    /// Merge statistics from another instance
    pub fn merge(&mut self, other: &TrieStatistics) {
        self.memory.merge(&other.memory);
        self.performance.merge(&other.performance);
        self.compression.merge(&other.compression);
        self.distribution.merge(&other.distribution);
        self.errors.merge(&other.errors);
        self.timing.merge(&other.timing);
    }
    
    /// Reset all statistics to initial state
    pub fn reset(&mut self) {
        self.memory.reset();
        self.performance.reset();
        self.compression.reset();
        self.distribution.reset();
        self.errors.reset();
        self.timing.reset();
    }
    
    /// Generate comprehensive report
    pub fn generate_report(&self) -> String {
        format!(
            "=== Trie Statistics Report ===\n{}\n{}\n{}\n{}\n{}\n{}",
            self.memory.report(),
            self.performance.report(),
            self.compression.report(),
            self.distribution.report(),
            self.errors.report(),
            self.timing.report()
        )
    }
}

/// Memory usage statistics
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
            total_allocated: AtomicUsize::new(0),
            nodes_memory: AtomicUsize::new(0),
            cache_memory: AtomicUsize::new(0),
            overhead_memory: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
        }
    }
    
    pub fn record_allocation(&self, size: usize, category: MemoryCategory) {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        let new_total = self.total_allocated.fetch_add(size, Ordering::Relaxed) + size;
        
        // Update peak memory if necessary
        let current_peak = self.peak_memory.load(Ordering::Relaxed);
        if new_total > current_peak {
            self.peak_memory.store(new_total, Ordering::Relaxed);
        }
        
        // Update category-specific memory
        match category {
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
    }
    
    pub fn record_deallocation(&self, size: usize, category: MemoryCategory) {
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
        self.total_allocated.fetch_sub(size, Ordering::Relaxed);
        
        match category {
            MemoryCategory::Nodes => {
                self.nodes_memory.fetch_sub(size, Ordering::Relaxed);
            }
            MemoryCategory::Cache => {
                self.cache_memory.fetch_sub(size, Ordering::Relaxed);
            }
            MemoryCategory::Overhead => {
                self.overhead_memory.fetch_sub(size, Ordering::Relaxed);
            }
        }
    }
    
    pub fn merge(&mut self, other: &MemoryStats) {
        let other_total = other.total_allocated.load(Ordering::Relaxed);
        let other_nodes = other.nodes_memory.load(Ordering::Relaxed);
        let other_cache = other.cache_memory.load(Ordering::Relaxed);
        let other_overhead = other.overhead_memory.load(Ordering::Relaxed);
        let other_peak = other.peak_memory.load(Ordering::Relaxed);
        let other_alloc_count = other.allocation_count.load(Ordering::Relaxed);
        let other_dealloc_count = other.deallocation_count.load(Ordering::Relaxed);
        
        self.total_allocated.fetch_add(other_total, Ordering::Relaxed);
        self.nodes_memory.fetch_add(other_nodes, Ordering::Relaxed);
        self.cache_memory.fetch_add(other_cache, Ordering::Relaxed);
        self.overhead_memory.fetch_add(other_overhead, Ordering::Relaxed);
        self.allocation_count.fetch_add(other_alloc_count, Ordering::Relaxed);
        self.deallocation_count.fetch_add(other_dealloc_count, Ordering::Relaxed);
        
        let current_peak = self.peak_memory.load(Ordering::Relaxed);
        if other_peak > current_peak {
            self.peak_memory.store(other_peak, Ordering::Relaxed);
        }
    }
    
    pub fn reset(&mut self) {
        self.total_allocated.store(0, Ordering::Relaxed);
        self.nodes_memory.store(0, Ordering::Relaxed);
        self.cache_memory.store(0, Ordering::Relaxed);
        self.overhead_memory.store(0, Ordering::Relaxed);
        self.peak_memory.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.deallocation_count.store(0, Ordering::Relaxed);
    }
    
    pub fn report(&self) -> String {
        format!(
            "Memory Stats:\n  Total: {} bytes\n  Nodes: {} bytes\n  Cache: {} bytes\n  Overhead: {} bytes\n  Peak: {} bytes\n  Allocations: {}\n  Deallocations: {}",
            self.total_allocated.load(Ordering::Relaxed),
            self.nodes_memory.load(Ordering::Relaxed),
            self.cache_memory.load(Ordering::Relaxed),
            self.overhead_memory.load(Ordering::Relaxed),
            self.peak_memory.load(Ordering::Relaxed),
            self.allocation_count.load(Ordering::Relaxed),
            self.deallocation_count.load(Ordering::Relaxed)
        )
    }
}

/// Memory allocation categories
#[derive(Debug, Clone, Copy)]
pub enum MemoryCategory {
    Nodes,
    Cache,
    Overhead,
}

/// Performance counters for operation tracking
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
        self.insert_count.fetch_add(1, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_lookup(&self, hit: bool) {
        self.lookup_count.fetch_add(1, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        
        if hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    pub fn record_delete(&self) {
        self.delete_count.fetch_add(1, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_operation_time(&self, duration_ns: u64) {
        // Simple moving average
        let current_avg = self.average_operation_time_ns.load(Ordering::Relaxed);
        let total_ops = self.total_operations.load(Ordering::Relaxed);
        
        if total_ops > 0 {
            let new_avg = (current_avg * (total_ops - 1) + duration_ns) / total_ops;
            self.average_operation_time_ns.store(new_avg, Ordering::Relaxed);
        }
    }
    
    pub fn record_failure(&self) {
        self.failed_operations.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed) as f64;
        let total_lookups = self.lookup_count.load(Ordering::Relaxed) as f64;
        
        if total_lookups > 0.0 {
            hits / total_lookups
        } else {
            0.0
        }
    }
    
    pub fn failure_rate(&self) -> f64 {
        let failures = self.failed_operations.load(Ordering::Relaxed) as f64;
        let total = self.total_operations.load(Ordering::Relaxed) as f64;
        
        if total > 0.0 {
            failures / total
        } else {
            0.0
        }
    }
    
    pub fn merge(&mut self, other: &PerformanceStats) {
        self.insert_count.fetch_add(other.insert_count.load(Ordering::Relaxed), Ordering::Relaxed);
        self.lookup_count.fetch_add(other.lookup_count.load(Ordering::Relaxed), Ordering::Relaxed);
        self.delete_count.fetch_add(other.delete_count.load(Ordering::Relaxed), Ordering::Relaxed);
        self.cache_hits.fetch_add(other.cache_hits.load(Ordering::Relaxed), Ordering::Relaxed);
        self.cache_misses.fetch_add(other.cache_misses.load(Ordering::Relaxed), Ordering::Relaxed);
        self.total_operations.fetch_add(other.total_operations.load(Ordering::Relaxed), Ordering::Relaxed);
        self.failed_operations.fetch_add(other.failed_operations.load(Ordering::Relaxed), Ordering::Relaxed);
        
        // Merge average operation time (weighted average)
        let self_total = self.total_operations.load(Ordering::Relaxed);
        let other_total = other.total_operations.load(Ordering::Relaxed);
        let combined_total = self_total + other_total;
        
        if combined_total > 0 {
            let self_avg = self.average_operation_time_ns.load(Ordering::Relaxed);
            let other_avg = other.average_operation_time_ns.load(Ordering::Relaxed);
            let weighted_avg = (self_avg * self_total + other_avg * other_total) / combined_total;
            self.average_operation_time_ns.store(weighted_avg, Ordering::Relaxed);
        }
    }
    
    pub fn reset(&mut self) {
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
        format!(
            "Performance Stats:\n  Inserts: {}\n  Lookups: {}\n  Deletes: {}\n  Cache Hit Rate: {:.2}%\n  Failure Rate: {:.2}%\n  Avg Operation Time: {} ns",
            self.insert_count.load(Ordering::Relaxed),
            self.lookup_count.load(Ordering::Relaxed),
            self.delete_count.load(Ordering::Relaxed),
            self.cache_hit_rate() * 100.0,
            self.failure_rate() * 100.0,
            self.average_operation_time_ns.load(Ordering::Relaxed)
        )
    }
}

/// Compression metrics tracking
#[derive(Debug)]
pub struct CompressionStats {
    pub original_size: AtomicUsize,
    pub compressed_size: AtomicUsize,
    pub compression_operations: AtomicU64,
    pub decompression_operations: AtomicU64,
    pub compression_time_ns: AtomicU64,
    pub decompression_time_ns: AtomicU64,
}

impl CompressionStats {
    pub fn new() -> Self {
        Self {
            original_size: AtomicUsize::new(0),
            compressed_size: AtomicUsize::new(0),
            compression_operations: AtomicU64::new(0),
            decompression_operations: AtomicU64::new(0),
            compression_time_ns: AtomicU64::new(0),
            decompression_time_ns: AtomicU64::new(0),
        }
    }
    
    pub fn record_compression(&self, original: usize, compressed: usize, time_ns: u64) {
        self.original_size.fetch_add(original, Ordering::Relaxed);
        self.compressed_size.fetch_add(compressed, Ordering::Relaxed);
        self.compression_operations.fetch_add(1, Ordering::Relaxed);
        self.compression_time_ns.fetch_add(time_ns, Ordering::Relaxed);
    }
    
    pub fn record_decompression(&self, time_ns: u64) {
        self.decompression_operations.fetch_add(1, Ordering::Relaxed);
        self.decompression_time_ns.fetch_add(time_ns, Ordering::Relaxed);
    }
    
    pub fn compression_ratio(&self) -> f64 {
        let original = self.original_size.load(Ordering::Relaxed) as f64;
        let compressed = self.compressed_size.load(Ordering::Relaxed) as f64;
        
        if original > 0.0 {
            compressed / original
        } else {
            0.0
        }
    }
    
    pub fn space_savings(&self) -> f64 {
        1.0 - self.compression_ratio()
    }
    
    pub fn merge(&mut self, other: &CompressionStats) {
        self.original_size.fetch_add(other.original_size.load(Ordering::Relaxed), Ordering::Relaxed);
        self.compressed_size.fetch_add(other.compressed_size.load(Ordering::Relaxed), Ordering::Relaxed);
        self.compression_operations.fetch_add(other.compression_operations.load(Ordering::Relaxed), Ordering::Relaxed);
        self.decompression_operations.fetch_add(other.decompression_operations.load(Ordering::Relaxed), Ordering::Relaxed);
        self.compression_time_ns.fetch_add(other.compression_time_ns.load(Ordering::Relaxed), Ordering::Relaxed);
        self.decompression_time_ns.fetch_add(other.decompression_time_ns.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    
    pub fn reset(&mut self) {
        self.original_size.store(0, Ordering::Relaxed);
        self.compressed_size.store(0, Ordering::Relaxed);
        self.compression_operations.store(0, Ordering::Relaxed);
        self.decompression_operations.store(0, Ordering::Relaxed);
        self.compression_time_ns.store(0, Ordering::Relaxed);
        self.decompression_time_ns.store(0, Ordering::Relaxed);
    }
    
    pub fn report(&self) -> String {
        format!(
            "Compression Stats:\n  Original Size: {} bytes\n  Compressed Size: {} bytes\n  Compression Ratio: {:.2}\n  Space Savings: {:.2}%\n  Operations: {} compress, {} decompress",
            self.original_size.load(Ordering::Relaxed),
            self.compressed_size.load(Ordering::Relaxed),
            self.compression_ratio(),
            self.space_savings() * 100.0,
            self.compression_operations.load(Ordering::Relaxed),
            self.decompression_operations.load(Ordering::Relaxed)
        )
    }
}

/// Distribution analysis statistics
#[derive(Debug)]
pub struct DistributionStats {
    pub total_samples: AtomicU64,
    pub min_value: AtomicU64,
    pub max_value: AtomicU64,
    pub sum: AtomicU64,
    pub sum_squares: AtomicU64,
}

impl DistributionStats {
    pub fn new() -> Self {
        Self {
            total_samples: AtomicU64::new(0),
            min_value: AtomicU64::new(u64::MAX),
            max_value: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            sum_squares: AtomicU64::new(0),
        }
    }
    
    pub fn add_sample(&self, value: u64) {
        self.total_samples.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(value, Ordering::Relaxed);
        self.sum_squares.fetch_add(value * value, Ordering::Relaxed);
        
        // Update min value
        let mut current_min = self.min_value.load(Ordering::Relaxed);
        while value < current_min {
            match self.min_value.compare_exchange_weak(
                current_min,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }
        
        // Update max value
        let mut current_max = self.max_value.load(Ordering::Relaxed);
        while value > current_max {
            match self.max_value.compare_exchange_weak(
                current_max,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }
    
    pub fn mean(&self) -> f64 {
        let samples = self.total_samples.load(Ordering::Relaxed);
        if samples > 0 {
            self.sum.load(Ordering::Relaxed) as f64 / samples as f64
        } else {
            0.0
        }
    }
    
    pub fn variance(&self) -> f64 {
        let samples = self.total_samples.load(Ordering::Relaxed);
        if samples > 1 {
            let mean = self.mean();
            let sum_sq = self.sum_squares.load(Ordering::Relaxed) as f64;
            (sum_sq / samples as f64) - (mean * mean)
        } else {
            0.0
        }
    }
    
    pub fn standard_deviation(&self) -> f64 {
        self.variance().sqrt()
    }
    
    pub fn merge(&mut self, other: &DistributionStats) {
        self.total_samples.fetch_add(other.total_samples.load(Ordering::Relaxed), Ordering::Relaxed);
        self.sum.fetch_add(other.sum.load(Ordering::Relaxed), Ordering::Relaxed);
        self.sum_squares.fetch_add(other.sum_squares.load(Ordering::Relaxed), Ordering::Relaxed);
        
        let other_min = other.min_value.load(Ordering::Relaxed);
        let current_min = self.min_value.load(Ordering::Relaxed);
        if other_min < current_min {
            self.min_value.store(other_min, Ordering::Relaxed);
        }
        
        let other_max = other.max_value.load(Ordering::Relaxed);
        let current_max = self.max_value.load(Ordering::Relaxed);
        if other_max > current_max {
            self.max_value.store(other_max, Ordering::Relaxed);
        }
    }
    
    pub fn reset(&mut self) {
        self.total_samples.store(0, Ordering::Relaxed);
        self.min_value.store(u64::MAX, Ordering::Relaxed);
        self.max_value.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.sum_squares.store(0, Ordering::Relaxed);
    }
    
    pub fn report(&self) -> String {
        let samples = self.total_samples.load(Ordering::Relaxed);
        if samples > 0 {
            format!(
                "Distribution Stats:\n  Samples: {}\n  Min: {}\n  Max: {}\n  Mean: {:.2}\n  Std Dev: {:.2}",
                samples,
                self.min_value.load(Ordering::Relaxed),
                self.max_value.load(Ordering::Relaxed),
                self.mean(),
                self.standard_deviation()
            )
        } else {
            "Distribution Stats: No samples collected".to_string()
        }
    }
}

/// Error tracking statistics
#[derive(Debug)]
pub struct ErrorStats {
    pub total_errors: AtomicU64,
    pub memory_errors: AtomicU64,
    pub io_errors: AtomicU64,
    pub corruption_errors: AtomicU64,
    pub timeout_errors: AtomicU64,
    pub other_errors: AtomicU64,
}

impl ErrorStats {
    pub fn new() -> Self {
        Self {
            total_errors: AtomicU64::new(0),
            memory_errors: AtomicU64::new(0),
            io_errors: AtomicU64::new(0),
            corruption_errors: AtomicU64::new(0),
            timeout_errors: AtomicU64::new(0),
            other_errors: AtomicU64::new(0),
        }
    }
    
    pub fn record_error(&self, error_type: ErrorType) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
        
        match error_type {
            ErrorType::Memory => self.memory_errors.fetch_add(1, Ordering::Relaxed),
            ErrorType::Io => self.io_errors.fetch_add(1, Ordering::Relaxed),
            ErrorType::Corruption => self.corruption_errors.fetch_add(1, Ordering::Relaxed),
            ErrorType::Timeout => self.timeout_errors.fetch_add(1, Ordering::Relaxed),
            ErrorType::Other => self.other_errors.fetch_add(1, Ordering::Relaxed),
        };
    }
    
    pub fn merge(&mut self, other: &ErrorStats) {
        self.total_errors.fetch_add(other.total_errors.load(Ordering::Relaxed), Ordering::Relaxed);
        self.memory_errors.fetch_add(other.memory_errors.load(Ordering::Relaxed), Ordering::Relaxed);
        self.io_errors.fetch_add(other.io_errors.load(Ordering::Relaxed), Ordering::Relaxed);
        self.corruption_errors.fetch_add(other.corruption_errors.load(Ordering::Relaxed), Ordering::Relaxed);
        self.timeout_errors.fetch_add(other.timeout_errors.load(Ordering::Relaxed), Ordering::Relaxed);
        self.other_errors.fetch_add(other.other_errors.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    
    pub fn reset(&mut self) {
        self.total_errors.store(0, Ordering::Relaxed);
        self.memory_errors.store(0, Ordering::Relaxed);
        self.io_errors.store(0, Ordering::Relaxed);
        self.corruption_errors.store(0, Ordering::Relaxed);
        self.timeout_errors.store(0, Ordering::Relaxed);
        self.other_errors.store(0, Ordering::Relaxed);
    }
    
    pub fn report(&self) -> String {
        format!(
            "Error Stats:\n  Total: {}\n  Memory: {}\n  I/O: {}\n  Corruption: {}\n  Timeout: {}\n  Other: {}",
            self.total_errors.load(Ordering::Relaxed),
            self.memory_errors.load(Ordering::Relaxed),
            self.io_errors.load(Ordering::Relaxed),
            self.corruption_errors.load(Ordering::Relaxed),
            self.timeout_errors.load(Ordering::Relaxed),
            self.other_errors.load(Ordering::Relaxed)
        )
    }
}

/// Error type classification
#[derive(Debug, Clone, Copy)]
pub enum ErrorType {
    Memory,
    Io,
    Corruption,
    Timeout,
    Other,
}

/// Timing statistics with high precision
#[derive(Debug)]
pub struct TimingStats {
    pub creation_time: Instant,
    pub last_access_time: AtomicU64,
    pub total_runtime_ns: AtomicU64,
    pub active_time_ns: AtomicU64,
    pub idle_time_ns: AtomicU64,
}

impl TimingStats {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            creation_time: now,
            last_access_time: AtomicU64::new(now.elapsed().as_nanos() as u64),
            total_runtime_ns: AtomicU64::new(0),
            active_time_ns: AtomicU64::new(0),
            idle_time_ns: AtomicU64::new(0),
        }
    }
    
    pub fn record_access(&self) {
        let elapsed_ns = self.creation_time.elapsed().as_nanos() as u64;
        self.last_access_time.store(elapsed_ns, Ordering::Relaxed);
    }
    
    pub fn record_active_time(&self, duration_ns: u64) {
        self.active_time_ns.fetch_add(duration_ns, Ordering::Relaxed);
        self.total_runtime_ns.fetch_add(duration_ns, Ordering::Relaxed);
    }
    
    pub fn record_idle_time(&self, duration_ns: u64) {
        self.idle_time_ns.fetch_add(duration_ns, Ordering::Relaxed);
        self.total_runtime_ns.fetch_add(duration_ns, Ordering::Relaxed);
    }
    
    pub fn uptime(&self) -> Duration {
        self.creation_time.elapsed()
    }
    
    pub fn utilization_rate(&self) -> f64 {
        let total = self.total_runtime_ns.load(Ordering::Relaxed) as f64;
        let active = self.active_time_ns.load(Ordering::Relaxed) as f64;
        
        if total > 0.0 {
            active / total
        } else {
            0.0
        }
    }
    
    pub fn merge(&mut self, other: &TimingStats) {
        self.total_runtime_ns.fetch_add(other.total_runtime_ns.load(Ordering::Relaxed), Ordering::Relaxed);
        self.active_time_ns.fetch_add(other.active_time_ns.load(Ordering::Relaxed), Ordering::Relaxed);
        self.idle_time_ns.fetch_add(other.idle_time_ns.load(Ordering::Relaxed), Ordering::Relaxed);
        
        let other_last_access = other.last_access_time.load(Ordering::Relaxed);
        let current_last_access = self.last_access_time.load(Ordering::Relaxed);
        if other_last_access > current_last_access {
            self.last_access_time.store(other_last_access, Ordering::Relaxed);
        }
    }
    
    pub fn reset(&mut self) {
        let now = Instant::now();
        self.creation_time = now;
        self.last_access_time.store(0, Ordering::Relaxed);
        self.total_runtime_ns.store(0, Ordering::Relaxed);
        self.active_time_ns.store(0, Ordering::Relaxed);
        self.idle_time_ns.store(0, Ordering::Relaxed);
    }
    
    pub fn report(&self) -> String {
        format!(
            "Timing Stats:\n  Uptime: {:.2}s\n  Total Runtime: {:.2}s\n  Active Time: {:.2}s\n  Utilization: {:.2}%",
            self.uptime().as_secs_f64(),
            self.total_runtime_ns.load(Ordering::Relaxed) as f64 / 1_000_000_000.0,
            self.active_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000_000.0,
            self.utilization_rate() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_trie_statistics_creation() {
        let stats = TrieStatistics::new();
        assert_eq!(stats.memory.total_allocated.load(Ordering::Relaxed), 0);
        assert_eq!(stats.performance.total_operations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_memory_stats_tracking() {
        let stats = MemoryStats::new();
        
        stats.record_allocation(1024, MemoryCategory::Nodes);
        assert_eq!(stats.total_allocated.load(Ordering::Relaxed), 1024);
        assert_eq!(stats.nodes_memory.load(Ordering::Relaxed), 1024);
        assert_eq!(stats.allocation_count.load(Ordering::Relaxed), 1);
        
        stats.record_deallocation(512, MemoryCategory::Nodes);
        assert_eq!(stats.total_allocated.load(Ordering::Relaxed), 512);
        assert_eq!(stats.nodes_memory.load(Ordering::Relaxed), 512);
        assert_eq!(stats.deallocation_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_performance_stats_cache_hit_rate() {
        let stats = PerformanceStats::new();
        
        stats.record_lookup(true);  // hit
        stats.record_lookup(true);  // hit
        stats.record_lookup(false); // miss
        
        assert_eq!(stats.cache_hit_rate(), 2.0 / 3.0);
        assert_eq!(stats.lookup_count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_compression_stats_ratio() {
        let stats = CompressionStats::new();
        
        stats.record_compression(1000, 500, 1000000); // 50% compression
        assert_eq!(stats.compression_ratio(), 0.5);
        assert_eq!(stats.space_savings(), 0.5);
    }

    #[test]
    fn test_distribution_stats_calculations() {
        let stats = DistributionStats::new();
        
        stats.add_sample(10);
        stats.add_sample(20);
        stats.add_sample(30);
        
        assert_eq!(stats.mean(), 20.0);
        assert_eq!(stats.min_value.load(Ordering::Relaxed), 10);
        assert_eq!(stats.max_value.load(Ordering::Relaxed), 30);
        
        // Variance should be 100 for values [10, 20, 30]
        let variance = stats.variance();
        assert!((variance - 66.66666666666667).abs() < 0.0001); // Close to expected variance
    }

    #[test]
    fn test_error_stats_tracking() {
        let stats = ErrorStats::new();
        
        stats.record_error(ErrorType::Memory);
        stats.record_error(ErrorType::Io);
        stats.record_error(ErrorType::Memory);
        
        assert_eq!(stats.total_errors.load(Ordering::Relaxed), 3);
        assert_eq!(stats.memory_errors.load(Ordering::Relaxed), 2);
        assert_eq!(stats.io_errors.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_timing_stats_utilization() {
        let stats = TimingStats::new();
        
        stats.record_active_time(500_000_000); // 0.5 seconds
        stats.record_idle_time(500_000_000);   // 0.5 seconds
        
        assert_eq!(stats.utilization_rate(), 0.5);
        assert_eq!(stats.total_runtime_ns.load(Ordering::Relaxed), 1_000_000_000);
    }

    #[test]
    fn test_statistics_merging() {
        let mut stats1 = TrieStatistics::new();
        let stats2 = TrieStatistics::new();
        
        stats1.memory.record_allocation(1000, MemoryCategory::Nodes);
        stats2.memory.record_allocation(2000, MemoryCategory::Cache);
        
        stats1.merge(&stats2);
        
        assert_eq!(stats1.memory.total_allocated.load(Ordering::Relaxed), 3000);
        assert_eq!(stats1.memory.nodes_memory.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_statistics_reset() {
        let mut stats = TrieStatistics::new();
        
        stats.memory.record_allocation(1000, MemoryCategory::Nodes);
        stats.performance.record_insert();
        
        stats.reset();
        
        assert_eq!(stats.memory.total_allocated.load(Ordering::Relaxed), 0);
        assert_eq!(stats.performance.insert_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_memory_breakdown() {
        let mut breakdown = MemoryBreakdown::new();
        breakdown.add_component("nodes", 1000);
        breakdown.add_component("cache", 500);
        
        assert_eq!(breakdown.total, 1500);
        assert_eq!(breakdown.components.len(), 2);
        assert_eq!(breakdown.components["nodes"], 1000);
    }

    #[test]
    fn test_concurrent_statistics() {
        let stats = std::sync::Arc::new(MemoryStats::new());
        let mut handles = vec![];
        
        // Spawn multiple threads to test thread safety
        for i in 0..10 {
            let stats_clone = stats.clone();
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    stats_clone.record_allocation(i * 10, MemoryCategory::Nodes);
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Should have recorded 1000 allocations total
        assert_eq!(stats.allocation_count.load(Ordering::Relaxed), 1000);
    }
}