//! Cache statistics and performance monitoring

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Comprehensive cache statistics
#[derive(Debug)]
pub struct CacheStatistics {
    /// Hit counts by type
    hit_counts: [AtomicU64; 7],
    
    /// Total cache hits
    total_hits: AtomicU64,
    
    /// Total cache misses
    total_misses: AtomicU64,
    
    /// Total bytes read
    bytes_read: AtomicU64,
    
    /// Total bytes cached
    bytes_cached: AtomicU64,
    
    /// Number of evictions
    evictions: AtomicU64,
    
    /// Hash collision statistics
    hash_collisions: AtomicU64,
    
    /// Average probe distance
    total_probe_distance: AtomicU64,
    total_probes: AtomicU64,
    
    /// Timing statistics
    total_read_time_ns: AtomicU64,
    total_reads: AtomicU64,
    
    /// Memory usage
    memory_allocated: AtomicU64,
    memory_used: AtomicU64,
    
    /// Lock contention
    lock_contentions: AtomicU64,
    lock_acquisitions: AtomicU64,
    
    /// Background maintenance
    maintenance_cycles: AtomicU64,
    defragmentation_runs: AtomicU64,
    
    /// Error counts
    allocation_failures: AtomicU64,
    load_failures: AtomicU64,
    
    /// Start time for rate calculations
    start_time: Instant,
}

impl CacheStatistics {
    /// Create new statistics instance
    pub fn new() -> Self {
        Self {
            hit_counts: [
                AtomicU64::new(0), // Hit
                AtomicU64::new(0), // EvictedOthers
                AtomicU64::new(0), // InitialFree
                AtomicU64::new(0), // DroppedFree
                AtomicU64::new(0), // HitOthersLoad
                AtomicU64::new(0), // Mix
                AtomicU64::new(0), // Miss
            ],
            total_hits: AtomicU64::new(0),
            total_misses: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            bytes_cached: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            hash_collisions: AtomicU64::new(0),
            total_probe_distance: AtomicU64::new(0),
            total_probes: AtomicU64::new(0),
            total_read_time_ns: AtomicU64::new(0),
            total_reads: AtomicU64::new(0),
            memory_allocated: AtomicU64::new(0),
            memory_used: AtomicU64::new(0),
            lock_contentions: AtomicU64::new(0),
            lock_acquisitions: AtomicU64::new(0),
            maintenance_cycles: AtomicU64::new(0),
            defragmentation_runs: AtomicU64::new(0),
            allocation_failures: AtomicU64::new(0),
            load_failures: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }
    
    /// Record cache hit
    pub fn record_hit(&self, hit_type: CacheHitType) {
        self.hit_counts[hit_type.as_index()].fetch_add(1, Ordering::Relaxed);
        self.total_hits.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record cache miss
    pub fn record_miss(&self) {
        self.total_misses.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record bytes read
    pub fn record_bytes_read(&self, bytes: u64) {
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        self.total_reads.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record bytes cached
    pub fn record_bytes_cached(&self, bytes: u64) {
        self.bytes_cached.fetch_add(bytes, Ordering::Relaxed);
    }
    
    /// Record eviction
    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record hash collision
    pub fn record_hash_collision(&self) {
        self.hash_collisions.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record probe distance
    pub fn record_probe_distance(&self, distance: u64) {
        self.total_probe_distance.fetch_add(distance, Ordering::Relaxed);
        self.total_probes.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record read timing
    pub fn record_read_time(&self, duration: Duration) {
        self.total_read_time_ns.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }
    
    /// Record memory usage
    pub fn record_memory_allocated(&self, bytes: u64) {
        self.memory_allocated.fetch_add(bytes, Ordering::Relaxed);
    }
    
    /// Record memory usage change
    pub fn record_memory_used(&self, bytes: u64) {
        self.memory_used.store(bytes, Ordering::Relaxed);
    }
    
    /// Record lock contention
    pub fn record_lock_contention(&self) {
        self.lock_contentions.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record lock acquisition
    pub fn record_lock_acquisition(&self) {
        self.lock_acquisitions.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record maintenance cycle
    pub fn record_maintenance_cycle(&self) {
        self.maintenance_cycles.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record defragmentation
    pub fn record_defragmentation(&self) {
        self.defragmentation_runs.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record allocation failure
    pub fn record_allocation_failure(&self) {
        self.allocation_failures.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record load failure
    pub fn record_load_failure(&self) {
        self.load_failures.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.total_hits.load(Ordering::Relaxed);
        let misses = self.total_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
    
    /// Get cache miss ratio
    pub fn miss_ratio(&self) -> f64 {
        1.0 - self.hit_ratio()
    }
    
    /// Get average probe distance
    pub fn average_probe_distance(&self) -> f64 {
        let total_distance = self.total_probe_distance.load(Ordering::Relaxed);
        let total_probes = self.total_probes.load(Ordering::Relaxed);
        
        if total_probes == 0 {
            0.0
        } else {
            total_distance as f64 / total_probes as f64
        }
    }
    
    /// Get average read time in nanoseconds
    pub fn average_read_time_ns(&self) -> f64 {
        let total_time = self.total_read_time_ns.load(Ordering::Relaxed);
        let total_reads = self.total_reads.load(Ordering::Relaxed);
        
        if total_reads == 0 {
            0.0
        } else {
            total_time as f64 / total_reads as f64
        }
    }
    
    /// Get read throughput in bytes per second
    pub fn read_throughput_bps(&self) -> f64 {
        let bytes = self.bytes_read.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        
        if elapsed == 0.0 {
            0.0
        } else {
            bytes as f64 / elapsed
        }
    }
    
    /// Get read rate in operations per second
    pub fn read_rate_ops(&self) -> f64 {
        let reads = self.total_reads.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        
        if elapsed == 0.0 {
            0.0
        } else {
            reads as f64 / elapsed
        }
    }
    
    /// Get memory utilization ratio
    pub fn memory_utilization(&self) -> f64 {
        let allocated = self.memory_allocated.load(Ordering::Relaxed);
        let used = self.memory_used.load(Ordering::Relaxed);
        
        if allocated == 0 {
            0.0
        } else {
            used as f64 / allocated as f64
        }
    }
    
    /// Get lock contention ratio
    pub fn lock_contention_ratio(&self) -> f64 {
        let contentions = self.lock_contentions.load(Ordering::Relaxed);
        let acquisitions = self.lock_acquisitions.load(Ordering::Relaxed);
        
        if acquisitions == 0 {
            0.0
        } else {
            contentions as f64 / acquisitions as f64
        }
    }
    
    /// Get hit counts by type
    pub fn hit_counts(&self) -> [u64; 7] {
        [
            self.hit_counts[0].load(Ordering::Relaxed),
            self.hit_counts[1].load(Ordering::Relaxed),
            self.hit_counts[2].load(Ordering::Relaxed),
            self.hit_counts[3].load(Ordering::Relaxed),
            self.hit_counts[4].load(Ordering::Relaxed),
            self.hit_counts[5].load(Ordering::Relaxed),
            self.hit_counts[6].load(Ordering::Relaxed),
        ]
    }
    
    /// Get detailed statistics snapshot
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hit_counts: self.hit_counts(),
            total_hits: self.total_hits.load(Ordering::Relaxed),
            total_misses: self.total_misses.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_cached: self.bytes_cached.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            hash_collisions: self.hash_collisions.load(Ordering::Relaxed),
            average_probe_distance: self.average_probe_distance(),
            average_read_time_ns: self.average_read_time_ns(),
            read_throughput_bps: self.read_throughput_bps(),
            read_rate_ops: self.read_rate_ops(),
            memory_allocated: self.memory_allocated.load(Ordering::Relaxed),
            memory_used: self.memory_used.load(Ordering::Relaxed),
            memory_utilization: self.memory_utilization(),
            lock_contentions: self.lock_contentions.load(Ordering::Relaxed),
            lock_acquisitions: self.lock_acquisitions.load(Ordering::Relaxed),
            lock_contention_ratio: self.lock_contention_ratio(),
            maintenance_cycles: self.maintenance_cycles.load(Ordering::Relaxed),
            defragmentation_runs: self.defragmentation_runs.load(Ordering::Relaxed),
            allocation_failures: self.allocation_failures.load(Ordering::Relaxed),
            load_failures: self.load_failures.load(Ordering::Relaxed),
            hit_ratio: self.hit_ratio(),
            miss_ratio: self.miss_ratio(),
            uptime_seconds: self.start_time.elapsed().as_secs_f64(),
        }
    }
    
    /// Reset all statistics
    pub fn reset(&self) {
        for hit_count in &self.hit_counts {
            hit_count.store(0, Ordering::Relaxed);
        }
        
        self.total_hits.store(0, Ordering::Relaxed);
        self.total_misses.store(0, Ordering::Relaxed);
        self.bytes_read.store(0, Ordering::Relaxed);
        self.bytes_cached.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.hash_collisions.store(0, Ordering::Relaxed);
        self.total_probe_distance.store(0, Ordering::Relaxed);
        self.total_probes.store(0, Ordering::Relaxed);
        self.total_read_time_ns.store(0, Ordering::Relaxed);
        self.total_reads.store(0, Ordering::Relaxed);
        self.lock_contentions.store(0, Ordering::Relaxed);
        self.lock_acquisitions.store(0, Ordering::Relaxed);
        self.maintenance_cycles.store(0, Ordering::Relaxed);
        self.defragmentation_runs.store(0, Ordering::Relaxed);
        self.allocation_failures.store(0, Ordering::Relaxed);
        self.load_failures.store(0, Ordering::Relaxed);
    }
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Point-in-time cache statistics snapshot
#[derive(Debug, Clone)]
pub struct CacheStatsSnapshot {
    pub hit_counts: [u64; 7],
    pub total_hits: u64,
    pub total_misses: u64,
    pub bytes_read: u64,
    pub bytes_cached: u64,
    pub evictions: u64,
    pub hash_collisions: u64,
    pub average_probe_distance: f64,
    pub average_read_time_ns: f64,
    pub read_throughput_bps: f64,
    pub read_rate_ops: f64,
    pub memory_allocated: u64,
    pub memory_used: u64,
    pub memory_utilization: f64,
    pub lock_contentions: u64,
    pub lock_acquisitions: u64,
    pub lock_contention_ratio: f64,
    pub maintenance_cycles: u64,
    pub defragmentation_runs: u64,
    pub allocation_failures: u64,
    pub load_failures: u64,
    pub hit_ratio: f64,
    pub miss_ratio: f64,
    pub uptime_seconds: f64,
}

impl CacheStatsSnapshot {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "Cache Statistics:\n\
             Hit Ratio: {:.2}% ({} hits, {} misses)\n\
             Hit Types: Hit={}, Evicted={}, Free={}, Dropped={}, Loading={}, Mix={}, Miss={}\n\
             Throughput: {:.2} MB/s ({:.1} ops/s)\n\
             Memory: {:.1}% utilization ({:.2} MB used / {:.2} MB allocated)\n\
             Hash: {:.2} avg probe distance, {} collisions\n\
             Performance: {:.2}μs avg read time\n\
             Lock Contention: {:.2}% ({} / {})\n\
             Maintenance: {} cycles, {} defrag runs\n\
             Errors: {} allocation failures, {} load failures\n\
             Uptime: {:.1} seconds",
            self.hit_ratio * 100.0, self.total_hits, self.total_misses,
            self.hit_counts[0], self.hit_counts[1], self.hit_counts[2], 
            self.hit_counts[3], self.hit_counts[4], self.hit_counts[5], self.hit_counts[6],
            self.read_throughput_bps / 1_048_576.0, self.read_rate_ops,
            self.memory_utilization * 100.0, 
            self.memory_used as f64 / 1_048_576.0,
            self.memory_allocated as f64 / 1_048_576.0,
            self.average_probe_distance, self.hash_collisions,
            self.average_read_time_ns / 1000.0,
            self.lock_contention_ratio * 100.0, self.lock_contentions, self.lock_acquisitions,
            self.maintenance_cycles, self.defragmentation_runs,
            self.allocation_failures, self.load_failures,
            self.uptime_seconds
        )
    }
    
    /// Export as JSON-like structure
    pub fn to_metrics(&self) -> std::collections::HashMap<String, f64> {
        let mut metrics = std::collections::HashMap::new();
        
        metrics.insert("hit_ratio".to_string(), self.hit_ratio);
        metrics.insert("miss_ratio".to_string(), self.miss_ratio);
        metrics.insert("total_hits".to_string(), self.total_hits as f64);
        metrics.insert("total_misses".to_string(), self.total_misses as f64);
        metrics.insert("bytes_read".to_string(), self.bytes_read as f64);
        metrics.insert("bytes_cached".to_string(), self.bytes_cached as f64);
        metrics.insert("read_throughput_bps".to_string(), self.read_throughput_bps);
        metrics.insert("read_rate_ops".to_string(), self.read_rate_ops);
        metrics.insert("memory_utilization".to_string(), self.memory_utilization);
        metrics.insert("average_probe_distance".to_string(), self.average_probe_distance);
        metrics.insert("average_read_time_ns".to_string(), self.average_read_time_ns);
        metrics.insert("lock_contention_ratio".to_string(), self.lock_contention_ratio);
        metrics.insert("evictions".to_string(), self.evictions as f64);
        metrics.insert("hash_collisions".to_string(), self.hash_collisions as f64);
        metrics.insert("uptime_seconds".to_string(), self.uptime_seconds);
        
        for i in 0..7 {
            let hit_type = match i {
                0 => "hit",
                1 => "evicted_others",
                2 => "initial_free",
                3 => "dropped_free",
                4 => "hit_others_load",
                5 => "mix",
                6 => "miss",
                _ => "unknown",
            };
            metrics.insert(format!("hit_count_{}", hit_type), self.hit_counts[i] as f64);
        }
        
        metrics
    }
}