//! Advanced Profiling Utilities
//!
//! Provides sophisticated profiling capabilities for performance analysis and optimization
//! with integration into the comprehensive statistics framework.

use crate::error::ZiporaError;
use crate::statistics::{TrieStatistics, MemorySize, MemoryBreakdown, TimingStats, PerformanceStats};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Comprehensive profiler for operations and data structures
#[derive(Debug)]
pub struct Profiler {
    /// Operation profiles indexed by name
    operation_profiles: Arc<RwLock<HashMap<String, OperationProfile>>>,
    /// Global profiling statistics
    global_stats: Arc<Mutex<GlobalProfilingStats>>,
    /// Configuration
    config: ProfilerConfig,
    /// Session start time
    session_start: Instant,
}

/// Configuration for profiler behavior
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Whether to enable detailed timing
    pub enable_timing: bool,
    /// Whether to track memory allocations
    pub track_memory: bool,
    /// Maximum number of operation profiles to keep
    pub max_profiles: usize,
    /// Whether to automatically clean up old profiles
    pub auto_cleanup: bool,
    /// Cleanup interval in seconds
    pub cleanup_interval_secs: u64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_timing: true,
            track_memory: true,
            max_profiles: 1000,
            auto_cleanup: true,
            cleanup_interval_secs: 300, // 5 minutes
        }
    }
}

/// Profile information for a specific operation
#[derive(Debug, Clone)]
pub struct OperationProfile {
    /// Operation name
    pub name: String,
    /// Number of times operation was called
    pub call_count: u64,
    /// Total time spent in operation
    pub total_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Average execution time
    pub avg_time: Duration,
    /// Memory allocations during operation
    pub total_memory_allocated: usize,
    /// Peak memory usage during operation
    pub peak_memory_usage: usize,
    /// Last execution time
    pub last_execution: Instant,
    /// Error count
    pub error_count: u64,
}

impl OperationProfile {
    fn new(name: String) -> Self {
        let now = Instant::now();
        Self {
            name,
            call_count: 0,
            total_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            total_memory_allocated: 0,
            peak_memory_usage: 0,
            last_execution: now,
            error_count: 0,
        }
    }

    fn update_timing(&mut self, duration: Duration) {
        self.call_count += 1;
        self.total_time += duration;
        
        if duration < self.min_time {
            self.min_time = duration;
        }
        
        if duration > self.max_time {
            self.max_time = duration;
        }
        
        self.avg_time = self.total_time / self.call_count as u32;
        self.last_execution = Instant::now();
    }

    fn update_memory(&mut self, allocated: usize, peak: usize) {
        self.total_memory_allocated += allocated;
        if peak > self.peak_memory_usage {
            self.peak_memory_usage = peak;
        }
    }

    fn record_error(&mut self) {
        self.error_count += 1;
    }
}

/// Global profiling statistics
#[derive(Debug, Default, Clone)]
pub struct GlobalProfilingStats {
    pub total_operations: u64,
    pub total_time: Duration,
    pub total_memory_allocated: usize,
    pub peak_memory_usage: usize,
    pub error_count: u64,
    pub session_duration: Duration,
}

impl Profiler {
    /// Create new profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            operation_profiles: Arc::new(RwLock::new(HashMap::new())),
            global_stats: Arc::new(Mutex::new(GlobalProfilingStats::default())),
            config,
            session_start: Instant::now(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(ProfilerConfig::default())
    }

    /// Start profiling an operation
    pub fn start_operation(&self, operation_name: &str) -> ProfiledOperation {
        ProfiledOperation::new(operation_name.to_string(), self.clone())
    }

    /// Record operation completion
    pub fn record_operation(
        &self,
        operation_name: &str,
        duration: Duration,
        memory_allocated: Option<usize>,
        memory_peak: Option<usize>,
        success: bool,
    ) -> Result<(), ZiporaError> {
        // Update operation profile
        {
            let mut profiles = self.operation_profiles.write().map_err(|_| {
                ZiporaError::system_error("Failed to acquire write lock on operation profiles")
            })?;

            let profile = profiles.entry(operation_name.to_string())
                .or_insert_with(|| OperationProfile::new(operation_name.to_string()));

            profile.update_timing(duration);
            
            if let (Some(allocated), Some(peak)) = (memory_allocated, memory_peak) {
                profile.update_memory(allocated, peak);
            }
            
            if !success {
                profile.record_error();
            }
        }

        // Update global statistics
        {
            let mut global = self.global_stats.lock().map_err(|_| {
                ZiporaError::system_error("Failed to acquire lock on global stats")
            })?;

            global.total_operations += 1;
            global.total_time += duration;
            
            if let Some(allocated) = memory_allocated {
                global.total_memory_allocated += allocated;
            }
            
            if let Some(peak) = memory_peak {
                if peak > global.peak_memory_usage {
                    global.peak_memory_usage = peak;
                }
            }
            
            if !success {
                global.error_count += 1;
            }
            
            global.session_duration = self.session_start.elapsed();
        }

        Ok(())
    }

    /// Get profile for specific operation
    pub fn get_operation_profile(&self, operation_name: &str) -> Result<Option<OperationProfile>, ZiporaError> {
        let profiles = self.operation_profiles.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on operation profiles")
        })?;

        Ok(profiles.get(operation_name).cloned())
    }

    /// Get all operation names
    pub fn get_operation_names(&self) -> Result<Vec<String>, ZiporaError> {
        let profiles = self.operation_profiles.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on operation profiles")
        })?;

        Ok(profiles.keys().cloned().collect())
    }

    /// Get global statistics
    pub fn get_global_stats(&self) -> Result<GlobalProfilingStats, ZiporaError> {
        let mut global = self.global_stats.lock().map_err(|_| {
            ZiporaError::system_error("Failed to acquire lock on global stats")
        })?;

        global.session_duration = self.session_start.elapsed();
        Ok(global.clone())
    }

    /// Profile a data structure's memory usage
    pub fn profile_memory<T: MemorySize>(&self, name: &str, object: &T) -> Result<(), ZiporaError> {
        let breakdown = object.detailed_mem_size();
        
        self.record_operation(
            &format!("memory_profile_{}", name),
            Duration::ZERO, // No timing for memory profiling
            Some(breakdown.total),
            Some(breakdown.total),
            true,
        )
    }

    /// Integrate with TrieStatistics
    pub fn integrate_trie_stats(&self, trie_stats: &TrieStatistics) -> Result<(), ZiporaError> {
        // Extract timing information
        let timing_ns = trie_stats.timing.total_runtime_ns.load(std::sync::atomic::Ordering::Relaxed);
        let timing_duration = Duration::from_nanos(timing_ns);

        // Extract performance information
        let total_ops = trie_stats.performance.total_operations.load(std::sync::atomic::Ordering::Relaxed);
        let failed_ops = trie_stats.performance.failed_operations.load(std::sync::atomic::Ordering::Relaxed);

        // Extract memory information
        let total_memory = trie_stats.memory.total_allocated.load(std::sync::atomic::Ordering::Relaxed);
        let peak_memory = trie_stats.memory.peak_memory.load(std::sync::atomic::Ordering::Relaxed);

        // Record integrated statistics
        self.record_operation(
            "trie_operations",
            timing_duration,
            Some(total_memory),
            Some(peak_memory),
            failed_ops == 0,
        )?;

        // Record specific operation types
        let inserts = trie_stats.performance.insert_count.load(std::sync::atomic::Ordering::Relaxed);
        let lookups = trie_stats.performance.lookup_count.load(std::sync::atomic::Ordering::Relaxed);
        let deletes = trie_stats.performance.delete_count.load(std::sync::atomic::Ordering::Relaxed);

        if inserts > 0 {
            let insert_time = if total_ops > 0 {
                timing_duration / total_ops as u32
            } else {
                Duration::ZERO
            };
            
            for _ in 0..inserts {
                self.record_operation("trie_insert", insert_time, None, None, true)?;
            }
        }

        if lookups > 0 {
            let lookup_time = if total_ops > 0 {
                timing_duration / total_ops as u32
            } else {
                Duration::ZERO
            };
            
            for _ in 0..lookups {
                self.record_operation("trie_lookup", lookup_time, None, None, true)?;
            }
        }

        if deletes > 0 {
            let delete_time = if total_ops > 0 {
                timing_duration / total_ops as u32
            } else {
                Duration::ZERO
            };
            
            for _ in 0..deletes {
                self.record_operation("trie_delete", delete_time, None, None, true)?;
            }
        }

        Ok(())
    }

    /// Generate comprehensive profiling report
    pub fn generate_report(&self) -> Result<String, ZiporaError> {
        let profiles = self.operation_profiles.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on operation profiles")
        })?;

        let global = self.get_global_stats()?;

        let mut report = String::from("=== Profiling Report ===\n\n");

        // Global statistics
        report.push_str("Global Statistics:\n");
        report.push_str(&format!("  Session Duration: {:.2}s\n", global.session_duration.as_secs_f64()));
        report.push_str(&format!("  Total Operations: {}\n", global.total_operations));
        report.push_str(&format!("  Total Time: {:.2}s\n", global.total_time.as_secs_f64()));
        report.push_str(&format!("  Total Memory Allocated: {} bytes\n", global.total_memory_allocated));
        report.push_str(&format!("  Peak Memory Usage: {} bytes\n", global.peak_memory_usage));
        report.push_str(&format!("  Error Count: {}\n", global.error_count));
        
        if global.total_operations > 0 {
            let avg_time = global.total_time.as_secs_f64() / global.total_operations as f64;
            let error_rate = (global.error_count as f64 / global.total_operations as f64) * 100.0;
            report.push_str(&format!("  Average Operation Time: {:.6}s\n", avg_time));
            report.push_str(&format!("  Error Rate: {:.2}%\n", error_rate));
        }
        
        report.push_str("\n");

        // Operation profiles (sorted by total time)
        if !profiles.is_empty() {
            report.push_str("Operation Profiles:\n");
            
            let mut sorted_profiles: Vec<_> = profiles.values().collect();
            sorted_profiles.sort_by(|a, b| b.total_time.cmp(&a.total_time));

            for profile in sorted_profiles.iter().take(20) { // Top 20 operations
                report.push_str(&format!("  {}:\n", profile.name));
                report.push_str(&format!("    Calls: {}\n", profile.call_count));
                report.push_str(&format!("    Total Time: {:.6}s\n", profile.total_time.as_secs_f64()));
                report.push_str(&format!("    Avg Time: {:.6}s\n", profile.avg_time.as_secs_f64()));
                report.push_str(&format!("    Min Time: {:.6}s\n", profile.min_time.as_secs_f64()));
                report.push_str(&format!("    Max Time: {:.6}s\n", profile.max_time.as_secs_f64()));
                
                if profile.total_memory_allocated > 0 {
                    report.push_str(&format!("    Memory Allocated: {} bytes\n", profile.total_memory_allocated));
                    report.push_str(&format!("    Peak Memory: {} bytes\n", profile.peak_memory_usage));
                }
                
                if profile.error_count > 0 {
                    let error_rate = (profile.error_count as f64 / profile.call_count as f64) * 100.0;
                    report.push_str(&format!("    Errors: {} ({:.2}%)\n", profile.error_count, error_rate));
                }
                
                report.push_str("\n");
            }
        }

        Ok(report)
    }

    /// Clear all profiling data
    pub fn clear(&self) -> Result<(), ZiporaError> {
        {
            let mut profiles = self.operation_profiles.write().map_err(|_| {
                ZiporaError::system_error("Failed to acquire write lock on operation profiles")
            })?;
            profiles.clear();
        }

        {
            let mut global = self.global_stats.lock().map_err(|_| {
                ZiporaError::system_error("Failed to acquire lock on global stats")
            })?;
            *global = GlobalProfilingStats::default();
        }

        Ok(())
    }

    /// Cleanup old profiles
    pub fn cleanup_old_profiles(&self) -> Result<usize, ZiporaError> {
        let cleanup_threshold = Duration::from_secs(self.config.cleanup_interval_secs);
        let now = Instant::now();
        let mut removed_count = 0;

        {
            let mut profiles = self.operation_profiles.write().map_err(|_| {
                ZiporaError::system_error("Failed to acquire write lock on operation profiles")
            })?;

            let initial_count = profiles.len();
            profiles.retain(|_, profile| {
                now.duration_since(profile.last_execution) < cleanup_threshold
            });
            removed_count = initial_count - profiles.len();

            // Also enforce max_profiles limit
            if profiles.len() > self.config.max_profiles {
                // Remove least recently used profiles
                let mut profile_vec: Vec<_> = profiles.drain().collect();
                profile_vec.sort_by(|a, b| b.1.last_execution.cmp(&a.1.last_execution));
                profile_vec.truncate(self.config.max_profiles);
                
                *profiles = profile_vec.into_iter().collect();
            }
        }

        Ok(removed_count)
    }
}

impl Clone for Profiler {
    fn clone(&self) -> Self {
        Self {
            operation_profiles: self.operation_profiles.clone(),
            global_stats: self.global_stats.clone(),
            config: self.config.clone(),
            session_start: self.session_start,
        }
    }
}

/// RAII wrapper for automatic operation profiling
pub struct ProfiledOperation {
    operation_name: String,
    profiler: Profiler,
    start_time: Instant,
    memory_start: Option<usize>,
}

impl ProfiledOperation {
    fn new(operation_name: String, profiler: Profiler) -> Self {
        Self {
            operation_name,
            profiler,
            start_time: Instant::now(),
            memory_start: None, // Could be enhanced to track actual memory
        }
    }

    /// Set memory tracking for this operation
    pub fn track_memory(&mut self, current_memory: usize) {
        self.memory_start = Some(current_memory);
    }

    /// Complete operation with success status
    pub fn complete(self, success: bool) -> Result<(), ZiporaError> {
        let duration = self.start_time.elapsed();
        
        self.profiler.record_operation(
            &self.operation_name,
            duration,
            self.memory_start,
            self.memory_start, // Simplified: would need proper peak tracking
            success,
        )
    }

    /// Complete operation with error
    pub fn complete_with_error(self) -> Result<(), ZiporaError> {
        self.complete(false)
    }
}

impl Drop for ProfiledOperation {
    fn drop(&mut self) {
        // Auto-complete with success if not manually completed
        let duration = self.start_time.elapsed();
        let _ = self.profiler.record_operation(
            &self.operation_name,
            duration,
            self.memory_start,
            self.memory_start,
            true, // Assume success if not specified
        );
    }
}

/// Macro for easy operation profiling
#[macro_export]
macro_rules! profile_operation {
    ($profiler:expr, $operation_name:expr, $code:block) => {{
        let _profiled_op = $profiler.start_operation($operation_name);
        let result = $code;
        result
    }};
}

/// Global profiler instance for convenience
static GLOBAL_PROFILER: std::sync::OnceLock<Profiler> = std::sync::OnceLock::new();

/// Get or initialize global profiler
pub fn global_profiler() -> &'static Profiler {
    GLOBAL_PROFILER.get_or_init(|| Profiler::default())
}

/// Initialize global profiler with custom config
pub fn init_global_profiler(config: ProfilerConfig) -> Result<(), ZiporaError> {
    GLOBAL_PROFILER.set(Profiler::new(config))
        .map_err(|_| ZiporaError::system_error("Global profiler already initialized"))
}

// Simple MemorySize implementation for Vec<u8> for testing
impl MemorySize for Vec<u8> {
    fn mem_size(&self) -> usize {
        std::mem::size_of::<Vec<u8>>() + self.len()
    }

    fn detailed_mem_size(&self) -> MemoryBreakdown {
        let mut breakdown = MemoryBreakdown::new();
        breakdown.add_component("vec_overhead", std::mem::size_of::<Vec<u8>>());
        breakdown.add_component("data", self.len());
        breakdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_profiler_basic() {
        let profiler = Profiler::default();
        
        profiler.record_operation(
            "test_op",
            Duration::from_millis(10),
            Some(1000),
            Some(1500),
            true,
        ).unwrap();

        let profile = profiler.get_operation_profile("test_op").unwrap().unwrap();
        assert_eq!(profile.call_count, 1);
        assert_eq!(profile.total_memory_allocated, 1000);
        assert_eq!(profile.peak_memory_usage, 1500);
    }

    #[test]
    fn test_profiled_operation() {
        let profiler = Profiler::default();
        
        {
            let _op = profiler.start_operation("test_scoped");
            thread::sleep(Duration::from_millis(1));
        } // Automatically completed here

        let profile = profiler.get_operation_profile("test_scoped").unwrap().unwrap();
        assert_eq!(profile.call_count, 1);
        assert!(profile.total_time >= Duration::from_millis(1));
    }

    #[test]
    fn test_operation_statistics() {
        let profiler = Profiler::default();
        
        // Record multiple operations
        for i in 0..10 {
            profiler.record_operation(
                "repeated_op",
                Duration::from_millis(i),
                Some(i as usize * 100),
                Some(i as usize * 100),
                i % 3 != 0, // Some failures
            ).unwrap();
        }

        let profile = profiler.get_operation_profile("repeated_op").unwrap().unwrap();
        assert_eq!(profile.call_count, 10);
        assert!(profile.error_count > 0);
        assert!(profile.min_time < profile.max_time);
    }

    #[test]
    fn test_global_statistics() {
        let profiler = Profiler::default();
        
        profiler.record_operation("op1", Duration::from_millis(5), Some(500), Some(500), true).unwrap();
        profiler.record_operation("op2", Duration::from_millis(3), Some(300), Some(300), false).unwrap();

        let global = profiler.get_global_stats().unwrap();
        assert_eq!(global.total_operations, 2);
        assert_eq!(global.error_count, 1);
        assert_eq!(global.total_memory_allocated, 800);
        assert_eq!(global.peak_memory_usage, 500);
    }

    #[test]
    fn test_memory_profiling() {
        let profiler = Profiler::default();
        let test_vec = vec![1u8; 1000];
        
        profiler.profile_memory("test_vector", &test_vec).unwrap();
        
        let profile = profiler.get_operation_profile("memory_profile_test_vector").unwrap().unwrap();
        assert_eq!(profile.call_count, 1);
        assert!(profile.total_memory_allocated > 0);
    }

    #[test]
    fn test_profiler_cleanup() {
        let config = ProfilerConfig {
            max_profiles: 5,
            cleanup_interval_secs: 0, // Immediate cleanup
            ..Default::default()
        };
        
        let profiler = Profiler::new(config);
        
        // Add more profiles than the limit
        for i in 0..10 {
            profiler.record_operation(
                &format!("op_{}", i),
                Duration::from_millis(1),
                None,
                None,
                true,
            ).unwrap();
        }

        let names_before = profiler.get_operation_names().unwrap();
        assert_eq!(names_before.len(), 10);

        thread::sleep(Duration::from_millis(1)); // Ensure time passes
        let _removed = profiler.cleanup_old_profiles().unwrap();

        let names_after = profiler.get_operation_names().unwrap();
        assert!(names_after.len() <= 5);
    }

    #[test]
    fn test_report_generation() {
        let profiler = Profiler::default();
        
        profiler.record_operation("fast_op", Duration::from_micros(100), Some(100), Some(100), true).unwrap();
        profiler.record_operation("slow_op", Duration::from_millis(10), Some(1000), Some(1000), true).unwrap();
        profiler.record_operation("error_op", Duration::from_millis(1), None, None, false).unwrap();

        let report = profiler.generate_report().unwrap();
        
        assert!(report.contains("Global Statistics"));
        assert!(report.contains("Operation Profiles"));
        assert!(report.contains("fast_op"));
        assert!(report.contains("slow_op"));
        assert!(report.contains("error_op"));
    }

    #[test]
    fn test_global_profiler() {
        let profiler = global_profiler();
        
        profiler.record_operation(
            "global_test",
            Duration::from_millis(1),
            None,
            None,
            true,
        ).unwrap();

        let profile = profiler.get_operation_profile("global_test").unwrap().unwrap();
        assert_eq!(profile.call_count, 1);
    }

    #[test]
    fn test_profiler_clear() {
        let profiler = Profiler::default();
        
        profiler.record_operation("test", Duration::from_millis(1), None, None, true).unwrap();
        assert!(!profiler.get_operation_names().unwrap().is_empty());

        profiler.clear().unwrap();
        assert!(profiler.get_operation_names().unwrap().is_empty());

        let global = profiler.get_global_stats().unwrap();
        assert_eq!(global.total_operations, 0);
    }
}