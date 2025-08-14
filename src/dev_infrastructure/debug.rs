//! Comprehensive Debugging Framework
//!
//! Advanced debugging utilities and helpers for development and production environments.
//! Provides high-precision timing, memory debugging, performance profiling, and flexible
//! logging infrastructure. Inspired by production debugging frameworks while leveraging
//! Rust's compile-time optimization and zero-cost abstractions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crate::error::{ZiporaError, Result};

/// High-precision timer for performance measurements
#[derive(Debug, Clone)]
pub struct HighPrecisionTimer {
    name: String,
    start_time: Instant,
    start_system_time: SystemTime,
}

impl HighPrecisionTimer {
    /// Create a new timer with a name
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start_time: Instant::now(),
            start_system_time: SystemTime::now(),
        }
    }

    /// Create a new anonymous timer
    pub fn new() -> Self {
        Self::named("timer")
    }

    /// Get elapsed time since timer creation
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get elapsed time in nanoseconds
    pub fn elapsed_nanos(&self) -> u128 {
        self.elapsed().as_nanos()
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_micros(&self) -> u128 {
        self.elapsed().as_micros()
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_millis(&self) -> u128 {
        self.elapsed().as_millis()
    }

    /// Get elapsed time in seconds as floating point
    pub fn elapsed_secs_f64(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    /// Restart the timer
    pub fn restart(&mut self) {
        self.start_time = Instant::now();
        self.start_system_time = SystemTime::now();
    }

    /// Print elapsed time with automatic unit selection
    pub fn print_elapsed(&self) {
        let duration = self.elapsed();
        let formatted = format_duration(duration);
        println!("{}: {}", self.name, formatted);
    }

    /// Print elapsed time with custom message
    pub fn print_elapsed_with(&self, message: &str) {
        let duration = self.elapsed();
        let formatted = format_duration(duration);
        println!("{}: {} ({})", self.name, message, formatted);
    }

    /// Get elapsed time as a formatted string with automatic unit selection
    pub fn elapsed_string(&self) -> String {
        format_duration(self.elapsed())
    }
}

impl Default for HighPrecisionTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Format duration with automatic unit selection for readability
pub fn format_duration(duration: Duration) -> String {
    let nanos = duration.as_nanos();
    
    if nanos < 1_000 {
        format!("{}ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.3}μs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.3}ms", nanos as f64 / 1_000_000.0)
    } else if nanos < 60_000_000_000 {
        format!("{:.3}s", nanos as f64 / 1_000_000_000.0)
    } else {
        format!("{:.2}m", duration.as_secs_f64() / 60.0)
    }
}

/// Scoped timer that automatically prints elapsed time when dropped
pub struct ScopedTimer {
    timer: HighPrecisionTimer,
    message: Option<String>,
}

impl ScopedTimer {
    /// Create a new scoped timer with a name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            timer: HighPrecisionTimer::named(name),
            message: None,
        }
    }

    /// Create a new scoped timer with a custom completion message
    pub fn with_message(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            timer: HighPrecisionTimer::named(name),
            message: Some(message.into()),
        }
    }

    /// Get the underlying timer
    pub fn timer(&self) -> &HighPrecisionTimer {
        &self.timer
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if let Some(ref message) = self.message {
            self.timer.print_elapsed_with(message);
        } else {
            self.timer.print_elapsed();
        }
    }
}

/// Benchmark suite for performance testing
pub struct BenchmarkSuite {
    name: String,
    benchmarks: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            benchmarks: Vec::new(),
        }
    }

    /// Add a benchmark and run it
    pub fn add_benchmark<F>(&mut self, name: &str, iterations: usize, mut operation: F)
    where
        F: FnMut(),
    {
        let _timer = HighPrecisionTimer::named(name);
        
        // Warmup
        for _ in 0..std::cmp::min(iterations / 10, 100) {
            operation();
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        let duration = start.elapsed();
        
        let result = BenchmarkResult {
            name: name.to_string(),
            iterations,
            total_duration: duration,
            avg_duration: duration / iterations as u32,
            ops_per_sec: if duration.as_secs_f64() > 0.0 {
                iterations as f64 / duration.as_secs_f64()
            } else {
                f64::INFINITY
            },
        };
        
        self.benchmarks.push(result);
    }

    /// Run all benchmarks and print results
    pub fn run_all(&self) {
        println!("Benchmark Suite: {}", self.name);
        println!("{:-<80}", "");
        println!("{:<30} {:>12} {:>15} {:>15}", "Name", "Iterations", "Avg Time", "Ops/sec");
        println!("{:-<80}", "");
        
        for benchmark in &self.benchmarks {
            println!("{:<30} {:>12} {:>15} {:>15.0}",
                benchmark.name,
                benchmark.iterations,
                format_duration(benchmark.avg_duration),
                benchmark.ops_per_sec
            );
        }
        println!("{:-<80}", "");
    }

    /// Get benchmark results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.benchmarks
    }
}

/// Individual benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub ops_per_sec: f64,
}

/// Memory debugging utilities
pub struct MemoryDebugger {
    allocations: Arc<Mutex<HashMap<usize, AllocationInfo>>>,
    total_allocated: AtomicU64,
    total_deallocated: AtomicU64,
    peak_usage: AtomicU64,
    allocation_count: AtomicUsize,
}

#[derive(Debug, Clone)]
struct AllocationInfo {
    size: usize,
    timestamp: Instant,
    location: String,
}

impl MemoryDebugger {
    /// Create a new memory debugger
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: AtomicU64::new(0),
            total_deallocated: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    /// Record an allocation (for custom allocators)
    pub fn record_allocation(&self, ptr: usize, size: usize, location: &str) {
        let info = AllocationInfo {
            size,
            timestamp: Instant::now(),
            location: location.to_string(),
        };

        if let Ok(mut allocations) = self.allocations.lock() {
            allocations.insert(ptr, info);
        }

        let current_allocated = self.total_allocated.fetch_add(size as u64, Ordering::SeqCst) + size as u64;
        self.allocation_count.fetch_add(1, Ordering::SeqCst);

        // Update peak usage
        let current_peak = self.peak_usage.load(Ordering::SeqCst);
        if current_allocated > current_peak {
            self.peak_usage.store(current_allocated, Ordering::SeqCst);
        }
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, ptr: usize) -> Option<usize> {
        let size = if let Ok(mut allocations) = self.allocations.lock() {
            allocations.remove(&ptr).map(|info| info.size)
        } else {
            None
        };

        if let Some(size) = size {
            self.total_deallocated.fetch_add(size as u64, Ordering::SeqCst);
            Some(size)
        } else {
            None
        }
    }

    /// Get current memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        let allocated = self.total_allocated.load(Ordering::SeqCst);
        let deallocated = self.total_deallocated.load(Ordering::SeqCst);
        let current_usage = allocated.saturating_sub(deallocated);
        let peak_usage = self.peak_usage.load(Ordering::SeqCst);
        let allocation_count = self.allocation_count.load(Ordering::SeqCst);
        
        let active_allocations = if let Ok(allocations) = self.allocations.lock() {
            allocations.len()
        } else {
            0
        };

        MemoryStats {
            total_allocated: allocated,
            total_deallocated: deallocated,
            current_usage,
            peak_usage,
            allocation_count,
            active_allocations,
        }
    }

    /// Print memory usage report
    pub fn print_report(&self) {
        let stats = self.get_stats();
        println!("Memory Usage Report:");
        println!("  Total Allocated: {} bytes", stats.total_allocated);
        println!("  Total Deallocated: {} bytes", stats.total_deallocated);
        println!("  Current Usage: {} bytes", stats.current_usage);
        println!("  Peak Usage: {} bytes", stats.peak_usage);
        println!("  Allocation Count: {}", stats.allocation_count);
        println!("  Active Allocations: {}", stats.active_allocations);
    }

    /// Check for memory leaks
    pub fn check_leaks(&self) -> Vec<(usize, AllocationInfo)> {
        if let Ok(allocations) = self.allocations.lock() {
            allocations.iter().map(|(&ptr, info)| (ptr, info.clone())).collect()
        } else {
            Vec::new()
        }
    }
}

impl Default for MemoryDebugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: u64,
    pub total_deallocated: u64,
    pub current_usage: u64,
    pub peak_usage: u64,
    pub allocation_count: usize,
    pub active_allocations: usize,
}

/// Performance profiler for tracking operation performance
pub struct PerformanceProfiler {
    profiles: Arc<RwLock<HashMap<String, ProfileData>>>,
}

#[derive(Debug, Clone)]
struct ProfileData {
    call_count: u64,
    total_duration: Duration,
    min_duration: Duration,
    max_duration: Duration,
    avg_duration: Duration,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Profile a function call
    pub fn profile<T, F>(&self, name: &str, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();

        self.record_measurement(name, duration)?;
        result
    }

    /// Record a measurement manually
    pub fn record_measurement(&self, name: &str, duration: Duration) -> Result<()> {
        let mut profiles = self.profiles.write()
            .map_err(|_| ZiporaError::io_error("Failed to acquire write lock on profiles"))?;

        let profile = profiles.entry(name.to_string()).or_insert_with(|| ProfileData {
            call_count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            avg_duration: Duration::ZERO,
        });

        profile.call_count += 1;
        profile.total_duration += duration;
        profile.min_duration = profile.min_duration.min(duration);
        profile.max_duration = profile.max_duration.max(duration);
        profile.avg_duration = profile.total_duration / profile.call_count as u32;

        Ok(())
    }

    /// Get profile data for a specific operation
    pub fn get_profile(&self, name: &str) -> Result<Option<ProfileData>> {
        let profiles = self.profiles.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock on profiles"))?;
        
        Ok(profiles.get(name).cloned())
    }

    /// Print performance report
    pub fn print_report(&self) -> Result<()> {
        let profiles = self.profiles.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock on profiles"))?;

        println!("Performance Profiling Report:");
        println!("{:-<100}", "");
        println!("{:<30} {:>10} {:>15} {:>15} {:>15} {:>15}",
            "Operation", "Calls", "Total", "Average", "Min", "Max");
        println!("{:-<100}", "");

        for (name, profile) in profiles.iter() {
            println!("{:<30} {:>10} {:>15} {:>15} {:>15} {:>15}",
                name,
                profile.call_count,
                format_duration(profile.total_duration),
                format_duration(profile.avg_duration),
                format_duration(profile.min_duration),
                format_duration(profile.max_duration),
            );
        }
        println!("{:-<100}", "");

        Ok(())
    }

    /// Clear all profile data
    pub fn clear(&self) -> Result<()> {
        let mut profiles = self.profiles.write()
            .map_err(|_| ZiporaError::io_error("Failed to acquire write lock on profiles"))?;
        
        profiles.clear();
        Ok(())
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Debug assertion with custom message and optional panic
#[macro_export]
macro_rules! debug_assert_msg {
    ($cond:expr, $msg:expr) => {
        #[cfg(debug_assertions)]
        {
            if !$cond {
                eprintln!("Debug assertion failed: {}", $msg);
                eprintln!("  at {}:{}", file!(), line!());
                panic!("Debug assertion failed");
            }
        }
    };
    ($cond:expr, $msg:expr, no_panic) => {
        #[cfg(debug_assertions)]
        {
            if !$cond {
                eprintln!("Debug assertion failed: {}", $msg);
                eprintln!("  at {}:{}", file!(), line!());
            }
        }
    };
}

/// Conditional debug print macro
#[macro_export]
macro_rules! debug_print {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[DEBUG {}:{}] {}", file!(), line!(), format!($($arg)*));
        }
    };
}

/// Performance measurement macro
#[macro_export]
macro_rules! measure_time {
    ($name:expr, $code:block) => {{
        let _timer = $crate::dev_infrastructure::debug::ScopedTimer::new($name);
        $code
    }};
}

/// Global performance profiler instance
static GLOBAL_PROFILER: std::sync::LazyLock<PerformanceProfiler> = 
    std::sync::LazyLock::new(|| PerformanceProfiler::new());

/// Get the global performance profiler
pub fn global_profiler() -> &'static PerformanceProfiler {
    &GLOBAL_PROFILER
}

/// Global memory debugger instance
static GLOBAL_MEMORY_DEBUGGER: std::sync::LazyLock<MemoryDebugger> = 
    std::sync::LazyLock::new(|| MemoryDebugger::new());

/// Get the global memory debugger
pub fn global_memory_debugger() -> &'static MemoryDebugger {
    &GLOBAL_MEMORY_DEBUGGER
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_high_precision_timer() {
        let timer = HighPrecisionTimer::named("test_timer");
        thread::sleep(Duration::from_millis(10));
        
        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed < Duration::from_millis(50));
        
        let formatted = timer.elapsed_string();
        assert!(formatted.contains("ms"));
    }

    #[test]
    fn test_scoped_timer() {
        {
            let _timer = ScopedTimer::new("scoped_test");
            thread::sleep(Duration::from_millis(5));
        }
        // Timer should print elapsed time when dropped
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new("test_suite");
        
        suite.add_benchmark("simple_op", 1000, || {
            // Simulate some work
            let _x = (0..100).sum::<i32>();
        });
        
        assert_eq!(suite.results().len(), 1);
        let result = &suite.results()[0];
        assert_eq!(result.name, "simple_op");
        assert_eq!(result.iterations, 1000);
        assert!(result.ops_per_sec > 0.0);
    }

    #[test]
    fn test_memory_debugger() {
        let debugger = MemoryDebugger::new();
        
        // Record some allocations
        debugger.record_allocation(0x1000, 1024, "test:1");
        debugger.record_allocation(0x2000, 2048, "test:2");
        
        let stats = debugger.get_stats();
        assert_eq!(stats.total_allocated, 3072);
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.active_allocations, 2);
        
        // Record deallocation
        let size = debugger.record_deallocation(0x1000);
        assert_eq!(size, Some(1024));
        
        let stats = debugger.get_stats();
        assert_eq!(stats.total_deallocated, 1024);
        assert_eq!(stats.current_usage, 2048);
        assert_eq!(stats.active_allocations, 1);
    }

    #[test]
    fn test_performance_profiler() {
        let profiler = PerformanceProfiler::new();
        
        // Profile some operations
        let result = profiler.profile("test_op", || {
            thread::sleep(Duration::from_millis(1));
            Ok(42)
        }).unwrap();
        
        assert_eq!(result, 42);
        
        let profile = profiler.get_profile("test_op").unwrap().unwrap();
        assert_eq!(profile.call_count, 1);
        assert!(profile.total_duration >= Duration::from_millis(1));
        
        // Profile same operation again
        profiler.profile("test_op", || {
            thread::sleep(Duration::from_millis(1));
            Ok(24)
        }).unwrap();
        
        let profile = profiler.get_profile("test_op").unwrap().unwrap();
        assert_eq!(profile.call_count, 2);
    }

    #[test]
    fn test_format_duration() {
        assert!(format_duration(Duration::from_nanos(500)).contains("ns"));
        assert!(format_duration(Duration::from_micros(500)).contains("μs"));
        assert!(format_duration(Duration::from_millis(500)).contains("ms"));
        assert!(format_duration(Duration::from_secs(5)).contains("s"));
    }

    #[test]
    fn test_global_instances() {
        let profiler = global_profiler();
        let debugger = global_memory_debugger();
        
        // These should be the same instances across calls
        assert!(std::ptr::eq(profiler, global_profiler()));
        assert!(std::ptr::eq(debugger, global_memory_debugger()));
    }
}