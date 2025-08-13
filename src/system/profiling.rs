//! # Performance Profiling Framework
//!
//! High-precision timing and benchmarking utilities for performance-critical code.
//! Inspired by production-grade profiling systems with minimal overhead design.

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Mutex;
use std::fmt;

/// High-precision timer with nanosecond accuracy
#[derive(Debug, Clone)]
pub struct HighPrecisionTimer {
    start: Instant,
    name: String,
}

impl HighPrecisionTimer {
    /// Create a new timer with a descriptive name
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            name: "timer".to_string(),
        }
    }

    /// Create a named timer
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            name: name.into(),
        }
    }

    /// Reset the timer
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }

    /// Get elapsed time in nanoseconds
    pub fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// Get elapsed time in microseconds  
    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Get elapsed time as floating-point seconds
    pub fn elapsed_sec_f64(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Get elapsed duration
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Print elapsed time with automatic unit selection
    pub fn print_elapsed(&self) {
        let elapsed_ns = self.elapsed_ns();
        let formatted = format_duration_auto(elapsed_ns);
        println!("[{}] Elapsed: {}", self.name, formatted);
    }
}

/// Performance timer that automatically tracks and reports timing
pub struct PerfTimer {
    timer: HighPrecisionTimer,
    auto_report: bool,
}

impl PerfTimer {
    /// Create a new performance timer with automatic reporting
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            timer: HighPrecisionTimer::named(name),
            auto_report: true,
        }
    }

    /// Create a silent timer (no automatic reporting)
    pub fn silent(name: impl Into<String>) -> Self {
        Self {
            timer: HighPrecisionTimer::named(name),
            auto_report: false,
        }
    }

    /// Get elapsed time in nanoseconds
    pub fn elapsed_ns(&self) -> u64 {
        self.timer.elapsed_ns()
    }

    /// Get elapsed time as Duration
    pub fn elapsed(&self) -> Duration {
        self.timer.elapsed()
    }

    /// Manually print the elapsed time
    pub fn report(&self) {
        self.timer.print_elapsed();
    }
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        if self.auto_report {
            self.timer.print_elapsed();
        }
    }
}

/// Profiled function wrapper that automatically times execution
pub struct ProfiledFunction<T> {
    name: String,
    func: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T> ProfiledFunction<T> {
    /// Create a new profiled function
    pub fn new<F>(name: impl Into<String>, func: F) -> Self 
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            func: Box::new(func),
        }
    }

    /// Execute the function with timing
    pub fn execute(&self) -> (T, Duration) {
        let timer = HighPrecisionTimer::named(&self.name);
        let result = (self.func)();
        let elapsed = timer.elapsed();
        (result, elapsed)
    }

    /// Execute the function and print timing
    pub fn execute_with_report(&self) -> T {
        let timer = HighPrecisionTimer::named(&self.name);
        let result = (self.func)();
        timer.print_elapsed();
        result
    }
}

/// Benchmark suite for systematic performance testing
pub struct BenchmarkSuite {
    name: String,
    benchmarks: Vec<BenchmarkCase>,
    results: Mutex<HashMap<String, BenchmarkResult>>,
}

/// Individual benchmark case
#[derive(Clone)]
pub struct BenchmarkCase {
    name: String,
    iterations: usize,
    warmup_iterations: usize,
    func: fn(),
}

/// Results from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub ops_per_sec: f64,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            benchmarks: Vec::new(),
            results: Mutex::new(HashMap::new()),
        }
    }

    /// Add a benchmark case
    pub fn add_benchmark(&mut self, name: impl Into<String>, iterations: usize, func: fn()) {
        self.benchmarks.push(BenchmarkCase {
            name: name.into(),
            iterations,
            warmup_iterations: iterations / 10, // 10% warmup
            func,
        });
    }

    /// Add a benchmark with custom warmup iterations
    pub fn add_benchmark_with_warmup(
        &mut self, 
        name: impl Into<String>, 
        iterations: usize, 
        warmup_iterations: usize,
        func: fn()
    ) {
        self.benchmarks.push(BenchmarkCase {
            name: name.into(),
            iterations,
            warmup_iterations,
            func,
        });
    }

    /// Run all benchmarks
    pub fn run_all(&self) {
        println!("Running benchmark suite: {}", self.name);
        println!("{:=^60}", "");

        for benchmark in &self.benchmarks {
            let result = self.run_single_benchmark(benchmark);
            
            // Store result
            if let Ok(mut results) = self.results.lock() {
                results.insert(benchmark.name.clone(), result.clone());
            }

            // Print result
            self.print_benchmark_result(&result);
        }

        println!("{:=^60}", "");
        println!("Benchmark suite completed: {}", self.name);
    }

    /// Run a single benchmark
    fn run_single_benchmark(&self, benchmark: &BenchmarkCase) -> BenchmarkResult {
        // Warmup phase
        for _ in 0..benchmark.warmup_iterations {
            (benchmark.func)();
        }

        // Actual benchmark
        let mut durations = Vec::with_capacity(benchmark.iterations);
        for _ in 0..benchmark.iterations {
            let timer = Instant::now();
            (benchmark.func)();
            durations.push(timer.elapsed());
        }

        // Calculate statistics
        let total_time: Duration = durations.iter().sum();
        let avg_time = total_time / benchmark.iterations as u32;
        let min_time = durations.iter().min().copied().unwrap_or(Duration::ZERO);
        let max_time = durations.iter().max().copied().unwrap_or(Duration::ZERO);
        
        let ops_per_sec = if avg_time.as_nanos() > 0 {
            1_000_000_000.0 / avg_time.as_nanos() as f64
        } else {
            f64::INFINITY
        };

        BenchmarkResult {
            name: benchmark.name.clone(),
            iterations: benchmark.iterations,
            total_time,
            avg_time,
            min_time,
            max_time,
            ops_per_sec,
        }
    }

    /// Print benchmark result
    fn print_benchmark_result(&self, result: &BenchmarkResult) {
        println!("{:<30} | {:>12} | {:>12} | {:>15}", 
                 result.name,
                 format_duration_auto(result.avg_time.as_nanos() as u64),
                 format_duration_auto(result.min_time.as_nanos() as u64),
                 format_ops_per_sec(result.ops_per_sec));
    }

    /// Get results for a specific benchmark
    pub fn get_result(&self, name: &str) -> Option<BenchmarkResult> {
        self.results.lock().ok()?.get(name).cloned()
    }

    /// Get all results
    pub fn get_all_results(&self) -> HashMap<String, BenchmarkResult> {
        self.results.lock().map(|guard| guard.clone()).unwrap_or_else(|_| HashMap::new())
    }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: avg={}, min={}, max={}, ops/sec={:.2}", 
               self.name,
               format_duration_auto(self.avg_time.as_nanos() as u64),
               format_duration_auto(self.min_time.as_nanos() as u64),
               format_duration_auto(self.max_time.as_nanos() as u64),
               self.ops_per_sec)
    }
}

/// Format duration with automatic unit selection
pub fn format_duration_auto(nanos: u64) -> String {
    if nanos < 1_000 {
        format!("{}ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.2}μs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2}ms", nanos as f64 / 1_000_000.0)
    } else if nanos < 60_000_000_000 {
        format!("{:.2}s", nanos as f64 / 1_000_000_000.0)
    } else {
        format!("{:.2}min", nanos as f64 / 60_000_000_000.0)
    }
}

/// Format operations per second with appropriate units
pub fn format_ops_per_sec(ops: f64) -> String {
    if ops >= 1_000_000_000.0 {
        format!("{:.2}G ops/s", ops / 1_000_000_000.0)
    } else if ops >= 1_000_000.0 {
        format!("{:.2}M ops/s", ops / 1_000_000.0)
    } else if ops >= 1_000.0 {
        format!("{:.2}K ops/s", ops / 1_000.0)
    } else {
        format!("{:.2} ops/s", ops)
    }
}

/// Macro for timing code blocks
#[macro_export]
macro_rules! time_block {
    ($name:expr, $block:block) => {
        {
            let _timer = $crate::system::profiling::PerfTimer::new($name);
            $block
        }
    };
}

/// Macro for timing expressions
#[macro_export]
macro_rules! time_expr {
    ($name:expr, $expr:expr) => {
        {
            let timer = $crate::system::profiling::HighPrecisionTimer::named($name);
            let result = $expr;
            timer.print_elapsed();
            result
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_high_precision_timer() {
        let timer = HighPrecisionTimer::new();
        thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed_ms();
        
        // Should be approximately 10ms (allow some variance)
        assert!(elapsed >= 8 && elapsed <= 15);
    }

    #[test]
    fn test_perf_timer() {
        {
            let _timer = PerfTimer::silent("test_timer");
            thread::sleep(Duration::from_millis(1));
        } // Timer should not print since it's silent
    }

    #[test]
    fn test_profiled_function() {
        let func = ProfiledFunction::new("test_function", || {
            thread::sleep(Duration::from_millis(1));
            42
        });

        let (result, duration) = func.execute();
        assert_eq!(result, 42);
        assert!(duration.as_millis() >= 1);
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new("test_suite");
        
        suite.add_benchmark("fast_operation", 1000, || {
            // Fast operation
            let _ = 1 + 1;
        });

        suite.add_benchmark("slow_operation", 10, || {
            // Slow operation
            thread::sleep(Duration::from_micros(100));
        });

        suite.run_all();

        // Check that results were stored
        let fast_result = suite.get_result("fast_operation").unwrap();
        let slow_result = suite.get_result("slow_operation").unwrap();

        assert!(fast_result.avg_time < slow_result.avg_time);
        assert!(fast_result.ops_per_sec > slow_result.ops_per_sec);
    }

    #[test]
    fn test_duration_formatting() {
        assert_eq!(format_duration_auto(500), "500ns");
        assert_eq!(format_duration_auto(1_500), "1.50μs");
        assert_eq!(format_duration_auto(2_500_000), "2.50ms");
        assert_eq!(format_duration_auto(3_500_000_000), "3.50s");
    }

    #[test]
    fn test_ops_per_sec_formatting() {
        assert_eq!(format_ops_per_sec(500.0), "500.00 ops/s");
        assert_eq!(format_ops_per_sec(1_500.0), "1.50K ops/s");
        assert_eq!(format_ops_per_sec(2_500_000.0), "2.50M ops/s");
        assert_eq!(format_ops_per_sec(3_500_000_000.0), "3.50G ops/s");
    }

    #[test]
    fn test_macros() {
        // Test timing macros (should compile and run)
        let result = time_expr!("test_expr", 42);
        assert_eq!(result, 42);

        time_block!("test_block", {
            let _ = 1 + 1;
        });
    }
}