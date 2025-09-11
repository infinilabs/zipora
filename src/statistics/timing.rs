//! High-Precision Timing Framework
//!
//! Provides nanosecond-resolution timing capabilities for comprehensive performance monitoring.
//! Based on sophisticated timing patterns from high-performance systems.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;

/// High-precision profiling timer with nanosecond resolution
#[derive(Debug)]
pub struct Profiling {
    #[cfg(target_os = "windows")]
    freq: i64,
    creation_time: Instant,
}

impl Profiling {
    pub fn new() -> Self {
        Self {
            #[cfg(target_os = "windows")]
            freq: Self::get_frequency(),
            creation_time: Instant::now(),
        }
    }

    #[cfg(target_os = "windows")]
    fn get_frequency() -> i64 {
        use std::mem;
        unsafe {
            let mut freq = mem::zeroed();
            winapi::um::profileapi::QueryPerformanceFrequency(&mut freq);
            freq
        }
    }

    /// Get current high-precision timestamp in nanoseconds
    pub fn now(&self) -> i64 {
        #[cfg(target_os = "windows")]
        {
            use std::mem;
            unsafe {
                let mut counter = mem::zeroed();
                winapi::um::profileapi::QueryPerformanceCounter(&mut counter);
                counter
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            self.creation_time.elapsed().as_nanos() as i64
        }
    }

    /// Convert timestamp to nanoseconds
    pub fn ns(&self, timestamp: i64) -> i64 {
        #[cfg(target_os = "windows")]
        {
            (timestamp * 1_000_000_000) / self.freq
        }
        #[cfg(not(target_os = "windows"))]
        {
            timestamp
        }
    }

    /// Convert timestamp to microseconds
    pub fn us(&self, timestamp: i64) -> i64 {
        self.ns(timestamp) / 1_000
    }

    /// Convert timestamp to milliseconds
    pub fn ms(&self, timestamp: i64) -> i64 {
        self.ns(timestamp) / 1_000_000
    }

    /// Convert timestamp difference to nanoseconds
    pub fn ns_between(&self, start: i64, end: i64) -> i64 {
        self.ns(end - start)
    }

    /// Convert timestamp difference to microseconds
    pub fn us_between(&self, start: i64, end: i64) -> i64 {
        self.us(end - start)
    }

    /// Convert timestamp difference to milliseconds
    pub fn ms_between(&self, start: i64, end: i64) -> i64 {
        self.ms(end - start)
    }

    /// Convert timestamp to floating-point nanoseconds
    pub fn nf(&self, timestamp: i64) -> f64 {
        self.ns(timestamp) as f64
    }

    /// Convert timestamp to floating-point microseconds
    pub fn uf(&self, timestamp: i64) -> f64 {
        self.ns(timestamp) as f64 / 1_000.0
    }

    /// Convert timestamp to floating-point milliseconds
    pub fn mf(&self, timestamp: i64) -> f64 {
        self.ns(timestamp) as f64 / 1_000_000.0
    }

    /// Convert timestamp to floating-point seconds
    pub fn sf(&self, timestamp: i64) -> f64 {
        self.ns(timestamp) as f64 / 1_000_000_000.0
    }

    /// Convert timestamp difference to floating-point nanoseconds
    pub fn nf_between(&self, start: i64, end: i64) -> f64 {
        self.nf(end - start)
    }

    /// Convert timestamp difference to floating-point microseconds
    pub fn uf_between(&self, start: i64, end: i64) -> f64 {
        self.uf(end - start)
    }

    /// Convert timestamp difference to floating-point milliseconds
    pub fn mf_between(&self, start: i64, end: i64) -> f64 {
        self.mf(end - start)
    }

    /// Convert timestamp difference to floating-point seconds
    pub fn sf_between(&self, start: i64, end: i64) -> f64 {
        self.sf(end - start)
    }
}

impl Default for Profiling {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick time measurement with high precision
#[derive(Debug, Clone, Copy)]
pub struct QTime {
    timestamp_ns: u64,
}

impl QTime {
    /// Create a new QTime with current timestamp
    pub fn now() -> Self {
        Self {
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        }
    }

    /// Create QTime from nanoseconds since epoch
    pub fn from_nanos(nanos: u64) -> Self {
        Self {
            timestamp_ns: nanos,
        }
    }

    /// Get nanoseconds between two times
    pub fn ns(&self, other: &QTime) -> i64 {
        (other.timestamp_ns as i64) - (self.timestamp_ns as i64)
    }

    /// Get microseconds between two times
    pub fn us(&self, other: &QTime) -> i64 {
        self.ns(other) / 1_000
    }

    /// Get milliseconds between two times
    pub fn ms(&self, other: &QTime) -> i64 {
        self.ns(other) / 1_000_000
    }

    /// Get floating-point nanoseconds between two times
    pub fn nf(&self, other: &QTime) -> f64 {
        self.ns(other) as f64
    }

    /// Get floating-point microseconds between two times
    pub fn uf(&self, other: &QTime) -> f64 {
        self.ns(other) as f64 / 1_000.0
    }

    /// Get floating-point milliseconds between two times
    pub fn mf(&self, other: &QTime) -> f64 {
        self.ns(other) as f64 / 1_000_000.0
    }

    /// Get floating-point seconds between two times
    pub fn sf(&self, other: &QTime) -> f64 {
        self.ns(other) as f64 / 1_000_000_000.0
    }

    /// Get floating-point minutes between two times
    pub fn minutes_f(&self, other: &QTime) -> f64 {
        self.sf(other) / 60.0
    }

    /// Get floating-point hours between two times
    pub fn hours_f(&self, other: &QTime) -> f64 {
        self.sf(other) / 3600.0
    }

    /// Get floating-point days between two times
    pub fn days_f(&self, other: &QTime) -> f64 {
        self.sf(other) / 86400.0
    }

    /// Create a duration from the difference
    pub fn duration(&self, other: &QTime) -> QDuration {
        QDuration {
            duration_ns: self.ns(other),
        }
    }
}

/// Duration measurement with conversion utilities
#[derive(Debug, Clone, Copy)]
pub struct QDuration {
    duration_ns: i64,
}

impl QDuration {
    /// Create duration from nanoseconds
    pub fn from_nanos(nanos: i64) -> Self {
        Self { duration_ns: nanos }
    }

    /// Get duration in nanoseconds
    pub fn ns(&self) -> i64 {
        self.duration_ns
    }

    /// Get duration in microseconds
    pub fn us(&self) -> i64 {
        self.duration_ns / 1_000
    }

    /// Get duration in milliseconds
    pub fn ms(&self) -> i64 {
        self.duration_ns / 1_000_000
    }

    /// Get duration as floating-point nanoseconds
    pub fn nf(&self) -> f64 {
        self.duration_ns as f64
    }

    /// Get duration as floating-point microseconds
    pub fn uf(&self) -> f64 {
        self.duration_ns as f64 / 1_000.0
    }

    /// Get duration as floating-point milliseconds
    pub fn mf(&self) -> f64 {
        self.duration_ns as f64 / 1_000_000.0
    }

    /// Get duration as floating-point seconds
    pub fn sf(&self) -> f64 {
        self.duration_ns as f64 / 1_000_000_000.0
    }

    /// Get duration as floating-point minutes
    pub fn minutes_f(&self) -> f64 {
        self.sf() / 60.0
    }

    /// Get duration as floating-point hours
    pub fn hours_f(&self) -> f64 {
        self.sf() / 3600.0
    }

    /// Get duration as floating-point days
    pub fn days_f(&self) -> f64 {
        self.sf() / 86400.0
    }
}

/// Performance timer for measuring operation durations
#[derive(Debug)]
pub struct PerfTimer {
    profiling: Profiling,
    start_time: Option<i64>,
    lap_times: Vec<i64>,
    total_time: AtomicU64,
    lap_count: AtomicU64,
}

impl PerfTimer {
    pub fn new() -> Self {
        Self {
            profiling: Profiling::new(),
            start_time: None,
            lap_times: Vec::new(),
            total_time: AtomicU64::new(0),
            lap_count: AtomicU64::new(0),
        }
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start_time = Some(self.profiling.now());
    }

    /// Stop timing and return duration in nanoseconds
    pub fn stop(&mut self) -> Option<i64> {
        if let Some(start) = self.start_time.take() {
            let end = self.profiling.now();
            let duration = self.profiling.ns_between(start, end);
            self.total_time.fetch_add(duration as u64, Ordering::Relaxed);
            self.lap_count.fetch_add(1, Ordering::Relaxed);
            Some(duration)
        } else {
            None
        }
    }

    /// Record lap time without stopping
    pub fn lap(&mut self) -> Option<i64> {
        if let Some(start) = self.start_time {
            let now = self.profiling.now();
            let duration = self.profiling.ns_between(start, now);
            self.lap_times.push(duration);
            Some(duration)
        } else {
            None
        }
    }

    /// Get total accumulated time in nanoseconds
    pub fn total_time_ns(&self) -> u64 {
        self.total_time.load(Ordering::Relaxed)
    }

    /// Get average time per operation in nanoseconds
    pub fn average_time_ns(&self) -> f64 {
        let total = self.total_time.load(Ordering::Relaxed) as f64;
        let count = self.lap_count.load(Ordering::Relaxed) as f64;
        if count > 0.0 {
            total / count
        } else {
            0.0
        }
    }

    /// Get lap times
    pub fn lap_times(&self) -> &[i64] {
        &self.lap_times
    }

    /// Reset timer
    pub fn reset(&mut self) {
        self.start_time = None;
        self.lap_times.clear();
        self.total_time.store(0, Ordering::Relaxed);
        self.lap_count.store(0, Ordering::Relaxed);
    }
}

impl Default for PerfTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// High-resolution timer collection for multiple operations
#[derive(Debug)]
pub struct TimerCollection {
    timers: HashMap<String, PerfTimer>,
    global_profiling: Profiling,
}

impl TimerCollection {
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            global_profiling: Profiling::new(),
        }
    }

    /// Get or create a timer by name
    pub fn timer(&mut self, name: &str) -> &mut PerfTimer {
        self.timers.entry(name.to_string()).or_insert_with(PerfTimer::new)
    }

    /// Start timing for a named operation
    pub fn start(&mut self, name: &str) {
        self.timer(name).start();
    }

    /// Stop timing for a named operation
    pub fn stop(&mut self, name: &str) -> Option<i64> {
        if let Some(timer) = self.timers.get_mut(name) {
            timer.stop()
        } else {
            None
        }
    }

    /// Record lap for a named operation
    pub fn lap(&mut self, name: &str) -> Option<i64> {
        if let Some(timer) = self.timers.get_mut(name) {
            timer.lap()
        } else {
            None
        }
    }

    /// Get statistics for a named timer
    pub fn stats(&self, name: &str) -> Option<TimerStats> {
        self.timers.get(name).map(|timer| TimerStats {
            total_time_ns: timer.total_time_ns(),
            average_time_ns: timer.average_time_ns(),
            operation_count: timer.lap_count.load(Ordering::Relaxed),
            lap_times: timer.lap_times().to_vec(),
        })
    }

    /// Get all timer names
    pub fn timer_names(&self) -> Vec<String> {
        self.timers.keys().cloned().collect()
    }

    /// Generate comprehensive timing report
    pub fn report(&self) -> String {
        let mut report = String::from("=== Timing Report ===\n");
        
        for (name, timer) in &self.timers {
            let total_ms = timer.total_time_ns() as f64 / 1_000_000.0;
            let avg_us = timer.average_time_ns() / 1_000.0;
            let count = timer.lap_count.load(Ordering::Relaxed);
            
            report.push_str(&format!(
                "Timer '{}': {} operations, {:.2}ms total, {:.2}μs average\n",
                name, count, total_ms, avg_us
            ));
        }
        
        report
    }

    /// Reset all timers
    pub fn reset(&mut self) {
        for timer in self.timers.values_mut() {
            timer.reset();
        }
    }
}

impl Default for TimerCollection {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer statistics summary
#[derive(Debug, Clone)]
pub struct TimerStats {
    pub total_time_ns: u64,
    pub average_time_ns: f64,
    pub operation_count: u64,
    pub lap_times: Vec<i64>,
}

impl TimerStats {
    /// Get minimum lap time
    pub fn min_time_ns(&self) -> Option<i64> {
        self.lap_times.iter().min().copied()
    }

    /// Get maximum lap time
    pub fn max_time_ns(&self) -> Option<i64> {
        self.lap_times.iter().max().copied()
    }

    /// Get median lap time
    pub fn median_time_ns(&self) -> Option<f64> {
        if self.lap_times.is_empty() {
            return None;
        }

        let mut sorted = self.lap_times.clone();
        sorted.sort_unstable();
        
        let len = sorted.len();
        if len % 2 == 0 {
            Some((sorted[len / 2 - 1] + sorted[len / 2]) as f64 / 2.0)
        } else {
            Some(sorted[len / 2] as f64)
        }
    }

    /// Calculate standard deviation of lap times
    pub fn std_deviation_ns(&self) -> f64 {
        if self.lap_times.len() <= 1 {
            return 0.0;
        }

        let mean = self.average_time_ns;
        let variance: f64 = self.lap_times
            .iter()
            .map(|&time| {
                let diff = time as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / (self.lap_times.len() - 1) as f64;

        variance.sqrt()
    }
}

/// Get current date/time as string for logging
pub fn str_date_time_now() -> String {
    use std::time::SystemTime;
    
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();
    
    // Convert to human-readable format
    // This is a simplified version - in production you might want to use chrono
    format!("timestamp_{}_{:09}", secs, nanos)
}

/// Scoped timer that automatically measures duration
pub struct ScopedTimer<'a> {
    timer: &'a mut PerfTimer,
}

impl<'a> ScopedTimer<'a> {
    pub fn new(timer: &'a mut PerfTimer) -> Self {
        timer.start();
        Self { timer }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        self.timer.stop();
    }
}

/// Macro for easy scoped timing
#[macro_export]
macro_rules! scoped_timer {
    ($timer:expr) => {
        let _scoped_timer = $crate::statistics::timing::ScopedTimer::new($timer);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_profiling_basic() {
        let profiling = Profiling::new();
        let start = profiling.now();
        
        thread::sleep(Duration::from_millis(10));
        
        let end = profiling.now();
        let duration_ns = profiling.ns_between(start, end);
        
        // Should be at least 10ms (10_000_000 ns)
        assert!(duration_ns >= 10_000_000);
        
        // Convert to other units
        let duration_us = profiling.us_between(start, end);
        let duration_ms = profiling.ms_between(start, end);
        
        assert!(duration_us >= 10_000);
        assert!(duration_ms >= 10);
    }

    #[test]
    fn test_qtime_operations() {
        let time1 = QTime::now();
        thread::sleep(Duration::from_millis(5));
        let time2 = QTime::now();
        
        let duration_ns = time1.ns(&time2);
        assert!(duration_ns >= 5_000_000); // At least 5ms
        
        let duration = time1.duration(&time2);
        assert!(duration.ms() >= 5);
    }

    #[test]
    fn test_perf_timer() {
        let mut timer = PerfTimer::new();
        
        timer.start();
        thread::sleep(Duration::from_millis(1));
        let duration = timer.stop().unwrap();
        
        assert!(duration >= 1_000_000); // At least 1ms
        assert_eq!(timer.lap_count.load(Ordering::Relaxed), 1);
        assert!(timer.total_time_ns() >= 1_000_000);
    }

    #[test]
    fn test_timer_collection() {
        let mut collection = TimerCollection::new();
        
        collection.start("test_op");
        thread::sleep(Duration::from_millis(1));
        let duration = collection.stop("test_op").unwrap();
        
        assert!(duration >= 1_000_000);
        
        let stats = collection.stats("test_op").unwrap();
        assert_eq!(stats.operation_count, 1);
        assert!(stats.total_time_ns >= 1_000_000);
    }

    #[test]
    fn test_timer_stats() {
        let mut timer = PerfTimer::new();
        
        // Record multiple operations
        for _ in 0..5 {
            timer.start();
            thread::sleep(Duration::from_millis(1));
            timer.stop();
        }
        
        let stats = TimerStats {
            total_time_ns: timer.total_time_ns(),
            average_time_ns: timer.average_time_ns(),
            operation_count: timer.lap_count.load(Ordering::Relaxed),
            lap_times: timer.lap_times().to_vec(),
        };
        
        assert_eq!(stats.operation_count, 5);
        assert!(stats.average_time_ns >= 1_000_000.0);
    }

    #[test]
    fn test_scoped_timer() {
        let mut timer = PerfTimer::new();
        
        {
            let _scoped = ScopedTimer::new(&mut timer);
            thread::sleep(Duration::from_millis(1));
        } // Timer stops here
        
        assert_eq!(timer.lap_count.load(Ordering::Relaxed), 1);
        assert!(timer.total_time_ns() >= 1_000_000);
    }

    #[test]
    fn test_qduration_conversions() {
        let duration = QDuration::from_nanos(1_500_000_000); // 1.5 seconds
        
        assert_eq!(duration.ns(), 1_500_000_000);
        assert_eq!(duration.us(), 1_500_000);
        assert_eq!(duration.ms(), 1_500);
        assert_eq!(duration.sf(), 1.5);
    }

    #[test]
    fn test_timer_statistics() {
        let lap_times = vec![1_000_000, 2_000_000, 1_500_000, 3_000_000, 2_500_000];
        let stats = TimerStats {
            total_time_ns: lap_times.iter().sum::<i64>() as u64,
            average_time_ns: lap_times.iter().sum::<i64>() as f64 / lap_times.len() as f64,
            operation_count: lap_times.len() as u64,
            lap_times: lap_times.clone(),
        };
        
        assert_eq!(stats.min_time_ns(), Some(1_000_000));
        assert_eq!(stats.max_time_ns(), Some(3_000_000));
        assert_eq!(stats.median_time_ns(), Some(2_000_000.0));
        
        let std_dev = stats.std_deviation_ns();
        assert!(std_dev > 0.0);
    }

    #[test]
    fn test_date_time_string() {
        let time_str = str_date_time_now();
        assert!(time_str.starts_with("timestamp_"));
        assert!(time_str.contains('_'));
    }
}