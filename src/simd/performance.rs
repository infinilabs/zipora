//! # Performance Monitoring and History Tracking
//!
//! Provides continuous performance monitoring with exponential moving averages
//! and adaptive degradation detection for SIMD operations.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Individual performance sample
#[derive(Debug, Clone)]
pub struct Sample {
    /// Operations per second throughput
    pub throughput: f64,
    /// Timestamp when sample was recorded
    pub timestamp: Instant,
    /// Number of operations in this sample
    pub operations: u64,
}

/// Performance history with EMA tracking and degradation detection
#[derive(Debug, Clone)]
pub struct PerformanceHistory {
    /// Exponential moving average of throughput (ops/sec)
    ema_throughput: f64,

    /// Total operations executed
    total_operations: u64,

    /// Total duration
    total_duration: Duration,

    /// Recent samples (ring buffer, last N samples)
    recent_samples: VecDeque<Sample>,

    /// Performance degradation counter
    degradation_counter: u32,

    /// Maximum recent samples to keep
    max_recent_samples: usize,

    /// EMA smoothing factor (alpha)
    ema_alpha: f64,
}

impl PerformanceHistory {
    /// Create new performance history tracker
    pub fn new() -> Self {
        PerformanceHistory {
            ema_throughput: 0.0,
            total_operations: 0,
            total_duration: Duration::ZERO,
            recent_samples: VecDeque::with_capacity(100),
            degradation_counter: 0,
            max_recent_samples: 100,
            ema_alpha: 0.1,
        }
    }

    /// Create with custom parameters
    pub fn with_params(max_recent_samples: usize, ema_alpha: f64) -> Self {
        PerformanceHistory {
            ema_throughput: 0.0,
            total_operations: 0,
            total_duration: Duration::ZERO,
            recent_samples: VecDeque::with_capacity(max_recent_samples),
            degradation_counter: 0,
            max_recent_samples,
            ema_alpha,
        }
    }

    /// Record a new performance sample
    pub fn record_sample(&mut self, duration: Duration, operations: u64) {
        if duration.is_zero() || operations == 0 {
            return;
        }

        let throughput = operations as f64 / duration.as_secs_f64();

        // Update EMA
        if self.total_operations == 0 {
            // First sample, initialize EMA
            self.ema_throughput = throughput;
        } else {
            // Update EMA: EMA_new = alpha * value + (1 - alpha) * EMA_old
            self.ema_throughput =
                self.ema_alpha * throughput + (1.0 - self.ema_alpha) * self.ema_throughput;
        }

        // Add to recent samples
        self.recent_samples.push_back(Sample {
            throughput,
            timestamp: Instant::now(),
            operations,
        });

        // Limit recent samples
        while self.recent_samples.len() > self.max_recent_samples {
            self.recent_samples.pop_front();
        }

        // Update totals
        self.total_operations += operations;
        self.total_duration += duration;
    }

    /// Check for sustained performance degradation
    ///
    /// Returns true if performance has degraded below expected threshold
    /// for a sustained period (trigger_count operations).
    pub fn check_performance_degradation(
        &mut self,
        expected_throughput: f64,
        trigger_count: u32,
    ) -> bool {
        if expected_throughput <= 0.0 || self.total_operations == 0 {
            return false;
        }

        // Check if current EMA is significantly below expected throughput
        // Default threshold: 90% of expected performance
        const DEGRADATION_THRESHOLD: f64 = 0.9;

        if self.ema_throughput < expected_throughput * DEGRADATION_THRESHOLD {
            self.degradation_counter += 1;

            // Trigger re-benchmarking after sustained degradation
            if self.degradation_counter >= trigger_count {
                self.degradation_counter = 0; // Reset counter
                return true;
            }
        } else {
            // Performance is acceptable, reset counter
            self.degradation_counter = 0;
        }

        false
    }

    /// Get current EMA throughput
    pub fn ema_throughput(&self) -> f64 {
        self.ema_throughput
    }

    /// Get total operations executed
    pub fn total_operations(&self) -> u64 {
        self.total_operations
    }

    /// Get total duration
    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    /// Get average throughput over all samples
    pub fn average_throughput(&self) -> f64 {
        if self.total_duration.is_zero() {
            0.0
        } else {
            self.total_operations as f64 / self.total_duration.as_secs_f64()
        }
    }

    /// Get recent samples
    pub fn recent_samples(&self) -> &VecDeque<Sample> {
        &self.recent_samples
    }

    /// Get current degradation counter
    pub fn degradation_counter(&self) -> u32 {
        self.degradation_counter
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.ema_throughput = 0.0;
        self.total_operations = 0;
        self.total_duration = Duration::ZERO;
        self.recent_samples.clear();
        self.degradation_counter = 0;
    }

    /// Calculate variance of recent samples
    pub fn recent_variance(&self) -> f64 {
        if self.recent_samples.len() < 2 {
            return 0.0;
        }

        let mean: f64 = self
            .recent_samples
            .iter()
            .map(|s| s.throughput)
            .sum::<f64>()
            / self.recent_samples.len() as f64;

        let variance: f64 = self
            .recent_samples
            .iter()
            .map(|s| {
                let diff = s.throughput - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.recent_samples.len() as f64;

        variance
    }

    /// Calculate standard deviation of recent samples
    pub fn recent_std_dev(&self) -> f64 {
        self.recent_variance().sqrt()
    }

    /// Get percentile from recent samples
    pub fn recent_percentile(&self, percentile: f64) -> Option<f64> {
        if self.recent_samples.is_empty() {
            return None;
        }

        let mut throughputs: Vec<f64> = self.recent_samples.iter().map(|s| s.throughput).collect();
        throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((percentile / 100.0) * (throughputs.len() - 1) as f64).round() as usize;
        Some(throughputs[index])
    }
}

impl Default for PerformanceHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_history_creation() {
        let history = PerformanceHistory::new();
        assert_eq!(history.total_operations(), 0);
        assert_eq!(history.ema_throughput(), 0.0);
        assert_eq!(history.degradation_counter(), 0);
    }

    #[test]
    fn test_record_sample() {
        let mut history = PerformanceHistory::new();

        history.record_sample(Duration::from_secs(1), 1000);

        assert_eq!(history.total_operations(), 1000);
        assert!(history.ema_throughput() > 0.0);
        assert_eq!(history.recent_samples().len(), 1);
    }

    #[test]
    fn test_ema_calculation() {
        let mut history = PerformanceHistory::with_params(100, 0.1);

        // First sample
        history.record_sample(Duration::from_secs(1), 1000);
        assert_eq!(history.ema_throughput(), 1000.0);

        // Second sample (higher throughput)
        history.record_sample(Duration::from_secs(1), 2000);

        // EMA should be between 1000 and 2000
        assert!(history.ema_throughput() > 1000.0);
        assert!(history.ema_throughput() < 2000.0);

        // EMA = 0.1 * 2000 + 0.9 * 1000 = 200 + 900 = 1100
        assert!((history.ema_throughput() - 1100.0).abs() < 0.1);
    }

    #[test]
    fn test_performance_degradation_detection() {
        let mut history = PerformanceHistory::new();

        // Establish baseline with good performance
        for _ in 0..10 {
            history.record_sample(Duration::from_secs(1), 10000);
        }

        let baseline_throughput = history.ema_throughput();
        println!("Baseline throughput: {}", baseline_throughput);

        // Simulate sustained degradation (much slower, below 90% threshold)
        let mut triggered = false;
        for i in 0..1500 {
            history.record_sample(Duration::from_secs(1), 500); // Much worse than baseline

            if history.check_performance_degradation(baseline_throughput, 1000) {
                println!("Degradation detected at iteration {}", i);
                triggered = true;
                break;
            }
        }

        // Should detect degradation
        assert!(triggered, "Degradation should have been detected");
    }

    #[test]
    fn test_no_false_degradation_detection() {
        let mut history = PerformanceHistory::new();

        // Record good performance
        for _ in 0..100 {
            history.record_sample(Duration::from_secs(1), 1000);
        }

        // Should not detect degradation when performance is good
        assert!(!history.check_performance_degradation(900.0, 1000));
    }

    #[test]
    fn test_recent_samples_limit() {
        let mut history = PerformanceHistory::with_params(10, 0.1);

        // Add more samples than limit
        for i in 0..20 {
            history.record_sample(Duration::from_millis(100), 100 + i);
        }

        // Should only keep last 10
        assert_eq!(history.recent_samples().len(), 10);

        // Oldest sample should be from iteration 10
        assert_eq!(history.recent_samples()[0].operations, 110);
    }

    #[test]
    fn test_average_throughput() {
        let mut history = PerformanceHistory::new();

        history.record_sample(Duration::from_secs(1), 1000);
        history.record_sample(Duration::from_secs(1), 2000);

        let avg = history.average_throughput();

        // Total: 3000 ops in 2 seconds = 1500 ops/sec
        assert!((avg - 1500.0).abs() < 0.1);
    }

    #[test]
    fn test_variance_calculation() {
        let mut history = PerformanceHistory::new();

        // Add samples with known variance
        history.record_sample(Duration::from_secs(1), 1000);
        history.record_sample(Duration::from_secs(1), 2000);
        history.record_sample(Duration::from_secs(1), 3000);

        let variance = history.recent_variance();
        let std_dev = history.recent_std_dev();

        assert!(variance > 0.0);
        assert!(std_dev > 0.0);
        assert!((std_dev.powi(2) - variance).abs() < 0.01);
    }

    #[test]
    fn test_percentile_calculation() {
        let mut history = PerformanceHistory::new();

        for i in 0..100 {
            history.record_sample(Duration::from_secs(1), 1000 + i * 10);
        }

        let p50 = history.recent_percentile(50.0).unwrap();
        let p95 = history.recent_percentile(95.0).unwrap();
        let p99 = history.recent_percentile(99.0).unwrap();

        assert!(p50 > 0.0);
        assert!(p95 > p50);
        assert!(p99 > p95);
    }

    #[test]
    fn test_reset() {
        let mut history = PerformanceHistory::new();

        history.record_sample(Duration::from_secs(1), 1000);
        assert!(history.total_operations() > 0);

        history.reset();

        assert_eq!(history.total_operations(), 0);
        assert_eq!(history.ema_throughput(), 0.0);
        assert_eq!(history.recent_samples().len(), 0);
    }

    #[test]
    fn test_ignore_zero_duration() {
        let mut history = PerformanceHistory::new();

        history.record_sample(Duration::ZERO, 1000);

        // Should ignore the sample
        assert_eq!(history.total_operations(), 0);
    }

    #[test]
    fn test_ignore_zero_operations() {
        let mut history = PerformanceHistory::new();

        history.record_sample(Duration::from_secs(1), 0);

        // Should ignore the sample
        assert_eq!(history.total_operations(), 0);
    }
}
