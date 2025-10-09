//! # Micro-Benchmarking Framework for SIMD Selection
//!
//! Provides runtime benchmarking capabilities to measure actual performance
//! of different SIMD implementations and guide adaptive selection.

use std::time::{Duration, Instant};

/// Results from micro-benchmarking a SIMD operation
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Median latency (nanoseconds)
    pub median_latency: Duration,

    /// 95th percentile latency
    pub p95_latency: Duration,

    /// 99th percentile latency
    pub p99_latency: Duration,

    /// Throughput (operations per second)
    pub throughput: f64,

    /// Number of samples collected
    pub samples: usize,

    /// Timestamp of last benchmark run
    pub last_update: Instant,
}

impl BenchmarkResults {
    /// Create new benchmark results
    pub fn new(
        median_latency: Duration,
        p95_latency: Duration,
        p99_latency: Duration,
        throughput: f64,
        samples: usize,
    ) -> Self {
        BenchmarkResults {
            median_latency,
            p95_latency,
            p99_latency,
            throughput,
            samples,
            last_update: Instant::now(),
        }
    }

    /// Calculate statistics from latency samples
    pub fn from_samples(latencies: &mut [Duration]) -> Self {
        assert!(!latencies.is_empty(), "Cannot create benchmark results from empty samples");

        latencies.sort();

        let samples = latencies.len();
        let median = latencies[samples / 2];
        let p95 = latencies[(samples * 95) / 100];
        let p99 = latencies[(samples * 99) / 100];

        let total_duration: Duration = latencies.iter().sum();
        let throughput = samples as f64 / total_duration.as_secs_f64();

        BenchmarkResults {
            median_latency: median,
            p95_latency: p95,
            p99_latency: p99,
            throughput,
            samples,
            last_update: Instant::now(),
        }
    }

    /// Check if benchmark results are stale (older than threshold)
    pub fn is_stale(&self, threshold: Duration) -> bool {
        self.last_update.elapsed() > threshold
    }
}

impl Default for BenchmarkResults {
    fn default() -> Self {
        BenchmarkResults {
            median_latency: Duration::from_micros(1),
            p95_latency: Duration::from_micros(2),
            p99_latency: Duration::from_micros(3),
            throughput: 1_000_000.0, // 1M ops/sec default
            samples: 0,
            last_update: Instant::now(),
        }
    }
}

/// Micro-benchmark runner for SIMD operations
pub struct MicroBenchmark {
    /// Warmup iterations
    warmup_iterations: usize,
    /// Measurement iterations
    measurement_iterations: usize,
}

impl MicroBenchmark {
    /// Create new micro-benchmark runner
    pub fn new(warmup_iterations: usize, measurement_iterations: usize) -> Self {
        MicroBenchmark {
            warmup_iterations,
            measurement_iterations,
        }
    }

    /// Run benchmark on provided closure
    pub fn run<F>(&self, mut operation: F) -> BenchmarkResults
    where
        F: FnMut(),
    {
        // Warmup phase
        for _ in 0..self.warmup_iterations {
            operation();
        }

        // Measurement phase
        let mut latencies = Vec::with_capacity(self.measurement_iterations);

        for _ in 0..self.measurement_iterations {
            let start = Instant::now();
            operation();
            latencies.push(start.elapsed());
        }

        BenchmarkResults::from_samples(&mut latencies)
    }

    /// Run benchmark with custom data generator
    pub fn run_with_data<D, F>(&self, mut data_gen: D, mut operation: F) -> BenchmarkResults
    where
        D: FnMut() -> Vec<u8>,
        F: FnMut(&[u8]),
    {
        // Generate test data
        let test_data = data_gen();

        // Warmup phase
        for _ in 0..self.warmup_iterations {
            operation(&test_data);
        }

        // Measurement phase
        let mut latencies = Vec::with_capacity(self.measurement_iterations);

        for _ in 0..self.measurement_iterations {
            let start = Instant::now();
            operation(&test_data);
            latencies.push(start.elapsed());
        }

        BenchmarkResults::from_samples(&mut latencies)
    }
}

impl Default for MicroBenchmark {
    fn default() -> Self {
        MicroBenchmark {
            warmup_iterations: 10,
            measurement_iterations: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_results_creation() {
        let results = BenchmarkResults::new(
            Duration::from_micros(100),
            Duration::from_micros(150),
            Duration::from_micros(200),
            10_000.0,
            100,
        );

        assert_eq!(results.median_latency, Duration::from_micros(100));
        assert_eq!(results.throughput, 10_000.0);
        assert_eq!(results.samples, 100);
    }

    #[test]
    fn test_benchmark_results_from_samples() {
        let mut latencies = vec![
            Duration::from_micros(100),
            Duration::from_micros(200),
            Duration::from_micros(150),
            Duration::from_micros(180),
            Duration::from_micros(120),
        ];

        let results = BenchmarkResults::from_samples(&mut latencies);

        assert_eq!(results.samples, 5);
        assert!(results.median_latency >= Duration::from_micros(100));
        assert!(results.median_latency <= Duration::from_micros(200));
        assert!(results.throughput > 0.0);
    }

    #[test]
    fn test_micro_benchmark_run() {
        let benchmark = MicroBenchmark::new(5, 10);

        let mut counter = 0;
        let results = benchmark.run(|| {
            counter += 1;
            std::hint::black_box(counter);
        });

        assert_eq!(results.samples, 10);
        assert!(results.median_latency > Duration::ZERO);
        assert!(results.throughput > 0.0);
    }

    #[test]
    fn test_micro_benchmark_with_data() {
        let benchmark = MicroBenchmark::default();

        let data_gen = || vec![0x01; 1024];
        let operation = |data: &[u8]| {
            // Use fold to avoid overflow
            let sum = data.iter().fold(0u64, |acc, &x| acc + x as u64);
            std::hint::black_box(sum);
        };

        let results = benchmark.run_with_data(data_gen, operation);

        assert_eq!(results.samples, 100);
        assert!(results.throughput > 0.0);
    }

    #[test]
    fn test_benchmark_staleness() {
        let results = BenchmarkResults::default();

        // Should not be stale immediately
        assert!(!results.is_stale(Duration::from_secs(1)));

        // Create old results
        let old_results = BenchmarkResults {
            last_update: Instant::now() - Duration::from_secs(10),
            ..Default::default()
        };

        // Should be stale after threshold
        assert!(old_results.is_stale(Duration::from_secs(5)));
    }
}
