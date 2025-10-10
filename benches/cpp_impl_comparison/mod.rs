//! Common utilities for C++ implementation comparison benchmarks
//!
//! This module provides data generators and metric collection utilities
//! that exactly match C++ implementation's benchmark methodology for fair
//! apples-to-apples comparison.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Data generation patterns matching C++ implementation methodology
#[derive(Debug, Clone, Copy)]
pub enum DataPattern {
    /// 25% all-ones, 20% all-zeros, 55% random (C++ implementation default)
    CppImplDefault,
    /// Sequential patterns for testing cache efficiency
    Sequential,
    /// Completely random data
    Random,
    /// Sparse data (1% density)
    Sparse,
    /// Dense data (75% density)
    Dense,
}

/// Access pattern for benchmarks
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    /// Sequential access (0, 1, 2, ...)
    Sequential,
    /// Random shuffled access
    Random,
}

/// Key generation patterns for trie/hashmap benchmarks
#[derive(Debug, Clone, Copy)]
pub enum KeyPattern {
    /// Sequential keys: "key_00000001", "key_00000002", ...
    Sequential,
    /// Random hexadecimal keys
    RandomHex,
    /// Prefix-heavy keys with common prefixes
    PrefixHeavy,
}

/// Data generator matching C++ implementation patterns
pub struct CppImplDataGenerator {
    seed: u64,
}

impl CppImplDataGenerator {
    /// Create new generator with seed
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate bitvector matching C++ implementation pattern
    ///
    /// Pattern: 25% all-ones, 20% all-zeros, 55% random
    /// This exactly matches C++ implementation's benchmark data generation.
    pub fn generate_bitvector(&mut self, num_bits: usize, pattern: DataPattern) -> Vec<u64> {
        let num_words = (num_bits + 63) / 64;
        let mut data = vec![0u64; num_words];

        match pattern {
            DataPattern::CppImplDefault => {
                // Match C++ implementation: 25% all-ones, 20% all-zeros, 55% random
                for word in data.iter_mut() {
                    let r = self.next_u64();
                    *word = match r % 5 {
                        0 => 0,                    // 20% all-zeros
                        _ if r % 4 == 0 => !0,     // 25% all-ones (1 in 4 of remaining)
                        _ => self.next_u64(),      // 55% random
                    };
                }
            }
            DataPattern::Sequential => {
                // Alternating pattern for cache testing
                for (i, word) in data.iter_mut().enumerate() {
                    *word = if i % 2 == 0 { 0xAAAAAAAAAAAAAAAA } else { 0x5555555555555555 };
                }
            }
            DataPattern::Random => {
                // Fully random
                for word in data.iter_mut() {
                    *word = self.next_u64();
                }
            }
            DataPattern::Sparse => {
                // ~1% density - sparse data
                for word in data.iter_mut() {
                    let r = self.next_u64();
                    *word = if r % 100 == 0 { self.next_u64() } else { 0 };
                }
            }
            DataPattern::Dense => {
                // ~75% density - dense data
                for word in data.iter_mut() {
                    let r = self.next_u64();
                    *word = if r % 4 != 3 { !0 } else { self.next_u64() };
                }
            }
        }

        // Clear extra bits in last word to match exact bit count
        if num_bits % 64 != 0 {
            let last_idx = num_words - 1;
            let valid_bits = num_bits % 64;
            let mask = (1u64 << valid_bits) - 1;
            data[last_idx] &= mask;
        }

        data
    }

    /// Generate test positions for rank operations
    pub fn generate_positions(&mut self, size: usize, count: usize, pattern: AccessPattern) -> Vec<usize> {
        match pattern {
            AccessPattern::Sequential => {
                // Sequential: 0, size/count, 2*size/count, ...
                (0..count).map(|i| (i * size) / count.max(1)).collect()
            }
            AccessPattern::Random => {
                // Random shuffled positions
                let mut positions: Vec<usize> = (0..count).map(|i| (i * size) / count.max(1)).collect();
                self.shuffle(&mut positions);
                positions
            }
        }
    }

    /// Generate test indices for select operations
    pub fn generate_indices(&mut self, total_ones: usize, count: usize, pattern: AccessPattern) -> Vec<usize> {
        if total_ones == 0 {
            return vec![];
        }

        let count = count.min(total_ones);

        match pattern {
            AccessPattern::Sequential => {
                // Sequential: 0, total_ones/count, 2*total_ones/count, ...
                (0..count).map(|i| (i * total_ones) / count.max(1)).collect()
            }
            AccessPattern::Random => {
                // Random shuffled indices
                let mut indices: Vec<usize> = (0..count).map(|i| (i * total_ones) / count.max(1)).collect();
                self.shuffle(&mut indices);
                indices
            }
        }
    }

    /// Generate keys for trie/hashmap benchmarks
    pub fn generate_keys(&mut self, count: usize, pattern: KeyPattern) -> Vec<Vec<u8>> {
        match pattern {
            KeyPattern::Sequential => {
                (0..count)
                    .map(|i| format!("key_{:08}", i).into_bytes())
                    .collect()
            }
            KeyPattern::RandomHex => {
                (0..count)
                    .map(|_| {
                        let value = self.next_u64();
                        format!("{:016x}", value).into_bytes()
                    })
                    .collect()
            }
            KeyPattern::PrefixHeavy => {
                let prefixes = ["app", "compress", "data", "index", "query", "search", "store", "table"];
                (0..count)
                    .map(|i| {
                        let prefix = prefixes[i % prefixes.len()];
                        format!("{}_{:08}", prefix, i).into_bytes()
                    })
                    .collect()
            }
        }
    }

    /// Simple LCG random number generator (deterministic, reproducible)
    fn next_u64(&mut self) -> u64 {
        // Linear Congruential Generator (LCG) - simple and fast
        // Constants from Numerical Recipes
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        self.seed
    }

    /// Fisher-Yates shuffle
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        for i in 0..len {
            let j = (self.next_u64() as usize) % (len - i) + i;
            slice.swap(i, j);
        }
    }
}

/// Benchmark metrics matching C++ implementation reporting format
#[derive(Debug, Clone)]
pub struct BenchmarkMetrics {
    pub operation: String,
    pub impl_name: String,
    pub data_size: usize,
    pub access_pattern: String,

    // Timing metrics
    pub avg_ns: f64,
    pub median_ns: f64,
    pub p95_ns: f64,
    pub p99_ns: f64,
    pub std_dev_ns: f64,

    // Throughput metrics
    pub ops_per_sec: f64,
    pub gops_per_sec: Option<f64>,  // For rank/select (billions)

    // Memory metrics
    pub memory_bytes: usize,
    pub overhead_ratio: f64,

    // Correctness validation
    pub checksum: u64,
}

impl BenchmarkMetrics {
    /// Create new metrics
    pub fn new(operation: String, impl_name: String, data_size: usize, access_pattern: String) -> Self {
        Self {
            operation,
            impl_name,
            data_size,
            access_pattern,
            avg_ns: 0.0,
            median_ns: 0.0,
            p95_ns: 0.0,
            p99_ns: 0.0,
            std_dev_ns: 0.0,
            ops_per_sec: 0.0,
            gops_per_sec: None,
            memory_bytes: 0,
            overhead_ratio: 0.0,
            checksum: 0,
        }
    }

    /// Calculate percentiles from sorted times
    pub fn calculate_percentiles(&mut self, mut times_ns: Vec<f64>) {
        if times_ns.is_empty() {
            return;
        }

        times_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = times_ns.len();
        self.median_ns = times_ns[len / 2];
        self.p95_ns = times_ns[(len * 95) / 100];
        self.p99_ns = times_ns[(len * 99) / 100];

        // Calculate average
        let sum: f64 = times_ns.iter().sum();
        self.avg_ns = sum / len as f64;

        // Calculate standard deviation
        let variance: f64 = times_ns.iter()
            .map(|&x| {
                let diff = x - self.avg_ns;
                diff * diff
            })
            .sum::<f64>() / len as f64;
        self.std_dev_ns = variance.sqrt();

        // Calculate throughput
        if self.avg_ns > 0.0 {
            self.ops_per_sec = 1_000_000_000.0 / self.avg_ns;
            self.gops_per_sec = Some(self.ops_per_sec / 1_000_000_000.0);
        }
    }

    /// Format as markdown table row
    pub fn to_markdown_row(&self) -> String {
        let gops_str = self.gops_per_sec
            .map(|g| format!("{:.3}", g))
            .unwrap_or_else(|| "N/A".to_string());

        format!(
            "| {} | {} | {} | {:.2} ns | {:.2} ns | {:.2} ns | {:.3} Gops/s | {:.2}x | {:016x} |",
            self.operation,
            self.impl_name,
            self.access_pattern,
            self.avg_ns,
            self.median_ns,
            self.p95_ns,
            gops_str,
            self.overhead_ratio,
            self.checksum
        )
    }
}

/// Generate comparison table header
pub fn markdown_table_header() -> String {
    String::from(
        "| Operation | Implementation | Pattern | Avg (ns) | Median (ns) | P95 (ns) | Throughput | Memory | Checksum |\n\
         |-----------|----------------|---------|----------|-------------|----------|------------|----------|---------|\n"
    )
}

/// Calculate checksum for correctness validation
pub fn calculate_checksum(results: &[usize]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for &result in results {
        result.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generator_reproducibility() {
        let mut gen1 = CppImplDataGenerator::new(42);
        let mut gen2 = CppImplDataGenerator::new(42);

        let data1 = gen1.generate_bitvector(1000, DataPattern::CppImplDefault);
        let data2 = gen2.generate_bitvector(1000, DataPattern::CppImplDefault);

        assert_eq!(data1, data2, "Same seed should produce same data");
    }

    #[test]
    fn test_data_generator_pattern_distribution() {
        let mut gen = CppImplDataGenerator::new(12345);
        let data = gen.generate_bitvector(10000, DataPattern::CppImplDefault);

        // Count all-zero and all-one words
        let mut zeros = 0;
        let mut ones = 0;
        let mut mixed = 0;

        for &word in &data {
            if word == 0 {
                zeros += 1;
            } else if word == !0 {
                ones += 1;
            } else {
                mixed += 1;
            }
        }

        // Approximate check: 20% zeros, 25% ones, 55% random
        let total = data.len() as f64;
        let zero_ratio = zeros as f64 / total;
        let one_ratio = ones as f64 / total;
        let mixed_ratio = mixed as f64 / total;

        // Allow ±10% variance due to randomness
        assert!(zero_ratio >= 0.10 && zero_ratio <= 0.30, "Zero ratio: {:.2}", zero_ratio);
        assert!(one_ratio >= 0.15 && one_ratio <= 0.35, "One ratio: {:.2}", one_ratio);
        assert!(mixed_ratio >= 0.45 && mixed_ratio <= 0.65, "Mixed ratio: {:.2}", mixed_ratio);
    }

    #[test]
    fn test_position_generation() {
        let mut gen = CppImplDataGenerator::new(999);

        // Sequential positions
        let seq_pos = gen.generate_positions(1000, 10, AccessPattern::Sequential);
        assert_eq!(seq_pos.len(), 10);
        assert_eq!(seq_pos[0], 0);
        assert_eq!(seq_pos[9], 900);

        // Random positions should be different from sequential
        let mut gen2 = CppImplDataGenerator::new(999);
        let rand_pos = gen2.generate_positions(1000, 10, AccessPattern::Random);
        assert_eq!(rand_pos.len(), 10);
        assert_ne!(seq_pos, rand_pos);
    }

    #[test]
    fn test_key_generation() {
        let mut gen = CppImplDataGenerator::new(777);

        // Sequential keys
        let keys = gen.generate_keys(5, KeyPattern::Sequential);
        assert_eq!(keys.len(), 5);
        assert_eq!(keys[0], b"key_00000000");
        assert_eq!(keys[4], b"key_00000004");

        // RandomHex keys
        let hex_keys = gen.generate_keys(3, KeyPattern::RandomHex);
        assert_eq!(hex_keys.len(), 3);
        assert_eq!(hex_keys[0].len(), 16); // 16 hex chars

        // Prefix-heavy keys
        let prefix_keys = gen.generate_keys(10, KeyPattern::PrefixHeavy);
        assert_eq!(prefix_keys.len(), 10);
        assert!(prefix_keys[0].starts_with(b"app_"));
        assert!(prefix_keys[1].starts_with(b"compress_"));
    }

    #[test]
    fn test_metrics_calculation() {
        let mut metrics = BenchmarkMetrics::new(
            "rank1".to_string(),
            "zipora".to_string(),
            1_000_000,
            "sequential".to_string(),
        );

        // Sample times in nanoseconds
        let times = vec![5.0, 6.0, 5.5, 5.2, 7.0, 5.8, 6.2, 5.1, 8.0, 5.3];
        metrics.calculate_percentiles(times);

        assert!(metrics.avg_ns > 0.0);
        assert!(metrics.median_ns > 0.0);
        assert!(metrics.p95_ns > metrics.median_ns);
        assert!(metrics.p99_ns >= metrics.p95_ns);
        assert!(metrics.ops_per_sec > 0.0);
        assert!(metrics.gops_per_sec.is_some());
    }

    #[test]
    fn test_checksum_consistency() {
        let data1 = vec![1, 2, 3, 4, 5];
        let data2 = vec![1, 2, 3, 4, 5];
        let data3 = vec![1, 2, 3, 4, 6];

        let cs1 = calculate_checksum(&data1);
        let cs2 = calculate_checksum(&data2);
        let cs3 = calculate_checksum(&data3);

        assert_eq!(cs1, cs2, "Same data should produce same checksum");
        assert_ne!(cs1, cs3, "Different data should produce different checksum");
    }
}
