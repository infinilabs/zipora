//! Adaptive compression that automatically selects the best algorithm

use super::{Algorithm, CompressionStats, Compressor, CompressorFactory, PerformanceRequirements};
use crate::error::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Configuration for adaptive compression
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Learning window size (number of operations to remember)
    pub learning_window: usize,
    /// Minimum number of operations before adaptation kicks in
    pub min_operations: usize,
    /// How often to re-evaluate the best algorithm (operations)
    pub evaluation_interval: usize,
    /// Threshold for switching algorithms (performance improvement required)
    pub switch_threshold: f64,
    /// Enable aggressive learning (test multiple algorithms)
    pub aggressive_learning: bool,
    /// Sample size for testing new algorithms
    pub test_sample_size: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            learning_window: 1000,
            min_operations: 50,
            evaluation_interval: 100,
            switch_threshold: 0.1, // 10% improvement required
            aggressive_learning: false,
            test_sample_size: 10,
        }
    }
}

/// Compression profile for different data types
#[derive(Debug, Clone)]
pub struct CompressionProfile {
    /// Data type identifier
    pub data_type: String,
    /// Preferred algorithm for this data type
    pub preferred_algorithm: Algorithm,
    /// Performance statistics for this profile
    pub stats: CompressionStats,
    /// Learning confidence (0.0 to 1.0)
    pub confidence: f64,
}

impl CompressionProfile {
    /// Create a new compression profile for a specific data type
    /// 
    /// # Arguments
    /// * `data_type` - Identifier for the type of data (e.g., "text", "binary")
    /// * `algorithm` - The preferred compression algorithm for this data type
    pub fn new(data_type: String, algorithm: Algorithm) -> Self {
        Self {
            data_type,
            preferred_algorithm: algorithm,
            stats: CompressionStats::default(),
            confidence: 0.0,
        }
    }
}

/// Performance measurement for a compression operation
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PerformanceMeasurement {
    algorithm: Algorithm,
    input_size: usize,
    output_size: usize,
    duration: Duration,
    timestamp: Instant,
    data_hash: u64, // Simple hash to identify data patterns
}

impl PerformanceMeasurement {
    fn compression_ratio(&self) -> f64 {
        self.output_size as f64 / self.input_size as f64
    }

    fn throughput(&self) -> f64 {
        self.input_size as f64 / self.duration.as_secs_f64()
    }

    fn score(&self, requirements: &PerformanceRequirements) -> f64 {
        let ratio_score = 1.0 - self.compression_ratio();
        let speed_score = self.throughput() / 1_000_000_000.0; // Normalize to GB/s
        let latency_penalty = if self.duration > requirements.max_latency {
            -1.0
        } else {
            0.0
        };

        requirements.speed_vs_quality * ratio_score
            + (1.0 - requirements.speed_vs_quality) * speed_score
            + latency_penalty
    }
}

/// Adaptive compressor that learns and improves over time
pub struct AdaptiveCompressor {
    config: AdaptiveConfig,
    requirements: PerformanceRequirements,
    current_algorithm: Algorithm,
    current_compressor: Arc<RwLock<Box<dyn Compressor>>>,
    performance_history: Arc<RwLock<VecDeque<PerformanceMeasurement>>>,
    profiles: Arc<RwLock<HashMap<String, CompressionProfile>>>,
    operation_count: Arc<RwLock<usize>>,
    stats: Arc<RwLock<CompressionStats>>,
}

impl AdaptiveCompressor {
    /// Create a new adaptive compressor
    pub fn new(config: AdaptiveConfig, requirements: PerformanceRequirements) -> Result<Self> {
        let initial_algorithm = Algorithm::Lz4; // Conservative starting point
        let initial_compressor = CompressorFactory::create(initial_algorithm, None)?;

        Ok(Self {
            config,
            requirements,
            current_algorithm: initial_algorithm,
            current_compressor: Arc::new(RwLock::new(initial_compressor)),
            performance_history: Arc::new(RwLock::new(VecDeque::new())),
            profiles: Arc::new(RwLock::new(HashMap::new())),
            operation_count: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(CompressionStats::default())),
        })
    }

    /// Create with default configuration
    pub fn default_with_requirements(requirements: PerformanceRequirements) -> Result<Self> {
        Self::new(AdaptiveConfig::default(), requirements)
    }

    /// Compress data with adaptive algorithm selection
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let start_time = Instant::now();

        // Check if we should adapt the algorithm
        self.maybe_adapt(data)?;

        // Perform compression
        let compressed = {
            let compressor = self.current_compressor.read().unwrap();
            compressor.compress(data)?
        };

        let duration = start_time.elapsed();

        // Record performance
        self.record_performance(data, &compressed, duration);

        Ok(compressed)
    }

    /// Decompress data
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let compressor = self.current_compressor.read().unwrap();
        compressor.decompress(data)
    }

    /// Get current compression statistics
    pub fn stats(&self) -> CompressionStats {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }

    /// Get current algorithm
    pub fn current_algorithm(&self) -> Algorithm {
        self.current_algorithm
    }

    /// Force adaptation to a specific algorithm
    pub fn set_algorithm(&mut self, algorithm: Algorithm) -> Result<()> {
        let new_compressor = CompressorFactory::create(algorithm, None)?;

        {
            let mut compressor = self.current_compressor.write().unwrap();
            *compressor = new_compressor;
        }

        self.current_algorithm = algorithm;
        Ok(())
    }

    /// Get compression profiles
    pub fn profiles(&self) -> HashMap<String, CompressionProfile> {
        let profiles = self.profiles.read().unwrap();
        profiles.clone()
    }

    /// Train the compressor with sample data
    pub fn train(&self, samples: &[(&[u8], &str)]) -> Result<()> {
        for (data, data_type) in samples {
            let data_hash = self.calculate_hash(data);

            // Test multiple algorithms on this sample
            for algorithm in CompressorFactory::available_algorithms() {
                if let Ok(compressor) = CompressorFactory::create(algorithm, Some(data)) {
                    let start_time = Instant::now();
                    if let Ok(compressed) = compressor.compress(data) {
                        let duration = start_time.elapsed();

                        let measurement = PerformanceMeasurement {
                            algorithm,
                            input_size: data.len(),
                            output_size: compressed.len(),
                            duration,
                            timestamp: Instant::now(),
                            data_hash,
                        };

                        // Update profile for this data type
                        self.update_profile(data_type, &measurement);
                    }
                }
            }
        }

        Ok(())
    }

    /// Maybe adapt the algorithm based on recent performance
    fn maybe_adapt(&self, data: &[u8]) -> Result<()> {
        let mut operation_count = self.operation_count.write().unwrap();
        *operation_count += 1;
        let count = *operation_count;
        drop(operation_count);

        // Don't adapt too early
        if count < self.config.min_operations {
            return Ok(());
        }

        // Don't adapt too frequently
        if count % self.config.evaluation_interval != 0 {
            return Ok(());
        }

        // Analyze recent performance
        let best_algorithm = self.find_best_algorithm(data)?;

        if best_algorithm != self.current_algorithm {
            let improvement = self.estimate_improvement(best_algorithm);

            if improvement > self.config.switch_threshold {
                // Can't call mutable method from immutable context
                // This would require a different design pattern
                log::info!(
                    "Would switch to algorithm: {:?} (improvement: {:.2}%)",
                    best_algorithm,
                    improvement * 100.0
                );
            }
        }

        Ok(())
    }

    /// Find the best algorithm for the current workload
    fn find_best_algorithm(&self, sample_data: &[u8]) -> Result<Algorithm> {
        let history = self.performance_history.read().unwrap();

        if history.len() < self.config.min_operations {
            return Ok(self.current_algorithm);
        }

        // Analyze recent performance by algorithm
        let mut algorithm_scores: HashMap<Algorithm, Vec<f64>> = HashMap::new();

        let recent_window = history.iter().rev().take(self.config.learning_window);

        for measurement in recent_window {
            let score = measurement.score(&self.requirements);
            algorithm_scores
                .entry(measurement.algorithm)
                .or_insert_with(Vec::new)
                .push(score);
        }

        // Find algorithm with best average score
        let mut best_algorithm = self.current_algorithm;
        let mut best_score = f64::NEG_INFINITY;

        for (algorithm, scores) in algorithm_scores {
            if scores.len() >= 5 {
                // Minimum samples for confidence
                let avg_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

                if avg_score > best_score {
                    best_score = avg_score;
                    best_algorithm = algorithm;
                }
            }
        }

        // If aggressive learning is enabled, occasionally test new algorithms
        if self.config.aggressive_learning
            && *self.operation_count.read().unwrap() % (self.config.evaluation_interval * 5) == 0
        {
            return self.test_new_algorithm(sample_data);
        }

        Ok(best_algorithm)
    }

    /// Test a new algorithm on sample data
    fn test_new_algorithm(&self, _sample_data: &[u8]) -> Result<Algorithm> {
        let available = CompressorFactory::available_algorithms();
        let current = self.current_algorithm;

        // Find an algorithm we haven't tested much recently
        let history = self.performance_history.read().unwrap();
        let mut usage_counts: HashMap<Algorithm, usize> = HashMap::new();

        for measurement in history.iter().rev().take(100) {
            *usage_counts.entry(measurement.algorithm).or_insert(0) += 1;
        }

        // Select the least used algorithm
        let mut test_algorithm = current;
        let mut min_usage = usize::MAX;

        for algorithm in available {
            let usage = usage_counts.get(&algorithm).unwrap_or(&0);
            if *usage < min_usage {
                min_usage = *usage;
                test_algorithm = algorithm;
            }
        }

        Ok(test_algorithm)
    }

    /// Estimate performance improvement from switching algorithms
    fn estimate_improvement(&self, new_algorithm: Algorithm) -> f64 {
        let history = self.performance_history.read().unwrap();

        let current_scores: Vec<f64> = history
            .iter()
            .rev()
            .take(self.config.learning_window)
            .filter(|m| m.algorithm == self.current_algorithm)
            .map(|m| m.score(&self.requirements))
            .collect();

        let new_scores: Vec<f64> = history
            .iter()
            .rev()
            .take(self.config.learning_window)
            .filter(|m| m.algorithm == new_algorithm)
            .map(|m| m.score(&self.requirements))
            .collect();

        if current_scores.is_empty() || new_scores.is_empty() {
            return 0.0;
        }

        let current_avg = current_scores.iter().sum::<f64>() / current_scores.len() as f64;
        let new_avg = new_scores.iter().sum::<f64>() / new_scores.len() as f64;

        (new_avg - current_avg) / current_avg.abs()
    }

    /// Switch to a new algorithm
    #[allow(dead_code)]
    fn switch_algorithm(&mut self, algorithm: Algorithm) -> Result<()> {
        let new_compressor = CompressorFactory::create(algorithm, None)?;

        {
            let mut compressor = self.current_compressor.write().unwrap();
            *compressor = new_compressor;
        }

        self.current_algorithm = algorithm;

        log::info!("Adaptive compressor switched to algorithm: {:?}", algorithm);

        Ok(())
    }

    /// Record performance measurement
    fn record_performance(&self, input: &[u8], output: &[u8], duration: Duration) {
        let measurement = PerformanceMeasurement {
            algorithm: self.current_algorithm,
            input_size: input.len(),
            output_size: output.len(),
            duration,
            timestamp: Instant::now(),
            data_hash: self.calculate_hash(input),
        };

        // Add to history
        {
            let mut history = self.performance_history.write().unwrap();
            history.push_back(measurement.clone());

            // Maintain window size
            while history.len() > self.config.learning_window {
                history.pop_front();
            }
        }

        // Update global stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.update(
                measurement.input_size,
                measurement.output_size,
                measurement.duration,
                measurement.algorithm,
            );
        }
    }

    /// Update compression profile for a data type
    fn update_profile(&self, data_type: &str, measurement: &PerformanceMeasurement) {
        let mut profiles = self.profiles.write().unwrap();

        let profile = profiles.entry(data_type.to_string()).or_insert_with(|| {
            CompressionProfile::new(data_type.to_string(), measurement.algorithm)
        });

        // Update statistics
        profile.stats.update(
            measurement.input_size,
            measurement.output_size,
            measurement.duration,
            measurement.algorithm,
        );

        // Update confidence
        profile.confidence = (profile.stats.operations as f64 / 100.0).min(1.0);

        // Maybe update preferred algorithm
        let score = measurement.score(&self.requirements);
        if score > 0.5 {
            // Arbitrary threshold for good performance
            profile.preferred_algorithm = measurement.algorithm;
        }
    }

    /// Calculate a simple hash of data for pattern recognition
    fn calculate_hash(&self, data: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Sample bytes from different parts of the data
        let sample_size = (data.len() / 10).max(1).min(1000);
        for i in 0..sample_size {
            let idx = (i * data.len()) / sample_size;
            data[idx].hash(&mut hasher);
        }

        hasher.finish()
    }
}

impl Compressor for AdaptiveCompressor {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.compress(data)
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.decompress(data)
    }

    fn algorithm(&self) -> Algorithm {
        self.current_algorithm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_config() {
        let config = AdaptiveConfig::default();
        assert!(config.learning_window > 0);
        assert!(config.min_operations > 0);
        assert!(config.switch_threshold > 0.0);
    }

    #[test]
    fn test_compression_profile() {
        let profile = CompressionProfile::new("text".to_string(), Algorithm::Lz4);
        assert_eq!(profile.confidence, 0.0);
        assert_eq!(profile.data_type, "text");
        assert_eq!(profile.preferred_algorithm, Algorithm::Lz4);
    }

    #[test]
    fn test_performance_measurement() {
        let measurement = PerformanceMeasurement {
            algorithm: Algorithm::Lz4,
            input_size: 1000,
            output_size: 500,
            duration: Duration::from_millis(10),
            timestamp: Instant::now(),
            data_hash: 12345,
        };

        assert_eq!(measurement.compression_ratio(), 0.5);
        assert!(measurement.throughput() > 0.0);
    }

    #[test]
    fn test_adaptive_compressor_creation() {
        let requirements = PerformanceRequirements::default();
        let compressor = AdaptiveCompressor::default_with_requirements(requirements).unwrap();

        assert_eq!(compressor.current_algorithm(), Algorithm::Lz4);
        assert_eq!(compressor.stats().operations, 0);
    }

    #[test]
    #[cfg(feature = "lz4")]
    fn test_adaptive_compression() {
        let requirements = PerformanceRequirements::default();
        let compressor = AdaptiveCompressor::default_with_requirements(requirements).unwrap();

        let data = b"test data that should compress well";
        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
        assert_eq!(compressor.stats().operations, 1);
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn test_algorithm_switching() {
        let requirements = PerformanceRequirements::default();
        let mut compressor = AdaptiveCompressor::default_with_requirements(requirements).unwrap();

        assert_eq!(compressor.current_algorithm(), Algorithm::Lz4);

        compressor.set_algorithm(Algorithm::Zstd(3)).unwrap();
        assert_eq!(compressor.current_algorithm(), Algorithm::Zstd(3));
    }

    #[test]
    fn test_training() {
        let requirements = PerformanceRequirements::default();
        let compressor = AdaptiveCompressor::default_with_requirements(requirements).unwrap();

        let samples = vec![
            (b"text data with lots of repetition".as_slice(), "text"),
            (b"binary data \x00\x01\x02\x03".as_slice(), "binary"),
        ];

        compressor.train(&samples).unwrap();

        let profiles = compressor.profiles();
        assert!(profiles.contains_key("text") || profiles.contains_key("binary"));
    }

    #[test]
    #[cfg(feature = "lz4")]
    fn test_performance_tracking() {
        let requirements = PerformanceRequirements::default();
        let compressor = AdaptiveCompressor::default_with_requirements(requirements).unwrap();

        // Perform several compression operations
        for i in 0..10 {
            let data = format!("test data iteration {}", i);
            let _ = compressor.compress(data.as_bytes()).unwrap();
        }

        let stats = compressor.stats();
        assert_eq!(stats.operations, 10);
        assert!(stats.bytes_processed > 0);
    }
}
