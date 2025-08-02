//! Real-time compression with strict latency guarantees

use super::{Algorithm, Compressor, CompressorFactory, CompressionStats};
use crate::error::{ToplingError, Result};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// Compression mode for real-time scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMode {
    /// Ultra-low latency (< 1ms)
    UltraLowLatency,
    /// Low latency (< 10ms)
    LowLatency,
    /// Balanced latency vs compression (< 100ms)
    Balanced,
    /// High compression with acceptable latency (< 1s)
    HighCompression,
}

impl CompressionMode {
    /// Get the target latency for this mode
    pub fn target_latency(&self) -> Duration {
        match self {
            CompressionMode::UltraLowLatency => Duration::from_millis(1),
            CompressionMode::LowLatency => Duration::from_millis(10),
            CompressionMode::Balanced => Duration::from_millis(100),
            CompressionMode::HighCompression => Duration::from_millis(1000),
        }
    }
    
    /// Get the preferred algorithm for this mode
    pub fn preferred_algorithm(&self) -> Algorithm {
        match self {
            CompressionMode::UltraLowLatency => Algorithm::None,
            CompressionMode::LowLatency => Algorithm::Lz4,
            CompressionMode::Balanced => Algorithm::Zstd(3),
            CompressionMode::HighCompression => Algorithm::Zstd(9),
        }
    }
    
    /// Get the maximum acceptable memory usage (bytes per input byte)
    pub fn max_memory_ratio(&self) -> f64 {
        match self {
            CompressionMode::UltraLowLatency => 0.0,
            CompressionMode::LowLatency => 0.1,
            CompressionMode::Balanced => 1.0,
            CompressionMode::HighCompression => 4.0,
        }
    }
}

/// Configuration for real-time compression
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Compression mode
    pub mode: CompressionMode,
    /// Maximum number of concurrent compression operations
    pub max_concurrent: usize,
    /// Enable deadline-based scheduling
    pub enable_deadlines: bool,
    /// Fallback to no compression if deadline exceeded
    pub fallback_on_timeout: bool,
    /// Buffer size for batching small operations
    pub batch_size: usize,
    /// Batch timeout for collecting operations
    pub batch_timeout: Duration,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            mode: CompressionMode::LowLatency,
            max_concurrent: num_cpus::get(),
            enable_deadlines: true,
            fallback_on_timeout: true,
            batch_size: 10,
            batch_timeout: Duration::from_millis(1),
        }
    }
}

/// Real-time compressor with strict latency guarantees
pub struct RealtimeCompressor {
    config: RealtimeConfig,
    compressor: Arc<RwLock<Box<dyn Compressor>>>,
    fallback_compressor: Arc<Box<dyn Compressor>>, // No-op for timeouts
    semaphore: Arc<Semaphore>,
    stats: Arc<RwLock<RealtimeStats>>,
}

/// Statistics specific to real-time compression
#[derive(Debug, Clone, Default)]
pub struct RealtimeStats {
    /// Base compression stats
    pub base_stats: CompressionStats,
    /// Number of operations that met deadline
    pub deadline_met: u64,
    /// Number of operations that missed deadline
    pub deadline_missed: u64,
    /// Number of fallback operations
    pub fallback_operations: u64,
    /// Average latency in microseconds
    pub avg_latency_us: u64,
    /// Maximum latency observed
    pub max_latency_us: u64,
    /// 95th percentile latency
    pub p95_latency_us: u64,
    /// 99th percentile latency  
    pub p99_latency_us: u64,
    /// Recent latency measurements (for percentile calculation)
    latency_samples: Vec<u64>,
}

impl RealtimeStats {
    /// Calculate deadline success rate
    pub fn deadline_success_rate(&self) -> f64 {
        let total = self.deadline_met + self.deadline_missed;
        if total == 0 {
            0.0
        } else {
            self.deadline_met as f64 / total as f64
        }
    }
    
    /// Update with a new latency measurement
    fn update_latency(&mut self, latency_us: u64, met_deadline: bool) {
        if met_deadline {
            self.deadline_met += 1;
        } else {
            self.deadline_missed += 1;
        }
        
        self.latency_samples.push(latency_us);
        
        // Keep only recent samples for percentile calculation
        if self.latency_samples.len() > 1000 {
            self.latency_samples.drain(0..500); // Remove older half
        }
        
        // Update averages
        let total_ops = self.deadline_met + self.deadline_missed;
        self.avg_latency_us = (self.avg_latency_us * (total_ops - 1) + latency_us) / total_ops;
        self.max_latency_us = self.max_latency_us.max(latency_us);
        
        // Calculate percentiles
        if self.latency_samples.len() >= 20 {
            let mut sorted = self.latency_samples.clone();
            sorted.sort_unstable();
            
            let p95_idx = (sorted.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted.len() as f64 * 0.99) as usize;
            
            self.p95_latency_us = sorted[p95_idx.min(sorted.len() - 1)];
            self.p99_latency_us = sorted[p99_idx.min(sorted.len() - 1)];
        }
    }
}

impl RealtimeCompressor {
    /// Create a new real-time compressor
    pub fn new(config: RealtimeConfig) -> Result<Self> {
        let algorithm = config.mode.preferred_algorithm();
        let compressor = CompressorFactory::create(algorithm, None)?;
        let fallback_compressor = Arc::new(Box::new(super::NoCompressor) as Box<dyn Compressor>);
        
        Ok(Self {
            config: config.clone(),
            compressor: Arc::new(RwLock::new(compressor)),
            fallback_compressor,
            semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
            stats: Arc::new(RwLock::new(RealtimeStats::default())),
        })
    }
    
    /// Create with compression mode
    pub fn with_mode(mode: CompressionMode) -> Result<Self> {
        let config = RealtimeConfig {
            mode,
            ..Default::default()
        };
        Self::new(config)
    }
    
    /// Compress data with deadline guarantee
    pub async fn compress_with_deadline(&self, data: &[u8], deadline: Instant) -> Result<Vec<u8>> {
        let start_time = Instant::now();
        
        // Check if we already missed the deadline
        if Instant::now() >= deadline {
            return self.handle_timeout(data, start_time).await;
        }
        
        // Acquire semaphore permit for concurrency control
        let _permit = self.semaphore.acquire().await
            .map_err(|_| ToplingError::configuration("semaphore acquire failed"))?;
        
        // Check deadline again after acquiring permit
        if Instant::now() >= deadline {
            return self.handle_timeout(data, start_time).await;
        }
        
        // Perform compression with timeout
        let remaining_time = deadline.saturating_duration_since(Instant::now());
        
        let compression_result = tokio::time::timeout(
            remaining_time,
            self.compress_internal(data)
        ).await;
        
        let latency = start_time.elapsed();
        let met_deadline = Instant::now() <= deadline;
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.update_latency(latency.as_micros() as u64, met_deadline);
        }
        
        match compression_result {
            Ok(Ok(compressed)) => Ok(compressed),
            Ok(Err(e)) => Err(e),
            Err(_) => self.handle_timeout(data, start_time).await,
        }
    }
    
    /// Compress data with mode-specific deadline
    pub async fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let deadline = Instant::now() + self.config.mode.target_latency();
        self.compress_with_deadline(data, deadline).await
    }
    
    /// Decompress data
    pub async fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let compressor = self.compressor.read().unwrap();
        compressor.decompress(data)
    }
    
    /// Batch compress multiple items
    pub async fn compress_batch(&self, items: Vec<&[u8]>) -> Result<Vec<Vec<u8>>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        
        let deadline = Instant::now() + self.config.mode.target_latency();
        let mut results = Vec::with_capacity(items.len());
        
        for item in items {
            let result = self.compress_with_deadline(item, deadline).await?;
            results.push(result);
            
            // Check if we're running out of time
            if Instant::now() >= deadline {
                break;
            }
        }
        
        Ok(results)
    }
    
    /// Get real-time statistics
    pub fn stats(&self) -> RealtimeStats {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }
    
    /// Switch compression mode
    pub fn set_mode(&self, mode: CompressionMode) -> Result<()> {
        let algorithm = mode.preferred_algorithm();
        let new_compressor = CompressorFactory::create(algorithm, None)?;
        
        {
            let mut compressor = self.compressor.write().unwrap();
            *compressor = new_compressor;
        }
        
        Ok(())
    }
    
    /// Check if the compressor can meet deadline for given data size
    pub fn can_meet_deadline(&self, data_size: usize, deadline: Duration) -> bool {
        let algorithm = self.config.mode.preferred_algorithm();
        let expected_time = data_size as f64 / algorithm.compression_speed();
        
        Duration::from_secs_f64(expected_time) <= deadline
    }
    
    /// Internal compression implementation
    async fn compress_internal(&self, data: &[u8]) -> Result<Vec<u8>> {
        // For very small data, consider skipping compression
        if data.len() < 64 && self.config.mode == CompressionMode::UltraLowLatency {
            return Ok(data.to_vec());
        }
        
        let compressor = self.compressor.read().unwrap();
        compressor.compress(data)
    }
    
    /// Handle timeout by falling back to no compression
    async fn handle_timeout(&self, data: &[u8], start_time: Instant) -> Result<Vec<u8>> {
        let latency = start_time.elapsed();
        
        // Update timeout statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.update_latency(latency.as_micros() as u64, false);
            stats.fallback_operations += 1;
        }
        
        if self.config.fallback_on_timeout {
            // Use fallback compressor (no-op)
            self.fallback_compressor.compress(data)
        } else {
            Err(ToplingError::configuration("compression deadline exceeded"))
        }
    }
}

/// Builder for real-time compressor configuration
pub struct RealtimeCompressorBuilder {
    config: RealtimeConfig,
}

impl RealtimeCompressorBuilder {
    pub fn new() -> Self {
        Self {
            config: RealtimeConfig::default(),
        }
    }
    
    pub fn mode(mut self, mode: CompressionMode) -> Self {
        self.config.mode = mode;
        self
    }
    
    pub fn max_concurrent(mut self, max_concurrent: usize) -> Self {
        self.config.max_concurrent = max_concurrent;
        self
    }
    
    pub fn enable_deadlines(mut self, enable: bool) -> Self {
        self.config.enable_deadlines = enable;
        self
    }
    
    pub fn fallback_on_timeout(mut self, fallback: bool) -> Self {
        self.config.fallback_on_timeout = fallback;
        self
    }
    
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }
    
    pub fn build(self) -> Result<RealtimeCompressor> {
        RealtimeCompressor::new(self.config)
    }
}

impl Default for RealtimeCompressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    
    #[test]
    fn test_compression_mode() {
        assert!(CompressionMode::UltraLowLatency.target_latency() < CompressionMode::LowLatency.target_latency());
        assert_eq!(CompressionMode::LowLatency.preferred_algorithm(), Algorithm::Lz4);
    }
    
    #[test]
    fn test_realtime_config() {
        let config = RealtimeConfig::default();
        assert_eq!(config.mode, CompressionMode::LowLatency);
        assert!(config.max_concurrent > 0);
    }
    
    #[tokio::test]
    async fn test_realtime_compressor_creation() {
        let compressor = RealtimeCompressor::with_mode(CompressionMode::LowLatency).unwrap();
        let stats = compressor.stats();
        
        assert_eq!(stats.deadline_met, 0);
        assert_eq!(stats.deadline_missed, 0);
    }
    
    #[tokio::test]
    #[cfg(feature = "lz4")]
    async fn test_realtime_compression() {
        let compressor = RealtimeCompressor::with_mode(CompressionMode::LowLatency).unwrap();
        let data = b"test data for real-time compression";
        
        let compressed = compressor.compress(data).await.unwrap();
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        
        assert_eq!(decompressed, data);
        
        let stats = compressor.stats();
        assert_eq!(stats.deadline_met + stats.deadline_missed, 1);
    }
    
    #[tokio::test]
    #[cfg(feature = "lz4")]
    async fn test_deadline_compression() {
        let compressor = RealtimeCompressor::with_mode(CompressionMode::LowLatency).unwrap();
        let data = b"test data for deadline-based compression";
        let deadline = Instant::now() + Duration::from_millis(50);
        
        let compressed = compressor.compress_with_deadline(data, deadline).await.unwrap();
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        
        assert_eq!(decompressed, data);
    }
    
    #[tokio::test]
    #[cfg(feature = "lz4")]
    async fn test_batch_compression() {
        let compressor = RealtimeCompressor::with_mode(CompressionMode::LowLatency).unwrap();
        let items = vec![
            b"item 1".as_slice(),
            b"item 2".as_slice(),
            b"item 3".as_slice(),
        ];
        
        let compressed_items = compressor.compress_batch(items).await.unwrap();
        assert_eq!(compressed_items.len(), 3);
    }
    
    #[tokio::test]
    async fn test_timeout_handling() {
        let compressor = RealtimeCompressor::with_mode(CompressionMode::UltraLowLatency).unwrap();
        let data = vec![0u8; 10000]; // Large data that might timeout
        let deadline = Instant::now(); // Already passed
        
        let result = compressor.compress_with_deadline(&data, deadline).await;
        // Should either succeed with fallback or return timeout error
        match result {
            Ok(_) => {
                let stats = compressor.stats();
                assert!(stats.fallback_operations > 0);
            }
            Err(_) => {
                // Timeout error is also acceptable
            }
        }
    }
    
    #[test]
    fn test_deadline_prediction() {
        let compressor = RealtimeCompressor::with_mode(CompressionMode::LowLatency).unwrap();
        
        // Small data should meet deadline
        assert!(compressor.can_meet_deadline(1000, Duration::from_millis(10)));
        
        // Very large data might not meet tight deadline
        assert!(!compressor.can_meet_deadline(10_000_000, Duration::from_micros(1)));
    }
    
    #[test]
    fn test_statistics_tracking() {
        let mut stats = RealtimeStats::default();
        
        // Add some latency measurements
        stats.update_latency(1000, true);  // 1ms, met deadline
        stats.update_latency(5000, true);  // 5ms, met deadline
        stats.update_latency(15000, false); // 15ms, missed deadline
        
        assert_eq!(stats.deadline_met, 2);
        assert_eq!(stats.deadline_missed, 1);
        assert!(stats.deadline_success_rate() > 0.6);
        assert_eq!(stats.max_latency_us, 15000);
    }
    
    #[test]
    fn test_builder_pattern() {
        let compressor = RealtimeCompressorBuilder::new()
            .mode(CompressionMode::Balanced)
            .max_concurrent(8)
            .enable_deadlines(false)
            .fallback_on_timeout(false)
            .build()
            .unwrap();
        
        assert_eq!(compressor.config.mode, CompressionMode::Balanced);
        assert_eq!(compressor.config.max_concurrent, 8);
        assert!(!compressor.config.enable_deadlines);
        assert!(!compressor.config.fallback_on_timeout);
    }
}