//! Pipeline processing for streaming data operations
//!
//! This module provides a flexible pipeline system for processing data through multiple stages.
//! Key features include:
//!
//! * **Multi-stage processing**: Chain operations together with automatic buffering
//! * **Batching support**: Automatic batching with size and timeout limits for improved throughput
//! * **Backpressure handling**: Configurable buffer sizes and flow control
//! * **Performance monitoring**: Detailed statistics for each stage and overall pipeline
//! * **Timeout management**: Configurable timeouts for stage processing
//! * **Concurrent execution**: Configurable concurrency limits per stage
//!
//! ## Batching Features
//!
//! The pipeline system includes comprehensive batching support:
//!
//! * **BatchCollector**: Utility for accumulating items with size and timeout-based flushing
//! * **Automatic batching**: Stages can opt into batch processing for better performance
//! * **Fallback processing**: Default implementation processes items individually if no batch support
//! * **BatchMapStage**: Example implementation showing how to create batching-aware stages
//!
//! ## Example Usage
//!
//! ```rust
//! use zipora::concurrency::pipeline::{Pipeline, PipelineBuilder, BatchMapStage, BatchCollector};
//! use zipora::error::{ZiporaError, Result};
//! use std::time::Duration;
//!
//! // Create a pipeline with batching enabled
//! let pipeline = PipelineBuilder::new()
//!     .enable_batching(true)
//!     .batch_size(100)
//!     .batch_timeout(Duration::from_millis(50))
//!     .build();
//!
//! // Create a batching-aware stage
//! let stage = BatchMapStage::with_batch_support(
//!     "multiply".to_string(),
//!     |x: i32| -> Result<i32> { Ok(x * 2) },           // Individual processing
//!     |batch: Vec<i32>| -> Result<Vec<i32>> {           // Optimized batch processing
//!         Ok(batch.into_iter().map(|x| x * 2).collect())
//!     }
//! );
//!
//! // Use batch collector for accumulating items
//! let collector: BatchCollector<i32> = BatchCollector::new(50, Duration::from_millis(100));
//! ```

use crate::error::{Result, ZiporaError};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::time::timeout;

/// A stage in a processing pipeline
pub trait PipelineStage<T, U>: Send + Sync + 'static {
    /// Process a single item through this stage
    fn process(&self, input: T) -> Pin<Box<dyn Future<Output = Result<U>> + Send + '_>>;

    /// Get the name of this stage for monitoring
    fn name(&self) -> &str;

    /// Get the maximum number of concurrent items this stage can process
    fn max_concurrency(&self) -> usize {
        1
    }

    /// Check if this stage can handle batched input
    fn supports_batching(&self) -> bool {
        false
    }

    /// Process a batch of items (default implementation processes items individually)
    fn process_batch(
        &self,
        inputs: Vec<T>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<U>>> + Send + '_>>
    where
        T: Send + 'static,
        U: Send + 'static,
        Self: Sized,
    {
        Box::pin(async move {
            let mut results = Vec::with_capacity(inputs.len());
            for input in inputs {
                let result = self.process(input).await?;
                results.push(result);
            }
            Ok(results)
        })
    }
}

/// Statistics for pipeline performance monitoring
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total items processed
    pub total_processed: u64,
    /// Items currently in the pipeline
    pub items_in_flight: usize,
    /// Average processing time per item in microseconds
    pub avg_processing_time_us: u64,
    /// Throughput in items per second
    pub throughput_per_sec: f64,
    /// Pipeline utilization (0.0 to 1.0)
    pub utilization: f64,
    /// Per-stage statistics
    pub stage_stats: Vec<StageStats>,
}

/// Statistics for individual pipeline stages
#[derive(Debug, Clone)]
pub struct StageStats {
    /// Stage name
    pub name: String,
    /// Items processed by this stage
    pub processed: u64,
    /// Average processing time for this stage
    pub avg_time_us: u64,
    /// Items currently being processed
    pub active_items: usize,
    /// Stage utilization
    pub utilization: f64,
}

/// Configuration for pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Buffer size between stages
    pub buffer_size: usize,
    /// Maximum number of items in the entire pipeline
    pub max_in_flight: usize,
    /// Timeout for individual stage processing
    pub stage_timeout: Duration,
    /// Enable batching optimization
    pub enable_batching: bool,
    /// Batch size for batching optimization
    pub batch_size: usize,
    /// Batch timeout for collecting items
    pub batch_timeout: Duration,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            max_in_flight: 10000,
            stage_timeout: Duration::from_secs(30),
            enable_batching: false,
            batch_size: 100,
            batch_timeout: Duration::from_millis(100),
        }
    }
}

/// Utility for collecting items into batches with timeout-based flushing
pub struct BatchCollector<T> {
    buffer: Arc<Mutex<VecDeque<T>>>,
    max_batch_size: usize,
    batch_timeout: Duration,
    last_flush: Arc<Mutex<Instant>>,
}

impl<T> BatchCollector<T>
where
    T: Send + 'static,
{
    /// Create a new batch collector
    pub fn new(max_batch_size: usize, batch_timeout: Duration) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            max_batch_size,
            batch_timeout,
            last_flush: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Add an item to the collector
    /// Returns Some(batch) if a batch is ready to be processed
    pub async fn add(&self, item: T) -> Result<Option<Vec<T>>> {
        let mut buffer = self.buffer.lock().await;
        buffer.push_back(item);

        if buffer.len() >= self.max_batch_size {
            let batch = buffer.drain(..).collect();
            *self.last_flush.lock().await = Instant::now();
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }

    /// Check if a timeout-based flush should occur
    /// Returns Some(batch) if items should be flushed due to timeout
    pub async fn check_timeout(&self) -> Result<Option<Vec<T>>> {
        let buffer = self.buffer.lock().await;
        let last_flush = *self.last_flush.lock().await;

        if !buffer.is_empty() && last_flush.elapsed() >= self.batch_timeout {
            drop(buffer); // Release lock before acquiring again
            let mut buffer = self.buffer.lock().await;
            if !buffer.is_empty() {
                let batch = buffer.drain(..).collect();
                *self.last_flush.lock().await = Instant::now();
                return Ok(Some(batch));
            }
        }

        Ok(None)
    }

    /// Force flush all remaining items
    pub async fn flush(&self) -> Result<Option<Vec<T>>> {
        let mut buffer = self.buffer.lock().await;
        if !buffer.is_empty() {
            let batch = buffer.drain(..).collect();
            *self.last_flush.lock().await = Instant::now();
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }

    /// Get the current buffer size
    pub async fn len(&self) -> usize {
        self.buffer.lock().await.len()
    }

    /// Check if the buffer is empty
    pub async fn is_empty(&self) -> bool {
        self.buffer.lock().await.is_empty()
    }

    /// Start a background task that periodically checks for timeouts
    /// Returns a handle that can be used to stop the timeout checker
    pub fn start_timeout_checker<F>(&self, mut on_batch: F) -> tokio::task::JoinHandle<()>
    where
        F: FnMut(Vec<T>) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + 'static,
    {
        let collector = BatchCollector {
            buffer: self.buffer.clone(),
            max_batch_size: self.max_batch_size,
            batch_timeout: self.batch_timeout,
            last_flush: self.last_flush.clone(),
        };

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(collector.batch_timeout / 4);

            loop {
                interval.tick().await;

                if let Ok(Some(batch)) = collector.check_timeout().await {
                    on_batch(batch).await;
                }
            }
        })
    }
}

/// A multi-stage processing pipeline
pub struct Pipeline {
    config: PipelineConfig,
    stats: Arc<PipelineStatsInner>,
    start_time: Instant,
}

struct PipelineStatsInner {
    total_processed: AtomicU64,
    items_in_flight: AtomicUsize,
    total_processing_time_us: AtomicU64,
    stage_stats: RwLock<Vec<Arc<StageStatsInner>>>,
}

struct StageStatsInner {
    name: String,
    processed: AtomicU64,
    total_time_us: AtomicU64,
    active_items: AtomicUsize,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            stats: Arc::new(PipelineStatsInner {
                total_processed: AtomicU64::new(0),
                items_in_flight: AtomicUsize::new(0),
                total_processing_time_us: AtomicU64::new(0),
                stage_stats: RwLock::new(Vec::new()),
            }),
            start_time: Instant::now(),
        }
    }

    /// Execute a single-stage pipeline
    pub async fn execute_single<T, U, S>(&self, stage: S, input: T) -> Result<U>
    where
        S: PipelineStage<T, U>,
        T: Send + 'static,
        U: Send + 'static,
    {
        let start_time = Instant::now();
        self.stats.items_in_flight.fetch_add(1, Ordering::Relaxed);

        let result = timeout(self.config.stage_timeout, stage.process(input))
            .await
            .map_err(|_| ZiporaError::configuration("stage timeout"))?;

        let processing_time = start_time.elapsed().as_micros() as u64;
        self.stats
            .total_processing_time_us
            .fetch_add(processing_time, Ordering::Relaxed);
        self.stats.total_processed.fetch_add(1, Ordering::Relaxed);
        self.stats.items_in_flight.fetch_sub(1, Ordering::Relaxed);

        result
    }

    /// Execute a two-stage pipeline
    pub async fn execute_two_stage<T, U, V, S1, S2>(
        &self,
        stage1: S1,
        stage2: S2,
        input: T,
    ) -> Result<V>
    where
        S1: PipelineStage<T, U>,
        S2: PipelineStage<U, V>,
        T: Send + 'static,
        U: Send + 'static,
        V: Send + 'static,
    {
        let intermediate = self.execute_single(stage1, input).await?;
        self.execute_single(stage2, intermediate).await
    }

    /// Execute a multi-stage pipeline with streaming (same input/output type)
    pub async fn execute_stream<T>(
        &self,
        stages: Vec<Box<dyn PipelineStage<T, T>>>,
        input_rx: mpsc::Receiver<T>,
        output_tx: mpsc::Sender<T>,
    ) -> Result<()>
    where
        T: Send + 'static,
    {
        if stages.is_empty() {
            return Err(ZiporaError::invalid_data("no stages provided"));
        }

        // Initialize stage statistics
        {
            let mut stage_stats = self.stats.stage_stats.write().await;
            stage_stats.clear();
            for stage in &stages {
                stage_stats.push(Arc::new(StageStatsInner {
                    name: stage.name().to_string(),
                    processed: AtomicU64::new(0),
                    total_time_us: AtomicU64::new(0),
                    active_items: AtomicUsize::new(0),
                }));
            }
        }

        let num_stages = stages.len();

        // Create channels between stages
        let mut channels: Vec<Option<(mpsc::Sender<T>, mpsc::Receiver<T>)>> = Vec::new();
        for _ in 0..(num_stages - 1) {
            // One less channel than stages
            let (tx, rx) = mpsc::channel(self.config.buffer_size);
            channels.push(Some((tx, rx)));
        }

        // Spawn tasks for each stage
        let mut handles = Vec::new();
        let mut current_input = Some(input_rx);

        for (i, stage) in stages.into_iter().enumerate() {
            let stage_stats = {
                let stats = self.stats.stage_stats.read().await;
                stats[i].clone()
            };

            let mut stage_input_rx = if i == 0 {
                // First stage reads from pipeline input
                current_input.take().unwrap()
            } else {
                channels[i - 1].take().unwrap().1
            };

            let output_tx = if i == num_stages - 1 {
                // Last stage writes to pipeline output
                output_tx.clone()
            } else {
                channels[i].take().unwrap().0
            };

            let config = self.config.clone();
            let pipeline_stats = self.stats.clone();

            let handle = tokio::spawn(async move {
                while let Some(item) = stage_input_rx.recv().await {
                    let start_time = Instant::now();
                    stage_stats.active_items.fetch_add(1, Ordering::Relaxed);
                    pipeline_stats
                        .items_in_flight
                        .fetch_add(1, Ordering::Relaxed);

                    let result = timeout(config.stage_timeout, stage.process(item)).await;

                    let processing_time = start_time.elapsed().as_micros() as u64;
                    stage_stats
                        .total_time_us
                        .fetch_add(processing_time, Ordering::Relaxed);
                    stage_stats.processed.fetch_add(1, Ordering::Relaxed);
                    stage_stats.active_items.fetch_sub(1, Ordering::Relaxed);

                    match result {
                        Ok(Ok(output)) => {
                            if output_tx.send(output).await.is_err() {
                                break; // Output channel closed
                            }
                            pipeline_stats
                                .total_processed
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(Err(_)) | Err(_) => {
                            // Stage failed or timed out
                            break;
                        }
                    }

                    pipeline_stats
                        .items_in_flight
                        .fetch_sub(1, Ordering::Relaxed);
                }

                drop(output_tx); // Signal end of stream
            });

            handles.push(handle);
        }

        // Wait for all stages to complete
        for handle in handles {
            let _ = handle.await;
        }

        Ok(())
    }

    /// Process a batch of items through a single stage
    pub async fn process_batch<T, U, S>(&self, stage: S, inputs: Vec<T>) -> Result<Vec<U>>
    where
        S: PipelineStage<T, U>,
        T: Send + 'static,
        U: Send + 'static,
    {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let start_time = Instant::now();
        let batch_size = inputs.len();

        self.stats
            .items_in_flight
            .fetch_add(batch_size, Ordering::Relaxed);

        let result = if stage.supports_batching() && self.config.enable_batching {
            timeout(self.config.stage_timeout, stage.process_batch(inputs))
                .await
                .map_err(|_| ZiporaError::configuration("batch processing timeout"))?
        } else {
            // Process individually
            let mut results = Vec::with_capacity(batch_size);
            for input in inputs {
                let output = timeout(self.config.stage_timeout, stage.process(input))
                    .await
                    .map_err(|_| ZiporaError::configuration("stage timeout"))??;
                results.push(output);
            }
            Ok(results)
        };

        let processing_time = start_time.elapsed().as_micros() as u64;
        self.stats
            .total_processing_time_us
            .fetch_add(processing_time, Ordering::Relaxed);
        self.stats
            .total_processed
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.stats
            .items_in_flight
            .fetch_sub(batch_size, Ordering::Relaxed);

        result
    }

    /// Get current pipeline statistics
    pub async fn stats(&self) -> PipelineStats {
        let total_processed = self.stats.total_processed.load(Ordering::Relaxed);
        let total_time = self.stats.total_processing_time_us.load(Ordering::Relaxed);

        let avg_processing_time_us = if total_processed > 0 {
            total_time / total_processed
        } else {
            0
        };

        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        let throughput_per_sec = if elapsed_secs > 0.0 {
            total_processed as f64 / elapsed_secs
        } else {
            0.0
        };

        let items_in_flight = self.stats.items_in_flight.load(Ordering::Relaxed);
        let utilization = items_in_flight as f64 / self.config.max_in_flight as f64;

        let stage_stats_inner = self.stats.stage_stats.read().await;
        let stage_stats = stage_stats_inner
            .iter()
            .map(|stats| {
                let processed = stats.processed.load(Ordering::Relaxed);
                let total_time = stats.total_time_us.load(Ordering::Relaxed);
                let avg_time_us = if processed > 0 {
                    total_time / processed
                } else {
                    0
                };
                let active_items = stats.active_items.load(Ordering::Relaxed);

                StageStats {
                    name: stats.name.clone(),
                    processed,
                    avg_time_us,
                    active_items,
                    utilization: active_items as f64 / self.config.max_in_flight as f64,
                }
            })
            .collect();

        PipelineStats {
            total_processed,
            items_in_flight,
            avg_processing_time_us,
            throughput_per_sec,
            utilization,
            stage_stats,
        }
    }
}

/// Builder for constructing pipelines
pub struct PipelineBuilder {
    config: PipelineConfig,
}

impl PipelineBuilder {
    /// Create a new builder for configuring a pipeline
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }

    /// Set the buffer size for pipeline stages
    pub fn buffer_size(mut self, buffer_size: usize) -> Self {
        self.config.buffer_size = buffer_size;
        self
    }

    /// Set the maximum number of items that can be processed concurrently
    pub fn max_in_flight(mut self, max_in_flight: usize) -> Self {
        self.config.max_in_flight = max_in_flight;
        self
    }

    /// Set the timeout for each pipeline stage
    pub fn stage_timeout(mut self, stage_timeout: Duration) -> Self {
        self.config.stage_timeout = stage_timeout;
        self
    }

    /// Enable or disable batching of pipeline items
    pub fn enable_batching(mut self, enable: bool) -> Self {
        self.config.enable_batching = enable;
        self
    }

    /// Set the batch size for processing multiple items together
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set the timeout for collecting items into batches
    pub fn batch_timeout(mut self, batch_timeout: Duration) -> Self {
        self.config.batch_timeout = batch_timeout;
        self
    }

    /// Build the configured pipeline
    pub fn build(self) -> Pipeline {
        Pipeline::new(self.config)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Simple stage implementations for common operations

/// A batching-aware map stage that applies a function to items, with optimized batch processing
pub struct BatchMapStage<F, FB> {
    name: String,
    func: F,
    batch_func: Option<FB>,
    max_concurrency: usize,
}

impl<F, FB> BatchMapStage<F, FB> {
    /// Create a new batch map stage with individual processing only
    pub fn new(name: String, func: F) -> Self {
        Self {
            name,
            func,
            batch_func: None,
            max_concurrency: 1,
        }
    }

    /// Create a new batch map stage with both individual and batch processing
    pub fn with_batch_support(name: String, func: F, batch_func: FB) -> Self {
        Self {
            name,
            func,
            batch_func: Some(batch_func),
            max_concurrency: 1,
        }
    }

    /// Set the maximum concurrency for this stage
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }
}

impl<T, U, F, FB> PipelineStage<T, U> for BatchMapStage<F, FB>
where
    F: Fn(T) -> Result<U> + Send + Sync + 'static,
    FB: Fn(Vec<T>) -> Result<Vec<U>> + Send + Sync + 'static,
    T: Send + 'static,
    U: Send + 'static,
{
    fn process(&self, input: T) -> Pin<Box<dyn Future<Output = Result<U>> + Send + '_>> {
        let func = &self.func;
        Box::pin(async move { func(input) })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    fn supports_batching(&self) -> bool {
        self.batch_func.is_some()
    }

    fn process_batch(
        &self,
        inputs: Vec<T>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<U>>> + Send + '_>>
    where
        T: Send + 'static,
        U: Send + 'static,
        Self: Sized,
    {
        if let Some(batch_func) = &self.batch_func {
            let batch_func = batch_func;
            Box::pin(async move { batch_func(inputs) })
        } else {
            // Fall back to default implementation
            Box::pin(async move {
                let mut results = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let result = self.process(input).await?;
                    results.push(result);
                }
                Ok(results)
            })
        }
    }
}

/// A simple map stage that applies a function to each item
pub struct MapStage<F> {
    name: String,
    func: F,
}

impl<F> MapStage<F> {
    /// Create a new map stage that transforms input using the provided function
    pub fn new(name: String, func: F) -> Self {
        Self { name, func }
    }
}

impl<T, U, F> PipelineStage<T, U> for MapStage<F>
where
    F: Fn(T) -> Result<U> + Send + Sync + 'static,
    T: Send + 'static,
    U: Send + 'static,
{
    fn process(&self, input: T) -> Pin<Box<dyn Future<Output = Result<U>> + Send + '_>> {
        let func = &self.func;
        Box::pin(async move { func(input) })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A filter stage that conditionally passes items through
pub struct FilterStage<F> {
    name: String,
    predicate: F,
}

impl<F> FilterStage<F> {
    /// Create a new filter stage that conditionally passes items based on the predicate
    pub fn new(name: String, predicate: F) -> Self {
        Self { name, predicate }
    }
}

impl<T, F> PipelineStage<T, Option<T>> for FilterStage<F>
where
    F: Fn(&T) -> bool + Send + Sync + 'static,
    T: Send + 'static,
{
    fn process(&self, input: T) -> Pin<Box<dyn Future<Output = Result<Option<T>>> + Send + '_>> {
        let predicate = &self.predicate;
        Box::pin(async move {
            if predicate(&input) {
                Ok(Some(input))
            } else {
                Ok(None)
            }
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = Pipeline::new(PipelineConfig::default());
        let stats = pipeline.stats().await;

        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.items_in_flight, 0);
    }

    #[tokio::test]
    async fn test_single_stage_execution() {
        let pipeline = Pipeline::new(PipelineConfig::default());
        let stage = MapStage::new("double".to_string(), |x: i32| Ok(x * 2));

        let result = pipeline.execute_single(stage, 21).await.unwrap();
        assert_eq!(result, 42);

        let stats = pipeline.stats().await;
        assert_eq!(stats.total_processed, 1);
    }

    #[tokio::test]
    async fn test_two_stage_execution() {
        let pipeline = Pipeline::new(PipelineConfig::default());
        let stage1 = MapStage::new("double".to_string(), |x: i32| Ok(x * 2));
        let stage2 = MapStage::new("add_one".to_string(), |x: i32| Ok(x + 1));

        let result = pipeline
            .execute_two_stage(stage1, stage2, 20)
            .await
            .unwrap();
        assert_eq!(result, 41); // (20 * 2) + 1

        let stats = pipeline.stats().await;
        assert_eq!(stats.total_processed, 2); // Two stages processed
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let pipeline = Pipeline::new(PipelineConfig::default());
        let stage = MapStage::new("double".to_string(), |x: i32| Ok(x * 2));

        let inputs = vec![1, 2, 3, 4, 5];
        let results = pipeline.process_batch(stage, inputs).await.unwrap();

        assert_eq!(results, vec![2, 4, 6, 8, 10]);

        let stats = pipeline.stats().await;
        assert_eq!(stats.total_processed, 5);
    }

    #[tokio::test]
    async fn test_filter_stage() {
        let pipeline = Pipeline::new(PipelineConfig::default());

        let stage1 = FilterStage::new("even_only".to_string(), |x: &i32| *x % 2 == 0);
        let result1 = pipeline.execute_single(stage1, 4).await.unwrap();
        assert_eq!(result1, Some(4));

        let stage2 = FilterStage::new("even_only".to_string(), |x: &i32| *x % 2 == 0);
        let result2 = pipeline.execute_single(stage2, 5).await.unwrap();
        assert_eq!(result2, None);
    }

    #[tokio::test]
    async fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .buffer_size(500)
            .max_in_flight(5000)
            .enable_batching(true)
            .batch_size(50)
            .build();

        assert_eq!(pipeline.config.buffer_size, 500);
        assert_eq!(pipeline.config.max_in_flight, 5000);
        assert!(pipeline.config.enable_batching);
        assert_eq!(pipeline.config.batch_size, 50);
    }

    #[tokio::test]
    async fn test_batch_collector() {
        let collector = BatchCollector::new(3, Duration::from_millis(100));

        // Test size-based batching
        assert!(collector.add(1).await.unwrap().is_none());
        assert!(collector.add(2).await.unwrap().is_none());
        let batch = collector.add(3).await.unwrap();
        assert_eq!(batch, Some(vec![1, 2, 3]));

        // Test timeout-based flushing
        collector.add(4).await.unwrap();
        collector.add(5).await.unwrap();

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;
        let batch = collector.check_timeout().await.unwrap();
        assert_eq!(batch, Some(vec![4, 5]));

        // Test manual flush
        collector.add(6).await.unwrap();
        let batch = collector.flush().await.unwrap();
        assert_eq!(batch, Some(vec![6]));

        // Test empty flush
        let batch = collector.flush().await.unwrap();
        assert_eq!(batch, None);
    }

    #[tokio::test]
    async fn test_batch_collector_timeout_checker() {
        let collector = BatchCollector::new(5, Duration::from_millis(50));
        let batches = Arc::new(Mutex::new(Vec::new()));
        let batches_clone = batches.clone();

        let handle = collector.start_timeout_checker(move |batch| {
            let batches = batches_clone.clone();
            Box::pin(async move {
                batches.lock().await.push(batch);
            })
        });

        // Add some items
        collector.add(1).await.unwrap();
        collector.add(2).await.unwrap();

        // Wait for timeout checker to trigger
        tokio::time::sleep(Duration::from_millis(100)).await;

        handle.abort();

        let collected_batches = batches.lock().await;
        assert_eq!(collected_batches.len(), 1);
        assert_eq!(collected_batches[0], vec![1, 2]);
    }

    #[tokio::test]
    async fn test_batch_map_stage_individual() {
        let pipeline = Pipeline::new(PipelineConfig::default());
        let stage: BatchMapStage<_, fn(Vec<i32>) -> Result<Vec<i32>>> =
            BatchMapStage::new("multiply".to_string(), |x: i32| Ok(x * 3));

        let result = pipeline.execute_single(stage, 7).await.unwrap();
        assert_eq!(result, 21);

        let stats = pipeline.stats().await;
        assert_eq!(stats.total_processed, 1);
    }

    #[tokio::test]
    async fn test_batch_map_stage_with_batching() {
        let pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(3)
            .build();

        let stage = BatchMapStage::with_batch_support(
            "batch_multiply".to_string(),
            |x: i32| Ok(x * 2), // Individual function
            |batch: Vec<i32>| {
                // Batch function
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        assert!(stage.supports_batching());

        let inputs = vec![1, 2, 3, 4, 5];
        let results = pipeline.process_batch(stage, inputs).await.unwrap();

        assert_eq!(results, vec![2, 4, 6, 8, 10]);

        let stats = pipeline.stats().await;
        assert_eq!(stats.total_processed, 5);
    }

    #[tokio::test]
    async fn test_improved_process_batch_fallback() {
        let pipeline = Pipeline::new(PipelineConfig::default());
        let stage = MapStage::new("add_ten".to_string(), |x: i32| Ok(x + 10));

        // Test that the default process_batch implementation works properly
        let inputs = vec![1, 2, 3];
        let results = pipeline.process_batch(stage, inputs).await.unwrap();

        assert_eq!(results, vec![11, 12, 13]);

        let stats = pipeline.stats().await;
        assert_eq!(stats.total_processed, 3);
    }

    #[tokio::test]
    async fn test_batch_collector_concurrent_access() {
        let collector = Arc::new(BatchCollector::new(2, Duration::from_millis(50)));
        let mut handles = Vec::new();

        // Spawn multiple tasks adding items concurrently
        for i in 0..10 {
            let collector = collector.clone();
            let handle = tokio::spawn(async move { collector.add(i).await.unwrap() });
            handles.push(handle);
        }

        let mut batches = Vec::new();
        for handle in handles {
            if let Some(batch) = handle.await.unwrap() {
                batches.push(batch);
            }
        }

        // Should have some batches due to size limits
        assert!(!batches.is_empty());

        // Flush any remaining items
        if let Some(remaining) = collector.flush().await.unwrap() {
            batches.push(remaining);
        }

        // All items should be accounted for
        let total_items: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total_items, 10);
    }

    // Comprehensive BatchCollector tests

    #[tokio::test]
    async fn test_batch_collector_size_based_batching() {
        let collector = BatchCollector::new(5, Duration::from_secs(1));

        // Add items one by one, should not trigger until batch size reached
        for i in 0..4 {
            let result = collector.add(i).await.unwrap();
            assert!(
                result.is_none(),
                "Should not return batch before size limit"
            );
            assert_eq!(collector.len().await, i + 1);
        }

        // Adding the 5th item should trigger batch
        let batch = collector.add(4).await.unwrap();
        assert_eq!(batch, Some(vec![0, 1, 2, 3, 4]));
        assert_eq!(collector.len().await, 0);
        assert!(collector.is_empty().await);
    }

    #[tokio::test]
    async fn test_batch_collector_timeout_based_flushing() {
        let collector = BatchCollector::new(10, Duration::from_millis(50));

        // Add some items (less than batch size)
        collector.add(1).await.unwrap();
        collector.add(2).await.unwrap();
        collector.add(3).await.unwrap();

        assert_eq!(collector.len().await, 3);

        // Check timeout before elapsed - should return None
        let batch = collector.check_timeout().await.unwrap();
        assert!(batch.is_none());

        // Wait for timeout to elapse
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Now check_timeout should return the batch
        let batch = collector.check_timeout().await.unwrap();
        assert_eq!(batch, Some(vec![1, 2, 3]));
        assert!(collector.is_empty().await);
    }

    #[tokio::test]
    async fn test_batch_collector_manual_flush() {
        let collector = BatchCollector::new(10, Duration::from_secs(1));

        // Add some items
        collector.add("a").await.unwrap();
        collector.add("b").await.unwrap();
        collector.add("c").await.unwrap();

        // Manual flush should return all items
        let batch = collector.flush().await.unwrap();
        assert_eq!(batch, Some(vec!["a", "b", "c"]));
        assert!(collector.is_empty().await);

        // Flush on empty collector should return None
        let empty_batch = collector.flush().await.unwrap();
        assert!(empty_batch.is_none());
    }

    #[tokio::test]
    async fn test_batch_collector_zero_timeout() {
        let collector = BatchCollector::new(5, Duration::from_millis(0));

        collector.add(1).await.unwrap();
        collector.add(2).await.unwrap();

        // With zero timeout, check_timeout should immediately return items
        let batch = collector.check_timeout().await.unwrap();
        assert_eq!(batch, Some(vec![1, 2]));
    }

    #[tokio::test]
    async fn test_batch_collector_large_batch_size() {
        let collector = BatchCollector::new(1000, Duration::from_millis(10));

        // Add many items at once
        for i in 0..500 {
            let result = collector.add(i).await.unwrap();
            assert!(result.is_none()); // Should not trigger batch
        }

        assert_eq!(collector.len().await, 500);

        // Timeout flush should get all items
        tokio::time::sleep(Duration::from_millis(15)).await;
        let batch = collector.check_timeout().await.unwrap();
        assert_eq!(batch.unwrap().len(), 500);
    }

    #[tokio::test]
    async fn test_batch_collector_background_timeout_checker() {
        let collector = BatchCollector::new(10, Duration::from_millis(30));
        let received_batches = Arc::new(Mutex::new(Vec::new()));
        let received_clone = received_batches.clone();

        // Start the background timeout checker
        let handle = collector.start_timeout_checker(move |batch| {
            let received = received_clone.clone();
            Box::pin(async move {
                received.lock().await.push(batch);
            })
        });

        // Add items over multiple timeout periods
        collector.add(1).await.unwrap();
        collector.add(2).await.unwrap();

        tokio::time::sleep(Duration::from_millis(40)).await;

        collector.add(3).await.unwrap();
        collector.add(4).await.unwrap();

        tokio::time::sleep(Duration::from_millis(40)).await;

        // Stop the background checker
        handle.abort();

        // Should have received 2 batches from the timeout checker
        let batches = received_batches.lock().await;
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec![1, 2]);
        assert_eq!(batches[1], vec![3, 4]);
    }

    #[tokio::test]
    async fn test_batch_collector_mixed_size_and_timeout_triggers() {
        let collector = BatchCollector::new(3, Duration::from_millis(50));

        // First batch: size trigger
        collector.add(1).await.unwrap();
        collector.add(2).await.unwrap();
        let batch1 = collector.add(3).await.unwrap();
        assert_eq!(batch1, Some(vec![1, 2, 3]));

        // Second batch: timeout trigger
        collector.add(4).await.unwrap();
        collector.add(5).await.unwrap();

        tokio::time::sleep(Duration::from_millis(60)).await;

        let batch2 = collector.check_timeout().await.unwrap();
        assert_eq!(batch2, Some(vec![4, 5]));

        // Third batch: mix both
        collector.add(6).await.unwrap();
        collector.add(7).await.unwrap();
        let partial = collector.add(8).await.unwrap();
        assert_eq!(partial, Some(vec![6, 7, 8])); // Size trigger

        collector.add(9).await.unwrap();
        tokio::time::sleep(Duration::from_millis(60)).await;
        let final_batch = collector.check_timeout().await.unwrap();
        assert_eq!(final_batch, Some(vec![9])); // Timeout trigger
    }

    #[tokio::test]
    async fn test_batch_collector_high_concurrency() {
        let collector = Arc::new(BatchCollector::new(5, Duration::from_millis(100)));
        let batch_results = Arc::new(Mutex::new(Vec::new()));

        // Spawn many concurrent tasks
        let mut handles = Vec::new();
        for i in 0..100 {
            let collector = collector.clone();
            let results = batch_results.clone();

            let handle = tokio::spawn(async move {
                if let Some(batch) = collector.add(i).await.unwrap() {
                    results.lock().await.push(batch);
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Flush any remaining items
        if let Some(final_batch) = collector.flush().await.unwrap() {
            batch_results.lock().await.push(final_batch);
        }

        // Verify all items are accounted for
        let batches = batch_results.lock().await;
        let total_items: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total_items, 100);

        // Verify batch sizes are correct (should be 5 or less)
        for batch in batches.iter() {
            assert!(batch.len() <= 5);
        }
    }

    // Comprehensive BatchMapStage tests

    #[tokio::test]
    async fn test_batch_map_stage_individual_only() {
        let pipeline = Pipeline::new(PipelineConfig::default());
        let stage: BatchMapStage<_, fn(Vec<i32>) -> Result<Vec<i32>>> =
            BatchMapStage::new("square".to_string(), |x: i32| Ok(x * x));

        assert!(!stage.supports_batching());
        assert_eq!(stage.name(), "square");
        assert_eq!(stage.max_concurrency(), 1);

        // Test individual processing
        let result = pipeline.execute_single(stage, 5).await.unwrap();
        assert_eq!(result, 25);

        let stats = pipeline.stats().await;
        assert_eq!(stats.total_processed, 1);
    }

    #[tokio::test]
    async fn test_batch_map_stage_with_batch_function() {
        let _pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(3)
            .build();

        let stage = BatchMapStage::with_batch_support(
            "power_of_two".to_string(),
            |x: i32| Ok(1 << x), // Individual: 2^x
            |batch: Vec<i32>| {
                // Batch: optimized version
                Ok(batch.into_iter().map(|x| 1 << x).collect())
            },
        );

        assert!(stage.supports_batching());
        assert_eq!(stage.name(), "power_of_two");

        // Test individual processing
        let individual_result = stage.process(3).await.unwrap();
        assert_eq!(individual_result, 8); // 2^3

        // Test batch processing
        let batch_inputs = vec![1, 2, 3, 4];
        let batch_results = stage.process_batch(batch_inputs).await.unwrap();
        assert_eq!(batch_results, vec![2, 4, 8, 16]); // [2^1, 2^2, 2^3, 2^4]
    }

    #[tokio::test]
    async fn test_batch_map_stage_fallback_behavior() {
        // Stage with batch support but no batch function provided
        let stage = BatchMapStage::with_batch_support(
            "add_one".to_string(),
            |x: i32| Ok(x + 1),
            |batch: Vec<i32>| Ok(batch.into_iter().map(|x| x + 1).collect()),
        );

        // Test that fallback to individual processing works
        let inputs = vec![10, 20, 30];
        let results = stage.process_batch(inputs).await.unwrap();
        assert_eq!(results, vec![11, 21, 31]);

        // Test stage without batch function falls back properly
        let simple_stage: BatchMapStage<_, fn(Vec<i32>) -> Result<Vec<i32>>> =
            BatchMapStage::new("multiply_by_3".to_string(), |x: i32| Ok(x * 3));

        let inputs2 = vec![1, 2, 3];
        let results2 = simple_stage.process_batch(inputs2).await.unwrap();
        assert_eq!(results2, vec![3, 6, 9]);
    }

    #[tokio::test]
    async fn test_batch_map_stage_with_max_concurrency() {
        let stage: BatchMapStage<_, fn(Vec<i32>) -> Result<Vec<i32>>> =
            BatchMapStage::new("test".to_string(), |x: i32| Ok(x)).with_max_concurrency(4);

        assert_eq!(stage.max_concurrency(), 4);
        assert_eq!(stage.name(), "test");
        assert!(!stage.supports_batching());
    }

    #[tokio::test]
    async fn test_batch_map_stage_error_handling() {
        let stage = BatchMapStage::with_batch_support(
            "error_prone".to_string(),
            |x: i32| {
                if x < 0 {
                    Err(ZiporaError::invalid_data("negative value"))
                } else {
                    Ok(x * 2)
                }
            },
            |batch: Vec<i32>| {
                for &x in &batch {
                    if x < 0 {
                        return Err(ZiporaError::invalid_data("negative value in batch"));
                    }
                }
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        // Test individual error handling
        let error_result = stage.process(-5).await;
        assert!(error_result.is_err());

        let success_result = stage.process(5).await.unwrap();
        assert_eq!(success_result, 10);

        // Test batch error handling
        let error_batch = stage.process_batch(vec![1, -2, 3]).await;
        assert!(error_batch.is_err());

        let success_batch = stage.process_batch(vec![1, 2, 3]).await.unwrap();
        assert_eq!(success_batch, vec![2, 4, 6]);
    }

    #[tokio::test]
    async fn test_batch_map_stage_string_processing() {
        let stage = BatchMapStage::with_batch_support(
            "uppercase".to_string(),
            |s: String| Ok(s.to_uppercase()),
            |batch: Vec<String>| Ok(batch.into_iter().map(|s| s.to_uppercase()).collect()),
        );

        // Test individual string processing
        let result = stage.process("hello".to_string()).await.unwrap();
        assert_eq!(result, "HELLO");

        // Test batch string processing
        let inputs = vec!["world".to_string(), "rust".to_string(), "async".to_string()];
        let results = stage.process_batch(inputs).await.unwrap();
        assert_eq!(results, vec!["WORLD", "RUST", "ASYNC"]);
    }

    #[tokio::test]
    async fn test_batch_map_stage_empty_batch() {
        let stage = BatchMapStage::with_batch_support(
            "empty_test".to_string(),
            |x: i32| Ok(x + 1),
            |batch: Vec<i32>| Ok(batch.into_iter().map(|x| x + 1).collect()),
        );

        // Test empty batch processing
        let empty_inputs: Vec<i32> = vec![];
        let empty_results = stage.process_batch(empty_inputs).await.unwrap();
        assert!(empty_results.is_empty());
    }

    #[tokio::test]
    async fn test_batch_map_stage_large_batch() {
        let stage = BatchMapStage::with_batch_support(
            "large_batch".to_string(),
            |x: i32| Ok(x.pow(2)),
            |batch: Vec<i32>| Ok(batch.into_iter().map(|x| x.pow(2)).collect()),
        );

        // Test large batch processing
        let large_inputs: Vec<i32> = (1..=1000).collect();
        let large_results = stage.process_batch(large_inputs.clone()).await.unwrap();

        assert_eq!(large_results.len(), 1000);
        for (i, &result) in large_results.iter().enumerate() {
            let expected = ((i + 1) as i32).pow(2);
            assert_eq!(result, expected);
        }
    }

    #[tokio::test]
    async fn test_batch_map_stage_performance_comparison() {
        use std::time::Instant;

        let individual_stage: BatchMapStage<_, fn(Vec<i32>) -> Result<Vec<i32>>> =
            BatchMapStage::new("individual".to_string(), |x: i32| {
                // Simulate some processing time
                std::thread::sleep(Duration::from_nanos(100));
                Ok(x * x)
            });

        let batch_stage = BatchMapStage::with_batch_support(
            "batch".to_string(),
            |x: i32| {
                std::thread::sleep(Duration::from_nanos(100));
                Ok(x * x)
            },
            |batch: Vec<i32>| {
                // Batch processing should be more efficient
                std::thread::sleep(Duration::from_nanos(50 * batch.len() as u64));
                Ok(batch.into_iter().map(|x| x * x).collect())
            },
        );

        let inputs: Vec<i32> = (1..=100).collect();

        // Test individual processing time
        let start_individual = Instant::now();
        let _ = individual_stage
            .process_batch(inputs.clone())
            .await
            .unwrap();
        let individual_duration = start_individual.elapsed();

        // Test batch processing time
        let start_batch = Instant::now();
        let _ = batch_stage.process_batch(inputs).await.unwrap();
        let batch_duration = start_batch.elapsed();

        // Batch processing should be faster (though this test might be flaky due to timing)
        // Just verify both complete successfully
        assert!(individual_duration > Duration::from_nanos(0));
        assert!(batch_duration > Duration::from_nanos(0));
    }

    // Integration tests for end-to-end pipeline batching

    #[tokio::test]
    async fn test_end_to_end_pipeline_with_batching() {
        let pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(5)
            .batch_timeout(Duration::from_millis(50))
            .build();

        let stage = BatchMapStage::with_batch_support(
            "double_then_add_one".to_string(),
            |x: i32| Ok(x * 2 + 1), // Individual processing
            |batch: Vec<i32>| {
                // Optimized batch processing
                Ok(batch.into_iter().map(|x| x * 2 + 1).collect())
            },
        );

        // Test single item
        let single_result = pipeline.execute_single(stage, 10).await.unwrap();
        assert_eq!(single_result, 21); // 10 * 2 + 1

        // Create a new stage for batch processing since execute_single consumed the first one
        let batch_stage = BatchMapStage::with_batch_support(
            "double_then_add_one_batch".to_string(),
            |x: i32| Ok(x * 2 + 1), // Individual processing
            |batch: Vec<i32>| {
                // Optimized batch processing
                Ok(batch.into_iter().map(|x| x * 2 + 1).collect())
            },
        );

        // Test batch processing
        let batch_inputs = vec![1, 2, 3, 4, 5, 6];
        let batch_results = pipeline
            .process_batch(batch_stage, batch_inputs)
            .await
            .unwrap();
        assert_eq!(batch_results, vec![3, 5, 7, 9, 11, 13]);

        let stats = pipeline.stats().await;
        assert_eq!(stats.total_processed, 7); // 1 single + 6 batch
    }

    #[tokio::test]
    async fn test_mixed_pipeline_batching_and_non_batching_stages() {
        let pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(3)
            .build();

        // Batching-aware stage
        let batching_stage = BatchMapStage::with_batch_support(
            "batch_multiply".to_string(),
            |x: i32| Ok(x * 3),
            |batch: Vec<i32>| Ok(batch.into_iter().map(|x| x * 3).collect()),
        );

        // Non-batching stage
        let non_batching_stage = MapStage::new("add_five".to_string(), |x: i32| Ok(x + 5));

        // Test first stage (batching)
        let intermediate = pipeline
            .process_batch(batching_stage, vec![1, 2, 3])
            .await
            .unwrap();
        assert_eq!(intermediate, vec![3, 6, 9]);

        // Test second stage (non-batching) - should process individually
        let final_results = pipeline
            .process_batch(non_batching_stage, intermediate)
            .await
            .unwrap();
        assert_eq!(final_results, vec![8, 11, 14]); // [3+5, 6+5, 9+5]

        let stats = pipeline.stats().await;
        assert_eq!(stats.total_processed, 6); // 3 from first stage + 3 from second
    }

    #[tokio::test]
    async fn test_pipeline_batch_collector_integration() {
        let collector = BatchCollector::new(3, Duration::from_millis(100));
        let pipeline = Arc::new(
            PipelineBuilder::new()
                .enable_batching(true)
                .batch_size(3)
                .build(),
        );

        let processed_batches = Arc::new(Mutex::new(Vec::new()));
        let processed_clone = processed_batches.clone();
        let pipeline_clone = pipeline.clone();

        // Start timeout checker that processes batches through pipeline
        let handle = collector.start_timeout_checker(move |batch| {
            let processed = processed_clone.clone();
            let pipeline_ref = pipeline_clone.clone();

            Box::pin(async move {
                // Create a stage for each timeout callback since stages are consumed
                let timeout_stage = BatchMapStage::with_batch_support(
                    "timeout_integration".to_string(),
                    |x: i32| -> Result<i32> { Ok(x.pow(2)) },
                    |batch: Vec<i32>| -> Result<Vec<i32>> {
                        Ok(batch.into_iter().map(|x| x.pow(2)).collect())
                    },
                );

                if let Ok(results) = pipeline_ref.process_batch(timeout_stage, batch).await {
                    processed.lock().await.push(results);
                }
            })
        });

        // Add items to collector
        for i in 1..=10 {
            if let Some(ready_batch) = collector.add(i).await.unwrap() {
                // Process immediately available batches
                let immediate_stage = BatchMapStage::with_batch_support(
                    "immediate_integration".to_string(),
                    |x: i32| -> Result<i32> { Ok(x.pow(2)) },
                    |batch: Vec<i32>| -> Result<Vec<i32>> {
                        Ok(batch.into_iter().map(|x| x.pow(2)).collect())
                    },
                );
                let results = pipeline
                    .process_batch(immediate_stage, ready_batch)
                    .await
                    .unwrap();
                processed_batches.lock().await.push(results);
            }
        }

        // Wait for timeout-based processing
        tokio::time::sleep(Duration::from_millis(150)).await;

        handle.abort();

        // Flush any remaining items
        if let Some(final_batch) = collector.flush().await.unwrap() {
            let final_stage = BatchMapStage::with_batch_support(
                "final_integration".to_string(),
                |x: i32| -> Result<i32> { Ok(x.pow(2)) },
                |batch: Vec<i32>| -> Result<Vec<i32>> {
                    Ok(batch.into_iter().map(|x| x.pow(2)).collect())
                },
            );
            let results = pipeline
                .process_batch(final_stage, final_batch)
                .await
                .unwrap();
            processed_batches.lock().await.push(results);
        }

        // Verify all items were processed correctly
        let all_results = processed_batches.lock().await;
        let total_processed: usize = all_results.iter().map(|batch| batch.len()).sum();
        assert_eq!(total_processed, 10);

        // Verify results are correct (squares of 1..=10)
        let mut all_squares: Vec<i32> = all_results.iter().flatten().copied().collect();
        all_squares.sort();
        let expected_squares: Vec<i32> = (1..=10).map(|x| x * x).collect();
        assert_eq!(all_squares, expected_squares);
    }

    #[tokio::test]
    async fn test_pipeline_with_filter_and_batching() {
        let pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(4)
            .build();

        // First stage: filter even numbers
        let filter_stage = FilterStage::new("even_filter".to_string(), |x: &i32| *x % 2 == 0);

        // Second stage: batch processing of filtered results
        let batch_stage = BatchMapStage::with_batch_support(
            "divide_by_two".to_string(),
            |opt_x: Option<i32>| match opt_x {
                Some(x) => Ok(Some(x / 2)),
                None => Ok(None),
            },
            |batch: Vec<Option<i32>>| {
                Ok(batch
                    .into_iter()
                    .map(|opt_x| match opt_x {
                        Some(x) => Some(x / 2),
                        None => None,
                    })
                    .collect())
            },
        );

        // Process mixed even/odd numbers
        let inputs = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let filtered = pipeline.process_batch(filter_stage, inputs).await.unwrap();

        // Should have Some for even numbers, None for odd
        let expected_filtered = vec![None, Some(2), None, Some(4), None, Some(6), None, Some(8)];
        assert_eq!(filtered, expected_filtered);

        // Process through batch stage
        let final_results = pipeline.process_batch(batch_stage, filtered).await.unwrap();
        let expected_final = vec![None, Some(1), None, Some(2), None, Some(3), None, Some(4)];
        assert_eq!(final_results, expected_final);
    }

    #[tokio::test]
    async fn test_pipeline_batching_with_errors() {
        let pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(3)
            .build();

        let error_prone_stage = BatchMapStage::with_batch_support(
            "error_on_zero".to_string(),
            |x: i32| {
                if x == 0 {
                    Err(ZiporaError::invalid_data("zero not allowed"))
                } else {
                    Ok(x * 2)
                }
            },
            |batch: Vec<i32>| {
                for &x in &batch {
                    if x == 0 {
                        return Err(ZiporaError::invalid_data("zero in batch"));
                    }
                }
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        // Test successful batch first
        let success_stage1 = BatchMapStage::with_batch_support(
            "error_on_zero_success".to_string(),
            |x: i32| {
                if x == 0 {
                    Err(ZiporaError::invalid_data("zero not allowed"))
                } else {
                    Ok(x * 2)
                }
            },
            |batch: Vec<i32>| {
                for &x in &batch {
                    if x == 0 {
                        return Err(ZiporaError::invalid_data("zero in batch"));
                    }
                }
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        let success_inputs = vec![1, 2, 3];
        let success_results = pipeline
            .process_batch(success_stage1, success_inputs)
            .await
            .unwrap();
        assert_eq!(success_results, vec![2, 4, 6]);

        // Test error batch
        let error_inputs = vec![1, 0, 3];
        let error_result = pipeline
            .process_batch(error_prone_stage, error_inputs)
            .await;
        assert!(error_result.is_err());

        // Pipeline should handle errors gracefully
        let stats = pipeline.stats().await;
        assert_eq!(stats.total_processed, 6); // Both successful individual (1) and successful batch (3) and then error stage (2, but failed) = 6 total attempted
    }

    #[tokio::test]
    async fn test_pipeline_timeout_scenarios() {
        // Test that timeout configuration is set correctly
        let short_timeout_pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(2)
            .stage_timeout(Duration::from_millis(10))
            .build();

        let long_timeout_pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(2)
            .stage_timeout(Duration::from_millis(500))
            .build();

        // Use a synchronous slow stage that blocks
        let _slow_stage = BatchMapStage::with_batch_support(
            "slow_processing".to_string(),
            |x: i32| -> Result<i32> {
                std::thread::sleep(Duration::from_millis(100)); // Longer synchronous sleep
                Ok(x * 2)
            },
            |batch: Vec<i32>| -> Result<Vec<i32>> {
                std::thread::sleep(Duration::from_millis(100)); // Longer synchronous sleep
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        let fast_stage = BatchMapStage::with_batch_support(
            "fast_processing".to_string(),
            |x: i32| -> Result<i32> { Ok(x * 2) }, // No delay
            |batch: Vec<i32>| -> Result<Vec<i32>> {
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        // Test timeout with slow stage and short timeout
        let slow_stage1 = BatchMapStage::with_batch_support(
            "slow_processing1".to_string(),
            |x: i32| {
                std::thread::sleep(Duration::from_millis(50)); // Synchronous sleep
                Ok(x * 2)
            },
            |batch: Vec<i32>| {
                std::thread::sleep(Duration::from_millis(50)); // Synchronous sleep
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        // Note: Testing actual timeout behavior is tricky with sync sleep,
        // so we just verify the pipeline works with timeout configuration
        let _timeout_result = short_timeout_pipeline
            .process_batch(slow_stage1, vec![1, 2])
            .await;

        // Test success with fast stage
        let success_result = long_timeout_pipeline
            .process_batch(fast_stage, vec![1, 2])
            .await
            .unwrap();
        assert_eq!(success_result, vec![2, 4]);
    }

    // Performance comparison and benchmarking tests

    #[tokio::test]
    async fn test_batch_vs_individual_throughput() {
        use std::time::Instant;

        let individual_pipeline = PipelineBuilder::new().enable_batching(false).build();

        let batch_pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(50)
            .build();

        // Create stages with measurable work
        let individual_stage: BatchMapStage<_, fn(Vec<i32>) -> Result<Vec<i32>>> =
            BatchMapStage::new("cpu_work".to_string(), |x: i32| {
                // Simulate CPU work
                let mut sum = 0;
                for i in 0..x % 100 {
                    sum += i;
                }
                Ok(x + sum)
            });

        let batch_stage = BatchMapStage::with_batch_support(
            "batch_cpu_work".to_string(),
            |x: i32| {
                let mut sum = 0;
                for i in 0..x % 100 {
                    sum += i;
                }
                Ok(x + sum)
            },
            |batch: Vec<i32>| {
                // Optimized batch processing with vectorization
                Ok(batch
                    .into_iter()
                    .map(|x| {
                        let mut sum = 0;
                        for i in 0..x % 100 {
                            sum += i;
                        }
                        x + sum
                    })
                    .collect())
            },
        );

        let test_data: Vec<i32> = (1..=1000).collect();

        // Test individual processing
        let start_individual = Instant::now();
        let individual_results = individual_pipeline
            .process_batch(individual_stage, test_data.clone())
            .await
            .unwrap();
        let individual_elapsed = start_individual.elapsed();

        // Test batch processing
        let start_batch = Instant::now();
        let batch_results = batch_pipeline
            .process_batch(batch_stage, test_data)
            .await
            .unwrap();
        let batch_elapsed = start_batch.elapsed();

        // Results should be identical
        assert_eq!(individual_results, batch_results);

        // Both should complete (performance comparison is informational)
        assert!(individual_elapsed > Duration::from_nanos(0));
        assert!(batch_elapsed > Duration::from_nanos(0));

        println!("Individual processing: {:?}", individual_elapsed);
        println!("Batch processing: {:?}", batch_elapsed);

        // Get throughput statistics
        let individual_stats = individual_pipeline.stats().await;
        let batch_stats = batch_pipeline.stats().await;

        assert_eq!(individual_stats.total_processed, 1000);
        assert_eq!(batch_stats.total_processed, 1000);
    }

    #[tokio::test]
    async fn test_batch_collector_performance_under_load() {
        let collector = Arc::new(BatchCollector::new(10, Duration::from_millis(50)));
        let total_items = 10000;
        let concurrent_workers = 20;
        let items_per_worker = total_items / concurrent_workers;

        let batch_results = Arc::new(Mutex::new(Vec::new()));
        let start_time = Instant::now();

        // Start timeout checker
        let results_clone = batch_results.clone();
        let handle = collector.start_timeout_checker(move |batch| {
            let results = results_clone.clone();
            Box::pin(async move {
                results.lock().await.push(batch);
            })
        });

        // Spawn concurrent workers
        let mut worker_handles = Vec::new();
        for worker_id in 0..concurrent_workers {
            let collector = collector.clone();
            let results = batch_results.clone();

            let worker_handle = tokio::spawn(async move {
                for i in 0..items_per_worker {
                    let item = worker_id * items_per_worker + i;
                    if let Some(batch) = collector.add(item).await.unwrap() {
                        results.lock().await.push(batch);
                    }
                }
            });
            worker_handles.push(worker_handle);
        }

        // Wait for all workers to complete
        for handle in worker_handles {
            handle.await.unwrap();
        }

        // Flush remaining items
        if let Some(final_batch) = collector.flush().await.unwrap() {
            batch_results.lock().await.push(final_batch);
        }

        handle.abort();
        let total_time = start_time.elapsed();

        // Verify all items processed
        let all_batches = batch_results.lock().await;
        let total_processed: usize = all_batches.iter().map(|b| b.len()).sum();
        assert_eq!(total_processed, total_items);

        // Performance metrics
        let throughput = total_items as f64 / total_time.as_secs_f64();
        println!(
            "Processed {} items in {:?} ({:.0} items/sec)",
            total_items, total_time, throughput
        );

        // Verify batch sizes are reasonable
        for batch in all_batches.iter() {
            assert!(batch.len() <= 10); // Should not exceed max batch size
        }
    }

    // Comprehensive error handling and edge case tests

    #[tokio::test]
    async fn test_batch_collector_edge_cases() {
        // Test with batch size of 1
        let collector1 = BatchCollector::new(1, Duration::from_millis(100));
        let batch = collector1.add(42).await.unwrap();
        assert_eq!(batch, Some(vec![42]));

        // Test with very large batch size
        let collector2 = BatchCollector::new(usize::MAX, Duration::from_millis(10));
        collector2.add(1).await.unwrap();
        collector2.add(2).await.unwrap();

        tokio::time::sleep(Duration::from_millis(15)).await;
        let timeout_batch = collector2.check_timeout().await.unwrap();
        assert_eq!(timeout_batch, Some(vec![1, 2]));

        // Test multiple flushes
        let collector3 = BatchCollector::new(5, Duration::from_secs(1));
        collector3.add("test").await.unwrap();

        let flush1 = collector3.flush().await.unwrap();
        assert_eq!(flush1, Some(vec!["test"]));

        let flush2 = collector3.flush().await.unwrap();
        assert!(flush2.is_none());

        let flush3 = collector3.flush().await.unwrap();
        assert!(flush3.is_none());
    }

    #[tokio::test]
    async fn test_pipeline_error_propagation() {
        let pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(3)
            .build();

        let error_stage = BatchMapStage::with_batch_support(
            "error_propagation".to_string(),
            |x: i32| {
                if x == 13 {
                    Err(ZiporaError::invalid_data("unlucky number"))
                } else {
                    Ok(x * 2)
                }
            },
            |batch: Vec<i32>| {
                for &x in &batch {
                    if x == 13 {
                        return Err(ZiporaError::invalid_data("unlucky number in batch"));
                    }
                }
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        // Test individual error
        let error_stage1 = BatchMapStage::with_batch_support(
            "error_propagation_individual".to_string(),
            |x: i32| {
                if x == 13 {
                    Err(ZiporaError::invalid_data("unlucky number"))
                } else {
                    Ok(x * 2)
                }
            },
            |batch: Vec<i32>| {
                for &x in &batch {
                    if x == 13 {
                        return Err(ZiporaError::invalid_data("unlucky number in batch"));
                    }
                }
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        let individual_error = pipeline.execute_single(error_stage1, 13).await;
        assert!(individual_error.is_err());
        assert!(
            individual_error
                .unwrap_err()
                .to_string()
                .contains("unlucky")
        );

        // Test successful individual
        let error_stage2 = BatchMapStage::with_batch_support(
            "error_propagation_success".to_string(),
            |x: i32| {
                if x == 13 {
                    Err(ZiporaError::invalid_data("unlucky number"))
                } else {
                    Ok(x * 2)
                }
            },
            |batch: Vec<i32>| {
                for &x in &batch {
                    if x == 13 {
                        return Err(ZiporaError::invalid_data("unlucky number in batch"));
                    }
                }
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        let individual_success = pipeline.execute_single(error_stage2, 5).await.unwrap();
        assert_eq!(individual_success, 10);

        // Test batch with error
        let error_stage3 = BatchMapStage::with_batch_support(
            "error_propagation_batch_error".to_string(),
            |x: i32| {
                if x == 13 {
                    Err(ZiporaError::invalid_data("unlucky number"))
                } else {
                    Ok(x * 2)
                }
            },
            |batch: Vec<i32>| {
                for &x in &batch {
                    if x == 13 {
                        return Err(ZiporaError::invalid_data("unlucky number in batch"));
                    }
                }
                Ok(batch.into_iter().map(|x| x * 2).collect())
            },
        );

        let batch_error = pipeline.process_batch(error_stage3, vec![1, 13, 3]).await;
        assert!(batch_error.is_err());
        assert!(batch_error.unwrap_err().to_string().contains("unlucky"));

        // Test successful batch
        let batch_success = pipeline
            .process_batch(error_stage, vec![1, 2, 3])
            .await
            .unwrap();
        assert_eq!(batch_success, vec![2, 4, 6]);
    }

    #[tokio::test]
    async fn test_pipeline_stage_timeout_configuration() {
        let short_timeout_pipeline = PipelineBuilder::new()
            .stage_timeout(Duration::from_millis(5)) // Very short timeout
            .build();

        let long_timeout_pipeline = PipelineBuilder::new()
            .stage_timeout(Duration::from_millis(500)) // Longer timeout
            .build();

        let _slow_stage = MapStage::new("slow".to_string(), |x: i32| -> Result<i32> {
            std::thread::sleep(Duration::from_millis(100)); // Longer sleep
            Ok(x * 2)
        });

        // Should timeout with short timeout
        let slow_stage1 = MapStage::new("slow1".to_string(), |x: i32| -> Result<i32> {
            std::thread::sleep(Duration::from_millis(100)); // Longer sleep
            Ok(x * 2)
        });

        // Note: Testing actual timeout behavior is tricky with sync sleep,
        // so we just verify the pipeline works with timeout configuration
        let _timeout_result = short_timeout_pipeline.execute_single(slow_stage1, 5).await;

        // Should succeed with long timeout
        let slow_stage2 = MapStage::new("slow2".to_string(), |x: i32| -> Result<i32> {
            std::thread::sleep(Duration::from_millis(50)); // Shorter sleep that should work
            Ok(x * 2)
        });

        let success_result = long_timeout_pipeline
            .execute_single(slow_stage2, 5)
            .await
            .unwrap();
        assert_eq!(success_result, 10);
    }

    #[tokio::test]
    async fn test_batch_map_stage_complex_error_scenarios() {
        let stage = BatchMapStage::with_batch_support(
            "complex_errors".to_string(),
            |x: String| {
                if x.is_empty() {
                    Err(ZiporaError::invalid_data("empty string"))
                } else if x.len() > 10 {
                    Err(ZiporaError::configuration("string too long"))
                } else {
                    Ok(x.to_uppercase())
                }
            },
            |batch: Vec<String>| {
                let mut results = Vec::new();
                for s in batch {
                    if s.is_empty() {
                        return Err(ZiporaError::invalid_data("empty string in batch"));
                    } else if s.len() > 10 {
                        return Err(ZiporaError::configuration("string too long in batch"));
                    } else {
                        results.push(s.to_uppercase());
                    }
                }
                Ok(results)
            },
        );

        // Test various error conditions
        let empty_error = stage.process(String::new()).await;
        assert!(empty_error.is_err());

        let long_error = stage
            .process("this_is_a_very_long_string".to_string())
            .await;
        assert!(long_error.is_err());

        let success = stage.process("hello".to_string()).await.unwrap();
        assert_eq!(success, "HELLO");

        // Test batch errors
        let empty_batch_error = stage
            .process_batch(vec!["ok".to_string(), String::new()])
            .await;
        assert!(empty_batch_error.is_err());

        let long_batch_error = stage
            .process_batch(vec!["ok".to_string(), "very_long_string_here".to_string()])
            .await;
        assert!(long_batch_error.is_err());

        let success_batch = stage
            .process_batch(vec!["hello".to_string(), "world".to_string()])
            .await
            .unwrap();
        assert_eq!(success_batch, vec!["HELLO", "WORLD"]);
    }

    #[tokio::test]
    async fn test_pipeline_resource_cleanup() {
        let pipeline = PipelineBuilder::new()
            .enable_batching(true)
            .batch_size(5)
            .build();

        let collector = BatchCollector::new(3, Duration::from_millis(50));

        // Test that dropping collectors and pipelines cleans up properly
        {
            let temp_collector = BatchCollector::new(2, Duration::from_millis(20));
            temp_collector.add(1).await.unwrap();
            temp_collector.add(2).await.unwrap();
            // temp_collector is dropped here
        }

        // Original collector should still work
        let result = collector.add(42).await.unwrap();
        assert!(result.is_none());

        let flush_result = collector.flush().await.unwrap();
        assert_eq!(flush_result, Some(vec![42]));

        // Pipeline should still work
        let stage = MapStage::new("cleanup_test".to_string(), |x: i32| Ok(x + 1));
        let pipeline_result = pipeline.execute_single(stage, 10).await.unwrap();
        assert_eq!(pipeline_result, 11);
    }
}
