//! Pipeline processing for streaming data operations

use crate::error::{ToplingError, Result};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, RwLock};
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
    
    /// Process a batch of items (must be implemented if batching is supported)
    fn process_batch(&self, _inputs: Vec<T>) -> Pin<Box<dyn Future<Output = Result<Vec<U>>> + Send + '_>>
    where
        T: 'static,
        U: 'static,
        Self: Sized,
    {
        Box::pin(async move {
            Err(ToplingError::not_supported("batching not implemented for this stage"))
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
        
        let result = timeout(self.config.stage_timeout, stage.process(input)).await
            .map_err(|_| ToplingError::configuration("stage timeout"))?;
        
        let processing_time = start_time.elapsed().as_micros() as u64;
        self.stats.total_processing_time_us.fetch_add(processing_time, Ordering::Relaxed);
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
            return Err(ToplingError::invalid_data("no stages provided"));
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
        for _ in 0..(num_stages - 1) {  // One less channel than stages
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
                    pipeline_stats.items_in_flight.fetch_add(1, Ordering::Relaxed);
                    
                    let result = timeout(config.stage_timeout, stage.process(item)).await;
                    
                    let processing_time = start_time.elapsed().as_micros() as u64;
                    stage_stats.total_time_us.fetch_add(processing_time, Ordering::Relaxed);
                    stage_stats.processed.fetch_add(1, Ordering::Relaxed);
                    stage_stats.active_items.fetch_sub(1, Ordering::Relaxed);
                    
                    match result {
                        Ok(Ok(output)) => {
                            if output_tx.send(output).await.is_err() {
                                break; // Output channel closed
                            }
                            pipeline_stats.total_processed.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(Err(_)) | Err(_) => {
                            // Stage failed or timed out
                            break;
                        }
                    }
                    
                    pipeline_stats.items_in_flight.fetch_sub(1, Ordering::Relaxed);
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
    pub async fn process_batch<T, U, S>(
        &self,
        stage: S,
        inputs: Vec<T>,
    ) -> Result<Vec<U>>
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
        
        self.stats.items_in_flight.fetch_add(batch_size, Ordering::Relaxed);
        
        let result = if stage.supports_batching() && self.config.enable_batching {
            timeout(self.config.stage_timeout, stage.process_batch(inputs)).await
                .map_err(|_| ToplingError::configuration("batch processing timeout"))?
        } else {
            // Process individually
            let mut results = Vec::with_capacity(batch_size);
            for input in inputs {
                let output = timeout(self.config.stage_timeout, stage.process(input)).await
                    .map_err(|_| ToplingError::configuration("stage timeout"))??;
                results.push(output);
            }
            Ok(results)
        };
        
        let processing_time = start_time.elapsed().as_micros() as u64;
        self.stats.total_processing_time_us.fetch_add(processing_time, Ordering::Relaxed);
        self.stats.total_processed.fetch_add(batch_size as u64, Ordering::Relaxed);
        self.stats.items_in_flight.fetch_sub(batch_size, Ordering::Relaxed);
        
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
        let stage_stats = stage_stats_inner.iter().map(|stats| {
            let processed = stats.processed.load(Ordering::Relaxed);
            let total_time = stats.total_time_us.load(Ordering::Relaxed);
            let avg_time_us = if processed > 0 { total_time / processed } else { 0 };
            let active_items = stats.active_items.load(Ordering::Relaxed);
            
            StageStats {
                name: stats.name.clone(),
                processed,
                avg_time_us,
                active_items,
                utilization: active_items as f64 / self.config.max_in_flight as f64,
            }
        }).collect();
        
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
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }
    
    pub fn buffer_size(mut self, buffer_size: usize) -> Self {
        self.config.buffer_size = buffer_size;
        self
    }
    
    pub fn max_in_flight(mut self, max_in_flight: usize) -> Self {
        self.config.max_in_flight = max_in_flight;
        self
    }
    
    pub fn stage_timeout(mut self, stage_timeout: Duration) -> Self {
        self.config.stage_timeout = stage_timeout;
        self
    }
    
    pub fn enable_batching(mut self, enable: bool) -> Self {
        self.config.enable_batching = enable;
        self
    }
    
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }
    
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

/// A simple map stage that applies a function to each item
pub struct MapStage<F> {
    name: String,
    func: F,
}

impl<F> MapStage<F> {
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
        
        let result = pipeline.execute_two_stage(stage1, stage2, 20).await.unwrap();
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
        let stage = FilterStage::new("even_only".to_string(), |x: &i32| *x % 2 == 0);
        
        let result1 = pipeline.execute_single(&stage, 4).await.unwrap();
        assert_eq!(result1, Some(4));
        
        let result2 = pipeline.execute_single(&stage, 5).await.unwrap();
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
}