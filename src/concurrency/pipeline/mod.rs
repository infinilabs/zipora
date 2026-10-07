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
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
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
    /// Timeout for individual stage processing.
    ///
    /// Asynchronous stages are cancelled at `.await` points via `tokio::time::timeout`.
    /// Stages that block the executor thread synchronously without yielding (such as
    /// [`MapStage`] or [`FilterStage`] closures) cannot be preempted mid-poll and are
    /// instead checked against `stage_timeout` after the synchronous poll returns.
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

mod stages;
pub use stages::{BatchCollector, BatchMapStage, FilterStage, MapStage};

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

struct AtomicCountGuard<'a> {
    counter: &'a AtomicUsize,
    count: usize,
}

impl<'a> AtomicCountGuard<'a> {
    fn new(counter: &'a AtomicUsize, count: usize) -> Self {
        // ORDERING: Relaxed is sufficient for telemetry gauge increments.
        counter.fetch_add(count, Ordering::Relaxed);
        Self { counter, count }
    }
}

impl Drop for AtomicCountGuard<'_> {
    fn drop(&mut self) {
        // ORDERING: Relaxed is sufficient for telemetry gauge decrements.
        self.counter.fetch_sub(self.count, Ordering::Relaxed);
    }
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

    /// Execute a single-stage pipeline.
    ///
    /// Enforces [`PipelineConfig::stage_timeout`] via `tokio::time::timeout` across
    /// `.await` points and a post-hoc `elapsed > stage_timeout` check after synchronous
    /// single-poll stages return.
    pub async fn execute_single<T, U, S>(&self, stage: S, input: T) -> Result<U>
    where
        S: PipelineStage<T, U>,
        T: Send + 'static,
        U: Send + 'static,
    {
        let start_time = Instant::now();
        let _in_flight = AtomicCountGuard::new(&self.stats.items_in_flight, 1);

        let result = timeout(self.config.stage_timeout, stage.process(input))
            .await
            .map_err(|_| ZiporaError::configuration("stage timeout"))?;

        let elapsed = start_time.elapsed();
        let processing_time = elapsed.as_micros() as u64;
        // ORDERING: Relaxed is sufficient for cumulative processing time telemetry.
        self.stats
            .total_processing_time_us
            .fetch_add(processing_time, Ordering::Relaxed);
        // ORDERING: Relaxed is sufficient for cumulative item count telemetry.
        self.stats.total_processed.fetch_add(1, Ordering::Relaxed);

        if elapsed > self.config.stage_timeout {
            return Err(ZiporaError::configuration("stage timeout"));
        }

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

        // Initialize stage statistics locally first so concurrent execute_stream calls
        // on the same Pipeline cannot overwrite stage_stats mid-loop with a shorter Vec.
        let local_stage_stats: Vec<Arc<StageStatsInner>> = stages
            .iter()
            .map(|stage| {
                Arc::new(StageStatsInner {
                    name: stage.name().to_string(),
                    processed: AtomicU64::new(0),
                    total_time_us: AtomicU64::new(0),
                    active_items: AtomicUsize::new(0),
                })
            })
            .collect();
        {
            let mut stage_stats = self.stats.stage_stats.write().await;
            *stage_stats = local_stage_stats.clone();
        }

        let num_stages = stages.len();
        let channel_cap = self.config.buffer_size.max(1);

        // Keep inter-stage senders and receivers in separate vectors so taking stage i's
        // output sender does not consume stage (i + 1)'s input receiver.
        let mut inter_txs: Vec<Option<mpsc::Sender<T>>> = Vec::with_capacity(num_stages - 1);
        let mut inter_rxs: Vec<Option<mpsc::Receiver<T>>> = Vec::with_capacity(num_stages - 1);
        for _ in 0..(num_stages - 1) {
            let (tx, rx) = mpsc::channel(channel_cap);
            inter_txs.push(Some(tx));
            inter_rxs.push(Some(rx));
        }

        // Spawn tasks for each stage
        let mut handles = Vec::with_capacity(num_stages);
        let mut current_input = Some(input_rx);
        let mut final_output = Some(output_tx);

        for (i, stage) in stages.into_iter().enumerate() {
            let stage_stats = local_stage_stats[i].clone();

            let mut stage_input_rx = if i == 0 {
                current_input.take().ok_or_else(|| {
                    ZiporaError::invalid_state("pipeline: input missing for first stage")
                })?
            } else {
                inter_rxs[i - 1].take().ok_or_else(|| {
                    ZiporaError::invalid_state("pipeline: channel missing for prior stage")
                })?
            };

            let stage_output_tx = if i == num_stages - 1 {
                final_output.take().ok_or_else(|| {
                    ZiporaError::invalid_state("pipeline: output missing for final stage")
                })?
            } else {
                inter_txs[i].take().ok_or_else(|| {
                    ZiporaError::invalid_state("pipeline: channel missing for next stage")
                })?
            };

            let config = self.config.clone();
            let pipeline_stats = self.stats.clone();

            let handle = tokio::spawn(async move {
                let mut stage_result: Result<()> = Ok(());

                loop {
                    if stage_output_tx.is_closed() {
                        break;
                    }
                    let maybe_item = match stage_input_rx.try_recv() {
                        Ok(item) => Some(item),
                        Err(mpsc::error::TryRecvError::Empty) => {
                            tokio::select! {
                                biased;
                                () = stage_output_tx.closed() => break,
                                item = stage_input_rx.recv() => item,
                            }
                        }
                        Err(mpsc::error::TryRecvError::Disconnected) => None,
                    };
                    let Some(item) = maybe_item else {
                        break;
                    };

                    let start_time = Instant::now();
                    let _stage_active = AtomicCountGuard::new(&stage_stats.active_items, 1);
                    let _in_flight = AtomicCountGuard::new(&pipeline_stats.items_in_flight, 1);

                    let result = timeout(config.stage_timeout, stage.process(item)).await;

                    let elapsed = start_time.elapsed();
                    let processing_time = elapsed.as_micros() as u64;
                    // ORDERING: Relaxed is sufficient for per-stage cumulative time telemetry.
                    stage_stats
                        .total_time_us
                        .fetch_add(processing_time, Ordering::Relaxed);
                    // ORDERING: Relaxed is sufficient for per-stage processed count telemetry.
                    stage_stats.processed.fetch_add(1, Ordering::Relaxed);
                    // ORDERING: Relaxed is sufficient for pipeline cumulative time telemetry.
                    pipeline_stats
                        .total_processing_time_us
                        .fetch_add(processing_time, Ordering::Relaxed);

                    match result {
                        Ok(Ok(output)) if elapsed <= config.stage_timeout => {
                            if stage_output_tx.send(output).await.is_err() {
                                break; // Downstream receiver closed
                            }
                            // ORDERING: Relaxed is sufficient for pipeline processed count telemetry.
                            pipeline_stats
                                .total_processed
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(Err(err)) => {
                            stage_result = Err(err);
                            break;
                        }
                        Ok(Ok(_)) | Err(_) => {
                            stage_result = Err(ZiporaError::configuration("stage timeout"));
                            break;
                        }
                    }
                }

                drop(stage_input_rx);
                drop(stage_output_tx);
                stage_result
            });

            handles.push(handle);
        }

        // Wait for all stages to complete and propagate the first stage failure if any.
        let mut first_err: Option<ZiporaError> = None;
        for handle in handles {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    if first_err.is_none() {
                        first_err = Some(err);
                    }
                }
                Err(join_err) => {
                    if first_err.is_none() {
                        first_err = Some(ZiporaError::system_error(format!(
                            "pipeline stage task failed: {}",
                            join_err
                        )));
                    }
                }
            }
        }

        if let Some(err) = first_err {
            return Err(err);
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
        let _in_flight = AtomicCountGuard::new(&self.stats.items_in_flight, batch_size);

        let result = if stage.supports_batching() && self.config.enable_batching {
            let batch_res = timeout(self.config.stage_timeout, stage.process_batch(inputs))
                .await
                .map_err(|_| ZiporaError::configuration("batch processing timeout"))?;
            if start_time.elapsed() > self.config.stage_timeout {
                return Err(ZiporaError::configuration("batch processing timeout"));
            }
            batch_res
        } else {
            // Process individually
            let mut results = Vec::with_capacity(batch_size);
            for input in inputs {
                let item_start = Instant::now();
                let output = timeout(self.config.stage_timeout, stage.process(input))
                    .await
                    .map_err(|_| ZiporaError::configuration("stage timeout"))??;
                if item_start.elapsed() > self.config.stage_timeout {
                    return Err(ZiporaError::configuration("stage timeout"));
                }
                results.push(output);
            }
            Ok(results)
        };

        let processing_time = start_time.elapsed().as_micros() as u64;
        // ORDERING: Relaxed is sufficient for cumulative processing time telemetry.
        self.stats
            .total_processing_time_us
            .fetch_add(processing_time, Ordering::Relaxed);
        // ORDERING: Relaxed is sufficient for cumulative item count telemetry.
        self.stats
            .total_processed
            .fetch_add(batch_size as u64, Ordering::Relaxed);

        result
    }

    /// Get current pipeline statistics
    pub async fn stats(&self) -> PipelineStats {
        // ORDERING: Relaxed is sufficient for point-in-time telemetry reads.
        let total_processed = self.stats.total_processed.load(Ordering::Relaxed);
        // ORDERING: Relaxed is sufficient for point-in-time telemetry reads.
        let total_time = self.stats.total_processing_time_us.load(Ordering::Relaxed);

        let avg_processing_time_us = total_time.checked_div(total_processed).unwrap_or(0);

        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        let throughput_per_sec = if elapsed_secs > 0.0 {
            total_processed as f64 / elapsed_secs
        } else {
            0.0
        };

        // ORDERING: Relaxed is sufficient for point-in-time telemetry reads.
        let items_in_flight = self.stats.items_in_flight.load(Ordering::Relaxed);
        let utilization = if self.config.max_in_flight > 0 {
            items_in_flight as f64 / self.config.max_in_flight as f64
        } else {
            0.0
        };

        let stage_stats_inner = self.stats.stage_stats.read().await;
        let stage_stats = stage_stats_inner
            .iter()
            .map(|stats| {
                // ORDERING: Relaxed is sufficient for point-in-time stage telemetry reads.
                let processed = stats.processed.load(Ordering::Relaxed);
                // ORDERING: Relaxed is sufficient for point-in-time stage telemetry reads.
                let total_time = stats.total_time_us.load(Ordering::Relaxed);
                let avg_time_us = total_time.checked_div(processed).unwrap_or(0);
                // ORDERING: Relaxed is sufficient for point-in-time stage telemetry reads.
                let active_items = stats.active_items.load(Ordering::Relaxed);
                let stage_utilization = if self.config.max_in_flight > 0 {
                    active_items as f64 / self.config.max_in_flight as f64
                } else {
                    0.0
                };

                StageStats {
                    name: stats.name.clone(),
                    processed,
                    avg_time_us,
                    active_items,
                    utilization: stage_utilization,
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

#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod tests;
