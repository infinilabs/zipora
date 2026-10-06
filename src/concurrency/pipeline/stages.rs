//! Batch collector and built-in stage implementations for [`super::Pipeline`].

use super::PipelineStage;
use crate::error::Result;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

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
        let was_empty = buffer.is_empty();
        buffer.push_back(item);

        if buffer.len() >= self.max_batch_size {
            let batch = buffer.drain(..).collect();
            *self.last_flush.lock().await = Instant::now();
            Ok(Some(batch))
        } else {
            if was_empty {
                // Start the batch timeout window from the arrival of the first item in a batch
                // rather than the previous flush timestamp across an idle period.
                *self.last_flush.lock().await = Instant::now();
            }
            Ok(None)
        }
    }

    /// Check if a timeout-based flush should occur
    /// Returns Some(batch) if items should be flushed due to timeout
    pub async fn check_timeout(&self) -> Result<Option<Vec<T>>> {
        let mut buffer = self.buffer.lock().await;
        if buffer.is_empty() {
            return Ok(None);
        }
        let mut last_flush = self.last_flush.lock().await;
        if last_flush.elapsed() >= self.batch_timeout {
            let batch = buffer.drain(..).collect();
            *last_flush = Instant::now();
            return Ok(Some(batch));
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
            let check_period = (collector.batch_timeout / 4).max(Duration::from_millis(1));
            let mut interval = tokio::time::interval(check_period);

            loop {
                interval.tick().await;

                if let Ok(Some(batch)) = collector.check_timeout().await {
                    on_batch(batch).await;
                }
            }
        })
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
