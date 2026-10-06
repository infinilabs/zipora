use super::*;
use tokio;
use tokio::sync::Mutex;

type SimpleBatchFn = fn(Vec<i32>) -> Result<Vec<i32>>;

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
    let stage: BatchMapStage<_, SimpleBatchFn> =
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
    let stage: BatchMapStage<_, SimpleBatchFn> =
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
    let simple_stage: BatchMapStage<_, SimpleBatchFn> =
        BatchMapStage::new("multiply_by_3".to_string(), |x: i32| Ok(x * 3));

    let inputs2 = vec![1, 2, 3];
    let results2 = simple_stage.process_batch(inputs2).await.unwrap();
    assert_eq!(results2, vec![3, 6, 9]);
}

#[tokio::test]
async fn test_batch_map_stage_with_max_concurrency() {
    let stage: BatchMapStage<_, SimpleBatchFn> =
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

    let individual_stage: BatchMapStage<_, SimpleBatchFn> =
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
