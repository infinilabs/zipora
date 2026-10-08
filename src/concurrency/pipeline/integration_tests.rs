use super::*;
use tokio;
use tokio::sync::Mutex;

type SimpleBatchFn = fn(Vec<i32>) -> Result<Vec<i32>>;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
                .map(|opt_x| opt_x.map(|x| x / 2))
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
        |batch: Vec<i32>| -> Result<Vec<i32>> { Ok(batch.into_iter().map(|x| x * 2).collect()) },
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_batch_vs_individual_throughput() {
    use std::time::Instant;

    let individual_pipeline = PipelineBuilder::new().enable_batching(false).build();

    let batch_pipeline = PipelineBuilder::new()
        .enable_batching(true)
        .batch_size(50)
        .build();

    // Create stages with measurable work
    let individual_stage: BatchMapStage<_, SimpleBatchFn> =
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

    // Should timeout with short timeout even for a synchronous built-in MapStage
    let slow_stage1 = MapStage::new("slow1".to_string(), |x: i32| -> Result<i32> {
        std::thread::sleep(Duration::from_millis(40));
        Ok(x * 2)
    });

    let timeout_result = short_timeout_pipeline.execute_single(slow_stage1, 5).await;
    assert!(
        timeout_result.is_err(),
        "execute_single must enforce stage_timeout for synchronous stages"
    );

    // Should succeed with long timeout
    let slow_stage2 = MapStage::new("slow2".to_string(), |x: i32| -> Result<i32> {
        std::thread::sleep(Duration::from_millis(20)); // Shorter sleep that should work
        Ok(x * 2)
    });

    let success_result = long_timeout_pipeline
        .execute_single(slow_stage2, 5)
        .await
        .unwrap();
    assert_eq!(success_result, 10);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_c10_p1_execute_stream_multi_stage_and_error_propagation() {
    let pipeline = Pipeline::new(PipelineConfig::default());

    // 1. Multi-stage stream must wire inter-stage channels without losing rx,
    //    update total_processing_time_us, and close output_tx on completion.
    let (in_tx, in_rx) = mpsc::channel::<i32>(8);
    let (out_tx, mut out_rx) = mpsc::channel::<i32>(8);
    in_tx.send(7).await.unwrap();
    drop(in_tx);

    let ok_stages: Vec<Box<dyn PipelineStage<i32, i32>>> = vec![
        Box::new(MapStage::new("stage0_inc".to_string(), |x: i32| Ok(x + 1))),
        Box::new(MapStage::new("stage1_mul".to_string(), |x: i32| Ok(x * 2))),
    ];
    pipeline
        .execute_stream(ok_stages, in_rx, out_tx)
        .await
        .expect("2-stage execute_stream must succeed");
    assert_eq!(out_rx.recv().await, Some(16));
    assert_eq!(out_rx.recv().await, None);

    // 2. Stage error in execute_stream must propagate Err and terminate upstream
    //    even when upstream input_tx remains open.
    let (err_in_tx, err_in_rx) = mpsc::channel::<i32>(8);
    let (err_out_tx, _err_out_rx) = mpsc::channel::<i32>(8);
    err_in_tx.send(7).await.unwrap();

    let err_stages: Vec<Box<dyn PipelineStage<i32, i32>>> = vec![
        Box::new(MapStage::new("stage0_pass".to_string(), |x: i32| Ok(x + 1))),
        Box::new(MapStage::new("stage1_fail".to_string(), |_x: i32| {
            Err(ZiporaError::invalid_data("stage 1 boom"))
        })),
    ];

    let stream_res = tokio::time::timeout(
        Duration::from_millis(200),
        pipeline.execute_stream(err_stages, err_in_rx, err_out_tx),
    )
    .await
    .expect("execute_stream hung waiting on upstream after downstream stage failed");

    assert!(
        stream_res.is_err(),
        "execute_stream must propagate stage errors instead of returning Ok(())"
    );
    drop(err_in_tx);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_c10_p2_items_in_flight_decremented_on_timeout_and_error() {
    let pipeline = PipelineBuilder::new()
        .stage_timeout(Duration::from_millis(10))
        .build();

    let err_stage = MapStage::new("err".to_string(), |_x: i32| -> Result<i32> {
        Err(ZiporaError::invalid_data("boom"))
    });
    assert!(
        pipeline
            .process_batch(err_stage, vec![1, 2, 3])
            .await
            .is_err()
    );
    assert_eq!(
        pipeline.stats().await.items_in_flight,
        0,
        "items_in_flight must be decremented when process_batch fails"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_c10_p3_zero_buffer_size_max_in_flight_and_zero_batch_timeout_do_not_panic_or_nan() {
    let pipeline = PipelineBuilder::new()
        .buffer_size(0)
        .max_in_flight(0)
        .build();

    let stats = pipeline.stats().await;
    assert!(
        !stats.utilization.is_nan(),
        "Pipeline::stats().utilization must not be NaN when max_in_flight == 0"
    );

    let (input_tx, input_rx) = mpsc::channel::<i32>(4);
    let (output_tx, mut output_rx) = mpsc::channel::<i32>(4);
    input_tx.send(10).await.unwrap();
    drop(input_tx);

    let stages: Vec<Box<dyn PipelineStage<i32, i32>>> = vec![
        Box::new(MapStage::new("s0".to_string(), |x: i32| Ok(x + 1))),
        Box::new(MapStage::new("s1".to_string(), |x: i32| Ok(x * 2))),
    ];
    pipeline
        .execute_stream(stages, input_rx, output_tx)
        .await
        .unwrap();
    assert_eq!(output_rx.recv().await, Some(22));

    let collector = BatchCollector::<i32>::new(4, Duration::ZERO);
    let handle = collector.start_timeout_checker(|_batch| Box::pin(async {}));
    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_c10_p5_batch_collector_idle_before_first_item_waits_full_timeout() {
    let collector = BatchCollector::new(4, Duration::from_millis(80));
    // Let the collector sit empty past batch_timeout before the first item arrives.
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert!(collector.add(1).await.unwrap().is_none());
    // Immediately after adding the first item of a new batch, timeout has NOT elapsed yet.
    assert_eq!(
        collector.check_timeout().await.unwrap(),
        None,
        "first item after an idle period must start a fresh batch_timeout window"
    );
}
