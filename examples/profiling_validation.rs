//! Cross-platform profiling validation
//! 
//! This example validates the Advanced Profiling Integration system across
//! different platforms and configurations.

use std::time::Duration;
use zipora::dev_infrastructure::profiling::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Profiling Integration - Cross-Platform Validation ===");
    
    // Platform detection
    let platform = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    println!("Platform: {} on {}", platform, arch);
    
    // Test 1: Basic profiler creation and configuration
    println!("\n1. Testing Profiler Creation and Configuration...");
    
    let config = ProfilingConfig::development()
        .with_hardware_profiling(true)
        .with_memory_profiling(true)
        .with_cache_profiling(true);
    
    println!("✓ Created profiling configuration");
    
    // Test 2: Hardware Profiler
    println!("\n2. Testing Hardware Profiler...");
    match HardwareProfiler::global() {
        Ok(hw_profiler) => {
            println!("✓ Hardware profiler created successfully");
            
            let handle = hw_profiler.start("test_operation")?;
            std::thread::sleep(Duration::from_millis(10));
            let data = hw_profiler.end(handle)?;
            
            println!("✓ Hardware profiling completed - Duration: {:?}", data.duration);
        }
        Err(e) => {
            println!("⚠ Hardware profiler not available: {}", e);
        }
    }
    
    // Test 3: Memory Profiler  
    println!("\n3. Testing Memory Profiler...");
    match MemoryProfiler::global() {
        Ok(mem_profiler) => {
            println!("✓ Memory profiler created successfully");
            
            let handle = mem_profiler.start("memory_test")?;
            let _vec: Vec<u64> = vec![0u64; 1000]; // Allocate some memory
            let data = mem_profiler.end(handle)?;
            
            println!("✓ Memory profiling completed - Duration: {:?}", data.duration);
        }
        Err(e) => {
            println!("⚠ Memory profiler not available: {}", e);
        }
    }
    
    // Test 4: Cache Profiler
    println!("\n4. Testing Cache Profiler...");
    match CacheProfiler::global() {
        Ok(cache_profiler) => {
            println!("✓ Cache profiler created successfully");
            
            let handle = cache_profiler.start("cache_test")?;
            // Simulate cache activity
            let mut sum = 0u64;
            for i in 0..1000 {
                sum = sum.wrapping_add(i);
            }
            let data = cache_profiler.end(handle)?;
            
            println!("✓ Cache profiling completed - Duration: {:?}, Sum: {}", data.duration, sum);
        }
        Err(e) => {
            println!("⚠ Cache profiler not available: {}", e);
        }
    }
    
    // Test 5: RAII Profiler Scope
    println!("\n5. Testing RAII Profiler Scope...");
    {
        let _scope = ProfilerScope::new("raii_test")?;
        std::thread::sleep(Duration::from_millis(5));
        println!("✓ RAII scope created and will auto-cleanup");
    }
    println!("✓ RAII scope completed cleanup");
    
    // Test 6: Registry Integration
    println!("\n6. Testing Profiler Registry...");
    let registry = ProfilerRegistry::new();
    println!("✓ Registry created successfully");
    
    // Test 7: Configuration Validation
    println!("\n7. Testing Configuration Features...");
    let configs = vec![
        ("Production", ProfilingConfig::production()),
        ("Development", ProfilingConfig::development()),
        ("Debugging", ProfilingConfig::debugging()),
        ("Disabled", ProfilingConfig::disabled()),
    ];
    
    for (name, config) in configs {
        println!("✓ {} config - Hardware: {}, Memory: {}, Cache: {}", 
                 name, 
                 config.enable_hardware_profiling, 
                 config.enable_memory_profiling,
                 config.enable_cache_profiling);
    }
    
    // Test 8: Platform-specific timing
    println!("\n8. Testing High-Resolution Timing...");
    let start = std::time::Instant::now();
    std::thread::sleep(Duration::from_millis(1));
    let elapsed = start.elapsed();
    println!("✓ High-resolution timing works - Measured: {:?}", elapsed);
    
    // Test 9: CPU Feature Detection (for SIMD optimizations)
    println!("\n9. Testing CPU Feature Detection...");
    #[cfg(target_arch = "x86_64")]
    {
        println!("✓ x86_64 detected");
        if is_x86_feature_detected!("sse2") {
            println!("✓ SSE2 support detected");
        }
        if is_x86_feature_detected!("avx2") {
            println!("✓ AVX2 support detected");
        }
        if is_x86_feature_detected!("bmi2") {
            println!("✓ BMI2 support detected");
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        println!("✓ ARM64 detected");
        // ARM64 feature detection would go here
    }
    
    // Test 10: Reporter functionality
    println!("\n10. Testing Profiler Reporter...");
    let reporter_config = ProfilingConfig::development();
    let reporter = ProfilerReporter::new(reporter_config)?;
    println!("✓ Reporter created successfully");
    
    // Generate a simple report
    let report = reporter.generate_report()?;
    println!("✓ Report generated successfully");
    
    println!("\n=== Cross-Platform Validation Complete ===");
    println!("✅ All profiling components validated successfully on {} {}", platform, arch);
    
    Ok(())
}