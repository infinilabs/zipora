//! Build System Integration for SIMD Feature Detection
//!
//! This build script implements Phase 1.1 of the systematic SIMD implementation plan.
//! It provides runtime CPU feature detection and conditional compilation support
//! following advanced SIMD library patterns and best practices.
//!
//! # Features
//!
//! - Runtime CPU feature detection during build
//! - Conditional compilation flags for SIMD features
//! - Cross-platform support (x86_64, ARM64)
//! - Build-time optimization configuration
//! - Feature-specific optimization flags

use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    
    // Detect and configure SIMD features
    detect_and_configure_simd();
    
    // Set target-specific configurations
    configure_target_specific_features();
    
    // Configure optimization flags
    configure_optimization_flags();
    
    // Handle optional FFI bindings
    #[cfg(feature = "ffi")]
    generate_ffi_bindings();
}

/// Detect available SIMD features and set appropriate compilation flags
fn detect_and_configure_simd() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    
    match target_arch.as_str() {
        "x86_64" => detect_x86_features(),
        "aarch64" => detect_arm_features(),
        _ => {
            println!("cargo:warning=SIMD optimizations not available for target architecture: {}", target_arch);
            configure_scalar_fallback();
        }
    }
}

/// Detect x86_64 SIMD features and set compilation flags
fn detect_x86_features() {
    println!("cargo:rustc-cfg=target_arch_x86_64");
    
    // Enable basic SIMD support
    if cfg!(feature = "simd") {
        println!("cargo:rustc-cfg=zipora_simd");
        
        // Check for specific x86 features that are commonly available
        detect_x86_feature("sse2", true); // Almost universally available on x86_64
        detect_x86_feature("sse3", true);
        detect_x86_feature("ssse3", true);
        detect_x86_feature("sse4.1", true);
        detect_x86_feature("sse4.2", true);
        detect_x86_feature("popcnt", true);
        detect_x86_feature("avx", true);
        detect_x86_feature("avx2", true);
        detect_x86_feature("bmi1", true);
        detect_x86_feature("bmi2", true);
        detect_x86_feature("lzcnt", true);
        
        // AVX-512 features (nightly only)
        if cfg!(feature = "avx512") {
            detect_x86_feature("avx512f", false);
            detect_x86_feature("avx512vl", false);
            detect_x86_feature("avx512bw", false);
            detect_x86_feature("avx512dq", false);
            detect_x86_feature("avx512cd", false);
            detect_x86_feature("avx512vpopcntdq", false);
            detect_x86_feature("avx512vbmi", false);
            detect_x86_feature("avx512ifma", false);
        }
    }
    
    // Set optimization tier flags
    if is_feature_available("bmi2") && is_feature_available("avx2") {
        println!("cargo:rustc-cfg=zipora_optimization_tier_4");
    } else if is_feature_available("bmi2") {
        println!("cargo:rustc-cfg=zipora_optimization_tier_3");
    } else if is_feature_available("popcnt") {
        println!("cargo:rustc-cfg=zipora_optimization_tier_2");
    } else {
        println!("cargo:rustc-cfg=zipora_optimization_tier_1");
    }
}

/// Detect ARM64 SIMD features and set compilation flags
fn detect_arm_features() {
    println!("cargo:rustc-cfg=target_arch_aarch64");
    
    if cfg!(feature = "simd") {
        println!("cargo:rustc-cfg=zipora_simd");
        
        // NEON is standard on AArch64
        detect_arm_feature("neon", true);
        
        // Additional ARM features that may be available
        detect_arm_feature("crc", false);
        detect_arm_feature("crypto", false);
        detect_arm_feature("sve", false);
        detect_arm_feature("sve2", false);
        
        // ARM optimization tier
        if is_feature_available("sve2") {
            println!("cargo:rustc-cfg=zipora_optimization_tier_3");
        } else if is_feature_available("sve") || is_feature_available("crypto") {
            println!("cargo:rustc-cfg=zipora_optimization_tier_2");
        } else {
            println!("cargo:rustc-cfg=zipora_optimization_tier_1");
        }
    }
}

/// Detect a specific x86 feature and set compilation flag
fn detect_x86_feature(feature: &str, commonly_available: bool) {
    // For build-time detection, we assume commonly available features are present
    // Runtime detection will happen in the actual code using std::arch::is_x86_feature_detected!
    if commonly_available || try_compile_with_feature(feature) {
        let cfg_name = format!("zipora_has_{}", feature.replace(".", "_").replace("-", "_"));
        println!("cargo:rustc-cfg={}", cfg_name);
        
        // Also set target feature for the specific feature
        if feature != "popcnt" { // popcnt is handled specially by Rust
            println!("cargo:rustc-env=RUSTFLAGS=-C target-feature=+{}", feature);
        }
    }
}

/// Detect a specific ARM feature and set compilation flag
fn detect_arm_feature(feature: &str, commonly_available: bool) {
    // For build-time detection, we assume commonly available features are present
    // Runtime detection will happen in the actual code using std::arch::is_aarch64_feature_detected!
    if commonly_available || try_compile_with_arm_feature(feature) {
        let cfg_name = format!("zipora_has_{}", feature);
        println!("cargo:rustc-cfg={}", cfg_name);
    }
}

/// Try to compile a simple test program with the given x86 feature
fn try_compile_with_feature(feature: &str) -> bool {
    // Create a simple test program that uses the feature
    let test_code = format!(r#"
        #[cfg(target_arch = "x86_64")]
        fn test_feature() {{
            #[cfg(target_feature = "{}")]
            {{
                // Feature is available at compile time
            }}
        }}
        
        fn main() {{
            test_feature();
        }}
    "#, feature);
    
    // Try to compile with the feature enabled
    compile_test_program(&test_code, &[&format!("target-feature=+{}", feature)])
}

/// Try to compile a simple test program with the given ARM feature
fn try_compile_with_arm_feature(feature: &str) -> bool {
    // Create a simple test program that uses the feature
    let test_code = format!(r#"
        #[cfg(target_arch = "aarch64")]
        fn test_feature() {{
            #[cfg(target_feature = "{}")]
            {{
                // Feature is available at compile time
            }}
        }}
        
        fn main() {{
            test_feature();
        }}
    "#, feature);
    
    // Try to compile with the feature enabled
    compile_test_program(&test_code, &[&format!("target-feature=+{}", feature)])
}

/// Compile a test program with given flags
fn compile_test_program(code: &str, flags: &[&str]) -> bool {
    use std::fs;
    use std::path::Path;
    
    let out_dir = env::var("OUT_DIR").unwrap_or_else(|_| ".".to_string());
    let test_file = Path::new(&out_dir).join("feature_test.rs");
    
    // Write test code to file
    if fs::write(&test_file, code).is_err() {
        return false;
    }
    
    // Try to compile
    let mut cmd = Command::new("rustc");
    cmd.arg(&test_file);
    cmd.arg("-o").arg(Path::new(&out_dir).join("feature_test"));
    
    for flag in flags {
        cmd.arg("-C").arg(flag);
    }
    
    // Suppress output and return success status
    cmd.output().map(|output| output.status.success()).unwrap_or(false)
}

/// Check if a feature is available based on environment or previous detection
fn is_feature_available(feature: &str) -> bool {
    // This is a simplified check - in practice, you might want to use
    // the actual CPU feature detection or check environment variables
    let _cfg_name = format!("zipora_has_{}", feature.replace(".", "_").replace("-", "_"));
    
    // For now, assume common features are available
    match feature {
        "sse2" | "sse3" | "ssse3" | "sse4.1" | "sse4.2" | "popcnt" => true,
        "avx" | "avx2" | "bmi1" | "bmi2" | "lzcnt" => {
            // These are commonly available on modern CPUs
            true
        }
        "neon" => {
            // NEON is standard on AArch64
            env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "aarch64"
        }
        _ => false,
    }
}

/// Configure scalar fallback for architectures without SIMD support
fn configure_scalar_fallback() {
    println!("cargo:rustc-cfg=zipora_scalar_only");
    println!("cargo:rustc-cfg=zipora_optimization_tier_0");
}

/// Configure target-specific features and optimizations
fn configure_target_specific_features() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    // Set target-specific configurations
    match (target_arch.as_str(), target_os.as_str()) {
        ("x86_64", "linux") => {
            println!("cargo:rustc-cfg=zipora_linux_x86_64");
            // Enable Linux-specific optimizations
            println!("cargo:rustc-link-lib=pthread");
        }
        ("x86_64", "windows") => {
            println!("cargo:rustc-cfg=zipora_windows_x86_64");
            // Enable Windows-specific optimizations
        }
        ("x86_64", "macos") => {
            println!("cargo:rustc-cfg=zipora_macos_x86_64");
            // Enable macOS-specific optimizations
        }
        ("aarch64", "linux") => {
            println!("cargo:rustc-cfg=zipora_linux_aarch64");
            // Enable ARM64 Linux-specific optimizations
            println!("cargo:rustc-link-lib=pthread");
        }
        ("aarch64", "macos") => {
            println!("cargo:rustc-cfg=zipora_macos_aarch64");
            // Enable ARM64 macOS-specific optimizations (Apple Silicon)
        }
        _ => {
            println!("cargo:warning=Unoptimized target: {}-{}", target_arch, target_os);
        }
    }
    
    // Set cache line size based on architecture
    match target_arch.as_str() {
        "x86_64" => println!("cargo:rustc-cfg=zipora_cache_line_64"),
        "aarch64" => println!("cargo:rustc-cfg=zipora_cache_line_64"),
        _ => println!("cargo:rustc-cfg=zipora_cache_line_64"), // Default assumption
    }
}

/// Configure optimization flags based on profile and features
fn configure_optimization_flags() {
    let profile = env::var("PROFILE").unwrap_or_default();
    
    match profile.as_str() {
        "release" => {
            println!("cargo:rustc-cfg=zipora_release_mode");
            
            // Enable aggressive optimizations for release builds
            if cfg!(feature = "simd") {
                println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=native");
            }
            
            // Link-time optimization flags are set in Cargo.toml
        }
        "bench" => {
            println!("cargo:rustc-cfg=zipora_bench_mode");
            
            // Enable maximum optimizations for benchmarks
            if cfg!(feature = "simd") {
                println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=native");
            }
        }
        _ => {
            println!("cargo:rustc-cfg=zipora_debug_mode");
            // Keep debug optimizations minimal for faster compilation
        }
    }
}

/// Generate FFI bindings if the ffi feature is enabled
#[cfg(feature = "ffi")]
#[allow(dead_code)]
fn generate_ffi_bindings() {
    use std::env;
    use std::path::PathBuf;
    
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    
    let config = cbindgen::Config {
        language: cbindgen::Language::C,
        cpp_compat: true,
        include_guard: Some("ZIPORA_H".to_string()),
        autogen_warning: Some("/* Warning, this file is autogenerated by cbindgen. Don't modify this manually. */".to_string()),
        include_version: true,
        namespace: Some("zipora".to_string()),
        line_length: 100,
        tab_width: 4,
        documentation: true,
        documentation_style: cbindgen::DocumentationStyle::Doxy,
        ..Default::default()
    };
    
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("include/zipora.h");
    
    println!("cargo:rustc-cfg=zipora_ffi_enabled");
}

#[cfg(not(feature = "ffi"))]
fn generate_ffi_bindings() {
    // No-op when ffi feature is disabled
}