//! Debug example for critical-bit trie implementation
//! 
//! This example demonstrates the issue with critical-bit trie lookup
//! and provides detailed debugging information.

use infini_zip::debug_crit_bit::{CritBitDebugger, analyze_key_bits, compare_keys_bitwise};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Critical-Bit Trie Debug Session");
    println!("=====================================\n");

    // Run the main debug session
    CritBitDebugger::run_debug_session()?;
    
    println!("\n🔬 DETAILED BIT ANALYSIS");
    println!("========================");
    
    // Analyze the problematic keys bit by bit
    let keys: &[&[u8]] = &[b"cat", b"car", b"card"];
    
    for key in keys {
        analyze_key_bits(key);
        println!();
    }
    
    println!("🔬 PAIRWISE COMPARISONS");
    println!("=======================");
    
    // Compare keys pairwise
    compare_keys_bitwise(b"cat", b"car");
    println!();
    compare_keys_bitwise(b"car", b"card");
    println!();
    compare_keys_bitwise(b"cat", b"card");
    
    println!("\n✅ Debug session completed");
    
    Ok(())
}