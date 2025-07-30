//! Debug the zero byte issue in critical-bit trie

use infini_zip::debug_crit_bit::{CritBitDebugger, analyze_key_bits, compare_keys_bitwise};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Zero Byte Debug Session");
    println!("=========================\n");

    let mut debugger = CritBitDebugger::new();
    
    let key1 = b"test";
    let key2 = b"test\x00";
    
    println!("Testing keys:");
    analyze_key_bits(key1);
    analyze_key_bits(key2);
    
    println!("\nComparing keys:");
    compare_keys_bitwise(key1, key2);
    
    println!("\nInserting keys:");
    debugger.debug_insert(key1)?;
    debugger.debug_insert(key2)?;
    
    println!("\nLooking up keys:");
    debugger.debug_lookup(key1);
    debugger.debug_lookup(key2);
    
    Ok(())
}