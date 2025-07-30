//! Debug program for critical-bit trie implementation
//! 
//! This module provides comprehensive debugging utilities to trace
//! the construction and lookup process in critical-bit tries.

use crate::fsa::crit_bit_trie::CritBitTrie;
use crate::fsa::traits::Trie;

/// Debug structure to trace trie operations
pub struct CritBitDebugger {
    trie: CritBitTrie,
    operation_log: Vec<String>,
}

impl CritBitDebugger {
    pub fn new() -> Self {
        Self {
            trie: CritBitTrie::new(),
            operation_log: Vec::new(),
        }
    }

    /// Insert a key with detailed debugging
    pub fn debug_insert(&mut self, key: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let key_str = String::from_utf8_lossy(key);
        println!("\n=== INSERTING: '{}' ===", key_str);
        
        self.operation_log.push(format!("INSERT: '{}'", key_str));
        
        // Show state before insertion
        self.print_trie_structure();
        
        // Perform insertion
        let result = self.trie.insert(key);
        
        match result {
            Ok(_) => {
                println!("✓ Insertion successful");
                self.operation_log.push(format!("SUCCESS: '{}'", key_str));
            }
            Err(e) => {
                println!("✗ Insertion failed: {}", e);
                self.operation_log.push(format!("FAILED: '{}' - {}", key_str, e));
                return Err(Box::new(e));
            }
        }
        
        // Show state after insertion
        println!("\nTrie after insertion:");
        self.print_trie_structure();
        
        Ok(())
    }

    /// Lookup a key with detailed debugging
    pub fn debug_lookup(&self, key: &[u8]) -> bool {
        let key_str = String::from_utf8_lossy(key);
        println!("\n=== LOOKING UP: '{}' ===", key_str);
        
        // Manual traversal with debugging
        let found = self.trace_lookup_path(key);
        
        // Compare with trie's lookup method
        let trie_result = self.trie.contains(key);
        
        println!("Manual trace result: {}", found);
        println!("Trie lookup result: {}", trie_result);
        
        if found != trie_result {
            println!("⚠️  MISMATCH between manual trace and trie lookup!");
        }
        
        trie_result
    }

    /// Trace the lookup path manually with debugging
    fn trace_lookup_path(&self, key: &[u8]) -> bool {
        let key_str = String::from_utf8_lossy(key);
        println!("Tracing lookup path for '{}'", key_str);
        
        // Access the internal structure - we'll need to add debug methods to CritBitTrie
        // For now, let's use the public interface and add detailed logging
        
        if self.trie.len() == 0 {
            println!("  Trie is empty");
            return false;
        }
        
        // We need access to internal nodes to trace properly
        // Let's check what the public interface gives us
        let result = self.trie.contains(key);
        println!("  Final result: {}", result);
        
        result
    }

    /// Print the entire trie structure
    pub fn print_trie_structure(&self) {
        println!("\n--- TRIE STRUCTURE ---");
        println!("Number of keys: {}", self.trie.len());
        
        if self.trie.len() == 0 {
            println!("(empty trie)");
            return;
        }
        
        // For detailed structure printing, we need access to internal nodes
        // Let's use the available public methods
        println!("Trie contains {} keys", self.trie.len());
        
        // Test some common keys to understand structure
        let test_keys: &[&[u8]] = &[b"", b"c", b"ca", b"car", b"cars", b"cat", b"cats", b"card"];
        for key in test_keys {
            let contains = self.trie.contains(key);
            if contains {
                println!("  ✓ Contains: '{}'", String::from_utf8_lossy(key));
            }
        }
        println!("--- END STRUCTURE ---\n");
    }

    /// Print operation log
    pub fn print_operation_log(&self) {
        println!("\n=== OPERATION LOG ===");
        for (i, op) in self.operation_log.iter().enumerate() {
            println!("{:2}: {}", i + 1, op);
        }
        println!("=== END LOG ===\n");
    }

    /// Test the critical bit calculation
    pub fn test_critical_bit_calculation(&self) {
        println!("\n=== CRITICAL BIT TESTS ===");
        
        let test_pairs: &[(&[u8], &[u8])] = &[
            (b"cat", b"car"),
            (b"car", b"card"),
            (b"cat", b"card"),
            (b"a", b"ab"),
            (b"hello", b"help"),
        ];
        
        for (key1, key2) in test_pairs {
            // We need to expose the find_critical_bit method or implement it here
            let key1_str = String::from_utf8_lossy(key1);
            let key2_str = String::from_utf8_lossy(key2);
            
            println!("Comparing '{}' vs '{}':", key1_str, key2_str);
            
            // Manual critical bit calculation for debugging
            let (byte_pos, bit_pos) = self.find_critical_bit_debug(key1, key2);
            println!("  Critical byte: {}, bit: {}", byte_pos, bit_pos);
            
            // Show the bit values at that position
            let bit1 = self.test_bit_debug(key1, byte_pos, bit_pos);
            let bit2 = self.test_bit_debug(key2, byte_pos, bit_pos);
            println!("  '{}' bit: {}, '{}' bit: {}", key1_str, bit1, key2_str, bit2);
        }
        
        println!("=== END CRITICAL BIT TESTS ===\n");
    }

    /// Debug version of critical bit finding
    fn find_critical_bit_debug(&self, key1: &[u8], key2: &[u8]) -> (usize, u8) {
        let mut byte_pos = 0;
        let min_len = key1.len().min(key2.len());
        
        println!("    Finding critical bit between keys of length {} and {}", key1.len(), key2.len());
        
        // Find first differing byte
        while byte_pos < min_len && key1[byte_pos] == key2[byte_pos] {
            println!("    Byte {}: both have 0x{:02x}", byte_pos, key1[byte_pos]);
            byte_pos += 1;
        }
        
        // If one key is a prefix of the other
        if byte_pos == min_len {
            if key1.len() != key2.len() {
                println!("    One key is prefix of other, using virtual end-of-string bit");
                return (byte_pos, 8);
            } else {
                println!("    Keys are identical");
                return (byte_pos, 0);
            }
        }
        
        // Find the critical bit within the differing byte
        let byte1 = key1[byte_pos];
        let byte2 = key2[byte_pos];
        let diff = byte1 ^ byte2;
        
        println!("    Byte {}: 0x{:02x} vs 0x{:02x}, diff: 0x{:02x}", byte_pos, byte1, byte2, diff);
        
        // Find the most significant differing bit
        let bit_pos = 7 - diff.leading_zeros() as u8;
        
        println!("    Most significant differing bit: {}", bit_pos);
        
        (byte_pos, bit_pos)
    }

    /// Debug version of bit testing
    fn test_bit_debug(&self, key: &[u8], byte_pos: usize, bit_pos: u8) -> bool {
        if bit_pos == 8 {
            let result = byte_pos >= key.len();
            println!("    Bit test: virtual end-of-string bit at byte {}, key length {}, result = {}", 
                     byte_pos, key.len(), result);
            result
        } else if byte_pos >= key.len() {
            println!("    Bit test: byte {} beyond key length {}, treating as 0", byte_pos, key.len());
            false
        } else {
            let byte_val = key[byte_pos];
            let bit_val = (byte_val >> bit_pos) & 1 == 1;
            println!("    Bit test: byte {} (0x{:02x}) >> {} & 1 = {}", byte_pos, byte_val, bit_pos, bit_val);
            bit_val
        }
    }

    /// Run a comprehensive debug session
    pub fn run_debug_session() -> Result<(), Box<dyn std::error::Error>> {
        let mut debugger = CritBitDebugger::new();
        
        println!("🔍 CRITICAL-BIT TRIE DEBUG SESSION");
        println!("=================================");
        
        // Test critical bit calculations first
        debugger.test_critical_bit_calculation();
        
        // Insert keys step by step
        let keys: &[&[u8]] = &[b"cat", b"car", b"card"];
        
        for key in keys {
            debugger.debug_insert(key)?;
        }
        
        println!("\n🔍 TESTING LOOKUPS");
        println!("=================");
        
        // Test lookups
        let lookup_keys: &[&[u8]] = &[b"cat", b"car", b"card", b"ca", b"cars", b"care"];
        
        for key in lookup_keys {
            let found = debugger.debug_lookup(key);
            let key_str = String::from_utf8_lossy(key);
            println!("Lookup '{}': {}", key_str, if found { "FOUND" } else { "NOT FOUND" });
        }
        
        // Show final operation log
        debugger.print_operation_log();
        
        // Additional analysis
        println!("\n🔍 ANALYSIS");
        println!("==========");
        
        // Check if "car" lookup issue exists
        let car_found = debugger.debug_lookup(b"car");
        if !car_found {
            println!("🚨 CONFIRMED: 'car' lookup fails after inserting cat, car, card");
            println!("This suggests an issue with the trie structure or lookup algorithm");
        } else {
            println!("✅ 'car' lookup works correctly");
        }
        
        Ok(())
    }
}

/// Extended debug functions that analyze bit patterns
pub fn analyze_key_bits(key: &[u8]) {
    let key_str = String::from_utf8_lossy(key);
    println!("Bit analysis for '{}':", key_str);
    
    for (i, &byte) in key.iter().enumerate() {
        println!("  Byte {}: 0x{:02x} = {:08b}", i, byte, byte);
        
        // Show individual bits
        for bit in 0..8 {
            let bit_val = (byte >> (7 - bit)) & 1;
            print!("    Bit {}: {} ", 7 - bit, bit_val);
        }
        println!();
    }
}

/// Compare two keys bit by bit
pub fn compare_keys_bitwise(key1: &[u8], key2: &[u8]) {
    let key1_str = String::from_utf8_lossy(key1);
    let key2_str = String::from_utf8_lossy(key2);
    
    println!("Bitwise comparison: '{}' vs '{}'", key1_str, key2_str);
    
    let max_len = key1.len().max(key2.len());
    
    for i in 0..max_len {
        let byte1 = if i < key1.len() { key1[i] } else { 0 };
        let byte2 = if i < key2.len() { key2[i] } else { 0 };
        
        println!("  Byte {}: 0x{:02x} vs 0x{:02x}", i, byte1, byte2);
        
        if byte1 != byte2 {
            let diff = byte1 ^ byte2;
            println!("    DIFFER: XOR = 0x{:02x} = {:08b}", diff, diff);
            
            for bit in 0..8 {
                let bit1 = (byte1 >> (7 - bit)) & 1;
                let bit2 = (byte2 >> (7 - bit)) & 1;
                if bit1 != bit2 {
                    println!("      First differing bit: {} ({} vs {})", 7 - bit, bit1, bit2);
                    break;
                }
            }
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_session() {
        // This test will run our debug session and help identify the issue
        CritBitDebugger::run_debug_session().expect("Debug session failed");
    }

    #[test]
    fn test_bit_analysis() {
        println!("\nBit analysis for test keys:");
        analyze_key_bits(b"cat");
        analyze_key_bits(b"car");
        analyze_key_bits(b"card");
        
        println!("\nBitwise comparisons:");
        compare_keys_bitwise(b"cat", b"car");
        compare_keys_bitwise(b"car", b"card");
        compare_keys_bitwise(b"cat", b"card");
    }
}