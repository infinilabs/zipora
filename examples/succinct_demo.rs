//! Demonstration of succinct data structures in infini-zip
//!
//! This example shows how to use BitVector and RankSelect256 for efficient
//! bit operations with constant-time rank and select queries.

use infini_zip::{BitVector, RankSelect256, Result};

fn main() -> Result<()> {
    println!("=== Infini-Zip Succinct Data Structures Demo ===\n");

    // Create a bit vector with a known pattern
    let mut bit_vector = BitVector::new();

    println!("Creating bit vector with pattern (every 3rd bit set):");
    for i in 0..32 {
        let bit = i % 3 == 0;
        bit_vector.push(bit)?;
        print!("{}", if bit { '1' } else { '0' });
        if (i + 1) % 8 == 0 {
            print!(" ");
        }
    }
    println!("\n");

    // Display basic statistics
    println!("Bit vector statistics:");
    println!("  Length: {} bits", bit_vector.len());
    println!("  Set bits (1s): {}", bit_vector.count_ones());
    println!("  Clear bits (0s): {}", bit_vector.count_zeros());
    println!();

    // Demonstrate basic bit operations
    println!("Basic bit operations:");
    println!("  bit_vector[0] = {:?}", bit_vector.get(0));
    println!("  bit_vector[3] = {:?}", bit_vector.get(3));
    println!("  bit_vector[31] = {:?}", bit_vector.get(31));
    println!();

    // Demonstrate rank operations (counting bits up to position)
    println!("Rank operations (count of 1s up to position):");
    for pos in [0, 5, 10, 15, 20, 25, 30, 32] {
        println!("  rank1({}) = {}", pos, bit_vector.rank1(pos));
    }
    println!();

    // Create RankSelect256 for advanced operations
    println!("Building RankSelect256 index...");
    let rank_select = RankSelect256::new(bit_vector.clone())?;

    println!("RankSelect256 statistics:");
    println!(
        "  Space overhead: {:.2}%",
        rank_select.space_overhead_percent()
    );
    println!("  Total set bits: {}", rank_select.count_ones());
    println!();

    // Demonstrate select operations (finding position of nth set bit)
    println!("Select operations (position of nth set bit):");
    for n in 0..rank_select.count_ones().min(8) {
        match rank_select.select1(n) {
            Ok(pos) => println!(
                "  select1({}) = {} ({}th set bit at position {})",
                n,
                pos,
                n + 1,
                pos
            ),
            Err(e) => println!("  select1({}) = Error: {}", n, e),
        }
    }
    println!();

    // Demonstrate consistency between implementations
    println!("Verifying rank consistency:");
    let mut mismatches = 0;
    for pos in 0..=bit_vector.len() {
        let bv_rank = bit_vector.rank1(pos);
        let rs_rank = rank_select.rank1(pos);
        if bv_rank != rs_rank {
            println!(
                "  Mismatch at pos {}: BitVector={}, RankSelect={}",
                pos, bv_rank, rs_rank
            );
            mismatches += 1;
        }
    }
    if mismatches == 0 {
        println!("  ✓ All rank operations match between implementations");
    }
    println!();

    // Performance demonstration with larger dataset
    println!("Performance test with larger dataset:");
    let mut large_bv = BitVector::with_capacity(100_000)?;
    for i in 0..100_000 {
        large_bv.push(i % 7 == 0)?; // Every 7th bit set
    }

    println!("  Created bit vector with {} bits", large_bv.len());
    println!("  Set bits: {}", large_bv.count_ones());

    let large_rs = RankSelect256::new(large_bv)?;
    println!(
        "  RankSelect256 space overhead: {:.2}%",
        large_rs.space_overhead_percent()
    );

    // Time some operations
    let start = std::time::Instant::now();
    let mut sum = 0;
    for i in (0..large_rs.len()).step_by(1000) {
        sum += large_rs.rank1(i);
    }
    let elapsed = start.elapsed();
    println!(
        "  100 rank operations took: {:?} (checksum: {})",
        elapsed, sum
    );

    // Test select operations
    let set_bits = large_rs.count_ones();
    if set_bits > 0 {
        let start = std::time::Instant::now();
        let mut positions = Vec::new();
        for i in (0..set_bits).step_by(set_bits / 10) {
            if let Ok(pos) = large_rs.select1(i) {
                positions.push(pos);
            }
        }
        let elapsed = start.elapsed();
        println!(
            "  10 select operations took: {:?} (found {} positions)",
            elapsed,
            positions.len()
        );
    }

    println!("\n=== Demo completed successfully! ===");
    Ok(())
}
