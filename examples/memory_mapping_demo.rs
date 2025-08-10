//! Memory-Mapped I/O Demonstration
//!
//! This example demonstrates the memory-mapped I/O capabilities of zipora,
//! including zero-copy file operations, automatic file growth, and performance benefits.

use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;
#[cfg(feature = "mmap")]
use zipora::{DataInput, DataOutput, MemoryMappedInput, MemoryMappedOutput};

#[cfg(feature = "mmap")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗺️  Memory-Mapped I/O Demo for zipora");
    println!("==========================================\n");

    // Create a temporary file for our demonstration
    let temp_file = NamedTempFile::new()?;
    let file_path = temp_file.path();

    println!("📁 Created temporary file: {:?}", file_path);
    println!();

    // === PART 1: Memory-Mapped Output (Writing) ===
    println!("📤 PART 1: Memory-Mapped Output (Writing)");
    println!("------------------------------------------");

    {
        // Create memory-mapped output with initial capacity
        let mut output = MemoryMappedOutput::create(file_path, 512)?;
        println!("✅ Created MemoryMappedOutput with 512 bytes initial capacity");

        // Write various data types
        output.write_u32(0x12345678)?;
        println!("   Wrote u32: 0x12345678");

        output.write_u64(0x9ABCDEF012345678)?;
        println!("   Wrote u64: 0x9ABCDEF012345678");

        output.write_var_int(300)?;
        println!("   Wrote var_int: 300");

        output.write_length_prefixed_string("Hello, Memory Mapping!")?;
        println!("   Wrote string: \"Hello, Memory Mapping!\"");

        // Write a large array to test automatic growth
        let large_data = vec![0xAB; 1000];
        output.write_slice(&large_data)?;
        println!("   Wrote 1000 bytes of data (testing automatic growth)");

        println!("   Current position: {} bytes", output.position());
        println!("   Current capacity: {} bytes", output.capacity());
        println!("   Remaining space: {} bytes", output.remaining());

        // Flush to ensure data is written
        output.flush()?;
        println!("✅ Flushed data to disk");

        // Truncate to actual size for efficient storage
        output.truncate()?;
        println!(
            "✅ Truncated file to actual data size: {} bytes",
            output.capacity()
        );
    } // output is dropped here, ensuring proper cleanup

    println!();

    // === PART 2: Memory-Mapped Input (Reading) ===
    println!("📥 PART 2: Memory-Mapped Input (Reading)");
    println!("----------------------------------------");

    {
        // Open the file for memory-mapped reading
        let file = File::open(file_path)?;
        let mut input = MemoryMappedInput::new(file)?;
        println!("✅ Created MemoryMappedInput");

        println!("   File size: {} bytes", input.len());
        println!("   Initial position: {}", input.position());
        println!("   Available bytes: {}", input.remaining());

        // Read back the data we wrote
        let value1 = input.read_u32()?;
        println!(
            "   Read u32: 0x{:08X} ({})",
            value1,
            if value1 == 0x12345678 {
                "✅ correct"
            } else {
                "❌ incorrect"
            }
        );

        let value2 = input.read_u64()?;
        println!(
            "   Read u64: 0x{:016X} ({})",
            value2,
            if value2 == 0x9ABCDEF012345678 {
                "✅ correct"
            } else {
                "❌ incorrect"
            }
        );

        let var_int = input.read_var_int()?;
        println!(
            "   Read var_int: {} ({})",
            var_int,
            if var_int == 300 {
                "✅ correct"
            } else {
                "❌ incorrect"
            }
        );

        let text = input.read_length_prefixed_string()?;
        println!(
            "   Read string: \"{}\" ({})",
            text,
            if text == "Hello, Memory Mapping!" {
                "✅ correct"
            } else {
                "❌ incorrect"
            }
        );

        // Zero-copy slice reading
        let slice = input.read_slice(10)?;
        println!("   Read 10 bytes (zero-copy): {:02X?}", slice);

        println!("   Position after reads: {}", input.position());
        println!("   Remaining bytes: {}", input.remaining());
    } // input is dropped here

    println!();

    // === PART 3: Advanced Operations ===
    println!("🔧 PART 3: Advanced Operations");
    println!("------------------------------");

    {
        let file = File::open(file_path)?;
        let mut input = MemoryMappedInput::new(file)?;

        // Demonstrate seeking
        input.seek(4)?; // Skip the first u32
        println!("✅ Seeked to position 4");

        let value = input.read_u64()?;
        println!("   Read u64 after seek: 0x{:016X}", value);

        // Demonstrate peek (reading without advancing position)
        input.seek(0)?;
        let peeked = input.peek_slice(4)?;
        println!("   Peeked first 4 bytes: {:02X?}", peeked);
        println!(
            "   Position after peek: {} (should still be 0)",
            input.position()
        );

        // Skip some data
        input.skip(12)?; // Skip u32 + u64
        println!("   Skipped 12 bytes, new position: {}", input.position());

        let var_int = input.read_var_int()?;
        println!("   Read var_int after skip: {}", var_int);
    }

    println!();

    // === PART 4: Performance Comparison ===
    println!("🚀 PART 4: Performance Benefits");
    println!("-------------------------------");

    println!("Memory-mapped I/O advantages:");
    println!("• Zero-copy operations - no intermediate buffers");
    println!("• Operating system handles caching and paging");
    println!("• Efficient random access patterns");
    println!("• Automatic file growth for writes");
    println!("• Cross-platform compatibility");
    println!("• Memory safety through Rust's type system");

    println!();
    println!("🎯 Use cases:");
    println!("• Large file processing");
    println!("• Database storage engines");
    println!("• Index file management");
    println!("• Log file processing");
    println!("• Scientific data analysis");

    println!();
    println!("✅ Memory-mapping demonstration completed successfully!");

    Ok(())
}

#[cfg(not(feature = "mmap"))]
fn main() {
    println!("⚠️  Memory mapping feature is not enabled!");
    println!("To run this example, enable the 'mmap' feature:");
    println!("cargo run --example memory_mapping_demo --features mmap");
}
