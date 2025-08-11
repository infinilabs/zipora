//! Integration tests for I/O & Serialization stream implementations
//!
//! This module tests the new StreamBuffer, RangeStream, and Zero-Copy optimizations
//! to ensure they work correctly together and provide the expected performance benefits.

use std::io::{Cursor, Read, Write, Seek, SeekFrom};
use tempfile::NamedTempFile;
use zipora::io::*;
use zipora::error::Result;
use std::io::Read as _;

/// Test data generator for consistent test inputs
fn generate_test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

/// Test StreamBufferedReader with various configurations
#[test]
fn test_stream_buffered_reader_configurations() {
    let test_data = generate_test_data(1024 * 1024); // 1MB test data
    
    // Test default configuration
    let cursor = Cursor::new(&test_data);
    let mut reader = StreamBufferedReader::new(cursor).unwrap();
    let mut read_data = Vec::new();
    reader.read_to_end(&mut read_data).unwrap();
    assert_eq!(read_data, test_data);
    
    // Test performance optimized configuration
    let cursor = Cursor::new(&test_data);
    let mut reader = StreamBufferedReader::performance_optimized(cursor).unwrap();
    assert!(reader.capacity() >= 128 * 1024);
    let mut read_data = Vec::new();
    reader.read_to_end(&mut read_data).unwrap();
    assert_eq!(read_data, test_data);
    
    // Test memory efficient configuration
    let cursor = Cursor::new(&test_data);
    let mut reader = StreamBufferedReader::memory_efficient(cursor).unwrap();
    assert_eq!(reader.capacity(), 16 * 1024);
    let mut read_data = Vec::new();
    reader.read_to_end(&mut read_data).unwrap();
    assert_eq!(read_data, test_data);
    
    // Test low latency configuration
    let cursor = Cursor::new(&test_data);
    let mut reader = StreamBufferedReader::low_latency(cursor).unwrap();
    assert_eq!(reader.capacity(), 8 * 1024);
    let mut read_data = Vec::new();
    reader.read_to_end(&mut read_data).unwrap();
    assert_eq!(read_data, test_data);
}

/// Test StreamBufferedReader fast byte reading
#[test]
fn test_stream_buffered_reader_fast_operations() {
    let test_data = b"Hello, World! This is a test of fast byte reading.";
    let cursor = Cursor::new(test_data);
    let mut reader = StreamBufferedReader::new(cursor).unwrap();
    
    // Test fast byte reading
    assert_eq!(reader.read_byte_fast().unwrap(), b'H');
    assert_eq!(reader.read_byte_fast().unwrap(), b'e');
    assert_eq!(reader.read_byte_fast().unwrap(), b'l');
    
    // Test slice reading
    if let Some(slice) = reader.read_slice(2).unwrap() {
        assert_eq!(slice, b"lo");
    } else {
        panic!("Expected slice data");
    }
    
    // Test bulk reading
    let mut buf = vec![0u8; 20];
    let bytes_read = reader.read_bulk(&mut buf).unwrap();
    assert_eq!(bytes_read, 20);
    assert_eq!(&buf, b", World! This is a t");
    
}

/// Test StreamBufferedWriter operations
#[test]
fn test_stream_buffered_writer_operations() {
    let mut buffer = Vec::new();
    {
        let cursor = Cursor::new(&mut buffer);
        let mut writer = StreamBufferedWriter::new(cursor).unwrap();
        
        // Test fast byte writing
        writer.write_byte_fast(b'H').unwrap();
        writer.write_byte_fast(b'e').unwrap();
        writer.write_byte_fast(b'l').unwrap();
        writer.write_byte_fast(b'l').unwrap();
        writer.write_byte_fast(b'o').unwrap();
        
        // Test normal writing
        writer.write_all(b", World!").unwrap();
        writer.flush().unwrap();
    }
    
    assert_eq!(buffer, b"Hello, World!");
}

/// Test RangeReader functionality
#[test]
fn test_range_reader_operations() {
    let test_data = b"The quick brown fox jumps over the lazy dog.";
    let cursor = Cursor::new(test_data);
    let mut reader = RangeReader::new_and_seek(cursor, 10, 9).unwrap(); // "brown fox"
    
    // Test basic range reading
    let mut buf = String::new();
    reader.read_to_string(&mut buf).unwrap();
    assert_eq!(buf, "brown fox");
    
    // Test position tracking
    assert_eq!(reader.start_position(), 10);
    assert_eq!(reader.end_position(), 19);
    assert_eq!(reader.range_length(), 9);
    assert!(reader.is_at_end());
    assert_eq!(reader.progress(), 1.0);
    
}

/// Test RangeReader with DataInput trait
#[test]
fn test_range_reader_data_input() {
    // Create test data with specific byte patterns
    let mut test_data = Vec::new();
    test_data.extend_from_slice(&42u8.to_le_bytes());      // u8
    test_data.extend_from_slice(&0x1234u16.to_le_bytes()); // u16
    test_data.extend_from_slice(&0x12345678u32.to_le_bytes()); // u32
    test_data.extend_from_slice(&0x123456789ABCDEF0u64.to_le_bytes()); // u64
    
    let cursor = Cursor::new(&test_data);
    let mut reader = RangeReader::new(cursor, 0, test_data.len() as u64);
    
    // Test reading structured data
    assert_eq!(reader.read_u8().unwrap(), 42);
    assert_eq!(reader.read_u16().unwrap(), 0x1234);
    assert_eq!(reader.read_u32().unwrap(), 0x12345678);
    assert_eq!(reader.read_u64().unwrap(), 0x123456789ABCDEF0);
    
}

/// Test RangeWriter functionality
#[test]
fn test_range_writer_operations() {
    let mut buffer = vec![0u8; 50];
    {
        let cursor = Cursor::new(&mut buffer);
        let mut writer = RangeWriter::new_and_seek(cursor, 10, 13).unwrap(); // Write "Hello, World!" at position 10
        
        writer.write_all(b"Hello, World!").unwrap();
        writer.flush().unwrap();
        
        assert_eq!(writer.bytes_written(), 13);
        assert!(writer.is_at_end());
    }
    
    // Check that data was written in the correct range
    assert_eq!(&buffer[10..23], b"Hello, World!");
    assert_eq!(&buffer[..10], &[0u8; 10]); // Unchanged
    assert_eq!(&buffer[23..], &[0u8; 27]); // Unchanged
    
}

/// Test MultiRangeReader for discontinuous ranges
#[test]
fn test_multi_range_reader() {
    let test_data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let cursor = Cursor::new(test_data);
    
    // Read ranges: A-C (0-3), G-I (6-9), M-O (12-15), 1-3 (27-30)
    let ranges = vec![(0, 3), (6, 9), (12, 15), (27, 30)];
    let mut reader = MultiRangeReader::new(cursor, ranges);
    
    assert_eq!(reader.total_length(), 12); // 3+3+3+3 = 12 bytes total
    
    let mut result = String::new();
    reader.read_to_string(&mut result).unwrap();
    assert_eq!(result, "ABCGHIMNO123");
    
}

/// Test ZeroCopyBuffer basic operations
#[test]
fn test_zero_copy_buffer_operations() {
    let mut buffer = ZeroCopyBuffer::new(1024).unwrap();
    
    // Test initial state
    assert_eq!(buffer.capacity(), 1024);
    assert_eq!(buffer.available(), 0);
    assert_eq!(buffer.write_available(), 1024);
    assert!(buffer.is_empty());
    assert!(!buffer.is_full());
    
    // Test zero-copy writing
    if let Some(write_buf) = buffer.zc_write(13).unwrap() {
        write_buf.copy_from_slice(b"Hello, World!");
        buffer.zc_commit(13).unwrap();
    }
    
    assert_eq!(buffer.available(), 13);
    assert_eq!(buffer.write_available(), 1011);
    
    // Test zero-copy reading
    if let Some(read_buf) = buffer.zc_read(5).unwrap() {
        assert_eq!(read_buf, b"Hello");
        buffer.zc_advance(5).unwrap();
    }
    
    assert_eq!(buffer.available(), 8);
    
    // Test remaining data
    if let Some(read_buf) = buffer.zc_read(8).unwrap() {
        assert_eq!(read_buf, b", World!");
        buffer.zc_advance(8).unwrap();
    }
    
    assert!(buffer.is_empty());
    
}

/// Test ZeroCopyReader with large data
#[test]
fn test_zero_copy_reader_large_data() {
    let test_data = generate_test_data(1024 * 1024); // 1MB
    let cursor = Cursor::new(&test_data);
    let mut reader = ZeroCopyReader::new(cursor).unwrap();
    
    // Test peek operation
    let peeked = reader.peek(1024).unwrap();
    assert_eq!(peeked.len(), 1024);
    assert_eq!(peeked, &test_data[..1024]);
    
    // Test zero-copy reading of first chunk
    if let Some(zc_data) = reader.zc_read(1024).unwrap() {
        assert_eq!(zc_data, &test_data[..1024]);
        reader.zc_advance(1024).unwrap();
    }
    
    // Test skipping data
    reader.skip_bytes(1024).unwrap();
    
    // Test reading remaining data
    let mut remaining_data = Vec::new();
    reader.read_to_end(&mut remaining_data).unwrap();
    assert_eq!(remaining_data, &test_data[2048..]);
    
}

/// Test ZeroCopyWriter with flush operations
#[test]
fn test_zero_copy_writer_flush() {
    let mut buffer = Vec::new();
    {
        let cursor = Cursor::new(&mut buffer);
        let mut writer = ZeroCopyWriter::new(cursor).unwrap();
        
        // Write some data using zero-copy
        if let Some(zc_buf) = writer.zc_write(13).unwrap() {
            zc_buf.copy_from_slice(b"Hello, World!");
            writer.zc_commit(13).unwrap();
        }
        
        // Write more data using regular write
        writer.write_all(b" How are you?").unwrap();
        
        // Test buffer usage before flush
        assert!(writer.get_ref().get_ref().is_empty()); // Data still in internal buffer
        
        writer.flush().unwrap();
    }
    
    assert_eq!(buffer, b"Hello, World! How are you?");
}

/// Test VectoredIO operations
#[test]
fn test_vectored_io_operations() {
    use std::io::{IoSlice, IoSliceMut};
    
    // Test vectored reading
    let test_data = b"Hello, World! This is a test.";
    let mut cursor = Cursor::new(test_data);
    
    let mut buf1 = [0u8; 5];  // "Hello"
    let mut buf2 = [0u8; 2];  // ", "
    let mut buf3 = [0u8; 6];  // "World!"
    let mut buf4 = [0u8; 16]; // " This is a test"
    
    let mut bufs = [
        IoSliceMut::new(&mut buf1),
        IoSliceMut::new(&mut buf2),
        IoSliceMut::new(&mut buf3),
        IoSliceMut::new(&mut buf4),
    ];
    
    let bytes_read = VectoredIO::read_vectored(&mut cursor, &mut bufs).unwrap();
    assert_eq!(bytes_read, 29); // 5 + 2 + 6 + 16 = 29
    assert_eq!(&buf1, b"Hello");
    assert_eq!(&buf2, b", ");
    assert_eq!(&buf3, b"World!");
    assert_eq!(&buf4[..15], b" This is a test"); // First 15 bytes
    
    // Test vectored writing
    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);
    
    let bufs = [
        IoSlice::new(b"Hello"),
        IoSlice::new(b", "),
        IoSlice::new(b"World!"),
        IoSlice::new(b" Vectored I/O test."),
    ];
    
    let bytes_written = VectoredIO::write_vectored(&mut cursor, &bufs).unwrap();
    assert_eq!(bytes_written, 32);
    assert_eq!(buffer, b"Hello, World! Vectored I/O test.");
    
}

/// Test memory-mapped zero-copy operations
#[cfg(feature = "mmap")]
#[test]
fn test_mmap_zero_copy_operations() {
    use std::io::Write;
    
    let mut temp_file = NamedTempFile::new().unwrap();
    let test_data = generate_test_data(64 * 1024); // 64KB test data
    temp_file.write_all(&test_data).unwrap();
    temp_file.flush().unwrap();
    
    let file = temp_file.reopen().unwrap();
    let mut reader = zero_copy::mmap::MmapZeroCopyReader::new(file).unwrap();
    
    assert_eq!(reader.len(), test_data.len());
    assert_eq!(reader.as_slice(), &test_data);
    
    // Test zero-copy reading from different positions
    for chunk_size in [1024, 4096, 16384] {
        reader.set_position(0).unwrap();
        
        let mut pos = 0;
        while pos < test_data.len() {
            let remaining = test_data.len() - pos;
            let to_read = chunk_size.min(remaining);
            
            if let Some(zc_data) = reader.zc_read(to_read).unwrap() {
                assert_eq!(zc_data, &test_data[pos..pos + to_read]);
                reader.zc_advance(to_read).unwrap();
                pos += to_read;
            } else {
                break;
            }
        }
        
        assert_eq!(pos, test_data.len());
    }
    
}

/// Integration test combining all three implementations
#[test]
fn test_combined_stream_operations() {
    let test_data = generate_test_data(256 * 1024); // 256KB test data
    
    // Phase 1: Write data using StreamBufferedWriter
    let mut buffer = Vec::new();
    {
        let cursor = Cursor::new(&mut buffer);
        let config = StreamBufferConfig::performance_optimized();
        let mut writer = StreamBufferedWriter::with_config(cursor, config).unwrap();
        writer.write_all(&test_data).unwrap();
        writer.flush().unwrap();
    }
    
    // Phase 2: Read specific ranges using RangeReader with ZeroCopyReader
    let cursor = Cursor::new(&buffer);
    let range_reader = RangeReader::new_and_seek(cursor, 1024, 64 * 1024).unwrap(); // Read 64KB starting at 1KB
    let mut zc_reader = ZeroCopyReader::new(range_reader).unwrap();
    
    // Phase 3: Use zero-copy operations to process the range
    let mut processed_data = Vec::new();
    while processed_data.len() < 64 * 1024 {
        let remaining = 64 * 1024 - processed_data.len();
        let chunk_size = 4096.min(remaining);
        
        if let Some(zc_data) = zc_reader.zc_read(chunk_size).unwrap() {
            processed_data.extend_from_slice(zc_data);
            zc_reader.zc_advance(chunk_size).unwrap();
        } else {
            // Fall back to regular read
            let mut buf = vec![0u8; chunk_size];
            let bytes_read = zc_reader.read(&mut buf).unwrap();
            if bytes_read == 0 {
                break;
            }
            processed_data.extend_from_slice(&buf[..bytes_read]);
        }
    }
    
    // Verify the processed data matches the expected range
    assert_eq!(processed_data, &test_data[1024..1024 + 64 * 1024]);
    
}

/// Performance comparison test
#[test]
fn test_stream_performance_comparison() {
    let test_data = generate_test_data(1024 * 1024); // 1MB test data
    
    // Test 1: Standard BufReader vs StreamBufferedReader
    let cursor1 = Cursor::new(&test_data);
    let mut std_reader = std::io::BufReader::new(cursor1);
    let start = std::time::Instant::now();
    let mut std_data = Vec::new();
    std_reader.read_to_end(&mut std_data).unwrap();
    let std_duration = start.elapsed();
    
    let cursor2 = Cursor::new(&test_data);
    let mut stream_reader = StreamBufferedReader::performance_optimized(cursor2).unwrap();
    let start = std::time::Instant::now();
    let mut stream_data = Vec::new();
    stream_reader.read_to_end(&mut stream_data).unwrap();
    let stream_duration = start.elapsed();
    
    assert_eq!(std_data, stream_data);
    println!("Standard BufReader: {:?}, StreamBufferedReader: {:?}", std_duration, stream_duration);
    
    // Test 2: Regular read vs Zero-copy read (when possible)
    let cursor3 = Cursor::new(&test_data);
    let mut zc_reader = ZeroCopyReader::new(cursor3).unwrap();
    let start = std::time::Instant::now();
    let mut zc_data = Vec::new();
    zc_reader.read_to_end(&mut zc_data).unwrap();
    let zc_duration = start.elapsed();
    
    assert_eq!(std_data, zc_data);
    println!("ZeroCopyReader: {:?}", zc_duration);
    
}

/// Stress test with large data and multiple operations
#[test]
fn test_stress_operations() {
    const DATA_SIZE: usize = 10 * 1024 * 1024; // 10MB
    let test_data = generate_test_data(DATA_SIZE);
    
    // Test with multiple concurrent-style operations
    for i in 0..10 {
        let offset = i * 1024 * 1024;
        let length = 1024 * 1024; // 1MB chunks
        
        if offset + length > DATA_SIZE {
            break;
        }
        
        // Use RangeReader for each chunk
        let cursor = Cursor::new(&test_data);
        let range_reader = RangeReader::new_and_seek(cursor, offset as u64, length as u64).unwrap();
        let mut buffered_reader = StreamBufferedReader::new(range_reader).unwrap();
        
        let mut chunk_data = Vec::new();
        buffered_reader.read_to_end(&mut chunk_data).unwrap();
        
        assert_eq!(chunk_data, &test_data[offset..offset + length]);
    }
    
}