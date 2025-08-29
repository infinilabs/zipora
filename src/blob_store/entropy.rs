//! Entropy coding blob store implementations
//!
//! This module provides blob store wrappers that use entropy coding for compression.

use crate::blob_store::{BlobStore, BlobStoreStats};
use crate::entropy::{
    DictionaryBuilder, DictionaryCompressor, EntropyStats, HuffmanDecoder, HuffmanEncoder,
    HuffmanTree,
};
use crate::entropy::rans::{Rans64Encoder, ParallelX1};
use crate::error::{Result, ZiporaError};

/// Compression algorithm type for entropy blob store
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyAlgorithm {
    /// Huffman coding
    Huffman,
    /// rANS (range Asymmetric Numeral Systems)
    Rans,
    /// Dictionary-based compression
    Dictionary,
}

/// Statistics for entropy compression
#[derive(Debug, Clone, PartialEq)]
pub struct EntropyCompressionStats {
    /// Basic blob store statistics
    pub blob_stats: BlobStoreStats,
    /// Entropy coding statistics
    pub entropy_stats: EntropyStats,
    /// Compression algorithm used
    pub algorithm: EntropyAlgorithm,
    /// Number of successful compressions
    pub compressions: u64,
    /// Number of successful decompressions
    pub decompressions: u64,
    /// Total time spent compressing (microseconds)
    pub compression_time_us: u64,
    /// Total time spent decompressing (microseconds)
    pub decompression_time_us: u64,
}

impl EntropyCompressionStats {
    /// Create new entropy compression statistics
    pub fn new(algorithm: EntropyAlgorithm) -> Self {
        Self {
            blob_stats: BlobStoreStats::default(),
            entropy_stats: EntropyStats::new(0, 0, 0.0),
            algorithm,
            compressions: 0,
            decompressions: 0,
            compression_time_us: 0,
            decompression_time_us: 0,
        }
    }

    /// Get average compression time per operation
    pub fn avg_compression_time_us(&self) -> f64 {
        if self.compressions > 0 {
            self.compression_time_us as f64 / self.compressions as f64
        } else {
            0.0
        }
    }

    /// Get average decompression time per operation
    pub fn avg_decompression_time_us(&self) -> f64 {
        if self.decompressions > 0 {
            self.decompression_time_us as f64 / self.decompressions as f64
        } else {
            0.0
        }
    }
}

/// Huffman coding blob store wrapper
pub struct HuffmanBlobStore<S: BlobStore> {
    inner: S,
    stats: EntropyCompressionStats,
    training_data: Vec<u8>,
    encoder: Option<HuffmanEncoder>,
    tree: Option<HuffmanTree>,
}

impl<S: BlobStore> HuffmanBlobStore<S> {
    /// Create new Huffman blob store
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            stats: EntropyCompressionStats::new(EntropyAlgorithm::Huffman),
            training_data: Vec::new(),
            encoder: None,
            tree: None,
        }
    }

    /// Add training data for building Huffman tree
    pub fn add_training_data(&mut self, data: &[u8]) {
        self.training_data.extend_from_slice(data);
    }

    /// Build Huffman tree from training data
    pub fn build_tree(&mut self) -> Result<()> {
        if self.training_data.is_empty() {
            return Err(ZiporaError::invalid_data("No training data provided"));
        }

        let tree = HuffmanTree::from_data(&self.training_data)?;
        let encoder = HuffmanEncoder::new(&self.training_data)?;

        self.tree = Some(tree);
        self.encoder = Some(encoder);

        Ok(())
    }

    /// Get compression statistics
    pub fn compression_stats(&self) -> &EntropyCompressionStats {
        &self.stats
    }

    /// Compress data using Huffman coding
    fn compress_data(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();

        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| ZiporaError::invalid_data("Huffman tree not built"))?;

        let compressed = encoder.encode(data)?;

        self.stats.compression_time_us += start.elapsed().as_micros() as u64;
        self.stats.compressions += 1;

        // Update entropy statistics
        let entropy = EntropyStats::calculate_entropy(data);
        self.stats.entropy_stats = EntropyStats::new(data.len(), compressed.len(), entropy);

        Ok(compressed)
    }

    /// Decompress data using Huffman coding
    #[allow(dead_code)]
    fn decompress_data(&mut self, compressed: &[u8], original_length: usize) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();

        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| ZiporaError::invalid_data("Huffman tree not built"))?;

        let decoder = HuffmanDecoder::new(tree.clone());
        let decompressed = decoder.decode(compressed, original_length)?;

        self.stats.decompression_time_us += start.elapsed().as_micros() as u64;
        self.stats.decompressions += 1;

        Ok(decompressed)
    }
}

impl<S: BlobStore> BlobStore for HuffmanBlobStore<S> {
    fn get(&self, id: crate::RecordId) -> Result<Vec<u8>> {
        // For now, delegate to inner store (would need metadata for decompression)
        self.inner.get(id)
    }

    fn put(&mut self, data: &[u8]) -> Result<crate::RecordId> {
        if self.encoder.is_some() && !data.is_empty() {
            match self.compress_data(data) {
                Ok(compressed) => {
                    let id = self.inner.put(&compressed)?;
                    self.stats.blob_stats.put_count += 1;
                    Ok(id)
                }
                Err(_) => {
                    // Fall back to uncompressed
                    self.inner.put(data)
                }
            }
        } else {
            self.inner.put(data)
        }
    }

    fn remove(&mut self, id: crate::RecordId) -> Result<()> {
        self.inner.remove(id)
    }

    fn contains(&self, id: crate::RecordId) -> bool {
        self.inner.contains(id)
    }

    fn size(&self, id: crate::RecordId) -> Result<Option<usize>> {
        self.inner.size(id)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }

    fn stats(&self) -> BlobStoreStats {
        self.inner.stats()
    }
}

/// rANS coding blob store wrapper
pub struct RansBlobStore<S: BlobStore> {
    inner: S,
    stats: EntropyCompressionStats,
    encoder: Option<Rans64Encoder<ParallelX1>>,
}

impl<S: BlobStore> RansBlobStore<S> {
    /// Create new rANS blob store
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            stats: EntropyCompressionStats::new(EntropyAlgorithm::Rans),
            encoder: None,
        }
    }

    /// Train rANS encoder with data
    pub fn train(&mut self, data: &[u8]) -> Result<()> {
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        let encoder = Rans64Encoder::<ParallelX1>::new(&frequencies)?;
        self.encoder = Some(encoder);

        Ok(())
    }

    /// Get compression statistics
    pub fn compression_stats(&self) -> &EntropyCompressionStats {
        &self.stats
    }
}

impl<S: BlobStore> BlobStore for RansBlobStore<S> {
    fn get(&self, id: crate::RecordId) -> Result<Vec<u8>> {
        self.inner.get(id)
    }

    fn put(&mut self, data: &[u8]) -> Result<crate::RecordId> {
        // For now, delegate to inner store (would need full implementation)
        self.inner.put(data)
    }

    fn remove(&mut self, id: crate::RecordId) -> Result<()> {
        self.inner.remove(id)
    }

    fn contains(&self, id: crate::RecordId) -> bool {
        self.inner.contains(id)
    }

    fn size(&self, id: crate::RecordId) -> Result<Option<usize>> {
        self.inner.size(id)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }

    fn stats(&self) -> BlobStoreStats {
        self.inner.stats()
    }
}

/// Dictionary compression blob store wrapper
pub struct DictionaryBlobStore<S: BlobStore> {
    inner: S,
    stats: EntropyCompressionStats,
    compressor: Option<DictionaryCompressor>,
}

impl<S: BlobStore> DictionaryBlobStore<S> {
    /// Create new dictionary blob store
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            stats: EntropyCompressionStats::new(EntropyAlgorithm::Dictionary),
            compressor: None,
        }
    }

    /// Train dictionary with data
    pub fn train(&mut self, data: &[u8]) -> Result<()> {
        let builder = DictionaryBuilder::new();
        let dictionary = builder.build(data);
        let compressor = DictionaryCompressor::new(dictionary);

        self.compressor = Some(compressor);

        Ok(())
    }

    /// Get compression statistics
    pub fn compression_stats(&self) -> &EntropyCompressionStats {
        &self.stats
    }
}

impl<S: BlobStore> BlobStore for DictionaryBlobStore<S> {
    fn get(&self, id: crate::RecordId) -> Result<Vec<u8>> {
        self.inner.get(id)
    }

    fn put(&mut self, data: &[u8]) -> Result<crate::RecordId> {
        // For now, delegate to inner store (would need full implementation)
        self.inner.put(data)
    }

    fn remove(&mut self, id: crate::RecordId) -> Result<()> {
        self.inner.remove(id)
    }

    fn contains(&self, id: crate::RecordId) -> bool {
        self.inner.contains(id)
    }

    fn size(&self, id: crate::RecordId) -> Result<Option<usize>> {
        self.inner.size(id)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }

    fn stats(&self) -> BlobStoreStats {
        self.inner.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob_store::MemoryBlobStore;

    #[test]
    fn test_huffman_blob_store_creation() {
        let inner = MemoryBlobStore::new();
        let huffman_store = HuffmanBlobStore::new(inner);

        assert_eq!(
            huffman_store.compression_stats().algorithm,
            EntropyAlgorithm::Huffman
        );
        assert_eq!(huffman_store.compression_stats().compressions, 0);
    }

    #[test]
    fn test_huffman_blob_store_training() {
        let inner = MemoryBlobStore::new();
        let mut huffman_store = HuffmanBlobStore::new(inner);

        huffman_store.add_training_data(b"hello world hello world");
        let result = huffman_store.build_tree();
        assert!(result.is_ok());
    }

    #[test]
    fn test_rans_blob_store_creation() {
        let inner = MemoryBlobStore::new();
        let rans_store = RansBlobStore::new(inner);

        assert_eq!(
            rans_store.compression_stats().algorithm,
            EntropyAlgorithm::Rans
        );
    }

    #[test]
    fn test_dictionary_blob_store_creation() {
        let inner = MemoryBlobStore::new();
        let dict_store = DictionaryBlobStore::new(inner);

        assert_eq!(
            dict_store.compression_stats().algorithm,
            EntropyAlgorithm::Dictionary
        );
    }

    #[test]
    fn test_entropy_compression_stats() {
        let mut stats = EntropyCompressionStats::new(EntropyAlgorithm::Huffman);

        stats.compressions = 10;
        stats.compression_time_us = 1000;

        assert_eq!(stats.avg_compression_time_us(), 100.0);

        stats.decompressions = 5;
        stats.decompression_time_us = 500;

        assert_eq!(stats.avg_decompression_time_us(), 100.0);
    }

    #[test]
    fn test_huffman_blob_store_basic_operations() {
        let inner = MemoryBlobStore::new();
        let mut huffman_store = HuffmanBlobStore::new(inner);

        // Test basic blob store operations
        let data = b"test data";
        let id = huffman_store.put(data).unwrap();

        assert!(huffman_store.contains(id));
        assert_eq!(huffman_store.len(), 1);

        let retrieved = huffman_store.get(id).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_rans_blob_store_training() {
        let inner = MemoryBlobStore::new();
        let mut rans_store = RansBlobStore::new(inner);

        let training_data = b"hello world hello world hello";
        let result = rans_store.train(training_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dictionary_blob_store_training() {
        let inner = MemoryBlobStore::new();
        let mut dict_store = DictionaryBlobStore::new(inner);

        let training_data = b"hello world hello world hello";
        let result = dict_store.train(training_data);
        assert!(result.is_ok());
    }
}
