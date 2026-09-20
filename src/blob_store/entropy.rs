//! Entropy coding blob store implementations
//!
//! This module provides blob store wrappers that use entropy coding for compression.

use crate::blob_store::{BlobStore, BlobStoreStats};
use crate::entropy::rans::{ParallelX1, Rans64Decoder, Rans64Encoder};
use crate::entropy::{
    DictionaryBuilder, DictionaryCompressor, EntropyStats, HuffmanDecoder, HuffmanEncoder,
    HuffmanTree,
};
use crate::error::{Result, ZiporaError};

/// Helper to frame a payload with `[flag: u8][uncompressed_len: u32 LE][payload]`
fn frame_entropy_blob(flag: u8, uncompressed_len: usize, payload: &[u8]) -> Result<Vec<u8>> {
    let len_u32 = u32::try_from(uncompressed_len).map_err(|_| {
        ZiporaError::invalid_data("Entropy blob uncompressed length exceeds u32::MAX")
    })?;
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(flag);
    out.extend_from_slice(&len_u32.to_le_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

/// Helper to parse the 5-byte frame header `(flag, uncompressed_len, payload)`
fn parse_entropy_frame(raw: &[u8]) -> Result<(u8, usize, &[u8])> {
    let (&flag, rest) = raw
        .split_first()
        .ok_or_else(|| ZiporaError::invalid_data("Truncated entropy blob header"))?;
    let len_bytes = rest
        .first_chunk::<4>()
        .ok_or_else(|| ZiporaError::invalid_data("Truncated entropy blob length"))?;
    let uncompressed_len = u32::from_le_bytes(*len_bytes) as usize;
    let payload = rest
        .get(4..)
        .ok_or_else(|| ZiporaError::invalid_data("Truncated entropy blob payload"))?;
    Ok((flag, uncompressed_len, payload))
}

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
    decoder: Option<HuffmanDecoder>,
    record_sizes: std::collections::HashMap<crate::RecordId, usize>,
}

impl<S: BlobStore> HuffmanBlobStore<S> {
    /// Create new Huffman blob store
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            stats: EntropyCompressionStats::new(EntropyAlgorithm::Huffman),
            training_data: Vec::new(),
            encoder: None,
            decoder: None,
            record_sizes: std::collections::HashMap::new(),
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
        let decoder = HuffmanDecoder::new(tree);

        self.decoder = Some(decoder);
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
}

impl<S: BlobStore> BlobStore for HuffmanBlobStore<S> {
    fn get(&self, id: crate::RecordId) -> Result<Vec<u8>> {
        let raw = self.inner.get(id)?;
        let (flag, uncompressed_len, payload) = parse_entropy_frame(&raw)?;
        if flag == 0 {
            if payload.len() != uncompressed_len {
                return Err(ZiporaError::invalid_data(
                    "Uncompressed Huffman blob length mismatch",
                ));
            }
            return Ok(payload.to_vec());
        }
        let decoder = self
            .decoder
            .as_ref()
            .ok_or_else(|| ZiporaError::invalid_data("Huffman decoder not initialized"))?;
        decoder.decode(payload, uncompressed_len)
    }

    fn put(&mut self, data: &[u8]) -> Result<crate::RecordId> {
        let id = if self.encoder.is_some() && !data.is_empty() {
            match self.compress_data(data) {
                Ok(compressed) => {
                    let framed = frame_entropy_blob(1, data.len(), &compressed)?;
                    let id = self.inner.put(&framed)?;
                    self.stats.blob_stats.put_count += 1;
                    id
                }
                Err(_) => {
                    let framed = frame_entropy_blob(0, data.len(), data)?;
                    self.inner.put(&framed)?
                }
            }
        } else {
            let framed = frame_entropy_blob(0, data.len(), data)?;
            self.inner.put(&framed)?
        };
        self.record_sizes.insert(id, data.len());
        Ok(id)
    }

    fn remove(&mut self, id: crate::RecordId) -> Result<()> {
        self.record_sizes.remove(&id);
        self.inner.remove(id)
    }

    fn contains(&self, id: crate::RecordId) -> bool {
        self.inner.contains(id)
    }

    fn size(&self, id: crate::RecordId) -> Result<Option<usize>> {
        if !self.inner.contains(id) {
            return Ok(None);
        }
        if let Some(&sz) = self.record_sizes.get(&id) {
            return Ok(Some(sz));
        }
        let raw = self.inner.get(id)?;
        let (_, uncompressed_len, _) = parse_entropy_frame(&raw)?;
        Ok(Some(uncompressed_len))
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
    decoder: Option<Rans64Decoder<ParallelX1>>,
    record_sizes: std::collections::HashMap<crate::RecordId, usize>,
}

impl<S: BlobStore> RansBlobStore<S> {
    /// Create new rANS blob store
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            stats: EntropyCompressionStats::new(EntropyAlgorithm::Rans),
            encoder: None,
            decoder: None,
            record_sizes: std::collections::HashMap::new(),
        }
    }

    /// Train rANS encoder with data
    pub fn train(&mut self, data: &[u8]) -> Result<()> {
        let mut frequencies = [0u32; 256];
        for &byte in data {
            frequencies[byte as usize] += 1;
        }

        let encoder = Rans64Encoder::<ParallelX1>::new(&frequencies)?;
        let decoder = Rans64Decoder::<ParallelX1>::new(&encoder);
        self.encoder = Some(encoder);
        self.decoder = Some(decoder);

        Ok(())
    }

    /// Get compression statistics
    pub fn compression_stats(&self) -> &EntropyCompressionStats {
        &self.stats
    }
}

impl<S: BlobStore> BlobStore for RansBlobStore<S> {
    fn get(&self, id: crate::RecordId) -> Result<Vec<u8>> {
        let raw = self.inner.get(id)?;
        let (flag, uncompressed_len, payload) = parse_entropy_frame(&raw)?;
        if flag == 0 {
            if payload.len() != uncompressed_len {
                return Err(ZiporaError::invalid_data(
                    "Uncompressed rANS blob length mismatch",
                ));
            }
            return Ok(payload.to_vec());
        }
        let decoder = self
            .decoder
            .as_ref()
            .ok_or_else(|| ZiporaError::invalid_data("rANS decoder not initialized"))?;
        decoder.decode(payload, uncompressed_len)
    }

    fn put(&mut self, data: &[u8]) -> Result<crate::RecordId> {
        if let Some(ref encoder) = self.encoder
            && !data.is_empty()
        {
            let start = std::time::Instant::now();
            if let Ok(compressed) = encoder.encode(data) {
                self.stats.compression_time_us += start.elapsed().as_micros() as u64;
                self.stats.compressions += 1;
                let entropy = EntropyStats::calculate_entropy(data);
                self.stats.entropy_stats =
                    EntropyStats::new(data.len(), compressed.len(), entropy);
                let framed = frame_entropy_blob(1, data.len(), &compressed)?;
                let id = self.inner.put(&framed)?;
                self.record_sizes.insert(id, data.len());
                return Ok(id);
            }
        }
        let framed = frame_entropy_blob(0, data.len(), data)?;
        let id = self.inner.put(&framed)?;
        self.record_sizes.insert(id, data.len());
        Ok(id)
    }

    fn remove(&mut self, id: crate::RecordId) -> Result<()> {
        self.record_sizes.remove(&id);
        self.inner.remove(id)
    }

    fn contains(&self, id: crate::RecordId) -> bool {
        self.inner.contains(id)
    }

    fn size(&self, id: crate::RecordId) -> Result<Option<usize>> {
        if !self.inner.contains(id) {
            return Ok(None);
        }
        if let Some(&sz) = self.record_sizes.get(&id) {
            return Ok(Some(sz));
        }
        let raw = self.inner.get(id)?;
        let (_, uncompressed_len, _) = parse_entropy_frame(&raw)?;
        Ok(Some(uncompressed_len))
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
    record_sizes: std::collections::HashMap<crate::RecordId, usize>,
}

impl<S: BlobStore> DictionaryBlobStore<S> {
    /// Create new dictionary blob store
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            stats: EntropyCompressionStats::new(EntropyAlgorithm::Dictionary),
            compressor: None,
            record_sizes: std::collections::HashMap::new(),
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
        let raw = self.inner.get(id)?;
        let (flag, uncompressed_len, payload) = parse_entropy_frame(&raw)?;
        if flag == 0 {
            if payload.len() != uncompressed_len {
                return Err(ZiporaError::invalid_data(
                    "Uncompressed dictionary blob length mismatch",
                ));
            }
            return Ok(payload.to_vec());
        }
        let compressor = self
            .compressor
            .as_ref()
            .ok_or_else(|| ZiporaError::invalid_data("Dictionary compressor not initialized"))?;
        let decompressed = compressor.decompress(payload)?;
        if decompressed.len() != uncompressed_len {
            return Err(ZiporaError::invalid_data(
                "Dictionary blob decompressed length mismatch",
            ));
        }
        Ok(decompressed)
    }

    fn put(&mut self, data: &[u8]) -> Result<crate::RecordId> {
        if let Some(ref compressor) = self.compressor
            && !data.is_empty()
        {
            let start = std::time::Instant::now();
            if let Ok(compressed) = compressor.compress(data) {
                self.stats.compression_time_us += start.elapsed().as_micros() as u64;
                self.stats.compressions += 1;
                let entropy = EntropyStats::calculate_entropy(data);
                self.stats.entropy_stats =
                    EntropyStats::new(data.len(), compressed.len(), entropy);
                let framed = frame_entropy_blob(1, data.len(), &compressed)?;
                let id = self.inner.put(&framed)?;
                self.record_sizes.insert(id, data.len());
                return Ok(id);
            }
        }
        let framed = frame_entropy_blob(0, data.len(), data)?;
        let id = self.inner.put(&framed)?;
        self.record_sizes.insert(id, data.len());
        Ok(id)
    }

    fn remove(&mut self, id: crate::RecordId) -> Result<()> {
        self.record_sizes.remove(&id);
        self.inner.remove(id)
    }

    fn contains(&self, id: crate::RecordId) -> bool {
        self.inner.contains(id)
    }

    fn size(&self, id: crate::RecordId) -> Result<Option<usize>> {
        if !self.inner.contains(id) {
            return Ok(None);
        }
        if let Some(&sz) = self.record_sizes.get(&id) {
            return Ok(Some(sz));
        }
        let raw = self.inner.get(id)?;
        let (_, uncompressed_len, _) = parse_entropy_frame(&raw)?;
        Ok(Some(uncompressed_len))
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

    #[test]
    fn test_trained_entropy_blob_stores_roundtrip_and_size() {
        // RED test (C2 / D1): Trained HuffmanBlobStore, RansBlobStore, and DictionaryBlobStore
        // must round-trip put() -> get() losslessly and report uncompressed size via size().
        let sample = b"hello world hello world hello world";

        // 1. Trained HuffmanBlobStore
        let mut huff = HuffmanBlobStore::new(MemoryBlobStore::new());
        huff.add_training_data(sample);
        huff.build_tree().unwrap();
        let id = huff.put(sample).unwrap();
        assert_eq!(huff.size(id).unwrap(), Some(sample.len()));
        assert_eq!(huff.get(id).unwrap(), sample);

        // 2. Trained RansBlobStore
        let mut rans = RansBlobStore::new(MemoryBlobStore::new());
        rans.train(sample).unwrap();
        let id_r = rans.put(sample).unwrap();
        assert_eq!(rans.size(id_r).unwrap(), Some(sample.len()));
        assert_eq!(rans.get(id_r).unwrap(), sample);

        // 3. Trained DictionaryBlobStore
        let mut dict = DictionaryBlobStore::new(MemoryBlobStore::new());
        dict.train(sample).unwrap();
        let id_d = dict.put(sample).unwrap();
        assert_eq!(dict.size(id_d).unwrap(), Some(sample.len()));
        assert_eq!(dict.get(id_d).unwrap(), sample);
    }
}
