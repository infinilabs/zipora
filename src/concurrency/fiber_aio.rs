//! Asynchronous I/O integration with fiber-based concurrency
//!
//! This module provides high-performance async I/O operations optimized for fiber-based
//! concurrency with adaptive provider selection and hardware-aware optimizations.

use crate::error::{Result, ZiporaError};
use crate::system::RuntimeCpuFeatures;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::fs::File;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt, AsyncWrite, AsyncWriteExt};

/// Configuration for async I/O provider selection
#[derive(Debug, Clone)]
pub struct FiberAioConfig {
    /// Preferred I/O provider
    pub io_provider: IoProvider,
    /// Buffer size for read operations
    pub read_buffer_size: usize,
    /// Buffer size for write operations
    pub write_buffer_size: usize,
    /// Enable vectored I/O operations
    pub enable_vectored_io: bool,
    /// Enable direct I/O when available
    pub enable_direct_io: bool,
    /// Read-ahead size for sequential access
    pub read_ahead_size: usize,
}

impl Default for FiberAioConfig {
    fn default() -> Self {
        Self {
            io_provider: IoProvider::auto_detect(),
            read_buffer_size: 64 * 1024,     // 64KB
            write_buffer_size: 64 * 1024,    // 64KB
            enable_vectored_io: true,
            enable_direct_io: false,
            read_ahead_size: 256 * 1024,     // 256KB
        }
    }
}

/// Available async I/O providers
#[derive(Debug, Clone, PartialEq)]
pub enum IoProvider {
    /// Standard tokio async I/O
    Tokio,
    /// Linux io_uring (when available)
    #[cfg(target_os = "linux")]
    IoUring,
    /// POSIX AIO
    #[cfg(unix)]
    PosixAio,
    /// Windows IOCP
    #[cfg(windows)]
    Iocp,
    /// Auto-detect best provider for platform
    Auto,
}

impl IoProvider {
    /// Automatically detect the best I/O provider for the current platform
    pub fn auto_detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            // Check if io_uring is available
            if Self::is_io_uring_available() {
                return Self::IoUring;
            }
        }

        #[cfg(unix)]
        {
            // Check if POSIX AIO is available
            if Self::is_posix_aio_available() {
                return Self::PosixAio;
            }
        }

        #[cfg(windows)]
        {
            return Self::Iocp;
        }

        // Fallback to tokio
        Self::Tokio
    }

    #[cfg(target_os = "linux")]
    fn is_io_uring_available() -> bool {
        // Check if io_uring is supported (kernel 5.1+)
        use std::fs;
        fs::metadata("/proc/sys/kernel/io_uring_disabled").is_ok()
    }

    #[cfg(unix)]
    fn is_posix_aio_available() -> bool {
        // Check if POSIX AIO is available
        unsafe { libc::sysconf(libc::_SC_ASYNCHRONOUS_IO) > 0 }
    }
}

/// High-performance fiber-aware async I/O manager
pub struct FiberAio {
    config: FiberAioConfig,
    runtime_features: Arc<RuntimeCpuFeatures>,
    #[cfg(target_os = "linux")]
    io_uring: Option<Arc<IoUringContext>>,
}

#[cfg(target_os = "linux")]
struct IoUringContext {
    // Placeholder for io_uring integration
    // Would contain io_uring ring setup and management
}

impl FiberAio {
    /// Create a new fiber AIO manager with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(FiberAioConfig::default())
    }

    /// Create a new fiber AIO manager with custom configuration
    pub fn with_config(mut config: FiberAioConfig) -> Result<Self> {
        // Auto-detect provider if needed
        if config.io_provider == IoProvider::Auto {
            config.io_provider = IoProvider::auto_detect();
        }

        let runtime_features = Arc::new(RuntimeCpuFeatures::new());

        #[cfg(target_os = "linux")]
        let io_uring = if config.io_provider == IoProvider::IoUring {
            Some(Arc::new(IoUringContext {}))
        } else {
            None
        };

        Ok(Self {
            config,
            runtime_features,
            #[cfg(target_os = "linux")]
            io_uring,
        })
    }

    /// Open a file for async operations
    pub async fn open<P: AsRef<Path>>(&self, path: P) -> Result<FiberFile> {
        let file = File::open(path).await?;

        Ok(FiberFile::new(file, self.config.clone()))
    }

    /// Create a new file for async operations
    pub async fn create<P: AsRef<Path>>(&self, path: P) -> Result<FiberFile> {
        let file = File::create(path).await?;

        Ok(FiberFile::new(file, self.config.clone()))
    }

    /// Read entire file contents asynchronously
    pub async fn read_to_vec<P: AsRef<Path>>(&self, path: P) -> Result<Vec<u8>> {
        let mut file = self.open(path).await?;
        file.read_to_end().await
    }

    /// Write data to file asynchronously
    pub async fn write_all<P: AsRef<Path>>(&self, path: P, data: &[u8]) -> Result<()> {
        let mut file = self.create(path).await?;
        file.write_all(data).await
    }

    /// Copy file asynchronously with optimized buffering
    pub async fn copy<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        from: P,
        to: Q,
    ) -> Result<u64> {
        let mut src = self.open(from).await?;
        let mut dst = self.create(to).await?;

        src.copy_to(&mut dst).await
    }

    /// Get the current I/O provider being used
    pub fn io_provider(&self) -> &IoProvider {
        &self.config.io_provider
    }

    /// Get I/O configuration
    pub fn config(&self) -> &FiberAioConfig {
        &self.config
    }
}

/// Fiber-aware asynchronous file handle
pub struct FiberFile {
    inner: File,
    config: FiberAioConfig,
    position: u64,
    read_ahead_buffer: Vec<u8>,
    read_ahead_valid: usize,
    read_ahead_offset: u64,
}

impl FiberFile {
    fn new(file: File, config: FiberAioConfig) -> Self {
        let read_ahead_buffer = vec![0u8; config.read_ahead_size];

        Self {
            inner: file,
            config,
            position: 0,
            read_ahead_buffer,
            read_ahead_valid: 0,
            read_ahead_offset: u64::MAX,
        }
    }

    /// Read data from the file with read-ahead optimization
    pub async fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        // Check if we can serve from read-ahead buffer
        if self.position >= self.read_ahead_offset
            && self.position < self.read_ahead_offset + self.read_ahead_valid as u64
        {
            let offset_in_buffer = (self.position - self.read_ahead_offset) as usize;
            let available = self.read_ahead_valid - offset_in_buffer;
            let to_copy = buf.len().min(available);

            buf[..to_copy].copy_from_slice(
                &self.read_ahead_buffer[offset_in_buffer..offset_in_buffer + to_copy],
            );

            self.position += to_copy as u64;
            return Ok(to_copy);
        }

        // Direct read for large requests or misaligned access
        if buf.len() >= self.config.read_buffer_size {
            let bytes_read = self.inner.read(buf).await?;

            self.position += bytes_read as u64;
            return Ok(bytes_read);
        }

        // Perform read-ahead
        self.inner
            .seek(tokio::io::SeekFrom::Start(self.position))
            .await?;

        let bytes_read = self
            .inner
            .read(&mut self.read_ahead_buffer)
            .await?;

        self.read_ahead_valid = bytes_read;
        self.read_ahead_offset = self.position;

        // Serve from read-ahead buffer
        let to_copy = buf.len().min(bytes_read);
        buf[..to_copy].copy_from_slice(&self.read_ahead_buffer[..to_copy]);

        self.position += to_copy as u64;
        Ok(to_copy)
    }

    /// Read data at specific offset without changing file position
    pub async fn read_at(&mut self, buf: &mut [u8], offset: u64) -> Result<usize> {
        let old_position = self.position;

        self.seek(tokio::io::SeekFrom::Start(offset)).await?;
        let result = self.read(buf).await;

        // Restore position
        self.position = old_position;

        result
    }

    /// Write data to the file
    pub async fn write(&mut self, buf: &[u8]) -> Result<usize> {
        // Invalidate read-ahead buffer if write affects it
        if self.position >= self.read_ahead_offset
            && self.position < self.read_ahead_offset + self.read_ahead_valid as u64
        {
            self.read_ahead_valid = 0;
            self.read_ahead_offset = u64::MAX;
        }

        let bytes_written = self.inner.write(buf).await?;

        self.position += bytes_written as u64;
        Ok(bytes_written)
    }

    /// Write all data to the file
    pub async fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        self.inner.write_all(buf).await?;

        self.position += buf.len() as u64;

        // Invalidate read-ahead buffer
        self.read_ahead_valid = 0;
        self.read_ahead_offset = u64::MAX;

        Ok(())
    }

    /// Seek to a position in the file
    pub async fn seek(&mut self, pos: tokio::io::SeekFrom) -> Result<u64> {
        let new_position = self.inner.seek(pos).await?;

        self.position = new_position;

        // Invalidate read-ahead buffer if seeking outside it
        if new_position < self.read_ahead_offset
            || new_position >= self.read_ahead_offset + self.read_ahead_valid as u64
        {
            self.read_ahead_valid = 0;
            self.read_ahead_offset = u64::MAX;
        }

        Ok(new_position)
    }

    /// Get current file position
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Read entire file contents
    pub async fn read_to_end(&mut self) -> Result<Vec<u8>> {
        let mut contents = Vec::new();
        self.inner.read_to_end(&mut contents).await?;

        Ok(contents)
    }

    /// Copy all data to another file
    pub async fn copy_to(&mut self, dst: &mut FiberFile) -> Result<u64> {
        let mut buffer = vec![0u8; self.config.read_buffer_size];
        let mut total_copied = 0u64;

        loop {
            let bytes_read = self.read(&mut buffer).await?;
            if bytes_read == 0 {
                break;
            }

            dst.write_all(&buffer[..bytes_read]).await?;
            total_copied += bytes_read as u64;

            // Yield control to allow other fibers to run
            tokio::task::yield_now().await;
        }

        Ok(total_copied)
    }

    /// Flush any buffered data
    pub async fn flush(&mut self) -> Result<()> {
        Ok(self.inner.flush().await?)
    }

    /// Sync all data and metadata to disk
    pub async fn sync_all(&mut self) -> Result<()> {
        Ok(self.inner.sync_all().await?)
    }

    /// Sync data to disk (without metadata)
    pub async fn sync_data(&mut self) -> Result<()> {
        Ok(self.inner.sync_data().await?)
    }
}

/// Vectored I/O operations for multiple buffers
pub struct VectoredIo;

impl VectoredIo {
    /// Read data into multiple buffers
    pub async fn read_vectored<R>(
        reader: &mut R,
        bufs: &mut [tokio::io::ReadBuf<'_>],
    ) -> Result<usize>
    where
        R: AsyncRead + Unpin,
    {
        let mut total_read = 0;

        for buf in bufs {
            let bytes_read = reader.read_buf(buf).await?;

            total_read += bytes_read;

            if bytes_read == 0 {
                break;
            }
        }

        Ok(total_read)
    }

    /// Write data from multiple buffers
    pub async fn write_vectored<W>(writer: &mut W, bufs: &[std::io::IoSlice<'_>]) -> Result<usize>
    where
        W: AsyncWrite + Unpin,
    {
        let mut total_written = 0;

        for buf in bufs {
            let bytes_written = writer.write(buf).await?;

            total_written += bytes_written;

            if bytes_written < buf.len() {
                break;
            }
        }

        Ok(total_written)
    }
}

/// Async I/O utilities optimized for fiber-based concurrency
pub struct FiberIoUtils;

impl FiberIoUtils {
    /// Parallel file processing with controlled concurrency
    pub async fn process_files_parallel<P, F, R>(
        paths: Vec<P>,
        max_concurrent: usize,
        processor: F,
    ) -> Result<Vec<R>>
    where
        P: AsRef<Path> + Send + 'static,
        F: Fn(P) -> Pin<Box<dyn Future<Output = Result<R>> + Send>> + Send + Sync + Clone + 'static,
        R: Send + 'static,
    {
        use futures::stream::{self, StreamExt};

        let results = stream::iter(paths)
            .map(|path| {
                let processor = processor.clone();
                async move { processor(path).await }
            })
            .buffer_unordered(max_concurrent)
            .collect::<Vec<_>>()
            .await;

        // Collect results, propagating any errors
        let mut output = Vec::with_capacity(results.len());
        for result in results {
            output.push(result?);
        }

        Ok(output)
    }

    /// Batch file operations with automatic yielding
    pub async fn batch_process<I, F, T, R>(
        items: I,
        batch_size: usize,
        processor: F,
    ) -> Result<Vec<R>>
    where
        I: IntoIterator<Item = T>,
        F: Fn(Vec<T>) -> Pin<Box<dyn Future<Output = Result<Vec<R>>> + Send>> + Send + Sync + Clone,
        T: Send + Clone + 'static,
        R: Send + 'static,
    {
        let items: Vec<T> = items.into_iter().collect();
        let mut results = Vec::new();

        for chunk in items.chunks(batch_size) {
            let chunk_vec: Vec<T> = chunk.to_vec();
            let chunk_results = processor(chunk_vec).await?;
            results.extend(chunk_results);

            // Yield to allow other fibers to run
            tokio::task::yield_now().await;
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_fiber_aio_creation() {
        let aio = FiberAio::new().unwrap();
        
        // The provider should be a concrete provider after auto-detection
        match aio.io_provider() {
            IoProvider::Tokio => {},
            #[cfg(target_os = "linux")]
            IoProvider::IoUring => {},
            #[cfg(unix)]
            IoProvider::PosixAio => {},
            #[cfg(windows)]
            IoProvider::Iocp => {},
            IoProvider::Auto => panic!("Auto provider should have been resolved to a concrete provider"),
        }
    }

    #[tokio::test]
    async fn test_file_operations() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_data = b"Hello, Fiber AIO!";
        temp_file.write_all(test_data).unwrap();
        temp_file.flush().unwrap();

        let aio = FiberAio::new().unwrap();
        let mut file = aio.open(temp_file.path()).await.unwrap();

        let mut buffer = vec![0u8; test_data.len()];
        let bytes_read = file.read(&mut buffer).await.unwrap();

        assert_eq!(bytes_read, test_data.len());
        assert_eq!(&buffer, test_data);
    }

    #[tokio::test]
    async fn test_read_ahead() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_data = vec![42u8; 1024]; // 1KB of test data
        temp_file.write_all(&test_data).unwrap();
        temp_file.flush().unwrap();

        let config = FiberAioConfig {
            read_ahead_size: 512,
            ..Default::default()
        };

        let aio = FiberAio::with_config(config).unwrap();
        let mut file = aio.open(temp_file.path()).await.unwrap();

        // First read should trigger read-ahead
        let mut buffer1 = vec![0u8; 256];
        let bytes_read1 = file.read(&mut buffer1).await.unwrap();
        assert_eq!(bytes_read1, 256);

        // Second read should be served from read-ahead buffer
        let mut buffer2 = vec![0u8; 256];
        let bytes_read2 = file.read(&mut buffer2).await.unwrap();
        assert_eq!(bytes_read2, 256);

        assert_eq!(buffer1, vec![42u8; 256]);
        assert_eq!(buffer2, vec![42u8; 256]);
    }

    #[tokio::test]
    async fn test_write_operations() {
        let temp_file = NamedTempFile::new().unwrap();
        let test_data = b"Test write data";

        let aio = FiberAio::new().unwrap();
        aio.write_all(temp_file.path(), test_data).await.unwrap();

        // Verify written data
        let read_data = aio.read_to_vec(temp_file.path()).await.unwrap();
        assert_eq!(read_data, test_data);
    }

    #[tokio::test]
    async fn test_copy_file() {
        let mut src_file = NamedTempFile::new().unwrap();
        let dst_file = NamedTempFile::new().unwrap();
        let test_data = b"Copy test data";

        src_file.write_all(test_data).unwrap();
        src_file.flush().unwrap();

        let aio = FiberAio::new().unwrap();
        let copied_bytes = aio.copy(src_file.path(), dst_file.path()).await.unwrap();

        assert_eq!(copied_bytes, test_data.len() as u64);

        // Verify copied data
        let read_data = aio.read_to_vec(dst_file.path()).await.unwrap();
        assert_eq!(read_data, test_data);
    }

    #[tokio::test]
    async fn test_vectored_io() {
        let test_data = b"Vectored I/O test data";
        let mut cursor = std::io::Cursor::new(test_data);

        let mut buf1 = vec![0u8; 8];
        let mut buf2 = vec![0u8; 8];
        let mut buf3 = vec![0u8; 8];

        let mut read_bufs = [
            tokio::io::ReadBuf::new(&mut buf1),
            tokio::io::ReadBuf::new(&mut buf2),
            tokio::io::ReadBuf::new(&mut buf3),
        ];

        let total_read = VectoredIo::read_vectored(&mut cursor, &mut read_bufs)
            .await
            .unwrap();

        assert_eq!(total_read, test_data.len());
        assert_eq!(&buf1, b"Vectored");
        assert_eq!(&buf2, b" I/O tes");
        assert_eq!(&buf3[..6], b"t data");
    }
}