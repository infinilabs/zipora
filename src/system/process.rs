//! # Process Management Utilities
//!
//! High-performance process spawning, control, and bidirectional communication.
//! Inspired by production-grade process management with async/await integration.

use std::process::{Command, Stdio};
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{Mutex, RwLock};
use tokio::process::{Command as AsyncCommand};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader as AsyncBufReader};
use crate::error::{Result, ZiporaError};

/// Process execution configuration
#[derive(Debug, Clone)]
pub struct ProcessConfig {
    /// Working directory for the process
    pub working_dir: Option<String>,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
    /// Timeout in milliseconds (None = no timeout)
    pub timeout_ms: Option<u64>,
    /// Enable bidirectional communication
    pub bidirectional: bool,
    /// Buffer size for I/O operations
    pub buffer_size: usize,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            working_dir: None,
            env_vars: HashMap::new(),
            timeout_ms: Some(30_000), // 30 second default timeout
            bidirectional: false,
            buffer_size: 8192,
        }
    }
}

/// Process execution result
#[derive(Debug, Clone)]
pub struct ProcessResult {
    /// Exit code of the process
    pub exit_code: i32,
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

/// Bidirectional pipe for process communication
pub struct BidirectionalPipe {
    process: Arc<Mutex<Option<tokio::process::Child>>>,
    stdin_writer: Arc<Mutex<Option<tokio::process::ChildStdin>>>,
    stdout_reader: Arc<Mutex<Option<AsyncBufReader<tokio::process::ChildStdout>>>>,
    stderr_reader: Arc<Mutex<Option<AsyncBufReader<tokio::process::ChildStderr>>>>,
    config: ProcessConfig,
}

impl BidirectionalPipe {
    /// Create a new bidirectional pipe for a command
    pub async fn new(command: &str, args: &[&str], config: ProcessConfig) -> Result<Self> {
        let mut cmd = AsyncCommand::new(command);
        cmd.args(args);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Set working directory if specified
        if let Some(ref dir) = config.working_dir {
            cmd.current_dir(dir);
        }

        // Set environment variables
        for (key, value) in &config.env_vars {
            cmd.env(key, value);
        }

        // Spawn the process
        let mut child = cmd.spawn()
            .map_err(|e| ZiporaError::invalid_data(&format!("Failed to spawn process: {}", e)))?;

        // Extract pipes
        let stdin = child.stdin.take()
            .ok_or_else(|| ZiporaError::invalid_data("Failed to get stdin pipe"))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| ZiporaError::invalid_data("Failed to get stdout pipe"))?;
        let stderr = child.stderr.take()
            .ok_or_else(|| ZiporaError::invalid_data("Failed to get stderr pipe"))?;

        Ok(Self {
            process: Arc::new(Mutex::new(Some(child))),
            stdin_writer: Arc::new(Mutex::new(Some(stdin))),
            stdout_reader: Arc::new(Mutex::new(Some(AsyncBufReader::new(stdout)))),
            stderr_reader: Arc::new(Mutex::new(Some(AsyncBufReader::new(stderr)))),
            config,
        })
    }

    /// Write data to the process stdin
    pub async fn write_line(&self, line: &str) -> Result<()> {
        let mut stdin_guard = self.stdin_writer.lock().await;
        if let Some(ref mut stdin) = *stdin_guard {
            stdin.write_all(line.as_bytes()).await
                .map_err(|e| ZiporaError::invalid_data(&format!("Failed to write to stdin: {}", e)))?;
            stdin.write_all(b"\n").await
                .map_err(|e| ZiporaError::invalid_data(&format!("Failed to write newline: {}", e)))?;
            stdin.flush().await
                .map_err(|e| ZiporaError::invalid_data(&format!("Failed to flush stdin: {}", e)))?;
            Ok(())
        } else {
            Err(ZiporaError::invalid_data("Stdin pipe is not available"))
        }
    }

    /// Read a line from process stdout
    pub async fn read_stdout_line(&self) -> Result<Option<String>> {
        let mut stdout_guard = self.stdout_reader.lock().await;
        if let Some(ref mut stdout) = *stdout_guard {
            let mut line = String::new();
            match stdout.read_line(&mut line).await {
                Ok(0) => Ok(None), // EOF
                Ok(_) => {
                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }
                    Ok(Some(line))
                }
                Err(e) => Err(ZiporaError::invalid_data(&format!("Failed to read stdout: {}", e))),
            }
        } else {
            Err(ZiporaError::invalid_data("Stdout pipe is not available"))
        }
    }

    /// Read a line from process stderr
    pub async fn read_stderr_line(&self) -> Result<Option<String>> {
        let mut stderr_guard = self.stderr_reader.lock().await;
        if let Some(ref mut stderr) = *stderr_guard {
            let mut line = String::new();
            match stderr.read_line(&mut line).await {
                Ok(0) => Ok(None), // EOF
                Ok(_) => {
                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }
                    Ok(Some(line))
                }
                Err(e) => Err(ZiporaError::invalid_data(&format!("Failed to read stderr: {}", e))),
            }
        } else {
            Err(ZiporaError::invalid_data("Stderr pipe is not available"))
        }
    }

    /// Wait for the process to complete and get the exit status
    pub async fn wait(&self) -> Result<i32> {
        let mut process_guard = self.process.lock().await;
        if let Some(mut child) = process_guard.take() {
            let status = child.wait().await
                .map_err(|e| ZiporaError::invalid_data(&format!("Failed to wait for process: {}", e)))?;
            Ok(status.code().unwrap_or(-1))
        } else {
            Err(ZiporaError::invalid_data("Process is not available"))
        }
    }

    /// Terminate the process forcefully
    pub async fn kill(&self) -> Result<()> {
        let mut process_guard = self.process.lock().await;
        if let Some(ref mut child) = *process_guard {
            child.kill().await
                .map_err(|e| ZiporaError::invalid_data(&format!("Failed to kill process: {}", e)))?;
        }
        Ok(())
    }
}

/// Process executor for simple command execution
pub struct ProcessExecutor {
    config: ProcessConfig,
}

impl ProcessExecutor {
    /// Create a new process executor with default configuration
    pub fn new() -> Self {
        Self {
            config: ProcessConfig::default(),
        }
    }

    /// Create a process executor with custom configuration
    pub fn with_config(config: ProcessConfig) -> Self {
        Self { config }
    }

    /// Execute a command synchronously
    pub fn execute(&self, command: &str, args: &[&str]) -> Result<ProcessResult> {
        let start_time = std::time::Instant::now();

        let mut cmd = Command::new(command);
        cmd.args(args);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Set working directory if specified
        if let Some(ref dir) = self.config.working_dir {
            cmd.current_dir(dir);
        }

        // Set environment variables
        for (key, value) in &self.config.env_vars {
            cmd.env(key, value);
        }

        // Execute the command
        let output = cmd.output()
            .map_err(|e| ZiporaError::invalid_data(&format!("Failed to execute command: {}", e)))?;

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(ProcessResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            execution_time_ms,
        })
    }

    /// Execute a command asynchronously
    pub async fn execute_async(&self, command: &str, args: &[&str]) -> Result<ProcessResult> {
        let start_time = std::time::Instant::now();

        let mut cmd = AsyncCommand::new(command);
        cmd.args(args);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Set working directory if specified
        if let Some(ref dir) = self.config.working_dir {
            cmd.current_dir(dir);
        }

        // Set environment variables
        for (key, value) in &self.config.env_vars {
            cmd.env(key, value);
        }

        // Execute the command with timeout
        let output = if let Some(timeout_ms) = self.config.timeout_ms {
            let timeout_duration = std::time::Duration::from_millis(timeout_ms);
            tokio::time::timeout(timeout_duration, cmd.output()).await
                .map_err(|_| ZiporaError::invalid_data("Process execution timed out"))?
                .map_err(|e| ZiporaError::invalid_data(&format!("Failed to execute command: {}", e)))?
        } else {
            cmd.output().await
                .map_err(|e| ZiporaError::invalid_data(&format!("Failed to execute command: {}", e)))?
        };

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(ProcessResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            execution_time_ms,
        })
    }

    /// Execute a shell command (cross-platform)
    pub fn execute_shell(&self, command: &str) -> Result<ProcessResult> {
        #[cfg(target_os = "windows")]
        {
            self.execute("cmd", &["/C", command])
        }
        #[cfg(not(target_os = "windows"))]
        {
            self.execute("sh", &["-c", command])
        }
    }

    /// Execute a shell command asynchronously (cross-platform)
    pub async fn execute_shell_async(&self, command: &str) -> Result<ProcessResult> {
        #[cfg(target_os = "windows")]
        {
            self.execute_async("cmd", &["/C", command]).await
        }
        #[cfg(not(target_os = "windows"))]
        {
            self.execute_async("sh", &["-c", command]).await
        }
    }
}

impl Default for ProcessExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Process pool for managing multiple concurrent processes
pub struct ProcessPool {
    max_concurrent: usize,
    active_processes: Arc<RwLock<HashMap<u64, Arc<Mutex<tokio::process::Child>>>>>,
    next_id: Arc<Mutex<u64>>,
}

impl ProcessPool {
    /// Create a new process pool with a maximum number of concurrent processes
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            active_processes: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Get the number of active processes
    pub async fn active_count(&self) -> usize {
        self.active_processes.read().await.len()
    }

    /// Check if the pool has capacity for more processes
    pub async fn has_capacity(&self) -> bool {
        self.active_count().await < self.max_concurrent
    }

    /// Spawn a new process and add it to the pool
    pub async fn spawn_process(&self, command: &str, args: &[&str], config: ProcessConfig) -> Result<u64> {
        // Check capacity
        if !self.has_capacity().await {
            return Err(ZiporaError::invalid_data("Process pool at maximum capacity"));
        }

        // Create the command
        let mut cmd = AsyncCommand::new(command);
        cmd.args(args);

        if let Some(ref dir) = config.working_dir {
            cmd.current_dir(dir);
        }

        for (key, value) in &config.env_vars {
            cmd.env(key, value);
        }

        // Spawn the process
        let child = cmd.spawn()
            .map_err(|e| ZiporaError::invalid_data(&format!("Failed to spawn process: {}", e)))?;

        // Generate unique ID
        let mut id_guard = self.next_id.lock().await;
        let process_id = *id_guard;
        *id_guard += 1;

        // Add to active processes
        let mut processes = self.active_processes.write().await;
        processes.insert(process_id, Arc::new(Mutex::new(child)));

        Ok(process_id)
    }

    /// Wait for a specific process to complete
    pub async fn wait_for_process(&self, process_id: u64) -> Result<i32> {
        let process_arc = {
            let processes = self.active_processes.read().await;
            processes.get(&process_id).cloned()
                .ok_or_else(|| ZiporaError::invalid_data("Process not found in pool"))?
        };

        let mut child = process_arc.lock().await;
        let status = child.wait().await
            .map_err(|e| ZiporaError::invalid_data(&format!("Failed to wait for process: {}", e)))?;

        // Remove from active processes
        let mut processes = self.active_processes.write().await;
        processes.remove(&process_id);

        Ok(status.code().unwrap_or(-1))
    }

    /// Kill a specific process
    pub async fn kill_process(&self, process_id: u64) -> Result<()> {
        let process_arc = {
            let processes = self.active_processes.read().await;
            processes.get(&process_id).cloned()
                .ok_or_else(|| ZiporaError::invalid_data("Process not found in pool"))?
        };

        let mut child = process_arc.lock().await;
        child.kill().await
            .map_err(|e| ZiporaError::invalid_data(&format!("Failed to kill process: {}", e)))?;

        // Remove from active processes
        let mut processes = self.active_processes.write().await;
        processes.remove(&process_id);

        Ok(())
    }

    /// Kill all active processes
    pub async fn kill_all(&self) -> Result<()> {
        let process_ids: Vec<u64> = {
            let processes = self.active_processes.read().await;
            processes.keys().cloned().collect()
        };

        for process_id in process_ids {
            let _ = self.kill_process(process_id).await; // Ignore errors for individual kills
        }

        Ok(())
    }
}

/// Process manager for high-level process management operations
pub struct ProcessManager {
    executor: ProcessExecutor,
    pool: ProcessPool,
}

impl ProcessManager {
    /// Create a new process manager
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            executor: ProcessExecutor::new(),
            pool: ProcessPool::new(max_concurrent),
        }
    }

    /// Create a process manager with custom configuration
    pub fn with_config(config: ProcessConfig, max_concurrent: usize) -> Self {
        Self {
            executor: ProcessExecutor::with_config(config),
            pool: ProcessPool::new(max_concurrent),
        }
    }

    /// Get a reference to the process executor
    pub fn executor(&self) -> &ProcessExecutor {
        &self.executor
    }

    /// Get a reference to the process pool
    pub fn pool(&self) -> &ProcessPool {
        &self.pool
    }

    /// Execute a command with automatic resource management
    pub async fn execute_managed(&self, command: &str, args: &[&str]) -> Result<ProcessResult> {
        self.executor.execute_async(command, args).await
    }

    /// Create a bidirectional pipe for interactive processes
    pub async fn create_pipe(&self, command: &str, args: &[&str], config: ProcessConfig) -> Result<BidirectionalPipe> {
        BidirectionalPipe::new(command, args, config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_process_executor() {
        let executor = ProcessExecutor::new();
        
        // Test simple command execution
        let result = executor.execute_shell("echo hello").unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_process_executor_async() {
        let executor = ProcessExecutor::new();
        
        // Test async command execution
        let result = executor.execute_shell_async("echo async_hello").await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("async_hello"));
    }

    #[tokio::test]
    async fn test_process_pool() {
        let pool = ProcessPool::new(2);
        
        // Test capacity checking
        assert!(pool.has_capacity().await);
        assert_eq!(pool.active_count().await, 0);
    }

    #[test]
    fn test_process_config() {
        let mut config = ProcessConfig::default();
        config.env_vars.insert("TEST_VAR".to_string(), "test_value".to_string());
        config.working_dir = Some("/tmp".to_string());
        
        assert_eq!(config.env_vars.get("TEST_VAR"), Some(&"test_value".to_string()));
        assert_eq!(config.working_dir.as_ref(), Some(&"/tmp".to_string()));
    }

    #[tokio::test]
    async fn test_process_manager() {
        let manager = ProcessManager::new(5);
        
        // Test managed execution
        let result = manager.execute_managed("echo", &["manager_test"]).await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("manager_test"));
    }
}