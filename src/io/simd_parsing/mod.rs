//! # SIMD-Accelerated Parsing Operations
//!
//! High-performance parsing operations using SIMD acceleration for various data formats.
//! Follows the 6-tier SIMD framework with runtime adaptive selection.
//!
//! ## Modules
//!
//! - **json**: simdjson-style two-stage JSON parser with 2-3 GB/s throughput
//! - **csv**: High-performance CSV parser with delimiter detection and quote handling (1.8 GB/s)

pub mod json;
pub mod csv;

pub use json::{JsonParser, JsonValue, parse_json};
pub use csv::{CsvParser, CsvConfig, find_delimiter, find_newline, find_delimiters_bulk, parse_csv_line};
