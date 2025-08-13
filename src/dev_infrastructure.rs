//! Development Infrastructure
//! 
//! This module provides essential development infrastructure components including:
//! - Generic factory pattern implementation for object creation
//! - Comprehensive debugging framework with advanced utilities  
//! - Statistical analysis tools for performance monitoring
//!
//! The implementation draws inspiration from production-ready C++ infrastructure
//! while leveraging Rust's type system for enhanced safety and performance.

pub mod factory;
pub mod debug;
pub mod statistics;

pub use factory::*;
pub use debug::*;
pub use statistics::*;