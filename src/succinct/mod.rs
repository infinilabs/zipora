//! Succinct data structures with constant-time rank and select operations
//!
//! This module provides space-efficient bit vectors and arrays with support for
//! rank (count of set bits up to position) and select (find position of nth set bit)
//! operations in constant time with ~3% space overhead.

pub mod bit_vector;
pub mod rank_select;

pub use bit_vector::BitVector;
pub use rank_select::{RankSelect256, RankSelectSe256};
