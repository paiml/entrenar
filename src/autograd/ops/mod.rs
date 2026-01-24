//! Autograd operations with backward passes
//!
//! This module provides differentiable operations for automatic differentiation.

mod activations;
mod attention;
mod basic;
mod matmul;
mod normalize;

// Re-export all public operations
pub use activations::{gelu, relu, softmax, swish};
pub use attention::attention;
pub use basic::{add, add_scaled, mul, scale, sum};
pub use matmul::{matmul, matmul_compute, transpose};
pub use normalize::layer_norm;
