//! CUDA-accelerated forward kernels for autograd
//!
//! This module wraps trueno-gpu forward kernels for GPU-accelerated forward passes.
//! Provides 10-100x speedup over CPU ndarray implementations.
//!
//! # Architecture (SPEC-FT-001 v3.0.0)
//!
//! ```text
//! entrenar autograd
//!     └── cuda_forward (this module)
//!             └── trueno-gpu/kernels
//!                     └── PTX generation + CUDA driver
//! ```
//!
//! # Available Kernels
//!
//! - `relu_forward` - ReLU activation
//! - `softmax_forward` - Numerically stable softmax with warp shuffle
//! - `batched_softmax_forward` - Row-wise batched softmax for attention
//! - `layer_norm_forward` - Fused layer normalization
//! - `rms_norm_forward` - RMS normalization (LLaMA-style)
//! - `gelu_forward` - GELU activation
//! - `silu_forward` - SiLU/Swish activation
//! - `gemm_forward` - Matrix multiplication
//! - `batched_4d_gemm_forward` - 4D batched GEMM for multi-head attention
//! - `residual_add_forward` - Element-wise addition (residual connections)
//! - `elementwise_mul_forward` - Element-wise multiplication
//! - `scale_forward` - Scalar multiplication

mod activations;
mod cache;
mod elementwise;
mod matmul;
mod normalization;
#[cfg(test)]
mod tests;

pub use activations::{
    batched_softmax_forward, gelu_forward, relu_forward, silu_forward, softmax_forward,
};
pub use cache::init_forward_kernel_cache;
pub use elementwise::{
    batched_to_interleaved_forward, batched_transpose_forward, elementwise_mul_forward,
    expand_kv_heads, interleaved_to_batched_forward, residual_add_forward, scale_forward,
};
pub use matmul::{batched_4d_gemm_forward, fused_swiglu_forward, gemm_forward};
pub use normalization::{layer_norm_forward, rms_norm_forward};
