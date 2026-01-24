//! Transformer layers with automatic differentiation support
//!
//! This module provides transformer building blocks that work with entrenar's
//! tape-based autograd engine. All operations support gradient computation.
//!
//! ## Architecture Components
//!
//! - `MultiHeadAttention`: Multi-head self-attention mechanism
//! - `FeedForward`: Position-wise feed-forward network (MLP)
//! - `TransformerBlock`: Complete transformer block (attention + FFN + residuals)
//! - `TransformerConfig`: Configuration for transformer models
//! - `RMSNorm`: RMS normalization layer
//! - `Embedding`: Token embedding layer
//! - `Transformer`: Complete transformer model
//!
//! ## Example
//!
//! ```ignore
//! use entrenar::transformer::{TransformerConfig, TransformerBlock};
//!
//! let config = TransformerConfig::llama2_7b();
//! let block = TransformerBlock::new(&config, 0);
//! let output = block.forward(&input);
//! ```

mod attention;
mod block;
mod config;
mod embedding;
mod feedforward;
mod model;
mod norm;
mod weights;

pub use attention::MultiHeadAttention;
pub use block::TransformerBlock;
pub use config::TransformerConfig;
pub use embedding::Embedding;
pub use feedforward::FeedForward;
pub use model::Transformer;
pub use norm::RMSNorm;
pub use weights::{load_safetensors_weights, validate_weights, Architecture};
