//! Transformer module with full model implementation and weight loading
//!
//! This module provides:
//! - `Transformer` - Complete transformer model for language modeling
//! - `TransformerConfig` - Model architecture configuration
//! - `load_safetensors_weights` - Load weights from SafeTensors files
//! - `Architecture` - Model architecture type for weight mapping
//! - `MultiHeadAttentionWithLoRA` - Attention with deep LoRA injection
//! - `LoRAProjection` - Linear projection with LoRA adapters

mod attention;
mod block;
mod config;
mod embedding;
mod feedforward;
mod model;
mod norm;
mod weights;

pub use attention::{LoRAProjection, MultiHeadAttention, MultiHeadAttentionWithLoRA};
pub use config::TransformerConfig;
pub use model::Transformer;
pub use weights::{load_safetensors_weights, validate_weights, Architecture};
