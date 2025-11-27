//! # Entrenar: Training & Optimization Library
//!
//! Entrenar provides a tape-based autograd engine with optimizers, LoRA/QLoRA,
//! quantization (QAT/PTQ), model merging (TIES/DARE/SLERP), and knowledge distillation.
//!
//! ## Architecture
//!
//! - **autograd**: Tape-based automatic differentiation
//! - **optim**: Optimizers (SGD, Adam, AdamW)
//! - **lora**: Low-rank adaptation with QLoRA support
//! - **quant**: Quantization-aware training and post-training quantization
//! - **merge**: Model merging methods
//! - **distill**: Knowledge distillation
//! - **config**: Declarative YAML configuration
//! - **train**: High-level training loop
//! - **io**: Model saving and loading (JSON, YAML formats)

pub mod autograd;
pub mod config;
pub mod distill;
pub mod io;
pub mod lora;
pub mod merge;
pub mod monitor;
pub mod optim;
pub mod quant;
pub mod train;

pub mod error;

// Re-export commonly used types
pub use autograd::{backward, Context, Tensor};
pub use error::{Error, Result};
