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
//! - **hf_pipeline**: HuggingFace model fetching and distillation
//! - **citl**: Compiler-in-the-Loop training with RAG-based fix suggestions (feature-gated)
//! - **efficiency**: Cost tracking, device detection, and performance benchmarking
//! - **sovereign**: Air-gapped deployment and distribution packaging
//! - **research**: Academic research artifacts, citations, and archive deposits
//! - **ecosystem**: PAIML stack integrations (Batuta, Realizar, Ruchy)
//! - **dashboard**: Real-time training monitoring and WASM bindings
//! - **yaml_mode**: Declarative YAML Mode Training (v1.0 spec)

pub mod autograd;
#[cfg(feature = "citl")]
pub mod citl;
pub mod config;
pub mod dashboard;
pub mod distill;
pub mod ecosystem;
pub mod efficiency;
pub mod hf_pipeline;
pub mod integrity;
pub mod io;
pub mod lora;
pub mod merge;
pub mod monitor;
pub mod optim;
pub mod quality;
pub mod quant;
pub mod research;
pub mod run;
pub mod sovereign;
pub mod storage;
pub mod train;
pub mod yaml_mode;

pub mod error;

// Re-export commonly used types
pub use autograd::{backward, Context, Tensor};
pub use error::{Error, Result};
