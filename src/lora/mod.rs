//! LoRA (Low-Rank Adaptation) implementation
//!
//! LoRA enables parameter-efficient fine-tuning of large pretrained models
//! by adding trainable low-rank decomposition matrices to frozen weights.

mod adapter;
mod config;
mod layer;
mod qlora;

#[cfg(test)]
mod gradient_tests;

pub use adapter::{load_adapter, save_adapter, AdapterError, AdapterMetadata, LoRAAdapter};
pub use config::LoRAConfig;
pub use layer::LoRALayer;
pub use qlora::{MemoryStats, QLoRALayer};
