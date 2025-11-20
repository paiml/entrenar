//! LoRA (Low-Rank Adaptation) implementation
//!
//! LoRA enables parameter-efficient fine-tuning of large pretrained models
//! by adding trainable low-rank decomposition matrices to frozen weights.

mod layer;

pub use layer::LoRALayer;
