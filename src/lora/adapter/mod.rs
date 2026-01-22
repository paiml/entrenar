//! LoRA adapter serialization and deserialization
//!
//! Enables saving and loading LoRA adapters independently of base models.
//! This allows:
//! - Training adapters and saving them separately
//! - Loading adapters into frozen base models for inference
//! - Sharing adapters without sharing full model weights
//! - Switching between multiple adapters for the same base model

mod error;
mod io;
mod lora_adapter;
mod metadata;

pub use error::AdapterError;
pub use io::{load_adapter, save_adapter};
pub use lora_adapter::LoRAAdapter;
pub use metadata::AdapterMetadata;

#[cfg(test)]
mod tests;
