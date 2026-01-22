//! Transformer-specific training utilities
//!
//! Provides specialized training components for transformer language models,
//! including tokenized batch creation and language modeling training loops.

mod batch;
mod config;
mod trainer;
mod utils;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use batch::LMBatch;
pub use config::TransformerTrainConfig;
pub use trainer::TransformerTrainer;
pub use utils::{perplexity, tokens_per_second};
