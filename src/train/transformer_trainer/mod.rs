//! Transformer-specific training utilities
//!
//! Provides specialized training components for transformer language models,
//! including tokenized batch creation and language modeling training loops.

mod batch;
mod config;
mod cuda_trainer;
pub mod distributed_checkpoint;
mod distributed_trainer;
pub mod grad_accumulator;
mod trainer;
mod utils;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use batch::LMBatch;
pub use config::{
    DistributedBackend, DistributedRole, DistributedTrainConfig, TransformerTrainConfig,
};
pub use grad_accumulator::{BlockGradientSet, PerBlockGradientAccumulator};
pub use distributed_checkpoint::DistributedCheckpointCoordinator;
#[cfg(feature = "cuda")]
pub use distributed_trainer::{DistributedCudaTrainer, DistributedComm, GradientMessage};
pub use cuda_trainer::CudaTransformerTrainer;
pub use trainer::TransformerTrainer;
pub use utils::{perplexity, tokens_per_second};
