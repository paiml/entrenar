//! High-level training loop
//!
//! This module provides a complete training framework with:
//! - Loss functions (MSE, Cross-Entropy)
//! - Trainer abstraction
//! - Training configuration
//! - Metrics tracking
//! - Checkpoint support
//!
//! # Example
//!
//! ```no_run
//! use entrenar::train::{Trainer, TrainConfig, Batch};
//! use entrenar::optim::Adam;
//! use entrenar::Tensor;
//!
//! let params = vec![Tensor::zeros(10, true)];
//! let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
//! let config = TrainConfig::default();
//!
//! let mut trainer = Trainer::new(params, Box::new(optimizer), config);
//!
//! // Training loop
//! // for epoch in 0..10 {
//! //     let loss = trainer.train_epoch(&dataloader);
//! //     println!("Epoch {}: loss={:.4}", epoch, loss);
//! // }
//! ```

mod loss;
mod trainer;
mod config;
mod batch;

#[cfg(test)]
mod tests;

pub use loss::{LossFn, MSELoss, CrossEntropyLoss};
pub use trainer::Trainer;
pub use config::{TrainConfig, MetricsTracker};
pub use batch::Batch;
