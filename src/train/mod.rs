//! High-level training loop
//!
//! This module provides a complete training framework with:
//! - Loss functions (MSE, Cross-Entropy, Huber/SmoothL1, L1)
//! - Evaluation metrics (Accuracy, Precision, Recall, F1, R², MAE, RMSE)
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

mod batch;
pub mod callback;
mod config;
mod loss;
mod metrics;
mod trainer;

#[cfg(test)]
mod tests;

pub use batch::Batch;
pub use callback::{
    CallbackAction, CallbackContext, CallbackManager, CheckpointCallback, EarlyStopping,
    LRSchedulerCallback, MonitorCallback, ProgressCallback, TrainerCallback,
};
pub use config::{MetricsTracker, TrainConfig};
pub use loss::{CrossEntropyLoss, HuberLoss, L1Loss, LossFn, MSELoss, SmoothL1Loss};
pub use metrics::{Accuracy, F1Score, Metric, Precision, R2Score, Recall, MAE, RMSE};
pub use trainer::{TrainResult, Trainer};
