//! High-level training loop
//!
//! This module provides a complete training framework with:
//! - Loss functions (MSE, Cross-Entropy, Huber/SmoothL1, L1)
//! - Evaluation metrics (Accuracy, Precision, Recall, F1, R², MAE, RMSE)
//! - Curriculum learning (Linear, Tiered, Adaptive)
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
mod curriculum;
mod loss;
mod metrics;
mod trainer;
mod transformer_trainer;
pub mod tui;

#[cfg(test)]
mod tests;

pub use batch::Batch;
pub use callback::{
    CallbackAction, CallbackContext, CallbackManager, CheckpointCallback, EarlyStopping,
    ExplainMethod, ExplainabilityCallback, FeatureImportanceResult, LRSchedulerCallback,
    MonitorCallback, ProgressCallback, TrainerCallback,
};
pub use config::{MetricsTracker, TrainConfig};
pub use curriculum::{
    efficiency_score, select_optimal_tier, AdaptiveCurriculum, CurriculumScheduler,
    LinearCurriculum, TieredCurriculum,
};
pub use loss::{
    BCEWithLogitsLoss, CausalLMLoss, CrossEntropyLoss, HuberLoss, L1Loss, LossFn, MSELoss,
    SampleWeightedLoss, SmoothL1Loss, WeightedLoss,
};
pub use metrics::{Accuracy, F1Score, Metric, Precision, R2Score, Recall, MAE, RMSE};
pub use trainer::{TrainResult, Trainer};
pub use transformer_trainer::{
    perplexity, tokens_per_second, LMBatch, TransformerTrainConfig, TransformerTrainer,
};
pub use tui::{
    format_duration, sparkline, sparkline_range, Alert, AlertLevel, AndonSystem, DashboardLayout,
    FeatureImportanceChart, GradientFlowHeatmap, KalmanEta, LossCurveDisplay, MetricsBuffer,
    MonitorConfig, ProgressBar, ReferenceCurve, RefreshPolicy, SeriesSummaryTuple,
    TerminalCapabilities, TerminalMode, TerminalMonitorCallback, SPARK_CHARS,
};
