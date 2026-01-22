//! Training hyperparameters configuration

use serde::{Deserialize, Serialize};

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of epochs
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    /// Batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Learning rate
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    /// Weight decay
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    /// Warmup ratio
    #[serde(default = "default_warmup_ratio")]
    pub warmup_ratio: f32,
    /// Gradient accumulation steps
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,
    /// Maximum gradient norm
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f32,
    /// Enable gradient checkpointing
    #[serde(default)]
    pub gradient_checkpointing: bool,
    /// Mixed precision mode
    pub mixed_precision: Option<String>,
    /// Random seed
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_epochs() -> usize {
    3
}
fn default_batch_size() -> usize {
    16
}
fn default_learning_rate() -> f64 {
    2e-4
}
fn default_weight_decay() -> f64 {
    0.01
}
fn default_warmup_ratio() -> f32 {
    0.03
}
fn default_grad_accum() -> usize {
    1
}
fn default_max_grad_norm() -> f32 {
    1.0
}
fn default_seed() -> u64 {
    42
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 16,
            learning_rate: 2e-4,
            weight_decay: 0.01,
            warmup_ratio: 0.03,
            gradient_accumulation_steps: 1,
            max_grad_norm: 1.0,
            gradient_checkpointing: false,
            mixed_precision: None,
            seed: 42,
        }
    }
}
