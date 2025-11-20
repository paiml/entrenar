//! YAML schema definitions for declarative training configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Complete training specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainSpec {
    /// Model configuration
    pub model: ModelRef,

    /// Data configuration
    pub data: DataConfig,

    /// Optimizer configuration
    pub optimizer: OptimSpec,

    /// Optional LoRA configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoRASpec>,

    /// Optional quantization configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantize: Option<QuantSpec>,

    /// Optional model merging configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge: Option<MergeSpec>,

    /// Training hyperparameters
    #[serde(default)]
    pub training: TrainingParams,
}

/// Model reference and target layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    /// Path to base model (GGUF, safetensors, etc.)
    pub path: PathBuf,

    /// Target layers for LoRA (if applicable)
    #[serde(default)]
    pub layers: Vec<String>,
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Training data path
    pub train: PathBuf,

    /// Optional validation data path
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val: Option<PathBuf>,

    /// Batch size
    pub batch_size: usize,

    /// Auto-infer feature types from data
    #[serde(default = "default_true")]
    pub auto_infer_types: bool,

    /// Sequence length (for transformers)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seq_len: Option<usize>,
}

/// Optimizer specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimSpec {
    /// Optimizer name: "adam" | "adamw" | "sgd"
    pub name: String,

    /// Learning rate
    pub lr: f32,

    /// Optimizer-specific parameters (beta1, beta2, momentum, etc.)
    #[serde(flatten)]
    pub params: HashMap<String, serde_json::Value>,
}

/// LoRA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRASpec {
    /// Rank of low-rank decomposition
    pub rank: usize,

    /// Scaling factor (alpha)
    pub alpha: f32,

    /// Target modules (e.g., [q_proj, v_proj])
    pub target_modules: Vec<String>,

    /// Dropout probability
    #[serde(default)]
    pub dropout: f32,
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantSpec {
    /// Quantization bits (4 or 8)
    pub bits: u8,

    /// Symmetric quantization
    #[serde(default = "default_true")]
    pub symmetric: bool,

    /// Per-channel quantization
    #[serde(default = "default_true")]
    pub per_channel: bool,
}

/// Model merging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeSpec {
    /// Merge method: "ties" | "dare" | "slerp"
    pub method: String,

    /// Method-specific parameters
    #[serde(flatten)]
    pub params: HashMap<String, serde_json::Value>,
}

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Number of epochs
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    /// Gradient clipping threshold
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_clip: Option<f32>,

    /// Learning rate scheduler
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr_scheduler: Option<String>,

    /// Warmup steps
    #[serde(default)]
    pub warmup_steps: usize,

    /// Save checkpoint every N epochs
    #[serde(default = "default_save_interval")]
    pub save_interval: usize,

    /// Output directory for checkpoints
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            grad_clip: None,
            lr_scheduler: None,
            warmup_steps: 0,
            save_interval: default_save_interval(),
            output_dir: default_output_dir(),
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_epochs() -> usize {
    10
}

fn default_save_interval() -> usize {
    1
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("./checkpoints")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_minimal_config() {
        let yaml = r#"
model:
  path: model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001
"#;

        let spec: TrainSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.model.path, PathBuf::from("model.gguf"));
        assert_eq!(spec.data.batch_size, 8);
        assert_eq!(spec.optimizer.name, "adam");
        assert_eq!(spec.optimizer.lr, 0.001);
    }

    #[test]
    fn test_deserialize_full_config() {
        let yaml = r#"
model:
  path: llama-7b.gguf
  layers: [q_proj, k_proj, v_proj, o_proj]

data:
  train: train.parquet
  val: val.parquet
  batch_size: 32
  auto_infer_types: true
  seq_len: 2048

optimizer:
  name: adamw
  lr: 0.0001
  beta1: 0.9
  beta2: 0.999
  weight_decay: 0.01

lora:
  rank: 64
  alpha: 16
  target_modules: [q_proj, v_proj]
  dropout: 0.1

quantize:
  bits: 4
  symmetric: true
  per_channel: true

training:
  epochs: 3
  grad_clip: 1.0
  lr_scheduler: cosine
  warmup_steps: 100
  save_interval: 1
  output_dir: ./outputs
"#;

        let spec: TrainSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.model.layers.len(), 4);
        assert!(spec.lora.is_some());
        assert_eq!(spec.lora.as_ref().unwrap().rank, 64);
        assert!(spec.quantize.is_some());
        assert_eq!(spec.quantize.as_ref().unwrap().bits, 4);
        assert_eq!(spec.training.epochs, 3);
    }

    #[test]
    fn test_default_training_params() {
        let params = TrainingParams::default();
        assert_eq!(params.epochs, 10);
        assert_eq!(params.save_interval, 1);
        assert!(params.grad_clip.is_none());
    }
}
