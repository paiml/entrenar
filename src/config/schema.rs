//! YAML schema definitions for declarative training configuration
//!
//! ENT-114: Added `ModelMode` and `TrainingMode` for LLM training support.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Deserialize a bool from either a YAML boolean (`true`) or a quoted string (`"true"`).
/// This supports CB-950 compliance where all truthy values must be quoted in YAML.
fn deserialize_bool_lenient<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BoolOrString {
        Bool(bool),
        Str(String),
    }

    match BoolOrString::deserialize(deserializer)? {
        BoolOrString::Bool(b) => Ok(b),
        BoolOrString::Str(s) => match s.to_lowercase().as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            other => Err(serde::de::Error::custom(format!(
                "expected 'true' or 'false', got '{other}'"
            ))),
        },
    }
}

/// Model execution mode
///
/// Determines whether to use generic tabular ML training or transformer-based LLM training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelMode {
    /// Generic tabular ML (uses Trainer + MSELoss)
    #[default]
    Tabular,
    /// Transformer-based LLM (uses TransformerTrainer + CausalLMLoss)
    Transformer,
}

/// Training loss mode
///
/// Determines which loss function to use during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    /// Mean squared error loss (regression)
    #[default]
    Regression,
    /// Cross-entropy loss for next-token prediction (language modeling)
    CausalLm,
}

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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelRef {
    /// Path to base model (GGUF, safetensors, etc.)
    #[serde(default)]
    pub path: PathBuf,

    /// Target layers for LoRA (if applicable)
    #[serde(default)]
    pub layers: Vec<String>,

    /// Model execution mode (tabular or transformer)
    /// ENT-114: Routes to TransformerTrainer when mode=transformer
    #[serde(default)]
    pub mode: ModelMode,

    /// Transformer architecture config preset (e.g., "qwen2_1_5b", "llama2_7b")
    /// Only used when mode=transformer
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<String>,
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Training data path
    #[serde(default)]
    pub train: PathBuf,

    /// Optional validation data path
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val: Option<PathBuf>,

    /// Batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Auto-infer feature types from data
    #[serde(
        default = "default_true",
        deserialize_with = "deserialize_bool_lenient"
    )]
    pub auto_infer_types: bool,

    /// Sequence length (for transformers)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seq_len: Option<usize>,

    // === ENT-114: LLM training fields ===
    /// Path to HuggingFace tokenizer.json (for transformer mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<PathBuf>,

    /// Input text column name (for transformer mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_column: Option<String>,

    /// Output/target text column name (for transformer mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_column: Option<String>,

    /// Maximum sequence length for tokenization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train: PathBuf::new(),
            val: None,
            batch_size: 8,
            auto_infer_types: true,
            seq_len: None,
            tokenizer: None,
            input_column: None,
            output_column: None,
            max_length: None,
        }
    }
}

fn default_batch_size() -> usize {
    8
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
    #[serde(
        default = "default_true",
        deserialize_with = "deserialize_bool_lenient"
    )]
    pub symmetric: bool,

    /// Per-channel quantization
    #[serde(
        default = "default_true",
        deserialize_with = "deserialize_bool_lenient"
    )]
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
#[serde(default)]
pub struct TrainingParams {
    /// Number of epochs
    pub epochs: usize,

    /// Gradient clipping threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grad_clip: Option<f32>,

    /// Learning rate scheduler
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_scheduler: Option<String>,

    /// Warmup steps
    pub warmup_steps: usize,

    /// Save checkpoint every N epochs
    pub save_interval: usize,

    /// Output directory for checkpoints
    pub output_dir: PathBuf,

    // === ENT-114: LLM training fields ===
    /// Training mode (regression or causal_lm)
    /// ENT-114: Uses CausalLMLoss when mode=causal_lm
    pub mode: TrainingMode,

    /// Gradient accumulation steps (for large batch simulation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_accumulation: Option<usize>,

    /// Number of gradient checkpoints (for memory optimization)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoints: Option<usize>,

    /// Use mixed precision training (bf16 or fp16)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mixed_precision: Option<String>,

    /// Scheduler-specific parameters (t_max, gamma, step_size, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler_params: Option<HashMap<String, serde_json::Value>>,

    /// Global random seed for reproducibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            epochs: 10,
            grad_clip: None,
            lr_scheduler: None,
            warmup_steps: 0,
            save_interval: 1,
            output_dir: PathBuf::from("./checkpoints"),
            mode: TrainingMode::default(),
            gradient_accumulation: None,
            checkpoints: None,
            mixed_precision: None,
            scheduler_params: None,
            seed: None,
        }
    }
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_minimal_config() {
        let yaml = r"
model:
  path: model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001
";

        let spec: TrainSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.model.path, PathBuf::from("model.gguf"));
        assert_eq!(spec.data.batch_size, 8);
        assert_eq!(spec.optimizer.name, "adam");
        assert_eq!(spec.optimizer.lr, 0.001);
    }

    #[test]
    fn test_deserialize_full_config() {
        let yaml = r"
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
";

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

    // === ENT-114 Tests: LLM Training Schema ===

    #[test]
    fn test_model_mode_default_is_tabular() {
        let mode = ModelMode::default();
        assert_eq!(mode, ModelMode::Tabular);
    }

    #[test]
    fn test_training_mode_default_is_regression() {
        let mode = TrainingMode::default();
        assert_eq!(mode, TrainingMode::Regression);
    }

    #[test]
    fn test_model_mode_serde_roundtrip() {
        // Tabular mode
        let yaml = "tabular";
        let mode: ModelMode = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(mode, ModelMode::Tabular);

        // Transformer mode
        let yaml = "transformer";
        let mode: ModelMode = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(mode, ModelMode::Transformer);
    }

    #[test]
    fn test_training_mode_serde_roundtrip() {
        // Regression mode
        let yaml = "regression";
        let mode: TrainingMode = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(mode, TrainingMode::Regression);

        // CausalLM mode
        let yaml = "causal_lm";
        let mode: TrainingMode = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(mode, TrainingMode::CausalLm);
    }

    #[test]
    fn test_deserialize_transformer_config() {
        let yaml = r"
model:
  path: qwen2.5-coder-1.5b.safetensors
  mode: transformer
  config: qwen2_1_5b
  layers: [q_proj, v_proj]

data:
  train: corpus/train.parquet
  batch_size: 4
  tokenizer: tokenizer.json
  input_column: input
  output_column: output
  max_length: 512

optimizer:
  name: adamw
  lr: 0.0001

training:
  epochs: 3
  mode: causal_lm
  gradient_accumulation: 4
  checkpoints: 6
  mixed_precision: bf16
";

        let spec: TrainSpec = serde_yaml::from_str(yaml).unwrap();

        // Model assertions
        assert_eq!(spec.model.mode, ModelMode::Transformer);
        assert_eq!(spec.model.config, Some("qwen2_1_5b".to_string()));

        // Data assertions
        assert_eq!(spec.data.tokenizer, Some(PathBuf::from("tokenizer.json")));
        assert_eq!(spec.data.input_column, Some("input".to_string()));
        assert_eq!(spec.data.output_column, Some("output".to_string()));
        assert_eq!(spec.data.max_length, Some(512));

        // Training assertions
        assert_eq!(spec.training.mode, TrainingMode::CausalLm);
        assert_eq!(spec.training.gradient_accumulation, Some(4));
        assert_eq!(spec.training.checkpoints, Some(6));
        assert_eq!(spec.training.mixed_precision, Some("bf16".to_string()));
    }

    #[test]
    fn test_backward_compatible_minimal_config() {
        // Ensure old configs still work (defaults to tabular/regression)
        let yaml = r"
model:
  path: model.gguf

data:
  train: data.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001
";

        let spec: TrainSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.model.mode, ModelMode::Tabular);
        assert_eq!(spec.training.mode, TrainingMode::Regression);
        assert!(spec.data.tokenizer.is_none());
    }

    #[test]
    fn test_training_params_new_fields_default() {
        let params = TrainingParams::default();
        assert_eq!(params.mode, TrainingMode::Regression);
        assert!(params.gradient_accumulation.is_none());
        assert!(params.checkpoints.is_none());
        assert!(params.mixed_precision.is_none());
        assert!(params.scheduler_params.is_none());
        assert!(params.seed.is_none());
    }

    #[test]
    fn test_deserialize_scheduler_params_and_seed() {
        let yaml = r"
model:
  path: model.gguf

data:
  train: data.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001

training:
  epochs: 5
  seed: 42
  lr_scheduler: cosine
  scheduler_params:
    t_max: 1000
    eta_min: 0.000001
";

        let spec: TrainSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.training.seed, Some(42));
        let params = spec.training.scheduler_params.unwrap();
        assert_eq!(params["t_max"], serde_json::json!(1000));
        assert_eq!(params["eta_min"], serde_json::json!(0.000001));
    }

    /// CB-950: Verify that quoted boolean strings ("true"/"false") deserialize correctly.
    /// PMAT compliance requires all YAML truthy values to be quoted.
    #[test]
    fn test_cb950_quoted_booleans_deserialize() {
        let yaml = r#"
model:
  path: model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8
  auto_infer_types: "true"

optimizer:
  name: adam
  lr: 0.001

quantize:
  bits: 4
  symmetric: "true"
  per_channel: "false"
"#;

        let spec: TrainSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.data.auto_infer_types);
        let quant = spec.quantize.unwrap();
        assert!(quant.symmetric);
        assert!(!quant.per_channel);
    }
}
