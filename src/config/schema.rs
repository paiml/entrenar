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
            other => {
                Err(serde::de::Error::custom(format!("expected 'true' or 'false', got '{other}'")))
            }
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

    /// Optional auto-publish after training completes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publish: Option<PublishSpec>,
}

/// Auto-publish configuration for uploading to HuggingFace Hub after training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishSpec {
    /// HuggingFace repo ID (e.g., "myuser/my-model")
    pub repo: String,

    /// Make the repository private
    #[serde(default)]
    pub private: bool,

    /// Generate and upload a model card
    #[serde(default = "default_true")]
    pub model_card: bool,

    /// Merge LoRA adapters before publishing
    #[serde(default)]
    pub merge_adapters: bool,

    /// Export format (safetensors or gguf)
    #[serde(default = "default_safetensors")]
    pub format: String,
}

fn default_safetensors() -> String {
    "safetensors".to_string()
}

/// Architecture override parameters from YAML manifest.
///
/// These override individual fields of the `TransformerConfig` resolved from
/// `config.json` or preset defaults. Only `Some` fields apply; `None` fields
/// are left as-is from the base config.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchitectureOverrides {
    /// Hidden size (embedding dimension)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<usize>,
    /// Number of transformer layers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_hidden_layers: Option<usize>,
    /// Number of attention heads
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_attention_heads: Option<usize>,
    /// Number of key-value heads (for grouped-query attention)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_heads: Option<usize>,
    /// FFN intermediate dimension
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intermediate_size: Option<usize>,
    /// Vocabulary size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,
    /// Maximum sequence/position length
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_position_embeddings: Option<usize>,
    /// RMS normalization epsilon
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rms_norm_eps: Option<f32>,
    /// RoPE theta (rotary positional encoding base)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f32>,
    /// Whether to use bias in linear layers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_bias: Option<bool>,
}

impl ArchitectureOverrides {
    /// Returns true if no overrides are set.
    pub fn is_empty(&self) -> bool {
        self.hidden_size.is_none()
            && self.num_hidden_layers.is_none()
            && self.num_attention_heads.is_none()
            && self.num_kv_heads.is_none()
            && self.intermediate_size.is_none()
            && self.vocab_size.is_none()
            && self.max_position_embeddings.is_none()
            && self.rms_norm_eps.is_none()
            && self.rope_theta.is_none()
            && self.use_bias.is_none()
    }
}

/// Model reference and target layers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelRef {
    /// Path to base model — local path (GGUF, safetensors, etc.) or HuggingFace repo ID
    /// (e.g., "Qwen/Qwen2.5-Coder-0.5B"). HF repo IDs are auto-detected and downloaded.
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

    /// Architecture parameter overrides from YAML manifest.
    /// Applied on top of config.json / preset values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<ArchitectureOverrides>,
}

impl ModelRef {
    /// Check if the model path looks like a HuggingFace repo ID (e.g., "org/model-name").
    ///
    /// Detection: contains exactly one `/`, no file extension, and both parts are non-empty.
    pub fn is_hf_repo_id(&self) -> bool {
        let s = self.path.to_string_lossy();
        is_hf_repo_id(&s)
    }
}

/// Check if a string looks like a HuggingFace repo ID.
///
/// Returns true if the string has the format "org/name" where:
/// - There is exactly one `/`
/// - Both parts are non-empty
/// - The name doesn't end with a known model file extension
/// - The string doesn't start with `.` or `/` (not a filesystem path)
pub fn is_hf_repo_id(s: &str) -> bool {
    // Must not start with `.` or `/` (those are filesystem paths)
    if s.starts_with('.') || s.starts_with('/') {
        return false;
    }

    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 2 {
        return false;
    }

    let (org, name) = (parts[0], parts[1]);

    // Both parts must be non-empty
    if org.is_empty() || name.is_empty() {
        return false;
    }

    // Reject if name ends with a known model file extension
    let file_extensions = [
        ".safetensors",
        ".gguf",
        ".bin",
        ".pt",
        ".pth",
        ".onnx",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
    ];
    let name_lower = name.to_lowercase();
    !file_extensions.iter().any(|ext| name_lower.ends_with(ext))
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
    #[serde(default = "default_true", deserialize_with = "deserialize_bool_lenient")]
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
    #[serde(default = "default_true", deserialize_with = "deserialize_bool_lenient")]
    pub symmetric: bool,

    /// Per-channel quantization
    #[serde(default = "default_true", deserialize_with = "deserialize_bool_lenient")]
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

    /// Maximum training steps (overrides epochs if set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,

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
            max_steps: None,
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

        let spec: TrainSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
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

        let spec: TrainSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
        assert_eq!(spec.model.layers.len(), 4);
        assert!(spec.lora.is_some());
        assert_eq!(spec.lora.as_ref().expect("operation should succeed").rank, 64);
        assert!(spec.quantize.is_some());
        assert_eq!(spec.quantize.as_ref().expect("operation should succeed").bits, 4);
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
        let mode: ModelMode = serde_yaml::from_str(yaml).expect("operation should succeed");
        assert_eq!(mode, ModelMode::Tabular);

        // Transformer mode
        let yaml = "transformer";
        let mode: ModelMode = serde_yaml::from_str(yaml).expect("operation should succeed");
        assert_eq!(mode, ModelMode::Transformer);
    }

    #[test]
    fn test_training_mode_serde_roundtrip() {
        // Regression mode
        let yaml = "regression";
        let mode: TrainingMode = serde_yaml::from_str(yaml).expect("operation should succeed");
        assert_eq!(mode, TrainingMode::Regression);

        // CausalLM mode
        let yaml = "causal_lm";
        let mode: TrainingMode = serde_yaml::from_str(yaml).expect("operation should succeed");
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

        let spec: TrainSpec = serde_yaml::from_str(yaml).expect("operation should succeed");

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

        let spec: TrainSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
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

        let spec: TrainSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
        assert_eq!(spec.training.seed, Some(42));
        let params = spec.training.scheduler_params.expect("operation should succeed");
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

        let spec: TrainSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
        assert!(spec.data.auto_infer_types);
        let quant = spec.quantize.expect("operation should succeed");
        assert!(quant.symmetric);
        assert!(!quant.per_channel);
    }

    // === HF Repo ID Detection Tests ===

    #[test]
    fn test_is_hf_repo_id_valid() {
        assert!(is_hf_repo_id("Qwen/Qwen2.5-Coder-0.5B"));
        assert!(is_hf_repo_id("meta-llama/Llama-2-7b"));
        assert!(is_hf_repo_id("google/gemma-2b"));
        assert!(is_hf_repo_id("myuser/my-model"));
    }

    #[test]
    fn test_is_hf_repo_id_local_paths() {
        assert!(!is_hf_repo_id("model.gguf"));
        assert!(!is_hf_repo_id("./models/model.safetensors"));
        assert!(!is_hf_repo_id("/absolute/path/model.bin"));
        assert!(!is_hf_repo_id("relative/path/model.gguf"));
    }

    #[test]
    fn test_is_hf_repo_id_edge_cases() {
        assert!(!is_hf_repo_id(""));
        assert!(!is_hf_repo_id("/"));
        assert!(!is_hf_repo_id("single-part"));
        assert!(!is_hf_repo_id("too/many/parts"));
        assert!(!is_hf_repo_id(".hidden/path"));
        assert!(!is_hf_repo_id("/org/name"));
        assert!(!is_hf_repo_id("org/"));
        assert!(!is_hf_repo_id("/name"));
    }

    #[test]
    fn test_is_hf_repo_id_with_extension_rejected() {
        // Files with extensions are local paths, not HF IDs
        assert!(!is_hf_repo_id("org/model.safetensors"));
        assert!(!is_hf_repo_id("user/model.gguf"));
    }

    #[test]
    fn test_model_ref_is_hf_repo_id() {
        let model =
            ModelRef { path: PathBuf::from("Qwen/Qwen2.5-Coder-0.5B"), ..Default::default() };
        assert!(model.is_hf_repo_id());

        let model = ModelRef { path: PathBuf::from("model.gguf"), ..Default::default() };
        assert!(!model.is_hf_repo_id());
    }

    #[test]
    fn test_deserialize_hf_repo_id_as_model_path() {
        let yaml = r"
model:
  path: Qwen/Qwen2.5-Coder-0.5B
  mode: transformer

data:
  train: data.parquet
  batch_size: 8

optimizer:
  name: adamw
  lr: 0.0001
";
        let spec: TrainSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
        assert!(spec.model.is_hf_repo_id());
        assert_eq!(spec.model.path, PathBuf::from("Qwen/Qwen2.5-Coder-0.5B"));
    }

    // === Publish Section Tests ===

    #[test]
    fn test_deserialize_with_publish_section() {
        let yaml = r"
model:
  path: model.gguf

data:
  train: data.parquet
  batch_size: 8

optimizer:
  name: adamw
  lr: 0.0001

publish:
  repo: myuser/my-model
  private: false
  model_card: true
  merge_adapters: true
  format: safetensors
";
        let spec: TrainSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
        let publish = spec.publish.expect("operation should succeed");
        assert_eq!(publish.repo, "myuser/my-model");
        assert!(!publish.private);
        assert!(publish.model_card);
        assert!(publish.merge_adapters);
        assert_eq!(publish.format, "safetensors");
    }

    #[test]
    fn test_deserialize_without_publish_section() {
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
        let spec: TrainSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
        assert!(spec.publish.is_none());
    }

    #[test]
    fn test_publish_spec_defaults() {
        let yaml = r"
model:
  path: model.gguf

data:
  train: data.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001

publish:
  repo: org/model
";
        let spec: TrainSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
        let publish = spec.publish.expect("operation should succeed");
        assert_eq!(publish.repo, "org/model");
        assert!(!publish.private);
        assert!(publish.model_card);
        assert!(!publish.merge_adapters);
        assert_eq!(publish.format, "safetensors");
    }
}
