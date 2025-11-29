//! YAML Configuration for Distillation Training
//!
//! Provides declarative configuration for the entire distillation pipeline.
//!
//! # Example Config
//!
//! ```yaml
//! teacher:
//!   model_id: "microsoft/codebert-base"
//!
//! student:
//!   model_id: "distilbert-base-uncased"
//!   lora:
//!     rank: 16
//!     alpha: 32
//!     target_modules: ["q_proj", "v_proj"]
//!
//! distillation:
//!   temperature: 4.0
//!   alpha: 0.7
//!   progressive:
//!     layer_mapping: [[0, 3], [1, 7], [2, 11]]
//!
//! training:
//!   epochs: 3
//!   batch_size: 16
//!   learning_rate: 2.0e-4
//! ```

use crate::hf_pipeline::error::{FetchError, Result};
use crate::hf_pipeline::fine_tune::{FineTuneConfig, MixedPrecision};
use crate::hf_pipeline::trainer::TrainerConfig;
use crate::lora::LoRAConfig;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Teacher model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherConfig {
    /// Model ID on HuggingFace
    pub model_id: String,
    /// Revision/branch (default: "main")
    #[serde(default = "default_revision")]
    pub revision: String,
    /// Use 8-bit quantization for teacher
    #[serde(default)]
    pub load_in_8bit: bool,
}

fn default_revision() -> String {
    "main".to_string()
}

/// Student model configuration with LoRA/QLoRA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentConfig {
    /// Model ID (can be same as teacher or smaller model)
    pub model_id: String,
    /// Revision/branch
    #[serde(default = "default_revision")]
    pub revision: String,
    /// LoRA configuration (if None, full fine-tuning)
    pub lora: Option<LoRAYamlConfig>,
    /// Use 4-bit quantization (QLoRA)
    #[serde(default)]
    pub load_in_4bit: bool,
}

/// LoRA configuration in YAML format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAYamlConfig {
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha (scaling factor)
    pub alpha: f32,
    /// Target modules
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,
    /// Target layers (optional)
    pub layers: Option<Vec<usize>>,
}

fn default_target_modules() -> Vec<String> {
    vec!["q_proj".to_string(), "v_proj".to_string()]
}

impl From<&LoRAYamlConfig> for LoRAConfig {
    fn from(yaml: &LoRAYamlConfig) -> Self {
        let mut config = LoRAConfig::new(yaml.rank, yaml.alpha);
        let modules: Vec<&str> = yaml.target_modules.iter().map(String::as_str).collect();
        config = config.target_modules(&modules);
        if let Some(ref layers) = yaml.layers {
            config = config.target_layers(layers);
        }
        config
    }
}

/// Distillation loss configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Temperature for softening distributions
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Alpha weight for soft vs hard loss
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    /// Progressive distillation config
    pub progressive: Option<ProgressiveConfig>,
    /// Attention transfer config
    pub attention_transfer: Option<AttentionTransferConfig>,
}

fn default_temperature() -> f32 {
    4.0
}

fn default_alpha() -> f32 {
    0.7
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,
            progressive: None,
            attention_transfer: None,
        }
    }
}

/// Progressive distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Layer mapping [[student_layer, teacher_layer], ...]
    pub layer_mapping: Vec<[usize; 2]>,
    /// Weight for hidden state loss
    #[serde(default = "default_hidden_weight")]
    pub hidden_weight: f32,
}

fn default_hidden_weight() -> f32 {
    1.0
}

/// Attention transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionTransferConfig {
    /// Weight for attention transfer loss
    #[serde(default = "default_attention_weight")]
    pub weight: f32,
}

fn default_attention_weight() -> f32 {
    0.1
}

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

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset ID or path
    pub path: String,
    /// Maximum sequence length
    #[serde(default = "default_max_seq_length")]
    pub max_seq_length: usize,
    /// Maximum training examples (None = all)
    pub max_train_examples: Option<usize>,
    /// Maximum validation examples
    pub max_eval_examples: Option<usize>,
}

fn default_max_seq_length() -> usize {
    512
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory
    pub dir: String,
    /// Save checkpoints every N steps
    #[serde(default = "default_save_steps")]
    pub save_steps: usize,
    /// Evaluate every N steps
    #[serde(default = "default_eval_steps")]
    pub eval_steps: usize,
    /// Log every N steps
    #[serde(default = "default_log_steps")]
    pub log_steps: usize,
    /// Push to HuggingFace Hub
    #[serde(default)]
    pub push_to_hub: bool,
    /// Hub repository ID
    pub hub_repo_id: Option<String>,
}

fn default_save_steps() -> usize {
    500
}
fn default_eval_steps() -> usize {
    100
}
fn default_log_steps() -> usize {
    10
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: "./output".to_string(),
            save_steps: 500,
            eval_steps: 100,
            log_steps: 10,
            push_to_hub: false,
            hub_repo_id: None,
        }
    }
}

/// Complete distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationYamlConfig {
    /// Teacher model config
    pub teacher: TeacherConfig,
    /// Student model config
    pub student: StudentConfig,
    /// Distillation loss config
    #[serde(default)]
    pub distillation: DistillationConfig,
    /// Training hyperparameters
    #[serde(default)]
    pub training: TrainingConfig,
    /// Dataset config
    pub dataset: DatasetConfig,
    /// Output config
    #[serde(default)]
    pub output: OutputConfig,
}

impl DistillationYamlConfig {
    /// Load configuration from YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content =
            std::fs::read_to_string(path.as_ref()).map_err(|e| FetchError::ConfigParseError {
                message: format!("Failed to read config file: {e}"),
            })?;

        Self::from_yaml(&content)
    }

    /// Parse configuration from YAML string
    pub fn from_yaml(content: &str) -> Result<Self> {
        serde_yaml::from_str(content).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to parse YAML: {e}"),
        })
    }

    /// Save configuration to YAML file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = self.to_yaml()?;
        std::fs::write(path, content).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to write config file: {e}"),
        })
    }

    /// Serialize to YAML string
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to serialize YAML: {e}"),
        })
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate teacher
        if self.teacher.model_id.is_empty() {
            return Err(FetchError::ConfigParseError {
                message: "teacher.model_id cannot be empty".into(),
            });
        }

        // Validate student
        if self.student.model_id.is_empty() {
            return Err(FetchError::ConfigParseError {
                message: "student.model_id cannot be empty".into(),
            });
        }

        // Validate distillation
        if self.distillation.temperature <= 0.0 {
            return Err(FetchError::ConfigParseError {
                message: "distillation.temperature must be positive".into(),
            });
        }

        if !(0.0..=1.0).contains(&self.distillation.alpha) {
            return Err(FetchError::ConfigParseError {
                message: "distillation.alpha must be between 0 and 1".into(),
            });
        }

        // Validate training
        if self.training.batch_size == 0 {
            return Err(FetchError::ConfigParseError {
                message: "training.batch_size must be > 0".into(),
            });
        }

        if self.training.learning_rate <= 0.0 {
            return Err(FetchError::ConfigParseError {
                message: "training.learning_rate must be positive".into(),
            });
        }

        // Validate dataset
        if self.dataset.path.is_empty() {
            return Err(FetchError::ConfigParseError {
                message: "dataset.path cannot be empty".into(),
            });
        }

        Ok(())
    }

    /// Convert to TrainerConfig
    pub fn to_trainer_config(&self) -> Result<TrainerConfig> {
        self.validate()?;

        let mut config = TrainerConfig::new(&self.teacher.model_id, &self.student.model_id)
            .temperature(self.distillation.temperature)
            .alpha(self.distillation.alpha)
            .epochs(self.training.epochs)
            .output_dir(&self.output.dir);

        // Add progressive distillation
        if let Some(ref prog) = self.distillation.progressive {
            let mapping: Vec<(usize, usize)> =
                prog.layer_mapping.iter().map(|[s, t]| (*s, *t)).collect();
            config = config.with_progressive(mapping);
        }

        // Add attention transfer
        if let Some(ref at) = self.distillation.attention_transfer {
            config = config.with_attention_transfer(at.weight);
        }

        // Set up fine-tuning config
        let mut fine_tune = FineTuneConfig::new(&self.student.model_id)
            .learning_rate(self.training.learning_rate)
            .epochs(self.training.epochs)
            .batch_size(self.training.batch_size);

        // Set LoRA if configured
        if let Some(ref lora_yaml) = self.student.lora {
            let lora_config = LoRAConfig::from(lora_yaml);
            if self.student.load_in_4bit {
                fine_tune = fine_tune.with_qlora(lora_config, 4);
            } else {
                fine_tune = fine_tune.with_lora(lora_config);
            }
        } else if !self.student.load_in_4bit {
            fine_tune = fine_tune.full_fine_tune();
        }

        // Set mixed precision
        if let Some(ref mp) = self.training.mixed_precision {
            fine_tune = fine_tune.mixed_precision(match mp.as_str() {
                "fp16" => Some(MixedPrecision::Fp16),
                "bf16" => Some(MixedPrecision::Bf16),
                _ => None,
            });
        }

        fine_tune = fine_tune.gradient_checkpointing(self.training.gradient_checkpointing);

        config.fine_tune = fine_tune;
        config.max_grad_norm = self.training.max_grad_norm;
        config.seed = self.training.seed;
        config.log_every_n_steps = self.output.log_steps;
        config.save_every_n_steps = self.output.save_steps;
        config.eval_every_n_steps = self.output.eval_steps;

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_CONFIG: &str = r#"
teacher:
  model_id: "microsoft/codebert-base"
student:
  model_id: "distilbert-base-uncased"
dataset:
  path: "wikitext"
"#;

    const FULL_CONFIG: &str = r#"
teacher:
  model_id: "microsoft/codebert-base"
  revision: "main"
  load_in_8bit: false

student:
  model_id: "distilbert-base-uncased"
  lora:
    rank: 16
    alpha: 32.0
    target_modules: ["q_proj", "v_proj"]
  load_in_4bit: true

distillation:
  temperature: 6.0
  alpha: 0.8
  progressive:
    layer_mapping: [[0, 3], [1, 7], [2, 11]]
    hidden_weight: 0.5
  attention_transfer:
    weight: 0.1

training:
  epochs: 5
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 0.01
  gradient_checkpointing: true
  mixed_precision: "bf16"

dataset:
  path: "wikitext"
  max_seq_length: 256
  max_train_examples: 10000

output:
  dir: "./distillation_output"
  save_steps: 1000
  eval_steps: 200
"#;

    // =========================================================================
    // Parsing Tests
    // =========================================================================

    #[test]
    fn test_parse_minimal_config() {
        let config = DistillationYamlConfig::from_yaml(MINIMAL_CONFIG);
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.teacher.model_id, "microsoft/codebert-base");
        assert_eq!(config.student.model_id, "distilbert-base-uncased");
    }

    #[test]
    fn test_parse_full_config() {
        let config = DistillationYamlConfig::from_yaml(FULL_CONFIG);
        assert!(config.is_ok());
        let config = config.unwrap();

        // Teacher
        assert_eq!(config.teacher.model_id, "microsoft/codebert-base");
        assert!(!config.teacher.load_in_8bit);

        // Student
        assert!(config.student.lora.is_some());
        let lora = config.student.lora.as_ref().unwrap();
        assert_eq!(lora.rank, 16);
        assert_eq!(lora.alpha, 32.0);
        assert!(config.student.load_in_4bit);

        // Distillation
        assert_eq!(config.distillation.temperature, 6.0);
        assert_eq!(config.distillation.alpha, 0.8);
        assert!(config.distillation.progressive.is_some());
        assert!(config.distillation.attention_transfer.is_some());

        // Training
        assert_eq!(config.training.epochs, 5);
        assert_eq!(config.training.batch_size, 32);
        assert!(config.training.gradient_checkpointing);
    }

    #[test]
    fn test_defaults() {
        let config = DistillationYamlConfig::from_yaml(MINIMAL_CONFIG).unwrap();

        // Check defaults
        assert_eq!(config.distillation.temperature, 4.0);
        assert_eq!(config.distillation.alpha, 0.7);
        assert_eq!(config.training.epochs, 3);
        assert_eq!(config.training.batch_size, 16);
        assert_eq!(config.teacher.revision, "main");
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_minimal() {
        let config = DistillationYamlConfig::from_yaml(MINIMAL_CONFIG).unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_teacher() {
        let yaml = r#"
teacher:
  model_id: ""
student:
  model_id: "test"
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_temperature() {
        let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
distillation:
  temperature: -1.0
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_alpha() {
        let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
distillation:
  alpha: 1.5
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert!(config.validate().is_err());
    }

    // =========================================================================
    // Conversion Tests
    // =========================================================================

    #[test]
    fn test_to_trainer_config() {
        let config = DistillationYamlConfig::from_yaml(FULL_CONFIG).unwrap();
        let trainer_config = config.to_trainer_config();
        assert!(trainer_config.is_ok());

        let trainer = trainer_config.unwrap();
        assert_eq!(trainer.teacher_model, "microsoft/codebert-base");
        assert_eq!(trainer.epochs, 5);
        assert!(trainer.progressive.is_some());
    }

    #[test]
    fn test_lora_config_conversion() {
        let yaml = LoRAYamlConfig {
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string(), "k_proj".to_string()],
            layers: Some(vec![0, 1, 2]),
        };

        let lora_config = LoRAConfig::from(&yaml);
        assert_eq!(lora_config.rank, 8);
        assert_eq!(lora_config.alpha, 16.0);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_roundtrip() {
        let config = DistillationYamlConfig::from_yaml(FULL_CONFIG).unwrap();
        let yaml = config.to_yaml().unwrap();
        let config2 = DistillationYamlConfig::from_yaml(&yaml).unwrap();

        assert_eq!(config.teacher.model_id, config2.teacher.model_id);
        assert_eq!(config.training.epochs, config2.training.epochs);
    }

    #[test]
    fn test_save_load() {
        let config = DistillationYamlConfig::from_yaml(MINIMAL_CONFIG).unwrap();
        let path = "/tmp/test_distill_config.yaml";

        config.save(path).unwrap();
        let loaded = DistillationYamlConfig::load(path).unwrap();

        assert_eq!(config.teacher.model_id, loaded.teacher.model_id);
        std::fs::remove_file(path).ok();
    }

    // =========================================================================
    // Progressive Config Tests
    // =========================================================================

    #[test]
    fn test_progressive_config() {
        let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
distillation:
  progressive:
    layer_mapping: [[0, 2], [1, 5], [2, 8]]
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        let prog = config.distillation.progressive.unwrap();
        assert_eq!(prog.layer_mapping.len(), 3);
        assert_eq!(prog.layer_mapping[0], [0, 2]);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_validate_empty_student() {
        let yaml = r#"
teacher:
  model_id: "teacher_model"
student:
  model_id: ""
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_batch_size() {
        let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
training:
  batch_size: 0
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_negative_learning_rate() {
        let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
training:
  learning_rate: -0.001
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_empty_dataset_path() {
        let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
dataset:
  path: ""
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_attention_transfer_config() {
        let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
distillation:
  attention_transfer:
    weight: 0.3
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        let at = config.distillation.attention_transfer.unwrap();
        assert_eq!(at.weight, 0.3);
    }

    #[test]
    fn test_mixed_precision_fp16() {
        let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
training:
  mixed_precision: "fp16"
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.training.mixed_precision, Some("fp16".to_string()));
    }

    #[test]
    fn test_mixed_precision_bf16() {
        let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
training:
  mixed_precision: "bf16"
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.training.mixed_precision, Some("bf16".to_string()));
    }

    #[test]
    fn test_full_fine_tune_no_lora() {
        let yaml = r#"
teacher:
  model_id: "teacher"
student:
  model_id: "student"
  load_in_4bit: false
dataset:
  path: "data"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        let trainer_config = config.to_trainer_config().unwrap();
        // Should have fine_tune set for full fine-tuning
        assert!(!trainer_config.fine_tune.model_id.is_empty());
    }

    #[test]
    fn test_teacher_revision() {
        let yaml = r#"
teacher:
  model_id: "test"
  revision: "v1.0.0"
student:
  model_id: "test"
dataset:
  path: "test"
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.teacher.revision, "v1.0.0");
    }

    #[test]
    fn test_output_config() {
        let yaml = r#"
teacher:
  model_id: "test"
student:
  model_id: "test"
dataset:
  path: "test"
output:
  dir: "/custom/output"
  save_steps: 500
  eval_steps: 250
"#;
        let config = DistillationYamlConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.output.dir, "/custom/output");
        assert_eq!(config.output.save_steps, 500);
        assert_eq!(config.output.eval_steps, 250);
    }

    #[test]
    fn test_lora_with_layers() {
        let yaml = LoRAYamlConfig {
            rank: 16,
            alpha: 32.0,
            target_modules: vec!["q_proj".to_string()],
            layers: Some(vec![0, 1, 2, 3]),
        };
        assert_eq!(yaml.layers.as_ref().unwrap().len(), 4);
    }

    #[test]
    fn test_lora_without_layers() {
        let yaml = LoRAYamlConfig {
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["v_proj".to_string()],
            layers: None,
        };
        assert!(yaml.layers.is_none());
    }
}
