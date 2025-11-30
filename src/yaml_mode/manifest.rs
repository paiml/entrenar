//! Training Manifest Schema
//!
//! Defines the complete YAML Mode Training manifest structure as specified in
//! docs/specifications/yaml-mode-train.md

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete training manifest (root structure)
///
/// # Required Fields
/// - `entrenar`: Specification version (must be "1.0")
/// - `name`: Experiment identifier
/// - `version`: Experiment version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingManifest {
    /// Specification version (required)
    pub entrenar: String,

    /// Experiment name (required)
    pub name: String,

    /// Experiment version (required)
    pub version: String,

    /// Human-readable description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Global random seed for reproducibility
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Dataset configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<DataConfig>,

    /// Model configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelConfig>,

    /// Optimizer configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<OptimizerConfig>,

    /// Learning rate scheduler configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<SchedulerConfig>,

    /// Training loop configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingConfig>,

    /// LoRA fine-tuning configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraConfig>,

    /// Quantization configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantize: Option<QuantizeConfig>,

    /// Monitoring configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monitoring: Option<MonitoringConfig>,

    /// Training callbacks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub callbacks: Option<Vec<CallbackConfig>>,

    /// Output and artifact configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputConfig>,

    // Extended configurations for YAML Mode QA Epic
    /// CITL (Compiler-in-the-Loop) configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citl: Option<CitlConfig>,

    /// RAG configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rag: Option<RagConfig>,

    /// Graph output configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<GraphConfig>,

    /// Distillation configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distillation: Option<DistillationConfig>,

    /// Inspection configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inspect: Option<InspectConfig>,

    /// Privacy configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy: Option<PrivacyConfig>,

    /// Audit configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit: Option<AuditConfig>,

    /// Session configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionConfig>,

    /// Stress testing configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stress: Option<StressConfig>,

    /// Benchmark configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark: Option<BenchmarkConfig>,

    /// Debug configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<DebugConfig>,

    /// Signing configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signing: Option<SigningConfig>,

    /// Verification configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification: Option<VerificationConfig>,

    /// Lockfile path for reproducibility
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lockfile: Option<String>,

    /// Strict mode for lockfile enforcement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,

    /// Strict validation mode
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_validation: Option<bool>,

    /// Require peer review for production
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_peer_review: Option<bool>,
}

// ============================================================================
// DATA CONFIGURATION
// ============================================================================

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Data source URI (pacha://, hf://, s3://, or local path)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Explicit format (auto-detected if omitted)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Data split configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<DataSplit>,

    /// Explicit training data path
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train: Option<String>,

    /// Explicit validation data path
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val: Option<String>,

    /// Explicit test data path
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test: Option<String>,

    /// Preprocessing pipeline
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preprocessing: Option<Vec<PreprocessingStep>>,

    /// Data augmentation pipeline
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub augmentation: Option<Vec<HashMap<String, serde_json::Value>>>,

    /// DataLoader settings
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loader: Option<DataLoader>,
}

/// Data split ratios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSplit {
    /// Training set ratio (0.0-1.0)
    pub train: f64,

    /// Validation set ratio (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val: Option<f64>,

    /// Test set ratio (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test: Option<f64>,

    /// Column name for stratified sampling
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stratify: Option<String>,

    /// Split seed (inherits global if omitted)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Preprocessing step (normalize, encode, drop, fillna, tokenize)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PreprocessingStep {
    /// Normalization step
    Normalize { normalize: NormalizeConfig },
    /// Encoding step
    Encode { encode: EncodeConfig },
    /// Drop columns step
    Drop { drop: DropConfig },
    /// Fill NA step
    FillNa { fillna: FillNaConfig },
    /// Tokenization step
    Tokenize { tokenize: TokenizeConfig },
}

/// Normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizeConfig {
    pub columns: Vec<String>,
    pub method: String,
}

/// Encoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeConfig {
    pub columns: Vec<String>,
    pub method: String,
}

/// Drop columns configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropConfig {
    pub columns: Vec<String>,
}

/// Fill NA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillNaConfig {
    pub strategy: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
}

/// Tokenization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeConfig {
    pub tokenizer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub padding: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
}

/// DataLoader settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoader {
    /// Batch size
    pub batch_size: usize,

    /// Shuffle data each epoch
    pub shuffle: bool,

    /// Number of worker processes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_workers: Option<usize>,

    /// Pin memory for GPU transfer
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin_memory: Option<bool>,

    /// Drop incomplete last batch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drop_last: Option<bool>,

    /// Prefetch factor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefetch_factor: Option<usize>,
}

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model source URI (pacha://, hf://, or local path)
    pub source: String,

    /// Model format (safetensors, gguf, apr, pt)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Architecture override
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<ArchitectureConfig>,

    /// Layers to freeze
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freeze: Option<Vec<String>>,

    /// Device placement (auto, cpu, cuda, cuda:0, mps)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,

    /// Data type (float32, float16, bfloat16)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
}

/// Model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Architecture type (transformer, sequential)
    #[serde(rename = "type")]
    pub arch_type: String,

    /// Hidden size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<usize>,

    /// Number of layers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_layers: Option<usize>,

    /// Number of attention heads
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_heads: Option<usize>,

    /// Vocabulary size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,

    /// Maximum sequence length
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seq_length: Option<usize>,

    /// Sequential layers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<HashMap<String, serde_json::Value>>>,
}

// ============================================================================
// OPTIMIZER CONFIGURATION
// ============================================================================

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer name (sgd, adam, adamw, rmsprop, adagrad, lamb)
    pub name: String,

    /// Learning rate
    pub lr: f64,

    /// Weight decay (L2 regularization)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f64>,

    /// Adam/AdamW betas
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub betas: Option<Vec<f64>>,

    /// Adam epsilon
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eps: Option<f64>,

    /// AMSGrad variant
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amsgrad: Option<bool>,

    /// SGD momentum
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub momentum: Option<f64>,

    /// Nesterov momentum
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nesterov: Option<bool>,

    /// SGD dampening
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dampening: Option<f64>,

    /// RMSprop alpha
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,

    /// RMSprop centered
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub centered: Option<bool>,

    /// Per-parameter groups
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_groups: Option<Vec<ParamGroup>>,
}

/// Per-parameter group configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamGroup {
    pub params: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f64>,
}

// ============================================================================
// SCHEDULER CONFIGURATION
// ============================================================================

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler name (step, cosine, linear, exponential, plateau, one_cycle)
    pub name: String,

    /// Warmup configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup: Option<WarmupConfig>,

    /// Cosine annealing T_max
    #[serde(rename = "T_max", default, skip_serializing_if = "Option::is_none")]
    pub t_max: Option<usize>,

    /// Cosine annealing eta_min
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta_min: Option<f64>,

    /// Step scheduler step_size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_size: Option<usize>,

    /// Step/exponential gamma
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gamma: Option<f64>,

    /// Plateau scheduler mode (min, max)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,

    /// Plateau scheduler factor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factor: Option<f64>,

    /// Plateau scheduler patience
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patience: Option<usize>,

    /// Plateau scheduler threshold
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,

    /// One-cycle max_lr
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_lr: Option<f64>,

    /// One-cycle pct_start
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pct_start: Option<f64>,

    /// One-cycle anneal_strategy
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anneal_strategy: Option<String>,

    /// One-cycle div_factor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub div_factor: Option<f64>,

    /// One-cycle final_div_factor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_div_factor: Option<f64>,
}

/// Warmup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    /// Warmup steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<usize>,

    /// Warmup ratio (alternative to steps)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ratio: Option<f64>,

    /// Starting learning rate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_lr: Option<f64>,
}

// ============================================================================
// TRAINING CONFIGURATION
// ============================================================================

/// Training loop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of epochs
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epochs: Option<usize>,

    /// Maximum training steps (mutually exclusive with epochs)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,

    /// Maximum wall-clock duration (mutually exclusive with epochs/max_steps)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<String>,

    /// Gradient settings
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient: Option<GradientConfig>,

    /// Mixed precision training
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixed_precision: Option<MixedPrecisionConfig>,

    /// Distributed training
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distributed: Option<DistributedConfig>,

    /// Checkpointing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<CheckpointConfig>,

    /// Early stopping (Jidoka)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_stopping: Option<EarlyStoppingConfig>,

    /// Validation configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationConfig>,

    /// Deterministic mode
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deterministic: Option<bool>,

    /// Benchmark mode (cuDNN autotuner)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark: Option<bool>,
}

/// Gradient settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConfig {
    /// Gradient accumulation steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accumulation_steps: Option<usize>,

    /// Gradient clipping (L2 norm)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_norm: Option<f64>,

    /// Gradient clipping (absolute value)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_value: Option<f64>,
}

/// Mixed precision training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Enable mixed precision
    pub enabled: bool,

    /// Data type (float16, bfloat16)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// Loss scale (dynamic, static, or float)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_scale: Option<String>,
}

/// Distributed training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Strategy (ddp, fsdp, deepspeed)
    pub strategy: String,

    /// World size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_size: Option<usize>,

    /// Gradient as bucket view
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_as_bucket_view: Option<bool>,

    /// Find unused parameters
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub find_unused_parameters: Option<bool>,
}

/// Checkpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Save every N steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_every: Option<usize>,

    /// Keep last N checkpoints
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_last: Option<usize>,

    /// Save best model by metric
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_best: Option<bool>,

    /// Metric for best model selection
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,

    /// Metric mode (min, max)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

/// Early stopping configuration (Jidoka - automatic halt on quality degradation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,

    /// Metric to monitor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,

    /// Patience (epochs without improvement)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patience: Option<usize>,

    /// Minimum delta for improvement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_delta: Option<f64>,

    /// Metric mode (min, max)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validate every N steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every: Option<usize>,

    /// Validate each epoch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every_epoch: Option<bool>,

    /// Metrics to compute
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<String>>,

    /// Cross-validation configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_validation: Option<CrossValidationConfig>,
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub folds: usize,

    /// Stratified sampling
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stratified: Option<bool>,

    /// Shuffle data
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle: Option<bool>,
}

// ============================================================================
// LORA CONFIGURATION
// ============================================================================

/// LoRA (Low-Rank Adaptation) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Enable LoRA
    pub enabled: bool,

    /// Rank of low-rank matrices (r)
    pub rank: usize,

    /// Scaling factor (alpha)
    pub alpha: f64,

    /// LoRA dropout
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropout: Option<f64>,

    /// Target modules for LoRA
    pub target_modules: Vec<String>,

    /// Target modules pattern (regex)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_modules_pattern: Option<String>,

    /// Bias training (none, all, lora_only)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bias: Option<String>,

    /// Weight initialization (gaussian, xavier, kaiming)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_weights: Option<String>,

    /// QLoRA: Quantize base model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantize_base: Option<bool>,

    /// QLoRA: Quantization bits
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantize_bits: Option<u8>,

    /// QLoRA: Double quantization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub double_quantize: Option<bool>,

    /// QLoRA: Quantization type (nf4, fp4)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_type: Option<String>,
}

// ============================================================================
// QUANTIZATION CONFIGURATION
// ============================================================================

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeConfig {
    /// Enable quantization
    pub enabled: bool,

    /// Quantization bits (2, 4, 8)
    pub bits: u8,

    /// Quantization scheme (symmetric, asymmetric, dynamic)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,

    /// Granularity (per_tensor, per_channel, per_group)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub granularity: Option<String>,

    /// Group size for per_group quantization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_size: Option<usize>,

    /// QAT configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qat: Option<QatConfig>,

    /// PTQ calibration configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationConfig>,

    /// Layers to exclude from quantization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<Vec<String>>,
}

/// Quantization-aware training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QatConfig {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observer: Option<String>,
}

/// Post-training quantization calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub percentile: Option<f64>,
}

// ============================================================================
// MONITORING CONFIGURATION
// ============================================================================

/// Monitoring configuration (genchi genbutsu - go and see)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Terminal visualization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<TerminalMonitor>,

    /// Experiment tracking
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracking: Option<TrackingConfig>,

    /// System metrics
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemMonitorConfig>,

    /// Alerts (Andon system)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alerts: Option<Vec<AlertConfig>>,

    /// Drift detection configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift_detection: Option<DriftDetectionConfig>,
}

/// Drift detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetectionConfig {
    /// Whether drift detection is enabled
    pub enabled: bool,

    /// Path to baseline statistics
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<String>,

    /// Threshold for triggering drift alert (e.g., PSI threshold)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,

    /// Window size for drift detection
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,

    /// Features to monitor for drift
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<String>>,

    /// Drift detection method (e.g., "psi", "ks", "wasserstein")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
}

/// Terminal monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalMonitor {
    /// Enable terminal monitoring
    pub enabled: bool,

    /// Refresh rate in ms
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_rate: Option<usize>,

    /// Metrics to display
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<String>>,

    /// Charts configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub charts: Option<Vec<ChartConfig>>,
}

/// Chart configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfig {
    #[serde(rename = "type")]
    pub chart_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_eta: Option<bool>,
}

/// Experiment tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingConfig {
    /// Enable tracking
    pub enabled: bool,

    /// Backend (trueno-db, mlflow, wandb, tensorboard)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,

    /// Project name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,

    /// Experiment name (supports templates)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment: Option<String>,

    /// Tags
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
}

/// System monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMonitorConfig {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<String>>,
}

/// Alert configuration (Andon system)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub condition: String,
    pub action: String,
    pub message: String,
}

// ============================================================================
// CALLBACKS CONFIGURATION
// ============================================================================

/// Callback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallbackConfig {
    /// Callback type
    #[serde(rename = "type")]
    pub callback_type: CallbackType,

    /// Trigger event
    pub trigger: String,

    /// Interval (for step-based triggers)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<usize>,

    /// Callback-specific configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<HashMap<String, serde_json::Value>>,

    /// Custom script (for custom callbacks)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
}

/// Callback type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CallbackType {
    Checkpoint,
    LrMonitor,
    GradientMonitor,
    SamplePredictions,
    Custom,
}

// ============================================================================
// OUTPUT CONFIGURATION
// ============================================================================

/// Output and artifact configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory (supports templates)
    pub dir: String,

    /// Model output configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelOutputConfig>,

    /// Metrics export configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<MetricsOutputConfig>,

    /// Training report configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<ReportConfig>,

    /// Artifact registry configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry: Option<RegistryConfig>,
}

/// Model output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutputConfig {
    /// Output format (safetensors, pt, gguf, apr)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Save optimizer state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_optimizer: Option<bool>,

    /// Save scheduler state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_scheduler: Option<bool>,
}

/// Metrics output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsOutputConfig {
    /// Output format (parquet, csv, json)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Metrics to include
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
}

/// Training report configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_plots: Option<bool>,
}

/// Artifact registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_config: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_metrics: Option<bool>,
}

// ============================================================================
// EXTENDED CONFIGURATIONS (YAML Mode QA Epic)
// ============================================================================

/// CITL (Compiler-in-the-Loop) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitlConfig {
    /// Mode: suggest, trace, index, tarantula
    pub mode: String,

    /// Error code for suggestions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,

    /// Top K suggestions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,

    /// Workspace mode for cross-crate analysis
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<bool>,

    /// Include dependencies in analysis
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_deps: Option<bool>,
}

/// RAG (Retrieval-Augmented Generation) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Pattern store path
    pub store: String,

    /// Similarity threshold
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity_threshold: Option<f64>,

    /// Max results to return
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
}

/// Graph output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Output file path
    pub output: String,

    /// Output format (dot, json, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Include edges in graph
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_edges: Option<bool>,
}

/// Distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Teacher model configuration
    pub teacher: DistillModelRef,

    /// Student model configuration
    pub student: DistillModelRef,

    /// Distillation temperature
    pub temperature: f64,

    /// Alpha weight for distillation loss vs hard labels
    pub alpha: f64,

    /// Loss function (kl_div, mse, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss: Option<String>,
}

/// Model reference for distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillModelRef {
    /// Model source path
    pub source: String,

    /// Device placement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

/// Inspection configuration for data analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectConfig {
    /// Inspection mode: outliers, distribution, correlation
    pub mode: String,

    /// Z-score threshold for outlier detection
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_threshold: Option<f64>,

    /// Action on detection: log, drop, flag
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,

    /// Columns to inspect
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub columns: Option<Vec<String>>,
}

/// Privacy configuration for differential privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Enable differential privacy
    pub differential: bool,

    /// Privacy budget epsilon
    pub epsilon: f64,

    /// Privacy budget delta
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<f64>,

    /// Maximum gradient norm for clipping
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_grad_norm: Option<f64>,

    /// Noise multiplier
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_multiplier: Option<f64>,

    /// Privacy accountant type (rdp, gdp, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accountant: Option<String>,
}

/// Audit configuration for bias and fairness testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Audit type: bias, fairness, security
    #[serde(rename = "type")]
    pub audit_type: String,

    /// Protected attribute for bias testing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protected_attr: Option<String>,

    /// Favorable outcome value
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub favorable_outcome: Option<i32>,

    /// Metrics to compute
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<String>>,

    /// Threshold for passing audit
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,

    /// Subgroups to analyze
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subgroups: Option<Vec<String>>,
}

/// Session configuration for stateful training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Unique session identifier
    pub id: String,

    /// Auto-save session state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_save: Option<bool>,

    /// Resume on crash
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_on_crash: Option<bool>,

    /// State directory
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_dir: Option<String>,
}

/// Stress testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressConfig {
    /// Number of parallel jobs
    pub parallel_jobs: usize,

    /// Test duration (e.g., "24h")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<String>,

    /// Memory limit as fraction (0.0-1.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<f64>,

    /// Backpressure configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backpressure: Option<BackpressureConfig>,
}

/// Backpressure configuration for stress testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureConfig {
    /// Enable backpressure handling
    pub enabled: bool,

    /// Queue size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_size: Option<usize>,

    /// Drop policy: oldest, newest, random
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drop_policy: Option<String>,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Benchmark mode: inference, training, throughput
    pub mode: String,

    /// Warmup iterations
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup: Option<usize>,

    /// Number of iterations
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterations: Option<usize>,

    /// Batch sizes to test
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_sizes: Option<Vec<usize>>,

    /// Percentiles to report
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub percentiles: Option<Vec<String>>,
}

/// Debug configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Enable memory profiling
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_profile: Option<bool>,

    /// Log interval in steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_interval: Option<usize>,

    /// GC interval in steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gc_interval: Option<usize>,
}

/// Signing configuration for model artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigningConfig {
    /// Enable signing
    pub enabled: bool,

    /// Signing algorithm (ed25519, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,

    /// Signing key (env var reference)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

/// Verification configuration for production releases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Require all 25 QA checks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all_25_checks: Option<bool>,

    /// QA lead sign-off requirement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qa_lead_sign_off: Option<String>,

    /// Engineering lead sign-off requirement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eng_lead_sign_off: Option<String>,

    /// Safety officer sign-off requirement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_officer_sign_off: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callback_type_serde() {
        let json = r#""checkpoint""#;
        let ct: CallbackType = serde_json::from_str(json).unwrap();
        assert_eq!(ct, CallbackType::Checkpoint);
    }
}
