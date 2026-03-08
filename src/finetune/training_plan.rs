//! Training plan — forjar-style plan/apply for ML training.
//!
//! A `TrainingPlan` captures everything needed to execute a training run:
//! data audit results, model configuration, hyperparameter strategy, resource
//! estimates, and pre-flight check results. The plan is generated without
//! touching the GPU, so validation is fast and cheap.
//!
//! # Architecture (mirrors forjar plan/apply)
//!
//! ```text
//! PlanConfig → validate data → check model → build HPO → estimate cost → pre-flight → TrainingPlan
//!                                                                                          │
//!                                   TrainingPlan → apply (future) → checkpoint + lock
//! ```

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::classification::{corpus_stats, load_safety_corpus};
use super::classify_tuner::{default_classify_search_space, extract_trial_params, TuneStrategy};

// ═══════════════════════════════════════════════════════════════════════
// Plan input configuration
// ═══════════════════════════════════════════════════════════════════════

/// Input configuration for plan generation.
///
/// This is the user's intent — what they want to train. The plan builder
/// validates this against reality and produces a `TrainingPlan`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanConfig {
    /// Task type (currently only "classify").
    pub task: String,
    /// Path to training data (JSONL).
    pub data_path: PathBuf,
    /// Optional validation data (JSONL). If absent, split from training data.
    pub val_path: Option<PathBuf>,
    /// Optional test data (JSONL). Used for post-train eval.
    pub test_path: Option<PathBuf>,
    /// Model size hint (e.g. "0.5B", "9B").
    pub model_size: String,
    /// Path to model weights directory.
    pub model_path: Option<PathBuf>,
    /// Number of output classes.
    pub num_classes: usize,
    /// Output directory for checkpoints.
    pub output_dir: PathBuf,
    /// HPO strategy: "tpe", "grid", "random", or "manual".
    pub strategy: String,
    /// HPO budget (number of trials). Ignored if strategy is "manual".
    pub budget: usize,
    /// Scout mode (1 epoch per trial).
    pub scout: bool,
    /// Maximum epochs per trial.
    pub max_epochs: usize,
    /// Manual hyperparameters (used when strategy is "manual").
    pub manual_lr: Option<f32>,
    /// Manual LoRA rank.
    pub manual_lora_rank: Option<usize>,
    /// Manual batch size.
    pub manual_batch_size: Option<usize>,
    /// Manual LoRA alpha.
    pub manual_lora_alpha: Option<f32>,
    /// Manual warmup fraction.
    pub manual_warmup: Option<f32>,
    /// Manual gradient clip norm.
    pub manual_gradient_clip: Option<f32>,
    /// Manual LR min ratio.
    pub manual_lr_min_ratio: Option<f32>,
    /// Manual class weight strategy.
    pub manual_class_weights: Option<String>,
    /// Manual target modules.
    pub manual_target_modules: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════
// Training plan output
// ═══════════════════════════════════════════════════════════════════════

/// Complete training plan — the serializable artifact that describes
/// exactly what a training run will do.
///
/// Analogous to forjar's `ExecutionPlan`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPlan {
    /// Plan format version.
    pub version: String,
    /// Task type.
    pub task: String,

    // ── Data audit ─────────────────────────────────────────────────────
    /// Data audit summary.
    pub data: DataAudit,

    // ── Model ──────────────────────────────────────────────────────────
    /// Model configuration summary.
    pub model: ModelInfo,

    // ── Hyperparameters ────────────────────────────────────────────────
    /// Hyperparameter configuration.
    pub hyperparameters: HyperparameterPlan,

    // ── Resource estimates ─────────────────────────────────────────────
    /// Estimated resource usage.
    pub resources: ResourceEstimate,

    // ── Pre-flight checks ──────────────────────────────────────────────
    /// Pre-flight check results.
    pub pre_flight: Vec<PreFlightCheck>,

    // ── Output config ──────────────────────────────────────────────────
    /// Output directory.
    pub output_dir: String,
    /// Whether to auto-diagnose after training.
    pub auto_diagnose: bool,

    // ── Plan-level verdict ─────────────────────────────────────────────
    /// Overall plan status.
    pub verdict: PlanVerdict,
    /// Issues found during planning.
    pub issues: Vec<PlanIssue>,
}

/// Data audit results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAudit {
    /// Path to training data.
    pub train_path: String,
    /// Total training samples.
    pub train_samples: usize,
    /// Average input length in characters.
    pub avg_input_len: usize,
    /// Per-class sample counts.
    pub class_counts: Vec<usize>,
    /// Imbalance ratio (max/min class count).
    pub imbalance_ratio: f64,
    /// Whether class weighting will be auto-applied.
    pub auto_class_weights: bool,
    /// Validation samples (if separate file provided).
    pub val_samples: Option<usize>,
    /// Test samples (if separate file provided).
    pub test_samples: Option<usize>,
    /// Number of duplicate inputs detected.
    pub duplicates: usize,
    /// Number of samples with shell preamble.
    pub preamble_count: usize,
}

/// Model configuration summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model size label.
    pub size: String,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Architecture name.
    pub architecture: String,
    /// Whether model weights are loadable.
    pub weights_available: bool,
    /// LoRA trainable parameters (estimated).
    pub lora_trainable_params: usize,
    /// Classifier head parameters.
    pub classifier_params: usize,
}

/// Hyperparameter plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterPlan {
    /// Strategy: "tpe", "grid", "random", or "manual".
    pub strategy: String,
    /// Number of HPO trials (0 if manual).
    pub budget: usize,
    /// Scout mode.
    pub scout: bool,
    /// Maximum epochs per trial.
    pub max_epochs: usize,
    /// Search space parameter count (0 if manual).
    pub search_space_params: usize,
    /// Sample trial configurations (first 3 from searcher).
    pub sample_configs: Vec<TrialPreview>,
    /// Manual config (if strategy is "manual").
    pub manual: Option<ManualConfig>,
    /// Recommendation: should user switch strategy?
    pub recommendation: Option<String>,
}

/// Preview of a single HPO trial configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialPreview {
    /// Trial index.
    pub trial: usize,
    /// Learning rate.
    pub learning_rate: f32,
    /// LoRA rank.
    pub lora_rank: usize,
    /// LoRA alpha.
    pub lora_alpha: f32,
    /// Batch size.
    pub batch_size: usize,
    /// Warmup fraction.
    pub warmup: f32,
    /// Gradient clip norm.
    pub gradient_clip: f32,
    /// Class weight strategy name.
    pub class_weights: String,
    /// Target modules.
    pub target_modules: String,
    /// LR min ratio.
    pub lr_min_ratio: f32,
}

/// Manual hyperparameter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualConfig {
    /// Learning rate.
    pub learning_rate: f32,
    /// LoRA rank.
    pub lora_rank: usize,
    /// Batch size.
    pub batch_size: usize,
    /// LoRA alpha (defaults to rank if absent).
    #[serde(default)]
    pub lora_alpha: Option<f32>,
    /// Warmup fraction (defaults to 0.1 if absent).
    #[serde(default)]
    pub warmup_fraction: Option<f32>,
    /// Gradient clip norm (defaults to 1.0 if absent).
    #[serde(default)]
    pub gradient_clip_norm: Option<f32>,
    /// LR min ratio for cosine decay (defaults to 0.01 if absent).
    #[serde(default)]
    pub lr_min_ratio: Option<f32>,
    /// Class weight strategy: "uniform", "inverse_freq", "sqrt_inverse".
    #[serde(default)]
    pub class_weights: Option<String>,
    /// Target modules: "qv", "qkv", "all_linear".
    #[serde(default)]
    pub target_modules: Option<String>,
}

/// Resource usage estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEstimate {
    /// Estimated VRAM usage in GB.
    pub estimated_vram_gb: f64,
    /// Estimated time per epoch in minutes.
    pub estimated_minutes_per_epoch: f64,
    /// Estimated total training time in minutes.
    pub estimated_total_minutes: f64,
    /// Estimated checkpoint storage in MB.
    pub estimated_checkpoint_mb: f64,
    /// Steps per epoch (train_samples / batch_size).
    pub steps_per_epoch: usize,
    /// Detected GPU device name (if available).
    pub gpu_device: Option<String>,
}

/// A single pre-flight check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFlightCheck {
    /// Check name.
    pub name: String,
    /// Check status.
    pub status: CheckStatus,
    /// Detail message.
    pub detail: String,
}

/// Status of a pre-flight check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckStatus {
    /// Check passed.
    Pass,
    /// Warning (non-blocking).
    Warn,
    /// Failed (blocks apply).
    Fail,
}

/// Overall plan verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanVerdict {
    /// All checks pass, ready to apply.
    Ready,
    /// Warnings present but can proceed.
    WarningsPresent,
    /// Failures detected, cannot apply.
    Blocked,
}

/// An issue found during planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanIssue {
    /// Issue severity.
    pub severity: CheckStatus,
    /// Issue category.
    pub category: String,
    /// Issue description.
    pub message: String,
    /// Suggested fix.
    pub fix: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════
// Plan generation
// ═══════════════════════════════════════════════════════════════════════

/// Generate a training plan from a `PlanConfig`.
///
/// This is the pure validation phase — no GPU allocation, no weight loading,
/// no training. Reads data files, validates schemas, estimates costs.
pub fn plan(config: &PlanConfig) -> crate::Result<TrainingPlan> {
    let mut issues: Vec<PlanIssue> = Vec::new();
    let mut pre_flight: Vec<PreFlightCheck> = Vec::new();

    // ── 1. Data audit ──────────────────────────────────────────────────

    let data = audit_data(config, &mut issues, &mut pre_flight)?;

    // ── 2. Model info ──────────────────────────────────────────────────

    let model = resolve_model(config, &mut pre_flight);

    // ── 3. Hyperparameter plan ─────────────────────────────────────────

    let hyperparameters = build_hpo_plan(config, data.train_samples, &mut issues);

    // ── 4. Resource estimation ─────────────────────────────────────────
    //
    // For HPO, use the median batch size from search space (64) to get a
    // representative estimate. For manual, use the configured batch size.
    let batch_size = hyperparameters.manual.as_ref().map_or(64, |m| m.batch_size);
    let resources = estimate_resources(config, &model, &data, batch_size);

    // ── 5. Additional pre-flight checks ────────────────────────────────

    // Output directory
    if config.output_dir.exists() {
        let has_checkpoints = config.output_dir.join("metadata.json").exists()
            || config.output_dir.join("epoch_001").exists();
        if has_checkpoints {
            pre_flight.push(PreFlightCheck {
                name: "output_dir".to_string(),
                status: CheckStatus::Warn,
                detail: format!(
                    "Output directory {} already contains checkpoints — may overwrite",
                    config.output_dir.display()
                ),
            });
            issues.push(PlanIssue {
                severity: CheckStatus::Warn,
                category: "Output".to_string(),
                message: "Checkpoint directory already contains previous run".to_string(),
                fix: Some("Use a fresh output directory or rename existing one".to_string()),
            });
        } else {
            pre_flight.push(PreFlightCheck {
                name: "output_dir".to_string(),
                status: CheckStatus::Pass,
                detail: format!("Output directory {} exists", config.output_dir.display()),
            });
        }
    } else {
        pre_flight.push(PreFlightCheck {
            name: "output_dir".to_string(),
            status: CheckStatus::Pass,
            detail: format!("Output directory {} will be created", config.output_dir.display()),
        });
    }

    // Class weights persistence check
    pre_flight.push(PreFlightCheck {
        name: "class_weights_persist".to_string(),
        status: CheckStatus::Pass,
        detail: "class_weights saved in checkpoint metadata (entrenar ≥0.7.5)".to_string(),
    });

    // ── 6. Verdict ─────────────────────────────────────────────────────

    let has_fail = pre_flight.iter().any(|c| c.status == CheckStatus::Fail)
        || issues.iter().any(|i| i.severity == CheckStatus::Fail);
    let has_warn = pre_flight.iter().any(|c| c.status == CheckStatus::Warn)
        || issues.iter().any(|i| i.severity == CheckStatus::Warn);

    let verdict = if has_fail {
        PlanVerdict::Blocked
    } else if has_warn {
        PlanVerdict::WarningsPresent
    } else {
        PlanVerdict::Ready
    };

    Ok(TrainingPlan {
        version: "1.0".to_string(),
        task: config.task.clone(),
        data,
        model,
        hyperparameters,
        resources,
        pre_flight,
        output_dir: config.output_dir.display().to_string(),
        auto_diagnose: true,
        verdict,
        issues,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Internal plan builders
// ═══════════════════════════════════════════════════════════════════════

/// Audit training data without loading into GPU.
fn audit_data(
    config: &PlanConfig,
    issues: &mut Vec<PlanIssue>,
    pre_flight: &mut Vec<PreFlightCheck>,
) -> crate::Result<DataAudit> {
    // Validate data file exists
    if !config.data_path.exists() {
        pre_flight.push(PreFlightCheck {
            name: "data_file".to_string(),
            status: CheckStatus::Fail,
            detail: format!("Training data not found: {}", config.data_path.display()),
        });
        return Err(crate::Error::Io(format!(
            "Training data not found: {}",
            config.data_path.display()
        )));
    }

    // Load and validate corpus
    let corpus = load_safety_corpus(&config.data_path, config.num_classes)?;
    let stats = corpus_stats(&corpus, config.num_classes);

    pre_flight.push(PreFlightCheck {
        name: "data_file".to_string(),
        status: CheckStatus::Pass,
        detail: format!("{} samples loaded from {}", stats.total, config.data_path.display()),
    });

    // Check for empty classes
    let empty_classes: Vec<usize> =
        stats.class_counts.iter().enumerate().filter(|(_, &c)| c == 0).map(|(i, _)| i).collect();
    if empty_classes.is_empty() {
        pre_flight.push(PreFlightCheck {
            name: "class_coverage".to_string(),
            status: CheckStatus::Pass,
            detail: format!("All {} classes have samples", config.num_classes),
        });
    } else {
        pre_flight.push(PreFlightCheck {
            name: "class_coverage".to_string(),
            status: CheckStatus::Fail,
            detail: format!("Classes with zero samples: {empty_classes:?}"),
        });
        issues.push(PlanIssue {
            severity: CheckStatus::Fail,
            category: "Data".to_string(),
            message: format!("Classes {empty_classes:?} have zero training samples"),
            fix: Some("Add samples for missing classes or reduce num_classes".to_string()),
        });
    }

    // Imbalance analysis
    let min_count = stats.class_counts.iter().copied().min().unwrap_or(1).max(1);
    let max_count = stats.class_counts.iter().copied().max().unwrap_or(1);
    let imbalance_ratio = max_count as f64 / min_count as f64;
    let auto_class_weights = imbalance_ratio > 2.0;

    if imbalance_ratio > 5.0 {
        issues.push(PlanIssue {
            severity: CheckStatus::Warn,
            category: "Data".to_string(),
            message: format!(
                "Severe class imbalance ({imbalance_ratio:.1}:1) — sqrt-inverse weights will be auto-applied"
            ),
            fix: Some("Consider oversampling minority classes: apr data balance --strategy oversample".to_string()),
        });
    }

    // Duplicate detection (fast: hash inputs)
    let mut seen = std::collections::HashSet::new();
    let mut duplicates = 0usize;
    for s in &corpus {
        if !seen.insert(&s.input) {
            duplicates += 1;
        }
    }
    if duplicates > 0 {
        issues.push(PlanIssue {
            severity: CheckStatus::Warn,
            category: "Data".to_string(),
            message: format!(
                "{duplicates} duplicate inputs detected ({:.1}%)",
                duplicates as f64 / stats.total as f64 * 100.0
            ),
            fix: Some("Remove duplicates: apr data dedup".to_string()),
        });
    }

    // Preamble detection
    let preamble_count = corpus
        .iter()
        .filter(|s| {
            s.input.starts_with("#!/")
                || s.input.starts_with("#! /")
                || s.input.starts_with("set -")
        })
        .count();
    if preamble_count > stats.total / 10 {
        issues.push(PlanIssue {
            severity: CheckStatus::Warn,
            category: "Data".to_string(),
            message: format!(
                "{preamble_count} samples ({:.0}%) have shell preamble",
                preamble_count as f64 / stats.total as f64 * 100.0
            ),
            fix: Some("Strip preambles: use --strip-preamble during data export".to_string()),
        });
    }

    // Minimum sample count
    if stats.total < 100 {
        issues.push(PlanIssue {
            severity: CheckStatus::Warn,
            category: "Data".to_string(),
            message: format!("Only {} samples — may be insufficient for fine-tuning", stats.total),
            fix: None,
        });
    }

    // Validate val/test if provided
    let val_samples = count_file_samples(config.val_path.as_ref(), config.num_classes);
    let test_samples = count_file_samples(config.test_path.as_ref(), config.num_classes);

    Ok(DataAudit {
        train_path: config.data_path.display().to_string(),
        train_samples: stats.total,
        avg_input_len: stats.avg_input_len,
        class_counts: stats.class_counts,
        imbalance_ratio,
        auto_class_weights,
        val_samples,
        test_samples,
        duplicates,
        preamble_count,
    })
}

/// Count samples in an optional JSONL file.
pub(crate) fn count_file_samples(path: Option<&PathBuf>, num_classes: usize) -> Option<usize> {
    path.and_then(|p| {
        if p.exists() {
            load_safety_corpus(p, num_classes).ok().map(|c| c.len())
        } else {
            None
        }
    })
}

/// Resolve model architecture from size hint.
pub(crate) fn resolve_model(
    config: &PlanConfig,
    pre_flight: &mut Vec<PreFlightCheck>,
) -> ModelInfo {
    let (hidden_size, num_layers, architecture) = match config.model_size.as_str() {
        "0.5B" | "500M" | "qwen2-0.5b" => (896, 24, "qwen2"),
        "9B" | "qwen3.5-9b" => (4096, 48, "qwen3.5"),
        "7B" | "llama2-7b" => (4096, 32, "llama2"),
        "13B" | "llama2-13b" => (5120, 40, "llama2"),
        _ => (896, 24, "unknown"),
    };

    // Check if model weights are available
    let weights_available = config.model_path.as_ref().is_some_and(|p| p.is_dir());
    if let Some(ref path) = config.model_path {
        if weights_available {
            // Check for key files
            let has_safetensors = path.join("model.safetensors").exists()
                || path.join("model-00001-of-00002.safetensors").exists();
            let has_tokenizer = path.join("tokenizer.json").exists();

            if has_safetensors && has_tokenizer {
                pre_flight.push(PreFlightCheck {
                    name: "model_weights".to_string(),
                    status: CheckStatus::Pass,
                    detail: format!("Model weights found at {}", path.display()),
                });
            } else {
                let mut missing = Vec::new();
                if !has_safetensors {
                    missing.push("model.safetensors");
                }
                if !has_tokenizer {
                    missing.push("tokenizer.json");
                }
                pre_flight.push(PreFlightCheck {
                    name: "model_weights".to_string(),
                    status: CheckStatus::Warn,
                    detail: format!("Model directory exists but missing: {}", missing.join(", ")),
                });
            }
        } else {
            pre_flight.push(PreFlightCheck {
                name: "model_weights".to_string(),
                status: CheckStatus::Fail,
                detail: format!("Model path not found: {}", path.display()),
            });
        }
    } else {
        pre_flight.push(PreFlightCheck {
            name: "model_weights".to_string(),
            status: CheckStatus::Warn,
            detail: "No model path specified — will use default model resolution".to_string(),
        });
    }

    // Estimate trainable parameters
    // LoRA: 2 * rank * hidden_size * num_adapters (Q,V per layer = 2 * num_layers)
    let default_rank = config.manual_lora_rank.unwrap_or(16);
    let lora_trainable_params = 2 * default_rank * hidden_size * 2 * num_layers;
    let classifier_params = hidden_size * config.num_classes + config.num_classes;

    ModelInfo {
        size: config.model_size.clone(),
        hidden_size,
        num_layers,
        architecture: architecture.to_string(),
        weights_available,
        lora_trainable_params,
        classifier_params,
    }
}

/// Build HPO plan with search space and sample configs.
pub(crate) fn build_hpo_plan(
    config: &PlanConfig,
    train_samples: usize,
    issues: &mut Vec<PlanIssue>,
) -> HyperparameterPlan {
    let strategy = config.strategy.as_str();

    if strategy == "manual" {
        let lr = config.manual_lr.unwrap_or(1e-4);
        let rank = config.manual_lora_rank.unwrap_or(16);
        let batch = config.manual_batch_size.unwrap_or(32);

        // Warn about manual mode when HPO is available
        issues.push(PlanIssue {
            severity: CheckStatus::Warn,
            category: "Hyperparameters".to_string(),
            message: "Using manual hyperparameters — HPO (--strategy tpe) searches 9 parameters automatically".to_string(),
            fix: Some(format!(
                "apr train plan --strategy tpe --budget 20 --scout --data {}",
                config.data_path.display()
            )),
        });

        return HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: config.max_epochs,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: Some(ManualConfig {
                learning_rate: lr,
                lora_rank: rank,
                batch_size: batch,
                lora_alpha: config.manual_lora_alpha,
                warmup_fraction: config.manual_warmup,
                gradient_clip_norm: config.manual_gradient_clip,
                lr_min_ratio: config.manual_lr_min_ratio,
                class_weights: config.manual_class_weights.clone(),
                target_modules: config.manual_target_modules.clone(),
            }),
            recommendation: Some(
                "Consider using --strategy tpe for automated hyperparameter search".to_string(),
            ),
        };
    }

    // Parse strategy for searcher
    let tune_strategy: TuneStrategy = strategy.parse().unwrap_or(TuneStrategy::Tpe);

    // Build searcher and sample configs
    let space = default_classify_search_space();
    let mut searcher: Box<dyn super::classify_tuner::TuneSearcher> = match tune_strategy {
        TuneStrategy::Tpe => {
            let n_startup = (config.budget / 3).max(3);
            Box::new(super::tune_searchers::TpeSearcher::new(space.clone(), n_startup))
        }
        TuneStrategy::Grid => Box::new(super::tune_searchers::GridSearcher::new(space.clone(), 3)),
        TuneStrategy::Random => Box::new(super::tune_searchers::RandomSearcher::new(space.clone())),
    };

    let num_previews = config.budget.min(3);
    let mut sample_configs = Vec::new();
    for i in 0..num_previews {
        if let Ok(trial) = searcher.suggest() {
            let (lr, rank, alpha, batch, warmup, clip, weights, targets, lr_min) =
                extract_trial_params(&trial.config);
            sample_configs.push(TrialPreview {
                trial: i + 1,
                learning_rate: lr,
                lora_rank: rank,
                lora_alpha: alpha,
                batch_size: batch,
                warmup,
                gradient_clip: clip,
                class_weights: weights,
                target_modules: targets,
                lr_min_ratio: lr_min,
            });
        }
    }

    // Budget sanity check
    if config.budget < 5 && tune_strategy == TuneStrategy::Tpe {
        issues.push(PlanIssue {
            severity: CheckStatus::Warn,
            category: "Hyperparameters".to_string(),
            message: format!(
                "TPE budget {} is low — needs ≥5 trials for Bayesian optimization to converge",
                config.budget
            ),
            fix: Some("Use --budget 20 for better results".to_string()),
        });
    }

    // Scout recommendation for large datasets
    if !config.scout && train_samples > 10_000 && config.max_epochs > 1 {
        issues.push(PlanIssue {
            severity: CheckStatus::Warn,
            category: "Hyperparameters".to_string(),
            message: format!(
                "Full HPO with {} samples × {} epochs × {} trials = ~{:.0} GPU hours",
                train_samples,
                config.max_epochs,
                config.budget,
                estimate_gpu_hours(train_samples, config.max_epochs, config.budget)
            ),
            fix: Some(
                "Use --scout for 1-epoch trials first, then --from-scout for full run".to_string(),
            ),
        });
    }

    HyperparameterPlan {
        strategy: strategy.to_string(),
        budget: config.budget,
        scout: config.scout,
        max_epochs: if config.scout { 1 } else { config.max_epochs },
        search_space_params: 9,
        sample_configs,
        manual: None,
        recommendation: None,
    }
}

/// Estimate GPU hours for a full HPO run.
pub(crate) fn estimate_gpu_hours(train_samples: usize, max_epochs: usize, budget: usize) -> f64 {
    // Based on observed RTX 4090 throughput: ~58 sec/step, median batch_size=64
    let batch_size = 64;
    let steps_per_epoch = train_samples.div_ceil(batch_size);
    let seconds_per_epoch = steps_per_epoch as f64 * 58.0;
    let total_seconds = seconds_per_epoch * max_epochs as f64 * budget as f64;
    total_seconds / 3600.0
}

/// Estimate resource usage.
pub(crate) fn estimate_resources(
    config: &PlanConfig,
    model: &ModelInfo,
    data: &DataAudit,
    batch_size: usize,
) -> ResourceEstimate {
    // VRAM estimate: model weights + optimizer state + activations
    // Qwen2 0.5B: ~1.0 GB weights, ~0.8 GB optimizer, ~0.5 GB activations
    let base_vram = match model.hidden_size {
        896 => 2.5,   // 0.5B
        4096 => 18.0, // 7B/9B
        5120 => 26.0, // 13B
        _ => 3.0,
    };

    let steps_per_epoch = data.train_samples.div_ceil(batch_size);

    // Time estimate based on observed GPU training throughput
    // RTX 4090 measured: ~58 seconds/step for 0.5B at batch_size=40
    // (measured over 245 steps in v3 training run)
    // For larger models, scale by layer count ratio
    let seconds_per_step = match model.hidden_size {
        896 => 58.0,   // 0.5B: observed on RTX 4090
        4096 => 270.0, // 7B/9B: estimated ~4.7x slower
        5120 => 450.0, // 13B: estimated ~7.8x slower
        _ => 90.0,
    };
    let minutes_per_epoch = (steps_per_epoch as f64 * seconds_per_step) / 60.0;

    let total_epochs = if config.scout { 1 } else { config.max_epochs };
    let total_trials = if config.strategy == "manual" { 1 } else { config.budget };
    let total_minutes = minutes_per_epoch * total_epochs as f64 * total_trials as f64;

    // Checkpoint size: LoRA adapters + classifier head
    let checkpoint_mb =
        (model.lora_trainable_params + model.classifier_params) as f64 * 4.0 / 1_048_576.0;

    // Try to detect GPU
    let gpu_device = detect_gpu_device();

    ResourceEstimate {
        estimated_vram_gb: base_vram,
        estimated_minutes_per_epoch: minutes_per_epoch,
        estimated_total_minutes: total_minutes,
        estimated_checkpoint_mb: checkpoint_mb,
        steps_per_epoch,
        gpu_device,
    }
}

/// Detect GPU device name (best-effort, no NVML required).
pub(crate) fn detect_gpu_device() -> Option<String> {
    // Try reading from sysfs (Linux)
    if let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus") {
        for entry in entries.flatten() {
            let info_path = entry.path().join("information");
            if let Ok(info) = std::fs::read_to_string(&info_path) {
                for line in info.lines() {
                    if let Some(name) = line.strip_prefix("Model:") {
                        return Some(name.trim().to_string());
                    }
                }
            }
        }
    }
    // Fallback: check CUDA_VISIBLE_DEVICES
    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
        return Some("CUDA device (unknown model)".to_string());
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════
// Plan execution (apply phase)
// ═══════════════════════════════════════════════════════════════════════

/// Runtime configuration for plan execution.
///
/// Supplements the `TrainingPlan` with execution-time parameters that
/// cannot be determined at plan time (e.g. actual model path resolution).
#[derive(Debug, Clone)]
pub struct ApplyConfig {
    /// Path to model weights directory.
    pub model_path: PathBuf,
    /// Path to training data (JSONL).
    pub data_path: PathBuf,
    /// Output directory for checkpoints and leaderboard.
    pub output_dir: PathBuf,
    /// Callback invoked after each trial completes.
    /// Arguments: (trial_id, total_budget, summary).
    pub on_trial_complete: Option<fn(usize, usize, &super::classify_tuner::TrialSummary)>,
}

/// Execute a training plan (the "apply" phase).
///
/// For HPO plans, iterates over trials:
/// 1. `ClassifyTuner.suggest()` → hyperparameters
/// 2. Build `ClassifyConfig` → `ClassifyPipeline::from_pretrained()`
/// 3. Load corpus → `ClassifyTrainer::new()` → `trainer.train()`
/// 4. Record trial results → check scheduler
/// 5. Return `TuneResult` with leaderboard
///
/// For manual plans, executes a single trial with the specified params.
///
/// # Errors
///
/// Returns error if model weights cannot be loaded, data is invalid,
/// or all trials fail.
pub fn execute_plan(
    plan: &TrainingPlan,
    apply: &ApplyConfig,
) -> crate::Result<super::classify_tuner::TuneResult> {
    use super::classify_pipeline::ClassifyConfig;
    use super::classify_tuner::{
        ClassifyTuner, SchedulerKind, TrialSummary, TuneConfig, TuneStrategy,
    };
    use crate::optim::ParameterValue;
    use crate::transformer::TransformerConfig;
    use std::collections::HashMap;

    // ── Verify pre-conditions ──────────────────────────────────────────
    if plan.verdict == PlanVerdict::Blocked {
        return Err(crate::Error::ConfigError(
            "Cannot apply a blocked plan — resolve all failures first".to_string(),
        ));
    }

    if !apply.model_path.is_dir() {
        return Err(crate::Error::ConfigError(format!(
            "Model path does not exist: {}",
            apply.model_path.display()
        )));
    }

    if !apply.data_path.exists() {
        return Err(crate::Error::Io(format!(
            "Training data not found: {}",
            apply.data_path.display()
        )));
    }

    // Create output directory
    std::fs::create_dir_all(&apply.output_dir).map_err(|e| {
        crate::Error::Io(format!(
            "Failed to create output directory {}: {e}",
            apply.output_dir.display()
        ))
    })?;

    // ── Open project-local experiment store ────────────────────────────
    let mut tracker = ExperimentTracker::open(&apply.output_dir, plan);

    // GH-377: Resolve model config — error on unknown instead of silent tiny()
    let model_config = TransformerConfig::from_size_str(&plan.model.size)
        .map_err(|e| crate::Error::ConfigError(e))?;

    let total_start = std::time::Instant::now();

    // Auto-enable NF4 quantization for large models.
    // Full fp32 weights for hidden_size >= 2048 (roughly >= 1B params)
    // exceed RTX 4090 VRAM (24 GB) after scratch + kernel cache overhead.
    let auto_nf4 = model_config.hidden_size >= 2048;
    if auto_nf4 {
        eprintln!(
            "[plan] Auto-enabling NF4 quantization (hidden_size={} >= 2048)",
            model_config.hidden_size
        );
    }

    // ── Manual strategy: single trial ──────────────────────────────────
    if plan.hyperparameters.strategy == "manual" {
        let manual = plan.hyperparameters.manual.as_ref().ok_or_else(|| {
            crate::Error::ConfigError(
                "Manual strategy requires manual hyperparameters in plan".to_string(),
            )
        })?;

        let num_classes = plan.data.class_counts.len();
        let lora_alpha = manual.lora_alpha.unwrap_or(manual.lora_rank as f32);
        let gradient_clip = manual.gradient_clip_norm.unwrap_or(1.0);
        let warmup = manual.warmup_fraction.unwrap_or(0.1);
        let lr_min_ratio = manual.lr_min_ratio.unwrap_or(0.01);

        let class_weights = manual
            .class_weights
            .as_deref()
            .and_then(|s| resolve_class_weights(s, &plan.data.class_counts, num_classes));

        let classify_config = ClassifyConfig {
            num_classes,
            lora_rank: manual.lora_rank,
            lora_alpha,
            learning_rate: manual.learning_rate,
            epochs: plan.hyperparameters.max_epochs,
            batch_size: manual.batch_size,
            gradient_clip_norm: Some(gradient_clip),
            class_weights,
            quantize_nf4: auto_nf4,
            ..ClassifyConfig::default()
        };

        let trial_start = std::time::Instant::now();
        let result = run_single_trial_with_warmup(
            &apply.model_path,
            &apply.data_path,
            &apply.output_dir.join("trial_001"),
            &model_config,
            classify_config,
            plan.hyperparameters.max_epochs,
            warmup,
            lr_min_ratio,
            &plan.model.size,
        )?;

        let mut config_map = HashMap::new();
        config_map.insert(
            "learning_rate".to_string(),
            ParameterValue::Float(f64::from(manual.learning_rate)),
        );
        config_map.insert("lora_rank".to_string(), ParameterValue::Int(manual.lora_rank as i64));
        config_map.insert(
            "batch_size".to_string(),
            ParameterValue::Categorical(manual.batch_size.to_string()),
        );

        let summary = TrialSummary {
            id: 0,
            val_loss: f64::from(result.best_val_loss),
            val_accuracy: result
                .epoch_metrics
                .get(result.best_epoch)
                .map_or(0.0, |m| f64::from(m.val_accuracy)),
            train_loss: result.epoch_metrics.last().map_or(0.0, |m| f64::from(m.train_loss)),
            train_accuracy: result
                .epoch_metrics
                .last()
                .map_or(0.0, |m| f64::from(m.train_accuracy)),
            epochs_run: result.epoch_metrics.len(),
            time_ms: trial_start.elapsed().as_millis() as u64,
            config: config_map,
            status: if result.stopped_early {
                "stopped_early".to_string()
            } else {
                "completed".to_string()
            },
        };

        tracker.log_manual_trial(manual, &result);

        if let Some(cb) = apply.on_trial_complete {
            cb(0, 1, &summary);
        }

        return Ok(super::classify_tuner::TuneResult {
            strategy: "manual".to_string(),
            mode: "manual".to_string(),
            budget: 1,
            trials: vec![summary],
            best_trial_id: 0,
            total_time_ms: total_start.elapsed().as_millis() as u64,
        });
    }

    // ── HPO strategy: multiple trials ──────────────────────────────────
    let strategy: TuneStrategy = plan.hyperparameters.strategy.parse().unwrap_or(TuneStrategy::Tpe);

    let num_classes = plan.data.class_counts.len();

    let tune_config = TuneConfig {
        budget: plan.hyperparameters.budget,
        strategy,
        scheduler: SchedulerKind::Asha,
        scout: plan.hyperparameters.scout,
        max_epochs: plan.hyperparameters.max_epochs,
        num_classes,
        seed: 42,
        time_limit_secs: None,
    };

    let mut tuner = ClassifyTuner::new(tune_config)?;
    let mut searcher = tuner.build_searcher();
    let scheduler = tuner.build_scheduler();

    let budget = plan.hyperparameters.budget;

    // Save plan as YAML in output dir for reproducibility
    let plan_path = apply.output_dir.join("plan.yaml");
    let _ = std::fs::write(&plan_path, plan.to_yaml());

    for trial_idx in 0..budget {
        // ── Suggest hyperparameters ────────────────────────────────────
        let suggestion = match searcher.suggest() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  Trial {}: searcher exhausted ({e}), stopping", trial_idx + 1);
                break;
            }
        };

        let (lr, rank, alpha, batch_size, warmup, clip, weights_strategy, _targets, lr_min_ratio) =
            super::classify_tuner::extract_trial_params(&suggestion.config);

        // ── Build ClassifyConfig from trial params ─────────────────────
        let class_weights =
            resolve_class_weights(&weights_strategy, &plan.data.class_counts, num_classes);

        let epochs = if plan.hyperparameters.scout { 1 } else { plan.hyperparameters.max_epochs };

        let classify_config = ClassifyConfig {
            num_classes,
            lora_rank: rank,
            lora_alpha: alpha,
            learning_rate: lr,
            epochs,
            batch_size,
            gradient_clip_norm: Some(clip),
            class_weights,
            quantize_nf4: auto_nf4,
            ..ClassifyConfig::default()
        };

        let trial_dir = apply.output_dir.join(format!("trial_{:03}", trial_idx + 1));
        let trial_start = std::time::Instant::now();

        eprintln!(
            "  Trial {}/{}: lr={:.2e} rank={} alpha={:.1} batch={} warmup={:.2} clip={:.1} weights={}",
            trial_idx + 1, budget, lr, rank, alpha, batch_size, warmup, clip, weights_strategy
        );

        // ── Execute trial ──────────────────────────────────────────────
        let trial_result = run_single_trial_with_warmup(
            &apply.model_path,
            &apply.data_path,
            &trial_dir,
            &model_config,
            classify_config,
            epochs,
            warmup,
            lr_min_ratio,
            &plan.model.size,
        );

        let trial_time_ms = trial_start.elapsed().as_millis() as u64;

        match trial_result {
            Ok(result) => {
                let val_loss = f64::from(result.best_val_loss);
                let val_accuracy = result
                    .epoch_metrics
                    .get(result.best_epoch)
                    .map_or(0.0, |m| f64::from(m.val_accuracy));

                // ── Check scheduler for early stopping ─────────────────
                let was_pruned = scheduler.should_stop(trial_idx, result.best_epoch, val_loss);

                let status = resolve_trial_status(was_pruned, result.stopped_early);

                let summary = TrialSummary {
                    id: trial_idx,
                    val_loss,
                    val_accuracy,
                    train_loss: result
                        .epoch_metrics
                        .last()
                        .map_or(0.0, |m| f64::from(m.train_loss)),
                    train_accuracy: result
                        .epoch_metrics
                        .last()
                        .map_or(0.0, |m| f64::from(m.train_accuracy)),
                    epochs_run: result.epoch_metrics.len(),
                    time_ms: trial_time_ms,
                    config: suggestion.config.clone(),
                    status: status.to_string(),
                };

                eprintln!(
                    "    => val_loss={:.4} val_acc={:.1}% epochs={} [{}]",
                    val_loss,
                    val_accuracy * 100.0,
                    result.epoch_metrics.len(),
                    status,
                );

                tracker.log_hpo_trial(&suggestion.config, &result, was_pruned);

                // Record for Bayesian learner
                searcher.record(suggestion.clone(), val_loss, result.epoch_metrics.len());
                tuner.record_trial(summary.clone());

                if let Some(cb) = apply.on_trial_complete {
                    cb(trial_idx, budget, &summary);
                }
            }
            Err(e) => {
                eprintln!("    => FAILED: {e}");
                tracker.log_failed_trial();

                let summary = TrialSummary {
                    id: trial_idx,
                    val_loss: f64::INFINITY,
                    val_accuracy: 0.0,
                    train_loss: f64::INFINITY,
                    train_accuracy: 0.0,
                    epochs_run: 0,
                    time_ms: trial_time_ms,
                    config: suggestion.config.clone(),
                    status: "failed".to_string(),
                };
                searcher.record(suggestion, f64::INFINITY, 0);
                tuner.record_trial(summary);
            }
        }
    }

    let total_time_ms = total_start.elapsed().as_millis() as u64;

    // Save leaderboard
    let result = tuner.into_result(total_time_ms);
    let leaderboard_path = apply.output_dir.join("leaderboard.json");
    let _ = std::fs::write(
        &leaderboard_path,
        serde_json::to_string_pretty(&result).unwrap_or_default(),
    );

    Ok(result)
}

/// Execute a single training trial with explicit warmup/LR min parameters.
fn run_single_trial_with_warmup(
    model_path: &std::path::Path,
    data_path: &std::path::Path,
    checkpoint_dir: &std::path::Path,
    model_config: &crate::transformer::TransformerConfig,
    classify_config: super::classify_pipeline::ClassifyConfig,
    epochs: usize,
    warmup_fraction: f32,
    lr_min_ratio: f32,
    model_name: &str,
) -> crate::Result<super::classify_trainer::TrainResult> {
    use super::classify_pipeline::ClassifyPipeline;
    use super::classify_trainer::{ClassifyTrainer, TrainingConfig};

    // Create checkpoint directory
    std::fs::create_dir_all(checkpoint_dir).map_err(|e| {
        crate::Error::Io(format!(
            "Failed to create checkpoint dir {}: {e}",
            checkpoint_dir.display()
        ))
    })?;

    // Load pipeline with pretrained weights
    let pipeline = ClassifyPipeline::from_pretrained(model_path, model_config, classify_config)?;

    // Load corpus
    let samples = pipeline.load_corpus(data_path)?;

    let lr_min = pipeline.config.learning_rate * lr_min_ratio;

    // Build training config
    let training_config = TrainingConfig {
        epochs,
        val_split: 0.2,
        save_every: 1,
        early_stopping_patience: 5,
        checkpoint_dir: checkpoint_dir.to_path_buf(),
        seed: 42,
        log_interval: 1,
        warmup_fraction,
        lr_min,
        ..TrainingConfig::default()
    };

    // Create trainer
    let mut trainer = ClassifyTrainer::new(pipeline, samples, training_config)?;

    // Attach monitor writer
    let experiment_id = format!(
        "trial-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let writer =
        crate::monitor::tui::TrainingStateWriter::new(checkpoint_dir, &experiment_id, model_name);
    trainer.set_monitor_writer(writer);

    // Run training
    Ok(trainer.train())
}

/// Resolve class weights from strategy name and class counts.
pub(crate) fn resolve_class_weights(
    strategy: &str,
    class_counts: &[usize],
    num_classes: usize,
) -> Option<Vec<f32>> {
    use super::classification::{compute_class_weights, ClassWeightStrategy, SafetyCorpusStats};

    match strategy {
        "uniform" => None,
        "inverse_freq" => {
            let stats = SafetyCorpusStats {
                total: class_counts.iter().sum(),
                class_counts: class_counts.to_vec(),
                avg_input_len: 0,
            };
            Some(compute_class_weights(&stats, ClassWeightStrategy::InverseFreq, num_classes))
        }
        "sqrt_inverse" => {
            let stats = SafetyCorpusStats {
                total: class_counts.iter().sum(),
                class_counts: class_counts.to_vec(),
                avg_input_len: 0,
            };
            Some(compute_class_weights(&stats, ClassWeightStrategy::SqrtInverse, num_classes))
        }
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Display helpers (for CLI consumption)
// ═══════════════════════════════════════════════════════════════════════

impl TrainingPlan {
    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Serialize to YAML.
    pub fn to_yaml(&self) -> String {
        serde_yaml::to_string(self).unwrap_or_default()
    }

    /// Deserialize from a string (auto-detects JSON or YAML).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> crate::Result<Self> {
        // Try JSON first (faster, more common for programmatic use)
        if let Ok(plan) = serde_json::from_str::<TrainingPlan>(s) {
            return Ok(plan);
        }
        // Fall back to YAML
        serde_yaml::from_str::<TrainingPlan>(s).map_err(|e| {
            crate::Error::ConfigError(format!("Failed to parse plan as JSON or YAML: {e}"))
        })
    }

    /// Count pre-flight checks by status.
    pub fn check_counts(&self) -> (usize, usize, usize) {
        let pass = self.pre_flight.iter().filter(|c| c.status == CheckStatus::Pass).count();
        let warn = self.pre_flight.iter().filter(|c| c.status == CheckStatus::Warn).count();
        let fail = self.pre_flight.iter().filter(|c| c.status == CheckStatus::Fail).count();
        (pass, warn, fail)
    }
}

/// Map pruned/stopped_early flags to a status string.
pub(crate) fn resolve_trial_status(was_pruned: bool, stopped_early: bool) -> &'static str {
    if was_pruned {
        "pruned"
    } else if stopped_early {
        "stopped_early"
    } else {
        "completed"
    }
}

// ── Experiment store integration ──────────────────────────────────────────

/// Thin wrapper around SqliteBackend for logging training experiments.
/// All methods are best-effort (errors silently ignored) so training is
/// never blocked by storage failures.
pub(crate) struct ExperimentTracker {
    pub(crate) store: Option<crate::storage::SqliteBackend>,
    pub(crate) exp_id: Option<String>,
}

impl ExperimentTracker {
    pub(crate) fn open(output_dir: &std::path::Path, plan: &TrainingPlan) -> Self {
        use crate::storage::{ExperimentStorage, SqliteBackend};

        let mut store = SqliteBackend::open_project(output_dir).ok();
        let exp_id = store.as_mut().and_then(|s| {
            let config_json = serde_json::json!({
                "model": &plan.model.architecture,
                "size": &plan.model.size,
                "strategy": &plan.hyperparameters.strategy,
                "budget": plan.hyperparameters.budget,
                "num_classes": plan.data.class_counts.len(),
            });
            s.create_experiment(&plan.model.architecture, Some(config_json)).ok()
        });
        Self { store, exp_id }
    }

    fn log_manual_trial(
        &mut self,
        manual: &ManualConfig,
        result: &super::classify_trainer::TrainResult,
    ) {
        use crate::storage::{ExperimentStorage, ParameterValue as SPV};
        let (store, eid) = match (self.store.as_mut(), self.exp_id.as_ref()) {
            (Some(s), Some(e)) => (s, e),
            _ => return,
        };
        let run_id = match store.create_run(eid) {
            Ok(id) => id,
            Err(_) => return,
        };
        let _ = store.start_run(&run_id);
        let _ =
            store.log_param(&run_id, "learning_rate", SPV::Float(f64::from(manual.learning_rate)));
        let _ = store.log_param(&run_id, "lora_rank", SPV::Int(manual.lora_rank as i64));
        let _ = store.log_param(&run_id, "batch_size", SPV::Int(manual.batch_size as i64));
        Self::log_epoch_metrics(store, &run_id, &result.epoch_metrics);
        let _ = store.complete_run(&run_id, crate::storage::RunStatus::Success);
    }

    fn log_hpo_trial(
        &mut self,
        config: &std::collections::HashMap<String, crate::optim::ParameterValue>,
        result: &super::classify_trainer::TrainResult,
        was_pruned: bool,
    ) {
        use crate::optim::ParameterValue as OPV;
        use crate::storage::{ExperimentStorage, ParameterValue as SPV};
        let (store, eid) = match (self.store.as_mut(), self.exp_id.as_ref()) {
            (Some(s), Some(e)) => (s, e),
            _ => return,
        };
        let run_id = match store.create_run(eid) {
            Ok(id) => id,
            Err(_) => return,
        };
        let _ = store.start_run(&run_id);
        for (k, v) in config {
            let sv = match v {
                OPV::Float(f) => SPV::Float(*f),
                OPV::Int(i) => SPV::Int(*i),
                OPV::Categorical(s) => SPV::String(s.clone()),
            };
            let _ = store.log_param(&run_id, k, sv);
        }
        Self::log_epoch_metrics(store, &run_id, &result.epoch_metrics);
        let status = if was_pruned {
            crate::storage::RunStatus::Cancelled
        } else {
            crate::storage::RunStatus::Success
        };
        let _ = store.complete_run(&run_id, status);
    }

    pub(crate) fn log_failed_trial(&mut self) {
        use crate::storage::ExperimentStorage;
        let (store, eid) = match (self.store.as_mut(), self.exp_id.as_ref()) {
            (Some(s), Some(e)) => (s, e),
            _ => return,
        };
        if let Ok(run_id) = store.create_run(eid) {
            let _ = store.start_run(&run_id);
            let _ = store.complete_run(&run_id, crate::storage::RunStatus::Failed);
        }
    }

    fn log_epoch_metrics(
        store: &mut crate::storage::SqliteBackend,
        run_id: &str,
        epochs: &[super::classify_trainer::EpochMetrics],
    ) {
        use crate::storage::ExperimentStorage;
        for (i, epoch) in epochs.iter().enumerate() {
            let _ = store.log_metric(run_id, "train_loss", i as u64, f64::from(epoch.train_loss));
            let _ = store.log_metric(run_id, "val_loss", i as u64, f64::from(epoch.val_loss));
            let _ =
                store.log_metric(run_id, "val_accuracy", i as u64, f64::from(epoch.val_accuracy));
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_missing_data_file() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/nonexistent/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: PathBuf::from("/tmp/test-plan-out"),
            strategy: "manual".to_string(),
            budget: 10,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let result = plan(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_plan_manual_strategy_warns() {
        // Create a temp JSONL file
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..50 {
            lines.push(format!(r#"{{"input": "echo test {i}", "label": {}}}"#, i % 5));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 10,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).expect("valid");
        assert_eq!(p.hyperparameters.strategy, "manual");
        assert!(p.issues.iter().any(|i| i.category == "Hyperparameters"));
        assert!(p.hyperparameters.recommendation.is_some());
    }

    #[test]
    fn test_plan_tpe_strategy_generates_previews() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..100 {
            lines.push(format!(r#"{{"input": "echo test {i}", "label": {}}}"#, i % 5));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "tpe".to_string(),
            budget: 20,
            scout: true,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).expect("valid");
        assert_eq!(p.hyperparameters.strategy, "tpe");
        assert_eq!(p.hyperparameters.budget, 20);
        assert!(p.hyperparameters.scout);
        assert!(!p.hyperparameters.sample_configs.is_empty());
        assert_eq!(p.hyperparameters.search_space_params, 9);
    }

    #[test]
    fn test_plan_detects_imbalance() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        // 80 class 0, 10 each for classes 1-4 = 8:1 imbalance
        for i in 0..80 {
            lines.push(format!(r#"{{"input": "safe command {i}", "label": 0}}"#));
        }
        for c in 1..5 {
            for i in 0..10 {
                lines.push(format!(r#"{{"input": "class {c} cmd {i}", "label": {c}}}"#));
            }
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "tpe".to_string(),
            budget: 10,
            scout: true,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).expect("valid");
        assert!(p.data.imbalance_ratio > 5.0);
        assert!(p.data.auto_class_weights);
        assert!(p.issues.iter().any(|i| i.message.contains("imbalance")));
    }

    #[test]
    fn test_plan_detects_duplicates() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..50 {
            lines.push(format!(r#"{{"input": "echo test {i}", "label": {}}}"#, i % 5));
        }
        // Add 5 exact duplicates
        for _ in 0..5 {
            lines.push(r#"{"input": "echo test 0", "label": 0}"#.to_string());
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).expect("valid");
        assert!(p.data.duplicates > 0);
        assert!(p.issues.iter().any(|i| i.message.contains("duplicate")));
    }

    #[test]
    fn test_plan_serialization_roundtrip() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..20 {
            lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 5));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "tpe".to_string(),
            budget: 5,
            scout: true,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).expect("valid");

        // JSON roundtrip
        let json = p.to_json();
        let deserialized: TrainingPlan = serde_json::from_str(&json).expect("valid");
        assert_eq!(deserialized.version, "1.0");
        assert_eq!(deserialized.task, "classify");
        assert_eq!(deserialized.data.train_samples, p.data.train_samples);

        // YAML roundtrip
        let yaml = p.to_yaml();
        let deserialized_yaml: TrainingPlan = serde_yaml::from_str(&yaml).expect("valid");
        assert_eq!(deserialized_yaml.data.train_samples, p.data.train_samples);
    }

    #[test]
    fn test_plan_resource_estimation() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..100 {
            lines.push(format!(r#"{{"input": "echo test {i}", "label": {}}}"#, i % 5));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).expect("valid");
        assert!(p.resources.estimated_vram_gb > 0.0);
        assert!(p.resources.steps_per_epoch > 0);
        assert!(p.resources.estimated_checkpoint_mb > 0.0);
    }

    #[test]
    fn test_plan_verdict_ready() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..100 {
            lines.push(format!(r#"{{"input": "echo test {i}", "label": {}}}"#, i % 5));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "tpe".to_string(),
            budget: 20,
            scout: true,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).expect("valid");
        // Should be WarningsPresent (model_path not specified)
        assert_ne!(p.verdict, PlanVerdict::Blocked);
    }

    #[test]
    fn test_plan_model_info_qwen2() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..20 {
            lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 5));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).expect("valid");
        assert_eq!(p.model.hidden_size, 896);
        assert_eq!(p.model.num_layers, 24);
        assert_eq!(p.model.architecture, "qwen2");
        assert!(p.model.lora_trainable_params > 0);
        assert!(p.model.classifier_params > 0);
    }

    #[test]
    fn test_execute_plan_rejects_blocked() {
        let blocked_plan = TrainingPlan {
            version: "1.0".to_string(),
            task: "classify".to_string(),
            data: DataAudit {
                train_path: "/tmp/nonexistent.jsonl".to_string(),
                train_samples: 0,
                avg_input_len: 0,
                class_counts: vec![0; 5],
                imbalance_ratio: 1.0,
                auto_class_weights: false,
                val_samples: None,
                test_samples: None,
                duplicates: 0,
                preamble_count: 0,
            },
            model: ModelInfo {
                size: "0.5B".to_string(),
                hidden_size: 896,
                num_layers: 24,
                architecture: "qwen2".to_string(),
                weights_available: false,
                lora_trainable_params: 0,
                classifier_params: 0,
            },
            hyperparameters: HyperparameterPlan {
                strategy: "manual".to_string(),
                budget: 0,
                scout: false,
                max_epochs: 1,
                search_space_params: 0,
                sample_configs: Vec::new(),
                manual: None,
                recommendation: None,
            },
            resources: ResourceEstimate {
                estimated_vram_gb: 0.0,
                estimated_minutes_per_epoch: 0.0,
                estimated_total_minutes: 0.0,
                estimated_checkpoint_mb: 0.0,
                steps_per_epoch: 0,
                gpu_device: None,
            },
            pre_flight: vec![PreFlightCheck {
                name: "data_file".to_string(),
                status: CheckStatus::Fail,
                detail: "Data not found".to_string(),
            }],
            output_dir: "/tmp/test".to_string(),
            auto_diagnose: false,
            verdict: PlanVerdict::Blocked,
            issues: Vec::new(),
        };

        let apply = ApplyConfig {
            model_path: PathBuf::from("/tmp/nonexistent"),
            data_path: PathBuf::from("/tmp/nonexistent.jsonl"),
            output_dir: PathBuf::from("/tmp/test-apply"),
            on_trial_complete: None,
        };

        let result = execute_plan(&blocked_plan, &apply);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("blocked"), "Error should mention blocked: {err_msg}");
    }

    #[test]
    fn test_resolve_class_weights_uniform() {
        let weights = resolve_class_weights("uniform", &[100, 200, 300], 3);
        assert!(weights.is_none());
    }

    #[test]
    fn test_resolve_class_weights_sqrt_inverse() {
        let weights = resolve_class_weights("sqrt_inverse", &[100, 200, 300], 3);
        assert!(weights.is_some());
        let w = weights.expect("valid");
        assert_eq!(w.len(), 3);
        // Largest class should have smallest weight
        assert!(
            w[0] > w[2],
            "class 0 (100 samples) should have higher weight than class 2 (300 samples)"
        );
    }

    #[test]
    fn test_resolve_class_weights_inverse_freq() {
        let weights = resolve_class_weights("inverse_freq", &[100, 200, 300], 3);
        assert!(weights.is_some());
        let w = weights.expect("valid");
        assert_eq!(w.len(), 3);
        assert!(w[0] > w[2]);
    }

    #[test]
    fn test_resolve_class_weights_unknown() {
        let weights = resolve_class_weights("bogus", &[100, 200], 2);
        assert!(weights.is_none());
    }

    #[test]
    fn test_execute_plan_rejects_missing_model_path() {
        let plan = TrainingPlan {
            version: "1.0".to_string(),
            task: "classify".to_string(),
            data: DataAudit {
                train_path: "/tmp/data.jsonl".to_string(),
                train_samples: 100,
                avg_input_len: 50,
                class_counts: vec![50, 50],
                imbalance_ratio: 1.0,
                auto_class_weights: false,
                val_samples: None,
                test_samples: None,
                duplicates: 0,
                preamble_count: 0,
            },
            model: ModelInfo {
                size: "0.5B".to_string(),
                hidden_size: 896,
                num_layers: 24,
                architecture: "qwen2".to_string(),
                weights_available: true,
                lora_trainable_params: 1000,
                classifier_params: 100,
            },
            hyperparameters: HyperparameterPlan {
                strategy: "manual".to_string(),
                budget: 0,
                scout: false,
                max_epochs: 1,
                search_space_params: 0,
                sample_configs: Vec::new(),
                manual: Some(ManualConfig {
                    learning_rate: 1e-4,
                    lora_rank: 16,
                    batch_size: 32,
                    lora_alpha: None,
                    warmup_fraction: None,
                    gradient_clip_norm: None,
                    lr_min_ratio: None,
                    class_weights: None,
                    target_modules: None,
                }),
                recommendation: None,
            },
            resources: ResourceEstimate {
                estimated_vram_gb: 2.5,
                estimated_minutes_per_epoch: 1.0,
                estimated_total_minutes: 1.0,
                estimated_checkpoint_mb: 1.0,
                steps_per_epoch: 4,
                gpu_device: None,
            },
            pre_flight: Vec::new(),
            output_dir: "/tmp/test".to_string(),
            auto_diagnose: false,
            verdict: PlanVerdict::Ready,
            issues: Vec::new(),
        };

        let apply = ApplyConfig {
            model_path: PathBuf::from("/tmp/definitely-not-a-real-model-dir"),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            output_dir: PathBuf::from("/tmp/test-apply"),
            on_trial_complete: None,
        };

        let result = execute_plan(&plan, &apply);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Model path") || err_msg.contains("does not exist"),
            "Error should mention model path: {err_msg}"
        );
    }

    // ── resolve_trial_status tests ──────────────────────────────────────

    #[test]
    fn test_resolve_trial_status_completed() {
        assert_eq!(resolve_trial_status(false, false), "completed");
    }

    #[test]
    fn test_resolve_trial_status_pruned() {
        assert_eq!(resolve_trial_status(true, false), "pruned");
    }

    #[test]
    fn test_resolve_trial_status_stopped_early() {
        assert_eq!(resolve_trial_status(false, true), "stopped_early");
    }

    #[test]
    fn test_resolve_trial_status_pruned_takes_priority() {
        // When both pruned and stopped_early, pruned wins
        assert_eq!(resolve_trial_status(true, true), "pruned");
    }

    // ── estimate_gpu_hours tests ────────────────────────────────────────

    #[test]
    fn test_estimate_gpu_hours_basic() {
        let hours = estimate_gpu_hours(128, 1, 1);
        // 128 samples / 64 batch = 2 steps, 2 * 58 = 116 seconds / 3600 ~ 0.032 hours
        assert!(hours > 0.0);
        assert!(hours < 1.0);
    }

    #[test]
    fn test_estimate_gpu_hours_scales_with_budget() {
        let h1 = estimate_gpu_hours(100, 1, 1);
        let h10 = estimate_gpu_hours(100, 1, 10);
        assert!((h10 - h1 * 10.0).abs() < 1e-6, "Budget should scale linearly");
    }

    #[test]
    fn test_estimate_gpu_hours_scales_with_epochs() {
        let h1 = estimate_gpu_hours(100, 1, 1);
        let h5 = estimate_gpu_hours(100, 5, 1);
        assert!((h5 - h1 * 5.0).abs() < 1e-6, "Epochs should scale linearly");
    }

    #[test]
    fn test_estimate_gpu_hours_zero_budget() {
        let hours = estimate_gpu_hours(100, 5, 0);
        assert!((hours).abs() < 1e-10);
    }

    // ── resolve_model tests ─────────────────────────────────────────────

    #[test]
    fn test_resolve_model_qwen2_05b() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut pf = Vec::new();
        let model = resolve_model(&config, &mut pf);
        assert_eq!(model.hidden_size, 896);
        assert_eq!(model.num_layers, 24);
        assert_eq!(model.architecture, "qwen2");
        assert!(!model.weights_available);
    }

    #[test]
    fn test_resolve_model_500m_alias() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "500M".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut pf = Vec::new();
        let model = resolve_model(&config, &mut pf);
        assert_eq!(model.hidden_size, 896);
        assert_eq!(model.architecture, "qwen2");
    }

    #[test]
    fn test_resolve_model_9b() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "9B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut pf = Vec::new();
        let model = resolve_model(&config, &mut pf);
        assert_eq!(model.hidden_size, 4096);
        assert_eq!(model.num_layers, 48);
        assert_eq!(model.architecture, "qwen3.5");
    }

    #[test]
    fn test_resolve_model_7b() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "7B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut pf = Vec::new();
        let model = resolve_model(&config, &mut pf);
        assert_eq!(model.hidden_size, 4096);
        assert_eq!(model.num_layers, 32);
        assert_eq!(model.architecture, "llama2");
    }

    #[test]
    fn test_resolve_model_13b() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "13B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut pf = Vec::new();
        let model = resolve_model(&config, &mut pf);
        assert_eq!(model.hidden_size, 5120);
        assert_eq!(model.num_layers, 40);
        assert_eq!(model.architecture, "llama2");
    }

    #[test]
    fn test_resolve_model_unknown_size() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "99B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut pf = Vec::new();
        let model = resolve_model(&config, &mut pf);
        assert_eq!(model.architecture, "unknown");
        assert_eq!(model.hidden_size, 896); // defaults to smallest
    }

    #[test]
    fn test_resolve_model_with_model_path_dir() {
        // model_path pointing to an existing directory
        let dir = tempfile::tempdir().unwrap();
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: Some(dir.path().to_path_buf()),
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut pf = Vec::new();
        let model = resolve_model(&config, &mut pf);
        assert!(model.weights_available);
        // Should warn about missing files (no model.safetensors, no tokenizer.json)
        assert!(pf.iter().any(|c| c.name == "model_weights" && c.status == CheckStatus::Warn));
    }

    #[test]
    fn test_resolve_model_with_nonexistent_path() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: Some(PathBuf::from("/nonexistent/model/dir")),
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut pf = Vec::new();
        let model = resolve_model(&config, &mut pf);
        assert!(!model.weights_available);
        assert!(pf.iter().any(|c| c.name == "model_weights" && c.status == CheckStatus::Fail));
    }

    #[test]
    fn test_resolve_model_with_complete_model_dir() {
        // model_path with model.safetensors and tokenizer.json
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: Some(dir.path().to_path_buf()),
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut pf = Vec::new();
        let model = resolve_model(&config, &mut pf);
        assert!(model.weights_available);
        assert!(pf.iter().any(|c| c.name == "model_weights" && c.status == CheckStatus::Pass));
    }

    #[test]
    fn test_resolve_model_lora_params_calculation() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: Some(8),
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut pf = Vec::new();
        let model = resolve_model(&config, &mut pf);
        // lora_trainable = 2 * rank * hidden * 2 * layers = 2 * 8 * 896 * 2 * 24
        assert_eq!(model.lora_trainable_params, 2 * 8 * 896 * 2 * 24);
        // classifier_params = hidden * num_classes + num_classes = 896 * 5 + 5
        assert_eq!(model.classifier_params, 896 * 5 + 5);
    }

    // ── estimate_resources tests ────────────────────────────────────────

    #[test]
    fn test_estimate_resources_05b() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 3,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let model = ModelInfo {
            size: "0.5B".to_string(),
            hidden_size: 896,
            num_layers: 24,
            architecture: "qwen2".to_string(),
            weights_available: false,
            lora_trainable_params: 100_000,
            classifier_params: 4485,
        };
        let data = DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 1000,
            avg_input_len: 50,
            class_counts: vec![500, 500],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        };
        let res = estimate_resources(&config, &model, &data, 64);
        assert!((res.estimated_vram_gb - 2.5).abs() < 0.01);
        assert!(res.steps_per_epoch > 0);
        assert!(res.estimated_minutes_per_epoch > 0.0);
        assert!(res.estimated_total_minutes > 0.0);
        assert!(res.estimated_checkpoint_mb > 0.0);
    }

    #[test]
    fn test_estimate_resources_7b() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "7B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let model = ModelInfo {
            size: "7B".to_string(),
            hidden_size: 4096,
            num_layers: 32,
            architecture: "llama2".to_string(),
            weights_available: false,
            lora_trainable_params: 1_000_000,
            classifier_params: 8194,
        };
        let data = DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 100,
            avg_input_len: 50,
            class_counts: vec![50, 50],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        };
        let res = estimate_resources(&config, &model, &data, 32);
        assert!((res.estimated_vram_gb - 18.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_resources_13b() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "13B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let model = ModelInfo {
            size: "13B".to_string(),
            hidden_size: 5120,
            num_layers: 40,
            architecture: "llama2".to_string(),
            weights_available: false,
            lora_trainable_params: 2_000_000,
            classifier_params: 10242,
        };
        let data = DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 100,
            avg_input_len: 50,
            class_counts: vec![50, 50],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        };
        let res = estimate_resources(&config, &model, &data, 32);
        assert!((res.estimated_vram_gb - 26.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_resources_unknown_hidden_size() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "custom".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let model = ModelInfo {
            size: "custom".to_string(),
            hidden_size: 2048,
            num_layers: 16,
            architecture: "custom".to_string(),
            weights_available: false,
            lora_trainable_params: 500_000,
            classifier_params: 4098,
        };
        let data = DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 100,
            avg_input_len: 50,
            class_counts: vec![50, 50],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        };
        let res = estimate_resources(&config, &model, &data, 32);
        // Unknown hidden size falls through to default 3.0 GB
        assert!((res.estimated_vram_gb - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_resources_scout_mode() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "tpe".to_string(),
            budget: 10,
            scout: true,
            max_epochs: 5,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let model = ModelInfo {
            size: "0.5B".to_string(),
            hidden_size: 896,
            num_layers: 24,
            architecture: "qwen2".to_string(),
            weights_available: false,
            lora_trainable_params: 100_000,
            classifier_params: 1794,
        };
        let data = DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 200,
            avg_input_len: 50,
            class_counts: vec![100, 100],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        };
        let res = estimate_resources(&config, &model, &data, 64);
        // Scout: total_epochs = 1, total_trials = 10
        // Non-scout with same config would be 5 * 10 = 50
        let non_scout_config = PlanConfig { scout: false, ..config.clone() };
        let res_full = estimate_resources(&non_scout_config, &model, &data, 64);
        assert!(res.estimated_total_minutes < res_full.estimated_total_minutes);
    }

    // ── count_file_samples tests ────────────────────────────────────────

    #[test]
    fn test_count_file_samples_none() {
        assert!(count_file_samples(None, 2).is_none());
    }

    #[test]
    fn test_count_file_samples_nonexistent() {
        let p = PathBuf::from("/nonexistent/file.jsonl");
        assert!(count_file_samples(Some(&p), 2).is_none());
    }

    #[test]
    fn test_count_file_samples_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("val.jsonl");
        let mut lines = Vec::new();
        for i in 0..10 {
            lines.push(format!(r#"{{"input": "test {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&path, lines.join("\n")).unwrap();
        let count = count_file_samples(Some(&path), 2);
        assert_eq!(count, Some(10));
    }

    // ── build_hpo_plan tests ────────────────────────────────────────────

    #[test]
    fn test_build_hpo_plan_manual() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 5,
            manual_lr: Some(2e-5),
            manual_lora_rank: Some(8),
            manual_batch_size: Some(16),
            manual_lora_alpha: Some(16.0),
            manual_warmup: Some(0.05),
            manual_gradient_clip: Some(0.5),
            manual_lr_min_ratio: Some(0.001),
            manual_class_weights: Some("sqrt_inverse".to_string()),
            manual_target_modules: Some("qkv".to_string()),
        };
        let mut issues = Vec::new();
        let hpo = build_hpo_plan(&config, 1000, &mut issues);
        assert_eq!(hpo.strategy, "manual");
        assert_eq!(hpo.budget, 0);
        assert!(!hpo.scout);
        assert_eq!(hpo.max_epochs, 5);
        assert_eq!(hpo.search_space_params, 0);
        assert!(hpo.sample_configs.is_empty());
        let manual = hpo.manual.unwrap();
        assert!((manual.learning_rate - 2e-5).abs() < 1e-10);
        assert_eq!(manual.lora_rank, 8);
        assert_eq!(manual.batch_size, 16);
        assert_eq!(manual.lora_alpha, Some(16.0));
        assert_eq!(manual.warmup_fraction, Some(0.05));
        assert_eq!(manual.gradient_clip_norm, Some(0.5));
        assert_eq!(manual.lr_min_ratio, Some(0.001));
        assert_eq!(manual.class_weights.as_deref(), Some("sqrt_inverse"));
        assert_eq!(manual.target_modules.as_deref(), Some("qkv"));
        // Should have a warning about manual mode
        assert!(issues.iter().any(|i| i.category == "Hyperparameters"));
        assert!(hpo.recommendation.is_some());
    }

    #[test]
    fn test_build_hpo_plan_manual_defaults() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut issues = Vec::new();
        let hpo = build_hpo_plan(&config, 100, &mut issues);
        let manual = hpo.manual.unwrap();
        // Defaults: lr=1e-4, rank=16, batch=32
        assert!((manual.learning_rate - 1e-4).abs() < 1e-10);
        assert_eq!(manual.lora_rank, 16);
        assert_eq!(manual.batch_size, 32);
    }

    #[test]
    fn test_build_hpo_plan_tpe() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "tpe".to_string(),
            budget: 20,
            scout: true,
            max_epochs: 5,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut issues = Vec::new();
        let hpo = build_hpo_plan(&config, 1000, &mut issues);
        assert_eq!(hpo.strategy, "tpe");
        assert_eq!(hpo.budget, 20);
        assert!(hpo.scout);
        assert_eq!(hpo.max_epochs, 1); // scout mode forces 1 epoch
        assert_eq!(hpo.search_space_params, 9);
        assert!(!hpo.sample_configs.is_empty());
        assert!(hpo.manual.is_none());
    }

    #[test]
    fn test_build_hpo_plan_grid() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "grid".to_string(),
            budget: 10,
            scout: false,
            max_epochs: 3,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut issues = Vec::new();
        let hpo = build_hpo_plan(&config, 100, &mut issues);
        assert_eq!(hpo.strategy, "grid");
        assert_eq!(hpo.max_epochs, 3);
    }

    #[test]
    fn test_build_hpo_plan_random() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "random".to_string(),
            budget: 5,
            scout: false,
            max_epochs: 2,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut issues = Vec::new();
        let hpo = build_hpo_plan(&config, 100, &mut issues);
        assert_eq!(hpo.strategy, "random");
    }

    #[test]
    fn test_build_hpo_plan_low_budget_warning() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "tpe".to_string(),
            budget: 3,
            scout: false,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut issues = Vec::new();
        build_hpo_plan(&config, 100, &mut issues);
        assert!(issues
            .iter()
            .any(|i| i.message.contains("TPE budget") && i.message.contains("low")));
    }

    #[test]
    fn test_build_hpo_plan_large_dataset_scout_warning() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "tpe".to_string(),
            budget: 20,
            scout: false,
            max_epochs: 5,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let mut issues = Vec::new();
        build_hpo_plan(&config, 50_000, &mut issues);
        // Should warn about GPU hours for large dataset without scout
        assert!(issues.iter().any(|i| i.message.contains("GPU hours")));
    }

    // ── detect_gpu_device tests ─────────────────────────────────────────

    #[test]
    fn test_detect_gpu_device_returns_option() {
        // Just verify it doesn't panic; result depends on hardware
        let _gpu = detect_gpu_device();
    }

    // ── TrainingPlan from_str tests ─────────────────────────────────────

    #[test]
    fn test_training_plan_from_str_invalid() {
        let result = TrainingPlan::from_str("not valid json or yaml {{{{");
        assert!(result.is_err());
    }

    #[test]
    fn test_training_plan_from_str_json() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..20 {
            lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&data_path, lines.join("\n")).unwrap();
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(8),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).unwrap();
        let json = p.to_json();
        let parsed = TrainingPlan::from_str(&json).unwrap();
        assert_eq!(parsed.task, "classify");
    }

    // ── ExperimentTracker tests ─────────────────────────────────────────

    #[test]
    fn test_experiment_tracker_open() {
        let dir = tempfile::tempdir().unwrap();
        let plan_data = DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 100,
            avg_input_len: 50,
            class_counts: vec![50, 50],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        };
        let test_plan = TrainingPlan {
            version: "1.0".to_string(),
            task: "classify".to_string(),
            data: plan_data,
            model: ModelInfo {
                size: "0.5B".to_string(),
                hidden_size: 896,
                num_layers: 24,
                architecture: "qwen2".to_string(),
                weights_available: false,
                lora_trainable_params: 100_000,
                classifier_params: 1794,
            },
            hyperparameters: HyperparameterPlan {
                strategy: "manual".to_string(),
                budget: 0,
                scout: false,
                max_epochs: 1,
                search_space_params: 0,
                sample_configs: Vec::new(),
                manual: None,
                recommendation: None,
            },
            resources: ResourceEstimate {
                estimated_vram_gb: 2.5,
                estimated_minutes_per_epoch: 1.0,
                estimated_total_minutes: 1.0,
                estimated_checkpoint_mb: 1.0,
                steps_per_epoch: 4,
                gpu_device: None,
            },
            pre_flight: Vec::new(),
            output_dir: dir.path().display().to_string(),
            auto_diagnose: false,
            verdict: PlanVerdict::Ready,
            issues: Vec::new(),
        };
        let tracker = ExperimentTracker::open(dir.path(), &test_plan);
        // Should either succeed or fail gracefully — no panic
        drop(tracker);
    }

    #[test]
    fn test_experiment_tracker_log_failed_trial_no_store() {
        let mut tracker = ExperimentTracker { store: None, exp_id: None };
        // Should be a no-op, not panic
        tracker.log_failed_trial();
    }

    // ── PlanConfig serialization tests ──────────────────────────────────

    #[test]
    fn test_plan_config_serde_roundtrip() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            val_path: Some(PathBuf::from("/tmp/val.jsonl")),
            test_path: Some(PathBuf::from("/tmp/test.jsonl")),
            model_size: "0.5B".to_string(),
            model_path: Some(PathBuf::from("/tmp/model")),
            num_classes: 5,
            output_dir: PathBuf::from("/tmp/out"),
            strategy: "tpe".to_string(),
            budget: 20,
            scout: true,
            max_epochs: 10,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
            manual_lora_alpha: Some(32.0),
            manual_warmup: Some(0.1),
            manual_gradient_clip: Some(1.0),
            manual_lr_min_ratio: Some(0.01),
            manual_class_weights: Some("sqrt_inverse".to_string()),
            manual_target_modules: Some("qkv".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PlanConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.task, "classify");
        assert_eq!(deserialized.budget, 20);
        assert_eq!(deserialized.manual_lr, Some(1e-4));
        assert!(deserialized.val_path.is_some());
        assert!(deserialized.test_path.is_some());
    }

    // ── ManualConfig serde defaults test ────────────────────────────────

    #[test]
    fn test_manual_config_serde_defaults() {
        let json = r#"{"learning_rate": 0.001, "lora_rank": 8, "batch_size": 32}"#;
        let mc: ManualConfig = serde_json::from_str(json).unwrap();
        assert!((mc.learning_rate - 0.001).abs() < 1e-6);
        assert_eq!(mc.lora_rank, 8);
        assert_eq!(mc.batch_size, 32);
        // All optional fields should be None (serde default)
        assert!(mc.lora_alpha.is_none());
        assert!(mc.warmup_fraction.is_none());
        assert!(mc.gradient_clip_norm.is_none());
        assert!(mc.lr_min_ratio.is_none());
        assert!(mc.class_weights.is_none());
        assert!(mc.target_modules.is_none());
    }

    #[test]
    fn test_manual_config_serde_all_fields() {
        let mc = ManualConfig {
            learning_rate: 5e-5,
            lora_rank: 4,
            batch_size: 64,
            lora_alpha: Some(8.0),
            warmup_fraction: Some(0.05),
            gradient_clip_norm: Some(0.5),
            lr_min_ratio: Some(0.001),
            class_weights: Some("inverse_freq".to_string()),
            target_modules: Some("all_linear".to_string()),
        };
        let json = serde_json::to_string(&mc).unwrap();
        let deserialized: ManualConfig = serde_json::from_str(&json).unwrap();
        assert!((deserialized.learning_rate - 5e-5).abs() < 1e-10);
        assert_eq!(deserialized.lora_alpha, Some(8.0));
        assert_eq!(deserialized.class_weights.as_deref(), Some("inverse_freq"));
    }

    // ── PlanIssue tests ─────────────────────────────────────────────────

    #[test]
    fn test_plan_issue_serde() {
        let issue = PlanIssue {
            severity: CheckStatus::Warn,
            category: "Data".to_string(),
            message: "test issue".to_string(),
            fix: Some("do this".to_string()),
        };
        let json = serde_json::to_string(&issue).unwrap();
        let deserialized: PlanIssue = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.severity, CheckStatus::Warn);
        assert_eq!(deserialized.category, "Data");
        assert_eq!(deserialized.fix.as_deref(), Some("do this"));
    }

    #[test]
    fn test_plan_issue_no_fix() {
        let issue = PlanIssue {
            severity: CheckStatus::Fail,
            category: "Model".to_string(),
            message: "error".to_string(),
            fix: None,
        };
        let json = serde_json::to_string(&issue).unwrap();
        let deserialized: PlanIssue = serde_json::from_str(&json).unwrap();
        assert!(deserialized.fix.is_none());
    }

    // ── TrialPreview serde tests ────────────────────────────────────────

    #[test]
    fn test_trial_preview_serde() {
        let tp = TrialPreview {
            trial: 1,
            learning_rate: 1e-4,
            lora_rank: 16,
            lora_alpha: 32.0,
            batch_size: 64,
            warmup: 0.1,
            gradient_clip: 1.0,
            class_weights: "sqrt_inverse".to_string(),
            target_modules: "qv".to_string(),
            lr_min_ratio: 0.01,
        };
        let json = serde_json::to_string(&tp).unwrap();
        let deserialized: TrialPreview = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.trial, 1);
        assert_eq!(deserialized.lora_rank, 16);
        assert!((deserialized.lora_alpha - 32.0).abs() < 1e-6);
    }

    // ── execute_plan missing data test ──────────────────────────────────

    #[test]
    fn test_execute_plan_rejects_missing_data() {
        let plan = TrainingPlan {
            version: "1.0".to_string(),
            task: "classify".to_string(),
            data: DataAudit {
                train_path: "/tmp/data.jsonl".to_string(),
                train_samples: 100,
                avg_input_len: 50,
                class_counts: vec![50, 50],
                imbalance_ratio: 1.0,
                auto_class_weights: false,
                val_samples: None,
                test_samples: None,
                duplicates: 0,
                preamble_count: 0,
            },
            model: ModelInfo {
                size: "0.5B".to_string(),
                hidden_size: 896,
                num_layers: 24,
                architecture: "qwen2".to_string(),
                weights_available: true,
                lora_trainable_params: 1000,
                classifier_params: 100,
            },
            hyperparameters: HyperparameterPlan {
                strategy: "manual".to_string(),
                budget: 0,
                scout: false,
                max_epochs: 1,
                search_space_params: 0,
                sample_configs: Vec::new(),
                manual: Some(ManualConfig {
                    learning_rate: 1e-4,
                    lora_rank: 16,
                    batch_size: 32,
                    lora_alpha: None,
                    warmup_fraction: None,
                    gradient_clip_norm: None,
                    lr_min_ratio: None,
                    class_weights: None,
                    target_modules: None,
                }),
                recommendation: None,
            },
            resources: ResourceEstimate {
                estimated_vram_gb: 2.5,
                estimated_minutes_per_epoch: 1.0,
                estimated_total_minutes: 1.0,
                estimated_checkpoint_mb: 1.0,
                steps_per_epoch: 4,
                gpu_device: None,
            },
            pre_flight: Vec::new(),
            output_dir: "/tmp/test".to_string(),
            auto_diagnose: false,
            verdict: PlanVerdict::Ready,
            issues: Vec::new(),
        };
        // Model path is a directory (use tempdir) but data doesn't exist
        let dir = tempfile::tempdir().unwrap();
        let apply = ApplyConfig {
            model_path: dir.path().to_path_buf(),
            data_path: PathBuf::from("/tmp/nonexistent_data_file.jsonl"),
            output_dir: PathBuf::from("/tmp/test-apply-missing-data"),
            on_trial_complete: None,
        };
        let result = execute_plan(&plan, &apply);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not found") || err_msg.contains("data"),
            "Error should mention data: {err_msg}"
        );
    }

    // ── Plan with preamble detection ────────────────────────────────────

    #[test]
    fn test_plan_detects_preambles() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        // Many entries with shell preamble (> 10% of total)
        for i in 0..20 {
            lines.push(format!("{{\"input\": \"#!/bin/bash\\necho {i}\", \"label\": {}}}", i % 2));
        }
        for i in 0..5 {
            lines.push(format!(r#"{{"input": "echo clean {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&data_path, lines.join("\n")).unwrap();
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(8),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).unwrap();
        assert!(p.data.preamble_count > 0);
        assert!(p.issues.iter().any(|i| i.message.contains("preamble")));
    }

    // ── Plan with small dataset warning ─────────────────────────────────

    #[test]
    fn test_plan_small_dataset_warning() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..30 {
            lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&data_path, lines.join("\n")).unwrap();
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(8),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).unwrap();
        // < 100 samples should generate a warning
        assert!(p.issues.iter().any(|i| i.message.contains("insufficient")));
    }

    // ── Plan output_dir with existing checkpoints ───────────────────────

    #[test]
    fn test_plan_output_dir_existing_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..20 {
            lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&data_path, lines.join("\n")).unwrap();
        // Create a metadata.json in output dir to simulate existing checkpoints
        let output_dir = dir.path().join("output");
        std::fs::create_dir_all(&output_dir).unwrap();
        std::fs::write(output_dir.join("metadata.json"), "{}").unwrap();
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir,
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(8),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).unwrap();
        assert!(p
            .pre_flight
            .iter()
            .any(|c| c.name == "output_dir" && c.status == CheckStatus::Warn));
        // Output dir warning is surfaced in pre_flight; issues may or may not reference it
        assert!(!p.pre_flight.is_empty());
    }

    // ── Plan verdict logic tests ────────────────────────────────────────

    #[test]
    fn test_plan_verdict_blocked_on_empty_class() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        // Only classes 0 and 1, but num_classes = 3 (class 2 is empty)
        for i in 0..50 {
            lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&data_path, lines.join("\n")).unwrap();
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 3,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(8),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).unwrap();
        assert_eq!(p.verdict, PlanVerdict::Blocked);
        assert!(p
            .pre_flight
            .iter()
            .any(|c| c.name == "class_coverage" && c.status == CheckStatus::Fail));
    }

    // ── DataAudit serde test ────────────────────────────────────────────

    #[test]
    fn test_data_audit_serde() {
        let da = DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 500,
            avg_input_len: 42,
            class_counts: vec![300, 200],
            imbalance_ratio: 1.5,
            auto_class_weights: false,
            val_samples: Some(50),
            test_samples: Some(25),
            duplicates: 3,
            preamble_count: 10,
        };
        let json = serde_json::to_string(&da).unwrap();
        let deserialized: DataAudit = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.train_samples, 500);
        assert_eq!(deserialized.val_samples, Some(50));
        assert_eq!(deserialized.test_samples, Some(25));
        assert_eq!(deserialized.duplicates, 3);
    }

    // ── ResourceEstimate serde test ─────────────────────────────────────

    #[test]
    fn test_resource_estimate_serde() {
        let re = ResourceEstimate {
            estimated_vram_gb: 6.5,
            estimated_minutes_per_epoch: 2.0,
            estimated_total_minutes: 100.0,
            estimated_checkpoint_mb: 50.0,
            steps_per_epoch: 32,
            gpu_device: Some("RTX 4090".to_string()),
        };
        let json = serde_json::to_string(&re).unwrap();
        let deserialized: ResourceEstimate = serde_json::from_str(&json).unwrap();
        assert!((deserialized.estimated_vram_gb - 6.5).abs() < 1e-6);
        assert_eq!(deserialized.gpu_device.as_deref(), Some("RTX 4090"));
    }

    // ── CheckStatus equality ────────────────────────────────────────────

    #[test]
    fn test_check_status_equality() {
        assert_eq!(CheckStatus::Pass, CheckStatus::Pass);
        assert_ne!(CheckStatus::Pass, CheckStatus::Warn);
        assert_ne!(CheckStatus::Warn, CheckStatus::Fail);
    }

    // ── PlanVerdict equality ────────────────────────────────────────────

    #[test]
    fn test_plan_verdict_equality() {
        assert_eq!(PlanVerdict::Ready, PlanVerdict::Ready);
        assert_ne!(PlanVerdict::Ready, PlanVerdict::Blocked);
        assert_ne!(PlanVerdict::WarningsPresent, PlanVerdict::Blocked);
    }

    // ── Plan with val_path and test_path ────────────────────────────────

    #[test]
    fn test_plan_with_val_and_test_paths() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("train.jsonl");
        let val_path = dir.path().join("val.jsonl");
        let test_path = dir.path().join("test.jsonl");
        let mut lines = Vec::new();
        for i in 0..50 {
            lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&data_path, lines.join("\n")).unwrap();
        let mut val_lines = Vec::new();
        for i in 0..10 {
            val_lines.push(format!(r#"{{"input": "val {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&val_path, val_lines.join("\n")).unwrap();
        let mut test_lines = Vec::new();
        for i in 0..5 {
            test_lines.push(format!(r#"{{"input": "test {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&test_path, test_lines.join("\n")).unwrap();
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: Some(val_path),
            test_path: Some(test_path),
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(8),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).unwrap();
        assert_eq!(p.data.val_samples, Some(10));
        assert_eq!(p.data.test_samples, Some(5));
    }

    // ── Plan with nonexistent val/test paths ────────────────────────────

    #[test]
    fn test_plan_with_nonexistent_val_test_paths() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..50 {
            lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&data_path, lines.join("\n")).unwrap();
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: Some(PathBuf::from("/nonexistent/val.jsonl")),
            test_path: Some(PathBuf::from("/nonexistent/test.jsonl")),
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 2,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(8),
            manual_batch_size: Some(32),
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        let p = plan(&config).unwrap();
        // Nonexistent paths should result in None
        assert!(p.data.val_samples.is_none());
        assert!(p.data.test_samples.is_none());
    }

    // ── execute_plan manual without manual config ───────────────────────

    #[test]
    fn test_execute_plan_manual_no_manual_config() {
        let dir = tempfile::tempdir().unwrap();
        // Create a data file so data check passes
        let data_path = dir.path().join("data.jsonl");
        let mut lines = Vec::new();
        for i in 0..20 {
            lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
        }
        std::fs::write(&data_path, lines.join("\n")).unwrap();

        let plan = TrainingPlan {
            version: "1.0".to_string(),
            task: "classify".to_string(),
            data: DataAudit {
                train_path: data_path.display().to_string(),
                train_samples: 20,
                avg_input_len: 10,
                class_counts: vec![10, 10],
                imbalance_ratio: 1.0,
                auto_class_weights: false,
                val_samples: None,
                test_samples: None,
                duplicates: 0,
                preamble_count: 0,
            },
            model: ModelInfo {
                size: "0.5B".to_string(),
                hidden_size: 896,
                num_layers: 24,
                architecture: "qwen2".to_string(),
                weights_available: true,
                lora_trainable_params: 100_000,
                classifier_params: 1794,
            },
            hyperparameters: HyperparameterPlan {
                strategy: "manual".to_string(),
                budget: 0,
                scout: false,
                max_epochs: 1,
                search_space_params: 0,
                sample_configs: Vec::new(),
                manual: None, // No manual config!
                recommendation: None,
            },
            resources: ResourceEstimate {
                estimated_vram_gb: 2.5,
                estimated_minutes_per_epoch: 1.0,
                estimated_total_minutes: 1.0,
                estimated_checkpoint_mb: 1.0,
                steps_per_epoch: 1,
                gpu_device: None,
            },
            pre_flight: Vec::new(),
            output_dir: dir.path().display().to_string(),
            auto_diagnose: false,
            verdict: PlanVerdict::Ready,
            issues: Vec::new(),
        };
        let model_dir = dir.path().join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        let apply = ApplyConfig {
            model_path: model_dir,
            data_path,
            output_dir: dir.path().join("out"),
            on_trial_complete: None,
        };
        let result = execute_plan(&plan, &apply);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("manual") || err_msg.contains("Manual"),
            "Error should mention manual config: {err_msg}"
        );
    }
}
