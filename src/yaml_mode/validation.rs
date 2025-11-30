//! Manifest Validation (Poka-yoke)
//!
//! Schema validation catches errors at parse time, not runtime.
//! Implements the Toyota Way's poka-yoke principle of defect prevention at source.

use super::manifest::TrainingManifest;
use thiserror::Error;

/// Validation result type
pub type ValidationResult<T> = Result<T, ManifestError>;

/// Manifest validation errors
#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("Unsupported entrenar version: {0}. Supported versions: 1.0")]
    UnsupportedVersion(String),

    #[error("Empty required field: {0}")]
    EmptyRequiredField(String),

    #[error("Invalid range for {field}: {value} (expected {constraint})")]
    InvalidRange {
        field: String,
        value: String,
        constraint: String,
    },

    #[error("Mutually exclusive fields specified: {field1} and {field2}")]
    MutuallyExclusive { field1: String, field2: String },

    #[error("Invalid split ratios: sum is {sum} (expected 1.0)")]
    InvalidSplitRatios { sum: f64 },

    #[error("Invalid quantization bits: {bits}. Valid values: 2, 4, 8")]
    InvalidQuantBits { bits: u8 },

    #[error("Dependency error: {0}")]
    DependencyError(String),

    #[error("Invalid optimizer: {0}")]
    InvalidOptimizer(String),

    #[error("Invalid scheduler: {0}")]
    InvalidScheduler(String),
}

/// Supported entrenar specification versions
const SUPPORTED_VERSIONS: &[&str] = &["1.0"];

/// Valid optimizer names
const VALID_OPTIMIZERS: &[&str] = &["sgd", "adam", "adamw", "rmsprop", "adagrad", "lamb"];

/// Valid scheduler names
const VALID_SCHEDULERS: &[&str] = &[
    "step",
    "cosine",
    "cosine_annealing",
    "linear",
    "exponential",
    "plateau",
    "one_cycle",
];

/// Valid quantization bit widths
const VALID_QUANT_BITS: &[u8] = &[2, 4, 8];

/// Validate a training manifest
///
/// Performs comprehensive validation including:
/// 1. Version compatibility
/// 2. Required fields presence
/// 3. Type constraints
/// 4. Range constraints
/// 5. Mutual exclusivity
/// 6. Cross-field dependencies
pub fn validate_manifest(manifest: &TrainingManifest) -> ValidationResult<()> {
    // 1. Version validation
    validate_version(&manifest.entrenar)?;

    // 2. Required field validation
    validate_required_fields(manifest)?;

    // 3. Optimizer validation
    if let Some(ref optim) = manifest.optimizer {
        validate_optimizer(optim)?;
    }

    // 4. Scheduler validation
    if let Some(ref sched) = manifest.scheduler {
        validate_scheduler(sched)?;
    }

    // 5. Training config validation
    if let Some(ref training) = manifest.training {
        validate_training(training)?;
    }

    // 6. Data config validation
    if let Some(ref data) = manifest.data {
        validate_data(data)?;
    }

    // 7. LoRA validation
    if let Some(ref lora) = manifest.lora {
        validate_lora(lora)?;
    }

    // 8. Quantization validation
    if let Some(ref quant) = manifest.quantize {
        validate_quantize(quant)?;
    }

    Ok(())
}

/// Validate specification version
fn validate_version(version: &str) -> ValidationResult<()> {
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(ManifestError::UnsupportedVersion(version.to_string()));
    }
    Ok(())
}

/// Validate required fields
fn validate_required_fields(manifest: &TrainingManifest) -> ValidationResult<()> {
    if manifest.name.is_empty() {
        return Err(ManifestError::EmptyRequiredField("name".to_string()));
    }

    if manifest.version.is_empty() {
        return Err(ManifestError::EmptyRequiredField("version".to_string()));
    }

    Ok(())
}

/// Validate optimizer configuration
fn validate_optimizer(optim: &super::manifest::OptimizerConfig) -> ValidationResult<()> {
    // Validate optimizer name
    let name_lower = optim.name.to_lowercase();
    if !VALID_OPTIMIZERS.contains(&name_lower.as_str()) {
        return Err(ManifestError::InvalidOptimizer(format!(
            "Unknown optimizer '{}'. Valid options: {:?}",
            optim.name, VALID_OPTIMIZERS
        )));
    }

    // Validate learning rate > 0
    if optim.lr <= 0.0 {
        return Err(ManifestError::InvalidRange {
            field: "optimizer.lr".to_string(),
            value: optim.lr.to_string(),
            constraint: "> 0".to_string(),
        });
    }

    // Validate weight_decay >= 0
    if let Some(wd) = optim.weight_decay {
        if wd < 0.0 {
            return Err(ManifestError::InvalidRange {
                field: "optimizer.weight_decay".to_string(),
                value: wd.to_string(),
                constraint: ">= 0".to_string(),
            });
        }
    }

    // Validate betas in (0, 1)
    if let Some(ref betas) = optim.betas {
        for (i, beta) in betas.iter().enumerate() {
            if *beta <= 0.0 || *beta >= 1.0 {
                return Err(ManifestError::InvalidRange {
                    field: format!("optimizer.betas[{i}]"),
                    value: beta.to_string(),
                    constraint: "in (0, 1)".to_string(),
                });
            }
        }
    }

    Ok(())
}

/// Validate scheduler configuration
fn validate_scheduler(sched: &super::manifest::SchedulerConfig) -> ValidationResult<()> {
    let name_lower = sched.name.to_lowercase();
    if !VALID_SCHEDULERS.contains(&name_lower.as_str()) {
        return Err(ManifestError::InvalidScheduler(format!(
            "Unknown scheduler '{}'. Valid options: {:?}",
            sched.name, VALID_SCHEDULERS
        )));
    }

    Ok(())
}

/// Validate training configuration
fn validate_training(training: &super::manifest::TrainingConfig) -> ValidationResult<()> {
    // Check mutual exclusivity of duration options
    let duration_options = [
        training.epochs.is_some(),
        training.max_steps.is_some(),
        training.duration.is_some(),
    ];

    let count = duration_options.iter().filter(|&&x| x).count();
    if count > 1 {
        if training.epochs.is_some() && training.max_steps.is_some() {
            return Err(ManifestError::MutuallyExclusive {
                field1: "training.epochs".to_string(),
                field2: "training.max_steps".to_string(),
            });
        }
        if training.epochs.is_some() && training.duration.is_some() {
            return Err(ManifestError::MutuallyExclusive {
                field1: "training.epochs".to_string(),
                field2: "training.duration".to_string(),
            });
        }
        if training.max_steps.is_some() && training.duration.is_some() {
            return Err(ManifestError::MutuallyExclusive {
                field1: "training.max_steps".to_string(),
                field2: "training.duration".to_string(),
            });
        }
    }

    // Validate epochs > 0
    if let Some(epochs) = training.epochs {
        if epochs == 0 {
            return Err(ManifestError::InvalidRange {
                field: "training.epochs".to_string(),
                value: epochs.to_string(),
                constraint: ">= 1".to_string(),
            });
        }
    }

    // Validate gradient config
    if let Some(ref grad) = training.gradient {
        if let Some(accum) = grad.accumulation_steps {
            if accum == 0 {
                return Err(ManifestError::InvalidRange {
                    field: "training.gradient.accumulation_steps".to_string(),
                    value: accum.to_string(),
                    constraint: ">= 1".to_string(),
                });
            }
        }
    }

    Ok(())
}

/// Validate data configuration
fn validate_data(data: &super::manifest::DataConfig) -> ValidationResult<()> {
    // Validate batch_size > 0
    if let Some(ref loader) = data.loader {
        if loader.batch_size == 0 {
            return Err(ManifestError::InvalidRange {
                field: "data.loader.batch_size".to_string(),
                value: "0".to_string(),
                constraint: ">= 1".to_string(),
            });
        }
    }

    // Validate split ratios sum to 1.0 (with tolerance)
    if let Some(ref split) = data.split {
        let mut sum = split.train;
        if let Some(val) = split.val {
            sum += val;
        }
        if let Some(test) = split.test {
            sum += test;
        }

        // Allow small tolerance for floating point
        if (sum - 1.0).abs() > 0.001 {
            return Err(ManifestError::InvalidSplitRatios { sum });
        }

        // Validate individual ratios in [0, 1]
        if split.train < 0.0 || split.train > 1.0 {
            return Err(ManifestError::InvalidRange {
                field: "data.split.train".to_string(),
                value: split.train.to_string(),
                constraint: "in [0, 1]".to_string(),
            });
        }
    }

    Ok(())
}

/// Validate LoRA configuration
fn validate_lora(lora: &super::manifest::LoraConfig) -> ValidationResult<()> {
    // Only validate if enabled
    if !lora.enabled {
        return Ok(());
    }

    // Target modules required when enabled
    if lora.target_modules.is_empty() && lora.target_modules_pattern.is_none() {
        return Err(ManifestError::EmptyRequiredField(
            "lora.target_modules".to_string(),
        ));
    }

    // Validate rank > 0
    if lora.rank == 0 {
        return Err(ManifestError::InvalidRange {
            field: "lora.rank".to_string(),
            value: "0".to_string(),
            constraint: ">= 1".to_string(),
        });
    }

    // Validate alpha > 0
    if lora.alpha <= 0.0 {
        return Err(ManifestError::InvalidRange {
            field: "lora.alpha".to_string(),
            value: lora.alpha.to_string(),
            constraint: "> 0".to_string(),
        });
    }

    // Validate dropout in [0, 1)
    if let Some(dropout) = lora.dropout {
        if !(0.0..1.0).contains(&dropout) {
            return Err(ManifestError::InvalidRange {
                field: "lora.dropout".to_string(),
                value: dropout.to_string(),
                constraint: "in [0, 1)".to_string(),
            });
        }
    }

    // Validate QLoRA bits
    if let Some(bits) = lora.quantize_bits {
        if !VALID_QUANT_BITS.contains(&bits) {
            return Err(ManifestError::InvalidQuantBits { bits });
        }
    }

    Ok(())
}

/// Validate quantization configuration
fn validate_quantize(quant: &super::manifest::QuantizeConfig) -> ValidationResult<()> {
    // Only validate if enabled
    if !quant.enabled {
        return Ok(());
    }

    // Validate bits
    if !VALID_QUANT_BITS.contains(&quant.bits) {
        return Err(ManifestError::InvalidQuantBits { bits: quant.bits });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_version() {
        assert!(validate_version("1.0").is_ok());
        assert!(validate_version("2.0").is_err());
    }

    #[test]
    fn test_valid_optimizers() {
        for opt in VALID_OPTIMIZERS {
            let optim = super::super::manifest::OptimizerConfig {
                name: opt.to_string(),
                lr: 0.001,
                weight_decay: None,
                betas: None,
                eps: None,
                amsgrad: None,
                momentum: None,
                nesterov: None,
                dampening: None,
                alpha: None,
                centered: None,
                param_groups: None,
            };
            assert!(validate_optimizer(&optim).is_ok(), "Optimizer {} should be valid", opt);
        }
    }

    #[test]
    fn test_valid_quant_bits() {
        for bits in VALID_QUANT_BITS {
            let quant = super::super::manifest::QuantizeConfig {
                enabled: true,
                bits: *bits,
                scheme: None,
                granularity: None,
                group_size: None,
                qat: None,
                calibration: None,
                exclude: None,
            };
            assert!(validate_quantize(&quant).is_ok(), "Quant bits {} should be valid", bits);
        }
    }
}
