//! Configuration validation logic
//!
//! Validates training specifications for correctness before execution.

use super::error::ValidationError;
use crate::config::schema::TrainSpec;

/// Validate a training specification
///
/// Checks:
/// - File paths exist
/// - Numeric values are in valid ranges
/// - Enums match allowed values
pub fn validate_config(spec: &TrainSpec) -> Result<(), ValidationError> {
    // Validate model path (skip in tests where files may not exist)
    #[cfg(not(test))]
    if !spec.model.path.exists() {
        return Err(ValidationError::ModelPathNotFound(
            spec.model.path.display().to_string(),
        ));
    }

    // Validate data paths
    #[cfg(not(test))]
    {
        if !spec.data.train.exists() {
            return Err(ValidationError::TrainDataNotFound(
                spec.data.train.display().to_string(),
            ));
        }

        if let Some(val_path) = &spec.data.val {
            if !val_path.exists() {
                return Err(ValidationError::ValDataNotFound(
                    val_path.display().to_string(),
                ));
            }
        }
    }

    // Validate batch size
    if spec.data.batch_size == 0 {
        return Err(ValidationError::InvalidBatchSize(spec.data.batch_size));
    }

    // Validate learning rate (must be positive and reasonable)
    if spec.optimizer.lr <= 0.0 || spec.optimizer.lr > 1.0 {
        return Err(ValidationError::InvalidLearningRate(spec.optimizer.lr));
    }

    // Validate optimizer name
    let valid_optimizers = ["adam", "adamw", "sgd"];
    if !valid_optimizers.contains(&spec.optimizer.name.as_str()) {
        return Err(ValidationError::InvalidOptimizer(
            spec.optimizer.name.clone(),
        ));
    }

    // Validate epochs
    if spec.training.epochs == 0 {
        return Err(ValidationError::InvalidEpochs(spec.training.epochs));
    }

    // Validate gradient clipping
    if let Some(grad_clip) = spec.training.grad_clip {
        if grad_clip <= 0.0 {
            return Err(ValidationError::InvalidGradClip(grad_clip));
        }
    }

    // Validate sequence length if specified
    if let Some(seq_len) = spec.data.seq_len {
        if seq_len == 0 {
            return Err(ValidationError::InvalidSeqLen(seq_len));
        }
    }

    // Validate save interval
    if spec.training.save_interval == 0 {
        return Err(ValidationError::InvalidSaveInterval(
            spec.training.save_interval,
        ));
    }

    // Validate LR scheduler if specified
    if let Some(scheduler) = &spec.training.lr_scheduler {
        let valid_schedulers = ["cosine", "linear", "constant"];
        if !valid_schedulers.contains(&scheduler.as_str()) {
            return Err(ValidationError::InvalidLRScheduler(scheduler.clone()));
        }
    }

    // Validate LoRA config if present
    if let Some(lora) = &spec.lora {
        if lora.rank == 0 || lora.rank > 1024 {
            return Err(ValidationError::InvalidLoRARank(lora.rank));
        }
        if lora.alpha <= 0.0 {
            return Err(ValidationError::InvalidLoRAAlpha(lora.alpha));
        }
        if lora.dropout < 0.0 || lora.dropout >= 1.0 {
            return Err(ValidationError::InvalidLoRADropout(lora.dropout));
        }
        if lora.target_modules.is_empty() {
            return Err(ValidationError::EmptyLoRATargets);
        }
    }

    // Validate quantization config if present
    if let Some(quant) = &spec.quantize {
        if quant.bits != 4 && quant.bits != 8 {
            return Err(ValidationError::InvalidQuantBits(quant.bits));
        }
    }

    // Validate merge config if present
    if let Some(merge) = &spec.merge {
        let valid_methods = ["ties", "dare", "slerp"];
        if !valid_methods.contains(&merge.method.as_str()) {
            return Err(ValidationError::InvalidMergeMethod(merge.method.clone()));
        }
    }

    Ok(())
}
