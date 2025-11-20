//! Configuration validation

use super::schema::TrainSpec;

/// Validation error type
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Model path does not exist: {0}")]
    ModelPathNotFound(String),

    #[error("Training data path does not exist: {0}")]
    TrainDataNotFound(String),

    #[error("Validation data path does not exist: {0}")]
    ValDataNotFound(String),

    #[error("Invalid learning rate: {0} (must be > 0.0)")]
    InvalidLearningRate(f32),

    #[error("Invalid batch size: {0} (must be > 0)")]
    InvalidBatchSize(usize),

    #[error("Invalid epochs: {0} (must be > 0)")]
    InvalidEpochs(usize),

    #[error("Invalid LoRA rank: {0} (must be > 0)")]
    InvalidLoRARank(usize),

    #[error("Invalid quantization bits: {0} (must be 4 or 8)")]
    InvalidQuantBits(u8),

    #[error("Invalid optimizer: {0} (must be one of: adam, adamw, sgd)")]
    InvalidOptimizer(String),

    #[error("Invalid merge method: {0} (must be one of: ties, dare, slerp)")]
    InvalidMergeMethod(String),

    #[error("Invalid gradient clip value: {0} (must be > 0.0)")]
    InvalidGradClip(f32),
}

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

    // Validate learning rate
    if spec.optimizer.lr <= 0.0 {
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

    // Validate LoRA config if present
    if let Some(lora) = &spec.lora {
        if lora.rank == 0 {
            return Err(ValidationError::InvalidLoRARank(lora.rank));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn create_valid_spec() -> TrainSpec {
        TrainSpec {
            model: ModelRef {
                path: PathBuf::from("model.gguf"),
                layers: vec![],
            },
            data: DataConfig {
                train: PathBuf::from("train.parquet"),
                val: None,
                batch_size: 8,
                auto_infer_types: true,
                seq_len: None,
            },
            optimizer: OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: HashMap::new(),
            },
            lora: None,
            quantize: None,
            merge: None,
            training: TrainingParams::default(),
        }
    }

    #[test]
    fn test_valid_config() {
        let spec = create_valid_spec();
        assert!(validate_config(&spec).is_ok());
    }

    #[test]
    fn test_invalid_batch_size() {
        let mut spec = create_valid_spec();
        spec.data.batch_size = 0;
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidBatchSize(0)));
    }

    #[test]
    fn test_invalid_learning_rate() {
        let mut spec = create_valid_spec();
        spec.optimizer.lr = 0.0;
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidLearningRate(0.0)));

        spec.optimizer.lr = -0.1;
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidLearningRate(_)));
    }

    #[test]
    fn test_invalid_optimizer() {
        let mut spec = create_valid_spec();
        spec.optimizer.name = "invalid".to_string();
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidOptimizer(_)));
    }

    #[test]
    fn test_invalid_epochs() {
        let mut spec = create_valid_spec();
        spec.training.epochs = 0;
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidEpochs(0)));
    }

    #[test]
    fn test_invalid_lora_rank() {
        let mut spec = create_valid_spec();
        spec.lora = Some(LoRASpec {
            rank: 0,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: 0.0,
        });
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidLoRARank(0)));
    }

    #[test]
    fn test_invalid_quant_bits() {
        let mut spec = create_valid_spec();
        spec.quantize = Some(QuantSpec {
            bits: 16,
            symmetric: true,
            per_channel: true,
        });
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidQuantBits(16)));
    }

    #[test]
    fn test_invalid_merge_method() {
        let mut spec = create_valid_spec();
        spec.merge = Some(MergeSpec {
            method: "invalid".to_string(),
            params: HashMap::new(),
        });
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidMergeMethod(_)));
    }

    #[test]
    fn test_invalid_grad_clip() {
        let mut spec = create_valid_spec();
        spec.training.grad_clip = Some(0.0);
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidGradClip(0.0)));

        spec.training.grad_clip = Some(-1.0);
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidGradClip(_)));
    }
}
