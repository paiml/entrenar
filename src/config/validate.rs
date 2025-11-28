//! Configuration validation
//!
//! Validates training specifications for correctness before execution.

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

    #[error("Invalid learning rate: {0} (must be > 0.0 and <= 1.0)")]
    InvalidLearningRate(f32),

    #[error("Invalid batch size: {0} (must be > 0)")]
    InvalidBatchSize(usize),

    #[error("Invalid epochs: {0} (must be > 0)")]
    InvalidEpochs(usize),

    #[error("Invalid LoRA rank: {0} (must be > 0 and <= 1024)")]
    InvalidLoRARank(usize),

    #[error("Invalid LoRA alpha: {0} (must be > 0.0)")]
    InvalidLoRAAlpha(f32),

    #[error("Invalid LoRA dropout: {0} (must be in [0.0, 1.0))")]
    InvalidLoRADropout(f32),

    #[error("Invalid quantization bits: {0} (must be 4 or 8)")]
    InvalidQuantBits(u8),

    #[error("Invalid optimizer: {0} (must be one of: adam, adamw, sgd)")]
    InvalidOptimizer(String),

    #[error("Invalid merge method: {0} (must be one of: ties, dare, slerp)")]
    InvalidMergeMethod(String),

    #[error("Invalid gradient clip value: {0} (must be > 0.0)")]
    InvalidGradClip(f32),

    #[error("Invalid sequence length: {0} (must be > 0)")]
    InvalidSeqLen(usize),

    #[error("Invalid save interval: {0} (must be > 0)")]
    InvalidSaveInterval(usize),

    #[error("LoRA target modules cannot be empty")]
    EmptyLoRATargets,

    #[error("Invalid LR scheduler: {0} (must be one of: cosine, linear, constant)")]
    InvalidLRScheduler(String),
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

    #[test]
    fn test_invalid_lr_too_high() {
        let mut spec = create_valid_spec();
        spec.optimizer.lr = 1.5;
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidLearningRate(_)));
    }

    #[test]
    fn test_invalid_lora_alpha() {
        let mut spec = create_valid_spec();
        spec.lora = Some(LoRASpec {
            rank: 64,
            alpha: 0.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: 0.0,
        });
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidLoRAAlpha(_)));
    }

    #[test]
    fn test_invalid_lora_dropout() {
        let mut spec = create_valid_spec();
        spec.lora = Some(LoRASpec {
            rank: 64,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: 1.0,
        });
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidLoRADropout(_)));
    }

    #[test]
    fn test_empty_lora_targets() {
        let mut spec = create_valid_spec();
        spec.lora = Some(LoRASpec {
            rank: 64,
            alpha: 16.0,
            target_modules: vec![],
            dropout: 0.0,
        });
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::EmptyLoRATargets));
    }

    #[test]
    fn test_invalid_lora_rank_too_high() {
        let mut spec = create_valid_spec();
        spec.lora = Some(LoRASpec {
            rank: 2000,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: 0.0,
        });
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidLoRARank(_)));
    }

    #[test]
    fn test_invalid_seq_len() {
        let mut spec = create_valid_spec();
        spec.data.seq_len = Some(0);
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidSeqLen(0)));
    }

    #[test]
    fn test_invalid_lr_scheduler() {
        let mut spec = create_valid_spec();
        spec.training.lr_scheduler = Some("invalid".to_string());
        let err = validate_config(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidLRScheduler(_)));
    }

    #[test]
    fn test_valid_lr_schedulers() {
        for scheduler in ["cosine", "linear", "constant"] {
            let mut spec = create_valid_spec();
            spec.training.lr_scheduler = Some(scheduler.to_string());
            assert!(validate_config(&spec).is_ok());
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::config::schema::*;
    use proptest::prelude::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn arb_valid_spec() -> impl Strategy<Value = TrainSpec> {
        (
            1usize..256,                        // batch_size
            1e-6f32..1.0,                       // lr
            1usize..100,                        // epochs
            proptest::option::of(0.1f32..10.0), // grad_clip
        )
            .prop_map(|(batch_size, lr, epochs, grad_clip)| TrainSpec {
                model: ModelRef {
                    path: PathBuf::from("model.gguf"),
                    layers: vec![],
                },
                data: DataConfig {
                    train: PathBuf::from("train.parquet"),
                    val: None,
                    batch_size,
                    auto_infer_types: true,
                    seq_len: None,
                },
                optimizer: OptimSpec {
                    name: "adam".to_string(),
                    lr,
                    params: HashMap::new(),
                },
                lora: None,
                quantize: None,
                merge: None,
                training: TrainingParams {
                    epochs,
                    grad_clip,
                    lr_scheduler: None,
                    warmup_steps: 0,
                    save_interval: 1,
                    output_dir: PathBuf::from("./checkpoints"),
                },
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_valid_spec_passes(spec in arb_valid_spec()) {
            prop_assert!(validate_config(&spec).is_ok());
        }

        #[test]
        fn prop_zero_batch_size_fails(spec in arb_valid_spec()) {
            let mut spec = spec;
            spec.data.batch_size = 0;
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidBatchSize(0))
            ));
        }

        #[test]
        fn prop_zero_lr_fails(spec in arb_valid_spec()) {
            let mut spec = spec;
            spec.optimizer.lr = 0.0;
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidLearningRate(_))
            ));
        }

        #[test]
        fn prop_negative_lr_fails(
            spec in arb_valid_spec(),
            neg_lr in -1.0f32..-1e-6
        ) {
            let mut spec = spec;
            spec.optimizer.lr = neg_lr;
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidLearningRate(_))
            ));
        }

        #[test]
        fn prop_lr_above_one_fails(
            spec in arb_valid_spec(),
            high_lr in 1.01f32..10.0
        ) {
            let mut spec = spec;
            spec.optimizer.lr = high_lr;
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidLearningRate(_))
            ));
        }

        #[test]
        fn prop_zero_epochs_fails(spec in arb_valid_spec()) {
            let mut spec = spec;
            spec.training.epochs = 0;
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidEpochs(0))
            ));
        }

        #[test]
        fn prop_valid_lora_passes(
            spec in arb_valid_spec(),
            rank in 1usize..1024,
            alpha in 0.1f32..100.0,
            dropout in 0.0f32..0.99
        ) {
            let mut spec = spec;
            spec.lora = Some(LoRASpec {
                rank,
                alpha,
                target_modules: vec!["q_proj".to_string()],
                dropout,
            });
            prop_assert!(validate_config(&spec).is_ok());
        }

        #[test]
        fn prop_lora_rank_zero_fails(spec in arb_valid_spec()) {
            let mut spec = spec;
            spec.lora = Some(LoRASpec {
                rank: 0,
                alpha: 16.0,
                target_modules: vec!["q_proj".to_string()],
                dropout: 0.0,
            });
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidLoRARank(0))
            ));
        }

        #[test]
        fn prop_lora_rank_too_high_fails(
            spec in arb_valid_spec(),
            rank in 1025usize..10000
        ) {
            let mut spec = spec;
            spec.lora = Some(LoRASpec {
                rank,
                alpha: 16.0,
                target_modules: vec!["q_proj".to_string()],
                dropout: 0.0,
            });
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidLoRARank(_))
            ));
        }

        #[test]
        fn prop_lora_alpha_zero_fails(spec in arb_valid_spec()) {
            let mut spec = spec;
            spec.lora = Some(LoRASpec {
                rank: 64,
                alpha: 0.0,
                target_modules: vec!["q_proj".to_string()],
                dropout: 0.0,
            });
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidLoRAAlpha(_))
            ));
        }

        #[test]
        fn prop_lora_negative_alpha_fails(
            spec in arb_valid_spec(),
            neg_alpha in -100.0f32..-0.01
        ) {
            let mut spec = spec;
            spec.lora = Some(LoRASpec {
                rank: 64,
                alpha: neg_alpha,
                target_modules: vec!["q_proj".to_string()],
                dropout: 0.0,
            });
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidLoRAAlpha(_))
            ));
        }

        #[test]
        fn prop_lora_dropout_one_fails(spec in arb_valid_spec()) {
            let mut spec = spec;
            spec.lora = Some(LoRASpec {
                rank: 64,
                alpha: 16.0,
                target_modules: vec!["q_proj".to_string()],
                dropout: 1.0,
            });
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidLoRADropout(_))
            ));
        }

        #[test]
        fn prop_lora_negative_dropout_fails(
            spec in arb_valid_spec(),
            neg_dropout in -1.0f32..-0.01
        ) {
            let mut spec = spec;
            spec.lora = Some(LoRASpec {
                rank: 64,
                alpha: 16.0,
                target_modules: vec!["q_proj".to_string()],
                dropout: neg_dropout,
            });
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidLoRADropout(_))
            ));
        }

        #[test]
        fn prop_valid_quant_bits(
            spec in arb_valid_spec(),
            bits in prop_oneof![Just(4u8), Just(8u8)]
        ) {
            let mut spec = spec;
            spec.quantize = Some(QuantSpec {
                bits,
                symmetric: true,
                per_channel: true,
            });
            prop_assert!(validate_config(&spec).is_ok());
        }

        #[test]
        fn prop_invalid_quant_bits_fails(
            spec in arb_valid_spec(),
            bits in 0u8..4
        ) {
            let mut spec = spec;
            spec.quantize = Some(QuantSpec {
                bits,
                symmetric: true,
                per_channel: true,
            });
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidQuantBits(_))
            ));
        }

        #[test]
        fn prop_valid_merge_methods(
            spec in arb_valid_spec(),
            method in prop_oneof!["ties", "dare", "slerp"]
        ) {
            let mut spec = spec;
            spec.merge = Some(MergeSpec {
                method: method.to_string(),
                params: HashMap::new(),
            });
            prop_assert!(validate_config(&spec).is_ok());
        }

        #[test]
        fn prop_zero_grad_clip_fails(spec in arb_valid_spec()) {
            let mut spec = spec;
            spec.training.grad_clip = Some(0.0);
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidGradClip(_))
            ));
        }

        #[test]
        fn prop_negative_grad_clip_fails(
            spec in arb_valid_spec(),
            neg_clip in -10.0f32..-0.01
        ) {
            let mut spec = spec;
            spec.training.grad_clip = Some(neg_clip);
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidGradClip(_))
            ));
        }

        #[test]
        fn prop_valid_seq_len(
            spec in arb_valid_spec(),
            seq_len in 1usize..8192
        ) {
            let mut spec = spec;
            spec.data.seq_len = Some(seq_len);
            prop_assert!(validate_config(&spec).is_ok());
        }

        #[test]
        fn prop_zero_seq_len_fails(spec in arb_valid_spec()) {
            let mut spec = spec;
            spec.data.seq_len = Some(0);
            prop_assert!(matches!(
                validate_config(&spec),
                Err(ValidationError::InvalidSeqLen(0))
            ));
        }

        #[test]
        fn prop_valid_lr_schedulers(
            spec in arb_valid_spec(),
            scheduler in prop_oneof!["cosine", "linear", "constant"]
        ) {
            let mut spec = spec;
            spec.training.lr_scheduler = Some(scheduler.to_string());
            prop_assert!(validate_config(&spec).is_ok());
        }
    }
}
