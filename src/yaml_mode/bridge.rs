//! Bridge converter: TrainingManifest → TrainSpec
//!
//! Maps the declarative YAML Mode manifest format to the legacy TrainSpec
//! so that `entrenar train manifest.yaml` works with the existing pipeline.

use crate::config::{
    DataConfig as SpecDataConfig, LoRASpec, ModelMode, ModelRef, OptimSpec, QuantSpec, TrainSpec,
    TrainingMode, TrainingParams,
};
use crate::yaml_mode::TrainingManifest;
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

/// Result of converting a TrainingManifest to TrainSpec
#[derive(Debug)]
pub struct BridgeResult {
    /// The converted TrainSpec
    pub spec: TrainSpec,
    /// Warnings about unsupported/ignored manifest fields
    pub warnings: Vec<String>,
}

/// Errors that can occur during manifest-to-spec conversion
#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("Missing required field: {0}")]
    MissingRequired(String),

    #[error("Invalid value for {field}: {reason}")]
    InvalidValue { field: String, reason: String },
}

/// Convert a TrainingManifest into a TrainSpec
///
/// Maps declarative manifest fields to the legacy spec format.
/// Returns warnings for manifest features not supported by TrainSpec.
pub fn manifest_to_spec(manifest: &TrainingManifest) -> Result<BridgeResult, BridgeError> {
    let mut warnings = Vec::new();

    let model = convert_model(manifest, &mut warnings)?;
    let data = convert_data(manifest, &mut warnings)?;
    let optimizer = convert_optimizer(manifest)?;
    let training = convert_training(manifest, &mut warnings);
    let lora = convert_lora(manifest, &mut warnings);
    let quantize = convert_quantize(manifest);

    // Merge config is not in manifest format — skip
    if manifest.monitoring.is_some() {
        warnings.push("monitoring config is not supported in legacy TrainSpec".into());
    }
    if manifest.callbacks.is_some() {
        warnings.push("callbacks config is not supported in legacy TrainSpec".into());
    }
    if manifest.distillation.is_some() {
        warnings.push("distillation config is not supported in legacy TrainSpec".into());
    }

    let spec = TrainSpec {
        model,
        data,
        optimizer,
        training,
        lora,
        quantize,
        merge: None,
    };

    Ok(BridgeResult { spec, warnings })
}

/// Convert model config from manifest to spec
fn convert_model(
    manifest: &TrainingManifest,
    warnings: &mut Vec<String>,
) -> Result<ModelRef, BridgeError> {
    let model_cfg = manifest
        .model
        .as_ref()
        .ok_or_else(|| BridgeError::MissingRequired("model".into()))?;

    // Determine model mode from architecture
    let mode = if let Some(ref arch) = model_cfg.architecture {
        if arch.arch_type == "transformer" {
            ModelMode::Transformer
        } else {
            ModelMode::Tabular
        }
    } else {
        ModelMode::Tabular
    };

    // Use LoRA target_modules as model layers
    let layers = manifest
        .lora
        .as_ref()
        .map(|l| l.target_modules.clone())
        .unwrap_or_default();

    if model_cfg.freeze.is_some() {
        warnings.push("model.freeze is not supported in legacy TrainSpec".into());
    }
    if model_cfg.device.is_some() {
        warnings.push("model.device is not supported in legacy TrainSpec".into());
    }

    Ok(ModelRef {
        path: PathBuf::from(&model_cfg.source),
        layers,
        mode,
        config: None,
    })
}

/// Convert data config from manifest to spec
fn convert_data(
    manifest: &TrainingManifest,
    warnings: &mut Vec<String>,
) -> Result<SpecDataConfig, BridgeError> {
    let data_cfg = manifest
        .data
        .as_ref()
        .ok_or_else(|| BridgeError::MissingRequired("data".into()))?;

    // Determine training data path: explicit train field, or source field
    let train = if let Some(ref train_path) = data_cfg.train {
        PathBuf::from(train_path)
    } else if let Some(ref source) = data_cfg.source {
        PathBuf::from(source)
    } else {
        return Err(BridgeError::MissingRequired(
            "data.source or data.train".into(),
        ));
    };

    let val = data_cfg.val.as_ref().map(PathBuf::from);

    let batch_size = data_cfg.loader.as_ref().map_or(8, |l| l.batch_size);

    if data_cfg.preprocessing.is_some() {
        warnings.push("data.preprocessing is not supported in legacy TrainSpec".into());
    }
    if data_cfg.augmentation.is_some() {
        warnings.push("data.augmentation is not supported in legacy TrainSpec".into());
    }

    Ok(SpecDataConfig {
        train,
        val,
        batch_size,
        auto_infer_types: true,
        seq_len: None,
        tokenizer: None,
        input_column: None,
        output_column: None,
        max_length: None,
    })
}

/// Convert optimizer config from manifest to spec
fn convert_optimizer(manifest: &TrainingManifest) -> Result<OptimSpec, BridgeError> {
    let optim_cfg = manifest
        .optimizer
        .as_ref()
        .ok_or_else(|| BridgeError::MissingRequired("optimizer".into()))?;

    let name = optim_cfg.name.to_lowercase();

    // f64 -> f32 conversion for learning rate
    let lr = optim_cfg.lr as f32;

    // Pack optional parameters into HashMap
    let mut params: HashMap<String, serde_json::Value> = HashMap::new();

    if let Some(ref betas) = optim_cfg.betas {
        if betas.len() >= 2 {
            params.insert("beta1".into(), serde_json::json!(betas[0]));
            params.insert("beta2".into(), serde_json::json!(betas[1]));
        }
    }
    if let Some(eps) = optim_cfg.eps {
        params.insert("eps".into(), serde_json::json!(eps));
    }
    if let Some(wd) = optim_cfg.weight_decay {
        params.insert("weight_decay".into(), serde_json::json!(wd));
    }
    if let Some(momentum) = optim_cfg.momentum {
        params.insert("momentum".into(), serde_json::json!(momentum));
    }

    Ok(OptimSpec { name, lr, params })
}

/// Convert training config from manifest to spec
fn convert_training(manifest: &TrainingManifest, warnings: &mut Vec<String>) -> TrainingParams {
    let training_cfg = manifest.training.as_ref();
    let scheduler_cfg = manifest.scheduler.as_ref();
    let output_cfg = manifest.output.as_ref();

    let epochs = training_cfg.and_then(|t| t.epochs).unwrap_or(10);

    let grad_clip = training_cfg
        .and_then(|t| t.gradient.as_ref())
        .and_then(|g| g.clip_norm)
        .map(|v| v as f32);

    let gradient_accumulation = training_cfg
        .and_then(|t| t.gradient.as_ref())
        .and_then(|g| g.accumulation_steps);

    let mixed_precision = training_cfg
        .and_then(|t| t.mixed_precision.as_ref())
        .and_then(|mp| if mp.enabled { mp.dtype.clone() } else { None });

    let save_interval = training_cfg
        .and_then(|t| t.checkpoint.as_ref())
        .and_then(|c| c.save_every)
        .unwrap_or(1);

    let lr_scheduler = scheduler_cfg.map(|s| s.name.to_lowercase());

    let warmup_steps = scheduler_cfg
        .and_then(|s| s.warmup.as_ref())
        .and_then(|w| w.steps)
        .unwrap_or(0);

    let output_dir =
        output_cfg.map_or_else(|| PathBuf::from("./checkpoints"), |o| PathBuf::from(&o.dir));

    if training_cfg
        .and_then(|t| t.early_stopping.as_ref())
        .is_some()
    {
        warnings.push("training.early_stopping is not supported in legacy TrainSpec".into());
    }
    if training_cfg.and_then(|t| t.distributed.as_ref()).is_some() {
        warnings.push("training.distributed is not supported in legacy TrainSpec".into());
    }

    TrainingParams {
        epochs,
        grad_clip,
        lr_scheduler,
        warmup_steps,
        save_interval,
        output_dir,
        mode: TrainingMode::default(),
        gradient_accumulation,
        checkpoints: None,
        mixed_precision,
    }
}

/// Convert LoRA config from manifest to spec
fn convert_lora(manifest: &TrainingManifest, warnings: &mut Vec<String>) -> Option<LoRASpec> {
    let lora_cfg = manifest.lora.as_ref()?;

    if !lora_cfg.enabled {
        return None;
    }

    if lora_cfg.quantize_base.unwrap_or(false) {
        warnings
            .push("lora.quantize_base (QLoRA) is not fully supported in legacy TrainSpec".into());
    }

    Some(LoRASpec {
        rank: lora_cfg.rank,
        alpha: lora_cfg.alpha as f32,
        target_modules: lora_cfg.target_modules.clone(),
        dropout: lora_cfg.dropout.map_or(0.0, |d| d as f32),
    })
}

/// Convert quantization config from manifest to spec
fn convert_quantize(manifest: &TrainingManifest) -> Option<QuantSpec> {
    let quant_cfg = manifest.quantize.as_ref()?;

    if !quant_cfg.enabled {
        return None;
    }

    let symmetric = quant_cfg.scheme.as_deref().is_none_or(|s| s == "symmetric");

    let per_channel = quant_cfg
        .granularity
        .as_deref()
        .is_none_or(|g| g == "per_channel");

    Some(QuantSpec {
        bits: quant_cfg.bits,
        symmetric,
        per_channel,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yaml_mode::manifest::core::TrainingManifest;
    use crate::yaml_mode::manifest::data::{DataConfig, DataLoader};
    use crate::yaml_mode::manifest::lora::LoraConfig;
    use crate::yaml_mode::manifest::model::{ArchitectureConfig, ModelConfig};
    use crate::yaml_mode::manifest::optimizer::OptimizerConfig;
    use crate::yaml_mode::manifest::output::OutputConfig;
    use crate::yaml_mode::manifest::quantize::QuantizeConfig;
    use crate::yaml_mode::manifest::scheduler::{SchedulerConfig, WarmupConfig};
    use crate::yaml_mode::manifest::training::{
        CheckpointConfig, GradientConfig, MixedPrecisionConfig, TrainingConfig,
    };

    /// Create a minimal valid manifest for testing
    fn minimal_manifest() -> TrainingManifest {
        TrainingManifest {
            entrenar: "1.0".into(),
            name: "test-experiment".into(),
            version: "1.0.0".into(),
            description: None,
            seed: None,
            data: Some(DataConfig {
                source: Some("./data/train.parquet".into()),
                format: None,
                split: None,
                train: None,
                val: None,
                test: None,
                preprocessing: None,
                augmentation: None,
                loader: None,
            }),
            model: Some(ModelConfig {
                source: "./models/base.safetensors".into(),
                format: None,
                architecture: None,
                freeze: None,
                device: None,
                dtype: None,
            }),
            optimizer: Some(OptimizerConfig {
                name: "adam".into(),
                lr: 1e-4,
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
            }),
            scheduler: None,
            training: None,
            lora: None,
            quantize: None,
            monitoring: None,
            callbacks: None,
            output: None,
            citl: None,
            rag: None,
            graph: None,
            distillation: None,
            inspect: None,
            privacy: None,
            audit: None,
            session: None,
            stress: None,
            benchmark: None,
            debug: None,
            signing: None,
            verification: None,
            lockfile: None,
            strict: None,
            strict_validation: None,
            require_peer_review: None,
        }
    }

    #[test]
    fn test_minimal_manifest_converts() {
        let manifest = minimal_manifest();
        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(
            result.spec.model.path,
            PathBuf::from("./models/base.safetensors")
        );
        assert_eq!(result.spec.model.mode, ModelMode::Tabular);
        assert_eq!(
            result.spec.data.train,
            PathBuf::from("./data/train.parquet")
        );
        assert_eq!(result.spec.optimizer.name, "adam");
        assert!((result.spec.optimizer.lr - 1e-4).abs() < 1e-6);
        assert_eq!(result.spec.training.epochs, 10);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_missing_model_errors() {
        let mut manifest = minimal_manifest();
        manifest.model = None;
        let err = manifest_to_spec(&manifest).unwrap_err();
        assert!(matches!(err, BridgeError::MissingRequired(_)));
    }

    #[test]
    fn test_missing_data_errors() {
        let mut manifest = minimal_manifest();
        manifest.data = None;
        let err = manifest_to_spec(&manifest).unwrap_err();
        assert!(matches!(err, BridgeError::MissingRequired(_)));
    }

    #[test]
    fn test_missing_optimizer_errors() {
        let mut manifest = minimal_manifest();
        manifest.optimizer = None;
        let err = manifest_to_spec(&manifest).unwrap_err();
        assert!(matches!(err, BridgeError::MissingRequired(_)));
    }

    #[test]
    fn test_missing_data_source_errors() {
        let mut manifest = minimal_manifest();
        manifest.data = Some(DataConfig {
            source: None,
            format: None,
            split: None,
            train: None,
            val: None,
            test: None,
            preprocessing: None,
            augmentation: None,
            loader: None,
        });
        let err = manifest_to_spec(&manifest).unwrap_err();
        assert!(matches!(err, BridgeError::MissingRequired(_)));
    }

    #[test]
    fn test_explicit_train_path_preferred_over_source() {
        let mut manifest = minimal_manifest();
        manifest.data.as_mut().unwrap().source = Some("./source.parquet".into());
        manifest.data.as_mut().unwrap().train = Some("./explicit_train.parquet".into());
        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(
            result.spec.data.train,
            PathBuf::from("./explicit_train.parquet")
        );
    }

    #[test]
    fn test_val_path_converted() {
        let mut manifest = minimal_manifest();
        manifest.data.as_mut().unwrap().val = Some("./val.parquet".into());
        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(result.spec.data.val, Some(PathBuf::from("./val.parquet")));
    }

    #[test]
    fn test_batch_size_from_loader() {
        let mut manifest = minimal_manifest();
        manifest.data.as_mut().unwrap().loader = Some(DataLoader {
            batch_size: 32,
            shuffle: true,
            num_workers: None,
            pin_memory: None,
            drop_last: None,
            prefetch_factor: None,
        });
        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(result.spec.data.batch_size, 32);
    }

    #[test]
    fn test_batch_size_default_without_loader() {
        let manifest = minimal_manifest();
        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(result.spec.data.batch_size, 8);
    }

    #[test]
    fn test_transformer_mode_from_architecture() {
        let mut manifest = minimal_manifest();
        manifest.model.as_mut().unwrap().architecture = Some(ArchitectureConfig {
            arch_type: "transformer".into(),
            hidden_size: None,
            num_layers: None,
            num_heads: None,
            vocab_size: None,
            max_seq_length: None,
            layers: None,
        });
        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(result.spec.model.mode, ModelMode::Transformer);
    }

    #[test]
    fn test_optimizer_params_converted() {
        let mut manifest = minimal_manifest();
        let opt = manifest.optimizer.as_mut().unwrap();
        opt.name = "adamw".into();
        opt.lr = 3e-4;
        opt.betas = Some(vec![0.9, 0.999]);
        opt.eps = Some(1e-8);
        opt.weight_decay = Some(0.01);

        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(result.spec.optimizer.name, "adamw");
        assert!((result.spec.optimizer.lr - 3e-4).abs() < 1e-6);
        assert_eq!(
            result.spec.optimizer.params["beta1"],
            serde_json::json!(0.9)
        );
        assert_eq!(
            result.spec.optimizer.params["beta2"],
            serde_json::json!(0.999)
        );
        assert!(result.spec.optimizer.params.contains_key("eps"));
        assert!(result.spec.optimizer.params.contains_key("weight_decay"));
    }

    #[test]
    fn test_training_config_converted() {
        let mut manifest = minimal_manifest();
        manifest.training = Some(TrainingConfig {
            epochs: Some(5),
            max_steps: None,
            duration: None,
            gradient: Some(GradientConfig {
                accumulation_steps: Some(4),
                clip_norm: Some(1.0),
                clip_value: None,
            }),
            mixed_precision: Some(MixedPrecisionConfig {
                enabled: true,
                dtype: Some("bf16".into()),
                loss_scale: None,
            }),
            distributed: None,
            checkpoint: Some(CheckpointConfig {
                save_every: Some(2),
                keep_last: None,
                save_best: None,
                metric: None,
                mode: None,
            }),
            early_stopping: None,
            validation: None,
            deterministic: None,
            benchmark: None,
        });

        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(result.spec.training.epochs, 5);
        assert_eq!(result.spec.training.grad_clip, Some(1.0));
        assert_eq!(result.spec.training.gradient_accumulation, Some(4));
        assert_eq!(result.spec.training.mixed_precision, Some("bf16".into()));
        assert_eq!(result.spec.training.save_interval, 2);
    }

    #[test]
    fn test_scheduler_converted() {
        let mut manifest = minimal_manifest();
        manifest.scheduler = Some(SchedulerConfig {
            name: "cosine".into(),
            warmup: Some(WarmupConfig {
                steps: Some(100),
                ratio: None,
                start_lr: None,
            }),
            t_max: None,
            eta_min: None,
            step_size: None,
            gamma: None,
            mode: None,
            factor: None,
            patience: None,
            threshold: None,
            max_lr: None,
            pct_start: None,
            anneal_strategy: None,
            div_factor: None,
            final_div_factor: None,
        });

        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(result.spec.training.lr_scheduler, Some("cosine".into()));
        assert_eq!(result.spec.training.warmup_steps, 100);
    }

    #[test]
    fn test_output_dir_converted() {
        let mut manifest = minimal_manifest();
        manifest.output = Some(OutputConfig {
            dir: "./outputs/my-model".into(),
            model: None,
            metrics: None,
            report: None,
            registry: None,
        });

        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(
            result.spec.training.output_dir,
            PathBuf::from("./outputs/my-model")
        );
    }

    #[test]
    fn test_lora_converted() {
        let mut manifest = minimal_manifest();
        manifest.lora = Some(LoraConfig {
            enabled: true,
            rank: 64,
            alpha: 16.0,
            dropout: Some(0.1),
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            target_modules_pattern: None,
            bias: None,
            init_weights: None,
            quantize_base: None,
            quantize_bits: None,
            double_quantize: None,
            quant_type: None,
        });

        let result = manifest_to_spec(&manifest).unwrap();
        let lora = result.spec.lora.unwrap();
        assert_eq!(lora.rank, 64);
        assert!((lora.alpha - 16.0).abs() < 1e-6);
        assert!((lora.dropout - 0.1).abs() < 1e-6);
        assert_eq!(lora.target_modules, vec!["q_proj", "v_proj"]);
        // Also check that target_modules get mapped to model layers
        assert_eq!(result.spec.model.layers, vec!["q_proj", "v_proj"]);
    }

    #[test]
    fn test_lora_disabled_not_converted() {
        let mut manifest = minimal_manifest();
        manifest.lora = Some(LoraConfig {
            enabled: false,
            rank: 64,
            alpha: 16.0,
            dropout: None,
            target_modules: vec!["q_proj".into()],
            target_modules_pattern: None,
            bias: None,
            init_weights: None,
            quantize_base: None,
            quantize_bits: None,
            double_quantize: None,
            quant_type: None,
        });

        let result = manifest_to_spec(&manifest).unwrap();
        assert!(result.spec.lora.is_none());
    }

    #[test]
    fn test_quantize_converted() {
        let mut manifest = minimal_manifest();
        manifest.quantize = Some(QuantizeConfig {
            enabled: true,
            bits: 4,
            scheme: Some("symmetric".into()),
            granularity: Some("per_channel".into()),
            group_size: None,
            qat: None,
            calibration: None,
            exclude: None,
        });

        let result = manifest_to_spec(&manifest).unwrap();
        let quant = result.spec.quantize.unwrap();
        assert_eq!(quant.bits, 4);
        assert!(quant.symmetric);
        assert!(quant.per_channel);
    }

    #[test]
    fn test_quantize_disabled_not_converted() {
        let mut manifest = minimal_manifest();
        manifest.quantize = Some(QuantizeConfig {
            enabled: false,
            bits: 4,
            scheme: None,
            granularity: None,
            group_size: None,
            qat: None,
            calibration: None,
            exclude: None,
        });

        let result = manifest_to_spec(&manifest).unwrap();
        assert!(result.spec.quantize.is_none());
    }

    #[test]
    fn test_quantize_asymmetric() {
        let mut manifest = minimal_manifest();
        manifest.quantize = Some(QuantizeConfig {
            enabled: true,
            bits: 8,
            scheme: Some("asymmetric".into()),
            granularity: Some("per_tensor".into()),
            group_size: None,
            qat: None,
            calibration: None,
            exclude: None,
        });

        let result = manifest_to_spec(&manifest).unwrap();
        let quant = result.spec.quantize.unwrap();
        assert!(!quant.symmetric);
        assert!(!quant.per_channel);
    }

    #[test]
    fn test_unsupported_fields_produce_warnings() {
        let mut manifest = minimal_manifest();
        manifest.monitoring = Some(crate::yaml_mode::MonitoringConfig {
            terminal: None,
            tracking: None,
            system: None,
            alerts: None,
            drift_detection: None,
        });

        let result = manifest_to_spec(&manifest).unwrap();
        assert!(!result.warnings.is_empty());
        assert!(result.warnings.iter().any(|w| w.contains("monitoring")));
    }

    #[test]
    fn test_training_defaults_without_config() {
        let manifest = minimal_manifest();
        let result = manifest_to_spec(&manifest).unwrap();
        assert_eq!(result.spec.training.epochs, 10);
        assert!(result.spec.training.grad_clip.is_none());
        assert!(result.spec.training.lr_scheduler.is_none());
        assert_eq!(result.spec.training.warmup_steps, 0);
        assert_eq!(result.spec.training.save_interval, 1);
        assert_eq!(
            result.spec.training.output_dir,
            PathBuf::from("./checkpoints")
        );
    }

    #[test]
    fn test_bridge_error_display() {
        let e = BridgeError::MissingRequired("model".into());
        assert!(e.to_string().contains("model"));

        let e = BridgeError::InvalidValue {
            field: "lr".into(),
            reason: "must be positive".into(),
        };
        assert!(e.to_string().contains("lr"));
        assert!(e.to_string().contains("must be positive"));
    }

    #[test]
    fn test_full_manifest_roundtrip_from_yaml() {
        let yaml = r#"
entrenar: "1.0"
name: "full-test"
version: "1.0.0"

model:
  source: "./models/llama.safetensors"
  architecture:
    type: transformer
    hidden_size: 768

data:
  source: "./data/train.jsonl"
  val: "./data/val.jsonl"
  loader:
    batch_size: 16
    shuffle: true

optimizer:
  name: adamw
  lr: 0.0003
  betas: [0.9, 0.95]
  weight_decay: 0.1

scheduler:
  name: cosine
  warmup:
    steps: 200

training:
  epochs: 3
  gradient:
    clip_norm: 1.0
    accumulation_steps: 8
  mixed_precision:
    enabled: true
    dtype: bf16
  checkpoint:
    save_every: 1

lora:
  enabled: true
  rank: 32
  alpha: 64.0
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

quantize:
  enabled: true
  bits: 4
  scheme: symmetric
  granularity: per_channel

output:
  dir: "./outputs/full-test"
"#;

        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = manifest_to_spec(&manifest).unwrap();

        assert_eq!(result.spec.model.mode, ModelMode::Transformer);
        assert_eq!(
            result.spec.model.path,
            PathBuf::from("./models/llama.safetensors")
        );
        assert_eq!(
            result.spec.model.layers,
            vec!["q_proj", "k_proj", "v_proj", "o_proj"]
        );
        assert_eq!(result.spec.data.train, PathBuf::from("./data/train.jsonl"));
        assert_eq!(
            result.spec.data.val,
            Some(PathBuf::from("./data/val.jsonl"))
        );
        assert_eq!(result.spec.data.batch_size, 16);
        assert_eq!(result.spec.optimizer.name, "adamw");
        assert!((result.spec.optimizer.lr - 0.0003).abs() < 1e-6);
        assert_eq!(result.spec.training.epochs, 3);
        assert_eq!(result.spec.training.grad_clip, Some(1.0));
        assert_eq!(result.spec.training.gradient_accumulation, Some(8));
        assert_eq!(result.spec.training.mixed_precision, Some("bf16".into()));
        assert_eq!(result.spec.training.lr_scheduler, Some("cosine".into()));
        assert_eq!(result.spec.training.warmup_steps, 200);
        assert_eq!(result.spec.training.save_interval, 1);
        assert_eq!(
            result.spec.training.output_dir,
            PathBuf::from("./outputs/full-test")
        );

        let lora = result.spec.lora.unwrap();
        assert_eq!(lora.rank, 32);

        let quant = result.spec.quantize.unwrap();
        assert_eq!(quant.bits, 4);
        assert!(quant.symmetric);
        assert!(quant.per_channel);
    }
}
