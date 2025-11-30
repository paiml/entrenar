//! TDD tests for YAML Mode Training
//!
//! Following Extreme TDD methodology - tests written first, implementation follows.

use super::*;

// ============================================================================
// MANIFEST PARSING TESTS (TDD RED)
// ============================================================================

mod manifest_parsing {
    use super::*;

    #[test]
    fn test_parse_minimal_manifest() {
        let yaml = r#"
entrenar: "1.0"
name: "test-experiment"
version: "1.0.0"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(manifest.entrenar, "1.0");
        assert_eq!(manifest.name, "test-experiment");
        assert_eq!(manifest.version, "1.0.0");
    }

    #[test]
    fn test_parse_with_description_and_seed() {
        let yaml = r#"
entrenar: "1.0"
name: "mnist-classifier"
version: "1.0.0"
description: "MNIST digit classification experiment"
seed: 42
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(manifest.description, Some("MNIST digit classification experiment".to_string()));
        assert_eq!(manifest.seed, Some(42));
    }

    #[test]
    fn test_parse_data_config_with_source() {
        let yaml = r#"
entrenar: "1.0"
name: "data-test"
version: "1.0.0"

data:
  source: "./data/train.parquet"
  format: "parquet"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let data = manifest.data.unwrap();
        assert_eq!(data.source, Some("./data/train.parquet".to_string()));
        assert_eq!(data.format, Some("parquet".to_string()));
    }

    #[test]
    fn test_parse_data_config_with_split() {
        let yaml = r#"
entrenar: "1.0"
name: "split-test"
version: "1.0.0"

data:
  source: "hf://ylecun/mnist"
  split:
    train: 0.8
    val: 0.1
    test: 0.1
    stratify: "label"
    seed: 42
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let data = manifest.data.unwrap();
        let split = data.split.unwrap();
        assert_eq!(split.train, 0.8);
        assert_eq!(split.val, Some(0.1));
        assert_eq!(split.test, Some(0.1));
        assert_eq!(split.stratify, Some("label".to_string()));
        assert_eq!(split.seed, Some(42));
    }

    #[test]
    fn test_parse_data_config_with_loader() {
        let yaml = r#"
entrenar: "1.0"
name: "loader-test"
version: "1.0.0"

data:
  source: "./data/train.parquet"
  loader:
    batch_size: 32
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: false
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let loader = manifest.data.unwrap().loader.unwrap();
        assert_eq!(loader.batch_size, 32);
        assert!(loader.shuffle);
        assert_eq!(loader.num_workers, Some(4));
        assert_eq!(loader.pin_memory, Some(true));
        assert_eq!(loader.drop_last, Some(false));
    }

    #[test]
    fn test_parse_model_config() {
        let yaml = r#"
entrenar: "1.0"
name: "model-test"
version: "1.0.0"

model:
  source: "hf://meta-llama/Llama-2-7b"
  device: "auto"
  dtype: "float16"
  freeze:
    - "embed_tokens"
    - "layers.0"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let model = manifest.model.unwrap();
        assert_eq!(model.source, "hf://meta-llama/Llama-2-7b");
        assert_eq!(model.device, Some("auto".to_string()));
        assert_eq!(model.dtype, Some("float16".to_string()));
        assert_eq!(model.freeze.unwrap().len(), 2);
    }

    #[test]
    fn test_parse_optimizer_config() {
        let yaml = r#"
entrenar: "1.0"
name: "optim-test"
version: "1.0.0"

optimizer:
  name: "adamw"
  lr: 0.001
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let optim = manifest.optimizer.unwrap();
        assert_eq!(optim.name, "adamw");
        assert_eq!(optim.lr, 0.001);
        assert_eq!(optim.weight_decay, Some(0.01));
        assert_eq!(optim.betas, Some(vec![0.9, 0.999]));
        assert_eq!(optim.eps, Some(1e-8));
    }

    #[test]
    fn test_parse_scheduler_config() {
        let yaml = r#"
entrenar: "1.0"
name: "scheduler-test"
version: "1.0.0"

scheduler:
  name: "cosine_annealing"
  warmup:
    steps: 1000
    start_lr: 1e-7
  T_max: 10000
  eta_min: 1e-6
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let sched = manifest.scheduler.unwrap();
        assert_eq!(sched.name, "cosine_annealing");
        let warmup = sched.warmup.unwrap();
        assert_eq!(warmup.steps, Some(1000));
        assert_eq!(warmup.start_lr, Some(1e-7));
        assert_eq!(sched.t_max, Some(10000));
        assert_eq!(sched.eta_min, Some(1e-6));
    }

    #[test]
    fn test_parse_training_config() {
        let yaml = r#"
entrenar: "1.0"
name: "training-test"
version: "1.0.0"

training:
  epochs: 10
  gradient:
    accumulation_steps: 4
    clip_norm: 1.0
  mixed_precision:
    enabled: true
    dtype: "float16"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let training = manifest.training.unwrap();
        assert_eq!(training.epochs, Some(10));
        let grad = training.gradient.unwrap();
        assert_eq!(grad.accumulation_steps, Some(4));
        assert_eq!(grad.clip_norm, Some(1.0));
        let mp = training.mixed_precision.unwrap();
        assert!(mp.enabled);
        assert_eq!(mp.dtype, Some("float16".to_string()));
    }

    #[test]
    fn test_parse_lora_config() {
        let yaml = r#"
entrenar: "1.0"
name: "lora-test"
version: "1.0.0"

lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
  bias: "none"
  quantize_base: true
  quantize_bits: 4
  quant_type: "nf4"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let lora = manifest.lora.unwrap();
        assert!(lora.enabled);
        assert_eq!(lora.rank, 16);
        assert_eq!(lora.alpha, 32.0);
        assert_eq!(lora.dropout, Some(0.05));
        assert_eq!(lora.target_modules.len(), 3);
        assert_eq!(lora.bias, Some("none".to_string()));
        assert!(lora.quantize_base.unwrap());
        assert_eq!(lora.quantize_bits, Some(4));
        assert_eq!(lora.quant_type, Some("nf4".to_string()));
    }

    #[test]
    fn test_parse_quantize_config() {
        let yaml = r#"
entrenar: "1.0"
name: "quant-test"
version: "1.0.0"

quantize:
  enabled: true
  bits: 4
  scheme: "symmetric"
  granularity: "per_channel"
  group_size: 128
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let quant = manifest.quantize.unwrap();
        assert!(quant.enabled);
        assert_eq!(quant.bits, 4);
        assert_eq!(quant.scheme, Some("symmetric".to_string()));
        assert_eq!(quant.granularity, Some("per_channel".to_string()));
        assert_eq!(quant.group_size, Some(128));
    }

    #[test]
    fn test_parse_monitoring_config() {
        let yaml = r#"
entrenar: "1.0"
name: "monitor-test"
version: "1.0.0"

monitoring:
  terminal:
    enabled: true
    refresh_rate: 100
    metrics:
      - loss
      - accuracy
  tracking:
    enabled: true
    backend: "trueno-db"
    project: "my-project"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let mon = manifest.monitoring.unwrap();
        let term = mon.terminal.unwrap();
        assert!(term.enabled);
        assert_eq!(term.refresh_rate, Some(100));
        assert_eq!(term.metrics.unwrap().len(), 2);
        let track = mon.tracking.unwrap();
        assert!(track.enabled);
        assert_eq!(track.backend, Some("trueno-db".to_string()));
    }

    #[test]
    fn test_parse_output_config() {
        let yaml = r#"
entrenar: "1.0"
name: "output-test"
version: "1.0.0"

output:
  dir: "./experiments/{{ name }}/{{ timestamp }}"
  model:
    format: "safetensors"
    save_optimizer: true
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let output = manifest.output.unwrap();
        assert_eq!(output.dir, "./experiments/{{ name }}/{{ timestamp }}");
        let model = output.model.unwrap();
        assert_eq!(model.format, Some("safetensors".to_string()));
        assert!(model.save_optimizer.unwrap());
    }

    #[test]
    fn test_parse_complete_llama_finetune_example() {
        // Complete example from spec section 11
        let yaml = r#"
entrenar: "1.0"
name: "llama2-alpaca-finetune"
version: "1.0.0"
description: "Fine-tune LLaMA-2-7B on Alpaca dataset using QLoRA"
seed: 42

data:
  source: "hf://tatsu-lab/alpaca"
  split:
    train: 0.9
    val: 0.1
    seed: 42
  loader:
    batch_size: 4
    shuffle: true
    num_workers: 4

model:
  source: "hf://meta-llama/Llama-2-7b"
  device: "auto"
  dtype: "float16"

optimizer:
  name: "adamw"
  lr: 0.0002
  betas: [0.9, 0.999]
  weight_decay: 0.01

scheduler:
  name: "cosine_annealing"
  warmup:
    steps: 100
  T_max: 10000
  eta_min: 1e-6

training:
  epochs: 3
  gradient:
    accumulation_steps: 16
    clip_norm: 1.0
  mixed_precision:
    enabled: true
    dtype: "bfloat16"

lora:
  enabled: true
  rank: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
  quantize_base: true
  quantize_bits: 4
  quant_type: "nf4"

monitoring:
  terminal:
    enabled: true
    refresh_rate: 100
  tracking:
    enabled: true
    backend: "trueno-db"
    project: "llama-finetune"

output:
  dir: "./experiments/llama2-alpaca/{{ timestamp }}"
  model:
    format: "safetensors"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();

        // Verify all sections parsed correctly
        assert_eq!(manifest.entrenar, "1.0");
        assert_eq!(manifest.name, "llama2-alpaca-finetune");
        assert_eq!(manifest.seed, Some(42));
        assert!(manifest.data.is_some());
        assert!(manifest.model.is_some());
        assert!(manifest.optimizer.is_some());
        assert!(manifest.scheduler.is_some());
        assert!(manifest.training.is_some());
        assert!(manifest.lora.is_some());
        assert!(manifest.monitoring.is_some());
        assert!(manifest.output.is_some());
    }
}

// ============================================================================
// VALIDATION TESTS (TDD RED)
// ============================================================================

mod validation_tests {
    use super::*;

    #[test]
    fn test_validate_minimal_valid_manifest() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_ok(), "Minimal manifest should be valid");
    }

    #[test]
    fn test_validate_rejects_unsupported_version() {
        let yaml = r#"
entrenar: "2.0"
name: "test"
version: "1.0.0"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ManifestError::UnsupportedVersion(_)));
    }

    #[test]
    fn test_validate_rejects_empty_name() {
        let yaml = r#"
entrenar: "1.0"
name: ""
version: "1.0.0"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ManifestError::EmptyRequiredField(_)));
    }

    #[test]
    fn test_validate_rejects_invalid_learning_rate() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

optimizer:
  name: "adam"
  lr: -0.001
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ManifestError::InvalidRange { .. }));
    }

    #[test]
    fn test_validate_rejects_zero_batch_size() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

data:
  source: "./data.parquet"
  loader:
    batch_size: 0
    shuffle: true
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ManifestError::InvalidRange { .. }));
    }

    #[test]
    fn test_validate_rejects_mutually_exclusive_duration() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

training:
  epochs: 10
  max_steps: 5000
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ManifestError::MutuallyExclusive { .. }));
    }

    #[test]
    fn test_validate_rejects_invalid_split_ratios() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

data:
  source: "./data.parquet"
  split:
    train: 0.5
    val: 0.3
    test: 0.3
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ManifestError::InvalidSplitRatios { .. }));
    }

    #[test]
    fn test_validate_rejects_invalid_quantization_bits() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

quantize:
  enabled: true
  bits: 5
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ManifestError::InvalidQuantBits { .. }));
    }

    #[test]
    fn test_validate_accepts_valid_lora_config() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules:
    - q_proj
    - v_proj
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_rejects_lora_without_target_modules() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules: []
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ManifestError::EmptyRequiredField(_)));
    }

    #[test]
    fn test_validate_accepts_disabled_lora_without_modules() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

lora:
  enabled: false
  rank: 16
  alpha: 32
  target_modules: []
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let result = validate_manifest(&manifest);
        assert!(result.is_ok(), "Disabled LoRA should not require target_modules");
    }
}

// ============================================================================
// DEFAULT VALUES TESTS (TDD RED)
// ============================================================================

mod default_values {
    use super::*;

    #[test]
    fn test_default_data_loader_values() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

data:
  source: "./data.parquet"
  loader:
    batch_size: 16
    shuffle: false
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let loader = manifest.data.unwrap().loader.unwrap();

        // Verify defaults are applied via Default trait or serde defaults
        assert_eq!(loader.batch_size, 16);
        assert!(!loader.shuffle);
        // num_workers should default to 0
        assert_eq!(loader.num_workers.unwrap_or(0), 0);
        // pin_memory should default to false
        assert!(!loader.pin_memory.unwrap_or(false));
        // drop_last should default to false
        assert!(!loader.drop_last.unwrap_or(false));
    }

    #[test]
    fn test_default_optimizer_values() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

optimizer:
  name: "adam"
  lr: 0.001
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let optim = manifest.optimizer.unwrap();

        // Default weight_decay is 0.01 for adamw, but adam has no default
        assert!(optim.weight_decay.is_none() || optim.weight_decay == Some(0.0));
    }

    #[test]
    fn test_default_training_epochs() {
        let yaml = r#"
entrenar: "1.0"
name: "test"
version: "1.0.0"

training: {}
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let training = manifest.training.unwrap();

        // Default epochs is 10 per spec Appendix A
        assert_eq!(training.epochs.unwrap_or(10), 10);
    }
}

// ============================================================================
// SERIALIZATION ROUNDTRIP TESTS (TDD RED)
// ============================================================================

mod serialization {
    use super::*;

    #[test]
    fn test_roundtrip_minimal_manifest() {
        let yaml = r#"
entrenar: "1.0"
name: "roundtrip-test"
version: "1.0.0"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let serialized = serde_yaml::to_string(&manifest).unwrap();
        let deserialized: TrainingManifest = serde_yaml::from_str(&serialized).unwrap();

        assert_eq!(manifest.entrenar, deserialized.entrenar);
        assert_eq!(manifest.name, deserialized.name);
        assert_eq!(manifest.version, deserialized.version);
    }

    #[test]
    fn test_roundtrip_complete_manifest() {
        let yaml = r#"
entrenar: "1.0"
name: "complete-test"
version: "1.0.0"
description: "Test description"
seed: 42

data:
  source: "./train.parquet"
  split:
    train: 0.8
    val: 0.2

model:
  source: "hf://test/model"
  dtype: "float16"

optimizer:
  name: "adamw"
  lr: 0.001
  weight_decay: 0.01

training:
  epochs: 5
  gradient:
    clip_norm: 1.0

lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules:
    - q_proj
    - v_proj

output:
  dir: "./output"
"#;
        let manifest: TrainingManifest = serde_yaml::from_str(yaml).unwrap();
        let serialized = serde_yaml::to_string(&manifest).unwrap();
        let deserialized: TrainingManifest = serde_yaml::from_str(&serialized).unwrap();

        assert_eq!(manifest.seed, deserialized.seed);
        assert_eq!(manifest.lora.as_ref().unwrap().rank, deserialized.lora.as_ref().unwrap().rank);
    }
}
