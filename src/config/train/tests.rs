//! Tests for training configuration and batch loading

use super::*;
use crate::config::schema::TrainSpec;
use crate::train::Batch;
use std::io::Write;
use tempfile::{NamedTempFile, TempDir};

#[test]
fn test_load_valid_config() {
    let yaml = r"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001
";

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let spec = load_config(temp_file.path()).unwrap();
    assert_eq!(spec.optimizer.name, "adam");
    assert_eq!(spec.data.batch_size, 8);
}

#[test]
fn test_load_invalid_config() {
    let yaml = r"
model:
  path: nonexistent_test_model.gguf

data:
  train: train.parquet
  batch_size: 0

optimizer:
  name: adam
  lr: 0.001
";

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let result = load_config(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_load_malformed_yaml() {
    let yaml = "this is not valid yaml: [}";

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let result = load_config(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_load_config_nonexistent_file() {
    let result = load_config("/nonexistent/path/to/config.yaml");
    assert!(result.is_err());
}

#[test]
fn test_load_config_with_lora() {
    let yaml = r"
model:
  path: nonexistent_test_model.gguf
  layers: [q_proj, v_proj]

data:
  train: train.parquet
  batch_size: 4

optimizer:
  name: adamw
  lr: 0.0001

lora:
  rank: 16
  alpha: 32
  target_modules: [q_proj, v_proj]
";
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let spec = load_config(temp_file.path()).unwrap();
    assert!(spec.lora.is_some());
    let lora = spec.lora.unwrap();
    assert_eq!(lora.rank, 16);
    assert_eq!(lora.alpha, 32.0);
}

#[test]
fn test_load_config_with_quantize() {
    let yaml = r"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 4

optimizer:
  name: adam
  lr: 0.001

quantize:
  bits: 4
  symmetric: true
";
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let spec = load_config(temp_file.path()).unwrap();
    assert!(spec.quantize.is_some());
    let quant = spec.quantize.unwrap();
    assert_eq!(quant.bits, 4);
}

#[test]
fn test_load_config_with_training_options() {
    let yaml = r"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: sgd
  lr: 0.01

training:
  epochs: 5
  grad_clip: 1.0
";
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let spec = load_config(temp_file.path()).unwrap();
    assert_eq!(spec.training.epochs, 5);
    assert_eq!(spec.training.grad_clip, Some(1.0));
}

#[test]
fn test_train_from_yaml_nonexistent() {
    let result = train_from_yaml("/nonexistent/config.yaml");
    assert!(result.is_err());
}

#[test]
fn test_train_from_yaml_success() {
    let output_dir = TempDir::new().unwrap();
    let yaml = format!(
        r#"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001

training:
  epochs: 2
  output_dir: "{}"
"#,
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());

    // Check that model file was saved
    let output_path = output_dir.path().join("final_model.json");
    assert!(output_path.exists());
}

#[test]
fn test_train_from_yaml_with_grad_clip() {
    let output_dir = TempDir::new().unwrap();
    let yaml = format!(
        r#"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 4

optimizer:
  name: sgd
  lr: 0.01

training:
  epochs: 1
  grad_clip: 1.0
  output_dir: "{}"
"#,
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

#[test]
fn test_train_from_yaml_with_lora() {
    let output_dir = TempDir::new().unwrap();
    let yaml = format!(
        r#"
model:
  path: nonexistent_test_model.gguf
  layers: [q_proj, v_proj]

data:
  train: train.parquet
  batch_size: 4

optimizer:
  name: adamw
  lr: 0.0001

lora:
  rank: 16
  alpha: 32
  target_modules: [q_proj, v_proj]

training:
  epochs: 1
  output_dir: "{}"
"#,
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

#[test]
fn test_train_from_yaml_with_quantize() {
    let output_dir = TempDir::new().unwrap();
    let yaml = format!(
        r#"
model:
  path: nonexistent_test_model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 4

optimizer:
  name: adam
  lr: 0.001

quantize:
  bits: 4
  symmetric: true

training:
  epochs: 1
  output_dir: "{}"
"#,
        output_dir.path().display()
    );

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_ok());
}

#[test]
fn test_train_from_yaml_malformed() {
    let yaml = "not: [valid yaml";
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_train_from_yaml_invalid_config() {
    let yaml = r"
model:
  path: nonexistent_test_model.gguf

data:
  train: train.parquet
  batch_size: 0

optimizer:
  name: adam
  lr: 0.001
";
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(yaml.as_bytes()).unwrap();

    let result = train_from_yaml(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_create_demo_batches_default() {
    let batches = create_demo_batches(4);
    assert!(!batches.is_empty());
    // Each batch should have 4 * 4 = 16 elements (batch_size * feature_dim)
    assert_eq!(batches[0].inputs.len(), 16);
    assert_eq!(batches[0].targets.len(), 16);
}

#[test]
fn test_create_demo_batches_small_batch() {
    let batches = create_demo_batches(1);
    assert!(!batches.is_empty());
    // With batch_size 1, should create multiple batches
    assert!(batches.len() >= 2);
}

#[test]
fn test_create_demo_batches_zero_batch_size() {
    // Should handle zero gracefully (uses max(1))
    let batches = create_demo_batches(0);
    assert!(!batches.is_empty());
}

#[test]
fn test_create_demo_batches_large_batch() {
    let batches = create_demo_batches(16);
    assert!(!batches.is_empty());
    // With large batch size, should have at least 2 batches (from the max)
    assert!(batches.len() >= 2);
}

#[test]
fn test_rebatch_empty() {
    let batches: Vec<Batch> = Vec::new();
    let result = rebatch(batches, 4);
    assert!(result.is_empty());
}

#[test]
fn test_rebatch_single_batch() {
    use crate::Tensor;
    // Create batch with 4 examples, each with 2 features (8 elements total)
    // rebatch determines input_dim from first batch's length = 8
    // So this represents 1 example (8/8=1)
    let batch = Batch::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false),
        Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], false),
    );
    // With input_dim=4 and 4 elements, we have 1 example
    // Rebatching 1 example into batch_size 2 gives 1 batch
    let result = rebatch(vec![batch], 2);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_rebatch_multiple_batches() {
    use crate::Tensor;
    // Two batches with same dimensions
    let batch1 = Batch::new(
        Tensor::from_vec(vec![1.0, 2.0], false),
        Tensor::from_vec(vec![3.0, 4.0], false),
    );
    let batch2 = Batch::new(
        Tensor::from_vec(vec![5.0, 6.0], false),
        Tensor::from_vec(vec![7.0, 8.0], false),
    );
    // input_dim = 2 (from first batch), total 4 elements = 2 examples
    // Rebatching 2 examples with batch_size 2 = 1 batch
    let result = rebatch(vec![batch1, batch2], 2);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_rebatch_creates_multiple_batches() {
    use crate::Tensor;
    // Create 4 batches each with 2 elements (input_dim=2)
    let batches: Vec<Batch> = (0..4)
        .map(|i| {
            Batch::new(
                Tensor::from_vec(vec![(i * 2) as f32, (i * 2 + 1) as f32], false),
                Tensor::from_vec(vec![(i * 2 + 10) as f32, (i * 2 + 11) as f32], false),
            )
        })
        .collect();
    // input_dim = 2, total 8 elements = 4 examples
    // Rebatching 4 examples with batch_size 2 = 2 batches
    let result = rebatch(batches, 2);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_rebatch_uneven_split() {
    use crate::Tensor;
    // Create 5 batches each with 2 elements
    let batches: Vec<Batch> = (0..5)
        .map(|i| {
            Batch::new(
                Tensor::from_vec(vec![(i * 2) as f32, (i * 2 + 1) as f32], false),
                Tensor::from_vec(vec![(i * 2 + 10) as f32, (i * 2 + 11) as f32], false),
            )
        })
        .collect();
    // input_dim = 2, total 10 elements = 5 examples
    // Rebatching 5 examples with batch_size 2 = 3 batches (2, 2, 1)
    let result = rebatch(batches, 2);
    assert_eq!(result.len(), 3);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_load_json_batches_structured_format() {
    let json = r#"
{
    "examples": [
        {"input": [1.0, 2.0], "target": [3.0, 4.0]},
        {"input": [5.0, 6.0], "target": [7.0, 8.0]}
    ]
}
"#;
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(json.as_bytes()).unwrap();

    let result = load_json_batches(temp_file.path(), 2);
    assert!(result.is_ok());
    let batches = result.unwrap();
    assert!(!batches.is_empty());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_load_json_batches_array_format() {
    let json = r#"[
    {"input": [1.0, 2.0], "target": [3.0, 4.0]},
    {"input": [5.0, 6.0], "target": [7.0, 8.0]}
]"#;
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(json.as_bytes()).unwrap();

    let result = load_json_batches(temp_file.path(), 1);
    assert!(result.is_ok());
    let batches = result.unwrap();
    assert_eq!(batches.len(), 2);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_load_json_batches_invalid_format() {
    let json = r#"{"invalid": "format"}"#;
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(json.as_bytes()).unwrap();

    // Should fall back to demo data
    let result = load_json_batches(temp_file.path(), 4);
    assert!(result.is_ok());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_load_json_batches_nonexistent_file() {
    let result = load_json_batches(std::path::Path::new("/nonexistent/file.json"), 4);
    assert!(result.is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_arrow_array_to_f32_float32() {
    use ::arrow::array::Float32Array;
    let array: ::arrow::array::ArrayRef =
        std::sync::Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0]));
    let result = arrow_array_to_f32(&array).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_arrow_array_to_f32_float64() {
    use ::arrow::array::Float64Array;
    let array: ::arrow::array::ArrayRef =
        std::sync::Arc::new(Float64Array::from(vec![1.0f64, 2.0, 3.0]));
    let result = arrow_array_to_f32(&array).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_arrow_array_to_f32_int32() {
    use ::arrow::array::Int32Array;
    let array: ::arrow::array::ArrayRef = std::sync::Arc::new(Int32Array::from(vec![1i32, 2, 3]));
    let result = arrow_array_to_f32(&array).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_arrow_array_to_f32_int64() {
    use ::arrow::array::Int64Array;
    let array: ::arrow::array::ArrayRef = std::sync::Arc::new(Int64Array::from(vec![1i64, 2, 3]));
    let result = arrow_array_to_f32(&array).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_arrow_array_to_f32_unsupported_type() {
    use ::arrow::array::StringArray;
    let array: ::arrow::array::ArrayRef = std::sync::Arc::new(StringArray::from(vec!["a", "b"]));
    let result = arrow_array_to_f32(&array);
    assert!(result.is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_load_training_batches_missing_file() {
    use crate::config::schema::{DataConfig, ModelRef, OptimSpec, TrainingParams};
    use std::collections::HashMap;

    let spec = TrainSpec {
        model: ModelRef {
            path: std::path::PathBuf::from("model.gguf"),
            layers: vec![],
        },
        data: DataConfig {
            train: std::path::PathBuf::from("/nonexistent/data.parquet"),
            val: None,
            batch_size: 4,
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
    };

    // Should fall back to demo batches
    let result = load_training_batches(&spec);
    assert!(result.is_ok());
    let batches = result.unwrap();
    assert!(!batches.is_empty());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_load_training_batches_unsupported_extension() {
    use crate::config::schema::{DataConfig, ModelRef, OptimSpec, TrainingParams};
    use std::collections::HashMap;

    let temp_file = NamedTempFile::with_suffix(".txt").unwrap();
    std::fs::write(temp_file.path(), "test data").unwrap();

    let spec = TrainSpec {
        model: ModelRef {
            path: std::path::PathBuf::from("model.gguf"),
            layers: vec![],
        },
        data: DataConfig {
            train: temp_file.path().to_path_buf(),
            val: None,
            batch_size: 4,
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
    };

    // Should fall back to demo batches for unsupported format
    let result = load_training_batches(&spec);
    assert!(result.is_ok());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_load_training_batches_json() {
    use crate::config::schema::{DataConfig, ModelRef, OptimSpec, TrainingParams};
    use std::collections::HashMap;

    let json = r#"[
    {"input": [1.0, 2.0, 3.0, 4.0], "target": [5.0, 6.0, 7.0, 8.0]},
    {"input": [9.0, 10.0, 11.0, 12.0], "target": [13.0, 14.0, 15.0, 16.0]}
]"#;
    let temp_file = NamedTempFile::with_suffix(".json").unwrap();
    std::fs::write(temp_file.path(), json).unwrap();

    let spec = TrainSpec {
        model: ModelRef {
            path: std::path::PathBuf::from("model.gguf"),
            layers: vec![],
        },
        data: DataConfig {
            train: temp_file.path().to_path_buf(),
            val: None,
            batch_size: 1,
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
    };

    let result = load_training_batches(&spec);
    assert!(result.is_ok());
    let batches = result.unwrap();
    assert_eq!(batches.len(), 2);
}
