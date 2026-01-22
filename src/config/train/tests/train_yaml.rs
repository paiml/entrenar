//! Tests for train_from_yaml functionality

use crate::config::train::train_from_yaml;
use std::io::Write;
use tempfile::{NamedTempFile, TempDir};

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
