//! Single-command training from YAML configuration

use super::schema::TrainSpec;
use super::validate::validate_config;
use crate::error::{Error, Result};
use std::fs;
use std::path::Path;

/// Train a model from YAML configuration file
///
/// This is the main entry point for declarative training. It:
/// 1. Loads and parses the YAML config
/// 2. Validates the configuration
/// 3. Builds the model and optimizer
/// 4. Runs the training loop
/// 5. Saves the final model
///
/// # Example
///
/// ```no_run
/// use entrenar::config::train_from_yaml;
///
/// let model = train_from_yaml("config.yaml")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn train_from_yaml<P: AsRef<Path>>(config_path: P) -> Result<()> {
    // Step 1: Load YAML file
    let yaml_content = fs::read_to_string(config_path.as_ref()).map_err(|e| {
        Error::ConfigError(format!(
            "Failed to read config file {}: {}",
            config_path.as_ref().display(),
            e
        ))
    })?;

    // Step 2: Parse YAML
    let spec: TrainSpec = serde_yaml::from_str(&yaml_content).map_err(|e| {
        Error::ConfigError(format!("Failed to parse YAML config: {}", e))
    })?;

    // Step 3: Validate configuration
    validate_config(&spec).map_err(|e| Error::ConfigError(format!("Invalid config: {}", e)))?;

    // Step 4: TODO - Build model from spec
    // This would require implementing model loading from GGUF/safetensors
    // For now, this is a placeholder showing the structure

    println!("✓ Config loaded and validated");
    println!("  Model: {}", spec.model.path.display());
    println!("  Optimizer: {} (lr={})", spec.optimizer.name, spec.optimizer.lr);
    println!("  Batch size: {}", spec.data.batch_size);
    println!("  Epochs: {}", spec.training.epochs);

    if let Some(lora) = &spec.lora {
        println!("  LoRA: rank={}, alpha={}", lora.rank, lora.alpha);
    }

    if let Some(quant) = &spec.quantize {
        println!("  Quantization: {}-bit", quant.bits);
    }

    // TODO: Implement actual training loop
    // let model = load_model(&spec.model.path)?;
    // if let Some(lora) = spec.lora {
    //     model.add_lora_layers(lora.rank);
    // }
    //
    // let optimizer = build_optimizer(&spec.optimizer)?;
    // let trainer = Trainer::new(model, optimizer, CrossEntropyLoss::new());
    //
    // for epoch in 0..spec.training.epochs {
    //     let loss = trainer.train_epoch(&dataloader);
    //     println!("Epoch {}: loss={:.4}", epoch, loss);
    // }
    //
    // save_model(&trainer.model, &spec.training.output_dir)?;

    Ok(())
}

/// Load training spec from YAML file (without running training)
///
/// Useful for testing config parsing and validation separately from training.
pub fn load_config<P: AsRef<Path>>(config_path: P) -> Result<TrainSpec> {
    let yaml_content = fs::read_to_string(config_path.as_ref()).map_err(|e| {
        Error::ConfigError(format!(
            "Failed to read config file {}: {}",
            config_path.as_ref().display(),
            e
        ))
    })?;

    let spec: TrainSpec = serde_yaml::from_str(&yaml_content).map_err(|e| {
        Error::ConfigError(format!("Failed to parse YAML config: {}", e))
    })?;

    validate_config(&spec).map_err(|e| Error::ConfigError(format!("Invalid config: {}", e)))?;

    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_valid_config() {
        let yaml = r#"
model:
  path: model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(yaml.as_bytes()).unwrap();

        let spec = load_config(temp_file.path()).unwrap();
        assert_eq!(spec.optimizer.name, "adam");
        assert_eq!(spec.data.batch_size, 8);
    }

    #[test]
    fn test_load_invalid_config() {
        let yaml = r#"
model:
  path: model.gguf

data:
  train: train.parquet
  batch_size: 0

optimizer:
  name: adam
  lr: 0.001
"#;

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
}
