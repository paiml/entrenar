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
    let spec: TrainSpec = serde_yaml::from_str(&yaml_content)
        .map_err(|e| Error::ConfigError(format!("Failed to parse YAML config: {}", e)))?;

    // Step 3: Validate configuration
    validate_config(&spec).map_err(|e| Error::ConfigError(format!("Invalid config: {}", e)))?;

    println!("✓ Config loaded and validated");
    println!("  Model: {}", spec.model.path.display());
    println!(
        "  Optimizer: {} (lr={})",
        spec.optimizer.name, spec.optimizer.lr
    );
    println!("  Batch size: {}", spec.data.batch_size);
    println!("  Epochs: {}", spec.training.epochs);

    if let Some(lora) = &spec.lora {
        println!("  LoRA: rank={}, alpha={}", lora.rank, lora.alpha);
    }

    if let Some(quant) = &spec.quantize {
        println!("  Quantization: {}-bit", quant.bits);
    }
    println!();

    // Step 4: Build model and optimizer
    println!("Building model and optimizer...");
    let model = crate::config::build_model(&spec)?;
    let optimizer = crate::config::build_optimizer(&spec.optimizer)?;

    // Step 5: Setup trainer
    use crate::train::{Batch, MSELoss, TrainConfig, Trainer};

    let mut train_config = TrainConfig::new().with_log_interval(100);

    if let Some(clip) = spec.training.grad_clip {
        train_config = train_config.with_grad_clip(clip);
    }

    let mut trainer = Trainer::new(
        model.parameters.into_iter().map(|(_, t)| t).collect(),
        optimizer,
        train_config,
    );
    trainer.set_loss(Box::new(MSELoss));

    println!("✓ Trainer initialized");
    println!();

    // Step 6: Create dummy training data (placeholder until real data loading)
    println!("Creating training batches...");
    let batches = vec![
        Batch::new(
            crate::Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false),
            crate::Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], false),
        ),
        Batch::new(
            crate::Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], false),
            crate::Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], false),
        ),
    ];
    println!("✓ {} batches created", batches.len());
    println!();

    // Step 7: Training loop
    println!("Starting training...");
    println!();

    for epoch in 0..spec.training.epochs {
        let avg_loss = trainer.train_epoch(batches.clone(), |x| x.clone());
        println!(
            "Epoch {}/{}: loss={:.6}",
            epoch + 1,
            spec.training.epochs,
            avg_loss
        );
    }

    println!();
    println!("✓ Training complete");
    println!(
        "  Final loss: {:.6}",
        trainer.metrics.losses.last().copied().unwrap_or(0.0)
    );
    println!(
        "  Best loss: {:.6}",
        trainer.metrics.best_loss().unwrap_or(0.0)
    );
    println!();

    // Step 8: Save the trained model
    let output_path = spec.training.output_dir.join("final_model.json");
    println!("Saving model to {}...", output_path.display());

    // Reconstruct model for saving
    let final_model = crate::io::Model::new(
        model.metadata.clone(),
        trainer
            .params()
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("param_{}", i), t.clone()))
            .collect(),
    );

    use crate::io::{save_model, ModelFormat, SaveConfig};
    let save_config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
    save_model(&final_model, &output_path, &save_config)?;

    println!("✓ Model saved successfully");
    println!();

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

    let spec: TrainSpec = serde_yaml::from_str(&yaml_content)
        .map_err(|e| Error::ConfigError(format!("Failed to parse YAML config: {}", e)))?;

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
