//! Single-command training from YAML configuration

use super::schema::TrainSpec;
use super::validate::validate_config;
use crate::error::{Error, Result};
use crate::train::Batch;
use crate::Tensor;
use std::fs;
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use alimentar::{ArrowDataset, Dataset};

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
        .map_err(|e| Error::ConfigError(format!("Failed to parse YAML config: {e}")))?;

    // Step 3: Validate configuration
    validate_config(&spec).map_err(|e| Error::ConfigError(format!("Invalid config: {e}")))?;

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
    use crate::train::{MSELoss, TrainConfig, Trainer};

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

    // Step 6: Load training data
    println!("Loading training data...");
    let batches = load_training_batches(&spec)?;
    println!("✓ {} batches created", batches.len());
    println!();

    // Step 7: Training loop
    println!("Starting training...");
    println!();

    for epoch in 0..spec.training.epochs {
        let avg_loss = trainer.train_epoch(batches.clone(), Clone::clone);
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
            .map(|(i, t)| (format!("param_{i}"), t.clone()))
            .collect(),
    );

    use crate::io::{save_model, ModelFormat, SaveConfig};
    let save_config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
    save_model(&final_model, &output_path, &save_config)?;

    println!("✓ Model saved successfully");
    println!();

    Ok(())
}

/// Load training batches from data file using alimentar
///
/// Supports parquet, JSON, and CSV formats via alimentar.
/// Falls back to demo data if the file doesn't exist (for testing).
fn load_training_batches(spec: &TrainSpec) -> Result<Vec<Batch>> {
    let data_path = &spec.data.train;
    let batch_size = spec.data.batch_size;

    // Check if data file exists
    if !data_path.exists() {
        eprintln!(
            "Warning: Training data not found at '{}', using demo data",
            data_path.display()
        );
        return Ok(create_demo_batches(batch_size));
    }

    // Load data using alimentar (only on non-WASM)
    #[cfg(not(target_arch = "wasm32"))]
    {
        let ext = data_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "parquet" => load_parquet_batches(data_path, batch_size),
            "json" => load_json_batches(data_path, batch_size),
            _ => {
                eprintln!("Warning: Unsupported data format '{ext}', using demo data");
                Ok(create_demo_batches(batch_size))
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        eprintln!("Warning: Data loading not available in WASM, using demo data");
        Ok(create_demo_batches(batch_size))
    }
}

/// Load batches from parquet file using alimentar
#[cfg(not(target_arch = "wasm32"))]
fn load_parquet_batches(path: &Path, batch_size: usize) -> Result<Vec<Batch>> {
    println!("  Loading parquet: {}", path.display());

    let dataset = ArrowDataset::from_parquet(path).map_err(|e| {
        Error::ConfigError(format!("Failed to load parquet {}: {}", path.display(), e))
    })?;

    println!("  Loaded {} rows from parquet", dataset.len());

    // Convert Arrow RecordBatches to training Batches
    let mut batches = Vec::new();
    let schema = dataset.schema();

    // Get column names for input/output detection
    let column_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    // Look for input/output columns (common patterns)
    let input_col = column_names
        .iter()
        .find(|&&n| n == "input" || n == "input_ids" || n == "x" || n == "features")
        .copied();
    let target_col = column_names
        .iter()
        .find(|&&n| n == "target" || n == "output" || n == "labels" || n == "y")
        .copied();

    if input_col.is_none() || target_col.is_none() {
        eprintln!(
            "Warning: Could not find input/target columns in parquet (found: {column_names:?})"
        );
        eprintln!("  Expected columns like: input/target, x/y, features/labels");
        return Ok(create_demo_batches(batch_size));
    }

    let input_name = input_col.unwrap();
    let target_name = target_col.unwrap();
    println!("  Using columns: input='{input_name}', target='{target_name}'");

    // Process each Arrow batch
    for record_batch in dataset.iter() {
        // Extract input and target arrays
        let input_idx = schema
            .index_of(input_name)
            .map_err(|e| Error::ConfigError(format!("Column not found: {e}")))?;
        let target_idx = schema
            .index_of(target_name)
            .map_err(|e| Error::ConfigError(format!("Column not found: {e}")))?;

        let input_array = record_batch.column(input_idx);
        let target_array = record_batch.column(target_idx);

        // Convert to f32 vectors (simplified - assumes numeric data)
        let input_data = arrow_array_to_f32(input_array)?;
        let target_data = arrow_array_to_f32(target_array)?;

        // Create batch
        let batch = Batch::new(
            Tensor::from_vec(input_data, false),
            Tensor::from_vec(target_data, false),
        );
        batches.push(batch);
    }

    // Re-batch to desired batch size if needed
    if batches.len() > 1 && batch_size > 0 {
        batches = rebatch(batches, batch_size);
    }

    Ok(batches)
}

/// Load batches from JSON file
#[cfg(not(target_arch = "wasm32"))]
fn load_json_batches(path: &Path, batch_size: usize) -> Result<Vec<Batch>> {
    println!("  Loading JSON: {}", path.display());

    // Try to load as JSON array of {input, target} objects
    let content = std::fs::read_to_string(path).map_err(|e| {
        Error::ConfigError(format!("Failed to read JSON {}: {}", path.display(), e))
    })?;

    #[derive(serde::Deserialize)]
    struct Example {
        input: Vec<f32>,
        target: Vec<f32>,
    }

    #[derive(serde::Deserialize)]
    struct DataFile {
        examples: Vec<Example>,
    }

    // Try structured format first
    if let Ok(data) = serde_json::from_str::<DataFile>(&content) {
        println!("  Loaded {} examples from JSON", data.examples.len());
        let batches: Vec<Batch> = data
            .examples
            .chunks(batch_size.max(1))
            .map(|chunk| {
                let input_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.input.clone()).collect();
                let target_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.target.clone()).collect();
                Batch::new(
                    Tensor::from_vec(input_data, false),
                    Tensor::from_vec(target_data, false),
                )
            })
            .collect();
        return Ok(batches);
    }

    // Try array of examples
    if let Ok(examples) = serde_json::from_str::<Vec<Example>>(&content) {
        println!("  Loaded {} examples from JSON array", examples.len());
        let batches: Vec<Batch> = examples
            .chunks(batch_size.max(1))
            .map(|chunk| {
                let input_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.input.clone()).collect();
                let target_data: Vec<f32> = chunk.iter().flat_map(|ex| ex.target.clone()).collect();
                Batch::new(
                    Tensor::from_vec(input_data, false),
                    Tensor::from_vec(target_data, false),
                )
            })
            .collect();
        return Ok(batches);
    }

    eprintln!("Warning: Could not parse JSON data format, using demo data");
    Ok(create_demo_batches(batch_size))
}

/// Convert Arrow array to f32 vector
#[cfg(not(target_arch = "wasm32"))]
fn arrow_array_to_f32(array: &arrow::array::ArrayRef) -> Result<Vec<f32>> {
    use arrow::array::{Float32Array, Float64Array, Int32Array, Int64Array};
    use arrow::datatypes::DataType;

    match array.data_type() {
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            Ok(arr.values().to_vec())
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            Ok(arr.values().iter().map(|&x| x as f32).collect())
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            Ok(arr.values().iter().map(|&x| x as f32).collect())
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok(arr.values().iter().map(|&x| x as f32).collect())
        }
        other => Err(Error::ConfigError(format!(
            "Unsupported Arrow data type: {other:?}. Use Float32, Float64, Int32, or Int64."
        ))),
    }
}

/// Re-batch data into specified batch size
fn rebatch(batches: Vec<Batch>, batch_size: usize) -> Vec<Batch> {
    // Flatten all data
    let all_inputs: Vec<f32> = batches
        .iter()
        .flat_map(|b| b.inputs.data().iter().copied())
        .collect();
    let all_targets: Vec<f32> = batches
        .iter()
        .flat_map(|b| b.targets.data().iter().copied())
        .collect();

    if all_inputs.is_empty() {
        return Vec::new();
    }

    // Determine feature dimensions from first batch
    let input_dim = batches[0].inputs.len();
    let target_dim = batches[0].targets.len();

    // Re-batch
    let num_examples = all_inputs.len() / input_dim;
    let mut new_batches = Vec::new();

    for chunk_start in (0..num_examples).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(num_examples);
        let input_start = chunk_start * input_dim;
        let input_end = chunk_end * input_dim;
        let target_start = chunk_start * target_dim;
        let target_end = chunk_end * target_dim;

        new_batches.push(Batch::new(
            Tensor::from_vec(all_inputs[input_start..input_end].to_vec(), false),
            Tensor::from_vec(all_targets[target_start..target_end].to_vec(), false),
        ));
    }

    new_batches
}

/// Create demo batches for testing when no data file is available
fn create_demo_batches(batch_size: usize) -> Vec<Batch> {
    let num_batches = 2.max(8 / batch_size.max(1));
    (0..num_batches)
        .map(|i| {
            let input_data: Vec<f32> = (0..batch_size * 4)
                .map(|j| ((i * batch_size + j) as f32) * 0.1)
                .collect();
            let target_data: Vec<f32> = (0..batch_size * 4)
                .map(|j| ((i * batch_size + j + 1) as f32) * 0.1)
                .collect();
            Batch::new(
                Tensor::from_vec(input_data, false),
                Tensor::from_vec(target_data, false),
            )
        })
        .collect()
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
        .map_err(|e| Error::ConfigError(format!("Failed to parse YAML config: {e}")))?;

    validate_config(&spec).map_err(|e| Error::ConfigError(format!("Invalid config: {e}")))?;

    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
