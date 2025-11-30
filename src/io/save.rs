//! Model saving functionality

use super::format::{ModelFormat, SaveConfig};
use super::model::Model;
use crate::{Error, Result};
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Save a model to a file
///
/// # Arguments
///
/// * `model` - The model to save
/// * `path` - Output file path
/// * `config` - Save configuration (format, options)
///
/// # Example
///
/// ```no_run
/// use entrenar::io::{Model, ModelMetadata, save_model, SaveConfig, ModelFormat};
/// # use entrenar::Tensor;
///
/// let params = vec![
///     ("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], true)),
/// ];
/// let model = Model::new(ModelMetadata::new("my-model", "linear"), params);
/// let config = SaveConfig::new(ModelFormat::Json);
///
/// save_model(&model, "model.json", &config).unwrap();
/// ```
pub fn save_model(model: &Model, path: impl AsRef<Path>, config: &SaveConfig) -> Result<()> {
    let path = path.as_ref();

    // Convert model to serializable state
    let state = model.to_state();

    // Serialize based on format
    match config.format {
        ModelFormat::SafeTensors => {
            // SafeTensors is binary format - handle separately
            return save_safetensors(model, path);
        }
        ModelFormat::Json => {
            let data = if config.pretty {
                serde_json::to_string_pretty(&state)
                    .map_err(|e| Error::Serialization(format!("JSON serialization failed: {e}")))?
            } else {
                serde_json::to_string(&state)
                    .map_err(|e| Error::Serialization(format!("JSON serialization failed: {e}")))?
            };
            let mut file = File::create(path)?;
            file.write_all(data.as_bytes())?;
        }
        ModelFormat::Yaml => {
            let data = serde_yaml::to_string(&state)
                .map_err(|e| Error::Serialization(format!("YAML serialization failed: {e}")))?;
            let mut file = File::create(path)?;
            file.write_all(data.as_bytes())?;
        }
        #[cfg(feature = "gguf")]
        ModelFormat::Gguf => {
            return Err(Error::Serialization(
                "GGUF format not yet implemented. Enable 'gguf' feature and use realizar integration.".to_string()
            ));
        }
    }

    Ok(())
}

/// Save model in SafeTensors format (HuggingFace compatible)
fn save_safetensors(model: &Model, path: &Path) -> Result<()> {
    // Collect tensor data with proper lifetime management
    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = model
        .parameters
        .iter()
        .map(|(name, tensor)| {
            let data = tensor.data();
            let bytes: Vec<u8> = bytemuck::cast_slice(data.as_slice().unwrap()).to_vec();
            let shape = vec![tensor.len()];
            (name.clone(), bytes, shape)
        })
        .collect();

    // Create TensorViews from collected data
    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, bytes, shape)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap();
            (name.as_str(), view)
        })
        .collect();

    // Create metadata with model info
    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), model.metadata.name.clone());
    metadata.insert(
        "architecture".to_string(),
        model.metadata.architecture.clone(),
    );
    metadata.insert("version".to_string(), model.metadata.version.clone());

    // Serialize to SafeTensors format
    let safetensor_bytes = safetensors::serialize(views, Some(metadata))
        .map_err(|e| Error::Serialization(format!("SafeTensors serialization failed: {e}")))?;

    // Write to file
    std::fs::write(path, safetensor_bytes)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{Model, ModelMetadata};
    use crate::Tensor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_model_json() {
        let params = vec![
            (
                "weight".to_string(),
                Tensor::from_vec(vec![1.0, 2.0, 3.0], true),
            ),
            ("bias".to_string(), Tensor::from_vec(vec![0.1], false)),
        ];

        let model = Model::new(ModelMetadata::new("test-model", "linear"), params);
        let config = SaveConfig::new(ModelFormat::Json);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        // Verify file was created and has content
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(!content.is_empty());
        assert!(content.contains("test-model"));
        assert!(content.contains("linear"));
    }

    #[test]
    fn test_save_model_yaml() {
        let params = vec![("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], true))];

        let model = Model::new(ModelMetadata::new("test", "simple"), params);
        let config = SaveConfig::new(ModelFormat::Yaml);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("test"));
        assert!(content.contains("simple"));
    }

    #[test]
    fn test_save_model_json_pretty() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("pretty-test", "test"), params);
        let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        // Pretty JSON should have newlines
        assert!(content.contains('\n'));
    }

    #[test]
    fn test_save_model_json_compact() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("compact-test", "test"), params);
        let config = SaveConfig::new(ModelFormat::Json).with_pretty(false);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        // Compact JSON should be single line (minus trailing)
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_save_model_empty_params() {
        let model = Model::new(ModelMetadata::new("empty", "test"), vec![]);
        let config = SaveConfig::new(ModelFormat::Json);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("empty"));
    }

    #[test]
    fn test_save_model_large_tensor() {
        let large_data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        let params = vec![("large".to_string(), Tensor::from_vec(large_data, false))];
        let model = Model::new(ModelMetadata::new("large", "test"), params);
        let config = SaveConfig::new(ModelFormat::Json);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.len() > 1000);
    }

    #[test]
    fn test_save_config_builder() {
        let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
        assert!(config.pretty);
        assert_eq!(config.format, ModelFormat::Json);
    }

    #[test]
    fn test_save_model_with_compress_option() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("compress-test", "test"), params);
        let config = SaveConfig::new(ModelFormat::Json).with_compress(true);

        let temp_file = NamedTempFile::new().unwrap();
        // Currently compress is not implemented, but we can still save
        save_model(&model, temp_file.path(), &config).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("compress-test"));
    }

    #[test]
    fn test_save_model_multiple_tensors() {
        let params = vec![
            (
                "layer1.weight".to_string(),
                Tensor::from_vec(vec![1.0, 2.0], true),
            ),
            ("layer1.bias".to_string(), Tensor::from_vec(vec![0.1], true)),
            (
                "layer2.weight".to_string(),
                Tensor::from_vec(vec![3.0, 4.0], false),
            ),
        ];
        let model = Model::new(ModelMetadata::new("multi", "deep"), params);
        let config = SaveConfig::new(ModelFormat::Yaml);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("layer1.weight"));
        assert!(content.contains("layer2.weight"));
    }

    #[test]
    fn test_save_model_with_metadata() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let meta = ModelMetadata::new("meta-test", "test")
            .with_custom("version", serde_json::json!("1.0.0"))
            .with_custom("author", serde_json::json!("test"));
        let model = Model::new(meta, params);
        let config = SaveConfig::new(ModelFormat::Json);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("version"));
    }

    #[test]
    fn test_save_config_default() {
        let config = SaveConfig::default();
        assert_eq!(config.format, ModelFormat::Json);
        assert!(config.pretty);
        assert!(!config.compress);
    }

    #[test]
    fn test_save_model_invalid_path() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("test", "test"), params);
        let config = SaveConfig::new(ModelFormat::Json);

        // Try to save to an invalid directory
        let result = save_model(&model, "/nonexistent/directory/model.json", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_model_safetensors() {
        let params = vec![
            (
                "weight".to_string(),
                Tensor::from_vec(vec![1.0, 2.0, 3.0], true),
            ),
            ("bias".to_string(), Tensor::from_vec(vec![0.1], false)),
        ];

        let model = Model::new(ModelMetadata::new("safetensor-test", "linear"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        // Verify file was created and is binary (starts with safetensors magic)
        let content = std::fs::read(temp_file.path()).unwrap();
        assert!(!content.is_empty());
        // SafeTensors files start with a header length (8 bytes)
        assert!(content.len() > 8);
    }

    #[test]
    fn test_save_model_safetensors_can_be_loaded() {
        let params = vec![
            (
                "layer1.weight".to_string(),
                Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true),
            ),
            (
                "layer1.bias".to_string(),
                Tensor::from_vec(vec![0.5], false),
            ),
        ];

        let model = Model::new(ModelMetadata::new("roundtrip-test", "mlp"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        // Verify we can load it back with safetensors crate
        let data = std::fs::read(temp_file.path()).unwrap();
        let loaded = safetensors::SafeTensors::deserialize(&data).unwrap();

        // Check tensor names exist - names() returns Vec<&str>
        let names = loaded.names();
        assert!(names.contains(&"layer1.weight"));
        assert!(names.contains(&"layer1.bias"));

        // Check tensor data
        let weight = loaded.tensor("layer1.weight").unwrap();
        assert_eq!(weight.shape(), &[4]);
        let weight_data: &[f32] = bytemuck::cast_slice(weight.data());
        assert_eq!(weight_data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_save_safetensors_metadata() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("meta-model", "transformer"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        // Load and check metadata using read_metadata
        let data = std::fs::read(temp_file.path()).unwrap();
        let (_, st_metadata) = safetensors::SafeTensors::read_metadata(&data).unwrap();

        let metadata = st_metadata.metadata();
        assert!(metadata.is_some());
        let meta = metadata.as_ref().unwrap();
        assert_eq!(meta.get("name").unwrap(), "meta-model");
        assert_eq!(meta.get("architecture").unwrap(), "transformer");
    }

    #[test]
    fn test_save_safetensors_large_tensor() {
        let large_data: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();
        let params = vec![(
            "large_weights".to_string(),
            Tensor::from_vec(large_data.clone(), false),
        )];
        let model = Model::new(ModelMetadata::new("large", "test"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        // Verify data integrity
        let data = std::fs::read(temp_file.path()).unwrap();
        let loaded = safetensors::SafeTensors::deserialize(&data).unwrap();
        let tensor = loaded.tensor("large_weights").unwrap();
        let tensor_data: &[f32] = bytemuck::cast_slice(tensor.data());
        assert_eq!(tensor_data.len(), 10000);
        assert!((tensor_data[0] - 0.0).abs() < 1e-6);
        assert!((tensor_data[9999] - 9.999).abs() < 1e-3);
    }

    #[test]
    fn test_save_safetensors_invalid_path() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let model = Model::new(ModelMetadata::new("test", "test"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let result = save_model(&model, "/nonexistent/directory/model.safetensors", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_safetensors_empty_params() {
        let model = Model::new(ModelMetadata::new("empty", "test"), vec![]);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        // Should still create valid file with metadata
        let data = std::fs::read(temp_file.path()).unwrap();
        let loaded = safetensors::SafeTensors::deserialize(&data).unwrap();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_save_safetensors_multiple_tensors() {
        let params = vec![
            (
                "encoder.layer1.weight".to_string(),
                Tensor::from_vec(vec![1.0, 2.0], true),
            ),
            (
                "encoder.layer1.bias".to_string(),
                Tensor::from_vec(vec![0.1], true),
            ),
            (
                "encoder.layer2.weight".to_string(),
                Tensor::from_vec(vec![3.0, 4.0, 5.0], false),
            ),
            (
                "decoder.layer1.weight".to_string(),
                Tensor::from_vec(vec![6.0, 7.0], false),
            ),
        ];
        let model = Model::new(ModelMetadata::new("encoder-decoder", "transformer"), params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        let temp_file = NamedTempFile::new().unwrap();
        save_model(&model, temp_file.path(), &config).unwrap();

        let data = std::fs::read(temp_file.path()).unwrap();
        let loaded = safetensors::SafeTensors::deserialize(&data).unwrap();
        assert_eq!(loaded.len(), 4);

        // names() returns Vec<&str> directly
        let names = loaded.names();
        assert!(names.contains(&"encoder.layer1.weight"));
        assert!(names.contains(&"decoder.layer1.weight"));
    }
}
