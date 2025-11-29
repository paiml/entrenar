//! Model loading functionality

use super::format::ModelFormat;
use super::model::{Model, ModelMetadata, ModelState};
use crate::{Error, Result, Tensor};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Load a model from a file
///
/// # Arguments
///
/// * `path` - Input file path
///
/// The format is automatically detected from the file extension.
///
/// # Example
///
/// ```no_run
/// use entrenar::io::load_model;
///
/// let model = load_model("model.json").unwrap();
/// println!("Loaded model: {}", model.metadata.name);
/// ```
pub fn load_model(path: impl AsRef<Path>) -> Result<Model> {
    let path = path.as_ref();

    // Detect format from extension
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::Serialization("File has no extension".to_string()))?;

    let format = ModelFormat::from_extension(ext)
        .ok_or_else(|| Error::Serialization(format!("Unsupported file extension: {ext}")))?;

    // Handle SafeTensors separately (binary format)
    if format == ModelFormat::SafeTensors {
        return load_safetensors(path);
    }

    // Read file content (text formats)
    let mut file = File::open(path)?;

    let mut content = String::new();
    file.read_to_string(&mut content)?;

    // Deserialize based on format
    let state: ModelState = match format {
        ModelFormat::Json => serde_json::from_str(&content)
            .map_err(|e| Error::Serialization(format!("JSON deserialization failed: {e}")))?,
        ModelFormat::Yaml => serde_yaml::from_str(&content)
            .map_err(|e| Error::Serialization(format!("YAML deserialization failed: {e}")))?,
        ModelFormat::SafeTensors => unreachable!(), // Handled above
        #[cfg(feature = "gguf")]
        ModelFormat::Gguf => {
            return Err(Error::Serialization(
                "GGUF format not yet implemented. Enable 'gguf' feature and use realizar integration.".to_string()
            ));
        }
    };

    // Convert state to model
    Ok(Model::from_state(state))
}

/// Load model from SafeTensors format (HuggingFace compatible)
fn load_safetensors(path: &Path) -> Result<Model> {
    // Read binary file
    let data =
        std::fs::read(path).map_err(|e| Error::Serialization(format!("Failed to read file: {e}")))?;

    // Parse SafeTensors and get metadata
    let (_, st_metadata) = safetensors::SafeTensors::read_metadata(&data)
        .map_err(|e| Error::Serialization(format!("SafeTensors parsing failed: {e}")))?;

    // Extract custom metadata
    let custom_meta = st_metadata.metadata();
    let name = custom_meta
        .as_ref()
        .and_then(|m| m.get("name").cloned())
        .unwrap_or_else(|| "unknown".to_string());
    let architecture = custom_meta
        .as_ref()
        .and_then(|m| m.get("architecture").cloned())
        .unwrap_or_else(|| "unknown".to_string());

    let metadata = ModelMetadata::new(name, architecture);

    // Deserialize to access tensors
    let safetensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| Error::Serialization(format!("SafeTensors parsing failed: {e}")))?;

    // Convert tensors - names() returns Vec<&str>, not an iterator
    let parameters: Vec<(String, Tensor)> = safetensors
        .names()
        .into_iter()
        .map(|name| {
            let tensor_view = safetensors.tensor(name).unwrap();
            let data: &[f32] = bytemuck::cast_slice(tensor_view.data());
            let tensor = Tensor::from_vec(data.to_vec(), false); // Default to no grad
            (name.to_string(), tensor)
        })
        .collect();

    Ok(Model::new(metadata, parameters))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{save_model, Model, ModelMetadata, SaveConfig};
    use crate::Tensor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_model_json() {
        // Create and save a model
        let params = vec![
            (
                "weight".to_string(),
                Tensor::from_vec(vec![1.0, 2.0, 3.0], true),
            ),
            ("bias".to_string(), Tensor::from_vec(vec![0.1], false)),
        ];

        let original = Model::new(ModelMetadata::new("test-model", "linear"), params);

        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("json");

        let config = SaveConfig::new(ModelFormat::Json);
        save_model(&original, &temp_path, &config).unwrap();

        // Load it back
        let loaded = load_model(&temp_path).unwrap();

        // Verify
        assert_eq!(original.metadata.name, loaded.metadata.name);
        assert_eq!(original.metadata.architecture, loaded.metadata.architecture);
        assert_eq!(original.parameters.len(), loaded.parameters.len());

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_model_yaml() {
        let params = vec![("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], true))];

        let original = Model::new(ModelMetadata::new("yaml-test", "simple"), params);

        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("yaml");

        let config = SaveConfig::new(ModelFormat::Yaml);
        save_model(&original, &temp_path, &config).unwrap();

        let loaded = load_model(&temp_path).unwrap();

        assert_eq!(original.metadata.name, loaded.metadata.name);
        assert_eq!(original.parameters.len(), loaded.parameters.len());

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_unsupported_extension() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("unknown");

        let result = load_model(&temp_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_load_round_trip() {
        // Create a model with multiple parameters
        let params = vec![
            (
                "layer1.weight".to_string(),
                Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true),
            ),
            (
                "layer1.bias".to_string(),
                Tensor::from_vec(vec![0.1, 0.2], true),
            ),
            (
                "layer2.weight".to_string(),
                Tensor::from_vec(vec![5.0, 6.0], false),
            ),
        ];

        let meta = ModelMetadata::new("round-trip-test", "multi-layer")
            .with_custom("layers", serde_json::json!(2))
            .with_custom("hidden_size", serde_json::json!(4));

        let original = Model::new(meta, params);

        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("json");

        // Save and load
        let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
        save_model(&original, &temp_path, &config).unwrap();
        let loaded = load_model(&temp_path).unwrap();

        // Verify all parameters match
        assert_eq!(original.parameters.len(), loaded.parameters.len());

        for (orig_name, orig_tensor) in &original.parameters {
            let loaded_tensor = loaded.get_parameter(orig_name).unwrap();
            assert_eq!(orig_tensor.data(), loaded_tensor.data());
            assert_eq!(orig_tensor.requires_grad(), loaded_tensor.requires_grad());
        }

        // Verify metadata
        assert_eq!(original.metadata.custom.len(), loaded.metadata.custom.len());

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_model_file_not_found() {
        let result = load_model("nonexistent_file.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_model_no_extension() {
        let result = load_model("model_without_extension");
        assert!(result.is_err());
        // Use match instead of unwrap_err since Model doesn't implement Debug
        if let Err(err) = result {
            assert!(err.to_string().contains("no extension"));
        }
    }

    #[test]
    fn test_load_model_invalid_json() {
        use std::io::Write;
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("json");

        // Write invalid JSON
        let mut f = File::create(&temp_path).unwrap();
        f.write_all(b"{ invalid json }").unwrap();
        drop(f);

        let result = load_model(&temp_path);
        assert!(result.is_err());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_model_invalid_yaml() {
        use std::io::Write;
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("yaml");

        // Write invalid YAML
        let mut f = File::create(&temp_path).unwrap();
        f.write_all(b"this: is: not: valid: yaml: [}").unwrap();
        drop(f);

        let result = load_model(&temp_path);
        assert!(result.is_err());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_yml_extension() {
        let params = vec![("weight".to_string(), Tensor::from_vec(vec![1.0], true))];
        let original = Model::new(ModelMetadata::new("yml-test", "simple"), params);

        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("yml");

        let config = SaveConfig::new(ModelFormat::Yaml);
        save_model(&original, &temp_path, &config).unwrap();

        let loaded = load_model(&temp_path).unwrap();
        assert_eq!(original.metadata.name, loaded.metadata.name);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_model_safetensors() {
        let params = vec![
            (
                "weight".to_string(),
                Tensor::from_vec(vec![1.0, 2.0, 3.0], true),
            ),
            ("bias".to_string(), Tensor::from_vec(vec![0.1], false)),
        ];

        let original = Model::new(ModelMetadata::new("safetensor-test", "linear"), params);

        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("safetensors");

        let config = SaveConfig::new(ModelFormat::SafeTensors);
        save_model(&original, &temp_path, &config).unwrap();

        let loaded = load_model(&temp_path).unwrap();

        assert_eq!(original.metadata.name, loaded.metadata.name);
        assert_eq!(original.metadata.architecture, loaded.metadata.architecture);
        assert_eq!(original.parameters.len(), loaded.parameters.len());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_safetensors_round_trip_data_integrity() {
        let params = vec![
            (
                "layer1.weight".to_string(),
                Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true),
            ),
            (
                "layer1.bias".to_string(),
                Tensor::from_vec(vec![0.5, 0.6], false),
            ),
        ];

        let original = Model::new(ModelMetadata::new("round-trip", "mlp"), params);

        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("safetensors");

        let config = SaveConfig::new(ModelFormat::SafeTensors);
        save_model(&original, &temp_path, &config).unwrap();

        let loaded = load_model(&temp_path).unwrap();

        // Verify data matches
        for (name, orig_tensor) in &original.parameters {
            let loaded_tensor = loaded.get_parameter(name).unwrap();
            assert_eq!(orig_tensor.data(), loaded_tensor.data());
        }

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_safetensors_file_not_found() {
        let result = load_model("nonexistent.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_safetensors_invalid_data() {
        use std::io::Write;
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("safetensors");

        // Write invalid safetensors data
        let mut f = File::create(&temp_path).unwrap();
        f.write_all(b"not valid safetensors binary data").unwrap();
        drop(f);

        let result = load_model(&temp_path);
        assert!(result.is_err());

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_safetensors_large_model() {
        let large_data: Vec<f32> = (0..5000).map(|i| i as f32 * 0.001).collect();
        let params = vec![
            ("large_weight".to_string(), Tensor::from_vec(large_data.clone(), false)),
            ("small_bias".to_string(), Tensor::from_vec(vec![0.1, 0.2], false)),
        ];

        let original = Model::new(ModelMetadata::new("large-model", "test"), params);

        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("safetensors");

        let config = SaveConfig::new(ModelFormat::SafeTensors);
        save_model(&original, &temp_path, &config).unwrap();

        let loaded = load_model(&temp_path).unwrap();

        let loaded_large = loaded.get_parameter("large_weight").unwrap();
        assert_eq!(loaded_large.len(), 5000);

        // Verify some values
        let data = loaded_large.data();
        assert!((data[[0]] - 0.0).abs() < 1e-6);
        assert!((data[[4999]] - 4.999).abs() < 1e-3);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_load_safetensors_metadata_preserved() {
        let params = vec![("w".to_string(), Tensor::from_vec(vec![1.0], false))];
        let original = Model::new(ModelMetadata::new("meta-model", "transformer"), params);

        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("safetensors");

        let config = SaveConfig::new(ModelFormat::SafeTensors);
        save_model(&original, &temp_path, &config).unwrap();

        let loaded = load_model(&temp_path).unwrap();

        assert_eq!(loaded.metadata.name, "meta-model");
        assert_eq!(loaded.metadata.architecture, "transformer");

        std::fs::remove_file(temp_path).ok();
    }
}
