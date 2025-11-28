//! Model loading functionality

use super::format::ModelFormat;
use super::model::{Model, ModelState};
use crate::{Error, Result};
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
        .ok_or_else(|| Error::Serialization(format!("Unsupported file extension: {}", ext)))?;

    // Read file content
    let mut file = File::open(path)?;

    let mut content = String::new();
    file.read_to_string(&mut content)?;

    // Deserialize based on format
    let state: ModelState = match format {
        ModelFormat::Json => serde_json::from_str(&content)
            .map_err(|e| Error::Serialization(format!("JSON deserialization failed: {}", e)))?,
        ModelFormat::Yaml => serde_yaml::from_str(&content)
            .map_err(|e| Error::Serialization(format!("YAML deserialization failed: {}", e)))?,
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
}
