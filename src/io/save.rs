//! Model saving functionality

use super::format::{ModelFormat, SaveConfig};
use super::model::Model;
use crate::{Error, Result};
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
    let data = match config.format {
        ModelFormat::Json => {
            if config.pretty {
                serde_json::to_string_pretty(&state)
                    .map_err(|e| Error::Serialization(format!("JSON serialization failed: {e}")))?
            } else {
                serde_json::to_string(&state)
                    .map_err(|e| Error::Serialization(format!("JSON serialization failed: {e}")))?
            }
        }
        ModelFormat::Yaml => serde_yaml::to_string(&state)
            .map_err(|e| Error::Serialization(format!("YAML serialization failed: {e}")))?,
        #[cfg(feature = "gguf")]
        ModelFormat::Gguf => {
            return Err(Error::Serialization(
                "GGUF format not yet implemented. Enable 'gguf' feature and use realizar integration.".to_string()
            ));
        }
    };

    // Write to file
    let mut file = File::create(path)?;
    file.write_all(data.as_bytes())?;

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
}
