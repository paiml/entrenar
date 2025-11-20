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
                    .map_err(|e| Error::Serialization(format!("JSON serialization failed: {}", e)))?
            } else {
                serde_json::to_string(&state)
                    .map_err(|e| Error::Serialization(format!("JSON serialization failed: {}", e)))?
            }
        }
        ModelFormat::Yaml => serde_yaml::to_string(&state)
            .map_err(|e| Error::Serialization(format!("YAML serialization failed: {}", e)))?,
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
    use crate::io::{ModelMetadata, Model};
    use crate::Tensor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_model_json() {
        let params = vec![
            ("weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], true)),
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
}
