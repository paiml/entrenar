//! Model exporter implementation.

use crate::hf_pipeline::error::{FetchError, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::format::ExportFormat;
use super::result::ExportResult;
use super::weights::{ModelMetadata, ModelWeights};

/// Model exporter
pub struct Exporter {
    /// Output directory
    pub(super) output_dir: PathBuf,
    /// Default format
    pub(super) default_format: ExportFormat,
    /// Include metadata
    pub(super) include_metadata: bool,
}

impl Default for Exporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Exporter {
    /// Create new exporter
    #[must_use]
    pub fn new() -> Self {
        Self {
            output_dir: PathBuf::from("."),
            default_format: ExportFormat::SafeTensors,
            include_metadata: true,
        }
    }

    /// Set output directory
    #[must_use]
    pub fn output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Set default format
    #[must_use]
    pub fn default_format(mut self, format: ExportFormat) -> Self {
        self.default_format = format;
        self
    }

    /// Set whether to include metadata
    #[must_use]
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Export weights to file
    pub fn export(
        &self,
        weights: &ModelWeights,
        format: ExportFormat,
        filename: impl AsRef<Path>,
    ) -> Result<ExportResult> {
        let path = self.output_dir.join(filename);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| FetchError::ConfigParseError {
                message: format!("Failed to create output directory: {e}"),
            })?;
        }

        match format {
            ExportFormat::SafeTensors => self.export_safetensors(weights, &path),
            ExportFormat::APR => self.export_apr(weights, &path),
            ExportFormat::GGUF => self.export_gguf(weights, &path),
            ExportFormat::PyTorch => Err(FetchError::PickleSecurityRisk),
        }
    }

    /// Export to SafeTensors format
    fn export_safetensors(&self, weights: &ModelWeights, path: &Path) -> Result<ExportResult> {
        // Mock implementation - actual safetensors serialization would use the safetensors crate
        let mut output = Vec::new();

        // Header
        let header = serde_json::json!({
            "__metadata__": {
                "format": "safetensors",
                "version": "0.1.0",
                "num_tensors": weights.tensors.len(),
                "num_params": weights.param_count(),
            }
        });
        let header_bytes =
            serde_json::to_vec(&header).map_err(|e| FetchError::ConfigParseError {
                message: format!("Failed to serialize header: {e}"),
            })?;

        // Write header length (8 bytes, little-endian)
        output.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        output.extend_from_slice(&header_bytes);

        // Write tensor data (mock - just count bytes)
        let data_size: usize = weights.tensors.values().map(|t| t.len() * 4).sum();
        output.extend(vec![0u8; data_size.min(1024)]); // Truncate for mock

        std::fs::write(path, &output).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to write file: {e}"),
        })?;

        Ok(ExportResult {
            path: path.to_path_buf(),
            format: ExportFormat::SafeTensors,
            size_bytes: output.len() as u64,
            num_tensors: weights.tensors.len(),
        })
    }

    /// Export to APR format (JSON-based)
    fn export_apr(&self, weights: &ModelWeights, path: &Path) -> Result<ExportResult> {
        #[derive(Serialize)]
        struct AprFormat {
            version: String,
            metadata: ModelMetadata,
            tensors: HashMap<String, AprTensor>,
        }

        #[derive(Serialize)]
        struct AprTensor {
            shape: Vec<usize>,
            dtype: String,
            data: Vec<f32>,
        }

        let apr = AprFormat {
            version: "1.0".to_string(),
            metadata: weights.metadata.clone(),
            tensors: weights
                .tensors
                .iter()
                .map(|(name, data)| {
                    let shape = weights.shapes.get(name).cloned().unwrap_or_default();
                    (
                        name.clone(),
                        AprTensor {
                            shape,
                            dtype: "f32".to_string(),
                            data: data.clone(),
                        },
                    )
                })
                .collect(),
        };

        let json =
            serde_json::to_string_pretty(&apr).map_err(|e| FetchError::ConfigParseError {
                message: format!("Failed to serialize APR: {e}"),
            })?;

        std::fs::write(path, &json).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to write file: {e}"),
        })?;

        Ok(ExportResult {
            path: path.to_path_buf(),
            format: ExportFormat::APR,
            size_bytes: json.len() as u64,
            num_tensors: weights.tensors.len(),
        })
    }

    /// Export to GGUF format
    fn export_gguf(&self, weights: &ModelWeights, path: &Path) -> Result<ExportResult> {
        // Mock GGUF export - actual implementation would use gguf crate
        let mut output = Vec::new();

        // GGUF magic number
        output.extend_from_slice(b"GGUF");
        output.extend_from_slice(&3u32.to_le_bytes()); // Version 3

        // Tensor count
        output.extend_from_slice(&(weights.tensors.len() as u64).to_le_bytes());

        // Metadata count (mock)
        output.extend_from_slice(&0u64.to_le_bytes());

        std::fs::write(path, &output).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to write file: {e}"),
        })?;

        Ok(ExportResult {
            path: path.to_path_buf(),
            format: ExportFormat::GGUF,
            size_bytes: output.len() as u64,
            num_tensors: weights.tensors.len(),
        })
    }

    /// Export with automatic format detection from filename
    pub fn export_auto(
        &self,
        weights: &ModelWeights,
        filename: impl AsRef<Path>,
    ) -> Result<ExportResult> {
        let path = filename.as_ref();
        let format = ExportFormat::from_path(path).unwrap_or(self.default_format);
        self.export(weights, format, path)
    }
}
