//! Model Export Formats
//!
//! Supports exporting trained/distilled models to various formats:
//! - SafeTensors: Safe, fast tensor serialization
//! - APR: Aprender format for entrenar ecosystem
//! - GGUF: Quantized format for llama.cpp compatibility
//!
//! # Example
//!
//! ```ignore
//! use entrenar::hf_pipeline::export::{Exporter, ExportFormat};
//!
//! let exporter = Exporter::new();
//! exporter.export(&model_weights, ExportFormat::SafeTensors, "output/model.safetensors")?;
//! ```

use crate::hf_pipeline::error::{FetchError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Export format selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// SafeTensors format (recommended)
    SafeTensors,
    /// Aprender format (JSON-based)
    APR,
    /// GGUF quantized format
    GGUF,
    /// PyTorch state dict (for compatibility)
    PyTorch,
}

impl ExportFormat {
    /// Get file extension for format
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::APR => "apr.json",
            Self::GGUF => "gguf",
            Self::PyTorch => "pt",
        }
    }

    /// Check if format is safe (no pickle/arbitrary code)
    #[must_use]
    pub fn is_safe(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::APR | Self::GGUF)
    }

    /// Detect format from file path
    #[must_use]
    pub fn from_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?;
        if name.ends_with(".safetensors") {
            Some(Self::SafeTensors)
        } else if name.ends_with(".apr.json") || name.ends_with(".apr") {
            Some(Self::APR)
        } else if name.ends_with(".gguf") {
            Some(Self::GGUF)
        } else if name.ends_with(".pt") || name.ends_with(".bin") {
            Some(Self::PyTorch)
        } else {
            None
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::APR => write!(f, "APR"),
            Self::GGUF => write!(f, "GGUF"),
            Self::PyTorch => write!(f, "PyTorch"),
        }
    }
}

/// Tensor metadata for export
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub(super) struct TensorMetadata {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: DataType,
    /// Shape
    pub shape: Vec<usize>,
    /// Byte offset in file
    pub offset: usize,
    /// Byte size
    pub size: usize,
}

/// Supported data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub(super) enum DataType {
    F32,
    F16,
    BF16,
    I32,
    I8,
    U8,
    Q4_0,
    Q8_0,
}

#[allow(dead_code)]
impl DataType {
    /// Bytes per element
    #[must_use]
    pub(super) fn bytes_per_element(&self) -> f32 {
        match self {
            Self::F32 | Self::I32 => 4.0,
            Self::F16 | Self::BF16 => 2.0,
            Self::I8 | Self::U8 => 1.0,
            Self::Q4_0 => 0.5,
            Self::Q8_0 => 1.0,
        }
    }
}

/// Model weights container for export
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Tensor data by name
    pub tensors: HashMap<String, Vec<f32>>,
    /// Tensor shapes by name
    pub shapes: HashMap<String, Vec<usize>>,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Model metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model architecture
    pub architecture: Option<String>,
    /// Model name
    pub model_name: Option<String>,
    /// Number of parameters
    pub num_params: u64,
    /// Hidden size
    pub hidden_size: Option<usize>,
    /// Number of layers
    pub num_layers: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Training info
    pub training: Option<TrainingMetadata>,
}

/// Training metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Training epochs completed
    pub epochs: usize,
    /// Final training loss
    pub final_loss: Option<f32>,
    /// Final validation loss
    pub final_val_loss: Option<f32>,
    /// Learning rate used
    pub learning_rate: Option<f64>,
    /// Batch size used
    pub batch_size: Option<usize>,
    /// Distillation temperature (if applicable)
    pub temperature: Option<f32>,
    /// Teacher model (if distilled)
    pub teacher_model: Option<String>,
}

impl ModelWeights {
    /// Create new empty weights container
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            shapes: HashMap::new(),
            metadata: ModelMetadata::default(),
        }
    }

    /// Add a tensor
    pub fn add_tensor(&mut self, name: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) {
        let name = name.into();
        self.tensors.insert(name.clone(), data);
        self.shapes.insert(name, shape);
    }

    /// Get tensor by name
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<(&Vec<f32>, &Vec<usize>)> {
        let data = self.tensors.get(name)?;
        let shape = self.shapes.get(name)?;
        Some((data, shape))
    }

    /// Get all tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Count total parameters
    #[must_use]
    pub fn param_count(&self) -> u64 {
        self.tensors.values().map(|t| t.len() as u64).sum()
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Create mock weights for testing
    #[must_use]
    pub fn mock(num_layers: usize, hidden_size: usize) -> Self {
        let mut weights = Self::new();

        for layer in 0..num_layers {
            // Q, K, V, O projections
            for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                let name = format!("layer.{layer}.attention.{proj}.weight");
                let size = hidden_size * hidden_size;
                let data = vec![0.01; size];
                weights.add_tensor(name, data, vec![hidden_size, hidden_size]);
            }

            // MLP layers
            let mlp_size = hidden_size * 4;
            weights.add_tensor(
                format!("layer.{layer}.mlp.up.weight"),
                vec![0.01; hidden_size * mlp_size],
                vec![mlp_size, hidden_size],
            );
            weights.add_tensor(
                format!("layer.{layer}.mlp.down.weight"),
                vec![0.01; mlp_size * hidden_size],
                vec![hidden_size, mlp_size],
            );
        }

        weights.metadata = ModelMetadata {
            num_params: weights.param_count(),
            hidden_size: Some(hidden_size),
            num_layers: Some(num_layers),
            ..Default::default()
        };

        weights
    }
}

impl Default for ModelWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Model exporter
pub struct Exporter {
    /// Output directory
    output_dir: PathBuf,
    /// Default format
    default_format: ExportFormat,
    /// Include metadata
    include_metadata: bool,
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

/// Export result
#[derive(Debug, Clone)]
pub struct ExportResult {
    /// Output path
    pub path: PathBuf,
    /// Format used
    pub format: ExportFormat,
    /// File size in bytes
    pub size_bytes: u64,
    /// Number of tensors exported
    pub num_tensors: usize,
}

impl ExportResult {
    /// Format size as human-readable string
    #[must_use]
    pub fn size_human(&self) -> String {
        if self.size_bytes >= 1_000_000_000 {
            format!("{:.2} GB", self.size_bytes as f64 / 1e9)
        } else if self.size_bytes >= 1_000_000 {
            format!("{:.2} MB", self.size_bytes as f64 / 1e6)
        } else if self.size_bytes >= 1_000 {
            format!("{:.2} KB", self.size_bytes as f64 / 1e3)
        } else {
            format!("{} B", self.size_bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ExportFormat Tests
    // =========================================================================

    #[test]
    fn test_format_extension() {
        assert_eq!(ExportFormat::SafeTensors.extension(), "safetensors");
        assert_eq!(ExportFormat::APR.extension(), "apr.json");
        assert_eq!(ExportFormat::GGUF.extension(), "gguf");
        assert_eq!(ExportFormat::PyTorch.extension(), "pt");
    }

    #[test]
    fn test_format_is_safe() {
        assert!(ExportFormat::SafeTensors.is_safe());
        assert!(ExportFormat::APR.is_safe());
        assert!(ExportFormat::GGUF.is_safe());
        assert!(!ExportFormat::PyTorch.is_safe());
    }

    #[test]
    fn test_format_from_path() {
        assert_eq!(
            ExportFormat::from_path(Path::new("model.safetensors")),
            Some(ExportFormat::SafeTensors)
        );
        assert_eq!(
            ExportFormat::from_path(Path::new("model.apr.json")),
            Some(ExportFormat::APR)
        );
        assert_eq!(
            ExportFormat::from_path(Path::new("model.gguf")),
            Some(ExportFormat::GGUF)
        );
        assert_eq!(
            ExportFormat::from_path(Path::new("model.pt")),
            Some(ExportFormat::PyTorch)
        );
        assert_eq!(ExportFormat::from_path(Path::new("model.txt")), None);
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", ExportFormat::SafeTensors), "SafeTensors");
        assert_eq!(format!("{}", ExportFormat::APR), "APR");
    }

    // =========================================================================
    // DataType Tests
    // =========================================================================

    #[test]
    fn test_dtype_bytes() {
        assert_eq!(DataType::F32.bytes_per_element(), 4.0);
        assert_eq!(DataType::F16.bytes_per_element(), 2.0);
        assert_eq!(DataType::Q4_0.bytes_per_element(), 0.5);
    }

    // =========================================================================
    // ModelWeights Tests
    // =========================================================================

    #[test]
    fn test_weights_new() {
        let weights = ModelWeights::new();
        assert!(weights.tensors.is_empty());
        assert_eq!(weights.param_count(), 0);
    }

    #[test]
    fn test_weights_add_tensor() {
        let mut weights = ModelWeights::new();
        weights.add_tensor("layer.0.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        assert_eq!(weights.tensors.len(), 1);
        assert_eq!(weights.param_count(), 4);

        let (data, shape) = weights.get_tensor("layer.0.weight").unwrap();
        assert_eq!(data.len(), 4);
        assert_eq!(shape, &vec![2, 2]);
    }

    #[test]
    fn test_weights_tensor_names() {
        let mut weights = ModelWeights::new();
        weights.add_tensor("a", vec![1.0], vec![1]);
        weights.add_tensor("b", vec![2.0], vec![1]);

        let names = weights.tensor_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }

    #[test]
    fn test_weights_mock() {
        let weights = ModelWeights::mock(4, 256);

        // 4 layers * (4 attention + 2 MLP) = 24 tensors
        assert_eq!(weights.tensors.len(), 24);
        assert!(weights.param_count() > 0);
        assert_eq!(weights.metadata.num_layers, Some(4));
        assert_eq!(weights.metadata.hidden_size, Some(256));
    }

    #[test]
    fn test_weights_with_metadata() {
        let metadata = ModelMetadata {
            model_name: Some("test_model".to_string()),
            num_params: 1000,
            ..Default::default()
        };

        let weights = ModelWeights::new().with_metadata(metadata);
        assert_eq!(weights.metadata.model_name, Some("test_model".to_string()));
    }

    // =========================================================================
    // Exporter Tests
    // =========================================================================

    #[test]
    fn test_exporter_new() {
        let exporter = Exporter::new();
        assert_eq!(exporter.default_format, ExportFormat::SafeTensors);
        assert!(exporter.include_metadata);
    }

    #[test]
    fn test_exporter_builder() {
        let exporter = Exporter::new()
            .output_dir("/tmp/export")
            .default_format(ExportFormat::APR)
            .include_metadata(false);

        assert_eq!(exporter.output_dir, PathBuf::from("/tmp/export"));
        assert_eq!(exporter.default_format, ExportFormat::APR);
        assert!(!exporter.include_metadata);
    }

    #[test]
    fn test_export_safetensors() {
        let weights = ModelWeights::mock(2, 64);
        let exporter = Exporter::new().output_dir("/tmp");

        let result = exporter.export(
            &weights,
            ExportFormat::SafeTensors,
            "test_model.safetensors",
        );
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.format, ExportFormat::SafeTensors);
        assert!(result.size_bytes > 0);

        // Cleanup
        std::fs::remove_file(&result.path).ok();
    }

    #[test]
    fn test_export_apr() {
        let weights = ModelWeights::mock(2, 64);
        let exporter = Exporter::new().output_dir("/tmp");

        let result = exporter.export(&weights, ExportFormat::APR, "test_model.apr.json");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.format, ExportFormat::APR);

        // Verify JSON is valid
        let content = std::fs::read_to_string(&result.path).unwrap();
        assert!(content.contains("\"version\""));
        assert!(content.contains("\"tensors\""));

        // Cleanup
        std::fs::remove_file(&result.path).ok();
    }

    #[test]
    fn test_export_gguf() {
        let weights = ModelWeights::mock(2, 64);
        let exporter = Exporter::new().output_dir("/tmp");

        let result = exporter.export(&weights, ExportFormat::GGUF, "test_model.gguf");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.format, ExportFormat::GGUF);

        // Cleanup
        std::fs::remove_file(&result.path).ok();
    }

    #[test]
    fn test_export_pytorch_rejected() {
        let weights = ModelWeights::mock(2, 64);
        let exporter = Exporter::new().output_dir("/tmp");

        let result = exporter.export(&weights, ExportFormat::PyTorch, "test_model.pt");
        assert!(result.is_err());
        assert!(matches!(result, Err(FetchError::PickleSecurityRisk)));
    }

    #[test]
    fn test_export_auto() {
        let weights = ModelWeights::mock(2, 64);
        let exporter = Exporter::new().output_dir("/tmp");

        let result = exporter.export_auto(&weights, "test_auto.safetensors");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().format, ExportFormat::SafeTensors);

        // Cleanup
        std::fs::remove_file("/tmp/test_auto.safetensors").ok();
    }

    // =========================================================================
    // ExportResult Tests
    // =========================================================================

    #[test]
    fn test_result_size_human() {
        let result = ExportResult {
            path: PathBuf::from("test"),
            format: ExportFormat::SafeTensors,
            size_bytes: 500,
            num_tensors: 1,
        };
        assert_eq!(result.size_human(), "500 B");

        let result = ExportResult {
            size_bytes: 1_500_000,
            ..result
        };
        assert!(result.size_human().contains("MB"));

        let result = ExportResult {
            size_bytes: 2_500_000_000,
            ..result
        };
        assert!(result.size_human().contains("GB"));
    }

    // =========================================================================
    // TrainingMetadata Tests
    // =========================================================================

    #[test]
    fn test_training_metadata() {
        let meta = TrainingMetadata {
            epochs: 10,
            final_loss: Some(0.5),
            teacher_model: Some("bert-base".to_string()),
            ..Default::default()
        };

        assert_eq!(meta.epochs, 10);
        assert_eq!(meta.final_loss, Some(0.5));
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_dtype_bytes_all_types() {
        // Cover all DataType branches
        assert_eq!(DataType::I32.bytes_per_element(), 4.0);
        assert_eq!(DataType::BF16.bytes_per_element(), 2.0);
        assert_eq!(DataType::I8.bytes_per_element(), 1.0);
        assert_eq!(DataType::U8.bytes_per_element(), 1.0);
        assert_eq!(DataType::Q8_0.bytes_per_element(), 1.0);
    }

    #[test]
    fn test_format_from_path_apr_short() {
        // Test .apr without .json
        assert_eq!(
            ExportFormat::from_path(Path::new("model.apr")),
            Some(ExportFormat::APR)
        );
    }

    #[test]
    fn test_format_from_path_bin() {
        // Test .bin (PyTorch)
        assert_eq!(
            ExportFormat::from_path(Path::new("model.bin")),
            Some(ExportFormat::PyTorch)
        );
    }

    #[test]
    fn test_format_display_gguf_pytorch() {
        assert_eq!(format!("{}", ExportFormat::GGUF), "GGUF");
        assert_eq!(format!("{}", ExportFormat::PyTorch), "PyTorch");
    }

    #[test]
    fn test_weights_get_tensor_not_found() {
        let weights = ModelWeights::new();
        assert!(weights.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_result_size_kb() {
        let result = ExportResult {
            path: PathBuf::from("test"),
            format: ExportFormat::SafeTensors,
            size_bytes: 5_000,
            num_tensors: 1,
        };
        let size_str = result.size_human();
        assert!(size_str.contains("KB"), "Expected KB, got: {size_str}");
    }

    #[test]
    fn test_model_metadata_default() {
        let meta = ModelMetadata::default();
        assert!(meta.architecture.is_none());
        assert!(meta.model_name.is_none());
        assert_eq!(meta.num_params, 0);
    }

    #[test]
    fn test_training_metadata_full() {
        let meta = TrainingMetadata {
            epochs: 5,
            final_loss: Some(0.1),
            final_val_loss: Some(0.15),
            learning_rate: Some(1e-4),
            batch_size: Some(32),
            temperature: Some(2.0),
            teacher_model: Some("gpt2".to_string()),
        };
        assert_eq!(meta.batch_size, Some(32));
        assert_eq!(meta.temperature, Some(2.0));
    }

    #[test]
    fn test_weights_default() {
        let weights = ModelWeights::default();
        assert!(weights.tensors.is_empty());
    }

    #[test]
    fn test_exporter_default() {
        let exporter = Exporter::default();
        assert_eq!(exporter.default_format, ExportFormat::SafeTensors);
    }

    #[test]
    fn test_export_result_clone() {
        let result = ExportResult {
            path: PathBuf::from("/tmp/test"),
            format: ExportFormat::APR,
            size_bytes: 1000,
            num_tensors: 5,
        };
        let cloned = result.clone();
        assert_eq!(result.path, cloned.path);
        assert_eq!(result.num_tensors, cloned.num_tensors);
    }

    #[test]
    fn test_format_from_path_no_extension() {
        // Path with no recognizable extension
        assert!(ExportFormat::from_path(Path::new("model")).is_none());
    }
}
