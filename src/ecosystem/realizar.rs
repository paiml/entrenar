//! Realizar GGUF Export Integration (ENT-032)
//!
//! Provides GGUF model export with quantization support via Realizar.
//! Includes experiment provenance tracking in model metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Errors that can occur during GGUF export.
#[derive(Debug, thiserror::Error)]
pub enum GgufExportError {
    /// Invalid quantization configuration
    #[error("Invalid quantization: {0}")]
    InvalidQuantization(String),

    /// Model data validation failed
    #[error("Model validation failed: {0}")]
    ValidationFailed(String),

    /// I/O error during export
    #[error("Export I/O error: {0}")]
    IoError(String),

    /// Unsupported model architecture
    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Metadata serialization error
    #[error("Metadata error: {0}")]
    MetadataError(String),
}

/// Quantization type for GGUF export.
///
/// These correspond to llama.cpp quantization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 4-bit quantization with k-quants (recommended for most use cases)
    Q4KM,
    /// 5-bit quantization with k-quants (higher quality than Q4_K_M)
    Q5KM,
    /// 8-bit quantization (highest quality, larger size)
    Q80,
    /// 16-bit floating point (no quantization)
    F16,
    /// 32-bit floating point (no quantization, largest)
    F32,
    /// 2-bit quantization (extreme compression, quality loss)
    Q2K,
    /// 3-bit quantization (aggressive compression)
    Q3KM,
    /// 6-bit quantization (high quality)
    Q6K,
}

impl QuantizationType {
    /// Get the GGUF type string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Q4KM => "Q4_K_M",
            Self::Q5KM => "Q5_K_M",
            Self::Q80 => "Q8_0",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::Q2K => "Q2_K",
            Self::Q3KM => "Q3_K_M",
            Self::Q6K => "Q6_K",
        }
    }

    /// Get estimated bits per weight.
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Q2K => 2.5,
            Self::Q3KM => 3.5,
            Self::Q4KM => 4.5,
            Self::Q5KM => 5.5,
            Self::Q6K => 6.5,
            Self::Q80 => 8.0,
            Self::F16 => 16.0,
            Self::F32 => 32.0,
        }
    }

    /// Get relative quality score (0-100).
    pub fn quality_score(&self) -> u8 {
        match self {
            Self::Q2K => 50,
            Self::Q3KM => 65,
            Self::Q4KM => 78,
            Self::Q5KM => 85,
            Self::Q6K => 92,
            Self::Q80 => 97,
            Self::F16 => 100,
            Self::F32 => 100,
        }
    }

    /// Estimate output size given input size in bytes.
    pub fn estimate_size(&self, original_bytes: u64) -> u64 {
        let ratio = self.bits_per_weight() / 32.0;
        (original_bytes as f32 * ratio) as u64
    }

    /// Parse from string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        let normalized = s.to_uppercase().replace(['-', '_'], "");
        match normalized.as_str() {
            "Q4KM" | "Q4K" => Some(Self::Q4KM),
            "Q5KM" | "Q5K" => Some(Self::Q5KM),
            "Q80" | "Q8" => Some(Self::Q80),
            "F16" | "FP16" => Some(Self::F16),
            "F32" | "FP32" => Some(Self::F32),
            "Q2K" | "Q2" => Some(Self::Q2K),
            "Q3KM" | "Q3K" => Some(Self::Q3KM),
            "Q6K" | "Q6" => Some(Self::Q6K),
            _ => None,
        }
    }
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Experiment provenance for tracking model lineage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentProvenance {
    /// Experiment identifier
    pub experiment_id: String,
    /// Run identifier within experiment
    pub run_id: String,
    /// Training configuration hash
    pub config_hash: String,
    /// Dataset identifier
    pub dataset_id: Option<String>,
    /// Base model identifier (for fine-tuned models)
    pub base_model_id: Option<String>,
    /// Training metrics at export time
    pub metrics: HashMap<String, f64>,
    /// Timestamp of export
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Git commit hash (if available)
    pub git_commit: Option<String>,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl ExperimentProvenance {
    /// Create new provenance with required fields.
    pub fn new(experiment_id: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            experiment_id: experiment_id.into(),
            run_id: run_id.into(),
            config_hash: String::new(),
            dataset_id: None,
            base_model_id: None,
            metrics: HashMap::new(),
            timestamp: chrono::Utc::now(),
            git_commit: None,
            custom: HashMap::new(),
        }
    }

    /// Set configuration hash.
    pub fn with_config_hash(mut self, hash: impl Into<String>) -> Self {
        self.config_hash = hash.into();
        self
    }

    /// Set dataset identifier.
    pub fn with_dataset(mut self, dataset_id: impl Into<String>) -> Self {
        self.dataset_id = Some(dataset_id.into());
        self
    }

    /// Set base model identifier.
    pub fn with_base_model(mut self, model_id: impl Into<String>) -> Self {
        self.base_model_id = Some(model_id.into());
        self
    }

    /// Add a metric.
    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    /// Add multiple metrics.
    pub fn with_metrics(mut self, metrics: impl IntoIterator<Item = (String, f64)>) -> Self {
        self.metrics.extend(metrics);
        self
    }

    /// Set git commit hash.
    pub fn with_git_commit(mut self, commit: impl Into<String>) -> Self {
        self.git_commit = Some(commit.into());
        self
    }

    /// Add custom metadata.
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Convert to GGUF metadata key-value pairs.
    pub fn to_metadata_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = vec![
            ("entrenar.experiment_id".to_string(), self.experiment_id.clone()),
            ("entrenar.run_id".to_string(), self.run_id.clone()),
            ("entrenar.timestamp".to_string(), self.timestamp.to_rfc3339()),
        ];

        if !self.config_hash.is_empty() {
            pairs.push(("entrenar.config_hash".to_string(), self.config_hash.clone()));
        }

        if let Some(ref dataset) = self.dataset_id {
            pairs.push(("entrenar.dataset_id".to_string(), dataset.clone()));
        }

        if let Some(ref base) = self.base_model_id {
            pairs.push(("entrenar.base_model_id".to_string(), base.clone()));
        }

        if let Some(ref commit) = self.git_commit {
            pairs.push(("entrenar.git_commit".to_string(), commit.clone()));
        }

        for (key, value) in &self.metrics {
            pairs.push((format!("entrenar.metric.{key}"), value.to_string()));
        }

        for (key, value) in &self.custom {
            pairs.push((format!("entrenar.custom.{key}"), value.clone()));
        }

        pairs
    }
}

/// GGUF metadata container.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GgufMetadata {
    /// General model information
    pub general: GeneralMetadata,
    /// Experiment provenance (optional)
    pub provenance: Option<ExperimentProvenance>,
    /// Custom key-value pairs
    pub custom: HashMap<String, String>,
}

/// General model metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneralMetadata {
    /// Model architecture (e.g., "llama", "mistral", "qwen")
    pub architecture: String,
    /// Model name
    pub name: String,
    /// Author or organization
    pub author: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// License identifier
    pub license: Option<String>,
    /// URL for more information
    pub url: Option<String>,
    /// File type (quantization level)
    pub file_type: Option<String>,
}

impl GeneralMetadata {
    /// Create new general metadata.
    pub fn new(architecture: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            architecture: architecture.into(),
            name: name.into(),
            author: None,
            description: None,
            license: None,
            url: None,
            file_type: None,
        }
    }

    /// Set author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set license.
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }
}

/// GGUF exporter for model conversion.
#[derive(Debug, Clone)]
pub struct GgufExporter {
    /// Quantization type to apply
    quantization: QuantizationType,
    /// Metadata to embed
    metadata: GgufMetadata,
    /// Whether to validate model structure
    validate: bool,
    /// Number of threads for quantization
    threads: usize,
}

impl Default for GgufExporter {
    fn default() -> Self {
        Self::new(QuantizationType::Q4KM)
    }
}

impl GgufExporter {
    /// Create a new exporter with specified quantization.
    pub fn new(quantization: QuantizationType) -> Self {
        Self {
            quantization,
            metadata: GgufMetadata::default(),
            validate: true,
            threads: num_cpus(),
        }
    }

    /// Set metadata.
    pub fn with_metadata(mut self, metadata: GgufMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set general metadata.
    pub fn with_general(mut self, general: GeneralMetadata) -> Self {
        self.metadata.general = general;
        self
    }

    /// Set experiment provenance.
    pub fn with_provenance(mut self, provenance: ExperimentProvenance) -> Self {
        self.metadata.provenance = Some(provenance);
        self
    }

    /// Disable validation.
    pub fn without_validation(mut self) -> Self {
        self.validate = false;
        self
    }

    /// Set thread count.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Get the quantization type.
    pub fn quantization(&self) -> QuantizationType {
        self.quantization
    }

    /// Get the metadata.
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Export model to GGUF format.
    ///
    /// This is a placeholder that prepares export configuration.
    /// Actual export requires Realizar crate integration.
    pub fn export(
        &self,
        _input_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
    ) -> Result<GgufExportResult, GgufExportError> {
        let output = output_path.as_ref();

        // Validate output path
        if let Some(parent) = output.parent() {
            if !parent.exists() {
                return Err(GgufExportError::IoError(format!(
                    "Output directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        // In a real implementation, this would:
        // 1. Load model from input_path using Realizar
        // 2. Apply quantization
        // 3. Embed metadata
        // 4. Write to output_path

        // Return export result with metadata
        Ok(GgufExportResult {
            output_path: output.to_path_buf(),
            quantization: self.quantization,
            metadata_keys: self.metadata.provenance.as_ref().map_or(0, |p| {
                p.to_metadata_pairs().len()
            }) + self.metadata.custom.len(),
            estimated_size_bytes: 0, // Would be calculated from actual model
        })
    }

    /// Collect all metadata as key-value pairs.
    pub fn collect_metadata(&self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();

        // General metadata
        pairs.push((
            "general.architecture".to_string(),
            self.metadata.general.architecture.clone(),
        ));
        pairs.push((
            "general.name".to_string(),
            self.metadata.general.name.clone(),
        ));

        if let Some(ref author) = self.metadata.general.author {
            pairs.push(("general.author".to_string(), author.clone()));
        }
        if let Some(ref desc) = self.metadata.general.description {
            pairs.push(("general.description".to_string(), desc.clone()));
        }
        if let Some(ref license) = self.metadata.general.license {
            pairs.push(("general.license".to_string(), license.clone()));
        }
        if let Some(ref url) = self.metadata.general.url {
            pairs.push(("general.url".to_string(), url.clone()));
        }

        pairs.push((
            "general.file_type".to_string(),
            self.quantization.as_str().to_string(),
        ));

        // Provenance metadata
        if let Some(ref prov) = self.metadata.provenance {
            pairs.extend(prov.to_metadata_pairs());
        }

        // Custom metadata
        for (key, value) in &self.metadata.custom {
            pairs.push((format!("custom.{key}"), value.clone()));
        }

        pairs
    }
}

/// Result of a GGUF export operation.
#[derive(Debug, Clone)]
pub struct GgufExportResult {
    /// Path to exported file
    pub output_path: std::path::PathBuf,
    /// Quantization type used
    pub quantization: QuantizationType,
    /// Number of metadata keys embedded
    pub metadata_keys: usize,
    /// Estimated file size in bytes
    pub estimated_size_bytes: u64,
}

/// Get number of CPUs (simplified).
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_as_str() {
        assert_eq!(QuantizationType::Q4KM.as_str(), "Q4_K_M");
        assert_eq!(QuantizationType::Q5KM.as_str(), "Q5_K_M");
        assert_eq!(QuantizationType::Q80.as_str(), "Q8_0");
        assert_eq!(QuantizationType::F16.as_str(), "F16");
        assert_eq!(QuantizationType::F32.as_str(), "F32");
    }

    #[test]
    fn test_quantization_type_bits() {
        assert!((QuantizationType::Q4KM.bits_per_weight() - 4.5).abs() < f32::EPSILON);
        assert!((QuantizationType::Q80.bits_per_weight() - 8.0).abs() < f32::EPSILON);
        assert!((QuantizationType::F16.bits_per_weight() - 16.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quantization_type_quality() {
        assert!(QuantizationType::Q2K.quality_score() < QuantizationType::Q4KM.quality_score());
        assert!(QuantizationType::Q4KM.quality_score() < QuantizationType::Q80.quality_score());
        assert!(QuantizationType::Q80.quality_score() < QuantizationType::F16.quality_score());
    }

    #[test]
    fn test_quantization_type_estimate_size() {
        let original = 1_000_000_000u64; // 1GB (F32 weights)

        let q4_size = QuantizationType::Q4KM.estimate_size(original);
        let f16_size = QuantizationType::F16.estimate_size(original);

        assert!(q4_size < f16_size);
        assert!(f16_size < original);
    }

    #[test]
    fn test_quantization_type_parse() {
        assert_eq!(QuantizationType::parse("Q4_K_M"), Some(QuantizationType::Q4KM));
        assert_eq!(QuantizationType::parse("q4km"), Some(QuantizationType::Q4KM));
        assert_eq!(QuantizationType::parse("F16"), Some(QuantizationType::F16));
        assert_eq!(QuantizationType::parse("fp16"), Some(QuantizationType::F16));
        assert_eq!(QuantizationType::parse("invalid"), None);
    }

    #[test]
    fn test_experiment_provenance_creation() {
        let prov = ExperimentProvenance::new("exp-001", "run-123")
            .with_config_hash("abc123")
            .with_dataset("imagenet-1k")
            .with_base_model("llama-7b")
            .with_metric("loss", 0.125)
            .with_metric("accuracy", 0.92)
            .with_git_commit("deadbeef")
            .with_custom("framework", "entrenar");

        assert_eq!(prov.experiment_id, "exp-001");
        assert_eq!(prov.run_id, "run-123");
        assert_eq!(prov.config_hash, "abc123");
        assert_eq!(prov.dataset_id, Some("imagenet-1k".to_string()));
        assert_eq!(prov.base_model_id, Some("llama-7b".to_string()));
        assert_eq!(prov.metrics.get("loss"), Some(&0.125));
        assert_eq!(prov.metrics.get("accuracy"), Some(&0.92));
        assert_eq!(prov.git_commit, Some("deadbeef".to_string()));
        assert_eq!(prov.custom.get("framework"), Some(&"entrenar".to_string()));
    }

    #[test]
    fn test_experiment_provenance_to_metadata() {
        let prov = ExperimentProvenance::new("exp-001", "run-123")
            .with_metric("loss", 0.125);

        let pairs = prov.to_metadata_pairs();

        assert!(pairs.iter().any(|(k, v)| k == "entrenar.experiment_id" && v == "exp-001"));
        assert!(pairs.iter().any(|(k, v)| k == "entrenar.run_id" && v == "run-123"));
        assert!(pairs.iter().any(|(k, _)| k == "entrenar.timestamp"));
        assert!(pairs.iter().any(|(k, _)| k == "entrenar.metric.loss"));
    }

    #[test]
    fn test_general_metadata_creation() {
        let general = GeneralMetadata::new("llama", "my-model")
            .with_author("PAIML")
            .with_description("Fine-tuned LLaMA model")
            .with_license("MIT");

        assert_eq!(general.architecture, "llama");
        assert_eq!(general.name, "my-model");
        assert_eq!(general.author, Some("PAIML".to_string()));
        assert_eq!(general.description, Some("Fine-tuned LLaMA model".to_string()));
        assert_eq!(general.license, Some("MIT".to_string()));
    }

    #[test]
    fn test_gguf_exporter_creation() {
        let exporter = GgufExporter::new(QuantizationType::Q5KM)
            .with_threads(8)
            .without_validation();

        assert_eq!(exporter.quantization(), QuantizationType::Q5KM);
    }

    #[test]
    fn test_gguf_exporter_with_provenance() {
        let prov = ExperimentProvenance::new("exp-001", "run-123");
        let general = GeneralMetadata::new("llama", "test-model");

        let exporter = GgufExporter::new(QuantizationType::Q4KM)
            .with_general(general)
            .with_provenance(prov);

        assert!(exporter.metadata().provenance.is_some());
        assert_eq!(exporter.metadata().general.architecture, "llama");
    }

    #[test]
    fn test_gguf_exporter_collect_metadata() {
        let prov = ExperimentProvenance::new("exp-001", "run-123")
            .with_metric("loss", 0.1);
        let general = GeneralMetadata::new("llama", "test-model")
            .with_author("PAIML");

        let exporter = GgufExporter::new(QuantizationType::Q4KM)
            .with_general(general)
            .with_provenance(prov);

        let pairs = exporter.collect_metadata();

        assert!(pairs.iter().any(|(k, v)| k == "general.architecture" && v == "llama"));
        assert!(pairs.iter().any(|(k, v)| k == "general.name" && v == "test-model"));
        assert!(pairs.iter().any(|(k, v)| k == "general.author" && v == "PAIML"));
        assert!(pairs.iter().any(|(k, v)| k == "general.file_type" && v == "Q4_K_M"));
        assert!(pairs.iter().any(|(k, _)| k == "entrenar.experiment_id"));
    }

    #[test]
    fn test_gguf_export_result() {
        let result = GgufExportResult {
            output_path: std::path::PathBuf::from("/tmp/model.gguf"),
            quantization: QuantizationType::Q4KM,
            metadata_keys: 10,
            estimated_size_bytes: 4_000_000_000,
        };

        assert_eq!(result.quantization, QuantizationType::Q4KM);
        assert_eq!(result.metadata_keys, 10);
    }
}
