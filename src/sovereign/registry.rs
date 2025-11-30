//! Offline Model Registry (ENT-017)
//!
//! Provides local model storage and verification for air-gapped deployments.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Model source type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelSource {
    /// HuggingFace Hub model
    HuggingFace {
        /// Repository ID (e.g., "bert-base-uncased")
        repo_id: String,
    },
    /// Local file path
    LocalFile {
        /// Path to the model file
        path: PathBuf,
    },
    /// Custom URL source
    Custom {
        /// Download URL
        url: String,
    },
}

impl ModelSource {
    /// Create HuggingFace source
    pub fn huggingface(repo_id: impl Into<String>) -> Self {
        Self::HuggingFace {
            repo_id: repo_id.into(),
        }
    }

    /// Create local file source
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::LocalFile { path: path.into() }
    }

    /// Create custom URL source
    pub fn custom(url: impl Into<String>) -> Self {
        Self::Custom { url: url.into() }
    }

    /// Get a display string for the source
    pub fn display_string(&self) -> String {
        match self {
            Self::HuggingFace { repo_id } => format!("hf://{repo_id}"),
            Self::LocalFile { path } => format!("file://{}", path.display()),
            Self::Custom { url } => url.clone(),
        }
    }
}

/// Model entry in the registry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Model name (e.g., "bert-base-uncased")
    pub name: String,
    /// Model version
    pub version: String,
    /// SHA-256 checksum
    pub sha256: String,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Model source
    pub source: ModelSource,
    /// Local path if mirrored
    pub local_path: Option<PathBuf>,
    /// Model format (gguf, safetensors, etc.)
    pub format: Option<String>,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

impl ModelEntry {
    /// Create a new model entry
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        sha256: impl Into<String>,
        size_bytes: u64,
        source: ModelSource,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            sha256: sha256.into(),
            size_bytes,
            source,
            local_path: None,
            format: None,
            metadata: HashMap::new(),
        }
    }

    /// Set local path
    pub fn with_local_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.local_path = Some(path.into());
        self
    }

    /// Set format
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if model is available locally
    pub fn is_local(&self) -> bool {
        self.local_path.as_ref().is_some_and(|p| p.exists())
    }

    /// Get size in megabytes
    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get size in gigabytes
    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Registry manifest containing all model entries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegistryManifest {
    /// List of model entries
    pub models: Vec<ModelEntry>,
    /// Last sync timestamp
    pub last_sync: Option<DateTime<Utc>>,
    /// Registry version
    pub version: String,
}

impl RegistryManifest {
    /// Create a new empty manifest
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            last_sync: None,
            version: "1.0".to_string(),
        }
    }

    /// Add a model entry
    pub fn add(&mut self, entry: ModelEntry) {
        // Update or insert
        if let Some(existing) = self.models.iter_mut().find(|m| m.name == entry.name) {
            *existing = entry;
        } else {
            self.models.push(entry);
        }
    }

    /// Find a model by name
    pub fn find(&self, name: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Find a model by name (mutable)
    pub fn find_mut(&mut self, name: &str) -> Option<&mut ModelEntry> {
        self.models.iter_mut().find(|m| m.name == name)
    }

    /// List all available models (those with local paths)
    pub fn available(&self) -> Vec<&ModelEntry> {
        self.models.iter().filter(|m| m.is_local()).collect()
    }

    /// Update sync timestamp
    pub fn mark_synced(&mut self) {
        self.last_sync = Some(Utc::now());
    }

    /// Get total size of all models
    pub fn total_size_bytes(&self) -> u64 {
        self.models.iter().map(|m| m.size_bytes).sum()
    }

    /// Get count of models
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if manifest is empty
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

/// Offline model registry
#[derive(Debug)]
pub struct OfflineModelRegistry {
    /// Root path for model storage
    pub root_path: PathBuf,
    /// Registry manifest
    pub manifest: RegistryManifest,
    /// Manifest file path
    manifest_path: PathBuf,
}

impl OfflineModelRegistry {
    /// Create a new registry at the given root path
    pub fn new(root: PathBuf) -> Self {
        let manifest_path = root.join("manifest.json");
        let manifest = if manifest_path.exists() {
            Self::load_manifest(&manifest_path).unwrap_or_default()
        } else {
            RegistryManifest::new()
        };

        Self {
            root_path: root,
            manifest,
            manifest_path,
        }
    }

    /// Create registry at default location (~/.entrenar/models/)
    pub fn default_location() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        Self::new(home.join(".entrenar").join("models"))
    }

    /// Load manifest from file
    fn load_manifest(path: &Path) -> Result<RegistryManifest> {
        let content = fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| Error::Io(format!("Invalid manifest data: {e}")))
    }

    /// Save manifest to file
    pub fn save_manifest(&self) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.manifest_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| Error::Io(format!("Failed to serialize manifest: {e}")))?;
        fs::write(&self.manifest_path, content)?;
        Ok(())
    }

    /// Add a model entry to the registry
    pub fn add_model(&mut self, entry: ModelEntry) {
        self.manifest.add(entry);
    }

    /// Mirror a model from HuggingFace Hub (simulated for offline scenarios)
    ///
    /// In a real implementation, this would download the model.
    /// For air-gapped scenarios, models are pre-downloaded and registered.
    pub fn mirror_from_hub(&mut self, repo_id: &str) -> Result<ModelEntry> {
        // Create model entry with HuggingFace source
        let name = repo_id.split('/').next_back().unwrap_or(repo_id);
        let local_path = self.root_path.join(name);

        let entry = ModelEntry::new(
            name,
            "1.0",
            "", // Checksum computed after download
            0,  // Size computed after download
            ModelSource::huggingface(repo_id),
        )
        .with_local_path(&local_path);

        self.manifest.add(entry.clone());
        Ok(entry)
    }

    /// Register a local model file
    pub fn register_local(&mut self, name: &str, path: &Path) -> Result<ModelEntry> {
        if !path.exists() {
            return Err(Error::ConfigError(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        let metadata = fs::metadata(path)?;
        let size_bytes = metadata.len();

        // Compute SHA-256
        let sha256 = Self::compute_file_sha256(path)?;

        // Determine format from extension
        let format = path.extension().and_then(|e| e.to_str()).map(String::from);

        let entry = ModelEntry::new(name, "local", sha256, size_bytes, ModelSource::local(path))
            .with_local_path(path);

        let entry = if let Some(fmt) = format {
            entry.with_format(fmt)
        } else {
            entry
        };

        self.manifest.add(entry.clone());
        self.manifest.mark_synced();
        self.save_manifest()?;

        Ok(entry)
    }

    /// Compute SHA-256 hash of a file
    fn compute_file_sha256(path: &Path) -> Result<String> {
        let mut file = fs::File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Load a model by name, returning its local path
    pub fn load(&self, name: &str) -> Result<PathBuf> {
        let entry = self
            .manifest
            .find(name)
            .ok_or_else(|| Error::ConfigError(format!("Model not found: {name}")))?;

        let path = entry
            .local_path
            .as_ref()
            .ok_or_else(|| Error::ConfigError(format!("Model not available locally: {name}")))?;

        if !path.exists() {
            return Err(Error::ConfigError(format!(
                "Model file missing: {}",
                path.display()
            )));
        }

        Ok(path.clone())
    }

    /// Verify a model entry's checksum
    pub fn verify(&self, entry: &ModelEntry) -> Result<bool> {
        let path = entry
            .local_path
            .as_ref()
            .ok_or_else(|| Error::ConfigError("Model has no local path".into()))?;

        if !path.exists() {
            return Ok(false);
        }

        if entry.sha256.is_empty() {
            // No checksum to verify against
            return Ok(true);
        }

        let computed = Self::compute_file_sha256(path)?;
        Ok(computed == entry.sha256)
    }

    /// List all available (locally cached) models
    pub fn list_available(&self) -> Vec<&ModelEntry> {
        self.manifest.available()
    }

    /// List all models in registry
    pub fn list_all(&self) -> &[ModelEntry] {
        &self.manifest.models
    }

    /// Get a model entry by name
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.manifest.find(name)
    }

    /// Remove a model from registry (does not delete files)
    pub fn remove(&mut self, name: &str) -> Option<ModelEntry> {
        let pos = self.manifest.models.iter().position(|m| m.name == name)?;
        Some(self.manifest.models.remove(pos))
    }

    /// Get total size of all models
    pub fn total_size(&self) -> u64 {
        self.manifest.total_size_bytes()
    }

    /// Get root path
    pub fn root(&self) -> &Path {
        &self.root_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_source_huggingface() {
        let source = ModelSource::huggingface("bert-base-uncased");
        assert_eq!(source.display_string(), "hf://bert-base-uncased");
    }

    #[test]
    fn test_model_source_local() {
        let source = ModelSource::local("/path/to/model.gguf");
        assert!(source.display_string().contains("/path/to/model.gguf"));
    }

    #[test]
    fn test_model_source_custom() {
        let source = ModelSource::custom("https://example.com/model.bin");
        assert_eq!(source.display_string(), "https://example.com/model.bin");
    }

    #[test]
    fn test_model_entry_new() {
        let entry = ModelEntry::new(
            "test-model",
            "1.0.0",
            "abc123",
            1024 * 1024 * 100, // 100 MB
            ModelSource::huggingface("test/model"),
        );

        assert_eq!(entry.name, "test-model");
        assert_eq!(entry.version, "1.0.0");
        assert_eq!(entry.sha256, "abc123");
        assert_eq!(entry.size_bytes, 100 * 1024 * 1024);
        assert!(!entry.is_local());
    }

    #[test]
    fn test_model_entry_size_conversions() {
        let entry = ModelEntry::new(
            "test",
            "1.0",
            "",
            1024 * 1024 * 1024, // 1 GB
            ModelSource::huggingface("test"),
        );

        assert!((entry.size_mb() - 1024.0).abs() < 0.01);
        assert!((entry.size_gb() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_model_entry_with_format() {
        let entry = ModelEntry::new("test", "1.0", "", 0, ModelSource::huggingface("test"))
            .with_format("gguf");

        assert_eq!(entry.format, Some("gguf".to_string()));
    }

    #[test]
    fn test_model_entry_with_metadata() {
        let entry = ModelEntry::new("test", "1.0", "", 0, ModelSource::huggingface("test"))
            .with_metadata("architecture", "llama")
            .with_metadata("quantization", "q4_0");

        assert_eq!(
            entry.metadata.get("architecture"),
            Some(&"llama".to_string())
        );
        assert_eq!(
            entry.metadata.get("quantization"),
            Some(&"q4_0".to_string())
        );
    }

    #[test]
    fn test_registry_manifest_new() {
        let manifest = RegistryManifest::new();

        assert!(manifest.models.is_empty());
        assert!(manifest.last_sync.is_none());
        assert_eq!(manifest.version, "1.0");
    }

    #[test]
    fn test_registry_manifest_add_and_find() {
        let mut manifest = RegistryManifest::new();

        let entry = ModelEntry::new("test", "1.0", "", 100, ModelSource::huggingface("test"));
        manifest.add(entry);

        assert_eq!(manifest.len(), 1);
        assert!(manifest.find("test").is_some());
        assert!(manifest.find("nonexistent").is_none());
    }

    #[test]
    fn test_registry_manifest_update_existing() {
        let mut manifest = RegistryManifest::new();

        let entry1 = ModelEntry::new("test", "1.0", "", 100, ModelSource::huggingface("test"));
        manifest.add(entry1);

        let entry2 = ModelEntry::new("test", "2.0", "", 200, ModelSource::huggingface("test"));
        manifest.add(entry2);

        assert_eq!(manifest.len(), 1);
        assert_eq!(manifest.find("test").unwrap().version, "2.0");
    }

    #[test]
    fn test_registry_manifest_total_size() {
        let mut manifest = RegistryManifest::new();

        manifest.add(ModelEntry::new(
            "a",
            "1",
            "",
            100,
            ModelSource::huggingface("a"),
        ));
        manifest.add(ModelEntry::new(
            "b",
            "1",
            "",
            200,
            ModelSource::huggingface("b"),
        ));

        assert_eq!(manifest.total_size_bytes(), 300);
    }

    #[test]
    fn test_offline_registry_new() {
        let temp = TempDir::new().unwrap();
        let registry = OfflineModelRegistry::new(temp.path().to_path_buf());

        assert_eq!(registry.root(), temp.path());
        assert!(registry.manifest.is_empty());
    }

    #[test]
    fn test_offline_registry_add_model() {
        let temp = TempDir::new().unwrap();
        let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());

        let entry = ModelEntry::new("test", "1.0", "", 100, ModelSource::huggingface("test"));
        registry.add_model(entry);

        assert_eq!(registry.manifest.len(), 1);
        assert!(registry.get("test").is_some());
    }

    #[test]
    fn test_offline_registry_mirror_from_hub() {
        let temp = TempDir::new().unwrap();
        let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());

        let entry = registry.mirror_from_hub("bert-base-uncased").unwrap();

        assert_eq!(entry.name, "bert-base-uncased");
        assert!(matches!(entry.source, ModelSource::HuggingFace { .. }));
    }

    #[test]
    fn test_offline_registry_register_local() {
        let temp = TempDir::new().unwrap();
        let model_file = temp.path().join("test.gguf");
        fs::write(&model_file, "test model content").unwrap();

        let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());
        let entry = registry.register_local("test-model", &model_file).unwrap();

        assert_eq!(entry.name, "test-model");
        assert!(entry.is_local());
        assert!(!entry.sha256.is_empty());
        assert_eq!(entry.format, Some("gguf".to_string()));
    }

    #[test]
    fn test_offline_registry_load() {
        let temp = TempDir::new().unwrap();
        let model_file = temp.path().join("test.gguf");
        fs::write(&model_file, "test content").unwrap();

        let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());
        registry.register_local("test", &model_file).unwrap();

        let loaded = registry.load("test").unwrap();
        assert_eq!(loaded, model_file);
    }

    #[test]
    fn test_offline_registry_load_not_found() {
        let temp = TempDir::new().unwrap();
        let registry = OfflineModelRegistry::new(temp.path().to_path_buf());

        let result = registry.load("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_offline_registry_verify() {
        let temp = TempDir::new().unwrap();
        let model_file = temp.path().join("test.bin");
        fs::write(&model_file, "test content").unwrap();

        let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());
        let entry = registry.register_local("test", &model_file).unwrap();

        assert!(registry.verify(&entry).unwrap());

        // Modify file
        fs::write(&model_file, "modified content").unwrap();
        assert!(!registry.verify(&entry).unwrap());
    }

    #[test]
    fn test_offline_registry_list_available() {
        let temp = TempDir::new().unwrap();
        let model_file = temp.path().join("local.bin");
        fs::write(&model_file, "content").unwrap();

        let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());

        // Add local model
        registry.register_local("local", &model_file).unwrap();

        // Add remote model (not available locally)
        registry.add_model(ModelEntry::new(
            "remote",
            "1.0",
            "",
            100,
            ModelSource::huggingface("remote"),
        ));

        let available = registry.list_available();
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].name, "local");
    }

    #[test]
    fn test_offline_registry_remove() {
        let temp = TempDir::new().unwrap();
        let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());

        registry.add_model(ModelEntry::new(
            "test",
            "1.0",
            "",
            100,
            ModelSource::huggingface("test"),
        ));
        assert_eq!(registry.manifest.len(), 1);

        let removed = registry.remove("test");
        assert!(removed.is_some());
        assert_eq!(registry.manifest.len(), 0);
    }

    #[test]
    fn test_offline_registry_save_and_load() {
        let temp = TempDir::new().unwrap();

        {
            let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());
            registry.add_model(ModelEntry::new(
                "test",
                "1.0",
                "abc",
                100,
                ModelSource::huggingface("test"),
            ));
            registry.save_manifest().unwrap();
        }

        // Load in new instance
        let registry = OfflineModelRegistry::new(temp.path().to_path_buf());
        assert_eq!(registry.manifest.len(), 1);
        assert!(registry.get("test").is_some());
    }

    #[test]
    fn test_model_entry_serialization() {
        let entry = ModelEntry::new(
            "test",
            "1.0",
            "abc123",
            1000,
            ModelSource::huggingface("test/model"),
        )
        .with_format("gguf")
        .with_metadata("arch", "llama");

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: ModelEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(entry.name, parsed.name);
        assert_eq!(entry.format, parsed.format);
        assert_eq!(entry.metadata, parsed.metadata);
    }
}
