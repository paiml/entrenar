//! Offline model registry implementation.

use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

use super::manifest::RegistryManifest;
use super::types::{ModelEntry, ModelSource};

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
        serde_json::from_str(&content).map_err(|e| Error::Io(format!("Invalid manifest data: {e}")))
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
