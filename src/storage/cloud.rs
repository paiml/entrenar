//! Cloud Storage Backends (#72)
//!
//! Artifact storage abstraction supporting local, S3, Azure, and GCS backends.
//!
//! # Toyota Principle: Heijunka (平準化)
//!
//! Level workloads across storage tiers - artifacts are stored content-addressably
//! to enable efficient caching and deduplication across storage backends.
//!
//! # Example
//!
//! ```
//! use entrenar::storage::cloud::{ArtifactBackend, LocalBackend};
//! use std::path::PathBuf;
//!
//! let backend = LocalBackend::new(PathBuf::from("/tmp/artifacts"));
//! let hash = backend.put("model.safetensors", b"test data").unwrap();
//! let data = backend.get(&hash).unwrap();
//! ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Cloud storage errors
#[derive(Debug, Error)]
pub enum CloudError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Artifact not found: {0}")]
    NotFound(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Network error: {0}")]
    Network(String),
}

/// Result type for cloud operations
pub type Result<T> = std::result::Result<T, CloudError>;

/// Artifact metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    /// Original filename
    pub name: String,
    /// Content-addressable hash (SHA-256)
    pub hash: String,
    /// Size in bytes
    pub size: u64,
    /// Content type (MIME)
    pub content_type: Option<String>,
    /// Creation timestamp (Unix seconds)
    pub created_at: u64,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl ArtifactMetadata {
    /// Create new artifact metadata
    pub fn new(name: &str, hash: &str, size: u64) -> Self {
        Self {
            name: name.to_string(),
            hash: hash.to_string(),
            size,
            content_type: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            metadata: HashMap::new(),
        }
    }

    /// Set content type
    pub fn with_content_type(mut self, content_type: &str) -> Self {
        self.content_type = Some(content_type.to_string());
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Trait for artifact storage backends
pub trait ArtifactBackend: Send + Sync {
    /// Store an artifact and return its content-addressable hash
    fn put(&self, name: &str, data: &[u8]) -> Result<String>;

    /// Retrieve an artifact by hash
    fn get(&self, hash: &str) -> Result<Vec<u8>>;

    /// Check if an artifact exists
    fn exists(&self, hash: &str) -> Result<bool>;

    /// Delete an artifact by hash
    fn delete(&self, hash: &str) -> Result<()>;

    /// Get artifact metadata
    fn get_metadata(&self, hash: &str) -> Result<ArtifactMetadata>;

    /// List all artifacts
    fn list(&self) -> Result<Vec<ArtifactMetadata>>;

    /// Get backend type name
    fn backend_type(&self) -> &'static str;
}

/// Compute SHA-256 hash of data
pub fn compute_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex::encode(result)
}

// =============================================================================
// Local Filesystem Backend
// =============================================================================

/// Local filesystem artifact backend
#[derive(Debug)]
pub struct LocalBackend {
    base_path: PathBuf,
    metadata: Arc<RwLock<HashMap<String, ArtifactMetadata>>>,
}

impl LocalBackend {
    /// Create a new local backend
    pub fn new(base_path: PathBuf) -> Self {
        Self {
            base_path,
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new local backend and ensure directory exists
    pub fn new_and_init(base_path: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;
        Ok(Self::new(base_path))
    }

    /// Get the file path for a hash
    fn hash_to_path(&self, hash: &str) -> PathBuf {
        // Use subdirectories based on hash prefix for better filesystem performance
        let prefix = &hash[..2];
        self.base_path.join(prefix).join(hash)
    }
}

impl ArtifactBackend for LocalBackend {
    fn put(&self, name: &str, data: &[u8]) -> Result<String> {
        let hash = compute_hash(data);
        let path = self.hash_to_path(&hash);

        // Create parent directory
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write data
        let mut file = std::fs::File::create(&path)?;
        file.write_all(data)?;

        // Store metadata
        let metadata = ArtifactMetadata::new(name, &hash, data.len() as u64);
        self.metadata
            .write()
            .unwrap()
            .insert(hash.clone(), metadata);

        Ok(hash)
    }

    fn get(&self, hash: &str) -> Result<Vec<u8>> {
        let path = self.hash_to_path(hash);

        if !path.exists() {
            return Err(CloudError::NotFound(hash.to_string()));
        }

        let mut file = std::fs::File::open(&path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        // Verify hash
        let computed = compute_hash(&data);
        if computed != hash {
            return Err(CloudError::Backend(format!(
                "Hash mismatch: expected {hash}, got {computed}"
            )));
        }

        Ok(data)
    }

    fn exists(&self, hash: &str) -> Result<bool> {
        let path = self.hash_to_path(hash);
        Ok(path.exists())
    }

    fn delete(&self, hash: &str) -> Result<()> {
        let path = self.hash_to_path(hash);

        if !path.exists() {
            return Err(CloudError::NotFound(hash.to_string()));
        }

        std::fs::remove_file(&path)?;
        self.metadata.write().unwrap().remove(hash);

        Ok(())
    }

    fn get_metadata(&self, hash: &str) -> Result<ArtifactMetadata> {
        self.metadata
            .read()
            .unwrap()
            .get(hash)
            .cloned()
            .ok_or_else(|| CloudError::NotFound(hash.to_string()))
    }

    fn list(&self) -> Result<Vec<ArtifactMetadata>> {
        Ok(self.metadata.read().unwrap().values().cloned().collect())
    }

    fn backend_type(&self) -> &'static str {
        "local"
    }
}

// =============================================================================
// In-Memory Backend (for testing)
// =============================================================================

/// In-memory artifact backend for testing
#[derive(Debug, Default)]
pub struct InMemoryBackend {
    data: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    metadata: Arc<RwLock<HashMap<String, ArtifactMetadata>>>,
}

impl InMemoryBackend {
    /// Create a new in-memory backend
    pub fn new() -> Self {
        Self::default()
    }
}

impl ArtifactBackend for InMemoryBackend {
    fn put(&self, name: &str, data: &[u8]) -> Result<String> {
        let hash = compute_hash(data);

        self.data
            .write()
            .unwrap()
            .insert(hash.clone(), data.to_vec());

        let metadata = ArtifactMetadata::new(name, &hash, data.len() as u64);
        self.metadata
            .write()
            .unwrap()
            .insert(hash.clone(), metadata);

        Ok(hash)
    }

    fn get(&self, hash: &str) -> Result<Vec<u8>> {
        self.data
            .read()
            .unwrap()
            .get(hash)
            .cloned()
            .ok_or_else(|| CloudError::NotFound(hash.to_string()))
    }

    fn exists(&self, hash: &str) -> Result<bool> {
        Ok(self.data.read().unwrap().contains_key(hash))
    }

    fn delete(&self, hash: &str) -> Result<()> {
        let removed = self.data.write().unwrap().remove(hash);
        if removed.is_none() {
            return Err(CloudError::NotFound(hash.to_string()));
        }
        self.metadata.write().unwrap().remove(hash);
        Ok(())
    }

    fn get_metadata(&self, hash: &str) -> Result<ArtifactMetadata> {
        self.metadata
            .read()
            .unwrap()
            .get(hash)
            .cloned()
            .ok_or_else(|| CloudError::NotFound(hash.to_string()))
    }

    fn list(&self) -> Result<Vec<ArtifactMetadata>> {
        Ok(self.metadata.read().unwrap().values().cloned().collect())
    }

    fn backend_type(&self) -> &'static str {
        "memory"
    }
}

// =============================================================================
// S3 Configuration (for future implementation)
// =============================================================================

/// S3 backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3Config {
    /// S3 bucket name
    pub bucket: String,
    /// Key prefix within bucket
    pub prefix: String,
    /// AWS region (e.g., "us-east-1")
    pub region: Option<String>,
    /// Custom endpoint (for MinIO, R2, etc.)
    pub endpoint: Option<String>,
    /// Access key ID (if not using IAM role)
    pub access_key_id: Option<String>,
    /// Secret access key (if not using IAM role)
    pub secret_access_key: Option<String>,
}

impl S3Config {
    /// Create a new S3 configuration
    pub fn new(bucket: &str, prefix: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            region: None,
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
        }
    }

    /// Set region
    pub fn with_region(mut self, region: &str) -> Self {
        self.region = Some(region.to_string());
        self
    }

    /// Set custom endpoint (for S3-compatible services)
    pub fn with_endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = Some(endpoint.to_string());
        self
    }

    /// Set credentials
    pub fn with_credentials(mut self, access_key_id: &str, secret_access_key: &str) -> Self {
        self.access_key_id = Some(access_key_id.to_string());
        self.secret_access_key = Some(secret_access_key.to_string());
        self
    }

    /// Get the full key path for a hash
    pub fn key_for_hash(&self, hash: &str) -> String {
        if self.prefix.is_empty() {
            hash.to_string()
        } else {
            format!("{}/{}", self.prefix.trim_end_matches('/'), hash)
        }
    }
}

/// Mock S3 backend for testing (simulates S3 behavior in memory)
#[derive(Debug)]
pub struct MockS3Backend {
    config: S3Config,
    inner: InMemoryBackend,
}

impl MockS3Backend {
    /// Create a new mock S3 backend
    pub fn new(config: S3Config) -> Self {
        Self {
            config,
            inner: InMemoryBackend::new(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &S3Config {
        &self.config
    }
}

impl ArtifactBackend for MockS3Backend {
    fn put(&self, name: &str, data: &[u8]) -> Result<String> {
        self.inner.put(name, data)
    }

    fn get(&self, hash: &str) -> Result<Vec<u8>> {
        self.inner.get(hash)
    }

    fn exists(&self, hash: &str) -> Result<bool> {
        self.inner.exists(hash)
    }

    fn delete(&self, hash: &str) -> Result<()> {
        self.inner.delete(hash)
    }

    fn get_metadata(&self, hash: &str) -> Result<ArtifactMetadata> {
        self.inner.get_metadata(hash)
    }

    fn list(&self) -> Result<Vec<ArtifactMetadata>> {
        self.inner.list()
    }

    fn backend_type(&self) -> &'static str {
        "s3"
    }
}

// =============================================================================
// Azure Configuration (for future implementation)
// =============================================================================

/// Azure Blob Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureConfig {
    /// Storage account name
    pub account: String,
    /// Container name
    pub container: String,
    /// Blob prefix
    pub prefix: String,
    /// Connection string (if not using managed identity)
    pub connection_string: Option<String>,
}

impl AzureConfig {
    /// Create a new Azure configuration
    pub fn new(account: &str, container: &str) -> Self {
        Self {
            account: account.to_string(),
            container: container.to_string(),
            prefix: String::new(),
            connection_string: None,
        }
    }

    /// Set blob prefix
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    /// Set connection string
    pub fn with_connection_string(mut self, conn_str: &str) -> Self {
        self.connection_string = Some(conn_str.to_string());
        self
    }
}

// =============================================================================
// GCS Configuration (for future implementation)
// =============================================================================

/// Google Cloud Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCSConfig {
    /// GCS bucket name
    pub bucket: String,
    /// Object prefix
    pub prefix: String,
    /// Project ID
    pub project_id: Option<String>,
    /// Service account JSON key path
    pub service_account_key: Option<String>,
}

impl GCSConfig {
    /// Create a new GCS configuration
    pub fn new(bucket: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            prefix: String::new(),
            project_id: None,
            service_account_key: None,
        }
    }

    /// Set object prefix
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    /// Set project ID
    pub fn with_project(mut self, project_id: &str) -> Self {
        self.project_id = Some(project_id.to_string());
        self
    }
}

// =============================================================================
// Unified Backend Enum
// =============================================================================

/// Unified artifact backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendConfig {
    /// Local filesystem
    Local { path: PathBuf },
    /// In-memory (for testing)
    Memory,
    /// Amazon S3
    S3(S3Config),
    /// Azure Blob Storage
    Azure(AzureConfig),
    /// Google Cloud Storage
    GCS(GCSConfig),
}

impl BackendConfig {
    /// Create a local backend configuration
    pub fn local(path: PathBuf) -> Self {
        Self::Local { path }
    }

    /// Create an in-memory backend configuration
    pub fn memory() -> Self {
        Self::Memory
    }

    /// Create an S3 backend configuration
    pub fn s3(bucket: &str, prefix: &str) -> Self {
        Self::S3(S3Config::new(bucket, prefix))
    }

    /// Create a backend from this configuration
    pub fn build(&self) -> Result<Box<dyn ArtifactBackend>> {
        match self {
            Self::Local { path } => Ok(Box::new(LocalBackend::new_and_init(path.clone())?)),
            Self::Memory => Ok(Box::new(InMemoryBackend::new())),
            Self::S3(config) => Ok(Box::new(MockS3Backend::new(config.clone()))),
            Self::Azure(_config) => {
                // For now, return a mock (real implementation would use azure SDK)
                Ok(Box::new(InMemoryBackend::new()))
            }
            Self::GCS(_config) => {
                // For now, return a mock (real implementation would use GCS SDK)
                Ok(Box::new(InMemoryBackend::new()))
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_compute_hash() {
        let data = b"hello world";
        let hash = compute_hash(data);
        assert_eq!(hash.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let data = b"test data";
        let hash1 = compute_hash(data);
        let hash2 = compute_hash(data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_artifact_metadata_new() {
        let meta = ArtifactMetadata::new("test.bin", "abc123", 1024);
        assert_eq!(meta.name, "test.bin");
        assert_eq!(meta.hash, "abc123");
        assert_eq!(meta.size, 1024);
    }

    #[test]
    fn test_artifact_metadata_with_content_type() {
        let meta = ArtifactMetadata::new("model.safetensors", "abc", 100)
            .with_content_type("application/octet-stream");
        assert_eq!(
            meta.content_type,
            Some("application/octet-stream".to_string())
        );
    }

    // -------------------------------------------------------------------------
    // In-Memory Backend Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_in_memory_backend_put_get() {
        let backend = InMemoryBackend::new();
        let data = b"test data";
        let hash = backend.put("test.bin", data).unwrap();

        let retrieved = backend.get(&hash).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_in_memory_backend_exists() {
        let backend = InMemoryBackend::new();
        let hash = backend.put("test.bin", b"data").unwrap();

        assert!(backend.exists(&hash).unwrap());
        assert!(!backend.exists("nonexistent").unwrap());
    }

    #[test]
    fn test_in_memory_backend_delete() {
        let backend = InMemoryBackend::new();
        let hash = backend.put("test.bin", b"data").unwrap();

        backend.delete(&hash).unwrap();
        assert!(!backend.exists(&hash).unwrap());
    }

    #[test]
    fn test_in_memory_backend_delete_not_found() {
        let backend = InMemoryBackend::new();
        let result = backend.delete("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_in_memory_backend_get_metadata() {
        let backend = InMemoryBackend::new();
        let data = b"test data 123";
        let hash = backend.put("model.bin", data).unwrap();

        let meta = backend.get_metadata(&hash).unwrap();
        assert_eq!(meta.name, "model.bin");
        assert_eq!(meta.size, data.len() as u64);
    }

    #[test]
    fn test_in_memory_backend_list() {
        let backend = InMemoryBackend::new();
        backend.put("file1.bin", b"data1").unwrap();
        backend.put("file2.bin", b"data2").unwrap();

        let list = backend.list().unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_in_memory_backend_type() {
        let backend = InMemoryBackend::new();
        assert_eq!(backend.backend_type(), "memory");
    }

    // -------------------------------------------------------------------------
    // Local Backend Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_local_backend_put_get() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalBackend::new_and_init(tmp.path().to_path_buf()).unwrap();

        let data = b"local test data";
        let hash = backend.put("test.bin", data).unwrap();

        let retrieved = backend.get(&hash).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_local_backend_exists() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalBackend::new_and_init(tmp.path().to_path_buf()).unwrap();

        let hash = backend.put("test.bin", b"data").unwrap();
        assert!(backend.exists(&hash).unwrap());
        assert!(!backend.exists("nonexistent").unwrap());
    }

    #[test]
    fn test_local_backend_delete() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalBackend::new_and_init(tmp.path().to_path_buf()).unwrap();

        let hash = backend.put("test.bin", b"data").unwrap();
        backend.delete(&hash).unwrap();
        assert!(!backend.exists(&hash).unwrap());
    }

    #[test]
    fn test_local_backend_type() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalBackend::new_and_init(tmp.path().to_path_buf()).unwrap();
        assert_eq!(backend.backend_type(), "local");
    }

    // -------------------------------------------------------------------------
    // S3 Config Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_s3_config_new() {
        let config = S3Config::new("my-bucket", "artifacts/");
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.prefix, "artifacts/");
    }

    #[test]
    fn test_s3_config_with_region() {
        let config = S3Config::new("bucket", "prefix").with_region("us-west-2");
        assert_eq!(config.region, Some("us-west-2".to_string()));
    }

    #[test]
    fn test_s3_config_with_endpoint() {
        let config = S3Config::new("bucket", "prefix").with_endpoint("http://minio:9000");
        assert_eq!(config.endpoint, Some("http://minio:9000".to_string()));
    }

    #[test]
    fn test_s3_config_key_for_hash() {
        let config = S3Config::new("bucket", "artifacts");
        assert_eq!(config.key_for_hash("abc123"), "artifacts/abc123");

        let config = S3Config::new("bucket", "");
        assert_eq!(config.key_for_hash("abc123"), "abc123");
    }

    // -------------------------------------------------------------------------
    // Mock S3 Backend Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mock_s3_backend_put_get() {
        let config = S3Config::new("test-bucket", "prefix");
        let backend = MockS3Backend::new(config);

        let data = b"s3 test data";
        let hash = backend.put("file.bin", data).unwrap();

        let retrieved = backend.get(&hash).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_mock_s3_backend_type() {
        let config = S3Config::new("bucket", "prefix");
        let backend = MockS3Backend::new(config);
        assert_eq!(backend.backend_type(), "s3");
    }

    // -------------------------------------------------------------------------
    // Azure Config Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_azure_config_new() {
        let config = AzureConfig::new("myaccount", "mycontainer");
        assert_eq!(config.account, "myaccount");
        assert_eq!(config.container, "mycontainer");
    }

    #[test]
    fn test_azure_config_with_prefix() {
        let config = AzureConfig::new("account", "container").with_prefix("models/");
        assert_eq!(config.prefix, "models/");
    }

    // -------------------------------------------------------------------------
    // GCS Config Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_gcs_config_new() {
        let config = GCSConfig::new("my-bucket");
        assert_eq!(config.bucket, "my-bucket");
    }

    #[test]
    fn test_gcs_config_with_project() {
        let config = GCSConfig::new("bucket").with_project("my-project");
        assert_eq!(config.project_id, Some("my-project".to_string()));
    }

    // -------------------------------------------------------------------------
    // Backend Config Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_backend_config_local() {
        let config = BackendConfig::local(PathBuf::from("/tmp/artifacts"));
        match config {
            BackendConfig::Local { path } => {
                assert_eq!(path, PathBuf::from("/tmp/artifacts"));
            }
            _ => panic!("Expected Local"),
        }
    }

    #[test]
    fn test_backend_config_memory() {
        let config = BackendConfig::memory();
        assert!(matches!(config, BackendConfig::Memory));
    }

    #[test]
    fn test_backend_config_s3() {
        let config = BackendConfig::s3("bucket", "prefix");
        match config {
            BackendConfig::S3(c) => {
                assert_eq!(c.bucket, "bucket");
                assert_eq!(c.prefix, "prefix");
            }
            _ => panic!("Expected S3"),
        }
    }

    #[test]
    fn test_backend_config_build_memory() {
        let config = BackendConfig::memory();
        let backend = config.build().unwrap();
        assert_eq!(backend.backend_type(), "memory");
    }

    #[test]
    fn test_backend_config_build_s3() {
        let config = BackendConfig::s3("bucket", "prefix");
        let backend = config.build().unwrap();
        assert_eq!(backend.backend_type(), "s3");
    }

    #[test]
    fn test_artifact_metadata_with_metadata() {
        let meta = ArtifactMetadata::new("model.bin", "abc123", 1024)
            .with_metadata("model_type", "bert")
            .with_metadata("version", "1.0");
        assert_eq!(meta.metadata.get("model_type"), Some(&"bert".to_string()));
        assert_eq!(meta.metadata.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_cloud_error_display() {
        let io_err = CloudError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(io_err.to_string().contains("IO error"));

        let not_found = CloudError::NotFound("abc123".to_string());
        assert!(not_found.to_string().contains("abc123"));

        let backend = CloudError::Backend("connection failed".to_string());
        assert!(backend.to_string().contains("connection failed"));

        let config = CloudError::Config("invalid config".to_string());
        assert!(config.to_string().contains("invalid config"));

        let permission = CloudError::PermissionDenied("access denied".to_string());
        assert!(permission.to_string().contains("access denied"));

        let network = CloudError::Network("timeout".to_string());
        assert!(network.to_string().contains("timeout"));
    }

    #[test]
    fn test_local_backend_get_not_found() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalBackend::new_and_init(tmp.path().to_path_buf()).unwrap();

        let result = backend.get("nonexistent_hash");
        assert!(result.is_err());
        match result {
            Err(CloudError::NotFound(hash)) => assert_eq!(hash, "nonexistent_hash"),
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn test_local_backend_delete_not_found() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalBackend::new_and_init(tmp.path().to_path_buf()).unwrap();

        let result = backend.delete("nonexistent_hash");
        assert!(result.is_err());
    }

    #[test]
    fn test_local_backend_get_metadata() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalBackend::new_and_init(tmp.path().to_path_buf()).unwrap();

        let data = b"test data for metadata";
        let hash = backend.put("test_file.bin", data).unwrap();

        let meta = backend.get_metadata(&hash).unwrap();
        assert_eq!(meta.name, "test_file.bin");
        assert_eq!(meta.size, data.len() as u64);
    }

    #[test]
    fn test_local_backend_get_metadata_not_found() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalBackend::new_and_init(tmp.path().to_path_buf()).unwrap();

        let result = backend.get_metadata("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_local_backend_list() {
        let tmp = TempDir::new().unwrap();
        let backend = LocalBackend::new_and_init(tmp.path().to_path_buf()).unwrap();

        backend.put("file1.bin", b"data1").unwrap();
        backend.put("file2.bin", b"data2").unwrap();
        backend.put("file3.bin", b"data3").unwrap();

        let list = backend.list().unwrap();
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_s3_config_with_credentials() {
        let config = S3Config::new("bucket", "prefix").with_credentials("access_key", "secret_key");
        assert_eq!(config.access_key_id, Some("access_key".to_string()));
        assert_eq!(config.secret_access_key, Some("secret_key".to_string()));
    }

    #[test]
    fn test_s3_config_key_for_hash_with_trailing_slash() {
        let config = S3Config::new("bucket", "artifacts/");
        assert_eq!(config.key_for_hash("abc123"), "artifacts/abc123");
    }

    #[test]
    fn test_azure_config_with_connection_string() {
        let config = AzureConfig::new("account", "container")
            .with_connection_string("DefaultEndpointsProtocol=https;...");
        assert!(config.connection_string.is_some());
    }

    #[test]
    fn test_gcs_config_with_prefix() {
        let config = GCSConfig::new("bucket").with_prefix("models/v1/");
        assert_eq!(config.prefix, "models/v1/");
    }

    #[test]
    fn test_backend_config_build_local() {
        let tmp = TempDir::new().unwrap();
        let config = BackendConfig::local(tmp.path().to_path_buf());
        let backend = config.build().unwrap();
        assert_eq!(backend.backend_type(), "local");
    }

    #[test]
    fn test_backend_config_build_azure() {
        let config = BackendConfig::Azure(AzureConfig::new("account", "container"));
        let backend = config.build().unwrap();
        // Currently returns InMemoryBackend as placeholder
        assert_eq!(backend.backend_type(), "memory");
    }

    #[test]
    fn test_backend_config_build_gcs() {
        let config = BackendConfig::GCS(GCSConfig::new("bucket"));
        let backend = config.build().unwrap();
        // Currently returns InMemoryBackend as placeholder
        assert_eq!(backend.backend_type(), "memory");
    }

    #[test]
    fn test_mock_s3_backend_exists() {
        let config = S3Config::new("bucket", "prefix");
        let backend = MockS3Backend::new(config);

        let hash = backend.put("file.bin", b"data").unwrap();
        assert!(backend.exists(&hash).unwrap());
        assert!(!backend.exists("nonexistent").unwrap());
    }

    #[test]
    fn test_mock_s3_backend_delete() {
        let config = S3Config::new("bucket", "prefix");
        let backend = MockS3Backend::new(config);

        let hash = backend.put("file.bin", b"data").unwrap();
        backend.delete(&hash).unwrap();
        assert!(!backend.exists(&hash).unwrap());
    }

    #[test]
    fn test_mock_s3_backend_get_metadata() {
        let config = S3Config::new("bucket", "prefix");
        let backend = MockS3Backend::new(config);

        let hash = backend.put("model.bin", b"model data").unwrap();
        let meta = backend.get_metadata(&hash).unwrap();
        assert_eq!(meta.name, "model.bin");
    }

    #[test]
    fn test_mock_s3_backend_list() {
        let config = S3Config::new("bucket", "prefix");
        let backend = MockS3Backend::new(config);

        backend.put("file1.bin", b"data1").unwrap();
        backend.put("file2.bin", b"data2").unwrap();

        let list = backend.list().unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_mock_s3_backend_config() {
        let config = S3Config::new("test-bucket", "models/").with_region("us-east-1");
        let backend = MockS3Backend::new(config);

        assert_eq!(backend.config().bucket, "test-bucket");
        assert_eq!(backend.config().prefix, "models/");
        assert_eq!(backend.config().region, Some("us-east-1".to_string()));
    }

    #[test]
    fn test_in_memory_backend_get_not_found() {
        let backend = InMemoryBackend::new();
        let result = backend.get("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_in_memory_backend_get_metadata_not_found() {
        let backend = InMemoryBackend::new();
        let result = backend.get_metadata("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_artifact_metadata_serde() {
        let meta = ArtifactMetadata::new("test.bin", "abc123", 1024)
            .with_content_type("application/octet-stream")
            .with_metadata("key", "value");

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: ArtifactMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(meta.name, parsed.name);
        assert_eq!(meta.hash, parsed.hash);
        assert_eq!(meta.size, parsed.size);
        assert_eq!(meta.content_type, parsed.content_type);
    }

    #[test]
    fn test_s3_config_serde() {
        let config = S3Config::new("bucket", "prefix")
            .with_region("us-west-2")
            .with_endpoint("http://localhost:9000")
            .with_credentials("key", "secret");

        let json = serde_json::to_string(&config).unwrap();
        let parsed: S3Config = serde_json::from_str(&json).unwrap();

        assert_eq!(config.bucket, parsed.bucket);
        assert_eq!(config.region, parsed.region);
        assert_eq!(config.endpoint, parsed.endpoint);
    }

    #[test]
    fn test_azure_config_serde() {
        let config = AzureConfig::new("account", "container")
            .with_prefix("models/")
            .with_connection_string("conn");

        let json = serde_json::to_string(&config).unwrap();
        let parsed: AzureConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.account, parsed.account);
        assert_eq!(config.container, parsed.container);
        assert_eq!(config.prefix, parsed.prefix);
    }

    #[test]
    fn test_gcs_config_serde() {
        let config = GCSConfig::new("bucket")
            .with_prefix("artifacts/")
            .with_project("my-project");

        let json = serde_json::to_string(&config).unwrap();
        let parsed: GCSConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.bucket, parsed.bucket);
        assert_eq!(config.prefix, parsed.prefix);
        assert_eq!(config.project_id, parsed.project_id);
    }

    #[test]
    fn test_backend_config_serde() {
        let configs = vec![
            BackendConfig::local(PathBuf::from("/tmp/test")),
            BackendConfig::memory(),
            BackendConfig::s3("bucket", "prefix"),
            BackendConfig::Azure(AzureConfig::new("account", "container")),
            BackendConfig::GCS(GCSConfig::new("bucket")),
        ];

        for config in configs {
            let json = serde_json::to_string(&config).unwrap();
            let parsed: BackendConfig = serde_json::from_str(&json).unwrap();
            // Just verify it parses without panic
            let _ = parsed;
        }
    }

    #[test]
    fn test_local_backend_new() {
        let backend = LocalBackend::new(PathBuf::from("/tmp/test"));
        assert_eq!(backend.backend_type(), "local");
    }

    #[test]
    fn test_in_memory_backend_default() {
        let backend = InMemoryBackend::default();
        assert_eq!(backend.backend_type(), "memory");
    }

    #[test]
    fn test_artifact_metadata_created_at_is_set() {
        let meta = ArtifactMetadata::new("test.bin", "hash", 100);
        assert!(meta.created_at > 0);
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_hash_deterministic(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            let hash1 = compute_hash(&data);
            let hash2 = compute_hash(&data);
            prop_assert_eq!(hash1, hash2);
        }

        #[test]
        fn prop_hash_length_constant(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            let hash = compute_hash(&data);
            prop_assert_eq!(hash.len(), 64);
        }

        #[test]
        fn prop_memory_backend_roundtrip(
            name in "[a-zA-Z0-9_]{1,50}",
            data in prop::collection::vec(any::<u8>(), 1..1000)
        ) {
            let backend = InMemoryBackend::new();
            let hash = backend.put(&name, &data).unwrap();
            let retrieved = backend.get(&hash).unwrap();
            prop_assert_eq!(retrieved, data);
        }

        #[test]
        fn prop_s3_key_contains_hash(
            prefix in "[a-zA-Z0-9/]{0,20}",
            hash in "[a-f0-9]{64}"
        ) {
            let config = S3Config::new("bucket", &prefix);
            let key = config.key_for_hash(&hash);
            prop_assert!(key.contains(&hash));
        }

        #[test]
        fn prop_metadata_size_matches(
            name in "[a-zA-Z0-9_]{1,20}",
            size in 0u64..1_000_000
        ) {
            let meta = ArtifactMetadata::new(&name, "hash", size);
            prop_assert_eq!(meta.size, size);
        }
    }
}
