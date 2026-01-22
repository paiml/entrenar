//! HuggingFace model fetcher implementation.
//!
//! Downloads models from HuggingFace Hub with authentication and caching.

use crate::hf_pipeline::error::{FetchError, Result};
use std::path::PathBuf;

use super::options::FetchOptions;
use super::types::{ModelArtifact, WeightFormat};

/// HuggingFace model fetcher
pub struct HfModelFetcher {
    /// Authentication token
    pub(crate) token: Option<String>,
    /// Cache directory
    pub(crate) cache_dir: PathBuf,
    /// API base URL (for future HTTP client integration)
    #[allow(dead_code)]
    pub(crate) api_base: String,
}

impl HfModelFetcher {
    /// Create new fetcher using HF_TOKEN environment variable
    ///
    /// # Errors
    ///
    /// Does not error on missing token (allows anonymous pulls).
    pub fn new() -> Result<Self> {
        let token = Self::resolve_token();
        let cache_dir = Self::default_cache_dir();

        Ok(Self {
            token,
            cache_dir,
            api_base: "https://huggingface.co".into(),
        })
    }

    /// Create fetcher with explicit token
    #[must_use]
    pub fn with_token(token: impl Into<String>) -> Self {
        Self {
            token: Some(token.into()),
            cache_dir: Self::default_cache_dir(),
            api_base: "https://huggingface.co".into(),
        }
    }

    /// Set cache directory
    #[must_use]
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = dir.into();
        self
    }

    /// Resolve token from multiple sources
    ///
    /// Priority:
    /// 1. HF_TOKEN environment variable
    /// 2. ~/.huggingface/token file
    #[must_use]
    pub fn resolve_token() -> Option<String> {
        // Try environment variable first
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if !token.is_empty() {
                return Some(token);
            }
        }

        // Try ~/.huggingface/token file
        if let Some(home) = dirs::home_dir() {
            let token_path = home.join(".huggingface").join("token");
            if let Ok(token) = std::fs::read_to_string(token_path) {
                let token = token.trim().to_string();
                if !token.is_empty() {
                    return Some(token);
                }
            }
        }

        None
    }

    /// Get default cache directory
    pub(crate) fn default_cache_dir() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("huggingface")
            .join("hub")
    }

    /// Check if client has authentication
    #[must_use]
    pub fn is_authenticated(&self) -> bool {
        self.token.is_some()
    }

    /// Parse and validate repository ID
    pub(crate) fn parse_repo_id(repo_id: &str) -> Result<(&str, &str)> {
        let parts: Vec<&str> = repo_id.split('/').collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
            return Err(FetchError::InvalidRepoId {
                repo_id: repo_id.to_string(),
            });
        }
        Ok((parts[0], parts[1]))
    }

    /// Download a model from HuggingFace Hub
    ///
    /// # Arguments
    ///
    /// * `repo_id` - Repository ID in "org/name" format
    /// * `options` - Fetch options
    ///
    /// # Errors
    ///
    /// Returns error if download fails, repo not found, or security check fails.
    pub fn download_model(&self, repo_id: &str, options: FetchOptions) -> Result<ModelArtifact> {
        let (_org, _name) = Self::parse_repo_id(repo_id)?;

        // Determine files to download
        let files = if options.files.is_empty() {
            vec!["model.safetensors".to_string(), "config.json".to_string()]
        } else {
            options.files.clone()
        };

        // Check for security risks
        for file in &files {
            if let Some(format) = WeightFormat::from_filename(file) {
                if !format.is_safe() && !options.allow_pytorch_pickle {
                    return Err(FetchError::PickleSecurityRisk);
                }
            }
        }

        // Create local cache path
        let cache_path = options
            .cache_dir
            .clone()
            .unwrap_or_else(|| self.cache_dir.clone())
            .join(repo_id.replace('/', "--"))
            .join(&options.revision);

        std::fs::create_dir_all(&cache_path)?;

        // Detect format from files
        let format = files
            .iter()
            .find_map(|f| WeightFormat::from_filename(f))
            .unwrap_or(WeightFormat::SafeTensors);

        // Use hf-hub for actual downloads
        let mut api_builder =
            hf_hub::api::sync::ApiBuilder::new().with_cache_dir(cache_path.clone());

        if let Some(token) = &self.token {
            api_builder = api_builder.with_token(Some(token.clone()));
        }

        let api = api_builder
            .build()
            .map_err(|e| FetchError::ConfigParseError {
                message: format!("Failed to initialize HF API: {e}"),
            })?;

        let repo = api.model(repo_id.to_string());

        // Download each requested file
        for file in &files {
            let download_result = if options.revision == "main" {
                repo.get(file)
            } else {
                // For non-main revisions, we need to use repo.revision()
                let revision_repo = api.repo(hf_hub::Repo::with_revision(
                    repo_id.to_string(),
                    hf_hub::RepoType::Model,
                    options.revision.clone(),
                ));
                revision_repo.get(file)
            };

            match download_result {
                Ok(path) => {
                    // Copy to our cache structure if not already there
                    let dest = cache_path.join(file);
                    if path != dest {
                        if let Some(parent) = dest.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        // Only copy if source exists and dest doesn't
                        if path.exists() && !dest.exists() {
                            std::fs::copy(&path, &dest)?;
                        }
                    }
                }
                Err(hf_hub::api::sync::ApiError::RequestError(e)) => {
                    // Check if it's a 404
                    if e.to_string().contains("404") {
                        return Err(FetchError::FileNotFound {
                            repo: repo_id.to_string(),
                            file: file.clone(),
                        });
                    }
                    return Err(FetchError::ConfigParseError {
                        message: format!("Download failed: {e}"),
                    });
                }
                Err(e) => {
                    return Err(FetchError::ConfigParseError {
                        message: format!("Download failed: {e}"),
                    });
                }
            }
        }

        Ok(ModelArtifact {
            path: cache_path,
            format,
            architecture: None,
            sha256: options.verify_sha256,
        })
    }

    /// Estimate memory required to load a model
    #[must_use]
    pub fn estimate_memory(param_count: u64, dtype_bytes: u8) -> u64 {
        param_count * u64::from(dtype_bytes)
    }
}

impl Default for HfModelFetcher {
    fn default() -> Self {
        Self::new().expect("Failed to create HfModelFetcher")
    }
}
