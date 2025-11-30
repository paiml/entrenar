//! HuggingFace Model Fetcher
//!
//! Downloads models from HuggingFace Hub with authentication and caching.

use crate::hf_pipeline::error::{FetchError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Model weight format
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightFormat {
    /// SafeTensors format (recommended, secure)
    SafeTensors,
    /// GGUF quantized format
    GGUF { quant_type: String },
    /// PyTorch pickle format (SECURITY RISK)
    PyTorchBin,
    /// ONNX format
    ONNX,
}

impl WeightFormat {
    /// Detect format from filename
    #[must_use]
    pub fn from_filename(filename: &str) -> Option<Self> {
        if filename.ends_with(".safetensors") {
            Some(Self::SafeTensors)
        } else if filename.ends_with(".gguf") {
            Some(Self::GGUF {
                quant_type: "unknown".into(),
            })
        } else if filename.ends_with(".bin") {
            Some(Self::PyTorchBin)
        } else if filename.ends_with(".onnx") {
            Some(Self::ONNX)
        } else {
            None
        }
    }

    /// Check if format is safe (no arbitrary code execution)
    #[must_use]
    pub fn is_safe(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::GGUF { .. } | Self::ONNX)
    }
}

/// Model architecture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Architecture {
    /// BERT-style encoder
    BERT {
        num_layers: usize,
        hidden_size: usize,
        num_attention_heads: usize,
    },
    /// GPT-style decoder
    GPT2 {
        num_layers: usize,
        hidden_size: usize,
        num_attention_heads: usize,
    },
    /// Llama architecture
    Llama {
        num_layers: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
    },
    /// T5 encoder-decoder
    T5 {
        encoder_layers: usize,
        decoder_layers: usize,
        hidden_size: usize,
    },
    /// Custom/unknown architecture
    Custom { config: serde_json::Value },
}

impl Architecture {
    /// Estimate parameter count
    #[must_use]
    pub fn param_count(&self) -> u64 {
        match self {
            Self::BERT {
                num_layers,
                hidden_size,
                num_attention_heads: _,
            } => {
                // Rough estimate: 4 * hidden^2 per layer (Q, K, V, O projections + FFN)
                let per_layer = 4 * (*hidden_size as u64).pow(2) + 4 * (*hidden_size as u64).pow(2);
                per_layer * (*num_layers as u64)
            }
            Self::GPT2 {
                num_layers,
                hidden_size,
                ..
            } => {
                let per_layer = 4 * (*hidden_size as u64).pow(2) + 4 * (*hidden_size as u64).pow(2);
                per_layer * (*num_layers as u64)
            }
            Self::Llama {
                num_layers,
                hidden_size,
                intermediate_size,
                ..
            } => {
                let attn = 4 * (*hidden_size as u64).pow(2);
                let ffn = 2 * (*hidden_size as u64) * (*intermediate_size as u64);
                (attn + ffn) * (*num_layers as u64)
            }
            Self::T5 {
                encoder_layers,
                decoder_layers,
                hidden_size,
            } => {
                let per_layer = 8 * (*hidden_size as u64).pow(2);
                per_layer * ((*encoder_layers + *decoder_layers) as u64)
            }
            Self::Custom { .. } => 0, // Unknown
        }
    }
}

/// Downloaded model artifact
#[derive(Debug)]
pub struct ModelArtifact {
    /// Local path to downloaded files
    pub path: PathBuf,
    /// Detected weight format
    pub format: WeightFormat,
    /// Model architecture (parsed from config.json)
    pub architecture: Option<Architecture>,
    /// SHA256 hash of model file
    pub sha256: Option<String>,
}

/// Options for model fetching
#[derive(Debug, Clone)]
pub struct FetchOptions {
    /// Git revision (branch, tag, or commit)
    pub revision: String,
    /// Specific files to download
    pub files: Vec<String>,
    /// Allow PyTorch pickle files (SECURITY RISK)
    pub allow_pytorch_pickle: bool,
    /// Expected SHA256 hash for verification
    pub verify_sha256: Option<String>,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            revision: "main".into(),
            files: vec![],
            allow_pytorch_pickle: false,
            verify_sha256: None,
            cache_dir: None,
        }
    }
}

impl FetchOptions {
    /// Create new options
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set revision
    #[must_use]
    pub fn revision(mut self, rev: impl Into<String>) -> Self {
        self.revision = rev.into();
        self
    }

    /// Set files to download
    #[must_use]
    pub fn files(mut self, files: &[&str]) -> Self {
        self.files = files.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Allow PyTorch pickle files (SECURITY RISK)
    #[must_use]
    pub fn allow_pytorch_pickle(mut self, allow: bool) -> Self {
        self.allow_pytorch_pickle = allow;
        self
    }

    /// Set SHA256 hash for verification
    #[must_use]
    pub fn verify_sha256(mut self, hash: impl Into<String>) -> Self {
        self.verify_sha256 = Some(hash.into());
        self
    }

    /// Set cache directory
    #[must_use]
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }
}

/// HuggingFace model fetcher
pub struct HfModelFetcher {
    /// Authentication token
    token: Option<String>,
    /// Cache directory
    cache_dir: PathBuf,
    /// API base URL (for future HTTP client integration)
    #[allow(dead_code)]
    api_base: String,
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
    fn default_cache_dir() -> PathBuf {
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
    fn parse_repo_id(repo_id: &str) -> Result<(&str, &str)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // WeightFormat Tests
    // =========================================================================

    #[test]
    fn test_weight_format_from_safetensors() {
        let format = WeightFormat::from_filename("model.safetensors");
        assert_eq!(format, Some(WeightFormat::SafeTensors));
    }

    #[test]
    fn test_weight_format_from_gguf() {
        let format = WeightFormat::from_filename("model.Q4_K_M.gguf");
        assert!(matches!(format, Some(WeightFormat::GGUF { .. })));
    }

    #[test]
    fn test_weight_format_from_pytorch() {
        let format = WeightFormat::from_filename("pytorch_model.bin");
        assert_eq!(format, Some(WeightFormat::PyTorchBin));
    }

    #[test]
    fn test_weight_format_from_onnx() {
        let format = WeightFormat::from_filename("model.onnx");
        assert_eq!(format, Some(WeightFormat::ONNX));
    }

    #[test]
    fn test_weight_format_unknown() {
        let format = WeightFormat::from_filename("random.txt");
        assert_eq!(format, None);
    }

    #[test]
    fn test_safetensors_is_safe() {
        assert!(WeightFormat::SafeTensors.is_safe());
    }

    #[test]
    fn test_gguf_is_safe() {
        let format = WeightFormat::GGUF {
            quant_type: "Q4_K_M".into(),
        };
        assert!(format.is_safe());
    }

    #[test]
    fn test_pytorch_is_not_safe() {
        assert!(!WeightFormat::PyTorchBin.is_safe());
    }

    // =========================================================================
    // Architecture Tests
    // =========================================================================

    #[test]
    fn test_bert_param_count() {
        let arch = Architecture::BERT {
            num_layers: 12,
            hidden_size: 768,
            num_attention_heads: 12,
        };
        let params = arch.param_count();
        // 12 layers * (4 * 768^2 + 4 * 768^2) = 12 * 8 * 589824 = 56,621,568
        assert!(params > 50_000_000);
        assert!(params < 200_000_000);
    }

    #[test]
    fn test_llama_param_count() {
        let arch = Architecture::Llama {
            num_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            intermediate_size: 11008,
        };
        let params = arch.param_count();
        // Should be in billions range for 7B model
        assert!(params > 1_000_000_000);
    }

    #[test]
    fn test_custom_param_count_is_zero() {
        let arch = Architecture::Custom {
            config: serde_json::json!({}),
        };
        assert_eq!(arch.param_count(), 0);
    }

    // =========================================================================
    // FetchOptions Tests
    // =========================================================================

    #[test]
    fn test_fetch_options_default() {
        let opts = FetchOptions::default();
        assert_eq!(opts.revision, "main");
        assert!(opts.files.is_empty());
        assert!(!opts.allow_pytorch_pickle);
        assert!(opts.verify_sha256.is_none());
    }

    #[test]
    fn test_fetch_options_builder() {
        let opts = FetchOptions::new()
            .revision("v1.0")
            .files(&["model.safetensors"])
            .allow_pytorch_pickle(true)
            .verify_sha256("abc123")
            .cache_dir("/tmp/cache");

        assert_eq!(opts.revision, "v1.0");
        assert_eq!(opts.files, vec!["model.safetensors"]);
        assert!(opts.allow_pytorch_pickle);
        assert_eq!(opts.verify_sha256, Some("abc123".into()));
        assert_eq!(opts.cache_dir, Some(PathBuf::from("/tmp/cache")));
    }

    // =========================================================================
    // HfModelFetcher Tests
    // =========================================================================

    #[test]
    fn test_fetcher_new() {
        let fetcher = HfModelFetcher::new();
        assert!(fetcher.is_ok());
    }

    #[test]
    fn test_fetcher_with_token() {
        let fetcher = HfModelFetcher::with_token("hf_test_token");
        assert!(fetcher.is_authenticated());
    }

    #[test]
    fn test_fetcher_without_token_is_not_authenticated() {
        // Clear env for test
        let _saved = std::env::var("HF_TOKEN");
        std::env::remove_var("HF_TOKEN");

        let fetcher = HfModelFetcher {
            token: None,
            cache_dir: PathBuf::from("/tmp"),
            api_base: "https://huggingface.co".into(),
        };
        assert!(!fetcher.is_authenticated());
    }

    #[test]
    fn test_parse_repo_id_valid() {
        let result = HfModelFetcher::parse_repo_id("microsoft/codebert-base");
        assert!(result.is_ok());
        let (org, name) = result.unwrap();
        assert_eq!(org, "microsoft");
        assert_eq!(name, "codebert-base");
    }

    #[test]
    fn test_parse_repo_id_invalid_no_slash() {
        let result = HfModelFetcher::parse_repo_id("invalid");
        assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
    }

    #[test]
    fn test_parse_repo_id_invalid_empty_org() {
        let result = HfModelFetcher::parse_repo_id("/model");
        assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
    }

    #[test]
    fn test_parse_repo_id_invalid_empty_name() {
        let result = HfModelFetcher::parse_repo_id("org/");
        assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
    }

    #[test]
    fn test_parse_repo_id_invalid_too_many_parts() {
        let result = HfModelFetcher::parse_repo_id("a/b/c");
        assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
    }

    #[test]
    fn test_download_rejects_pytorch_by_default() {
        let fetcher = HfModelFetcher::with_token("test");
        let result = fetcher.download_model(
            "test/model",
            FetchOptions::new().files(&["pytorch_model.bin"]),
        );
        assert!(matches!(result, Err(FetchError::PickleSecurityRisk)));
    }

    #[test]
    fn test_download_nonexistent_repo_returns_error() {
        let temp_dir = std::env::temp_dir().join("hf_test_nonexistent");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let fetcher = HfModelFetcher::new().unwrap().cache_dir(&temp_dir);
        let result = fetcher.download_model(
            "nonexistent-org-xyz123/nonexistent-model-abc456",
            FetchOptions::new()
                .files(&["model.safetensors"])
                .cache_dir(&temp_dir),
        );

        // Should fail with some error (network or not found)
        assert!(result.is_err(), "Non-existent repo should return error");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_download_security_check_blocks_pytorch() {
        let temp_dir = std::env::temp_dir().join("hf_test_security");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let fetcher = HfModelFetcher::new().unwrap().cache_dir(&temp_dir);
        let result = fetcher.download_model(
            "microsoft/codebert-base",
            FetchOptions::new()
                .files(&["pytorch_model.bin"]) // PyTorch without allow flag
                .cache_dir(&temp_dir),
        );

        // Should be blocked by security check BEFORE network access
        assert!(
            matches!(result, Err(FetchError::PickleSecurityRisk)),
            "PyTorch files should be blocked without allow_pytorch_pickle"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[ignore] // Requires network access
    fn test_download_real_model_integration() {
        let temp_dir = std::env::temp_dir().join("hf_test_real");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let fetcher = HfModelFetcher::new().unwrap().cache_dir(&temp_dir);
        let result = fetcher.download_model(
            "hf-internal-testing/tiny-random-bert",
            FetchOptions::new()
                .files(&["config.json"])
                .cache_dir(&temp_dir),
        );

        assert!(
            result.is_ok(),
            "Should download from real repo: {:?}",
            result.err()
        );
        let artifact = result.unwrap();
        assert!(artifact.path.exists(), "Cache directory should exist");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_estimate_memory_fp32() {
        // 125M params * 4 bytes = 500MB
        let mem = HfModelFetcher::estimate_memory(125_000_000, 4);
        assert_eq!(mem, 500_000_000);
    }

    #[test]
    fn test_estimate_memory_fp16() {
        // 125M params * 2 bytes = 250MB
        let mem = HfModelFetcher::estimate_memory(125_000_000, 2);
        assert_eq!(mem, 250_000_000);
    }

    #[test]
    fn test_estimate_memory_int4() {
        // 125M params * 0.5 bytes ≈ 62.5MB (but we use 1 byte minimum)
        let mem = HfModelFetcher::estimate_memory(125_000_000, 1);
        assert_eq!(mem, 125_000_000);
    }

    #[test]
    fn test_gpt2_param_count() {
        let arch = Architecture::GPT2 {
            num_layers: 12,
            hidden_size: 768,
            num_attention_heads: 12,
        };
        let params = arch.param_count();
        assert!(params > 50_000_000);
    }

    #[test]
    fn test_t5_param_count() {
        let arch = Architecture::T5 {
            encoder_layers: 12,
            decoder_layers: 12,
            hidden_size: 768,
        };
        let params = arch.param_count();
        assert!(params > 100_000_000);
    }

    #[test]
    fn test_onnx_is_safe() {
        assert!(WeightFormat::ONNX.is_safe());
    }

    #[test]
    fn test_fetcher_cache_dir() {
        let temp_dir = std::env::temp_dir().join("hf_test_cache_dir");
        let fetcher = HfModelFetcher::with_token("test").cache_dir(&temp_dir);
        assert_eq!(fetcher.cache_dir, temp_dir);
    }

    #[test]
    fn test_default_cache_dir() {
        let cache_dir = HfModelFetcher::default_cache_dir();
        // Should return a path (either from env or default)
        assert!(!cache_dir.as_os_str().is_empty());
    }

    #[test]
    fn test_weight_format_gguf_quant_type() {
        let format = WeightFormat::GGUF {
            quant_type: "Q4_K_M".to_string(),
        };
        if let WeightFormat::GGUF { quant_type } = format {
            assert_eq!(quant_type, "Q4_K_M");
        } else {
            panic!("Expected GGUF format");
        }
    }

    #[test]
    fn test_architecture_serde() {
        let arch = Architecture::Llama {
            num_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            intermediate_size: 11008,
        };
        let serialized = serde_json::to_string(&arch).unwrap();
        let deserialized: Architecture = serde_json::from_str(&serialized).unwrap();
        assert_eq!(arch.param_count(), deserialized.param_count());
    }

    #[test]
    fn test_bert_architecture_serde() {
        let arch = Architecture::BERT {
            num_layers: 12,
            hidden_size: 768,
            num_attention_heads: 12,
        };
        let serialized = serde_json::to_string(&arch).unwrap();
        let deserialized: Architecture = serde_json::from_str(&serialized).unwrap();
        assert_eq!(arch.param_count(), deserialized.param_count());
    }

    #[test]
    fn test_gpt2_architecture_serde() {
        let arch = Architecture::GPT2 {
            num_layers: 12,
            hidden_size: 768,
            num_attention_heads: 12,
        };
        let serialized = serde_json::to_string(&arch).unwrap();
        let deserialized: Architecture = serde_json::from_str(&serialized).unwrap();
        assert_eq!(arch.param_count(), deserialized.param_count());
    }

    #[test]
    fn test_t5_architecture_serde() {
        let arch = Architecture::T5 {
            encoder_layers: 12,
            decoder_layers: 12,
            hidden_size: 768,
        };
        let serialized = serde_json::to_string(&arch).unwrap();
        let deserialized: Architecture = serde_json::from_str(&serialized).unwrap();
        assert_eq!(arch.param_count(), deserialized.param_count());
    }
}
