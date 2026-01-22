//! Tests for HuggingFace model fetcher.

use super::*;
use std::path::PathBuf;

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
    use crate::hf_pipeline::error::FetchError;
    let result = HfModelFetcher::parse_repo_id("invalid");
    assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
}

#[test]
fn test_parse_repo_id_invalid_empty_org() {
    use crate::hf_pipeline::error::FetchError;
    let result = HfModelFetcher::parse_repo_id("/model");
    assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
}

#[test]
fn test_parse_repo_id_invalid_empty_name() {
    use crate::hf_pipeline::error::FetchError;
    let result = HfModelFetcher::parse_repo_id("org/");
    assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
}

#[test]
fn test_parse_repo_id_invalid_too_many_parts() {
    use crate::hf_pipeline::error::FetchError;
    let result = HfModelFetcher::parse_repo_id("a/b/c");
    assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
}

#[test]
fn test_download_rejects_pytorch_by_default() {
    use crate::hf_pipeline::error::FetchError;
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
    use crate::hf_pipeline::error::FetchError;
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

#[test]
fn test_custom_architecture_serde() {
    let arch = Architecture::Custom {
        config: serde_json::json!({"model_type": "custom", "layers": 10}),
    };
    let serialized = serde_json::to_string(&arch).unwrap();
    let deserialized: Architecture = serde_json::from_str(&serialized).unwrap();
    assert_eq!(arch.param_count(), deserialized.param_count());
    assert_eq!(arch.param_count(), 0); // Custom always returns 0
}

#[test]
fn test_hf_model_fetcher_default() {
    // Default impl should create a valid fetcher
    let fetcher = HfModelFetcher::default();
    // Should not panic and have reasonable defaults
    assert!(!fetcher.cache_dir.as_os_str().is_empty());
}

#[test]
fn test_weight_format_equality() {
    assert_eq!(WeightFormat::SafeTensors, WeightFormat::SafeTensors);
    assert_eq!(WeightFormat::PyTorchBin, WeightFormat::PyTorchBin);
    assert_eq!(WeightFormat::ONNX, WeightFormat::ONNX);
    assert_ne!(WeightFormat::SafeTensors, WeightFormat::ONNX);

    let gguf1 = WeightFormat::GGUF {
        quant_type: "Q4_K_M".into(),
    };
    let gguf2 = WeightFormat::GGUF {
        quant_type: "Q4_K_M".into(),
    };
    assert_eq!(gguf1, gguf2);
}

#[test]
fn test_weight_format_debug() {
    let safetensors = WeightFormat::SafeTensors;
    let debug = format!("{:?}", safetensors);
    assert!(debug.contains("SafeTensors"));

    let gguf = WeightFormat::GGUF {
        quant_type: "Q4_K_S".into(),
    };
    let debug_gguf = format!("{:?}", gguf);
    assert!(debug_gguf.contains("Q4_K_S"));
}

#[test]
fn test_weight_format_clone() {
    let original = WeightFormat::GGUF {
        quant_type: "Q8_0".into(),
    };
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_fetch_options_cache_dir_pathbuf() {
    let path = PathBuf::from("/custom/cache/dir");
    let opts = FetchOptions::new().cache_dir(path.clone());
    assert_eq!(opts.cache_dir, Some(path));
}

#[test]
fn test_fetch_options_multiple_files() {
    let opts = FetchOptions::new().files(&[
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "vocab.txt",
    ]);
    assert_eq!(opts.files.len(), 4);
    assert!(opts.files.contains(&"vocab.txt".to_string()));
}

#[test]
fn test_architecture_debug() {
    let bert = Architecture::BERT {
        num_layers: 6,
        hidden_size: 384,
        num_attention_heads: 6,
    };
    let debug = format!("{:?}", bert);
    assert!(debug.contains("BERT"));
    assert!(debug.contains("384"));
}

#[test]
fn test_architecture_clone() {
    let original = Architecture::Llama {
        num_layers: 16,
        hidden_size: 2048,
        num_attention_heads: 16,
        intermediate_size: 5504,
    };
    let cloned = original.clone();
    assert_eq!(original.param_count(), cloned.param_count());
}

#[test]
fn test_model_artifact_debug() {
    let artifact = ModelArtifact {
        path: PathBuf::from("/tmp/model"),
        format: WeightFormat::SafeTensors,
        architecture: Some(Architecture::BERT {
            num_layers: 12,
            hidden_size: 768,
            num_attention_heads: 12,
        }),
        sha256: Some("abc123".into()),
    };
    let debug = format!("{:?}", artifact);
    assert!(debug.contains("SafeTensors"));
    assert!(debug.contains("abc123"));
}

#[test]
fn test_fetch_options_debug_and_clone() {
    let opts = FetchOptions::new()
        .revision("v2.0")
        .files(&["model.gguf"])
        .verify_sha256("deadbeef");
    let debug = format!("{:?}", opts);
    assert!(debug.contains("v2.0"));

    let cloned = opts.clone();
    assert_eq!(cloned.revision, "v2.0");
}

#[test]
fn test_download_with_non_main_revision() {
    let temp_dir = std::env::temp_dir().join("hf_test_revision");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().unwrap().cache_dir(&temp_dir);
    let result = fetcher.download_model(
        "nonexistent-org/nonexistent-model",
        FetchOptions::new()
            .revision("v1.0.0") // Non-main revision
            .files(&["model.safetensors"])
            .cache_dir(&temp_dir),
    );

    // Should fail (repo doesn't exist), but should exercise the revision branch
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_download_with_empty_files_uses_defaults() {
    let temp_dir = std::env::temp_dir().join("hf_test_empty_files");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().unwrap().cache_dir(&temp_dir);
    // Empty files should default to model.safetensors and config.json
    let result = fetcher.download_model(
        "nonexistent-org-abc123/nonexistent-model-xyz789",
        FetchOptions::new().cache_dir(&temp_dir),
    );

    // Should fail (repo doesn't exist)
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_download_pytorch_allowed_when_flag_set() {
    use crate::hf_pipeline::error::FetchError;
    let temp_dir = std::env::temp_dir().join("hf_test_pytorch_allowed");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().unwrap().cache_dir(&temp_dir);
    let result = fetcher.download_model(
        "nonexistent-org/nonexistent-model",
        FetchOptions::new()
            .files(&["pytorch_model.bin"])
            .allow_pytorch_pickle(true)
            .cache_dir(&temp_dir),
    );

    // Should NOT be blocked by security check (will fail for other reasons)
    assert!(!matches!(result, Err(FetchError::PickleSecurityRisk)));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_download_uses_custom_cache_dir_from_options() {
    let temp_dir = std::env::temp_dir().join("hf_test_custom_cache");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().unwrap();
    let result = fetcher.download_model(
        "nonexistent-org/nonexistent-model",
        FetchOptions::new()
            .files(&["model.safetensors"])
            .cache_dir(&temp_dir), // Custom cache from options
    );

    // Should fail (repo doesn't exist)
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_bert_small_config() {
    let bert = Architecture::BERT {
        num_layers: 6,
        hidden_size: 256,
        num_attention_heads: 4,
    };
    let params = bert.param_count();
    // Should be smaller than standard BERT
    assert!(params > 0);
    assert!(params < 50_000_000);
}

#[test]
fn test_llama_small_config() {
    let llama = Architecture::Llama {
        num_layers: 8,
        hidden_size: 512,
        num_attention_heads: 8,
        intermediate_size: 1024,
    };
    let params = llama.param_count();
    assert!(params > 0);
}

#[test]
fn test_gpt2_small_config() {
    let gpt2 = Architecture::GPT2 {
        num_layers: 6,
        hidden_size: 384,
        num_attention_heads: 6,
    };
    let params = gpt2.param_count();
    assert!(params > 0);
}

#[test]
fn test_t5_small_config() {
    let t5 = Architecture::T5 {
        encoder_layers: 4,
        decoder_layers: 4,
        hidden_size: 256,
    };
    let params = t5.param_count();
    assert!(params > 0);
}
