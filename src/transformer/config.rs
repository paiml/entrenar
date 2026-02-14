//! Transformer configuration module
//!
//! This module provides configuration structures for transformer models.

use serde::{Deserialize, Serialize};

// Well-known model architecture constants
const LLAMA2_7B_INTERMEDIATE_SIZE: usize = 11008;
const LLAMA2_13B_HIDDEN_SIZE: usize = 5120;
const LLAMA2_13B_INTERMEDIATE_SIZE: usize = 13824;
const LLAMA_VOCAB_SIZE: usize = 32000;
const MISTRAL_INTERMEDIATE_SIZE: usize = 14336;
const MISTRAL_MAX_SEQ_LEN: usize = 32768;
const QWEN2_0_5B_HIDDEN_SIZE: usize = 896;
const QWEN2_0_5B_INTERMEDIATE_SIZE: usize = 4864;
const QWEN2_VOCAB_SIZE: usize = 151936;
const QWEN2_MAX_SEQ_LEN: usize = 32768;
const QWEN2_ROPE_THETA: f32 = 1_000_000.0;
const DEFAULT_ROPE_THETA: f32 = 10000.0;

/// Configuration for transformer models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Hidden dimension (embedding size)
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for grouped-query attention)
    pub num_kv_heads: usize,
    /// Feed-forward network intermediate dimension
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// RMS normalization epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta base
    pub rope_theta: f32,
    /// Whether to use bias in linear layers
    pub use_bias: bool,
}

impl TransformerConfig {
    /// LLaMA 2 7B configuration
    pub fn llama2_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_kv_heads: 32,
            intermediate_size: LLAMA2_7B_INTERMEDIATE_SIZE,
            num_hidden_layers: 32,
            vocab_size: LLAMA_VOCAB_SIZE,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: DEFAULT_ROPE_THETA,
            use_bias: false,
        }
    }

    /// LLaMA 2 13B configuration
    pub fn llama2_13b() -> Self {
        Self {
            hidden_size: LLAMA2_13B_HIDDEN_SIZE,
            num_attention_heads: 40,
            num_kv_heads: 40,
            intermediate_size: LLAMA2_13B_INTERMEDIATE_SIZE,
            num_hidden_layers: 40,
            vocab_size: LLAMA_VOCAB_SIZE,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: DEFAULT_ROPE_THETA,
            use_bias: false,
        }
    }

    /// Mistral 7B configuration
    pub fn mistral_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_kv_heads: 8, // Grouped-query attention
            intermediate_size: MISTRAL_INTERMEDIATE_SIZE,
            num_hidden_layers: 32,
            vocab_size: LLAMA_VOCAB_SIZE,
            max_position_embeddings: MISTRAL_MAX_SEQ_LEN,
            rms_norm_eps: 1e-5,
            rope_theta: DEFAULT_ROPE_THETA,
            use_bias: false,
        }
    }

    /// Qwen2 0.5B configuration (good for testing)
    pub fn qwen2_0_5b() -> Self {
        Self {
            hidden_size: QWEN2_0_5B_HIDDEN_SIZE,
            num_attention_heads: 14,
            num_kv_heads: 2,
            intermediate_size: QWEN2_0_5B_INTERMEDIATE_SIZE,
            num_hidden_layers: 24,
            vocab_size: QWEN2_VOCAB_SIZE,
            max_position_embeddings: QWEN2_MAX_SEQ_LEN,
            rms_norm_eps: 1e-6,
            rope_theta: QWEN2_ROPE_THETA,
            use_bias: true,
        }
    }

    /// Tiny configuration for testing
    pub fn tiny() -> Self {
        Self {
            hidden_size: 64,
            num_attention_heads: 2,
            num_kv_heads: 2,
            intermediate_size: 256,
            num_hidden_layers: 2,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: DEFAULT_ROPE_THETA,
            use_bias: false,
        }
    }

    /// Per-head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_config_llama2() {
        let config = TransformerConfig::llama2_7b();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_transformer_config_tiny() {
        let config = TransformerConfig::tiny();
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_attention_heads, 2);
        assert_eq!(config.head_dim(), 32);
    }

    #[test]
    fn test_config_serialization() {
        let config = TransformerConfig::llama2_7b();
        let json = serde_json::to_string(&config).unwrap();
        let restored: TransformerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.hidden_size, config.hidden_size);
        assert_eq!(restored.num_attention_heads, config.num_attention_heads);
    }

    #[test]
    fn test_mistral_config() {
        let config = TransformerConfig::mistral_7b();
        assert_eq!(config.num_kv_heads, 8); // Grouped-query attention
        assert_eq!(config.num_attention_heads, 32);
        // 32 / 8 = 4 query heads per KV head
    }

    #[test]
    fn test_qwen2_config() {
        let config = TransformerConfig::qwen2_0_5b();
        assert!(config.use_bias);
        assert_eq!(config.vocab_size, 151936);
    }

    #[test]
    fn test_llama2_13b_config() {
        let config = TransformerConfig::llama2_13b();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_attention_heads, 40);
        assert_eq!(config.num_hidden_layers, 40);
        assert_eq!(config.head_dim(), 128); // 5120 / 40 = 128
    }

    #[test]
    fn test_config_yaml_serialization() {
        let config = TransformerConfig::tiny();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: TransformerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.hidden_size, config.hidden_size);
        assert_eq!(restored.num_hidden_layers, config.num_hidden_layers);
    }

    #[test]
    fn test_grouped_query_attention_ratio() {
        let config = TransformerConfig::mistral_7b();
        let heads_per_kv = config.num_attention_heads / config.num_kv_heads;
        assert_eq!(heads_per_kv, 4); // 32 / 8 = 4
    }

    #[test]
    fn test_config_clone() {
        let config = TransformerConfig::llama2_7b();
        let cloned = config.clone();
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.vocab_size, cloned.vocab_size);
    }
}
