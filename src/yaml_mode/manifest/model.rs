//! Model Configuration
//!
//! Contains model-related configuration types for training manifests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model source URI (pacha://, hf://, or local path)
    pub source: String,

    /// Model format (safetensors, gguf, apr, pt)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Architecture override
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<ArchitectureConfig>,

    /// Layers to freeze
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freeze: Option<Vec<String>>,

    /// Device placement (auto, cpu, cuda, cuda:0, mps)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,

    /// Data type (float32, float16, bfloat16)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
}

/// Model architecture configuration
///
/// Supports both preset names and custom architecture parameters.
/// Custom params override values from config.json or preset defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Architecture type (transformer, sequential)
    #[serde(rename = "type")]
    pub arch_type: String,

    /// Hidden size (embedding dimension)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<usize>,

    /// Number of transformer layers
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "num_hidden_layers"
    )]
    pub num_layers: Option<usize>,

    /// Number of attention heads
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "num_attention_heads"
    )]
    pub num_heads: Option<usize>,

    /// Number of key-value heads (for grouped-query attention)
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "num_key_value_heads"
    )]
    pub num_kv_heads: Option<usize>,

    /// FFN intermediate dimension
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intermediate_size: Option<usize>,

    /// Vocabulary size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,

    /// Maximum sequence/position length
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "max_position_embeddings"
    )]
    pub max_seq_length: Option<usize>,

    /// RMS normalization epsilon
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rms_norm_eps: Option<f32>,

    /// RoPE theta (rotary positional encoding base)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f32>,

    /// Whether to use bias in linear layers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_bias: Option<bool>,

    /// Sequential layers (for sequential architecture type)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<HashMap<String, serde_json::Value>>>,
}
