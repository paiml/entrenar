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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Architecture type (transformer, sequential)
    #[serde(rename = "type")]
    pub arch_type: String,

    /// Hidden size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<usize>,

    /// Number of layers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_layers: Option<usize>,

    /// Number of attention heads
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_heads: Option<usize>,

    /// Vocabulary size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,

    /// Maximum sequence length
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seq_length: Option<usize>,

    /// Sequential layers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<HashMap<String, serde_json::Value>>>,
}
