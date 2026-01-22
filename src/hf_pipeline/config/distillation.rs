//! Distillation loss configuration

use serde::{Deserialize, Serialize};

/// Distillation loss configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DistillationConfig {
    /// Temperature for softening distributions
    pub temperature: f32,
    /// Alpha weight for soft vs hard loss
    pub alpha: f32,
    /// Progressive distillation config
    pub progressive: Option<ProgressiveConfig>,
    /// Attention transfer config
    pub attention_transfer: Option<AttentionTransferConfig>,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,
            progressive: None,
            attention_transfer: None,
        }
    }
}

/// Progressive distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Layer mapping [[student_layer, teacher_layer], ...]
    pub layer_mapping: Vec<[usize; 2]>,
    /// Weight for hidden state loss
    #[serde(default = "default_hidden_weight")]
    pub hidden_weight: f32,
}

fn default_hidden_weight() -> f32 {
    1.0
}

/// Attention transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionTransferConfig {
    /// Weight for attention transfer loss
    #[serde(default = "default_attention_weight")]
    pub weight: f32,
}

fn default_attention_weight() -> f32 {
    0.1
}
