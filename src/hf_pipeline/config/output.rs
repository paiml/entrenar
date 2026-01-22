//! Output configuration

use serde::{Deserialize, Serialize};

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory
    pub dir: String,
    /// Save checkpoints every N steps
    #[serde(default = "default_save_steps")]
    pub save_steps: usize,
    /// Evaluate every N steps
    #[serde(default = "default_eval_steps")]
    pub eval_steps: usize,
    /// Log every N steps
    #[serde(default = "default_log_steps")]
    pub log_steps: usize,
    /// Push to HuggingFace Hub
    #[serde(default)]
    pub push_to_hub: bool,
    /// Hub repository ID
    pub hub_repo_id: Option<String>,
}

fn default_save_steps() -> usize {
    500
}
fn default_eval_steps() -> usize {
    100
}
fn default_log_steps() -> usize {
    10
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: "./output".to_string(),
            save_steps: 500,
            eval_steps: 100,
            log_steps: 10,
            push_to_hub: false,
            hub_repo_id: None,
        }
    }
}
