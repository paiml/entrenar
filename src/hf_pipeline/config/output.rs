//! Output configuration

use serde::{Deserialize, Serialize};

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Output directory
    pub dir: String,
    /// Save checkpoints every N steps
    pub save_steps: usize,
    /// Evaluate every N steps
    pub eval_steps: usize,
    /// Log every N steps
    pub log_steps: usize,
    /// Push to HuggingFace Hub
    pub push_to_hub: bool,
    /// Hub repository ID
    pub hub_repo_id: Option<String>,
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
