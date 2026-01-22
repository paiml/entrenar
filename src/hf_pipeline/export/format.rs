//! Export format selection and detection.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Export format selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// SafeTensors format (recommended)
    SafeTensors,
    /// Aprender format (JSON-based)
    APR,
    /// GGUF quantized format
    GGUF,
    /// PyTorch state dict (for compatibility)
    PyTorch,
}

impl ExportFormat {
    /// Get file extension for format
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::APR => "apr.json",
            Self::GGUF => "gguf",
            Self::PyTorch => "pt",
        }
    }

    /// Check if format is safe (no pickle/arbitrary code)
    #[must_use]
    pub fn is_safe(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::APR | Self::GGUF)
    }

    /// Detect format from file path
    #[must_use]
    pub fn from_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?;
        if name.ends_with(".safetensors") {
            Some(Self::SafeTensors)
        } else if name.ends_with(".apr.json") || name.ends_with(".apr") {
            Some(Self::APR)
        } else if name.ends_with(".gguf") {
            Some(Self::GGUF)
        } else if name.ends_with(".pt") || name.ends_with(".bin") {
            Some(Self::PyTorch)
        } else {
            None
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::APR => write!(f, "APR"),
            Self::GGUF => write!(f, "GGUF"),
            Self::PyTorch => write!(f, "PyTorch"),
        }
    }
}
