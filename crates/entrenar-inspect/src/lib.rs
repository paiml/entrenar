//! SafeTensors model inspection and format conversion.
//!
//! This crate provides tools for:
//! - Inspecting model architectures and layer structures
//! - Analyzing memory requirements
//! - Converting between formats (SafeTensors, GGUF, APR)
//!
//! # Toyota Way Principles
//!
//! - **Andon**: Integrity checker surfaces data problems immediately
//! - **Genchi Genbutsu**: Direct inspection of actual model weights
//! - **SMED**: Quick format conversion for deployment changeover

pub mod architecture;
pub mod convert;
pub mod inspect;
pub mod validate;

pub use architecture::{Architecture, ArchitectureDetector};
pub use convert::{ConversionResult, FormatConverter};
pub use inspect::{ModelInfo, TensorInfo};
pub use validate::{IntegrityChecker, ValidationResult};

use entrenar_common::Result;
use std::path::Path;

/// Inspect a model file and return detailed information.
pub fn inspect(path: impl AsRef<Path>) -> Result<ModelInfo> {
    inspect::inspect_model(path)
}

/// Validate a model file's integrity.
pub fn validate(path: impl AsRef<Path>) -> Result<ValidationResult> {
    validate::validate_model(path)
}

/// Convert a model between formats.
pub fn convert(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    format: OutputFormat,
) -> Result<ConversionResult> {
    convert::convert_model(input, output, format)
}

/// Output format for model conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// SafeTensors format (secure, recommended)
    SafeTensors,
    /// GGUF format (llama.cpp compatible)
    Gguf,
    /// APR format (JSON metadata)
    Apr,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "safetensors" | "st" => Ok(Self::SafeTensors),
            "gguf" => Ok(Self::Gguf),
            "apr" => Ok(Self::Apr),
            _ => Err(format!(
                "Unknown format: {}. Use: safetensors, gguf, apr",
                s
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_parsing() {
        assert_eq!(
            "safetensors".parse::<OutputFormat>().unwrap(),
            OutputFormat::SafeTensors
        );
        assert_eq!("GGUF".parse::<OutputFormat>().unwrap(), OutputFormat::Gguf);
        assert_eq!("apr".parse::<OutputFormat>().unwrap(), OutputFormat::Apr);
    }
}
