//! Model Export Formats
//!
//! Supports exporting trained/distilled models to various formats:
//! - SafeTensors: Safe, fast tensor serialization
//! - APR: Aprender format for entrenar ecosystem
//! - GGUF: Quantized format for llama.cpp compatibility
//!
//! # Example
//!
//! ```ignore
//! use entrenar::hf_pipeline::export::{Exporter, ExportFormat};
//!
//! let exporter = Exporter::new();
//! exporter.export(&model_weights, ExportFormat::SafeTensors, "output/model.safetensors")?;
//! ```

mod exporter;
mod format;
mod result;
mod types;
mod weights;

#[cfg(test)]
mod tests;

// Public re-exports
pub use exporter::Exporter;
pub use format::ExportFormat;
pub use result::ExportResult;
#[allow(unused_imports)]
pub use weights::TrainingMetadata;
pub use weights::{ModelMetadata, ModelWeights};
