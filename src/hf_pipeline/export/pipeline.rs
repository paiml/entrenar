//! Quantize-Export pipeline
//!
//! Quantizes model weights and exports them as GGUF files in a single operation.

use crate::hf_pipeline::error::{FetchError, Result};
use crate::hf_pipeline::export::gguf_writer::GgufQuantization;
use std::path::Path;

use super::exporter::Exporter;
use super::format::ExportFormat;
use super::result::ExportResult;
use super::weights::ModelWeights;

/// Result of the quantize-export pipeline
#[derive(Debug, Clone)]
pub struct QuantExportResult {
    /// Export result with file path and size
    pub export: ExportResult,
    /// Quantization mode used
    pub quantization: GgufQuantization,
    /// Generated README content (if any)
    pub readme: Option<String>,
}

/// Quantize model weights and export to GGUF format
///
/// Performs the full quantize→export pipeline:
/// 1. Quantize all tensors according to the config
/// 2. Export as GGUF with tensor data
/// 3. Generate a README with quantization metadata
pub fn quantize_and_export(
    weights: &ModelWeights,
    quantization: GgufQuantization,
    output_dir: impl AsRef<Path>,
    filename: impl AsRef<Path>,
) -> Result<QuantExportResult> {
    let output_dir = output_dir.as_ref();

    // Determine output filename
    let filename = filename.as_ref();

    // Build exporter with quantization config
    let exporter = Exporter::new()
        .output_dir(output_dir)
        .gguf_quantization(quantization);

    let export = exporter.export(weights, ExportFormat::GGUF, filename)?;

    // Generate README with quantization metadata
    let readme = generate_quant_readme(weights, quantization, &export);

    // Write README alongside the model
    let readme_path = output_dir.join("README.md");
    std::fs::write(&readme_path, &readme).map_err(|e| FetchError::GgufWriteError {
        message: format!("Failed to write README: {e}"),
    })?;

    Ok(QuantExportResult {
        export,
        quantization,
        readme: Some(readme),
    })
}

/// Generate a README with quantization metadata
fn generate_quant_readme(
    weights: &ModelWeights,
    quantization: GgufQuantization,
    export: &ExportResult,
) -> String {
    let quant_name = match quantization {
        GgufQuantization::None => "F32 (unquantized)",
        GgufQuantization::Q4_0 => "Q4_0 (4-bit)",
        GgufQuantization::Q8_0 => "Q8_0 (8-bit)",
    };

    let model_name = weights
        .metadata
        .model_name
        .as_deref()
        .unwrap_or("Unknown Model");

    let arch = weights
        .metadata
        .architecture
        .as_deref()
        .unwrap_or("unknown");

    format!(
        "---\ntags:\n- entrenar\n- gguf\n- quantized\n---\n\n\
         # {model_name} ({quant_name})\n\n\
         Quantized with [Entrenar](https://github.com/paiml/entrenar).\n\n\
         ## Model Details\n\n\
         | Property | Value |\n\
         |----------|-------|\n\
         | Architecture | {arch} |\n\
         | Parameters | {} |\n\
         | Quantization | {quant_name} |\n\
         | File Size | {} |\n\
         | Tensors | {} |\n\
         | Format | GGUF v3 |\n",
        weights.metadata.num_params,
        export.size_human(),
        export.num_tensors,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf_pipeline::export::weights::{ModelMetadata, ModelWeights};
    use tempfile::TempDir;

    fn make_test_weights() -> ModelWeights {
        let mut weights = ModelWeights::new();
        weights.add_tensor("layer.0.weight", vec![1.0; 256], vec![16, 16]);
        weights.add_tensor("layer.0.bias", vec![0.1; 16], vec![16]);
        weights.metadata = ModelMetadata {
            model_name: Some("test-model".to_string()),
            architecture: Some("llama".to_string()),
            num_params: 272,
            ..Default::default()
        };
        weights
    }

    #[test]
    fn test_quantize_export_f32() {
        let weights = make_test_weights();
        let tmp = TempDir::new().unwrap();

        let result =
            quantize_and_export(&weights, GgufQuantization::None, tmp.path(), "model.gguf")
                .unwrap();

        assert_eq!(result.quantization, GgufQuantization::None);
        assert!(result.export.size_bytes > 0);
        assert!(result.readme.is_some());
        assert!(tmp.path().join("model.gguf").exists());
        assert!(tmp.path().join("README.md").exists());
    }

    #[test]
    fn test_quantize_export_q4_0() {
        let weights = make_test_weights();
        let tmp = TempDir::new().unwrap();

        let result = quantize_and_export(
            &weights,
            GgufQuantization::Q4_0,
            tmp.path(),
            "model-q4.gguf",
        )
        .unwrap();

        assert_eq!(result.quantization, GgufQuantization::Q4_0);
        assert!(result.export.size_bytes > 0);
    }

    #[test]
    fn test_quantize_export_q8_0() {
        let weights = make_test_weights();
        let tmp = TempDir::new().unwrap();

        let result = quantize_and_export(
            &weights,
            GgufQuantization::Q8_0,
            tmp.path(),
            "model-q8.gguf",
        )
        .unwrap();

        assert_eq!(result.quantization, GgufQuantization::Q8_0);
    }

    #[test]
    fn test_quantize_export_readme_content() {
        let weights = make_test_weights();
        let tmp = TempDir::new().unwrap();

        let result =
            quantize_and_export(&weights, GgufQuantization::Q4_0, tmp.path(), "model.gguf")
                .unwrap();

        let readme = result.readme.unwrap();
        assert!(readme.contains("test-model"));
        assert!(readme.contains("Q4_0"));
        assert!(readme.contains("llama"));
        assert!(readme.contains("entrenar"));
    }

    #[test]
    fn test_quantize_export_q4_smaller_than_f32() {
        let weights = make_test_weights();
        let tmp_f32 = TempDir::new().unwrap();
        let tmp_q4 = TempDir::new().unwrap();

        let f32_result = quantize_and_export(
            &weights,
            GgufQuantization::None,
            tmp_f32.path(),
            "model.gguf",
        )
        .unwrap();
        let q4_result = quantize_and_export(
            &weights,
            GgufQuantization::Q4_0,
            tmp_q4.path(),
            "model.gguf",
        )
        .unwrap();

        assert!(
            q4_result.export.size_bytes < f32_result.export.size_bytes,
            "Q4_0 ({}) should be smaller than F32 ({})",
            q4_result.export.size_bytes,
            f32_result.export.size_bytes
        );
    }
}
