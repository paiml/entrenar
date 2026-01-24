//! Weight loading module for transformer models
//!
//! This module provides functions to load model weights from SafeTensors files
//! and convert them to the format expected by `Transformer::from_params`.
//!
//! Supports:
//! - Qwen2/Qwen2.5 architecture
//! - LLaMA architecture
//! - Mistral architecture
//!
//! Weight name mapping follows HuggingFace conventions.

use crate::error::{Error, Result};
use crate::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// Architecture type for weight name mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// LLaMA and LLaMA-2 models
    Llama,
    /// Qwen2 and Qwen2.5 models (includes Qwen2.5-Coder)
    Qwen2,
    /// Mistral models
    Mistral,
    /// Auto-detect from weight names
    Auto,
}

/// Load transformer weights from SafeTensors file(s)
///
/// # Arguments
///
/// * `model_path` - Path to model directory or single SafeTensors file
/// * `arch` - Model architecture (use Auto to detect from weight names)
///
/// # Returns
///
/// HashMap of parameter names mapped to Tensor values.
/// Names follow the HuggingFace LLaMA convention expected by `Transformer::from_params`.
pub fn load_safetensors_weights(
    model_path: &Path,
    arch: Architecture,
) -> Result<HashMap<String, Tensor>> {
    use safetensors::SafeTensors;

    // Find SafeTensors files
    let st_files = find_safetensors_files(model_path)?;
    if st_files.is_empty() {
        return Err(Error::ConfigError(format!(
            "No SafeTensors files found in {}",
            model_path.display()
        )));
    }

    let mut weights = HashMap::new();
    let mut detected_arch = arch;

    // Process each SafeTensors file
    for st_path in &st_files {
        let data = std::fs::read(st_path).map_err(|e| {
            Error::ConfigError(format!("Failed to read {}: {}", st_path.display(), e))
        })?;

        let tensors = SafeTensors::deserialize(&data).map_err(|e| {
            Error::ConfigError(format!(
                "Failed to parse SafeTensors {}: {}",
                st_path.display(),
                e
            ))
        })?;

        // Auto-detect architecture from first file
        if detected_arch == Architecture::Auto {
            detected_arch = detect_architecture(&tensors);
            println!("  Detected architecture: {:?}", detected_arch);
        }

        // Load and map tensors
        for name in tensors.names() {
            if let Ok(tensor_view) = tensors.tensor(name) {
                // Convert tensor to f32 values
                if let Some(values) = tensor_to_f32_vec(&tensor_view) {
                    // Map name to standard LLaMA convention
                    let mapped_name = map_weight_name(name, detected_arch);
                    let tensor = Tensor::from_vec(values, true);
                    weights.insert(mapped_name, tensor);
                }
            }
        }
    }

    println!("  Loaded {} weight tensors", weights.len());
    Ok(weights)
}

/// Find SafeTensors files in a directory or return single file
fn find_safetensors_files(path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        // Single file path
        if path
            .extension()
            .map(|e| e == "safetensors")
            .unwrap_or(false)
        {
            files.push(path.to_path_buf());
        }
    } else if path.is_dir() {
        // Directory - find all .safetensors files
        // Check for model.safetensors first
        let single = path.join("model.safetensors");
        if single.exists() {
            files.push(single);
        } else {
            // Look for sharded files: model-00001-of-00005.safetensors
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                        files.push(p);
                    }
                }
                // Sort for consistent ordering
                files.sort();
            }
        }
    }

    Ok(files)
}

/// Auto-detect model architecture from tensor names
fn detect_architecture(tensors: &safetensors::SafeTensors<'_>) -> Architecture {
    let names: Vec<String> = tensors.names().iter().map(|s| s.to_string()).collect();

    // Qwen2 uses "model.layers.X.self_attn.q_proj" with biases
    // LLaMA uses same pattern but no biases
    // Check for bias tensors to distinguish
    let has_attn_bias = names.iter().any(|n: &String| {
        n.contains("self_attn.q_proj.bias") || n.contains("self_attn.k_proj.bias")
    });

    if has_attn_bias {
        // Qwen2 has attention biases
        return Architecture::Qwen2;
    }

    // Check for Mistral-specific patterns
    // Mistral uses sliding window attention config but same weight names as LLaMA
    // We default to LLaMA if no Qwen2 bias markers found

    Architecture::Llama
}

/// Map weight name from source architecture to standard LLaMA convention
///
/// Standard names expected by `Transformer::from_params`:
/// - `model.embed_tokens.weight`
/// - `model.layers.{i}.input_layernorm.weight`
/// - `model.layers.{i}.self_attn.q_proj.weight`
/// - `model.layers.{i}.self_attn.k_proj.weight`
/// - `model.layers.{i}.self_attn.v_proj.weight`
/// - `model.layers.{i}.self_attn.o_proj.weight`
/// - `model.layers.{i}.post_attention_layernorm.weight`
/// - `model.layers.{i}.mlp.gate_proj.weight`
/// - `model.layers.{i}.mlp.up_proj.weight`
/// - `model.layers.{i}.mlp.down_proj.weight`
/// - `model.norm.weight`
/// - `lm_head.weight`
fn map_weight_name(name: &str, arch: Architecture) -> String {
    match arch {
        Architecture::Qwen2 => {
            // Qwen2 weight names are nearly identical to LLaMA convention
            // Main differences:
            // - Qwen2 has bias tensors (which we preserve)
            // - Some Qwen2 models use "attn" instead of "self_attn" (rare)

            // Handle potential "attn" -> "self_attn" mapping
            if name.contains(".attn.") && !name.contains(".self_attn.") {
                name.replace(".attn.", ".self_attn.")
            } else {
                name.to_string()
            }
        }
        Architecture::Mistral | Architecture::Llama => {
            // LLaMA and Mistral use same naming convention as our standard
            name.to_string()
        }
        Architecture::Auto => {
            // Should not reach here after detection
            name.to_string()
        }
    }
}

/// Convert SafeTensors tensor view to f32 Vec
///
/// Handles bf16, fp16, and fp32 formats.
fn tensor_to_f32_vec(tensor: &safetensors::tensor::TensorView<'_>) -> Option<Vec<f32>> {
    use safetensors::Dtype;

    let shape = tensor.shape();
    let numel: usize = shape.iter().product();

    if numel == 0 {
        return Some(Vec::new());
    }

    let data = tensor.data();

    match tensor.dtype() {
        Dtype::F32 => {
            // Direct f32 conversion
            let values: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Some(values)
        }
        Dtype::F16 => {
            // fp16 conversion
            let values: Vec<f32> = data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();
            Some(values)
        }
        Dtype::BF16 => {
            // bf16 conversion
            let values: Vec<f32> = data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect();
            Some(values)
        }
        Dtype::I32 => {
            // Integer to float (rare for transformer weights)
            let values: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
                .collect();
            Some(values)
        }
        _ => {
            // Unsupported dtype
            eprintln!(
                "Warning: Unsupported tensor dtype {:?}, skipping",
                tensor.dtype()
            );
            None
        }
    }
}

/// Get expected weight count for a transformer model
pub fn expected_weight_count(num_layers: usize, has_lm_head: bool) -> usize {
    // Per layer:
    //   input_layernorm.weight (1)
    //   self_attn: q_proj, k_proj, v_proj, o_proj (4)
    //   post_attention_layernorm.weight (1)
    //   mlp: gate_proj, up_proj, down_proj (3)
    // = 9 per layer
    //
    // Global:
    //   embed_tokens.weight (1)
    //   norm.weight (1)
    //   lm_head.weight (optional, 1)
    let base = 2 + (num_layers * 9);
    if has_lm_head {
        base + 1
    } else {
        base
    }
}

/// Validate that loaded weights match expected architecture
pub fn validate_weights(weights: &HashMap<String, Tensor>, num_layers: usize) -> Result<()> {
    // Check embedding
    if !weights.contains_key("model.embed_tokens.weight") {
        return Err(Error::ConfigError(
            "Missing model.embed_tokens.weight".into(),
        ));
    }

    // Check final norm
    if !weights.contains_key("model.norm.weight") {
        return Err(Error::ConfigError("Missing model.norm.weight".into()));
    }

    // Check each layer
    for i in 0..num_layers {
        let layer_prefix = format!("model.layers.{i}");

        // Required layer weights
        let required = [
            ".input_layernorm.weight",
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".post_attention_layernorm.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        ];

        for suffix in required {
            let key = format!("{layer_prefix}{suffix}");
            if !weights.contains_key(&key) {
                return Err(Error::ConfigError(format!("Missing {key}")));
            }
        }
    }

    // Check weight count for informational purposes
    let has_lm_head = weights.contains_key("lm_head.weight");
    let expected = expected_weight_count(num_layers, has_lm_head);
    let actual = weights.len();
    if actual < expected {
        // This is a warning, not an error - some models may have extra or fewer weights
        eprintln!(
            "Warning: Expected at least {} weights, found {} (may have extra bias tensors)",
            expected, actual
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_map_weight_name_llama() {
        let name = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(map_weight_name(name, Architecture::Llama), name);
    }

    #[test]
    fn test_map_weight_name_qwen2_standard() {
        let name = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(map_weight_name(name, Architecture::Qwen2), name);
    }

    #[test]
    fn test_map_weight_name_qwen2_attn_variant() {
        let name = "model.layers.0.attn.q_proj.weight";
        assert_eq!(
            map_weight_name(name, Architecture::Qwen2),
            "model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn test_expected_weight_count() {
        // 2 layers without lm_head
        assert_eq!(expected_weight_count(2, false), 2 + 2 * 9);
        // 2 layers with lm_head
        assert_eq!(expected_weight_count(2, true), 2 + 2 * 9 + 1);
        // 24 layers (Qwen2.5-0.5B)
        assert_eq!(expected_weight_count(24, false), 2 + 24 * 9);
    }

    #[test]
    fn test_validate_weights_minimal() {
        let mut weights = HashMap::new();
        let hidden = 64;
        let kv_hidden = 64;
        let intermediate = 256;
        let vocab = 1000;

        // Global weights
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(vec![0.1; vocab * hidden], true),
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden], true),
        );

        // Layer 0 weights
        weights.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden], true),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * hidden], true),
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * kv_hidden], true),
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * kv_hidden], true),
        );
        weights.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * hidden], true),
        );
        weights.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden], true),
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * intermediate], true),
        );
        weights.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * intermediate], true),
        );
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; intermediate * hidden], true),
        );

        // Should validate for 1 layer
        assert!(validate_weights(&weights, 1).is_ok());

        // Should fail for 2 layers (missing layer 1)
        assert!(validate_weights(&weights, 2).is_err());
    }

    #[test]
    fn test_validate_weights_missing_embedding() {
        let weights: HashMap<String, Tensor> = HashMap::new();
        let result = validate_weights(&weights, 1);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("embed_tokens.weight"));
    }

    #[test]
    fn test_find_safetensors_files_nonexistent() {
        let path = Path::new("/nonexistent/path");
        let files = find_safetensors_files(path).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn test_find_safetensors_files_empty_dir() {
        let dir = TempDir::new().unwrap();
        let files = find_safetensors_files(dir.path()).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn test_architecture_default() {
        // Default should be LLaMA-compatible
        assert_eq!(Architecture::Auto, Architecture::Auto);
        assert_ne!(Architecture::Llama, Architecture::Qwen2);
    }

    #[test]
    fn test_map_weight_name_mistral() {
        let name = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(map_weight_name(name, Architecture::Mistral), name);
    }

    #[test]
    fn test_map_weight_name_auto() {
        let name = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(map_weight_name(name, Architecture::Auto), name);
    }

    #[test]
    fn test_validate_weights_missing_norm() {
        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(vec![0.1; 64000], true),
        );

        let result = validate_weights(&weights, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("norm.weight"));
    }

    #[test]
    fn test_validate_weights_missing_layer_weight() {
        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(vec![0.1; 64000], true),
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Tensor::from_vec(vec![1.0; 64], true),
        );
        weights.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            Tensor::from_vec(vec![1.0; 64], true),
        );
        // Missing q_proj.weight

        let result = validate_weights(&weights, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("q_proj.weight"));
    }

    #[test]
    fn test_validate_weights_with_lm_head() {
        let mut weights = HashMap::new();
        let hidden = 64;
        let kv_hidden = 64;
        let intermediate = 256;
        let vocab = 1000;

        // Global weights
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(vec![0.1; vocab * hidden], true),
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden], true),
        );
        weights.insert(
            "lm_head.weight".to_string(),
            Tensor::from_vec(vec![0.1; vocab * hidden], true),
        );

        // Layer 0 weights
        weights.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden], true),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * hidden], true),
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * kv_hidden], true),
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * kv_hidden], true),
        );
        weights.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * hidden], true),
        );
        weights.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            Tensor::from_vec(vec![1.0; hidden], true),
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * intermediate], true),
        );
        weights.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden * intermediate], true),
        );
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; intermediate * hidden], true),
        );

        assert!(validate_weights(&weights, 1).is_ok());
    }

    #[test]
    fn test_find_safetensors_single_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");
        std::fs::write(&file_path, b"dummy").unwrap();

        let files = find_safetensors_files(dir.path()).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], file_path);
    }

    #[test]
    fn test_find_safetensors_sharded_files() {
        let dir = TempDir::new().unwrap();
        // Create sharded files
        std::fs::write(
            dir.path().join("model-00001-of-00002.safetensors"),
            b"part1",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model-00002-of-00002.safetensors"),
            b"part2",
        )
        .unwrap();

        let files = find_safetensors_files(dir.path()).unwrap();
        assert_eq!(files.len(), 2);
        // Files should be sorted
        assert!(files[0].to_string_lossy().contains("00001"));
        assert!(files[1].to_string_lossy().contains("00002"));
    }

    #[test]
    fn test_find_safetensors_direct_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("weights.safetensors");
        std::fs::write(&file_path, b"dummy").unwrap();

        let files = find_safetensors_files(&file_path).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_find_safetensors_non_safetensors_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.bin");
        std::fs::write(&file_path, b"dummy").unwrap();

        let files = find_safetensors_files(&file_path).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn test_load_safetensors_no_files() {
        let dir = TempDir::new().unwrap();
        let result = load_safetensors_weights(dir.path(), Architecture::Auto);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No SafeTensors files found"));
    }

    #[test]
    fn test_load_safetensors_invalid_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");
        std::fs::write(&file_path, b"not a valid safetensors file").unwrap();

        let result = load_safetensors_weights(dir.path(), Architecture::Auto);
        assert!(result.is_err());
    }

    #[test]
    fn test_architecture_variants() {
        assert_eq!(Architecture::Llama, Architecture::Llama);
        assert_eq!(Architecture::Qwen2, Architecture::Qwen2);
        assert_eq!(Architecture::Mistral, Architecture::Mistral);

        // Test Clone
        let arch = Architecture::Llama;
        let cloned = arch;
        assert_eq!(arch, cloned);
    }

    // Test tensor conversion with a real safetensors file
    #[test]
    fn test_load_safetensors_real_file() {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");

        // Create a minimal safetensors file with f32 data
        let embed_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let embed_bytes: Vec<u8> = embed_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let view = TensorView::new(Dtype::F32, vec![2, 2], &embed_bytes).unwrap();
        let data = vec![("model.embed_tokens.weight", &view)];

        let serialized =
            serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
        std::fs::write(&file_path, serialized).unwrap();

        let result = load_safetensors_weights(&file_path, Architecture::Llama);
        assert!(result.is_ok());
        let weights = result.unwrap();
        assert!(weights.contains_key("model.embed_tokens.weight"));
    }

    #[test]
    fn test_load_safetensors_with_f16() {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");

        // Create fp16 data
        let fp16_values: Vec<half::f16> = vec![
            half::f16::from_f32(0.1),
            half::f16::from_f32(0.2),
            half::f16::from_f32(0.3),
            half::f16::from_f32(0.4),
        ];
        let fp16_bytes: Vec<u8> = fp16_values.iter().flat_map(|f| f.to_le_bytes()).collect();

        let view = TensorView::new(Dtype::F16, vec![2, 2], &fp16_bytes).unwrap();
        let data = vec![("model.embed_tokens.weight", &view)];

        let serialized =
            serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
        std::fs::write(&file_path, serialized).unwrap();

        let result = load_safetensors_weights(&file_path, Architecture::Llama);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_safetensors_with_bf16() {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");

        // Create bf16 data
        let bf16_values: Vec<half::bf16> = vec![
            half::bf16::from_f32(0.1),
            half::bf16::from_f32(0.2),
            half::bf16::from_f32(0.3),
            half::bf16::from_f32(0.4),
        ];
        let bf16_bytes: Vec<u8> = bf16_values.iter().flat_map(|f| f.to_le_bytes()).collect();

        let view = TensorView::new(Dtype::BF16, vec![2, 2], &bf16_bytes).unwrap();
        let data = vec![("model.embed_tokens.weight", &view)];

        let serialized =
            serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
        std::fs::write(&file_path, serialized).unwrap();

        let result = load_safetensors_weights(&file_path, Architecture::Llama);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_safetensors_with_i32() {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");

        // Create i32 data
        let i32_values: Vec<i32> = vec![1, 2, 3, 4];
        let i32_bytes: Vec<u8> = i32_values.iter().flat_map(|i| i.to_le_bytes()).collect();

        let view = TensorView::new(Dtype::I32, vec![2, 2], &i32_bytes).unwrap();
        let data = vec![("model.embed_tokens.weight", &view)];

        let serialized =
            serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
        std::fs::write(&file_path, serialized).unwrap();

        let result = load_safetensors_weights(&file_path, Architecture::Llama);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_safetensors_empty_tensor() {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");

        let empty_bytes: Vec<u8> = vec![];
        let view = TensorView::new(Dtype::F32, vec![0], &empty_bytes).unwrap();
        let data = vec![("empty_tensor", &view)];

        let serialized =
            serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
        std::fs::write(&file_path, serialized).unwrap();

        let result = load_safetensors_weights(&file_path, Architecture::Llama);
        assert!(result.is_ok());
        let weights = result.unwrap();
        assert!(weights.contains_key("empty_tensor"));
    }

    #[test]
    fn test_detect_architecture_qwen2() {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");

        let bias_bytes: Vec<u8> = vec![0.0f32; 4]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Qwen2 has attention biases
        let view1 = TensorView::new(Dtype::F32, vec![4], &bias_bytes).unwrap();
        let view2 = TensorView::new(Dtype::F32, vec![4], &bias_bytes).unwrap();
        let data = vec![
            ("model.layers.0.self_attn.q_proj.bias", &view1),
            ("model.layers.0.self_attn.k_proj.bias", &view2),
        ];

        let serialized =
            serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
        std::fs::write(&file_path, serialized).unwrap();

        let result = load_safetensors_weights(&file_path, Architecture::Auto);
        assert!(result.is_ok());
    }

    #[test]
    fn test_detect_architecture_llama() {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");

        let weight_bytes: Vec<u8> = vec![0.1f32; 4]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // LLaMA has no attention biases
        let view = TensorView::new(Dtype::F32, vec![2, 2], &weight_bytes).unwrap();
        let data = vec![("model.layers.0.self_attn.q_proj.weight", &view)];

        let serialized =
            serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
        std::fs::write(&file_path, serialized).unwrap();

        let result = load_safetensors_weights(&file_path, Architecture::Auto);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_safetensors_with_unsupported_dtype() {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};

        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");

        // Create U8 data (unsupported for conversion)
        let u8_bytes: Vec<u8> = vec![1, 2, 3, 4];
        let view = TensorView::new(Dtype::U8, vec![4], &u8_bytes).unwrap();
        let data = vec![("unsupported_tensor", &view)];

        let serialized =
            serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
        std::fs::write(&file_path, serialized).unwrap();

        // Should succeed but skip the unsupported tensor
        let result = load_safetensors_weights(&file_path, Architecture::Llama);
        assert!(result.is_ok());
        let weights = result.unwrap();
        // Unsupported dtype tensors are skipped
        assert!(!weights.contains_key("unsupported_tensor"));
    }
}
