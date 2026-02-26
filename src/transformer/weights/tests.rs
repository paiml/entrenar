//! Tests for weight loading module

use super::*;
use std::collections::HashMap;
use std::path::Path;
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
    weights.insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; hidden], true));

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
    assert!(result.unwrap_err().to_string().contains("embed_tokens.weight"));
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
    weights
        .insert("model.embed_tokens.weight".to_string(), Tensor::from_vec(vec![0.1; 64000], true));

    let result = validate_weights(&weights, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("norm.weight"));
}

#[test]
fn test_validate_weights_missing_layer_weight() {
    let mut weights = HashMap::new();
    weights
        .insert("model.embed_tokens.weight".to_string(), Tensor::from_vec(vec![0.1; 64000], true));
    weights.insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; 64], true));
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
    weights.insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; hidden], true));
    weights.insert("lm_head.weight".to_string(), Tensor::from_vec(vec![0.1; vocab * hidden], true));

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
    std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"part1").unwrap();
    std::fs::write(dir.path().join("model-00002-of-00002.safetensors"), b"part2").unwrap();

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
    assert!(result.unwrap_err().to_string().contains("No SafeTensors files found"));
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

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
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

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
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

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
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

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
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

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
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

    let bias_bytes: Vec<u8> = vec![0.0f32; 4].iter().flat_map(|f| f.to_le_bytes()).collect();

    // Qwen2 has attention biases
    let view1 = TensorView::new(Dtype::F32, vec![4], &bias_bytes).unwrap();
    let view2 = TensorView::new(Dtype::F32, vec![4], &bias_bytes).unwrap();
    let data = vec![
        ("model.layers.0.self_attn.q_proj.bias", &view1),
        ("model.layers.0.self_attn.k_proj.bias", &view2),
    ];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
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

    let weight_bytes: Vec<u8> = vec![0.1f32; 4].iter().flat_map(|f| f.to_le_bytes()).collect();

    // LLaMA has no attention biases
    let view = TensorView::new(Dtype::F32, vec![2, 2], &weight_bytes).unwrap();
    let data = vec![("model.layers.0.self_attn.q_proj.weight", &view)];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
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

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>).unwrap();
    std::fs::write(&file_path, serialized).unwrap();

    // Should succeed but skip the unsupported tensor
    let result = load_safetensors_weights(&file_path, Architecture::Llama);
    assert!(result.is_ok());
    let weights = result.unwrap();
    // Unsupported dtype tensors are skipped
    assert!(!weights.contains_key("unsupported_tensor"));
}
