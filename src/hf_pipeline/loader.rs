//! Model loading and teacher model abstraction
//!
//! Provides format-agnostic model loading with memory estimation.

use crate::hf_pipeline::error::{FetchError, Result};
use ndarray::Array2;
use std::path::Path;

/// Memory estimation for model loading
#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    /// Memory for model weights
    pub weights: u64,
    /// Memory for activations during forward pass
    pub activations: u64,
    /// Memory for gradients (0 for frozen teacher)
    pub gradients: u64,
}

impl MemoryEstimate {
    /// Total memory required
    #[must_use]
    pub fn total(&self) -> u64 {
        self.weights + self.activations + self.gradients
    }

    /// Check if model fits in available memory
    #[must_use]
    pub fn fits_in(&self, available: u64) -> bool {
        self.total() <= available
    }

    /// Create estimate for FP32 model
    #[must_use]
    pub fn fp32(param_count: u64, batch_size: usize, seq_len: usize, hidden_size: usize) -> Self {
        Self {
            weights: param_count * 4,
            activations: (batch_size * seq_len * hidden_size * 4) as u64,
            gradients: 0, // Frozen teacher
        }
    }

    /// Create estimate for FP16 model
    #[must_use]
    pub fn fp16(param_count: u64, batch_size: usize, seq_len: usize, hidden_size: usize) -> Self {
        Self {
            weights: param_count * 2,
            activations: (batch_size * seq_len * hidden_size * 2) as u64,
            gradients: 0,
        }
    }

    /// Create estimate for INT4/Q4 model
    #[must_use]
    pub fn int4(param_count: u64, batch_size: usize, seq_len: usize, hidden_size: usize) -> Self {
        Self {
            weights: param_count / 2, // 4-bit = 0.5 bytes per param
            // Activations still in FP16 for compute
            activations: (batch_size * seq_len * hidden_size * 2) as u64,
            gradients: 0,
        }
    }
}

/// Teacher model trait for distillation
///
/// Provides interface for frozen teacher models used in knowledge distillation.
pub trait TeacherModel: Send + Sync {
    /// Run forward pass, returning output logits
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [batch_size, seq_len, hidden_size]
    ///
    /// # Returns
    ///
    /// Output logits [batch_size, seq_len, vocab_size]
    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>>;

    /// Get intermediate hidden states for progressive distillation
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    ///
    /// # Returns
    ///
    /// Hidden states for each layer
    fn hidden_states(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>>;

    /// Get attention weights for attention transfer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    ///
    /// # Returns
    ///
    /// Attention weights [batch, heads, seq, seq] for each layer
    fn attention_weights(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>>;

    /// Estimate memory requirements
    fn estimate_memory(&self, batch_size: usize, seq_len: usize) -> MemoryEstimate;

    /// Get number of parameters
    fn param_count(&self) -> u64;

    /// Get number of layers
    fn num_layers(&self) -> usize;

    /// Get hidden size
    fn hidden_size(&self) -> usize;
}

/// SafeTensors-based teacher model
pub struct SafeTensorsTeacher {
    /// Model weights by tensor name
    weights: std::collections::HashMap<String, Array2<f32>>,
    /// Tensor names (in order)
    tensor_names: Vec<String>,
    /// Number of layers
    num_layers: usize,
    /// Hidden dimension
    hidden_size: usize,
    /// Total parameter count
    param_count: u64,
}

impl SafeTensorsTeacher {
    /// Load model from SafeTensors file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to model directory containing model.safetensors
    ///
    /// # Errors
    ///
    /// Returns error if file not found or parsing fails.
    pub fn load(path: &Path) -> Result<Self> {
        use safetensors::SafeTensors;

        let model_path = path.join("model.safetensors");
        if !model_path.exists() {
            return Err(FetchError::FileNotFound {
                repo: path.display().to_string(),
                file: "model.safetensors".into(),
            });
        }

        // Read the file into memory (safe approach for models up to ~10GB)
        let data = std::fs::read(&model_path)?;

        // Parse SafeTensors
        let tensors =
            SafeTensors::deserialize(&data).map_err(|e| FetchError::SafeTensorsParseError {
                message: e.to_string(),
            })?;

        // Extract tensor names and compute statistics
        let tensor_names: Vec<String> = tensors.names().iter().map(|s| (*s).to_string()).collect();

        // Calculate total parameter count
        let mut param_count: u64 = 0;
        for name in &tensor_names {
            if let Ok(info) = tensors.tensor(name) {
                let numel: u64 = info.shape().iter().map(|&x| x as u64).product();
                param_count += numel;
            }
        }

        // Detect number of layers from tensor naming convention
        // Common patterns: "encoder.layer.N.", "layers.N.", "h.N."
        let num_layers = detect_layer_count(&tensor_names);

        // Detect hidden size from weight tensor shapes
        let hidden_size = detect_hidden_size(&tensors, &tensor_names);

        Ok(Self {
            weights: std::collections::HashMap::new(), // Lazy load on demand
            tensor_names,
            num_layers,
            hidden_size,
            param_count,
        })
    }

    /// Get list of tensor names in the model
    #[must_use]
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    /// Get model weights by tensor name
    ///
    /// Note: Currently returns an empty map as weights are loaded on-demand
    /// for memory efficiency. Future versions will support lazy loading.
    #[must_use]
    pub fn weights(&self) -> &std::collections::HashMap<String, Array2<f32>> {
        &self.weights
    }

    /// Create mock teacher for testing
    #[cfg(test)]
    pub fn mock(num_layers: usize, hidden_size: usize) -> Self {
        let param_count = (num_layers as u64) * (hidden_size as u64).pow(2) * 4;
        Self {
            weights: std::collections::HashMap::new(),
            tensor_names: Vec::new(),
            num_layers,
            hidden_size,
            param_count,
        }
    }
}

/// Detect number of layers from tensor naming patterns
fn detect_layer_count(names: &[String]) -> usize {
    use std::collections::HashSet;

    let mut layer_indices: HashSet<usize> = HashSet::new();

    for name in names {
        // Match patterns like "encoder.layer.0.", "layers.0.", "h.0."
        if let Some(idx) = extract_layer_index(name) {
            layer_indices.insert(idx);
        }
    }

    if layer_indices.is_empty() {
        // Default to 12 if can't detect (BERT-base assumption)
        12
    } else {
        layer_indices.len()
    }
}

/// Extract layer index from tensor name
fn extract_layer_index(name: &str) -> Option<usize> {
    // Common patterns for layer indices
    let patterns = [".layer.", ".layers.", ".h."];

    for pattern in patterns {
        if let Some(pos) = name.find(pattern) {
            let after_pattern = &name[pos + pattern.len()..];
            if let Some(end) = after_pattern.find('.') {
                if let Ok(idx) = after_pattern[..end].parse::<usize>() {
                    return Some(idx);
                }
            } else if let Ok(idx) = after_pattern.parse::<usize>() {
                return Some(idx);
            }
        }
    }

    None
}

/// Detect hidden size from tensor shapes
fn detect_hidden_size(tensors: &safetensors::SafeTensors<'_>, names: &[String]) -> usize {
    // Look for attention query weight which is typically [hidden_size, hidden_size]
    let query_patterns = [
        ".query.weight",
        ".q_proj.weight",
        ".self_attn.q_proj.weight",
    ];

    for name in names {
        for pattern in query_patterns {
            if name.ends_with(pattern) {
                if let Ok(tensor) = tensors.tensor(name) {
                    let shape = tensor.shape();
                    if shape.len() == 2 && shape[0] == shape[1] {
                        return shape[0];
                    }
                }
            }
        }
    }

    // Fallback: look for any large square weight matrix
    for name in names {
        if name.contains("weight") {
            if let Ok(tensor) = tensors.tensor(name) {
                let shape = tensor.shape();
                if shape.len() == 2 && shape[0] == shape[1] && shape[0] >= 256 {
                    return shape[0];
                }
            }
        }
    }

    // Default to 768 (BERT-base)
    768
}

impl TeacherModel for SafeTensorsTeacher {
    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // Mock implementation - just pass through
        Ok(input.clone())
    }

    fn hidden_states(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        // Return one hidden state per layer
        Ok(vec![input.clone(); self.num_layers])
    }

    fn attention_weights(&self, input: &Array2<f32>) -> Result<Vec<Array2<f32>>> {
        // Return attention weights per layer
        let (batch, _seq) = input.dim();
        let attn = Array2::<f32>::ones((batch, batch));
        Ok(vec![attn; self.num_layers])
    }

    fn estimate_memory(&self, batch_size: usize, seq_len: usize) -> MemoryEstimate {
        MemoryEstimate::fp16(self.param_count, batch_size, seq_len, self.hidden_size)
    }

    fn param_count(&self) -> u64 {
        self.param_count
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // =========================================================================
    // MemoryEstimate Tests
    // =========================================================================

    #[test]
    fn test_memory_estimate_total() {
        let est = MemoryEstimate {
            weights: 100,
            activations: 50,
            gradients: 25,
        };
        assert_eq!(est.total(), 175);
    }

    #[test]
    fn test_memory_estimate_fits_in() {
        let est = MemoryEstimate {
            weights: 100,
            activations: 50,
            gradients: 0,
        };
        assert!(est.fits_in(200));
        assert!(est.fits_in(150));
        assert!(!est.fits_in(100));
    }

    #[test]
    fn test_memory_estimate_fp32() {
        // 125M params in FP32 = 500MB
        let est = MemoryEstimate::fp32(125_000_000, 1, 512, 768);
        assert_eq!(est.weights, 500_000_000);
        assert!(est.activations > 0);
        assert_eq!(est.gradients, 0); // Frozen teacher
    }

    #[test]
    fn test_memory_estimate_fp16() {
        // 125M params in FP16 = 250MB
        let est = MemoryEstimate::fp16(125_000_000, 1, 512, 768);
        assert_eq!(est.weights, 250_000_000);
    }

    #[test]
    fn test_memory_estimate_int4() {
        // 125M params in INT4 = ~62.5MB
        let est = MemoryEstimate::int4(125_000_000, 1, 512, 768);
        assert_eq!(est.weights, 62_500_000);
    }

    #[test]
    fn test_codebert_memory() {
        // CodeBERT: 125M params
        let est = MemoryEstimate::fp16(125_000_000, 32, 512, 768);
        // Should fit in 8GB GPU
        assert!(est.fits_in(8 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_llama_7b_memory() {
        // Llama-7B: 7B params
        let est = MemoryEstimate::fp16(7_000_000_000, 1, 2048, 4096);
        // Needs ~14GB for weights alone
        assert!(est.weights > 10 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_llama_7b_int4_memory() {
        // Llama-7B quantized: ~3.5GB
        let est = MemoryEstimate::int4(7_000_000_000, 1, 2048, 4096);
        assert!(est.weights < 5 * 1024 * 1024 * 1024);
    }

    // =========================================================================
    // SafeTensorsTeacher Tests
    // =========================================================================

    #[test]
    fn test_mock_teacher_creation() {
        let teacher = SafeTensorsTeacher::mock(12, 768);
        assert_eq!(teacher.num_layers(), 12);
        assert_eq!(teacher.hidden_size(), 768);
        assert!(teacher.param_count() > 0);
    }

    #[test]
    fn test_teacher_forward() {
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let input = Array2::<f32>::zeros((4, 768));
        let output = teacher.forward(&input);
        assert!(output.is_ok());
        assert_eq!(output.unwrap().dim(), (4, 768));
    }

    #[test]
    fn test_teacher_hidden_states() {
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let input = Array2::<f32>::zeros((4, 768));
        let hidden = teacher.hidden_states(&input);
        assert!(hidden.is_ok());
        let hidden = hidden.unwrap();
        assert_eq!(hidden.len(), 12); // One per layer
    }

    #[test]
    fn test_teacher_attention_weights() {
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let input = Array2::<f32>::zeros((4, 768));
        let attn = teacher.attention_weights(&input);
        assert!(attn.is_ok());
        let attn = attn.unwrap();
        assert_eq!(attn.len(), 12);
    }

    #[test]
    fn test_teacher_memory_estimate() {
        let teacher = SafeTensorsTeacher::mock(12, 768);
        let est = teacher.estimate_memory(32, 512);
        assert!(est.weights > 0);
        assert!(est.activations > 0);
        assert_eq!(est.gradients, 0);
    }

    #[test]
    fn test_load_nonexistent() {
        let result = SafeTensorsTeacher::load(Path::new("/nonexistent/path"));
        assert!(matches!(result, Err(FetchError::FileNotFound { .. })));
    }

    // =========================================================================
    // SafeTensors Parsing Tests (TDD - these define expected behavior)
    // =========================================================================

    #[test]
    fn test_load_valid_safetensors_file() {
        use tempfile::TempDir;

        // Create a minimal valid safetensors file
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.safetensors");

        // Create minimal safetensors with one tensor
        let data = create_test_safetensors(&[("weight", &[2, 3])]);
        std::fs::write(&model_path, data).unwrap();

        let teacher = SafeTensorsTeacher::load(temp_dir.path());
        assert!(teacher.is_ok(), "Should load valid safetensors file");

        let teacher = teacher.unwrap();
        assert!(teacher.param_count() > 0, "Should have non-zero params");
    }

    #[test]
    fn test_safetensors_extracts_tensor_names() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.safetensors");

        // Create safetensors with named tensors
        let data = create_test_safetensors(&[
            ("encoder.layer.0.attention.query.weight", &[768, 768]),
            ("encoder.layer.0.attention.key.weight", &[768, 768]),
        ]);
        std::fs::write(&model_path, data).unwrap();

        let teacher = SafeTensorsTeacher::load(temp_dir.path()).unwrap();
        assert!(teacher
            .tensor_names()
            .contains(&"encoder.layer.0.attention.query.weight".to_string()));
    }

    #[test]
    fn test_safetensors_param_count_matches_tensors() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.safetensors");

        // Create safetensors with known parameter count
        // 2 tensors of 768x768 = 2 * 589,824 = 1,179,648 params
        let data = create_test_safetensors(&[
            ("layer.0.weight", &[768, 768]),
            ("layer.1.weight", &[768, 768]),
        ]);
        std::fs::write(&model_path, data).unwrap();

        let teacher = SafeTensorsTeacher::load(temp_dir.path()).unwrap();
        assert_eq!(teacher.param_count(), 768 * 768 * 2);
    }

    #[test]
    fn test_safetensors_detects_layer_count() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.safetensors");

        // Create safetensors with 12 layers
        let mut tensors: Vec<(&str, &[usize])> = Vec::new();
        let layer_names: Vec<String> = (0..12)
            .map(|i| format!("encoder.layer.{}.attention.weight", i))
            .collect();

        for name in &layer_names {
            tensors.push((name, &[768, 768]));
        }

        let data = create_test_safetensors_from_names(&tensors);
        std::fs::write(&model_path, data).unwrap();

        let teacher = SafeTensorsTeacher::load(temp_dir.path()).unwrap();
        assert_eq!(teacher.num_layers(), 12);
    }

    #[test]
    fn test_safetensors_detects_hidden_size() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.safetensors");

        // Create safetensors with 1024 hidden size
        let data =
            create_test_safetensors(&[("encoder.layer.0.attention.query.weight", &[1024, 1024])]);
        std::fs::write(&model_path, data).unwrap();

        let teacher = SafeTensorsTeacher::load(temp_dir.path()).unwrap();
        assert_eq!(teacher.hidden_size(), 1024);
    }

    #[test]
    fn test_safetensors_corrupt_file_error() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.safetensors");

        // Write garbage data
        std::fs::write(&model_path, b"not a valid safetensors file").unwrap();

        let result = SafeTensorsTeacher::load(temp_dir.path());
        assert!(result.is_err(), "Should fail on corrupt file");
    }

    // Helper function to create minimal safetensors for testing
    fn create_test_safetensors(tensors: &[(&str, &[usize])]) -> Vec<u8> {
        use safetensors::tensor::{Dtype, TensorView};

        let tensor_data: Vec<(String, Vec<f32>, Vec<usize>)> = tensors
            .iter()
            .map(|(name, shape)| {
                let numel: usize = shape.iter().product();
                ((*name).to_string(), vec![0.0f32; numel], shape.to_vec())
            })
            .collect();

        let views: Vec<(&str, TensorView<'_>)> = tensor_data
            .iter()
            .map(|(name, data, shape)| {
                let view =
                    TensorView::new(Dtype::F32, shape.clone(), bytemuck::cast_slice(data)).unwrap();
                (name.as_str(), view)
            })
            .collect();

        safetensors::serialize(views, None::<std::collections::HashMap<String, String>>).unwrap()
    }

    fn create_test_safetensors_from_names(tensors: &[(&str, &[usize])]) -> Vec<u8> {
        create_test_safetensors(tensors)
    }
}
