//! SafeTensors-based teacher model implementation

use crate::hf_pipeline::error::{FetchError, Result};
use ndarray::Array2;
use std::path::Path;

use super::{MemoryEstimate, TeacherModel};

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
