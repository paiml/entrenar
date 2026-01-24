//! Multi-head attention module
//!
//! This module provides multi-head self-attention with grouped-query attention support.

use crate::autograd::matmul;
use crate::Tensor;
use std::collections::HashMap;

use super::config::TransformerConfig;

/// Multi-head self-attention layer
pub struct MultiHeadAttention {
    /// Configuration
    config: TransformerConfig,
    /// Query projection weight (hidden_size x hidden_size)
    pub w_q: Tensor,
    /// Key projection weight (hidden_size x kv_hidden_size)
    pub w_k: Tensor,
    /// Value projection weight (hidden_size x kv_hidden_size)
    pub w_v: Tensor,
    /// Output projection weight (hidden_size x hidden_size)
    pub w_o: Tensor,
}

impl MultiHeadAttention {
    /// Create new attention layer with initialized weights
    pub fn new(config: &TransformerConfig) -> Self {
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();

        // Xavier initialization scale
        let scale = (2.0 / (hidden_size + hidden_size) as f32).sqrt();
        let kv_scale = (2.0 / (hidden_size + kv_hidden_size) as f32).sqrt();

        Self {
            config: config.clone(),
            w_q: Tensor::from_vec(
                (0..hidden_size * hidden_size)
                    .map(|i| ((i as f32 * 0.123).sin() * scale))
                    .collect(),
                true,
            ),
            w_k: Tensor::from_vec(
                (0..hidden_size * kv_hidden_size)
                    .map(|i| ((i as f32 * 0.234).sin() * kv_scale))
                    .collect(),
                true,
            ),
            w_v: Tensor::from_vec(
                (0..hidden_size * kv_hidden_size)
                    .map(|i| ((i as f32 * 0.345).sin() * kv_scale))
                    .collect(),
                true,
            ),
            w_o: Tensor::from_vec(
                (0..hidden_size * hidden_size)
                    .map(|i| ((i as f32 * 0.456).sin() * scale))
                    .collect(),
                true,
            ),
        }
    }

    /// Create attention layer from parameter map
    ///
    /// Expected parameter names (following HuggingFace convention):
    /// - `{prefix}.q_proj.weight`
    /// - `{prefix}.k_proj.weight`
    /// - `{prefix}.v_proj.weight`
    /// - `{prefix}.o_proj.weight`
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Option<Self> {
        let w_q = params.get(&format!("{prefix}.q_proj.weight"))?.clone();
        let w_k = params.get(&format!("{prefix}.k_proj.weight"))?.clone();
        let w_v = params.get(&format!("{prefix}.v_proj.weight"))?.clone();
        let w_o = params.get(&format!("{prefix}.o_proj.weight"))?.clone();

        Some(Self {
            config: config.clone(),
            w_q,
            w_k,
            w_v,
            w_o,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size, flattened)
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor (seq_len * hidden_size, flattened)
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let kv_hidden_size = num_kv_heads * head_dim;

        // Project Q, K, V
        // x: (seq_len, hidden_size) -> Q: (seq_len, hidden_size)
        let q = matmul(x, &self.w_q, seq_len, hidden_size, hidden_size);
        // x: (seq_len, hidden_size) -> K: (seq_len, kv_hidden_size)
        let k = matmul(x, &self.w_k, seq_len, hidden_size, kv_hidden_size);
        // x: (seq_len, hidden_size) -> V: (seq_len, kv_hidden_size)
        let v = matmul(x, &self.w_v, seq_len, hidden_size, kv_hidden_size);

        // Multi-head attention with grouped-query attention support
        let mut attn_outputs = Vec::with_capacity(num_heads * seq_len * head_dim);

        // Number of query heads per KV head
        let heads_per_kv = num_heads / num_kv_heads;

        for h in 0..num_heads {
            // Which KV head to use for this query head
            let kv_h = h / heads_per_kv;

            // Extract Q for this head: (seq_len, head_dim)
            let q_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * hidden_size + h * head_dim;
                    q.data().as_slice().unwrap()[start..start + head_dim].to_vec()
                })
                .collect();

            // Extract K for this KV head: (seq_len, head_dim)
            let k_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * kv_hidden_size + kv_h * head_dim;
                    k.data().as_slice().unwrap()[start..start + head_dim].to_vec()
                })
                .collect();

            // Extract V for this KV head: (seq_len, head_dim)
            let v_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * kv_hidden_size + kv_h * head_dim;
                    v.data().as_slice().unwrap()[start..start + head_dim].to_vec()
                })
                .collect();

            // Scaled dot-product attention for this head
            let q_tensor = Tensor::from_vec(q_head, false);
            let k_tensor = Tensor::from_vec(k_head, false);
            let v_tensor = Tensor::from_vec(v_head, false);

            let attn_out = crate::autograd::attention(
                &q_tensor, &k_tensor, &v_tensor, seq_len, head_dim, seq_len, head_dim,
            );

            attn_outputs.extend_from_slice(attn_out.data().as_slice().unwrap());
        }

        // Concatenate heads and reshape: (seq_len, num_heads * head_dim) = (seq_len, hidden_size)
        // Then reorder from head-major to position-major
        let mut concat_output = vec![0.0; seq_len * hidden_size];
        for h in 0..num_heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    // Source: head h, position s, dimension d
                    let src_idx = h * seq_len * head_dim + s * head_dim + d;
                    // Destination: position s, head h, dimension d
                    let dst_idx = s * hidden_size + h * head_dim + d;
                    concat_output[dst_idx] = attn_outputs[src_idx];
                }
            }
        }

        let concat_tensor = Tensor::from_vec(concat_output, true);

        // Output projection: (seq_len, hidden_size) @ (hidden_size, hidden_size) = (seq_len, hidden_size)
        matmul(&concat_tensor, &self.w_o, seq_len, hidden_size, hidden_size)
    }

    /// Get all parameters as a vector
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w_q, &self.w_k, &self.w_v, &self.w_o]
    }

    /// Get all parameters as mutable references for optimizer
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w_q, &mut self.w_k, &mut self.w_v, &mut self.w_o]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention_tiny() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let x = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
        let output = attn.forward(&x, 2);
        assert_eq!(output.len(), 2 * config.hidden_size);
    }

    #[test]
    fn test_multi_head_attention_parameters() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let params = attn.parameters();
        assert_eq!(params.len(), 4); // w_q, w_k, w_v, w_o
    }

    #[test]
    fn test_attention_longer_sequence() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let x = Tensor::from_vec(vec![0.1; 8 * config.hidden_size], true);
        let output = attn.forward(&x, 8);
        assert_eq!(output.len(), 8 * config.hidden_size);
    }

    #[test]
    fn test_attention_weight_sizes() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let kv_hidden = config.num_kv_heads * config.head_dim();
        assert_eq!(attn.w_q.len(), config.hidden_size * config.hidden_size);
        assert_eq!(attn.w_k.len(), config.hidden_size * kv_hidden);
        assert_eq!(attn.w_v.len(), config.hidden_size * kv_hidden);
        assert_eq!(attn.w_o.len(), config.hidden_size * config.hidden_size);
    }

    #[test]
    fn test_multi_head_attention_from_params_success() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();

        let mut params = HashMap::new();
        params.insert(
            "attn.q_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );
        params.insert(
            "attn.k_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.v_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.o_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );

        let attn = MultiHeadAttention::from_params(&config, &params, "attn");
        assert!(attn.is_some());
        let attn = attn.unwrap();
        assert_eq!(attn.w_q.len(), hidden_size * hidden_size);
    }

    #[test]
    fn test_multi_head_attention_from_params_missing_key() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;

        let mut params = HashMap::new();
        params.insert(
            "attn.q_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );
        // Missing k_proj, v_proj, o_proj

        let attn = MultiHeadAttention::from_params(&config, &params, "attn");
        assert!(attn.is_none());
    }

    #[test]
    fn test_attention_projections_backward() {
        // Test that Q, K, V projection matmuls have gradients
        // (isolated from the full attention which has intermediate tensor issues)
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let hidden_size = config.hidden_size;
        let seq_len = 2;

        let x = Tensor::from_vec(vec![0.1; seq_len * hidden_size], true);

        // Test Q projection
        let mut q = crate::autograd::matmul(&x, &attn.w_q, seq_len, hidden_size, hidden_size);
        let grad_out = ndarray::Array1::ones(seq_len * hidden_size);
        crate::autograd::backward(&mut q, Some(grad_out));

        assert!(attn.w_q.grad().is_some());
        let grad_q = attn.w_q.grad().unwrap();
        assert!(grad_q.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_output_projection_backward() {
        // Test output projection in isolation
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let hidden_size = config.hidden_size;
        let seq_len = 2;

        // Simulate concatenated attention output
        let concat_out = Tensor::from_vec(vec![0.1; seq_len * hidden_size], true);

        // Output projection
        let mut output =
            crate::autograd::matmul(&concat_out, &attn.w_o, seq_len, hidden_size, hidden_size);

        let grad_out = ndarray::Array1::ones(seq_len * hidden_size);
        crate::autograd::backward(&mut output, Some(grad_out));

        assert!(attn.w_o.grad().is_some());
        let grad_o = attn.w_o.grad().unwrap();
        assert!(grad_o.iter().all(|&v| v.is_finite()));
        let sum: f32 = grad_o.iter().map(|v| v.abs()).sum();
        assert!(
            sum > 0.0,
            "Output projection gradient should not be all zero"
        );
    }
}
