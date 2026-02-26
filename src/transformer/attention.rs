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
    /// # Contract (PMAT-331)
    /// Validates Q/K/V/O projection shapes against config dimensions.
    /// Returns None if any key is missing or shape is wrong.
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Option<Self> {
        let w_q = params.get(&format!("{prefix}.q_proj.weight"))?.clone();
        let w_k = params.get(&format!("{prefix}.k_proj.weight"))?.clone();
        let w_v = params.get(&format!("{prefix}.v_proj.weight"))?.clone();
        let w_o = params.get(&format!("{prefix}.o_proj.weight"))?.clone();

        let hidden = config.hidden_size;
        let kv_hidden = config.num_kv_heads * config.head_dim();

        // PMAT-331: Shape validation for attention projections
        let checks: &[(&str, &Tensor, usize)] = &[
            ("q_proj", &w_q, hidden * hidden),
            ("k_proj", &w_k, hidden * kv_hidden),
            ("v_proj", &w_v, hidden * kv_hidden),
            ("o_proj", &w_o, hidden * hidden),
        ];
        for &(name, tensor, expected) in checks {
            if tensor.len() != expected {
                eprintln!(
                    "[PMAT-331] {prefix}.{name}: shape mismatch — got {} elements, expected {expected}",
                    tensor.len()
                );
                return None;
            }
        }

        Some(Self { config: config.clone(), w_q, w_k, w_v, w_o })
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
                    q.data().as_slice().expect("contiguous Q tensor")[start..start + head_dim]
                        .to_vec()
                })
                .collect();

            // Extract K for this KV head: (seq_len, head_dim)
            let k_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * kv_hidden_size + kv_h * head_dim;
                    k.data().as_slice().expect("contiguous K tensor")[start..start + head_dim]
                        .to_vec()
                })
                .collect();

            // Extract V for this KV head: (seq_len, head_dim)
            let v_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * kv_hidden_size + kv_h * head_dim;
                    v.data().as_slice().expect("contiguous V tensor")[start..start + head_dim]
                        .to_vec()
                })
                .collect();

            // Scaled dot-product attention for this head
            let q_tensor = Tensor::from_vec(q_head, false);
            let k_tensor = Tensor::from_vec(k_head, false);
            let v_tensor = Tensor::from_vec(v_head, false);

            let attn_out = crate::autograd::attention(
                &q_tensor, &k_tensor, &v_tensor, seq_len, head_dim, seq_len, head_dim,
            );

            attn_outputs.extend_from_slice(
                attn_out.data().as_slice().expect("contiguous attention output"),
            );
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

/// LoRA-enabled linear projection
///
/// Computes: y = x @ W + scale * (x @ A) @ B
/// Where W is frozen base weight, A and B are trainable LoRA adapters
pub struct LoRAProjection {
    /// Base weight (frozen), shape (d_in × d_out)
    pub base_weight: Tensor,
    /// LoRA A matrix (down-projection), shape (d_in × rank)
    pub lora_a: Tensor,
    /// LoRA B matrix (up-projection), shape (rank × d_out)
    pub lora_b: Tensor,
    /// Input dimension
    pub d_in: usize,
    /// Output dimension
    pub d_out: usize,
    /// LoRA rank
    pub rank: usize,
    /// Scaling factor (alpha / rank)
    pub scale: f32,
}

impl LoRAProjection {
    /// Create a new LoRA projection
    ///
    /// # Arguments
    /// * `base_weight` - Frozen base weight [d_in × d_out]
    /// * `d_in` - Input dimension
    /// * `d_out` - Output dimension
    /// * `rank` - LoRA rank (typically 4, 8, 16, 32, or 64)
    /// * `alpha` - LoRA scaling parameter
    pub fn new(base_weight: Tensor, d_in: usize, d_out: usize, rank: usize, alpha: f32) -> Self {
        assert_eq!(base_weight.len(), d_in * d_out, "Base weight size mismatch");

        // Initialize A with small Gaussian-like noise
        let lora_a = Tensor::from_vec(
            (0..d_in * rank).map(|i| ((i as f32 * 0.123).sin() * 0.01)).collect(),
            true, // requires_grad
        );

        // Initialize B with zeros (standard LoRA init ensures ΔW = 0 at start)
        // But for immediate effect in experiments, use small values
        let lora_b = Tensor::from_vec(
            (0..rank * d_out).map(|i| ((i as f32 * 0.234).sin() * 0.005)).collect(),
            true, // requires_grad
        );

        Self { base_weight, lora_a, lora_b, d_in, d_out, rank, scale: alpha / rank as f32 }
    }

    /// Forward pass with LoRA
    ///
    /// Computes: y = x @ W + scale * (x @ A) @ B
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len × d_in]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor [seq_len × d_out]
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        // Base projection: x @ W, (seq × d_in) @ (d_in × d_out) = (seq × d_out)
        let base_out = matmul(x, &self.base_weight, seq_len, self.d_in, self.d_out);

        // LoRA path: scale * (x @ A) @ B
        // Step 1: x @ A, (seq × d_in) @ (d_in × rank) = (seq × rank)
        let lora_intermediate = matmul(x, &self.lora_a, seq_len, self.d_in, self.rank);

        // Step 2: (x @ A) @ B, (seq × rank) @ (rank × d_out) = (seq × d_out)
        let lora_out = matmul(&lora_intermediate, &self.lora_b, seq_len, self.rank, self.d_out);

        // Combine: base + scale * lora
        // Use autograd-compatible addition
        crate::autograd::add_scaled(&base_out, &lora_out, self.scale)
    }

    /// Get trainable LoRA parameters
    pub fn lora_params(&self) -> Vec<&Tensor> {
        vec![&self.lora_a, &self.lora_b]
    }

    /// Get mutable trainable LoRA parameters
    pub fn lora_params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.lora_a, &mut self.lora_b]
    }
}

/// Multi-head attention with deep LoRA injection
///
/// LoRA adapters are applied to Q, K, V, O projections during forward pass
pub struct MultiHeadAttentionWithLoRA {
    /// Configuration
    pub config: TransformerConfig,
    /// Query projection with LoRA
    pub q_proj: LoRAProjection,
    /// Key projection with LoRA
    pub k_proj: LoRAProjection,
    /// Value projection with LoRA
    pub v_proj: LoRAProjection,
    /// Output projection with LoRA
    pub o_proj: LoRAProjection,
}

impl MultiHeadAttentionWithLoRA {
    /// Create LoRA-enabled attention from existing attention weights
    ///
    /// # Arguments
    /// * `attn` - Base MultiHeadAttention with pretrained weights
    /// * `rank` - LoRA rank
    /// * `alpha` - LoRA alpha scaling factor
    pub fn from_attention(attn: &MultiHeadAttention, rank: usize, alpha: f32) -> Self {
        let hidden_size = attn.config.hidden_size;
        let kv_hidden_size = attn.config.num_kv_heads * attn.config.head_dim();

        Self {
            config: attn.config.clone(),
            q_proj: LoRAProjection::new(attn.w_q.clone(), hidden_size, hidden_size, rank, alpha),
            k_proj: LoRAProjection::new(attn.w_k.clone(), hidden_size, kv_hidden_size, rank, alpha),
            v_proj: LoRAProjection::new(attn.w_v.clone(), hidden_size, kv_hidden_size, rank, alpha),
            o_proj: LoRAProjection::new(attn.w_o.clone(), hidden_size, hidden_size, rank, alpha),
        }
    }

    /// Forward pass with deep LoRA injection
    ///
    /// LoRA is applied to all Q, K, V, O projections
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let kv_hidden_size = num_kv_heads * head_dim;

        // Project Q, K, V with LoRA
        let q = self.q_proj.forward(x, seq_len);
        let k = self.k_proj.forward(x, seq_len);
        let v = self.v_proj.forward(x, seq_len);

        // Multi-head attention with grouped-query attention support
        let mut attn_outputs = Vec::with_capacity(num_heads * seq_len * head_dim);
        let heads_per_kv = num_heads / num_kv_heads;

        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;

            // Extract Q for this head
            let q_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * hidden_size + h * head_dim;
                    q.data().as_slice().expect("contiguous Q tensor")[start..start + head_dim]
                        .to_vec()
                })
                .collect();

            // Extract K for this KV head
            let k_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * kv_hidden_size + kv_h * head_dim;
                    k.data().as_slice().expect("contiguous K tensor")[start..start + head_dim]
                        .to_vec()
                })
                .collect();

            // Extract V for this KV head
            let v_head: Vec<f32> = (0..seq_len)
                .flat_map(|s| {
                    let start = s * kv_hidden_size + kv_h * head_dim;
                    v.data().as_slice().expect("contiguous V tensor")[start..start + head_dim]
                        .to_vec()
                })
                .collect();

            // Scaled dot-product attention
            let q_tensor = Tensor::from_vec(q_head, false);
            let k_tensor = Tensor::from_vec(k_head, false);
            let v_tensor = Tensor::from_vec(v_head, false);

            let attn_out = crate::autograd::attention(
                &q_tensor, &k_tensor, &v_tensor, seq_len, head_dim, seq_len, head_dim,
            );

            attn_outputs.extend_from_slice(
                attn_out.data().as_slice().expect("contiguous attention output"),
            );
        }

        // Concatenate heads and reorder
        let mut concat_output = vec![0.0; seq_len * hidden_size];
        for h in 0..num_heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    let src_idx = h * seq_len * head_dim + s * head_dim + d;
                    let dst_idx = s * hidden_size + h * head_dim + d;
                    concat_output[dst_idx] = attn_outputs[src_idx];
                }
            }
        }

        let concat_tensor = Tensor::from_vec(concat_output, true);

        // Output projection with LoRA
        self.o_proj.forward(&concat_tensor, seq_len)
    }

    /// Get all trainable LoRA parameters
    pub fn lora_params(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.lora_params());
        params.extend(self.k_proj.lora_params());
        params.extend(self.v_proj.lora_params());
        params.extend(self.o_proj.lora_params());
        params
    }

    /// Get all trainable LoRA parameters as mutable references
    pub fn lora_params_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.lora_params_mut());
        params.extend(self.k_proj.lora_params_mut());
        params.extend(self.v_proj.lora_params_mut());
        params.extend(self.o_proj.lora_params_mut());
        params
    }

    /// Count total LoRA parameters
    pub fn lora_param_count(&self) -> usize {
        // Each projection has A (d_in × rank) + B (rank × d_out)
        let hidden = self.config.hidden_size;
        let kv_hidden = self.config.num_kv_heads * self.config.head_dim();
        let rank = self.q_proj.rank;

        // Q: (hidden × rank) + (rank × hidden)
        // K: (hidden × rank) + (rank × kv_hidden)
        // V: (hidden × rank) + (rank × kv_hidden)
        // O: (hidden × rank) + (rank × hidden)
        (hidden * rank + rank * hidden)      // Q
            + (hidden * rank + rank * kv_hidden) // K
            + (hidden * rank + rank * kv_hidden) // V
            + (hidden * rank + rank * hidden) // O
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
        assert!(sum > 0.0, "Output projection gradient should not be all zero");
    }

    // ============================================================================
    // LoRAProjection tests
    // ============================================================================

    #[test]
    fn test_lora_projection_new() {
        let d_in = 32;
        let d_out = 16;
        let rank = 4;
        let alpha = 8.0;

        let base_weight = Tensor::from_vec(vec![0.1; d_in * d_out], false);
        let lora = LoRAProjection::new(base_weight, d_in, d_out, rank, alpha);

        assert_eq!(lora.d_in, d_in);
        assert_eq!(lora.d_out, d_out);
        assert_eq!(lora.rank, rank);
        assert!((lora.scale - 2.0).abs() < 1e-6); // alpha / rank = 8 / 4 = 2
        assert_eq!(lora.lora_a.len(), d_in * rank);
        assert_eq!(lora.lora_b.len(), rank * d_out);
    }

    #[test]
    fn test_lora_projection_forward() {
        let d_in = 32;
        let d_out = 16;
        let rank = 4;
        let alpha = 8.0;
        let seq_len = 2;

        let base_weight = Tensor::from_vec(vec![0.1; d_in * d_out], false);
        let lora = LoRAProjection::new(base_weight, d_in, d_out, rank, alpha);

        let x = Tensor::from_vec(vec![0.1; seq_len * d_in], false);
        let output = lora.forward(&x, seq_len);

        assert_eq!(output.len(), seq_len * d_out);
        // Check output is finite
        assert!(output.data().iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_lora_projection_params() {
        let d_in = 32;
        let d_out = 16;
        let rank = 4;

        let base_weight = Tensor::from_vec(vec![0.1; d_in * d_out], false);
        let lora = LoRAProjection::new(base_weight, d_in, d_out, rank, 8.0);

        let params = lora.lora_params();
        assert_eq!(params.len(), 2); // lora_a and lora_b
    }

    #[test]
    fn test_lora_projection_params_mut() {
        let d_in = 32;
        let d_out = 16;
        let rank = 4;

        let base_weight = Tensor::from_vec(vec![0.1; d_in * d_out], false);
        let mut lora = LoRAProjection::new(base_weight, d_in, d_out, rank, 8.0);

        let params = lora.lora_params_mut();
        assert_eq!(params.len(), 2);
    }

    #[test]
    #[should_panic(expected = "Base weight size mismatch")]
    fn test_lora_projection_size_mismatch() {
        let d_in = 32;
        let d_out = 16;
        let rank = 4;

        // Wrong base weight size
        let base_weight = Tensor::from_vec(vec![0.1; d_in * d_out + 1], false);
        let _ = LoRAProjection::new(base_weight, d_in, d_out, rank, 8.0);
    }

    // ============================================================================
    // MultiHeadAttentionWithLoRA tests
    // ============================================================================

    #[test]
    fn test_mha_with_lora_creation() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let rank = 4;
        let alpha = 8.0;

        let lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, rank, alpha);

        assert_eq!(lora_attn.q_proj.rank, rank);
        assert_eq!(lora_attn.k_proj.rank, rank);
        assert_eq!(lora_attn.v_proj.rank, rank);
        assert_eq!(lora_attn.o_proj.rank, rank);
    }

    #[test]
    fn test_mha_with_lora_forward() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, 4, 8.0);

        let seq_len = 2;
        let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], false);
        let output = lora_attn.forward(&x, seq_len);

        assert_eq!(output.len(), seq_len * config.hidden_size);
        // Check output is finite and non-zero
        assert!(output.data().iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_mha_with_lora_params() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, 4, 8.0);

        let params = lora_attn.lora_params();
        // 4 projections × 2 params each = 8
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_mha_with_lora_params_mut() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let mut lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, 4, 8.0);

        let params = lora_attn.lora_params_mut();
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_mha_with_lora_param_count() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let rank = 4;
        let lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, rank, 8.0);

        let param_count = lora_attn.lora_param_count();

        // Calculate expected:
        let hidden = config.hidden_size;
        let kv_hidden = config.num_kv_heads * config.head_dim();
        let expected = (hidden * rank + rank * hidden)      // Q
            + (hidden * rank + rank * kv_hidden) // K
            + (hidden * rank + rank * kv_hidden) // V
            + (hidden * rank + rank * hidden); // O

        assert_eq!(param_count, expected);
        assert!(param_count > 0);
    }

    #[test]
    fn test_mha_with_lora_longer_sequence() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, 4, 8.0);

        let seq_len = 8;
        let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], false);
        let output = lora_attn.forward(&x, seq_len);

        assert_eq!(output.len(), seq_len * config.hidden_size);
    }

    #[test]
    fn test_parameters_mut() {
        let config = TransformerConfig::tiny();
        let mut attn = MultiHeadAttention::new(&config);

        let params = attn.parameters_mut();
        assert_eq!(params.len(), 4);
    }

    // =========================================================================
    // FALSIFY-A: §2.1.3 Attention Projections — Five-Whys Gap Analysis (Refs PMAT-331)
    //
    // Contract: tensor-layout-v1.yaml §tensors.q_proj/k_proj/v_proj/o_proj
    //   q_proj: [num_heads*head_dim, hidden] (= [hidden, hidden] for MHA)
    //   k_proj: [num_kv_heads*head_dim, hidden] (smaller for GQA)
    //   v_proj: [num_kv_heads*head_dim, hidden] (smaller for GQA)
    //   o_proj: [hidden, num_heads*head_dim]
    //
    // Five-Whys:
    //   Why 1: Trained model's attention weights could be wrong shape
    //   Why 2: from_params accepts any tensor without shape validation
    //   Why 3: No ValidatedWeight in entrenar
    //   Why 4: entrenar predates the Poka-Yoke contract
    //   Why 5: No cross-crate contract enforcement for training weights
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // entrenar's attention weight handling prevents degenerate models."
    // =========================================================================

    /// FALSIFY-A1e: from_params rejects wrong-shape Q weight (PMAT-331 fix)
    ///
    /// from_params now validates Q projection shape against config dimensions.
    /// A tensor of 50 elements is rejected when hidden*hidden is expected.
    #[test]
    fn falsify_a1e_from_params_rejects_wrong_shape_q_weight() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();

        let mut params = HashMap::new();
        // WRONG-SHAPE q_proj: 50 elements instead of hidden*hidden
        params.insert("attn.q_proj.weight".to_string(), Tensor::from_vec(vec![0.1; 50], true));
        // Correct k, v, o
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
        // FIXED (PMAT-331): now rejected
        assert!(
            attn.is_none(),
            "FALSIFY-A1e: PMAT-331 fix — from_params MUST reject wrong-shape q_proj"
        );
    }

    /// FALSIFY-A2e: GQA init produces correct K/V dimensions
    ///
    /// For GQA (num_kv_heads < num_heads), K/V must be smaller than Q.
    /// If init uses num_heads for K/V, the shapes are wrong.
    #[test]
    fn falsify_a2e_gqa_init_correct_kv_dimensions() {
        let mut config = TransformerConfig::tiny();
        config.num_kv_heads = 1; // Force GQA: 1 KV head, but num_heads > 1

        let attn = MultiHeadAttention::new(&config);
        let head_dim = config.head_dim();
        let kv_hidden = config.num_kv_heads * head_dim; // 1 * head_dim

        // Q: hidden * hidden
        assert_eq!(
            attn.w_q.len(),
            config.hidden_size * config.hidden_size,
            "FALSIFY-A2e: Q projection must be hidden*hidden"
        );

        // K: hidden * kv_hidden (smaller than Q for GQA)
        assert_eq!(
            attn.w_k.len(),
            config.hidden_size * kv_hidden,
            "FALSIFY-A2e: K projection must use num_kv_heads, not num_heads"
        );

        // V: hidden * kv_hidden (same as K)
        assert_eq!(
            attn.w_v.len(),
            config.hidden_size * kv_hidden,
            "FALSIFY-A2e: V projection must use num_kv_heads, not num_heads"
        );

        // O: hidden * hidden (matches Q output)
        assert_eq!(
            attn.w_o.len(),
            config.hidden_size * config.hidden_size,
            "FALSIFY-A2e: O projection must be hidden*hidden"
        );

        // K/V must be SMALLER than Q for GQA
        assert!(
            attn.w_k.len() < attn.w_q.len(),
            "FALSIFY-A2e: For GQA, K weight must be smaller than Q weight"
        );
    }

    /// FALSIFY-A3e: GQA forward produces correct output dimensions
    ///
    /// With num_kv_heads < num_heads, the forward pass must still produce
    /// [seq_len, hidden_size] output (not [seq_len, kv_hidden]).
    #[test]
    fn falsify_a3e_gqa_forward_correct_output_dims() {
        let mut config = TransformerConfig::tiny();
        config.num_kv_heads = 1; // Force GQA

        let attn = MultiHeadAttention::new(&config);
        let seq_len = 3;
        let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], true);
        let output = attn.forward(&x, seq_len);

        assert_eq!(
            output.len(),
            seq_len * config.hidden_size,
            "FALSIFY-A3e: GQA output must be seq_len * hidden_size, not seq_len * kv_hidden"
        );
    }

    /// FALSIFY-A4e: Attention init produces non-degenerate values
    ///
    /// Like FALSIFY-E7a for embeddings: init must produce varied, finite values.
    #[test]
    fn falsify_a4e_init_produces_valid_attention_weights() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);

        for (name, w) in
            [("w_q", &attn.w_q), ("w_k", &attn.w_k), ("w_v", &attn.w_v), ("w_o", &attn.w_o)]
        {
            let data = w.data();
            let slice = data.as_slice().expect("data as slice");

            // No NaN
            let nan_count = slice.iter().filter(|v| v.is_nan()).count();
            assert_eq!(nan_count, 0, "FALSIFY-A4e: {name} init must not contain NaN");

            // No Inf
            let inf_count = slice.iter().filter(|v| v.is_infinite()).count();
            assert_eq!(inf_count, 0, "FALSIFY-A4e: {name} init must not contain Inf");

            // Values vary
            let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (max - min).abs() > 1e-6,
                "FALSIFY-A4e: {name} init values are constant ({min}..{max}) — degenerate weight"
            );
        }
    }

    /// FALSIFY-A5e: Attention forward produces finite outputs
    ///
    /// If any attention weight is degenerate, output should still be finite
    /// (the init is designed to prevent this).
    #[test]
    fn falsify_a5e_forward_produces_finite_output() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let seq_len = 4;
        let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], true);
        let output = attn.forward(&x, seq_len);

        let data = output.data();
        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        let inf_count = data.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(nan_count, 0, "FALSIFY-A5e: Attention output must not contain NaN");
        assert_eq!(inf_count, 0, "FALSIFY-A5e: Attention output must not contain Inf");
    }

    // =========================================================================
    // FALSIFY-GQ: gqa-kernel-v1.yaml contract (entrenar MultiHeadAttention GQA)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had FALSIFY-A tests but zero FALSIFY-GQ-* tests
    //   Why 2: FALSIFY-A tests verify projections/shapes, not GQA invariants
    //   Why 3: no mapping from gqa-kernel-v1.yaml to entrenar test names
    //   Why 4: entrenar's GQA support added after FALSIFY-A tests
    //   Why 5: GQA was "obviously correct" (just index K/V by h/heads_per_kv)
    //
    // References:
    //   - provable-contracts/contracts/gqa-kernel-v1.yaml
    //   - Ainslie et al. (2023) "GQA: Training Generalized MQT Models"
    // =========================================================================

    /// FALSIFY-GQ-001e: GQA output shape correct for various head configs
    #[test]
    fn falsify_gq_001e_output_shape() {
        for (num_heads, num_kv_heads) in [(2, 2), (4, 2), (4, 1), (2, 1)] {
            let mut config = TransformerConfig::tiny();
            config.num_attention_heads = num_heads;
            config.num_kv_heads = num_kv_heads;

            let attn = MultiHeadAttention::new(&config);
            let seq_len = 3;
            let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], true);
            let output = attn.forward(&x, seq_len);

            assert_eq!(
                output.len(),
                seq_len * config.hidden_size,
                "FALSIFIED GQ-001e: output len mismatch for heads={num_heads},kv={num_kv_heads}"
            );
        }
    }

    /// FALSIFY-GQ-002e: MHA degeneration — kv_heads == num_heads produces finite output
    #[test]
    fn falsify_gq_002e_mha_degeneration() {
        let config = TransformerConfig::tiny(); // num_heads == num_kv_heads == 2
        assert_eq!(config.num_attention_heads, config.num_kv_heads);

        let attn = MultiHeadAttention::new(&config);
        let seq_len = 4;
        let x = Tensor::from_vec(
            (0..seq_len * config.hidden_size).map(|i| (i as f32 * 0.37).sin()).collect(),
            true,
        );
        let output = attn.forward(&x, seq_len);

        let data = output.data();
        for (i, v) in data.iter().enumerate() {
            assert!(v.is_finite(), "FALSIFIED GQ-002e: MHA output[{i}] = {v} (not finite)");
        }
    }

    /// FALSIFY-GQ-004e: Head divisibility — GQA requires num_heads % num_kv_heads == 0
    #[test]
    fn falsify_gq_004e_head_divisibility() {
        // Valid configurations should not panic
        for (nh, nkv) in [(2, 1), (2, 2), (4, 1), (4, 2), (4, 4), (8, 2), (8, 4)] {
            let mut config = TransformerConfig::tiny();
            config.num_attention_heads = nh;
            config.num_kv_heads = nkv;
            assert_eq!(nh % nkv, 0, "FALSIFIED GQ-004e: test config has invalid head ratio");
            // Should not panic during construction or forward
            let attn = MultiHeadAttention::new(&config);
            let x = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
            let _ = attn.forward(&x, 2);
        }
    }

    /// FALSIFY-GQ-006e: MQA boundary — kv_heads=1 broadcasts single KV to all heads
    #[test]
    fn falsify_gq_006e_mqa_boundary() {
        let mut config = TransformerConfig::tiny();
        config.num_attention_heads = 4;
        config.num_kv_heads = 1;
        // Adjust hidden_size to be divisible by 4 heads
        config.hidden_size = 64;

        let attn = MultiHeadAttention::new(&config);
        let seq_len = 3;
        let x = Tensor::from_vec(
            (0..seq_len * config.hidden_size).map(|i| (i as f32 * 0.73).cos()).collect(),
            true,
        );
        let output = attn.forward(&x, seq_len);

        assert_eq!(
            output.len(),
            seq_len * config.hidden_size,
            "FALSIFIED GQ-006e: MQA output size wrong"
        );

        // All finite
        let data = output.data();
        for (i, v) in data.iter().enumerate() {
            assert!(v.is_finite(), "FALSIFIED GQ-006e: MQA output[{i}] = {v} (not finite)");
        }
    }

    mod gq_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // FALSIFY-GQ-001e-prop: GQA output shape for random configs
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn falsify_gq_001e_prop_output_shape(
                config_idx in 0..4usize,
                seq_len in 2..=6usize,
                seed in 0..500u32,
            ) {
                let configs: [(usize, usize); 4] = [
                    (2, 2), (2, 1), (4, 2), (4, 1),
                ];
                let (num_heads, num_kv_heads) = configs[config_idx];
                let mut config = TransformerConfig::tiny();
                config.num_attention_heads = num_heads;
                config.num_kv_heads = num_kv_heads;

                let attn = MultiHeadAttention::new(&config);
                let data: Vec<f32> = (0..seq_len * config.hidden_size)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                    .collect();
                let x = Tensor::from_vec(data, true);
                let output = attn.forward(&x, seq_len);

                prop_assert_eq!(
                    output.len(),
                    seq_len * config.hidden_size,
                    "FALSIFIED GQ-001e-prop: output len mismatch"
                );

                // All finite
                for v in output.data() {
                    prop_assert!(
                        v.is_finite(),
                        "FALSIFIED GQ-001e-prop: non-finite output"
                    );
                }
            }
        }

        // FALSIFY-GQ-006e-prop: MQA boundary with random inputs
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(30))]

            #[test]
            fn falsify_gq_006e_prop_mqa_boundary(
                seed in 0..500u32,
                seq_len in 2..=5usize,
            ) {
                let mut config = TransformerConfig::tiny();
                config.num_attention_heads = 4;
                config.num_kv_heads = 1;
                config.hidden_size = 64;

                let attn = MultiHeadAttention::new(&config);
                let data: Vec<f32> = (0..seq_len * config.hidden_size)
                    .map(|i| ((i as f32 + seed as f32) * 0.73).cos())
                    .collect();
                let x = Tensor::from_vec(data, true);
                let output = attn.forward(&x, seq_len);

                prop_assert_eq!(
                    output.len(),
                    seq_len * config.hidden_size,
                    "FALSIFIED GQ-006e-prop: MQA output len mismatch"
                );

                for v in output.data() {
                    prop_assert!(
                        v.is_finite(),
                        "FALSIFIED GQ-006e-prop: non-finite MQA output"
                    );
                }
            }
        }
    }
}
