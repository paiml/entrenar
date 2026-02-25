//! Feed-forward network module
//!
//! This module provides position-wise feed-forward networks with SwiGLU activation.

use crate::autograd::matmul;
use crate::Tensor;
use std::collections::HashMap;

use super::config::TransformerConfig;

/// Position-wise Feed-Forward Network
pub struct FeedForward {
    /// Configuration
    config: TransformerConfig,
    /// Gate projection weight (hidden_size x intermediate_size)
    pub w_gate: Tensor,
    /// Up projection weight (hidden_size x intermediate_size)
    pub w_up: Tensor,
    /// Down projection weight (intermediate_size x hidden_size)
    pub w_down: Tensor,
}

impl FeedForward {
    /// Create new FFN layer with initialized weights
    pub fn new(config: &TransformerConfig) -> Self {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        // Xavier initialization scale
        let scale_up = (2.0 / (hidden_size + intermediate_size) as f32).sqrt();
        let scale_down = (2.0 / (intermediate_size + hidden_size) as f32).sqrt();

        Self {
            config: config.clone(),
            w_gate: Tensor::from_vec(
                (0..hidden_size * intermediate_size)
                    .map(|i| ((i as f32 * 0.567).sin() * scale_up))
                    .collect(),
                true,
            ),
            w_up: Tensor::from_vec(
                (0..hidden_size * intermediate_size)
                    .map(|i| ((i as f32 * 0.678).sin() * scale_up))
                    .collect(),
                true,
            ),
            w_down: Tensor::from_vec(
                (0..intermediate_size * hidden_size)
                    .map(|i| ((i as f32 * 0.789).sin() * scale_down))
                    .collect(),
                true,
            ),
        }
    }

    /// Create FFN layer from parameter map
    ///
    /// Expected parameter names (following HuggingFace convention):
    /// - `{prefix}.gate_proj.weight`
    /// - `{prefix}.up_proj.weight`
    /// - `{prefix}.down_proj.weight`
    /// # Contract (PMAT-333)
    /// Validates gate/up/down projection shapes against config dimensions.
    /// gate/up: hidden_size * intermediate_size, down: intermediate_size * hidden_size
    /// Returns None if any key is missing or shape is wrong.
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Option<Self> {
        let w_gate = params.get(&format!("{prefix}.gate_proj.weight"))?.clone();
        let w_up = params.get(&format!("{prefix}.up_proj.weight"))?.clone();
        let w_down = params.get(&format!("{prefix}.down_proj.weight"))?.clone();

        let expected_gate_up = config.hidden_size * config.intermediate_size;
        let expected_down = config.intermediate_size * config.hidden_size;

        // PMAT-333: Shape validation for FFN projections
        let checks: &[(&str, &Tensor, usize)] = &[
            ("gate_proj", &w_gate, expected_gate_up),
            ("up_proj", &w_up, expected_gate_up),
            ("down_proj", &w_down, expected_down),
        ];
        for &(name, tensor, expected) in checks {
            if tensor.len() != expected {
                eprintln!(
                    "[PMAT-333] {prefix}.{name}: shape mismatch — got {} elements, expected {expected}",
                    tensor.len()
                );
                return None;
            }
        }

        Some(Self {
            config: config.clone(),
            w_gate,
            w_up,
            w_down,
        })
    }

    /// Forward pass with SwiGLU activation
    ///
    /// FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size, flattened)
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor (seq_len * hidden_size, flattened)
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        // Gate projection: (seq_len, hidden) @ (hidden, intermediate) = (seq_len, intermediate)
        let gate = matmul(x, &self.w_gate, seq_len, hidden_size, intermediate_size);

        // Up projection: (seq_len, hidden) @ (hidden, intermediate) = (seq_len, intermediate)
        let up = matmul(x, &self.w_up, seq_len, hidden_size, intermediate_size);

        // SwiGLU: SiLU(gate) * up
        let gate_activated = crate::autograd::swish(&gate);
        let hidden = crate::autograd::mul(&gate_activated, &up);

        // Down projection: (seq_len, intermediate) @ (intermediate, hidden) = (seq_len, hidden)
        matmul(
            &hidden,
            &self.w_down,
            seq_len,
            intermediate_size,
            hidden_size,
        )
    }

    /// Get all parameters as a vector
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w_gate, &self.w_up, &self.w_down]
    }

    /// Get all parameters as mutable references for optimizer
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w_gate, &mut self.w_up, &mut self.w_down]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward_tiny() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let x = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
        let output = ffn.forward(&x, 2);
        assert_eq!(output.len(), 2 * config.hidden_size);
    }

    #[test]
    fn test_feed_forward_parameters() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let params = ffn.parameters();
        assert_eq!(params.len(), 3); // w_gate, w_up, w_down
    }

    #[test]
    fn test_ffn_longer_sequence() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let x = Tensor::from_vec(vec![0.1; 8 * config.hidden_size], true);
        let output = ffn.forward(&x, 8);
        assert_eq!(output.len(), 8 * config.hidden_size);
    }

    #[test]
    fn test_ffn_weight_sizes() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        assert_eq!(
            ffn.w_gate.len(),
            config.hidden_size * config.intermediate_size
        );
        assert_eq!(
            ffn.w_up.len(),
            config.hidden_size * config.intermediate_size
        );
        assert_eq!(
            ffn.w_down.len(),
            config.intermediate_size * config.hidden_size
        );
    }

    #[test]
    fn test_feed_forward_from_params_success() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let mut params = HashMap::new();
        params.insert(
            "ffn.gate_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
        );
        params.insert(
            "ffn.up_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
        );
        params.insert(
            "ffn.down_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; intermediate_size * hidden_size], true),
        );

        let ffn = FeedForward::from_params(&config, &params, "ffn");
        assert!(ffn.is_some());
        let ffn = ffn.unwrap();
        assert_eq!(ffn.w_gate.len(), hidden_size * intermediate_size);
    }

    #[test]
    fn test_feed_forward_from_params_missing_key() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let mut params = HashMap::new();
        params.insert(
            "ffn.gate_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
        );
        // Missing up_proj, down_proj

        let ffn = FeedForward::from_params(&config, &params, "ffn");
        assert!(ffn.is_none());
    }

    // =========================================================================
    // FALSIFY-F: §2.1.4 FFN Projections — Five-Whys Gap Analysis (Refs PMAT-333)
    //
    // Contract: tensor-layout-v1.yaml §tensors.gate_proj/up_proj/down_proj
    //   gate_proj: [intermediate, hidden], up_proj: [intermediate, hidden]
    //   down_proj: [hidden, intermediate]
    //   SwiGLU: FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    //
    // Five-Whys:
    //   Why 1: from_params accepts ANY tensor shape without validation
    //   Why 2: FeedForward stores raw Tensor, no ValidatedWeight wrapper
    //   Why 3: entrenar uses flattened 1D Tensors — no shape metadata
    //   Why 4: Shape errors only manifest at matmul time (runtime panic)
    //   Why 5: No constructor-time shape check exists (PMAT-333 gap)
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // entrenar's FFN construction prevents shape-related runtime panics."
    // =========================================================================

    /// FALSIFY-F1e: from_params rejects wrong-shape gate_proj (PMAT-333 fix)
    ///
    /// gate_proj should be [hidden_size * intermediate_size] elements.
    /// from_params now validates shape against config dimensions.
    #[test]
    fn falsify_f1e_from_params_rejects_wrong_shape_gate() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let mut params = HashMap::new();
        // WRONG: gate_proj has 42 elements instead of hidden*intermediate
        params.insert(
            "ffn.gate_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; 42], true),
        );
        params.insert(
            "ffn.up_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * intermediate_size], true),
        );
        params.insert(
            "ffn.down_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; intermediate_size * hidden_size], true),
        );

        // FIXED (PMAT-333): now rejected
        let ffn = FeedForward::from_params(&config, &params, "ffn");
        assert!(
            ffn.is_none(),
            "FALSIFY-F1e: PMAT-333 fix — from_params MUST reject wrong-shape gate_proj"
        );
    }

    /// FALSIFY-F2e: SwiGLU forward produces correct output dimensions
    ///
    /// For correct weights, output.len() == seq_len * hidden_size.
    #[test]
    fn falsify_f2e_swiglu_forward_correct_dims() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let seq_len = 4;
        let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], true);
        let output = ffn.forward(&x, seq_len);
        assert_eq!(
            output.len(),
            seq_len * config.hidden_size,
            "FALSIFY-F2e: FFN output must be seq_len * hidden_size"
        );
    }

    /// FALSIFY-F3e: FFN output is finite for valid inputs
    ///
    /// SwiGLU with bounded inputs must produce finite outputs.
    /// If gate/up/down weights contain NaN/Inf, output would be NaN.
    #[test]
    fn falsify_f3e_ffn_output_finite() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let x = Tensor::from_vec(vec![0.5; 2 * config.hidden_size], true);
        let output = ffn.forward(&x, 2);
        assert!(
            output.data().iter().all(|v| v.is_finite()),
            "FALSIFY-F3e: FFN output must be finite for bounded inputs"
        );
    }

    /// FALSIFY-F4e: gate_proj and up_proj share dimensions
    ///
    /// SwiGLU requires SiLU(gate(x)) * up(x) — element-wise multiply.
    /// gate_proj and up_proj must produce identically-sized outputs.
    #[test]
    fn falsify_f4e_gate_up_shape_parity() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        assert_eq!(
            ffn.w_gate.len(),
            ffn.w_up.len(),
            "FALSIFY-F4e: gate_proj and up_proj must have identical size for SwiGLU multiply"
        );
    }

    /// FALSIFY-F5e: down_proj dimensions reversed from gate/up
    ///
    /// gate/up: [hidden, intermediate] (transposed to row-major)
    /// down: [intermediate, hidden] (reversed)
    /// Total elements should still be hidden * intermediate.
    #[test]
    fn falsify_f5e_down_proj_reversed_same_total() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        assert_eq!(
            ffn.w_gate.len(),
            ffn.w_down.len(),
            "FALSIFY-F5e: gate and down must have same total elements (H*I)"
        );
        assert_eq!(
            ffn.w_down.len(),
            config.hidden_size * config.intermediate_size,
            "FALSIFY-F5e: down_proj must have hidden*intermediate elements"
        );
    }

    #[test]
    fn test_ffn_backward_gradient_exists() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let x = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
        let mut output = ffn.forward(&x, 2);

        // Backward pass
        let grad_out = ndarray::Array1::ones(2 * config.hidden_size);
        crate::autograd::backward(&mut output, Some(grad_out));

        // All FFN weights should have gradients
        assert!(ffn.w_gate.grad().is_some());
        assert!(ffn.w_up.grad().is_some());
        assert!(ffn.w_down.grad().is_some());
    }

    #[test]
    fn test_ffn_backward_gradients_finite() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let x = Tensor::from_vec(vec![0.5; 2 * config.hidden_size], true);
        let mut output = ffn.forward(&x, 2);

        let grad_out = ndarray::Array1::ones(2 * config.hidden_size);
        crate::autograd::backward(&mut output, Some(grad_out));

        // All gradients should be finite
        let grad_gate = ffn.w_gate.grad().unwrap();
        let grad_up = ffn.w_up.grad().unwrap();
        let grad_down = ffn.w_down.grad().unwrap();

        assert!(grad_gate.iter().all(|&v| v.is_finite()));
        assert!(grad_up.iter().all(|&v| v.is_finite()));
        assert!(grad_down.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_ffn_backward_swiglu_activation() {
        // Test that SwiGLU activation in FFN has proper gradients
        let config = TransformerConfig::tiny();

        // Test with various input magnitudes
        for scale in [0.1, 1.0, 2.0] {
            let ffn = FeedForward::new(&config);
            let x = Tensor::from_vec(
                (0..2 * config.hidden_size)
                    .map(|i| (i as f32 * 0.01).sin() * scale)
                    .collect(),
                true,
            );
            let mut output = ffn.forward(&x, 2);

            let grad_out = ndarray::Array1::ones(2 * config.hidden_size);
            crate::autograd::backward(&mut output, Some(grad_out));

            let grad_gate = ffn.w_gate.grad().unwrap();
            assert!(
                grad_gate.iter().all(|&v| v.is_finite()),
                "Gradients not finite for scale {scale}"
            );
        }
    }

    #[test]
    fn test_ffn_backward_gradient_nonzero() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let x = Tensor::from_vec(vec![0.5; 2 * config.hidden_size], true);
        let mut output = ffn.forward(&x, 2);

        let grad_out = ndarray::Array1::ones(2 * config.hidden_size);
        crate::autograd::backward(&mut output, Some(grad_out));

        // Gradients should not be all zero
        let grad_gate = ffn.w_gate.grad().unwrap();
        let sum: f32 = grad_gate.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "FFN gate gradients should not be all zero");
    }

    #[test]
    fn test_ffn_backward_different_seq_lengths() {
        let config = TransformerConfig::tiny();

        for seq_len in [1, 2, 4, 8] {
            let ffn = FeedForward::new(&config);
            let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], true);
            let mut output = ffn.forward(&x, seq_len);

            let grad_out = ndarray::Array1::ones(seq_len * config.hidden_size);
            crate::autograd::backward(&mut output, Some(grad_out));

            let grad_gate = ffn.w_gate.grad().unwrap();
            assert!(
                grad_gate.iter().all(|&v| v.is_finite()),
                "Non-finite gradient for seq_len {seq_len}"
            );
        }
    }

    #[test]
    fn test_ffn_backward_gradient_accumulation() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);

        // First forward-backward
        let x1 = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
        let mut output1 = ffn.forward(&x1, 2);
        let grad_out1 = ndarray::Array1::ones(2 * config.hidden_size);
        crate::autograd::backward(&mut output1, Some(grad_out1));
        let grad1 = ffn.w_gate.grad().unwrap().to_vec();

        // Second forward-backward should accumulate
        let x2 = Tensor::from_vec(vec![0.2; 2 * config.hidden_size], true);
        let mut output2 = ffn.forward(&x2, 2);
        let grad_out2 = ndarray::Array1::ones(2 * config.hidden_size);
        crate::autograd::backward(&mut output2, Some(grad_out2));
        let grad2 = ffn.w_gate.grad().unwrap().to_vec();

        // Gradients should have accumulated (different from first)
        assert!(
            grad2
                .iter()
                .zip(grad1.iter())
                .any(|(g2, g1)| g2.abs() != g1.abs()),
            "Gradients should accumulate across backward passes"
        );
    }

    #[test]
    fn test_ffn_backward_with_zero_input() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let x = Tensor::from_vec(vec![0.0; 2 * config.hidden_size], true);
        let mut output = ffn.forward(&x, 2);

        let grad_out = ndarray::Array1::ones(2 * config.hidden_size);
        crate::autograd::backward(&mut output, Some(grad_out));

        // Should still produce finite gradients
        let grad_gate = ffn.w_gate.grad().unwrap();
        assert!(grad_gate.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_ffn_backward_large_input() {
        let config = TransformerConfig::tiny();
        let ffn = FeedForward::new(&config);
        let x = Tensor::from_vec(vec![10.0; 2 * config.hidden_size], true);
        let mut output = ffn.forward(&x, 2);

        let grad_out = ndarray::Array1::ones(2 * config.hidden_size);
        crate::autograd::backward(&mut output, Some(grad_out));

        // Should still produce finite gradients
        let grad_gate = ffn.w_gate.grad().unwrap();
        assert!(grad_gate.iter().all(|&v| v.is_finite()));
    }
}
