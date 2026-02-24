//! RMS Normalization module
//!
//! This module provides RMS normalization layers for transformer models.

use crate::autograd::scale;
use crate::Tensor;
use std::collections::HashMap;

/// RMS Normalization layer
pub struct RMSNorm {
    /// Weight (scale) parameter
    pub weight: Tensor,
    /// Epsilon for numerical stability
    eps: f32,
}

impl RMSNorm {
    /// Create new RMS normalization layer
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::ones(hidden_size, true),
            eps,
        }
    }

    /// Create from parameters
    ///
    /// # Contract (PMAT-332 norm)
    /// Validates weight.len() == hidden_size.
    /// Returns None if key is missing or length is wrong.
    pub fn from_params(
        params: &HashMap<String, Tensor>,
        prefix: &str,
        eps: f32,
        hidden_size: usize,
    ) -> Option<Self> {
        let weight = params.get(&format!("{prefix}.weight"))?.clone();
        if weight.len() != hidden_size {
            eprintln!(
                "[PMAT-332] {prefix}.weight: length mismatch — got {}, expected {hidden_size}",
                weight.len()
            );
            return None;
        }
        Some(Self { weight, eps })
    }

    /// Forward pass
    ///
    /// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let n = x.len() as f32;

        // Compute RMS
        let sq_sum: f32 = x.data().iter().map(|v| v * v).sum();
        let rms = (sq_sum / n + self.eps).sqrt();

        // Normalize and scale
        let normalized = scale(x, 1.0 / rms);
        crate::autograd::mul(&normalized, &self.weight)
    }

    /// Forward pass for batched input
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size, flattened)
    /// * `seq_len` - Sequence length
    /// * `hidden_size` - Hidden dimension
    pub fn forward_batched(&self, x: &Tensor, seq_len: usize, hidden_size: usize) -> Tensor {
        let mut output = vec![0.0; seq_len * hidden_size];

        for s in 0..seq_len {
            let start = s * hidden_size;
            let end = start + hidden_size;
            let slice = &x.data().as_slice().unwrap()[start..end];

            // Compute RMS for this position
            let sq_sum: f32 = slice.iter().map(|v| v * v).sum();
            let rms = (sq_sum / hidden_size as f32 + self.eps).sqrt();

            // Normalize and scale
            for (i, &val) in slice.iter().enumerate() {
                output[start + i] = (val / rms) * self.weight.data()[i];
            }
        }

        Tensor::from_vec(output, x.requires_grad())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_forward() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
        let output = norm.forward(&x);
        assert_eq!(output.len(), 4);
        // Output should be normalized and scaled
        let data = output.data();
        assert!(data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_batched() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
        let output = norm.forward_batched(&x, 2, 4);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_rms_norm_normalization_property() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], true);
        let output = norm.forward(&x);
        // After RMS normalization, if weights are 1, output should be x / rms(x)
        // rms(x) = sqrt(mean(x^2)) = sqrt(4) = 2
        // so output = [2/2, 2/2, 2/2, 2/2] = [1, 1, 1, 1]
        let data = output.data();
        for &val in data.iter() {
            assert!((val - 1.0).abs() < 1e-5, "Expected ~1.0, got {val}");
        }
    }

    #[test]
    fn test_rms_norm_with_zeros() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], true);
        let output = norm.forward(&x);
        // With zeros input and eps, output should be finite (zeros)
        let data = output.data();
        assert!(data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_weight_requires_grad() {
        let norm = RMSNorm::new(4, 1e-6);
        assert!(norm.weight.requires_grad());
    }

    #[test]
    fn test_rms_norm_from_params() {
        let mut params = HashMap::new();
        params.insert(
            "test.weight".to_string(),
            Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], true),
        );
        let norm = RMSNorm::from_params(&params, "test", 1e-6, 4);
        assert!(norm.is_some());
        let norm = norm.unwrap();
        assert_eq!(norm.weight.len(), 4);
    }

    #[test]
    fn test_rms_norm_from_params_missing() {
        let params: HashMap<String, Tensor> = HashMap::new();
        let norm = RMSNorm::from_params(&params, "missing", 1e-6, 4);
        assert!(norm.is_none());
    }

    #[test]
    fn test_rms_norm_backward_gradient_exists() {
        let norm = RMSNorm::new(8, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
        let mut output = norm.forward(&x);

        let grad_out = ndarray::Array1::ones(8);
        crate::autograd::backward(&mut output, Some(grad_out));

        assert!(norm.weight.grad().is_some());
        let grad = norm.weight.grad().unwrap();
        assert!(grad.iter().all(|&v| v.is_finite()));
    }

    // =========================================================================
    // FALSIFY-N: §2.1.5-6 Layer Norms — Five-Whys Gap Analysis (Refs PMAT-332)
    //
    // Contract: tensor-layout-v1.yaml §tensors.input_layernorm/post_attention_layernorm/final_norm
    //   apr_shape: "[hidden]"
    //   transpose: "false"
    //   kernel: "element-wise multiply"
    //
    // Five-Whys:
    //   Why 1: from_params accepts ANY tensor length without validation
    //   Why 2: RMSNorm stores raw Tensor with no length check
    //   Why 3: Wrong-length norm produces wrong-scale hidden states
    //   Why 4: Mismatched norm length panics at element-wise multiply
    //   Why 5: No constructor-time length check exists
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // entrenar's norm handling prevents shape-related runtime errors."
    // =========================================================================

    /// FALSIFY-N1e: from_params rejects wrong-length norm weight (PMAT-332 norm fix)
    ///
    /// RMSNorm.from_params now validates weight.len() == hidden_size.
    /// A wrong-length weight is rejected at construction time.
    #[test]
    fn falsify_n1e_from_params_rejects_wrong_length_norm() {
        let mut params = HashMap::new();
        // WRONG: weight has 7 elements, should be hidden_size=4
        params.insert(
            "test.weight".to_string(),
            Tensor::from_vec(vec![1.0; 7], true),
        );
        let norm = RMSNorm::from_params(&params, "test", 1e-6, 4);
        // FIXED: from_params now rejects wrong-length weight
        assert!(norm.is_none(),
            "FALSIFY-N1e: PMAT-332 fix — from_params MUST reject wrong-length norm weight");
    }

    /// FALSIFY-N2e: RMSNorm forward produces finite output for valid input
    #[test]
    fn falsify_n2e_norm_output_finite() {
        let norm = RMSNorm::new(8, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
        let output = norm.forward(&x);
        assert!(output.data().iter().all(|v| v.is_finite()),
            "FALSIFY-N2e: RMSNorm output must be finite for valid input");
    }

    /// FALSIFY-N3e: RMSNorm weight length matches hidden_size when constructed via new()
    #[test]
    fn falsify_n3e_new_constructor_correct_length() {
        let hidden_sizes = [64, 128, 256, 896, 4096];
        for &hidden in &hidden_sizes {
            let norm = RMSNorm::new(hidden, 1e-6);
            assert_eq!(norm.weight.len(), hidden,
                "FALSIFY-N3e: RMSNorm::new({hidden}) weight must have {hidden} elements");
        }
    }

    /// FALSIFY-N4e: Batched forward preserves sequence*hidden dimension
    #[test]
    fn falsify_n4e_batched_forward_preserves_dims() {
        let hidden = 8;
        let seq_len = 4;
        let norm = RMSNorm::new(hidden, 1e-6);
        let x = Tensor::from_vec(vec![0.5; seq_len * hidden], true);
        let output = norm.forward_batched(&x, seq_len, hidden);
        assert_eq!(output.len(), seq_len * hidden,
            "FALSIFY-N4e: Batched norm must preserve seq_len * hidden dimension");
        assert!(output.data().iter().all(|v| v.is_finite()),
            "FALSIFY-N4e: Batched norm output must be finite");
    }

    /// FALSIFY-N5e: RMSNorm handles extreme but finite input values
    ///
    /// Very large values should still produce finite output due to normalization.
    #[test]
    fn falsify_n5e_extreme_input_still_finite() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![1e30, -1e30, 1e30, -1e30], true);
        let output = norm.forward(&x);
        assert!(output.data().iter().all(|v| v.is_finite()),
            "FALSIFY-N5e: RMSNorm must handle extreme values without Inf/NaN");
    }
}
