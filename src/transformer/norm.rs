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
    pub fn from_params(params: &HashMap<String, Tensor>, prefix: &str, eps: f32) -> Option<Self> {
        let weight = params.get(&format!("{prefix}.weight"))?.clone();
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
        let norm = RMSNorm::from_params(&params, "test", 1e-6);
        assert!(norm.is_some());
        let norm = norm.unwrap();
        assert_eq!(norm.weight.len(), 4);
    }

    #[test]
    fn test_rms_norm_from_params_missing() {
        let params: HashMap<String, Tensor> = HashMap::new();
        let norm = RMSNorm::from_params(&params, "missing", 1e-6);
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
}
