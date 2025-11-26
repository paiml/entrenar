//! QLoRA (Quantized LoRA) implementation
//!
//! QLoRA reduces memory usage by storing frozen base weights in 4-bit quantized format.
//! During forward pass, weights are dequantized on-the-fly.
//!
//! Memory savings: ~75% for base weights (4-bit vs 32-bit)
//! LoRA adapters remain in full precision for training.

use crate::autograd::matmul;
use crate::lora::LoRALayer;
use crate::quant::{dequantize_4bit, quantize_4bit, Quantized4Bit};
use crate::Tensor;

/// QLoRA layer with 4-bit quantized base weight
///
/// Memory-efficient variant of LoRALayer that stores the frozen base weight
/// in 4-bit quantized format, reducing memory usage by ~75%.
pub struct QLoRALayer {
    /// Quantized base weight (4-bit)
    base_weight_quantized: Quantized4Bit,
    /// LoRA matrix A [r * d_in] - full precision, trainable
    lora_a: Tensor,
    /// LoRA matrix B [d_out * r] - full precision, trainable
    lora_b: Tensor,
    /// Output dimension
    d_out: usize,
    /// Input dimension
    d_in: usize,
    /// LoRA rank
    rank: usize,
    /// Scaling factor (alpha/rank)
    scale: f32,
    /// Whether the adapter is merged (not supported for quantized weights)
    merged: bool,
}

impl QLoRALayer {
    /// Create QLoRA layer from existing LoRA layer
    ///
    /// # Arguments
    /// * `lora_layer` - Existing LoRALayer to convert
    ///
    /// # Returns
    /// QLoRALayer with quantized base weight
    pub fn from_lora(lora_layer: LoRALayer) -> Self {
        let base_weight_data = lora_layer.base_weight().data().to_vec();
        let base_weight_quantized = quantize_4bit(&base_weight_data);

        Self {
            base_weight_quantized,
            lora_a: lora_layer.lora_a().clone(),
            lora_b: lora_layer.lora_b().clone(),
            d_out: lora_layer.d_out(),
            d_in: lora_layer.d_in(),
            rank: lora_layer.rank(),
            scale: lora_layer.scale(),
            merged: false,
        }
    }

    /// Create QLoRA layer directly with quantized base weight
    ///
    /// # Arguments
    /// * `base_weight` - Base weight to quantize [d_out * d_in]
    /// * `d_out` - Output dimension
    /// * `d_in` - Input dimension
    /// * `rank` - LoRA rank
    /// * `alpha` - LoRA alpha parameter
    pub fn new(base_weight: Tensor, d_out: usize, d_in: usize, rank: usize, alpha: f32) -> Self {
        // Create LoRALayer first, then convert
        let lora_layer = LoRALayer::new(base_weight, d_out, d_in, rank, alpha);
        Self::from_lora(lora_layer)
    }

    /// Forward pass with on-the-fly dequantization
    ///
    /// # Arguments
    /// * `x` - Input tensor [d_in]
    ///
    /// # Returns
    /// Output tensor [d_out]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.len(), self.d_in, "Input size must match d_in");

        // Dequantize base weight on-the-fly
        let base_weight_data = dequantize_4bit(&self.base_weight_quantized);
        let base_weight = Tensor::new(ndarray::arr1(&base_weight_data), false);

        // Base forward: W @ x
        let base_output = matmul(&base_weight, x, self.d_out, self.d_in, 1);

        if !self.merged {
            // LoRA forward: scale * (B @ (A @ x))
            let lora_out_a = matmul(&self.lora_a, x, self.rank, self.d_in, 1);
            let lora_out_b = matmul(&self.lora_b, &lora_out_a, self.d_out, self.rank, 1);

            // Scale and add
            let mut scaled_lora_data = lora_out_b.data().to_owned();
            for val in scaled_lora_data.iter_mut() {
                *val *= self.scale;
            }
            let scaled_lora = Tensor::new(scaled_lora_data, false);

            let mut result_data = base_output.data().to_owned();
            for (i, val) in result_data.iter_mut().enumerate() {
                *val += scaled_lora.data()[i];
            }
            Tensor::new(result_data, base_output.requires_grad())
        } else {
            base_output
        }
    }

    /// Get reference to LoRA A matrix
    pub fn lora_a(&self) -> &Tensor {
        &self.lora_a
    }

    /// Get mutable reference to LoRA A matrix
    pub fn lora_a_mut(&mut self) -> &mut Tensor {
        &mut self.lora_a
    }

    /// Get reference to LoRA B matrix
    pub fn lora_b(&self) -> &Tensor {
        &self.lora_b
    }

    /// Get mutable reference to LoRA B matrix
    pub fn lora_b_mut(&mut self) -> &mut Tensor {
        &mut self.lora_b
    }

    /// Get trainable parameters (A and B)
    pub fn trainable_params(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.lora_a, &mut self.lora_b]
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get scale factor
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get output dimension
    pub fn d_out(&self) -> usize {
        self.d_out
    }

    /// Get input dimension
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let base_unquantized_bytes = self.d_out * self.d_in * 4; // f32
        let base_quantized_bytes = self.base_weight_quantized.memory_bytes();
        let lora_a_bytes = self.lora_a.len() * 4;
        let lora_b_bytes = self.lora_b.len() * 4;

        MemoryStats {
            base_unquantized_bytes,
            base_quantized_bytes,
            lora_bytes: lora_a_bytes + lora_b_bytes,
            total_bytes: base_quantized_bytes + lora_a_bytes + lora_b_bytes,
            compression_ratio: self.base_weight_quantized.compression_ratio(),
        }
    }

    /// Check if merged (always false for quantized layers)
    pub fn is_merged(&self) -> bool {
        self.merged
    }
}

/// Memory usage statistics for QLoRA layer
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Base weight size if unquantized (bytes)
    pub base_unquantized_bytes: usize,
    /// Base weight size quantized (bytes)
    pub base_quantized_bytes: usize,
    /// LoRA adapters size (bytes)
    pub lora_bytes: usize,
    /// Total memory usage (bytes)
    pub total_bytes: usize,
    /// Compression ratio for base weights
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    // ========================================================================
    // PROPERTY TESTS - Mathematical correctness validation
    // ========================================================================

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(200))]

        /// Memory savings should be consistent with dimensions
        #[test]
        fn prop_qlora_memory_savings_consistent(
            d in 8usize..32,
            rank in 1usize..8,
            alpha in 1.0f32..32.0
        ) {
            let size = d * d;
            let base_weight = Tensor::from_vec(vec![0.5; size], false);
            let qlora = QLoRALayer::new(base_weight, d, d, rank, alpha);

            let stats = qlora.memory_stats();

            // Quantized should always be smaller than unquantized
            prop_assert!(stats.base_quantized_bytes <= stats.base_unquantized_bytes);

            // Compression ratio should be > 1.0
            prop_assert!(stats.compression_ratio >= 1.0);

            // Total bytes = quantized base + lora
            prop_assert_eq!(
                stats.total_bytes,
                stats.base_quantized_bytes + stats.lora_bytes
            );

            // LoRA bytes = (d*rank + d*rank) * 4 bytes
            let expected_lora_bytes = (d * rank + d * rank) * 4;
            prop_assert_eq!(stats.lora_bytes, expected_lora_bytes);
        }

        /// LoRA parameters should be preserved during quantization
        #[test]
        fn prop_lora_params_preserved_after_quantization(
            d_out in 4usize..16,
            d_in in 4usize..16,
            rank in 1usize..4,
            alpha in 1.0f32..16.0
        ) {
            let size = d_out * d_in;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);
            let lora = LoRALayer::new(base_weight.clone(), d_out, d_in, rank, alpha);

            let qlora = QLoRALayer::from_lora(lora.clone());

            // Dimensions should match exactly
            prop_assert_eq!(qlora.d_out(), lora.d_out());
            prop_assert_eq!(qlora.d_in(), lora.d_in());
            prop_assert_eq!(qlora.rank(), lora.rank());

            // Scale should match
            prop_assert!((qlora.scale() - lora.scale()).abs() < 1e-6);

            // LoRA A and B data should be identical
            prop_assert_eq!(qlora.lora_a().data().len(), lora.lora_a().data().len());
            prop_assert_eq!(qlora.lora_b().data().len(), lora.lora_b().data().len());

            for (a, b) in qlora.lora_a().data().iter().zip(lora.lora_a().data().iter()) {
                prop_assert!((a - b).abs() < 1e-6);
            }
            for (a, b) in qlora.lora_b().data().iter().zip(lora.lora_b().data().iter()) {
                prop_assert!((a - b).abs() < 1e-6);
            }
        }

        /// Quantization error should be bounded
        #[test]
        fn prop_quantization_error_bounded(
            d in 8usize..24,
        ) {
            let size = d * d;
            // Use values in reasonable range for 4-bit quantization
            let base_weight = Tensor::from_vec(
                (0..size).map(|i| ((i % 16) as f32 - 8.0) * 0.1).collect(),
                false
            );
            let lora = LoRALayer::new(base_weight.clone(), d, d, 2, 4.0);
            let qlora = QLoRALayer::from_lora(lora.clone());

            // Forward with same input
            let x = Tensor::from_vec(vec![0.1; d], true);
            let lora_out = lora.forward(&x);
            let qlora_out = qlora.forward(&x);

            // Outputs should be close (quantization introduces some error)
            prop_assert_eq!(lora_out.len(), qlora_out.len());
            for i in 0..lora_out.len() {
                let diff = (lora_out.data()[i] - qlora_out.data()[i]).abs();
                // Allow 30% relative error due to 4-bit quantization
                let max_diff = lora_out.data()[i].abs() * 0.3 + 0.5;
                prop_assert!(
                    diff < max_diff,
                    "Quantization error {} > {} at index {}",
                    diff, max_diff, i
                );
            }
        }

        /// Forward output dimensions should always be correct
        #[test]
        fn prop_forward_dimensions_correct(
            d_out in 4usize..16,
            d_in in 4usize..16,
            rank in 1usize..4,
        ) {
            let size = d_out * d_in;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);
            let qlora = QLoRALayer::new(base_weight, d_out, d_in, rank, 4.0);

            let x = Tensor::from_vec(vec![0.5; d_in], true);
            let output = qlora.forward(&x);

            prop_assert_eq!(output.len(), d_out);
        }

        /// Trainable params should have correct dimensions
        #[test]
        fn prop_trainable_params_dimensions(
            d_out in 4usize..16,
            d_in in 4usize..16,
            rank in 1usize..4,
        ) {
            let size = d_out * d_in;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);
            let mut qlora = QLoRALayer::new(base_weight, d_out, d_in, rank, 4.0);

            let params = qlora.trainable_params();
            prop_assert_eq!(params.len(), 2);

            // A: [rank * d_in]
            prop_assert_eq!(params[0].len(), rank * d_in);
            // B: [d_out * rank]
            prop_assert_eq!(params[1].len(), d_out * rank);

            // Both should require grad
            prop_assert!(params[0].requires_grad());
            prop_assert!(params[1].requires_grad());
        }
    }

    // ========================================================================
    // UNIT TESTS
    // ========================================================================

    #[test]
    fn test_qlora_creation() {
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
        let qlora = QLoRALayer::new(base_weight, 2, 2, 1, 2.0);

        assert_eq!(qlora.rank(), 1);
        assert_eq!(qlora.d_out(), 2);
        assert_eq!(qlora.d_in(), 2);
        assert_abs_diff_eq!(qlora.scale(), 2.0, epsilon = 1e-6); // alpha/rank = 2/1
        assert!(!qlora.is_merged());
    }

    #[test]
    fn test_qlora_forward_matches_lora() {
        // Test that QLoRA forward pass approximates LoRA
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora = LoRALayer::new(base_weight.clone(), 2, 2, 1, 1.0);
        *lora.lora_a_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);
        *lora.lora_b_mut().data_mut() = ndarray::arr1(&[0.3, 0.3]);

        let mut qlora = QLoRALayer::new(base_weight, 2, 2, 1, 1.0);
        *qlora.lora_a_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);
        *qlora.lora_b_mut().data_mut() = ndarray::arr1(&[0.3, 0.3]);

        let x = Tensor::from_vec(vec![2.0, 3.0], true);

        let lora_output = lora.forward(&x);
        let qlora_output = qlora.forward(&x);

        // Outputs should be close (within quantization error)
        assert_eq!(lora_output.len(), qlora_output.len());
        for i in 0..lora_output.len() {
            let diff = (lora_output.data()[i] - qlora_output.data()[i]).abs();
            assert!(
                diff < 0.2,
                "Output mismatch at {}: {} vs {} (diff: {})",
                i,
                lora_output.data()[i],
                qlora_output.data()[i],
                diff
            );
        }
    }

    #[test]
    fn test_qlora_memory_savings() {
        // Test with large enough weight to show compression (use perfect square)
        let d = 16; // 16x16 = 256 elements
        let size = d * d;
        let base_weight = Tensor::from_vec(vec![1.0; size], false);
        let qlora = QLoRALayer::new(base_weight, d, d, 8, 16.0);

        let stats = qlora.memory_stats();

        // Should see significant memory savings
        assert!(
            stats.base_quantized_bytes < stats.base_unquantized_bytes,
            "Quantized should use less memory"
        );

        // Compression ratio should be > 6x
        assert!(
            stats.compression_ratio > 6.0,
            "Compression ratio {} should be > 6.0",
            stats.compression_ratio
        );
    }

    #[test]
    fn test_qlora_trainable_params() {
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
        let mut qlora = QLoRALayer::new(base_weight, 2, 2, 2, 4.0);

        let params = qlora.trainable_params();
        assert_eq!(params.len(), 2);

        // Both should be trainable
        assert!(params[0].requires_grad());
        assert!(params[1].requires_grad());
    }

    #[test]
    fn test_qlora_from_lora() {
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false);
        let lora = LoRALayer::new(base_weight, 3, 2, 2, 8.0);

        let qlora = QLoRALayer::from_lora(lora);

        assert_eq!(qlora.rank(), 2);
        assert_eq!(qlora.d_out(), 3);
        assert_eq!(qlora.d_in(), 2);
        assert_abs_diff_eq!(qlora.scale(), 4.0, epsilon = 1e-6); // 8/2 = 4
    }

    #[test]
    fn test_qlora_large_matrix() {
        // Test with realistic transformer dimensions
        let d_model = 256;
        let base_weight = Tensor::from_vec(vec![1.0; d_model * d_model], false);
        let qlora = QLoRALayer::new(base_weight, d_model, d_model, 16, 32.0);

        let x = Tensor::from_vec(vec![0.5; d_model], true);
        let output = qlora.forward(&x);

        assert_eq!(output.len(), d_model);

        // Check memory savings
        let stats = qlora.memory_stats();
        let savings_percent =
            (1.0 - stats.base_quantized_bytes as f32 / stats.base_unquantized_bytes as f32) * 100.0;

        assert!(
            savings_percent > 70.0,
            "Should save > 70% memory, got {}%",
            savings_percent
        );
    }
}
