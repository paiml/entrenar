//! LoRA (Low-Rank Adaptation) layer implementation
//!
//! LoRA enables parameter-efficient fine-tuning by adding trainable low-rank
//! decomposition matrices to frozen pretrained weights.
//!
//! For a frozen weight matrix W ∈ ℝ^(d_out × d_in), LoRA adds:
//! ΔW = B @ A where A ∈ ℝ^(r × d_in) and B ∈ ℝ^(d_out × r)
//!
//! Forward pass: y = (W + α·B·A) @ x = W@x + α·(B@(A@x))
//! where α is a scaling factor (typically alpha/r)

use crate::autograd::matmul;
use crate::Tensor;

/// LoRA layer: adds trainable low-rank adaptation to a frozen base weight
pub struct LoRALayer {
    /// Frozen base weight matrix stored as 1D [d_out * d_in]
    base_weight: Tensor,
    /// LoRA matrix A stored as 1D [r * d_in] - downprojection
    lora_a: Tensor,
    /// LoRA matrix B stored as 1D [d_out * r] - upprojection
    lora_b: Tensor,
    /// Output dimension
    d_out: usize,
    /// Input dimension
    d_in: usize,
    /// LoRA rank
    rank: usize,
    /// Scaling factor (alpha/rank)
    scale: f32,
    /// Whether the adapter is merged into base_weight
    merged: bool,
}

impl LoRALayer {
    /// Create a new LoRA layer
    ///
    /// # Arguments
    /// * `base_weight` - Frozen pretrained weight [d_out * d_in]
    /// * `d_out` - Output dimension
    /// * `d_in` - Input dimension
    /// * `rank` - LoRA rank (typically 4, 8, 16, 32, or 64)
    /// * `alpha` - LoRA scaling parameter (often same as rank)
    ///
    /// # Returns
    /// LoRA layer with randomly initialized A (Gaussian) and zero-initialized B
    pub fn new(base_weight: Tensor, d_out: usize, d_in: usize, rank: usize, alpha: f32) -> Self {
        assert_eq!(
            base_weight.len(),
            d_out * d_in,
            "Base weight size must match d_out * d_in"
        );

        // Initialize A with small Gaussian noise, B with zeros (standard LoRA init)
        // This ensures that initially ΔW = B·A = 0
        let lora_a_data: Vec<f32> = (0..rank * d_in)
            .map(|i| {
                // Simple deterministic "random" init for reproducibility in tests
                let x = (i as f32 * 0.1).sin();
                x * 0.01 // Small values
            })
            .collect();
        let lora_a = Tensor::from_vec(lora_a_data, true);

        let lora_b = Tensor::zeros(d_out * rank, true);

        let scale = alpha / rank as f32;

        Self {
            base_weight,
            lora_a,
            lora_b,
            d_out,
            d_in,
            rank,
            scale,
            merged: false,
        }
    }

    /// Forward pass: y = W@x + scale * (B @ (A @ x))
    ///
    /// # Arguments
    /// * `x` - Input tensor [d_in]
    ///
    /// # Returns
    /// Output tensor [d_out]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.len(), self.d_in, "Input size must match d_in");

        // Base forward: W @ x [d_out, d_in] @ [d_in, 1] -> [d_out, 1]
        let base_output = matmul(&self.base_weight, x, self.d_out, self.d_in, 1);

        if !self.merged {
            // LoRA forward: scale * (B @ (A @ x))
            // Step 1: A @ x [r, d_in] @ [d_in, 1] -> [r, 1]
            let lora_out_a = matmul(&self.lora_a, x, self.rank, self.d_in, 1);

            // Step 2: B @ (A @ x) [d_out, r] @ [r, 1] -> [d_out, 1]
            let lora_out_b = matmul(&self.lora_b, &lora_out_a, self.d_out, self.rank, 1);

            // Step 3: scale * LoRA output
            let mut scaled_lora_data = lora_out_b.data().to_owned();
            for val in scaled_lora_data.iter_mut() {
                *val *= self.scale;
            }
            let scaled_lora = Tensor::new(scaled_lora_data, false);

            // Step 4: base + LoRA
            let mut result_data = base_output.data().to_owned();
            for (i, val) in result_data.iter_mut().enumerate() {
                *val += scaled_lora.data()[i];
            }
            Tensor::new(result_data, base_output.requires_grad())
        } else {
            // If merged, W already includes LoRA adaptation
            base_output
        }
    }

    /// Merge LoRA weights into base weight: W' = W + scale * (B @ A)
    ///
    /// After merging, forward pass only uses W' (more efficient).
    /// This is typically done for inference.
    pub fn merge(&mut self) {
        if self.merged {
            return; // Already merged
        }

        // Compute B @ A [d_out, r] @ [r, d_in] -> [d_out, d_in]
        let ba = matmul(&self.lora_b, &self.lora_a, self.d_out, self.rank, self.d_in);

        // Scale and add to base weight: W' = W + scale * B @ A
        for (i, val) in self.base_weight.data_mut().iter_mut().enumerate() {
            *val += self.scale * ba.data()[i];
        }

        self.merged = true;
    }

    /// Unmerge LoRA weights from base weight: W = W' - scale * (B @ A)
    ///
    /// Reverses the merge operation. Useful for continuing training or
    /// switching adapters.
    pub fn unmerge(&mut self) {
        if !self.merged {
            return; // Not merged
        }

        // Compute B @ A
        let ba = matmul(&self.lora_b, &self.lora_a, self.d_out, self.rank, self.d_in);

        // Subtract from base weight: W = W' - scale * B @ A
        for (i, val) in self.base_weight.data_mut().iter_mut().enumerate() {
            *val -= self.scale * ba.data()[i];
        }

        self.merged = false;
    }

    /// Get reference to base weight matrix
    pub fn base_weight(&self) -> &Tensor {
        &self.base_weight
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

    /// Check if LoRA is merged
    pub fn is_merged(&self) -> bool {
        self.merged
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    // ========================================================================
    // PROPERTY TESTS
    // ========================================================================

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(200))]

        #[test]
        fn prop_zero_b_gives_base_output(
            d_out in 2usize..10,
            d_in in 2usize..10,
            rank in 1usize..5,
        ) {
            // When B is zeros, LoRA output should equal base output
            let size = d_out * d_in;
            let base_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
            let base_weight = Tensor::from_vec(base_data, false);
            let lora = LoRALayer::new(base_weight.clone(), d_out, d_in, rank, 1.0);

            // B is initialized to zeros by default
            let x_data: Vec<f32> = (0..d_in).map(|i| i as f32 * 0.5).collect();
            let x = Tensor::from_vec(x_data.clone(), true);

            let lora_output = lora.forward(&x);

            // Compute expected base output: W @ x
            let base_output = matmul(&base_weight, &Tensor::from_vec(x_data, false), d_out, d_in, 1);

            for i in 0..d_out {
                prop_assert!(
                    (lora_output.data()[i] - base_output.data()[i]).abs() < 1e-4,
                    "Zero B should give base output at index {}", i
                );
            }
        }

        #[test]
        fn prop_merge_preserves_forward_output(
            d_out in 2usize..8,
            d_in in 2usize..8,
            rank in 1usize..4,
        ) {
            let size = d_out * d_in;
            let base_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).cos()).collect();
            let base_weight = Tensor::from_vec(base_data, false);
            let mut lora = LoRALayer::new(base_weight, d_out, d_in, rank, 2.0);

            // Set non-zero LoRA weights
            let a_data: Vec<f32> = (0..rank * d_in).map(|i| (i as f32 * 0.2).sin() * 0.1).collect();
            let b_data: Vec<f32> = (0..d_out * rank).map(|i| (i as f32 * 0.3).cos() * 0.1).collect();
            *lora.lora_a_mut().data_mut() = ndarray::Array1::from_vec(a_data);
            *lora.lora_b_mut().data_mut() = ndarray::Array1::from_vec(b_data);

            let x_data: Vec<f32> = (0..d_in).map(|i| i as f32 + 1.0).collect();
            let x = Tensor::from_vec(x_data.clone(), true);

            // Forward before merge
            let output_before = lora.forward(&x);

            // Merge
            lora.merge();
            prop_assert!(lora.is_merged());

            // Forward after merge
            let x2 = Tensor::from_vec(x_data, true);
            let output_after = lora.forward(&x2);

            // Outputs should match
            for i in 0..d_out {
                prop_assert!(
                    (output_before.data()[i] - output_after.data()[i]).abs() < 1e-3,
                    "Merge should preserve output at index {}: before={} after={}",
                    i, output_before.data()[i], output_after.data()[i]
                );
            }
        }

        #[test]
        fn prop_unmerge_restores_weights(
            d_out in 2usize..8,
            d_in in 2usize..8,
            rank in 1usize..4,
        ) {
            let size = d_out * d_in;
            let base_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();
            let base_weight = Tensor::from_vec(base_data.clone(), false);
            let mut lora = LoRALayer::new(base_weight, d_out, d_in, rank, 1.0);

            // Set non-zero LoRA weights
            let a_data: Vec<f32> = (0..rank * d_in).map(|i| i as f32 * 0.01).collect();
            let b_data: Vec<f32> = (0..d_out * rank).map(|i| i as f32 * 0.02).collect();
            *lora.lora_a_mut().data_mut() = ndarray::Array1::from_vec(a_data);
            *lora.lora_b_mut().data_mut() = ndarray::Array1::from_vec(b_data);

            // Merge then unmerge
            lora.merge();
            lora.unmerge();

            // Base weights should be restored
            for i in 0..size {
                prop_assert!(
                    (lora.base_weight().data()[i] - base_data[i]).abs() < 1e-4,
                    "Unmerge should restore weight at index {}", i
                );
            }
        }

        #[test]
        fn prop_scale_factor_correct(
            rank in 1usize..32,
            alpha in 1.0f32..64.0,
        ) {
            let base_weight = Tensor::from_vec(vec![1.0], false);
            let lora = LoRALayer::new(base_weight, 1, 1, rank, alpha);

            let expected_scale = alpha / rank as f32;
            prop_assert!(
                (lora.scale() - expected_scale).abs() < 1e-6,
                "Scale should be alpha/rank: expected {} got {}", expected_scale, lora.scale()
            );
        }

        #[test]
        fn prop_lora_dimensions_correct(
            d_out in 2usize..20,
            d_in in 2usize..20,
            rank in 1usize..10,
        ) {
            let size = d_out * d_in;
            let base_data: Vec<f32> = vec![0.0; size];
            let base_weight = Tensor::from_vec(base_data, false);
            let lora = LoRALayer::new(base_weight, d_out, d_in, rank, 1.0);

            // Verify all dimensions
            prop_assert_eq!(lora.d_out(), d_out);
            prop_assert_eq!(lora.d_in(), d_in);
            prop_assert_eq!(lora.rank(), rank);
            prop_assert_eq!(lora.lora_a().len(), rank * d_in);
            prop_assert_eq!(lora.lora_b().len(), d_out * rank);
            prop_assert_eq!(lora.base_weight().len(), d_out * d_in);
        }
    }

    // ========================================================================
    // DETERMINISTIC UNIT TESTS
    // ========================================================================

    #[test]
    fn test_lora_layer_creation() {
        // 3x2 weight matrix
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false);
        let lora = LoRALayer::new(base_weight, 3, 2, 2, 2.0);

        assert_eq!(lora.rank(), 2);
        assert_eq!(lora.d_out(), 3);
        assert_eq!(lora.d_in(), 2);
        assert_abs_diff_eq!(lora.scale(), 1.0, epsilon = 1e-6); // alpha/rank = 2/2 = 1
        assert!(!lora.is_merged());

        // Check dimensions
        assert_eq!(lora.lora_a().len(), 2 * 2); // [r * d_in]
        assert_eq!(lora.lora_b().len(), 3 * 2); // [d_out * r]
    }

    #[test]
    fn test_lora_forward_unmerged() {
        // Simple 2x2 identity weight matrix
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        // Set LoRA weights to known values for testing
        // A: [1, 2] (1x2 matrix)
        *lora.lora_a_mut().data_mut() = ndarray::arr1(&[1.0, 2.0]);
        // B: [3, 4] (2x1 matrix) - stored as column-major [3, 4]
        *lora.lora_b_mut().data_mut() = ndarray::arr1(&[3.0, 4.0]);

        // Input vector [1, 2]
        let x = Tensor::from_vec(vec![1.0, 2.0], true);

        // Forward pass
        let output = lora.forward(&x);

        // Expected:
        // Base: [[1, 0], [0, 1]] @ [1, 2] = [1, 2]
        // LoRA:
        //   A @ x: [1, 2] @ [1, 2] = 1*1 + 2*2 = 5 (scalar)
        //   B @ (A@x): [[3], [4]] @ [5] = [15, 20]
        //   scale = 1.0, so LoRA output = [15, 20]
        // Total: [1, 2] + [15, 20] = [16, 22]
        assert_eq!(output.len(), 2);
        assert_abs_diff_eq!(output.data()[0], 16.0, epsilon = 1e-4);
        assert_abs_diff_eq!(output.data()[1], 22.0, epsilon = 1e-4);
    }

    #[test]
    fn test_lora_merge_unmerge() {
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        // Set LoRA weights: A = [1, 2], B = [0.5, 0.5]
        *lora.lora_a_mut().data_mut() = ndarray::arr1(&[1.0, 2.0]);
        *lora.lora_b_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);

        // Original base weight
        let original_weight = lora.base_weight.data().to_owned();

        // Merge
        lora.merge();
        assert!(lora.is_merged());

        // After merge, base weight should be W + scale * (B @ A)
        // B @ A = [[0.5], [0.5]] @ [[1, 2]] = [[0.5, 1.0], [0.5, 1.0]]
        //       = [0.5, 1.0, 0.5, 1.0] in row-major
        // scale = 1.0, so delta = [0.5, 1.0, 0.5, 1.0]
        // W' = [1, 0, 0, 1] + [0.5, 1.0, 0.5, 1.0] = [1.5, 1.0, 0.5, 2.0]
        let merged_weight = lora.base_weight.data();
        assert_abs_diff_eq!(merged_weight[0], 1.5, epsilon = 1e-4);
        assert_abs_diff_eq!(merged_weight[1], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(merged_weight[2], 0.5, epsilon = 1e-4);
        assert_abs_diff_eq!(merged_weight[3], 2.0, epsilon = 1e-4);

        // Unmerge
        lora.unmerge();
        assert!(!lora.is_merged());

        // Should restore original weight
        let restored_weight = lora.base_weight.data();
        for i in 0..4 {
            assert_abs_diff_eq!(restored_weight[i], original_weight[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_lora_forward_merged() {
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        *lora.lora_a_mut().data_mut() = ndarray::arr1(&[1.0, 1.0]);
        *lora.lora_b_mut().data_mut() = ndarray::arr1(&[1.0, 1.0]);

        let x = Tensor::from_vec(vec![1.0, 1.0], true);

        // Forward before merge
        let output_unmerged = lora.forward(&x);

        // Merge
        lora.merge();

        // Forward after merge - should give same result
        let output_merged = lora.forward(&x);

        assert_eq!(output_unmerged.len(), output_merged.len());
        for i in 0..output_unmerged.len() {
            assert_abs_diff_eq!(
                output_unmerged.data()[i],
                output_merged.data()[i],
                epsilon = 1e-4
            );
        }
    }

    #[test]
    fn test_lora_trainable_params() {
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
        let mut lora = LoRALayer::new(base_weight, 2, 2, 2, 4.0);

        let params = lora.trainable_params();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].len(), 2 * 2); // A: [r * d_in]
        assert_eq!(params[1].len(), 2 * 2); // B: [d_out * r]

        // All should require gradients
        assert!(params[0].requires_grad());
        assert!(params[1].requires_grad());
    }

    #[test]
    fn test_lora_zero_initialization() {
        // With B initialized to zeros, initial LoRA contribution should be zero
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let lora = LoRALayer::new(base_weight, 2, 2, 2, 2.0);

        let x = Tensor::from_vec(vec![2.0, 3.0], true);

        // Forward pass
        let output = lora.forward(&x);

        // Should match base forward since B is zeros
        // Base: [[1, 0], [0, 1]] @ [2, 3] = [2, 3]
        assert_abs_diff_eq!(output.data()[0], 2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(output.data()[1], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_lora_rank_scaling() {
        let base_weight = Tensor::from_vec(vec![1.0], false);

        // Different ranks with same alpha should give different scales
        let lora_r4 = LoRALayer::new(base_weight.clone(), 1, 1, 4, 8.0);
        let lora_r8 = LoRALayer::new(base_weight, 1, 1, 8, 8.0);

        assert_abs_diff_eq!(lora_r4.scale(), 2.0, epsilon = 1e-6); // 8/4 = 2
        assert_abs_diff_eq!(lora_r8.scale(), 1.0, epsilon = 1e-6); // 8/8 = 1
    }
}
