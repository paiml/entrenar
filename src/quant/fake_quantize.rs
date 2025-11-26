//! Fake Quantization for Quantization-Aware Training (QAT)
//!
//! Fake quantization simulates the effects of quantization during training:
//! - Forward: quantize → dequantize (simulates quantization noise)
//! - Backward: Straight-Through Estimator (STE) passes gradients unchanged
//!
//! This allows models to adapt to quantization noise during training,
//! resulting in better accuracy after actual quantization.

use crate::Tensor;

/// Fake quantization configuration
#[derive(Clone, Debug)]
pub struct FakeQuantConfig {
    /// Number of bits for quantization (e.g., 4, 8)
    pub bits: usize,
    /// Whether quantization is symmetric (centered at 0)
    pub symmetric: bool,
    /// Quantization range: min value
    pub qmin: i32,
    /// Quantization range: max value
    pub qmax: i32,
}

impl FakeQuantConfig {
    /// Create symmetric fake quantization config
    ///
    /// # Arguments
    /// * `bits` - Number of bits (4-bit: qmin=-7, qmax=7; 8-bit: qmin=-127, qmax=127)
    pub fn symmetric(bits: usize) -> Self {
        let qmax = (1 << (bits - 1)) - 1; // 2^(bits-1) - 1
        let qmin = -qmax;
        Self {
            bits,
            symmetric: true,
            qmin,
            qmax,
        }
    }

    /// Create asymmetric fake quantization config
    ///
    /// # Arguments
    /// * `bits` - Number of bits (4-bit: qmin=0, qmax=15; 8-bit: qmin=0, qmax=255)
    pub fn asymmetric(bits: usize) -> Self {
        let qmax = (1 << bits) - 1; // 2^bits - 1
        Self {
            bits,
            symmetric: false,
            qmin: 0,
            qmax,
        }
    }

    /// 4-bit symmetric quantization
    pub fn q4_symmetric() -> Self {
        Self::symmetric(4)
    }

    /// 8-bit symmetric quantization
    pub fn q8_symmetric() -> Self {
        Self::symmetric(8)
    }
}

impl Default for FakeQuantConfig {
    fn default() -> Self {
        Self::q8_symmetric()
    }
}

/// Fake quantization operation with Straight-Through Estimator (STE)
///
/// This struct holds the state for fake quantization including learned
/// or calibrated scale and zero_point parameters.
#[derive(Clone, Debug)]
pub struct FakeQuantize {
    /// Quantization configuration
    pub config: FakeQuantConfig,
    /// Scale factor for quantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i32,
    /// Whether scale has been initialized
    pub initialized: bool,
}

impl FakeQuantize {
    /// Create new fake quantization operation
    pub fn new(config: FakeQuantConfig) -> Self {
        Self {
            config,
            scale: 1.0,
            zero_point: 0,
            initialized: false,
        }
    }

    /// Create with 4-bit symmetric quantization
    pub fn q4() -> Self {
        Self::new(FakeQuantConfig::q4_symmetric())
    }

    /// Create with 8-bit symmetric quantization
    pub fn q8() -> Self {
        Self::new(FakeQuantConfig::q8_symmetric())
    }

    /// Initialize scale from data (min-max calibration)
    ///
    /// For symmetric: scale = max(|min|, |max|) / qmax
    /// For asymmetric: scale = (max - min) / (qmax - qmin)
    pub fn calibrate(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if self.config.symmetric {
            // Symmetric: scale from max absolute value
            let max_abs = min_val.abs().max(max_val.abs());
            self.scale = max_abs / self.config.qmax as f32;
            self.zero_point = 0;
        } else {
            // Asymmetric: scale from range
            self.scale = (max_val - min_val) / (self.config.qmax - self.config.qmin) as f32;
            self.zero_point =
                (self.config.qmin as f32 - min_val / self.scale).round() as i32;
            self.zero_point = self.zero_point.clamp(self.config.qmin, self.config.qmax);
        }

        // Prevent division by zero
        if self.scale < 1e-10 {
            self.scale = 1e-10;
        }

        self.initialized = true;
    }

    /// Forward pass: fake quantize (quantize → dequantize)
    ///
    /// Simulates quantization effects while keeping values in floating point.
    /// Output = dequantize(quantize(input))
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let data: Vec<f32> = input
            .data()
            .iter()
            .map(|&x| self.fake_quantize_value(x))
            .collect();

        Tensor::new(ndarray::arr1(&data), input.requires_grad())
    }

    /// Forward pass with auto-calibration
    ///
    /// If not initialized, calibrates from input data first.
    pub fn forward_with_calibration(&mut self, input: &Tensor) -> Tensor {
        if !self.initialized {
            self.calibrate(input.data().as_slice().unwrap());
        }
        self.forward(input)
    }

    /// Backward pass: Straight-Through Estimator (STE)
    ///
    /// The gradient passes through unchanged:
    /// ∂L/∂x = ∂L/∂y (where y = fake_quantize(x))
    ///
    /// This allows gradients to flow during training despite the
    /// non-differentiable quantization operation.
    pub fn backward(&self, grad_output: &Tensor) -> Tensor {
        // STE: gradient passes through unchanged
        grad_output.clone()
    }

    /// Backward pass with gradient clipping (clamped STE)
    ///
    /// Clips gradients to zero outside the quantization range.
    /// This can improve training stability.
    pub fn backward_clamped(&self, grad_output: &Tensor, input: &Tensor) -> Tensor {
        let qmin_float = self.config.qmin as f32 * self.scale;
        let qmax_float = self.config.qmax as f32 * self.scale;

        let data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(input.data().iter())
            .map(|(&grad, &x)| {
                // Zero gradient outside quantization range
                if x < qmin_float || x > qmax_float {
                    0.0
                } else {
                    grad
                }
            })
            .collect();

        Tensor::new(ndarray::arr1(&data), grad_output.requires_grad())
    }

    /// Fake quantize a single value
    fn fake_quantize_value(&self, x: f32) -> f32 {
        // Quantize
        let q = if self.config.symmetric {
            (x / self.scale).round().clamp(
                self.config.qmin as f32,
                self.config.qmax as f32,
            ) as i32
        } else {
            ((x / self.scale) + self.zero_point as f32)
                .round()
                .clamp(self.config.qmin as f32, self.config.qmax as f32)
                as i32
        };

        // Dequantize
        if self.config.symmetric {
            q as f32 * self.scale
        } else {
            (q - self.zero_point) as f32 * self.scale
        }
    }

    /// Get the quantization scale
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get the zero point
    pub fn zero_point(&self) -> i32 {
        self.zero_point
    }

    /// Check if calibrated
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get number of quantization levels
    pub fn num_levels(&self) -> usize {
        (self.config.qmax - self.config.qmin + 1) as usize
    }
}

/// Convenience function for fake quantization forward pass
pub fn fake_quantize(input: &Tensor, bits: usize, symmetric: bool) -> Tensor {
    let config = if symmetric {
        FakeQuantConfig::symmetric(bits)
    } else {
        FakeQuantConfig::asymmetric(bits)
    };
    let mut fq = FakeQuantize::new(config);
    fq.forward_with_calibration(input)
}

/// Convenience function for STE backward pass
pub fn ste_backward(grad_output: &Tensor) -> Tensor {
    // STE: gradient passes through unchanged
    grad_output.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    // ========================================================================
    // PROPERTY TESTS - Fake quantization correctness
    // ========================================================================

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(200))]

        /// STE backward should always pass gradients unchanged
        #[test]
        fn prop_ste_backward_identity(
            grad in prop::collection::vec(-10.0f32..10.0, 1..32),
        ) {
            let grad_tensor = Tensor::from_vec(grad.clone(), true);
            let fq = FakeQuantize::q8();

            let backward = fq.backward(&grad_tensor);

            // STE should pass through unchanged
            prop_assert_eq!(backward.len(), grad.len());
            for (i, &g) in grad.iter().enumerate() {
                prop_assert!(
                    (backward.data()[i] - g).abs() < 1e-6,
                    "STE should preserve gradient at index {}", i
                );
            }
        }

        /// Fake quantize should produce values that are multiples of scale
        #[test]
        fn prop_fake_quantize_produces_quantized_values(
            values in prop::collection::vec(-5.0f32..5.0, 4..32),
            bits in 4usize..9,
        ) {
            let input = Tensor::from_vec(values.clone(), false);
            let config = FakeQuantConfig::symmetric(bits);
            let mut fq = FakeQuantize::new(config);
            fq.calibrate(&values);

            let output = fq.forward(&input);

            // Each output value should be a valid quantized level
            let scale = fq.scale();
            for &val in output.data().iter() {
                // Value should be approximately q * scale for some integer q
                let q = (val / scale).round();
                let reconstructed = q * scale;
                prop_assert!(
                    (val - reconstructed).abs() < 1e-5,
                    "Value {} should be quantized (q={}, scale={})",
                    val, q, scale
                );
            }
        }

        /// Fake quantize output should be bounded by qmin*scale and qmax*scale
        #[test]
        fn prop_fake_quantize_bounded_output(
            values in prop::collection::vec(-100.0f32..100.0, 4..32),
            bits in 4usize..9,
        ) {
            let input = Tensor::from_vec(values.clone(), false);
            let config = FakeQuantConfig::symmetric(bits);
            let mut fq = FakeQuantize::new(config);
            fq.calibrate(&values);

            let output = fq.forward(&input);

            let qmin_float = fq.config.qmin as f32 * fq.scale();
            let qmax_float = fq.config.qmax as f32 * fq.scale();

            for &val in output.data().iter() {
                prop_assert!(
                    val >= qmin_float - 1e-5 && val <= qmax_float + 1e-5,
                    "Output {} should be in [{}, {}]",
                    val, qmin_float, qmax_float
                );
            }
        }

        /// Calibration should set scale based on data range
        #[test]
        fn prop_calibration_sets_appropriate_scale(
            values in prop::collection::vec(-10.0f32..10.0, 4..32),
            bits in 4usize..9,
        ) {
            let config = FakeQuantConfig::symmetric(bits);
            let mut fq = FakeQuantize::new(config);

            fq.calibrate(&values);

            prop_assert!(fq.is_initialized());
            prop_assert!(fq.scale() > 0.0);

            // For symmetric, scale should be max_abs / qmax
            let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let expected_scale = max_abs / fq.config.qmax as f32;

            // Allow small tolerance for numerical precision
            if max_abs > 1e-8 {
                prop_assert!(
                    (fq.scale() - expected_scale).abs() < 1e-5,
                    "Scale {} should be {} (max_abs={}, qmax={})",
                    fq.scale(), expected_scale, max_abs, fq.config.qmax
                );
            }
        }

        /// Number of quantization levels should be correct
        #[test]
        fn prop_num_levels_correct(bits in 2usize..10) {
            let config = FakeQuantConfig::symmetric(bits);
            let fq = FakeQuantize::new(config);

            // Symmetric: qmin = -(2^(bits-1)-1), qmax = 2^(bits-1)-1
            // Levels = qmax - qmin + 1 = 2 * (2^(bits-1)-1) + 1 = 2^bits - 1
            let expected = (1 << bits) - 1;
            prop_assert_eq!(fq.num_levels(), expected);
        }
    }

    // ========================================================================
    // UNIT TESTS
    // ========================================================================

    #[test]
    fn test_fake_quantize_config_symmetric() {
        let config = FakeQuantConfig::symmetric(4);
        assert_eq!(config.bits, 4);
        assert!(config.symmetric);
        assert_eq!(config.qmin, -7);
        assert_eq!(config.qmax, 7);

        let config8 = FakeQuantConfig::symmetric(8);
        assert_eq!(config8.qmin, -127);
        assert_eq!(config8.qmax, 127);
    }

    #[test]
    fn test_fake_quantize_config_asymmetric() {
        let config = FakeQuantConfig::asymmetric(4);
        assert_eq!(config.bits, 4);
        assert!(!config.symmetric);
        assert_eq!(config.qmin, 0);
        assert_eq!(config.qmax, 15);

        let config8 = FakeQuantConfig::asymmetric(8);
        assert_eq!(config8.qmin, 0);
        assert_eq!(config8.qmax, 255);
    }

    #[test]
    fn test_fake_quantize_forward() {
        let input = Tensor::from_vec(vec![0.0, 1.0, -1.0, 0.5, -0.5], false);
        let mut fq = FakeQuantize::q8();
        fq.calibrate(input.data().as_slice().unwrap());

        let output = fq.forward(&input);

        assert_eq!(output.len(), 5);
        // 0 should stay 0
        assert_abs_diff_eq!(output.data()[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_fake_quantize_forward_with_calibration() {
        let input = Tensor::from_vec(vec![0.0, 1.0, -1.0, 0.5, -0.5], false);
        let mut fq = FakeQuantize::q8();

        assert!(!fq.is_initialized());

        let output = fq.forward_with_calibration(&input);

        assert!(fq.is_initialized());
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_ste_backward() {
        let grad = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
        let fq = FakeQuantize::q8();

        let backward = fq.backward(&grad);

        // STE: gradient should pass through unchanged
        assert_eq!(backward.len(), 4);
        for i in 0..4 {
            assert_abs_diff_eq!(backward.data()[i], grad.data()[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_clamped_ste_backward() {
        let grad = Tensor::from_vec(vec![1.0, 1.0, 1.0], true);
        let input = Tensor::from_vec(vec![0.5, 10.0, -10.0], false); // 10, -10 outside range

        let mut fq = FakeQuantize::q4();
        fq.scale = 1.0; // Set scale so range is [-7, 7]
        fq.initialized = true;

        let backward = fq.backward_clamped(&grad, &input);

        // 0.5 is in range: gradient passes
        assert_abs_diff_eq!(backward.data()[0], 1.0, epsilon = 1e-6);
        // 10.0 is outside range: gradient clipped to 0
        assert_abs_diff_eq!(backward.data()[1], 0.0, epsilon = 1e-6);
        // -10.0 is outside range: gradient clipped to 0
        assert_abs_diff_eq!(backward.data()[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_calibration_symmetric() {
        let mut fq = FakeQuantize::q8();
        let data = vec![0.0, 1.0, -2.0, 1.5, -1.5];

        fq.calibrate(&data);

        // max_abs = 2.0, qmax = 127
        // scale = 2.0 / 127
        let expected_scale = 2.0 / 127.0;
        assert_abs_diff_eq!(fq.scale(), expected_scale, epsilon = 1e-6);
        assert_eq!(fq.zero_point(), 0);
    }

    #[test]
    fn test_fake_quantize_convenience_function() {
        let input = Tensor::from_vec(vec![0.0, 1.0, -1.0], false);

        let output = fake_quantize(&input, 8, true);

        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_ste_backward_convenience_function() {
        let grad = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);

        let backward = ste_backward(&grad);

        for i in 0..3 {
            assert_abs_diff_eq!(backward.data()[i], grad.data()[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_num_levels() {
        let fq4 = FakeQuantize::q4();
        assert_eq!(fq4.num_levels(), 15); // -7 to 7 = 15 levels

        let fq8 = FakeQuantize::q8();
        assert_eq!(fq8.num_levels(), 255); // -127 to 127 = 255 levels
    }

    #[test]
    fn test_quantize_dequantize_round_trip() {
        let input = Tensor::from_vec(vec![0.0, 0.5, 1.0, -0.5, -1.0], false);
        let mut fq = FakeQuantize::q8();
        fq.calibrate(input.data().as_slice().unwrap());

        let output = fq.forward(&input);

        // Output should be close to input (with quantization noise)
        for (i, (&orig, &out)) in input.data().iter().zip(output.data().iter()).enumerate() {
            let error = (orig - out).abs();
            assert!(
                error < 0.1,
                "Error {} at index {} too large: {} vs {}",
                error,
                i,
                orig,
                out
            );
        }
    }
}
