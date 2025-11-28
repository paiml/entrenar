//! PTQ (Post-Training Quantization) Calibration
//!
//! Calibration methods for determining quantization parameters (scale, zero_point)
//! from representative data:
//! - Min-Max: Uses the full range of observed values
//! - Percentile: Uses percentiles to be robust to outliers
//! - Moving Average: Smooths calibration over multiple batches

use crate::Tensor;

/// Calibration method for PTQ
#[derive(Clone, Debug, PartialEq, Default)]
pub enum CalibrationMethod {
    /// Min-max calibration: scale from actual min/max values
    #[default]
    MinMax,
    /// Percentile calibration: scale from percentile values (more robust to outliers)
    Percentile {
        /// Lower percentile (e.g., 0.01 for 0.01%)
        lower: f32,
        /// Upper percentile (e.g., 99.99 for 99.99%)
        upper: f32,
    },
    /// Moving average: smoothed min/max over multiple batches
    MovingAverage {
        /// Smoothing factor (0 = no smoothing, 1 = fully use new value)
        momentum: f32,
    },
}

/// Calibration result containing scale and zero_point
#[derive(Clone, Debug)]
pub struct CalibrationResult {
    /// Scale factor for quantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i32,
    /// Observed minimum value
    pub observed_min: f32,
    /// Observed maximum value
    pub observed_max: f32,
    /// Method used for calibration
    pub method: CalibrationMethod,
}

/// PTQ Calibrator for collecting statistics and computing quantization parameters
#[derive(Clone, Debug)]
pub struct Calibrator {
    /// Calibration method
    method: CalibrationMethod,
    /// Whether quantization is symmetric
    symmetric: bool,
    /// Number of bits for quantization
    bits: usize,
    /// Running minimum (for moving average)
    running_min: Option<f32>,
    /// Running maximum (for moving average)
    running_max: Option<f32>,
    /// Collected samples (for percentile)
    samples: Vec<f32>,
    /// Maximum samples to collect (for percentile)
    max_samples: usize,
    /// Number of batches observed
    num_batches: usize,
}

impl Calibrator {
    /// Create new calibrator with min-max method
    pub fn min_max(bits: usize, symmetric: bool) -> Self {
        Self {
            method: CalibrationMethod::MinMax,
            symmetric,
            bits,
            running_min: None,
            running_max: None,
            samples: Vec::new(),
            max_samples: 0,
            num_batches: 0,
        }
    }

    /// Create new calibrator with percentile method
    ///
    /// # Arguments
    /// * `bits` - Number of quantization bits
    /// * `symmetric` - Whether to use symmetric quantization
    /// * `lower` - Lower percentile (e.g., 0.01 for 0.01%)
    /// * `upper` - Upper percentile (e.g., 99.99 for 99.99%)
    /// * `max_samples` - Maximum number of samples to collect
    pub fn percentile(
        bits: usize,
        symmetric: bool,
        lower: f32,
        upper: f32,
        max_samples: usize,
    ) -> Self {
        Self {
            method: CalibrationMethod::Percentile { lower, upper },
            symmetric,
            bits,
            running_min: None,
            running_max: None,
            samples: Vec::with_capacity(max_samples.min(10000)),
            max_samples,
            num_batches: 0,
        }
    }

    /// Create new calibrator with moving average method
    pub fn moving_average(bits: usize, symmetric: bool, momentum: f32) -> Self {
        Self {
            method: CalibrationMethod::MovingAverage { momentum },
            symmetric,
            bits,
            running_min: None,
            running_max: None,
            samples: Vec::new(),
            max_samples: 0,
            num_batches: 0,
        }
    }

    /// Observe a batch of data for calibration
    pub fn observe(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        match &self.method {
            CalibrationMethod::MinMax => {
                self.observe_min_max(data);
            }
            CalibrationMethod::Percentile { .. } => {
                self.observe_percentile(data);
            }
            CalibrationMethod::MovingAverage { momentum } => {
                let momentum = *momentum;
                self.observe_moving_average(data, momentum);
            }
        }

        self.num_batches += 1;
    }

    /// Observe a tensor for calibration
    pub fn observe_tensor(&mut self, tensor: &Tensor) {
        if let Some(slice) = tensor.data().as_slice() {
            self.observe(slice);
        }
    }

    /// Observe multiple tensors
    pub fn observe_tensors(&mut self, tensors: &[&Tensor]) {
        for tensor in tensors {
            self.observe_tensor(tensor);
        }
    }

    /// Compute calibration result
    pub fn compute(&self) -> CalibrationResult {
        let (observed_min, observed_max) = match &self.method {
            CalibrationMethod::MinMax | CalibrationMethod::MovingAverage { .. } => (
                self.running_min.unwrap_or(0.0),
                self.running_max.unwrap_or(0.0),
            ),
            CalibrationMethod::Percentile { lower, upper } => {
                self.compute_percentile_bounds(*lower, *upper)
            }
        };

        let (scale, zero_point) = self.compute_scale_zero_point(observed_min, observed_max);

        CalibrationResult {
            scale,
            zero_point,
            observed_min,
            observed_max,
            method: self.method.clone(),
        }
    }

    /// Get number of batches observed
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    /// Get calibration method
    pub fn method(&self) -> &CalibrationMethod {
        &self.method
    }

    /// Check if any data has been observed
    pub fn has_data(&self) -> bool {
        self.num_batches > 0
    }

    /// Reset calibration state
    pub fn reset(&mut self) {
        self.running_min = None;
        self.running_max = None;
        self.samples.clear();
        self.num_batches = 0;
    }

    // Internal methods

    fn observe_min_max(&mut self, data: &[f32]) {
        let batch_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let batch_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        self.running_min = Some(
            self.running_min
                .map(|m| m.min(batch_min))
                .unwrap_or(batch_min),
        );
        self.running_max = Some(
            self.running_max
                .map(|m| m.max(batch_max))
                .unwrap_or(batch_max),
        );
    }

    fn observe_percentile(&mut self, data: &[f32]) {
        // Collect samples (with reservoir sampling if needed)
        if self.samples.len() < self.max_samples {
            let remaining = self.max_samples - self.samples.len();
            self.samples.extend(data.iter().take(remaining).cloned());
        } else {
            // Reservoir sampling for samples beyond max_samples
            let total_seen = self.num_batches * data.len() + data.len();
            for (i, &val) in data.iter().enumerate() {
                let j = rand_simple(total_seen + i);
                if j < self.max_samples {
                    self.samples[j] = val;
                }
            }
        }

        // Also track min/max for fallback
        self.observe_min_max(data);
    }

    fn observe_moving_average(&mut self, data: &[f32], momentum: f32) {
        let batch_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let batch_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        self.running_min = Some(
            self.running_min
                .map(|m| m * (1.0 - momentum) + batch_min * momentum)
                .unwrap_or(batch_min),
        );
        self.running_max = Some(
            self.running_max
                .map(|m| m * (1.0 - momentum) + batch_max * momentum)
                .unwrap_or(batch_max),
        );
    }

    fn compute_percentile_bounds(&self, lower: f32, upper: f32) -> (f32, f32) {
        if self.samples.is_empty() {
            return (
                self.running_min.unwrap_or(0.0),
                self.running_max.unwrap_or(0.0),
            );
        }

        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let lower_idx = ((lower / 100.0) * n as f32) as usize;
        let upper_idx = ((upper / 100.0) * n as f32).min((n - 1) as f32) as usize;

        (sorted[lower_idx], sorted[upper_idx])
    }

    fn compute_scale_zero_point(&self, min_val: f32, max_val: f32) -> (f32, i32) {
        let qmax = (1 << (self.bits - 1)) - 1;
        let qmin = if self.symmetric { -qmax } else { 0 };
        let qmax_full = if self.symmetric {
            qmax
        } else {
            (1 << self.bits) - 1
        };

        if self.symmetric {
            // Symmetric: scale from max absolute value
            let max_abs = min_val.abs().max(max_val.abs());
            let scale = if max_abs < 1e-10 {
                1e-10
            } else {
                max_abs / qmax as f32
            };
            (scale, 0)
        } else {
            // Asymmetric: scale from range
            let range = max_val - min_val;
            let scale = if range < 1e-10 {
                1e-10
            } else {
                range / (qmax_full - qmin) as f32
            };
            let zero_point = (qmin as f32 - min_val / scale).round() as i32;
            let zero_point = zero_point.clamp(qmin, qmax_full);
            (scale, zero_point)
        }
    }
}

/// Simple deterministic pseudo-random for reservoir sampling
fn rand_simple(seed: usize) -> usize {
    // Simple LCG-based PRNG
    let a: usize = 1103515245;
    let c: usize = 12345;
    let m: usize = 1 << 31;
    (a.wrapping_mul(seed).wrapping_add(c)) % m
}

/// Convenience function for min-max calibration
pub fn calibrate_min_max(data: &[f32], bits: usize, symmetric: bool) -> CalibrationResult {
    let mut calibrator = Calibrator::min_max(bits, symmetric);
    calibrator.observe(data);
    calibrator.compute()
}

/// Convenience function for percentile calibration
pub fn calibrate_percentile(
    data: &[f32],
    bits: usize,
    symmetric: bool,
    lower: f32,
    upper: f32,
) -> CalibrationResult {
    let mut calibrator = Calibrator::percentile(bits, symmetric, lower, upper, data.len());
    calibrator.observe(data);
    calibrator.compute()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    // ========================================================================
    // PROPERTY TESTS - Calibration correctness
    // ========================================================================

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(200))]

        /// Min-max calibration should capture the full range
        #[test]
        fn prop_min_max_captures_range(
            data in prop::collection::vec(-100.0f32..100.0, 10..100),
        ) {
            let result = calibrate_min_max(&data, 8, true);

            let actual_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let actual_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            prop_assert!((result.observed_min - actual_min).abs() < 1e-5);
            prop_assert!((result.observed_max - actual_max).abs() < 1e-5);
        }

        /// Symmetric calibration should have zero_point = 0
        #[test]
        fn prop_symmetric_zero_point(
            data in prop::collection::vec(-10.0f32..10.0, 10..50),
            bits in 4usize..9,
        ) {
            let result = calibrate_min_max(&data, bits, true);
            prop_assert_eq!(result.zero_point, 0);
        }

        /// Scale should be positive and reasonable
        #[test]
        fn prop_scale_positive(
            data in prop::collection::vec(-10.0f32..10.0, 10..50),
            bits in 4usize..9,
        ) {
            let result = calibrate_min_max(&data, bits, true);

            prop_assert!(result.scale > 0.0);
            prop_assert!(result.scale < 1e10);
        }

        /// Percentile calibration should produce bounds within data range
        #[test]
        fn prop_percentile_within_range(
            data in prop::collection::vec(-10.0f32..10.0, 100..500),
        ) {
            let result = calibrate_percentile(&data, 8, true, 1.0, 99.0);

            let actual_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let actual_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Percentile bounds should be within actual range
            prop_assert!(result.observed_min >= actual_min - 1e-5);
            prop_assert!(result.observed_max <= actual_max + 1e-5);
        }

        /// Multiple batch observation should accumulate correctly
        #[test]
        fn prop_multi_batch_accumulates(
            batch1 in prop::collection::vec(-5.0f32..5.0, 10..30),
            batch2 in prop::collection::vec(-10.0f32..10.0, 10..30),
        ) {
            let mut calibrator = Calibrator::min_max(8, true);
            calibrator.observe(&batch1);
            calibrator.observe(&batch2);

            let result = calibrator.compute();

            let all_data: Vec<f32> = batch1.iter().chain(batch2.iter()).cloned().collect();
            let expected_min = all_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let expected_max = all_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            prop_assert!((result.observed_min - expected_min).abs() < 1e-5);
            prop_assert!((result.observed_max - expected_max).abs() < 1e-5);
            prop_assert_eq!(calibrator.num_batches(), 2);
        }
    }

    // ========================================================================
    // UNIT TESTS
    // ========================================================================

    #[test]
    fn test_min_max_calibration() {
        let data = vec![0.0, 1.0, -2.0, 1.5, -1.5, 3.0];
        let result = calibrate_min_max(&data, 8, true);

        assert_abs_diff_eq!(result.observed_min, -2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.observed_max, 3.0, epsilon = 1e-6);
        assert_eq!(result.zero_point, 0);

        // Scale = max_abs / qmax = 3.0 / 127
        let expected_scale = 3.0 / 127.0;
        assert_abs_diff_eq!(result.scale, expected_scale, epsilon = 1e-6);
    }

    #[test]
    fn test_percentile_calibration() {
        // Create data with outliers
        let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        data.push(1000.0); // Outlier
        data.push(-1000.0); // Outlier

        let result = calibrate_percentile(&data, 8, true, 1.0, 99.0);

        // Percentile should ignore outliers
        // 1% of 102 ≈ 1, 99% ≈ 100
        // So bounds should be close to 0.1 and 9.9 (not -1000 and 1000)
        assert!(
            result.observed_min > -100.0,
            "Should ignore negative outlier"
        );
        assert!(
            result.observed_max < 100.0,
            "Should ignore positive outlier"
        );
    }

    #[test]
    fn test_moving_average_calibration() {
        let mut calibrator = Calibrator::moving_average(8, true, 0.5);

        calibrator.observe(&[0.0, 1.0, -1.0]); // min=-1, max=1
        let r1 = calibrator.compute();
        assert_abs_diff_eq!(r1.observed_min, -1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(r1.observed_max, 1.0, epsilon = 1e-5);

        calibrator.observe(&[0.0, 2.0, -2.0]); // batch min=-2, max=2
        let r2 = calibrator.compute();
        // With momentum=0.5: new_min = -1*0.5 + -2*0.5 = -1.5
        assert_abs_diff_eq!(r2.observed_min, -1.5, epsilon = 1e-5);
        assert_abs_diff_eq!(r2.observed_max, 1.5, epsilon = 1e-5);
    }

    #[test]
    fn test_asymmetric_calibration() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // All positive
        let result = calibrate_min_max(&data, 8, false);

        assert_abs_diff_eq!(result.observed_min, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.observed_max, 4.0, epsilon = 1e-6);

        // Asymmetric should have non-zero zero_point
        // scale = (4-0) / 255 ≈ 0.0157
        // zero_point = round(0 - 0/scale) = 0
        assert!(result.scale > 0.0);
    }

    #[test]
    fn test_calibrator_reset() {
        let mut calibrator = Calibrator::min_max(8, true);
        calibrator.observe(&[1.0, 2.0, 3.0]);
        assert!(calibrator.has_data());

        calibrator.reset();
        assert!(!calibrator.has_data());
        assert_eq!(calibrator.num_batches(), 0);
    }

    #[test]
    fn test_calibrator_observe_tensor() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], false);
        let mut calibrator = Calibrator::min_max(8, true);

        calibrator.observe_tensor(&tensor);

        let result = calibrator.compute();
        assert_abs_diff_eq!(result.observed_min, -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.observed_max, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_calibration_method_default() {
        let method = CalibrationMethod::default();
        assert_eq!(method, CalibrationMethod::MinMax);
    }

    #[test]
    fn test_calibration_with_zeros() {
        let data = vec![0.0; 100];
        let result = calibrate_min_max(&data, 8, true);

        // Should handle zero data without division by zero
        assert!(result.scale > 0.0);
        assert!(result.scale.is_finite());
    }

    #[test]
    fn test_calibration_single_value() {
        let data = vec![5.0; 50];
        let result = calibrate_min_max(&data, 8, true);

        // Single value should work
        assert_abs_diff_eq!(result.observed_min, 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.observed_max, 5.0, epsilon = 1e-6);
        assert!(result.scale.is_finite());
    }

    #[test]
    fn test_4bit_calibration() {
        let data = vec![0.0, 1.0, -1.0];
        let result = calibrate_min_max(&data, 4, true);

        // 4-bit symmetric: qmax = 7
        let expected_scale = 1.0 / 7.0;
        assert_abs_diff_eq!(result.scale, expected_scale, epsilon = 1e-6);
    }
}
