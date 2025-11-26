//! Per-channel vs Per-tensor Quantization Granularity
//!
//! Provides quantization at different granularities:
//! - **Per-tensor**: Single scale/zero-point for entire tensor (fastest, least accurate)
//! - **Per-channel**: Separate scale/zero-point per channel (slower, more accurate)
//! - **Per-group**: Scale/zero-point per group of values (balance of speed/accuracy)
//!
//! Per-channel is critical for weight quantization where channels have different ranges.

use serde::{Deserialize, Serialize};

/// Quantization granularity options
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantGranularity {
    /// Single scale/zero-point for entire tensor
    #[default]
    PerTensor,
    /// Separate scale/zero-point per channel (axis 0 for weights)
    PerChannel,
    /// Separate scale/zero-point per group of n elements
    PerGroup(usize),
}

/// Quantization mode: symmetric or asymmetric
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantMode {
    /// Symmetric: zero-point = 0, range = [-max_abs, max_abs]
    #[default]
    Symmetric,
    /// Asymmetric: zero-point != 0, range = [min, max]
    Asymmetric,
}

/// Quantization parameters for a tensor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantParams {
    /// Scale factor(s)
    pub scales: Vec<f32>,
    /// Zero point(s) - empty for symmetric quantization
    pub zero_points: Vec<i32>,
    /// Quantization granularity
    pub granularity: QuantGranularity,
    /// Quantization mode
    pub mode: QuantMode,
    /// Bit width (4 or 8)
    pub bits: u8,
}

impl QuantParams {
    /// Get number of scale/zero-point groups
    pub fn num_groups(&self) -> usize {
        self.scales.len()
    }

    /// Check if asymmetric quantization
    pub fn is_asymmetric(&self) -> bool {
        self.mode == QuantMode::Asymmetric
    }
}

/// Quantized tensor with per-channel or per-tensor quantization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// Quantized integer data (8-bit representation)
    pub data: Vec<i8>,
    /// Quantization parameters
    pub params: QuantParams,
    /// Original shape
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let data_bytes = self.data.len();
        let scale_bytes = self.params.scales.len() * 4;
        let zp_bytes = self.params.zero_points.len() * 4;
        data_bytes + scale_bytes + zp_bytes
    }
}

/// Calibrate quantization parameters for per-tensor quantization
///
/// # Arguments
/// * `values` - Input tensor values
/// * `bits` - Bit width (4 or 8)
/// * `mode` - Symmetric or asymmetric quantization
pub fn calibrate_per_tensor(values: &[f32], bits: u8, mode: QuantMode) -> QuantParams {
    let (scale, zero_point) = match mode {
        QuantMode::Symmetric => {
            let max_abs = values
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(1e-8)
                .max(1e-8);

            let qmax = ((1i32 << (bits - 1)) - 1) as f32;
            let scale = max_abs / qmax;
            (scale, 0)
        }
        QuantMode::Asymmetric => {
            let (min_val, max_val) = values.iter().fold((f32::MAX, f32::MIN), |(min, max), &v| {
                (min.min(v), max.max(v))
            });

            let range = (max_val - min_val).max(1e-8);
            let qmax = ((1i32 << bits) - 1) as f32;
            let scale = range / qmax;
            let zero_point = ((-min_val / scale).round() as i32).clamp(0, qmax as i32);
            (scale, zero_point)
        }
    };

    QuantParams {
        scales: vec![scale],
        zero_points: if mode == QuantMode::Asymmetric {
            vec![zero_point]
        } else {
            vec![]
        },
        granularity: QuantGranularity::PerTensor,
        mode,
        bits,
    }
}

/// Calibrate quantization parameters for per-channel quantization
///
/// # Arguments
/// * `values` - Input tensor values (row-major: [channels, features])
/// * `num_channels` - Number of channels (first dimension)
/// * `bits` - Bit width (4 or 8)
/// * `mode` - Symmetric or asymmetric quantization
pub fn calibrate_per_channel(
    values: &[f32],
    num_channels: usize,
    bits: u8,
    mode: QuantMode,
) -> QuantParams {
    if num_channels == 0 || values.is_empty() {
        return QuantParams {
            scales: vec![1.0],
            zero_points: if mode == QuantMode::Asymmetric {
                vec![0]
            } else {
                vec![]
            },
            granularity: QuantGranularity::PerChannel,
            mode,
            bits,
        };
    }

    let features_per_channel = values.len() / num_channels;
    let qmax_signed = ((1i32 << (bits - 1)) - 1) as f32;
    let qmax_unsigned = ((1i32 << bits) - 1) as f32;

    let mut scales = Vec::with_capacity(num_channels);
    let mut zero_points = Vec::with_capacity(num_channels);

    for ch in 0..num_channels {
        let start = ch * features_per_channel;
        let end = start + features_per_channel;
        let channel_values = &values[start..end];

        match mode {
            QuantMode::Symmetric => {
                let max_abs = channel_values
                    .iter()
                    .map(|v| v.abs())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(1e-8)
                    .max(1e-8);

                scales.push(max_abs / qmax_signed);
            }
            QuantMode::Asymmetric => {
                let (min_val, max_val) =
                    channel_values
                        .iter()
                        .fold((f32::MAX, f32::MIN), |(min, max), &v| {
                            (min.min(v), max.max(v))
                        });

                let range = (max_val - min_val).max(1e-8);
                let scale = range / qmax_unsigned;
                let zp = ((-min_val / scale).round() as i32).clamp(0, qmax_unsigned as i32);

                scales.push(scale);
                zero_points.push(zp);
            }
        }
    }

    QuantParams {
        scales,
        zero_points,
        granularity: QuantGranularity::PerChannel,
        mode,
        bits,
    }
}

/// Calibrate quantization parameters for per-group quantization
///
/// # Arguments
/// * `values` - Input tensor values
/// * `group_size` - Number of elements per group
/// * `bits` - Bit width (4 or 8)
/// * `mode` - Symmetric or asymmetric quantization
pub fn calibrate_per_group(
    values: &[f32],
    group_size: usize,
    bits: u8,
    mode: QuantMode,
) -> QuantParams {
    let group_size = group_size.max(1);
    let num_groups = values.len().div_ceil(group_size);
    let qmax_signed = ((1i32 << (bits - 1)) - 1) as f32;
    let qmax_unsigned = ((1i32 << bits) - 1) as f32;

    let mut scales = Vec::with_capacity(num_groups);
    let mut zero_points = Vec::with_capacity(num_groups);

    for group_idx in 0..num_groups {
        let start = group_idx * group_size;
        let end = (start + group_size).min(values.len());
        let group_values = &values[start..end];

        match mode {
            QuantMode::Symmetric => {
                let max_abs = group_values
                    .iter()
                    .map(|v| v.abs())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(1e-8)
                    .max(1e-8);

                scales.push(max_abs / qmax_signed);
            }
            QuantMode::Asymmetric => {
                let (min_val, max_val) =
                    group_values
                        .iter()
                        .fold((f32::MAX, f32::MIN), |(min, max), &v| {
                            (min.min(v), max.max(v))
                        });

                let range = (max_val - min_val).max(1e-8);
                let scale = range / qmax_unsigned;
                let zp = ((-min_val / scale).round() as i32).clamp(0, qmax_unsigned as i32);

                scales.push(scale);
                zero_points.push(zp);
            }
        }
    }

    QuantParams {
        scales,
        zero_points,
        granularity: QuantGranularity::PerGroup(group_size),
        mode,
        bits,
    }
}

/// Quantize values using given parameters
///
/// # Arguments
/// * `values` - Input f32 values
/// * `params` - Quantization parameters
pub fn quantize_with_params(values: &[f32], params: &QuantParams) -> Vec<i8> {
    let qmax_signed = ((1i32 << (params.bits - 1)) - 1) as f32;
    let qmin_signed = -qmax_signed - 1.0;
    let qmax_unsigned = ((1i32 << params.bits) - 1) as f32;

    let group_size = match params.granularity {
        QuantGranularity::PerTensor => values.len(),
        QuantGranularity::PerChannel => values.len() / params.scales.len().max(1),
        QuantGranularity::PerGroup(size) => size,
    };

    let mut result = Vec::with_capacity(values.len());

    for (i, &val) in values.iter().enumerate() {
        let group_idx = i / group_size.max(1);
        let scale = params.scales.get(group_idx).copied().unwrap_or(1.0);

        let q_val = match params.mode {
            QuantMode::Symmetric => (val / scale).round().clamp(qmin_signed, qmax_signed) as i8,
            QuantMode::Asymmetric => {
                let zp = params.zero_points.get(group_idx).copied().unwrap_or(0);
                let q = (val / scale + zp as f32).round().clamp(0.0, qmax_unsigned);
                // Store as signed for uniform representation
                (q as i32 - 128) as i8
            }
        };

        result.push(q_val);
    }

    result
}

/// Dequantize values using given parameters
///
/// # Arguments
/// * `quantized` - Quantized i8 values
/// * `params` - Quantization parameters
pub fn dequantize_with_params(quantized: &[i8], params: &QuantParams) -> Vec<f32> {
    let group_size = match params.granularity {
        QuantGranularity::PerTensor => quantized.len(),
        QuantGranularity::PerChannel => quantized.len() / params.scales.len().max(1),
        QuantGranularity::PerGroup(size) => size,
    };

    let mut result = Vec::with_capacity(quantized.len());

    for (i, &q_val) in quantized.iter().enumerate() {
        let group_idx = i / group_size.max(1);
        let scale = params.scales.get(group_idx).copied().unwrap_or(1.0);

        let val = match params.mode {
            QuantMode::Symmetric => (q_val as f32) * scale,
            QuantMode::Asymmetric => {
                let zp = params.zero_points.get(group_idx).copied().unwrap_or(0);
                // Convert back from signed storage
                let q_unsigned = (q_val as i32 + 128) as f32;
                (q_unsigned - zp as f32) * scale
            }
        };

        result.push(val);
    }

    result
}

/// Quantize tensor with specified granularity
///
/// # Arguments
/// * `values` - Input tensor values
/// * `shape` - Tensor shape
/// * `granularity` - Quantization granularity
/// * `mode` - Quantization mode
/// * `bits` - Bit width (4 or 8)
pub fn quantize_tensor(
    values: &[f32],
    shape: &[usize],
    granularity: QuantGranularity,
    mode: QuantMode,
    bits: u8,
) -> QuantizedTensor {
    let params = match granularity {
        QuantGranularity::PerTensor => calibrate_per_tensor(values, bits, mode),
        QuantGranularity::PerChannel => {
            let num_channels = shape.first().copied().unwrap_or(1);
            calibrate_per_channel(values, num_channels, bits, mode)
        }
        QuantGranularity::PerGroup(group_size) => calibrate_per_group(values, group_size, bits, mode),
    };

    let data = quantize_with_params(values, &params);

    QuantizedTensor {
        data,
        params,
        shape: shape.to_vec(),
    }
}

/// Dequantize tensor
pub fn dequantize_tensor(quantized: &QuantizedTensor) -> Vec<f32> {
    dequantize_with_params(&quantized.data, &quantized.params)
}

/// Compute quantization error (MSE)
pub fn quantization_mse(original: &[f32], dequantized: &[f32]) -> f32 {
    if original.len() != dequantized.len() || original.is_empty() {
        return f32::MAX;
    }

    let sum_sq: f32 = original
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    sum_sq / original.len() as f32
}

/// Compare per-channel vs per-tensor quantization error
///
/// # Arguments
/// * `values` - Input tensor values (row-major)
/// * `num_channels` - Number of channels
/// * `bits` - Bit width
///
/// # Returns
/// (per_tensor_mse, per_channel_mse)
pub fn compare_granularities(values: &[f32], num_channels: usize, bits: u8) -> (f32, f32) {
    // Per-tensor
    let pt_params = calibrate_per_tensor(values, bits, QuantMode::Symmetric);
    let pt_quantized = quantize_with_params(values, &pt_params);
    let pt_dequantized = dequantize_with_params(&pt_quantized, &pt_params);
    let pt_mse = quantization_mse(values, &pt_dequantized);

    // Per-channel
    let pc_params = calibrate_per_channel(values, num_channels, bits, QuantMode::Symmetric);
    let pc_quantized = quantize_with_params(values, &pc_params);
    let pc_dequantized = dequantize_with_params(&pc_quantized, &pc_params);
    let pc_mse = quantization_mse(values, &pc_dequantized);

    (pt_mse, pc_mse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    #[test]
    fn test_per_tensor_symmetric_8bit() {
        let values = vec![1.0, -2.0, 3.0, -4.0, 5.0, -5.0];
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);

        assert_eq!(params.scales.len(), 1);
        assert!(params.zero_points.is_empty());
        assert_eq!(params.granularity, QuantGranularity::PerTensor);

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert_abs_diff_eq!(orig, deq, epsilon = 0.1);
        }
    }

    #[test]
    fn test_per_tensor_asymmetric_8bit() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; // All positive
        let params = calibrate_per_tensor(&values, 8, QuantMode::Asymmetric);

        assert_eq!(params.scales.len(), 1);
        assert_eq!(params.zero_points.len(), 1);
        assert_eq!(params.mode, QuantMode::Asymmetric);

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert_abs_diff_eq!(orig, deq, epsilon = 0.1);
        }
    }

    #[test]
    fn test_per_channel_symmetric_8bit() {
        // 2 channels, 4 features each
        // Channel 0: small values, Channel 1: large values
        let values = vec![
            0.1, 0.2, -0.1, -0.2, // Channel 0
            10.0, 20.0, -10.0, -20.0, // Channel 1
        ];
        let params = calibrate_per_channel(&values, 2, 8, QuantMode::Symmetric);

        assert_eq!(params.scales.len(), 2);
        assert!(params.zero_points.is_empty());

        // Different scales for different channels
        assert!(params.scales[0] < params.scales[1]);

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            let rel_error = (orig - deq).abs() / orig.abs().max(0.01);
            assert!(rel_error < 0.1, "Error too large: {} vs {}", orig, deq);
        }
    }

    #[test]
    fn test_per_channel_better_than_per_tensor() {
        // Values with very different scales per channel
        let values = vec![
            0.01, 0.02, -0.01, -0.02, // Channel 0: tiny
            100.0, 200.0, -100.0, -200.0, // Channel 1: huge
        ];

        let (pt_mse, pc_mse) = compare_granularities(&values, 2, 8);

        // Per-channel should have lower error
        assert!(
            pc_mse <= pt_mse,
            "Per-channel MSE ({}) should be <= per-tensor MSE ({})",
            pc_mse,
            pt_mse
        );
    }

    #[test]
    fn test_per_group_quantization() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let params = calibrate_per_group(&values, 10, 8, QuantMode::Symmetric);

        assert_eq!(params.scales.len(), 10); // 100 values / 10 per group

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        let mse = quantization_mse(&values, &dequantized);
        assert!(mse < 0.01, "MSE {} too large", mse);
    }

    #[test]
    fn test_4bit_quantization() {
        let values = vec![1.0, -2.0, 3.0, -4.0, 5.0, -5.0, 6.0, -7.0];
        let params = calibrate_per_tensor(&values, 4, QuantMode::Symmetric);

        // 4-bit symmetric: qmax = 7
        assert!(params.scales[0] == 7.0 / 7.0); // max_abs / 7

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        // 4-bit has lower precision
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert_abs_diff_eq!(orig, deq, epsilon = 1.5);
        }
    }

    #[test]
    fn test_quantized_tensor_struct() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];

        let quantized = quantize_tensor(&values, &shape, QuantGranularity::PerChannel, QuantMode::Symmetric, 8);

        assert_eq!(quantized.shape, vec![2, 3]);
        assert_eq!(quantized.params.scales.len(), 2);
        assert_eq!(quantized.data.len(), 6);

        let dequantized = dequantize_tensor(&quantized);
        assert_eq!(dequantized.len(), 6);
    }

    #[test]
    fn test_memory_bytes() {
        let values = vec![1.0; 100];
        let shape = vec![100];

        let quantized = quantize_tensor(&values, &shape, QuantGranularity::PerTensor, QuantMode::Symmetric, 8);

        // 100 bytes data + 4 bytes scale = 104 bytes
        assert_eq!(quantized.memory_bytes(), 104);
    }

    #[test]
    fn test_empty_values() {
        let values: Vec<f32> = vec![];
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
        assert_eq!(params.scales[0], 1e-8 / 127.0);
    }

    #[test]
    fn test_zeros() {
        let values = vec![0.0; 10];
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        for val in dequantized {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-6);
        }
    }

    // Property tests

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_per_tensor_round_trip(values in proptest::collection::vec(-100.0f32..100.0, 1..100)) {
            let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let quantized = quantize_with_params(&values, &params);
            let dequantized = dequantize_with_params(&quantized, &params);

            prop_assert_eq!(dequantized.len(), values.len());

            let mse = quantization_mse(&values, &dequantized);
            // 8-bit should have low error
            prop_assert!(mse < 10.0, "MSE {} too large", mse);
        }

        #[test]
        fn prop_per_channel_scales_count(
            num_channels in 1usize..10,
            features_per_channel in 1usize..20
        ) {
            let values: Vec<f32> = (0..num_channels * features_per_channel)
                .map(|i| i as f32 * 0.1)
                .collect();

            let params = calibrate_per_channel(&values, num_channels, 8, QuantMode::Symmetric);

            prop_assert_eq!(params.scales.len(), num_channels);
        }

        #[test]
        fn prop_per_group_scales_count(
            total_values in 10usize..200,
            group_size in 1usize..20
        ) {
            let values: Vec<f32> = (0..total_values)
                .map(|i| i as f32 * 0.1)
                .collect();

            let params = calibrate_per_group(&values, group_size, 8, QuantMode::Symmetric);

            let expected_groups = total_values.div_ceil(group_size);
            prop_assert_eq!(params.scales.len(), expected_groups);
        }

        #[test]
        fn prop_per_channel_better_or_equal(
            num_channels in 2usize..5,
            features_per_channel in 5usize..20,
            scale_factor in 1.0f32..100.0
        ) {
            // Generate values where channels have different scales
            let values: Vec<f32> = (0..num_channels)
                .flat_map(|ch| {
                    let ch_scale = (ch as f32 + 1.0) * scale_factor;
                    (0..features_per_channel).map(move |i| (i as f32 * 0.1 - 0.5) * ch_scale)
                })
                .collect();

            let (pt_mse, pc_mse) = compare_granularities(&values, num_channels, 8);

            // Per-channel should be at least as good as per-tensor
            prop_assert!(
                pc_mse <= pt_mse * 1.01, // Small tolerance for floating point
                "Per-channel MSE ({}) should be <= per-tensor MSE ({})",
                pc_mse,
                pt_mse
            );
        }

        #[test]
        fn prop_symmetric_zero_mean(values in proptest::collection::vec(-100.0f32..100.0, 10..100)) {
            // For symmetric quantization, zero should map to zero
            let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);

            let zero_quantized = quantize_with_params(&[0.0], &params);
            let zero_dequantized = dequantize_with_params(&zero_quantized, &params);

            prop_assert!(zero_dequantized[0].abs() < 0.01, "Zero should map to ~zero");
        }

        #[test]
        fn prop_4bit_vs_8bit_accuracy(values in proptest::collection::vec(-100.0f32..100.0, 10..100)) {
            let params_8bit = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let params_4bit = calibrate_per_tensor(&values, 4, QuantMode::Symmetric);

            let q8 = quantize_with_params(&values, &params_8bit);
            let q4 = quantize_with_params(&values, &params_4bit);

            let d8 = dequantize_with_params(&q8, &params_8bit);
            let d4 = dequantize_with_params(&q4, &params_4bit);

            let mse_8bit = quantization_mse(&values, &d8);
            let mse_4bit = quantization_mse(&values, &d4);

            // 8-bit should generally be better than 4-bit
            prop_assert!(
                mse_8bit <= mse_4bit * 1.01,
                "8-bit MSE ({}) should be <= 4-bit MSE ({})",
                mse_8bit,
                mse_4bit
            );
        }
    }
}
