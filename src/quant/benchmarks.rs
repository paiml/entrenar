//! Accuracy Degradation Benchmarks
//!
//! Provides benchmarks for measuring quantization accuracy degradation:
//! - Synthetic workload benchmarks
//! - Bit-width comparison (4-bit vs 8-bit)
//! - Granularity comparison (per-tensor vs per-channel vs per-group)
//! - Model-like weight pattern tests
//! - Numerical precision edge cases

use super::error_analysis::analyze_error;
use super::granularity::{
    calibrate_per_channel, calibrate_per_group, calibrate_per_tensor, dequantize_with_params,
    quantization_mse, quantize_with_params, QuantGranularity, QuantMode,
};
use serde::{Deserialize, Serialize};

/// Benchmark results for quantization accuracy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantBenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Number of elements tested
    pub num_elements: usize,
    /// Bits used for quantization
    pub bits: u8,
    /// Granularity used
    pub granularity: QuantGranularity,
    /// Mode used (symmetric/asymmetric)
    pub mode: QuantMode,
    /// MSE error
    pub mse: f32,
    /// Max error
    pub max_error: f32,
    /// SQNR in dB
    pub sqnr_db: f32,
    /// Compression ratio
    pub compression_ratio: f32,
}

impl QuantBenchmarkResult {
    /// Quality score (higher is better): SQNR / compression overhead
    pub fn quality_score(&self) -> f32 {
        if self.compression_ratio > 0.0 {
            self.sqnr_db / self.compression_ratio.max(1.0)
        } else {
            0.0
        }
    }
}

/// Suite of benchmark results
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct BenchmarkSuite {
    pub results: Vec<QuantBenchmarkResult>,
}

impl BenchmarkSuite {
    /// Add a benchmark result
    pub fn add(&mut self, result: QuantBenchmarkResult) {
        self.results.push(result);
    }

    /// Get best result by SQNR
    pub fn best_by_sqnr(&self) -> Option<&QuantBenchmarkResult> {
        self.results
            .iter()
            .max_by(|a, b| a.sqnr_db.partial_cmp(&b.sqnr_db).unwrap())
    }

    /// Get best result by MSE (lowest)
    pub fn best_by_mse(&self) -> Option<&QuantBenchmarkResult> {
        self.results
            .iter()
            .min_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap())
    }

    /// Get results sorted by quality score
    pub fn sorted_by_quality(&self) -> Vec<&QuantBenchmarkResult> {
        let mut sorted: Vec<_> = self.results.iter().collect();
        sorted.sort_by(|a, b| b.quality_score().partial_cmp(&a.quality_score()).unwrap());
        sorted
    }
}

/// Run benchmark on given values with specified configuration
pub fn run_benchmark(
    name: &str,
    values: &[f32],
    bits: u8,
    granularity: QuantGranularity,
    mode: QuantMode,
) -> QuantBenchmarkResult {
    let params = match granularity {
        QuantGranularity::PerTensor => calibrate_per_tensor(values, bits, mode),
        QuantGranularity::PerChannel => {
            // Assume square-ish shape for simplicity
            let num_channels = (values.len() as f32).sqrt() as usize;
            calibrate_per_channel(values, num_channels.max(1), bits, mode)
        }
        QuantGranularity::PerGroup(size) => calibrate_per_group(values, size, bits, mode),
    };

    let stats = analyze_error(values, &params, 0.1);

    // Calculate compression ratio
    let original_bytes = values.len() * 4; // f32 = 4 bytes
    let scale_bytes = params.scales.len() * 4;
    let zp_bytes = params.zero_points.len() * 4;
    let data_bytes = if bits == 4 {
        values.len().div_ceil(2)
    } else {
        values.len()
    };
    let compressed_bytes = scale_bytes + zp_bytes + data_bytes;
    let compression_ratio = original_bytes as f32 / compressed_bytes.max(1) as f32;

    QuantBenchmarkResult {
        name: name.to_string(),
        num_elements: values.len(),
        bits,
        granularity,
        mode,
        mse: stats.mse,
        max_error: stats.max_error,
        sqnr_db: stats.sqnr_db,
        compression_ratio,
    }
}

/// Generate Gaussian-like weight distribution (common in neural networks)
pub fn generate_gaussian_weights(n: usize, mean: f32, std_dev: f32, seed: u64) -> Vec<f32> {
    // Simple LCG for reproducibility
    let mut state = seed;
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    (0..n)
        .map(|_| {
            // Box-Muller transform (simplified)
            state = (a.wrapping_mul(state).wrapping_add(c)) % m;
            let u1 = (state as f32) / (m as f32);
            state = (a.wrapping_mul(state).wrapping_add(c)) % m;
            let u2 = (state as f32) / (m as f32);

            let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            mean + std_dev * z
        })
        .collect()
}

/// Generate uniform weights in range
pub fn generate_uniform_weights(n: usize, min: f32, max: f32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    (0..n)
        .map(|_| {
            state = (a.wrapping_mul(state).wrapping_add(c)) % m;
            let t = (state as f32) / (m as f32);
            min + t * (max - min)
        })
        .collect()
}

/// Generate weights with outliers (to test robustness)
pub fn generate_weights_with_outliers(
    n: usize,
    outlier_ratio: f32,
    outlier_magnitude: f32,
    seed: u64,
) -> Vec<f32> {
    let mut weights = generate_gaussian_weights(n, 0.0, 1.0, seed);
    let num_outliers = (n as f32 * outlier_ratio) as usize;

    let mut state = seed.wrapping_add(12345);
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    for _ in 0..num_outliers {
        state = (a.wrapping_mul(state).wrapping_add(c)) % m;
        let idx = (state as usize) % n;
        state = (a.wrapping_mul(state).wrapping_add(c)) % m;
        let sign = if state.is_multiple_of(2) { 1.0 } else { -1.0 };
        weights[idx] = sign * outlier_magnitude;
    }

    weights
}

/// Generate multi-channel weights (like conv/linear layer)
pub fn generate_multi_channel_weights(
    num_channels: usize,
    features_per_channel: usize,
    scale_variance: f32,
    seed: u64,
) -> Vec<f32> {
    let mut weights = Vec::with_capacity(num_channels * features_per_channel);
    let mut state = seed;
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    for ch in 0..num_channels {
        state = (a.wrapping_mul(state).wrapping_add(c)) % m;
        let channel_scale = 1.0 + (ch as f32 / num_channels as f32) * scale_variance;

        let channel_weights =
            generate_gaussian_weights(features_per_channel, 0.0, channel_scale, state);
        weights.extend(channel_weights);
    }

    weights
}

/// Run full benchmark suite on various weight patterns
pub fn run_full_benchmark_suite(size: usize) -> BenchmarkSuite {
    let mut suite = BenchmarkSuite::default();

    // Gaussian weights
    let gaussian = generate_gaussian_weights(size, 0.0, 1.0, 42);

    // Test different configurations
    for bits in [4u8, 8] {
        for granularity in [
            QuantGranularity::PerTensor,
            QuantGranularity::PerChannel,
            QuantGranularity::PerGroup(32),
        ] {
            let name = format!(
                "gaussian_{}bit_{:?}",
                bits,
                match granularity {
                    QuantGranularity::PerTensor => "tensor",
                    QuantGranularity::PerChannel => "channel",
                    QuantGranularity::PerGroup(_) => "group",
                }
            );
            suite.add(run_benchmark(
                &name,
                &gaussian,
                bits,
                granularity,
                QuantMode::Symmetric,
            ));
        }
    }

    // Uniform weights
    let uniform = generate_uniform_weights(size, -1.0, 1.0, 43);
    suite.add(run_benchmark(
        "uniform_8bit_tensor",
        &uniform,
        8,
        QuantGranularity::PerTensor,
        QuantMode::Symmetric,
    ));

    // Weights with outliers
    let outliers = generate_weights_with_outliers(size, 0.01, 10.0, 44);
    suite.add(run_benchmark(
        "outliers_8bit_tensor",
        &outliers,
        8,
        QuantGranularity::PerTensor,
        QuantMode::Symmetric,
    ));
    suite.add(run_benchmark(
        "outliers_8bit_group32",
        &outliers,
        8,
        QuantGranularity::PerGroup(32),
        QuantMode::Symmetric,
    ));

    // Multi-channel weights
    let multi_ch = generate_multi_channel_weights(16, size / 16, 5.0, 45);
    suite.add(run_benchmark(
        "multi_channel_8bit_tensor",
        &multi_ch,
        8,
        QuantGranularity::PerTensor,
        QuantMode::Symmetric,
    ));
    suite.add(run_benchmark(
        "multi_channel_8bit_channel",
        &multi_ch,
        8,
        QuantGranularity::PerChannel,
        QuantMode::Symmetric,
    ));

    suite
}

/// Compare accuracy degradation across bit widths
pub fn compare_bit_width_degradation(values: &[f32]) -> Vec<(u8, f32, f32)> {
    let mut results = Vec::new();

    for bits in [4u8, 8] {
        let params = calibrate_per_tensor(values, bits, QuantMode::Symmetric);
        let quantized = quantize_with_params(values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);
        let mse = quantization_mse(values, &dequantized);

        let compression = if bits == 4 { 8.0 } else { 4.0 }; // vs f32
        results.push((bits, mse, compression));
    }

    results
}

/// Calculate accuracy retention percentage
pub fn accuracy_retention(original_mse: f32, quantized_mse: f32) -> f32 {
    if quantized_mse > 1e-10 {
        (1.0 - (quantized_mse - original_mse).abs() / quantized_mse.max(original_mse)) * 100.0
    } else if original_mse > 1e-10 {
        0.0
    } else {
        100.0
    }
}

#[cfg(test)]
mod tests {
    use super::super::granularity::dequantize_with_params;
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_run_benchmark() {
        let values = generate_gaussian_weights(1000, 0.0, 1.0, 42);
        let result = run_benchmark(
            "test",
            &values,
            8,
            QuantGranularity::PerTensor,
            QuantMode::Symmetric,
        );

        assert_eq!(result.name, "test");
        assert_eq!(result.num_elements, 1000);
        assert_eq!(result.bits, 8);
        assert!(result.mse >= 0.0);
        assert!(result.sqnr_db > 0.0);
        assert!(result.compression_ratio > 1.0);
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::default();

        suite.add(QuantBenchmarkResult {
            name: "a".to_string(),
            num_elements: 100,
            bits: 8,
            granularity: QuantGranularity::PerTensor,
            mode: QuantMode::Symmetric,
            mse: 0.01,
            max_error: 0.1,
            sqnr_db: 40.0,
            compression_ratio: 4.0,
        });

        suite.add(QuantBenchmarkResult {
            name: "b".to_string(),
            num_elements: 100,
            bits: 4,
            granularity: QuantGranularity::PerTensor,
            mode: QuantMode::Symmetric,
            mse: 0.1,
            max_error: 0.5,
            sqnr_db: 20.0,
            compression_ratio: 8.0,
        });

        assert_eq!(suite.results.len(), 2);
        assert_eq!(suite.best_by_sqnr().unwrap().name, "a");
        assert_eq!(suite.best_by_mse().unwrap().name, "a");
    }

    #[test]
    fn test_quality_score() {
        let result = QuantBenchmarkResult {
            name: "test".to_string(),
            num_elements: 100,
            bits: 8,
            granularity: QuantGranularity::PerTensor,
            mode: QuantMode::Symmetric,
            mse: 0.01,
            max_error: 0.1,
            sqnr_db: 40.0,
            compression_ratio: 4.0,
        };

        assert_eq!(result.quality_score(), 10.0); // 40 / 4
    }

    #[test]
    fn test_generate_gaussian_weights() {
        let weights = generate_gaussian_weights(1000, 0.0, 1.0, 42);

        assert_eq!(weights.len(), 1000);

        // Mean should be approximately 0
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        assert!(mean.abs() < 0.2, "Mean {} should be close to 0", mean);

        // Std dev should be approximately 1
        let variance: f32 =
            weights.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        let std_dev = variance.sqrt();
        assert!(
            (std_dev - 1.0).abs() < 0.3,
            "Std dev {} should be close to 1",
            std_dev
        );
    }

    #[test]
    fn test_generate_uniform_weights() {
        let weights = generate_uniform_weights(1000, -1.0, 1.0, 42);

        assert_eq!(weights.len(), 1000);

        for &w in &weights {
            assert!(w >= -1.0 && w <= 1.0, "Weight {} out of range", w);
        }
    }

    #[test]
    fn test_generate_weights_with_outliers() {
        let weights = generate_weights_with_outliers(1000, 0.01, 10.0, 42);

        assert_eq!(weights.len(), 1000);

        // Should have some large values
        let large_count = weights.iter().filter(|&&w| w.abs() > 5.0).count();
        assert!(large_count > 0, "Should have outliers");
    }

    #[test]
    fn test_generate_multi_channel_weights() {
        let weights = generate_multi_channel_weights(16, 64, 5.0, 42);

        assert_eq!(weights.len(), 16 * 64);
    }

    #[test]
    fn test_full_benchmark_suite() {
        let suite = run_full_benchmark_suite(256);

        // Should have multiple results
        assert!(suite.results.len() >= 5);

        // All results should be valid
        for result in &suite.results {
            assert!(result.mse >= 0.0);
            assert!(result.compression_ratio >= 1.0);
        }
    }

    #[test]
    fn test_bit_width_comparison() {
        let values = generate_gaussian_weights(1000, 0.0, 1.0, 42);
        let results = compare_bit_width_degradation(&values);

        assert_eq!(results.len(), 2);

        // 8-bit should have lower MSE than 4-bit
        let (_, mse_4bit, _) = results.iter().find(|(b, _, _)| *b == 4).unwrap();
        let (_, mse_8bit, _) = results.iter().find(|(b, _, _)| *b == 8).unwrap();

        assert!(
            mse_8bit <= mse_4bit,
            "8-bit MSE ({}) should be <= 4-bit MSE ({})",
            mse_8bit,
            mse_4bit
        );
    }

    #[test]
    fn test_accuracy_retention() {
        assert_eq!(accuracy_retention(0.0, 0.0), 100.0);
        assert!(accuracy_retention(0.0, 0.01) >= 0.0);
        assert!(accuracy_retention(0.0, 0.01) <= 100.0);
    }

    // Property tests

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_benchmark_compression_positive(
            size in 100usize..500,
            bits in prop::sample::select(vec![4u8, 8])
        ) {
            let values = generate_gaussian_weights(size, 0.0, 1.0, 42);
            let result = run_benchmark(
                "test",
                &values,
                bits,
                QuantGranularity::PerTensor,
                QuantMode::Symmetric,
            );

            prop_assert!(result.compression_ratio > 0.0);
        }

        #[test]
        fn prop_8bit_better_than_4bit(size in 100usize..500) {
            let values = generate_gaussian_weights(size, 0.0, 1.0, 42);

            let result_4bit = run_benchmark(
                "4bit",
                &values,
                4,
                QuantGranularity::PerTensor,
                QuantMode::Symmetric,
            );
            let result_8bit = run_benchmark(
                "8bit",
                &values,
                8,
                QuantGranularity::PerTensor,
                QuantMode::Symmetric,
            );

            prop_assert!(
                result_8bit.mse <= result_4bit.mse * 1.01,
                "8-bit MSE ({}) should be <= 4-bit MSE ({})",
                result_8bit.mse,
                result_4bit.mse
            );
        }

        #[test]
        fn prop_per_channel_helps_multi_scale(
            num_channels in 4usize..16,
            scale_variance in 5.0f32..20.0 // Higher variance to ensure per-channel helps
        ) {
            let features = 64;
            let values = generate_multi_channel_weights(num_channels, features, scale_variance, 42);

            // Use calibration directly with correct num_channels
            let params_tensor = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let params_channel = calibrate_per_channel(&values, num_channels, 8, QuantMode::Symmetric);

            let q_tensor = quantize_with_params(&values, &params_tensor);
            let q_channel = quantize_with_params(&values, &params_channel);

            let d_tensor = dequantize_with_params(&q_tensor, &params_tensor);
            let d_channel = dequantize_with_params(&q_channel, &params_channel);

            let mse_tensor = quantization_mse(&values, &d_tensor);
            let mse_channel = quantization_mse(&values, &d_channel);

            // Per-channel should be at least as good when scales vary significantly
            prop_assert!(
                mse_channel <= mse_tensor * 1.01,
                "Per-channel MSE ({}) should be <= per-tensor MSE ({})",
                mse_channel,
                mse_tensor
            );
        }

        #[test]
        fn prop_benchmark_deterministic(size in 100usize..500) {
            let values1 = generate_gaussian_weights(size, 0.0, 1.0, 42);
            let values2 = generate_gaussian_weights(size, 0.0, 1.0, 42);

            // Same seed should produce same weights
            prop_assert_eq!(values1, values2);
        }

        #[test]
        fn prop_sqnr_positive_for_signal(size in 100usize..500) {
            let values = generate_gaussian_weights(size, 0.0, 1.0, 42);
            let result = run_benchmark(
                "test",
                &values,
                8,
                QuantGranularity::PerTensor,
                QuantMode::Symmetric,
            );

            prop_assert!(result.sqnr_db > 0.0, "SQNR must be positive");
        }
    }
}
