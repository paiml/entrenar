//! Calibration data collection for pruning
//!
//! Provides utilities for collecting activation statistics needed by
//! activation-weighted pruning methods like Wanda and SparseGPT.
//!
//! # Toyota Way: Genchi Genbutsu (Go and See)
//! Uses real activation data from calibration samples, not theoretical estimates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for calibration data collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Number of calibration samples to collect.
    num_samples: usize,
    /// Sequence length for text models.
    sequence_length: usize,
    /// Dataset identifier for calibration.
    dataset: String,
    /// Batch size for calibration forward passes.
    batch_size: usize,
    /// Whether to normalize activation statistics.
    normalize: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_samples: 128,
            sequence_length: 2048,
            dataset: "c4".to_string(),
            batch_size: 1,
            normalize: true,
        }
    }
}

impl CalibrationConfig {
    /// Create a new calibration configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of calibration samples.
    pub fn with_num_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    /// Set the sequence length.
    pub fn with_sequence_length(mut self, len: usize) -> Self {
        self.sequence_length = len;
        self
    }

    /// Set the dataset name.
    pub fn with_dataset(mut self, dataset: impl Into<String>) -> Self {
        self.dataset = dataset.into();
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set whether to normalize statistics.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Get the number of samples.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Get the sequence length.
    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Get the dataset name.
    pub fn dataset(&self) -> &str {
        &self.dataset
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Check if normalization is enabled.
    pub fn normalize(&self) -> bool {
        self.normalize
    }
}

/// Per-layer activation statistics collected during calibration.
///
/// Uses Welford's algorithm for numerically stable online computation.
#[derive(Debug, Clone, Default)]
pub struct LayerActivationStats {
    /// Running sum of L2 norms per input channel.
    input_norm_sum: Vec<f32>,
    /// Running sum of squared activations per channel.
    squared_sum: Vec<f32>,
    /// Number of samples processed.
    count: usize,
    /// Input feature dimension.
    input_dim: usize,
}

impl LayerActivationStats {
    /// Create new statistics tracker for a given input dimension.
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_norm_sum: vec![0.0; input_dim],
            squared_sum: vec![0.0; input_dim],
            count: 0,
            input_dim,
        }
    }

    /// Update statistics with a new batch of activations.
    ///
    /// # Arguments
    ///
    /// * `activations` - Batch of activations [batch_size, input_dim]
    ///
    /// # Panics
    ///
    /// Panics if activation dimensions don't match.
    pub fn update(&mut self, activations: &[Vec<f32>]) {
        if activations.is_empty() {
            return;
        }

        for sample in activations {
            assert_eq!(
                sample.len(),
                self.input_dim,
                "Activation dimension mismatch: expected {}, got {}",
                self.input_dim,
                sample.len()
            );

            for (i, &val) in sample.iter().enumerate() {
                // Accumulate squared values for L2 norm computation
                self.squared_sum[i] += val * val;
                self.input_norm_sum[i] += val.abs();
            }
            self.count += 1;
        }
    }

    /// Get the mean L2 norm for each input channel.
    ///
    /// Returns sqrt(mean(x^2)) for each channel.
    pub fn input_norms(&self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.input_dim];
        }

        self.squared_sum
            .iter()
            .map(|&sum| (sum / self.count as f32).sqrt())
            .collect()
    }

    /// Get the mean absolute value for each input channel.
    pub fn mean_abs(&self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.input_dim];
        }

        self.input_norm_sum
            .iter()
            .map(|&sum| sum / self.count as f32)
            .collect()
    }

    /// Get the number of samples processed.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Check if any statistics have been collected.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.input_norm_sum.fill(0.0);
        self.squared_sum.fill(0.0);
        self.count = 0;
    }
}

/// Calibration data collector for pruning.
///
/// Collects and stores per-layer activation statistics during
/// calibration forward passes.
#[derive(Debug, Clone)]
pub struct CalibrationCollector {
    /// Configuration for calibration.
    config: CalibrationConfig,
    /// Per-layer activation statistics.
    layer_stats: HashMap<String, LayerActivationStats>,
    /// Total samples processed.
    samples_processed: usize,
    /// Whether calibration is complete.
    complete: bool,
}

impl CalibrationCollector {
    /// Create a new calibration collector.
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            config,
            layer_stats: HashMap::new(),
            samples_processed: 0,
            complete: false,
        }
    }

    /// Register a layer for calibration.
    ///
    /// # Arguments
    ///
    /// * `name` - Layer name
    /// * `input_dim` - Input feature dimension
    pub fn register_layer(&mut self, name: impl Into<String>, input_dim: usize) {
        let name = name.into();
        self.layer_stats
            .entry(name)
            .or_insert_with(|| LayerActivationStats::new(input_dim));
    }

    /// Record activations for a layer.
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Layer name
    /// * `activations` - Batch of activations
    pub fn record_activations(&mut self, layer_name: &str, activations: &[Vec<f32>]) {
        if self.complete {
            return;
        }

        if let Some(stats) = self.layer_stats.get_mut(layer_name) {
            stats.update(activations);
        }
    }

    /// Mark a batch as processed.
    pub fn batch_complete(&mut self, batch_size: usize) {
        self.samples_processed += batch_size;

        if self.samples_processed >= self.config.num_samples {
            self.complete = true;
        }
    }

    /// Get activation statistics for a layer.
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&LayerActivationStats> {
        self.layer_stats.get(layer_name)
    }

    /// Get all layer names.
    pub fn layer_names(&self) -> Vec<&String> {
        self.layer_stats.keys().collect()
    }

    /// Check if calibration is complete.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Get the number of samples processed.
    pub fn samples_processed(&self) -> usize {
        self.samples_processed
    }

    /// Get the configuration.
    pub fn config(&self) -> &CalibrationConfig {
        &self.config
    }

    /// Reset all collected statistics.
    pub fn reset(&mut self) {
        for stats in self.layer_stats.values_mut() {
            stats.reset();
        }
        self.samples_processed = 0;
        self.complete = false;
    }

    /// Get progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        if self.config.num_samples == 0 {
            return 1.0;
        }
        (self.samples_processed as f32 / self.config.num_samples as f32).min(1.0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CalibrationConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default_values() {
        // TEST_ID: CAL-001
        let config = CalibrationConfig::default();
        assert_eq!(
            config.num_samples(),
            128,
            "CAL-001 FALSIFIED: Default num_samples should be 128"
        );
        assert_eq!(
            config.sequence_length(),
            2048,
            "CAL-001 FALSIFIED: Default sequence_length should be 2048"
        );
        assert_eq!(
            config.dataset(),
            "c4",
            "CAL-001 FALSIFIED: Default dataset should be c4"
        );
        assert_eq!(
            config.batch_size(),
            1,
            "CAL-001 FALSIFIED: Default batch_size should be 1"
        );
        assert!(
            config.normalize(),
            "CAL-001 FALSIFIED: Default normalize should be true"
        );
    }

    #[test]
    fn test_config_builder_pattern() {
        // TEST_ID: CAL-002
        let config = CalibrationConfig::new()
            .with_num_samples(256)
            .with_sequence_length(1024)
            .with_dataset("wikitext")
            .with_batch_size(4)
            .with_normalize(false);

        assert_eq!(config.num_samples(), 256);
        assert_eq!(config.sequence_length(), 1024);
        assert_eq!(config.dataset(), "wikitext");
        assert_eq!(config.batch_size(), 4);
        assert!(!config.normalize());
    }

    #[test]
    fn test_config_serialize_json() {
        // TEST_ID: CAL-003
        let config = CalibrationConfig::new().with_num_samples(64);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CalibrationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            config.num_samples(),
            deserialized.num_samples(),
            "CAL-003 FALSIFIED: Serialization roundtrip failed"
        );
    }

    // =========================================================================
    // LayerActivationStats Tests
    // =========================================================================

    #[test]
    fn test_layer_stats_new() {
        // TEST_ID: CAL-010
        let stats = LayerActivationStats::new(512);
        assert_eq!(
            stats.input_dim(),
            512,
            "CAL-010 FALSIFIED: input_dim should be 512"
        );
        assert_eq!(
            stats.count(),
            0,
            "CAL-010 FALSIFIED: initial count should be 0"
        );
        assert!(
            stats.is_empty(),
            "CAL-010 FALSIFIED: should be empty initially"
        );
    }

    #[test]
    fn test_layer_stats_update_single_sample() {
        // TEST_ID: CAL-011
        let mut stats = LayerActivationStats::new(4);
        let sample = vec![1.0, 2.0, 3.0, 4.0];
        stats.update(&[sample]);

        assert_eq!(
            stats.count(),
            1,
            "CAL-011 FALSIFIED: count should be 1 after one sample"
        );
        assert!(
            !stats.is_empty(),
            "CAL-011 FALSIFIED: should not be empty after update"
        );
    }

    #[test]
    fn test_layer_stats_input_norms_single_sample() {
        // TEST_ID: CAL-012
        // For a single sample [1, 2, 3, 4]:
        // L2 norm per channel = sqrt(x^2 / 1) = |x|
        let mut stats = LayerActivationStats::new(4);
        let sample = vec![1.0, 2.0, 3.0, 4.0];
        stats.update(&[sample]);

        let norms = stats.input_norms();
        assert!(
            (norms[0] - 1.0).abs() < 1e-6,
            "CAL-012 FALSIFIED: norm[0] should be 1.0"
        );
        assert!(
            (norms[1] - 2.0).abs() < 1e-6,
            "CAL-012 FALSIFIED: norm[1] should be 2.0"
        );
        assert!(
            (norms[2] - 3.0).abs() < 1e-6,
            "CAL-012 FALSIFIED: norm[2] should be 3.0"
        );
        assert!(
            (norms[3] - 4.0).abs() < 1e-6,
            "CAL-012 FALSIFIED: norm[3] should be 4.0"
        );
    }

    #[test]
    fn test_layer_stats_input_norms_multiple_samples() {
        // TEST_ID: CAL-013
        // Two samples: [1, 2] and [3, 4]
        // squared_sum[0] = 1 + 9 = 10, mean = 5, sqrt = sqrt(5) ≈ 2.236
        // squared_sum[1] = 4 + 16 = 20, mean = 10, sqrt = sqrt(10) ≈ 3.162
        let mut stats = LayerActivationStats::new(2);
        stats.update(&[vec![1.0, 2.0], vec![3.0, 4.0]]);

        let norms = stats.input_norms();
        let expected_0 = (5.0_f32).sqrt();
        let expected_1 = (10.0_f32).sqrt();

        assert!(
            (norms[0] - expected_0).abs() < 1e-5,
            "CAL-013 FALSIFIED: norm[0] should be sqrt(5), got {}",
            norms[0]
        );
        assert!(
            (norms[1] - expected_1).abs() < 1e-5,
            "CAL-013 FALSIFIED: norm[1] should be sqrt(10), got {}",
            norms[1]
        );
    }

    #[test]
    fn test_layer_stats_mean_abs() {
        // TEST_ID: CAL-014
        let mut stats = LayerActivationStats::new(2);
        stats.update(&[vec![1.0, -2.0], vec![3.0, -4.0]]);

        let mean_abs = stats.mean_abs();
        // mean_abs[0] = (1 + 3) / 2 = 2.0
        // mean_abs[1] = (2 + 4) / 2 = 3.0
        assert!(
            (mean_abs[0] - 2.0).abs() < 1e-6,
            "CAL-014 FALSIFIED: mean_abs[0] should be 2.0"
        );
        assert!(
            (mean_abs[1] - 3.0).abs() < 1e-6,
            "CAL-014 FALSIFIED: mean_abs[1] should be 3.0"
        );
    }

    #[test]
    fn test_layer_stats_empty_batch() {
        // TEST_ID: CAL-015
        let mut stats = LayerActivationStats::new(4);
        stats.update(&[]);
        assert!(
            stats.is_empty(),
            "CAL-015 FALSIFIED: Empty batch should not update stats"
        );
    }

    #[test]
    fn test_layer_stats_reset() {
        // TEST_ID: CAL-016
        let mut stats = LayerActivationStats::new(4);
        stats.update(&[vec![1.0, 2.0, 3.0, 4.0]]);
        assert!(!stats.is_empty());

        stats.reset();
        assert!(
            stats.is_empty(),
            "CAL-016 FALSIFIED: Should be empty after reset"
        );
        assert_eq!(
            stats.count(),
            0,
            "CAL-016 FALSIFIED: Count should be 0 after reset"
        );
    }

    #[test]
    fn test_layer_stats_empty_returns_zeros() {
        // TEST_ID: CAL-017
        let stats = LayerActivationStats::new(3);
        let norms = stats.input_norms();
        assert_eq!(norms.len(), 3);
        assert!(norms.iter().all(|&x| x == 0.0));

        let mean_abs = stats.mean_abs();
        assert!(mean_abs.iter().all(|&x| x == 0.0));
    }

    // =========================================================================
    // CalibrationCollector Tests
    // =========================================================================

    #[test]
    fn test_collector_new() {
        // TEST_ID: CAL-020
        let config = CalibrationConfig::new().with_num_samples(64);
        let collector = CalibrationCollector::new(config);

        assert!(!collector.is_complete());
        assert_eq!(collector.samples_processed(), 0);
        assert!(collector.layer_names().is_empty());
    }

    #[test]
    fn test_collector_register_layer() {
        // TEST_ID: CAL-021
        let config = CalibrationConfig::default();
        let mut collector = CalibrationCollector::new(config);

        collector.register_layer("layer.0.attn", 512);
        collector.register_layer("layer.0.mlp", 2048);

        assert_eq!(
            collector.layer_names().len(),
            2,
            "CAL-021 FALSIFIED: Should have 2 registered layers"
        );
        assert!(collector.get_layer_stats("layer.0.attn").is_some());
        assert!(collector.get_layer_stats("layer.0.mlp").is_some());
    }

    #[test]
    fn test_collector_duplicate_registration() {
        // TEST_ID: CAL-022
        let mut collector = CalibrationCollector::new(CalibrationConfig::default());

        collector.register_layer("layer.0", 512);
        collector.register_layer("layer.0", 1024); // Different dim, should not override

        let stats = collector.get_layer_stats("layer.0").unwrap();
        assert_eq!(
            stats.input_dim(),
            512,
            "CAL-022 FALSIFIED: Duplicate registration should not override"
        );
    }

    #[test]
    fn test_collector_record_activations() {
        // TEST_ID: CAL-023
        let mut collector = CalibrationCollector::new(CalibrationConfig::default());
        collector.register_layer("layer.0", 4);

        let activations = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        collector.record_activations("layer.0", &activations);

        let stats = collector.get_layer_stats("layer.0").unwrap();
        assert_eq!(
            stats.count(),
            2,
            "CAL-023 FALSIFIED: Should have recorded 2 samples"
        );
    }

    #[test]
    fn test_collector_unregistered_layer() {
        // TEST_ID: CAL-024
        let mut collector = CalibrationCollector::new(CalibrationConfig::default());

        // Recording to unregistered layer should be a no-op
        collector.record_activations("nonexistent", &[vec![1.0, 2.0]]);

        assert!(
            collector.get_layer_stats("nonexistent").is_none(),
            "CAL-024 FALSIFIED: Unregistered layer should not exist"
        );
    }

    #[test]
    fn test_collector_batch_complete() {
        // TEST_ID: CAL-025
        let config = CalibrationConfig::new().with_num_samples(10);
        let mut collector = CalibrationCollector::new(config);

        collector.batch_complete(5);
        assert_eq!(collector.samples_processed(), 5);
        assert!(!collector.is_complete());

        collector.batch_complete(5);
        assert_eq!(collector.samples_processed(), 10);
        assert!(collector.is_complete());
    }

    #[test]
    fn test_collector_progress() {
        // TEST_ID: CAL-026
        let config = CalibrationConfig::new().with_num_samples(100);
        let mut collector = CalibrationCollector::new(config);

        assert!(
            collector.progress().abs() < 1e-6,
            "CAL-026 FALSIFIED: Initial progress should be 0"
        );

        collector.batch_complete(25);
        assert!(
            (collector.progress() - 0.25).abs() < 1e-6,
            "CAL-026 FALSIFIED: Progress should be 0.25"
        );

        collector.batch_complete(75);
        assert!(
            (collector.progress() - 1.0).abs() < 1e-6,
            "CAL-026 FALSIFIED: Progress should be 1.0"
        );
    }

    #[test]
    fn test_collector_progress_exceeds_target() {
        // TEST_ID: CAL-027
        let config = CalibrationConfig::new().with_num_samples(10);
        let mut collector = CalibrationCollector::new(config);

        collector.batch_complete(20);
        assert!(
            (collector.progress() - 1.0).abs() < 1e-6,
            "CAL-027 FALSIFIED: Progress should be clamped to 1.0"
        );
    }

    #[test]
    fn test_collector_stops_recording_when_complete() {
        // TEST_ID: CAL-028
        let config = CalibrationConfig::new().with_num_samples(2);
        let mut collector = CalibrationCollector::new(config);
        collector.register_layer("layer.0", 2);

        // Complete calibration
        collector.record_activations("layer.0", &[vec![1.0, 2.0], vec![3.0, 4.0]]);
        collector.batch_complete(2);
        assert!(collector.is_complete());

        // Try to record more (should be ignored)
        collector.record_activations("layer.0", &[vec![5.0, 6.0]]);

        let stats = collector.get_layer_stats("layer.0").unwrap();
        assert_eq!(
            stats.count(),
            2,
            "CAL-028 FALSIFIED: Recording should stop when complete"
        );
    }

    #[test]
    fn test_collector_reset() {
        // TEST_ID: CAL-029
        let config = CalibrationConfig::new().with_num_samples(10);
        let mut collector = CalibrationCollector::new(config);
        collector.register_layer("layer.0", 4);

        collector.record_activations("layer.0", &[vec![1.0, 2.0, 3.0, 4.0]]);
        collector.batch_complete(10);
        assert!(collector.is_complete());

        collector.reset();
        assert!(!collector.is_complete());
        assert_eq!(collector.samples_processed(), 0);
        let stats = collector.get_layer_stats("layer.0").unwrap();
        assert!(
            stats.is_empty(),
            "CAL-029 FALSIFIED: Layer stats should be reset"
        );
    }

    #[test]
    fn test_collector_zero_samples_config() {
        // TEST_ID: CAL-030
        let config = CalibrationConfig::new().with_num_samples(0);
        let collector = CalibrationCollector::new(config);
        assert!(
            (collector.progress() - 1.0).abs() < 1e-6,
            "CAL-030 FALSIFIED: Progress with 0 samples should be 1.0"
        );
    }

    #[test]
    fn test_collector_config_access() {
        // TEST_ID: CAL-031
        let config = CalibrationConfig::new()
            .with_num_samples(256)
            .with_dataset("custom");
        let collector = CalibrationCollector::new(config);

        assert_eq!(collector.config().num_samples(), 256);
        assert_eq!(collector.config().dataset(), "custom");
    }

    // =========================================================================
    // Welford's Algorithm Numerical Stability Tests
    // =========================================================================

    #[test]
    fn test_welford_large_values() {
        // TEST_ID: CAL-040
        // Test with large values that could cause overflow in naive implementations
        let mut stats = LayerActivationStats::new(2);

        // Large values
        stats.update(&[vec![1e6, 1e6]]);
        stats.update(&[vec![1e6 + 1.0, 1e6 + 1.0]]);

        let norms = stats.input_norms();
        // Should not overflow or produce NaN/Inf
        assert!(
            norms[0].is_finite(),
            "CAL-040 FALSIFIED: Large values should produce finite results"
        );
    }

    #[test]
    fn test_welford_small_values() {
        // TEST_ID: CAL-041
        let mut stats = LayerActivationStats::new(2);

        // Very small values
        stats.update(&[vec![1e-10, 1e-10]]);
        stats.update(&[vec![1e-10, 1e-10]]);

        let norms = stats.input_norms();
        assert!(
            norms[0].is_finite(),
            "CAL-041 FALSIFIED: Small values should produce finite results"
        );
        assert!(
            norms[0] > 0.0,
            "CAL-041 FALSIFIED: Small positive values should produce positive norms"
        );
    }

    #[test]
    fn test_welford_mixed_sign() {
        // TEST_ID: CAL-042
        let mut stats = LayerActivationStats::new(2);

        // Mixed positive and negative values
        stats.update(&[vec![1.0, -2.0], vec![-3.0, 4.0]]);

        let norms = stats.input_norms();
        // L2 norm should be positive regardless of sign
        assert!(
            norms[0] > 0.0,
            "CAL-042 FALSIFIED: L2 norm should be positive"
        );
        assert!(
            norms[1] > 0.0,
            "CAL-042 FALSIFIED: L2 norm should be positive"
        );
    }
}
