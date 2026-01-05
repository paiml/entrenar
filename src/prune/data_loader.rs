//! Calibration data loader for pruning
//!
//! Provides data loading utilities for collecting activation statistics
//! during calibration for pruning methods like Wanda and SparseGPT.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::prune::{CalibrationDataLoader, CalibrationDataConfig};
//!
//! let config = CalibrationDataConfig::new()
//!     .with_num_samples(128)
//!     .with_batch_size(4);
//!
//! let loader = CalibrationDataLoader::new(config);
//! for batch in loader.iter() {
//!     // Process batch for calibration
//! }
//! ```

use crate::train::Batch;
use crate::Tensor;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for calibration data loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationDataConfig {
    /// Number of calibration samples to load.
    num_samples: usize,
    /// Batch size for calibration.
    batch_size: usize,
    /// Sequence length (for text models).
    sequence_length: usize,
    /// Dataset name or path.
    dataset: String,
    /// Cache directory for downloaded data.
    cache_dir: Option<PathBuf>,
    /// Random seed for sampling.
    seed: u64,
}

impl Default for CalibrationDataConfig {
    fn default() -> Self {
        Self {
            num_samples: 128,
            batch_size: 1,
            sequence_length: 2048,
            dataset: "c4".to_string(),
            cache_dir: None,
            seed: 42,
        }
    }
}

impl CalibrationDataConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of calibration samples.
    pub fn with_num_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Set the sequence length.
    pub fn with_sequence_length(mut self, len: usize) -> Self {
        self.sequence_length = len;
        self
    }

    /// Set the dataset name or path.
    pub fn with_dataset(mut self, dataset: impl Into<String>) -> Self {
        self.dataset = dataset.into();
        self
    }

    /// Set the cache directory.
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Get the number of samples.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the sequence length.
    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Get the dataset name.
    pub fn dataset(&self) -> &str {
        &self.dataset
    }

    /// Get the cache directory.
    pub fn cache_dir(&self) -> Option<&PathBuf> {
        self.cache_dir.as_ref()
    }

    /// Get the random seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Get the number of batches.
    pub fn num_batches(&self) -> usize {
        self.num_samples.div_ceil(self.batch_size)
    }
}

/// Calibration data loader for pruning.
///
/// Provides an iterator over calibration batches for collecting
/// activation statistics during pruning.
#[derive(Debug, Clone)]
pub struct CalibrationDataLoader {
    /// Configuration.
    config: CalibrationDataConfig,
    /// Pre-loaded data (if available).
    data: Option<Vec<Batch>>,
    /// Current position in iteration.
    position: usize,
}

impl CalibrationDataLoader {
    /// Create a new calibration data loader.
    pub fn new(config: CalibrationDataConfig) -> Self {
        Self {
            config,
            data: None,
            position: 0,
        }
    }

    /// Create a data loader with pre-loaded synthetic data for testing.
    pub fn with_synthetic_data(config: CalibrationDataConfig) -> Self {
        let mut loader = Self::new(config);
        loader.generate_synthetic_data();
        loader
    }

    /// Generate synthetic calibration data for testing.
    fn generate_synthetic_data(&mut self) {
        use rand::prelude::*;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut batches = Vec::with_capacity(self.config.num_batches());

        let mut samples_remaining = self.config.num_samples;

        while samples_remaining > 0 {
            let batch_size = samples_remaining.min(self.config.batch_size);
            samples_remaining -= batch_size;

            // Generate random input data
            let input_size = batch_size * self.config.sequence_length;
            let inputs: Vec<f32> = (0..input_size).map(|_| rng.random::<f32>()).collect();
            let targets: Vec<f32> = (0..batch_size).map(|_| rng.random::<f32>()).collect();

            batches.push(Batch::new(
                Tensor::from_vec(inputs, false),
                Tensor::from_vec(targets, false),
            ));
        }

        self.data = Some(batches);
    }

    /// Load data from the configured source.
    ///
    /// This is a placeholder for actual dataset loading.
    /// In production, this would load from C4, WikiText, etc.
    pub fn load(&mut self) -> Result<(), String> {
        if self.data.is_some() {
            return Ok(());
        }

        // For now, generate synthetic data
        // Real implementation would load from dataset
        self.generate_synthetic_data();
        Ok(())
    }

    /// Get the configuration.
    pub fn config(&self) -> &CalibrationDataConfig {
        &self.config
    }

    /// Check if data is loaded.
    pub fn is_loaded(&self) -> bool {
        self.data.is_some()
    }

    /// Get the number of batches.
    pub fn num_batches(&self) -> usize {
        self.data.as_ref().map_or(0, Vec::len)
    }

    /// Reset the iterator position.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get a batch by index.
    pub fn get_batch(&self, index: usize) -> Option<&Batch> {
        self.data.as_ref().and_then(|d| d.get(index))
    }

    /// Create an iterator over batches.
    pub fn iter(&self) -> CalibrationDataIter<'_> {
        CalibrationDataIter {
            loader: self,
            position: 0,
        }
    }
}

/// Iterator over calibration batches.
pub struct CalibrationDataIter<'a> {
    loader: &'a CalibrationDataLoader,
    position: usize,
}

impl<'a> Iterator for CalibrationDataIter<'a> {
    type Item = &'a Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.loader.get_batch(self.position)?;
        self.position += 1;
        Some(batch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.loader.num_batches().saturating_sub(self.position);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CalibrationDataIter<'_> {}

impl<'a> IntoIterator for &'a CalibrationDataLoader {
    type Item = &'a Batch;
    type IntoIter = CalibrationDataIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CalibrationDataConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        // TEST_ID: DL-001
        let config = CalibrationDataConfig::default();
        assert_eq!(config.num_samples(), 128);
        assert_eq!(config.batch_size(), 1);
        assert_eq!(config.sequence_length(), 2048);
        assert_eq!(config.dataset(), "c4");
        assert_eq!(config.seed(), 42);
    }

    #[test]
    fn test_config_builder() {
        // TEST_ID: DL-002
        let config = CalibrationDataConfig::new()
            .with_num_samples(256)
            .with_batch_size(4)
            .with_sequence_length(1024)
            .with_dataset("wikitext")
            .with_seed(123);

        assert_eq!(config.num_samples(), 256);
        assert_eq!(config.batch_size(), 4);
        assert_eq!(config.sequence_length(), 1024);
        assert_eq!(config.dataset(), "wikitext");
        assert_eq!(config.seed(), 123);
    }

    #[test]
    fn test_config_batch_size_minimum() {
        // TEST_ID: DL-003
        let config = CalibrationDataConfig::new().with_batch_size(0);
        assert_eq!(
            config.batch_size(),
            1,
            "DL-003 FALSIFIED: Batch size should be minimum 1"
        );
    }

    #[test]
    fn test_config_num_batches() {
        // TEST_ID: DL-004
        let config = CalibrationDataConfig::new()
            .with_num_samples(10)
            .with_batch_size(3);
        // 10 samples / 3 per batch = ceil(10/3) = 4 batches
        assert_eq!(
            config.num_batches(),
            4,
            "DL-004 FALSIFIED: 10 samples with batch_size 3 should be 4 batches"
        );
    }

    #[test]
    fn test_config_cache_dir() {
        // TEST_ID: DL-005
        let config = CalibrationDataConfig::new().with_cache_dir("/tmp/cache");
        assert_eq!(
            config.cache_dir().map(|p| p.to_str().unwrap()),
            Some("/tmp/cache")
        );
    }

    #[test]
    fn test_config_serialize_json() {
        // TEST_ID: DL-006
        let config = CalibrationDataConfig::new().with_num_samples(64);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CalibrationDataConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.num_samples(), deserialized.num_samples());
    }

    // =========================================================================
    // CalibrationDataLoader Tests
    // =========================================================================

    #[test]
    fn test_loader_new() {
        // TEST_ID: DL-010
        let config = CalibrationDataConfig::new();
        let loader = CalibrationDataLoader::new(config);
        assert!(!loader.is_loaded());
        assert_eq!(loader.num_batches(), 0);
    }

    #[test]
    fn test_loader_with_synthetic_data() {
        // TEST_ID: DL-011
        let config = CalibrationDataConfig::new()
            .with_num_samples(10)
            .with_batch_size(3);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        assert!(loader.is_loaded());
        assert_eq!(
            loader.num_batches(),
            4,
            "DL-011 FALSIFIED: Should have 4 batches"
        );
    }

    #[test]
    fn test_loader_load() {
        // TEST_ID: DL-012
        let config = CalibrationDataConfig::new().with_num_samples(5);
        let mut loader = CalibrationDataLoader::new(config);

        assert!(!loader.is_loaded());
        loader.load().unwrap();
        assert!(loader.is_loaded());
    }

    #[test]
    fn test_loader_get_batch() {
        // TEST_ID: DL-013
        let config = CalibrationDataConfig::new()
            .with_num_samples(10)
            .with_batch_size(5);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        assert!(loader.get_batch(0).is_some());
        assert!(loader.get_batch(1).is_some());
        assert!(loader.get_batch(2).is_none()); // Only 2 batches
    }

    #[test]
    fn test_loader_iter() {
        // TEST_ID: DL-014
        let config = CalibrationDataConfig::new()
            .with_num_samples(9)
            .with_batch_size(3);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(
            batches.len(),
            3,
            "DL-014 FALSIFIED: Should iterate over 3 batches"
        );
    }

    #[test]
    fn test_loader_iter_size_hint() {
        // TEST_ID: DL-015
        let config = CalibrationDataConfig::new()
            .with_num_samples(6)
            .with_batch_size(2);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        let iter = loader.iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_loader_reset() {
        // TEST_ID: DL-016
        let config = CalibrationDataConfig::new().with_num_samples(5);
        let mut loader = CalibrationDataLoader::with_synthetic_data(config);
        loader.position = 3;
        loader.reset();
        assert_eq!(
            loader.position, 0,
            "DL-016 FALSIFIED: Reset should set position to 0"
        );
    }

    #[test]
    fn test_loader_deterministic_with_seed() {
        // TEST_ID: DL-017
        let config = CalibrationDataConfig::new()
            .with_num_samples(5)
            .with_seed(12345);

        let loader1 = CalibrationDataLoader::with_synthetic_data(config.clone());
        let loader2 = CalibrationDataLoader::with_synthetic_data(config);

        let batch1 = loader1.get_batch(0).unwrap();
        let batch2 = loader2.get_batch(0).unwrap();

        // Same seed should produce same data
        let data1: Vec<f32> = batch1.inputs.data().to_vec();
        let data2: Vec<f32> = batch2.inputs.data().to_vec();
        assert_eq!(
            data1, data2,
            "DL-017 FALSIFIED: Same seed should produce same data"
        );
    }

    #[test]
    fn test_loader_different_seeds_different_data() {
        // TEST_ID: DL-018
        let config1 = CalibrationDataConfig::new()
            .with_num_samples(5)
            .with_seed(111);
        let config2 = CalibrationDataConfig::new()
            .with_num_samples(5)
            .with_seed(222);

        let loader1 = CalibrationDataLoader::with_synthetic_data(config1);
        let loader2 = CalibrationDataLoader::with_synthetic_data(config2);

        let batch1 = loader1.get_batch(0).unwrap();
        let batch2 = loader2.get_batch(0).unwrap();

        let data1: Vec<f32> = batch1.inputs.data().to_vec();
        let data2: Vec<f32> = batch2.inputs.data().to_vec();
        assert_ne!(
            data1, data2,
            "DL-018 FALSIFIED: Different seeds should produce different data"
        );
    }

    #[test]
    fn test_loader_batch_sizes_correct() {
        // TEST_ID: DL-019
        let config = CalibrationDataConfig::new()
            .with_num_samples(10)
            .with_batch_size(4)
            .with_sequence_length(128);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        // First two batches should have 4 samples each (4 * 128 = 512 elements)
        // Last batch should have 2 samples (2 * 128 = 256 elements)
        let batch0 = loader.get_batch(0).unwrap();
        let batch1 = loader.get_batch(1).unwrap();
        let batch2 = loader.get_batch(2).unwrap();

        assert_eq!(batch0.inputs.len(), 4 * 128);
        assert_eq!(batch1.inputs.len(), 4 * 128);
        assert_eq!(batch2.inputs.len(), 2 * 128);
    }

    #[test]
    fn test_loader_config_access() {
        // TEST_ID: DL-020
        let config = CalibrationDataConfig::new().with_num_samples(50);
        let loader = CalibrationDataLoader::new(config);
        assert_eq!(loader.config().num_samples(), 50);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_loader_single_sample() {
        // TEST_ID: DL-030
        let config = CalibrationDataConfig::new()
            .with_num_samples(1)
            .with_batch_size(1);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        assert_eq!(loader.num_batches(), 1);
        assert!(loader.get_batch(0).is_some());
    }

    #[test]
    fn test_loader_batch_size_larger_than_samples() {
        // TEST_ID: DL-031
        let config = CalibrationDataConfig::new()
            .with_num_samples(3)
            .with_batch_size(10);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        assert_eq!(
            loader.num_batches(),
            1,
            "DL-031 FALSIFIED: Should have 1 batch when batch_size > num_samples"
        );
        let batch = loader.get_batch(0).unwrap();
        // Should have only 3 samples worth of data
        assert_eq!(batch.inputs.len(), 3 * 2048); // 3 samples * default seq len
    }

    #[test]
    fn test_loader_empty_iter() {
        // TEST_ID: DL-032
        let config = CalibrationDataConfig::new();
        let loader = CalibrationDataLoader::new(config);

        let count = loader.iter().count();
        assert_eq!(
            count, 0,
            "DL-032 FALSIFIED: Unloaded loader should have empty iterator"
        );
    }

    #[test]
    fn test_loader_clone() {
        // TEST_ID: DL-033
        let config = CalibrationDataConfig::new().with_num_samples(5);
        let loader = CalibrationDataLoader::with_synthetic_data(config);
        let cloned = loader.clone();

        assert_eq!(loader.num_batches(), cloned.num_batches());
        assert!(cloned.is_loaded());
    }
}
