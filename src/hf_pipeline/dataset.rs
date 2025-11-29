//! HuggingFace Dataset Fetcher and Collator
//!
//! Provides dataset loading and batching for distillation training.
//!
//! # Features
//!
//! - Streaming support for large datasets
//! - Parquet file loading
//! - Dynamic padding and batching
//! - Teacher output caching

use crate::hf_pipeline::error::{FetchError, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Dataset split type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Split {
    /// Training split
    Train,
    /// Validation split
    Validation,
    /// Test split
    Test,
}

impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Train => write!(f, "train"),
            Self::Validation => write!(f, "validation"),
            Self::Test => write!(f, "test"),
        }
    }
}

/// Dataset fetch options
#[derive(Debug, Clone)]
pub struct DatasetOptions {
    /// Dataset split to load
    pub split: Split,
    /// Maximum number of examples (None = all)
    pub max_examples: Option<usize>,
    /// Stream data instead of loading all at once
    pub streaming: bool,
    /// Shuffle data
    pub shuffle: bool,
    /// Random seed for shuffling
    pub seed: Option<u64>,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
}

impl Default for DatasetOptions {
    fn default() -> Self {
        Self {
            split: Split::Train,
            max_examples: None,
            streaming: false,
            shuffle: true,
            seed: Some(42),
            cache_dir: None,
        }
    }
}

impl DatasetOptions {
    /// Create new options for training split
    #[must_use]
    pub fn train() -> Self {
        Self::default()
    }

    /// Create new options for validation split
    #[must_use]
    pub fn validation() -> Self {
        Self {
            split: Split::Validation,
            shuffle: false,
            ..Default::default()
        }
    }

    /// Create new options for test split
    #[must_use]
    pub fn test() -> Self {
        Self {
            split: Split::Test,
            shuffle: false,
            ..Default::default()
        }
    }

    /// Set maximum examples
    #[must_use]
    pub fn max_examples(mut self, n: usize) -> Self {
        self.max_examples = Some(n);
        self
    }

    /// Enable streaming
    #[must_use]
    pub fn streaming(mut self, enabled: bool) -> Self {
        self.streaming = enabled;
        self
    }

    /// Set shuffle
    #[must_use]
    pub fn shuffle(mut self, enabled: bool) -> Self {
        self.shuffle = enabled;
        self
    }

    /// Set random seed
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// A single example from a dataset
#[derive(Debug, Clone)]
pub struct Example {
    /// Input token IDs
    pub input_ids: Vec<u32>,
    /// Attention mask (1 = attend, 0 = ignore)
    pub attention_mask: Vec<u8>,
    /// Target labels (for supervised learning)
    pub labels: Option<Vec<u32>>,
    /// Text content (if available)
    pub text: Option<String>,
}

impl Example {
    /// Create new example from token IDs
    #[must_use]
    pub fn from_tokens(input_ids: Vec<u32>) -> Self {
        let len = input_ids.len();
        Self {
            input_ids,
            attention_mask: vec![1; len],
            labels: None,
            text: None,
        }
    }

    /// Set labels
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<u32>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Set text
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Get sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}

/// Dataset abstraction
pub struct Dataset {
    /// Dataset name/ID
    name: String,
    /// Examples
    examples: Vec<Example>,
    /// Current position for iteration
    position: usize,
}

impl Dataset {
    /// Create new dataset from examples
    #[must_use]
    pub fn new(name: impl Into<String>, examples: Vec<Example>) -> Self {
        Self {
            name: name.into(),
            examples,
            position: 0,
        }
    }

    /// Create mock dataset for testing
    #[must_use]
    pub fn mock(num_examples: usize, seq_len: usize) -> Self {
        let examples: Vec<Example> = (0..num_examples)
            .map(|i| {
                Example::from_tokens((0..seq_len).map(|j| ((i + j) % 30000) as u32).collect())
                    .with_labels((0..seq_len).map(|j| ((i + j + 1) % 30000) as u32).collect())
            })
            .collect();

        Self::new("mock_dataset", examples)
    }

    /// Get dataset name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get number of examples
    #[must_use]
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get example by index
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Example> {
        self.examples.get(index)
    }

    /// Get all examples
    #[must_use]
    pub fn examples(&self) -> &[Example] {
        &self.examples
    }

    /// Reset iteration position
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Shuffle examples
    pub fn shuffle(&mut self, seed: u64) {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        self.examples.shuffle(&mut rng);
    }

    /// Take a subset of examples
    #[must_use]
    pub fn take(mut self, n: usize) -> Self {
        self.examples.truncate(n);
        self
    }
}

impl Iterator for Dataset {
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.examples.len() {
            let example = self.examples[self.position].clone();
            self.position += 1;
            Some(example)
        } else {
            None
        }
    }
}

/// HuggingFace dataset fetcher
pub struct HfDatasetFetcher {
    /// HuggingFace token
    #[allow(dead_code)]
    token: Option<String>,
    /// Cache directory
    cache_dir: PathBuf,
}

impl HfDatasetFetcher {
    /// Create new fetcher
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("huggingface")
            .join("datasets");

        Ok(Self {
            token: std::env::var("HF_TOKEN").ok(),
            cache_dir,
        })
    }

    /// Set cache directory
    #[must_use]
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = path.into();
        self
    }

    /// Fetch dataset from HuggingFace
    ///
    /// # Arguments
    ///
    /// * `dataset_id` - Dataset ID (e.g., "wikitext", "squad")
    /// * `options` - Fetch options
    pub fn fetch(&self, dataset_id: &str, options: DatasetOptions) -> Result<Dataset> {
        // Validate dataset ID
        if dataset_id.is_empty() {
            return Err(FetchError::InvalidRepoId {
                repo_id: dataset_id.into(),
            });
        }

        // For now, create mock dataset (actual HF API integration later)
        let num_examples = options.max_examples.unwrap_or(1000);
        let mut dataset = Dataset::mock(num_examples, 128);

        if options.shuffle {
            if let Some(seed) = options.seed {
                dataset.shuffle(seed);
            }
        }

        Ok(dataset)
    }

    /// Load dataset from local parquet file
    pub fn load_parquet(&self, path: &std::path::Path) -> Result<Dataset> {
        if !path.exists() {
            return Err(FetchError::FileNotFound {
                repo: path.parent().unwrap_or(path).display().to_string(),
                file: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into(),
            });
        }

        // Mock implementation - actual parquet parsing later
        Ok(Dataset::mock(100, 128))
    }
}

impl Default for HfDatasetFetcher {
    fn default() -> Self {
        Self::new().expect("Failed to create dataset fetcher")
    }
}

/// Batch of examples for training
#[derive(Debug, Clone)]
pub struct Batch {
    /// Input IDs [batch_size, max_seq_len]
    pub input_ids: Array2<u32>,
    /// Attention mask [batch_size, max_seq_len]
    pub attention_mask: Array2<u8>,
    /// Labels [batch_size, max_seq_len] (optional)
    pub labels: Option<Array2<u32>>,
    /// Original sequence lengths
    pub lengths: Vec<usize>,
}

impl Batch {
    /// Get batch size
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.input_ids.nrows()
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.input_ids.ncols()
    }
}

/// Collator for batching examples with dynamic padding
pub struct DistillationCollator {
    /// Padding token ID
    pub pad_token_id: u32,
    /// Maximum sequence length (truncate if longer)
    pub max_length: usize,
    /// Padding side (true = left, false = right)
    pub pad_left: bool,
}

impl Default for DistillationCollator {
    fn default() -> Self {
        Self {
            pad_token_id: 0,
            max_length: 512,
            pad_left: false,
        }
    }
}

impl DistillationCollator {
    /// Create new collator
    #[must_use]
    pub fn new(pad_token_id: u32) -> Self {
        Self {
            pad_token_id,
            ..Default::default()
        }
    }

    /// Set maximum length
    #[must_use]
    pub fn max_length(mut self, len: usize) -> Self {
        self.max_length = len;
        self
    }

    /// Set padding side
    #[must_use]
    pub fn pad_left(mut self, left: bool) -> Self {
        self.pad_left = left;
        self
    }

    /// Collate examples into a batch
    #[must_use]
    pub fn collate(&self, examples: &[Example]) -> Batch {
        if examples.is_empty() {
            return Batch {
                input_ids: Array2::zeros((0, 0)),
                attention_mask: Array2::zeros((0, 0)),
                labels: None,
                lengths: vec![],
            };
        }

        // Find max length in batch (capped at max_length)
        let max_len = examples
            .iter()
            .map(|e| e.len().min(self.max_length))
            .max()
            .unwrap_or(0);

        let batch_size = examples.len();
        let mut input_ids = Array2::from_elem((batch_size, max_len), self.pad_token_id);
        let mut attention_mask = Array2::zeros((batch_size, max_len));
        let mut lengths = Vec::with_capacity(batch_size);

        let has_labels = examples.iter().any(|e| e.labels.is_some());
        let mut labels = if has_labels {
            Some(Array2::from_elem((batch_size, max_len), self.pad_token_id))
        } else {
            None
        };

        for (i, example) in examples.iter().enumerate() {
            let seq_len = example.len().min(self.max_length);
            lengths.push(seq_len);

            let (start, end) = if self.pad_left {
                (max_len - seq_len, max_len)
            } else {
                (0, seq_len)
            };

            // Copy input IDs
            for (j, &token) in example.input_ids.iter().take(seq_len).enumerate() {
                input_ids[[i, start + j]] = token;
            }

            // Set attention mask
            for j in start..end {
                attention_mask[[i, j]] = 1;
            }

            // Copy labels if present
            if let (Some(ref mut label_arr), Some(ref ex_labels)) = (&mut labels, &example.labels) {
                for (j, &token) in ex_labels.iter().take(seq_len).enumerate() {
                    label_arr[[i, start + j]] = token;
                }
            }
        }

        Batch {
            input_ids,
            attention_mask,
            labels,
            lengths,
        }
    }

    /// Create batches from dataset
    pub fn batch_dataset(&self, dataset: &Dataset, batch_size: usize) -> Vec<Batch> {
        dataset
            .examples()
            .chunks(batch_size)
            .map(|chunk| self.collate(chunk))
            .collect()
    }
}

/// Cached teacher outputs for distillation
#[derive(Debug, Clone)]
pub struct TeacherCache {
    /// Cached logits by example index
    logits: std::collections::HashMap<usize, Array2<f32>>,
    /// Cached hidden states by example index
    hidden_states: std::collections::HashMap<usize, Vec<Array2<f32>>>,
    /// Cache hit count
    hits: usize,
    /// Cache miss count
    misses: usize,
}

impl Default for TeacherCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TeacherCache {
    /// Create new empty cache
    #[must_use]
    pub fn new() -> Self {
        Self {
            logits: std::collections::HashMap::new(),
            hidden_states: std::collections::HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Get cached logits
    pub fn get_logits(&mut self, index: usize) -> Option<&Array2<f32>> {
        if self.logits.contains_key(&index) {
            self.hits += 1;
            self.logits.get(&index)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Cache logits
    pub fn cache_logits(&mut self, index: usize, logits: Array2<f32>) {
        self.logits.insert(index, logits);
    }

    /// Get cached hidden states
    pub fn get_hidden_states(&mut self, index: usize) -> Option<&Vec<Array2<f32>>> {
        if self.hidden_states.contains_key(&index) {
            self.hits += 1;
            self.hidden_states.get(&index)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Cache hidden states
    pub fn cache_hidden_states(&mut self, index: usize, states: Vec<Array2<f32>>) {
        self.hidden_states.insert(index, states);
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            logits_cached: self.logits.len(),
            hidden_states_cached: self.hidden_states.len(),
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.logits.clear();
        self.hidden_states.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

/// Cache statistics
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of cached logits entries
    pub logits_cached: usize,
    /// Number of cached hidden state entries
    pub hidden_states_cached: usize,
}

impl CacheStats {
    /// Get hit rate
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f32 / total as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Split Tests
    // =========================================================================

    #[test]
    fn test_split_display() {
        assert_eq!(format!("{}", Split::Train), "train");
        assert_eq!(format!("{}", Split::Validation), "validation");
        assert_eq!(format!("{}", Split::Test), "test");
    }

    // =========================================================================
    // DatasetOptions Tests
    // =========================================================================

    #[test]
    fn test_dataset_options_default() {
        let opts = DatasetOptions::default();
        assert_eq!(opts.split, Split::Train);
        assert!(opts.shuffle);
        assert!(!opts.streaming);
    }

    #[test]
    fn test_dataset_options_builders() {
        let train = DatasetOptions::train();
        assert_eq!(train.split, Split::Train);
        assert!(train.shuffle);

        let val = DatasetOptions::validation();
        assert_eq!(val.split, Split::Validation);
        assert!(!val.shuffle);

        let test = DatasetOptions::test();
        assert_eq!(test.split, Split::Test);
        assert!(!test.shuffle);
    }

    #[test]
    fn test_dataset_options_chaining() {
        let opts = DatasetOptions::train()
            .max_examples(100)
            .streaming(true)
            .shuffle(false)
            .seed(123);

        assert_eq!(opts.max_examples, Some(100));
        assert!(opts.streaming);
        assert!(!opts.shuffle);
        assert_eq!(opts.seed, Some(123));
    }

    // =========================================================================
    // Example Tests
    // =========================================================================

    #[test]
    fn test_example_creation() {
        let example = Example::from_tokens(vec![1, 2, 3, 4, 5]);
        assert_eq!(example.len(), 5);
        assert!(!example.is_empty());
        assert_eq!(example.attention_mask, vec![1, 1, 1, 1, 1]);
        assert!(example.labels.is_none());
    }

    #[test]
    fn test_example_with_labels() {
        let example = Example::from_tokens(vec![1, 2, 3]).with_labels(vec![2, 3, 4]);
        assert_eq!(example.labels, Some(vec![2, 3, 4]));
    }

    #[test]
    fn test_example_with_text() {
        let example = Example::from_tokens(vec![1, 2]).with_text("hello world");
        assert_eq!(example.text, Some("hello world".to_string()));
    }

    // =========================================================================
    // Dataset Tests
    // =========================================================================

    #[test]
    fn test_dataset_mock() {
        let dataset = Dataset::mock(10, 32);
        assert_eq!(dataset.len(), 10);
        assert_eq!(dataset.name(), "mock_dataset");

        let first = dataset.get(0).unwrap();
        assert_eq!(first.len(), 32);
        assert!(first.labels.is_some());
    }

    #[test]
    fn test_dataset_iteration() {
        let dataset = Dataset::mock(3, 16);
        let collected: Vec<_> = dataset.collect();
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn test_dataset_shuffle() {
        let mut dataset1 = Dataset::mock(100, 16);
        let mut dataset2 = Dataset::mock(100, 16);

        dataset1.shuffle(42);
        dataset2.shuffle(42);

        // Same seed should produce same order
        for (e1, e2) in dataset1.examples().iter().zip(dataset2.examples().iter()) {
            assert_eq!(e1.input_ids, e2.input_ids);
        }
    }

    #[test]
    fn test_dataset_take() {
        let dataset = Dataset::mock(100, 16).take(10);
        assert_eq!(dataset.len(), 10);
    }

    // =========================================================================
    // HfDatasetFetcher Tests
    // =========================================================================

    #[test]
    fn test_fetcher_creation() {
        let fetcher = HfDatasetFetcher::new();
        assert!(fetcher.is_ok());
    }

    #[test]
    fn test_fetcher_fetch_mock() {
        let fetcher = HfDatasetFetcher::default();
        let dataset = fetcher.fetch("wikitext", DatasetOptions::train().max_examples(50));
        assert!(dataset.is_ok());
        assert_eq!(dataset.unwrap().len(), 50);
    }

    #[test]
    fn test_fetcher_invalid_id() {
        let fetcher = HfDatasetFetcher::default();
        let result = fetcher.fetch("", DatasetOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_fetcher_load_nonexistent_parquet() {
        let fetcher = HfDatasetFetcher::default();
        let result = fetcher.load_parquet(std::path::Path::new("/nonexistent/file.parquet"));
        assert!(result.is_err());
    }

    // =========================================================================
    // Batch Tests
    // =========================================================================

    #[test]
    fn test_batch_dimensions() {
        let examples = vec![
            Example::from_tokens(vec![1, 2, 3]),
            Example::from_tokens(vec![4, 5]),
        ];
        let collator = DistillationCollator::default();
        let batch = collator.collate(&examples);

        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.max_seq_len(), 3);
    }

    // =========================================================================
    // DistillationCollator Tests
    // =========================================================================

    #[test]
    fn test_collator_default() {
        let collator = DistillationCollator::default();
        assert_eq!(collator.pad_token_id, 0);
        assert_eq!(collator.max_length, 512);
        assert!(!collator.pad_left);
    }

    #[test]
    fn test_collator_builder() {
        let collator = DistillationCollator::new(1).max_length(256).pad_left(true);
        assert_eq!(collator.pad_token_id, 1);
        assert_eq!(collator.max_length, 256);
        assert!(collator.pad_left);
    }

    #[test]
    fn test_collator_empty_batch() {
        let collator = DistillationCollator::default();
        let batch = collator.collate(&[]);
        assert_eq!(batch.batch_size(), 0);
    }

    #[test]
    fn test_collator_right_padding() {
        let examples = vec![
            Example::from_tokens(vec![1, 2, 3]),
            Example::from_tokens(vec![4, 5]),
        ];
        let collator = DistillationCollator::new(0);
        let batch = collator.collate(&examples);

        // First example: [1, 2, 3]
        assert_eq!(batch.input_ids[[0, 0]], 1);
        assert_eq!(batch.input_ids[[0, 1]], 2);
        assert_eq!(batch.input_ids[[0, 2]], 3);

        // Second example: [4, 5, PAD]
        assert_eq!(batch.input_ids[[1, 0]], 4);
        assert_eq!(batch.input_ids[[1, 1]], 5);
        assert_eq!(batch.input_ids[[1, 2]], 0); // Padded

        // Attention mask
        assert_eq!(batch.attention_mask[[0, 2]], 1); // Not padded
        assert_eq!(batch.attention_mask[[1, 2]], 0); // Padded
    }

    #[test]
    fn test_collator_left_padding() {
        let examples = vec![
            Example::from_tokens(vec![1, 2, 3]),
            Example::from_tokens(vec![4, 5]),
        ];
        let collator = DistillationCollator::new(0).pad_left(true);
        let batch = collator.collate(&examples);

        // Second example: [PAD, 4, 5]
        assert_eq!(batch.input_ids[[1, 0]], 0); // Padded
        assert_eq!(batch.input_ids[[1, 1]], 4);
        assert_eq!(batch.input_ids[[1, 2]], 5);

        // Attention mask
        assert_eq!(batch.attention_mask[[1, 0]], 0); // Padded
        assert_eq!(batch.attention_mask[[1, 1]], 1);
    }

    #[test]
    fn test_collator_truncation() {
        let examples = vec![Example::from_tokens(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])];
        let collator = DistillationCollator::new(0).max_length(5);
        let batch = collator.collate(&examples);

        assert_eq!(batch.max_seq_len(), 5);
        assert_eq!(batch.input_ids[[0, 0]], 1);
        assert_eq!(batch.input_ids[[0, 4]], 5);
    }

    #[test]
    fn test_collator_with_labels() {
        let examples = vec![
            Example::from_tokens(vec![1, 2]).with_labels(vec![2, 3]),
            Example::from_tokens(vec![4, 5, 6]).with_labels(vec![5, 6, 7]),
        ];
        let collator = DistillationCollator::new(0);
        let batch = collator.collate(&examples);

        assert!(batch.labels.is_some());
        let labels = batch.labels.unwrap();
        assert_eq!(labels[[0, 0]], 2);
        assert_eq!(labels[[1, 2]], 7);
    }

    #[test]
    fn test_collator_lengths() {
        let examples = vec![
            Example::from_tokens(vec![1, 2, 3]),
            Example::from_tokens(vec![4, 5]),
            Example::from_tokens(vec![6]),
        ];
        let collator = DistillationCollator::default();
        let batch = collator.collate(&examples);

        assert_eq!(batch.lengths, vec![3, 2, 1]);
    }

    #[test]
    fn test_collator_batch_dataset() {
        let dataset = Dataset::mock(10, 16);
        let collator = DistillationCollator::default();
        let batches = collator.batch_dataset(&dataset, 3);

        assert_eq!(batches.len(), 4); // 10 / 3 = 3 full + 1 partial
        assert_eq!(batches[0].batch_size(), 3);
        assert_eq!(batches[3].batch_size(), 1); // Last batch is partial
    }

    // =========================================================================
    // TeacherCache Tests
    // =========================================================================

    #[test]
    fn test_cache_new() {
        let cache = TeacherCache::new();
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_logits() {
        let mut cache = TeacherCache::new();
        let logits = Array2::<f32>::zeros((4, 100));

        // Miss first
        assert!(cache.get_logits(0).is_none());

        // Cache and hit
        cache.cache_logits(0, logits.clone());
        assert!(cache.get_logits(0).is_some());

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_hidden_states() {
        let mut cache = TeacherCache::new();
        let states = vec![Array2::<f32>::zeros((4, 768)); 12];

        cache.cache_hidden_states(0, states);
        assert!(cache.get_hidden_states(0).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = TeacherCache::new();
        cache.cache_logits(0, Array2::<f32>::zeros((4, 100)));
        cache.clear();

        let stats = cache.stats();
        assert_eq!(stats.logits_cached, 0);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = TeacherCache::new();
        let logits = Array2::<f32>::zeros((4, 100));

        // 1 miss, then 3 hits
        let _ = cache.get_logits(0);
        cache.cache_logits(0, logits);
        let _ = cache.get_logits(0);
        let _ = cache.get_logits(0);
        let _ = cache.get_logits(0);

        let stats = cache.stats();
        assert_eq!(stats.hit_rate(), 0.75); // 3 hits / 4 total
    }
}
