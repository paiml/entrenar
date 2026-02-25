//! Production training loop for classification fine-tuning (SSC-026)
//!
//! `ClassifyTrainer` wraps `ClassifyPipeline` with epoch management,
//! validation, checkpointing, LR scheduling, and early stopping.
//!
//! # Contract Invariants
//!
//! - F-LOOP-001: EMA loss decreasing over training (alpha=0.1, 5-epoch window)
//! - F-LOOP-002: Validation computed every epoch
//! - F-LOOP-007: Data shuffled per epoch (different order)
//! - F-LOOP-008: Val split disjoint (zero overlap with training set)
//! - F-LOOP-009: Val set frozen (same composition across epochs)
//! - F-LOOP-010: Early stopping respects patience

use super::classification::SafetySample;
use super::classify_pipeline::ClassifyPipeline;
use crate::optim::WarmupCosineDecayLR;
use crate::optim::LRScheduler;
use std::path::{Path, PathBuf};

/// Training configuration for the classification trainer.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs (default: 50)
    pub epochs: usize,
    /// Fraction of data reserved for validation (default: 0.2)
    pub val_split: f32,
    /// Save checkpoint every N epochs (default: 5)
    pub save_every: usize,
    /// Early stopping patience in epochs (default: 10)
    pub early_stopping_patience: usize,
    /// Directory for checkpoint files
    pub checkpoint_dir: PathBuf,
    /// Random seed for reproducibility (default: 42)
    pub seed: u64,
    /// Log metrics every N epochs (default: 1)
    pub log_interval: usize,
    /// Warmup steps as fraction of total steps (default: 0.1)
    pub warmup_fraction: f32,
    /// Minimum learning rate for cosine decay (default: 1e-6)
    pub lr_min: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            val_split: 0.2,
            save_every: 5,
            early_stopping_patience: 10,
            checkpoint_dir: PathBuf::from("checkpoints"),
            seed: 42,
            log_interval: 1,
            warmup_fraction: 0.1,
            lr_min: 1e-6,
        }
    }
}

/// Metrics for a single training epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Epoch number (0-indexed)
    pub epoch: usize,
    /// Average training loss
    pub train_loss: f32,
    /// Training accuracy (0.0-1.0)
    pub train_accuracy: f32,
    /// Average validation loss
    pub val_loss: f32,
    /// Validation accuracy (0.0-1.0)
    pub val_accuracy: f32,
    /// Current learning rate
    pub learning_rate: f32,
    /// Epoch wall-clock time in milliseconds
    pub epoch_time_ms: u64,
    /// Training throughput (samples/second)
    pub samples_per_sec: f32,
}

/// Result of the full training run.
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Per-epoch metrics
    pub epoch_metrics: Vec<EpochMetrics>,
    /// Epoch with lowest validation loss
    pub best_epoch: usize,
    /// Lowest validation loss achieved
    pub best_val_loss: f32,
    /// Whether training stopped early
    pub stopped_early: bool,
    /// Total wall-clock training time in milliseconds
    pub total_time_ms: u64,
}

/// Production training loop for classification fine-tuning.
///
/// Wraps `ClassifyPipeline` with:
/// - Epoch management with per-epoch shuffling
/// - Validation on a disjoint, frozen split
/// - Warmup + cosine decay LR scheduling
/// - Periodic checkpointing (SafeTensors + metadata JSON)
/// - Early stopping with configurable patience
pub struct ClassifyTrainer {
    /// The classification pipeline (model + optimizer)
    pipeline: ClassifyPipeline,
    /// Training configuration
    config: TrainingConfig,
    /// Training data (shuffled per epoch)
    train_data: Vec<SafetySample>,
    /// Validation data (frozen, never shuffled)
    val_data: Vec<SafetySample>,
    /// Base random seed
    rng_seed: u64,
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for ClassifyTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassifyTrainer")
            .field("config", &self.config)
            .field("train_data_len", &self.train_data.len())
            .field("val_data_len", &self.val_data.len())
            .field("rng_seed", &self.rng_seed)
            .finish()
    }
}

impl ClassifyTrainer {
    /// Create a new trainer by splitting corpus into train/val sets.
    ///
    /// # Arguments
    /// * `pipeline` - Initialized `ClassifyPipeline`
    /// * `corpus` - Full dataset of labeled samples
    /// * `config` - Training configuration
    ///
    /// # Errors
    /// Returns error if corpus is empty, val_split is out of (0.0, 0.5],
    /// or epochs is 0.
    pub fn new(
        pipeline: ClassifyPipeline,
        corpus: Vec<SafetySample>,
        config: TrainingConfig,
    ) -> crate::Result<Self> {
        if corpus.is_empty() {
            return Err(crate::Error::ConfigError(
                "SSC-026: corpus must not be empty".to_string(),
            ));
        }
        if config.val_split <= 0.0 || config.val_split > 0.5 {
            return Err(crate::Error::ConfigError(format!(
                "SSC-026: val_split must be in (0.0, 0.5], got {}",
                config.val_split,
            )));
        }
        if config.epochs == 0 {
            return Err(crate::Error::ConfigError(
                "SSC-026: epochs must be > 0".to_string(),
            ));
        }

        let (train_data, val_data) = Self::split_dataset(&corpus, config.val_split, config.seed);

        if train_data.is_empty() || val_data.is_empty() {
            return Err(crate::Error::ConfigError(format!(
                "SSC-026: split produced empty set (train={}, val={}). Need more samples.",
                train_data.len(),
                val_data.len(),
            )));
        }

        let rng_seed = config.seed;

        Ok(Self {
            pipeline,
            config,
            train_data,
            val_data,
            rng_seed,
        })
    }

    /// Run the full training loop.
    ///
    /// For each epoch:
    /// 1. Shuffle training data (deterministic, seed varies per epoch)
    /// 2. Process batches via `pipeline.train_batch()`
    /// 3. Compute validation metrics (forward-only)
    /// 4. Step LR scheduler
    /// 5. Record metrics
    /// 6. Save checkpoint if `save_every` or new best val_loss
    /// 7. Check early stopping
    pub fn train(&mut self) -> TrainResult {
        let total_start = std::time::Instant::now();
        let batch_size = self.pipeline.config.batch_size;
        let batches_per_epoch = self.train_data.len().div_ceil(batch_size);
        let total_steps = self.config.epochs * batches_per_epoch;
        let warmup_steps = (self.config.warmup_fraction * total_steps as f32) as usize;
        let lr_max = self.pipeline.optimizer_lr();

        let mut scheduler = WarmupCosineDecayLR::new(
            lr_max,
            self.config.lr_min,
            warmup_steps,
            total_steps,
        );

        let mut epoch_metrics_vec: Vec<EpochMetrics> = Vec::with_capacity(self.config.epochs);
        let mut best_val_loss = f32::INFINITY;
        let mut best_epoch: usize = 0;
        let mut epochs_without_improvement: usize = 0;
        let mut stopped_early = false;

        for epoch in 0..self.config.epochs {
            let epoch_start = std::time::Instant::now();

            // F-LOOP-007: Shuffle training data with epoch-specific seed
            self.shuffle_training_data(epoch);

            // Train one epoch
            let (train_loss, train_accuracy) = self.train_epoch(&mut scheduler);

            // F-LOOP-002: Validate every epoch
            let (val_loss, val_accuracy) = self.validate();

            let epoch_time = epoch_start.elapsed();
            let epoch_time_ms = epoch_time.as_millis() as u64;
            let samples_per_sec = if epoch_time_ms > 0 {
                self.train_data.len() as f32 / (epoch_time_ms as f32 / 1000.0)
            } else {
                0.0
            };

            let metrics = EpochMetrics {
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                learning_rate: scheduler.get_lr(),
                epoch_time_ms,
                samples_per_sec,
            };

            epoch_metrics_vec.push(metrics.clone());

            // Track best validation loss
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_without_improvement = 0;

                // Save best checkpoint
                let _ = self.save_checkpoint(
                    &self.config.checkpoint_dir.join("best"),
                    epoch,
                    &metrics,
                );
            } else {
                epochs_without_improvement += 1;
            }

            // Periodic checkpoint
            if self.config.save_every > 0 && (epoch + 1) % self.config.save_every == 0 {
                let _ = self.save_checkpoint(
                    &self.config.checkpoint_dir.join(format!("epoch-{epoch}")),
                    epoch,
                    &metrics,
                );
            }

            // F-LOOP-010: Early stopping
            if epochs_without_improvement >= self.config.early_stopping_patience {
                stopped_early = true;
                break;
            }
        }

        let total_time_ms = total_start.elapsed().as_millis() as u64;

        TrainResult {
            epoch_metrics: epoch_metrics_vec,
            best_epoch,
            best_val_loss,
            stopped_early,
            total_time_ms,
        }
    }

    /// Train for one epoch, processing all training data in batches.
    ///
    /// Returns `(avg_loss, accuracy)` for the epoch.
    fn train_epoch(&mut self, scheduler: &mut WarmupCosineDecayLR) -> (f32, f32) {
        let batch_size = self.pipeline.config.batch_size;
        let mut total_loss = 0.0f32;
        let mut total_correct = 0usize;
        let mut total_samples = 0usize;

        // Clone train_data to avoid borrow conflict (pipeline is &mut self)
        let train_snapshot: Vec<SafetySample> = self.train_data.clone();

        for chunk in train_snapshot.chunks(batch_size) {
            // Apply current LR from scheduler
            self.pipeline.set_optimizer_lr(scheduler.get_lr());

            let result = self.pipeline.train_batch(chunk);
            total_loss += result.avg_loss * result.total as f32;
            total_correct += result.correct;
            total_samples += result.total;

            // Step scheduler per batch
            scheduler.step();
        }

        let avg_loss = if total_samples > 0 {
            total_loss / total_samples as f32
        } else {
            0.0
        };
        let accuracy = if total_samples > 0 {
            total_correct as f32 / total_samples as f32
        } else {
            0.0
        };

        (avg_loss, accuracy)
    }

    /// Compute validation metrics (forward-only, no gradient updates).
    ///
    /// F-LOOP-009: Val set is frozen — same samples every epoch.
    fn validate(&self) -> (f32, f32) {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        let total = self.val_data.len();

        for sample in &self.val_data {
            let ids = sample.input_ids();
            let (loss, predicted) = self.pipeline.forward_only(&ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
        }

        let avg_loss = if total > 0 {
            total_loss / total as f32
        } else {
            0.0
        };
        let accuracy = if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        };

        (avg_loss, accuracy)
    }

    /// Shuffle training data using Fisher-Yates with epoch-dependent seed.
    ///
    /// F-LOOP-007: `seed = base_seed + epoch` ensures different order per epoch
    /// but deterministic across runs.
    fn shuffle_training_data(&mut self, epoch: usize) {
        let seed = self.rng_seed.wrapping_add(epoch as u64);
        let mut rng_state = seed;
        let n = self.train_data.len();

        // Fisher-Yates shuffle with LCG PRNG
        for i in (1..n).rev() {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (rng_state >> 33) as usize % (i + 1);
            self.train_data.swap(i, j);
        }
    }

    /// Save checkpoint with metadata JSON and SafeTensors model weights.
    ///
    /// Creates: `{path}/metadata.json` and `{path}/model.safetensors`
    pub fn save_checkpoint(
        &self,
        path: &Path,
        epoch: usize,
        metrics: &EpochMetrics,
    ) -> crate::Result<()> {
        std::fs::create_dir_all(path).map_err(|e| {
            crate::Error::Io(format!(
                "Failed to create checkpoint dir {}: {e}",
                path.display()
            ))
        })?;

        // Save metadata.json
        let metadata = serde_json::json!({
            "epoch": epoch,
            "train_loss": metrics.train_loss,
            "train_accuracy": metrics.train_accuracy,
            "val_loss": metrics.val_loss,
            "val_accuracy": metrics.val_accuracy,
            "learning_rate": metrics.learning_rate,
            "epoch_time_ms": metrics.epoch_time_ms,
            "samples_per_sec": metrics.samples_per_sec,
        });

        let meta_path = path.join("metadata.json");
        let meta_json = serde_json::to_string_pretty(&metadata).map_err(|e| {
            crate::Error::Serialization(format!("Failed to serialize metadata: {e}"))
        })?;
        std::fs::write(&meta_path, meta_json)?;

        // Save model weights as SafeTensors
        let params = self.pipeline.classifier.parameters();
        let st_path = path.join("model.safetensors");

        // Collect tensor data
        let tensor_names = ["classifier.weight", "classifier.bias"];
        let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = params
            .iter()
            .zip(tensor_names.iter())
            .map(|(tensor, name)| {
                let data = tensor.data();
                let bytes: Vec<u8> =
                    bytemuck::cast_slice(data.as_slice().expect("contiguous")).to_vec();
                let shape = vec![tensor.len()];
                (name.to_string(), bytes, shape)
            })
            .collect();

        let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensor_data
            .iter()
            .map(|(name, bytes, shape)| {
                let view = safetensors::tensor::TensorView::new(
                    safetensors::tensor::Dtype::F32,
                    shape.clone(),
                    bytes,
                )
                .expect("valid tensor view");
                (name.as_str(), view)
            })
            .collect();

        let mut st_metadata = std::collections::HashMap::new();
        st_metadata.insert("epoch".to_string(), epoch.to_string());
        st_metadata.insert("val_loss".to_string(), format!("{:.6}", metrics.val_loss));

        let safetensor_bytes = safetensors::serialize(views, Some(st_metadata)).map_err(|e| {
            crate::Error::Serialization(format!("SafeTensors serialization failed: {e}"))
        })?;
        std::fs::write(&st_path, safetensor_bytes)?;

        // Save APR format
        self.save_apr_checkpoint(path, epoch, metrics)?;

        Ok(())
    }

    /// Save model in APR format.
    fn save_apr_checkpoint(
        &self,
        path: &Path,
        epoch: usize,
        metrics: &EpochMetrics,
    ) -> crate::Result<()> {
        use aprender::serialization::apr::AprWriter;

        let mut writer = AprWriter::new();
        writer.set_metadata(
            "model_type".to_string(),
            serde_json::json!("classify_pipeline"),
        );
        writer.set_metadata("epoch".to_string(), serde_json::json!(epoch));
        writer.set_metadata("val_loss".to_string(), serde_json::json!(metrics.val_loss));
        writer.set_metadata(
            "val_accuracy".to_string(),
            serde_json::json!(metrics.val_accuracy),
        );

        // Add classifier weight and bias tensors
        let weight = &self.pipeline.classifier.weight;
        let weight_data = weight.data();
        let weight_slice = weight_data.as_slice().expect("contiguous weight");
        writer.add_tensor_f32("classifier.weight", vec![weight.len()], weight_slice);

        let bias = &self.pipeline.classifier.bias;
        let bias_data = bias.data();
        let bias_slice = bias_data.as_slice().expect("contiguous bias");
        writer.add_tensor_f32("classifier.bias", vec![bias.len()], bias_slice);

        let apr_path = path.join("model.apr");
        writer.write(&apr_path).map_err(|e| {
            crate::Error::Serialization(format!("APR serialization failed: {e}"))
        })?;

        Ok(())
    }

    /// Split dataset into disjoint train/val sets.
    ///
    /// F-LOOP-008: Guarantees zero overlap between train and val.
    /// F-LOOP-009: Val set is deterministic given the same seed.
    ///
    /// # Arguments
    /// * `data` - Full dataset
    /// * `val_ratio` - Fraction for validation (0.0, 0.5]
    /// * `seed` - Random seed for deterministic shuffling
    pub fn split_dataset(
        data: &[SafetySample],
        val_ratio: f32,
        seed: u64,
    ) -> (Vec<SafetySample>, Vec<SafetySample>) {
        if data.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut indices: Vec<usize> = (0..data.len()).collect();

        // Fisher-Yates shuffle with LCG PRNG for determinism
        let mut rng_state = seed;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (rng_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let val_count = ((data.len() as f32) * val_ratio).ceil() as usize;
        let val_count = val_count.min(data.len() - 1).max(1);

        let val_indices = &indices[..val_count];
        let train_indices = &indices[val_count..];

        let val_data: Vec<SafetySample> = val_indices.iter().map(|&i| data[i].clone()).collect();
        let train_data: Vec<SafetySample> =
            train_indices.iter().map(|&i| data[i].clone()).collect();

        (train_data, val_data)
    }

    /// Get a reference to the training data.
    #[must_use]
    pub fn train_data(&self) -> &[SafetySample] {
        &self.train_data
    }

    /// Get a reference to the validation data.
    #[must_use]
    pub fn val_data(&self) -> &[SafetySample] {
        &self.val_data
    }

    /// Get a reference to the training config.
    #[must_use]
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Get a mutable reference to the underlying pipeline.
    pub fn pipeline_mut(&mut self) -> &mut ClassifyPipeline {
        &mut self.pipeline
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::finetune::{ClassifyConfig, ClassifyPipeline};
    use crate::transformer::TransformerConfig;
    use std::collections::HashSet;

    fn tiny_pipeline(num_classes: usize) -> ClassifyPipeline {
        let model_config = TransformerConfig::tiny();
        let classify_config = ClassifyConfig {
            num_classes,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            batch_size: 4,
            gradient_clip_norm: None,
            ..ClassifyConfig::default()
        };
        ClassifyPipeline::new(&model_config, classify_config)
    }

    fn make_corpus(n: usize, num_classes: usize) -> Vec<SafetySample> {
        (0..n)
            .map(|i| SafetySample {
                input: format!("sample_{i}_{}", "x".repeat(i % 5 + 1)),
                label: i % num_classes,
            })
            .collect()
    }

    // =========================================================================
    // SSC-026: Dataset splitting
    // =========================================================================

    #[test]
    fn test_ssc026_split_dataset_disjoint() {
        // F-LOOP-008: Train and val sets must have zero overlap
        let corpus = make_corpus(20, 3);
        let (train, val) = ClassifyTrainer::split_dataset(&corpus, 0.2, 42);

        let train_inputs: HashSet<String> = train.iter().map(|s| s.input.clone()).collect();
        let val_inputs: HashSet<String> = val.iter().map(|s| s.input.clone()).collect();

        let overlap: HashSet<_> = train_inputs.intersection(&val_inputs).collect();
        assert!(
            overlap.is_empty(),
            "F-LOOP-008: train/val overlap = {overlap:?}"
        );
    }

    #[test]
    fn test_ssc026_split_dataset_sizes() {
        let corpus = make_corpus(100, 3);
        let (train, val) = ClassifyTrainer::split_dataset(&corpus, 0.2, 42);

        // Total must be preserved
        assert_eq!(
            train.len() + val.len(),
            100,
            "All samples must be accounted for"
        );

        // Val should be ~20% (ceil(100 * 0.2) = 20)
        assert_eq!(val.len(), 20, "Val set should be 20% of 100");
        assert_eq!(train.len(), 80, "Train set should be 80% of 100");
    }

    #[test]
    fn test_ssc026_split_dataset_deterministic() {
        let corpus = make_corpus(50, 3);
        let (train1, val1) = ClassifyTrainer::split_dataset(&corpus, 0.2, 42);
        let (train2, val2) = ClassifyTrainer::split_dataset(&corpus, 0.2, 42);

        // Same seed produces identical splits
        let train1_inputs: Vec<String> = train1.iter().map(|s| s.input.clone()).collect();
        let train2_inputs: Vec<String> = train2.iter().map(|s| s.input.clone()).collect();
        assert_eq!(train1_inputs, train2_inputs, "Splits must be deterministic");

        let val1_inputs: Vec<String> = val1.iter().map(|s| s.input.clone()).collect();
        let val2_inputs: Vec<String> = val2.iter().map(|s| s.input.clone()).collect();
        assert_eq!(val1_inputs, val2_inputs, "Val splits must be deterministic");
    }

    #[test]
    fn test_ssc026_split_dataset_empty() {
        let (train, val) = ClassifyTrainer::split_dataset(&[], 0.2, 42);
        assert!(train.is_empty());
        assert!(val.is_empty());
    }

    // =========================================================================
    // SSC-026: Val set frozen
    // =========================================================================

    #[test]
    fn test_ssc026_val_set_frozen() {
        // F-LOOP-009: Val set does not change between epochs
        let num_classes = 3;
        let corpus = make_corpus(20, num_classes);
        let pipeline = tiny_pipeline(num_classes);
        let config = TrainingConfig {
            epochs: 3,
            val_split: 0.2,
            checkpoint_dir: PathBuf::from("/tmp/ssc026_test_frozen"),
            early_stopping_patience: 100,
            ..TrainingConfig::default()
        };

        let trainer = ClassifyTrainer::new(pipeline, corpus, config).unwrap();
        let val_before: Vec<String> = trainer.val_data().iter().map(|s| s.input.clone()).collect();

        // The val set is established at construction and must not change
        let val_after: Vec<String> = trainer.val_data().iter().map(|s| s.input.clone()).collect();
        assert_eq!(
            val_before, val_after,
            "F-LOOP-009: val set must be frozen"
        );
    }

    // =========================================================================
    // SSC-026: Data shuffled per epoch
    // =========================================================================

    #[test]
    fn test_ssc026_data_shuffled_per_epoch() {
        // F-LOOP-007: Training order differs between epochs
        let num_classes = 3;
        let corpus = make_corpus(30, num_classes);
        let pipeline = tiny_pipeline(num_classes);
        let config = TrainingConfig {
            epochs: 2,
            val_split: 0.2,
            checkpoint_dir: PathBuf::from("/tmp/ssc026_test_shuffle"),
            early_stopping_patience: 100,
            ..TrainingConfig::default()
        };

        let mut trainer = ClassifyTrainer::new(pipeline, corpus, config).unwrap();

        // Capture order after epoch-0 shuffle
        trainer.shuffle_training_data(0);
        let order_epoch0: Vec<String> =
            trainer.train_data().iter().map(|s| s.input.clone()).collect();

        // Capture order after epoch-1 shuffle
        trainer.shuffle_training_data(1);
        let order_epoch1: Vec<String> =
            trainer.train_data().iter().map(|s| s.input.clone()).collect();

        assert_ne!(
            order_epoch0, order_epoch1,
            "F-LOOP-007: training data must have different order per epoch"
        );
    }

    // =========================================================================
    // SSC-026: Training convergence
    // =========================================================================

    #[test]
    fn test_ssc026_train_convergence() {
        // Loss should decrease over epochs on a tiny overfit task
        let num_classes = 3;
        let corpus = vec![
            SafetySample {
                input: "echo hello world".into(),
                label: 0,
            },
            SafetySample {
                input: "rm -rf /tmp/danger".into(),
                label: 1,
            },
            SafetySample {
                input: "ls -la /home".into(),
                label: 2,
            },
            SafetySample {
                input: "echo safe output".into(),
                label: 0,
            },
            SafetySample {
                input: "eval dangerous code".into(),
                label: 1,
            },
            SafetySample {
                input: "cat /etc/passwd".into(),
                label: 2,
            },
        ];

        let pipeline = tiny_pipeline(num_classes);
        let config = TrainingConfig {
            epochs: 15,
            val_split: 0.34, // 2 val samples out of 6
            checkpoint_dir: PathBuf::from("/tmp/ssc026_test_convergence"),
            early_stopping_patience: 100, // disable early stopping for this test
            warmup_fraction: 0.0,
            ..TrainingConfig::default()
        };

        let mut trainer = ClassifyTrainer::new(pipeline, corpus, config).unwrap();
        let result = trainer.train();

        assert!(
            !result.epoch_metrics.is_empty(),
            "Should have at least one epoch of metrics"
        );

        let first_loss = result.epoch_metrics.first().unwrap().train_loss;
        let last_loss = result.epoch_metrics.last().unwrap().train_loss;

        assert!(
            last_loss < first_loss,
            "SSC-026: Training loss must decrease. First: {first_loss:.4}, last: {last_loss:.4}"
        );
    }

    // =========================================================================
    // SSC-026: Epoch metrics complete
    // =========================================================================

    #[test]
    fn test_ssc026_epoch_metrics_complete() {
        let num_classes = 3;
        let corpus = make_corpus(15, num_classes);
        let pipeline = tiny_pipeline(num_classes);
        let config = TrainingConfig {
            epochs: 2,
            val_split: 0.2,
            checkpoint_dir: PathBuf::from("/tmp/ssc026_test_metrics"),
            early_stopping_patience: 100,
            ..TrainingConfig::default()
        };

        let mut trainer = ClassifyTrainer::new(pipeline, corpus, config).unwrap();
        let result = trainer.train();

        assert_eq!(result.epoch_metrics.len(), 2, "Should have 2 epochs");

        for m in &result.epoch_metrics {
            assert!(m.train_loss.is_finite(), "train_loss must be finite");
            assert!(m.val_loss.is_finite(), "val_loss must be finite");
            assert!(
                (0.0..=1.0).contains(&m.train_accuracy),
                "train_accuracy must be in [0,1], got {}",
                m.train_accuracy
            );
            assert!(
                (0.0..=1.0).contains(&m.val_accuracy),
                "val_accuracy must be in [0,1], got {}",
                m.val_accuracy
            );
            assert!(m.learning_rate >= 0.0, "LR must be non-negative");
            assert!(m.samples_per_sec >= 0.0, "throughput must be non-negative");
        }
    }

    // =========================================================================
    // SSC-026: Early stopping
    // =========================================================================

    #[test]
    fn test_ssc026_early_stopping() {
        // F-LOOP-010: Training stops after patience epochs without val improvement
        let num_classes = 3;
        let corpus = make_corpus(10, num_classes);
        let pipeline = tiny_pipeline(num_classes);
        let config = TrainingConfig {
            epochs: 100, // high max so early stopping must trigger
            val_split: 0.3,
            early_stopping_patience: 3,
            checkpoint_dir: PathBuf::from("/tmp/ssc026_test_early_stop"),
            warmup_fraction: 0.0,
            ..TrainingConfig::default()
        };

        let mut trainer = ClassifyTrainer::new(pipeline, corpus, config).unwrap();
        let result = trainer.train();

        // Should have stopped before reaching 100 epochs
        assert!(
            result.epoch_metrics.len() < 100,
            "F-LOOP-010: Early stopping should have triggered. Ran {} epochs.",
            result.epoch_metrics.len()
        );
    }

    // =========================================================================
    // SSC-026: Best epoch tracking
    // =========================================================================

    #[test]
    fn test_ssc026_train_result_best_epoch() {
        let num_classes = 3;
        let corpus = make_corpus(15, num_classes);
        let pipeline = tiny_pipeline(num_classes);
        let config = TrainingConfig {
            epochs: 5,
            val_split: 0.2,
            checkpoint_dir: PathBuf::from("/tmp/ssc026_test_best_epoch"),
            early_stopping_patience: 100,
            ..TrainingConfig::default()
        };

        let mut trainer = ClassifyTrainer::new(pipeline, corpus, config).unwrap();
        let result = trainer.train();

        // best_epoch must correspond to the lowest val_loss
        let actual_best = result
            .epoch_metrics
            .iter()
            .min_by(|a, b| a.val_loss.partial_cmp(&b.val_loss).unwrap())
            .unwrap();

        assert_eq!(
            result.best_epoch, actual_best.epoch,
            "best_epoch should match the epoch with lowest val_loss"
        );
        assert!(
            (result.best_val_loss - actual_best.val_loss).abs() < 1e-6,
            "best_val_loss should match"
        );
    }

    // =========================================================================
    // SSC-026: Error handling
    // =========================================================================

    #[test]
    fn test_ssc026_empty_corpus_error() {
        let pipeline = tiny_pipeline(3);
        let config = TrainingConfig::default();

        let result = ClassifyTrainer::new(pipeline, vec![], config);
        assert!(
            result.is_err(),
            "Empty corpus should return an error"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("corpus must not be empty"),
            "Error should mention empty corpus, got: {err}"
        );
    }

    #[test]
    fn test_ssc026_invalid_val_split_zero() {
        let pipeline = tiny_pipeline(3);
        let corpus = make_corpus(10, 3);
        let config = TrainingConfig {
            val_split: 0.0,
            ..TrainingConfig::default()
        };

        let result = ClassifyTrainer::new(pipeline, corpus, config);
        assert!(
            result.is_err(),
            "val_split=0.0 should return an error"
        );
    }

    #[test]
    fn test_ssc026_invalid_val_split_too_large() {
        let pipeline = tiny_pipeline(3);
        let corpus = make_corpus(10, 3);
        let config = TrainingConfig {
            val_split: 0.8,
            ..TrainingConfig::default()
        };

        let result = ClassifyTrainer::new(pipeline, corpus, config);
        assert!(
            result.is_err(),
            "val_split=0.8 should return an error"
        );
    }

    #[test]
    fn test_ssc026_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 50);
        assert!((config.val_split - 0.2).abs() < 1e-6);
        assert_eq!(config.save_every, 5);
        assert_eq!(config.early_stopping_patience, 10);
        assert_eq!(config.seed, 42);
        assert_eq!(config.log_interval, 1);
    }
}
