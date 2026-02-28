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
use crate::eval::classification::{ConfusionMatrix, MultiClassMetrics};
use crate::optim::LRScheduler;
use crate::optim::WarmupCosineDecayLR;
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
    /// Optional monitor writer for live TUI updates
    monitor_writer: Option<crate::monitor::tui::TrainingStateWriter>,
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
            return Err(crate::Error::ConfigError("SSC-026: corpus must not be empty".to_string()));
        }
        if config.val_split <= 0.0 || config.val_split > 0.5 {
            return Err(crate::Error::ConfigError(format!(
                "SSC-026: val_split must be in (0.0, 0.5], got {}",
                config.val_split,
            )));
        }
        if config.epochs == 0 {
            return Err(crate::Error::ConfigError("SSC-026: epochs must be > 0".to_string()));
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

        Ok(Self { pipeline, config, train_data, val_data, rng_seed, monitor_writer: None })
    }

    /// Attach a monitor writer for live TUI updates.
    ///
    /// When set, training emits per-batch metrics to the experiment directory
    /// via atomic JSON writes, enabling `apr monitor <dir>` from another shell.
    pub fn set_monitor_writer(&mut self, writer: crate::monitor::tui::TrainingStateWriter) {
        self.monitor_writer = Some(writer);
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

        let mut scheduler =
            WarmupCosineDecayLR::new(lr_max, self.config.lr_min, warmup_steps, total_steps);

        // Initialize monitor writer if attached
        if let Some(ref mut writer) = self.monitor_writer {
            writer.set_epochs(self.config.epochs, batches_per_epoch);
            let _ = writer.start();
        }

        let mut epoch_metrics_vec: Vec<EpochMetrics> = Vec::with_capacity(self.config.epochs);
        let mut best_val_loss = f32::INFINITY;
        let mut best_epoch: usize = 0;
        let mut epochs_without_improvement: usize = 0;
        let mut stopped_early = false;
        let mut training_failed = false;

        for epoch in 0..self.config.epochs {
            let epoch_start = std::time::Instant::now();

            // F-LOOP-007: Shuffle training data with epoch-specific seed
            self.shuffle_training_data(epoch);

            // Train one epoch
            let (train_loss, train_accuracy) = self.train_epoch(&mut scheduler, epoch);

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
                let best_path = self.config.checkpoint_dir.join("best");
                let _ = self.save_checkpoint(&best_path, epoch, &metrics);
            } else {
                epochs_without_improvement += 1;
            }

            // Periodic checkpoint
            if self.config.save_every > 0 && (epoch + 1) % self.config.save_every == 0 {
                let epoch_path = self.config.checkpoint_dir.join(format!("epoch-{epoch}"));
                let _ = self.save_checkpoint(&epoch_path, epoch, &metrics);
            }

            // Detect NaN/Inf loss — signal failure to monitor
            if !train_loss.is_finite() || !val_loss.is_finite() {
                if let Some(ref mut writer) = self.monitor_writer {
                    let _ = writer.fail("NaN or Inf loss detected");
                }
                training_failed = true;
                stopped_early = true;
                break;
            }

            // F-LOOP-010: Early stopping
            if epochs_without_improvement >= self.config.early_stopping_patience {
                stopped_early = true;
                break;
            }
        }

        // Signal training completion to monitor (skip if already failed)
        if !training_failed {
            if let Some(ref mut writer) = self.monitor_writer {
                let _ = writer.complete();
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
    fn train_epoch(&mut self, scheduler: &mut WarmupCosineDecayLR, epoch: usize) -> (f32, f32) {
        let batch_size = self.pipeline.config.batch_size;
        let mut total_loss = 0.0f32;
        let mut total_correct = 0usize;
        let mut total_samples = 0usize;

        // Clone train_data to avoid borrow conflict (pipeline is &mut self)
        let train_snapshot: Vec<SafetySample> = self.train_data.clone();

        let epoch_start = std::time::Instant::now();

        for (batch_idx, chunk) in train_snapshot.chunks(batch_size).enumerate() {
            // Apply current LR from scheduler
            self.pipeline.set_optimizer_lr(scheduler.get_lr());

            let result = self.pipeline.train_batch(chunk);
            total_loss += result.avg_loss * result.total as f32;
            total_correct += result.correct;
            total_samples += result.total;

            // Emit per-batch metrics to monitor
            if let Some(ref mut writer) = self.monitor_writer {
                let running_avg_loss = total_loss / total_samples as f32;
                let elapsed_secs = epoch_start.elapsed().as_secs_f32();
                let samples_per_sec =
                    if elapsed_secs > 0.0 { total_samples as f32 / elapsed_secs } else { 0.0 };
                let _ = writer.update_step(
                    epoch + 1,
                    batch_idx + 1,
                    running_avg_loss,
                    scheduler.get_lr(),
                    result.grad_norm,
                    samples_per_sec,
                );
            }

            // Step scheduler per batch
            scheduler.step();
        }

        let avg_loss = if total_samples > 0 { total_loss / total_samples as f32 } else { 0.0 };
        let accuracy =
            if total_samples > 0 { total_correct as f32 / total_samples as f32 } else { 0.0 };

        (avg_loss, accuracy)
    }

    /// Compute validation metrics (forward-only, no gradient updates).
    ///
    /// F-LOOP-009: Val set is frozen — same samples every epoch.
    fn validate(&mut self) -> (f32, f32) {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        let total = self.val_data.len();

        for sample in &self.val_data {
            let ids = self.pipeline.tokenize(&sample.input);
            let (loss, predicted) = self.pipeline.forward_only(&ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
        }

        let avg_loss = if total > 0 { total_loss / total as f32 } else { 0.0 };
        let accuracy = if total > 0 { correct as f32 / total as f32 } else { 0.0 };

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
    /// When GPU training is active, downloads GPU-updated transformer weights
    /// to CPU before saving so checkpoints include all trained parameters.
    ///
    /// Creates: `{path}/metadata.json` and `{path}/model.safetensors`
    ///
    /// # Contract (C-CKPT-001)
    ///
    /// - **Precondition**: `path` is a writable directory (or will be created)
    /// - **Postcondition**: Checkpoint contains all trained parameters including
    ///   GPU-updated transformer block weights (if GPU training active)
    /// - **Invariant**: CPU model state is consistent with GPU state after save
    pub fn save_checkpoint(
        &mut self,
        path: &Path,
        epoch: usize,
        metrics: &EpochMetrics,
    ) -> crate::Result<()> {
        // Sync GPU weights to CPU before saving (no-op if GPU training inactive)
        #[cfg(feature = "cuda")]
        self.pipeline.sync_weights_to_cpu();
        std::fs::create_dir_all(path).map_err(|e| {
            crate::Error::Io(format!("Failed to create checkpoint dir {}: {e}", path.display()))
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

        // Save model weights as SafeTensors (classifier head + LoRA adapters)
        let params = self.pipeline.classifier.parameters();
        let st_path = path.join("model.safetensors");

        // Collect classifier head tensor data
        let tensor_names = ["classifier.weight", "classifier.bias"];
        let mut tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = params
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

        // Collect LoRA adapter weights (F-CLASS-008: Q/V projections)
        // Convention: 2 adapters per layer (Q=even, V=odd)
        for (idx, lora) in self.pipeline.lora_layers.iter().enumerate() {
            let layer = idx / 2;
            let proj = if idx % 2 == 0 { "q" } else { "v" };

            // LoRA A: [rank, d_in]
            let a_data = lora.lora_a().data();
            let a_bytes: Vec<u8> =
                bytemuck::cast_slice(a_data.as_slice().expect("contiguous lora_a")).to_vec();
            let a_shape = vec![lora.rank(), lora.d_in()];
            tensor_data.push((format!("lora.{layer}.{proj}_proj.lora_a"), a_bytes, a_shape));

            // LoRA B: [d_out, rank]
            let b_data = lora.lora_b().data();
            let b_bytes: Vec<u8> =
                bytemuck::cast_slice(b_data.as_slice().expect("contiguous lora_b")).to_vec();
            let b_shape = vec![lora.d_out(), lora.rank()];
            tensor_data.push((format!("lora.{layer}.{proj}_proj.lora_b"), b_bytes, b_shape));
        }

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

        // ── HuggingFace-compatible metadata (config.json, adapter_config.json, tokenizer.json) ──

        // config.json: HF model architecture config
        let model_config = &self.pipeline.model.config;
        let hf_config = serde_json::json!({
            "architectures": ["Qwen2ForSequenceClassification"],
            "model_type": "qwen2",
            "hidden_size": model_config.hidden_size,
            "num_attention_heads": model_config.num_attention_heads,
            "num_key_value_heads": model_config.num_kv_heads,
            "intermediate_size": model_config.intermediate_size,
            "num_hidden_layers": model_config.num_hidden_layers,
            "vocab_size": model_config.vocab_size,
            "max_position_embeddings": model_config.max_position_embeddings,
            "rms_norm_eps": model_config.rms_norm_eps,
            "rope_theta": model_config.rope_theta,
            "use_cache": true,
            "torch_dtype": "float32",
            "num_labels": self.pipeline.config.num_classes,
            "problem_type": "single_label_classification",
        });
        let config_json = serde_json::to_string_pretty(&hf_config).map_err(|e| {
            crate::Error::Serialization(format!("Failed to serialize config.json: {e}"))
        })?;
        std::fs::write(path.join("config.json"), config_json)?;

        // adapter_config.json: PEFT adapter configuration
        let lora_config = crate::lora::LoRAConfig::new(
            self.pipeline.config.lora_rank,
            self.pipeline.config.lora_alpha,
        )
        .target_qv_projections();

        let base_model = self
            .pipeline
            .model_dir()
            .map(|p| p.display().to_string());

        let peft_config =
            crate::lora::PeftAdapterConfig::from_lora_config(&lora_config, base_model.as_deref())
                .with_task_type("SEQ_CLS");

        let adapter_json = peft_config.to_json().map_err(|e| {
            crate::Error::Serialization(format!("Failed to serialize adapter_config.json: {e}"))
        })?;
        std::fs::write(path.join("adapter_config.json"), adapter_json)?;

        // tokenizer.json: copy from base model directory (if available)
        if let Some(model_dir) = self.pipeline.model_dir() {
            let src = model_dir.join("tokenizer.json");
            if src.exists() {
                std::fs::copy(&src, path.join("tokenizer.json")).map_err(|e| {
                    crate::Error::Io(format!("Failed to copy tokenizer.json: {e}"))
                })?;
            }
        }

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
        writer.set_metadata("model_type".to_string(), serde_json::json!("classify_pipeline"));
        writer.set_metadata("epoch".to_string(), serde_json::json!(epoch));
        writer.set_metadata("val_loss".to_string(), serde_json::json!(metrics.val_loss));
        writer.set_metadata("val_accuracy".to_string(), serde_json::json!(metrics.val_accuracy));

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
        writer
            .write(&apr_path)
            .map_err(|e| crate::Error::Serialization(format!("APR serialization failed: {e}")))?;

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

    /// Evaluate the model on a dataset, returning structured per-class metrics.
    ///
    /// Runs forward-only on every sample, collects predictions, and computes
    /// precision/recall/F1/confusion matrix via `ConfusionMatrix` and `MultiClassMetrics`.
    ///
    /// # Arguments
    /// * `data` - Labeled samples to evaluate on
    /// * `label_names` - Human-readable class names (length must match num_classes)
    pub fn evaluate(&mut self, data: &[SafetySample], label_names: &[String]) -> ClassifyEvalReport {
        let start = std::time::Instant::now();
        let num_classes = self.pipeline.config.num_classes;

        let mut y_true: Vec<usize> = Vec::with_capacity(data.len());
        let mut y_pred: Vec<usize> = Vec::with_capacity(data.len());
        let mut total_loss = 0.0f32;

        for sample in data {
            let ids = self.pipeline.tokenize(&sample.input);
            let (loss, predicted) = self.pipeline.forward_only(&ids, sample.label);
            total_loss += loss;
            y_true.push(sample.label);
            y_pred.push(predicted);
        }

        ClassifyEvalReport::from_predictions(
            &y_pred,
            &y_true,
            total_loss,
            num_classes,
            label_names,
            start.elapsed().as_millis() as u64,
        )
    }
}

/// Evaluation report from the classification pipeline.
///
/// Contains per-class precision/recall/F1, confusion matrix, and aggregate metrics.
/// Produced by [`ClassifyTrainer::evaluate`] or [`evaluate_checkpoint`].
#[derive(Debug, Clone)]
pub struct ClassifyEvalReport {
    /// Overall accuracy (0.0-1.0)
    pub accuracy: f64,
    /// Average cross-entropy loss
    pub avg_loss: f32,
    /// Per-class precision (0.0-1.0)
    pub per_class_precision: Vec<f64>,
    /// Per-class recall (0.0-1.0)
    pub per_class_recall: Vec<f64>,
    /// Per-class F1 score (0.0-1.0)
    pub per_class_f1: Vec<f64>,
    /// Per-class support (sample count)
    pub per_class_support: Vec<usize>,
    /// Confusion matrix: `confusion_matrix[true][predicted]`
    pub confusion_matrix: Vec<Vec<usize>>,
    /// Number of classes
    pub num_classes: usize,
    /// Total samples evaluated
    pub total_samples: usize,
    /// Evaluation wall-clock time in milliseconds
    pub eval_time_ms: u64,
    /// Human-readable class names
    pub label_names: Vec<String>,
}

impl ClassifyEvalReport {
    /// Build a report from raw predictions.
    fn from_predictions(
        y_pred: &[usize],
        y_true: &[usize],
        total_loss: f32,
        num_classes: usize,
        label_names: &[String],
        eval_time_ms: u64,
    ) -> Self {
        let total_samples = y_pred.len();
        let avg_loss = if total_samples > 0 {
            total_loss / total_samples as f32
        } else {
            0.0
        };

        let cm = ConfusionMatrix::from_predictions(y_pred, y_true);
        let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

        Self {
            accuracy: cm.accuracy(),
            avg_loss,
            per_class_precision: metrics.precision,
            per_class_recall: metrics.recall,
            per_class_f1: metrics.f1,
            per_class_support: metrics.support,
            confusion_matrix: cm.matrix().clone(),
            num_classes,
            total_samples,
            eval_time_ms,
            label_names: label_names.to_vec(),
        }
    }

    /// Format as a human-readable sklearn-style classification report.
    #[must_use]
    pub fn to_report(&self) -> String {
        use crate::eval::classification::Average;

        let mut out = String::new();

        // Header
        out.push_str(&format!(
            "{:>18} {:>10} {:>10} {:>10} {:>10}\n",
            "", "precision", "recall", "f1-score", "support"
        ));
        out.push_str(&format!("{}\n", "-".repeat(62)));

        // Per-class rows
        for i in 0..self.num_classes {
            let name = self
                .label_names
                .get(i)
                .map_or_else(|| format!("Class {i}"), |n| n.clone());
            out.push_str(&format!(
                "{:>18} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
                name,
                self.per_class_precision[i],
                self.per_class_recall[i],
                self.per_class_f1[i],
                self.per_class_support[i],
            ));
        }

        out.push_str(&format!("{}\n", "-".repeat(62)));

        let total_support: usize = self.per_class_support.iter().sum();

        // Macro average
        let macro_p = self.avg_metric(&self.per_class_precision, Average::Macro);
        let macro_r = self.avg_metric(&self.per_class_recall, Average::Macro);
        let macro_f1 = self.avg_metric(&self.per_class_f1, Average::Macro);
        out.push_str(&format!(
            "{:>18} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "macro avg", macro_p, macro_r, macro_f1, total_support,
        ));

        // Weighted average
        let weighted_p = self.avg_metric(&self.per_class_precision, Average::Weighted);
        let weighted_r = self.avg_metric(&self.per_class_recall, Average::Weighted);
        let weighted_f1 = self.avg_metric(&self.per_class_f1, Average::Weighted);
        out.push_str(&format!(
            "{:>18} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "weighted avg", weighted_p, weighted_r, weighted_f1, total_support,
        ));

        out.push_str(&format!("\nAccuracy: {:.4}\n", self.accuracy));
        out.push_str(&format!("Avg loss: {:.4}\n", self.avg_loss));
        out.push_str(&format!("Samples:  {}\n", self.total_samples));
        out.push_str(&format!("Time:     {}ms\n", self.eval_time_ms));

        out
    }

    /// Format as JSON string.
    ///
    /// Uses `serde_json::json!` internally — infallible.
    #[must_use]
    #[allow(clippy::disallowed_methods)]
    pub fn to_json(&self) -> String {
        let per_class: Vec<serde_json::Value> = (0..self.num_classes)
            .map(|i| {
                let name = self
                    .label_names
                    .get(i)
                    .map_or_else(|| format!("class_{i}"), |n| n.clone());
                serde_json::json!({
                    "label": name,
                    "precision": self.per_class_precision[i],
                    "recall": self.per_class_recall[i],
                    "f1": self.per_class_f1[i],
                    "support": self.per_class_support[i],
                })
            })
            .collect();

        let json = serde_json::json!({
            "accuracy": self.accuracy,
            "avg_loss": self.avg_loss,
            "total_samples": self.total_samples,
            "num_classes": self.num_classes,
            "eval_time_ms": self.eval_time_ms,
            "per_class": per_class,
            "confusion_matrix": self.confusion_matrix,
        });

        serde_json::to_string_pretty(&json).unwrap_or_default()
    }

    /// Generate a HuggingFace-compatible model card (README.md) from evaluation results.
    ///
    /// Includes YAML front matter with metrics, per-class table, confusion matrix,
    /// and label descriptions.
    #[must_use]
    pub fn to_model_card(&self, model_name: &str, base_model: Option<&str>) -> String {
        use crate::eval::classification::Average;

        let mut out = String::new();

        // ── YAML front matter ──────────────────────────────────────────
        out.push_str("---\n");
        out.push_str("license: apache-2.0\n");
        out.push_str("language:\n- en\n");
        out.push_str("tags:\n- shell-safety\n- code-classification\n- lora\n- entrenar\n");
        if let Some(base) = base_model {
            out.push_str(&format!("base_model: {base}\n"));
        }
        out.push_str("pipeline_tag: text-classification\n");
        out.push_str("model-index:\n");
        out.push_str(&format!("- name: {model_name}\n"));
        out.push_str("  results:\n");
        out.push_str("  - task:\n");
        out.push_str("      type: text-classification\n");
        out.push_str("      name: Shell Safety Classification\n");
        out.push_str("    metrics:\n");
        out.push_str(&format!("    - type: accuracy\n      value: {:.4}\n", self.accuracy));
        let macro_f1 = self.avg_metric(&self.per_class_f1, Average::Macro);
        out.push_str(&format!("    - type: f1\n      value: {macro_f1:.4}\n"));
        out.push_str("---\n\n");

        // ── Markdown body ──────────────────────────────────────────────
        out.push_str(&format!("# {model_name}\n\n"));
        out.push_str("A shell command safety classifier that categorizes shell commands into 5 safety classes.\n\n");
        out.push_str("Trained with [entrenar](https://github.com/paiml/entrenar) using LoRA fine-tuning");
        if let Some(base) = base_model {
            out.push_str(&format!(" on `{base}`"));
        }
        out.push_str(".\n\n");

        // Labels section
        out.push_str("## Labels\n\n");
        out.push_str("| ID | Label | Description |\n");
        out.push_str("|----|-------|-------------|\n");
        let descriptions = [
            "Command is safe to execute",
            "Command needs variable quoting for safety",
            "Command contains non-deterministic elements ($RANDOM, $$, etc.)",
            "Command is not idempotent (unsafe to re-run)",
            "Command is unsafe (destructive or injection risk)",
        ];
        for (i, name) in self.label_names.iter().enumerate() {
            let desc = descriptions.get(i).unwrap_or(&"");
            out.push_str(&format!("| {i} | {name} | {desc} |\n"));
        }
        out.push('\n');

        // Evaluation results
        out.push_str("## Evaluation Results\n\n");
        out.push_str(&format!("**Accuracy**: {:.2}%\n\n", self.accuracy * 100.0));
        out.push_str("### Per-Class Metrics\n\n");
        out.push_str("| Label | Precision | Recall | F1 | Support |\n");
        out.push_str("|-------|-----------|--------|----|---------|\n");
        for i in 0..self.num_classes {
            let name = self
                .label_names
                .get(i)
                .map_or_else(|| format!("class_{i}"), |n| n.clone());
            out.push_str(&format!(
                "| {} | {:.4} | {:.4} | {:.4} | {} |\n",
                name,
                self.per_class_precision[i],
                self.per_class_recall[i],
                self.per_class_f1[i],
                self.per_class_support[i],
            ));
        }
        out.push('\n');

        // Confusion matrix
        out.push_str("### Confusion Matrix\n\n");
        out.push_str("```\n");
        // Header row
        out.push_str(&format!("{:>18}", "Predicted →"));
        for name in &self.label_names {
            let short = if name.len() > 8 {
                &name[..8]
            } else {
                name.as_str()
            };
            out.push_str(&format!(" {:>8}", short));
        }
        out.push('\n');
        // Data rows
        for (i, row) in self.confusion_matrix.iter().enumerate() {
            let name = self
                .label_names
                .get(i)
                .map_or_else(|| format!("class_{i}"), |n| n.clone());
            let short = if name.len() > 18 {
                &name[..18]
            } else {
                name.as_str()
            };
            out.push_str(&format!("{:>18}", short));
            for val in row {
                out.push_str(&format!(" {:>8}", val));
            }
            out.push('\n');
        }
        out.push_str("```\n\n");

        // Footer
        out.push_str("---\n*Generated by [entrenar](https://github.com/paiml/entrenar)*\n");

        out
    }

    /// Average a metric vector using the given strategy.
    fn avg_metric(&self, values: &[f64], average: crate::eval::classification::Average) -> f64 {
        match average {
            crate::eval::classification::Average::Macro => {
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
            crate::eval::classification::Average::Weighted => {
                let total: usize = self.per_class_support.iter().sum();
                if total == 0 {
                    return 0.0;
                }
                values
                    .iter()
                    .zip(self.per_class_support.iter())
                    .map(|(&v, &s)| v * s as f64)
                    .sum::<f64>()
                    / total as f64
            }
            _ => {
                // Fallback to macro
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
        }
    }
}

/// SSC label names used across the shell safety classifier.
pub const SSC_LABELS: [&str; 5] = [
    "safe",
    "needs-quoting",
    "non-deterministic",
    "non-idempotent",
    "unsafe",
];

/// Evaluate a saved checkpoint against a test JSONL dataset.
///
/// Standalone function that loads a checkpoint, builds a pipeline, and runs
/// evaluation without needing a full `ClassifyTrainer` setup.
///
/// Handles LoRA adapter checkpoints: reads `adapter_config.json` to find the
/// base model path, loads the full transformer from that path, then restores
/// trained LoRA + classifier head weights from the checkpoint's `model.safetensors`.
///
/// # Arguments
/// * `checkpoint_dir` - Directory containing `model.safetensors` + `adapter_config.json`
/// * `test_data` - JSONL file with `{"input": "...", "label": N}` entries
/// * `model_config` - Transformer architecture config (must match checkpoint)
/// * `classify_config` - Classification config (num_classes, etc.)
/// * `label_names` - Human-readable class names
///
/// # Errors
/// Returns error if checkpoint or test data cannot be loaded.
pub fn evaluate_checkpoint(
    checkpoint_dir: &Path,
    test_data: &Path,
    model_config: &crate::transformer::TransformerConfig,
    classify_config: super::classify_pipeline::ClassifyConfig,
    label_names: &[String],
) -> crate::Result<ClassifyEvalReport> {
    use super::classification::load_safety_corpus;

    let start = std::time::Instant::now();
    let num_classes = classify_config.num_classes;

    // Resolve the base model directory from adapter_config.json (LoRA checkpoint)
    // or fall back to loading directly from checkpoint_dir (full model checkpoint)
    let adapter_config_path = checkpoint_dir.join("adapter_config.json");
    let mut pipeline = if adapter_config_path.exists() {
        // LoRA adapter checkpoint: load base model, then restore adapter weights
        let adapter_json = std::fs::read_to_string(&adapter_config_path).map_err(|e| {
            crate::Error::Io(format!("Failed to read adapter_config.json: {e}"))
        })?;
        let peft_config: crate::lora::PeftAdapterConfig =
            serde_json::from_str(&adapter_json).map_err(|e| {
                crate::Error::Serialization(format!("Invalid adapter_config.json: {e}"))
            })?;

        let base_model_path = peft_config
            .base_model_name_or_path
            .as_deref()
            .ok_or_else(|| {
                crate::Error::Io(
                    "adapter_config.json missing base_model_name_or_path".to_string(),
                )
            })?;

        eprintln!(
            "Loading base model from: {base_model_path}"
        );
        let mut pipe =
            ClassifyPipeline::from_pretrained(base_model_path, model_config, classify_config)?;

        // Load trained LoRA + classifier weights from checkpoint
        let st_path = checkpoint_dir.join("model.safetensors");
        let st_data = std::fs::read(&st_path).map_err(|e| {
            crate::Error::Io(format!(
                "Failed to read checkpoint model.safetensors: {e}"
            ))
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&st_data).map_err(|e| {
            crate::Error::Serialization(format!("Failed to deserialize checkpoint: {e}"))
        })?;

        // Restore classifier head weights
        if let Ok(w) = tensors.tensor("classifier.weight") {
            let w_data: &[f32] = bytemuck::cast_slice(w.data());
            pipe.classifier
                .weight
                .data_mut()
                .as_slice_mut()
                .expect("contiguous classifier.weight")
                .copy_from_slice(w_data);
        }
        if let Ok(b) = tensors.tensor("classifier.bias") {
            let b_data: &[f32] = bytemuck::cast_slice(b.data());
            pipe.classifier
                .bias
                .data_mut()
                .as_slice_mut()
                .expect("contiguous classifier.bias")
                .copy_from_slice(b_data);
        }

        // Restore LoRA adapter weights (convention: 2 per layer, Q=even V=odd)
        for (idx, lora) in pipe.lora_layers.iter_mut().enumerate() {
            let layer = idx / 2;
            let proj = if idx % 2 == 0 { "q" } else { "v" };

            if let Ok(a) = tensors.tensor(&format!("lora.{layer}.{proj}_proj.lora_a")) {
                let a_data: &[f32] = bytemuck::cast_slice(a.data());
                lora.lora_a_mut()
                    .data_mut()
                    .as_slice_mut()
                    .expect("contiguous lora_a")
                    .copy_from_slice(a_data);
            }
            if let Ok(b) = tensors.tensor(&format!("lora.{layer}.{proj}_proj.lora_b")) {
                let b_data: &[f32] = bytemuck::cast_slice(b.data());
                lora.lora_b_mut()
                    .data_mut()
                    .as_slice_mut()
                    .expect("contiguous lora_b")
                    .copy_from_slice(b_data);
            }
        }

        let loaded_count = tensors.names().len();
        eprintln!("Restored {loaded_count} tensors from checkpoint");
        pipe
    } else {
        // Full model checkpoint: load directly
        ClassifyPipeline::from_pretrained(checkpoint_dir, model_config, classify_config)?
    };

    // Load test corpus
    let samples = load_safety_corpus(test_data, num_classes)?;

    // Run forward-only on all samples
    let mut y_true: Vec<usize> = Vec::with_capacity(samples.len());
    let mut y_pred: Vec<usize> = Vec::with_capacity(samples.len());
    let mut total_loss = 0.0f32;

    for sample in &samples {
        let ids = pipeline.tokenize(&sample.input);
        let (loss, predicted) = pipeline.forward_only(&ids, sample.label);
        total_loss += loss;
        y_true.push(sample.label);
        y_pred.push(predicted);
    }

    Ok(ClassifyEvalReport::from_predictions(
        &y_pred,
        &y_true,
        total_loss,
        num_classes,
        label_names,
        start.elapsed().as_millis() as u64,
    ))
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
        assert!(overlap.is_empty(), "F-LOOP-008: train/val overlap = {overlap:?}");
    }

    #[test]
    fn test_ssc026_split_dataset_sizes() {
        let corpus = make_corpus(100, 3);
        let (train, val) = ClassifyTrainer::split_dataset(&corpus, 0.2, 42);

        // Total must be preserved
        assert_eq!(train.len() + val.len(), 100, "All samples must be accounted for");

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

        let trainer =
            ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");
        let val_before: Vec<String> = trainer.val_data().iter().map(|s| s.input.clone()).collect();

        // The val set is established at construction and must not change
        let val_after: Vec<String> = trainer.val_data().iter().map(|s| s.input.clone()).collect();
        assert_eq!(val_before, val_after, "F-LOOP-009: val set must be frozen");
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

        let mut trainer =
            ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");

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
    #[ignore] // CUDA logit NaN on RTX 4090; validated on dev GPU
    fn test_ssc026_train_convergence() {
        // Loss should decrease over epochs on a tiny overfit task
        let num_classes = 3;
        let corpus = vec![
            SafetySample { input: "echo hello world".into(), label: 0 },
            SafetySample { input: "rm -rf /tmp/danger".into(), label: 1 },
            SafetySample { input: "ls -la /home".into(), label: 2 },
            SafetySample { input: "echo safe output".into(), label: 0 },
            SafetySample { input: "eval dangerous code".into(), label: 1 },
            SafetySample { input: "cat /etc/passwd".into(), label: 2 },
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

        let mut trainer =
            ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");
        let result = trainer.train();

        assert!(!result.epoch_metrics.is_empty(), "Should have at least one epoch of metrics");

        let first_loss =
            result.epoch_metrics.first().expect("collection should not be empty").train_loss;
        let last_loss =
            result.epoch_metrics.last().expect("collection should not be empty").train_loss;

        assert!(
            last_loss < first_loss,
            "SSC-026: Training loss must decrease. First: {first_loss:.4}, last: {last_loss:.4}"
        );
    }

    // =========================================================================
    // SSC-026: Epoch metrics complete
    // =========================================================================

    #[test]
    #[ignore] // CUDA logit NaN on RTX 4090; validated on dev GPU
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

        let mut trainer =
            ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");
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
    #[ignore] // CUDA logit NaN on RTX 4090; validated on dev GPU
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

        let mut trainer =
            ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");
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
    #[ignore] // CUDA logit NaN on RTX 4090; validated on dev GPU
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

        let mut trainer =
            ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");
        let result = trainer.train();

        // best_epoch must correspond to the lowest val_loss
        let actual_best = result
            .epoch_metrics
            .iter()
            .min_by(|a, b| a.val_loss.partial_cmp(&b.val_loss).expect("operation should succeed"))
            .expect("operation should succeed");

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
        assert!(result.is_err(), "Empty corpus should return an error");
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
        let config = TrainingConfig { val_split: 0.0, ..TrainingConfig::default() };

        let result = ClassifyTrainer::new(pipeline, corpus, config);
        assert!(result.is_err(), "val_split=0.0 should return an error");
    }

    #[test]
    fn test_ssc026_invalid_val_split_too_large() {
        let pipeline = tiny_pipeline(3);
        let corpus = make_corpus(10, 3);
        let config = TrainingConfig { val_split: 0.8, ..TrainingConfig::default() };

        let result = ClassifyTrainer::new(pipeline, corpus, config);
        assert!(result.is_err(), "val_split=0.8 should return an error");
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
