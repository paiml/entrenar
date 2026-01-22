//! Transformer trainer implementation

use crate::autograd::{checkpoint, GradScaler};
use crate::train::{CausalLMLoss, LossFn, MetricsTracker};
use crate::transformer::Transformer;
use crate::Tensor;

use super::batch::LMBatch;
use super::config::TransformerTrainConfig;

/// Transformer training state
pub struct TransformerTrainer {
    /// Model
    model: Transformer,
    /// Loss function
    loss_fn: CausalLMLoss,
    /// Gradient scaler for mixed precision
    grad_scaler: GradScaler,
    /// Configuration
    config: TransformerTrainConfig,
    /// Metrics tracker
    pub metrics: MetricsTracker,
    /// Current step
    step: usize,
    /// Accumulated gradients (for gradient accumulation)
    accumulated_loss: f32,
    /// Number of accumulated batches
    accumulated_batches: usize,
}

impl TransformerTrainer {
    /// Create a new transformer trainer
    pub fn new(config: TransformerTrainConfig) -> Self {
        let model = Transformer::new(&config.model_config);
        let loss_fn = CausalLMLoss::new(config.model_config.vocab_size);
        let grad_scaler = GradScaler::from_config(&config.precision_config);

        Self {
            model,
            loss_fn,
            grad_scaler,
            config,
            metrics: MetricsTracker::new(),
            step: 0,
            accumulated_loss: 0.0,
            accumulated_batches: 0,
        }
    }

    /// Create trainer from existing model
    pub fn with_model(model: Transformer, config: TransformerTrainConfig) -> Self {
        let loss_fn = CausalLMLoss::new(config.model_config.vocab_size);
        let grad_scaler = GradScaler::from_config(&config.precision_config);

        Self {
            model,
            loss_fn,
            grad_scaler,
            config,
            metrics: MetricsTracker::new(),
            step: 0,
            accumulated_loss: 0.0,
            accumulated_batches: 0,
        }
    }

    /// Forward pass on a single batch item
    ///
    /// Returns (loss_value, logits)
    pub fn forward_single(&self, input_ids: &[u32], target_ids: &[u32]) -> (f32, Tensor) {
        // Forward through transformer
        let logits = if self.config.checkpoint_config.enabled {
            // With checkpointing (recompute activations during backward)
            checkpoint(|_| self.model.forward(input_ids), &Tensor::zeros(1, false))
        } else {
            self.model.forward(input_ids)
        };

        // Compute loss
        let targets = Tensor::from_vec(target_ids.iter().map(|&id| id as f32).collect(), false);
        let loss = self.loss_fn.forward(&logits, &targets);

        (loss.data()[0], logits)
    }

    /// Process a batch (forward + backward for each item)
    ///
    /// Returns average loss for the batch
    pub fn train_batch(&mut self, batch: &LMBatch) -> f32 {
        if batch.batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for i in 0..batch.batch_size {
            let input_ids = batch.get_input(i).unwrap();
            let target_ids = batch.get_target(i).unwrap();

            let (loss, _logits) = self.forward_single(input_ids, target_ids);

            // Scale loss for gradient accumulation
            let scaled_loss = loss / self.config.accumulation_steps as f32;
            total_loss += scaled_loss;
        }

        let avg_loss = total_loss / batch.batch_size as f32;

        // Track accumulation
        self.accumulated_loss += avg_loss;
        self.accumulated_batches += 1;

        // Perform optimizer step if accumulation is complete
        if self.accumulated_batches >= self.config.accumulation_steps {
            self.step += 1;
            self.metrics.losses.push(self.accumulated_loss);
            self.metrics.increment_step();

            // Reset accumulation
            self.accumulated_loss = 0.0;
            self.accumulated_batches = 0;
        }

        avg_loss
    }

    /// Train for one epoch over batches
    pub fn train_epoch(&mut self, batches: &[LMBatch]) -> f32 {
        if batches.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for batch in batches {
            let batch_loss = self.train_batch(batch);
            total_loss += batch_loss;
        }

        total_loss / batches.len() as f32
    }

    /// Get current step count
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get reference to model
    pub fn model(&self) -> &Transformer {
        &self.model
    }

    /// Get mutable reference to model
    pub fn model_mut(&mut self) -> &mut Transformer {
        &mut self.model
    }

    /// Get current learning rate (with warmup applied)
    pub fn current_lr(&self) -> f32 {
        let base_lr = self.config.lr;

        if self.step < self.config.warmup_steps {
            // Linear warmup
            base_lr * (self.step as f32 / self.config.warmup_steps as f32)
        } else {
            base_lr
        }
    }

    /// Get gradient scaler stats
    pub fn grad_scaler_stats(&self) -> (f32, usize, usize) {
        (
            self.grad_scaler.scale(),
            self.grad_scaler.overflow_count(),
            self.grad_scaler.successful_steps(),
        )
    }

    /// Check if using mixed precision
    pub fn is_mixed_precision(&self) -> bool {
        self.config.precision_config.is_mixed()
    }

    /// Check if using gradient checkpointing
    pub fn is_checkpointing(&self) -> bool {
        self.config.checkpoint_config.enabled
    }
}
