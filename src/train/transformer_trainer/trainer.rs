//! Transformer trainer implementation

use crate::autograd::{checkpoint, GradScaler};
use crate::io::{save_model, Model, ModelFormat, ModelMetadata, SaveConfig};
use crate::optim::{AdamW, Optimizer};
use crate::train::{CausalLMLoss, LossFn, MetricsTracker};
use crate::transformer::Transformer;
use crate::Tensor;
use std::path::Path;

use super::batch::LMBatch;
use super::config::TransformerTrainConfig;

/// Transformer training state
pub struct TransformerTrainer {
    /// Model
    model: Transformer,
    /// Loss function
    loss_fn: CausalLMLoss,
    /// Optimizer
    optimizer: AdamW,
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
        let optimizer = AdamW::default_params(config.lr);
        let grad_scaler = GradScaler::from_config(&config.precision_config);

        Self {
            model,
            loss_fn,
            optimizer,
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
        let optimizer = AdamW::default_params(config.lr);
        let grad_scaler = GradScaler::from_config(&config.precision_config);

        Self {
            model,
            loss_fn,
            optimizer,
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
    /// Returns (loss_value, loss_tensor, logits)
    pub fn forward_single(&self, input_ids: &[u32], target_ids: &[u32]) -> (f32, Tensor, Tensor) {
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
        let loss_val = loss.data()[0];

        (loss_val, loss, logits)
    }

    /// Compute forward + backward for all items in a batch, returning average loss.
    fn compute_batch_gradients(&self, batch: &LMBatch) -> f32 {
        let mut total_loss = 0.0;

        for i in 0..batch.batch_size {
            let Some(input_ids) = batch.get_input(i) else {
                continue;
            };
            let Some(target_ids) = batch.get_target(i) else {
                continue;
            };

            let (loss_val, loss, _logits) = self.forward_single(input_ids, target_ids);

            if let Some(backward_op) = loss.backward_op() {
                backward_op.backward();
            }

            total_loss += loss_val / self.config.accumulation_steps as f32;
        }

        total_loss / batch.batch_size as f32
    }

    /// Apply gradient clipping and run the optimizer step, then reset accumulation.
    fn clip_and_step(&mut self) {
        if let Some(max_norm) = self.config.base.max_grad_norm {
            let params = self.model.parameters();
            let total_norm: f32 = params
                .iter()
                .filter_map(|p| p.grad())
                .map(|g| g.iter().map(|x| x * x).sum::<f32>())
                .sum::<f32>()
                .sqrt();

            if total_norm > max_norm {
                let scale = max_norm / (total_norm + 1e-6);
                // FUTURE: in-place gradient scaling for strict clipping
                let _ = scale;
            }
        }

        let mut params = self.model.parameters_mut();
        self.optimizer.step_refs(&mut params);

        self.step += 1;
        self.metrics.losses.push(self.accumulated_loss);
        self.metrics.increment_step();

        self.accumulated_loss = 0.0;
        self.accumulated_batches = 0;
    }

    /// Process a batch (forward + backward + optimizer step)
    ///
    /// Returns average loss for the batch
    pub fn train_batch(&mut self, batch: &LMBatch) -> f32 {
        if batch.batch_size == 0 {
            return 0.0;
        }

        if self.accumulated_batches == 0 {
            let mut params = self.model.parameters_mut();
            self.optimizer.zero_grad_refs(&mut params);
        }

        let avg_loss = self.compute_batch_gradients(batch);

        self.accumulated_loss += avg_loss;
        self.accumulated_batches += 1;

        if self.accumulated_batches >= self.config.accumulation_steps {
            self.clip_and_step();
        }

        avg_loss
    }

    /// Train for one epoch over batches
    pub fn train_epoch(&mut self, batches: &[LMBatch]) -> f32 {
        self.train_epoch_with_callback(batches, |_, _, _| {})
    }

    /// Train for one epoch with a per-step callback.
    ///
    /// The callback receives (batch_index, batch_loss, &self) after each batch.
    /// Use this for progress logging, checkpointing, or early stopping.
    ///
    /// Stops early if `max_steps` is set and the step count reaches it.
    /// Returns `(avg_loss, reached_max_steps)`.
    pub fn train_epoch_with_callback<F>(&mut self, batches: &[LMBatch], mut on_batch: F) -> f32
    where
        F: FnMut(usize, f32, &Self),
    {
        if batches.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut batches_processed = 0;

        for (i, batch) in batches.iter().enumerate() {
            // Check max_steps before processing
            if let Some(max) = self.config.max_steps {
                if self.step >= max {
                    break;
                }
            }

            let batch_loss = self.train_batch(batch);
            total_loss += batch_loss;
            batches_processed += 1;
            on_batch(i, batch_loss, self);
        }

        total_loss / batches_processed.max(1) as f32
    }

    /// Returns true if max_steps has been reached.
    pub fn reached_max_steps(&self) -> bool {
        self.config
            .max_steps
            .map_or(false, |max| self.step >= max)
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

    /// Save model weights to a SafeTensors file
    ///
    /// This persists the trained transformer weights to disk.
    /// Call this after training completes to preserve the learned parameters.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path (should end in .safetensors)
    /// * `name` - Model name for metadata
    /// * `architecture` - Model architecture description (e.g., "Qwen2ForCausalLM")
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(
        &self,
        path: impl AsRef<Path>,
        name: &str,
        architecture: &str,
    ) -> crate::Result<()> {
        // Collect all model parameters with proper names
        let model_params = self.model.parameters();
        let num_total = model_params.len();

        // Build parameter names matching transformer architecture conventions
        let num_layers = (num_total - 2) / 9;
        let mut names: Vec<String> = Vec::with_capacity(num_total);

        names.push("model.embed_tokens.weight".to_string());
        names.push("model.norm.weight".to_string());

        // Each layer has: input_norm, post_attn_norm, 4 attention weights, 3 FFN weights = 9 params
        for layer in 0..num_layers {
            names.push(format!("model.layers.{layer}.input_layernorm.weight"));
            names.push(format!("model.layers.{layer}.post_attention_layernorm.weight"));
            names.push(format!("model.layers.{layer}.self_attn.q_proj.weight"));
            names.push(format!("model.layers.{layer}.self_attn.k_proj.weight"));
            names.push(format!("model.layers.{layer}.self_attn.v_proj.weight"));
            names.push(format!("model.layers.{layer}.self_attn.o_proj.weight"));
            names.push(format!("model.layers.{layer}.mlp.gate_proj.weight"));
            names.push(format!("model.layers.{layer}.mlp.up_proj.weight"));
            names.push(format!("model.layers.{layer}.mlp.down_proj.weight"));
        }

        // LM head if present
        if names.len() < num_total {
            names.push("lm_head.weight".to_string());
        }

        // Pair names with cloned parameters (clone outside loop via iterator)
        let params: Vec<(String, Tensor)> = names
            .into_iter()
            .zip(model_params)
            .map(|(name, tensor)| (name, tensor.clone()))
            .collect();

        let metadata = ModelMetadata::new(name, architecture);
        let model = Model::new(metadata, params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        save_model(&model, path, &config)
    }
}
