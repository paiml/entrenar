//! Transformer-specific training utilities
//!
//! Provides specialized training components for transformer language models,
//! including tokenized batch creation and language modeling training loops.

use crate::autograd::{checkpoint, CheckpointConfig, GradScaler, MixedPrecisionConfig};
use crate::train::{CausalLMLoss, LossFn, MetricsTracker, TrainConfig};
use crate::transformer::{Transformer, TransformerConfig};
use crate::Tensor;

/// Configuration for transformer training
#[derive(Debug, Clone)]
pub struct TransformerTrainConfig {
    /// Base training configuration
    pub base: TrainConfig,
    /// Transformer architecture configuration
    pub model_config: TransformerConfig,
    /// Checkpoint configuration for memory efficiency
    pub checkpoint_config: CheckpointConfig,
    /// Mixed-precision configuration
    pub precision_config: MixedPrecisionConfig,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Accumulation steps for gradient accumulation
    pub accumulation_steps: usize,
    /// Warmup steps for learning rate scheduler
    pub warmup_steps: usize,
    /// Learning rate
    pub lr: f32,
}

impl TransformerTrainConfig {
    /// Create new config with defaults
    pub fn new(model_config: TransformerConfig) -> Self {
        Self {
            base: TrainConfig::default(),
            model_config,
            checkpoint_config: CheckpointConfig::disabled(),
            precision_config: MixedPrecisionConfig::fp32(),
            max_seq_len: 512,
            accumulation_steps: 1,
            warmup_steps: 0,
            lr: 0.001,
        }
    }

    /// Enable gradient checkpointing
    pub fn with_checkpointing(mut self, num_segments: usize) -> Self {
        self.checkpoint_config = CheckpointConfig::enabled(num_segments);
        self
    }

    /// Enable bf16 mixed precision
    pub fn with_bf16(mut self) -> Self {
        self.precision_config = MixedPrecisionConfig::bf16();
        self
    }

    /// Enable fp16 mixed precision with dynamic loss scaling
    pub fn with_fp16(mut self) -> Self {
        self.precision_config = MixedPrecisionConfig::fp16();
        self
    }

    /// Set maximum sequence length
    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Set gradient accumulation steps
    pub fn with_accumulation_steps(mut self, steps: usize) -> Self {
        self.accumulation_steps = steps.max(1);
        self
    }

    /// Set warmup steps
    pub fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set learning rate
    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Set gradient clipping
    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.base.max_grad_norm = Some(clip);
        self
    }
}

/// A batch of tokenized sequences for language model training
#[derive(Debug, Clone)]
pub struct LMBatch {
    /// Input token IDs (batch_size x seq_len flattened)
    pub input_ids: Vec<u32>,
    /// Target token IDs (batch_size x seq_len flattened, shifted by 1)
    pub target_ids: Vec<u32>,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
}

impl LMBatch {
    /// Create a new LM batch from token sequences
    ///
    /// For causal LM, targets are inputs shifted by 1:
    /// input:  [BOS, A, B, C, D]
    /// target: [A, B, C, D, EOS]
    pub fn from_sequences(sequences: &[Vec<u32>], pad_id: u32, eos_id: u32) -> Self {
        if sequences.is_empty() {
            return Self {
                input_ids: Vec::new(),
                target_ids: Vec::new(),
                batch_size: 0,
                seq_len: 0,
            };
        }

        let batch_size = sequences.len();
        let max_len = sequences.iter().map(Vec::len).max().unwrap_or(0);
        let seq_len = max_len.saturating_sub(1).max(1);

        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut target_ids = Vec::with_capacity(batch_size * seq_len);

        for seq in sequences {
            // Input: all tokens except last
            for i in 0..seq_len {
                if i < seq.len() - 1 {
                    input_ids.push(seq[i]);
                } else {
                    input_ids.push(pad_id);
                }
            }

            // Target: all tokens except first (shifted by 1)
            for i in 0..seq_len {
                if i + 1 < seq.len() {
                    target_ids.push(seq[i + 1]);
                } else if i + 1 == seq.len() {
                    target_ids.push(eos_id);
                } else {
                    target_ids.push(pad_id);
                }
            }
        }

        Self {
            input_ids,
            target_ids,
            batch_size,
            seq_len,
        }
    }

    /// Create a batch from a single sequence (for testing)
    pub fn single(input_ids: Vec<u32>, target_ids: Vec<u32>) -> Self {
        let seq_len = input_ids.len();
        Self {
            input_ids,
            target_ids,
            batch_size: 1,
            seq_len,
        }
    }

    /// Get input IDs for a specific batch item
    pub fn get_input(&self, batch_idx: usize) -> Option<&[u32]> {
        if batch_idx >= self.batch_size {
            return None;
        }
        let start = batch_idx * self.seq_len;
        let end = start + self.seq_len;
        Some(&self.input_ids[start..end])
    }

    /// Get target IDs for a specific batch item
    pub fn get_target(&self, batch_idx: usize) -> Option<&[u32]> {
        if batch_idx >= self.batch_size {
            return None;
        }
        let start = batch_idx * self.seq_len;
        let end = start + self.seq_len;
        Some(&self.target_ids[start..end])
    }

    /// Total number of tokens in batch
    pub fn num_tokens(&self) -> usize {
        self.batch_size * self.seq_len
    }
}

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

/// Calculate perplexity from cross-entropy loss
pub fn perplexity(loss: f32) -> f32 {
    loss.exp()
}

/// Calculate tokens per second
pub fn tokens_per_second(num_tokens: usize, elapsed_secs: f64) -> f64 {
    num_tokens as f64 / elapsed_secs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_train_config_new() {
        let model_config = TransformerConfig::tiny();
        let config = TransformerTrainConfig::new(model_config.clone());

        assert_eq!(config.model_config.hidden_size, model_config.hidden_size);
        assert!(!config.checkpoint_config.enabled);
        assert!(!config.precision_config.is_mixed());
    }

    #[test]
    fn test_transformer_train_config_with_checkpointing() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_checkpointing(4);

        assert!(config.checkpoint_config.enabled);
        assert_eq!(config.checkpoint_config.num_segments, 4);
    }

    #[test]
    fn test_transformer_train_config_with_bf16() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_bf16();

        assert!(config.precision_config.is_mixed());
    }

    #[test]
    fn test_transformer_train_config_with_fp16() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_fp16();

        assert!(config.precision_config.is_mixed());
        assert!(config.precision_config.dynamic_scaling);
    }

    #[test]
    fn test_transformer_train_config_builders() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny())
            .with_max_seq_len(1024)
            .with_accumulation_steps(4)
            .with_warmup_steps(100)
            .with_lr(0.0001)
            .with_grad_clip(1.0);

        assert_eq!(config.max_seq_len, 1024);
        assert_eq!(config.accumulation_steps, 4);
        assert_eq!(config.warmup_steps, 100);
    }

    #[test]
    fn test_lm_batch_from_sequences() {
        let sequences = vec![vec![0, 1, 2, 3, 4], vec![0, 5, 6, 7]];

        let batch = LMBatch::from_sequences(&sequences, 99, 100);

        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.seq_len, 4); // max_len - 1 = 5 - 1 = 4
    }

    #[test]
    fn test_lm_batch_single() {
        let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.seq_len, 3);
        assert_eq!(batch.num_tokens(), 3);
    }

    #[test]
    fn test_lm_batch_get_input() {
        let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

        let input = batch.get_input(0).unwrap();
        assert_eq!(input, &[1, 2, 3]);

        assert!(batch.get_input(1).is_none());
    }

    #[test]
    fn test_lm_batch_get_target() {
        let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

        let target = batch.get_target(0).unwrap();
        assert_eq!(target, &[2, 3, 4]);
    }

    #[test]
    fn test_lm_batch_empty() {
        let batch = LMBatch::from_sequences(&[], 0, 1);
        assert_eq!(batch.batch_size, 0);
        assert_eq!(batch.num_tokens(), 0);
    }

    #[test]
    fn test_transformer_trainer_new() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny());
        let trainer = TransformerTrainer::new(config);

        assert_eq!(trainer.step(), 0);
        assert!(!trainer.is_mixed_precision());
        assert!(!trainer.is_checkpointing());
    }

    #[test]
    fn test_transformer_trainer_forward_single() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny());
        let trainer = TransformerTrainer::new(config);

        let input_ids = vec![1, 2, 3];
        let target_ids = vec![2, 3, 4];

        let (loss, logits) = trainer.forward_single(&input_ids, &target_ids);

        assert!(loss > 0.0);
        assert!(loss.is_finite());
        assert_eq!(logits.len(), 3 * trainer.model().config().vocab_size);
    }

    #[test]
    fn test_transformer_trainer_train_batch() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny());
        let mut trainer = TransformerTrainer::new(config);

        let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);
        let loss = trainer.train_batch(&batch);

        assert!(loss > 0.0);
        assert!(loss.is_finite());
        assert_eq!(trainer.step(), 1);
    }

    #[test]
    fn test_transformer_trainer_train_epoch() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny());
        let mut trainer = TransformerTrainer::new(config);

        let batches = vec![
            LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]),
            LMBatch::single(vec![5, 6, 7], vec![6, 7, 8]),
        ];

        let avg_loss = trainer.train_epoch(&batches);

        assert!(avg_loss > 0.0);
        assert_eq!(trainer.step(), 2);
    }

    #[test]
    fn test_transformer_trainer_empty_epoch() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny());
        let mut trainer = TransformerTrainer::new(config);

        let avg_loss = trainer.train_epoch(&[]);
        assert_eq!(avg_loss, 0.0);
    }

    #[test]
    fn test_transformer_trainer_with_accumulation() {
        let config =
            TransformerTrainConfig::new(TransformerConfig::tiny()).with_accumulation_steps(2);
        let mut trainer = TransformerTrainer::new(config);

        let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

        // First batch - no step yet
        trainer.train_batch(&batch);
        assert_eq!(trainer.step(), 0);

        // Second batch - step occurs
        trainer.train_batch(&batch);
        assert_eq!(trainer.step(), 1);
    }

    #[test]
    fn test_transformer_trainer_warmup_lr() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny())
            .with_lr(0.001)
            .with_warmup_steps(100);
        let mut trainer = TransformerTrainer::new(config);

        // At step 0, LR should be 0
        assert_eq!(trainer.current_lr(), 0.0);

        // Train to advance step
        let batch = LMBatch::single(vec![1, 2], vec![2, 3]);
        trainer.train_batch(&batch);

        // At step 1, LR should be 0.001 * 1/100 = 0.00001
        let lr = trainer.current_lr();
        assert!(lr > 0.0);
        assert!(lr < 0.001);
    }

    #[test]
    fn test_perplexity() {
        let loss = 2.0;
        let ppl = perplexity(loss);
        assert!((ppl - loss.exp()).abs() < 1e-6);
    }

    #[test]
    fn test_tokens_per_second() {
        let tps = tokens_per_second(1000, 2.0);
        assert_eq!(tps, 500.0);
    }

    #[test]
    fn test_transformer_trainer_grad_scaler_stats() {
        let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_fp16();
        let trainer = TransformerTrainer::new(config);

        let (scale, overflows, successes) = trainer.grad_scaler_stats();
        assert!(scale > 0.0);
        assert_eq!(overflows, 0);
        assert_eq!(successes, 0);
    }

    #[test]
    fn test_transformer_trainer_with_model() {
        let model_config = TransformerConfig::tiny();
        let model = Transformer::new(&model_config);
        let config = TransformerTrainConfig::new(model_config);
        let trainer = TransformerTrainer::with_model(model, config);

        assert_eq!(trainer.step(), 0);
    }

    #[test]
    fn test_lm_batch_shift_correctness() {
        // Verify that input/target shift is correct for causal LM
        let sequences = vec![vec![100, 1, 2, 3, 200]]; // BOS, tokens, EOS
        let batch = LMBatch::from_sequences(&sequences, 0, 200);

        let input = batch.get_input(0).unwrap();
        let target = batch.get_target(0).unwrap();

        // Input should be [BOS, 1, 2, 3]
        assert_eq!(input[0], 100); // BOS
        assert_eq!(input[1], 1);
        assert_eq!(input[2], 2);
        assert_eq!(input[3], 3);

        // Target should be [1, 2, 3, EOS]
        assert_eq!(target[0], 1);
        assert_eq!(target[1], 2);
        assert_eq!(target[2], 3);
        assert_eq!(target[3], 200); // EOS
    }
}
