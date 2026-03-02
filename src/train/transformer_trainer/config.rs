//! Configuration for transformer training

use crate::autograd::{CheckpointConfig, MixedPrecisionConfig};
use crate::train::TrainConfig;
use crate::transformer::TransformerConfig;

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
    /// Maximum training steps (stop after this many optimizer steps)
    pub max_steps: Option<usize>,
    /// Use CUDA GPU training when available (default: true = auto-detect)
    pub use_cuda: bool,
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
            max_steps: None,
            use_cuda: true,
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

    /// Set maximum training steps
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = Some(steps);
        self
    }

    /// Enable or disable CUDA GPU training (default: true = auto-detect)
    pub fn with_use_cuda(mut self, use_cuda: bool) -> Self {
        self.use_cuda = use_cuda;
        self
    }
}
