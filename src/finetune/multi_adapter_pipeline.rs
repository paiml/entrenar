//! Multi-Adapter Pipeline (GPU-SHARE Phase 2.1, GH-203)
//!
//! Trains N LoRA adapter sets concurrently on a single frozen NF4 base model.
//! The base model is loaded once to GPU, and each adapter maintains independent:
//! - LoRA A/B matrices (Q and V projections)
//! - AdamW optimizer state
//! - Training data iterator
//! - Checkpoint directory
//!
//! # VRAM Savings
//!
//! Compared to N separate processes (MPS), this saves (N-1) × base_model_vram:
//! - MPS (3 adapters on 7B): 3 × 7.3 GB = 21.9 GB
//! - Multi-adapter (3 adapters on 7B): 7.3 GB + 3 × 0.02 GB = 7.36 GB
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │         Frozen NF4 Base Model        │ ← loaded once
//! │    (CudaNf4TransformerBlock × L)     │
//! └──────────┬───────────┬───────────┬───┘
//!            │           │           │
//!     ┌──────┴──┐ ┌──────┴──┐ ┌──────┴──┐
//!     │Adapter 0│ │Adapter 1│ │Adapter 2│
//!     │LoRA A/B │ │LoRA A/B │ │LoRA A/B │
//!     │Optimizer│ │Optimizer│ │Optimizer│
//!     │  Data   │ │  Data   │ │  Data   │
//!     └─────────┘ └─────────┘ └─────────┘
//! ```

use super::instruct_corpus::InstructSample;
use super::instruct_pipeline::{InstructConfig, InstructPipeline, InstructStepResult};
use super::instruct_trainer::InstructEpochMetrics;
use crate::lora::LoRALayer;
use std::path::PathBuf;

/// Scheduling strategy for multi-adapter training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterSchedule {
    /// All adapters process one sample each per step (synchronized).
    Synchronized,
    /// Round-robin: each step trains one adapter.
    RoundRobin,
    /// Priority: adapter with highest validation loss gets the next step.
    PriorityValLoss,
}

impl Default for AdapterSchedule {
    fn default() -> Self {
        Self::RoundRobin
    }
}

/// Configuration for a single adapter slot.
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Path to training data (JSONL instruct corpus).
    pub data_path: PathBuf,
    /// Directory for adapter checkpoints.
    pub checkpoint_dir: PathBuf,
    /// Per-adapter hyperparameters (lora_rank, lr, epochs, etc.)
    pub instruct_config: InstructConfig,
}

/// Runtime state for one adapter during training.
pub struct AdapterSlot {
    /// Per-adapter LoRA layers (Q and V projections, per transformer layer).
    pub lora_layers: Vec<LoRALayer>,
    /// Training data for this adapter.
    pub train_samples: Vec<InstructSample>,
    /// Validation data for this adapter.
    pub val_samples: Vec<InstructSample>,
    /// Checkpoint directory for this adapter.
    pub checkpoint_dir: PathBuf,
    /// Per-adapter metrics history.
    pub metrics: Vec<InstructEpochMetrics>,
    /// Per-adapter config.
    pub config: InstructConfig,
    /// Current sample index within the training data.
    pub cursor: usize,
    /// Best validation loss (for early stopping / priority scheduling).
    pub best_val_loss: f32,

    /// Per-adapter GPU LoRA optimizer states.
    #[cfg(feature = "cuda")]
    pub(crate) optimizer_states: Option<Vec<crate::transformer::GpuLoraOptimizerState>>,
    /// NF4 LoRA optimizer step counter.
    #[cfg(feature = "cuda")]
    pub lora_step: u32,
}

/// Multi-adapter training pipeline.
///
/// Trains N LoRA adapter sets on a shared frozen NF4 base model.
/// GPU memory is dominated by the base model (~7 GB for 7B NF4);
/// each adapter adds only ~20 MB (LoRA A/B matrices + optimizer state).
pub struct MultiAdapterPipeline {
    /// The base InstructPipeline (owns the frozen transformer + CUDA blocks).
    pub base_pipeline: InstructPipeline,
    /// Independent adapter slots.
    pub adapters: Vec<AdapterSlot>,
    /// Scheduling strategy.
    pub schedule: AdapterSchedule,
    /// Current step counter (global across all adapters).
    pub global_step: usize,
}

impl MultiAdapterPipeline {
    /// Create a new multi-adapter pipeline.
    ///
    /// The `base_pipeline` should be a fully initialized InstructPipeline
    /// (with CUDA blocks uploaded if GPU training is desired). Adapter slots
    /// are initially empty — call `add_adapter()` to register each one.
    pub fn new(base_pipeline: InstructPipeline, schedule: AdapterSchedule) -> Self {
        Self {
            base_pipeline,
            adapters: Vec::new(),
            schedule,
            global_step: 0,
        }
    }

    /// Add an adapter slot with its own training data and checkpoint directory.
    pub fn add_adapter(
        &mut self,
        config: AdapterConfig,
        train_samples: Vec<InstructSample>,
        val_samples: Vec<InstructSample>,
    ) {
        let model_config = &self.base_pipeline.model.config;
        let lora_layers = InstructPipeline::build_lora_layers(
            &self.base_pipeline.model,
            model_config,
            &config.instruct_config,
        );

        let slot = AdapterSlot {
            lora_layers,
            train_samples,
            val_samples,
            checkpoint_dir: config.checkpoint_dir,
            metrics: Vec::new(),
            config: config.instruct_config,
            cursor: 0,
            best_val_loss: f32::INFINITY,
            #[cfg(feature = "cuda")]
            optimizer_states: None,
            #[cfg(feature = "cuda")]
            lora_step: 0,
        };

        self.adapters.push(slot);
    }

    /// Number of registered adapters.
    pub fn num_adapters(&self) -> usize {
        self.adapters.len()
    }

    /// Select which adapter index to train next based on the schedule.
    pub fn select_next_adapter(&self) -> Option<usize> {
        if self.adapters.is_empty() {
            return None;
        }
        match self.schedule {
            AdapterSchedule::Synchronized => {
                // All adapters train — caller should iterate all
                Some(0)
            }
            AdapterSchedule::RoundRobin => {
                Some(self.global_step % self.adapters.len())
            }
            AdapterSchedule::PriorityValLoss => {
                // Pick adapter with highest (worst) validation loss
                self.adapters
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.best_val_loss
                            .partial_cmp(&b.best_val_loss)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
            }
        }
    }

    /// Train one step on the specified adapter.
    ///
    /// Swaps the adapter's LoRA layers into the base pipeline, runs one
    /// training step, then swaps them back out.
    ///
    /// # Returns
    ///
    /// Training step result (loss, perplexity) or `None` if the adapter's
    /// data is exhausted.
    pub fn train_step_adapter(&mut self, adapter_idx: usize) -> Option<InstructStepResult> {
        let slot = &mut self.adapters[adapter_idx];

        // Check if data is exhausted
        if slot.cursor >= slot.train_samples.len() {
            return None;
        }

        let sample = &slot.train_samples[slot.cursor];
        slot.cursor += 1;

        // Tokenize prompt and response separately
        if !self.base_pipeline.has_tokenizer() {
            return None;
        }
        let prompt_ids = self.base_pipeline.tokenize(&sample.instruction);
        let response_ids = self.base_pipeline.tokenize(&sample.response);

        if prompt_ids.is_empty() || response_ids.is_empty() {
            return None;
        }

        // Swap adapter's LoRA layers into the base pipeline
        std::mem::swap(&mut slot.lora_layers, &mut self.base_pipeline.lora_layers);

        // Run training step through base pipeline (uses shared CUDA blocks)
        let result = self
            .base_pipeline
            .train_step(&prompt_ids, &response_ids);

        // Swap LoRA layers back
        std::mem::swap(&mut slot.lora_layers, &mut self.base_pipeline.lora_layers);

        self.global_step += 1;

        Some(result)
    }

    /// Reset all adapter cursors for a new epoch.
    pub fn reset_epoch(&mut self, seed: u64) {
        for (i, slot) in self.adapters.iter_mut().enumerate() {
            slot.cursor = 0;
            // Shuffle training data with per-adapter seed
            shuffle_samples(&mut slot.train_samples, seed.wrapping_add(i as u64));
        }
    }

    /// Check if all adapters have exhausted their training data.
    pub fn all_exhausted(&self) -> bool {
        self.adapters
            .iter()
            .all(|s| s.cursor >= s.train_samples.len())
    }
}

/// Simple Fisher-Yates shuffle with a deterministic seed.
fn shuffle_samples(samples: &mut [InstructSample], seed: u64) {
    let mut rng = seed;
    for i in (1..samples.len()).rev() {
        // xorshift64
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng as usize) % (i + 1);
        samples.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_round_robin() {
        let sched = AdapterSchedule::RoundRobin;
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![dummy_slot(), dummy_slot(), dummy_slot()],
            schedule: sched,
            global_step: 0,
        };

        assert_eq!(pipeline.select_next_adapter(), Some(0));

        let pipeline = MultiAdapterPipeline {
            global_step: 1,
            ..pipeline
        };
        assert_eq!(pipeline.select_next_adapter(), Some(1));

        let pipeline = MultiAdapterPipeline {
            global_step: 5,
            ..pipeline
        };
        assert_eq!(pipeline.select_next_adapter(), Some(2));
    }

    #[test]
    fn test_schedule_priority_val_loss() {
        let mut slot0 = dummy_slot();
        slot0.best_val_loss = 1.0;
        let mut slot1 = dummy_slot();
        slot1.best_val_loss = 3.0; // worst
        let mut slot2 = dummy_slot();
        slot2.best_val_loss = 2.0;

        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![slot0, slot1, slot2],
            schedule: AdapterSchedule::PriorityValLoss,
            global_step: 0,
        };

        assert_eq!(pipeline.select_next_adapter(), Some(1)); // highest loss
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = MultiAdapterPipeline {
            base_pipeline: create_dummy_pipeline(),
            adapters: vec![],
            schedule: AdapterSchedule::RoundRobin,
            global_step: 0,
        };
        assert_eq!(pipeline.select_next_adapter(), None);
        assert!(pipeline.all_exhausted());
    }

    #[test]
    fn test_shuffle_deterministic() {
        let mut samples1 = vec![
            InstructSample {
                instruction: "a".into(),
                response: "1".into(),
                system: None,
                metadata: None,
            },
            InstructSample {
                instruction: "b".into(),
                response: "2".into(),
                system: None,
                metadata: None,
            },
            InstructSample {
                instruction: "c".into(),
                response: "3".into(),
                system: None,
                metadata: None,
            },
        ];
        let mut samples2 = samples1.clone();

        shuffle_samples(&mut samples1, 42);
        shuffle_samples(&mut samples2, 42);

        // Same seed → same order
        for (s1, s2) in samples1.iter().zip(samples2.iter()) {
            assert_eq!(s1.instruction, s2.instruction);
        }
    }

    fn create_dummy_pipeline() -> InstructPipeline {
        use crate::transformer::TransformerConfig;
        let config = TransformerConfig::tiny();
        InstructPipeline::new(&config, InstructConfig::default())
    }

    fn dummy_slot() -> AdapterSlot {
        AdapterSlot {
            lora_layers: Vec::new(),
            train_samples: Vec::new(),
            val_samples: Vec::new(),
            checkpoint_dir: PathBuf::from("/tmp/test"),
            metrics: Vec::new(),
            config: InstructConfig::default(),
            cursor: 0,
            best_val_loss: f32::INFINITY,
            #[cfg(feature = "cuda")]
            optimizer_states: None,
            #[cfg(feature = "cuda")]
            lora_step: 0,
        }
    }
}
